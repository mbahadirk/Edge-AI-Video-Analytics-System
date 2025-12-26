import time
import numpy as np
import cv2
import torch
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO


class Detector:
    def __init__(self, backend, model_path, input_size=(640, 640), conf_thres=0.25, iou_thres=0.45):
        self.backend = backend
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.backend == "pytorch":
            self.model = YOLO(model_path).model.to(self.device).eval()
            
        elif self.backend == "onnx":
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
        elif self.backend == "tensorrt":
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs = []
            self.cuda_inputs = []
            self.host_outputs = []
            self.cuda_outputs = []
            self.bindings = []
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_tensor_shape(binding)) * self.engine.get_tensor_dtype(binding).itemsize
                host_mem = cuda.pagelocked_empty(trt.volume(self.engine.get_tensor_shape(binding)), trt.nptype(self.engine.get_tensor_dtype(binding)))
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(cuda_mem))
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    self.host_inputs.append(host_mem)
                    self.cuda_inputs.append(cuda_mem)
                else:
                    self.host_outputs.append(host_mem)
                    self.cuda_outputs.append(cuda_mem)
                    
        self._warmup()

    def _warmup(self):
        dummy_input = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        for _ in range(3):
            self(dummy_input)

    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (nw, nh))
        
        dw, dh = self.input_size[1] - nw, self.input_size[0] - nh
        dw //= 2
        dh //= 2
        
        padded_image = cv2.copyMakeBorder(image_resized, dh, self.input_size[0]-nh-dh, dw, self.input_size[1]-nw-dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        blob = padded_image.astype(np.float32)
        blob /= 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale, (dw, dh)

    def postprocess(self, output, scale, pad):
        dw, dh = pad
        output = np.squeeze(output)
        if output.shape[0] != 84:
             output = output.T
             
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(output.shape[1]):
            class_score = output[4:, i]
            class_id = np.argmax(class_score)
            score = class_score[class_id]
            
            if score > self.conf_thres:
                cx, cy, w, h = output[:4, i]
                x1 = (cx - w / 2 - dw) / scale
                y1 = (cy - h / 2 - dh) / scale
                w = w / scale
                h = h / scale
                
                boxes.append([x1, y1, w, h])
                scores.append(float(score))
                class_ids.append(class_id)
                
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "box": [x, y, x + w, y + h],
                    "score": scores[i],
                    "class_id": class_ids[i]
                })
                
        return detections

    def __call__(self, image):
        t_start = time.time()
        
        if isinstance(image, list):
             image = image[0] 
        
        preprocessed_img, scale, pad = self.preprocess(image)
        
        t_infer_start = time.time()
        
        if self.backend == "pytorch":
            input_tensor = torch.from_numpy(preprocessed_img).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)[0].cpu().numpy()
                
        elif self.backend == "onnx":
            output = self.session.run(self.output_names, {self.input_name: preprocessed_img})[0]
            
        elif self.backend == "tensorrt":
            np.copyto(self.host_inputs[0], preprocessed_img.ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()
            output = self.host_outputs[0].reshape(1, -1, 8400) 
            
        t_infer_end = time.time()
        
        results = self.postprocess(output, scale, pad)
        
        t_end = time.time()
        
        stats = {
            "preprocess_ms": (t_infer_start - t_start) * 1000,
            "inference_ms": (t_infer_end - t_infer_start) * 1000,
            "postprocess_ms": (t_end - t_infer_end) * 1000,
            "total_ms": (t_end - t_start) * 1000
        }
        
        return results, stats

if __name__ == "__main__":
    img = cv2.imread("../test_image.jpg")
    if img is None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)

    detector = Detector(backend="pytorch", model_path="../models/best.pt")
    dets, stats = detector(img)
    print(f"Detections: {len(dets)}")
    print(f"Timing: {stats}")