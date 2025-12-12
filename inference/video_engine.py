import cv2
import numpy as np
import time
import threading
import queue
from inference import Detector

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


class VideoPipeline:
    def __init__(self, source, model_path, backend="tensorrt", skip_frames=5):
        self.capture = cv2.VideoCapture(source)
        self.detector = Detector(backend=backend, model_path=model_path)
        self.skip_frames = skip_frames
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        self.stopped = False
        self.trackers = []
        self.frame_id = 0
        
        self.fps = 0.0
        self.mode = "Init"

    def start(self):
        self.t_capture = threading.Thread(target=self.capture_loop, daemon=True)
        self.t_process = threading.Thread(target=self.process_loop, daemon=True)
        
        self.t_capture.start()
        self.t_process.start()
        
        self.display_loop()

    def capture_loop(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                self.stopped = True
                break
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

    def process_loop(self):
        while not self.stopped:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            self.frame_id += 1
            start_time = time.time()
            
            final_boxes = []
            
            if self.frame_id % self.skip_frames == 0:
                self.mode = "Detection"
                detections, _ = self.detector(frame)
                
                valid_trackers = []
                
                for det in detections:
                    det_box = det["box"]
                    matched = False
                    
                    for trk in self.trackers:
                        success, trk_box = trk.update(frame)
                        if success:
                            iou = compute_iou(det_box, trk_box)
                            if iou > 0.5:
                                valid_trackers.append(trk)
                                matched = True
                                break
                    
                    if not matched:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, tuple(det_box))
                        valid_trackers.append(tracker)
                    
                    final_boxes.append(det_box)
                
                self.trackers = valid_trackers
                
            else:
                self.mode = "Tracking"
                new_trackers = []
                for tracker in self.trackers:
                    success, box = tracker.update(frame)
                    if success:
                        final_boxes.append(box)
                        new_trackers.append(tracker)
                self.trackers = new_trackers

            end_time = time.time()
            self.fps = 1.0 / (end_time - start_time)
            
            if not self.result_queue.full():
                self.result_queue.put((frame, final_boxes, self.fps, self.mode))

    def display_loop(self):
        while not self.stopped:
            if self.result_queue.empty():
                time.sleep(0.01)
                continue

            frame, boxes, fps, mode = self.result_queue.get()
            
            cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {mode}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            for box in boxes:
                x, y, w, h = map(int, box[:4])
                
                # Handling different box formats (xywh vs xyxy)
                # If tracker returns xywh (x,y,w,h)
                # If detector returns xyxy (x1,y1,x2,y2) -> need check
                # Based on Parts 2/3, we assume consistent format or handle it here:
                # Assuming output is [x, y, w, h] for drawing:
                if len(box) == 4:
                     if w > frame.shape[1]: # likely xyxy
                         w = w - x
                         h = h - y
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            cv2.imshow("Real-Time Engine", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break
        
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = VideoPipeline(
        source="../test_video.mp4",
        model_path="../models/yolov8l_fp16.engine",
        backend="tensorrt",
        skip_frames=5
    )
    pipeline.start()