import tensorrt as trt
import os
import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

class INT8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=1, height=640, width=640):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = training_data
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * self.height * self.width * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None

        batch = []
        for i in range(self.batch_size):
            img_path = self.data[self.current_index + i]
            img = Image.open(img_path).resize((self.width, self.height))
            img_np = np.array(img).transpose(2, 0, 1).astype(np.float32)
            img_np /= 255.0
            batch.append(img_np)
        
        self.current_index += self.batch_size
        batch_np = np.ascontiguousarray(np.array(batch).ravel())
        cuda.memcpy_htod(self.device_input, batch_np)
        
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def get_calibration_files(data_dir, limit=500):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                images.append(os.path.join(root, file))
                if len(images) >= limit:
                    return images
    return images
