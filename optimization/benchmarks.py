import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import json
import pynvml
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Benchmark:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        input_shape = input_data.shape
        self.context.set_input_shape(self.engine.get_tensor_name(0), input_shape)
        
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_size = 1
        for binding in self.engine:
            if not self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
                dims = self.context.get_tensor_shape(binding)
                for dim in dims:
                    if dim > 0: output_size *= dim
                output_size *= np.dtype(dtype).itemsize

        d_output = cuda.mem_alloc(output_size)
        cuda.memcpy_htod_async(d_input, input_data, self.stream)
        
        bindings = [int(d_input), int(d_output)]
        
        start = time.time()
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        end = time.time()
        
        return (end - start) * 1000

    def run(self, iterations=100, warmup=10):
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
        
        for _ in range(warmup):
            self.infer(input_data)
            
        latencies = []
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        for _ in range(iterations):
            lat = self.infer(input_data)
            latencies.append(lat)
            
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        fps = 1000 / avg_latency
        
        return {
            "model": os.path.basename(self.engine_path),
            "latency_avg_ms": round(avg_latency, 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "throughput_fps": round(fps, 2),
            "gpu_utilization_pct": gpu_util
        }

if __name__ == "__main__":
    engines = ["models/yolov8l_fp16.engine", "models/yolov8l_int8.engine"]
    results = []
    
    for engine in engines:
        if os.path.exists(engine):
            b = Benchmark(engine)
            results.append(b.run())
            
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)