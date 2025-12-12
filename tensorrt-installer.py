print("⬇️ Installing TensorRT 10.0.1 from NVIDIA Index...")

import os
try:
    import tensorrt as trt
    print(f"\n✅ TensorRT Successfully Imported! Version: {trt.__version__}")
    print("INFO: TensorRT python bindings are working.")
except ImportError as e:
    print(f"❌ TensorRT import failed: {e}")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"✅ PyCUDA Version: {cuda.get_version()}")
except:
    print("⚠️ PyCUDA warning")
