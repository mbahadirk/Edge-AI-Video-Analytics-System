import sys
import os
import time
import cv2
import numpy as np
import uvicorn
import pynvml
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from typing import List

# --- Path Setup ---
# Add project root to path so we can import 'monitoring' and 'schemas'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.schemas import DetectionResponse, HealthResponse, MetricsResponse, BoundingBox
from monitoring.monitor import PerformanceMonitor

# --- Configuration ---
MODEL_PATH = "models/model.trt"
CONFIDENCE_THRESHOLD = 0.5


# --- TensorRT Engine Wrapper ---
# In a real scenario, this might be in 'api/engine.py'.
# We include it here for a complete, standalone server file.
class TRTEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # Check if model exists
        if not os.path.exists(engine_path):
            print(f"⚠️ Warning: Model not found at {engine_path}. Running in Mock Mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            print(f"Loading TensorRT Engine: {engine_path}")
            # TODO: Add your actual TensorRT loading logic here (Part 3/4 content)
            # self.load_engine()
            # self.allocate_buffers()

    def infer(self, image: np.ndarray):
        """
        Performs inference on the image.
        Returns a list of dicts: {'bbox': [x1, y1, x2, y2], 'conf': 0.95, 'class': 0, 'label': 'person'}
        """
        if self.mock_mode:
            # Simulate processing delay (15ms)
            time.sleep(0.015)
            # Return dummy detection
            h, w, _ = image.shape
            return [{
                "bbox": [int(w * 0.2), int(h * 0.2), int(w * 0.5), int(h * 0.6)],
                "conf": 0.96,
                "class": 0,
                "label": "person"
            }]

        # TODO: Add actual TensorRT inference logic here
        # 1. Preprocess (Resize, Normalize, NHWC -> NCHW)
        # 2. Transfer to GPU
        # 3. Execute
        # 4. Transfer back
        # 5. Postprocess (NMS)
        return []


# --- Global State ---
model: TRTEngine = None
perf_monitor = PerformanceMonitor(window_size=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the model and initializes GPU monitoring.
    """
    global model
    print("🚀 Server starting up...")

    # 1. Load Model
    try:
        model = TRTEngine(MODEL_PATH)
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")

    # 2. Initialize GPU (NVML) for Monitoring
    # Note: monitor.py handles its own init, but we double-check here.
    if perf_monitor.gpu_available:
        print("✅ GPU Monitoring Initialized (NVML)")
    else:
        print("⚠️ GPU Monitoring Disabled")

    yield

    # Shutdown logic
    print("🛑 Server shutting down...")
    try:
        pynvml.nvmlShutdown()
    except:
        pass


# --- API Application ---
app = FastAPI(
    title="Real-Time Detection API (TensorRT)",
    version="1.0.0",
    lifespan=lifespan
)


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decodes bytes into an OpenCV image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the health status of the API and GPU."""
    return {
        "status": "healthy" if model else "degraded",
        "gpu_available": perf_monitor.gpu_available
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Main inference endpoint.
    1. Decodes image.
    2. Runs inference (monitored).
    3. Logs performance metrics in background.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Read & Decode
    try:
        contents = await file.read()
        image = decode_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Inference with Performance Monitoring
    t0 = perf_monitor.start_inference()  # Start Timer

    raw_results = model.infer(image)  # Run Model

    latency_ms = perf_monitor.stop_inference(t0)  # Stop Timer & Record

    # 3. Queue Logging Task (Non-blocking)
    # Log every 10th request or if latency spikes > 100ms
    if len(perf_monitor.latency_buffer) % 10 == 0 or latency_ms > 100:
        background_tasks.add_task(perf_monitor.log_snapshot)

    # 4. Format Response
    formatted_detections = []
    for res in raw_results:
        formatted_detections.append(BoundingBox(
            x_min=res["bbox"][0],
            y_min=res["bbox"][1],
            x_max=res["bbox"][2],
            y_max=res["bbox"][3],
            confidence=res["conf"],
            class_id=res["class"],
            label=res["label"]
        ))

    return {
        "detections": formatted_detections,
        "inference_time_ms": round(latency_ms, 2),
        "fps": round(perf_monitor._calculate_fps(), 1),
        "model_name": "yolov8m_trt"
    }


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Exposes internal performance metrics (P50, P95, GPU usage)
    for dashboards or manual checks.
    """
    lat_stats = perf_monitor._calculate_percentiles()
    gpu_stats = perf_monitor._get_gpu_stats()
    current_fps = perf_monitor._calculate_fps()

    return {
        "avg_latency_ms": lat_stats["avg"],
        "current_fps": round(current_fps, 1),
        "gpu_usage_percent": gpu_stats["gpu_util_pct"],
        "gpu_memory_used_mb": gpu_stats["mem_used_mb"]
    }


if __name__ == "__main__":
    # Running this file directly starts the server
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)