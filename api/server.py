import sys
import os
import cv2
import numpy as np
import uvicorn
import pynvml
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from typing import List

# --- Path Setup ---
# Proje ana dizinini path'e ekliyoruz ki 'inference' ve 'monitoring' modüllerini bulabilsin
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.schemas import DetectionResponse, HealthResponse, MetricsResponse, BoundingBox
# Önceki adımlarda yazdığımız güçlü Detector sınıfını kullanıyoruz
from inference.inference import Detector
# İzleme için Dashboard sınıfını kullanıyoruz
from monitoring.dashboard import Dashboard

# --- Configuration ---
MODEL_PATH = "../models/yolov8l_int8.engine"  # Veya .engine
CONFIDENCE_THRESHOLD = 0.5

# --- Global State ---
# Detector ve Dashboard'u global olarak tanımlıyoruz
detector: Detector = None
# Dashboard'u burada başlatıyoruz, her istekte yeniden başlatmak (file'daki hata) performansı öldürür
dashboard = Dashboard(window_size=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama başlarken modeli yükler ve GPU izlemeyi kontrol eder.
    """
    global detector
    print("🚀 Server starting up...")

    # 1. Load Model (Detector Sınıfı üzerinden)
    if os.path.exists(MODEL_PATH):
        try:
            # Backend olarak 'tensorrt' seçiyoruz. inference.py bu işi halledecek.
            print(f"Loading TensorRT Engine: {MODEL_PATH}")
            detector = Detector(backend="tensorrt", model_path=MODEL_PATH, conf_thres=CONFIDENCE_THRESHOLD)
            # Isınma turu (Warmup) Detector __init__ içinde otomatik yapılıyor.
        except Exception as e:
            print(f"❌ Critical Error loading model: {e}")
            detector = None
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}. API will return errors.")

    # 2. GPU Monitoring Kontrolü
    if dashboard.gpu_available:
        print("✅ GPU Monitoring Initialized (NVML)")
    else:
        print("⚠️ GPU Monitoring Disabled (NVML Init Failed)")

    yield

    # Shutdown logic
    print("🛑 Server shutting down...")
    try:
        if dashboard.gpu_handle:
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
    """Bytes verisini OpenCV formatına çevirir."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """API ve GPU durumunu kontrol eder."""
    return {
        "status": "healthy" if detector else "degraded",
        "gpu_available": dashboard.gpu_available
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Ana tespit endpoint'i.
    1. Resmi decode eder.
    2. Detector ile tahmin yapar (Süreyi ölçerek).
    3. Sonuçları loglar.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # 1. Read & Decode
    try:
        contents = await file.read()
        image = decode_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Inference & Monitoring
    # Timer'ı başlat
    t0 = dashboard.start_recording()

    try:
        # Detector sınıfı hem sonuçları hem de süre istatistiklerini (stats) döner
        # inference.py içindeki __call__ metodunu kullanıyoruz
        detections_raw, stats = detector(image)
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    # Timer'ı durdur ve süreyi kaydet
    latency_ms = dashboard.stop_recording(t0)

    # 3. Background Logging
    # Yanıtı geciktirmemek için loglamayı arka plana atıyoruz
    background_tasks.add_task(dashboard.capture_snapshot)

    # 4. Format Response
    # Detector sınıfı zaten {box, score, class_id} formatında dönüyor, şemaya uyarlıyoruz
    formatted_detections = []

    # inference.py'den gelen formatı API şemasına çeviriyoruz
    # Gelen format: [{'box': [x1, y1, x2, y2], 'score': 0.95, 'class_id': 0}, ...]
    for det in detections_raw:
        box = det['box']
        formatted_detections.append(BoundingBox(
            x_min=int(box[0]),
            y_min=int(box[1]),
            x_max=int(box[2]),
            y_max=int(box[3]),
            confidence=det['score'],
            class_id=det['class_id'],
            label=f"class_{det['class_id']}"  # Eğer class names listen varsa buradan maple
        ))

    # FPS'i dashboard üzerinden anlık hesaplıyoruz
    current_fps = dashboard.meter.get_fps()

    return {
        "detections": formatted_detections,
        "inference_time_ms": round(latency_ms, 2),
        "fps": round(current_fps, 1),
        "model_name": "yolov8_trt"
    }


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Dashboard üzerinden canlı performans verilerini çeker.
    """
    # Dashboard sınıfındaki FPSMeter'dan istatistikleri alıyoruz
    lat_stats = dashboard.meter.get_latency_statistics()
    gpu_stats = dashboard._get_gpu_stats()
    current_fps = dashboard.meter.get_fps()

    return {
        "avg_latency_ms": lat_stats["avg"],
        "current_fps": round(current_fps, 1),
        "gpu_usage_percent": gpu_stats["util_pct"],
        "gpu_memory_used_mb": gpu_stats["mem_used_mb"]
    }


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)