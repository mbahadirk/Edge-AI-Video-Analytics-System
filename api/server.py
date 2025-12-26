import sys
import os
import cv2
import numpy as np
import uvicorn
import pynvml
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, FileResponse
from typing import List

# --- Path Setup ---
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
sys.path.append(PROJECT_ROOT)

# Klasör Tanımları
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")

# --- Kullanıcı Ayarı ---
DEFAULT_MODEL_NAME = "best.onnx"

from api.schemas import DetectionResponse, HealthResponse, MetricsResponse, BoundingBox, ModelLoadRequest
from inference.inference import Detector
from monitoring.dashboard import Dashboard

# --- Global State ---
detector: Detector = None
dashboard = Dashboard(window_size=100)
current_model_name = "None"


def determine_backend(model_file: str) -> str:
    ext = os.path.splitext(model_file)[1].lower()
    if ext in [".pt", ".pth"]:
        return "pytorch"
    elif ext == ".onnx":
        return "onnx"
    elif ext in [".engine", ".trt"]:
        return "tensorrt"
    else:
        return "unknown"


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    vis_img = image.copy()
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf = det['score']
        cls_id = det['class_id']

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{cls_id} {conf:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(vis_img, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(vis_img, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return vis_img


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, current_model_name

    # Klasörleri oluştur (Garanti olsun)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    print("\n" + "=" * 40)
    print(f"🚀 Server Başlatılıyor...")

    if os.path.exists(MODELS_DIR):
        files = os.listdir(MODELS_DIR)
        target_path = os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)

        if os.path.exists(target_path):
            print(f"🎯 Varsayılan model bulundu: {DEFAULT_MODEL_NAME}")
            load_model_logic(DEFAULT_MODEL_NAME)
        else:
            valid_files = [f for f in files if f.endswith(('.pt', '.onnx', '.engine', '.trt'))]
            if valid_files:
                print(f"🔄 Alternatif yükleniyor: {valid_files[0]}")
                load_model_logic(valid_files[0])
            else:
                print("⚠️ Klasörde hiç uygun model YOK!")
    else:
        print(f"❌ HATA: '{MODELS_DIR}' klasörü yok!")

    if dashboard.gpu_available:
        print("✅ GPU Monitoring Aktif")

    yield

    print("🛑 Server Kapatılıyor...")
    # Temp klasörünü temizle ama klasörü silme (Permission hatası olmasın diye)
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass
    try:
        if dashboard.gpu_handle: pynvml.nvmlShutdown()
    except:
        pass


def load_model_logic(model_name: str):
    global detector, current_model_name
    model_path = os.path.join(MODELS_DIR, model_name)
    backend = determine_backend(model_name)
    print(f"📥 Model Yükleniyor: {model_name} ({backend})")
    detector = Detector(backend=backend, model_path=model_path)
    current_model_name = model_name
    print(f"✅ Model Başarıyla Yüklendi!")


# --- API ---
app = FastAPI(title="Dynamic AI Server", version="3.1 - Fixes")


@app.get("/models")
def list_models():
    files = [f for f in os.listdir(MODELS_DIR)] if os.path.exists(MODELS_DIR) else []
    return {"available_models": files, "current_loaded_model": current_model_name}


@app.post("/load")
def load_model_endpoint(request: ModelLoadRequest):
    try:
        load_model_logic(request.model_name)
        return {"status": "success", "message": f"Loaded {request.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy" if detector else "waiting", "model": current_model_name}


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")
    return img


@app.post("/detect")
async def detect_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        return_image: bool = Query(False)
):
    if not detector: raise HTTPException(status_code=503, detail="Model yüklü değil.")

    contents = await file.read()
    image = decode_image(contents)

    t0 = dashboard.start_recording()
    detections, stats = detector(image)
    dashboard.stop_recording(t0)
    background_tasks.add_task(dashboard.capture_snapshot)

    # 1. Görsel Kayıt (Her zaman yapıyoruz)
    vis_img = draw_detections(image, detections)
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Klasör yoksa oluştur
    save_path = os.path.join(RESULTS_DIR, "latest_image.jpg")
    cv2.imwrite(save_path, vis_img)

    # 2. Return Image True ise Resim Dön
    if return_image:
        _, encoded_img = cv2.imencode('.jpg', vis_img)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

    # 3. Return Image False ise JSON Dön (DÜZELTME BURADA)
    formatted_detections = []
    for det in detections:
        box = det['box']
        formatted_detections.append({
            "x_min": int(box[0]),
            "y_min": int(box[1]),
            "x_max": int(box[2]),
            "y_max": int(box[3]),
            # --- KRİTİK DÜZELTME: NumPy tiplerini Python tiplerine çevir ---
            "confidence": float(det['score']),
            "class_id": int(det['class_id']),
            "label": str(det['class_id'])
            # -------------------------------------------------------------
        })

    return {
        "detections": formatted_detections,
        "count": len(detections),
        "model": current_model_name,
        "inference_time_ms": stats['inference_ms']
    }


@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    if not detector: raise HTTPException(status_code=503, detail="Model yüklü değil.")

    # --- DÜZELTME: Klasörü oluştur (No such file hatası için) ---
    os.makedirs(TEMP_DIR, exist_ok=True)
    # ----------------------------------------------------------

    input_path = os.path.join(TEMP_DIR, "input_video.mp4")
    output_path = os.path.join(TEMP_DIR, "output_processed.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"🎥 Video işleme başladı: {file.filename}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Video açılamadı.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0  # Hata önleyici

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        detections, _ = detector(frame)
        processed_frame = draw_detections(frame, detections)
        out.write(processed_frame)
        frame_count += 1

        if frame_count % 10 == 0: print(f"   İşlenen Kare: {frame_count}")

    cap.release()
    out.release()
    print(f"✅ Video tamamlandı. Kare: {frame_count}")

    return FileResponse(output_path, media_type="video/mp4", filename="processed_result.mp4")


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
