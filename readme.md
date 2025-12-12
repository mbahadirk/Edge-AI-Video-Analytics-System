# 🚀 Real-Time Object Detection Pipeline (YOLOv8 + TensorRT + FastAPI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-24.0-blue)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-76B900)

## 📌 Overview
This project represents a production-grade, end-to-end AI pipeline designed for high-performance object detection. It covers the full lifecycle of an AI product:
1.  **Training:** Fine-tuning YOLOv8m with advanced augmentation strategies.
2.  **Optimization:** Converting the model to **TensorRT (FP16)** for low-latency inference.
3.  **Deployment:** Serving the model via a **FastAPI** wrapper inside a GPU-accelerated Docker container.
4.  **Observability:** Built-in performance monitoring (Latency, FPS, GPU Usage).

## 🏗️ Architecture
The system is designed as a microservice that exposes a REST API.

```mermaid
graph LR
    User[Client / Stress Test] -->|POST Image| API[FastAPI Server]
    API -->|Pre-process| TRT[TensorRT Engine]
    TRT -->|Inference (GPU)| TRT
    TRT -->|Post-process| API
    API -->|JSON Response| User
    
    subgraph Monitoring
    Monitor[Performance Monitor] -.->|Log Metrics| API
    NVML[NVIDIA NVML] -.->|GPU Stats| Monitor
    end
📂 Project StructurePlaintext├── api/
│   ├── server.py           # Main FastAPI application
│   ├── schemas.py          # Pydantic data models
│   └── docker/             # Docker configuration
├── monitoring/
│   └── monitor.py          # Real-time latency & GPU usage tracker
├── training/
│   ├── train.py            # Training script
│   └── augmentations.py    # Strong augmentation policies (Mosaic, MixUp)
├── models/
│   └── model.trt           # Optimized TensorRT Engine
├── stress_test.py          # Client-side benchmarking script
└── requirements.txt        # Python dependencies
🚀 Quick StartPrerequisitesNVIDIA GPU (CUDA 11.x / 12.x compatible)Docker & NVIDIA Container ToolkitPython 3.10+1. Build the Docker ImageThe Dockerfile uses the official NVIDIA TensorRT runtime as a base.Bashdocker build -t detection-api -f api/docker/Dockerfile .
2. Run the ContainerLaunch the API with GPU access enabled.Bashdocker run -d \
  --name detector \
  --gpus all \
  -p 8000:8000 \
  detection-api
3. Verify HealthCheck if the model is loaded and GPU is accessible:Bashcurl http://localhost:8000/health
# Output: {"status": "healthy", "gpu_available": true}
⚡ Performance BenchmarksPerformance metrics were collected using stress_test.py.MetricResultDescriptionThroughput~19 FPSEnd-to-end (HTTP + Inference)Avg Latency53 msTotal round-trip timemAP@500.96Model precision after trainingFormatTensorRT (FP16)Optimized engine formatTo reproduce these results, run the stress test script:Bashpython stress_test.py
📡 API Documentation1. Detect ObjectsEndpoint: /detectMethod: POSTBody: multipart/form-data (Key: file)Response:JSON{
  "detections": [
    {
      "x_min": 100,
      "y_min": 50,
      "x_max": 200,
      "y_max": 300,
      "confidence": 0.95,
      "class_id": 0,
      "label": "person"
    }
  ],
  "inference_time_ms": 15.4,
  "fps": 64.9
}
2. Real-Time MetricsEndpoint: /metricsMethod: GETDescription: Returns live monitoring stats for dashboards.Response:JSON{
  "avg_latency_ms": 53.2,
  "current_fps": 18.8,
  "gpu_usage_percent": 45,
  "gpu_memory_used_mb": 1250
}
🧠 Training StrategyThe model was trained using YOLOv8m (Medium) to balance speed and accuracy.Key HyperparametersOptimizer: AdamW with Cosine Learning Rate Scheduler.Precision: AMP (Automatic Mixed Precision).Batch Size: Optimized for 16GB VRAM.Strong AugmentationsTo ensure robustness in varied environments, the following augmentations were applied (via training/augmentations.py):Mosaic & MixUp: To improve small object detection and generalization.HSV Jitter: Robustness against lighting changes.Random Erasing: Simulating occlusions.🛠️ Technical DecisionsWhy TensorRT?Standard PyTorch inference has high overhead. TensorRT optimizes the graph (layer fusion) and uses FP16 precision to boost inference speed by ~40% on NVIDIA hardware.Why FastAPI?Its asynchronous nature (async def) allows handling concurrent requests efficiently without blocking the inference loop.Why Docker?Ensures reproducibility. The nvcr.io/nvidia/tensorrt base image eliminates "CUDA hell" by pre-packaging all driver dependencies.📜 LicenseTechnical Assessment Submission - December 2025