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