# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    class_id: int
    label: str

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]
    inference_time_ms: float
    fps: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool

class MetricsResponse(BaseModel):
    avg_latency_ms: float
    current_fps: float
    gpu_usage_percent: float
    gpu_memory_used_mb: float

class ModelLoadRequest(BaseModel):
    model_name: str