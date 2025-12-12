from ultralytics import YOLO
import sys

def export_model(weights_path):
    model = YOLO(weights_path)
    model.export(
        format='onnx',
        dynamic=True,
        opset=12,
        simplify=True
    )

if __name__ == "__main__":
    weights = "../models/best.pt"
    if len(sys.argv) > 1:
        weights = sys.argv[1]
    export_model(weights)