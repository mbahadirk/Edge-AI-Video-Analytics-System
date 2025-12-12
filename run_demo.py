import argparse
from inference.video_engine import VideoEngine
import os


def main():
    # Modellerin varlığını kontrol et
    if os.path.exists("../models/best.onnx"):
        backend = "onnx"
        model = "../models/best.onnx"
    else:
        backend = "pytorch"
        model = "../models/best.pt"
    
    print(f"🚀 Starting Demo with {backend} backend...")
    
    # Girdi Videosu (Colab'e yüklediğin videonun adı)
    input_video = "test_video.mp4" 
    
    # Çıktı Videosu
    output_video = "output_demo.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Error: {input_video} not found. Please upload a video.")
        # Fallback to webcam/dummy? No, just exit.
        return

    engine = VideoEngine(model_path=model, backend=backend, source=input_video, output=output_video)
    engine.run()

if __name__ == "__main__":
    main()