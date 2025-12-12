def get_hyperparameters():
    """
    Defines 'Strong' augmentation hyperparameters for YOLOv8.
    """
    return {
        # --- Training Mechanics ---
        "epochs": 55,  # Adjust based on your schedule
        "imgsz": 640,  # Base image size
        "batch": 48,  # Adjust based on GPU VRAM
        "patience": 20,  # Early stopping

        # --- Optimizer & Schedule ---
        "optimizer": "AdamW",  # Stable optimizer
        "lr0": 0.001,  # Initial learning rate
        "lrf": 0.01,  # Final learning rate (cosine schedule)
        "cos_lr": True,  # Cosine LR scheduler (Required)
        "warmup_epochs": 3.0,  # Warmup period
        "amp": True,  # Automatic Mixed Precision (Required)

        # --- Strong Augmentations (Geometric) ---
        "mosaic": 1.0,  # Mosaic (1.0 = 100% probability)
        "mixup": 0.15,  # MixUp (combine images)
        "copy_paste": 0.3,  # Copy-Paste augmentation
        "degrees": 10.0,  # Rotation (+/- deg)
        "translate": 0.1,  # Translation (+/- fraction)
        "scale": 0.5,  # Scale gain (Multi-scale effect)
        "shear": 0.0,  # Shear
        "perspective": 0.0005,  # Perspective distortion
        "flipud": 0.0,  # Flip Up-Down
        "fliplr": 0.5,  # Flip Left-Right
        # --- Strong Augmentations (Pixel-Level & Albumentations) ---
        # YOLOv8 maps these natively to efficient transforms
        "hsv_h": 0.015,  # Hue jitter
        "hsv_s": 0.7,  # Saturation jitter
        "hsv_v": 0.4,  # Value (Brightness) jitter

        # --- CutOut / Occlusion ---
        "erasing": 0.4,  # Random Erasing (CutOut equivalent) - 40% prob
    }


def check_albumentations():
    """
    Verifies that albumentations is installed for the extra
    pixel-level effects (Blur, CLAHE) that YOLO applies automatically.
    """
    try:
        import albumentations
        print(f"✅ Albumentations {albumentations.__version__} detected.")
        print("   YOLOv8 will automatically apply Blur, MedianBlur, and CLAHE during training.")
    except ImportError:
        print("⚠️ Albumentations not found! Install it for full augmentation support:")
        print("   pip install albumentations")
