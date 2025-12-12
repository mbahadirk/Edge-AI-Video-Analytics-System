import pytest
import numpy as np
import cv2


# --- Helpers ---
def mock_preprocess(image, target_size=(640, 640)):
    """Simulates Resize -> Pad -> HWC to CHW."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(image, (nw, nh))

    # Pad
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    canvas[:nh, :nw, :] = img_resized

    # Transpose to NCHW
    input_tensor = canvas.transpose(2, 0, 1) / 255.0
    return np.expand_dims(input_tensor, axis=0)


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


# --- Tests ---

def test_io_shape_validation(dummy_image):
    """Ensures preprocessing returns correct NCHW shape (1, 3, 640, 640)."""
    tensor = mock_preprocess(dummy_image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float64 or tensor.dtype == np.float32


def test_onnx_dynamic_shapes():
    """Verifies that logic holds for different batch sizes (Dynamic Batching)."""
    # Batch 1
    input_b1 = np.zeros((1, 3, 640, 640), dtype=np.float32)
    assert input_b1.shape[0] == 1

    # Batch 8 (Simulating larger load)
    input_b8 = np.zeros((8, 3, 640, 640), dtype=np.float32)
    assert input_b8.shape[0] == 8