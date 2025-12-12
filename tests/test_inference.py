import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# --- Helpers ---
def mock_postprocess(bbox_norm, scale):
    """Simulates rescaling boxes back to original image."""
    # bbox_norm is [x1, y1, x2, y2] in model space
    x1, y1, x2, y2 = bbox_norm
    return [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]


@pytest.fixture
def trt_engine_path():
    path = "models/model.trt"
    if not os.path.exists(path):
        pytest.skip("⚠️ model.trt not found. Skipping engine integration tests.")
    return path


# --- Tests ---

def test_trt_engine_loading(trt_engine_path):
    """Tries to load the engine and run one inference warmup."""
    try:
        from api.server import TRTEngine
        engine = TRTEngine(trt_engine_path)
        assert engine is not None

        # Warmup with dummy data
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        engine.infer(dummy)

    except ImportError:
        pytest.skip("API module not found.")
    except Exception as e:
        pytest.fail(f"Engine loading failed: {e}")


def test_pre_post_consistency():
    """
    Verifies that scaling coordinates down and back up results
    in the same coordinates (handling aspect ratio).
    """
    orig_h, orig_w = 1080, 1920
    target_size = 640

    # Original Box
    original_box = [100, 100, 200, 200]

    # 1. Simulate Preprocess Scale
    scale = min(target_size / orig_h, target_size / orig_w)

    # 2. Transform to model space
    model_box = [c * scale for c in original_box]

    # 3. Transform back (Postprocess)
    restored_box = mock_postprocess(model_box, scale)

    # 4. Check deviation (Allowing small rounding errors)
    diff = np.abs(np.array(original_box) - np.array(restored_box))
    assert np.all(diff < 2), f"Coordinate drift too high: {diff} pixels"