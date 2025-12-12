import pytest


# --- Helpers ---
def calculate_iou(box1, box2):
    """Calculates Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / float(union_area)


# --- Tests ---

def test_tracker_drift():
    """
    Simulates a tracker by verifying that small movements 
    maintain high IoU (Same Object), while jumps drop IoU (New Object).
    """
    # Frame 1: Object A
    box_t0 = [100, 100, 200, 200]

    # Frame 2: Object A moves slightly (Drift)
    box_t1 = [105, 105, 205, 205]

    # Frame 3: Object B appears elsewhere
    box_t2 = [300, 300, 400, 400]

    # Test 1: Same Object Verification
    iou_same = calculate_iou(box_t0, box_t1)
    assert iou_same > 0.8, f"Tracker lost ID on small move. IoU: {iou_same}"

    # Test 2: Different Object Verification
    iou_diff = calculate_iou(box_t0, box_t2)
    assert iou_diff < 0.1, f"Tracker failed to distinguish objects. IoU: {iou_diff}"