import numpy as np

from spatial_text.geometric.bbox import compute_iou


def test_compute_iou_perfect():
    bbox1 = np.array([0, 0, 1, 1])
    bbox2 = np.array([0, 0, 1, 1])
    assert compute_iou(bbox1, bbox2) == 1.0


def test_compute_iou_no_overlap():
    bbox1 = np.array([0, 0, 1, 1])
    bbox2 = np.array([2, 2, 3, 3])
    assert compute_iou(bbox1, bbox2) == 0.0
    assert compute_iou(bbox2, bbox1) == 0.0


def compute_iou_partial_overlap():
    bbox1 = np.array([0, 0, 2, 2])
    bbox2 = np.array([1, 1, 3, 3])
    assert compute_iou(bbox1, bbox2) == 0.25
    assert compute_iou(bbox2, bbox1) == 0.25
