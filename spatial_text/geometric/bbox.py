from typing import Dict

import numpy as np


def compute_iop_or_iou(pred_bbox: np.ndarray, target_bbox: np.ndarray) -> float:
    return max(compute_iop(pred_bbox, target_bbox), compute_iou(pred_bbox, target_bbox))


def compute_iop(pred_bbox: np.ndarray, target_bbox: np.ndarray) -> float:
    """
    Computes the IOT (Intersection over pred) between a predicted bounding box
    and a target bounding box.
    """
    min_x = max(pred_bbox[0], target_bbox[0])
    max_x = min(pred_bbox[2], target_bbox[2])
    min_y = max(pred_bbox[1], target_bbox[1])
    max_y = min(pred_bbox[3], target_bbox[3])
    intersection = max(0, max_x - min_x) * max(0, max_y - min_y)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    return intersection / pred_area


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Computes the IoU (intersection over union) between two bounding boxes.

    Args:
        bbox1: Bounding box given in the format (xmin, ymin, xmax, ymax).
        bbox2: Bounding box given in the format (xmin, ymin, xmax, ymax).

    Returns:
        IoU between the two bounding boxes.
    """
    min_x = max(bbox1[0], bbox2[0])
    max_x = min(bbox1[2], bbox2[2])
    min_y = max(bbox1[1], bbox2[1])
    max_y = min(bbox1[3], bbox2[3])
    intersection = max(0, max_x - min_x) * max(0, max_y - min_y)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union


def map_evaluation(
    pred: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    p=0.90,
    denominator='target',
) -> float:
    """
    Computes the MAP (Mean Average Precision) evaluation metric.  The map metric
    is defined as the percentage of the target extractions that are correctly
    predicted by the predicted extractions.  A predicted extraction is considered
    correct if the IoU between the predicted bounding box and the target bounding box
    is greater than p.

    Args:
        pred: Dictionary of the keys to their corresponding predicted bounding boxes.
            Bounding boxes are given in the format (xmin, ymin, xmax, ymax).
        target: Dictionary of the keys to their corresponding target bounding boxes.
            Bounding boxes are given in the format (xmin, ymin, xmax, ymax).
        p: IoU threshold for a predicted extraction to be considered correct.
    """
    correct_predictions = 0
    for key, target_bbox in target.items():
        if key in pred:
            pred_bbox = pred[key]
            iou = compute_iop_or_iou(pred_bbox, target_bbox)
            if iou >= p:
                correct_predictions += 1
    if denominator == 'pred':
        return correct_predictions / len(pred)
    return correct_predictions / len(target)
