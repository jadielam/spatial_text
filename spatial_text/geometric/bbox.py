from functools import lru_cache
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


@lru_cache(maxsize=4)
def rotation_matrix(angle_in_degres: float) -> np.ndarray:
    """
    Returns a rotation matrix for a given angle in degrees.

    Uses lru_cache because the four angles of the coordinate
    system (0, 90, 180, 270) are used repeatedly.
    """
    angle_in_radians = angle_in_degres * np.pi / 180
    return np.array(
        [
            [np.cos(angle_in_radians), -np.sin(angle_in_radians)],
            [np.sin(angle_in_radians), np.cos(angle_in_radians)],
        ],
    )


def rotate_points(points: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """
    Rotates a set of points by a given angle and returns the rotated
    points.

    Arguments:
        points: Array of shape (nb_points, 2). Points given in the format (x, y).
        angle_in_degrees: Angle in degrees to rotate the points.

    Returns:
        Rotated points.
    """
    rm = rotation_matrix(angle_in_degrees)
    return np.dot(points, rm)


def derive_points_of_bboxes(bboxes: np.ndarray) -> np.ndarray:
    """
    Derives the four points of a set of bounding boxes from the bounding boxes in the format
    (xmin, ymin, xmax, ymax).

    Arguments:
        bboxes: Array of shape (nb_boxes, 4). Bounding boxes given in the format (xmin, ymin, xmax, ymax).

    Returns:
        Four points of the bounding boxes in the format (x, y). The shape of the returned
        array is (nb_bboxes, 4, 2).
    """
    points = np.zeros((bboxes.shape[0], 4, 2))
    points[:, 0, 0] = bboxes[:, 0]
    points[:, 0, 1] = bboxes[:, 1]
    points[:, 1, 0] = bboxes[:, 0]
    points[:, 1, 1] = bboxes[:, 3]
    points[:, 2, 0] = bboxes[:, 2]
    points[:, 2, 1] = bboxes[:, 1]
    points[:, 3, 0] = bboxes[:, 2]
    points[:, 3, 1] = bboxes[:, 3]
    return points


def derive_bboxes_from_points(rotated_bbox_points: np.ndarray) -> np.ndarray:
    """
    Derives the bounding boxes in (xmin, ymin, xmax, ymax) format from
    from the four points of the bboxes

    Arguments:
        rotated_bbox_points: Array of shape (nb_bboxes, 4, 2) with the four points
            corresponding to the corners of a bbox. Points are given in format (x, y)

    Returns:
        Bounding boxes: np.ndarray of shape (nb_bboxes, 4). Bboxes are given in the format
            (xmin, ymin, xmax, ymax)
    """
    bboxes = np.zeros((rotated_bbox_points.shape[0], 4))
    bboxes[:, 0] = np.min(rotated_bbox_points[:, :, 0], axis=1)
    bboxes[:, 1] = np.min(rotated_bbox_points[:, :, 1], axis=1)
    bboxes[:, 2] = np.max(rotated_bbox_points[:, :, 0], axis=1)
    bboxes[:, 3] = np.max(rotated_bbox_points[:, :, 1], axis=1)
    return bboxes


def rotate_bboxes(bboxes: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """
    Rotates a set of bounding boxes by a given angle.
    The process to rotate the bounding boxes is as follows:
    1. Convert the bounding boxes in two points format (xmin, ymin) and (xmax, ymax)
        to four points format.
    2. Rotate the four points.
    3. Derive the two point format bboxes from the four points.
    4. Return the bounding boxes in the two points format and in the four points format.

    The bounding boxes need to be rotated in both formats because returning them
    only in the two points format makes it impossible to revert the rotation operation.


    Arguments:
        bboxes: Bounding boxes given in the format (xmin, ymin, xmax, ymax).
        angle_in_degres: Angle in degrees to rotate the bounding boxes. Most divisible
            by 90

    Returns:
        bboxes in the two points format (xmin, ymin, xmax, ymax)
    """
    if angle_in_degrees % 90 != 0:
        raise ValueError('Angle needs to be divisible by 90')
    bbox_points = derive_points_of_bboxes(bboxes)
    rotated_bbox_points = rotate_points(bbox_points, angle_in_degrees)
    return derive_bboxes_from_points(rotated_bbox_points)


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
        return correct_predictions / len(pred) if len(pred) > 0 else 0
    return correct_predictions / len(target) if len(target) > 0 else 0
