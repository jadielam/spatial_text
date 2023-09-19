import numpy as np
import pytest
from spatial_text.geometric.bbox import (
    compute_iou,
    derive_bboxes_from_points,
    derive_points_of_bboxes,
    rotate_bboxes,
    rotate_points,
)


@pytest.fixture
def points():
    return np.array([[[1, 2], [1, 3]], [[1, 3], [3, 5]], [[458, -23], [239, -242]]])


@pytest.fixture
def bboxes():
    return np.array([[0, 0, 4, 2], [4, 0, 8, 2], [8, 0, 12, 2], [12, 0, 16, 2]])


@pytest.fixture
def negative_bboxes():
    return np.array([[0, -2, 4, 2], [4, -3, 8, 2], [8, 0, 12, 2], [-4, 20, 16, 2]])


def test_compute_iou_perfect():
    bbox1 = np.array([0, 0, 1, 1])
    bbox2 = np.array([0, 0, 1, 1])
    assert compute_iou(bbox1, bbox2) == 1.0


def test_compute_iou_no_overlap():
    bbox1 = np.array([0, 0, 1, 1])
    bbox2 = np.array([2, 2, 3, 3])
    assert compute_iou(bbox1, bbox2) == 0.0
    assert compute_iou(bbox2, bbox1) == 0.0


@pytest.mark.parametrize(
    'angle,expected_bboxes',
    [
        (90, np.array([[0, -4, 2, 0], [0, -8, 2, -4], [0, -12, 2, -8], [0, -16, 2, -12]])),
        (
            180,
            np.array(
                [
                    [-4, -2, 0, 0],
                    [-8, -2, -4, 0],
                    [-12, -2, -8, 0],
                    [-16, -2, -12, 0],
                ],
            ),
        ),
    ],
)
def test_rotate_bboxes(bboxes, angle, expected_bboxes):
    rotated_bboxes = rotate_bboxes(bboxes, angle)
    np.testing.assert_almost_equal(rotated_bboxes, expected_bboxes, decimal=4)


def test_inverse_bbox_rotation(bboxes):
    for angle in [90, 180, 270, 360]:
        rotated_bboxes = rotate_bboxes(bboxes, angle)
        inverse_rotated = rotate_bboxes(rotated_bboxes, -angle)
        np.testing.assert_almost_equal(bboxes, inverse_rotated, decimal=4)


def test_inverse_points_rotation(points):
    for angle in [90, 100, 180, 270, 345, 360]:
        rotated_points = rotate_points(points, angle)
        inverse_rotated = rotate_points(rotated_points, -angle)
        np.testing.assert_almost_equal(points, inverse_rotated, decimal=4)


@pytest.mark.parametrize(
    'bboxes,points',
    [
        (
            np.array([[0, 0, 4, 2], [4, 0, 8, 2], [8, 0, 12, 2], [12, 0, 16, 2]]),
            np.array(
                [
                    [[0, 0], [0, 2], [4, 0], [4, 2]],
                    [[4, 0], [4, 2], [8, 0], [8, 2]],
                    [[8, 0], [8, 2], [12, 0], [12, 2]],
                    [[12, 0], [12, 2], [16, 0], [16, 2]],
                ],
            ),
        ),
        (
            np.array([[26, 52, 12, 16], [49, 80, 63, 40], [5, 96, 9, 60]]),
            np.array(
                [
                    [[26, 52], [26, 16], [12, 52], [12, 16]],
                    [[49, 80], [49, 40], [63, 80], [63, 40]],
                    [[5, 96], [5, 60], [9, 96], [9, 60]],
                ],
            ),
        ),
    ],
)
def test_derive_points_of_bboxes(bboxes, points):
    derived_points = derive_points_of_bboxes(bboxes)
    np.testing.assert_almost_equal(derived_points, points, decimal=4)


@pytest.mark.parametrize(
    'points,bboxes',
    [
        (
            np.array(
                [
                    [[0, 0], [0, 2], [4, 0], [4, 2]],
                    [[4, 0], [4, 2], [8, 0], [8, 2]],
                    [[8, 0], [8, 2], [12, 0], [12, 2]],
                    [[12, 0], [12, 2], [16, 0], [16, 2]],
                ],
            ),
            np.array([[0, 0, 4, 2], [4, 0, 8, 2], [8, 0, 12, 2], [12, 0, 16, 2]]),
        ),
        (
            np.array(
                [
                    [[26, 52], [26, 16], [12, 52], [12, 16]],
                    [[49, 80], [49, 40], [63, 80], [63, 40]],
                    [[5, 96], [5, 60], [9, 96], [9, 60]],
                ],
            ),
            np.array([[12, 16, 26, 52], [49, 40, 63, 80], [5, 60, 9, 96]]),
        ),
        (np.array([[[16, 21], [29, 7], [14, 4], [76, 69]]]), np.array([[14, 4, 76, 69]])),
    ],
)
def test_derive_bboxes_from_points(points, bboxes):
    derived_bboxes = derive_bboxes_from_points(points)
    np.testing.assert_almost_equal(derived_bboxes, bboxes, decimal=4)


def test_rotate_bboxes_exception(bboxes):
    with pytest.raises(ValueError):
        rotate_bboxes(bboxes, 45)
