from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from spatial_text.geometric.beta_skeleton import (
    _beta_skeleton_delaunay,
    _beta_skeleton_naive,
    beta_skeleton,
    compute_angle,
)


def compare_edge_lists(el_1: Sequence, el_2: Sequence) -> bool:
    """
    Arguments:
    ----------
    - el_1: edge list 1
    - el_2: edge list 2

    Returns:
    --------
    - `True` if the two lists of edges are the same, `False` otherwise
    """
    set_1 = set([tuple(sorted(a)) for a in el_1])
    set_2 = set([tuple(sorted(a)) for a in el_2])
    return set_1 == set_2


def test_compute_angle_pi():
    p = np.array([1, 2])
    q = np.array([3, 4])
    r = np.array([2, 3])

    angle = compute_angle(p, r, q)
    np.testing.assert_almost_equal(angle, np.pi, decimal=7)


def test_compute_angle_pi_2():
    p = np.array([1, 0])
    q = np.array([0, 1])
    r = np.array([0, 0])

    angle = compute_angle(p, r, q)
    np.testing.assert_almost_equal(angle, np.pi / 2, decimal=7)


@pytest.mark.parametrize(
    'test_input,expected',
    [
        ([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[0, 1], [1, 2]]),
        ([[1.0, 2.0], [2.0, 3.0]], [[0, 1]]),
        ([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [2.0, 40.0]], [[0, 1], [1, 2], [1, 3]]),
    ],
)
def test_beta_skeleton_naive_beta_1(test_input, expected):
    points = np.array(test_input)
    edges = _beta_skeleton_naive(points, 1.0)
    expected_edges = np.array(expected, dtype=np.int32)
    np.testing.assert_array_equal(edges, expected_edges)


def test_beta_skeleton_delaunay_beta_1():
    """
    Uses the beta skeleton naive implementation to test
    the Delaunay implementation which is more complex.
    """
    inputs = [
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 20.0]],
        [[1.0, 21.0], [7.0, 3.0], [5.0, 4.0], [4.0, 20.0]],
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [-7.0, 4.0],
            [-4.0, 20.0],
            [1.0, 21.0],
            [7.0, 3.0],
            [5.0, 4.0],
            [4.0, 20.0],
        ],
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [-7.0, 4.0],
            [-4.0, 20.0],
            [1.0, 21.0],
            [7.0, 3.0],
            [5.0, 4.0],
            [4.0, 20.0],
            [1.0, 0.0],
            [7.0, 1.0],
            [9.0, 4.0],
            [4.0, 10.0],
        ],
    ]
    for test_input in inputs:
        points = np.array(test_input)
        edges_d = _beta_skeleton_delaunay(points, 1.0)
        edges_n = _beta_skeleton_naive(points, 1.0)
        assert compare_edge_lists(edges_d, edges_n)


def test_beta_skeleton():
    inputs = [
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[1.0, 21.0], [7.0, 3.0], [5.0, 4.0], [4.0, 20.0]],
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [-7.0, 4.0],
            [-4.0, 20.0],
            [1.0, 21.0],
            [7.0, 3.0],
            [5.0, 4.0],
            [4.0, 20.0],
        ],
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [-7.0, 4.0],
            [-4.0, 20.0],
            [1.0, 21.0],
            [7.0, 3.0],
            [5.0, 4.0],
            [4.0, 20.0],
            [1.0, 0.0],
            [7.0, 1.0],
            [9.0, 4.0],
            [4.0, 10.0],
        ],
    ]
    for test_input in inputs:
        beta_skeleton(np.array(test_input))
