from __future__ import annotations

from collections import defaultdict

import numpy as np
from spatial_text.geometric.utils import compute_angle
from scipy.spatial import Delaunay, QhullError  # type: ignore


def _beta_skeleton_naive(points: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Tests each triple of points p, q, r for membership of r in the region R_{pq}
    and will construct the beta-skeleton in time O(n^3)

    Arguments:
    ----------
    - points: An `np.ndarray` of shape (nb_points, 2) where the second dimension
        represents the x, y coordinates a point in the Cartesian plane

    Returns:
    --------
    - graph_edges: An `np.ndarray` of  of shape (nb_edges, 2) where the
        second dimension represent the indexes of points from the `point` array that
        form an undirected edge in the skeleton graph.
    """
    list_of_edges = []
    theta = np.arcsin(1 / beta) if beta >= 1 else np.pi - np.arcsin(beta)

    for p_idx in range(points.shape[0]):
        p = points[p_idx]
        for q_idx in range(p_idx + 1, points.shape[0]):
            q = points[q_idx]
            points_in_region = 0
            for r_idx in range(points.shape[0]):
                if r_idx != p_idx and r_idx != q_idx:
                    r = points[r_idx]

                    # Test if point r is in region R_{pq}
                    angle_prq = compute_angle(p, r, q)
                    if angle_prq >= theta:
                        points_in_region += 1
            if points_in_region == 0:
                list_of_edges.append([p_idx, q_idx])

    return np.array(list_of_edges, dtype=np.int32)


def _beta_skeleton_delaunay(points: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    For `beta` >= 1, the beta-skeleton is a subgraph of the Delaunay triangulation.
    If pq is an edge of the Delaunay triangulation that is not an edge of the
    beta-skeleton, then a point r that forms a large angle prq can be found as
    one of the at most two points forming a triangle with p and q in the
    Delaunay triangulation. The circle based beta-skeleton for a set of points
    can be constructed in time O(nlogn) by computing the Delaunay triangulation
    and using this test to filter its edges.

    Arguments:
    ----------
    - points: An `np.ndarray` of shape (nb_points, 2) where the second dimension
        represents the x, y coordinates a point in the Cartesian plane

    Returns:
    --------
    - graph_edges: An `np.ndarray` of integer type, of shape (nb_edges, 2) where the
        second dimension represent the indexes of points from the `point` array that
        form an undirected edge in the skeleton graph.
    """
    if beta < 1:
        raise ValueError('beta most be greater than or equal to 1')
    theta = np.arcsin(1 / beta)

    # 1. Use Delaunay triangulation to compute, for each edge, the at most
    #   two points forming a triangle with that edge.
    tri = Delaunay(points)
    edge_points = defaultdict(list)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edge_points[edge].append(simplex[(i + 2) % 3])

    # 2. For each edge, test the triangle points to see if they form
    # large angles that disqualify the edge from being part of the
    # beta-skeleton
    to_return = []
    for (p_i, q_i), r_i_l in edge_points.items():
        p, q = points[p_i], points[q_i]
        if not any([compute_angle(p, points[r_i], q) >= theta for r_i in r_i_l]):
            to_return.append([p_i, q_i])
    return np.array(to_return, dtype=np.int32)


def beta_skeleton(points: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    The definition used to compute the beta-skeleton graph is the one
    provided by Wikipedia here: https://en.wikipedia.org/wiki/Beta_skeleton

    Let n=len(points). Two different algorithms are used, depending on
    the value of `beta`.

    1. For `beta` < 1, a Naive algorithm tests each triple of points p, q, r for
        membership of r in the region R_{pq} and will construct the beta-skeleton
        in time O(n^3)
    2. For `beta` >= 1, the beta-skeleton is a subgraph of the Delaunay triangulation.
        If pq is an edge of the Delaunay triangulation that is not an edge of the
        beta-skeleton, then a point r that forms a large angle prq can be found as
        one of the at most two points forming a triangle with p and q in the
        Delaunay triangulation. The circle based beta-skeleton for a set of points
        can be constructed in time O(nlogn) by computing the Delaunay triangulation
        and using this test to filter its edges

    Arguments:
    ----------
    - points: An `np.ndarray` of shape (nb_points, 2) where the second dimension
        represents the x, y coordinates a point in the Cartesian plane

    Returns:
    --------
    - graph_edges: An `np.ndarray` of integer type, of shape (nb_edges, 2) where the
        second dimension represent the indexes of points from the `point` array that
        form an undirected edge in the skeleton graph.
    """
    if beta >= 1 and len(points) >= 4:
        try:
            edges = _beta_skeleton_delaunay(points, beta)
        except QhullError:
            # We arrive here when the initial simplex is flat
            edges = _beta_skeleton_naive(points, beta)
        return edges
    else:
        return _beta_skeleton_naive(points, beta)
