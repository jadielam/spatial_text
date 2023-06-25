import numpy as np
from spatial_text.data_model import Token


def compute_angle(p: np.ndarray, r: np.ndarray, q: np.ndarray) -> float:
    """
    Computes angle prq formed by the three points:

    Arguments:
    ----------
    - p: `np.ndarray` of shape (2,) representing the x-y coordinates of
        the point
    - r: `np.ndarray` of shape (2,) representing the x-y coordinates of
        the point
    - q: `np.ndarray` of shape (2,) representing the x-y coordinates of
        the point

    Returns:
    --------
    - angle in radians of the angle between the points
    """
    rp = p - r
    rq = q - r
    cosine = np.dot(rp, rq) / (np.linalg.norm(rp) * np.linalg.norm(rq))
    angle = np.arccos(cosine)
    return angle


def is_to_the_right(t1: Token, t2: Token, threshold: float = np.pi / 16) -> bool:
    """
    Returns True if token2 is spatially to the right of token1.
    Function assumes that bounding boxes of token1 and token2 do not overlap,
    otherwise the result is not well behaved.

    Arguments:
    ----------
    - t1: Token 1
    - t2: Token 2
    - threshold: Threshold of angle in radians.
    """
    t1_xr = t1.bbox[2]
    t2_xl = t2.bbox[0]

    if t2_xl < t1_xr:
        return False

    t1_ym = (t1.bbox[3] - t1.bbox[1]) / 2
    t2_ym = (t2.bbox[3] - t2.bbox[1]) / 2

    angle_in_radians = compute_angle(
        np.array([t1_xr, t1_ym]),
        np.array([0, 0]),
        np.array([t2_xl, t2_ym]),
    )
    if angle_in_radians < threshold:
        return True
    return False
