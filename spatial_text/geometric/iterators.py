from typing import Generator, List, Tuple

import numpy as np


def line_by_line_itr(
    ocr: List[Tuple[str, np.ndarray]],
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Given a list of tokens with their bounding boxes, it returns
    a generator of tuples with the token and its bounding box.
    Bounding boxes are represented as np.ndarray with format
    (xmin, ymin, xmax, ymax)

    It orders the tokens from top to bottom and left to right.

    Args:
        ocr: List of tuples with the token and its bounding box.
            Bounding boxes are represented as np.ndarray with format
            (xmin, ymin, xmax, ymax)

    Returns:
        Generator of tuples with the token and its bounding box.
    """
    raise NotImplementedError()


def column_by_column_itr(
    ocr: List[Tuple[str, np.ndarray]],
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Given a list of tokens with their bounding boxes, it returns
    a generator of tuples with the token and its bounding box.
    Bounding boxes are represented as np.ndarray with format
    (xmin, ymin, xmax, ymax)

    It orders the tokens from left to right and top to bottom."""
    raise NotImplementedError()
