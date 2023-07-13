from typing import List, Tuple

import numpy as np

from spatial_text.data_model import Block


def layout(ocr: List[Tuple[str, np.ndarray]]) -> List[Block]:
    """
    Given a list of tokens with their bounding boxes, it returns
    a list of blocks that represent the layout of the page.  Blocks
    are ordered left-to-right and top-to-bottom.

    The idea of the algorithm is this one:

    1. Iterate over the tokens in the OCR line by line.
    2. At any given moment in time, we have a list of open blocks
        to which a token can be attached to.
    3. We also build a graph where the nodes represent the state
        of the layout at a given moment in time, and the edges
        represent transitions from one state to another.
    3. For each token, probe all the open blocks at that moment in tim

    Args:
        ocr: List of tuples with the token and its bounding box.
            Bounding boxes are represented as np.ndarray with format
            (xmin, ymin, xmax, ymax)

    Returns:
        List of blocks that represent the layout of the page.
    """
    return []
