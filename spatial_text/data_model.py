import itertools
from functools import cached_property
from typing import List

import numpy as np


class Token:
    """
    A token as a word and a bbox position. It also has an id
    `tid` that is automatically generated.

    Arguments:
    ----------
    - word: str
    - bbox: np.ndarray with format (xmin, ymin, xmax, ymax)

    """

    id_iter = itertools.count()

    def __init__(self, word: str, bbox: np.ndarray):
        assert word is not None
        assert bbox is not None
        self.__word = word
        self.__bbox = bbox
        self.__tid = next(Token.id_iter)

    @property
    def tid(self):
        return self.__tid

    @property
    def word(self):
        return self.__word

    @cached_property
    def bbox(self):
        """
        Notice that it won't be able to modify the bbox inside the token
        """
        return self.__bbox.copy()

    def __eq__(self, other):
        """
        Comparison only takes into consideration the token id.
        This means that a token should only be created once.
        """
        return self.__tid == other.tid


class Line:
    """
    A line is a list of tokens that are read from left to right.
    """

    def __init__(self, tokens: List[Token] = []):
        assert tokens is not None
        self.__tokens = tokens

    @cached_property
    def tokens(self) -> List[Token]:
        return list(self.__tokens)

    @cached_property
    def bbox(self) -> np.ndarray:
        min_x = min([t.bbox[0] for t in self.__tokens])
        max_x = max([t.bbox[2] for t in self.__tokens])
        min_y = min([t.bbox[1] for t in self.__tokens])
        max_y = max([t.bbox[3] for t in self.__tokens])
        return np.array([min_x, min_y, max_x, max_y])

    def __len__(self):
        return len(self.tokens)

    def __eq__(self, other):
        return self.tokens == other.tokens


class Block:
    """
    A block is a list of lines that overflow from top to bottom.
    """

    def __init__(self, lines: List[Line] = []):
        self.__lines = lines

    @cached_property
    def lines(self):
        return list(self.__lines)

    @cached_property
    def bbox(self) -> np.ndarray:
        min_x = min([line.bbox[0] for line in self.__lines])
        max_x = max([line.bbox[2] for line in self.__lines])
        min_y = min([line.bbox[1] for line in self.__lines])
        max_y = max([line.bbox[3] for line in self.__lines])
        return np.array([min_x, min_y, max_x, max_y])

    def __eq__(self, other):
        return self.__lines == other.lines
