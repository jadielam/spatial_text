from typing import List, Optional

import numpy as np

from spatial_text import config
from spatial_text.utils.running_stats import RunningStats


class TextContainer:
    __slots__ = ['text']

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class Token(TextContainer):
    __slots__ = ['text']

    def __init__(self, text: str):
        super().__init__(text)

    def __eq__(self, other):
        if type(other) != Token:
            return False
        return self.text == other.text

    def __hash__(self):
        return hash(self.text)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()


class SpatialToken(Token):
    """
    A token as a word and a bbox position. It also has an id
    `tid` that is automatically generated.

    Arguments:
    ----------
    - word: str
    - bbox: np.ndarray with format (xmin, ymin, xmax, ymax)
    """

    __slots__ = ['__bbox', '__area']

    def __init__(self, text: str, bbox: np.ndarray):
        assert text is not None
        assert bbox is not None
        super().__init__(text)
        self.__bbox = bbox
        self.__area = (self.__bbox[2] - self.__bbox[0]) * (self.__bbox[3] - self.__bbox[1])

    @property
    def bbox(self):
        """
        Notice that it won't be able to modify the bbox inside the token
        """
        if self.__bbox is None:
            return None
        return self.__bbox

    @property
    def avg_char_length(self):
        return (self.bbox[2] - self.bbox[0]) / len(self.text)

    def area(self):
        """
        Computes the area of the token.
        """
        return self.__area

    def __eq__(self, other):
        """
        Comparison only takes into consideration the token id.
        This means that a token should only be created once.
        """
        if type(other) != SpatialToken:
            return False
        return self.text == other.text and np.array_equal(self.__bbox, other.bbox)

    def __hash__(self):
        return hash((self.text, self.__bbox))


class Line:
    """
    A line is a list of tokens that are read from left to right.
    """

    __slots__ = ['__tokens', '__char_length_stats']

    def __init__(self, tokens: Optional[List[SpatialToken]] = None):
        if tokens is None:
            tokens = []
        assert tokens is not None
        self.__tokens = tokens
        self.__char_length_stats = RunningStats()
        for t in self.__tokens:
            t_bbox = t.bbox
            if t_bbox is not None:
                self.__char_length_stats.push((t_bbox[2] - t_bbox[0]) / len(t.text))

    def tokens(self) -> List[SpatialToken]:
        """
        Returns a copy of the tokens in the line.
        """
        return list(self.__tokens)

    def add_token(self, token: SpatialToken) -> bool:
        """
        Returns True if the token is added to the line, False otherwise.
        Token is not added to the line if it is too far away from the line.
        """
        if len(self.__tokens) == 0:
            self.__tokens.append(token)
            self.__char_length_stats.push((token.bbox[2] - token.bbox[0]) / len(token.text))
            return True

        # add token to line if:
        # (1) token is to the right of the last token in the line
        # (2) token is not too far away from the last token in the line
        # (3) token is aligned in the y-axis with the last token in the line.
        if (
            token.bbox[0] > self.__tokens[-1].bbox[2]
            and token.bbox[0]
            < self.__tokens[-1].bbox[2] + self.avg_char_len() * config.DATA_MODEL_LINE_RIGHT_CHAR
            and abs(token.bbox[1] - self.__tokens[-1].bbox[1])
            < config.DATA_MODEL_LINE_NEXT_LINE * (token.bbox[3] - token.bbox[1])
        ):
            self.__tokens.append(token)
            self.__char_length_stats.push((token.bbox[2] - token.bbox[0]) / len(token.text))
            return True
        return False

    def avg_char_len(self) -> float:
        """
        This class keeps a running average on the length of the characters
        in the line. The average is updated every time a new token is added
        to the line. This function returns the average character length at
        the time of the call.
        """
        return self.__char_length_stats.mean()

    def bbox(self) -> np.ndarray:
        """
        Returns:
        --------
        - np.ndarray with format (xmin, ymin, xmax, ymax)
        """
        min_x = min([t.bbox[0] for t in self.__tokens])
        max_x = max([t.bbox[2] for t in self.__tokens])
        min_y = min([t.bbox[1] for t in self.__tokens])
        max_y = max([t.bbox[3] for t in self.__tokens])
        return np.array([min_x, min_y, max_x, max_y])

    def __len__(self) -> int:
        return len(self.__tokens)

    def __eq__(self, other) -> bool:
        return self.__tokens == other.tokens

    def area(self) -> float:
        """
        Computes the area of the line.
        """
        if len(self.__tokens) == 0:
            return 0
        line_bbox = self.bbox()
        return (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1])

    def compactness(self) -> float:
        """
        Computes the compactness of the line. The compactness is defined as
        the ratio between the sum of the areas of individual tokens in the line
        and the area of the bounding box of the line.
        """
        tokens_areas = sum([t.area() for t in self.__tokens])
        line_area = self.area()
        if line_area == 0:
            return 0
        return tokens_areas / line_area

    def copy(self) -> 'Line':
        return Line(list(self.__tokens))


class BlockDataClass(TextContainer):
    __slots__ = ['text', 'bbox', 'avg_char_length']

    def __init__(self, text: str, bbox: np.ndarray, avg_char_length: int):
        """
        Arguments:
        ----------
        - text: str
        - bbox: np.ndarray with format (xmin, ymin, xmax, ymax)
        - avg_char_length: int
        """
        super().__init__(text)
        self.bbox = bbox
        self.avg_char_length = avg_char_length


class Block:
    """
    A block is a list of lines that overflow from top to bottom.
    """

    __slots__ = ['__lines']

    def __init__(self, lines: Optional[List[Line]] = None):
        if lines is None:
            lines = []
        self.__lines = lines

    def lines(self):
        return list(self.__lines)

    def bbox(self) -> np.ndarray:
        """
        Returns:
        --------
        - np.ndarray with format (xmin, ymin, xmax, ymax)
        """
        line_bboxes = [line.bbox() for line in self.__lines]
        min_x = min([bbox[0] for bbox in line_bboxes])
        max_x = max([bbox[2] for bbox in line_bboxes])
        min_y = min([bbox[1] for bbox in line_bboxes])
        max_y = max([bbox[3] for bbox in line_bboxes])
        return np.array([min_x, min_y, max_x, max_y])

    def area(self) -> float:
        """
        Computes the area of the block.
        """
        if len(self.__lines) == 0:
            return 0
        block_bbox = self.bbox()
        return (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])

    def add_token(self, token: SpatialToken) -> bool:
        """
        Returns True if the token is added to the block, False otherwise.
        Token is not added to the block if it is too far away from the
        other tokens in the block.
        """
        if len(self.__lines) == 0:
            self.__lines.append(Line([token]))
            return True

        last_line = self.__lines[-1]
        if last_line.add_token(token):
            return True

        return self.add_line(Line([token]))

    def add_line(self, line: Line) -> bool:
        """
        Returns True if the line is added to the block, False otherwise.
        Line is not added to the block if it is too far away from the
        other lines in the block.
        """
        if len(self.__lines) == 0:
            self.__lines.append(line)
            return True

        last_line = self.__lines[-1]
        last_line_bbox = last_line.bbox()
        new_line_bbox = line.bbox()
        last_line_height = last_line_bbox[3] - last_line_bbox[1]
        last_line_avg_char_len = last_line.avg_char_len()
        # Add line to block if:
        # (1) line is below the last line in the block
        # (2) line is not too far away from the last line in the block
        # (3) line is aligned in the x-axis with the last line in the block.
        if (
            new_line_bbox[1] > last_line_bbox[3]
            and new_line_bbox[1] - last_line_bbox[3] < last_line_height
            and abs(new_line_bbox[0] - last_line_bbox[0])
            < config.DATA_MODEL_BLOCK_LINE_ALIGNMENT * last_line_avg_char_len
        ):
            self.__lines.append(line)
            return True
        return False

    def __eq__(self, other) -> bool:
        return self.__lines == other.lines

    def tokens(self) -> List[SpatialToken]:
        tokens = []
        for line in self.__lines:
            tokens.extend(line.tokens())
        return tokens

    def __str__(self) -> str:
        tokens = self.tokens()
        return ' '.join([t.text for t in tokens])

    def __repr__(self) -> str:
        return self.__str__()

    def compactness(self) -> float:
        """
        Computes the compactness of the block. The compactness is defined as
        the ratio between the sum of the areas of individual tokens in the block
        and the area of the bounding box of the block.
        """
        tokens_areas = sum([token.area() for token in self.tokens()])
        block_area = self.area()
        if block_area == 0:
            return 0
        return tokens_areas / block_area

    def avg_char_len(self) -> float:
        return sum([line.avg_char_len() for line in self.__lines]) / len(self.__lines)

    def copy(self) -> 'Block':
        return Block([a.copy() for a in self.__lines])
