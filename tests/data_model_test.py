from __future__ import annotations

import numpy as np

from spatial_text.data_model import Block, Line, SpatialToken, Token


def test_token_equality():
    token1 = SpatialToken('test', [1, 2, 4, 5])
    token2 = SpatialToken('test', [1, 2, 4, 5])
    assert token1 == token2
    assert token1 == token1


def test_token_equality_with_none():
    token1 = Token('test')
    token2 = Token('test')
    assert token1 == token2


def test_token_inequality():
    token1 = SpatialToken('test', [1, 2, 4, 5])
    token2 = SpatialToken('test', [1, 2, 4, 6])
    assert token1 != token2


def test_token_area():
    token1 = SpatialToken('test', [1, 2, 4, 5])
    assert token1.area() == 9


def test_token_inequality_with_none():
    token1 = Token('test')
    token2 = SpatialToken('test', [1, 2, 4, 6])
    assert token1 != token2


def test_line_add_token_with_empty_line():
    line = Line()
    token = SpatialToken('test', [1, 2, 4, 5])
    assert line.add_token(token) is True
    assert line.tokens() == [token]


def test_line_add_token_inline():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    token = SpatialToken('test', [5, 2, 8, 6])
    assert line.add_token(token) is True


def test_line_add_token_slightly_below():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    token = SpatialToken('test', [5, 3, 8, 7])
    assert line.add_token(token) is True


def test_line_not_add_token_too_far_below():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    token = SpatialToken('test', [5, 5, 8, 9])
    assert line.add_token(token) is False


def test_line_not_add_token_to_the_left():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    token = SpatialToken('test', [0, 2, 2, 5])
    assert line.add_token(token) is False


def test_line_not_add_token_to_the_right():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    token = SpatialToken('test', [10, 2, 15, 5])
    assert line.add_token(token) is False


def test_block_add_token_with_empty_block():
    block = Block()
    token = SpatialToken('test', [1, 2, 4, 5])
    assert block.add_token(token) is True


def test_block_add_token_inline():
    block = Block([Line([SpatialToken('test', [1, 2, 4, 5])])])
    token = SpatialToken('test', [6, 2, 9, 5])
    assert block.add_token(token) is True
    assert len(block.lines()) == 1


def test_block_add_token_below():
    block = Block([Line([SpatialToken('test', [1, 2, 4, 5])])])
    token = SpatialToken('test', [1, 6, 4, 9])
    assert block.add_token(token) is True
    assert len(block.lines()) == 2


def test_block_not_add_token_too_far_below():
    block = Block([Line([SpatialToken('test', [1, 2, 4, 5])])])
    token = SpatialToken('test', [1, 12, 4, 15])
    assert block.add_token(token) is False
    assert len(block.lines()) == 1


def test_line_bbox():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    assert np.array_equal(line.bbox(), [1, 2, 4, 5])

    line = Line(
        [SpatialToken('test', [1, 2, 4, 5]), SpatialToken('test', [5, 2, 8, 6])],
    )
    assert np.array_equal(line.bbox(), [1, 2, 8, 6])


def test_line_avg_char_len():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    assert line.avg_char_len() == 0.75

    line = Line(
        [SpatialToken('test', [1, 2, 4, 5]), SpatialToken('test', [5, 2, 11, 6])],
    )
    assert line.avg_char_len() == 1.125


def test_line_area():
    line = Line([SpatialToken('test', [1, 2, 4, 5])])
    assert line.area() == 9

    line = Line(
        [SpatialToken('test', [1, 2, 4, 5]), SpatialToken('test', [5, 2, 11, 6])],
    )
    assert line.area() == 40
