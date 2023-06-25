from __future__ import annotations

import random

from spatial_text.data_model import Block, Line, Token


def test_data_models_equality():
    tokens1 = [Token('test', [1, 2, 4, 5]) for _ in range(10)]
    tokens2 = list(tokens1)
    line1 = Line(tokens1)
    line2 = Line(tokens2)
    assert line1 == line2

    block1 = Block([line1])
    block2 = Block([line2])
    assert block1 == block2


def test_data_models_differences():
    tokens = [Token('word', [1, 2, 3, 4]) for _ in range(10)]
    tokens1 = random.sample(tokens, len(tokens))
    line1 = Line(tokens)
    line2 = Line(tokens1)
    assert line1 != line2

    block1 = Block([line1])
    block2 = Block([line2])
    assert block1 != block2
