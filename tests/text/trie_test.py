import numpy as np
import pytest

from spatial_text.data_model import SpatialToken, Token
from spatial_text.text.trie import (
    TrieNode, find_node, fuzzy_search,
    prefix_search, search_word,
)


@pytest.fixture
def trie():
    trie = TrieNode()
    trie.insert(Token('abc'))
    trie.insert(Token('abcd'))
    trie.insert(Token('abd'))
    trie.insert(Token('a'))
    trie.insert(Token('b'))
    trie.insert(Token('c'))
    trie.insert(Token('jadiel'))

    return trie


@pytest.fixture
def spatial_token_trie():
    trie = TrieNode()
    trie.insert(SpatialToken('abc', np.array([1, 2, 4, 5])))
    trie.insert(SpatialToken('abc', np.array([2, 3, 4, 5])))
    trie.insert(SpatialToken('abc', np.array([1, 2, 3, 4])))
    trie.insert(SpatialToken('text', np.array([1, 2, 3, 4])))
    return trie


def test_search_word(trie):
    assert search_word(trie, Token('abc')) is True
    assert search_word(trie, Token('abcd')) is True
    assert search_word(trie, Token('test')) is False
    assert search_word(trie, Token('a')) is True
    assert search_word(trie, Token('ba')) is False


def test_search_word_with_spatial_tokens(spatial_token_trie):
    assert search_word(spatial_token_trie, Token('abc')) is True
    assert search_word(spatial_token_trie, Token('abcd')) is False
    assert search_word(spatial_token_trie, Token('text')) is True


def test_find_node(trie):
    assert find_node(trie, Token('ab')) is not None
    assert find_node(trie, Token('')) is trie
    assert find_node(trie, Token('test')) is None
    assert find_node(trie, Token('a')) is not None


def test_prefix_search(trie):
    assert set(prefix_search(trie, Token('ab'))) == {
        Token('abc'),
        Token('abcd'),
        Token('abd'),
    }
    assert set(prefix_search(trie, Token('a'))) == {
        Token('a'),
        Token('abc'),
        Token('abcd'),
        Token('abd'),
    }
    assert set(prefix_search(trie, Token('test'))) == set()
    assert set(prefix_search(trie, Token(''))) == {
        Token('a'),
        Token('abc'),
        Token('abcd'),
        Token('abd'),
        Token('b'),
        Token('c'),
        Token('jadiel'),
    }


def test_prefix_search_with_spatial_tokens(spatial_token_trie):
    assert len(prefix_search(spatial_token_trie, Token('ab'))) == 3
    assert len(prefix_search(spatial_token_trie, Token(''))) == 4


def test_fuzzy_search(trie):
    assert set(fuzzy_search(trie, Token('ab'), 2)) == {
        (Token('a'), 1),
        (Token('b'), 1),
        (Token('abcd'), 2),
        (Token('abd'), 1),
        (Token('abc'), 1),
        (Token('c'), 2),
    }
