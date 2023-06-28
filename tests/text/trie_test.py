import pytest

from spatial_text.text.trie import (
    TrieNode,
    find_node,
    fuzzy_search,
    prefix_search,
    search_word,
)


@pytest.fixture
def trie():
    trie = TrieNode()
    trie.insert('abc')
    trie.insert('abcd')
    trie.insert('abd')
    trie.insert('a')
    trie.insert('b')
    trie.insert('c')
    trie.insert('jadiel')

    return trie


def test_search_word(trie):
    assert search_word(trie, 'abc') is True
    assert search_word(trie, 'abcd') is True
    assert search_word(trie, 'test') is False
    assert search_word(trie, 'a') is True
    assert search_word(trie, 'ba') is False


def test_find_node(trie):
    assert find_node(trie, 'ab') is not None
    assert find_node(trie, '') is trie
    assert find_node(trie, 'test') is None
    assert find_node(trie, 'a') is not None


def test_prefix_search(trie):
    assert set(prefix_search(trie, 'ab')) == {'abc', 'abcd', 'abd'}
    assert set(prefix_search(trie, 'a')) == {'a', 'abc', 'abcd', 'abd'}
    assert set(prefix_search(trie, 'test')) == set()
    assert set(prefix_search(trie, '')) == {'a', 'abc', 'abcd', 'abd', 'b', 'c', 'jadiel'}


def test_fuzzy_search(trie):
    assert set(fuzzy_search(trie, 'ab', 2)) == {
        ('a', 1),
        ('b', 1),
        ('abcd', 2),
        ('abd', 1),
        ('abc', 1),
        ('c', 2),
    }
