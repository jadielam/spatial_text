from collections import deque
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from spatial_text.data_model import TextContainer

T = TypeVar('T', bound=TextContainer)


class TrieNode(Generic[T]):
    def __init__(self):
        self.tokens: List[T] = []
        self.children: Dict[str, TrieNode[T]] = {}

    def insert(self, token: T):
        node = self
        for letter in token.text:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.tokens.append(token)


def fuzzy_search(
    trie: TrieNode[T],
    token: TextContainer,
    max_distance: int,
) -> List[Tuple[T, int]]:
    """
    Searches for all the words in the trie that are at most
    max_distance away from word. This function is useful for
    fuzzy search. The algorithm is based on the Levenshtein
    algorithm for computing edit distance.

    Arguments:
    ----------
    - trie: Root of trie
    - word: Query word
    - max_distance: Maximum distance away from query

    Returns:
    --------
    - List of tuples of words that are at most max_distance away from word and their
    distance.
    """

    def _search_recursive(
        node: TrieNode[T],
        letter: str,
        token: TextContainer,
        previous_row,
        results: List[Tuple[T, int]],
        max_distance: int,
    ):
        """
        This recursive helper is used by the outer search function. It assumes that
        the previousRow has been filled in already.
        """
        columns = len(token.text) + 1
        current_row = [previous_row[0] + 1]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):
            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if token.text[column - 1] != letter:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if current_row[-1] <= max_distance and len(node.tokens) > 0:
            for node_token in node.tokens:
                results.append((node_token, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(current_row) <= max_distance:
            for letter in node.children:
                _search_recursive(
                    node.children[letter],
                    letter,
                    token,
                    current_row,
                    results,
                    max_distance,
                )

    # build first row
    current_row = range(len(token.text) + 1)

    results: List[Tuple[T, int]] = []

    # recursively search each branch of the trie
    for letter in trie.children:
        _search_recursive(
            trie.children[letter],
            letter,
            token,
            current_row,
            results,
            max_distance,
        )

    return results


def find_node(trie_node: TrieNode[T], prefix: TextContainer) -> Optional[TrieNode[T]]:
    """
    Searches for prefix on trie rooted at trie_node and returns the
    last TrieNode of the search. If it cannot find the entire prefix,
    returns None

    Arguments:
    ----------
    - trie_node: Root of trie
    - prefix: Prefix to search for

    Returns:
    --------
    - Last TrieNode of the search if word is found, None otherwise
    """
    current_node = trie_node
    for ch in prefix.text:
        if ch not in current_node.children:
            return None
        current_node = current_node.children[ch]
    return current_node


def search_word(trie_node: TrieNode[T], token: TextContainer) -> bool:
    """
    Returns true if word was found in trie, otherwise returns false

    Arguments:
    ----------
    - trie_node: Root of trie
    - word: Word to search for

    Returns:
    --------
    - True if word was found in trie, False otherwise
    """
    final_node: Optional[TrieNode[T]] = find_node(trie_node, token)
    return bool(final_node is not None and len(final_node.tokens) > 0)


def prefix_search(trie_node: TrieNode[T], prefix: TextContainer) -> List[T]:
    """
    Returns all words in trie that have the given prefix

    Arguments:
    ----------
    - trie_node: Root of trie
    - prefix: Prefix to search for

    Returns:
    --------
    - List of words that have the given prefix
    """
    to_return: List[T] = []
    final_node: Optional[TrieNode[T]] = find_node(trie_node, prefix)

    if final_node is None:
        return to_return

    # Do BFS on nodes, appending words as I find them as terminals
    queue = deque([final_node])
    while len(queue) > 0:
        node = queue.popleft()
        if len(node.tokens) > 0:
            for token in node.tokens:
                to_return.append(token)
        for child_node in node.children.values():
            queue.append(child_node)

    return to_return
