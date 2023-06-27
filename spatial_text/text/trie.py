from collections import deque
from typing import Dict, List, Optional


class TrieNode:
    def __init__(self):
        self.word = None
        self.children: Dict[str, TrieNode] = {}

    def insert(self, word: str):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.word = word


def fuzzy_search(trie: TrieNode, word: str, max_distance: int) -> List[str]:
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
    - List of words that are at most max_distance away from word
    """

    def _search_recursive(
        node: TrieNode,
        letter: str,
        word: str,
        previous_row,
        results,
        max_distance,
    ):
        """
        This recursive helper is used by the outer search function. It assumes that
        the previousRow has been filled in already.
        """
        columns = len(word) + 1
        current_row = [previous_row[0] + 1]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):
            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word[column - 1] != letter:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if current_row[-1] <= max_distance and node.word is not None:
            results.append((node.word, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(current_row) <= max_distance:
            for letter in node.children:
                _search_recursive(
                    node.children[letter],
                    letter,
                    word,
                    current_row,
                    results,
                    max_distance,
                )

    # build first row
    currentRow = range(len(word) + 1)

    results: List[str] = []

    # recursively search each branch of the trie
    for letter in trie.children:
        _search_recursive(trie.children[letter], letter, word, currentRow, results, max_distance)

    return results


def find_node(trie_node: TrieNode, prefix: str) -> Optional[TrieNode]:
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
    for ch in prefix:
        if ch not in current_node.children:
            return None
        current_node = current_node.children[ch]
    return current_node


def search_word(trie_node: TrieNode, word: str) -> bool:
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
    final_node = find_node(trie_node, word)
    return bool(final_node is not None and final_node.word is not None)


def prefix_search(trie_node: TrieNode, prefix: str) -> List[str]:
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
    to_return: List[str] = []
    final_node = find_node(trie_node, prefix)

    if final_node is None:
        return to_return

    # Do BFS on nodes, appending words as I find them as terminals
    queue = deque([final_node])
    while len(queue) > 0:
        node = queue.popleft()
        if node.word is not None:
            to_return.append(node.word)
        for child_node in node.children.values():
            queue.append(child_node)

    return to_return
