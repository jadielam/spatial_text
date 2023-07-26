import itertools
from typing import List, Optional, Tuple, Union

import numpy as np

from spatial_text.data_model import SpatialToken, Token
from spatial_text.text.trie import TrieNode, fuzzy_search
from spatial_text.utils.numeric_strings import (
    generate_numeric_candidates,
    is_numeric,
)
from spatial_text.utils.running_stats import RunningStats


class RootNode:
    def __init__(self):
        self.children: List[LayoutNode] = []
        self.char_len_stats = RunningStats()

    def add_token(self, token_str: str, bbox: np.ndarray) -> 'LayoutNode':
        """
        Adds a token to the layout graph. The token is added as a child of this node.
        The token is either added to the right of the current last line, or to a
        new line.  If token cannot be added because is too far away from tokens in
        this layout, it returns None. Otherwise, it returns the node that was added.

        Args:
            token_str: The token string to add.
            bbox: The bounding box of the token to add. The bounding box is a numpy
                array of shape (4,) with the following format: [x_min, y_min, x_max, y_max].
        """
        layout_node = LayoutNode(token_str, bbox, self.char_len_stats.copy(), None)
        self.children.append(layout_node)
        return layout_node


class LayoutNode:
    id_iter = itertools.count()

    def __init__(
        self,
        token: str,
        bbox: np.ndarray,
        char_length_stats: RunningStats,
        line_start_node: Optional['LayoutNode'],
    ):
        """
        Args:
            token: The token represented by this node.
            bbox: The bounding box of the token represented by this node. The
                bounding box is a numpy array of shape (4,) with the following
                format: [x_min, y_min, x_max, y_max].
            line_start_node: The node that starts the line of this node. If this node is
                supposed to start a new block sequence or a new line, then line_start_node
                should be passed as None
            char_length_stats: The running stats of the average character length of the sequence
        """
        self.id = next(LayoutNode.id_iter)
        self.token = token
        self.bbox = bbox
        self.children: List[LayoutNode] = []

        # Char length stats
        char_length_stats.push((bbox[2] - bbox[0]) / len(token))
        self.avg_char_length = char_length_stats.mean()
        self.char_length_stats = char_length_stats

        # Handling optional arguments
        if line_start_node is None:
            self.line_start_node = self
        else:
            self.line_start_node = line_start_node

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def add_token(self, token_str: str, bbox: np.ndarray) -> Optional['LayoutNode']:
        """
        Adds a token to the layout graph. The token is added as a child of this node.
        The token is either added to the right of the current last line, or to a
        new line.  If token cannot be added because is too far away from tokens in
        this layout, it returns None. Otherwise, it returns the node that was added.

        Args:
            token_str: The token string to add.
            bbox: The bounding box of the token to add. The bounding box is a numpy
                array of shape (4,) with the following format: [x_min, y_min, x_max, y_max].
        """
        # 2.1 Probe that token can be added at the end of current line
        # add token to line if:
        # (1) token is to the right of the last token in the line
        # (2) token is not too far away from the last token in the line
        # (3) token is aligned in the y-axis with the last token in the line.
        if (
            bbox[0] > self.bbox[2]
            and bbox[0] < self.bbox[2] + self.avg_char_length * 3
            and abs(bbox[1] - self.bbox[1]) < 0.5 * (bbox[3] - bbox[1])
        ):
            layout_node = LayoutNode(
                token_str,
                bbox,
                self.char_length_stats.copy(),
                self.line_start_node,
            )
            self.children.append(layout_node)
            return layout_node

        # 2.2 Probe that token can be added as a new line.
        # Add new line to layout if:
        # (1) line is below the last line in the block
        # (2) line is not too far away from the last line in the block
        # (3) line is aligned in the x-axis with the last line in the block.
        last_line_height = self.line_start_node.bbox[3] - self.line_start_node.bbox[1]
        if (
            bbox[1] > self.line_start_node.bbox[3]
            and bbox[1] - self.line_start_node.bbox[3] < last_line_height
            and abs(bbox[0] - self.line_start_node.bbox[0]) < 5 * self.avg_char_length
        ):
            layout_node = LayoutNode(token_str, bbox, self.char_length_stats.copy(), None)
            self.children.append(layout_node)
            return layout_node

        return None


def build_layout_graph_for_query(
    trie: TrieNode[SpatialToken],
    query: Tuple[str],
    pruning_distance: int = -1,
) -> RootNode:
    """
    Takes a trie that contains all the individual tokens in an unordered collection, and a query
    that represents the search of a sequence of tokens over that unordered collection,
    and returns a graph that represents the possible sequences of tokens in the collection
    that match the query. The graph is represented as a tree, where each sequence starts at the
    root, and each path from the root to a terminal node represents a possible sequence of tokens
    that closely match the query.

    Args:
        trie: The trie that contains the ocr text.
        query: The query to search for in the ocr text.
        pruning_distance: When looking at the query as a joined sequence of characters
            s (i.e.: s=' '.join(query)), then, if a token in the query starts at
            position i in s with i > pruning_distance, thenwe do not add
            the token to the root layout token.  This is to create simpler trees with
            unecessary branches going out of the root. If pruning_distance is -1,
            then no pruning is done.
    """
    layout_root_node = RootNode()
    layout_nodes: List[LayoutNode] = []

    query_prunning_distance = 0
    for token_str in query:
        query_prunning_distance += len(token_str) + 1

        candidates: List[Tuple[SpatialToken, int]] = []
        if is_numeric(token_str):
            for numeric_candidate in generate_numeric_candidates(token_str):
                candidates.extend(
                    fuzzy_search(
                        trie,
                        Token(numeric_candidate),
                        int(len(token_str) * 0.3),
                    ),  # type: ignore
                )
        else:
            token = Token(token_str)
            candidates = fuzzy_search(trie, token, int(len(token_str) * 0.3))

        layout_nodes_length = len(layout_nodes)
        for candidate in candidates:
            added_nodes_count = 0
            for layout_node in layout_nodes[:layout_nodes_length]:
                added_node = layout_node.add_token(candidate[0].word, candidate[0].bbox)
                if added_node is not None:
                    layout_nodes.append(added_node)
                    added_nodes_count += 1

            # if candidate is does not pass any probe, create a new Layout Node and
            # add connect it to the layout root node, and add it to the layout nodes
            if added_nodes_count == 0 and (
                query_prunning_distance < pruning_distance or pruning_distance == -1
            ):
                layout_node = layout_root_node.add_token(candidate[0].word, candidate[0].bbox)
                layout_nodes.append(layout_node)

    return layout_root_node


def seqs_from_layout_graph(layout_root_node: RootNode) -> List[Tuple[List[str], np.ndarray]]:
    """
    Does search in the graph collecting all the sequences that are possible, together
    with their bounding boxes. All possible sequences are determined by all the paths from the
    root to a terminal.
    """
    stack: List[Tuple[List[str], Union[RootNode, LayoutNode], np.ndarray]] = [
        ([], layout_root_node, np.array([float('inf'), float('inf'), 0, 0])),
    ]
    to_return = []

    while stack:
        seq, node, bbox = stack.pop()
        if node.children:
            for child in node.children:
                new_bbox = np.array(
                    [
                        min(bbox[0], child.bbox[0]),
                        min(bbox[1], child.bbox[1]),
                        max(bbox[2], child.bbox[2]),
                        max(bbox[3], child.bbox[3]),
                    ],
                )
                stack.append((seq + [child.token], child, new_bbox))
        else:
            to_return.append((seq, bbox))

    return to_return


def find_best_sequences(
    trie: TrieNode[SpatialToken],
    query: Tuple[str],
    k=1,
) -> List[Tuple[str, np.ndarray]]:
    """
    Finds the top k best sequences that match the query. The sequences are returned in order of
    relevance. The relevance is determined the edit distance between sequence and query.

    Args:
        trie: The trie that contains the ocr text.
        query: The query to search for in the ocr text.
        k: The number of sequences to return. if k is -1, return all the sequences found.
    """
    layout_root_node = build_layout_graph_for_query(trie, query)

    candidates_trie: TrieNode[SpatialToken] = TrieNode()
    candidate_seqs = seqs_from_layout_graph(layout_root_node)
    for candidate_seq, bbox in candidate_seqs:
        candidate_seq_str = ' '.join(candidate_seq)
        # Note: we are using spatial token here, but in reality, this is not a token,
        # but a complete string.
        seq_token = SpatialToken(candidate_seq_str, bbox)
        candidates_trie.insert(seq_token)

    query_str = ' '.join(query)
    seqs_and_distances = fuzzy_search(candidates_trie, Token(query_str), int(len(query_str) * 0.3))
    seqs_and_distances.sort(key=lambda x: x[1])
    return [(a[0].word, a[0].bbox) for a in seqs_and_distances][:k]
