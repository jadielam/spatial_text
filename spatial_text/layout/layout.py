import heapq
from collections import defaultdict
from typing import Dict, List

import numpy as np
from spatial_text.data_model import Line, Token
from spatial_text.geometric.beta_skeleton import beta_skeleton
from spatial_text.geometric.utils import is_to_the_right


def _build_adjacency_lists(edges: np.ndarray) -> Dict[int, List[int]]:
    """
    Arguments:
    ----------
    - edges: An `np.ndarray` of  of shape (nb_edges, 2) where the
        second dimension represent the indexes of points from the `point` array that
        form an undirected edge in the skeleton graph.
    """
    adj = defaultdict(list)
    for edge in edges:
        adj[edge[0]].append(edge[1])
    return adj


def _find_leftmost_in_row(
    idx: int,
    adj: Dict[int, List[int]],
    unseen_nodes: set,
    tokens: List[Token],
):
    """
    Finds the leftmost token that can be reached by walking left from token with index
    idx.

    This function assumes that this is a beta skeleton network with beta parameter
    of 1, which means that every node will only have one adjacent node to the right
    of it.

    Arguments:
    ---------
    - idx: index of the reference token
    - adj: Adjacency list from the beta-skeleton algorithm with theta of 1
    - unseen_nodes: set of nodes that have not yet been seen.  Only unseen nodes
        are valid to be explored.
    - tokens: The universe of tokens.

    """
    current_node = idx
    while True:
        for n in adj.get(current_node, []):
            if is_to_the_right(tokens[n], tokens[current_node]) and n in unseen_nodes:
                current_node = n
                break
        else:
            return current_node


def tblr_layout(tokens: List[Token]) -> List[Line]:
    """
    Returns a list of lines that go spatially from top to bottom.  The list of tokens
    on each line go spatially from left to right.
    """
    points = np.array(
        [[(t.bbox[2] - t.bbox[0]) / 2, (t.bbox[3] - t.bbox[1] / 2)] for t in tokens],
    )
    adj = _build_adjacency_lists(beta_skeleton(points, 1))
    # Storing in the heap by the y midpoint
    heap = [
        ((tokens[i].bbox[3] - tokens[i].bbox[1]) / 2, i) for i in range(len(tokens))
    ]
    heapq.heapify(heap)
    unseen_nodes = set(range(len(tokens)))

    # 2. Yielding lines of tokens in top-bottom left-right order.
    to_return = []
    while unseen_nodes:
        while True:
            try:
                (_, top_idx) = heapq.heappop(heap)
            except IndexError:
                to_break = True
            finally:
                if top_idx in unseen_nodes or to_break:
                    break
        leftmost = _find_leftmost_in_row(top_idx, adj, unseen_nodes, tokens)
        if leftmost in unseen_nodes:
            unseen_nodes.remove(leftmost)
            line_idxs = [leftmost]
            for n in adj.get(leftmost, []):
                if n in unseen_nodes and is_to_the_right(
                    tokens[leftmost],
                    tokens[n],
                ):
                    line_idxs.append(n)
                    unseen_nodes.remove(n)
            to_return.append(Line([tokens[i] for i in line_idxs]))

    return to_return
