# Spatial Text
Algorithms for analysis of text in 2D space

# Things to do:
## Speed up computations of bounding box derivation.
1. Speed up the computations for [bounding box derivation](spatial_text/bbox_derivation.py).  The deep copy of data structures in the
`find_candidate_sequences` function is very slow.
2. A better alternative is to substitute the `List[Block]` datastructure by graph datastructure that represents the candidate sequences. In this graph, a path from the root to a terminal represents a candidate sequence (or block). In this path, the node immediately after the root node is the first item of the sequence (or block), and the terminal node is the last node of the block.
    - We start by creating a root node.
    - For each node A in the sequence being searched
    in the collection:
        - Use the trie datastructure to find a list `L` of close candidate tokens.
        - For each candidate `c`, probe each terminal in the graph to see if the node can be attached to it. A node can be attached to another node if:
            1. It is right next to the other node, that is, a continuation of a line.
            2. It is right below the previous node in the sequence that started a line. In order to support this point 2, each node most contain a pointer to the node that started a given line. The first node in a line points to itself.

        Notice that I need to keep track of the terminals that have been already used by previous candidates `c` so as to not attach to those: with the exception of the root node, of course, which receives all the attachments that did not find good candidate terminals.
    - To select the best candidate sequence, simply find all paths from root to a terminal, and for each path, compute its score: (compactness and closeness to query sequence). I believe that we can efficiently implement this using dfs, because we can probably define the scores so that they can be computed in an online manner (similar to how [spatial_text/utils/running_stats.py](spatial_text/utils/running_stats.py) does).

    ## Create [unsupervised layout derivation algorithm](spatial_text/geometric/layout.py)
    1.
