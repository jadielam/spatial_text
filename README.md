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
    1. Read tokens from ocr row by row.
    2. Review the graph datastructure used in the section "Speed Up of Bounding Box computation" to represent blocks, because we will use a derivation of it here:
        - Each node in that graph will have a pointer to the node that starts a line
        - Each node can have potentially multiple pointers to next nodes in a block, indicating var
    3. For each token T in ocr:
        1. Probe recursively each of the terminals of the graph to see if it belongs to one of them and add them. For each edge, add the timestep at which it was added.
    4. The tricky part: finding the optimal layout. The idea that I have is this one.
        1. We have collected a list L of the timesteps at which the last token in ocr was added to a terminal. This list of timesteps represents a different layout configuration to be computed.
        2. For each timestep l in L, the actual layout corresponds to exploring the graph following edges that have timestep value less than or equal to l.
        3. The issue why we are in trouble here is because we need to explore multiple blocks at the same time.
        4. Therefore, this exploration needs to be done doing BFS going in order of edge ids (node ids too, because this is a tree). Maybe is even possible to do a BFS-DFS combination.
