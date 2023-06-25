import numpy as np


def edit_distance(self, word1: str, word2: str, wi = 1, wd = 1, wr = 1) -> int:
    '''
    Computes the edit distance between two strings. The edit distance is the
    minimum number of operations (insertion, deletion, substitution) required
    to transform one string into the other.

    Arguments:
    ----------
    - word1: First string
    - word2: Second string
    - wi: Weight of insertion operation
    - wd: Weight of deletion operation
    - wr: Weight of substitution operation

    Returns:
    --------
    - edit distance between the two strings

    '''
        
    #1. Compute for the case of empty strings
    ed = np.zeros((len(word1) + 1, len(word2) + 1), dtype = np.int32)
    for m in range(1, ed.shape[0]):
        ed[m, 0] = ed[m - 1, 0] + wi
    for n in range(1, ed.shape[1]):
        ed[0, n] = ed[0, n - 1] + wd
            
    #2. Compute the rest of the columns and rows.
    for m in range(len(word1)):
        for n in range(len(word2)):
            if word1[m] == word2[n]:
                ed[m + 1, n + 1] = ed[m, n]
            else:
                ed[m + 1, n + 1] = min(
                    ed[m, n + 1] + wi,
                    ed[m + 1, n] + wd,
                    ed[m, n] + wr
                )
                    
    return ed[ed.shape[0] - 1, ed.shape[1] - 1]
