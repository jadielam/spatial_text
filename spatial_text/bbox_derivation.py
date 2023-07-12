from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore
from shapely import geometry  # type: ignore

from spatial_text.data_model import Block, Line, SpatialToken, Token
from spatial_text.text.edit_distance import edit_distance
from spatial_text.text.trie import TrieNode, fuzzy_search
from spatial_text.utils.numeric_strings import (
    generate_numeric_candidates,
    is_numeric,
)

PRACTICAL_INFINITY = 100_000_000


def find_candidate_sequences(
    root: TrieNode[SpatialToken],
    query: Tuple[str],
) -> List[Block]:
    """
    Collects all the sequences in the ocr text that closely match
    the tokens in the query.  The function does a best effort to
    return a list of candidates that are at least as long as the
    min_length argument.

    Args:
        root: ocr text represented as a trie.
        query: Tuple of tokens to search for in the OCR text.

    Returns:
        List of candidate sequences.
    """
    candidate_blocks: List[Block] = []

    for word in query:
        token = Token(word)

        if is_numeric(word):
            candidate_tokens = []
            for candidate in generate_numeric_candidates(word):
                candidate_tokens.extend(
                    fuzzy_search(
                        root,
                        Token(candidate),
                        int(len(word) * 0.3),
                    ),  # type: ignore
                )
        else:
            candidate_tokens = fuzzy_search(root, token, int(len(word) * 0.3))

        for ct, _ in candidate_tokens:
            added_blocks_count = 0
            for block in candidate_blocks:
                block_to_test = block.copy()
                if block_to_test.add_token(ct):
                    added_blocks_count += 1
                    candidate_blocks.append(block_to_test)

            if added_blocks_count == 0:
                candidate_blocks.append(Block([Line([ct])]))

    return candidate_blocks


def compute_distance_to_query(
    candidates: List[Block],
    query: Tuple[str],
    alpha=0.5,
    beta=0.5,
) -> List[float]:
    """
    Computes the distance of each candidate sequence to the query sequence.
    The distance is computed using two factors:
    (1) Closeness of the sequence of characters in the candidate to the query.
    (2) Compactness of the of the candidate sequence in the 2D space.

    Args:
        candidates: List of candidate sequences.
        query: Tuple of tokens that will be compared to the candidates.
        alpha: Weight of the closeness factor.
        beta: Weight of the compactness factor.

    Returns:
        List of distances. The length of the list is the same as the length
        of the candidates list.
    """
    distances = []
    for candidate in candidates:
        candidate_seq = ' '.join([t.word for t in candidate.tokens()])
        query_seq = ' '.join(query)
        closeness = edit_distance(candidate_seq, query_seq) / max(
            len(query_seq),
            len(candidate_seq),
        )
        compactness = candidate.compactness()
        distances.append(alpha * closeness + beta * (max(0, 1 - compactness)))
    return distances


def distance_labels_candidates(
    candidate_labels: List[Block],
    candidate_sequences: List[Block],
):
    """
    Computes the distance between the candidate sequences for values and the candidate
    labelsfor those sequences. The distance is computed using the distance between the
    top left corner of the candidate sequence and the top left corner of the candidate
    label. The distance is computed only if the top left corner of the candidate
    sequence is inside a specified polygon around the candidate label, otherwise
    the distance is set to infinity.
    """
    distances = np.full(
        (len(candidate_labels), len(candidate_sequences)),
        PRACTICAL_INFINITY,
    )
    for i, candidate_label in enumerate(candidate_labels):
        for j in range(i, len(candidate_sequences)):
            candidate_sequence = candidate_sequences[j]
            cs_bbox = candidate_sequence.bbox()
            cl_bbox = candidate_label.bbox()
            cl_avg_char_len = candidate_label.avg_char_len()
            cl_right_polygon = geometry.Polygon(
                [
                    (cl_bbox[2], cl_bbox[1]),
                    (cl_bbox[2] + 100 * cl_avg_char_len, cl_bbox[1]),
                    (cl_bbox[2] + 100 * cl_avg_char_len, cl_bbox[3]),
                    (cl_bbox[2], cl_bbox[3]),
                ],
            )
            cl_below_polygon = geometry.Polygon(
                [
                    (cl_bbox[0], cl_bbox[3]),
                    (cl_bbox[2], cl_bbox[3]),
                    (cl_bbox[2], cl_bbox[3] + 10 * cl_avg_char_len),
                    (cl_bbox[0], cl_bbox[3] + 10 * cl_avg_char_len),
                ],
            )
            if cl_right_polygon.contains(
                geometry.Point(cs_bbox[0], (cs_bbox[1] + cs_bbox[3]) / 2),
            ) or cl_below_polygon.contains(geometry.Point(cs_bbox[0], cs_bbox[1])):
                distances[i, j] = np.linalg.norm(
                    np.array(cs_bbox[:2]) - np.array(cl_bbox[:2]),
                )
    return distances


def find_best_sequences(
    root: TrieNode,
    query: Tuple[str],
    scoring_fn,
    top_k=1,
) -> List[Block]:
    candidates = find_candidate_sequences(root, query)
    distances = scoring_fn(candidates, query)
    distances_argsort = np.argsort(distances)
    to_return = []
    for i in range(min(top_k, len(distances))):
        to_return.append(candidates[distances_argsort[i]])
    return to_return


def derive_bboxes_of_extractions(
    ocr: List[Tuple[str, np.ndarray]],
    extractions: Dict[str, Tuple[str]],
    keys_labels: Optional[Dict[str, List[Tuple[str]]]],
) -> Dict[str, Optional[Block]]:
    """
    Derive bounding boxes of extractions from OCR.

    Args:
        ocr: List of tuples of OCR tokens and bounding boxes. Bounding boxes
            are given in the format (xmin, ymin, xmax, ymax)
        extractions: Dictionary of extractions. Note that the extraction
            values are tuples of tokens. This means that the user of this
            function must tokenize the extractions before passing them as
            an argument.
        key_labels: Dictionary of the keys to their corresponding labels.
            The keys here are the same as the keys in the extractions. Also notice,
            as in the case with extractions, that the values are tuples of tokens.
            This means that the user of this function must tokenize the labels
            before passing them as an argument. This is an optional argument.

    Returns:
        Dictionary of the keys to their corresponding bounding boxes. Bounding boxes
        are given in the format (xmin, ymin, xmax, ymax). Note that the bounding boxes
        are identified by the keys, but they correspond to the location of the values
        corresponding to those keys. The value of the key is None if the value is not
        found in the OCR text.
    """
    to_return: Dict[str, Optional[Block]] = {}
    value_to_keys_d: Dict[Tuple[str], List[str]] = defaultdict(list)
    for key, value in extractions.items():
        value_to_keys_d[value].append(key)

    root: TrieNode[SpatialToken] = TrieNode()
    for _i, (word, bbox) in enumerate(ocr):
        token = SpatialToken(word, bbox)
        root.insert(token)

    for value in value_to_keys_d:
        keys = value_to_keys_d[value]
        if len(keys) == 1:
            best_candidates = find_best_sequences(
                root,
                value,
                compute_distance_to_query,
                top_k=1,
            )
            if best_candidates:
                to_return[keys[0]] = best_candidates[0]
            else:
                print(f'Did not find candidate value for key={key}, value={value}')
        elif len(keys) > 1 and keys_labels is not None:
            candidate_sequences = find_best_sequences(
                root,
                value,
                compute_distance_to_query,
                top_k=len(keys),
            )

            candidate_labels: List[Block] = []
            labelCandidate_idx_to_key = {}
            for _, key in enumerate(keys):
                if key in keys_labels:
                    candidate_labels_length = len(candidate_labels)
                    for label in keys_labels[key]:
                        candidate_labels.extend(
                            find_best_sequences(
                                root,
                                label,
                                compute_distance_to_query,
                                top_k=1,
                            ),
                        )

                    for i in range(candidate_labels_length, len(candidate_labels)):
                        labelCandidate_idx_to_key[i] = key

            distance_matrix = distance_labels_candidates(
                candidate_labels,
                candidate_sequences,
            )
            label_ind, value_ind = linear_sum_assignment(distance_matrix)
            assert len(label_ind) == len(value_ind)
            for k in range(len(value_ind)):
                to_return[
                    labelCandidate_idx_to_key[label_ind[k]]
                ] = candidate_sequences[value_ind[k]]
        elif len(keys) > 1 and keys_labels is None:
            for key in keys:
                to_return[key] = None
            print(f'Finding value for multiple keys not supported yet: {keys}={value}')

    return to_return
