# ============================================================
# scoring.py – Domain-specific hybrid scoring (INNOVATION FEATURE)
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
#
# INNOVATION:
#   Rather than combining vector and BM25 scores with a flat average,
#   this module applies a weighted scoring function that also accounts for:
#     - source match bonus (reward chunks from the likely-relevant dataset)
#     - keyword overlap bonus (reward chunks sharing query keywords)
#     - year/numeric bonus (reward chunks mentioning numbers/years in query)
#
#   This mimics domain knowledge: election queries should prefer election
#   chunks, budget queries should prefer budget chunks, numeric questions
#   should prefer chunks that actually contain numbers.
# ============================================================

import re
from typing import List, Dict, Tuple


# Weights for each scoring component
WEIGHT_VECTOR   = 0.45
WEIGHT_BM25     = 0.30
WEIGHT_SOURCE   = 0.10
WEIGHT_KEYWORD  = 0.10
WEIGHT_NUMERIC  = 0.05


def _normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _keyword_overlap(query: str, chunk_text: str) -> float:
    """
    Fraction of query tokens (non-stopword) that appear in chunk text.
    Returns a value in [0, 1].
    """
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "to", "is", "are"}
    q_tokens = set(re.findall(r"\b[a-z]{3,}\b", query.lower())) - stopwords
    if not q_tokens:
        return 0.0
    c_tokens = set(re.findall(r"\b[a-z]{3,}\b", chunk_text.lower()))
    overlap = q_tokens & c_tokens
    return len(overlap) / len(q_tokens)


def _source_match_bonus(query_type: str, chunk_source: str) -> float:
    """
    Return 1.0 if the chunk source matches the detected query domain,
    0.5 for mixed queries, 0.0 for mismatch.
    """
    if query_type == "election" and "election" in chunk_source.lower():
        return 1.0
    if query_type == "budget" and "budget" in chunk_source.lower():
        return 1.0
    if query_type == "mixed":
        return 0.5
    return 0.0


def _numeric_bonus(query: str, chunk_text: str) -> float:
    """
    If the query contains numbers or years, reward chunks that also
    contain numbers (likely statistical/financial content).
    """
    query_has_nums = bool(re.search(r"\d", query))
    if not query_has_nums:
        return 0.0
    chunk_has_nums = bool(re.search(r"\d", chunk_text))
    return 1.0 if chunk_has_nums else 0.0


def compute_final_scores(
    query: str,
    query_type: str,
    candidates: List[Dict],
    vector_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
) -> List[Tuple[Dict, float, float, float]]:
    """
    Compute the final domain-aware score for each candidate chunk.

    Parameters
    ----------
    query        : the user's query string
    query_type   : 'election', 'budget', or 'mixed'
    candidates   : list of chunk dicts (merged from vector + BM25 results)
    vector_scores: { chunk_id: raw_vector_score }
    bm25_scores  : { chunk_id: raw_bm25_score }

    Returns
    -------
    List of (chunk, final_score, vector_score, bm25_score) sorted descending.
    """
    all_vector = [vector_scores.get(c["chunk_id"], 0.0) for c in candidates]
    all_bm25   = [bm25_scores.get(c["chunk_id"], 0.0) for c in candidates]

    norm_vector = _normalize_scores(all_vector)
    norm_bm25   = _normalize_scores(all_bm25)

    results = []
    for i, chunk in enumerate(candidates):
        nv = norm_vector[i]
        nb = norm_bm25[i]
        src = _source_match_bonus(query_type, chunk.get("source", ""))
        kw  = _keyword_overlap(query, chunk.get("text", ""))
        num = _numeric_bonus(query, chunk.get("text", ""))

        final = (
            WEIGHT_VECTOR  * nv +
            WEIGHT_BM25    * nb +
            WEIGHT_SOURCE  * src +
            WEIGHT_KEYWORD * kw +
            WEIGHT_NUMERIC * num
        )
        results.append((chunk, round(final, 4), round(nv, 4), round(nb, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
