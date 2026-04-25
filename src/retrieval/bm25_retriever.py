# ============================================================
# bm25_retriever.py – BM25 keyword-based retrieval
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import re
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Wraps rank_bm25's BM25Okapi for keyword-based retrieval over chunk texts.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []

    def build(self, chunks: List[Dict]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.
        Each chunk must have 'chunk_id' and 'text'.
        """
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[bm25] Built index over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return (chunk_id, raw_bm25_score) for the top_k matches.
        """
        if self.bm25 is None:
            return []

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Pair chunk_ids with scores and sort descending
        paired = list(zip(self.chunk_ids, scores.tolist()))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_k]
