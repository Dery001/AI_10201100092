# ============================================================
# hybrid_retriever.py – Combine FAISS vector search + BM25 keyword search
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

from typing import List, Dict, Tuple
import numpy as np

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import embed_query
from src.retrieval.scoring import compute_final_scores
from src.utils.helpers import deduplicate_chunks


# ─── Query type classifier ────────────────────────────────────

ELECTION_KEYWORDS = {
    "election", "vote", "votes", "candidate", "party", "parliament",
    "presidential", "ndc", "npp", "mahama", "akufo", "constituency",
    "polling", "winner", "result", "incumbent", "seats", "region",
}

BUDGET_KEYWORDS = {
    "budget", "expenditure", "revenue", "gdp", "fiscal", "tax",
    "ministry", "allocation", "appropriation", "debt", "deficit",
    "surplus", "cedi", "ghc", "billion", "million", "programme",
    "2025", "economic", "growth", "inflation", "spending",
}


def classify_query(query: str) -> str:
    """
    Simple keyword-rule classifier.
    Returns 'election', 'budget', or 'mixed'.
    """
    tokens = set(query.lower().split())
    e_hits = tokens & ELECTION_KEYWORDS
    b_hits = tokens & BUDGET_KEYWORDS

    if e_hits and not b_hits:
        return "election"
    if b_hits and not e_hits:
        return "budget"
    return "mixed"


# ─── Hybrid retrieval ─────────────────────────────────────────

class HybridRetriever:
    """
    Retrieves candidates from both FAISS (semantic) and BM25 (keyword),
    then merges and re-ranks using the domain-specific scoring function.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        chunks_by_id: Dict[str, Dict],
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.chunks_by_id = chunks_by_id

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        pool_size: int = 20,
    ) -> Tuple[List[Dict], Dict, Dict, List[float], str]:
        """
        Full hybrid retrieval pipeline.

        Returns
        -------
        selected_chunks : top_k ranked chunks
        vector_scores   : { chunk_id: score }
        bm25_scores     : { chunk_id: score }
        final_scores    : list of final scores for selected_chunks
        query_type      : classified query domain
        """
        query_type = classify_query(query)

        # 1. Vector search
        q_emb = embed_query(query)
        vec_results = self.vector_store.search(q_emb, top_k=pool_size)
        vector_scores = {cid: sc for cid, sc in vec_results}

        # 2. BM25 keyword search
        bm25_results = self.bm25.search(query, top_k=pool_size)
        bm25_scores = {cid: sc for cid, sc in bm25_results}

        # 3. Merge candidate IDs
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        candidates = [
            self.chunks_by_id[cid]
            for cid in all_ids
            if cid in self.chunks_by_id
        ]

        # 4. Deduplicate
        candidates = deduplicate_chunks(candidates)

        # 5. Domain-specific scoring
        scored = compute_final_scores(
            query=query,
            query_type=query_type,
            candidates=candidates,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
        )

        # 6. Take top_k
        top = scored[:top_k]
        selected_chunks = [item[0] for item in top]
        final_scores    = [item[1] for item in top]

        return selected_chunks, vector_scores, bm25_scores, final_scores, query_type
