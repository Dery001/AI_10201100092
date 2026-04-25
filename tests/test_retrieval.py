# ============================================================
# test_retrieval.py – Unit tests for retrieval and scoring
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.scoring import compute_final_scores, _normalize_scores, _keyword_overlap
from src.generation.prompt_builder import build_prompt


# ── Shared fixtures ──────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "source": "election_csv", "text": "NDC won the 2020 presidential election in Ghana with John Mahama as candidate.", "keywords": ["ndc", "election", "ghana"]},
    {"chunk_id": "c2", "source": "budget_pdf",   "text": "Ghana 2025 budget allocates 12 billion cedis to education sector spending.", "keywords": ["budget", "education", "ghana"]},
    {"chunk_id": "c3", "source": "election_csv", "text": "NPP secured 137 parliamentary seats in the 2020 general election results.", "keywords": ["npp", "parliament", "seats"]},
    {"chunk_id": "c4", "source": "budget_pdf",   "text": "Total revenue and grants for 2025 is projected at 180 billion Ghana cedis.", "keywords": ["revenue", "grants", "2025"]},
]

EMBEDDING_DIM = 8  # small for testing


def _fake_embeddings(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generate random unit-normalised embeddings."""
    rng = np.random.default_rng(42)
    raw = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ── VectorStore tests ─────────────────────────────────────────

def test_vector_store_add_and_search():
    vs = VectorStore(embedding_dim=EMBEDDING_DIM)
    embeddings = _fake_embeddings(len(SAMPLE_CHUNKS))
    vs.add(SAMPLE_CHUNKS, embeddings)
    assert vs.total_vectors() == len(SAMPLE_CHUNKS)

    query_emb = _fake_embeddings(1)
    results = vs.search(query_emb, top_k=2)
    assert len(results) == 2
    for cid, score in results:
        assert isinstance(cid, str)
        assert -1.0 <= score <= 1.01  # cosine similarity range


def test_vector_store_returns_ranked():
    """Results should be in descending order of similarity."""
    vs = VectorStore(embedding_dim=EMBEDDING_DIM)
    embeddings = _fake_embeddings(len(SAMPLE_CHUNKS))
    vs.add(SAMPLE_CHUNKS, embeddings)
    query_emb = _fake_embeddings(1)
    results = vs.search(query_emb, top_k=4)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be in descending score order"


# ── BM25 tests ────────────────────────────────────────────────

def test_bm25_build_and_search():
    bm25 = BM25Retriever()
    bm25.build(SAMPLE_CHUNKS)
    results = bm25.search("Ghana election NDC 2020", top_k=2)
    assert len(results) == 2
    # election-related chunk should rank higher than pure budget chunk
    top_cid = results[0][0]
    assert top_cid in {"c1", "c3"}, f"Expected election chunk, got {top_cid}"


def test_bm25_ranked_descending():
    bm25 = BM25Retriever()
    bm25.build(SAMPLE_CHUNKS)
    results = bm25.search("revenue budget 2025", top_k=4)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── Scoring tests ─────────────────────────────────────────────

def test_normalize_scores():
    scores = [0.0, 0.5, 1.0, 0.25]
    normed = _normalize_scores(scores)
    assert min(normed) == 0.0
    assert max(normed) == 1.0


def test_keyword_overlap():
    overlap = _keyword_overlap("Ghana election results", "The Ghana election results are available.")
    assert overlap > 0.5, f"Expected high overlap, got {overlap}"

    no_overlap = _keyword_overlap("quantum physics", "Budget allocation for education in Ghana.")
    assert no_overlap == 0.0


def test_compute_final_scores_sorted():
    vec_scores = {"c1": 0.9, "c2": 0.3, "c3": 0.6, "c4": 0.2}
    bm25_scores = {"c1": 5.0, "c2": 1.0, "c3": 3.0, "c4": 0.5}
    scored = compute_final_scores(
        query="Ghana election 2020",
        query_type="election",
        candidates=SAMPLE_CHUNKS,
        vector_scores=vec_scores,
        bm25_scores=bm25_scores,
    )
    final_vals = [s[1] for s in scored]
    assert final_vals == sorted(final_vals, reverse=True), "Scored results must be descending"


# ── Prompt builder tests ──────────────────────────────────────

def test_prompt_v1_contains_query():
    prompt = build_prompt("Who won?", SAMPLE_CHUNKS[:2], [0.9, 0.7], version="v1")
    assert "Who won?" in prompt
    assert "Context:" in prompt


def test_prompt_v3_contains_chunk_ids():
    prompt = build_prompt("Who won?", SAMPLE_CHUNKS[:2], [0.9, 0.7], version="v3")
    assert "c1" in prompt or "c2" in prompt
    assert "do not have enough information" in prompt.lower() or "Rules:" in prompt


if __name__ == "__main__":
    test_vector_store_add_and_search()
    test_vector_store_returns_ranked()
    test_bm25_build_and_search()
    test_bm25_ranked_descending()
    test_normalize_scores()
    test_keyword_overlap()
    test_compute_final_scores_sorted()
    test_prompt_v1_contains_query()
    test_prompt_v3_contains_chunk_ids()
    print("✅ All retrieval and scoring tests passed.")
