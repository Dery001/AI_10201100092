# ============================================================
# vector_store.py – FAISS-backed vector index (built manually)
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

from typing import List, Tuple, Dict
import numpy as np
import faiss


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP index.
    Since embeddings are L2-normalised, inner product == cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        # IndexFlatIP: exact inner-product search (no approximation)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_ids: List[str] = []

    def add(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.
        chunks: list of chunk dicts (must have 'chunk_id')
        embeddings: float32 array of shape (n, dim)
        """
        assert len(chunks) == embeddings.shape[0], "Chunks and embeddings must match."
        self.index.add(embeddings)
        self.chunk_ids.extend([c["chunk_id"] for c in chunks])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the top_k most similar chunks.
        Returns list of (chunk_id, cosine_score) tuples.
        """
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunk_ids[idx], float(score)))

        return results

    def total_vectors(self) -> int:
        return self.index.ntotal
