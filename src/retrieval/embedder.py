# ============================================================
# embedder.py – Generate sentence embeddings via sentence-transformers
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

# Singleton model to avoid re-loading on every call
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate L2-normalised embeddings for a list of strings.
    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 norm → cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a float32 array of shape (1, embedding_dim).
    """
    return embed_texts([query])
