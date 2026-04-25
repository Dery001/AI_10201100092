# ============================================================
# test_chunking.py – Unit tests for chunking functions
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.chunking import fixed_size_chunks, paragraph_aware_chunks


SAMPLE_PAGES = [
    {
        "page_number": 1,
        "text": " ".join([f"word{i}" for i in range(600)]),  # 600-word synthetic page
    },
    {
        "page_number": 2,
        "text": " ".join([f"wordB{i}" for i in range(400)]),
    },
]


def test_fixed_size_chunks_produced():
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    assert len(chunks) > 0, "Should produce at least one chunk"


def test_fixed_chunk_metadata():
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "source" in c
        assert len(c["text"].split()) <= 410  # small tolerance


def test_fixed_overlap_creates_extra_chunks():
    # With 1000 total words, chunk_size=400, overlap=80 → stride=320
    # → ceil(1000/320) ≈ 4 chunks
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    assert len(chunks) >= 3, f"Expected ≥3 chunks, got {len(chunks)}"


def test_paragraph_aware_chunks_produced():
    # Create pages with paragraph breaks
    pages = [{"page_number": 1, "text": ("This is a sentence. " * 50 + "\n\n") * 4}]
    chunks = paragraph_aware_chunks(pages, min_words=50, max_words=200)
    assert len(chunks) > 0, "Should produce at least one paragraph-aware chunk"


def test_paragraph_chunk_metadata():
    pages = [{"page_number": 1, "text": ("Word number one. " * 40 + "\n\n") * 3}]
    chunks = paragraph_aware_chunks(pages, min_words=30, max_words=150)
    for c in chunks:
        assert "chunk_id" in c
        assert c["chunk_type"] == "paragraph_aware"


if __name__ == "__main__":
    test_fixed_size_chunks_produced()
    test_fixed_chunk_metadata()
    test_fixed_overlap_creates_extra_chunks()
    test_paragraph_aware_chunks_produced()
    test_paragraph_chunk_metadata()
    print("✅ All chunking tests passed.")
