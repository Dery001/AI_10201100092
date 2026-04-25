# ============================================================
# helpers.py – Shared utility functions
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_years(text: str) -> List[str]:
    """Pull 4-digit years (1990–2030) from a string."""
    return re.findall(r"\b(19[9][0-9]|20[0-3][0-9])\b", text)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Simple frequency-based keyword extraction.
    Strips stopwords and returns the most common meaningful tokens.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
        "was", "were", "be", "been", "by", "for", "with", "on", "at",
        "from", "as", "it", "its", "this", "that", "which", "we", "our",
        "their", "he", "she", "they", "have", "has", "had", "will",
        "would", "could", "should", "may", "also", "than", "more",
        "into", "about", "up", "out", "not", "no", "but", "if",
    }
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq: dict = {}
    for t in tokens:
        if t not in stopwords:
            freq[t] = freq.get(t, 0) + 1
    sorted_kw = sorted(freq, key=lambda k: freq[k], reverse=True)
    return sorted_kw[:top_n]


def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def deduplicate_chunks(chunks: List[dict], key: str = "chunk_id") -> List[dict]:
    """Remove duplicate chunks by key."""
    seen = set()
    out = []
    for c in chunks:
        val = c.get(key)
        if val not in seen:
            seen.add(val)
            out.append(c)
    return out
