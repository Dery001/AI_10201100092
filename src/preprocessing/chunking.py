# ============================================================
# chunking.py – Fixed-size and paragraph-aware chunking strategies
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# CS4241 – Introduction to Artificial Intelligence
#
# DESIGN JUSTIFICATION:
#   Fixed-size (400 words, 80 overlap): provides predictable context
#   window usage and avoids very long chunks that dilute relevance.
#   Paragraph-aware (300-500 word target): preserves logical boundaries
#   so retrieved chunks are semantically coherent. Overlap between
#   adjacent paragraph groups prevents answer truncation at boundaries.
# ============================================================

import re
from typing import List, Dict, Optional
from src.utils.helpers import extract_keywords, extract_years


# ─── helpers ────────────────────────────────────────────────

def _split_into_words(text: str) -> List[str]:
    return text.split()


def _words_to_text(words: List[str]) -> str:
    return " ".join(words)


def _guess_section_title(text: str) -> Optional[str]:
    """Heuristic: first line that looks like a heading (short + title-case)."""
    for line in text.split("."):
        line = line.strip()
        if 3 <= len(line.split()) <= 10 and line.istitle():
            return line
    return None


def _make_chunk(
    chunk_id: str,
    text: str,
    source: str,
    chunk_type: str,
    page_number: Optional[int] = None,
) -> Dict:
    years = extract_years(text)
    return {
        "chunk_id": chunk_id,
        "source": source,
        "chunk_type": chunk_type,
        "text": text,
        "section_title": _guess_section_title(text),
        "year": years[0] if years else "2025",
        "keywords": extract_keywords(text, top_n=10),
        "page_number": page_number,
    }


# ─── Strategy A: Fixed-size chunking ────────────────────────

def fixed_size_chunks(
    pages: List[Dict],
    source: str = "budget_pdf",
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict]:
    """
    Concatenate all page text, then slide a window of `chunk_size` words
    with `overlap` words between consecutive chunks.

    Justification:
    - chunk_size=400 fits comfortably in the LLM context while providing
      enough context for factual questions.
    - overlap=80 (~20%) prevents answers that span chunk boundaries from
      being missed entirely.
    """
    # Flatten all pages into one word list, tracking page boundaries
    all_words = []
    word_pages = []  # parallel list: which page each word came from
    for page in pages:
        words = _split_into_words(page["text"])
        all_words.extend(words)
        word_pages.extend([page["page_number"]] * len(words))

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(all_words):
        end = min(start + chunk_size, len(all_words))
        word_slice = all_words[start:end]
        page_num = word_pages[start]  # page of the first word in the chunk

        text = _words_to_text(word_slice)
        chunk = _make_chunk(
            chunk_id=f"fixed_{chunk_idx}",
            text=text,
            source=source,
            chunk_type="fixed_size",
            page_number=page_num,
        )
        chunks.append(chunk)

        chunk_idx += 1
        start += chunk_size - overlap  # slide forward by (size - overlap)

    return chunks


# ─── Strategy B: Paragraph-aware chunking ───────────────────

def paragraph_aware_chunks(
    pages: List[Dict],
    source: str = "budget_pdf",
    min_words: int = 300,
    max_words: int = 500,
    sentence_overlap: int = 1,  # number of sentences to overlap
) -> List[Dict]:
    """
    Split text on paragraph breaks (double newlines or line-end patterns),
    then group paragraphs until the target word count range is reached.

    Justification:
    - Preserves section/topic coherence: a retrieved chunk will contain
      complete thoughts rather than cutting mid-sentence.
    - Range 300-500 words is wide enough to capture multi-sentence answers
      without becoming so long that relevance is diluted.
    - One-sentence overlap ensures boundary sentences appear in both chunks.
    """
    # Collect all paragraphs with their source page
    all_paragraphs: List[Dict] = []
    for page in pages:
        raw_paragraphs = re.split(r"\n{2,}|\.\s{2,}", page["text"])
        for para in raw_paragraphs:
            para = para.strip()
            if len(para.split()) >= 10:  # skip very short fragments
                all_paragraphs.append({"text": para, "page": page["page_number"]})

    chunks = []
    chunk_idx = 0
    i = 0
    overlap_sentences: List[str] = []

    while i < len(all_paragraphs):
        accumulated_words = list(overlap_sentences)  # carry-over sentences
        accumulated_pages = []

        while i < len(all_paragraphs):
            para = all_paragraphs[i]
            words_in_para = para["text"].split()
            accumulated_words.extend(words_in_para)
            accumulated_pages.append(para["page"])
            i += 1

            word_count = len(accumulated_words)
            if word_count >= min_words:
                # Check if next paragraph would exceed max
                if i < len(all_paragraphs):
                    next_count = word_count + len(all_paragraphs[i]["text"].split())
                    if next_count > max_words:
                        break
                else:
                    break

        if not accumulated_words:
            break

        text = " ".join(accumulated_words)
        page_num = accumulated_pages[0] if accumulated_pages else None
        chunk = _make_chunk(
            chunk_id=f"para_{chunk_idx}",
            text=text,
            source=source,
            chunk_type="paragraph_aware",
            page_number=page_num,
        )
        chunks.append(chunk)
        chunk_idx += 1

        # Extract last sentence for overlap with next chunk
        sentences = re.split(r"(?<=[.!?])\s+", text)
        overlap_sentences = sentences[-sentence_overlap:] if sentences else []

    return chunks
