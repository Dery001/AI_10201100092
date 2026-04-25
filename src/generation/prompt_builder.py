# ============================================================
# prompt_builder.py – Three prompt templates for RAG generation
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
#
# V1 – Basic: minimal instructions, raw context injection
# V2 – Hallucination-controlled: explicit "only use context" rule
# V3 – Structured (production): chunk IDs, strict grounding, format rules
# ============================================================

from typing import List, Dict
from src.utils.helpers import truncate_text


MAX_CHUNK_WORDS = 200  # per chunk in the prompt context block


def _build_context_block(chunks: List[Dict], final_scores: List[float]) -> str:
    """Format retrieved chunks into a numbered context section."""
    lines = []
    for i, (chunk, score) in enumerate(zip(chunks, final_scores), start=1):
        text = truncate_text(chunk["text"], max_words=MAX_CHUNK_WORDS)
        source = chunk.get("source", "unknown")
        cid = chunk.get("chunk_id", f"chunk_{i}")
        lines.append(
            f"[{i}] (chunk_id={cid}, source={source}, score={score:.3f})\n{text}"
        )
    return "\n\n".join(lines)


def build_prompt_v1(query: str, chunks: List[Dict], final_scores: List[float]) -> str:
    """V1 – Basic prompt: context + question, minimal guidance."""
    context = _build_context_block(chunks, final_scores)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def build_prompt_v2(query: str, chunks: List[Dict], final_scores: List[float]) -> str:
    """V2 – Hallucination-controlled: tell the model to stay grounded."""
    context = _build_context_block(chunks, final_scores)
    return (
        "You are a helpful assistant. Answer ONLY using the context provided below. "
        "Do NOT use any external knowledge. If the answer is not in the context, "
        "respond: 'I do not have enough information from the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def build_prompt_v3(query: str, chunks: List[Dict], final_scores: List[float]) -> str:
    """
    V3 – Production-grade structured prompt.
    References chunk IDs, enforces precise factual answers,
    and controls context window by limiting per-chunk length.
    """
    context = _build_context_block(chunks, final_scores)
    chunk_ids = ", ".join(c.get("chunk_id", "?") for c in chunks)

    return (
        "You are an academic AI assistant for Academic City, Ghana.\n"
        "You must answer STRICTLY using the provided document chunks.\n"
        "Rules:\n"
        "1. Use only information found in the context below.\n"
        "2. Cite the chunk_id when referencing specific facts (e.g., [chunk_id=csv_42]).\n"
        "3. Be concise and precise. Prefer numbers, names, and dates over vague statements.\n"
        "4. If the answer cannot be found in the context, respond exactly:\n"
        "   'I do not have enough information from the provided documents.'\n"
        "5. Do not speculate or add background knowledge.\n\n"
        f"Retrieved chunks ({chunk_ids}):\n"
        "──────────────────────────────────────\n"
        f"{context}\n"
        "──────────────────────────────────────\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


PROMPT_BUILDERS = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}


def build_prompt(
    query: str,
    chunks: List[Dict],
    final_scores: List[float],
    version: str = "v3",
) -> str:
    """Dispatch to the correct prompt version."""
    builder = PROMPT_BUILDERS.get(version, build_prompt_v3)
    return builder(query, chunks, final_scores)
