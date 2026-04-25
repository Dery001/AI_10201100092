# ============================================================
# logger.py – Structured logging for every RAG pipeline stage
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# CS4241 – Introduction to Artificial Intelligence
# ============================================================

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


LOG_PATH = os.path.join("outputs", "logs.json")


def _load_logs() -> List[Dict]:
    """Load existing logs from disk, or return empty list."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_logs(logs: List[Dict]) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def log_query_session(
    query: str,
    query_type: str,
    retrieved_chunks: List[Dict],
    vector_scores: List[float],
    bm25_scores: List[float],
    final_scores: List[float],
    selected_context: str,
    final_prompt: str,
    response: str,
    mode: str = "rag",
    prompt_version: str = "v3",
) -> str:
    """
    Log a complete RAG query session.
    Returns the session_id for reference.
    """
    session_id = str(uuid.uuid4())[:8]
    entry = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "prompt_version": prompt_version,
        "query": query,
        "query_type": query_type,
        "retrieved_chunks": retrieved_chunks,
        "vector_scores": vector_scores,
        "bm25_scores": bm25_scores,
        "final_scores": final_scores,
        "selected_context": selected_context,
        "final_prompt": final_prompt,
        "response": response,
    }
    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)
    return session_id


def get_all_logs() -> List[Dict]:
    return _load_logs()
