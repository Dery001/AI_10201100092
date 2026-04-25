# ============================================================
# run_evaluation.py – Evaluate RAG vs pure LLM on test queries
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import json
import os
from datetime import datetime
from typing import Dict, List

from src.evaluation.adversarial_tests import ADVERSARIAL_QUERIES
from src.generation.llm_client import generate_pure_llm_response

EVAL_OUTPUT_PATH = os.path.join("outputs", "evaluation_results.json")

INSUFFICIENT_PHRASES = [
    "i do not have enough information",
    "not enough information",
    "cannot be found",
    "not mentioned",
    "no information",
]


def _likely_hallucination(response: str, chunks: List[Dict]) -> bool:
    """
    Heuristic: if the response contains confident numeric or named claims
    but none of those tokens appear in any retrieved chunk, flag as
    potential hallucination.
    """
    import re
    numbers_in_response = set(re.findall(r"\b\d[\d,.]+\b", response))
    if not numbers_in_response:
        return False  # no numeric claims to check

    all_chunk_text = " ".join(c.get("text", "") for c in chunks)
    for num in numbers_in_response:
        if num not in all_chunk_text:
            return True
    return False


def _admits_insufficient(response: str) -> bool:
    r = response.lower()
    return any(phrase in r for phrase in INSUFFICIENT_PHRASES)


def run_evaluation(pipeline) -> List[Dict]:
    """
    Run all test queries through both RAG and pure-LLM modes.
    Returns a list of evaluation records.
    """
    results = []

    for test in ADVERSARIAL_QUERIES:
        print(f"[eval] Running: {test['id']} – {test['query'][:60]}")

        # RAG mode
        try:
            rag_result = pipeline.query(test["query"], mode="rag", prompt_version="v3")
            rag_response = rag_result["response"]
            rag_chunks = rag_result["selected_chunks"]
            rag_final_scores = rag_result["final_scores"]
        except Exception as e:
            rag_response = f"[Error] {e}"
            rag_chunks = []
            rag_final_scores = []

        # Pure LLM mode
        try:
            llm_response = generate_pure_llm_response(test["query"])
        except Exception as e:
            llm_response = f"[Error] {e}"

        # Scoring heuristics
        rag_hallucination = _likely_hallucination(rag_response, rag_chunks)
        llm_hallucination = _likely_hallucination(llm_response, [])
        rag_admits = _admits_insufficient(rag_response)

        record = {
            "id": test["id"],
            "query": test["query"],
            "query_type": test["type"],
            "expected_behaviour": test["expected_behaviour"],
            "rag_response": rag_response,
            "llm_response": llm_response,
            "rag_top_chunk_score": rag_final_scores[0] if rag_final_scores else 0.0,
            "rag_num_chunks": len(rag_chunks),
            "rag_potential_hallucination": rag_hallucination,
            "llm_potential_hallucination": llm_hallucination,
            "rag_admits_insufficient": rag_admits,
            "timestamp": datetime.utcnow().isoformat(),
        }
        results.append(record)

    # Save
    os.makedirs("outputs", exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[eval] Saved {len(results)} results to {EVAL_OUTPUT_PATH}")
    return results


def print_summary(results: List[Dict]) -> None:
    total = len(results)
    rag_hall = sum(1 for r in results if r["rag_potential_hallucination"])
    llm_hall = sum(1 for r in results if r["llm_potential_hallucination"])
    admits = sum(1 for r in results if r["rag_admits_insufficient"])

    print("\n══════════════════════════════════════")
    print("  EVALUATION SUMMARY")
    print("══════════════════════════════════════")
    print(f"  Total queries evaluated : {total}")
    print(f"  RAG potential hallucinations : {rag_hall} / {total}")
    print(f"  LLM potential hallucinations : {llm_hall} / {total}")
    print(f"  RAG correctly admitted insufficient info : {admits}")
    print("══════════════════════════════════════\n")
