# ============================================================
# rag_pipeline.py – Main RAG orchestrator
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
#
# Pipeline: Query → Classify → Retrieve → Score → Select Context
#           → Build Prompt → Generate Response → Log
# ============================================================

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.ingestion.load_csv import load_election_csv
from src.ingestion.load_pdf import load_pdf_pages
from src.preprocessing.clean_csv import clean_election_df, csv_to_chunks
from src.preprocessing.clean_pdf import clean_pdf_pages
from src.preprocessing.chunking import fixed_size_chunks, paragraph_aware_chunks
from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_response, generate_pure_llm_response
from src.utils.logger import log_query_session

CHUNKS_PATH = os.path.join("outputs", "chunks.json")


class RAGPipeline:
    """
    Encapsulates the complete RAG system:
    ingestion → chunking → embedding → indexing → retrieval → generation.
    Call `build()` once at startup, then `query()` for each user request.
    """

    def __init__(
        self,
        csv_path: str = "data/Ghana_Election_Result.csv",
        pdf_path: str = "data/2025_Budget_Statement.pdf",
        chunking_strategy: str = "paragraph",  # 'fixed' or 'paragraph'
        top_k: int = 4,
    ):
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.chunking_strategy = chunking_strategy
        self.top_k = top_k

        self.all_chunks: List[Dict] = []
        self.chunks_by_id: Dict[str, Dict] = {}
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.retriever: Optional[HybridRetriever] = None
        self.ready = False

    # ─── Build / index ───────────────────────────────────────

    def build(self, force_rebuild: bool = False) -> None:
        """
        Load data, chunk, embed, and build indexes.
        If chunks.json exists and force_rebuild is False, load from cache.
        """
        if not force_rebuild and os.path.exists(CHUNKS_PATH):
            print("[pipeline] Loading chunks from cache…")
            self._load_chunks_from_cache()
        else:
            print("[pipeline] Building from raw data…")
            self._ingest_and_chunk()
            self._save_chunks_to_cache()

        self._build_indexes()
        self.ready = True
        print(f"[pipeline] Ready. Total chunks: {len(self.all_chunks)}")

    def _ingest_and_chunk(self) -> None:
        """Load documents, clean, and chunk them."""
        chunks: List[Dict] = []

        # ── CSV ─────────────────────────────────────────────
        df = load_election_csv(self.csv_path)
        if df is not None:
            df = clean_election_df(df)
            csv_chunks = csv_to_chunks(df, source="election_csv")
            chunks.extend(csv_chunks)
            print(f"[pipeline] CSV → {len(csv_chunks)} chunks")
        else:
            print("[pipeline] WARNING: CSV not found, skipping.")

        # ── PDF ─────────────────────────────────────────────
        pages = load_pdf_pages(self.pdf_path)
        if pages is not None:
            pages = clean_pdf_pages(pages)
            if self.chunking_strategy == "fixed":
                pdf_chunks = fixed_size_chunks(pages, source="budget_pdf")
            else:
                pdf_chunks = paragraph_aware_chunks(pages, source="budget_pdf")
            chunks.extend(pdf_chunks)
            print(f"[pipeline] PDF → {len(pdf_chunks)} chunks ({self.chunking_strategy})")
        else:
            print("[pipeline] WARNING: PDF not found, skipping.")

        self.all_chunks = chunks

    def _save_chunks_to_cache(self) -> None:
        os.makedirs("outputs", exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.all_chunks, f, indent=2, ensure_ascii=False)
        print(f"[pipeline] Saved {len(self.all_chunks)} chunks to {CHUNKS_PATH}")

    def _load_chunks_from_cache(self) -> None:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.all_chunks = json.load(f)

    def _build_indexes(self) -> None:
        """Embed all chunks and build FAISS + BM25 indexes."""
        print("[pipeline] Generating embeddings…")
        texts = [c["text"] for c in self.all_chunks]
        embeddings = embed_texts(texts)

        self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(self.all_chunks, embeddings)

        self.bm25 = BM25Retriever()
        self.bm25.build(self.all_chunks)

        self.chunks_by_id = {c["chunk_id"]: c for c in self.all_chunks}

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25,
            chunks_by_id=self.chunks_by_id,
        )

    # ─── Query ───────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        prompt_version: str = "v3",
        mode: str = "rag",
    ) -> Dict:
        """
        Run a full RAG query and return a result dict with all intermediate
        artefacts for display in the UI and logging.
        """
        if not self.ready:
            raise RuntimeError("Pipeline not built. Call build() first.")

        # ── Step 1: Retrieval ────────────────────────────────
        selected_chunks, vector_scores, bm25_scores, final_scores, query_type = \
            self.retriever.retrieve(user_query, top_k=self.top_k)

        # ── Step 2: Context selection (already done in retriever) ──
        # Build context string for logging
        selected_context = "\n\n".join(
            f"[{c['chunk_id']}] {c['text'][:200]}…" for c in selected_chunks
        )

        # ── Step 3: Prompt construction ──────────────────────
        final_prompt = build_prompt(
            query=user_query,
            chunks=selected_chunks,
            final_scores=final_scores,
            version=prompt_version,
        )

        # ── Step 4: LLM generation ───────────────────────────
        if mode == "rag":
            response = generate_response(final_prompt)
        else:
            response = generate_pure_llm_response(user_query)

        # ── Step 5: Logging ──────────────────────────────────
        vec_score_list = [vector_scores.get(c["chunk_id"], 0.0) for c in selected_chunks]
        bm25_score_list = [bm25_scores.get(c["chunk_id"], 0.0) for c in selected_chunks]

        log_query_session(
            query=user_query,
            query_type=query_type,
            retrieved_chunks=[
                {"chunk_id": c["chunk_id"], "source": c["source"], "text": c["text"][:300]}
                for c in selected_chunks
            ],
            vector_scores=vec_score_list,
            bm25_scores=bm25_score_list,
            final_scores=final_scores,
            selected_context=selected_context,
            final_prompt=final_prompt,
            response=response,
            mode=mode,
            prompt_version=prompt_version,
        )

        return {
            "query": user_query,
            "query_type": query_type,
            "selected_chunks": selected_chunks,
            "vector_scores": vec_score_list,
            "bm25_scores": bm25_score_list,
            "final_scores": final_scores,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "response": response,
            "mode": mode,
        }
