# Architecture Notes

**CS4241 – Introduction to Artificial Intelligence**  
**Student:** [YOUR FULL NAME] | **Index:** [YOUR INDEX NUMBER]

---

## System Overview

The Academic City RAG Assistant is a modular, end-to-end Retrieval-Augmented Generation system. It answers domain-specific questions using two document sources: a Ghana election results CSV and a Ghana 2025 Budget Statement PDF. All components are implemented from scratch.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         DATA LAYER                          │
│                                                             │
│  Ghana_Election_Result.csv      2025_Budget_Statement.pdf   │
│         │                                │                  │
│  load_csv.py (pandas)          load_pdf.py (PyMuPDF)        │
│         │                                │                  │
│  clean_csv.py                  clean_pdf.py                 │
│  (normalise, fill NaN)         (remove boilerplate)         │
│         │                                │                  │
│  csv_to_chunks()               fixed_size_chunks()          │
│  (row → NL text)            OR paragraph_aware_chunks()     │
│         │                                │                  │
│         └──────────────┬─────────────────┘                  │
│                    all_chunks[]                             │
│                 (JSON, saved to outputs/)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      INDEX LAYER                            │
│                                                             │
│  embedder.py                                                │
│  sentence-transformers/all-MiniLM-L6-v2                     │
│  → 384-dim L2-normalised float32 vectors                    │
│         │                        │                          │
│  vector_store.py           bm25_retriever.py                │
│  FAISS IndexFlatIP         BM25Okapi (rank-bm25)            │
│  (exact cosine via IP)     (tokenised corpus)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    RETRIEVAL LAYER                          │
│                                                             │
│  User Query                                                 │
│       │                                                     │
│  hybrid_retriever.py                                        │
│  ┌────────────────────────────────────────────┐             │
│  │ 1. classify_query()  → election/budget/mixed│             │
│  │ 2. FAISS top-20       → (chunk_id, score)  │             │
│  │ 3. BM25 top-20        → (chunk_id, score)  │             │
│  │ 4. merge + deduplicate candidates          │             │
│  │ 5. scoring.py – domain-specific re-rank    │             │
│  │    final = 0.45·vec + 0.30·bm25            │             │
│  │          + 0.10·source + 0.10·kw           │             │
│  │          + 0.05·numeric                    │             │
│  │ 6. top-4 selected chunks                   │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   GENERATION LAYER                          │
│                                                             │
│  prompt_builder.py                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ v1: Basic (context + question)                       │   │
│  │ v2: Hallucination-controlled                         │   │
│  │ v3: Structured (chunk IDs + strict grounding rules)  │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                   │
│  llm_client.py                                              │
│  OpenAI gpt-4o-mini (temp=0.2, max_tokens=512)              │
│         │                                                   │
│  Response text                                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  LOGGING + EVALUATION                       │
│                                                             │
│  logger.py → outputs/logs.json                              │
│  run_evaluation.py → outputs/evaluation_results.json        │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      UI LAYER                               │
│                                                             │
│  app.py (Streamlit)                                         │
│  - Query input                                              │
│  - Retrieved chunks with scores                             │
│  - Score table (vector / BM25 / final)                      │
│  - Final prompt display (optional)                          │
│  - Answer output                                            │
│  - Evaluation runner panel                                  │
│  - Sidebar: top_k, prompt version, chunking strategy, mode  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Ingestion

- `load_csv.py`: Uses `pandas.read_csv` to load the election dataset.
- `load_pdf.py`: Uses `PyMuPDF` (`fitz`) to extract plain text per page. Preserves page numbers as metadata.

### 2. Preprocessing

- `clean_csv.py`: Normalises column names (lowercase, underscores), drops fully-null rows, fills remaining NaNs with `"unknown"`, strips whitespace. Converts each row to a natural-language sentence using `row_to_text()`.
- `clean_pdf.py`: Strips header/footer boilerplate (lone page numbers, repeated headers) using regex patterns. Collapses whitespace.

### 3. Chunking

Two strategies are implemented for PDF:

**Fixed-size chunking (Strategy A)**
- Window of 400 words, stride of 320 words (80-word overlap).
- Provides predictable chunk lengths for uniform FAISS performance.
- Overlap prevents answer truncation at chunk boundaries.

**Paragraph-aware chunking (Strategy B)**
- Splits on paragraph breaks (`\n\n` or multi-space sentence endings).
- Groups paragraphs until target range of 300–500 words is reached.
- One-sentence overlap between adjacent groups.
- Preserves semantic coherence: a retrieved chunk reads as a complete thought.

Both strategies attach metadata: `chunk_id`, `source`, `chunk_type`, `year`, `keywords`, `section_title`, `page_number`.

### 4. Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Embeddings are L2-normalised so inner product equals cosine similarity.
- Batch size 64 for GPU/CPU efficiency.

### 5. FAISS Vector Store

- Index type: `IndexFlatIP` (exact brute-force inner product).
- Exact search chosen over approximate (HNSW/IVF) because the corpus is small enough that latency is acceptable, and exact results are preferable in an academic setting.

### 6. BM25 Keyword Retrieval

- Uses `BM25Okapi` from `rank-bm25`.
- Corpus tokenised to lowercase alphanumeric tokens.
- Provides complementary keyword signal to the semantic vector search.

### 7. Hybrid Retrieval & Domain-Specific Scoring (Innovation)

The `HybridRetriever` class merges candidates from FAISS and BM25, then applies the domain-specific scoring function:

```
final_score = 0.45 × norm(vector_score)
            + 0.30 × norm(bm25_score)
            + 0.10 × source_match_bonus
            + 0.10 × keyword_overlap_bonus
            + 0.05 × numeric_bonus
```

Weight rationale:
- Vector search dominates (0.45) because semantic similarity is generally more reliable.
- BM25 contributes strongly (0.30) for exact keyword matches (party names, budget line items).
- Source bonus (0.10) steers election queries toward election chunks and vice versa.
- Keyword overlap (0.10) rewards literal term matches beyond BM25 scoring.
- Numeric bonus (0.05) surfaces statistical chunks for numeric questions.

### 8. Prompt Builder

Three prompt versions with increasing strictness:
- **v1**: Minimal — injects context and asks the question.
- **v2**: Adds explicit instruction to only use context, with fallback phrase.
- **v3**: Numbered rules, chunk ID citations, strict anti-hallucination clause, concise factual style instruction.

Context window is managed by truncating each chunk to 200 words within the prompt.

### 9. LLM Response Generation

- API: OpenAI `gpt-4o-mini`
- Temperature: 0.2 (for factual accuracy)
- max_tokens: 512
- Pure-LLM mode (no retrieval) implemented for evaluation comparison.

### 10. Streamlit UI

- Two-column chunk display with colour-coded score badges.
- Expandable score table, prompt viewer, and debug panel.
- Sidebar controls: top_k, prompt version, chunking strategy, query mode.
- Inline evaluation runner.

### 11. Evaluation & Logging

- Every query is logged to `outputs/logs.json` with full pipeline trace.
- `run_evaluation.py` tests 12 queries (adversarial + factual) in both RAG and pure-LLM modes.
- Hallucination detection heuristic: numeric claims in response not present in retrieved chunks.

---

## Failure Cases & Fixes

**Failure case 1 – Vague single-word queries**  
Query: `"votes"` → BM25 returns high scores for many chunks; vector search finds only loosely related chunks. Result: low top-score, possible generic answer.  
Fix: source bonus and keyword overlap bonus down-weight irrelevant chunks; v3 prompt instructs model to admit insufficient information.

**Failure case 2 – Cross-domain queries with misleading keywords**  
Query: `"How much did Ghana spend on the 2024 election campaign?"` → Budget chunks score high due to numeric keywords, but no election-spending data exists.  
Fix: query classifier routes to "election", reducing budget chunk source bonuses. Prompt v3 forces admission when data is absent.
