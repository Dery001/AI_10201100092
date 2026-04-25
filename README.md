# Academic City RAG Assistant

**Course:** CS4241 – Introduction to Artificial Intelligence  
**Student:** [YOUR FULL NAME]  
**Index Number:** [YOUR INDEX NUMBER]  
**Institution:** Academic City University College, Ghana

---

## Overview

This project implements a **custom Retrieval-Augmented Generation (RAG)** chatbot that answers questions about:

1. **Ghana General Election Results** (CSV dataset)
2. **Ghana 2025 Budget Statement** (PDF document)

All core RAG components are built from scratch — no LangChain, no LlamaIndex.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Data loading | pandas, PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatIP`) |
| Keyword search | rank-bm25 (BM25Okapi) |
| LLM generation | Ollama Cloud (`llama3.2`) |
| Utilities | numpy, scikit-learn, python-dotenv |

---

## Architecture Summary

```
User Query
    │
    ▼
Query Classifier  ──→  domain: election / budget / mixed
    │
    ▼
┌──────────────────────────────────────────────┐
│              Hybrid Retrieval                │
│  FAISS (semantic)  +  BM25 (keyword)         │
│  → merge candidates                          │
│  → domain-specific scoring                  │
│     (vector + BM25 + source + kw + numeric)  │
└──────────────────────────────────────────────┘
    │
    ▼
Context Selection  (top-4 deduped chunks)
    │
    ▼
Prompt Builder  (v1 / v2 / v3 templates)
    │
    ▼
Ollama Cloud (llama3.2)
    │
    ▼
Response  +  Logging
```

---

## Setup Steps

### 1. Clone / unzip the project

```bash
cd ai_[YOUR_INDEX_NUMBER]
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your Ollama Cloud API key
```

### 4. Add data files

Place the following files in the `data/` directory:

```
data/Ghana_Election_Result.csv
data/2025_Budget_Statement.pdf
```

### 5. Run the app

```bash
streamlit run app.py
```

The first run will build the chunk index and embeddings (approximately 30–90 seconds depending on PDF size). Subsequent runs load from `outputs/chunks.json`.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OLLAMA_API_KEY` | Your Ollama Cloud API key (required) |

---

## How Retrieval Works

1. **Ingestion** – CSV rows are converted to natural-language sentences. PDF pages are extracted using PyMuPDF.
2. **Chunking** – PDF is chunked using either fixed-size (400 words, 80 overlap) or paragraph-aware (300–500 word target) strategy.
3. **Embedding** – All chunks are embedded using `all-MiniLM-L6-v2` (384-dim).
4. **FAISS Index** – Embeddings are stored in an `IndexFlatIP` index (exact cosine similarity via normalised inner product).
5. **BM25 Index** – All chunk texts are indexed for keyword-based retrieval.
6. **Hybrid Retrieval** – Both FAISS and BM25 retrieve a candidate pool; candidates are merged and re-ranked.
7. **Domain-Specific Scoring** – Final score = weighted sum of vector score + BM25 score + source-match bonus + keyword-overlap bonus + numeric bonus.

---

## Innovation Feature: Domain-Specific Scoring

Rather than naively averaging vector and BM25 scores, this project applies a **weighted multi-factor scoring function**:

```
final_score = 0.45 × vector_score
            + 0.30 × bm25_score
            + 0.10 × source_match_bonus
            + 0.10 × keyword_overlap_bonus
            + 0.05 × numeric_bonus
```

- **source_match_bonus**: rewards chunks from the predicted domain (election → election chunks)
- **keyword_overlap_bonus**: rewards chunks that share non-stopword query tokens
- **numeric_bonus**: rewards chunks containing numbers when the query asks for statistics

---

## Prompt Versions

| Version | Description |
|---|---|
| v1 | Basic: context + question with no guardrails |
| v2 | Hallucination-controlled: explicit instruction to only use context |
| v3 | Structured (production): chunk IDs, strict grounding, format rules |

---

## Evaluation Summary

Run the evaluation suite from the Streamlit UI or directly:

```bash
python -c "
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.run_evaluation import run_evaluation, print_summary
p = RAGPipeline(); p.build()
results = run_evaluation(p); print_summary(results)
"
```

Key metrics measured:
- Potential hallucination rate (RAG vs pure LLM)
- Correct "insufficient information" admissions on out-of-scope queries
- Top-chunk relevance score on factual queries

Results are saved to `outputs/evaluation_results.json`.

---

## Project Structure

```
ai_index_number/
├── app.py                        # Streamlit UI
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── Ghana_Election_Result.csv
│   └── 2025_Budget_Statement.pdf
├── docs/
│   ├── architecture.md
│   ├── experiment_logs.md
│   ├── evaluation_report.md
│   └── walkthrough_notes.md
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   ├── evaluation/
│   └── utils/
├── outputs/
│   ├── chunks.json
│   ├── logs.json
│   └── evaluation_results.json
└── tests/
    ├── test_chunking.py
    └── test_retrieval.py
```
