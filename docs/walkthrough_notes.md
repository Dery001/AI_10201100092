# Video Walkthrough Notes

**CS4241 – Introduction to Artificial Intelligence**  
**Student:** [YOUR FULL NAME] | **Index:** [YOUR INDEX NUMBER]

---

## Walkthrough Structure (suggested ~10 minutes)

---

### 1. Project Introduction (0:00 – 1:00)

- State your name and index number.
- Introduce the project: "This is a custom RAG chatbot built for Academic City that answers questions about Ghana's election results and the 2025 budget statement."
- Briefly mention the tech stack: Python, Streamlit, sentence-transformers, FAISS, BM25, OpenAI.

---

### 2. Codebase Tour (1:00 – 3:00)

Open the `src/` folder and briefly walk through:

- `ingestion/` – loading CSV and PDF
- `preprocessing/` – cleaning + two chunking strategies
- `retrieval/` – embedder, vector store, BM25, hybrid retriever, scoring
- `generation/` – prompt builder (3 versions), LLM client
- `pipeline/` – the orchestrator that ties everything together
- `evaluation/` – adversarial test suite and evaluation runner
- `utils/` – logger and helpers

**Key point to highlight:** "There is no LangChain or LlamaIndex here. Every component is implemented manually."

---

### 3. Chunking Strategies (3:00 – 4:30)

Open `src/preprocessing/chunking.py` and explain:

- **Fixed-size**: "I use a 400-word window with 80-word overlap. The overlap prevents answers from being cut at chunk boundaries."
- **Paragraph-aware**: "This preserves logical paragraph boundaries, so each retrieved chunk reads as a coherent thought rather than an arbitrary text slice."
- Show the `_make_chunk` function and explain the metadata fields.

---

### 4. Domain-Specific Scoring (Innovation) (4:30 – 5:30)

Open `src/retrieval/scoring.py` and explain:

- "The final score is a weighted combination of five factors."
- Walk through the weights: 0.45 vector, 0.30 BM25, 0.10 source match, 0.10 keyword overlap, 0.05 numeric.
- "This is the innovation feature – it uses domain knowledge about the data to improve ranking beyond a simple average."

---

### 5. Live Demo – Factual Query (5:30 – 7:30)

Run `streamlit run app.py` and demonstrate:

1. Enter query: *"How many votes did John Mahama receive in the 2020 election?"*
2. Show the retrieved chunks section – point out chunk IDs, sources, and the three score columns.
3. Expand the score details table.
4. Show the final prompt (expand the panel).
5. Point out the answer and note it is grounded in the retrieved chunk.

---

### 6. Adversarial Query Demo (7:30 – 8:30)

Enter: *"What is the capital of France?"*

- Show that the system retrieves some chunks but they are irrelevant (low scores).
- The answer should say: "I do not have enough information from the provided documents."
- Explain: "This is the hallucination control working. The v3 prompt instructs the model to admit when data is absent."

---

### 7. Evaluation Suite (8:30 – 9:30)

- Expand the evaluation panel in the Streamlit app.
- Click "Run Evaluation" (or show a pre-run results file if time is limited).
- Highlight the hallucination comparison between RAG and pure LLM.
- Note the "RAG Admits Insufficient?" column.

---

### 8. Closing (9:30 – 10:00)

- Summarise: "This project implements all core RAG components manually, demonstrates two chunking strategies, a hybrid retrieval system with domain-specific scoring, three prompt versions, and a full evaluation suite."
- Mention the outputs folder: `chunks.json`, `logs.json`, `evaluation_results.json`.
- Thank the examiner.

---

## Tips for Recording

- Show your terminal and browser side-by-side.
- Zoom in on code sections you explain.
- Keep energy up when walking through the scoring function – this is the innovation piece.
- Have the app already running to avoid long startup waits in the recording.
