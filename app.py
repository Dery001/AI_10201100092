# ============================================================
# app.py – Academic City RAG Chatbot (Refactored UX)
# ============================================================

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title { 
    font-size: 2.5rem; 
    font-weight: 800; 
    background: -webkit-linear-gradient(45deg, #4F46E5, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title { 
    font-size: 1.1rem; 
    color: #6B7280; 
    margin-bottom: 2rem; 
}

.answer-box { 
    background: #FFFFFF; 
    border: 1px solid #E5E7EB;
    border-top: 4px solid #10B981;
    padding: 1.5rem; 
    border-radius: 12px; 
    font-size: 1.05rem; 
    color: #1F2937;
    line-height: 1.6;
}

.chunk-card { 
    background: #FFFFFF; 
    border: 1px solid #E5E7EB;
    border-left: 4px solid #4F46E5;
    padding: 1rem; 
    border-radius: 10px; 
    margin-bottom: 1rem;
}

.score-badge { 
    display: inline-block;
    background: #EEF2FF; 
    color: #4F46E5;
    border-radius: 999px; 
    padding: 3px 10px; 
    font-size: 0.75rem;
    margin-right: 5px;
}

.prompt-box { 
    background: #F9FAFB; 
    border: 1px solid #E5E7EB;
    border-radius: 8px; 
    padding: 1rem; 
    font-size: 0.85rem;
    font-family: monospace; 
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building RAG index…")
def load_pipeline(chunking_strategy: str):
    from src.pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(
        csv_path="data/Ghana_Election_Result.csv",
        pdf_path="data/2025_Budget_Statement.pdf",
        chunking_strategy=chunking_strategy,
    )
    pipeline.build(force_rebuild=False)
    return pipeline

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/3/3b/Academic_City_University_College_Logo.png/200px-Academic_City_University_College_Logo.png",
        width=120,
    )

    st.header("⚙️ Settings")

    top_k = st.slider("Top-K chunks", 1, 8, 4)

    prompt_version = st.selectbox(
        "Prompt version",
        ["v3 (Production)", "v2 (Safe)", "v1 (Basic)"]
    )
    pv = prompt_version[:2]

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        ["paragraph", "fixed"]
    )

    mode = st.radio("Mode", ["rag", "pure_llm"])

    show_prompt = st.checkbox("Show prompt", False)
    show_debug = st.checkbox("Debug mode", False)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Academic City RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions about Ghana elections & 2025 budget.</div>', unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────
pipeline = load_pipeline(chunking_strategy)
pipeline.top_k = top_k

# ── Input ────────────────────────────────────────────────────
example_queries = [
    "Which party won the 2020 presidential election in Ghana?",
    "What is Ghana's revenue target for 2025?",
]

example = st.selectbox("Example:", ["Custom"] + example_queries)

user_query = st.text_input(
    "Ask a question:",
    value="" if example == "Custom" else example
)

submit = st.button("🔍 Ask")

# ── Session State ─────────────────────────────────────────────
if "show_more" not in st.session_state:
    st.session_state.show_more = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Run Query ────────────────────────────────────────────────
if submit and user_query.strip():
    with st.spinner("Thinking..."):
        st.session_state.last_result = pipeline.query(user_query, prompt_version=pv, mode=mode)

if st.session_state.last_result:
    result = st.session_state.last_result
    chunks = result.get("selected_chunks", [])
    vec_scores = result.get("vector_scores", [])
    bm25_scores = result.get("bm25_scores", [])
    final_scores = result.get("final_scores", [])

    st.markdown("---")

    # ✅ PRIMARY OUTPUT ONLY
    st.markdown("### ✅ Answer")
    st.markdown(
        f'<div class="answer-box">{result["response"]}</div>',
        unsafe_allow_html=True,
    )

    # Confidence (nice touch)
    if final_scores:
        st.caption(f"Confidence: {max(final_scores):.2f}")

    # Toggle button
    if st.button("🔎 More Info"):
        st.session_state.show_more = not st.session_state.show_more

    # ── SECONDARY INFO ───────────────────────────────────────
    if st.session_state.show_more:
        st.markdown("---")
        st.markdown("## 🔍 Details")

        # Chunks
        st.markdown("### 📄 Retrieved Chunks")

        if chunks:
            cols = st.columns(min(len(chunks), 2))
            for i, chunk in enumerate(chunks):
                with cols[i % 2]:
                    st.markdown(f"""
<div class="chunk-card">
<b>{chunk['chunk_id']}</b> | {chunk.get('source','?')}<br>
<span class="score-badge">V: {vec_scores[i]:.2f}</span>
<span class="score-badge">B: {bm25_scores[i]:.2f}</span>
<span class="score-badge">F: {final_scores[i]:.2f}</span>
<br><br>
{chunk['text'][:300]}...
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No chunks retrieved.")

        # Scores
        with st.expander("📊 Scores"):
            import pandas as pd
            df = pd.DataFrame({
                "Chunk": [c["chunk_id"] for c in chunks],
                "Vector": vec_scores,
                "BM25": bm25_scores,
                "Final": final_scores
            })
            st.dataframe(df)

        # Prompt
        if show_prompt:
            with st.expander("🧾 Prompt"):
                st.markdown(
                    f'<div class="prompt-box">{result["final_prompt"]}</div>',
                    unsafe_allow_html=True,
                )

        # Debug
        if show_debug:
            with st.expander("🐛 Debug"):
                st.json({
                    "query_type": result["query_type"],
                    "mode": result["mode"],
                    "chunks": len(chunks),
                })

elif submit:
    st.warning("Enter a question.")