import os
import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import config
from src.utils.helpers import load_jsonl
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG
from src.rag_pipelines.cross_encoder_rag import CrossEncoderRAG

# Page Configuration
st.set_page_config(
    page_title="LawSaathi-RAG Leaderboard & Playground",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">LawSaathi-RAG ⚖️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Benchmarking Retrieval-Augmented Generation Architectures for Indian Legal Question Answering</div>', unsafe_allow_html=True)

# Sidebar System Controls
st.sidebar.title("⚙️ System Settings")
st.sidebar.info(f"**Model Endpoint**: {config.MODEL_NAME}")
st.sidebar.info(f"**Local Ollama**: {config.USE_LOCAL_MODEL}")
st.sidebar.info(f"**Cross-Encoder Model**: {config.RERANKER_MODEL}")

@st.cache_data
def get_sample_corpus():
    test_path = project_root / "data" / "test.jsonl"
    if test_path.exists():
        records = load_jsonl(test_path)
        chunks = []
        for r in records:
            chunks.extend(r.get("context_chunks", []))
        return list(set(chunks)), records
    return [
        "[Section 302 IPC] Punishment for murder: Whoever commits murder shall be punished with death, or imprisonment for life, and fine.",
        "[Section 420 IPC] Cheating and dishonestly inducing delivery of property: Punished with imprisonment up to seven years and fine.",
        "[Article 21] Protection of life and personal liberty: No person shall be deprived of life or liberty except according to procedure law.",
        "[Section 438 CrPC] Direction for grant of bail to person apprehending arrest: Anticipatory bail provisions by High Court or Sessions Court."
    ], []

corpus, test_records = get_sample_corpus()

# Tabs
tab1, tab2, tab3 = st.tabs(["⚖️ Live QA Playground", "📊 Benchmark Leaderboard", "⚙️ Ablation Configurator"])

# ==================== TAB 1: LIVE QA PLAYGROUND ====================
with tab1:
    st.subheader("Interactive Multi-Pipeline Legal QA")
    st.write("Compare real-time retrieval and generation across Naive, Dense, Hybrid, and Cross-Encoder RAG architectures.")

    example_queries = [
        "What is the punishment for cheating under Section 420 IPC?",
        "What is the procedure for anticipatory bail during police arrest?",
        "How is personal liberty protected under Article 21 of the Indian Constitution?",
        "What are the statutory punishments for murder under Indian Penal Code?"
    ]

    selected_example = st.selectbox("📌 Select an example query or type below:", ["Custom Query..."] + example_queries)
    
    if selected_example != "Custom Query...":
        default_query = selected_example
    else:
        default_query = "What is the punishment for cheating under Section 420 IPC?"

    user_query = st.text_input("💬 Enter your legal question:", value=default_query)

    selected_pipelines = st.multiselect(
        "🛠️ Select Architectures to Run:",
        ["NaiveRAG", "DenseRAG", "HybridRAG", "CrossEncoderRAG"],
        default=["HybridRAG", "CrossEncoderRAG"]
    )

    if st.button("🚀 Run Retrieval & QA Benchmark", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            st.divider()
            cols = st.columns(len(selected_pipelines))

            for idx, pipe_name in enumerate(selected_pipelines):
                with cols[idx]:
                    st.markdown(f"### {pipe_name}")
                    with st.spinner(f"Indexing & retrieving via {pipe_name}..."):
                        start_t = time.perf_counter()
                        
                        if pipe_name == "NaiveRAG":
                            rag = NaiveRAG()
                        elif pipe_name == "DenseRAG":
                            rag = DenseRAG()
                        elif pipe_name == "HybridRAG":
                            rag = HybridRAG()
                        else:
                            rag = CrossEncoderRAG()

                        rag.index_documents(corpus)
                        ans_dict = rag.answer(user_query, k=5)
                        elapsed_ms = (time.perf_counter() - start_t) * 1000

                    st.metric("Latency", f"{elapsed_ms:.1f} ms")
                    st.success("**Predicted Answer:**")
                    st.write(ans_dict.get("predicted_answer", "N/A"))

                    with st.expander("📚 Retrieved Source Chunks (Top-5)", expanded=False):
                        for c_idx, chunk in enumerate(ans_dict.get("retrieved_chunks", []), 1):
                            st.caption(f"**Chunk {c_idx}**")
                            st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))

# ==================== TAB 2: LEADERBOARD ====================
with tab2:
    st.subheader("Systematic Architectural Leaderboard")
    csv_path = project_root / "results" / "comparison_table.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_index=False if "Unnamed: 0" in pd.read_csv(csv_path).columns else None)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Architecture"})

        st.dataframe(df.style.highlight_max(axis=0, color="rgba(16, 185, 129, 0.3)"), use_container_width=True)

        st.subheader("Metric Comparison Charts")
        numeric_cols = [c for c in df.columns if c != "Architecture" and pd.api.types.is_numeric_dtype(df[c])]
        
        selected_metric = st.selectbox("Select Metric to Visualize:", numeric_cols)
        fig = px.bar(
            df,
            x="Architecture",
            y=selected_metric,
            color="Architecture",
            title=f"Architectural Comparison: {selected_metric}",
            color_discrete_sequence=["#a78bfa", "#6366f1", "#10b981", "#f59e0b"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No benchmark results file found at `results/comparison_table.csv`. Run `python src/evaluation/evaluator.py` to populate.")

# ==================== TAB 3: ABLATION CONFIGURATOR ====================
with tab3:
    st.subheader("Hyperparameter & Ablation Trade-off Analysis")
    ablation_path = project_root / "results" / "ablation_study.csv"

    if ablation_path.exists():
        adf = pd.read_csv(ablation_path)
        st.dataframe(adf, use_container_width=True)

        fig2 = px.scatter(
            adf,
            x="avg_latency_ms",
            y="avg_f1",
            size="chunk_size",
            color="config_name",
            hover_name="config_name",
            title="Latency vs F1 Score Trade-off by Chunk Size & Configuration"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No ablation study file found at `results/ablation_study.csv`. Run `python src/evaluation/ablation.py` to populate.")
