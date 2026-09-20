# LawSaathi-RAG ⚖️

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21863845.svg)](https://doi.org/10.5281/zenodo.21863845)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

> **Comparing Retrieval-Augmented Generation (RAG) Architectures for Indian Legal Question Answering**

---

## 📖 Overview

Access to timely, accurate legal information remains a major barrier in India due to the sheer complexity and volume of Supreme Court and High Court precedents. While Large Language Models (LLMs) offer strong natural language capabilities, they often suffer from factual hallucinations on domain-specific statutory and case law texts.

**LawSaathi-RAG** is a systematic benchmarking and experimentation framework that evaluates and compares different RAG architectures on Indian Supreme Court judgment datasets.

---

## 🏛️ RAG Architectures Benchmarked

| Pipeline | Retrieval Mechanism | Underlying Technologies | Key Strength |
| :--- | :--- | :--- | :--- |
| **NaiveRAG** | Lexical Matching | Okapi BM25 (`rank-bm25`) | High precision for explicit statutory sections and exact legal terms |
| **DenseRAG** | Semantic Vector Search | `all-MiniLM-L6-v2` + Ephemeral ChromaDB | Semantic understanding of abstract legal concepts and synonyms |
| **HybridRAG** | Ensemble Weighted Fusion | BM25 + Dense Vectors (Min-Max normalized, $\alpha=0.7$) | Balanced lexical recall and semantic relevance with lowest latency |

---

## 📊 Benchmark Results

Evaluated across Indian legal question-answer pairs:

| Architecture | Exact Match | Token F1 | Precision@5 | Recall@5 | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NaiveRAG** | 0.0 | 0.057 | **0.644** | **0.644** | 3818.45 ms |
| **DenseRAG** | 0.0 | 0.057 | 0.375 | 0.375 | 2072.25 ms |
| **HybridRAG** | 0.0 | 0.057 | 0.516 | 0.516 | **1898.01 ms** |

### 🔍 Key Findings from Ablation Studies
- **Chunk-size ablation**: In the checked-in 10-question ablation, 256-, 512-, and 1024-token chunks all achieve F1 0.044; 1024 tokens has the lowest measured latency, so the results do not establish a unique 512-token optimum.
- **Fusion Weight**: Setting dense weight $\alpha = 0.7$ and lexical weight $1 - \alpha = 0.3$ strikes an effective balance for hybrid legal querying.

---

## 📁 Repository Structure

```text
LawSathi-Rag/
├── data/                       # Benchmark data splits (train, val, test)
│   ├── benchmark.jsonl
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── demo/                       # Visual dashboard & leaderboard
│   └── index.html
├── report/                     # Academic paper source
│   └── main.tex
├── results/                    # Benchmark predictions, error cases & charts
│   ├── ablation_study.csv
│   ├── comparison_table.csv
│   ├── error_breakdown.png
│   └── error_cases.jsonl
├── src/                        # Core codebase
│   ├── evaluation/             # Metrics, ablation & error analysis
│   │   ├── ablation.py
│   │   ├── error_analysis.py
│   │   └── evaluator.py
│   ├── preprocessing/          # Cleaning, token chunking & curation
│   │   ├── chunker.py
│   │   ├── cleaners.py
│   │   └── curate_benchmark.py
│   ├── rag_pipelines/          # RAG implementations
│   │   ├── dense_rag.py
│   │   ├── hybrid_rag.py
│   │   └── naive_rag.py
│   ├── utils/                  # Configuration and LLM helpers
│   │   ├── config.py
│   │   └── helpers.py
│   └── run_full_benchmark.py   # Benchmark execution orchestrator
├── .env.example                # Environment variables template
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/codeXsidd/LawSathi-Rag.git
cd LawSathi-Rag
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API key or local model preference:
```bash
cp .env.example .env
```

Key configuration options in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `USE_LOCAL_MODEL`: Set to `true` to use a local Ollama instance (default: `false`).
- `MODEL_NAME`: e.g. `gpt-4o-mini` or `llama3.1`.

---

## 💻 Running the Benchmark Suite

To run all three architectures (Naive, Dense, Hybrid) against the evaluation dataset:
```bash
python src/run_full_benchmark.py
```

To run evaluation metrics, error analysis, and generate comparison charts:
```bash
python src/evaluation/evaluator.py
python src/evaluation/error_analysis.py
python src/evaluation/ablation.py
```

To view the leaderboard in your browser:
Open `demo/index.html` in any web browser.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on setting up your development environment, coding standards, and submitting pull requests.

Potential areas for contribution:
- [ ] Integration of Cross-Encoder Rerankers (e.g. `BAAI/bge-reranker-base`).
- [ ] Statutory section-aware chunking for Indian Penal Code and Constitution articles.
- [ ] Interactive query CLI and Streamlit/Gradio web demo.
- [ ] RAGAS evaluation metrics integration.

---

## 📄 Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@software{singh_lawsathi_rag_2026,
  author       = {Sidak Singh},
  title        = {LawSaathi-RAG: Comparing Retrieval-Augmented Generation Architectures for Indian Legal Question Answering},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21863845},
  url          = {https://doi.org/10.5281/zenodo.21863845}
}
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
