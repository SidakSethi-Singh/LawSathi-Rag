# LawSathi-RAG

A retrieval-augmented generation benchmark for Indian legal question answering.

## Overview

LawSathi-RAG is a research-oriented project for benchmarking different retrieval strategies on Indian legal judgment data. The repository compares three RAG architectures on a legal QA task:

- NaiveRAG using BM25 lexical retrieval
- DenseRAG using semantic embedding retrieval
- HybridRAG combining lexical and semantic signals

The system is designed to answer questions grounded in Indian Supreme Court and High Court judgments while evaluating retrieval quality, generation quality, and latency.

## Why this project exists

Legal information is often spread across large case law repositories, and navigating it requires both domain expertise and careful citation to authoritative sources. This project explores whether a modern RAG pipeline can help answer legal questions from existing judgments while measuring which retrieval strategy performs best on a legal benchmark.

The benchmark is built around Indian legal QA examples, where the goal is to retrieve relevant chunks and produce answers grounded in the text.

## Key features

- Multi-architecture RAG comparison
  - lexical BM25 retrieval
  - dense vector retrieval with sentence embeddings
  - hybrid retrieval combining both approaches
- Legal QA dataset processing with train/validation/test splits
- Evaluation pipeline for:
  - Exact Match
  - Token F1
  - Precision@5
  - Recall@5
  - average latency
- Ablation analysis for chunk size and retrieval settings
- Visualization and HTML reporting output
- Reproducible benchmarking flow from raw records to result tables

## Architecture

The repository implements a standard retrieval-augmented generation flow:

1. Load legal question-answer records from the dataset.
2. Build a corpus of retrieved context chunks.
3. Run retrieval for each query using one of the configured pipelines.
4. Generate an answer from the retrieved context using an LLM API or a local model.
5. Evaluate the prediction against the ground truth.

### Retrieval pipelines

#### NaiveRAG

- Uses BM25 through `rank-bm25`
- Performs lexical matching over chunked legal text
- Best suited for keyword-heavy legal queries

#### DenseRAG

- Uses SentenceTransformers embeddings
- Stores document embeddings in ChromaDB
- Uses semantic similarity for retrieval

#### HybridRAG

- Combines BM25 lexical scores and dense semantic scores
- Normalizes both score sets and merges them with a weighted scheme
- Designed to capture both exact legal terminology and broader semantic relevance

## Dataset

The project uses legal QA records stored under `data/`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `benchmark.jsonl`

Each sample contains a question, answer, and relevant context chunks extracted from Indian legal judgments. The dataset is focused on legal cases and judicial reasoning from Indian courts.

## Project structure

```text
.
├── data/
│   ├── benchmark.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── demo/
│   └── index.html
├── report/
│   └── main.tex
├── results/
│   ├── ablation_study.csv
│   ├── comparison_table.csv
│   ├── error_cases.jsonl
│   └── figures/
├── src/
│   ├── evaluation/
│   │   ├── ablation.py
│   │   ├── error_analysis.py
│   │   └── evaluator.py
│   ├── preprocessing/
│   │   ├── chunker.py
│   │   ├── cleaners.py
│   │   └── curate_benchmark.py
│   ├── rag_pipelines/
│   │   ├── dense_rag.py
│   │   ├── hybrid_rag.py
│   │   ├── naive_rag.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── helpers.py
│   │   └── __init__.py
│   └── run_full_benchmark.py
├── LICENSE
├── README.md
├── requirements.txt
└── report/
```

## Requirements

The project dependencies are defined in `requirements.txt` and include:

- `openai`
- `sentence-transformers`
- `rank-bm25`
- `chromadb`
- `tiktoken`
- `ragas`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `python-dotenv`
- `tqdm`

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/LawSathi-RAG.git
cd LawSathi-RAG
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment configuration

The project reads settings from `src/utils/config.py` and environment variables. If you are using a hosted LLM, create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
USE_LOCAL_MODEL=false
API_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=meta/llama-3.1-8b-instruct
```

If `USE_LOCAL_MODEL=true`, the code expects a local Ollama-compatible server to be available.

## Usage

### Run the full benchmark

This script runs all three architectures over the test split and saves predictions:

```bash
python src/run_full_benchmark.py
```

The output is written under `results/predictions/`.

### Evaluate predictions

This generates comparison metrics and charts:

```bash
python src/evaluation/evaluator.py
```

Outputs include:

- `results/comparison_table.csv`
- `results/figures/*.png`
- `demo/index.html`

### Run ablation study

This script tests chunk-size and retrieval tuning for the hybrid setup:

```bash
python src/evaluation/ablation.py
```

Outputs include:

- `results/ablation_study.csv`
- `results/figures/ablation_study.png`

## Research and evaluation

The repository includes a research write-up in `report/main.tex` comparing retrieval architectures for Indian legal QA. It documents the motivation, benchmark methodology, evaluation metrics, and results.

The evaluation logic is implemented in:

- `src/evaluation/evaluator.py`
- `src/evaluation/ablation.py`
- `src/evaluation/error_analysis.py`

### Metrics used

- Exact Match (EM)
- Token F1
- Precision@5
- Recall@5
- Average latency

## Demo

The generated dashboard is available in:

- `demo/index.html`

This page presents the comparison table and plots produced by the evaluation pipeline.

## Reproducibility notes

To reproduce the benchmark end-to-end:

```bash
python src/run_full_benchmark.py
python src/evaluation/evaluator.py
python src/evaluation/ablation.py
```

The project is designed so results can be regenerated consistently from the dataset and pipeline modules.

## Future work

Potential improvements include:

- multilingual legal retrieval for Indian language documents
- better citation-aware answer generation
- better handling of ambiguous legal questions
- larger benchmark coverage and more evaluation samples
- domain-specific legal embedding models
- stronger temporal and jurisdictional validation

## Contributing

Contributions are welcome. If you want to improve the benchmark, add new pipelines, or improve evaluation, please open a pull request or discuss your idea in an issue.

## License

This project is licensed under the terms of the repository's license. See `LICENSE` for details.
