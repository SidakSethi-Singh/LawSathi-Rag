# LawSathi-RAG

Benchmark comparing three RAG architectures (BM25, dense, hybrid) for question answering over Indian Supreme Court judgments. The code produces the numbers in `results/` and the paper in `report/main.tex`, so whether a measurement is correct matters more than anything else.

## Layout

- `src/preprocessing/`: `curate_benchmark.py` builds `data/{benchmark,train,val,test}.jsonl` (shuffled with seed 42). `cleaners.py` normalizes text; `chunker.py` splits with tiktoken `cl100k_base`, 512 tokens, 50 overlap.
- `src/rag_pipelines/`: `NaiveRAG` (BM25) is the base class. `DenseRAG` and `HybridRAG` subclass it and override `index_documents`, `retrieve` and `generate`, but all three share `NaiveRAG.answer()`, which records `latency_ms`. A change to `answer()` affects every architecture.
- `src/run_full_benchmark.py`: runs all three pipelines over `data/test.jsonl` (200 questions) and writes `results/predictions/*_full.jsonl` (gitignored).
- `src/evaluation/evaluator.py`: computes EM, token F1, Precision@5, Recall@5, average latency and optional RAGAS metrics; writes `results/comparison_table.csv` and `results/figures/`, and regenerates `demo/index.html`.
- `src/evaluation/ablation.py`, `src/evaluation/error_analysis.py`: chunk-size ablation and failure cases.
- `report/main.tex`: the paper. Its tables mirror `results/*.csv`.

Each record in `data/*.jsonl` has `id`, `question`, `answer` and `context_chunks`.

## Running

Each script adds the repo root to `sys.path` and imports `src.*`, so run them directly:

```
python src/run_full_benchmark.py
python src/evaluation/evaluator.py
```

LLM settings come from `.env`, which is never committed: `OPENAI_API_KEY`, `USE_LOCAL_MODEL`, `API_BASE_URL` (defaults to NVIDIA NIM) and `MODEL_NAME`. The pipelines need a live LLM endpoint and there is no test suite or CI, so review by reading code. `python3 -m py_compile <file>` is the only check that runs without credentials.

## Reviewing pull requests

Most PRs come from outside contributors and are paired with an issue. Check, in order:

1. **Benchmark validity.** Does the change alter what a metric or timer measures? Compare metric code against the metric's name and against how `report/main.tex` defines it (for example, the paper describes latency as end-to-end).
2. **Reproducibility.** Results must not depend on `set()` iteration order or unseeded randomness. Seeds stay at 42 unless the PR explains why.
3. **Consistency across architectures.** `generate()` and `run_main()` are duplicated in all three pipeline files; a fix to one copy should be applied to all of them.
4. **Reported results.** If the change would move numbers in `results/` or the paper, the PR should say so, and CSVs or figures should be regenerated rather than hand-edited.
5. **Failure handling.** Don't add `except Exception` blocks that log and return empty values; a broken run then looks like a low score.
6. **Repo hygiene.** No secrets, `.env` files, generated predictions, LaTeX build files, `__pycache__` or large new data files.
7. **Overlap.** Several contributors often fix the same bug. Point out competing PRs so a maintainer can pick one.

Only report problems you have verified in the code, and don't flag style the surrounding code already uses.
