# Embedding Benchmark Configuration

Dense and hybrid retrieval use the configured `EMBEDDING_MODEL_NAME` by default.

The configuration can be supplied through environment variables:

- `EMBEDDING_MODEL_NAME`: default dense embedding model. Defaults to `all-MiniLM-L6-v2`.
- `LEGAL_EMBEDDING_MODEL_CANDIDATE`: documented legal-domain candidate. Defaults to `epequeno/legal-embeddings-bge-base`.
- `BENCHMARK_EMBEDDING_MODELS`: comma-separated model identifiers used when the embedding matrix is enabled.
- `RUN_EMBEDDING_MATRIX=true`: runs DenseRAG and HybridRAG for every configured embedding model.

The default benchmark path remains backward compatible and uses the configured default model. Matrix mode writes model-specific prediction files under `results/predictions/` and records the selected model set in `results/embedding_benchmark_manifest.jsonl`.

Example:

```text
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LEGAL_EMBEDDING_MODEL_CANDIDATE=epequeno/legal-embeddings-bge-base
BENCHMARK_EMBEDDING_MODELS=all-MiniLM-L6-v2,epequeno/legal-embeddings-bge-base
RUN_EMBEDDING_MATRIX=true
```

The legal candidate is a SentenceTransformers-compatible model fine-tuned for legal document retrieval. The benchmark should be run on the same dataset, retrieval settings, and evaluation configuration for each candidate so the resulting measurements remain comparable.
