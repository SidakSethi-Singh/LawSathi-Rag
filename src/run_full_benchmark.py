import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.helpers import save_jsonl
from src.utils import config
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)

def _load_jsonl(path: Path) -> List[Dict]:
    """Helper function to load line-delimited JSON rows into a list."""
    if not path.exists():
        logger.error(f"Target test file not found at: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_full_corpus(records: List[Dict]) -> List[str]:
    """Extract and deduplicate context chunks across all test records."""
    all_chunks = [c for r in records for c in r.get("context_chunks", [])]
    return list(set(all_chunks))

def run_architecture_benchmark(
    arch_class: type,
    name: str,
    corpus: List[str],
    records: List[Dict],
    out_path: Path,
    embed_model: str | None = None,
) -> None:
    """Run full benchmark for one architecture and optional embedding model."""
    label = f"{name}[{embed_model}]" if embed_model else name
    logger.info(f"Starting full benchmark execution for {label}...")
    start_time = time.perf_counter()
    rag = (
        arch_class(embed_model=embed_model)
        if embed_model is not None and arch_class in (DenseRAG, HybridRAG)
        else arch_class()
    )
    rag.index_documents(corpus)
    results = []
    total = len(records)
    for idx, rec in enumerate(records):
        ans = rag.answer(rec["question"])
        results.append(ans)
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info(f"{name}: {idx + 1}/{total} questions processed")
        time.sleep(2.0)
    duration = time.perf_counter() - start_time
    logger.info(f"{label} completed in {duration:.2f} seconds.")
    save_jsonl(out_path, results)

def main() -> None:
    """Orchestrate full evaluation run across NaiveRAG, DenseRAG, and HybridRAG."""
    test_path = project_root / "data" / "test.jsonl"
    records = _load_jsonl(test_path)
    corpus = get_full_corpus(records)
    logger.info(f"Loaded {len(records)} test records and {len(corpus)} corpus chunks.")
    benchmarks = [
        (NaiveRAG, "NaiveRAG", "naive_rag_full.jsonl"),
    ]
    preds_dir = project_root / "results" / "predictions"
    for arch_cls, name, filename in benchmarks:
        run_architecture_benchmark(arch_cls, name, corpus, records, preds_dir / filename)

    embedding_models = (
        config.BENCHMARK_EMBEDDING_MODELS
        if config.RUN_EMBEDDING_MATRIX
        else (config.EMBEDDING_MODEL_NAME,)
    )
    manifest = {
        "embedding_models": list(embedding_models),
        "default_embedding_model": config.EMBEDDING_MODEL_NAME,
        "legal_embedding_candidate": config.LEGAL_EMBEDDING_MODEL_CANDIDATE,
        "matrix_enabled": config.RUN_EMBEDDING_MATRIX,
    }
    for arch_cls, name in ((DenseRAG, "DenseRAG"), (HybridRAG, "HybridRAG")):
        for embed_model in embedding_models:
            slug = "".join(
                ch.lower() if ch.isalnum() else "_"
                for ch in embed_model
            ).strip("_")
            filename = f"{name.lower()}__{slug}_full.jsonl"
            run_architecture_benchmark(
                arch_cls,
                name,
                corpus,
                records,
                preds_dir / filename,
                embed_model=embed_model,
            )

    save_jsonl(
        project_root / "results" / "embedding_benchmark_manifest.json",
        [manifest],
    )
    logger.info("Full benchmark complete. Predictions saved to results/predictions/")

if __name__ == "__main__":
    main()
