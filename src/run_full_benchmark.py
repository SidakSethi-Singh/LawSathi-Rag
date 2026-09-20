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
from src.utils.helpers import load_jsonl
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)

def get_full_corpus(records: List[Dict]) -> List[str]:
    """Extract and stably deduplicate context chunks across all test records.

    Order is preserved by first occurrence so repeated benchmark runs build the
    retrieval corpus in the same sequence. Using ``set`` here would make corpus
    order dependent on Python hash iteration order and can introduce avoidable
    run-to-run variation in index construction and tie-breaking.
    """
    seen = set()
    corpus = []
    for record in records:
        for chunk in record.get("context_chunks", []):
            if chunk not in seen:
                seen.add(chunk)
                corpus.append(chunk)
    return corpus


def run_architecture_benchmark(
    arch_class: type, name: str, corpus: List[str], records: List[Dict], out_path: Path
) -> None:
    """Run full benchmark for a single architecture, logging progress and saving results."""
    logger.info(f"Starting full benchmark execution for {name}...")
    start_time = time.perf_counter()
    rag = arch_class()
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
    logger.info(f"{name} completed in {duration:.2f} seconds.")
    save_jsonl(out_path, results)


def main() -> None:
    """Orchestrate full evaluation run across NaiveRAG, DenseRAG, and HybridRAG."""
    # Define the path to the benchmark file
    benchmark_path = project_root / "data" / "test.jsonl"
    
    # Load the benchmark data and handle the exit condition explicitly
    benchmark_data = load_jsonl(benchmark_path)
    if not benchmark_data:
        logger.error(f"Benchmark dataset missing or empty at {benchmark_path}")
        sys.exit(1)

    # Extract the deduplicated corpus using the verified dataset
    corpus = get_full_corpus(benchmark_data)
    logger.info(f"Loaded {len(benchmark_data)} test records and {len(corpus)} corpus chunks.")
    benchmarks = [
        (NaiveRAG, "NaiveRAG", "naive_rag_full.jsonl"),
        (DenseRAG, "DenseRAG", "dense_rag_full.jsonl"),
        (HybridRAG, "HybridRAG", "hybrid_rag_full.jsonl")
    ]
    preds_dir = project_root / "results" / "predictions"
    for arch_cls, name, filename in benchmarks:
        # Passed benchmark_data into the runner instead of the old records variable
        run_architecture_benchmark(arch_cls, name, corpus, benchmark_data, preds_dir / filename)
        
    logger.info("Full benchmark complete. Predictions saved to results/predictions/")

if __name__ == "__main__":
    main()
