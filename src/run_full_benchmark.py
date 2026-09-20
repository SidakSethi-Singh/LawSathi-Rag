import argparse
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.helpers import save_jsonl, load_jsonl
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LawSaathi-RAG benchmark evaluation.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=project_root / "data" / "test.jsonl",
        help="Path to the JSONL test split (default: data/test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "predictions",
        help="Directory to write prediction JSONL files (default: results/predictions)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between API calls to avoid rate limits (default: 2.0)",
    )
    return parser.parse_args()


def get_full_corpus(records: List[Dict]) -> List[str]:
    """Extract and deduplicate context chunks across all test records."""
    all_chunks = [c for r in records for c in r.get("context_chunks", [])]
    return list(set(all_chunks))

def run_architecture_benchmark(
    arch_class: type, name: str, corpus: List[str], records: List[Dict],
    out_path: Path, top_k: int = 5, sleep: float = 2.0,
) -> None:
    """Run full benchmark for a single architecture, logging progress and saving results."""
    logger.info(f"Starting full benchmark execution for {name} (top_k={top_k})...")
    start_time = time.perf_counter()
    rag = arch_class()
    rag.index_documents(corpus)
    results = []
    total = len(records)
    for idx, rec in enumerate(records):
        ans = rag.answer(rec["question"], k=top_k)
        results.append(ans)
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info(f"{name}: {idx + 1}/{total} questions processed")
        time.sleep(sleep)
    duration = time.perf_counter() - start_time
    logger.info(f"{name} completed in {duration:.2f} seconds.")
    save_jsonl(out_path, results)


def main() -> None:
    """Orchestrate full evaluation run across NaiveRAG, DenseRAG, and HybridRAG."""
    args = parse_args()
    if not args.test_file.exists():
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)
    records = load_jsonl(args.test_file)
    corpus = get_full_corpus(records)
    logger.info(f"Loaded {len(records)} test records and {len(corpus)} corpus chunks.")
    benchmarks = [
        (NaiveRAG, "NaiveRAG", "naive_rag_full.jsonl"),
        (DenseRAG, "DenseRAG", "dense_rag_full.jsonl"),
        (HybridRAG, "HybridRAG", "hybrid_rag_full.jsonl"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arch_cls, name, filename in benchmarks:
        run_architecture_benchmark(
            arch_cls, name, corpus, records,
            args.output_dir / filename,
            top_k=args.top_k,
            sleep=args.sleep,
        )
    logger.info(f"Full benchmark complete. Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()
