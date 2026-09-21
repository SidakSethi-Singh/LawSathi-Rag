import os
import sys
import json
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from src.utils import config
from src.preprocessing.chunker import chunk_documents
from src.rag_pipelines.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)

# Toggle this flag to True for the final 50-question comprehensive ablation run
RUN_FULL_ABLATION: bool = False

def load_test_questions(test_path: Path, num_questions: int) -> List[Dict]:
    """Load test questions and select a random subset using seed 42."""
    if not test_path.exists():
        logger.error(f"Test split not found at {test_path}")
        sys.exit(1)
    with open(test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    return random.sample(records, min(num_questions, len(records)))

def compute_f1(pred: str, gt: str) -> float:
    """Compute Token F1 score for generated vs target answers."""
    p_tokens = pred.strip().lower().split()
    g_tokens = gt.strip().lower().split()
    if not p_tokens or not g_tokens:
        return 1.0 if p_tokens == g_tokens else 0.0
    common = set(p_tokens).intersection(set(g_tokens))
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2.0 * precision * recall / (precision + recall)

def get_ablation_configs(run_full: bool) -> Tuple[int, List[Dict]]:
    """Return number of questions and configurations for ablation run."""
    configs = [
        {"name": "chunk_256", "chunk_size": 256, "overlap": 50, "top_k": 5, "alpha": 0.7},
        {"name": "chunk_512", "chunk_size": 512, "overlap": 50, "top_k": 5, "alpha": 0.7},
        {"name": "chunk_1024", "chunk_size": 1024, "overlap": 50, "top_k": 5, "alpha": 0.7},
    ]
    if run_full:
        configs.extend([
            {"name": "topk_3", "chunk_size": 512, "overlap": 50, "top_k": 3, "alpha": 0.7},
            {"name": "topk_5", "chunk_size": 512, "overlap": 50, "top_k": 5, "alpha": 0.7},
            {"name": "topk_10", "chunk_size": 512, "overlap": 50, "top_k": 10, "alpha": 0.7},
            {"name": "alpha_0.5", "chunk_size": 512, "overlap": 50, "top_k": 5, "alpha": 0.5},
            {"name": "alpha_0.7", "chunk_size": 512, "overlap": 50, "top_k": 5, "alpha": 0.7},
            {"name": "alpha_0.9", "chunk_size": 512, "overlap": 50, "top_k": 5, "alpha": 0.9},
        ])
    return (50 if run_full else 10), configs

def build_ablation_chunks(questions: List[Dict], chunk_size: int, overlap: int) -> List[str]:
    """Chunk each source context independently so generated chunks never cross boundaries."""
    source_documents = list(dict.fromkeys(
        chunk
        for question in questions
        for chunk in question.get("context_chunks", [])
        if chunk and chunk.strip()
    ))
    return chunk_documents(source_documents, chunk_size=chunk_size, overlap=overlap)

def run_config(config_info: Dict, questions: List[Dict]) -> Tuple[float, float]:
    """Run evaluation for a single ablation configuration, returning avg F1 and latency."""
    name = config_info["name"]
    c_size = config_info["chunk_size"]
    overlap = config_info["overlap"]
    top_k = config_info["top_k"]
    alpha = config_info["alpha"]
    logger.info(f"Ablation: running '{name}' (chunk_size={c_size}, top_k={top_k}, alpha={alpha})...")
    re_chunked = build_ablation_chunks(questions, chunk_size=c_size, overlap=overlap)
    rag = HybridRAG(alpha=alpha)
    rag.index_documents(re_chunked)
    f1_scores = []
    latencies = []
    for rec in questions:
        ans = rag.answer(rec["question"], k=top_k)
        f1 = compute_f1(ans["predicted_answer"], rec["answer"])
        f1_scores.append(f1)
        latencies.append(ans["latency_ms"])
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return avg_f1, avg_latency

def plot_ablation_results(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate and save comparison bar charts for ablation study configurations."""
    try:
        import matplotlib.pyplot as plt
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        df.plot(x="config_name", y="avg_f1", kind="bar", color="#6366f1", ax=ax1, legend=False)
        ax1.set_title("Average F1 Score per Configuration", fontsize=11, fontweight="bold", pad=10)
        ax1.set_ylabel("F1 Score")
        ax1.set_xticklabels(df["config_name"], rotation=45, ha="right")
        df.plot(x="config_name", y="avg_latency_ms", kind="bar", color="#a78bfa", ax=ax2, legend=False)
        ax2.set_title("Average Latency (ms) per Configuration", fontsize=11, fontweight="bold", pad=10)
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xticklabels(df["config_name"], rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / "ablation_study.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to generate figures: {e}")

def main() -> None:
    """Main orchestrator for RAG ablation study benchmark pipeline."""
    test_path = project_root / "data" / "test.jsonl"
    num_q, configs = get_ablation_configs(RUN_FULL_ABLATION)
    questions = load_test_questions(test_path, num_q)
    results = []
    for cfg in configs:
        f1, latency = run_config(cfg, questions)
        results.append({
            "config_name": cfg["name"],
            "chunk_size": cfg["chunk_size"],
            "top_k": cfg["top_k"],
            "alpha": cfg["alpha"],
            "avg_f1": round(f1, 3),
            "avg_latency_ms": round(latency, 3)
        })
    df = pd.DataFrame(results)
    out_path = project_root / "results" / "ablation_study.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    plot_ablation_results(df, project_root / "results" / "figures")
    logger.info("Ablation study complete. Results saved to results/ablation_study.csv")

if __name__ == "__main__":
    main()
