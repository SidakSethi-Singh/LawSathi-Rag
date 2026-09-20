import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from src.utils.helpers import load_jsonl

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from src.utils import config

logger = logging.getLogger(__name__)

def load_predictions_and_ground_truth() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Load evaluation records from predictions JSONL files and ground truth test file."""
    preds_dir = project_root / "results" / "predictions"
    naive = load_jsonl(preds_dir / "naive_rag_full.jsonl")
    dense = load_jsonl(preds_dir / "dense_rag_full.jsonl")
    hybrid = load_jsonl(preds_dir / "hybrid_rag_full.jsonl")
    gt = load_jsonl(project_root / "data" / "test.jsonl")
    return naive, dense, hybrid, gt

def check_chunk_relevance(chunk: str, gt_answer: str) -> bool:
    """Check if chunk is relevant to ground truth (contains >= 2 case-insensitive alphanumeric words)."""
    gt_words = set(re.findall(r"\w+", gt_answer.lower()))
    chunk_words = set(re.findall(r"\w+", chunk.lower()))
    return len(gt_words.intersection(chunk_words)) >= 2

def compute_em(pred: str, gt: str) -> float:
    """Compute Exact Match score (1.0 if identical after normalizations, else 0.0)."""
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0

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

def evaluate_custom(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Compute custom metrics (EM, Token F1, Precision@5, Recall@5, and Avg Latency)."""
    em_scores, f1_scores, p5_scores, r5_scores, latencies = [], [], [], [], []
    pred_map = {p["question"]: p for p in predictions}
    for gt in ground_truth:
        q = gt["question"]
        if q not in pred_map:
            continue
        p = pred_map[q]
        pred_ans = p.get("predicted_answer", "")
        gt_ans = gt.get("answer", "")
        chunks = p.get("retrieved_chunks", [])
        em_scores.append(compute_em(pred_ans, gt_ans))
        f1_scores.append(compute_f1(pred_ans, gt_ans))
        rel = sum(1 for c in chunks[:5] if check_chunk_relevance(c, gt_ans))
        p5_scores.append(rel / 5.0)
        r5_scores.append(rel / 5.0)
        latencies.append(p.get("latency_ms", 0.0))
    n = len(em_scores) or 1
    return {
        "Exact Match": sum(em_scores) / n,
        "Token F1": sum(f1_scores) / n,
        "Precision@5": sum(p5_scores) / n,
        "Recall@5": sum(r5_scores) / n,
        "Avg Latency (ms)": sum(latencies) / n
    }

def evaluate_ragas(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Evaluate using RAGAS framework if possible, falling back to empty dict on failure."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        pred_map = {p["question"]: p for p in predictions}
        aligned_q = [g["question"] for g in ground_truth if g["question"] in pred_map]
        data = {
            "question": aligned_q,
            "answer": [pred_map[q]["predicted_answer"] for q in aligned_q],
            "contexts": [pred_map[q]["retrieved_chunks"] for q in aligned_q],
            "ground_truth": [g["answer"] for g in ground_truth if g["question"] in pred_map]
        }
        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        res = evaluate(dataset, metrics=metrics)
        return {k: float(v) for k, v in res.items()}
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed or skipped: {e}")
        return {}

def generate_comparison_table(results: Dict[str, Dict[str, float]], output_path: Path) -> pd.DataFrame:
    """Build and save architecture comparison table rounded to 3 decimal places."""
    try:
        df = pd.DataFrame.from_dict(results, orient="index")
        df = df.round(3)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        return df
    except Exception as e:
        logger.error(f"Failed to generate comparison table: {e}")
        return pd.DataFrame()

def generate_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate and save comparison bar charts for each metric."""
    try:
        import matplotlib.pyplot as plt
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.ioff()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#a78bfa", "#6366f1", "#10b981"]
            df[col].plot(kind="bar", color=colors[:len(df)], ax=ax)
            ax.set_title(col, fontsize=12, fontweight="bold", pad=15)
            ax.set_ylabel("Value")
            ax.set_xticklabels(df.index, rotation=0)
            plt.tight_layout()
            sanitized = col.replace(" ", "_").replace("@", "_").replace("(", "_").replace(")", "_")
            fig.savefig(figures_dir / f"{sanitized}.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to generate figures: {e}")

def generate_html_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a responsive HTML dashboard visualizing architecture benchmarks."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table_html = df.to_html(classes="table", border=0)
        charts = []
        for col in df.columns:
            san_col = col.replace(" ", "_").replace("@", "_").replace("(", "_").replace(")", "_")
            img_rel_path = f"../results/figures/{san_col}.png"
            charts.append(f"""
            <div class="fig-card">
                <h3>{col}</h3>
                <img src="{img_rel_path}" alt="{col} chart">
            </div>
            """)
        charts_html = "\n".join(charts)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LawSaathi-RAG Leaderboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent: #6366f1;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            max-width: 1000px;
            width: 100%;
        }}
        header {{
            text-align: center;
            margin-bottom: 50px;
        }}
        h1 {{
            font-size: 2.5rem;
            margin: 0 0 10px 0;
            background: linear-gradient(135deg, #a78bfa, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        p.subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin: 0;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            margin-bottom: 40px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-secondary);
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .fig-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .fig-card img {{
            max-width: 100%;
            border-radius: 8px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LawSaathi-RAG Leaderboard</h1>
            <p class="subtitle">Systematic performance comparison of Naive, Dense, and Hybrid RAG architectures on Indian Legal QA</p>
        </header>
        <div class="card">
            <h2>Evaluation Metrics Summary</h2>
            {table_html}
        </div>
        <h2>Visualization Charts</h2>
        <div class="grid">
            {charts_html}
        </div>
    </div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")

def main() -> None:
    """Main orchestrator for benchmark load, evaluation, table, chart and HTML report generation."""
    n_preds, d_preds, h_preds, gt = load_predictions_and_ground_truth()
    results = {}
    architectures = [("NaiveRAG", n_preds), ("DenseRAG", d_preds), ("HybridRAG", h_preds)]
    for name, preds in architectures:
        if not preds:
            logger.warning(f"No predictions found for {name}. Skipping.")
            continue
        results[name] = {**evaluate_custom(preds, gt), **evaluate_ragas(preds, gt)}
    if not results:
        logger.error("No evaluation results generated.")
        sys.exit(1)
    df = generate_comparison_table(results, project_root / "results" / "comparison_table.csv")
    generate_figures(df, project_root / "results" / "figures")
    generate_html_report(df, project_root / "demo" / "index.html")
    logger.info("Evaluation complete. Results saved to results/ and demo/")

if __name__ == "__main__":
    main()
