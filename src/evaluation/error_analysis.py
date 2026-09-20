import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.helpers import load_jsonl

logger = logging.getLogger(__name__)

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

def find_failure_cases(preds_list: List[Tuple[str, List[Dict]]], gt_list: List[Dict], threshold: float) -> List[Dict]:
    """Find and return failure cases with F1 score below a certain threshold."""
    failure_cases = []
    seen_questions = set()
    gt_map = {g["question"]: g for g in gt_list}
    for arch_name, predictions in preds_list:
        for p in predictions:
            q = p["question"]
            if q not in gt_map or q in seen_questions:
                continue
            gt_rec = gt_map[q]
            f1 = compute_f1(p.get("predicted_answer", ""), gt_rec.get("answer", ""))
            if f1 < threshold:
                seen_questions.add(q)
                failure_cases.append({
                    "question_id": gt_rec.get("id", ""),
                    "question": q,
                    "ground_truth": gt_rec.get("answer", ""),
                    "predicted_answer": p.get("predicted_answer", ""),
                    "architecture": arch_name,
                    "retrieved_chunks": p.get("retrieved_chunks", []),
                    "f1_score": round(f1, 3)
                })
    return failure_cases

def save_error_cases(output_path: Path, cases: List[Dict]) -> None:
    """Save error cases in JSONL format, keeping output list capped at 30 items."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subset = cases[:30]
        with open(output_path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Error analysis complete. {len(subset)} failure cases saved. Manual review required.")
    except Exception as e:
        logger.error(f"Failed to save error cases JSONL: {e}")

def plot_error_breakdown(output_path: Path) -> None:
    """Generate and save the placeholder pie chart for error category distribution."""
    try:
        import matplotlib.pyplot as plt
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()
        labels = ["retrieval_failure", "generation_failure", "both", "unanswerable"]
        sizes = [25, 25, 25, 25]
        colors = ["#a78bfa", "#6366f1", "#10b981", "#f59e0b"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
        ax.set_title("Error Category Distribution\n(Manual review required)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot error breakdown: {e}")

def main() -> None:
    """Main orchestrator for error cases loading, filtering, and report plotting."""
    preds_dir = project_root / "results" / "predictions"
    naive = load_jsonl(preds_dir / "naive_rag.jsonl")
    dense = load_jsonl(preds_dir / "dense_rag.jsonl")
    hybrid = load_jsonl(preds_dir / "hybrid_rag.jsonl")
    gt = load_jsonl(project_root / "data" / "test.jsonl")
    
    architectures = [("NaiveRAG", naive), ("DenseRAG", dense), ("HybridRAG", hybrid)]
    failures = find_failure_cases(architectures, gt, 0.3)
    if len(failures) < 10:
        logger.info("Fewer than 10 cases found at threshold 0.3. Retrying with threshold 0.5...")
        failures = find_failure_cases(architectures, gt, 0.5)
    if len(failures) < 5:
        logger.warning(f"Only {len(failures)} failure cases found overall.")
        
    save_error_cases(project_root / "results" / "error_cases.jsonl", failures)
    plot_error_breakdown(project_root / "results" / "error_breakdown.png")

if __name__ == "__main__":
    main()
