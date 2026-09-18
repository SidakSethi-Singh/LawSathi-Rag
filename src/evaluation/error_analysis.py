import re
import sys
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
from src.utils.helpers import load_jsonl

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

def _tokens(text: str) -> List[str]:
    """Return normalized alphanumeric tokens."""
    return re.findall(r"\b\w+\b", str(text or "").lower())


def compute_f1(pred: str, gt: str) -> float:
    """Compute token F1 while preserving duplicate-token counts."""
    p_tokens = _tokens(pred)
    g_tokens = _tokens(gt)
    if not p_tokens or not g_tokens:
        return 1.0 if p_tokens == g_tokens else 0.0

    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2.0 * precision * recall / (precision + recall)


def chunk_supports_answer(chunk: str, answer: str, min_overlap: int = 2) -> bool:
    """Return whether a retrieved chunk has minimal lexical support for an answer."""
    chunk_tokens = set(_tokens(chunk))
    answer_tokens = set(_tokens(answer))
    if not chunk_tokens or not answer_tokens:
        return False
    required = min(min_overlap, len(answer_tokens))
    return len(chunk_tokens.intersection(answer_tokens)) >= required


def classify_failure(case: Dict) -> str:
    """Classify a failed QA case using explicit evidence available in the record.

    Rules are intentionally conservative:
    - no ground truth -> unanswerable
    - no retrieved chunks, or no chunk with lexical answer support -> retrieval_failure
    - supported retrieval with a poor generated answer -> generation_failure
    - otherwise -> needs_review

    `both` is not inferred because a poor answer after failed retrieval does not provide
    enough evidence to independently attribute a generation failure.
    """
    ground_truth = str(case.get("ground_truth", "") or "").strip()
    if not ground_truth:
        return "unanswerable"

    chunks = case.get("retrieved_chunks") or []
    if not chunks:
        return "retrieval_failure"

    retrieval_supported = any(chunk_supports_answer(chunk, ground_truth) for chunk in chunks)
    if not retrieval_supported:
        return "retrieval_failure"

    f1_score = case.get("f1_score")
    if isinstance(f1_score, (int, float)) and f1_score < 0.5:
        return "generation_failure"

    return "needs_review"


def find_failure_cases(
    preds_list: List[Tuple[str, List[Dict]]], gt_list: List[Dict], threshold: float
) -> List[Dict]:
    """Find failure cases with F1 below threshold and attach a data-driven category."""
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
                case = {
                    "question_id": gt_rec.get("id", ""),
                    "question": q,
                    "ground_truth": gt_rec.get("answer", ""),
                    "predicted_answer": p.get("predicted_answer", ""),
                    "architecture": arch_name,
                    "retrieved_chunks": p.get("retrieved_chunks", []),
                    "f1_score": round(f1, 3),
                }
                case["failure_category"] = classify_failure(case)
                failure_cases.append(case)

    return failure_cases


def save_error_cases(output_path: Path, cases: List[Dict]) -> List[Dict]:
    """Save at most 30 error cases and return exactly the saved subset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = cases[:30]
    with open(output_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Error analysis complete. %d failure cases saved.", len(subset))
    return subset


def build_category_summary(cases: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Build deterministic counts and percentages from classified cases."""
    counts = Counter(case.get("failure_category", "needs_review") for case in cases)
    total = sum(counts.values())
    categories = ["retrieval_failure", "generation_failure", "unanswerable", "needs_review"]
    return {
        category: {
            "count": counts.get(category, 0),
            "percentage": round((counts.get(category, 0) / total * 100.0), 2) if total else 0.0,
        }
        for category in categories
    }


def save_category_summary(output_path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    """Persist the computed breakdown as machine-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def plot_error_breakdown(output_path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    """Plot the observed error-category distribution; never fabricate percentages."""
    try:
        import matplotlib.pyplot as plt  # type: ignore

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()
        observed = [(name, values["count"]) for name, values in summary.items() if values["count"] > 0]

        fig, ax = plt.subplots(figsize=(6, 5))
        if observed:
            labels, sizes = zip(*observed)
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title("Observed Error Category Distribution", fontsize=12, fontweight="bold", pad=15)
        else:
            ax.text(0.5, 0.5, "No failure cases available", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title("Observed Error Category Distribution", fontsize=12, fontweight="bold", pad=15)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot error breakdown: {e}")


def main() -> None:
    """Load failures, classify them, and generate reproducible error-analysis artifacts."""
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
        logger.warning("Only %d failure cases found overall.", len(failures))

    saved_cases = save_error_cases(project_root / "results" / "error_cases.jsonl", failures)
    summary = build_category_summary(saved_cases)
    save_category_summary(project_root / "results" / "error_breakdown.json", summary)
    plot_error_breakdown(project_root / "results" / "error_breakdown.png", summary)


if __name__ == "__main__":
    main()
