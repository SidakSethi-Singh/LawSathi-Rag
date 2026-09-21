import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import config
from src.utils.helpers import save_jsonl

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """RAGAS (Retrieval-Augmented Generation Assessment) Legal Evaluation Engine."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (project_root / "results")

    def _compute_fallback_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Heuristic Faithfulness: proportion of predicted answer words grounded in retrieved context."""
        if not answer or not contexts:
            return 0.0
        full_context = " ".join(contexts).lower()
        context_words = set(re.findall(r"\w+", full_context))
        ans_words = re.findall(r"\w+", answer.lower())
        if not ans_words:
            return 1.0
        matched = sum(1 for w in ans_words if w in context_words or len(w) <= 3)
        return round(matched / len(ans_words), 3)

    def _compute_fallback_relevancy(self, question: str, answer: str) -> float:
        """Heuristic Answer Relevancy: lexical alignment between question and predicted answer."""
        if not question or not answer:
            return 0.0
        q_words = set(re.findall(r"\w+", question.lower()))
        ans_words = set(re.findall(r"\w+", answer.lower()))
        if not q_words or not ans_words:
            return 0.0
        overlap = len(q_words.intersection(ans_words))
        return round(min(1.0, (overlap / len(q_words)) * 1.5), 3)

    def _compute_fallback_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """Heuristic Context Recall: proportion of ground truth key terms present in retrieved contexts."""
        if not ground_truth or not contexts:
            return 0.0
        gt_words = set(re.findall(r"\w+", ground_truth.lower()))
        if not gt_words:
            return 1.0
        full_context = " ".join(contexts).lower()
        matched = sum(1 for w in gt_words if w in full_context)
        return round(matched / len(gt_words), 3)

    def _compute_fallback_context_precision(self, ground_truth: str, contexts: List[str]) -> float:
        """Heuristic Context Precision: proportion of retrieved chunks containing ground truth terms."""
        if not ground_truth or not contexts:
            return 0.0
        gt_words = set(re.findall(r"\w+", ground_truth.lower()))
        if not gt_words:
            return 1.0
        relevant_chunks = 0
        for chunk in contexts:
            chunk_words = set(re.findall(r"\w+", chunk.lower()))
            if len(gt_words.intersection(chunk_words)) >= 2:
                relevant_chunks += 1
        return round(relevant_chunks / len(contexts), 3)

    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate predictions using official RAGAS metrics with robust heuristic fallbacks."""
        if not predictions or not ground_truth:
            return {}

        pred_map = {p["question"]: p for p in predictions}
        aligned_records = [g for g in ground_truth if g["question"] in pred_map]

        if not aligned_records:
            return {}

        # Attempt official RAGAS evaluation library call
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

            data = {
                "question": [g["question"] for g in aligned_records],
                "answer": [pred_map[g["question"]].get("predicted_answer", "") for g in aligned_records],
                "contexts": [pred_map[g["question"]].get("retrieved_chunks", []) for g in aligned_records],
                "ground_truth": [g.get("answer", "") for g in aligned_records]
            }
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            res = evaluate(dataset, metrics=metrics)
            results_dict = {
                "Faithfulness": float(res.get("faithfulness", 0.0)),
                "Answer Relevancy": float(res.get("answer_relevancy", 0.0)),
                "Context Precision": float(res.get("context_precision", 0.0)),
                "Context Recall": float(res.get("context_recall", 0.0))
            }
            logger.info("RAGAS evaluation executed cleanly via official framework.")
            return results_dict
        except Exception as e:
            logger.warning(f"RAGAS framework execution fallback triggered ({e}). Running domain heuristic fallback...")

        # Domain Heuristic Fallback Evaluation
        faith_scores, rel_scores, cp_scores, cr_scores = [], [], [], []
        for g in aligned_records:
            q = g["question"]
            p = pred_map[q]
            pred_ans = p.get("predicted_answer", "")
            gt_ans = g.get("answer", "")
            chunks = p.get("retrieved_chunks", [])

            faith_scores.append(self._compute_fallback_faithfulness(pred_ans, chunks))
            rel_scores.append(self._compute_fallback_relevancy(q, pred_ans))
            cp_scores.append(self._compute_fallback_context_precision(gt_ans, chunks))
            cr_scores.append(self._compute_fallback_context_recall(gt_ans, chunks))

        n = len(aligned_records)
        metrics_summary = {
            "Faithfulness": round(sum(faith_scores) / n, 3),
            "Answer Relevancy": round(sum(rel_scores) / n, 3),
            "Context Precision": round(sum(cp_scores) / n, 3),
            "Context Recall": round(sum(cr_scores) / n, 3)
        }

        # Save result artifact
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / "ragas_evaluation.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info(f"Saved RAGAS evaluation summary to {out_file}")
        except Exception as ex:
            logger.error(f"Failed to save RAGAS JSON summary: {ex}")

        return metrics_summary

def run_main() -> None:
    """Sanity test for RAGAS Evaluator."""
    evaluator = RAGASEvaluator = RagasEvaluator()
    sample_preds = [
        {
            "question": "What is the punishment for murder under IPC?",
            "predicted_answer": "Under Section 302 IPC, murder is punishable with death or imprisonment for life and fine.",
            "retrieved_chunks": ["Section 302 Indian Penal Code: Punishment for murder with death or imprisonment for life."]
        }
    ]
    sample_gt = [
        {
            "question": "What is the punishment for murder under IPC?",
            "answer": "Section 302 provides punishment with death or life imprisonment."
        }
    ]
    scores = evaluator.evaluate_predictions(sample_preds, sample_gt)
    print("RAGAS Metric Scores:", scores)

if __name__ == "__main__":
    run_main()
