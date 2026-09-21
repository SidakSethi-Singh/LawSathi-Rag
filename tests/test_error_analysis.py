import unittest

from src.evaluation.error_analysis import (
    build_category_summary,
    chunk_supports_answer,
    classify_failure,
    compute_f1,
    find_failure_cases,
    _load_prediction_file,
)
from pathlib import Path
from tempfile import TemporaryDirectory


class ErrorAnalysisTests(unittest.TestCase):
    def test_compute_f1_preserves_duplicate_tokens(self):
        self.assertAlmostEqual(compute_f1("law law court", "law court"), 0.8)

    def test_chunk_supports_short_answer(self):
        self.assertTrue(chunk_supports_answer("The court granted bail.", "bail"))

    def test_missing_ground_truth_is_unanswerable(self):
        case = {"ground_truth": "", "retrieved_chunks": [], "f1_score": 0.0}
        self.assertEqual(classify_failure(case), "unanswerable")

    def test_missing_retrieval_is_retrieval_failure(self):
        case = {"ground_truth": "anticipatory bail", "retrieved_chunks": [], "f1_score": 0.0}
        self.assertEqual(classify_failure(case), "retrieval_failure")

    def test_irrelevant_retrieval_is_retrieval_failure(self):
        case = {
            "ground_truth": "anticipatory bail",
            "retrieved_chunks": ["tax assessment procedure"],
            "f1_score": 0.0,
        }
        self.assertEqual(classify_failure(case), "retrieval_failure")

    def test_supported_retrieval_with_bad_answer_is_generation_failure(self):
        case = {
            "ground_truth": "anticipatory bail",
            "retrieved_chunks": ["The court considered an anticipatory bail application."],
            "f1_score": 0.2,
        }
        self.assertEqual(classify_failure(case), "generation_failure")

    def test_uncertain_case_is_needs_review(self):
        case = {
            "ground_truth": "anticipatory bail",
            "retrieved_chunks": ["The court considered anticipatory bail."],
            "f1_score": 0.7,
        }
        self.assertEqual(classify_failure(case), "needs_review")

    def test_summary_is_data_driven(self):
        cases = [
            {"failure_category": "retrieval_failure"},
            {"failure_category": "retrieval_failure"},
            {"failure_category": "generation_failure"},
            {"failure_category": "needs_review"},
        ]
        summary = build_category_summary(cases)
        self.assertEqual(summary["retrieval_failure"]["count"], 2)
        self.assertEqual(summary["retrieval_failure"]["percentage"], 50.0)
        self.assertEqual(summary["generation_failure"]["percentage"], 25.0)
        self.assertEqual(summary["unanswerable"]["percentage"], 0.0)

    def test_empty_summary_has_zero_percentages(self):
        summary = build_category_summary([])
        self.assertTrue(all(item["percentage"] == 0.0 for item in summary.values()))

    def test_failure_cases_retained_across_multiple_architectures(self):
        """Failure on the same question must be captured for each failing architecture."""
        gt = [{"id": "q_001", "question": "What is Section 302 IPC?", "answer": "punishment for murder"}]
        naive_preds = [
            {"question": "What is Section 302 IPC?", "predicted_answer": "theft rules", "retrieved_chunks": []}
        ]
        dense_preds = [
            {"question": "What is Section 302 IPC?", "predicted_answer": "taxation", "retrieved_chunks": []}
        ]
        preds_list = [("NaiveRAG", naive_preds), ("DenseRAG", dense_preds)]
        cases = find_failure_cases(preds_list, gt, threshold=0.5)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["architecture"], "NaiveRAG")
        self.assertEqual(cases[1]["architecture"], "DenseRAG")

    def test_failure_cases_deduplicates_within_same_architecture(self):
        """Duplicate question entries within the same architecture should not create duplicate errors."""
        gt = [{"id": "q_001", "question": "What is Section 302 IPC?", "answer": "punishment for murder"}]
        naive_preds = [
            {"question": "What is Section 302 IPC?", "predicted_answer": "theft rules", "retrieved_chunks": []},
            {"question": "What is Section 302 IPC?", "predicted_answer": "theft rules duplicate", "retrieved_chunks": []}
        ]
        preds_list = [("NaiveRAG", naive_preds)]
        cases = find_failure_cases(preds_list, gt, threshold=0.5)

        self.assertEqual(len(cases), 1)

    def test_load_prediction_file_prefers_full_predictions(self):
        """_load_prediction_file should load _full.jsonl if present."""
        with TemporaryDirectory() as tmpdir:
            preds_dir = Path(tmpdir)
            full_file = preds_dir / "naive_rag_full.jsonl"
            sample_file = preds_dir / "naive_rag.jsonl"
            full_file.write_text('{"question": "q_full", "predicted_answer": "a_full"}\n', encoding="utf-8")
            sample_file.write_text('{"question": "q_sample", "predicted_answer": "a_sample"}\n', encoding="utf-8")

            res = _load_prediction_file(preds_dir, "naive_rag")
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["question"], "q_full")


if __name__ == "__main__":
    unittest.main()
