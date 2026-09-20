import unittest

from src.evaluation.error_analysis import (
    build_category_summary,
    chunk_supports_answer,
    classify_failure,
    compute_f1,
)


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


if __name__ == "__main__":
    unittest.main()
