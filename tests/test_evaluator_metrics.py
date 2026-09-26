import unittest

from src.evaluation.evaluator import (
    check_chunk_relevance,
    compute_em,
    compute_f1,
    evaluate_custom
)

class TestEvaluatorMetrics(unittest.TestCase):
    def test_check_chunk_relevance(self):
        gt_answer = "The Supreme Court dismissed the appeal regarding excise duty classification."
        relevant_chunk = "In this judgment, the Supreme Court discussed excise duty classification extensively."
        irrelevant_chunk = "A completely unrelated traffic violation dispute in municipal court."

        self.assertTrue(check_chunk_relevance(relevant_chunk, gt_answer))
        self.assertFalse(check_chunk_relevance(irrelevant_chunk, gt_answer))

    def test_compute_em(self):
        self.assertEqual(compute_em("Dismissed", "dismissed "), 1.0)
        self.assertEqual(compute_em("Granted", "Denied"), 0.0)

    def test_compute_f1(self):
        # Full overlap
        self.assertEqual(compute_f1("excise duty classification", "excise duty classification"), 1.0)
        # No overlap
        self.assertEqual(compute_f1("apple orange", "car truck"), 0.0)
        # Partial overlap: 2 common ("excise", "duty")
        f1 = compute_f1("central excise duty tariff", "excise duty classification")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_precision_and_recall_are_distinct(self):
        """
        Verify that Precision@5 and Recall@5 are mathematically distinct
        and that Recall@5 is properly normalized by total ground-truth relevant passages.
        """
        gt_ans = "The appellant's prosecution under Section 138 of the Negotiable Instruments Act was quashed."
        gt_chunks = [
            "Section 138 Negotiable Instruments Act prosecution validity.",
            "The High Court held that the prosecution under Section 138 was unsustainable."
        ]
        
        # Ground truth has 2 relevant chunks
        ground_truth = [{
            "question": "Was prosecution under Section 138 valid?",
            "answer": gt_ans,
            "context_chunks": gt_chunks
        }]

        # Retrieved top 5 chunks: 2 relevant, 3 irrelevant
        retrieved = [
            gt_chunks[0],
            gt_chunks[1],
            "Unrelated property boundary dispute.",
            "Arbitration clause enforcement ruling.",
            "Motor vehicles act third party claim."
        ]

        predictions = [{
            "question": "Was prosecution under Section 138 valid?",
            "predicted_answer": "Prosecution under Section 138 was quashed.",
            "retrieved_chunks": retrieved,
            "latency_ms": 120.0
        }]

        res = evaluate_custom(predictions, ground_truth)

        # Precision@5 should be 2 relevant / 5 retrieved = 0.40
        self.assertEqual(res["Precision@5"], 0.4)
        # Recall@5 should be 2 retrieved / 2 ground truth relevant = 1.00
        self.assertEqual(res["Recall@5"], 1.0)
        # They MUST NOT be identical
        self.assertNotEqual(res["Precision@5"], res["Recall@5"])

    def test_recall_partial_coverage(self):
        """Verify partial recall when top-5 retrieves 2 out of 4 relevant chunks."""
        gt_ans = "Commercial parlance test governs classification under excise tariff."
        gt_chunks = [
            "Commercial parlance test definition.",
            "Common market perception in excise tariff.",
            "Central excise tariff heading 2107.",
            "Trade classification under excise law."
        ]

        ground_truth = [{
            "question": "What test governs classification?",
            "answer": gt_ans,
            "context_chunks": gt_chunks
        }]

        # Retrieve 2 relevant, 3 irrelevant
        retrieved = [
            gt_chunks[0],
            gt_chunks[1],
            "Unrelated tax dispute.",
            "Customs import duty exception.",
            "Income tax section 80C exemption."
        ]

        predictions = [{
            "question": "What test governs classification?",
            "predicted_answer": gt_ans,
            "retrieved_chunks": retrieved,
            "latency_ms": 150.0
        }]

        res = evaluate_custom(predictions, ground_truth)
        # 2 / 5 = 0.4
        self.assertEqual(res["Precision@5"], 0.4)
        # 2 / 4 = 0.5
        self.assertEqual(res["Recall@5"], 0.5)

if __name__ == "__main__":
    unittest.main()
