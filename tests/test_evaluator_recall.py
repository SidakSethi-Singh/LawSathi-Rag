import sys
import types
import unittest

fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = object
fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("pandas", fake_pandas)
sys.modules.setdefault("dotenv", fake_dotenv)

from src.evaluation.evaluator import evaluate_custom


class TestEvaluatorRecall(unittest.TestCase):
    def test_recall_uses_ground_truth_relevant_count(self):
        answer = "section 138 prosecution was quashed"
        gt_chunks = [
            "section 138 prosecution",
            "prosecution was quashed",
            "unrelated civil limitation issue",
        ]
        predictions = [{
            "question": "Was the prosecution valid?",
            "predicted_answer": answer,
            "retrieved_chunks": [gt_chunks[0], gt_chunks[1], "tax unrelated", "customs unrelated", "rent unrelated"],
        }]
        ground_truth = [{
            "question": "Was the prosecution valid?",
            "answer": answer,
            "context_chunks": gt_chunks,
        }]

        result = evaluate_custom(predictions, ground_truth)

        self.assertEqual(result["Precision@5"], 0.4)
        self.assertEqual(result["Recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
