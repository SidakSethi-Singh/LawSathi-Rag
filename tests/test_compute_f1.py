import unittest
from src.evaluation.evaluator import compute_f1


class TestComputeF1(unittest.TestCase):

    def test_repeated_tokens_in_prediction(self):
        # "law" appears twice in pred but once in gt
        # set() would give precision=1.0, recall=1.0, F1=1.0 (wrong)
        # Counter gives precision=2/3, recall=2/2=1.0, F1=0.8
        pred = "law law court"
        gt = "law court"
        result = compute_f1(pred, gt)
        self.assertAlmostEqual(result, 0.8, places=9)

    def test_repeated_tokens_in_ground_truth(self):
        # "law" appears once in pred but twice in gt
        pred = "law court"
        gt = "law law court"
        result = compute_f1(pred, gt)
        # common = 1 law + 1 court = 2
        # precision = 2/2 = 1.0, recall = 2/3
        expected = 2 * 1.0 * (2/3) / (1.0 + 2/3)
        self.assertAlmostEqual(result, expected, places=9)

    def test_exact_match(self):
        self.assertAlmostEqual(compute_f1("hello world", "hello world"), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(compute_f1("hello world", "foo bar"), 0.0)

    def test_empty_prediction(self):
        self.assertAlmostEqual(compute_f1("", "hello world"), 0.0)


if __name__ == "__main__":
    unittest.main()
