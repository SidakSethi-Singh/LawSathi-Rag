import sys
import types
import unittest

fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = object
fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("pandas", fake_pandas)
sys.modules.setdefault("dotenv", fake_dotenv)

from src.evaluation.evaluator import compute_f1


class TestEvaluatorF1(unittest.TestCase):
    def test_duplicate_tokens_count_only_matched_occurrences(self):
        score = compute_f1("tax tax duty", "tax duty duty")

        self.assertAlmostEqual(score, 2 / 3)


if __name__ == "__main__":
    unittest.main()
