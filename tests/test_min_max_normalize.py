"""
Tests for the min_max_normalize fix.
The function is copied here to run without the heavy dependency chain
(chromadb, openai, sentence_transformers) that hybrid_rag.py imports.
"""
import unittest
from typing import Dict


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-Max normalization — copy of the fixed function in hybrid_rag.py."""
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    diff = max_val - min_val
    if diff == 0:
        uniform = 0.0 if max_val == 0.0 else 1.0
        return {k: uniform for k in scores}
    return {k: (v - min_val) / diff for k, v in scores.items()}


class TestMinMaxNormalize(unittest.TestCase):

    def test_all_zero_scores_return_zero(self):
        # BM25 returns all 0.0 when query has no lexical overlap with any chunk.
        # Before fix: returned {k: 1.0} — incorrectly promoted irrelevant chunks.
        scores = {"chunk_a": 0.0, "chunk_b": 0.0, "chunk_c": 0.0}
        result = min_max_normalize(scores)
        for v in result.values():
            self.assertEqual(v, 0.0)

    def test_all_equal_nonzero_return_one(self):
        # All chunks equally relevant — capped at 1.0 is correct.
        scores = {"chunk_a": 3.5, "chunk_b": 3.5}
        result = min_max_normalize(scores)
        for v in result.values():
            self.assertEqual(v, 1.0)

    def test_normal_range(self):
        scores = {"a": 0.0, "b": 5.0, "c": 10.0}
        result = min_max_normalize(scores)
        self.assertAlmostEqual(result["a"], 0.0)
        self.assertAlmostEqual(result["b"], 0.5)
        self.assertAlmostEqual(result["c"], 1.0)

    def test_empty_scores_return_empty(self):
        self.assertEqual(min_max_normalize({}), {})


if __name__ == "__main__":
    unittest.main()
