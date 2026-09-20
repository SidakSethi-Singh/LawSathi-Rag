import sys
import unittest
from unittest.mock import MagicMock

for mod in [
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
    "rank_bm25",
    "openai",
    "dotenv",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from src.rag_pipelines.hybrid_rag import calibrate_bm25_scores, HybridRAG


class BM25CalibrationTests(unittest.TestCase):
    def test_score_is_invariant_to_weaker_candidates(self):
        base = calibrate_bm25_scores({
            "exact_match": 8.0,
            "weak_match": 2.0,
        })
        expanded = calibrate_bm25_scores({
            "exact_match": 8.0,
            "weak_match": 2.0,
            "weaker_match": 0.25,
        })

        self.assertAlmostEqual(base["exact_match"], expanded["exact_match"])
        self.assertAlmostEqual(base["weak_match"], expanded["weak_match"])
        self.assertLess(expanded["weaker_match"], expanded["weak_match"])

    def test_non_positive_scores_do_not_gain_lexical_relevance(self):
        normalized = calibrate_bm25_scores({
            "zero": 0.0,
            "negative": -1.0,
            "positive": 3.0,
        })

        self.assertEqual(normalized["zero"], 0.0)
        self.assertEqual(normalized["negative"], 0.0)
        self.assertAlmostEqual(normalized["positive"], 0.75)

    def test_positive_scores_remain_monotonic(self):
        normalized = calibrate_bm25_scores({
            "low": 1.0,
            "medium": 4.0,
            "high": 9.0,
        })

        self.assertGreater(normalized["medium"], normalized["low"])
        self.assertGreater(normalized["high"], normalized["medium"])


class HybridRAGLexicalStabilityTests(unittest.TestCase):
    def test_weaker_bm25_distractor_does_not_change_existing_order(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["lexical_exact", "dense_exact", "lexical_distractor"]
        rag.alpha = 0.5

        rag._retrieve_dense = MagicMock(return_value={
            "dense_exact": 0.95,
            "lexical_exact": 0.90,
        })

        rag._retrieve_bm25 = MagicMock(return_value={
            "lexical_exact": 8.0,
            "dense_exact": 1.0,
        })
        without_distractor = rag.retrieve("query", k=2)

        rag._retrieve_bm25.return_value = {
            "lexical_exact": 8.0,
            "dense_exact": 1.0,
            "lexical_distractor": 0.2,
        }
        with_distractor = rag.retrieve("query", k=2)

        self.assertEqual(without_distractor, with_distractor)


if __name__ == "__main__":
    unittest.main()
