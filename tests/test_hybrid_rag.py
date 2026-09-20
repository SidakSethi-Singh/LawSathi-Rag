import sys
import unittest
from unittest.mock import MagicMock

# Stub heavy external dependencies if not installed in the test environment,
# enabling tests to import the real functions and classes directly from hybrid_rag.py
for mod in ["chromadb", "chromadb.config", "sentence_transformers", "rank_bm25", "openai"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

# Import the ACTUAL implementations under test
from src.rag_pipelines.hybrid_rag import min_max_normalize, HybridRAG


class HybridRAGNormalizationTests(unittest.TestCase):
    """Unit tests exercising the actual min_max_normalize implementation."""

    def test_empty_scores_returns_empty(self):
        """Empty input dictionary must return an empty dictionary."""
        self.assertEqual(min_max_normalize({}), {})

    def test_all_zero_scores_normalize_to_zero(self):
        """All-zero scores (no lexical matches) must normalize to 0.0, not 1.0."""
        scores = {"chunk_a": 0.0, "chunk_b": 0.0, "chunk_c": 0.0}
        normalized = min_max_normalize(scores)
        self.assertEqual(normalized, {"chunk_a": 0.0, "chunk_b": 0.0, "chunk_c": 0.0})

    def test_equal_positive_scores_normalize_to_one(self):
        """Equal positive match scores must retain their match signal as 1.0."""
        scores = {"chunk_a": 3.5, "chunk_b": 3.5}
        normalized = min_max_normalize(scores)
        self.assertEqual(normalized, {"chunk_a": 1.0, "chunk_b": 1.0})

    def test_equal_negative_scores_normalize_to_zero(self):
        """Equal negative scores must normalize to 0.0 rather than being inflated."""
        scores = {"chunk_a": -1.2, "chunk_b": -1.2}
        normalized = min_max_normalize(scores)
        self.assertEqual(normalized, {"chunk_a": 0.0, "chunk_b": 0.0})

    def test_standard_varying_scores_normalize_linearly(self):
        """Standard varying scores normalize linearly into [0.0, 1.0]."""
        scores = {"chunk_min": 2.0, "chunk_mid": 5.0, "chunk_max": 8.0}
        normalized = min_max_normalize(scores)
        self.assertAlmostEqual(normalized["chunk_min"], 0.0)
        self.assertAlmostEqual(normalized["chunk_mid"], 0.5)
        self.assertAlmostEqual(normalized["chunk_max"], 1.0)


class HybridRAGIntegrationRankingTests(unittest.TestCase):
    """Integration test verifying end-to-end retrieve() ranking with zero-match BM25."""

    def test_zero_bm25_does_not_displace_dense_candidates(self):
        """When BM25 scores are all zero, zero-score BM25 candidates must not displace dense retrieval candidates."""
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = [f"doc_{i}" for i in range(10)]
        rag.alpha = 0.7

        # BM25 finds zero matches for all retrieved chunks
        rag._retrieve_bm25 = MagicMock(return_value={f"junk_bm25_{i}": 0.0 for i in range(5)})

        # Dense retrieval provides the candidates used for ranking
        rag._retrieve_dense = MagicMock(return_value={
            "dense_relevant_1": 0.90,
            "dense_relevant_2": 0.75,
            "dense_relevant_3": 0.60,
            "dense_relevant_4": 0.45,
            "dense_relevant_5": 0.30,
        })

        results = rag.retrieve("query with zero keyword overlap", k=4)

        # Dense candidates with positive normalized scores must all rank ahead of zero-score BM25 chunks
        expected_top_dense = [
            "dense_relevant_1",
            "dense_relevant_2",
            "dense_relevant_3",
            "dense_relevant_4",
        ]
        self.assertEqual(results[:4], expected_top_dense)


if __name__ == "__main__":
    unittest.main()
