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

from src.rag_pipelines.hybrid_rag import HybridRAG


class HybridRAGCandidateLimitTests(unittest.TestCase):
    def _make_rag(self, candidate_k=20, corpus_size=25):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = [f"doc_{i}" for i in range(corpus_size)]
        rag.alpha = 0.7
        rag.candidate_k = candidate_k
        rag._retrieve_bm25 = MagicMock(
            return_value={f"bm25_{i}": float(i + 1) for i in range(candidate_k)}
        )
        rag._retrieve_dense = MagicMock(
            return_value={f"dense_{i}": float(i + 1) for i in range(candidate_k)}
        )
        return rag

    def test_k_20_passes_at_least_20_candidates_to_each_retriever(self):
        rag = self._make_rag(candidate_k=20, corpus_size=25)

        results = rag.retrieve("query", k=20)

        rag._retrieve_bm25.assert_called_once_with("query", 20)
        rag._retrieve_dense.assert_called_once_with("query", 20)
        self.assertEqual(len(results), 20)

    def test_candidate_k_is_configurable_and_must_cover_requested_k(self):
        rag = self._make_rag(candidate_k=8, corpus_size=25)

        rag.retrieve("query", k=5)
        rag._retrieve_bm25.assert_called_with("query", 8)
        rag._retrieve_dense.assert_called_with("query", 8)

        rag._retrieve_bm25.reset_mock()
        rag._retrieve_dense.reset_mock()

        rag.retrieve("query", k=12)
        rag._retrieve_bm25.assert_called_once_with("query", 12)
        rag._retrieve_dense.assert_called_once_with("query", 12)

    def test_candidate_limit_never_exceeds_available_corpus(self):
        rag = self._make_rag(candidate_k=20, corpus_size=7)
        rag._retrieve_bm25.return_value = {
            f"bm25_{i}": float(i + 1) for i in range(7)
        }
        rag._retrieve_dense.return_value = {
            f"dense_{i}": float(i + 1) for i in range(7)
        }

        results = rag.retrieve("query", k=20)

        rag._retrieve_bm25.assert_called_once_with("query", 7)
        rag._retrieve_dense.assert_called_once_with("query", 7)
        self.assertLessEqual(len(results), 7)

    def test_non_positive_k_returns_no_results(self):
        rag = self._make_rag(candidate_k=20)

        self.assertEqual(rag.retrieve("query", k=0), [])
        self.assertEqual(rag.retrieve("query", k=-1), [])
        rag._retrieve_bm25.assert_not_called()
        rag._retrieve_dense.assert_not_called()


if __name__ == "__main__":
    unittest.main()
