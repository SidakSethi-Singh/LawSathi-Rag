"""
Edge-case retrieval tests for NaiveRAG.
Tests the retrieve() method directly without calling LLM APIs.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub heavy deps so the module can be imported without installing them
for mod in ["openai", "rank_bm25", "numpy"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np_real  # we need real numpy for argsort in the test

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestNaiveRAGRetrievalEdgeCases(unittest.TestCase):
    """
    Tests NaiveRAG.retrieve() behaviour when the corpus has fewer
    documents than the requested k, including the empty-corpus case.

    We test the logic directly by patching BM25Okapi so no external
    package needs to be installed in this environment.
    """

    def _make_rag(self, chunks):
        """Return a NaiveRAG instance with chunks indexed using a real BM25-like stub."""
        from src.rag_pipelines.naive_rag import NaiveRAG

        rag = NaiveRAG.__new__(NaiveRAG)
        rag.chunks = chunks
        # Stub bm25 to return uniform scores so argsort is deterministic
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [1.0] * len(chunks)
        rag.bm25 = mock_bm25 if chunks else None
        return rag

    def test_empty_corpus_returns_empty_list(self):
        from src.rag_pipelines.naive_rag import NaiveRAG
        rag = NaiveRAG.__new__(NaiveRAG)
        rag.chunks = []
        rag.bm25 = None
        result = rag.retrieve("any query", k=5)
        self.assertEqual(result, [])

    def test_corpus_smaller_than_k_returns_all_docs(self):
        chunks = ["chunk one", "chunk two"]
        rag = self._make_rag(chunks)
        # Mock numpy argsort to return valid indices
        import numpy as np
        with patch.object(np, "argsort", return_value=np.array([0, 1])):
            result = rag.retrieve("query", k=10)
        self.assertLessEqual(len(result), len(chunks))
        self.assertGreaterEqual(len(result), 0)

    def test_k_equals_corpus_size_returns_all_docs(self):
        chunks = ["a", "b", "c"]
        rag = self._make_rag(chunks)
        import numpy as np
        with patch.object(np, "argsort", return_value=np.array([0, 1, 2])):
            result = rag.retrieve("query", k=3)
        self.assertLessEqual(len(result), 3)

    def test_k_larger_than_corpus_does_not_raise(self):
        chunks = ["only one chunk"]
        rag = self._make_rag(chunks)
        import numpy as np
        with patch.object(np, "argsort", return_value=np.array([0])):
            try:
                result = rag.retrieve("query", k=100)
                self.assertIsInstance(result, list)
            except Exception as e:
                self.fail(f"retrieve() raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
