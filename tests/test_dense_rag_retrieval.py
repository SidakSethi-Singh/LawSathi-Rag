import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_import_stubs() -> None:
    chromadb = types.ModuleType("chromadb")
    chromadb.Client = MagicMock()
    chromadb.config = types.ModuleType("chromadb.config")
    chromadb.config.Settings = MagicMock
    sys.modules.setdefault("chromadb", chromadb)
    sys.modules.setdefault("chromadb.config", chromadb.config)

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = MagicMock
    sys.modules.setdefault("sentence_transformers", sentence_transformers)

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = MagicMock
    sys.modules.setdefault("rank_bm25", rank_bm25)

    openai = types.ModuleType("openai")
    openai.OpenAI = MagicMock
    sys.modules.setdefault("openai", openai)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)


_install_import_stubs()

from src.rag_pipelines.dense_rag import DenseRAG


class TestDenseRAGRetrieval(unittest.TestCase):
    def make_rag(self, chunk_count: int) -> DenseRAG:
        rag = DenseRAG.__new__(DenseRAG)
        rag.chunks = [f"chunk-{index}" for index in range(chunk_count)]
        rag.encoder = MagicMock()
        rag.encoder.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2]])
        rag.collection = MagicMock()
        rag.collection.query.return_value = {
            "documents": [[f"chunk-{index}" for index in range(min(chunk_count, 3))]]
        }
        return rag

    def test_retrieval_clamps_k_to_indexed_corpus_size(self):
        rag = self.make_rag(2)

        results = rag.retrieve("query", k=5)

        self.assertEqual(results, ["chunk-0", "chunk-1"])
        rag.collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=2,
        )

    def test_retrieval_with_k_zero_returns_no_results(self):
        rag = self.make_rag(2)

        results = rag.retrieve("query", k=0)

        self.assertEqual(results, [])
        rag.collection.query.assert_not_called()
        rag.encoder.encode.assert_not_called()

    def test_retrieval_with_negative_k_returns_no_results(self):
        rag = self.make_rag(2)

        results = rag.retrieve("query", k=-1)

        self.assertEqual(results, [])
        rag.collection.query.assert_not_called()
        rag.encoder.encode.assert_not_called()

    def test_retrieval_preserves_requested_k_when_within_corpus_size(self):
        rag = self.make_rag(5)

        results = rag.retrieve("query", k=3)

        self.assertEqual(results, ["chunk-0", "chunk-1", "chunk-2"])
        rag.collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=3,
        )

    def test_empty_index_returns_no_results(self):
        rag = self.make_rag(0)

        results = rag.retrieve("query", k=5)

        self.assertEqual(results, [])
        rag.collection.query.assert_not_called()
        rag.encoder.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
