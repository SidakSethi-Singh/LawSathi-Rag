import sys
import types
import unittest
from unittest.mock import MagicMock

for module_name in [
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
    "rank_bm25",
    "openai",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from src.rag_pipelines.hybrid_rag import HybridRAG


class TestHybridRAGChunkIdentity(unittest.TestCase):
    def test_duplicate_text_chunks_remain_distinct_during_fusion(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["same passage", "same passage", "different passage"]
        rag.alpha = 0.5
        rag._retrieve_bm25 = MagicMock(return_value={0: 2.0, 1: 1.0})
        rag._retrieve_dense = MagicMock(return_value={0: 0.9, 1: 0.8})

        self.assertEqual(
            rag.retrieve("query", k=2),
            ["same passage", "same passage"],
        )

    def test_dense_retrieval_uses_chroma_ids_for_duplicate_text(self):
        class FakeEmbedding(list):
            def tolist(self):
                return list(self)

        class FakeEncoder:
            def encode(self, queries):
                return FakeEmbedding([[0.1, 0.2]])

        class FakeCollection:
            def query(self, query_embeddings, n_results):
                return {
                    "documents": [["same passage", "same passage"]],
                    "ids": [["chunk_1", "chunk_0"]],
                    "distances": [[0.1, 0.2]],
                }

        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["same passage", "same passage"]
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()

        self.assertEqual(
            rag._retrieve_dense("query", limit=2),
            {1: 0.9, 0: 0.8},
        )


if __name__ == "__main__":
    unittest.main()
