import sys
import unittest
from unittest.mock import MagicMock

for module_name in ["chromadb", "chromadb.config", "sentence_transformers", "rank_bm25", "openai"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG


class FakeEmbeddings:
    def tolist(self):
        return [[0.1, 0.2]]


class FakeEncoder:
    def encode(self, texts, show_progress_bar=False):
        return FakeEmbeddings()


class FakeCollection:
    def __init__(self):
        self.added_metadatas = None
        self.last_query = None

    def add(self, ids, documents, embeddings, metadatas):
        self.added_metadatas = metadatas

    def query(self, **kwargs):
        self.last_query = kwargs
        return {"documents": [["matching chunk"]], "distances": [[0.1]]}


class FakeBM25:
    def get_scores(self, query_tokens):
        return [2.0, 1.0]


class TestDenseMetadataFiltering(unittest.TestCase):
    def test_index_stores_metadata_and_retrieve_passes_filter(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()

        rag.index_documents(
            ["chunk one", "chunk two"],
            metadatas=[
                {"court": "Supreme Court of India", "year": 2024},
                {"court": "Delhi High Court", "year": 2023},
            ],
        )

        self.assertEqual(
            rag.collection.added_metadatas,
            [
                {"chunk_id": "chunk_0", "court": "Supreme Court of India", "year": 2024},
                {"chunk_id": "chunk_1", "court": "Delhi High Court", "year": 2023},
            ],
        )

        self.assertEqual(
            rag.retrieve(
                "query",
                metadata_filter={"court": "Supreme Court of India", "year": 2024},
            ),
            ["matching chunk"],
        )
        self.assertEqual(
            rag.collection.last_query["where"],
            {"court": "Supreme Court of India", "year": 2024},
        )

    def test_rejects_metadata_length_mismatch(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()

        with self.assertRaises(ValueError):
            rag.index_documents(["chunk one"], metadatas=[])


class TestHybridMetadataFiltering(unittest.TestCase):
    def test_bm25_filter_uses_metadata_before_ranking(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["chunk one", "chunk two"]
        rag.metadatas = [
            {"chunk_id": "chunk_0", "court": "A"},
            {"chunk_id": "chunk_1", "court": "B"},
        ]
        rag.bm25 = FakeBM25()

        self.assertEqual(
            rag._retrieve_bm25("query", 5, {"court": "B"}),
            {"chunk two": 1.0},
        )

    def test_dense_filter_reaches_chroma(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()
        rag.chunks = ["matching chunk"]
        rag.metadatas = [{"chunk_id": "chunk_0", "court": "A"}]

        self.assertEqual(
            rag._retrieve_dense("query", 1, {"court": "A"}),
            {"matching chunk": 0.9},
        )
        self.assertEqual(rag.collection.last_query["where"], {"court": "A"})

    def test_serializes_non_scalar_metadata_for_chroma(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()

        rag.index_documents(
            ["chunk one"],
            metadatas=[{"act": ["Indian Penal Code", "Evidence Act"]}],
        )

        self.assertEqual(
            rag.collection.added_metadatas[0]["act"],
            '["Indian Penal Code", "Evidence Act"]',
        )


if __name__ == "__main__":
    unittest.main()
