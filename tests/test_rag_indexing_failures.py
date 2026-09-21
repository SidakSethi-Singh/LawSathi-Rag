import sys
import types
import unittest
from unittest.mock import Mock


class FakeEmbeddings:
    def tolist(self):
        return [[0.1, 0.2], [0.3, 0.4]]


class FakeBM25:
    def __init__(self, tokenized_chunks):
        self.tokenized_chunks = tokenized_chunks


chromadb = types.ModuleType("chromadb")
chromadb.Client = Mock()
chromadb_config = types.ModuleType("chromadb.config")
chromadb_config.Settings = Mock
chromadb.config = chromadb_config
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = Mock
rank_bm25 = types.ModuleType("rank_bm25")
rank_bm25.BM25Okapi = FakeBM25
numpy = types.ModuleType("numpy")
numpy.argsort = Mock()
openai = types.ModuleType("openai")
openai.OpenAI = Mock

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = Mock()

sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("chromadb.config", chromadb_config)
sys.modules.setdefault("sentence_transformers", sentence_transformers)
sys.modules.setdefault("rank_bm25", rank_bm25)
sys.modules.setdefault("numpy", numpy)
sys.modules.setdefault("openai", openai)
sys.modules.setdefault("dotenv", dotenv)

from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG


class TestRAGIndexingFailures(unittest.TestCase):
    def test_dense_encoder_failure_does_not_publish_chunks(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.chunks = []
        rag.encoder = Mock()
        rag.collection = Mock()
        indexing_error = RuntimeError("embedding failure")
        rag.encoder.encode.side_effect = indexing_error

        with self.assertRaisesRegex(RuntimeError, "embedding failure"):
            rag.index_documents(["chunk one", "chunk two"])

        self.assertEqual(rag.chunks, [])
        rag.collection.add.assert_not_called()

    def test_dense_collection_failure_does_not_publish_chunks(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.chunks = []
        rag.encoder = Mock()
        rag.collection = Mock()
        rag.encoder.encode.return_value = FakeEmbeddings()
        indexing_error = RuntimeError("chroma insertion failure")
        rag.collection.add.side_effect = indexing_error

        with self.assertRaisesRegex(RuntimeError, "chroma insertion failure"):
            rag.index_documents(["chunk one", "chunk two"])

        self.assertEqual(rag.chunks, [])

    def test_dense_success_publishes_chunks_after_indexing(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.chunks = []
        rag.encoder = Mock()
        rag.collection = Mock()
        rag.encoder.encode.return_value = FakeEmbeddings()

        chunks = ["chunk one", "chunk two"]
        rag.index_documents(chunks)

        self.assertIs(rag.chunks, chunks)
        rag.collection.add.assert_called_once()

    def test_hybrid_encoder_failure_does_not_publish_index_state(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = []
        rag.bm25 = None
        rag.encoder = Mock()
        rag.collection = Mock()
        indexing_error = RuntimeError("embedding failure")
        rag.encoder.encode.side_effect = indexing_error

        with self.assertRaisesRegex(RuntimeError, "embedding failure"):
            rag.index_documents(["chunk one", "chunk two"])

        self.assertEqual(rag.chunks, [])
        self.assertIsNone(rag.bm25)
        rag.collection.add.assert_not_called()

    def test_hybrid_collection_failure_does_not_publish_index_state(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = []
        rag.bm25 = None
        rag.encoder = Mock()
        rag.collection = Mock()
        rag.encoder.encode.return_value = FakeEmbeddings()
        indexing_error = RuntimeError("chroma insertion failure")
        rag.collection.add.side_effect = indexing_error

        with self.assertRaisesRegex(RuntimeError, "chroma insertion failure"):
            rag.index_documents(["chunk one", "chunk two"])

        self.assertEqual(rag.chunks, [])
        self.assertIsNone(rag.bm25)

    def test_hybrid_success_publishes_index_state_after_indexing(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = []
        rag.bm25 = None
        rag.encoder = Mock()
        rag.collection = Mock()
        rag.encoder.encode.return_value = FakeEmbeddings()

        chunks = ["chunk one", "chunk two"]
        rag.index_documents(chunks)

        self.assertIs(rag.chunks, chunks)
        self.assertIsInstance(rag.bm25, FakeBM25)
        rag.collection.add.assert_called_once()


if __name__ == "__main__":
    unittest.main()
