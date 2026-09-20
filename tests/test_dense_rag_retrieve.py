import sys
import types
import unittest

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
fake_requests = types.ModuleType("requests")
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_numpy = types.ModuleType("numpy")
fake_rank_bm25 = types.ModuleType("rank_bm25")
fake_rank_bm25.BM25Okapi = object
fake_chromadb = types.ModuleType("chromadb")
fake_chromadb.Client = object
fake_chromadb_config = types.ModuleType("chromadb.config")
fake_chromadb_config.Settings = object
fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("dotenv", fake_dotenv)
sys.modules.setdefault("requests", fake_requests)
sys.modules.setdefault("openai", fake_openai)
sys.modules.setdefault("numpy", fake_numpy)
sys.modules.setdefault("rank_bm25", fake_rank_bm25)
sys.modules.setdefault("chromadb", fake_chromadb)
sys.modules.setdefault("chromadb.config", fake_chromadb_config)
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

from src.rag_pipelines.dense_rag import DenseRAG


class FakeEmbedding(list):
    def tolist(self):
        return list(self)


class FakeEncoder:
    def encode(self, queries):
        return FakeEmbedding([[0.1, 0.2]])


class FakeCollection:
    def __init__(self):
        self.n_results = None

    def query(self, query_embeddings, n_results):
        self.n_results = n_results
        return {"documents": [["a", "b"]]}


class TestDenseRAGRetrieve(unittest.TestCase):
    def test_retrieve_clamps_k_to_index_size(self):
        rag = DenseRAG.__new__(DenseRAG)
        rag.chunks = ["a", "b"]
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()

        self.assertEqual(rag.retrieve("question", k=5), ["a", "b"])
        self.assertEqual(rag.collection.n_results, 2)


if __name__ == "__main__":
    unittest.main()
