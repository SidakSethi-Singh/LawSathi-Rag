import inspect
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

from src.rag_pipelines import dense_rag, hybrid_rag
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG
from src.run_full_benchmark import _embedding_model_slug
from src.utils import config


class EmbeddingConfigurationTests(unittest.TestCase):
    def test_dense_and_hybrid_use_configured_default_embedding_model(self):
        self.assertEqual(
            inspect.signature(DenseRAG).parameters["embed_model"].default,
            config.EMBEDDING_MODEL_NAME,
        )
        self.assertEqual(
            inspect.signature(HybridRAG).parameters["embed_model"].default,
            config.EMBEDDING_MODEL_NAME,
        )

    def test_legal_embedding_candidate_is_registered(self):
        self.assertTrue(config.LEGAL_EMBEDDING_MODEL_CANDIDATE)
        self.assertIn(
            config.LEGAL_EMBEDDING_MODEL_CANDIDATE,
            config.BENCHMARK_EMBEDDING_MODELS,
        )

    def test_model_slug_is_stable_and_filesystem_safe(self):
        model = "org/legal-embed-base:v1.5"
        self.assertEqual(
            _embedding_model_slug(model),
            "org_legal_embed_base_v1_5",
        )


if __name__ == "__main__":
    unittest.main()
