import sys
import unittest
from unittest.mock import MagicMock, patch

# Stub heavy external dependencies if not installed in the test environment,
# enabling tests to import the real classes directly from rag_pipelines.
for mod in [
    "dotenv",
    "requests",
    "openai",
    "numpy",
    "rank_bm25",
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.dense_rag import DenseRAG
from src.rag_pipelines.hybrid_rag import HybridRAG


class RAGPipelineInheritanceTests(unittest.TestCase):
    """Verify that DenseRAG and HybridRAG properly inherit generate() from NaiveRAG."""

    def test_subclasses_inherit_from_naive_rag(self):
        """DenseRAG and HybridRAG must be subclasses of NaiveRAG."""
        self.assertTrue(issubclass(DenseRAG, NaiveRAG))
        self.assertTrue(issubclass(HybridRAG, NaiveRAG))

    def test_generate_is_not_duplicated_in_subclasses(self):
        """DenseRAG and HybridRAG must not define their own generate() method."""
        self.assertNotIn("generate", DenseRAG.__dict__)
        self.assertNotIn("generate", HybridRAG.__dict__)

    def test_generate_resolves_to_naive_rag_generate(self):
        """DenseRAG.generate and HybridRAG.generate must resolve to NaiveRAG.generate."""
        self.assertIs(DenseRAG.generate, NaiveRAG.generate)
        self.assertIs(HybridRAG.generate, NaiveRAG.generate)

    def test_inherited_generate_produces_identical_output(self):
        """All three pipelines produce identical generated answers for identical inputs."""
        naive = NaiveRAG.__new__(NaiveRAG)
        naive.model_name = "gpt-4o-mini"

        dense = DenseRAG.__new__(DenseRAG)
        dense.model_name = "gpt-4o-mini"

        hybrid = HybridRAG.__new__(HybridRAG)
        hybrid.model_name = "gpt-4o-mini"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Granted under Section 438 CrPC."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client), patch("src.utils.config.USE_LOCAL_MODEL", False):
            ans_naive = naive.generate("What was granted?", ["The court granted bail under Section 438."])
            ans_dense = dense.generate("What was granted?", ["The court granted bail under Section 438."])
            ans_hybrid = hybrid.generate("What was granted?", ["The court granted bail under Section 438."])

        self.assertEqual(ans_naive, "Granted under Section 438 CrPC.")
        self.assertEqual(ans_dense, "Granted under Section 438 CrPC.")
        self.assertEqual(ans_hybrid, "Granted under Section 438 CrPC.")

    def test_answer_method_calls_inherited_generate(self):
        """DenseRAG.answer and HybridRAG.answer utilize inherited generate() and return standard schema."""
        dense = DenseRAG.__new__(DenseRAG)
        dense.model_name = "gpt-4o-mini"
        dense.retrieve = MagicMock(return_value=["Context chunk 1"])

        hybrid = HybridRAG.__new__(HybridRAG)
        hybrid.model_name = "gpt-4o-mini"
        hybrid.retrieve = MagicMock(return_value=["Context chunk 2"])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Standard legal answer."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client), patch("src.utils.config.USE_LOCAL_MODEL", False):
            dense_res = dense.answer("What is the holding?", k=1)
            hybrid_res = hybrid.answer("What is the holding?", k=1)

        self.assertEqual(dense_res["predicted_answer"], "Standard legal answer.")
        self.assertEqual(dense_res["retrieved_chunks"], ["Context chunk 1"])
        self.assertIn("latency_ms", dense_res)
        self.assertEqual(dense_res["model_used"], "gpt-4o-mini")

        self.assertEqual(hybrid_res["predicted_answer"], "Standard legal answer.")
        self.assertEqual(hybrid_res["retrieved_chunks"], ["Context chunk 2"])
        self.assertIn("latency_ms", hybrid_res)
        self.assertEqual(hybrid_res["model_used"], "gpt-4o-mini")


    def test_inherited_generate_local_model_path(self):
        """DenseRAG and HybridRAG correctly call local Ollama model via inherited generate()."""
        dense = DenseRAG.__new__(DenseRAG)
        hybrid = HybridRAG.__new__(HybridRAG)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Local model answer."}
        mock_resp.raise_for_status = MagicMock()

        import requests
        with patch.object(requests, "post", return_value=mock_resp, create=True), patch("src.utils.config.USE_LOCAL_MODEL", True):
            ans_dense = dense.generate("Query?", ["Context."])
            ans_hybrid = hybrid.generate("Query?", ["Context."])

        self.assertEqual(ans_dense, "Local model answer.")
        self.assertEqual(ans_hybrid, "Local model answer.")


if __name__ == "__main__":
    unittest.main()
