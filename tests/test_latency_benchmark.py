import unittest
import time
from unittest.mock import MagicMock

from src.rag_pipelines.naive_rag import NaiveRAG

class TestLatencyBenchmark(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            "Section 304 of the Indian Penal Code deals with punishment for culpable homicide not amounting to murder.",
            "The Constitution of India is the supreme legal document governing the legislative, executive, and judicial powers.",
            "Under Section 138 of the Negotiable Instruments Act, dishonour of a cheque for insufficiency of funds is an offence."
        ]

    def test_answer_latency_structure(self):
        rag = NaiveRAG()
        rag.index_documents(self.sample_chunks)
        # Mock generate to avoid external API calls during unit test
        rag.generate = MagicMock(return_value="This is a mocked answer for Section 304 IPC.")

        result = rag.answer("What does Section 304 IPC specify?", k=2)

        self.assertIn("latency_ms", result)
        self.assertIn("retrieval_latency_ms", result)
        self.assertIn("generation_latency_ms", result)
        self.assertGreater(result["latency_ms"], 0)
        self.assertGreaterEqual(result["retrieval_latency_ms"], 0)
        self.assertGreaterEqual(result["generation_latency_ms"], 0)
        # Total latency should encompass both retrieval and generation
        self.assertGreaterEqual(
            result["latency_ms"],
            result["retrieval_latency_ms"] + result["generation_latency_ms"] - 1.0
        )

    def test_retrieval_and_generation_timed_accurately(self):
        rag = NaiveRAG()
        # Mock retrieve with 0.02s sleep and generate with 0.03s sleep
        def mock_retrieve(query, k=5):
            time.sleep(0.02)
            return ["chunk 1", "chunk 2"]

        def mock_generate(query, contexts):
            time.sleep(0.03)
            return "Generated answer"

        rag.retrieve = mock_retrieve
        rag.generate = mock_generate

        res = rag.answer("Test query", k=2)

        # 20ms retrieval sleep should be reflected in retrieval_latency_ms
        self.assertGreaterEqual(res["retrieval_latency_ms"], 15.0)
        # 30ms generation sleep should be reflected in generation_latency_ms
        self.assertGreaterEqual(res["generation_latency_ms"], 25.0)
        # Total latency should be at least ~50ms (0.05s)
        self.assertGreaterEqual(res["latency_ms"], 45.0)

    def test_subclass_inheritance_contract(self):
        """Verify subclass contract that any subclass inheriting from NaiveRAG uses this answer implementation."""
        class MockSubclassRAG(NaiveRAG):
            def retrieve(self, query, k=5):
                time.sleep(0.01)
                return ["mocked passage"]
            def generate(self, query, contexts):
                time.sleep(0.01)
                return "mocked answer"

        sub = MockSubclassRAG()
        res = sub.answer("Hello", k=1)
        self.assertIn("retrieval_latency_ms", res)
        self.assertIn("generation_latency_ms", res)
        self.assertGreaterEqual(res["retrieval_latency_ms"], 8.0)
        self.assertGreaterEqual(res["generation_latency_ms"], 8.0)
        self.assertGreaterEqual(res["latency_ms"], 18.0)

if __name__ == "__main__":
    unittest.main()
