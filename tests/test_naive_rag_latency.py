import unittest
from unittest.mock import patch

from src.rag_pipelines.naive_rag import NaiveRAG


class NaiveRAGLatencyTests(unittest.TestCase):
    def test_answer_measures_retrieval_and_generation_latency(self):
        rag = NaiveRAG.__new__(NaiveRAG)
        rag.model_name = "test-model"

        events = []

        def retrieve(query, k=5):
            events.append("retrieve")
            return ["context"]

        def generate(query, contexts):
            events.append("generate")
            return "answer"

        rag.retrieve = retrieve
        rag.generate = generate

        with patch(
            "src.rag_pipelines.naive_rag.time.perf_counter",
            side_effect=[10.0, 12.5],
        ) as timer:
            result = rag.answer("question", k=1)

        self.assertEqual(events, ["retrieve", "generate"])
        self.assertEqual(timer.call_count, 2)
        self.assertEqual(result["retrieved_chunks"], ["context"])
        self.assertEqual(result["predicted_answer"], "answer")
        self.assertEqual(result["latency_ms"], 2500.0)


if __name__ == "__main__":
    unittest.main()
