import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

rank_bm25_stub = types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

naive_rag = importlib.import_module("src.rag_pipelines.naive_rag")


class TestModelNameProvenance(unittest.TestCase):
    def test_local_generation_uses_selected_model(self):
        rag = naive_rag.NaiveRAG(model_name="custom-local-model")
        response = MagicMock()
        response.json.return_value = {"response": "local answer"}

        with patch.object(naive_rag.config, "USE_LOCAL_MODEL", True), patch.object(
            naive_rag.requests, "post", return_value=response
        ) as post:
            result = rag.generate("question", ["context"])

        self.assertEqual(result, "local answer")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["model"], "custom-local-model")

    def test_remote_generation_uses_selected_model(self):
        rag = naive_rag.NaiveRAG(model_name="custom-remote-model")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="remote answer"))]
        client = MagicMock()
        client.chat.completions.create.return_value = response
        openai_provider_stub = types.ModuleType("openai")
        openai_provider_stub.OpenAI = MagicMock(return_value=client)

        with patch.object(naive_rag.config, "USE_LOCAL_MODEL", False), patch.dict(
            sys.modules, {"openai": openai_provider_stub}
        ):
            result = rag.generate("question", ["context"])

        self.assertEqual(result, "remote answer")
        client.chat.completions.create.assert_called_once()
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "custom-remote-model",
        )

    def test_answer_reports_selected_model(self):
        rag = naive_rag.NaiveRAG(model_name="custom-model")

        with patch.object(rag, "retrieve", return_value=["context"]), patch.object(
            rag, "generate", return_value="answer"
        ):
            result = rag.answer("question")

        self.assertEqual(result["model_used"], "custom-model")


if __name__ == "__main__":
    unittest.main()
