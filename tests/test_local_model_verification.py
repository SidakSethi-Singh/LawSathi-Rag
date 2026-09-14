import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_pipelines import naive_rag


class TestLocalModelVerification(unittest.TestCase):
    @patch.object(naive_rag, "NaiveRAG")
    @patch.object(naive_rag, "save_jsonl")
    @patch.object(naive_rag, "requests")
    @patch.object(naive_rag.config, "USE_LOCAL_MODEL", True)
    @patch.object(naive_rag.config, "API_BASE_URL", "http://localhost:11434")
    @patch("builtins.open")
    def test_local_health_check_uses_configured_base_url(
        self, mock_open, mock_requests, mock_save_jsonl, mock_rag
    ):
        records = [
            json.dumps({"question": "q1", "context_chunks": []}),
            json.dumps({"question": "q2", "context_chunks": []}),
        ]
        mock_open.return_value.__enter__.return_value = iter(
            [record + "\n" for record in records]
        )
        mock_requests.get.return_value = MagicMock()
        mock_rag.return_value.answer.return_value = {
            "question": "q",
            "predicted_answer": "a",
            "retrieved_chunks": [],
            "latency_ms": 1.0,
            "model_used": "model",
        }

        naive_rag.run_main()

        mock_requests.get.assert_called_once_with("http://localhost:11434", timeout=5.0)
        mock_rag.assert_called_once()
        mock_save_jsonl.assert_called_once()

    @patch.object(naive_rag, "requests")
    @patch.object(naive_rag.config, "USE_LOCAL_MODEL", True)
    @patch.object(naive_rag.config, "API_BASE_URL", "http://localhost:11434")
    @patch.object(naive_rag, "project_root", Path("/tmp/nonexistent-lawsathi-rag-test"))
    def test_offline_local_model_exits_before_loading_records(self, mock_requests):
        mock_requests.get.side_effect = OSError("connection refused")

        with self.assertRaises(SystemExit) as error:
            naive_rag.run_main()

        self.assertEqual(error.exception.code, 1)
        mock_requests.get.assert_called_once_with("http://localhost:11434", timeout=5.0)


if __name__ == "__main__":
    unittest.main()
