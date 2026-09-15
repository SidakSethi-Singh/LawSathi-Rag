import unittest
from unittest.mock import Mock, patch
from src.rag_pipelines.naive_rag import NaiveRAG
class TestNaiveRAGClientReuse(unittest.TestCase):
    @patch("src.rag_pipelines.naive_rag.OpenAI")
    @patch("src.utils.config.USE_LOCAL_MODEL", False)
    def test_openai_client_created_once(self, mock_openai):
        fake_client = Mock()
        fake_response = Mock(
            choices=[
                Mock(
                    message=Mock(content="test answer")
                )
            ]
        )
        fake_client.chat.completions.create.return_value = fake_response
        mock_openai.return_value = fake_client
        rag = NaiveRAG()
        rag.generate("question 1", ["context"])
        rag.generate("question 2", ["context"])
        mock_openai.assert_called_once()
        self.assertEqual(
            fake_client.chat.completions.create.call_count,
            2
        )
    @patch("src.rag_pipelines.naive_rag.requests.Session")
    @patch("src.utils.config.USE_LOCAL_MODEL", True)
    def test_requests_session_created_once(self, mock_session):
        fake_session = Mock()
        fake_response = Mock()
        fake_response.json.return_value = {"response": "test answer"}
        fake_session.post.return_value = fake_response
        mock_session.return_value = fake_session
        rag = NaiveRAG()
        rag.generate("question 1", ["context"])
        rag.generate("question 2", ["context"])
        mock_session.assert_called_once()
        self.assertEqual(fake_session.post.call_count, 2)
if __name__ == "__main__":
    unittest.main()
