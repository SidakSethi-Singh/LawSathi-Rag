import sys
import types
import unittest
from unittest import mock

fake_requests = types.ModuleType("requests")
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("requests", fake_requests)
sys.modules.setdefault("openai", fake_openai)
sys.modules.setdefault("dotenv", fake_dotenv)

from src.utils import helpers


class TestRetryLogging(unittest.TestCase):
    def test_final_failure_does_not_claim_retry(self):
        def always_fail():
            raise ValueError("boom")

        with (
            mock.patch.object(helpers.time, "sleep") as sleep,
            self.assertLogs(helpers.logger, level="WARNING") as logs,
            self.assertRaises(RuntimeError),
        ):
            helpers.retry_with_backoff(always_fail, max_retries=1, initial_delay=0)

        self.assertFalse(sleep.called)
        self.assertIn("No retries left", logs.output[0])
        self.assertNotIn("Retrying", logs.output[0])

    def test_call_openai_api_uses_config_base_url_and_model(self):
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="Mocked answer"))]
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("src.utils.helpers.OpenAI", return_value=mock_client) as mock_openai_cls:
            result = helpers.call_openai_api(prompt="Test prompt", system_prompt="System prompt")
            self.assertEqual(result, "Mocked answer")

            mock_openai_cls.assert_called_once_with(
                api_key=helpers.OPENAI_API_KEY,
                base_url=helpers.API_BASE_URL
            )
            mock_client.chat.completions.create.assert_called_once_with(
                model=helpers.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Test prompt"}
                ],
                temperature=0.2,
                timeout=30.0
            )


if __name__ == "__main__":
    unittest.main()
