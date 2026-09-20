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


if __name__ == "__main__":
    unittest.main()
