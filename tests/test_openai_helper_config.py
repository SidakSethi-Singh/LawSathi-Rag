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


class FakeMessage:
    content = "configured"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeClient:
    last_init_kwargs = None
    last_completions = None

    def __init__(self, **kwargs):
        self.__class__.last_init_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=FakeCompletions())
        self.__class__.last_completions = self.chat.completions


class TestOpenAIHelperConfig(unittest.TestCase):
    def test_uses_configured_base_url_and_model(self):
        with (
            mock.patch.object(helpers, "OpenAI", FakeClient),
            mock.patch.object(helpers, "API_BASE_URL", "https://example.test/v1"),
            mock.patch.object(helpers, "OPENAI_API_KEY", "test-key"),
            mock.patch.object(helpers, "MODEL_NAME", "model-from-config"),
        ):
            result = helpers.call_openai_api("prompt", "system")

        self.assertEqual(result, "configured")
        self.assertEqual(FakeClient.last_init_kwargs["base_url"], "https://example.test/v1")
        self.assertEqual(FakeClient.last_init_kwargs["api_key"], "test-key")
        self.assertEqual(FakeClient.last_completions.kwargs["model"], "model-from-config")


if __name__ == "__main__":
    unittest.main()
