import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
fake_requests = types.ModuleType("requests")
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("dotenv", fake_dotenv)
sys.modules.setdefault("requests", fake_requests)
sys.modules.setdefault("openai", fake_openai)
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.curate_benchmark import find_data_file


class TestCurateDiscovery(unittest.TestCase):
    def test_finds_uppercase_jsonl_suffix(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "DATA.JSONL"
            path.write_text("{}\n", encoding="utf-8")

            self.assertEqual(find_data_file(Path(tmpdir)), path)


if __name__ == "__main__":
    unittest.main()
