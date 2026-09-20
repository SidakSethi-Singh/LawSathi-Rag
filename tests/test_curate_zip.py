import io
import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
fake_pandas = types.ModuleType("pandas")
fake_requests = types.ModuleType("requests")
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("dotenv", fake_dotenv)
sys.modules.setdefault("pandas", fake_pandas)
sys.modules.setdefault("requests", fake_requests)
sys.modules.setdefault("openai", fake_openai)
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.curate_benchmark import safe_extract_zip


class TestSafeExtractZip(unittest.TestCase):
    def test_rejects_members_outside_extract_dir(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            zip_file.writestr("../escaped.jsonl", "{}\n")
        buffer.seek(0)

        with TemporaryDirectory() as tmpdir, zipfile.ZipFile(buffer) as zip_file:
            with self.assertRaises(ValueError):
                safe_extract_zip(zip_file, Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
