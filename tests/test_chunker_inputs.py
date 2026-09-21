import sys
import types
import unittest

fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.chunker import chunk_text


class TestChunkerInputs(unittest.TestCase):
    def test_non_string_input_returns_no_chunks(self):
        self.assertEqual(chunk_text(None), [])
        self.assertEqual(chunk_text(123), [])


if __name__ == "__main__":
    unittest.main()
