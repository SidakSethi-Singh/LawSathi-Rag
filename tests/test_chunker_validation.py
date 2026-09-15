import sys
import types
import unittest

fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.chunker import chunk_text


class TestChunkerValidation(unittest.TestCase):
    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("legal text", chunk_size=50, overlap=50)

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            chunk_text("legal text", chunk_size=0, overlap=0)


if __name__ == "__main__":
    unittest.main()
