import sys
import types
import unittest

fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.chunker import chunk_text, chunk_documents


class TestChunkerInputs(unittest.TestCase):
    def test_non_string_input_returns_no_chunks(self):
        self.assertEqual(chunk_text(None), [])
        self.assertEqual(chunk_text(123), [])

    def test_zero_or_negative_chunk_size_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            chunk_text("legal text", chunk_size=0, overlap=0)
        self.assertIn("chunk_size must be a positive integer", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            chunk_text("legal text", chunk_size=-10, overlap=0)
        self.assertIn("chunk_size must be a positive integer", str(ctx.exception))

    def test_overlap_greater_than_or_equal_to_chunk_size_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            chunk_text("legal text", chunk_size=100, overlap=100)
        self.assertIn("overlap must be non-negative and strictly less than chunk_size", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            chunk_text("legal text", chunk_size=100, overlap=150)
        self.assertIn("overlap must be non-negative and strictly less than chunk_size", str(ctx.exception))

    def test_negative_overlap_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            chunk_text("legal text", chunk_size=100, overlap=-5)
        self.assertIn("overlap must be non-negative", str(ctx.exception))

    def test_chunk_documents_validates_parameters(self):
        with self.assertRaises(ValueError):
            chunk_documents(["doc1", "doc2"], chunk_size=50, overlap=50)

        with self.assertRaises(ValueError):
            chunk_documents(["doc1"], chunk_size=-1, overlap=0)

        self.assertEqual(chunk_documents([]), [])


if __name__ == "__main__":
    unittest.main()
