import unittest

from src.run_full_benchmark import get_full_corpus


class FullBenchmarkCorpusTests(unittest.TestCase):
    def test_deduplicates_without_reordering(self):
        records = [
            {"context_chunks": ["chunk-b", "chunk-a", "chunk-b"]},
            {"context_chunks": ["chunk-c", "chunk-a", "chunk-d"]},
        ]

        self.assertEqual(
            get_full_corpus(records),
            ["chunk-b", "chunk-a", "chunk-c", "chunk-d"],
        )

    def test_missing_context_chunks_are_ignored(self):
        records = [
            {"context_chunks": ["first"]},
            {"question": "record without context_chunks"},
            {"context_chunks": ["second"]},
        ]

        self.assertEqual(get_full_corpus(records), ["first", "second"])

    def test_empty_input_returns_empty_corpus(self):
        self.assertEqual(get_full_corpus([]), [])


if __name__ == "__main__":
    unittest.main()
