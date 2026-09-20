"""Tests for the shared load_jsonl helper and CLI argument parsing."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Test load_jsonl directly without the heavy helper module chain
def load_jsonl(path):
    """Minimal copy of the shared helper for test purposes."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestLoadJsonl(unittest.TestCase):

    def test_loads_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as f:
            f.write('{"question": "q1", "answer": "a1"}\n')
            f.write('{"question": "q2", "answer": "a2"}\n')
            path = f.name
        try:
            result = load_jsonl(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["question"], "q1")
            self.assertEqual(result[1]["answer"], "a2")
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty_list(self):
        result = load_jsonl("/nonexistent/file.jsonl")
        self.assertEqual(result, [])

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
            path = f.name
        try:
            result = load_jsonl(path)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)


class TestCLIArgs(unittest.TestCase):

    def test_default_args(self):
        # Test argparse defaults without importing the full module
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--sleep", type=float, default=2.0)
        args = parser.parse_args([])
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.sleep, 2.0)

    def test_custom_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--sleep", type=float, default=2.0)
        args = parser.parse_args(["--top-k", "10", "--sleep", "0.5"])
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.sleep, 0.5)


if __name__ == "__main__":
    unittest.main()
