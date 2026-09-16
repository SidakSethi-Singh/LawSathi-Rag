import json
import tempfile
import unittest
from pathlib import Path

from src.utils.helpers import load_jsonl


class JsonlHelperTests(unittest.TestCase):
    def test_loads_utf8_records_and_ignores_empty_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.jsonl"
            path.write_text(
                json.dumps({"question": "¿Qué es el hábeas corpus?"}, ensure_ascii=False)
                + "\n\n"
                + json.dumps({"answer": "A remedy"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_jsonl(path),
                [
                    {"question": "¿Qué es el hábeas corpus?"},
                    {"answer": "A remedy"},
                ],
            )

    def test_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(load_jsonl(Path(temp_dir) / "missing.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
