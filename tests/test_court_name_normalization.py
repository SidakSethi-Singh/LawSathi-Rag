import sys
import types
import unittest
from unittest.mock import patch

for module_name in ["dotenv", "pandas", "tiktoken", "tqdm", "openai"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None
sys.modules["pandas"].read_csv = lambda *args, **kwargs: None
sys.modules["tiktoken"].get_encoding = lambda *args, **kwargs: None
sys.modules["tqdm"].tqdm = lambda iterable, **kwargs: iterable
sys.modules["openai"].OpenAI = object

from src.preprocessing import curate_benchmark
from src.preprocessing.court_normalizer import extract_court_name, normalize_court_name


class TestCourtNameNormalization(unittest.TestCase):
    def test_supreme_court_aliases_share_one_identifier(self):
        variants = [
            "Supreme Court",
            "Supreme Court of India",
            "SC",
        ]
        self.assertEqual(
            {normalize_court_name(value) for value in variants},
            {"supreme_court_of_india"},
        )

    def test_high_court_aliases_share_one_identifier(self):
        variants = [
            "Delhi High Court",
            "High Court of Delhi",
            "Delhi HC",
            "High Court at Delhi",
        ]
        self.assertEqual(
            {normalize_court_name(value) for value in variants},
            {"delhi_high_court"},
        )

    def test_court_name_is_extracted_from_nested_metadata(self):
        record = {
            "question": "What was decided?",
            "answer": "The petition was dismissed.",
            "context": "Judgment text",
            "metadata": {
                "courtName": "High Court of Karnataka",
            },
        }

        self.assertEqual(
            extract_court_name(record),
            "karnataka_high_court",
        )

    def test_process_record_persists_canonical_court(self):
        record = {
            "question": "What was decided?",
            "answer": "The petition was dismissed.",
            "context": "Judgment text",
            "court": "Madras HC",
        }

        with patch.object(
            curate_benchmark,
            "chunk_text",
            return_value=["Judgment text"],
        ):
            processed = curate_benchmark.process_record(record, 1)

        self.assertEqual(processed["court"], "madras_high_court")


if __name__ == "__main__":
    unittest.main()
