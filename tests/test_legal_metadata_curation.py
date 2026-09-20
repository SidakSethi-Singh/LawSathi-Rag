import sys
import types
import unittest
from unittest.mock import patch
from unittest.mock import patch

for module_name in ["dotenv", "pandas", "tiktoken", "tqdm", "openai"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None
sys.modules["pandas"].read_csv = lambda *args, **kwargs: None
sys.modules["tiktoken"].get_encoding = lambda *args, **kwargs: None
sys.modules["tqdm"].tqdm = lambda iterable, **kwargs: iterable

from src.preprocessing import curate_benchmark


class TestLegalMetadataCuration(unittest.TestCase):
    def test_extracts_supported_top_level_fields_without_changing_values(self):
        metadata = curate_benchmark.extract_legal_metadata(
            {
                "case_name": "A v. B",
                "court": "Supreme Court of India",
                "citation": "2024 INSC 123",
                "year": 2024,
                "act": ["Indian Penal Code", "Evidence Act"],
                "section": "302",
                "context": "Judgment text",
            }
        )

        self.assertEqual(
            metadata,
            {
                "case_name": "A v. B",
                "court": "Supreme Court of India",
                "citation": "2024 INSC 123",
                "year": 2024,
                "act": ["Indian Penal Code", "Evidence Act"],
                "section": "302",
            },
        )

    def test_supports_nested_metadata_and_aliases(self):
        metadata = curate_benchmark.extract_legal_metadata(
            {
                "question": "What happened?",
                "answer": "Something.",
                "context": "Judgment text",
                "metadata": {
                    "caseName": "A v. B",
                    "court_name": "Supreme Court of India",
                    "reporter_citation": "2024 INSC 123",
                    "decision_year": 2024,
                    "decision_date": "2024-01-15",
                    "statute": "Indian Penal Code",
                    "provision": "Section 302",
                },
            }
        )

        self.assertEqual(
            metadata,
            {
                "case_name": "A v. B",
                "court": "Supreme Court of India",
                "citation": "2024 INSC 123",
                "year": 2024,
                "date": "2024-01-15",
                "act": "Indian Penal Code",
                "section": "Section 302",
            },
        )

    def test_process_record_persists_legal_metadata(self):
        record = {
            "question": "What was decided?",
            "answer": "The court decided X.",
            "context": "The court considered Section 302.",
            "case_name": "A v. B",
            "court": "Supreme Court of India",
            "citation": "2024 INSC 123",
            "year": 2024,
            "act": "Indian Penal Code",
            "section": "302",
        }

        with patch.object(curate_benchmark, "chunk_text", return_value=["The court considered Section 302."]):
            processed = curate_benchmark.process_record(record, 7)

        self.assertIsNotNone(processed)
        self.assertEqual(
            processed["metadata"],
            {
                "case_name": "A v. B",
                "court": "Supreme Court of India",
                "citation": "2024 INSC 123",
                "year": 2024,
                "act": "Indian Penal Code",
                "section": "302",
            },
        )
        self.assertEqual(processed["id"], "q_0007")


if __name__ == "__main__":
    unittest.main()
