import unittest
from pathlib import Path
import tempfile
import json
import shutil

from src.preprocessing.curate_benchmark import (
    extract_qa_fields,
    validate_context,
    process_record,
    load_judgment_corpus,
    QUESTION_KEYS,
    ANSWER_KEYS,
    CONTEXT_KEYS
)

class TestCurateBenchmark(unittest.TestCase):
    def setUp(self):
        self.substantive_context = (
            "The Supreme Court of India held that when evaluating excise duty classifications under the "
            "Central Excise Tariff Act, 1986, the primary test remains the commercial parlance test. Goods "
            "must be classified according to how they are known in common trade and market perception, rather "
            "than purely scientific or technical definitions. The respondent's product constitutes a flavored beverage base "
            "and does not qualify for exemptions granted exclusively to natural mineral water preparations under Chapter 22."
        )
        self.case_name = "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_qa_fields_with_explicit_context(self):
        record = {
            "question": "What is the primary test for excise duty classification?",
            "answer": "The commercial parlance test based on common market perception.",
            "context": self.substantive_context,
            "case_name": self.case_name
        }
        q, a, c, case = extract_qa_fields(record)
        self.assertEqual(q, record["question"])
        self.assertEqual(a, record["answer"])
        self.assertEqual(c, self.substantive_context)
        self.assertEqual(case, self.case_name)

    def test_case_name_not_treated_as_context(self):
        """Verify that case_name is extracted as metadata, NOT as context passage."""
        record = {
            "question": "What was decided?",
            "answer": "The appeal was dismissed.",
            "case_name": self.case_name
        }
        q, a, c, case = extract_qa_fields(record)
        self.assertEqual(case, self.case_name)
        # Context must NOT fall back to case_name
        self.assertEqual(c, "")

    def test_alternative_context_keys(self):
        """Verify candidate context keys like judgment_text and full_text are mapped."""
        for key in ["judgment_text", "full_text", "passage", "document"]:
            rec = {
                "question": "Test question?",
                "answer": "Test answer.",
                key: self.substantive_context
            }
            q, a, c, case = extract_qa_fields(rec)
            self.assertEqual(c, self.substantive_context, f"Failed for key {key}")

    def test_validate_context_rejects_empty_and_titles(self):
        # Empty context
        is_valid, reason = validate_context("")
        self.assertFalse(is_valid)
        self.assertIn("empty", reason.lower())

        # Context identical to case title
        is_valid, reason = validate_context(self.case_name, case_name=self.case_name)
        self.assertFalse(is_valid)
        self.assertIn("identical to case title", reason.lower())

        # Short snippet / metadata header (below 30 tokens)
        short_snippet = "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        is_valid, reason = validate_context(short_snippet, min_tokens=30)
        self.assertFalse(is_valid)
        self.assertIn("too short", reason.lower())

    def test_validate_context_accepts_substantive_text(self):
        is_valid, reason = validate_context(self.substantive_context, case_name=self.case_name, min_tokens=30)
        self.assertTrue(is_valid)
        self.assertIn("valid", reason.lower())

    def test_external_judgment_corpus_lookup(self):
        # Create a mock judgment file in temp dir
        judgments_dir = Path(self.temp_dir) / "judgments"
        judgments_dir.mkdir()
        file_path = judgments_dir / "hamdard_judgment.txt"
        file_path.write_text(self.substantive_context, encoding="utf-8")

        # Load corpus with normalized stem
        lookup = {"union of india & ors. vs. m/s. hamdard (waqf) laboratories": self.substantive_context}
        rec = {
            "question": "What is the excise duty test?",
            "answer": "Commercial parlance test.",
            "case_name": self.case_name
        }
        q, a, c, case = extract_qa_fields(rec, judgments_lookup=lookup)
        self.assertEqual(c, self.substantive_context)

    def test_process_record_filters_metadata_only_records(self):
        """Verify process_record rejects records where context is only a case title."""
        rec = {
            "question": "What was the issue?",
            "answer": "Excise classification.",
            "case_name": self.case_name
            # No context provided, only case_name
        }
        res = process_record(rec, idx=0, min_tokens=30, strict_context=True)
        self.assertIsNone(res, "Expected record to be filtered out due to missing substantive context.")

    def test_process_record_success(self):
        rec = {
            "question": "What was the rule?",
            "answer": "Commercial parlance rule.",
            "case_name": self.case_name,
            "context": self.substantive_context
        }
        res = process_record(rec, idx=1, min_tokens=30, strict_context=True)
        self.assertIsNotNone(res)
        self.assertEqual(res["id"], "q_0001")
        self.assertEqual(res["case_name"], self.case_name)
        self.assertTrue(len(res["context_chunks"]) > 0)
        self.assertIn("commercial parlance", res["context_chunks"][0].lower())

if __name__ == "__main__":
    unittest.main()
