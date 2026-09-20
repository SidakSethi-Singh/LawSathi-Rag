import unittest

from src.rag_pipelines.legal_relation_normalizer import (
    normalize_legal_relation_operators,
    tokenize_legal_relation_text,
)


class TestLegalRelationNormalization(unittest.TestCase):
    def test_under_section_aliases_share_one_form(self):
        variants = [
            "u/s 302",
            "u / s 302",
            "under section 302",
            "under Sec. 302",
            "under S. 302",
            "under § 302",
        ]

        for variant in variants:
            self.assertEqual(
                normalize_legal_relation_operators(variant),
                "under section 302",
            )

    def test_read_with_aliases_share_one_form(self):
        variants = [
            "r/w 34",
            "r / w 34",
            "read with section 34",
            "read with Sec. 34",
            "read along with section 34",
        ]

        for variant in variants:
            self.assertEqual(
                normalize_legal_relation_operators(variant),
                "read with section 34",
            )

    def test_relation_operator_is_kept_distinct(self):
        under_tokens = set(tokenize_legal_relation_text("u/s 302"))
        read_with_tokens = set(tokenize_legal_relation_text("r/w 302"))

        self.assertIn("under", under_tokens)
        self.assertIn("section", under_tokens)
        self.assertIn("read", read_with_tokens)
        self.assertIn("with", read_with_tokens)
        self.assertNotIn("under", read_with_tokens)

    def test_unrelated_text_is_preserved(self):
        text = "The court considered the evidence before deciding the application."
        self.assertEqual(
            normalize_legal_relation_operators(text),
            text,
        )


if __name__ == "__main__":
    unittest.main()
