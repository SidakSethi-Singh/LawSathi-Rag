import unittest

from src.rag_pipelines import section_reference_normalizer


class TestSectionReferenceNormalization(unittest.TestCase):
    def test_parent_and_nested_reference_tokens_are_distinct(self):
        text = "Section 302(1)(a) applies to the offence."
        normalized = section_reference_normalizer.normalize_section_references(text)

        self.assertIn("legal_section_302", normalized)
        self.assertIn("legal_subsection_302_1", normalized)
        self.assertIn("legal_clause_302_1_a", normalized)

    def test_parent_section_matches_nested_reference(self):
        parent = section_reference_normalizer.tokenize_legal_reference_text("section 302")
        nested = section_reference_normalizer.tokenize_legal_reference_text("section 302(1)(a)")

        self.assertIn("legal_section_302", parent)
        self.assertIn("legal_section_302", nested)
        self.assertIn("legal_clause_302_1_a", nested)
        self.assertNotIn("legal_clause_302_1_a", parent)

    def test_range_expands_small_numeric_ranges(self):
        normalized = section_reference_normalizer.normalize_section_references(
            "Sections 302-304 apply."
        )

        self.assertIn("legal_section_range_302_304", normalized)
        self.assertIn("legal_section_302", normalized)
        self.assertIn("legal_section_303", normalized)
        self.assertIn("legal_section_304", normalized)

    def test_large_ranges_keep_the_range_identity_without_expanding(self):
        normalized = section_reference_normalizer.normalize_section_references(
            "Sections 100-150 apply."
        )

        self.assertIn("legal_section_range_100_150", normalized)
        self.assertNotIn("legal_section_125", normalized)

    def test_aliases_use_the_same_canonical_section(self):
        variants = ["Section 302", "Sec. 302", "S. 302", "§ 302"]

        tokens = [
            set(section_reference_normalizer.tokenize_legal_reference_text(value))
            for value in variants
        ]

        self.assertTrue(all(token_set == tokens[0] for token_set in tokens))


if __name__ == "__main__":
    unittest.main()
