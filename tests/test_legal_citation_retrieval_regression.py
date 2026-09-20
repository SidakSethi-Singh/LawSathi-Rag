import sys
import unittest
from unittest.mock import MagicMock

for mod in ["rank_bm25", "openai", "dotenv"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from src.rag_pipelines.naive_rag import NaiveRAG


LEGAL_FIXTURES = [
    {
        "id": "section_302",
        "text": "Indian Penal Code Section 302 prescribes punishment for murder.",
        "queries": ["Section 302", "murder"],
        "distractors": ["Section 304", "Section 307"],
    },
    {
        "id": "section_302_subsection",
        "text": "Section 302(1)(a) applies where the statutory condition is satisfied.",
        "queries": ["Section 302(1)(a)", "302(1)(a)"],
        "distractors": ["Section 302(1)(b)", "Section 304(1)(a)"],
    },
    {
        "id": "scc_citation",
        "text": "The precedent is reported at (2019) 1 SCC 234 and concerns statutory interpretation.",
        "queries": ["(2019) 1 SCC 234", "SCC 234"],
        "distractors": ["(2019) 2 SCC 234", "AIR 2020 SC 123"],
    },
    {
        "id": "case_name",
        "text": "Kesavananda Bharati v State of Kerala established a constitutional doctrine.",
        "queries": ["Kesavananda Bharati v State of Kerala"],
        "distractors": [
            "Golaknath v State of Punjab",
            "Maneka Gandhi v Union of India",
        ],
    },
    {
        "id": "court",
        "text": "The Supreme Court of India considered the constitutional question.",
        "queries": ["Supreme Court of India", "Supreme Court"],
        "distractors": ["Delhi High Court", "Bombay High Court"],
    },
    {
        "id": "metadata",
        "text": "Case: State of Kerala v Thomas Court: Supreme Court of India Year: 2018 Act: Evidence Act Section: 65B.",
        "queries": ["State of Kerala v Thomas", "2018", "Evidence Act", "65B"],
        "distractors": [
            "State of Kerala v Joseph",
            "2017",
            "Arbitration Act",
            "Section 45",
        ],
    },
    {
        "id": "duplicate_case_a",
        "text": "The court held that the petition was maintainable.",
        "queries": ["petition maintainable"],
        "distractors": [],
    },
    {
        "id": "duplicate_case_b",
        "text": "The court held that the petition was maintainable.",
        "queries": ["petition maintainable"],
        "distractors": [],
    },
]


class LegalCitationRetrievalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = NaiveRAG()
        cls.rag.index_documents([fixture["text"] for fixture in LEGAL_FIXTURES])

    def retrieve_top(self, query, k=3):
        return self.rag.retrieve(query, k=k)

    def assert_contains_fixture(self, fixture_id, query):
        fixture = next(item for item in LEGAL_FIXTURES if item["id"] == fixture_id)
        self.assertIn(fixture["text"], self.retrieve_top(query, k=3))

    def test_statutory_section_positive_and_adversarial_matches(self):
        self.assert_contains_fixture("section_302", "Section 302")
        results = self.retrieve_top("Section 302", k=1)
        self.assertEqual(
            results[0],
            next(item for item in LEGAL_FIXTURES if item["id"] == "section_302")["text"],
        )

    def test_subsection_reference_is_more_specific_than_sibling_clause(self):
        self.assert_contains_fixture("section_302_subsection", "Section 302(1)(a)")
        results = self.retrieve_top("Section 302(1)(a)", k=3)
        sibling = next(
            item for item in LEGAL_FIXTURES if item["id"] == "section_302"
        )["text"]
        self.assertIn(
            next(
                item for item in LEGAL_FIXTURES if item["id"] == "section_302_subsection"
            )["text"],
            results,
        )
        self.assertTrue(results.index(
            next(
                item for item in LEGAL_FIXTURES if item["id"] == "section_302_subsection"
            )["text"]
        ) < results.index(sibling))

    def test_reporter_citation_positive_and_adversarial_matches(self):
        self.assert_contains_fixture("scc_citation", "(2019) 1 SCC 234")
        self.assert_contains_fixture("scc_citation", "SCC 234")
        results = self.retrieve_top("SCC 234", k=1)
        self.assertEqual(
            results[0],
            next(item for item in LEGAL_FIXTURES if item["id"] == "scc_citation")["text"],
        )

    def test_case_name_positive_and_adversarial_matches(self):
        self.assert_contains_fixture(
            "case_name", "Kesavananda Bharati v State of Kerala"
        )
        results = self.retrieve_top("Kesavananda Bharati v State of Kerala", k=3)
        self.assertEqual(
            results[0],
            next(item for item in LEGAL_FIXTURES if item["id"] == "case_name")["text"],
        )

    def test_court_positive_and_adversarial_matches(self):
        self.assert_contains_fixture("court", "Supreme Court of India")
        results = self.retrieve_top("Supreme Court of India", k=3)
        self.assertEqual(
            results[0],
            next(item for item in LEGAL_FIXTURES if item["id"] == "court")["text"],
        )

    def test_legal_metadata_fields_are_retrievable(self):
        fixture = next(item for item in LEGAL_FIXTURES if item["id"] == "metadata")
        for query in fixture["queries"]:
            self.assertIn(fixture["text"], self.retrieve_top(query, k=3))

    def test_metadata_queries_retrieve_exact_record(self):
        fixture = next(item for item in LEGAL_FIXTURES if item["id"] == "metadata")
        for query in ["2018", "Evidence Act", "65B"]:
            results = self.retrieve_top(query, k=3)
            self.assertIn(fixture["text"], results)

    def test_duplicate_text_fixture_is_retained_as_distinct_source_data(self):
        duplicate_text = next(
            item["text"]
            for item in LEGAL_FIXTURES
            if item["id"] == "duplicate_case_a"
        )
        duplicate_ids = [
            item["id"]
            for item in LEGAL_FIXTURES
            if item["text"] == duplicate_text
        ]
        self.assertEqual(
            duplicate_ids,
            ["duplicate_case_a", "duplicate_case_b"],
        )

    def test_citation_should_rank_above_semantic_distractor(self):
        fixture = next(item for item in LEGAL_FIXTURES if item["id"] == "scc_citation")
        results = self.retrieve_top("(2019) 1 SCC 234", k=1)
        self.assertEqual(results[0], fixture["text"])

    def test_every_fixture_has_positive_and_adversarial_cases(self):
        for fixture in LEGAL_FIXTURES:
            self.assertTrue(fixture["queries"], fixture["id"])
            if fixture["id"] != "duplicate_case_a" and fixture["id"] != "duplicate_case_b":
                self.assertTrue(
                    fixture["distractors"],
                    fixture["id"],
                )


if __name__ == "__main__":
    unittest.main()
