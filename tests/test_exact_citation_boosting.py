import sys
import types
import unittest
from unittest.mock import MagicMock

for mod in ["chromadb", "chromadb.config", "sentence_transformers", "rank_bm25", "openai"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from src.rag_pipelines.citation_matcher import extract_exact_citation_signatures
from src.rag_pipelines.hybrid_rag import HybridRAG


class TestExactCitationBoosting(unittest.TestCase):
    def test_section_citation_signature_is_canonical(self):
        self.assertEqual(
            extract_exact_citation_signatures("See Section 302(1)(a)."),
            {"section:302(1)(a)"},
        )
        self.assertEqual(
            extract_exact_citation_signatures("See § 302 (1) (a)."),
            {"section:302(1)(a)"},
        )

    def test_reporter_citation_signatures_are_canonical(self):
        self.assertIn(
            "scc:2019:1:234",
            extract_exact_citation_signatures("AIR reference aside: (2019) 1 SCC 234"),
        )
        self.assertIn(
            "air:1980:sc:123",
            extract_exact_citation_signatures("AIR 1980 SC 123"),
        )

    def test_exact_citation_receives_positive_boost(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = [
            "Section 302. Exact statutory authority.",
            "Bail and sentencing principles discussed in detail.",
        ]
        rag.alpha = 0.7
        rag.exact_citation_boost = 0.5
        rag._retrieve_bm25 = MagicMock(
            return_value={
                rag.chunks[0]: 1.0,
                rag.chunks[1]: 0.0,
            }
        )
        rag._retrieve_dense = MagicMock(
            return_value={
                rag.chunks[0]: 0.4,
                rag.chunks[1]: 0.9,
            }
        )

        results = rag.retrieve("What does Section 302 provide?", k=2)

        self.assertEqual(results[0], rag.chunks[0])

    def test_ordinary_query_without_citation_is_unchanged(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["first", "second"]
        rag.alpha = 0.7
        rag.exact_citation_boost = 0.5
        rag._retrieve_bm25 = MagicMock(
            return_value={"first": 0.0, "second": 1.0}
        )
        rag._retrieve_dense = MagicMock(
            return_value={"first": 0.8, "second": 0.2}
        )

        results = rag.retrieve("bail", k=2)

        self.assertEqual(results, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
