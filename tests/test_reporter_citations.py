import sys
import types
import unittest

if "openai" not in sys.modules:
    sys.modules["openai"] = types.ModuleType("openai")
sys.modules["openai"].OpenAI = object

if "rank_bm25" not in sys.modules:
    rank_bm25 = types.ModuleType("rank_bm25")

    class FakeBM25Okapi:
        last_tokens = None

        def __init__(self, tokenized_corpus):
            self.tokenized_corpus = tokenized_corpus
            FakeBM25Okapi.last_tokens = tokenized_corpus

        def get_scores(self, tokenized_query):
            self.query_tokens = tokenized_query
            return [1.0 if any(token in tokenized_query for token in doc) else 0.0 for doc in self.tokenized_corpus]

    rank_bm25.BM25Okapi = FakeBM25Okapi
    sys.modules["rank_bm25"] = rank_bm25

from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.citation_normalizer import extract_reporter_citations, citation_tokens


class TestReporterCitationNormalization(unittest.TestCase):
    def test_extracts_common_reporter_components(self):
        text = "The judgment is reported at (2019) 1 SCC 234."
        self.assertEqual(
            extract_reporter_citations(text),
            [{"year": "2019", "volume": "1", "reporter": "scc", "page": "234"}],
        )

    def test_parses_air_citation_with_court(self):
        text = "See AIR 2020 SC 123."
        self.assertEqual(
            extract_reporter_citations(text),
            [{"year": "2020", "reporter": "air", "court": "sc", "page": "123"}],
        )

    def test_partial_reporter_query_matches_indexed_citation_tokens(self):
        citation = "Union of India v. Example (2019) 1 SCC 234"
        rag = NaiveRAG()
        rag.index_documents([citation])
        tokens = citation_tokens("SCC 234")
        self.assertIn("__citation_reporter_scc__", tokens)
        self.assertIn("__citation_page_234__", tokens)
        self.assertTrue(
            any(token in rag.bm25.tokenized_corpus[0] for token in tokens),
            "Canonical citation tokens were not indexed with the document.",
        )
        self.assertEqual(rag.retrieve("SCC 234", k=1), [citation])


if __name__ == "__main__":
    unittest.main()
