import sys
import types
import unittest

for module_name in ["dotenv", "openai"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None
sys.modules["openai"].OpenAI = object

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
        self.assertTrue(rag.citation_metadata[0])
        self.assertEqual(
            rag.citation_metadata[0][0],
            {"year": "2019", "volume": "1", "reporter": "scc", "page": "234"},
        )
        self.assertEqual(rag.retrieve("SCC 234", k=1), [citation])


if __name__ == "__main__":
    unittest.main()
