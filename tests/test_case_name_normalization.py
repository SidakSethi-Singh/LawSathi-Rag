import sys
import types
import unittest
from unittest.mock import patch

for module_name in ["dotenv", "pandas", "tiktoken", "tqdm", "openai", "rank_bm25"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None
sys.modules["pandas"].read_csv = lambda *args, **kwargs: None
sys.modules["tiktoken"].get_encoding = lambda *args, **kwargs: None
sys.modules["tqdm"].tqdm = lambda iterable, **kwargs: iterable
sys.modules["openai"].OpenAI = object

class FakeBM25Okapi:
    def __init__(self, tokenized_corpus):
        self.tokenized_corpus = tokenized_corpus

    def get_scores(self, tokenized_query):
        return [
            1.0 if any(token in tokenized_query for token in document) else 0.0
            for document in self.tokenized_corpus
        ]

sys.modules["rank_bm25"].BM25Okapi = FakeBM25Okapi

from src.rag_pipelines.case_name_normalizer import (
    case_name_tokens,
    normalize_case_name,
)
from src.rag_pipelines.naive_rag import NaiveRAG
from src.preprocessing import curate_benchmark


class TestCaseNameNormalization(unittest.TestCase):
    def test_common_variants_share_one_canonical_name(self):
        variants = [
            "A v B",
            "A vs B",
            "A vs. B",
            "A versus B",
        ]
        normalized = {
            normalize_case_name(value)
            for value in variants
        }
        self.assertEqual(normalized, {"a v b"})

    def test_party_suffix_aliases_share_one_canonical_form(self):
        variants = [
            "A vs B Ltd.",
            "A versus B Limited",
            "A v B pvt limited",
            "A vs B private ltd",
        ]
        normalized = {
            normalize_case_name(value)
            for value in variants
        }
        self.assertEqual(normalized, {"a v b ltd", "a v b pvt ltd"})

    def test_query_variants_retrieve_the_same_case(self):
        chunk = "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        rag = NaiveRAG()
        rag.index_documents([chunk])

        queries = [
            "What was the issue in the case of Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories?",
            "What was the issue in the case of Union of India and Ors versus M/s Hamdard Waqf Laboratories?",
            "Union of India & Ors. v. M/s. Hamdard (Waqf) Laboratories",
        ]

        for query in queries:
            self.assertEqual(
                rag.retrieve(query, k=1),
                [chunk],
                query,
            )

        self.assertTrue(case_name_tokens(queries[0]))

    def test_process_record_persists_canonical_case_name(self):
        record = {
            "question": "What happened in the case?",
            "answer": "The appeal was dismissed.",
            "context": "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories",
            "caseName": "Union of India and Ors versus M/S. Hamdard (Waqf) Laboratories",
        }

        with patch.object(
            curate_benchmark,
            "chunk_text",
            return_value=[record["context"]],
        ):
            processed = curate_benchmark.process_record(record, 1)

        self.assertEqual(
            processed["case_name"],
            "union of india and ors v ms hamdard waqf laboratories",
        )


if __name__ == "__main__":
    unittest.main()
