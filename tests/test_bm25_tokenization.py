import unittest

from src.preprocessing.cleaners import tokenize_legal_text
from src.rag_pipelines.naive_rag import NaiveRAG


class BM25TokenizationTests(unittest.TestCase):
    def test_punctuation_does_not_change_legal_term_tokens(self):
        self.assertEqual(
            tokenize_legal_text("302, IPC"),
            tokenize_legal_text("302 IPC"),
        )

    def test_section_reference_is_tokenized_consistently(self):
        self.assertEqual(
            tokenize_legal_text("Section 302, IPC"),
            ["section", "302", "ipc"],
        )

    def test_document_and_query_use_compatible_tokens(self):
        document_tokens = tokenize_legal_text("Section 302, IPC")
        query_tokens = tokenize_legal_text("section 302 IPC")
        self.assertEqual(document_tokens, query_tokens)

    def test_normal_bm25_retrieval_still_matches_query(self):
        rag = NaiveRAG()
        rag.index_documents(
            [
                "Section 302, IPC penal code.",
                "Section 420, IPC civil procedure.",
                "The court reviewed a procedural motion.",
            ]
        )

        self.assertEqual(
            rag.retrieve("section 302 IPC penal", k=1),
            ["Section 302, IPC penal code."],
        )


if __name__ == "__main__":
    unittest.main()
