import importlib
import sys
import types
import unittest
from unittest.mock import patch


pd_stub = types.ModuleType("pandas")
pd_stub.DataFrame = object
sys.modules.setdefault("pandas", pd_stub)

chunker_stub = types.ModuleType("src.preprocessing.chunker")
chunker_stub.chunk_documents = None
sys.modules["src.preprocessing.chunker"] = chunker_stub

ablation = None


class TestAblationDocumentBoundaries(unittest.TestCase):
    def test_build_ablation_chunks_preserves_source_boundaries(self):
        module = importlib.import_module("src.evaluation.ablation")
        questions = [
            {"context_chunks": ["DOC_A alpha beta", "DOC_B gamma delta"]},
        ]

        observed_documents = []

        def fake_chunk_documents(documents, chunk_size, overlap):
            observed_documents.extend(documents)
            return [f"chunk:{document}" for document in documents]

        with patch.object(module, "chunk_documents", side_effect=fake_chunk_documents):
            chunks = module.build_ablation_chunks(questions, chunk_size=256, overlap=50)

        self.assertEqual(observed_documents, ["DOC_A alpha beta", "DOC_B gamma delta"])
        self.assertEqual(chunks, ["chunk:DOC_A alpha beta", "chunk:DOC_B gamma delta"])

    def test_build_ablation_chunks_deduplicates_contexts_without_merging_boundaries(self):
        module = importlib.import_module("src.evaluation.ablation")
        questions = [
            {"context_chunks": ["DOC_A alpha", "DOC_A alpha", "DOC_B beta"]},
        ]

        observed_documents = []

        def fake_chunk_documents(documents, chunk_size, overlap):
            observed_documents.extend(documents)
            return list(documents)

        with patch.object(module, "chunk_documents", side_effect=fake_chunk_documents):
            chunks = module.build_ablation_chunks(questions, chunk_size=512, overlap=50)

        self.assertEqual(observed_documents, ["DOC_A alpha", "DOC_B beta"])
        self.assertEqual(chunks, ["DOC_A alpha", "DOC_B beta"])


if __name__ == "__main__":
    unittest.main()
