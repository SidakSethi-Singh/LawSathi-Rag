import unittest
from src.evaluation.evaluator import check_chunk_relevance
class TestChunkRelevance(unittest.TestCase):
    def test_matching_gold_context_is_relevant(self):
        chunk = "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        gold_contexts = [
            "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        ]
        self.assertTrue(check_chunk_relevance(chunk, gold_contexts))
    def test_unrelated_chunk_is_not_relevant(self):
        chunk = "Delhi Development Authority vs. Vandana Gupta"
        gold_contexts = [
            "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        ]
        self.assertFalse(check_chunk_relevance(chunk, gold_contexts))
    def test_answer_word_overlap_does_not_make_chunk_relevant(self):
        chunk = "The main issue was the classification of the product."
        gold_contexts = [
            "Union of India & Ors. vs. M/s. Hamdard (Waqf) Laboratories"
        ]
        self.assertFalse(check_chunk_relevance(chunk, gold_contexts))
if __name__ == "__main__":
    unittest.main()
