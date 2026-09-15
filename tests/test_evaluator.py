import pytest

from src.evaluation.evaluator import evaluate_custom


def _make_pred(question, chunks, answer="answer", latency=10.0):
    return {
        "question": question,
        "predicted_answer": answer,
        "retrieved_chunks": chunks,
        "latency_ms": latency,
    }


def _make_gt(question, answer):
    return {"question": question, "answer": answer}


class TestRecallAt5:
    def test_recall_differs_from_precision_when_pool_exceeds_five(self):
        relevant = "indian penal code section 302"
        irrelevant = "completely unrelated chunk about weather"
        chunks = [relevant] * 3 + [irrelevant] * 2 + [relevant] * 4

        preds = [_make_pred("q1", chunks)]
        gt = [_make_gt("q1", "indian penal code section 302")]

        result = evaluate_custom(preds, gt)

        assert result["Precision@5"] == pytest.approx(3 / 5.0)
        assert result["Recall@5"] == pytest.approx(3 / 7.0)
        assert result["Recall@5"] != result["Precision@5"]

    def test_recall_is_one_when_all_relevant_in_top_five(self):
        relevant = "supreme court judgment on bail"
        irrelevant = "random text"
        chunks = [relevant] * 3 + [irrelevant] * 2

        preds = [_make_pred("q1", chunks)]
        gt = [_make_gt("q1", "supreme court judgment on bail")]

        result = evaluate_custom(preds, gt)

        assert result["Recall@5"] == pytest.approx(1.0)

    def test_recall_zero_when_no_relevant_chunks(self):
        chunks = ["nothing here"] * 5

        preds = [_make_pred("q1", chunks)]
        gt = [_make_gt("q1", "criminal procedure code")]

        result = evaluate_custom(preds, gt)

        assert result["Recall@5"] == pytest.approx(0.0)
        assert result["Precision@5"] == pytest.approx(0.0)

    def test_precision_at_5_unchanged(self):
        relevant = "article 21 right to life"
        chunks = [relevant] * 4 + ["irrelevant"] * 1

        preds = [_make_pred("q1", chunks)]
        gt = [_make_gt("q1", "article 21 right to life")]

        result = evaluate_custom(preds, gt)

        assert result["Precision@5"] == pytest.approx(4 / 5.0)