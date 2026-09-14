from src.evaluation.evaluator import compute_recall_at_k, evaluate_custom


def test_compute_recall_at_k_uses_relevant_ground_truth_denominator():
    answer = "breach of contract damages"
    relevant = [
        "The court considered breach of contract and damages.",
        "Damages were awarded for the contractual breach.",
        "This chunk is unrelated to the legal issue.",
    ]
    retrieved = [
        relevant[0],
        "Noise about an unrelated procedural matter.",
    ]

    assert compute_recall_at_k(retrieved, relevant, answer, k=5) == 0.5


def test_compute_recall_at_k_returns_zero_without_relevant_reference_chunks():
    assert compute_recall_at_k(
        ["breach of contract damages"],
        ["unrelated material"],
        "breach of contract damages",
        k=5,
    ) == 0.0


def test_evaluate_custom_no_longer_duplicates_precision_and_recall():
    predictions = [
        {
            "question": "q1",
            "predicted_answer": "breach of contract damages",
            "retrieved_chunks": [
                "The court considered breach of contract and damages.",
                "Noise one",
                "Noise two",
                "Noise three",
                "Noise four",
            ],
            "latency_ms": 10.0,
        }
    ]
    ground_truth = [
        {
            "question": "q1",
            "answer": "breach of contract damages",
            "context_chunks": [
                "The court considered breach of contract and damages.",
                "Damages were awarded for the contractual breach.",
            ],
        }
    ]

    metrics = evaluate_custom(predictions, ground_truth)

    assert metrics["Precision@5"] == 0.2
    assert metrics["Recall@5"] == 0.5
