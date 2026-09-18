import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import tempfile

from src.evaluation.evaluator import (
    evaluate_ragas,
    generate_comparison_table,
    generate_figures,
    generate_html_report,
    RAGAS_METRIC_NAMES
)
from src.utils import config


class EvaluatorRagasTests(unittest.TestCase):
    """Unit tests for RAGAS evaluation, failure handling, and table generation."""

    def setUp(self):
        self.predictions = [
            {
                "question": "What is article 21?",
                "predicted_answer": "Protection of life and personal liberty.",
                "retrieved_chunks": ["Article 21 guarantees protection of life and liberty."],
                "latency_ms": 120.0
            }
        ]
        self.ground_truth = [
            {
                "question": "What is article 21?",
                "answer": "Article 21 provides that no person shall be deprived of life or personal liberty."
            }
        ]

    def test_evaluate_ragas_returns_none_on_exception(self):
        """Verify evaluate_ragas catches exceptions, logs them, and returns None for each metric."""
        with patch("src.evaluation.evaluator.logger.exception") as mock_log:
            with patch.dict("sys.modules", {"ragas": None}):
                scores = evaluate_ragas(self.predictions, self.ground_truth)

        for metric in RAGAS_METRIC_NAMES:
            self.assertIn(metric, scores)
            self.assertIsNone(scores[metric])
        mock_log.assert_called_once()

    def test_evaluate_ragas_preserves_columns_in_comparison_table(self):
        """Verify that failed RAGAS metrics preserve all 9 columns in comparison table CSV."""
        results = {
            "NaiveRAG": {
                "Exact Match": 0.0,
                "Token F1": 0.057,
                "Precision@5": 0.644,
                "Recall@5": 0.644,
                "Avg Latency (ms)": 3818.454,
                "Faithfulness": None,
                "Answer Relevancy": None,
                "Context Precision": None,
                "Context Recall": None
            },
            "DenseRAG": {
                "Exact Match": 0.0,
                "Token F1": 0.057,
                "Precision@5": 0.375,
                "Recall@5": 0.375,
                "Avg Latency (ms)": 2072.249,
                "Faithfulness": None,
                "Answer Relevancy": None,
                "Context Precision": None,
                "Context Recall": None
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "comparison_table.csv"
            df = generate_comparison_table(results, csv_path)

            expected_cols = [
                "Exact Match",
                "Token F1",
                "Precision@5",
                "Recall@5",
                "Avg Latency (ms)",
                "Faithfulness",
                "Answer Relevancy",
                "Context Precision",
                "Context Recall"
            ]
            self.assertEqual(list(df.columns), expected_cols)
            self.assertTrue(csv_path.exists())

            # Read back CSV and ensure column headers are preserved
            saved_df = pd.read_csv(csv_path, index_col=0)
            self.assertEqual(list(saved_df.columns), expected_cols)
            self.assertTrue(saved_df["Faithfulness"].isna().all())

    def test_evaluate_ragas_success_mocked(self):
        """Verify that a successful RAGAS evaluate call maps metrics to title case names."""
        mock_evaluate = MagicMock()
        mock_res_df = pd.DataFrame([{
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.75,
            "context_recall": 0.80
        }])
        mock_res = MagicMock()
        mock_res.to_pandas.return_value = mock_res_df
        mock_evaluate.return_value = mock_res

        mock_ragas = MagicMock()
        mock_ragas.evaluate = mock_evaluate

        mock_datasets = MagicMock()
        mock_datasets.Dataset.from_dict.return_value = MagicMock()

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "ragas": mock_ragas,
            "ragas.run_config": MagicMock(),
            "ragas.llms": MagicMock(),
            "ragas.embeddings": MagicMock(),
            "ragas.metrics": MagicMock(),
            "datasets": mock_datasets,
            "openai": mock_openai,
            "langchain_community.embeddings": MagicMock()
        }):
            scores = evaluate_ragas(self.predictions, self.ground_truth)

        self.assertAlmostEqual(scores["Faithfulness"], 0.85)
        self.assertAlmostEqual(scores["Answer Relevancy"], 0.90)
        self.assertAlmostEqual(scores["Context Precision"], 0.75)
        self.assertAlmostEqual(scores["Context Recall"], 0.80)

    def test_embedding_model_not_aliased_to_llm(self):
        """Verify that embedding configuration is distinct from LLM generator name."""
        self.assertTrue(hasattr(config, "EMBEDDING_MODEL_NAME"))
        self.assertNotEqual(config.EMBEDDING_MODEL_NAME, config.MODEL_NAME)
        # LLM should not be used as embedding model identifier
        self.assertNotIn("llama", config.EMBEDDING_MODEL_NAME.lower())

    def test_generate_figures_skips_all_nan_columns(self):
        """Verify generate_figures does not create blank charts for all-NaN columns."""
        df = pd.DataFrame({
            "Exact Match": [0.0, 0.0],
            "Faithfulness": [None, None]
        }, index=["NaiveRAG", "DenseRAG"])

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = Path(tmpdir) / "figures"
            generate_figures(df, fig_dir)
            self.assertTrue((fig_dir / "Exact_Match.png").exists())
            self.assertFalse((fig_dir / "Faithfulness.png").exists())

    def test_generate_html_report_handles_missing_figures(self):
        """Verify generate_html_report handles missing figures gracefully."""
        df = pd.DataFrame({
            "Exact Match": [0.0, 0.0],
            "Faithfulness": [None, None]
        }, index=["NaiveRAG", "DenseRAG"])

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "index.html"
            generate_html_report(df, html_path)
            self.assertTrue(html_path.exists())
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Exact Match", content)


if __name__ == "__main__":
    unittest.main()
