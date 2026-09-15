import importlib
import sys
import types
import unittest
from unittest import mock


class TestEvaluatorImport(unittest.TestCase):
    def test_metric_helpers_import_without_pandas(self):
        real_import = __import__

        def blocked_pandas_import(name, *args, **kwargs):
            if name == "pandas":
                raise ModuleNotFoundError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        fake_config = types.ModuleType("src.utils.config")
        with (
            mock.patch.dict(sys.modules, {"src.utils.config": fake_config}),
            mock.patch("builtins.__import__", side_effect=blocked_pandas_import),
        ):
            evaluator = importlib.import_module("src.evaluation.evaluator")

        self.assertEqual(evaluator.compute_em("Allowed", " allowed "), 1.0)


if __name__ == "__main__":
    unittest.main()
