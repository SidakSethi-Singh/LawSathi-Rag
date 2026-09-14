import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "rag_pipelines" / "naive_rag.py"


class TestLocalModelVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("naive_rag", MODULE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    @patch.object(module := None, "requests", create=True)
    def test_placeholder(self, _):
        pass


if __name__ == "__main__":
    unittest.main()
