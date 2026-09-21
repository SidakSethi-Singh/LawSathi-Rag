import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

helpers = importlib.import_module("src.utils.helpers")


class TestSaveJsonl(unittest.TestCase):
    def test_successful_write_preserves_jsonl_output(self):
        data = [
            {"question": "What is a contract?", "answer": "An agreement."},
            {"question": "Who is a party?", "answer": "A participant."},
        ]
        expected = (
            '{"question": "What is a contract?", "answer": "An agreement."}\n'
            '{"question": "Who is a party?", "answer": "A participant."}\n'
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "output.jsonl"
            helpers.save_jsonl(file_path, data)
            self.assertEqual(file_path.read_text(encoding="utf-8"), expected)

    def test_write_failure_is_propagated(self):
        file_path = Path("output.jsonl")
        data = [{"question": "What is a contract?", "answer": "An agreement."}]
        write_error = OSError("disk full")

        with patch.object(helpers.Path, "mkdir"), patch.object(
            helpers.tempfile, "NamedTemporaryFile"
        ) as temp_file_factory:
            temp_file = temp_file_factory.return_value.__enter__.return_value
            temp_file.name = "temporary-output.jsonl"
            temp_file.write.side_effect = write_error
            with patch.object(helpers.Path, "unlink") as unlink, patch.object(
                helpers.os, "replace"
            ) as replace:
                with self.assertRaises(OSError) as raised:
                    helpers.save_jsonl(file_path, data)

        self.assertIs(raised.exception, write_error)
        replace.assert_not_called()
        unlink.assert_called_once_with(missing_ok=True)

    def test_replace_failure_is_propagated(self):
        file_path = Path("output.jsonl")
        data = [{"question": "What is a contract?", "answer": "An agreement."}]
        replace_error = OSError("permission denied")

        with patch.object(helpers.Path, "mkdir"), patch.object(
            helpers.tempfile, "NamedTemporaryFile"
        ) as temp_file_factory:
            temp_file = temp_file_factory.return_value.__enter__.return_value
            temp_file.name = "temporary-output.jsonl"
            with patch.object(helpers.os, "replace", side_effect=replace_error):
                with patch.object(helpers.Path, "unlink") as unlink:
                    with self.assertRaises(OSError) as raised:
                        helpers.save_jsonl(file_path, data)

        self.assertIs(raised.exception, replace_error)
        unlink.assert_called_once_with(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
