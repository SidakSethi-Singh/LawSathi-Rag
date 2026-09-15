import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.utils.helpers import save_jsonl


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output.jsonl"


class TestSaveJsonl:
    def test_writes_valid_jsonl(self, tmp_output):
        data = [{"a": 1}, {"b": 2}]
        save_jsonl(tmp_output, data)
        lines = tmp_output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "output.jsonl"
        save_jsonl(nested, [{"x": 1}])
        assert nested.exists()

    def test_propagates_serialization_error(self, tmp_output):
        unserializable = [{"bad": object()}]
        with pytest.raises(TypeError):
            save_jsonl(tmp_output, unserializable)

    def test_no_partial_file_on_failure(self, tmp_output):
        unserializable = [{"good": 1}, {"bad": object()}]
        with pytest.raises(TypeError):
            save_jsonl(tmp_output, unserializable)
        assert not tmp_output.exists()

    def test_does_not_corrupt_existing_file_on_failure(self, tmp_output):
        save_jsonl(tmp_output, [{"original": True}])
        with pytest.raises(TypeError):
            save_jsonl(tmp_output, [{"bad": object()}])
        lines = tmp_output.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0]) == {"original": True}

    def test_propagates_permission_error(self, tmp_path):
        target = tmp_path / "no_write" / "output.jsonl"
        with patch("src.utils.helpers.tempfile.mkstemp", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                save_jsonl(target, [{"a": 1}])
