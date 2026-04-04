"""Tests for webapp output directory validation."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from resume_refinery.webapp import _validate_output_dir


def test_validate_output_dir_existing_directory(tmp_path):
    result = _validate_output_dir(str(tmp_path))
    assert result == tmp_path.resolve()


def test_validate_output_dir_new_subdirectory(tmp_path):
    new_dir = tmp_path / "new_subdir"
    result = _validate_output_dir(str(new_dir))
    assert result == new_dir.resolve()


def test_validate_output_dir_rejects_empty_string():
    with pytest.raises(HTTPException) as exc_info:
        _validate_output_dir("   ")
    assert exc_info.value.status_code == 400
    assert "required" in exc_info.value.detail.lower()


def test_validate_output_dir_rejects_file_path(tmp_path):
    file_path = tmp_path / "somefile.txt"
    file_path.write_text("hello")
    with pytest.raises(HTTPException) as exc_info:
        _validate_output_dir(str(file_path))
    assert exc_info.value.status_code == 400
    assert "not a directory" in exc_info.value.detail


def test_validate_output_dir_rejects_missing_parent(tmp_path):
    deep = tmp_path / "nonexistent_parent" / "child"
    with pytest.raises(HTTPException) as exc_info:
        _validate_output_dir(str(deep))
    assert exc_info.value.status_code == 400
    assert "Parent directory does not exist" in exc_info.value.detail
