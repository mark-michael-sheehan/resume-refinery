"""Tests for CLI output directory validation."""

from pathlib import Path

import pytest
import typer

from resume_refinery.cli import _validate_output_dir


def test_validate_output_dir_existing_directory(tmp_path):
    result = _validate_output_dir(tmp_path)
    assert result == tmp_path.resolve()


def test_validate_output_dir_new_subdirectory(tmp_path):
    new_dir = tmp_path / "new_subdir"
    result = _validate_output_dir(new_dir)
    assert result == new_dir.resolve()


def test_validate_output_dir_rejects_file_path(tmp_path):
    file_path = tmp_path / "somefile.txt"
    file_path.write_text("hello")
    with pytest.raises(typer.BadParameter, match="not a directory"):
        _validate_output_dir(file_path)


def test_validate_output_dir_rejects_missing_parent(tmp_path):
    deep = tmp_path / "nonexistent_parent" / "child"
    with pytest.raises(typer.BadParameter, match="Parent directory does not exist"):
        _validate_output_dir(deep)
