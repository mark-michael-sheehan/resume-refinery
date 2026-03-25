"""Tests for resume_refinery.utils — specifically apply_edits."""

import pytest

from resume_refinery.utils import EditApplicationError, apply_edits


def test_apply_edits_basic():
    """Single find/replace edit should work."""
    doc = "I am a passionate innovator."
    edits = [{"find": "passionate innovator", "replace": "backend engineer"}]
    result = apply_edits(doc, edits, fail_threshold=0)
    assert result == "I am a backend engineer."


def test_apply_edits_multiple():
    """Multiple non-overlapping edits should all be applied."""
    doc = "AAA and BBB and CCC."
    edits = [
        {"find": "AAA", "replace": "111"},
        {"find": "BBB", "replace": "222"},
        {"find": "CCC", "replace": "333"},
    ]
    result = apply_edits(doc, edits, fail_threshold=0)
    assert result == "111 and 222 and 333."


def test_apply_edits_reverse_order():
    """Edits should be applied in reverse document order so offsets don't drift."""
    doc = "short and a_longer_word here"
    edits = [
        {"find": "short", "replace": "extremely_long_replacement"},
        {"find": "a_longer_word", "replace": "x"},
    ]
    result = apply_edits(doc, edits, fail_threshold=0)
    assert result == "extremely_long_replacement and x here"


def test_apply_edits_no_match_under_threshold():
    """Edits that don't match should be skipped when under threshold."""
    doc = "Hello world."
    edits = [
        {"find": "Hello", "replace": "Hi"},
        {"find": "MISSING", "replace": "X"},
    ]
    result = apply_edits(doc, edits, fail_threshold=1)
    assert result == "Hi world."


def test_apply_edits_exceeds_threshold():
    """Should raise EditApplicationError when too many edits fail."""
    doc = "Hello world."
    edits = [
        {"find": "NOT_HERE_1", "replace": "A"},
        {"find": "NOT_HERE_2", "replace": "B"},
    ]
    with pytest.raises(EditApplicationError) as exc_info:
        apply_edits(doc, edits, fail_threshold=1)
    assert len(exc_info.value.failed) == 2
    assert exc_info.value.threshold == 1


def test_apply_edits_empty_find_counted_as_failure():
    """An edit with an empty 'find' should be counted as a failure."""
    doc = "Hello world."
    edits = [{"find": "", "replace": "X"}]
    result = apply_edits(doc, edits, fail_threshold=1)
    assert result == "Hello world."


def test_apply_edits_empty_list():
    """An empty edit list should return the document unchanged."""
    doc = "No changes."
    assert apply_edits(doc, []) == "No changes."


def test_apply_edits_deletion():
    """Replacing with empty string should delete the matched text."""
    doc = "Remove this_phrase from the text."
    edits = [{"find": "this_phrase ", "replace": ""}]
    result = apply_edits(doc, edits, fail_threshold=0)
    assert result == "Remove from the text."
