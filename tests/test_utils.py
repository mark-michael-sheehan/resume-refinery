"""Tests for resume_refinery.utils — specifically apply_edits."""

import pytest

from resume_refinery.utils import EditApplicationError, apply_edits


def test_apply_edits_basic():
    """Single find/replace edit should work."""
    doc = "I am a passionate innovator."
    edits = [{"find": "passionate innovator", "replace": "backend engineer"}]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "I am a backend engineer."


def test_apply_edits_multiple():
    """Multiple non-overlapping edits should all be applied."""
    doc = "AAA and BBB and CCC."
    edits = [
        {"find": "AAA", "replace": "111"},
        {"find": "BBB", "replace": "222"},
        {"find": "CCC", "replace": "333"},
    ]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "111 and 222 and 333."


def test_apply_edits_reverse_order():
    """Edits should be applied in reverse document order so offsets don't drift."""
    doc = "short and a_longer_word here"
    edits = [
        {"find": "short", "replace": "extremely_long_replacement"},
        {"find": "a_longer_word", "replace": "x"},
    ]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "extremely_long_replacement and x here"


def test_apply_edits_no_match_under_threshold():
    """Edits that don't match should be skipped when under threshold."""
    doc = "Hello world."
    edits = [
        {"find": "Hello", "replace": "Hi"},
        {"find": "MISSING", "replace": "X"},
    ]
    result, regions = apply_edits(doc, edits, fail_threshold=1)
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
    result, regions = apply_edits(doc, edits, fail_threshold=1)
    assert result == "Hello world."


def test_apply_edits_empty_list():
    """An empty edit list should return the document unchanged."""
    doc = "No changes."
    result, regions = apply_edits(doc, [])
    assert result == "No changes."
    assert regions == []


def test_apply_edits_deletion():
    """Replacing with empty string should delete the matched text."""
    doc = "Remove this_phrase from the text."
    edits = [{"find": "this_phrase ", "replace": ""}]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "Remove from the text."
    # Deletion (empty replace) should NOT produce a region
    assert regions == []


def test_apply_edits_contained_overlap():
    """When one edit's find range is inside another's, the contained edit is dropped."""
    doc = "I am a quick brown fox jumping."
    edits = [
        {"find": "quick brown fox", "replace": "fast red dog"},
        {"find": "brown", "replace": "crimson"},
    ]
    # "brown" at pos 13 is inside "quick brown fox" at pos 8 (end 23).
    # The contained edit is dropped; only the broader edit applies.
    result, regions = apply_edits(doc, edits, fail_threshold=1)
    assert result == "I am a fast red dog jumping."


def test_apply_edits_partial_overlap():
    """Partially overlapping edits: the later-starting one is dropped."""
    doc = "AABBCCDD"
    edits = [
        {"find": "AABB", "replace": "XX"},
        {"find": "BBCC", "replace": "YY"},
    ]
    # "AABB" at 0 (end 4), "BBCC" at 2 (end 6) — overlap at positions 2-3.
    result, regions = apply_edits(doc, edits, fail_threshold=1)
    assert result == "XXCCDD"


def test_apply_edits_duplicate_find_text():
    """Two edits with the same find text: each applies to successive occurrences."""
    doc = "hello world hello"
    edits = [
        {"find": "hello", "replace": "HI"},
        {"find": "hello", "replace": "BYE"},
    ]
    # Both map to position 0 in the original.  Sequential re-find:
    # edit 1 replaces the first "hello" → "HI world hello"
    # edit 2 re-finds "hello" → matches the second occurrence.
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "HI world BYE"


def test_apply_edits_adjacent_non_overlapping():
    """Adjacent edits (end-of-A == start-of-B) should both be applied."""
    doc = "AAABBB"
    edits = [
        {"find": "AAA", "replace": "XX"},
        {"find": "BBB", "replace": "YY"},
    ]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "XXYY"


def test_apply_edits_overlap_counted_as_failure():
    """Dropped overlapping edits count toward the failure threshold."""
    doc = "one two three"
    edits = [
        {"find": "one two", "replace": "X"},
        {"find": "two", "replace": "Y"},  # overlapping, will be dropped
        {"find": "MISSING", "replace": "Z"},  # not found
    ]
    # 2 failures (overlap + missing) with threshold 1 → should raise
    with pytest.raises(EditApplicationError) as exc_info:
        apply_edits(doc, edits, fail_threshold=1)
    assert len(exc_info.value.failed) == 2


# ------------------------------------------------------------------
# Edit region tracking tests
# ------------------------------------------------------------------


def test_apply_edits_returns_regions():
    """apply_edits should return EditRegion objects for each successful edit."""
    doc = "AAA BBB CCC"
    edits = [
        {"find": "AAA", "replace": "XX"},
        {"find": "CCC", "replace": "YYYY"},
    ]
    result, regions = apply_edits(
        doc, edits, fail_threshold=0, reviewer="truthfulness", pass_num=2,
    )
    assert result == "XX BBB YYYY"
    assert len(regions) == 2
    # First region: "XX" starts at 0, length 2
    assert regions[0].start == 0
    assert regions[0].end == 2
    assert regions[0].reviewer == "truthfulness"
    assert regions[0].pass_num == 2
    # Second region: "YYYY" — "XX BBB " is 7 chars, so starts at 7
    assert regions[1].start == 7
    assert regions[1].end == 11
    assert regions[1].reviewer == "truthfulness"


def test_apply_edits_deletion_no_region():
    """Deleting text (replace='') should not produce a region."""
    doc = "keep DELETE keep"
    edits = [{"find": "DELETE ", "replace": ""}]
    result, regions = apply_edits(doc, edits, fail_threshold=0)
    assert result == "keep keep"
    assert regions == []


def test_edit_region_overlaps():
    """EditRegion.overlaps should detect overlapping spans."""
    from resume_refinery.models import EditRegion
    region = EditRegion(start=10, end=20, reviewer="truthfulness", pass_num=0)
    assert region.overlaps(15, 25)  # overlaps right
    assert region.overlaps(5, 15)   # overlaps left
    assert region.overlaps(12, 18)  # contained
    assert region.overlaps(5, 25)   # contains
    assert not region.overlaps(20, 30)  # adjacent, no overlap
    assert not region.overlaps(0, 10)   # adjacent left, no overlap
    assert not region.overlaps(25, 30)  # disjoint


def test_document_edit_history_is_protected():
    """Phrases inside a high-prio region are protected from lower-prio reviewers."""
    from resume_refinery.models import DocumentEditHistory, EditRegion
    hist = DocumentEditHistory(regions=[
        EditRegion(start=10, end=30, reviewer="truthfulness", pass_num=0),
    ])
    doc = "0123456789PROTECTED_REGION_HERE0rest"
    # "PROTECTED_REGION_HERE" starts at 10, inside the region
    assert hist.is_protected("PROTECTED_REGION_HERE", doc, "voice")  # lower prio
    assert hist.is_protected("PROTECTED_REGION_HERE", doc, "truthfulness")  # same prio
    # Text outside the region should not be protected
    assert not hist.is_protected("rest", doc, "voice")


def test_document_edit_history_lower_prio_not_protected():
    """A region from a low-prio reviewer should NOT protect against a higher-prio one."""
    from resume_refinery.models import DocumentEditHistory, EditRegion
    hist = DocumentEditHistory(regions=[
        EditRegion(start=0, end=10, reviewer="voice", pass_num=0),
    ])
    doc = "voice_edit rest of doc"
    # Truthfulness has higher priority than voice — should NOT be protected
    assert not hist.is_protected("voice_edit", doc, "truthfulness")
