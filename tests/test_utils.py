"""Tests for resume_refinery.utils — specifically apply_edits."""

import pytest

from resume_refinery.utils import EditApplicationError, apply_edits


def test_apply_edits_basic():
    """Single find/replace edit should work."""
    doc = "I am a passionate innovator."
    edits = [{"find": "passionate innovator", "replace": "backend engineer"}]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "I am a backend engineer."


def test_apply_edits_multiple():
    """Multiple non-overlapping edits should all be applied."""
    doc = "AAA and BBB and CCC."
    edits = [
        {"find": "AAA", "replace": "111"},
        {"find": "BBB", "replace": "222"},
        {"find": "CCC", "replace": "333"},
    ]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "111 and 222 and 333."


def test_apply_edits_reverse_order():
    """Edits should be applied in reverse document order so offsets don't drift."""
    doc = "short and a_longer_word here"
    edits = [
        {"find": "short", "replace": "extremely_long_replacement"},
        {"find": "a_longer_word", "replace": "x"},
    ]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "extremely_long_replacement and x here"


def test_apply_edits_no_match_under_threshold():
    """Edits that don't match should be skipped when under threshold."""
    doc = "Hello world."
    edits = [
        {"find": "Hello", "replace": "Hi"},
        {"find": "MISSING", "replace": "X"},
    ]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=1)
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
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=1)
    assert result == "Hello world."


def test_apply_edits_empty_list():
    """An empty edit list should return the document unchanged."""
    doc = "No changes."
    result, regions, _failed = apply_edits(doc, [])
    assert result == "No changes."
    assert regions == []


def test_apply_edits_deletion():
    """Replacing with empty string should delete the matched text."""
    doc = "Remove this_phrase from the text."
    edits = [{"find": "this_phrase ", "replace": ""}]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "Remove from the text."
    # Deletion (empty replace) should NOT produce a region
    assert regions == []


def test_apply_edits_contained_overlap():
    """When one edit's find range is inside another's, the contained edit is a collision."""
    doc = "I am a quick brown fox jumping."
    edits = [
        {"find": "quick brown fox", "replace": "fast red dog"},
        {"find": "brown", "replace": "crimson"},
    ]
    # "brown" at pos 13 is inside "quick brown fox" at pos 8 (end 23).
    # They form a collision cluster — first edit kept, second skipped.
    # Collision is NOT a failure, so fail_threshold=0 is fine.
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "I am a fast red dog jumping."


def test_apply_edits_partial_overlap():
    """Partially overlapping edits: the later-starting one is a collision."""
    doc = "AABBCCDD"
    edits = [
        {"find": "AABB", "replace": "XX"},
        {"find": "BBCC", "replace": "YY"},
    ]
    # "AABB" at 0 (end 4), "BBCC" at 2 (end 6) — overlap at positions 2-3.
    # Collision — first kept, second skipped. Not a failure.
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
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
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "HI world BYE"


def test_apply_edits_adjacent_non_overlapping():
    """Adjacent edits (end-of-A == start-of-B) should both be applied."""
    doc = "AAABBB"
    edits = [
        {"find": "AAA", "replace": "XX"},
        {"find": "BBB", "replace": "YY"},
    ]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "XXYY"


def test_apply_edits_overlap_not_counted_as_failure():
    """Dropped overlapping edits are collisions, not failures — they don't count toward threshold."""
    doc = "one two three"
    edits = [
        {"find": "one two", "replace": "X"},
        {"find": "two", "replace": "Y"},  # overlapping, will be merged/skipped as collision
        {"find": "MISSING", "replace": "Z"},  # not found — genuine failure
    ]
    # Only 1 failure (MISSING), collision is not a failure.
    # With threshold 1, this should NOT raise.
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=1)
    assert result == "X three"


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
    result, regions, _failed = apply_edits(
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
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
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


# ------------------------------------------------------------------
# Whitespace-normalized fallback tests
# ------------------------------------------------------------------


def test_apply_edits_whitespace_normalized_fallback():
    """LLM returns find text with different whitespace — should match via fallback."""
    doc = "Hello   world\nfoo bar."
    edits = [{"find": "Hello world foo", "replace": "Greetings universe foo"}]
    # Exact match fails, but whitespace-normalized match should succeed.
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    # The actual matched span is "Hello   world\nfoo" which gets replaced.
    assert "Greetings universe foo" in result
    assert result.endswith(" bar.")


def test_apply_edits_whitespace_normalized_newline():
    """Find text with space where document has newline."""
    doc = "line one\nline two"
    edits = [{"find": "line one line two", "replace": "combined"}]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "combined"


def test_apply_edits_whitespace_normalized_extra_spaces():
    """Find text with single space where document has multiple spaces."""
    doc = "a    b    c"
    edits = [{"find": "a b c", "replace": "x y z"}]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0)
    assert result == "x y z"


# ------------------------------------------------------------------
# Merge callback tests
# ------------------------------------------------------------------


def test_apply_edits_merge_fn_called_for_overlap():
    """When edits overlap and merge_fn is provided, it should be called."""
    doc = "The quick brown fox jumps over"
    edits = [
        {"find": "quick brown", "replace": "fast red"},
        {"find": "brown fox", "replace": "crimson dog"},
    ]
    # These overlap at "brown". merge_fn should receive the union span.
    merge_calls = []

    def fake_merge(context_text, overlapping_edits):
        merge_calls.append((context_text, overlapping_edits))
        return {"find": context_text, "replace": "speedy scarlet hound", "reason": "merged"}

    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0, merge_fn=fake_merge)
    assert len(merge_calls) == 1
    assert merge_calls[0][0] == "quick brown fox"  # union of the two spans
    assert len(merge_calls[0][1]) == 2
    assert result == "The speedy scarlet hound jumps over"


def test_apply_edits_merge_fn_returns_none_uses_first():
    """When merge_fn returns None, the first edit in the cluster is used."""
    doc = "AABBCC"
    edits = [
        {"find": "AABB", "replace": "XX"},
        {"find": "BBCC", "replace": "YY"},
    ]

    def failing_merge(context_text, overlapping_edits):
        return None

    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0, merge_fn=failing_merge)
    assert result == "XXCC"


def test_apply_edits_merge_fn_not_called_for_singletons():
    """merge_fn should NOT be called for non-overlapping edits."""
    doc = "AAA BBB CCC"
    edits = [
        {"find": "AAA", "replace": "111"},
        {"find": "CCC", "replace": "333"},
    ]
    merge_calls = []

    def spy_merge(context_text, overlapping_edits):
        merge_calls.append(True)
        return None

    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0, merge_fn=spy_merge)
    assert result == "111 BBB 333"
    assert merge_calls == []


# ------------------------------------------------------------------
# Offset-based application tests
# ------------------------------------------------------------------


def test_apply_edits_offset_tracking_correct_regions():
    """Offset tracking should produce correct region spans after variable-length replacements."""
    doc = "AAA BBB CCC DDD"
    edits = [
        {"find": "AAA", "replace": "XXXXX"},   # 3 → 5, delta +2
        {"find": "CCC", "replace": "Y"},       # 3 → 1, delta -2
    ]
    result, regions, _failed = apply_edits(doc, edits, fail_threshold=0, reviewer="voice", pass_num=1)
    assert result == "XXXXX BBB Y DDD"
    assert len(regions) == 2
    # First: "XXXXX" at 0-5
    assert regions[0].start == 0
    assert regions[0].end == 5
    # Second: "Y" — offset shifted by +2 from first edit.
    # Original CCC at pos 8, adjusted to 8+2=10. Length 1, so end=11.
    assert regions[1].start == 10
    assert regions[1].end == 11
