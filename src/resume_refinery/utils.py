"""Miscellaneous utilities."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, TypedDict

from dotenv import load_dotenv

from .models import EditRegion, ReviewerPriority

load_dotenv()

_EDIT_FAIL_THRESHOLD = int(
    os.environ.get("RESUME_REFINERY_EDIT_FAIL_THRESHOLD", "3")
)

log = logging.getLogger(__name__)


class EditOp(TypedDict, total=False):
    find: str
    replace: str
    reason: str
    insert_after: bool


class EditApplicationError(Exception):
    """Raised when too many edits fail to match the document."""

    def __init__(self, failed: list[EditOp], threshold: int) -> None:
        self.failed = failed
        self.threshold = threshold
        super().__init__(
            f"{len(failed)} edit(s) failed to match (threshold={threshold}): "
            + "; ".join(e.get("find", "")[:60] for e in failed)
        )


# ------------------------------------------------------------------
# Whitespace-normalized matching
# ------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single space and strip."""
    return _WS_RUN.sub(" ", text).strip()


def _find_normalized(document: str, find_text: str) -> int:
    """Find *find_text* in *document* using whitespace-normalized matching.

    Returns the start index in the **original** (non-normalized) document,
    or -1 if not found.
    """
    norm_find = _normalize_ws(find_text)
    if not norm_find:
        return -1

    # Build a mapping from normalized-string positions back to original
    # document positions.  We walk the original document character by
    # character, building the normalized version and recording where each
    # normalized character came from.
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []  # norm_to_orig[i] = original index of norm_chars[i]
    prev_was_space = True  # treat start of string as preceded by space (strip leading)
    for i, ch in enumerate(document):
        if ch in (" ", "\t", "\n", "\r", "\f", "\v"):
            if not prev_was_space:
                norm_chars.append(" ")
                norm_to_orig.append(i)
            prev_was_space = True
        else:
            norm_chars.append(ch)
            norm_to_orig.append(i)
            prev_was_space = False

    # Strip trailing space from normalized
    if norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        norm_to_orig.pop()

    norm_doc = "".join(norm_chars)
    idx = norm_doc.find(norm_find)
    if idx == -1:
        return -1

    return norm_to_orig[idx]


def _find_end_normalized(document: str, find_text: str, start: int) -> int:
    """Return the end index in *document* of the span matching *find_text* starting at *start*.

    Uses whitespace-normalized matching: the span in the original document
    may contain different whitespace than *find_text* but matches after
    normalisation.
    """
    norm_find = _normalize_ws(find_text)
    doc_pos = start
    norm_pos = 0
    while norm_pos < len(norm_find) and doc_pos < len(document):
        # Skip leading whitespace runs in document when norm expects a single space
        if norm_find[norm_pos] == " ":
            # Advance past whitespace in document
            if document[doc_pos] not in (" ", "\t", "\n", "\r", "\f", "\v"):
                return -1  # mismatch
            while doc_pos < len(document) and document[doc_pos] in (" ", "\t", "\n", "\r", "\f", "\v"):
                doc_pos += 1
            norm_pos += 1
        else:
            if document[doc_pos] != norm_find[norm_pos]:
                return -1  # mismatch
            doc_pos += 1
            norm_pos += 1
    if norm_pos < len(norm_find):
        return -1  # ran out of document
    return doc_pos


# ------------------------------------------------------------------
# Collision clustering
# ------------------------------------------------------------------


class _LocatedEdit:
    """An edit with its located span in the original document."""

    __slots__ = ("start", "end", "edit")

    def __init__(self, start: int, end: int, edit: EditOp) -> None:
        self.start = start
        self.end = end
        self.edit = edit


def _cluster_overlapping(located: list[_LocatedEdit]) -> list[list[_LocatedEdit]]:
    """Group located edits into clusters of overlapping spans.

    Input must be sorted by start position.  Non-overlapping edits
    become singleton clusters.  Overlapping or contained edits are
    grouped together.
    """
    if not located:
        return []
    clusters: list[list[_LocatedEdit]] = [[located[0]]]
    cluster_end = located[0].end
    for le in located[1:]:
        if le.start < cluster_end:
            # Overlaps the current cluster
            clusters[-1].append(le)
            cluster_end = max(cluster_end, le.end)
        else:
            clusters.append([le])
            cluster_end = le.end
    return clusters


# ------------------------------------------------------------------
# Type for the merge callback
# ------------------------------------------------------------------

MergeFn = Callable[[str, list[EditOp]], EditOp | None]
"""Signature: merge_fn(context_text, overlapping_edits) -> merged EditOp or None.

``context_text`` is the span of the original document that covers all
of the overlapping edits.  ``overlapping_edits`` is the list of edits
(2+) whose find ranges overlap.  The callback should return a single
merged EditOp whose ``find`` equals ``context_text`` and whose ``replace``
satisfies all the overlapping edits' intents, or ``None`` on failure.
"""


# ------------------------------------------------------------------
# apply_edits — main entry point
# ------------------------------------------------------------------


def apply_edits(
    document: str,
    edits: list[EditOp],
    *,
    fail_threshold: int | None = None,
    reviewer: ReviewerPriority = "truthfulness",
    pass_num: int = 0,
    merge_fn: MergeFn | None = None,
) -> tuple[str, list[EditRegion], list[EditOp]]:
    """Apply surgical find/replace edits to *document*.

    **Phase 1 — Locate:** Each edit's ``find`` text is located in the
    original document (exact match first, whitespace-normalized fallback
    second).  Edits that cannot be located at all are counted as failures.

    **Phase 2 — Cluster:** Located edits are sorted left-to-right and
    grouped into clusters of overlapping spans.  Non-overlapping edits
    become singleton clusters.

    **Phase 3 — Merge collisions:** For clusters with 2+ edits, the
    *merge_fn* callback is called to produce a single merged edit that
    satisfies all overlapping edits' intents.  If no *merge_fn* is
    provided (or it returns ``None``), only the first (leftmost) edit
    in the cluster is kept and the rest are counted as collisions (not
    failures).

    **Phase 4 — Apply with offset tracking:** Resolved edits are applied
    left-to-right using offset tracking — no re-find step, so earlier
    edits cannot invalidate later ones.

    Returns ``(modified_document, edit_regions, failed_edits)``.
    """
    threshold = fail_threshold if fail_threshold is not None else _EDIT_FAIL_THRESHOLD

    # ------------------------------------------------------------------
    # Phase 1 — locate every edit in the original document.
    # ------------------------------------------------------------------
    located: list[_LocatedEdit] = []
    failed: list[EditOp] = []

    for edit in edits:
        find_text = edit.get("find", "")
        is_insert = bool(edit.get("insert_after", False))
        if not find_text:
            log.warning("Skipping edit with empty 'find': %s", edit)
            failed.append(edit)
            continue

        # Try exact match first.
        idx = document.find(find_text)
        if idx != -1:
            if is_insert:
                # insert_after: anchor is preserved; span is zero-width right after it.
                anchor_end = idx + len(find_text)
                located.append(_LocatedEdit(anchor_end, anchor_end, edit))
            else:
                located.append(_LocatedEdit(idx, idx + len(find_text), edit))
            continue

        # Fallback: whitespace-normalized match.
        idx = _find_normalized(document, find_text)
        if idx != -1:
            end = _find_end_normalized(document, find_text, idx)
            if end != -1:
                log.info(
                    "Edit matched via whitespace-normalized fallback: %.80s",
                    find_text,
                )
                # Rewrite the edit's 'find' to the actual text in the document
                # so downstream consumers see the real text.
                edit = dict(edit)  # shallow copy to avoid mutating caller's dict
                edit["find"] = document[idx:end]
                if is_insert:
                    located.append(_LocatedEdit(end, end, edit))
                else:
                    located.append(_LocatedEdit(idx, end, edit))
                continue

        log.warning("Edit find text not found in document: %.80s", find_text)
        failed.append(edit)

    # Handle duplicate find texts: when two edits map to the same start
    # position, assign the second to the next occurrence of its find text.
    seen_starts: dict[int, int] = {}  # start -> count
    relocated: list[_LocatedEdit] = []
    for le in located:
        count = seen_starts.get(le.start, 0)
        if count == 0:
            seen_starts[le.start] = 1
            relocated.append(le)
        else:
            # Find the (count+1)-th occurrence
            find_text = le.edit.get("find", "")
            idx = -1
            search_from = 0
            for _ in range(count + 1):
                idx = document.find(find_text, search_from)
                if idx == -1:
                    break
                search_from = idx + 1
            if idx != -1:
                seen_starts[le.start] = count + 1
                relocated.append(_LocatedEdit(idx, idx + len(find_text), le.edit))
            else:
                log.warning(
                    "Duplicate edit find text has no further occurrence: %.80s",
                    find_text,
                )
                failed.append(le.edit)
    located = relocated

    # Sort by start position for left-to-right processing.
    located.sort(key=lambda le: le.start)

    # ------------------------------------------------------------------
    # Phase 2 — cluster overlapping spans.
    # ------------------------------------------------------------------
    clusters = _cluster_overlapping(located)

    # ------------------------------------------------------------------
    # Phase 3 — resolve each cluster to a single edit.
    # ------------------------------------------------------------------
    resolved: list[_LocatedEdit] = []
    collision_count = 0

    for cluster in clusters:
        if len(cluster) == 1:
            resolved.append(cluster[0])
            continue

        # Multiple edits overlap — try merge_fn.
        union_start = min(le.start for le in cluster)
        union_end = max(le.end for le in cluster)
        context_text = document[union_start:union_end]
        cluster_edits = [le.edit for le in cluster]

        merged: EditOp | None = None
        if merge_fn is not None:
            try:
                merged = merge_fn(context_text, cluster_edits)
            except Exception as exc:
                log.warning("merge_fn raised %s for cluster at %d-%d; falling back to first edit",
                            exc, union_start, union_end)

        if merged is not None:
            log.info(
                "Merged %d overlapping edits at %d-%d into one",
                len(cluster), union_start, union_end,
            )
            resolved.append(_LocatedEdit(union_start, union_end, merged))
        else:
            # No merge — keep the first (leftmost) edit, skip the rest.
            log.warning(
                "Collision: %d overlapping edits at %d-%d; keeping first, skipping %d",
                len(cluster), union_start, union_end, len(cluster) - 1,
            )
            resolved.append(cluster[0])
            collision_count += len(cluster) - 1

    # ------------------------------------------------------------------
    # Phase 4 — apply resolved edits with offset tracking.
    # ------------------------------------------------------------------
    applied_count = 0
    regions: list[EditRegion] = []
    offset = 0  # accumulated length delta from prior edits

    for le in resolved:
        find_text = le.edit.get("find", le.edit.get("find", ""))
        replace_text = le.edit.get("replace", "")
        is_insert = bool(le.edit.get("insert_after", False))
        adjusted_start = le.start + offset
        adjusted_end = le.end + offset

        # Sanity check: the text at the adjusted position should match.
        # For insert_after edits, start == end (zero-width); nothing to verify.
        if not is_insert:
            actual = document[adjusted_start:adjusted_end]
            if actual != find_text:
                log.warning(
                    "Offset-based apply mismatch at %d: expected %r, got %r",
                    adjusted_start, find_text[:60], actual[:60],
                )
                failed.append(le.edit)
                continue

        document = document[:adjusted_start] + replace_text + document[adjusted_end:]
        delta = len(replace_text) - (le.end - le.start)
        offset += delta

        if replace_text:
            regions.append(EditRegion(
                start=adjusted_start,
                end=adjusted_start + len(replace_text),
                reviewer=reviewer,
                pass_num=pass_num,
            ))
        applied_count += 1

    # ------------------------------------------------------------------
    # Threshold check.
    # ------------------------------------------------------------------
    if len(failed) > threshold:
        raise EditApplicationError(failed, threshold)
    if failed:
        log.warning(
            "Applied %d edit(s), %d failed to match, %d collision(s) resolved (threshold=%d)",
            applied_count, len(failed), collision_count, threshold,
        )
    else:
        log.info(
            "Applied %d edit(s), 0 failures, %d collision(s) resolved",
            applied_count, collision_count,
        )

    return document, regions, list(failed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
