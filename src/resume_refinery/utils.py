"""Miscellaneous utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

_EDIT_FAIL_THRESHOLD = int(
    os.environ.get("RESUME_REFINERY_EDIT_FAIL_THRESHOLD", "3")
)

log = logging.getLogger(__name__)


class EditOp(TypedDict, total=False):
    find: str
    replace: str
    reason: str


class EditApplicationError(Exception):
    """Raised when too many edits fail to match the document."""

    def __init__(self, failed: list[EditOp], threshold: int) -> None:
        self.failed = failed
        self.threshold = threshold
        super().__init__(
            f"{len(failed)} edit(s) failed to match (threshold={threshold}): "
            + "; ".join(e.get("find", "")[:60] for e in failed)
        )


def apply_edits(
    document: str,
    edits: list[EditOp],
    *,
    fail_threshold: int | None = None,
) -> str:
    """Apply surgical find/replace edits to *document*.

    Edits are located in the original document to determine processing
    order (left-to-right), then applied **sequentially**: each edit
    re-finds its target in the *current* document state before applying.

    This handles overlapping edits safely — if a prior edit already
    transformed the region, the later edit simply won't match and is
    counted as a failure rather than silently corrupting the document.

    If the number of edits that fail to match exceeds *fail_threshold*,
    an ``EditApplicationError`` is raised.
    """
    threshold = fail_threshold if fail_threshold is not None else _EDIT_FAIL_THRESHOLD

    # Phase 1 — locate every edit in the *original* document to determine
    # left-to-right processing order.
    ordered: list[tuple[int, EditOp]] = []
    failed: list[EditOp] = []

    for edit in edits:
        find_text = edit.get("find", "")
        if not find_text:
            log.warning("Skipping edit with empty 'find': %s", edit)
            failed.append(edit)
            continue
        idx = document.find(find_text)
        if idx == -1:
            log.warning("Edit find text not found in document: %.80s", find_text)
            failed.append(edit)
            continue
        ordered.append((idx, edit))

    # Sort by original position so edits are processed left-to-right.
    ordered.sort(key=lambda t: t[0])

    # Phase 2 — apply each edit sequentially, re-finding the target in
    # the current document.  If a prior edit changed the region, the
    # find text won't match and the edit is counted as a failure.
    applied_count = 0
    for _orig_idx, edit in ordered:
        find_text = edit["find"]
        replace_text = edit.get("replace", "")
        idx = document.find(find_text)
        if idx == -1:
            log.warning(
                "Edit find text no longer present after prior edits: %.80s",
                find_text,
            )
            failed.append(edit)
            continue
        document = document[:idx] + replace_text + document[idx + len(find_text):]
        applied_count += 1

    # Check threshold after all edits have been attempted.
    if len(failed) > threshold:
        raise EditApplicationError(failed, threshold)
    if failed:
        log.warning(
            "Applied %d edit(s), %d failed to match (threshold=%d)",
            applied_count, len(failed), threshold,
        )
    else:
        log.info("Applied %d edit(s), 0 failures", applied_count)

    return document


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
