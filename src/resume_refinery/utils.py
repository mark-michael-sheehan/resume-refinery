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

    Edits are applied in reverse document order so that earlier replacements
    don't shift the positions of later ones.  If the number of edits that
    fail to match exceeds *fail_threshold*, an ``EditApplicationError`` is
    raised.
    """
    threshold = fail_threshold if fail_threshold is not None else _EDIT_FAIL_THRESHOLD

    # Determine application order: sort by position in document (reverse)
    # so offset drift doesn't affect later edits.
    positioned: list[tuple[int, EditOp]] = []
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
        positioned.append((idx, edit))

    # Check threshold before applying anything
    if len(failed) > threshold:
        raise EditApplicationError(failed, threshold)

    # Apply in reverse order so earlier edits don't shift later positions
    positioned.sort(key=lambda t: t[0], reverse=True)

    for idx, edit in positioned:
        find_text = edit["find"]
        replace_text = edit.get("replace", "")
        document = document[:idx] + replace_text + document[idx + len(find_text):]

    # Log summary
    applied_count = len(positioned)
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
