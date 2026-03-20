"""Parse user-provided markdown files into model objects.

Parsing is intentionally lenient — the raw content is always preserved and
passed to Claude directly. Structural extraction is only used for session
naming and display.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import CareerProfile, JobDescription, VoiceProfile


def load_voice_profile(path: str | Path) -> VoiceProfile:
    """Read a voice profile markdown file."""
    content = Path(path).read_text(encoding="utf-8")
    return parse_voice_profile_content(content)


def load_career_profile(path: str | Path) -> CareerProfile:
    """Read a career profile markdown file and extract basic contact info."""
    content = Path(path).read_text(encoding="utf-8")

    return parse_career_profile_content(content)


def parse_voice_profile_content(content: str) -> VoiceProfile:
    """Build a VoiceProfile from raw file content."""
    return VoiceProfile(raw_content=content)


def parse_career_profile_content(content: str) -> CareerProfile:
    """Build a CareerProfile from raw file content."""

    name = _extract_name(content)
    email = _extract_email(content)
    phone = _extract_phone(content)
    location = _extract_location(content)

    return CareerProfile(
        raw_content=content,
        name=name,
        email=email,
        phone=phone,
        location=location,
    )


def load_job_description(path: str | Path) -> JobDescription:
    """Read a job description markdown/text file and extract title + company."""
    content = Path(path).read_text(encoding="utf-8")

    return parse_job_description_content(content)


def parse_job_description_content(content: str) -> JobDescription:
    """Build a JobDescription from raw file content."""

    title = _extract_job_title(content)
    company = _extract_company(content)

    return JobDescription(raw_content=content, title=title, company=company)


# ---------------------------------------------------------------------------
# Simple extraction helpers (regex-based, best-effort)
# ---------------------------------------------------------------------------


def _extract_name(text: str) -> str | None:
    """Look for a top-level H1 heading as the user's name."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        candidate = m.group(1).strip()
        # Sanity: likely a name if it's 2-5 words with no special characters
        if re.match(r"^[A-Za-z\s\-'.]{2,50}$", candidate) and len(candidate.split()) <= 5:
            return candidate
    return None


def _extract_email(text: str) -> str | None:
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> str | None:
    m = re.search(r"(\+?[\d][\d\s\-().]{7,15}\d)", text)
    return m.group(1).strip() if m else None


def _extract_location(text: str) -> str | None:
    """Look for common location patterns (City, ST or City, Country)."""
    m = re.search(
        r"\b([A-Z][a-zA-Z\s]+,\s*(?:[A-Z]{2}|[A-Za-z]+))\b",
        text,
    )
    return m.group(1).strip() if m else None


def _extract_job_title(text: str) -> str | None:
    """Look for 'Title:' label or top-level H1/H2."""
    m = re.search(r"(?:title|role|position)\s*[:–-]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^#{1,2}\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def _extract_company(text: str) -> str | None:
    m = re.search(r"(?:company|employer|organization)\s*[:–-]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None
