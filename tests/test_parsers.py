"""Tests for file parsers."""

import tempfile
from pathlib import Path

import pytest

from resume_refinery.parsers import (
    _extract_company,
    _extract_email,
    _extract_job_title,
    _extract_location,
    _extract_name,
    _extract_phone,
    load_career_profile,
    load_job_description,
    load_voice_profile,
    parse_career_profile_content,
    parse_job_description_content,
    parse_voice_profile_content,
)


def test_load_voice_profile(tmp_path):
    f = tmp_path / "voice.md"
    f.write_text("# Voice\n\nDirect. Precise.", encoding="utf-8")
    vp = load_voice_profile(f)
    assert "Direct. Precise." in vp.raw_content


def test_load_career_profile_extracts_name(tmp_path):
    f = tmp_path / "career.md"
    f.write_text("# Jordan Lee\njordan@example.com\n", encoding="utf-8")
    cp = load_career_profile(f)
    assert cp.name == "Jordan Lee"
    assert cp.email == "jordan@example.com"


def test_load_career_profile_no_name(tmp_path):
    f = tmp_path / "career.md"
    f.write_text("## Work Experience\nBuilt things.", encoding="utf-8")
    cp = load_career_profile(f)
    assert cp.name is None


def test_extract_email():
    assert _extract_email("Contact: alice@example.com today") == "alice@example.com"
    assert _extract_email("no email here") is None


def test_extract_phone():
    assert _extract_phone("Call 415-555-1234 anytime") is not None


def test_extract_job_title_label():
    assert _extract_job_title("Title: Staff Engineer") == "Staff Engineer"


def test_extract_job_title_heading():
    assert _extract_job_title("# Senior Data Scientist") == "Senior Data Scientist"


def test_extract_company():
    assert _extract_company("Company: Acme Corp") == "Acme Corp"
    assert _extract_company("No company here") is None


def test_load_job_description(tmp_path):
    f = tmp_path / "job.md"
    f.write_text("Title: ML Engineer\nCompany: BigCo\n\nWe need a great engineer.", encoding="utf-8")
    jd = load_job_description(f)
    assert jd.title == "ML Engineer"
    assert jd.company == "BigCo"
    assert "great engineer" in jd.raw_content


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------


def test_extract_location():
    assert _extract_location("San Francisco, CA") == "San Francisco, CA"


def test_extract_location_none():
    assert _extract_location("no location info 12345") is None


# ---------------------------------------------------------------------------
# Phone edge case
# ---------------------------------------------------------------------------


def test_extract_phone_none():
    assert _extract_phone("no phone number here") is None


# ---------------------------------------------------------------------------
# Name extraction edge cases
# ---------------------------------------------------------------------------


def test_extract_name_non_name_heading():
    """A heading with special characters shouldn't be treated as a name."""
    assert _extract_name("# Features & Benefits (v2)") is None


def test_extract_name_too_long():
    assert _extract_name("# This Is A Very Long Name That Exceeds Five Words Definitely") is None


# ---------------------------------------------------------------------------
# Job title edge cases
# ---------------------------------------------------------------------------


def test_extract_job_title_no_match():
    assert _extract_job_title("Some plain text with no headings or labels.") is None


def test_extract_job_title_role_label():
    assert _extract_job_title("Role: Platform Engineer") == "Platform Engineer"


# ---------------------------------------------------------------------------
# Content-based parse functions (no file I/O)
# ---------------------------------------------------------------------------


def test_parse_voice_profile_content():
    vp = parse_voice_profile_content("Direct and concise writing style.")
    assert "Direct" in vp.raw_content


def test_parse_career_profile_content():
    cp = parse_career_profile_content("# Alice Smith\nalice@test.com\n415-555-0099\nNew York, NY")
    assert cp.name == "Alice Smith"
    assert cp.email == "alice@test.com"
    assert cp.phone is not None
    assert cp.location is not None


def test_parse_career_profile_content_empty():
    cp = parse_career_profile_content("Nothing useful here.")
    assert cp.name is None
    assert cp.email is None


def test_parse_job_description_content():
    jd = parse_job_description_content("# Data Engineer\nCompany: WidgetCo\n\nBuild data pipelines.")
    assert jd.title == "Data Engineer"
    assert jd.company == "WidgetCo"


# ---------------------------------------------------------------------------
# .docx loading
# ---------------------------------------------------------------------------


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    """Create a minimal .docx file with the given paragraph texts."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(str(path))


def test_load_voice_profile_docx(tmp_path):
    f = tmp_path / "voice.docx"
    _make_docx(f, ["Direct.", "Precise."])
    vp = load_voice_profile(f)
    assert "Direct." in vp.raw_content


def test_load_career_profile_docx(tmp_path):
    f = tmp_path / "career.docx"
    _make_docx(f, ["# Jordan Lee", "jordan@example.com", "Austin, TX"])
    cp = load_career_profile(f)
    assert cp.email == "jordan@example.com"
    assert "Jordan" in cp.raw_content


def test_load_job_description_docx(tmp_path):
    f = tmp_path / "job.docx"
    _make_docx(f, ["Title: ML Engineer", "Company: BigCo", "We need a great engineer."])
    jd = load_job_description(f)
    assert jd.title == "ML Engineer"
    assert jd.company == "BigCo"
    assert "great engineer" in jd.raw_content


def test_parse_job_description_content_no_fields():
    jd = parse_job_description_content("We need someone great with no structure.")
    assert jd.title is None
    assert jd.company is None
