"""Tests for file parsers."""

import tempfile
from pathlib import Path

import pytest

from resume_refinery.parsers import (
    _extract_company,
    _extract_email,
    _extract_job_title,
    _extract_name,
    _extract_phone,
    load_career_profile,
    load_job_description,
    load_voice_profile,
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
