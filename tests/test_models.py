"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from resume_refinery.models import (
    AIDetectionResult,
    CareerProfile,
    DocumentSet,
    JobDescription,
    ReviewBundle,
    Session,
    VoiceProfile,
    VoiceReviewResult,
)


def test_voice_profile_stores_raw_content():
    vp = VoiceProfile(raw_content="Direct. Analytical. No fluff.")
    assert vp.raw_content == "Direct. Analytical. No fluff."


def test_career_profile_optional_extracted_fields():
    cp = CareerProfile(raw_content="Some career text.")
    assert cp.name is None
    assert cp.email is None


def test_career_profile_with_extracted_fields(career_profile):
    assert career_profile.name == "Jordan Lee"
    assert career_profile.email == "jordan@example.com"


def test_job_description_raw_content(job_description):
    assert "Staff Engineer" in job_description.raw_content
    assert job_description.title == "Staff Engineer, Platform"


def test_document_set_get_set():
    ds = DocumentSet()
    assert ds.get("cover_letter") is None
    ds.set("cover_letter", "Hello world")
    assert ds.get("cover_letter") == "Hello world"


def test_document_set_all_present(document_set):
    assert document_set.all_present()


def test_document_set_not_all_present():
    ds = DocumentSet(cover_letter="x", resume="y")
    assert not ds.all_present()


def test_voice_review_valid_literals(voice_review):
    assert voice_review.overall_match in ("strong", "moderate", "weak")


def test_voice_review_invalid_literal():
    with pytest.raises(ValidationError):
        VoiceReviewResult(
            overall_match="excellent",  # not a valid literal
            cover_letter_assessment="",
            resume_assessment="",
            interview_guide_assessment="",
        )


def test_ai_detection_valid_literals(ai_detection):
    assert ai_detection.risk_level in ("low", "medium", "high")


def test_review_bundle_optional_fields():
    rb = ReviewBundle()
    assert rb.voice is None
    assert rb.ai_detection is None


def test_session_structure(sample_session):
    assert sample_session.current_version == 1
    assert len(sample_session.versions) == 1
    assert sample_session.versions[0].has_reviews is False
