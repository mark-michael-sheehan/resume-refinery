"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from resume_refinery.models import (
    AIDetectionResult,
    CareerProfile,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    EvidencePack,
    JobDescription,
    OrchestrationResult,
    ReviewBundle,
    Session,
    TruthfulnessResult,
    VersionInfo,
    VoiceProfile,
    VoiceReviewResult,
    VoiceStyleGuide,
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


def test_voice_review_per_doc_match_defaults():
    """Per-doc match fields default to 'moderate' when not explicitly set."""
    vr = VoiceReviewResult(
        overall_match="strong",
        cover_letter_assessment="Good",
        resume_assessment="Good",
        interview_guide_assessment="Good",
    )
    assert vr.cover_letter_match == "moderate"
    assert vr.resume_match == "moderate"
    assert vr.interview_guide_match == "moderate"


def test_voice_review_per_doc_match_explicit():
    """Per-doc match fields can be set explicitly."""
    vr = VoiceReviewResult(
        overall_match="weak",
        cover_letter_match="strong",
        resume_match="weak",
        interview_guide_match="moderate",
        cover_letter_assessment="On-voice",
        resume_assessment="Off-voice",
        interview_guide_assessment="Okay",
    )
    assert vr.cover_letter_match == "strong"
    assert vr.resume_match == "weak"
    assert vr.interview_guide_match == "moderate"


def test_voice_review_per_doc_match_invalid_literal():
    with pytest.raises(ValidationError):
        VoiceReviewResult(
            overall_match="strong",
            cover_letter_match="excellent",  # invalid
            cover_letter_assessment="",
            resume_assessment="",
            interview_guide_assessment="",
        )


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


# ---------------------------------------------------------------------------
# DocumentSet extras
# ---------------------------------------------------------------------------


def test_document_set_overwrite():
    ds = DocumentSet(cover_letter="original")
    ds.set("cover_letter", "updated")
    assert ds.get("cover_letter") == "updated"


def test_document_set_empty_is_not_all_present():
    ds = DocumentSet()
    assert not ds.all_present()


# ---------------------------------------------------------------------------
# Model defaults and factory fields
# ---------------------------------------------------------------------------


def test_evidence_pack_defaults():
    pack = EvidencePack()
    assert pack.job_requirements == []
    assert pack.matched_evidence == []
    assert pack.gaps == []
    assert pack.source_summary == []


def test_voice_style_guide_defaults():
    guide = VoiceStyleGuide()
    assert guide.core_adjectives == []
    assert guide.style_rules == []
    assert guide.preferred_phrases == []
    assert guide.phrases_to_avoid == []
    assert guide.writing_samples == []


def test_drafting_context_requires_both_fields():
    pack = EvidencePack()
    guide = VoiceStyleGuide()
    ctx = DraftingContext(evidence_pack=pack, voice_style_guide=guide)
    assert ctx.evidence_pack is not None
    assert ctx.voice_style_guide is not None


# ---------------------------------------------------------------------------
# DocumentTruthResult / TruthfulnessResult
# ---------------------------------------------------------------------------


def test_document_truth_result_defaults():
    dtr = DocumentTruthResult(pass_strict=True)
    assert dtr.unsupported_claims == []
    assert dtr.evidence_examples == []


def test_truthfulness_result_all_supported():
    doc_pass = DocumentTruthResult(pass_strict=True)
    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
    tr = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_pass,
        resume=doc_fail,
        interview_guide=doc_pass,
    )
    assert tr.all_supported is False
    assert len(tr.resume.unsupported_claims) == 1


# ---------------------------------------------------------------------------
# AI detection invalid literal
# ---------------------------------------------------------------------------


def test_ai_detection_invalid_literal():
    with pytest.raises(ValidationError):
        AIDetectionResult(risk_level="extreme")


# ---------------------------------------------------------------------------
# Review bundle with all three populated
# ---------------------------------------------------------------------------


def test_review_bundle_all_populated():
    doc = DocumentTruthResult(pass_strict=True)
    rb = ReviewBundle(
        truthfulness=TruthfulnessResult(
            all_supported=True,
            cover_letter=doc, resume=doc, interview_guide=doc,
        ),
        voice=VoiceReviewResult(
            overall_match="strong",
            cover_letter_assessment="Good",
            resume_assessment="Good",
            interview_guide_assessment="Good",
        ),
        ai_detection=AIDetectionResult(risk_level="low"),
    )
    assert rb.truthfulness is not None
    assert rb.voice is not None
    assert rb.ai_detection is not None


# ---------------------------------------------------------------------------
# OrchestrationResult defaults
# ---------------------------------------------------------------------------


def test_orchestration_result_defaults(sample_session, document_set):
    result = OrchestrationResult(session=sample_session, documents=document_set)
    assert result.reviews.voice is None
    assert result.exported_paths == {}
    assert result.strict_truth_failed is False


# ---------------------------------------------------------------------------
# VersionInfo
# ---------------------------------------------------------------------------


def test_version_info_with_feedback():
    vi = VersionInfo(
        version=2,
        created_at="2026-03-20T11:00:00+00:00",
        feedback="Shorten the cover letter",
        docs_regenerated=["cover_letter"],
    )
    assert vi.feedback == "Shorten the cover letter"
    assert vi.docs_regenerated == ["cover_letter"]
    assert vi.has_reviews is False
