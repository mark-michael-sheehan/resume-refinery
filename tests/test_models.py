"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from resume_refinery.models import (
    AIDetectionResult,
    CandidacyNarrative,
    CareerProfile,
    CoverageGap,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    JobDescription,
    NarrativeCoherenceIssue,
    NarrativeCoherenceResult,
    NarrativeCoverageResult,
    NarrativePillar,
    OrchestrationResult,
    ReviewBundle,
    Session,
    TruthfulnessResult,
    VersionInfo,
    VoiceData,
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
    assert ds.get("resume") is None
    ds.set("resume", "Hello world")
    assert ds.get("resume") == "Hello world"


def test_document_set_all_present(document_set):
    assert document_set.all_present()


def test_document_set_not_all_present():
    ds = DocumentSet()
    assert not ds.all_present()


def test_voice_review_valid_literals(voice_review):
    assert voice_review.overall_match in ("strong", "moderate", "weak")


def test_voice_review_per_doc_match_defaults():
    """Per-doc match fields default to 'moderate' when not explicitly set."""
    vr = VoiceReviewResult(
        overall_match="strong",
        resume_assessment="Good",
    )
    assert vr.resume_match == "moderate"


def test_voice_review_per_doc_match_explicit():
    """Per-doc match fields can be set explicitly."""
    vr = VoiceReviewResult(
        overall_match="weak",
        resume_match="weak",
        resume_assessment="Off-voice",
    )
    assert vr.resume_match == "weak"


def test_voice_review_per_doc_issues_defaults():
    """Per-doc issues default to empty lists."""
    vr = VoiceReviewResult(
        overall_match="strong",
        resume_assessment="Good",
    )
    assert vr.resume_issues == []


def test_voice_review_per_doc_issues_explicit():
    """Per-doc issues can be set explicitly."""
    vr = VoiceReviewResult(
        overall_match="weak",
        resume_assessment="Good",
        resume_issues=["too formal"],
    )
    assert vr.resume_issues == ["too formal"]


def test_voice_review_per_doc_match_invalid_literal():
    with pytest.raises(ValidationError):
        VoiceReviewResult(
            overall_match="strong",
            resume_match="excellent",  # invalid
            resume_assessment="",
        )


def test_voice_review_invalid_literal():
    with pytest.raises(ValidationError):
        VoiceReviewResult(
            overall_match="excellent",  # not a valid literal
            resume_assessment="",
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
    ds = DocumentSet(resume="original")
    ds.set("resume", "updated")
    assert ds.get("resume") == "updated"


def test_document_set_empty_is_not_all_present():
    ds = DocumentSet()
    assert not ds.all_present()


# ---------------------------------------------------------------------------
# Model defaults and factory fields
# ---------------------------------------------------------------------------


def test_narrative_defaults():
    narrative = CandidacyNarrative(thesis="", pillars=[], gap_framing=[], raw_narrative="")
    assert narrative.thesis == ""
    assert narrative.pillars == []
    assert narrative.gap_framing == []
    assert narrative.raw_narrative == ""


def test_voice_style_guide_defaults():
    guide = VoiceStyleGuide()
    assert guide.core_adjectives == []
    assert guide.style_rules == []
    assert guide.preferred_phrases == []
    assert guide.phrases_to_avoid == []
    assert guide.writing_samples == []


def test_voice_data_defaults():
    v = VoiceData()
    assert v.core_adjectives == []
    assert v.has_content() is False


def test_voice_data_has_content():
    v = VoiceData(core_adjectives=["Direct", "analytical"])
    assert v.has_content() is True


def test_voice_data_to_markdown():
    v = VoiceData(
        core_adjectives=["Direct", "analytical"],
        style_notes=["Short sentences"],
        preferred_phrases=["The key insight was..."],
        avoid_phrases=["Passionate about"],
        writing_samples=["I start simple."],
    )
    md = v.to_markdown(name="Test")
    assert "## Core Adjectives" in md
    assert "- Direct" in md
    assert "- analytical" in md
    assert "## Style Notes" in md
    assert "## Phrases I Actually Use" in md
    assert "## Phrases to Avoid" in md
    assert "## Writing Sample 1" in md
    assert "I start simple." in md


def test_voice_data_from_markdown():
    raw = (
        "# Voice Profile\n\n"
        "## Core Adjectives\n- Direct\n- Analytical\n\n"
        "## Style Notes\n- Short sentences\n\n"
        '## Phrases I Actually Use\n- "The key insight was..."\n\n'
        '## Phrases to Avoid\n- "Passionate about"\n\n'
        "## Writing Sample 1\nI start simple.\n"
    )
    v = VoiceData.from_markdown(raw)
    assert v.core_adjectives == ["Direct", "Analytical"]
    assert v.style_notes == ["Short sentences"]
    assert v.preferred_phrases == ["The key insight was..."]
    assert v.avoid_phrases == ["Passionate about"]
    assert v.writing_samples == ["I start simple."]


def test_voice_data_from_markdown_empty():
    v = VoiceData.from_markdown("")
    assert v.has_content() is False


def test_drafting_context_requires_both_fields():
    narrative = CandidacyNarrative(
        thesis="Strong fit",
        pillars=[NarrativePillar(theme="Backend", argument="Led migrations", career_evidence=["Cut costs"])],
        gap_framing=[],
        raw_narrative="Narrative text.",
    )
    guide = VoiceStyleGuide()
    ctx = DraftingContext(narrative=narrative, voice_style_guide=guide)
    assert ctx.narrative is not None
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
        resume=doc_fail,
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
            resume=doc,
        ),
        voice=VoiceReviewResult(
            overall_match="strong",
            resume_assessment="Good",
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
        feedback="Shorten the resume",
        docs_regenerated=["resume"],
    )
    assert vi.feedback == "Shorten the resume"
    assert vi.docs_regenerated == ["resume"]
    assert vi.has_reviews is False


# ---------------------------------------------------------------------------
# NarrativeCoherenceResult
# ---------------------------------------------------------------------------


def test_narrative_coherence_result_defaults():
    nc = NarrativeCoherenceResult(alignment="strong")
    assert nc.alignment == "strong"
    assert nc.resume_issues == []


def test_narrative_coherence_result_with_issues():
    issue = NarrativeCoherenceIssue(
        phrase="Proficient in Kubernetes orchestration",
        issue="Not connected to any pillar",
        suggestion="Tie to backend migration pillar",
        severity="medium",
    )
    nc = NarrativeCoherenceResult(alignment="moderate", resume_issues=[issue])
    assert nc.alignment == "moderate"
    assert len(nc.resume_issues) == 1
    assert nc.resume_issues[0].phrase == "Proficient in Kubernetes orchestration"


def test_narrative_coherence_issue_defaults():
    issue = NarrativeCoherenceIssue(
        phrase="Led a team",
        issue="Disconnected from thesis",
        severity="high",
    )
    assert issue.suggestion == ""


def test_review_bundle_includes_narrative_coherence():
    nc = NarrativeCoherenceResult(alignment="weak", resume_issues=[])
    bundle = ReviewBundle(narrative_coherence=nc)
    assert bundle.narrative_coherence is not None
    assert bundle.narrative_coherence.alignment == "weak"


def test_review_bundle_narrative_coherence_defaults_none():
    bundle = ReviewBundle()
    assert bundle.narrative_coherence is None


# ---------------------------------------------------------------------------
# Narrative coverage models
# ---------------------------------------------------------------------------


def test_coverage_gap_defaults():
    gap = CoverageGap(pillar_theme="Backend")
    assert gap.pillar_theme == "Backend"
    assert gap.career_evidence == []
    assert gap.suggested_content == ""
    assert gap.anchor_section == ""
    assert gap.confidence == "medium"


def test_coverage_gap_full():
    gap = CoverageGap(
        pillar_theme="Cost Optimisation",
        career_evidence=["$180K savings"],
        suggested_content="- Reduced costs by $180K",
        anchor_section="Experience",
        confidence="high",
    )
    assert gap.confidence == "high"
    assert len(gap.career_evidence) == 1


def test_narrative_coverage_result_defaults():
    result = NarrativeCoverageResult()
    assert result.gaps == []
    assert result.coverage_summary == ""
    assert result.pillars_covered == 0
    assert result.pillars_total == 0


def test_narrative_coverage_result_with_gaps():
    result = NarrativeCoverageResult(
        gaps=[CoverageGap(pillar_theme="X")],
        coverage_summary="1 gap found.",
        pillars_covered=1,
        pillars_total=2,
    )
    assert len(result.gaps) == 1
    assert result.pillars_total == 2


def test_orchestration_result_includes_coverage():
    cov = NarrativeCoverageResult(pillars_covered=2, pillars_total=2)
    r = OrchestrationResult(
        session=Session(
            session_id="test",
            job_description=JobDescription(raw_content="job"),
            created_at="2026-01-01T00:00:00Z",
        ),
        documents=DocumentSet(),
        coverage_result=cov,
    )
    assert r.coverage_result is not None
    assert r.coverage_result.pillars_covered == 2


def test_orchestration_result_coverage_defaults_none():
    r = OrchestrationResult(
        session=Session(
            session_id="test",
            job_description=JobDescription(raw_content="job"),
            created_at="2026-01-01T00:00:00Z",
        ),
        documents=DocumentSet(),
    )
    assert r.coverage_result is None
