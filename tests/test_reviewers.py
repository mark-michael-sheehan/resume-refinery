"""Tests for review agents (mocked API)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    ATSKeywordResult,
    DocumentSet,
    GrammarResult,
    HiringManagerReview,
    NarrativeCoherenceResult,
    RelevancePruningResult,
    TruthfulnessResult,
    VoiceReviewResult,
)
from resume_refinery.reviewers import DocumentReviewer, _normalize_llm_json


def _make_mock_response(response_text: str):
    """Build an ollama chat response mock."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_response = MagicMock()
    mock_response.message = mock_message
    return mock_response


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_voice_returns_result(mock_client_cls, document_set, voice_profile):
    payload = json.dumps({
        "overall_match": "strong",
        "assessment": "Great voice match.",
        "issues": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert isinstance(result, VoiceReviewResult)
    assert result.overall_match == "strong"
    assert result.resume_assessment == "Great voice match."


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ai_detection_returns_result(mock_client_cls, document_set):
    payload = json.dumps({
        "risk_level": "low",
        "flags": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)

    assert isinstance(result, AIDetectionResult)
    assert result.risk_level == "low"


@patch("resume_refinery.reviewers.ollama.Client")
def test_reviewer_strips_json_fences(mock_client_cls, document_set):
    payload = json.dumps({
        "risk_level": "medium",
        "flags": ["test flag"],
    })
    fenced = f"```json\n{payload}\n```"
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(fenced)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)
    assert result.risk_level == "medium"
    assert "test flag" in result.resume_flags


# ---------------------------------------------------------------------------
# review_truthfulness
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_all_pass(mock_client_cls, document_set, career_profile, job_description):
    payload = json.dumps({
        "pass_strict": True,
        "unsupported_claims": [],
        "evidence_examples": ["Led backend migration"],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(document_set, career_profile, job_description)

    assert isinstance(result, TruthfulnessResult)
    assert result.all_supported is True
    assert result.resume.pass_strict is True


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_one_fails(mock_client_cls, document_set, career_profile, job_description):
    fail_payload = json.dumps({
        "pass_strict": False,
        "unsupported_claims": ["Led a team of 50"],
        "evidence_examples": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(fail_payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(document_set, career_profile, job_description)

    assert result.all_supported is False
    assert result.resume.pass_strict is False
    assert "Led a team of 50" in result.resume.unsupported_claims


# ---------------------------------------------------------------------------
# _normalize_llm_json
# ---------------------------------------------------------------------------


def test_normalize_already_valid_json():
    raw = '{"key": "value"}'
    assert _normalize_llm_json(raw) == raw


def test_normalize_trailing_comma():
    raw = '{"key": "value",}'
    result = _normalize_llm_json(raw)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_python_literals():
    raw = "{'pass_strict': True, 'value': 42}"
    result = _normalize_llm_json(raw)
    parsed = json.loads(result)
    assert parsed["pass_strict"] is True
    assert parsed["value"] == 42


def test_normalize_returns_raw_on_total_failure():
    """When nothing can parse it, return raw unchanged."""
    raw = "this is not json at all {{{]]}"
    result = _normalize_llm_json(raw)
    # Should not raise — just returns the raw string
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _call: empty response raises ValueError
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_call_empty_response_raises(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("")
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    with pytest.raises(ValueError, match="empty content"):
        reviewer._call("system prompt", "user message")


# ---------------------------------------------------------------------------
# _call: strips <think> blocks
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_call_strips_think_blocks(mock_client_cls):
    payload = json.dumps({"risk_level": "low", "flags": []})
    wrapped = f"<think>reasoning here</think>{payload}"
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(wrapped)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    raw = reviewer._call("system", "user")
    parsed = json.loads(raw)
    assert parsed["risk_level"] == "low"


# ---------------------------------------------------------------------------
# Voice review: worst-of aggregation
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_voice_review_returns_match_for_single_doc(mock_client_cls, document_set, voice_profile):
    """Voice review should return the match level for the resume."""
    weak = json.dumps({"overall_match": "weak", "assessment": "Poor", "issues": ["too formal"]})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(weak)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert result.overall_match == "weak"
    assert "too formal" in result.specific_issues


# ---------------------------------------------------------------------------
# AI detection: worst-of (max risk) aggregation
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_ai_detection_returns_flags(mock_client_cls, document_set):
    """AI detection should return flags from the resume."""
    high = json.dumps({"risk_level": "high", "flags": ["passionate about innovation"]})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(high)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)

    assert result.risk_level == "high"
    assert "passionate about innovation" in result.resume_flags


# ---------------------------------------------------------------------------
# Missing documents: skipped cleanly
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_voice_skips_missing_docs(mock_client_cls, voice_profile):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(docs, voice_profile)

    # No API call when resume is missing
    assert mock_client.chat.call_count == 0
    assert result.resume_assessment == "(not generated)"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_skips_missing_docs(mock_client_cls, career_profile, job_description):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(docs, career_profile, job_description)

    assert result.all_supported is True
    assert mock_client.chat.call_count == 0  # No docs to review
    assert result.resume.pass_strict is True  # Default for missing


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ai_detection_skips_missing_docs(mock_client_cls):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(docs)

    assert result.risk_level == "low"
    assert mock_client.chat.call_count == 0
    assert result.resume_flags == []


# ---------------------------------------------------------------------------
# Voice review: per-doc match levels are stored
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_voice_review_stores_per_doc_match(mock_client_cls, document_set, voice_profile):
    """Per-doc overall_match value from the LLM is stored in the result."""
    weak = json.dumps({"overall_match": "weak", "assessment": "Off-voice", "issues": []})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(weak)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert result.resume_match == "weak"
    assert result.overall_match == "weak"


# ---------------------------------------------------------------------------
# Same-model warning
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_same_model_warning_logged(mock_client_cls, caplog):
    """When review model matches generation model, a warning should be logged."""
    import logging
    import resume_refinery.reviewers as rev_mod

    original_model = rev_mod.MODEL
    original_gen_model = rev_mod._GEN_MODEL
    try:
        rev_mod.MODEL = "same-model"
        rev_mod._GEN_MODEL = "same-model"
        mock_client_cls.return_value = MagicMock()

        with caplog.at_level(logging.WARNING, logger="resume_refinery.reviewers"):
            reviewer = DocumentReviewer(api_key="test-key")

        assert any("same as generation model" in rec.message for rec in caplog.records)
    finally:
        rev_mod.MODEL = original_model
        rev_mod._GEN_MODEL = original_gen_model


@patch("resume_refinery.reviewers.ollama.Client")
def test_different_model_no_warning(mock_client_cls, caplog):
    """When review model differs from generation model, no warning should appear."""
    import logging
    import resume_refinery.reviewers as rev_mod

    original_model = rev_mod.MODEL
    original_gen_model = rev_mod._GEN_MODEL
    try:
        rev_mod.MODEL = "review-model"
        rev_mod._GEN_MODEL = "gen-model"
        mock_client_cls.return_value = MagicMock()

        with caplog.at_level(logging.WARNING, logger="resume_refinery.reviewers"):
            reviewer = DocumentReviewer(api_key="test-key")

        assert not any("same as generation model" in rec.message for rec in caplog.records)
    finally:
        rev_mod.MODEL = original_model
        rev_mod._GEN_MODEL = original_gen_model


# ---------------------------------------------------------------------------
# review_hiring_manager
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_hiring_manager_returns_result(mock_client_cls, document_set, job_description):
    payload = json.dumps({
        "advance_likelihood": 72,
        "summary": "Strong technical match with relevant distributed systems experience.",
        "strengths": ["Quantified achievements", "Relevant tech stack"],
        "concerns": ["No staff-level leadership evidence"],
        "improvements": [
            {
                "area": "resume",
                "suggestion": "Add a bullet highlighting cross-team influence.",
                "impact": "high",
            },
            {
                "area": "resume",
                "suggestion": "Open with the deploy-time reduction metric.",
                "impact": "medium",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_hiring_manager(document_set, job_description)

    assert isinstance(result, HiringManagerReview)
    assert result.advance_likelihood == 72
    assert "Strong technical match" in result.summary
    assert len(result.strengths) == 2
    assert len(result.concerns) == 1
    assert len(result.improvements) == 2
    assert result.improvements[0].area == "resume"
    assert result.improvements[0].impact == "high"
    assert result.improvements[1].area == "resume"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_hiring_manager_clamps_likelihood(mock_client_cls, document_set, job_description):
    """Likelihood values outside 0-100 are clamped."""
    payload = json.dumps({
        "advance_likelihood": 150,
        "summary": "Excellent.",
        "strengths": [],
        "concerns": [],
        "improvements": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_hiring_manager(document_set, job_description)

    assert result.advance_likelihood == 100


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_hiring_manager_invalid_area_defaults(mock_client_cls, document_set, job_description):
    """Invalid area values default to 'resume'."""
    payload = json.dumps({
        "advance_likelihood": 55,
        "summary": "Decent candidate.",
        "strengths": ["Good skills"],
        "concerns": [],
        "improvements": [
            {"area": "cover_letter", "suggestion": "Fix something", "impact": "low"},
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_hiring_manager(document_set, job_description)

    assert result.improvements[0].area == "resume"


# ---------------------------------------------------------------------------
# Voice review: per-doc issues
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_voice_review_stores_per_doc_issues(mock_client_cls, document_set, voice_profile):
    """Per-doc issues from the LLM are stored in the result."""
    resume_resp = json.dumps({
        "overall_match": "weak",
        "assessment": "Off-voice",
        "issues": ["opener too formal"],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(resume_resp)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    # Per-doc fields
    assert result.resume_issues == ["opener too formal"]
    # Aggregated fields
    assert "opener too formal" in result.specific_issues


# ---------------------------------------------------------------------------
# review_relevance_pruning
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_relevance_pruning_returns_result(mock_client_cls, document_set, job_description):
    resume_payload = json.dumps({
        "overall_density": "bloated",
        "removal_candidates": [
            {
                "phrase": "Senior Engineer",
                "reason": "Old role with no JD relevance",
                "category": "space_waste",
                "severity": "high",
            },
            {
                "phrase": "jordan@example.com",
                "reason": "Redundant contact line",
                "category": "redundant",
                "severity": "low",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(resume_payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_relevance_pruning(document_set, job_description)

    assert isinstance(result, RelevancePruningResult)
    assert result.overall_density == "bloated"
    assert len(result.resume_issues) == 2
    assert result.resume_issues[0].phrase == "Senior Engineer"
    assert result.resume_issues[0].severity == "high"
    assert result.resume_issues[1].category == "redundant"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_relevance_pruning_skips_missing_docs(mock_client_cls, job_description):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_relevance_pruning(docs, job_description)

    assert mock_client.chat.call_count == 0
    assert result.resume_issues == []


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_relevance_pruning_density(mock_client_cls, document_set, job_description):
    """Density should reflect the resume review result."""
    bloated = json.dumps({
        "overall_density": "bloated",
        "removal_candidates": [
            {"phrase": "some phrase", "reason": "filler", "category": "filler", "severity": "low"},
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(bloated)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_relevance_pruning(document_set, job_description)

    assert result.overall_density == "bloated"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_relevance_pruning_invalid_values_default(mock_client_cls, document_set, job_description):
    """Invalid category/severity/density values default to safe values."""
    payload = json.dumps({
        "overall_density": "unknown",
        "removal_candidates": [
            {
                "phrase": "some phrase",
                "reason": "filler",
                "category": "nonexistent_category",
                "severity": "critical",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_relevance_pruning(document_set, job_description)

    assert result.overall_density == "balanced"  # unknown → default
    assert result.resume_issues[0].category == "filler"  # nonexistent → default
    assert result.resume_issues[0].severity == "medium"  # critical → default


# ---------------------------------------------------------------------------
# review_ats_keyword
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ats_keyword_returns_result(mock_client_cls, document_set, job_description, career_profile):
    payload = json.dumps({
        "alignment_score": "moderate",
        "missing_keywords": [
            {
                "keyword": "distributed systems",
                "section": "Experience",
                "suggestion": "Add distributed systems to backend migration bullet",
                "priority": "high",
            },
        ],
        "stuffing_keywords": [
            {
                "keyword": "Python",
                "section": "Skills",
                "suggestion": "Reduce to one mention",
                "priority": "low",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ats_keyword(document_set, job_description, career_profile)

    assert isinstance(result, ATSKeywordResult)
    assert result.alignment_score == "moderate"
    assert len(result.missing_keywords) == 1
    assert result.missing_keywords[0].keyword == "distributed systems"
    assert result.missing_keywords[0].issue_type == "missing"
    assert result.missing_keywords[0].priority == "high"
    assert len(result.stuffing_keywords) == 1
    assert result.stuffing_keywords[0].keyword == "Python"
    assert result.stuffing_keywords[0].issue_type == "stuffing"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ats_keyword_skips_missing_resume(mock_client_cls, job_description, career_profile):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ats_keyword(docs, job_description, career_profile)

    assert result.alignment_score == "strong"
    assert result.missing_keywords == []
    assert result.stuffing_keywords == []
    assert mock_client.chat.call_count == 0


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ats_keyword_invalid_values_default(mock_client_cls, document_set, job_description, career_profile):
    """Invalid alignment_score/priority values default to safe values."""
    payload = json.dumps({
        "alignment_score": "excellent",
        "missing_keywords": [
            {"keyword": "k1", "priority": "critical"},
        ],
        "stuffing_keywords": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ats_keyword(document_set, job_description, career_profile)

    assert result.alignment_score == "moderate"  # excellent → default
    assert result.missing_keywords[0].priority == "medium"  # critical → default


# ---------------------------------------------------------------------------
# review_grammar
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_grammar_returns_result(mock_client_cls, document_set):
    resume_payload = json.dumps({
        "clean": False,
        "issues": [
            {
                "phrase": "I've spent five years building",
                "issue": "Contraction in formal letter",
                "suggestion": "I have spent five years building",
                "category": "grammar",
                "severity": "low",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(resume_payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_grammar(document_set)

    assert isinstance(result, GrammarResult)
    assert result.clean is False
    assert len(result.resume_issues) == 1
    assert result.resume_issues[0].phrase == "I've spent five years building"
    assert result.resume_issues[0].document == "resume"
    assert result.resume_issues[0].category == "grammar"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_grammar_skips_missing_docs(mock_client_cls):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_grammar(docs)

    assert result.clean is True
    assert mock_client.chat.call_count == 0
    assert result.resume_issues == []


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_grammar_invalid_values_default(mock_client_cls, document_set):
    """Invalid category/severity values default to safe values."""
    payload = json.dumps({
        "clean": False,
        "issues": [
            {
                "phrase": "some text",
                "issue": "bad grammar",
                "category": "syntax",
                "severity": "critical",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_grammar(document_set)

    assert result.resume_issues[0].category == "grammar"  # syntax → default
    assert result.resume_issues[0].severity == "medium"  # critical → default


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_grammar_all_clean(mock_client_cls, document_set):
    """When resume is clean, result should be clean."""
    payload = json.dumps({"clean": True, "issues": []})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_grammar(document_set)

    assert result.clean is True
    assert result.resume_issues == []
    assert mock_client.chat.call_count == 1


# ---------------------------------------------------------------------------
# Narrative coherence reviewer
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_returns_result(mock_client_cls, document_set, candidacy_narrative):
    payload = json.dumps({
        "alignment": "moderate",
        "issues": [
            {
                "phrase": "Proficient in Kubernetes orchestration",
                "issue": "Not connected to any narrative pillar",
                "suggestion": "Tie to backend migration pillar",
                "severity": "medium",
            },
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(document_set, candidacy_narrative)

    assert isinstance(result, NarrativeCoherenceResult)
    assert result.alignment == "moderate"
    assert len(result.resume_issues) == 1
    assert result.resume_issues[0].phrase == "Proficient in Kubernetes orchestration"
    assert result.resume_issues[0].severity == "medium"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_strong_no_issues(mock_client_cls, document_set, candidacy_narrative):
    payload = json.dumps({"alignment": "strong", "issues": []})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(document_set, candidacy_narrative)

    assert result.alignment == "strong"
    assert result.resume_issues == []


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_skips_missing_resume(mock_client_cls, candidacy_narrative):
    docs = DocumentSet(resume=None)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(docs, candidacy_narrative)

    assert result.alignment == "strong"
    assert result.resume_issues == []
    assert mock_client.chat.call_count == 0


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_skips_missing_thesis(mock_client_cls, document_set):
    from resume_refinery.models import CandidacyNarrative
    narrative = CandidacyNarrative(thesis="", pillars=[], gap_framing=[], raw_narrative="")
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(document_set, narrative)

    assert result.alignment == "strong"
    assert mock_client.chat.call_count == 0


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_invalid_alignment_defaults(mock_client_cls, document_set, candidacy_narrative):
    payload = json.dumps({"alignment": "perfect", "issues": []})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(document_set, candidacy_narrative)

    assert result.alignment == "moderate"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_narrative_coherence_invalid_severity_defaults(mock_client_cls, document_set, candidacy_narrative):
    payload = json.dumps({
        "alignment": "weak",
        "issues": [
            {"phrase": "Led team", "issue": "No pillar link", "severity": "critical"},
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_narrative_coherence(document_set, candidacy_narrative)

    assert result.resume_issues[0].severity == "medium"

