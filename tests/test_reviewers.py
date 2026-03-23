"""Tests for review agents (mocked API)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    DocumentSet,
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
    # Per-doc schema: one call per document (cover_letter, resume, interview_guide)
    payload = json.dumps({
        "overall_match": "strong",
        "assessment": "Great voice match.",
        "issues": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert isinstance(result, VoiceReviewResult)
    # All three per-doc calls return "strong" → overall is "strong"
    assert result.overall_match == "strong"
    assert result.cover_letter_assessment == "Great voice match."


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ai_detection_returns_result(mock_client_cls, document_set):
    # Per-doc schema: one call per document
    payload = json.dumps({
        "risk_level": "low",
        "flags": [],
        "suggestions": [],
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
    # Per-doc schema wrapped in markdown fences
    payload = json.dumps({
        "risk_level": "medium",
        "flags": ["test flag"],
        "suggestions": [],
    })
    fenced = f"```json\n{payload}\n```"
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(fenced)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)
    # All three docs return risk_level "medium" → max = "medium"
    assert result.risk_level == "medium"
    # cover_letter is the first doc reviewed → its flags land in cover_letter_flags
    assert "test flag" in result.cover_letter_flags


# ---------------------------------------------------------------------------
# review_truthfulness
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_all_pass(mock_client_cls, document_set, career_profile):
    payload = json.dumps({
        "pass_strict": True,
        "unsupported_claims": [],
        "evidence_examples": ["Led backend migration"],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(document_set, career_profile)

    assert isinstance(result, TruthfulnessResult)
    assert result.all_supported is True
    assert result.cover_letter.pass_strict is True
    assert result.resume.pass_strict is True
    assert result.interview_guide.pass_strict is True


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_one_fails(mock_client_cls, document_set, career_profile):
    pass_payload = json.dumps({
        "pass_strict": True,
        "unsupported_claims": [],
        "evidence_examples": [],
        "suggestions": [],
    })
    fail_payload = json.dumps({
        "pass_strict": False,
        "unsupported_claims": ["Led a team of 50"],
        "evidence_examples": [],
        "suggestions": ["Remove '50' or reduce to actual team size"],
    })
    mock_client = MagicMock()
    # First call (cover letter) passes, second (resume) fails, third (interview guide) passes
    mock_client.chat.side_effect = [
        _make_mock_response(pass_payload),
        _make_mock_response(fail_payload),
        _make_mock_response(pass_payload),
    ]
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(document_set, career_profile)

    assert result.all_supported is False
    assert result.cover_letter.pass_strict is True
    assert result.resume.pass_strict is False
    assert "Led a team of 50" in result.resume.unsupported_claims
    assert result.interview_guide.pass_strict is True


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
    payload = json.dumps({"risk_level": "low", "flags": [], "suggestions": []})
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
def test_voice_review_worst_of_aggregation(mock_client_cls, document_set, voice_profile):
    """Overall match should be the worst (minimum) across all per-doc matches."""
    strong = json.dumps({"overall_match": "strong", "assessment": "Good", "issues": [], "suggestions": []})
    weak = json.dumps({"overall_match": "weak", "assessment": "Poor", "issues": ["too formal"], "suggestions": []})
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_mock_response(strong),   # cover letter
        _make_mock_response(weak),     # resume
        _make_mock_response(strong),   # interview guide
    ]
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert result.overall_match == "weak"
    assert "too formal" in result.specific_issues


# ---------------------------------------------------------------------------
# AI detection: worst-of (max risk) aggregation
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_ai_detection_worst_of_risk(mock_client_cls, document_set):
    """Overall risk should be the worst (maximum) across all per-doc risks."""
    low = json.dumps({"risk_level": "low", "flags": [], "suggestions": []})
    high = json.dumps({"risk_level": "high", "flags": ["passionate about innovation"], "suggestions": []})
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_mock_response(low),    # cover letter
        _make_mock_response(low),    # resume
        _make_mock_response(high),   # interview guide
    ]
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)

    assert result.risk_level == "high"
    assert "passionate about innovation" in result.interview_guide_flags


# ---------------------------------------------------------------------------
# Missing documents: skipped cleanly
# ---------------------------------------------------------------------------


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_voice_skips_missing_docs(mock_client_cls, voice_profile):
    docs = DocumentSet(cover_letter="Some content.", resume=None, interview_guide=None)
    payload = json.dumps({
        "overall_match": "moderate",
        "assessment": "Okay match",
        "issues": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(docs, voice_profile)

    assert result.overall_match == "moderate"
    # Only 1 API call for the single present doc
    assert mock_client.chat.call_count == 1
    assert result.resume_assessment == "(not generated)"


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_truthfulness_skips_missing_docs(mock_client_cls, career_profile):
    docs = DocumentSet(cover_letter=None, resume="# Resume content", interview_guide=None)
    payload = json.dumps({
        "pass_strict": True,
        "unsupported_claims": [],
        "evidence_examples": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_truthfulness(docs, career_profile)

    assert result.all_supported is True
    assert mock_client.chat.call_count == 1  # Only resume reviewed
    assert result.cover_letter.pass_strict is True  # Default for missing
    assert result.interview_guide.pass_strict is True  # Default for missing


@patch("resume_refinery.reviewers.ollama.Client")
def test_review_ai_detection_skips_missing_docs(mock_client_cls):
    docs = DocumentSet(cover_letter=None, resume=None, interview_guide="guide content")
    payload = json.dumps({"risk_level": "medium", "flags": ["generic phrase"], "suggestions": []})
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response(payload)
    mock_client_cls.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(docs)

    assert result.risk_level == "medium"
    assert mock_client.chat.call_count == 1
    assert result.cover_letter_flags == []
    assert result.resume_flags == []
    assert "generic phrase" in result.interview_guide_flags

