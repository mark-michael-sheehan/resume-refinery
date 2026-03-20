"""Tests for review agents (mocked API)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import AIDetectionResult, VoiceReviewResult
from resume_refinery.reviewers import DocumentReviewer


def _make_mock_stream(response_text: str):
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = response_text

    mock_final = MagicMock()
    mock_final.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.get_final_message.return_value = mock_final
    return mock_stream


@patch("resume_refinery.reviewers.anthropic.Anthropic")
def test_review_voice_returns_result(mock_anthropic, document_set, voice_profile):
    payload = json.dumps({
        "overall_match": "strong",
        "cover_letter_assessment": "Great voice match.",
        "resume_assessment": "Consistent.",
        "interview_guide_assessment": "Slightly formal.",
        "specific_issues": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream(payload)
    mock_anthropic.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert isinstance(result, VoiceReviewResult)
    assert result.overall_match == "strong"


@patch("resume_refinery.reviewers.anthropic.Anthropic")
def test_review_ai_detection_returns_result(mock_anthropic, document_set):
    payload = json.dumps({
        "risk_level": "low",
        "cover_letter_flags": [],
        "resume_flags": [],
        "interview_guide_flags": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream(payload)
    mock_anthropic.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)

    assert isinstance(result, AIDetectionResult)
    assert result.risk_level == "low"


@patch("resume_refinery.reviewers.anthropic.Anthropic")
def test_reviewer_strips_json_fences(mock_anthropic, document_set):
    payload = json.dumps({
        "risk_level": "medium",
        "cover_letter_flags": ["test flag"],
        "resume_flags": [],
        "interview_guide_flags": [],
        "suggestions": [],
    })
    fenced = f"```json\n{payload}\n```"
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream(fenced)
    mock_anthropic.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)
    assert result.risk_level == "medium"
    assert "test flag" in result.cover_letter_flags
