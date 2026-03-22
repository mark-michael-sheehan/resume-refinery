"""Tests for review agents (mocked API)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import AIDetectionResult, VoiceReviewResult
from resume_refinery.reviewers import DocumentReviewer


def _make_mock_response(response_text: str):
    """Build a non-streaming ChatCompletion mock."""
    mock_message = MagicMock()
    mock_message.content = response_text

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch("resume_refinery.reviewers.openai.OpenAI")
def test_review_voice_returns_result(mock_openai, document_set, voice_profile):
    payload = json.dumps({
        "overall_match": "strong",
        "cover_letter_assessment": "Great voice match.",
        "resume_assessment": "Consistent.",
        "interview_guide_assessment": "Slightly formal.",
        "specific_issues": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(payload)
    mock_openai.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_voice(document_set, voice_profile)

    assert isinstance(result, VoiceReviewResult)
    assert result.overall_match == "strong"


@patch("resume_refinery.reviewers.openai.OpenAI")
def test_review_ai_detection_returns_result(mock_openai, document_set):
    payload = json.dumps({
        "risk_level": "low",
        "cover_letter_flags": [],
        "resume_flags": [],
        "interview_guide_flags": [],
        "suggestions": [],
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(payload)
    mock_openai.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)

    assert isinstance(result, AIDetectionResult)
    assert result.risk_level == "low"


@patch("resume_refinery.reviewers.openai.OpenAI")
def test_reviewer_strips_json_fences(mock_openai, document_set):
    payload = json.dumps({
        "risk_level": "medium",
        "cover_letter_flags": ["test flag"],
        "resume_flags": [],
        "interview_guide_flags": [],
        "suggestions": [],
    })
    fenced = f"```json\n{payload}\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(fenced)
    mock_openai.return_value = mock_client

    reviewer = DocumentReviewer(api_key="test-key")
    result = reviewer.review_ai_detection(document_set)
    assert result.risk_level == "medium"
    assert "test flag" in result.cover_letter_flags
