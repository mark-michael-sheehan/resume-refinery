"""Tests for the generation agent (mocked API)."""

from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.agent import ResumeRefineryAgent
from resume_refinery.prompts import generation_user_message


def test_generation_user_message_contains_all_sections(career_profile, voice_profile, job_description):
    msg = generation_user_message(
        career_profile_content=career_profile.raw_content,
        voice_profile_content=voice_profile.raw_content,
        job_description_content=job_description.raw_content,
        doc_prompt="Write a cover letter.",
    )
    assert "Career Profile" in msg
    assert "Voice Profile" in msg
    assert "Job Description" in msg
    assert "Jordan Lee" in msg
    assert "Acme Cloud" in msg


def test_generation_user_message_with_feedback(career_profile, voice_profile, job_description):
    msg = generation_user_message(
        career_profile_content=career_profile.raw_content,
        voice_profile_content=voice_profile.raw_content,
        job_description_content=job_description.raw_content,
        doc_prompt="Write a cover letter.",
        feedback="Make it shorter.",
        previous_version="Previous draft text.",
    )
    assert "Make it shorter." in msg
    assert "Previous draft text." in msg
    assert "User Feedback" in msg
    assert "Previous Version" in msg


def _make_mock_stream(text: str):
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = text

    mock_final = MagicMock()
    mock_final.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.get_final_message.return_value = mock_final
    mock_stream.text_stream = iter([text])
    return mock_stream


@patch("resume_refinery.agent.anthropic.Anthropic")
def test_generate_document_returns_string(mock_anthropic, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream("# Jordan Lee\n\nCover letter text.")
    mock_anthropic.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    result = agent.generate_document("cover_letter", career_profile, voice_profile, job_description)
    assert "Jordan Lee" in result


@patch("resume_refinery.agent.anthropic.Anthropic")
def test_generate_all_returns_document_set(mock_anthropic, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream("Generated content.")
    mock_anthropic.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    docs = agent.generate_all(career_profile, voice_profile, job_description)

    assert docs.cover_letter == "Generated content."
    assert docs.resume == "Generated content."
    assert docs.interview_guide == "Generated content."


@patch("resume_refinery.agent.anthropic.Anthropic")
def test_stream_document_yields_chunks(mock_anthropic, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_mock_stream("chunk1")
    mock_anthropic.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    chunks = list(agent.stream_document("resume", career_profile, voice_profile, job_description))
    assert len(chunks) > 0
