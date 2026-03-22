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


def _make_mock_response(text: str):
    """Build a non-streaming ChatCompletion mock."""
    mock_message = MagicMock()
    mock_message.content = text

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _make_mock_stream_response(text: str):
    """Build an iterable of streaming chunks."""
    mock_delta = MagicMock()
    mock_delta.content = text

    mock_chunk_choice = MagicMock()
    mock_chunk_choice.delta = mock_delta

    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_chunk_choice]
    return iter([mock_chunk])


@patch("resume_refinery.agent.openai.OpenAI")
def test_generate_document_returns_string(mock_openai, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("# Jordan Lee\n\nCover letter text.")
    mock_openai.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    result = agent.generate_document("cover_letter", career_profile, voice_profile, job_description)
    assert "Jordan Lee" in result


@patch("resume_refinery.agent.openai.OpenAI")
def test_generate_all_returns_document_set(mock_openai, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("Generated content.")
    mock_openai.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    docs = agent.generate_all(career_profile, voice_profile, job_description)

    assert docs.cover_letter == "Generated content."
    assert docs.resume == "Generated content."
    assert docs.interview_guide == "Generated content."


@patch("resume_refinery.agent.openai.OpenAI")
def test_stream_document_yields_chunks(mock_openai, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_stream_response("chunk1")
    mock_openai.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    chunks = list(agent.stream_document("resume", career_profile, voice_profile, job_description))
    assert len(chunks) > 0
