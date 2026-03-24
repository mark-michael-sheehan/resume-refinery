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
    """Build a non-streaming ollama chat response mock."""
    mock_message = MagicMock()
    mock_message.content = text
    mock_response = MagicMock()
    mock_response.message = mock_message
    return mock_response


def _make_mock_stream_response(text: str):
    """Build an iterable of streaming ollama chunks."""
    mock_message = MagicMock()
    mock_message.content = text
    mock_chunk = MagicMock()
    mock_chunk.message = mock_message
    return iter([mock_chunk])


@patch("resume_refinery.agent.ollama.Client")
def test_generate_document_returns_string(mock_client_cls, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("# Jordan Lee\n\nCover letter text.")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    result = agent.generate_document("cover_letter", career_profile, voice_profile, job_description)
    assert "Jordan Lee" in result


@patch("resume_refinery.agent.ollama.Client")
def test_generate_all_returns_document_set(mock_client_cls, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("Generated content.")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    docs = agent.generate_all(career_profile, voice_profile, job_description)

    assert docs.cover_letter == "Generated content."
    assert docs.resume == "Generated content."
    assert docs.interview_guide == "Generated content."


@patch("resume_refinery.agent.ollama.Client")
def test_stream_document_yields_chunks(mock_client_cls, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_stream_response("chunk1")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    chunks = list(agent.stream_document("resume", career_profile, voice_profile, job_description))
    assert len(chunks) > 0


@patch("resume_refinery.agent.ollama.Client")
def test_generate_document_with_feedback(mock_client_cls, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("Revised cover letter.")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    result = agent.generate_document(
        "cover_letter", career_profile, voice_profile, job_description,
        feedback="Make it shorter.", previous_version="Old draft.",
    )
    assert result == "Revised cover letter."
    # Verify the user message includes feedback and previous version
    call_args = mock_client.chat.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    assert "Make it shorter." in user_content
    assert "Old draft." in user_content


@patch("resume_refinery.agent.ollama.Client")
def test_generate_document_strips_whitespace(mock_client_cls, career_profile, voice_profile, job_description):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("  padded output  \n\n")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    result = agent.generate_document("resume", career_profile, voice_profile, job_description)
    assert result == "padded output"


@patch("resume_refinery.agent.ollama.Client")
def test_stream_document_skips_empty_chunks(mock_client_cls, career_profile, voice_profile, job_description):
    """Empty content chunks should not be yielded."""
    mock_client = MagicMock()
    empty_chunk = MagicMock()
    empty_chunk.message = MagicMock()
    empty_chunk.message.content = ""
    real_chunk = MagicMock()
    real_chunk.message = MagicMock()
    real_chunk.message.content = "hello"
    mock_client.chat.return_value = iter([empty_chunk, real_chunk])
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    chunks = list(agent.stream_document("cover_letter", career_profile, voice_profile, job_description))
    assert chunks == ["hello"]


@patch("resume_refinery.agent.ollama.Client")
def test_generate_all_calls_all_three_keys(mock_client_cls, career_profile, voice_profile, job_description):
    """generate_all should call chat exactly 3 times (one per doc type)."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("content")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    docs = agent.generate_all(career_profile, voice_profile, job_description)
    assert mock_client.chat.call_count == 3
    assert docs.all_present()


@patch("resume_refinery.agent.ollama.Client")
def test_ollama_called_with_think_true(mock_client_cls, career_profile, voice_profile, job_description):
    """Generation calls should include think=True so the model can self-check."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_mock_response("output")
    mock_client_cls.return_value = mock_client

    agent = ResumeRefineryAgent(api_key="test-key")
    agent.generate_document("resume", career_profile, voice_profile, job_description)
    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["think"] is True


def test_generation_user_message_no_feedback(career_profile, voice_profile, job_description):
    """When no feedback or previous_version, those sections should be absent."""
    msg = generation_user_message(
        career_profile_content=career_profile.raw_content,
        voice_profile_content=voice_profile.raw_content,
        job_description_content=job_description.raw_content,
        doc_prompt="Generate a resume.",
    )
    assert "User Feedback" not in msg
    assert "Previous Version" not in msg
