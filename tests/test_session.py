"""Tests for session management."""

import json
import os

import pytest

from resume_refinery.models import DocumentSet, ReviewBundle
from resume_refinery.session import SessionStore, _slugify


def test_slugify():
    assert _slugify("Acme Cloud Corp!") == "acme-cloud-corp"
    assert _slugify("Staff Engineer, Platform") == "staff-engineer-platform"


def test_create_session(tmp_path, career_profile, voice_profile, job_description, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)

    assert session.session_id.startswith("acme-cloud")
    assert session.current_version == 0
    assert (tmp_path / session.session_id / "inputs" / "career_profile.md").exists()
    assert (tmp_path / session.session_id / "inputs" / "voice_profile.md").exists()
    assert (tmp_path / session.session_id / "inputs" / "job_description.md").exists()


def test_save_and_load_documents(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    assert session.current_version == 1
    loaded = store.load_documents(session)
    assert loaded.cover_letter == document_set.cover_letter
    assert loaded.resume == document_set.resume


def test_save_and_load_reviews(tmp_path, career_profile, voice_profile, job_description, document_set, review_bundle, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)
    session = store.save_reviews(session, review_bundle)

    assert session.versions[0].has_reviews is True

    loaded = store.load_reviews(session)
    assert loaded.voice.overall_match == review_bundle.voice.overall_match
    assert loaded.ai_detection.risk_level == review_bundle.ai_detection.risk_level


def test_list_sessions(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    store.create(job_description, career_profile, voice_profile)
    sessions = store.list_sessions()
    assert len(sessions) == 1


def test_get_session(tmp_path, career_profile, voice_profile, job_description, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    created = store.create(job_description, career_profile, voice_profile)
    retrieved = store.get(created.session_id)
    assert retrieved.session_id == created.session_id


def test_get_session_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    with pytest.raises(FileNotFoundError):
        store.get("nonexistent-session")


def test_versioning_increments(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)
    session = store.save_documents(session, document_set, feedback="Make it shorter")

    assert session.current_version == 2
    assert len(session.versions) == 2
    assert session.versions[1].feedback == "Make it shorter"


def test_load_inputs(tmp_path, career_profile, voice_profile, job_description, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    loaded_career, loaded_voice = store.load_inputs(session)

    assert loaded_career.raw_content == career_profile.raw_content
    assert loaded_voice.raw_content == voice_profile.raw_content
