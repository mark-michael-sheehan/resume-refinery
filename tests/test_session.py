"""Tests for session management."""

import json
import os

import pytest

from resume_refinery.models import (
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    EvidencePack,
    ReviewBundle,
    TruthfulnessResult,
    VoiceReviewResult,
    VoiceStyleGuide,
    AIDetectionResult,
)
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


# ---------------------------------------------------------------------------
# Session ID deduplication on collision
# ---------------------------------------------------------------------------


def test_session_id_dedup(tmp_path, career_profile, voice_profile, job_description, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    first = store.create(job_description, career_profile, voice_profile)
    second = store.create(job_description, career_profile, voice_profile)

    assert first.session_id != second.session_id
    assert second.session_id.endswith("_2") or second.session_id > first.session_id


# ---------------------------------------------------------------------------
# Load documents for a specific version
# ---------------------------------------------------------------------------


def test_load_documents_specific_version(tmp_path, career_profile, voice_profile, job_description, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    docs_v1 = DocumentSet(cover_letter="v1 letter", resume="v1 resume", interview_guide="v1 guide")
    session = store.save_documents(session, docs_v1)
    docs_v2 = DocumentSet(cover_letter="v2 letter", resume="v2 resume", interview_guide="v2 guide")
    session = store.save_documents(session, docs_v2)

    loaded_v1 = store.load_documents(session, version=1)
    loaded_v2 = store.load_documents(session, version=2)

    assert loaded_v1.cover_letter == "v1 letter"
    assert loaded_v2.cover_letter == "v2 letter"


# ---------------------------------------------------------------------------
# Save/load reviews with truthfulness
# ---------------------------------------------------------------------------


def test_save_and_load_truthfulness_review(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    doc = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=True,
        cover_letter=doc, resume=doc, interview_guide=doc,
        suggestions=["Keep it up"],
    )
    bundle = ReviewBundle(truthfulness=truth)
    session = store.save_reviews(session, bundle)

    loaded = store.load_reviews(session)
    assert loaded.truthfulness is not None
    assert loaded.truthfulness.all_supported is True
    assert loaded.voice is None
    assert loaded.ai_detection is None


# ---------------------------------------------------------------------------
# Load reviews when none exist
# ---------------------------------------------------------------------------


def test_load_reviews_when_none_exist(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    loaded = store.load_reviews(session)
    assert loaded.voice is None
    assert loaded.ai_detection is None
    assert loaded.truthfulness is None


# ---------------------------------------------------------------------------
# save_documents with specific docs_regenerated list
# ---------------------------------------------------------------------------


def test_save_documents_with_docs_regenerated(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set, docs_regenerated=["cover_letter"])

    assert session.versions[0].docs_regenerated == ["cover_letter"]


# ---------------------------------------------------------------------------
# slugify edge cases
# ---------------------------------------------------------------------------


def test_slugify_special_chars():
    # Trailing special chars become trailing dashes after substitution
    result = _slugify("Hello World!!! @#$")
    assert result.startswith("hello-world")
    assert "!!!" not in result


def test_slugify_long_string():
    result = _slugify("a" * 100)
    assert len(result) <= 40


# ---------------------------------------------------------------------------
# Context persistence (EvidencePack + VoiceStyleGuide)
# ---------------------------------------------------------------------------


def test_save_and_load_context(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    context = DraftingContext(
        evidence_pack=EvidencePack(gaps=["Rust experience"], source_summary=["profile"]),
        voice_style_guide=VoiceStyleGuide(core_adjectives=["direct", "analytical"]),
    )
    store.save_context(session, context)

    loaded = store.load_context(session)
    assert loaded is not None
    assert loaded.evidence_pack.gaps == ["Rust experience"]
    assert loaded.voice_style_guide.core_adjectives == ["direct", "analytical"]


def test_load_context_when_none_exist(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    loaded = store.load_context(session)
    assert loaded is None


# ---------------------------------------------------------------------------
# Repair pass snapshots
# ---------------------------------------------------------------------------


def test_save_and_load_repair_pass(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    pass0_docs = DocumentSet(cover_letter="pass 0 CL", resume="pass 0 resume", interview_guide="pass 0 guide")
    pass1_docs = DocumentSet(cover_letter="pass 1 CL", resume="pass 1 resume", interview_guide="pass 1 guide")
    store.save_repair_pass(session, 0, pass0_docs)
    store.save_repair_pass(session, 1, pass1_docs)

    docs0, reviews0 = store.load_repair_pass(session, 0)
    docs1, reviews1 = store.load_repair_pass(session, 1)
    assert docs0 is not None
    assert docs0.cover_letter == "pass 0 CL"
    assert reviews0 is None
    assert docs1 is not None
    assert docs1.resume == "pass 1 resume"
    assert reviews1 is None


def test_load_repair_pass_nonexistent(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    docs, reviews = store.load_repair_pass(session, 99)
    assert docs is None
    assert reviews is None


def test_save_and_load_repair_pass_with_reviews(tmp_path, career_profile, voice_profile, job_description, document_set, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    session = store.create(job_description, career_profile, voice_profile)
    session = store.save_documents(session, document_set)

    docs = DocumentSet(cover_letter="CL text", resume="Resume text", interview_guide="Guide text")
    truth = TruthfulnessResult(
        all_supported=True,
        cover_letter=DocumentTruthResult(pass_strict=True, suggestions=["looks good"]),
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
    )
    voice = VoiceReviewResult(
        overall_match="strong",
        cover_letter_assessment="good",
        resume_assessment="good",
        interview_guide_assessment="good",
    )
    ai = AIDetectionResult(
        risk_level="low",
        cover_letter_suggestions=["cl suggestions"],
        resume_suggestions=["resume suggestions"],
    )
    bundle = ReviewBundle(truthfulness=truth, voice=voice, ai_detection=ai)

    store.save_repair_pass(session, 0, docs, reviews=bundle)

    loaded_docs, loaded_reviews = store.load_repair_pass(session, 0)
    assert loaded_docs is not None
    assert loaded_docs.cover_letter == "CL text"
    assert loaded_reviews is not None
    assert loaded_reviews.truthfulness.all_supported is True
    assert loaded_reviews.voice.overall_match == "strong"
    assert loaded_reviews.ai_detection.cover_letter_suggestions == ["cl suggestions"]
