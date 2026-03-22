"""Tests for workflow orchestration over specialist agents."""

from __future__ import annotations

from pathlib import Path

from resume_refinery.models import (
    AIDetectionResult,
    DocumentSet,
    DocumentTruthResult,
    EvidencePack,
    JobRequirement,
    ReviewBundle,
    TruthfulnessResult,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from resume_refinery.orchestrator import ResumeRefineryOrchestrator
from resume_refinery.session import SessionStore


class FakeEvidenceAgent:
    def build_evidence_pack(self, career, job):
        return EvidencePack(
            job_requirements=[JobRequirement(requirement="distributed systems")],
            matched_evidence=[],
            gaps=[],
            source_summary=["Reduced infra costs by $180K/year"],
        )


class FakeVoiceAgent:
    def build_style_guide(self, voice):
        return VoiceStyleGuide(
            core_adjectives=["direct", "analytical"],
            style_rules=["Short declarative sentences"],
        )


class FakeDraftingAgent:
    def stream_document(self, key, career, voice, job, context, feedback=None, previous_version=None):
        yield f"{key} draft"

    def generate_document(self, key, career, voice, job, context, feedback=None, previous_version=None):
        return f"{key} rewritten"


class FakeVerificationAgent:
    def __init__(self):
        self.truth_calls = 0

    def review_truthfulness(self, docs, career):
        self.truth_calls += 1
        passed = self.truth_calls > 1
        truth_doc = DocumentTruthResult(pass_strict=passed, unsupported_claims=[] if passed else ["unsupported claim"], evidence_examples=[])
        return TruthfulnessResult(
            all_supported=passed,
            cover_letter=truth_doc,
            resume=truth_doc,
            interview_guide=truth_doc,
            suggestions=[],
        )

    def review_all(self, docs, career, voice):
        truth_doc = DocumentTruthResult(pass_strict=True, unsupported_claims=[], evidence_examples=[])
        return ReviewBundle(
            truthfulness=TruthfulnessResult(
                all_supported=True,
                cover_letter=truth_doc,
                resume=truth_doc,
                interview_guide=truth_doc,
                suggestions=[],
            ),
            voice=VoiceReviewResult(
                overall_match="strong",
                cover_letter_assessment="Good",
                resume_assessment="Good",
                interview_guide_assessment="Good",
                specific_issues=[],
                suggestions=[],
            ),
            ai_detection=AIDetectionResult(
                risk_level="low",
                cover_letter_flags=[],
                resume_flags=[],
                interview_guide_flags=[],
                suggestions=[],
            ),
        )


class FakeRepairAgent:
    def __init__(self):
        self.calls = 0

    def repair_documents(self, docs, truth, career, voice, job, context, feedback=None):
        self.calls += 1
        docs.cover_letter = "cover_letter repaired"
        docs.resume = "resume repaired"
        docs.interview_guide = "interview_guide repaired"
        return docs


def test_orchestrator_create_session_run_builds_artifacts_and_exports(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    assert result.session.current_version == 1
    assert result.evidence_pack is not None
    assert result.voice_style_guide is not None
    assert result.exported_paths
    assert repair.calls == 1
    assert Path(next(iter(result.exported_paths.values()))).exists()


def test_orchestrator_refine_session_run_updates_selected_doc(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    first = orchestrator.create_session_run(career_profile, voice_profile, job_description, skip_review=True)
    second = orchestrator.refine_session_run(first.session.session_id, "Tighten the opener", doc="cover_letter", skip_review=True)

    assert second.session.current_version == 2
    assert second.documents.cover_letter is not None
    assert second.documents.resume is not None
    assert second.documents.interview_guide is not None
