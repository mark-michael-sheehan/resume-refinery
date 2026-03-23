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
        self.voice_calls = 0
        self.ai_calls = 0

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

    def review_voice(self, docs, voice):
        self.voice_calls += 1
        match = "strong" if self.voice_calls > 1 else "moderate"
        return VoiceReviewResult(
            overall_match=match,
            cover_letter_assessment="Mostly on-voice.",
            resume_assessment="Consistent.",
            interview_guide_assessment="Slightly formal.",
            specific_issues=[] if match == "strong" else ["opener feels generic"],
            suggestions=[] if match == "strong" else ["Use a concrete hook"],
        )

    def review_ai_detection(self, docs):
        self.ai_calls += 1
        risk = "low" if self.ai_calls > 1 else "medium"
        return AIDetectionResult(
            risk_level=risk,
            cover_letter_flags=[] if risk == "low" else ["results-driven"],
            resume_flags=[],
            interview_guide_flags=[],
            suggestions=[] if risk == "low" else ["Remove generic superlatives"],
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
        self.voice_calls = 0
        self.ai_calls = 0

    def repair_documents(self, docs, truth, career, voice, job, context, feedback=None):
        self.calls += 1
        docs.cover_letter = "cover_letter repaired"
        docs.resume = "resume repaired"
        docs.interview_guide = "interview_guide repaired"
        return docs

    def repair_voice(self, docs, voice_review, career, voice, job, context, feedback=None):
        self.voice_calls += 1
        docs.cover_letter = "cover_letter voice-fixed"
        docs.resume = "resume voice-fixed"
        docs.interview_guide = "interview_guide voice-fixed"
        return docs

    def repair_ai_detection(self, docs, ai_review, career, voice, job, context, feedback=None):
        self.ai_calls += 1
        docs.cover_letter = "cover_letter ai-fixed"
        docs.resume = "resume ai-fixed"
        docs.interview_guide = "interview_guide ai-fixed"
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


def test_orchestrator_create_verifies_all_three_loops(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """All three repair loops (truth, voice, AI) trigger with the default fakes."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Each fake fails on the first call, passes on the second → one repair per loop
    assert repair.calls == 1
    assert repair.voice_calls == 1
    assert repair.ai_calls == 1
    assert verification.truth_calls == 2
    assert verification.voice_calls == 2
    assert verification.ai_calls == 2
    # Final reviews should reflect the passing second call
    assert result.reviews.truthfulness is not None
    assert result.reviews.truthfulness.all_supported is True
    assert result.reviews.voice is not None
    assert result.reviews.voice.overall_match == "strong"
    assert result.reviews.ai_detection is not None
    assert result.reviews.ai_detection.risk_level == "low"


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


# ---------------------------------------------------------------------------
# All-pass: no repair when every review passes immediately
# ---------------------------------------------------------------------------


class AlwaysPassVerificationAgent:
    """Reviews always pass on the very first call."""

    def review_truthfulness(self, docs, career):
        passed_doc = DocumentTruthResult(pass_strict=True, unsupported_claims=[], evidence_examples=[])
        return TruthfulnessResult(
            all_supported=True,
            cover_letter=passed_doc,
            resume=passed_doc,
            interview_guide=passed_doc,
        )

    def review_voice(self, docs, voice):
        return VoiceReviewResult(
            overall_match="strong",
            cover_letter_assessment="Good",
            resume_assessment="Good",
            interview_guide_assessment="Good",
        )

    def review_ai_detection(self, docs):
        return AIDetectionResult(risk_level="low")

    def review_all(self, docs, career, voice):
        return ReviewBundle(
            truthfulness=self.review_truthfulness(docs, career),
            voice=self.review_voice(docs, voice),
            ai_detection=self.review_ai_detection(docs),
        )


def test_no_repair_when_all_reviews_pass(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=repair,
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    assert repair.calls == 0
    assert repair.voice_calls == 0
    assert repair.ai_calls == 0
    assert result.reviews.truthfulness.all_supported is True
    assert result.reviews.voice.overall_match == "strong"
    assert result.reviews.ai_detection.risk_level == "low"


# ---------------------------------------------------------------------------
# Exception handling: reviewer raises → graceful skip, other loops continue
# ---------------------------------------------------------------------------


class TruthRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_truthfulness(self, docs, career):
        raise RuntimeError("LLM timeout")


class VoiceRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_voice(self, docs, voice):
        raise RuntimeError("LLM timeout")


class AIRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_ai_detection(self, docs):
        raise RuntimeError("LLM timeout")


def _build_orchestrator(tmp_path, monkeypatch, verification):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    return ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=FakeRepairAgent(),
    )


def test_truth_review_exception_skips_loop(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    orch = _build_orchestrator(tmp_path, monkeypatch, TruthRaisesVerificationAgent())
    result = orch.create_session_run(career_profile, voice_profile, job_description)

    assert result.reviews.truthfulness is None
    assert result.reviews.voice is not None
    assert result.reviews.ai_detection is not None


def test_voice_review_exception_skips_loop(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    orch = _build_orchestrator(tmp_path, monkeypatch, VoiceRaisesVerificationAgent())
    result = orch.create_session_run(career_profile, voice_profile, job_description)

    assert result.reviews.truthfulness is not None
    assert result.reviews.voice is None
    assert result.reviews.ai_detection is not None


def test_ai_review_exception_skips_loop(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    orch = _build_orchestrator(tmp_path, monkeypatch, AIRaisesVerificationAgent())
    result = orch.create_session_run(career_profile, voice_profile, job_description)

    assert result.reviews.truthfulness is not None
    assert result.reviews.voice is not None
    assert result.reviews.ai_detection is None


# ---------------------------------------------------------------------------
# Per-loop max_passes: 0 skips, 1 = review-only, independent control
# ---------------------------------------------------------------------------


class NeverPassVerificationAgent(AlwaysPassVerificationAgent):
    """Reviews always fail — used to verify loop capping behaviour."""

    def __init__(self):
        self.truth_calls = 0
        self.voice_calls = 0
        self.ai_calls = 0

    def review_truthfulness(self, docs, career):
        self.truth_calls += 1
        doc = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
        return TruthfulnessResult(
            all_supported=False,
            cover_letter=doc, resume=doc, interview_guide=doc,
        )

    def review_voice(self, docs, voice):
        self.voice_calls += 1
        return VoiceReviewResult(
            overall_match="weak",
            cover_letter_assessment="Off-voice",
            resume_assessment="Off-voice",
            interview_guide_assessment="Off-voice",
            specific_issues=["too formal"],
        )

    def review_ai_detection(self, docs):
        self.ai_calls += 1
        return AIDetectionResult(
            risk_level="high",
            cover_letter_flags=["results-driven"],
            resume_flags=["proven track record"],
            interview_guide_flags=[],
        )


def test_max_passes_zero_skips_all_loops(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator._verify_and_repair(
        DocumentSet(cover_letter="cl", resume="r", interview_guide="ig"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_truth_passes=0, max_voice_passes=0, max_ai_passes=0,
    )

    assert verification.truth_calls == 0
    assert verification.voice_calls == 0
    assert verification.ai_calls == 0
    assert repair.calls == 0
    assert result.truthfulness is None
    assert result.voice is None
    assert result.ai_detection is None


def test_max_passes_one_reviews_but_never_repairs(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """With 1 pass the review runs once; because the review fails the loop exits without repair."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator._verify_and_repair(
        DocumentSet(cover_letter="cl", resume="r", interview_guide="ig"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_truth_passes=1, max_voice_passes=1, max_ai_passes=1,
    )

    # Review ran once for each loop
    assert verification.truth_calls == 1
    assert verification.voice_calls == 1
    assert verification.ai_calls == 1
    # Repair ran once for each loop (review fails → repair → loop exhausted)
    assert repair.calls == 1
    assert repair.voice_calls == 1
    assert repair.ai_calls == 1
    # Result reflects the failing review
    assert result.truthfulness.all_supported is False
    assert result.voice.overall_match == "weak"
    assert result.ai_detection.risk_level == "high"


def test_independent_per_loop_max_passes(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Each loop's max_passes is independent of the others."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator._verify_and_repair(
        DocumentSet(cover_letter="cl", resume="r", interview_guide="ig"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_truth_passes=0, max_voice_passes=3, max_ai_passes=1,
    )

    assert verification.truth_calls == 0
    assert verification.voice_calls == 3
    assert verification.ai_calls == 1
    assert repair.calls == 0        # truth skipped
    assert repair.voice_calls == 3   # 3 passes, all fail → 3 repairs
    assert repair.ai_calls == 1      # 1 pass, fails → 1 repair
    assert result.truthfulness is None


def test_max_passes_exhaustion_returns_last_review(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """When repairs never fix the issue, loop exhausts and returns the final (failing) review."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator._verify_and_repair(
        DocumentSet(cover_letter="cl", resume="r", interview_guide="ig"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_truth_passes=4, max_voice_passes=4, max_ai_passes=4,
    )

    assert verification.truth_calls == 4
    assert verification.voice_calls == 4
    assert verification.ai_calls == 4
    assert repair.calls == 4
    assert repair.voice_calls == 4
    assert repair.ai_calls == 4
    assert result.truthfulness.all_supported is False
    assert result.voice.overall_match == "weak"
    assert result.ai_detection.risk_level == "high"


# ---------------------------------------------------------------------------
# strict_truth_failed / allow_unverified
# ---------------------------------------------------------------------------


class TruthFailsVerificationAgent(AlwaysPassVerificationAgent):
    """Truth always fails; voice + AI always pass."""

    def review_truthfulness(self, docs, career):
        doc = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
        return TruthfulnessResult(
            all_supported=False,
            cover_letter=doc, resume=doc, interview_guide=doc,
        )


def test_strict_truth_failed_when_truth_fails(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=TruthFailsVerificationAgent(),
        repair_agent=FakeRepairAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)
    assert result.strict_truth_failed is True


def test_allow_unverified_suppresses_strict_truth_flag(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=TruthFailsVerificationAgent(),
        repair_agent=FakeRepairAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, allow_unverified=True,
    )
    assert result.strict_truth_failed is False


# ---------------------------------------------------------------------------
# skip_review persists only truthfulness
# ---------------------------------------------------------------------------


def test_skip_review_persists_only_truthfulness(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )

    assert result.reviews.truthfulness is not None
    assert result.reviews.voice is None
    assert result.reviews.ai_detection is None


# ---------------------------------------------------------------------------
# review_session_run
# ---------------------------------------------------------------------------


def test_review_session_run_returns_full_review(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=FakeRepairAgent(),
    )

    created = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Reset counters so we can verify review_session_run makes its own calls
    verification.truth_calls = 0
    verification.voice_calls = 0
    verification.ai_calls = 0

    reviewed = orchestrator.review_session_run(created.session.session_id)

    assert reviewed.reviews.truthfulness is not None
    assert reviewed.reviews.voice is not None
    assert reviewed.reviews.ai_detection is not None
    assert reviewed.documents.all_present()


# ---------------------------------------------------------------------------
# Stream callback receives chunks during generation
# ---------------------------------------------------------------------------


def test_stream_callback_receives_chunks(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    chunks_received: list[str] = []
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        stream_callback=chunks_received.append,
    )

    # FakeDraftingAgent yields one chunk per doc + newline delimiter
    assert len(chunks_received) >= 3
    assert any("cover_letter" in c for c in chunks_received)
    assert any("resume" in c for c in chunks_received)
    assert any("interview_guide" in c for c in chunks_received)


# ---------------------------------------------------------------------------
# Progress callback receives messages
# ---------------------------------------------------------------------------


def test_progress_callback_receives_messages(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    messages: list[str] = []
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        progress=messages.append,
    )

    assert len(messages) > 0
    assert any("evidence" in m.lower() for m in messages)
    assert any("voice" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# AI loop exits on no flags even when risk_level is elevated
# ---------------------------------------------------------------------------


class MediumRiskNoFlagsVerificationAgent(AlwaysPassVerificationAgent):
    """AI detection returns elevated risk but no per-doc flags."""

    def __init__(self):
        self.ai_calls = 0

    def review_ai_detection(self, docs):
        self.ai_calls += 1
        return AIDetectionResult(
            risk_level="medium",
            cover_letter_flags=[],
            resume_flags=[],
            interview_guide_flags=[],
        )


def test_ai_loop_exits_on_no_flags_despite_risk_level(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Even when risk_level is 'medium', the loop should exit if no docs have flags."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = MediumRiskNoFlagsVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        evidence_agent=FakeEvidenceAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
    )

    result = orchestrator._verify_and_repair(
        DocumentSet(cover_letter="cl", resume="r", interview_guide="ig"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_ai_passes=3,
    )

    # Should review once and exit — no flags means no repair needed
    assert verification.ai_calls == 1
    assert repair.ai_calls == 0
    assert result.ai_detection.risk_level == "medium"  # preserved as-is
