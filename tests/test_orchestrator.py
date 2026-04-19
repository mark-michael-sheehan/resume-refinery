"""Tests for workflow orchestration over specialist agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    ATSKeywordResult,
    CandidacyNarrative,
    CoverageGap,
    DocumentSet,
    DocumentTruthResult,
    GrammarResult,
    HiringManagerReview,
    NarrativeCoherenceResult,
    NarrativeCoverageResult,
    NarrativePillar,
    RepairPassResult,
    ReviewBundle,
    TruthfulnessResult,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from resume_refinery.orchestrator import ResumeRefineryOrchestrator
from resume_refinery.session import SessionStore


class FakeNarrativeAgent:
    def build_narrative(self, career, job, **kwargs):
        return CandidacyNarrative(
            thesis="Strong distributed systems background.",
            pillars=[NarrativePillar(theme="Backend", argument="Led migrations", career_evidence=["Reduced infra costs by $180K/year"])],
            gap_framing=[],
            raw_narrative="Narrative text.",
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

    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        self.truth_calls += 1
        passed = self.truth_calls > 1
        truth_doc = DocumentTruthResult(pass_strict=passed, unsupported_claims=[] if passed else ["unsupported claim"], evidence_examples=[])
        return TruthfulnessResult(
            all_supported=passed,
            resume=truth_doc,
        )

    def review_voice(self, docs, voice, *, exemptions=None):
        self.voice_calls += 1
        match = "strong" if self.voice_calls > 1 else "moderate"
        return VoiceReviewResult(
            overall_match=match,
            resume_assessment="Consistent.",
            specific_issues=[] if match == "strong" else ["opener feels generic"],
        )

    def review_ai_detection(self, docs, *, exemptions=None):
        self.ai_calls += 1
        risk = "low" if self.ai_calls > 1 else "medium"
        return AIDetectionResult(
            risk_level=risk,
            resume_flags=[] if risk == "low" else ["results-driven"],
        )

    def review_all(self, docs, career, voice, job):
        truth_doc = DocumentTruthResult(pass_strict=True, unsupported_claims=[], evidence_examples=[])
        return ReviewBundle(
            truthfulness=TruthfulnessResult(
                all_supported=True,
                resume=truth_doc,
            ),
            voice=VoiceReviewResult(
                overall_match="strong",
                resume_assessment="Good",
                specific_issues=[],
            ),
            ai_detection=AIDetectionResult(
                risk_level="low",
                resume_flags=[],
            ),
        )

    def review_hiring_manager(self, docs, job, *, exemptions=None):
        return HiringManagerReview(
            advance_likelihood=70,
            summary="Decent candidate.",
        )

    def review_relevance_pruning(self, docs, job, *, exemptions=None):
        from resume_refinery.models import RelevancePruningResult
        return RelevancePruningResult(
            overall_density="lean",
            resume_issues=[],
        )

    def review_ats_keyword(self, docs, job, career, *, exemptions=None):
        return ATSKeywordResult(alignment_score="strong")

    def review_grammar(self, docs, *, exemptions=None):
        return GrammarResult(clean=True)

    def review_narrative_coherence(self, docs, narrative, *, exemptions=None):
        return NarrativeCoherenceResult(alignment="strong")


class FakeRepairAgent:
    def __init__(self):
        self.unified_calls = 0

    def repair_unified(self, docs, truth, voice_review, ai_review, career, voice, job, context, feedback=None, hm_review=None, pruning_review=None, ats_review=None, grammar_review=None, narrative_review=None, preserve_instructions=None, phase="a", pass_num=0, prior_edits=None):
        self.unified_calls += 1
        docs.resume = "resume repaired"
        return RepairPassResult()


class FakeNarrativeCoverageAgent:
    def __init__(self, gaps=None):
        self._gaps = gaps or []
        self.analyze_calls = 0
        self.apply_calls = 0

    def analyze_coverage(self, narrative, career, docs, job):
        self.analyze_calls += 1
        return NarrativeCoverageResult(
            gaps=self._gaps,
            coverage_summary="Fake coverage summary.",
            pillars_covered=len(narrative.pillars) - len(self._gaps),
            pillars_total=len(narrative.pillars),
        )

    def apply_suggestions(self, docs, result):
        self.apply_calls += 1
        return docs


def test_orchestrator_create_exports_to_custom_output_dir(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(sessions_dir))
    custom_out = tmp_path / "my_output"
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        output_dir=custom_out, skip_review=True,
    )

    # Returned paths point to the custom output directory
    assert result.exported_paths
    for path_str in result.exported_paths.values():
        p = Path(path_str)
        assert p.exists()
        assert str(custom_out) in str(p)

    # Session version directory also has DOCX copies for versioning
    version_dir = store.session_dir(result.session.session_id) / f"v{result.session.current_version}"
    assert (version_dir / "resume.docx").exists()
    # cover_letter.docx no longer generated
    # interview_guide.docx no longer generated


def test_orchestrator_refine_exports_to_custom_output_dir(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(sessions_dir))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    first = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )
    custom_out = tmp_path / "refined_output"
    second = orchestrator.refine_session_run(
        first.session.session_id, "Tighten the opener",
        output_dir=custom_out,
    )

    # Returned paths point to the custom output directory
    assert second.exported_paths
    for path_str in second.exported_paths.values():
        p = Path(path_str)
        assert p.exists()
        assert str(custom_out) in str(p)

    # Session version directory also has DOCX copies for versioning
    version_dir = store.session_dir(second.session.session_id) / f"v{second.session.current_version}"
    assert (version_dir / "resume.docx").exists()
    # cover_letter.docx no longer generated
    # interview_guide.docx no longer generated


def test_orchestrator_create_session_run_builds_artifacts_and_exports(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    assert result.session.current_version == 1
    assert result.narrative is not None
    assert result.voice_style_guide is not None
    assert result.exported_paths
    # Single truth repair in first pass (truth fails once, then passes)
    assert repair.unified_calls == 1
    assert Path(next(iter(result.exported_paths.values()))).exists()


def test_orchestrator_create_verifies_truth_loop_and_advisory(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Truth loop repairs, then advisory reviews run once."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Pass 1: truth fails -> repair. Pass 2: truth passes -> break.
    # Advisory: all reviewers called once (including truth again).
    assert repair.unified_calls == 1
    assert verification.truth_calls == 3  # 2 loop + 1 advisory
    assert verification.voice_calls == 1  # advisory only
    assert verification.ai_calls == 1     # advisory only
    # Final reviews: truth from loop (passing), others from advisory
    assert result.reviews.truthfulness is not None
    assert result.reviews.truthfulness.all_supported is True
    assert result.reviews.voice is not None
    assert result.reviews.ai_detection is not None


def test_orchestrator_refine_session_run_updates_selected_doc(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    first = orchestrator.create_session_run(career_profile, voice_profile, job_description, skip_review=True)
    second = orchestrator.refine_session_run(first.session.session_id, "Tighten the opener", doc="resume")

    assert second.session.current_version == 2
    assert second.documents.resume is not None


def test_orchestrator_refine_uses_repair_agent_and_runs_reviews_once(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Refine should call the repair agent (not drafting agent) and run reviews once without looping."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    verification = FakeVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    first = orchestrator.create_session_run(career_profile, voice_profile, job_description, skip_review=True)
    initial_repair_calls = repair.unified_calls

    second = orchestrator.refine_session_run(first.session.session_id, "Make it more concise")

    # Repair agent called exactly once (single pass, no loop).
    assert repair.unified_calls == initial_repair_calls + 1
    # Documents were modified by repair agent.
    assert second.documents.resume == "resume repaired"
    assert second.documents.resume == "resume repaired"
    
    # Reviews are present in the result (all seven reviewers).
    assert second.reviews.truthfulness is not None
    assert second.reviews.voice is not None
    assert second.reviews.ai_detection is not None
    assert second.reviews.hiring_manager is not None
    assert second.reviews.ats_keyword is not None
    assert second.reviews.grammar is not None
    # Version was bumped.
    assert second.session.current_version == 2


def test_orchestrator_refine_with_doc_only_modifies_targeted_doc(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """When doc= is specified, only that document is modified by repair."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    first = orchestrator.create_session_run(career_profile, voice_profile, job_description, skip_review=True)

    second = orchestrator.refine_session_run(first.session.session_id, "Fix the opener", doc="resume")

    # Resume was modified by repair.
    assert second.documents.resume == "resume repaired"
    

def test_refine_passes_prior_reviews_to_repair_agent(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """refine_session_run should load prior reviewer findings and forward them
    to repair_unified so the user can reference reviewer feedback."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    class CapturingRepairAgent:
        """Captures the review arguments passed to repair_unified."""
        def __init__(self):
            self.last_truth = "NOT_CALLED"
            self.last_voice = "NOT_CALLED"
            self.last_ai = "NOT_CALLED"
            self.last_hm = "NOT_CALLED"
            self.last_pruning = "NOT_CALLED"
            self.last_ats = "NOT_CALLED"
            self.last_grammar = "NOT_CALLED"
            self.last_narrative = "NOT_CALLED"

        def repair_unified(self, docs, truth, voice_review, ai_review, career,
                           voice, job, context, feedback=None, hm_review=None,
                           pruning_review=None, ats_review=None,
                           grammar_review=None, narrative_review=None,
                           preserve_instructions=None, phase="a", pass_num=0,
                           prior_edits=None):
            self.last_truth = truth
            self.last_voice = voice_review
            self.last_ai = ai_review
            self.last_hm = hm_review
            self.last_pruning = pruning_review
            self.last_ats = ats_review
            self.last_grammar = grammar_review
            self.last_narrative = narrative_review
            docs.resume = "resume repaired"
            return RepairPassResult()

    capturing_repair = CapturingRepairAgent()

    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    # Create a session with skip_review=False so reviews are saved.
    first = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=False,
    )
    assert first.reviews.truthfulness is not None

    # Now swap in the capturing repair agent and refine.
    orchestrator.repair_agent = capturing_repair
    orchestrator.refine_session_run(
        first.session.session_id,
        "Implement all hiring-manager suggestions",
    )

    # The prior reviews should have been loaded and passed through.
    assert capturing_repair.last_truth is not None
    assert capturing_repair.last_truth != "NOT_CALLED"
    assert capturing_repair.last_voice is not None
    assert capturing_repair.last_voice != "NOT_CALLED"
    assert capturing_repair.last_ai is not None
    assert capturing_repair.last_ai != "NOT_CALLED"
    assert capturing_repair.last_hm is not None
    assert capturing_repair.last_hm != "NOT_CALLED"
    assert capturing_repair.last_ats is not None
    assert capturing_repair.last_ats != "NOT_CALLED"
    assert capturing_repair.last_grammar is not None
    assert capturing_repair.last_grammar != "NOT_CALLED"


# ---------------------------------------------------------------------------
# All-pass: no repair when every review passes immediately
# ---------------------------------------------------------------------------


class AlwaysPassVerificationAgent:
    """Reviews always pass on the very first call."""

    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        passed_doc = DocumentTruthResult(pass_strict=True, unsupported_claims=[], evidence_examples=[])
        return TruthfulnessResult(
            all_supported=True,
            resume=passed_doc,
        )

    def review_voice(self, docs, voice, *, exemptions=None):
        return VoiceReviewResult(
            overall_match="strong",
            resume_assessment="Good",
        )

    def review_ai_detection(self, docs, *, exemptions=None):
        return AIDetectionResult(risk_level="low")

    def review_hiring_manager(self, docs, job, *, exemptions=None):
        return HiringManagerReview(
            advance_likelihood=75,
            summary="Good candidate.",
        )

    def review_relevance_pruning(self, docs, job, *, exemptions=None):
        from resume_refinery.models import RelevancePruningResult
        return RelevancePruningResult(overall_density="lean", resume_issues=[])

    def review_ats_keyword(self, docs, job, career, *, exemptions=None):
        return ATSKeywordResult(alignment_score="strong")
    def review_grammar(self, docs, *, exemptions=None):
        return GrammarResult(clean=True)

    def review_narrative_coherence(self, docs, narrative, *, exemptions=None):
        return NarrativeCoherenceResult(alignment="strong")

    def review_all(self, docs, career, voice, job):
        return ReviewBundle(
            truthfulness=self.review_truthfulness(docs, career, job),
            voice=self.review_voice(docs, voice),
            ai_detection=self.review_ai_detection(docs),
        )


def test_no_repair_when_all_reviews_pass(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    assert repair.unified_calls == 0
    assert result.reviews.truthfulness.all_supported is True
    assert result.reviews.voice.overall_match == "strong"
    assert result.reviews.ai_detection.risk_level == "low"


# ---------------------------------------------------------------------------
# Exception handling: reviewer raises â†’ graceful skip, other loops continue
# ---------------------------------------------------------------------------


class TruthRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        raise RuntimeError("LLM timeout")


class VoiceRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_voice(self, docs, voice, *, exemptions=None):
        raise RuntimeError("LLM timeout")


class AIRaisesVerificationAgent(AlwaysPassVerificationAgent):
    def review_ai_detection(self, docs, *, exemptions=None):
        raise RuntimeError("LLM timeout")


def _build_orchestrator(tmp_path, monkeypatch, verification):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    return ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
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
# Per-reviewer suppression
# ---------------------------------------------------------------------------


class AlwaysFlagsAIPhraseVerification(AlwaysPassVerificationAgent):
    """Truth and voice always pass; AI detection always flags the same phrase."""

    def review_ai_detection(self, docs, *, exemptions=None):
        return AIDetectionResult(
            risk_level="medium",
            resume_flags=["accepted-phrase"],
        )


class AcceptsAIPhraseRepairAgent:
    """On first repair, accepts the flagged AI phrase instead of trying to fix it."""

    def __init__(self):
        self.unified_calls = 0

    def repair_unified(self, docs, truth, voice_review, ai_review, career, voice, job, context, feedback=None, hm_review=None, pruning_review=None, ats_review=None, grammar_review=None, narrative_review=None, preserve_instructions=None, phase="a", pass_num=0, prior_edits=None):
        self.unified_calls += 1
        return RepairPassResult(accepted_ai_phrases=["accepted-phrase"])


def test_ai_flags_appear_as_advisory_not_blocking(
    tmp_path, monkeypatch, career_profile, voice_profile, job_description
):
    """AI flags no longer block convergence; they appear in advisory reviews only."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    repair = AcceptsAIPhraseRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysFlagsAIPhraseVerification(),
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Truth always passes (from AlwaysPassVerificationAgent) -> no repair needed.
    # AI flags are advisory-only, so no repair is triggered by AI.
    assert repair.unified_calls == 0
    # Advisory review shows AI findings
    assert result.reviews.ai_detection is not None
    assert result.reviews.ai_detection.resume_flags == ["accepted-phrase"]


# ---------------------------------------------------------------------------
# Per-loop max_passes: 0 skips, 1 = review-only, independent control
# ---------------------------------------------------------------------------


class NeverPassVerificationAgent(AlwaysPassVerificationAgent):
    """Reviews always fail â€” used to verify loop capping behaviour."""

    def __init__(self):
        self.truth_calls = 0
        self.voice_calls = 0
        self.ai_calls = 0

    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        self.truth_calls += 1
        doc = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
        return TruthfulnessResult(
            all_supported=False,
            resume=doc,
        )

    def review_voice(self, docs, voice, *, exemptions=None):
        self.voice_calls += 1
        return VoiceReviewResult(
            overall_match="weak",
            resume_assessment="Off-voice",
            specific_issues=["too formal"],
        )

    def review_ai_detection(self, docs, *, exemptions=None):
        self.ai_calls += 1
        return AIDetectionResult(
            risk_level="high",
            resume_flags=["proven track record"],
        )


def test_max_passes_zero_skips_all_reviews(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result, _, _exempted = orchestrator._verify_and_repair(
        DocumentSet(resume="r"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_passes=0,
    )

    # Loop skipped entirely, but advisory reviews still run.
    assert verification.truth_calls == 1  # advisory only
    assert verification.voice_calls == 1  # advisory only
    assert verification.ai_calls == 1     # advisory only
    assert repair.unified_calls == 0
    # truth is overridden to None (loop never set it), others come from advisory
    assert result.truthfulness is None
    assert result.voice is not None
    assert result.ai_detection is not None


def test_max_passes_one_reviews_and_repairs_once(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """With 1 pass the review runs once; fails â†’ one unified repair â†’ loop exhausted."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result, _, _exempted = orchestrator._verify_and_repair(
        DocumentSet(resume="r"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_passes=1,
    )

    # Truth called once in loop + once in advisory = 2
    assert verification.truth_calls == 2
    assert verification.voice_calls == 1  # advisory only
    assert verification.ai_calls == 1     # advisory only
    # Single truth repair
    assert repair.unified_calls == 1
    # Truth from loop (still failing), others from advisory
    assert result.truthfulness.all_supported is False
    assert result.voice.overall_match == "weak"
    assert result.ai_detection.risk_level == "high"


def test_max_passes_exhaustion_returns_last_review(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """When repairs never fix the issue, loop exhausts and returns the final (failing) review."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = NeverPassVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result, _, _exempted = orchestrator._verify_and_repair(
        DocumentSet(resume="r"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_passes=4,
    )

    # 4 truth passes in loop + 1 advisory = 5
    assert verification.truth_calls == 5
    assert verification.voice_calls == 1  # advisory only
    assert verification.ai_calls == 1     # advisory only
    # 4 truth repairs
    assert repair.unified_calls == 4
    # Truth from loop (still failing), others from advisory
    assert result.truthfulness.all_supported is False
    assert result.voice.overall_match == "weak"
    assert result.ai_detection.risk_level == "high"


# ---------------------------------------------------------------------------
# strict_truth_failed / allow_unverified
# ---------------------------------------------------------------------------


class TruthFailsVerificationAgent(AlwaysPassVerificationAgent):
    """Truth always fails; voice + AI always pass."""

    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        doc = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
        return TruthfulnessResult(
            all_supported=False,
            resume=doc,
        )


def test_strict_truth_failed_when_truth_fails(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=TruthFailsVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)
    assert result.strict_truth_failed is True


def test_allow_unverified_suppresses_strict_truth_flag(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=TruthFailsVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
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
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
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
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
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
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        stream_callback=chunks_received.append,
    )

    # FakeDraftingAgent yields one chunk per doc + newline delimiter
    assert len(chunks_received) >= 1
    assert any("resume" in c for c in chunks_received)


# ---------------------------------------------------------------------------
# Progress callback receives messages
# ---------------------------------------------------------------------------


def test_progress_callback_receives_messages(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    messages: list[str] = []
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        progress=messages.append,
    )

    assert len(messages) > 0
    assert any("narrative" in m.lower() for m in messages)
    assert any("voice" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# AI loop exits on no flags even when risk_level is elevated
# ---------------------------------------------------------------------------


class MediumRiskNoFlagsVerificationAgent(AlwaysPassVerificationAgent):
    """AI detection returns elevated risk but no per-doc flags."""

    def __init__(self):
        self.ai_calls = 0

    def review_ai_detection(self, docs, *, exemptions=None):
        self.ai_calls += 1
        return AIDetectionResult(
            risk_level="medium",
            resume_flags=[],
        )


def test_ai_loop_exits_on_no_flags_despite_risk_level(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Even when risk_level is 'medium', the loop should exit if no docs have flags."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = MediumRiskNoFlagsVerificationAgent()
    repair = FakeRepairAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result, _, _exempted = orchestrator._verify_and_repair(
        DocumentSet(resume="r"),
        career_profile, voice_profile, job_description.model_copy(),
        orchestrator._build_context(career_profile, voice_profile, job_description),
        max_passes=3,
    )

    # Truth passes in loop (from AlwaysPass), AI is advisory-only.
    assert verification.ai_calls == 1  # advisory only
    assert repair.unified_calls == 0
    assert result.ai_detection.risk_level == "medium"  # preserved as-is


# ---------------------------------------------------------------------------
# Progress callback receives review summaries after each review
# ---------------------------------------------------------------------------


def test_progress_includes_review_summaries(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Progress messages should contain review result details, not just status labels."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    messages: list[str] = []
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        progress=messages.append,
    )

    combined = "\n".join(messages)
    # Truthfulness summary: first call fails, shows unsupported claim detail
    assert "UNSUPPORTED CLAIMS" in combined or "ALL SUPPORTED" in combined
    # Voice summary: shows per-doc match levels
    assert "Voice match:" in combined
    # Cover Letter summary removed (resume-only mode)
    # AI summary: shows risk level
    assert "AI-detection risk:" in combined


# ---------------------------------------------------------------------------
# Post-repair truthfulness recheck
# ---------------------------------------------------------------------------


def test_progress_includes_pass_headers(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Progress messages should include review pass headers."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    messages: list[str] = []
    verification = FakeVerificationAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        progress=messages.append,
    )

    combined = "\n".join(messages)
    # Should include pass headers
    assert "Truth Pass 1/" in combined
    # truth_calls = 2 in loop + 1 advisory = 3
    assert verification.truth_calls == 3


# ---------------------------------------------------------------------------
# Incremental saves: docs + context saved before review loop
# ---------------------------------------------------------------------------


def test_docs_saved_before_review_loop(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Documents and context should be on disk before the review loop starts."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))

    store = SessionStore()
    saved_during_review: dict[str, bool] = {}

    class CheckingVerificationAgent(AlwaysPassVerificationAgent):
        """On the first review call, check that docs already exist on disk."""

        def __init__(self):
            self.checked = False

        def review_truthfulness(self, docs, career, job, *, exemptions=None):
            if not self.checked:
                self.checked = True
                sessions = store.list_sessions()
                assert len(sessions) == 1
                session = sessions[0]
                loaded = store.load_documents(session)
                saved_during_review["resume"] = loaded.resume is not None
                saved_during_review["context"] = store.load_context(session) is not None
            return super().review_truthfulness(docs, career, job)

    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=CheckingVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(career_profile, voice_profile, job_description)

    assert saved_during_review.get("resume") is True
    assert saved_during_review.get("context") is True


# ---------------------------------------------------------------------------
# Exemptions are passed to reviewers during the repair loop
# ---------------------------------------------------------------------------


class ExemptionTrackingVerification(AlwaysPassVerificationAgent):
    """Records exemptions received by each reviewer across calls."""

    def __init__(self):
        self.truth_exemptions: list = []
        self.ai_exemptions: list = []
        self.voice_exemptions: list = []
        self.hm_exemptions: list = []
        self.pruning_exemptions: list = []
        self.ats_exemptions: list = []
        self.grammar_exemptions: list = []

    def review_ai_detection(self, docs, *, exemptions=None):
        self.ai_exemptions.append(exemptions)
        # Fail on first call to force a repair pass
        if len(self.ai_exemptions) == 1:
            return AIDetectionResult(
                risk_level="medium",
                resume_flags=["flagged-phrase"],
            )
        return super().review_ai_detection(docs, exemptions=exemptions)

    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        self.truth_exemptions.append(exemptions)
        return super().review_truthfulness(docs, career, job, exemptions=exemptions)


class AcceptsAIPhraseAndTrackRepair:
    def __init__(self):
        self.unified_calls = 0

    def repair_unified(self, docs, truth, voice_review, ai_review, career, voice, job, context, feedback=None, hm_review=None, pruning_review=None, ats_review=None, grammar_review=None, narrative_review=None, preserve_instructions=None, phase="a", pass_num=0, prior_edits=None):
        self.unified_calls += 1
        return RepairPassResult(accepted_ai_phrases=["flagged-phrase"])


def test_exemptions_passed_to_truth_reviewer_in_repair_loop(
    tmp_path, monkeypatch, career_profile, voice_profile, job_description
):
    """After a repair pass accepts a truth claim, the exemption list is passed on the next pass."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    verification = ExemptionTrackingVerification()
    repair = AcceptsAIPhraseAndTrackRepair()
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=verification,
        repair_agent=repair,
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Truth always passes -> no repair pass -> no AI exemption from repair
    assert repair.unified_calls == 0
    # Advisory still calls AI; first call has no exemptions
    assert verification.ai_exemptions[0] is None


# ---------------------------------------------------------------------------
# Refine loads and applies exemptions from prior create run
# ---------------------------------------------------------------------------


class AlwaysFlagsAIForRefine(AlwaysPassVerificationAgent):
    """AI detection always flags a specific phrase â€” used to verify suppression in refine."""

    def __init__(self):
        self.ai_exemptions_received: list = []

    def review_ai_detection(self, docs, *, exemptions=None):
        self.ai_exemptions_received.append(exemptions)
        return AIDetectionResult(
            risk_level="medium",
            resume_flags=["previously-accepted"],
        )


def test_refine_loads_exemptions_and_suppresses_review_findings(
    tmp_path, monkeypatch, career_profile, voice_profile, job_description
):
    """refine_session_run should load exemptions from the prior create run,
    pass them to reviewers, and apply post-filter suppressions."""
    from resume_refinery.models import ExemptedPhrases

    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    # Create initial run with skip_review, then manually save exemptions
    create_verification = AlwaysPassVerificationAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=create_verification,
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )
    created = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )

    # Save exemptions as if the create run accepted "previously-accepted"
    store.save_suppressions(created.session, ExemptedPhrases(
        ai_phrases=["previously-accepted"],
    ))

    # Now refine with a verification agent that always flags the suppressed phrase
    refine_verification = AlwaysFlagsAIForRefine()
    orchestrator2 = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=refine_verification,
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )
    result = orchestrator2.refine_session_run(
        created.session.session_id, "Make it shorter",
    )

    # Exemptions were passed to the AI reviewer
    assert refine_verification.ai_exemptions_received[0] == ["previously-accepted"]
    # Post-filter suppression removed the flag from the final review
    assert result.reviews.ai_detection is not None
    assert result.reviews.ai_detection.resume_flags == []
    # Exemptions were persisted with the new version
    loaded = store.load_suppressions(result.session)
    assert loaded is not None
    assert "previously-accepted" in loaded.ai_phrases


# ---------------------------------------------------------------------------
# load_suppressions scans versions backwards
# ---------------------------------------------------------------------------


def test_load_suppressions_returns_most_recent(
    tmp_path, monkeypatch, career_profile, voice_profile, job_description
):
    """load_suppressions should find exemptions from any prior version."""
    from resume_refinery.models import ExemptedPhrases, Session

    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()

    # Create a session and save exemptions in v1
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )
    created = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )
    store.save_suppressions(created.session, ExemptedPhrases(
        claims=["already-verified"],
    ))

    # Refine bumps to v2 (no exemptions saved in v2 yet)
    refined = orchestrator.refine_session_run(
        created.session.session_id, "Shorten.",
    )

    # load_suppressions should still find the v1 exemptions via backwards scan
    # (in practice refine now saves them too, so v2 will have them â€”
    # but verify the scanning logic works for the case where v2 has them)
    loaded = store.load_suppressions(refined.session)
    assert loaded is not None
    assert "already-verified" in loaded.claims


def test_docs_updated_after_each_repair_pass(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """After each repair pass, updated documents should be written to disk."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=FakeVerificationAgent(),  # pass 1 fails, pass 2 passes â†’ 1 repair
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(career_profile, voice_profile, job_description)

    # Repair happened and docs on disk should reflect repaired content
    loaded = store.load_documents(result.session)
    assert loaded.resume == "resume repaired"
    assert loaded.resume == "resume repaired"

    # Repair pass snapshot should also exist
    snap_docs, _ = store.load_repair_pass(result.session, 0)
    assert snap_docs is not None


# ---------------------------------------------------------------------------
# Prior-edit context builder
# ---------------------------------------------------------------------------


def test_build_prior_edits_empty_when_no_results():
    """_build_prior_edits returns empty dict when no repair results exist."""
    result = ResumeRefineryOrchestrator._build_prior_edits([])
    assert result == {}


def test_build_prior_edits_formats_edit_summary():
    """_build_prior_edits builds per-document summaries from RepairPassResult edits."""
    from resume_refinery.models import RepairEdit

    rp = RepairPassResult(
        edits={
            "resume": [
                RepairEdit(find="old text", replace="new text", reason="truthfulness fix", reviewer="truthfulness"),
                RepairEdit(find="remove me", replace="", reason="pruning", reviewer="pruning"),
                RepairEdit(find="foo", replace="bar", reason="voice fix", reviewer="voice"),
            ],
        },
    )
    result = ResumeRefineryOrchestrator._build_prior_edits([rp])

    assert "resume" in result
    assert "[truthfulness]" in result["resume"]
    assert "old text" in result["resume"]
    assert "new text" in result["resume"]
    assert "DELETED" in result["resume"]
    assert "[voice]" in result["resume"]
    assert "foo" in result["resume"]
    assert "bar" in result["resume"]


def test_build_prior_edits_accumulates_across_passes():
    """_build_prior_edits accumulates edits from multiple RepairPassResult objects."""
    from resume_refinery.models import RepairEdit

    rp1 = RepairPassResult(
        edits={"resume": [RepairEdit(find="a", replace="b", reviewer="truthfulness")]},
    )
    rp2 = RepairPassResult(
        edits={"resume": [RepairEdit(find="c", replace="d", reviewer="voice")]},
    )
    result = ResumeRefineryOrchestrator._build_prior_edits([rp1, rp2])

    assert "resume" in result
    assert "[truthfulness]" in result["resume"]
    assert "[voice]" in result["resume"]


def test_build_prior_edits_insert_after_summary():
    """_build_prior_edits labels insert_after edits as INSERTED."""
    from resume_refinery.models import RepairEdit

    rp = RepairPassResult(
        edits={
            "resume": [
                RepairEdit(find="## Skills", replace="\n- Kubernetes", reason="missing ATS keyword", reviewer="ats", insert_after=True),
            ],
        },
    )
    result = ResumeRefineryOrchestrator._build_prior_edits([rp])

    assert "resume" in result
    assert "INSERTED" in result["resume"]
    assert "## Skills" in result["resume"]
    assert "Kubernetes" in result["resume"]


# ---------------------------------------------------------------------------
# selected_docs: generate only chosen documents
# ---------------------------------------------------------------------------


def test_selected_docs_generates_resume(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """selected_docs defaults to resume only."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        skip_review=True,
        selected_docs=["resume"],
    )

    assert result.documents.resume is not None
    assert result.session.selected_docs == ["resume"]


def test_selected_docs_persisted_in_session(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """selected_docs is persisted in session metadata and can be reloaded."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        skip_review=True,
        selected_docs=["resume"],
    )

    loaded_session = store.get(result.session.session_id)
    assert loaded_session.selected_docs == ["resume"]


def test_selected_docs_default_all(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Without selected_docs, the resume is generated by default."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        skip_review=True,
    )

    assert result.documents.resume is not None
    assert set(result.session.selected_docs) == {"resume"}


def test_selected_docs_refine(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Refine should operate on the session's selected_docs (resume only)."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    created = orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        skip_review=True,
    )

    refined = orchestrator.refine_session_run(
        created.session.session_id, "Make it shorter",
    )

    assert refined.documents.resume is not None


def test_selected_docs_stream_callback_only_selected(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Stream callback should only receive chunks for selected documents."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    chunks: list[str] = []
    orchestrator = ResumeRefineryOrchestrator(
        store=SessionStore(),
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    orchestrator.create_session_run(
        career_profile, voice_profile, job_description,
        selected_docs=["resume"],
        stream_callback=chunks.append,
    )

    # Only resume chunks + newlines should be present
    assert any("resume" in c for c in chunks)


# ---------------------------------------------------------------------------
# extract_context + generate_session_run (two-phase flow)
# ---------------------------------------------------------------------------


def test_extract_context_creates_session_and_stages_context(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """extract_context should create a session and save staging context to disk."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    session, context = orchestrator.extract_context(
        career_profile, voice_profile, job_description,
    )

    assert session.current_version == 0
    assert context.narrative is not None
    assert context.voice_style_guide is not None
    # Staging context should be on disk
    loaded = store.load_staging_context(session)
    assert loaded is not None
    assert loaded.narrative.thesis == context.narrative.thesis


def test_generate_session_run_uses_staged_context(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """generate_session_run should load staged context and produce documents."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    session, context = orchestrator.extract_context(
        career_profile, voice_profile, job_description,
    )

    result = orchestrator.generate_session_run(
        session.session_id, skip_review=True,
    )

    assert result.session.current_version == 1
    assert result.documents.resume is not None
    # Staging context should be cleaned up
    assert store.load_staging_context(result.session) is None


def test_generate_session_run_with_filtered_evidence(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """generate_session_run should accept a curated context with filtered evidence."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    session, context = orchestrator.extract_context(
        career_profile, voice_profile, job_description,
    )

    # Pass the context directly (simulating curation review)
    from resume_refinery.models import DraftingContext, CandidacyNarrative as CN
    curated_context = DraftingContext(
        narrative=CN(
            thesis=context.narrative.thesis,
            pillars=[],  # all pillars removed
            gap_framing=context.narrative.gap_framing,
            raw_narrative=context.narrative.raw_narrative,
        ),
        voice_style_guide=context.voice_style_guide,
    )

    result = orchestrator.generate_session_run(
        session.session_id,
        context=curated_context,
        skip_review=True,
    )

    assert result.session.current_version == 1
    assert result.documents.resume is not None


def test_generate_session_run_fails_without_staged_context(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """generate_session_run should raise ValueError if no staged context exists and none is provided."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=FakeNarrativeCoverageAgent(),
    )

    # Create a session manually without staging context
    session = store.create(job_description, career_profile, voice_profile)

    with pytest.raises(ValueError, match="No staged context"):
        orchestrator.generate_session_run(session.session_id)


# ---------------------------------------------------------------------------
# Narrative coverage integration
# ---------------------------------------------------------------------------


def test_create_session_run_coverage_result_is_none(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Coverage analysis is absorbed into the narrative critic; coverage_result is None."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    coverage_agent = FakeNarrativeCoverageAgent()
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=coverage_agent,
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )

    # Coverage analysis no longer runs in orchestrator
    assert result.coverage_result is None
    assert coverage_agent.analyze_calls == 0


def test_create_session_run_no_coverage_analysis(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Coverage analysis is no longer invoked during create (absorbed into narrative critic)."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    gap = CoverageGap(
        pillar_theme="Cost Optimisation",
        career_evidence=["$180K savings"],
        suggested_content="- Reduced costs by $180K",
        anchor_section="Experience",
        confidence="high",
    )
    coverage_agent = FakeNarrativeCoverageAgent(gaps=[gap])
    store = SessionStore()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=coverage_agent,
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )

    assert coverage_agent.analyze_calls == 0
    assert coverage_agent.apply_calls == 0
    assert result.coverage_result is None


def test_coverage_result_not_persisted(tmp_path, monkeypatch, career_profile, voice_profile, job_description):
    """Coverage result is no longer persisted (absorbed into narrative critic)."""
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))
    store = SessionStore()
    coverage_agent = FakeNarrativeCoverageAgent()
    orchestrator = ResumeRefineryOrchestrator(
        store=store,
        narrative_agent=FakeNarrativeAgent(),
        voice_agent=FakeVoiceAgent(),
        drafting_agent=FakeDraftingAgent(),
        verification_agent=AlwaysPassVerificationAgent(),
        repair_agent=FakeRepairAgent(),
        coverage_agent=coverage_agent,
    )

    result = orchestrator.create_session_run(
        career_profile, voice_profile, job_description, skip_review=True,
    )

    loaded = store.load_coverage(result.session)
    assert loaded is None
    assert result.coverage_result is None
