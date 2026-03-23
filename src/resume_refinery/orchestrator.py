"""Workflow orchestrator for bounded multi-agent execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from .exporters import export_document_set
from .models import (
    AIDetectionResult,
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    OrchestrationResult,
    ReviewBundle,
    Session,
    TruthfulnessResult,
    VoiceProfile,
    VoiceReviewResult,
    JobDescription,
)
from .session import SessionStore
from .specialist_agents import DraftingAgent, EvidenceAgent, RepairAgent, VerificationAgent, VoiceAgent

load_dotenv()

MAX_TRUTH_PASSES = int(os.environ.get("RESUME_REFINERY_MAX_TRUTH_PASSES", "2"))
MAX_VOICE_PASSES = int(os.environ.get("RESUME_REFINERY_MAX_VOICE_PASSES", "2"))
MAX_AI_PASSES = int(os.environ.get("RESUME_REFINERY_MAX_AI_PASSES", "2"))

ProgressCallback = Callable[[str], None]
StreamCallback = Callable[[str], None]


class ResumeRefineryOrchestrator:
    """Deterministic coordinator over specialist agents and persistence."""

    def __init__(
        self,
        store: SessionStore | None = None,
        evidence_agent: EvidenceAgent | None = None,
        voice_agent: VoiceAgent | None = None,
        drafting_agent: DraftingAgent | None = None,
        verification_agent: VerificationAgent | None = None,
        repair_agent: RepairAgent | None = None,
    ) -> None:
        self.store = store or SessionStore()
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.voice_agent = voice_agent or VoiceAgent()
        self.drafting_agent = drafting_agent or DraftingAgent()
        self.verification_agent = verification_agent or VerificationAgent()
        self.repair_agent = repair_agent or RepairAgent(self.drafting_agent)

    def create_session_run(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        *,
        skip_review: bool = False,
        allow_unverified: bool = False,
        progress: ProgressCallback | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> OrchestrationResult:
        session = self.store.create(job, career, voice)
        self._progress(progress, f"Session created: {session.session_id}")
        context = self._build_context(career, voice, job, progress)

        docs = DocumentSet()
        for key, label in self._doc_labels().items():
            self._progress(progress, f"Generating {label}...")
            chunks: list[str] = []
            for chunk in self.drafting_agent.stream_document(key, career, voice, job, context):
                chunks.append(chunk)
                if stream_callback:
                    stream_callback(chunk)
            if stream_callback:
                stream_callback("\n")
            docs.set(key, "".join(chunks).strip())

        reviews = self._verify_and_repair(docs, career, voice, job, context, progress=progress)
        session = self.store.save_documents(session, docs)
        exported = self._export(session, docs)

        if skip_review:
            persisted = ReviewBundle(truthfulness=reviews.truthfulness)
        else:
            persisted = reviews
        self.store.save_reviews(session, persisted)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=persisted,
            evidence_pack=context.evidence_pack,
            voice_style_guide=context.voice_style_guide,
            exported_paths={key: str(path) for key, path in exported.items()},
            strict_truth_failed=strict_failed and not allow_unverified,
        )

    def refine_session_run(
        self,
        session_id: str,
        feedback: str,
        *,
        doc: DocumentKey | None = None,
        skip_review: bool = False,
        allow_unverified: bool = False,
        progress: ProgressCallback | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> OrchestrationResult:
        session = self.store.get(session_id)
        career, voice = self.store.load_inputs(session)
        job = session.job_description
        current_docs = self.store.load_documents(session)
        context = self._build_context(career, voice, job, progress)

        keys_to_regen = [doc] if doc else list(self._doc_labels().keys())
        for key in keys_to_regen:
            if key is None:
                continue
            self._progress(progress, f"Regenerating {key.replace('_', ' ')}...")
            previous = current_docs.get(key)
            if stream_callback:
                chunks: list[str] = []
                for chunk in self.drafting_agent.stream_document(
                    key, career, voice, job, context,
                    feedback=feedback, previous_version=previous,
                ):
                    chunks.append(chunk)
                    stream_callback(chunk)
                stream_callback("\n")
                current_docs.set(key, "".join(chunks).strip())
            else:
                regenerated = self.drafting_agent.generate_document(
                    key, career, voice, job, context,
                    feedback=feedback, previous_version=previous,
                )
                current_docs.set(key, regenerated)

        reviews = self._verify_and_repair(
            current_docs,
            career,
            voice,
            job,
            context,
            feedback=feedback,
            progress=progress,
        )
        session = self.store.save_documents(
            session,
            current_docs,
            feedback=feedback,
            docs_regenerated=[key for key in keys_to_regen if key is not None],
        )
        exported = self._export(session, current_docs)

        if skip_review:
            persisted = ReviewBundle(truthfulness=reviews.truthfulness)
        else:
            persisted = reviews
        self.store.save_reviews(session, persisted)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=current_docs,
            reviews=persisted,
            evidence_pack=context.evidence_pack,
            voice_style_guide=context.voice_style_guide,
            exported_paths={key: str(path) for key, path in exported.items()},
            strict_truth_failed=strict_failed and not allow_unverified,
        )

    def review_session_run(
        self,
        session_id: str,
        *,
        version: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        session = self.store.get(session_id)
        career, voice = self.store.load_inputs(session)
        docs = self.store.load_documents(session, version=version)
        context = self._build_context(career, voice, session.job_description, progress)
        reviews = self.verification_agent.review_all(docs, career, voice)
        self.store.save_reviews(session, reviews)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=reviews,
            evidence_pack=context.evidence_pack,
            voice_style_guide=context.voice_style_guide,
            strict_truth_failed=bool(reviews.truthfulness and not reviews.truthfulness.all_supported),
        )

    def _build_context(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        progress: ProgressCallback | None = None,
    ) -> DraftingContext:
        self._progress(progress, "Extracting evidence pack...")
        evidence_pack = self.evidence_agent.build_evidence_pack(career, job)
        self._progress(progress, "Distilling voice guide...")
        style_guide = self.voice_agent.build_style_guide(voice)
        return DraftingContext(evidence_pack=evidence_pack, voice_style_guide=style_guide)

    def _verify_and_repair(
        self,
        docs: DocumentSet,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        *,
        feedback: str | None = None,
        progress: ProgressCallback | None = None,
        max_truth_passes: int = MAX_TRUTH_PASSES,
        max_voice_passes: int = MAX_VOICE_PASSES,
        max_ai_passes: int = MAX_AI_PASSES,
    ) -> ReviewBundle:
        import logging

        # Track suggestions from previous passes to avoid repeating the same fixes
        previous_truth_suggestions: list[str] = []
        previous_voice_suggestions: list[str] = []
        previous_ai_suggestions: list[str] = []

        # --- Truthfulness loop ---
        truth = None
        for _ in range(max_truth_passes):
            self._progress(progress, "Running strict truthfulness review...")
            try:
                truth = self.verification_agent.review_truthfulness(docs, career)
            except Exception as exc:
                logging.warning("Truthfulness review failed (%s); skipping repair loop.", exc)
                self._progress(progress, f"[yellow]Truth review skipped: {exc}[/yellow]")
                break
            self._progress(progress, self._summarise_truth(truth))
            if truth.all_supported:
                break
            self._progress(progress, "Repairing unsupported claims...")
            self.repair_agent.repair_documents(
                docs, truth, career, voice, job, context,
                feedback=feedback,
                previous_suggestions=previous_truth_suggestions,
            )
            previous_truth_suggestions.extend(truth.suggestions)

        # --- Voice-match loop ---
        voice_result = None
        for _ in range(max_voice_passes):
            self._progress(progress, "Running voice-match review...")
            try:
                voice_result = self.verification_agent.review_voice(docs, voice)
            except Exception as exc:
                logging.warning("Voice review failed (%s); skipping repair loop.", exc)
                self._progress(progress, f"[yellow]Voice review skipped: {exc}[/yellow]")
                break
            self._progress(progress, self._summarise_voice(voice_result))
            if voice_result.overall_match == "strong":
                break
            self._progress(progress, "Repairing voice-match issues...")
            self.repair_agent.repair_voice(
                docs, voice_result, career, voice, job, context,
                feedback=feedback,
                previous_suggestions=previous_voice_suggestions,
            )
            previous_voice_suggestions.extend(voice_result.suggestions)

        # --- AI-detection loop ---
        ai_result = None
        for _ in range(max_ai_passes):
            self._progress(progress, "Running AI-detection review...")
            try:
                ai_result = self.verification_agent.review_ai_detection(docs)
            except Exception as exc:
                logging.warning("AI-detection review failed (%s); skipping repair loop.", exc)
                self._progress(progress, f"[yellow]AI-detection review skipped: {exc}[/yellow]")
                break
            self._progress(progress, self._summarise_ai(ai_result))
            # Exit when no document has flagged phrases, regardless of aggregate risk_level
            has_flags = (
                ai_result.cover_letter_flags
                or ai_result.resume_flags
                or ai_result.interview_guide_flags
            )
            if not has_flags:
                break
            self._progress(progress, "Repairing AI-flagged content...")
            self.repair_agent.repair_ai_detection(
                docs, ai_result, career, voice, job, context,
                feedback=feedback,
                previous_suggestions=previous_ai_suggestions,
            )
            previous_ai_suggestions.extend(ai_result.suggestions)

        # --- Post-repair truthfulness recheck ---
        # Voice and AI repairs may introduce fabricated claims;
        # re-verify truthfulness on the final documents.
        if truth and not truth.all_supported or (voice_result or ai_result):
            self._progress(progress, "Running final truthfulness recheck...")
            try:
                final_truth = self.verification_agent.review_truthfulness(docs, career)
                self._progress(progress, self._summarise_truth(final_truth))
                truth = final_truth
            except Exception as exc:
                logging.warning("Final truth recheck failed (%s); keeping previous result.", exc)
                self._progress(progress, f"[yellow]Final truth recheck skipped: {exc}[/yellow]")

        return ReviewBundle(
            truthfulness=truth,
            voice=voice_result,
            ai_detection=ai_result,
        )

    def _export(self, session: Session, docs: DocumentSet) -> dict[str, Path]:
        version_dir = self.store.session_dir(session.session_id) / f"v{session.current_version}"
        return export_document_set(docs, version_dir)

    def _progress(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    # ------------------------------------------------------------------
    # Review-result summaries emitted via the progress callback
    # ------------------------------------------------------------------

    def _summarise_truth(self, truth: TruthfulnessResult) -> str:
        if truth.all_supported:
            return "[green]Truthfulness: ALL SUPPORTED[/green]"
        parts = ["[red]Truthfulness: UNSUPPORTED CLAIMS DETECTED[/red]"]
        for label, doc in [("Cover Letter", truth.cover_letter),
                           ("Resume", truth.resume),
                           ("Interview Guide", truth.interview_guide)]:
            if not doc.pass_strict:
                claims = doc.unsupported_claims
                if claims:
                    parts.append(f"  {label}: {len(claims)} unsupported")
                    for claim in claims:
                        parts.append(f"    • {claim}")
                else:
                    parts.append(f"  {label}: strict check failed (no specific claims listed)")
                if doc.evidence_examples:
                    parts.append(f"  {label} evidence examples:")
                    for ex in doc.evidence_examples:
                        parts.append(f"    ✓ {ex}")
        if truth.suggestions:
            parts.append("  Suggestions:")
            for s in truth.suggestions:
                parts.append(f"    • {s}")
        return "\n".join(parts)

    def _summarise_voice(self, voice: VoiceReviewResult) -> str:
        color = {"strong": "green", "moderate": "yellow", "weak": "red"}[voice.overall_match]
        parts = [f"[{color}]Voice match: {voice.overall_match.upper()}[/{color}]"]
        per_doc_issues: dict[str, list[str]] = {
            "Cover Letter": voice.cover_letter_issues,
            "Resume": voice.resume_issues,
            "Interview Guide": voice.interview_guide_issues,
        }
        per_doc_suggestions: dict[str, list[str]] = {
            "Cover Letter": voice.cover_letter_suggestions,
            "Resume": voice.resume_suggestions,
            "Interview Guide": voice.interview_guide_suggestions,
        }
        for label, match, assessment in [
            ("Cover Letter", voice.cover_letter_match, voice.cover_letter_assessment),
            ("Resume", voice.resume_match, voice.resume_assessment),
            ("Interview Guide", voice.interview_guide_match, voice.interview_guide_assessment),
        ]:
            mc = {"strong": "green", "moderate": "yellow", "weak": "red"}[match]
            parts.append(f"  {label}: [{mc}]{match}[/{mc}]")
            parts.append(f"    {assessment}")
            issues = per_doc_issues.get(label, [])
            if issues:
                for issue in issues:
                    parts.append(f"    • Issue: {issue}")
            suggestions = per_doc_suggestions.get(label, [])
            if suggestions:
                for s in suggestions:
                    parts.append(f"    → {s}")
        return "\n".join(parts)

    def _summarise_ai(self, ai: AIDetectionResult) -> str:
        color = {"low": "green", "medium": "yellow", "high": "red"}[ai.risk_level]
        parts = [f"[{color}]AI-detection risk: {ai.risk_level.upper()}[/{color}]"]
        for label, flags in [("Cover Letter", ai.cover_letter_flags),
                             ("Resume", ai.resume_flags),
                             ("Interview Guide", ai.interview_guide_flags)]:
            if flags:
                parts.append(f"  {label}:")
                for f in flags:
                    parts.append(f'    • "{f}"')
        if ai.suggestions:
            parts.append("  Suggestions:")
            for s in ai.suggestions:
                parts.append(f"    • {s}")
        return "\n".join(parts)

    def _doc_labels(self) -> dict[DocumentKey, str]:
        return {
            "cover_letter": "Cover Letter",
            "resume": "Resume",
            "interview_guide": "Interview Guide",
        }
