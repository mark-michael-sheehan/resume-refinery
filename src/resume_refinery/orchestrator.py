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

MAX_REPAIR_PASSES = int(os.environ.get("RESUME_REFINERY_MAX_REPAIR_PASSES", "3"))

# On later passes, relax voice/AI thresholds to help convergence.
# Truthfulness always stays strict.
_AI_FLAG_TOLERANCE_LATE = int(os.environ.get("RESUME_REFINERY_AI_FLAG_TOLERANCE", "2"))
# 0-based pass index at which relaxed thresholds kick in (default: pass 2, i.e. the second pass).
_RELAXED_PASS_START = int(os.environ.get("RESUME_REFINERY_RELAXED_PASS_START", "1"))

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
            self._progress(progress, f"Generating {label} (model is thinking, output appears after reasoning)...")
            chunks: list[str] = []
            for chunk in self.drafting_agent.stream_document(key, career, voice, job, context):
                chunks.append(chunk)
                if stream_callback:
                    stream_callback(chunk)
            if stream_callback:
                stream_callback("\n")
            text = "".join(chunks).strip()
            if not text:
                raise ValueError(
                    f"'{label}' generated empty content — the model may have "
                    "exhausted its context window on reasoning. Try raising "
                    "RESUME_REFINERY_NUM_CTX in your .env."
                )
            docs.set(key, text)

        repair_snapshots: list[tuple[int, DocumentSet, ReviewBundle]] = []
        reviews = self._verify_and_repair(
            docs, career, voice, job, context, progress=progress,
            on_repair_pass=lambda p, d, r: repair_snapshots.append((p, d.model_copy(deep=True), r)),
        )
        session = self.store.save_documents(session, docs)
        self.store.save_context(session, context)
        for pass_num, snap, pass_reviews in repair_snapshots:
            self.store.save_repair_pass(session, pass_num, snap, pass_reviews)
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
            label = self._doc_labels()[key]
            self._progress(progress, f"Regenerating {label} (model is thinking, output appears after reasoning)...")
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
                text = "".join(chunks).strip()
                if not text:
                    raise ValueError(
                        f"'{label}' generated empty content — the model may have "
                        "exhausted its context window on reasoning. Try raising "
                        "RESUME_REFINERY_NUM_CTX in your .env."
                    )
                current_docs.set(key, text)
            else:
                regenerated = self.drafting_agent.generate_document(
                    key, career, voice, job, context,
                    feedback=feedback, previous_version=previous,
                )
                current_docs.set(key, regenerated)

        repair_snapshots: list[tuple[int, DocumentSet, ReviewBundle]] = []
        reviews = self._verify_and_repair(
            current_docs,
            career,
            voice,
            job,
            context,
            feedback=feedback,
            progress=progress,
            on_repair_pass=lambda p, d, r: repair_snapshots.append((p, d.model_copy(deep=True), r)),
        )
        session = self.store.save_documents(
            session,
            current_docs,
            feedback=feedback,
            docs_regenerated=[key for key in keys_to_regen if key is not None],
        )
        self.store.save_context(session, context)
        for pass_num, snap, pass_reviews in repair_snapshots:
            self.store.save_repair_pass(session, pass_num, snap, pass_reviews)
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
        max_passes: int = MAX_REPAIR_PASSES,
        on_repair_pass: Callable[[int, DocumentSet, ReviewBundle], None] | None = None,
    ) -> ReviewBundle:
        import logging

        previous_truth_suggestions: list[str] = []
        previous_voice_suggestions: list[str] = []
        previous_ai_suggestions: list[str] = []

        truth = None
        voice_result = None
        ai_result = None

        for pass_num in range(max_passes):
            self._progress(progress, f"─── Review Pass {pass_num + 1}/{max_passes} ───")

            # --- Run all three reviews ---
            self._progress(progress, "  Truthfulness review (3 LLM calls, thinking enabled)...")
            try:
                truth = self.verification_agent.review_truthfulness(docs, career)
            except Exception as exc:
                logging.warning("Truthfulness review failed (%s)", exc)
                self._progress(progress, f"[yellow]Truth review skipped: {exc}[/yellow]")
                truth = None

            self._progress(progress, "  Voice review (2 LLM calls)...")
            try:
                voice_result = self.verification_agent.review_voice(docs, voice)
            except Exception as exc:
                logging.warning("Voice review failed (%s)", exc)
                self._progress(progress, f"[yellow]Voice review skipped: {exc}[/yellow]")
                voice_result = None

            self._progress(progress, "  AI-detection review (2 LLM calls)...")
            try:
                ai_result = self.verification_agent.review_ai_detection(docs)
            except Exception as exc:
                logging.warning("AI-detection review failed (%s)", exc)
                self._progress(progress, f"[yellow]AI-detection review skipped: {exc}[/yellow]")
                ai_result = None

            # --- Summarise all three ---
            if truth:
                self._progress(progress, self._summarise_truth(truth, previous_truth_suggestions))
            if voice_result:
                self._progress(progress, self._summarise_voice(voice_result, previous_voice_suggestions))
            if ai_result:
                self._progress(progress, self._summarise_ai(ai_result, previous_ai_suggestions))

            # --- Check if all passing ---
            # Truthfulness is always strict (no relaxation).
            truth_ok = truth is None or truth.all_supported

            # On pass RELAXED_PASS_START+, relax voice and AI thresholds to help convergence.
            is_late_pass = pass_num >= _RELAXED_PASS_START
            if is_late_pass:
                voice_ok = voice_result is None or voice_result.overall_match in ("strong", "moderate")
                total_ai_flags = (
                    len(ai_result.cover_letter_flags)
                    + len(ai_result.resume_flags)
                ) if ai_result else 0
                ai_ok = ai_result is None or total_ai_flags <= _AI_FLAG_TOLERANCE_LATE
            else:
                voice_ok = voice_result is None or voice_result.overall_match == "strong"
                ai_ok = ai_result is None or not any([
                    ai_result.cover_letter_flags,
                    ai_result.resume_flags,
                ])

            if truth_ok and voice_ok and ai_ok:
                break

            # --- Unified repair ---
            self._progress(progress, "  Repairing documents (up to 3 LLM calls, thinking enabled)...")
            self.repair_agent.repair_unified(
                docs, truth, voice_result, ai_result,
                career, voice, job, context,
                feedback=feedback,
                previous_suggestions=(
                    previous_truth_suggestions
                    + previous_voice_suggestions
                    + previous_ai_suggestions
                ),
            )

            # Snapshot documents and reviews after this repair pass for auditing.
            if on_repair_pass is not None:
                pass_reviews = ReviewBundle(
                    truthfulness=truth,
                    voice=voice_result,
                    ai_detection=ai_result,
                )
                on_repair_pass(pass_num, docs, pass_reviews)

            # Collect suggestions for next pass
            if truth:
                for doc in (truth.cover_letter, truth.resume, truth.interview_guide):
                    previous_truth_suggestions.extend(doc.suggestions)
            if voice_result and voice_result.suggestions:
                previous_voice_suggestions.extend(voice_result.suggestions)
            if ai_result:
                previous_ai_suggestions.extend(ai_result.cover_letter_suggestions)
                previous_ai_suggestions.extend(ai_result.resume_suggestions)

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

    def _summarise_truth(self, truth: TruthfulnessResult, previous_suggestions: list[str] | None = None) -> str:
        if truth.all_supported:
            parts = ["[green]Truthfulness: ALL SUPPORTED[/green]"]
        else:
            parts = ["[red]Truthfulness: UNSUPPORTED CLAIMS DETECTED[/red]"]
        for label, doc in [("Cover Letter", truth.cover_letter),
                           ("Resume", truth.resume),
                           ("Interview Guide", truth.interview_guide)]:
            status = "[green]✓[/green]" if doc.pass_strict else f"[red]✗ ({len(doc.unsupported_claims)} unsupported)[/red]"
            parts.append(f"  {label}: {status}")
            if doc.suggestions:
                for s in doc.suggestions:
                    parts.append(f"    • {s}")
        if previous_suggestions:
            parts.append("  Previously attempted:")
            for s in previous_suggestions:
                parts.append(f"    ◦ {s}")
        return "\n".join(parts)

    def _summarise_voice(self, voice: VoiceReviewResult, previous_suggestions: list[str] | None = None) -> str:
        color = {"strong": "green", "moderate": "yellow", "weak": "red"}[voice.overall_match]
        parts = [f"[{color}]Voice match: {voice.overall_match.upper()}[/{color}]"]
        per_doc_suggestions: dict[str, list[str]] = {
            "Cover Letter": voice.cover_letter_suggestions,
            "Resume": voice.resume_suggestions,
            "Interview Guide": voice.interview_guide_suggestions,
        }
        for label, match in [
            ("Cover Letter", voice.cover_letter_match),
            ("Resume", voice.resume_match),
            ("Interview Guide", voice.interview_guide_match),
        ]:
            mc = {"strong": "green", "moderate": "yellow", "weak": "red"}[match]
            parts.append(f"  {label}: [{mc}]{match}[/{mc}]")
            suggestions = per_doc_suggestions.get(label, [])
            if suggestions:
                for s in suggestions:
                    parts.append(f"    • {s}")
        if previous_suggestions:
            parts.append("  Previously attempted:")
            for s in previous_suggestions:
                parts.append(f"    ◦ {s}")
        return "\n".join(parts)

    def _summarise_ai(self, ai: AIDetectionResult, previous_suggestions: list[str] | None = None) -> str:
        color = {"low": "green", "medium": "yellow", "high": "red"}[ai.risk_level]
        parts = [f"[{color}]AI-detection risk: {ai.risk_level.upper()}[/{color}]"]
        per_doc_suggestions: dict[str, list[str]] = {
            "Cover Letter": ai.cover_letter_suggestions,
            "Resume": ai.resume_suggestions,
        }
        per_doc_flags: dict[str, list[str]] = {
            "Cover Letter": ai.cover_letter_flags,
            "Resume": ai.resume_flags,
        }
        for label in ("Cover Letter", "Resume"):
            flags = per_doc_flags.get(label, [])
            suggestions = per_doc_suggestions.get(label, [])
            if flags or suggestions:
                flag_count = len(flags)
                parts.append(f"  {label}: {flag_count} flag(s)")
                for s in suggestions:
                    parts.append(f"    • {s}")
        if previous_suggestions:
            parts.append("  Previously attempted:")
            for s in previous_suggestions:
                parts.append(f"    ◦ {s}")
        return "\n".join(parts)

    def _doc_labels(self) -> dict[DocumentKey, str]:
        return {
            "cover_letter": "Cover Letter",
            "resume": "Resume",
            "interview_guide": "Interview Guide",
        }
