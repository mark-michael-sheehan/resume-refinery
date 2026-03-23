"""Workflow orchestrator for bounded multi-agent execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from .exporters import export_document_set
from .models import (
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    OrchestrationResult,
    ReviewBundle,
    Session,
    VoiceProfile,
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
            if truth.all_supported:
                break
            self._progress(progress, "Repairing unsupported claims...")
            self.repair_agent.repair_documents(docs, truth, career, voice, job, context, feedback=feedback)

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
            if voice_result.overall_match == "strong":
                break
            self._progress(progress, "Repairing voice-match issues...")
            self.repair_agent.repair_voice(docs, voice_result, career, voice, job, context, feedback=feedback)

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
            # Exit when no document has flagged phrases, regardless of aggregate risk_level
            has_flags = (
                ai_result.cover_letter_flags
                or ai_result.resume_flags
                or ai_result.interview_guide_flags
            )
            if not has_flags:
                break
            self._progress(progress, "Repairing AI-flagged content...")
            self.repair_agent.repair_ai_detection(docs, ai_result, career, voice, job, context, feedback=feedback)

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

    def _doc_labels(self) -> dict[DocumentKey, str]:
        return {
            "cover_letter": "Cover Letter",
            "resume": "Resume",
            "interview_guide": "Interview Guide",
        }
