"""Workflow orchestrator for bounded multi-agent execution."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from .exporters import export_document_set
from .models import (
    AIDetectionResult,
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    ExemptedPhrases,
    HiringManagerReview,
    OrchestrationResult,
    RepairPassResult,
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
MAX_WORKERS = int(os.environ.get("RESUME_REFINERY_MAX_WORKERS", "1"))

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
        output_dir: Path | None = None,
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
        if skip_review:
            self._progress(progress, "  Truthfulness review (3 LLM calls)...")
            try:
                truth = self.verification_agent.review_truthfulness(docs, career, job)
            except Exception as exc:
                logging.warning("Truthfulness review failed (%s)", exc)
                truth = None
            reviews: ReviewBundle = ReviewBundle(truthfulness=truth)
            repair_passes: list[RepairPassResult] = []
            exempted = ExemptedPhrases()
        else:
            reviews, repair_passes, exempted = self._verify_and_repair(
                docs, career, voice, job, context, progress=progress,
                on_repair_pass=lambda p, d, r: repair_snapshots.append((p, d.model_copy(deep=True), r)),
            )
        session = self.store.save_documents(session, docs)
        self.store.save_context(session, context)
        for pass_num, snap, pass_reviews in repair_snapshots:
            self.store.save_repair_pass(session, pass_num, snap, pass_reviews)
        if exempted.claims or exempted.ai_phrases or exempted.voice_issues:
            self.store.save_suppressions(session, exempted)
        exported = self._export(session, docs, output_dir=output_dir)

        # --- Hiring manager review (runs after repair, before final save) ---
        self._progress(progress, "Running hiring-manager review (1 LLM call)...")
        try:
            hm_review = self.verification_agent.review_hiring_manager(docs, job)
            reviews = reviews.model_copy(update={"hiring_manager": hm_review})
            self._progress(progress, self._summarise_hiring_manager(hm_review))
        except Exception as exc:
            logging.warning("Hiring-manager review failed (%s)", exc)
            self._progress(progress, f"[yellow]Hiring-manager review skipped: {exc}[/yellow]")

        self.store.save_reviews(session, reviews)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=reviews,
            repair_passes=repair_passes,
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
        output_dir: Path | None = None,
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
        if skip_review:
            self._progress(progress, "  Truthfulness review (3 LLM calls)...")
            try:
                truth = self.verification_agent.review_truthfulness(current_docs, career, job)
            except Exception as exc:
                logging.warning("Truthfulness review failed (%s)", exc)
                truth = None
            reviews: ReviewBundle = ReviewBundle(truthfulness=truth)
            repair_passes: list[RepairPassResult] = []
            exempted = ExemptedPhrases()
        else:
            reviews, repair_passes, exempted = self._verify_and_repair(
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
        if exempted.claims or exempted.ai_phrases or exempted.voice_issues:
            self.store.save_suppressions(session, exempted)
        exported = self._export(session, current_docs, output_dir=output_dir)

        # --- Hiring manager review (runs after repair, before final save) ---
        self._progress(progress, "Running hiring-manager review (1 LLM call)...")
        try:
            hm_review = self.verification_agent.review_hiring_manager(current_docs, job)
            reviews = reviews.model_copy(update={"hiring_manager": hm_review})
            self._progress(progress, self._summarise_hiring_manager(hm_review))
        except Exception as exc:
            logging.warning("Hiring-manager review failed (%s)", exc)
            self._progress(progress, f"[yellow]Hiring-manager review skipped: {exc}[/yellow]")

        self.store.save_reviews(session, reviews)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=current_docs,
            reviews=reviews,
            repair_passes=repair_passes,
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
        job = session.job_description
        context = self._build_context(career, voice, job, progress)
        reviews = self.verification_agent.review_all(docs, career, voice, job)
        try:
            hm_review = self.verification_agent.review_hiring_manager(docs, job)
            reviews = reviews.model_copy(update={"hiring_manager": hm_review})
        except Exception as exc:
            logging.warning("Hiring-manager review failed (%s)", exc)
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
    ) -> tuple[ReviewBundle, list[RepairPassResult], ExemptedPhrases]:
        import logging

        repair_results: list[RepairPassResult] = []

        truth = None
        voice_result = None
        ai_result = None

        # Per-reviewer suppression sets — accumulated across all repair passes.
        # Each reviewer has its own independent set so a voice false positive
        # cannot accidentally suppress a truthfulness finding (and vice versa).
        suppressed_claims: set[str] = set()
        suppressed_ai_phrases: set[str] = set()
        suppressed_voice_issues: set[str] = set()

        for pass_num in range(max_passes):
            self._progress(progress, f"─── Review Pass {pass_num + 1}/{max_passes} ───")

            # --- Run all three reviews (parallel when MAX_WORKERS > 1) ---
            self._progress(progress, "  Running reviews (7 LLM calls)...")

            def _run_truth():
                try:
                    return self.verification_agent.review_truthfulness(docs, career, job)
                except Exception as exc:
                    logging.warning("Truthfulness review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Truth review skipped: {exc}[/yellow]")
                    return None

            def _run_voice():
                try:
                    return self.verification_agent.review_voice(docs, voice)
                except Exception as exc:
                    logging.warning("Voice review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Voice review skipped: {exc}[/yellow]")
                    return None

            def _run_ai():
                try:
                    return self.verification_agent.review_ai_detection(docs)
                except Exception as exc:
                    logging.warning("AI-detection review failed (%s)", exc)
                    self._progress(progress, f"[yellow]AI-detection review skipped: {exc}[/yellow]")
                    return None

            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 3)) as pool:
                truth_future = pool.submit(_run_truth)
                voice_future = pool.submit(_run_voice)
                ai_future = pool.submit(_run_ai)
                truth = truth_future.result()
                voice_result = voice_future.result()
                ai_result = ai_future.result()

            # Filter out items accepted as false positives in earlier passes.
            truth, voice_result, ai_result = self._apply_suppressions(
                truth, voice_result, ai_result,
                suppressed_claims, suppressed_ai_phrases, suppressed_voice_issues,
            )

            # --- Summarise all three ---
            if truth:
                self._progress(progress, self._summarise_truth(truth))
            if voice_result:
                self._progress(progress, self._summarise_voice(voice_result))
            if ai_result:
                self._progress(progress, self._summarise_ai(ai_result))

            # --- Check if all passing ---
            # Truthfulness is always strict (no relaxation).
            truth_ok = truth is None or truth.all_supported

            # Voice accepts "moderate" from pass 0 to avoid unnecessary rewrites
            # that destabilise truthfulness.  On late passes, AI threshold relaxes.
            voice_ok = voice_result is None or voice_result.overall_match in ("strong", "moderate")

            is_late_pass = pass_num >= _RELAXED_PASS_START
            if is_late_pass:
                total_ai_flags = (
                    len(ai_result.cover_letter_flags)
                    + len(ai_result.resume_flags)
                ) if ai_result else 0
                ai_ok = ai_result is None or total_ai_flags <= _AI_FLAG_TOLERANCE_LATE
            else:
                ai_ok = ai_result is None or not any([
                    ai_result.cover_letter_flags,
                    ai_result.resume_flags,
                ])

            if truth_ok and voice_ok and ai_ok:
                break

            # --- Unified repair ---
            self._progress(progress, "  Repairing documents (up to 3 LLM calls, thinking enabled)...")
            repair_pass = self.repair_agent.repair_unified(
                docs, truth, voice_result, ai_result,
                career, voice, job, context,
                feedback=feedback,
            )
            repair_results.append(repair_pass)
            # Accumulate per-reviewer acceptances into suppression sets.
            suppressed_claims.update(repair_pass.accepted_claims)
            suppressed_ai_phrases.update(repair_pass.accepted_ai_phrases)
            suppressed_voice_issues.update(repair_pass.accepted_voice_issues)
            if repair_pass.edits:
                self._progress(progress, self._summarise_repair(repair_pass))
            # Emit explicit output for every item accepted as a false positive.
            if repair_pass.accepted_claims or repair_pass.accepted_ai_phrases or repair_pass.accepted_voice_issues:
                self._progress(progress, self._summarise_acceptances(repair_pass))

            # Snapshot documents and reviews after this repair pass for auditing.
            if on_repair_pass is not None:
                pass_reviews = ReviewBundle(
                    truthfulness=truth,
                    voice=voice_result,
                    ai_detection=ai_result,
                )
                on_repair_pass(pass_num, docs, pass_reviews)

        return ReviewBundle(
            truthfulness=truth,
            voice=voice_result,
            ai_detection=ai_result,
        ), repair_results, ExemptedPhrases(
            claims=sorted(suppressed_claims),
            ai_phrases=sorted(suppressed_ai_phrases),
            voice_issues=sorted(suppressed_voice_issues),
        )

    def _export(
        self,
        session: Session,
        docs: DocumentSet,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        # Always export to the session version directory for versioning integrity
        version_dir = self.store.session_dir(session.session_id) / f"v{session.current_version}"
        export_document_set(docs, version_dir)
        # If the user specified a separate output directory, also copy there
        if output_dir is not None:
            return export_document_set(docs, output_dir)
        return export_document_set(docs, version_dir)

    def _progress(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    # ------------------------------------------------------------------
    # Per-reviewer false-positive suppression
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_suppressions(
        truth: TruthfulnessResult | None,
        voice_result: VoiceReviewResult | None,
        ai_result: AIDetectionResult | None,
        suppressed_claims: set[str],
        suppressed_ai_phrases: set[str],
        suppressed_voice_issues: set[str],
    ) -> tuple[TruthfulnessResult | None, VoiceReviewResult | None, AIDetectionResult | None]:
        """Return copies of review results with suppressed items removed.

        Each reviewer has its own independent suppression set so that accepting
        a voice false positive cannot silence a truthfulness finding (and
        vice versa).
        """
        # --- Truthfulness ---
        filtered_truth = truth
        if truth and suppressed_claims:
            def _filter_doc(doc: DocumentTruthResult) -> DocumentTruthResult:
                remaining = [c for c in doc.unsupported_claims if c not in suppressed_claims]
                return doc.model_copy(update={"unsupported_claims": remaining, "pass_strict": not remaining})
            cl = _filter_doc(truth.cover_letter)
            res = _filter_doc(truth.resume)
            ig = _filter_doc(truth.interview_guide)
            filtered_truth = truth.model_copy(update={
                "cover_letter": cl,
                "resume": res,
                "interview_guide": ig,
                "all_supported": cl.pass_strict and res.pass_strict and ig.pass_strict,
            })

        # --- AI detection ---
        filtered_ai = ai_result
        if ai_result and suppressed_ai_phrases:
            cl_flags = [f for f in ai_result.cover_letter_flags if f not in suppressed_ai_phrases]
            res_flags = [f for f in ai_result.resume_flags if f not in suppressed_ai_phrases]
            ig_flags = [f for f in ai_result.interview_guide_flags if f not in suppressed_ai_phrases]
            total = len(cl_flags) + len(res_flags) + len(ig_flags)
            risk = "low" if total <= 1 else "medium" if total <= 3 else "high"
            filtered_ai = ai_result.model_copy(update={
                "cover_letter_flags": cl_flags,
                "resume_flags": res_flags,
                "interview_guide_flags": ig_flags,
                "risk_level": risk,
            })

        # --- Voice ---
        filtered_voice = voice_result
        if voice_result and suppressed_voice_issues:
            filtered_voice = voice_result.model_copy(update={
                "cover_letter_issues": [i for i in voice_result.cover_letter_issues if i not in suppressed_voice_issues],
                "resume_issues": [i for i in voice_result.resume_issues if i not in suppressed_voice_issues],
                "interview_guide_issues": [i for i in voice_result.interview_guide_issues if i not in suppressed_voice_issues],
                "specific_issues": [i for i in voice_result.specific_issues if i not in suppressed_voice_issues],
            })

        return filtered_truth, filtered_voice, filtered_ai

    # ------------------------------------------------------------------
    # Review-result summaries emitted via the progress callback
    # ------------------------------------------------------------------

    def _summarise_truth(self, truth: TruthfulnessResult) -> str:
        if truth.all_supported:
            parts = ["[green]Truthfulness: ALL SUPPORTED[/green]"]
        else:
            parts = ["[red]Truthfulness: UNSUPPORTED CLAIMS DETECTED[/red]"]
        for label, doc in [("Cover Letter", truth.cover_letter),
                           ("Resume", truth.resume),
                           ("Interview Guide", truth.interview_guide)]:
            status = "[green]✓[/green]" if doc.pass_strict else f"[red]✗ ({len(doc.unsupported_claims)} unsupported)[/red]"
            parts.append(f"  {label}: {status}")
            if not doc.pass_strict:
                for claim in doc.unsupported_claims:
                    parts.append(f"    • {claim}")
        return "\n".join(parts)

    def _summarise_voice(self, voice: VoiceReviewResult) -> str:
        color = {"strong": "green", "moderate": "yellow", "weak": "red"}[voice.overall_match]
        parts = [f"[{color}]Voice match: {voice.overall_match.upper()}[/{color}]"]
        for label, match, issues in [
            ("Cover Letter", voice.cover_letter_match, voice.cover_letter_issues),
            ("Resume", voice.resume_match, voice.resume_issues),
            ("Interview Guide", voice.interview_guide_match, voice.interview_guide_issues),
        ]:
            mc = {"strong": "green", "moderate": "yellow", "weak": "red"}[match]
            parts.append(f"  {label}: [{mc}]{match}[/{mc}]")
            if match != "strong" and issues:
                for issue in issues:
                    parts.append(f"    • {issue}")
        return "\n".join(parts)

    def _summarise_ai(self, ai: AIDetectionResult) -> str:
        color = {"low": "green", "medium": "yellow", "high": "red"}[ai.risk_level]
        parts = [f"[{color}]AI-detection risk: {ai.risk_level.upper()}[/{color}]"]
        for label, flags in [
            ("Cover Letter", ai.cover_letter_flags),
            ("Resume", ai.resume_flags),
            ("Interview Guide", ai.interview_guide_flags),
        ]:
            if flags:
                parts.append(f"  {label}: {len(flags)} flag(s)")
                for flag in flags:
                    parts.append(f'    • "{flag}"')
        return "\n".join(parts)

    def _summarise_hiring_manager(self, hm: HiringManagerReview) -> str:
        pct = hm.advance_likelihood
        color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
        parts = [f"[{color}]Hiring-manager advance likelihood: {pct}%[/{color}]"]
        if hm.summary:
            parts.append(f"  {hm.summary}")
        if hm.strengths:
            parts.append("  Strengths:")
            for s in hm.strengths:
                parts.append(f"    • {s}")
        if hm.concerns:
            parts.append("  Concerns:")
            for c in hm.concerns:
                parts.append(f"    • {c}")
        if hm.improvements:
            parts.append("  Suggested improvements:")
            for imp in hm.improvements:
                parts.append(f"    [{imp.impact.upper()}] ({imp.area}) {imp.suggestion}")
        return "\n".join(parts)

    def _summarise_repair(self, repair_pass: RepairPassResult) -> str:
        doc_labels = self._doc_labels()
        parts = ["[bold]Repair edits applied:[/bold]"]
        for key, edits in repair_pass.edits.items():
            label = doc_labels.get(key, key)
            parts.append(f"  {label}: {len(edits)} edit(s)")
            for edit in edits:
                parts.append(f'    [red]- "{edit.find}"[/red]')
                parts.append(f'    [green]+ "{edit.replace}"[/green]')
                if edit.reason:
                    parts.append(f"      ({edit.reason})")
        return "\n".join(parts)

    def _summarise_acceptances(self, repair_pass: RepairPassResult) -> str:
        """Build a Rich-tagged summary of all phrases/claims accepted as false positives."""
        parts = ["[bold cyan]Repair agent accepted the following as false positives (will be exempted from future passes):[/bold cyan]"]
        if repair_pass.accepted_claims:
            parts.append("  [cyan]Truthfulness claims (accepted as supported by career evidence):[/cyan]")
            for claim in repair_pass.accepted_claims:
                parts.append(f'    [cyan]✓ "{claim}"[/cyan]')
        if repair_pass.accepted_ai_phrases:
            parts.append("  [cyan]AI-detection flags (accepted as natural human language):[/cyan]")
            for phrase in repair_pass.accepted_ai_phrases:
                parts.append(f'    [cyan]✓ "{phrase}"[/cyan]')
        if repair_pass.accepted_voice_issues:
            parts.append("  [cyan]Voice-match issues (accepted as reviewer false positives):[/cyan]")
            for issue in repair_pass.accepted_voice_issues:
                parts.append(f'    [cyan]✓ "{issue}"[/cyan]')
        return "\n".join(parts)

    def _doc_labels(self) -> dict[DocumentKey, str]:
        return {
            "cover_letter": "Cover Letter",
            "resume": "Resume",
            "interview_guide": "Interview Guide",
        }
