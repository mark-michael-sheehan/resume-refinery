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
    ALL_DOC_KEYS,
    AIDetectionResult,
    ATSKeywordResult,
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    ExemptedPhrases,
    GrammarResult,
    HiringManagerReview,
    OrchestrationResult,
    RelevancePruningResult,
    RepairPassResult,
    ReviewBundle,
    Session,
    TruthfulnessResult,
    VoiceProfile,
    VoiceReviewResult,
    JobDescription,
)
from .session import SessionStore
from .specialist_agents import DraftingAgent, NarrativeAgent, RepairAgent, VerificationAgent, VoiceAgent

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
        narrative_agent: NarrativeAgent | None = None,
        voice_agent: VoiceAgent | None = None,
        drafting_agent: DraftingAgent | None = None,
        verification_agent: VerificationAgent | None = None,
        repair_agent: RepairAgent | None = None,
    ) -> None:
        self.store = store or SessionStore()
        self.narrative_agent = narrative_agent or NarrativeAgent()
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
        selected_docs: list[DocumentKey] | None = None,
        progress: ProgressCallback | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> OrchestrationResult:
        active_docs = selected_docs or list(ALL_DOC_KEYS)
        session = self.store.create(job, career, voice, selected_docs=active_docs)
        self._progress(progress, f"Session created: {session.session_id}")
        context = self._build_context(career, voice, job, progress)

        docs = DocumentSet()
        for key, label in self._doc_labels(active_docs).items():
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
                    f"'{label}' generated empty content â€” the model may have "
                    "exhausted its context window on reasoning. Try raising "
                    "RESUME_REFINERY_NUM_CTX in your .env."
                )
            docs.set(key, text)

        # Save documents + context immediately after generation so results
        # are available on disk before the (potentially long) review loop.
        session = self.store.save_documents(session, docs, docs_regenerated=active_docs)
        self.store.save_context(session, context)
        self._export(session, docs, output_dir=output_dir)

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
            def _on_repair_pass(p: int, d: DocumentSet, r: ReviewBundle) -> None:
                self.store.save_repair_pass(session, p, d.model_copy(deep=True), r)
                self.store.update_documents(session, d)
                self._export(session, d, output_dir=output_dir)

            reviews, repair_passes, exempted = self._verify_and_repair(
                docs, career, voice, job, context, progress=progress,
                on_repair_pass=_on_repair_pass,
            )
        if exempted.claims or exempted.ai_phrases or exempted.voice_issues or exempted.hm_issues or exempted.pruning_issues or exempted.ats_issues or exempted.grammar_issues:
            self.store.save_suppressions(session, exempted)
        # Final export with the fully-repaired documents.
        exported = self._export(session, docs, output_dir=output_dir)

        self.store.save_reviews(session, reviews)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=reviews,
            repair_passes=repair_passes,
            narrative=context.narrative,
            voice_style_guide=context.voice_style_guide,
            exported_paths={key: str(path) for key, path in exported.items()},
            strict_truth_failed=strict_failed and not allow_unverified,
        )

    def extract_context(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        *,
        selected_docs: list[DocumentKey] | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Session, DraftingContext]:
        """Create a session, build narrative + voice context, and stage it.

        Returns the session and context so the caller can present the
        narrative for review before calling :meth:`generate_session_run`.
        """
        active_docs = selected_docs or list(ALL_DOC_KEYS)
        session = self.store.create(job, career, voice, selected_docs=active_docs)
        self._progress(progress, f"Session created: {session.session_id}")
        context = self._build_context(career, voice, job, progress)
        self.store.save_staging_context(session, context)
        return session, context

    def generate_session_run(
        self,
        session_id: str,
        *,
        context: DraftingContext | None = None,
        output_dir: Path | None = None,
        skip_review: bool = False,
        allow_unverified: bool = False,
        progress: ProgressCallback | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> OrchestrationResult:
        """Generate documents using a pre-built (and possibly curated) context.

        If *context* is ``None``, loads the staged context from disk.
        """
        session = self.store.get(session_id)
        career, voice = self.store.load_inputs(session)
        job = session.job_description
        active_docs = session.selected_docs or list(ALL_DOC_KEYS)

        if context is None:
            context = self.store.load_staging_context(session)
            if context is None:
                raise ValueError(
                    f"No staged context found for session {session_id}. "
                    "Run extract_context() first."
                )

        docs = DocumentSet()
        for key, label in self._doc_labels(active_docs).items():
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
                    f"'{label}' generated empty content â€” the model may have "
                    "exhausted its context window on reasoning. Try raising "
                    "RESUME_REFINERY_NUM_CTX in your .env."
                )
            docs.set(key, text)

        session = self.store.save_documents(session, docs, docs_regenerated=active_docs)
        self.store.save_context(session, context)
        self._export(session, docs, output_dir=output_dir)

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
            def _on_repair_pass(p: int, d: DocumentSet, r: ReviewBundle) -> None:
                self.store.save_repair_pass(session, p, d.model_copy(deep=True), r)
                self.store.update_documents(session, d)
                self._export(session, d, output_dir=output_dir)

            reviews, repair_passes, exempted = self._verify_and_repair(
                docs, career, voice, job, context, progress=progress,
                on_repair_pass=_on_repair_pass,
            )
        if exempted.claims or exempted.ai_phrases or exempted.voice_issues or exempted.hm_issues or exempted.pruning_issues or exempted.ats_issues or exempted.grammar_issues:
            self.store.save_suppressions(session, exempted)
        exported = self._export(session, docs, output_dir=output_dir)
        self.store.save_reviews(session, reviews)

        # Clean up staging files only after successful completion, so the
        # curate page remains accessible if generation/reviews fail.
        self.store.clear_staging_context(session)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=reviews,
            repair_passes=repair_passes,
            narrative=context.narrative,
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
        allow_unverified: bool = False,
        progress: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        session = self.store.get(session_id)
        career, voice = self.store.load_inputs(session)
        job = session.job_description
        current_docs = self.store.load_documents(session)
        context = self._build_context(career, voice, job, progress)

        # Load exemptions accumulated from prior runs in this session.
        exempted = self.store.load_suppressions(session) or ExemptedPhrases()

        keys_to_refine = [doc] if doc else list(self._doc_labels(session.selected_docs).keys())

        # Preserve originals so we can restore docs the user didn't target
        # and docs outside the session's selected_docs.
        originals = current_docs.model_copy(deep=True)

        # Apply user's instructions via the repair agent (single pass).
        self._progress(progress, "Applying refinement instructions...")
        repair_pass = self.repair_agent.repair_unified(
            current_docs, None, None, None,
            career, voice, job, context,
            feedback=feedback,
        )
        if repair_pass.edits:
            self._progress(progress, self._summarise_repair(repair_pass))

        # Accumulate any new acceptances from the repair pass.
        suppressed_claims = set(exempted.claims)
        suppressed_ai_phrases = set(exempted.ai_phrases)
        suppressed_voice_issues = set(exempted.voice_issues)
        suppressed_hm_issues = set(exempted.hm_issues)
        suppressed_pruning_issues = set(exempted.pruning_issues)
        suppressed_ats_issues = set(exempted.ats_issues)
        suppressed_grammar_issues = set(exempted.grammar_issues)
        suppressed_claims.update(repair_pass.accepted_claims)
        suppressed_ai_phrases.update(repair_pass.accepted_ai_phrases)
        suppressed_voice_issues.update(repair_pass.accepted_voice_issues)
        suppressed_hm_issues.update(repair_pass.accepted_hm_issues)
        suppressed_pruning_issues.update(repair_pass.accepted_pruning_issues)
        suppressed_ats_issues.update(repair_pass.accepted_ats_issues)
        suppressed_grammar_issues.update(repair_pass.accepted_grammar_issues)

        updated_exempted = ExemptedPhrases(
            claims=sorted(suppressed_claims),
            ai_phrases=sorted(suppressed_ai_phrases),
            voice_issues=sorted(suppressed_voice_issues),
            hm_issues=sorted(suppressed_hm_issues),
            pruning_issues=sorted(suppressed_pruning_issues),
            ats_issues=sorted(suppressed_ats_issues),
            grammar_issues=sorted(suppressed_grammar_issues),
        )

        # Restore documents that weren't targeted or aren't in selected_docs.
        for key in self._doc_labels():
            if key not in keys_to_refine or key not in session.selected_docs:
                current_docs.set(key, originals.get(key))

        # Save repaired documents as a new version.
        session = self.store.save_documents(
            session,
            current_docs,
            feedback=feedback,
            docs_regenerated=[k for k in keys_to_refine if k is not None],
        )
        self.store.save_context(session, context)
        self._export(session, current_docs, output_dir=output_dir)

        # Run all reviewers once (no repair loop), passing exemptions.
        self._progress(progress, "Running reviews...")
        reviews = self._run_all_reviews(
            current_docs, career, voice, job, progress,
            exempted=updated_exempted,
        )

        # Apply post-filter suppressions to review results.
        (reviews_truth, reviews_voice, reviews_ai, reviews_hm,
         reviews_pruning, reviews_ats, reviews_grammar) = self._apply_suppressions(
            reviews.truthfulness, reviews.voice, reviews.ai_detection,
            reviews.hiring_manager, reviews.relevance_pruning,
            reviews.ats_keyword, reviews.grammar,
            suppressed_claims, suppressed_ai_phrases, suppressed_voice_issues,
            suppressed_hm_issues, suppressed_pruning_issues,
            suppressed_ats_issues, suppressed_grammar_issues,
        )
        reviews = ReviewBundle(
            truthfulness=reviews_truth,
            voice=reviews_voice,
            ai_detection=reviews_ai,
            hiring_manager=reviews_hm,
            relevance_pruning=reviews_pruning,
            ats_keyword=reviews_ats,
            grammar=reviews_grammar,
        )

        # Persist the combined exemptions with the new version.
        if any([updated_exempted.claims, updated_exempted.ai_phrases,
                updated_exempted.voice_issues, updated_exempted.hm_issues,
                updated_exempted.pruning_issues, updated_exempted.ats_issues,
                updated_exempted.grammar_issues]):
            self.store.save_suppressions(session, updated_exempted)

        exported = self._export(session, current_docs, output_dir=output_dir)
        self.store.save_reviews(session, reviews)

        strict_failed = bool(reviews.truthfulness and not reviews.truthfulness.all_supported)
        return OrchestrationResult(
            session=session,
            documents=current_docs,
            reviews=reviews,
            repair_passes=[repair_pass],
            narrative=context.narrative,
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
        reviews = self._run_all_reviews(docs, career, voice, job, progress)
        self.store.save_reviews(session, reviews)
        return OrchestrationResult(
            session=session,
            documents=docs,
            reviews=reviews,
            narrative=context.narrative,
            voice_style_guide=context.voice_style_guide,
            strict_truth_failed=bool(reviews.truthfulness and not reviews.truthfulness.all_supported),
        )

    def _run_all_reviews(
        self,
        docs: DocumentSet,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        progress: ProgressCallback | None = None,
        exempted: ExemptedPhrases | None = None,
    ) -> ReviewBundle:
        """Run every reviewer once and return the combined bundle."""
        truth = None
        voice_review = None
        ai_review = None
        try:
            truth = self.verification_agent.review_truthfulness(
                docs, career, job,
                exemptions=exempted.claims if exempted and exempted.claims else None,
            )
        except Exception as exc:
            logging.warning("Truthfulness review failed (%s)", exc)
        try:
            voice_review = self.verification_agent.review_voice(
                docs, voice,
                exemptions=exempted.voice_issues if exempted and exempted.voice_issues else None,
            )
        except Exception as exc:
            logging.warning("Voice review failed (%s)", exc)
        try:
            ai_review = self.verification_agent.review_ai_detection(
                docs,
                exemptions=exempted.ai_phrases if exempted and exempted.ai_phrases else None,
            )
        except Exception as exc:
            logging.warning("AI-detection review failed (%s)", exc)
        reviews = ReviewBundle(truthfulness=truth, voice=voice_review, ai_detection=ai_review)
        if reviews.truthfulness:
            self._progress(progress, self._summarise_truth(reviews.truthfulness))
        if reviews.voice:
            self._progress(progress, self._summarise_voice(reviews.voice))
        if reviews.ai_detection:
            self._progress(progress, self._summarise_ai(reviews.ai_detection))
        try:
            hm_review = self.verification_agent.review_hiring_manager(
                docs, job,
                exemptions=exempted.hm_issues if exempted and exempted.hm_issues else None,
            )
            reviews = reviews.model_copy(update={"hiring_manager": hm_review})
            self._progress(progress, self._summarise_hiring_manager(hm_review))
        except Exception as exc:
            logging.warning("Hiring-manager review failed (%s)", exc)
        try:
            pruning_review = self.verification_agent.review_relevance_pruning(
                docs, job,
                exemptions=exempted.pruning_issues if exempted and exempted.pruning_issues else None,
            )
            reviews = reviews.model_copy(update={"relevance_pruning": pruning_review})
            self._progress(progress, self._summarise_relevance_pruning(pruning_review))
        except Exception as exc:
            logging.warning("Relevance-pruning review failed (%s)", exc)
        try:
            ats_review = self.verification_agent.review_ats_keyword(
                docs, job, career,
                exemptions=exempted.ats_issues if exempted and exempted.ats_issues else None,
            )
            reviews = reviews.model_copy(update={"ats_keyword": ats_review})
            self._progress(progress, self._summarise_ats_keyword(ats_review))
        except Exception as exc:
            logging.warning("ATS-keyword review failed (%s)", exc)
        try:
            grammar_review = self.verification_agent.review_grammar(
                docs,
                exemptions=exempted.grammar_issues if exempted and exempted.grammar_issues else None,
            )
            reviews = reviews.model_copy(update={"grammar": grammar_review})
            self._progress(progress, self._summarise_grammar(grammar_review))
        except Exception as exc:
            logging.warning("Grammar review failed (%s)", exc)
        return reviews

    def _build_context(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        progress: ProgressCallback | None = None,
    ) -> DraftingContext:
        self._progress(progress, "Building candidacy narrative...")
        narrative = self.narrative_agent.build_narrative(career, job)
        self._progress(progress, "Distilling voice guide...")
        style_guide = self.voice_agent.build_style_guide(voice)
        return DraftingContext(narrative=narrative, voice_style_guide=style_guide)

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
        hm_result: HiringManagerReview | None = None
        pruning_result: RelevancePruningResult | None = None
        ats_result: ATSKeywordResult | None = None
        grammar_result: GrammarResult | None = None

        # Per-reviewer suppression sets â€” accumulated across all repair passes.
        # Each reviewer has its own independent set so a voice false positive
        # cannot accidentally suppress a truthfulness finding (and vice versa).
        suppressed_claims: set[str] = set()
        suppressed_ai_phrases: set[str] = set()
        suppressed_voice_issues: set[str] = set()
        suppressed_hm_issues: set[str] = set()
        suppressed_pruning_issues: set[str] = set()
        suppressed_ats_issues: set[str] = set()
        suppressed_grammar_issues: set[str] = set()

        repair_sub_pass = 0  # running counter for on_repair_pass snapshots

        for pass_num in range(max_passes):
            self._progress(progress, f"â”€â”€â”€ Review Pass {pass_num + 1}/{max_passes} â”€â”€â”€")

            # ============================================================
            # Run all 8 reviewers concurrently
            # ============================================================
            self._progress(progress, "  Running all reviews (truth, ATS, grammar, voice, AI, HM, pruning)...")

            def _run_truth():
                try:
                    result = self.verification_agent.review_truthfulness(
                        docs, career, job,
                        exemptions=sorted(suppressed_claims) if suppressed_claims else None,
                    )
                    self._progress(progress, "    \u2713 Truth review complete")
                    return result
                except Exception as exc:
                    logging.warning("Truthfulness review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Truth review skipped: {exc}[/yellow]")
                    return None

            def _run_ats():
                try:
                    result = self.verification_agent.review_ats_keyword(
                        docs, job, career,
                        exemptions=sorted(suppressed_ats_issues) if suppressed_ats_issues else None,
                    )
                    self._progress(progress, "    \u2713 ATS keyword review complete")
                    return result
                except Exception as exc:
                    logging.warning("ATS-keyword review failed (%s)", exc)
                    self._progress(progress, f"[yellow]ATS-keyword review skipped: {exc}[/yellow]")
                    return None

            def _run_grammar():
                try:
                    result = self.verification_agent.review_grammar(
                        docs,
                        exemptions=sorted(suppressed_grammar_issues) if suppressed_grammar_issues else None,
                    )
                    self._progress(progress, "    \u2713 Grammar review complete")
                    return result
                except Exception as exc:
                    logging.warning("Grammar review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Grammar review skipped: {exc}[/yellow]")
                    return None

            def _run_voice():
                try:
                    result = self.verification_agent.review_voice(
                        docs, voice,
                        exemptions=sorted(suppressed_voice_issues) if suppressed_voice_issues else None,
                    )
                    self._progress(progress, "    \u2713 Voice review complete")
                    return result
                except Exception as exc:
                    logging.warning("Voice review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Voice review skipped: {exc}[/yellow]")
                    return None

            def _run_ai():
                try:
                    result = self.verification_agent.review_ai_detection(
                        docs,
                        exemptions=sorted(suppressed_ai_phrases) if suppressed_ai_phrases else None,
                    )
                    self._progress(progress, "    \u2713 AI detection review complete")
                    return result
                except Exception as exc:
                    logging.warning("AI-detection review failed (%s)", exc)
                    self._progress(progress, f"[yellow]AI-detection review skipped: {exc}[/yellow]")
                    return None

            def _run_hm():
                try:
                    result = self.verification_agent.review_hiring_manager(
                        docs, job,
                        exemptions=sorted(suppressed_hm_issues) if suppressed_hm_issues else None,
                    )
                    self._progress(progress, "    \u2713 Hiring manager review complete")
                    return result
                except Exception as exc:
                    logging.warning("Hiring-manager review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Hiring-manager review skipped: {exc}[/yellow]")
                    return None

            def _run_pruning():
                try:
                    result = self.verification_agent.review_relevance_pruning(
                        docs, job,
                        exemptions=sorted(suppressed_pruning_issues) if suppressed_pruning_issues else None,
                    )
                    self._progress(progress, "    \u2713 Relevance pruning review complete")
                    return result
                except Exception as exc:
                    logging.warning("Relevance-pruning review failed (%s)", exc)
                    self._progress(progress, f"[yellow]Relevance-pruning review skipped: {exc}[/yellow]")
                    return None

            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 7)) as pool:
                truth_future = pool.submit(_run_truth)
                ats_future = pool.submit(_run_ats)
                grammar_future = pool.submit(_run_grammar)
                voice_future = pool.submit(_run_voice)
                ai_future = pool.submit(_run_ai)
                hm_future = pool.submit(_run_hm)
                pruning_future = pool.submit(_run_pruning)

                truth = truth_future.result()
                ats_result = ats_future.result()
                grammar_result = grammar_future.result()
                voice_result = voice_future.result()
                ai_result = ai_future.result()
                hm_result = hm_future.result()
                pruning_result = pruning_future.result()

            # Filter out items accepted as false positives in earlier passes.
            truth, voice_result, ai_result, hm_result, pruning_result, ats_result, grammar_result = self._apply_suppressions(
                truth, voice_result, ai_result, hm_result, pruning_result,
                ats_result, grammar_result,
                suppressed_claims, suppressed_ai_phrases, suppressed_voice_issues,
                suppressed_hm_issues, suppressed_pruning_issues,
                suppressed_ats_issues, suppressed_grammar_issues,
            )

            # Summarise all reviews
            if truth:
                self._progress(progress, self._summarise_truth(truth))
            if ats_result:
                self._progress(progress, self._summarise_ats_keyword(ats_result))
            if grammar_result:
                self._progress(progress, self._summarise_grammar(grammar_result))
            if voice_result:
                self._progress(progress, self._summarise_voice(voice_result))
            if ai_result:
                self._progress(progress, self._summarise_ai(ai_result))
            if hm_result:
                self._progress(progress, self._summarise_hiring_manager(hm_result))
            if pruning_result:
                self._progress(progress, self._summarise_relevance_pruning(pruning_result))

            # Check all gates
            truth_ok = truth is None or truth.all_supported
            ats_ok = ats_result is None or ats_result.alignment_score in ("strong", "moderate")

            is_late_pass = pass_num >= _RELAXED_PASS_START
            if is_late_pass:
                grammar_ok = grammar_result is None or (
                    len(grammar_result.resume_issues)
                ) <= 2
            else:
                grammar_ok = grammar_result is None or grammar_result.clean

            voice_ok = voice_result is None or voice_result.overall_match in ("strong", "moderate")

            if is_late_pass:
                total_ai_flags = (
                    len(ai_result.resume_flags)
                ) if ai_result else 0
                ai_ok = ai_result is None or total_ai_flags <= _AI_FLAG_TOLERANCE_LATE
            else:
                ai_ok = ai_result is None or not ai_result.resume_flags

            # Hiring manager and relevance pruning are advisory â€” they feed
            # findings into repair but never block convergence (no hard gate).
            all_ok = truth_ok and ats_ok and grammar_ok and voice_ok and ai_ok

            if all_ok:
                break

            # Single unified repair with all findings
            self._progress(progress, "  Repair (up to 3 LLM calls, thinking enabled)...")
            repair_pass = self.repair_agent.repair_unified(
                docs, truth, voice_result, ai_result,
                career, voice, job, context,
                feedback=feedback,
                hm_review=hm_result,
                pruning_review=pruning_result,
                ats_review=ats_result,
                grammar_review=grammar_result,
                pass_num=pass_num,
                prior_edits=self._build_prior_edits(repair_results),
            )
            repair_results.append(repair_pass)
            suppressed_claims.update(repair_pass.accepted_claims)
            suppressed_ai_phrases.update(repair_pass.accepted_ai_phrases)
            suppressed_voice_issues.update(repair_pass.accepted_voice_issues)
            suppressed_hm_issues.update(repair_pass.accepted_hm_issues)
            suppressed_pruning_issues.update(repair_pass.accepted_pruning_issues)
            suppressed_ats_issues.update(repair_pass.accepted_ats_issues)
            suppressed_grammar_issues.update(repair_pass.accepted_grammar_issues)
            if repair_pass.edits:
                self._progress(progress, self._summarise_repair(repair_pass))
            if repair_pass.accepted_claims or repair_pass.accepted_ai_phrases or repair_pass.accepted_voice_issues or repair_pass.accepted_hm_issues or repair_pass.accepted_pruning_issues or repair_pass.accepted_ats_issues or repair_pass.accepted_grammar_issues:
                self._progress(progress, self._summarise_acceptances(repair_pass))
            if repair_pass.failed_edits:
                self._progress(progress, self._summarise_failed_edits(repair_pass))

            if on_repair_pass is not None:
                pass_reviews = ReviewBundle(
                    truthfulness=truth,
                    voice=voice_result,
                    ai_detection=ai_result,
                    hiring_manager=hm_result,
                    relevance_pruning=pruning_result,
                    ats_keyword=ats_result,
                    grammar=grammar_result,
                )
                on_repair_pass(repair_sub_pass, docs, pass_reviews)
            repair_sub_pass += 1

        return ReviewBundle(
            truthfulness=truth,
            voice=voice_result,
            ai_detection=ai_result,
            hiring_manager=hm_result,
            relevance_pruning=pruning_result,
            ats_keyword=ats_result,
            grammar=grammar_result,
        ), repair_results, ExemptedPhrases(
            claims=sorted(suppressed_claims),
            ai_phrases=sorted(suppressed_ai_phrases),
            voice_issues=sorted(suppressed_voice_issues),
            hm_issues=sorted(suppressed_hm_issues),
            pruning_issues=sorted(suppressed_pruning_issues),
            ats_issues=sorted(suppressed_ats_issues),
            grammar_issues=sorted(suppressed_grammar_issues),
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
        hm_result: HiringManagerReview | None,
        pruning_result: RelevancePruningResult | None,
        ats_result: ATSKeywordResult | None,
        grammar_result: GrammarResult | None,
        suppressed_claims: set[str],
        suppressed_ai_phrases: set[str],
        suppressed_voice_issues: set[str],
        suppressed_hm_issues: set[str],
        suppressed_pruning_issues: set[str],
        suppressed_ats_issues: set[str],
        suppressed_grammar_issues: set[str],
    ) -> tuple[TruthfulnessResult | None, VoiceReviewResult | None, AIDetectionResult | None, HiringManagerReview | None, RelevancePruningResult | None, ATSKeywordResult | None, GrammarResult | None]:
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
            res = _filter_doc(truth.resume)
            filtered_truth = truth.model_copy(update={
                "resume": res,
                "all_supported": res.pass_strict,
            })

        # --- AI detection ---
        filtered_ai = ai_result
        if ai_result and suppressed_ai_phrases:
            res_flags = [f for f in ai_result.resume_flags if f not in suppressed_ai_phrases]
            total = len(res_flags)
            risk = "low" if total <= 1 else "medium" if total <= 3 else "high"
            filtered_ai = ai_result.model_copy(update={
                "resume_flags": res_flags,
                "risk_level": risk,
            })

        # --- Voice ---
        filtered_voice = voice_result
        if voice_result and suppressed_voice_issues:
            filtered_voice = voice_result.model_copy(update={
                "resume_issues": [i for i in voice_result.resume_issues if i not in suppressed_voice_issues],
                "specific_issues": [i for i in voice_result.specific_issues if i not in suppressed_voice_issues],
            })

        # --- Hiring manager ---
        filtered_hm = hm_result
        if hm_result and suppressed_hm_issues:
            filtered_hm = hm_result.model_copy(update={
                "resume_issues": [
                    i for i in hm_result.resume_issues
                    if i.phrase not in suppressed_hm_issues
                ],
            })

        # --- Relevance pruning ---
        filtered_pruning = pruning_result
        if pruning_result and suppressed_pruning_issues:
            filtered_pruning = pruning_result.model_copy(update={
                "resume_issues": [
                    i for i in pruning_result.resume_issues
                    if i.phrase not in suppressed_pruning_issues
                ],
            })

        # --- ATS keyword ---
        filtered_ats = ats_result
        if ats_result and suppressed_ats_issues:
            filtered_missing = [
                i for i in ats_result.missing_keywords
                if i.keyword not in suppressed_ats_issues
            ]
            filtered_stuffing = [
                i for i in ats_result.stuffing_keywords
                if i.keyword not in suppressed_ats_issues
            ]
            remaining = len(filtered_missing)
            score = "strong" if remaining <= 1 else "moderate" if remaining <= 3 else "weak"
            filtered_ats = ats_result.model_copy(update={
                "missing_keywords": filtered_missing,
                "stuffing_keywords": filtered_stuffing,
                "alignment_score": score,
            })

        # --- Grammar ---
        filtered_grammar = grammar_result
        if grammar_result and suppressed_grammar_issues:
            res_issues = [i for i in grammar_result.resume_issues if i.phrase not in suppressed_grammar_issues]
            filtered_grammar = grammar_result.model_copy(update={
                "resume_issues": res_issues,
                "clean": not res_issues,
            })

        return filtered_truth, filtered_voice, filtered_ai, filtered_hm, filtered_pruning, filtered_ats, filtered_grammar

    # ------------------------------------------------------------------
    # Prior-edit context builder (annotated pass-through)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prior_edits(
        repair_results: list[RepairPassResult],
    ) -> dict[str, str]:
        """Build per-document summaries of prior edits for annotated pass-through.

        Instead of silently filtering findings that overlap previously edited
        regions, we pass this context to the repair agent so it can make
        informed fix/merge/accept decisions.

        Returns a dict mapping document key â†’ human-readable prior-edit summary.
        """
        from .models import REVIEWER_PRIORITY_RANK

        # Accumulate edits per document across all prior repair passes.
        doc_edits: dict[str, list[tuple[str, str, str, str, bool]]] = {}  # key -> [(reviewer, find, replace, reason, insert_after)]
        for rp in repair_results:
            for doc_key, edits in rp.edits.items():
                entries = doc_edits.setdefault(doc_key, [])
                for edit in edits:
                    entries.append((edit.reviewer, edit.find, edit.replace, edit.reason, edit.insert_after))

        result: dict[str, str] = {}
        for doc_key, entries in doc_edits.items():
            if not entries:
                continue
            lines: list[str] = []
            for reviewer, find_text, replace_text, reason, is_insert in entries:
                if is_insert:
                    lines.append(
                        f'- [{reviewer}] INSERTED after "{find_text[:80]}": "{replace_text[:80]}"'
                        + (f"  ({reason[:60]})" if reason else "")
                    )
                elif replace_text:
                    lines.append(
                        f'- [{reviewer}] "{find_text[:80]}" â†’ "{replace_text[:80]}"'
                        + (f"  ({reason[:60]})" if reason else "")
                    )
                else:
                    lines.append(
                        f'- [{reviewer}] DELETED "{find_text[:80]}"'
                        + (f"  ({reason[:60]})" if reason else "")
                    )
            if lines:
                result[doc_key] = "\n".join(lines)

        return result

    # ------------------------------------------------------------------
    # Review-result summaries emitted via the progress callback
    # ------------------------------------------------------------------

    def _summarise_truth(self, truth: TruthfulnessResult) -> str:
        if truth.all_supported:
            parts = ["[green]Truthfulness: ALL SUPPORTED[/green]"]
        else:
            parts = ["[red]Truthfulness: UNSUPPORTED CLAIMS DETECTED[/red]"]
        doc = truth.resume
        status = "[green]âœ“[/green]" if doc.pass_strict else f"[red]âœ— ({len(doc.unsupported_claims)} unsupported)[/red]"
        parts.append(f"  Resume: {status}")
        if not doc.pass_strict:
            for claim in doc.unsupported_claims:
                parts.append(f"    â€¢ {claim}")
        return "\n".join(parts)

    def _summarise_voice(self, voice: VoiceReviewResult) -> str:
        color = {"strong": "green", "moderate": "yellow", "weak": "red"}[voice.overall_match]
        parts = [f"[{color}]Voice match: {voice.overall_match.upper()}[/{color}]"]
        mc = {"strong": "green", "moderate": "yellow", "weak": "red"}[voice.resume_match]
        parts.append(f"  Resume: [{mc}]{voice.resume_match}[/{mc}]")
        if voice.resume_match != "strong" and voice.resume_issues:
            for issue in voice.resume_issues:
                parts.append(f"    â€¢ {issue}")
        return "\n".join(parts)

    def _summarise_ai(self, ai: AIDetectionResult) -> str:
        color = {"low": "green", "medium": "yellow", "high": "red"}[ai.risk_level]
        parts = [f"[{color}]AI-detection risk: {ai.risk_level.upper()}[/{color}]"]
        if ai.resume_flags:
            parts.append(f"  Resume: {len(ai.resume_flags)} flag(s)")
            for flag in ai.resume_flags:
                parts.append(f'    â€¢ "{flag}"')
        return "\n".join(parts)

    def _summarise_hiring_manager(self, hm: HiringManagerReview) -> str:
        pct = hm.advance_likelihood
        color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
        total_issues = len(hm.resume_issues)
        parts = [f"[{color}]Hiring-manager advance likelihood: {pct}% ({total_issues} issue(s))[/{color}]"]
        if hm.summary:
            parts.append(f"  {hm.summary}")
        if hm.resume_issues:
            parts.append("  Resume:")
            for i in hm.resume_issues:
                parts.append(f'    [{i.impact.upper()}] "{i.phrase[:80]}" â€” {i.issue}')
        return "\n".join(parts)

    def _summarise_relevance_pruning(self, pruning: RelevancePruningResult) -> str:
        density = pruning.overall_density
        color = "green" if density == "lean" else "yellow" if density == "balanced" else "red"
        total = len(pruning.resume_issues)
        parts = [f"[{color}]Relevance pruning: {density} ({total} removal candidate(s))[/{color}]"]
        if pruning.resume_issues:
            parts.append("  Resume:")
            for issue in pruning.resume_issues:
                parts.append(f'    [{issue.severity.upper()}] ({issue.category}) "{issue.phrase[:80]}" — {issue.reason}')
        return "\n".join(parts)

    def _summarise_ats_keyword(self, ats: ATSKeywordResult) -> str:
        color = {"strong": "green", "moderate": "yellow", "weak": "red"}[ats.alignment_score]
        total_missing = len(ats.missing_keywords)
        total_stuffing = len(ats.stuffing_keywords)
        parts = [f"[{color}]ATS keyword alignment: {ats.alignment_score.upper()} ({total_missing} missing, {total_stuffing} stuffing)[/{color}]"]
        if ats.missing_keywords:
            parts.append("  Missing keywords:")
            for kw in ats.missing_keywords:
                parts.append(f'    [{kw.priority.upper()}] "{kw.keyword}" â€” {kw.suggestion}')
        if ats.stuffing_keywords:
            parts.append("  Keyword stuffing:")
            for kw in ats.stuffing_keywords:
                parts.append(f'    "{kw.keyword}" in {kw.section} â€” {kw.suggestion}')
        return "\n".join(parts)

    def _summarise_grammar(self, grammar: GrammarResult) -> str:
        total = len(grammar.resume_issues)
        if grammar.clean:
            return "[green]Grammar & mechanics: CLEAN[/green]"
        parts = [f"[red]Grammar & mechanics: {total} issue(s)[/red]"]
        if grammar.resume_issues:
            parts.append("  Resume:")
            for issue in grammar.resume_issues:
                parts.append(f'    [{issue.severity.upper()}] ({issue.category}) "{issue.phrase[:60]}" â€” {issue.issue}')
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
                parts.append(f'    [cyan]âœ“ "{claim}"[/cyan]')
        if repair_pass.accepted_ai_phrases:
            parts.append("  [cyan]AI-detection flags (accepted as natural human language):[/cyan]")
            for phrase in repair_pass.accepted_ai_phrases:
                parts.append(f'    [cyan]âœ“ "{phrase}"[/cyan]')
        if repair_pass.accepted_voice_issues:
            parts.append("  [cyan]Voice-match issues (accepted as reviewer false positives):[/cyan]")
            for issue in repair_pass.accepted_voice_issues:
                parts.append(f'    [cyan]âœ“ "{issue}"[/cyan]')
        if repair_pass.accepted_hm_issues:
            parts.append("  [cyan]Hiring-manager issues (accepted as false positives):[/cyan]")
            for issue in repair_pass.accepted_hm_issues:
                parts.append(f'    [cyan]âœ“ "{issue}"[/cyan]')
        if repair_pass.accepted_pruning_issues:
            parts.append("  [cyan]Relevance-pruning issues (accepted as valuable content):[/cyan]")
            for issue in repair_pass.accepted_pruning_issues:
                parts.append(f'    [cyan]âœ“ "{issue}"[/cyan]')
        if repair_pass.accepted_ats_issues:
            parts.append("  [cyan]ATS-keyword issues (accepted as adequately represented):[/cyan]")
            for issue in repair_pass.accepted_ats_issues:
                parts.append(f'    [cyan]âœ“ "{issue}"[/cyan]')
        if repair_pass.accepted_grammar_issues:
            parts.append("  [cyan]Grammar issues (accepted as correct/intentional):[/cyan]")
            for issue in repair_pass.accepted_grammar_issues:
                parts.append(f'    [cyan]âœ“ "{issue}"[/cyan]')
        return "\n".join(parts)

    def _summarise_failed_edits(self, repair_pass: RepairPassResult) -> str:
        """Build a Rich-tagged summary of edits that failed Phase 1 locate."""
        doc_labels = self._doc_labels()
        parts = ["[bold yellow]Failed edits (could not locate find-text in document):[/bold yellow]"]
        for key, failures in repair_pass.failed_edits.items():
            label = doc_labels.get(key, key)
            parts.append(f"  {label}: {len(failures)} failed edit(s)")
            for fail in failures:
                find_snippet = fail.get("find", "")[:80]
                reason = fail.get("reason", "")
                parts.append(f'    [yellow]âœ— "{find_snippet}"[/yellow]')
                if reason:
                    parts.append(f"      ({reason})")
        return "\n".join(parts)

    def _doc_labels(self, selected: list[DocumentKey] | None = None) -> dict[DocumentKey, str]:
        all_labels: dict[DocumentKey, str] = {
            "resume": "Resume",
        }
        if selected is None:
            return all_labels
        return {k: v for k, v in all_labels.items() if k in selected}
