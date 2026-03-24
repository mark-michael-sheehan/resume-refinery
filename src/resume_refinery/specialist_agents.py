"""Bounded specialist agents used by the workflow orchestrator."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable, Iterator

import ollama
from dotenv import load_dotenv

from .agent import ResumeRefineryAgent
from .models import (
    AIDetectionResult,
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    EvidenceItem,
    EvidencePack,
    JobDescription,
    JobRequirement,
    ReviewBundle,
    TruthfulnessResult,
    VoiceProfile,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from .prompts import (
    EVIDENCE_MATCHING_SYSTEM_PROMPT,
    EVIDENCE_MATCHING_USER_TEMPLATE,
    REQUIREMENT_EXTRACTION_SYSTEM_PROMPT,
    REQUIREMENT_EXTRACTION_USER_TEMPLATE,
)
from .reviewers import DocumentReviewer

load_dotenv()

_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")
_NUM_CTX = int(os.environ.get("RESUME_REFINERY_NUM_CTX", "16384"))
_MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "4096"))

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


class EvidenceAgent:
    """Extracts job requirements and grounded evidence from raw inputs.

    Uses LLM calls for semantic understanding with keyword-based fallbacks
    when the LLM is unavailable or returns invalid results.
    """

    def __init__(self, client: ollama.Client | None = None) -> None:
        self.client = client or ollama.Client(host=_BASE_URL)

    def build_evidence_pack(self, career: CareerProfile, job: JobDescription) -> EvidencePack:
        requirements = self._extract_requirements(job.raw_content)
        matched: list[EvidenceItem] = []
        gaps: list[str] = []

        for requirement in requirements:
            evidence_items = self._match_evidence(requirement.requirement, career.raw_content)
            if evidence_items:
                matched.extend(evidence_items)
            else:
                gaps.append(requirement.requirement)

        career_lines = self._career_lines(career.raw_content)
        summary = [line for line in career_lines if len(line) > 20][:8]
        return EvidencePack(
            job_requirements=requirements,
            matched_evidence=matched,
            gaps=gaps,
            source_summary=summary,
        )

    def _extract_requirements(self, raw_job: str) -> list[JobRequirement]:
        """Extract requirements using LLM, falling back to keyword heuristics."""
        try:
            return self._extract_requirements_llm(raw_job)
        except Exception as exc:
            logging.warning("LLM requirement extraction failed (%s); using keyword fallback.", exc)
            return self._extract_requirements_keyword(raw_job)

    def _extract_requirements_llm(self, raw_job: str) -> list[JobRequirement]:
        """Use the LLM to extract structured requirements from the job description."""
        user_msg = REQUIREMENT_EXTRACTION_USER_TEMPLATE.format(job_description=raw_job)
        raw = self._call_llm(REQUIREMENT_EXTRACTION_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        requirements: list[JobRequirement] = []
        for item in data[:10]:
            if isinstance(item, dict) and "requirement" in item:
                category = item.get("category", "other")
                if category not in ("skill", "experience", "leadership", "domain", "other"):
                    category = "other"
                requirements.append(
                    JobRequirement(
                        requirement=item["requirement"],
                        category=category,
                        source_excerpt=item.get("source_excerpt", item["requirement"]),
                    )
                )
        if not requirements:
            raise ValueError("LLM returned empty requirements list")
        return requirements

    def _extract_requirements_keyword(self, raw_job: str) -> list[JobRequirement]:
        """Keyword heuristic fallback for requirement extraction."""
        requirements: list[JobRequirement] = []
        seen: set[str] = set()
        for line in raw_job.splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            lowered = clean.lower()
            if any(token in lowered for token in ("required", "must", "need", "experience", "skills", "responsible")):
                for piece in self._split_requirement_line(clean):
                    normalized = piece.strip()
                    if normalized and normalized.lower() not in seen:
                        seen.add(normalized.lower())
                        requirements.append(
                            JobRequirement(
                                requirement=normalized,
                                category=self._categorize_requirement(normalized),
                                source_excerpt=clean,
                            )
                        )
        if not requirements:
            fallback = [line.strip() for line in raw_job.splitlines() if line.strip()][:5]
            for line in fallback:
                if line.lower().startswith("company:"):
                    continue
                requirements.append(JobRequirement(requirement=line, source_excerpt=line))
        return requirements[:10]

    def _match_evidence(self, requirement: str, career_content: str) -> list[EvidenceItem]:
        """Match evidence using LLM, falling back to keyword overlap."""
        try:
            return self._match_evidence_llm(requirement, career_content)
        except Exception as exc:
            logging.warning("LLM evidence matching failed (%s); using keyword fallback.", exc)
            career_lines = self._career_lines(career_content)
            return self._match_evidence_keyword(requirement, career_lines)

    def _match_evidence_llm(self, requirement: str, career_content: str) -> list[EvidenceItem]:
        """Use the LLM to find semantically relevant evidence for a requirement."""
        user_msg = EVIDENCE_MATCHING_USER_TEMPLATE.format(
            requirement=requirement,
            career_profile=career_content,
        )
        raw = self._call_llm(EVIDENCE_MATCHING_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        items: list[EvidenceItem] = []
        for entry in data[:3]:
            if isinstance(entry, dict) and "evidence" in entry:
                score = entry.get("relevance_score", 3)
                if not isinstance(score, int) or score < 1 or score > 5:
                    score = 3
                items.append(
                    EvidenceItem(
                        requirement=requirement,
                        evidence=entry["evidence"],
                        source_excerpt=entry["evidence"],
                        relevance_score=score,
                    )
                )
        return items

    def _match_evidence_keyword(self, requirement: str, career_lines: list[str]) -> list[EvidenceItem]:
        """Keyword overlap fallback for evidence matching."""
        req_keywords = self._keywords(requirement)
        scored: list[tuple[int, str]] = []
        for line in career_lines:
            line_keywords = self._keywords(line)
            overlap = len(req_keywords & line_keywords)
            if overlap:
                scored.append((overlap, line))
        scored.sort(key=lambda item: (-item[0], -len(item[1])))
        items: list[EvidenceItem] = []
        for rank, (_, evidence) in enumerate(scored[:3], start=1):
            items.append(
                EvidenceItem(
                    requirement=requirement,
                    evidence=evidence,
                    source_excerpt=evidence,
                    relevance_score=max(1, 6 - rank),
                )
            )
        return items

    def _call_llm(self, system: str, user_msg: str) -> str:
        """Make an Ollama API call and return cleaned JSON text."""
        response = self.client.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "/no_think\n" + user_msg},
            ],
            think=False,
            format="json",
            options={"num_ctx": _NUM_CTX, "num_predict": _MAX_TOKENS},
        )
        raw = response.message.content.strip()
        # Strip residual think blocks
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        # Repair LLM JSON quirks
        from .reviewers import _normalize_llm_json
        raw = _normalize_llm_json(raw)
        if not raw:
            raise ValueError("LLM returned empty content")
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        return raw

    def _split_requirement_line(self, line: str) -> list[str]:
        if ":" in line:
            _, tail = line.split(":", 1)
        else:
            tail = line
        return [piece.strip() for piece in re.split(r",|;", tail) if piece.strip()]

    def _categorize_requirement(self, requirement: str) -> str:
        lowered = requirement.lower()
        if any(word in lowered for word in ("python", "sql", "aws", "system", "distributed", "architecture")):
            return "skill"
        if any(word in lowered for word in ("lead", "mentor", "stakeholder", "strategy")):
            return "leadership"
        if any(word in lowered for word in ("years", "experience", "background")):
            return "experience"
        if any(word in lowered for word in ("fintech", "healthcare", "platform", "data")):
            return "domain"
        return "other"

    def _career_lines(self, raw_career: str) -> list[str]:
        return [line.strip(" -\t") for line in raw_career.splitlines() if line.strip()]

    def _keywords(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
            if token not in _STOPWORDS
        }


class VoiceAgent:
    """Distills a reusable style guide from the voice profile."""

    def build_style_guide(self, voice: VoiceProfile) -> VoiceStyleGuide:
        sections = self._section_map(voice.raw_content)
        return VoiceStyleGuide(
            core_adjectives=self._collect_list_items(sections.get("adjectives", "")),
            style_rules=self._collect_list_items(sections.get("style notes", voice.raw_content)),
            preferred_phrases=self._collect_list_items(sections.get("phrases you actually use", "")),
            phrases_to_avoid=self._collect_list_items(sections.get("phrases to avoid", "")),
            writing_samples=self._collect_paragraphs(sections.get("writing samples", "")),
        )

    def _section_map(self, raw: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current = ""
        buffer: list[str] = []
        for line in raw.splitlines():
            heading = re.match(r"^##\s+(.+)$", line.strip())
            if heading:
                if current:
                    sections[current] = "\n".join(buffer).strip()
                current = heading.group(1).strip().lower()
                buffer = []
                continue
            buffer.append(line)
        if current:
            sections[current] = "\n".join(buffer).strip()
        return sections

    def _collect_list_items(self, raw: str) -> list[str]:
        items = [line.strip(" -*\t") for line in raw.splitlines() if line.strip().startswith(("-", "*"))]
        if items:
            return items
        return [part.strip() for part in raw.splitlines() if part.strip()][:6]

    def _collect_paragraphs(self, raw: str) -> list[str]:
        paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
        return paragraphs[:3]


class DraftingAgent:
    """Uses distilled context to draft documents through the core LLM generator."""

    def __init__(self, generator: ResumeRefineryAgent | None = None) -> None:
        self.generator = generator or ResumeRefineryAgent()

    def generate_all(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
    ) -> DocumentSet:
        docs = DocumentSet()
        for key in ("cover_letter", "resume", "interview_guide"):
            docs.set(key, self.generate_document(key, career, voice, job, context))
        return docs

    def generate_document(
        self,
        key: DocumentKey,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_version: str | None = None,
    ) -> str:
        return self.generator.generate_document(
            key,
            self._career_context(career, context.evidence_pack),
            self._voice_context(voice, context.voice_style_guide),
            job,
            feedback=feedback,
            previous_version=previous_version,
        )

    def stream_document(
        self,
        key: DocumentKey,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_version: str | None = None,
    ) -> Iterator[str]:
        yield from self.generator.stream_document(
            key,
            self._career_context(career, context.evidence_pack),
            self._voice_context(voice, context.voice_style_guide),
            job,
            feedback=feedback,
            previous_version=previous_version,
        )

    def _career_context(self, career: CareerProfile, evidence_pack: EvidencePack) -> CareerProfile:
        summary_lines = [
            "## Evidence Pack",
            "**Use the evidence pack below as your PRIMARY source for claims. "
            "The full career profile is provided only for additional detail.**",
            "",
            "### Top Job Requirements",
        ]
        summary_lines.extend(f"- {item.requirement}" for item in evidence_pack.job_requirements[:8])

        # Sort matched evidence by relevance score descending
        sorted_evidence = sorted(
            evidence_pack.matched_evidence[:12],
            key=lambda e: e.relevance_score,
            reverse=True,
        )
        summary_lines.append("\n### Matched Evidence (highest relevance first)")
        for item in sorted_evidence:
            priority = "HIGH PRIORITY" if item.relevance_score >= 4 else "supporting"
            summary_lines.append(
                f"- [{priority}] Requirement: {item.requirement} | "
                f"Evidence: {item.evidence} (relevance: {item.relevance_score}/5)"
            )
        if evidence_pack.gaps:
            summary_lines.append("\n### Potential Gaps")
            summary_lines.extend(f"- {gap}" for gap in evidence_pack.gaps[:6])
            summary_lines.append(
                "\n**Important**: The gaps above are requirements from the job description "
                "that have no direct evidence in the career profile. Do NOT fabricate "
                "experience to cover them. Either omit them or frame related transferable "
                "skills honestly."
            )
        summary_lines.append("\n### Full Career Profile")
        summary_lines.append(career.raw_content)
        return career.model_copy(update={"raw_content": "\n".join(summary_lines)})

    def _voice_context(self, voice: VoiceProfile, guide: VoiceStyleGuide) -> VoiceProfile:
        lines = ["## Distilled Voice Guide"]
        if guide.core_adjectives:
            lines.append("### Core Adjectives")
            lines.extend(f"- {item}" for item in guide.core_adjectives[:8])
        if guide.style_rules:
            lines.append("\n### Style Rules")
            lines.extend(f"- {item}" for item in guide.style_rules[:8])
        if guide.preferred_phrases:
            lines.append("\n### Preferred Phrases")
            lines.extend(f"- {item}" for item in guide.preferred_phrases[:8])
        if guide.phrases_to_avoid:
            lines.append("\n### Avoid")
            lines.extend(f"- {item}" for item in guide.phrases_to_avoid[:8])
        lines.append("\n### Full Voice Profile")
        lines.append(voice.raw_content)
        return voice.model_copy(update={"raw_content": "\n".join(lines)})


class VerificationAgent:
    """Runs bounded verification passes on drafted documents."""

    def __init__(self, reviewer: DocumentReviewer | None = None) -> None:
        self.reviewer = reviewer or DocumentReviewer()

    def review_all(self, docs: DocumentSet, career: CareerProfile, voice: VoiceProfile) -> ReviewBundle:
        return ReviewBundle(
            truthfulness=self.reviewer.review_truthfulness(docs, career),
            voice=self.reviewer.review_voice(docs, voice),
            ai_detection=self.reviewer.review_ai_detection(docs),
        )

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile) -> TruthfulnessResult:
        return self.reviewer.review_truthfulness(docs, career)

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile) -> VoiceReviewResult:
        return self.reviewer.review_voice(docs, voice)

    def review_ai_detection(self, docs: DocumentSet) -> AIDetectionResult:
        return self.reviewer.review_ai_detection(docs)


class RepairAgent:
    """Performs bounded rewrites in response to verifier findings."""

    def __init__(self, drafting_agent: DraftingAgent | None = None) -> None:
        self.drafting_agent = drafting_agent or DraftingAgent()

    def repair_unified(
        self,
        docs: DocumentSet,
        truth: TruthfulnessResult | None,
        voice_review: VoiceReviewResult | None,
        ai_review: AIDetectionResult | None,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_suggestions: list[str] | None = None,
    ) -> DocumentSet:
        """Unified repair: combine feedback from all reviewers and repair each doc once."""
        for key in ("cover_letter", "resume", "interview_guide"):
            parts = self._build_unified_feedback(
                key, truth, voice_review, ai_review, feedback, previous_suggestions,
            )
            if not parts:
                continue
            previous = docs.get(key)
            if not previous:
                continue
            repair_feedback = "\n\n".join(parts)
            regenerated = self.drafting_agent.generate_document(
                key, career, voice, job, context,
                feedback=repair_feedback, previous_version=previous,
            )
            docs.set(key, regenerated)
        return docs

    def _build_unified_feedback(
        self,
        key: DocumentKey,
        truth: TruthfulnessResult | None,
        voice_review: VoiceReviewResult | None,
        ai_review: AIDetectionResult | None,
        feedback: str | None,
        previous_suggestions: list[str] | None,
    ) -> list[str]:
        parts: list[str] = []
        has_issues = False

        if feedback:
            parts.append(feedback)

        # --- Truthfulness ---
        if truth:
            truth_map = {
                "cover_letter": truth.cover_letter,
                "resume": truth.resume,
                "interview_guide": truth.interview_guide,
            }
            doc_truth = truth_map[key]
            if not doc_truth.pass_strict:
                has_issues = True
                if doc_truth.unsupported_claims:
                    parts.append(
                        "TRUTHFULNESS — Unsupported claims to remove or directly evidence:\n"
                        + "\n".join(f"- {c}" for c in doc_truth.unsupported_claims[:8])
                    )
                else:
                    parts.append(
                        "TRUTHFULNESS — The truthfulness check failed. Review every claim "
                        "and ensure each is directly supported by evidence from the career profile."
                    )
                if doc_truth.evidence_examples:
                    parts.append(
                        "Evidence examples to reference:\n"
                        + "\n".join(f"- {e}" for e in doc_truth.evidence_examples[:8])
                    )
                if doc_truth.suggestions:
                    new = self._deduplicate(doc_truth.suggestions, previous_suggestions)
                    if new:
                        parts.append(
                            "Truthfulness suggestions:\n"
                            + "\n".join(f"- {s}" for s in new[:8])
                        )

        # --- Voice (skip for interview guide — personal prep) ---
        if voice_review and key != "interview_guide":
            voice_matches: dict[DocumentKey, str] = {
                "cover_letter": voice_review.cover_letter_match,
                "resume": voice_review.resume_match,
                "interview_guide": voice_review.interview_guide_match,
            }
            if voice_matches[key] != "strong":
                has_issues = True
                assessments: dict[DocumentKey, str] = {
                    "cover_letter": voice_review.cover_letter_assessment,
                    "resume": voice_review.resume_assessment,
                    "interview_guide": voice_review.interview_guide_assessment,
                }
                doc_issues_map: dict[DocumentKey, list[str]] = {
                    "cover_letter": voice_review.cover_letter_issues,
                    "resume": voice_review.resume_issues,
                    "interview_guide": voice_review.interview_guide_issues,
                }
                doc_suggestions_map: dict[DocumentKey, list[str]] = {
                    "cover_letter": voice_review.cover_letter_suggestions,
                    "resume": voice_review.resume_suggestions,
                    "interview_guide": voice_review.interview_guide_suggestions,
                }
                parts.append(f"VOICE — Assessment: {assessments[key]}")
                issues = doc_issues_map[key] or voice_review.specific_issues
                if issues:
                    parts.append(
                        "Voice issues to fix:\n"
                        + "\n".join(f"- {i}" for i in issues[:8])
                    )
                suggestions = doc_suggestions_map[key] or voice_review.suggestions
                if suggestions:
                    new = self._deduplicate(suggestions, previous_suggestions)
                    if new:
                        parts.append(
                            "Voice suggestions:\n"
                            + "\n".join(f"- {s}" for s in new[:8])
                        )

        # --- AI detection (skip for interview guide — personal prep) ---
        if ai_review and key != "interview_guide":
            flag_map: dict[DocumentKey, list[str]] = {
                "cover_letter": ai_review.cover_letter_flags,
                "resume": ai_review.resume_flags,
                "interview_guide": ai_review.interview_guide_flags,
            }
            ai_suggestions_map: dict[DocumentKey, list[str]] = {
                "cover_letter": ai_review.cover_letter_suggestions,
                "resume": ai_review.resume_suggestions,
                "interview_guide": [],
            }
            flags = flag_map[key]
            if flags:
                has_issues = True
                parts.append(
                    "AI DETECTION — Flagged phrases:\n"
                    + "\n".join(f'- "{f}"' for f in flags[:8])
                )
                # Use per-doc suggestions; fall back to aggregated for backward compat
                ai_suggestions = ai_suggestions_map.get(key) or ai_review.suggestions
                if ai_suggestions:
                    new = self._deduplicate(ai_suggestions, previous_suggestions)
                    if new:
                        parts.append(
                            "AI-detection suggestions:\n"
                            + "\n".join(f"- {s}" for s in new[:8])
                        )

        if not has_issues:
            return []

        if previous_suggestions:
            parts.append(
                "Previously attempted suggestions (already tried — try a different approach):\n"
                + "\n".join(f"- {s}" for s in previous_suggestions[:12])
            )

        parts.append(
            "REPAIR PROCEDURE (follow these phases in order during your thinking):\n"
            "Phase 1 — TRUTHFULNESS: Identify every unsupported claim listed above. "
            "For each one, either remove it or replace it with a fact directly from "
            "the Career Profile. Do NOT rephrase surrounding text. Write out the "
            "corrected draft.\n"
            "Phase 2 — VOICE & AI: Apply voice and AI-detection fixes ONLY to "
            "sentences that were NOT changed in Phase 1. If a voice/AI fix would "
            "alter a factual claim, skip it.\n"
            "Phase 3 — TRUTHFULNESS SELF-CHECK: Re-read your Phase 2 draft "
            "sentence by sentence. For each sentence, verify the claim still "
            "appears verbatim or is directly supported in the Career Profile. "
            "If any new unsupported claim was introduced in Phase 2, revert that "
            "sentence to the Phase 1 version.\n"
            "Repeat Phases 2-3 until a full pass introduces ZERO new unsupported "
            "claims. Usually 1-2 iterations suffice.\n"
            "FINAL — OUTPUT: Emit the final document. It must pass truthfulness."
        )
        parts.append(
            "MINIMAL EDITS ONLY: Make the smallest changes necessary to address the feedback "
            "above. Preserve all text that is not specifically flagged. Do not restructure "
            "paragraphs, rephrase sentences, or rewrite sections that already pass all checks. "
            "If a flagged phrase can be fixed by changing a few words, do that — do not rewrite "
            "the surrounding paragraph."
        )
        return parts

    def repair_documents(
        self,
        docs: DocumentSet,
        truth: TruthfulnessResult,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_suggestions: list[str] | None = None,
    ) -> DocumentSet:
        for key in self._docs_to_fix(truth):
            previous = docs.get(key)
            repair_feedback = self._feedback_for_doc(key, truth, feedback, previous_suggestions)
            regenerated = self.drafting_agent.generate_document(
                key,
                career,
                voice,
                job,
                context,
                feedback=repair_feedback,
                previous_version=previous,
            )
            docs.set(key, regenerated)
        return docs

    def _docs_to_fix(self, truth: TruthfulnessResult) -> list[DocumentKey]:
        docs: list[DocumentKey] = []
        if not truth.cover_letter.pass_strict:
            docs.append("cover_letter")
        if not truth.resume.pass_strict:
            docs.append("resume")
        if not truth.interview_guide.pass_strict:
            docs.append("interview_guide")
        return docs

    def _feedback_for_doc(self, key: DocumentKey, truth: TruthfulnessResult, feedback: str | None, previous_suggestions: list[str] | None = None) -> str:
        if key == "cover_letter":
            doc_truth = truth.cover_letter
        elif key == "resume":
            doc_truth = truth.resume
        else:
            doc_truth = truth.interview_guide
        parts = []
        if feedback:
            parts.append(feedback)
        if doc_truth.unsupported_claims:
            parts.append("Unsupported claims to remove or directly evidence:\n" + "\n".join(f"- {claim}" for claim in doc_truth.unsupported_claims[:8]))
        elif not doc_truth.pass_strict:
            # Strict check failed but no specific claims listed — give explicit guidance
            parts.append("The truthfulness check failed but no specific unsupported claims were identified. "
                         "Review every claim in the document and ensure each one is directly supported by "
                         "concrete evidence from the career profile. Remove or rephrase any statements that "
                         "embellish, generalize, or infer beyond what the evidence supports.")
        if doc_truth.evidence_examples:
            parts.append("Evidence examples to reference:\n" + "\n".join(f"- {e}" for e in doc_truth.evidence_examples[:8]))
        if doc_truth.suggestions:
            # Deduplicate: only include suggestions not already attempted
            new_suggestions = self._deduplicate(doc_truth.suggestions, previous_suggestions)
            if new_suggestions:
                parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in new_suggestions[:8]))
        if previous_suggestions:
            parts.append("Previously attempted suggestions (already tried — try a different approach):\n" + "\n".join(f"- {s}" for s in previous_suggestions[:8]))
        parts.append("Rewrite strictly using only evidence from the career profile and evidence pack.")
        return "\n\n".join(parts)

    def repair_voice(
        self,
        docs: DocumentSet,
        voice_review: VoiceReviewResult,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_suggestions: list[str] | None = None,
    ) -> DocumentSet:
        """Re-generate only documents that don't already have a 'strong' voice match."""
        doc_assessments: dict[DocumentKey, str] = {
            "cover_letter": voice_review.cover_letter_assessment,
            "resume": voice_review.resume_assessment,
            "interview_guide": voice_review.interview_guide_assessment,
        }
        doc_matches: dict[DocumentKey, str] = {
            "cover_letter": voice_review.cover_letter_match,
            "resume": voice_review.resume_match,
            "interview_guide": voice_review.interview_guide_match,
        }
        doc_issues: dict[DocumentKey, list[str]] = {
            "cover_letter": voice_review.cover_letter_issues,
            "resume": voice_review.resume_issues,
            "interview_guide": voice_review.interview_guide_issues,
        }
        doc_suggestions: dict[DocumentKey, list[str]] = {
            "cover_letter": voice_review.cover_letter_suggestions,
            "resume": voice_review.resume_suggestions,
            "interview_guide": voice_review.interview_guide_suggestions,
        }
        for key in ("cover_letter", "resume", "interview_guide"):
            if doc_matches[key] == "strong":
                continue  # already on-voice — skip
            previous = docs.get(key)
            if not previous:
                continue
            parts = []
            if feedback:
                parts.append(feedback)
            parts.append(f"Voice-match assessment: {doc_assessments[key]}")
            # Use per-doc issues/suggestions when available, fall back to aggregated
            issues = doc_issues[key] or voice_review.specific_issues
            suggestions = doc_suggestions[key] or voice_review.suggestions
            if issues:
                parts.append("Specific voice issues to fix:\n" + "\n".join(f"- {i}" for i in issues[:8]))
            if suggestions:
                new_suggestions = self._deduplicate(suggestions, previous_suggestions)
                if new_suggestions:
                    parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in new_suggestions[:8]))
            if previous_suggestions:
                parts.append("Previously attempted suggestions (already tried — try a different approach):\n" + "\n".join(f"- {s}" for s in previous_suggestions[:8]))
            parts.append("Rewrite to match the voice profile more closely. Keep all factual content intact.")
            repair_feedback = "\n\n".join(parts)
            regenerated = self.drafting_agent.generate_document(
                key, career, voice, job, context,
                feedback=repair_feedback, previous_version=previous,
            )
            docs.set(key, regenerated)
        return docs

    def repair_ai_detection(
        self,
        docs: DocumentSet,
        ai_review: AIDetectionResult,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
        previous_suggestions: list[str] | None = None,
    ) -> DocumentSet:
        """Re-generate documents flagged by the AI-detection review."""
        flag_map: dict[DocumentKey, list[str]] = {
            "cover_letter": ai_review.cover_letter_flags,
            "resume": ai_review.resume_flags,
            "interview_guide": ai_review.interview_guide_flags,
        }
        per_doc_suggestions: dict[DocumentKey, list[str]] = {
            "cover_letter": ai_review.cover_letter_suggestions,
            "resume": ai_review.resume_suggestions,
            "interview_guide": [],
        }
        for key, flags in flag_map.items():
            if not flags:
                continue
            previous = docs.get(key)
            if not previous:
                continue
            parts = []
            if feedback:
                parts.append(feedback)
            parts.append("AI-detection flagged the following phrases as generic or AI-sounding:\n" + "\n".join(f'- "{f}"' for f in flags[:8]))
            # Use per-doc suggestions; fall back to aggregated suggestions for backward compat
            suggestions = per_doc_suggestions.get(key) or ai_review.suggestions
            if suggestions:
                new_suggestions = self._deduplicate(suggestions, previous_suggestions)
                if new_suggestions:
                    parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in new_suggestions[:8]))
            if previous_suggestions:
                parts.append("Previously attempted suggestions (already tried — try a different approach):\n" + "\n".join(f"- {s}" for s in previous_suggestions[:8]))
            parts.append("Rewrite these passages to sound more authentically human. Use the voice profile. Keep all factual content intact.")
            repair_feedback = "\n\n".join(parts)
            regenerated = self.drafting_agent.generate_document(
                key, career, voice, job, context,
                feedback=repair_feedback, previous_version=previous,
            )
            docs.set(key, regenerated)
        return docs

    @staticmethod
    def _deduplicate(current: list[str], previous: list[str] | None) -> list[str]:
        """Return suggestions from *current* not already in *previous*."""
        if not previous:
            return current
        seen = {s.strip().lower() for s in previous}
        return [s for s in current if s.strip().lower() not in seen]
