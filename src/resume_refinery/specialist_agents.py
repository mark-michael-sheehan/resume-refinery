"""Bounded specialist agents used by the workflow orchestrator."""

from __future__ import annotations

import re
from typing import Iterable, Iterator

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
from .reviewers import DocumentReviewer

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
    """Extracts job requirements and grounded evidence from raw inputs."""

    def build_evidence_pack(self, career: CareerProfile, job: JobDescription) -> EvidencePack:
        requirements = self._extract_requirements(job.raw_content)
        career_lines = self._career_lines(career.raw_content)
        matched: list[EvidenceItem] = []
        gaps: list[str] = []

        for requirement in requirements:
            evidence_lines = self._match_evidence(requirement.requirement, career_lines)
            if evidence_lines:
                for score, evidence in enumerate(evidence_lines[:3], start=1):
                    matched.append(
                        EvidenceItem(
                            requirement=requirement.requirement,
                            evidence=evidence,
                            source_excerpt=evidence,
                            relevance_score=max(1, 6 - score),
                        )
                    )
            else:
                gaps.append(requirement.requirement)

        summary = [line for line in career_lines if len(line) > 20][:8]
        return EvidencePack(
            job_requirements=requirements,
            matched_evidence=matched,
            gaps=gaps,
            source_summary=summary,
        )

    def _extract_requirements(self, raw_job: str) -> list[JobRequirement]:
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

    def _match_evidence(self, requirement: str, career_lines: list[str]) -> list[str]:
        req_keywords = self._keywords(requirement)
        scored: list[tuple[int, str]] = []
        for line in career_lines:
            line_keywords = self._keywords(line)
            overlap = len(req_keywords & line_keywords)
            if overlap:
                scored.append((overlap, line))
        scored.sort(key=lambda item: (-item[0], -len(item[1])))
        return [line for _, line in scored]

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
            "### Top Job Requirements",
        ]
        summary_lines.extend(f"- {item.requirement}" for item in evidence_pack.job_requirements[:8])
        summary_lines.append("\n### Matched Evidence")
        summary_lines.extend(
            f"- Requirement: {item.requirement} | Evidence: {item.evidence}"
            for item in evidence_pack.matched_evidence[:12]
        )
        if evidence_pack.gaps:
            summary_lines.append("\n### Potential Gaps")
            summary_lines.extend(f"- {gap}" for gap in evidence_pack.gaps[:6])
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

    def repair_documents(
        self,
        docs: DocumentSet,
        truth: TruthfulnessResult,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        context: DraftingContext,
        feedback: str | None = None,
    ) -> DocumentSet:
        for key in self._docs_to_fix(truth):
            previous = docs.get(key)
            repair_feedback = self._feedback_for_doc(key, truth, feedback)
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

    def _feedback_for_doc(self, key: DocumentKey, truth: TruthfulnessResult, feedback: str | None) -> str:
        if key == "cover_letter":
            claims = truth.cover_letter.unsupported_claims
        elif key == "resume":
            claims = truth.resume.unsupported_claims
        else:
            claims = truth.interview_guide.unsupported_claims
        parts = []
        if feedback:
            parts.append(feedback)
        if claims:
            parts.append("Unsupported claims to remove or directly evidence:\n" + "\n".join(f"- {claim}" for claim in claims[:8]))
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
    ) -> DocumentSet:
        """Re-generate documents flagged by the voice-match review."""
        doc_assessments: dict[DocumentKey, str] = {
            "cover_letter": voice_review.cover_letter_assessment,
            "resume": voice_review.resume_assessment,
            "interview_guide": voice_review.interview_guide_assessment,
        }
        for key in ("cover_letter", "resume", "interview_guide"):
            previous = docs.get(key)
            if not previous:
                continue
            parts = []
            if feedback:
                parts.append(feedback)
            parts.append(f"Voice-match assessment: {doc_assessments[key]}")
            if voice_review.specific_issues:
                parts.append("Specific voice issues to fix:\n" + "\n".join(f"- {i}" for i in voice_review.specific_issues[:8]))
            if voice_review.suggestions:
                parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in voice_review.suggestions[:8]))
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
    ) -> DocumentSet:
        """Re-generate documents flagged by the AI-detection review."""
        flag_map: dict[DocumentKey, list[str]] = {
            "cover_letter": ai_review.cover_letter_flags,
            "resume": ai_review.resume_flags,
            "interview_guide": ai_review.interview_guide_flags,
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
            if ai_review.suggestions:
                parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in ai_review.suggestions[:8]))
            parts.append("Rewrite these passages to sound more authentically human. Use the voice profile. Keep all factual content intact.")
            repair_feedback = "\n\n".join(parts)
            regenerated = self.drafting_agent.generate_document(
                key, career, voice, job, context,
                feedback=repair_feedback, previous_version=previous,
            )
            docs.set(key, regenerated)
        return docs
