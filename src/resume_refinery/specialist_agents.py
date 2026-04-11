"""Bounded specialist agents used by the workflow orchestrator."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

import ollama
from dotenv import load_dotenv

from .agent import ResumeRefineryAgent
from .models import (
    AIDetectionResult,
    ATSKeywordResult,
    CareerProfile,
    ConsistencyResult,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    EvidenceItem,
    EvidencePack,
    GrammarResult,
    HiringManagerReview,
    JobDescription,
    JobRequirement,
    RelevancePruningResult,
    RepairEdit,
    RepairPassResult,
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
_MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "8192"))
_MAX_WORKERS = int(os.environ.get("RESUME_REFINERY_MAX_WORKERS", "1"))

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

        def _match_one(req: JobRequirement) -> tuple[str, list[EvidenceItem]]:
            items = self._match_evidence(req.requirement, career.raw_content)
            return req.requirement, items

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(requirements) or 1)) as pool:
            futures = [pool.submit(_match_one, req) for req in requirements]
            for future in futures:
                req_text, evidence_items = future.result()
                if evidence_items:
                    matched.extend(evidence_items)
                else:
                    gaps.append(req_text)

        career_lines = self._career_lines(career.raw_content)
        summary = [line for line in career_lines if len(line) > 20][:15]
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
        for item in data[:15]:
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
        return requirements[:15]

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
        for entry in data[:5]:
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
        for rank, (_, evidence) in enumerate(scored[:5], start=1):
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
            "The career summary provides structure, contact details, and constraints only.**",
            "",
            "### Job Requirements",
        ]
        summary_lines.extend(f"- {item.requirement}" for item in evidence_pack.job_requirements)

        # Sort matched evidence by relevance score descending
        sorted_evidence = sorted(
            evidence_pack.matched_evidence,
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
            summary_lines.extend(f"- {gap}" for gap in evidence_pack.gaps)
            summary_lines.append(
                "\n**Important**: The gaps above are requirements from the job description "
                "that have no direct evidence in the career profile. Do NOT fabricate "
                "experience to cover them. Either omit them or frame related transferable "
                "skills honestly."
            )
        summary_lines.append("\n### Career Summary (structure & constraints)")
        summary_lines.append(self._compact_career_summary(career))
        return career.model_copy(update={"raw_content": "\n".join(summary_lines)})

    def _compact_career_summary(self, career: CareerProfile) -> str:
        """Build a compact career summary for the drafting prompt.

        Retains identity/contact, role timeline (titles/dates/tech only),
        education, certifications, skills, domain knowledge, story titles,
        and strategic meta (anti-claims, gaps, differentiators).
        Drops verbose role narratives already captured in the evidence pack.
        """
        raw = career.raw_content
        section_pattern = re.compile(r"^(## .+)$", re.MULTILINE)
        parts = section_pattern.split(raw)

        sections: list[tuple[str, str]] = [("_header", parts[0])]
        for i in range(1, len(parts), 2):
            heading = parts[i]
            content = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((heading, content))

        compact: list[str] = []
        for heading, content in sections:
            if heading == "_header":
                compact.append(content.strip())
            elif "Work Experience" in heading:
                compact.append(f"\n{heading}")
                compact.append("*(Narratives omitted — see Evidence Pack for details)*")
                in_anti = False
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("### "):
                        in_anti = False
                        compact.append(stripped)
                    elif stripped.startswith("**Technologies:**"):
                        in_anti = False
                        compact.append(stripped)
                    elif stripped.startswith("**Do NOT claim:**"):
                        in_anti = True
                        compact.append(stripped)
                    elif in_anti and stripped and not stripped.startswith("**") and stripped != "---":
                        compact.append(stripped)
                    elif stripped.startswith("**"):
                        in_anti = False
            elif "Key Stories" in heading:
                compact.append(f"\n{heading}")
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("### ") or stripped.startswith("Tags:") or stripped.startswith("**What this shows:**"):
                        compact.append(stripped)
            else:
                compact.append(f"\n{heading}")
                compact.append(content.rstrip())

        return "\n".join(compact)

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

    def review_all(self, docs: DocumentSet, career: CareerProfile, voice: VoiceProfile, job: JobDescription) -> ReviewBundle:
        return ReviewBundle(
            truthfulness=self.reviewer.review_truthfulness(docs, career, job),
            voice=self.reviewer.review_voice(docs, voice),
            ai_detection=self.reviewer.review_ai_detection(docs),
        )

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile, job: JobDescription) -> TruthfulnessResult:
        return self.reviewer.review_truthfulness(docs, career, job)

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile) -> VoiceReviewResult:
        return self.reviewer.review_voice(docs, voice)

    def review_ai_detection(self, docs: DocumentSet) -> AIDetectionResult:
        return self.reviewer.review_ai_detection(docs)

    def review_hiring_manager(self, docs: DocumentSet, job: JobDescription) -> HiringManagerReview:
        return self.reviewer.review_hiring_manager(docs, job)

    def review_relevance_pruning(self, docs: DocumentSet, job: JobDescription) -> RelevancePruningResult:
        return self.reviewer.review_relevance_pruning(docs, job)

    def review_ats_keyword(self, docs: DocumentSet, job: JobDescription, career: CareerProfile) -> ATSKeywordResult:
        return self.reviewer.review_ats_keyword(docs, job, career)

    def review_consistency(self, docs: DocumentSet) -> ConsistencyResult:
        return self.reviewer.review_consistency(docs)

    def review_grammar(self, docs: DocumentSet) -> GrammarResult:
        return self.reviewer.review_grammar(docs)


class RepairAgent:
    """Produces surgical find/replace edits and applies them programmatically."""

    def __init__(self, drafting_agent: DraftingAgent | None = None) -> None:
        # drafting_agent param retained for call-site compatibility; not used internally.
        self.client = ollama.Client(host=_BASE_URL)

    # ------------------------------------------------------------------
    # Public API (called by orchestrator)
    # ------------------------------------------------------------------

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
        hm_review: HiringManagerReview | None = None,
        pruning_review: RelevancePruningResult | None = None,
        ats_review: ATSKeywordResult | None = None,
        consistency_review: ConsistencyResult | None = None,
        grammar_review: GrammarResult | None = None,
        preserve_instructions: str | None = None,
        phase: str = "a",
        pass_num: int = 0,
        prior_edits: dict[str, str] | None = None,
    ) -> RepairPassResult:
        """Surgical repair: ask LLM for JSON edits, then apply programmatically."""
        from .models import DocumentEditHistory, EditRegion, ReviewerPriority
        from .prompts import REPAIR_SYSTEM_PROMPT, repair_user_message
        from .utils import apply_edits

        # Determine the dominant reviewer for this phase so that edit
        # regions are tagged with the correct priority.
        phase_reviewer: ReviewerPriority = self._phase_reviewer(phase, truth, ats_review, consistency_review, grammar_review, voice_review, ai_review)

        all_edits: dict[str, list[RepairEdit]] = {}
        all_regions: dict[str, list[EditRegion]] = {}
        all_accepted_claims: list[str] = []
        all_accepted_ai_phrases: list[str] = []
        all_accepted_voice_issues: list[str] = []
        all_accepted_hm_issues: list[str] = []
        all_accepted_pruning_issues: list[str] = []
        all_accepted_ats_issues: list[str] = []
        all_accepted_consistency_issues: list[str] = []
        all_accepted_grammar_issues: list[str] = []

        def _plan_for_key(key: str) -> tuple[str, list[dict], dict[str, list[str]]] | None:
            review_findings = self._build_review_findings(
                key, truth, voice_review, ai_review, feedback, hm_review, pruning_review,
                ats_review, consistency_review, grammar_review,
            )
            if not review_findings:
                return None
            if preserve_instructions:
                review_findings = preserve_instructions + "\n\n" + review_findings
            doc_content = docs.get(key)
            if not doc_content:
                return None

            user_msg = repair_user_message(
                doc_content=doc_content,
                career_profile=career.raw_content,
                voice_profile=voice.raw_content,
                job_description=job.raw_content,
                review_findings=review_findings,
                prior_edits=(prior_edits or {}).get(key, ""),
            )

            edits, acceptances = self._plan_edits(REPAIR_SYSTEM_PROMPT, user_msg)
            return key, edits, acceptances

        keys = ["cover_letter", "resume", "interview_guide"]
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, 3)) as pool:
            futures = [pool.submit(_plan_for_key, key) for key in keys]
            results = [f.result() for f in futures]

        for result in results:
            if result is None:
                continue
            key, edits, acceptances = result
            all_accepted_claims.extend(acceptances.get("accepted_claims", []))
            all_accepted_ai_phrases.extend(acceptances.get("accepted_ai_phrases", []))
            all_accepted_voice_issues.extend(acceptances.get("accepted_voice_issues", []))
            all_accepted_hm_issues.extend(acceptances.get("accepted_hm_issues", []))
            all_accepted_pruning_issues.extend(acceptances.get("accepted_pruning_issues", []))
            all_accepted_ats_issues.extend(acceptances.get("accepted_ats_issues", []))
            all_accepted_consistency_issues.extend(acceptances.get("accepted_consistency_issues", []))
            all_accepted_grammar_issues.extend(acceptances.get("accepted_grammar_issues", []))
            logging.debug(
                "[repair:%s] LLM returned %d edit(s), %d/%d/%d/%d/%d/%d/%d/%d accepted (claims/ai/voice/hm/pruning/ats/consistency/grammar)",
                key, len(edits),
                len(acceptances.get("accepted_claims", [])),
                len(acceptances.get("accepted_ai_phrases", [])),
                len(acceptances.get("accepted_voice_issues", [])),
                len(acceptances.get("accepted_hm_issues", [])),
                len(acceptances.get("accepted_pruning_issues", [])),
                len(acceptances.get("accepted_ats_issues", [])),
                len(acceptances.get("accepted_consistency_issues", [])),
                len(acceptances.get("accepted_grammar_issues", [])),
            )
            if edits:
                for i, e in enumerate(edits):
                    logging.debug(
                        "[repair:%s] edit %d/%d — find=%r  replace=%r  reason=%r",
                        key, i + 1, len(edits),
                        e.get("find", "")[:120],
                        e.get("replace", "")[:120],
                        e.get("reason", "")[:120],
                    )
                repaired, regions = apply_edits(
                    docs.get(key), edits,
                    reviewer=phase_reviewer,
                    pass_num=pass_num,
                )
                docs.set(key, repaired)
                all_regions[key] = regions
                all_edits[key] = [
                    RepairEdit(
                        find=e.get("find", ""),
                        replace=e.get("replace", ""),
                        reason=e.get("reason", ""),
                        reviewer=phase_reviewer,
                    )
                    for e in edits
                ]
        return RepairPassResult(
            edits=all_edits,
            edit_regions=all_regions,
            accepted_claims=all_accepted_claims,
            accepted_ai_phrases=all_accepted_ai_phrases,
            accepted_voice_issues=all_accepted_voice_issues,
            accepted_hm_issues=all_accepted_hm_issues,
            accepted_pruning_issues=all_accepted_pruning_issues,
            accepted_ats_issues=all_accepted_ats_issues,
            accepted_consistency_issues=all_accepted_consistency_issues,
            accepted_grammar_issues=all_accepted_grammar_issues,
        )

    @staticmethod
    def _phase_reviewer(
        phase: str,
        truth: TruthfulnessResult | None,
        ats_review: ATSKeywordResult | None,
        consistency_review: ConsistencyResult | None,
        grammar_review: GrammarResult | None,
        voice_review: VoiceReviewResult | None,
        ai_review: AIDetectionResult | None,
    ) -> str:
        """Return the highest-priority reviewer that has findings in this phase."""
        from .models import ReviewerPriority
        if phase == "a":
            # Phase A priority order: truth > consistency > ats > grammar
            if truth and not truth.all_supported:
                return "truthfulness"
            if consistency_review and not consistency_review.consistent:
                return "consistency"
            if ats_review and ats_review.alignment_score not in ("strong", "moderate"):
                return "ats"
            return "grammar"
        else:
            # Phase B priority order: voice > ai > hm > pruning
            if voice_review and voice_review.overall_match not in ("strong", "moderate"):
                return "voice"
            if ai_review and (ai_review.cover_letter_flags or ai_review.resume_flags):
                return "ai"
            return "voice"

    # ------------------------------------------------------------------
    # LLM call for edit planning
    # ------------------------------------------------------------------

    def _plan_edits(self, system: str, user_msg: str) -> tuple[list[dict], dict[str, list[str]]]:
        """Call the LLM and return (edits, acceptances).

        edits: list of {find, replace, reason} dicts.
        acceptances: dict with keys accepted_claims, accepted_ai_phrases,
            accepted_voice_issues, accepted_hm_issues, accepted_pruning_issues
            — verbatim phrases the repairer determined are reviewer false
            positives that should be suppressed going forward.
        """
        from .reviewers import _normalize_llm_json

        response = self.client.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            think=True,
            format={
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "find":    {"type": "string"},
                                "replace": {"type": "string"},
                                "reason":  {"type": "string"},
                            },
                            "required": ["find", "replace"],
                        },
                    },
                    "accepted_claims":        {"type": "array", "items": {"type": "string"}},
                    "accepted_ai_phrases":    {"type": "array", "items": {"type": "string"}},
                    "accepted_voice_issues":  {"type": "array", "items": {"type": "string"}},
                    "accepted_hm_issues":     {"type": "array", "items": {"type": "string"}},
                    "accepted_pruning_issues":{"type": "array", "items": {"type": "string"}},
                    "accepted_ats_issues":    {"type": "array", "items": {"type": "string"}},
                    "accepted_consistency_issues":{"type": "array", "items": {"type": "string"}},
                    "accepted_grammar_issues":{"type": "array", "items": {"type": "string"}},
                },
                "required": ["edits", "accepted_claims", "accepted_ai_phrases", "accepted_voice_issues", "accepted_hm_issues", "accepted_pruning_issues", "accepted_ats_issues", "accepted_consistency_issues", "accepted_grammar_issues"],
            },
            options={"num_ctx": _NUM_CTX, "num_predict": _MAX_TOKENS * 2},
        )
        raw = (response.message.content or "").strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not raw:
            logging.warning("Repair LLM returned empty content")
            return [], {"accepted_claims": [], "accepted_ai_phrases": [], "accepted_voice_issues": [], "accepted_hm_issues": [], "accepted_pruning_issues": [], "accepted_ats_issues": [], "accepted_consistency_issues": [], "accepted_grammar_issues": []}
        raw = _normalize_llm_json(raw)
        _empty: dict[str, list[str]] = {
            "accepted_claims": [], "accepted_ai_phrases": [], "accepted_voice_issues": [], "accepted_hm_issues": [], "accepted_pruning_issues": [], "accepted_ats_issues": [], "accepted_consistency_issues": [], "accepted_grammar_issues": []
        }

        def _extract_acceptances(d: dict) -> dict[str, list[str]]:
            return {
                k: [x for x in d.get(k, []) if isinstance(x, str)]
                for k in ("accepted_claims", "accepted_ai_phrases", "accepted_voice_issues", "accepted_hm_issues", "accepted_pruning_issues", "accepted_ats_issues", "accepted_consistency_issues", "accepted_grammar_issues")
            }

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("Repair LLM returned non-JSON after normalization; skipping repair pass")
            return [], dict(_empty)

        if isinstance(data, dict):
            edits = data.get("edits")
            if isinstance(edits, list):
                edits = self._filter_valid_edits(edits)
                return edits, _extract_acceptances(data)
            # Fallback: older bare-list keys the model may emit
            for k in ("changes", "replacements"):
                if isinstance(data.get(k), list):
                    logging.warning("Repair LLM returned object with '%s' key instead of 'edits'", k)
                    return self._filter_valid_edits(data[k]), _extract_acceptances(data)
            if "find" in data:
                logging.warning("Repair LLM returned a single edit object instead of object with 'edits'")
                return self._filter_valid_edits([data]), dict(_empty)
        if isinstance(data, list):
            # Backward compat: bare array (pre-schema)
            logging.warning("Repair LLM returned a bare array instead of an object with 'edits'")
            return self._filter_valid_edits(data), dict(_empty)
        return [], dict(_empty)

    @staticmethod
    def _filter_valid_edits(edits: list[dict]) -> list[dict]:
        """Drop malformed edit entries (missing or empty 'find')."""
        valid = []
        for e in edits:
            if not isinstance(e, dict):
                continue
            find = e.get("find", "")
            if not isinstance(find, str) or not find.strip():
                logging.debug("Dropping malformed edit (empty/missing find): %s", e)
                continue
            valid.append(e)
        return valid

    # ------------------------------------------------------------------
    # Build review findings text from reviewer results
    # ------------------------------------------------------------------

    def _build_review_findings(
        self,
        key: DocumentKey,
        truth: TruthfulnessResult | None,
        voice_review: VoiceReviewResult | None,
        ai_review: AIDetectionResult | None,
        feedback: str | None,
        hm_review: HiringManagerReview | None = None,
        pruning_review: RelevancePruningResult | None = None,
        ats_review: ATSKeywordResult | None = None,
        consistency_review: ConsistencyResult | None = None,
        grammar_review: GrammarResult | None = None,
    ) -> str:
        """Return a human-readable summary of review findings for *key*.

        Returns empty string if no issues were found for this document.
        """
        parts: list[str] = []
        has_issues = False

        if feedback:
            parts.append(f"USER FEEDBACK:\n{feedback}")
            has_issues = True

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
                    n = len(doc_truth.unsupported_claims)
                    logging.debug(
                        "[repair:%s] truthfulness: %d unsupported claim(s) — passing ALL to repair",
                        key, n,
                    )
                    parts.append(
                        "TRUTHFULNESS — Unsupported claims (verbatim from document):\n"
                        + "\n".join(f"- {c}" for c in doc_truth.unsupported_claims)
                    )
                else:
                    parts.append(
                        "TRUTHFULNESS — The truthfulness check failed but no specific "
                        "claims were listed. Review every factual claim."
                    )
                if doc_truth.evidence_examples:
                    parts.append(
                        "Supporting evidence from Career Profile:\n"
                        + "\n".join(f"- {e}" for e in doc_truth.evidence_examples)
                    )

        # --- Voice (cover letter and resume only) ---
        if voice_review and key in ("cover_letter", "resume"):
            voice_matches: dict[str, str] = {
                "cover_letter": voice_review.cover_letter_match,
                "resume": voice_review.resume_match,
            }
            if voice_matches[key] not in ("strong",):
                has_issues = True
                doc_issues_map: dict[str, list[str]] = {
                    "cover_letter": voice_review.cover_letter_issues,
                    "resume": voice_review.resume_issues,
                }
                issues = doc_issues_map[key] or voice_review.specific_issues
                if issues:
                    logging.debug(
                        "[repair:%s] voice: %d off-voice issue(s) — passing ALL to repair",
                        key, len(issues),
                    )
                    parts.append(
                        "VOICE — Off-voice phrases (verbatim from document):\n"
                        + "\n".join(f"- {i}" for i in issues)
                    )

        # --- AI detection (cover letter and resume only) ---
        if ai_review and key in ("cover_letter", "resume"):
            flag_map: dict[str, list[str]] = {
                "cover_letter": ai_review.cover_letter_flags,
                "resume": ai_review.resume_flags,
            }
            flags = flag_map.get(key, [])
            if flags:
                has_issues = True
                logging.debug(
                    "[repair:%s] ai-detection: %d flagged phrase(s) — passing ALL to repair",
                    key, len(flags),
                )
                parts.append(
                    "AI DETECTION — Flagged phrases (verbatim from document):\n"
                    + "\n".join(f'- "{f}"' for f in flags)
                )

        # --- Hiring manager (cover letter and resume only) ---
        if hm_review and key in ("cover_letter", "resume"):
            doc_issues = (
                hm_review.cover_letter_issues if key == "cover_letter"
                else hm_review.resume_issues
            )
            if doc_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] hiring-manager: %d issue(s) — passing ALL to repair",
                    key, len(doc_issues),
                )
                parts.append(
                    "HIRING MANAGER — Issues (verbatim from document):\n"
                    + "\n".join(
                        f'- "{i.phrase}" — {i.issue}. Suggestion: {i.suggestion}'
                        for i in doc_issues
                    )
                )

        # --- Relevance pruning (cover letter and resume only) ---
        if pruning_review and key in ("cover_letter", "resume"):
            doc_pruning_issues = (
                pruning_review.cover_letter_issues if key == "cover_letter"
                else pruning_review.resume_issues
            )
            if doc_pruning_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] relevance-pruning: %d issue(s) — passing ALL to repair",
                    key, len(doc_pruning_issues),
                )
                parts.append(
                    "RELEVANCE PRUNING — Content flagged for removal (verbatim from document):\n"
                    + "\n".join(
                        f'- "{i.phrase}" — {i.reason} (category: {i.category}, severity: {i.severity})'
                        for i in doc_pruning_issues
                    )
                )

        # --- ATS keyword alignment (resume only) ---
        if ats_review and key == "resume":
            ats_issues = ats_review.missing_keywords + ats_review.stuffing_keywords
            if ats_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] ats-keyword: %d issue(s) — passing ALL to repair",
                    key, len(ats_issues),
                )
                parts.append(
                    "ATS KEYWORD — Alignment issues:\n"
                    + "\n".join(
                        f'- [{i.issue_type.upper()}] "{i.keyword}" — section: {i.section}. Suggestion: {i.suggestion}'
                        for i in ats_issues
                    )
                )

        # --- Cross-document consistency (all documents) ---
        if consistency_review and consistency_review.issues:
            # Only include issues where this document is involved
            doc_consistency_issues = [
                i for i in consistency_review.issues
                if i.document_a == key or i.document_b == key
            ]
            if doc_consistency_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] consistency: %d issue(s) — passing ALL to repair",
                    key, len(doc_consistency_issues),
                )
                lines = []
                for i in doc_consistency_issues:
                    lines.append(
                        f'- {i.field}: "{i.quote_a}" (in {i.document_a}) vs "{i.quote_b}" (in {i.document_b}) — severity: {i.severity}'
                    )
                parts.append(
                    "CROSS-DOCUMENT CONSISTENCY — Contradictions involving this document:\n"
                    + "\n".join(lines)
                )

        # --- Grammar & mechanics ---
        if grammar_review and key in ("cover_letter", "resume", "interview_guide"):
            doc_grammar_map = {
                "cover_letter": grammar_review.cover_letter_issues,
                "resume": grammar_review.resume_issues,
                "interview_guide": grammar_review.interview_guide_issues,
            }
            doc_grammar_issues = doc_grammar_map.get(key, [])
            if doc_grammar_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] grammar: %d issue(s) — passing ALL to repair",
                    key, len(doc_grammar_issues),
                )
                parts.append(
                    "GRAMMAR & MECHANICS — Issues (verbatim from document):\n"
                    + "\n".join(
                        f'- "{i.phrase}" — {i.issue}. Suggestion: {i.suggestion} (category: {i.category})'
                        for i in doc_grammar_issues
                    )
                )

        if not has_issues:
            logging.debug("[repair:%s] no issues found — skipping repair for this document", key)
            return ""

        findings = "\n\n".join(parts)
        logging.debug(
            "[repair:%s] full review findings sent to LLM (%d chars):\n%s",
            key, len(findings), findings,
        )
        return findings
