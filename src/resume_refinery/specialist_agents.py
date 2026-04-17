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
    CandidacyNarrative,
    CareerProfile,
    CoverageGap,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    GrammarResult,
    HiringManagerReview,
    JobDescription,
    NarrativeCoherenceResult,
    NarrativeCoverageResult,
    NarrativePillar,
    RelevancePruningResult,
    RepairEdit,
    RepairPassResult,
    ReviewBundle,
    SupportingEvidence,
    TruthfulnessResult,
    VoiceProfile,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from .prompts import (
    NARRATIVE_SYSTEM_PROMPT,
    NARRATIVE_USER_TEMPLATE,
    NARRATIVE_CRITIQUE_SYSTEM_PROMPT,
    NARRATIVE_CRITIQUE_USER_TEMPLATE,
    NARRATIVE_REVISION_USER_TEMPLATE,
)
from .reviewers import DocumentReviewer

load_dotenv()

_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")
_NUM_CTX = int(os.environ.get("RESUME_REFINERY_NUM_CTX", "16384"))
_MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "8192"))
_MAX_WORKERS = int(os.environ.get("RESUME_REFINERY_MAX_WORKERS", "1"))
_MAX_NARRATIVE_CRITIQUE_PASSES = int(os.environ.get("RESUME_REFINERY_MAX_NARRATIVE_CRITIQUE_PASSES", "2"))

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


class NarrativeAgent:
    """Builds a candidacy narrative from career profile and job description.

    The narrative frames why the applicant is a strong fit and guides
    resume generation.
    """

    def __init__(self, client: ollama.Client | None = None) -> None:
        self.client = client or ollama.Client(host=_BASE_URL)

    def build_narrative(
        self,
        career: CareerProfile,
        job: JobDescription,
        *,
        max_critique_passes: int = _MAX_NARRATIVE_CRITIQUE_PASSES,
        progress: object | None = None,
    ) -> CandidacyNarrative:
        try:
            narrative = self._build_narrative_llm(career.raw_content, job.raw_content)
        except Exception as exc:
            logging.warning("LLM narrative generation failed (%s); using keyword fallback.", exc)
            return self._build_narrative_fallback(career.raw_content, job.raw_content)

        if max_critique_passes <= 0:
            return narrative

        critic = NarrativeCriticAgent(client=self.client)
        for pass_num in range(max_critique_passes):
            try:
                critique = critic.critique(narrative, career, job)
            except Exception as exc:
                logging.warning("Narrative critique pass %d failed (%s); keeping current narrative.", pass_num, exc)
                break

            if critique.get("passes", False):
                logging.info("Narrative passed critique on pass %d.", pass_num)
                break

            issues = critique.get("issues", [])
            if not issues:
                break

            logging.info(
                "Narrative critique pass %d found %d issue(s); revising.",
                pass_num, len(issues),
            )

            try:
                narrative = self._revise_narrative(narrative, career, job, issues)
            except Exception as exc:
                logging.warning("Narrative revision failed on pass %d (%s); keeping current.", pass_num, exc)
                break

        return narrative

    def _revise_narrative(
        self,
        narrative: CandidacyNarrative,
        career: CareerProfile,
        job: JobDescription,
        issues: list[dict],
    ) -> CandidacyNarrative:
        """Ask the LLM to revise the narrative based on critique findings."""
        pillars_text = "\n".join(
            f"- **{p.theme}**: {p.argument} (evidence: {', '.join(ev.evidence for ev in p.career_evidence)})"
            for p in narrative.pillars
        )
        gap_text = "\n".join(f"- {g}" for g in narrative.gap_framing) or "None"

        critique_text = "\n".join(
            f"- [{i.get('severity', 'medium').upper()}] {i.get('criterion', 'unknown')}: "
            f"{i.get('description', '')} → Suggestion: {i.get('suggestion', '')}"
            for i in issues
        )

        user_msg = NARRATIVE_REVISION_USER_TEMPLATE.format(
            career_profile=career.raw_content,
            job_description=job.raw_content,
            thesis=narrative.thesis,
            pillars=pillars_text,
            gap_framing=gap_text,
            raw_narrative=narrative.raw_narrative or "(none)",
            critique_findings=critique_text,
        )
        raw = self._call_llm(NARRATIVE_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        pillars: list[NarrativePillar] = []
        for p in data.get("pillars", [])[:5]:
            if isinstance(p, dict) and "theme" in p:
                raw_evidence = p.get("career_evidence", [])
                evidence: list[SupportingEvidence] = []
                for item in raw_evidence:
                    if isinstance(item, dict):
                        evidence.append(SupportingEvidence(
                            evidence=item.get("evidence", ""),
                            justification=item.get("justification", ""),
                        ))
                    elif isinstance(item, str):
                        evidence.append(SupportingEvidence(evidence=item))
                pillars.append(
                    NarrativePillar(
                        theme=p["theme"],
                        argument=p.get("argument", ""),
                        career_evidence=evidence,
                    )
                )

        return CandidacyNarrative(
            thesis=data.get("thesis", ""),
            pillars=pillars,
            gap_framing=data.get("gap_framing", []),
            raw_narrative=data.get("raw_narrative", ""),
        )

    def _build_narrative_llm(self, career_content: str, job_content: str) -> CandidacyNarrative:
        user_msg = NARRATIVE_USER_TEMPLATE.format(
            career_profile=career_content,
            job_description=job_content,
        )
        raw = self._call_llm(NARRATIVE_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        pillars: list[NarrativePillar] = []
        for p in data.get("pillars", [])[:5]:
            if isinstance(p, dict) and "theme" in p:
                raw_evidence = p.get("career_evidence", [])
                evidence: list[SupportingEvidence] = []
                for item in raw_evidence:
                    if isinstance(item, dict):
                        evidence.append(SupportingEvidence(
                            evidence=item.get("evidence", ""),
                            justification=item.get("justification", ""),
                        ))
                    elif isinstance(item, str):
                        evidence.append(SupportingEvidence(evidence=item))
                pillars.append(
                    NarrativePillar(
                        theme=p["theme"],
                        argument=p.get("argument", ""),
                        career_evidence=evidence,
                    )
                )

        return CandidacyNarrative(
            thesis=data.get("thesis", ""),
            pillars=pillars,
            gap_framing=data.get("gap_framing", []),
            raw_narrative=data.get("raw_narrative", ""),
        )

    def _build_narrative_fallback(self, career_content: str, job_content: str) -> CandidacyNarrative:
        """Keyword heuristic fallback for narrative generation."""
        career_keywords = self._keywords(career_content)
        job_keywords = self._keywords(job_content)
        overlap = career_keywords & job_keywords
        gaps = job_keywords - career_keywords

        pillars = []
        overlap_list = sorted(overlap)[:5]
        for kw in overlap_list:
            pillars.append(
                NarrativePillar(
                    theme=kw.title(),
                    argument=f"Candidate has demonstrated experience with {kw}.",
                    career_evidence=[SupportingEvidence(
                        evidence=kw,
                        justification=f"Keyword '{kw}' appears in both career profile and job description.",
                    )],
                )
            )

        gap_framing = [
            f"Gap: {g} — consider highlighting transferable skills"
            for g in sorted(gaps)[:3]
        ]

        return CandidacyNarrative(
            thesis="Candidate's experience aligns with key role requirements.",
            pillars=pillars,
            gap_framing=gap_framing,
            raw_narrative="",
        )

    def _call_llm(self, system: str, user_msg: str) -> str:
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
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        from .reviewers import _normalize_llm_json
        raw = _normalize_llm_json(raw)
        if not raw:
            raise ValueError("LLM returned empty content")
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        return raw

    def _keywords(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
            if token not in _STOPWORDS
        }


class NarrativeCriticAgent:
    """Critiques a candidacy narrative for quality, completeness, and strategic fitness.

    Used by NarrativeAgent in a tight self-critique loop to strengthen the
    narrative before it reaches the DraftingAgent.
    """

    def __init__(self, client: ollama.Client | None = None) -> None:
        self.client = client or ollama.Client(host=_BASE_URL)

    def critique(
        self,
        narrative: CandidacyNarrative,
        career: CareerProfile,
        job: JobDescription,
    ) -> dict:
        """Return a critique dict with 'passes' bool and 'issues' list."""
        pillars_text = "\n".join(
            f"- **{p.theme}**: {p.argument} (evidence: {', '.join(ev.evidence for ev in p.career_evidence)})"
            for p in narrative.pillars
        )
        gap_text = "\n".join(f"- {g}" for g in narrative.gap_framing) or "None"

        user_msg = NARRATIVE_CRITIQUE_USER_TEMPLATE.format(
            career_profile=career.raw_content,
            job_description=job.raw_content,
            thesis=narrative.thesis,
            pillars=pillars_text,
            gap_framing=gap_text,
            raw_narrative=narrative.raw_narrative or "(none)",
        )
        raw = self._call_llm(NARRATIVE_CRITIQUE_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        return data

    def _call_llm(self, system: str, user_msg: str) -> str:
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
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        from .reviewers import _normalize_llm_json
        raw = _normalize_llm_json(raw)
        if not raw:
            raise ValueError("LLM returned empty content")
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        return raw


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


class NarrativeCoverageAgent:
    """Identifies narrative pillars not reflected in the resume despite having career evidence.

    Runs after initial drafting but before the review loop. Advisory only —
    its suggestions are applied to enrich the resume but never block convergence.
    """

    def __init__(self, client: ollama.Client | None = None) -> None:
        self.client = client or ollama.Client(host=_BASE_URL)

    def analyze_coverage(
        self,
        narrative: CandidacyNarrative,
        career: CareerProfile,
        docs: DocumentSet,
        job: JobDescription,
    ) -> NarrativeCoverageResult:
        """Compare narrative pillars against the resume and return gaps."""
        if not narrative.pillars or not docs.resume:
            return NarrativeCoverageResult(
                pillars_total=len(narrative.pillars),
                pillars_covered=len(narrative.pillars),
                coverage_summary="No pillars or resume to compare.",
            )
        try:
            return self._analyze_llm(narrative, career, docs, job)
        except Exception as exc:
            logging.warning("LLM coverage analysis failed (%s); using fallback.", exc)
            return self._analyze_fallback(narrative, docs)

    def _analyze_llm(
        self,
        narrative: CandidacyNarrative,
        career: CareerProfile,
        docs: DocumentSet,
        job: JobDescription,
    ) -> NarrativeCoverageResult:
        from .prompts import NARRATIVE_COVERAGE_SYSTEM_PROMPT, NARRATIVE_COVERAGE_USER_TEMPLATE

        pillars_text = "\n".join(
            f"- **{p.theme}**: {p.argument} (evidence: {', '.join(ev.evidence for ev in p.career_evidence)})"
            for p in narrative.pillars
        )
        gap_text = "\n".join(f"- {g}" for g in narrative.gap_framing) or "None"

        user_msg = NARRATIVE_COVERAGE_USER_TEMPLATE.format(
            thesis=narrative.thesis,
            pillars=pillars_text,
            gap_framing=gap_text,
            career_profile=career.raw_content,
            resume=docs.resume or "",
            job_description=job.raw_content,
        )
        raw = self._call_llm(NARRATIVE_COVERAGE_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        gaps: list[CoverageGap] = []
        for g in data.get("gaps", [])[:5]:
            if isinstance(g, dict) and "pillar_theme" in g:
                gaps.append(
                    CoverageGap(
                        pillar_theme=g["pillar_theme"],
                        career_evidence=g.get("career_evidence", []),
                        suggested_content=g.get("suggested_content", ""),
                        anchor_section=g.get("anchor_section", ""),
                        confidence=g.get("confidence", "medium"),
                    )
                )

        return NarrativeCoverageResult(
            gaps=gaps,
            coverage_summary=data.get("coverage_summary", ""),
            pillars_covered=data.get("pillars_covered", 0),
            pillars_total=data.get("pillars_total", len(narrative.pillars)),
        )

    def _analyze_fallback(
        self,
        narrative: CandidacyNarrative,
        docs: DocumentSet,
    ) -> NarrativeCoverageResult:
        """Keyword heuristic: check if each pillar's theme appears in the resume."""
        resume_lower = (docs.resume or "").lower()
        gaps: list[CoverageGap] = []
        covered = 0
        for pillar in narrative.pillars:
            theme_words = set(pillar.theme.lower().split())
            if any(w in resume_lower for w in theme_words if len(w) > 3):
                covered += 1
            elif pillar.career_evidence:
                gaps.append(
                    CoverageGap(
                        pillar_theme=pillar.theme,
                        career_evidence=[ev.evidence for ev in pillar.career_evidence[:3]],
                        suggested_content=f"Consider adding evidence related to: {pillar.theme}",
                        anchor_section="Experience",
                        confidence="low",
                    )
                )
            else:
                covered += 1  # No evidence to draw from, skip

        return NarrativeCoverageResult(
            gaps=gaps,
            coverage_summary=f"{covered} of {len(narrative.pillars)} pillars covered.",
            pillars_covered=covered,
            pillars_total=len(narrative.pillars),
        )

    def apply_suggestions(self, docs: DocumentSet, result: NarrativeCoverageResult) -> DocumentSet:
        """Insert suggested content into the resume at anchor sections.

        Modifies docs in place and returns it for chaining.
        """
        if not result.gaps or not docs.resume:
            return docs

        resume = docs.resume
        for gap in result.gaps:
            if not gap.suggested_content or not gap.anchor_section:
                continue
            # Find the anchor section heading and insert after it
            pattern = re.compile(
                r"(^#{1,3}\s+" + re.escape(gap.anchor_section) + r".*$)",
                re.MULTILINE | re.IGNORECASE,
            )
            match = pattern.search(resume)
            if match:
                insert_pos = match.end()
                resume = resume[:insert_pos] + "\n" + gap.suggested_content + resume[insert_pos:]
        docs.resume = resume
        return docs

    def _call_llm(self, system: str, user_msg: str) -> str:
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
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        from .reviewers import _normalize_llm_json
        raw = _normalize_llm_json(raw)
        if not raw:
            raise ValueError("LLM returned empty content")
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        return raw


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
        docs.set("resume", self.generate_document("resume", career, voice, job, context))
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
            self._career_context(career, context.narrative),
            self._voice_context(voice, context.voice_style_guide),
            job,
            narrative_text=context.narrative.raw_narrative if context.narrative else "",
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
            self._career_context(career, context.narrative),
            self._voice_context(voice, context.voice_style_guide),
            job,
            narrative_text=context.narrative.raw_narrative if context.narrative else "",
            feedback=feedback,
            previous_version=previous_version,
        )

    def _career_context(self, career: CareerProfile, narrative: CandidacyNarrative | None) -> CareerProfile:
        if not narrative:
            return career

        summary_lines = [
            "## Candidacy Narrative",
            "**Use the narrative below as your PRIMARY guide for emphasis, ordering, and framing.**",
            "",
            f"### Thesis\n{narrative.thesis}",
            "",
            "### Supporting Pillars",
        ]
        for pillar in narrative.pillars:
            summary_lines.append(f"**{pillar.theme}**: {pillar.argument}")
            for ev in pillar.career_evidence:
                line = f"  - {ev.evidence}"
                if ev.justification:
                    line += f" — {ev.justification}"
                summary_lines.append(line)
        if narrative.gap_framing:
            summary_lines.append("\n### Gap Framing")
            for gap in narrative.gap_framing:
                summary_lines.append(f"- {gap}")
            summary_lines.append(
                "\n**Important**: The gaps above are areas where the candidate's experience "
                "does not perfectly match. Do NOT fabricate experience to cover them. "
                "Instead, frame related transferable skills honestly."
            )
        summary_lines.append("\n### Career Profile")
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

    def review_all(self, docs: DocumentSet, career: CareerProfile, voice: VoiceProfile, job: JobDescription) -> ReviewBundle:
        return ReviewBundle(
            truthfulness=self.reviewer.review_truthfulness(docs, career, job),
            voice=self.reviewer.review_voice(docs, voice),
            ai_detection=self.reviewer.review_ai_detection(docs),
        )

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile, job: JobDescription, *, exemptions: list[str] | None = None) -> TruthfulnessResult:
        return self.reviewer.review_truthfulness(docs, career, job, exemptions=exemptions)

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile, *, exemptions: list[str] | None = None) -> VoiceReviewResult:
        return self.reviewer.review_voice(docs, voice, exemptions=exemptions)

    def review_ai_detection(self, docs: DocumentSet, *, exemptions: list[str] | None = None) -> AIDetectionResult:
        return self.reviewer.review_ai_detection(docs, exemptions=exemptions)

    def review_hiring_manager(self, docs: DocumentSet, job: JobDescription, *, exemptions: list[str] | None = None) -> HiringManagerReview:
        return self.reviewer.review_hiring_manager(docs, job, exemptions=exemptions)

    def review_relevance_pruning(self, docs: DocumentSet, job: JobDescription, *, exemptions: list[str] | None = None) -> RelevancePruningResult:
        return self.reviewer.review_relevance_pruning(docs, job, exemptions=exemptions)

    def review_ats_keyword(self, docs: DocumentSet, job: JobDescription, career: CareerProfile, *, exemptions: list[str] | None = None) -> ATSKeywordResult:
        return self.reviewer.review_ats_keyword(docs, job, career, exemptions=exemptions)

    def review_grammar(self, docs: DocumentSet, *, exemptions: list[str] | None = None) -> GrammarResult:
        return self.reviewer.review_grammar(docs, exemptions=exemptions)

    def review_narrative_coherence(self, docs: DocumentSet, narrative: CandidacyNarrative, *, exemptions: list[str] | None = None) -> NarrativeCoherenceResult:
        return self.reviewer.review_narrative_coherence(docs, narrative, exemptions=exemptions)


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
        grammar_review: GrammarResult | None = None,
        narrative_review: NarrativeCoherenceResult | None = None,
        preserve_instructions: str | None = None,
        phase: str = "a",
        pass_num: int = 0,
        prior_edits: dict[str, str] | None = None,
    ) -> RepairPassResult:
        """Surgical repair: ask LLM for JSON edits, then apply programmatically."""
        from .models import DocumentEditHistory, EditRegion, ReviewerPriority
        from .prompts import REPAIR_SYSTEM_PROMPT, repair_user_message
        from .utils import apply_edits

        phase_reviewer: ReviewerPriority = self._phase_reviewer(phase, truth, ats_review, grammar_review, voice_review, ai_review)

        all_edits: dict[str, list[RepairEdit]] = {}
        all_regions: dict[str, list[EditRegion]] = {}
        all_failed_edits: dict[str, list[dict]] = {}
        all_accepted_claims: list[str] = []
        all_accepted_ai_phrases: list[str] = []
        all_accepted_voice_issues: list[str] = []
        all_accepted_hm_issues: list[str] = []
        all_accepted_pruning_issues: list[str] = []
        all_accepted_ats_issues: list[str] = []
        all_accepted_grammar_issues: list[str] = []
        all_accepted_narrative_issues: list[str] = []

        def _plan_for_key(key: str) -> tuple[str, list[dict], dict[str, list[str]]] | None:
            review_findings = self._build_review_findings(
                key, truth, voice_review, ai_review, feedback, hm_review, pruning_review,
                ats_review, grammar_review, narrative_review,
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

        keys = ["resume"]
        with ThreadPoolExecutor(max_workers=1) as pool:
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
            all_accepted_grammar_issues.extend(acceptances.get("accepted_grammar_issues", []))
            all_accepted_narrative_issues.extend(acceptances.get("accepted_narrative_issues", []))
            logging.debug(
                "[repair:%s] LLM returned %d edit(s), %d/%d/%d/%d/%d/%d/%d/%d accepted (claims/ai/voice/hm/pruning/ats/grammar/narrative)",
                key, len(edits),
                len(acceptances.get("accepted_claims", [])),
                len(acceptances.get("accepted_ai_phrases", [])),
                len(acceptances.get("accepted_voice_issues", [])),
                len(acceptances.get("accepted_hm_issues", [])),
                len(acceptances.get("accepted_pruning_issues", [])),
                len(acceptances.get("accepted_ats_issues", [])),
                len(acceptances.get("accepted_grammar_issues", [])),
                len(acceptances.get("accepted_narrative_issues", [])),
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
                repaired, regions, edit_failures = apply_edits(
                    docs.get(key), edits,
                    reviewer=phase_reviewer,
                    pass_num=pass_num,
                    merge_fn=self._merge_overlapping_edits,
                )
                docs.set(key, repaired)
                all_regions[key] = regions
                if edit_failures:
                    all_failed_edits[key] = edit_failures
                all_edits[key] = [
                    RepairEdit(
                        find=e.get("find", ""),
                        replace=e.get("replace", ""),
                        reason=e.get("reason", ""),
                        reviewer=phase_reviewer,
                        insert_after=bool(e.get("insert_after", False)),
                    )
                    for e in edits
                ]
        return RepairPassResult(
            edits=all_edits,
            edit_regions=all_regions,
            failed_edits=all_failed_edits,
            accepted_claims=all_accepted_claims,
            accepted_ai_phrases=all_accepted_ai_phrases,
            accepted_voice_issues=all_accepted_voice_issues,
            accepted_hm_issues=all_accepted_hm_issues,
            accepted_pruning_issues=all_accepted_pruning_issues,
            accepted_ats_issues=all_accepted_ats_issues,
            accepted_grammar_issues=all_accepted_grammar_issues,
            accepted_narrative_issues=all_accepted_narrative_issues,
        )

    @staticmethod
    def _phase_reviewer(
        phase: str,
        truth: TruthfulnessResult | None,
        ats_review: ATSKeywordResult | None,
        grammar_review: GrammarResult | None,
        voice_review: VoiceReviewResult | None,
        ai_review: AIDetectionResult | None,
    ) -> str:
        """Return the highest-priority reviewer that has findings."""
        if truth and not truth.all_supported:
            return "truthfulness"
        if ats_review and ats_review.alignment_score not in ("strong", "moderate"):
            return "ats"
        if grammar_review and not grammar_review.clean:
            return "grammar"
        if voice_review and voice_review.overall_match not in ("strong", "moderate"):
            return "voice"
        if ai_review and ai_review.resume_flags:
            return "ai"
        return "grammar"

    # ------------------------------------------------------------------
    # LLM call for merging overlapping edits
    # ------------------------------------------------------------------

    def _merge_overlapping_edits(
        self, context_text: str, overlapping_edits: list[dict]
    ) -> dict | None:
        """Merge overlapping edits via a lightweight LLM call.

        Called by ``apply_edits`` when two or more edits' ``find`` spans
        overlap in the original document.  Returns a single merged
        ``{find, replace, reason}`` dict, or ``None`` on failure.
        """
        from .prompts import MERGE_EDITS_SYSTEM_PROMPT, MERGE_EDITS_USER_TEMPLATE
        from .reviewers import _normalize_llm_json

        edits_desc = "\n".join(
            f'{i+1}. find: {e.get("find", "")!r}\n'
            f'   replace: {e.get("replace", "")!r}\n'
            f'   reason: {e.get("reason", "N/A")!r}'
            for i, e in enumerate(overlapping_edits)
        )
        user_msg = MERGE_EDITS_USER_TEMPLATE.format(
            context_text=context_text,
            edits_description=edits_desc,
        )

        try:
            response = self.client.chat(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": MERGE_EDITS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format={
                    "type": "object",
                    "properties": {
                        "find": {"type": "string"},
                        "replace": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["find", "replace"],
                },
                options={"num_ctx": _NUM_CTX, "num_predict": _MAX_TOKENS},
            )
        except Exception as exc:
            logging.warning("Merge LLM call failed: %s", exc)
            return None

        raw = (response.message.content or "").strip()
        if not raw:
            return None
        raw = _normalize_llm_json(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("Merge LLM returned non-JSON: %.200s", raw)
            return None

        if not isinstance(data, dict) or "replace" not in data:
            return None

        # Force the find to be the context_text — the LLM may echo it
        # imperfectly.
        return {
            "find": context_text,
            "replace": data.get("replace", ""),
            "reason": data.get("reason", "merged overlapping edits"),
        }

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
                                "insert_after": {"type": "boolean"},
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
                    "accepted_grammar_issues":{"type": "array", "items": {"type": "string"}},
                    "accepted_narrative_issues":{"type": "array", "items": {"type": "string"}},
                },
                "required": ["edits", "accepted_claims", "accepted_ai_phrases", "accepted_voice_issues", "accepted_hm_issues", "accepted_pruning_issues", "accepted_ats_issues", "accepted_grammar_issues", "accepted_narrative_issues"],
            },
            options={"num_ctx": _NUM_CTX, "num_predict": _MAX_TOKENS * 2},
        )
        raw = (response.message.content or "").strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not raw:
            logging.warning("Repair LLM returned empty content")
            return [], {"accepted_claims": [], "accepted_ai_phrases": [], "accepted_voice_issues": [], "accepted_hm_issues": [], "accepted_pruning_issues": [], "accepted_ats_issues": [], "accepted_grammar_issues": [], "accepted_narrative_issues": []}
        raw = _normalize_llm_json(raw)
        _empty: dict[str, list[str]] = {
            "accepted_claims": [], "accepted_ai_phrases": [], "accepted_voice_issues": [], "accepted_hm_issues": [], "accepted_pruning_issues": [], "accepted_ats_issues": [], "accepted_grammar_issues": [], "accepted_narrative_issues": []
        }

        def _extract_acceptances(d: dict) -> dict[str, list[str]]:
            return {
                k: [x for x in d.get(k, []) if isinstance(x, str)]
                for k in ("accepted_claims", "accepted_ai_phrases", "accepted_voice_issues", "accepted_hm_issues", "accepted_pruning_issues", "accepted_ats_issues", "accepted_grammar_issues", "accepted_narrative_issues")
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
        grammar_review: GrammarResult | None = None,
        narrative_review: NarrativeCoherenceResult | None = None,
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
            doc_truth = truth.resume
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

        # --- Voice ---
        if voice_review:
            if voice_review.resume_match not in ("strong",):
                has_issues = True
                issues = voice_review.resume_issues or voice_review.specific_issues
                if issues:
                    logging.debug(
                        "[repair:%s] voice: %d off-voice issue(s) — passing ALL to repair",
                        key, len(issues),
                    )
                    parts.append(
                        "VOICE — Off-voice phrases (verbatim from document):\n"
                        + "\n".join(f"- {i}" for i in issues)
                    )

        # --- AI detection ---
        if ai_review:
            flags = ai_review.resume_flags
            if flags:
                has_issues = True
                logging.debug(
                    "[repair:%s] ai-detection: %d flagged phrase(s) — passing ALL to repair",
                    key, len(flags),
                )
                parts.append(
                    "AI DETECTION — Flagged phrases (verbatim from document):\n"
                    + "\n".join(f'"- "{f}"' for f in flags)
                )

        # --- Hiring manager ---
        if hm_review:
            doc_issues = hm_review.resume_issues
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

        # --- Relevance pruning ---
        if pruning_review:
            doc_pruning_issues = pruning_review.resume_issues
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

        # --- Grammar & mechanics ---
        if grammar_review:
            doc_grammar_issues = grammar_review.resume_issues
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

        # --- Narrative coherence ---
        if narrative_review:
            if narrative_review.resume_issues:
                has_issues = True
                logging.debug(
                    "[repair:%s] narrative-coherence: %d issue(s) — passing ALL to repair",
                    key, len(narrative_review.resume_issues),
                )
                parts.append(
                    "NARRATIVE COHERENCE — Misaligned content (verbatim from document):\n"
                    + "\n".join(
                        f'- "{i.phrase}" — {i.issue}. Suggestion: {i.suggestion}'
                        for i in narrative_review.resume_issues
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
