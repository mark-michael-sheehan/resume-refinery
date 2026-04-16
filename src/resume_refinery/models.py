"""Pydantic data models for all inputs, outputs, and session state."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, BeforeValidator, Field, model_validator


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a list to list[str], converting any non-string items via str()."""
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, str) else str(item) for item in value]


StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class VoiceProfile(BaseModel):
    """The user's writing voice, loaded directly from a markdown/text file."""

    raw_content: str = Field(description="Full contents of the voice profile file")


class CareerProfile(BaseModel):
    """The user's professional history, loaded from a markdown file."""

    raw_content: str = Field(description="Full contents of the career profile file")

    # Extracted fields — populated by parsers.py when structure is present.
    # Claude uses raw_content directly; these aid session naming only.
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class JobDescription(BaseModel):
    """The target job, loaded from a markdown or plain-text file."""

    raw_content: str = Field(description="Full job description text")

    # Extracted for session naming and display
    title: Optional[str] = None
    company: Optional[str] = None


# ---------------------------------------------------------------------------
# Career repository models
# ---------------------------------------------------------------------------

WizardPhase = Literal[
    "identity",
    "roles",
    "role_deepdive",
    "skills",
    "stories",
    "meta",
    "voice",
    "complete",
]


class RoleEntry(BaseModel):
    """A single work experience role in the career repository."""

    company: str
    title: str
    start_date: str = Field(description="e.g. 'Mar 2021'")
    end_date: str = Field(default="Present", description="e.g. 'Feb 2023' or 'Present'")
    company_context: str = Field(default="", description="What the company does, size, stage")
    team_context: str = Field(default="", description="Team size, reporting structure")
    ownership: str = Field(default="", description="What the user was responsible for")
    accomplishments: str = Field(default="", description="Key accomplishments, narrative form")
    technologies: str = Field(default="", description="Comma-separated or prose")
    learnings: str = Field(default="", description="What the user learned in this role")
    anti_claims: str = Field(default="", description="Things NOT to claim about this role")
    extraction_confidence: Literal["high", "medium", "low"] = "medium"
    confidence_notes: str = Field(default="", description="What the LLM found thin or missing")

    def slug(self) -> str:
        """URL-safe identifier for this role."""
        import re
        raw = f"{self.start_date[:4]}_{self.company}".lower()
        return re.sub(r"[^\w]+", "-", raw).strip("-")[:40]


class SkillEntry(BaseModel):
    """A single skill with proficiency and evidence."""

    name: str
    category: Literal["language", "infrastructure", "tool", "framework", "non_technical", "other"] = "other"
    proficiency: Literal["expert", "strong", "working", "familiar"] = "working"
    years: Optional[str] = None
    evidence: str = Field(default="", description="Concrete evidence of this skill")


class StoryEntry(BaseModel):
    """A behavioral STAR story for interview prep."""

    title: str
    tags: list[str] = Field(default_factory=list)
    situation: str = ""
    task: str = ""
    action: str = ""
    result: str = ""
    what_it_shows: str = Field(default="", description="What this story demonstrates about the user")
    extraction_confidence: Literal["high", "medium", "low"] = "medium"
    confidence_notes: str = Field(default="", description="Which STAR components were inferred vs explicit")


class CareerIdentity(BaseModel):
    """Basic contact and identity info."""

    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    headline: str = Field(default="", description="One-line professional headline")
    target_roles: list[str] = Field(default_factory=list)


class CareerMeta(BaseModel):
    """Strategic metadata about the career."""

    career_arc: str = Field(default="", description="Narrative career trajectory")
    differentiators: str = Field(default="", description="What makes this person unique")
    themes_to_emphasize: list[str] = Field(default_factory=list)
    anti_claims: list[str] = Field(default_factory=list, description="Things NEVER to claim")
    known_gaps: list[str] = Field(default_factory=list)


class VoiceData(BaseModel):
    """Structured voice profile data collected from the career wizard."""

    core_adjectives: list[str] = Field(default_factory=list, description="5-7 words describing communication style")
    style_notes: list[str] = Field(default_factory=list, description="Writing style rules, one per entry")
    preferred_phrases: list[str] = Field(default_factory=list, description="Characteristic phrases the user naturally uses")
    avoid_phrases: list[str] = Field(default_factory=list, description="Cliches or language the user rejects")
    writing_samples: list[str] = Field(default_factory=list, description="Paragraphs in the user's natural voice")

    def has_content(self) -> bool:
        """Return True if any voice field is populated."""
        return bool(
            self.core_adjectives or self.style_notes or self.preferred_phrases
            or self.avoid_phrases or self.writing_samples
        )

    def to_markdown(self, name: str = "") -> str:
        """Render the structured voice data as markdown for LLM consumption."""
        title = f"# Voice Profile — {name}\n" if name else "# Voice Profile\n"
        parts: list[str] = [title]
        if self.core_adjectives:
            parts.append("## Core Adjectives")
            parts.extend(f"- {a}" for a in self.core_adjectives)
            parts.append("")
        if self.style_notes:
            parts.append("## Style Notes")
            parts.extend(f"- {n}" for n in self.style_notes)
            parts.append("")
        if self.preferred_phrases:
            parts.append("## Phrases I Actually Use")
            parts.extend(f'- "{p}"' for p in self.preferred_phrases)
            parts.append("")
        if self.avoid_phrases:
            parts.append("## Phrases to Avoid")
            parts.extend(f'- "{p}"' for p in self.avoid_phrases)
            parts.append("")
        for i, sample in enumerate(self.writing_samples, 1):
            parts.append(f"## Writing Sample {i}")
            parts.append(sample)
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def from_markdown(raw: str) -> "VoiceData":
        """Parse a legacy markdown voice profile into structured VoiceData."""
        if not raw or not raw.strip():
            return VoiceData()

        def _extract_section(text: str, heading: str) -> str:
            pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
            m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            return m.group(1).strip() if m else ""

        def _extract_list(text: str, heading: str) -> list[str]:
            block = _extract_section(text, heading)
            if not block:
                return []
            items: list[str] = []
            for line in block.splitlines():
                line = line.strip()
                line = re.sub(r'^- "?(.*?)"?$', r"\1", line)
                if line:
                    items.append(line)
            return items

        adjectives = _extract_list(raw, "Core Adjectives")
        style_notes = _extract_list(raw, "Style Notes")
        preferred = _extract_list(raw, "Phrases I Actually Use")
        avoid = _extract_list(raw, "Phrases to Avoid")

        samples: list[str] = []
        for i in range(1, 4):
            sample = _extract_section(raw, f"Writing Sample {i}")
            if sample:
                samples.append(sample)
        # Also try bare "Writing Samples" heading
        if not samples:
            bare = _extract_section(raw, "Writing Samples")
            if bare:
                samples = [p.strip() for p in bare.split("\n\n") if p.strip()]

        return VoiceData(
            core_adjectives=adjectives,
            style_notes=style_notes,
            preferred_phrases=preferred,
            avoid_phrases=avoid,
            writing_samples=samples,
        )


class CareerRepository(BaseModel):
    """Complete structured career repository for guided elicitation."""

    repo_id: str = Field(description="URL-safe identifier")
    created_at: str = Field(default="", description="ISO 8601 datetime")
    updated_at: str = Field(default="", description="ISO 8601 datetime")
    current_phase: WizardPhase = "identity"
    deepdive_role_index: int = Field(default=0, description="Which role is being deep-dived")
    needs_consolidation: bool = Field(default=False, description="True after ingestion, before user confirms extracted roles")

    identity: CareerIdentity = Field(default_factory=CareerIdentity)
    roles: list[RoleEntry] = Field(default_factory=list)
    skills: list[SkillEntry] = Field(default_factory=list)
    stories: list[StoryEntry] = Field(default_factory=list)
    education: str = Field(default="", description="Education section, free-form markdown")
    certifications: str = Field(default="", description="Certifications, free-form markdown")
    domain_knowledge: str = Field(default="", description="Industry/domain expertise")
    meta: CareerMeta = Field(default_factory=CareerMeta)
    voice: VoiceData = Field(default_factory=VoiceData, description="Structured voice profile data")

    @model_validator(mode="before")
    @classmethod
    def _migrate_voice_raw(cls, data: dict) -> dict:  # type: ignore[override]
        """Migrate legacy ``voice_raw`` string to structured ``voice`` field."""
        if not isinstance(data, dict):
            return data
        voice_raw = data.pop("voice_raw", None)
        if voice_raw and "voice" not in data:
            data["voice"] = VoiceData.from_markdown(voice_raw)
        return data

    @property
    def voice_raw(self) -> str:
        """Render the structured voice data as markdown (backward compat)."""
        return self.voice.to_markdown(name=self.identity.name)

    def to_career_profile(self) -> "CareerProfile":
        """Flatten the repository into a single CareerProfile for the pipeline."""
        sections: list[str] = []

        # Identity
        ident = self.identity
        if ident.name:
            sections.append(f"# {ident.name}")
        contact_parts = [p for p in [ident.email, ident.phone, ident.location] if p]
        if contact_parts:
            sections.append(" | ".join(contact_parts))
        for link in [ident.linkedin, ident.github]:
            if link:
                sections.append(link)
        if ident.headline:
            sections.append(f"\n## Professional Summary\n{ident.headline}")

        # Experience
        if self.roles:
            sections.append("\n## Work Experience\n")
            for role in self.roles:
                sections.append(f"### {role.title} @ {role.company} ({role.start_date} – {role.end_date})")
                if role.company_context:
                    sections.append(role.company_context)
                if role.team_context:
                    sections.append(role.team_context)
                if role.ownership:
                    sections.append(f"\n**What I Owned:**\n{role.ownership}")
                if role.accomplishments:
                    sections.append(f"\n**What I actually did:**\n{role.accomplishments}")
                if role.technologies:
                    sections.append(f"\n**Technologies:** {role.technologies}")
                if role.learnings:
                    sections.append(f"\n**What I learned:**\n{role.learnings}")
                if role.anti_claims:
                    sections.append(f"\n**Do NOT claim:**\n{role.anti_claims}")
                sections.append("\n---\n")

        # Education & Certifications
        if self.education:
            sections.append(f"\n## Education\n{self.education}")
        if self.certifications:
            sections.append(f"\n## Certifications\n{self.certifications}")

        # Projects / Stories
        if self.stories:
            sections.append("\n## Key Stories\n")
            for story in self.stories:
                sections.append(f"### {story.title}")
                if story.tags:
                    sections.append(f"Tags: {', '.join(story.tags)}")
                if story.situation:
                    sections.append(f"**Situation:** {story.situation}")
                if story.task:
                    sections.append(f"**Task:** {story.task}")
                if story.action:
                    sections.append(f"**Action:** {story.action}")
                if story.result:
                    sections.append(f"**Result:** {story.result}")
                if story.what_it_shows:
                    sections.append(f"**What this shows:** {story.what_it_shows}")
                sections.append("")

        # Skills
        if self.skills:
            sections.append("\n## Skills\n")
            for skill in self.skills:
                line = f"- **{skill.name}** ({skill.proficiency})"
                if skill.years:
                    line += f" — {skill.years}"
                if skill.evidence:
                    line += f" — {skill.evidence}"
                sections.append(line)

        # Domain knowledge
        if self.domain_knowledge:
            sections.append(f"\n## Domain Knowledge\n{self.domain_knowledge}")

        # Meta
        meta = self.meta
        meta_parts: list[str] = []
        if meta.career_arc:
            meta_parts.append(f"**Career Arc:** {meta.career_arc}")
        if meta.differentiators:
            meta_parts.append(f"**What Makes Me Different:** {meta.differentiators}")
        if meta.themes_to_emphasize:
            meta_parts.append("**Themes to Emphasize:**\n" + "\n".join(f"- {t}" for t in meta.themes_to_emphasize))
        if meta.anti_claims:
            meta_parts.append("**Things I Do NOT Want Claimed:**\n" + "\n".join(f"- {c}" for c in meta.anti_claims))
        if meta.known_gaps:
            meta_parts.append("**Gaps I'm Aware Of:**\n" + "\n".join(f"- {g}" for g in meta.known_gaps))
        if meta_parts:
            sections.append("\n## Key Points to Draw From\n" + "\n\n".join(meta_parts))

        raw = "\n".join(sections)
        return CareerProfile(
            raw_content=raw,
            name=ident.name or None,
            email=ident.email or None,
            phone=ident.phone or None,
            location=ident.location or None,
        )


class SupportingEvidence(BaseModel):
    """A career example that supports a narrative pillar, with justification."""

    evidence: str = Field(description="Specific fact or example from the career profile")
    justification: str = Field(default="", description="Why this evidence supports the pillar theme and should be included")


def _coerce_evidence_list(value: object) -> list[SupportingEvidence]:
    """Coerce plain strings into SupportingEvidence for backward compatibility."""
    if not isinstance(value, list):
        return []
    result: list[SupportingEvidence] = []
    for item in value:
        if isinstance(item, SupportingEvidence):
            result.append(item)
        elif isinstance(item, dict):
            result.append(SupportingEvidence(**item))
        elif isinstance(item, str):
            result.append(SupportingEvidence(evidence=item))
        else:
            result.append(SupportingEvidence(evidence=str(item)))
    return result


EvidenceList = Annotated[list[SupportingEvidence], BeforeValidator(_coerce_evidence_list)]


class NarrativePillar(BaseModel):
    """A supporting theme in the candidacy narrative."""

    theme: str = Field(description="Short theme label")
    argument: str = Field(description="How this theme supports the thesis")
    career_evidence: EvidenceList = Field(default_factory=list, description="Supporting examples from the career profile, ordered by importance (strongest first)")


class CandidacyNarrative(BaseModel):
    """Strategic narrative framing why the candidate is a strong fit for the target role."""

    thesis: str = Field(default="", description="Core 1-2 sentence argument for candidacy")
    pillars: list[NarrativePillar] = Field(default_factory=list, description="3-5 supporting themes with evidence")
    gap_framing: list[str] = Field(default_factory=list, description="Honest framing for gaps between candidate and requirements")
    raw_narrative: str = Field(default="", description="Full generated narrative text for reference")


class VoiceStyleGuide(BaseModel):
    core_adjectives: list[str] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)
    preferred_phrases: list[str] = Field(default_factory=list)
    phrases_to_avoid: list[str] = Field(default_factory=list)
    writing_samples: list[str] = Field(default_factory=list)


class DraftingContext(BaseModel):
    narrative: CandidacyNarrative
    voice_style_guide: VoiceStyleGuide


# ---------------------------------------------------------------------------
# Narrative coverage models
# ---------------------------------------------------------------------------


class CoverageGap(BaseModel):
    """A narrative pillar that is backed by career evidence but not reflected in the resume."""

    pillar_theme: str = Field(description="Theme label of the uncovered pillar")
    career_evidence: list[str] = Field(default_factory=list, description="Evidence from the career profile supporting this pillar")
    suggested_content: str = Field(default="", description="Concrete bullet or sentence to add to the resume")
    anchor_section: str = Field(default="", description="Resume section where the addition fits best")
    confidence: Literal["high", "medium", "low"] = Field(default="medium", description="Confidence that adding this gap would improve the resume")


class NarrativeCoverageResult(BaseModel):
    """Result of comparing narrative pillars against resume content."""

    gaps: list[CoverageGap] = Field(default_factory=list, description="Narrative pillars not covered in the resume")
    coverage_summary: str = Field(default="", description="Brief summary of overall narrative coverage")
    pillars_covered: int = Field(default=0, description="Number of pillars adequately represented in the resume")
    pillars_total: int = Field(default=0, description="Total number of narrative pillars")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


DocumentKey = Literal["resume"]


class DocumentSet(BaseModel):
    """Markdown source for the generated resume."""

    resume: Optional[str] = None

    def get(self, key: DocumentKey) -> Optional[str]:
        return getattr(self, key, None)

    def set(self, key: DocumentKey, value: str) -> None:
        setattr(self, key, value)

    def all_present(self) -> bool:
        return self.resume is not None


# ---------------------------------------------------------------------------
# Review models
# ---------------------------------------------------------------------------


class VoiceReviewResult(BaseModel):
    overall_match: Literal["strong", "moderate", "weak"]
    resume_match: Literal["strong", "moderate", "weak"] = "moderate"
    resume_assessment: str = ""
    specific_issues: StrList = Field(default_factory=list)
    resume_issues: StrList = Field(default_factory=list)


class AIDetectionResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    resume_flags: StrList = Field(default_factory=list)


class DocumentTruthResult(BaseModel):
    pass_strict: bool
    unsupported_claims: StrList = Field(default_factory=list)
    evidence_examples: StrList = Field(default_factory=list)


class TruthfulnessResult(BaseModel):
    all_supported: bool
    resume: DocumentTruthResult


class RelevancePruningIssue(BaseModel):
    """A single content item flagged for potential removal."""

    document: Literal["resume"]
    phrase: str = Field(description="Verbatim quote from the document to consider removing")
    reason: str = Field(description="Why this content does not add to the story")
    category: Literal["redundant", "irrelevant", "filler", "low_impact", "space_waste"] = "filler"
    severity: Literal["high", "medium", "low"] = "medium"


class RelevancePruningResult(BaseModel):
    """Result of relevance pruning review on the resume."""

    overall_density: Literal["lean", "balanced", "bloated"] = "balanced"
    resume_issues: list[RelevancePruningIssue] = Field(default_factory=list)


class HiringManagerImprovementItem(BaseModel):
    """A single improvement suggestion from the hiring manager review."""

    area: str = Field(default="resume", description="Which document or section the suggestion targets")
    suggestion: str = Field(description="Specific actionable improvement")
    impact: Literal["high", "medium", "low"] = "medium"


class HiringManagerIssue(BaseModel):
    """A single targeted finding from the hiring manager review.

    Each issue quotes a verbatim phrase from the document so the repair
    agent can produce a surgical find/replace edit."""

    document: Literal["resume"]
    phrase: str = Field(description="Verbatim quote from the document to improve")
    issue: str = Field(description="What is weak from a hiring-manager perspective")
    suggestion: str = Field(description="How to improve it")
    impact: Literal["high", "medium", "low"] = "medium"


class HiringManagerReview(BaseModel):
    """Simulated hiring-manager assessment of the resume."""

    advance_likelihood: int = Field(
        ge=0, le=100,
        description="Percentage likelihood of advancing the candidate to the next stage",
    )
    summary: str = Field(default="", description="Overall hiring-manager impression")
    strengths: StrList = Field(default_factory=list)
    concerns: StrList = Field(default_factory=list)
    improvements: list[HiringManagerImprovementItem] = Field(default_factory=list)
    resume_issues: list[HiringManagerIssue] = Field(default_factory=list)


class ATSKeywordIssue(BaseModel):
    """A single keyword alignment finding."""

    keyword: str = Field(description="The JD keyword/phrase that is missing or over-used")
    issue_type: Literal["missing", "stuffing"] = "missing"
    section: str = Field(default="", description="Resume section where the issue was found (or should appear)")
    suggestion: str = Field(default="", description="How to address the issue")
    priority: Literal["high", "medium", "low"] = "medium"


class ATSKeywordResult(BaseModel):
    """Result of ATS keyword alignment review on the resume."""

    alignment_score: Literal["strong", "moderate", "weak"] = "moderate"
    missing_keywords: list[ATSKeywordIssue] = Field(default_factory=list)
    stuffing_keywords: list[ATSKeywordIssue] = Field(default_factory=list)


class GrammarIssue(BaseModel):
    """A single grammar or mechanics finding."""

    document: Literal["resume"]
    phrase: str = Field(description="Verbatim quote containing the error")
    issue: str = Field(description="Description of the grammatical or mechanical problem")
    suggestion: str = Field(default="", description="Corrected version")
    category: Literal["grammar", "tense", "punctuation", "capitalization", "formatting"] = "grammar"
    severity: Literal["high", "medium", "low"] = "medium"


class GrammarResult(BaseModel):
    """Result of grammar & mechanics review."""

    clean: bool = True
    resume_issues: list[GrammarIssue] = Field(default_factory=list)


class NarrativeCoherenceIssue(BaseModel):
    """A single resume item that does not connect to the candidacy narrative."""

    phrase: str = Field(description="Verbatim quote from the resume")
    issue: str = Field(description="Why this content does not support any narrative pillar")
    suggestion: str = Field(default="", description="How to realign or remove it")
    severity: Literal["high", "medium", "low"] = "medium"


class NarrativeCoherenceResult(BaseModel):
    """Result of narrative coherence review — checks that every resume point supports the candidacy narrative."""

    alignment: Literal["strong", "moderate", "weak"] = "moderate"
    resume_issues: list[NarrativeCoherenceIssue] = Field(default_factory=list)


class ReviewBundle(BaseModel):
    voice: Optional[VoiceReviewResult] = None
    ai_detection: Optional[AIDetectionResult] = None
    truthfulness: Optional[TruthfulnessResult] = None
    hiring_manager: Optional[HiringManagerReview] = None
    relevance_pruning: Optional[RelevancePruningResult] = None
    ats_keyword: Optional[ATSKeywordResult] = None
    grammar: Optional[GrammarResult] = None
    narrative_coherence: Optional[NarrativeCoherenceResult] = None


ReviewerPriority = Literal[
    "truthfulness",
    "ats",
    "grammar",
    "voice",
    "ai",
    "hm",
    "narrative",
    "pruning",
]

# Higher value = higher priority.  Truthfulness edits are never overwritten
# by lower-priority style repairs.
REVIEWER_PRIORITY_RANK: dict[str, int] = {
    "truthfulness": 80,
    "ats": 60,
    "grammar": 50,
    "voice": 40,
    "narrative": 35,
    "ai": 30,
    "hm": 20,
    "pruning": 10,
}


class EditRegion(BaseModel):
    """A character span in a document that was modified by a repair edit."""

    start: int = Field(description="Start character offset (inclusive)")
    end: int = Field(description="End character offset (exclusive)")
    reviewer: ReviewerPriority = Field(description="Which reviewer triggered this edit")
    pass_num: int = Field(description="0-based repair pass number")

    @property
    def priority(self) -> int:
        return REVIEWER_PRIORITY_RANK.get(self.reviewer, 0)

    def overlaps(self, start: int, end: int) -> bool:
        """Return True if [start, end) overlaps this region."""
        return self.start < end and start < self.end


class DocumentEditHistory(BaseModel):
    """Accumulates edit regions per document across repair passes."""

    regions: list[EditRegion] = Field(default_factory=list)

    def add_region(self, start: int, end: int, reviewer: ReviewerPriority, pass_num: int) -> None:
        self.regions.append(EditRegion(start=start, end=end, reviewer=reviewer, pass_num=pass_num))

    def is_protected(self, phrase: str, document: str, by_reviewer: ReviewerPriority) -> bool:
        """Return True if *phrase* falls inside a region edited by an equal-or-higher-priority reviewer.

        This implements the merge-conflict rule: higher-priority edits
        protect their region from being re-flagged by lower-priority reviewers.
        """
        incoming_rank = REVIEWER_PRIORITY_RANK.get(by_reviewer, 0)
        idx = document.find(phrase)
        if idx == -1:
            return False
        phrase_end = idx + len(phrase)
        for region in self.regions:
            if region.overlaps(idx, phrase_end) and region.priority >= incoming_rank:
                return True
        return False


class RepairEdit(BaseModel):
    find: str
    replace: str
    reason: str = ""
    reviewer: str = Field(default="", description="Which reviewer triggered this edit")
    insert_after: bool = Field(default=False, description="When True, 'find' is an anchor; 'replace' is inserted after the anchor without removing it")


class RepairPassResult(BaseModel):
    """Edits applied during a single repair pass, keyed by document."""
    edits: dict[str, list[RepairEdit]] = Field(default_factory=dict)
    # Edit regions produced by apply_edits, keyed by document.
    edit_regions: dict[str, list[EditRegion]] = Field(default_factory=dict)
    # Per-reviewer false-positive acceptances — phrases the repairer determined
    # are reviewer false positives and should be suppressed in future passes.
    accepted_claims: StrList = Field(default_factory=list)
    accepted_ai_phrases: StrList = Field(default_factory=list)
    accepted_voice_issues: StrList = Field(default_factory=list)
    accepted_hm_issues: StrList = Field(default_factory=list)
    accepted_pruning_issues: StrList = Field(default_factory=list)
    accepted_ats_issues: StrList = Field(default_factory=list)
    accepted_grammar_issues: StrList = Field(default_factory=list)
    accepted_narrative_issues: StrList = Field(default_factory=list)
    # Edits that failed Phase 1 locate (could not find the ``find`` text in
    # the document).  Keyed by document, each value is a list of EditOp dicts.
    failed_edits: dict[str, list[dict]] = Field(default_factory=dict)


class ExemptedPhrases(BaseModel):
    """Cumulative set of phrases/claims/issues accepted as false positives across all repair passes."""
    claims: StrList = Field(
        default_factory=list,
        description="Truthfulness claims accepted as already supported by career evidence",
    )
    ai_phrases: StrList = Field(
        default_factory=list,
        description="AI-detection flags accepted as natural human-written language",
    )
    voice_issues: StrList = Field(
        default_factory=list,
        description="Voice-match issues accepted as reviewer false positives",
    )
    hm_issues: StrList = Field(
        default_factory=list,
        description="Hiring-manager issues accepted as reviewer false positives",
    )
    pruning_issues: StrList = Field(
        default_factory=list,
        description="Relevance-pruning issues accepted as reviewer false positives",
    )
    ats_issues: StrList = Field(
        default_factory=list,
        description="ATS-keyword issues accepted as reviewer false positives",
    )
    grammar_issues: StrList = Field(
        default_factory=list,
        description="Grammar/mechanics issues accepted as reviewer false positives",
    )
    narrative_issues: StrList = Field(
        default_factory=list,
        description="Narrative-coherence issues accepted as reviewer false positives",
    )


class OrchestrationResult(BaseModel):
    session: Session
    documents: DocumentSet
    reviews: ReviewBundle = Field(default_factory=ReviewBundle)
    repair_passes: list[RepairPassResult] = Field(default_factory=list)
    narrative: Optional[CandidacyNarrative] = None
    voice_style_guide: Optional[VoiceStyleGuide] = None
    coverage_result: Optional[NarrativeCoverageResult] = None
    exported_paths: dict[str, str] = Field(default_factory=dict)
    strict_truth_failed: bool = False


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class VersionInfo(BaseModel):
    version: int
    created_at: str = Field(description="ISO 8601 datetime string")
    feedback: Optional[str] = None
    docs_regenerated: list[DocumentKey] = Field(default_factory=list)
    has_reviews: bool = False


ALL_DOC_KEYS: list[DocumentKey] = ["resume"]


class Session(BaseModel):
    session_id: str
    job_description: JobDescription
    created_at: str
    current_version: int = 1
    versions: list[VersionInfo] = Field(default_factory=list)
    selected_docs: list[DocumentKey] = Field(
        default_factory=lambda: list(ALL_DOC_KEYS),
        description="Which documents the user chose to generate for this session",
    )
