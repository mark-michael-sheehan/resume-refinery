"""Pydantic data models for all inputs, outputs, and session state."""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, BeforeValidator, Field


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
    voice_raw: str = Field(default="", description="Voice profile content, structured markdown")

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


class JobRequirement(BaseModel):
    requirement: str
    category: Literal["skill", "experience", "leadership", "domain", "other"] = "other"
    source_excerpt: Optional[str] = None


class EvidenceItem(BaseModel):
    requirement: str
    evidence: str
    source_excerpt: str
    relevance_score: int = Field(default=3, ge=1, le=5)


class EvidencePack(BaseModel):
    job_requirements: list[JobRequirement] = Field(default_factory=list)
    matched_evidence: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    source_summary: list[str] = Field(default_factory=list)


class VoiceStyleGuide(BaseModel):
    core_adjectives: list[str] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)
    preferred_phrases: list[str] = Field(default_factory=list)
    phrases_to_avoid: list[str] = Field(default_factory=list)
    writing_samples: list[str] = Field(default_factory=list)


class DraftingContext(BaseModel):
    evidence_pack: EvidencePack
    voice_style_guide: VoiceStyleGuide


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


DocumentKey = Literal["cover_letter", "resume", "interview_guide"]


class DocumentSet(BaseModel):
    """Markdown source for each generated document."""

    cover_letter: Optional[str] = None
    resume: Optional[str] = None
    interview_guide: Optional[str] = None

    def get(self, key: DocumentKey) -> Optional[str]:
        return getattr(self, key, None)

    def set(self, key: DocumentKey, value: str) -> None:
        setattr(self, key, value)

    def all_present(self) -> bool:
        return all(
            v is not None
            for v in (self.cover_letter, self.resume, self.interview_guide)
        )


# ---------------------------------------------------------------------------
# Review models
# ---------------------------------------------------------------------------


class VoiceReviewResult(BaseModel):
    overall_match: Literal["strong", "moderate", "weak"]
    cover_letter_match: Literal["strong", "moderate", "weak"] = "moderate"
    resume_match: Literal["strong", "moderate", "weak"] = "moderate"
    interview_guide_match: Literal["strong", "moderate", "weak"] = "moderate"
    cover_letter_assessment: str
    resume_assessment: str
    interview_guide_assessment: str
    specific_issues: StrList = Field(default_factory=list)
    # Per-document issues for targeted repair
    cover_letter_issues: StrList = Field(default_factory=list)
    resume_issues: StrList = Field(default_factory=list)
    interview_guide_issues: StrList = Field(default_factory=list)


class AIDetectionResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    cover_letter_flags: StrList = Field(default_factory=list)
    resume_flags: StrList = Field(default_factory=list)
    interview_guide_flags: StrList = Field(default_factory=list)


class DocumentTruthResult(BaseModel):
    pass_strict: bool
    unsupported_claims: StrList = Field(default_factory=list)
    evidence_examples: StrList = Field(default_factory=list)


class TruthfulnessResult(BaseModel):
    all_supported: bool
    cover_letter: DocumentTruthResult
    resume: DocumentTruthResult
    interview_guide: DocumentTruthResult


class ReviewBundle(BaseModel):
    voice: Optional[VoiceReviewResult] = None
    ai_detection: Optional[AIDetectionResult] = None
    truthfulness: Optional[TruthfulnessResult] = None


class RepairEdit(BaseModel):
    find: str
    replace: str
    reason: str = ""


class RepairPassResult(BaseModel):
    """Edits applied during a single repair pass, keyed by document."""
    edits: dict[str, list[RepairEdit]] = Field(default_factory=dict)
    # Per-reviewer false-positive acceptances — phrases the repairer determined
    # are reviewer false positives and should be suppressed in future passes.
    accepted_claims: StrList = Field(default_factory=list)
    accepted_ai_phrases: StrList = Field(default_factory=list)
    accepted_voice_issues: StrList = Field(default_factory=list)


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


class OrchestrationResult(BaseModel):
    session: Session
    documents: DocumentSet
    reviews: ReviewBundle = Field(default_factory=ReviewBundle)
    repair_passes: list[RepairPassResult] = Field(default_factory=list)
    evidence_pack: Optional[EvidencePack] = None
    voice_style_guide: Optional[VoiceStyleGuide] = None
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


class Session(BaseModel):
    session_id: str
    job_description: JobDescription
    created_at: str
    current_version: int = 1
    versions: list[VersionInfo] = Field(default_factory=list)
