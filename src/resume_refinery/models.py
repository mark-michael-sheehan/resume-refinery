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


class WorkExperience(BaseModel):
    title: str
    company: str
    dates: str = Field(description="e.g. 'Jan 2021 – Present'")
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[str] = None
    details: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str
    outcome: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)


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


class VerificationReport(BaseModel):
    reviews: ReviewBundle
    passed_strict_truth: bool = False


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
