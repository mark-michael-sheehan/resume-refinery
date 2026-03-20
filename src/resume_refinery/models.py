"""Pydantic data models for all inputs, outputs, and session state."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


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
    cover_letter_assessment: str
    resume_assessment: str
    interview_guide_assessment: str
    specific_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class AIDetectionResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    cover_letter_flags: list[str] = Field(default_factory=list)
    resume_flags: list[str] = Field(default_factory=list)
    interview_guide_flags: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class DocumentTruthResult(BaseModel):
    pass_strict: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    evidence_examples: list[str] = Field(default_factory=list)


class TruthfulnessResult(BaseModel):
    all_supported: bool
    cover_letter: DocumentTruthResult
    resume: DocumentTruthResult
    interview_guide: DocumentTruthResult
    suggestions: list[str] = Field(default_factory=list)


class ReviewBundle(BaseModel):
    voice: Optional[VoiceReviewResult] = None
    ai_detection: Optional[AIDetectionResult] = None
    truthfulness: Optional[TruthfulnessResult] = None


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
