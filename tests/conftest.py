"""Shared pytest fixtures."""

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    CareerProfile,
    DocumentSet,
    JobDescription,
    ReviewBundle,
    Session,
    VersionInfo,
    VoiceProfile,
    VoiceReviewResult,
)


@pytest.fixture
def voice_profile() -> VoiceProfile:
    return VoiceProfile(
        raw_content=(
            "# Voice Profile\n\n"
            "## Adjectives\n- Direct, precise, analytical\n- Warm but not effusive\n\n"
            "## Style Notes\n- Short declarative sentences\n- Outcome before method\n"
        )
    )


@pytest.fixture
def career_profile() -> CareerProfile:
    return CareerProfile(
        raw_content=(
            "# Jordan Lee\njordan@example.com | 415-555-0100 | San Francisco, CA\n\n"
            "## Work Experience\n### Senior Engineer @ DataFlow Inc (2021–Present)\n"
            "- Led backend migration, cut deploy time 60%\n- Reduced infra costs by $180K/year\n\n"
            "## Education\n### B.S. Computer Science, UC Berkeley, 2019\n\n"
            "## Key Points\n- Strong distributed systems background\n- Loves mentoring\n"
        ),
        name="Jordan Lee",
        email="jordan@example.com",
        phone="415-555-0100",
        location="San Francisco, CA",
    )


@pytest.fixture
def job_description() -> JobDescription:
    return JobDescription(
        raw_content=(
            "# Staff Engineer, Platform\nCompany: Acme Cloud\n\n"
            "We need a Staff Engineer to own our data ingestion platform.\n"
            "Required: Python, distributed systems, technical leadership.\n"
        ),
        title="Staff Engineer, Platform",
        company="Acme Cloud",
    )


@pytest.fixture
def document_set() -> DocumentSet:
    return DocumentSet(
        cover_letter="Dear Hiring Manager,\n\nI've spent five years building...",
        resume="# Jordan Lee\n\njordan@example.com\n\n## Experience\n\n### Senior Engineer",
        interview_guide="## Interview Guide\n\n### Key Talking Points\n\n- Distributed systems",
    )


@pytest.fixture
def voice_review() -> VoiceReviewResult:
    return VoiceReviewResult(
        overall_match="moderate",
        cover_letter_assessment="Mostly on-voice but opener feels generic.",
        resume_assessment="Quantified well; matches direct tone.",
        interview_guide_assessment="Slightly formal compared to profile.",
        specific_issues=["'results-driven' opener feels off-voice"],
        suggestions=["Replace opener with a concrete hook from their experience"],
    )


@pytest.fixture
def ai_detection() -> AIDetectionResult:
    return AIDetectionResult(
        risk_level="medium",
        cover_letter_flags=["'passionate about innovation'"],
        resume_flags=[],
        interview_guide_flags=["'demonstrated track record'"],
        suggestions=["Replace hollow phrases with specific examples"],
    )


@pytest.fixture
def review_bundle(voice_review, ai_detection) -> ReviewBundle:
    return ReviewBundle(voice=voice_review, ai_detection=ai_detection)


@pytest.fixture
def sample_session(job_description) -> Session:
    return Session(
        session_id="acme-cloud_staff-engineer_2026-03-20",
        job_description=job_description,
        created_at="2026-03-20T10:00:00+00:00",
        current_version=1,
        versions=[
            VersionInfo(
                version=1,
                created_at="2026-03-20T10:00:00+00:00",
                docs_regenerated=["cover_letter", "resume", "interview_guide"],
                has_reviews=False,
            )
        ],
    )
