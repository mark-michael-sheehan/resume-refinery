"""Tests for the IngestAgent — document-to-career-repo extraction."""

import json

import pytest

from resume_refinery.ingest_agent import (
    IngestAgent,
    build_repo_from_parsed,
    parse_ingest_response,
    _extract_json,
    _strip_think_tags,
    INGEST_SYSTEM_PROMPT,
)
from resume_refinery.models import CareerRepository


# ---------------------------------------------------------------------------
# Helper: sample LLM response
# ---------------------------------------------------------------------------

SAMPLE_LLM_JSON = json.dumps({
    "identity": {
        "name": "Jordan Lee",
        "email": "jordan@example.com",
        "phone": "415-555-0100",
        "location": "San Francisco, CA",
        "linkedin": "https://linkedin.com/in/jordanlee",
        "github": "https://github.com/jordanlee",
        "headline": "Senior Backend Engineer with 5+ years in distributed systems",
        "target_roles": ["Staff Engineer", "Principal Engineer"],
    },
    "roles": [
        {
            "company": "DataFlow Inc",
            "title": "Senior Engineer",
            "start_date": "Mar 2021",
            "end_date": "Present",
            "company_context": "Series B data infrastructure startup",
            "team_context": "Platform team of 8",
            "ownership": "Backend data pipeline",
            "accomplishments": "Led backend migration, cut deploy time 60%. Reduced infra costs by $180K/year.",
            "technologies": "Python, Postgres, Kafka, Kubernetes",
            "learnings": "Scaling event-driven systems under pressure",
            "anti_claims": "",
        },
        {
            "company": "TechCorp",
            "title": "Software Engineer",
            "start_date": "Jun 2019",
            "end_date": "Feb 2021",
            "company_context": "",
            "team_context": "",
            "ownership": "",
            "accomplishments": "Built REST APIs serving 10M requests/day",
            "technologies": "Java, Spring Boot",
            "learnings": "",
            "anti_claims": "",
        },
    ],
    "skills": [
        {"name": "Python", "category": "language", "proficiency": "expert", "years": "5", "evidence": "Primary language at DataFlow"},
        {"name": "Kubernetes", "category": "infrastructure", "proficiency": "strong", "years": "3", "evidence": ""},
        {"name": "Leadership", "category": "non_technical", "proficiency": "working", "years": "", "evidence": "Led team of 8"},
    ],
    "stories": [],
    "education": "B.S. Computer Science, UC Berkeley, 2019",
    "certifications": "",
    "domain_knowledge": "Data infrastructure, event-driven architecture",
    "meta": {
        "career_arc": "IC engineer progressing toward staff-level platform roles",
        "differentiators": "Deep cost optimization expertise combined with hands-on distributed systems",
        "themes_to_emphasize": ["cost optimization", "distributed systems"],
        "anti_claims": [],
        "known_gaps": [],
    },
})


# ---------------------------------------------------------------------------
# Unit tests — strip_think_tags
# ---------------------------------------------------------------------------


def test_strip_think_tags():
    assert _strip_think_tags("hello") == "hello"
    assert _strip_think_tags("<think>inner</think>result") == "result"
    assert _strip_think_tags("before<think>x</think>after") == "beforeafter"


# ---------------------------------------------------------------------------
# Unit tests — extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    assert _extract_json('{"key": "val"}') == '{"key": "val"}'


def test_extract_json_with_fences():
    text = '```json\n{"key": "val"}\n```'
    assert _extract_json(text) == '{"key": "val"}'


def test_extract_json_with_bare_fences():
    text = '```\n{"key": "val"}\n```'
    assert _extract_json(text) == '{"key": "val"}'


# ---------------------------------------------------------------------------
# Unit tests — parse_ingest_response
# ---------------------------------------------------------------------------


def test_parse_ingest_response_clean_json():
    result = parse_ingest_response(SAMPLE_LLM_JSON)
    assert isinstance(result, dict)
    assert result["identity"]["name"] == "Jordan Lee"
    assert len(result["roles"]) == 2


def test_parse_ingest_response_with_think_tags():
    raw = f"<think>Let me analyze the documents...</think>{SAMPLE_LLM_JSON}"
    result = parse_ingest_response(raw)
    assert result["identity"]["name"] == "Jordan Lee"


def test_parse_ingest_response_with_fences():
    raw = f"```json\n{SAMPLE_LLM_JSON}\n```"
    result = parse_ingest_response(raw)
    assert result["identity"]["name"] == "Jordan Lee"


def test_parse_ingest_response_empty():
    with pytest.raises(ValueError):
        parse_ingest_response("")


def test_parse_ingest_response_non_object():
    with pytest.raises(ValueError, match="Expected JSON object"):
        parse_ingest_response("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Unit tests — build_repo_from_parsed
# ---------------------------------------------------------------------------


def _make_repo() -> CareerRepository:
    return CareerRepository(repo_id="test-repo", created_at="", updated_at="")


def test_build_repo_identity():
    data = json.loads(SAMPLE_LLM_JSON)
    repo = build_repo_from_parsed(data, _make_repo())
    assert repo.identity.name == "Jordan Lee"
    assert repo.identity.email == "jordan@example.com"
    assert repo.identity.headline == "Senior Backend Engineer with 5+ years in distributed systems"
    assert repo.identity.target_roles == ["Staff Engineer", "Principal Engineer"]


def test_build_repo_roles():
    data = json.loads(SAMPLE_LLM_JSON)
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.roles) == 2
    assert repo.roles[0].company == "DataFlow Inc"
    assert repo.roles[0].title == "Senior Engineer"
    assert repo.roles[0].accomplishments == "Led backend migration, cut deploy time 60%. Reduced infra costs by $180K/year."
    assert repo.roles[1].company == "TechCorp"


def test_build_repo_skills():
    data = json.loads(SAMPLE_LLM_JSON)
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.skills) == 3
    assert repo.skills[0].name == "Python"
    assert repo.skills[0].category == "language"
    assert repo.skills[0].proficiency == "expert"
    assert repo.skills[2].category == "non_technical"


def test_build_repo_education_and_meta():
    data = json.loads(SAMPLE_LLM_JSON)
    repo = build_repo_from_parsed(data, _make_repo())
    assert "UC Berkeley" in repo.education
    assert "Data infrastructure" in repo.domain_knowledge
    assert "cost optimization" in repo.meta.themes_to_emphasize
    assert repo.meta.career_arc.startswith("IC engineer")


def test_build_repo_empty_data():
    repo = build_repo_from_parsed({}, _make_repo())
    assert repo.identity.name == ""
    assert repo.roles == []
    assert repo.skills == []


def test_build_repo_skips_invalid_roles():
    data = {
        "roles": [
            {"company": "Valid", "title": "Eng"},
            {"company": "", "title": ""},  # Should be skipped
            "not a dict",  # Should be skipped
            {"company": "Also Valid", "title": "SRE"},
        ]
    }
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.roles) == 2
    assert repo.roles[0].company == "Valid"
    assert repo.roles[1].company == "Also Valid"


def test_build_repo_skips_invalid_skills():
    data = {
        "skills": [
            {"name": "Python", "category": "language"},
            {"name": ""},  # Should be skipped
            {"name": "Go", "category": "INVALID_CAT", "proficiency": "INVALID_PROF"},
        ]
    }
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.skills) == 2
    assert repo.skills[1].name == "Go"
    assert repo.skills[1].category == "other"  # Defaulted
    assert repo.skills[1].proficiency == "working"  # Defaulted


def test_build_repo_skips_invalid_stories():
    data = {
        "stories": [
            {"title": "Good Story", "situation": "We had a problem"},
            {"title": ""},  # Should be skipped
            "not a dict",  # Should be skipped
        ]
    }
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.stories) == 1
    assert repo.stories[0].title == "Good Story"


# ---------------------------------------------------------------------------
# Integration test with mocked LLM
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


def test_ingest_with_mocked_llm(monkeypatch):
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(SAMPLE_LLM_JSON),
    )

    data = agent.ingest("# Jordan Lee\nSenior Engineer at DataFlow Inc")
    assert data["identity"]["name"] == "Jordan Lee"
    assert len(data["roles"]) == 2


def test_ingest_to_repo_with_mocked_llm(monkeypatch):
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(SAMPLE_LLM_JSON),
    )

    repo = _make_repo()
    result = agent.ingest_to_repo("# Jordan Lee\nSenior Engineer at DataFlow Inc", repo)
    assert result.identity.name == "Jordan Lee"
    assert len(result.roles) == 2
    assert len(result.skills) == 3


def test_ingest_empty_text():
    agent = IngestAgent()
    with pytest.raises(ValueError, match="No document text"):
        agent.ingest("")


def test_ingest_llm_empty_response(monkeypatch):
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(""),
    )

    with pytest.raises(ValueError, match="LLM returned empty content"):
        agent.ingest("some document text")


def test_ingest_with_think_tags_in_response(monkeypatch):
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(f"<think>analyzing...</think>{SAMPLE_LLM_JSON}"),
    )

    data = agent.ingest("# Jordan Lee\nSenior Engineer")
    assert data["identity"]["name"] == "Jordan Lee"


def test_system_prompt_has_json_schema():
    """Verify the system prompt includes all expected top-level keys."""
    for key in ("identity", "roles", "skills", "stories", "education",
                "certifications", "domain_knowledge", "meta"):
        assert f'"{key}"' in INGEST_SYSTEM_PROMPT
