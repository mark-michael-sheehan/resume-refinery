"""Tests for the IngestAgent — document-to-career-repo extraction."""

import json

import pytest

from resume_refinery.ingest_agent import (
    IngestAgent,
    build_repo_from_parsed,
    consolidate_repo,
    parse_ingest_response,
    _coerce_str,
    _consolidation_call,
    _extract_json,
    _has_duplicate_skills,
    _log_token_usage,
    _normalize_skill_name,
    _repo_to_consolidation_json,
    _strip_think_tags,
    INGEST_SYSTEM_PROMPT,
    STORY_COMPOSITION_PROMPT,
    CONSOLIDATION_PASS1_PROMPT,
    CONSOLIDATION_PASS2_PROMPT,
)
from resume_refinery.models import CareerRepository, RoleEntry, SkillEntry


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
            "extraction_confidence": "high",
            "confidence_notes": "",
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
            "extraction_confidence": "low",
            "confidence_notes": "no company context, team context, or ownership found",
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
# Unit tests — _coerce_str
# ---------------------------------------------------------------------------


def test_coerce_str_plain_string():
    assert _coerce_str("hello") == "hello"


def test_coerce_str_list_of_strings():
    result = _coerce_str(["Led migration to AWS", "Reduced costs by 30%"])
    assert result == "Led migration to AWS\nReduced costs by 30%"


def test_coerce_str_empty_list():
    assert _coerce_str([]) == ""


def test_coerce_str_none():
    assert _coerce_str(None) == ""


def test_coerce_str_empty_string():
    assert _coerce_str("") == ""


def test_coerce_str_list_with_empty_items():
    result = _coerce_str(["a", "", "b", None])
    assert result == "a\nb"


def test_build_repo_handles_list_accomplishments():
    """Accomplishments returned as a list should be joined, not repr'd."""
    data = {
        "roles": [{
            "company": "Acme",
            "title": "Engineer",
            "accomplishments": ["Led migration to AWS", "Reduced costs by 30%"],
        }],
    }
    repo = CareerRepository(repo_id="test-list", created_at="", updated_at="")
    build_repo_from_parsed(data, repo)
    assert repo.roles[0].accomplishments == "Led migration to AWS\nReduced costs by 30%"
    assert "[" not in repo.roles[0].accomplishments


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
    assert repo.roles[0].extraction_confidence == "high"
    assert repo.roles[1].company == "TechCorp"
    assert repo.roles[1].extraction_confidence == "low"
    assert "no company context" in repo.roles[1].confidence_notes


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
            {"title": "Good Story", "situation": "We had a problem",
             "extraction_confidence": "high", "confidence_notes": "solid"},
            {"title": ""},  # Should be skipped
            "not a dict",  # Should be skipped
        ]
    }
    repo = build_repo_from_parsed(data, _make_repo())
    assert len(repo.stories) == 1
    assert repo.stories[0].title == "Good Story"
    assert repo.stories[0].extraction_confidence == "high"


# ---------------------------------------------------------------------------
# Integration test with mocked LLM
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str, prompt_eval_count: int = 500,
                 eval_count: int = 1000, done_reason: str = "stop"):
        self.message = _FakeMessage(content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done_reason = done_reason


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


def test_system_prompt_has_field_guidance():
    """Verify the enriched prompt includes field-level guidance."""
    assert "extraction_confidence" in INGEST_SYSTEM_PROMPT
    assert "confidence_notes" in INGEST_SYSTEM_PROMPT
    assert "THE MOST CRITICAL FIELD" in INGEST_SYSTEM_PROMPT
    assert "company_context" in INGEST_SYSTEM_PROMPT
    assert "team_context" in INGEST_SYSTEM_PROMPT


def test_story_composition_prompt_exists():
    """Verify the story composition prompt is defined with STAR guidance."""
    assert "situation" in STORY_COMPOSITION_PROMPT.lower()
    assert "task" in STORY_COMPOSITION_PROMPT.lower()
    assert "action" in STORY_COMPOSITION_PROMPT.lower()
    assert "result" in STORY_COMPOSITION_PROMPT.lower()
    assert "extraction_confidence" in STORY_COMPOSITION_PROMPT


# ---------------------------------------------------------------------------
# Unit tests — consolidate_repo (LLM-based, mocked)
# ---------------------------------------------------------------------------

# Sample consolidated output — pass 1 (identity + roles)
SAMPLE_CONSOLIDATED_PASS1_JSON = json.dumps({
    "identity": {
        "name": "Jordan Lee",
        "email": "jordan@example.com",
        "phone": "",
        "location": "San Francisco, CA",
        "linkedin": "",
        "github": "",
        "headline": "Senior Backend Engineer",
        "target_roles": [],
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
            "technologies": "Python, Kafka, Kubernetes",
            "learnings": "Scaling event-driven systems under pressure",
            "anti_claims": "",
            "extraction_confidence": "medium",
            "confidence_notes": "merged from 2 documents",
        },
    ],
})

# Sample consolidated output — pass 2 (skills + education + meta)
SAMPLE_CONSOLIDATED_PASS2_JSON = json.dumps({
    "skills": [
        {"name": "Python", "category": "language", "proficiency": "expert",
         "years": "5", "evidence": "Primary language at DataFlow"},
    ],
    "education": "B.S. Computer Science, UC Berkeley, 2019",
    "certifications": "",
    "domain_knowledge": "Data infrastructure",
    "meta": {
        "career_arc": "IC engineer progressing toward staff-level roles",
        "differentiators": "Cost optimization + distributed systems",
        "themes_to_emphasize": ["cost optimization"],
        "anti_claims": [],
        "known_gaps": [],
    },
})


class _FakeClient:
    """Fake ollama.Client for consolidation tests."""

    def __init__(self, responses: list[str] | str):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = responses
        self.call_count = 0

    def chat(self, **kw):
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return _FakeResponse(self._responses[idx])


def test_consolidate_merges_via_llm():
    """consolidate_repo should call the LLM twice and rebuild the repo from both passes."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(
            company="DataFlow Inc", title="Senior Engineer",
            start_date="Mar 2021", end_date="Present",
            accomplishments="Led backend migration.",
            technologies="Python, Kafka",
        ),
        RoleEntry(
            company="DataFlow Inc", title="Senior Engineer",
            start_date="Jan 2022", end_date="Present",
            accomplishments="Reduced infra costs by $180K/year.",
            technologies="Python, Kubernetes",
        ),
    ]
    fake_client = _FakeClient([
        SAMPLE_CONSOLIDATED_PASS1_JSON,
        SAMPLE_CONSOLIDATED_PASS2_JSON,
    ])
    result = consolidate_repo(repo, client=fake_client)
    assert fake_client.call_count == 2
    # Pass 1 results
    assert len(result.roles) == 1
    assert "Led backend migration" in result.roles[0].accomplishments
    assert "$180K" in result.roles[0].accomplishments
    assert "Kafka" in result.roles[0].technologies
    assert "Kubernetes" in result.roles[0].technologies
    # Pass 2 results
    assert len(result.skills) == 1
    assert result.skills[0].name == "Python"
    assert "UC Berkeley" in result.education


def test_consolidate_skips_when_nothing_to_merge():
    """consolidate_repo should not call LLM when only 1 role and 1 skill."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="DataFlow Inc", title="Senior Engineer",
                  start_date="Mar 2021", end_date="Present"),
    ]
    repo.skills = [SkillEntry(name="Python", category="language")]
    fake_client = _FakeClient("should not be called")
    result = consolidate_repo(repo, client=fake_client)
    assert fake_client.call_count == 0
    assert len(result.roles) == 1


def test_consolidate_empty_repo():
    """Consolidation on an empty repo should be a no-op."""
    repo = _make_repo()
    fake_client = _FakeClient("should not be called")
    result = consolidate_repo(repo, client=fake_client)
    assert fake_client.call_count == 0
    assert result.roles == []
    assert result.skills == []


def test_consolidate_falls_back_on_llm_failure():
    """consolidate_repo should return original data if both LLM passes fail."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    repo.skills = [SkillEntry(name="Python", category="language")]

    class _FailClient:
        call_count = 0
        def chat(self, **kw):
            self.call_count += 1
            raise RuntimeError("LLM down")

    fail_client = _FailClient()
    result = consolidate_repo(repo, client=fail_client)
    # Both passes called, both fell back to originals
    assert fail_client.call_count == 2
    assert len(result.roles) == 2
    assert result.roles[0].company == "A"
    assert len(result.skills) == 1


def test_consolidate_falls_back_on_empty_response():
    """consolidate_repo should return original data if LLM returns empty."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    fake_client = _FakeClient(["", ""])
    result = consolidate_repo(repo, client=fake_client)
    assert len(result.roles) == 2


def test_consolidate_pass1_fails_pass2_succeeds():
    """If pass 1 fails, roles fall back to original; pass 2 skills still consolidated."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    repo.skills = [
        SkillEntry(name="Python", category="language", proficiency="expert"),
        SkillEntry(name="python", category="other", proficiency="working"),
    ]
    # Pass 1 returns empty (fails), pass 2 returns consolidated skills
    fake_client = _FakeClient(["", SAMPLE_CONSOLIDATED_PASS2_JSON])
    result = consolidate_repo(repo, client=fake_client)
    # Roles fell back to original (2 separate roles)
    assert len(result.roles) == 2
    # Skills came from pass 2 (consolidated)
    assert len(result.skills) == 1
    assert result.skills[0].name == "Python"


def test_consolidate_preserves_repo_metadata():
    """consolidate_repo should carry over repo_id, created_at, voice_raw, etc."""
    repo = _make_repo()
    repo.voice_raw = "my voice profile"
    repo.current_phase = "role_deepdive"
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    fake_client = _FakeClient([
        SAMPLE_CONSOLIDATED_PASS1_JSON,
        SAMPLE_CONSOLIDATED_PASS2_JSON,
    ])
    result = consolidate_repo(repo, client=fake_client)
    assert result.repo_id == "test-repo"
    assert result.voice_raw == "my voice profile"
    assert result.current_phase == "role_deepdive"


def test_repo_to_consolidation_json():
    """_repo_to_consolidation_json should produce valid JSON with all key sections."""
    repo = _make_repo()
    repo.roles = [RoleEntry(company="Test", title="Eng", start_date="2020", end_date="2021")]
    repo.skills = [SkillEntry(name="Python")]
    result = json.loads(_repo_to_consolidation_json(repo))
    assert "identity" in result
    assert "roles" in result
    assert "skills" in result
    assert len(result["roles"]) == 1
    assert result["roles"][0]["company"] == "Test"


def test_repo_to_consolidation_json_filtered_keys():
    """_repo_to_consolidation_json with keys param should only include those keys."""
    repo = _make_repo()
    repo.roles = [RoleEntry(company="Test", title="Eng", start_date="2020", end_date="2021")]
    repo.skills = [SkillEntry(name="Python")]
    result = json.loads(_repo_to_consolidation_json(repo, keys=["identity", "roles"]))
    assert "identity" in result
    assert "roles" in result
    assert "skills" not in result
    assert "meta" not in result


# ---------------------------------------------------------------------------
# Unit tests — _normalize_skill_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("Python", "python"),
    ("React.js", "reactjs"),
    ("React JS", "reactjs"),
    ("ReactJS", "reactjs"),
    ("Vue.js", "vuejs"),
    ("  Kubernetes  ", "kubernetes"),
    ("CI/CD", "ci/cd"),
    ("Node.js", "nodejs"),
])
def test_normalize_skill_name(raw, expected):
    assert _normalize_skill_name(raw) == expected


# ---------------------------------------------------------------------------
# Unit tests — _has_duplicate_skills
# ---------------------------------------------------------------------------


def test_has_duplicate_skills_exact_match():
    """Exact normalised-name match should be detected."""
    skills = [
        SkillEntry(name="Python"),
        SkillEntry(name="Go"),
        SkillEntry(name="python"),
    ]
    assert _has_duplicate_skills(skills) is True


def test_has_duplicate_skills_no_dupes():
    """Distinct skills should not trigger."""
    skills = [
        SkillEntry(name="Python"),
        SkillEntry(name="Go"),
        SkillEntry(name="Kubernetes"),
    ]
    assert _has_duplicate_skills(skills) is False


def test_has_duplicate_skills_empty():
    assert _has_duplicate_skills([]) is False


def test_has_duplicate_skills_fuzzy_match():
    """Fuzzy near-miss (e.g. 'React.js' vs 'ReactJS') should be caught."""
    skills = [
        SkillEntry(name="React.js"),
        SkillEntry(name="ReactJS"),
    ]
    # These normalise to the same string, so caught by exact match
    assert _has_duplicate_skills(skills) is True


def test_has_duplicate_skills_fuzzy_close_but_different():
    """Genuinely different skills with similar names should not match."""
    skills = [
        SkillEntry(name="Java"),
        SkillEntry(name="JavaScript"),
    ]
    assert _has_duplicate_skills(skills) is False


def test_has_duplicate_skills_fuzzy_threshold():
    """Names that are very close but not identical should trigger above threshold."""
    skills = [
        SkillEntry(name="Machine Learning"),
        SkillEntry(name="Machine  Learning"),  # extra space
    ]
    # Normalisation collapses spaces, so these become identical
    assert _has_duplicate_skills(skills) is True


# ---------------------------------------------------------------------------
# Integration test — consolidate_repo retries on duplicate skills
# ---------------------------------------------------------------------------


def test_consolidate_retries_when_dupes_remain():
    """If pass 2 returns duplicate skills, consolidate_repo retries the LLM."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    repo.skills = [
        SkillEntry(name="Python", category="language", proficiency="expert"),
        SkillEntry(name="python", category="other", proficiency="working"),
    ]

    # Pass 2 returns duplicates the first time, clean the second time
    pass2_with_dupes = json.dumps({
        "skills": [
            {"name": "Python", "category": "language", "proficiency": "expert",
             "years": "5", "evidence": "Primary language"},
            {"name": "python", "category": "language", "proficiency": "working",
             "years": "3", "evidence": "Used in scripts"},
        ],
        "education": "B.S. CS",
        "certifications": "",
        "domain_knowledge": "",
        "meta": {"career_arc": "", "differentiators": "",
                 "themes_to_emphasize": [], "anti_claims": [], "known_gaps": []},
    })
    pass2_clean = json.dumps({
        "skills": [
            {"name": "Python", "category": "language", "proficiency": "expert",
             "years": "5", "evidence": "Primary language. Used in scripts."},
        ],
        "education": "B.S. CS",
        "certifications": "",
        "domain_knowledge": "",
        "meta": {"career_arc": "", "differentiators": "",
                 "themes_to_emphasize": [], "anti_claims": [], "known_gaps": []},
    })

    fake_client = _FakeClient([
        SAMPLE_CONSOLIDATED_PASS1_JSON,  # pass 1
        pass2_with_dupes,                # pass 2 — still has dupes
        pass2_clean,                     # retry pass 2 — deduped
    ])
    result = consolidate_repo(repo, client=fake_client)
    assert fake_client.call_count == 3  # 2 initial passes + 1 retry
    assert len(result.skills) == 1
    assert result.skills[0].name == "Python"


def test_consolidate_no_retry_when_skills_clean():
    """No retry when pass 2 already returns deduplicated skills."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    repo.skills = [
        SkillEntry(name="Python", category="language"),
        SkillEntry(name="Go", category="language"),
    ]
    fake_client = _FakeClient([
        SAMPLE_CONSOLIDATED_PASS1_JSON,
        SAMPLE_CONSOLIDATED_PASS2_JSON,  # returns 1 skill, no dupes
    ])
    result = consolidate_repo(repo, client=fake_client)
    assert fake_client.call_count == 2  # no retry


def test_consolidation_prompts_have_key_instructions():
    """Verify the consolidation prompts include appropriate merge guidance."""
    # Pass 1 — identity + roles
    assert "IDENTITY" in CONSOLIDATION_PASS1_PROMPT
    assert "ROLES" in CONSOLIDATION_PASS1_PROMPT
    assert "merge" in CONSOLIDATION_PASS1_PROMPT.lower()
    assert "extraction_confidence" in CONSOLIDATION_PASS1_PROMPT
    assert "SKILLS" not in CONSOLIDATION_PASS1_PROMPT
    # Pass 2 — skills + meta
    assert "SKILLS" in CONSOLIDATION_PASS2_PROMPT
    assert "META" in CONSOLIDATION_PASS2_PROMPT
    assert "deduplicate" in CONSOLIDATION_PASS2_PROMPT.lower()
    assert "ROLES" not in CONSOLIDATION_PASS2_PROMPT


def test_consolidation_prompt_preserves_distinct_skills():
    """Pass 2 prompt should instruct LLM to keep related-but-different skills separate."""
    assert "related-but-different" in CONSOLIDATION_PASS2_PROMPT.lower() or \
           "SEPARATE" in CONSOLIDATION_PASS2_PROMPT
    # Key examples of skills NOT to merge
    assert "JavaScript" in CONSOLIDATION_PASS2_PROMPT
    assert "TypeScript" in CONSOLIDATION_PASS2_PROMPT


def test_consolidate_reverts_when_too_many_skills_lost():
    """If pass 2 drops >30% of skills, original skills are kept."""
    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="A", title="Eng", start_date="2020", end_date="2021"),
        RoleEntry(company="B", title="SRE", start_date="2021", end_date="2022"),
    ]
    repo.skills = [
        SkillEntry(name="Python", category="language"),
        SkillEntry(name="Go", category="language"),
        SkillEntry(name="Kubernetes", category="infrastructure"),
        SkillEntry(name="Docker", category="tool"),
        SkillEntry(name="SQL", category="language"),
    ]
    # Pass 2 aggressively drops most skills (returns only 2 of 5)
    aggressive_pass2 = json.dumps({
        "skills": [
            {"name": "Python", "category": "language", "proficiency": "expert",
             "years": "5", "evidence": "Primary language"},
            {"name": "Go", "category": "language", "proficiency": "strong",
             "years": "3", "evidence": "Secondary language"},
        ],
        "education": "", "certifications": "", "domain_knowledge": "",
        "meta": {"career_arc": "", "differentiators": "",
                 "themes_to_emphasize": [], "anti_claims": [], "known_gaps": []},
    })
    fake_client = _FakeClient([
        SAMPLE_CONSOLIDATED_PASS1_JSON,
        aggressive_pass2,
    ])
    result = consolidate_repo(repo, client=fake_client)
    # Safety check should revert to original 5 skills
    assert len(result.skills) == 5


def test_build_repo_identity_first_nonempty_wins():
    """When building from multiple documents, first non-empty identity field wins."""
    repo = _make_repo()
    # First doc fills name
    build_repo_from_parsed({"identity": {"name": "Jordan Lee", "email": ""}}, repo)
    # Second doc also has name — should NOT overwrite
    build_repo_from_parsed({"identity": {"name": "J. Lee", "email": "jordan@example.com"}}, repo)
    assert repo.identity.name == "Jordan Lee"
    assert repo.identity.email == "jordan@example.com"


def test_build_repo_meta_lists_merge():
    """Meta list fields should accumulate across documents, not overwrite."""
    repo = _make_repo()
    build_repo_from_parsed({"meta": {"themes_to_emphasize": ["cost optimization"]}}, repo)
    build_repo_from_parsed({"meta": {"themes_to_emphasize": ["distributed systems", "cost optimization"]}}, repo)
    assert "cost optimization" in repo.meta.themes_to_emphasize
    assert "distributed systems" in repo.meta.themes_to_emphasize
    # No duplicates
    assert repo.meta.themes_to_emphasize.count("cost optimization") == 1


# ---------------------------------------------------------------------------
# Unit tests — compose_stories (mocked LLM)
# ---------------------------------------------------------------------------


SAMPLE_STORIES_JSON = json.dumps([
    {
        "title": "Backend Migration Cost Savings",
        "tags": ["cost-optimization", "migration"],
        "situation": "Legacy monolith was costing $300K/year in infra",
        "task": "Lead migration to microservices architecture",
        "action": "Designed new service mesh, migrated 12 services over 6 months",
        "result": "Reduced infra costs by $180K/year, cut deploy time 60%",
        "what_it_shows": "Can drive large-scale technical initiatives with measurable business impact",
        "extraction_confidence": "high",
        "confidence_notes": "All 4 STAR components have concrete evidence from source",
    }
])


def test_compose_stories_with_mocked_llm(monkeypatch):
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(SAMPLE_STORIES_JSON),
    )

    repo = _make_repo()
    repo.roles = [
        RoleEntry(
            company="DataFlow Inc", title="Senior Engineer",
            start_date="Mar 2021", end_date="Present",
            accomplishments="Led backend migration, cut deploy time 60%. Reduced infra costs by $180K/year.",
        ),
    ]
    agent.compose_stories(repo)
    assert len(repo.stories) == 1
    assert repo.stories[0].title == "Backend Migration Cost Savings"
    assert repo.stories[0].extraction_confidence == "high"
    assert "cost-optimization" in repo.stories[0].tags


def test_compose_stories_empty_roles(monkeypatch):
    """compose_stories should no-op when there are no roles."""
    agent = IngestAgent()
    # Should not even call the LLM
    call_count = 0

    def _fail(**kw):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Should not be called")

    monkeypatch.setattr(agent.client, "chat", _fail)

    repo = _make_repo()
    agent.compose_stories(repo)
    assert repo.stories == []
    assert call_count == 0


def test_compose_stories_llm_failure(monkeypatch):
    """compose_stories should gracefully handle LLM failures."""
    agent = IngestAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("LLM down")),
    )

    repo = _make_repo()
    repo.roles = [
        RoleEntry(company="Test", title="Eng", start_date="2020", end_date="2021",
                  accomplishments="Did stuff"),
    ]
    # Should not raise
    agent.compose_stories(repo)
    assert repo.stories == []


# ---------------------------------------------------------------------------
# Unit tests — _log_token_usage (truncation detection)
# ---------------------------------------------------------------------------


def test_log_token_usage_normal(caplog):
    """Normal responses should log info but no warnings."""
    import logging
    with caplog.at_level(logging.INFO, logger="resume_refinery.ingest_agent"):
        resp = _FakeResponse("content", prompt_eval_count=2000, eval_count=500, done_reason="stop")
        _log_token_usage("test-step", resp)
    assert "[test-step] tokens" in caplog.text
    assert "TRUNCATED" not in caplog.text
    assert "NEAR LIMIT" not in caplog.text


def test_log_token_usage_output_truncated(caplog):
    """done_reason='length' should trigger an OUTPUT TRUNCATED warning."""
    import logging
    with caplog.at_level(logging.WARNING, logger="resume_refinery.ingest_agent"):
        resp = _FakeResponse("content", prompt_eval_count=2000, eval_count=8192, done_reason="length")
        _log_token_usage("test-step", resp)
    assert "OUTPUT TRUNCATED" in caplog.text
    assert "RESUME_REFINERY_MAX_TOKENS" in caplog.text


def test_log_token_usage_input_near_limit(caplog):
    """Prompt tokens near NUM_CTX should trigger an INPUT NEAR LIMIT warning."""
    import logging
    from resume_refinery.ingest_agent import NUM_CTX
    with caplog.at_level(logging.WARNING, logger="resume_refinery.ingest_agent"):
        near_limit = int(NUM_CTX * 0.96)
        resp = _FakeResponse("content", prompt_eval_count=near_limit, eval_count=100, done_reason="stop")
        _log_token_usage("test-step", resp)
    assert "INPUT NEAR LIMIT" in caplog.text
    assert "RESUME_REFINERY_NUM_CTX" in caplog.text


def test_log_token_usage_missing_attrs(caplog):
    """Responses without token attrs (e.g. test fakes) should not crash."""
    import logging

    class _BareResponse:
        def __init__(self):
            self.message = _FakeMessage("content")

    with caplog.at_level(logging.INFO, logger="resume_refinery.ingest_agent"):
        _log_token_usage("test-step", _BareResponse())
    assert "[test-step] tokens" in caplog.text
    assert "TRUNCATED" not in caplog.text
