"""Tests for the career wizard web routes."""

import pytest
from fastapi.testclient import TestClient

from resume_refinery.career_repo import CareerRepoStore
from resume_refinery.career_wizard import career_store, router
from resume_refinery.webapp import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path / "sessions"))
    # Reinitialise the module-level store so it picks up the env var
    from resume_refinery import career_wizard
    career_wizard.career_store = CareerRepoStore()
    return TestClient(app)


@pytest.fixture
def repo_id(client):
    """Create a career repo and return its ID."""
    resp = client.post("/career/new", data={"name": "Test User"}, follow_redirects=False)
    assert resp.status_code == 303
    # Extract repo_id from redirect Location header
    location = resp.headers["location"]
    return location.split("/career/")[1]


def test_career_index(client):
    resp = client.get("/career")
    assert resp.status_code == 200
    assert "Career Builder" in resp.text


def test_create_career(client):
    resp = client.post("/career/new", data={"name": "Jordan Lee"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "/career/jordan-lee" in resp.headers["location"]


def test_career_show_identity_phase(client, repo_id):
    resp = client.get(f"/career/{repo_id}")
    assert resp.status_code == 200
    assert "Identity" in resp.text
    assert "Full Name" in resp.text


def test_save_identity(client, repo_id):
    resp = client.post(f"/career/{repo_id}/identity", data={
        "name": "Test User",
        "email": "test@example.com",
        "phone": "555-1234",
        "location": "NYC",
        "linkedin": "",
        "github": "",
        "headline": "Engineer",
        "target_roles": "Staff Eng, EM",
        "education": "BS CS",
        "certifications": "AWS SA",
    }, follow_redirects=False)
    assert resp.status_code == 303

    # Should now be on roles phase
    resp = client.get(f"/career/{repo_id}")
    assert resp.status_code == 200
    assert "Role Timeline" in resp.text


def test_add_role(client, repo_id):
    # Advance to roles phase first
    client.post(f"/career/{repo_id}/identity", data={
        "name": "Test User", "email": "", "phone": "", "location": "",
        "linkedin": "", "github": "", "headline": "", "target_roles": "",
        "education": "", "certifications": "",
    })

    resp = client.post(f"/career/{repo_id}/roles", data={
        "company": "Acme Corp",
        "title": "Engineer",
        "start_date": "2020",
        "end_date": "Present",
    }, follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get(f"/career/{repo_id}")
    assert "Acme Corp" in resp.text


def test_delete_role(client, repo_id):
    # Setup: add a role
    client.post(f"/career/{repo_id}/identity", data={
        "name": "Test User", "email": "", "phone": "", "location": "",
        "linkedin": "", "github": "", "headline": "", "target_roles": "",
        "education": "", "certifications": "",
    })
    client.post(f"/career/{repo_id}/roles", data={
        "company": "Deleteme Corp", "title": "Eng", "start_date": "2020", "end_date": "2021",
    })

    resp = client.post(f"/career/{repo_id}/roles/0/delete", follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get(f"/career/{repo_id}")
    assert "Deleteme Corp" not in resp.text


def test_advance_phase(client, repo_id):
    # Advance to skills
    resp = client.post(f"/career/{repo_id}/advance/skills", follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get(f"/career/{repo_id}")
    assert "Skills Inventory" in resp.text


def test_advance_invalid_phase(client, repo_id):
    resp = client.post(f"/career/{repo_id}/advance/bogus")
    assert resp.status_code == 400


def test_add_skill(client, repo_id):
    client.post(f"/career/{repo_id}/advance/skills")

    resp = client.post(f"/career/{repo_id}/skills", data={
        "name": "Python",
        "category": "language",
        "proficiency": "expert",
        "years": "6+",
        "evidence": "Primary language at work",
    }, follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get(f"/career/{repo_id}/skills")
    assert "Python" in resp.text


def test_delete_skill(client, repo_id):
    client.post(f"/career/{repo_id}/advance/skills")
    client.post(f"/career/{repo_id}/skills", data={
        "name": "DeleteMe", "category": "other", "proficiency": "familiar",
        "years": "", "evidence": "",
    })

    resp = client.post(f"/career/{repo_id}/skills/0/delete", follow_redirects=False)
    assert resp.status_code == 303


def test_add_story(client, repo_id):
    client.post(f"/career/{repo_id}/advance/stories")

    resp = client.post(f"/career/{repo_id}/stories", data={
        "title": "Cost Savings",
        "tags": "cost, initiative",
        "situation": "Bills were high",
        "task": "Investigate",
        "action": "Redesigned caching",
        "result": "Saved $180K",
        "what_it_shows": "Initiative",
    }, follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get(f"/career/{repo_id}/stories")
    assert "Cost Savings" in resp.text


def test_save_meta(client, repo_id):
    client.post(f"/career/{repo_id}/advance/meta")

    resp = client.post(f"/career/{repo_id}/meta", data={
        "career_arc": "IC to staff",
        "differentiators": "Simplification over complexity",
        "themes": "Ownership\nImpact",
        "anti_claims": "Not a manager",
        "known_gaps": "No hyperscale",
        "domain_knowledge": "SaaS analytics",
    }, follow_redirects=False)
    assert resp.status_code == 303

    # Should advance to voice phase
    resp = client.get(f"/career/{repo_id}")
    assert "Voice Profile" in resp.text


def test_save_voice(client, repo_id):
    client.post(f"/career/{repo_id}/advance/voice")

    resp = client.post(f"/career/{repo_id}/voice", data={
        "adjectives": "Direct, analytical",
        "style_notes": "Short sentences\nOutcome first",
        "preferred_phrases": "The key insight was...",
        "avoid_phrases": "Passionate about",
        "sample_1": "When I look at a system problem, I start simple.",
        "sample_2": "",
    }, follow_redirects=False)
    assert resp.status_code == 303

    # Should advance to review phase
    resp = client.get(f"/career/{repo_id}")
    assert "Career Repository" in resp.text


def test_probe_endpoint(client, repo_id, monkeypatch):
    # Mock the elicitation agent so we don't need a running LLM
    from resume_refinery import career_wizard
    from resume_refinery.elicitation import _static_probes, ProbeResult
    from unittest.mock import MagicMock

    mock_agent = MagicMock()
    mock_agent.probe_role = lambda role: ProbeResult(probes=_static_probes(role), llm_used=False)
    monkeypatch.setattr(career_wizard, "elicitation_agent", mock_agent)

    # Add a role with thin data
    client.post(f"/career/{repo_id}/advance/roles")
    client.post(f"/career/{repo_id}/roles", data={
        "company": "Acme", "title": "Eng", "start_date": "2020", "end_date": "Present",
    })
    client.post(f"/career/{repo_id}/advance/role_deepdive")

    # Save minimal accomplishments
    client.post(f"/career/{repo_id}/role_deepdive/0", data={
        "company_context": "", "team_context": "", "ownership": "",
        "accomplishments": "Did stuff", "technologies": "", "learnings": "", "anti_claims": "",
    })

    resp = client.post(f"/career/{repo_id}/role_deepdive/0/probe")
    assert resp.status_code == 200
    # Should get follow-up probes since data is thin
    assert "quantify" in resp.text.lower() or "detail" in resp.text.lower()
    # Should show LLM unavailable warning since mock returns llm_used=False
    assert "LLM is unavailable" in resp.text


def test_review_page(client, repo_id):
    client.post(f"/career/{repo_id}/advance/complete")
    resp = client.get(f"/career/{repo_id}")
    assert resp.status_code == 200
    assert "Career Repository" in resp.text


def test_career_not_found(client):
    resp = client.get("/career/nonexistent")
    assert resp.status_code == 404


def test_voice_prepopulation(client, repo_id):
    """Voice form should pre-populate fields from a previously saved voice_raw."""
    client.post(f"/career/{repo_id}/advance/voice")
    # Save voice data
    client.post(f"/career/{repo_id}/voice", data={
        "adjectives": "Direct, concise",
        "style_notes": "Short sentences\nOutcome first",
        "preferred_phrases": "The key insight was...",
        "avoid_phrases": "Passionate about",
        "sample_1": "I start with the simplest solution.",
        "sample_2": "",
    })
    # Go back to voice phase to check pre-population
    client.post(f"/career/{repo_id}/advance/voice")
    resp = client.get(f"/career/{repo_id}/voice")
    assert resp.status_code == 200
    assert "Direct" in resp.text
    assert "Short sentences" in resp.text
    assert "The key insight was" in resp.text
    assert "Passionate about" in resp.text
    assert "simplest solution" in resp.text


def test_htmx_probe_button_present(client, repo_id):
    """The deep-dive form should include an HTMX probe button."""
    client.post(f"/career/{repo_id}/identity", data={
        "name": "Test User", "email": "", "phone": "", "location": "",
        "linkedin": "", "github": "", "headline": "", "target_roles": "",
        "education": "", "certifications": "",
    })
    client.post(f"/career/{repo_id}/roles", data={
        "company": "Acme", "title": "Eng", "start_date": "2020", "end_date": "Present",
    })
    client.post(f"/career/{repo_id}/advance/role_deepdive")
    resp = client.get(f"/career/{repo_id}")
    assert "hx-post" in resp.text
    assert "probe-area" in resp.text


def test_ingest_advances_to_role_deepdive(client, monkeypatch):
    """After document ingestion, phase should be roles with needs_consolidation."""
    from resume_refinery import career_wizard
    from unittest.mock import MagicMock
    import re

    mock_ingest = MagicMock()
    # Simulate ingest populating a role
    def fake_ingest(text, repo):
        from resume_refinery.models import RoleEntry
        repo.roles.append(RoleEntry(company="Acme", title="Engineer",
                                     start_date="2020", end_date="Present"))
    mock_ingest.ingest_to_repo = fake_ingest
    monkeypatch.setattr(career_wizard, "ingest_agent", mock_ingest)

    # Mock consolidate_roles so it doesn't call the LLM
    monkeypatch.setattr(career_wizard, "consolidate_roles", lambda repo, client=None: repo)

    resp = client.post("/career/ingest", data={"name": "Ingest User"},
                       files=[("files", ("resume.txt", b"Worked at Acme as Engineer", "text/plain"))])
    assert resp.status_code == 200
    # Streaming progress page embeds a JS redirect to the new profile
    match = re.search(r"window\.location\.href='(/career/[^']+)'", resp.text)
    assert match, "Expected JS redirect in progress page"
    repo_id = match.group(1).split("/career/")[1]

    # Load the repo and verify phase
    repo = career_wizard.career_store.get(repo_id)
    assert repo.current_phase == "roles"
    assert repo.needs_consolidation is True


def test_ingest_roles_page_shows_finalize_button(client, monkeypatch):
    """After ingestion the roles page should show 'Finalize & Build Stories'."""
    from resume_refinery import career_wizard
    from unittest.mock import MagicMock
    import re

    mock_ingest = MagicMock()
    def fake_ingest(text, repo):
        from resume_refinery.models import RoleEntry
        repo.roles.append(RoleEntry(company="Acme", title="Engineer",
                                     start_date="2020", end_date="Present"))
    mock_ingest.ingest_to_repo = fake_ingest
    monkeypatch.setattr(career_wizard, "ingest_agent", mock_ingest)
    monkeypatch.setattr(career_wizard, "consolidate_roles", lambda repo, client=None: repo)

    resp = client.post("/career/ingest", data={"name": "Gate User"},
                       files=[("files", ("resume.txt", b"Worked at Acme", "text/plain"))])
    match = re.search(r"window\.location\.href='(/career/[^']+)'", resp.text)
    assert match
    redirect_url = match.group(1)

    resp = client.get(redirect_url)
    assert resp.status_code == 200
    assert "Finalize" in resp.text
    assert "Build Stories" in resp.text
    # Should NOT show the normal "Continue to Deep Dive" button
    assert "Continue to Deep Dive" not in resp.text


def test_finalize_runs_consolidation_and_stories(client, monkeypatch):
    """The finalize endpoint should consolidate skills/meta, compose stories, and advance phase."""
    from resume_refinery import career_wizard
    from resume_refinery.models import RoleEntry
    from unittest.mock import MagicMock, patch
    import re

    mock_ingest = MagicMock()
    def fake_ingest(text, repo):
        repo.roles.append(RoleEntry(company="Acme", title="Engineer",
                                     start_date="2020", end_date="Present"))
    mock_ingest.ingest_to_repo = fake_ingest
    monkeypatch.setattr(career_wizard, "ingest_agent", mock_ingest)
    monkeypatch.setattr(career_wizard, "consolidate_roles", lambda repo, client=None: repo)

    # Ingest first
    resp = client.post("/career/ingest", data={"name": "Final User"},
                       files=[("files", ("resume.txt", b"Worked at Acme", "text/plain"))])
    match = re.search(r"window\.location\.href='(/career/[^']+)'", resp.text)
    assert match
    repo_id = match.group(1).split("/career/")[1]

    # Mock consolidate_skills_meta to be a no-op that returns the same repo
    def fake_consolidate(repo, client=None):
        return repo
    monkeypatch.setattr(career_wizard, "consolidate_skills_meta", fake_consolidate)

    # Finalize
    resp = client.post(f"/career/{repo_id}/finalize")
    assert resp.status_code == 200
    assert "Consolidating" in resp.text
    assert "Composing" in resp.text

    # Verify repo state after finalize
    repo = career_wizard.career_store.get(repo_id)
    assert repo.current_phase == "role_deepdive"
    assert repo.needs_consolidation is False
