"""Tests for career repository models and storage."""

import json
import pytest

from resume_refinery.models import (
    CareerIdentity,
    CareerMeta,
    CareerRepository,
    RoleEntry,
    SkillEntry,
    StoryEntry,
)
from resume_refinery.career_repo import CareerRepoStore


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_career_repository_defaults():
    repo = CareerRepository(repo_id="test")
    assert repo.current_phase == "identity"
    assert repo.identity.name == ""
    assert repo.roles == []
    assert repo.skills == []
    assert repo.stories == []
    assert repo.voice_raw == ""


def test_role_entry_slug():
    role = RoleEntry(company="DataFlow Inc", title="Senior Engineer", start_date="Mar 2021")
    slug = role.slug()
    assert " " not in slug
    assert "dataflow" in slug.lower()


def test_career_repository_to_career_profile():
    repo = CareerRepository(repo_id="test")
    repo.identity = CareerIdentity(
        name="Jordan Lee",
        email="jordan@example.com",
        phone="415-555-0100",
        location="San Francisco, CA",
        headline="Distributed systems engineer",
    )
    repo.roles = [
        RoleEntry(
            company="DataFlow Inc",
            title="Senior Engineer",
            start_date="2021",
            end_date="Present",
            accomplishments="Reduced infra costs by $180K/year",
            technologies="Python, FastAPI, Kafka",
        ),
    ]
    repo.skills = [
        SkillEntry(name="Python", proficiency="expert", years="6+", evidence="Primary language"),
    ]
    repo.stories = [
        StoryEntry(
            title="Redis Cost Savings",
            tags=["cost-optimization"],
            situation="Infra bill climbing",
            action="Redesigned caching",
            result="$180K/year saved",
        ),
    ]
    repo.meta = CareerMeta(
        career_arc="IC to staff+",
        differentiators="Finds and fixes unassigned problems",
        themes_to_emphasize=["Ownership mentality"],
        anti_claims=["Not a frontend engineer"],
        known_gaps=["No hyperscale experience"],
    )

    profile = repo.to_career_profile()

    assert profile.name == "Jordan Lee"
    assert profile.email == "jordan@example.com"
    assert "Jordan Lee" in profile.raw_content
    assert "Senior Engineer @ DataFlow Inc" in profile.raw_content
    assert "$180K/year" in profile.raw_content
    assert "Python" in profile.raw_content
    assert "Redis Cost Savings" in profile.raw_content
    assert "Ownership mentality" in profile.raw_content
    assert "Not a frontend engineer" in profile.raw_content


def test_career_repository_to_career_profile_empty():
    repo = CareerRepository(repo_id="empty")
    profile = repo.to_career_profile()
    assert profile.raw_content is not None
    assert profile.name is None


def test_career_repository_voice_raw():
    repo = CareerRepository(repo_id="test", voice_raw="# Voice\nDirect and precise")
    assert "Direct and precise" in repo.voice_raw


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


def test_career_repo_store_create(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    repo = store.create("Jordan Lee")

    assert repo.repo_id == "jordan-lee"
    assert repo.identity.name == "Jordan Lee"
    assert (tmp_path / "jordan-lee" / "career.json").exists()


def test_career_repo_store_create_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    repo1 = store.create("Jordan Lee")
    repo2 = store.create("Jordan Lee")

    assert repo1.repo_id == "jordan-lee"
    assert repo2.repo_id == "jordan-lee-2"


def test_career_repo_store_save_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    repo = store.create("Test User")
    repo.identity.email = "test@example.com"
    repo.roles.append(RoleEntry(company="Acme", title="Engineer", start_date="2020"))
    repo.current_phase = "roles"
    store.save(repo)

    loaded = store.get(repo.repo_id)
    assert loaded.identity.email == "test@example.com"
    assert len(loaded.roles) == 1
    assert loaded.current_phase == "roles"


def test_career_repo_store_list(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    store.create("Alice")
    store.create("Bob")

    repos = store.list_repos()
    assert len(repos) == 2
    names = {r.identity.name for r in repos}
    assert names == {"Alice", "Bob"}


def test_career_repo_store_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    repo = store.create("Disposable")
    assert (tmp_path / repo.repo_id).exists()

    store.delete(repo.repo_id)
    assert not (tmp_path / repo.repo_id).exists()


def test_career_repo_store_get_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    with pytest.raises(FileNotFoundError):
        store.get("nonexistent")


def test_career_repo_store_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_REFINERY_CAREERS_DIR", str(tmp_path))
    store = CareerRepoStore()

    repo = store.create("Bad Json")
    json_path = tmp_path / repo.repo_id / "career.json"
    json_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted"):
        store.get(repo.repo_id)
