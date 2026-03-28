"""Tests for the ElicitationAgent."""

import pytest

from resume_refinery.elicitation import (
    ElicitationAgent,
    _build_role_context,
    _parse_probes,
    _static_probes,
    _strip_think_tags,
)
from resume_refinery.models import RoleEntry


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_build_role_context_full():
    role = RoleEntry(
        company="Acme", title="Eng", start_date="2020", end_date="2023",
        company_context="SaaS startup", team_context="Team of 5",
        ownership="Auth system", accomplishments="Cut latency 40%",
        technologies="Python, Postgres", learnings="Scaling under pressure",
    )
    ctx = _build_role_context(role)
    assert "Acme" in ctx
    assert "Cut latency 40%" in ctx
    assert "SaaS startup" in ctx
    assert "Team of 5" in ctx


def test_build_role_context_minimal():
    role = RoleEntry(company="Acme", title="Eng", start_date="2020")
    ctx = _build_role_context(role)
    assert "Acme" in ctx
    # Empty fields should be absent
    assert "Company context:" not in ctx


def test_parse_probes_numbered():
    text = "1. What did you do?\n2. Can you quantify that?\n3. How big was the team?"
    probes = _parse_probes(text)
    assert len(probes) == 3
    assert probes[0] == "What did you do?"


def test_parse_probes_bulleted():
    text = "- First question?\n- Second question?"
    probes = _parse_probes(text)
    assert len(probes) == 2


def test_parse_probes_max_four():
    text = "\n".join(f"{i}. Question {i}?" for i in range(1, 8))
    probes = _parse_probes(text)
    assert len(probes) == 4


def test_parse_probes_empty():
    assert _parse_probes("") == []
    assert _parse_probes("   \n  ") == []


def test_strip_think_tags():
    assert _strip_think_tags("hello") == "hello"
    assert _strip_think_tags("<think>inner</think>result") == "result"
    assert _strip_think_tags("before<think>x</think>after") == "beforeafter"


# ---------------------------------------------------------------------------
# Static fallback probes
# ---------------------------------------------------------------------------


def test_static_probes_thin_data():
    role = RoleEntry(
        company="Acme", title="Eng", start_date="2020",
        accomplishments="Did stuff",
    )
    probes = _static_probes(role)
    assert len(probes) >= 2
    assert any("quantify" in p.lower() for p in probes)


def test_static_probes_quantified_data():
    role = RoleEntry(
        company="Acme", title="Eng", start_date="2020",
        company_context="B2B SaaS, 200 employees",
        ownership="Auth pipeline",
        accomplishments="Improved throughput by 40%, saving $120K/year in compute costs. "
                        "Migrated 50 services to new auth system over 8 weeks.",
    )
    probes = _static_probes(role)
    # Already quantified and detailed — context and ownership present
    assert len(probes) == 0


def test_static_probes_missing_context():
    role = RoleEntry(
        company="Acme", title="Eng", start_date="2020",
        accomplishments="Cut latency by 50%, saved $200K. Redesigned entire auth pipeline "
                        "and migrated 20 services.",
        ownership="Auth pipeline",
    )
    probes = _static_probes(role)
    assert any("company" in p.lower() for p in probes)


# ---------------------------------------------------------------------------
# Integration test with mocked LLM
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


def test_probe_role_with_mocked_llm(monkeypatch):
    agent = ElicitationAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse("1. Can you quantify the impact?\n2. What was the team size?"),
    )

    role = RoleEntry(company="Acme", title="Eng", start_date="2020", accomplishments="Did stuff")
    probes = agent.probe_role(role)
    assert len(probes) == 2
    assert "quantify" in probes[0].lower()


def test_probe_role_looks_good(monkeypatch):
    agent = ElicitationAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse("LOOKS_GOOD"),
    )

    role = RoleEntry(
        company="Acme", title="Eng", start_date="2020",
        accomplishments="Cut latency by 50%, saving $200K.",
    )
    probes = agent.probe_role(role)
    assert probes == []


def test_probe_role_llm_failure_falls_back(monkeypatch):
    agent = ElicitationAgent()

    def _fail(**kw):
        raise ConnectionError("LLM down")

    monkeypatch.setattr(agent.client, "chat", _fail)

    role = RoleEntry(company="Acme", title="Eng", start_date="2020", accomplishments="Did stuff")
    probes = agent.probe_role(role)
    # Should get static fallback probes
    assert len(probes) >= 1


def test_probe_role_with_think_tags(monkeypatch):
    agent = ElicitationAgent()
    monkeypatch.setattr(
        agent.client, "chat",
        lambda **kw: _FakeResponse(
            "<think>Let me analyze...</think>1. How many users were affected?\n2. What was the timeline?"
        ),
    )

    role = RoleEntry(company="Acme", title="Eng", start_date="2020", accomplishments="Did stuff")
    probes = agent.probe_role(role)
    assert len(probes) == 2
    assert "think" not in probes[0].lower()
