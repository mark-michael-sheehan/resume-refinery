"""Tests for bounded specialist agents."""

from resume_refinery.specialist_agents import EvidenceAgent, VoiceAgent


def test_evidence_agent_extracts_requirements_and_matches(career_profile, job_description):
    agent = EvidenceAgent()
    pack = agent.build_evidence_pack(career_profile, job_description)

    assert pack.job_requirements
    assert any("distributed" in item.requirement.lower() or "python" in item.requirement.lower() for item in pack.job_requirements)
    assert pack.matched_evidence
    assert any("distributed systems" in item.evidence.lower() or "engineer" in item.evidence.lower() for item in pack.matched_evidence)


def test_voice_agent_distills_style_guide(voice_profile):
    agent = VoiceAgent()
    guide = agent.build_style_guide(voice_profile)

    assert guide.core_adjectives
    assert any("direct" in item.lower() for item in guide.core_adjectives)
    assert guide.style_rules
    assert any("short declarative sentences" in item.lower() for item in guide.style_rules)
