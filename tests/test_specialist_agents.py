"""Tests for bounded specialist agents."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    EvidencePack,
    JobRequirement,
    RepairPassResult,
    ReviewBundle,
    TruthfulnessResult,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from resume_refinery.specialist_agents import (
    DraftingAgent,
    EvidenceAgent,
    RepairAgent,
    VerificationAgent,
    VoiceAgent,
)


# ---------------------------------------------------------------------------
# EvidenceAgent
# ---------------------------------------------------------------------------


def _make_llm_resp(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def test_evidence_agent_extracts_requirements_and_matches(career_profile, job_description):
    req_json = json.dumps([
        {"requirement": "Python expertise", "category": "skill"},
        {"requirement": "Distributed systems experience", "category": "skill"},
    ])
    ev_json = json.dumps([
        {"evidence": "Led distributed systems migration, cut deploy time 60%", "relevance_score": 5}
    ])
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_llm_resp(req_json),
        _make_llm_resp(ev_json),
        _make_llm_resp(ev_json),
    ]
    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job_description)

    assert pack.job_requirements
    assert any("python" in item.requirement.lower() or "distributed" in item.requirement.lower() for item in pack.job_requirements)
    assert pack.matched_evidence
    assert any("distributed" in item.evidence.lower() or "engineer" in item.evidence.lower() for item in pack.matched_evidence)


def test_evidence_agent_identifies_gaps(career_profile):
    """When the job requires something not in the career profile, it should appear in gaps."""
    from resume_refinery.models import JobDescription
    job = JobDescription(
        raw_content="Required: Rust experience, quantum computing, blockchain architecture.",
        title="Quantum Engineer",
        company="QuantumCo",
    )
    req_json = json.dumps([
        {"requirement": "Rust experience", "category": "skill"},
        {"requirement": "Quantum computing", "category": "skill"},
    ])
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_llm_resp(req_json),
        _make_llm_resp(json.dumps([])),  # no evidence for Rust
        _make_llm_resp(json.dumps([])),  # no evidence for Quantum
    ]
    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job)

    assert len(pack.gaps) > 0


def test_evidence_agent_fallback_requirements():
    """When the LLM fails, keyword/line fallback should still produce requirements."""
    from resume_refinery.models import CareerProfile, JobDescription
    job = JobDescription(
        raw_content="# Data Scientist\nCompany: BigCo\n\nBuild models.\nImprove metrics.",
        title="Data Scientist",
        company="BigCo",
    )
    career = CareerProfile(raw_content="# Someone\nDoes stuff.")
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")
    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career, job)

    assert pack.job_requirements


def test_evidence_agent_source_summary(career_profile, job_description):
    """source_summary is derived from career profile lines — no LLM needed."""
    req_json = json.dumps([{"requirement": "Python", "category": "skill"}])
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_llm_resp(req_json),
        _make_llm_resp(json.dumps([])),
    ]
    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job_description)

    assert pack.source_summary  # Non-empty


def test_evidence_agent_limits_requirements():
    """LLM extraction already caps at 10 via [:10] slice."""
    from resume_refinery.models import CareerProfile, JobDescription
    lines = "\n".join(f"- Required: skill_{i} experience" for i in range(20))
    job = JobDescription(raw_content=f"# Job\n{lines}", title="Job", company="Co")
    career = CareerProfile(raw_content="# Name\nDoes things.")
    # Return 15 requirements from LLM — slice should cap at 10
    big_reqs = json.dumps([{"requirement": f"skill_{i}", "category": "skill"} for i in range(15)])
    mock_client = MagicMock()
    # First call = extraction; subsequent calls = evidence matching (10 calls for 10 reqs)
    mock_client.chat.side_effect = [_make_llm_resp(big_reqs)] + [
        _make_llm_resp(json.dumps([])) for _ in range(10)
    ]
    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career, job)

    assert len(pack.job_requirements) <= 10


# ---------------------------------------------------------------------------
# VoiceAgent
# ---------------------------------------------------------------------------


def test_voice_agent_distills_style_guide(voice_profile):
    agent = VoiceAgent()
    guide = agent.build_style_guide(voice_profile)

    assert guide.core_adjectives
    assert any("direct" in item.lower() for item in guide.core_adjectives)
    assert guide.style_rules
    assert any("short declarative sentences" in item.lower() for item in guide.style_rules)


# ---------------------------------------------------------------------------
# VoiceAgent
# ---------------------------------------------------------------------------


def test_voice_agent_phrases_to_avoid():
    from resume_refinery.models import VoiceProfile
    voice = VoiceProfile(raw_content=(
        "# Voice\n"
        "## Adjectives\n- Bold\n"
        "## Phrases to Avoid\n- results-driven\n- industry-leading\n"
    ))
    agent = VoiceAgent()
    guide = agent.build_style_guide(voice)
    assert any("results-driven" in p for p in guide.phrases_to_avoid)


def test_voice_agent_writing_samples():
    from resume_refinery.models import VoiceProfile
    voice = VoiceProfile(raw_content=(
        "# Voice\n"
        "## Writing Samples\n"
        "I built the thing and it worked.\n\n"
        "We shipped on time. No drama.\n"
    ))
    agent = VoiceAgent()
    guide = agent.build_style_guide(voice)
    assert guide.writing_samples


def test_voice_agent_empty_profile():
    from resume_refinery.models import VoiceProfile
    voice = VoiceProfile(raw_content="nothing structured here")
    agent = VoiceAgent()
    guide = agent.build_style_guide(voice)
    # Should not crash — returns empty/minimal guide
    assert isinstance(guide, VoiceStyleGuide)


# ---------------------------------------------------------------------------
# DraftingAgent
# ---------------------------------------------------------------------------


def _make_context():
    return DraftingContext(
        evidence_pack=EvidencePack(
            job_requirements=[JobRequirement(requirement="Python")],
            matched_evidence=[],
            gaps=["Rust"],
            source_summary=["Built things"],
        ),
        voice_style_guide=VoiceStyleGuide(
            core_adjectives=["direct"],
            style_rules=["Short sentences"],
        ),
    )


def test_drafting_agent_generate_all(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "Generated."

    agent = DraftingAgent(generator=mock_generator)
    docs = agent.generate_all(career_profile, voice_profile, job_description, _make_context())

    assert docs.cover_letter == "Generated."
    assert docs.resume == "Generated."
    assert docs.interview_guide == "Generated."
    assert mock_generator.generate_document.call_count == 3


def test_drafting_agent_generate_document_with_feedback(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "Revised."

    agent = DraftingAgent(generator=mock_generator)
    result = agent.generate_document(
        "cover_letter", career_profile, voice_profile, job_description, _make_context(),
        feedback="Shorten it", previous_version="Old draft",
    )

    assert result == "Revised."
    call_kwargs = mock_generator.generate_document.call_args
    assert call_kwargs.kwargs["feedback"] == "Shorten it"
    assert call_kwargs.kwargs["previous_version"] == "Old draft"


def test_drafting_agent_stream_document(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.stream_document.return_value = iter(["chunk1", "chunk2"])

    agent = DraftingAgent(generator=mock_generator)
    chunks = list(agent.stream_document("resume", career_profile, voice_profile, job_description, _make_context()))

    assert chunks == ["chunk1", "chunk2"]


def test_drafting_agent_career_context_includes_evidence(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "doc"

    agent = DraftingAgent(generator=mock_generator)
    agent.generate_document("resume", career_profile, voice_profile, job_description, _make_context())

    # The career profile passed to the generator should include evidence pack info
    enriched_career = mock_generator.generate_document.call_args.args[1]
    assert "Evidence Pack" in enriched_career.raw_content
    assert "Python" in enriched_career.raw_content


def test_drafting_agent_voice_context_includes_guide(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "doc"

    agent = DraftingAgent(generator=mock_generator)
    agent.generate_document("resume", career_profile, voice_profile, job_description, _make_context())

    enriched_voice = mock_generator.generate_document.call_args.args[2]
    assert "Distilled Voice Guide" in enriched_voice.raw_content
    assert "direct" in enriched_voice.raw_content


# ---------------------------------------------------------------------------
# VerificationAgent
# ---------------------------------------------------------------------------


class FakeReviewer:
    def review_truthfulness(self, docs, career, job):
        doc = DocumentTruthResult(pass_strict=True)
        return TruthfulnessResult(
            all_supported=True,
            cover_letter=doc, resume=doc, interview_guide=doc,
        )

    def review_voice(self, docs, voice):
        return VoiceReviewResult(
            overall_match="strong",
            cover_letter_assessment="Good",
            resume_assessment="Good",
            interview_guide_assessment="Good",
        )

    def review_ai_detection(self, docs):
        return AIDetectionResult(risk_level="low")


def test_verification_agent_review_all(document_set, career_profile, voice_profile, job_description):
    agent = VerificationAgent(reviewer=FakeReviewer())
    bundle = agent.review_all(document_set, career_profile, voice_profile, job_description)

    assert bundle.truthfulness is not None
    assert bundle.voice is not None
    assert bundle.ai_detection is not None
    assert bundle.truthfulness.all_supported is True


def test_verification_agent_review_truthfulness(document_set, career_profile, job_description):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_truthfulness(document_set, career_profile, job_description)
    assert result.all_supported is True


def test_verification_agent_review_voice(document_set, voice_profile):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_voice(document_set, voice_profile)
    assert result.overall_match == "strong"


def test_verification_agent_review_ai_detection(document_set):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_ai_detection(document_set)
    assert result.risk_level == "low"


# ---------------------------------------------------------------------------
# LLM-powered EvidenceAgent tests
# ---------------------------------------------------------------------------


def _make_llm_response(response_text: str):
    """Build an ollama chat response mock."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_response = MagicMock()
    mock_response.message = mock_message
    return mock_response


def test_evidence_agent_llm_requirement_extraction(career_profile, job_description):
    """When LLM is available, requirements should come from the LLM."""
    llm_response = json.dumps([
        {"requirement": "Python expertise", "category": "skill"},
        {"requirement": "Distributed systems experience", "category": "skill"},
        {"requirement": "Technical leadership", "category": "leadership"},
    ])
    mock_client = MagicMock()
    # First call = requirement extraction, subsequent calls = evidence matching (one per requirement)
    evidence_response = json.dumps([
        {"evidence": "Led backend migration, cut deploy time 60%", "relevance_score": 5}
    ])
    mock_client.chat.side_effect = [
        _make_llm_response(llm_response),
        _make_llm_response(evidence_response),
        _make_llm_response(evidence_response),
        _make_llm_response(evidence_response),
    ]

    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job_description)

    assert len(pack.job_requirements) == 3
    assert pack.job_requirements[0].requirement == "Python expertise"
    assert pack.job_requirements[0].category == "skill"
    assert pack.matched_evidence  # Should have evidence from LLM


def test_evidence_agent_llm_evidence_matching(career_profile, job_description):
    """LLM evidence matching should return EvidenceItems with relevance scores."""
    req_response = json.dumps([{"requirement": "Python", "category": "skill"}])
    evidence_response = json.dumps([
        {"evidence": "Led backend migration in Python", "relevance_score": 5},
        {"evidence": "Built data pipeline", "relevance_score": 3},
    ])
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_llm_response(req_response),
        _make_llm_response(evidence_response),
    ]

    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job_description)

    assert len(pack.matched_evidence) == 2
    assert pack.matched_evidence[0].relevance_score == 5


def test_evidence_agent_llm_returns_gaps_for_unmatched(career_profile):
    """When LLM finds no evidence, the requirement should appear as a gap."""
    from resume_refinery.models import JobDescription
    job = JobDescription(raw_content="Required: Quantum computing", title="QC", company="QCo")

    req_response = json.dumps([{"requirement": "Quantum computing", "category": "skill"}])
    empty_evidence = json.dumps([])
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        _make_llm_response(req_response),
        _make_llm_response(empty_evidence),
    ]

    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job)

    assert "Quantum computing" in pack.gaps


def test_evidence_agent_falls_back_on_llm_failure(career_profile, job_description):
    """When LLM calls fail, keyword fallback should still produce results."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")

    agent = EvidenceAgent(client=mock_client)
    pack = agent.build_evidence_pack(career_profile, job_description)

    # Should still get results from keyword fallback
    assert pack.job_requirements
    assert pack.matched_evidence or pack.gaps  # Either found evidence or identified gaps


# ---------------------------------------------------------------------------
# RepairAgent  (surgical find/replace edits)
# ---------------------------------------------------------------------------


def test_repair_unified_applies_surgical_edits(career_profile, voice_profile, job_description):
    """repair_unified should call _plan_edits and apply edits programmatically."""
    agent = RepairAgent()

    docs = DocumentSet(
        cover_letter="I am a passionate innovator with quantum AI expertise.",
        resume="old resume",
        interview_guide="old ig",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["quantum AI expertise"],
    )
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=doc_pass,
        interview_guide=doc_pass,
    )
    context = _make_context()

    # Mock _plan_edits to return a surgical edit
    agent._plan_edits = MagicMock(return_value=[
        {"find": "quantum AI expertise", "replace": "backend migration experience", "reason": "truthfulness"},
    ])

    agent.repair_unified(
        docs, truth, None, None,
        career_profile, voice_profile, job_description, context,
    )

    # Only cover_letter was repaired
    assert "backend migration experience" in docs.cover_letter
    assert "quantum AI expertise" not in docs.cover_letter
    # Other docs unchanged
    assert docs.resume == "old resume"
    assert docs.interview_guide == "old ig"
    # _plan_edits called exactly once (only cover_letter had issues)
    assert agent._plan_edits.call_count == 1


def test_repair_unified_returns_repair_pass_result(career_profile, voice_profile, job_description):
    """repair_unified should return a RepairPassResult with the applied edits."""
    agent = RepairAgent()

    docs = DocumentSet(
        cover_letter="I am a passionate innovator.",
        resume="old resume",
        interview_guide="old ig",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["passionate innovator"],
    )
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=doc_pass,
        interview_guide=doc_pass,
    )
    context = _make_context()

    agent._plan_edits = MagicMock(return_value=[
        {"find": "passionate innovator", "replace": "experienced engineer", "reason": "truthfulness"},
    ])

    result = agent.repair_unified(
        docs, truth, None, None,
        career_profile, voice_profile, job_description, context,
    )

    assert isinstance(result, RepairPassResult)
    assert "cover_letter" in result.edits
    assert len(result.edits["cover_letter"]) == 1
    assert result.edits["cover_letter"][0].find == "passionate innovator"
    assert result.edits["cover_letter"][0].replace == "experienced engineer"
    assert result.edits["cover_letter"][0].reason == "truthfulness"


def test_repair_unified_skips_passing_docs(career_profile, voice_profile, job_description):
    """When all reviewers pass for a doc, it should not be repaired."""
    agent = RepairAgent()
    agent._plan_edits = MagicMock(return_value=[])

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=True,
        cover_letter=doc_pass, resume=doc_pass, interview_guide=doc_pass,
    )
    voice_review = VoiceReviewResult(
        overall_match="strong",
        cover_letter_match="strong",
        resume_match="strong",
        interview_guide_match="strong",
        cover_letter_assessment="Good",
        resume_assessment="Good",
        interview_guide_assessment="Good",
    )
    ai_review = AIDetectionResult(risk_level="low")
    context = _make_context()

    agent.repair_unified(
        docs, truth, voice_review, ai_review,
        career_profile, voice_profile, job_description, context,
    )

    assert agent._plan_edits.call_count == 0
    assert docs.cover_letter == "cl"
    assert docs.resume == "r"
    assert docs.interview_guide == "ig"


def test_repair_unified_handles_none_reviews(career_profile, voice_profile, job_description):
    """repair_unified should handle None review results gracefully."""
    agent = RepairAgent()
    agent._plan_edits = MagicMock(return_value=[])

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    context = _make_context()

    agent.repair_unified(
        docs, None, None, None,
        career_profile, voice_profile, job_description, context,
    )

    assert agent._plan_edits.call_count == 0


def test_repair_unified_combines_all_findings(career_profile, voice_profile, job_description):
    """Review findings from all three reviewers should be combined in the repair call."""
    agent = RepairAgent()

    docs = DocumentSet(
        cover_letter="I am a passionate innovator with quantum AI.",
        resume="r",
        interview_guide="ig",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["quantum AI"],
    )
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
    )
    voice_review = VoiceReviewResult(
        overall_match="weak",
        cover_letter_match="weak",
        resume_match="strong",
        interview_guide_match="strong",
        cover_letter_assessment="Off-voice",
        resume_assessment="Good",
        interview_guide_assessment="Good",
        cover_letter_issues=["opener too formal"],
    )
    ai_review = AIDetectionResult(
        risk_level="high",
        cover_letter_flags=["passionate innovator"],
        resume_flags=[],
        interview_guide_flags=[],
    )
    context = _make_context()

    # Capture the user_msg sent to _plan_edits
    agent._plan_edits = MagicMock(return_value=[
        {"find": "passionate innovator with quantum AI", "replace": "software engineer with backend experience", "reason": "combined"},
    ])

    agent.repair_unified(
        docs, truth, voice_review, ai_review,
        career_profile, voice_profile, job_description, context,
    )

    # Verify _plan_edits was called and we can check the user_msg
    assert agent._plan_edits.call_count == 1
    # The user message is the second arg (index 1) of the call
    call_args = agent._plan_edits.call_args
    user_msg = call_args[0][1]  # positional arg 1
    assert "quantum AI" in user_msg  # truthfulness finding
    assert "opener too formal" in user_msg  # voice finding
    assert "passionate innovator" in user_msg  # AI detection finding


def test_repair_plan_edits_parses_json_array():
    """_plan_edits should parse a JSON array response from the LLM."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = json.dumps([
        {"find": "old text", "replace": "new text", "reason": "fix"},
    ])
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits = agent._plan_edits("system", "user")

    assert len(edits) == 1
    assert edits[0]["find"] == "old text"
    assert edits[0]["replace"] == "new text"


def test_repair_plan_edits_handles_wrapped_object():
    """_plan_edits should handle LLM response wrapped in {edits: [...]}."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "edits": [{"find": "a", "replace": "b", "reason": "fix"}]
    })
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits = agent._plan_edits("system", "user")

    assert len(edits) == 1
    assert edits[0]["find"] == "a"


def test_repair_plan_edits_handles_empty_response():
    """_plan_edits should return empty list on empty LLM response."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = ""
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits = agent._plan_edits("system", "user")

    assert edits == []


def test_repair_build_review_findings_truthfulness():
    """_build_review_findings should format truthfulness issues."""
    agent = RepairAgent()
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["quantum AI expertise"],
        evidence_examples=["Led backend migration"],
    )
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
    )

    findings = agent._build_review_findings("cover_letter", truth, None, None, None)

    assert "quantum AI expertise" in findings
    assert "Led backend migration" in findings
    assert "TRUTHFULNESS" in findings


def test_repair_build_review_findings_empty_when_passing():
    """_build_review_findings should return empty string when all checks pass."""
    agent = RepairAgent()
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=True,
        cover_letter=doc_pass, resume=doc_pass, interview_guide=doc_pass,
    )
    voice_review = VoiceReviewResult(
        overall_match="strong",
        cover_letter_match="strong",
        resume_match="strong",
        interview_guide_match="strong",
        cover_letter_assessment="Good",
        resume_assessment="Good",
        interview_guide_assessment="Good",
    )
    ai_review = AIDetectionResult(risk_level="low")

    findings = agent._build_review_findings("cover_letter", truth, voice_review, ai_review, None)

    assert findings == ""
