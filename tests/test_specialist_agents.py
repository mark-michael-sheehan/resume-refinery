"""Tests for bounded specialist agents."""

import json
from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    ATSKeywordResult,
    CandidacyNarrative,
    CareerProfile,
    CoverageGap,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    GrammarResult,
    JobDescription,
    NarrativeCoherenceResult,
    NarrativeCoverageResult,
    NarrativePillar,
    RepairPassResult,
    ReviewBundle,
    TruthfulnessResult,
    VoiceReviewResult,
    VoiceStyleGuide,
)
from resume_refinery.specialist_agents import (
    DraftingAgent,
    NarrativeAgent,
    NarrativeCoverageAgent,
    RepairAgent,
    VerificationAgent,
    VoiceAgent,
)


# ---------------------------------------------------------------------------
# NarrativeAgent
# ---------------------------------------------------------------------------


def _make_llm_resp(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def test_narrative_agent_builds_narrative(career_profile, job_description):
    """NarrativeAgent should produce a CandidacyNarrative from LLM JSON."""
    narrative_json = json.dumps({
        "thesis": "Strong distributed systems background.",
        "pillars": [
            {
                "theme": "Backend",
                "argument": "Led migrations",
                "career_evidence": ["Cut deploy time 60%"],
            }
        ],
        "gap_framing": ["No Rust experience"],
        "raw_narrative": "Full narrative text.",
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_llm_resp(narrative_json)

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job_description)

    assert narrative.thesis == "Strong distributed systems background."
    assert len(narrative.pillars) == 1
    assert narrative.pillars[0].theme == "Backend"
    assert narrative.gap_framing == ["No Rust experience"]
    assert narrative.raw_narrative == "Full narrative text."


def test_narrative_agent_fallback_on_failure(career_profile, job_description):
    """When LLM fails, keyword fallback should still produce a narrative."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job_description)

    assert isinstance(narrative, CandidacyNarrative)
    assert narrative.thesis  # Should have a default thesis


def test_narrative_agent_limits_pillars(career_profile, job_description):
    """NarrativeAgent should cap pillars at 5."""
    pillars = [
        {"theme": f"Skill {i}", "argument": f"Arg {i}", "career_evidence": [f"ev{i}"]}
        for i in range(10)
    ]
    narrative_json = json.dumps({
        "thesis": "Strong",
        "pillars": pillars,
        "gap_framing": [],
        "raw_narrative": "",
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_llm_resp(narrative_json)

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job_description)

    assert len(narrative.pillars) <= 5


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
        narrative=CandidacyNarrative(
            thesis="Strong fit for the role.",
            pillars=[NarrativePillar(theme="Backend", argument="Led migrations", career_evidence=["Reduced costs"])],
            gap_framing=["Rust"],
            raw_narrative="Full narrative text.",
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

    assert docs.resume == "Generated."
    assert mock_generator.generate_document.call_count == 1


def test_drafting_agent_generate_document_with_feedback(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "Revised."

    agent = DraftingAgent(generator=mock_generator)
    result = agent.generate_document(
        "resume", career_profile, voice_profile, job_description, _make_context(),
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

    # The career profile passed to the generator should include narrative info
    enriched_career = mock_generator.generate_document.call_args.args[1]
    assert "Candidacy Narrative" in enriched_career.raw_content
    assert "Backend" in enriched_career.raw_content


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
    def review_truthfulness(self, docs, career, job, *, exemptions=None):
        doc = DocumentTruthResult(pass_strict=True)
        return TruthfulnessResult(
            all_supported=True,
            resume=doc,
        )

    def review_voice(self, docs, voice, *, exemptions=None):
        return VoiceReviewResult(
            overall_match="strong",
            resume_assessment="Good",
        )

    def review_ai_detection(self, docs, *, exemptions=None):
        return AIDetectionResult(risk_level="low")

    def review_ats_keyword(self, docs, job, career, *, exemptions=None):
        return ATSKeywordResult(alignment_score="strong")

    def review_grammar(self, docs, *, exemptions=None):
        return GrammarResult(clean=True)

    def review_narrative_coherence(self, docs, narrative, *, exemptions=None):
        return NarrativeCoherenceResult(alignment="strong")


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


def test_verification_agent_review_ats_keyword(document_set, career_profile, job_description):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_ats_keyword(document_set, job_description, career_profile)
    assert result.alignment_score == "strong"


def test_verification_agent_review_grammar(document_set):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_grammar(document_set)
    assert result.clean is True


def test_verification_agent_review_narrative_coherence(document_set, candidacy_narrative):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_narrative_coherence(document_set, candidacy_narrative)
    assert isinstance(result, NarrativeCoherenceResult)
    assert result.alignment == "strong"


# ---------------------------------------------------------------------------
# LLM-powered NarrativeAgent tests
# ---------------------------------------------------------------------------


def _make_llm_response(response_text: str):
    """Build an ollama chat response mock."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_response = MagicMock()
    mock_response.message = mock_message
    return mock_response


def test_narrative_agent_llm_builds_narrative(career_profile, job_description):
    """When LLM is available, narrative should come from the LLM."""
    narrative_json = json.dumps({
        "thesis": "Strong distributed systems background makes this candidate ideal.",
        "pillars": [
            {"theme": "Backend Engineering", "argument": "Led critical infrastructure projects", "career_evidence": ["Cut deploy time 60%"]},
            {"theme": "Technical Leadership", "argument": "Mentored junior engineers", "career_evidence": ["Led team of 5"]},
        ],
        "gap_framing": ["No explicit Rust experience — transferable from Go/Python"],
        "raw_narrative": "Full narrative for context.",
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_llm_response(narrative_json)

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job_description)

    assert narrative.thesis == "Strong distributed systems background makes this candidate ideal."
    assert len(narrative.pillars) == 2
    assert narrative.pillars[0].theme == "Backend Engineering"
    assert narrative.gap_framing == ["No explicit Rust experience — transferable from Go/Python"]
    assert narrative.raw_narrative == "Full narrative for context."


def test_narrative_agent_llm_gap_detection(career_profile):
    """When career profile lacks required skills, gap_framing should capture that."""
    from resume_refinery.models import JobDescription
    job = JobDescription(raw_content="Required: Quantum computing", title="QC", company="QCo")

    narrative_json = json.dumps({
        "thesis": "Partial fit.",
        "pillars": [],
        "gap_framing": ["Quantum computing — no direct experience, but strong physics background"],
        "raw_narrative": "",
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_llm_response(narrative_json)

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job)

    assert len(narrative.gap_framing) > 0
    assert "Quantum" in narrative.gap_framing[0]


def test_narrative_agent_falls_back_on_llm_failure(career_profile, job_description):
    """When LLM calls fail, keyword fallback should still produce a narrative."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")

    agent = NarrativeAgent(client=mock_client)
    narrative = agent.build_narrative(career_profile, job_description)

    assert isinstance(narrative, CandidacyNarrative)
    assert narrative.thesis  # Should have a default thesis
    # Fallback should identify some overlapping keywords as pillars
    assert narrative.pillars or narrative.gap_framing


# ---------------------------------------------------------------------------
# RepairAgent  (surgical find/replace edits)
# ---------------------------------------------------------------------------


def test_repair_unified_applies_surgical_edits(career_profile, voice_profile, job_description):
    """repair_unified should call _plan_edits and apply edits programmatically."""
    agent = RepairAgent()

    docs = DocumentSet(
        resume="I am a passionate innovator with quantum AI expertise.",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["quantum AI expertise"],
    )
    truth = TruthfulnessResult(
        all_supported=False,
        resume=doc_fail,
    )
    context = _make_context()

    # Mock _plan_edits to return a surgical edit
    agent._plan_edits = MagicMock(return_value=(
        [{"find": "quantum AI expertise", "replace": "backend migration experience", "reason": "truthfulness"}],
        {},
    ))

    agent.repair_unified(
        docs, truth, None, None,
        career_profile, voice_profile, job_description, context,
    )

    # Resume was repaired
    assert "backend migration experience" in docs.resume
    assert "quantum AI expertise" not in docs.resume
    # _plan_edits called exactly once (resume had issues)
    assert agent._plan_edits.call_count == 1


def test_repair_unified_returns_repair_pass_result(career_profile, voice_profile, job_description):
    """repair_unified should return a RepairPassResult with the applied edits."""
    agent = RepairAgent()

    docs = DocumentSet(
        resume="I am a passionate innovator.",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["passionate innovator"],
    )
    truth = TruthfulnessResult(
        all_supported=False,
        resume=doc_fail,
    )
    context = _make_context()

    agent._plan_edits = MagicMock(return_value=(
        [{"find": "passionate innovator", "replace": "experienced engineer", "reason": "truthfulness"}],
        {},
    ))

    result = agent.repair_unified(
        docs, truth, None, None,
        career_profile, voice_profile, job_description, context,
    )

    assert isinstance(result, RepairPassResult)
    assert "resume" in result.edits
    assert len(result.edits["resume"]) == 1
    assert result.edits["resume"][0].find == "passionate innovator"
    assert result.edits["resume"][0].replace == "experienced engineer"
    assert result.edits["resume"][0].reason == "truthfulness"


def test_repair_unified_skips_passing_docs(career_profile, voice_profile, job_description):
    """When all reviewers pass for a doc, it should not be repaired."""
    agent = RepairAgent()
    agent._plan_edits = MagicMock(return_value=([], {}))

    docs = DocumentSet(resume="r")
    context = _make_context()

    agent.repair_unified(
        docs, None, None, None,
        career_profile, voice_profile, job_description, context,
    )

    assert agent._plan_edits.call_count == 0


def test_repair_unified_combines_all_findings(career_profile, voice_profile, job_description):
    """Review findings from all reviewers should be combined in the repair call."""
    agent = RepairAgent()

    docs = DocumentSet(
        resume="I am a passionate innovator with quantum AI.",
    )
    doc_fail = DocumentTruthResult(
        pass_strict=False,
        unsupported_claims=["quantum AI"],
    )
    truth = TruthfulnessResult(
        all_supported=False,
        resume=doc_fail,
    )
    voice_review = VoiceReviewResult(
        overall_match="weak",
        resume_match="weak",
        resume_assessment="Off-voice",
        resume_issues=["opener too formal"],
    )
    ai_review = AIDetectionResult(
        risk_level="high",
        resume_flags=["passionate innovator"],
    )
    context = _make_context()

    # Capture the user_msg sent to _plan_edits
    agent._plan_edits = MagicMock(return_value=(
        [{"find": "passionate innovator with quantum AI", "replace": "software engineer with backend experience", "reason": "combined"}],
        {},
    ))

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


def test_repair_unified_populates_accepted_phrases(career_profile, voice_profile, job_description):
    """repair_unified should populate RepairPassResult accepted_* fields from _plan_edits."""
    agent = RepairAgent()

    docs = DocumentSet(
        resume="I am a Senior Software Engineer with Python experience.",
    )
    ai_review = AIDetectionResult(
        risk_level="medium",
        resume_flags=["I am a Senior Software Engineer"],
    )
    context = _make_context()

    # Repairer decides the AI flag is a false positive and accepts it
    agent._plan_edits = MagicMock(return_value=(
        [],
        {"accepted_claims": [], "accepted_ai_phrases": ["I am a Senior Software Engineer"], "accepted_voice_issues": [], "accepted_hm_issues": []},
    ))

    result = agent.repair_unified(
        docs, None, None, ai_review,
        career_profile, voice_profile, job_description, context,
    )

    assert isinstance(result, RepairPassResult)
    assert result.accepted_ai_phrases == ["I am a Senior Software Engineer"]
    assert result.accepted_claims == []
    assert result.accepted_voice_issues == []
    # No edits applied — doc unchanged
    assert "I am a Senior Software Engineer" in docs.resume


def test_repair_plan_edits_parses_json_object():
    """_plan_edits should parse the new JSON object schema and return (edits, acceptances)."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "edits": [{"find": "old text", "replace": "new text", "reason": "fix"}],
        "accepted_claims": [],
        "accepted_ai_phrases": [],
        "accepted_voice_issues": [],
        "accepted_hm_issues": [],
    })
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits, acceptances = agent._plan_edits("system", "user")

    assert len(edits) == 1
    assert edits[0]["find"] == "old text"
    assert edits[0]["replace"] == "new text"
    assert acceptances["accepted_claims"] == []
    assert acceptances["accepted_ai_phrases"] == []
    assert acceptances["accepted_voice_issues"] == []
    assert acceptances["accepted_hm_issues"] == []


def test_repair_plan_edits_returns_acceptances():
    """_plan_edits should return populated acceptance arrays from the LLM response."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = json.dumps({
        "edits": [{"find": "a", "replace": "b", "reason": "fix"}],
        "accepted_claims": ["unsupported claim"],
        "accepted_ai_phrases": [],
        "accepted_voice_issues": ["off voice phrase"],
        "accepted_hm_issues": [],
    })
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits, acceptances = agent._plan_edits("system", "user")

    assert len(edits) == 1
    assert edits[0]["find"] == "a"
    assert acceptances["accepted_claims"] == ["unsupported claim"]
    assert acceptances["accepted_voice_issues"] == ["off voice phrase"]


def test_repair_plan_edits_handles_empty_response():
    """_plan_edits should return ([], empty acceptances) on empty LLM response."""
    agent = RepairAgent()
    mock_response = MagicMock()
    mock_response.message.content = ""
    agent.client = MagicMock()
    agent.client.chat.return_value = mock_response

    edits, acceptances = agent._plan_edits("system", "user")

    assert edits == []
    assert acceptances == {"accepted_claims": [], "accepted_ai_phrases": [], "accepted_voice_issues": [], "accepted_hm_issues": [], "accepted_pruning_issues": [], "accepted_ats_issues": [], "accepted_grammar_issues": [], "accepted_narrative_issues": []}


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
        resume=doc_fail,
    )

    findings = agent._build_review_findings("resume", truth, None, None, None)

    assert "quantum AI expertise" in findings
    assert "Led backend migration" in findings
    assert "TRUTHFULNESS" in findings


def test_repair_build_review_findings_empty_when_passing():
    """_build_review_findings should return empty string when all checks pass."""
    agent = RepairAgent()
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=True,
        resume=doc_pass,
    )
    voice_review = VoiceReviewResult(
        overall_match="strong",
        resume_match="strong",
        resume_assessment="Good",
    )
    ai_review = AIDetectionResult(risk_level="low")

    findings = agent._build_review_findings("resume", truth, voice_review, ai_review, None)

    assert findings == ""


# ---------------------------------------------------------------------------
# NarrativeCoverageAgent
# ---------------------------------------------------------------------------


def _coverage_narrative():
    return CandidacyNarrative(
        thesis="Distributed systems expert.",
        pillars=[
            NarrativePillar(
                theme="Backend Migration",
                argument="Led migrations cutting deploy time.",
                career_evidence=["Cut deploy time 60%"],
            ),
            NarrativePillar(
                theme="Cost Optimisation",
                argument="Reduced infra spend by $180K.",
                career_evidence=["Reduced infra costs by $180K/year"],
            ),
        ],
        gap_framing=["No Rust experience"],
        raw_narrative="Full narrative.",
    )


def test_coverage_agent_llm_success(career_profile, job_description):
    """NarrativeCoverageAgent should parse LLM JSON into a NarrativeCoverageResult."""
    coverage_json = json.dumps({
        "coverage_summary": "1 of 2 pillars covered.",
        "pillars_covered": 1,
        "pillars_total": 2,
        "gaps": [
            {
                "pillar_theme": "Cost Optimisation",
                "career_evidence": ["Reduced infra costs by $180K/year"],
                "suggested_content": "- Reduced infrastructure costs by $180K/year through resource optimisation",
                "anchor_section": "Experience",
                "confidence": "high",
            }
        ],
    })
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_llm_resp(coverage_json)

    agent = NarrativeCoverageAgent(client=mock_client)
    docs = DocumentSet(resume="# Jordan Lee\n\n## Experience\n\n- Led backend migration")
    result = agent.analyze_coverage(_coverage_narrative(), career_profile, docs, job_description)

    assert result.pillars_covered == 1
    assert result.pillars_total == 2
    assert len(result.gaps) == 1
    assert result.gaps[0].pillar_theme == "Cost Optimisation"
    assert result.gaps[0].confidence == "high"


def test_coverage_agent_fallback_on_failure(career_profile, job_description):
    """When LLM fails, keyword fallback should still produce a result."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")

    agent = NarrativeCoverageAgent(client=mock_client)
    docs = DocumentSet(resume="# Jordan Lee\n\n## Experience\n\n- Led backend migration")
    result = agent.analyze_coverage(_coverage_narrative(), career_profile, docs, job_description)

    assert isinstance(result, NarrativeCoverageResult)
    assert result.pillars_total == 2


def test_coverage_agent_no_pillars():
    """Empty pillars should return full coverage."""
    narrative = CandidacyNarrative(thesis="Expert.", pillars=[])
    agent = NarrativeCoverageAgent(client=MagicMock())
    career = CareerProfile(raw_content="Career text.")
    docs = DocumentSet(resume="Resume text.")
    job = JobDescription(raw_content="Job text.")

    result = agent.analyze_coverage(narrative, career, docs, job)

    assert result.pillars_covered == 0
    assert result.pillars_total == 0
    assert result.gaps == []


def test_coverage_agent_no_resume():
    """Missing resume should return full coverage."""
    narrative = _coverage_narrative()
    agent = NarrativeCoverageAgent(client=MagicMock())
    career = CareerProfile(raw_content="Career text.")
    docs = DocumentSet(resume=None)
    job = JobDescription(raw_content="Job text.")

    result = agent.analyze_coverage(narrative, career, docs, job)

    assert result.pillars_covered == 2
    assert result.gaps == []


def test_coverage_agent_apply_suggestions():
    """apply_suggestions should insert content at anchor sections."""
    agent = NarrativeCoverageAgent(client=MagicMock())
    docs = DocumentSet(resume="# Jordan Lee\n\n## Experience\n\n- Point 1\n\n## Skills\n\n- Python")
    result = NarrativeCoverageResult(
        gaps=[
            CoverageGap(
                pillar_theme="Cost Optimisation",
                career_evidence=["$180K savings"],
                suggested_content="- Reduced infrastructure costs by $180K/year",
                anchor_section="Experience",
                confidence="high",
            ),
        ],
        pillars_covered=1,
        pillars_total=2,
    )
    agent.apply_suggestions(docs, result)

    assert "Reduced infrastructure costs by $180K/year" in docs.resume
    assert docs.resume.index("Reduced infrastructure") < docs.resume.index("## Skills")


def test_coverage_agent_apply_suggestions_no_anchor():
    """apply_suggestions should skip gaps without an anchor_section."""
    agent = NarrativeCoverageAgent(client=MagicMock())
    original = "# Resume\n\n## Experience\n\n- Point 1"
    docs = DocumentSet(resume=original)
    result = NarrativeCoverageResult(
        gaps=[CoverageGap(pillar_theme="X", suggested_content="New bullet", anchor_section="", confidence="low")],
        pillars_covered=0,
        pillars_total=1,
    )
    agent.apply_suggestions(docs, result)

    assert docs.resume == original


def test_coverage_agent_fallback_themes_in_resume(career_profile, job_description):
    """Fallback should count pillars whose theme words appear in the resume."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("fail")

    agent = NarrativeCoverageAgent(client=mock_client)
    # "migration" is in theme "Backend Migration"
    docs = DocumentSet(resume="Led backend migration cutting deploy time.")
    result = agent.analyze_coverage(_coverage_narrative(), career_profile, docs, job_description)

    assert result.pillars_covered >= 1
