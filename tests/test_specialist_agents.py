"""Tests for bounded specialist agents."""

from unittest.mock import MagicMock, patch

import pytest

from resume_refinery.models import (
    AIDetectionResult,
    DocumentSet,
    DocumentTruthResult,
    DraftingContext,
    EvidencePack,
    JobRequirement,
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


def test_evidence_agent_identifies_gaps(career_profile):
    """When the job requires something not in the career profile, it should appear in gaps."""
    from resume_refinery.models import JobDescription
    job = JobDescription(
        raw_content="Required: Rust experience, quantum computing, blockchain architecture.",
        title="Quantum Engineer",
        company="QuantumCo",
    )
    agent = EvidenceAgent()
    pack = agent.build_evidence_pack(career_profile, job)

    # These niche requirements should not match the career profile
    assert len(pack.gaps) > 0


def test_evidence_agent_fallback_requirements():
    """When no requirement keywords are found, fallback to first lines."""
    from resume_refinery.models import CareerProfile, JobDescription
    job = JobDescription(
        raw_content="# Data Scientist\nCompany: BigCo\n\nBuild models.\nImprove metrics.",
        title="Data Scientist",
        company="BigCo",
    )
    career = CareerProfile(raw_content="# Someone\nDoes stuff.")
    agent = EvidenceAgent()
    pack = agent.build_evidence_pack(career, job)

    # Should still produce some requirements from fallback
    assert pack.job_requirements


def test_evidence_agent_source_summary(career_profile, job_description):
    agent = EvidenceAgent()
    pack = agent.build_evidence_pack(career_profile, job_description)
    assert pack.source_summary  # Non-empty


def test_evidence_agent_limits_requirements():
    """Should cap at 10 requirements."""
    from resume_refinery.models import CareerProfile, JobDescription
    lines = "\n".join(f"- Required: skill_{i} experience" for i in range(20))
    job = JobDescription(raw_content=f"# Job\n{lines}", title="Job", company="Co")
    career = CareerProfile(raw_content="# Name\nDoes things.")
    agent = EvidenceAgent()
    pack = agent.build_evidence_pack(career, job)
    assert len(pack.job_requirements) <= 10


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
    def review_truthfulness(self, docs, career):
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


def test_verification_agent_review_all(document_set, career_profile, voice_profile):
    agent = VerificationAgent(reviewer=FakeReviewer())
    bundle = agent.review_all(document_set, career_profile, voice_profile)

    assert bundle.truthfulness is not None
    assert bundle.voice is not None
    assert bundle.ai_detection is not None
    assert bundle.truthfulness.all_supported is True


def test_verification_agent_review_truthfulness(document_set, career_profile):
    agent = VerificationAgent(reviewer=FakeReviewer())
    result = agent.review_truthfulness(document_set, career_profile)
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
# RepairAgent
# ---------------------------------------------------------------------------


def test_repair_agent_repair_documents(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "repaired content"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="old cl", resume="old resume", interview_guide="old ig")
    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim A"])
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=doc_pass,
        interview_guide=doc_fail,
    )
    context = _make_context()

    agent.repair_documents(docs, truth, career_profile, voice_profile, job_description, context)

    # cover_letter and interview_guide should be repaired; resume should stay
    assert docs.cover_letter == "repaired content"
    assert docs.resume == "old resume"
    assert docs.interview_guide == "repaired content"
    assert mock_generator.generate_document.call_count == 2


def test_repair_agent_repair_voice(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "voice-fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    voice_review = VoiceReviewResult(
        overall_match="weak",
        cover_letter_assessment="Off-voice",
        resume_assessment="Off-voice",
        interview_guide_assessment="Off-voice",
        specific_issues=["too formal"],
        suggestions=["Use contractions"],
    )
    context = _make_context()

    agent.repair_voice(docs, voice_review, career_profile, voice_profile, job_description, context)

    assert docs.cover_letter == "voice-fixed"
    assert docs.resume == "voice-fixed"
    assert docs.interview_guide == "voice-fixed"
    assert mock_generator.generate_document.call_count == 3

    # Verify feedback includes voice issues
    feedback_arg = mock_generator.generate_document.call_args.kwargs["feedback"]
    assert "too formal" in feedback_arg


def test_repair_agent_repair_ai_detection(career_profile, voice_profile, job_description):
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "ai-fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    ai_review = AIDetectionResult(
        risk_level="high",
        cover_letter_flags=["passionate about innovation"],
        resume_flags=[],
        interview_guide_flags=["demonstrated track record"],
        suggestions=["Be specific"],
    )
    context = _make_context()

    agent.repair_ai_detection(docs, ai_review, career_profile, voice_profile, job_description, context)

    # Only cover_letter and interview_guide have flags → only those are repaired
    assert docs.cover_letter == "ai-fixed"
    assert docs.resume == "r"  # No flags → not repaired
    assert docs.interview_guide == "ai-fixed"
    assert mock_generator.generate_document.call_count == 2


def test_repair_agent_repair_documents_no_failures(career_profile, voice_profile, job_description):
    """When all docs pass strict truth, no repairs should happen."""
    mock_generator = MagicMock()
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=True,
        cover_letter=doc_pass, resume=doc_pass, interview_guide=doc_pass,
    )
    context = _make_context()

    agent.repair_documents(docs, truth, career_profile, voice_profile, job_description, context)

    assert mock_generator.generate_document.call_count == 0
    assert docs.cover_letter == "cl"  # Unchanged


def test_repair_agent_repair_ai_no_flags(career_profile, voice_profile, job_description):
    """When no documents have AI flags, no repairs should happen."""
    mock_generator = MagicMock()
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    ai_review = AIDetectionResult(
        risk_level="low",
        cover_letter_flags=[],
        resume_flags=[],
        interview_guide_flags=[],
    )
    context = _make_context()

    agent.repair_ai_detection(docs, ai_review, career_profile, voice_profile, job_description, context)

    assert mock_generator.generate_document.call_count == 0


def test_repair_voice_skips_strong_docs(career_profile, voice_profile, job_description):
    """When a doc's per-doc match is 'strong', repair_voice should skip it."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "voice-fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    voice_review = VoiceReviewResult(
        overall_match="moderate",
        cover_letter_match="strong",      # should be skipped
        resume_match="weak",              # should be repaired
        interview_guide_match="strong",   # should be skipped
        cover_letter_assessment="On-voice",
        resume_assessment="Off-voice",
        interview_guide_assessment="On-voice",
        specific_issues=["too formal"],
        suggestions=["Use contractions"],
    )
    context = _make_context()

    agent.repair_voice(docs, voice_review, career_profile, voice_profile, job_description, context)

    # Only resume regenerated
    assert docs.cover_letter == "cl"  # unchanged (strong)
    assert docs.resume == "voice-fixed"
    assert docs.interview_guide == "ig"  # unchanged (strong)
    assert mock_generator.generate_document.call_count == 1


def test_feedback_for_doc_empty_claims_but_fails(career_profile, voice_profile, job_description):
    """When pass_strict=False but unsupported_claims is empty, feedback should still be specific."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "repaired"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=[], evidence_examples=["Led migration project"])
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
    )

    feedback = agent._feedback_for_doc("cover_letter", truth, None)

    # Should contain specific guidance even without unsupported_claims
    assert "truthfulness check failed" in feedback.lower()
    assert "Led migration project" in feedback  # evidence_examples surfaced
    assert "Rewrite strictly" in feedback


def test_feedback_for_doc_with_claims(career_profile, voice_profile, job_description):
    """When unsupported_claims are present, they should appear in feedback."""
    mock_generator = MagicMock()
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["Increased revenue by 500%"])
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
    )

    feedback = agent._feedback_for_doc("cover_letter", truth, None)

    assert "Increased revenue by 500%" in feedback
    assert "truthfulness check failed" not in feedback.lower()  # uses claims path, not empty-claims path


def test_feedback_includes_truth_suggestions(career_profile, voice_profile, job_description):
    """Top-level TruthfulnessResult.suggestions should be included in repair feedback."""
    mock_generator = MagicMock()
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["Led 50-person team"])
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
        suggestions=["Stick to verifiable metrics", "Avoid leadership claims without evidence"],
    )

    feedback = agent._feedback_for_doc("cover_letter", truth, None)

    assert "Stick to verifiable metrics" in feedback
    assert "Avoid leadership claims without evidence" in feedback


# ---------------------------------------------------------------------------
# LLM-powered EvidenceAgent tests
# ---------------------------------------------------------------------------

import json


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
# RepairAgent dedup and previous_suggestions
# ---------------------------------------------------------------------------


def test_repair_agent_deduplicate():
    """_deduplicate should remove suggestions already in previous list."""
    agent = RepairAgent()
    current = ["Fix claim A", "Rewrite intro", "Add metrics"]
    previous = ["Fix claim A", "add metrics"]  # case-insensitive dedup

    result = agent._deduplicate(current, previous)

    assert result == ["Rewrite intro"]


def test_repair_agent_deduplicate_no_previous():
    """With no previous suggestions, all current should be returned."""
    agent = RepairAgent()
    assert agent._deduplicate(["A", "B"], None) == ["A", "B"]
    assert agent._deduplicate(["A", "B"], []) == ["A", "B"]


def test_repair_documents_accepts_previous_suggestions(career_profile, voice_profile, job_description):
    """repair_documents should accept and pass previous_suggestions without error."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "repaired"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="old", resume="old", interview_guide="old")
    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
        suggestions=["Fix claim X", "Fix claim Y"],
    )
    context = _make_context()

    agent.repair_documents(
        docs, truth, career_profile, voice_profile, job_description, context,
        previous_suggestions=["Fix claim X"],
    )

    feedback = mock_generator.generate_document.call_args.kwargs["feedback"]
    # "Fix claim X" was already attempted — should appear in "previously attempted"
    assert "previously attempted" in feedback.lower()
    # "Fix claim Y" is new — should appear in suggestions
    assert "Fix claim Y" in feedback


def test_repair_voice_uses_per_doc_issues(career_profile, voice_profile, job_description):
    """repair_voice should use per-doc issues when available."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    voice_review = VoiceReviewResult(
        overall_match="weak",
        cover_letter_match="weak",
        resume_match="strong",
        interview_guide_match="strong",
        cover_letter_assessment="Off-voice",
        resume_assessment="Good",
        interview_guide_assessment="Good",
        specific_issues=["global issue"],
        suggestions=["global suggestion"],
        cover_letter_issues=["opener too formal"],
        cover_letter_suggestions=["Use contractions in opener"],
    )
    context = _make_context()

    agent.repair_voice(docs, voice_review, career_profile, voice_profile, job_description, context)

    # Only cover_letter should be repaired (others are "strong")
    assert docs.cover_letter == "fixed"
    assert docs.resume == "r"
    feedback = mock_generator.generate_document.call_args.kwargs["feedback"]
    # Should use per-doc issues, not global
    assert "opener too formal" in feedback
    assert "Use contractions in opener" in feedback


# ---------------------------------------------------------------------------
# RepairAgent.repair_unified
# ---------------------------------------------------------------------------


def test_repair_unified_combines_all_feedback(career_profile, voice_profile, job_description):
    """repair_unified should send combined feedback from all three reviewers in one call per doc."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "unified-fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["Invented quantum AI"])
    doc_pass = DocumentTruthResult(pass_strict=True)
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=doc_pass,
        interview_guide=doc_pass,
        suggestions=["Remove unverifiable claims"],
    )
    voice_review = VoiceReviewResult(
        overall_match="weak",
        cover_letter_match="weak",
        resume_match="strong",
        interview_guide_match="moderate",
        cover_letter_assessment="Off-voice",
        resume_assessment="Good",
        interview_guide_assessment="Slightly formal",
        specific_issues=["too formal"],
        suggestions=["Use contractions"],
        cover_letter_issues=["opener too formal"],
        cover_letter_suggestions=["Use a concrete hook"],
    )
    ai_review = AIDetectionResult(
        risk_level="high",
        cover_letter_flags=["passionate about innovation"],
        resume_flags=[],
        interview_guide_flags=[],
        suggestions=["Be specific"],
    )
    context = _make_context()

    agent.repair_unified(
        docs, truth, voice_review, ai_review,
        career_profile, voice_profile, job_description, context,
    )

    # cover_letter has truth + voice + AI issues → repaired
    assert docs.cover_letter == "unified-fixed"
    # resume passes all checks → not repaired
    assert docs.resume == "r"
    # interview_guide: voice + AI checks are skipped (personal prep), truth passes → not repaired
    assert docs.interview_guide == "ig"

    # Only cover_letter was repaired (1 call)
    assert mock_generator.generate_document.call_count == 1
    first_call_feedback = mock_generator.generate_document.call_args_list[0].kwargs["feedback"]
    assert "TRUTHFULNESS" in first_call_feedback
    assert "Invented quantum AI" in first_call_feedback
    assert "VOICE" in first_call_feedback
    assert "opener too formal" in first_call_feedback
    assert "AI DETECTION" in first_call_feedback
    assert "passionate about innovation" in first_call_feedback


def test_repair_unified_skips_passing_docs(career_profile, voice_profile, job_description):
    """When all reviewers pass for a doc, it should not be repaired."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

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

    assert mock_generator.generate_document.call_count == 0
    assert docs.cover_letter == "cl"
    assert docs.resume == "r"
    assert docs.interview_guide == "ig"


def test_repair_unified_handles_none_reviews(career_profile, voice_profile, job_description):
    """repair_unified should handle None review results gracefully."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    context = _make_context()

    # All None → no issues → no repairs
    agent.repair_unified(
        docs, None, None, None,
        career_profile, voice_profile, job_description, context,
    )

    assert mock_generator.generate_document.call_count == 0


def test_repair_unified_includes_previous_suggestions(career_profile, voice_profile, job_description):
    """Previously attempted suggestions should appear in the unified feedback."""
    mock_generator = MagicMock()
    mock_generator.generate_document.return_value = "fixed"
    drafting = DraftingAgent(generator=mock_generator)
    agent = RepairAgent(drafting_agent=drafting)

    docs = DocumentSet(cover_letter="cl", resume="r", interview_guide="ig")
    doc_fail = DocumentTruthResult(pass_strict=False, unsupported_claims=["claim"])
    truth = TruthfulnessResult(
        all_supported=False,
        cover_letter=doc_fail,
        resume=DocumentTruthResult(pass_strict=True),
        interview_guide=DocumentTruthResult(pass_strict=True),
        suggestions=["Fix claim X", "Fix claim Y"],
    )
    context = _make_context()

    agent.repair_unified(
        docs, truth, None, None,
        career_profile, voice_profile, job_description, context,
        previous_suggestions=["Fix claim X"],
    )

    feedback = mock_generator.generate_document.call_args.kwargs["feedback"]
    assert "previously attempted" in feedback.lower()
    assert "Fix claim Y" in feedback
