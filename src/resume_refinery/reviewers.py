"""Review agents — voice-match and AI-detection checks on generated documents."""

from __future__ import annotations

import json
import logging
import os
import re

import ollama
from dotenv import load_dotenv

from .models import (
    AIDetectionResult,
    ATSKeywordIssue,
    ATSKeywordResult,
    DocumentSet,
    DocumentTruthResult,
    GrammarIssue,
    GrammarResult,
    HiringManagerImprovementItem,
    HiringManagerIssue,
    HiringManagerReview,
    JobDescription,
    RelevancePruningIssue,
    RelevancePruningResult,
    ReviewBundle,
    TruthfulnessResult,
    CareerProfile,
    VoiceProfile,
    VoiceReviewResult,
)
from .prompts import (
    AI_DETECTION_DOC_USER_TEMPLATE,
    AI_DETECTION_SYSTEM_PROMPT,
    ATS_KEYWORD_SYSTEM_PROMPT,
    ATS_KEYWORD_USER_TEMPLATE,
    GRAMMAR_DOC_USER_TEMPLATE,
    GRAMMAR_SYSTEM_PROMPT,
    HIRING_MANAGER_REVIEW_SYSTEM_PROMPT,
    HIRING_MANAGER_REVIEW_USER_TEMPLATE,
    RELEVANCE_PRUNING_DOC_USER_TEMPLATE,
    RELEVANCE_PRUNING_SYSTEM_PROMPT,
    TRUTHFULNESS_DOC_USER_TEMPLATE,
    TRUTHFULNESS_SYSTEM_PROMPT,
    VOICE_REVIEW_DOC_USER_TEMPLATE,
    VOICE_REVIEW_SYSTEM_PROMPT,
)

load_dotenv()

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("RESUME_REFINERY_REVIEW_MODEL", "qwen3.5:9b")
MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_REVIEW_MAX_TOKENS", "4096"))
NUM_CTX = int(os.environ.get("RESUME_REFINERY_NUM_CTX", "16384"))

_MATCH_RANK = {"strong": 3, "moderate": 2, "weak": 1}
_RISK_RANK = {"low": 1, "medium": 2, "high": 3}


def _normalize_llm_json(raw: str) -> str:
    """Return *raw* as valid JSON, repairing common LLM mistakes.

    Handles: unescaped quotes inside strings, Python-style ``None`` / ``True``
    / ``False``, trailing commas, single-quoted strings, etc.  Falls back to
    the ``json_repair`` library which is purpose-built for LLM output.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # json_repair handles unescaped inner quotes, trailing commas, etc.
    from json_repair import repair_json
    repaired = repair_json(raw, return_objects=False)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass
    # Last resort: Python literal syntax (None/True/False → null/true/false)
    try:
        import ast as _ast
        return json.dumps(_ast.literal_eval(raw))
    except Exception:
        return raw  # let the caller surface the original error


_GEN_MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")


def _exemption_section(items: list[str] | None, label: str) -> str:
    """Build a user-message section listing previously accepted items."""
    if not items:
        return ""
    bullet_list = "\n".join(f'- "{item}"' for item in items)
    return (
        f"\n\n## Previously Accepted (DO NOT flag these)\n"
        f"The following {label} were reviewed in prior passes and accepted as "
        f"legitimate — they are not issues. Do not flag them:\n{bullet_list}\n"
    )


class DocumentReviewer:
    """Runs voice-match and AI-detection reviews on a DocumentSet."""

    def __init__(self, api_key: str | None = None) -> None:  # api_key unused; retained for call-site compatibility
        self.client = ollama.Client(host=BASE_URL)
        if MODEL == _GEN_MODEL:
            logging.warning(
                "Review model (%s) is the same as generation model. "
                "Set RESUME_REFINERY_REVIEW_MODEL to a different model for more "
                "objective reviews.",
                MODEL,
            )

    def review_all(self, docs: DocumentSet, voice: VoiceProfile) -> ReviewBundle:
        """Run both review passes and return a ReviewBundle."""
        return ReviewBundle(
            voice=self.review_voice(docs, voice),
            ai_detection=self.review_ai_detection(docs),
        )

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile, job: JobDescription, *, exemptions: list[str] | None = None) -> TruthfulnessResult:
        """Verify resume claims are explicitly supported by the career profile."""
        exempt_block = _exemption_section(exemptions, "claims")

        if not docs.resume:
            resume_result = DocumentTruthResult(pass_strict=True)
        else:
            user_msg = TRUTHFULNESS_DOC_USER_TEMPLATE.format(
                career_profile=career.raw_content,
                job_description=job.raw_content,
                doc_type="Resume",
                doc_content=docs.resume,
            ) + exempt_block
            raw = self._call(TRUTHFULNESS_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)
            resume_result = DocumentTruthResult(
                pass_strict=True if data.get("pass_strict") is None else data["pass_strict"],
                unsupported_claims=data.get("unsupported_claims") or [],
                evidence_examples=data.get("evidence_examples") or [],
            )

        return TruthfulnessResult(
            all_supported=resume_result.pass_strict,
            resume=resume_result,
        )

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile, *, exemptions: list[str] | None = None) -> VoiceReviewResult:
        """Check how well the resume matches the user's voice profile."""
        exempt_block = _exemption_section(exemptions, "voice issues")

        if not docs.resume:
            return VoiceReviewResult(
                overall_match="strong",
                resume_match="strong",
                resume_assessment="(not generated)",
            )

        user_msg = VOICE_REVIEW_DOC_USER_TEMPLATE.format(
            voice_profile=voice.raw_content,
            doc_type="Resume",
            doc_content=docs.resume,
        ) + exempt_block
        raw = self._call(VOICE_REVIEW_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)

        resume_match = data.get("overall_match") or "moderate"
        resume_issues = data.get("issues") or []

        return VoiceReviewResult(
            overall_match=resume_match,
            resume_match=resume_match,
            resume_assessment=data.get("assessment") or "",
            specific_issues=resume_issues,
            resume_issues=resume_issues,
        )

    def review_ai_detection(self, docs: DocumentSet, *, exemptions: list[str] | None = None) -> AIDetectionResult:
        """Identify AI-sounding or generic content in the resume."""
        exempt_block = _exemption_section(exemptions, "phrases")

        if not docs.resume:
            return AIDetectionResult(risk_level="low")

        user_msg = AI_DETECTION_DOC_USER_TEMPLATE.format(
            doc_type="Resume",
            doc_content=docs.resume,
        ) + exempt_block
        raw = self._call(AI_DETECTION_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)

        raw_flags = data.get("flags") or []
        seen: set[str] = set()
        deduped: list[str] = []
        for f in raw_flags:
            if f not in seen:
                seen.add(f)
                deduped.append(f)

        risk_level = data.get("risk_level") or "low"
        if risk_level not in ("low", "medium", "high"):
            risk_level = "low"

        return AIDetectionResult(
            risk_level=risk_level,
            resume_flags=deduped,
        )

    def review_hiring_manager(
        self, docs: DocumentSet, job: JobDescription, *, exemptions: list[str] | None = None,
    ) -> HiringManagerReview:
        """Simulate a hiring-manager review of the resume."""
        user_msg = HIRING_MANAGER_REVIEW_USER_TEMPLATE.format(
            job_description=job.raw_content,
            resume=docs.resume or "(not provided)",
        ) + _exemption_section(exemptions, "phrases")
        raw = self._call(HIRING_MANAGER_REVIEW_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)

        likelihood = data.get("advance_likelihood")
        if likelihood is None:
            likelihood = 50
        if not isinstance(likelihood, int):
            try:
                likelihood = int(likelihood)
            except (TypeError, ValueError):
                likelihood = 50
        likelihood = max(0, min(100, likelihood))

        improvements = []
        for item in data.get("improvements") or []:
            if isinstance(item, dict) and "suggestion" in item:
                area = item.get("area", "resume")
                if area != "resume":
                    area = "resume"
                impact = item.get("impact", "medium")
                if impact not in ("high", "medium", "low"):
                    impact = "medium"
                improvements.append(HiringManagerImprovementItem(
                    area=area,
                    suggestion=item["suggestion"],
                    impact=impact,
                ))

        # Parse per-document issues (verbatim-quote-based findings for repair)
        resume_issues: list[HiringManagerIssue] = []
        for item in data.get("issues") or []:
            if not isinstance(item, dict) or "phrase" not in item:
                continue
            impact = item.get("impact", "medium")
            if impact not in ("high", "medium", "low"):
                impact = "medium"
            resume_issues.append(HiringManagerIssue(
                document="resume",
                phrase=item["phrase"],
                issue=item.get("issue") or "",
                suggestion=item.get("suggestion") or "",
                impact=impact,
            ))

        return HiringManagerReview(
            advance_likelihood=likelihood,
            summary=data.get("summary") or "",
            strengths=data.get("strengths") or [],
            concerns=data.get("concerns") or [],
            improvements=improvements,
            resume_issues=resume_issues,
        )

    def review_relevance_pruning(
        self, docs: DocumentSet, job: JobDescription, *, exemptions: list[str] | None = None,
    ) -> RelevancePruningResult:
        """Identify content that can be removed without weakening the resume."""
        exempt_block = _exemption_section(exemptions, "phrases")
        resume_issues: list[RelevancePruningIssue] = []
        density = "balanced"

        if docs.resume:
            user_msg = RELEVANCE_PRUNING_DOC_USER_TEMPLATE.format(
                job_description=job.raw_content,
                doc_type="Resume",
                doc_content=docs.resume,
            ) + exempt_block
            raw = self._call(RELEVANCE_PRUNING_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)

            density = data.get("overall_density") or "balanced"
            if density not in ("lean", "balanced", "bloated"):
                density = "balanced"

            for item in data.get("removal_candidates") or []:
                if not isinstance(item, dict) or "phrase" not in item:
                    continue
                category = item.get("category", "filler")
                if category not in ("redundant", "irrelevant", "filler", "low_impact", "space_waste"):
                    category = "filler"
                severity = item.get("severity", "medium")
                if severity not in ("high", "medium", "low"):
                    severity = "medium"
                resume_issues.append(RelevancePruningIssue(
                    document="resume",
                    phrase=item["phrase"],
                    reason=item.get("reason") or "",
                    category=category,
                    severity=severity,
                ))

        return RelevancePruningResult(
            overall_density=density,
            resume_issues=resume_issues,
        )

    def review_ats_keyword(
        self, docs: DocumentSet, job: JobDescription, career: CareerProfile, *, exemptions: list[str] | None = None,
    ) -> ATSKeywordResult:
        """Check resume for ATS keyword alignment against the job description."""
        if not docs.resume:
            return ATSKeywordResult(alignment_score="strong")

        user_msg = ATS_KEYWORD_USER_TEMPLATE.format(
            job_description=job.raw_content,
            career_profile=career.raw_content,
            resume=docs.resume,
        ) + _exemption_section(exemptions, "keywords")
        raw = self._call(ATS_KEYWORD_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)

        score = data.get("alignment_score") or "moderate"
        if score not in ("strong", "moderate", "weak"):
            score = "moderate"

        missing: list[ATSKeywordIssue] = []
        for item in data.get("missing_keywords") or []:
            if not isinstance(item, dict) or "keyword" not in item:
                continue
            priority = item.get("priority", "medium")
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            missing.append(ATSKeywordIssue(
                keyword=item["keyword"],
                issue_type="missing",
                section=item.get("section") or "",
                suggestion=item.get("suggestion") or "",
                priority=priority,
            ))

        stuffing: list[ATSKeywordIssue] = []
        for item in data.get("stuffing_keywords") or []:
            if not isinstance(item, dict) or "keyword" not in item:
                continue
            priority = item.get("priority", "medium")
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            stuffing.append(ATSKeywordIssue(
                keyword=item["keyword"],
                issue_type="stuffing",
                section=item.get("section") or "",
                suggestion=item.get("suggestion") or "",
                priority=priority,
            ))

        return ATSKeywordResult(
            alignment_score=score,
            missing_keywords=missing,
            stuffing_keywords=stuffing,
        )

    def review_grammar(self, docs: DocumentSet, *, exemptions: list[str] | None = None) -> GrammarResult:
        """Check the resume for grammar, tense, and mechanics errors."""
        exempt_block = _exemption_section(exemptions, "phrases")
        resume_issues: list[GrammarIssue] = []
        all_clean = True

        if docs.resume:
            user_msg = GRAMMAR_DOC_USER_TEMPLATE.format(
                doc_type="Resume",
                doc_content=docs.resume,
            ) + exempt_block
            raw = self._call(GRAMMAR_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)

            doc_clean = data.get("clean")
            if doc_clean is not None and not doc_clean:
                all_clean = False

            for item in data.get("issues") or []:
                if not isinstance(item, dict) or "phrase" not in item:
                    continue
                category = item.get("category", "grammar")
                if category not in ("grammar", "tense", "punctuation", "capitalization", "formatting"):
                    category = "grammar"
                severity = item.get("severity", "medium")
                if severity not in ("high", "medium", "low"):
                    severity = "medium"
                resume_issues.append(GrammarIssue(
                    document="resume",
                    phrase=item["phrase"],
                    issue=item.get("issue") or "",
                    suggestion=item.get("suggestion") or "",
                    category=category,
                    severity=severity,
                ))
                all_clean = False

        return GrammarResult(
            clean=all_clean,
            resume_issues=resume_issues,
        )

    def _call(self, system: str, user_msg: str, *, think: bool = False) -> str:
        """Make an Ollama API call and return the text response."""
        if think:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": "/no_think\n" + user_msg},
            ]
        # Cap num_predict when thinking is enabled to prevent unbounded
        # reasoning loops that make the call appear to hang.  Ollama
        # separates thinking from content, so format="json" still works.
        predict = MAX_TOKENS * 2 if think else MAX_TOKENS
        response = self.client.chat(
            model=MODEL,
            messages=messages,
            think=think,
            format="json",
            options={"num_ctx": NUM_CTX, "num_predict": predict, "temperature": 0},
        )
        raw = (response.message.content or "").strip()

        # Strip any residual <think>...</think> blocks as a defensive fallback
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

        # Repair Python-style literals that models sometimes emit (None→null, True→true, False→false)
        raw = _normalize_llm_json(raw)

        if not raw:
            # Log any thinking the model produced to aid debugging.
            thinking_text = getattr(response.message, "thinking", None) or ""
            logging.warning(
                "Ollama reviewer returned empty content. "
                "Thinking length: %d chars. Full content: %r",
                len(thinking_text),
                response.message.content,
            )
            raise ValueError(
                "Reviewer returned empty content — model may have run out of context tokens."
            )

        # Strip markdown fences if present (defensive fallback for non-think path)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        return raw

