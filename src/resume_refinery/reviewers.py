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
    ConsistencyIssue,
    ConsistencyResult,
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
    CONSISTENCY_SYSTEM_PROMPT,
    CONSISTENCY_USER_TEMPLATE,
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

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile, job: JobDescription) -> TruthfulnessResult:
        """Verify document claims are explicitly supported by the career profile.

        Each document is reviewed in its own call so the context window is never
        filled with all three documents at once.
        """
        doc_map = [
            ("Cover Letter", docs.cover_letter),
            ("Resume", docs.resume),
            ("Interview Guide", docs.interview_guide),
        ]

        results: dict[str, DocumentTruthResult] = {}

        for doc_type, content in doc_map:
            if not content:
                results[doc_type] = DocumentTruthResult(pass_strict=True)
                continue
            user_msg = TRUTHFULNESS_DOC_USER_TEMPLATE.format(
                career_profile=career.raw_content,
                job_description=job.raw_content,
                doc_type=doc_type,
                doc_content=content,
            )
            raw = self._call(TRUTHFULNESS_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)
            results[doc_type] = DocumentTruthResult(
                pass_strict=True if data.get("pass_strict") is None else data["pass_strict"],
                unsupported_claims=data.get("unsupported_claims") or [],
                evidence_examples=data.get("evidence_examples") or [],
            )

        cl = results["Cover Letter"]
        resume = results["Resume"]
        ig = results["Interview Guide"]

        return TruthfulnessResult(
            all_supported=cl.pass_strict and resume.pass_strict and ig.pass_strict,
            cover_letter=cl,
            resume=resume,
            interview_guide=ig,
        )

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile) -> VoiceReviewResult:
        """Check how well each document matches the user's voice profile.

        Only the cover letter and resume are reviewed — the interview guide
        is personal preparation and is not subject to voice checks.

        Each document is reviewed in its own call to stay within context limits.
        """
        doc_map = [
            ("Cover Letter", docs.cover_letter),
            ("Resume", docs.resume),
        ]

        assessments: dict[str, str] = {}
        all_issues: list[str] = []
        per_doc_issues: dict[str, list[str]] = {}
        match_scores: list[str] = []
        per_doc_match: dict[str, str] = {}

        for doc_type, content in doc_map:
            if not content:
                assessments[doc_type] = "(not generated)"
                per_doc_match[doc_type] = "strong"  # nothing to review
                per_doc_issues[doc_type] = []
                continue
            user_msg = VOICE_REVIEW_DOC_USER_TEMPLATE.format(
                voice_profile=voice.raw_content,
                doc_type=doc_type,
                doc_content=content,
            )
            raw = self._call(VOICE_REVIEW_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)
            assessments[doc_type] = data.get("assessment") or ""
            doc_issues = data.get("issues") or []
            per_doc_issues[doc_type] = doc_issues
            all_issues.extend(doc_issues)
            doc_match = data.get("overall_match") or "moderate"
            per_doc_match[doc_type] = doc_match
            match_scores.append(doc_match)

        overall_match = (
            min(match_scores, key=lambda m: _MATCH_RANK.get(m, 2))
            if match_scores else "moderate"
        )

        return VoiceReviewResult(
            overall_match=overall_match,
            cover_letter_match=per_doc_match.get("Cover Letter", "moderate"),
            resume_match=per_doc_match.get("Resume", "moderate"),
            cover_letter_assessment=assessments.get("Cover Letter", "(not reviewed)"),
            resume_assessment=assessments.get("Resume", "(not reviewed)"),
            specific_issues=all_issues,
            cover_letter_issues=per_doc_issues.get("Cover Letter", []),
            resume_issues=per_doc_issues.get("Resume", []),
        )

    def review_ai_detection(self, docs: DocumentSet) -> AIDetectionResult:
        """Identify AI-sounding or generic content in the documents.

        Each document is reviewed in its own call to stay within context limits.
        """
        doc_map = [
            ("Cover Letter", docs.cover_letter),
            ("Resume", docs.resume),
            ("Interview Guide", docs.interview_guide),
        ]

        flags_by_doc: dict[str, list[str]] = {
            "Cover Letter": [],
            "Resume": [],
            "Interview Guide": [],
        }
        risk_scores: list[str] = []

        for doc_type, content in doc_map:
            if not content:
                continue
            if doc_type == "Interview Guide":
                # Interview guides are personal prep — skip AI detection
                continue
            user_msg = AI_DETECTION_DOC_USER_TEMPLATE.format(
                doc_type=doc_type,
                doc_content=content,
            )
            raw = self._call(AI_DETECTION_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)
            # Deduplicate flags — LLMs sometimes repeat the same phrase many times.
            raw_flags = data.get("flags") or []
            seen: set[str] = set()
            deduped: list[str] = []
            for f in raw_flags:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            flags_by_doc[doc_type] = deduped
            if r := data.get("risk_level"):
                risk_scores.append(r)

        risk_level = (
            max(risk_scores, key=lambda r: _RISK_RANK.get(r, 2))
            if risk_scores else "low"
        )

        return AIDetectionResult(
            risk_level=risk_level,
            cover_letter_flags=flags_by_doc["Cover Letter"],
            resume_flags=flags_by_doc["Resume"],
            interview_guide_flags=flags_by_doc["Interview Guide"],
        )

    def review_hiring_manager(
        self, docs: DocumentSet, job: JobDescription,
    ) -> HiringManagerReview:
        """Simulate a hiring-manager review of the resume + cover letter."""
        user_msg = HIRING_MANAGER_REVIEW_USER_TEMPLATE.format(
            job_description=job.raw_content,
            resume=docs.resume or "(not provided)",
            cover_letter=docs.cover_letter or "(not provided)",
        )
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
                if area not in ("resume", "cover_letter"):
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
        cover_letter_issues: list[HiringManagerIssue] = []
        resume_issues: list[HiringManagerIssue] = []
        for item in data.get("issues") or []:
            if not isinstance(item, dict) or "phrase" not in item:
                continue
            document = item.get("document", "resume")
            if document not in ("resume", "cover_letter"):
                document = "resume"
            impact = item.get("impact", "medium")
            if impact not in ("high", "medium", "low"):
                impact = "medium"
            hm_issue = HiringManagerIssue(
                document=document,
                phrase=item["phrase"],
                issue=item.get("issue") or "",
                suggestion=item.get("suggestion") or "",
                impact=impact,
            )
            if document == "cover_letter":
                cover_letter_issues.append(hm_issue)
            else:
                resume_issues.append(hm_issue)

        return HiringManagerReview(
            advance_likelihood=likelihood,
            summary=data.get("summary") or "",
            strengths=data.get("strengths") or [],
            concerns=data.get("concerns") or [],
            improvements=improvements,
            cover_letter_issues=cover_letter_issues,
            resume_issues=resume_issues,
        )

    def review_relevance_pruning(
        self, docs: DocumentSet, job: JobDescription,
    ) -> RelevancePruningResult:
        """Identify content that can be removed without weakening the application."""
        doc_map = [
            ("Cover Letter", "cover_letter", docs.cover_letter),
            ("Resume", "resume", docs.resume),
        ]

        cover_letter_issues: list[RelevancePruningIssue] = []
        resume_issues: list[RelevancePruningIssue] = []
        density_scores: list[str] = []

        for doc_type, doc_key, content in doc_map:
            if not content:
                continue
            user_msg = RELEVANCE_PRUNING_DOC_USER_TEMPLATE.format(
                job_description=job.raw_content,
                doc_type=doc_type,
                doc_content=content,
            )
            raw = self._call(RELEVANCE_PRUNING_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)

            density = data.get("overall_density") or "balanced"
            if density not in ("lean", "balanced", "bloated"):
                density = "balanced"
            density_scores.append(density)

            for item in data.get("removal_candidates") or []:
                if not isinstance(item, dict) or "phrase" not in item:
                    continue
                category = item.get("category", "filler")
                if category not in ("redundant", "irrelevant", "filler", "low_impact", "space_waste"):
                    category = "filler"
                severity = item.get("severity", "medium")
                if severity not in ("high", "medium", "low"):
                    severity = "medium"
                issue = RelevancePruningIssue(
                    document=doc_key,
                    phrase=item["phrase"],
                    reason=item.get("reason") or "",
                    category=category,
                    severity=severity,
                )
                if doc_key == "cover_letter":
                    cover_letter_issues.append(issue)
                else:
                    resume_issues.append(issue)

        _DENSITY_RANK = {"lean": 1, "balanced": 2, "bloated": 3}
        overall_density = (
            max(density_scores, key=lambda d: _DENSITY_RANK.get(d, 2))
            if density_scores else "balanced"
        )

        return RelevancePruningResult(
            overall_density=overall_density,
            cover_letter_issues=cover_letter_issues,
            resume_issues=resume_issues,
        )

    def review_ats_keyword(
        self, docs: DocumentSet, job: JobDescription, career: CareerProfile,
    ) -> ATSKeywordResult:
        """Check resume for ATS keyword alignment against the job description."""
        if not docs.resume:
            return ATSKeywordResult(alignment_score="strong")

        user_msg = ATS_KEYWORD_USER_TEMPLATE.format(
            job_description=job.raw_content,
            career_profile=career.raw_content,
            resume=docs.resume,
        )
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

    def review_consistency(self, docs: DocumentSet) -> ConsistencyResult:
        """Compare facts across documents and flag contradictions."""
        # Need at least two documents to compare
        doc_count = sum(1 for d in [docs.cover_letter, docs.resume, docs.interview_guide] if d)
        if doc_count < 2:
            return ConsistencyResult(consistent=True)

        user_msg = CONSISTENCY_USER_TEMPLATE.format(
            resume=docs.resume or "(not provided)",
            cover_letter=docs.cover_letter or "(not provided)",
            interview_guide=docs.interview_guide or "(not provided)",
        )
        raw = self._call(CONSISTENCY_SYSTEM_PROMPT, user_msg)
        data = json.loads(raw)

        issues: list[ConsistencyIssue] = []
        for item in data.get("issues") or []:
            if not isinstance(item, dict) or "quote_a" not in item or "quote_b" not in item:
                continue
            doc_a = item.get("document_a", "resume")
            doc_b = item.get("document_b", "cover_letter")
            valid_docs = ("resume", "cover_letter", "interview_guide")
            if doc_a not in valid_docs:
                doc_a = "resume"
            if doc_b not in valid_docs:
                doc_b = "cover_letter"
            severity = item.get("severity", "medium")
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            issues.append(ConsistencyIssue(
                field=item.get("field") or "",
                document_a=doc_a,
                quote_a=item["quote_a"],
                document_b=doc_b,
                quote_b=item["quote_b"],
                severity=severity,
            ))

        consistent = True if data.get("consistent") is None else data["consistent"]
        if issues:
            consistent = False

        return ConsistencyResult(consistent=consistent, issues=issues)

    def review_grammar(self, docs: DocumentSet) -> GrammarResult:
        """Check each document for grammar, tense, and mechanics errors."""
        doc_map = [
            ("Cover Letter", "cover_letter", docs.cover_letter),
            ("Resume", "resume", docs.resume),
            ("Interview Guide", "interview_guide", docs.interview_guide),
        ]

        cover_letter_issues: list[GrammarIssue] = []
        resume_issues: list[GrammarIssue] = []
        interview_guide_issues: list[GrammarIssue] = []
        all_clean = True

        issue_lists = {
            "cover_letter": cover_letter_issues,
            "resume": resume_issues,
            "interview_guide": interview_guide_issues,
        }

        for doc_type, doc_key, content in doc_map:
            if not content:
                continue
            user_msg = GRAMMAR_DOC_USER_TEMPLATE.format(
                doc_type=doc_type,
                doc_content=content,
            )
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
                issue = GrammarIssue(
                    document=doc_key,
                    phrase=item["phrase"],
                    issue=item.get("issue") or "",
                    suggestion=item.get("suggestion") or "",
                    category=category,
                    severity=severity,
                )
                issue_lists[doc_key].append(issue)
                all_clean = False

        return GrammarResult(
            clean=all_clean,
            cover_letter_issues=cover_letter_issues,
            resume_issues=resume_issues,
            interview_guide_issues=interview_guide_issues,
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

