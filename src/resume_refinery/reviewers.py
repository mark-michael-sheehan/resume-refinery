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
    DocumentSet,
    DocumentTruthResult,
    JobDescription,
    ReviewBundle,
    TruthfulnessResult,
    CareerProfile,
    VoiceProfile,
    VoiceReviewResult,
)
from .prompts import (
    AI_DETECTION_DOC_USER_TEMPLATE,
    AI_DETECTION_SYSTEM_PROMPT,
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


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from *text*.

    When ``format="json"`` is not used (e.g. with thinking mode), the model
    may emit preamble/postamble text or wrap JSON in markdown fences.  This
    helper finds the outermost ``{…}`` or ``[…]`` substring.
    """
    # Strip markdown code fences first.
    fenced = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    # Find the first { or [ and match to its closing counterpart.
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text  # fallback: return as-is and let caller handle errors


_GEN_MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")


class DocumentReviewer:
    """Runs voice-match and AI-detection reviews on a DocumentSet."""

    def __init__(self, api_key: str | None = None) -> None:
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
                pass_strict=data.get("pass_strict", True),
                unsupported_claims=data.get("unsupported_claims", []),
                evidence_examples=data.get("evidence_examples", []),
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

        Each document is reviewed in its own call to stay within context limits.
        """
        doc_map = [
            ("Cover Letter", docs.cover_letter),
            ("Resume", docs.resume),
            ("Interview Guide", docs.interview_guide),
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
            if doc_type == "Interview Guide":
                # Interview guides are personal prep — skip voice check
                assessments[doc_type] = "(skipped — personal prep)"
                per_doc_match[doc_type] = "strong"
                per_doc_issues[doc_type] = []
                continue
            user_msg = VOICE_REVIEW_DOC_USER_TEMPLATE.format(
                voice_profile=voice.raw_content,
                doc_type=doc_type,
                doc_content=content,
            )
            raw = self._call(VOICE_REVIEW_SYSTEM_PROMPT, user_msg)
            data = json.loads(raw)
            assessments[doc_type] = data.get("assessment", "")
            doc_issues = data.get("issues", [])
            per_doc_issues[doc_type] = doc_issues
            all_issues.extend(doc_issues)
            doc_match = data.get("overall_match", "moderate")
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
            interview_guide_match=per_doc_match.get("Interview Guide", "moderate"),
            cover_letter_assessment=assessments.get("Cover Letter", "(not reviewed)"),
            resume_assessment=assessments.get("Resume", "(not reviewed)"),
            interview_guide_assessment=assessments.get("Interview Guide", "(not reviewed)"),
            specific_issues=all_issues,
            cover_letter_issues=per_doc_issues.get("Cover Letter", []),
            resume_issues=per_doc_issues.get("Resume", []),
            interview_guide_issues=per_doc_issues.get("Interview Guide", []),
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
            flags_by_doc[doc_type] = data.get("flags", [])
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

