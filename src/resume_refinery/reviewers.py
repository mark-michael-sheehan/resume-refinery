"""Review agents — voice-match and AI-detection checks on generated documents."""

from __future__ import annotations

import json
import os

import openai
from dotenv import load_dotenv

from .models import (
    AIDetectionResult,
    DocumentSet,
    ReviewBundle,
    TruthfulnessResult,
    CareerProfile,
    VoiceProfile,
    VoiceReviewResult,
)
from .prompts import (
    AI_DETECTION_SYSTEM_PROMPT,
    AI_DETECTION_USER_TEMPLATE,
    TRUTHFULNESS_SYSTEM_PROMPT,
    TRUTHFULNESS_USER_TEMPLATE,
    VOICE_REVIEW_SYSTEM_PROMPT,
    VOICE_REVIEW_USER_TEMPLATE,
)

load_dotenv()

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RESUME_REFINERY_REVIEW_MODEL", "qwen3.5:9b")
MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_REVIEW_MAX_TOKENS", "4096"))


class DocumentReviewer:
    """Runs voice-match and AI-detection reviews on a DocumentSet."""

    def __init__(self, api_key: str | None = None) -> None:
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "ollama"),
            base_url=BASE_URL,
        )

    def review_all(self, docs: DocumentSet, voice: VoiceProfile) -> ReviewBundle:
        """Run both review passes and return a ReviewBundle."""
        return ReviewBundle(
            voice=self.review_voice(docs, voice),
            ai_detection=self.review_ai_detection(docs),
        )

    def review_truthfulness(self, docs: DocumentSet, career: CareerProfile) -> TruthfulnessResult:
        """Verify that document claims are explicitly supported by career-profile evidence."""
        user_msg = TRUTHFULNESS_USER_TEMPLATE.format(
            career_profile=career.raw_content,
            cover_letter=docs.cover_letter or "(not generated)",
            resume=docs.resume or "(not generated)",
            interview_guide=docs.interview_guide or "(not generated)",
        )
        raw = self._call(TRUTHFULNESS_SYSTEM_PROMPT, user_msg)
        return TruthfulnessResult(**json.loads(raw))

    def review_voice(self, docs: DocumentSet, voice: VoiceProfile) -> VoiceReviewResult:
        """Check how well the documents match the user's voice profile."""
        user_msg = VOICE_REVIEW_USER_TEMPLATE.format(
            voice_profile=voice.raw_content,
            cover_letter=docs.cover_letter or "(not generated)",
            resume=docs.resume or "(not generated)",
            interview_guide=docs.interview_guide or "(not generated)",
        )
        raw = self._call(VOICE_REVIEW_SYSTEM_PROMPT, user_msg)
        return VoiceReviewResult(**json.loads(raw))

    def review_ai_detection(self, docs: DocumentSet) -> AIDetectionResult:
        """Identify AI-sounding or generic content in the documents."""
        user_msg = AI_DETECTION_USER_TEMPLATE.format(
            cover_letter=docs.cover_letter or "(not generated)",
            resume=docs.resume or "(not generated)",
            interview_guide=docs.interview_guide or "(not generated)",
        )
        raw = self._call(AI_DETECTION_SYSTEM_PROMPT, user_msg)
        return AIDetectionResult(**json.loads(raw))

    def _call(self, system: str, user_msg: str) -> str:
        """Make an Ollama API call and return the text response."""
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            extra_body={"think": False},
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present (defensive fallback)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        return raw
