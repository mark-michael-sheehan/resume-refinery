"""Core agent — generates individual career documents using an OpenAI-compatible API (Ollama)."""

from __future__ import annotations

import os
from typing import Iterator

import openai
from dotenv import load_dotenv

from .models import CareerProfile, DocumentKey, DocumentSet, JobDescription, VoiceProfile
from .prompts import (
    COVER_LETTER_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    INTERVIEW_GUIDE_PROMPT,
    RESUME_PROMPT,
    generation_user_message,
)

load_dotenv()

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")
MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "4096"))

_DOC_PROMPTS: dict[DocumentKey, str] = {
    "cover_letter": COVER_LETTER_PROMPT,
    "resume": RESUME_PROMPT,
    "interview_guide": INTERVIEW_GUIDE_PROMPT,
}


class ResumeRefineryAgent:
    """Generates and refines career documents using an OpenAI-compatible API (Ollama)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "ollama"),
            base_url=BASE_URL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
    ) -> DocumentSet:
        """Generate all three documents from scratch. Returns a DocumentSet."""
        docs = DocumentSet()
        for key in ("cover_letter", "resume", "interview_guide"):
            docs.set(key, self._generate_one(key, career, voice, job))  # type: ignore[arg-type]
        return docs

    def generate_document(
        self,
        key: DocumentKey,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        feedback: str | None = None,
        previous_version: str | None = None,
    ) -> str:
        """Generate (or regenerate with feedback) a single document. Returns Markdown."""
        return self._generate_one(key, career, voice, job, feedback, previous_version)

    def stream_document(
        self,
        key: DocumentKey,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        feedback: str | None = None,
        previous_version: str | None = None,
    ) -> Iterator[str]:
        """Stream a single document as text delta chunks."""
        user_msg = generation_user_message(
            career_profile_content=career.raw_content,
            voice_profile_content=voice.raw_content,
            job_description_content=job.raw_content,
            doc_prompt=_DOC_PROMPTS[key],
            feedback=feedback,
            previous_version=previous_version,
        )
        stream = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            stream=True,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            extra_body={"think": False},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        key: DocumentKey,
        career: CareerProfile,
        voice: VoiceProfile,
        job: JobDescription,
        feedback: str | None = None,
        previous_version: str | None = None,
    ) -> str:
        user_msg = generation_user_message(
            career_profile_content=career.raw_content,
            voice_profile_content=voice.raw_content,
            job_description_content=job.raw_content,
            doc_prompt=_DOC_PROMPTS[key],
            feedback=feedback,
            previous_version=previous_version,
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            extra_body={"think": False},
        )
        return response.choices[0].message.content.strip()
