"""ElicitationAgent — LLM-powered follow-up probes for career elicitation.

Given a user's answers to career questions, the agent generates contextual
follow-up probes that push for specificity, quantification, and evidence.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import ollama
from dotenv import load_dotenv

from .models import RoleEntry


@dataclass
class ProbeResult:
    """Result from a probe_role call, indicating source and probes."""

    probes: list[str] = field(default_factory=list)
    llm_used: bool = True

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")
MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "8192"))
NUM_CTX = int(os.environ.get("RESUME_REFINERY_NUM_CTX", "16384"))

ELICITATION_SYSTEM_PROMPT = """\
You are an expert career coach helping someone build a detailed career \
repository. Your job is to read their answers about a specific work role \
and generate 1-4 short, specific follow-up questions that will strengthen \
their profile.

Focus on:
- Missing quantification (numbers, percentages, dollar amounts, timelines)
- Vague accomplishments that need concrete examples
- Missing personal ownership (what YOU did vs. what the team did)
- Missing context (company size, team size, scope)
- Missing outcomes or results

Rules:
- Be encouraging but direct.
- Each probe should be a single question, max two sentences.
- Do NOT repeat information the user already provided.
- If the answers are already detailed and specific, respond with exactly: LOOKS_GOOD
- Return probes as a numbered list (1. ... 2. ...) with nothing else.
"""


def _build_role_context(role: RoleEntry) -> str:
    """Format a role's answers into a readable block for the LLM."""
    parts = [f"Role: {role.title} @ {role.company} ({role.start_date} – {role.end_date})"]
    if role.company_context:
        parts.append(f"Company context: {role.company_context}")
    if role.team_context:
        parts.append(f"Team context: {role.team_context}")
    if role.ownership:
        parts.append(f"What they owned: {role.ownership}")
    if role.accomplishments:
        parts.append(f"Accomplishments: {role.accomplishments}")
    if role.technologies:
        parts.append(f"Technologies: {role.technologies}")
    if role.learnings:
        parts.append(f"Learnings: {role.learnings}")
    return "\n".join(parts)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from LLM output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ElicitationAgent:
    """Generates contextual follow-up probes for career elicitation."""

    def __init__(self) -> None:
        self.client = ollama.Client(host=BASE_URL)

    def probe_role(self, role: RoleEntry) -> ProbeResult:
        """Return follow-up probes for a role, or empty list if answers are solid.

        Falls back to static probes if the LLM call fails.
        """
        context = _build_role_context(role)
        user_msg = (
            "Here are a candidate's answers about one of their work roles. "
            "Generate follow-up probes to strengthen their answers, or respond "
            "with LOOKS_GOOD if the answers are already detailed.\n\n"
            f"{context}"
        )

        try:
            response = self.client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ELICITATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                options={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS},
            )
            raw = _strip_think_tags(response.message.content or "")
        except Exception:
            log.warning("ElicitationAgent LLM call failed, falling back to static probes", exc_info=True)
            return ProbeResult(probes=_static_probes(role), llm_used=False)

        if not raw or "LOOKS_GOOD" in raw:
            return ProbeResult(probes=[], llm_used=True)

        return ProbeResult(probes=_parse_probes(raw), llm_used=True)


def _parse_probes(text: str) -> list[str]:
    """Extract numbered or bulleted probe lines from LLM output."""
    import re
    lines = text.strip().splitlines()
    probes: list[str] = []
    for line in lines:
        line = line.strip()
        # Strip leading number/bullet: "1. ", "- ", "* "
        cleaned = re.sub(r"^(?:\d+[.)]\s*|[-*]\s+)", "", line).strip()
        if cleaned:
            probes.append(cleaned)
    return probes[:4]


def _static_probes(role: RoleEntry) -> list[str]:
    """Static fallback probes when LLM is unavailable."""
    probes: list[str] = []
    if role.accomplishments and "%" not in role.accomplishments and "$" not in role.accomplishments:
        probes.append(
            "Can you quantify any of those accomplishments? "
            "Percentages, dollar amounts, time saved?"
        )
    if role.accomplishments and len(role.accomplishments) < 100:
        probes.append(
            "Could you walk through one of those accomplishments in more detail? "
            "What specifically did YOU do vs. the team?"
        )
    if not role.company_context:
        probes.append("What did this company do? How big was it?")
    if not role.ownership:
        probes.append(
            "What were you specifically responsible for — "
            "not just the team, but you personally?"
        )
    return probes
