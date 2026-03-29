"""IngestAgent — extract structured career data from uploaded documents.

Given raw text extracted from one or more documents (PDF, DOCX, TXT), the
agent uses the LLM to populate a CareerRepository as a first-pass starter
that the user then refines via the career wizard.
"""

from __future__ import annotations

import logging
import os
import re

import ollama
from dotenv import load_dotenv
from json_repair import repair_json

from .models import (
    CareerIdentity,
    CareerMeta,
    CareerRepository,
    RoleEntry,
    SkillEntry,
    StoryEntry,
)

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("RESUME_REFINERY_MODEL", "qwen3.5:9b")
MAX_TOKENS = int(os.environ.get("RESUME_REFINERY_MAX_TOKENS", "8192"))
NUM_CTX = int(os.environ.get("RESUME_REFINERY_NUM_CTX", "16384"))

INGEST_SYSTEM_PROMPT = """\
You are an expert career data extractor. Given raw text from one or more \
career-related documents (resumes, CVs, LinkedIn exports, cover letters, \
professional bios), extract structured career information.

Return a single JSON object with exactly these top-level keys:

{
  "identity": {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "headline": "",
    "target_roles": []
  },
  "roles": [
    {
      "company": "",
      "title": "",
      "start_date": "",
      "end_date": "",
      "company_context": "",
      "team_context": "",
      "ownership": "",
      "accomplishments": "",
      "technologies": "",
      "learnings": "",
      "anti_claims": ""
    }
  ],
  "skills": [
    {
      "name": "",
      "category": "language|infrastructure|tool|framework|non_technical|other",
      "proficiency": "expert|strong|working|familiar",
      "years": "",
      "evidence": ""
    }
  ],
  "stories": [],
  "education": "",
  "certifications": "",
  "domain_knowledge": "",
  "meta": {
    "career_arc": "",
    "differentiators": "",
    "themes_to_emphasize": [],
    "anti_claims": [],
    "known_gaps": []
  }
}

Rules:
- Extract ONLY information explicitly present in the documents. Do NOT invent \
or infer anything not stated.
- For roles, preserve chronological order (most recent first).
- Dates should be formatted like "Mar 2021" or "2021" if only the year is known.
- If end_date is not given for the most recent role, use "Present".
- For accomplishments, preserve the original phrasing — do not embellish.
- For skills, infer category and proficiency from context when possible, \
defaulting to "other" and "working".
- If certifications or education are present, capture them as free-form markdown.
- Leave fields empty ("" or []) when the information is not available.
- Respond with ONLY the JSON object. No explanation, no markdown fences.
"""


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output, stripping markdown fences if present."""
    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_ingest_response(raw: str) -> dict:
    """Parse and repair the LLM's JSON response into a dict.

    Uses json_repair to handle common LLM JSON issues (trailing commas,
    missing quotes, etc.).
    """
    cleaned = _strip_think_tags(raw)
    cleaned = _extract_json(cleaned)
    result = repair_json(cleaned, return_objects=True)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    return result


def build_repo_from_parsed(data: dict, repo: CareerRepository) -> CareerRepository:
    """Populate a CareerRepository from the parsed LLM response dict.

    Only sets fields that are present and non-empty in the data.
    """
    # Identity
    if "identity" in data and isinstance(data["identity"], dict):
        ident_data = data["identity"]
        for field in ("name", "email", "phone", "location", "linkedin", "github", "headline"):
            val = ident_data.get(field, "")
            if val and isinstance(val, str):
                setattr(repo.identity, field, val)
        targets = ident_data.get("target_roles", [])
        if isinstance(targets, list):
            repo.identity.target_roles = [str(t) for t in targets if t]

    # Roles
    if "roles" in data and isinstance(data["roles"], list):
        for role_data in data["roles"]:
            if not isinstance(role_data, dict):
                continue
            company = role_data.get("company", "")
            title = role_data.get("title", "")
            if not company or not title:
                continue
            role = RoleEntry(
                company=str(company),
                title=str(title),
                start_date=str(role_data.get("start_date", "")),
                end_date=str(role_data.get("end_date", "Present")),
                company_context=str(role_data.get("company_context", "")),
                team_context=str(role_data.get("team_context", "")),
                ownership=str(role_data.get("ownership", "")),
                accomplishments=str(role_data.get("accomplishments", "")),
                technologies=str(role_data.get("technologies", "")),
                learnings=str(role_data.get("learnings", "")),
                anti_claims=str(role_data.get("anti_claims", "")),
            )
            repo.roles.append(role)

    # Skills
    if "skills" in data and isinstance(data["skills"], list):
        for skill_data in data["skills"]:
            if not isinstance(skill_data, dict):
                continue
            name = skill_data.get("name", "")
            if not name:
                continue
            category = skill_data.get("category", "other")
            valid_categories = {"language", "infrastructure", "tool", "framework", "non_technical", "other"}
            if category not in valid_categories:
                category = "other"
            proficiency = skill_data.get("proficiency", "working")
            valid_proficiencies = {"expert", "strong", "working", "familiar"}
            if proficiency not in valid_proficiencies:
                proficiency = "working"
            skill = SkillEntry(
                name=str(name),
                category=category,
                proficiency=proficiency,
                years=str(skill_data.get("years", "") or ""),
                evidence=str(skill_data.get("evidence", "")),
            )
            repo.skills.append(skill)

    # Stories
    if "stories" in data and isinstance(data["stories"], list):
        for story_data in data["stories"]:
            if not isinstance(story_data, dict):
                continue
            title = story_data.get("title", "")
            if not title:
                continue
            story = StoryEntry(
                title=str(title),
                tags=[str(t) for t in story_data.get("tags", []) if t],
                situation=str(story_data.get("situation", "")),
                task=str(story_data.get("task", "")),
                action=str(story_data.get("action", "")),
                result=str(story_data.get("result", "")),
                what_it_shows=str(story_data.get("what_it_shows", "")),
            )
            repo.stories.append(story)

    # Free-form text fields
    for field in ("education", "certifications", "domain_knowledge"):
        val = data.get(field, "")
        if val and isinstance(val, str):
            setattr(repo, field, val)

    # Meta
    if "meta" in data and isinstance(data["meta"], dict):
        meta_data = data["meta"]
        for field in ("career_arc", "differentiators"):
            val = meta_data.get(field, "")
            if val and isinstance(val, str):
                setattr(repo.meta, field, val)
        for list_field in ("themes_to_emphasize", "anti_claims", "known_gaps"):
            val = meta_data.get(list_field, [])
            if isinstance(val, list):
                setattr(repo.meta, list_field, [str(v) for v in val if v])

    return repo


class IngestAgent:
    """Extracts structured career data from raw document text via LLM."""

    def __init__(self) -> None:
        self.client = ollama.Client(host=BASE_URL)

    def ingest(self, document_text: str) -> dict:
        """Send document text to the LLM and return parsed career data.

        Returns a dict conforming to the CareerRepository schema.
        Raises ValueError if the LLM returns no usable content.
        """
        if not document_text.strip():
            raise ValueError("No document text provided")

        response = self.client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": INGEST_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Extract structured career data from the following documents. "
                    "Return ONLY the JSON object.\n\n"
                    f"---\n{document_text}\n---"
                )},
            ],
            options={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS, "temperature": 0},
        )

        raw = response.message.content or ""
        if not raw.strip():
            raise ValueError(
                "LLM returned empty content — the document may be too long for the "
                "current context window. Try raising RESUME_REFINERY_NUM_CTX."
            )

        return parse_ingest_response(raw)

    def ingest_to_repo(self, document_text: str, repo: CareerRepository) -> CareerRepository:
        """Extract career data from documents and populate the given repo.

        Convenience method that calls ingest() then build_repo_from_parsed().
        Falls back gracefully if the LLM call fails.
        """
        data = self.ingest(document_text)
        return build_repo_from_parsed(data, repo)
