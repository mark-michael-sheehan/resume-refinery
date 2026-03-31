"""IngestAgent — extract structured career data from uploaded documents.

Given raw text extracted from one or more documents (PDF, DOCX, TXT), the
agent uses one LLM call *per document* to populate a CareerRepository.
After all documents are ingested, a two-pass LLM consolidation step merges
duplicate roles and skills (Pass 1: identity + roles, Pass 2: skills + meta),
and a STAR composition pass generates behavioural stories from the merged
accomplishments.

Pipeline:
    Per-document extraction  →  consolidate_repo() (2 passes)  →  compose_stories()
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

# ---------------------------------------------------------------------------
# Extraction prompt — field-level guidance mirrors the career wizard
# ---------------------------------------------------------------------------

INGEST_SYSTEM_PROMPT = """\
You are an expert career data extractor. Given raw text from a single \
career-related document (resume, CV, LinkedIn export, cover letter, \
professional bio, performance review, self-assessment, or similar), \
extract as much structured career information as the document contains.

Return a single JSON object with exactly these top-level keys.  Each field \
includes a description of what to extract and how to format it.

{
  "identity": {
    "name": "Full legal or professional name",
    "email": "Email address",
    "phone": "Phone number in original format",
    "location": "City, State or City, Country — e.g. 'San Francisco, CA'",
    "linkedin": "Full LinkedIn profile URL",
    "github": "Full GitHub profile URL",
    "headline": "One sentence that captures who this person is professionally — e.g. 'Senior Backend Engineer with 5+ years in distributed systems'",
    "target_roles": ["Comma-separated role types the person is targeting — e.g. 'Staff Engineer', 'Engineering Manager'"]
  },
  "roles": [
    {
      "company": "Company or organisation name",
      "title": "The person's job title at this company",
      "start_date": "Start date formatted as 'Mar 2021' or '2021' if only year is known",
      "end_date": "End date formatted as 'Feb 2023', or 'Present' if this is the current role",
      "company_context": "What did this company do? How big was it (employees, revenue)? What stage (startup, enterprise, government)? What industry? Example: 'Series B fintech startup, ~200 employees, building payment infrastructure'",
      "team_context": "Team size, who this person reported to, scope of the team. Example: 'Platform team of 8, reporting to VP Engineering'",
      "ownership": "What was this person specifically responsible for? Systems, teams, budgets, outcomes they owned. Use their own words where possible.",
      "accomplishments": "THE MOST CRITICAL FIELD. Extract ALL accomplishments, metrics, awards, recognition, ratings, project outcomes, and impact statements. Include specific numbers, percentages, dollar amounts, team sizes, and timeframes. Walk through every win — be specific about what the person did and what happened as a result. If the document mentions performance ratings (e.g. 'Exceeds Expectations'), include them. Consolidate scattered references to the same accomplishment into one coherent entry. Err on the side of extracting TOO MUCH rather than too little.",
      "technologies": "Languages, frameworks, tools, infrastructure, platforms mentioned in context of this role. Comma-separated.",
      "learnings": "What did this role teach the person? What skills, insights, or capabilities did they develop that they couldn't have learned elsewhere?",
      "anti_claims": "Anything mentioned as a limitation, area for improvement, or development need for this role. Things that should NOT appear as strengths on a resume.",
      "extraction_confidence": "high | medium | low — Rate how complete this role's extraction is. 'high' = clear, detailed info with metrics. 'medium' = reasonable but some fields thin. 'low' = minimal info found, user should expand.",
      "confidence_notes": "Brief note on what was thin or missing — e.g. 'accomplishments section was sparse, no metrics found' or 'no team context in source document'"
    }
  ],
  "skills": [
    {
      "name": "Skill name",
      "category": "language | infrastructure | tool | framework | non_technical | other",
      "proficiency": "expert | strong | working | familiar — infer from context (years used, depth of work, how centrally it featured)",
      "years": "Approximate years of experience, e.g. '6+' or '3'",
      "evidence": "Concrete evidence of this skill grounded in the document — e.g. 'Primary language at DataFlow, built all backend services' or 'Led 3 Kubernetes migrations'"
    }
  ],
  "stories": [],
  "education": "Degrees, schools, graduation years, honors, GPA if mentioned. Free-form markdown.",
  "certifications": "Professional certifications, courses, training programs. Free-form markdown.",
  "domain_knowledge": "Industries, problem spaces, or subject areas this person knows deeply — e.g. 'Healthcare compliance, HIPAA, insurance claims processing'",
  "meta": {
    "career_arc": "Describe this person's career trajectory in 2-3 sentences. Where did they start, what's the progression, where are they headed?",
    "differentiators": "If you could tell a hiring manager only 3 things about this person, what would they be?",
    "themes_to_emphasize": ["Recurring themes across the career — e.g. 'Ownership mentality', 'Measurable cost savings', 'Cross-team influence'"],
    "anti_claims": ["Things that should NEVER be claimed on a resume based on what the document says — hard limits and honest boundaries"],
    "known_gaps": ["Areas where the person is aware they are weaker or where the document notes development needs"]
  }
}

Rules:
- Extract as much information as possible that is grounded in the document. \
Include all details, metrics, context, and evidence you can find.
- Do NOT fabricate information, but DO synthesize scattered references to the \
same topic into coherent entries.
- For accomplishments, extract EVERYTHING: numbers, percentages, dollar amounts, \
timelines, ratings, awards, recognition. This is the most valuable field.
- For roles, preserve chronological order (most recent first).
- Dates should be formatted like "Mar 2021" or "2021" if only the year is known.
- If end_date is not given for the most recent role, use "Present".
- For skills, infer category and proficiency from context when possible, \
defaulting to "other" and "working".
- If certifications or education are present, capture them as free-form markdown.
- Leave fields empty ("" or []) when the information is genuinely not available.
- Leave the "stories" array empty — stories are composed in a later step.
- Respond with ONLY the JSON object. No explanation, no markdown fences.
"""

# ---------------------------------------------------------------------------
# STAR story composition prompt
# ---------------------------------------------------------------------------

STORY_COMPOSITION_PROMPT = """\
You are an expert career coach who helps professionals articulate their \
accomplishments as compelling STAR (Situation-Task-Action-Result) stories.

Given the structured career data below, identify accomplishments that contain \
enough detail to construct behavioural STAR stories.

For each story, return a JSON object with these fields:
{
  "title": "Short, memorable title — e.g. 'Redis Caching Cost Savings'",
  "tags": ["Relevant skill/theme tags — e.g. 'cost-optimization', 'initiative', 'architecture'"],
  "situation": "What was the context? What was happening? What problem or opportunity existed?",
  "task": "What was this person's responsibility or what did they decide to take on?",
  "action": "What specifically did THIS PERSON do? Be concrete — technologies, decisions, collaborations.",
  "result": "What happened? Quantify if possible — numbers, percentages, dollar amounts, timelines.",
  "what_it_shows": "What trait or capability does this story demonstrate about this person?",
  "extraction_confidence": "high | medium | low — 'high' = all 4 STAR components have concrete evidence. 'medium' = 3 components solid, 1 inferred. 'low' = 2 or fewer components have evidence.",
  "confidence_notes": "Which STAR components were directly stated vs inferred from context"
}

Rules:
- Only create stories where the source material provides concrete evidence for \
at least 2 of the 4 STAR components (situation, task, action, result).
- Ground every statement in the career data provided. Do NOT invent details.
- Prefer stories with quantified results (metrics, dollar amounts, percentages).
- A single accomplishment may become one story. Multi-role accomplishments \
(e.g. a project spanning 2 review periods) should be combined into one story.
- Return a JSON array of story objects. If no stories can be composed, return [].
- Respond with ONLY the JSON array. No explanation, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Consolidation prompt — merge duplicates across documents
# ---------------------------------------------------------------------------

CONSOLIDATION_PASS1_PROMPT = """\
You are an expert career data consolidator. You are given structured career \
data that was extracted from MULTIPLE documents about the SAME person \
(e.g. a resume, multiple annual performance reviews, a LinkedIn export). \
Because each document was processed independently, the data contains \
duplicate and overlapping entries that must be merged.

Your job is to produce a single, clean, consolidated JSON object containing \
ONLY the "identity" and "roles" keys, following these rules:

IDENTITY:
- Produce one identity block using the most complete information available \
across all sources. Prefer non-empty values.

ROLES:
- If the same role appears multiple times (same company and title, with \
overlapping or adjacent dates), merge them into ONE role entry.
- When merging roles, combine ALL accomplishments, ownership details, \
learnings, and context from every source. Do NOT drop any information — \
the merged entry should be RICHER than any single source.
- Use the earliest start_date and latest end_date across all sources.
- Merge technologies into a single deduplicated comma-separated list.
- If roles are at different companies or have different titles, keep them \
as separate entries even if dates overlap (the person may have held \
multiple jobs simultaneously).

CONFIDENCE:
- For each merged role, set extraction_confidence to the LOWEST confidence \
from the source entries (most conservative).
- Update confidence_notes to reflect the merge.

Return a JSON object with exactly two keys: "identity" and "roles". \
Use the EXACT same field schema as the input. \
Respond with ONLY the JSON object. No explanation, no markdown fences.
"""

CONSOLIDATION_PASS2_PROMPT = """\
You are an expert career data consolidator. You are given structured career \
data that was extracted from MULTIPLE documents about the SAME person \
(e.g. a resume, multiple annual performance reviews, a LinkedIn export). \
Because each document was processed independently, the data contains \
duplicate and overlapping entries that must be merged.

Your job is to produce a single, clean, consolidated JSON object containing \
ONLY the "skills", "education", "certifications", "domain_knowledge", and \
"meta" keys, following these rules:

SKILLS:
- Deduplicate skills by name (case-insensitive). When merging duplicates:
  - Keep the highest proficiency level (expert > strong > working > familiar).
  - Keep the longest years value.
  - Combine evidence from all sources.
  - Prefer a specific category (language, framework, etc.) over "other".

META:
- Merge career_arc and differentiators into the most complete version.
- Combine and deduplicate list fields (themes_to_emphasize, anti_claims, known_gaps).

EDUCATION / CERTIFICATIONS / DOMAIN KNOWLEDGE:
- Deduplicate entries that refer to the same degree, certification, or domain.
- Keep all unique entries.

Return a JSON object with exactly these keys: "skills", "education", \
"certifications", "domain_knowledge", "meta". \
Use the EXACT same field schema as the input. \
Respond with ONLY the JSON object. No explanation, no markdown fences.
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


def _coerce_str(val: object) -> str:
    """Coerce a value to a string, joining lists with newlines.

    LLMs sometimes return a JSON array of strings where we expect a single
    string.  ``str(["a", "b"])`` would produce ``"['a', 'b']"`` — instead
    we join list items with newlines so the text reads naturally.
    """
    if isinstance(val, list):
        return "\n".join(str(item) for item in val if item)
    return str(val) if val else ""


def build_repo_from_parsed(data: dict, repo: CareerRepository) -> CareerRepository:
    """Populate a CareerRepository from the parsed LLM response dict.

    Only sets fields that are present and non-empty in the data.
    """
    # Identity — first non-empty value wins (don't overwrite with blanks)
    if "identity" in data and isinstance(data["identity"], dict):
        ident_data = data["identity"]
        for field in ("name", "email", "phone", "location", "linkedin", "github", "headline"):
            val = ident_data.get(field, "")
            if val and isinstance(val, str) and not getattr(repo.identity, field):
                setattr(repo.identity, field, val)
        targets = ident_data.get("target_roles", [])
        if isinstance(targets, list) and targets and not repo.identity.target_roles:
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
            # Validate confidence
            confidence = role_data.get("extraction_confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            role = RoleEntry(
                company=str(company),
                title=str(title),
                start_date=str(role_data.get("start_date", "")),
                end_date=str(role_data.get("end_date", "Present")),
                company_context=_coerce_str(role_data.get("company_context", "")),
                team_context=_coerce_str(role_data.get("team_context", "")),
                ownership=_coerce_str(role_data.get("ownership", "")),
                accomplishments=_coerce_str(role_data.get("accomplishments", "")),
                technologies=_coerce_str(role_data.get("technologies", "")),
                learnings=_coerce_str(role_data.get("learnings", "")),
                anti_claims=_coerce_str(role_data.get("anti_claims", "")),
                extraction_confidence=confidence,
                confidence_notes=_coerce_str(role_data.get("confidence_notes", "")),
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
                evidence=_coerce_str(skill_data.get("evidence", "")),
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
            confidence = story_data.get("extraction_confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            story = StoryEntry(
                title=str(title),
                tags=[str(t) for t in story_data.get("tags", []) if t],
                situation=_coerce_str(story_data.get("situation", "")),
                task=_coerce_str(story_data.get("task", "")),
                action=_coerce_str(story_data.get("action", "")),
                result=_coerce_str(story_data.get("result", "")),
                what_it_shows=_coerce_str(story_data.get("what_it_shows", "")),
                extraction_confidence=confidence,
                confidence_notes=_coerce_str(story_data.get("confidence_notes", "")),
            )
            repo.stories.append(story)

    # Free-form text fields — append new content (dedup at consolidation)
    for field in ("education", "certifications", "domain_knowledge"):
        val = _coerce_str(data.get(field, ""))
        if val:
            existing = getattr(repo, field, "")
            if existing:
                setattr(repo, field, existing + "\n" + val)
            else:
                setattr(repo, field, val)

    # Meta — merge lists, keep first non-empty strings
    if "meta" in data and isinstance(data["meta"], dict):
        meta_data = data["meta"]
        for field in ("career_arc", "differentiators"):
            val = _coerce_str(meta_data.get(field, ""))
            if val and not getattr(repo.meta, field):
                setattr(repo.meta, field, val)
        for list_field in ("themes_to_emphasize", "anti_claims", "known_gaps"):
            val = meta_data.get(list_field, [])
            if isinstance(val, list):
                existing = getattr(repo.meta, list_field)
                new_items = [str(v) for v in val if v and str(v) not in existing]
                setattr(repo.meta, list_field, existing + new_items)

    return repo


# ---------------------------------------------------------------------------
# Cross-document consolidation (LLM-based)
# ---------------------------------------------------------------------------

def _repo_to_consolidation_json(repo: CareerRepository, keys: list[str] | None = None) -> str:
    """Serialize selected repo sections to JSON for the consolidation LLM.

    If *keys* is None, serializes all consolidation-relevant sections.
    Otherwise only the named keys are included.
    """
    import json

    all_data: dict = {
        "identity": repo.identity.model_dump(),
        "roles": [r.model_dump() for r in repo.roles],
        "skills": [s.model_dump() for s in repo.skills],
        "education": repo.education,
        "certifications": repo.certifications,
        "domain_knowledge": repo.domain_knowledge,
        "meta": repo.meta.model_dump(),
    }
    if keys is not None:
        all_data = {k: v for k, v in all_data.items() if k in keys}
    return json.dumps(all_data, indent=2)


def _consolidation_call(
    client: ollama.Client,
    system_prompt: str,
    repo_json: str,
) -> dict | None:
    """Make a single consolidation LLM call and return parsed data, or None on failure."""
    try:
        response = client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "Consolidate the following career data extracted from multiple "
                    "documents about the same person. Merge duplicate entries, "
                    "keeping ALL information. Return ONLY the JSON object.\n\n"
                    f"---\n{repo_json}\n---"
                )},
            ],
            options={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS, "temperature": 0},
        )
        raw = response.message.content or ""
        if not raw.strip():
            log.warning("Consolidation LLM returned empty response for a pass")
            return None
        return parse_ingest_response(raw)
    except Exception:
        log.warning("Consolidation LLM call failed for a pass", exc_info=True)
        return None


def consolidate_repo(repo: CareerRepository, client: ollama.Client | None = None) -> CareerRepository:
    """Merge duplicate roles and skills from multi-document extraction via LLM.

    Uses two LLM passes to stay within output-token limits:
      Pass 1 — identity + roles  (accomplishment-heavy, gets full output budget)
      Pass 2 — skills + education + certifications + domain knowledge + meta

    Falls back to the original repo data for any pass that fails.
    """
    # Skip if there's nothing meaningful to consolidate
    if len(repo.roles) <= 1 and len(repo.skills) <= 1:
        return repo

    if client is None:
        client = ollama.Client(host=BASE_URL)

    # Build a fresh repo preserving metadata
    consolidated = CareerRepository(
        repo_id=repo.repo_id,
        created_at=repo.created_at,
        updated_at=repo.updated_at,
        current_phase=repo.current_phase,
        deepdive_role_index=repo.deepdive_role_index,
        voice_raw=repo.voice_raw,
    )

    # Pass 1 — identity + roles
    pass1_json = _repo_to_consolidation_json(repo, keys=["identity", "roles"])
    pass1_data = _consolidation_call(client, CONSOLIDATION_PASS1_PROMPT, pass1_json)
    if pass1_data:
        build_repo_from_parsed(pass1_data, consolidated)
    else:
        # Fall back: copy original identity + roles
        build_repo_from_parsed(
            {"identity": repo.identity.model_dump(),
             "roles": [r.model_dump() for r in repo.roles]},
            consolidated,
        )

    # Pass 2 — skills + education + certifications + domain_knowledge + meta
    pass2_keys = ["skills", "education", "certifications", "domain_knowledge", "meta"]
    pass2_json = _repo_to_consolidation_json(repo, keys=pass2_keys)
    pass2_data = _consolidation_call(client, CONSOLIDATION_PASS2_PROMPT, pass2_json)
    if pass2_data:
        build_repo_from_parsed(pass2_data, consolidated)
    else:
        # Fall back: copy original pass-2 fields
        import json as _json
        fallback = {k: v for k, v in _json.loads(
            _repo_to_consolidation_json(repo, keys=pass2_keys)
        ).items()}
        build_repo_from_parsed(fallback, consolidated)

    # Deterministic fuzzy-match check — if duplicates remain, re-run LLM
    if _has_duplicate_skills(consolidated.skills):
        log.info("Duplicate skills detected after consolidation, re-running pass 2")
        retry_json = _repo_to_consolidation_json(consolidated, keys=pass2_keys)
        retry_data = _consolidation_call(client, CONSOLIDATION_PASS2_PROMPT, retry_json)
        if retry_data and "skills" in retry_data:
            # Replace only skills from the retry (keep education/meta from first pass)
            consolidated.skills = []
            build_repo_from_parsed({"skills": retry_data["skills"]}, consolidated)

    return consolidated


# ---------------------------------------------------------------------------
# Fuzzy skill duplicate detection
# ---------------------------------------------------------------------------

def _normalize_skill_name(name: str) -> str:
    """Lower-case, strip, collapse whitespace, remove trailing version-like suffixes."""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Collapse e.g. "React.js" / "ReactJS" / "React JS" → "reactjs"
    s = re.sub(r"[.\-\s]", "", s)
    return s


def _has_duplicate_skills(skills: list[SkillEntry], threshold: float = 0.85) -> bool:
    """Return True if any two skills are duplicates.

    Uses exact-match on normalised names first, then falls back to
    ``SequenceMatcher`` ratio for fuzzy near-misses.
    """
    from difflib import SequenceMatcher

    names = [_normalize_skill_name(s.name) for s in skills]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            return True
        seen.add(name)

    # Fuzzy pass — only needed if no exact dupes found
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if SequenceMatcher(None, names[i], names[j]).ratio() >= threshold:
                return True
    return False


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
                    "Extract structured career data from the following document. "
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
        """Extract career data from a single document and append to the repo.

        Convenience method that calls ingest() then build_repo_from_parsed().
        """
        data = self.ingest(document_text)
        return build_repo_from_parsed(data, repo)

    def compose_stories(self, repo: CareerRepository) -> CareerRepository:
        """Generate STAR stories from the repo's roles and accomplishments.

        Makes one LLM call against the already-structured career data.
        Stories are appended to repo.stories.
        """
        # Build a summary of roles + accomplishments for the LLM
        role_summaries: list[str] = []
        for role in repo.roles:
            parts = [f"## {role.title} @ {role.company} ({role.start_date} – {role.end_date})"]
            if role.ownership:
                parts.append(f"Owned: {role.ownership}")
            if role.accomplishments:
                parts.append(f"Accomplishments: {role.accomplishments}")
            if role.technologies:
                parts.append(f"Technologies: {role.technologies}")
            if role.learnings:
                parts.append(f"Learnings: {role.learnings}")
            role_summaries.append("\n".join(parts))

        if not role_summaries:
            return repo

        career_text = "\n\n".join(role_summaries)

        try:
            response = self.client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": STORY_COMPOSITION_PROMPT},
                    {"role": "user", "content": (
                        "Compose STAR stories from the following career data. "
                        "Return ONLY the JSON array.\n\n"
                        f"---\n{career_text}\n---"
                    )},
                ],
                options={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS, "temperature": 0},
            )

            raw = response.message.content or ""
            if not raw.strip():
                log.warning("Story composition returned empty response")
                return repo

            cleaned = _strip_think_tags(raw)
            cleaned = _extract_json(cleaned)
            stories_data = repair_json(cleaned, return_objects=True)

            if isinstance(stories_data, dict) and "stories" in stories_data:
                stories_data = stories_data["stories"]

            if not isinstance(stories_data, list):
                log.warning("Story composition returned non-list: %s", type(stories_data).__name__)
                return repo

            for story_data in stories_data:
                if not isinstance(story_data, dict):
                    continue
                title = story_data.get("title", "")
                if not title:
                    continue
                confidence = story_data.get("extraction_confidence", "medium")
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                story = StoryEntry(
                    title=str(title),
                    tags=[str(t) for t in story_data.get("tags", []) if t],
                    situation=str(story_data.get("situation", "")),
                    task=str(story_data.get("task", "")),
                    action=str(story_data.get("action", "")),
                    result=str(story_data.get("result", "")),
                    what_it_shows=str(story_data.get("what_it_shows", "")),
                    extraction_confidence=confidence,
                    confidence_notes=str(story_data.get("confidence_notes", "")),
                )
                repo.stories.append(story)

        except Exception:
            log.warning("Story composition failed, continuing without stories", exc_info=True)

        return repo
