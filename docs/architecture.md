# Architecture

## Overview

Resume Refinery uses a deterministic orchestrator with bounded specialist agents.
A session is created for each job application and stores all versions of generated
documents, review outputs, and source inputs.

## Pipeline

```
voice_profile.md + career_profile.md + job_description.md
        (or CareerRepository via career_wizard.py)
            │
            ▼
     parsers.py / CareerRepository.to_career_profile()
            │
            ▼
  ResumeRefineryOrchestrator
  ┌──────────────────────────────────────────────────────────────────────┐
  │ EvidenceAgent      -> EvidencePack (requirements + matched evidence) │
  │ VoiceAgent         -> VoiceStyleGuide                                │
  │ DraftingAgent      -> DocumentSet                                    │
  │ VerificationAgent  -> Truth/Voice/AI review bundle                  │
  │ RepairAgent        -> targeted rewrites for unsupported claims      │
  └──────────────────────────────────────────────────────────────────────┘
            │
            ▼
    SessionStore.save_documents() + save_reviews() + DOCX export
            │
            ▼
        Session versions (v1, v2, ...)
```

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | Domain models + intermediate orchestration artifacts |
| `parsers.py` | Read markdown files → models |
| `agent.py` | Low-level Claude document generation client |
| `specialist_agents.py` | Evidence, voice, drafting, verification, and repair agents |
| `orchestrator.py` | Deterministic coordinator over specialist agents |
| `reviewers.py` | LLM review client implementations |
| `webapp.py` | Local FastAPI browser app |
| `session.py` | Session CRUD, versioning, disk I/O |
| `exporters.py` | Markdown → DOCX via python-docx |
| `cli.py` | CLI commands calling orchestrator |
| `career_repo.py` | Career repository CRUD, disk I/O for structured career data |
| `career_wizard.py` | HTMX-powered guided elicitation wizard (FastAPI sub-router) |
| `elicitation.py` | LLM-powered follow-up probe agent for career elicitation |
| `ingest_agent.py` | LLM-powered document ingest — extracts structured career data from uploaded PDFs, DOCX, and text files |

## Career Repository Storage

Career repositories live in `~/.resume_refinery/careers/` by default.
Override with `RESUME_REFINERY_CAREERS_DIR` env var.

```
~/.resume_refinery/careers/
└── jordan-lee/
    └── career.json          ← Full structured career data (single JSON)
```

A `CareerRepository` can be flattened into a `CareerProfile` via
`to_career_profile()`, making it a drop-in replacement for file-uploaded
career profiles in the existing pipeline.

### Document Ingest

The `IngestAgent` (in `ingest_agent.py`) processes each uploaded document
(PDF, DOCX, TXT/MD) in a separate LLM call, giving each file the full
context window. The extraction prompt includes field-level guidance that
mirrors the wizard's helper text, ensuring the LLM fills each field
appropriately and completely.

After all documents are extracted, `consolidate_repo()` merges duplicate
roles (matched by company + title + overlapping dates) and deduplicates
skills (case-insensitive name match, keeping highest proficiency).
`compose_stories()` then generates STAR behavioural stories by making **one
LLM call per role**, giving each role the full output-token budget.  This
per-role approach produces significantly more stories than a single
monolithic call because output tokens are not shared across all roles.
Each role's company context, team context, ownership, accomplishments,
technologies, and learnings are included in the prompt so the LLM has rich
material for the Situation and Task components.

Each role and story carries an `extraction_confidence` rating (`high` /
`medium` / `low`) and `confidence_notes` so the wizard can surface
low-confidence areas for user review.

```
Upload: resume.pdf + perf_review_2024.pdf + perf_review_2025.pdf
         │
         ▼
    parsers._read_file_content()  (per file: PDF, DOCX, TXT)
         │
         ▼
    IngestAgent.ingest_to_repo()  ×N  (one LLM call per document)
         │
         ▼
    consolidate_repo()  (2-pass LLM merge)
         │               Pass 1: identity + roles
         │               Pass 2: skills + education + meta
         │               Fuzzy dupe check → retry pass 2 if needed
         │
         ▼
    IngestAgent.compose_stories()  (one LLM call per role)
         │
         ▼
    CareerRepository (pre-filled with confidence scores)
         │
         ▼
    Wizard Phase 3 (role deep-dive) — low-confidence roles first
```

## Session Storage

Sessions live in `~/.resume_refinery/sessions/` by default.
Override with `RESUME_REFINERY_SESSIONS_DIR` env var.

```
~/.resume_refinery/sessions/
└── acme-cloud_staff-engineer_2026-03-20/
    ├── session.json            ← metadata + version history
    ├── inputs/
    │   ├── career_profile.md
    │   ├── voice_profile.md
    │   └── job_description.md
    ├── v1/
    │   ├── cover_letter.md         ← Markdown source (intermediate)
    │   ├── cover_letter.docx       ← Final Word document
    │   ├── resume.md
    │   ├── resume.docx
    │   ├── interview_guide.md
    │   ├── interview_guide.docx
    │   ├── voice_review.json
    │   ├── ai_review.json
    │   ├── exempted_phrases.json   ← Phrases/claims exempted during repair (only when any exemptions occurred)
    │   └── repair_pass_0/         ← Snapshot after each repair pass (if repair ran)
    │       └── ...
    └── v2/
        └── ...
```

## Review and Artifact JSON Schemas

Each version directory can contain the following JSON files. All are emitted by the
orchestrator and correspond to Pydantic models in `models.py`.

### `truth_review.json` — `TruthfulnessResult`

```json
{
  "all_supported": false,
  "cover_letter": {
    "pass_strict": true,
    "unsupported_claims": [],
    "evidence_examples": ["Reduced infra costs by $180K/year (career profile, Acme Corp)"]
  },
  "resume": {
    "pass_strict": false,
    "unsupported_claims": ["Led a team of 12 engineers"],
    "evidence_examples": []
  },
  "interview_guide": {
    "pass_strict": true,
    "unsupported_claims": [],
    "evidence_examples": []
  }
}
```

### `voice_review.json` — `VoiceReviewResult`

```json
{
  "overall_match": "moderate",
  "cover_letter_match": "strong",
  "resume_match": "moderate",
  "interview_guide_match": "moderate",
  "cover_letter_assessment": "Matches the direct, analytical tone well.",
  "resume_assessment": "Slightly more formal than the voice profile suggests.",
  "interview_guide_assessment": "Good conversational tone.",
  "specific_issues": ["Resume bullet 3 uses passive voice"],
  "cover_letter_issues": [],
  "resume_issues": ["Resume bullet 3 uses passive voice"],
  "interview_guide_issues": []
}
```

### `ai_review.json` — `AIDetectionResult`

```json
{
  "risk_level": "medium",
  "cover_letter_flags": ["results-driven", "passionate about"],
  "resume_flags": [],
  "interview_guide_flags": []
}
```

### `exempted_phrases.json` — `ExemptedPhrases`

Only written when the repair agent accepted at least one item as a false positive.

```json
{
  "claims": ["Led cross-functional initiatives"],
  "ai_phrases": ["results-driven"],
  "voice_issues": []
}
```

### `evidence_pack.json` — `EvidencePack`

```json
{
  "job_requirements": [
    {"requirement": "distributed systems", "category": "skill", "source_excerpt": "..."}
  ],
  "matched_evidence": [
    {"requirement": "distributed systems", "evidence": "Built event-driven pipeline...", "source_excerpt": "...", "relevance_score": 4}
  ],
  "gaps": ["No Kubernetes experience mentioned"],
  "source_summary": ["Reduced infra costs by $180K/year"]
}
```

### `voice_guide.json` — `VoiceStyleGuide`

```json
{
  "core_adjectives": ["direct", "analytical"],
  "style_rules": ["Short declarative sentences", "Avoid adverbs"],
  "preferred_phrases": ["I built", "We shipped"],
  "phrases_to_avoid": ["passionate about", "results-driven"],
  "writing_samples": ["Sample paragraph from voice profile..."]
}
```

## Design Decisions

**Bounded agentic design:** Specialist agents are role-constrained and never control
the global workflow. The orchestrator owns step order, retries, and persistence.

**Intermediate artifacts for explainability:** `EvidencePack` and `VoiceStyleGuide`
are explicit artifacts that can be inspected in the UI and reasoned about in reviews.

**Per-document generation:** Each document remains a separate Claude API call, which keeps
targeted refinement cheap and traceable.

**Adaptive thinking enabled:** All Claude calls use `thinking: {type: "adaptive"}`. This
is especially valuable for the review passes, where the model needs to reason carefully
about voice match and AI-detection signals before producing a JSON result.

**Verification gates:** Truthfulness, voice match, and AI-detection are treated as
separate verification concerns. The truthfulness reviewer receives the career profile
and job description as grounding sources; voice and AI-detection reviewers operate
only on the documents and voice profile. Truth checks run before final acceptance,
and repair passes target only failing documents.

**Raw content over structured parsing:** Input files are passed to Claude as raw text.
This is intentional — flexible, user-friendly input formats are more important than
schema rigidity at the ingestion stage. Structured extraction is only used for session
naming.

**DOCX output:** python-docx produces Word documents without requiring Pandoc (an
external binary). The Markdown source is also preserved alongside the DOCX for easy
diffing and re-export.
