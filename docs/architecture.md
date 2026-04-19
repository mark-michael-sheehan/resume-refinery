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
  │ NarrativeAgent     -> CandidacyNarrative (thesis + pillars + gaps)   │
  │   NarrativeCriticAgent  -> critique loop (up to N passes)            │
  │   Revises narrative until critique passes or max passes exhausted    │
  │ VoiceAgent         -> VoiceStyleGuide                                │
  │                                                                      │
  │   ── User edits narrative via /curate page (optional) ──             │
  │                                                                      │
  │ DraftingAgent      -> DocumentSet (resume only)                      │
  │ VerificationAgent  -> Truthfulness-only repair loop                  │
  │ RepairAgent        -> targeted rewrites with prior-edit context       │
  │ All 8 reviewers    -> single advisory pass (no repair)               │
  └──────────────────────────────────────────────────────────────────────┘
            │
            ▼
    SessionStore.save_documents() + save_context() + DOCX export
    (saved immediately after generation, before review loop)
            │
            ▼
    Truthfulness repair loop (per pass):
      Only the truthfulness reviewer runs in the loop.
      If unsupported claims remain after suppression, a repair pass
        targets only truthfulness findings.
      Edit-region tracking (annotated pass-through): each repair records
        edits tagged with the reviewer that triggered them. On subsequent
        passes, the repair agent receives a "Prior Edits" summary listing
        all earlier edits with reviewer attribution and priority hierarchy.
        The repair agent decides whether to fix, merge, or accept findings
        that overlap prior edits — findings are never silently suppressed.
      Repair edits support three modes: find/replace (default), deletion
        (replace=""), and insert_after (anchor preserved, new content
        appended after anchor) for adding new content to documents.
      Intra-pass collision resolution: when multiple edits target
        overlapping text spans, they are merged via a lightweight LLM call
        that combines all overlapping edits' intents into one replacement.
        Whitespace-normalized matching handles LLM quoting imprecision.
      Loop repeats until truthfulness passes or max passes exhausted
            │
            ▼
    Advisory reviews (single pass, all 8 reviewers):
      All 8 reviewers run once (truth, ATS, grammar, voice, AI detection,
        HM, pruning, narrative coherence).
      Results are merged with the loop's truthfulness result.
      Advisory scores are informational — they never trigger repair.
            │
            ▼
    SessionStore.save_reviews() + final DOCX export
            │
            ▼
        Session versions (v1, v2, ...)
```

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | Domain models + intermediate orchestration artifacts |
| `parsers.py` | Read markdown files → models |
| `agent.py` | Low-level Ollama document generation client |
| `specialist_agents.py` | Narrative, voice, drafting, verification, and repair agents |
| `orchestrator.py` | Deterministic coordinator over specialist agents |
| `reviewers.py` | LLM review client implementations |
| `webapp.py` | Local FastAPI browser app with streaming progress feedback |
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

After per-document extraction, `consolidate_roles()` (Pass 1) immediately
merges duplicate roles — matching by company + title + overlapping dates —
so the user sees a clean timeline rather than one entry per source document.
The user is then landed on the **Role Timeline** page
(`current_phase = "roles"`, `needs_consolidation = True`) where they can
verify, edit, delete, and add roles before further LLM work runs.

Once the user clicks **Finalize & Build Stories**, `consolidate_skills_meta()`
(Pass 2) deduplicates skills (case-insensitive name match, keeping highest
proficiency) and merges education/certifications/meta. A final
`compose_stories()` LLM call generates STAR behavioural stories from the
merged accomplishments.

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
    consolidate_roles()  (Pass 1: identity + roles)
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  USER VERIFICATION GATE (roles phase)       │
    │  Edit / delete / add roles, fix dates and   │
    │  company names on the merged timeline.      │
    └──────────────────┬──────────────────────────┘
                       │  "Finalize & Build Stories"
                       ▼
    consolidate_skills_meta()  (Pass 2: skills + education + meta)
         │               Fuzzy dupe check → retry pass 2 if needed
         │
         ▼
    IngestAgent.compose_stories()  (one LLM call on merged data)
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
    │   ├── resume.md
    │   ├── resume.docx
    │   ├── voice_review.json
    │   ├── ai_review.json
    │   ├── hiring_manager_review.json
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
  "resume": {
    "pass_strict": false,
    "unsupported_claims": ["Led a team of 12 engineers"],
    "evidence_examples": []
  }
}
```

### `voice_review.json` — `VoiceReviewResult`

```json
{
  "overall_match": "moderate",
  "resume_match": "moderate",
  "resume_assessment": "Slightly more formal than the voice profile suggests.",
  "specific_issues": ["Resume bullet 3 uses passive voice"],
  "resume_issues": ["Resume bullet 3 uses passive voice"]
}
```

### `ai_review.json` — `AIDetectionResult`

```json
{
  "risk_level": "medium",
  "resume_flags": ["results-driven", "passionate about"]
}
```

### `exempted_phrases.json` — `ExemptedPhrases`

Only written when the repair agent accepted at least one item as a false positive.

```json
{
  "claims": ["Led cross-functional initiatives"],
  "ai_phrases": ["results-driven"],
  "voice_issues": [],
  "hm_issues": [],
  "narrative_issues": []
}
```

### `candidacy_narrative.json` — `CandidacyNarrative`

```json
{
  "thesis": "Strong distributed systems background makes this candidate ideal.",
  "pillars": [
    {
      "theme": "Backend Engineering",
      "argument": "Led critical infrastructure projects",
      "career_evidence": [
        {
          "evidence": "Cut deploy time 60%",
          "justification": "Directly demonstrates infrastructure optimization skill required by the role"
        },
        {
          "evidence": "Migrated monolith to microservices",
          "justification": "Shows hands-on distributed systems experience at scale"
        }
      ]
    }
  ],
  "gap_framing": ["No Kubernetes experience — transferable from Docker/ECS background"],
  "raw_narrative": "Full narrative text for context."
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

**Compact career context for drafting:** The DraftingAgent receives a compact career
summary (identity, role timeline, technologies, anti-claims, education, certifications,
skills, story titles, and strategic meta) rather than the full career profile. Role
narratives are omitted since the CandidacyNarrative already captures the argument for
candidacy. This frees token budget for richer narrative context. Reviewers and the
RepairAgent still receive the full career profile independently for fact-checking.

**Narrative creation:** The NarrativeAgent reviews the job description against the
career profile and writes a CandidacyNarrative — a structured argument for why the
candidate is a strong fit. The narrative includes a thesis statement, themed argument
pillars with career evidence, and gap framing. Each pillar gathers ALL solid
supporting examples from the career profile (not capped), ordered from strongest to
weakest. Each piece of evidence includes a justification explaining why it supports
the pillar theme. Evidence may overlap across pillars when it genuinely supports
multiple themes. When the LLM is unavailable, a keyword overlap fallback produces a
basic narrative. Pillars are capped at 5 to keep the narrative focused.

**Narrative self-critique:** After the initial narrative is generated, a
`NarrativeCriticAgent` evaluates it against seven quality criteria (thesis specificity,
pillar-JD alignment, evidence exhaustiveness, pillar quality, evidence omission, gap
framing honesty, and pillar coverage of resume content). If the critique identifies
issues, the NarrativeAgent revises the narrative using the critique findings. This
loop runs up to `RESUME_REFINERY_MAX_NARRATIVE_CRITIQUE_PASSES` times (default 2).
This is the highest-leverage intervention: a strong narrative prevents most downstream
quality issues (voice, ATS, coherence) at generation time.

**Intermediate artifacts for explainability:** `CandidacyNarrative` and `VoiceStyleGuide`
are explicit artifacts that can be inspected in the UI and reasoned about in reviews.

**Per-document generation:** The resume is generated as a single Ollama LLM call, which keeps
targeted refinement cheap and traceable.

**Adaptive thinking enabled:** All Ollama calls use `think=True`. This
is especially valuable for the review passes, where the model needs to reason carefully
about voice match and AI-detection signals before producing a JSON result.

**Verification gates:** Truthfulness is the only hard gate in the repair loop —
unsupported claims must be fixed before a resume ships. All other reviewers (voice
match, AI detection, ATS alignment, grammar, hiring-manager, relevance pruning,
narrative coherence) run once after the truthfulness loop as an advisory pass. Their
scores are recorded and displayed but never trigger repair. This design eliminates
cross-reviewer oscillation (e.g., the HM reviewer requesting bolder claims that the
truthfulness reviewer then rejects) and dramatically reduces LLM calls per run.

**Raw content over structured parsing:** Input files are passed to the LLM as raw text.
This is intentional — flexible, user-friendly input formats are more important than
schema rigidity at the ingestion stage. Structured extraction is only used for session
naming.

**DOCX output:** python-docx produces Word documents without requiring Pandoc (an
external binary). The Markdown source is also preserved alongside the DOCX for easy
diffing and re-export.

**User-driven refinement:** `refine_session_run` loads the prior version's reviewer
findings (truthfulness, voice, AI detection, hiring-manager, ATS alignment, grammar,
relevance pruning, narrative coherence) and passes them all to the repair agent
alongside the user's free-form feedback. The candidacy narrative (thesis, pillars,
gap framing) is also included as a reference section so users can say things like
"strengthen the second pillar" or "lean harder into the gap framing". This lets
users reference reviewer output and narrative structure directly (e.g. "implement
all hiring-manager suggestions except the one about the summary section") without
re-running the reviewers up front. Reviewer findings are labeled as "PRIOR REVIEWER
CONTEXT — REFERENCE ONLY" in the repair prompt so the LLM only acts on them when
the user's feedback explicitly requests it — they are never applied autonomously.
After the repair pass, a fresh set of advisory reviews runs once on the updated
documents.
