# Architecture

## Overview

Resume Refinery uses a deterministic orchestrator with bounded specialist agents.
A session is created for each job application and stores all versions of generated
documents, review outputs, and source inputs.

## Pipeline

```
voice_profile.md + career_profile.md + job_description.md
            │
            ▼
     parsers.py
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
    │   ├── cover_letter.md     ← Markdown source (intermediate)
    │   ├── cover_letter.docx   ← Final Word document
    │   ├── resume.md
    │   ├── resume.docx
    │   ├── interview_guide.md
    │   ├── interview_guide.docx
    │   ├── voice_review.json
    │   └── ai_review.json
    └── v2/
        └── ...
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
separate verification concerns. Truth checks run before final acceptance, and repair
passes target only failing documents.

**Raw content over structured parsing:** Input files are passed to Claude as raw text.
This is intentional — flexible, user-friendly input formats are more important than
schema rigidity at the ingestion stage. Structured extraction is only used for session
naming.

**DOCX output:** python-docx produces Word documents without requiring Pandoc (an
external binary). The Markdown source is also preserved alongside the DOCX for easy
diffing and re-export.
