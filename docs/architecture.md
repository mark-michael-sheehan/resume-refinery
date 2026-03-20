# Architecture

## Overview

Resume Refinery is a multi-step, session-aware pipeline. A session is created for each
job application and stores all versions of the generated documents, review results, and
the original input files.

## Pipeline

```
voice_profile.md + career_profile.md + job_description.md
          │
          ▼
     parsers.py ── extract metadata for session naming
          │
          ▼
    SessionStore.create()
          │
          ▼
  (CLI or local web app)
            │
            ▼
  ResumeRefineryAgent.generate_all()
  ┌──────────────────────────────────┐
  │  generate cover_letter           │  Each doc is a separate
  │  generate resume                 │  Claude API call (streaming,
  │  generate interview_guide        │  adaptive thinking)
  └──────────────────────────────────┘
          │
          ▼
    SessionStore.save_documents() → v1/ (Markdown + DOCX)
          │
          ▼
  DocumentReviewer.review_truthfulness()
  ┌──────────────────────────────────┐
  │ strict claim-evidence verification │
  │ optional rewrite pass for unsupported claims │
  └──────────────────────────────────┘
            │
            ▼
  DocumentReviewer.review_all()
  ┌──────────────────────────────────┐
  │  review_voice()    → VoiceReviewResult    │
  │  review_ai_detection() → AIDetectionResult │
  └──────────────────────────────────┘
          │
          ▼
    SessionStore.save_reviews() → v1/voice_review.json, ai_review.json
          │
          ▼
  User reviews results → gives feedback
          │
          ▼
  ResumeRefineryAgent.generate_document("cover_letter", feedback=...)
          │
          ▼
    SessionStore.save_documents() → v2/
          │
          ▼
  DocumentReviewer.review_all() → v2/reviews
```

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | All Pydantic data models |
| `parsers.py` | Read markdown files → models |
| `agent.py` | Generate documents via Claude |
| `reviewers.py` | Voice-match and AI-detection review |
| `webapp.py` | Local FastAPI browser app |
| `session.py` | Session CRUD, versioning, disk I/O |
| `exporters.py` | Markdown → DOCX via python-docx |
| `cli.py` | CLI commands (Typer + Rich) |

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

**Per-document generation:** Each document (cover letter, resume, interview guide) is a
separate Claude API call. This makes targeted refinement cheap — regenerating just the
cover letter only costs one call — and allows streaming per document.

**Adaptive thinking enabled:** All Claude calls use `thinking: {type: "adaptive"}`. This
is especially valuable for the review passes, where the model needs to reason carefully
about voice match and AI-detection signals before producing a JSON result.

**Two-stage review:** Voice review and AI-detection are separate agents with distinct
system prompts. Running them as one combined call risks the model conflating the two
concerns. Separate calls allow each reviewer to focus on its specific criteria.

**Raw content over structured parsing:** Input files are passed to Claude as raw text.
This is intentional — flexible, user-friendly input formats are more important than
schema rigidity at the ingestion stage. Structured extraction is only used for session
naming.

**DOCX output:** python-docx produces Word documents without requiring Pandoc (an
external binary). The Markdown source is also preserved alongside the DOCX for easy
diffing and re-export.
