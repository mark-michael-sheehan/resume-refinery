# Resume Refinery

An LLM-powered career document generation tool. Provide your career history, writing
voice, and a job description — get a tailored cover letter, resume, and interview guide,
with automatic voice-match and AI-detection review built in.

Powered by **Claude Opus 4.6** with adaptive thinking.

---

## Features

- **Voice matching**: Define your writing style with adjectives, style notes, and writing
  samples. The agent mirrors your authentic voice, not generic AI prose.
- **Three documents per application**: Cover letter, ATS-friendly resume, and interview
  prep guide — each generated in a separate focused pass.
- **Local web app**: Run a browser UI locally for upload, generation, refinement, and
  session browsing.
- **Targeted refinement**: Give feedback on specific documents and regenerate only those.
  Every refinement is saved as a new version.
- **Strict truthfulness pass**: Generated content is checked against the career profile.
  Unsupported claims trigger automatic rewrite attempts and are blocked by default unless
  you opt in to `--allow-unverified`.
- **Built-in review pipeline**: After every generation, two separate LLM reviewers assess:
  1. How well the documents match your voice
  2. Whether any content sounds AI-generated or generic
- **Session tracking**: Each job application is a session with full version history,
  stored at `~/.resume_refinery/sessions/`.
- **Word document output**: Documents are exported as `.docx` files via `python-docx`.

---

## Setup

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

Optional low-cost model settings:

```bash
# generation defaults to claude-3-5-haiku-latest
RESUME_REFINERY_MODEL=claude-3-5-haiku-latest
RESUME_REFINERY_MAX_TOKENS=4096

# review defaults to claude-3-5-haiku-latest
RESUME_REFINERY_REVIEW_MODEL=claude-3-5-haiku-latest
RESUME_REFINERY_REVIEW_MAX_TOKENS=4096
```

---

## Quick Start

```bash
# Start the local web app (recommended)
resume-refinery-web
# Opens on http://127.0.0.1:8765

# 1. Generate documents for a job
resume-refinery new \
  examples/career_profile.md \
  examples/voice_profile.md \
  examples/job_description.md

# 2. Refine based on the review feedback
resume-refinery refine acme-cloud_staff-engineer_2026-03-20 \
  --doc cover_letter \
  --feedback "Lead with the Redis cost-saving story, not the migration."

# 3. List all your sessions
resume-refinery list

# 4. Open the output folder
resume-refinery show acme-cloud_staff-engineer_2026-03-20 --open
```

---

## Input Files

| File | Purpose |
|---|---|
| `career_profile.md` | Work history, education, projects, key points |
| `voice_profile.md` | Your writing style: adjectives, style notes, writing samples, phrases to avoid |
| `job_description.md` | The target job description |

See the [`examples/`](examples/) directory for complete, annotated templates.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest                                            # all tests
pytest --cov=resume_refinery --cov-report=term-missing  # with coverage
```

---

## Project Structure

```
resume_refinery/
├── src/
│   └── resume_refinery/
│       ├── agent.py         ← Document generation (Claude API, streaming)
│       ├── reviewers.py     ← Voice-match + AI-detection review agents
│       ├── session.py       ← Session CRUD and versioning
│       ├── exporters.py     ← Markdown → DOCX conversion
│       ├── parsers.py       ← Read markdown files → models
│       ├── models.py        ← Pydantic data models
│       ├── prompts.py       ← All system/generation/review prompts
│       └── cli.py           ← CLI (Typer + Rich)
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_parsers.py
│   ├── test_agent.py
│   ├── test_reviewers.py
│   ├── test_session.py
│   └── test_exporters.py
├── docs/
│   ├── architecture.md
│   └── usage.md
├── examples/
│   ├── career_profile.md    ← Complete template
│   ├── voice_profile.md     ← Complete template
│   └── job_description.md   ← Complete template
├── .env.example
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

---

## Documentation

- [Usage Guide](docs/usage.md) — commands and input file formats
- [Architecture](docs/architecture.md) — design decisions and data flow
