# Resume Refinery

An LLM-powered career document generation tool. Provide your career history, writing
voice, and a job description — get a tailored cover letter, resume, and interview guide,
with strict truth verification, voice-match review, and AI-detection review.

Built as a deterministic workflow orchestrator with bounded specialist agents.

---

## Features

- **Voice matching**: Define your writing style with adjectives, style notes, and writing
  samples. The agent mirrors your authentic voice, not generic AI prose.
- **Three documents per application**: Cover letter, ATS-friendly resume, and interview
  prep guide — each generated in a separate focused pass.
- **Local web app**: Run a browser UI locally for upload, generation, refinement, and
  session browsing.
- **Career Builder**: Use a guided wizard (`/career`) to build a structured career
  repository through directed questions — or import from existing resume files
  (PDF, DOCX, TXT, MD) and let an AI agent extract the data for you.
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

## Prerequisites

1. **Python 3.11+**
2. **[Ollama](https://ollama.com/)** installed and running locally. Verify with:
   ```bash
   ollama --version
   ```
3. **Pull the default model** (or whichever model you configure in `.env`):
   ```bash
   ollama pull qwen3.5:9b
   ```
   Ollama must be serving before you run Resume Refinery. Start it with `ollama serve`
   if it isn't running as a system service.

---

## Setup

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env if Ollama runs on a non-default URL (default: http://localhost:11434)
```

Optional model/token settings:

```bash
# Generation model — defaults to qwen3.5:9b (lighter: qwen3.5:4b)
RESUME_REFINERY_MODEL=qwen3.5:9b
RESUME_REFINERY_MAX_TOKENS=8192

# Review model — can be the same as generation or a lighter model
RESUME_REFINERY_REVIEW_MODEL=qwen3.5:9b
RESUME_REFINERY_REVIEW_MAX_TOKENS=4096

# Context window size in tokens (shared by generation and review)
# Ollama pre-allocates this as KV cache; 16384 ≈ 2 GB extra RAM
RESUME_REFINERY_NUM_CTX=16384

# Custom session storage path (default: ~/.resume_refinery/sessions)
# Both the web app and CLI read/write sessions here.
# RESUME_REFINERY_SESSIONS_DIR=/path/to/sessions
```

See [`.env.example`](.env.example) for the full reference with descriptions, or the
[Environment Variables](docs/usage.md#environment-variables) section of the Usage Guide
for a complete table.

---

## Quick Start

### Web App (recommended)

Make sure Ollama is running (`ollama serve`) before starting the web app.

```bash
resume-refinery-web
# Opens on http://127.0.0.1:8765
```

The web app gives you:

| Page | What it does |
|---|---|
| **Home** (`/`) | Upload a career profile, voice profile, and job description to generate documents. Links to Sessions and Career Builder. |
| **Career Builder** (`/career`) | Build a structured career repository through a guided wizard, or import from existing resume files (PDF, DOCX, TXT, MD). The result can be selected directly when creating a new session. |
| **Sessions** (`/sessions`) | Browse all sessions, view generated documents side-by-side, download DOCX files, and submit refinement feedback. |

> **Tip — fastest on-ramp:** Go to `/career`, click **Import from Documents**, upload
> your existing resume, then start a new session from the home page using the imported
> career repository.

### CLI

```bash
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
│       ├── orchestrator.py  ← Deterministic workflow coordinator
│       ├── specialist_agents.py ← Evidence/voice/drafting/verification/repair agents
│       ├── agent.py         ← Low-level Ollama document generation client
│       ├── reviewers.py     ← Voice-match + AI-detection + truth review clients
│       ├── session.py       ← Session CRUD and versioning
│       ├── exporters.py     ← Markdown → DOCX conversion
│       ├── parsers.py       ← Read markdown files → models
│       ├── models.py        ← Domain + intermediate artifact models
│       ├── prompts.py       ← All system/generation/review prompts
│       ├── cli.py           ← CLI (Typer + Rich)
│       └── webapp.py        ← Local FastAPI web app
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

- [Usage Guide](docs/usage.md) — full web-app walkthrough, CLI commands, input file
  formats, environment variables, and library API
- [Architecture](docs/architecture.md) — orchestrator, specialist agents, and data flow
