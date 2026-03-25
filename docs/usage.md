# Usage Guide

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Configure Ollama endpoint
cp .env.example .env
# Edit .env if Ollama runs on a non-default URL (default: http://localhost:11434)
# See the Environment Variables section below for all available options.
```

---

## Preparing Your Input Files

You need three files:

### 1. Voice Profile (`voice_profile.md`)

Describes how you write and communicate. The more specific, the better the voice match.

Include:
- **Core adjectives** that describe your style (e.g., "direct", "analytical", "concise")
- **Style notes** about sentence structure, tone, and habits
- **Phrases you actually use** — pull from emails, performance reviews, or previous cover letters
- **Phrases to avoid** — generic AI tells you want the system to avoid
- **2–3 writing samples** from your own work (cover letters, LinkedIn About sections, emails
  you felt good about, performance self-reviews)

See [examples/voice_profile.md](../examples/voice_profile.md) for a complete template.

### 2. Career Profile (`career_profile.md`)

Your full professional history. Structured Markdown works best, but the format is flexible.

Include:
- Contact information
- Work experience with responsibilities and quantified achievements
- Education
- Projects (context, your role, outcome, technologies)
- A **Key Points** section for anything you specifically want drawn upon

See [examples/career_profile.md](../examples/career_profile.md) for a complete template.

### 3. Job Description (`job_description.md`)

The target role. Paste the full job description text. Include:
- `Title:` and `Company:` lines at the top for session naming
- The full description text (more context = better tailoring)

See [examples/job_description.md](../examples/job_description.md) for a template.

---

## Commands

### Web App (recommended)

```bash
resume-refinery-web
```

Then open `http://127.0.0.1:8765` in your browser.

### `new` — Start a new session

```bash
resume-refinery new career_profile.md voice_profile.md job_description.md
```

This will:
1. Create a new session with a unique ID (e.g. `acme-cloud_staff-engineer_2026-03-20`)
2. Generate all three documents (streaming to terminal as they're written)
3. Export DOCX files to `~/.resume_refinery/sessions/<session_id>/v1/`
4. Run strict truthfulness verification and targeted repair passes
5. Run voice-match and AI-detection reviews automatically
6. Print a summary of truthfulness + review results

Skip voice/AI style reviews with `--skip-review`.

Strict truthfulness is enforced by default. To keep output despite unsupported
claims, pass `--allow-unverified`.

---

### `refine` — Regenerate with feedback

```bash
# Refine a specific document
resume-refinery refine acme-cloud_staff-engineer_2026-03-20 \
  --doc cover_letter \
  --feedback "The opener is too generic. Lead with the Redis cost-saving story instead."

# Refine all documents at once
resume-refinery refine acme-cloud_staff-engineer_2026-03-20 \
  --feedback "Focus more on technical leadership and architectural decisions, less on day-to-day tasks."
```

The agent sees the previous version and your feedback. A new version (`v2`, `v3`, etc.) is
created automatically. Reviews run automatically after refinement.

---

### `review` — Re-run reviews without regenerating

```bash
resume-refinery review acme-cloud_staff-engineer_2026-03-20

# Review a specific version
resume-refinery review acme-cloud_staff-engineer_2026-03-20 --version 1
```

Runs strict truthfulness (against career profile and job description), voice-match, and AI-detection reviewers on the current (or specified) version.

---

### `list` — See all sessions

```bash
resume-refinery list
```

Shows all sessions with job title, company, creation date, and number of versions.

---

### `show` — Inspect a session

```bash
resume-refinery show acme-cloud_staff-engineer_2026-03-20

# Open the output folder in Explorer/Finder
resume-refinery show acme-cloud_staff-engineer_2026-03-20 --open
```

---

## Output Files

For each version, the following files are created:

| File | Description |
|---|---|
| `cover_letter.docx` | Cover letter as a Word document |
| `resume.docx` | Resume as a Word document |
| `interview_guide.docx` | Interview prep guide as a Word document |
| `cover_letter.md` | Markdown source (for diffing between versions) |
| `resume.md` | Markdown source |
| `interview_guide.md` | Markdown source |

---

## Environment Variables

All settings are read from a `.env` file in the project root (or from the real environment).
Copy `.env.example` to `.env` to get started — every variable has a sensible default.

### Ollama connection

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server. Use the bare host — do **not** add `/v1`. |

### Document generation

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_MODEL` | `qwen3.5:9b` | Model used to write the cover letter, resume, and interview guide. Any model pulled in Ollama works; `qwen3.5:4b` is faster and lighter. |
| `RESUME_REFINERY_MAX_TOKENS` | `4096` | Maximum new tokens the generation model may produce per document. Raise if output is cut off mid-section. |

### Review / verification

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_REVIEW_MODEL` | `qwen3.5:9b` | Model used for truthfulness, voice-match, and AI-detection reviews. Can be set to a lighter model to save RAM (e.g. `qwen3.5:4b`). |
| `RESUME_REFINERY_REVIEW_MAX_TOKENS` | `4096` | Maximum new tokens the review model may produce per review call. Each call covers one document at a time, so 4096 is generally sufficient. |

### Context window (shared)

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_NUM_CTX` | `16384` | KV-cache size (tokens) requested from Ollama for every call — both generation and review. Ollama pre-allocates this on the GPU/CPU at call time: `16384` uses ~2 GB extra RAM and comfortably fits a full career profile plus one document. Raise to `32768` for very long profiles; lower to `8192` if you hit out-of-memory errors. If you see `WARNING: Ollama reviewer returned empty content`, increase this value. |

### Repair loop pass limits

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_MAX_TRUTH_PASSES` | `2` | Max review+repair passes for the truthfulness loop. Each pass checks claims against the career profile; failing docs are re-generated before the next pass. Set to `1` to review without repair, `0` to skip entirely. |
| `RESUME_REFINERY_MAX_VOICE_PASSES` | `2` | Max passes for the voice-match loop. Each pass rates voice fidelity; if not `"strong"`, all docs are re-generated with voice feedback. |
| `RESUME_REFINERY_MAX_AI_PASSES` | `2` | Max passes for the AI-detection loop. Each pass flags generic/AI-sounding phrases; flagged docs are re-generated with the quoted phrases as repair feedback. |

### Session storage

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_SESSIONS_DIR` | `~/.resume_refinery/sessions` | Directory where sessions are persisted. Each session is a subdirectory containing input copies, versioned Markdown sources, DOCX exports, and review JSON files. |
| `voice_review.json` | Voice-match review results |
| `ai_review.json` | AI-detection review results |
| `truth_review.json` | Strict claim-support verification results |

---

## Using as a Library

```python
from resume_refinery.orchestrator import ResumeRefineryOrchestrator
from resume_refinery.parsers import load_career_profile, load_job_description, load_voice_profile
from resume_refinery.session import SessionStore

career = load_career_profile("career_profile.md")
voice = load_voice_profile("voice_profile.md")
job = load_job_description("job_description.md")

orchestrator = ResumeRefineryOrchestrator(store=SessionStore())
result = orchestrator.create_session_run(career, voice, job)

print(result.session.session_id)
print(result.reviews.truthfulness.all_supported)
print(result.reviews.voice.overall_match)      # "strong" / "moderate" / "weak"
print(result.reviews.ai_detection.risk_level)  # "low" / "medium" / "high"
```
