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

### 2. Career Profile (`career_profile.md`) or Career Repository

Your full professional history. You have two options:

**Option A: Markdown file** — Structured Markdown works best, but the format is flexible.

Include:
- Contact information
- Work experience with responsibilities and quantified achievements
- Education
- Projects (context, your role, outcome, technologies)
- A **Key Points** section for anything you specifically want drawn upon

See [examples/career_profile.md](../examples/career_profile.md) for a complete template.

**Option B: Career Builder** — Use the guided web wizard at `/career` to build a
structured career repository through directed questions. The wizard walks you through
identity, roles, accomplishments (with follow-up probes), skills, STAR stories,
career strategy, and voice profile. The result is stored as JSON and can be selected
directly when creating a new session.

**Option C: Import from Documents** — On the Career Builder page (`/career`), use
the "Import from Documents" form to upload one or more files (PDF, DOCX, TXT, MD).
An AI agent extracts structured career data (identity, roles, skills, education, etc.)
from the combined document text in a single pass. The result is saved as a new career
repository and you enter the wizard to review and refine the extracted data. This is
the fastest way to get started if you already have a resume or CV on hand.

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

The home page includes links to **Browse Sessions** and **Career Builder**.
Use the Career Builder (`/career`) to create a structured career repository through
guided questions before generating documents.

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
6. Print per-document issues from each reviewer and find/replace edits applied per repair pass

Pass `--skip-review` to skip voice/AI style reviews and the repair loop entirely.
Only the truthfulness review runs; voice and AI-detection are not executed and no
repair passes are performed. This saves significant LLM calls when reviews are not needed.

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

For each version, the following files are created in `~/.resume_refinery/sessions/<session_id>/v<N>/`:

### Documents

| File | Description |
|---|---|
| `cover_letter.docx` | Cover letter as a Word document |
| `resume.docx` | Resume as a Word document |
| `interview_guide.docx` | Interview prep guide as a Word document |
| `cover_letter.md` | Markdown source (for diffing between versions) |
| `resume.md` | Markdown source |
| `interview_guide.md` | Markdown source |

### Context artifacts

| File | Description |
|---|---|
| `evidence_pack.json` | Extracted job requirements matched to career evidence, plus identified gaps. Used by the drafting and truthfulness agents. |
| `voice_guide.json` | Distilled voice style guide (adjectives, rules, preferred phrases, phrases to avoid, writing samples). Used by the drafting and voice-review agents. |

### Review results

| File | Description |
|---|---|
| `truth_review.json` | Per-document truthfulness verification — whether every claim is supported by the career profile. Contains `all_supported`, and per-document `pass_strict`, `unsupported_claims`, and `evidence_examples`. |
| `voice_review.json` | Voice-match review — `overall_match` (`strong`/`moderate`/`weak`), per-document match level, assessment, and specific issues. |
| `ai_review.json` | AI-detection review — `risk_level` (`low`/`medium`/`high`) and per-document flagged phrases that sound AI-generated. |
| `exempted_phrases.json` | Cumulative list of phrases/claims the repair agent accepted as false positives during the repair loop. Only written when exemptions occurred. Contains `claims` (truthfulness), `ai_phrases` (AI-detection), and `voice_issues` (voice). |

### Repair pass snapshots

Each repair pass creates a `repair_pass_<N>/` subdirectory containing the document and review state at that point:

| File | Description |
|---|---|
| `repair_pass_<N>/cover_letter.md` | Document state after repair pass N |
| `repair_pass_<N>/resume.md` | Document state after repair pass N |
| `repair_pass_<N>/interview_guide.md` | Document state after repair pass N |
| `repair_pass_<N>/truth_review.json` | Truthfulness result that preceded repair pass N |
| `repair_pass_<N>/voice_review.json` | Voice-match result that preceded repair pass N |
| `repair_pass_<N>/ai_review.json` | AI-detection result that preceded repair pass N |

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

### Storage

| Variable | Default | Description |
|---|---|---|
| `RESUME_REFINERY_SESSIONS_DIR` | `~/.resume_refinery/sessions` | Directory where sessions are persisted. Each session is a subdirectory containing input copies, versioned Markdown sources, DOCX exports, and review JSON files. |
| `RESUME_REFINERY_CAREERS_DIR` | `~/.resume_refinery/careers` | Directory where career repositories are persisted. Each repository is a subdirectory containing a `career.json` file with the full `CareerRepository` model. |
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

---

## Model Selection Guide

Resume Refinery works with any model available in Ollama. The default is `qwen3.5:9b`,
which balances quality and resource usage. Here are some practical guidelines:

| Model | VRAM / RAM | Quality | Speed | Notes |
|---|---|---|---|---|
| `qwen3.5:4b` | ~3 GB | Good | Fast | Lighter option — may produce more AI-sounding language that triggers extra repair passes. Good for quick iterations or machines with limited RAM. |
| `qwen3.5:9b` | ~6 GB | Very good | Moderate | Default. Strong reasoning, good voice matching, reliable JSON output from reviewers. |
| `qwen3:8b` | ~5 GB | Good | Moderate | Previous-generation Qwen. Solid but `qwen3.5:9b` is generally preferred. |
| `qwen3:14b` | ~10 GB | Excellent | Slower | Highest quality for truthfulness and voice fidelity. Choose if you have the VRAM and prioritise output quality over speed. |

### Tips

- **Generation and review models can differ.** Use a larger model for generation
  (`RESUME_REFINERY_MODEL`) and a lighter one for reviews (`RESUME_REFINERY_REVIEW_MODEL`)
  to save RAM during the review loop.
- **Context window matters.** `RESUME_REFINERY_NUM_CTX` controls how much of your career
  profile the model can see. If your career profile is long (>5 pages), raise this to
  `32768` or higher. Each 16K tokens ≈ 2 GB extra RAM.
- **Smaller models produce more repair passes.** Review flags and unsupported-claim counts
  tend to be higher with smaller models, which means more repair iterations. You can raise
  `RESUME_REFINERY_AI_FLAG_TOLERANCE` or `RESUME_REFINERY_MAX_REPAIR_PASSES` to compensate.
- **Pull your model before running.** Ollama must already have the model downloaded:
  ```bash
  ollama pull qwen3.5:9b
  ```

---

## End-to-End Walkthrough

This section walks through a complete run using the example files in `examples/`.

### 1. Start Ollama and pull the model

```bash
ollama serve            # if not already running
ollama pull qwen3.5:9b
```

### 2. Set up the project

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env             # edit if needed
```

### 3. Generate documents via CLI

```bash
resume-refinery new \
  examples/career_profile.md \
  examples/voice_profile.md \
  examples/job_description.md
```

The tool will:
1. Extract an evidence pack (job requirements matched to career evidence).
2. Distill a voice style guide from your voice profile.
3. Generate a cover letter, resume, and interview guide (streamed to the terminal).
4. Run truthfulness, voice-match, and AI-detection reviews.
5. If any review fails, run surgical repair passes (up to `MAX_REPAIR_PASSES` times).
6. Export final `.docx` files.

You'll see output like:

```
Session created: acme-cloud_staff-engineer_2026-03-25
Extracting evidence pack...
Distilling voice guide...
Generating Cover Letter (model is thinking, output appears after reasoning)...
[streaming output...]
Generating Resume (model is thinking, output appears after reasoning)...
[streaming output...]
Generating Interview Guide (model is thinking, output appears after reasoning)...
[streaming output...]
─── Review Pass 1/3 ───
  Truthfulness review (3 LLM calls)...
  Voice review (2 LLM calls)...
  AI-detection review (2 LLM calls)...
Truthfulness: ALL SUPPORTED
Voice match: STRONG
AI-detection risk: LOW
```

### 4. Find your output

```bash
resume-refinery show acme-cloud_staff-engineer_2026-03-25 --open
```

This opens `~/.resume_refinery/sessions/acme-cloud_staff-engineer_2026-03-25/v1/`
which contains:

```
v1/
├── cover_letter.docx         ← Open these in Word
├── resume.docx
├── interview_guide.docx
├── cover_letter.md           ← Markdown source for diffing
├── resume.md
├── interview_guide.md
├── truth_review.json         ← Review results (JSON)
├── voice_review.json
├── ai_review.json
├── evidence_pack.json        ← Extracted evidence and gaps
└── voice_guide.json          ← Distilled voice rules
```

### 5. Refine

```bash
resume-refinery refine acme-cloud_staff-engineer_2026-03-25 \
  --doc cover_letter \
  --feedback "Lead with the Redis cost-saving story, not the migration."
```

A new version (`v2/`) is created with fresh documents, reviews, and exports.

### 6. Or use the web app instead

```bash
resume-refinery-web
# Open http://127.0.0.1:8765
```

Upload your three files, click **Generate**, and browse/refine sessions from the sidebar.

---

## Troubleshooting

### "Empty content" or "model may have exhausted its context window"

The model's context window is too small for your career profile plus the document being
generated. Raise the KV-cache size in `.env`:

```
RESUME_REFINERY_NUM_CTX=32768
```

Each 16K tokens ≈ 2 GB extra RAM. If you hit out-of-memory errors, lower it back to
`16384` and try a smaller model instead.

### Ollama not running / connection refused

Resume Refinery expects Ollama at `http://localhost:11434` by default. Check that:

```bash
ollama serve          # start the server
curl http://localhost:11434   # should return "Ollama is running"
```

If Ollama is on a different host or port, set `OLLAMA_BASE_URL` in your `.env`.

### Model not found

If you see an error about a missing model, pull it first:

```bash
ollama pull qwen3.5:9b
```

Replace `qwen3.5:9b` with whatever `RESUME_REFINERY_MODEL` is set to in your `.env`.

### JSON parse failures / garbled review results

Smaller models sometimes produce malformed JSON in review responses. Resume Refinery
normalises JSON output (strips markdown fences, repairs trailing commas, etc.) but very
small models (<4B parameters) can still produce unparseable output. Fixes:

- Use `qwen3.5:9b` or larger for the review model.
- If your generation model is small, set `RESUME_REFINERY_REVIEW_MODEL` to a more capable
  model separately.

### "WARNING: Ollama reviewer returned empty content"

The review model ran out of context or produced only `<think>` tags with no actual JSON.
Increase `RESUME_REFINERY_NUM_CTX` in your `.env`.

### Repair loop never converges

If the repair loop exhausts all passes without all reviews passing:

- **Truthfulness failures:** Check that your career profile actually contains the evidence
  the reviewer is looking for. The truthfulness gate never relaxes — if a claim isn't
  traceable to your career profile, it won't pass.
- **AI-detection flags:** Raise `RESUME_REFINERY_AI_FLAG_TOLERANCE` (e.g. to `4`) to allow
  more flags on later passes. Or lower `RESUME_REFINERY_RELAXED_PASS_START` (e.g. to `0`)
  to relax earlier.
- **Voice match stuck at "weak":** Improve your voice profile — add more writing samples
  and specific style notes. The models need concrete examples to match voice well.
- **Increase passes:** Raise `RESUME_REFINERY_MAX_REPAIR_PASSES` (e.g. to `5` or `7`).

### Review model == generation model warning

If the review model is the same as the generation model, you'll see a warning at startup.
This is fine for most uses, but the reviewer may be less likely to flag issues in text it
would have written itself. Set `RESUME_REFINERY_REVIEW_MODEL` to a different model if you
want stricter reviews.
