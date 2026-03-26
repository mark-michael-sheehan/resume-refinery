# Contributing to Resume Refinery

## Development Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd resume_refinery

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install runtime + dev dependencies
pip install -r requirements-dev.txt
pip install -e .

# 4. Configure environment
cp .env.example .env
# Edit .env as needed (see docs/usage.md for all variables)
```

## Running Tests

All tests run without a live Ollama instance — LLM calls are mocked.

```bash
pytest                                              # all tests
pytest tests/test_orchestrator.py -x                # single file, stop on first failure
pytest --cov=resume_refinery --cov-report=term-missing  # with coverage
```

The full suite must pass in under 15 seconds. If a change breaks tests, fix them before
committing. Always run `pytest tests/` after making changes — not just the directly
affected test file.

## Project Layout

```
src/resume_refinery/     ← Application code
tests/                   ← Unit tests (mirror src/ structure)
docs/                    ← Architecture, usage, requirements
examples/                ← Template input files
```

## Code Style

- **Python 3.12+** — use modern syntax (type unions with `|`, etc.)
- **Type hints on public APIs** — function signatures, method returns
- **Pydantic `BaseModel`** for all domain models
- **LLM calls go through `ollama.Client`** — never raw HTTP
- No auto-formatters are enforced yet; match the style of surrounding code

## How Tests Work

Tests use **fake agents** (defined in `tests/conftest.py` and per-test-file) that return
canned responses instead of calling Ollama. This makes tests fast and deterministic.

Key patterns:
- `FakeEvidenceAgent`, `FakeVoiceAgent`, `FakeDraftingAgent`, etc. — return static data
- `FakeVerificationAgent` — simulates review results, often toggling pass/fail across calls
- `FakeRepairAgent` — returns `RepairPassResult` with or without edits
- `monkeypatch.setenv("RESUME_REFINERY_SESSIONS_DIR", str(tmp_path))` — isolates session
  I/O to a temp directory

When writing new tests, prefer constructing fake agents over patching internal methods.

## Documentation Rules

When a code change affects behaviour, inputs, outputs, or contracts:
- Update the relevant docs in the **same commit**, not as a follow-up.
- Files to check:
  - `docs/architecture.md` — pipeline overview, module responsibilities, session layout
  - `docs/usage.md` — CLI commands, environment variables, output files, troubleshooting
  - `docs/requirements/functional.md` — functional requirements (FR-*)
  - `docs/requirements/non-functional.md` — non-functional requirements (NFR-*)
  - `docs/requirements/convergence.md` — convergence and repair loop requirements (CR-*)
- Keep requirement IDs stable — update the text, don't renumber.
- New environment variables must be added to `docs/usage.md` and `.env.example`.

## Adding a New Specialist Agent

1. Add the agent class to `src/resume_refinery/specialist_agents.py`.
2. Add any new models to `src/resume_refinery/models.py`.
3. Wire it into `ResumeRefineryOrchestrator.__init__()` and the relevant orchestration method.
4. Add a fake agent class to the test file and write tests.
5. Update `docs/architecture.md` with the new module responsibility.

## Adding a New Environment Variable

1. Add it to `.env.example` with a comment.
2. Read it via `os.environ.get()` with a default in the module that uses it.
3. Document it in `docs/usage.md` (Environment Variables section).
4. Add it to the table in `docs/requirements/non-functional.md` (NFR-4).
