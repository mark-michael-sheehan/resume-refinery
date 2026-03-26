# Project Guidelines

## Documentation and Requirements

When a code change affects behavior, inputs, outputs, or contracts of the system, update the relevant documentation and requirements files as part of the same change. Do not treat documentation updates as a separate follow-up task.

Files to check:
- `docs/architecture.md` — pipeline overview, module responsibilities, design decisions
- `docs/usage.md` — CLI commands, environment variables, library usage examples
- `docs/requirements/functional.md` — functional requirements (FR-*)
- `docs/requirements/non-functional.md` — non-functional requirements (NFR-*)
- `docs/requirements/convergence.md` — convergence and repair loop requirements (CR-*)

Rules:
- If a function signature, prompt template, or reviewer input changes, update the corresponding requirement and any architecture/usage docs that describe that behavior.
- If a new environment variable or configuration knob is added, add it to `docs/usage.md`.
- If a new module or agent is introduced, add it to `docs/architecture.md`.
- Keep requirement IDs stable — update the text of existing IDs rather than renumbering.

## Testing

All code changes must pass `pytest tests/` before being considered complete. Run the full test suite after making changes, not just the directly affected test file.

## Code Style

- Python 3.12+, type hints on public APIs.
- Models use Pydantic `BaseModel`.
- LLM calls go through Ollama client (`ollama.Client`), never raw HTTP.
