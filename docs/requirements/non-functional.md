# Non-Functional Requirements

## NFR-1 Cost

| ID | Requirement |
|---|---|
| NFR-1.1 | The system runs entirely on a local Ollama instance — zero API costs. |
| NFR-1.2 | Low cost is prioritised over low latency. Thinking mode is used for generation and repair even though it increases wall-clock time. |

## NFR-2 Truthfulness

| ID | Requirement |
|---|---|
| NFR-2.1 | Every factual statement in generated documents MUST be directly traceable to the career profile or the job description. |
| NFR-2.2 | The system must never fabricate, embellish, or extrapolate achievements beyond what the career profile states. |
| NFR-2.3 | Truthfulness is the highest-priority review gate and never relaxes. |

## NFR-3 Model Compatibility

| ID | Requirement |
|---|---|
| NFR-3.1 | Default model: `qwen3.5:9b`. Generation and review models are independently configurable via environment variables. |
| NFR-3.2 | The system must handle models that wrap responses in `<think>` tags by stripping them before JSON parsing. |
| NFR-3.3 | JSON responses from reviewers must be normalised (trailing commas, extra text before/after JSON stripped) before parsing. |

## NFR-4 Configurability

All tuning knobs are exposed as environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `RESUME_REFINERY_MODEL` | `qwen3.5:9b` | Generation model |
| `RESUME_REFINERY_REVIEW_MODEL` | `qwen3.5:9b` | Review model |
| `RESUME_REFINERY_NUM_CTX` | `16384` | KV-cache size (tokens) |
| `RESUME_REFINERY_MAX_TOKENS` | `8192` | Max tokens for non-thinking generation |
| `RESUME_REFINERY_REVIEW_MAX_TOKENS` | `4096` | Max tokens for review calls |
| `RESUME_REFINERY_MAX_REPAIR_PASSES` | `3` | Max review+repair iterations |
| `RESUME_REFINERY_RELAXED_PASS_START` | `1` | Pass index where voice/AI relax |
| `RESUME_REFINERY_AI_FLAG_TOLERANCE` | `2` | Allowed AI flags on relaxed passes |
| `RESUME_REFINERY_EDIT_FAIL_THRESHOLD` | `3` | Max edit match failures before error |
| `RESUME_REFINERY_SESSIONS_DIR` | `~/.resume_refinery/sessions` | Session storage path |
| `RESUME_REFINERY_CAREERS_DIR` | `~/.resume_refinery/careers` | Career repository storage path |
| `RESUME_REFINERY_MAX_WORKERS` | `1` | Max concurrent threads for independent LLM calls |

## NFR-5 Testability

| ID | Requirement |
|---|---|
| NFR-5.1 | All specialist agents are unit-testable with mocked LLM clients. |
| NFR-5.2 | The full test suite must run in under 15 seconds with no network calls. |
| NFR-5.3 | Tests must not depend on a running Ollama instance. |

## NFR-6 Delivery

| ID | Requirement |
|---|---|
| NFR-6.1 | pip-installable (`pip install -e .`). |
| NFR-6.2 | Primary UI is a local web application served by FastAPI. |
| NFR-6.3 | No external cloud services required. |
