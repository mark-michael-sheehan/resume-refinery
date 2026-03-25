# Functional Requirements

## FR-1 Inputs

| ID | Requirement |
|---|---|
| FR-1.1 | The system accepts three Markdown files: **career profile**, **voice profile**, and **job description**. |
| FR-1.2 | Input files are treated as raw text — no rigid schema. Free-form content is preferred over structured fields. |
| FR-1.3 | Inputs are persisted inside the session directory (`inputs/`) for reproducibility. |

## FR-2 Pipeline

| ID | Requirement |
|---|---|
| FR-2.1 | **Evidence extraction** — The EvidenceAgent analyses the career profile against the job description and produces an `EvidencePack` (matched evidence + gaps). |
| FR-2.2 | **Voice extraction** — The VoiceAgent analyses the voice profile and produces a `VoiceStyleGuide` used to shape document tone. |
| FR-2.3 | **Drafting** — The DraftingAgent generates three documents: cover letter, resume, and interview guide. Each document is a separate LLM call using thinking mode. |
| FR-2.4 | **Verification** — The VerificationAgent runs three independent reviewers (truthfulness, voice match, AI detection) on each document. |
| FR-2.5 | **Repair** — The RepairAgent fixes documents that fail verification using surgical find/replace edits (see [convergence.md](convergence.md)). |
| FR-2.6 | **Iteration** — Verification and repair repeat up to `MAX_REPAIR_PASSES` times or until all documents pass. |

## FR-3 Outputs

| ID | Requirement |
|---|---|
| FR-3.1 | Each run produces three documents: **cover letter**, **resume**, and **interview guide** in both Markdown and DOCX format. |
| FR-3.2 | Documents are versioned (v1, v2, …) within a session directory. Each version includes Markdown source, DOCX export, and review JSON. |
| FR-3.3 | DOCX generation uses python-docx (no external Pandoc dependency). |

## FR-4 Session Management

| ID | Requirement |
|---|---|
| FR-4.1 | Each job application creates a named session under `~/.resume_refinery/sessions/` (overridable via `RESUME_REFINERY_SESSIONS_DIR`). |
| FR-4.2 | Session context (evidence pack, voice guide, drafting context) can be saved and loaded for resumption. |
| FR-4.3 | Session naming is derived from company + role + date. |

## FR-5 Delivery

| ID | Requirement |
|---|---|
| FR-5.1 | Primary delivery is a local web application (FastAPI + browser). |
| FR-5.2 | A CLI interface is also available for headless/scripted use. |
| FR-5.3 | The tool is pip-installable (`pip install -e .`). |

## FR-6 Reviewers

| ID | Requirement |
|---|---|
| FR-6.1 | **Truthfulness reviewer** — Every factual claim in a document must be directly supported by evidence in the career profile or the job description. Claims not found in either source are flagged as unsupported. |
| FR-6.2 | **Voice reviewer** — Documents must match the writing style described in the voice profile. Per-document match strength is rated as "strong", "moderate", or "weak". |
| FR-6.3 | **AI detection reviewer** — Documents are scanned for phrases that commonly trigger AI-detection tools. Flagged phrases are listed per document. |
| FR-6.4 | All reviewers use JSON-formatted output, temperature 0, and thinking disabled to maximise determinism. |
