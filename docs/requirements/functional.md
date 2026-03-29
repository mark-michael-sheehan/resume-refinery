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

## FR-7 Career Repository

| ID | Requirement |
|---|---|
| FR-7.1 | The system provides a **Career Builder** — a guided, multi-phase web wizard that elicits structured career data (identity, roles, skills, STAR stories, strategy, voice). |
| FR-7.2 | Career data is stored as a `CareerRepository` model persisted in `~/.resume_refinery/careers/<repo_id>/career.json`. Override with `RESUME_REFINERY_CAREERS_DIR`. |
| FR-7.3 | A `CareerRepository` can be flattened into a `CareerProfile` (`to_career_profile()`) and used as a direct replacement for a file-uploaded career profile in session creation. |
| FR-7.4 | The wizard uses HTMX for partial-page updates. No JavaScript build step is required. |
| FR-7.5 | Each phase saves progress incrementally — the user can stop and resume at any point. |
| FR-7.6 | An `ElicitationAgent` uses the LLM to analyse role answers and generate contextual follow-up probes. Falls back to static heuristic probes when the LLM is unavailable. The probe endpoint returns HTML fragments swapped into the page via HTMX. |
| FR-7.7 | The session creation form in the web app allows selecting a saved career repository instead of uploading files. |
| FR-7.8 | The system accepts document uploads (PDF, DOCX, TXT, MD) via a **Document Ingest** endpoint. An `IngestAgent` extracts structured career data from the combined document text using a single LLM call and populates a new `CareerRepository` as a first-pass starting point. The user reviews and refines the extracted data via the standard wizard phases. |

## FR-6 Reviewers

| ID | Requirement |
|---|---|
| FR-6.1 | **Truthfulness reviewer** — Every factual claim in a document must be directly supported by evidence in the career profile or the job description. Claims not found in either source are flagged as unsupported. |
| FR-6.2 | **Voice reviewer** — Documents must match the writing style described in the voice profile. Per-document match strength is rated as "strong", "moderate", or "weak". |
| FR-6.3 | **AI detection reviewer** — Documents are scanned for phrases that commonly trigger AI-detection tools. Flagged phrases are listed per document. |
| FR-6.4 | All reviewers use JSON-formatted output, temperature 0, and thinking disabled to maximise determinism. |
