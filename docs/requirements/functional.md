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
| FR-2.1 | **Narrative creation** — The NarrativeAgent analyses the career profile against the job description and produces a `CandidacyNarrative` (thesis, argument pillars with career evidence, and gap framing). When the LLM is available, it generates a structured argument for why the candidate is a strong fit. A keyword-overlap fallback produces a basic narrative when the LLM is unavailable. Pillars are capped at 5. Each pillar gathers ALL solid supporting examples from the career profile (unlimited), ordered from strongest to weakest. Each evidence item includes a justification explaining why it supports the pillar. Evidence may overlap across pillars. After initial generation, a `NarrativeCriticAgent` evaluates the narrative against quality criteria and the NarrativeAgent revises it if issues are found. This critique loop runs up to `RESUME_REFINERY_MAX_NARRATIVE_CRITIQUE_PASSES` times (default 2). |
| FR-2.2 | **Voice extraction** — The VoiceAgent analyses the voice profile and produces a `VoiceStyleGuide` used to shape document tone. |
| FR-2.3 | **Drafting** — The DraftingAgent generates the resume. The document is produced in a single LLM call using thinking mode. |
| FR-2.3a | **Narrative coverage analysis** — _(Removed from orchestrator flow.)_ The NarrativeCoverageAgent class remains available for ad-hoc use but is no longer called during `create_session_run` or `generate_session_run`. Coverage analysis was found to be redundant with the narrative self-critique loop, which catches pillar/evidence gaps at generation time. |
| FR-2.4 | **Verification** — The truthfulness reviewer runs in a repair loop (up to `MAX_REPAIR_PASSES` iterations). After the loop completes, all eight independent reviewers (truthfulness, voice match, AI detection, hiring-manager, relevance pruning, ATS keyword alignment, grammar & mechanics, narrative coherence) run once as an advisory pass. Only truthfulness findings trigger repair; all other reviewer scores are informational. |
| FR-2.5 | **Repair** — The RepairAgent fixes documents that fail verification using surgical find/replace edits (see [convergence.md](convergence.md)). |
| FR-2.6 | **Iteration** — The truthfulness repair loop repeats up to `MAX_REPAIR_PASSES` times or until truthfulness passes. All other reviewers run once after the loop as advisory. |
| FR-2.7 | **Refinement** — The `refine` operation loads the prior version's reviewer findings (all eight reviewers) and passes them to the RepairAgent alongside the user's free-form feedback (single pass, no loop). Reviewer findings are labeled as reference-only context — the LLM only acts on them when the user's feedback explicitly requests it (e.g. "implement all hiring-manager suggestions except X"). They are never applied autonomously. After the repair pass, all eight reviewers run once on the result. The updated document is saved as a new version with review feedback. |
| FR-2.8 | _(Reserved)_ |

## FR-3 Outputs

| ID | Requirement |
|---|---|
| FR-3.1 | Each run produces a **resume** in both Markdown and DOCX format. |
| FR-3.2 | Documents are versioned (v1, v2, …) within a session directory. Each version includes Markdown source, DOCX export, and review JSON. |
| FR-3.3 | DOCX generation uses python-docx (no external Pandoc dependency). |
| FR-3.4 | The user must specify an **output directory** for generated DOCX files. In the CLI it is a required positional argument; in the web app it is a required text field. If the path is invalid (not a directory, parent does not exist) the system raises an error before generation begins. DOCX files are always also saved in the session version directory to preserve version history. |

## FR-4 Session Management

| ID | Requirement |
|---|---|
| FR-4.1 | Each job application creates a named session under `~/.resume_refinery/sessions/` (overridable via `RESUME_REFINERY_SESSIONS_DIR`). |
| FR-4.2 | Session context (candidacy narrative, voice guide, drafting context) can be saved and loaded for resumption. |
| FR-4.3 | Session naming is derived from company + role + date. |

## FR-5 Delivery

| ID | Requirement |
|---|---|
| FR-5.1 | Primary delivery is a local web application (FastAPI + browser). The **generate** and **refine** endpoints stream real-time progress to the browser using `StreamingResponse`. Each orchestrator step (narrative creation, document generation, review passes, repairs) is reported as it completes, with multi-line detail (review summaries, repair edits, false-positive acceptances) rendered in collapsible `<details>` blocks. On completion the page auto-redirects to the session view. |
| FR-5.2 | A CLI interface is also available for headless/scripted use. |
| FR-5.3 | The tool is pip-installable (`pip install -e .`). |
| FR-5.4 | After narrative extraction, the web app presents an editable **curate** page (`/sessions/{id}/curate`) where the user can revise the thesis, pillars (theme, argument, evidence), and gap framing before document generation begins. Edits are serialised as JSON and saved to the staging context so the drafting agent uses the user-revised narrative. |

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
| FR-7.8 | The system accepts document uploads (PDF, DOCX, TXT, MD) via a **Document Ingest** endpoint. An `IngestAgent` extracts structured career data using **one LLM call per document** (giving each file the full context window). The extraction prompt includes field-level guidance mirroring the wizard's helper text. After extraction, `consolidate_roles()` **(Pass 1)** immediately merges duplicate roles (matched by company + title + overlapping dates) so the user sees a clean timeline rather than one entry per source document. The user is then landed on the **Role Timeline** page (`needs_consolidation = True`) where they can verify, edit, delete, and add roles. When the user clicks **Finalize & Build Stories**, `consolidate_skills_meta()` **(Pass 2)** consolidates skills + education + certifications + domain knowledge + meta. After pass 2, a **fuzzy duplicate detection** step (`_has_duplicate_skills`) checks for remaining skill duplicates using normalised name matching and `SequenceMatcher` similarity (threshold 0.85); if duplicates are found, pass 2 is automatically re-run. `compose_stories()` then generates STAR behavioural stories from the merged accomplishments. Each role and story carries an `extraction_confidence` (`high`/`medium`/`low`) and `confidence_notes` field. If either consolidation pass fails, the original data for that pass is preserved. Both the ingest and finalize steps return **streamed HTML progress pages** that display each pipeline step in real time as it completes, then auto-redirect to the next page. |

## FR-6 Reviewers

| ID | Requirement |
|---|---|
| FR-6.1 | **Truthfulness reviewer** — Every factual claim in a document must be directly supported by evidence in the career profile or the job description. Claims not found in either source are flagged as unsupported. |
| FR-6.2 | **Voice reviewer** — Documents must match the writing style described in the voice profile. Per-document match strength is rated as "strong", "moderate", or "weak". Advisory only — never blocks convergence or triggers repair. |
| FR-6.3 | **AI detection reviewer** — Documents are scanned for phrases that commonly trigger AI-detection tools. Flagged phrases are listed per document. Advisory only — never blocks convergence or triggers repair. |
| FR-6.4 | All reviewers use JSON-formatted output, temperature 0, and thinking disabled to maximise determinism. |
| FR-6.5 | **Hiring-manager reviewer** — After the truthfulness loop and advisory pass, a simulated hiring-manager review evaluates the resume against the job description. It returns: an `advance_likelihood` percentage (0–100), strengths, concerns, and specific actionable improvement suggestions targeting the resume. The review is displayed on the session detail page in the web app and emitted via the progress callback during generation/refinement. Advisory only — never blocks convergence or triggers repair. |
| FR-6.6 | **Relevance-pruning reviewer** — Identifies bullets, sentences, sections, or entire role entries in the resume that do not meaningfully strengthen the applicant's case for the target role. Flags content as redundant, irrelevant, filler, low-impact, or space-wasting. Preserves content that demonstrates transferable skills, differentiation, or narrative coherence. Advisory only — never blocks convergence or triggers repair. |
| FR-6.7 | **ATS keyword alignment reviewer** — Reviews the resume against the job description and career profile. Flags missing high-priority keywords the candidate genuinely possesses but that are absent from the resume, phrasing mismatches (synonym vs. exact JD term), and keyword stuffing. Never flags skills the candidate lacks. Returns an `alignment_score` ("strong" / "moderate" / "weak") plus per-keyword issue lists. Advisory only — never blocks convergence or triggers repair. |
| FR-6.8 | _(Reserved — cross-document consistency reviewer removed; only one document type exists.)_ |
| FR-6.9 | **Grammar & mechanics reviewer** — Reviews the resume for grammar errors, tense inconsistency, punctuation problems, capitalisation issues, and formatting inconsistencies. Never flags intentional fragments, industry jargon, or stylistic preferences. Returns a `clean` boolean plus issue lists. Advisory only — never blocks convergence or triggers repair. |
| FR-6.10 | **Narrative coherence reviewer** — Checks that every point in the resume connects back to the candidacy narrative's thesis and pillars. Flags phrases that are disconnected from the narrative structure. Returns an `alignment` rating ("strong" / "moderate" / "weak") plus issue lists. Advisory only — never blocks convergence or triggers repair. Skipped when no candidacy narrative is available. |
