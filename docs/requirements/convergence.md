# Convergence Requirements

The review-repair loop must converge — i.e., documents should improve (or at least
not regress) with each pass. This document captures the design constraints that
ensure convergence.

## CR-1 Surgical Repair (not full regeneration)

| ID | Requirement |
|---|---|
| CR-1.1 | The repair step MUST NOT regenerate an entire document. Repairs are expressed as a list of `{find, replace, reason}` JSON edits produced by the LLM. |
| CR-1.2 | Edits are applied programmatically via string find/replace (`utils.apply_edits`), not by asking the LLM to produce a new complete document. |
| CR-1.3 | Edits are applied in reverse document order to prevent offset drift. |
| CR-1.4 | If the number of edits that fail to match the document text exceeds `RESUME_REFINERY_EDIT_FAIL_THRESHOLD` (default 3), an `EditApplicationError` is raised. |

## CR-2 Reviewer Determinism

| ID | Requirement |
|---|---|
| CR-2.1 | All reviewers pin `temperature=0` to minimise non-determinism between passes. |
| CR-2.2 | Reviewers use `think=False` and `format="json"` so that raw output is parseable without stripping thinking tags. |
| CR-2.3 | The truthfulness reviewer is the strictest gate — it NEVER relaxes. It receives both the career profile and the job description as grounding sources. Voice and AI detection may relax on later passes (see CR-3). |

## CR-3 Acceptance Thresholds

| ID | Requirement |
|---|---|
| CR-3.1 | Voice: "moderate" or "strong" per-document match is accepted from pass 0 onward. |
| CR-3.2 | AI detection: on passes before `RELAXED_PASS_START`, cover letter + resume must have zero flags. From `RELAXED_PASS_START` onward, total flags ≤ `AI_FLAG_TOLERANCE`. |
| CR-3.3 | Truthfulness: `pass_strict=True` is required on every pass. No relaxation. |
| CR-3.4 | Interview guide is exempt from voice and AI-detection reviews (it is personal preparation, not a submitted document). |
| CR-3.5 | Hiring-manager review is advisory only — it feeds findings into the repair agent but NEVER blocks convergence. This prevents feedback loops where the HM asks for bolder claims that the truthfulness reviewer then rejects. |
| CR-3.6 | Relevance-pruning review is advisory only — it feeds findings into the repair agent but NEVER blocks convergence. |
| CR-3.7 | ATS keyword alignment: "strong" or "moderate" `alignment_score` is accepted. "weak" blocks convergence and triggers repair. |
| CR-3.8 | Cross-document consistency: `consistent=True` is accepted. Any contradictions (`consistent=False`) block convergence and trigger repair. |
| CR-3.9 | Grammar & mechanics: on passes before `RELAXED_PASS_START`, `clean=True` is required (zero issues). From `RELAXED_PASS_START` onward, total issues ≤ 2 is accepted. |

## CR-4 Feedback Hygiene

| ID | Requirement |
|---|---|
| CR-4.1 | Previous suggestions are cleared at the start of each pass and only the most recent pass's suggestions are retained. This prevents unbounded prompt growth. |
| CR-4.2 | Stale suggestions are listed under "Previously attempted fixes" in the repair prompt so the LLM tries a different approach. |
| CR-4.3 | Per-document truthfulness suggestions are de-duplicated against the previous suggestions list (case-insensitive). |

## CR-5 Repair Prompt Alignment

| ID | Requirement |
|---|---|
| CR-5.1 | The repair system prompt (`REPAIR_SYSTEM_PROMPT`) embeds the exact criteria used by each reviewer. This ensures the repairer "knows" the same rules as the reviewers, eliminating the reviewer/repairer divergence problem. |
| CR-5.2 | Review findings sent to the repair LLM include verbatim quotes from the reviewers (unsupported claims, off-voice phrases, AI-flagged phrases). |
| CR-5.3 | The repair LLM uses thinking mode (`think=True`) and unlimited token generation (`num_predict=-1`) to reason carefully about edits. |

## CR-6 Pass Limits

| ID | Requirement |
|---|---|
| CR-6.1 | The maximum number of review+repair passes is bounded by `RESUME_REFINERY_MAX_REPAIR_PASSES` (default 3). |
| CR-6.2 | If all documents pass all reviewers on any pass, the loop exits early. |
| CR-6.3 | If the loop exhausts all passes without convergence, the best version so far is kept and a warning is logged. |
| CR-6.4 | Each outer pass runs two sequential phases: Phase A (hard-gate: truthfulness, consistency, ATS, grammar) and Phase B (soft-gate: voice, AI detection, HM, pruning). Each phase runs its reviewers concurrently, checks gates, and repairs only if its gates fail. |
| CR-6.5 | Phase B repair receives a `preserve_instructions` note instructing the LLM not to alter text corrected by Phase A, reducing cross-phase regressions. |
| CR-6.6 | The outer loop re-runs both phases, so any Phase B regression of a Phase A fix is caught and re-repaired on the next pass. |

## CR-7 Per-Reviewer Suppression

| ID | Requirement |
|---|---|
| CR-7.1 | The repair agent may signal that a reviewer's finding is a false positive by populating one of seven per-reviewer acceptance arrays in its output: `accepted_claims` (truthfulness), `accepted_ai_phrases` (AI-detection), `accepted_voice_issues` (voice), `accepted_hm_issues` (hiring manager), `accepted_pruning_issues` (relevance pruning), `accepted_ats_issues` (ATS keyword), `accepted_consistency_issues` (consistency), `accepted_grammar_issues` (grammar). |
| CR-7.2 | The orchestrator maintains seven independent suppression sets — one per reviewer — that accumulate accepted phrases across all repair passes within a single run. |
| CR-7.3 | Before each pass's gate check and repair call, raw reviewer results are filtered through the corresponding suppression set. Suppressed items are removed from flag/issue/claim lists; truthfulness `pass_strict` and `all_supported` are recalculated; AI `risk_level` is recalculated from the remaining flag count; ATS `alignment_score` is recalculated from remaining missing/stuffing keywords; consistency `consistent` is recalculated from remaining issues; grammar `clean` is recalculated from remaining issue counts. Voice match levels are preserved as-is (they reflect holistic LLM judgment, not issue count). |
| CR-7.4 | A phrase accepted in any pass is suppressed for all subsequent passes in the same run. Suppression sets do not persist beyond a single `create_session_run` or `refine_session_run` call. |
| CR-7.5 | Each reviewer's suppression set is independent — accepting a voice false positive cannot suppress a truthfulness or AI-detection finding (and vice versa). |
| CR-7.6 | Whenever the repair agent adds items to any acceptance list, the orchestrator emits an explicit progress message naming each accepted phrase/claim/issue and the reviewer it came from, before proceeding to the next pass. |
| CR-7.7 | At the end of each `create_session_run` or `refine_session_run` call, if any items were exempted, the cumulative suppression sets are persisted to `exempted_phrases.json` in the active version directory as an `ExemptedPhrases` model (fields: `claims`, `ai_phrases`, `voice_issues`, `hm_issues`, `pruning_issues`, `ats_issues`, `consistency_issues`, `grammar_issues`). No file is written when no items were exempted. |
