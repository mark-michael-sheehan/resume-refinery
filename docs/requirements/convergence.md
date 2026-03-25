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

## CR-7 Per-Reviewer Suppression

| ID | Requirement |
|---|---|
| CR-7.1 | The repair agent may signal that a reviewer's finding is a false positive by populating one of three per-reviewer acceptance arrays in its output: `accepted_claims` (truthfulness), `accepted_ai_phrases` (AI-detection), `accepted_voice_issues` (voice). |
| CR-7.2 | The orchestrator maintains three independent suppression sets — one per reviewer — that accumulate accepted phrases across all repair passes within a single run. |
| CR-7.3 | Before each pass's gate check and repair call, raw reviewer results are filtered through the corresponding suppression set. Suppressed items are removed from flag/issue/claim lists; truthfulness `pass_strict` and `all_supported` are recalculated; AI `risk_level` is recalculated from the remaining flag count. Voice match levels are preserved as-is (they reflect holistic LLM judgment, not issue count). |
| CR-7.4 | A phrase accepted in any pass is suppressed for all subsequent passes in the same run. Suppression sets do not persist beyond a single `create_session_run` or `refine_session_run` call. |
| CR-7.5 | Each reviewer's suppression set is independent — accepting a voice false positive cannot suppress a truthfulness or AI-detection finding (and vice versa). |
