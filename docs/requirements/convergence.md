# Convergence Requirements

The review-repair loop must converge — i.e., documents should improve (or at least
not regress) with each pass. This document captures the design constraints that
ensure convergence.

## CR-1 Surgical Repair (not full regeneration)

| ID | Requirement |
|---|---|
| CR-1.1 | The repair step MUST NOT regenerate an entire document. Repairs are expressed as a list of `{find, replace, reason}` JSON edits produced by the LLM. |
| CR-1.2 | Edits are applied programmatically via string find/replace (`utils.apply_edits`), not by asking the LLM to produce a new complete document. |
| CR-1.3 | Edits are located in the original document, clustered by overlapping spans, and applied left-to-right with offset tracking (no re-find step). Overlapping edits are merged via a lightweight LLM call when possible; otherwise the leftmost edit in the cluster is kept. |
| CR-1.4 | If the number of edits that fail to locate in the document text exceeds `RESUME_REFINERY_EDIT_FAIL_THRESHOLD` (default 3), an `EditApplicationError` is raised. Overlapping-edit collisions are NOT counted as failures. Whitespace-normalized matching is attempted before declaring a locate failure. |

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
| CR-6.4 | Each pass runs all 8 reviewers concurrently (truthfulness, consistency, ATS, grammar, voice, AI detection, HM, pruning), checks all gates, and makes a single unified repair call if any gate fails. |
| CR-6.5 | The repair prompt includes prior-edit context with conflict resolution instructions (fix/merge/accept), replacing the former Phase B preserve note. Cross-reviewer regressions are handled by the prior-edits annotated pass-through system (CR-8). |
| CR-6.6 | The loop re-runs all reviewers after each repair, so any regression introduced by a repair is caught on the next pass. |

## CR-7 Per-Reviewer Suppression

| ID | Requirement |
|---|---|
| CR-7.1 | The repair agent may signal that a reviewer's finding is a false positive by populating one of seven per-reviewer acceptance arrays in its output: `accepted_claims` (truthfulness), `accepted_ai_phrases` (AI-detection), `accepted_voice_issues` (voice), `accepted_hm_issues` (hiring manager), `accepted_pruning_issues` (relevance pruning), `accepted_ats_issues` (ATS keyword), `accepted_consistency_issues` (consistency), `accepted_grammar_issues` (grammar). |
| CR-7.2 | The orchestrator maintains seven independent suppression sets — one per reviewer — that accumulate accepted phrases across all repair passes within a single run. |
| CR-7.3 | Before each pass's gate check and repair call, raw reviewer results are filtered through the corresponding suppression set. Suppressed items are removed from flag/issue/claim lists; truthfulness `pass_strict` and `all_supported` are recalculated; AI `risk_level` is recalculated from the remaining flag count; ATS `alignment_score` is recalculated from remaining missing/stuffing keywords; consistency `consistent` is recalculated from remaining issues; grammar `clean` is recalculated from remaining issue counts. Voice match levels are preserved as-is (they reflect holistic LLM judgment, not issue count). |
| CR-7.4 | A phrase accepted in any pass is suppressed for all subsequent passes in the same run. Suppression sets do not persist beyond a single `create_session_run` call. The `refine_session_run` call runs a single repair pass (no loop) so suppression does not apply. |
| CR-7.5 | Each reviewer's suppression set is independent — accepting a voice false positive cannot suppress a truthfulness or AI-detection finding (and vice versa). |
| CR-7.6 | Whenever the repair agent adds items to any acceptance list, the orchestrator emits an explicit progress message naming each accepted phrase/claim/issue and the reviewer it came from, before proceeding to the next pass. |
| CR-7.7 | At the end of each `create_session_run` call, if any items were exempted, the cumulative suppression sets are persisted to `exempted_phrases.json` in the active version directory as an `ExemptedPhrases` model (fields: `claims`, `ai_phrases`, `voice_issues`, `hm_issues`, `pruning_issues`, `ats_issues`, `consistency_issues`, `grammar_issues`). No file is written when no items were exempted. The `refine_session_run` call does not persist suppressions (single repair pass, no loop). |

## CR-8 Edit Region Tracking (Annotated Pass-Through)

| ID | Requirement |
|---|---|
| CR-8.1 | Every successful edit applied by `apply_edits` produces an `EditRegion(start, end, reviewer, pass_num)` recording the character span of the replacement text, the reviewer that triggered the edit, and the pass number. |
| CR-8.2 | Each `RepairEdit` stores the `reviewer` that triggered it, enabling per-edit attribution in prior-edit summaries. |
| CR-8.3 | Each reviewer has a numeric priority rank: truthfulness (80) > consistency (70) > ATS (60) > grammar (50) > voice (40) > AI (30) > HM (20) > pruning (10). |
| CR-8.4 | On pass 1+, the orchestrator builds a per-document "prior edits" summary from accumulated `RepairPassResult.edits`, listing each prior edit's reviewer, original text, replacement text, and reason. This summary is passed to the repair agent as prompt context — findings are NOT pre-filtered. |
| CR-8.5 | The repair prompt includes a "Prior Edits" section (between the Job Description and Review Findings) when prior edits exist, and the system prompt includes conflict resolution instructions (fix/merge/accept). |
| CR-8.6 | When a review finding targets text that was previously edited by a higher-priority reviewer, the repair agent decides: FIX (genuinely new concern), MERGE (satisfy both constraints), or ACCEPT (noise from prior edit). The orchestrator does not silently suppress findings. |
| CR-8.7 | Deletions (empty replacement text) do not produce edit regions because there is no replacement span to protect. They are shown as "DELETED" in prior-edit summaries. |
| CR-8.8 | The `RepairPassResult` model includes an `edit_regions` field (dict mapping document key to list of `EditRegion`s) alongside the existing `edits` field. |
| CR-8.9 | The `repair_unified` method determines the dominant reviewer by checking all reviewers in priority order (truthfulness > consistency > ATS > grammar > voice > AI) and tags all edits and edit regions with that reviewer's priority. |

## CR-9 Intra-Pass Edit Collision Resolution

| ID | Requirement |
|---|---|
| CR-9.1 | `apply_edits` locates each edit in the original document using exact match first, then a whitespace-normalized fallback (collapsing all whitespace runs to single spaces). |
| CR-9.2 | Located edits are sorted left-to-right and grouped into clusters of overlapping spans. Non-overlapping edits become singleton clusters. |
| CR-9.3 | For clusters with 2+ overlapping edits, a `merge_fn` callback is invoked with the union span text and the list of overlapping edits. The callback produces a single merged edit that satisfies all overlapping edits' intents. |
| CR-9.4 | `RepairAgent._merge_overlapping_edits` implements `merge_fn` via a lightweight LLM call using `MERGE_EDITS_SYSTEM_PROMPT`. The LLM receives the overlapping passage and all proposed edits, and returns a single merged `{find, replace, reason}`. The `find` is forced to the exact context text regardless of LLM output. |
| CR-9.5 | If `merge_fn` is not provided or returns `None`, only the first (leftmost) edit in the cluster is kept and the rest are skipped. |
| CR-9.6 | Collision-skipped edits are NOT counted toward the failure threshold — they are expected intra-batch overlaps, not match failures. |
| CR-9.7 | After collision resolution, edits are applied left-to-right using offset tracking (accumulated length delta from prior edits). No re-find is performed. |
| CR-9.8 | Duplicate `find` texts (two edits targeting the same string at the same position) are reassigned to successive occurrences of that string in the document. |
| CR-9.9 | `apply_edits` returns a 3-tuple `(document, edit_regions, failed_edits)`. The third element is a list of `EditOp` dicts for edits that failed Phase 1 locate (could not find the `find` text via exact or whitespace-normalized matching). |
| CR-9.10 | `RepairPassResult` includes a `failed_edits` field (dict mapping document key to list of failed edit dicts). The orchestrator emits a progress message listing each failed edit's `find` snippet and reason when any locate failures occur. |
