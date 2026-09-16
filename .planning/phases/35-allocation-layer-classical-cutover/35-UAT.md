---
status: testing
phase: 35-allocation-layer-classical-cutover
source: [35-VERIFICATION.md]
started: 2026-09-13T09:30:00Z
updated: 2026-09-16T15:31:32Z
---

## Current Test

number: 4
name: Decide whether an in-place `Observatory` position/timezone correction must re-mint already-projected allocation nights (round-5 verifier escalation)
expected: |
  DECISION, not a manual test. Correcting an `Observatory` row's `lat`/`lon`/`altitude`/`timezone` in the Django admin WITHOUT changing `run.site` leaves every allocation night already projected at that site permanently stale: the verifier reproduced `ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0)` with boundaries unchanged at 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00 while the corrected site's true sunset/sunrise are 2026-07-09 07:20:39 / 20:57:12 (~15 h, silent, uncounted). The recorded token was `v2|1|none|none` before and after because it carries `site_id`, not the site's coordinates. Decide: fix in round 6 (alongside 35-REVIEW.md iteration-9 CR-04/CR-05), file as a follow-up, or defer with WR-01/WR-02/WR-03.
awaiting: user response

## Tests

### 1. Cutover before/after diff in the reconciler demo notebook and the one surviving blank-url row
expected: Open docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb and read the executed cutover before/after table and the unexplained list. The before/after numbers are real figures from a copy of the developer database (241 -> 233 total, 56 -> 0 RUN:{pk}:{date}, 16 -> 16 bare RUN:{pk} containers, 10 -> 1 blank-url, 0 -> 57 ALLOC:, 159 -> 159 facility-url), the four end-state assertion cells executed without raising, and the single remaining blank-url row (pk=334, title 'tmp') is one you recognise as pre-existing junk rather than a real observing night the cutover failed to convert.
why_human: Whether the unexplained list contains only rows an operator recognises is a judgement about what the calendar MEANS on this specific database, not a property any test can assert. Harvested from 35-06-PLAN.md task 3 and 35-07-PLAN.md task 3 <human-check> blocks (deferred to end-of-phase).
result: pass

### 2. Three-group reconciliation sums in the 35-06 real-database cutover record
expected: Read the 35-06 real-database cutover record in 35-VALIDATION.md ('Manual-Only Verifications' table) and confirm 48 rekeyed + 8 legacy_deleted + 0 retired-by-observation = 56, which is exactly the before-count of RUN:{pk}:{date} events; the bare RUN:{pk} container count is unchanged at 16; all 159 facility-url-keyed observation events are reported byte-identical. The 8 containers whose stale pre-Phase-33 title this first post-D-12 sweep corrected read as a pre-existing-staleness correction, not a Phase 35 regression.
why_human: The arithmetic is checkable but the judgement -- that the 8 corrected container titles are acceptable churn rather than an unwanted rewrite -- is an operator call about the real calendar.
result: pass

### 3. Judgment-tier prohibition verdicts
expected: Review the seven judgment-tier prohibitions listed in the 'Prohibitions' section of 35-VERIFICATION.md and confirm each verdict: each prohibition is upheld by the cited code and test evidence.
why_human: unverified-prohibition -- human review recommended. Autonomous verify records a NON-AUTHORITATIVE LLM-judge verdict for judgment-tier prohibitions; these are never silently passed.
result: pass

### 4. Decide whether an in-place `Observatory` position/timezone correction must re-mint already-projected allocation nights (round-5 verifier escalation)
expected: DECISION, not a manual test. Correcting an `Observatory` row's `lat`/`lon`/`altitude`/`timezone` in the Django admin WITHOUT changing `run.site` leaves every allocation night already projected at that site permanently stale: the verifier reproduced `ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0)` with boundaries unchanged at 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00 while the corrected site's true sunset/sunrise are 2026-07-09 07:20:39 / 20:57:12 (~15 h, silent, uncounted). The recorded token was `v2|1|none|none` before and after because it carries `site_id`, not the site's coordinates. Decide: fix in round 6 (alongside 35-REVIEW.md iteration-9 CR-04/CR-05), file as a follow-up, or defer with WR-01/WR-02/WR-03.
why_human: Same defect signature as iteration 8's CR-02 one input further out, but no must-have as worded is falsified (35-21 truth 2 defines the boundary inputs as `(night_start_utc, night_end_utc, run.site, night)` and the token does carry `run.site`); entirely pre-existing and narrowed by this round; reachable only by a manual admin edit of a site definition after projection (`MPCObscodeFetcher.to_observatory()` builds new rows, never updates existing ones). Product decision, not a verification call. Source: 35-VERIFICATION.md (round-5 re-verification) human_verification.
result: [pending]

## Summary

total: 4
passed: 3
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

_None recorded yet. Note: the deep code review committed as `35-REVIEW.md` (`eb6a595`) reports 6 reproduced
blockers (CR-01..CR-06) and 11 warnings that the automated verification did not surface; resolve or triage
those before treating a passing UAT as phase completion._
