---
status: complete
phase: 35-allocation-layer-classical-cutover
source: [35-VERIFICATION.md]
started: 2026-09-13T09:30:00Z
updated: 2026-09-16T21:10:00Z
---

## Current Test

[testing complete]

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
result: pass
reported: "fix in round 6"
resolution: "Diagnosis (.planning/debug/observatory-edit-leaves-nights-stale.md) showed round 6 (35-23/35-24) had already carried out the 'fix in round 6' decision, including the owner's >1 min re-mint threshold; owner closed G-35-4 as already satisfied on 2026-09-16."

### 5. Decide what to do about a declined re-mint writing a provenance token for boundaries it did not prove (round-6 verifier, seam between plans 35-23 and 35-24)
expected: DECISION, not a manual test. Project a fully-set sub-night run (`night_start_utc=23:00`, `night_end_utc=05:00`); confirm its companion row; then edit the sub-night start to `22:00` AND correct the same `Observatory` row's position in place; reconcile. Verifier's probe at HEAD `0bc1ccd`: `token_before=v3|1|5884a60fe2946a56|23:00:00|05:00:00`, result `updated=1, remint_declined=1, retired=0`, boundaries and pk preserved (correct), but `token_after=v3|1|f49f304a6749ba00|22:00:00|05:00:00` -- plan 35-23's CR-04 fall-through reaches plan 35-24's update-path refresh, which records the run's CURRENT token onto a night whose `start_time` is still the 23:00-derived value. Contradicts 35-24's prohibition 2 ("on the declined path nothing is recorded at all"); the comment at `allocation_projector.py:1362-1368` ("step 1 found them equal on THIS SAME sweep") is false on this path. Consequence checked, not assumed: none observable -- a fully-set run never consults the token and the second sweep still reports `remint_declined=1`. Options: fix now (guard the `_record_sub_night_provenance()` call at `:1441` on "the re-mint was not declined"), file as a follow-up, or accept and correct the two comments. 35-REVIEW.md iteration 10 reports the same seam as WR-01.
why_human: No must-have as worded is falsified and there is no demonstrated behavioural consequence; it is a reproduced prohibition counterexample at the seam between two plans in the same round that neither plan's tests cover. Product/priority decision, not a verification call. Source: 35-VERIFICATION.md (round-6 re-verification) human_verification #1.
result: skipped
reason: "Deferred follow-up: Observatory's positions are read once from the MPC site code API and almost never changed. So this collision between editing a position, which would change the rise/set times and night starts feels like a 'tip of the icecube' problem that will basically never happen"

### 6. Acknowledge that CR-05 permanently narrows ROADMAP Success Criterion 3 / ALLOC-03
expected: ACKNOWLEDGEMENT, not a manual test. A night whose companion row a person has CONFIRMED is no longer deleted when an `ObservationRecord` links to it; the calendar then shows both that allocation night and the observation's own entry until the confirmation is cleared and the sweep re-run. SC-3 as written says linking removes the night, full stop. The exception follows the UAT-2026-09-09 "human outranks machine" decision and was demanded by 35-REVIEW.md iteration 9's CR-05 (the unguarded delete destroyed a confirmation, both observation links and `is_verified` through a CASCADE, silently, counted as ordinary `retired` work). It is counted under `detach_declined`, logged with a named warning, documented in the runbook's `detach_declined` section with its remedy, and demonstrated in the notebook's executed cell 20. The default (unconfirmed) path is unchanged and still retires; unlink-restores is untouched. Confirm you accept the narrowed SC-3 behaviour, or say whether ALLOC-03 / SC-3 should be reworded to carry the exception explicitly.
why_human: A roadmap success criterion is now deliberately narrower than its wording; only the owner can accept the narrowing or ask for the criterion to be reworded. Source: 35-VERIFICATION.md (round-6 re-verification) human_verification #2.
result: pass

## Summary

total: 6
passed: 5
issues: 0
pending: 0
skipped: 1
blocked: 0

## Deferred Follow-Ups

- test: 5
  idea: "Observatory's positions are read once from the MPC site code API and almost never changed. So this collision between editing a position, which would change the rise/set times and night starts feels like a 'tip of the icecube' problem that will basically never happen"
  deferred_at: 2026-09-16
  context: "Declined re-mint records the run's current sub-night token onto a night whose start_time is still the pre-edit value (allocation_projector.py:1362-1368 comment false on this path; _record_sub_night_provenance() call at :1441 unguarded). No observable consequence; same seam as 35-REVIEW.md iteration 10 WR-01."

## Gaps

- gap_id: G-35-4
  truth: "Correcting an `Observatory` row's `lat`/`lon`/`altitude`/`timezone` in place (without changing `run.site`) causes the next reconcile sweep to re-mint already-projected allocation nights at that site with boundaries derived from the corrected coordinates, counted and logged, rather than leaving them silently stale with `unchanged=1`."
  status: resolved
  resolved_by: 35-24-PLAN.md (with 35-23-PLAN.md; both executed, SUMMARYs present)
  resolved_at: 2026-09-16
  resolution: "Owner decision (UAT 2026-09-16, after diagnosis): close as already satisfied -- the truth and the >1 min threshold hold at HEAD 5b0f431; the residual warning-text mismatch stays as 35-VERIFICATION.md advisory #1, no round-7 fix plan."
  reason: "User reported: fix in round 6"
  severity: major
  test: 4
  root_cause: "Already closed in code before diagnosis: round-6 plans 35-23/35-24 (up to 0bc1ccd) added the v3 position-fingerprinted token and a fall-through in _span_needs_remint() that recomputes the night boundaries and re-mints only when they differ from the stored ones by more than _UNRECORDED_PROVENANCE_TOLERANCE (1 minute) -- the owner's >1 min threshold is already in place. Debugger's probe at HEAD 5b0f431 on the round-5 fixture: ReconcileResult(created=1, retired=1) with boundaries 2026-07-09 07:20:39 -> 20:57:12 (= corrected site's sun events) plus a warning; a few-metre tweak (~1 s drift) reports unchanged=1 with no churn; a confirmed companion row is declined under remint_declined with two warnings per sweep (the owner-accepted 'human outranks machine' rule re-affirmed by UAT Test 6). The round-5 reproduction no longer reproduces. Remaining delta is message accuracy only: the step-4 staleness warning at allocation_projector.py:739-750 hard-codes 'unrecorded-provenance night' for a branch that is also reached by fully-recorded, repositioned nights, so an operator who corrected a site position is pointed at the runbook's legacy-audit reason instead of the site-correction paragraph. Already recorded as advisory #1 / an Anti-Pattern row in 35-VERIFICATION.md."
  artifacts:
    - path: "solsys_code/allocation_projector.py"
      issue: "lines 739-750: warning text says 'unrecorded-provenance night' on both entry paths of the step-4 boundary comparison; the trusted-token-but-repositioned path deserves its own wording (e.g. 'site position changed')"
    - path: "solsys_code/allocation_projector.py"
      issue: "line 77: _UNRECORDED_PROVENANCE_TOLERANCE now serves two callers; name too narrow (cosmetic)"
  missing:
    - "Branch the step-4 warning text on which entry path reached it (a token_trusted-and-fingerprint-differed flag is already in scope) so a site-position correction is described as such"
    - "If planned: regenerate docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb (cell 22 prints the old wording) and touch the runbook's retired reason (5) / site-definition paragraphs only if operator-visible wording changes"
  debug_session: ".planning/debug/observatory-edit-leaves-nights-stale.md"
  regression_guards: "solsys_code/tests/test_allocation_projector.py: test_in_place_observatory_correction_remints_to_the_corrected_positions_sun_event (:2387), test_changing_the_sites_position_in_place_remints (:3021), test_a_boundary_exactly_at_the_tolerance_resolves_as_correct (:2888), test_a_boundary_one_microsecond_beyond_the_tolerance_remints (:2905)"
  notes: "Owner decision (UAT 2026-09-16): fix in round 6 alongside 35-REVIEW.md iteration-9 CR-04/CR-05. Owner's use case, verbatim: 'someone might create an Observatory with a rough position in a hurry with runs associated with it and the Observatory position could be refined later which would be worth a re-sweep and reproject with new times if the positions changed enough to make >1 minute differences'. Design constraint from that: the re-mint must trigger only when the recomputed night boundaries differ from the projected ones by more than 1 minute -- a sub-minute drift from a trivial coordinate tweak must NOT churn every night at the site. Verifier reproduction: ReconcileResult(created=0, updated=0, unchanged=1, retired=0) with boundaries 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00 while the corrected site's true sunset/sunrise are 2026-07-09 07:20:39 / 20:57:12 (~15 h stale). Token `v2|1|none|none` unchanged before/after because it carries `site_id`, not the site's coordinates."
