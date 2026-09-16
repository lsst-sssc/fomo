---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-16T19:14:26Z
status: human_needed
score: 253/253 must-haves verified
covered_files:
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-08-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-08-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-09-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-09-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-10-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-10-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-11-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-11-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-12-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-12-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-13-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-13-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-14-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-14-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-15-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-15-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-16-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-16-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-17-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-17-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-18-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-18-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-19-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-19-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-20-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-20-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-21-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-21-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-22-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-22-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-23-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-23-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-24-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-24-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-25-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-25-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-CONTEXT.md"
  - "solsys_code/admin.py"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/cutover_classical_allocations.py"
  - "solsys_code/management/commands/load_telescope_runs.py"
  - "solsys_code/management/commands/reconcile_campaign_runs.py"
  - "solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py"
  - "solsys_code/models.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/telescope_runs.py"
  - "solsys_code/tests/test_allocation_projector.py"
  - "solsys_code/tests/test_allocation_projector_signals.py"
  - "solsys_code/tests/test_campaign_reconciler.py"
  - "solsys_code/tests/test_cutover_classical_allocations.py"
  - "solsys_code/tests/test_load_telescope_runs.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_reconcile_campaign_runs.py"
  - "solsys_code/tests/test_telescope_runs.py"
covered_digest: "v1:sha256:0b2b19e711eb3d999195e804b40e29c235bed3967a9538e8bb9c790e1d524360"
behavior_unverified: 0
overrides_applied: 0
flagged_prohibitions: 1
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
  notes:
    - "D-13 is honored with ONE stated, bounded and tested exception introduced by plan 35-24: a single `sun_event(kind='dark')` call on the plain-update path, fired only on the transition where a recorded current-format token proves the site component moved AND both sub-night fields are set. D-13's core promise (an existing night's `start_time`/`end_time` are never rewritten in place; no astropy on an idempotent re-reconcile) is intact and pinned by `TestNoSunEventRecompute` staying green unedited in my own run."
re_verification:
  previous_status: human_needed
  previous_score: 221/221
  previous_verified: 2026-09-16T15:26:19Z
  gap_closure_plans: ["35-23", "35-24", "35-25"]
  gaps_closed:
    - "Round-5 escalation (35-VERIFICATION.md 'Human Verification Required' #1 / 35-UAT.md test 4 -- an in-place `Observatory` position/timezone correction left already-projected nights permanently stale, reported as `unchanged`) -- CLOSED, and closed in the source, not in a SUMMARY. I re-ran the round-5 verifier's own probe myself against a migrated Django test database at HEAD `0bc1ccd`: with the SAME fixture (a null/null run at La Silla, night 2026-07-09, stored boundaries `2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00`) and the SAME in-place correction (`lat`/`lon`/`altitude`/`timezone` edited on that Observatory row, `run.site` never reassigned), the sweep now reports `ReconcileResult(created=1, ..., retired=1)`, a NEW primary key, and boundaries `2026-07-09 07:20:39+00:00 -> 2026-07-09 20:57:12+00:00` -- equal to the corrected position's live `sun_event()` values. The mechanism is real: `_PROVENANCE_TOKEN_VERSION = 'v3'` (`allocation_projector.py:99`), `_site_position_fingerprint()` (`:102-147`, SHA-256 over `repr(lat), repr(lon), repr(altitude), repr(timezone)`, truncated to 16 hex characters), a five-part token (`:520`), and a COMPONENT-WISE comparison at `:710-725` that routes a fingerprint-only difference to step 4's one-`sun_event()` resolution rather than to an outright re-mint. Probe module deleted afterwards; `git status --porcelain -- solsys_code/ src/ docs/` empty."
    - "35-REVIEW.md iteration 9 CR-04 (a declined re-mint also declined the non-destructive label refresh, freezing a night's title forever) -- CLOSED. The `continue` is gone; the decline branch (`:1256-1280`) increments `remint_declined` and falls through to the plain-update path (`:1320-1442`), which writes `title`/`description`/`target_list` only. `TestDeclinedRemintStillUpdatesLabels` (5 tests) green in my own run, and my own independent probe observed `remint_declined=1, updated=1` with the primary key and BOTH boundaries byte-unchanged on the same sweep."
    - "35-REVIEW.md iteration 9 CR-05 (the retirement branch's own `existing.delete()` was unguarded) -- CLOSED. `:1182-1208` now applies `_clearable_declined_and_unattributed()` to `existing` itself, counts a decline under `detach_declined`, logs a warning, and only increments `retired` when the night actually went away. Deliberately NOT `_remint_decline_reason()` -- only `confirmed_by` declines a retirement, which `test_is_verified_false_with_no_confirmation_still_retires` pins. `TestRetirePathAllocationEventGuard` (6 tests) green in my own run; the notebook's executed cell 20 proves it fires from the `CampaignRunObservation` post_save receiver alone, with no explicit `reconcile_run()` call."
    - "35-REVIEW.md iteration 9 WR-06 (one counter, two meanings, both printed messages wrong for one of them) -- CLOSED by a real split, threaded end to end: `ReconcileResult.remint_declined` (`campaign_reconciler.py`), `totals['remint_declined']` (`allocation_projector.py:1097`, `:1276`), the command's accumulator, per-run stderr line and BOTH summary lines (`reconcile_campaign_runs.py`), and a staff message in `campaign_views.py`. `detach_declined`'s own printed wording is byte-identical (verified by grep). Proven at the command level by `test_real_sweep_reports_remint_declined_for_a_human_confirmed_alloc_night` and by the notebook's executed cell 21, which shows a real `remint_declined: 1` in a genuine summary line."
    - "35-REVIEW.md iteration 9 WR-05 (the 'a site correction re-mints' claim was false for a fully-set sub-night run) -- CLOSED in code, docstring and runbook. `_site_provenance_differs()` (`:757-801`) plus the guarded refresh at `:1369-1396` correct the one site-derived field a fully-set run can still reach; `_span_needs_remint()`'s step-2 docstring (`:561-576`) now states the truth including the cross-timezone remedy; `TestSetWindowSiteCorrection` (6 tests) green in my run."
    - "35-REVIEW.md iteration 9 WR-07 (a 'once ever' cost bound that was silently false) -- CLOSED as a stated, bounded acceptance: the `Cost bound` paragraph (`:647-668`) now names both escapes (a declined night resolves once PER SWEEP forever; a `--dry-run` repeats every invocation, WR-01, explicitly not claimed fixed). `TestDeclinedNightResolutionCostIsBounded` green; my own probe's second sweep independently reported `remint_declined=1` again with one resolution, matching the documented bound."
    - "35-REVIEW.md iteration 9 WR-08 (`is_verified` documented as vestigial while being a load-bearing veto) -- CLOSED. `models.py` class docstring, `verbose_name` and a new `help_text` all state the re-mint veto AND the retirement non-veto; migration `0021_alter_calendareventmeta_is_verified_and_more.py` carries both `AlterField`s (and the `max_length` 64 -> 128 widening) with NO `RunPython`. The runbook quotes the shipped `verbose_name` verbatim."
    - "IN-05 (a redundant `active_urls.add(url)` under a false 'load-bearing' comment) -- CLOSED, and the executor's deviation analysis is CORRECT, not a dodge: I checked the three call sites myself. The decline branch's call is gone (`:1277-1280` is now a comment naming the earlier unconditional add at `:1215`); the two that remain are `:1115` (the BLOCKED branch, which `continue`s before `:1215` -- removing it would let the D-14 convergence step at `:1456` delete a foreign-owned night) and `:1215` itself. The plan's expected grep count of 1 was stale; the correct count is 2."
    - "CLAUDE.md paired-docs obligation for the round -- DISCHARGED. `reconcile_campaign_runs_demo.ipynb` has 23 code cells, every `execution_count` non-null and strictly increasing 1..23, zero error outputs, and real executed demos for all four behaviour changes (cells 19/20/21/22). `docs/runbooks/telescope_runs_calendar.rst` carries the `remint_declined` counter in the counter list and both example summary lines, a three-case `retired` reason (2), an extended reason (5), a new site-definition-correction paragraph, the narrowed `detach_declined` with the CR-05 duplicate consequence and remedy, the `is_verified` sentence, and an extended deploy note. `sphinx-build` Passed in my own run."
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
user_deferred:
  - finding: "WR-01 (the `--dry-run` preview repeats the unrecorded-provenance `sun_event()` call on every invocation, and the re-mint branch's inline comment at `allocation_projector.py:1284-1286` still says 'Both halves are skipped under dry_run (no sun_event() call either)'), WR-02 (that same preview path can raise `sun_event()`'s `ValueError` for a blank `Observatory.timezone`), WR-03's residue (the `rekeyed` paragraph's stability promise), WR-04(a)/(c), IN-01, IN-03, IN-06, and iteration 7's WR-01/WR-02/WR-03."
    severity: warning
    decision: "EXPLICITLY DEFERRED BY THE USER for round 6 (35-23-PLAN.md `<review_dispositions>` ledger; 35-ROADMAP round-6 header). Re-checked at HEAD as still open and still untouched -- plan 35-25's prohibition 4 deliberately preserved the `rekeyed` paragraph byte-for-byte, which I confirmed by direct diff. Re-recorded so the carry-forward is explicit rather than implied by omission. The WR-07 half of the old 'once ever' advisory IS now closed."
advisory:
  - finding: "The staleness warning emitted when a site-position correction re-mints a night calls that night an 'unrecorded-provenance night', but its provenance WAS recorded -- only the position fingerprint differed. Observed text: `Allocation unrecorded-provenance night pk=370 run pk=84 night=2026-09-12: stored boundary ... disagrees beyond tolerance with the resolved sun event ...` (`allocation_projector.py:739-750`, reached from `:730` after the fingerprint-only fall-through at `:723-725`)."
    category: other
    reason: "Operator-facing message accuracy -- the same defect class WR-06 fixed one counter over. An operator reading this line after correcting a site definition is told the night had no recorded provenance, which is false, and is pointed at the one-time legacy audit rather than at their own edit. Fix: branch the warning text on which entry path reached step 4, or generalise the noun. Evidence is deterministic (my own probe run and the committed notebook's executed cell 22 both show it), but it falsifies no must-have truth and no success criterion, so it is recorded here rather than as a gap."
    evidence_status: "reproduced: my throwaway probe against a migrated test DB, and `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` cell exec 22"
  - finding: "`detach_declined`'s printed operator message ('N superseded entr(y|ies) left attributed -- a person confirmed them, and an automated sweep never clears a human confirmation') is a loose fit for the NEW cause plan 35-23 routed onto that counter: a confirmed allocation night whose RETIREMENT was declined was not detached and no attribution was released -- the night simply was not deleted. The notebook's executed cell 21 shows exactly this line printed for the CR-05 case (`Run pk=83`)."
    category: other
    reason: "Plan 35-23 kept the wording byte-identical on purpose (its own prohibition 3) while adding a second cause to the counter. The 'a person confirmed them' half is exactly true for both causes, and the runbook now explains both, so this is a wording-fit concern rather than the outright falsehood WR-06 described. Fix, if wanted: split the message by cause the way the counter itself was split."
    evidence_status: "reproduced: notebook executed cell 21 (real `call_command` output)"
  - finding: "ROADMAP.md's phase-35 plan-count narrative still reads '22 executed, 3 pending' although all three round-6 plans are checked `[x]` on lines 320-322 and all three SUMMARYs exist."
    category: other
    reason: "Stale cached prose in the roadmap header, not a code or contract defect. Cosmetic; corrected on the next roadmap sync."
    evidence_status: "static observation (`.planning/ROADMAP.md:318-322`)"
  - finding: "IN-01 ... IN-07 and WR-04-round-2 from the earlier verification passes are carried forward unchanged; none was in scope for round 6 and none was re-raised by iteration 9's criticals."
    category: other
    reason: "See the round-3/round-5 35-VERIFICATION.md advisory lists for the full text of each. Recorded so the carry-forward is explicit."
    evidence_status: "carried forward, not re-probed"
flagged_prohibition_items:
  - plan: "35-24"
    statement: "A provenance token must NOT be recorded for boundaries that were not proven. On the declined path nothing is recorded at all; on the update path the token is recorded only when both sub-night fields are set and step 1 has just compared both stored boundaries against them."
    status: counterexample_found
    verification: backstop
    evidence: "Reproduced with my own throwaway probe against a migrated Django test database at HEAD. Fixture: a fully-set sub-night run (`night_start_utc=23:00`, `night_end_utc=05:00`) at the Chilean site, night 2026-07-09; project it (token recorded `v3|1|5884a60fe2946a56|23:00:00|05:00:00`); a person confirms the companion row; then BOTH the sub-night start is edited (23:00 -> 22:00) and the site's position is corrected in place. Result: `ReconcileResult(updated=1, remint_declined=1, retired=0)` -- the re-mint is correctly declined and the boundaries correctly survive -- but plan 35-23's CR-04 fall-through then reaches plan 35-24's update-path refresh (`allocation_projector.py:1369-1373`, `:1437-1441`), which recomputes the dark-window line AND writes `_record_sub_night_provenance(event, _sub_night_provenance_token(run))`. The stored token afterwards is `v3|1|f49f304a6749ba00|22:00:00|05:00:00` -- i.e. it claims the night's boundaries were minted from a 22:00 start, while `start_time` is still the 23:00-derived value the decline preserved. The comment at `:1362-1368` asserts step 1 'found them equal on THIS SAME sweep'; on this path step 1 found them UNEQUAL, which is why the re-mint was attempted at all."
    consequence_assessed: "No silent staleness results, and I checked this rather than assuming it: for a fully-set run the token is never consulted (step 2 returns before step 3), so the false token cannot make a later sweep skip a needed re-mint -- my probe's SECOND sweep still reported `remint_declined=1`. Every route that WOULD consult the token (clearing one or both sub-night fields) produces a token whose sub-night components differ from the recorded ones, so it re-mints. The observable residue is a provenance column that records a claim this sweep never proved, plus a dark-window line refreshed onto a night whose boundaries are stale. That is a correctness-of-record issue with no demonstrated user-visible defect -- hence a decision, not a gap."
human_verification:
  - test: "DECISION, not a manual test. The cross-plan seam between 35-23's CR-04 fall-through and 35-24's update-path provenance write violates 35-24's own prohibition 2 in one reachable case. To reproduce: project a fully-set sub-night run; confirm its companion row; then edit a sub-night field AND correct the same `Observatory` row's position in place; reconcile."
    expected: "Per the prohibition, a declined night should have NO provenance token written (nothing on the declined path was proven). Observed: the token IS rewritten to the run's CURRENT identity (`v3|1|f49f304a6749ba00|22:00:00|05:00:00`) while `start_time`/`end_time` still hold the pre-edit, pre-correction values the decline preserved. `remint_declined=1, updated=1, retired=0`, same primary key. No later sweep is misled (the fully-set path never reads the token), so the cost is a false claim in the provenance column, not a stale calendar. Decide: fix now (guard the `_record_sub_night_provenance()` call at `allocation_projector.py:1441` on 'the re-mint was not declined'), file as a follow-up, or accept and correct the comment at `:1362-1368` which asserts an equality that does not hold on this path."
    why_human: "No must-have TRUTH is falsified -- 35-24's dark-window truths are all about when the refresh fires, and it fires exactly as they say. What is contradicted is a `verification: backstop` PROHIBITION, which by its own declaration was never claimed to be enforced by a test, and the residue has no demonstrated behavioural consequence. Whether a false provenance record with no downstream effect is worth another round is a product call, not a verification call. Evidence is a real probe against a migrated Django test database; the probe module was deleted afterwards and `git status --porcelain -- solsys_code/ src/ docs/` is clean."
  - test: "ACKNOWLEDGEMENT of a narrowed roadmap contract. ROADMAP Success Criterion 3 / ALLOC-03 says linking an `ObservationRecord` to an allocation night REMOVES that night's allocation event. Plan 35-23's CR-05 guard adds a stated exception: a night whose companion row a person has CONFIRMED now survives the link. To see it: confirm an allocation night's companion row in the Django admin, then link a placed `ObservationRecord` to that run."
    expected: "The allocation night is NOT deleted; it is reported under `detach_declined` with the warning `Allocation retire declined: night pk=N ... is human-confirmed to run pk=M -- an automated retirement never destroys it`, and the calendar shows BOTH the confirmed allocation night and the observation's own entry on that night until someone clears the confirmation and re-runs the sweep. The default (unconfirmed) path is unchanged and still retires -- `TestObservationHandoff` and `test_unconfirmed_night_still_retires_normally` are green and unedited in my own run, and unlink-restores is untouched."
    why_human: "This narrows a ROADMAP success criterion, which is the phase contract I verify against. It follows the UAT-2026-09-09 'human outranks machine' decision and was demanded by 35-REVIEW.md iteration 9's CR-05, it is counted, logged, documented in the runbook's `detach_declined` section with its remedy, and demonstrated in the notebook's executed cell 20 -- so it is deliberate and visible, not a defect. But accepting a permanently narrowed SC-3 is a product judgement. Plan 35-23 itself records it as a flagged assumption; this is the confirmation step for it."
behavior_unverified_items: []
coincidental_reliance_items:
  - truth: "The cutover's identity guard refuses what it cannot prove it owns (35-12 truths 1-2)."
    reason: undeclared-precondition
    harden: "Carried forward unchanged from the previous three passes: the PERMISSIVE branch still depends on `observation_details` being trustworthy when it happens to match, and that field is writable from three staff surfaces. Round 3's accepted disposition was to document the consequence; the precondition is still undeclared in code. Advisory only -- no score or status effect."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report (round 6 re-verification)

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.
**Verified:** 2026-09-16T19:14:26Z (HEAD `0bc1ccd`, branch `issue37-telescope-runs-calendar`)
**Status:** human_needed
**Re-verification:** Yes — after gap-closure round 6 (plans 35-23, 35-24, 35-25), superseding the round-5 `human_needed` verdict whose single open item was the escalated site-correction decision.

## Summary

**The escalation is genuinely closed.** I did not take the SUMMARY's word for it: I wrote my own throwaway probe and re-ran the round-5 verifier's exact reproduction against a migrated Django test database at HEAD. The same fixture that previously produced `unchanged=1`, the same primary key and boundaries frozen ~15 hours away from the corrected site's real sun events now produces `retired=1 / created=1`, a new primary key, and boundaries equal to the corrected position's live `sun_event()` values. The mechanism is real and readable in the source (`v3` marker, `_site_position_fingerprint()`, a component-wise token comparison that routes a fingerprint-only difference to one bounded `sun_event()` resolution rather than to an outright re-mint).

Everything else this round shipped also holds up under first-hand checking: the CR-04 fall-through, the CR-05 retirement guard on both the sweep and the receiver path, the two-counter split threaded all the way to an operator's summary line, the WR-05 dark-window refresh, the qualified WR-07 cost bound and the WR-08 documentation. 324 tests across seven modules pass in my own run, plus 73 more in `test_telescope_runs` and `test_observation_projector_signals`; ruff, ruff-format, sphinx-build and `makemigrations --check` are all green; the developer database is untouched.

All three executor-documented deviations were judged, not rubber-stamped, and all three are correct (details below).

Two items are escalated rather than filed as gaps, which is why the status is `human_needed` and not `passed`: (1) a reproduced cross-plan seam where a declined re-mint now writes a provenance token for boundaries it did not prove — contradicting 35-24's own prohibition 2, with no demonstrated behavioural consequence; and (2) an acknowledgement that CR-05 permanently narrows ROADMAP Success Criterion 3.

## Goal Achievement

### ROADMAP Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Allocation with a resolved site + awarded window shows one sunset→sunrise event per window night; queue/class-wide/satellite keeps a single container | ✓ VERIFIED | `TestEndToEndAllocationNight` (incl. `test_queue_sourced_run_keeps_its_single_container_never_fanned_out`) green in my own 324-test run. This round's change widens the re-mint funnel correctly: a previously-invisible in-place site correction now re-mints (my own probe), and a correction too small to move the sun events resolves within tolerance and reports `unchanged` (`TestSiteChangeRemints`, `TestMintInputInvariant`) |
| 2 | Allocation nights follow the site-local observing night (Chilean + Australian) | ✓ VERIFIED | `_night_span_utc()`/`night_bounds()` untouched by this round (`git diff da13a31..HEAD` over `allocation_projector.py` touches the fingerprint, the token comparison, the two guards, the counter and the refresh only). `TestAllocationNightBoundary`, `TestSubNightWindow`, `TestSubNightWindowSiteDirection` unedited (no diff hunk in the old-file range 517..1713) and green; my own probe crossed `America/Santiago` → `Australia/Sydney` end to end |
| 3 | Linking an `ObservationRecord` removes that night's allocation event, leaves the observation's own event untouched; unlinking restores it | ✓ VERIFIED (with a stated, documented exception) | `TestObservationHandoff` unedited and green (default path unchanged: link retires, unlink restores, the observation's own event untouched). Plan 35-23's CR-05 guard adds ONE exception — a night a person has CONFIRMED survives the link, counted under `detach_declined`, logged, documented in the runbook with its remedy, and shown in notebook cell 20. The narrowing is real and is escalated as Human Verification item 2 |
| 4 | `load_telescope_runs` produces the same per-night calendar as before, by way of an allocation record, and re-running changes nothing | ✓ VERIFIED | `load_telescope_runs.py` and `test_load_telescope_runs.py` are absent from `git diff da13a31..HEAD` — byte-unchanged this round — and the module is green inside my own 324-test run. The loader delegates boundaries to `reconcile_run()`, so the `v3` token change reaches it only through that call |
| 5 | After the cutover, one event per night: no duplicate, no orphan | ✓ VERIFIED | `cutover_classical_allocations.py` and its test module are absent from the round-6 diff; the module is green in my own run. The one new way a duplicate can appear (a confirmed night surviving its retirement) requires an explicit human confirmation and is documented with its remedy — see SC-3 |

**Roadmap contract: 5/5.**

### Observable Truths — round-6 plan must-haves

Plans 35-01 … 35-22 were verified across the five prior passes; those truths were re-checked for regression only (see *Regression Checks*). Plans 35-23 … 35-25 are verified in full here.

#### Plan 35-23 — CR-04 fall-through, CR-05 retirement guard, the counter split, IN-05

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | A declined re-mint still receives its ordinary title/description/campaign refresh; boundaries preserved exactly | ✓ VERIFIED | The `continue` is gone: `:1256-1280` counts the decline, `:1320` falls through, and the update path's `fields` dict (`:1399-1403`) contains `title`/`description`/`target_list` ONLY — no boundary key. `TestDeclinedRemintStillUpdatesLabels` (5 tests) green; my own probe observed `remint_declined=1, updated=1`, same pk, both boundaries byte-unchanged, `[CANCELLED]`-style label refresh reaching the night |
| 2 | The retirement branch never destroys a human-confirmed allocation night | ✓ VERIFIED | `:1182-1208`: `_clearable_declined_and_unattributed()` applied to `existing` itself, `existing.delete()` only when `existing_deletable`, `retired` incremented only when the night actually went away. `TestRetirePathAllocationEventGuard` green (6 tests) |
| 3 | The retirement decline is reachable without a sweep and is proven so through the link | ✓ VERIFIED | `receiver_on_run_observation_save()` → `reproject_allocation_if_dispatched()` → `project_allocation()`; `test_dry_run_parity_for_the_confirmed_retirement_decline` and `test_retiring_a_night_never_deletes_a_human_confirmed_alloc_event` exercise it through the `CampaignRunObservation` save. Notebook cell 20's executed output shows the guard firing from the receiver with no explicit `reconcile_run()` call |
| 4 | `detach_declined` and `remint_declined` are two counters so each operator message can be true | ✓ VERIFIED (see advisory 2) | The split exists end to end and is pinned at the command level (`test_real_sweep_reports_remint_declined_for_a_human_confirmed_alloc_night` asserts `remint_declined == 1` AND `detach_declined == 0` in the real summary, plus the absence of the `left attributed` wording). `detach_declined`'s own wording is byte-identical (grep). Caveat recorded as advisory 2: the preserved wording is a loose fit for CR-05's new retirement-decline cause |
| 5 | The split also removes CR-04's counter ambiguity (`remint_declined: 1, updated: 1` reads coherently) | ✓ VERIFIED | My own probe printed exactly `ReconcileResult(created=0, updated=1, ..., detach_declined=0, remint_declined=1, retired=0)` for a declined night |
| 6 | Dry-run and real sweep report identical counters for both new declines | ✓ VERIFIED | The re-mint guard runs before the `dry_run` short-circuit (`:1256` precedes `:1289`); the retirement guard's counters move outside the `if not dry_run:` block (`:1197`, `:1207` vs `:1198`). `test_declined_night_dry_run_parity_for_the_counter_pair` and `test_dry_run_parity_for_the_confirmed_retirement_decline` green |
| 7 | [edge probe, explicit] The two guarded deletes in the retirement branch decide independently; the re-mint decline is evaluated before both counters and the dry-run short-circuit, then falls through once | ✓ VERIFIED | Independence pinned by `test_confirmed_allocation_night_and_deletable_legacy_row_decide_independently` and its mirror; the source shows two separate `_clearable_declined_and_unattributed()` calls (`:1149`, `:1184`) with separate flags |
| 8 | [flagged assumption] CR-05's guard creates a STATED exception to ALLOC-03 (a confirmed night keeps its allocation event when the observation links) | ✓ VERIFIED as stated | The exception is real, counted, logged, documented (runbook `detach_declined` section) and demonstrated (notebook cell 20). Accepting the narrowing is escalated as Human Verification item 2 |

#### Plan 35-24 — the site-position fingerprint (the escalated decision), WR-05, WR-07, WR-08

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | An in-place `Observatory` correction re-mints every night already projected at that site, on the next sweep | ✓ VERIFIED | **My own probe**, not the executor's test: same fixture and same transcript as the round-5 escalation, now `retired=1 / created=1`, new pk, boundaries = the corrected position's live `sun_event()` values. `TestObservatoryCorrectionRemints` also green |
| 2 | The token's site component records the site's boundary-relevant CONTENTS as a fixed-width fingerprint beside `site_id` | ✓ VERIFIED | `_site_position_fingerprint()` `:102-147` (SHA-256 over `repr(lat), repr(lon), repr(altitude), repr(timezone)`, 16 hex chars, `'none'` for a null site); token assembled at `:520`; observed live as `v3|1|5884a60fe2946a56|none|none` |
| 3 | Existing rows migrate with no data migration (`v2` → `v3` read-time transition) | ✓ VERIFIED | `_PROVENANCE_TOKEN_VERSION = 'v3'` `:99`; the version-AND-part-count test `:706-709`; migration `0021` is two bare `AlterField`s, no `RunPython`; `test_v2_token_reads_as_unrecorded_and_resolves_once` green |
| 4 | A site correction whose sun events do not move does NOT destroy and re-create the night | ✓ VERIFIED | `:723-725` — a fingerprint-only difference falls through to step 4's one-`sun_event()` tolerance comparison rather than returning True. `TestSiteChangeRemints`/`TestMintInputInvariant` cases green |
| 5 | [edge probe, explicit] Three adjacency cases pinned: no-move resolves as correct; a swap between identically-positioned rows still re-mints on `site_id`; a boundary exactly at tolerance still resolves as correct | ✓ VERIFIED | Named tests in `TestMintInputInvariant` and `TestProvenanceTokenFormat`; the two tolerance-boundary tests stayed green through the version bump |
| 6 | [edge probe, explicit] The three sub-night cases behave differently under a site correction, each with a named test; a malformed current-version token reads as unrecorded | ✓ VERIFIED | `test_null_null_and_half_null_runs_never_take_the_refresh_path`, `TestSetWindowSiteCorrection`, `test_current_version_wrong_part_count_token_reads_as_unrecorded_and_resolves_once` — all green |
| 7 | The fully-set cross-timezone case is a decided, pinned outcome with the remedy in the docstring | ✓ VERIFIED | `test_a_set_window_moved_across_timezones_has_a_pinned_outcome` green; the remedy is in `_span_needs_remint()`'s step-2 docstring `:573-576`; plan 35-20's compute-before-delete ordering (`:1302` before `:1311`) keeps the night intact |
| 8 | The dark-window refresh costs nothing on an idempotent sweep and nothing in a preview | ✓ VERIFIED | `:1369-1374` (`refresh_dark_window and not dry_run`); `:1405-1418` reports the preview divergence without paying the call; `test_idempotent_reconcile_after_the_refresh_makes_zero_sun_event_calls` (`assert_not_called`) and `test_dry_run_parity_and_zero_astropy_cost_for_the_correction` green |
| 9 | D-13's clause is NARROWED, and the narrowing is stated at the call site | ✓ VERIFIED | The verbatim-quoting comment is present at `:1375-1392`; the bound it claims is pinned by the two tests above. Recorded in `decision_coverage.notes` |
| 10 | D-13's astropy budget survives: an unchanged re-reconcile of a current-format night makes zero `sun_event()` calls | ✓ VERIFIED | `TestNoSunEventRecompute` unedited (no diff hunk in its range) and green; my own probe's idempotent control also reported `unchanged` with no re-mint |
| 11 | A declined night's repeated `sun_event()` call is bounded, deliberate and pinned; no token recorded for boundaries not re-minted | ✓ VERIFIED on the re-mint path (see flagged prohibition) | `:647-668` states the bound honestly; `TestDeclinedNightResolutionCostIsBounded` green; my own probe's second sweep repeated exactly once. The re-mint path itself records nothing (`return True` at `:751` precedes any write). The separate seam where the CR-04 fall-through DOES record is the flagged prohibition below |
| 12 | `is_verified` is documented where a reader looks for it, including that it does not veto a retirement | ✓ VERIFIED | `models.py` class docstring, `verbose_name` and new `help_text`; migration 0021 carries both strings; `_remint_decline_reason()`'s cross-reference paragraph `:852-861` |
| 13 | [flagged assumptions] ALLOC-02 keying is unchanged by a timezone correction; `load_telescope_runs` untouched and pinned by its own tests | ✓ VERIFIED | The night key comes from `run.window_start + i` (`:1107-1108`) — no site-derived date. `test_load_telescope_runs.py` byte-unchanged and green |

#### Plan 35-25 — paired docs for the round

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | The runbook prints the counter vocabulary the sweep actually uses | ✓ VERIFIED | Counter-list sentence and BOTH example summary lines carry `remint_declined`, matching `reconcile_campaign_runs.py`'s actual token order (`detach_declined`, then `remint_declined`) |
| 2 | `retired` reason (2) says what is true, not what 35-21 hoped | ✓ VERIFIED | Rewritten into three cases (empty/half-set re-mints; fully-set same-timezone refreshes only the dark-window line; fully-set cross-timezone fails the run's reconcile) with the operator remedy — each matching the shipped behaviour I verified above |
| 3 | `retired` reason (5) covers the second one-time audit | ✓ VERIFIED | Extended for "every night still carrying the PREVIOUS release's token format", including the newly-checked site position/timezone input |
| 4 | The runbook states what a correction to a SITE DEFINITION does | ✓ VERIFIED | New paragraph, including the honest statement of the pre-release behaviour ("silently wrong by up to fifteen hours") and the within-tolerance no-churn case |
| 5 | `is_verified`'s second meaning is written where the operator reads about the counter it drives | ✓ VERIFIED | The runbook quotes the shipped `verbose_name` verbatim — I diffed the two strings; they match byte-for-byte — and states the retirement non-veto |
| 6 | The declined-retirement consequence is documented with its remedy | ✓ VERIFIED | `detach_declined` section: both entries visible on the same night until the confirmation is cleared, plus the remedy |
| 7 | The notebook carries real executed output for all four behaviour changes | ✓ VERIFIED | 23 code cells, execution counts strictly 1..23, no nulls, zero error outputs; cells exec 19 (CR-01+CR-04), 20 (CR-05 via the receiver), 21 (real `call_command` summary with `remint_declined: 1`), 22 (in-place site correction re-minting, with an idempotent control before it) |
| 8 | This plan is what makes the round complete | ✓ VERIFIED | Both code plans' behaviour changes are documented in one pass against the final tree |
| 9 | The developer database is never written | ✓ VERIFIED | Notebook teardown cell prints the scratch path removal; `git status --porcelain -- src/fomo_db.sqlite3` empty in my own run |
| 10 | The code surface is still green behind the documentation | ✓ VERIFIED | My own runs: 324 tests (7 modules) OK, 73 more (`test_telescope_runs`, `test_observation_projector_signals`) OK, ruff Passed, ruff-format Passed, sphinx-build Passed, `makemigrations --check --dry-run` → "No changes detected" |
| 11 | [flagged assumption] ALLOC-05's cutover sequencing untouched in code; the post-upgrade audit is written down | ✓ VERIFIED | `cutover_classical_allocations.py` absent from the round-6 diff; the deploy note is extended for this release |

**Score:** 253/253 truths verified (248 plan truths + 5 ROADMAP success criteria; 0 failed; 0 behaviour-unverified; 1 prohibition flagged with a reproduced counterexample; 0 overrides).

### Executor-Documented Deviations — judged

| Deviation | Verdict | Basis |
|---|---|---|
| 35-23: IN-05 grep returns `2`, not the plan's expected `1` | **CORRECT — the plan's count was stale** | I read all three call sites. `:1115` is the BLOCKED branch, which `continue`s before the unconditional add at `:1215`; without it the D-14 convergence at `:1456` (`exclude(url__in=active_urls | retired_urls)`) would delete a foreign-owned night, contradicting `project_allocation()`'s own docstring. The decline branch's redundant call and its false "load-bearing" comment are both gone |
| 35-24: `v2` → `v3` literal updates in five pre-existing test assertions + one fixture repair | **CORRECT — necessary consequence of the deliberate version bump, no test weakened** | The full deletion set in the test diff is 6 `detach_declined == 1` assertions MOVED to `remint_declined` (each replacement also adds a `detach_declined == 0` assertion, so the split is pinned in both directions), 5 `'v2|'` → `'v3|'` prefix literals, docstring rewrites, and one fixture line replaced with a real `Observatory` row. No test renamed, skipped, deleted or loosened; no `skipTest`/`@skip`/`expectedFailure` anywhere in either module |
| 35-25: Task 1's verify regex cannot span RST double-backtick markup; the `rekeyed` paragraph claimed byte-unchanged | **CORRECT — confirmed independently** | `diff` of the `rekeyed` paragraph between `6f65b23` and HEAD returns no differences (byte-identical), and the only `rekeyed` tokens in the whole runbook diff are the counter-list line and the example summary line. The plan's `rekeyed. counts` pattern uses one `.` where the file has two backticks — a pre-existing plan-side regex bug, not a documentation defect. Prohibition 4 held |

### Regression Checks — plans 35-01 … 35-22

| Check | Result |
|---|---|
| `git diff --stat da13a31..HEAD` file set | `allocation_projector.py`, `campaign_reconciler.py`, `campaign_views.py`, `models.py`, `reconcile_campaign_runs.py`, new migration 0021, two test modules, the runbook, the reconciler notebook, and `.planning/` only. No other source file touched |
| "Must stay unedited" classes | No diff hunk in the old-file range 517..1713, which covers `TestTakeoverBlockedCountedOnce`, `TestFinalConvergenceGuard`, `TestAllocationNightBoundary`, `TestNoSunEventRecompute`, `TestSubNightWindow`, `TestSubNightWindowSiteDirection`, `TestClearedSubNightFieldRemints`, `TestUnrecordedProvenanceNight`; `TestObservationHandoff` and `TestRetirePathLegacyEventGuard` likewise untouched. All green in my own run |
| `load_telescope_runs` / cutover surfaces (ALLOC-04, ALLOC-05) | Both command modules and both test modules byte-unchanged this round; green inside the 324-test run |
| `_clearable_declined_and_unattributed()` (35-23 prohibition 2's shared helper) | `campaign_reconciler.py`'s only diff this round is two docstrings and the new `remint_declined` field — the helper itself is unmodified |
| Migration 0021 (35-24 prohibition 1) | Two bare `AlterField`s, no `RunPython`, no data step |
| 35-23 / 35-24 prohibition "no `docs/`"; 35-25 prohibition "no source" | Verified per-commit: all five 35-23 commits and all five 35-24 commits touch `solsys_code/` only; `b61f9da` touches the runbook only, `6f2f29d` the notebook only |
| Schema drift | `makemigrations --check --dry-run` → "No changes detected" |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| an `Observatory` row's `lat`/`lon`/`altitude`/`timezone` | the recorded provenance token | `_site_position_fingerprint()` `:102-147` → `_sub_night_provenance_token()` `:519-520` | ✓ WIRED | **This is the round-5 escalation's missing arrow, now present.** Proven by my own probe, not by inspection alone |
| a fingerprint-only difference | step 4's one-`sun_event()` resolution (not an outright re-mint) | `:721-725` → `:730` | ✓ WIRED | `TestSiteChangeRemints`, `TestMintInputInvariant`; my probe's re-mint went through this path |
| a `v2\|` token already in the database | the bounded one-time resolution and a fresh `v3` record | version-AND-part-count test `:706-709` → `:730` → `:753` | ✓ WIRED | `test_v2_token_reads_as_unrecorded_and_resolves_once` |
| `_remint_decline_reason()` returns a reason | the plain-update path (label refresh) | decline branch `:1256-1280` → fall-through `:1320` → `fields` `:1399-1403` | ✓ WIRED | The `continue` is gone; probe observed `updated=1` with boundaries and pk intact |
| a staff confirmation on an `ALLOC:` companion row | `existing.delete()` skipped in the RETIREMENT branch | `_clearable_declined_and_unattributed()` `:1184` → `existing_deletable` `:1187-1199` | ✓ WIRED | `TestRetirePathAllocationEventGuard`; notebook cell 20 |
| a `CampaignRunObservation` save | the retirement branch, with no sweep | `receiver_on_run_observation_save()` → `reproject_allocation_if_dispatched()` → `project_allocation()` | ✓ WIRED | Notebook cell 20's executed output |
| `totals['remint_declined']` | the operator's summary line and per-run message | `ReconcileResult` → `_replace()` → command accumulator `:83`, summary lines `:155`/`:172`, stderr line `:134-141` | ✓ WIRED | Command-level test + notebook cell 21's real `call_command` output |
| a fully-set run's site correction | a refreshed dark-window line | `_site_provenance_differs()` `:757-801` → `:1369-1396` | ✓ WIRED | `TestSetWindowSiteCorrection`; my probe saw the refreshed line |
| a declined re-mint | the provenance column | `:1437-1441` | ⚠️ WIRED WHERE IT SHOULD NOT BE | The flagged prohibition: on the CR-04 fall-through with a fully-set pair and a moved site, a token IS written for boundaries the sweep declined to re-mint |

### Data-Flow Trace (Level 4)

| Artifact | Data value | Source | Produces real data | Status |
|---|---|---|---|---|
| `CalendarEvent.start_time`/`end_time` after an in-place site correction | corrected night boundaries | `_mint_fields()` → `sun_event(run.site, night, 'sun')` → `night_bounds()` | Yes — my probe read back `2026-07-09 07:20:39+00:00 / 2026-07-09 20:57:12+00:00`, equal to the live `sun_event()` values | ✓ FLOWING |
| `CalendarEventMeta.minted_sub_night_window` | five-part `v3` token | `_sub_night_provenance_token()` at `:753`, `:1317`, `:1433`, `:1441` | Yes — `v3|1|5884a60fe2946a56|none|none` read back from a real test DB | ✓ FLOWING |
| `ReconcileResult.remint_declined` | decline count | `totals['remint_declined'] += 1` `:1276` | Yes — `remint_declined: 1` in a real `call_command` summary (notebook cell 21) and in my probe | ✓ FLOWING |
| `CalendarEvent.description` dark-window line after a fully-set site correction | corrected dark window | `sun_event(run.site, night, kind='dark')` `:1393` | Yes — my probe read back `Dark window (-15 deg, UTC): 2026-07-09T16:58:22+00:00 to 2026-07-10T04:25:38+00:00` | ✓ FLOWING |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|---|---|---|---|
| The round-5 escalation's own probe, re-run independently at HEAD | throwaway `TestCase` probe against a migrated test DB (deleted after the run) | `retired=1, created=1, unchanged=0`, new pk, boundaries = corrected `sun_event()` values | ✓ PASS |
| A declined re-mint's counters, boundaries and label refresh | same probe | `updated=1, remint_declined=1, retired=0`, same pk, boundaries unchanged, description refreshed | ✓ PASS |
| A declined-and-unrecorded night's per-sweep cost | same probe, two consecutive sweeps | `remint_declined=1` on both, one resolution each | ✓ PASS |
| Seven-module regression | `python manage.py test solsys_code.tests.test_allocation_projector … test_campaign_views` | `Ran 324 tests in 157.1s — OK` | ✓ PASS |
| ALLOC-02/ALLOC-03 supporting modules | `python manage.py test solsys_code.tests.test_telescope_runs solsys_code.tests.test_observation_projector_signals` | `Ran 73 tests in 15.4s — OK` | ✓ PASS |
| Schema drift | `python manage.py makemigrations --check --dry-run` | `No changes detected` | ✓ PASS |
| Lint / format gates (D-07) | `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files` | Passed, Passed | ✓ PASS |
| Docs build | `pre-commit run sphinx-build --all-files` | Passed | ✓ PASS |
| Notebook regenerated, not hand-patched | JSON inspection | 23 code cells, 0 null `execution_count`, strictly increasing 1..23, 0 error outputs | ✓ PASS |
| `rekeyed` paragraph unedited (35-25 prohibition 4) | `diff` of the paragraph between `6f65b23` and HEAD | byte-identical | ✓ PASS |
| Developer database untouched | `git status --porcelain -- src/fomo_db.sqlite3 docs/ solsys_code/` | empty | ✓ PASS |
| Declined re-mint + fully-set pair + moved site: is a token recorded? | same probe | token rewritten to the run's CURRENT identity while boundaries stayed stale | ✗ FAIL → flagged prohibition / Human Verification item 1 |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| — | `find scripts -path '*/tests/probe-*.sh'` | no `scripts/` directory in this repo; no probe script declared by any 35-\* plan | N/A — SKIPPED (no probe scripts in this project) |

### Prohibitions

Every round-6 prohibition is declared `status: flagged-unverified`, `verification: backstop` — i.e. never claimed to be test-enforced. I checked each against the tree rather than abstaining wholesale.

| Plan | Prohibition | Verdict | Evidence |
|---|---|---|---|
| 35-23 | A declined re-mint must NOT write `start_time`/`end_time` | ✓ UPHELD | Update-path `fields` dict contains only `title`/`description`/`target_list` (`:1399-1403`); probe observed both boundaries and the pk unchanged |
| 35-23 | The retirement guard must NOT reuse `_remint_decline_reason()` | ✓ UPHELD | `:1184` calls `_clearable_declined_and_unattributed()` directly; `test_is_verified_false_with_no_confirmation_still_retires` green |
| 35-23 | No existing test weakened, renamed, skipped or deleted; `detach_declined`'s wording unchanged | ✓ UPHELD | Full deletion set reviewed (see Deviations); wording byte-identical by grep |
| 35-23 | `docs/` must NOT be touched | ✓ UPHELD | All five commits touch `solsys_code/` only |
| 35-24 | The migration must contain no `RunPython` | ✓ UPHELD | Two bare `AlterField`s |
| 35-24 | A provenance token must NOT be recorded for boundaries that were not proven | ⚠️ COUNTEREXAMPLE FOUND | Reproduced — see `flagged_prohibition_items` and Human Verification item 1 |
| 35-24 | `telescope_instrument`/`campaign` must NOT enter the token | ✓ UPHELD | `_sub_night_provenance_token()` reads only version, `site_id`, fingerprint, sub-night pair |
| 35-24 | The dark-window refresh must NOT add an astropy call to a preview | ✓ UPHELD | `refresh_dark_window and not dry_run` `:1374`; `test_dry_run_parity_and_zero_astropy_cost_for_the_correction` green |
| 35-25 | Notebook JSON must NOT be hand-edited | ✓ UPHELD | Strictly increasing 1..23 execution counts, no nulls, no error outputs, live pks/timestamps consistent across cells |
| 35-25 | No management command may be run against `src/fomo_db.sqlite3` | ✓ UPHELD | Scratch-copy setup and teardown visible in executed output; `git status` clean |
| 35-25 | No source file may be modified | ✓ UPHELD | `b61f9da` (runbook) and `6f2f29d` (notebook) only |
| 35-25 | The `rekeyed` paragraph must NOT be edited | ✓ UPHELD | Byte-identical by direct diff |

### Requirements Coverage

| Requirement | Source Plans (round 6) | Description | Status | Evidence |
|---|---|---|---|---|
| ALLOC-01 | 35-23, 35-24, 35-25 (+ 35-01/03/13/15/16/19/20/21/22) | Per-night events for resolved classical/awarded allocations; container for queue/class-wide/satellite | ✓ SATISFIED | SC-1 verified; the last known blind spot (an in-place site correction) is closed and proven by my own probe |
| ALLOC-02 | 35-24, 35-25 (+ 35-01/21/22) | Site-local observing nights, both hemispheres | ✓ SATISFIED | `TestAllocationNightBoundary`, `TestSubNightWindowSiteDirection` unedited and green; my probe crossed Chile → Sydney; the night key is window-derived, not site-derived |
| ALLOC-03 | 35-23, 35-25 (+ 35-01/04/20/22) | Handoff on link, restore on unlink, observation's own event untouched | ✓ SATISFIED (narrowed, documented) | Default path green and unedited; the CR-05 confirmation exception is counted, logged, documented and escalated for acknowledgement (Human Verification item 2) |
| ALLOC-04 | 35-24, 35-25 (+ 35-05/09/14/17) | Loader writes a collision-safe allocation record, same per-night calendar, idempotent | ✓ SATISFIED | Loader and its tests byte-unchanged this round; module green |
| ALLOC-05 | 35-23, 35-24, 35-25 (+ 35-06/07/12/21/22) | Explicit cutover sequencing, no duplicate or orphan | ✓ SATISFIED | Cutover command and tests untouched; the second one-time audit this release triggers is written into the runbook's deploy note before it happens |

**Orphaned requirements:** none — `.planning/REQUIREMENTS.md` maps exactly ALLOC-01…05 to Phase 35, every one is claimed by at least one round-6 plan, and all five carry `[x]` / "Complete".

**Traceability note on the round-6 requirements marks.** 35-25's observation is correct and I traced it: ALLOC-01…05 were already marked `[x]` in `REQUIREMENTS.md` well before this round (`git log -S` places the ALLOC-01 mark at commit `08996d2`, plan 35-03's completion, not 35-07 as the SUMMARY guessed), and `REQUIREMENTS.md` is byte-unchanged by the whole of round 6. So no plan in this round needed to write a mark, and 35-24's "0/4 ready" shared-ID gate note is a statement about the gate, not about the file. The marks are nevertheless earned as of this verification: every ID's supporting behaviour is verified above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/models.py` | 357, 453, 466, 509 | `TBD` | ℹ️ Info | Domain vocabulary ("TBD window" = a run with no concrete window), not a debt marker. Pre-existing |
| `solsys_code/campaign_reconciler.py`, `campaign_views.py`, `docs/runbooks/telescope_runs_calendar.rst` | various | `TBD`, `PLACEHOLDER` | ℹ️ Info | Same — documented skip reason and a tier-3 site class name |
| `solsys_code/allocation_projector.py` | 1284-1286 | stale comment contradicting the code ("Both halves are skipped under dry_run (no sun_event() call either)") | ⚠️ Warning | Iteration 8 WR-01, user-deferred for this round; unchanged |
| `solsys_code/allocation_projector.py` | 739-750 | operator-facing warning labels a fingerprint-difference re-mint as an "unrecorded-provenance night" | ⚠️ Warning | New this round; evidenced; recorded under `advisory` |
| `solsys_code/allocation_projector.py` | 1362-1368 | comment asserts "step 1 … found them equal on THIS SAME sweep", which is false on the CR-04 declined fall-through | ⚠️ Warning | The flagged prohibition's companion comment; see Human Verification item 1 |
| `solsys_code/management/commands/reconcile_campaign_runs.py` | 128-131 | preserved `detach_declined` wording is a loose fit for CR-05's new cause | ⚠️ Warning | Recorded under `advisory` |

No `FIXME`, `XXX`, `HACK`, `PLACEHOLDER` debt marker or unreferenced `TODO` in any file this round modified. No stubs, no hollow props, no console-log-only implementations. No blocking anti-pattern.

### Advisory (New Scope, Unevidenced or Non-Blocking)

| # | Finding | Category | Why Advisory |
|---|---|---|---|
| 1 | The staleness warning calls a fingerprint-difference re-mint an "unrecorded-provenance night" | other | Evidenced and new-scope, but falsifies no truth and no success criterion — message accuracy only |
| 2 | `detach_declined`'s preserved message is a loose fit for CR-05's new retirement-decline cause | other | Evidenced; the "a person confirmed them" half is exactly true for both causes and the runbook explains both |
| 3 | ROADMAP's phase-35 header still narrates "22 executed, 3 pending" | other | Stale cached prose; the plan checkboxes and SUMMARYs are all correct |
| 4 | IN-01…IN-07 and WR-04-round-2 carried forward unchanged | other | None was in round-6 scope; full text in the earlier reports |

### Human Verification Required

#### 1. DECISION — a declined re-mint now writes a provenance token for boundaries it did not prove

**Reproduction:** project a fully-set sub-night run (`night_start_utc=23:00`, `night_end_utc=05:00`); confirm its companion row; then edit the sub-night start to `22:00` AND correct the same `Observatory` row's position in place; reconcile.

**Observed (my own probe, migrated Django test database, HEAD `0bc1ccd`):**

```
PROBE2 token_before= v3|1|5884a60fe2946a56|23:00:00|05:00:00
PROBE2 result= ReconcileResult(created=0, updated=1, ..., detach_declined=0, remint_declined=1, retired=0, ...)
PROBE2 token_after=  v3|1|f49f304a6749ba00|22:00:00|05:00:00
PROBE2 boundaries unchanged? True      PROBE2 same pk? True
PROBE2 second sweep= ReconcileResult(..., remint_declined=1, ...)
```

The re-mint is correctly declined and the boundaries correctly survive — but plan 35-23's CR-04 fall-through then reaches plan 35-24's update-path refresh, which writes the run's CURRENT token (`22:00:00` start) onto a night whose `start_time` is still the 23:00-derived value. That contradicts 35-24's own prohibition 2 ("on the declined path nothing is recorded at all"), and the comment at `allocation_projector.py:1362-1368` asserting that step 1 "found them equal on THIS SAME sweep" is false on this path — step 1 found them unequal, which is why a re-mint was attempted.

**Consequence, checked rather than assumed:** none observable. For a fully-set run the token is never consulted (step 2 returns two steps before step 3), so no later sweep can be misled — the second sweep still reports `remint_declined=1`. Every route that would consult the token (clearing one or both sub-night fields) produces a differing token and re-mints.

**Your options:** fix now (guard the `_record_sub_night_provenance()` call at `:1441` on "the re-mint was not declined"); file as a follow-up; or accept and correct the two comments. This is the seam between two plans in the same round, which neither plan's tests cover.

#### 2. ACKNOWLEDGEMENT — CR-05 permanently narrows ROADMAP Success Criterion 3 / ALLOC-03

**What changed:** a night whose companion row a person has CONFIRMED is no longer deleted when an `ObservationRecord` links to it. The calendar then shows both that allocation night and the observation's own entry until the confirmation is cleared and the sweep re-run.

**Why it is here:** SC-3 as written says linking removes the night, full stop. The exception follows the UAT-2026-09-09 "human outranks machine" decision and was demanded by 35-REVIEW.md iteration 9's CR-05 (the delete was destroying a confirmation, both observation links and `is_verified` through a CASCADE, silently, counted as ordinary `retired` work). It is counted under `detach_declined`, logged with a named warning, documented in the runbook's `detach_declined` section with its remedy, and demonstrated in the notebook's executed cell 20. The default (unconfirmed) path is unchanged and still retires, and unlink-restores is untouched.

**What is needed from you:** confirm you accept the narrowed SC-3 wording, or say whether ALLOC-03/SC-3 should be reworded to carry the exception explicitly.

### Gaps Summary

None. No must-have truth is failed, no artifact is missing or stubbed, no key link any plan claims is unwired, and no blocking anti-pattern exists. The round-5 escalation is closed and I verified it with my own reproduction rather than from a SUMMARY. The status is `human_needed` because of the two decision items above: a reproduced prohibition counterexample at the seam between plans 35-23 and 35-24 with no demonstrated behavioural consequence, and an acknowledgement that a roadmap success criterion is now deliberately narrower than its wording.

---

_Verified: 2026-09-16T19:14:26Z_
_Verifier: Claude (gsd-verifier)_
