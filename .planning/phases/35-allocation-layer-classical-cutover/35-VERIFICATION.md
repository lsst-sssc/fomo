---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-16T15:26:19Z
status: human_needed
score: 221/221 must-haves verified
covered_files:
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
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
  - ".planning/phases/35-allocation-layer-classical-cutover/35-CONTEXT.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md"
  - ".planning/REQUIREMENTS.md"
  - "solsys_code/admin.py"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/apps.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_utils.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/cutover_classical_allocations.py"
  - "solsys_code/management/commands/load_telescope_runs.py"
  - "solsys_code/management/commands/reconcile_campaign_runs.py"
  - "solsys_code/migrations/0018_campaignrun_night_window_fields.py"
  - "solsys_code/migrations/0019_calendareventmeta_minted_sub_night_window.py"
  - "solsys_code/migrations/0020_alter_calendareventmeta_minted_sub_night_window.py"
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
covered_digest: "v1:sha256:16e0513b851bf2966779ac27b3b194a792bba96b4eff07963721976e319be355"
behavior_unverified: 0
overrides_applied: 0
flagged_prohibitions: 0
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
re_verification:
  previous_status: gaps_found
  previous_score: 183/186
  previous_verified: 2026-09-15T19:11:38Z
  gap_closure_plans: ["35-20", "35-21", "35-22"]
  gaps_closed:
    - "Prior gap (35-03 truth 2, 35-03 truth 5 and ROADMAP SC-1 — CR-01 iteration 7, the cleared-to-null sub-night field that never re-minted) — CLOSED, confirmed against the source and by running the tests myself, not from the SUMMARY. `_span_needs_remint()` (`allocation_projector.py:467-595`) now decides the null case in two ways: a recorded provenance token comparison (`:566-567`) and, when provenance is unrecorded, exactly ONE `sun_event(run.site, night, kind='sun')` call compared against `_UNRECORDED_PROVENANCE_TOLERANCE` with a warning log and a re-mint when the stored boundary is outside it (`:571-592`). `TestClearedSubNightFieldRemints` and `TestUnrecordedProvenanceNight` are present and green in my own run (44 tests across the eight regression classes, OK). The prior pass's three probe shapes (A both-cleared, B half-null's remaining field cleared, C one-of-two cleared) each have a named test. The recorded token is admin-readonly on BOTH staff surfaces (`admin.py:124`, `admin.py:340`)."
    - "35-REVIEW.md iteration 8 CR-01 (the re-mint delete path had no human-confirmation guard) — CLOSED. `_remint_decline_reason()` (`allocation_projector.py:598-649`) reuses `campaign_reconciler._clearable_declined_and_unattributed()` unmodified and adds a staff-state check (`observation_record_id`, `observation_group_id`, `is_verified is False`). It is called at `:987`, BEFORE both counters move and before the `dry_run` short-circuit at `:1020`, and a decline increments `detach_declined` (`:1007`) and adds the url to `active_urls` (`:1011`) so the D-14 convergence step at `:1114` cannot delete the night the guard just refused to delete. Seven named tests (confirmed, is_verified, observation_record, observation_group, dry-run parity, ordinary-re-mint-still-works, confirmed+unrecorded-provenance) all green in my own run; the decline warning lines appear in the captured log output."
    - "35-REVIEW.md iteration 8 CR-02 (the provenance token omitted the site, so a site correction was permanently invisible) — CLOSED. `_PROVENANCE_TOKEN_VERSION = 'v2'` (`:87`); `_sub_night_provenance_token()` emits `v2|{site_id}|{start}|{end}` (`:443-446`); the read predicate at `:566` is a version-prefix test, so `None`, `''` and any pre-release token all fall through to the bounded one-time resolution branch. `models.CalendarEventMeta.minted_sub_night_window` is `max_length=64` with migration `0020_alter_calendareventmeta_minted_sub_night_window.py` — a bare `AlterField`, no `RunPython`. `TestSiteChangeRemints` reproduces the reviewer's probe 1 (La Silla -> Siding Spring) and asserts `retired=1 / created=1 / unchanged=0`, a new pk, and boundaries equal to the AUSTRALIAN site's real `sun_event()` values resolved through `night_bounds()`; green in my own run."
    - "35-REVIEW.md iteration 8 CR-03 (the re-mint deleted before `_mint_fields()` could raise, with no transaction anywhere on the path) — CLOSED by both halves. `remint_fields = _mint_fields(run, night)` now runs at `:1033`, BEFORE `existing.delete()`; `from django.db import transaction` (`:37`) and `with transaction.atomic():` (`:1034`) wrap delete/create/link/record-provenance as one unit, scoped to the single night. `TestRemintAtomicity` has one test per mechanism (an inverted-span raise from the compute half, and a `RuntimeError` injected into `insert_or_create_calendar_event` strictly between the delete and the create); both assert same pk, same start_time, same end_time after the raise. Both green in my own run."
    - "35-REVIEW.md iteration 8 WR-03 (the runbook's `retired` enumeration was wrong and `docs/` had been left untouched) — CLOSED for the counter paragraphs. `docs/runbooks/telescope_runs_calendar.rst` now says five reasons, names the site correction inside reason (2), adds the one-time provenance audit as reason (5), splits `detach_declined` into declined-release vs declined-re-mint with an operator remedy for the latter, and adds a post-upgrade deploy note telling the operator to `--dry-run` first. `sphinx-build` pre-commit hook Passed in my own run."
    - "CLAUDE.md paired-docs obligation for the round — DISCHARGED. `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` was regenerated by re-execution: 20/20 code cells carry non-null, strictly increasing execution counts (1..20) and zero error outputs. Cell 18 (exec 11) shows a real site correction moving a night's boundaries from `2026-09-01T22:35:09Z/2026-09-02T10:49:50Z` to the corrected site's own `sun_event()` values `2026-09-01T07:52:18Z/2026-09-01T20:14:41Z` with `retired=3, created=3`; cell 19 (exec 12) shows `detach_declined=1` with the pk, boundaries and confirmation all surviving, plus the decline warning in the captured output. The 35-22 commits (`4385742`, `3ad0797`) touch `docs/` only; `git status --porcelain -- src/fomo_db.sqlite3` is empty."
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
user_deferred:
  - finding: "WR-01/WR-02/WR-03 from 35-REVIEW.md iteration 7 (dry-run cannot detect a half-null inverted span; the cutover re-run caveat under-enumerates re-applied fields; `cutover_classical_allocations.py:649` discards `adopt_event_into_run()`'s refusal return). Carried forward unchanged from the prior verification pass — none of them was in scope for round 5."
    severity: warning
    decision: "EXPLICITLY DEFERRED BY THE USER in the prior pass; re-checked as still open, re-recorded here so they are not silently dropped."
advisory:
  - finding: "35-REVIEW.md iteration 8 WR-01 is still open: the re-mint branch's inline comment (`allocation_projector.py:1016-1017`) still asserts 'Both halves are skipped under dry_run (no sun_event() call either)', and `_span_needs_remint()`'s cost bound (`:533-534`) still says 'once ever' unqualified — but `_span_needs_remint(..., dry_run=True)` does reach `:571` and calls `sun_event()` for every unrecorded night, deliberately skipping the recording at `:593`, so a dry run repeats the cost on every invocation."
    category: other
    reason: "Documentation-diverged-from-code, the same class this phase's iteration-7 CR-01 was found underneath. Round 5 was scoped to the three criticals only. Fix: qualify both the comment and the cost bound, and optionally pin the per-dry-run call count."
    evidence_status: "reviewer probe 2 in 35-REVIEW.md (dry-run #1 = 5 calls, dry-run #2 = 5 calls); I re-read both passages and confirmed the divergence in the shipped source, no new probe run"
  - finding: "35-REVIEW.md iteration 8 WR-02 is still open: `_span_needs_remint()`'s unrecorded-provenance branch calls `sun_event()` at `:571` on the dry-run path with no `try/except`, so a blank or malformed `Observatory.timezone` can now raise out of what the create branch's own comment (`:1050-1071`) documents as a read-only preview, aborting the whole preview sweep."
    category: architectural
    reason: "Behaviour is unchanged since iteration 8 raised it; the round did not take it on. Fix: either state the trade-off where the call is made, or degrade a preview to 'cannot decide, report unchanged'."
    evidence_status: "reviewer probe 3 in 35-REVIEW.md; not re-run in this pass — the call site and the absence of a guard were confirmed by reading `:571`"
  - finding: "35-REVIEW.md iteration 8 WR-03 is only partly closed: the `retired` and `detach_declined` paragraphs and the deploy note all landed, but the `rekeyed` paragraph (`telescope_runs_calendar.rst`, immediately after the `retired` block) still promises 'same primary key, same start/end time' with no note that the just-re-keyed night carries no provenance and may therefore be audited and re-minted on the NEXT sweep."
    category: other
    reason: "Outside plan 35-22's must_haves, which name the `retired`, `detach_declined` and deploy-note passages only. Fix: one sentence in the `rekeyed` paragraph pointing at reason (5)."
    evidence_status: "static observation; runbook diff read in full"
  - finding: "35-REVIEW.md iteration 8 WR-04(a) is still open: the premise that the round-hour test fixtures never represented a genuine legacy night is recorded only as a Rule-1 deviation note inside 35-19's SUMMARY, not anywhere a future reader can challenge it in the tree. The post-upgrade consequence IS now documented for operators (the new deploy note), which was WR-04's other half."
    category: other
    reason: "Documentation-of-premise only; no behavioural defect. Fix: a module-level comment next to `_UNRECORDED_PROVENANCE_TOLERANCE` naming the writer the premise refers to."
    evidence_status: "none provided — reasoned from the SUMMARY and the tree"
  - finding: "IN-01 … IN-07 and WR-04-round-2 from the prior verification pass are carried forward unchanged; none was in scope for round 5 and none was re-raised by iteration 8's criticals."
    category: other
    reason: "See the prior 35-VERIFICATION.md advisory list for the full text of each. Recorded here so the carry-forward is explicit rather than implied by omission."
    evidence_status: "carried forward, not re-probed"
behavior_unverified_items: []
coincidental_reliance_items:
  - truth: "The provenance token detects every change to the inputs a night's boundaries were minted from (35-21 truths 1, 2 and 9)."
    reason: undeclared-precondition
    harden: "It holds because the token carries `run.site_id` — the site's IDENTITY — while the boundaries actually depend on the site's CONTENTS (`lat`, `lon`, `altitude`, `timezone`, all of which feed `sun_event()`). Nothing in the code declares 'an Observatory row's coordinates never change after a night is minted', and `ObservatoryAdmin` (`solsys_code/solsys_code_observatory/admin.py:6-12`) declares no `readonly_fields`, so a staff correction in place is reachable. I reproduced the consequence (see Human Verification item 1). Hardening options: carry the resolved site coordinates (or a hash of them) in the token, or make the position fields read-only for sites with projected nights. Advisory only — it changes no score and no status, and the finding itself is escalated for a human decision below."
  - truth: "The cutover's identity guard refuses what it cannot prove it owns (35-12 truths 1-2)."
    reason: undeclared-precondition
    harden: "Carried forward unchanged from the prior pass: the PERMISSIVE branch still depends on `observation_details` being trustworthy when it happens to match, and that field is writable from three staff surfaces. Round 3's accepted disposition was to document the consequence; the precondition is still undeclared in code."
human_verification:
  - test: "DECISION, not a manual test. Decide whether an in-place correction to an `Observatory` row's position or timezone should re-mint the allocation nights already projected at that site — and whether it blocks this phase or becomes a follow-up. To reproduce: mint a night for a run with both sub-night fields null (so both boundaries are sun-derived), then edit that same `Observatory` row's `lat`/`lon`/`altitude`/`timezone` in the Django admin WITHOUT changing `run.site`, and reconcile again."
    expected: "Arguably the night should re-mint to the corrected site's real sun events. It does NOT: I reproduced `ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0, ...)`, same pk, boundaries unchanged at `2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00` while the corrected site's true sunset/sunrise are `2026-07-09 07:20:39 / 2026-07-09 20:57:12` — a ~15 hour error, silent, uncounted and permanent. Recorded token was `v2|1|none|none` before and after, because the token carries `site_id` (unchanged) rather than the site's coordinates."
    why_human: "This is the SAME defect signature as iteration 8's CR-02, one input further out, but it is NOT falsified by any must-have as worded: plan 35-21 truth 2 explicitly defines the boundary input set as `(run.night_start_utc, run.night_end_utc, run.site, night)` and the token does carry `run.site`. It is also entirely pre-existing — this round narrowed the hole (the site FK is now detected) rather than opening it, and it needs a manual admin edit of a site definition after projection (the `MPCObscodeFetcher` path builds NEW `Observatory()` rows, it does not update existing ones). Whether that residual justifies a round 6, a follow-up issue, or a deferral alongside WR-01/WR-02/WR-03 is a product decision, not a verification call. Evidence is a real probe against a migrated Django test database; the probe module was deleted afterwards and `git status --porcelain -- solsys_code/` is clean."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report (round 5 re-verification)

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.
**Verified:** 2026-09-16T15:26:19Z (HEAD `da13a31`)
**Status:** human_needed
**Re-verification:** Yes — after gap-closure round 5 (plans 35-20, 35-21, 35-22), which also carries plan 35-19 into verification for the first time (35-19 executed after the prior pass was written).

## Summary

Everything the prior pass failed is closed, and closed in the source rather than in a SUMMARY. I re-read `_span_needs_remint()`, `_remint_decline_reason()`, `_sub_night_provenance_token()`, the re-mint branch, `models.CalendarEventMeta`, migration 0020 and the full runbook diff, and I ran the relevant tests myself (20 new tests + 44 regression tests + 64 cutover/loader tests, all OK) rather than trusting the claimed counts. The three iteration-8 criticals are each pinned by named behavioural tests, not by symbol presence.

One new finding is escalated rather than filed as a gap: correcting an `Observatory` row's coordinates or timezone **in place** leaves every already-projected night at that site permanently stale, silently and uncounted. I reproduced it. It is the CR-02 signature one input further out, it is pre-existing rather than introduced by this round, and no must-have as worded is falsified by it — so it is a human decision (fix now / follow-up / defer), which is why the status is `human_needed` rather than `gaps_found`.

## Goal Achievement

### ROADMAP Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Allocation with resolved site + awarded window shows one sunset→sunrise event per window night; queue/class-wide/satellite keeps a single container | ✓ VERIFIED | The prior pass's FAIL cause (a cleared sub-night field never re-minting) is gone: all three probe shapes have named tests, green in my run. A site correction now re-mints too (`TestSiteChangeRemints`). Dispatch and the container rule are untouched and green. One DOCUMENTED exception now exists by design: a night carrying a human confirmation or staff state is left as-is and reported under `detach_declined` with a warning log and an operator remedy in the runbook — counted and visible, not silent |
| 2 | Allocation nights follow the site-local observing night (Chilean + Australian sites) | ✓ VERIFIED | `_night_span_utc()` / `night_bounds()` unchanged by this round (`git diff 61f0df04..HEAD` over `allocation_projector.py` touches the token, the guard, the reorder and the wrap only). `TestSubNightWindow`, `TestSubNightWindowSiteDirection` and `TestAllocationNightBoundary` green in my run; `TestSiteChangeRemints` exercises both hemispheres end to end |
| 3 | Linking an `ObservationRecord` removes that night's allocation event, leaves the observation's own event untouched; unlinking restores it | ✓ VERIFIED | `TestObservationHandoff` and `TestRetirePathLegacyEventGuard` unedited (`git diff 61f0df04..HEAD` shows no hunk inside either class) and green in my run. The new guard cannot reach the retire path — `night in retired` short-circuits above the re-mint branch. Notebook cell 17 (exec 10 region) shows a real handoff with `retired=1` |
| 4 | `load_telescope_runs` produces the same per-night calendar as before, by way of an allocation record, and re-running changes nothing | ✓ VERIFIED | `test_load_telescope_runs.py` is byte-unchanged since the review commit and its module is green in my own 64-test run. The loader delegates boundaries to `reconcile_run()`, so the token format change reaches it only through that call |
| 5 | After the cutover, one event per night: no duplicate, no orphan left behind | ✓ VERIFIED | `test_cutover_classical_allocations.py` unchanged this round and green in my own run (64 tests with the loader module). `cutover_classical_allocations.py` not modified by any round-5 commit |

**Roadmap contract: 5/5.**

### Observable Truths — round-5 plan must-haves

Plans 35-01 … 35-18 were verified across the four prior passes; those truths were re-checked for regression only (see *Regression Checks* below). Plans 35-19 … 35-22 are verified in full here — 35-19 because it executed after the prior report was written, and 35-20/21/22 because they are this round's gap-closure work.

#### Plan 35-19 — record mint provenance so a cleared sub-night field re-mints (the prior pass's gap)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Probe shape A (set/set → null/null) re-mints to the computed sun events, `retired 1 / created 1` | ✓ VERIFIED | `TestClearedSubNightFieldRemints`, green in my run (44-test regression batch) |
| 2 | Probe shape B (half-null → both null) re-mints the same way | ✓ VERIFIED | Same class, named test per shape |
| 3 | Probe shape C (one of two cleared, the other still matching) re-mints | ✓ VERIFIED | Same class; `_span_needs_remint()` reaches step 3's token comparison after both `is not None` gates pass |
| 4 | Every re-minted boundary asserted against the real `sun_event()` value, not a counter | ✓ VERIFIED | Assertions resolve `sun_event(...)` + `night_bounds(...)` live rather than hardcoding timestamps (e.g. `test_allocation_projector.py:2014-2018`) |
| 5 | D-13's astropy budget holds — an unchanged re-reconcile makes zero `sun_event()` calls | ✓ VERIFIED | `TestNoSunEventRecompute` unedited and green; `TestProvenanceTokenFormat`'s null/empty/pre-release tests each assert `mock_sun_event.call_count == 1` then zero on the following sweep |
| 6 | An unrecorded-provenance night is resolved by exactly one `sun_event()` call, logged when stale, recorded when correct | ✓ VERIFIED | `allocation_projector.py:571-595`; `TestUnrecordedProvenanceNight` green; the warning line appears in my captured test output |
| 7 | Dry run and real run reach the same decision, and the dry run writes nothing | ✓ VERIFIED | `:593` guards the recording on `not dry_run`; `test_allocation_projector.py:1675` asserts the token is still `None` after a dry run |
| 8 | The recorded provenance is not writable from any staff surface | ✓ VERIFIED | `admin.py:124` (`CalendarEventMetaInline.readonly_fields`) and `admin.py:340` (`CalendarEventMetaAdmin.readonly_fields`) |

#### Plan 35-20 — guard the re-mint branch (CR-01) and make it atomic (CR-03)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | A human-confirmed night is never destroyed by an automated re-mint: same pk, both boundaries, every companion field, counted under `detach_declined`, warning logged | ✓ VERIFIED | `_remint_decline_reason()` `:598-649` + call site `:987-1012`; `test_confirmed_night_survives_a_would_be_remint` asserts `detach_declined == 1`, `retired == 0`, `created == 0` on a FRESH query after the whole `reconcile_run()` returns. Green |
| 2 | The same protection covers `observation_record`, `observation_group` and `is_verified=False` (reviewer probe 9) | ✓ VERIFIED | `:647` checks all three; one named test each, all green; the `staff-set state` warning appears three times in my captured output |
| 3 | A declined night survives the WHOLE call — its url joins `active_urls` so the D-14 convergence step cannot delete it | ✓ VERIFIED | `:1011` `active_urls.add(url)`; convergence excludes `active_urls` at `:1114-1115`; the test re-queries after `reconcile_run()` has returned |
| 4 | A dry-run preview and the real run reach identical decisions and counters for a declined re-mint | ✓ VERIFIED | Guard at `:987` runs before the `dry_run` short-circuit at `:1020`; `_clearable_declined_and_unattributed()` is read-only by its own documented contract (re-read in `campaign_reconciler.py`); `test_dry_run_parity_for_a_declined_night` green |
| 5 | An inverted sub-night edit no longer destroys the night before failing — `_mint_fields()` runs before `existing.delete()` | ✓ VERIFIED | `:1033` precedes `:1041`; `test_compute_before_destroy_leaves_the_event_in_place_on_an_inverted_span` asserts same pk/start/end after `assertRaises(ValueError)`. Green |
| 6 | Delete and re-create are one unit — a failure between them rolls the delete back | ✓ VERIFIED | `from django.db import transaction` `:37`; `with transaction.atomic():` `:1034` wrapping delete/create/link/record; `test_a_failure_between_the_delete_and_the_create_rolls_back` injects a `RuntimeError` into `insert_or_create_calendar_event` and asserts the event survives. Green |
| 7 | The ordinary re-mint path did not regress — new pk, real `sun_event()` boundaries, `retired 1 / created 1` | ✓ VERIFIED | `test_unconfirmed_night_still_remints_normally`; plus `TestClearedSubNightFieldRemints` unedited and green |
| 8 | Each failure ordering has its own named test rather than sharing one | ✓ VERIFIED | `TestRemintAtomicity` has exactly two tests, one per mechanism, with a docstring saying why one test cannot distinguish them |
| 9 | ALLOC-03's handoff contract is untouched — pinned by `TestRetirePathLegacyEventGuard` / `TestObservationHandoff` staying green unedited | ✓ VERIFIED | `git diff 61f0df04..HEAD -- solsys_code/tests/test_allocation_projector.py` has no hunk inside either class (the only non-append hunks are the import block and one assertion inside `TestUnrecordedProvenanceNight`). Both classes green in my run |

#### Plan 35-21 — carry the site in the provenance token (CR-02)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | A site correction on an already-projected run re-mints every night to the new site's real sun events | ✓ VERIFIED | `TestSiteChangeRemints` reproduces the reviewer's probe 1 and asserts `retired 1 / created 1 / unchanged 0`, new pk, boundaries from the Australian site. Green |
| 2 | The token carries every input the minted boundaries depended on, as the plan defines that set: version, `site_id`, sub-night pair (night comes from the event key) | ✓ VERIFIED (coincidental-reliance) | `:443-446`. Verified as worded; see `coincidental_reliance_items` — the token carries the site's IDENTITY, not its coordinates, and an in-place `Observatory` edit is consequently invisible (escalated below) |
| 3 | `telescope_instrument` and `campaign` are deliberately NOT in the token | ✓ VERIFIED | Absent from `:443-446`; the rationale is in the function docstring `:422-426` |
| 4 | A pre-release-format token reads as unrecorded — a version-prefix test, not `is not None` | ✓ VERIFIED | `:566` `recorded_token.startswith(f'{_PROVENANCE_TOKEN_VERSION}|')`; `test_pre_release_token_reads_as_unrecorded_and_resolves_once` writes the literal `'none|none'` and asserts one `sun_event()` call plus a `v2|` token afterwards. Green |
| 5 | `NULL`, `''` and a pre-release token each read as unrecorded, each a named test | ✓ VERIFIED | Three named tests in `TestProvenanceTokenFormat`, all green |
| 6 | The tolerance boundary is pinned on both sides (`> tolerance`, not `>=`) | ✓ VERIFIED | `test_a_boundary_exactly_at_the_tolerance_resolves_as_correct` and `test_a_boundary_one_microsecond_beyond_the_tolerance_remints`; the beyond-tolerance warning appears in my captured output. Green |
| 7 | The column is wide enough, proven against the field's own declared `max_length` | ✓ VERIFIED | `test_worst_case_token_fits_within_the_declared_max_length` reads `CalendarEventMeta._meta.get_field('minted_sub_night_window').max_length` rather than hardcoding 64. Green |
| 8 | The token's meaning is stated in the class docstring, the field comment and the function docstring | ✓ VERIFIED | `models.py:38-56`, `models.py:122-129`, `allocation_projector.py:394-442` — all three rewritten; the old "sub-night pair alone" wording is gone from each |
| 9 | A change to ANY recorded mint input re-mints, walked by a named invariant test | ✓ VERIFIED | `TestMintInputInvariant` (sub-night pair, site, and a source-inspection test). Green. Its docstring states honestly that it cannot detect an input nobody wrote a case for |
| 10 | The round is not complete until 35-22 runs | ✓ VERIFIED | 35-22 executed (`4385742`, `3ad0797`, `0afdfe3`) |
| 11 | ALLOC-02 restored on a site correction, pinned by the cross-hemisphere test | ✓ VERIFIED | `TestSiteChangeRemints` (`America/Santiago` → `Australia/Sydney`) |
| 12 | `load_telescope_runs` untouched, pinned by its tests staying green unedited | ✓ VERIFIED | Not in `git diff 61f0df04..HEAD`; its module green in my own 64-test run |

#### Plan 35-22 — paired docs for the round

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | The runbook's `retired` and `detach_declined` paragraphs describe what the sweep now does, including the operator remedy for a declined re-mint | ✓ VERIFIED | Runbook diff read in full: four reasons → five, site correction folded into reason (2), provenance audit as reason (5), `detach_declined` split into declined-release and declined-re-mint with an explicit remedy ("clear the confirmation or the link ... and re-run the sweep") |
| 2 | The notebook carries executed output for BOTH behaviour changes, regenerated by re-execution | ✓ VERIFIED | Cell 18 (exec 11): site correction, `retired=3 created=3`, boundaries land on the corrected site's own `sun_event()` values. Cell 19 (exec 12): `detach_declined=1`, decline warning captured, pk/boundaries/confirmation all preserved. 20/20 code cells non-null, strictly increasing 1..20, zero error outputs |
| 3 | This plan is what completes the round | ✓ VERIFIED | Both behaviour changes are documented; no `docs/` residual from 35-20 or 35-21 remains except the `rekeyed` sentence recorded as advisory |
| 4 | The developer database is never written | ✓ VERIFIED | Notebook cell 2 copies `src/fomo_db.sqlite3` to a scratch temp file and exports `FOMO_DATABASE_PATH` BEFORE `django.setup()`; `git status --porcelain -- src/fomo_db.sqlite3` and `git diff --stat 61f0df04..HEAD -- src/fomo_db.sqlite3` are both empty |
| 5 | The code surface is still green behind the documentation | ✓ VERIFIED | My own runs: 20 new tests OK, 44 regression tests OK, 64 cutover+loader tests OK; `pre-commit run ruff` Passed, `ruff-format` Passed, `sphinx-build` Passed, `makemigrations --check --dry-run` → "No changes detected" |
| 6 | ALLOC-05's cutover sequencing is untouched in code, and the post-upgrade one-time audit is now written down | ✓ VERIFIED | `cutover_classical_allocations.py` absent from the round-5 diff; the deploy note is in the runbook immediately after the `detach_declined` block |

**Score:** 221/221 truths verified (216 plan truths + 5 ROADMAP success criteria; 0 failed; 0 behaviour-unverified; 0 prohibitions flagged; 0 overrides).

### Regression Checks — plans 35-01 … 35-18

| Check | Result |
|---|---|
| `git diff 61f0df04..HEAD` file set | `allocation_projector.py`, `models.py`, new migration 0020, `test_allocation_projector.py`, the runbook, the reconciler notebook, and `.planning/` only. No other source file touched by round 5 |
| Test-file deletions since the review commit | 3 lines: 2 are the single-line import reformatted into a multi-line import block, 1 is a documented Rule-1 fix replacing a hardcoded `'none|none'` literal with `_sub_night_provenance_token(run)` inside `TestUnrecordedProvenanceNight`. Every other change is appended at the end of the file |
| "Must stay unedited" classes named in 35-20/35-21 | `TestRetirePathLegacyEventGuard`, `TestFinalConvergenceGuard`, `TestObservationHandoff`, `TestClearedSubNightFieldRemints`, `TestNoSunEventRecompute`, `TestSubNightWindow` — no diff hunk inside any of them; all green in my own 44-test run. `test_load_telescope_runs.py` byte-unchanged |
| `_clearable_declined_and_unattributed()` (35-20 prohibition 3) | `campaign_reconciler.py` absent from the round-5 diff — unmodified |
| Migration 0020 (35-21 prohibition 2) | Bare `AlterField`, no `RunPython`, no data step |
| 35-22 prohibition 3 (no source file from a docs plan) | `4385742` and `3ad0797` touch `docs/` only |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `_span_needs_remint()` returns True | the guard | `_remint_decline_reason()` at `:987` | ✓ WIRED | Nothing runs between the decision and the guard; the guard runs before both counters and before the dry-run short-circuit |
| a declined night's url | the D-14 convergence step | `active_urls.add(url)` `:1011` → `exclude(url__in=active_urls | retired_urls)` `:1114` | ✓ WIRED | Pinned by the fresh re-query in `test_confirmed_night_survives_a_would_be_remint` |
| `_mint_fields()`'s raise | the caller, event intact | reorder `:1033` before `:1041` | ✓ WIRED | `TestRemintAtomicity` test 1 |
| delete / create / link / record | one rollback boundary | `transaction.atomic()` `:1034` | ✓ WIRED | `TestRemintAtomicity` test 2 |
| `run.site` | the recorded token | `_sub_night_provenance_token()` `:445` | ✓ WIRED | `site_id` present in the emitted token (`v2|1|none|none` observed live) |
| an `Observatory` row's coordinates | the recorded token | — | ✗ NOT WIRED | Escalated as Human Verification item 1: the token carries `site_id`, not the site's position, so an in-place correction is invisible. Pre-existing, outside every must-have's wording |
| a pre-release token | the bounded one-time resolution | version-prefix test `:566` → `:571` | ✓ WIRED | Three named tests |
| the sweep's counters | the operator | runbook `retired`/`detach_declined`/deploy note | ✓ WIRED | Runbook diff read in full; `sphinx-build` Passed |

### Data-Flow Trace (Level 4)

| Artifact | Data value | Source | Produces real data | Status |
|---|---|---|---|---|
| `CalendarEvent.start_time`/`end_time` on a re-mint | night boundaries | `_mint_fields()` → `sun_event(run.site, night, 'sun')` → `night_bounds()` | Yes — asserted against live `sun_event()` in tests, and observed live in notebook cell 18 | ✓ FLOWING |
| `CalendarEventMeta.minted_sub_night_window` | provenance token | `_sub_night_provenance_token(run)` written at `:1047` / `:1096` / `:594` | Yes — `v2|1|none|none` read back from a real test DB in my own probe | ✓ FLOWING |
| `ReconcileResult.detach_declined` | decline count | `totals['detach_declined'] += 1` at `:1007` (and `:937`) | Yes — `detach_declined=1` in notebook cell 19's executed output | ✓ FLOWING |
| Allocation night boundaries after an in-place `Observatory` edit | corrected sun events | nothing — the stored value from before the edit survives | No | ✗ DISCONNECTED (escalated, not a must-have) |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|---|---|---|---|
| The four new/changed test classes for CR-01/CR-02/CR-03 pass | `python manage.py test ...TestRemintHumanConfirmationGuard ...TestRemintAtomicity ...TestSiteChangeRemints ...TestProvenanceTokenFormat ...TestMintInputInvariant` | `Ran 20 tests in 11.6s — OK`, with the decline and beyond-tolerance warnings visible in the captured log | ✓ PASS |
| The must-stay-unedited regression classes still pass | `python manage.py test ...TestClearedSubNightFieldRemints ...TestUnrecordedProvenanceNight ...TestNoSunEventRecompute ...TestObservationHandoff ...TestRetirePathLegacyEventGuard ...TestFinalConvergenceGuard ...TestSubNightWindow ...TestSubNightWindowSiteDirection` | `Ran 44 tests in 32.9s — OK` | ✓ PASS |
| ALLOC-04 / ALLOC-05 modules unaffected by the token change | `python manage.py test solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs` | `Ran 64 tests in 37.1s — OK` | ✓ PASS |
| No schema drift | `python manage.py makemigrations --check --dry-run` | `No changes detected` | ✓ PASS |
| Lint / format gates (D-07) | `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files` | Passed, Passed | ✓ PASS |
| Docs build | `pre-commit run sphinx-build --all-files` | Passed | ✓ PASS |
| Notebook regenerated, not hand-patched | JSON inspection of `reconcile_campaign_runs_demo.ipynb` | 20 code cells, 0 null `execution_count`, strictly increasing 1..20, 0 error outputs | ✓ PASS |
| Developer database untouched | `git status --porcelain -- src/fomo_db.sqlite3` | empty | ✓ PASS |
| In-place `Observatory` correction re-mints? | throwaway `TestCase` probe against a migrated test DB (deleted after the run) | `unchanged=1`, same pk, boundaries frozen ~15 h from the corrected site's true sun events | ✗ FAIL → escalated as Human Verification item 1 (no must-have covers it) |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| — | `find scripts -path '*/tests/probe-*.sh'` | no `scripts/` directory in this repo; no probe declared by any 35-\* plan | N/A — SKIPPED (no probe scripts in this project) |

### Decision Coverage

All trackable `35-CONTEXT.md` decisions are honored by shipped artifacts — 18/18, none missing. (Non-blocking gate.)

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| ALLOC-01 | 35-01, 35-03, 35-13, 35-15, 35-16, 35-19, 35-20, 35-21, 35-22 | Per-night events for resolved classical/awarded allocations; container for queue/class-wide/satellite | ✓ SATISFIED | SC-1 restored. The prior pass's BLOCKED mark is cleared: the cleared-to-null transition, the site correction and the unrecorded-provenance legacy night are each detected, each counted, each logged. REQUIREMENTS.md line 113's "Complete" mark is now earned |
| ALLOC-02 | 35-01, 35-21, 35-22 | Site-local observing nights, both hemispheres | ✓ SATISFIED | `TestSubNightWindowSiteDirection`, `TestAllocationNightBoundary`, `TestSiteChangeRemints` (Chile → Australia). Green |
| ALLOC-03 | 35-01, 35-04, 35-20, 35-22 | Handoff on link, restore on unlink, observation's own event untouched | ✓ SATISFIED | `TestObservationHandoff` / `TestRetirePathLegacyEventGuard` unedited and green; notebook handoff cell shows `retired=1` with the observation's own event intact |
| ALLOC-04 | 35-05, 35-09, 35-14, 35-17 | Loader writes a collision-safe allocation record, same per-night calendar, idempotent | ✓ SATISFIED | Loader untouched this round; module green in my own run |
| ALLOC-05 | 35-06, 35-07, 35-12, 35-21, 35-22 | Explicit cutover sequencing, no duplicate or orphan | ✓ SATISFIED | Cutover command untouched; its module green; the post-upgrade one-time audit is now stated in the runbook deploy note |

**Orphaned requirements:** none — REQUIREMENTS.md maps exactly ALLOC-01…05 to Phase 35, and every one is claimed by at least one plan.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/models.py` | 322, 418, 431, 474 | `TBD` | ℹ️ Info | Domain vocabulary ("TBD window" = a run with no concrete window), not a debt marker. Pre-existing, outside the round-5 diff |
| `docs/runbooks/telescope_runs_calendar.rst` | 1195 | `TBD` | ℹ️ Info | Same — the documented skip reason `TBD window` |
| `solsys_code/allocation_projector.py` | 1016-1017 | stale comment contradicting the code | ⚠️ Warning | Iteration 8 WR-01, unclosed; recorded under `advisory:` |

No `FIXME`, `XXX`, `HACK`, `PLACEHOLDER` or unreferenced `TODO` in any file this round modified. No stubs, no hollow props, no console-log-only implementations.

### Advisory (New Scope, Unevidenced)

New-scope findings with no deterministic evidence — reported, not blocking, and they do not revert a completed must-have.

| # | Finding | Category | Why Advisory |
|---|---|---|---|
| 1 | WR-01 (iteration 8): the re-mint comment and the `once ever` cost bound both still contradict the dry-run behaviour of the unrecorded-provenance branch | other | Out of round-5 scope (criticals only); reviewer probe exists but I did not re-run it |
| 2 | WR-02 (iteration 8): a `--dry-run` preview can now raise `sun_event()`'s `ValueError` for a blank `Observatory.timezone` | architectural | Out of round-5 scope; confirmed by reading the call site, not re-probed |
| 3 | WR-03 (iteration 8), residual: the `rekeyed` paragraph still promises "same start/end time" with no pointer to the new audit | other | Outside plan 35-22's must_haves, which name three passages only |
| 4 | WR-04(a) (iteration 8): the "round-hour fixtures were never genuine legacy nights" premise lives only in a SUMMARY deviation note | other | Documentation of premise; no behavioural defect |
| 5 | IN-01…IN-07 and WR-04-round-2 carried forward from the prior pass, unchanged | other | None was in round-5 scope; full text in the prior 35-VERIFICATION.md |

### Human Verification Required

#### 1. DECISION — should an in-place `Observatory` correction re-mint the nights already projected at that site?

**Test (reproduction):** Project a night for a run with both sub-night fields null, then edit that same `Observatory` row's `lat` / `lon` / `altitude` / `timezone` in the Django admin without changing `run.site`, and reconcile again.

**Expected vs observed:** I ran this against a migrated Django test database (probe module deleted afterwards; `git status --porcelain -- solsys_code/` clean):

```
PROBE token= v2|1|none|none
PROBE before start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE result= ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0, ...)
PROBE after  start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE true corrected sunset/sunrise= 2026-07-09 07:20:39  2026-07-09 20:57:12
PROBE same pk? True
```

A ~15-hour error, reported as `unchanged`, permanently, with no counter and no log line.

**Why this is a decision and not a gap:** it is the CR-02 signature one input further out, but (a) no must-have as worded is falsified — plan 35-21 truth 2 explicitly defines the boundary input set as `(run.night_start_utc, run.night_end_utc, run.site, night)`, and the token does carry `run.site`; (b) it is entirely pre-existing — this round narrowed the hole rather than opening it; (c) reachability is a manual staff edit of a site definition after projection (`ObservatoryAdmin` declares no `readonly_fields`), not an automated path — `MPCObscodeFetcher.to_observatory()` constructs a NEW `Observatory()` and does not update existing rows. Your options: open a round 6, file it as a follow-up issue, or defer it alongside WR-01/WR-02/WR-03. Candidate fix if actioned: carry the resolved site position (or a hash of it) inside the token, which the `max_length` test already makes safe to widen.

### Gaps Summary

None. No must-have is failed, no artifact is missing or stubbed, no key link that any plan claims is unwired, and no blocking anti-pattern exists. The status is `human_needed` solely because of the escalated decision above — a reproduced, pre-existing staleness path that sits outside every declared must-have and needs a product call rather than a verification call.

---

_Verified: 2026-09-16T15:26:19Z_
_Verifier: Claude (gsd-verifier)_
