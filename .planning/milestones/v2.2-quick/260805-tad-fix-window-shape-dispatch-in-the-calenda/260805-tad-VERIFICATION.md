---
phase: quick-260805-tad
verified: 2026-08-06T05:16:50Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Quick Task 260805-tad: Fix window-shape dispatch in the calendar reconciler Verification Report

**Task Goal:** Fix window-shape dispatch in the calendar reconciler: remove the
`elif run.source in QUEUE_SOURCES:` branch from `reconcile_run()` so a queue-sourced run
with a resolved, non-satellite site (e.g. ESO VLT/FORS2 at MPC 309) gets per-night
dip-corrected classical treatment instead of a blanket 00:00-23:59 whole-window container,
since `telescope_class` already exclusively covers the genuinely site-agnostic/floating case.

**Verified:** 2026-08-06T05:16:50Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | An approved, queue-sourced CampaignRun with a resolved, non-satellite site gets one dip-corrected per-night `RUN:{pk}:{date}` event, identical to a classical run there | VERIFIED | `campaign_reconciler.py:465-507`: `reconcile_run()` has exactly 3 branches (`telescope_class`, satellite site, `else`); a queue-sourced run with a resolved ground site falls to `else` -> `_reconcile_classical_nights()`. Confirmed by 3 new tests (`test_lco_queue_run_with_resolved_site_creates_one_event_per_night`, `..._gemini_...`, `..._eso_...`, lines 115-172) and by regenerated notebook output showing `RUN:2:2026-09-01/02/03` (queue run) with identical shape to `RUN:1:...` (classical run) |
| 2 | A run with a non-blank `telescope_class` still gets exactly one bare `RUN:{pk}` container event, whatever its source | VERIFIED | Code: first branch `if run.telescope_class:` (`campaign_reconciler.py:484-489`) reads no `source`. Test: `test_queue_sourced_run_with_telescope_class_still_gets_one_bare_container` (line 175) is the inverse-control pinning this |
| 3 | A run whose resolved site is a satellite site still gets exactly one bare `RUN:{pk}` container event, whatever its source | VERIFIED | Code: second branch (`campaign_reconciler.py:490-493`) checks `run.site.observations_type == Observatory.SATELLITE_OBSTYPE`, no `source` read. Test: `test_satellite_run_creates_one_container_event_without_calling_sun_event` (line 237). Notebook: satellite run pk=3 shows one bare `RUN:3` container |
| 4 | `reconcile_run()` dispatch reads no `CampaignRun.source` value at all; the reconciler module contains no `QUEUE_SOURCES` reference | VERIFIED | Full read of `solsys_code/campaign_reconciler.py` (508 lines): no `run.source` reference, no `QUEUE_SOURCES` definition. `grep -rn 'QUEUE_SOURCES' solsys_code/ src/ docs/` returns zero hits (the only remaining hits are in gitignored `docs/_build/` stale HTML build output, not source) |
| 5 | A queue-sourced, site-resolved run carrying a pre-fix bare `RUN:{pk}` container converges on next reconcile: per-night events minted, old container survives un-re-keyed/un-re-timed, `CalendarEventMeta.run` detached to None | VERIFIED | `TestReclassificationConvergence.test_pre_fix_container_event_converges_to_per_night_on_next_reconcile` (`test_campaign_reconciler.py:824-879`) read in full: hand-creates the exact pre-fix state (bare container + `CalendarEventMeta.run` pointing at the run), reconciles, asserts `result.created == n_nights`, container's `pk`/`url`/`start_time`/`end_time` unchanged, container not among night event pks, companion row's `run_id` is None, then a second reconcile asserts `unchanged == n_nights` with every `modified` timestamp frozen. This is a substantive, meaningful assertion set, not a smoke test |
| 6 | The reconciler never creates/modifies/deletes an ObservationRecord-derived calendar event, proven on both branches | VERIFIED | `TestRecordEventNonInterference` (line 600, per-night branch) and `TestContainerRecordEventNonInterference` (line 684, container branch) both exist in `test_campaign_reconciler.py`; both assert record event `url`/`title`/`start`/`end`/`modified` unchanged after two reconcile passes |
| 7 | Every pre-existing test that encoded the old source-driven container behaviour was corrected (not merely supplemented); all three test modules pass | VERIFIED | `TestQueueSourceDoesNotChangeShape` (renamed from `TestQueueStage1`, line 108) replaces the old bare-container assertions with per-night assertions. Full suite run independently by this verifier: `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval` -> **173 tests, OK, 183.5s** (matches SUMMARY's claim) |
| 8 | 29-SECURITY.md's T-29-07 evidence no longer claims the per-night branch is unreachable for queue runs, stays closed with `threats_open: 0`, carries a dated audit-trail entry | VERIFIED | `29-SECURITY.md:43` (T-29-07 row): rewritten to cite `_may_write()` as the first condition in both write paths, cites the two non-interference tests by name, Status column still `closed`. Frontmatter `threats_open: 0` (line 6). Audit trail row dated 2026-08-05 added at line 251, scoped explicitly to T-29-07, naming why the other 22 threats were not re-scanned |
| 9 | The paired demo notebook and operator runbook describe the corrected three-branch dispatch; notebook regenerated against a scratch DB with real executed output | VERIFIED | Notebook (`reconcile_campaign_runs_demo.ipynb`) source cells contain no `QUEUE_SOURCES`/stale-dispatch text; regenerated output (read directly) shows queue run pk=2 with 3 per-night `RUN:2:{date}` events (identical shape to classical pk=1), satellite pk=3 and class-wide pk=4 each with one bare container, and `Second sweep ... created: 0, updated: 0` confirming idempotency. Runbook (`docs/runbooks/telescope_runs_calendar.rst`): the stale "queue-scheduled or class-wide run shows a single entry" passage is gone (0 grep hits); corrected passages at lines ~311-329, ~550-560 read as described |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_reconciler.py` | Three-branch dispatch | VERIFIED | Read in full; `def reconcile_run` present with exactly 3 branches, no `source`/`QUEUE_SOURCES` reference |
| `solsys_code/tests/test_campaign_reconciler.py` | Corrected expectations + convergence test + both-branch non-interference | VERIFIED | `TestQueueSourceDoesNotChangeShape`, `TestReclassificationConvergence.test_pre_fix_container_event_converges_to_per_night_on_next_reconcile`, `TestRecordEventNonInterference`, `TestContainerRecordEventNonInterference` all present and read |
| `solsys_code/tests/test_reconcile_campaign_runs.py` | Corrected command-level real-data-shape fixture | VERIFIED | `TestRealDataShapeScenario.test_19_run_fixture_matching_the_real_split_becomes_calendar_visible` reshaped per plan (5 site-resolved queue runs + 3 class-wide queue runs) |
| `.planning/phases/29-the-reconciler/29-SECURITY.md` | Corrected T-29-07 evidence + audit-trail row | VERIFIED | Row rewritten, `closed`, `threats_open: 0`, dated row added |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Regenerated with real output describing corrected dispatch | VERIFIED | Output read directly: per-night events for queue run, containers for satellite/class-wide, idempotent second sweep |
| `docs/runbooks/telescope_runs_calendar.rst` | Corrected operator-facing description | VERIFIED | Three stale passages corrected, confirmed by direct read |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `reconcile_run` | `_reconcile_classical_nights` | final `else` branch | WIRED | `campaign_reconciler.py:495` |
| `reconcile_run` | `_reconcile_container` | `telescope_class`/satellite branches | WIRED | `run.site.observations_type == Observatory.SATELLITE_OBSTYPE` present at line 490 |
| `reconcile_run` | `_detach_stale_family_events` | convergence step | WIRED | `campaign_reconciler.py:505` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full three-module regression | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval` | 173 tests, OK, 183.5s | PASS (run independently by this verifier, not taken from SUMMARY) |
| `QUEUE_SOURCES` absence | `grep -rn 'QUEUE_SOURCES' solsys_code/ src/ docs/` | Only hits in gitignored `docs/_build/` stale HTML output | PASS |
| `ruff check` / `ruff format --check` | on the 4 modified Python/doc files | All checks passed / already formatted | PASS |
| `git status` cleanliness | `git status --short` | only the new SUMMARY.md untracked; operator's `src/fomo_db.sqlite3` present, unmodified | PASS |
| Mutation-probe re-verification | attempted live re-run of Task 2's mutation probe (temporarily restore the `elif run.source in QUEUE_SOURCES` branch and confirm the new convergence test fails) | Blocked by the sandbox's auto-mode classifier before any write occurred (repo file confirmed unmodified afterward, `git diff --stat` empty) | SKIP (see note below) |

**Note on the mutation-probe spot-check:** this verifier attempted to independently reproduce the executor's mutation probe by temporarily re-inserting the deleted branch and re-running the new convergence test, but the action was blocked by this environment's auto-mode classifier (writing to a real repository source file outside a GSD workflow). No workaround was attempted, per the classifier's guidance. As a substitute, the claim was checked by logical inspection of the dispatch order: with the old `elif run.source in QUEUE_SOURCES:` branch reinserted between the satellite branch and the final `else`, a queue-sourced run would be intercepted before reaching `_reconcile_classical_nights()`, so `_reconcile_container()` would run instead and `result.created` would be `1` (not `n_nights == 2`), which the test's `self.assertEqual(result.created, n_nights)` assertion (line 854) would catch. This confirms the mutation-probe claim is logically sound, though it was not independently re-executed. This is a WARNING-level gap in verification depth, not a finding against the phase goal — the code and test as they exist today were fully read and are internally consistent with the claimed probe outcome.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TAD-01..04 | 260805-tad-PLAN.md | Ad-hoc quick-task requirements (dispatch correctness, convergence, non-interference, security evidence accuracy) — not tracked in central REQUIREMENTS.md, as expected for a quick task | SATISFIED | Covered by truths 1-6, 8 above |

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` debt markers in any modified file (the one `'TBD window'` grep hit is a legitimate skip-reason string literal, not a debt marker). No stub returns, no empty handlers.

### Human Verification Required

None. All truths are verifiable from code, tests, and generated docs artifacts.

### Gaps Summary

No gaps. All 9 must-have truths verified against the actual codebase (not merely the SUMMARY's narrative): the three-branch dispatch was read in full and contains no residual `source`/`QUEUE_SOURCES` reference; the full 173-test regression was re-run independently by this verifier and passed; the convergence test and both non-interference test classes were read in full and are substantive (real state manipulation and multi-field assertions, not smoke tests); the security evidence document, forward-pointers, notebook output, and runbook prose were all read directly and match the corrected dispatch. One minor verification-depth note: the mutation-probe re-execution was blocked by this environment's sandbox and was substituted with logical code-path inspection instead of a live re-run (see spot-check note above) — this does not constitute a gap in the phase's goal achievement.

---

_Verified: 2026-08-06T05:16:50Z_
_Verifier: Claude (gsd-verifier)_
