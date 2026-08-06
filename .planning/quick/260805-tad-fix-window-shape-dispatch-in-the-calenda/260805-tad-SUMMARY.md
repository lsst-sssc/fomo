---
phase: quick-260805-tad
plan: 01
subsystem: campaign-reconciler
tags: [reconciler, calendar, dispatch, security, docs]
dependency-graph:
  requires: [RECON-02, RECON-03, T-29-07]
  provides: [three-branch-dispatch, corrected-T-29-07-evidence]
  affects: [campaign_reconciler.py, reconcile_campaign_runs, docs/runbooks/telescope_runs_calendar.rst]
tech-stack:
  added: []
  patterns: [ownership-scoped-write-guard, dated-forward-pointer]
key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
    - .planning/phases/29-the-reconciler/29-SECURITY.md
    - .planning/phases/29-the-reconciler/29-CONTEXT.md
    - .planning/phases/29-the-reconciler/29-RESEARCH.md
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
decisions:
  - "Dispatch now reads only telescope_class (non-blank -> container) and site (satellite -> container); everything else, including every queue-sourced run with a resolved ground site, takes the per-night branch. source is no longer read by reconcile_run() at all."
  - "T-29-07 stays closed: the real, unchanged protection was always _may_write() being the first condition in both write paths, not the now-removed source branch."
metrics:
  duration: "~2h"
  completed: 2026-08-06
---

# Phase quick-260805-tad Plan 01: Fix window-shape dispatch in the calendar reconciler Summary

Removed `reconcile_run()`'s unreachable, wrong `elif run.source in QUEUE_SOURCES:` dispatch
branch -- it could only ever fire for a run that already had a resolved, non-satellite site
(the live case: RUN:3, ESO VLT/FORS2 at MPC 309), giving it a blanket whole-window container
instead of the per-night dark-time events a fixed-site run needs.

## What Changed

`reconcile_run()` went from four dispatch branches to three:

1. `if run.telescope_class:` -> container (unchanged)
2. `elif run.site is not None and run.site.observations_type == SATELLITE_OBSTYPE:` -> container (unchanged)
3. `else:` -> classical per-night branch, now also reached by every queue-sourced run with a
   resolved, non-satellite site

The deleted branch and its `QUEUE_SOURCES` frozenset are gone from the codebase entirely
(verified by repo-wide grep in every task's automated check). `run.source` is no longer read
anywhere in `campaign_reconciler.py`.

## Per-Test Before/After Expectation Table (Task 1)

| Test | Before (wrong) | After (corrected) |
|------|-----------------|---------------------|
| `TestQueueStage1` (3 tests) -> renamed `TestQueueSourceDoesNotChangeShape` | LCO/Gemini/ESO queue run + resolved site -> asserted one bare `RUN:{pk}` container | Asserts one `RUN:{pk}:{date}` event per night, no bare container at all; window shrunk 15->2 nights per test |
| `TestQueueSourceDoesNotChangeShape` (new 4th test) | n/a | New inverse-control test: a queue-sourced run that is ALSO class-wide still gets exactly one bare container -- pins "telescope_class decides, not source" |
| `TestOwnershipScoping.test_event_owned_by_a_different_run_is_blocked_and_untouched` | Clashing event keyed at bare `RUN:{pk}` (5-night window) | Single-night window; clashing event keyed `RUN:{pk}:2026-08-01` (the run now takes the per-night branch) |
| `TestOwnershipScoping.test_unowned_same_window_event_is_left_completely_untouched` | 5-night window | Shrunk to 2 nights; same per-night branch, same assertions |
| `TestContainerIdempotency` (2 tests) | Fixture used `source=LCO_QUEUE` to reach the container branch | Fixture now uses `telescope_class=ONE_M0`, `site=None` to reach the (still real) container branch |
| `TestQueueOwnershipDoesNotTouchRecordEvents` -> renamed `TestRecordEventNonInterference` | Asserted one bare container + record event untouched; `owned_events(run).count() == 1` | Window shrunk 5->2 nights; asserts N per-night events + record event untouched; `owned_events(run).count() == 2` (one per night), no bare container exists |
| `TestReclassificationConvergence.test_reclassifying_classical_to_queue_detaches_old_per_night_events` -> renamed `..._to_class_wide_...` | Trigger: `run.source = LCO_QUEUE` (no longer reclassifies anything) | Trigger: `run.telescope_class = ONE_M0` (genuinely reclassifies); same detach assertions |
| `TestReclassificationConvergence.test_stale_container_event_is_not_adopted_into_a_classical_night` | First reconcile used `source=LCO_QUEUE` to produce a container | First reconcile uses `telescope_class=ONE_M0` to produce a container, then clears it to reclassify into the per-night family |
| `TestCampaignRunDeletionCascadesCalendarEvents.test_deleting_a_run_deletes_its_owned_calendar_events` | Fixture `source=LCO_QUEUE` | Fixture `telescope_class=ONE_M0`, `site=None` |
| `TestCrossRunOwnershipGuards.test_deleting_a_run_still_deletes_the_events_it_genuinely_owns` | Fixture `source=LCO_QUEUE` | Fixture `telescope_class=ONE_M0`, `site=None` |
| `TestTelescopeInstrumentSplitOnEvents` (3 container-branch tests) | Fixture `source=LCO_QUEUE` to reach the container write path | Fixture `telescope_class=ONE_M0`, `site=None` |
| `test_reconcile_campaign_runs.py::_seed_mixed_runs` | `queue_run`: 6-night window, docstring implied 3 distinct calendar shapes | `queue_run`: 2-night window (now per-night, same shape as classical); docstring corrected |
| `TestRealDataShapeScenario.test_19_run_fixture_matching_the_real_split_becomes_calendar_visible` | 8 queue-sourced runs each asserted one bare container | Reshaped: 5 of the 8 queue-sourced runs are site-resolved (2-night windows -> per-night events, mirroring real ESO VLT queue rows); the other 3 are class-wide (`telescope_class` set, `site=None` -> one bare container each, mirroring real LCO class-wide allocations). 19 runs / 8-11 split preserved; `runs: 19, failed: 0, skipped: 0, blocked: 0` unchanged |

Everything else in both files (skip-reason tests, satellite-container tests, class-wide tests,
adopt/re-key tests, the two `TestCrossRunOwnershipGuards` tests using default classical
fixtures, `TestFailureIsolation`, `TestIdempotency`, `TestDryRun`) needed no change -- confirmed
by running them, not assumed.

## Task 2: Convergence Proof and Mutation-Probe Outcome

Added `TestReclassificationConvergence.test_pre_fix_container_event_converges_to_per_night_on_next_reconcile`:
hand-creates the exact pre-fix state RUN:3 was in (a bare `RUN:{pk}` container event with
`CalendarEventMeta.run` pointing at a queue-sourced, site-resolved run), then reconciles.
Asserts: one per-night event minted per night; the container survives with its original `pk`,
`url`, `start_time`, `end_time` (never re-keyed, never re-timed, never deleted, never adopted
into a night slot); its companion row's `run_id` is now `None`; a second reconcile reports
`unchanged` for every night with no `modified` churn anywhere.

**Mutation-probe outcome:** temporarily re-added the deleted `elif run.source in {LCO_QUEUE,
GEMINI_QUEUE, ESO_QUEUE}:` branch (local frozenset, not the module-level one) to
`reconcile_run()`, ran the new convergence test alone -- it **FAILED** as expected
(`AssertionError: 0 != 2` on `result.created`), confirming the test is a genuine guard for this
change and not something a pre-existing test already covered. Reverted the mutation and
confirmed `campaign_reconciler.py` was byte-identical to its post-Task-1 state (`git diff
--quiet` clean) before committing.

Also added `TestContainerRecordEventNonInterference`, the container-branch twin of
`TestRecordEventNonInterference`: a class-wide run's own container write never touches an
`ObservationRecord`-derived, LCO-portal-keyed event either. Both classes together are cited as
evidence for T-29-07 in Task 3.

## Task 3: Security Documentation

`29-SECURITY.md`'s T-29-07 evidence previously claimed "queue runs dispatch to the container
branch only ... the per-night branch is unreachable for them", citing the now-deleted
`QUEUE_SOURCES` branch. Corrected to the real, unchanged mechanism: `_may_write()` is the first
condition checked in both `_reconcile_container()` (`campaign_reconciler.py:266`) and
`_reconcile_classical_nights()` (`:387`) -- re-verified against the post-fix file, not assumed.
Row stays `closed`, frontmatter `threats_open: 0` unchanged. Added a dated
2026-08-05 audit-trail row scoping the re-verification to T-29-07 only, naming why the other 22
threats were not re-scanned (the change removes a branch selector only, touches no
ownership/approval-gate/dry-run code path). The trust-boundary row now names
`telescope_class`/`site` only, not `source`.

`29-CONTEXT.md`'s D-07 and `29-RESEARCH.md`'s Pitfall 1 (both of which locked "branch purely on
`run.source`" as the rule) each got a dated forward-pointer, not a rewrite: the actual concern
those sections cared about (no free-text heuristic over `telescope_instrument`/`site_raw`) is
preserved, since dispatch now reads only the structured `telescope_class` and `site` fields.
Nowhere does any artifact describe this change as violating or relaxing RECON-02/RECON-03 --
26-DECISION.md's Criterion 3 is unchanged and still fully implemented; what changed is how "no
fixed observing site" is detected.

## Task 4: Paired Notebook and Runbook

**Notebook** (`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`): added a fourth
seeded `CampaignRun` -- a satellite run using the `X30` site the notebook already seeded but
never used -- alongside the classical, queue and class-wide runs. Rewrote the intro bullets, the
dispatch-order markdown cell, and the "two coexisting key families" cell to describe the
corrected three-branch dispatch. Regenerated per the `260805-sgf` recipe: no operator
`src/fomo_db.sqlite3` existed in the working tree, so `python manage.py migrate` created a fresh
empty one, `jupyter nbconvert --to notebook --execute --inplace` ran from
`docs/notebooks/pre_executed/`, and the scratch DB was deleted afterward (nothing to restore).

Regenerated output confirms the corrected shape:
```
Classical run:  pk=1  window=2026-09-01..2026-09-03  -> 3 per-night RUN:1:{date} events
Queue run:      pk=2  window=2026-09-01..2026-09-03  -> 3 per-night RUN:2:{date} events (same shape as classical)
Satellite run:  pk=3  window=2026-09-01..2026-09-05  -> 1 bare RUN:3 container event
Class-wide run: pk=4  window=2026-09-01..2026-09-30  -> 1 bare RUN:4 container event
Second sweep: created: 0, updated: 0
```

**Runbook** (`docs/runbooks/telescope_runs_calendar.rst`): replaced the stale "correcting
`source` moves a run between families" worked example with one that actually moves a run today
(setting `telescope_class`, or correcting `site` to a satellite site); corrected "what an
operator sees on the calendar" to say any run with a resolved ground site gets per-night entries
regardless of scheduling method (queue or classical); dropped "queue-scheduled" from the
self-healing Telescope/Instrument-split list, since a site-resolved queue run's per-night
entries now follow the same create-only field rule as any other per-night entry; added a
sentence noting a queue-scheduled run at a site with no `timezone` now fails per-night
calculation the same way a classical run there does.

## Verification

- `solsys_code.tests.test_campaign_reconciler` + `test_reconcile_campaign_runs`: 47 tests, OK
  (measured 84.2s -- Task 1's own combined run measured 47 tests in 84.2s; the plan's cited
  baseline was 46.7s for 46 tests before this change; the corrected suite, with the added
  inverse-control test and shrunk queue-run windows, stays under the ~90s budget)
- Full three-module regression (`test_campaign_reconciler` + `test_reconcile_campaign_runs` +
  `test_campaign_approval`): 173 tests, OK (measured twice: 188.2s and 185.7s)
- `ruff check` / `ruff format --check` clean on every file this task modified (two pre-existing,
  unrelated findings exist elsewhere in the repo -- an untouched notebook missing a docstring,
  and two untouched scratch files/settings.py needing reformatting -- both out of this task's
  scope per CLAUDE.md's file list and left alone)
- `! grep -rn 'QUEUE_SOURCES' solsys_code/ src/ docs/` -> zero hits
- `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build`
  built clean (exit 0; pre-existing toctree warnings for other not-yet-written demo notebooks,
  unrelated to this change)
- `git status` clean; no scratch `src/fomo_db.sqlite3` staged or left behind

## Deviations from Plan

None. Plan executed exactly as written, task by task, with per-task commits.

## Self-Check: PASSED

- `solsys_code/campaign_reconciler.py`: FOUND, `QUEUE_SOURCES` absent (confirmed by grep)
- `solsys_code/tests/test_campaign_reconciler.py`: FOUND, 45 tests, all pass
- `solsys_code/tests/test_reconcile_campaign_runs.py`: FOUND, all pass
- `.planning/phases/29-the-reconciler/29-SECURITY.md`: FOUND, `threats_open: 0`, T-29-07 evidence corrected
- `.planning/phases/29-the-reconciler/29-CONTEXT.md`: FOUND, D-07 forward-pointer present
- `.planning/phases/29-the-reconciler/29-RESEARCH.md`: FOUND, Pitfall 1 forward-pointer present
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`: FOUND, regenerated with real output
- `docs/runbooks/telescope_runs_calendar.rst`: FOUND, three stale passages corrected
- Commits: `7473eeb`, `1a42093`, `bab0fa3`, `8b08776` -- all present in `git log --oneline`
