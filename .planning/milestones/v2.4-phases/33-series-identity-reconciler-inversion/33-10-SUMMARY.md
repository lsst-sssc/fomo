---
phase: 33-series-identity-reconciler-inversion
plan: 10
subsystem: campaign-reconciler
tags: [django, campaign-reconciler, calendar-events, attribution, gap-closure]

# Dependency graph
requires:
  - phase: 33 (plan 09)
    provides: FOMO_DATABASE_PATH scratch-copy mechanism the paired notebooks now execute
      against, and the residue cleanup that keeps the campaign-attributed fixture clean
  - phase: 33 (plan 11)
    provides: the calendar pop-up runbook section, landed first so this plan's runbook
      edits (disjoint sections) apply to a page that already contains it
provides:
  - a human-confirmation guard (_stale_attributions()) that makes the automated reconciler
    sweep skip any companion row whose confirmed_by is set, closing the CR-04 confirm/erase
    loop (UAT option B, 2026-09-09)
  - ReconcileResult.detach_declined, reported in the reconcile_campaign_runs summary, a
    per-run stderr line, and a staff-facing message on all four staff actions
  - ownership decided before a classical night's outcome in _reconcile_classical_nights()
    (WR-13), so an attributed-and-contested night reports blocked, not skipped, and never
    detaches a foreign attribution
  - a --dry-run detach preview that reports a real number instead of "n/a" (WR-11), built
    on the same predicate the real sweep detaches on
  - _resolve_site()'s success message re-keyed on created/updated/skipped_nights so "run
    added to the calendar" only appears when something was added (WR-12)
  - both paired demo notebooks re-executed against the post-fix code, and the runbook's
    re-classification/counter/skip-rule sections corrected to match
affects: [34-observation-projector-and-trigger, 36-unattended-operation]

# Actuals (#2632)
actuals:
  tokens: 26558
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A human-confirmation guard that must NOT reach a shared clear-the-link helper's
      other (human-initiated) callers lives on the caller side that needs it, not inside
      the shared helper: _stale_attributions() splits stale rows on confirmed_by before
      calling campaign_utils.unlink_event_from_run(), rather than adding the guard to
      unlink_event_from_run()/UNLINK_CLEARED_FIELDS itself."
    - "Ownership (_may_write()) is evaluated before the outcome-specific skip/attribution
      check in a per-item resolution loop, and the blocked branch still adds the item's
      identity to whatever 'currently active' set a later convergence step reads -- so a
      contested item is neither written nor swept up by that later step."
    - "A dry-run preview and the real write share one read-only predicate function
      (_stale_attributions()), consumed by both the preview branch and the write branch,
      so the preview cannot drift from what a real run would do."

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/campaign_views.py
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_approval.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "_stale_attributions(run, active_urls) is the single read-only predicate consumed by
    both _detach_stale_family_events() (the write path) and reconcile_run()'s dry-run
    branch -- one query shape, two consumers, so the --dry-run preview number is
    guaranteed to match what a real sweep would detach (WR-11)."
  - "The human-confirmation guard was added to campaign_reconciler.py only, never to
    campaign_utils.unlink_event_from_run()/UNLINK_CLEARED_FIELDS -- Phase 28's undo view
    and the admin's clear branch are human callers and must keep clearing a confirmed row
    (this plan's own prohibition, verified by the unchanged
    test_campaign_attribution_views/test_admin suites)."
  - "_reconcile_classical_nights() now fetches the existing event and evaluates
    _may_write() BEFORE the attributed-night skip check, and the blocked branch adds the
    night's url to active_urls itself -- ownership is decided before the night's outcome
    (WR-13), so a night that is both attributed to this run and contested by a foreign
    RUN:-keyed attribution reports blocked==1, skipped_nights==0, and never detaches the
    foreign attribution."
  - "Pre-existing test test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched
    (33-04) asserted unconditional stamp-clearing on detach; it was split into two tests --
    one for the still-detached unconfirmed case (unchanged assertions) and a new one for
    the now-declined confirmed case -- rather than deleted, since both branches needed
    coverage under the new rule."
  - "No CalendarEventDismissal row is ever written on the automated detach path (verified
    by a grep gate on campaign_reconciler.py and by an explicit assertion in the new
    confirm/erase regression test) -- the UAT decision explicitly rejected the
    dismissal-row remedy from 33-REVIEW.md CR-04's fix block."

patterns-established:
  - "Reconciler docstrings on detach line paths say 'may or may not release a companion
    row -- see detach_declined' rather than asserting unconditional clearing, now that a
    detach step can legitimately decline."

requirements-completed: [ANNOT-01]

coverage:
  - id: D1
    description: "A human-confirmed attribution survives an unattended reconcile sweep;
      the sweep reports the declined count instead of silently repeating the erasure
      (CR-04 confirm/erase loop, UAT option B)"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_staff_reconfirmation_of_the_detached_run_keyed_event_survives_every_later_sweep"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_real_sweep_reports_declined_for_a_human_confirmed_superseded_row"
        status: pass
      - kind: integration
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb (confirm/erase cell, executed output: detached=0, detach_declined=1 across two further sweeps and a --dry-run preview)"
        status: pass
    human_judgment: false
  - id: D2
    description: "An unconfirmed superseded row is still detached exactly as 33-08 shipped
      it (CR-03 not weakened)"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third"
        status: pass
    human_judgment: false
  - id: D3
    description: "A night both attributed to this run and contested by a foreign
      RUN:-keyed attribution reports blocked==1, keeps the url active, and never detaches
      the foreign attribution (WR-13)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_attributed_and_contested_night_is_blocked_not_skipped_and_never_detached"
        status: pass
    human_judgment: false
  - id: D4
    description: "--dry-run previews the detach count on the same predicate the real
      sweep uses, and writes nothing (WR-11)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_dry_run_previews_the_detach_count_and_writes_nothing"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSkipAndDetachCounters.test_dry_run_previews_the_would_detach_count_and_writes_nothing"
        status: pass
    human_judgment: false
  - id: D5
    description: "All four staff actions (approve, resolve site, mark cancelled, mark
      weather failure) surface a release/decline via one shared message helper; 'run
      added to the calendar' only appears when something was added (WR-12)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_with_every_night_already_covered_reports_no_new_entries"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSitesNeedingReview.test_resolve_that_detaches_something_shows_the_warning"
        status: pass
    human_judgment: false
  - id: D6
    description: "Both paired demo notebooks re-executed against the post-fix code with
      output committed; the reconcile notebook's output demonstrates the confirm/erase
      survival; the runbook's three owned sections (re-classification, counter docs,
      skip-rule) are corrected and the calendar pop-up section (plan 33-11's) is
      byte-identical"
    verification:
      - kind: other
        ref: "python -c \"json cells-without-output scan\" on both notebooks, plus grep gates on telescope_runs_calendar.rst (re-confirm or discard: 0, would_detach: n/a: 0, detach_declined: >=1, Bootstrap 5: >=1)"
        status: pass
    human_judgment: false
  - id: D7
    description: "No CalendarEventDismissal row is ever written by an automated detach"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReconcileThenAttributeOrdering.test_staff_reconfirmation_of_the_detached_run_keyed_event_survives_every_later_sweep (CalendarEventDismissal.objects.count() == 0 assertion)"
        status: pass
      - kind: other
        ref: "grep -c 'CalendarEventDismissal' solsys_code/campaign_reconciler.py == 0"
        status: pass
    human_judgment: false

# Metrics
duration: ~70min
completed: 2026-09-09
status: complete
---

# Phase 33 Plan 10: Human Confirmation Outranks the Automated Reconciler Detach Summary

**The reconciler sweep now skips any companion row a human has confirmed (`_stale_attributions()` splits on `confirmed_by`), evaluates ownership before a classical night's outcome (WR-13), previews the detach count under `--dry-run` (WR-11), and every staff action plus both paired demo notebooks now report the new `detach_declined` counter truthfully.**

## Performance

- **Duration:** ~70 min
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Closed the CR-04 confirm/erase loop: a staff re-confirmation of a released `RUN:{pk}:{date}` event now survives every subsequent sweep, proven by a regression test that goes one step further than 33-08's existing detach test, and by a real executed notebook cell.
- Reordered `_reconcile_classical_nights()` so ownership (`_may_write()`) is decided before a night's skip/attribution outcome (WR-13): an attributed-and-contested night now reports `blocked`, not `skipped_nights`, and the foreign attribution is never detached.
- `--dry-run` now previews the exact detach count a real sweep would produce (WR-11), built on the same `_stale_attributions()` predicate the write path uses, and writes nothing.
- `reconcile_campaign_runs`'s summary and per-run lines, and all four staff actions (approve, resolve site, mark cancelled, mark weather failure) via a single shared `_message_reconcile_side_effects()` helper, now report both `detached` and the new `detach_declined` counter (WR-10/WR-12). `_resolve_site()`'s success message is re-keyed on `created`/`updated`/`skipped_nights` so "run added to the calendar" is never shown when nothing was added.
- Both paired demo notebooks re-executed against the post-fix code (dev database checksum unchanged, per 33-09's scratch-copy mechanism); the reconcile notebook gained a new cell demonstrating the confirm/erase survival end to end.
- The runbook's re-classification, counter-documentation and skip-rule sections corrected to describe both detach causes, the `detach_declined` counter, the `--dry-run` preview as a number, and that re-confirming a released entry is now permanent; plan 33-11's calendar pop-up section left byte-identical.

## Task Commits

Each task was committed atomically:

1. **Task 1: A human confirmation outranks the automated detach, and ownership is decided before the night's outcome** - `e0b0c87` (feat)
2. **Task 2: Operator surfaces tell the truth — sweep summary, per-run lines and the four staff actions** - `cdc5a29` (feat)
3. **Task 3: Paired docs — both notebooks re-executed, runbook wording corrected** - `f881ead` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `solsys_code/campaign_reconciler.py` - `_stale_attributions()` (new read-only predicate), `ReconcileResult.detach_declined`, `_detach_stale_family_events()` returns `(detached, declined)`, `reconcile_run()`'s dry-run branch previews via the same predicate, `_reconcile_classical_nights()` reordered per WR-13
- `solsys_code/campaign_views.py` - `_message_reconcile_side_effects()` (new shared staff-message helper), wired into approve/`_resolve_site()`/`_set_run_status()`; `_resolve_site()`'s success message re-keyed on `created`/`updated`/`skipped_nights`
- `solsys_code/management/commands/reconcile_campaign_runs.py` - accumulates and reports `detach_declined`; `would_detach` now a real number; per-run lines reworded (two detach causes, new declined line)
- `solsys_code/tests/test_campaign_reconciler.py` - confirm/erase regression, unconfirmed-still-reclaimable test, attributed-and-contested test, dry-run preview test, split the pre-existing unconditional-clear test into unconfirmed/confirmed branches
- `solsys_code/tests/test_reconcile_campaign_runs.py` - renamed/rewrote the dry-run counter test to assert a real previewed number; updated the detached-line wording assertion; added a declined-per-run-line test
- `solsys_code/tests/test_campaign_approval.py` - added an all-nights-already-covered test and a resolve-that-detaches-something warning test
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - new confirm/erase cell (executed output), "Two new counters" markdown updated to three counters, full notebook re-executed
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - re-executed against post-fix code (no source changes needed)
- `docs/runbooks/telescope_runs_calendar.rst` - re-classification, counter-documentation and skip-rule sections corrected; calendar pop-up section (33-11's) untouched

## Decisions Made

See `key-decisions` in frontmatter for the full rationale on each; summarized:
- The human-confirmation guard lives in `campaign_reconciler.py` only, never in the shared `campaign_utils.unlink_event_from_run()`/`UNLINK_CLEARED_FIELDS`, so Phase 28's human-initiated callers (undo view, admin clear) are unaffected.
- One `_stale_attributions()` predicate serves both the write path and the dry-run preview, closing the class of bug where a preview could drift from what a real sweep does.
- `_may_write()` is now evaluated before the attributed-night skip in the classical per-night loop, and a blocked night's url is added to `active_urls` from the blocked branch itself, so a contested night is neither written nor swept up by the detach convergence step.

## Observed `ReconcileResult` values across the four-sweep confirm/erase scenario

Captured directly from a fresh scratch run of the exact scenario (single-night `LCO_QUEUE` run, ground site, `RUN:{pk}:{date}` per-night branch):

```
1st reconcile (mints):                 ReconcileResult(created=1, updated=0, unchanged=0, blocked=0, skipped_nights=0, detached=0, detach_declined=0, skipped_reason=None)
2nd reconcile (facility event attributed, detaches):  ReconcileResult(created=0, updated=0, unchanged=0, blocked=0, skipped_nights=1, detached=1, detach_declined=0, skipped_reason=None)
3rd reconcile (staff re-confirms the detached RUN:-keyed event to the SAME run, declines):  ReconcileResult(created=0, updated=0, unchanged=0, blocked=0, skipped_nights=1, detached=0, detach_declined=1, skipped_reason=None)
4th reconcile (a further sweep changes nothing):      ReconcileResult(created=0, updated=0, unchanged=0, blocked=0, skipped_nights=1, detached=0, detach_declined=1, skipped_reason=None)

Final companion-row stamp: run_id=<run.pk>, confirmed_by_id=<staffer.pk>, confirmed_at=2026-08-02 09:00:00+00:00 (unchanged from what the staff member set)
```

The same scenario is proven end to end with real executed output in `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`'s new confirm/erase cell (using `classical_run`, pk=74 in that run): `detached=0, detach_declined=1` on both the re-confirmation sweep and a further sweep, plus a `--dry-run` preview reporting the identical live counters.

## Before/after text of the two corrected runbook instructions

### 1. Re-classification section (`telescope_runs_calendar.rst`, "What happens to an already-reconciled run's calendar events...")

**Before:**
> ...next reconcile (either a full `reconcile_campaign_runs` sweep, or the run's own next staff-action reconcile) automatically detaches the old family's events from the run rather than leaving them on the calendar looking like a live commitment forever. Detaching, not deleting: the old events stay on the calendar but return to the attribution page's worklist (`campaigns:attribution`, see "How do I attribute existing calendar events and observation records to a run?" above), where a staff member can **re-confirm or discard them**. The correction itself does not trigger this -- it happens on the *next* reconcile, same as any other calendar-visibility change only renders correctly once a sweep runs afterward. **That detach also clears the row's confirmation stamps** (who confirmed the attribution, and when) together with the link, so a detached row never goes on displaying a confirmation for an attribution that no longer exists.

**After:**
> ...next reconcile (either a full `reconcile_campaign_runs` sweep, or the run's own next staff-action reconcile) automatically detaches the old family's events from the run rather than leaving them on the calendar looking like a live commitment forever -- **unless a staff member has already confirmed one of those events to this run, in which case the sweep leaves it alone entirely** (see `detach_declined` below): a human decision always outranks an automated sweep. Detaching, not deleting: the old events stay on the calendar but return to the attribution page's worklist (`campaigns:attribution`, see "How do I attribute existing calendar events and observation records to a run?" above), where a staff member can **re-confirm them**. **Re-confirming a released entry is a permanent decision -- once a person has confirmed it, no later automated sweep releases it again.** The correction itself does not trigger the detach -- it happens on the *next* reconcile, same as any other calendar-visibility change only renders correctly once a sweep runs afterward.

### 2. Skip-rule section (`telescope_runs_calendar.rst`, "The rule applies whether or not the reconciler had already made its own entry for that night.")

**Before:**
> ...that earlier reconciler-created entry is released (never deleted) back into the attribution queue **for a human to re-confirm or discard, and its "confirmed by"/"confirmed at" record is cleared along with the release**. Clearing the other entry's attribution later brings the reconciler's own entry back, in place (same record, same url), on the next sweep.

**After:**
> ...that earlier reconciler-created entry is released (never deleted) back into the attribution queue, **where a staff member can re-confirm it. Re-confirming it is a permanent decision: once a person has confirmed it, no later automated sweep releases it again.** Clearing the other entry's attribution instead brings the reconciler's own entry back, in place (same record, same url), on the next sweep.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Split a pre-existing test that asserted unconditional confirmation-stamp clearing on detach**
- **Found during:** Task 1, running the full `test_campaign_reconciler` suite
- **Issue:** `test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched` (from plan 33-04) set `confirmed_by`/`confirmed_at` on a stale row and asserted the detach step cleared them — behavior this plan's UAT option B deliberately narrows (a confirmed row is now declined, not detached).
- **Fix:** Split into `test_detach_clears_audit_fields_for_an_unconfirmed_row_leaves_event_and_verification_flag_untouched` (confirmed_by left null, asserts the still-detached branch, unchanged from the original test's assertions) and `test_detach_declines_a_confirmed_row_leaving_run_and_audit_stamps_untouched` (a new test for the now-declined confirmed branch).
- **Files modified:** `solsys_code/tests/test_campaign_reconciler.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_reconciler` — both new tests pass
- **Commit:** `e0b0c87`

**2. [Rule 1 - Bug] Two pre-existing command-level tests asserted the retired `n/a` dry-run placeholder and the retired "confirmation stamp cleared" wording**
- **Found during:** Task 2, running `test_reconcile_campaign_runs`
- **Issue:** `test_dry_run_reports_skipped_nights_and_would_detach_na_and_writes_nothing` asserted `'would_detach: n/a (dry-run)'` in the output (now a real number per WR-11); `test_real_sweep_reports_detached_for_a_superseded_run_keyed_event` asserted the word `'confirmation'` in the detached-line output (removed from the reworded line, since a released row never carries one after Task 1).
- **Fix:** Renamed/rewrote the dry-run test to build a genuine stale-row scenario and assert `would_detach == 1`; updated the detached-line assertion to the new wording (`'released back into the attribution queue'`).
- **Files modified:** `solsys_code/tests/test_reconcile_campaign_runs.py`
- **Verification:** `python manage.py test solsys_code.tests.test_reconcile_campaign_runs` passes
- **Commit:** `cdc5a29`

**3. [Rule 1 - Bug] Removed the literal string `CalendarEventDismissal` from a docstring to satisfy the plan's own grep gate**
- **Found during:** Task 1, running the plan's acceptance-criteria checks
- **Issue:** `grep -c 'CalendarEventDismissal' solsys_code/campaign_reconciler.py` must output 0, but `_stale_attributions()`'s docstring named the class verbatim while explaining why no such row is written.
- **Fix:** Reworded to "no dismissal-row model instance" instead of naming the class.
- **Files modified:** `solsys_code/campaign_reconciler.py`
- **Verification:** `grep -c 'CalendarEventDismissal' solsys_code/campaign_reconciler.py` outputs 0
- **Commit:** `e0b0c87`

---

**Total deviations:** 3 auto-fixed (all Rule 1 — updating pre-existing tests/docstrings to match this plan's own intentionally narrowed/changed behavior, not scope creep).
**Impact on plan:** All three were necessary consequences of the plan's own must-haves (option B narrows detach behavior; the grep gate is the plan's own acceptance criterion). No unrelated files touched.

## Issues Encountered

None beyond the deviations above. The scratch-verification script used to capture real command output for the runbook's example summary lines (and for this SUMMARY's four-sweep table) ran against a full copy of the developer database via `FOMO_DATABASE_PATH`, mirroring 33-09's notebook mechanism — the developer database's checksum was confirmed unchanged (`md5sum src/fomo_db.sqlite3`) before and after every notebook re-execution and scratch run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Both 33-VERIFICATION.md gaps this plan targeted (Gap 1/CR-04 human-confirmed-attribution loss, Gap 2/WR-13 swallowed `blocked` signal) are closed, along with the three operator-facing findings riding on the same code path (WR-10, WR-11, WR-12). This was the last plan of the second gap-closure wave (wave 3); phase-level verification/UAT can now re-run against all three gap-closure plans (33-09, 33-11, 33-10) together.

## Self-Check: PASSED

All 9 key-files confirmed present on disk (`[ -f ]`). All 3 task commit hashes
(`e0b0c87`, `cdc5a29`, `f881ead`) confirmed present in `git log --oneline --all`.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-09*
