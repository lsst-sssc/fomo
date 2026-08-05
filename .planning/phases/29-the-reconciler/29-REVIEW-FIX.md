---
phase: 29-the-reconciler
fixed_at: 2026-08-05T21:10:00Z
review_path: .planning/phases/29-the-reconciler/29-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 29: Code Review Fix Report

**Fixed at:** 2026-08-05T21:10:00Z
**Source review:** .planning/phases/29-the-reconciler/29-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (1 critical, 2 warning, 1 info -- `fix_scope: all`)
- Fixed: 4
- Skipped: 0

Two of these findings (CR-01, WR-01) required a product decision beyond a mechanical fix.
The user was asked directly and chose: **CR-01 -- detach (`CalendarEventMeta.run = None`),
never delete, a run's stale-family calendar events on reclassification, returning them to
Phase 28's attribution queue**; **WR-01 -- cascade-delete a `CampaignRun`'s owned
`CalendarEvent` rows when the run itself is deleted**, via a `pre_delete` signal (not a
`CampaignRun.delete()` override alone), so both the admin's single-object delete and its
changelist bulk "Delete selected" action are covered. Both decisions are recorded in code
comments/docstrings at the fix site.

## Fixed Issues

### CR-01: Re-classifying a CampaignRun's branch leaves stale events behind, and can corrupt an adopted night's timing

**Files modified:** `solsys_code/campaign_reconciler.py`, `solsys_code/tests/test_campaign_reconciler.py`
**Commit:** `9db22f0`
**Applied fix:**
- Scoped `_adopted_event_for_night()`'s candidate query to blank-`url` events only
  (`CalendarEventMeta.objects.filter(run_id=run.pk, event__url='')`), matching its own
  docstring intent -- this closes the adopt-corruption vector where a run's own stale
  container event could be re-keyed into a per-night slot while its timing stayed at the
  original whole-window span.
- Added `_detach_stale_family_events(run, active_urls)`, a small bulk-`.update()` helper
  called from `reconcile_run()` after each branch computes which family (container vs.
  per-night) the run currently belongs to. It detaches (`CalendarEventMeta.run = None`) any
  `owned_events(run)` row whose `url` is not in the currently-active family's key set --
  a no-op when the run has not been reclassified since its last reconcile (preserves
  RECON-01 idempotency), and skipped entirely under `dry_run=True`.
- Added `TestReclassificationConvergence` (two tests): one proving a classical->queue
  reclassification detaches the old per-night events rather than leaving them dangling or
  duplicated; one proving a stale container event is never adopted into a classical night
  and ends up detached, not corrupted.

### WR-01: Deleting a CampaignRun permanently orphans its RUN:-namespaced calendar events

**Files modified:** `solsys_code/models.py`, `solsys_code/tests/test_campaign_reconciler.py`
**Commit:** `8dcdf58`
**Applied fix:** Added a `pre_delete` signal receiver on `CampaignRun`
(`_delete_owned_calendar_events_on_campaign_run_delete`) that deletes
`campaign_reconciler.owned_events(run)` before the run row itself is removed --
deleting the `CalendarEvent` rows cascades to their `CalendarEventMeta` companions via that
FK's own `on_delete=CASCADE`. A signal (not just a `CampaignRun.delete()` override) was
chosen specifically because the admin changelist's bulk "Delete selected" action calls
`QuerySet.delete()`, which sends `pre_delete`/`post_delete` per object but bypasses any
instance-level `delete()` override -- this covers both the admin's single-object delete
path and its bulk path, plus any direct ORM `.delete()` call. `campaign_reconciler` is
imported lazily inside the handler to avoid a circular import (it already imports
`CampaignRun`/`CalendarEventMeta` from `models.py`). Added
`TestCampaignRunDeletionCascadesCalendarEvents`, proving a deleted run's owned
`CalendarEvent` and companion `CalendarEventMeta` rows are both gone afterward.

### WR-02: No guard against window_end < window_start

**Files modified:** `solsys_code/campaign_reconciler.py` (code landed in commit `9db22f0`,
bundled with CR-01 since both touch `reconcile_run()`'s dispatch/guard logic in the same
file), `solsys_code/tests/test_campaign_reconciler.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `9db22f0` (code), `e695918` (test + runbook doc)
**Applied fix:** Added an explicit `_skip_reason()` branch --
`if run.window_end < run.window_start: return 'window_end before window_start'` -- evaluated
after the `TBD window` check and before `unresolved site`, so this data-integrity case is
surfaced through the same skip-reason vocabulary as every other unprojectable state instead
of `_reconcile_classical_nights()` silently iterating zero nights with no reported reason.
Added `TestWindowEndBeforeWindowStart`, asserting the run is skipped with the new reason and
`CalendarEvent.objects.count()` is unchanged. Also updated the runbook's itemized skip-reason
list ("How do I get every campaign run onto the calendar?") to include the new reason, since
that list is otherwise now stale.

**Note on commit boundaries:** the WR-02 code change (the `_skip_reason()` branch itself)
was written and committed together with CR-01's changes in `9db22f0` before the per-finding
test-file split was applied -- both changes are in `campaign_reconciler.py`'s
`_skip_reason()`/`reconcile_run()` area, and separating that single small `if` branch into
its own commit after the fact would have required an unwanted history rewrite. The test
coverage and doc update for WR-02 are committed separately in `e695918`, which references
`9db22f0` as the commit carrying the actual code fix.

### IN-01: Operator docs describe the exact workflow that triggers CR-01, without warning about it

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`,
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`
**Commit:** `21a97da`
**Applied fix:** Per the user's direction (CR-01 is now resolved, not merely risky), replaced
the originally-proposed "add a warning" fix with a description of the actual, now-safe
behavior. Added a note to the "Can I correct a run's source?" runbook section explaining that
correcting a run's classifying fields (`source`/`telescope_class`/`site`) automatically
detaches its old-family calendar events back into the attribution page's worklist on the
*next* reconcile (not immediately on save) -- detach, not delete, so a staff member can
re-confirm or discard them. Appended the equivalent explanation to the demo notebook's "Why
`source` is set explicitly here, and what it means for real data" markdown cell. This was a
markdown-only change (no code cell added or altered, no cell output affected), so no
`jupyter nbconvert --execute` re-run was needed; the notebook's JSON validity and the
`docs/` Sphinx build were both verified after the edit.

## Skipped Issues

None -- all four in-scope findings were fixed.

## Verification

- `python manage.py test` across all `solsys_code` test modules except `test_views` and
  `test_ephem_utils` (excluded per project memory: native ASSIST segfault, unrelated to this
  fix) -- **817 tests, OK** (includes the 4 new regression tests: 2 for CR-01, 1 for WR-01,
  1 for WR-02).
- `python -m pytest` (the separate `tests/`/`src/`/`docs/` suite) -- **1 passed, OK**.
- `ruff check .` -- 1 pre-existing `D103` finding in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` (unrelated file,
  confirmed present before this fix run).
- `ruff format --check .` -- 3 pre-existing reformat candidates
  (`.planning/quick/260619-f7u-.../verify_nb.py`, `verify_project.py`, `src/fomo/settings.py`;
  none touched by this fix, confirmed present before this fix run).
- `sphinx-build -b html docs ...` -- succeeds; no new warnings/errors attributable to
  `docs/runbooks/telescope_runs_calendar.rst`.

---

_Fixed: 2026-08-05T21:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
