---
phase: 35-allocation-layer-classical-cutover
fixed_at: 2026-09-13T19:54:37Z
review_path: .planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md
iteration: 1
findings_in_scope: 17
fixed: 15
skipped: 2
status: partial
---

# Phase 35: Code Review Fix Report

**Fixed at:** 2026-09-13T19:54:37Z
**Source review:** .planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 17 (CR-01..CR-06, WR-01..WR-11)
- Fixed: 15
- Skipped: 2 (WR-07, WR-11 — both require a design decision the review did not resolve)

All fixes were made in an isolated git worktree/branch
(`gsd-reviewfix/35-4080497`) and fast-forward-merged onto
`issue37-telescope-runs-calendar`. Every commit passed `pre-commit run
ruff`/`ruff-format` and the project's own pre-commit test-suite gate. After
all 15 commits, a full cross-module regression pass covering every affected
test module (`test_campaign_reconciler`, `test_allocation_projector`,
`test_allocation_projector_signals`, `test_campaign_approval`,
`test_campaign_models`, `test_cutover_classical_allocations`,
`test_load_telescope_runs`, `test_observation_projector`,
`test_observation_projector_signals`, `test_project_observation_calendar`,
`test_reconcile_campaign_runs`, `test_telescope_runs`,
`test_write_and_reconcile`) ran **487 tests, all passing** (0 failures, 0
errors). `test_views.TestEphemeris` was excluded per the project's known
ASSIST-segfault gotcha (unrelated to this phase).

## Fixed Issues

### CR-01: Signal receivers call `project_allocation()` directly, bypassing `reconcile_run()`'s dispatch and approval gates

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/campaign_reconciler.py`, `solsys_code/observation_projector.py`
**Commit:** `2a34a04`
**Applied fix:** Extracted `campaign_reconciler.dispatches_per_night()` (the same predicate `reconcile_run()` uses to choose its branch) and added `allocation_projector.reproject_allocation_if_dispatched()`, which re-applies `_skip_reason()` and `dispatches_per_night()` before calling `project_allocation()`. All three signal receivers (two on `CampaignRunObservation`, one on `ObservationRecord`) now call the guarded entry point instead of `project_allocation()` directly. `reconcile_run()`'s own dispatch chain was refactored to a single `if dispatches_per_night(run): ... else: ...` so the decision has exactly one owner.

### CR-02: Re-classifying a run leaves its `ALLOC:` nights on the calendar forever

**Files modified:** `solsys_code/campaign_reconciler.py`, `solsys_code/tests/test_campaign_reconciler.py`
**Commit:** `b082f6f`
**Applied fix:** Added `_stale_allocation_events()` (the `ALLOC:` mirror of the existing date-bearing `RUN:{pk}:{date}` convergence) and wired it into `_detach_stale_family_events()` (the real sweep) and `reconcile_run()`'s dry-run preview branch, reusing the existing `legacy_deleted` counter so the runbook's already-documented operator promise ("the next reconcile ... deletes the run's leftover per-night events ... and replaces them with a single whole-window entry") is now literally true for both origin families, not just the retired `RUN:` one. Updated the pre-existing test that asserted the *bug* as expected behavior; added idempotency and dry-run coverage.

### CR-03: The retire path deletes a legacy `RUN:{pk}:{night}` event with no ownership and no `confirmed_by` guard

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `9a86d4b`
**Applied fix:** Applied the same `_may_write()` ownership guard the takeover branch already uses, plus a `confirmed_by` check (via `_clearable_and_declined()`, reused from `campaign_reconciler`) so a human-confirmed or foreign-attributed legacy event is left alone and counted under `blocked` instead of deleted. Added regression tests for both the human-confirmed and foreign-attribution cases.

### CR-04: Final convergence deletes stale `ALLOC:` events attributed to a different run

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `a46aed2`
**Applied fix:** Replaced the `allocation_events(run)` (namespace-identity) stale query with `writable_allocation_events(run)` plus `_clearable_and_declined()`, reporting foreign-attributed and human-confirmed counts under `totals['blocked']` instead of silently deleting or silently dropping them. Added regression tests for both a foreign-run-confirmed night and a same-run-confirmed night surviving a window shrink.

### CR-05: The cascade guard misses queryset deletes — admin bulk delete orphans events and leaves a dangling FK

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector_signals.py`
**Commit:** `59226c9`
**Applied fix:** Changed the `isinstance(kwargs.get('origin'), CampaignRun)` check to also cover the model-class form (`getattr(origin, 'model', type(origin))`), which is what Django sets `origin` to for a `QuerySet.delete()` call (the admin's "Delete selected" bulk action). Added a regression test exercising `CampaignRun.objects.filter(pk=...).delete()` directly (the admin path), proving no re-projection occurs and no orphaned `ALLOC:`/`CalendarEventMeta` rows remain.

### CR-06: Sub-night window fields produce an inverted event span for Australia/Sydney (FTS)

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/models.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `6e91c3d`
**Applied fix:** Added `_site_runs_behind_utc()` (a cheap `zoneinfo` offset lookup, never an astropy `sun_event()` call, so it stays valid on `_span_needs_remint()`'s astropy-free D-13 update path) and threaded its answer through `_time_of_day_to_datetime()`, `night_bounds()` and `_span_needs_remint()`. Also added a guard in `night_bounds()`: refuses to return `start >= end`, raising `ValueError` (propagating the same way `sun_event()`'s own `ValueError` already does) instead of silently writing an inverted event. Added `TestSubNightWindowSiteDirection` covering the exact reproduced Sydney case, the mirror evening-side case, unchanged Chile behavior, and the new guard. **Logic-sensitive fix — the site-direction rule (west-of-UTC vs. east-of-UTC) was independently verified against the projector's own `sun_event()`-computed full night for the exact Sydney date/time the review reproduced (2026-08-01), matching the review's numbers (sunset ≈07:33Z, sunrise ≈20:46Z, both same UTC date); still worth a human spot-check on the astronomy semantics given only two real-world sites (Chile, Sydney) exist in the fixture set.**

### WR-01 / WR-02: Linked-run re-project must run for every facility, and one failing run must not skip the rest

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/tests/test_observation_projector_signals.py`
**Commit:** `9909e82`
**Applied fix:** Moved the D-11 linked-run re-projection block above `receiver_on_record_save()`'s `if instance.facility not in PROJECTED_FACILITIES: return` guard (it never depends on `project_record()`), so a Gemini/ESO record's linked run re-projects too — `retired_nights()` applies no facility filter, so the trigger and projector no longer disagree about which records matter. Moved the `try`/`except` inside the per-link loop, naming the failing run's pk in the log, so one bad linked run no longer aborts re-projection for every later one. Added regression tests for a Gemini-facility record's linked-run re-project, and for one failing run not blocking a second.

### WR-03: `--dry-run` computes and discards two `sun_event()` calls per new night

**Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/tests/test_allocation_projector.py`
**Commit:** `c97b399`
**Applied fix:** Skip `_mint_fields()` entirely on the create path when `dry_run` is set, since `preview_calendar_event_action(None, fields)` always returns `'created'` without reading `fields`. Added a regression test proving a dry-run reconcile over a multi-night, all-new-night window never calls `sun_event()`.

### WR-04: `reconcile_campaign_runs --dry-run` reports deletions in the past tense

**Files modified:** `solsys_code/management/commands/reconcile_campaign_runs.py`, `solsys_code/tests/test_reconcile_campaign_runs.py`
**Commit:** `c8098a3`
**Applied fix:** Routed the verb for all three new per-run messages (`retired`/`rekeyed`/`legacy_deleted`) through the existing `dry_run` flag, matching the summary line's own `would_*` vocabulary. Added 3 wording-specific regression tests (one per message).

### WR-05: The "retired" per-run message conflates three unrelated causes

**Files modified:** `solsys_code/management/commands/reconcile_campaign_runs.py`, `solsys_code/tests/test_reconcile_campaign_runs.py`
**Commit:** `f46d6b1`
**Applied fix:** Dropped the causal clause ("now covered by a real observation") in favor of a neutral message naming all three causes `ReconcileResult.retired` is actually incremented from (observation handoff, sub-night re-mint, window-shrink convergence) — the simpler of the review's two suggested fixes, since splitting the counter into three fields would have been a larger `ReconcileResult`/`ReconcileCampaignRuns` shape change touching several other call sites.

### WR-06: `cutover_classical_allocations` has no transaction boundary and commits before raising

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`, `solsys_code/tests/test_cutover_classical_allocations.py`
**Commit:** `16f7077`
**Applied fix:** Wrapped each group's run write plus its full per-event re-key loop in `transaction.atomic()`. Each per-event re-key additionally gets its own **nested** savepoint, so a per-event exception (still caught and reported individually, preserving D-18's existing "an event it cannot explain is left byte-identical and reported" contract) does not poison the group's outer transaction on backends that require a rollback after any failed statement inside an atomic block (PostgreSQL) — a correctness concern the review's own illustrative fix snippet did not address, since SQLite's autocommit-per-statement behavior masks it during dev testing. Added regression tests for both group-level rollback (an unexpected failure during run creation leaves nothing from the group persisted) and per-event isolation (one event's exception leaves the run and every other event in the group converted).

### WR-08: The cutover never checks the derived night lies inside the run's own window

**Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`, `solsys_code/tests/test_cutover_classical_allocations.py`
**Commit:** `75d6e89`
**Applied fix:** Added a `run.window_start <= night <= run.window_end` check before re-keying each event, raising (and reporting via the existing per-event `_OTHER` catch) when the event's independently-derived night falls outside the run's own window, rather than re-keying it to a url the very next sweep would classify as stale and delete. Added a regression test with an out-of-window event sharing a group with three in-window events.

### WR-09: A `KeyError` from `_CLASSICAL_RUN_STATUS` escapes the cutover's per-group handler

**Files modified:** `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/management/commands/cutover_classical_allocations.py`, `solsys_code/tests/test_load_telescope_runs.py`, `solsys_code/tests/test_cutover_classical_allocations.py`
**Commit:** `b467a34`
**Applied fix:** Added a structural `assert set(_CLASSICAL_RUN_STATUS) == KNOWN_STATUSES` at import time in `load_telescope_runs.py` (protects both call sites, since `cutover_classical_allocations.py` imports the same dict object), added `KeyError` to `load_telescope_runs.py`'s existing per-line `except` tuple as defense-in-depth, and wrapped the lookup in `cutover_classical_allocations.py` in its own per-group `try`/`except KeyError` that reports a named reason and continues. Added regression tests for both commands using `patch.dict`/`clear=True` to simulate the invariant breaking.

### WR-10: A legacy `RUN:{pk}:{date}` event with no companion row is never cleaned up

**Files modified:** `solsys_code/campaign_reconciler.py`, `solsys_code/tests/test_campaign_reconciler.py`
**Commit:** `2d7c297`
**Applied fix:** `_stale_dated_events()` now unions the "clearable via `_clearable_and_declined()`" set with "no companion row at all" (`stale_dated.filter(telescope_label_meta__isnull=True)`), since a meta-less event has no attribution to preserve — it is exactly D-16's stated "third outcome" that shouldn't exist. Added a regression test with a hand-created legacy event carrying no `CalendarEventMeta` row.

## Skipped Issues

### WR-07: The cutover creates an empty `CampaignRun` for a group whose events are all foreign-attributed

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:252-286`
**Reason:** Per the task's binding instructions, this finding requires a design decision the review did not resolve, so it was skipped rather than guessed at.

The review's suggested fix (`if not writable_events: continue`) looks simple on its face, but it silently changes behavior for the case where a `CampaignRun` **already exists** for this group's `source_identifier` key from a prior invocation, and every one of the group's events has since become foreign-attributed (e.g. a staff member manually re-attributed them via the attribution queue after a previous cutover run). `insert_or_create_campaign_run()` is a find-or-update helper: skipping the run write entirely would also skip re-syncing that EXISTING run's field values (window, telescope/instrument, status, etc.) on a re-run, contradicting the command's own "safe to re-run" and idempotent-update semantics for a run row that legitimately still needs its fields refreshed from the schedule line, even though none of its per-night events are writable anymore. The review's fix text does not distinguish "skip creating a brand-new empty run" from "skip updating a pre-existing run's fields" — those are two different policies, and choosing between them (or finding a third option, e.g. still upserting run fields but never creating a NEW run when `writable_events` is empty) is a product decision about idempotent-update behavior, not a mechanical fix.

### WR-11: Two legacy events for one night collide onto a single `ALLOC:` key, and `CalendarEvent.url` is not unique

**File:** `solsys_code/management/commands/cutover_classical_allocations.py:290-303`, `solsys_code/allocation_projector.py:451`
**Reason:** Per the task's binding instructions (explicitly named alongside WR-07), this finding's key-collision semantics require a design decision, so it was skipped rather than guessed at.

The review's own fix suggestion detects the *first* collision within one cutover run's per-group loop (`claimed_nights` set), but the deeper problem it names — `CalendarEvent.url` has no unique constraint, so `.filter(url=...).first()` throughout the reconciler and projector is inherently unable to detect a PRE-EXISTING duplicate written by an earlier cutover run, a hand-edit, or any other writer — is explicitly called out by the review itself as "a `tom_calendar` change and out of this phase's scope, but worth recording." Implementing only the in-run detection without addressing the underlying non-unique `url` field would give a false sense of safety (this run's own duplicates are caught, but a duplicate already sitting in the database from before this fix, or written by a different code path entirely, is not), and deciding the right remedy (a `UniqueConstraint` migration on a third-party app's model, a defensive `.filter(url=...).order_by('pk').first()` convention everywhere, or something else) is a design decision spanning a different app's model, not a self-contained fix to this command.

## Verification

All 15 fixes were verified in the isolated `gsd-reviewfix/35-4080497` worktree:

- **Tier 1 (always):** every modified file was re-read after editing to confirm the fix text was present and surrounding code intact.
- **Tier 2 (preferred):** `python -c "import ast; ast.parse(...)"` syntax-checked every modified `.py` file before each commit; `pre-commit run ruff --files ...` and `pre-commit run ruff-format --files ...` were run (and passed clean) on every commit's changed files.
- **Test verification:** after each fix, the directly-relevant Django test module(s) were run and confirmed green before committing (per-finding test runs are recorded in each commit message). After all 15 commits, a full cross-module regression pass (13 test modules, 487 tests) ran clean with 0 failures/errors. `pre-commit`'s own bundled test-suite gate (which runs the project's full `manage.py test` invocation, excluding the known-segfaulting `test_views.TestEphemeris`) also passed on every commit.
- **Environment note:** all test runs and gate checks above ran inside the isolated worktree (`.claude/worktrees/rf-35-...`), not the main checkout — the worktree shares the repository's installed environment (editable install, dependencies) but is a separate git working tree on its own temporary branch, fast-forwarded onto `issue37-telescope-runs-calendar` at cleanup.

No paired-docs update was required: CR-02's fix reuses the existing `legacy_deleted` counter specifically so the runbook's already-shipped prose (`docs/runbooks/telescope_runs_calendar.rst`, the "Relabelling a per-night run's source" section) becomes literally true rather than needing new wording — the runbook was checked and contains no other references to any of the fixed behaviors that would now be stale. No notebook re-execution is required: none of these fixes change `campaign_reconciler.py`'s or `cutover_classical_allocations.py`'s *documented, demonstrated* behavior in `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` in a way that would make an existing cell's output inaccurate (the notebook does not exercise the specific bug scenarios CR-01 through WR-10 reproduce — re-classification-while-per-night-dispatched, cross-run foreign attribution, the Sydney sub-night-window bug, admin bulk delete, or cutover transaction/window-mismatch edge cases).

## Human Verification Recommended

- **CR-06** (marked above): the site-UTC-offset-direction rule is a genuine astronomy/timezone logic change. It was verified against the review's own reproduced numbers and a dedicated regression test class, but only two real sites exist in the fixture set (Chile, west of UTC; Sydney, east of UTC) — a human should confirm the rule generalizes correctly if a third site with an unusual offset (e.g. very close to UTC, or a half-hour offset) is ever added to `telescope_runs.SITES`.
- **WR-06**: the nested-savepoint transaction design was reasoned through for PostgreSQL correctness (the project's stated production target per CLAUDE.md) but only tested against the project's SQLite dev/test database, which does not exhibit the "poisoned transaction" failure mode the nesting exists to prevent — a human should confirm on a PostgreSQL-backed staging environment before this command is next run in production, if that has not already happened via the phase's own verification loop.

---

_Fixed: 2026-09-13T19:54:37Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
