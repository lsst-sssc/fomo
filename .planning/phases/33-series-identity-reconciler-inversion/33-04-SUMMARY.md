---
phase: 33-series-identity-reconciler-inversion
plan: 04
subsystem: campaign-coordination
tags: [attribution, calendar-events, campaign-run, django-orm, audit-trail]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion (plan 01)
    provides: "the annotator-inversion skip-the-night rule and D-17 attribution-not-ownership wording this plan's helper is written into"
  - phase: 33-series-identity-reconciler-inversion (plan 03)
    provides: "CalendarEventMeta.observation_record/observation_group fields, used as a fixture value by this plan's D-15 test"
provides:
  - "unlink_event_from_run() in campaign_utils.py -- the single writer that clears an event's attribution, taking confirmed_by/confirmed_at with it"
  - "All three existing clear-the-link writers (attribution-undo view, reconciler detach step, admin standalone clear branch) routed through that one helper"
  - "The reconciler's detach step now clears confirmed_by/confirmed_at with the link (D-16) -- a deliberate behaviour change, previously an audit-stamp leak"
  - "D-15 proven, not assumed: an observation-backed, unattributed event still appears in the event attribution queue"
affects: [34-the-observation-projector-and-trigger, 37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 6877
  tasks: 3
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One shared clear-the-link helper, mirroring the existing one shared link-the-event helper (adopt_event_into_run): unlink_event_from_run() lives beside its inverse in campaign_utils.py, takes an event (instance/pk/queryset) and a run (instance/pk), resolves the run to a pk and bails out first on a falsy pk before building any queryset, and clears run/confirmed_by/confirmed_at together in one conditional bulk .update()."
    - "Local (function-body) import to break a circular import: campaign_utils.py imports reconcile_run/ReconcileResult from campaign_reconciler.py at module scope, so campaign_reconciler.py's own need for campaign_utils.unlink_event_from_run() is satisfied with an import inside _detach_stale_family_events() rather than at module top level."

key-files:
  created: []
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_reconciler.py
    - solsys_code/admin.py
    - solsys_code/tests/test_campaign_attribution_views.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_admin.py
    - solsys_code/tests/test_campaign_attribution.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "unlink_event_from_run(events, run) accepts either a single CalendarEvent instance, a bare event pk (int), or a queryset/iterable for the event side, and a CampaignRun instance or its pk for the run side -- one signature serving the view's single-pair call, the reconciler's bulk queryset call, and the admin's per-row call alike, dispatched by isinstance() rather than three separate functions."
  - "The run-side null guard (getattr(run, 'pk', run), then `if not run_pk: return 0`) is the first statement in the function body, before any queryset is built -- proven by a source-level verify script, not just by tests, per 33-REVIEWS.md Agreed Concern 2 / threat T-33-21."
  - "The reconciler's call to the helper uses a local (function-body) import, not a module-level one, because campaign_utils.py already imports reconcile_run/ReconcileResult from campaign_reconciler.py at its own top level -- a module-level import in the other direction would deadlock on load order regardless of which module Python imports first."
  - "admin.py's save_model branch 2 keeps its existing in-memory obj.confirmed_by/obj.confirmed_at nulling rather than also calling the shared helper -- the plan explicitly allowed either route. Calling the helper would add a second database write (a bulk .update()) that super().save_model()'s subsequent obj.save() would then immediately overwrite anyway with the in-memory obj's own (already-nulled) values, so the extra write buys nothing; the comment above the two assignments now names the helper as the shared definition of what a clear means, so a future reader does not \"simplify\" this branch by deleting them."
  - "[Rule 2 - CLAUDE.md paired-docs] The reconciler detach step's new audit-stamp-clearing behaviour is a real behaviour change to campaign_reconciler.py, which CLAUDE.md's paired-docs convention maps to reconcile_campaign_runs_demo.ipynb and docs/runbooks/telescope_runs_calendar.rst. Neither was in the plan's files_modified. Rather than adding a new executed code demo (the notebook has no cell that reclassifies a run today, only a markdown paragraph describing the detach), both existing explanatory passages were extended with one sentence stating the new audit-clearing behaviour -- a markdown-only edit needing no notebook re-execution, proportionate to the size of the actual change."

patterns-established:
  - "Mirror-pair helpers in campaign_utils.py: adopt_event_into_run() (link) and unlink_event_from_run() (clear) sit beside each other, share the same never-touch-a-CalendarEvent-field/never-delete-anything contract, and are the only two writers of CalendarEventMeta.run in the codebase."

requirements-completed: []
# ANNOT-01 is also declared by sibling plans 33-01 (already summarized) and 33-05 (not yet
# summarized) in this phase's shared-ID gate (execute-plan.md update_requirements step);
# `requirements.ready-ids` returned 0/1 ready for this plan's run, so REQUIREMENTS.md is left
# untouched here -- ANNOT-01 flips to Complete once 33-05 also finishes.

coverage:
  - id: D1
    description: "unlink_event_from_run() is the single writer that clears an event's attribution -- run, confirmed_by and confirmed_at together -- and returns the number of companion rows it changed"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_clears_the_matching_run_and_returns_one"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_clears_every_matching_row_in_a_multi_row_filter"
        status: pass
    human_judgment: false
  - id: D2
    description: "The helper refuses to clear a row attributed to a different run, a row with no companion row, an already-unlinked row, or a null/unresolvable run -- and a null run never wipes audit stamps on an already-unlinked row (T-33-21)"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_wrong_run_returns_zero_and_changes_nothing"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_null_run_returns_zero_and_never_touches_an_already_unlinked_rows_audit_fields"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_campaign_run_shaped_argument_with_no_pk_behaves_like_none"
        status: pass
      - kind: other
        ref: "python source-verify: the early `return 0` precedes any `CalendarEventMeta.objects` reference in the function body"
        status: pass
    human_judgment: false
  - id: D3
    description: "The helper never touches is_verified, never deletes a CalendarEvent or CalendarEventMeta row, and never writes any CalendarEvent field"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_verification_flag_is_never_touched"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun::test_neither_event_nor_companion_row_is_deleted"
        status: pass
      - kind: other
        ref: "grep-based source-verify: the function body contains no 'is_verified', '.delete(', 'observation_record' or 'observation_group' substring"
        status: pass
    human_judgment: false
  - id: D4
    description: "The attribution-undo view's event branch is routed through the helper, preserving the conditional event+run filter and the changed_count-gated dismissal write"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUndoConfirmationOrdering"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestConcurrencyAndTampering"
        status: pass
    human_judgment: false
  - id: D5
    description: "The reconciler's detach step is routed through the helper: it now clears confirmed_by/confirmed_at alongside run (a deliberate D-16 behaviour change), leaves is_verified and every CalendarEvent field untouched, deletes nothing, and never clears a foreign attribution in the same RUN: namespace"
    requirement: ANNOT-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReclassificationConvergence::test_detach_clears_audit_fields_leaves_event_and_verification_flag_untouched"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestReclassificationConvergence::test_detach_never_clears_a_foreign_attribution_in_the_same_namespace"
        status: pass
    human_judgment: false
  - id: D6
    description: "The admin standalone change form's clear branch still leaves the STORED row (re-fetched from the database, not the in-memory instance) with confirmed_by/confirmed_at both None, is_verified unchanged, and the CalendarEvent untouched"
    requirement: ANNOT-01
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_admin.py#CalendarEventMetaStandaloneAdminAuditStampTests::test_clearing_the_run_clears_the_audit_fields"
        status: pass
    human_judgment: false
  - id: D7
    description: "D-15 holds: an event whose companion row has observation_record set but run unset is still returned by orphan_calendar_events(), exactly like any other unattributed event"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution.py#TestOrphanQuerysets::test_event_with_observation_record_but_no_run_is_still_included"
        status: pass
    human_judgment: false

duration: 28min
completed: 2026-09-04
status: complete
---

# Phase 33 Plan 4: One Shared Unlink Helper Summary

**Added `unlink_event_from_run()` — the single writer that clears an event's campaign attribution and takes its `confirmed_by`/`confirmed_at` audit stamps with it — and routed the attribution-undo view, the reconciler's detach step, and the admin's standalone clear branch through it, closing the reconciler's pre-existing stale-confirmation-stamp leak.**

## Performance

- **Duration:** 28 min
- **Started:** 2026-09-04T16:32:00Z
- **Completed:** 2026-09-04T17:00:06Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments

- `unlink_event_from_run()` in `campaign_utils.py`, mirroring `adopt_event_into_run()`'s contract: resolves its `run` argument to a pk and returns `0` immediately — before building any queryset — when that pk is falsy, so a `None` run (or an unsaved `CampaignRun`) can never build a `run_id=None` filter that would wipe audit stamps on rows that were never attributed to anything (T-33-21, 33-REVIEWS.md Agreed Concern 2). Otherwise clears `run`, `confirmed_by` and `confirmed_at` together in one conditional bulk `.update()` and returns the number of rows changed.
- All three existing clear-the-link writers now route through it: `campaign_views._undo_confirmation()`'s event branch (behaviour-preserving — the conditional event+run filter and the `changed_count`-gated dismissal write are unchanged); `campaign_reconciler._detach_stale_family_events()` (a deliberate D-16 behaviour change — it now also clears the audit stamps, closing a leak where a detached row could go on displaying "confirmed by X at T" for an attribution that no longer existed); and `admin.CalendarEventMetaAdmin.save_model()` branch 2, which keeps its existing in-memory `confirmed_by`/`confirmed_at` nulling (documented as the in-memory half of the same clear the helper performs, since `super().save_model()`'s subsequent `obj.save()` would otherwise re-persist stale values over a helper-only clear — 33-REVIEWS.md Agreed Concern 3).
- A circular import between `campaign_utils.py` (which already imports `reconcile_run`/`ReconcileResult` from `campaign_reconciler.py` at module scope) and `campaign_reconciler.py`'s new need for the helper is resolved with a local (function-body) import inside `_detach_stale_family_events()`.
- New regression coverage proves ROADMAP criterion 4 end to end: the helper's own nine behaviors (Task 1); the undo view's and admin's existing tests re-verified unchanged (Task 2); and two new reconciler tests plus one admin extension proving the detach step's audit-clearing, its untouched-`CalendarEvent`/untouched-`is_verified` guarantee, its foreign-attribution protection, and the admin's clear branch re-fetched-from-database persistence (Task 3). A new D-15 test proves an observation-backed, unattributed event still appears in the event attribution queue.
- [Deviation, Rule 2] Extended the paired `reconcile_campaign_runs_demo.ipynb` explanation and the `telescope_runs_calendar.rst` runbook's matching "correct a run's source" section with one sentence each, noting the detach step's new audit-stamp-clearing behaviour (CLAUDE.md paired-docs convention; neither was in the plan's `files_modified`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Write the one shared unlink helper** - `6fedad8` (test)
2. **Task 2: Route all three existing clear-the-link call sites through the helper** - `789e76b` (feat)
3. **Task 3: Prove criterion 4 — unlink removes only the decoration, and the attribution queue is unchanged** - `7e1d7d8` (test)

Plus one deviation commit (see above): `b48e49d` (docs) — paired-docs update, not a numbered plan task.

**Plan metadata:** (this commit)

## Files Created/Modified

- `solsys_code/campaign_utils.py` - `unlink_event_from_run()`, the single writer that clears an attribution
- `solsys_code/campaign_views.py` - `_undo_confirmation()`'s event branch routed through the helper
- `solsys_code/campaign_reconciler.py` - `_detach_stale_family_events()` routed through the helper (D-16 behaviour change), local import to break the circular dependency
- `solsys_code/admin.py` - `save_model()` branch 2's comment names the helper as the shared definition of what a clear means; in-memory nulling unchanged
- `solsys_code/tests/test_campaign_attribution_views.py` - `TestUnlinkEventFromRun` (9 tests)
- `solsys_code/tests/test_campaign_reconciler.py` - two new `TestReclassificationConvergence` methods (audit-clearing, foreign-attribution protection)
- `solsys_code/tests/test_admin.py` - `test_clearing_the_run_clears_the_audit_fields` extended (re-fetch, `is_verified`, `CalendarEvent`/count invariance)
- `solsys_code/tests/test_campaign_attribution.py` - new D-15 test in `TestOrphanQuerysets`
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - one sentence added to the family-reclassification explanation
- `docs/runbooks/telescope_runs_calendar.rst` - one sentence added to the matching "correct a run's source" section

## Decisions Made

- `unlink_event_from_run(events, run)` dispatches on `isinstance()` to accept a single `CalendarEvent`, a bare event pk, or a queryset/iterable for `events` — one signature for the view's single-pair call, the reconciler's bulk queryset call, and any future per-row admin call, rather than three separate functions.
- The null-run guard is the function's first statement, verified by a source-level check (not just tests) that it precedes any `CalendarEventMeta.objects` reference in the body.
- The reconciler's helper call uses a local import rather than a module-level one, to avoid a circular import with `campaign_utils.py`'s existing top-level import of `campaign_reconciler.reconcile_run`/`ReconcileResult`.
- Admin `save_model()` branch 2 keeps its in-memory-only nulling rather than also calling the helper — calling it would add a redundant database write that `obj.save()` would immediately overwrite anyway with the same (already-nulled) in-memory values.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Local import to break a circular dependency between `campaign_reconciler.py` and `campaign_utils.py`**
- **Found during:** Task 2 (`_detach_stale_family_events()`)
- **Issue:** `campaign_utils.py` already imports `reconcile_run`/`ReconcileResult` from `campaign_reconciler.py` at module scope. A module-level `from solsys_code.campaign_utils import unlink_event_from_run` added to `campaign_reconciler.py` would deadlock on whichever module Python loads first (a genuine circular import, not a style choice).
- **Fix:** Import `unlink_event_from_run` inside `_detach_stale_family_events()`'s function body instead of at module scope.
- **Files modified:** `solsys_code/campaign_reconciler.py`
- **Verification:** Full test suite (980 + 40 tests) imports and runs both modules without `ImportError`.
- **Committed in:** `789e76b` (Task 2 commit)

**2. [Rule 2 - Missing Critical] CLAUDE.md paired-docs update for the reconciler's behaviour change**
- **Found during:** Task 2/3 (after `_detach_stale_family_events()`'s audit-clearing behaviour change landed)
- **Issue:** CLAUDE.md's paired-docs convention maps `campaign_reconciler.py` to `reconcile_campaign_runs_demo.ipynb` and, by directory scope, any `docs/runbooks/` page whose documented behaviour the change affects (`telescope_runs_calendar.rst`, which already documents the detach step in its "correct a run's source" section). Neither artifact was in the plan's `files_modified`, and the detach step's new confirmed_by/confirmed_at clearing is a genuine behaviour change, not a refactor.
- **Fix:** Added one sentence to the notebook's existing family-reclassification markdown explanation and one sentence to the runbook's matching section, both stating the new audit-clearing behaviour. No code demo cell was added (the notebook had no cell exercising the detach path at all, only markdown prose) and no notebook re-execution was needed since only a markdown cell (no code, no outputs) changed.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, `docs/runbooks/telescope_runs_calendar.rst`
- **Verification:** `nbformat.validate()` on the notebook; `pre-commit run ruff-format`/Sphinx-build hooks both passed.
- **Committed in:** `b48e49d`

---

**Total deviations:** 2 auto-fixed (1 blocking/circular-import, 1 missing-critical/paired-docs).
**Impact on plan:** Both fixes were necessary — the circular import would have broken every import of either module, and the paired-docs update keeps the operator-facing documentation from silently going stale against a real, deliberate behaviour change. No scope creep beyond what CLAUDE.md and the plan's own correctness requirements demand.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ROADMAP criterion 4 holds at all three call sites: clearing `CalendarEventMeta.run` removes only the decoration — the event untouched, nothing deleted — and now always takes the confirmation audit stamps with it.
- `unlink_event_from_run()` is the mirror-image twin of `adopt_event_into_run()` and the only other writer of `CalendarEventMeta.run` in the codebase, which Phase 34's observation projector and Phase 35's allocation layer can both reuse rather than re-deriving the clear semantics.
- D-15 holds, proven rather than assumed: the event attribution queue keys off the attribution link only, never the observation link, so Phase 34's projector-written `observation_record` links will not accidentally remove events from Phase 28's queue.
- No blockers for 33-05 (the only remaining plan in this phase, which also declares ANNOT-01/ANNOT-02).

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-04*

## Self-Check: PASSED

All 10 modified files verified present on disk. All 4 commit hashes (`6fedad8`, `789e76b`, `7e1d7d8`, `b48e49d`) verified present in `git log`. Full acceptance-criteria and `<verify>` re-run: `python manage.py test solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_admin solsys_code.tests.test_campaign_attribution solsys_code.tests.test_attribution_dismissals` (217 tests, OK); the project full-suite command (980 + 40 tests, OK); `pre-commit run ruff --all-files` and `ruff-format --all-files` (both Passed); `grep -c 'def unlink_event_from_run' solsys_code/campaign_utils.py` (1); the null-run-guard-precedes-queryset source verify (1); the forbidden-substring source verify (0); `grep -v '^ *#' solsys_code/campaign_reconciler.py | grep -c "update(run=None)"` (0); the admin in-memory-nulling source verify (1).
