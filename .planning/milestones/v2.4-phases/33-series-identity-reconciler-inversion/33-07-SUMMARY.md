---
phase: 33-series-identity-reconciler-inversion
plan: 07
subsystem: api
tags: [django-admin, campaign-attribution, unlink-event-from-run, gap-closure, type-safety]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion
    provides: "unlink_event_from_run() as the single writer that clears a CalendarEventMeta attribution, from plan 33-04"
provides:
  - "UNLINK_CLEARED_FIELDS: one exported field set in campaign_utils.py, consumed by both unlink_event_from_run()'s bulk .update() and CalendarEventMetaAdmin.save_model()'s in-memory clear (WR-02 closed)"
  - "unlink_event_from_run() raises TypeError for a str/bytes events argument instead of silently expanding it into a per-character event__in filter (WR-04 closed)"
  - "CalendarEventMetaInline's docstring corrected to match what Django actually renders -- a hidden parent-linkage field, never an editable run widget (WR-06 code-side half closed)"
affects: ["33-08"]

actuals:
  tokens: 5550
  tasks: 3
  commits: 5

tech-stack:
  added: []
  patterns:
    - "A single module-level constant (UNLINK_CLEARED_FIELDS) consumed by two writers via **kwargs unpacking (bulk .update()) and a setattr loop (in-memory clear), rather than each writer keeping its own copy of the field list"
    - "unittest.mock.patch.dict on a module-level dict constant to prove a drift guard actually derives from the constant, not from independently-matching hand-written code"
    - "str | bytes isinstance check inserted between a specific-type branch and an iterable catch-all, to reject an iterable-shaped input before it silently expands per-character"

key-files:
  created: []
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/admin.py
    - solsys_code/tests/test_admin.py
    - solsys_code/tests/test_campaign_attribution_views.py

key-decisions:
  - "UNLINK_CLEARED_FIELDS is imported locally inside CalendarEventMetaAdmin.save_model(), not at admin.py's module level -- campaign_utils imports campaign_reconciler at its own top level, which pulls telescope_runs and astropy into every Django admin autodiscover; mirrors campaign_reconciler._detach_stale_family_events()'s existing local-import pattern for the same reason."
  - "The fourth-key drift proof uses a plain, non-model sentinel attribute (patched into UNLINK_CLEARED_FIELDS via patch.dict) rather than a real CalendarEventMeta field -- the model's only other nullable columns are the PROJ-04 carrier fields this plan's prohibitions forbid touching, and a real field would need a migration."
  - "Task 3's docstring and test were corrected mid-task: Django's BaseInlineFormSet.add_fields() actually re-binds the fk_name field as a hidden InlineForeignKeyField (parent-linkage Django needs on POST), not a fully absent field as the plan's draft wording assumed. The docstring and the wired test were both revised to assert 'no editable widget' rather than 'no field at all', which is what the review finding (33-REVIEW.md WR-06) actually claims and what the codebase genuinely does."

requirements-completed: [ANNOT-01, PROJ-04]

coverage:
  - id: D1
    description: "UNLINK_CLEARED_FIELDS is importable, maps run/confirmed_by/confirmed_at to None, and both the bulk .update() and the admin's in-memory clear derive from it -- clearing an attribution through the admin standalone form leaves the stored row's fields null with is_verified/observation_record/observation_group/CalendarEvent/object-counts unchanged"
    requirement: "PROJ-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#CalendarEventMetaStandaloneAdminAuditStampTests.test_clearing_the_run_clears_the_audit_fields"
        status: pass
    human_judgment: false
  - id: D2
    description: "Adding a fourth key to UNLINK_CLEARED_FIELDS makes the admin clear path null that key too, with no edit to admin.py"
    requirement: "PROJ-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#CalendarEventMetaStandaloneAdminAuditStampTests.test_clearing_the_run_honours_a_fourth_key_added_to_unlink_cleared_fields"
        status: pass
    human_judgment: false
  - id: D3
    description: "unlink_event_from_run() called with a bare int primary key clears exactly that one event, and called with a queryset clears every event in it -- both branches directly asserted"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun.test_bare_int_primary_key_clears_exactly_that_one_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun.test_queryset_clears_every_matching_event"
        status: pass
    human_judgment: false
  - id: D4
    description: "unlink_event_from_run() called with a str or bytes events argument raises TypeError and writes nothing -- a string primary key is never silently expanded into a per-character event__in filter"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun.test_string_primary_key_raises_type_error_and_changes_nothing"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestUnlinkEventFromRun.test_bytes_primary_key_raises_type_error_and_changes_nothing"
        status: pass
    human_judgment: false
  - id: D5
    description: "CalendarEventMetaInline renders no editable attribution widget, and its docstring states the un-attribution operation as it actually exists (delete the row, or clear the value on the standalone change page)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#CampaignRunAdminInlinesTests.test_calendar_event_meta_inline_renders_no_editable_attribution_field"
        status: pass
    human_judgment: false

duration: 26min
completed: 2026-09-06
status: complete
---

# Phase 33 Plan 07: One Declaration, One Type Check, One Corrected Docstring Summary

**Closed two review warnings about 33-04's "one shared unlink helper" claim: `UNLINK_CLEARED_FIELDS` is now the single field-set declaration both writers consume, `unlink_event_from_run()` rejects a `str`/`bytes` primary key with `TypeError` instead of silently per-character-expanding it, and the inline's docstring now matches what Django actually renders.**

## Performance

- **Duration:** 26 min
- **Started:** 2026-09-06T00:28:00Z
- **Completed:** 2026-09-06T00:54:35Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- **WR-02 closed:** `campaign_utils.UNLINK_CLEARED_FIELDS` is the single declaration of what clearing a campaign attribution means (`run`, `confirmed_by`, `confirmed_at` -> `None`). `unlink_event_from_run()`'s bulk `.update()` unpacks it directly; `CalendarEventMetaAdmin.save_model()` branch 2 loops over it via a function-local import. A companion test proves this with a `patch.dict`-injected fourth sentinel key, distinguishing a real loop from three hand-written assignments that happened to look the same for three keys.
- **WR-04 closed:** `unlink_event_from_run()`'s type dispatch now raises `TypeError` for a `str` or `bytes` `events` argument, inserted between the `int` branch and the iterable catch-all, after the `if not run_pk: return 0` guard. Both currently-used call shapes (bare `int` from `_undo_confirmation()`, queryset from `_detach_stale_family_events()`) are now directly asserted rather than only inferred from the `CalendarEvent`-instance branch.
- **WR-06 (code side) closed:** `CalendarEventMetaInline`'s docstring previously told the reader that clearing the `run` value on a row un-attributes the event. Corrected to state what actually exists on this surface: deleting the row (Django's `BaseInlineFormSet.add_fields()` re-binds `fk_name='run'` as a hidden internal `InlineForeignKeyField`, never an editable widget), or clearing the value on the standalone `CalendarEventMetaAdmin` change page. A wired test confirms the hidden linkage field renders but no `<select>` widget for `run` does.

## Task Commits

Each task was committed atomically (Tasks 1-2 are TDD, producing a `test(...)` -> `feat(...)` pair each):

1. **Task 1 RED: failing UNLINK_CLEARED_FIELDS drift-guard tests** - `28c836f` (test)
2. **Task 1 GREEN: declare UNLINK_CLEARED_FIELDS, consume it from both writers** - `3f96186` (feat)
3. **Task 2 RED: failing str/bytes rejection tests** - `aa99f48` (test)
4. **Task 2 GREEN: reject str/bytes in unlink_event_from_run's type dispatch** - `b4cd339` (feat)
5. **Task 3: correct CalendarEventMetaInline's docstring, add the wired no-editable-widget test** - `8aa82c1` (docs)

_Task 1 is `type="tracer"` -- verified end-to-end (test_admin OK, grep counts, makemigrations --check clean) immediately after its GREEN commit, per the plan's `HUMAN_VERIFY_MODE=end-of-phase` + automated-only `<verify>` rule; no checkpoint synthesized, expansion to Task 2 proceeded on the automated pass._

## Files Created/Modified

- `solsys_code/campaign_utils.py` - added `UNLINK_CLEARED_FIELDS` module constant; `unlink_event_from_run()`'s `.update()` now unpacks it; added the `str`/`bytes` `TypeError` rejection branch; docstring updated (Args/Raises)
- `solsys_code/admin.py` - `CalendarEventMetaAdmin.save_model()` branch 2 replaced with a loop over `UNLINK_CLEARED_FIELDS.items()` behind a function-local import; `CalendarEventMetaInline`'s docstring corrected to name the two operations that actually exist
- `solsys_code/tests/test_admin.py` - `test_clearing_the_run_clears_the_audit_fields` rewritten to iterate `UNLINK_CLEARED_FIELDS`; added `test_clearing_the_run_honours_a_fourth_key_added_to_unlink_cleared_fields` and `test_calendar_event_meta_inline_renders_no_editable_attribution_field`
- `solsys_code/tests/test_campaign_attribution_views.py` - added four `TestUnlinkEventFromRun` methods: bare-int, queryset, str-rejection, bytes-rejection

## Decisions Made

- `UNLINK_CLEARED_FIELDS` is consumed via a function-local import inside `save_model()`, not a module-level import in `admin.py` -- keeps `campaign_reconciler`/`telescope_runs`/astropy out of the admin's autodiscover import graph, mirroring the existing local-import pattern in `campaign_reconciler._detach_stale_family_events()`.
- The fourth-key drift proof patches in a plain non-model sentinel attribute via `patch.dict`, rather than a real `CalendarEventMeta` field, avoiding a migration and avoiding any risk of writing a PROJ-04 carrier field.
- Task 3's docstring and test were revised mid-task after checking Django's actual rendering: `fk_name='run'` produces a hidden `InlineForeignKeyField`, not a fully absent field. Both the docstring and the wired test assert "no editable widget", which is what 33-REVIEW.md WR-06 actually claims -- not "no field at all", which the plan's draft wording assumed but the codebase does not do.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 3's test assertion was checking the wrong thing (`name=` presence instead of widget visibility)**
- **Found during:** Task 3, first test run
- **Issue:** The plan's action described asserting that "the prefixed `run` input name does not appear in the response body." Running the test against a real rendered admin page showed `name="calendar_event_metas-0-run"` DOES appear -- as a `type="hidden"` `InlineForeignKeyField` Django's `BaseInlineFormSet.add_fields()` adds back for parent-linkage validation on POST, not as an editable widget. The as-written assertion would have permanently failed against Django's genuine behavior.
- **Fix:** Rewrote the test to assert the hidden linkage field IS present (control: the row genuinely rendered) and that no `<select name="calendar_event_metas-0-run"` widget exists -- the actual, correct proof that no editable attribution field is offered. Updated `CalendarEventMetaInline`'s docstring to describe the hidden-field mechanism precisely rather than claiming the field is fully absent.
- **Files modified:** `solsys_code/admin.py`, `solsys_code/tests/test_admin.py`
- **Verification:** `python manage.py test solsys_code.tests.test_admin` -- 55 tests, OK
- **Committed in:** `8aa82c1` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in the plan's draft test assertion, corrected against Django's real rendering behavior).
**Impact on plan:** No scope creep -- the fix makes the test and docstring assert the review finding's actual claim ("no editable widget"), which is a strictly more accurate version of the plan's own intent, not a different behavior.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- WR-02, WR-04 and the code half of WR-06 are all closed. `UNLINK_CLEARED_FIELDS` has exactly two consumers (`unlink_event_from_run()`'s bulk `.update()` and `CalendarEventMetaAdmin.save_model()`'s in-memory clear); `grep -rn "confirmed_by=None"` across `solsys_code/` (excluding tests) shows no third hand-written copy.
- Plan 33-08 owns the operator-facing runbook prose for the remaining WR-06 half (`docs/runbooks/telescope_runs_calendar.rst`) and can now refer to this plan's corrected admin docstring and wired test.
- No `CalendarEvent` field, `is_verified` value, `observation_record`, or `observation_group` was written anywhere in this plan's diff -- confirmed by the unchanged assertions in `test_clearing_the_run_clears_the_audit_fields` and the full test suite passing with no regressions.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-06*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_utils.py
- FOUND: solsys_code/admin.py
- FOUND: solsys_code/tests/test_admin.py
- FOUND: solsys_code/tests/test_campaign_attribution_views.py
- FOUND commit 28c836f (Task 1 RED)
- FOUND commit 3f96186 (Task 1 GREEN)
- FOUND commit aa99f48 (Task 2 RED)
- FOUND commit b4cd339 (Task 2 GREEN)
- FOUND commit 8aa82c1 (Task 3)
- `python manage.py test solsys_code.tests.test_admin solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_campaign_attribution solsys_code.tests.test_campaign_reconciler` -- 210 tests, OK
- `python manage.py makemigrations solsys_code --check --dry-run` -- No changes detected
- Full project test-command (`.planning/config.json` `workflow.test_command`) -- OK, no regressions (no `data.minorplanetcenter.net` flake this run)
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` -- both Passed
