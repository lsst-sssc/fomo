---
phase: 28-operator-assisted-attribution
plan: 01
subsystem: database
tags: [django, migrations, admin, models, sqlite]

# Dependency graph
requires:
  - phase: 27-the-canonical-run-record
    provides: CalendarEventMeta.run and CampaignRunObservation, the two link models this
      phase's schema attaches to
provides:
  - CalendarEventDismissal / ObservationRecordDismissal typed per-pair dismissal models
    (D-05/D-06/D-08)
  - CalendarEventMeta.confirmed_by / confirmed_at audit fields (D-12), closing the
    event-side audit asymmetry Phase 27's D-05 deliberately left open
  - Migration 0013 applying all four schema changes in one hand-annotated, autodetected
    migration
  - CampaignRunAdmin.save_formset stamps the new CalendarEventMeta audit fields on a
    genuine run_id None-to-not-None transition
affects: [28-02, 28-03, 28-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Named UniqueConstraint per (orphan, run) pair with an explanatory comment (no bare
      unique_together), mirroring CampaignRunObservation/CampaignRun's existing convention"
    - "Transition-detection for a field that IS the primary key: fetch the prior DB value via
      a separate .filter(pk=...).values_list(...).first() query before the mutated in-memory
      instance hides it, rather than testing instance.pk is None"

key-files:
  created:
    - solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py
    - solsys_code/tests/test_attribution_dismissals.py
  modified:
    - solsys_code/models.py
    - solsys_code/admin.py
    - solsys_code/tests/test_admin.py

key-decisions:
  - "CalendarEventMetaInline.readonly_fields extended to cover confirmed_by/confirmed_at
    (T-28-02 mitigation) -- without this, Django's admin auto-generates editable fields for
    every non-excluded model field, which would let a staff member hand-type either audit
    value through the inline form, exactly the tampering the plan's own threat model flags"
  - "The 'link a previously-unowned CalendarEventMeta row to a run' test scenario is
    achieved by creating the companion row and the run link in one inline submission, not by
    pre-creating an unowned row and re-pointing it via the inline's blank row -- Django's
    ModelForm uniqueness validation on the OneToOneField-as-primary-key `event` field
    rejects an 'add' formset row whose event pk already has a companion row (a genuine
    mechanical limit of this admin surface, not a test simplification)"
  - "Seeded a local, gitignored AnonymousUser row directly into this worktree's dev SQLite
    file to unblock django-guardian's post_migrate signal, which was crashing due to a stale
    pre-Django-1.8-era auth_user schema already present in that file -- no tracked file
    touched, and Django's own test runner uses a fresh in-memory DB unaffected by this"

requirements-completed: [ATTRIB-03, ATTRIB-04]

# Metrics
duration: ~20min
completed: 2026-08-01
---

# Phase 28 Plan 01: Attribution Schema Summary

**Two typed per-pair dismissal models (CalendarEventDismissal, ObservationRecordDismissal), CalendarEventMeta.confirmed_by/confirmed_at audit fields, and admin-side stamping on a genuine run-link transition -- the schema every later plan in this phase reads or writes.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-08-01 (session start; first commit 08:41 PDT)
- **Completed:** 2026-08-01T08:50:34-07:00
- **Tasks:** 3
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments
- `CalendarEventDismissal` / `ObservationRecordDismissal`: real FKs, named per-pair
  `UniqueConstraint`s, `dismissed_by`/`dismissed_at`/`reason` audit fields, `CASCADE` on
  both sides (a dismissal row without its pair means nothing)
- `CalendarEventMeta.confirmed_by`/`confirmed_at` close the event-side audit gap Phase 27's
  D-05 deliberately left open; the stale "do not fix it here" comment is replaced with one
  describing what the code now does
- Migration 0013 (autodetected via `makemigrations`, hand-annotated header per migration
  0010's precedent) applies all four schema changes together; `makemigrations --check
  --dry-run` stays clean
- `CampaignRunAdmin.save_formset` gains a `CalendarEventMeta` branch keyed on a `run_id`
  None-to-not-None transition (not `instance.pk is None`, which is never true for this
  model), with a matching `readonly_fields` lock so the new audit fields can never be
  hand-typed through the inline form
- 15 new/rewritten tests: 13 in the new `test_attribution_dismissals.py` module, 2
  replacing the admin inline test whose assertion D-12 deliberately reverses

## Task Commits

Each task was committed atomically:

1. **Task 1: Two dismissal models, the event-side audit fields, and migration 0013** -
   `650f787` (feat)
2. **Task 2: Admin save_formset stamps the event-side audit fields on a genuine run-link
   transition** - `1ab4031` (feat)
3. **Task 3: Constraint, cascade and audit-default tests for both dismissal models** -
   `531a1be` (test)

## Files Created/Modified
- `solsys_code/models.py` - `CalendarEventDismissal`, `ObservationRecordDismissal`,
  `CalendarEventMeta.confirmed_by`/`confirmed_at`, replaced D-05 comment
- `solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py` -
  schema for the two new models and the two new fields
- `solsys_code/admin.py` - `save_formset` `CalendarEventMeta` branch,
  `CalendarEventMetaInline.readonly_fields` extended
- `solsys_code/tests/test_admin.py` - two new stamping tests, one superseded test removed
- `solsys_code/tests/test_attribution_dismissals.py` - new module, 13 tests across three
  test classes

## Decisions Made
- `CalendarEventMetaInline.readonly_fields` gains `confirmed_by`/`confirmed_at` (Rule 2 /
  T-28-02 mitigation) -- the plan's task text didn't call this out explicitly, but without
  it the inline form exposes both new fields as freely editable, directly contradicting the
  threat model's own "stamped server-side ... never bound from POST data" disposition. This
  mirrors `CampaignRunObservationInline`'s existing identical lock.
- Reworked the "link a previously-unowned row" test to create-and-link in one step rather
  than pre-create-then-repoint, after confirming Django's own ModelForm uniqueness
  validation rejects the pre-create-then-repoint shape (see key-decisions above for detail).
- Replaced `test_save_formset_does_not_stamp_calendar_event_meta_formset` (its assertion is
  what D-12 deliberately reverses) with
  `test_save_formset_stamps_calendar_event_meta_on_run_transition` and
  `test_save_formset_does_not_restamp_calendar_event_meta_on_unrelated_edit`, per the plan's
  explicit instruction to record this as a deliberate, D-12-mandated test revision.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Locked `CalendarEventMeta`'s new audit fields as read-only on the inline**
- **Found during:** Task 2
- **Issue:** Django's admin auto-generates form fields for every non-excluded, editable
  model field. Adding `confirmed_by`/`confirmed_at` to `CalendarEventMeta` without also
  adding them to `CalendarEventMetaInline.readonly_fields` would let a staff member submit
  arbitrary values for either field through the inline form -- exactly the tampering the
  plan's own threat model (T-28-02) requires mitigated, and the reason the second new test
  initially failed (a `None` `confirmed_by` on an "unrelated edit" the fields being silently
  cleared, not preserved).
- **Fix:** Added `readonly_fields = ['confirmed_by', 'confirmed_at']` to
  `CalendarEventMetaInline`, matching `CampaignRunObservationInline`'s existing identical
  lock, with a docstring note citing T-28-02.
- **Files modified:** `solsys_code/admin.py`
- **Verification:** `test_save_formset_does_not_restamp_calendar_event_meta_on_unrelated_edit`
  passes; both fields stay bound to their original values across an unrelated inline edit.
- **Committed in:** `1ab4031` (Task 2 commit)

**2. [Rule 3 - Blocking] Restored a gitignored build artifact so Django settings could import**
- **Found during:** Task 1 (before `makemigrations` could even run)
- **Issue:** `src/fomo/_version.py` (setuptools_scm's `write_to` target, gitignored, not
  committed) did not exist in this fresh worktree, so importing `src.fomo.settings` raised
  `ModuleNotFoundError: No module named 'src.fomo._version'` before any Django command could
  execute.
- **Fix:** Copied the existing generated file from the main checkout
  (`/home/tlister/git/fomo_devel/src/fomo/_version.py`) into the worktree at the same path.
  Not a tracked-file change (the file is gitignored in both locations) and not committed.
- **Files modified:** none tracked (local build artifact only)
- **Verification:** `python manage.py makemigrations` runs without import error.

**3. [Rule 3 - Blocking] Seeded a missing AnonymousUser row in this worktree's local dev DB**
- **Found during:** Task 1, verifying `python manage.py migrate solsys_code` exits 0
- **Issue:** Migration 0013 applied cleanly and was recorded in `django_migrations`, but the
  `migrate` command itself exited 1 because django-guardian's `post_migrate` signal handler
  (`create_anonymous_user`) crashed with `IntegrityError: NOT NULL constraint failed:
  auth_user.last_login`. Inspection showed this worktree's `src/fomo_db.sqlite3` (gitignored,
  not tracked, present at worktree creation) carries a stale, pre-Django-1.8-era `auth_user`
  schema (`username varchar(30)`, no nullable `last_login`) that predates the current model
  definitions -- unrelated to any change in this plan, and django-guardian's signal fires
  unconditionally on every `migrate` invocation regardless of what migrated.
- **Fix:** Inserted one `AnonymousUser` row (`id=-1`) directly into this worktree's local
  `auth_user` table via raw SQL, matching what the main checkout's DB already had (with an
  explicit non-null `last_login` to satisfy the stale schema's constraint). No tracked file
  touched; the DB file is gitignored and not part of this commit.
- **Files modified:** none tracked (local dev DB only)
- **Verification:** `python manage.py migrate solsys_code` now exits 0. Django's own test
  runner is unaffected either way -- it builds a fresh in-memory test database per run, so
  none of the 70 tests exercised in this plan's verification depended on this fix.

---

**Total deviations:** 3 auto-fixed (1 missing-critical, 2 blocking/environment)
**Impact on plan:** The readonly_fields fix (#1) is a real correctness/security fix inside
plan scope. Fixes #2 and #3 are local, untracked environment repairs with no effect on the
committed code or on `./manage.py test`'s isolated test database -- included here for
transparency, not because they touch any deliverable.

## Issues Encountered

`ruff check .` / `ruff format --check .` at the whole-project level report 4 pre-existing,
unrelated issues (a docstring gap in `docs/notebooks/pre_executed/sync_gemini_observation_
calendar_demo.ipynb` and formatting drift in `src/fomo/settings.py` plus two files under
`.planning/quick/260619-f7u.../`). Confirmed pre-existing on the plan's starting commit
(`575f3ec`) and untouched by any file this plan modifies; `ruff check`/`ruff format --check`
scoped to this plan's own files (`solsys_code/models.py`, `solsys_code/admin.py`,
`solsys_code/tests/test_admin.py`, `solsys_code/tests/test_attribution_dismissals.py`) both
pass cleanly, and the pre-commit hook (which lints only staged files) passed on all three
task commits. Migration files are excluded from ruff entirely by `pyproject.toml`'s
`[tool.ruff] exclude`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The schema this whole phase writes into now exists: `CalendarEventDismissal`,
  `ObservationRecordDismissal`, and `CalendarEventMeta.confirmed_by`/`confirmed_at`, all
  migrated and tested.
- 28-02 (the matcher) can now exclude dismissed pairs; 28-03 (the POST actions) can create
  dismissal rows and stamp the audit fields via the same `save_formset` pattern proven here
  for the admin path; 28-04 (the page) can render the Dismissed/Confirmed sections from
  these models.
- No blockers. The two local, untracked environment fixes (deviations #2/#3) are
  worktree-specific and do not need to propagate to other plans' worktrees unless they hit
  the identical stale-DB/missing-build-artifact conditions.

---
*Phase: 28-operator-assisted-attribution*
*Completed: 2026-08-01*

## Self-Check: PASSED

- FOUND: `solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py`
- FOUND: `solsys_code/tests/test_attribution_dismissals.py`
- FOUND: `.planning/phases/28-operator-assisted-attribution/28-01-SUMMARY.md`
- FOUND commit: `650f787` (Task 1)
- FOUND commit: `1ab4031` (Task 2)
- FOUND commit: `531a1be` (Task 3)
