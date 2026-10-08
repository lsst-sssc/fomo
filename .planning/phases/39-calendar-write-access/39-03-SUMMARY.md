---
phase: 39-calendar-write-access
plan: 03
subsystem: testing
tags: [playwright, django, tom_calendar, access-control, review-ledger]

requires:
  - phase: 39-calendar-write-access
    provides: "39-01's login guards on the five calendar write routes (commits 5e31c0f, 4d32e62) and 39-02's read-only presentation, six-item event_form.html header and signed-in session-cookie hand-off (commits 8d0eca5, 9b2df4d, 99b60b1)"
provides:
  - "Browser proof (Chromium) that a plain logged-in user creates, edits and deletes a CalendarEvent from the month view through the guarded routes"
  - "Phase 39 gate on the whole tree: full local suite OK, ruff twice clean, check shows only urls.W005, no migrations, tom_calendar untouched"
  - "WR-05 recorded fixed in 37.1-REVIEW-DISPOSITION.md (WARN-01) and 33-REVIEW.md (ACCESS-01), with a WR-04 note"
affects: [Phase 42 re-verification, /gsd-verify-work 39]

actuals:
  tokens: 9000
  tasks: 2
  commits: 2
plan_head_before: 7efff0fa848cc962692c36feafadb3eba0c98f51
plan_head_after: 798dfe98b3d03ed10917a6a14e181015458ba535
commits: 2

tech-stack:
  added: []
  patterns:
    - "Poll the live server's database (_wait_for_db) instead of sleeping, because the browser sees the modal close before the server thread's write is visible"
    - "Reload the 2026-08 month page before each reopen; upstream re-renders the current year's month after a save"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_bootstrap5_rendering.py
    - .planning/milestones/v2.4-phases/37.1-close-gap-alloc-06-exact-identity-system-links-on-ingest-int/37.1-REVIEW-DISPOSITION.md
    - .planning/milestones/v2.4-phases/33-series-identity-reconciler-inversion/33-REVIEW.md

key-decisions:
  - "Counted the update form specifically (form[hx-post*='/calendar/update/']) in the edit step, because upstream's pop-up for a saved event also holds a separate add-a-todo form"
  - "Cited 9b2df4d (not 8d0eca5) in the WR-04 note: that commit carries the calendar.html Bootstrap 5 names"

requirements-completed: [ACCESS-01, ACCESS-02, WARN-01]

coverage:
  - id: D1
    description: "A logged-in non-staff user creates an event from an empty day cell, edits its title, and deletes it from the month view; the modal closes each time and no page error is raised"
    requirement: ACCESS-01
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_signed_in_editor_creates_edits_and_deletes_from_month_view"
        status: pass
    human_judgment: false
  - id: D2
    description: "Whole tree is green with the phase's changes: full suite with functional tests, ruff twice, check, makemigrations, tom_calendar RECORD hashes"
    requirement: ACCESS-02
    verification:
      - kind: command
        ref: "python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault (Ran 2233 tests, OK)"
        status: pass
    human_judgment: false
  - id: D3
    description: "WR-05 recorded fixed in both review ledgers, no other disposition changed"
    requirement: WARN-01
    verification:
      - kind: command
        ref: "ledger check script (front matter, table row, 33-REVIEW notes, one commit with exactly the two files)"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-10-08
status: complete
---

# Phase 39 Plan 03: Editor Browser Proof, Phase Gate and Ledgers Summary

**A Chromium test shows a plain logged-in user still creates, retitles and deletes a calendar event from the month view through the guarded routes; the whole tree passes (2233 tests), and WR-05 is recorded fixed in both review ledgers**

## Performance

- **Duration:** about 45 min (the Task 1 pre-commit django-test hook and the full suite ran a few minutes each)
- **Completed:** 2026-10-08
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Phase 39 success criterion 2 is proven where it is stated, the month view: the editor clicks the empty 2026-08-20 cell (start pre-filled `2026-08-20T00:00`), saves `EditorTrip`, reopens it, saves `EditorTrip2` on the same row, then presses Delete and accepts the confirm dialog; the row is gone. The modal closes after every step and no page error is raised.
- The phase gate is green on the final tree and the installed `tom_calendar` is provably unedited.
- D-12: both WR-05 fixes are recorded, each pointing at the commits and tests that prove it.

## Results

**Task 1 functional run** (the five named tests): `Ran 5 tests in 9.647s` ... `OK`. The new test also passed twice more alone (about 5.5 s each). Commit `ba37d8e` (full pre-commit hook, django-test passed).

**Task 2 gate**

- Full suite `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault`: `Ran 2233 tests in 484.806s`, `OK`; no `FAILED`, no `skipped=` line; the flaky observatory test passed. Log: `$HOME/tmp/phase39-03-suite.log`.
- `pre-commit run ruff --all-files` and `ruff-format --all-files`, each run twice: all `Passed`; no tracked file outside `.planning` modified.
- `python manage.py check`: only `?: (urls.W005) URL namespace 'calendar' isn't unique` (System check identified 1 issue).
- `python manage.py makemigrations --check --dry-run`: `No changes detected`.
- RECORD hash check: `OK: 22 installed tom_calendar files match tomtoolkit 3.1.0 RECORD hashes`.
- Ledger check: `OK: 37.1 WR-05 fixed in front matter and table only; 33-REVIEW WR-05 and WR-04 notes present` (the changed disposition set was exactly `['WR-05']`) and `OK: one ledger commit holding exactly the two ledgers`.
- Ledger commit `798dfe9` (`SKIP=django-test`), `git diff --numstat`: `4 0 .../33-REVIEW.md`, `2 2 .../37.1-REVIEW-DISPOSITION.md`.

## Task Commits

1. **Task 1 (tracer): editor create, edit, delete in the browser** - `ba37d8e` (test)
2. **Task 2: gate, then WR-05 recorded fixed in both ledgers** - `798dfe9` (docs)

The tracer's `<verify>` is automated-only, so it was re-run end to end after the commit (the three-times-green runs above) before Task 2 started.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan assumption] The edit pop-up holds two forms, not one**
- **Found during:** Task 1 (first run)
- **Issue:** The plan asked to assert `#cal-modal-body form` has count 1 for the editor. Upstream's pop-up for a saved event also carries the separate add-a-todo form (`/calendar/todo/create/<id>/`), so Playwright raised a strict-mode violation (2 elements).
- **Fix:** The edit step waits for and counts `#cal-modal-body form[hx-post*="/calendar/update/"]` (exactly 1) and also asserts there is no `#cal-event-card`, which still proves the editor gets the form, not the visitor card.
- **Files modified:** `solsys_code/tests/test_bootstrap5_rendering.py`
- **Commit:** `ba37d8e`

**Total deviations:** 1 auto-fixed (test locator). **Impact:** none on behavior or scope.

## Paired docs

No notebook changed in this plan, and no module with a paired notebook changed. The runbook (`docs/runbooks/telescope_runs_calendar.rst`) was updated in 39-02; this plan changes no behavior.

## Issues Encountered

- `git commit` was started in a detached subshell, so the harness's completion notice fired early; the commit was confirmed by polling its log for the exit line and `git log -1 --stat`.
- The unrelated working-tree changes (`.planning/config.json`, `.planning/state.json`, `.planning/ui-reviews/.gitignore`, `reqgroup_2682493.json`, `.planning/agent-history.json`, `.planning/milestone.lock`) were left exactly as found.

## Known Stubs

None.

## Threat Flags

None. The only code change is a test; the browser logs in through the existing `force_login` session-cookie hand-off (T-39-17), the ledger edits touch only WR-05 and the WR-04 note (T-39-18), and the RECORD hash check passed (T-39-19).

## User Setup Required

None.

## Next Phase Readiness

Phase 39 is complete: ACCESS-01, ACCESS-02 and WARN-01 are all delivered and recorded. Ready for phase verification.

## Self-Check: PASSED

- Files found: `solsys_code/tests/test_bootstrap5_rendering.py`, both ledgers.
- Commits found: `ba37d8e`, `798dfe9` (ancestors of HEAD; `git rev-list --count 7efff0f..HEAD` = 2).

---
*Phase: 39-calendar-write-access*
*Completed: 2026-10-08*
