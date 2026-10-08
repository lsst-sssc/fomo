---
phase: 39-calendar-write-access
plan: 04
subsystem: testing
tags: [django, tom_calendar, access-control, csrf, template-snapshot, security-ledger]

requires:
  - phase: 39-calendar-write-access
    provides: "39-01's login guards (calendar_access.py), 39-02's read-only presentation and six-item event_form.html header, 39-03's browser proof and phase gate"
provides:
  - "AnonymousCsrfFailureWriteTest: the CSRF-failure refusal path (302 or HX-Redirect to login with next = the refused path) pinned for all five write routes"
  - "Runbook read-only paragraph and calendar_access.py module docstring that describe both refusal paths truthfully, plus the open self-registration acceptance"
  - "event_form.html's Save and Edit label restored to tomtoolkit 3.1.0's, and a pinned normalized diff of the body against upstream (10 regions)"
  - "CR-01 recorded as an accepted risk in 39-SECURITY.md (AR-39-01 corrected, T-39-22) and 39-REVIEW-DISPOSITION.md (skipped)"
affects: [Phase 39 re-verification, /gsd-verify-work 39, Phase 42]

actuals:
  tokens: 9000
  tasks: 3
  commits: 3
plan_head_before: b2de6c2da136e39b780ccbfa5baa0444e9d06da0
plan_head_after: a8548ea79f22c26235dbf23f906d6a2e306d6d72
commits: 3

tech-stack:
  added: []
  patterns:
    - "Test the CSRF-failing path with Client(enforce_csrf_checks=True); the default test Client skips the CSRF check, so its assertions only describe a write that passes it"
    - "Pin a template override's normalized difflib diff against the installed upstream file, so any new difference fails until the header list and the snapshot are updated together"

key-files:
  created:
    - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
    - .planning/phases/39-calendar-write-access/39-04-red-evidence-task2.json
  modified:
    - solsys_code/tests/test_calendar_write_access.py
    - solsys_code/calendar_access.py
    - docs/runbooks/telescope_runs_calendar.rst
    - src/templates/tom_calendar/partials/event_form.html
    - solsys_code/tests/test_calendar_template.py
    - .planning/phases/39-calendar-write-access/39-SECURITY.md
    - .planning/phases/39-calendar-write-access/39-REVIEW-DISPOSITION.md

key-decisions:
  - "Gap 1 closed by docs and tests, not by a CSRF_FAILURE_VIEW (developer decision 2026-10-08): no behaviour, setting or route changed"
  - "CR-01 (open self-registration) recorded as an accepted risk with the developer's rationale verbatim; TOM_REGISTRATION_STRATEGY, D-01 and the guard unchanged"
  - "Restored upstream's Save and Edit label rather than documenting the case difference; pinned the full body diff instead of mapping header items to regions"

requirements-completed: [ACCESS-01, WARN-01]

coverage:
  - id: E1
    description: "A tokenless anonymous POST to each of the five write routes is sent to login with next = that route's own path (htmx: 200 with HX-Redirect), repeats identically, a missing id gives 302 not 404, a signed-in GET replay of the refused path writes nothing, and no row ever changes"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#AnonymousCsrfFailureWriteTest"
        status: pass
    human_judgment: false
  - id: E2
    description: "Runbook paragraph and guard docstring name both refusal paths and the self-registration acceptance, with the verbatim rationale"
    requirement: ACCESS-01
    verification:
      - kind: command
        ref: "Task 1 structural check script (AST-identical calendar_access.py code, phrases present, old false clause gone)"
        status: pass
    human_judgment: true
  - id: E3
    description: "The create form's Save and Edit label matches upstream and the body's full diff against tomtoolkit 3.1.0 is pinned, so an unlisted line inside an anchored region fails a test"
    requirement: WARN-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventFormHeaderMatchesUpstreamTest.test_body_diff_matches_pinned_snapshot"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventFormHeaderMatchesUpstreamTest.test_snapshot_detects_an_unlisted_line_inside_an_anchored_region"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventModalReadOnlyCardTest.test_signed_in_create_form_uses_upstream_button_labels"
        status: pass
    human_judgment: false
  - id: E4
    description: "Whole tree green after this plan: full suite with functional tests, ruff twice, check, makemigrations, tom_calendar RECORD hashes"
    requirement: ACCESS-01
    verification:
      - kind: command
        ref: "python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault (Ran 2241 tests, OK)"
        status: pass
    human_judgment: false
  - id: E5
    description: "CR-01 accepted-risk record: verbatim rationale in runbook, 39-SECURITY.md and the review ledger; only the intended lines changed"
    requirement: ACCESS-01
    verification:
      - kind: command
        ref: "Task 3 records check script (OK: CR-01 recorded as skipped ...)"
        status: pass
    human_judgment: false

duration: 2h
completed: 2026-10-08
status: complete
---

# Phase 39 Plan 04: Gap Closure (CSRF Refusal Path, Accepted Self-Registration Risk, Upstream Label) Summary

**The CSRF-failure refusal path is pinned for all five calendar write routes and documented truthfully, open self-registration is on record as an accepted risk, and event_form.html carries upstream's "Save and Edit" label with its full diff against tomtoolkit 3.1.0 pinned in a snapshot**

## Performance

- **Duration:** about 2 h across the Task 1 session, a checkpoint pause, and this continuation (two full pre-commit django-test hooks, one full suite)
- **Completed:** 2026-10-08
- **Tasks:** 3 (Task 1 was a tracer ending at a human-verify checkpoint)
- **Files modified:** 9 (2 created)

## Accomplishments

- Verification gap 1 (39-REVIEW WR-01) is closed with no behaviour change: a logged-out write that fails the CSRF check is sent to `/accounts/login/?next=<its own path>` (htmx: `HX-Redirect` to the same), and every one of those refusals, repeats, missing ids and signed-in GET replays leaves all rows unchanged. The runbook and `calendar_access.py` now say so.
- CR-01 is recorded in all three places in the developer's words (runbook, 39-SECURITY.md AR-39-01 plus new T-39-22, review ledger CR-01 skipped).
- WR-02 is resolved: the label matches upstream, header item 3 says so, and a pinned snapshot now catches any unlisted difference, including inside the large inserted region the anchor rule could not see.

## Results

### Task 1 (tracer), commit `bc73bfd`

- `python manage.py test --noinput solsys_code.tests.test_calendar_write_access`: `Ran 30 tests in 0.954s` ... `OK`.
- Non-vacuity run (setUp temporarily built `Client()` without CSRF enforcement, then restored): `Ran 5 tests` ... `FAILED (failures=22)`. Failing tests: the Location test (5 subtests), the HX-Redirect test (5), the repeat test (5), the missing-event test (2) and the replay test (5), as the guard then answers `next=/calendar/`.
- Structural check: `OK: docstring-only guard change, behaviour files untouched, existing tests intact, precondition declared, runbook paragraph true for both paths`.
- The user approved the checkpoint on the rewritten runbook paragraph and the `calendar_access.py` module docstring ("approved"); neither was changed afterwards. Full pre-commit hook (django-test included) passed.

### Task 2, commit `65ba57c`

- Pre-change grep for `Save and` in `solsys_code/tests/` and `docs/` found nothing outside the new tests (the Playwright round trip clicks the exact name `Save`), so no other test or doc pinned the label.
- RED (`.planning/phases/39-calendar-write-access/39-04-red-evidence-task2.json`, `Ran 17 tests`, `FAILED (failures=3, errors=1)`):
  - `FAIL: test_signed_in_create_form_uses_upstream_button_labels` on `assertIn('>Save and Edit</button>', ...)` (the form rendered the lowercase-e label). This is the target test.
  - `FAIL: test_every_differing_region_is_listed_and_every_item_differs` (anchor `Save and Edit` in no differing region).
  - `FAIL: test_body_diff_matches_pinned_snapshot` (snapshot file missing).
  - `ERROR: test_snapshot_detects_an_unlisted_line_inside_an_anchored_region` (FileNotFoundError reading the not-yet-created snapshot; expected, the plan names the missing snapshot as a RED reason).
  - Classifier: `gsd-tools check tdd-red-evidence` printed `RED_EVIDENCE_OK`, reason `target_test_failed`.
- GREEN: `python manage.py test --noinput solsys_code.tests.test_calendar_template`: `Ran 97 tests` ... `OK` (also after the ruff-format pass). Functional run (`test_signed_in_editor_creates_edits_and_deletes_from_month_view`, `test_calendar_modal_opens_for_new_event_button_with_no_page_errors`): `Ran 2 tests in 4.460s` ... `OK`.
- Structural check: `OK: one form line differs from the pre-phase template (upstream label), header item 3 accurate, snapshot pinned with 10 regions`. The `<form>` block differs from f929f4e in exactly the `save_and_edit` line (VERIFICATION truth B6 deliberately adjusted).
- Full pre-commit hook (django-test included) passed.
- The 10 snapshot regions mapped to header items:

| # | Snapshot region | Header item |
|---|-----------------|-------------|
| 1 | `replace upstream 1-1 fomo-body 1-2` (load line plus the `is_authenticated` opener) | 1 (and 5) |
| 2 | `replace upstream 31-31 fomo-body 32-32` (`is_web_url` gate) | 2 |
| 3 | `replace upstream 33-33 fomo-body 34-34` (`rel="noopener noreferrer"`) | 2 |
| 4 | `insert upstream 35-34 fomo-body 36-37` (`(not a web link)`) | 2 |
| 5 | `replace upstream 65-65 fomo-body 68-68` (Save button) | 3 |
| 6 | `replace upstream 67-67 fomo-body 70-70` (Save and Edit button) | 3 |
| 7 | `replace upstream 69-72 fomo-body 72-76` (Delete button) | 3 |
| 8 | `insert upstream 75-74 fomo-body 79-275` (else branch, `cal-event-card`, series and campaign decoration) | 4 and 5 |
| 9 | `insert upstream 78-77 fomo-body 279-279` (todos `is_authenticated` opener) | 6 |
| 10 | `insert upstream 83-82 fomo-body 285-297` (read-only todo list from `event.todos.all`) | 6 |

### Task 3, commit `a8548ea` (`SKIP=django-test`)

- Full suite `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault`: `Ran 2241 tests in 470.316s`, `OK`; no `FAILED`, no `skipped=` line. Log: `$HOME/tmp/phase39-04-suite.log`.
- `pre-commit run ruff --all-files` and `ruff-format --all-files`, each run twice: all `Passed`; no tracked file outside `.planning` modified (`OK: both ruff hooks pass twice and change nothing`).
- `python manage.py check`: only `?: (urls.W005) URL namespace 'calendar' isn't unique` (1 issue). `makemigrations --check --dry-run`: `No changes detected`.
- RECORD hash check: `OK: 22 installed tom_calendar files match tomtoolkit 3.1.0 RECORD hashes`.
- Records check: `OK: CR-01 recorded as skipped with the verbatim rationale; SECURITY.md premise corrected and T-39-22 accepted; nothing else changed`; exactly three ledger lines differ; one commit holds exactly the two records.
- `git diff --numstat` for `a8548ea`: `3 3 .../39-REVIEW-DISPOSITION.md`, `3 2 .../39-SECURITY.md`.

## Task Commits

1. **Task 1 (tracer): CSRF-failure path tests, guard docstring, runbook paragraph** - `bc73bfd` (docs)
2. **Task 2: restore upstream's Save and Edit label, pinned diff snapshot** - `65ba57c` (fix)
3. **Task 3: record CR-01 as an accepted risk** - `a8548ea` (docs)

## Deviations from Plan

None - plan executed exactly as written. One note: the Task 2 RED run showed one ERROR (the snapshot file read) beside the three FAILs; the plan expects the snapshot tests to fail because the file does not exist, and the target test failed as a FAIL on the planned assertion, so the classifier gave `RED_EVIDENCE_OK`.

## Paired docs

The runbook `docs/runbooks/telescope_runs_calendar.rst` was updated in Task 1 in the same commit as the docstring it mirrors. No notebook changed: none pairs with `calendar_access.py` or the calendar templates, and none contains the button label.

## Issues Encountered

- `ruff-format` reformatted `solsys_code/tests/test_calendar_template.py` once before the Task 2 commit; re-checked (ruff and ruff-format `Passed`, module tests `OK`) and then committed.
- Task 3's full suite was started from a detached subshell, so the harness's completion notice fired early; completion was confirmed by waiting for the `.rc` file and reading the log.
- The unrelated working-tree changes (`.planning/config.json`, `.planning/state.json`, `.planning/ui-reviews/.gitignore`, `reqgroup_2682493.json`, `.planning/agent-history.json`, `.planning/milestone.lock`) were left as found.

## Known Stubs

None.

## Threat Flags

None. No endpoint, auth path, file access or schema changed; this plan changes tests, documentation, one button label and planning records. 39-01-PLAN.md, 39-VERIFICATION.md, `src/fomo/settings.py`, `solsys_code/calendar_urls.py`, `src/fomo/urls.py` and `calendar.html` were not touched.

## User Setup Required

None.

## Next Phase Readiness

Verification gaps 1 and 2 are closed (gap 2 by the override in 39-VERIFICATION.md), CR-01 and WR-02 are recorded, and the tree is green. IN-03 and human items 3-4 remain for `/gsd-verify-work`; Phase 39 is ready for re-verification.

## Self-Check: PASSED

- Files found: `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff`, `.planning/phases/39-calendar-write-access/39-04-red-evidence-task2.json`, and the seven modified files.
- Commits found: `bc73bfd`, `65ba57c`, `a8548ea` (ancestors of HEAD; `git rev-list --count b2de6c2..HEAD` = 3).

---
*Phase: 39-calendar-write-access*
*Completed: 2026-10-08*
