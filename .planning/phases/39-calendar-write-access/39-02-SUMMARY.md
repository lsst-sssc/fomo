---
phase: 39-calendar-write-access
plan: 02
subsystem: ui
tags: [django, tom_calendar, templates, access-control, htmx, bootstrap5, playwright]

requires:
  - phase: 39-calendar-write-access
    provides: "plan 39-01's login guards on the five calendar write routes (the anonymous pop-up GET stays 200)"
  - phase: 38-tomtoolkit-3-1-0-floor
    provides: "tomtoolkit 3.1.0, whose upstream event_form.html / calendar.html the overrides are diffed against"
provides:
  - "event_form.html: a request.user.is_authenticated branch giving a visitor who is not logged in a read-only cal-event-card and a read-only todo list instead of the form"
  - "calendar.html: '+ New Event' button and day-cell create hx-* attributes only for a signed-in user, a cal-header-spacer for visitors, Bootstrap 5 utility names"
  - "event_form.html header rewritten as six numbered items checked against the installed upstream file (WARN-01)"
  - "Playwright proof of the anonymous read-only pop-up and of the signed-in create targets"
  - "runbook paragraph 'Not logged in, the calendar is read-only.'"
affects: [39-03 browser proof / requirement close-out, future tomtoolkit upgrades (header re-diff)]

actuals:
  tokens: 55000
  tasks: 3
  commits: 3
plan_head_before: b261c9c4aa40604a1a111c3cf443f39e27fc2db0
plan_head_after: 99b60b169e66440985d09debf1d3e3192476ce17
commits: 3

tech-stack:
  added: []
  patterns:
    - "Template-side audience branch on request.user.is_authenticated (same idiom as the existing request.user.is_staff hint); no view-supplied flag, no third template override"
    - "Override header kept honest by a test: difflib region-by-region comparison of the body against the installed upstream template, with per-item anchor literals"

key-files:
  created:
    - .planning/phases/39-calendar-write-access/39-02-red-evidence-task1.json
    - .planning/phases/39-calendar-write-access/39-02-red-evidence-task2.json
  modified:
    - src/templates/tom_calendar/partials/event_form.html
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_template.py
    - solsys_code/tests/test_bootstrap5_rendering.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Read-only branch lives inside event_form.html; the observation-series, campaign and staff-hint blocks stay below both branches and render once"
  - "A non-web URL value is never echoed on the anonymous card, only '(not a web link)' (RESEARCH OQ3)"
  - "Todos stay readable to visitors (done / not done), per D-05"
  - "Declined re-scoping the day-cell hover tint: it would widen calendar.html's diff from upstream for no success criterion"

patterns-established:
  - "EventFormHeaderMatchesUpstreamTest: every differing region must contain an anchor, every anchor must occur in a differing region and in its own header item"

requirements-completed: []

coverage:
  - id: D1
    description: "An anonymous month view has no /calendar/create/ URL and no '+ New Event'; it keeps each event's update hx-get and the Bootstrap 5 modal handler; a signed-in user keeps both create targets"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#CalendarMonthViewReadOnlyTest"
        status: pass
    human_judgment: false
  - id: D2
    description: "An anonymous pop-up is a read-only card (no form control, no write URL, labelled fields, empty rows omitted, escaped markup, read-only todos, attributed-run block once, no login prompt); a signed-in pop-up is the unchanged editable form"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventModalReadOnlyCardTest, EventCardUrlLinkTest, EventFormUrlLinkTest"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two anonymous reads write nothing; an editor's render and a visitor's render share no output"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventModalReadOnlyCardTest.test_anonymous_reads_write_nothing, test_editor_then_anonymous_render_share_no_output"
        status: pass
    human_judgment: false
  - id: D4
    description: "event_form.html's header names tomtoolkit 3.1.0 and lists the six differing blocks; a diff against the installed upstream file matches the list"
    requirement: WARN-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventFormHeaderMatchesUpstreamTest"
        status: pass
    human_judgment: false
  - id: D5
    description: "calendar.html carries the Bootstrap 5 utility names and none of the six Bootstrap 4 ones (D-11)"
    requirement: WARN-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#CalendarTemplateBootstrap5ClassTest"
        status: pass
    human_judgment: false
  - id: D6
    description: "In Chromium an anonymous visitor sees no create target, opens the pop-up through the Bootstrap 5 API as a read-only card with the attributed-run block and no form control, and an empty day cell is inert; a logged-in user opens the modal from both create targets and sees a form"
    requirement: ACCESS-02
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering (seven calendar/attribution tests)"
        status: pass
    human_judgment: false
  - id: D7
    description: "The runbook says the calendar is read-only when not logged in"
    requirement: ACCESS-02
    verification: []
    human_judgment: true
    rationale: "Prose accuracy of the paired runbook paragraph is not asserted by any test; a reviewer should read it"

duration: 36min
completed: 2026-10-08
status: complete
---

# Phase 39 Plan 02: Read-only Calendar for Visitors Summary

**An anonymous visitor now gets a month view with no create click target and a plain-text event card (fields, read-only todos, attributed-run block) in place of the editable form, while event_form.html's header lists the six blocks that really differ from tomtoolkit 3.1.0 and a test re-diffs it against the installed upstream file**

## Performance

- **Duration:** 36 min of execution (plus three pre-commit django-test hook runs, roughly 10 to 15 minutes each, included)
- **Started:** 2026-10-08T15:18Z
- **Completed:** 2026-10-08T15:55Z
- **Tasks:** 3
- **Files modified:** 7 (5 code, test or docs, 2 RED evidence records)

## Accomplishments

- ACCESS-02: a visitor who is not logged in sees no `/calendar/create/` URL, no "+ New Event" text and no day-cell hx-* attributes in the month view; clicking an event still opens the pop-up (proved in Chromium, which settles RESEARCH A2: the inner container's handler fires with the day cell's own hx-* attributes gone).
- The pop-up for a visitor is `<div id="cal-event-card">`: Title, Start, End always, then Description, URL, Target list, User, Proposal, Telescope, Instrument only when set. No form, input, select, textarea or button; none of the delete, update or todo URLs; ALLOC:/RUN: keys and `javascript:` values are never echoed; markup in titles and descriptions is escaped.
- The Attributed campaign run block and its tally render once for both audiences; the Observation series block and staff candidate hint are untouched, so visitors still never see the group name or a candidate run.
- WARN-01: the header names tomtoolkit 3.1.0 and the installed upstream path, and lists six numbered items; the stale "exact copy", "one new block", 3.0.1 and 3.0.0a9 text is gone.
- D-11: calendar.html uses border-start, border-end, fw-bold, me-2, me-3 and var(--bs-white); `data-url` kept.

## RED / GREEN evidence

**Task 1 (tracer)**
- RED log: `Ran 90 tests`, `FAILED (failures=10)`, all FAIL headers and no ERROR. Target `test_anonymous_card_has_no_form_controls` failed on `AssertionError: '<form' unexpectedly found` (the anonymous GET rendered the whole editable form). The other nine were `test_anonymous_month_view_has_no_create_target`, three `EventCardUrlLinkTest` cases (non-web value echoed), `test_anonymous_card_escapes_markup`, `test_anonymous_card_lists_todos_read_only`, `test_anonymous_card_omits_empty_fields`, `test_anonymous_card_shows_every_field_as_text` and `test_editor_then_anonymous_render_share_no_output`. Every pre-existing test, including `EventFormUrlLinkTest` for a signed-in user, passed.
- Classifier (`gsd-tools check tdd-red-evidence`): `RED_EVIDENCE_OK`, reason `target_test_failed`.
- GREEN: `Ran 90 tests in 8.472s ... OK`. Structural check printed `OK: two branches per template, shared blocks once, event rows intact, no new override, runbook paragraph present`.
- Tracer gate: the tracer's `<verify>` is automated-only, so it was re-run end to end (pass); nothing expanded on a broken foundation.
- Commit `8d0eca5`.

**Task 2**
- RED log: `Ran 4 tests`, `FAILED (failures=4)`. Target `test_header_names_the_pinned_upstream` failed on `'tomtoolkit 3.1.0' not found` in the old 3.0.1 header; the numbering, region/item and Bootstrap 5 class tests failed too. The first RED run had one ERROR (a KeyError on a missing item number); I made `_item_texts` assert the 1-to-6 numbering first so it is a proper FAIL.
- Classifier: `RED_EVIDENCE_OK`, reason `target_test_failed`.
- GREEN: `Ran 94 tests in 8.633s ... OK`.
- Commit `9b2df4d`.

**Task 3**
- Functional run of the seven named tests: `Ran 7 tests in 5.937s ... OK`. The whole `solsys_code.tests.test_bootstrap5_rendering` module: `Ran 13 tests in 8.138s ... OK` (the known-flaky observatory-create test passed too).
- Commit `99b60b1`.

All three commits ran the full pre-commit hook (ruff, ruff-format, django-test), no SKIP and no `--no-verify`. Per the plan, tests, implementation and the RED record of each task landed in one commit (a failing-test commit would fail the django-test hook). Nothing pushed.

## event_form.html differing regions matched to header items

`difflib` regions of the 3.1.0 upstream file against the FOMO body (as printed by the Task 2 verify block), with the header item each belongs to:

| Region (upstream lines -> FOMO body lines) | Header item |
|---|---|
| replace 1 -> 1-2 (load line, plus the opening `request.user.is_authenticated` line) | 1 (load line), 5 (opening of the form/card branch) |
| replace 31 -> 32, replace 33 -> 34, insert -> 36-37 (the URL label: `is_web_url`, `rel="noopener noreferrer"`, `(not a web link)`) | 2 |
| replace 65 -> 68, 67 -> 70, 69-72 -> 72-76 (plain `<button>` Save, Save and edit, Delete) | 3 |
| insert -> 79-275 (`{% else %}`, the cal-event-card, then the series, campaign and staff-hint blocks with their comments) | 5 (card) and 4 (decoration blocks) |
| insert -> 279 (todo area `request.user.is_authenticated`) and insert -> 285-297 (`{% else %}` read-only list from `event.todos.all`) | 6 |

`diff -u` hunk headers for event_form.html against upstream: `@@ -1,4 +1,39 @@`, `@@ -28,10 +63,12 @@`, `@@ -62,22 +99,234 @@`. The calendar.html `diff -u` against upstream shows no remaining hunk about the Bootstrap utility names or the white variable (only `me-3` additions in FOMO-only legend lines and unchanged context lines for the border classes).

## Task Commits

1. **Task 1 (tracer): read-only card, month view without create targets, runbook paragraph** - `8d0eca5` (feat)
2. **Task 2: six-item header checked against upstream; Bootstrap 5 names** - `9b2df4d` (docs)
3. **Task 3: browser proof** - `99b60b1` (test)

**Plan metadata:** recorded in the docs(39-02) commit that follows this SUMMARY.

## Files Created/Modified

- `src/templates/tom_calendar/partials/event_form.html` - two `request.user.is_authenticated` branches (form vs card, todos include vs read-only list), the rewritten six-item header.
- `src/templates/tom_calendar/partials/calendar.html` - create targets for signed-in users only, `cal-header-spacer`, Bootstrap 5 names.
- `solsys_code/tests/test_calendar_template.py` - `CalendarMonthViewReadOnlyTest`, `EventModalReadOnlyCardTest`, `EventCardUrlLinkTest`, `EventFormHeaderMatchesUpstreamTest`, `CalendarTemplateBootstrap5ClassTest`; `EventFormUrlLinkTest` logs an editor in.
- `solsys_code/tests/test_bootstrap5_rendering.py` - `_calendar_editor`, `_log_in_browser`, two anonymous tests, two signed-in tests re-pointed.
- `docs/runbooks/telescope_runs_calendar.rst` - the "Not logged in, the calendar is read-only." paragraph (inserted before "Before chasing a missing attribution") and the "or -- when logged in -- on a day cell or the "+ New Event" button," wording.
- `.planning/phases/39-calendar-write-access/39-02-red-evidence-task1.json`, `...-task2.json` - classified RED evidence.

## Decisions Made

- Followed the plan's discretion choices (see key-decisions in the frontmatter). No view flag, no third template override.
- Requirements ACCESS-02 and WARN-01 are not marked complete here: plan 39-03 declares both too, so 39-03 closes them (same rule 39-01 followed for ACCESS-01).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test fixture attribute named `run` shadowed `TestCase.run`**
- **Found during:** Task 1 (first RED run)
- **Issue:** `cls.run = CampaignRun...` made every test in the class raise `TypeError: 'CampaignRun' object is not callable`.
- **Fix:** Renamed the fixture attribute to `card_run`.
- **Files modified:** `solsys_code/tests/test_calendar_template.py`
- **Commit:** `8d0eca5`

**2. [Rule 1 - Bug] Task 2 numbering/region test errored instead of failing in RED**
- **Found during:** Task 2 (RED run)
- **Issue:** `_item_texts` raised `KeyError` when the old header had no numbered items, giving an ERROR rather than a clean FAIL.
- **Fix:** Added an `assertEqual(sorted(starts), [1..6])` first.
- **Commit:** `9b2df4d`

**3. [Rule 3 - Blocking] Single-line runbook phrase**
- **Found during:** Task 1 (runbook edit)
- **Issue:** The plan's structural check needs `or -- when logged in -- on a day cell or the "+ New Event" button` on one line, so the surrounding paragraph was reflowed slightly beyond the file's usual ~75 columns for that line.
- **Commit:** `8d0eca5`

**4. [Addition] Browser tests wait for the form to attach**
- The two signed-in tests call `wait_for(state='attached')` on `#cal-modal-body form` before counting it, to avoid a race with the htmx swap. No plan criterion changes.
- **Commit:** `99b60b1`

---

**Total deviations:** 4 (2 test-bug fixes, 1 formatting constraint, 1 harmless wait). **Impact:** none on behavior or scope.

## Paired docs

The runbook page was updated in Task 1 (same commit as the visitor-visible template change). No notebook changed: `campaign_lifecycle_demo.ipynb`'s pop-up cells assert only on decoration text, which the card still renders, so its committed output stays accurate.

## Issues Encountered

- `Monitor` was not used; background commits were polled with an until-loop. Each django-test hook took about 10 to 15 minutes.
- Unrelated working-tree changes (`.planning/config.json`, `.planning/state.json`, `.planning/ui-reviews/.gitignore`, untracked `reqgroup_2682493.json`, `.planning/agent-history.json`, `.planning/milestone.lock`) were left exactly as found and survived pre-commit's stash and restore.

## Known Stubs

None.

## Threat Flags

None. The change removes controls from a public surface; it adds no endpoint, auth path or schema. T-39-11 to T-39-16 are mitigated by the tests named in the coverage block.

## User Setup Required

None.

## Next Phase Readiness

- 39-03 can rely on the read-only card, the hidden create targets and the six-item header; it owns closing ACCESS-02 and WARN-01 in REQUIREMENTS.md.
- A future tomtoolkit upgrade that changes the upstream event_form.html will fail `EventFormHeaderMatchesUpstreamTest` until the header is re-diffed (intended, T-27-20).

## TDD Gate Compliance

The plan's tasks are tdd="true" with test, implementation and RED record in one commit by design (a failing-test commit would fail the django-test hook). The RED gate is evidenced by two classified RED records (`RED_EVIDENCE_OK`) produced before any implementation. There is no separate `test(39-02)` RED commit; the `test(39-02)` commit is the browser-test task.

## Self-Check: PASSED

- Files found: both templates, both test modules, the runbook, both RED evidence JSON files.
- Commits found: `8d0eca5`, `9b2df4d`, `99b60b1` (all ancestors of HEAD; `git rev-list --count b261c9c..HEAD` = 3).

---
*Phase: 39-calendar-write-access*
*Completed: 2026-10-08*
