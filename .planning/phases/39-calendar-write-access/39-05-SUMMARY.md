---
phase: 39-calendar-write-access
plan: 05
subsystem: calendar
tags: [django, tom_calendar, template-tags, htmx, gap-closure, runbook]
gap_closure: true
gap_ids: [G-39-3, G-39-4]

requires:
  - phase: 39-calendar-write-access
    provides: write_requires_login gating, the event_form.html override and its header/pinned snapshot (39-01..39-04)
provides:
  - high_band_attribution_candidates guarded against a non-CalendarEvent value (never-raises docstring now true)
  - staff attribution hint gated on the edit form, so staff and superusers get the New Event create form again
  - six regression tests covering staff, superuser, plain-user equivalence, invalid create POST, tag guard, edit-form gate
  - runbook warning that the post-CSRF-login create and edit addresses are a bare copy of the event form
affects: [41-todo-triage, verify-work UAT Tests 3 and 4]

actuals:
  tokens: 27000
  tasks: 2
  commits: 2
plan_head_before: e8839e31bf6c68b86d7ab686ed2dd76ffa0bc087
plan_head_after: c71b7b0e071a814adbe627a498f2eed829108512
commits: 2

tech-stack:
  added: []
  patterns:
    - "isinstance(event, CalendarEvent) guard as the first statement of every template tag that receives a template variable that may resolve to the empty-string placeholder"
    - "action-first condition in event_form.html so a branch needing `event` is never evaluated on the create form"

key-files:
  created:
    - .planning/phases/39-calendar-write-access/39-05-red-evidence-task1.json
  modified:
    - solsys_code/templatetags/attribution_display_extras.py
    - src/templates/tom_calendar/partials/event_form.html
    - solsys_code/tests/test_calendar_template.py
    - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Fixed G-39-4 minimally: tag guard (a) plus action-first elif (b); calendar.html modal-on-any-response hardening and a guard in campaign_attribution.candidates_for_event were not planned and not done"
  - "Header item 4 and the pinned snapshot were updated in the same commit as the template change (D-10, WARN-01)"
  - "G-39-3 is runbook-only: the template method=post hardening (39-REVIEW WR-01 item 2) was offered at UAT and not requested"

patterns-established:
  - "Staff-visible regression tests use Client(raise_request_exception=False) so a server 500 is a FAIL on a status assertion, not an ERROR"

requirements-completed: [ACCESS-01, ACCESS-02, WARN-01]

coverage:
  - id: D1
    description: "A staff or superuser GET of /calendar/create/ (with and without ?date=, htmx or not) returns 200 with the create form and no hint"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_staff_and_superuser_get_the_create_form"
        status: pass
    human_judgment: false
  - id: D2
    description: "The staff create form is identical to a plain signed-in user's once the CSRF token is masked"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_staff_create_form_matches_the_plain_users_apart_from_the_csrf_token"
        status: pass
    human_judgment: false
  - id: D3
    description: "A staff htmx POST of an invalid create form re-renders the form into the pop-up and creates nothing"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_staff_invalid_create_post_re_renders_the_form"
        status: pass
    human_judgment: false
  - id: D4
    description: "high_band_attribution_candidates returns [] for '' and None and never raises"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_attribution_tag_returns_empty_list_for_a_non_event"
        status: pass
    human_judgment: false
  - id: D5
    description: "The staff hint shows on the edit form (staff and superuser) and never on the create form; non-staff and anonymous still never see it"
    requirement: ACCESS-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_hint_is_gated_on_the_edit_form"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#test_superuser_sees_high_band_hint_for_unlinked_event"
        status: pass
    human_judgment: false
  - id: D6
    description: "Header item 4 and the pinned 10-region snapshot match the template (WARN-01, D-10)"
    requirement: WARN-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventFormHeaderMatchesUpstreamTest.test_body_diff_matches_pinned_snapshot"
        status: pass
    human_judgment: false
  - id: D7
    description: "The runbook tells operators the post-CSRF-login create and edit addresses are a bare, unstyled copy of the event form, not to use it, and to go back to the calendar page"
    requirement: ACCESS-01
    verification:
      - kind: other
        ref: "runbook paragraph check script in 39-05-PLAN.md Task 2 verify (OK: runbook warns ...)"
        status: pass
    human_judgment: true
    rationale: "Operator-facing wording; the developer asked for this exact sentence at UAT, but a reader should confirm it reads clearly"
  - id: D8
    description: "A staff or superuser clicking + New Event or an empty day cell in a real browser sees the create form in the pop-up"
    requirement: ACCESS-02
    verification: []
    human_judgment: true
    rationale: "UAT Test 4 re-run as a staff account on the dev server is a human-check at /gsd-verify-work; the non-staff browser test already passes and the staff form is byte-identical to it apart from the CSRF token"

duration: 30min
completed: 2026-10-08
status: complete
---

# Phase 39 Plan 05: Staff New Event pop-up fix and runbook bare-form warning Summary

**Staff and superusers get the New Event create form again (isinstance guard on `high_band_attribution_candidates` plus an action-first gate on the staff hint in `event_form.html`), with header item 4 and the pinned snapshot kept in step, and the runbook now warns that the post-CSRF-login create/edit address is a bare form that saves nothing.**

## Performance

- **Duration:** 30 min
- **Started:** 2026-10-08T22:25:11Z
- **Completed:** 2026-10-08T22:55:04Z
- **Tasks:** 2
- **Files modified:** 6 (5 modified, 1 created)

## Accomplishments

- G-39-4 closed: a staff or superuser GET of `/calendar/create/` (with or without `?date=`, htmx or not) and an invalid create POST return 200 with the form instead of 500 (which htmx did not swap, leaving an empty pop-up shell). Latent since 27-07, not a Phase 39 regression.
- The tag's "Never raises" docstring is now true; the staff "Possible campaign run match" hint is only considered on the edit form, where it still shows for staff and superusers and never for anyone else.
- Header item 4 of `event_form.html` says the hint is edit-form-only, and the pinned snapshot is regenerated (still exactly 10 regions) in the same commit.
- G-39-3 closed: the runbook paragraph "A write attempt while logged out changes nothing" now says the create and edit addresses show a bare, unstyled copy of the event form and tells the operator not to use it.

## Task Commits

1. **Task 1 (tracer): staff/superuser create form, tag guard, hint gate, header item 4, snapshot** - `917a895` (fix)
2. **Task 2: runbook bare-form warning (G-39-3)** - `c71b7b0` (docs)

**Plan metadata:** the docs(39-05) closeout commit holding this SUMMARY, STATE.md, ROADMAP.md and REQUIREMENTS.md.

## Task 1 evidence

RED (`EventModalAttributionHintTest`, `Ran 12 tests`, `FAILED (failures=13)`, FAIL headers only, no ERROR):

- `test_staff_and_superuser_get_the_create_form`: 8 subTests, each `AssertionError: 500 != 200`
- `test_staff_create_form_matches_the_plain_users_apart_from_the_csrf_token`: `500 != 200`
- `test_staff_invalid_create_post_re_renders_the_form`: `500 != 200`
- `test_attribution_tag_returns_empty_list_for_a_non_event`: `ValueError: Field 'id' expected a number but got ''` and `AttributeError: 'NoneType' object has no attribute 'start_time'`, both reported via `self.fail`
- `test_hint_is_gated_on_the_edit_form`: create subTest `True != False`
- Classifier verdict: `RED_EVIDENCE_OK` (`target_test_failed`); record at `.planning/phases/39-calendar-write-access/39-05-red-evidence-task1.json`.

GREEN: `python manage.py test --noinput solsys_code.tests.test_calendar_template` -> `Ran 103 tests ... OK`. Functional: `Ran 2 tests ... OK` (`test_signed_in_editor_creates_edits_and_deletes_from_month_view`, `test_calendar_modal_opens_for_new_event_button_with_no_page_errors`).

Snapshot `git diff -U0` (regenerated with the class's own `current_diff()`; still exactly 10 `@@ ` lines): only the hint elif line changed to the gated form, three added comment lines (the G-39-4 reason sentence), and renumbered `@@ insert upstream ... fomo-body` headers for the regions at or after it (79-275 -> 79-278, 279-279 -> 282-282, 285-297 -> 288-300).

Structural check output: `OK: tag guarded and nothing else changed there; elif gated on the edit form; header item 4 and the 10-region snapshot in step; form block and other files untouched; existing tests intact`.

## Task 2 evidence

Runbook check: `OK: runbook warns that the create and edit addresses show a bare copy of the event form; nothing else in docs/ changed`. The rewrapped closing clause:

> nothing -- the delete and todo addresses refuse a plain visit, and the create and edit addresses show a bare, unstyled copy of the event form. Do not use that copy: its Save saves nothing and silently discards what was typed; go back to the calendar page and make the change there.

Commit check: `OK: one runbook commit holding exactly the runbook`.

Phase gate on the final tree:

- Full suite: `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` -> `Ran 2247 tests in 471.394s`, `OK`, exit 0, no `FAILED`, no `skipped=` (functional tests included; the known flaky Playwright test did not fail).
- Ruff: `pre-commit run ruff` / `ruff-format --all-files` run twice, all Passed, no tracked file changed outside `.planning`.
- `python manage.py check`: only `urls.W005` (calendar namespace not unique).
- `python manage.py makemigrations --check --dry-run`: `No changes detected`.
- RECORD: `OK: 22 installed tom_calendar files match tomtoolkit 3.1.0 RECORD hashes`.

## Files Created/Modified

- `solsys_code/templatetags/attribution_display_extras.py` - `if not isinstance(event, CalendarEvent): return []` as first statement, docstring made true
- `src/templates/tom_calendar/partials/event_form.html` - staff hint elif is now `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}`, G-39-4 reason sentence in its comment, header item 4 says edit form only
- `solsys_code/tests/test_calendar_template.py` - two fixtures, two helpers, six tests on `EventModalAttributionHintTest`; no pre-existing test changed
- `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff` - regenerated pinned snapshot
- `.planning/phases/39-calendar-write-access/39-05-red-evidence-task1.json` - RED evidence
- `docs/runbooks/telescope_runs_calendar.rst` - closing clause of one paragraph

## Decisions Made

- Minimal G-39-4 fix per the developer's UAT scope: tag guard plus action-first elif. The elif tests `action == "update"` before `event` so Django's short-circuiting `and` never resolves `event` on the create form.
- Runbook-only G-39-3: no `method="post"` on the event form (offered in 39-REVIEW WR-01 item 2, not requested).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The full hook (ruff, ruff-format, django-test) passed on both code commits.

## Surfaced, not fixed

- `campaign_attribution.candidates_for_event` (campaign_attribution.py:644) also raises on the empty-string placeholder although its docstring says it never raises. After the tag guard the template can no longer hand it a non-event, and changing it would bring `campaign_lifecycle_demo.ipynb` into scope. Candidate for Phase 41's todo triage.
- `calendar.html` still opens the pop-up after any response, so a future 5xx from either create trigger would again look like an empty pop-up. Optional hardening offered in the G-39-4 diagnosis, not requested.
- T-39-27 (accepted): if an operator uses the bare copy anyway, its native-GET Save puts the CSRF token in the address bar and history.

## Not touched

- No notebook changed (none pairs with `attribution_display_extras.py`, the calendar templates or the calendar tests); G-39-4 needed no runbook change because the staff hint still shows on an existing unlinked entry's pop-up exactly as the runbook's 27-UAT.md Test 9 paragraph describes, and the create form never showed a hint (for staff it failed outright).
- `39-REVIEW-DISPOSITION.md` (WR-01 still open there), `39-UAT.md`, `39-VERIFICATION.md`, `39-SECURITY.md`, earlier plans, `calendar.html`, `calendar_urls.py`, `calendar_access.py`, `settings.py`, `urls.py` and the installed `tom_calendar` were not touched.

## Threat Flags

None - no new network endpoint, auth path or schema change; T-39-25, T-39-26, T-39-28 and T-39-29 are mitigated as planned, T-39-27 accepted.

## Next Phase Readiness

Ready for `/gsd-verify-work` to re-run UAT Tests 3 and 4. The Task 1 human-check applies: as a staff or superuser account on the dev server, "+ New Event" and an empty day cell open the create form, and an existing unlinked entry with a High-band candidate still shows the "Possible campaign run match" hint.

## Self-Check: PASSED

- Created/modified files exist; `917a895` and `c71b7b0` are ancestors of HEAD; `commits: 2` measured from the ledger base `e8839e3`.

---
*Phase: 39-calendar-write-access*
*Completed: 2026-10-08*
