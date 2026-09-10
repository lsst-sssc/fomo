---
phase: 33-series-identity-reconciler-inversion
plan: 11
subsystem: campaign-coordination
tags: [django, bootstrap5, htmx, playwright, calendar, jquery-removal]

# Dependency graph
requires:
  - phase: 33 (plan 09)
    provides: the fixture receipt (65 surviving campaign-attributed CalendarEventMeta
      rows, July 2025 - July 2026) this plan's Task 2 human-check relies on, and
      ordering that avoids a race on src/fomo/settings.py during full-suite test runs
provides:
  - a Bootstrap-5-native calendar pop-up open path (calendar.html), closing UAT
    gap G-33-2
  - a browser-driven regression test suite proving the pop-up opens with no page
    error, for an attributed entry, an unattributed entry, the '+ New Event'
    button, and an empty day cell
  - a server-rendered guard pinning the served month partial to contain no
    jQuery-style selector call and to contain the Bootstrap 5 modal call
  - a defensive fix in campaign_decoration() so the create-event form path never
    raises AttributeError
  - the runbook's calendar pop-up section corrected to describe the Bootstrap 5
    open path and to distinguish it from a missing-attribution symptom
affects: [33-10, end-of-phase UAT (queued human browser re-check)]

# Actuals (#2632)
actuals:
  tokens: 4012
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Modal-opening handlers on tomtoolkit 3.x pages must call
      bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show()
      directly (not a jQuery selector, not the upstream showModal() indirection) --
      the tomtoolkit 3.x base loads Bootstrap 5.3.3, htmx and Alpine and no jQuery,
      and an htmx-swapped partial cannot rely on a script element in its own
      fragment having executed."
    - "Template tags documented as 'never raises' must guard against being called
      with Django's invalid-variable placeholder (an undefined context variable
      resolves to the empty string, not an exception) -- isinstance-check the
      expected model type before touching its attributes."

key-files:
  created: []
  modified:
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_bootstrap5_rendering.py
    - solsys_code/tests/test_calendar_template.py
    - docs/runbooks/telescope_runs_calendar.rst
    - solsys_code/templatetags/calendar_display_extras.py

key-decisions:
  - "Fixed the handler body inline as bootstrap.Modal.getOrCreateInstance(...).show()
    rather than upstream's showModal() function indirection, per the plan's explicit
    instruction: the partial is itself swapped by htmx, so an inline call has no
    dependency on a script element in the swapped fragment having executed."
  - "Built the browser-modal-open fixture (CampaignRun/CalendarEvent/CalendarEventMeta)
    in TestBootstrap5Rendering.setUp(), not setUpTestData -- StaticLiveServerTestCase is
    a TransactionTestCase subclass, which does not support setUpTestData's class-scoped,
    transaction-rolled-back fixture semantics; each TransactionTestCase test flushes the
    database in its own teardown, so per-class data would only reliably survive for
    whichever test happened to run first."
  - "Guarded campaign_decoration() with an isinstance(event, CalendarEvent) check
    rather than wrapping the {% campaign_decoration event %} call site in
    {% if event %} in event_form.html -- centralizes the function's own documented
    'never raises' contract in the function itself, rather than requiring every
    future template call site to remember the guard."

requirements-completed: [ANNOT-02]

coverage:
  - id: D1
    description: "A real headless browser opens the calendar pop-up via the Bootstrap 5
      modal API for a campaign-attributed entry (with the 'Attributed campaign run'
      block and 'View campaign' link), an unattributed entry, the '+ New Event' button,
      and the empty area of a day cell -- zero page errors in every case (UAT G-33-2,
      must_haves Truth 3)"
    requirement: "ANNOT-02"
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_calendar_modal_opens_for_campaign_attributed_event_with_no_page_errors"
        status: pass
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_calendar_modal_opens_for_new_event_button_with_no_page_errors"
        status: pass
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_calendar_modal_opens_for_unattributed_event_with_no_page_errors"
        status: pass
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_calendar_modal_opens_for_empty_day_cell_with_no_page_errors"
        status: pass
      - kind: other
        ref: "sensitivity check: template reverted to $('#cal-modal').modal('show') -> attributed-click test fails with playwright._impl._errors.TimeoutError: Locator.wait_for: Timeout 30000ms exceeded, waiting for locator(\"#cal-modal.show\") to be visible; template restored immediately after"
        status: pass
    human_judgment: false
  - id: D2
    description: "The served month partial contains no jQuery-style selector call
      ('$(') and does contain bootstrap.Modal.getOrCreateInstance -- a server-rendered
      regression guard independent of the browser test"
    requirement: "ANNOT-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#CalendarModalOpenerRenderTest.test_calendar_partial_contains_no_jquery_selector_call"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#CalendarModalOpenerRenderTest.test_calendar_partial_opens_modal_via_bootstrap5_api"
        status: pass
      - kind: other
        ref: "sensitivity check: template reverted to jQuery handler -> no-jQuery test fails with AssertionError: '$(' unexpectedly found in ... (71 pre-fix occurrences); template restored immediately after"
        status: pass
    human_judgment: false
  - id: D3
    description: "The runbook's calendar pop-up section documents the Bootstrap 5 open
      path and distinguishes it from the missing-attribution symptom; plan 33-10's
      three sections are left byte-identical"
    requirement: "ANNOT-02"
    verification:
      - kind: other
        ref: "grep -c 'Bootstrap 5' docs/runbooks/telescope_runs_calendar.rst == 2 (0 before); grep -c 're-confirm or discard' == 2 (unchanged); git diff shows added lines only"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files"
        status: pass
    human_judgment: false
  - id: D4
    description: "End-of-phase human browser re-verification of UAT test 2: click an
      attributed entry, follow 'View campaign |->', confirm the campaign table lands
      scrolled to and highlighting that run's own row"
    requirement: "ANNOT-02"
    verification: []
    human_judgment: true
    rationale: "Real-browser anchor-scroll and :target CSS highlight rendering is
      not observable by a server-side or headless-assertion test; queued for the
      end-of-phase human_verify_mode=end-of-phase UAT consolidation. Fixture: 65
      surviving campaign-attributed CalendarEventMeta rows recorded in 33-09-SUMMARY.md
      (July 2025 - July 2026) -- no seeding needed for this check."

# Metrics
duration: ~30 min
completed: 2026-09-10
status: complete
---

# Phase 33 Plan 11: Calendar Pop-up Bootstrap 5 Migration Summary

**FOMO's calendar month partial now opens `#cal-modal` via `bootstrap.Modal.getOrCreateInstance(...).show()` on all three click targets instead of a dead `$('#cal-modal').modal('show')` jQuery call, proven by a real headless-browser click with zero page errors, a server-rendered no-jQuery guard, and a corrected runbook section (UAT G-33-2).**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-09-10T04:20:00Z (approx.)
- **Completed:** 2026-09-10T05:08:00Z (approx.)
- **Tasks:** 3
- **Files modified:** 5 (4 in `files_modified` + 1 out-of-scope deviation fix)

## Accomplishments

- `src/templates/tom_calendar/partials/calendar.html`'s three `hx-on::after-request`
  handlers ('+ New Event' button, `.cal-day` div, inner event-container div) now call
  `bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show();`
  — the same fixed literal on all three, matching upstream tom_calendar 3.0.1's
  Bootstrap-5-native form but inlined (not the `showModal()` indirection) since the
  partial is itself htmx-swapped.
- Four new Playwright browser tests in `test_bootstrap5_rendering.py` prove the pop-up
  opens with zero page errors for a campaign-attributed entry (with the 'Attributed
  campaign run' block and 'View campaign' link), an unattributed entry, the
  '+ New Event' button, and an empty day cell.
- Two new server-rendered tests in `test_calendar_template.py`
  (`CalendarModalOpenerRenderTest`) pin the served month partial to contain zero
  `$(` occurrences and at least one `bootstrap.Modal.getOrCreateInstance` occurrence.
- Both revert-and-restore sensitivity checks performed during execution (see
  "Sensitivity Check Evidence" below), confirming both new guards actually exercise
  the fix rather than passing vacuously.
- The runbook's calendar pop-up section (`docs/runbooks/telescope_runs_calendar.rst`)
  now tells an operator that the pop-up opens via the Bootstrap 5 modal API and how
  to tell a client-side JavaScript fault (pop-up never opens at all) apart from a
  missing attribution (pop-up opens, no Attributed campaign run block).

## Task Commits

1. **Task 1: Open the pop-up with the Bootstrap 5 API, proven by a real browser click** - `ee9957a` (fix)
2. **Task 2: Server-rendered guard that the jQuery modal call cannot return** - `861a4ea` (test)
3. **Task 3: Paired docs — the runbook's calendar pop-up section matches the restored behaviour** - `80605f6` (docs)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Sensitivity Check Evidence

Both new guards were verified to actually fail against the pre-fix jQuery template,
then the template was restored immediately after observing the failure:

- **Task 1 (browser test):** with `calendar.html` temporarily reverted to
  `$('#cal-modal').modal('show');`, `test_calendar_modal_opens_for_campaign_attributed_event_with_no_page_errors`
  failed with:
  ```
  playwright._impl._errors.TimeoutError: Locator.wait_for: Timeout 30000ms exceeded.
  Call log:
    - waiting for locator("#cal-modal.show") to be visible
  ```
  (The jQuery handler throws a `ReferenceError` for the undefined `$` global; the
  modal's `show` class never appears, so the wait times out.)

- **Task 2 (server-rendered guard):** with the same revert,
  `test_calendar_partial_contains_no_jquery_selector_call` failed with:
  ```
  AssertionError: '$(' unexpectedly found in '...'
  ```
  — the pre-fix rendered page contains 71 occurrences of the jQuery call (one per
  day-cell click target plus the '+ New Event' button and the inner event-container
  div, repeated across all rendered day cells in the month grid).

## Files Created/Modified

- `src/templates/tom_calendar/partials/calendar.html` - three modal-opening handlers migrated from jQuery to the Bootstrap 5 API; one single-line template comment added
- `solsys_code/tests/test_bootstrap5_rendering.py` - four new browser-driven modal-open tests plus a per-test fixture in `setUp()`
- `solsys_code/tests/test_calendar_template.py` - new `CalendarModalOpenerRenderTest` class (two tests) pinning the served partial's rendered output
- `docs/runbooks/telescope_runs_calendar.rst` - one new paragraph in the calendar pop-up section
- `solsys_code/templatetags/calendar_display_extras.py` - `campaign_decoration()` guarded against a non-`CalendarEvent` argument (deviation, see below)

## Decisions Made

See `key-decisions` in the frontmatter above.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `campaign_decoration()` raised `AttributeError` on the create-event form path**

- **Found during:** Task 1, while implementing the '+ New Event' button browser test
- **Issue:** `event_form.html` is rendered for both the create and update actions of
  `tom_calendar`'s event form. The create action's view (`tom_calendar.views.create_event`)
  never puts `event` in the template context. Django resolves the missing `event`
  variable to the empty-string invalid-variable placeholder rather than raising, so
  `{% campaign_decoration event as deco %}` was invoked with `event=''`. Inside
  `campaign_decoration()`, `event.telescope_label_meta` then raised
  `AttributeError: 'str' object has no attribute 'telescope_label_meta'` — an
  uncaught exception (the function only catches `ObjectDoesNotExist`), producing a
  500 on every "+ New Event" click. This predates this plan; no existing test
  exercised the create-event GET path through a real request before this plan's new
  browser test did.
- **Fix:** Added `if not isinstance(event, CalendarEvent): return None` at the top of
  `campaign_decoration()`, before the `.telescope_label_meta` access — matching the
  function's own docstring contract ("Never raises").
- **Files modified:** `solsys_code/templatetags/calendar_display_extras.py`
- **Verification:** `test_calendar_modal_opens_for_new_event_button_with_no_page_errors`
  passes; full `test_bootstrap5_rendering.TestBootstrap5Rendering` (7 tests) and
  `test_calendar_template` (52 tests) both green; full project suite (1011 tests) green.
- **Committed in:** `ee9957a` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug blocking the plan's own required '+ New Event' browser test).
**Impact on plan:** No scope creep. The fix is a minimal defensive guard matching the
function's existing documented contract, discovered only because this plan's new
browser test was the first to exercise the create-event GET path end-to-end. Without
it, Task 1's required "+ New Event button opens the same modal" test could not pass.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required.

## Requirements Status

This plan carries `requirements: [ANNOT-02]`. ANNOT-02 is also declared by sibling
plans in this phase: 33-01, 33-02, 33-05, 33-06, and 33-09, all of which already have
their own `*-SUMMARY.md`. This plan is the last plan in the phase declaring ANNOT-02
without a completed sibling, so per the shared-ID gate (issue #2388) ANNOT-02 becomes
markable complete once this SUMMARY is written — handled via `requirements
mark-complete` in the state-update step below.

## Next Phase Readiness

- Plan 33-10 (wave 3, `depends_on` this plan) can now build against a tree that
  already contains this plan's runbook paragraph; it owns the re-classification
  section, the `reconcile_campaign_runs` counter documentation, and the skip-rule
  section — all three left byte-identical by this plan.
- The end-of-phase UAT consolidation should re-run UAT test 2 in a real browser
  (D4 in the coverage table above): open the month named in 33-09-SUMMARY.md's fixture
  receipt (any of the 65 surviving campaign-attributed rows, July 2025 - July 2026),
  click an attributed entry, follow 'View campaign |->', and confirm the campaign
  table lands scrolled to and highlighting that run's own row.
- No blockers.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-10*

## Self-Check: PASSED

- FOUND: src/templates/tom_calendar/partials/calendar.html
- FOUND: solsys_code/tests/test_bootstrap5_rendering.py
- FOUND: solsys_code/tests/test_calendar_template.py
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND: solsys_code/templatetags/calendar_display_extras.py
- FOUND: commit ee9957a (Task 1)
- FOUND: commit 861a4ea (Task 2)
- FOUND: commit 80605f6 (Task 3)
