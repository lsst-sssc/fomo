---
phase: 39-calendar-write-access
plan: 01
subsystem: auth
tags: [django, tom_calendar, access-control, htmx, csrf, url-conf]

requires:
  - phase: 33-calendar-modal-run-info
    provides: the FOMO-local calendar URL conf (solsys_code/calendar_urls.py) that shadows tom_calendar.urls; review WR-05 that flagged the unguarded writes
  - phase: 38-tomtoolkit-3-1-0-floor
    provides: tomtoolkit 3.1.0, whose tom_calendar views this plan wraps
provides:
  - solsys_code/calendar_access.py with write_requires_login and read_open_write_requires_login (each wrapper carries a calendar_guard marker)
  - all five calendar write routes guarded at FOMO's URL conf; delete-event, create-todo and update-todo POST-only
  - test_calendar_write_access.py (25 tests) proving ACCESS-01 per route, for htmx, signed-in users, CSRF and URL-conf shadowing
affects: [39-02 read-only card / hidden write targets, 39-03 browser proof, runbook sentence on logged-out writes]

actuals:
  tokens: 6000
  tasks: 2
  commits: 2
plan_head_before: f929f4ee7493d2b4b9ebaa9bfc8167133ab56b79
plan_head_after: 4d32e627c6ee11b43decbd1d5bde1e9841164e25
commits: 2

tech-stack:
  added: []
  patterns:
    - "Guard vendored views at FOMO's own URL conf with small stateless decorators; never edit or re-implement the upstream package"
    - "Guard outermost, require_POST inside, upstream callable innermost, so anonymous callers always get the login redirect and never a 405"
    - "Login next is the calendar page, not the refused URL, so a login cannot replay a GET-acting write"

key-files:
  created:
    - solsys_code/calendar_access.py
    - solsys_code/tests/test_calendar_write_access.py
    - .planning/phases/39-calendar-write-access/39-01-red-evidence-task1.json
    - .planning/phases/39-calendar-write-access/39-01-red-evidence-task2.json
  modified:
    - solsys_code/calendar_urls.py

key-decisions:
  - "Any logged-in user may write (D-01); no staff, permission or ownership check, so tightening later happens at the same wrapping point"
  - "require_POST added inside the guard on delete-event, create-todo and update-todo (OQ1): upstream deletes or blanks on a plain GET"
  - "Refused writes redirect to /accounts/login/?next=/calendar/ (OQ2); htmx gets 200 + HX-Redirect from tom_common middleware"
  - "A signed-in POST with no CSRF token ends as a 302 to login (next=/calendar/create/), not a 403, because tom_common's Raise403Middleware rewrites browser 403s; nothing is created either way"

patterns-established:
  - "AST allowlist check on the guard module (only wraps, redirect_to_login, reverse, the wrapped view): keeps upstream logic from being copied in"

requirements-completed: [ACCESS-01]

coverage:
  - id: D1
    description: "An anonymous POST to create-event, update-event and delete-event is a 302 to /accounts/login/?next=/calendar/ and changes no CalendarEvent field or count"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#AnonymousCalendarWriteTest (post_create_event, post_update_event, post_delete_event)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The two todo routes refuse anonymous POST and GET the same way; EventTodo unchanged; htmx gets HX-Redirect, never 403; missing id is a redirect, not 404"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#test_anonymous_post_create_todo_adds_nothing, test_anonymous_post_update_todo_changes_nothing, test_htmx_anonymous_writes_get_hx_redirect_never_403, test_anonymous_post_to_missing_event_is_redirected_not_404"
        status: pass
    human_judgment: false
  - id: D3
    description: "update-event GET and HEAD stay open (the pop-up) while PUT, PATCH, DELETE, OPTIONS and POST are refused"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#test_anonymous_get_and_head_update_event_stay_open, test_anonymous_other_methods_on_update_event_are_refused"
        status: pass
    human_judgment: false
  - id: D4
    description: "A plain signed-in user can still create, update and delete events and add and change todos; a signed-in GET on delete-event, create-todo and update-todo is 405 and changes nothing; CSRF still enforced"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#SignedInCalendarWriteTest"
        status: pass
    human_judgment: false
  - id: D5
    description: "FOMO's guarded URL conf wins over tom_common's unguarded copy of tom_calendar.urls for all five literal paths"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_write_access.py#CalendarUrlConfShadowingTest, test_literal_paths_are_guarded"
        status: pass
    human_judgment: false

duration: 34min
completed: 2026-10-08
status: complete
---

# Phase 39 Plan 01: Calendar Write Access Summary

**Two stateless login-guard decorators wrapped around tomtoolkit 3.1.0's five tom_calendar write views at FOMO's URL conf: anonymous writes become a login redirect that changes nothing, update-event's GET stays the open pop-up, and delete/todo routes are POST-only**

## Performance

- **Duration:** 34 min
- **Started:** 2026-10-08T14:42:08Z
- **Completed:** 2026-10-08T15:16:29Z
- **Tasks:** 2
- **Files modified:** 5 (3 code/test, 2 RED evidence records)

## Accomplishments

- ACCESS-01 closed: an anonymous request by any method can no longer create, change or delete a calendar event or todo through any of the five write routes; the refusal is a 302 to `/accounts/login/?next=/calendar/` (htmx: 200 + `HX-Redirect`), never a 403 or a 404 for a missing id.
- The upstream GET-acting views (delete-event, create-todo, update-todo) are POST-only now, so a crafted link can neither delete an event nor wipe a todo for a signed-in user, and a login cannot replay one.
- A plain non-staff signed-in user writes exactly as before (D-01, D-03); CSRF is still enforced.
- 25 tests cover the plan's truths, including exact per-field snapshots, idempotent repeats, literal paths and URL-conf shadowing.

## RED / GREEN evidence

**Task 1 (tracer, update-event)**
- RED log: `Ran 5 tests`, `FAILED (failures=6)`. FAIL headers: `test_anonymous_post_update_event_changes_nothing` (target, `AssertionError: 200 != 302`), `test_anonymous_repeat_post_update_event_is_refused_both_times`, and `test_anonymous_other_methods_on_update_event_are_refused` for put, patch, delete and options. GET/HEAD and signed-in tests passed.
- Classifier (`gsd-tools check tdd-red-evidence`): `RED_EVIDENCE_OK`, reason `target_test_failed`.
- GREEN: `Ran 5 tests ... OK`. AST check printed `OK: calendar_access.py has only the allowed calls, no mutable module state, docstrings; update-event wrapped`.
- Commit `5e31c0f`.

**Task 2 (other four routes)**
- RED log: `Ran 25 tests`, `FAILED (failures=25, errors=4)`. Target `test_get_delete_event_redirects_and_keeps_row` FAILed (`200 != 302`, the row was deleted by an anonymous GET). Other FAIL headers: anonymous POST create-event, delete-event, create-todo, update-todo; GET create-event and update-todo; missing id; repeat; literal paths; htmx create-todo/update-todo; signed-in 405; shadowing. The 4 ERRORs are the unguarded upstream views crashing (create_todo returns None on a GET; no HX-Redirect header on a non-redirect). Task 1 tests and signed-in write tests still passed.
- Classifier: `RED_EVIDENCE_OK`, reason `target_test_failed`.
- GREEN: `Ran 29 tests in 0.775s` (test_calendar_write_access + test_urls) `OK`.
- A3 caller check: `OK: five routes wrapped as planned; 4 template callers of the POST-only routes all use hx-post (A3)`.
- `python manage.py check`: only `?: (urls.W005) URL namespace 'calendar' isn't unique` (System check identified 1 issue).
- Commit `4d32e62`.

## Task Commits

1. **Task 1 (tracer): guard update-event end to end** - `5e31c0f` (feat)
2. **Task 2: guard the other four write routes, POST-only destructive routes** - `4d32e62` (feat)

Both ran the full pre-commit hook (ruff, ruff-format, django-test), no SKIP, no --no-verify. Per the plan, tests and implementation of each task landed in one commit (no separate RED commit, since a failing-test commit would fail the django-test hook); the RED evidence JSON rides in each commit. Nothing pushed.

**Plan metadata:** recorded in the docs(39-01) commit that follows this SUMMARY.

## Files Created/Modified

- `solsys_code/calendar_access.py` - `write_requires_login`, `read_open_write_requires_login`, `_login_redirect`; stateless, `calendar_guard` marker on each wrapper.
- `solsys_code/calendar_urls.py` - five write routes wrapped; docstring rewritten.
- `solsys_code/tests/test_calendar_write_access.py` - `AnonymousCalendarWriteTest`, `SignedInCalendarWriteTest`, `CalendarUrlConfShadowingTest`.
- `.planning/phases/39-calendar-write-access/39-01-red-evidence-task1.json`, `...-task2.json` - classified RED evidence.

## Decisions Made

- Followed the plan's dispositions: OQ1 (`require_POST` inside the guard) and OQ2 (`next` is the calendar page).
- No notebook or runbook changed in this plan; the runbook sentence about a logged-out write is 39-02's.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan assumption] CSRF refusal is a 302 to login, not a 403**
- **Found during:** Task 2 (RED run of `test_post_without_csrf_token_is_refused`)
- **Issue:** The plan said a signed-in POST without a CSRF token through `Client(enforce_csrf_checks=True)` returns 403. In FOMO's real middleware stack tom_common's `Raise403Middleware` rewrites every browser 403 into a redirect to `/accounts/login/?next=<refused path>`, so the response is 302 (Location `/accounts/login/?next=/calendar/create/`). This happens identically with the guard absent, so it is not caused by the guards.
- **Fix:** The test asserts the 302 and that exact Location (which also differs from the guard's `next=/calendar/`, so it proves the CSRF layer, not the guard, refused it) and that nothing was created. A control test (default client, no CSRF enforcement) shows the same POST creates the event.
- **Files modified:** `solsys_code/tests/test_calendar_write_access.py`
- **Committed in:** `4d32e62`

**2. [Rule 3 - Blocking] Task 1 commit imports only read_open_write_requires_login**
- **Found during:** Task 1 (ruff hook)
- **Issue:** The plan's acceptance criterion wanted the combined import of both decorators in Task 1, but only the read-open guard is used until Task 2, so ruff's F401 fix removed `write_requires_login` from the import.
- **Fix:** Left the ruff fix in place; Task 2 restored the combined import `from solsys_code.calendar_access import read_open_write_requires_login, write_requires_login`, which is the final state the key link expects.
- **Files modified:** `solsys_code/calendar_urls.py`
- **Committed in:** `5e31c0f`, then `4d32e62`

---

**Total deviations:** 2 auto-fixed (1 plan-assumption bug, 1 blocking lint)
**Impact on plan:** Neither changes behavior or scope; the CSRF behavior is a documented property of the existing middleware stack.

## Issues Encountered

- The pre-commit django-test hook takes about 15 minutes on this host; both commits ran it to completion.
- `Monitor` is unavailable here, so background commits were polled with a log-watching loop.

## Known Stubs

None.

## Threat Flags

None. The wrappers add no endpoint, auth path or schema; they only restrict existing routes. The CSRF-as-302 behavior comes from tom_common's existing middleware, not from this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 39-02 can rely on the anonymous GET of update-event returning 200 (the read-only card builds on it) and on the guarded write routes.
- Open assumption A1 (allauth honours `next=/calendar/` after login) was not exercised through a real login; the guard holds either way.

## TDD Gate Compliance

The plan was tdd="true" by task with test and implementation in one commit by design (a failing-test commit would fail the django-test hook). The RED gate is evidenced instead by the two classified RED records (`RED_EVIDENCE_OK`) produced before any guard code existed. There is no separate `test(39-01)` commit.

## Self-Check: PASSED

- Files found: `solsys_code/calendar_access.py`, `solsys_code/calendar_urls.py`, `solsys_code/tests/test_calendar_write_access.py`, both RED evidence JSON files.
- Commits found: `5e31c0f`, `4d32e62` (both ancestors of HEAD; `git rev-list --count f929f4e..HEAD` = 2).

---
*Phase: 39-calendar-write-access*
*Completed: 2026-10-08*
