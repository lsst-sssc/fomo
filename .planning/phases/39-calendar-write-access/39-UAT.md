---
status: testing
phase: 39-calendar-write-access
source: [39-VERIFICATION.md]
started: 2026-10-08T21:14:30Z
updated: 2026-10-08T23:21:54Z
---

## Current Test

number: 5
name: New Event pop-up as staff or superuser (re-run of Test 4 after gap closure 39-05)
expected: |
  On the dev server, signed in as a staff or superuser account, '+ New Event' and an empty day cell both open the pop-up with the create form (title, dates, Save and 'Save and Edit'), not an empty box; an existing unlinked entry with a High-band candidate still shows the 'Possible campaign run match' hint.
awaiting: user response

## Tests

### 1. Visitor affordance (judgment-tier prohibition, 39-02)
expected: Open /calendar/ logged out, click an entry and hover over the day cells. Nothing looks editable and nothing prompts a login. Decide whether the day-cell hover tint (.cal-day:hover) is acceptable for visitors or should be scoped to logged-in users. (Verifier's non-authoritative probe: the anonymous card has no form/input/select/textarea/hx-post/write URL; the month view's only control is the utc_offset display select.)
result: pass

### 2. Concurrency backstop truth (39-01 A12)
expected: The guard keeps no state between requests, so concurrent or interrupted anonymous requests are each refused independently. Accept the structural evidence (calendar_access.py code AST-identical to 798dfe9; module level holds only a tuple of string constants; no global/nonlocal) as sufficient, or ask for a concurrency test.
result: pass

### 3. Runbook paragraph readability (39-04 Task 1 human-check)
expected: docs/runbooks/telescope_runs_calendar.rst lines 2576-2605 read as plain operator guidance, name both refusal paths without jargon beyond "CSRF check", and state the self-registration acceptance with its quoted reason. Approved at the 39-04 Task 1 checkpoint on 2026-10-08; confirm, and decide at the same time whether to add the WR-01 sentence (the create and edit addresses show a bare copy of the form after login; go back to the calendar page instead) and the IN-02 note about the login-page flash message.
result: issue
reported: "pass, but add the WR-01 sentence"
severity: minor

### 4. New Event pop-up renders the create form (logged in)
expected: Clicking "+ New Event" (or an empty day cell) while logged in opens the event pop-up containing the create-event form (title, dates, Save / Save and Edit buttons).
result: issue
reported: "clicking on the '+ New Event' button on the background of a day brings up a wide but short blank box - see screenshot (Screenshot from 2026-10-08 14-52-26.png: modal shows only a close X, no form content; month view October 2026 behind it; user logged in)"
severity: major

### 5. New Event pop-up as staff or superuser (re-run of Test 4 after gap closure 39-05)
expected: On the dev server, signed in as a staff or superuser account (e.g. sssc_admin or talister), click "+ New Event" and click an empty day cell; then open an existing unlinked entry that has a High-band candidate. Both triggers open the pop-up with the create form (title, dates, Save and "Save and Edit"), not an empty box; the existing unlinked entry's pop-up still shows the "Possible campaign run match" hint. (Automated evidence: staff/superuser GETs return 200 with the form and the staff body equals the plain user's once the CSRF token is masked; the planner deferred the real-browser check to end of phase.)
result: [pending]

### 6. Runbook bare-form sentence reads as asked (re-check of Test 3 after gap closure 39-05)
expected: Read docs/runbooks/telescope_runs_calendar.rst lines 2596-2607 once. The new closing sentences ("show a bare, unstyled copy of the event form. Do not use that copy: its Save saves nothing and silently discards what was typed; go back to the calendar page and make the change there.") read clearly and match what was asked for at UAT Test 3.
result: [pending]

### 7. Decide on 39-REVIEW WR-04 (non-staff edit pop-up hint test)
expected: No repository test GETs the edit pop-up as a signed-in NON-staff user and asserts the "Possible campaign run match" hint is absent. Either add the test before shipping (recommended; about 6 lines in EventModalAttributionHintTest, reusing plain_user, _signed_in_client and unlinked_event_with_candidate) or accept WR-04 as an advisory. Behaviour is correct today: the verifier's scratch test of exactly that case passes on the real tree and fails when the template's gate is mutated to request.user.is_authenticated, while all 17 repository tests still pass under that mutation. Not a must-have failure; a ship decision.
result: [pending]

## Summary

total: 7
passed: 2
issues: 2
pending: 3
skipped: 0
blocked: 0

## Gaps

- gap_id: G-39-3
  truth: "The runbook's read-only paragraph warns that after a CSRF-failure login the create and edit addresses show a bare copy of the event form, and tells the operator to go back to the calendar page instead of using it (39-REVIEW WR-01)"
  status: failed
  reason: "User reported: pass, but add the WR-01 sentence"
  severity: minor
  test: 3
  root_cause: "Documentation omission, no code at fault: 39-04 (bc73bfd) rewrote the runbook's CSRF-failure paragraph to end '...the create and edit addresses only show a form' (docs/runbooks/telescope_runs_calendar.rst:2604-2605). Literally true, but it does not warn that after login those addresses render a bare, unstyled copy of the event form (event_form.html is a fragment with no base page and no htmx; its <form> has hx-post but no method/action), so its Save button does a plain GET that saves nothing, silently discards the typed input and puts the CSRF token in the URL — and it never tells the operator to go back to the calendar page (39-REVIEW WR-01, facts reproduced with a signed-in Client)."
  artifacts:
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      issue: "lines 2596-2605 (paragraph 'A write attempt while logged out changes nothing...'): closing clause at 2604-2605 lacks the bare-form warning and the go-back instruction"
    - path: "src/templates/tom_calendar/partials/event_form.html"
      issue: "supporting evidence only, no change requested: lines 40-45 <form hx-post=... hx-target=...> with no method/action; fragment has no <html>/<script>"
  missing:
    - "Extend the clause at runbook lines 2604-2605: the create and edit addresses show a bare, unstyled copy of the event form; do not use it — its Save saves nothing and silently discards what was typed; go back to the calendar page and make the change there (optionally: its Save also puts the page's security token into the address bar and browser history)"
    - "No template change (method=\"post\" was offered as optional and not requested); runbook is the paired doc for calendar_access.py and the calendar templates (CLAUDE.md)"
  debug_session: .planning/debug/runbook-bare-form-warning.md
- gap_id: G-39-4
  truth: "Clicking + New Event (or an empty day cell) while logged in opens the event pop-up with the create-event form rendered inside it"
  status: failed
  reason: "User reported: clicking on the '+ New Event' button on the background of a day brings up a wide but short blank box - see screenshot (modal with only a close X, no form content)"
  severity: major
  test: 4
  root_cause: "GET /calendar/create/ returns 500 for any staff or superuser account (plain users get the form): tom_calendar's create_event renders event_form.html with no `event` in the context, so `event` resolves to ''; the staff-only attribution-hint branch at event_form.html:284 (`{% elif not event.telescope_label_meta.run and request.user.is_staff %}`, the else-arm of `{% if deco %}` at :229, outside the is_authenticated form branch and with no action/event-exists gate) is therefore true for staff, and :295 calls `{% high_band_attribution_candidates event %}` with ''. That tag (attribution_display_extras.py:24-48) lacks the `isinstance(event, CalendarEvent)` guard its three sibling tags have, so its first statement `CalendarEventMeta.objects.filter(event=event, ...)` raises `ValueError: Field 'id' expected a number but got ''` while Django builds the lookup, before any SQL. calendar.html:222/:240 open the modal in hx-on::after-request whatever the status and htmx does not swap a 5xx, so the 500 shows as an empty modal-lg shell. NOT introduced by Phase 39: the elif and tag call are unchanged in content and nesting since fd79c1e (pre-39); latent since 27-07 (58998952) / 33-06 (d4c044f); the create modal could not open at all before 33-11. Today's Playwright test passed because its editor (`create_user`, test_bootstrap5_rendering.py:128) is not staff; no existing test GETs the create form as staff."
  artifacts:
    - path: "solsys_code/templatetags/attribution_display_extras.py"
      issue: "lines 24-48 high_band_attribution_candidates: no `isinstance(event, CalendarEvent)` guard; line 46 raises on '' although the docstring says it never raises"
    - path: "src/templates/tom_calendar/partials/event_form.html"
      issue: "line 284 staff-hint elif fires on the create form (no action == 'update' / event-exists gate); line 295 is the tag call"
    - path: "solsys_code/campaign_attribution.py"
      issue: "line 644 candidates_for_event is also unguarded (the tag's fallback path)"
    - path: "src/templates/tom_calendar/partials/calendar.html"
      issue: "contributing, not root: lines 222 and 240 show the modal in hx-on::after-request regardless of response status, hiding the 500 as a blank box"
  missing:
    - "Guard the tag: first line `if not isinstance(event, CalendarEvent): return []` (same pattern as campaign_decoration / run_tally / observation_series_decoration in calendar_display_extras.py:544/633/824), making the 'Never raises' docstring true"
    - "Optionally gate the template branch so the create form never evaluates the hint (e.g. `{% elif action == 'update' and request.user.is_staff and not event.telescope_label_meta.run %}`) — keep the event_form.html header list and the pinned snapshot solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff in step if the template changes"
    - "Regression tests in solsys_code/tests/test_calendar_template.py: GET /calendar/create/ and /calendar/create/?date=... as is_staff=True and as a superuser with HTTP_HX_REQUEST='true' -> 200, '<form', '>Save and Edit</button>'; non-staff neighbour; staff update-event on a real unlinked event still renders the hint"
    - "Optional hardening (touches the FOMO override calendar.html, paired-doc rule applies): open the modal only on success (`if(event.detail.successful)`) so a future 500 is not shown as a blank box"
  debug_session: .planning/debug/blank-new-event-popup.md
