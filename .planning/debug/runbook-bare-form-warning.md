---
status: diagnosed
trigger: "pass, but add the WR-01 sentence"
created: 2026-10-08T00:00:00Z
updated: 2026-10-08T00:10:00Z
---

## Current Focus
<!-- OVERWRITE on each update - always reflects NOW -->

hypothesis: CONFIRMED -- runbook paragraph at docs/runbooks/telescope_runs_calendar.rst:2596-2605 omits the WR-01 warning; the rendering facts it should warn about are reproduced.
test: done (signed-in Client GET /calendar/create/ in a throwaway test DB)
expecting: n/a
next_action: Return ROOT CAUSE FOUND to orchestrator; gap-closure planner writes the runbook sentence (no code change requested).
bug_class: bohrbug
reasoning_checkpoint: null
tdd_checkpoint: null

## Symptoms
<!-- Written during gathering, then immutable -->

expected: docs/runbooks/telescope_runs_calendar.rst, "Not logged in, the calendar is read-only." section (lines ~2576-2605), describes the CSRF-failure path's post-login landing page truthfully: the create and edit addresses render a bare copy of the event form, and the operator should return to the calendar page rather than use it.
actual: User reported at UAT Test 3: "pass, but add the WR-01 sentence". Runbook says the create and edit addresses "only show a form" with no warning that the form is a dead end.
errors: None reported
reproduction: Test 3 in .planning/phases/39-calendar-write-access/39-UAT.md (gap G-39-3)
started: Discovered during UAT of Phase 39 on 2026-10-08, after gap-closure plan 39-04 rewrote the paragraph (commit bc73bfd).

## Eliminated
<!-- APPEND only - prevents re-investigating after /clear -->

## Evidence
<!-- APPEND only - facts discovered during investigation -->

- timestamp: 2026-10-08T00:00:00Z
  checked: docs/runbooks/telescope_runs_calendar.rst lines 2596-2605
  found: Paragraph "A write attempt while logged out changes nothing ..." ends "after logging in the browser simply opens that address, which changes nothing -- the delete and todo addresses refuse a plain visit and the create and edit addresses only show a form." No warning that the form is bare/unusable, no instruction to go back to the calendar page.
  implication: The WR-01 sentence the user asked for is absent; this is the gap.

- timestamp: 2026-10-08T00:00:00Z
  checked: src/templates/tom_calendar/partials/event_form.html lines 38-46
  found: Fragment template (no extends, no <html>, only `{% load django_bootstrap5 ... %}`); authenticated branch opens `<form hx-post="{% url ... %}" hx-target="#calendar-partial">` (lines 40-45) with no method= and no action=, then `{% csrf_token %}` (line 46).
  implication: Without htmx loaded, a native submit is a GET to the current URL carrying csrfmiddlewaretoken in the query string; nothing is saved.

- timestamp: 2026-10-08T00:00:00Z
  checked: .planning/phases/39-calendar-write-access/39-REVIEW.md WR-01 (lines 77-137)
  found: Review proposes runbook wording "the create and edit addresses show a bare, unstyled copy of the form; do not use it: go back to the calendar page and make the change there." Optional template hardening (method="post") is item 2 and was not requested by the user.
  implication: Fix is scoped to one runbook sentence.

- timestamp: 2026-10-08T00:08:00Z
  checked: Scratch script (scratchpad/wr01_check.py via python manage.py shell, test DB) -- force_login'd Client(enforce_csrf_checks=True) GET /calendar/create/, then a GET to the same URL with title/start_time/end_time + csrfmiddlewaretoken (what a script-less native submit sends)
  found: GET -> 200; body starts directly with `<form hx-post="/calendar/create/" hx-target="#calendar-partial">`; no `<html`, no `<head`, no `<script` at all (so no htmx, no Bootstrap JS/CSS page); form tag has no method= and no action=; csrfmiddlewaretoken hidden input present. Native-submit GET -> 200, CalendarEvent count 0 before and 0 after, typed title not echoed back.
  implication: WR-01 confirmed: the post-login landing page is a bare, unstyled fragment; Save does a GET that writes nothing, discards the input silently, and puts the CSRF token in the URL. The runbook's "only show a form" is literally true but gives the operator no warning.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: Documentation omission. The 39-04 rewrite of the "Not logged in, the calendar is read-only." section (commit bc73bfd) ends the CSRF-failure paragraph (docs/runbooks/telescope_runs_calendar.rst:2596-2605, closing clause at 2604-2605) with "the create and edit addresses only show a form", without saying that this form is a bare, unstyled fragment (upstream create_event/update_event render tom_calendar/partials/event_form.html standalone: no page, no htmx) whose Save button (form at src/templates/tom_calendar/partials/event_form.html:40-45 has hx-post but no method/action) does a plain GET that saves nothing and silently discards the input -- and without telling the operator to go back to the calendar page instead.
fix: (not applied -- find_root_cause_only) Extend the closing clause of runbook line 2604-2605 with the WR-01 warning sentence.
verification:
oracle_type:
files_changed: []
