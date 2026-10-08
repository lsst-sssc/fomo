---
status: diagnosed
trigger: "So I don't know if it broke in Phase 39 but clicking on the '+ New Event' button on the background of a day brings up a wide but short blank box - see screenshot"
created: 2026-10-08T22:05:00Z
updated: 2026-10-08T22:45:00Z
---

## Current Focus
<!-- OVERWRITE on each update - always reflects NOW -->

hypothesis: CONFIRMED H1 -- for any STAFF user, GET /calendar/create/ (with or without ?date=, with or without HX-Request) returns 500 because event_form.html:284 `{% elif not event.telescope_label_meta.run and request.user.is_staff %}` is true on the create form (no `event` in context), so :295 calls high_band_attribution_candidates('') and attribution_display_extras.py:46 raises ValueError; htmx does not swap a 5xx, but calendar.html's hx-on::after-request still shows #cal-modal -> empty modal-lg shell
test: done -- django.test.Client GETs against dev DB (read-only), in-process render of the pre-Phase-39 template, direct call of candidates_for_event('')
expecting: n/a (diagnosed)
next_action: return ROOT CAUSE FOUND to orchestrator (goal find_root_cause_only); no fix applied
bug_class: bohrbug (deterministic: fails 4/4 variants for a staff user, 0/4 for a non-staff user; raised while building the lookup, before any SQL, so independent of DB contents)
reasoning_checkpoint:
  hypothesis: "The blank pop-up is a 500 from GET /calendar/create/ for staff viewers: event_form.html's staff-only attribution-hint branch (line 284) has no action/event gate, so on the create form it passes the missing `event` ('' placeholder) to high_band_attribution_candidates, whose unguarded ORM filter (line 46) raises ValueError; calendar.html shows the modal after any request, so the un-swapped 500 leaves an empty modal."
  confirming_evidence:
    - "Client GET as sssc_admin (is_staff, is_superuser): ValueError 'Field id expected a number but got ' at attribution_display_extras.py:46; non-raising client -> status 500, for /calendar/create/ and ?date=2026-10-02, HX and non-HX"
    - "Same GETs as non-staff account look_admin's peer AnonymousUser row (is_staff False): 200, body starts with <form hx-post=/calendar/create/"
    - "Developer's dev-server traceback (relayed by coordinator) for GET /calendar/create/?date=2026-09-29: identical chain calendar_access.py:59 -> tom_calendar/views.py:184 -> attribution_display_extras.py:46 ValueError"
    - "Pre-Phase-39 template fd79c1e rendered in-process: staff -> same ValueError; non-staff -> OK with <form"
  falsification_test: "A staff user getting 200 with the form from /calendar/create/ would refute it -- observed 500 instead"
  fix_rationale: "(not applied -- diagnose only) Gate the staff hint on an existing event and/or make the tag return [] for a non-CalendarEvent value, matching its three sibling tags"
  blind_spots: "Browser not observed directly (relied on Client + developer traceback); did not determine why earlier UATs never hit it (likely: the modal never opened before ee9957a 2026-09-09, and later UATs clicked existing events, not + New Event, as staff)"
  candidate_causes:
    - "code: unguarded template tag high_band_attribution_candidates + staff branch not gated on action/event (CONFIRMED)"
    - "environment/data: viewer role -- every real dev account the developer uses (sssc_admin, talister, wr06tmp) is staff/superuser while test fixtures use plain create_user (CONFIRMED as trigger condition)"
    - "config/session: 302/HX-Redirect from login or account middleware (ELIMINATED -- status is 500, no HX-* headers)"
  and_gate: "yes -- needs (a) staff viewer AND (b) create-form context with no `event`; the blank-modal presentation additionally needs (c) the after-request handler that shows the modal regardless of status"
tdd_checkpoint: null

## Symptoms
<!-- Written during gathering, then immutable -->

expected: The Bootstrap 5 modal #cal-modal opens with #cal-modal-body filled by the hx-get response from /calendar/create/ (event_form.html: title, start/end, Save / Save and Edit buttons).
actual: User report "clicking on the '+ New Event' button on the background of a day brings up a wide but short blank box" — screenshot shows the modal shell (wide, short white box with only a close X, no form content) over the October 2026 month view; user is logged in.
errors: None reported; dev-server console on /dev/pts/27 not readable.
reproduction: UAT Test 4 in .planning/phases/39-calendar-write-access/39-UAT.md (gap G-39-4), on `python manage.py runserver tlister-thinkmate.lco.gtn:8000` from this checkout with src/fomo_db.sqlite3.
started: Reported 2026-10-08 during Phase 39 UAT after 39-04; Playwright functional test test_calendar_modal_opens_for_new_event_button_with_no_page_errors PASSED on a fresh test DB today.

## Eliminated
<!-- APPEND only - prevents re-investigating after /clear -->

- hypothesis: H2 -- the response is a 302 / HX-Redirect (login, session or account-requirement middleware quirk for the real user)
  evidence: staff GET returns status 500 with no Location and no HX-* headers; non-staff returns 200 with no HX-* headers
  timestamp: 2026-10-08T22:25:00Z

- hypothesis: H3 -- 200 with an empty/whitespace body from a template branch (e.g. is_authenticated / read-only card)
  evidence: non-staff 200 body is 3533-3583 bytes beginning with <form ...>; staff never gets a 200 at all
  timestamp: 2026-10-08T22:25:00Z

- hypothesis: H4 -- client-side JS error before the swap
  evidence: not needed -- server returns 500, which htmx does not swap by default; the month view itself renders 200 for the superuser with the + New Event button present
  timestamp: 2026-10-08T22:25:00Z

- hypothesis: Phase 39 (8d0eca5 / 9b2df4d / 65ba57c) moved the staff hint block so it now renders for action == "create"
  evidence: 8d0eca5 only added {% if request.user.is_authenticated %} wrappers (HEAD 39/116/163 around the form, 316/322/334 around the todos include); the elif/tag call sit after {% if deco %} at fd79c1e 224/235, 798dfe9~1 281/292, HEAD 284/295 -- same content, same nesting, never gated by action; fd79c1e template reproduces the ValueError for staff
  timestamp: 2026-10-08T22:40:00Z

## Evidence
<!-- APPEND only - facts discovered during investigation -->

- timestamp: 2026-10-08T22:05:00Z
  checked: .planning/debug/knowledge-base.md (keyword scan: modal, htmx, blank, empty)
  found: no entry about the calendar modal, htmx swaps, or blank pop-ups
  implication: no known-pattern candidate; investigate from scratch

- timestamp: 2026-10-08T22:10:00Z
  checked: solsys_code/calendar_urls.py, solsys_code/calendar_access.py, upstream tom_calendar.views.create_event (tomtoolkit 3.1.0)
  found: create-event is write_requires_login(create_event); GET for a signed-in user renders tom_calendar/partials/event_form.html with context {"form", "action": "create"} only -- no `event` key
  implication: every `event` reference in event_form.html resolves to a missing variable on the create form

- timestamp: 2026-10-08T22:12:00Z
  checked: src/templates/tom_calendar/partials/event_form.html lines 215-312 and the four tags it calls
  found: campaign_decoration, run_tally and observation_series_decoration all start with `if not isinstance(event, CalendarEvent): return None` (33-11 Rule 1 fix, comment names the create-event context). high_band_attribution_candidates (solsys_code/templatetags/attribution_display_extras.py:24-48) has NO such guard; its first statement is CalendarEventMeta.objects.filter(event=event, observation_record__isnull=False).exists(). It is reached via line 284 `{% elif not event.telescope_label_meta.run and request.user.is_staff %}`, which on the create form is `not None and is_staff` -> True for any staff user
  implication: candidate H1 -- create form renders for non-staff users but should raise for staff users; Playwright functional test passing on a fresh DB would be consistent if its user is not staff

- timestamp: 2026-10-08T22:20:00Z
  checked: scratchpad/repro_create.py via `python manage.py shell` against src/fomo_db.sqlite3 (GET only); users in DB: AnonymousUser (non-staff), sssc_admin, talister, wr06tmp (staff+superuser), look_admin (non-staff)
  found: sssc_admin -> ValueError "Field 'id' expected a number but got ''." raised at solsys_code/templatetags/attribution_display_extras.py:46 (via calendar_access.py:59 -> tom_calendar/views.py:184 render), non-raising client status 500, for /calendar/create/ and /calendar/create/?date=2026-10-02, with and without HX-Request/HX-Target. Non-staff user -> 200, 3533/3583 bytes, body starts `<form hx-post="/calendar/create/" hx-target="#calendar-partial">`. GET /calendar/?month=10&year=2026 for sssc_admin -> 200 with "+ New Event"
  implication: H1 confirmed; failure is role-dependent (staff), not data-dependent

- timestamp: 2026-10-08T22:22:00Z
  checked: upstream tom_calendar/templates/tom_calendar/calendar_page.html lines 6-12; src/templates/tom_calendar/partials/calendar.html 220-222 and 238-240
  found: #cal-modal is `modal-dialog modal-lg` with a modal-header holding only the btn-close and an initially empty #cal-modal-body; both create triggers use hx-on::after-request="bootstrap.Modal.getOrCreateInstance(...).show();" (added ee9957a 33-11, 2026-09-09), which fires for error responses too, while htmx does not swap 4xx/5xx by default
  implication: exactly the "wide but short blank box with only a close X" in the screenshot

- timestamp: 2026-10-08T22:30:00Z
  checked: solsys_code/tests/test_bootstrap5_rendering.py:128 (functional test user), test_calendar_template.py:1613/1666, test_calendar_write_access.py:286/380 (editor users), staff fixtures at test_calendar_template.py:434/792/1074
  found: every user that GETs /calendar/create/ in tests is created with plain create_user (is_staff False); the staff fixtures only GET update-event for an existing event
  implication: explains why test_calendar_modal_opens_for_new_event_button_with_no_page_errors and the editor round trip pass -- the staff branch is never reached on the create path in any test. DB contents are irrelevant (ValueError raised while building the lookup, before SQL)

- timestamp: 2026-10-08T22:35:00Z
  checked: git blame + in-process render of `git show fd79c1e:src/templates/tom_calendar/partials/event_form.html` (parent of first Phase 39 commit 9c06ea4) as sssc_admin and look_admin; direct call campaign_attribution.candidates_for_event('')
  found: baseline template -> staff ValueError, non-staff OK with <form. Blame: event_form.html:284 condition d4c044f8 (33-06, 2026-09-05, rewrote 27-07's `{% elif not run and request.user.is_staff %}` with the same truthiness); :295 tag call 58998952 (27-07, 2026-08-06); attribution_display_extras.py:46 6761f9c (37.1 WR-06, 2026-10-02). candidates_for_event('') also raises the same ValueError at campaign_attribution.py:644 (file unchanged since before 37.1), so the pre-37.1 tag failed identically
  implication: NOT introduced by Phase 39; latent since 27-07 (2026-08-06) for staff on the create form. Plausibly unnoticed because the create modal did not open at all until ee9957a (2026-09-09) and later UAT clicks were on existing events (unverified)

- timestamp: 2026-10-08T22:40:00Z
  checked: coordinator-relayed developer dev-server traceback for GET /calendar/create/?date=2026-09-29
  found: identical chain and exception to the Client reproduction
  implication: reproduction matches the live failure

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: For any staff/superuser viewer, GET /calendar/create/ raises ValueError("Field 'id' expected a number but got ''.") and returns 500. tom_calendar's create_event renders event_form.html with only {form, action:"create"}; the staff-only attribution-hint branch at event_form.html:284 (`{% elif not event.telescope_label_meta.run and request.user.is_staff %}`, the else-arm of `{% if deco %}`, outside both the is_authenticated form branch and any action gate) evaluates `not None and is_staff` -> True, so :295 calls high_band_attribution_candidates with the invalid-variable placeholder ''; unlike its siblings campaign_decoration/run_tally/observation_series_decoration it has no isinstance(event, CalendarEvent) guard, and its first statement (attribution_display_extras.py:46) builds CalendarEventMeta.objects.filter(event='') -> ValueError (pre-37.1 the same '' reached candidates_for_event, which fails the same way at campaign_attribution.py:644). Contributing (presentation): calendar.html:222/:240 show the modal in hx-on::after-request regardless of status, and htmx does not swap a 5xx, so the 500 appears as an empty modal-lg shell. Not a Phase 39 regression: latent since 27-07 (58998952, 2026-08-06); every test that GETs the create form uses a non-staff user.
fix: (not applied -- goal find_root_cause_only)
verification:
oracle_type:
files_changed: []
