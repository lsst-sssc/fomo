---
phase: "39"
slug: "calendar-write-access"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-10-08"
---

# Phase 39 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner only (`django.test.TestCase` / `SimpleTestCase`; `StaticLiveServerTestCase` + Playwright 1.62 Chromium for `@tag('functional')` tests). No pytest (CLAUDE.md "Testing"). |
| **Config file** | none — `manage.py` sets `DJANGO_SETTINGS_MODULE=src.fomo.settings`; coverage config is `[tool.coverage.run]` in `pyproject.toml` |
| **Quick run command** | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access solsys_code.tests.test_calendar_template` |
| **Full suite command** | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (functional browser tests included; CI/pre-commit form: `python manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault`) |
| **Estimated runtime** | ~12 s process start per run (SPICE kernels load on import, measured 2026-10-08 with `test_urls`: 12.4 s total) plus ~10-60 s per targeted module; full suite ~10-11 min |

---

## Sampling Rate

- **After every task commit:** the task's own `<automated>` targeted run (the touched test module, or the named functional tests for a browser task); each code commit also runs the pre-commit `django-test` hook (whole suite minus `functional` and `ephemeris_segfault`, ~10 min, run in the background)
- **After every plan wave:** Run `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (after wave 1 this already includes 39-02's re-pointed and new browser tests, so the functional tests are consistent at the wave boundary)
- **Before `/gsd-verify-work`:** Full suite must be green (39-03 Task 2 runs it, plus both ruff hooks twice, `manage.py check`, `makemigrations --check --dry-run` and the tomtoolkit RECORD hash check)
- **Max feedback latency:** ~90 seconds for a targeted run (process start plus one module)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 39-01-01 | 01 | 1 | ACCESS-01 | T-39-01, T-39-03, T-39-04, T-39-08 | anonymous POST/PUT/PATCH/DELETE/OPTIONS on update-event is a 302 to `/accounts/login/?next=/calendar/` with every field unchanged; anonymous GET/HEAD (the pop-up) 200; plain signed-in POST saves; guard module has no mutable state and no upstream logic (AST check) | unit (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access` | ❌ W0 — created by this task (tests first) | ✅ green |
| 39-01-02 | 01 | 1 | ACCESS-01 | T-39-01..T-39-07, T-39-09 | all five routes refuse anonymous callers on every method (GET included on delete/todo routes); htmx gets 200 + HX-Redirect, never 403; signed-in GET on delete-event/create-todo/update-todo is 405; CSRF still enforced; literal paths resolve to FOMO's guarded conf (app_name 'calendar') | unit (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access solsys_code.tests.test_urls` | ❌ W0 — module from 39-01-01 | ✅ green |
| 39-02-01 | 02 | 1 | ACCESS-02 | T-39-11..T-39-15 | anonymous month view has no `/calendar/create/` target; anonymous pop-up is the read-only `cal-event-card` (no form/input/select/textarea/button/hx-post/csrf token), shows the attributed-run block once and read-only todos, escapes markup, never echoes ALLOC:/RUN: keys; editor form unchanged; runbook paragraph present | unit (TDD, RED recorded) + structural python check | `python manage.py test --noinput solsys_code.tests.test_calendar_template` | ✅ module exists — new classes added by this task | ✅ green |
| 39-02-02 | 02 | 1 | WARN-01 | T-39-16 | event_form.html header pins tomtoolkit 3.1.0 and lists six items; every region where the body differs from the installed upstream file matches an item's anchor and vice versa; calendar.html carries only Bootstrap 5 utility names, keeps `data-url` | unit, source-level (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_template` | ✅ module exists — new classes added by this task | ✅ green |
| 39-02-03 | 02 | 1 | ACCESS-02 | T-39-15 | Chromium, no cookie: no '+ New Event', no create hx-get, event click opens the read-only card via the Bootstrap 5 API with the attributed-run link and no form control; empty day cell opens nothing and sends no create request; logged-in user still opens the modal from both create targets | functional (Playwright) | `python manage.py test --noinput --tag functional` with the seven named `TestBootstrap5Rendering` tests (see 39-02 Task 3 verify) | ✅ module exists — tests added/re-pointed by this task | ✅ green |
| 39-03-01 | 03 | 2 | ACCESS-01, ACCESS-02 | T-39-17 | a logged-in plain user creates, edits and deletes an event from the month view through the guarded routes; no test-only login path | functional (Playwright) | `python manage.py test --noinput --tag functional` with the five named `TestBootstrap5Rendering` tests (see 39-03 Task 1 verify) | ✅ module exists — test added by this task | ✅ green |
| 39-03-02 | 03 | 2 | ACCESS-01, ACCESS-02, WARN-01 | T-39-18, T-39-19 | full suite OK with no skips; ruff clean twice; check only urls.W005; no migrations; tom_calendar files match tomtoolkit RECORD; ledgers record WR-05 fixed with no other disposition changed | full suite + gates + ledger check | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (plus the four gate commands in 39-03 Task 2 verify) | ✅ | ✅ green |
| 39-04-01 | 04 | 3 | ACCESS-01 | T-39-20, T-39-21 | a tokenless anonymous POST to each of the five write routes is a 302 (htmx: 200 + HX-Redirect) to `/accounts/login/?next=<the refused path>`, repeats identically, a missing id gives 302 not 404, a signed-in GET replay of the refused path writes nothing (405 on delete/todo routes, form render on create/update), no row changes; guard code AST-identical (docstring-only change); runbook names both refusal paths | unit (tracer; non-vacuity run with CSRF off FAILED 22) + structural python check | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access` | ✅ module exists — `CalendarRowSnapshotMixin`, `AnonymousCsrfFailureWriteTest` added by this task | ✅ green (30 tests) |
| 39-04-02 | 04 | 3 | WARN-01 | T-39-23 | create form's "Save and Edit" label matches tomtoolkit 3.1.0; the body's normalized diff against the installed upstream file matches the pinned snapshot `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff` (10 regions); an unlisted line inside an anchored region fails (non-vacuity test) | unit, source-level (TDD, RED recorded in `39-04-red-evidence-task2.json`) | `python manage.py test --noinput solsys_code.tests.test_calendar_template` (+ the two named functional tests) | ✅ module exists — tests added by this task | ✅ green (97 + 2 functional) |
| 39-04-03 | 04 | 3 | ACCESS-01 | T-39-22, T-39-24 | full suite OK with no skips; ruff clean twice; check only urls.W005; no migrations; tom_calendar files match tomtoolkit RECORD; CR-01 recorded as an accepted risk with the verbatim rationale in the runbook, 39-SECURITY.md (T-39-10 premise corrected, T-39-22 row, AR-39-01) and the review ledger, exactly three ledger lines changed | full suite + gates + records check | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (plus the gate commands and the records check script in 39-04 Task 3 verify) | ✅ | ✅ green (2241 tests) |
| 39-05-01 | 05 | 4 | ACCESS-01, ACCESS-02, WARN-01 | T-39-25, T-39-26, T-39-29 | staff and superuser GET /calendar/create/ (with/without ?date=, htmx or not) return 200 with the create form and no hint; the staff create form is byte-identical to a plain user's apart from the CSRF token; an invalid staff htmx create POST re-renders the form (200, HX-Retarget) and creates nothing; `high_band_attribution_candidates` returns [] for '' and None and never raises (isinstance guard, rest AST-identical to cf76780); the staff hint elif is gated on `action == "update"` and still shows on the edit form for staff and superusers; header item 4 and the pinned snapshot (10 regions) regenerated in the same commit; `<form>` block byte-identical | unit (tracer, TDD, RED recorded in `39-05-red-evidence-task1.json`) + structural python check | `python manage.py test --noinput solsys_code.tests.test_calendar_template` (+ the two named functional tests) | ✅ module exists — six tests added to `EventModalAttributionHintTest` by this task | ✅ green (103 + 2 functional) |
| 39-05-02 | 05 | 4 | ACCESS-01 | T-39-27, T-39-28 | runbook paragraph `A write attempt while logged out changes nothing` says the create and edit addresses show a bare, unstyled copy of the event form, not to use it (its Save saves nothing and silently discards what was typed) and to go back to the calendar page; the 39-04 phrases and the verbatim accepted-risk rationale retained; nothing else under docs/ changed; full suite OK with no skips; ruff clean twice; check only urls.W005; no migrations; tom_calendar files match tomtoolkit RECORD | structural python check + full suite + gates | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (plus the runbook check and the gate commands in 39-05 Task 2 verify) | ✅ | ✅ green (2247 tests) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_calendar_write_access.py` — ACCESS-01 tests (AnonymousCalendarWriteTest, SignedInCalendarWriteTest, CalendarUrlConfShadowingTest). Folded into 39-01 Task 1 (tracer, tests written and run RED before the guard) and extended in Task 2; no separate Wave 0 plan.
- [x] New classes in `solsys_code/tests/test_calendar_template.py` — CalendarMonthViewReadOnlyTest, EventModalReadOnlyCardTest, EventCardUrlLinkTest (39-02 Task 1, written RED first); EventFormHeaderMatchesUpstreamTest, CalendarTemplateBootstrap5ClassTest (39-02 Task 2, RED first). `EventFormUrlLinkTest` is re-pointed at a logged-in user in 39-02 Task 1 (RESEARCH Pitfall 3).
- [x] `solsys_code/tests/test_bootstrap5_rendering.py` — two anonymous create-target tests re-pointed at a logged-in user and two anonymous read-only tests added (39-02 Task 3); editor round trip added (39-03 Task 1).
- Framework install: none — Django runner, Playwright 1.62.0 and Chromium (`~/.cache/ms-playwright/chromium-1234`, `-1243`) are already present.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| After logging in from a refused write, django-allauth returns the visitor to `/calendar/` (RESEARCH A1) | ACCESS-01 | A scripted allauth login through tom_common's account-requirements middleware is brittle; the redirect URL itself (`/accounts/login/?next=/calendar/`) is asserted automatically, and if `next` were ignored the visitor lands on `/` with the guard still holding | Log out; open `/calendar/delete/<id>/` for an existing event in the browser; confirm the login page opens with `?next=/calendar/`; log in; confirm the browser lands on `/calendar/` and the event still exists |
| The month title stays roughly centred for a visitor once the '+ New Event' button is gone (RESEARCH Pitfall 6) | ACCESS-02 | Visual layout; the `cal-header-spacer` third child is asserted automatically, its visual effect is not | Log out; open `/calendar/`; confirm the month name sits near the middle of the header row, not at the right edge |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s (targeted runs; the full suite is the wave/phase gate)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-08 (execute-phase verify:post audit — 7/7 tasks green: 39-01 25 unit tests, 39-02 94 template tests + 13 Playwright, 39-03 full suite 2233 OK incl. functional; 2 manual-only items retained)

**Re-approval after gap closure 39-04:** approved 2026-10-08 (execute-phase verify:post audit — 10/10 tasks green: 39-04 write-access module 30 tests, template module 97 + 2 functional, full suite 2241 OK incl. functional; the 2 manual-only items are retained — the CSRF-failure variant of the first one is now asserted automatically by `test_replaying_the_refused_path_as_a_signed_in_get_changes_nothing`, the allauth landing page itself remains manual)

**Re-approval after gap closure 39-05:** approved 2026-10-08 (execute-phase verify:post audit — 12/12 tasks green: 39-05 template module 103 tests + 2 functional, RED evidence classified RED_EVIDENCE_OK, full suite 2247 OK incl. functional with no skips; the 2 manual-only items are retained unchanged; the `verify-failure-directions` check reports 52 commands, 0 blockers, 0 warnings)

## Validation Audit 2026-10-08

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tasks green | 7 |
| Manual-only | 2 |

## Validation Audit 2026-10-08

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tasks green | 10 |
| Manual-only | 2 |

## Validation Audit 2026-10-08

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tasks green | 12 |
| Manual-only | 2 |
