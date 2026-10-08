---
phase: "39"
slug: "calendar-write-access"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 39-01-01 | 01 | 1 | ACCESS-01 | T-39-01, T-39-03, T-39-04, T-39-08 | anonymous POST/PUT/PATCH/DELETE/OPTIONS on update-event is a 302 to `/accounts/login/?next=/calendar/` with every field unchanged; anonymous GET/HEAD (the pop-up) 200; plain signed-in POST saves; guard module has no mutable state and no upstream logic (AST check) | unit (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access` | ❌ W0 — created by this task (tests first) | ⬜ pending |
| 39-01-02 | 01 | 1 | ACCESS-01 | T-39-01..T-39-07, T-39-09 | all five routes refuse anonymous callers on every method (GET included on delete/todo routes); htmx gets 200 + HX-Redirect, never 403; signed-in GET on delete-event/create-todo/update-todo is 405; CSRF still enforced; literal paths resolve to FOMO's guarded conf (app_name 'calendar') | unit (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access solsys_code.tests.test_urls` | ❌ W0 — module from 39-01-01 | ⬜ pending |
| 39-02-01 | 02 | 1 | ACCESS-02 | T-39-11..T-39-15 | anonymous month view has no `/calendar/create/` target; anonymous pop-up is the read-only `cal-event-card` (no form/input/select/textarea/button/hx-post/csrf token), shows the attributed-run block once and read-only todos, escapes markup, never echoes ALLOC:/RUN: keys; editor form unchanged; runbook paragraph present | unit (TDD, RED recorded) + structural python check | `python manage.py test --noinput solsys_code.tests.test_calendar_template` | ✅ module exists — new classes added by this task | ⬜ pending |
| 39-02-02 | 02 | 1 | WARN-01 | T-39-16 | event_form.html header pins tomtoolkit 3.1.0 and lists six items; every region where the body differs from the installed upstream file matches an item's anchor and vice versa; calendar.html carries only Bootstrap 5 utility names, keeps `data-url` | unit, source-level (TDD, RED recorded) | `python manage.py test --noinput solsys_code.tests.test_calendar_template` | ✅ module exists — new classes added by this task | ⬜ pending |
| 39-02-03 | 02 | 1 | ACCESS-02 | T-39-15 | Chromium, no cookie: no '+ New Event', no create hx-get, event click opens the read-only card via the Bootstrap 5 API with the attributed-run link and no form control; empty day cell opens nothing and sends no create request; logged-in user still opens the modal from both create targets | functional (Playwright) | `python manage.py test --noinput --tag functional` with the seven named `TestBootstrap5Rendering` tests (see 39-02 Task 3 verify) | ✅ module exists — tests added/re-pointed by this task | ⬜ pending |
| 39-03-01 | 03 | 2 | ACCESS-01, ACCESS-02 | T-39-17 | a logged-in plain user creates, edits and deletes an event from the month view through the guarded routes; no test-only login path | functional (Playwright) | `python manage.py test --noinput --tag functional` with the five named `TestBootstrap5Rendering` tests (see 39-03 Task 1 verify) | ✅ module exists — test added by this task | ⬜ pending |
| 39-03-02 | 03 | 2 | ACCESS-01, ACCESS-02, WARN-01 | T-39-18, T-39-19 | full suite OK with no skips; ruff clean twice; check only urls.W005; no migrations; tom_calendar files match tomtoolkit RECORD; ledgers record WR-05 fixed with no other disposition changed | full suite + gates + ledger check | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` (plus the four gate commands in 39-03 Task 2 verify) | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_calendar_write_access.py` — ACCESS-01 tests (AnonymousCalendarWriteTest, SignedInCalendarWriteTest, CalendarUrlConfShadowingTest). Folded into 39-01 Task 1 (tracer, tests written and run RED before the guard) and extended in Task 2; no separate Wave 0 plan.
- [ ] New classes in `solsys_code/tests/test_calendar_template.py` — CalendarMonthViewReadOnlyTest, EventModalReadOnlyCardTest, EventCardUrlLinkTest (39-02 Task 1, written RED first); EventFormHeaderMatchesUpstreamTest, CalendarTemplateBootstrap5ClassTest (39-02 Task 2, RED first). `EventFormUrlLinkTest` is re-pointed at a logged-in user in 39-02 Task 1 (RESEARCH Pitfall 3).
- [ ] `solsys_code/tests/test_bootstrap5_rendering.py` — two anonymous create-target tests re-pointed at a logged-in user and two anonymous read-only tests added (39-02 Task 3); editor round trip added (39-03 Task 1).
- Framework install: none — Django runner, Playwright 1.62.0 and Chromium (`~/.cache/ms-playwright/chromium-1234`, `-1243`) are already present.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| After logging in from a refused write, django-allauth returns the visitor to `/calendar/` (RESEARCH A1) | ACCESS-01 | A scripted allauth login through tom_common's account-requirements middleware is brittle; the redirect URL itself (`/accounts/login/?next=/calendar/`) is asserted automatically, and if `next` were ignored the visitor lands on `/` with the guard still holding | Log out; open `/calendar/delete/<id>/` for an existing event in the browser; confirm the login page opens with `?next=/calendar/`; log in; confirm the browser lands on `/calendar/` and the event still exists |
| The month title stays roughly centred for a visitor once the '+ New Event' button is gone (RESEARCH Pitfall 6) | ACCESS-02 | Visual layout; the `cal-header-spacer` third child is asserted automatically, its visual effect is not | Log out; open `/calendar/`; confirm the month name sits near the middle of the header row, not at the right edge |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s (targeted runs; the full suite is the wave/phase gate)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** {pending / approved YYYY-MM-DD}
