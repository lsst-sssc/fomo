---
phase: 39-calendar-write-access
verified: 2026-10-08T16:55:00Z
status: gaps_found
score: 33/36 must-haves verified
covered_files:
  - .planning/phases/39-calendar-write-access/39-01-PLAN.md
  - .planning/phases/39-calendar-write-access/39-01-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-02-PLAN.md
  - .planning/phases/39-calendar-write-access/39-02-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-03-PLAN.md
  - .planning/phases/39-calendar-write-access/39-03-SUMMARY.md
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/calendar_access.py
  - solsys_code/calendar_urls.py
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_write_access.py
  - src/fomo/settings.py
  - src/fomo/urls.py
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
covered_digest: "v3:sha256:32cc64f243aa0dda510814243e92e5500775cf0d13db20c777b66b1cdb40acbc"
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Paired doc (CLAUDE.md paired-docs rule): the runbook's 'Not logged in, the calendar is read-only.' paragraph describes what happens to a logged-out write attempt truthfully"
    status: failed
    reason: "The paragraph says a logged-out write attempt from 'a stale browser tab, a script' is sent to login and 'after logging in, the browser returns to the calendar page, never to the refused write'. A write with no valid CSRF token (a script, or a forged cross-site post) is refused by CsrfViewMiddleware before the guard runs; tom_common's Raise403Middleware turns that 403 into /accounts/login/?next=<the refused write path> (htmx: HX-Redirect to the same). Reproduced for all five routes with an anonymous Client(enforce_csrf_checks=True). No row is written on that path, and replaying the next URL as a signed-in GET gives 405 for delete-event/create-todo/update-todo and a bare form fragment for create-event/update-event, so this is a truthfulness defect, not a write hole (39-REVIEW WR-01)."
    artifacts:
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "Lines ~2586-2589: 'after logging in, the browser returns to the calendar page, never to the refused write' is false for the CSRF-failure path, including the 'script' case the sentence itself names"
      - path: "solsys_code/tests/test_calendar_write_access.py"
        issue: "Every AnonymousCalendarWriteTest test uses the default test Client (CSRF checks off), so the exact Location ?next=/calendar/ is proven only for requests that pass the CSRF check; no anonymous CSRF-enforced test exists"
    missing:
      - "Correct the runbook sentence (e.g. a refused write that passes the CSRF check returns to the calendar page; one that fails it is sent to login with the refused path as next, where a GET changes nothing), OR make the behaviour match the doc with a CSRF_FAILURE_VIEW that returns redirect_to_login(reverse('calendar:calendar')) for anonymous /calendar/ requests"
      - "Add an anonymous Client(enforce_csrf_checks=True) test per write route that asserts no row changes and pins whichever Location is chosen"
      - "Optionally scope calendar_access.py's module-docstring sentence 'The login next is the calendar page, not the refused URL' to the guard's own redirect"
  - truth: "39-01 CSRF must-have: a signed-in POST to calendar:create-event without a CSRF token through Client(enforce_csrf_checks=True) returns 403 and creates nothing (the guards do not exempt CSRF)"
    status: failed
    reason: "As worded the truth is false: the response is 302 to /accounts/login/?next=/calendar/create/ because tom_common's Raise403Middleware rewrites every browser 403. The intent holds: the request is refused, nothing is created, and the Location (next=/calendar/create/) differs from the guard's next=/calendar/, proving the CSRF layer refused it. The executor recorded this as a Rule-1 deviation and changed the test to assert 302. This needs an override (no code change), not a fix."
    artifacts:
      - path: "solsys_code/tests/test_calendar_write_access.py"
        issue: "test_post_without_csrf_token_is_refused asserts 302 + Location next=/calendar/create/, not the 403 the must-have states"
    missing:
      - "Accept the override suggested in the report body (the 403 wording was a planning assumption about the middleware stack), or reword the must-have to 'is refused (302 to login via Raise403Middleware) and creates nothing'"
coincidental_reliance_items:
  - truth: "39-01 Pitfall 2 / SC1: an anonymous refused write is redirected to exactly /accounts/login/?next=/calendar/"
    reason: fixture-only
    harden: "The default django.test.Client disables CSRF checks, so every anonymous exact-Location assertion relies on a precondition (valid CSRF token) that a tokenless production request (script, forged post) does not have; declare the CSRF-passing precondition in the test docstring and add a CSRF-enforced anonymous test (see gap 1)"
human_verification:
  - test: "DECISION (39-REVIEW CR-01, highest priority): open self-registration. Effective settings (including local_settings.py) are TOM_REGISTRATION_STRATEGY='open' and ACCOUNT_EMAIL_VERIFICATION='none'; TomAccountAdapter.is_open_for_signup() returns True and /accounts/signup/ serves the signup form (200, password1 field). save_user() creates the account active and allauth logs it in at once."
    expected: "Developer decides whether D-01 ('any logged-in user may write') stands given that any member of the public can obtain a login in under a minute. Options: TOM_REGISTRATION_STRATEGY='approval_required' (or None) in settings/production local_settings; or reopen D-01 and gate writes on staff/permission at the same wrapping point; or accept explicitly and record why (e.g. production overrides the strategy). Also record that T-39-10/AR-39-01 in 39-SECURITY.md was accepted on the premise 'accounts are issued by the operator', which these settings contradict."
    why_human: "Not a must-have gap as written (the goal is about 'anyone not logged in'; D-01 is locked and 39-01's prohibition forbids narrowing beyond 'logged in'), so the phase could not close it in scope; it reopens a locked decision and depends on the production deployment's settings, which are not visible here."
  - test: "DECISION (39-REVIEW WR-02): event_form.html's 'Save and edit' button label differs from upstream's 'Save and Edit'. Header item 3 names the button block and quotes FOMO's label but describes the difference as markup only. EventFormHeaderMatchesUpstreamTest cannot detect an omission inside a region that already holds an anchor (difflib merges FOMO body lines 79-275, card plus series/campaign/high-band blocks, into one inserted region)."
    expected: "Either restore upstream's 'Save and Edit' label or add the case change to item 3; optionally harden the test (per-region item mapping or a pinned normalized diff snapshot)."
    why_human: "SC4 holds at block level (every differing block is named and the diff matches the list), so this is a precision and future-robustness choice, not a failed truth."
  - test: "Judgment-tier prohibition (39-02): 'MUST NOT show an anonymous visitor anything that looks editable or tells them how to get write access'. Non-authoritative LLM-judge verdict: satisfied for the card and the month partial (no form/input/select/textarea/button, no login prompt; tests assert both). Borderline item: 39-REVIEW IN-03, day cells still change background on hover for visitors (.cal-day:hover rules), which may read as clickable."
    expected: "A human opens /calendar/ logged out and confirms nothing looks editable, or asks for the hover tint to be scoped to logged-in users (39-02 declined that polish deliberately)."
    why_human: "Visual affordance judgement; judgment-tier prohibitions require human resolution and are never silently passed."
  - test: "Backstop truth (39-01 concurrency edge): the guard keeps no state between requests, so concurrent or interrupted anonymous requests are each refused independently."
    expected: "Accept the structural evidence (AST check re-run by the verifier: only a tuple of string constants at module level, no global/nonlocal, calls limited to wraps/redirect_to_login/reverse/view/_login_redirect) as sufficient, or ask for a concurrency test."
    why_human: "Marked verification: backstop (non-inferable); presence and structure never qualify on their own, and no concurrent-request test exists."
---

# Phase 39: Calendar Write Access Verification Report

**Phase Goal:** The public calendar is read-only to anyone not logged in — an anonymous visitor can neither create, change nor delete an event, and is not offered a control that would try — and FOMO's event pop-up override says truthfully how it differs from tomtoolkit 3.1.0's upstream template.
**Verified:** 2026-10-08T16:55:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

The core goal is achieved in code: no anonymous request writes through any of the five calendar write routes by any method (my own CSRF-enforced probe confirms it too), the visitor's month view and pop-up offer no write control, a plain signed-in user still creates, edits and deletes in Chromium, and the WARN-01 header matches the 3.1.0 diff block by block. Two gaps remain. One is real but small: the paired runbook makes a false claim about the CSRF-failure path. The other is a must-have whose wording was wrong (403 vs 302) and needs an override. Separately, CR-01 (open self-registration) is a developer decision that the phase could not close within D-01.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| R1 | SC1: anonymous POST to create-event / update-event / delete-event redirected or refused; a test per route proves count and fields unchanged | ✓ VERIFIED | `test_anonymous_post_{create,update,delete}_event_*` assert 302 + full-field snapshot; module ran OK (123 tests across 3 modules). Verifier probe: CSRF-enforced anonymous POSTs also write nothing |
| R2 | SC2: a logged-in user can still create, update and delete from the month view | ✓ VERIFIED | `test_signed_in_editor_creates_edits_and_deletes_from_month_view` re-run by verifier in Chromium: OK |
| R3 | SC3: anonymous visitor sees no create target, can open and read the pop-up incl. attributed-run link; editor still sees targets | ✓ VERIFIED | CalendarMonthViewReadOnlyTest, EventModalReadOnlyCardTest pass; functional `test_anonymous_visitor_opens_read_only_event_card_with_no_page_errors`, `test_anonymous_click_on_empty_day_cell_opens_nothing`, two signed-in create-target tests re-run: OK |
| R4 | SC4: event_form.html header names the blocks that differ from 3.1.0, diff matches the list | ✓ VERIFIED (warning WR-02) | Verifier difflib run: 10 differing regions, all map to items 1-6 (load line+branch→1/5; URL label→2; buttons→3; inserted card+decorations→5/4; todo branch→6). Label case change sits inside item 3's block, not called out — see human item 2 |
| A1 | D-02: anonymous POST to create-todo/update-todo → 302, EventTodo unchanged | ✓ VERIFIED | `test_anonymous_post_create_todo_adds_nothing`, `..._update_todo_changes_nothing` |
| A2 | D-06/Pitfall 1: anonymous GET to create-event, delete-event, create-todo, update-todo redirected, no row changes | ✓ VERIFIED | four GET tests pass |
| A3 | D-04: update-event GET/HEAD 200, POST/PUT/PATCH/DELETE/OPTIONS redirected | ✓ VERIFIED | `read_open_write_requires_login` (`_READ_METHODS = ('GET','HEAD')`); method tests pass |
| A4 | D-08: htmx anonymous writes → 200 + HX-Redirect, never 403 | ✓ VERIFIED | `test_htmx_anonymous_writes_get_hx_redirect_never_403` |
| A5 | Pitfall 2: guard's next is the calendar page; signed-in GET on destructive routes → 405 | ✓ VERIFIED (coincidental-reliance) | `_login_redirect()` uses `reverse('calendar:calendar')`; 405 test passes. Holds only for CSRF-passing requests (default test Client disables CSRF) — see gap 1 |
| A6 | D-01/D-03: plain signed-in user creates, updates, deletes, adds/changes todos; GET create/update render | ✓ VERIFIED | SignedInCalendarWriteTest passes |
| A7 | CSRF: signed-in tokenless POST returns 403 and creates nothing | ✗ FAILED (wording) | Response is 302 → `/accounts/login/?next=/calendar/create/` (Raise403Middleware); nothing created. Intent met; override suggested below |
| A8 | Shadowing: five literal paths resolve to app 'calendar', carry calendar_guard, unwrap to upstream callables | ✓ VERIFIED | CalendarUrlConfShadowingTest, `test_literal_paths_are_guarded`; `src/fomo/urls.py` include precedes tom_common.urls |
| A9 | Boundary edge: anonymous refused / plain user allowed; missing id → 302 not 404 | ✓ VERIFIED | `test_anonymous_post_to_missing_event_is_redirected_not_404` |
| A10 | Precision edge: every field compared after refresh | ✓ VERIFIED | `snapshot()` covers all 11 event fields + both todo fields + counts |
| A11 | Idempotency edge: repeated anonymous POSTs identical and unchanged | ✓ VERIFIED | two repeat tests pass |
| A12 | Concurrency edge (backstop): guard keeps no state | ? insufficient_spec | AST check re-run by verifier passes; backstop tier needs human acceptance (human item 4) |
| B1 | D-07: signed-in month view keeps '+ New Event' and day-cell hx-get; anonymous header keeps cal-header-spacer | ✓ VERIFIED | calendar.html lines 218-241; `test_signed_in_month_view_keeps_both_create_targets` |
| B2 | D-04/D-05: anonymous card has no form controls or write URLs, shows labelled fields, omits empty rows | ✓ VERIFIED | event_form.html 113-163; card tests pass |
| B3 | Attributed-run block once for both audiences | ✓ VERIFIED | one `campaign_decoration` tag below both branches; `..._renders_attributed_run_block_once`, editor test counts 1 |
| B4 | Read-only todos with (done)/(not done); 'No todos yet.' | ✓ VERIFIED | `cal-todos-readonly` list; two todo tests |
| B5 | URL row: web link with noopener; ALLOC:/RUN:/javascript: never echoed | ✓ VERIFIED | EventCardUrlLinkTest |
| B6 | Editor unchanged; EventFormUrlLinkTest passes signed-in | ✓ VERIFIED | `<form>` block byte-identical to pre-phase; test passes |
| B7 | XSS escaped on card | ✓ VERIFIED | `test_anonymous_card_escapes_markup`; no `|safe`/`autoescape off` (structural check) |
| B8 | Pre-existing modal gate classes pass unmodified | ✓ VERIFIED | `git diff b261c9c..HEAD` touches only EventFormUrlLinkTest (planned) plus appended classes |
| B9 | D-11: Bootstrap 5 names, data-url kept | ✓ VERIFIED | grep finds no BS4 names; `data-url` at line 194; `var(--light/--secondary/--primary)` identical to upstream 3.1.0 lines 8, 9, 33 |
| B10 | D-09 anonymous browser proof | ✓ VERIFIED | re-run OK |
| B11 | Signed-in browser create-target tests log in first and see a form | ✓ VERIFIED | re-run OK |
| B12 | Runbook paragraph present with the required content | ✓ VERIFIED (as worded) | paragraph at line 2576; logged-in-only wording present. One extra sentence is false — see D1 |
| B13 | Idempotency: anonymous reads write nothing | ✓ VERIFIED | `test_anonymous_reads_write_nothing` |
| B14 | Concurrency: editor then anonymous renders share no output | ✓ VERIFIED | `test_editor_then_anonymous_render_share_no_output` |
| C1 | Full local suite OK, no skips | ✓ VERIFIED | Reported by orchestrator at this HEAD (2233 tests incl. functional, OK) and in 39-03-SUMMARY; not re-run per instruction. Verifier's targeted runs: 123 unit + 6 functional OK; no skip/expectedFailure in the three modules |
| C2 | Quality gates: ruff, check, makemigrations | ✓ VERIFIED | ruff + ruff-format Passed on the phase's Python files; `check` only urls.W005; `makemigrations --check` No changes detected |
| C3 | Vendored tom_calendar untouched | ✓ VERIFIED | 22 RECORD-hashed tom_calendar files match tomtoolkit 3.1.0 |
| C4 | D-12: 37.1 ledger WR-05 fixed (front matter + table) | ✓ VERIFIED | line 200 `disposition: fixed`; table row cites 9b2df4d |
| C5 | D-12: 33-REVIEW WR-05 and WR-04 notes | ✓ VERIFIED | ledger script re-run: tokens and 5e31c0f/4d32e62 present; WR-04 names D-11 and campaignrun_table.html |
| D1 | CLAUDE.md paired-docs: runbook describes the logged-out write path truthfully | ✗ FAILED | "after logging in, the browser returns to the calendar page, never to the refused write" is false for tokenless writes (gap 1) |

**Score:** 33/36 truths verified (2 failed, 1 insufficient_spec routed to human; 0 present-but-behavior-unverified)

### Prohibitions

| Plan | Prohibition | Tier | Disposition |
|------|-------------|------|-------------|
| 39-01 | No edit/patch of tomtoolkit; no re-implemented view body | test | ✓ RECORD hash check, AST allowlist, `inspect.unwrap` shadowing test |
| 39-01 | No 403/error page for anonymous writes | test | ✓ guard path 302/HX-Redirect; CSRF-failure path also ends as 302 (Raise403Middleware) |
| 39-01 | Login next must not be the refused write URL | test | ⚠ flagged: true for the guard's redirect (tests); false for the CSRF-failure path, where tom_common sets next=refused path. No write results (verifier probe: 405 or read-only render on replay). Resolved with gap 1 |
| 39-01 | No narrowing beyond "logged in" | test | ✓ no is_staff/has_perm in guard; plain-user tests. Note: this is exactly what CR-01 would have to reopen |
| 39-02 | Nothing looks editable / no login prompt for visitors | judgment | ⚠ flagged, human item 3 (non-authoritative LLM verdict: satisfied; IN-03 hover tint borderline) |
| 39-02 | No internal identifiers on the anonymous card | test | ✓ EventCardUrlLinkTest, EventModalSeriesDecorationTest, EventModalAttributionHintTest |
| 39-02 | No duplicated series/campaign blocks | test | ✓ one tag each (structural check) |
| 39-02 | No third tom_calendar override / no upstream edit | test | ✓ partials dir holds exactly calendar.html, campaign_chip.html, event_form.html; RECORD hashes |
| 39-03 | No test-only login route/setting/hook | test | ✓ ba37d8e touches only the test module; `_log_in_browser` uses force_login cookie hand-off |
| 39-03 | Ledgers not marked fixed early; no other disposition changed | test | ✓ ledger commit 798dfe9 holds exactly the two ledgers; only WR-05 changed |
| 39-03 | No skip/tag-out/expectedFailure | test | ✓ none in the three modules |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/calendar_access.py` | two guard decorators | ✓ VERIFIED | 78 lines, both decorators with `calendar_guard` markers; imported by calendar_urls.py |
| `solsys_code/calendar_urls.py` | five guarded write routes | ✓ VERIFIED | exact wrappings incl. `write_requires_login(require_POST(...))` on the three destructive routes |
| `solsys_code/tests/test_calendar_write_access.py` | ACCESS-01 proof | ✓ VERIFIED | 3 classes, 25 tests, all pass |
| `src/templates/tom_calendar/partials/event_form.html` | six-item header + auth branches | ✓ VERIFIED | `cal-event-card`, `cal-todos-readonly`, 2 `is_authenticated` branches |
| `src/templates/tom_calendar/partials/calendar.html` | create targets for logged-in only; BS5 names | ✓ VERIFIED | `cal-header-spacer`; 3 event-row hx-gets intact |
| `solsys_code/tests/test_calendar_template.py` | ACCESS-02, header, BS5 tests | ✓ VERIFIED | 5 new classes; module passes |
| `solsys_code/tests/test_bootstrap5_rendering.py` | browser proofs | ✓ VERIFIED | anonymous card, inert cell, editor round trip — all re-run OK |
| `docs/runbooks/telescope_runs_calendar.rst` | read-only paragraph | ⚠ PRESENT, one false sentence | gap 1 |
| 37.1-REVIEW-DISPOSITION.md / 33-REVIEW.md | WR-05 recorded fixed | ✓ VERIFIED | ledger script passes |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| calendar_urls.py | calendar_access.py | import of both decorators | ✓ WIRED | `from solsys_code.calendar_access import read_open_write_requires_login, write_requires_login` |
| calendar_urls.py | tom_calendar.views.delete_event | guard → require_POST → upstream | ✓ WIRED | `write_requires_login(require_POST(delete_event))`; unwrap test |
| src/fomo/urls.py | calendar_urls.py | include before tom_common.urls | ✓ WIRED | line 30 precedes `include('tom_common.urls')`; resolve() test reports app 'calendar' |
| event_form.html | request.user | `request.user.is_authenticated` | ✓ WIRED | two branches; request context processor |
| calendar.html | calendar:update-event | event-row hx-get + inner container modal handler | ✓ WIRED | 3 hx-gets; anonymous browser test opens modal with the cell's hx-* absent |
| test_calendar_template.py | installed upstream event_form.html | `tom_calendar.__file__` | ✓ WIRED | `_upstream_lines()` asserts the file exists |
| test_bootstrap5_rendering.py | calendar_urls.py | signed-in hx-post through the guards | ✓ WIRED | round-trip test changes DB rows |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| event_form.html card | `event.*`, `event.todos.all` | upstream `update_event` GET passes the CalendarEvent instance | Yes (tests render real fixture values) | ✓ FLOWING |
| event_form.html decorations | `campaign_decoration event` | CalendarEventMeta.run at request time | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase unit tests | `python manage.py test solsys_code.tests.test_calendar_write_access solsys_code.tests.test_calendar_template solsys_code.tests.test_urls` | Ran 123 tests, OK | ✓ PASS |
| Browser proofs | `--tag functional` six named TestBootstrap5Rendering tests (editor round trip, two signed-in create targets, anonymous card, inert cell, attributed modal) | Found 6, OK | ✓ PASS |
| WR-01 reproduction | scratch test (outside repo) POSTing anonymously with `Client(enforce_csrf_checks=True)` to all 5 routes, plain and htmx, then replaying `next` as a signed-in GET | 302 → `/accounts/login/?next=<refused path>` (htmx: HX-Redirect same); replay 405 ×3, 200 form fragment ×2; no row changed | ✓ no write / ✗ doc claim |
| CR-01 reproduction | `manage.py shell`: settings + `TomAccountAdapter().is_open_for_signup()` + GET `/accounts/signup/` (host 127.0.0.1) | `open`, `none`, True, 200 with `password1` field | reproduced |
| Guard structure | AST allowlist check | AST OK | ✓ PASS |
| Vendored integrity | RECORD hash check | 22 files, 0 mismatches | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts declared in the phase; no `scripts/*/tests/probe-*.sh` relevant).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ACCESS-01 | 39-01, 39-03 | anonymous POST to any of the five write routes changes nothing; test per endpoint | ✓ SATISFIED | R1, A1-A11; tests per route; probe confirms the CSRF path too. Who may write = D-01 (see CR-01 decision) |
| ACCESS-02 | 39-02, 39-03 | month-view create/update click targets hidden from anonymous users | ✓ SATISFIED | R3, B1-B14 |
| WARN-01 | 39-02, 39-03 | event_form.html header states accurately which blocks differ | ✓ SATISFIED (warning) | R4; WR-02 label-case precision is human item 2 |

No orphaned requirements: REQUIREMENTS.md maps exactly ACCESS-01, ACCESS-02, WARN-01 to Phase 39, all claimed by plans.

### Code Review Findings (39-REVIEW.md) — Classification

| Finding | Classification | Reasoning |
|---------|----------------|-----------|
| CR-01 open self-registration | **Human decision** (not a must-have gap as written) | Reproduced. Goal and ACCESS-01 are about "anyone not logged in", which holds; D-01 locked "any logged-in user" and 39-01 forbids narrowing. But D-01's premise ("the same accounts as today", T-39-10 "accounts are issued by the operator") is contradicted by these settings, so the security gate's accepted risk rests on a false assumption. Top-priority decision before shipping (human item 1) |
| WR-01 CSRF-failure redirect | **Must-have gap** (paired doc, gap 1) + flagged prohibition | Reproduced; no data is written on any path; the runbook sentence is false; anonymous tests rely on CSRF-disabled client (coincidental-reliance advisory) |
| WR-02 header omits label change; test can't detect omissions | **Human decision** (warning) | SC4 / the 39-02 WARN-01 must-have hold as worded (every region has an anchor; every anchor in a region and its item); label case change lies inside item 3's block. Precision plus test robustness (human item 2) |
| IN-01 "UTC" label without `|utc` | Out of scope (info) | `TIME_ZONE = 'UTC'`, no `timezone.activate` anywhere in FOMO; correct today, latent only |
| IN-02 whitespace-only todo 500 | Out of scope (info) | Upstream bug that predates the phase; upstream view must not be edited; candidate for Phase 41 triage |
| IN-03 day-cell hover tint for visitors | Human item 3 (tied to the judgment-tier prohibition) | 39-02 deliberately declined it; cell is functionally inert (browser test), so the runbook's "does nothing when clicked" is true |
| IN-04 `--light/--secondary/--primary` | Out of scope | D-11 explicitly says keep them because upstream 3.1.0 uses them; verified identical at upstream lines 8, 9, 33 |

### Executor Deviations

| Deviation | Disposition |
|-----------|-------------|
| 39-01: CSRF test changed from 403 to 302 | Truth A7 FAILED as worded, intent met; override suggested (gap 2). Behaviour correctly reflects tom_common's Raise403Middleware |
| 39-03: edit step counts `form[hx-post*="/calendar/update/"]` instead of `#cal-modal-body form` | Acceptable. Upstream's saved-event pop-up also holds the add-todo form, so a plain count would be 2. No must-have depends on the plain count; the test also asserts no `#cal-event-card`. R2 VERIFIED |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/templates/tom_calendar/partials/event_form.html | 242 | "TBD" | ℹ️ Info | Descriptive ("a TBD run", undated run), pre-existing (2026-07-31), not a debt marker |
| docs/runbooks/telescope_runs_calendar.rst | ~2586-2589 | over-broad claim | 🛑 Blocker (paired-docs rule) | gap 1 |

No TODO/FIXME/XXX/placeholder or stub patterns in the phase's code files.

### Suggested Override (gap 2)

**This looks intentional.** To accept this deviation, add to VERIFICATION.md frontmatter:

```yaml
overrides:
  - must_have: "CSRF unchanged: a signed-in POST to calendar:create-event without a CSRF token through Client(enforce_csrf_checks=True) returns 403 and creates nothing"
    reason: "tom_common's Raise403Middleware rewrites every browser 403 into a 302 to login (next=/calendar/create/); the request is still refused and nothing is created, and the distinct next proves the CSRF layer refused it, not the guard"
    accepted_by: "{name}"
    accepted_at: "{ISO timestamp}"
```

### Human Verification Required

1. **CR-01 registration strategy (decision, highest priority).** Decide whether D-01 stands while `TOM_REGISTRATION_STRATEGY='open'` and `ACCOUNT_EMAIL_VERIFICATION='none'` let anyone self-register and write immediately. Options are `approval_required` or None, a staff/permission gate, or an explicit acceptance. Then correct AR-39-01/T-39-10 in 39-SECURITY.md either way.
2. **WR-02 label.** Restore "Save and Edit" or list the case change under header item 3; optionally harden EventFormHeaderMatchesUpstreamTest.
3. **Visitor affordance (judgment prohibition).** Open /calendar/ logged out and confirm nothing looks editable; decide on the day-cell hover tint (IN-03).
4. **Concurrency backstop truth.** Accept the AST structural evidence or request a concurrency test.

### Gaps Summary

Both gaps come from one root cause. A write that fails the CSRF check never reaches FOMO's guard: tom_common's Raise403Middleware turns the 403 into a login redirect whose `next` is the refused path.

- **Gap 1 (needs work):** the runbook paragraph, the paired doc for this phase, claims the browser "never" returns to the refused write. That is false for tokenless writes, including the "script" example the sentence itself names. The anonymous tests cannot see this path because the default test Client disables CSRF checks. No data is written on this path (verified), so the fix is small: correct the sentence and add a CSRF-enforced anonymous test, or add a `CSRF_FAILURE_VIEW` so the behaviour matches the doc.
- **Gap 2 (needs no code):** the 39-01 must-have expected a 403 that FOMO's middleware stack never returns. Accept the override above or reword the must-have.

CR-01 is not counted as a gap. As written, the goal holds for anyone not logged in, and the phase was bound by D-01. Even so, it is the most important finding here: in this configuration "logged in" is not a barrier, and the developer must decide on it before the security gate is treated as closed.

---

_Verified: 2026-10-08T16:55:00Z_
_Verifier: Claude (gsd-verifier)_
