---
phase: 39-calendar-write-access
verified: 2026-10-08T23:58:00Z
status: human_needed
score: 58/58 must-haves verified
covered_files:
  - .planning/phases/39-calendar-write-access/39-01-PLAN.md
  - .planning/phases/39-calendar-write-access/39-01-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-02-PLAN.md
  - .planning/phases/39-calendar-write-access/39-02-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-03-PLAN.md
  - .planning/phases/39-calendar-write-access/39-03-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-04-PLAN.md
  - .planning/phases/39-calendar-write-access/39-04-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-05-PLAN.md
  - .planning/phases/39-calendar-write-access/39-05-SUMMARY.md
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/calendar_access.py
  - solsys_code/calendar_urls.py
  - solsys_code/campaign_attribution.py
  - solsys_code/templatetags/attribution_display_extras.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_write_access.py
  - src/fomo/settings.py
  - src/fomo/urls.py
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
covered_digest: "v3:sha256:bcc8fed35cb47974b0dff3771ea2c787c8facf6d76d33cb2722efe7ddda806c9"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "CSRF unchanged: a signed-in POST to calendar:create-event without a CSRF token through Client(enforce_csrf_checks=True) returns 403 and creates nothing (the guards do not exempt CSRF)."
    reason: "tom_common's Raise403Middleware rewrites every browser 403 into a 302 to login (next=/calendar/create/); the request is still refused and nothing is created, and the distinct next proves the CSRF layer refused it, not the guard"
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-08T17:04:52Z"
re_verification:
  previous_status: human_needed
  previous_score: 48/49
  gaps_closed:
    - "G-39-4 (39-UAT Test 4, major): staff and superuser GET /calendar/create/ (with and without ?date=, htmx or not) and an invalid staff create POST now return 200 with the form, not 500 -- isinstance guard in high_band_attribution_candidates plus the action-first elif in event_form.html"
    - "G-39-3 (39-UAT Test 3, minor): the runbook's logged-out-write paragraph now says the create and edit addresses show a bare, unstyled copy of the event form, not to use it, and to go back to the calendar page"
    - "Previous human item 1 (39-02 judgment-tier visitor-affordance prohibition): resolved by the developer, 39-UAT Test 1 pass"
    - "Previous human item 2 (39-01 A12 backstop concurrency truth): structural evidence accepted by the developer, 39-UAT Test 2 pass"
    - "Previous human item 3 (39-04 runbook readability) and advisory WR-01: answered at 39-UAT Test 3 ('pass, but add the WR-01 sentence'); the sentence is now in the runbook (G-39-3); residual token-in-URL risk accepted as AR-39-02"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "39-REVIEW WR-02 (carried): event_form.html's header says the pinned snapshot makes 'any new difference fail until this list and that file are updated together'; regenerating the snapshot alone turns the test green for a line inside an already-anchored region, so the header list is not enforced on that path."
    category: other
    reason: "The six-item list is exact today (10 snapshot regions map to items 1-6; item 4 now says 'edit form only'), so the goal's 'says truthfully how it differs' holds; only the sentence about what the test enforces overclaims. One-line reword."
    evidence_status: "statically evident (header lines 9-12 unchanged; current_diff() regenerates from the body only)"
  - finding: "39-REVIEW WR-03 (carried): the snapshot is named 'vs tomtoolkit 3.1.0' but is diffed against the installed tomtoolkit; pyproject.toml line 20 is 'tomtoolkit>=3.1.0', unpinned."
    category: other
    reason: "Fails closed with a misleading remedy on an upstream upgrade; installed version is 3.1.0 so the comparison is true today."
    evidence_status: "statically evident"
  - finding: "39-REVIEW IN-06: the staff 'Save and Edit' create path (the third render of event_form.html, upstream create_event lines 154-163, action='update' with a real event) has no repository test."
    category: other
    reason: "Not a defect today: verifier scratch probe (subclass of EventModalAttributionHintTest, outside the repo) POSTed a valid create with save_and_edit as staff, superuser and plain user -> 200, the update form for the new event, no hint, OK. On that path the hint is evaluated with a real CalendarEvent, which the guarded tag handles. Worth a test, not a gap."
    evidence_status: "scratch probe passed (Ran 1 test, OK)"
  - finding: "39-REVIEW IN-07: the runbook's troubleshooting paragraph covers a pop-up that does not open (JavaScript fault) but not a pop-up that opens empty (a server error), which was G-39-4's actual symptom."
    category: other
    reason: "G-39-4's cause is fixed and pinned by tests; calendar.html still opens the modal on any response (hardening offered at UAT, not requested), so a future 5xx would look the same. Documentation completeness for a hypothetical future fault, not a statement in the runbook that is false. Candidate for Phase 41 todo triage alongside the calendar.html hardening and candidates_for_event's unguarded '' path."
    evidence_status: "statically evident (rst lines 2609-2620)"
human_verification:
  - test: "UAT Test 4 re-run as staff (39-05 Task 1 human-check, SUMMARY D8): on the dev server, signed in as a staff or superuser account (e.g. sssc_admin or talister), click '+ New Event' and click an empty day cell; then open an existing unlinked entry that has a High-band candidate."
    expected: "Both triggers open the pop-up with the create form (title, dates, Save and 'Save and Edit'), not an empty box; the existing unlinked entry's pop-up still shows the 'Possible campaign run match' hint."
    why_human: "Real-browser htmx swap and modal display as staff. Automated evidence is strong (staff/superuser GETs return 200 with the form; the staff body equals the plain user's once the CSRF token is masked, and the plain-user browser test passes), but the planner deferred this check to end of phase."
  - test: "UAT Test 3 re-check (G-39-3, SUMMARY D7): read docs/runbooks/telescope_runs_calendar.rst lines 2596-2607 once."
    expected: "The new closing sentences ('show a bare, unstyled copy of the event form. Do not use that copy: its Save saves nothing and silently discards what was typed; go back to the calendar page and make the change there.') read clearly and match what was asked for at UAT."
    why_human: "Operator-facing wording; the developer requested the sentence, so this is a confirmation, not a new judgement."
  - test: "Decide on 39-REVIEW WR-04: no repository test GETs the edit pop-up as a signed-in NON-staff user and asserts the 'Possible campaign run match' hint is absent."
    expected: "Either add the test before shipping (recommended; about 6 lines in EventModalAttributionHintTest, reusing plain_user, _signed_in_client and unlinked_event_with_candidate) or accept it as an advisory. Behaviour is correct today: the verifier's scratch test of exactly that case passes on the real tree, and fails when the template's gate is mutated to request.user.is_authenticated, while all 17 repository tests in EventModalAttributionHintTest and EventFormHeaderMatchesUpstreamTest still pass under that mutation."
    why_human: "Coverage gap on the T-39-26 information-disclosure mitigation, which 39-05 rewrote and 39-SECURITY.md marks closed on evidence that covers anonymous and staff but not signed-in non-staff viewers. With open self-registration (CR-01, accepted) every new account is such a viewer. It is not a must-have failure (no phase truth or requirement covers non-staff, and the behaviour holds), so it is a ship decision for the developer, not a gap."
---

# Phase 39: Calendar Write Access Verification Report

**Phase Goal:** The public calendar is read-only to anyone not logged in — an anonymous visitor can neither create, change nor delete an event, and is not offered a control that would try — and FOMO's event pop-up override says truthfully how it differs from tomtoolkit 3.1.0's upstream template.
**Verified:** 2026-10-08T23:58:00Z
**Status:** human_needed
**Re-verification:** Yes — after gap-closure plan 39-05 (commits 917a895, c71b7b0, ba8a53a, 3241a21)

The goal is achieved, and both UAT gaps are closed in the code:

- **G-39-4:** staff and superusers get the New Event create form again, with no 500 and no hint.
- **G-39-3:** the runbook now warns about the bare copy of the form.

The full local suite ran once: `Ran 2247 tests ... OK`, exit 0, no skips. Every file the 39-05 prohibitions protect is unchanged since cf76780.

At UAT the developer resolved the three human items from the previous report:

- Test 1 passed (the visitor-affordance prohibition).
- Test 2 passed (the backstop concurrency truth).
- Test 3 led to G-39-3, which is now fixed.

Three human items remain:

- the UAT Test 4 browser re-run as staff, which the planner deferred to end of phase;
- a one-read confirmation of the new runbook sentence;
- a decision on WR-04.

That is why the status is `human_needed` and not `passed`.

## Goal Achievement

### Re-checked gaps

| # | Gap | Now | Evidence |
|---|-----|-----|----------|
| G-39-4 | New Event pop-up blank for staff and superusers (GET /calendar/create/ was 500) | ✓ VERIFIED | Targeted run of `EventModalAttributionHintTest` and `EventFormHeaderMatchesUpstreamTest`: `Ran 17 tests ... OK`. The RED record (39-05-red-evidence-task1.json, exit 1, `Internal Server Error: /calendar/cr...`) proves the new tests failed before the fix |
| G-39-3 | Runbook did not warn about the bare form | ✓ VERIFIED | rst lines 2604-2607 now read "show a bare, unstyled copy of the event form. Do not use that copy: its Save saves nothing and silently discards what was typed; go back to the calendar page and make the change there." The old clause "only show a form" is gone. `git diff --stat cf76780 HEAD -- docs/` shows only this file, +4/-2 |
| Prev. human 1 | Judgment-tier visitor-affordance prohibition (39-02) | ✓ resolved by human | 39-UAT Test 1: pass |
| Prev. human 2 | A12 backstop concurrency truth | ✓ resolved by human | 39-UAT Test 2: pass. calendar_access.py is unchanged since cf76780 |
| Prev. human 3 / WR-01 | Runbook readability and the WR-01 sentence | ✓ resolved | 39-UAT Test 3 asked for the WR-01 sentence; it is now in the runbook (G-39-3). The residual token-in-URL risk is accepted as AR-39-02. The claim "saves nothing and discards what was typed" is true: the form has no method or action, so Save sends a GET to the same address, and upstream `create_event` reads only `date` from GET while `update_event` re-renders the stored event |

### Observable Truths — 39-05 (new, full verification)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| E1 | Staff and superuser GET of /calendar/create/ and ?date=2026-07-16, htmx and not: 200, `<form`, the create hx-post, `>Save and Edit</button>`, the date, and no hint | ✓ VERIFIED | `test_staff_and_superuser_get_the_create_form` (8 subTests) passes. It uses `Client(raise_request_exception=False)`, so a 500 would show up as a FAIL |
| E2 | The staff create form matches the plain user's once the CSRF token is masked | ✓ VERIFIED | `test_staff_create_form_matches_the_plain_users_apart_from_the_csrf_token` passes |
| E3 | A staff htmx invalid create POST: 200, HX-Retarget `#cal-modal-body`, `<form`, and no row created | ✓ VERIFIED | `test_staff_invalid_create_post_re_renders_the_form` passes |
| E4 | The tag returns [] for '' and None; the guard is its first statement after the docstring; the rest is AST-identical to cf76780 | ✓ VERIFIED | Verifier AST check: first statement `if not isinstance(event, CalendarEvent): return []`; the remaining body, signature and decorators are AST-identical; the other top-level code in the module is unchanged. `test_attribution_tag_returns_empty_list_for_a_non_event` passes |
| E5 | The elif reads exactly `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}`; the hint shows for update and not for create | ✓ VERIFIED | event_form.html line 284; it is the only staff elif. `test_hint_is_gated_on_the_edit_form` passes |
| E6 | Pre-existing hint tests are AST-identical and pass; superuser also sees the hint | ✓ VERIFIED | Verifier AST check: no pre-existing test method changed. setUpTestData keeps its original statements as a prefix and adds 2 (superuser, plain_user). `test_superuser_sees_high_band_hint_for_unlinked_event` passes |
| E7 | Header item 4 says "edit form only"; six items; snapshot regenerated with 10 regions and the new elif; `<form>` block identical to cf76780; exactly 3 partials | ✓ VERIFIED | Header items 1-6 present; 10 `@@ ` lines; snapshot line 199 holds the gated elif. Form block identical (verifier check). The partials directory holds calendar.html, campaign_chip.html and event_form.html. `test_body_diff_matches_pinned_snapshot` passes |
| E8 | Runbook paragraph warns about the bare copy; the 39-04 phrases and the verbatim rationale remain; nothing else in docs/ changed | ✓ VERIFIED | Phrase counts: "passes the CSRF check" 1, "fails the CSRF check" 1, "returns to the calendar page" 1, the verbatim rationale 1, "bare, unstyled copy of the event form" 1, old clause 0. The diff touches only that paragraph |
| E9 | Phase gate: full suite OK with no skips; ruff clean; check shows only W005; no migrations; RECORD intact | ✓ VERIFIED | Verifier ran each check once: `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` gave `Ran 2247 tests in 478.474s` / `OK` / exit 0, with no `skipped=` (functional tests included). `pre-commit run ruff` and `ruff-format` on the two 39-05 Python files: Passed, tree unchanged. `manage.py check`: urls.W005 only. `makemigrations --check --dry-run`: No changes detected. RECORD: 22 hashed tom_calendar files, 0 mismatches, tomtoolkit 3.1.0 |

### Observable Truths — roadmap and earlier plans (regression check)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| R1 | SC1: an anonymous POST to the five write routes is refused, and a test per route proves nothing changed | ✓ VERIFIED | `test_calendar_write_access` (both refusal paths) passes in the full suite; calendar_urls.py, calendar_access.py, settings.py and urls.py show an empty `git diff --stat cf76780 HEAD` |
| R2 | SC2: a logged-in user can still create, update and delete from the month view | ✓ VERIFIED (strengthened) | The functional round trip passes in the full suite. Staff and superusers can now create too (E1), which D-01 requires and which was broken before 39-05 |
| R3 | SC3: no create target for an anonymous visitor; the pop-up is readable | ✓ VERIFIED | calendar.html is unchanged since cf76780; the anonymous card and inert-cell tests pass in the full suite; UAT Test 1 passed |
| R4 | SC4: the header names every block that differs from 3.1.0 | ✓ VERIFIED (advisory WR-02) | 10 regions map to items 1-6, and item 4 is updated for the gated elif. `test_every_differing_region_is_listed_and_every_item_differs` passes |
| A1-A11 | 39-01 route, method, htmx, signed-in, shadowing, boundary, precision and idempotency truths | ✓ VERIFIED | Files unchanged; tests pass |
| A7 | CSRF: a signed-in POST without a token is refused and creates nothing | PASSED (override) | Accepted by Tim Lister on 2026-10-08T17:04:52Z; the test is unchanged |
| A12 | Concurrency edge (backstop): the guard keeps no state | ✓ VERIFIED (human-accepted) | 39-UAT Test 2 pass; calendar_access.py unchanged since cf76780 |
| B1-B14 | 39-02 presentation, card, URL, XSS, BS5, browser and idempotency truths | ✓ VERIFIED | test_calendar_template passes in the full suite; the `<form>` block is unchanged since cf76780 (B6 as adjusted by 39-04) |
| C1-C5 | 39-03 suite, quality gates, vendored package untouched, ledgers | ✓ VERIFIED | E9 re-runs the suite, ruff and RECORD checks; the ledgers are not regressed |
| D1 | Paired doc describes the logged-out write path truthfully | ✓ VERIFIED (strengthened by E8) | — |
| N1-N13 | 39-04 CSRF-failure path tests, docstrings, CR-01 records, label and pinned snapshot | ✓ VERIFIED | The AnonymousCsrfFailureWriteTest tests pass; `test_replaying_the_refused_path_as_a_signed_in_get_changes_nothing` still expects 200 for create-event and update-event (lines 352-353), which is the page the new runbook sentence describes |

**Score:** 58/58 truths verified. That is 49 carried from the previous report plus 9 new (E1-E9). It includes 1 passed by override (A7) and 1 accepted by a human at UAT (A12). None is present-but-behavior-unverified.

### Prohibitions (39-05)

| Prohibition | Tier | Disposition |
|-------------|------|-------------|
| No change to calendar.html, the `<form>` block, calendar_urls.py, calendar_access.py, settings.py or urls.py | test | ✓ `git diff --stat cf76780 HEAD` is empty for all of them; form block identical |
| No change to campaign_attribution.py or calendar_display_extras.py | test | ✓ empty diff |
| Snapshot not regenerated without header item 4 in the same commit; no tom_calendar edit; no fourth override | test | ✓ both are in 917a895; RECORD 0 mismatches; 3 partials |
| No existing test changed; no skip, tag-out or expected failure | test | ✓ AST check: only setUpTestData changed, by appending 2 statements; full suite has no skips |
| No runbook text outside the closing clause changed; 39-UAT/VERIFICATION/DISPOSITION/SECURITY and earlier plans not edited by 39-05 | test | ✓ the runbook diff is one hunk. 917a895 and c71b7b0 touch none of those files; the later ledger, security and verification edits came from the workflow's own steps (782c530, 798a433, bc0e22c, f373781, 1691f6e) |
| (earlier) 39-02 judgment-tier "nothing looks editable to a visitor" | judgment | ✓ resolved by human, 39-UAT Test 1 pass |

### Required Artifacts (39-05)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/templatetags/attribution_display_extras.py` | isinstance guard | ✓ VERIFIED | Line 52; docstring now true |
| `src/templates/tom_calendar/partials/event_form.html` | gated elif and header item 4 | ✓ VERIFIED | Lines 24-25 and 284; G-39-4 reason sentence in the comment |
| `solsys_code/tests/test_calendar_template.py` | six new tests | ✓ VERIFIED | Lines 922-998, all passing |
| `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff` | regenerated, 10 regions | ✓ VERIFIED | Gated elif at line 199 |
| `39-05-red-evidence-task1.json` | RED evidence | ✓ VERIFIED | Exit 1, failures before the fix |
| `docs/runbooks/telescope_runs_calendar.rst` | bare-form warning | ✓ VERIFIED | Lines 2604-2607 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| calendar.html | event_form.html | `hx-get="{% url 'calendar:create-event' %}"` (+ New Event and day cell) | ✓ WIRED | Lines 220 and 238; E1 shows 200 for staff |
| event_form.html | attribution_display_extras.py | tag call reached only through the action-first elif | ✓ WIRED | Line 284 elif, then the `{% high_band_attribution_candidates event %}` call |
| test_calendar_template.py | snapshot `.diff` | `EventFormHeaderMatchesUpstreamTest.SNAPSHOT` | ✓ WIRED | Line 2055 |
| runbook | test_calendar_write_access.py | the bare-form sentence describes the 200 for create-event and update-event | ✓ WIRED | Test lines 352-353 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| event_form.html hint | `attribution_candidates` | `high_band_attribution_candidates(event)` → `campaign_attribution.candidates_for_event` | Yes on the edit form (`test_staff_sees_high_band_hint_for_unlinked_event`); [] on create by design | ✓ FLOWING |
| event_form.html card and decorations | `event.*`, `campaign_decoration event` | upstream `update_event` | Yes | ✓ FLOWING (unchanged) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 39-05 targeted tests | `python manage.py test --noinput solsys_code.tests.test_calendar_template.EventModalAttributionHintTest solsys_code.tests.test_calendar_template.EventFormHeaderMatchesUpstreamTest` | Ran 17 tests, OK | ✓ PASS |
| Full local suite (run once) | `python manage.py test --noinput solsys_code --exclude-tag=ephemeris_segfault` | Ran 2247 tests in 478.474s, OK, exit 0 | ✓ PASS |
| WR-04: plain signed-in GET of the edit pop-up has no hint | scratch test outside the repo (subclass of EventModalAttributionHintTest) | real tree: Ran 1, OK | ✓ PASS (behaviour holds) |
| WR-04 mutation | the same run with a scratch settings module that puts in front a copy of event_form.html whose gate is mutated to `request.user.is_authenticated` | the scratch probe FAILs; all 17 repository tests in the two classes pass (the source-level snapshot test reads the repository file, so a mutation through the template loader escapes it) | confirms WR-04 |
| IN-06: Save and Edit create as staff, superuser and plain user | scratch test | 200, update form for the new event, no hint; OK | ✓ PASS |
| Ruff | `pre-commit run ruff` / `ruff-format --files` on the two Python files | Passed; tree unchanged | ✓ PASS |
| System, migrations and RECORD checks | `manage.py check`, `makemigrations --check --dry-run`, RECORD hashes | W005 only; No changes; 22/0 | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. The phase declares no probe scripts.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ACCESS-01 | 39-01, 39-03, 39-04, 39-05 | an anonymous POST to the five write routes changes nothing; a test per endpoint | ✓ SATISFIED | R1, A1-A12, N1-N6. The 39-05 runbook warning (E8) makes the CSRF-failure landing page accurately documented |
| ACCESS-02 | 39-02, 39-03, 39-05 | month-view create and update targets hidden from anonymous users | ✓ SATISFIED | R3, B1-B14; calendar.html unchanged. 39-05 restores the create path for staff (E1-E3) and changes nothing for anonymous visitors |
| WARN-01 | 39-02, 39-03, 39-04, 39-05 | the event_form.html header states accurately which blocks differ | ✓ SATISFIED | R4, N11-N13, E7. Advisories WR-02 and WR-03 are about what the test enforces, not about the list itself |

No orphaned requirements: REQUIREMENTS.md lines 90-92 map exactly ACCESS-01, ACCESS-02 and WARN-01 to Phase 39, and plans claim all three.

### Code Review Round 3 (39-REVIEW.md, bc0e22c) — Classification

| Finding | Classification | Reasoning |
|---------|----------------|-----------|
| **WR-04**: no behavioural test keeps the edit-form hint from a signed-in non-staff user | **Human decision** (human item 3; recommended: add the test) | Confirmed. The string "Possible campaign run match" appears in tests only in test_calendar_template.py, and no case there is a signed-in non-staff user on the edit form. My mutation run reproduces the reviewer's result: with the gate weakened to `is_authenticated`, all 17 repository tests in the two classes stay green. **Not a must-have gap**: ACCESS-02 is about anonymous visitors, and no 39-05 truth or prohibition claims non-staff coverage (E6 lists anonymous, non-candidate and record-backed cases). The behaviour is correct today (the scratch probe passes on the real tree). The gate already excluded non-staff users before this phase, without a test, since 27-07. **But** 39-05 rewrote that gate. T-39-26 in 39-SECURITY.md is marked closed against non-staff exposure on evidence that does not cover non-staff. SUMMARY D5 claims "non-staff ... still never see it" and cites no test for it. Under CR-01 every self-registered account is such a viewer. The fix is about 6 lines, so it should be added before shipping |
| **IN-06**: the staff Save and Edit create path is untested | 📋 Advisory, accepted | Works today (scratch probe: staff, superuser and plain user all get 200 with the update form). On that path the tag receives a real CalendarEvent. Coverage, not a defect |
| **IN-07**: runbook troubleshooting does not cover an empty pop-up | 📋 Advisory, accepted | The runbook says nothing false. The G-39-4 cause is fixed and has tests. The empty-pop-up symptom is only possible again if a future 5xx and the unhardened calendar.html (not requested at UAT) coincide. Phase 41 triage candidate |
| WR-01 | Resolved by documentation | Runbook lines 2604-2607 verified; AR-39-02 accepts the residual token-in-URL risk; the ledger marks it fixed (1691f6e) |
| WR-02 (carried) | 📋 Advisory | Header lines 9-12 unchanged; the list itself is exact |
| WR-03 (carried) | 📋 Advisory | pyproject.toml line 20 `tomtoolkit>=3.1.0`; installed 3.1.0 |
| IN-01..IN-05 (carried) | ℹ️ Info | Their files are unchanged since the previous report (calendar_access.py, calendar_urls.py and test_calendar_write_access.py show no diff since cf76780); the classifications are unchanged |
| CR-01 | Accepted risk | AR-39-01, unchanged |

Also surfaced by the 39-05 SUMMARY (not review findings) and accepted as follow-ups:

- `campaign_attribution.candidates_for_event` still raises on `''`, although its docstring says it never raises. It is unreachable from the template now.
- calendar.html still opens the modal on any response.

Both are recorded for Phase 41 triage.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| attribution_display_extras.py | 31, 48 | "placeholder" | ℹ️ Info | Refers to Django's invalid-variable placeholder; not a stub |
| telescope_runs_calendar.rst | 2604 | "todo" | ℹ️ Info | The calendar's todo feature; not a debt marker |

No TBD, FIXME or XXX appears in any line 39-05 added.

### Human Verification Required

1. **UAT Test 4 re-run as staff (39-05 Task 1 human-check).** Sign in on the dev server as a staff or superuser account. Click "+ New Event", then an empty day cell. Each should open the pop-up with the create form, not an empty box. Then open an existing unlinked entry that has a High-band candidate; it should still show the "Possible campaign run match" hint.
2. **UAT Test 3 re-check (G-39-3).** Read runbook lines 2596-2607 once and confirm the new bare-form sentences read as you asked at UAT.
3. **WR-04 decision.** Either add a test in which a signed-in non-staff user GETs the edit pop-up of `unlinked_event_with_candidate` and the test asserts the hint is absent (recommended before shipping, because T-39-26 is marked closed without it), or accept WR-04 as an advisory.

### Gaps Summary

No gaps remain.

- **G-39-4:** closed, and pinned by six new tests that were RED before the fix.
- **G-39-3:** closed, by a runbook-only change confined to the one paragraph.

Every earlier must-have still holds: the files the 39-05 prohibitions protect are byte-unchanged since cf76780, and the full suite passes. The previous report's three human items were resolved at UAT.

The remaining items are human items, not gaps:

- two end-of-phase confirmations (the staff browser re-run and the runbook sentence);
- the WR-04 ship decision. The behaviour it guards is correct today, but no repository test protects it.

---

_Verified: 2026-10-08T23:58:00Z_
_Verifier: Claude (gsd-verifier)_
