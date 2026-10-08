---
phase: 39-calendar-write-access
verified: 2026-10-08T21:11:50Z
status: human_needed
score: 48/49 must-haves verified
covered_files:
  - .planning/phases/39-calendar-write-access/39-01-PLAN.md
  - .planning/phases/39-calendar-write-access/39-01-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-02-PLAN.md
  - .planning/phases/39-calendar-write-access/39-02-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-03-PLAN.md
  - .planning/phases/39-calendar-write-access/39-03-SUMMARY.md
  - .planning/phases/39-calendar-write-access/39-04-PLAN.md
  - .planning/phases/39-calendar-write-access/39-04-SUMMARY.md
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/calendar_access.py
  - solsys_code/calendar_urls.py
  - solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff
  - solsys_code/tests/test_bootstrap5_rendering.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_write_access.py
  - src/fomo/settings.py
  - src/fomo/urls.py
  - src/templates/tom_calendar/partials/calendar.html
  - src/templates/tom_calendar/partials/event_form.html
covered_digest: "v3:sha256:c15fcc752d1521e801022c68b1283a32185ceccce8ce76a1b1d9b6edcb8a50d3"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "CSRF unchanged: a signed-in POST to calendar:create-event without a CSRF token through Client(enforce_csrf_checks=True) returns 403 and creates nothing (the guards do not exempt CSRF)."
    reason: "tom_common's Raise403Middleware rewrites every browser 403 into a 302 to login (next=/calendar/create/); the request is still refused and nothing is created, and the distinct next proves the CSRF layer refused it, not the guard"
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-08T17:04:52Z"
re_verification:
  previous_status: gaps_found
  previous_score: 33/36
  gaps_closed:
    - "Paired doc (CLAUDE.md paired-docs rule): the runbook's 'Not logged in, the calendar is read-only.' paragraph describes what happens to a logged-out write attempt truthfully"
    - "39-01 CSRF must-have (403 wording) -- closed by the accepted override above; test unchanged"
    - "Coincidental-reliance item (anonymous exact-Location assertions relied on the CSRF-disabled test client) -- precondition declared and the CSRF-failure path pinned by AnonymousCsrfFailureWriteTest"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "39-REVIEW WR-01 (post-39-04): on the CSRF-failure path a signed-in user lands, after login, on a bare stand-alone form fragment for create-event/update-event; its <form> has no method or action, so clicking Save without htmx sends a GET carrying the typed fields and csrfmiddlewaretoken in the query string. The runbook's 'only show a form' is accurate (verifier probe: the tag is <form hx-post=... hx-target=...>, no <html>, no htmx script; nothing is written) but does not warn the operator not to use that form."
    category: security
    reason: "Not a write path and not reachable by an anonymous visitor; the form block is upstream's markup (unchanged by this phase apart from the label), so the same bare fragment is reached by any signed-in direct visit to /calendar/update/<id>/ today. Resolve with a one-sentence runbook warning and/or method=\"post\" plus action on the FOMO-owned form (one header item and snapshot update)."
    evidence_status: "reproduced by verifier probe (form tag, no page shell); token-in-URL consequence follows from HTML default form method"
  - finding: "39-REVIEW WR-02 (post-39-04): event_form.html's header says the pinned snapshot makes 'any new difference fail until this list and that file are updated together'; regenerating the snapshot alone turns the test green for a line inside an already-anchored region, so the header list itself is not enforced on that path."
    category: other
    reason: "The six-item list of differences is accurate today (10 snapshot regions all map to items 1-6) and the goal's truthfulness about HOW the override differs holds; this sentence overstates what the drift test enforces. Resolve by rewording to 'fails until that file is regenerated; review it against this list', or by binding a header hash into the snapshot."
    evidence_status: "statically evident from test_calendar_template.py (current_diff regenerates from the body only); not executed"
  - finding: "39-REVIEW WR-03: the snapshot is 'vs tomtoolkit 3.1.0' by name, but the test diffs against the installed tomtoolkit and pyproject requires tomtoolkit>=3.1.0 unpinned."
    category: other
    reason: "Fails closed (an upstream change turns CI red) with a misleading message and remedy; installed version here is 3.1.0, so the header's comparison is true today. Resolve with an installed-version assertion and an upgrade-specific message, or pin tomtoolkit."
    evidence_status: "statically evident (pyproject.toml, _upstream_lines reads the installed package)"
human_verification:
  - test: "Judgment-tier prohibition (39-02): 'MUST NOT show an anonymous visitor anything that looks editable or tells them how to get write access'. Open /calendar/ logged out, click an entry and hover over day cells."
    expected: "Nothing looks editable and nothing prompts a login. Decide whether the day-cell hover tint (.cal-day:hover, IN-03 of the first review) is acceptable for visitors or should be scoped to logged-in users. Non-authoritative LLM verdict: satisfied (verifier probe: anonymous card has no form/input/select/textarea/hx-post/write URL; anonymous month view's only control is the utc_offset display select, an hx-get to /calendar/; no '+ New Event')."
    why_human: "Visual affordance judgement; judgment-tier prohibitions need explicit human resolution and are never passed silently."
  - test: "Backstop truth (39-01 concurrency edge, A12): the guard keeps no state between requests, so concurrent or interrupted anonymous requests are each refused independently."
    expected: "Accept the structural evidence (calendar_access.py code AST-identical to 798dfe9; module level holds only a tuple of string constants; no global/nonlocal) as sufficient, or ask for a concurrency test."
    why_human: "Marked verification: backstop (non-inferable); presence and structure never qualify on their own, and no concurrent-request test exists."
  - test: "39-04 Task 1 human-check: read the rewritten runbook paragraph (docs/runbooks/telescope_runs_calendar.rst lines 2576-2605) once."
    expected: "It reads as plain operator guidance, names both refusal paths without jargon beyond 'CSRF check', and states the self-registration acceptance with its quoted reason. (39-04-SUMMARY reports this was approved at the Task 1 checkpoint; confirm, and decide at the same time whether to add the WR-01 'do not use the bare form' sentence and the IN-02 login-page flash note.)"
    why_human: "Readability and tone of operator documentation."
---

# Phase 39: Calendar Write Access Verification Report

**Phase Goal:** The public calendar is read-only to anyone not logged in — an anonymous visitor can neither create, change nor delete an event, and is not offered a control that would try — and FOMO's event pop-up override says truthfully how it differs from tomtoolkit 3.1.0's upstream template.
**Verified:** 2026-10-08T21:11:50Z
**Status:** human_needed
**Re-verification:** Yes — after gap-closure plan 39-04 (commits bc73bfd, 65ba57c, a8548ea, b1fd597)

The goal is achieved in the code. Both gaps from the previous pass are closed:

- **Gap 1:** the runbook paragraph now describes both refusal paths, and new tests pin the path a write takes when it fails the CSRF check, for all five routes. I also ran my own probe of the "tab left open after logging out" example: it logs out through the real logout view, and that tab does pass the CSRF check and reaches the guard.
- **Gap 2:** closed by the override Tim Lister accepted.

As instructed, CR-01 (open self-registration) is now on record as an accepted risk and is not treated as a gap.

None of the eight new review findings blocks the goal; three are kept as advisories. The phase still needs a human for three items: the judgment-tier visitor-affordance prohibition, the backstop concurrency truth, and the 39-04 readability check on the runbook paragraph. That is why the status is `human_needed` and not `passed`.

## Goal Achievement

### Re-checked gaps from the previous report

| # | Previous gap | Now | Evidence |
|---|--------------|-----|----------|
| D1 | Runbook paragraph claimed every refused logged-out write returns to the calendar page | ✓ VERIFIED | Lines 2596-2605 now name two paths: a write that passes the CSRF check returns to the calendar page, while one that fails it is sent to login with the refused address. The old clause "never to the refused write" is gone, and so is the "script" example from that sentence. The 39-04 Task 1 structural check, re-run by me, printed OK. The first path's example (a tab left open after logging out) was checked by my scratch probe: login through Client(enforce_csrf_checks=True), read the pop-up's token, POST to the `logout` view, then POST to all 5 routes with the old token. The CSRF cookie was kept, all 5 got 302 → `/accounts/login/?next=/calendar/`, and no row changed. The second path is pinned by AnonymousCsrfFailureWriteTest (below). |
| A7 | 39-01 CSRF must-have said "returns 403" | PASSED (override) | Override: Raise403Middleware rewrites the 403 into a 302 to login; the request is refused and nothing is created. Accepted by Tim Lister on 2026-10-08T17:04:52Z. `test_post_without_csrf_token_is_refused` is AST-identical to 798dfe9, and 39-01-PLAN.md is unchanged (`git diff 798dfe9` is empty). |
| A5 (advisory) | Coincidental reliance: the exact `next=/calendar/` was proven only through the CSRF-disabled client | ✓ VERIFIED | The class docstring of AnonymousCalendarWriteTest, the `login_url` and `assert_login_redirect` docstrings, and the module docstring all state the CSRF-passing precondition. The other path has its own tests. |

### Observable Truths

Roadmap success criteria (R), 39-01 (A), 39-02 (B), 39-03 (C), paired doc (D) and 39-04 (N).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| R1 | SC1: anonymous POST to create/update/delete-event is redirected or refused; a test per route proves the count and fields are unchanged | ✓ VERIFIED | AnonymousCalendarWriteTest (CSRF-passing) and AnonymousCsrfFailureWriteTest (CSRF-failing) both pass |
| R2 | SC2: a logged-in user can still create, update and delete from the month view | ✓ VERIFIED | The functional round-trip test, re-run in Chromium with the restored "Save and Edit" label: OK |
| R3 | SC3: an anonymous visitor sees no create target and can read the pop-up | ✓ VERIFIED | Functional anonymous-card and inert-cell tests re-run: OK. Verifier probe: the anonymous card has no form controls or write URLs, and the month view has no "New Event" |
| R4 | SC4: the event_form.html header names every block that differs from 3.1.0 | ✓ VERIFIED (advisory WR-02) | The 10 snapshot regions map to items 1-6 (Task 2 check re-run). Item 3 now says the labels match upstream. The sentence about what the test enforces overstates it (advisory) |
| A1-A4, A6, A8-A11 | 39-01 route, method, htmx, signed-in, shadowing, boundary, precision and idempotency truths | ✓ VERIFIED | Regression: test_calendar_write_access (30 tests) and test_urls pass; calendar_urls.py and src/fomo/urls.py are unchanged since 798dfe9 |
| A5 | The guard's next is the calendar page; a signed-in GET on a destructive route gives 405 | ✓ VERIFIED | As above, with the precondition now declared |
| A7 | CSRF: tokenless signed-in POST is refused and creates nothing | PASSED (override) | See the re-checked gaps table |
| A12 | Concurrency edge (backstop): the guard keeps no state | ? insufficient_spec | calendar_access.py code is AST-identical to 798dfe9; the backstop tier needs human acceptance (human item 2) |
| B1-B5, B7-B14 | 39-02 presentation, card, URL, XSS, BS5, browser and idempotency/concurrency truths | ✓ VERIFIED | test_calendar_template (97 tests) passes; functional tests re-run OK |
| B6 | Editor form unchanged | ✓ VERIFIED (deliberately adjusted by 39-04) | The `<form>` block differs from f929f4e in exactly the save_and_edit line, and only in the label (Task 2 check re-run: OK) |
| C1 | Full local suite OK, no skips | ✓ VERIFIED | Not re-run, per instruction. 39-04 Task 3 reports 2241 tests OK; the orchestrator reports the regression gate at 2196 + 40 OK. My targeted runs: 131 unit tests OK and 4 functional tests OK. The only "skip" matches in the phase's test modules are in prose |
| C2 | Quality gates | ✓ VERIFIED | `pre-commit run ruff` and `ruff-format` both Passed on the four phase Python files. The scratch probe's system check shows only urls.W005 |
| C3 | Vendored tom_calendar untouched | ✓ VERIFIED | 22 RECORD-hashed files, 0 mismatches; the partials directory holds exactly calendar.html, campaign_chip.html and event_form.html |
| C4-C5 | D-12 ledgers | ✓ VERIFIED | Unchanged since the previous pass (no regressions; the 39-04 commits do not touch them) |
| D1 | Paired doc: the runbook describes the logged-out write path truthfully | ✓ VERIFIED | See the re-checked gaps table |
| N1 | Tokenless anonymous POST, all 5 routes: 302 to `/accounts/login/?next=<own path>`, not next=/calendar/, and no row changes | ✓ VERIFIED | `test_tokenless_anonymous_post_is_sent_to_login_with_its_own_path` passes; it posts through `Client(enforce_csrf_checks=True)` |
| N2 | htmx variant: 200 with HX-Redirect to the same address, never 403 | ✓ VERIFIED | `test_tokenless_anonymous_htmx_post_gets_hx_redirect_to_its_own_path` passes |
| N3 | Replaying the address as a signed-in GET writes nothing: 405 ×3, 200 ×2 | ✓ VERIFIED | `test_replaying_the_refused_path_as_a_signed_in_get_changes_nothing` passes |
| N4 | Idempotency: repeated tokenless POSTs give identical Locations and leave rows unchanged | ✓ VERIFIED | `test_tokenless_repeat_posts_are_refused_identically` |
| N5 | Boundary: a missing id gives 302, not 404 | ✓ VERIFIED | `test_tokenless_post_to_missing_event_is_refused_not_404` |
| N6 | Precision: a single shared `snapshot()` in CalendarRowSnapshotMixin; existing anonymous tests AST-identical | ✓ VERIFIED | Task 1 AST check re-run: OK |
| N7 | Precondition declared; calendar_access.py docstring scoped ("guard's own"), names CsrfViewMiddleware and Raise403Middleware; code AST-identical | ✓ VERIFIED | Task 1 check re-run: OK. IN-01 (the HX-Redirect actually comes from HTMXRedirectMiddleware) is a small attribution slip and is information only |
| N8 | CR-01 recorded in the runbook with the verbatim rationale | ✓ VERIFIED | Lines 2586-2594; the verbatim string appears once |
| N9 | CR-01 recorded in 39-SECURITY.md (T-39-10 premise corrected, T-39-22, AR-39-01) | ✓ VERIFIED | Rows 46, 58 and 72; verbatim rationale present; `threats_open: 0` |
| N10 | CR-01 recorded in 39-REVIEW-DISPOSITION.md as skipped | ✓ VERIFIED | At a8548ea: `open: 6`, `total: 7`, row `\| CR-01 \| critical \| skipped \|` with the rationale. The later code-review re-render (38ff537) keeps CR-01 skipped with the same Source text and adds the 8 new findings (open 8 / total 9). This is the workflow's normal re-render, not drift |
| N11 | The label reads "Save and Edit"; the create form renders `>Save and Edit</button>`; item 3 and ANCHORS[3] updated | ✓ VERIFIED | event_form.html line 105; `test_signed_in_create_form_uses_upstream_button_labels` passes; the RED evidence was classified RED_EVIDENCE_OK |
| N12 | Pinned normalized diff (10 regions); an unlisted line inside an anchored region fails a test | ✓ VERIFIED | `test_body_diff_matches_pinned_snapshot` and `test_snapshot_detects_an_unlisted_line_inside_an_anchored_region` pass; the snapshot has 10 `@@ ` lines |
| N13 | Header list, snapshot and installed upstream agree; the header names the snapshot file | ✓ VERIFIED | Task 2 check re-run: OK; tomtoolkit 3.1.0 installed |

**Score:** 48/49 truths verified (including 1 passed by override; 1 insufficient_spec routed to human; 0 present-but-behavior-unverified)

### Prohibitions

| Plan | Prohibition | Tier | Disposition |
|------|-------------|------|-------------|
| 39-01 | No edit/patch of tomtoolkit; no re-implemented view body | test | ✓ RECORD hashes; calendar_access.py code unchanged |
| 39-01 | No 403/error page for anonymous writes | test | ✓ guard path 302/HX-Redirect; CSRF path 302/HX-Redirect (N1, N2, with `assertNotEqual(403)`) |
| 39-01 | The login next must not be the refused write URL | test | ✓ for the guard's own redirect (the prohibition's subject). The CSRF-failure path, which is outside the guard, does set next to the refused path. That path is now documented, and test N3 proves a replay of that path writes nothing. This replaces the earlier flag |
| 39-01 | No narrowing beyond "logged in" | test | ✓ guard code unchanged; CR-01 accepted rather than fixed |
| 39-02 | Nothing looks editable or prompts a login for visitors | judgment | ⚠ flagged, human item 1 (non-authoritative verdict: satisfied) |
| 39-02 | No internal identifiers, no duplicated blocks, no third override | test | ✓ unchanged since the last pass; still exactly 3 partials |
| 39-03 | No test-only login hook; ledgers; no skip/expectedFailure | test | ✓ |
| 39-04 | No behaviour change (no CSRF_FAILURE_VIEW, no edit to settings, the URL confs, calendar.html, or calendar_access.py code) | test | ✓ `git diff 798dfe9` for those files is empty; AST check passes |
| 39-04 | No change to TOM_REGISTRATION_STRATEGY, D-01 or the guard | test | ✓ settings.py line 280 is still `'open'`; guard code identical |
| 39-04 | No edit to 39-01-PLAN.md; no 403 re-asserted; CSRF test unchanged | test | ✓ |
| 39-04 | Rationale not paraphrased; no other ledger finding changed at a8548ea | test | ✓ verbatim string in all three records; a8548ea changes exactly the CR-01 lines and `open:` |
| 39-04 | No tom_calendar edit, no third override, no skip; CSRF tests use the enforcing client | test | ✓ the AnonymousCsrfFailureWriteTest source has `enforce_csrf_checks=True` and no `self.client.post` |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_calendar_write_access.py` | CalendarRowSnapshotMixin and AnonymousCsrfFailureWriteTest | ✓ VERIFIED | 5 new tests, all passing; the mixin holds the only `snapshot()` |
| `solsys_code/calendar_access.py` | docstring scoped to the guard; CSRF path named | ✓ VERIFIED | Lines 8-19; code unchanged |
| `docs/runbooks/telescope_runs_calendar.rst` | both paths described; self-registration acceptance | ✓ VERIFIED | Lines 2586-2605 |
| `src/templates/tom_calendar/partials/event_form.html` | upstream label; item 3; snapshot named | ✓ VERIFIED | Lines 9-21 and 105 |
| `solsys_code/tests/test_calendar_template.py` | label test, pinned-snapshot test and non-vacuity test | ✓ VERIFIED | Module passes |
| `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff` | pinned diff | ✓ VERIFIED | Tracked; 10 regions; `+...>Save and Edit</button>` |
| `39-SECURITY.md` / `39-REVIEW-DISPOSITION.md` | CR-01 accepted | ✓ VERIFIED | See N9 and N10 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_calendar_write_access.py | tom_common Raise403Middleware | `Client(enforce_csrf_checks=True)` | ✓ WIRED | Middleware line 88 builds `reverse('login') + '?next=' + request.path`; the tests assert exactly that |
| runbook | test_calendar_write_access.py | "fails the CSRF check" / "passes the CSRF check" | ✓ WIRED | Each runbook path corresponds to one test class's assertions |
| test_calendar_template.py | the snapshot `.diff` | `EventFormHeaderMatchesUpstreamTest.SNAPSHOT` | ✓ WIRED | The test reads it, and it matches `current_diff()` |
| 39-SECURITY.md | 39-REVIEW-DISPOSITION.md | verbatim rationale | ✓ WIRED | Exact string in both |
| Previous links (calendar_urls → calendar_access, the include order, `request.user` branches) | — | — | ✓ WIRED | Files unchanged since 798dfe9; tests pass |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| event_form.html card | `event.*`, `event.todos.all` | upstream `update_event` GET | Yes | ✓ FLOWING (unchanged) |
| event_form.html decorations | `campaign_decoration event` | CalendarEventMeta.run | Yes | ✓ FLOWING (unchanged) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase unit modules | `python manage.py test --noinput solsys_code.tests.test_calendar_write_access solsys_code.tests.test_calendar_template solsys_code.tests.test_urls` | Ran 131 tests, OK | ✓ PASS |
| Browser proofs | `--tag functional`: editor round trip, anonymous card, inert day cell, New Event modal | Found 4, OK | ✓ PASS |
| 39-04 Task 1 structural check | plan script, re-run | `OK: docstring-only guard change, behaviour files untouched, ...` | ✓ PASS |
| 39-04 Task 2 structural check | plan script, re-run | 10 region headers; `OK: one form line differs ...` | ✓ PASS |
| Stale tab after logout (the runbook's first path, review IN-03) | scratch test outside the repo: enforcing client, force_login, read token, POST to `logout`, POST to all 5 routes with the old token | CSRF cookie kept; 5 × 302 → `/accounts/login/?next=/calendar/`; rows unchanged | ✓ PASS (note: Django's test `Client.logout()` clears all cookies and gives a false CSRF-path result; only the real logout view reproduces a browser) |
| WR-01 landing page | scratch: signed-in GET of create-event | `<form hx-post="/calendar/create/" hx-target="#calendar-partial">`; no `<html>`, no htmx script | facts confirmed; nothing written |
| Anonymous surfaces | scratch: anonymous GET of the month view and pop-up | card: no form/input/hx-post/write URL; month: only the `utc_offset` display select (hx-get /calendar/), no "New Event" | ✓ PASS |
| Vendored integrity | RECORD hash check | 22 files, 0 mismatches | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. The phase declares no probe scripts, and no `scripts/*/tests/probe-*.sh` is relevant.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ACCESS-01 | 39-01, 39-03, 39-04 | an anonymous POST to any of the five write routes changes nothing; a test per endpoint | ✓ SATISFIED | R1, A1-A11, N1-N6. Both refusal paths are now covered by tests. Who may write is set by D-01; self-registration is an accepted risk (CR-01) |
| ACCESS-02 | 39-02, 39-03 | month-view create/update targets hidden from anonymous users | ✓ SATISFIED | R3, B1-B14 |
| WARN-01 | 39-02, 39-03, 39-04 | the event_form.html header states accurately which blocks differ | ✓ SATISFIED | R4, N11-N13. The test-enforcement sentence overstates what the test enforces (advisory WR-02) |

No orphaned requirements: REQUIREMENTS.md maps exactly ACCESS-01, ACCESS-02 and WARN-01 to Phase 39, and plans claim all three.

### Code Review Findings after 39-04 (39-REVIEW.md) — Classification

| Finding | Classification | Reasoning |
|---------|----------------|-----------|
| WR-01: the CSRF-path landing page is a bare form whose Save sends a GET with the token | 📋 Advisory (security, follow-up) | I confirmed the facts. The finding does not affect the goal: no anonymous write is possible, and on the CSRF path no write is possible either. "Only show a form" is literally true. The exposure needs a logged-in user to click Save on an unstyled fragment, and that fragment is upstream's form markup, which this phase did not introduce; any signed-in direct visit to `/calendar/update/<id>/` reaches the same page. A one-sentence runbook warning or `method="post"` would close it. Worth doing before shipping, but it does not block the phase |
| WR-02: the header overstates the snapshot guard | 📋 Advisory (follow-up) | The goal asks for a truthful account of how the override differs, and the six-item list is exact. This sentence is about the drift check, and it overclaims for one case (regenerating the snapshot without touching the list). One-line reword |
| WR-03: the 3.1.0 snapshot is checked against an unpinned upstream | 📋 Advisory (follow-up) | The check fails closed; the message is misleading. It is true for the installed 3.1.0. A good candidate for Phase 41 triage |
| IN-01: HX-Redirect attributed to Raise403Middleware | ℹ️ Info | HTMXRedirectMiddleware (middleware.py lines 100-117) does the rewrite. The behaviour described is right; the middleware named is not |
| IN-02: runbook omits the login-page flash on the CSRF path | ℹ️ Info | 39-04 chose not to document it, and the reason is on record in its flagged-assumptions table. Can be added together with the WR-01 sentence |
| IN-03: no CSRF-enforcing test for the stale-tab example | ℹ️ Info | The behaviour holds today (verifier probe above), but no repository test pins it. Adding the probe as a test would guard against a future `CSRF_USE_SESSIONS = True` |
| IN-04: tests use `settings.LOGIN_URL`, code uses `reverse('login')` | ℹ️ Info | Both are `/accounts/login/` today. Only test robustness is affected |
| IN-05: calendar_urls.py docstring has the one-path wording | ℹ️ Info | It describes the guard's own redirect, which is true. It is a code docstring, not the paired runbook, and 39-04 deliberately left the file untouched |
| CR-01: open self-registration | Accepted risk (not a gap) | Developer decision recorded in the runbook, AR-39-01 / T-39-22 and the ledger, with the rationale verbatim |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/templates/tom_calendar/partials/event_form.html | 245 | "TBD" | ℹ️ Info | Describes an undated run ("a TBD run"); pre-existing, not a debt marker |

No TODO, FIXME, XXX, placeholder or stub patterns appear in the lines 39-04 added.

### Human Verification Required

1. **Visitor affordance (judgment-tier prohibition, 39-02).** Open /calendar/ logged out, click an entry and hover over the day cells. Confirm that nothing looks editable or prompts a login, and decide whether the `.cal-day:hover` tint should apply only to logged-in users.
2. **Concurrency backstop truth (A12).** Accept the structural evidence (guard code AST-identical, no module state) or ask for a concurrency test.
3. **Runbook paragraph readability (39-04 Task 1 human-check).** Read lines 2576-2605. The SUMMARY reports this was approved at the Task 1 checkpoint. At the same time, decide whether to add the WR-01 sentence ("the create and edit addresses show a bare copy of the form; go back to the calendar page instead") and the IN-02 sentence about the flash message.

### Gaps Summary

No gaps remain.

- **Gap 1 (paired runbook):** closed. The paragraph now separates the guard's refusal (returns to the calendar page) from the CSRF-failure refusal (returns to the refused address, where a GET writes nothing). Tests now pin the CSRF-failure path for every route: plain, htmx, repeated, missing id, and replay. My probe also confirmed the first path's real-browser example.
- **Gap 2 (the 403 wording):** closed by the accepted override.

The new review raises three warnings, and none of them touches the goal's two halves:

- No anonymous request writes or is offered a write control.
- The header's list of differences is exact.

They are recorded as advisories for a follow-up. WR-01's one-sentence runbook warning is the most worthwhile of them before shipping.

The status is `human_needed` only because of the judgment-tier prohibition, the backstop truth and the planned readability check.

---

_Verified: 2026-10-08T21:11:50Z_
_Verifier: Claude (gsd-verifier)_
