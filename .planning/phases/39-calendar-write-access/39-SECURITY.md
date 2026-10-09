---
phase: "39"
slug: "calendar-write-access"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-08"
---

# Phase 39 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| browser (anonymous or signed-in) -> Django URL conf | every request to /calendar/... is untrusted input; the session cookie decides request.user | HTTP method, form fields, session cookie |
| FOMO URL conf -> vendored tom_calendar views | the upstream views trust their caller completely (no login, no method check); FOMO's wrapper is the only gate | CalendarEvent / EventTodo writes |
| login page -> post-login redirect | the `next` parameter decides which URL the browser GETs after logging in | redirect target |
| server-rendered HTML -> anonymous browser | everything in the month partial and the pop-up is published to anyone; it must carry no write control and no internal identifier | event fields, todo text, attributed-run link |
| CalendarEvent field values -> HTML | titles, descriptions and URLs come from pipelines and from any logged-in user, so they are untrusted when rendered | user/pipeline strings |
| installed upstream template -> FOMO override | the override's header is the only record of how FOMO's copy differs; a wrong record hides drift on the next upgrade | template provenance |
| test harness -> live server | the browser test logs in by copying a force_login session cookie; nothing in production code knows about tests | session cookie (test only) |
| ledgers -> later verifiers and Phase 42 re-verification | a ledger row marked fixed is read as fact by later phases | review dispositions |
| installed tomtoolkit package -> FOMO | the guard's correctness assumes the vendored views were not edited in place | site-packages integrity |
| signed-in staff browser -> calendar:create-event | a staff viewer's GET or invalid POST renders event_form.html with no `event`, which reaches the staff-only hint branch (39-05) | form fields, `action`, missing `event` |
| template layer -> attribution matcher | the hint tag hands the template's `event` value to an ORM lookup and to campaign_attribution (39-05) | template context value |
| staff-only hint -> non-staff viewers | the hint can name a candidate run that is not yet publicly visible, so who sees it must not widen (39-05) | candidate run names |
| operator docs -> operators | the runbook is read as fact about what the post-login create and edit pages do (39-05) | operator behaviour |
| signed-in non-staff browser -> calendar:update-event (GET) | under D-01 and D-04 every signed-in account, self-registered ones included (AR-39-01), opens the edit pop-up, which reaches the staff-only hint branch; only its `is_staff` conjunct keeps the hint from that viewer (39-06) | request.user flags, `action`, `event` |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-39-01 | Tampering | anonymous POST to create-event / update-event / delete-event creates, changes or deletes a CalendarEvent | high | mitigate | `write_requires_login` / `read_open_write_requires_login` wrap all five routes in `solsys_code/calendar_urls.py` (6 refs); `AnonymousCalendarWriteTest` asserts 302 to login and unchanged rows per route (commits 5e31c0f, 4d32e62) | closed |
| T-39-02 | Tampering | anonymous POST to create-todo / update-todo adds or changes an EventTodo (D-02) | high | mitigate | same guards on both todo routes; 6 todo tests in `test_calendar_write_access.py` assert count/description/is_completed unchanged | closed |
| T-39-03 | Tampering | GET/HEAD-driven mutation: upstream delete_event, create_todo, update_todo act on any method | high | mitigate | `require_POST` inside the guard on delete-event, create-todo, update-todo (5 refs in calendar_urls.py); signed-in GET -> 405 and anonymous GET -> 302 tests | closed |
| T-39-04 | Tampering (CSRF) | login replay: a `next` of the write URL makes the post-login GET perform the refused delete or todo wipe | high | mitigate | `next` is always `reverse('calendar:calendar')` (calendar_access.py:35); test asserts Location exactly `/accounts/login/?next=/calendar/`; destructive routes are POST-only | closed |
| T-39-05 | Tampering (CSRF) | a signed-in session's POST forged from another site | medium | mitigate | guards do not exempt CSRF; `Client(enforce_csrf_checks=True)` test — refused (tom_common's Raise403Middleware turns the 403 into a login 302) and nothing created | closed |
| T-39-06 | Elevation of privilege | the unguarded upstream copy of the routes mounted by tom_common.urls under the same 'calendar' namespace | high | mitigate | `src/fomo/urls.py:30` includes `solsys_code.calendar_urls` before `tom_common.urls` (line 35); `CalendarUrlConfShadowingTest` resolves each literal path to FOMO's guarded callable | closed |
| T-39-07 | Spoofing | a stale-tab htmx save swapping the login page into the modal, or a 403 page that tells the visitor nothing | low | mitigate | HX-Request writes get 200 + `HX-Redirect` (tom_common HTMXRedirectMiddleware), asserted per route; never 403 | closed |
| T-39-08 | Spoofing (open redirect) | `next` taken from request data sends the user off-site after login | low | mitigate | calendar_access.py reads no request.GET/POST for `next` (0 refs); the value is a fixed reverse() | closed |
| T-39-09 | Information disclosure | an anonymous write to a missing id answered with 404 reveals which event ids exist | low | mitigate | guard runs before the object lookup; test: anonymous POST to /calendar/delete/999999/ is a 302 to login | closed |
| T-39-10 | Elevation of privilege | any logged-in user may edit or delete any event, including other users' and pipeline-created ones | medium | accept | accepted risk AR-39-01 (D-01, D-03): no staff/permission/ownership check by decision; guard has 0 refs to is_staff/has_perm; premise corrected 2026-10-08 (39-REVIEW CR-01): TOM_REGISTRATION_STRATEGY = 'open' lets any member of the public self-register and log in, so 'any logged-in user' includes self-registered accounts (see T-39-22); re-accepted by Tim Lister with no behaviour change | closed |
| T-39-11 | Information disclosure | anonymous pop-up rendering the full form: every TargetList name, widget values and a CSRF token | medium | mitigate | `request.user.is_authenticated` branch renders `cal-event-card`; `test_anonymous_card_has_no_form_controls` asserts no form/input/select/textarea/button/hx-post/csrfmiddlewaretoken (commit 8d0eca5) | closed |
| T-39-12 | Information disclosure | internal ALLOC:/RUN: namespace keys echoed on the anonymous card | low | mitigate | `EventCardUrlLinkTest`: namespace keys and javascript: values render only `(not a web link)`, raw value absent (9 asserts) | closed |
| T-39-13 | Tampering (stored XSS) | event title, description or url rendered unescaped on the card | medium | mitigate | Django autoescape kept; XSS test asserts `&lt;b&gt;` / `&lt;script&gt;` and no raw tags | closed |
| T-39-14 | Information disclosure | observation-group name or staff-only candidate run reaching an anonymous visitor | medium | mitigate | `EventModalSeriesDecorationTest` and `EventModalAttributionHintTest` pass unmodified; series/campaign blocks rendered once below both branches | closed |
| T-39-15 | Spoofing | the public month view advertising create/edit controls that the server refuses | low | mitigate | `CalendarMonthViewReadOnlyTest` (no /calendar/create/ URL, no '+ New Event'); Playwright `test_anonymous_visitor_opens_read_only_event_card_with_no_page_errors` + inert day cell (commit 99b60b1) | closed |
| T-39-16 | Repudiation | the event_form.html header misstating how FOMO's copy differs from upstream | low | mitigate | six-item header pinned to tomtoolkit 3.1.0; `EventFormHeaderMatchesUpstreamTest` diffs the body against the installed file (commit 9b2df4d) | closed |
| T-39-17 | Elevation of privilege | a test-only login shortcut leaking into production to make the browser test pass | medium | mitigate | login only via the test-side `_log_in_browser` force_login cookie hand-off in test_bootstrap5_rendering.py; settings.py and src/fomo/urls.py unchanged since f929f4e | closed |
| T-39-18 | Repudiation | a ledger marked fixed without evidence, or another finding's disposition changed by accident | low | mitigate | commit 798dfe9 touches exactly 33-REVIEW.md (+4/-0) and 37.1-REVIEW-DISPOSITION.md (+2/-2), WR-05 only, written after the 2233-test gate was green | closed |
| T-39-19 | Tampering | the installed tom_calendar package edited in place | medium | mitigate | 39-03 verified all 22 hashed tom_calendar files against tomtoolkit 3.1.0's RECORD sha256; no site-packages path in the phase diff | closed |
| T-39-20 | Repudiation | the runbook's read-only paragraph and calendar_access.py's module docstring claim every refused logged-out write returns to the calendar page, so an operator misreads where a tokenless refusal lands (39-REVIEW WR-01) | low | mitigate | both texts rewritten in 39-04 Task 1 (commit bc73bfd) to name the two refusal paths (runbook "fails the CSRF check" paragraph; calendar_access.py:15-19 names `CsrfViewMiddleware` and `AnonymousCsrfFailureWriteTest`); `AnonymousCsrfFailureWriteTest` pins `next=<refused path>` per route; the Task 1 structural check ties the phrases to the tested behaviour and the guard code is AST-identical to 798dfe9 | closed |
| T-39-21 | Tampering (login replay via the CSRF-failure path) | a tokenless write (script, forged cross-site post) gets `next=<the refused write path>`; after logging in, the browser GETs it | medium | mitigate | `require_POST` (405) unchanged on delete-event, create-todo, update-todo and render-only GET on create-event, update-event; `test_replaying_the_refused_path_as_a_signed_in_get_changes_nothing` (test_calendar_write_access.py:349) asserts no row changes for all five routes; tokenless repeat and missing-id POSTs write nothing (lines 320, 334); non-vacuity run with CSRF enforcement off FAILED 22 | closed |
| T-39-22 | Elevation of privilege | open self-registration (TOM_REGISTRATION_STRATEGY = 'open', src/fomo/settings.py): any member of the public can create an account, is logged in at once, and passes both calendar guards (39-REVIEW CR-01) | high | accept | accepted risk AR-39-01 (corrected 2026-10-08, Tim Lister) by developer sign-off, where ASVS L1 would otherwise mitigate a high threat; recorded in the runbook's read-only paragraph and in 39-REVIEW-DISPOSITION.md (CR-01 skipped); registration setting, D-01 and the guard unchanged | closed |
| T-39-23 | Repudiation | event_form.html's header omits a difference from upstream (the button label case), and `EventFormHeaderMatchesUpstreamTest` cannot see an unlisted line inside a region that already holds an anchor (39-REVIEW WR-02) | low | mitigate | 39-04 Task 2 (commit 65ba57c) restores upstream's "Save and Edit" label (event_form.html:107), lists it in header item 3 and `ANCHORS[3]` (test_calendar_template.py:1956), pins the full normalized body diff in `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff` (10 regions; `test_body_diff_matches_pinned_snapshot`) and proves with `test_snapshot_detects_an_unlisted_line_inside_an_anchored_region` that an unlisted line now fails; RED evidence in 39-04-red-evidence-task2.json | closed |
| T-39-24 | Repudiation | the accepted-risk record drifts: the rationale paraphrased, another ledger finding changed, or the CR-01 entry inconsistent between front matter and table | low | mitigate | 39-04 Task 3 records check printed OK against a saved pre-task copy; commit a8548ea touches exactly 39-REVIEW-DISPOSITION.md (+3/-3) and 39-SECURITY.md (+3/-2); the verbatim rationale is present in the runbook (lines 2592-2594), AR-39-01 / T-39-22 here and the CR-01 ledger row | closed |
| T-39-25 | Denial of service | GET /calendar/create/ and an invalid create POST for any staff or superuser viewer: the unguarded `high_band_attribution_candidates` raised ValueError on the create form's missing `event`, so the New Event pop-up was a 500 shown as an empty box (G-39-4; latent since 27-07, not a Phase 39 regression) | medium | mitigate | 39-05 Task 1 (commit 917a895): `if not isinstance(event, CalendarEvent): return []` is the tag's first statement (attribution_display_extras.py:52) and the staff-hint elif tests `action == "update"` first (event_form.html:284); `test_staff_and_superuser_get_the_create_form`, `test_staff_create_form_matches_the_plain_users_apart_from_the_csrf_token`, `test_staff_invalid_create_post_re_renders_the_form` and `test_attribution_tag_returns_empty_list_for_a_non_event` (test_calendar_template.py:922-978) were RED before the fix (39-05-red-evidence-task1.json, RED_EVIDENCE_OK) and are green | closed |
| T-39-26 | Information disclosure | the staff-only "Possible campaign run match" hint can name a run not yet publicly visible; changing its gate must not show it to anonymous or non-staff viewers nor drop it for staff on the edit form | medium | mitigate | the elif keeps `request.user.is_staff` (event_form.html:284) with only the action test added in front; `test_anonymous_does_not_see_hint` (test_calendar_template.py:891) and the other pre-existing `EventModalAttributionHintTest` tests are AST-identical and pass; `test_hint_is_gated_on_the_edit_form` (:979) proves update still shows it and `test_superuser_sees_high_band_hint_for_unlinked_event` (:994) covers superusers; the plain-user equivalence test (:940) proves no staff-only text reaches the create form | closed |
| T-39-27 | Information disclosure | the bare copy of the event form shown at the create or edit address after a CSRF-failure login: its Save is a native GET that carries the CSRF token in the query string (browser history, server logs) and silently drops the input (39-REVIEW WR-01) | low | accept | accepted risk AR-39-02: 39-05 Task 2 (commit c71b7b0) tells operators not to use the copy and to go back to the calendar page (telescope_runs_calendar.rst:2605-2607); the template `method="post"` hardening (39-REVIEW WR-01 item 2) was offered and not requested by the developer at UAT on 2026-10-08, so the residual token-in-URL risk if someone uses it anyway is accepted; not reachable by an anonymous visitor and not a write path | closed |
| T-39-28 | Repudiation | the runbook's closing clause presented the post-login create and edit pages as harmless forms, so an operator types a change there and believes it saved | low | mitigate | 39-05 Task 2 (commit c71b7b0, runbook only: +4/-2) rewrote the clause to say the addresses show a bare, unstyled copy of the event form, that its Save saves nothing and silently discards what was typed, and to go back to the calendar page (rst:2605-2607); the Task 2 structural check requires those phrases, forbids the old clause and proves no other text in the runbook or docs/ changed | closed |
| T-39-29 | Repudiation | event_form.html changes without its header list and pinned snapshot, so the record of how FOMO's copy differs from tomtoolkit 3.1.0 drifts (D-10, WARN-01) | low | mitigate | header item 4 now reads "edit form only" (event_form.html:25) and the snapshot was regenerated in the same commit 917a895 (`+{% elif action == "update" ...` at event_form_vs_tomtoolkit_3_1_0.diff:199, still exactly 10 regions); `test_body_diff_matches_pinned_snapshot` (test_calendar_template.py:2162) and the Task 1 structural check enforce it | closed |
| T-39-30 | Information disclosure | the "Possible campaign run match" hint in event_form.html (candidate run name, score, attribution-queue band=high link): a later edit weakening its `request.user.is_staff` conjunct (for example to is_authenticated) would show a not-yet-public candidate run to every signed-in account, and until now only the regenerable byte snapshot would notice (39-REVIEW WR-04 / WR-02; T-27-21; UAT G-39-7) | medium | mitigate | 39-06 Task 1 (commit cadb56c, test module only): `test_signed_in_non_staff_does_not_see_hint` (test_calendar_template.py:1010, view level: plain_user GET of the edit pop-up is 200 with the update `hx-post` and neither the hint text nor the `?band=high` link) and the `(self.plain_user, 'update', False)` row of `test_hint_is_gated_on_the_edit_form` (:984, row at :993, template level); the scratch-template mutation run (gate weakened to `is_authenticated` under $HOME/tmp, tracked template untouched) ends `FAILED (failures=2)` with exactly those two checks while the real tree runs 13 OK; the gate itself (event_form.html:284) is unchanged | closed |
| T-39-31 | Repudiation | 39-REVIEW-DISPOSITION.md marks WR-04 fixed without the test having landed, or citing the wrong commit, misstating the phase's open findings to Phase 41 triage and the ship decision | low | mitigate | 39-06 Task 2 (commit 8d0f26b, +3/-3, ledger only) ran after Task 1's commit; its check requires the WR-04 Source cell to cite the hash of the post-6ea4928 commit touching the test module (cadb56c), both test names, the is_authenticated mutation and G-39-7, `open:` 10 -> 9, and exactly three changed ledger lines (`OK: WR-04 recorded fixed`); no other row changed | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-39-01 | T-39-10, T-39-22 | D-01 / D-03 (39-CONTEXT.md): any logged-in user may create, edit or delete any calendar event, including other users' and pipeline-created ones; the boundary this phase sets is anonymous vs. logged-in, and tightening to staff/ownership is a later, separate decision at the same wrapping point (39-01 prohibition). Corrected 2026-10-08 (39-REVIEW CR-01): this risk was first accepted on the premise that accounts are issued by the operator (39-01-PLAN.md T-39-10); that premise is false while TOM_REGISTRATION_STRATEGY = 'open' -- any member of the public can obtain a login -- so the accepted risk covers self-registered accounts too. Re-accepted with the premise corrected: "Self-signup is wanted; collaborators should be able to join without an operator; calendar edits are visible, attributable and easily reverted." The registration setting, D-01 and the guard are unchanged. | Tim Lister (D-01 at discuss-phase 39; re-accepted after 39-REVIEW CR-01) | 2026-10-08 |
| AR-39-02 | T-39-27 | After a CSRF-failure login the create and edit addresses render event_form.html as a bare fragment (no page shell, no htmx); its Save is a native GET that saves nothing, discards the typed input and puts the CSRF token in the query string. Phase 39-05 closes this by documentation (runbook: do not use the copy, go back to the calendar page); the `method="post"` / `action` hardening of the FOMO-owned form (39-REVIEW WR-01 item 2) was offered and not requested at UAT on 2026-10-08. The path is not reachable by an anonymous visitor and is not a write path. | Tim Lister (UAT decision 2026-10-08, G-39-3 scope) | 2026-10-08 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-08 | 19 | 19 | 0 | execute-phase verify:post (L1 grep-depth short-circuit: plan-time register, ASVS 1) |
| 2026-10-08 | 24 | 24 | 0 | execute-phase verify:post after gap-closure plan 39-04 (L1 grep-depth short-circuit: plan-time register, ASVS 1; T-39-20, T-39-21, T-39-23, T-39-24 added from 39-04's threat model, T-39-22 recorded by the plan itself) |
| 2026-10-08 | 29 | 29 | 0 | execute-phase verify:post after gap-closure plan 39-05 (L1 grep-depth short-circuit: plan-time register, ASVS 1; T-39-25, T-39-26, T-39-28, T-39-29 mitigated and T-39-27 accepted as AR-39-02 from 39-05's threat model; the shared T-39-SC supply-chain row stays accepted/no installs as in every plan of this phase and is not tabulated) |
| 2026-10-09 | 31 | 31 | 0 | execute-phase verify:post after gap-closure plan 39-06 (L1 grep-depth short-circuit: plan-time register, ASVS 1; T-39-30 and T-39-31 mitigated from 39-06's threat model, test-only plan; the shared T-39-SC supply-chain row stays accepted/no installs and is not tabulated) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-08; re-verified 2026-10-08 after gap-closure plan 39-04 (24 threats, 24 closed, threats_open 0); re-verified 2026-10-08 after gap-closure plan 39-05 (29 threats, 29 closed, threats_open 0); re-verified 2026-10-09 after gap-closure plan 39-06 (31 threats, 31 closed, threats_open 0)

## Security Audit 2026-10-08

| Metric | Count |
|---|---|
| Threats found | 24 |
| Closed | 24 |
| Open | 0 |

## Security Audit 2026-10-08

| Metric | Count |
|---|---|
| Threats found | 29 |
| Closed | 29 |
| Open | 0 |

## Security Audit 2026-10-09

| Metric | Count |
|---|---|
| Threats found | 31 |
| Closed | 31 |
| Open | 0 |
