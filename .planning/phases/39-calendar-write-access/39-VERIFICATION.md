---
phase: 39-calendar-write-access
verified: 2026-10-09T04:07:17Z
status: passed
score: 64/64 must-haves verified
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
  - .planning/phases/39-calendar-write-access/39-06-PLAN.md
  - .planning/phases/39-calendar-write-access/39-06-SUMMARY.md
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
covered_digest: "v3:sha256:c53b45871371cef06220460b8b8f8010ff2364f890d5da0e51aa06e509e9b3e1"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "CSRF unchanged: a signed-in POST to calendar:create-event without a CSRF token through Client(enforce_csrf_checks=True) returns 403 and creates nothing (the guards do not exempt CSRF)."
    reason: "tom_common's Raise403Middleware rewrites every browser 403 into a 302 to login (next=/calendar/create/); the request is still refused and nothing is created, and the distinct next proves the CSRF layer refused it, not the guard"
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-08T17:04:52Z"
re_verification:
  previous_status: human_needed
  previous_score: 58/58
  gaps_closed:
    - "G-39-7 (39-UAT Test 7, 39-REVIEW WR-04): test_signed_in_non_staff_does_not_see_hint and a (plain_user, update, False) row in test_hint_is_gated_on_the_edit_form now pin the staff conjunct of the attribution-hint gate; an independent verifier mutation run (gate weakened to is_authenticated through a scratch template directory) fails exactly those two checks"
    - "Previous human item 1 (UAT Test 4 re-run as staff): 39-UAT Test 5 pass"
    - "Previous human item 2 (runbook bare-form sentence): 39-UAT Test 6 pass"
    - "Previous human item 3 (WR-04 decision): developer chose 'add the test' (39-UAT Test 7); added by 39-06 (cadb56c)"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "39-REVIEW IN-08 (new, round 4): test_signed_in_non_staff_does_not_see_hint has no positive control of its own; its hx-post anchor proves the signed-in non-create form rendered, not that the event still has a High-band candidate in that render, and the line-1017 comment ('so the absences below mean something') claims more than the anchor alone proves."
    category: other
    reason: "Not a must-have miss. The 39-06 truth claims only what the anchor does prove (edit form, not create form or anonymous card). Non-vacuity is established by the shared setUpTestData fixture plus two sibling positive controls on the same event (test_staff_sees_high_band_hint_for_unlinked_event and the staff/update row of test_hint_is_gated_on_the_edit_form), which would fail first if the candidate dropped out of the High band, and by the verifier's own mutation run. Resolve by folding a staff row into the test or rewording the comment; Phase 41 triage candidate with the other open findings."
    evidence_status: "statically evident; verifier mutation run shows the test catches the WR-04 regression as written"
  - finding: "39-REVIEW WR-02 (carried): event_form.html's header says the pinned snapshot makes any new difference fail until the list and the file are updated together; regenerating the snapshot alone turns the test green for a line inside an already-anchored region."
    category: other
    reason: "The six-item list is exact today; only the sentence about what the test enforces overclaims. Open in the ledger for Phase 41 triage."
    evidence_status: "statically evident (header unchanged since the previous report)"
  - finding: "39-REVIEW WR-03 (carried): the snapshot is named 'vs tomtoolkit 3.1.0' but diffed against the installed tomtoolkit; pyproject.toml has tomtoolkit>=3.1.0 unpinned."
    category: other
    reason: "Installed version is 3.1.0, so the comparison is true today. Open in the ledger for Phase 41 triage."
    evidence_status: "statically evident"
  - finding: "39-REVIEW IN-06 (carried): the staff 'Save and Edit' create path has no repository test."
    category: other
    reason: "Works today (previous verifier scratch probe passed). Coverage, not a defect."
    evidence_status: "scratch probe passed in the previous round"
  - finding: "39-REVIEW IN-07 (carried): runbook troubleshooting does not cover a pop-up that opens empty."
    category: other
    reason: "Nothing in the runbook is false; G-39-4's cause is fixed and tested. Phase 41 triage candidate."
    evidence_status: "statically evident"
---

# Phase 39: Calendar Write Access Verification Report

**Phase Goal:** The public calendar is read-only to anyone not logged in — an anonymous visitor can neither create, change nor delete an event, and is not offered a control that would try — and FOMO's event pop-up override says truthfully how it differs from tomtoolkit 3.1.0's upstream template.
**Verified:** 2026-10-09T04:07:17Z
**Status:** passed
**Re-verification:** Yes — after gap-closure plan 39-06 (commits cadb56c test, 8d0f26b ledger, 19cb95c summary)

The phase goal is achieved, and the last open item is closed.

- **G-39-7 / WR-04 is closed in the code.** The repository now has a test in which a signed-in non-staff user opens the edit pop-up and does not see the staff-only hint. A second, template-level test row pins the same gate. I ran my own mutation check: I copied the template to a scratch directory outside the repo and weakened the gate to `is_authenticated`. Exactly those two checks fail (`Ran 13 tests ... FAILED (failures=2)`). On the real tree the class passes (13 tests, OK).
- **The previous report's three human items are resolved at UAT.** Test 5 (staff browser re-run) passed. Test 6 (the runbook sentence) passed. Test 7 led to "add the test", which 39-06 delivered.
- **Nothing outside the test module changed.** Since the previous verification (69b2df1), the only change under `src/`, `docs/`, `solsys_code/` and `pyproject.toml` is `solsys_code/tests/test_calendar_template.py`, +26/-4. The template, tag module, runbook, snapshot and notebooks are byte-identical, so every earlier must-have still holds.

No human items remain, so the status is `passed`. One bookkeeping note for the orchestrator: `39-UAT.md` still records G-39-7 as `status: failed`, because 39-06 was forbidden from editing that file. `/gsd-verify-work` (or the orchestrator) should mark it resolved by 39-06-PLAN.md.

## Goal Achievement

### Observable Truths — 39-06 (new, full verification)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| F1 | A signed-in non-staff user (plain_user) who GETs calendar:update-event for unlinked_event_with_candidate gets 200 and the edit form (`hx-post="<update url>"`). The body has neither `Possible campaign run match` nor the `campaigns:attribution?band=high` link (`test_signed_in_non_staff_does_not_see_hint`) | ✓ VERIFIED | test_calendar_template.py lines 1010-1020 assert exactly that. The test passes in my targeted run. The anchor is a sound discriminator: `hx-post="{% url 'calendar:update-event' event.id %}"` appears only at event_form.html line 43, the not-create arm of the signed-in `<form>` branch. `plain_user` is `create_user` with defaults, so it is not staff |
| F2 | `test_hint_is_gated_on_the_edit_form` loops over exactly three (user, action, shown) rows: (staff, update, True), (staff, create, False), (plain, update, False). It sets `request.user` per row inside `subTest(user=..., action=...)` | ✓ VERIFIED | Lines 984-1002 match. The old pre-loop `request.user = self.staff_user` is removed, and `request.user = user` is the first statement in the subTest. The render call and the assertEqual line are unchanged |
| F3 | Non-vacuity: with only the hint elif changed to `is_authenticated` in a scratch copy searched first, the class runs 13 tests and ends `FAILED (failures=2)`, failing only the new test and the plain-user subTest. The real tree runs 13 OK | ✓ VERIFIED (independently reproduced) | My own run used a scratch dir in the session scratchpad, not the executor's: `verifier_mut_settings` prepends a copy whose only diff is line 284. Output: `FAIL: test_hint_is_gated_on_the_edit_form ... (user='attrmodalplain', action='update')`, `FAIL: test_signed_in_non_staff_does_not_see_hint`, `Ran 13 tests`, `FAILED (failures=2)`. Real tree: `Ran 18 tests ... OK` for EventModalAttributionHintTest (13) plus EventFormHeaderMatchesUpstreamTest (5). The tracked tree stayed clean |
| F4 | Test-only: nothing under src/, docs/ or solsys_code/ changed since 6ea4928 except the test module. Every other class and member is AST-identical, and the imports are unchanged | ✓ VERIFIED | `git diff --stat 6ea4928 HEAD -- src docs solsys_code pyproject.toml` lists only test_calendar_template.py. My AST check: no top-level class was added or removed; only EventModalAttributionHintTest changed. Within it, one method was added (the last one) and only test_hint_is_gated_on_the_edit_form changed. The class docstring keeps its old text as a prefix plus a G-39-7 paragraph, which plan line 193 permits. Imports are identical |
| F5 | 39-REVIEW-DISPOSITION.md records WR-04 as `fixed` in its front matter and its table row. `open:` goes from 10 to 9, the Source cell cites 39-06 Task 1, cadb56c, both test names, the mutation and G-39-7, and nothing else changes | ✓ VERIFIED | `git show 8d0f26b` is exactly a three-line edit. The ledger was later re-run by the round-4 review gate (65ff40e), which added IN-08 and set `open: 10` / `total: 13`. WR-04 is still `fixed` with the same Source cell |
| F6 | Gate: ruff and ruff-format pass and change no tracked file; the test commit went through the full hook; nothing is pushed | ✓ VERIFIED | `pre-commit run ruff` and `ruff-format` on the module both Passed, and the tree is unchanged. No line is longer than 120 characters. cadb56c is not on any remote branch: the branch is 86 commits ahead of `origin/issue37-telescope-runs-calendar`. The full django-test hook run is the executor's and orchestrator's claim (the commit message has no SKIP). The change touches only one test class, and I re-ran that class |

### Observable Truths — roadmap and earlier plans (regression check)

All of these were fully verified in the previous report (58/58). Since then (69b2df1 to HEAD), the only non-planning change is the test module. I spot-checked the following:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| R1 | SC1: an anonymous POST to each of the five write routes is refused, and a test per route proves nothing changed | ✓ VERIFIED | calendar_urls.py, calendar_access.py, settings.py, urls.py and test_calendar_write_access.py are byte-identical since the previous report. The suite passed in the 39-06 commit hook |
| R2 | SC2: a signed-in user can still create, update and delete from the month view (staff included since 39-05) | ✓ VERIFIED | The files are unchanged. EventModalAttributionHintTest's staff create-form tests pass in my run |
| R3 | SC3: an anonymous visitor gets no create target, and the pop-up is readable | ✓ VERIFIED | calendar.html is unchanged. UAT Test 1 passed earlier |
| R4 | SC4: the event_form.html header names every block that differs from 3.1.0 | ✓ VERIFIED (advisory WR-02) | event_form.html and the snapshot are unchanged. EventFormHeaderMatchesUpstreamTest (5 tests) passes in my run. The gate at line 284 still reads `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}` |
| A1-A12, B1-B14, C1-C5, D1, N1-N13, E1-E9 | 39-01 to 39-05 truths | ✓ VERIFIED | Carried. Every file they rest on is byte-identical since the previous verification. A7 still passes by override; A12 was accepted by a human at UAT |

**Score:** 64/64 truths verified: 58 carried plus 6 new (F1-F6). That includes 1 passed by override (A7) and 1 human-accepted (A12). None is present-but-behavior-unverified. No truth was flagged for coincidental reliance. IN-08's concern is a precondition (the event has a High-band candidate), but the shared `setUpTestData` fixture declares it and sibling tests assert it, so it is not an undeclared or fixture-only precondition in the Step 5c sense.

### Prohibitions (39-06)

| Prohibition | Tier | Disposition |
|-------------|------|-------------|
| No change to event_form.html, the snapshot, attribution_display_extras.py, the runbook or any notebook | test | ✓ Enforced by a deterministic check: `git diff --stat 6ea4928 HEAD -- src docs solsys_code` lists only the test module |
| No change to any test other than the two named; no new fixtures; no skip, tag-out or expected failure | test | ✓ AST check (F4). No decorator was added; setUpTestData and the helpers are unchanged |
| The mutant template and scratch settings never go inside the repo, and the tracked template is never edited | test | ✓ `git status --untracked-files=all` shows no mutant files. The executor's scratch files are under `$HOME/tmp/phase39-06-mutant`. event_form.html is unchanged in git |
| No edit to 39-UAT, 39-VERIFICATION, 39-REVIEW, 39-SECURITY or earlier plans; only the WR-04 row of the ledger changed | test | ✓ cadb56c touches only the test module. 8d0f26b touches only the ledger (3 lines). 19cb95c touches ROADMAP.md, STATE.md and 39-06-SUMMARY.md |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/tests/test_calendar_template.py` | `def test_signed_in_non_staff_does_not_see_hint` and the plain-user row | ✓ VERIFIED | Present at lines 1010-1020 and 993; it reuses the existing fixtures and passes |
| `.planning/phases/39-calendar-write-access/39-REVIEW-DISPOSITION.md` | `\| WR-04 \| warning \| fixed \|` | ✓ VERIFIED | Present, citing cadb56c |
| (earlier plans' artifacts) | — | ✓ VERIFIED | Byte-identical since the previous report |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_calendar_template.py | event_form.html | a signed-in plain_user GET of update-event reaches the hint elif, whose `request.user.is_staff` conjunct alone hides the hint | ✓ WIRED | The pattern `elif action == "update" and request.user.is_staff` is at event_form.html:284. The mutation run proves the test reaches that conjunct |
| 39-REVIEW-DISPOSITION.md | test_calendar_template.py | the WR-04 Source cell names the test | ✓ WIRED | The cell contains `test_signed_in_non_staff_does_not_see_hint` |
| (earlier key links) | — | — | ✓ WIRED | Their files are unchanged |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| event_form.html hint | `attribution_candidates` | `high_band_attribution_candidates(event)` → `campaign_attribution.candidates_for_event` | Yes for staff on the edit form (the staff positive controls pass); withheld for non-staff by the gate (F1, F3) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Targeted classes on the real tree | `python manage.py test --noinput solsys_code.tests.test_calendar_template.EventModalAttributionHintTest solsys_code.tests.test_calendar_template.EventFormHeaderMatchesUpstreamTest` | Ran 18 tests, OK | ✓ PASS |
| WR-04 mutant (`is_staff` → `is_authenticated`, scratch DIRS entry outside the repo) | `PYTHONPATH=<scratch> python manage.py test --noinput --settings=verifier_mut_settings ...EventModalAttributionHintTest` | Ran 13 tests, FAILED (failures=2): the plain-user subTest and the new test | ✓ PASS (the expected failure occurs) |
| Ruff gates | `pre-commit run ruff` / `ruff-format --files solsys_code/tests/test_calendar_template.py` | Passed, Passed; tree unchanged | ✓ PASS |
| Full suite | not re-run, per orchestrator instruction; it passed in the cadb56c commit hook and was reused as the regression gate | — | ? not re-run (test-only change; the affected class was re-run) |

### Probe Execution

Step 7c: SKIPPED. The phase declares no probe scripts.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ACCESS-01 | 39-01, 39-03, 39-04, 39-05, 39-06 | an anonymous POST to the five write routes changes nothing; a test per endpoint | ✓ SATISFIED | R1 and the carried A/N truths. 39-06 adds coverage of a signed-in non-staff viewer and changes no behaviour |
| ACCESS-02 | 39-02, 39-03, 39-05 | the month view's create and update targets are hidden from anonymous users | ✓ SATISFIED | R3 and the carried B truths; calendar.html is unchanged |
| WARN-01 | 39-02, 39-03, 39-04, 39-05, 39-06 | the event_form.html header states accurately which blocks differ | ✓ SATISFIED | R4. Header item 4 ("staff-only, edit form only") is now backed by a behavioural test of the staff conjunct (F2) |

There are no orphaned requirements. REQUIREMENTS.md lines 90-92 map exactly ACCESS-01, ACCESS-02 and WARN-01 to Phase 39, and the plans claim all three.

### Code Review Round 4 (39-REVIEW.md, b748445): Classification

| Finding | Classification | Reasoning |
|---------|----------------|-----------|
| **IN-08**: the new HTTP test has no positive control of its own | **Not a must-have miss; 📋 advisory. Neither a gap nor a human decision** | Confirmed as stated. The anchor shows a signed-in, non-create form, not `action == "update"` and not a High-band candidate in that render. Three reasons it is not a gap. (1) The 39-06 truth F1 claims only what the anchor proves. (2) Non-vacuity, which is truth F3, rests on the mutation run, not on the anchor, and I reproduced it independently. (3) The shared `setUpTestData` fixture is protected by two positive controls on the same event: `test_staff_sees_high_band_hint_for_unlinked_event` and the staff/update row of the gate test. Either would fail first if the candidate fell out of the High band. What remains is a test-reliability nit and a comment (line 1017) that overstates what the anchor proves. It is a fix-when-convenient item (fold in a staff row, or reword the comment), best handled in Phase 41 triage with the other open findings. It needs no ship decision |
| WR-04 | Resolved | F1-F3. Ledger `fixed` (F5) |
| WR-02, WR-03, IN-01..IN-07 | Carried; open in the ledger by prior decision (Phase 41 triage) | Their files are unchanged; not gaps of this phase |
| CR-01 | Accepted risk | AR-39-01, unchanged |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| test_calendar_template.py | 1017 | comment overstates what the anchor proves (IN-08) | ℹ️ Info | Wording only; non-vacuity is shown by the mutation run |

No line that 39-06 added contains TBD, FIXME or XXX.

### Human Verification Required

None. The previous report's three items were resolved at UAT (Tests 5, 6 and 7), and 39-06 declares no `<human-check>` block.

### Gaps Summary

No gaps.

- **G-39-7 is closed by a test-only change.** I confirmed both new checks exist and pass. They fail exactly when the gate is weakened to `is_authenticated`, which I reproduced independently. No production file was touched.
- **Every earlier must-have still holds.** The files they rest on are byte-identical since the previous verification.

The only new review finding, IN-08, is an Info-level test-reliability nit and not a must-have miss. It sits in the advisory list with the carried WR-02, WR-03, IN-06 and IN-07 for Phase 41 triage.

---

_Verified: 2026-10-09T04:07:17Z_
_Verifier: Claude (gsd-verifier)_
