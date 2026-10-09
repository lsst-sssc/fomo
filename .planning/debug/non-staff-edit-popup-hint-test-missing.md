---
status: diagnosed
trigger: "add the test"
created: 2026-10-09T02:33:33Z
updated: 2026-10-09T02:46:00Z
goal: find_root_cause_only
gap_id: G-39-7
---

## Current Focus
<!-- OVERWRITE on each update - always reflects NOW -->

hypothesis: CONFIRMED - the `request.user.is_staff` conjunct of event_form.html:284 has no behavioural test for a signed-in non-staff user on the edit pop-up; weakening it to is_authenticated is caught only by the regenerable snapshot.
test: done (differential mutation experiment, see Evidence 2026-10-09T02:44:00Z)
expecting: n/a
next_action: Hand root cause to the gap-closure planner (/gsd-plan-phase 39 --gaps); goal is find_root_cause_only, no fix applied.
bug_class: bohrbug (deterministic coverage gap; not a runtime failure)
reasoning_checkpoint:
  hypothesis: "The hint's staff gate is unpinned because every edit-form test uses a viewer whose is_staff equals is_authenticated (staff, superuser: both True; anonymous: both False); the only signed-in non-staff fixture (plain_user) is used solely on the create form, where `action == \"update\"` already hides the hint."
  confirming_evidence:
    - "User/action matrix of all 12 EventModalAttributionHintTest methods: no (signed-in, non-staff, update) cell."
    - "Mutation is_staff -> is_authenticated at line 284: all 12 existing tests pass (24 executions); the proposed test fails with the candidate name, score and band=high link in the plain user's body."
    - "Proposed test passes on the real tree."
  falsification_test: "Any existing test failing under the is_authenticated mutation (none did), or the proposed test passing under it (it failed)."
  fix_rationale: "Adding the missing matrix cell pins the conjunct behaviourally; no production change is needed because production behaviour is correct."
  blind_spots: "Mutation run used a scratch TEMPLATES DIR, so EventFormHeaderMatchesUpstreamTest (reads the file path directly) did not see it; the reviewer's in-place run showed those two snapshot tests fail and are regenerable. Other weakenings (e.g. dropping the is_staff conjunct entirely) behave the same for a signed-in user and are caught by the same test."
  candidate_causes:
    - "code (tests): missing non-staff edit-form test - CONFIRMED"
    - "config/process: the snapshot (data) being treated as the guard - contributing, the regenerable byte pin hid the gap (WR-02)"
  and_gate: "no - one missing test is sufficient to explain the gap; the regenerable snapshot only explains why the gap was not noticed."
tdd_checkpoint: null

## Symptoms
<!-- Written during gathering, then immutable -->

expected: A repository test GETs the edit pop-up (`calendar:update-event`) as a signed-in NON-staff user and asserts the "Possible campaign run match" hint is absent (39-REVIEW.md WR-04; UAT Test 7).
actual: No repository test does so. User response at UAT Test 7, verbatim: "add the test". Verifier's scratch test of exactly that case passes on the real tree and fails when the template's gate is mutated to request.user.is_authenticated, while all 17 repository tests still pass under that mutation.
errors: None - test-coverage gap, not a runtime failure. Production behaviour believed correct.
reproduction: Test 7 in .planning/phases/39-calendar-write-access/39-UAT.md
started: Discovered during UAT (raised earlier as 39-REVIEW.md WR-04)

## Eliminated
<!-- APPEND only - prevents re-investigating after /clear -->

- hypothesis: test_anonymous_does_not_see_hint already protects the staff gate
  evidence: AnonymousUser has is_authenticated False and is_staff False, so the test passes with either gate; confirmed - it passed under the is_authenticated mutation.
  timestamp: 2026-10-09T02:44:00Z

- hypothesis: another test module (test_calendar_write_access, test_bootstrap5_rendering) covers a signed-in non-staff edit pop-up with a candidate
  evidence: Their non-staff editors only open events with no target_list / no CampaignRun, so no candidate exists, and neither asserts on the hint text; repo-wide grep finds the hint text only in test_calendar_template.py.
  timestamp: 2026-10-09T02:39:00Z

## Evidence
<!-- APPEND only - facts discovered during investigation -->

- timestamp: 2026-10-09T02:33:33Z
  checked: .planning/debug/knowledge-base.md (Phase 0)
  found: No entry matches this coverage-gap pattern (staff-gate template branch untested for non-staff). Closest family is blank-new-event-popup (G-39-4, same template branch, different defect) - not yet in the KB as a resolved entry.
  implication: No known-pattern candidate; proceed with direct investigation.

- timestamp: 2026-10-09T02:36:00Z
  checked: 39-REVIEW.md WR-04 and 39-REVIEW-DISPOSITION.md
  found: WR-04 (warning, disposition open) - "No behavioural test keeps the edit-form hint from a signed-in non-staff user; 39-05 rewrote that gate, and weakening it is caught only by the regenerable snapshot". Reviewer's mutation (is_staff -> is_authenticated at event_form.html:284) across test_calendar_template + test_calendar_write_access (133 tests) failed only test_body_diff_matches_pinned_snapshot and test_snapshot_detects_an_unlisted_line_inside_an_anchored_region; regenerating the snapshot (WR-02 one-liner) turns the suite green. Reviewer proposed test_signed_in_non_staff_does_not_see_hint + a (plain, 'update', False) row in test_hint_is_gated_on_the_edit_form.
  implication: The finding is a coverage gap on one conjunct; the only current tripwire is a byte-level snapshot that is regenerable by design.

- timestamp: 2026-10-09T02:37:00Z
  checked: src/templates/tom_calendar/partials/event_form.html:228-315 and header item 4 (lines 23-25)
  found: Line 284 `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}` is the else-arm of `{% if deco %}` (line 229) and sits OUTSIDE the `{% if request.user.is_authenticated %}` form branch (line 39). Its comment (lines 290-293) states the security intent: "this hint must never reach an anonymous or non-staff visitor". The body renders the "Possible campaign run match" label (line 302) and the `{% url 'campaigns:attribution' %}?band=high` link. Header item 4 already says "staff-only ... hint (edit form only)".
  implication: `request.user.is_staff` is the single conjunct that separates signed-in non-staff from staff on the edit form; it is the guard under test.

- timestamp: 2026-10-09T02:38:00Z
  checked: solsys_code/tests/test_calendar_template.py EventModalAttributionHintTest (lines 785-998), user/action matrix of every test
  found: 12 test methods. Edit-form (update) GETs: staff_user (test_staff_sees_high_band_hint_for_unlinked_event, test_record_backed_event_shows_no_hint, test_no_candidate_event_shows_no_hint, test_linked_event_shows_run_block_not_hint), superuser (test_superuser_sees_high_band_hint_for_unlinked_event), anonymous (test_anonymous_does_not_see_hint). render_to_string as staff_user only (test_hint_is_gated_on_the_edit_form, actions update/create). plain_user (line 842) is used ONLY in test_staff_create_form_matches_the_plain_users_apart_from_the_csrf_token, on calendar:create-event, where the `action == "update"` conjunct already hides the hint. Helpers _modal_url (844), _signed_in_client (847, raise_request_exception=False + force_login) and fixture unlinked_event_with_candidate (812, high-band candidate for matched_run) are reusable as-is.
  implication: The (signed-in, non-staff, update) cell of the matrix is empty. Anonymous has is_authenticated False AND is_staff False, staff/superuser have both True, so no existing test distinguishes is_staff from is_authenticated.

- timestamp: 2026-10-09T02:39:00Z
  checked: repo-wide grep for 'Possible campaign run match', 'band=high', 'partials/event_form.html' render, and calendar:update-event GETs in other test modules
  found: The hint text is asserted only in test_calendar_template.py (lines 866, 889, 895, 903, 910, 938, 992, 998), none for a signed-in non-staff user on update. test_calendar_write_access.py GETs update-event as non-staff editors (calendar-editor, csrf-replay-editor) but its events have no target_list/CampaignRun, so no candidate exists and it never asserts on the hint. test_bootstrap5_rendering.py's non-staff editor (bs5-calendar-editor) opens only events without a target_list (no candidate) and never asserts on the hint.
  implication: Confirms no test anywhere in the repo pins the non-staff edit-form case.

- timestamp: 2026-10-09T02:41:00Z
  checked: python manage.py test solsys_code.tests.test_calendar_template.EventModalAttributionHintTest (baseline, real tree, venv devel_fomo311_venv, Django 5.2.17)
  found: Ran 12 tests, OK.
  implication: Baseline green; the class has 12 test methods (the "17" in 39-REVIEW/UAT counts the tests 39-05 touched across files, not this class).

- timestamp: 2026-10-09T02:44:00Z
  checked: Differential mutation experiment with no tracked file touched - scratch settings prepend a TEMPLATES DIR holding a copy of event_form.html with line 284 `request.user.is_staff` -> `request.user.is_authenticated`; scratch module subclasses EventModalAttributionHintTest and adds test_signed_in_non_staff_does_not_see_hint (plain_user via _signed_in_client, GET _modal_url(unlinked_event_with_candidate), assert 200, '<form' in body, 'Possible campaign run match' and the attribution ?band=high URL not in body). The loader collected the base class too (25 tests = 12 base + 12 inherited + 1 new).
  found: Real tree - Ran 25, OK. Mutated - Ran 25, FAILED (failures=1): only test_signed_in_non_staff_does_not_see_hint failed; all 24 existing-test executions passed. Failure body shows the plain user's edit form (hx-post="/calendar/update/1/") followed by "Possible campaign run match", candidate "#1 Didymos 2026 | FTS/MuSCAT4 | 2026-07-07..2026-07-21 | no site", score 0.82 and /campaigns/attribution/?band=high.
  implication: CONFIRMED. No test in the class detects the weakened gate; the proposed test does, and it passes on the real tree (not tautological). Under the mutation a not-yet-public candidate run's name and score leak to every signed-in (self-registered) account.

- timestamp: 2026-10-09T02:45:00Z
  checked: EventFormHeaderMatchesUpstreamTest (test_calendar_template.py:2050-2185), the pinned snapshot solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff (line 199 holds the line-284 text), header item 4, runbook lines 915-925
  found: The snapshot tests read event_form.html from disk and diff it against installed upstream - a source-level byte check, regenerable via the one-liner its failure message prints (WR-02), so it is the only current tripwire for the mutation and not a behavioural one. A test-only fix changes neither the template body (snapshot unaffected) nor the header (item 4 already says "staff-only ... hint (edit form only)"). The runbook mentions "the calendar modal's staff hint" - behaviour unchanged, so the paired-doc rule is not triggered.
  implication: The fix is test-only: no template, snapshot, header or runbook change.

- timestamp: 2026-10-09T02:45:30Z
  checked: git status --short src/ solsys_code/ docs/ and sha256sum -c of event_form.html against the pre-experiment hash
  found: Clean; event_form.html: OK (9ad16ad6...c1 unchanged).
  implication: Investigation left the tracked tree untouched.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: Test-coverage gap (no production defect). The `request.user.is_staff` conjunct in `{% elif action == "update" and request.user.is_staff and not event.telescope_label_meta.run %}` (src/templates/tom_calendar/partials/event_form.html:284) - the only thing keeping the "Possible campaign run match" hint (candidate run name, score, attribution-queue band=high link) from signed-in non-staff accounts on the edit pop-up - is not exercised by any behavioural test. EventModalAttributionHintTest GETs calendar:update-event only as staff_user, superuser (is_staff and is_authenticated both True) and anonymous (both False), and its render_to_string gate test renders only as staff_user; plain_user is used only on calendar:create-event, where the action conjunct already hides the hint. So no test distinguishes is_staff from is_authenticated: with the gate mutated to is_authenticated all 12 class tests pass, and in the reviewer's in-place run across test_calendar_template + test_calendar_write_access (133 tests) the only failures were the two regenerable byte-snapshot tests in EventFormHeaderMatchesUpstreamTest (39-REVIEW WR-04/WR-02).
fix: (not applied - goal find_root_cause_only) Suggested: add test_signed_in_non_staff_does_not_see_hint to EventModalAttributionHintTest reusing plain_user, _signed_in_client, _modal_url and unlinked_event_with_candidate; optionally a (plain_user, 'update', False) row in test_hint_is_gated_on_the_edit_form. Test-only; no template, snapshot, header or runbook change.
verification: Diagnosis verified by differential mutation (scratch settings, tracked tree untouched): proposed test passes on the real tree and is the only failure under is_staff -> is_authenticated.
oracle_type: specified (template comment lines 290-293 and header item 4: "staff-only", "must never reach an anonymous or non-staff visitor"; T-27-21)
files_changed: []
