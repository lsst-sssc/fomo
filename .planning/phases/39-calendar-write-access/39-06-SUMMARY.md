---
phase: 39-calendar-write-access
plan: 06
subsystem: calendar
tags: [django, tom_calendar, tests, gap-closure, attribution-hint]
gap_closure: true
gap_ids: [G-39-7]

requires:
  - phase: 39-calendar-write-access
    provides: the staff-only attribution hint gate in event_form.html (39-05) and the plain_user, _signed_in_client, _modal_url and unlinked_event_with_candidate fixtures
provides:
  - test_signed_in_non_staff_does_not_see_hint (view level pin of the hint's staff conjunct)
  - a (plain_user, update, False) row in test_hint_is_gated_on_the_edit_form (template level pin)
  - a repeatable scratch-template mutation run proving both checks are not vacuous
  - WR-04 recorded as fixed in 39-REVIEW-DISPOSITION.md
affects: [41-todo-triage, verify-work UAT Test 7]

actuals:
  tokens: 614
  tasks: 2
  commits: 2
plan_head_before: cdaf3c8af58a7e707c6955275b5f07856bed99e1
plan_head_after: 8d0f26b61c30b5f8b1c5c2885cb55fa874ece048
commits: 2

tech-stack:
  added: []
  patterns:
    - "Mutation check against a scratch template copy and scratch settings module under $HOME/tmp (tracked template never edited) to prove a regression test is not vacuous"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_calendar_template.py
    - .planning/phases/39-calendar-write-access/39-REVIEW-DISPOSITION.md

key-decisions:
  - "Test-only fix: the hint gate in event_form.html was correct as written (G-39-7 root cause was a coverage gap), so the template, pinned snapshot, tag module, runbook and notebooks are untouched"
  - "The optional plain-user gate row was included because it is three loop rows and also catches the weakened gate on its own"

requirements-completed: [ACCESS-01, WARN-01]

coverage:
  - id: D1
    description: "A signed-in non-staff user opening the edit pop-up of an unlinked event with a High-band candidate gets the edit form and sees neither the hint nor its band=high link"
    requirement: ACCESS-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventModalAttributionHintTest.test_signed_in_non_staff_does_not_see_hint"
        status: pass
    human_judgment: false
  - id: D2
    description: "The template-level gate test pins the staff conjunct as well as the action conjunct via a (plain_user, update, False) row"
    requirement: WARN-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_template.py#EventModalAttributionHintTest.test_hint_is_gated_on_the_edit_form"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both checks fail, and only they fail, when the hint gate is weakened to request.user.is_authenticated"
    requirement: WARN-01
    verification:
      - kind: other
        ref: "scratch-template mutation run, log $HOME/tmp/phase39-06-mutant/mutant.log (Ran 13 tests, FAILED (failures=2))"
        status: pass
    human_judgment: false
  - id: D4
    description: "WR-04 is recorded as fixed in the Phase 39 review disposition ledger, citing the test commit"
    verification:
      - kind: other
        ref: "ledger check script in 39-06-PLAN.md Task 2 verify (OK: WR-04 recorded fixed)"
        status: pass
    human_judgment: false

duration: 41min
completed: 2026-10-09
status: complete
---

# Phase 39 Plan 06: Non-staff edit pop-up hint test Summary

**Test-only closure of G-39-7 / 39-REVIEW WR-04: a signed-in non-staff user opening the edit pop-up is now tested for the absence of the staff-only attribution hint at view and template level, and a scratch-template mutation to `is_authenticated` makes exactly those two checks fail.**

## Performance

- **Duration:** 41 min
- **Started:** 2026-10-09T03:13:24Z
- **Completed:** 2026-10-09T03:54:05Z
- **Tasks:** 2
- **Files modified:** 2 (plus this SUMMARY)

## Accomplishments

- Added `test_signed_in_non_staff_does_not_see_hint` as the last method of `EventModalAttributionHintTest`: `plain_user` GETs `calendar:update-event` for `unlinked_event_with_candidate`, the test asserts 200, asserts `hx-post="<update url>"` (so the edit form, not the create form or the anonymous card, rendered), then asserts that neither `Possible campaign run match` nor the `campaigns:attribution` `?band=high` link is in the body.
- Reworked `test_hint_is_gated_on_the_edit_form` to loop over `(user, action, shown)` with rows `(staff_user, update, True)`, `(staff_user, create, False)`, `(plain_user, update, False)`, setting `request.user` per row inside `subTest(user=..., action=...)`.
- Appended a G-39-7 paragraph to the class docstring (existing text kept as its prefix).
- Marked WR-04 `fixed` in `39-REVIEW-DISPOSITION.md` (`open: 10` to `open: 9`), the same three-line edit 1691f6e made for WR-01.

## Task Commits

1. **Task 1 (tracer): add the non-staff edit pop-up test and the plain-user gate row** - `cadb56c` (test; full pre-commit hook including django-test passed)
2. **Task 2: record WR-04 as fixed in the review ledger** - `8d0f26b` (docs; `SKIP=django-test`, `.planning`-only)

**Plan metadata:** recorded in the final `docs(39-06)` commit that adds this SUMMARY with STATE.md, ROADMAP.md and REQUIREMENTS.md.

## Evidence

Real tree, `python manage.py test --noinput solsys_code.tests.test_calendar_template.EventModalAttributionHintTest`:

```
Ran 13 tests in 1.671s
OK
OK: EventModalAttributionHintTest runs 13 tests green on the real tree
```

Precondition baseline before any edit: `Ran 12 tests` / `OK`.

Mutation run (hint elif's `request.user.is_staff` changed to `request.user.is_authenticated` in a scratch copy under `$HOME/tmp/phase39-06-mutant`, loaded first through `--settings=phase39_06_mutant_settings`; the tracked template never touched), log kept at `$HOME/tmp/phase39-06-mutant/mutant.log`:

```
FAIL: test_hint_is_gated_on_the_edit_form (solsys_code.tests.test_calendar_template.EventModalAttributionHintTest.test_hint_is_gated_on_the_edit_form) (user='attrmodalplain', action='update')
FAIL: test_signed_in_non_staff_does_not_see_hint (solsys_code.tests.test_calendar_template.EventModalAttributionHintTest.test_signed_in_non_staff_does_not_see_hint)
Ran 13 tests in 1.574s
FAILED (failures=2)
OK: exactly the two new checks fail when the hint gate is weakened to is_authenticated; tracked template untouched
```

Structural check (AST comparison against 6ea4928):

```
OK: one new test last in the class, the gate test loops over the three rows, nothing else in the module or under src/, docs/, solsys_code/ changed
```

Ruff hooks (`pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`):

```
Lint code using ruff; sort and organize imports..........................Passed
Format code using ruff...................................................Passed
OK: ruff hooks clean, one commit after 6ea4928 holds exactly the test module, on the branch (cadb56c604ac742fefde7b0d900b9bb83545e02a)
```

Ledger check:

```
OK: WR-04 recorded fixed (open 10 -> 9), citing cadb56c; no other ledger line changed
OK: one ledger commit after 6ea4928 holding exactly the ledger, on the branch
```

## Files Created/Modified

- `solsys_code/tests/test_calendar_template.py` - one new test, one extended test loop, one class-docstring paragraph (26 insertions, 4 deletions).
- `.planning/phases/39-calendar-write-access/39-REVIEW-DISPOSITION.md` - WR-04 disposition, `open` count and table row (3 lines).

## Decisions Made

- Test-only change. No module behaviour changed, so the paired-docs rule in CLAUDE.md is not triggered: `event_form.html`, the pinned snapshot `solsys_code/tests/data/event_form_vs_tomtoolkit_3_1_0.diff`, `attribution_display_extras.py`, `docs/runbooks/telescope_runs_calendar.rst` and every notebook are byte-identical to 6ea4928.
- `39-UAT.md`, `39-VERIFICATION.md`, `39-REVIEW.md` and `39-SECURITY.md` were not touched, and no earlier plan file was edited.
- WR-02, WR-03 and the IN-01 to IN-07 findings remain `open` in the ledger for Phase 41 triage (CR-01 stays `skipped`, WR-01 `fixed`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Over-long explanatory comment in the new test (ruff E501)**
- **Found during:** Task 1 (ruff hook, before the commit)
- **Issue:** The one-line comment the plan asked for ran to 121 characters, over the 120-column limit.
- **Fix:** Reworded the comment to fit; same meaning. The first reword added a character and failed again; the second fits.
- **Files modified:** `solsys_code/tests/test_calendar_template.py`
- **Verification:** both ruff hooks pass; class re-run 13 tests OK; the full django-test hook passed in the commit.
- **Commit:** cadb56c

**Total deviations:** 1 auto-fixed (lint). **Impact:** none on behaviour or scope.

## Issues Encountered

None. `workflow.human_verify_mode` is end-of-phase and Task 1's verify carries only automated blocks, so the tracer feedback gate was the re-run of those blocks: `Tracer verified end-to-end - expanding`.

## Known Stubs

None.

## Threat Flags

None. The plan's T-39-30 (information disclosure of the hint to non-staff) is mitigated by the new tests; T-39-31 (ledger accuracy) is mitigated by the ledger check citing `cadb56c`.

## Next Phase Readiness

G-39-7 is closed; `/gsd-verify-work` can mark UAT Test 7 resolved. Nothing is pushed.

## Self-Check: PASSED

- `solsys_code/tests/test_calendar_template.py` and `39-REVIEW-DISPOSITION.md` exist and are committed.
- Commits `cadb56c` and `8d0f26b` are ancestors of HEAD on `issue37-telescope-runs-calendar`.
