---
phase: 261006-lsf
plan: 01
subsystem: calendar-reconciler
tags: [F12, calendar, proposal, legend, reconciler, allocation-projector]
requires: [ALLOC-01, PROJ-06, UNUSED-01]
provides:
  - 'RUN: containers and ALLOC: nights carry CampaignRun.proposal_code'
  - target-first container titles
  - '"No proposal recorded" legend entry'
affects: [solsys_code/campaign_reconciler.py, solsys_code/allocation_projector.py, solsys_code/templatetags/calendar_display_extras.py]
key-files:
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/allocation_projector.py
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_calendar_template.py
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
decisions:
  - "Container title rule: {target}{' — '}{rest}, with exactly one trailing ' — {target}' stripped from telescope_instrument (exact, case-sensitive); no-target runs keep the old title"
  - "proposal rides on the existing _mint_fields/_label_fields builders (legacy re-key now uses _label_fields); no change to calendar_utils.py, observation_projector.py or src/templates/"
status: complete
commits: 3
plan_head_before: 976f5d44c2fa312b905c0bd886a8de30daf81cd2
plan_head_after: 876a36f
actuals:
  tasks: 3
  commits: 3
completed: 2026-10-06
---

# Quick task 261006-lsf: Fix F12 (write run proposal code onto RUN:/ALLOC: events)

One-liner: Reconciler containers and allocation nights now carry their run's proposal code (so they take that proposal's colour and legend entry), container titles lead with the target without repeating it, and the grey legend entry reads "No proposal recorded".

## Commits

| Task | Hash | Message |
| ---- | ---- | ------- |
| 1 (tracer) | 3b2924c | fix(261006-lsf): write run.proposal_code onto RUN:/ALLOC: events and lead the container title with the target |
| 2 | 0f3fca3 | fix(261006-lsf): relabel the empty-proposal calendar legend entry "No proposal recorded" |
| 3 | 876a36f | docs(261006-lsf): demo proposal-coloured, target-first reconciler entries; runbook legend and title wording |

## What changed

- `campaign_reconciler.py`: `TARGET_TITLE_SEPARATOR`, `_container_label(run)`, `event_title()` builds from it, `_write_container_event()` writes `proposal`. Docstrings updated.
- `allocation_projector.py`: `_mint_fields()` and `_label_fields()` write `proposal`; legacy re-key uses `_label_fields(run, dark_line)`; label-field docstrings and comments list `proposal`. `allocation_night_title()` unchanged.
- `calendar_display_extras.py`: `NO_PROPOSAL_LABEL = 'No proposal recorded'` replaces `CLASSICAL_SCHEDULE_LABEL`; wording only elsewhere. Tag and CSS names, and `src/templates/`, untouched.
- Notebook `reconcile_campaign_runs_demo.ipynb`: new markdown+code F12 pair (ids `e5a1c3d7`, `7b2f9e04`) between `db8b9701` and `a5619fed`, `proposal=` print in `31a60753`, prose in `8b703ea8`, `dbe67c97`, `d2adacf8`; re-executed top to bottom on a scratch copy (17 s), no error output, `PASS: F12`. Real-data part: 9 coded runs (pks 1, 69-76), every event carries its run's code; example title `10P — LCO 1m0 / Sinistro (window 2026-08-01..2027-01-31)`.
- Runbook: three edits (bracketed-token paragraph, colour legend sentence, entry-title paragraph plus new "Every reconciler entry carries its run's proposal code." paragraph). Sphinx builds.
- `campaign_lifecycle_demo.ipynb` deliberately not re-executed; the Task 3 gate confirmed it sets no run code or run target.

## Test counts

| Scope | Before | After |
| ----- | ------ | ----- |
| Task 1 six-module set (reconciler, allocation projector, write_and_reconcile, null_campaign_guards, campaign_approval, reconcile_campaign_runs) | 361 | 378 (OK) |
| Task 2 (`test_calendar_display_extras` + `test_calendar_template`) | 173 | 174 (OK) |
| Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault --noinput --parallel 4` | 2095 | `Ran 2113 tests ... OK` |

17 new Task 1 tests (R1-R10, A1-A7) plus 1 new Task 2 test (`test_no_proposal_label_text`). Task 1 RED run matched the plan's prediction exactly (11 failing: R1, R3, R4, R6, R7, R9, R10, A1, A4, A5, A7; the other 6 passed as guards).

## Deviations from Plan

**1. [Rule 1 - lint] SIM108 in `_container_label`.** ruff flagged the if/else block; rewritten with a `strip_suffix` flag and a ternary. Behaviour identical (covered by R3, R4).

**2. [Minor] Runbook cross-reference wrap.** The bold lead-in in the legend paragraph is line-broken after "its" so the Task 3 gate's `t.index('Every reconciler entry carries its run')` finds the real paragraph and not the cross-reference. The bullet containing "**No proposal recorded**" was likewise wrapped so the phrase stays on one line (gate counts three occurrences).

**3. [Minor] Notebook demo cell** imports `date` explicitly rather than relying on an earlier cell's global.

No pre-existing test pinned the F12 defect, so none was updated. The only renamed tests are the two legend-label ones the plan named.

Note: under `--parallel`, a failing test run crashes with a "cannot pickle 'traceback'" error rather than printing the failure; the RED run was done serially.

## Operator follow-up (plan's human-check, not done by the executor)

After the first unattended tick that starts after commit 3b2924c, expect about 21 one-time `updated` in the reconcile line (7 containers + 10 + 4 nights), then 0 on the next tick. Read-only cross-check and ticking F12 in `.planning/v2.4-INTENT-REVIEW.md` are the operator's. The live database was never opened by the executor; the only access was the notebook's scratch copy (cron END banner confirmed before starting).

## Known Stubs

None.

## Threat Flags

None. No new endpoints, auth paths or schema changes.

## Self-Check: PASSED

- 3b2924c, 0f3fca3, 876a36f all in `git log 976f5d4..HEAD`.
- `git diff --quiet 976f5d4 HEAD -- solsys_code/observation_projector.py solsys_code/calendar_utils.py src/templates/` clean.
- The four operator-owned `.planning/` files are still modified and unstaged; the untracked files are still untracked.
