---
phase: 261002-gev
plan: 01
subsystem: unattended-runner / proposal-allocation
tags: [F7, proposal_allocation, unattended, portal, not-fetchable]
requires: []
provides:
  - proposal_codes_to_fetch() narrowed to LCO/SOAR-fetchable codes
  - proposal_codes_not_fetchable()
  - refresh_all() 5-tuple with not_fetchable
  - "not fetchable: N" in the proposal_allocation step summary
affects: [solsys_code/proposal_allocation.py, solsys_code/unattended.py, runbook, CLAUDE.md map]
tech-stack:
  added: []
  patterns: [run provenance rule derived from campaign_attribution alias tables]
key-files:
  created: []
  modified:
    - solsys_code/proposal_allocation.py
    - solsys_code/unattended.py
    - solsys_code/tests/test_proposal_allocation.py
    - solsys_code/tests/test_unattended.py
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - CLAUDE.md
    - .planning/v2.4-INTENT-REVIEW.md (working tree only, NOT committed)
decisions:
  - "Rule is run provenance (lco_queue/soar_queue source, or site obscode from campaign_attribution's LCO_SITE_CODE_TO_OBSCODE / OBSERVED_TELESCOPE_OBSCODES), not code shape: nothing records the portal's naming, so a regex would be a guess."
  - "telescope_class is deliberately not a branch: operators set it by hand and derive_telescope_class() infers it from free text, so a false positive would re-create F7; a false negative is benign (counted, tally says not yet known)."
metrics:
  duration: ~12 min
  completed: 2026-10-02
status: complete
commits: 3
plan_head_before: b9e9276941e06ce4b3ed32e234102e92f8442c23
plan_head_after: cfab05870a75a9e8c2c45d63c7b9e5d45da1adc0
actuals:
  tokens: 6000
  tasks: 3
  commits: 3
---

# Phase 261002-gev Plan 01: F7 proposal allocation fetches only LCO/SOAR-fetchable codes Summary

The unattended `proposal_allocation` step now sends a run's proposal code to the LCO Observation Portal only when the run is LCO/SOAR queue-sourced or sits at an FTN/FTS/SOAR observatory; every other run code is counted as `not fetchable: N`, never requested, and never fails the step.

## Commits

| Task | Hash | Subject |
| ---- | ---- | ------- |
| 1 (RED) | 0f56776 | test(261002-gev): pin that only LCO/SOAR-fetchable proposal codes reach the portal (F7) |
| 1 (GREEN) | 433eb9f | fix(261002-gev): fetch only LCO/SOAR-fetchable proposal codes; count the rest as not fetchable, never a step failure (F7) |
| 2 | cfab058 | docs(261002-gev): runbook and CLAUDE.md map cover not-fetchable proposal codes; load_telescope_runs notebook prose narrowed (F7) |

Task 3 made no commit (lint was already clean repo-wide, so no `style` commit; the F7 note is working-tree only).

## Test counts

- `test_proposal_allocation` + `test_unattended`: **126 before, 135 after** (9 new).
- With `test_campaign_tally`: 219 tests OK after the fix.
- Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`: **`Ran 1842 tests in 344.793s`, `OK (skipped=1)`, exit=0** (was 1833 at 261001-smo).
- RED run (before the fix): 135 tests, 4 failures (P5, P6, U1, U2) and 6 errors (P1, P2, P4, P7 and the two edited RefreshAllTests), exactly as planned; P3 and the edited union test passed as guards.

## Deviations from Plan

### Deliberate pre-existing test edits (planning finding 4)

1. `ProposalCodesToFetchTests.test_union_of_watched_and_run_codes_sorted_and_deduped` -- its two code-carrying runs now carry `source=CampaignRun.Source.LCO_QUEUE` (they were `legacy`/no-site and would no longer be fetchable). Assertion unchanged.
2. `RefreshAllTests.test_isolates_one_failing_proposal_from_the_rest` -- unpacks 5 values, asserts `not_fetchable == 0`.
3. `RefreshAllTests.test_empty_code_list_is_a_no_op` -- unpacks 5 values, expects `(0, 0, 0, None, 0)`.

The pin gate confirmed the other 134 pre-existing test methods are AST-identical to before.

### Auto-fixed / environmental

**[Rule 3 - Blocking] Sphinx pre-commit hook failed on the first commit attempt (RED commit).**
- `FileNotFoundError: docs/autoapi/fomo/asgi/index.rst does not exist`, caused by a `sphinx-autobuild` process on port 8001 (the operator's) racing the hook's autoapi regeneration of the untracked `docs/autoapi/`. Same failure mode as 261002-dsa. No commit was created; the identical commit succeeded on retry (no `--no-verify`, no amend). I did not touch the operator's autobuild process. A transient untracked `docs/autoapi/` reappeared afterwards; it is generated output and was never staged.

**P1 test unpacks 5 values** (as the plan's behavior text specified) instead of comparing the raw return tuple, so RED matched the planned "4 failures, 6 errors" split (an initial draft compared the tuple directly and gave 5/5).

### Otherwise
Plan executed as written. Production edits were made in the planned order (a, b+c, d+e back-to-back, f) between ticks (12:04-12:06 PDT, well before the 12:15 tick), with an import smoke check after each; no tick saw a half-edited module. No test was ever imported by the runner.

## Rule chosen

Run provenance derived from `campaign_attribution`'s existing alias tables (`Q(source__in=(lco_queue, soar_queue)) | Q(site__obscode__in=<FTN/FTS/SOAR obscodes>)`), never new obscode literals (AST gate confirms). A code-shape regex was rejected because nothing in the codebase records the portal's naming scheme; it would be a guess, and WR-07 deliberately keeps `_PROPOSAL_CODE_RE` a charset guard only.

## Live database read-only check (sqlite `mode=ro`, `query_only=ON`)

Active watched proposal: `KEY2026B-004`. Run codes and the rule's verdict:

| Runs | source | site | code | fetchable |
| ---- | ------ | ---- | ---- | --------- |
| 1 | legacy | E10 | LCO2026A-003 | yes |
| 69-75 | lco_queue | none | KEY2026B-004 | yes |
| 76 | classical_file | 809 | 117.2A2N.001 | **no** |

So fetchable: `KEY2026B-004`, `LCO2026A-003` (proposals: 2); not fetchable: `117.2A2N.001` (not fetchable: 1). The next tick should read `step proposal_allocation: ok | proposals: 2, rows written: ..., failed: 0, not fetchable: 1` with `exit=0`. At the time of writing the log's last tick was still the pre-fix 12:00 tick (`exit=1`); the 12:15 tick is the first on fixed code. Nothing was written to the live database.

## Paired docs

- Runbook (`docs/runbooks/telescope_runs_calendar.rst`): step 5 reworded, new paragraph after the step-failure paragraph, sample log line gained `not fetchable: 0`, troubleshooting entry gained the third cause and the fix sentence. Sphinx build passes.
- `CLAUDE.md`: exactly one new paired-docs map entry for `proposal_allocation.py` (runbook sections, not a notebook).
- `load_telescope_runs_demo.ipynb`: markdown cell `94a24de0` only (3 insertions, 1 deletion; verified every other cell and the metadata are identical; not re-executed, per the orchestrator decision).

## F7 note

The "Fix landed" paragraph is in `.planning/v2.4-INTENT-REVIEW.md` inside F7, **left uncommitted** beside the operator's own edits (`git status` still ` M`). Both F7 checkboxes are unticked; the live confirmation and ticking them are the operator's.

## Known Stubs

None.

## Threat Flags

None. No new network endpoints, auth paths or schema changes; the change only narrows which codes reach the existing credentialed GET.

## Self-Check: PASSED

- Commits 0f56776, 433eb9f, cfab058 exist on `issue37-telescope-runs-calendar`.
- Gates passed: import smoke (no heavy imports), AST gate, pin gate (134 unchanged), runbook/notebook/CLAUDE.md gates, Sphinx, `pre-commit run ruff` and `ruff-format --all-files`, full suite 1842 OK.
- `.planning/v2.4-INTENT-REVIEW.md` unstaged; untracked `.gsd/`, `.planning/agent-history.json`, `reqgroup_2682493.json`, `src/fomo_db_20260929.sqlite3` untouched and uncommitted.
