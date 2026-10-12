---
status: complete
phase: 40-notebook-isolation-attribution-page
source: [40-01-SUMMARY.md, 40-02-SUMMARY.md, 40-03-SUMMARY.md, 40-04-SUMMARY.md, 40-05-SUMMARY.md, 40-06-SUMMARY.md, 40-07-SUMMARY.md, 40-08-SUMMARY.md, 40-09-SUMMARY.md, 40-10-SUMMARY.md, 40-11-SUMMARY.md, 40-12-SUMMARY.md, 40-13-SUMMARY.md, 40-14-SUMMARY.md, 40-15-SUMMARY.md, 40-16-SUMMARY.md, 40-17-SUMMARY.md, 40-18-SUMMARY.md]
started: 2026-10-12T19:43:21Z
updated: 2026-10-12T20:19:00.941Z
---

<!--
Scope note: quick task 261012-g4y (2026-10-12) ended Phase 40 at 40-04 plus the 40-05 helper fix.
The guard (solsys_code/tests/test_pre_executed_notebooks.py) and pyproject.toml are back at 8ab5247,
the helper and its tests at 819321e (both confirmed with `git diff --stat`: no difference at HEAD).
Notebook source was kept from 17168ff, so notebook prose added by 40-07, 40-09 and 40-11 is still in
the tree and is checked under test 2. Deliverables of 40-06 and 40-08 to 40-18 that the rollback removed
are recorded as skipped with that reason, not dropped.

Commit-claim reconciliation: 40-01 (2) and 40-02 (3) match their plan windows. 40-03 to 40-18 have no
`commits:` claim (legacy, WARNING only).
Coverage blocks in 40-07, 40-09, 40-10 and 40-11 are malformed (verification kind not in the allowed set);
their entries were kept as checkpoints or skips, never dropped.
-->

## Current Test

[testing complete]

## Tests

### 1. Notebook data-dependent claims match the executed output
expected: In all eight pre-executed notebooks, markdown claims that depend on data agree with the output below them (incl. 40-07 IN-05 dir=scratch_db_dir, 40-09 IN-02 NF-14 labels and requirement-table headings, single-unit dip); no error output; counts 1..N. (40-01 D6, 40-07 D1/D2, 40-09 D1/D3)
result: pass

### 2. Notebook setup and teardown prose matches the 40-05 helper
expected: Each notebook's setup markdown says the helper checks Django's settings before django.setup() (default must be the scratch file, no other alias, no DATABASE_ROUTERS), that after a refusal or later failure no cell can reach a database until the kernel restarts, and that re-running setup raises RuntimeError. Each teardown says its second printed line ("developer database never opened for writing") is fixed text resting on the setup check, not a measurement. The setup/teardown cell output (scratch directory under /tmp/fomo-notebook-db-*, "Removed scratch database directory") agrees with that prose. (40-02 D7, 40-07 D3, 40-08 D4, 40-09 D2, 40-11 D4)
result: pass

### 3. README helper section reads plainly and is accurate
expected: docs/notebooks/README.md states the isolation rule (no notebook byte-copies or writes src/fomo_db.sqlite3; snapshot notebooks read it only through a read-only online backup; the rest start fresh), lists which notebooks snapshot and which start fresh, and explains plainly that settings are checked before Django starts, what a refusal leaves behind, and what teardown does. It reads correctly against the 819321e helper. (40-02 D7, 40-05 D7)
result: pass

### 4. Runbook attribution pass and lifecycle notebook describe the new behavior
expected: docs/runbooks/telescope_runs_calendar.rst's "How do I attribute existing calendar events and observation records to a run?" section still says the Dismissed section is folded away until you click its heading, and its "The page keeps your place" paragraph says Undo on a Dismissed row brings you back to the same page of the section, still open and with your band filter, and that Confirm, Dismiss, Confirm selected and the Confirmed-section Undo keep your place too. campaign_lifecycle_demo.ipynb's marker cell shows the High-band classes (border-0 border-start border-4 border-success) and its Undo cell shows the redirect to ?dismissed-page=N#attribution-dismissed-section. (40-04 D7)
result: pass

### 5. The proposed WARN-02 override is accurate and accepted
expected: Read .planning/quick/261012-g4y-end-phase-40-at-40-04-plus-the-40-05-hel/261012-g4y-OVERRIDE-DRAFT.md. It describes the restored 8ab5247 guard (R0, R2-R11) and the 819321e helper as they are, lists what the guard does not see, records the shell-escape residual and the unfixed round-2 WR-02 path as accepted by you, and you agree it should replace the 2026-10-11 override for ROADMAP SC 2 (first half) / WARN-02 at re-verification. (40-12 D3, 40-14 D4)
result: pass

### 6. 40-01 automated deliverables (helper, fresh routing, refusal under python -O, mode=ro snapshot, four notebooks on the helper)
expected: 40-01 D1-D5
result: pass
source: automated
coverage_id: 40-01 D1, D2, D3, D4, D5

### 7. 40-02 automated deliverables (guard R0-R11, real-notebook replay, eight notebooks pass, assert audit, dip in degrees, README rule)
expected: 40-02 D1-D6
result: pass
source: automated
coverage_id: 40-02 D1, D2, D3, D4, D5, D6

### 8. 40-03 automated deliverables (High-band marker render and Chromium tests, Bootstrap 5.3 names, class guard over every template directory)
expected: 40-03 D1-D4
result: pass
source: automated
coverage_id: 40-03 D1, D2, D3, D4

### 9. 40-04 automated deliverables (Undo keeps Dismissed open on same page, actions keep band/page/sort, clamp, tampering dropped, Chromium SC 5)
expected: 40-04 D1-D6
result: pass
source: automated
coverage_id: 40-04 D1, D2, D3, D4, D5, D6

### 10. 40-05 automated deliverables (refusal leaves no database, failure after setup, settings check, ready line, teardown refusals)
expected: 40-05 D1-D6
result: pass
source: automated
coverage_id: 40-05 D1, D2, D3, D4, D5, D6

### 11. 40-06 guard rule R12 and its docstring
expected: 40-06 D1-D5
result: skipped
reason: "Rolled back by quick task 261012-g4y (developer decision 2026-10-12): guard restored to 8ab5247"

### 12. 40-07 review ledger
expected: 40-07 D4
result: skipped
reason: "Planning ledger, not user-observable; superseded by the 261012-g4y record of decisions"

### 13. 40-08 guard capabilities, helper dummy-switch-first fix, teardown line
expected: 40-08 D1-D3
result: skipped
reason: "Rolled back by quick task 261012-g4y: c45edbc dropped by developer decision 2026-10-12, knowing round-2 WR-02"

### 14. 40-09 ledger and 2026-10-10 override
expected: 40-09 D4
result: skipped
reason: "Planning ledger and override superseded by the 261012-g4y override draft (test 5)"

### 15. 40-10 guard R0 allow-list, R6 and R12 widening, ledger
expected: 40-10 D1-D6
result: skipped
reason: "Rolled back by quick task 261012-g4y: guard restored to 8ab5247"

### 16. 40-11 guard and helper docstrings, README guard text, ledger
expected: 40-11 D1, D2, D3, D5
result: skipped
reason: "Rolled back by quick task 261012-g4y: 7af64d0 dropped with c45edbc; guard and helper restored. Notebook sentences (D4) kept and checked under test 2"

### 17. 40-12 to 40-18 guard rounds 4-7 (R0/R5/R6/R7 forms, ipykernel dev extra, docstrings, overrides)
expected: 40-12 D1-D2, 40-13 D1-D4, 40-14 D1-D3, 40-15 D1-D7, 40-16, 40-17 D1-D3, 40-18
result: skipped
reason: "Rolled back by quick task 261012-g4y: guard and pyproject.toml restored to 8ab5247, helper to 819321e; the override question these plans put to the developer is test 5"

## Summary

total: 17
passed: 10
issues: 0
pending: 0
skipped: 7
blocked: 0

## Gaps

[none yet]
