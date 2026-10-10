---
phase: 40-notebook-isolation-attribution-page
plan: 06
subsystem: testing
tags: [notebooks, ast-guard, django, database-routing, scratch-database, gap-closure, tdd]
gap_closure: true
gap_ids: [WR-01]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-04's guard module solsys_code/tests/test_pre_executed_notebooks.py with rules R0-R11 and the eight isolated notebooks"
provides:
  - "Rule R12 (_database_rerouting_problems): runs for every notebook, Django-backed or not, and reports one problem per (cell, name) for eleven database-routing names, two connection-handler attribute chains and any sqlite3 import"
  - "Both VERIFICATION gap 2 probes (settings_dict re-route, override_settings re-route) replayed on a copy of the real import_campaign_csv_demo.ipynb and reported"
  - "R0's unparseable-cell list built from the same constants, so a magic cell cannot hide a re-route and the two lists cannot drift"
  - "A module docstring that states what R0-R12 check and what the guard cannot see"
affects: [40-07 notebook re-execution and ledger, phase 40 verification]

actuals:
  tokens: 3300
  tasks: 2
  commits: 2

plan_head_before: 6fe9e071efb6e0a035edb815c67984351a95867f
plan_head_after: 26940ce3de81af1598261ea9c573db725ac59eef

tech-stack:
  added: []
  patterns:
    - "Report names, not intent: the guard fails closed on any notebook that names a database-routing surface, because a static rule cannot prove the target"
    - "Replay a verifier probe on a deep copy of a real notebook (append to the setup cell) so execution counts and committed output stay valid and only the rule under test can fire"
    - "Derive the R0 suspect-text list from the R12 constants instead of repeating values"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py

key-decisions:
  - "R12 reports names, so a future demo that only reads settings.DATABASES would also be reported; accepted because the helper already prints the resolved database"
  - "RED stage defined the R12 constants first (without the rule), so the RED failures are assertions on an empty problem list rather than NameErrors"
  - "R10, TEARDOWN_OUTPUT and REFERENCE_TEARDOWN_OUTPUT left untouched (developer decision on WR-02)"

patterns-established:
  - "Rule R12 is wired before the is_django_backed early return, beside R5-R7"

requirements-completed: [WARN-02]

coverage:
  - id: D1
    description: "A settings_dict re-route appended after the helper call to a copy of a real notebook is reported by the guard, every problem carrying the (R12: marker and naming settings_dict and developer_database_path"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestRerouteRuleOnARealNotebook.test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D2
    description: "An override_settings(DATABASES=...) re-route appended to a copy of a real notebook is reported, naming override_settings, DATABASES and developer_database_path"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestRerouteRuleOnARealNotebook.test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D3
    description: "R12 covers every spelling (settings_dict, DATABASES, DATABASE_ROUTERS, connections.settings/databases, override_settings, modify_settings, helper internals, sqlite3, backend wrappers, string and keyword forms), applies to notebooks that import no Django, ignores comments and markdown, and R0 flags an unparseable re-routing cell"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestRerouteRuleOnSyntheticNotebooks"
        status: pass
    human_judgment: false
  - id: D4
    description: "All eight real notebooks still pass the whole guard with R12 in place"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestPreExecutedNotebooksUseTheHelper.test_every_real_pre_executed_notebook_is_isolated"
        status: pass
    human_judgment: false
  - id: D5
    description: "The module docstring is accurate about what the guard checks and cannot see"
    verification: []
    human_judgment: true
    rationale: "Docstring accuracy was checked structurally (required terms present, old completeness claim absent) but whether the prose reads honestly is a reviewer judgment"

duration: 6min
completed: 2026-10-10
status: complete
---

# Phase 40 Plan 06: Notebook guard rule R12 (re-routing after setup) Summary

**AST rule R12 makes the pre-executed-notebook guard fail any notebook that re-routes Django's database after the helper call (settings_dict, DATABASES, override_settings, connections.settings, sqlite3 and the helper's own internals), proven by replaying the verifier's two probes on a copy of a real notebook.**

## Performance

- **Duration:** about 6 min
- **Started:** 2026-10-10T04:59Z (start of the first test run)
- **Completed:** 2026-10-10T05:06Z
- **Tasks:** 2 (a tracer, then an expansion task)
- **Files modified:** 1 (`solsys_code/tests/test_pre_executed_notebooks.py`)

## Accomplishments

- VERIFICATION gap 2 / review WR-01 is closed. Before this plan the guard returned `[]` for a copy of `import_campaign_csv_demo.ipynb` with either verifier probe appended to its setup cell. Now both are reported, every problem carrying `(R12:`.
- R12 runs before the `is_django_backed` early return, so it covers every notebook. It matches names, attributes, keyword names, import alias names/asnames and exact string constants for the eleven REROUTE_NAMES, the attribute chains ending in `connections.settings` / `connections.databases`, and any import with a `sqlite3` module component.
- R0's unparseable-cell suspect list now includes every R12 name and both suffixes, referencing the constants, so a magic cell cannot hide a re-route.
- The module docstring no longer claims the only blind spot is AST-unreadable code. It lists what R0-R12 check and names the blind spots: runtime-built code (`exec`/`eval`/`compile`, `importlib` or `getattr` with a computed name), code in imported modules, a re-route through an object bound to another name, and library internals.
- `REFERENCE_SETUP_OUTPUT`'s first line is now `Django ready: settings module='src.fomo.settings'`, matching 40-05's helper output.

## Task Commits

1. **Task 1 (tracer): rule R12 wired for every notebook; both verifier probes replayed on a real notebook** - `50ffb59` (test)
2. **Task 2: R12 covers every spelling, a synthetic test per form, accurate docstring** - `26940ce` (test)

**Plan metadata:** committed separately after this file (docs).

## TDD evidence

Per-task full-hook commits (CLAUDE.md: no SKIP, no --no-verify) make a separate failing RED commit impossible, so the RED runs were kept as logs, as 40-05 did. Each task still ran RED, then GREEN, inside the one commit.

**Task 1 RED** (`$HOME/tmp/phase40-06-t1-red.log`): the R12 constants (`REROUTE_NAMES`, `REROUTE_DOTTED_SUFFIXES`, `REROUTE_MARKER`) were defined first without the rule, so the failures are assertions on the planned behavior rather than NameErrors. Semantic assessment: both target tests executed and failed on `assertTrue(problems)` with `[]` (the guard missed the re-route); the unmodified-notebook test passed, as it should.

```
FAIL: test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook (...TestRerouteRuleOnARealNotebook...)
AssertionError: [] is not true : the guard missed a settings_dict re-route after the helper call
FAIL: test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook (...TestRerouteRuleOnARealNotebook...)
AssertionError: [] is not true : the guard missed an override_settings re-route after the helper call
Ran 3 tests in 0.081s
FAILED (failures=2)
```

**Task 1 GREEN:** `Ran 41 tests ... OK` for `solsys_code.tests.test_pre_executed_notebooks`.

**Task 2 RED** (`$HOME/tmp/phase40-06-t2-red.log`): `Ran 9 tests ... FAILED (failures=16)`. Eight of the nine tests failed, 16 FAIL lines counting subTests: `test_sqlite3_or_a_backend_wrapper_is_a_problem` (5 subtests), `test_connection_handler_settings_are_a_problem` (2), `test_databases_or_routers_in_any_spelling_is_a_problem` (2), `test_helper_internals_are_a_problem` (2), `test_settings_dict_in_any_spelling_is_a_problem` (1, the getattr string form), `test_override_or_modify_settings_is_a_problem` (1, modify_settings), `test_an_unparseable_cell_that_reroutes_is_a_problem` (1), and `test_the_rule_applies_to_a_notebook_that_does_not_import_django` (1). The one passing test, `test_names_in_comments_or_markdown_are_not_a_problem`, is a negative case that is correctly green before and after. All failures were assertions that no problem was reported, not setup errors.

**Task 2 GREEN:** `Ran 50 tests ... OK` for `solsys_code.tests.test_pre_executed_notebooks`; `Ran 55 tests ... OK` for that module plus `solsys_code.tests.test_projector_demo_notebook`.

## Verification

- Final count: `solsys_code.tests.test_pre_executed_notebooks` runs 50 tests (38 before this plan), all OK. With `test_projector_demo_notebook`, 55 OK.
- `test_every_real_pre_executed_notebook_is_isolated` passed with R12 in place at both commits, and so did `test_the_real_notebook_check_is_not_vacuous`: none of the eight real notebooks uses an R12 name.
- Structural checks: R12 called before the Django-backed return; R5/R6/R7/R4 calls, `EXPECTED_NOTEBOOKS`, `TEARDOWN_OUTPUT`, `REFERENCE_TEARDOWN(_OUTPUT)` unchanged since `0ad6991`; no test removed; no skip/expectedFailure/tag decorator; `REROUTE_NAMES` is exactly the eleven names.
- `pre-commit run ruff` and `ruff-format` clean; exactly two guard commits after `0ad6991`, each holding only the guard module.
- No notebook, `docs/notebooks/fomo_nb_utils.py` or `docs/notebooks/README.md` changed in this plan, so no paired doc (CLAUDE.md paired-docs rule) was triggered.

## Files Created/Modified

- `solsys_code/tests/test_pre_executed_notebooks.py` - rule R12 and its constants, `_append_to_setup_cell` / `_rerouting_names` helpers, `TestRerouteRuleOnARealNotebook` (3 tests), `TestRerouteRuleOnSyntheticNotebooks` (9 tests), R0 suspect list, `REFERENCE_SETUP_OUTPUT`, module docstring.

## Decisions Made

- Followed the plan as specified. R12 reports names, not intent (accepted limitation noted in the plan's flagged assumptions).
- Defined the R12 constants in the RED step so the RED failures are meaningful assertions (the plan allowed FAIL or ERROR; this is stricter).

## Deviations from Plan

None - plan executed exactly as written. (The formatter reflowed the new code once before the Task 1 commit; the hooks then passed.)

## Issues Encountered

None.

## Known Stubs

None.

## Threat Flags

None. The change is confined to a test module and adds no endpoint, auth path or file access.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for 40-07 (notebook re-execution and ledger). Commit hashes for its ledger: Task 1 `50ffb59`, Task 2 `26940ce`.
- Remaining blind spots are documented in the guard's docstring (runtime-built code, imported modules, renamed objects, library internals); they are accepted, not open defects.

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py` exists; commits `50ffb59` and `26940ce` are ancestors of HEAD.
- Both RED logs exist under `$HOME/tmp/` with FAIL lines for the named tests.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-10*
