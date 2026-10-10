---
phase: 40-notebook-isolation-attribution-page
plan: 05
subsystem: testing
tags: [notebooks, django, settings, scratch-database, fail-closed, subprocess-tests, gap-closure]
gap_closure: true
gap_ids: [CR-01, WR-02, IN-01, IN-02, IN-06, IN-07]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-01's scratch-database helper docs/notebooks/fomo_nb_utils.py and its 14 tests; 40-04 as the last wave-4 plan"
provides:
  - "setup_scratch_django reads and checks the Django settings before django.setup(); a refusal leaves apps.ready False and every database alias on Django's dummy backend"
  - "Any later setup failure (for example migrate) closes every connection, switches every alias to the dummy backend, removes the scratch directory and restores the routing environment before re-raising"
  - "check_database_settings: refuses any alias other than default, any non-empty DATABASE_ROUTERS and a missing or empty default NAME, in that order"
  - "teardown_scratch_django accepts only the directory this interpreter's setup created, directly inside the temp directory, and restores FOMO_DATABASE_PATH / DJANGO_ALLOW_ASYNC_UNSAFE"
  - "The Django ready line no longer prints the repository path; snapshot mode and a failed snapshot are tested for real"
  - "docs/notebooks/README.md helper section describes the check-before-setup order"
affects: [40-06 guard rule for re-routing after the helper call, 40-07 notebook re-execution and ledger, phase 40 verification]

actuals:
  tokens: 9600
  tasks: 3
  commits: 3

plan_head_before: a274e1697fbbeb75672c4c857a17a2afe51fb5ab
plan_head_after: 819321ec6e3fd1c3f2fff9a0d0cb4b0274415ddb

tech-stack:
  added: []
  patterns:
    - "Fail closed: on any failure once the settings are loaded, every alias in settings.DATABASES is switched in place to django.db.backends.dummy (the connection handler shares the same dict) and the cached wrappers are dropped"
    - "The refuse-up-front check sits outside the try block, so the disable step can only run on a setup the helper itself started"
    - "Module-level _active_setup record carries the scratch directory and previous environment between setup and teardown without changing the call shape"
    - "Subprocess tests live in separate SimpleTestCase classes so --parallel spreads them across workers"

key-files:
  created: []
  modified:
    - docs/notebooks/fomo_nb_utils.py
    - solsys_code/tests/test_notebook_helper.py
    - docs/notebooks/README.md

key-decisions:
  - "WR-02 (developer decision 2026-10-09): refuse extra aliases and routers rather than support them; the teardown line about the developer database keeps its wording"
  - "Already-configured Django settings are refused up front with a restart-the-kernel message, outside the try"
  - "_active_setup is module-level state (documented exception to the avoid-module-level-state note) because returning it would change the call shape all eight notebooks depend on"
  - "The environment restore happens after close_all(), which is async_unsafe"

patterns-established:
  - "Child-process refusal probe: run the helper, then try a raw cursor write on every alias and an ORM query, and assert ImproperlyConfigured"

requirements-completed: [WARN-03, WARN-05]

coverage:
  - id: D1
    description: "A refused setup leaves apps.ready False, every alias unreachable (ImproperlyConfigured), the override database never created, the routing environment restored and a second setup refused with a restart message (CR-01)"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRefusalLeavesNoDatabase.test_a_refused_setup_leaves_django_unable_to_reach_any_database"
        status: pass
    human_judgment: false
  - id: D2
    description: "A failure after django.setup() (forced migrate error) leaves a raw cursor and the ORM both raising ImproperlyConfigured and no scratch directory"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRefusalLeavesNoDatabase.test_a_failure_after_django_setup_leaves_no_usable_database"
        status: pass
    human_judgment: false
  - id: D3
    description: "The up-front refusal never disables the database of a Django that was already set up (the test runner's own settings and connection are unchanged)"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestSnapshotAndRouting.test_setup_refuses_when_django_is_already_set_up"
        status: pass
    human_judgment: false
  - id: D4
    description: "check_database_settings refuses a second alias, DATABASE_ROUTERS and a missing or empty default, in a fixed order, before django.setup() (WR-02)"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestDatabaseSettingsCheck"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRefusalLeavesNoDatabase.test_a_second_alias_or_a_router_is_refused_before_django_setup"
        status: pass
    human_judgment: false
  - id: D5
    description: "The Django ready line omits the repository path (IN-02); snapshot mode and a failed snapshot are executed by tests (IN-07 helper-test parts)"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRunsForReal.test_fresh_setup_routes_migrates_and_tears_down"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperSnapshotRunsForReal.test_snapshot_setup_copies_the_developer_rows_migrates_and_tears_down"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestSnapshotAndRouting.test_a_failed_snapshot_removes_the_scratch_directory"
        status: pass
    human_judgment: false
  - id: D6
    description: "Teardown refuses a fomo-notebook-db-* directory outside the temp directory or one setup did not create (IN-06) and restores the routing environment (IN-01)"
    requirement: WARN-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestSnapshotAndRouting.test_teardown_refuses_a_scratch_named_directory_outside_the_temp_directory"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_notebook_helper.py#TestHelperRestoresTheEnvironment.test_teardown_restores_the_routing_environment"
        status: pass
    human_judgment: false
  - id: D7
    description: "The README helper section reads plainly: settings checked before Django starts, what a refusal leaves behind, what the teardown refuses and restores"
    requirement: WARN-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestReadmeStatesTheIsolationRule"
        status: pass
    human_judgment: true
    rationale: "Test pins the D-04 sentence and mode lists and the AST/whitespace check pins the setup-check sentence; whether the surrounding prose reads clearly is a human judgment (the task's human-check at /gsd-verify-work)"

duration: 15min
completed: 2026-10-10
status: complete
---

# Phase 40 Plan 05: Notebook helper fails closed Summary

**The notebook scratch-database helper now checks Django's settings before django.setup() and, on any refusal or later failure, leaves every database alias on Django's dummy backend, so no later cell can write to a refused database (closes VERIFICATION gap 1 / CR-01).**

## Performance

- **Duration:** 15 min
- **Started:** 2026-10-10T04:42:28Z
- **Completed:** 2026-10-10T04:57:17Z
- **Tasks:** 3 (tracer, auto, auto)
- **Files modified:** 3

## Accomplishments

- CR-01 closed: the settings are read through the lazy `django.conf.settings` and checked before `django.setup()`. A child-process test proves that after a refusal `apps.ready` is False, a raw `cursor()` on every alias raises `ImproperlyConfigured`, the override database file is never created, both routing variables are absent again and a second setup call says "restart the kernel". A second child test forces `migrate` to fail after `django.setup()` and shows both a raw cursor and `User.objects.count()` raise `ImproperlyConfigured` (before the fix they raised `OperationalError`, safe only by accident).
- WR-02: `check_database_settings(databases, database_routers, scratch_db_path)` refuses any alias other than `default` (even one naming the scratch file; sorted, named with repr), then any non-empty `DATABASE_ROUTERS`, then a missing or empty default `NAME`. Five in-process tests plus a two-case child test cover it.
- IN-01, IN-06: `_active_setup` records the scratch directory and previous values of `FOMO_DATABASE_PATH` and `DJANGO_ALLOW_ASYNC_UNSAFE` only after migrate succeeds. Teardown refuses a directory that is not directly inside the resolved temp directory or was not created by this interpreter's setup (both checks before Django is imported), then closes connections, removes the directory, restores the environment and clears the record. A child test confirms a preset `FOMO_DATABASE_PATH` comes back and an absent `DJANGO_ALLOW_ASYNC_UNSAFE` is removed, never set to an empty string.
- IN-02: the `Django ready:` line prints only the settings module; the fresh test asserts the exact line and that the checkout path is absent.
- IN-07 helper-test parts: `re.escape(tmp)` in the Resolved regexes, an executed snapshot-mode child test against a temp-root "developer database" (marker row copied, migrations present, source sha256 unchanged, no directory left) and a forced snapshot failure test (directory created then removed).
- README helper section rewritten around the new order; the D-04 sentence and both mode lists are byte-equal after whitespace normalisation.

## Task Commits

1. **Task 1 (tracer): refusal leaves no reachable database; settings checked before django.setup()** - `222a25f` (fix) - closes CR-01 (VERIFICATION gap 1), IN-01 failure half; WARN-03/WARN-05
2. **Task 2: second alias or router refused; evidence line drops the repository path; snapshot mode tested** - `7cfd78e` (fix) - closes WR-02, IN-02, IN-07 helper-test parts
3. **Task 3: teardown location and recorded-directory checks, environment restore, README** - `819321e` (fix) - closes IN-06, IN-01 teardown half

**Plan metadata:** recorded by the SUMMARY commit that follows this file.

## TDD evidence

RED logs (`$HOME/tmp/phase40-05-t{1,2,3}-red.log`), each failing on the planned assertion for the intended reason:

- Task 1 RED: `FAIL: test_a_refused_setup_leaves_django_unable_to_reach_any_database` (`'APPS_READY False' not found ... APPS_READY True ... WRITE_SUCCEEDED default`) and `FAIL: test_a_failure_after_django_setup_leaves_no_usable_database` (`WRITE_REFUSED default OperationalError`, not `ImproperlyConfigured`). `FAILED (failures=2)`.
- Task 2 RED: `ERROR` x7 for `TestDatabaseSettingsCheck` (no `check_database_settings`), `FAIL` for `test_a_second_alias_or_a_router_is_refused_before_django_setup` (both subtests) and `test_fresh_setup_routes_migrates_and_tears_down` (repository path in the line). `FAILED (failures=3, errors=7)`. The snapshot and failed-snapshot tests passed against the existing code, as the plan expected (they cover untested existing behavior).
- Task 3 RED: `FAIL` for `test_teardown_restores_the_routing_environment` (`AFTER_FOMO False`, `AFTER_ASYNC 'true'`), `test_teardown_refuses_a_scratch_directory_setup_did_not_create` and `test_teardown_refuses_a_scratch_named_directory_outside_the_temp_directory` (`RuntimeError not raised`). `FAILED (failures=3)`.

GREEN: Task 1 `Ran 59 tests ... OK` (helper + guard + projector-demo modules); Task 2 `Ran 62 tests ... OK`; Task 3 `Ran 70 tests ... OK`. Final `solsys_code.tests.test_notebook_helper`: `Ran 27 tests ... OK` (14 pre-existing, none removed, skipped or tagged; 13 new).

AST checks (all printed OK): "up-front refusal outside the try, settings checked before django.setup(), fail-closed handler order disable -> rmtree -> restore -> raise"; "check_database_settings (aliases, then routers, then the default path) runs before django.setup(); no print formats the repository path"; "teardown checks location and record before touching Django, restores the environment, keeps its lines; README carries the setup-check sentence and unchanged rule and lists".

django-test pre-commit hook durations on the three commits: 77 s, 85 s, 79 s (full hook, parallel).

## Files Created/Modified

- `docs/notebooks/fomo_nb_utils.py` - fail-closed setup, `check_database_settings`, `_disable_django_databases`, `_restore_environ`, `_ROUTING_ENVIRON`, `_active_setup`, stricter teardown
- `solsys_code/tests/test_notebook_helper.py` - 13 new tests in four new classes plus two changed tests; module helpers `_write_override_settings` and `_refusal_probe_script`
- `docs/notebooks/README.md` - helper section describing the check-before-setup order, refusal state, teardown checks and environment restore

## Decisions Made

See `key-decisions` above. The WR-02 choice (refuse, do not support, extra aliases and routers) is the developer's decision of 2026-10-09; the up-front "already configured" refusal and the module-level `_active_setup` record are planner decisions carried out as written.

## Deviations from Plan

### TDD gate shape

**1. [Process - RED not committed separately] Tests and fix committed together per task**
- **Found during:** all three tasks
- **Issue:** the executor brief asks for a RED commit before GREEN for TDD tasks, but this plan specifies one full-hook commit per task holding both files ("Never SKIP= for these commits"). A separate RED commit would fail the django-test hook because the new tests fail by design, and bypassing the hook contradicts the plan.
- **Fix:** followed the plan. RED was run and its log kept for each task (see TDD evidence above); each task then went in as one commit with the full hook. `git log --grep '^test(40-05)'` finds no RED commit, which is a TDD gate flag for the reviewer.
- **Files modified:** none beyond the plan's
- **Committed in:** `222a25f`, `7cfd78e`, `819321e`

### Auto-fixed Issues

**2. [Rule 3 - Lint] Over-long test lines caught by ruff**
- **Found during:** Tasks 1 and 2
- **Issue:** two test lines exceeded 120 columns (E501) as first written.
- **Fix:** let `ruff-format` rewrap them, re-ran both ruff hooks clean.
- **Files modified:** `solsys_code/tests/test_notebook_helper.py`
- **Committed in:** `222a25f`, `7cfd78e`

---

**Total deviations:** 1 process deviation (no separate RED commit), 1 trivial lint fix.
**Impact on plan:** none on behavior or scope; every acceptance criterion and verify command passed.

## Issues Encountered

None. A stale `/tmp/fomo-notebook-db-i_k4rosi` directory dated 2026-10-02 exists on the machine; it predates this plan (all this plan's tests set their own `TMPDIR`) and was left alone.

## Paired docs

The paired notebooks (all eight under `docs/notebooks/pre_executed/`) are re-executed in plan 40-07, not here, because the printed `Django ready:` line changed. No `docs/runbooks/` page describes the helper (`grep -rn fomo_nb_utils docs/runbooks` is empty), so no runbook changes. `docs/notebooks/README.md` was updated here.

## Out of scope, left open

WR-03, IN-03, IN-04 and IN-07's `campaign_views.py` alias tidy-up (`_DISMISSED_TABLE_PREFIX` / `_CONFIRMED_TABLE_PREFIX`) were not touched; no notebook, `test_pre_executed_notebooks.py`, `campaign_views.py`, `campaign_tables.py`, `test_bootstrap5_rendering.py` or template changed. 40-07 records IN-07 as partly fixed.

## Known Stubs

None.

## Threat Flags

None. The plan's register (T-40-16 to T-40-21) is mitigated and tested; nothing new at a trust boundary.

## Next Phase Readiness

Ready for 40-06 (guard rule for source that re-routes Django's database after the helper call) and 40-07 (re-execute the eight notebooks, close the ledger). 40-07 should cite commits `222a25f` (CR-01, IN-01 failure half), `7cfd78e` (WR-02, IN-02, IN-07 helper-test parts) and `819321e` (IN-06, IN-01 teardown half).

## Self-Check: PASSED

- `docs/notebooks/fomo_nb_utils.py`, `solsys_code/tests/test_notebook_helper.py`, `docs/notebooks/README.md` exist.
- Commits `222a25f`, `7cfd78e`, `819321e` are ancestors of HEAD; `git rev-list --count a274e16..HEAD` was 3 before this SUMMARY.
- Final `python manage.py test --noinput solsys_code.tests.test_notebook_helper`: Ran 27 tests, OK; both ruff hooks clean; none of the developer's uncommitted files staged.
