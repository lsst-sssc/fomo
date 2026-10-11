---
phase: 40-notebook-isolation-attribution-page
plan: 17
subsystem: testing
tags: [notebook-isolation, static-guard, ipython, string-code, r12, r3, r4, r6, gap-closure]
gap_closure: true
gap_ids: [CR-01, CR-02, WR-01, IN-01]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-15/40-16 guard (R0-R12) in solsys_code/tests/test_pre_executed_notebooks.py at cd883a6"
provides:
  - "R12 lists IPython's shell modules, classes, app and entry points (fifteen names)"
  - "R3, R4 and the Django-backed gate read code inside strings, two levels deep"
  - "At most one write_bytes problem per cell (key by name, first text kept)"
  - "An import of tempfile's _shutil is followed for copy and move"
affects: [40-18]

actuals:
  tokens: 5355          # chars/4 over `git diff cd883a6 HEAD` of the guard module (21,422 chars)
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "_with_string_code(tree) returns (tree, *_string_code_trees(tree)) for a reader that must see string code"
    - "R6 write_bytes findings stored with setdefault under one ('write', name) key per cell"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py

key-decisions:
  - "D-G16 carried out: fifteen IPython shell names join the R12 lists; is_django_backed, R3 and R4 read code inside strings (WR-01 route: key by name, first text kept; IN-01 route: follow every SHUTIL_MODULE_NAMES import)"
  - "No docstring or comment at cd883a6 was edited (D-G17: 40-18 rewords them after the developer approves)"

requirements-completed: [WARN-02, WARN-05]

coverage:
  - id: D1
    description: "IPython's shell reached as InteractiveShell.instance() (alias then a one-line cell, run_cell, autocall, ex) is reported on a copy of a real notebook"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestSixthReviewFormsOnARealNotebook.test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D2
    description: "Django set up in a string before the helper call (timeit probe, settings.configure variant) is reported; a string-only Django notebook is Django-backed"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestSixthReviewFormsOnARealNotebook.test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestSixthReviewFormsOnARealNotebook.test_a_notebook_whose_only_django_import_is_in_string_code_is_django_backed"
        status: pass
    human_judgment: false
  - id: D3
    description: "A cell has at most one write_bytes problem; an import of tempfile's _shutil is followed for copy and move"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_a_cell_has_at_most_one_write_bytes_problem"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_a_name_bound_to_tempfile_shutil_by_an_import_is_followed"
        status: pass
    human_judgment: false

plan_head_before: d5f27e846640e999ab8c4993d3598ed1369ba7fd
plan_head_after: ca1cb7d83dc810ff6ab6bb980d071877809f7115
duration: 12min
completed: 2026-10-11
status: complete
---

# Phase 40 Plan 17: Guard code for the sixth review (CR-01, CR-02, WR-01, IN-01) Summary

**The pre-executed-notebook guard now reports IPython's shell by fifteen listed names, reads code inside strings in R3, R4 and the Django-backed gate, keeps one write_bytes problem per cell, and follows an import of tempfile's `_shutil`, with every sixth-review form pinned on a copy of a real notebook.**

## Performance

- **Duration:** about 12 min (2026-10-11T09:22Z to about 09:34Z, Django test runs on a loaded host)
- **Tasks:** 3 (one tracer, two auto/TDD), 1 file changed (the guard module), 96 guard tests (91 before)
- **Commits:** 3, each a full-hook plain `git commit` (ruff, ruff-format, notebook checks, parallel django-test)

## Commits

| Task | Hash | Closes |
|------|------|--------|
| 1 (tracer) | `c78e517283590a0ec0d23c950e382d05f2f7e26c` | CR-01 (verification gap 1): R12 lists IPython, ipykernel, traitlets and twelve shell names |
| 2 | `fdd8b3eefa7a4369ee64a33bfa29940ab9039852` | CR-02 (verification gap 1): R3, R4 and `is_django_backed` read code inside strings |
| 3 | `ca1cb7d83dc810ff6ab6bb980d071877809f7115` | WR-01 (gap 3, one write_bytes problem per cell) and IN-01 (`_shutil` import followed) |

## Verification evidence

RED logs (kept under `$HOME/tmp`, no RED commits, as in earlier rounds):

- `phase40-17-t1-red.log`: FAIL on `test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook` (subtests alias then a one-line cell, run_cell, autocall mode, ex, control get_ipython, IPython.display) and on `test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported` (import, subclass, string); 7 tests, 9 failures. The `control: %config`, wipe-alone and look-alike subtests passed before the change.
- `phase40-17-t2-red.log`: FAIL on `test_a_notebook_whose_only_django_import_is_in_string_code_is_django_backed` and on `test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook` (subtests review and verifier; both plain-code controls passed before the change); 3 tests, 3 failures.
- `phase40-17-t3-red.log`: FAIL on `test_a_cell_has_at_most_one_write_bytes_problem` (method then string part, string part then method, a real notebook) and on `test_a_name_bound_to_tempfile_shutil_by_an_import_is_followed` (`from tempfile import _shutil as sh` / `sh.move`, `import _shutil as sh` / `sh.copy`, the names set, a real notebook); 9 tests, 7 failures. The string-code-then-method, parameter-and-method and `from tempfile import _shutil` cases already passed.

GREEN (whole guard module): Task 1 `Ran 92 tests ... OK`; Task 2 `Ran 94 tests ... OK`; Task 3 `Ran 96 tests ... OK`. Final run of guard, helper and projector modules together (`--parallel auto`): `Ran 131 tests ... OK`.

Replays and checks (all printed their OK lines):

- Task 1 replay: "OK: independent replay -- the four CR-01 forms and the IPython.display import report IPython shell names on a real notebook; the get_ipython and %config controls stay reported; the wipe cell alone, exit -1 and six look-alikes are clean; the eight real notebooks name none of the fifteen names and stay clean and Django-backed".
- Task 2 replay: "OK: independent replay -- both CR-02 probes and their plain controls report django.setup() and a Django import before the helper call on a real notebook; ...".
- Task 3 replay: "OK: independent replay -- one write_bytes problem per cell (first text kept) ...; a name bound to tempfile's _shutil by an import is followed for copy and move; eight real notebooks clean".
- No-weakening checks at stages 1, 2 and 3 (against cd883a6): "OK: no weakening at stage N -- 92 / 94 / 96 guard tests (1 / 3 / 5 added, none removed or renamed; only the superseded IPyAutocall set widened)".
- Ruff and ruff-format clean via pre-commit on the guard module after each task; each commit holds only the guard module; `git status --short src/` empty; nothing pushed.

## Boundary lines the replays printed

- Task 1: `boundary (a one-line cell running an alias the notebook defined): not reported by R0; the definition is reported by R12`.
- Task 2: `boundary (Django code three strings deep): not reported` (two string levels are read).
- Shell-name counts in the eight real notebooks' code (whole-word, all fifteen names): 0 in each of backfill_lco_observations_demo, campaign_lifecycle_demo, import_campaign_csv_demo, load_telescope_runs_demo, project_observation_calendar_demo, reconcile_campaign_runs_demo, sync_gemini_observation_calendar_demo and telescope_runs_demo. All eight stay Django-backed and give `isolation_problems() == []`.

## Forms now reported (40-18's evidence table cites these)

All on a copy of `import_campaign_csv_demo.ipynb` with its committed output kept, unless noted. COMMAND is `cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput`.

| Form | Reported names / texts | Test |
|------|------------------------|------|
| `from IPython.core.interactiveshell import InteractiveShell` / `InteractiveShell.instance().alias_manager.define_alias('wipe', COMMAND)` in the setup cell, then a one-line `wipe` cell | IPython, InteractiveShell, alias_manager, define_alias | `TestSixthReviewFormsOnARealNotebook.test_ipython_shell_reached_without_get_ipython_is_reported_on_a_real_notebook` |
| same import / `InteractiveShell.instance().run_cell('!COMMAND')` | IPython, InteractiveShell, run_cell | same test |
| `from IPython import InteractiveShell` / `InteractiveShell.instance().autocall = 2` | IPython, InteractiveShell, autocall | same test |
| `from IPython import InteractiveShell` / `InteractiveShell.instance().ex('print(1)')` | IPython, InteractiveShell | same test |
| control: `get_ipython().alias_manager.define_alias('wipe', COMMAND)` | get_ipython, alias_manager, define_alias | same test |
| `from IPython.display import display` (a D-G16 consequence, pinned) | IPython | same test |
| control: a cell `%config AliasManager.user_aliases = [('wipe', COMMAND)]` | 'not parseable' | same test |
| the one-line `wipe` cell alone | `[]` (R0 knows IPython's default names only) | same test |
| look-alikes `run_cells = 1`, `x = autocall_mode`, `print('IPython-free')`, `x = ipython`, `alias_managers = []`, `x = my_InteractiveShell` | `[]` | same test |
| review probe: a cell before the setup cell, `import timeit` / `timeit.timeit("import django; django.setup(); ...flush...", number=1)` | 'calls django.setup()' and 'imports Django before the setup_scratch_django call' | `TestSixthReviewFormsOnARealNotebook.test_django_set_up_in_string_code_before_the_helper_is_reported_on_a_real_notebook` |
| verifier variant: in the setup cell just before the helper call, `timeit.timeit("from django.conf import settings; import src.fomo.settings as s; settings.configure(s); import django; django.setup(); ...", number=1)` | same two problems | same test |
| plain controls (`import django` / `django.setup()`) in both places | same two problems | same test |
| bare one-cell notebook `import timeit` / `timeit.timeit("import django; django.setup()", number=1)` | Django-backed; 'calls django.setup()' and 'no top-level' | `TestSixthReviewFormsOnARealNotebook.test_a_notebook_whose_only_django_import_is_in_string_code_is_django_backed` |
| bare one-cell notebook `print('django.setup is called by the helper')` | not Django-backed; `[]` | same test |
| `p.write_bytes(b'')` / `getattr(p, 'write_bytes')`; the reverse order; string code then method; parameter and method | exactly one write_bytes problem ('references write_bytes;' or 'names write_bytes in a string;', the first found) | `TestR6CopyAndWriteFunctions.test_a_cell_has_at_most_one_write_bytes_problem` |
| two cells, one method and one `getattr`; the method-plus-`getattr` loop appended to the real-notebook copy | one problem per cell (cells 3, 4); exactly one on the real copy | same test |
| `from tempfile import _shutil as sh` / `sh.move(a, b)`; `import _shutil as sh` / `sh.copy(a, b)`; the same import with a copy loop on the real copy | move and _shutil; copy and _shutil; copy | `TestR6CopyAndWriteFunctions.test_a_name_bound_to_tempfile_shutil_by_an_import_is_followed` |
| `_shutil_names([ast.parse('from tempfile import _shutil as sh')])` | `{'shutil', '_shutil', 'sh'}` | same test |

## Superseded on purpose

- **Widened IPyAutocall expected set.** `TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported`: its three IPyAutocall forms now also name IPython and autocall, so the one assertion changed from `{'IPyAutocall'}` to `{'IPython', 'autocall', 'IPyAutocall'}`. Nothing else in that test changed, its comment included. The matching 40-15 must_have truth 5 and 40-15 Task 2's replay assertion of the IPyAutocall-only set are superseded the same way.
- **No-gate-change prohibition.** D-G12's "no code change to `is_django_backed` or to the gating of R2-R4", 40-15's prohibition on changing `is_django_backed` and the gating, and 40-15 must_have 15.6 ("gate code unchanged") are superseded for exactly one change: `is_django_backed`, R3 and R4 also read code inside strings. `_django_imports`, the raw-source reading of each cell and the `if not is_django_backed(notebook): return problems` gating are unchanged; the `%%capture` gate pin (`test_the_django_backed_gate_reads_each_cell_as_written`) stays green.
- `from IPython.display import display` is now reported (it names IPython); pinned in the CR-01 test. No real notebook imports IPython.

## Facts 40-18's wording relies on

- R0's automagic set is IPython's default line magics and aliases, so the later one-line cell of a notebook-defined alias is not reported by R0; its definition is reported by R12 when it names a listed shell name (pinned: the `wipe` cell alone gives `[]`).
- R3, R4, R6, R12 and the Django-backed gate read code inside strings to two levels (STRING_CODE_DEPTH = 2). R5 and R7 match a string's whole text.
- `getattr(django, 'setup')()` after the helper is not reported; Django code three strings deep is not reported; a shell reached by a name or form the lists do not contain is not reported.
- A string that parses as a Django import (`x = 'import django'`) makes a notebook Django-backed and is reported by R3 before the setup call; after the setup call it stays allowed.
- No real notebook names any of the fifteen new names, imports Django in a string, or calls `django.setup()` in one.
- Near names separate: exactly equal identifiers or string parts collide, `run_cells`, `autocall_mode`, `IPython-free`, `ipython`, `alias_managers`, `my_InteractiveShell` stay allowed.
- Problem order: cell order; R12 one problem per (cell, name) sorted by name; R3 one per top-level statement before the setup call; R4 one per matching node, the cell's own code before its string code (an `import django` / `django.setup()` / `timeit.timeit('import django; django.setup()')` cell gives two identical 'calls django.setup()' problems); R6 one write_bytes problem per cell.

## Existing docstrings and comments left unchanged although now stale or claiming more than the lists (for 40-18's sweep)

By design (D-G17), none of these was edited:

- Functions: `is_django_backed` (describes code cells only), `_string_code_trees` (says the rules R6/R12 read it; R3, R4 and the gate now do too), `_django_setup_problems`, `_byte_copy_problems` (its docstring and the comment above the identifier pass now state a true count), `_shutil_names` (docstring says it follows `import shutil as X` only), `_ipython_line_magic_names`, `_automagic_word`, `_identifiers`, `_database_rerouting_problems`, `isolation_problems` (the R3 comment).
- The module docstring.
- Class docstrings: `TestCapabilityRulesOnSyntheticNotebooks`, `TestR12MatchesEveryIdentifierAndString`, `TestR6CopyAndWriteFunctions`, `TestR0AllowList` (it says the kernel's only autocall objects are exit and quit).
- Constant comments: DISTINCTIVE_COPY_NAMES and SHUTIL_MODULE_NAMES.
- The autocall test's comment (its wording says R12 names only `IPyAutocall`; the set it asserts is now wider).

## Deviations from Plan

None - plan executed exactly as written. (Ruff-format reflowed the new test source once in Task 1 and once in Task 3 before staging; formatting only.)

## Known Stubs

None.

## Threat Flags

None. The change adds no network endpoint, auth path or file-access surface; T-40-60..T-40-63 are mitigated by the tests above and T-40-64 (residual class) stays accepted only by the developer's approval in 40-18.

## Scope statement

No notebook, README, helper, helper test, projector test, docstring, comment, pyproject.toml, verification report, review report or ledger changed. 40-18 rewords the docstrings, comments and README and records the ledger and the override after the developer's approval. The D-G14 items still out of scope (the fifth review's IN-01 and IN-02, the fourth review's IN-01, round-2 IN-03, WR-03, IN-04, the IN-07 remainder, round-1 IN-03's runbook text, any attribution or template file) were not touched.

Ready for 40-18.

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py` exists and holds all three commits (`c78e517`, `fdd8b3e`, `ca1cb7d` are ancestors of HEAD, each touching only that file).
- `git rev-list --count d5f27e8..HEAD` is 3 at SUMMARY time.
