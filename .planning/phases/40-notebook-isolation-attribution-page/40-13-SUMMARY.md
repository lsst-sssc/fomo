---
phase: 40-notebook-isolation-attribution-page
plan: 13
subsystem: testing
tags: [notebooks, guard, gap-closure, automagic, shutil, bytes-literals, ipykernel]
gap_closure: true
gap_ids: [CR-01, WR-01, WR-02, IN-02]
status: complete

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-10 guard (R0 allow-list and automagic rule, R6 copy and write functions, R12) and 40-11/40-12 docstring and override"
provides:
  - "R0's automagic rule reports a one-line IPython magic or alias cell whatever an earlier cell binds (no source-level binding exemption)"
  - "R6 reports copy2, copyfile, copytree and copyfileobj at every identifier and string part, and follows shutil names across the whole notebook for copy and move"
  - "R5 and R7 read str and bytes constants"
  - "ipykernel>=7 declared in the dev extra"
affects: [40-14, verify-work, 40-re-verification]

actuals:
  tokens: 5570      # chars/4 over the added and removed lines of the realized diff (guard module and pyproject.toml); excludes this SUMMARY
  tasks: 3
  commits: 3
plan_head_before: 0d16c3aac004953a5a8d68b694d2677a25a4dd11
plan_head_after: ccfc578b243e286f2e61623c4734dc9396957219

tech-stack:
  added: []
  patterns:
    - "A static check never trusts a source-level binding for a decision IPython makes from the namespace at run time; only facts true at runtime whatever the source says (builtin, keyword, a following = or ,) stay exempt"
    - "Shutil names are collected once over the whole notebook (cells and string code) by a fixpoint over assignment-shaped bindings; an attribute is tied to shutil when its value mentions a shutil name"

key-files:
  created:
    - .planning/phases/40-notebook-isolation-attribution-page/40-13-SUMMARY.md
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - pyproject.toml

key-decisions:
  - "D-G7 (1): the exemption for a name an earlier cell binds is removed from R0's automagic rule; _module_bindings is deleted and _automagic_word takes the cell's view only"
  - "D-G7 (2): DISTINCTIVE_COPY_NAMES are reported wherever written; copy and move stay tied to shutil names, collected notebook-wide"
  - "D-G8 (1): R5 and R7 read bytes constants through _constant_text (latin-1)"
  - "D-G8 (2): ipykernel>=7 as a floor, not an exact pin (7.4 needs Python 3.11 while CI tests 3.10 to 3.12)"

requirements-completed: [WARN-02, WARN-05]

coverage:
  - id: D1
    description: "A one-line cell starting with an IPython magic or alias name is reported whatever an earlier cell binds; seven bindings of rm (bare annotation, if False, empty for, except as, plain assignment, import, assignment then del) before an rm alias cell are reported on a copy of a real notebook, each binding alone is clean"
    requirement: "WARN-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name"
        status: pass
    human_judgment: false
  - id: D2
    description: "R6 reports both review rebinding forms, the copy and move variants, every assignment-shaped, cross-cell, string-code, module-attribute and literal-lookup form, and the distinctive copy names at every identifier; look-alikes stay clean"
    requirement: "WARN-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem"
        status: pass
    human_judgment: false
  - id: D3
    description: "R5 and R7 report a bytes literal naming fomo_db.sqlite3 or FOMO_DATABASE_PATH"
    requirement: "WARN-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem"
        status: pass
    human_judgment: false
  - id: D4
    description: "pyproject.toml's dev extra lists ipykernel>=7 with a comment naming the guard; nothing else changed"
    verification:
      - kind: other
        ref: "40-13-PLAN.md Task 3 first <automated> script (printed 'OK: the dev extra gains exactly ipykernel>=7 ...')"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-10-10
---

# Phase 40 Plan 13: Notebook guard round 5 (code half) Summary

**The notebook guard no longer trusts source-level bindings for IPython's automagic, follows a rebound shutil across the whole notebook, reads bytes literals in R5/R7, and declares ipykernel in the dev extra.**

## Performance

- **Duration:** 11 min (2026-10-10T22:13Z to 22:24Z)
- **Tasks:** 3 (one tracer, two auto)
- **Files modified:** 2 (`solsys_code/tests/test_pre_executed_notebooks.py`, `pyproject.toml`)

## Commits

| Task | Hash | Closes |
|------|------|--------|
| 1 (tracer) | `905408a` | CR-01, verification gap 1 (D-G7 1) |
| 2 | `4f02628` | WR-01 (D-G7 2), WR-02 (D-G8 1) |
| 3 | `ccfc578` | IN-02 (D-G8 2) |

Base for every "unchanged since" check: `e08f8eb`. Nothing pushed. The metadata (this SUMMARY, STATE, ROADMAP) is committed separately.

## Accomplishments

- **R0 (CR-01).** `_module_bindings` is deleted, `_automagic_word(view)` has one parameter, and the R0 loop keeps no set of bound names. A one-line cell starting with a line magic or alias name is reported whatever an earlier cell binds. Only a builtin or keyword first word and a following `=` or `,` stay exempt, because they hold at runtime whatever the notebook's source says. The problem text now says the name is reported unless it is bound when the cell runs.
- **R6 (WR-01).** New `DISTINCTIVE_COPY_NAMES = ('copy2', 'copyfile', 'copytree', 'copyfileobj')` reported at every identifier `_identifiers` records and as string parts. New `_mentions_shutil` and `_shutil_names` (plus two small private helpers, `_binding_pairs` and `_target_names`). `copy` and `move` are reported when read as an attribute of an expression mentioning a shutil name, called or not.
- **R5/R7 (WR-02).** `_database_routing_problems` reads str and bytes constants through `_constant_text`.
- **IN-02.** `"ipykernel>=7", # Imported by the notebook guard (...) for KernelMagics` after `"ipython",`.

## TDD evidence

- Task 1 RED (`$HOME/tmp/phase40-13-t1-red.log`): `Ran 4 tests`, `FAILED (failures=8)`; FAIL lines for the renamed test with subTests `[time bound by an earlier import]`, `[bare annotation]`, `[if False branch]`, `[empty for loop]`, `[except as]`, `[plain assignment]`, `[import]`, `[assignment then del]`. GREEN (`phase40-13-t1.log`): `Ran 82 tests ... OK`.
- Task 2 RED (`$HOME/tmp/phase40-13-t2-red.log`, run on the committed Task 1 guard plus the new tests): `Ran 35 tests`, `FAILED (failures=27)`: 3 subTests of `test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed`, 22 of `test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem` (including the two review forms), 2 of `test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem`. The look-alike test passed already, as the plan expected. GREEN (`phase40-13-t2.log`): `Ran 91 tests ... OK` (guard plus projector demo tests).
- Final: `Ran 121 tests in 55.833s ... OK` (guard, notebook helper, projector demo).

Process note: the Task 2 implementation was written before its tests; RED evidence was produced afterwards by running the new tests against the committed Task 1 guard (the same file without the Task 2 implementation), so every RED failure is real. The test and implementation land in one commit, as the plan prescribes.

## Check results (OK lines)

- Task 1 replay: `OK: independent replay -- seven bindings of rm then an alias cell reported on a real notebook (each binding alone clean); import time then time reported; assignments and plain calls clean; no bound-name set in R0; eight real notebooks clean with no cell starting with a magic or alias name`
- Task 1 commit check: `OK: ruff clean; the automagic commit holds exactly the guard module (905408a...)`
- Task 2 replay: `OK: independent replay -- both review rebinding forms and the copy and move variants reported on a real notebook; every assignment-shaped, cross-cell, string-code, module-attribute and literal-lookup form reported; distinctive names reported at every identifier; look-alikes clean; bytes literals reported by R5 and R7; eight real notebooks clean`
- Task 2 no-weakening: `OK: no weakening -- 86 guard tests (one renamed, none removed or skipped), module docstring and frozen constants unchanged, deny-list and rule calls kept, eight real notebooks clean`
- Task 2 commit check: `OK: ruff clean; the R6 and R5/R7 commit holds only the guard module (4f026282...); two guard commits after e08f8eb`
- Task 3 pyproject check: `OK: the dev extra gains exactly ipykernel>=7 with a comment naming the guard (installed 7.3.0); nothing else in pyproject.toml changed; every workflow installs .[dev]; CLAUDE.md lists no dev extras to update`
- Task 3 commit check: `OK: validate-pyproject clean; the ipykernel commit holds only pyproject.toml (ccfc578b...) and follows the guard commits`
- Task 3 final: `OK: guard, helper and projector tests green after the three 40-13 commits`

## Boundary (what R6 does not follow) -- for 40-14's docstring

Task 2's replay printed:

- `boundary (return value): copy is not reported` -- `def get(): return shutil` then `get().copy(a, b)`.
- `boundary (call side effect): copy is not reported` -- `setattr(m, 'sh', shutil)` then `sh.copy(a, b)`.
- `boundary (module holding shutil under another name): copy is not reported` -- `import tempfile` then `tempfile._shutil.copy(a, b)`.

The distinctive names (copy2, copyfile, copytree, copyfileobj) are reported in all three shapes; only `copy` and `move` escape. Other documented over-reach: any name bound from a value that mentions shutil (for example `path = shutil.which('ls')`) becomes a shutil name, so a later `path.copy()` would be reported. A bytes literal is read as latin-1.

## Literal forms now reported (for 40-14's evidence table)

R0 (`TestR0AllowList.test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name`; the rm alias cell is `rm -r -- '../../src'` on a copy of `import_campaign_csv_demo.ipynb`):

- `rm: int` (bare annotation) then the alias cell
- `if False:` / `    rm = 1` then the alias cell
- `for rm in []:` / `    pass` then the alias cell
- `try: pass` / `except OSError as rm: pass` then the alias cell
- `rm = 1` then the alias cell
- `from os import remove as rm` then the alias cell
- `rm = 1` / `del rm` then the alias cell
- `import time` cell then a `time` cell (synthetic)
- every 40-10 case (`rm -r -- 'x'`, `cp`, `rehashx`, `time -x`, `rm += 3`, `man -k`, `store -r`, `ed`, trailing comment, trailing blank line)

R6 (`TestR6CopyAndWriteFunctions.test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem`, first four on a real-notebook copy):

- `sh = shutil` then `sh.copytree(scratch_db_dir, repo_root_path / 'src', dirs_exist_ok=True)` (review form 1)
- `from zipfile import shutil as s` then `s.copytree(...)` (review form 2)
- `sh = shutil` then `sh.copy(f, ...)` in a loop over `scratch_db_dir.iterdir()`
- `from zipfile import shutil as s` then `s.move(scratch_db_dir / 'x', ...)`
- `a = b = shutil` / `b.copy(x, y)`; `sh: object = shutil`; `if (sh := shutil):`; `sh, other = shutil, None`; `a = shutil` / `b = a` / `b.copy(x, y)`; `for sh in [shutil]:`; `with contextlib.nullcontext(shutil) as sh:`; `def f(sh=shutil):`
- `zipfile.shutil.copy(a, b)`; `sys.modules['shutil'].copy(a, b)`; `mods = [shutil]` / `mods[0].move(a, b)`; `class K: sh = shutil` / `K.sh.copy(a, b)`; `globals()['sh'] = shutil` / `sh.copy(a, b)`
- distinctive names at any identifier: `x = frame.copytree`, `def copyfile(a, b)`, `f(copy2=1)`, `from somewhere import copyfileobj`, `import pkg.copytree`

R6 across cells and string code (`test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed`): `import shutil as sh` in one cell with `sh.copy(a, b)` in the next; `sh = shutil` then `sh.move(a, b)` in the next; `timeit.timeit('sh.copy(a, b)', globals=globals())` after `import shutil as sh`.

Look-alikes kept clean (`test_copy_and_move_not_read_from_shutil_are_not_a_problem`): `df = frame.copy()`, `sh = frame` / `sh.copy()`, `import copy` / `copy.copy(x)`, `print('a copy of the run')`, `items.move_to_end('k')`, `print('copytree is not used here')`, `shutil.which('ls')`, `x = shutil_free` / `x.copy()`.

R5/R7 (`TestIsolationRulesOnSyntheticNotebooks.test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem`): `db = open(b'src/fomo_db.sqlite3', 'rb')` and `x = b'FOMO_DATABASE_PATH'`.

## Task 3 grep results

- Workflow install lines (`grep -n install .github/workflows/*.yml`): `smoke-test.yml:36` and `testing-and-coverage.yml:37` run `uv pip install --system -e .[dev]`; `testing-and-coverage.yml:68` runs `pip install -e .[dev]`; `publish-to-pypi.yml:34` installs `build` only. No workflow names ipython, jupyter or ipykernel, so none needs a change.
- CLAUDE.md (`grep -n -i "dev extra\|ipython\|jupyter"`): line 105 `tblib` (dev extra), line 301 ruff (dev extra), and prose about Jupyter notebooks (lines 131, 210, 345); no list of dev extras names ipython or jupyter, so no change.

## Deviations from Plan

None - plan executed exactly as written, with these notes:

- The Task 1 real-notebook subTests live inside the renamed test, as the plan's behavior block says ("Same test, on a copy ...").
- Four small ruff-driven fixes in the Task 2 code before its commit (two over-long lines, `strict=True` on two `zip` calls whose arguments have equal length by construction). These are style conformance of new code, not behavior changes.
- Two private helpers (`_binding_pairs`, `_target_names`) were added beside the plan's `_mentions_shutil` and `_shutil_names`, to keep `_shutil_names` readable; the plan's artifacts list names only the latter two.

## Not changed

No notebook, `docs/notebooks/README.md`, `fomo_nb_utils.py`, helper test module, guard module docstring, CLAUDE.md, `.github/` file, 40-VERIFICATION.md, 40-REVIEW.md or ledger changed in this plan. 40-14 rewrites the docstring and README and records the ledger and the developer-approved override. The fourth review's IN-01, round-2 IN-03, WR-03, IN-04, the IN-07 remainder and round-1 IN-03 were not touched.

## Known Stubs

None.

## Threat Flags

None. T-40-46, T-40-47, T-40-48 and T-40-50 are mitigated as planned; T-40-49 (the boundary above) is recorded here and accepted only by the developer at 40-14's checkpoint.

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py` and `pyproject.toml` exist and are committed (`905408a`, `4f02628`, `ccfc578` are ancestors of HEAD).
- `git rev-list --count 0d16c3a..ccfc578` is 3, matching `commits: 3`.
