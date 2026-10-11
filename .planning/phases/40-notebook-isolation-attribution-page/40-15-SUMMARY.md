---
phase: 40-notebook-isolation-attribution-page
plan: 15
subsystem: testing
tags: [notebook-isolation-guard, r6-byte-copy, r12-dynamic-code, ipython-autocall, gap-closure, tdd]
gap_closure: true
gap_ids: [CR-01, CR-02, WR-01, WR-02]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-14 guard (R0-R12) at 1dbdb6d, the helper, and the eight pre-executed notebooks"
provides:
  - "R6 reports shutil, tempfile's _shutil and write_bytes at every identifier and string part (D-G11)"
  - "IPyAutocall is an R12 dynamic-code name (D-G13)"
  - "Pin tests for IPython's autocall of exit and quit and for the raw-source Django-backed gate (D-G13, D-G12)"
  - "Post-fix boundary list (what R6 still does not report) for 40-16's docstring, README and override reason"
affects: [40-16, phase-40-verification, phase-41]

actuals:
  tokens: 14000
  tasks: 2
  commits: 2

plan_head_before: f5f50832315a2e4d00528d47cce1a37ebf117204
plan_head_after: 1c830b13f38310a7b8a156caed5f551bad1acb06

tech-stack:
  added: []
  patterns:
    - "Report a module's names wherever they are written (via _identifiers and string parts) instead of following every way the module can be handed on"
    - "Pin a documented exception with an inert static test on placeholder source or a deep copy of a real notebook"

key-files:
  created: []
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py

key-decisions:
  - "D-G11: SHUTIL_MODULE_NAMES = ('shutil', '_shutil') as its own constant; DISTINCTIVE_COPY_NAMES left exactly as 40-13 had it"
  - "_shutil_names starts from both names, so tempfile._shutil.copy is also reported as a copy"
  - "The identifier pass reports write_bytes with exactly the first pass's text, so a cell has at most one write_bytes problem"
  - "D-G12: no code change to is_django_backed or the gating; wording and a pin test only"
  - "D-G13: only 'IPyAutocall' added to DYNAMIC_CODE_NAMES; exit -1 and quit -1 stay allowed (the kernel's only autocall objects)"

requirements-completed: [WARN-02, WARN-05]

coverage:
  - id: D1
    description: "R6 reports the six CR-01 shutil forms (parameter, immediately called lambda, map over a lambda, keyword-only parameter, match capture, class pattern) and the parameter-default control on a copy of import_campaign_csv_demo.ipynb"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D2
    description: "R6 reports write_bytes as a class-pattern attribute, a parameter and a keyword argument (one problem per cell) plus the method control"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook"
        status: pass
    human_judgment: false
  - id: D3
    description: "shutil, _shutil and tempfile._shutil.copy are reported wherever they are named; one problem per cell; _shutil_names([]) is {'shutil','_shutil'}"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named"
        status: pass
    human_judgment: false
  - id: D4
    description: "Look-alikes that name no shutil, _shutil or write_bytes stay allowed (frame.copy(), shutil_free, my_shutil, write_bytes_count, 'shutil-free', ...); shutil.which is superseded and now reported"
    requirement: WARN-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR6CopyAndWriteFunctions.test_copy_and_move_not_read_from_shutil_are_not_a_problem"
        status: pass
    human_judgment: false
  - id: D5
    description: "IPyAutocall is an R12 name reported in three forms on a real notebook copy; exit -1, quit -1 and a type(exit) subclass's one-line call cell give no R0 problem while its __call__ body is read"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported"
        status: pass
    human_judgment: false
  - id: D6
    description: "The Django-backed gate reads each cell as written (pin test, no code change)"
    requirement: WARN-02
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_pre_executed_notebooks.py#TestR0AllowList.test_the_django_backed_gate_reads_each_cell_as_written"
        status: pass
    human_judgment: false
  - id: D7
    description: "Wording of the function and class docstrings and comments matches the post-fix behaviour (module docstring deliberately untouched; 40-16 rewrites it after developer approval)"
    verification: []
    human_judgment: true
    rationale: "Docstring accuracy is prose; the replay checks only that the named phrases are present, and the developer approves the module-level wording in 40-16"

duration: 8min
completed: 2026-10-11
status: complete
---

# Phase 40 Plan 15: R6 shutil/write_bytes everywhere, IPyAutocall, autocall and gate pins Summary

**R6 now reports `shutil`, tempfile's `_shutil` and `write_bytes` at every identifier and string part, so all six fifth-review shutil hand-off forms and the class-pattern write_bytes form are caught on a copy of a real notebook; `IPyAutocall` joins the R12 names, and two pin tests back the autocall and raw-source-gate wording that 40-16 will put in the docstring.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-10-11T03:16:01Z
- **Completed:** 2026-10-11T03:24Z
- **Tasks:** 2
- **Files modified:** 1 (`solsys_code/tests/test_pre_executed_notebooks.py`)

## Accomplishments

- Closed the code half of verification gap 1 (CR-01, CR-02 of the fifth review, D-G11): shutil handed to a parameter, an immediately called lambda, `map(lambda ...)`, a keyword-only parameter, a `match` capture or a class pattern is reported by its name; `write_bytes` is reported as a class-pattern attribute, a parameter or a keyword.
- Closed the code half of WR-02 (D-G13): `'IPyAutocall'` is the only addition to `DYNAMIC_CODE_NAMES` (and so to `R12_ALL_NAMES` and R0's unparseable-cell search).
- Pinned the facts 40-16's wording relies on: `exit -1` / `quit -1` and a `type(exit)` subclass's one-line call cell give no R0 problem, the subclass's `__call__` body is read (a `shutil.copy` there is reported), and the Django-backed gate decides from each cell as written (WR-01, D-G12; `is_django_backed` and the gating are unchanged).
- Rewrote the function docstrings (`_byte_copy_problems`, `_shutil_names`, `isolation_problems`) and the TestR6CopyAndWriteFunctions and TestR0AllowList class docstrings for the post-fix behaviour. The module docstring is untouched (40-16).

## Task Commits

1. **Task 1 (tracer): R6 reports shutil, tempfile's _shutil and write_bytes at every identifier and string part** - `bc99b0e` (test) - closes CR-01, CR-02.
2. **Task 2: R12 lists IPyAutocall; autocall and raw-source-gate pins** - `1c830b1` (test) - closes WR-02 (code), WR-01 (wording and pin).

Both commits went through the full pre-commit hook (ruff, ruff-format, notebook checks, parallel django-test, about 80 s each) and hold only `solsys_code/tests/test_pre_executed_notebooks.py`. Nothing was pushed.

## TDD evidence

- **RED, Task 1** (`$HOME/tmp/phase40-15-t1-red.log`, 7 tests, `FAILED (failures=21)`): FAIL lines for `test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook` (5 CR-01 subtests plus the parameter-default control, which needs the new shutil report), `test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook` (class pattern, parameter, keyword argument), and `test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named` (every case plus the start-set check). Each failed on the planned assertion (`assertTrue`/`assertEqual` on the reported problems); the `control: method` write_bytes subtest and the look-alike test passed already, as planned. Semantic assessment: targets executed, failed on the planned assertion for the planned reason; no load, import or fixture fault.
- **GREEN, Task 1:** `Ran 89 tests ... OK` (whole guard module, 86 + 3).
- **RED, Task 2** (`$HOME/tmp/phase40-15-t2-red.log`, 6 tests, `FAILED (failures=3)`): `test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported` with subtests `[import]`, `[subclass]`, `[string]`. The `exit -1`, `quit -1`, type(exit)-subclass subtests and the gate pin test passed already (they pin existing behaviour, as planned).
- **GREEN, Task 2:** `Ran 96 tests ... OK` (guard + projector demo); final run `Ran 126 tests ... OK` (guard + helper + projector).
- **Replays and checks (all printed their OK lines):** Task 1 independent replay ("OK: independent replay -- six CR-01 forms report shutil and three CR-02 forms report write_bytes once each on a real notebook; ..."), Task 1 no-weakening ("OK: no weakening -- 89 guard tests ..."), Task 1 commit/ruff ("OK: ruff clean; the R6 commit holds exactly the guard module (bc99b0e5b097cfa7aa8a91560ea20e7168c87640)"), Task 2 independent replay ("OK: independent replay -- IPyAutocall is an R12 name ..."), Task 2 no-weakening ("OK: no weakening -- 91 guard tests (exactly five added, none removed or renamed) ..."), final check ("OK: ruff clean; the R12 commit holds only the guard module (1c830b13f38310a7b8a156caed5f551bad1acb06); exactly two guard commits after 1dbdb6d; guard, helper and projector tests green").

## Boundary lines (what R6 still does not report; Task 1 replay output, for 40-16's docstring, README and override reason)

```
boundary (inspect.getmodule): not reported
boundary (a function's __module__): not reported
boundary (a search over sys.modules): not reported
boundary (a search over a module's vars()): not reported
boundary (a write_bytes method found by a search): not reported
boundary (Python 3.14's Path.copy): not reported
boundary (code three strings deep): not reported
```

The three old boundary forms are now reported because each names shutil or _shutil: a function's return value (`def get(): return shutil` then `get().copy(a, b)`), a call's side effect (`setattr(m, 'sh', shutil)`), and `tempfile._shutil.copy(a, b)`.

In words: not reported are `copy` or `move` read from anything that mentions no shutil name (another object's method, Python 3.14's `Path.copy`/`Path.move`, unreachable on the supported 3.10-3.12); a shutil module, shutil function or `write_bytes` method obtained at runtime without writing its name (`inspect.getmodule`, a function's `__module__`, a search over `sys.modules` or a module's `vars()`); and code nested deeper than `STRING_CODE_DEPTH` (2) strings, matched only by its '.'/':' parts. Already inside the approved residual: `getattr(zipfile, 'sh' + 'util')` (RES 1), `open(..., 'wb')`, `write_text` with latin-1, Django's `file_move_safe` (RES 2).

## Forms now reported (each literal form, with its test; for 40-16's evidence table)

All in `solsys_code/tests/test_pre_executed_notebooks.py`, class `TestR6CopyAndWriteFunctions` unless noted.

| Form | Test |
|------|------|
| `def put(sh):` loop `sh.copy(...)` then `put(shutil)` (parameter given shutil) | `test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook` |
| `(lambda sh: [sh.copy(...) ...])(shutil)` (lambda called at once) | same |
| `list(map(lambda sh: [sh.move(...) ...], [shutil]))` | same |
| `def put(*, sh):` ... `put(sh=shutil)` (keyword-only parameter) | same |
| `match shutil:` / `case sh:` ... `sh.copy(...)` (match capture) | same |
| `match shutil:` / `case object(copy=c):` ... `c(...)` (class pattern) | same |
| `def put(sh=shutil):` ... `sh.move(...)` (parameter default, control; reports move and shutil) | same |
| `match repo_root_path / 'src' / f.name:` / `case object(write_bytes=w):` / `w(f.read_bytes())` | `test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook` |
| `def put(p, write_bytes=None):` | same |
| `put(p, write_bytes=data)` | same |
| `(...).write_bytes(f.read_bytes())` (method, control; exactly one problem per cell) | same |
| `import shutil`; `from shutil import which`; `def f(shutil=None)`; `f(shutil=1)`; `case object(shutil=s)` | `test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named` |
| `import tempfile` then `tempfile._shutil.copy(a, b)` (reported as `references _shutil` and `calls tempfile._shutil.copy()`) | same |
| `x = 'shutil'`; `x = 'tempfile._shutil'` (string parts) | same |
| `import shutil` and `x = shutil` in two cells: exactly two `references shutil` problems, one per cell | same |
| `from IPython.core.autocall import IPyAutocall`; a subclass of `autocall.IPyAutocall`; `name = 'IPython.core.autocall.IPyAutocall'` (only R12 problems naming IPyAutocall) | `TestR0AllowList.test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported` |
| (allowed, pinned) `exit -1`, `quit -1`, a `type(exit)` subclass `bye` and a `bye -1` cell: no problem; with `shutil.copy(a, b)` in its `__call__`, R6 reports the copy and no R0 marker appears | same |
| (pinned) a `%%capture` cell with `x = 1; import django as dj` / `dj.setup()`: not Django-backed, no alias or helper-call problem; same code without the magic: Django-backed and both problems; the magic cell inserted into a real-notebook copy: alias problem | `TestR0AllowList.test_the_django_backed_gate_reads_each_cell_as_written` |

**Superseded on purpose:** `import shutil` then `path = shutil.which('ls')` was allowed by 40-13 and is now reported (it names shutil; developer decision D-G11). Three new look-alikes (`print('shutil-free')`, `x = my_shutil`, `x = write_bytes_count`) keep the look-alike list whole; the other seven look-alikes still give `[]`.

## Facts 40-16's wording relies on

- The kernel's only autocall objects are `exit` and `quit` (autocall mode 0); `exit -1` runs as `exit(-1)` and only ends the kernel. `exit` and `quit` are builtins and not line magics, so R0's automagic rule exempts them.
- No module in the stdlib or the venv's site-packages imports shutil under a name other than `shutil` or `_shutil` (`grep -rnE "shutil as [A-Za-z_]+"`, checked at planning).
- Python NFKC-normalizes identifiers when it parses, so a full-width `ｓｈｕｔｉｌ` is the identifier `shutil` and is reported; a full-width string is a different key at runtime and reaches nothing.
- All eight real notebooks give `isolation_problems() == []` and are Django-backed on raw source after each task; none names shutil, _shutil, write_bytes or IPyAutocall.

## Files Created/Modified

- `solsys_code/tests/test_pre_executed_notebooks.py` - new constant `SHUTIL_MODULE_NAMES`; `'IPyAutocall'` in `DYNAMIC_CODE_NAMES`; `_shutil_names` start set; `_byte_copy_problems` string-part and identifier passes; five new tests and one changed look-alike list; reworded function and class docstrings and two comments.

## Decisions Made

- D-G11 followed as planned: shutil names get their own constant rather than joining DISTINCTIVE_COPY_NAMES (which stays exactly as 40-13 left it).
- D-G12 followed: no code change to `is_django_backed`, `_django_imports` or the gating; confirmed identical by an ast comparison with docstrings removed.
- D-G13 followed: only `IPyAutocall` added; the documented `exit`/`quit` exception is pinned, not closed.

## Deviations from Plan

None - plan executed exactly as written.

Minor execution notes, not deviations: ruff-format re-wrapped several of the new test lines, one comment line was split by hand for the 120-column limit, and the `_byte_copy_problems` docstring was re-flowed once so the phrase "every identifier" sits on one line (the plan's own replay checks for it as a literal).

## Issues Encountered

None.

## Known Stubs

None.

## Threat Flags

None - this plan changes one test module and adds no endpoint, auth path, file access or schema.

## Scope statement

No notebook, `docs/notebooks/README.md`, `docs/notebooks/fomo_nb_utils.py`, helper test module, module docstring, `pyproject.toml`, `CLAUDE.md`, workflow file, `40-VERIFICATION.md`, `40-REVIEW.md`, `40-REVIEW-DISPOSITION.md` or ledger changed since 1dbdb6d; `git status --short src/` is empty. 40-16 rewrites the module docstring and README and records the ledger and the developer-approved override. The D-G14 items (the fifth review's IN-01 and IN-02, the fourth review's IN-01, round-2 IN-03, WR-03, IN-04, the IN-07 remainder, round-1 IN-03's runbook text, every attribution or template file) were not touched.

## Next Phase Readiness

Ready for 40-16: the guard now checks everything the revised docstring, README and override reason will claim, and the boundary list above is the source for what they must name as not reported.

## Self-Check: PASSED

- `solsys_code/tests/test_pre_executed_notebooks.py` exists and contains `SHUTIL_MODULE_NAMES`.
- Commits `bc99b0e` and `1c830b1` are ancestors of HEAD; `git rev-list --count f5f5083..HEAD` is 2.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-11*
