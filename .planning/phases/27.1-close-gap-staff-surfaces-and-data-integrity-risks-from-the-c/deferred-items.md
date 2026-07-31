# Phase 27.1 — Deferred Items

Out-of-scope discoveries found during execution that were not fixed, per the executor's
scope-boundary rule (only auto-fix issues directly caused by the current task's changes).

## Pre-existing repo-wide ruff/format drift (found during 27.1-01 Task 3)

`ruff check .` and `ruff format --check .`, run against the full repository as part of
Task 3's verification, surface failures unrelated to this plan's own changes -- the same
class of pre-existing drift previously logged in
`.planning/phases/26-canonical-record-spike/deferred-items.md`:

- `ruff check .`: `D103` (missing docstring in public function) in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` (cell 6).
- `ruff format --check .`: would reformat
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `src/fomo/settings.py`.

Confirmed pre-existing: `git diff 2872537392a31d392241162578ff724d576f0e65 -- <each path
above>` is empty for every one of these files -- none was touched by any 27.1-01 task.
This plan's own two files (`solsys_code/tests/test_calendar_template.py`,
`src/templates/tom_calendar/partials/event_form.html`) are individually clean:
`ruff check solsys_code/tests/test_calendar_template.py` and `ruff format --check
solsys_code/tests/test_calendar_template.py` both pass (the `.html` file is not linted by
ruff's Python-only default include patterns when discovered via directory walk, which is
why `ruff check .`/`ruff format --check .` never flag it, and explicitly naming it on the
command line falsely triggers Python-syntax parsing -- confirmed not a real finding).

Not fixed here -- out of scope for a plan whose only committed deliverables are the event
modal template and its test module. Whoever next runs a repo-wide `ruff check . --fix &&
ruff format .` cleanup pass should pick these up.
