# Deferred Items — quick-260619-jpr

Pre-existing issues discovered during Task 3 (quality-gate check) that are out
of scope for this quick task (not caused by the SOAR site-mapping fix).

## ruff check findings in the demo notebook (pre-existing, not caused by this task)

`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`:

- **cell 6** (`I001`): unsorted/unformatted import block (`from django.forms
  import model_to_dict` / `import pprint`).
- **cell 12** (`E501`): one line 139 chars, over the 120-char limit (an
  f-string building a CalendarEvent summary line with
  `event.telescope`/`event.instrument`/`event.proposal`).

Confirmed pre-existing by running `ruff check` against the notebook as of
commit `fa17e2c` (the commit immediately before this quick task's first
commit) — both findings are present there too, in cells unrelated to the two
SOAR fixture cells (20, 22) this task modified.

Not fixed here per the executor's scope-boundary rule: only auto-fix issues
directly caused by the current task's changes. `ruff check .`/`ruff format
--check .` were already non-clean for this notebook before this quick task;
this task does not regress that, and does not claim "ruff check . stays
clean" project-wide — only that the two Python files this task modified
(`sync_lco_observation_calendar.py`,
`test_sync_lco_observation_calendar.py`) pass both gates cleanly.
