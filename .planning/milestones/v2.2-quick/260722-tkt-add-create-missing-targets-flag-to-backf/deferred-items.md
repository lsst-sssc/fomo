# Deferred Items

Out-of-scope findings discovered during execution of quick task 260722-tkt, not fixed
per the executor's scope boundary rule (only auto-fix issues directly caused by the
current task's changes).

## ruff findings in unrelated notebook files

`ruff check .` (repo-wide) reports 5 pre-existing findings, all in
`docs/notebooks/pre_executed/*.ipynb` files not touched by this task:

- `load_telescope_runs_demo.ipynb` — E401 (multiple imports on one line), I001 (unsorted
  import block)
- `sync_gemini_observation_calendar_demo.ipynb` — D103 (missing docstring in public
  function `make_gem_record`)
- `sync_lco_observation_calendar_demo.ipynb` — I001 (unsorted import block), E501 (line
  too long)

Confirmed pre-existing: `git status --short` shows these files unmodified by this task,
and `git log -1` on each traces back to commit `0dcd4b0` (Phase 23), unrelated to
`backfill_lco_observation_records.py`.

The two files this task actually modified
(`solsys_code/management/commands/backfill_lco_observation_records.py` and
`solsys_code/tests/test_backfill_lco_observation_records.py`) are clean:
`ruff check` and `ruff format --check` both pass on them individually.
