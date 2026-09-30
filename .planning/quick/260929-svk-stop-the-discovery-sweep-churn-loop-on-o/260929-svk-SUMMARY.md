---
phase: 260929-svk
plan: 01
subsystem: discovery / projector-sweep interplay
tags: [backfill_lco_observations, observed-site, F1, churn, notebook, runbook]
requires: []
provides:
  - "backfill_lco_observations carries the sweep's observed_site/observed_telescope/observed_enclosure keys forward"
affects: [unattended runner tick time, calendar event titles]
key-files:
  modified:
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
decisions:
  - "Carry-forward lives only inside _changed_record_fields(), the one comparison both dry-run and real-run branches call (T-ik7-02)"
  - "Key presence is tested by membership, never truthiness, so a stored None enclosure survives"
  - "Notebook uses a fresh migrated scratch DB (not a copy of the live DB)"
status: complete
commits: 3
plan_head_before: 3fa7469357151a9a4d064905855c15f37d78bb3e
plan_head_after: 213b8baae2d9aa4ffa15e837e8341a6c33012414
actuals:
  tokens: 60000
  tasks: 3
  commits: 3
---

# Phase 260929-svk Plan 01: Stop the discovery/sweep churn loop (F1) Summary

Discovery (`backfill_lco_observations`) now carries the three observed-site keys the projector sweep stores forward via `_preserve_observed_site_keys()`, applied once inside `_changed_record_fields()`; unchanged portal data no longer saves the row, while a really moved window still updates `start`/`end`.

## Commits

| Task | Commit | Message |
| ---- | ------ | ------- |
| 1 (RED) | `6b3a731` | test(260929-svk): pin observed-site keys surviving discovery (F1) |
| 1 (GREEN) | `7f31677` | fix(260929-svk): discovery carries the sweep's observed-site keys forward (F1) |
| 2 | `213b8ba` | docs(260929-svk): demo observed-site carry-forward on a scratch database; runbook note |
| 3 | (no commit) | gates made no formatting change |

## Implementation

- Two production edits, each importable on its own (import smoke check run after each): Edit 1 added the `OBSERVED_SITE_PARAMETER_KEYS` import and the unused helper; Edit 2 added the single call at the top of `_changed_record_fields()` plus the docstring update. No call sites, `_build_parameters()`, `get_or_create` defaults or projector-side code were touched.
- RED confirmed for the F1 reason: with only the helper present, the four behavioral tests (real unchanged, real moved window, dry-run unchanged, None enclosure) failed/errored and the helper tests passed; after Edit 2 all pass. The AST gate confirms `_changed_record_fields` calls the helper.

## Test counts

- `solsys_code.tests.test_backfill_lco_observations`: 49 before -> 59 after (5 behavioral + 5 helper tests added), all pass.
- Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`: `Ran 1802 tests in 1429.324s`, `OK (skipped=1)`, exit 0.

## Notebook (`backfill_lco_observations_demo.ipynb`, executed top to bottom, 15 code cells, no error output)

- Setup cell output: `Resolved database: '/tmp/fomo-notebook-db-zxivo06u/fomo_db.sqlite3' (fresh scratch database, migrated from empty)`; last cell removes the scratch directory (confirmed gone).
- Dry run over unchanged data: `... would create: 0, would update: 0, unchanged: 2, ...`
- Real run over unchanged data: `... created: 0, updated: 0, unchanged: 2, ...` and both `modified` timestamps unchanged; `PASS: unchanged portal data leaves both records unchanged and every observed-site key in place`
- Moved window on 900101: `... created: 0, updated: 1, unchanged: 1, ...`; `parameters` shows new `start`/`end` (`2026-07-02T00:00:00`/`2026-07-03T00:00:00`) with `observed_site='elp'`, `observed_telescope='1m0a'`, `observed_enclosure='doma'` intact; `PASS: the moved window was refreshed from the portal and the observed-site keys survived the update`
- Read-only query of `src/fomo_db.sqlite3` (`mode=ro`) after execution: 0 demo observation records, 0 `BACKFILL-DEMO%` watched proposals, 0 `failed: KeyError%` summaries.

## Gates

- `pre-commit run ruff --all-files`: Passed. `pre-commit run ruff-format --all-files`: Passed (no file rewritten).
- Runbook bullet check and Sphinx build: Passed.

## Live ticks after the fix (read-only, `/var/log/fomo/unattended.log`)

The fix commit `7f31677` landed at 21:02:54 local (04:02:54Z). The 21:00 tick had already imported the pre-fix code.

```
START 2026-09-29T21:15:07 (first post-fix tick)
step project_sweep: ok | failed: 0 | LCO: created: 0, updated: 134, unchanged: 60, unprojectable: 0, site_lookups: 134, site_lookup_failed: 1
step discovery: ok | swept: 1, failed: 0
END   ... duration=483s

START 2026-09-29T21:30:08 (second post-fix tick)
step project_sweep: ok | failed: 0 | LCO: created: 0, updated: 0, unchanged: 194, unprojectable: 0, site_lookups: 0, site_lookup_failed: 1
step discovery: ok | swept: 1, failed: 0
END   ... duration=233s
```

This is the expected pattern: the first tick's sweep restored keys that the pre-fix discovery had erased; the second tick shows `updated: 0, site_lookups: 0` and a much shorter duration (233 s; the full-suite run was competing for CPU during part of it). The F1 checkbox in `.planning/v2.4-INTENT-REVIEW.md` was NOT ticked: the two-tick confirmation is the operator's to make (a third tick would strengthen it).

## Deviations from Plan

1. **[Process] Commit `7f31677` lacks the Co-Authored-By/Claude-Session trailers.** The first attempt at this commit (with trailers) was rejected because the Sphinx pre-commit hook failed transiently (it passed on immediate retry and standalone); the retry command omitted the trailers, and the commit succeeded. I did not amend (the instructions forbid `--amend`). Commits `6b3a731` and `213b8ba` carry the trailers. The Sphinx hook also failed once transiently on the first attempt of `213b8ba`; the retry passed.
2. **[Environment] Full-suite run took ~24 minutes** (1429 s) rather than ~8, because it shared the host with the cron ticks; the first foreground attempt was moved to the background at the 600 s limit and produced no summary line, so it was re-run to a log file with an explicit exit code.

No other deviations. Plan symbol names, line locations and fixture styles matched the code.

## Known Stubs

None.

## Threat Flags

None.

## Self-Check: PASSED

- Files exist: the four `files_modified` paths (verified via git show --stat on each commit).
- Commits exist: `6b3a731`, `7f31677`, `213b8ba` in `git log`.
- 3 commits measured from `3fa7469..HEAD`; unrelated dirty/untracked files remain uncommitted.
