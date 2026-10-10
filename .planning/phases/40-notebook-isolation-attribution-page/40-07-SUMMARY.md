---
phase: 40-notebook-isolation-attribution-page
plan: 07
subsystem: docs
tags: [notebooks, paired-docs, nbconvert, scratch-database, review-ledger, gap-closure]
gap_closure: true
gap_ids: [IN-02, IN-05]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-05's fail-closed helper (settings checked before django.setup(), Django ready line without the checkout path) and 40-06's guard rule R12"
provides:
  - "All eight pre-executed notebooks re-executed top to bottom on the 40-05 helper and committed with output"
  - "load_telescope_runs_demo creates its eight schedule files inside scratch_db_dir, so teardown removes them (IN-05)"
  - "Every notebook's setup markdown describes the new order (scratch directory, settings check, django.setup() and migrate) and carries the same setup-check sentence as docs/notebooks/README.md"
  - "40-REVIEW-DISPOSITION.md records CR-01, WR-01, WR-02, IN-01, IN-02, IN-05, IN-06 as fixed and IN-07 as deferred in part"
affects: [phase 40 verification, Phase 41 triage of WR-03, IN-03, IN-04 and IN-07's alias tidy-up]

actuals:
  tokens: 13800
  tasks: 3
  commits: 4

plan_head_before: 484bce84726802fac77ac206aff10cdbe8362e2d
plan_head_after: 49a05307fa09abf570b6d543c9ec1a4076212a4c

tech-stack:
  added: []
  patterns:
    - "Notebook prose edited with nbformat from a scratch script (round trip of an untouched notebook is byte-identical), then ruff-format, then nbconvert from docs/notebooks/pre_executed/ with TMPDIR=/tmp"
    - "Schedule files written inside the notebook's scratch directory so the helper's teardown owns their lifetime"

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
    - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
    - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - .planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md

key-decisions:
  - "The spiceypy 'Using fallback library next to module' stderr line in campaign_lifecycle_demo is third-party log output, not helper output: listed here and left in place (developer decision recorded in the plan)"
  - "The ledger's IN-02 Source cell names the spiceypy line without its cell id, because the id d687bfab is hexadecimal and the plan's ledger check reads any 7-40 character hex word as a commit hash"

patterns-established:
  - "Ledger Source cells cite 7-character commit hashes that exist, come after 0ad6991 and touch the finding's file"

requirements-completed: [WARN-05, WARN-02, WARN-03]

coverage:
  - id: D1
    description: "IN-05: all eight tempfile.NamedTemporaryFile calls in load_telescope_runs_demo pass dir=scratch_db_dir; the committed schedule path lies inside the run's /tmp/fomo-notebook-db-* directory, which is gone after teardown"
    requirement: WARN-03
    verification:
      - kind: command
        ref: "ast check of load_telescope_runs_demo.ipynb (8 calls, dir=scratch_db_dir, printed path inside scratch directory, directory absent)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Paired docs for 40-05's helper change: all eight notebooks re-executed (counts 1..N, no error output, exact Django ready line, /tmp Resolved line with the mode description, teardown last with output, no /home/ in helper output)"
    requirement: WARN-05
    verification:
      - kind: command
        ref: "per-notebook check script, all eight, plus the all-eight check with the home-path inventory"
        status: pass
      - kind: unit
        ref: "solsys_code.tests.test_pre_executed_notebooks, test_notebook_helper, test_projector_demo_notebook (82 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Each setup markdown cell carries the setup-check sentence and no cell keeps the old check-after-setup wording; no code cell changed except load_telescope_runs_demo's seven schedule-file cells"
    requirement: WARN-05
    verification:
      - kind: command
        ref: "per-notebook check script (sentence present, old wording absent, code cells identical to 0ad6991 except the allowed seven)"
        status: pass
    human_judgment: true
    rationale: "Whether the setup prose reads clearly and matches the output beside it is the reading VERIFICATION human item 1 asks for (human-check at /gsd-verify-work)"
  - id: D4
    description: "Review ledger: seven findings fixed, IN-07 deferred in part, WR-03/IN-03/IN-04 untouched and open, open 3, total 11, exactly 17 lines changed, every Source cell cites real post-0ad6991 commits touching the finding's file"
    requirement: WARN-02
    verification:
      - kind: command
        ref: "ledger check script (OK line printed)"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-10-10
status: complete
---

# Phase 40 Plan 07: Paired docs for the fail-closed notebook helper Summary

**All eight pre-executed notebooks were re-executed on the 40-05 helper with setup prose describing the settings check before django.setup(), load_telescope_runs_demo now writes its eight schedule files inside the scratch directory, and the review ledger records the round (seven findings fixed, IN-07 deferred in part).**

## Performance

- **Duration:** about 20 min (start of the first edit to the last commit; about 05:09Z to 05:28Z)
- **Tasks:** 3 (a tracer, then two auto tasks)
- **Files modified:** 9 (eight notebooks and the ledger)
- **Notebook commits:** three through the full django-test hook (77 to 85 s each); ledger commit with `SKIP=django-test`

## Accomplishments

- IN-05 closed: every `tempfile.NamedTemporaryFile(...)` in `load_telescope_runs_demo` has `dir=scratch_db_dir`; the committed `Schedule file written to:` path lies inside `/tmp/fomo-notebook-db-_4ruy6g9/`, which no longer exists. Markdown `2bd45621` says the sample file lives in `scratch_db_dir` and goes with the teardown.
- IN-02 closed for helper output in all eight notebooks: the setup cell prints exactly `Django ready: settings module='src.fomo.settings'`, and neither the setup nor the teardown output contains a `/home/` path.
- Paired docs for 40-05: all eight notebooks re-executed with `TMPDIR=/tmp`; each has execution counts exactly 1..N, no error output, a `/tmp/fomo-notebook-db-*` Resolved line with its mode's description, and `teardown_scratch_django(scratch_db_dir)` as its last cell with `Removed scratch database directory:` output.
- Setup markdown rewritten in all eight to the new order and given the same setup-check sentence as `docs/notebooks/README.md`; no markdown cell keeps the old "then checks that Django really resolved" wording.
- Ledger updated: 17 lines changed (see below).

## Task Commits

1. **Task 1 (tracer): load_telescope_runs_demo end to end** - `29490c3` (docs) - IN-05, IN-02, WARN-05
2. **Task 2: five fresh notebooks** - `8beeb5c` (docs) - backfill_lco_observations, campaign_lifecycle, import_campaign_csv, sync_gemini_observation_calendar, telescope_runs
3. **Task 3a: two snapshot notebooks** - `304022a` (docs) - project_observation_calendar, reconcile_campaign_runs
4. **Task 3b: review ledger** - `49a0530` (docs, `SKIP=django-test`, only this commit)

**Plan metadata:** the SUMMARY and state updates are committed separately after this file.

## Per-notebook record

Resolved lines are those committed (random directory names differ per run; every directory was gone after teardown).

| Notebook | Mode | Resolved line | Markdown cells changed | Code cells whose output changed (besides setup and teardown lines) |
|----------|------|---------------|------------------------|-----------------------------------------------------------|
| load_telescope_runs_demo | fresh | `/tmp/fomo-notebook-db-_4ruy6g9/fomo_db.sqlite3` (fresh scratch database, migrated from empty) | `49ff72df`, `2bd45621` | none (the seven schedule-file cells changed in source; their output is unchanged except the schedule path) |
| backfill_lco_observations_demo | fresh | `/tmp/fomo-notebook-db-8be8yo3n/...` (fresh) | `86dd4f0a` | `6a1f0c22` (`last_run_at` timestamps), `e2c7f5a4` (one log line moved: stdout/stderr interleaving) |
| campaign_lifecycle_demo | fresh | `/tmp/fomo-notebook-db-2htehcmo/...` (fresh) | `7dfef438` | `729bd69e` (`confirmed_at` timestamp), `f13a1c02` (two log lines reordered) |
| import_campaign_csv_demo | fresh | `/tmp/fomo-notebook-db-9qs0ye9r/...` (fresh) | `43e2c1c3` | none |
| sync_gemini_observation_calendar_demo | fresh | `/tmp/fomo-notebook-db-lss3u2if/...` (fresh) | `c674d60c` | `d0e1f2a3`, `b4c5d6e7`, `d6e7f8a9` (time-relative window values only) |
| telescope_runs_demo | fresh | `/tmp/fomo-notebook-db-rzp7fvb7/...` (fresh) | `23326d63` | none |
| project_observation_calendar_demo | snapshot | `/tmp/fomo-notebook-db-h3zmslsg/...` (scratch copy of the developer database, taken read-only with SQLite's online backup and migrated to head) | `9607e5f3` | `b9831b5d` (random suffixes in demo ids), `055eff76` (stage-marker counts), `250b5d0b` (pending-record count) |
| reconcile_campaign_runs_demo | snapshot | `/tmp/fomo-notebook-db-y_j8yvtt/...` (scratch copy, as above) | `dfcc0343` | `717711a4`, `db8b9701`, `98fc8daa`, `ed8be386`, `80b3d470` (random demo ids, hook-line and stderr ordering only) |

Prose read against the regenerated output: no sentence was contradicted. Reconcile's `8b703ea8`, `e1edbaa7`, `ce1fd748` and `d2adacf8` say the cutover section is a clean no-op on real data (`candidates: 0`, no `CommandError`); the regenerated output still shows exactly that, so they were left alone. No project_observation_calendar markdown cell quotes a count.

## Snapshot headline counts for this regeneration

The developer database file was not touched (`src/fomo_db.sqlite3` modification time unchanged, no `git status` entry). Both notebooks read it only through the helper's `mode=ro` online backup.

**reconcile_campaign_runs_demo** (previous values: 40-01 SUMMARY)

| Metric | This regeneration | 40-01 |
|--------|-------------------|-------|
| CalendarEvent total (before cutover) | 511 | 511 |
| facility-url events | 459 | 459 |
| ALLOC: events | 30 | 30 |
| bare RUN: containers | 22 | 22 |
| blank-url / date-bearing RUN: events | 0 / 0 | 0 / 0 |
| CampaignRun rows | 41 | 41 |
| cutover `candidates` (dry and real) | 0, no `CommandError` | 0 |
| first sweep | `runs: 41, created: 0, updated: 0, unchanged: 52, skipped: 4, retired: 5` | same |

**project_observation_calendar_demo** (previous values: 40-02 SUMMARY)

| Metric | This regeneration | 40-02 |
|--------|-------------------|-------|
| facility-url events | 459 | 459 |
| `RUN:` / `GEM:` / blank-url | 22 / 0 / 0 | 22 / 0 / 0 |
| dry run and both sweeps | `unchanged: 459`, created and updated 0 | same |
| stage markers | `[Q] 96, [S] 52, [O] 176, [X] 59, [C] 75, [F] 1, [?] 0` | `[Q] 82, [S] 68, [O] 175, [X] 58, [C] 75, [F] 1, [?] 0` |
| pending `KEY2026B-004` records in the copy | 148 | 150 |

The stage markers and pending count move with the snapshot (accepted, RESEARCH A6); `test_projector_demo_notebook` passes and cell ids `7022f987`, `05528b38`, `556d2a9f`, `250b5d0b` are intact.

## Home-path inventory (committed output lines still containing `/home/`)

```
HOME-PATH (not helper output): campaign_lifecycle_demo.ipynb  d687bfab  Using fallback library next to module: /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/spiceypy/utils/libcspice.so
```

This is the only line. It is a third-party spiceypy INFO log on stderr, emitted when the ephemeris module is imported; it is not printed by the helper and no setup or teardown cell contains it. It is left as is because silencing it would mean a logging change in the demo or settings, outside this round. `grep -l /home/ docs/notebooks/pre_executed/*.ipynb` lists only `campaign_lifecycle_demo.ipynb`.

## campaign_lifecycle_demo High-band marker and Undo cells

Confirmed unchanged: the scripted diff against `0ad6991` shows output changes only in `729bd69e` (a timestamp) and `f13a1c02` (two log lines reordered); the 40-04 marker and Undo cells show no output difference and the `Undo redirected to ...` line appears the same number of times (2) as before.

## Review ledger

`40-REVIEW-DISPOSITION.md`: CR-01, WR-01, WR-02, IN-01, IN-02, IN-05 and IN-06 are `fixed`; IN-07 is `deferred` and its Source cell says it is partly fixed in 40-05 (`7cfd78e`) with the `_DISMISSED_TABLE_PREFIX` and `_CONFIRMED_TABLE_PREFIX` alias tidy-up in `campaign_views.py` remaining; `open: 3`, `total: 11`; WR-03, IN-03 and IN-04 untouched. Commits cited: `222a25f`, `7cfd78e`, `819321e` (40-05), `50ffb59`, `26940ce` (40-06), `29490c3`, `8beeb5c`, `304022a` (this plan).

Ledger check output: `OK: 7 findings fixed, IN-07 deferred in part, WR-03/IN-03/IN-04 open, open 11 -> 3, every Source cell cites real commits touching its file, exactly 17 lines changed`.

WR-03, IN-03, IN-04 and IN-07's alias tidy-up remain for Phase 41 triage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cell id in the IN-02 ledger cell read as a commit hash**
- **Found during:** Task 3, ledger check
- **Issue:** the first draft of the IN-02 Source cell named the spiceypy cell by id (`d687bfab`). That id is eight hexadecimal characters, so the plan's own ledger check treated it as a commit hash and failed (`d687bfab is not a commit`).
- **Fix:** named the cell as "its ephemeris import cell" instead; the id stays in this SUMMARY.
- **Files modified:** `.planning/phases/40-notebook-isolation-attribution-page/40-REVIEW-DISPOSITION.md`
- **Committed in:** `49a0530` (the ledger was not yet committed when this was found)

**2. [Process] Ledger Source cells use "and" where the plan text used " / "**
- The IN-07 cell says "DISMISSED_TABLE_PREFIX and CONFIRMED_TABLE_PREFIX" rather than a slash, to keep the markdown table readable; the check still finds `_DISMISSED_TABLE_PREFIX` and `campaign_views.py` in the cell.

---

**Total deviations:** 1 auto-fixed (Rule 1), 1 wording difference. **Impact:** none on scope or acceptance criteria.

## Issues Encountered

- A stale `/tmp/fomo-notebook-db-i_k4rosi` (dated 2026-10-02, noted in 40-05) and 16 stale `/tmp/tmp*.txt` files from earlier regenerations are still on the machine. They predate this plan; the eight directories created by this plan's runs were all removed by their teardowns, and the schedule files of this run live inside the removed directory. Left alone.
- None of the plan's prohibitions was approached: no `--allow-errors`, no code cell changed outside the seven schedule cells, `FOMO_DATABASE_PATH` never set in the shell, `src/fomo_db.sqlite3` never opened by hand.

## Verification

- `python manage.py test --noinput solsys_code.tests.test_pre_executed_notebooks solsys_code.tests.test_notebook_helper solsys_code.tests.test_projector_demo_notebook`: 82 tests OK after Tasks 1 and 3; 77 OK (first two modules) after Task 2.
- Full suite `python manage.py test solsys_code --exclude-tag=ephemeris_segfault --parallel auto --noinput`: `Ran 2350 tests ... OK`.
- Per-notebook check scripts and the all-eight inventory printed their OK lines; the ledger check printed its OK line.
- `git rev-list --count 484bce8..HEAD` was 4 before this SUMMARY (3 notebook commits, 1 ledger commit). Nothing pushed.

## Paired docs

This plan is the paired-docs work for 40-05's change to `docs/notebooks/fomo_nb_utils.py`: all eight notebooks under `docs/notebooks/pre_executed/` were re-executed and committed with output. No `docs/runbooks/` page describes the helper, so no runbook changed. The runbook attribution-pass wording (IN-03) is out of scope this round.

## Known Stubs

None.

## Threat Flags

None. No endpoint, auth path or trust-boundary change; the only new file-access pattern is schedule files inside the existing scratch directory (T-40-26, mitigated and checked).

## Next Phase Readiness

Plan 7 of 7 is complete. The phase is ready for review and verification; WR-03, IN-03, IN-04 and IN-07's alias tidy-up are carried to Phase 41 triage.

## Self-Check: PASSED

- All eight notebooks and `40-REVIEW-DISPOSITION.md` exist; commits `29490c3`, `8beeb5c`, `304022a`, `49a0530` are ancestors of HEAD.
- `40-VERIFICATION.md`, `40-REVIEW.md` and every 40-01 to 40-04 PLAN and SUMMARY file are unchanged since `0ad6991`; none of the developer's uncommitted files was staged.
