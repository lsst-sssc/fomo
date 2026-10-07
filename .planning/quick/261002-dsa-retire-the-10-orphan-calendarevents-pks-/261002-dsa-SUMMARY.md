---
phase: 261002-dsa
plan: 01
subsystem: calendar
tags: [one-off-repair, calendar-events, allocation-nights, sqlite]
requires:
  - phase: 35
    provides: 'ALLOC: allocation-night projector (the nights that supersede the orphans)'
provides:
  - retire_orphan_events.py: reviewable, dry-run-default script that retires the ten orphan CalendarEvents (pks 44-52, 334)
affects: [v2.4 walkthrough Q1-Q7]
key-files:
  created:
    - .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py
  modified:
    - .planning/v2.4-INTENT-REVIEW.md (working tree only, uncommitted by design)
key-decisions:
  - "Description-line containment accepts an ALLOC: line equal to the orphan's line, or that line plus the loader's ' [<proposal id>]' tag"
requirements-completed: [ALLOC-05]
status: complete
duration: ~20min
completed: 2026-10-02
plan_head_before: 08ca7f674835d87b4cd50a4c869de7e59fc70c37
plan_head_after: c0be3a4be5e9d360f9879280f142fc8199548025
actuals:
  tokens: 3000
  tasks: 2
  commits: 1
---

# Phase 261002-dsa Plan 01: Retire the 10 orphan CalendarEvents Summary

**A dry-run-default repair script that, after a strict pre-flight, deletes pks 44-52 and 334 (plus pk 334's one cascaded run-68 companion row) in one transaction; proven on scratch copies only, the live run belongs to the operator.**

## Outcome

- Census re-check (read-only, `mode=ro`) before writing anything: passed. All ten rows have blank urls, 44-52 are companion-free and match their `ALLOC:76/77/78` night to the second, 334 has exactly one unconfirmed run-68 companion row, and no todos or dismissals hang off any of them.
- Commit: `c0be3a4` `chore(261002-dsa): add one-off script retiring the 10 orphan calendar events superseded by the Didymos ALLOC nights` (script only; committed with `DRY_RUN = True`).
- Lint: `.planning/` is excluded from the pre-commit ruff hooks, so the venv's ruff 0.2.1 was run directly on the script (`ruff check` and `ruff format --check`): clean.
- The live `src/fomo_db.sqlite3` was never written. After all runs it still holds all ten orphans (read through a `mode=ro` URI). Every run used a `.backup` copy via `FOMO_DATABASE_PATH`, and each run's own `database:` line proved the copy was in use. Copies were deleted.

## Dry run on a scratch copy (verbatim; Task 1)

```
45 objects imported automatically (use -v 2 for details).

database: <scratchpad>/261002-dsa/dry/fomo_db.sqlite3
mode: DRY RUN (nothing will be written)
pk | title | UTC span | superseded by
44 | NTT EFOSC2 | 2026-07-09T22:06:36+00:00 -> 2026-07-10T11:29:48+00:00 | ALLOC:76:2026-07-09 (event 409)
45 | NTT EFOSC2 | 2026-07-10T22:07:04+00:00 -> 2026-07-11T11:29:36+00:00 | ALLOC:76:2026-07-10 (event 410)
46 | NTT EFOSC2 | 2026-07-11T22:07:33+00:00 -> 2026-07-12T11:29:22+00:00 | ALLOC:76:2026-07-11 (event 411)
47 | NTT EFOSC2 | 2026-07-12T22:08:02+00:00 -> 2026-07-13T11:29:07+00:00 | ALLOC:76:2026-07-12 (event 412)
48 | Magellan-Baade IMACS | 2026-07-17T22:10:56+00:00 -> 2026-07-18T11:26:51+00:00 | ALLOC:77:2026-07-17 (event 413)
49 | Magellan-Baade IMACS | 2026-07-18T22:11:27+00:00 -> 2026-07-19T11:26:28+00:00 | ALLOC:77:2026-07-18 (event 414)
50 | Magellan-Clay Lightspeed | 2026-07-18T22:11:28+00:00 -> 2026-07-19T06:26:00+00:00 | ALLOC:78:2026-07-18 (event 415)
51 | Magellan-Clay Lightspeed | 2026-07-19T22:12:00+00:00 -> 2026-07-20T06:26:00+00:00 | ALLOC:78:2026-07-19 (event 416)
52 | Magellan-Clay Lightspeed | 2026-07-20T22:12:31+00:00 -> 2026-07-21T06:26:00+00:00 | ALLOC:78:2026-07-20 (event 417)
334 | tmp | 2025-07-04T22:00:00+00:00 -> 2025-07-05T06:00:00+00:00 | stray (companion row attributed to run 68 cascades)
asserted: 10 (9 superseded by ALLOC nights, 1 stray)
companion rows that will cascade: 1 (pk 334 -> run 68)
blank-url events in this database: 10
DRY RUN: nothing deleted. Run a copy with DRY_RUN = False to delete.
```

## Real run on a scratch copy (verbatim; Task 2)

```
45 objects imported automatically (use -v 2 for details).

database: <scratchpad>/261002-dsa/real/fomo_db.sqlite3
mode: DELETE
pk | title | UTC span | superseded by
44 | NTT EFOSC2 | 2026-07-09T22:06:36+00:00 -> 2026-07-10T11:29:48+00:00 | ALLOC:76:2026-07-09 (event 409)
45 | NTT EFOSC2 | 2026-07-10T22:07:04+00:00 -> 2026-07-11T11:29:36+00:00 | ALLOC:76:2026-07-10 (event 410)
46 | NTT EFOSC2 | 2026-07-11T22:07:33+00:00 -> 2026-07-12T11:29:22+00:00 | ALLOC:76:2026-07-11 (event 411)
47 | NTT EFOSC2 | 2026-07-12T22:08:02+00:00 -> 2026-07-13T11:29:07+00:00 | ALLOC:76:2026-07-12 (event 412)
48 | Magellan-Baade IMACS | 2026-07-17T22:10:56+00:00 -> 2026-07-18T11:26:51+00:00 | ALLOC:77:2026-07-17 (event 413)
49 | Magellan-Baade IMACS | 2026-07-18T22:11:27+00:00 -> 2026-07-19T11:26:28+00:00 | ALLOC:77:2026-07-18 (event 414)
50 | Magellan-Clay Lightspeed | 2026-07-18T22:11:28+00:00 -> 2026-07-19T06:26:00+00:00 | ALLOC:78:2026-07-18 (event 415)
51 | Magellan-Clay Lightspeed | 2026-07-19T22:12:00+00:00 -> 2026-07-20T06:26:00+00:00 | ALLOC:78:2026-07-19 (event 416)
52 | Magellan-Clay Lightspeed | 2026-07-20T22:12:31+00:00 -> 2026-07-21T06:26:00+00:00 | ALLOC:78:2026-07-20 (event 417)
334 | tmp | 2025-07-04T22:00:00+00:00 -> 2025-07-05T06:00:00+00:00 | stray (companion row attributed to run 68 cascades)
asserted: 10 (9 superseded by ALLOC nights, 1 stray)
companion rows that will cascade: 1 (pk 334 -> run 68)
blank-url events in this database: 10
deleted: 10
cascaded: 1 CalendarEventMeta (pk 334's companion row)
ALLOC nights unchanged: 9/9
blank-url events in this database: 0
```

Copy event count went 280 -> 270; none of the ten pks remains; pk 334's companion row is gone; the nine `ALLOC:` urls still carry the census spans; run 68 and event 357 are still present.

## Refusal and re-run proofs

- Broken copy (pk 44's `end_time` shifted by one second, on the scratch copy only): the real-mode script exited 1, printed `PRE-FLIGHT FAILED: pk 44: span differs from ALLOC:76:2026-07-09 (... 11:29:49 vs ... 11:29:48)`, printed no deletion line, and the copy still held all ten rows and pk 334's companion row.
- Re-run on the already-repaired copy: exited 1 with `PRE-FLIGHT FAILED: pk 44: event is missing` (and the same for every pk, plus `expected exactly 1 companion row (pk 334), found 0`), `11 pre-flight problem(s) found; nothing was written.`

## The pk 334 finding

pk 334 is not companion-free. One unconfirmed `CalendarEventMeta` row attributes it to run 68 (`FTN/MuSCAT3`, `source=legacy`, in `WR06 tmp campaign`, a Phase 35 WR-06 review leftover that owns its own `ALLOC:68:2025-07-04`, event 357). The delete cascades that row, and the script asserts it exactly. Run 68, TargetList 10 and event 357 were left alone for the operator to decide.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Description-containment check was stricter than the data (planning finding 3)**
- **Found during:** Task 1 (first dry run on the scratch copy; pre-flight correctly refused)
- **Issue:** The plan asserted that every orphan description line is an exact line of its `ALLOC:` event's description. For the four NTT nights (44-47) the loader appends ` [117.2A2N.001]` to the same `Source line:` line, so it is the orphan's line plus a tag, not an exact match. Pre-flight reported four failures, which is the safety net working as designed.
- **Fix:** A line now counts as carried when an `ALLOC:` line equals it, or equals it followed by ` [`... (the loader's bracketed proposal tag). Still strict: any other difference fails. Dry run, real run, refusal and re-run were all done after the fix.
- **Files modified:** `retire_orphan_events.py`
- **Commit:** `c0be3a4`

**2. [Rule 3 - Blocking] Sphinx pre-commit hook failed on the first commit attempt**
- **Found during:** Task 1 commit
- **Issue:** The `sphinx-build` hook failed (`docs/autoapi/fomo/asgi/index.rst does not exist`) because of the untracked, generated `docs/autoapi/` directory that was present at the start of the session; the autoapi extension removes and regenerates that output directory during the build. Unrelated to the script (`.planning/` is outside every hook's scope). The failed hook run left `docs/autoapi/` deleted.
- **Fix:** Re-ran the identical commit; all hooks passed (no `--no-verify`, no amend; the first attempt created no commit). `docs/autoapi/` is regenerated build output (untracked) and is no longer in the working tree; the next Sphinx build recreates it.

### Other notes

- The live database now holds 280 events (the plan said 294 at planning time). The ten orphans and every census fact still matched exactly, so this is unrelated cron drift, not a census change.
- `git add` / `git commit` ran only on the script path; `.planning/v2.4-INTENT-REVIEW.md` was never staged.

## Paired docs

No notebook or runbook was changed: this is a one-off data repair of ten literal pks on one database; it changes no module's behaviour, none of the mapped modules or `docs/runbooks/` pages is touched, and no runbook documents these rows (planning finding 9).

## Operator follow-up

- Setup step 5 in `.planning/v2.4-INTENT-REVIEW.md` is struck through with a Done note (script path, pre-flight, the pk 334 finding, scratch evidence, expected live output). The file is modified in the working tree beside the operator's other edits and is deliberately uncommitted.
- The live run is the operator's: wait for a `=== FOMO unattended run END` banner, take a fresh `.backup`, dry-run, then run a `DRY_RUN = False` copy with a `<` redirect. Expect `asserted: 10`, `deleted: 10`, `cascaded: 1 CalendarEventMeta (pk 334's companion row)`, `ALLOC nights unchanged: 9/9`, blank-url count 10 -> 0.
- Decide separately whether run 68 / `WR06 tmp campaign` / event 357 should also go.

## Known Stubs

None.

## Self-Check: PASSED

- Script present at `.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py`; commit `c0be3a4` exists on `issue37-telescope-runs-calendar` and touches only that file; `DRY_RUN = True` in the committed blob.
- Live DB read-only check after all runs: all ten orphans still present.
