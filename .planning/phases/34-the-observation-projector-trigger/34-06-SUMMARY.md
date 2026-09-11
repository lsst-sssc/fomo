---
phase: 34-the-observation-projector-trigger
plan: 06
subsystem: calendar-sync
tags: [django, jupyter, observation-projector, gap-closure, updatestatus, uat-evidence]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    provides: "34-05: coerce_schedule_datetime() closing G-34-2 at the unit/integration level"
provides:
  - "project_observation_calendar_demo.ipynb re-executed with the G-34-2 fix documented and demonstrated: the receiver-demo cell narrows an event from the LCO portal's own ISO-8601 strings, and the notebook is routable to a scratch database copy without spending the SCHED-06 evidence"
  - "Live, portal-backed proof that G-34-2 is closed: a real updatestatus run over a scratch copy of the developer database logs zero unprojectable lines, and the following dry-run sweep reports updated: 0, unprojectable: 0 for LCO"
  - "Runbook determination recorded: docs/runbooks/telescope_runs_calendar.rst needs no edit for this fix"
affects: [34-UAT.md, "the operator's next real updatestatus run against src/fomo_db.sqlite3"]

# Actuals (#2632)
actuals:
  tokens: 9200
  tasks: 2
  commits: 1
plan_head_before: 639e1a10a4e4af5e6dce4677c704d048bac3dd39

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SCRATCH_DB_OVERRIDE pattern: a demo notebook reads FOMO_DATABASE_PATH once into a module-level variable and branches its own assert/print/write behavior on whether it is set, so the same notebook can run against the real developer database (the default, still the point) or a scratch copy (when the developer database is carrying evidence a sweep or a baseline-write would destroy) — mirrors the pattern already used by reconcile_campaign_runs_demo.ipynb and campaign_lifecycle_demo.ipynb, extended here with a guarded evidence-file write rather than just a guarded assert."

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb

key-decisions:
  - "Removed the now-unused `datetime`/`timezone as dt_timezone` imports from the receiver-demo cell (cell 4) after switching its schedule assignment to raw ISO-8601 strings — the plan's edit instruction only named the assignment and print-statement changes explicitly, but leaving the two imports in place would have left dead code; ruff itself does not lint notebooks (types_or: [python, pyi] excludes jupyter in this repo's ruff hook), so this was a cleanliness choice, not a gate requirement."
  - "Task 2 produced no tracked-file commit of its own: `tmp/34-06-updatestatus.txt` and `tmp/34-06-dry-run.txt` are gitignored evidence files (matching the plan's own `files_modified` scope, which lists only the notebook), so there is nothing for git to stage once the evidence is captured. All of Task 2's proof lives in this SUMMARY and the gitignored tmp/ files."

requirements-completed: [PROJ-02, TRIG-01, TRIG-02, SCHED-06]

coverage:
  - id: D1
    description: "project_observation_calendar_demo.ipynb documents and demonstrates the G-34-2 fix with real executed output: the receiver-demo cell assigns scheduled_start/scheduled_end as the portal's own ISO-8601 strings and the event still narrows"
    requirement: PROJ-02
    verification:
      - kind: other
        ref: "notebook cell 4 output (this plan's re-execution): '2. After a schedule-only save ... assigned scheduled_start=\\'2026-09-20T03:00:00Z\\' scheduled_end=\\'2026-09-20T05:00:00Z\\' ... title=\\'[S] 2m0 observation-projector-demo-target\\''"
        status: pass
    human_judgment: false
  - id: D2
    description: "The notebook is re-executable against a scratch copy of the developer database without writing to src/fomo_db.sqlite3 or overwriting the committed SCHED-06 baseline JSON, and names which database the committed run used"
    requirement: SCHED-06
    verification:
      - kind: other
        ref: "test \"$(git status --porcelain docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json src/fomo_db.sqlite3 | wc -l)\" -eq 0 (Task 1 automated verify)"
        status: pass
      - kind: other
        ref: "notebook cell 2 output: 'Resolved database: .../tmp/fomo_g34_2_copy.sqlite3 -- routed to a scratch copy, not the developer database.'"
        status: pass
    human_judgment: false
  - id: D3
    description: "The 33 stale LCO events remain on the developer database, untouched by this plan, available for the operator's next real updatestatus run to repair through the receiver (SCHED-06/UAT Test 4 evidence)"
    requirement: SCHED-06
    verification:
      - kind: other
        ref: "src/fomo_db.sqlite3 mtime/size unchanged (1789157248 / 1232896 bytes, matching the value recorded at dispatch) and git status --porcelain src/fomo_db.sqlite3 empty, checked before and after every step in this plan"
        status: pass
    human_judgment: false
  - id: D4
    description: "A real, portal-backed updatestatus run against a copy of the developer database logs zero unprojectable lines, and the following dry-run sweep reports updated: 0, unprojectable: 0 for LCO -- the direct, live proof G-34-2 is closed"
    requirement: TRIG-01
    verification:
      - kind: other
        ref: "grep -c 'unprojectable' tmp/34-06-updatestatus.txt == 0; grep -o 'LCO: [^|]*' tmp/34-06-dry-run.txt | tail -1 matches 'updated: 0,' and 'unprojectable: 0' (Task 2 automated verifies)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The receiver never raises out of a real save (TRIG-02) -- updatestatus completed successfully with zero AttributeError/unprojectable lines over 52 non-terminal LCO records"
    requirement: TRIG-02
    verification:
      - kind: other
        ref: "tmp/34-06-updatestatus.txt: 'Update completed successfully', 0 AttributeError, 0 OperationalError, 0 unprojectable lines"
        status: pass
    human_judgment: false
  - id: D6
    description: "The runbook determination (docs/runbooks/telescope_runs_calendar.rst needs no edit) is confirmed and recorded with the specific line numbers checked"
    verification: []
    human_judgment: true
    rationale: "Confirming that the runbook's existing prose already describes post-fix behavior (rather than documenting the broken pre-fix path) is an interpretive reading of documentation text, not a mechanical check a grep alone can certify -- the grep only locates the three regions; a human/agent judgment call is what confirms their content matches the objective's pre-stated conclusion."

# Metrics
duration: ~10min
completed: 2026-09-11
status: complete
---

# Phase 34 Plan 06: G-34-2 Paired-Docs Closure & Live Proof Summary

**Re-executed `project_observation_calendar_demo.ipynb` against a scratch copy of the developer database to document and demonstrate the G-34-2 schedule-string fix, then proved it live with a real `updatestatus` run (zero `unprojectable`) and a converging dry-run sweep (`updated: 0, unprojectable: 0` for LCO) -- leaving the developer database and its SCHED-06 baseline evidence untouched throughout.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-09-11T21:21:20Z
- **Completed:** 2026-09-11T21:31:16Z
- **Tasks:** 2
- **Files modified:** 1 (tracked); 3 gitignored evidence/scratch files created under `tmp/`

## Accomplishments

- Made all seven planned edits to `project_observation_calendar_demo.ipynb` before any re-execution: a `SCRATCH_DB_OVERRIDE` guard in the Django-setup cell (reads `FOMO_DATABASE_PATH`, branches its assert/print between "real developer database" and "routed to a scratch copy"); the receiver-demo cell now assigns `scheduled_start`/`scheduled_end` as raw ISO-8601 strings with a trailing `Z` (matching `BaseObservationFacility.update_observation_status()`'s real contract) and prints the assigned raw values with `!r`; the SCHED-06 baseline cell only writes `project_observation_calendar_demo.sched06-baseline.json` when running against the real developer database, and otherwise reads back and reports the existing file's `captured_at` instead of overwriting it; four markdown cells updated to narrate the override, the G-34-2 path the receiver demo now exercises, the one-time-takeover framing for a re-execution routed to a copy, and the "what happens next" re-execution instructions.
- Copied `src/fomo_db.sqlite3` (1,232,896 bytes, no `-wal`/`-shm` siblings) to `tmp/fomo_g34_2_copy.sqlite3` and re-executed the notebook end to end with `FOMO_DATABASE_PATH` set to that absolute path. All 12 code cells carry fresh output; the notebook's own cell 2 output names the scratch copy it ran against verbatim.
- The notebook's own first sweep (cell 8) on the copy: `Done. failed: 0 | LCO: created: 0, updated: 33, unchanged: 126, unprojectable: 0, site_lookups: 14, site_lookup_failed: 1 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0` -- the `updated: 33` matches the 33 LCO events G-34-2 left stale on the developer database (the copy was cloned from it), repaired here on the copy by the sweep exactly as the receiver will repair them on the developer database at the operator's next `updatestatus`.
- The notebook's own second sweep (cell 10): `Done. failed: 0 | LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 1 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0` -- convergence, unchanged from the notebook's pre-existing assertion structure.
- The SCHED-06 baseline cell (cell 15), routed to the copy: printed `Routed to a scratch copy -- the committed SCHED-06 baseline is deliberately left as it stands (captured_at='2026-09-11T04:44:59.526430+00:00'); ...` and did not touch the committed baseline JSON -- confirmed via `git status --porcelain` reporting it clean throughout the plan.
- A real, portal-backed `python manage.py updatestatus` run against the same scratch copy (52 non-terminal LCO records) completed with `Update completed successfully`, zero `unprojectable`, zero `AttributeError`, zero `OperationalError` lines (`tmp/34-06-updatestatus.txt`, gitignored, quoted here credential-free).
- The following `python manage.py project_observation_calendar --dry-run` against the same copy reported `Done (dry run). failed: 0 | LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0` (`tmp/34-06-dry-run.txt`) -- the corroborating signal that every save `updatestatus` just made was already projected as it happened.
- Confirmed `src/fomo_db.sqlite3` untouched throughout: `git status --porcelain src/fomo_db.sqlite3` empty at every checkpoint, and its mtime/size (`1789157248` / `1232896` bytes) unchanged from the value recorded at dispatch. A read-only `sqlite3 -readonly` count of facility-url-keyed `CalendarEvent` rows on the developer database (159) matches the notebook's own pre-sweep snapshot of the copy, confirming no drift.
- Runbook determination confirmed and recorded (Step 5, below) -- `docs/runbooks/telescope_runs_calendar.rst` needs no edit.

## Runbook Determination

Grepped `docs/runbooks/telescope_runs_calendar.rst` for `updatestatus`, `post_save`, and `unprojectable` and read the three regions the hits land in:

- **Lines 47-65** ("How do LCO/SOAR queue observations get onto the calendar?"): already states that saving an `ObservationRecord` -- "whether FOMO submitted it, TOM's `updatestatus` refreshed it from the LCO portal, or `backfill_lco_observation_records` created it" -- draws or updates that record's own `CalendarEvent` with no operator command, via the `post_save` receiver "live since Phase 34." This is exactly the behavior 34-05 restores; the page describes intended behavior, not the broken G-34-2 path, and never claimed a real `updatestatus` save would fail.
- **Lines 144-198** ("When would I run the sweep?"): describes `project_observation_calendar`'s `unprojectable`/`created`/`updated`/`unchanged`/`site_lookups` counters and the sweep-vs-`post_save` division of labor. Unchanged by this fix -- the sweep's own behavior (re-fetching DB-stored `datetime` values) never hit the G-34-2 bug, which was specific to an in-memory post-save instance holding portal strings.
- **Lines 1144-1155** (troubleshooting table, `unprojectable` row): same conclusion -- describes the sweep's counter semantics, not the receiver's real-save path, and is unaffected by the fix.

**Conclusion: no edit needed.** Confirms the objective's pre-stated determination.

## Task Commits

1. **Task 1: Teach the demo notebook the portal-string path and re-execute it on a copy** - `73b465d` (feat) -- notebook edits + re-execution against `tmp/fomo_g34_2_copy.sqlite3`.
2. **Task 2: Prove the fix live against the copy and record the runbook determination** -- no tracked-file commit. Produces only gitignored evidence (`tmp/34-06-updatestatus.txt`, `tmp/34-06-dry-run.txt`) and the runbook determination recorded above and in this SUMMARY; per the plan's own `files_modified` scope, there was nothing to stage for this task.

**Plan metadata:** committed alongside this SUMMARY, STATE.md, and ROADMAP.md.

## Files Created/Modified

- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` - seven edits teaching the notebook the `FOMO_DATABASE_PATH` override and the portal ISO-string receiver path; re-executed against a scratch copy
- `tmp/fomo_g34_2_copy.sqlite3` (gitignored) - scratch copy of the developer database used for the notebook re-execution and both Task 2 commands
- `tmp/34-06-updatestatus.txt` (gitignored) - captured stdout+stderr of the live `updatestatus` run
- `tmp/34-06-dry-run.txt` (gitignored) - captured stdout+stderr of the following dry-run sweep

## Decisions Made

- Removed the now-unused `datetime`/`timezone as dt_timezone` imports from the receiver-demo cell after switching its schedule assignment to raw ISO-8601 strings, rather than leaving dead imports behind -- a cleanliness choice, not a gate requirement (ruff's lint hook in this repo does not lint notebooks; only `ruff-format` does, and that only reformats, never flags unused imports).
- Task 2 intentionally produced no tracked-file commit: both its outputs are gitignored evidence files matching the plan's own `files_modified` scope (which names only the notebook), so all its proof is captured in this SUMMARY and the gitignored `tmp/` files rather than in a git commit.

## Deviations from Plan

None - plan executed exactly as written (see Decisions Made above for two minor, in-scope implementation choices that did not deviate from the plan's must-haves or verify commands).

## Issues Encountered

None. The live `updatestatus` run against the 52 non-terminal LCO records on the copy completed quickly (Django's console log handler only surfaces `unprojectable`/warning-level lines, not a line per successful save, so a fast, quiet run with `Update completed successfully` and zero `unprojectable` lines is the expected, correct outcome, not a sign the command skipped work).

## Post-plan human check

Per the plan's Task 2 `<human-check>` verify item, the following is **operator work to be done after this plan is committed**, not something this execution performed:

> After this plan is committed, run `python manage.py updatestatus` -- and nothing else -- against the real developer database. It should repair the 33 stale LCO events through the `post_save` receiver, with no `unprojectable` lines. That run, and the dated re-check row it lets you fill in, is the SCHED-06 evidence UAT Test 4 has been waiting for (D-20). Do not run `project_observation_calendar` against the developer database before then.

This plan deliberately left `src/fomo_db.sqlite3` and its 33 stale LCO events untouched so that operator run remains available as live evidence.

## User Setup Required

None - no external service configuration required. Credentials (`LCO_API_KEY`) were already present in the gitignored `src/fomo/local_settings.py` and were confirmed working (portal reachable, `updatestatus` authenticated) without ever being printed, echoed, or pasted into any evidence file, this SUMMARY, or a commit message.

## Next Phase Readiness

- G-34-2 is now closed at the paired-docs and live-proof levels: the notebook demonstrates the fix with real executed output, and a real portal-backed `updatestatus` run over 52 non-terminal LCO records produced zero `unprojectable` lines with a converging dry-run sweep.
- `34-UAT.md`'s Test 4 (SCHED-06) remains open pending the operator's own `updatestatus` run against the real developer database -- this plan deliberately preserved the 33 stale LCO events there as that test's evidence (see "Post-plan human check" above).
- No further Phase 34 plan is pending after this one in the current phase directory listing; the operator's next step is the post-plan human check above, followed by updating `34-UAT.md`'s dated re-check table once real observing nights have elapsed.

## Self-Check: PASSED

- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`: FOUND, modified, committed
- Commit `73b465d` present in `git log --oneline --all --grep="(34-06)"`
- Acceptance criteria re-run: Task 1's two `<verify>` commands pass; Task 2's three automated `<verify>` commands pass (`grep -c unprojectable` == 0; dry-run LCO segment shows `updated: 0,`/`unprojectable: 0`; `git status --porcelain` for dev DB + baseline JSON == 0 lines)
- Plan-level `<verification>` re-run: zero `unprojectable` in `tmp/34-06-updatestatus.txt`; `tmp/34-06-dry-run.txt` LCO segment `updated: 0, unprojectable: 0`; notebook committed with output in all 12 code cells naming the scratch copy; `git status --porcelain` for `src/fomo_db.sqlite3` and the SCHED-06 baseline JSON empty across the whole plan; `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both pass repo-wide
- `src/fomo_db.sqlite3` mtime/size (`1789157248` / `1232896` bytes) unchanged from the value recorded at dispatch

---
*Phase: 34-the-observation-projector-trigger*
*Completed: 2026-09-11*
