---
phase: 36-unattended-operation
plan: 05
subsystem: docs
tags: [runbook, jupyter-notebook, sphinx, paired-docs, unattended, watched-proposal]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: "36-01's run_unattended runner + crontab/heartbeat/email design, 36-02's WatchedProposal model and bare-invocation sweep, 36-03's full four-step tick, 36-04's check_unattended preflight and logrotate example -- this plan documents all four as they actually shipped"
provides:
  - "docs/runbooks/telescope_runs_calendar.rst -- the 'How do I run everything unattended?' section (label unattended-operation), an amended backfill-without-a-campaign section, two new cheat-sheet rows plus one previously-missing row, and four new Troubleshooting entries"
  - "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb -- three new cell pairs proving the watched-proposal contract with real executed output, plus its own toctree entry"
  - "CLAUDE.md's paired-docs map -- the runner/check_unattended -> runbook-section pairing, with its recorded reason"
affects: []

# Actuals (#2632)
actuals:
  tokens: 9932
  tasks: 3
  commits: 4
  plan_head_before: 6bf8bf6

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A runner module with no CLI/API mock surface small enough to demo (portal + mail backend + heartbeat all mocked at once) is paired with a runbook section instead of a notebook, per 36-CONTEXT.md's discretion note -- recorded explicitly in CLAUDE.md's map so the next phase can tell a deliberate choice from an enforcement hole."

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/notebooks.rst
    - docs/installation.rst
    - CLAUDE.md

key-decisions:
  - "The command cheat-sheet had no row for backfill_lco_observations at all (only its campaign-bound sibling backfill_lco_observation_records was listed) -- the plan's instruction to 'update' that row assumed one already existed. Added a new row rather than treating this as blocking, since the command was already fully documented in its own runbook section; the missing row was itself the pre-existing gap RESEARCH.md's own audit should have caught."
  - "The watched-list demo's two proposal codes route through one proposal-aware make_request side_effect (parsing the portal URL's own 'proposal' query parameter via urllib.parse) rather than two separate patched calls, because a bare sweep invocation makes one call_command() that internally loops over both watched rows -- there is only one patch context to share between them."
  - "The runbook's 'Running it by hand' locking paragraph states the true, narrower guarantee verified directly from solsys_code/unattended.py's command_lock() call sites: two ticks (including a hand-started run_unattended --step <name>, which takes the same lock) never overlap, but a direct manage.py invocation of the underlying sweep command does not check that lock at all and can coincide with a tick. This was cross-checked against the step docstrings' own 'does not make a hand-run manage.py <command> wait for a tick' language before writing it, rather than assuming the discretion note's original aspiration held."

requirements-completed: [SCHED-08, SCHED-09, SCHED-10, DISCOVER-01]

coverage:
  - id: D1
    description: "The runbook's new 'How do I run everything unattended?' section (setup, schedule, both failure signals, the nothing-has-appeared checklist, running it by hand with the true locking guarantee) lets an operator go from a fresh host to a monitored schedule using one section, with no source reading required"
    requirement: SCHED-08
    verification:
      - kind: other
        ref: "pre-commit run sphinx-build --all-files (clean, no new warning); plan's own region-scoped grep probes over the cheat-sheet table and the new section's required terms"
        status: pass
      - kind: manual_procedural
        ref: "Cross-checked every setting name (FOMO_HEARTBEAT_URL, FOMO_BASE_URL, FOMO_LOCK_DIR, FOMO_LOG_FILE), the six check_unattended checks, and the exact locking behavior against solsys_code/unattended.py and check_unattended.py source, not against the plan's aspirational discretion note"
        status: pass
    human_judgment: false
  - id: D2
    description: "backfill_lco_observations's --proposal-optional/watched-list/failure-isolation contract is documented in the runbook, cross-referenced to the new unattended section, with a previously-missing cheat-sheet row added"
    requirement: DISCOVER-01
    verification:
      - kind: other
        ref: "python -c substring checks against the amended section and the cheat-sheet table region"
        status: pass
    human_judgment: false
  - id: D3
    description: "backfill_lco_observations_demo.ipynb gains three new cell pairs (seed + bare sweep, bookkeeping readback, per-proposal failure isolation) with real committed output against a mocked portal, cleans up every row it creates, and is now reachable from docs/notebooks.rst's toctree"
    requirement: DISCOVER-01
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace (exit 0, no live network call); post-execution WatchedProposal.objects.count() == 0; docs/notebooks.rst substring check"
        status: pass
    human_judgment: false
  - id: D4
    description: "CLAUDE.md's paired-docs map records the runner/check_unattended -> runbook-section pairing (with the one-sentence reason a notebook was not used) so the next phase touching unattended.py can tell a deliberate choice from an enforcement hole; installation.rst cross-references the new section"
    requirement: SCHED-08
    verification:
      - kind: other
        ref: "region-scoped grep confirming unattended.py/notifications.py sit inside the 'Paired docs are part of the deliverable' bullet; pre-commit run sphinx-build --all-files clean (no undefined-label warning)"
        status: pass
    human_judgment: false

# Metrics
duration: 33min
completed: 2026-09-17
status: complete
---

# Phase 36 Plan 5: The Unattended-Operation Paired Docs Summary

**A new runbook section takes an operator from a fresh host to a monitored 15-minute cron schedule using only `check_unattended`'s own output; three new pre-executed notebook cells prove the admin-editable watched-proposal contract end to end; CLAUDE.md now tells the next contributor which doc pairs with `unattended.py` and why it is a runbook section, not a notebook.**

## Performance

- **Duration:** 33 min
- **Started:** 2026-09-17T16:34:00Z
- **Completed:** 2026-09-17T17:07:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- New `docs/runbooks/telescope_runs_calendar.rst` section "How do I run everything unattended?" (labeled `unattended-operation`): what runs and in what order, fresh-host setup via `check_unattended`, adding a watched proposal from the admin, the two failure signals (email suppression/reminder/recovery, and the `/start`/`/<exit-code>` heartbeat), the ordered "nothing has appeared" checklist, and running it by hand with the *true* locking guarantee (two ticks never overlap; a direct sweep-command invocation is not locked against one)
- Amended the "How do I backfill ObservationRecords without a campaign?" section for the now-optional `--proposal`, the bare watched-list sweep, the quiet empty-list no-op, and per-proposal failure isolation, cross-referenced to the new section
- Added a `backfill_lco_observations` cheat-sheet row (previously missing entirely) plus `run_unattended`/`check_unattended` rows, and four new Troubleshooting entries
- Three new markdown+code cell pairs in `backfill_lco_observations_demo.ipynb`: seeding two `WatchedProposal` rows (one with a `target_list_name` override) and sweeping both with a bare `call_command('backfill_lco_observations')`; reading back `last_run_at`/`last_run_summary`; and per-proposal failure isolation (one proposal's `ConnectionError` recorded as `failed: ConnectionError` while the other's sweep still completes and creates its record) -- re-executed in place with real committed output, zero live network calls, and the cleanup cell extended so `WatchedProposal.objects.count()` is `0` afterward
- `docs/notebooks.rst` now lists the notebook in the Demonstration Notebooks toctree, closing the pre-existing gap RESEARCH.md found
- `CLAUDE.md`'s paired-docs map records `unattended.py`/`notifications.py`/`run_unattended.py`/`check_unattended.py` as paired with the runbook section (not a notebook), with the one-sentence reason; `docs/installation.rst` cross-references the new section

## Task Commits

Each task was committed atomically:

1. **Task 1: The runbook's unattended-operation section** - `c9c6fcf` (docs)
2. **Task 2: The backfill demo notebook's watched-proposal cells** - `b02f7bf` (docs)
3. **Task 3: CLAUDE.md's notebook map and the installation cross-reference** - `0f1f31a` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified
- `docs/runbooks/telescope_runs_calendar.rst` - new unattended-operation section, amended backfill-without-a-campaign section, cheat-sheet rows, Troubleshooting entries
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` - three new cell pairs, extended cleanup cell, re-executed in place
- `docs/notebooks.rst` - toctree entry for the backfill notebook
- `docs/installation.rst` - cross-reference to the new runbook section
- `CLAUDE.md` - paired-docs map entries for the four unattended-path modules

## Decisions Made
See `key-decisions` in frontmatter: the missing `backfill_lco_observations` cheat-sheet row (added rather than blocking on the plan's "update" framing), the single proposal-aware `make_request` side effect shared by both watched-list demo cells, and cross-checking the "Running it by hand" locking paragraph directly against `unattended.py`'s `command_lock()` call sites and docstrings rather than the discretion note's original aspiration.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added a missing `backfill_lco_observations` cheat-sheet row**
- **Found during:** Task 1, following the action's instruction to "update the `backfill_lco_observations` row's Key flags column to show `--proposal` as optional"
- **Issue:** No such row existed in the `command-cheat-sheet` table at all -- only the campaign-bound sibling `backfill_lco_observation_records` was listed. The command itself was already fully documented in its own runbook section, so this was a pre-existing documentation gap, not new-in-this-plan work.
- **Fix:** Added a new row for `backfill_lco_observations` with its optional `--proposal` (and other optional flags), placed immediately after the sibling command's row.
- **Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
- **Verification:** The plan's own region-scoped probe over the cheat-sheet table confirms both `run_unattended` and `check_unattended` are present; a follow-on manual check confirms `backfill_lco_observations` is now listed too.
- **Committed in:** `c9c6fcf` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug/missing-content)
**Impact on plan:** No scope creep -- the added row documents an already-shipped, already-optional flag; it does not introduce new behavior.

### Known Verify-Probe Imprecision (not a deviation, documented for the verifier)

The plan's Task 2 `<verify>` block includes a probe reading `sum(1 for c in code if c.get('outputs'))` and treats a value lower than the total code-cell count as a failure signal ("the notebook was committed without its output"). This notebook has always had two code cells (the Django-setup boilerplate and the mocking-helpers import cell) that legitimately produce no printed `stdout` -- they still execute (both carry a non-`null` `execution_count` after `nbconvert --execute`) but have an empty `outputs` list, exactly as they did before this plan touched the file (6 of 8 original code cells had output; now 9 of 11). All three new code cells added by this plan do carry real printed output, verified directly above. The probe as literally written would have flagged the pre-existing notebook before any change in this plan, so this is scope-boundary territory (a pre-existing structural fact, not something this plan introduced) rather than a defect to fix by rewriting unrelated existing cells.

## Issues Encountered

`pre-commit`'s `ruff-format` hook reformatted three of the new notebook's code cells (wrapping two over-120-column `print(...)` calls) on the first `git commit` attempt for Task 2, which is expected auto-fix-hook behavior (files were modified, the commit was blocked, the fix was re-staged, and the commit was retried and succeeded) -- not a deviation requiring documentation under the rules above, since this is exactly the "auto-fix hooks handle themselves transparently" case the executor workflow anticipates.

## User Setup Required

None - no external service configuration required. (The runbook's new section documents what a *future* operator does to set up a real host; this plan itself makes no host changes.)

## Next Phase Readiness

This is the last plan of Phase 36. All four Phase 36 requirements (SCHED-08, SCHED-09, SCHED-10, DISCOVER-01) are shared across the phase's five plans and become `Complete` from this plan's `update_requirements` step, since this is the last declaring plan for all four IDs. No blockers for Phase 37.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All key files found on disk and containing the expected content: `docs/runbooks/telescope_runs_calendar.rst` contains `How do I run everything unattended?` and `.. _unattended-operation:`; `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` contains `WatchedProposal` and has 23 cells (11 code cells, 9 with output); `docs/notebooks.rst` lists `backfill_lco_observations_demo`; `CLAUDE.md` names `solsys_code/unattended.py` inside its "Paired docs are part of the deliverable" bullet; `docs/installation.rst` contains ``:ref:`unattended-operation``` . All 3 task commit hashes (`c9c6fcf`, `b02f7bf`, `0f1f31a`) found in `git log`. `pre-commit run sphinx-build --all-files`, `pre-commit run ruff --all-files`, and `pre-commit run ruff-format --all-files` all pass at time of writing. `WatchedProposal.objects.count()` is `0` on the real developer database after the notebook's execution.
