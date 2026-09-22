---
phase: 260922-dva
plan: 01
subsystem: docs
tags: [unattended-runner, runbook, lock-directory, sched-08, sched-09]
status: complete
dependency-graph:
  requires: []
  provides:
    - "Corrected fresh-host lock/state directory setup instructions"
    - "Post-reboot heartbeat troubleshooting entry"
  affects:
    - docs/runbooks/telescope_runs_calendar.rst
tech-stack:
  added: []
  patterns:
    - "Durable-path-first, tmpfiles.d-alternative-second documentation pattern for cron-owned state directories"
key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
decisions:
  - "Recommended durable lock-directory path is ~/.local/state/fomo (cron-account-owned, no sudo needed), with /etc/tmpfiles.d/fomo.conf offered as the alternative for operators who want to keep /var/lock/fomo"
  - "FOMO_STATE_DIR must be set explicitly alongside FOMO_LOCK_DIR because settings.py captures its default at definition time, before local_settings.py is imported"
  - "Step 8's crontab-template comparison reworded to name FOMO_LOCK_DIR specifically as the value that will no longer match the template's hardcoded default, rather than grouping it with FOMO_LOG_FILE"
metrics:
  duration: ~25min
  completed: 2026-09-22
actuals:
  tokens: 1860
  tasks: 2
  commits: 3
  plan_head_before: 6e08361d966369e39364b99446fc141866da76d6
---

# Quick Task 260922-dva: Document the unattended runner's lock-directory outage Summary

Corrected the fresh-host setup step that told operators to create the unattended runner's lock
directory on a tmpfs (`/var/lock/fomo` → `/run/lock`), and added a troubleshooting entry for the
silent-stall failure mode that instruction produced on the real host on 2026-09-21.

## What Was Built

**Task 1 — Corrected the fresh-host lock and state directory step.** Rewrote step 1 of "Setting
it up on a fresh host" (`docs/runbooks/telescope_runs_calendar.rst`) to:

- State plainly that `/var/lock` is a symlink to `/run/lock`, a tmpfs, so anything created there
  is erased on every reboot — a property of the path, not a possibility.
- Explain why the failure is permanent rather than self-healing: the crontab's `flock` guard can't
  open its lock file after a reboot (tick dies before Python starts), and even with the crontab
  line repointed, `command_lock()`'s own `mkdir` fails because an unprivileged cron account cannot
  create anything inside root-owned `/run/lock`.
- Recommend `~/.local/state/fomo` (cron-account-owned, no `sudo`) as the primary route, with a
  `/etc/tmpfiles.d/fomo.conf` entry as the alternative for operators who want to keep the
  conventional `/var/lock/fomo` path — including that alternative's limitation (it restores the
  directory on boot, not what a prior boot left in it).
- Warn about the `FOMO_STATE_DIR` second-order trap: it defaults from `FOMO_LOCK_DIR` at the point
  `settings.py` defines it, before `local_settings.py` is imported, so setting only `FOMO_LOCK_DIR`
  leaves the D-11 suppression-state file on the tmpfs — dropped every reboot, causing repeated
  "newly failing" re-mails.
- Add the ordering note: set both variables before running the step-6 preflight, so the printed
  cron line already reflects the corrected paths.
- Removed the old "no third directory is needed" claim that this task directly contradicts.
- Left the WR-17 temp-directory fallback paragraph (a different mechanism — the state file going
  unwritable at run time, not the lock directory being wiped at boot) completely untouched.
- Reworded step 8's crontab-template comparison sentence so it no longer implies `FOMO_LOCK_DIR`
  is still at its default when comparing against `deploy/cron/fomo.crontab.example`.

**Task 2 — Added the post-reboot heartbeat troubleshooting entry.** Inserted "The heartbeat went
down after a reboot and no email arrived" between the two existing heartbeat entries in
`Troubleshooting`, in the sibling entries' exact `**Cause:**`/`**Fix:**` house style:

- Names the identifying signature (healthcheck down, no failure email) and explains why the
  absence of email is expected here, not a second fault.
- Ties the symptom to the tmpfs lock directory and cites the real incident's numbers (100
  consecutive ticks lost over about 25 hours).
- Gives the two diagnostic greps (`flock: cannot open`, `PermissionError` with errno 13) against
  `/var/log/fomo/unattended.log`, explaining they are two forms of one root cause rather than two
  separate problems.
- Fix: set `FOMO_LOCK_DIR` and `FOMO_STATE_DIR` to durable storage and change the crontab's
  `flock` path to match (all three), then re-run `check_unattended` as the cron account. Closes
  with a cross-reference to the corrected "Setting it up on a fresh host" step rather than
  restating it.

## Deviations from Plan

None — plan executed exactly as written. Both tasks' full automated verification gates (Sphinx
docutils-warning gate, region-scoped `awk`/`grep` content gates, `pre-commit run --all-files`, and
the `git status --porcelain solsys_code src` docs-only gate) passed on first attempt with no
retries needed.

### Post-execution correction (commit `181df9c`, orchestrator)

Orchestrator review caught one defect the plan's own gates could not: the Task 2
troubleshooting entry documented the diagnostic as `grep 'errno 13'`, but Python emits
`[Errno 13]` with a capital E. Tested against the real incident log, the pattern as written
returned **0 hits on a log that does contain the error** — a diagnostic that silently fails
the operator mid-incident. Corrected to `grep 'Errno 13'` and re-verified (1 hit). The
content gates checked that the greps were *present*, not that they *match real log output*;
worth remembering when writing gates for documented commands.

## Auth Gates

None encountered.

## Known Stubs

None. This is a documentation-only change with no code, no data flow, and no UI — there is nothing
that could stub a data source or defer wiring.

## Threat Flags

None. Per the plan's own threat model, all four in-scope STRIDE entries (T-260922-dva-01 through
-04) are fully mitigated by the two tasks above, and no new security-relevant surface (network
endpoint, auth path, file-access pattern, or schema change) was introduced — this plan touches one
`.rst` file and nothing under `solsys_code/` or `src/`.

## Out of Scope (recorded, not actioned)

Per the plan's `<out_of_scope>` block: `deploy/cron/fomo.crontab.example` hardcodes
`/var/lock/fomo` at lines 9, 67 and 71 and has the same defect this plan corrects in the runbook.
It was deliberately left unchanged (outside this task's single-file scope); step 8's reworded
sentence now tells an operator the template's lock path will not match a host that followed the
corrected step 1. **Recommend a follow-up quick task to fix the template itself.**

## Self-Check: PASSED

- `docs/runbooks/telescope_runs_calendar.rst` — FOUND (modified, both commits present)
- Commit `6cd9c81` (Task 1) — FOUND in `git log --oneline`
- Commit `142848f` (Task 2) — FOUND in `git log --oneline`
- Sphinx build reports zero `docutils`-category warnings naming this file (one pre-existing
  `ref.doc` warning about the excluded notebook remains, as expected per the plan's baseline)
- `git status --porcelain solsys_code src` — empty (nothing under `solsys_code/` or `src/` touched)
- `pre-commit run --all-files` — clean
