---
phase: 36-unattended-operation
fixed_at: 2026-09-17T17:43:08Z
review_path: .planning/phases/36-unattended-operation/36-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 36: Code Review Fix Report

**Fixed at:** 2026-09-17T17:43:08Z
**Source review:** .planning/phases/36-unattended-operation/36-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 10 (2 critical, 8 warning; fix_scope=critical_warning, Info findings excluded)
- Fixed: 10
- Skipped: 0

All fixes were applied in an isolated git worktree (`gsd-reviewfix/36-27188`, branched
from `issue37-telescope-runs-calendar`), committed one finding per commit, verified with
`pre-commit run ruff`/`ruff-format` and `python manage.py test` after each commit (the
project's own pre-commit hooks additionally ran the full ruff/ruff-format/Sphinx-docs/
unit-test gate on every commit), then fast-forwarded onto the working branch.

## Fixed Issues

### CR-01: The shipped cron line self-deadlocks — every scheduled tick runs nothing and exits 0

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `deploy/cron/fomo.crontab.example`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `5fbcb51`
**Applied fix:** Renamed the cron guard's lock file from `run_unattended.lock` to
`run_unattended.cron.lock` in `cron_line()` and the committed crontab template, so it can
never collide with `command_lock('run_unattended')`'s own internal lock file. Added a
regression test (`test_cron_lock_differs_from_the_runner_internal_lock`) asserting the two
paths are never equal, and updated the pinned-shape test for the new filename.

### CR-02: `--created-after`/`--created-before`/`--username`/`--target-list` are silently ignored on the bare `backfill_lco_observations` invocation

**Files modified:** `solsys_code/management/commands/backfill_lco_observations.py`, `solsys_code/tests/test_backfill_lco_observations.py`, `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`
**Commit:** `f5f4cfd`
**Applied fix:** `handle()` now raises `CommandError` before the watched-list loop when
any of the four `--proposal`-only flags is given without `--proposal`, naming which
flag(s) were rejected. Updated each flag's `--help` text and the docstring's `Raises:`
section to document the new contract, and added a short markdown note to the paired demo
notebook's "bare invocation" cell (no re-execution needed — prose only). Added five new
tests covering each rejected flag plus a "legitimate bare invocation still sweeps" guard.

### WR-01: The crontab's `|| echo "... lock held"` tail mislabels every failing tick as a skipped one

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `deploy/cron/fomo.crontab.example`, `docs/runbooks/telescope_runs_calendar.rst`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `abd236c`
**Applied fix:** `flock -n -E 99` now gives lock contention a dedicated exit code, and the
skip-line tail is gated on `[ $? -eq 99 ]` instead of a bare `||` that also fired on
`run_unattended`'s own exit 1 (a step failure). Mirrored in `cron_line()`, the crontab
template, and the runbook's "Repeated lock held lines" troubleshooting section (which now
also disambiguates the cron-guard lock file from the runner's internal one, tying into
CR-01). Added tests asserting the tail is gated on `-E 99`, not `||`.

### WR-02: A failed or unsent notification is still recorded as "staff were notified"

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `8dc2b7a`
**Applied fix:** `_send_notification()` now returns whether the mail was actually
attempted *and* delivered (`notifications.notify_staff()`'s own return value, or `False`
on a caught send exception). `run_tick()` only calls `save_state()` when that return
value is `True`. Added `test_failed_send_does_not_save_state`, which forces a send
failure, confirms no state was persisted, then confirms a later working tick still mails
staff (proving the failed send did not suppress it).

### WR-03: The suppression-state file is not fail-safe — three inputs abort the tail of every tick, permanently

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `c606f97`
**Applied fix:** `load_state()` now validates the top-level JSON is a dict (not, e.g., a
bare list), catches an unparseable/naive `notified_at` and normalizes it to UTC, and
returns an already-parsed `datetime` rather than a raw string — `decide_notification()`
no longer re-parses it. The whole
`load_state()`/`decide_notification()`/`_send_notification()`/`save_state()` block in
`run_tick()` is now wrapped in `try/except Exception` (logging the class name only), so
the END banner and exit-code heartbeat ping always run even if that block raises. Added
four tests: non-dict state file, malformed `notified_at`, naive `notified_at` normalized
to aware, and a `save_state()` failure that must not block the END banner/heartbeat.

### WR-04: The END banner reports the tick's *start* time, so no tick's duration is readable

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `3eb10e8`
**Applied fix:** `run_tick()` now samples a fresh `end_time = datetime.now(dt_timezone.utc)`
after all steps have run, and uses it for the END banner, `decide_notification()`, and
`save_state()` (previously all three reused the START-of-tick `now`). Added
`test_end_banner_timestamp_differs_from_start_banner_timestamp`, which patches
`datetime.now()` to return two distinct sampled values and asserts the END banner logs
the second one, not the first.

### WR-05: `cron_line()` hardcodes `/usr/bin/flock` while `check_flock()` resolves the real path

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `75aace1`
**Applied fix:** `cron_line()` now resolves `flock` via `shutil.which('flock') or
'/usr/bin/flock'` — the same resolution `check_flock()` already performs — instead of a
hardcoded path. Added `test_flock_path_is_resolved_not_hardcoded`, which patches
`shutil.which` to a non-default path and asserts the printed line uses it.

### WR-06: The writability preflight cannot check what the deploy docs claim it checks

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`, `deploy/logrotate/fomo.example`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `0b240d2`
**Applied fix:** `_check_directory_writable()`'s detail text now reports the resolved uid,
owner uid, and permission mode, and states explicitly whose write access was tested
(`writable by uid {geteuid()} ... run this check as the account that will actually run
unattended to verify it too`). Corrected the "verifies that directory is writable by the
user the runner will actually run as" claim in `deploy/logrotate/fomo.example` and added
an explicit "run as the cron account, not root" instruction to the runbook's setup
walkthrough (step 4). Added `test_writable_result_names_the_uid_that_was_actually_tested`.

### WR-07: The campaign-submission notice silently loses the real host in its approval-queue link

**Files modified:** `solsys_code/notifications.py`, `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`, `solsys_code/tests/test_campaign_submission.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `c5f6b64`
**Applied fix:** Added a new soft `check_base_url()` preflight check (warns when
`FOMO_BASE_URL` is unset or still the `http://localhost:8000` dev default), wired into
`handle()` and documented in the runbook's setup walkthrough and the command's own
`--help`/module docstring. Also guarded `notifications.absolute_url()` against a `None`
`FOMO_BASE_URL` (falls back to the documented default instead of raising
`AttributeError` from `None.rstrip()`), which would otherwise have escaped
`_notify_staff()` and broken the submission itself. Added tests: the new warning/ok
states for `check_base_url()`, a submission test asserting the approval-queue link
carries a configured non-default host, and a submission test proving a `None`
`FOMO_BASE_URL` no longer crashes the request.

### WR-08: The status-refresh step doubles portal traffic on every failure, uncapped

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `967fa07`
**Applied fix:** Added `_MAX_STATUS_RECHECKS = 20` and capped
`_refresh_one_facility()`'s per-record re-check to the first 20 failed records; the
omitted count is surfaced in the step's summary line (`recheck capped: N omitted`) rather
than silently dropped. `failed_record_count` (used for the step's pass/fail verdict and
the email's failure count) is unaffected — only the number of individual portal re-check
calls is capped. Added tests for the capped-call-count behavior and confirmed the note is
absent when the failure count is under the cap.

## Skipped Issues

None — all 10 in-scope findings (CR-01, CR-02, WR-01 through WR-08) were fixed.

## Notes for the developer

- Info findings (IN-01 through IN-06) were out of `fix_scope=critical_warning` and were
  not touched. IN-05 (the dangling `classes: ` fragment and the "failed 1" ambiguity for
  a whole-facility outage) is adjacent to the WR-08 fix but was left alone to keep that
  commit's diff scoped to the recheck cap only, per the instruction not to guess beyond
  what a finding asks for.
- CR-01 and WR-01 both touch `cron_line()`, the crontab template, and the runbook's
  troubleshooting section; they were applied and committed in dependency order (CR-01
  first, so the lock filename existed before WR-01 added the `-E 99`/exit-code-99 tail
  on top of it) — the final `cron_line()` output combines both fixes cleanly, verified
  by the full `TestCronLine` suite.
- All ten commits pass the full `pre-commit` gate (ruff, ruff-format, Sphinx docs build,
  and the project's unit-test suite), run inside the isolated worktree with the venv's
  editable install pointed at it (a local, gitignored `src/fomo/_version.py` was copied
  in from the main checkout purely to make `manage.py`/pytest importable inside the
  worktree — this is a generated file, never committed, and is not part of any fix diff).

---

_Fixed: 2026-09-17T17:43:08Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
