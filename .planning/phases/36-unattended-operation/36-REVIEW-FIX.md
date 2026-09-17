---
phase: 36-unattended-operation
fixed_at: 2026-09-17T21:27:26Z
review_path: .planning/phases/36-unattended-operation/36-REVIEW.md
iteration: 2
findings_in_scope: 21
fixed: 21
skipped: 0
status: all_fixed
---

# Phase 36: Code Review Fix Report (iteration 2)

**Fixed at:** 2026-09-17T21:27:26Z
**Source review:** `.planning/phases/36-unattended-operation/36-REVIEW.md`
**Iteration:** 2

**Summary:**
- Findings in scope: 21 (7 Warning + 8 Info from this iteration, plus 6 carried-forward
  Info findings IN-01..IN-06 re-confirmed still open — `fix_scope: all`)
- Fixed: 21
- Skipped: 0

All fixes were applied inside an isolated git worktree
(`.claude/worktrees/rf-36-294761-1789678103`, branch `gsd-reviewfix/36-294761`) and
fast-forwarded onto `issue37-telescope-runs-calendar`. Every commit was individually
verified with the project's own gates before being made: `python manage.py test` on the
four affected modules (`solsys_code.tests.test_unattended`,
`solsys_code.tests.test_check_unattended`, `solsys_code.tests.test_backfill_lco_observations`,
`solsys_code.tests.test_campaign_submission`), `pre-commit run ruff`, `pre-commit run
ruff-format`, and `pre-commit run sphinx-build` for `.rst`/notebook changes. The full
`.pre-commit-config.yaml` `Run unit tests` hook (the project's own `test_command`) also
ran and passed on every commit as part of `pre-commit`'s own hook chain. All gates ran
inside the isolated worktree, which is a checkout of the same repository (not a
stripped-down sandbox), so these results are reproducible from the fast-forwarded
`issue37-telescope-runs-calendar` branch after teardown.

## Fixed Issues

### WR-09: The new cron line's exit status is inverted

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`deploy/cron/fomo.crontab.example`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `2b93705`
**Applied fix:** `cron_line()` and the committed crontab template now capture flock's
exit code into `$rc` immediately after the guarded command and end with an explicit
`exit $rc`, so the line's own reported status is always `run_unattended`'s (0 healthy,
1 failing, 99 skipped) rather than the skip-tail's own status. Added a test asserting
the line ends with `exit $rc`, and updated the existing shape/token assertions.

### WR-10: `load_state()` still raises on a mixed-type `failing_steps` list

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `ce2c866`
**Applied fix:** `load_state()` now filters `failing_steps` to `str` elements before
`sorted()` runs, so a state file like `{"failing_steps": [1, "a"]}` is coerced to a
clean default instead of raising `TypeError` (which previously skipped `save_state()`
for the rest of the tick's life, permanently suppressing failure email). Added
`test_mixed_type_failing_steps_are_coerced_not_raised`.

### WR-11: `check_flock()` didn't verify `-E` support

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `fffe861`
**Applied fix:** `check_flock()` now probes `flock --help` for
`--conflict-exit-code` and fails hard when it's absent (util-linux < 2.27), rather than
only checking `shutil.which()`. Added `test_flock_without_conflict_exit_code_support_fails`.

### WR-15: Unwritable `FOMO_STATE_DIR` was never preflight-checked

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `60dae5e`
**Applied fix:** Added `check_state_dir()`, mirroring `check_lock_dir()`, registered as
an eighth hard check in `Command.handle()`. Added `test_unwritable_state_dir_fails` and
extended `test_all_hard_checks_passing_exits_zero`.

### WR-13: `FOMO_BASE_URL` documented as cron-only, but the web process needs it too

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`,
`deploy/cron/fomo.crontab.example`, `docs/installation.rst`
**Commit:** `26ee834`
**Applied fix:** Reworded the runbook's setup step 3 and the crontab template's comment
to say `FOMO_BASE_URL` must be set in both the cron and web-server environments (or once
in `local_settings.py`), and added a note to `docs/installation.rst`.

### WR-12: Runbook still documented `backfill_lco_observations`'s window flags as freely optional

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `e9ce217`
**Applied fix:** Added a sentence after the bare-invocation example stating the four
non-`--dry-run` flags require `--proposal`, and corrected the command cheat-sheet row's
"(all optional)" wording.

### WR-14: Stale-lock-file remedy described an unreachable condition

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `e4ba0ba`
**Applied fix:** Replaced the "if it has genuinely died without releasing the lock
file, remove it" remedy with the correct diagnosis: a `flock` is always released on
process exit (including a crash/OOM kill), so repeated skip lines always mean a tick is
still running; find it with `pgrep -af run_unattended` instead of deleting the lock
file.

### IN-07: `notify_staff()` discarded `send_mail()`'s return value

**Files modified:** `solsys_code/notifications.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `69d9418`
**Applied fix:** `notify_staff()` now returns `bool(send_mail(...))` instead of a
hardcoded `True`, matching `_send_notification()`'s stronger documented claim. Added
`TestNotifyStaffReturnValue` with three cases (real send, zero-sent, fail-silently
suppression).

### IN-08: Cron-line shape test broke on a host with no `flock`

**Files modified:** `solsys_code/tests/test_check_unattended.py`
**Commit:** `c5e45de`
**Applied fix:** Guarded the test's own `shutil.which('flock')` call with the same
`/usr/bin/flock` fallback `cron_line()` uses, and added a second test that reads
`deploy/cron/fomo.crontab.example`'s committed line and compares option tokens
directly, since the original test's name promised that agreement but never checked it.

### IN-09: Three stale `check_unattended` docstrings

**Files modified:** `solsys_code/management/commands/check_unattended.py`
**Commit:** `900caa7`
**Applied fix:** Updated the module docstring's interpolated-values claim (now
includes ownership/permission metadata), the `Command` class docstring (now mentions
the `FOMO_BASE_URL` warning), and the test module docstring's stale "six" checks count
(now "eight" — already partly updated by the WR-15 commit, verified consistent here).

### IN-10: Two runbook passages still described pre-fix `flock -n`

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `f3db68e`
**Applied fix:** Propagated `-E 99` / `run_unattended.cron.lock` into the "When
nothing has appeared" checklist item 4 and the "What the locking does and does not
cover" paragraph, and added a sentence distinguishing the cron guard's skip line from
the runner's own internal-lock message.

### IN-11: Paired notebook's CR-02 note was prose-only

**Files modified:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`
**Commit:** `7e3610c`
**Applied fix:** Added an executed cell calling
`call_command('backfill_lco_observations', created_after='2026-01-01')` inside a
`try/except CommandError`, printing the real message, and regenerated the notebook with
`jupyter nbconvert --to notebook --execute --inplace` (executed against a freshly
migrated worktree-local `src/fomo_db.sqlite3`; one incidental pre-existing PASS line in
an unrelated cell now prints because the fresh DB has no leftover sidereal target from
prior manual runs — not a behavior change).

### IN-12: `--proposal`-only guard ran after username resolution, used truthiness

**Files modified:** `solsys_code/management/commands/backfill_lco_observations.py`,
`solsys_code/tests/test_backfill_lco_observations.py`
**Commit:** `0dec266`
**Applied fix:** Moved the guard above `--username` resolution (so an unknown username
on the bare invocation reports the guard's message, not "Invalid username"), and
switched the predicate from truthiness to `options.get(key) is not None` (so
`--target-list ''` still trips it). Added two new tests.

### IN-13: `step_discovery()` duplicated `Command.handle()`'s watched-proposal loop

**Files modified:** `solsys_code/management/commands/backfill_lco_observations.py`,
`solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `3ac974d`
**Applied fix:** Extracted `sweep_watched_rows(*, dry_run, stdout=None, stderr=None) ->
tuple[int, int, list[str]]` in `backfill_lco_observations.py`; both callers now use it,
leaving each with only its own terminal reporting. Updated test patches that targeted
`solsys_code.unattended.sweep_proposal` to target the real definition site.

### IN-14: `FOMO_LOCK_DIR`/`FOMO_STATE_DIR`/`FOMO_LOG_FILE` unguarded against `None`; `_owner_mode()` could raise

**Files modified:** `solsys_code/unattended.py`,
`solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_unattended.py`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `f1712ed`
**Applied fix:** Guarded all three path settings the same way `FOMO_BASE_URL` already
is (`or '<documented default>'`) at every `Path(...)` construction site in both files,
and wrapped `_owner_mode()`'s `stat()` call in `try/except OSError` returning
`'owner/mode unavailable'`.

### IN-04: `run_unattended`'s docstring said steps come "from later plans in this phase"

**Files modified:** `solsys_code/management/commands/run_unattended.py`
**Commit:** `4390a63`
**Applied fix:** Updated the `Command` class docstring to state all four steps have
shipped.

### IN-02: Discovery discarded every per-request skip reason

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `cf5c7f4`
**Applied fix:** `step_discovery()` now passes captured `io.StringIO()` sinks to
`sweep_watched_rows()` and logs any captured stdout/stderr text at `DEBUG` instead of
letting it sink into a throwaway buffer. Added
`test_per_request_skip_reasons_are_logged_not_discarded`.

### IN-01: Heartbeat ping ignored the HTTP status code

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `178460b`
**Applied fix:** `ping_heartbeat()` now calls `response.raise_for_status()`, so a
non-2xx response (an expired/rotated URL, a rate limit) is caught by the same
`RequestException` handler a network failure already uses. Added
`test_non_2xx_response_is_logged_as_a_failed_ping`.

### IN-03: State file written non-atomically, with the process umask

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `154e1c1`
**Applied fix:** `save_state()` now writes to a fresh temp file in the same directory,
`chmod`s it `0o600`, then `os.replace()`s it into place, instead of writing directly to
the target path. Added `TestStateFileAtomicWrite` (mode, no leftover temp file, content
round-trip).

### IN-05: Whole-facility outage read as "failed 1"; clean run left a dangling `classes: ` fragment

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `8768622`
**Applied fix:** `_refresh_one_facility()` now returns a distinct `outage_class_name`
instead of a placeholder `failed_record_count=1` when the facility call itself raises;
`step_status_refresh()` reports that case as `"LCO: outage (ClassName)"` and only
appends the `classes: ...` segment when non-empty. Added two new tests.

### IN-06: `run_tick(only_step='typo')` silently ran nothing and reported healthy

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `874ae37`
**Applied fix:** `run_tick()` now validates `only_step` against the registered
`STEPS` names up front and raises `ValueError` on a mismatch, rather than silently
matching zero steps and returning a healthy `exit_code=0`. Added
`test_unknown_only_step_raises_instead_of_reporting_a_healthy_no_op`.

## Skipped Issues

None — all 21 in-scope findings were fixed.

---

_Fixed: 2026-09-17T21:27:26Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
