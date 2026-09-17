---
phase: 36-unattended-operation
plan: 04
subsystem: infra
tags: [management-command, cron, logrotate, preflight, credential-hygiene]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: "36-01's run_unattended.py runner (FOMO_LOCK_DIR/FOMO_LOG_FILE/FOMO_HEARTBEAT_URL settings, command_lock()) and its deploy/cron/fomo.crontab.example; 36-02's WatchedProposal model"
provides:
  - "solsys_code.management.commands.check_unattended -- the one-command fresh-host preflight (D-05, SC 5): CheckResult dataclass, check_flock()/check_lock_dir()/check_log_dir()/check_email()/check_heartbeat()/check_watched_proposals(), cron_line(), and --send-test-email"
  - "deploy/logrotate/fomo.example -- the committed daily/rotate-14 logrotate stanza for the unattended log file (D-18)"
affects: [36-05]

# Actuals (#2632)
actuals:
  tokens: 6362
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A read-only preflight command: every check function performs os.access()/shutil.which()/count() reads only, never mkdir/write/save -- proven by TestHardChecks.test_command_writes_nothing asserting the probed directories are still absent after a run"
    - "One CheckResult dataclass (name/ok/hard/detail) shared by six check functions plus the optional --send-test-email outcome, aggregated by handle() into a single [ok]/[WARN]/[FAIL] report and one CommandError naming every failed hard check at once"

key-files:
  created:
    - solsys_code/management/commands/check_unattended.py
    - solsys_code/tests/test_check_unattended.py
    - deploy/logrotate/fomo.example
  modified: []

key-decisions:
  - "check_email() returns a list of two CheckResults (EMAIL_BACKEND, staff_recipients) rather than one -- the plan's own <action> describes it as 'hard, two results', so handle() extends its results list with both rather than the command growing a seventh named check function."
  - "cron_line() and _send_test_email() are module-level, non-check helpers (not among the six check_ functions the plan names) -- keeps the six-callable verify probe in Task 1's <verify> stable across Task 2's additions."
  - "The lock-dir/log-dir writability tests use a directory chmod'd to 0o500 (read+execute, no write) as an unwritable parent, restored via addCleanup before the enclosing TemporaryDirectory tears down -- avoids a real root-owned path dependency in the test suite while still exercising both exists()/os.access() branches of _check_directory_writable()."

patterns-established:
  - "TDD RED verified manually per task (workflow.tdd_mode is false for this project, precedent: plans 34-07, 36-01, 36-02, 36-03) -- confirmed by running each task's full test module before the implementation existed and observing the exact ImportError/AttributeError it produced."

requirements-completed: [SCHED-08, SCHED-10]

coverage:
  - id: D1
    description: "check_unattended reports every prerequisite the unattended path needs (flock, lock dir, log dir, email backend + staff recipients, heartbeat URL, watched-proposal list) in one run, exiting non-zero only on a hard failure; warnings never trip the exit code"
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_check_unattended.py -- TestHardChecks (6 tests), TestWarningChecks (3 tests)"
        status: pass
      - kind: other
        ref: "python manage.py check_unattended on the developer host: exits non-zero, names FOMO_LOCK_DIR/FOMO_LOG_FILE/EMAIL_BACKEND among the failures, flock passes (found at /usr/bin/flock)"
        status: pass
    human_judgment: false
  - id: D2
    description: "cron_line() prints the exact cron line to install with real resolved sys.executable/manage.py/lock/log paths substituted for the committed template's placeholders, matching every element of deploy/cron/fomo.crontab.example, and is printed even when a hard check failed"
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_check_unattended.py -- TestCronLine (3 tests)"
        status: pass
      - kind: other
        ref: "shell probe from 36-04-PLAN.md Task 2 <verify>: printed True True False (all seven elements present, real interpreter substituted, no placeholder survived)"
        status: pass
    human_judgment: false
  - id: D3
    description: "--send-test-email sends one message through the configured backend to the same staff-with-an-email recipients the failure notice uses, reported as one more hard CheckResult"
    requirement: SCHED-08
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_check_unattended.py -- TestTestEmail (3 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "No check_unattended output surface (stdout, stderr, or the CommandError message) ever contains a credential or setting value -- only names, paths, and set/unset status/counts (D-15, SCHED-10, T-36-04/T-36-14)"
    requirement: SCHED-10
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_check_unattended.py -- TestNoValueLeakage.test_output_never_contains_a_seeded_value (seeds a fake heartbeat URL, mail password, and LCO API key; asserts absence under both a passing and a hard-failing configuration)"
        status: pass
    human_judgment: false
  - id: D5
    description: "deploy/logrotate/fomo.example rotates the unattended log file daily, keeps 14, and documents why copytruncate is used instead of create (D-18)"
    requirement: SCHED-08
    verification:
      - kind: other
        ref: "shell probes from 36-04-PLAN.md Task 3 <verify>: all five directives present, rotate 14 appears outside a comment, git ls-files confirms the file is tracked"
        status: pass
    human_judgment: false

# Metrics
duration: 18min
completed: 2026-09-17
status: complete
---

# Phase 36 Plan 4: check_unattended and the Logrotate Example Summary

**A single `check_unattended` management command reports every unattended-path prerequisite (flock, lock/log directories, email backend and staff recipients, heartbeat URL, watched-proposal list) in one run, prints the exact cron line to install with real resolved paths, can send a proof-of-life test email, and never prints a credential value -- paired with the committed `deploy/logrotate/fomo.example` that rotates the one log file daily and keeps a fortnight.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-17T16:14:00Z
- **Completed:** 2026-09-17T16:32:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- `check_unattended.py`: `CheckResult` dataclass plus `check_flock()`, `check_lock_dir()`, `check_log_dir()`, `check_email()` (two results), `check_heartbeat()`, `check_watched_proposals()` -- six read-only checks aggregated by `handle()` into one `[ok]`/`[WARN]`/`[FAIL]` report and a single `CommandError` naming every failed hard check
- `cron_line()`: the exact `flock`-guarded schedule line to install, built from `sys.executable`, the resolved `manage.py` path, and `FOMO_LOCK_DIR`/`FOMO_LOG_FILE` -- printed even when a hard check failed, matching every element of `deploy/cron/fomo.crontab.example`
- `--send-test-email`: sends one message via `notifications.notify_staff()` to the same staff-with-an-email recipients the failure notice uses, reported as one more hard `CheckResult` (a raised send is reported by exception class name only, D-17)
- `deploy/logrotate/fomo.example`: daily rotation, keep 14, `missingok`/`notifempty`/`compress`/`delaycompress`/`copytruncate`, with a comment explaining why `copytruncate` (cron's own redirect holds the descriptor open across a tick)
- `solsys_code/tests/test_check_unattended.py`: 17 tests across `TestHardChecks`, `TestWarningChecks`, `TestCronLine`, `TestTestEmail`, and `TestNoValueLeakage`, including a credential-hygiene regression seeding a fake heartbeat URL, mail password, and LCO API key and asserting their absence under both a passing and a hard-failing run

## Task Commits

Each task was committed atomically:

1. **Task 1: check_unattended -- the prerequisite report** - `fc2e2bc` (feat)
2. **Task 2: The printed cron line and the test email** - `d8d7f14` (feat)
3. **Task 3: The committed logrotate example** - `e42ec75` (feat)

**Plan metadata:** (this commit)

_Note: all three tasks carried `tdd="true"`/`type="auto"` per the plan; `workflow.tdd_mode` is `false` for this project (precedent: plans 34-07, 36-01, 36-02, 36-03), so each task's tests and implementation landed in a single commit rather than a split RED/GREEN pair._

## Files Created/Modified
- `solsys_code/management/commands/check_unattended.py` - the preflight command: `CheckResult`, six check functions, `cron_line()`, `_send_test_email()`, `Command`
- `solsys_code/tests/test_check_unattended.py` - 17 tests covering every case in the plan's `<behavior>` lists
- `deploy/logrotate/fomo.example` - committed logrotate drop-in for `/var/log/fomo/unattended.log`

## Decisions Made
See `key-decisions` in frontmatter: `check_email()`'s two-result return shape, `cron_line()`/`_send_test_email()` as non-check helpers kept out of the six-callable verify probe, and the `0o500`-directory technique used to exercise the unwritable-parent branch of `_check_directory_writable()` without depending on a real root-owned path.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' `<behavior>` test lists, `<action>` implementation steps, `<verify>` commands, and `<acceptance_criteria>` were satisfied without needing a Rule 1-4 deviation.

## Issues Encountered

`workflow.tdd_mode` is `false` for this project (per this plan's own project note and the precedent set by plans 34-07, 36-01, 36-02, 36-03), so `gsd_run check tdd-red-evidence` was not run. RED was verified manually for each task: running `python manage.py test solsys_code.tests.test_check_unattended` against the test file alone (before `check_unattended.py` existed, for Task 1; before `cron_line()`/`--send-test-email` existed, for Task 2) produced the expected `ImportError`/`AttributeError`/`CommandError: unrecognized arguments` failures, confirmed against the exact tests this plan added, before the corresponding implementation was written.

The plan's own environment note (also true for 36-01/36-02/36-03) holds here too: `python manage.py check_unattended` on this developer host genuinely fails its `FOMO_LOCK_DIR`/`FOMO_LOG_FILE`/`EMAIL_BACKEND` hard checks (the project defaults point at root-owned `/var/lock/fomo` and `/var/log/fomo`, and `EMAIL_BACKEND` is the console default) -- this is the acceptance criterion working as designed ("proving the hard-failure path is real, not decorative"), not a defect. `flock` itself is present at `/usr/bin/flock` on this host, so that one check passes.

## User Setup Required

None - no external service configuration required. (An operator sets up the real host's lock/log directories and email backend using this command's own output in a later plan's runbook section.)

## Next Phase Readiness

Ready for Plan 05 (the paired runbook section documenting `check_unattended` and the logrotate/crontab setup walkthrough, per CLAUDE.md's paired-docs rule and `36-02-SUMMARY.md`'s own forward pointer to `36-05-PLAN.md`). No blockers: `check_unattended.py` and `deploy/logrotate/fomo.example` are both committed, tested, and pass the full quality gate.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All key files found on disk (`solsys_code/management/commands/check_unattended.py` contains
`class Command(BaseCommand):`, `@dataclass`, `def check_flock(`, `def check_lock_dir(`,
`def check_log_dir(`, `def check_email(`, `def check_heartbeat(`, `def check_watched_proposals(`,
`def cron_line(`, `'--send-test-email'`; `solsys_code/tests/test_check_unattended.py`;
`deploy/logrotate/fomo.example` tracked by git). All 3 task commit hashes (`fc2e2bc`, `d8d7f14`,
`e42ec75`) found in `git log`. Full test module rerun (`python manage.py test
solsys_code.tests.test_check_unattended` -- 17 tests) and the plan's own `<verification>` block
(cron-probe script, `check_unattended` real-host run exiting 1 with the expected named failures,
`git ls-files -- deploy/` listing both example files) all pass. The project's full non-segfaulting
test suite (`solsys_code.tests.*` plus `solsys_code_observatory.tests.*`, excluding
`test_views.TestEphemeris`) reruns clean (exit code 0). Both quality gates
(`pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`) pass.
