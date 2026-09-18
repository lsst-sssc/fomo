---
phase: 36-unattended-operation
fixed_at: 2026-09-18T16:29:00Z
review_path: .planning/phases/36-unattended-operation/36-REVIEW.md
iteration: 1
findings_in_scope: 16
fixed: 16
skipped: 0
status: all_fixed
---

# Phase 36: Code Review Fix Report

**Fixed at:** 2026-09-18T16:29:00Z
**Source review:** .planning/phases/36-unattended-operation/36-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 16 (2 critical, 6 warning, 8 info -- `fix_scope: all`)
- Fixed: 16
- Skipped: 0

All fixes were applied and committed inside an isolated git worktree
(`gsd-reviewfix/36-1858218`, branched from `issue37-telescope-runs-calendar`),
then fast-forwarded back onto `issue37-telescope-runs-calendar`. Each commit
below was verified with `python -c "import ast; ast.parse(...)"` (Tier 2 syntax
check) followed by the phase's own targeted test command:
`python manage.py test solsys_code.tests.test_unattended
solsys_code.tests.test_check_unattended solsys_code.tests.test_settings_api_key_fold
solsys_code.tests.test_backfill_lco_observations`, and
`pre-commit run ruff --all-files` / `pre-commit run ruff-format --all-files`
before every commit (pre-commit's own hooks -- including a full Sphinx doc
build and the pytest suite -- also ran and passed on every commit). The final
targeted-suite count is 170 tests, OK (up from 160 at the start, reflecting
the new regression tests this run added).

## Fixed Issues

### CR-05: WR-17's fallback state file is a predictable, world-writable path whose mtime alone decides which suppression state the runner trusts

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`, `solsys_code/tests/test_check_unattended.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `14ae0bf`
**Applied fix:** Replaced the module-level `_FALLBACK_STATE_PATH` constant with
`_fallback_state_path()`, a function returning a path scoped to this deployment
(a hash of `settings.BASE_DIR`) and to this process's own uid. Added
`_fallback_is_trustworthy()` (lstat-based: regular file, owned by this euid, no
group/other permission bits) that `_newest_existing_state_path()` now requires
before considering the fallback candidate at all -- an untrusted fallback file
is ignored regardless of how new its mtime is. Updated the runbook's "How do I
run everything unattended?" setup section and its fallback-troubleshooting
entry to describe the scoped filename and the trust check, replacing the
"mode 0600 ... not a new exposure" claim the review flagged as overstating
what 0600 alone covers.
**Committed together with WR-33 and WR-34** (see their entries below for why):
WR-33's fix is identical to CR-05's (the review's own fix note says "as in
CR-05"), and WR-34's test fix is not separable from CR-05's source change --
removing the `_FALLBACK_STATE_PATH` constant would otherwise break
`test_unwritable_state_dir_after_setup_still_suppresses_repeat_mail` in the gap
between two separate commits.

### WR-33: the fallback state path has no per-deployment discriminator, so two FOMO instances on one host silently share one suppression state

**Files modified:** `solsys_code/unattended.py` (same commit as CR-05)
**Commit:** `14ae0bf`
**Applied fix:** Resolved by the same `_fallback_state_path()` change as CR-05 --
the BASE_DIR hash + uid scoping is exactly the per-deployment discriminator this
finding asked for.

### WR-34: the new WR-17 test reads, writes and deletes the real `/tmp` fallback path instead of an isolated one

**Files modified:** `solsys_code/tests/test_unattended.py`, `solsys_code/tests/test_check_unattended.py` (same commit as CR-05)
**Commit:** `14ae0bf`
**Applied fix:** `test_unwritable_state_dir_after_setup_still_suppresses_repeat_mail`
now patches `unattended._fallback_state_path` to return a path inside a
`TemporaryDirectory()` instead of touching the real `/tmp` constant (made
possible by CR-05 turning the fallback location into a function). Added
`@skipIf(os.geteuid() == 0, ...)` to this test and to
`_make_unwritable_parent`'s three callers in `test_check_unattended.py`
(`test_unwritable_lock_dir_fails`, `test_unwritable_log_dir_fails`,
`test_unwritable_state_dir_fails`), per the finding's explicit instruction.

### CR-06: a state file with a valid `failing_steps` but an unparseable `notified_at` permanently suppresses both the failure mail and the 24-hour reminder

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `5553724`
**Applied fix:** Implemented the review's "belt-and-braces" suggestion, both
halves: `load_state()` now discards the whole record (not just `notified_at`)
on a parse failure, honoring its own docstring's promise, and logs a warning;
`decide_notification()` additionally treats a matching failing set with
`notified_at=None` as due for notification now, as an independent second
guard. Added one test per half: a hand-written state file with an unparseable
`notified_at` must still mail on the next tick (end-to-end via
`call_command('run_unattended')`), and `decide_notification()` must return
`'failure'` (not `None`) for a matching failing set with `notified_at=None`
(direct unit test). Updated
`test_malformed_notified_at_is_treated_as_no_prior_notification`, which had
pinned the old partial-preservation behavior as correct.
**Note on scope:** the review's discussion of CR-05 also illustrates a
*separate*, narrower scenario -- a state file with a **valid but far-future**
`notified_at` (e.g. `2099-01-01`) suppressing the reminder indefinitely. That
is not the "unparseable" case CR-06's docstring promise covers, and is closed
in practice by CR-05's trust check (an attacker can no longer plant such a
value into the trusted fallback path); it was not separately re-implemented
here since it was not filed as its own finding ID and the review's own
"Fix" code sketch for CR-06 does not address it.

### WR-35: `check_flock()`'s IN-23 "confirm this is the flock you want" result is reported as `[ok]` and never reaches stderr, so the warning is invisible

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `8a4d2bc`
**Applied fix:** Changed the branch's `CheckResult` to `ok=False, hard=False`,
matching every other advisory result in this command. Renders as
`[WARN] flock: ...` and now reaches stderr. Updated the pinning test to assert
`[WARN] flock` in both stdout and stderr.

### WR-36: the 15-minute tick interval was re-duplicated in the same iteration that introduced the single-owner rule for shared constants

**Files modified:** `solsys_code/unattended.py`, `solsys_code/management/commands/check_unattended.py`
**Commit:** `8da8058`
**Applied fix:** Moved `_CRON_INTERVAL_MINUTES = 15` into `unattended.py` next
to the other single-owner defaults, and derived `_CRON_TICK_INTERVAL` from it.
`check_unattended.py` now imports the minutes constant instead of redeclaring
it. (Superseded in placement, not in effect, by IN-34 below -- see that entry.)

### WR-37: the cron-line/template drift guard does not cover the `rc=0` normalization that WR-16 just added

**Files modified:** `solsys_code/tests/test_check_unattended.py`
**Commit:** `8e6b39e`
**Applied fix:** Added `'rc=0'` and `'; }'` to the token comparison list in
`test_line_matches_the_committed_template_token_for_token`. Extracted the
0/1/99 stub-splice matrix into a shared helper
(`_assert_lock_held_exit_matrix_is_normalized`) and added a new test,
`test_committed_template_lock_held_exit_is_normalized_to_zero`, that runs the
same matrix against the committed template's own `*/15` line read from disk,
not just `cron_line()`'s generated output.

### WR-38: the WR-32 settings import guard is executed by no test

**Files modified:** `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `367f99f`
**Applied fix:** Added `_MissingModuleFinder`, a `sys.meta_path` finder that
raises a caller-supplied exception directly from `find_spec()` for exactly
`'fomo.local_settings'` -- propagating that exact exception (and its `.name`)
through the import machinery untouched, independent of what exists on disk.
Two new cases exec the fold tail with `sys.modules['fomo.local_settings']`
removed and this finder installed: (a) the module itself "missing"
(`name='fomo.local_settings'`) must be swallowed, `FACILITIES` untouched;
(b) some other import inside `fomo.local_settings` missing
(`name='some_missing_dependency'`) must propagate out of the exec.

### IN-33: `test_settings_api_key_fold.py`'s module docstring still advertises the test class WR-30 deleted

**Files modified:** `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `feb5ae9`
**Applied fix:** Rewrote the module docstring to enumerate the test classes
actually present (seven at the time of this commit; an eighth, added by the
subsequent IN-37 commit, is also named -- see that entry), naming each
class's own mechanism instead of the deleted `NameError` case.

### IN-34: the read-only preflight now imports the entire runner graph for two string constants

**Files modified:** `solsys_code/constants.py` (new), `solsys_code/unattended.py`, `solsys_code/management/commands/check_unattended.py`
**Commit:** `8d7b3d1`
**Applied fix:** Created `solsys_code/constants.py`, a leaf module with no
project-local imports, owning `DEFAULT_LOCK_DIR`, `DEFAULT_LOG_FILE`, and
`CRON_INTERVAL_MINUTES`. Both `unattended.py` and `check_unattended.py` now
import from this leaf module (via `as` aliases, so every existing internal
reference and every existing test patch target is unchanged) instead of one
importing from the other -- `check_unattended.py` no longer imports
`solsys_code.unattended` at all, verified empirically
(`'solsys_code.unattended' in sys.modules` is `False` after importing
`check_unattended` alone).

### IN-35: `_NON_DELIVERING_EMAIL_BACKENDS` is an exact dotted-path denylist that any subclass evades

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `74fe17b`
**Applied fix:** Added `_classify_email_backend()`, which resolves
`EMAIL_BACKEND` with `django.utils.module_loading.import_string()` and checks
`issubclass()` against the four non-delivering classes, falling back to the
dotted-path comparison only if the class cannot be imported. Rewrote the test
fixture (`_FakeDeliveringEmailBackend`) as a thin, direct subclass of
`BaseEmailBackend` (not `locmem`, which would now correctly fail its own
check) replicating just enough of `locmem`'s `send_messages()` to keep
populating `mail.outbox`. Added a test proving the evasion is closed: a bare
`locmem` subclass with no overrides must still fail `EMAIL_BACKEND`.

### IN-36: `_SYSTEM_BINARY_DIRECTORIES` omits `/usr/local/bin`, and the path is compared unresolved

**Files modified:** `solsys_code/management/commands/check_unattended.py`, `solsys_code/tests/test_check_unattended.py`
**Commit:** `05f03fb`
**Applied fix:** Added `/usr/local/bin` and `/usr/local/sbin` to
`_SYSTEM_BINARY_DIRECTORIES`. Compare `str(Path(path).resolve())` instead of
the raw `which()` result, with the resolved target named in the detail text
when it differs. Added two tests: `/usr/local/bin/flock` gets no sanity note;
a real filesystem symlink at a trusted path resolving outside every system
directory gets the `[WARN]` note (uses an actual symlink, not a mocked
`Path.resolve`, to keep the test exercising real filesystem resolution).

### IN-37: the settings fold leaves its loop variable bound in the settings module namespace

**Files modified:** `src/fomo/settings.py`, `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `3f1d493`
**Applied fix:** Added `del _facility` after the loop (inside the same
`if 'LCO_API_KEY' in globals():` guard the loop itself is under, since the
binding never exists otherwise). Added a regression test asserting directly
against the live, already-imported settings module -- this checkout's real
`src/fomo/local_settings.py` sets `LCO_API_KEY`, so the fold loop runs on
every settings import here, giving a real (not synthetic) case to assert
against.

### IN-38: `step_discovery()` logs the whole captured sweep buffer as one INFO record

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `a3da325`
**Applied fix:** `for line in captured_text.splitlines(): logger.info('discovery %s: %s', sink_name, line)`,
exactly as the review's fix sketch specified. Added a test with a two-line
captured buffer, asserting exactly two `discovery stdout` log records are
emitted (one per line).

### IN-39: the runbook's description of the heartbeat reminder still names only the ping interval

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `5ad7cc5`
**Applied fix:** Changed "...still needs its own expected ping interval set"
to "...still needs its own expected ping interval *and* grace time set",
exactly as the review's fix specified.

### IN-40: the template-parity test locates the template line by a hardcoded `*/15` while the code's schedule is now a constant

**Files modified:** `solsys_code/tests/test_check_unattended.py`
**Commit:** `27c722a`
**Applied fix:** Added module-level `_TEMPLATE_SCHEDULE_PREFIX = f'*/{_CRON_INTERVAL_MINUTES}'`
(imported from `check_unattended` itself) and used it everywhere a `'*/15'`
literal previously located or asserted the schedule field: both template-line
locators in `TestCronLine`, `test_line_matches_the_committed_template_shape`'s
assertion, and as a new token in
`test_line_matches_the_committed_template_token_for_token`'s comparison list.

## Skipped Issues

None -- all 16 in-scope findings were fixed.

---

_Fixed: 2026-09-18T16:29:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
