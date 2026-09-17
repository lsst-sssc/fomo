---
phase: 36-unattended-operation
reviewed: 2026-09-17T21:05:00Z
depth: deep
iteration: 2
files_reviewed: 12
files_reviewed_list:
  - deploy/cron/fomo.crontab.example
  - deploy/logrotate/fomo.example
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/notifications.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_check_unattended.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/unattended.py
findings:
  critical: 0
  warning: 7
  info: 8
  total: 15
carried_forward_open: 6
status: issues_found
---

# Phase 36: Code Review Report (iteration 2 — re-review after the fix pass)

**Reviewed:** 2026-09-17T21:05:00Z
**Depth:** deep
**Files Reviewed:** 12 (everything changed since `337b1e9`, the commit the first review was written against)
**Status:** issues_found

## Summary

This is an incremental re-review of the ten fixes recorded in `36-REVIEW-FIX.md`
(CR-01, CR-02, WR-01..WR-08), plus an adversarial pass over everything those fixes
touched. **All ten prior findings are genuinely fixed** — I verified the two critical ones
against the real code path rather than against the fix report's prose:

- **CR-01 (self-deadlocking cron line) is empirically resolved.** Running the real
  `command_lock('run_unattended')` under `flock -n -E 99 <FOMO_LOCK_DIR>/run_unattended.cron.lock`
  now acquires the internal lock (`RESULT: internal lock ACQUIRED -- tick would run`), while
  the same probe under the *old* shared filename still reproduces `LockContended`. The two
  names are distinct in `cron_line()` (`check_unattended.py:240`), in the committed template
  (`fomo.crontab.example:38`), and in the runbook, and `test_cron_lock_differs_from_the_runner_internal_lock`
  pins the difference.
- **CR-02 is resolved** by a fail-closed `CommandError` in `handle()`
  (`backfill_lco_observations.py:836-850`) with five new tests, and the paired demo notebook
  carries a note.
- All 146 tests in the four affected modules pass (`test_unattended`, `test_check_unattended`,
  `test_campaign_submission`, `test_backfill_lco_observations`), and
  `pre-commit run ruff`/`ruff-format` are clean on every changed Python file.

What the fix pass did **not** get right is concentrated in two places: the cron line's exit
status, which the WR-01 fix inverted, and the state file, where WR-03's "never raises"
contract is still not true.

**Two findings are regressions or residuals of the fixes themselves, and should be treated
as the priority:**

- **WR-09** — the new `[ $? -eq 99 ] && echo ...` tail makes the crontab line exit **1 on a
  healthy tick, 1 on a failing tick, and 0 on the one case where nothing ran**. Verified in a
  real `sh`. The cron line's own status is now precisely inverted.
- **WR-10** — `load_state()` still raises `TypeError` on a state file whose `failing_steps`
  mixes types (verified: `{"failing_steps": [1, "a"]}` → `TypeError: '<' not supported
  between instances of 'str' and 'int'`). The new outer `try/except` keeps the tick alive,
  but `save_state()` is never reached, so the file is never repaired and **every failure
  email is silently suppressed forever** from that point on.

No security defects were found in the changed code. Credential hygiene (D-15/D-17/SCHED-10)
holds throughout: `check_base_url()` never echoes the URL value, `cron_line()` interpolates
only paths/uids, and every caught exception is still reported by class name only.

## Narrative Findings (AI reviewer)

### Verification of the prior ten fixes

| Prior finding | Verdict | Evidence |
|---|---|---|
| CR-01 self-deadlocking cron lock | **Fixed** | Live `flock`/`command_lock()` probe; `check_unattended.py:240`, `fomo.crontab.example:38`, `test_check_unattended.py:236-242` |
| CR-02 silently-ignored `--proposal`-only flags | **Fixed** | `backfill_lco_observations.py:836-850`; 5 new tests |
| WR-01 `\|\|` tail mislabels failures | **Fixed** (but see WR-09) | `-E 99` + `[ $? -eq 99 ]` verified in `sh` |
| WR-02 unsent mail recorded as notified | **Fixed** (see IN-07 for a docstring overstatement) | `unattended.py:503-522`, `:587-591` |
| WR-03 state file not fail-safe | **Partly fixed** — see **WR-10**, **WR-15** | `unattended.py:373-408`, `:579-593` |
| WR-04 END banner reuses START time | **Fixed** | `unattended.py:569`, `:596` |
| WR-05 hardcoded `/usr/bin/flock` | **Fixed** | `check_unattended.py:246` |
| WR-06 writability preflight claim | **Fixed** (reporting-only remedy, doc corrected) | `check_unattended.py:73-121`, `fomo.example:6-9` |
| WR-07 localhost base URL in links | **Fixed** (see **WR-13** for the remaining hole) | `check_unattended.py:188-211`, `notifications.py:46` |
| WR-08 uncapped status re-check | **Fixed** | `unattended.py:56`, `:207`, `:218`, `:254-259` |

## Warnings

### WR-09: The new cron line's exit status is inverted — 0 only when nothing ran, 1 on both success and failure

**File:** `deploy/cron/fomo.crontab.example:38`, `solsys_code/management/commands/check_unattended.py:251-255`
**Issue:** The WR-01 fix replaced `... || echo ...` with `...; [ $? -eq 99 ] && echo ...`.
The `[ ... ] && echo ...` list is now the *last* command in the crontab line, so its status
is the line's status. Verified in a real `sh`:

```
$ sh -c "flock -n -E 99 $D/l true      >> $D/log 2>&1; [ \$? -eq 99 ] && echo skip >> $D/log"; echo $?
1                       # healthy tick  -> line exits 1
$ sh -c "flock -n -E 99 $D/l sh -c 'exit 1' >> $D/log 2>&1; [ \$? -eq 99 ] && echo skip >> $D/log"; echo $?
1                       # failing tick  -> line exits 1
$ sh -c "flock -n -E 99 $D/l true      >> $D/log 2>&1; [ \$? -eq 99 ] && echo skip >> $D/log"; echo $?   # lock held
0                       # skipped tick  -> line exits 0
```

So the only outcome that reports *success* to cron is the one where zero steps ran, and a
healthy tick is indistinguishable from a failed one. Anything that reads the cron job's
status — `cron`'s own syslog `CMD exit status`, a systemd timer if this is ever migrated, a
wrapper script, an `OnFailure=` hook, or a `run-parts`-style harness — now gets exactly the
wrong answer. The pre-fix `||` form at least returned 0 on a healthy tick. Neither the
crontab template's comment block (`:20-30`) nor the runbook mentions that the line's status
is no longer the tick's status, so the next person to wire monitoring onto it will be misled
in the same direction CR-01 misled the operator.
**Fix:** capture and re-emit `run_unattended`'s own status, in `cron_line()` and in the
committed template together:

```
*/15 * * * * /usr/bin/flock -n -E 99 /var/lock/fomo/run_unattended.cron.lock <python> <manage.py> run_unattended >> /var/log/fomo/unattended.log 2>&1; rc=$?; [ $rc -eq 99 ] && echo "$(date -Is) run_unattended skipped: lock held" >> /var/log/fomo/unattended.log; exit $rc
```

and add a `TestCronLine` case asserting the line ends with an explicit `exit $rc` (or
equivalent) rather than with a bare `[ ... ] && ...`.

### WR-10: `load_state()` still raises, and a corrupt state file silently disables every failure email forever

**File:** `solsys_code/unattended.py:393-408` (specifically `:406`), `:579-593`
**Issue:** WR-03's fix added `isinstance(data, dict)` and `isinstance(failing_steps, list)`
guards but never validates the list's *elements*, and `sorted(failing_steps)` at `:406` sits
outside any `try`. Executed against the real function with a temporary `FOMO_STATE_DIR`:

```
'{"failing_steps": [1, "a"], "notified_at": null}' RAISED TypeError '<' not supported between instances of 'str' and 'int'
'{"failing_steps": [true, "a"]}'                   RAISED TypeError '<' not supported between instances of 'str' and 'bool'
'{"failing_steps": {"a": 1}}'                      -> {'failing_steps': [], 'notified_at': None}   # ok
'null'                                             -> {'failing_steps': [], 'notified_at': None}   # ok
```

`load_state()`'s own docstring (`:379-382`) still promises "A missing file, an unparseable
one, one whose top level is not a dict/list ... is treated as 'no prior failure' -- never an
exception out of `run_tick()`". It is not true. The new outer `try/except` at `:579-593`
does keep the tick alive — but that is exactly what makes this dangerous: the exception is
raised on the **first** statement of the block, so `decide_notification()`,
`_send_notification()` and `save_state()` are all skipped. `save_state()` is the only writer
of that file, so the corrupt content is never overwritten, and the condition repeats on
every tick forever. Net effect: D-11/SCHED-09's primary alert channel is permanently and
silently off, leaving only one `logger.error('unattended notification/state handling raised:
TypeError')` line per tick in a log nobody reads until something already looks wrong.
`FOMO_STATE_DIR` defaults to `FOMO_LOCK_DIR` (`settings.py:418`), a directory the preflight
checks only for writability, so a hand-edit or a foreign writer is the realistic trigger —
the same trigger WR-03 itself cited.
**Fix:** coerce the elements and make the function total, as its docstring already claims:

```python
    failing_steps = data.get('failing_steps')
    if not isinstance(failing_steps, list):
        failing_steps = []
    failing_steps = [step for step in failing_steps if isinstance(step, str)]
```

and add a `TestStateFileRobustness` case for `{"failing_steps": [1, "a"]}` asserting
`load_state()` returns a clean default instead of raising. Consider also having the
`except Exception` handler at `:592` call `save_state([], None)` so a corrupt file
self-heals on the next tick rather than wedging the notification path permanently.

### WR-11: `check_flock()` verifies the binary exists but not the `-E` option the new cron line depends on

**File:** `solsys_code/management/commands/check_unattended.py:54-64`, `:252`
**Issue:** `cron_line()` now emits `flock -n -E 99`, and the whole WR-01 skip-detection
scheme depends on that flag. `-E/--conflict-exit-code` was added in util-linux 2.27 (2015);
`check_flock()` only calls `shutil.which('flock')` and reports `[ok] flock: found at <path>`.
On a host with an older util-linux (RHEL/CentOS 7 ships 2.23, still a live deployment
target), the preflight passes, the operator installs the printed line, and **every tick is a
no-op** — `flock` rejects the unknown option and exits before ever starting Python, while
`[ $? -eq 99 ]` is false so no skip line is written either. This is the CR-01 failure class
reproduced through a different door; it is only less severe because `flock: invalid option
-- 'E'` does land in the redirected log. WR-05's fix makes this *more* likely to bite, since
`cron_line()` will now happily emit a resolved path to a venv- or container-provided
`flock` of unknown vintage.
**Fix:** make the check prove the option, not just the binary:

```python
def check_flock() -> CheckResult:
    path = shutil.which('flock')
    if path is None:
        return CheckResult(name='flock', ok=False, hard=True, detail='not found on PATH -- install util-linux')
    probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False)  # noqa: S603
    if '--conflict-exit-code' not in (probe.stdout + probe.stderr):
        return CheckResult(
            name='flock',
            ok=False,
            hard=True,
            detail=f'{path} does not support -E/--conflict-exit-code (util-linux < 2.27) -- '
            'the cron line below needs it to distinguish a skipped tick from a failed one',
        )
    return CheckResult(name='flock', ok=True, hard=True, detail=f'found at {path}, supports -E')
```

### WR-12: The runbook still documents `backfill_lco_observations`'s window/attribution flags as freely optional, which CR-02's fix made untrue

**File:** `docs/runbooks/telescope_runs_calendar.rst:384-402` and `:1733-1736`; cf.
`solsys_code/management/commands/backfill_lco_observations.py:836-850`
**Issue:** CR-02's fix changed the command's CLI contract — `--created-after`,
`--created-before`, `--username` and `--target-list` now raise `CommandError` when
`--proposal` is omitted. The fixer updated the four `--help` strings, the `handle()`
docstring and the paired notebook, but not the runbook, which is the page an operator
actually reads:

- `:384-386` shows the bare invocation `python3 manage.py backfill_lco_observations` and then
  `:388-402` immediately describes `--created-after`/`--created-before`, `--username <user>`
  and `--target-list <NAME>` with no mention that any of them now requires `--proposal`.
- The command cheat-sheet row at `:1733-1736` reads "``--proposal <code>`` (optional -- omit
  to sweep every active Watched proposal row), ``--created-after``/``--created-before``,
  ``--username <user>``, ``--target-list <name>``, ``--dry-run`` (**all optional**)" —
  which now describes four combinations that abort with an error.

CLAUDE.md makes any `docs/runbooks/` page whose documented behavior a change affects part of
the deliverable, not follow-up polish; this is the same class of miss the file records for
quick task `260726-kdp`.
**Fix:** add one sentence after `:386` ("These four flags apply to the single-proposal
override only; combined with the bare invocation the command now fails with a
``CommandError`` rather than silently discarding them, because the watched-list sweep takes
its overrides from each ``WatchedProposal`` row"), and change the cheat-sheet row's "(all
optional)" to "(optional; the four non-``--dry-run`` flags require ``--proposal``)".

### WR-13: `FOMO_BASE_URL` is now required by the *web* process too, but is documented and preflighted only for cron

**File:** `docs/runbooks/telescope_runs_calendar.rst:1456-1458`,
`deploy/cron/fomo.crontab.example:16`, `solsys_code/management/commands/check_unattended.py:188-211`,
`solsys_code/campaign_views.py:338`
**Issue:** WR-07's fix added `check_base_url()` and a `None` guard, which is right as far as
it goes — but it leaves the deployment instruction wrong. The campaign-submission approval
queue link (`campaign_views._notify_staff` → `notifications.absolute_url()`) is built inside
the **WSGI/web** process, which reads its own environment at settings-import time
(`settings.py:409`). Every place the setting is documented scopes it to cron only: the
runbook's step 3 says "Export ``FOMO_HEARTBEAT_URL`` and ``FOMO_BASE_URL`` in the environment
**the cron daemon sees**", and the crontab template lists it under "The environment the cron
daemon must supply". `grep -rn FOMO_BASE_URL docs/ deploy/` finds no mention anywhere that
the web process needs it, and `docs/installation.rst` does not mention it at all. Worse,
WR-06's fix now tells the operator to run `check_unattended` **as the cron account** — so
the one check that would catch a default base URL is deliberately run in the process whose
environment is least likely to match the web server's. A deployment that exports
`FOMO_BASE_URL` for cron and not for gunicorn gets a green preflight and unusable
`http://localhost:8000/campaigns/...` links in every SUBMIT-05 notice.
**Fix:** reword runbook step 3 to "``FOMO_HEARTBEAT_URL`` in the cron environment;
``FOMO_BASE_URL`` in **both** the cron environment and the web server's (gunicorn/uWSGI)
environment — or, simpler, set it once in this host's ``local_settings.py`` so every process
picks it up", mirror the note in `deploy/cron/fomo.crontab.example:16`, and add it to
`docs/installation.rst`'s settings list.

### WR-14: The runbook's "remove the stale lock file" remedy describes a condition that cannot occur

**File:** `docs/runbooks/telescope_runs_calendar.rst:2000-2008`
**Issue:** The rewritten troubleshooting section now says: "if it has genuinely died without
releasing the lock file, remove the stale ``run_unattended.cron.lock`` file under
``FOMO_LOCK_DIR``". `flock(2)` locks are released by the kernel when the holding file
descriptor is closed, which happens unconditionally on process exit — including `SIGKILL`,
an OOM kill, and a crash. A dead process therefore *never* leaves a held lock, so the
documented condition is unreachable and the remedy can only ever be applied while a tick is
genuinely still running. Doing so unlinks the file the live `flock` holds; the next cron
invocation creates a *new* inode and takes its lock, so the cron guard is defeated (the
runner's own internal `run_unattended.lock` is what actually still prevents two overlapping
ticks — the cron guard silently stops contributing). CR-01 was in large part a finding about
the runbook teaching a wrong diagnosis; this paragraph, rewritten by the CR-01/WR-01 fix,
teaches another one.
**Fix:** replace the parenthetical with the truth — "a `flock` is always released when the
holding process exits, so a repeated skip line always means a tick is *still running*; find
it with `pgrep -af run_unattended` and investigate why it is stuck. Deleting the lock file
does not help and, while a tick is live, disables the cron guard until the next tick." Keep
the heartbeat-grace-period sentence as-is.

### WR-15: An unwritable `FOMO_STATE_DIR` now mails every staff user every 15 minutes, indefinitely

**File:** `solsys_code/unattended.py:579-593`, `:411-428`
**Issue:** WR-03's fix correctly stopped a `save_state()` failure from killing the END banner
and the exit-code ping — but it left the ordering that makes the failure self-amplifying.
With `FOMO_STATE_DIR` unwritable or full, each tick: `load_state()` returns the default (the
`OSError` is caught at `:388`) → `decide_notification()` sees an empty previous set and a
non-empty current one → `'failure'` → `_send_notification()` **sends** → `save_state()`
raises → the new handler logs `OSError` and moves on. Nothing records that staff were told,
so the identical email goes out on the next tick, and the next: 96 messages per staff address
per day for as long as one step keeps failing, which is the D-11 suppression rule failing
open in the most visible possible way. `check_unattended` checks `FOMO_LOCK_DIR` and the log
directory but never `FOMO_STATE_DIR` (`check_unattended.py:334-341`), even though it defaults
to `FOMO_LOCK_DIR` and can be pointed elsewhere.
**Fix:** two cheap mitigations, either of which closes it: (a) make persistence failure
suppress further mail by treating an unpersistable state as "already notified" for the
remainder of the process and logging one `logger.error` naming the state path; or (b) add a
`check_state_dir()` hard check to the preflight mirroring `check_lock_dir()`, so the
condition is caught at setup. (a) is the one that survives a directory that becomes
unwritable after setup.

## Info

### IN-07: `_send_notification()` documents "attempted *and* delivered", but `notify_staff()` discards `send_mail()`'s result

**File:** `solsys_code/unattended.py:510-515`, `solsys_code/notifications.py:79-90`
**Issue:** `notify_staff()` calls `send_mail(...)` and ignores its return value (the number
of messages actually sent), returning `True` purely because the recipient list was non-empty
and nothing raised. Its own docstring is honest about this ("True if there was at least one
recipient (an attempt was made, whether or not it succeeded)"), but `_send_notification()`'s
docstring promotes it to "True only when the mail was actually attempted *and* delivered",
and WR-02's whole suppression decision now rests on that stronger claim. With
`fail_silently=False` the gap is small in practice (Django's backends raise rather than
return 0), but the two docstrings should not disagree about the same boolean.
**Fix:** `return bool(send_mail(...))` in `notify_staff()` (keeping the `fail_silently`
semantics), or soften `_send_notification()`'s docstring to "attempted without raising".

### IN-08: The cron-line "committed template shape" test no longer pins the committed template, and breaks on a host without `flock`

**File:** `solsys_code/tests/test_check_unattended.py:196-210`
**Issue:** `test_line_matches_the_committed_template_shape` now asserts
`f'{shutil.which("flock")} -n -E 99'`. On a host with no `flock`, `shutil.which` returns
`None` and the assertion becomes the literal `'None -n -E 99'`, which can never match
`cron_line()`'s `/usr/bin/flock` fallback — the test fails for a reason unrelated to what it
checks. Separately, the test's name promises agreement with
`deploy/cron/fomo.crontab.example`, but it compares against a hand-maintained list of
fragments and never reads the committed file, which still hardcodes `/usr/bin/flock`
(`:38`). Drift between the two is exactly what CR-01/WR-01 were about.
**Fix:** guard with `flock_path = shutil.which('flock') or '/usr/bin/flock'`, and have the
test read `deploy/cron/fomo.crontab.example`'s `*/15` line and compare the option set
(`-n`, `-E 99`, `.cron.lock`, `>>`, `[ $? -eq 99 ]`) token by token.

### IN-09: `check_unattended`'s docstrings drifted from the code the fixes added

**File:** `solsys_code/management/commands/check_unattended.py:13-17`, `:300-305`;
`solsys_code/tests/test_check_unattended.py:3`
**Issue:** Three stale statements, all introduced by the WR-06/WR-07 fixes:
(1) the module docstring's "The only values ever interpolated into this command's output are
filesystem paths and ``sys.executable``" is no longer true — `_check_directory_writable()`
now interpolates `os.geteuid()`, the owner uid and the octal mode; (2) the `Command` class
docstring still says "an unset heartbeat URL and an empty watched-proposal list are warnings",
omitting the new `FOMO_BASE_URL` warning that the module docstring and `help` string both
list; (3) the test module docstring says "the six prerequisite checks" — there are now seven.
**Fix:** extend (1) to "paths, `sys.executable`, and filesystem ownership/permission
metadata", add `FOMO_BASE_URL` to (2), and change "six" to "seven" in (3).

### IN-10: Two runbook passages still describe the pre-fix `flock -n` cron guard

**File:** `docs/runbooks/telescope_runs_calendar.rst:1552-1562`, `:1575-1578`
**Issue:** The troubleshooting section at `:1985-2008` was updated to `flock -n -E 99` and
`run_unattended.cron.lock`, but the "When nothing has appeared" checklist item 4 (`:1552`)
and the "What the locking does and does not cover" paragraph (`:1576`) still say plain
``flock -n`` and never name which of the two lock files they mean. An operator following the
checklist reaches the *un*updated description first. The same passage also cannot help the
reader distinguish the cron tail's `... run_unattended skipped: lock held` from the runner's
own internal-lock message `run_unattended: lock held -- skipping this tick`
(`unattended.py:599-600`), which lands in the same log with the same phrase but at exit 0.
**Fix:** propagate `-E 99` / `run_unattended.cron.lock` into both passages and add one
sentence distinguishing the two "lock held" strings by their prefix.

### IN-11: The paired notebook got prose only — no executed cell exercising CR-02's new `CommandError`

**File:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (bare-invocation
markdown cell)
**Issue:** CLAUDE.md's paired-docs rule asks for cells or prose "exercising the new behavior
**with real executed output**". The fix added a markdown note and explicitly skipped
re-execution. The notebook therefore still demonstrates only the happy path; a reader cannot
see what the rejection actually looks like, and nothing in the committed outputs would catch a
future regression that silently dropped the guard.
**Fix:** add one short cell that calls
`call_command('backfill_lco_observations', '--created-after=2026-01-01')` inside a
`try/except CommandError` and prints the message, then regenerate with
`jupyter nbconvert --to notebook --execute --inplace`.

### IN-12: The `--proposal`-only guard runs after username resolution and uses truthiness

**File:** `solsys_code/management/commands/backfill_lco_observations.py:809-814`, `:836-845`
**Issue:** Two small rough edges in CR-02's otherwise-correct guard. (1) `--username` is
resolved to a `User` (a DB query, and a `CommandError` on an unknown name) at `:810-814`,
*before* the `ignored` check at `:846`, so
`backfill_lco_observations --username ghost` reports `Invalid username: 'ghost'` rather than
the more useful "``--username`` requires ``--proposal``". (2) the guard tests
`options.get(key)` for truthiness, so `--target-list ''` or `--created-after ''` — flags that
*were* given — slip past it. Neither causes harm today (an empty string is falsy everywhere
downstream), but the guard's intent is "the flag was supplied", which is `is not None`.
**Fix:** move the `ignored` block above the username resolution, and switch the predicate to
`options.get(key) is not None`.

### IN-13: `step_discovery()` duplicates `Command.handle()`'s watched-proposal loop, and CR-02's guard exists in only one of them

**File:** `solsys_code/unattended.py:307-361`, `solsys_code/management/commands/backfill_lco_observations.py:852-899`
**Issue:** The two blocks are near-identical: same `watched_rows()` query, same per-row
`sweep_proposal()` call with the same four keyword arguments, same broad `except Exception`
recording `f'failed: {type(exc).__name__}'`, same `last_run_at`/`last_run_summary` write
under the same `if not dry_run` guard, same failed-code accumulation. Only the terminal
reporting differs (a `StepResult` vs. a `CommandError`). Every prior fix to this behavior has
had to be reasoned about twice, and CR-02's new contract now lives in exactly one copy. This
is correct today only because the runner bypasses the CLI, which is subtle enough to be worth
removing.
**Fix:** extract a `sweep_watched_rows(*, dry_run, stdout=None, stderr=None) -> tuple[int, list[str]]`
helper in `backfill_lco_observations.py` and have both callers use it, leaving each with only
its own result translation.

### IN-14: `FOMO_BASE_URL` is now `None`-guarded but the three sibling path settings are not, and `_owner_mode()` can raise

**File:** `solsys_code/notifications.py:46`, `solsys_code/unattended.py:111`, `:384`, `:420`,
`:496`; `solsys_code/management/commands/check_unattended.py:67-70`
**Issue:** WR-07's fix established that a `local_settings.py` deriving a FOMO setting from an
unset environment variable yields `None` and must not crash the caller. The same hazard
applies unguarded to `FOMO_LOCK_DIR` (`Path(None)` → `TypeError` inside `command_lock()`,
i.e. the very first thing `run_tick()` does), `FOMO_STATE_DIR`, and `FOMO_LOG_FILE`
(interpolated into the failure-email body at `unattended.py:496` and `Path(...)`-ed at
`check_unattended.py:132`). Separately, `_owner_mode()` calls `path.stat()` after
`path.exists()` with no `try`, so an `EACCES`/`ENOENT` race turns the whole read-only
preflight into an uncaught traceback instead of a reported check.
**Fix:** either guard the three path settings the same way (`or '<documented default>'`) or
drop the `FOMO_BASE_URL` guard and validate all four once at settings load; wrap
`_owner_mode()`'s `stat()` in `try/except OSError` returning `'owner/mode unavailable'`.

### Previously reported, still open (out of the fix pass's `critical_warning` scope)

`36-REVIEW-FIX.md` correctly records that IN-01..IN-06 were not touched. All six were
re-checked against the current code and all six still apply unchanged:

- **IN-01** heartbeat ping ignores the HTTP status code — `unattended.py:140`.
- **IN-02** unattended discovery discards every per-request skip reason — `unattended.py:332-337`
  (no `stdout`/`stderr` passed, so `sweep_proposal()`'s default `io.StringIO()` sinks swallow them).
- **IN-03** suppression-state file written non-atomically and with the process umask —
  `unattended.py:427-428`.
- **IN-04** `run_unattended`'s command docstring still says "from later plans in this phase" —
  `run_unattended.py:20-24`; all four steps shipped, and `--help` shows this to operators.
- **IN-05** a whole-facility outage reports "failed 1", and a clean run still emits the
  dangling fragment `classes: ` — `unattended.py:204`, `:253`. The WR-08 fix made the first
  half slightly worse: with the re-check capped, `classes` can now be empty even on a large
  outage, so the summary reads `classes:  | recheck capped: 80 omitted`.
- **IN-06** `run_tick(only_step='typo')` runs nothing and reports a healthy tick —
  `unattended.py:551`.

---

_Reviewed: 2026-09-17T21:05:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 2 (re-review of the `36-REVIEW-FIX.md` fix pass)_
