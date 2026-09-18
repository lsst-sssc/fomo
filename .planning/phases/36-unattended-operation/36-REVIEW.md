---
phase: 36-unattended-operation
reviewed: 2026-09-17T23:55:00Z
depth: deep
iteration: 3
files_reviewed: 12
files_reviewed_list:
  - deploy/cron/fomo.crontab.example
  - docs/installation.rst
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/management/commands/run_unattended.py
  - solsys_code/notifications.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - solsys_code/tests/test_check_unattended.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/unattended.py
findings:
  critical: 0
  warning: 7
  info: 10
  total: 17
carried_forward_open: 1
status: issues_found
---

# Phase 36: Code Review Report (iteration 3 — re-review after the iteration-2 fix pass and the 36-06 gap closure)

**Reviewed:** 2026-09-17T23:55:00Z
**Depth:** deep
**Files Reviewed:** 12 (everything changed since `fe719d6d`, the commit iteration 2 was written against)
**Status:** issues_found

## Summary

This is an incremental re-review of the 21 fixes recorded in `36-REVIEW-FIX.md`
(iteration 2) plus gap-closure plan 36-06 (`12c51c6`, `f075a7f`, `1f3bbac`), which
corrected the heartbeat alert-window guidance.

**Verification of the prior 21 fixes.** All were checked against the real code rather than
against the fix report's prose. Twenty are genuinely and completely fixed. The two I
re-derived empirically:

- **WR-09 (inverted cron-line exit status) is fixed.** Running the committed line's exact
  shape in a real `sh` now yields `0` on a healthy tick, `1` on a failing tick, and `99`
  on a contended one — the inversion is gone. But the *semantics* of the new third case
  contradict the runner's own documented contract; see **WR-16**.
- **WR-10 (`load_state()` raising on a mixed-type `failing_steps`) is fixed** by the
  `isinstance(step, str)` filter at `unattended.py:434`, which sits before the `sorted()`
  at `:445`.

**One prior finding is only partially fixed and is carried forward.** WR-15's own
recommended remedies were "(a) suppress further mail when the state cannot be persisted"
**or** "(b) add a `check_state_dir()` preflight", with the review explicitly noting that
"(a) is the one that survives a directory that becomes unwritable after setup". The fix
pass implemented (b) only, and `36-REVIEW-FIX.md` records WR-15 as fixed. The runtime
email-storm loop is unchanged — see **WR-17**.

**The 36-06 gap closure is substantively correct.** The runbook's rewritten "Heartbeat."
paragraph now names both knobs, states the alert arithmetic, explains why the grace time
also bounds the `/start`→completion gap (accurate for healthchecks.io's start-signal
semantics), and adds a dedicated troubleshooting entry. Its propagation into
`check_unattended.py` is thinner than into the docs (see **IN-20**), and the runbook's
own description of what the preflight reports was not refreshed for the two *other*
checks the fix pass added (see **WR-21**).

**Credential hygiene (D-15/D-17/SCHED-10) mostly holds.** A scan of all twelve files for
`hc-ping`, healthchecks-shaped URLs, and bare UUIDs finds nothing; every fixture literal is
an obvious `example`/`FAKE-` placeholder; `check_heartbeat()` still prints set/unset only
and the new test pins `assertNotIn(_FAKE_HEARTBEAT_URL, stdout)`; the regenerated notebook
carries no absolute host path and no worktree path. `check_unattended` prints set/unset
only, never a value, so D-15 is satisfied. The one hole is **WR-22**: two `logger.debug()`
call sites interpolate a raw `str(exc)` from a portal call, which D-17 forbids — latent
only because the shipped `LOGGING` config drops `DEBUG`. No injection, path-traversal, or
deserialization defect was found. `ruff`'s 120-column limit is respected in every changed
Python file.

**Where the remaining defects cluster.** Four of the seven warnings are consequences of the
fix pass itself: a contract contradiction the WR-09 fix introduced (WR-16), the half-fix
of WR-15 (WR-17), an IN-02 fix whose output the project's own `LOGGING` config discards
(WR-18), and a `subprocess` call the WR-11 fix added without the error handling the same
commit gave `_owner_mode()` (WR-20). The remaining three are pre-existing gaps the earlier
iterations did not catch (WR-19, WR-21, WR-22). **WR-18 and WR-22 must be fixed together**:
WR-18's obvious remedy (raise the verbosity of the unattended log) is exactly what would
activate WR-22's leak.

## Narrative Findings (AI reviewer)

### Verification of the prior 21 fixes

| Prior finding | Verdict | Evidence |
|---|---|---|
| WR-09 inverted cron-line exit status | **Fixed** (but see **WR-16**) | Live `sh` probe: healthy `0`, failing `1`, contended `99`; `check_unattended.py:325-330`, `fomo.crontab.example:52`, `test_check_unattended.py` `test_line_ends_with_an_explicit_exit_of_the_captured_status` |
| WR-10 `load_state()` raises on mixed types | **Fixed** | `unattended.py:434` filters before `sorted()` at `:445`; `test_mixed_type_failing_steps_are_coerced_not_raised` |
| WR-11 `check_flock()` didn't prove `-E` | **Fixed** (but see **WR-20**) | `check_unattended.py:85-95` |
| WR-12 runbook flag docs | **Fixed** | `telescope_runs_calendar.rst:388-391`, `:1779-1780` |
| WR-13 `FOMO_BASE_URL` scoped to cron only | **Fixed** | `telescope_runs_calendar.rst:1463-1473`, `fomo.crontab.example:16-21`, `docs/installation.rst:110-117` |
| WR-14 unreachable stale-lock remedy | **Fixed** | `telescope_runs_calendar.rst:2045-2057` |
| WR-15 unwritable `FOMO_STATE_DIR` mail storm | **Partly fixed** — see **WR-17** | `check_unattended.py:177-191` adds the setup-time check; `unattended.py:646-660` is unchanged |
| IN-01 heartbeat ignored HTTP status | **Fixed** | `unattended.py:156` `raise_for_status()` |
| IN-02 skip reasons discarded | **Partly fixed** — see **WR-18** | `unattended.py:374-381` captures them, but at `DEBUG` |
| IN-03 non-atomic state write | **Fixed** | `unattended.py:473-483`; mkstemp + `chmod 0o600` + `os.replace` |
| IN-04 stale `run_unattended` docstring | **Fixed** | `run_unattended.py:20-25` (but see **IN-16** for two siblings missed) |
| IN-05 outage read as "failed 1" / dangling `classes:` | **Fixed** | `unattended.py:226`, `:281-290` |
| IN-06 unknown `only_step` silent no-op | **Fixed** | `unattended.py:605-606` |
| IN-07 `notify_staff()` discarded `send_mail()`'s result | **Fixed** | `notifications.py:84-95` |
| IN-08 cron-line shape test | **Fixed** | `test_check_unattended.py:263`, new token-for-token test |
| IN-09 stale `check_unattended` docstrings | **Fixed** | `check_unattended.py:15-19`, `:376-381`, `test_check_unattended.py:3` |
| IN-10 stale runbook `flock -n` passages | **Fixed** | `telescope_runs_calendar.rst:1587-1600` |
| IN-11 notebook prose-only | **Fixed** | Cell 18 is a real executed cell (`execution_count` 1..12 is sequential across all 12 code cells, so the notebook was genuinely re-run) |
| IN-12 guard ordering/truthiness | **Fixed** | `backfill_lco_observations.py:879-899` (`is not None`, above username resolution) |
| IN-13 duplicated watched-proposal loop | **Fixed** | `backfill_lco_observations.py:697-769`, both callers |
| IN-14 `None` path settings, `_owner_mode()` | **Fixed** | `unattended.py:121`, `:417`, `:466`, `:551`; `check_unattended.py:108-112` |

## Warnings

### WR-16: The cron line now reports a benign lock-contended skip as exit 99 — a failure to any supervisor — and both the docstring and the crontab comment misattribute that code to `run_unattended`

**File:** `solsys_code/management/commands/check_unattended.py:295-300` and `:325-330`,
`deploy/cron/fomo.crontab.example:47-52`; cf. `solsys_code/unattended.py:88-91`, `:590-594`,
`:665-668`
**Issue:** The WR-09 fix ends the line with `exit $rc`. Verified in a real `sh` against a
real `flock`:

```
healthy tick   -> exit=0
failing tick   -> exit=1
lock contended -> exit=99
```

That is correct for the first two cases and fixes the inversion. The third case is a new
contract violation. `run_tick()`'s own docstring is explicit that contention is *not* a
failure — "A contended whole-run lock is NOT a failure -- it returns ``exit_code=0`` with
no results ... the heartbeat (D-12) is the structural backstop" (`unattended.py:590-594`),
and `TickResult.exit_code` documents "0 on a healthy tick (including a lock-contended
skip)" (`:88-91`). The cron line now overrides that decision from the outside: any
supervisor reading the line's status — cron's own syslog `CMD exit status`, a systemd
timer if this is migrated, an `OnFailure=` hook, a `run-parts` harness, or a monitoring
wrapper — sees a **non-zero status on a routine tick overlap**, which the runbook itself
calls normal ("One occurrence is normal (an overrunning tick colliding with the next
scheduled one)", `telescope_runs_calendar.rst:1600`). The previous iteration's WR-09
complained that a healthy tick was indistinguishable from a failing one; this iteration's
line makes a *healthy skip* indistinguishable from a failing tick.

Both prose claims about the new line are also wrong in the same direction:

- `check_unattended.py:299-300`: "the line's own status is always `run_unattended`'s (0
  healthy, 1 failing, 99 skipped)". `run_unattended` never exits 99 — it exits **0** on
  contention. 99 is `flock`'s own `-E` code and is a status `run_unattended` cannot
  produce.
- `fomo.crontab.example:48-49`: the same sentence, same error.

**Fix:** decide which contract wins and make all three surfaces agree. The runner's own
contract (contention is benign, the heartbeat is the backstop) is the one the whole phase
is built on, so normalize the skip to 0 after the log line is written, in `cron_line()`
and the committed template together:

```
*/15 * * * * <flock> -n -E 99 <lock> <python> <manage.py> run_unattended >> <log> 2>&1; rc=$?; \
  [ $rc -eq 99 ] && { echo "$(date -Is) run_unattended skipped: lock held" >> <log>; rc=0; }; exit $rc
```

and correct both prose claims to "the line's own status is `run_unattended`'s (0 healthy,
1 failing); a lock-held skip is normalized to 0 after the skip line is logged, matching
`run_tick()`'s own decision that contention is not a failure". If the project instead
*wants* 99 surfaced, say so explicitly in both places ("99 means the tick was skipped —
benign in isolation, investigate only if repeated") and update `run_tick()`'s docstring to
note the divergence. Either way, add a `TestCronLine` case pinning whichever choice is made.

### WR-17: WR-15's runtime email storm is still open — only the setup-time preflight was added

**File:** `solsys_code/unattended.py:646-660`, `solsys_code/management/commands/check_unattended.py:177-191`
**Issue:** `36-REVIEW-FIX.md` records WR-15 as fixed, citing the new `check_state_dir()`.
That closes the *setup-time* case only. WR-15's own text named two remedies and said which
one mattered: "(a) is the one that survives a directory that becomes unwritable after
setup." (a) was not implemented, and the runtime path is byte-for-byte unchanged. Traced
against the current code with a state directory that becomes unwritable or full **after**
the preflight passed (a full `/var/lock` tmpfs is the realistic trigger; `FOMO_STATE_DIR`
defaults to `FOMO_LOCK_DIR`, `settings.py:418`):

1. `load_state()` (`:647`) → its `except (OSError, ValueError)` at `:421` returns
   `{'failing_steps': [], 'notified_at': None}`.
2. `decide_notification()` (`:648`) sees an empty previous set and a non-empty current one
   → `'failure'`.
3. `_send_notification()` (`:654`) **sends**.
4. `save_state()` (`:656`) raises `OSError` → caught at `:659`, logged as one
   `unattended notification/state handling raised: OSError` line.
5. Nothing records that staff were told, so step 2 reaches the identical conclusion on the
   next tick.

At the D-04 15-minute cadence that is 96 identical emails per staff address per day, for
as long as one step keeps failing — D-11's suppression rule failing open in the most
visible possible way, and the exact scenario `check_state_dir()`'s own docstring describes
("makes every tick send the same failure email again, forever") without preventing it once
the host is past setup.
**Fix:** implement remedy (a) alongside the preflight — a process-lifetime fallback so an
unpersistable state cannot re-notify:

```python
_state_write_failed = False  # module-level, reset per process


def _persist(failing_steps, when):
    global _state_write_failed
    try:
        save_state(failing_steps, when)
    except OSError:
        _state_write_failed = True
        logger.error(
            'could not persist unattended suppression state to %s -- further notifications '
            'for this failing set are suppressed for this process',
            Path(settings.FOMO_STATE_DIR or settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR) / _STATE_FILENAME,
        )
```

and skip `_send_notification()` when `_state_write_failed` is set and the decision is
`'failure'`/`'reminder'`. Add a `TestStateFileRobustness` case patching `save_state` to
raise `OSError` across two consecutive `run_tick()` calls and asserting
`len(mail.outbox) == 1`, not 2. Either way, un-mark WR-15 in `36-REVIEW-FIX.md`.

### WR-18: IN-02's "skip reasons are no longer discarded" fix logs at `DEBUG`, which this project's own `LOGGING` config drops — the reasons are still discarded in production

**File:** `solsys_code/unattended.py:374-381`; cf. `src/fomo/settings.py:193-202`,
`solsys_code/tests/test_unattended.py` `test_per_request_skip_reasons_are_logged_not_discarded`
**Issue:** The IN-02 fix captures `sweep_proposal()`'s per-request skip lines into
`io.StringIO()` sinks and re-emits them with `logger.debug('discovery %s: %s', ...)`
(`:381`). The project ships exactly one logging configuration, and its root logger is
pinned to `INFO`:

```python
LOGGING = {
    ...
    'loggers': {'': {'handlers': ['console'], 'level': 'INFO'}},
}
```

`solsys_code.unattended` declares no logger of its own in that config, so it inherits the
root level. Every one of those `DEBUG` records is therefore filtered out before it reaches
the `StreamHandler` whose stderr the crontab line redirects into
`/var/log/fomo/unattended.log`. Net effect on a real deployment: identical to before the
fix — the operator still sees only the bare `swept: N, failed: M` summary, and the skip
reasons are still gone. The new test passes only because
`self.assertLogs('solsys_code.unattended', level='DEBUG')` temporarily installs its own
handler at `DEBUG`, which is exactly the condition that does not hold at runtime; it
therefore verifies that `logger.debug()` was *called*, not that IN-02's stated outcome
("They must now reach the log") is achieved.
**Fix:** promote this one re-emission to `INFO`, which is the level the same function
already uses for its sibling operator-facing line at `:383` (`'0 watched proposals,
nothing to discover'`), and which is what the redirected log actually captures:

```python
                if captured_text:
                    logger.info('discovery %s: %s', sink_name, captured_text)
```

Skip reasons are structural (`Skipping request <id>: no configuration with a named
target.`), not credentials, so `INFO` is consistent with D-17 — see IN-18 for the wording.
Then assert the level in the test (`assertLogs(..., level='INFO')`) so a future downgrade
back to `DEBUG` fails. **Do not implement this by lowering the global log level to `DEBUG`
instead** — that would activate **WR-22**.

### WR-19: `check_email()`'s backend check rejects only the console backend, so `dummy`, `locmem` and `filebased` pass a check whose own docstring promises "the email backend can actually deliver"

**File:** `solsys_code/management/commands/check_unattended.py:194-217`
**Issue:** The check is a single equality test:

```python
    is_console = backend == 'django.core.mail.backends.console.EmailBackend'
```

Django ships four other non-delivering backends. `django.core.mail.backends.dummy.EmailBackend`
is the canonical "turn email off" idiom and is a realistic production setting on a host
where someone wanted to silence mail temporarily; `locmem` is what a half-finished
`local_settings.py` copied from a test config carries; `filebased` writes to a directory
nobody reads. All three pass this hard check and are reported `[ok] EMAIL_BACKEND` with
their own dotted path as the detail, and `--send-test-email` also "succeeds" against all
three (`dummy.EmailBackend.send_messages()` returns `len(email_messages)`, so
`notify_staff()` returns True and `_send_test_email()` reports
`sent one test email to staff recipients`). The operator then installs the crontab line
believing D-11's primary alert channel is proven, and no failure notice will ever arrive.
This is a strictly worse outcome than the console backend the check does catch, because
`--send-test-email` actively confirms it.
**Fix:** reject the whole non-delivering set by name rather than one member of it:

```python
_NON_DELIVERING_BACKENDS = {
    'django.core.mail.backends.console.EmailBackend',
    'django.core.mail.backends.dummy.EmailBackend',
    'django.core.mail.backends.locmem.EmailBackend',
    'django.core.mail.backends.filebased.EmailBackend',
}
...
    is_non_delivering = backend in _NON_DELIVERING_BACKENDS
```

with a detail naming which one and why it cannot deliver, and add one test per backend.
(A custom third-party backend still passes, which is the right default — the check can
only prove a *known* non-deliverer.)

### WR-20: The `check_flock()` probe the WR-11 fix added can hang or traceback the whole read-only preflight — the exact hardening the same commit gave `_owner_mode()`

**File:** `solsys_code/management/commands/check_unattended.py:85`
**Issue:**

```python
    probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False)  # noqa: S603
```

has neither a `timeout=` nor a `try/except OSError`. `shutil.which()` returning a path is
an `os.access(..., X_OK)` test, not a guarantee the `execve` will succeed: a dangling
symlink target, a `noexec` mount, an `ENOEXEC` wrapper script with a bad shebang, an
`ETXTBSY`, or a plain TOCTOU delete between the `which()` at `:77` and the `run()` at
`:85` all raise `OSError`/`PermissionError` out of `check_flock()`. Because `check_flock()`
is the **first** entry in `Command.handle()`'s list (`:411`), that exception aborts the
whole command: the operator loses the other seven check results *and* the printed cron
line, and gets a traceback out of a command whose module docstring opens with "This
command is read-only by construction" and whose whole purpose (`:8-10`) is "naming every
failed hard check in one `CommandError` so a fresh-host operator sees the whole list of
problems in one run". Separately, with no `timeout=` a `flock` binary on a stalled NFS
mount hangs the preflight indefinitely. The same commit range explicitly hardened
`_owner_mode()`'s `stat()` against precisely this class ("an unavailable stat should
degrade to a reported detail, never an uncaught traceback", `:105-106`); the new
`subprocess.run()` did not get the same treatment.
**Fix:**

```python
    try:
        probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False, timeout=5)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name='flock',
            ok=False,
            hard=True,
            detail=f'{path} could not be executed to verify -E support: {type(exc).__name__}',
        )
```

(reporting the class name only, matching D-17), and add a test patching `subprocess.run`
with `side_effect=OSError` that asserts a `CommandError` naming `flock` rather than an
`OSError` escaping.

### WR-21: The runbook's description of what `check_unattended` reports was not updated for the two hard checks the same fix pass added

**File:** `docs/runbooks/telescope_runs_calendar.rst:1449-1454` and `:1480-1496`; cf.
`solsys_code/management/commands/check_unattended.py:66-96`, `:177-191`, `:411-418`
**Issue:** The iteration-2 fixes added two hard checks — `FOMO_STATE_DIR` writability
(WR-15) and `flock -E` support (WR-11) — taking the total from six to eight. The
`check_unattended` module docstring and the test module docstring were updated (IN-09);
the runbook, which is the page an operator actually follows, was not:

- Step 4's enumeration (`:1480-1496`) still lists exactly the old set: "whether ``flock``
  is on ``PATH``, whether the lock and log directories exist and are writable, whether the
  email backend can actually deliver and at least one staff user has an email on file,
  whether ``FOMO_HEARTBEAT_URL`` is set ..., whether ``FOMO_BASE_URL`` has been changed
  ..., and whether at least one ``WatchedProposal`` row is active". Neither the state
  directory nor the `-E` probe appears, and the closing sentence still says the hard set is
  "(flock, the directories, or email)" without naming which directories.
- Setup step 1 (`:1449-1454`) still says "Create the **two** directories the schedule below
  assumes exist", listing `/var/lock/fomo` and `/var/log/fomo`. `FOMO_STATE_DIR` is a
  separately settable path (`settings.py:418`) that merely *defaults* to `FOMO_LOCK_DIR`; a
  host that points it elsewhere now gets a hard preflight failure for a directory the
  runbook never told the operator to create, and `grep -n FOMO_STATE_DIR docs/runbooks/`
  returns nothing at all.

CLAUDE.md makes any `docs/runbooks/` page whose documented behavior a change affects part
of the deliverable, not follow-up polish — the same rule WR-12 was raised under, and the
same rule quick task `260726-kdp` is recorded as breaching.
**Fix:** in step 4's enumeration, replace "whether the lock and log directories exist and
are writable" with "whether the lock, log and suppression-state directories exist and are
writable" and "whether ``flock`` is on ``PATH``" with "whether ``flock`` is on ``PATH``
*and* new enough to support ``-E`` (util-linux 2.27+), which the cron line's skip
detection needs"; in step 1, add a sentence that `FOMO_STATE_DIR` defaults to
`FOMO_LOCK_DIR` and only needs creating separately if it has been pointed elsewhere.

### WR-22: Two `logger.debug()` sites interpolate a raw portal-exception message, which D-17 forbids — latent only because the shipped config drops `DEBUG`

**File:** `solsys_code/management/commands/backfill_lco_observations.py:349`,
`solsys_code/unattended.py:191`; cf. `solsys_code/tests/test_unattended.py`
`TestCredentialHygiene`
**Issue:** The phase's D-17 discipline is "only the exception's class name ever reaches the
row, stderr, or the log — never `str(exc)`", and it is honoured at every `warning`/`error`
site in `unattended.py` (`:158`, `:225`, `:236`, `:576`, `:625`, `:660`) and in
`sweep_watched_rows()` (`backfill_lco_observations.py:742`). Two `DEBUG` sites break it:

```python
# backfill_lco_observations.py:346-349 -- the exception is from a live portal call
    try:
        result = facility.get_observation_status(observation_id)
    except Exception as exc:
        logger.debug(f'Observed-block lookup failed for observation_id={observation_id!r}: {exc}')

# unattended.py:188-191 -- `except Exception`, so not necessarily FOMO's own exception class
                except Exception as exc:  # noqa: BLE001 -- FOMO's own reconcile_run(), D-17's
                    logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
```

`facility.get_observation_status()` is an authenticated LCO Observation Portal call. This
codebase's own credential-hygiene test models exactly what such an exception's message can
carry — `TestCredentialHygiene` constructs
`ImproperCredentialsException(f'portal error key={_FAKE_LCO_API_KEY} url={_FAKE_HEARTBEAT_PING_URL}')`
— so `str(exc)` here can put the LCO API key and a heartbeat ping URL into
`/var/log/fomo/unattended.log`, a file `deploy/logrotate/fomo.example` keeps on disk and
that an operator is told to read first when triaging. Today this is inert because
`settings.LOGGING` pins the root logger to `INFO`; it becomes live the moment anyone sets
`DEBUG` to chase a problem — which is the natural response to WR-18, and which the runbook
does not warn against. The second site's `# noqa` comment claims the exception is "FOMO's
own `reconcile_run()`, D-17's second bucket", but the `except` clause is bare `Exception`,
so anything `reconcile_run()` propagates (including a `requests` error from deeper in the
call chain) is logged with its full message.
**Fix:** use the class name at both sites, matching every other call site in the phase:

```python
        logger.debug('Observed-block lookup failed for observation_id=%r: %s', observation_id, type(exc).__name__)
...
                    logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, type(exc).__name__)
```

and extend `TestCredentialHygiene` with a case that runs a tick under
`self.assertLogs(level='DEBUG')` and asserts the fake key and ping URL appear nowhere in
the captured output — the current suite only captures at the default level, which is why
this survived two review iterations.
**Disposition (2026-09-18, 36-UAT.md Test 2):** **accepted, not fixed.** The developer recorded
an explicit acceptance that the class-name-only discipline holds only while `settings.LOGGING`
keeps the root logger at `INFO`. The constraint is documented at the point an operator would
change it — a comment directly above `LOGGING` in `src/fomo/settings.py` naming both sites and
the required fix — and in `36-VERIFICATION.md` § Acknowledged Gaps. If the level is ever raised,
WR-22 and WR-18 must be fixed together before the change ships.

## Info

### IN-15: The crontab template's own comment contradicts the command line directly below it

**File:** `deploy/cron/fomo.crontab.example:29`
**Issue:** Line 29 still describes "the ``[ $? -eq 99 ] && echo ... skipped`` tail below",
but line 52 was rewritten by the WR-09 fix and now reads `[ $rc -eq 99 ]`. The WR-09
explanation block at `:47-51` correctly describes `[ $? -eq 99 ]` as the *earlier* form, so
the file now says both that `$?` is what the line uses and that `$?` is what the line no
longer uses.
**Fix:** change `:29` to `` `[ $rc -eq 99 ] && echo ... skipped` ``.

### IN-16: Two stale "a later plan in this phase" references survive in the committed crontab template

**File:** `deploy/cron/fomo.crontab.example:9-11` and `:60`
**Issue:** `:9-11` describes `python manage.py check_unattended` as "(a later plan in this
phase)" and `:60` describes `deploy/logrotate/fomo.example` as "(a later plan in this
phase)". Both shipped — `deploy/logrotate/fomo.example` exists on disk and was reviewed in
iteration 2. This is the same defect IN-04 raised and the fix pass corrected in
`run_unattended.py`; the sweep stopped at the Python file and missed the two instances in
the operator-facing template. `grep -rn "later plan in this phase"` outside `.planning/`
finds only these.
**Fix:** drop both parentheticals.

### IN-17: `sweep_watched_rows()`'s docstring and type hints were invalidated by the IN-02 fix that landed two commits later

**File:** `solsys_code/management/commands/backfill_lco_observations.py:697-714`, `:926-928`
**Issue:** Two drifts, both introduced by the ordering of the fix commits (`3ac974d`
IN-13, then `cf5c7f4` IN-02):
(1) `:712-713` says "``stdout``: forwarded to ``sweep_proposal()``, **unused (``None``) by
the runner**, which has no stdout of its own to write progress lines to". The runner now
passes a live `io.StringIO()` (`unattended.py:374-377`) precisely so it is *not* unused.
(2) the signature annotates `stdout: io.StringIO | None` / `stderr: io.StringIO | None`,
but `Command.handle()` passes `self.stdout`/`self.stderr`, which are Django
`OutputWrapper` instances, not `io.StringIO` — so the annotation is wrong for one of the
two callers it was extracted to serve.
**Fix:** reword (1) to describe the runner's capture-and-log use, and widen (2) to
`typing.TextIO | None` (matching `sweep_proposal()`'s own `Any` parameters at `:424-425`).

### IN-18: The IN-02 comment's D-17 justification overstates what the captured sinks contain

**File:** `solsys_code/unattended.py:366-373`
**Issue:** The comment asserts the captured text holds skip reasons that are "never portal
response content or a credential (D-17)". The credential half is right; the portal half is
not. `sweep_proposal()` writes portal-derived values to both sinks:
`stderr.write(f'Skipping request {observation_id}: ...')` (`backfill_lco_observations.py:519`,
`:529`, `:536`, `:550`) carries request ids from the payload, and the dry-run
`stdout.write(...)` at `:589-593` carries `target_name` and `status` straight from the
RequestGroup JSON. None of that is a credential and none is PII, so the *decision* to log
it is fine — but the stated reason is not the true one, and a future reader relying on
"never portal response content" to widen what gets logged would be relying on something
false.
**Fix:** reword to "these are structural skip reasons carrying only portal identifiers
(request/observation ids, target names, states) — never a credential and never a raw
response body, request URL, or caught exception's message (D-17)".

### IN-19: The `--proposal` guard's message is ungrammatical for the single-flag case, and that wording is now committed in the notebook's executed output

**File:** `solsys_code/management/commands/backfill_lco_observations.py:896-899`;
`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (cell 18 output)
**Issue:** `f'{", ".join(ignored)} require --proposal; ...'` uses the plural verb
unconditionally. The single-flag case — by far the common one, and the only one the
notebook and the two new tests exercise — reads
`--created-after require --proposal; the watched-list sweep takes its overrides from each
WatchedProposal row.` That exact string is now baked into the notebook's committed output
and is what an operator sees.
**Fix:**

```python
                verb = 'requires' if len(ignored) == 1 else 'require'
                raise CommandError(
                    f'{", ".join(ignored)} {verb} --proposal; the watched-list sweep takes its '
                    'overrides from each WatchedProposal row.'
                )
```

and regenerate the notebook (`jupyter nbconvert --to notebook --execute --inplace`) so the
committed output matches.

### IN-20: The 36-06 heartbeat guidance reached the runbook and the crontab with both knobs, but the preflight with only one

**File:** `solsys_code/management/commands/check_unattended.py:242-246`;
`solsys_code/tests/test_check_unattended.py` `test_set_heartbeat_reminds_about_the_check_period`;
cf. `deploy/cron/fomo.crontab.example:31-35`, `docs/runbooks/telescope_runs_calendar.rst:1548-1569`
**Issue:** 36-06's stated purpose was "correct heartbeat guidance with **both** alert-window
knobs". The runbook and the crontab template both name the expected ping interval
(`Period`, 15 min) *and* the grace time (`Grace`, ~20 min). The `[ok] heartbeat` detail —
the one surface the operator actually executes, and the only one that fires at setup time —
names only `Period`, and the new test pins only `assertIn('Period', stdout)`, so a future
edit that drops the grace half entirely would still pass. An operator who follows the
preflight's reminder alone sets `Period=15` and leaves healthchecks.io's 1-hour default
grace, producing a 75-minute alert window instead of the documented ~35. Separately, the
literal `15 min` in this string is now a third hardcoded copy of the schedule constant
(alongside `cron_line()`'s `*/15` and the template's), with nothing tying them together.
**Fix:** extend the detail to "confirm the check's own expected ping interval (Period) is
15 min, not its 1-day default, and its grace time is about 20 min", assert both substrings
in the test, and derive the "15" from the same source `cron_line()`'s `*/15` uses (a
module constant) so the three cannot drift.

### IN-21: `save_state()`'s temp files leak on a process kill, and nothing ever reaps them

**File:** `solsys_code/unattended.py:473-483`
**Issue:** The IN-03 fix's `except BaseException: tmp_path.unlink()` covers the exception
path only. A `SIGKILL`/OOM kill between `mkstemp()` (`:473`) and `os.replace()` (`:479`) —
the same failure class the fix's own docstring cites for the lock file — leaves a
`.unattended-state.json.<random>.tmp` file in `FOMO_STATE_DIR` forever. `load_state()`
reads only the exact `_STATE_FILENAME`, so there is no correctness impact, but on a host
that OOM-kills ticks the directory (which defaults to `/var/lock/fomo`, often a small
tmpfs) accumulates one file per occurrence with nothing to clean them.
**Fix:** at the top of `save_state()`, unlink any `.{_STATE_FILENAME}.*.tmp` older than a
tick interval, or note the leak in the docstring so an operator knows the files are safe
to delete.

### IN-22: `_DEFAULT_LOCK_DIR`/`_DEFAULT_LOG_FILE` now exist in three places

**File:** `solsys_code/unattended.py:58-59`,
`solsys_code/management/commands/check_unattended.py:42-43`, `src/fomo/settings.py:415`, `:422`
**Issue:** The IN-14 fix duplicated the same two literals into both modules, each with a
near-identical comment saying it "mirrors settings.py's own `os.getenv(..., <default>)`
defaults". Nothing enforces the mirroring, so a change to `settings.py`'s defaults now
silently desynchronizes two fallback paths whose entire purpose is to match it.
**Fix:** export them once — e.g. `solsys_code/unattended.py` as the single owner, imported
by `check_unattended.py` — or add one test asserting
`unattended._DEFAULT_LOCK_DIR == check_unattended._DEFAULT_LOCK_DIR` and that both match
`settings.py`'s literal.

### IN-23: `cron_line()` bakes a `PATH`-resolved binary into a line destined for a service crontab, and the new template test fails opaquely if the template's schedule line is renamed

**File:** `solsys_code/management/commands/check_unattended.py:320`;
`solsys_code/tests/test_check_unattended.py` `test_line_matches_the_committed_template_token_for_token`
**Issue:** Two small ones.
(1) WR-05's `shutil.which('flock')` resolves against the *preflight process's* `PATH` and
the result is printed for the operator to paste into a persistent, scheduled command. An
operator with a stale or user-writable directory early in `PATH` (a conda/venv `bin`, a
`~/bin`) can end up installing a non-system `flock` into a service crontab — a
low-likelihood but persistent outcome, and one the committed template's hardcoded
`/usr/bin/flock` did not have. A one-line sanity note ("resolved outside the usual system
directories — confirm this is the `flock` you want in a crontab") would keep WR-05's
benefit without the silent case.
(2) the new test's `next(...)` has no default, so if the template's `*/15` line is ever
reformatted or the file moved, the test fails with a bare `StopIteration` instead of an
assertion naming the problem; and `Path(__file__).resolve().parents[2]` assumes an editable
checkout layout.
**Fix:** (1) compare `flock_path` against a small allow-list of system directories and add
a note to the detail when it falls outside; (2) `next(..., None)` plus
`self.assertIsNotNone(template_line, f'no */15 line in {template_path}')`.

### IN-24: The regenerated notebook is the only pre-executed notebook with no `kernelspec` metadata

**File:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (top-level
`metadata`)
**Issue:** All seven other notebooks under `docs/notebooks/pre_executed/` carry
`metadata.kernelspec` = `python3`; this one carries `language_info` only. Verified
**pre-existing** — the same metadata was already missing at `fe719d6d`, so the IN-11
regeneration preserved rather than caused it. Flagged because the file is in scope and
because nbsphinx and JupyterLab both use `kernelspec` to pick an interpreter if the
notebook is ever re-executed by a reader or by a future `--execute` build.
**Fix:** add the standard block on the next regeneration:

```json
  "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
```

### Previously reported, still open

- **WR-15** (iteration 2) — recorded as fixed in `36-REVIEW-FIX.md`, but only remedy (b)
  was applied. Carried forward as **WR-17** above.

---

_Reviewed: 2026-09-17T23:55:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 3 (re-review of the `36-REVIEW-FIX.md` fix pass and the 36-06 gap closure)_
