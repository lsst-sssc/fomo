---
phase: 36-unattended-operation
reviewed: 2026-09-17T00:00:00Z
depth: deep
files_reviewed: 23
files_reviewed_list:
  - CLAUDE.md
  - deploy/cron/fomo.crontab.example
  - deploy/logrotate/fomo.example
  - docs/installation.rst
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/notebooks.rst
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/management/commands/run_unattended.py
  - solsys_code/migrations/0022_watchedproposal.py
  - solsys_code/models.py
  - solsys_code/notifications.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_check_unattended.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/tests/test_watched_proposal.py
  - solsys_code/unattended.py
  - src/fomo/settings.py
findings:
  critical: 2
  warning: 8
  info: 6
  total: 16
status: issues_found
---

# Phase 36: Code Review Report

**Reviewed:** 2026-09-17
**Depth:** deep
**Files Reviewed:** 23
**Status:** issues_found

## Summary

Phase 36 adds the unattended runner (`solsys_code/unattended.py`), its cron entry point,
a read-only preflight command, a shared request-free staff-notification helper, the
admin-editable `WatchedProposal` model, and the deploy/runbook artifacts. The in-process
design is careful — per-step failure isolation, class-name-only exception logging, no
`reverse()` anywhere on the unattended path (verified: no `reverse` call exists in
`unattended.py`, `notifications.py`, `campaign_reconciler.py`, `observation_projector.py`,
`project_observation_calendar.py`, or `backfill_lco_observations.py`), migration state is
clean (`makemigrations --check` reports no changes), and the credential-hygiene tests are
genuinely adversarial.

The serious defects are at the seams the unit tests cannot reach: the boundary between the
cron line and the process it starts, and the CLI boundary of the newly-optional
`--proposal` argument.

**CR-01 is the phase-defining defect: as shipped, the documented cron line makes every
scheduled tick a no-op.** The crontab template — and the line `check_unattended` prints for
the operator — guards the run with `flock -n <FOMO_LOCK_DIR>/run_unattended.lock`, which is
the exact same file `command_lock('run_unattended')` locks from inside the process. `flock(2)`
locks are per open file description, so the child's second `open()` of that file is denied by
the lock its own parent holds. Verified empirically against the real code path on util-linux
2.37.4 (the version 36-RESEARCH.md records for the target host):

```
$ flock -n $D/run_unattended.lock python probe.py     # probe calls the real command_lock()
RESULT: LockContended -- tick SKIPS: /tmp/.../run_unattended.lock is already locked
```

Every tick therefore returns `exit_code=0` with zero steps run, zero heartbeat pings, and a
"lock held -- skipping this tick" line — which the runbook then teaches the operator to
misdiagnose as a stuck previous tick. `36-RESEARCH.md` line 593 (Assumption A3) flags exactly
this as untested ("the raise-on-contention behavior itself was not exercised live"), and
line 183-186 recommended per-*step* locks only — the runner-level internal lock sharing the
cron file's name is an implementation addition, not a locked decision.

## Critical Issues

### CR-01: The shipped cron line self-deadlocks — every scheduled tick runs nothing and exits 0

**File:** `deploy/cron/fomo.crontab.example:27`, `solsys_code/management/commands/check_unattended.py:165-172`, `solsys_code/unattended.py:104-115` and `:497`
**Issue:** `cron_line()` and the committed template both build
`/usr/bin/flock -n <FOMO_LOCK_DIR>/run_unattended.lock <python> <manage.py> run_unattended`,
while `run_tick()` opens *the same path* (`command_lock('run_unattended')` →
`Path(settings.FOMO_LOCK_DIR) / 'run_unattended.lock'`) and takes `LOCK_EX | LOCK_NB` on a
fresh file descriptor. `flock(1)` holds its lock on an fd that is inherited across `exec`,
and `flock(2)` treats two open file descriptions of the same file as independent — so the
child's lock attempt is denied by its own parent. `run_tick()` catches `LockContended`,
writes the skip line, and returns `TickResult(exit_code=0, results=())`.

Consequences on a real host: none of `status_refresh`, `project_sweep`, `discovery`, or
`reconcile` ever runs; no `/start` or `/<exit-code>` heartbeat ping is ever sent (the ping
is inside the lock, `unattended.py:499-500`); no failure email can ever be generated; and
`flock` exits 0, so the crontab's `|| echo ... lock held` tail stays silent. The only signal
is the skip line on stderr, which `docs/runbooks/telescope_runs_calendar.rst:1972` and
`:1565-1570` tell the operator means "a previous tick is genuinely stuck" — sending them to
delete a lock file that will be re-taken on the next tick. `TestLocking.test_contended_lock_skips_every_step`
(`solsys_code/tests/test_unattended.py:289-306`) exercises the *same* two-fd contention and
asserts the skip is correct behavior, so the suite actively confirms the mechanism while
missing that cron triggers it.

Verified with the real code path (util-linux 2.37.4, matching the host recorded in
36-RESEARCH.md:620): running `command_lock('run_unattended')` under
`flock -n <same file>` raises `LockContended`.

**Fix:** give the cron guard and the in-process runner lock different files, and keep
`cron_line()`/the template/the runbook in agreement:

```python
# solsys_code/management/commands/check_unattended.py -- cron_line()
lock_file = Path(settings.FOMO_LOCK_DIR) / 'run_unattended.cron.lock'  # NOT run_unattended.lock
```

```
# deploy/cron/fomo.crontab.example
*/15 * * * * /usr/bin/flock -n /var/lock/fomo/run_unattended.cron.lock ... run_unattended ...
```

Add a regression test that asserts the two paths differ, e.g.
`assertNotEqual(cron_lock_path_in(cron_line()), str(Path(settings.FOMO_LOCK_DIR) / 'run_unattended.lock'))`,
or better, a subprocess test that runs `flock -n <cron lock> python manage.py run_unattended --step reconcile`
and asserts the step actually ran. (Dropping the runner-level internal lock entirely is the
alternative — but then `run_unattended --step <name>` loses the exclusivity the runbook
promises at `:1565`.)

### CR-02: `--created-after` / `--created-before` / `--username` / `--target-list` are silently ignored on the bare `backfill_lco_observations` invocation

**File:** `solsys_code/management/commands/backfill_lco_observations.py:790-853`
**Issue:** Making `--proposal` optional created a second code path that accepts, parses, and
then discards four other arguments. `handle()` reads `options.get('created_after')` /
`options.get('created_before')` only in the `if proposal:` branch (`:807-808`); the watched
loop calls `sweep_proposal(row.proposal_code, target_list_name=row.target_list_name or None,
user=row.attributed_to, dry_run=dry_run, ...)` (`:827-834`) with no window arguments at all,
and overrides `--username`'s resolved `user` and `--target-list` with the row's own values.
argparse accepts the flags, so `python manage.py backfill_lco_observations --created-after 2026-09-01`
runs a **full-history** sweep of every watched proposal — creating Targets, ObservationRecords,
ObservationGroups and TargetList memberships for RequestGroups the operator explicitly asked to
exclude — with no error, no warning, and no mention in the summary line. Before this phase the
combination was impossible (`--proposal` was `required=True`), and it is untested: no test in
`solsys_code/tests/test_backfill_lco_observations.py` passes a window flag without `--proposal`.
The invalid-ISO-8601 `CommandError` also disappears on this path, because `_parse_created_bound()`
is now only reached from inside `sweep_proposal()`.

**Fix:** fail closed on the incompatible combination, in `handle()` before the watched loop:

```python
if not proposal:
    ignored = [
        flag
        for flag, key in (
            ('--created-after', 'created_after'),
            ('--created-before', 'created_before'),
            ('--username', 'username'),
            ('--target-list', 'target_list'),
        )
        if options.get(key)
    ]
    if ignored:
        raise CommandError(
            f'{", ".join(ignored)} require --proposal; the watched-list sweep takes its '
            'overrides from each WatchedProposal row.'
        )
```

(Or pass the window through to every row's `sweep_proposal()` call — but then say so in the
help text and the runbook. Silently discarding is the one option that is not acceptable.)

## Warnings

### WR-01: The crontab's `|| echo "... lock held"` tail mislabels every failing tick as a skipped one

**File:** `deploy/cron/fomo.crontab.example:27`, `solsys_code/management/commands/check_unattended.py:170-171`
**Issue:** `run_unattended` exits 1 whenever any step failed (`run_unattended.py:57-58`), and
`flock` propagates the command's exit status. The `||` tail therefore fires on *any* non-zero
exit — a failing tick, an uncaught traceback, a missing interpreter, a missing `flock` binary —
and appends `run_unattended skipped: lock held` for a tick that in fact ran and failed. With a
persistent step failure the log fills with one false "lock held" line every 15 minutes, which
`docs/runbooks/telescope_runs_calendar.rst:1972-1996` tells the operator to diagnose as a stuck
process (and to delete the lock file for). This directly undermines SCHED-09's "failures are
visible" goal in the one place an operator looks first.
**Fix:** give lock contention its own exit code and test for it:

```
*/15 * * * * /usr/bin/flock -n -E 99 /var/lock/fomo/run_unattended.cron.lock <python> <manage.py> run_unattended >> /var/log/fomo/unattended.log 2>&1; \
  [ $? -eq 99 ] && echo "$(date -Is) run_unattended skipped: lock held" >> /var/log/fomo/unattended.log
```

Mirror the change in `cron_line()` and in the runbook's troubleshooting section.

### WR-02: A failed or unsent notification is still recorded as "staff were notified"

**File:** `solsys_code/unattended.py:464-474` and `:517-526`
**Issue:** `_send_notification()` swallows every send exception (logging the class name only)
and discards `notify_staff()`'s return value, which is `False` when there is no recipient at
all (`notifications.py:71-73`). `run_tick()` then unconditionally calls
`save_state(failing_steps, now)` for a `'failure'`/`'reminder'` decision. So on the *first*
failing tick, if the SMTP relay is down (or every staff user's email was cleared since
`check_unattended` last ran), nobody is told and the state file records `notified_at=<now>` —
suppressing all further mail for the same failing set for 24 hours, and again for each
reminder window if the outage persists. `test_mail_failure_never_raises`
(`test_unattended.py:231-248`) asserts only that the tick survives, not that the notification
is retried.
**Fix:** only record the notification when it was actually attempted *and* delivered:

```python
def _send_notification(decision, results) -> bool:
    subject, body = _build_notification_body(decision, results)
    try:
        return notifications.notify_staff(subject, body, fail_silently=False)
    except Exception as exc:  # noqa: BLE001
        logger.error('failed to send unattended notification: %s', type(exc).__name__)
        return False
...
sent = _send_notification(decision, results)
if sent and decision in ('failure', 'reminder'):
    save_state(failing_steps, now)
elif sent and decision == 'recovered':
    save_state([], None)
```

### WR-03: The suppression-state file is not fail-safe — three inputs abort the tail of every tick, permanently

**File:** `solsys_code/unattended.py:354-371`, `:394-412`, `:517-529`
**Issue:** `load_state()`'s docstring promises "A missing or unparseable file is treated as
'no prior failure' -- never an exception out of `run_tick()`", but it only catches
`(OSError, ValueError)` around `json.load`, and the notification block in `run_tick()` sits
outside any `try`. Three inputs were confirmed by executing the real functions against a
temporary `FOMO_STATE_DIR`:

```
load_state(list)             RAISED: AttributeError 'list' object has no attribute 'get'   # file contains [1,2]
decide_notification          RAISED: ValueError Invalid isoformat string: 'not-a-date'
decide_notification(naive)   RAISED: TypeError can't subtract offset-naive and offset-aware datetimes
```

`save_state()` raising (unwritable/full `FOMO_STATE_DIR`) has the same effect. In every case
the exception escapes `run_tick()` *after* the four steps have already run: the END banner is
never written, the `/<exit-code>` heartbeat ping is never sent, and the condition repeats on
every subsequent tick until a human notices the traceback. `FOMO_STATE_DIR` defaults to
`FOMO_LOCK_DIR` (`settings.py:413-415`), a directory the preflight only checks for
writability — not for exclusive ownership — so a hand-edit or a foreign writer is a realistic
trigger.
**Fix:** validate in `load_state()` and isolate the whole notification block:

```python
    if not isinstance(data, dict):
        return {'failing_steps': [], 'notified_at': None}
    raw = data.get('notified_at')
    try:
        notified_at = datetime.fromisoformat(raw) if raw else None
    except (TypeError, ValueError):
        notified_at = None
    if notified_at is not None and notified_at.tzinfo is None:
        notified_at = notified_at.replace(tzinfo=dt_timezone.utc)
```

(returning the parsed datetime, not the raw string), and wrap the
`load_state`/`decide_notification`/`_send_notification`/`save_state` block in
`try/except Exception` that logs the class name — the same discipline every other failure
path in this module already follows — so the END banner and the exit-code ping always run.

### WR-04: The END banner reports the tick's *start* time, so no tick's duration is readable

**File:** `solsys_code/unattended.py:493`, `:529`, `:425-431`
**Issue:** `now = datetime.now(dt_timezone.utc)` is captured once before the lock is taken and
reused for `_write_banner('END', now, exit_code=exit_code)`. Every tick's END line therefore
carries the identical timestamp as its START line. The runbook sells this banner as the
primary log diagnostic ("Every tick writes a START/per-step/END banner with a timestamp, so a
single tick is readable in isolation", `telescope_runs_calendar.rst:1520-1523`) — but the one
question an operator asks of it, "how long did this tick take / did it overrun its 15-minute
window?", is exactly what it cannot answer. The same stale `now` is also what
`decide_notification()` compares against `_REMINDER_INTERVAL`, so a long tick's reminder timing
drifts by the tick's own duration.
**Fix:**

```python
            _write_banner('END', datetime.now(dt_timezone.utc), exit_code=exit_code)
```

and pass a freshly-sampled time into `decide_notification()`/`save_state()`.

### WR-05: `cron_line()` hardcodes `/usr/bin/flock` while `check_flock()` resolves the real path

**File:** `solsys_code/management/commands/check_unattended.py:53-63` and `:170`
**Issue:** `check_flock()` reports `shutil.which('flock')` ("found at {path}") and passes the
hard check for a `flock` found anywhere on `PATH`, but the printed cron line always says
`/usr/bin/flock`. On a host where `flock` lives elsewhere (a non-merged-`/usr` layout, a
conda/venv-provided `util-linux`, a container image with it in `/bin` only), the preflight
says "ok" and hands the operator a cron line that cannot start — and, because of WR-01, the
resulting failure is logged as `run_unattended skipped: lock held`. `test_line_matches_the_committed_template_shape`
(`test_check_unattended.py:171-182`) pins the hardcoded string rather than the resolved one.
**Fix:** `flock_path = shutil.which('flock') or '/usr/bin/flock'` in `cron_line()`, and assert
in the test that the printed path is the resolved one.

### WR-06: The writability preflight cannot check what the deploy docs claim it checks

**File:** `solsys_code/management/commands/check_unattended.py:66-84`, `deploy/logrotate/fomo.example:6-8`
**Issue:** `_check_directory_writable()` calls `os.access(path, os.W_OK)`, which answers "can
*this* process's uid write here" — not the cron account's. `deploy/logrotate/fomo.example`
states `check_unattended` "verifies that directory is writable by the user the runner will
actually run as", and the runbook's setup step 4 implies the same. An operator running the
preflight as root (the natural thing to do while creating `/var/lock/fomo` and `/var/log/fomo`)
gets `[ok]` for directories the unprivileged cron account cannot write at all, because
`os.access` is effectively unconditional for uid 0. The check also cannot distinguish "writable
because the service account owns it" from "writable because it is mode 0777", which matters for
the lock and suppression-state files the runner keeps there.
**Fix:** report the resolved owner/mode alongside the verdict and say what was actually tested,
e.g. `detail=f'{path} writable by uid {os.geteuid()} (owner uid {path.stat().st_uid}, mode {oct(path.stat().st_mode & 0o777)})'`,
and correct the claim in `deploy/logrotate/fomo.example` and the runbook to "writable by the
account you run this check as — run it as the cron account".

### WR-07: The campaign-submission notice silently loses the real host in its approval-queue link

**File:** `solsys_code/campaign_views.py:338-343`, `src/fomo/settings.py:409`, `solsys_code/notifications.py:29-42`
**Issue:** The rewire replaced `self.request.build_absolute_uri(reverse(...))` — which always
produced a link on the host the submitter actually used — with
`notifications.absolute_url(reverse(...))`, i.e. `settings.FOMO_BASE_URL`, which defaults to
`'http://localhost:8000'`. On any deployment where `FOMO_BASE_URL` is not exported (it is
documented only for the *cron* environment: `crontab.example:16`, runbook step 3), every
SUBMIT-05 staff notification now carries an unusable `http://localhost:8000/campaigns/...`
link. Nothing catches this: `check_unattended` has no `FOMO_BASE_URL` check (it checks
`FOMO_HEARTBEAT_URL`, the lock/log dirs, mail, and watched proposals only), and no test in
`test_campaign_submission.py` asserts the link's host. The same default silently degrades the
Phase 36 failure email's own Admin/Calendar links (`unattended.py:459-460`).
**Fix:** add a preflight check (soft at minimum, hard if the runner is expected to mail links):

```python
def check_base_url() -> CheckResult:
    is_default = settings.FOMO_BASE_URL.rstrip('/') == 'http://localhost:8000'
    return CheckResult(
        name='FOMO_BASE_URL',
        ok=not is_default,
        hard=False,
        detail='FOMO_BASE_URL: still the localhost dev default -- emailed links will not work off this host'
        if is_default
        else 'FOMO_BASE_URL: set',
    )
```

and assert the host in a `TestStaffNotification` case. Guard `absolute_url()` against a `None`
setting too — `settings.FOMO_BASE_URL.rstrip('/')` raises `AttributeError` if a
`local_settings.py` sets it from an unset env var, and that exception escapes `run_tick()` via
`_build_notification_body()`, which is called outside `_send_notification()`'s `try`.

### WR-08: The status-refresh step doubles portal traffic on every failure, uncapped

**File:** `solsys_code/unattended.py:176-207`
**Issue:** `facility.update_all_observation_statuses()` already calls
`update_observation_status()` once per non-terminal record and returns the failures
(`tom_observations/facility.py:567-579`); `_refresh_one_facility()` then calls
`update_observation_status(observation_id)` a *second* time for each failed id, purely to name
the exception class. During a portal outage or an auth failure — precisely when this step
fails — every non-terminal LCO record fails, so the tick issues 2N portal requests with no
cap, no backoff, and no per-tick time budget, on a 15-minute schedule. A long enough tick
overruns its window and the next one is skipped by the cron `flock`, which is the "stuck tick"
state the runbook warns about. The re-check also mutates rows (it is the same write path as
the first attempt), so it is not the read-only probe the docstring's framing suggests.
**Fix:** cap the re-check (e.g. `for observation_id, _message in failed_records[:_MAX_RECHECKS]:`,
with the omitted count in the summary), or skip it entirely when
`len(failed_records)` exceeds a threshold — a whole-facility outage does not need N class
names, one is enough.

## Info

### IN-01: The heartbeat ping ignores the HTTP status code

**File:** `solsys_code/unattended.py:132-135`
**Issue:** `requests.get(...)` is issued and its response discarded — a `404`/`410` from a
deleted, mistyped, or expired healthchecks check is indistinguishable from a successful ping
in the log, so the "second visibility layer" can be silently dead while everything reports
healthy.
**Fix:** `response = requests.get(...)` then
`if response.status_code >= 400: logger.warning('heartbeat ping returned HTTP %s', response.status_code)`
— status codes are not credentials, so this is SCHED-10-safe.

### IN-02: Unattended discovery discards every per-request skip reason

**File:** `solsys_code/unattended.py:313-318`, `solsys_code/management/commands/backfill_lco_observations.py:463-466`
**Issue:** `step_discovery()` calls `sweep_proposal()` with no `stdout`/`stderr`, so the
function's default `io.StringIO()` sinks swallow every `Skipping request ...: <reason>` line.
The only surviving signal is the `skipped: N` counter in `last_run_summary`. An operator asking
"why did the tick skip 12 requests?" has nowhere to look — the reasons (no named target, bad
orbital-element scheme, no usable instrument_type) contain no credentials and would be safe to
log.
**Fix:** pass a sink that forwards to `logger.info`, or have `sweep_proposal()` log the skip
reasons itself in addition to writing them.

### IN-03: The suppression-state file is written non-atomically and with default permissions

**File:** `solsys_code/unattended.py:383-391`
**Issue:** `open('w')` + `json.dump` truncates first; a crash or a full filesystem mid-write
leaves a truncated file (recovered, but only because `JSONDecodeError` subclasses `ValueError`
— see WR-03 for the inputs that are not recovered). The file and the lock files are also
created with the process umask in a directory only checked for writability, so on a host where
`FOMO_LOCK_DIR` ends up group/world-writable another local account can hold the runner's lock
indefinitely or rewrite the suppression state.
**Fix:** write to `state_path.with_suffix('.tmp')` and `os.replace()` onto the final path, and
`os.chmod(state_path, 0o600)`.

### IN-04: `run_unattended`'s command docstring still describes the tracer-slice step set

**File:** `solsys_code/management/commands/run_unattended.py:20-24`
**Issue:** "Run one unattended tick: reconcile every CampaignRun's calendar projection (and,
**from later plans in this phase**, refresh LCO/SOAR observation statuses, sweep the
observation projector, and discover newly-scheduled observations...)" — all four steps shipped;
the parenthetical is stale, and `--help` shows this text to operators.
**Fix:** restate as the four steps in `STEPS` order.

### IN-05: A whole-facility outage is reported as "failed 1", and a clean run emits a dangling "classes:"

**File:** `solsys_code/unattended.py:191-194`, `:240`
**Issue:** When `update_all_observation_statuses()` raises outright, `_refresh_one_facility()`
returns `1` — so "LCO: failed 1" means either one bad observation or the entire facility being
unreachable, and the failure email cannot distinguish them. On a clean run the summary ends
with the empty fragment `classes: `.
**Fix:** return a sentinel (e.g. `-1`, or a separate `facility_down` flag) and render it as
`LCO: facility unreachable (HTTPError)`; drop the `classes:` fragment when the list is empty.

### IN-06: `run_tick(only_step='typo')` runs nothing and reports a healthy tick

**File:** `solsys_code/unattended.py:502-504`
**Issue:** `tuple(entry for entry in STEPS if entry[0] == only_step)` yields an empty tuple for
an unknown name, so `run_tick()` logs START/END, computes `exit_code=0`, and returns success
having done nothing. Today only argparse `choices` (`run_unattended.py:40`) prevents this, and
`run_tick()` is a public module-level function that other callers (tests, a future scheduler
shim) can call directly.
**Fix:** `raise ValueError(f'unknown step {only_step!r}')` when the filtered tuple is empty.

---

_Reviewed: 2026-09-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
