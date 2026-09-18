---
phase: 36-unattended-operation
reviewed: 2026-09-18T00:00:00Z
depth: deep
iteration: 6
diff_base: 0f3d10e078c16cb738c8779b2c5592200b2fef61
files_reviewed: 12
files_reviewed_list:
  - deploy/cron/fomo.crontab.example
  - docs/conf.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - solsys_code/tests/test_check_unattended.py
  - solsys_code/tests/test_settings_api_key_fold.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/unattended.py
  - src/fomo/settings.py
findings:
  critical: 2
  warning: 6
  info: 8
  total: 16
status: issues_found
---

# Phase 36: Code Review Report (iteration 6)

**Reviewed:** 2026-09-18
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

This is an incremental, adversarial review of the iteration-5 fix commits (CR-03, CR-04,
WR-16..WR-32, IN-17..IN-32) plus the two WR-16/WR-17 runbook commits, scoped to the diff
against `0f3d10e`.

**Verification of the iteration-5 fixes.** Each claimed fix was traced in the current
source, not taken from the fix report:

- CR-03 (`docs/conf.py`) — `autoapi_ignore` now carries `*/local_settings.py`; `autoapi_dirs`
  is `['../src']` and `sphinx.ext.viewcode` only renders modules autoapi documents, so the
  exclusion does close the path. Confirmed `src/fomo/local_settings.py` exists on this
  checkout (gitignored), so the exposure was real here.
- WR-16 (cron line) — `cron_line()` and `deploy/cron/fomo.crontab.example` both carry
  `[ $rc -eq 99 ] && { echo ...; rc=0; }; exit $rc`; the new shell-level test proves
  0/1/99 → 0/1/0. Fix is correct (but see WR-37 for the drift guard gap).
- WR-18 (discovery log level) — `logger.info` confirmed, and `settings.LOGGING` does pin the
  root logger to `INFO`, so the stated rationale holds.
- WR-19 (email backends), WR-20 (flock probe), WR-28/WR-32 (settings guards), WR-31
  (facility credentials), IN-19 (singular/plural verb), IN-20/IN-22/IN-23 — all present and
  behaving as described. `ImportError.name` for a missing `fomo.local_settings` was verified
  empirically to be `'fomo.local_settings'`, so the WR-32 guard does not break a stock dev
  checkout.
- Quality gates: `pre-commit run ruff --all-files` and `ruff-format --all-files` both pass;
  `python manage.py test solsys_code.tests.test_unattended solsys_code.tests.test_check_unattended
  solsys_code.tests.test_settings_api_key_fold solsys_code.tests.test_backfill_lco_observations`
  → 160 tests, OK.
- Paired docs: the `backfill_lco_observations.py` message change is reflected in a freshly
  re-executed `backfill_lco_observations_demo.ipynb` (the `--created-after requires --proposal`
  output cell changed); the runbook's "How do I run everything unattended?" section was
  extended for WR-16, WR-17, WR-31, CR-03 and the heartbeat knobs. The paired-docs rule is
  satisfied for this iteration, with the inaccuracies noted in CR-05 and IN-39.

**Key concerns.** The WR-17 fix — the highest-consequence change in this iteration — solved
the "re-mail every 15 minutes" failure by moving the suppression state into a fixed,
predictable filename in the shared system temp directory and by having `load_state()` trust
whichever of the two files has the newest mtime. That trades a noisy failure mode for a
silent one: the runner's alerting state is now readable, creatable and (via `utime`)
rank-controllable by any local account on the host, and by any second FOMO deployment or
test run sharing the same `/tmp`. Separately, `load_state()` and `decide_notification()`
disagree about what a state file with an unparseable `notified_at` means, and the disagreement
resolves to "never mail again for this failing set" — the exact outcome this phase exists to
prevent.

## Critical Issues

### CR-05: WR-17's fallback state file is a predictable, world-writable path whose mtime alone decides which suppression state the runner trusts

**File:** `solsys_code/unattended.py:64` (`_FALLBACK_STATE_PATH`), `solsys_code/unattended.py:432-448`
(`_newest_existing_state_path`), `solsys_code/unattended.py:561-598` (`save_state`);
`docs/runbooks/telescope_runs_calendar.rst:1458-1483`

**Issue:** `_FALLBACK_STATE_PATH` is `Path(tempfile.gettempdir()) / 'fomo-unattended-state.fallback.json'`
— a constant, guessable name in a directory that on every normal Linux host is mode `1777`
(world-writable, sticky). `load_state()` now calls `_newest_existing_state_path()`, which picks
whichever of the primary and fallback files has the larger `st_mtime` and reads it
unconditionally — there is no ownership check, no `S_ISREG`/`S_ISLNK` check, and no "only
consult the fallback while the primary is actually unwritable" condition. Verified on this
host: `unattended._FALLBACK_STATE_PATH` resolves to `/tmp/fomo-unattended-state.fallback.json`.

Consequences, in order of severity:

1. **Local tampering with the alerting state (integrity).** Any unprivileged local account —
   a threat actor this phase's own threat model already admits (T-36-02 reasons about a local
   `ps aux`) — can create that file before FOMO does and then set an arbitrary (e.g.
   far-future) mtime with `os.utime`, guaranteeing it always outranks the real state file.
   Writing `{"failing_steps": ["status_refresh","project_sweep","discovery","reconcile"],
   "notified_at": "2099-01-01T00:00:00+00:00"}` silences the failure mail for that set forever
   (`decide_notification()` sees an unchanged set, and `now - notified_at` is negative so the
   24-hour reminder never fires). Writing JSON garbage instead flips it the other way: every
   tick reads "no prior failure", decides `'failure'`, and mails staff every 15 minutes — an
   email-amplification vector. The runner cannot self-heal either: `/tmp`'s sticky bit forbids
   renaming over or unlinking another user's file, so both `_atomic_write_json()`'s
   `os.replace()` and `save_state()`'s `_FALLBACK_STATE_PATH.unlink()` fail with `EPERM`, and
   the unlink failure is swallowed by `with suppress(OSError)`.
2. **Tmp reapers.** `systemd-tmpfiles` ships a default `d /tmp 1777 root root 10d` rule; an
   outage lasting longer than the age threshold silently loses the state, re-arming the exact
   D-11 failing-open loop WR-17 was written to stop.
3. **The runbook overstates the safety of this.** The new paragraph ends "Both files are
   written atomically and with mode 0600, so a state file living in the shared system temp
   directory is not a new exposure." `0600` only addresses *confidentiality* of a file this
   process created; it says nothing about another user creating that name first, or about
   mtime being the sole arbiter. As written, an operator reading the runbook has no reason to
   check `/tmp` at all.

**Fix:** stop trusting a shared-directory path by name alone. Minimum viable fix — make the
fallback per-deployment and per-uid, and validate it before reading:

```python
import hashlib, os, stat

def _fallback_state_path() -> Path:
    # One fallback per (deployment, uid) instead of one per host, so two FOMO instances
    # (staging/prod, or a test run beside a live cron) can never read each other's state.
    tag = hashlib.sha256(str(settings.BASE_DIR).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f'fomo-unattended-state.{os.geteuid()}.{tag}.fallback.json'


def _fallback_is_trustworthy(path: Path) -> bool:
    try:
        info = path.lstat()                       # lstat: never follow a planted symlink
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not (info.st_mode & 0o077)
```

and have `_newest_existing_state_path()` skip any fallback that fails
`_fallback_is_trustworthy()`. Better still, prefer a directory this process owns
(`Path(settings.FOMO_LOCK_DIR).parent`, `~/.cache/fomo/`, or `$XDG_RUNTIME_DIR`) over
`tempfile.gettempdir()`. Then correct the runbook paragraph: say the fallback is
uid-scoped and validated, and that a fallback file the runner does not own is ignored rather
than trusted.

### CR-06: a state file with a valid `failing_steps` but an unparseable `notified_at` permanently suppresses both the failure mail and the 24-hour reminder

**File:** `solsys_code/unattended.py:451-502` (`load_state`), `solsys_code/unattended.py:601-631`
(`decide_notification`)

**Issue:** `load_state()`'s docstring promises that "a `notified_at` that is not a valid
ISO-8601 string is treated as 'no prior failure'". The code does not do that — it nulls
`notified_at` only, and still returns the parsed `failing_steps`:

```python
try:
    notified_at = datetime.fromisoformat(raw_notified_at) if raw_notified_at else None
except (TypeError, ValueError):
    notified_at = None
...
return {'failing_steps': sorted(failing_steps), 'notified_at': notified_at}
```

`decide_notification()` then reaches:

```python
if failing_steps:
    if failing_steps != previous_failing:
        return 'failure'
    if notified_at is not None and (now - notified_at) >= _REMINDER_INTERVAL:
        return 'reminder'
    return None
```

With `previous_failing == failing_steps` and `notified_at is None`, the reminder branch is
unreachable and the function returns `None` — *forever*, for as long as that failing set
persists. No failure email, no reminder, and (because `save_state()` is only called when a
notification was actually sent) nothing ever rewrites the offending file. Reproduced directly
against the module:

```
>>> load_state()                       # file: {"failing_steps": ["reconcile"], "notified_at": "not-a-date"}
{'failing_steps': ['reconcile'], 'notified_at': None}
>>> decide_notification({'failing_steps': ['reconcile'], 'notified_at': None}, ['reconcile'], now)
None
```

`run_tick()` never produces such a file itself (it always passes a real `end_time` alongside a
non-empty set), but it is reachable by: the CR-05 fallback path (any local account, or a
second deployment, can write one); a hand edit — the runbook's new troubleshooting section
now points operators at these files by name; a truncated/legacy file written by an older or
newer version of this module; and the `{"failing_steps": [...]}`-only model that
`load_state()`'s own `data.get('notified_at')` explicitly tolerates. The impact is the total
loss of the phase's primary deliverable (staff learn about a failing unattended pipeline),
with no log line saying so.

**Fix:** make the two functions agree, and fail *loud* rather than silent. Either honour the
docstring in `load_state()`:

```python
    raw_notified_at = data.get('notified_at')
    try:
        notified_at = datetime.fromisoformat(raw_notified_at) if raw_notified_at else None
    except (TypeError, ValueError):
        # An unparseable timestamp makes the whole record untrustworthy: keeping
        # failing_steps without it wedges decide_notification() on "same set, never
        # notified" -- no mail, and no reminder either. Treat it as no prior failure.
        logger.warning('unattended state file has an unparseable notified_at -- ignoring the whole record')
        return {'failing_steps': [], 'notified_at': None}
```

or, belt-and-braces, close the hole in `decide_notification()` too:

```python
    if failing_steps:
        if failing_steps != previous_failing:
            return 'failure'
        if notified_at is None:
            return 'failure'          # previously-recorded failure with no send time: treat as due now
        if (now - notified_at) >= _REMINDER_INTERVAL:
            return 'reminder'
        return None
```

Add a test for each half — a state file with a bad `notified_at` must still mail, and a
future-dated `notified_at` must not suppress indefinitely.

## Warnings

### WR-33: the fallback state path has no per-deployment discriminator, so two FOMO instances on one host silently share one suppression state

**File:** `solsys_code/unattended.py:64`

**Issue:** `_FALLBACK_STATE_PATH` is a module-level constant derived only from
`tempfile.gettempdir()`. Two checkouts on the same host — the common staging + production
pairing, or a developer checkout beside a live cron deployment — resolve to the identical
path even though their `FOMO_STATE_DIR`s are different. If either one's primary state
directory goes bad, `load_state()` in *both* processes may read that file (whenever it is the
newer of the two), so one deployment's failing-step set can suppress or trigger the other's
staff mail. Nothing in the code or the runbook flags this.

**Fix:** as in CR-05 — fold `settings.BASE_DIR` (or `FOMO_STATE_DIR`) and `os.geteuid()` into
the filename, and make it a function rather than an import-time constant so `override_settings`
can move it in tests.

### WR-34: the new WR-17 test reads, writes and deletes the real `/tmp` fallback path instead of an isolated one

**File:** `solsys_code/tests/test_unattended.py:223-253`
(`test_unwritable_state_dir_after_setup_still_suppresses_repeat_mail`)

**Issue:** the test takes `fallback_path = unattended._FALLBACK_STATE_PATH` — the real
`/tmp/fomo-unattended-state.fallback.json` — unlinks it up front, lets the runner write to it,
and unlinks it again in cleanup. Every other filesystem-touching test in this module and in
`test_check_unattended.py` works inside a `TemporaryDirectory()`; this one does not. Running
`python manage.py test` on a host that also runs the cron schedule therefore destroys that
host's live suppression state (whose only purpose is to stop a failure email storm), and two
concurrent test runs — or a test run concurrent with a real tick — race on the same file. The
test also loses its own premise when run as root (`chmod 0500` does not stop uid 0 writing),
where it fails on `assertTrue(fallback_path.exists())` for a reason unrelated to the behavior
under test.

**Fix:** make the fallback location injectable (`unattended._fallback_state_path()` reading a
setting, or simply `patch.object(unattended, '_FALLBACK_STATE_PATH', Path(tmp.name) / 'fb.json')`)
and point this test at a `TemporaryDirectory()`. Add
`@skipIf(os.geteuid() == 0, 'unwritable-directory tests are meaningless as root')` to this test
and to `_make_unwritable_parent`'s users.

### WR-35: `check_flock()`'s IN-23 "confirm this is the flock you want" result is reported as `[ok]` and never reaches stderr, so the warning is invisible

**File:** `solsys_code/management/commands/check_unattended.py:137-147`, consumed at
`check_unattended.py:538-548`; pinned by `solsys_code/tests/test_check_unattended.py:118-134`

**Issue:** the new branch returns `CheckResult(..., ok=True, hard=True, detail='... resolved
outside the usual system directories ... confirm this is the flock you want a service crontab
to run')`. `Command.handle()` maps `ok=True` to the literal status `ok` and only mirrors
non-`ok` lines to stderr:

```python
line = f'[{status}] {result.name}: {result.detail}'
self.stdout.write(line)
if status != 'ok':
    self.stderr.write(line)
```

So the one signal IN-23 exists to raise is printed as `[ok] flock: ...` in the middle of a
clean run, is absent from stderr, and is missed by any operator or wrapper that greps for
`FAIL`/`WARN` or watches stderr — which is precisely the "operator pastes the printed line
into a persistent crontab" workflow the finding was about. `hard=True` alongside `ok=True` is
also inert (`hard` is only consulted when `ok` is false). The new test asserts `[ok] flock`,
which locks the ineffective behavior in.

**Fix:** return `ok=False, hard=False` so it renders as `[WARN] flock: ...` and is echoed to
stderr — matching how every other advisory result in this command (`heartbeat`,
`FOMO_BASE_URL`, `facility_credentials`, `watched_proposals`) is surfaced — and update the
test to assert `[WARN] flock` plus its presence in stderr.

### WR-36: the 15-minute tick interval was re-duplicated in the same iteration that introduced the single-owner rule for shared constants

**File:** `solsys_code/unattended.py:55` (`_CRON_TICK_INTERVAL = timedelta(minutes=15)`) and
`solsys_code/management/commands/check_unattended.py:57` (`_CRON_INTERVAL_MINUTES = 15`)

**Issue:** IN-22's fix moved `_DEFAULT_LOCK_DIR`/`_DEFAULT_LOG_FILE` into `unattended.py` and
had `check_unattended.py` import them, explicitly because "a future change ... could silently
desynchronize" duplicated literals, and IN-20's comment declares `_CRON_INTERVAL_MINUTES` "the
single source for the schedule's own interval". The same iteration then added a *second*
independent copy of that same interval in `unattended.py`, in a module `check_unattended.py`
already imports — so nothing stops `*/10` in the cron line from coexisting with a 15-minute
temp-file reap threshold, and the reap comment's justification ("a full tick interval") would
silently become false.

**Fix:** keep one owner. Either define `_CRON_INTERVAL_MINUTES = 15` in `unattended.py` next
to `_DEFAULT_LOCK_DIR` and derive `_CRON_TICK_INTERVAL = timedelta(minutes=_CRON_INTERVAL_MINUTES)`
there, with `check_unattended.py` importing the minutes constant the same way it already
imports the two path defaults; or import `_CRON_TICK_INTERVAL` into `check_unattended.py` and
use `int(_CRON_TICK_INTERVAL.total_seconds() // 60)`.

### WR-37: the cron-line/template drift guard does not cover the `rc=0` normalization that WR-16 just added

**File:** `solsys_code/tests/test_check_unattended.py:447-471`
(`test_line_matches_the_committed_template_token_for_token`), and
`test_check_unattended.py:431-445`

**Issue:** the parity test's token list is `('-n', '-E 99', '.cron.lock', '>>', '2>&1',
'rc=$?', '[ $rc -eq 99 ]', 'lock held', 'exit $rc')` — it contains neither `rc=0` nor the
`{ ... ; }` grouping that makes WR-16's normalization work. The committed template could lose
`rc=0` (reverting to the "a routine tick overlap looks like a failure" behavior WR-16 fixed)
and every test would still pass. The companion shell-level test,
`test_lock_held_exit_is_normalized_to_zero`, splices a stub into `cron_line()`'s output only —
it never executes the committed template line — so the template half of the fix has no
executable coverage at all, in a phase whose CR-01/WR-01/WR-09 history is entirely about these
two artifacts drifting apart.

**Fix:** add `'rc=0'` (and ideally `'; }'`) to the token tuple, and extend
`test_lock_held_exit_is_normalized_to_zero` to run the same 0/1/99 matrix against the
`*/15` line read from `deploy/cron/fomo.crontab.example`, with the same stub splice.

### WR-38: the WR-32 settings import guard is executed by no test

**File:** `src/fomo/settings.py:436-443`; `solsys_code/tests/test_settings_api_key_fold.py:42-83`

**Issue:** the new guard changes settings-import control flow:

```python
except ImportError as exc:
    if exc.name != 'fomo.local_settings':
        raise
```

No test reaches it. `_FoldExecutionTestCase._run_fold()` always injects a working
`types.ModuleType('fomo.local_settings')` into `sys.modules` before exec'ing the tail, so the
`except` branch is dead in the suite; and this checkout has a real `src/fomo/local_settings.py`,
so even the ordinary settings import at test start takes the success path. The only new test
that touches this region, `TestFoldTailUsesStarImportIntoOwnNamespace`, is a source-token grep
(`assertIn('from fomo.local_settings import *')`, `assertNotRegex(r'except\s*:')`) and would
pass unchanged if the guard's condition were inverted. This is the same regression model as
WR-29 (a settings change that kept every test green while a configured host failed at import),
which is what the iteration-5 fix report cites as the reason this area needs executable cases.

**Fix:** add two cases that exec the fold tail with `sys.modules['fomo.local_settings']`
removed and a temporary `sys.meta_path` finder raising: (a)
`ModuleNotFoundError('...', name='fomo.local_settings')` — must be swallowed, `FACILITIES`
untouched; (b) `ModuleNotFoundError('...', name='some_missing_dependency')` — must propagate
out of the exec. Both run with no real local settings module on disk.

## Info

### IN-33: `test_settings_api_key_fold.py`'s module docstring still advertises the test class WR-30 deleted

**File:** `solsys_code/tests/test_settings_api_key_fold.py:4-11`
**Issue:** the docstring says "They pin four behaviors: ... the bracketed dict-subscript form
the old runbook wrongly documented raises `NameError` ...". `TestBracketedDictSubscriptRaisesNameError`
was removed by the WR-30 fix and replaced by `TestFoldTailUsesStarImportIntoOwnNamespace`;
there are now six test classes and that behavior is no longer pinned anywhere. The next
reader is told coverage exists that does not.
**Fix:** rewrite the docstring to enumerate the six classes actually present, naming the
`from ... import *` mechanism rather than the deleted `NameError` case.

### IN-34: the read-only preflight now imports the entire runner graph for two string constants

**File:** `solsys_code/management/commands/check_unattended.py:38`
**Issue:** IN-22's dedup added `from solsys_code.unattended import _DEFAULT_LOCK_DIR,
_DEFAULT_LOG_FILE`, which pulls in `campaign_reconciler`, `observation_projector`,
`backfill_lco_observations`, `project_observation_calendar`, `telescope_runs` (astropy) and
both facility classes at import time — into a command whose whole contract is "read-only,
reports, does not fix". Verified that none of these currently reaches `ephem_utils` (the
1.6 GB SPICE path), so this is a cost and coupling concern, not a breakage — but the guard
against it is now one accidental import away in any of five modules.
**Fix:** move the two default-path constants into a leaf module (e.g. `solsys_code/constants.py`,
or `src/fomo/settings.py` itself as `FOMO_DEFAULT_LOCK_DIR`/`FOMO_DEFAULT_LOG_FILE`) that both
`unattended.py` and `check_unattended.py` import, so neither command depends on the other's
import graph.

### IN-35: `_NON_DELIVERING_EMAIL_BACKENDS` is an exact dotted-path denylist that any subclass evades

**File:** `solsys_code/management/commands/check_unattended.py:255-283`;
`solsys_code/tests/test_check_unattended.py:34-46`
**Issue:** the check is `_NON_DELIVERING_EMAIL_BACKENDS.get(settings.EMAIL_BACKEND)`, so a
`local_settings.py` that subclasses or re-exports `locmem`/`dummy` passes. The suite itself
demonstrates the evasion: `_FakeDeliveringEmailBackend(_LocmemEmailBackend)` exists precisely
to be reported as deliverable while behaving exactly like the backend the check rejects.
**Fix:** resolve the class (`django.utils.module_loading.import_string(settings.EMAIL_BACKEND)`)
and test `issubclass()` against the four non-delivering classes, falling back to the dotted-path
comparison if the import fails. The test helper would then need a genuinely different stand-in
(e.g. a thin subclass of `BaseEmailBackend` that appends to `mail.outbox`).

### IN-36: `_SYSTEM_BINARY_DIRECTORIES` omits `/usr/local/bin`, and the path is compared unresolved

**File:** `solsys_code/management/commands/check_unattended.py:62, 137`
**Issue:** `/usr/local/bin` is a standard system location on many hosts (and the default
install prefix for a source-built util-linux), so a legitimate `/usr/local/bin/flock`
produces the "outside the usual system directories" note — noise that trains operators to
ignore it. Conversely the check compares `shutil.which()`'s raw result, so a symlink at
`/usr/bin/flock` pointing into a user-writable directory passes silently.
**Fix:** add `/usr/local/bin` and `/usr/local/sbin` to the tuple, and compare
`str(Path(path).resolve())` so the note follows the real target.

### IN-37: the settings fold leaves its loop variable bound in the settings module namespace

**File:** `src/fomo/settings.py:459-460`
**Issue:** `for _facility in ('LCO', 'SOAR'): FACILITIES.setdefault(...)` leaves `_facility`
as a module-level name after the loop. Harmless today (Django's `Settings` only copies
`isupper()` names), but it makes the settings module's final namespace carry a stray non-setting.
**Fix:** `del _facility` after the loop, or use a tuple-unrolled pair of statements.

### IN-38: `step_discovery()` logs the whole captured sweep buffer as one INFO record

**File:** `solsys_code/unattended.py:401-404`
**Issue:** `logger.info('discovery %s: %s', sink_name, captured_text)` emits the entire
accumulated `StringIO` contents as a single multi-line log record. In a `--dry-run` preview
`sweep_proposal()` writes one "Would create/reuse ..." line per portal request, so a large
proposal produces one enormous log line that no line-oriented tool (grep, logrotate's
size accounting, journald's field limits) handles gracefully. Content itself is fine — the
lines carry only observation ids, target names and skip reasons, never a credential or a raw
exception message.
**Fix:** `for line in captured_text.splitlines(): logger.info('discovery %s: %s', sink_name, line)`.

### IN-39: the runbook's description of the heartbeat reminder still names only the ping interval

**File:** `docs/runbooks/telescope_runs_calendar.rst:1580-1583`
**Issue:** step 6 says `check_unattended` "reminds you that the check at the other end still
needs its own expected ping interval set". After IN-20 the detail string names both `Period`
*and* `Grace`; the runbook's summary of that same line still names one knob — the exact
one-knob/two-knob asymmetry IN-20 was filed about, now surviving in the doc instead of the code.
**Fix:** change to "...still needs its own expected ping interval *and* grace time set".

### IN-40: the template-parity test locates the template line by a hardcoded `*/15` while the code's schedule is now a constant

**File:** `solsys_code/tests/test_check_unattended.py:459-467`
**Issue:** the test finds the committed line with `stripped_line.startswith('*/15')` and never
compares the schedule field itself (it is not among the compared tokens). Changing
`_CRON_INTERVAL_MINUTES` to anything but 15 leaves the test comparing against a template line
that no longer matches the generated one, and it still passes — the drift the test exists to
catch.
**Fix:** derive the prefix from the constant
(`f'*/{check_unattended._CRON_INTERVAL_MINUTES}'`) and add that same token to the comparison list.

---

_Reviewed: 2026-09-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
