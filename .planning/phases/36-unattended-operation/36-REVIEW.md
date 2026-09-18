---
phase: 36-unattended-operation
reviewed: 2026-09-18T00:00:00Z
depth: deep
iteration: 7
diff_base: 8f9b045498eba33d32b09513697612e5de123051
files_reviewed: 8
files_reviewed_list:
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/constants.py
  - solsys_code/management/commands/check_unattended.py
  - solsys_code/tests/test_check_unattended.py
  - solsys_code/tests/test_settings_api_key_fold.py
  - solsys_code/tests/test_unattended.py
  - solsys_code/unattended.py
  - src/fomo/settings.py
findings:
  critical: 0
  warning: 6
  info: 6
  total: 12
status: issues_found
---

# Phase 36: Code Review Report (iteration 7)

**Reviewed:** 2026-09-18
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Incremental, adversarial review of everything since `8f9b045`: plan 36-09's G-36-5 gap
closure (one result line, one stream, with a stdout flush before each stderr write) plus
the iteration-6 fix commits (CR-05, CR-06, WR-33..WR-38, IN-33..IN-40).

**Verification of the iteration-6 fixes.** Every claimed fix was traced in the current
source and, where possible, exercised — not taken from the fix report:

- **CR-05 / WR-33 / WR-34** — `unattended.py:435-478`: the fallback state path is now a
  function returning `fomo-unattended-state.<euid>.<sha256(BASE_DIR)[:12]>.fallback.json`,
  and `_newest_existing_state_path()` only considers it when `_fallback_is_trustworthy()`
  passes (`lstat`, regular file, owner == euid, no group/other bits — exactly what
  `_atomic_write_json()`'s `chmod 0o600` produces). The symlink-planting and
  foreign-owner paths are genuinely closed; `test_unattended.py` now patches
  `_fallback_state_path` to a `TemporaryDirectory()` instead of mutating the real `/tmp`
  file. One residual is filed as IN-44.
- **CR-06** — `load_state()` (`unattended.py:553-566`) now discards the whole record on an
  unparseable `notified_at` and logs a warning; `decide_notification()`
  (`unattended.py:707-711`) independently treats "same failing set, `notified_at is None`"
  as `'failure'`. Both halves are covered by new tests, and the old
  partial-preservation assertion was correctly inverted rather than deleted.
- **WR-35** — the non-system-`flock` note is now `ok=False, hard=False`, so it renders as
  `[WARN]` and reaches stderr. Correct in code; the operator doc was not updated to match
  (WR-42).
- **WR-36 / IN-34** — `solsys_code/constants.py` is a real leaf module (no project-local
  imports); `grep` confirms both `unattended.py` and `check_unattended.py` import the three
  literals from it and that `check_unattended.py` no longer imports `unattended.py` at all,
  so the read-only preflight no longer drags in the runner graph.
- **WR-37** — the 0/1/99 shell matrix now runs against the committed
  `deploy/cron/fomo.crontab.example` line as well as `cron_line()`'s output, and `rc=0`
  and `; }` are compared tokens.
- **WR-38** — `_MissingModuleFinder` forces both halves of the
  `except ImportError as exc: if exc.name != 'fomo.local_settings': raise` guard; both
  branches are now executed.
- **IN-33, IN-35, IN-36, IN-38, IN-39, IN-40** — all present and behaving as described
  (module docstring now lists exactly the 8 public test classes that exist; `issubclass()`
  resolution catches a locmem subclass; `/usr/local/bin`+`/usr/local/sbin` added and the
  path is `resolve()`d; the discovery sweep logs one record per line; the runbook names
  both heartbeat knobs; the template-parity test derives its prefix from the constant).
- **IN-37** — `del _facility` is present in `settings.py:463`. The regression guard added
  for it does not actually guard (WR-44).
- **G-36-5 itself** — reproduced end to end. `python manage.py check_unattended >> log 2>&1`
  on this host emits each `[ok]`/`[WARN]`/`[FAIL]` line exactly once, in check order, with
  no interleaving: the duplicate warning line the operator reported is gone, and Django
  5.2's `OutputWrapper.flush()` really does delegate to the wrapped stream (it is defined
  explicitly, not inherited as an `IOBase` no-op), so the flush is not a no-op.
- **Quality gates** — `pre-commit run ruff --all-files` and `ruff-format --all-files` both
  pass. `python manage.py test solsys_code.tests.test_check_unattended
  solsys_code.tests.test_settings_api_key_fold` → 57 tests OK;
  `python manage.py test solsys_code.tests.test_unattended` → 68 tests OK.
  `sphinx-build` produces no new warning for the changed runbook hunks (one pre-existing
  one is filed as IN-43).

**Key concerns for this iteration.** The G-36-5 fix is correct for the loop it covers but
its ordering guarantee stops at the loop: on the failure path — the one the runbook's new
capture recipe is written for — the cron-line block still lands *after* the `CommandError`
in the merged log, reproduced below (WR-39). Separately, the IN-35 rewrite of the email
backend check left its import-failure branch reporting `[ok]` for a backend that cannot be
imported at all, which is the one misconfiguration whose runtime symptom is silently
swallowed by `_send_notification()` (WR-40), and made `issubclass()` reachable with a
non-class (WR-41) — the same "a read-only preflight must never abort" failure class WR-20
was filed about. Finally, three of this iteration's new regression guards cannot fail in
the environment that runs them (WR-43, WR-44, IN-41): they are green by construction in
CI, so the behaviors they claim to pin are still unpinned.

## Warnings

### WR-39: the G-36-5 ordering fix stops at the loop — on the failure path the cron-line block still lands after the `CommandError` in a merged log

**File:** `solsys_code/management/commands/check_unattended.py:616-638`
**Issue:** the new `self.stdout.flush()` (line 626) only orders stdout against the *result
lines* written inside the loop. The trailing stdout block — the blank separator, the
`Cron line to install...` header and `cron_line()` itself (lines 631-633) — is never
flushed before `raise CommandError(...)` (line 638). When stdout is a file (the crontab
template's `>> ... 2>&1`, and the runbook's own newly documented
`>> preflight.log 2>&1` recipe) it is block-buffered and only flushed at interpreter exit,
while Django's `run_from_argv` writes the `CommandError` to the line-buffered stderr
immediately. Reproduced on this host with
`FOMO_LOCK_DIR=/proc/nonexistent/sub python manage.py check_unattended > log 2>&1`:

```
[WARN] watched_proposals: no active WatchedProposal rows -- ...
CommandError: check_unattended: 2 hard check(s) failed: FOMO_LOCK_DIR, FOMO_STATE_DIR

Cron line to install (both host directories above must exist first):
*/15 * * * * /usr/bin/flock -n -E 99 ...
```

This is exactly the operator workflow the block exists for ("Printed even when a hard check
failed -- an operator fixing prerequisites still wants to see the target state"), and it
contradicts the runbook's new promise of a report captured "in one file, in check order".
The success path is correctly ordered, so only the failing run — the run an operator
actually reads — is inverted.
**Fix:** flush after the trailing block, before the raise:

```python
        self.stdout.write('')
        self.stdout.write('Cron line to install (both host directories above must exist first):')
        self.stdout.write(cron_line())
        # Ordering, as in the loop above: a CommandError goes to the line-buffered
        # stderr immediately, while this block sits in stdout's buffer until exit.
        self.stdout.flush()

        failed_hard = [result for result in results if result.hard and not result.ok]
```

### WR-40: `check_email()` reports `[ok] EMAIL_BACKEND` for a backend that cannot be imported at all, and the send-time failure it defers to is swallowed by design

**File:** `solsys_code/management/commands/check_unattended.py:311-329, 341-354`
**Issue:** `_classify_email_backend()` catches `ImportError` from `import_string(backend)`
and falls back to `_NON_DELIVERING_EMAIL_BACKENDS.get(backend)`. That dict only holds the
four Django-shipped backends, which always import — so the fallback branch can only ever
return `None`, and a typo'd or removed dotted path in `local_settings.py` (e.g.
`django.core.mail.backends.smpt.EmailBackend`) is reported as `[ok] EMAIL_BACKEND`, i.e.
"this backend can deliver". The docstring's rationale ("an unresolvable `EMAIL_BACKEND`
fails for its own reasons at send time, not here") does not hold for this phase: at send
time `_send_notification()` (`unattended.py:776-778`) catches the resulting `ImportError`
and logs the class name only, so the failure notice is lost with no operator-visible
signal — precisely the outcome this hard check exists to prevent. As a side effect
`_NON_DELIVERING_EMAIL_BACKENDS` is now dead code in every reachable path (grep confirms
its only reference is that unreachable branch), and no test covers it.
**Fix:** treat an unimportable backend as a hard failure rather than a pass:

```python
def _classify_email_backend(backend: str) -> str | None:
    try:
        backend_cls = import_string(backend)
    except ImportError:
        return 'cannot be imported -- a failure notice would raise ImportError at send time and be swallowed'
    ...
```

and delete `_NON_DELIVERING_EMAIL_BACKENDS`, with a test asserting a bogus dotted path
fails the command.

### WR-41: `issubclass()` on a non-class `EMAIL_BACKEND` raises `TypeError` and aborts the whole read-only preflight

**File:** `solsys_code/management/commands/check_unattended.py:326-328`
**Issue:** `issubclass(backend_cls, non_delivering_cls)` assumes `import_string()` returned
a class. Django's own `django.core.mail.get_connection()` does
`import_string(settings.EMAIL_BACKEND)(...)`, so any callable — a factory function, a
`functools.partial`, a module-level instance — is a legal `EMAIL_BACKEND`. Verified:
`issubclass(import_string('json.dumps'), Exception)` raises
`TypeError: issubclass() arg 1 must be a class`. `check_email()` is called from
`Command.handle()` (line 599) with no guard, so that `TypeError` propagates as an uncaught
traceback, losing the heartbeat, base-URL, credential and watched-proposal results and the
printed cron line. This is the identical failure class WR-20 was filed and fixed for in
`check_flock()`, reintroduced one check later.
**Fix:** guard the type before comparing:

```python
    if not isinstance(backend_cls, type):
        return None  # a callable factory: not one of the four backends we can classify
    for non_delivering_cls, reason in _NON_DELIVERING_EMAIL_BACKEND_CLASSES.items():
```

### WR-42: the runbook still enumerates exactly four warning conditions and calls `flock` a hard prerequisite, which WR-35 made untrue

**File:** `docs/runbooks/telescope_runs_calendar.rst:1607-1613`
**Issue:** step 6 says the preflight "exits non-zero only when a hard prerequisite (flock,
the directories, or email) is missing; an unset heartbeat URL, a default base URL, an
unconfigured LCO/SOAR portal API key, and an empty watched-proposal list are warnings, not
failures (the tick still runs, and mail still sends, without any of the four)." WR-35
added a fifth warning condition in this same review round: a `flock` that works and
supports `-E` but resolves outside `/usr/bin`, `/bin`, `/usr/sbin`, `/sbin`,
`/usr/local/bin`, `/usr/local/sbin` now renders as `[WARN] flock: ...` with `hard=False`.
An operator on a conda/venv-provided `flock` therefore sees a `[WARN] flock` line that the
runbook says cannot exist ("flock" is listed only as a hard prerequisite) and a closed list
of "the four" that does not include it. CLAUDE.md makes the affected `docs/runbooks/` page
part of the deliverable for a behavior change like this one, not a follow-up.
**Fix:** add the fifth case to the enumeration and drop the "the four" count, e.g.
"...an unset heartbeat URL, a default base URL, an unconfigured LCO/SOAR portal API key, a
`flock` that works but resolves outside the usual system directories, and an empty
watched-proposal list are warnings, not failures".

### WR-43: `test_merged_capture_has_no_escape_bytes` cannot fail in CI, so the prohibition it claims to pin is unpinned

**File:** `solsys_code/tests/test_check_unattended.py:558-565`
**Issue:** the test's own comment says it "pins the prohibition against ever passing that
[style_func] argument". It cannot. `BaseCommand.__init__` sets `self.style =
color_style()`, and `django.core.management.color.supports_color()` keys off
`sys.stdout.isatty()`. Verified on this host: with stdout piped (every CI run, and every
`manage.py test` whose output is redirected), `supports_color()` is `False` and
`color_style().ERROR('x')` returns plain `'x'` — so even a deliberate
`self.stderr.write(line, self.style.ERROR)` would emit no escape bytes and the test would
still pass. It only has teeth on a developer's interactive terminal.
**Fix:** make the styling explicit rather than ambient, e.g. assert against a forced style
(`with patch.object(command, 'style', color_style(force_color=True))`) or assert the
source-level prohibition directly (inspect `Command.handle`'s `self.stderr.write` calls for
a second positional argument), so the test fails wherever it runs.

### WR-44: the IN-37 regression guard is vacuous on any checkout without a `local_settings.py` defining `LCO_API_KEY`

**File:** `solsys_code/tests/test_settings_api_key_fold.py:145-161`
**Issue:** `test_settings_module_does_not_carry_the_loop_variable` asserts
`not hasattr(settings_module, '_facility')` against the live settings module. The fold
loop only runs when `'LCO_API_KEY' in globals()`, i.e. only when this host has a
`local_settings.py` that sets it. On a checkout without one — every CI run, and any
developer who has not created the file — the loop never executes, `_facility` is never
bound whether or not `del _facility` exists, and the test passes trivially. The guard is
green by construction in exactly the environment that is supposed to enforce it, which is
the same regression model WR-29/WR-38 were filed about. The module already has the tool to
do this deterministically: `_FoldExecutionTestCase._run_fold()` execs the real fold tail
into a synthetic namespace.
**Fix:** assert against the executed namespace instead of the live module — have
`_run_fold()` return (or expose) the namespace and assert
`self.assertNotIn('_facility', namespace)` after
`_run_fold({'LCO_API_KEY': _FAKE_LCO_API_KEY})`, which exercises the `del` on every host.

## Info

### IN-41: the new `self.stdout.flush()` is covered by no test that could fail without it

**File:** `solsys_code/tests/test_check_unattended.py:91-106, 519-556`
**Issue:** `_run_merged()` binds one `io.StringIO` as both sinks. A `StringIO` has no
buffering asymmetry, so writes appear in call order with or without the flush — the
merged-sink class pins the *deduplication* half of G-36-5 but not the *ordering* half the
flush exists for (see WR-39, which is a live instance of that untested half being wrong).
**Fix:** add one subprocess-level case that runs
`sys.executable manage.py check_unattended` with stdout and stderr redirected to the same
file and asserts the `[FAIL]`/`[WARN]` lines appear after the `[ok]` lines that preceded
them (and, once WR-39 is fixed, that the cron-line block precedes the `CommandError`).

### IN-42: the routing-contract test anchors on `[ok] flock`, which WR-35 made host-dependent

**File:** `solsys_code/tests/test_check_unattended.py:567-575`
**Issue:** `test_warning_and_passing_lines_route_to_separate_streams` asserts
`'[ok] flock' in stdout` with `shutil.which` unpatched. After WR-35, a host whose `flock`
comes from a conda/venv bin (a normal setup for this repo, per `.setup_dev.sh`) produces
`[WARN] flock` instead, and this test fails for a reason unrelated to the routing contract
it exists to pin.
**Fix:** anchor on a line the fixture determines, e.g. `[ok] staff_recipients` (the base
class always creates a staff user with an email), or patch `shutil.which` to
`/usr/bin/flock` as the neighbouring tests already do.

### IN-43: the runbook renders a docutils error into the published HTML at line 1562

**File:** `docs/runbooks/telescope_runs_calendar.rst:1562`
**Issue:** ``` ``_readthedocs/html/`` and ``docs/_build/html/`` are both ``.gitignore``d, ```
— the inline literal is immediately followed by a word character, which docutils rejects:
`sphinx-build` emits `WARNING: Inline literal start-string without end-string` and the
built page renders a literal ``` `` ``` hyperlinked to an error message instead of the
intended text (confirmed in
`_docs/html/runbooks/telescope_runs_calendar.html`, `<span class="problematic">``</span>`).
Introduced by this phase (commit `e2ed553`, the CR-03 fix) and missed by iterations 5 and 6.
The pre-commit Sphinx hook has no `-W`, so it does not fail the build.
**Fix:** ``` ``.gitignore``\ d ``` (escaped space) or reword to "are both ignored by
``.gitignore``".

### IN-44: residual TOCTOU and a planted-file denial of service around the CR-05 fallback trust check

**File:** `solsys_code/unattended.py:461-478, 532-539, 656-667`
**Issue:** `_fallback_is_trustworthy()` validates by path (`lstat`), then `load_state()`
re-opens the same path by name (line 536) — a different inode may be there by then. On a
sticky `/tmp` no other account can win that race, but the guarantee comes from the sticky
bit rather than from anything this code does, and `TMPDIR` is operator-settable. The
mirror case is a denial of service rather than a spoof: a local account that creates a
regular file at the (predictable) fallback path first makes `os.replace()` fail with
`EPERM` under the sticky bit, so `save_state()`'s fallback write raises, the suppression
state is never persisted, and every tick re-mails the same failure notice — the WR-17
failure mode, reachable by an unprivileged local user.
**Fix:** open once and validate the descriptor:
`fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)`, then `os.fstat(fd)` for the
uid/regular-file/mode check, and read from that same descriptor. Optionally log the
planted-file case distinctly in `save_state()` so the runbook's troubleshooting section can
name it.

### IN-45: `_run_fold_with_missing_module()` duplicates `_run_fold()`'s anchor-slice-and-exec block

**File:** `solsys_code/tests/test_settings_api_key_fold.py:236-266` (vs `58-105`)
**Issue:** the import/read/`source.find(_FOLD_TAIL_ANCHOR)`/`self.fail(...)`/`compile(...,
'exec')` sequence is copied verbatim into the WR-38 helper, differing only in how
`fomo.local_settings` is made to fail. A future change to the anchor handling or to the
seeded namespace has to be made twice, and only one copy is obviously the canonical one.
**Fix:** extract the slice-and-exec into a small helper on `_FoldExecutionTestCase`
(`_exec_fold_tail(self) -> dict`) that both entry points call after they have set up their
respective `sys.modules`/`sys.meta_path` state.

### IN-46: `cron_line()` interpolates paths into a crontab line with no quoting or `%` escaping

**File:** `solsys_code/management/commands/check_unattended.py:477-482`
**Issue:** `flock_path`, `lock_file`, `python_path`, `manage_py_path` and `log_file` are
interpolated bare. A `FOMO_LOG_FILE`/`FOMO_LOCK_DIR`/venv path containing a space produces
a silently wrong cron line (word-split into extra arguments), and a `%` in any of them is
special to crontab (it terminates the command and starts stdin), truncating the line. The
printed line is explicitly "authoritative" per the runbook, so an operator would paste it
without review. Pre-existing, not introduced by this diff.
**Fix:** `shlex.quote()` each interpolated path and replace `%` with `\%`, or reject such
paths with a hard `CheckResult` naming the offending setting.

---

_Reviewed: 2026-09-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
