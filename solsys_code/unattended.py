"""The FOMO unattended runner (D-01, 36-CONTEXT.md).

``run_tick()`` is the single process a `flock -n`-guarded cron line invokes every 15
minutes. It runs each registered step in ``STEPS`` in order inside its own try/except
(D-02 -- a failing step never stops the later ones), brackets the whole tick with
healthchecks-style heartbeat pings (D-12), and mails staff once per newly-failing tick
with suppression, a daily reminder, and a one-shot recovery notice (D-11).

This module deliberately does not import the ephemeris/target-detail view module (whose
import triggers a large one-time SPICE kernel download) or that module's own heavy
dependency -- either would drag that cost into every cron tick. For the same reason, this
module never calls ``django.urls.reverse()``: from a management-command-only process
(no HTTP request has already forced the URL conf to load), the first such call would
resolve the *entire* project URL conf, which imports every view module wired into it --
reintroducing exactly the cost the previous sentence rules out. ``notifications.absolute_url()``
is given literal paths here for that reason, never a ``reverse()``'d one.
"""

import fcntl
import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path

import requests
from django.conf import settings

from solsys_code import notifications
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT_SECONDS = 10
_REMINDER_INTERVAL = timedelta(hours=24)
_STATE_FILENAME = 'unattended-state.json'


@dataclass
class StepResult:
    """Outcome of one unattended step.

    Attributes:
        name: the step's registry name (matches its ``STEPS`` entry).
        failed: True when the step's outcome should make the tick non-zero (D-10).
        summary: a short, credential- and PII-free counter/status line for the log,
            the failure email, and the end-of-tick banner.
    """

    name: str
    failed: bool
    summary: str


@dataclass
class TickResult:
    """Outcome of one ``run_tick()`` call.

    Attributes:
        exit_code: 0 on a healthy tick (including a lock-contended skip, D-12's grace
            period is the backstop for that case), 1 if any step failed.
        results: the per-step outcomes, in registry order. Empty for a lock-contended
            skip -- no step ran.
    """

    exit_code: int
    results: tuple[StepResult, ...] = field(default_factory=tuple)


class LockContended(Exception):  # noqa: N818 -- exact symbol name locked by 36-01-PLAN.md
    """Raised by ``command_lock()`` when the named lock is already held."""


@contextmanager
def command_lock(name: str) -> Iterator[None]:
    """Take a non-blocking exclusive ``fcntl`` lock on ``<FOMO_LOCK_DIR>/<name>.lock``.

    Args:
        name: the lock's name -- typically a command/step name, so a manual
            ``manage.py <command>`` run and a tick's own step both take the same named
            lock (defence in depth behind the runner-level lock this function also
            guards, per ``name='run_unattended'``).

    Yields:
        None. The lock is released (and the file descriptor closed) on exit, including
            on an exception raised inside the ``with`` block.

    Raises:
        LockContended: the lock is already held by another process.
    """
    lock_dir = Path(settings.FOMO_LOCK_DIR)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f'{name}.lock'
    with lock_path.open('a+') as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockContended(f'{lock_path} is already locked') from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ping_heartbeat(suffix: str) -> None:
    """Ping ``<FOMO_HEARTBEAT_URL>/<suffix>``, or do nothing when the URL is unset (D-12).

    Args:
        suffix: the path suffix -- ``'start'`` before the first step, or
            ``str(exit_code)`` after the last.

    Never raises: a network failure is logged by exception class name only (D-17) and
    swallowed, so a heartbeat outage never fails the tick.
    """
    url = settings.FOMO_HEARTBEAT_URL
    if not url:
        logger.info('FOMO_HEARTBEAT_URL is not set -- skipping heartbeat ping')
        return
    try:
        requests.get(f'{url.rstrip("/")}/{suffix}', timeout=_HEARTBEAT_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        logger.warning('heartbeat ping failed: %s', type(exc).__name__)


def step_reconcile(dry_run: bool) -> StepResult:
    """Sweep every ``CampaignRun`` through the shared reconciler.

    Mirrors ``reconcile_campaign_runs.Command.handle()``'s loop shape exactly, but
    reports only the pass/fail signal a ``StepResult`` needs -- the full per-field
    counters that command prints belong to that command, not to this step.

    Args:
        dry_run: report without writing when True.

    Returns:
        StepResult: ``failed`` is True if any run's ``reconcile_run()`` call raised.
            When the per-step lock is contended, returns a non-failing ``StepResult``
            noting the skip instead -- this per-step lock is defence in depth behind
            the runner-level lock and does not make a hand-run
            ``manage.py reconcile_campaign_runs`` wait for a tick (see
            36-01-PLAN.md's ``<decisions_this_plan_records>`` for why).
    """
    try:
        with command_lock('reconcile_campaign_runs'):
            runs = CampaignRun.objects.all().select_related('site', 'campaign').order_by('pk')
            failed_count = 0
            run_count = 0
            for run in runs:
                run_count += 1
                try:
                    reconcile_run(run, dry_run=dry_run)
                except Exception as exc:  # noqa: BLE001 -- FOMO's own reconcile_run(), D-17's
                    # second bucket; logged at DEBUG (not INFO/stderr) so this stays out of
                    # the tick's normal output surface even though the message may be kept.
                    logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
                    failed_count += 1
            summary = f'runs: {run_count}, failed: {failed_count}'
            return StepResult(name='reconcile', failed=failed_count > 0, summary=summary)
    except LockContended:
        return StepResult(name='reconcile', failed=False, summary='skipped -- lock held')


# D-01/D-04: the single source of the step order. Plan 03 prepends the other three steps
# in their D-01 order; nothing else may re-declare this tuple.
STEPS = (('reconcile', step_reconcile),)


def load_state() -> dict:
    """Read the D-11 suppression-state file.

    Returns:
        dict: ``{'failing_steps': [...], 'notified_at': <ISO-8601 str> | None}``. A
            missing or unparseable file is treated as "no prior failure" -- never an
            exception out of ``run_tick()``.
    """
    state_path = Path(settings.FOMO_STATE_DIR) / _STATE_FILENAME
    try:
        with state_path.open() as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {'failing_steps': [], 'notified_at': None}
    return {
        'failing_steps': sorted(data.get('failing_steps') or []),
        'notified_at': data.get('notified_at'),
    }


def save_state(failing_steps: list[str], notified_at: datetime | None) -> None:
    """Write the D-11 suppression-state file.

    Args:
        failing_steps: the currently-failing step names (any order -- sorted before
            writing).
        notified_at: when the notification for this state was sent, or None (the
            recovered/no-prior-failure state).
    """
    state_dir = Path(settings.FOMO_STATE_DIR)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / _STATE_FILENAME
    payload = {
        'failing_steps': sorted(failing_steps),
        'notified_at': notified_at.isoformat() if notified_at else None,
    }
    with state_path.open('w') as fh:
        json.dump(payload, fh)


def decide_notification(previous_state: dict, failing_steps: list[str], now: datetime) -> str | None:
    """Implement D-11's mail-once-per-newly-failing-set rule.

    Args:
        previous_state: the dict ``load_state()`` returned before this tick ran.
        failing_steps: step names failing on this tick (any order).
        now: the current time (an explicit parameter, not ``datetime.now()``, so tests
            can inject it).

    Returns:
        str | None: ``'failure'`` (a newly failing set), ``'reminder'`` (the same set
            persists and ``_REMINDER_INTERVAL`` has elapsed since the last
            notification), ``'recovered'`` (the set just became empty after a prior
            failure), or ``None`` (nothing to send).
    """
    previous_failing = sorted(previous_state.get('failing_steps') or [])
    failing_steps = sorted(failing_steps)
    notified_at_raw = previous_state.get('notified_at')
    notified_at = datetime.fromisoformat(notified_at_raw) if notified_at_raw else None

    if failing_steps:
        if failing_steps != previous_failing:
            return 'failure'
        if notified_at is not None and (now - notified_at) >= _REMINDER_INTERVAL:
            return 'reminder'
        return None
    if previous_failing:
        return 'recovered'
    return None


def _write_banner(kind: str, now: datetime, *, exit_code: int | None = None) -> None:
    """Write the D-01 start/end log banner (module constant format, 36-01-PLAN.md
    ``<decisions_this_plan_records>``)."""
    if kind == 'START':
        logger.info('=== FOMO unattended run START %s ===', now.isoformat())
    else:
        logger.info('=== FOMO unattended run END %s exit=%s ===', now.isoformat(), exit_code)


def _build_notification_body(decision: str, results: list[StepResult]) -> tuple[str, str]:
    """Build the D-14 subject/body for a 'failure'/'reminder'/'recovered' decision.

    Args:
        decision: one of ``decide_notification()``'s non-None return values.
        results: this tick's per-step results, in registry order.

    Returns:
        tuple[str, str]: (subject, body). Never includes a traceback, a request URL, or
            portal response text (D-14) -- only step names, each failed step's
            (already credential-free) summary line, the log file path, and two
            hardcoded-path admin/calendar links.
    """
    failing = [result for result in results if result.failed]
    if decision == 'recovered':
        subject = 'FOMO unattended run recovered'
        lines = ['The previously failing unattended run has recovered.']
    else:
        step_names = ', '.join(result.name for result in failing)
        subject = f'FOMO unattended run failed: {step_names}'
        lines = ['The following unattended step(s) failed:', '']
        lines.extend(f'- {result.name}: {result.summary}' for result in failing)
    lines.append('')
    lines.append(f'Log file: {settings.FOMO_LOG_FILE}')
    # Hardcoded, not reverse()'d -- see this module's docstring.
    lines.append(f'Admin: {notifications.absolute_url("/admin/")}')
    lines.append(f'Calendar: {notifications.absolute_url("/calendar/")}')
    return subject, '\n'.join(lines)


def _send_notification(decision: str, results: list[StepResult]) -> None:
    """Build and send the D-14 notification email for a non-None decision.

    Never raises (D-11/D-17): a mail outage is logged by exception class name only and
    swallowed, so it can never fail the tick.
    """
    subject, body = _build_notification_body(decision, results)
    try:
        notifications.notify_staff(subject, body, fail_silently=False)
    except Exception as exc:  # noqa: BLE001 -- mail sending is a network-ish call, D-17
        logger.error('failed to send unattended notification: %s', type(exc).__name__)


def run_tick(dry_run: bool = False, only_step: str | None = None) -> TickResult:
    """Run one unattended tick.

    Args:
        dry_run: pass ``dry_run=True`` to every step; never pings or mails (an operator
            preview).
        only_step: run exactly this one step and no other; like ``dry_run``, never
            pings or mails -- an operator debugging tool, and a manual run must never
            page staff.

    Returns:
        TickResult: ``exit_code`` is 0 on a healthy tick, 1 if any step failed. A
            contended whole-run lock is NOT a failure -- it returns ``exit_code=0``
            with no results and writes a skip line to stderr; the heartbeat's grace
            period (D-12) is the structural backstop for a permanently contended lock.
    """
    now = datetime.now(dt_timezone.utc)
    quiet = dry_run or only_step is not None

    try:
        with command_lock('run_unattended'):
            _write_banner('START', now)
            if not quiet:
                ping_heartbeat('start')

            steps: tuple[tuple[str, Callable[[bool], StepResult]], ...] = (
                STEPS if only_step is None else tuple(entry for entry in STEPS if entry[0] == only_step)
            )
            results: list[StepResult] = []
            for name, step_fn in steps:
                try:
                    result = step_fn(dry_run)
                except Exception as exc:  # noqa: BLE001 -- D-02: a raising step never aborts the tick
                    logger.warning('step %s raised: %s', name, type(exc).__name__)
                    result = StepResult(name=name, failed=True, summary=f'raised {type(exc).__name__}')
                results.append(result)
                logger.info('step %s: %s | %s', name, 'FAILED' if result.failed else 'ok', result.summary)

            exit_code = 1 if any(result.failed for result in results) else 0

            if not quiet:
                failing_steps = sorted(result.name for result in results if result.failed)
                previous_state = load_state()
                decision = decide_notification(previous_state, failing_steps, now)
                if decision is not None:
                    _send_notification(decision, results)
                if decision in ('failure', 'reminder'):
                    save_state(failing_steps, now)
                elif decision == 'recovered':
                    save_state([], None)
                ping_heartbeat(str(exit_code))

            _write_banner('END', now, exit_code=exit_code)
            return TickResult(exit_code=exit_code, results=tuple(results))
    except LockContended:
        sys.stderr.write('run_unattended: lock held -- skipping this tick\n')
        logger.warning('run_unattended: lock held -- skipping this tick')
        return TickResult(exit_code=0, results=())
