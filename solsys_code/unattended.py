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
from typing import Any

import requests
from django.conf import settings
from django.utils import timezone
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

from solsys_code import notifications
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.management.commands.backfill_lco_observations import sweep_proposal, watched_rows
from solsys_code.management.commands.project_observation_calendar import resolve_observed_site
from solsys_code.models import CampaignRun
from solsys_code.observation_projector import PROJECTED_FACILITIES, project_queryset

logger = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT_SECONDS = 10
_REMINDER_INTERVAL = timedelta(hours=24)
_STATE_FILENAME = 'unattended-state.json'
# WR-08 (36-REVIEW.md): during a whole-facility outage, every non-terminal record fails
# update_all_observation_statuses(), and _refresh_one_facility() re-checks each one
# individually purely to name the exception class -- uncapped, that is 2N portal
# requests on a 15-minute schedule, with no cap, no backoff, and no per-tick time
# budget. A handful of re-checks is enough to identify the failure mode; the rest add
# nothing but portal load and tick duration.
_MAX_STATUS_RECHECKS = 20


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


def _refresh_one_facility(facility: Any) -> tuple[int, list[str], int]:
    """Refresh every non-terminal ObservationRecord for one facility instance (D-03).

    Args:
        facility: an already-constructed ``LCOFacility``/``SOARFacility`` instance -- one
            per call, never shared across facilities or reused between ticks (Phase 34
            D-10).

    Returns:
        tuple[int, list[str], int]: ``(failed_record_count, class_names,
            omitted_recheck_count)``. ``class_names`` holds the distinct exception class
            name observed while re-checking each of the first ``_MAX_STATUS_RECHECKS``
            failed observation ids (WR-08, 36-REVIEW.md) -- a transient failure (the
            re-check succeeds) still counts toward ``failed_record_count`` but
            contributes no class name. ``omitted_recheck_count`` is how many of the
            failed records past that cap were never individually re-checked at all.
    """
    try:
        failed_records = facility.update_all_observation_statuses()
    except Exception as exc:  # noqa: BLE001 -- a portal call, D-17's first bucket
        logger.warning('status refresh raised for %s: %s', type(facility).__name__, type(exc).__name__)
        return 1, [type(exc).__name__], 0

    class_names: list[str] = []
    recheck_targets = failed_records[:_MAX_STATUS_RECHECKS]
    for observation_id, _message in recheck_targets:
        # The message half is discarded immediately, before any logging or string
        # building -- it can embed portal request/response content (SCHED-10, Pitfall 2).
        try:
            facility.update_observation_status(observation_id)
        except Exception as exc:  # noqa: BLE001 -- a portal call, D-17's first bucket
            logger.warning('observation_id=%s %s', observation_id, type(exc).__name__)
            class_names.append(type(exc).__name__)
        else:
            logger.warning('observation_id=%s no exception on re-check', observation_id)
    omitted = len(failed_records) - len(recheck_targets)
    return len(failed_records), class_names, omitted


def step_status_refresh(dry_run: bool) -> StepResult:
    """Refresh every LCO/SOAR ``ObservationRecord``'s status via TOM's own facility classes
    (D-03), replacing the stock ``updatestatus`` command's always-zero exit.

    A dry run returns immediately without instantiating either facility -- a status
    refresh is a portal read that mutates ``ObservationRecord`` rows through the Phase 34
    ``post_save`` receiver, so there is no meaningful read-only variant of it.

    Args:
        dry_run: report a no-op summary without calling either facility when True.

    Returns:
        StepResult: ``failed`` is True if either facility reported at least one failed
            record (or raised outright). When the per-step lock is contended, returns a
            non-failing ``StepResult`` noting the skip instead -- defence in depth behind
            the runner-level lock (see 36-01-PLAN.md's
            ``<decisions_this_plan_records>``).
    """
    if dry_run:
        return StepResult(name='status_refresh', failed=False, summary='skipped (dry run)')

    try:
        with command_lock('status_refresh'):
            lco_failed, lco_classes, lco_omitted = _refresh_one_facility(LCOFacility())
            soar_failed, soar_classes, soar_omitted = _refresh_one_facility(SOARFacility())
            total_failed = lco_failed + soar_failed
            total_omitted = lco_omitted + soar_omitted
            classes: list[str] = []
            for name in (*lco_classes, *soar_classes):
                if name not in classes:
                    classes.append(name)
            summary = f'LCO: failed {lco_failed} | SOAR: failed {soar_failed} | classes: {", ".join(classes)}'
            if total_omitted:
                # WR-08 (36-REVIEW.md): name how many failed records past the
                # _MAX_STATUS_RECHECKS cap were never individually re-checked, so the
                # failure email/log line does not silently imply every failure was
                # inspected.
                summary += f' | recheck capped: {total_omitted} omitted'
            return StepResult(name='status_refresh', failed=total_failed > 0, summary=summary)
    except LockContended:
        return StepResult(name='status_refresh', failed=False, summary='skipped -- lock held')


def step_project_sweep(dry_run: bool) -> StepResult:
    """Sweep every LCO/SOAR ``ObservationRecord`` through the observation projector.

    Reproduces ``project_observation_calendar.Command.handle()``'s logic directly
    (D-02) -- unfiltered by proposal or facility, matching 34 D-17's "the runner calls
    this sweep bare" -- rather than going through the command.

    Args:
        dry_run: report what would change without writing, and skip the one-time
            observed-site lookup entirely (D-08's dry-run caveat), when True.

    Returns:
        StepResult: ``failed`` is True if any row's projection came back
            ``'unprojectable'``. When the per-step lock is contended, returns a
            non-failing ``StepResult`` noting the skip instead.
    """
    try:
        with command_lock('project_observation_calendar'):
            records = ObservationRecord.objects.filter(facility__in=PROJECTED_FACILITIES)

            def hook(record: ObservationRecord, facility: Any) -> dict[str, int] | None:
                increment, message = resolve_observed_site(record, facility)
                if message:
                    # A fixed, generic message naming only the observation_id (D-08) --
                    # never a caught exception's value.
                    logger.warning(message)
                return increment

            result = project_queryset(records, dry_run=dry_run, pre_fields_hook=None if dry_run else hook)

            failed = sum(1 for row in result['rows'] if row['action'] == 'unprojectable')
            summary = ' | '.join(
                f'{facility}: created: {c["created"]}, updated: {c["updated"]}, unchanged: {c["unchanged"]}, '
                f'unprojectable: {c["unprojectable"]}, site_lookups: {c["site_lookups"]}, '
                f'site_lookup_failed: {c["site_lookup_failed"]}'
                for facility, c in result['counters'].items()
            )
            return StepResult(name='project_sweep', failed=failed > 0, summary=f'failed: {failed} | {summary}')
    except LockContended:
        return StepResult(name='project_sweep', failed=False, summary='skipped -- lock held')


def step_discovery(dry_run: bool) -> StepResult:
    """Sweep every active ``WatchedProposal`` row through ``sweep_proposal()`` (D-07..D-09).

    Args:
        dry_run: report what would change without writing any ``WatchedProposal``
            bookkeeping when True.

    Returns:
        StepResult: ``failed`` is True if any row's sweep raised -- the failing
            proposal code(s) are named in the summary so D-14's failure email can quote
            them. A zero-row watched list is a healthy tick, never a failure (D-08).
            When the per-step lock is contended, returns a non-failing ``StepResult``
            noting the skip instead.
    """
    try:
        with command_lock('backfill_lco_observations'):
            rows = list(watched_rows())
            if not rows:
                logger.info('0 watched proposals, nothing to discover')
                return StepResult(name='discovery', failed=False, summary='0 watched proposals, nothing to discover')

            failed_count = 0
            failed_codes: list[str] = []
            for row in rows:
                try:
                    summary = sweep_proposal(
                        row.proposal_code,
                        target_list_name=row.target_list_name or None,
                        user=row.attributed_to,
                        dry_run=dry_run,
                    )
                except Exception as exc:  # noqa: BLE001 -- portal/facility call, D-17's first bucket
                    # D-17: a portal exception's message can embed request/response
                    # content, so only the class name ever reaches the row or the log.
                    summary = f'failed: {type(exc).__name__}'
                    logger.debug(
                        'sweep_proposal() failed for proposal_code=%r: %s', row.proposal_code, type(exc).__name__
                    )
                    failed_count += 1
                    failed_codes.append(row.proposal_code)

                # D-09: bookkeeping is written whether the sweep succeeded or failed, so
                # the admin's last_run_at/last_run_summary are never stale for a row that
                # was actually swept -- but a dry run writes nothing at all.
                if not dry_run:
                    row.last_run_at = timezone.now()
                    row.last_run_summary = summary
                    row.save(update_fields=['last_run_at', 'last_run_summary'])

            overall_summary = f'swept: {len(rows)}, failed: {failed_count}'
            if failed_codes:
                overall_summary += f' ({", ".join(failed_codes)})'
            return StepResult(name='discovery', failed=failed_count > 0, summary=overall_summary)
    except LockContended:
        return StepResult(name='discovery', failed=False, summary='skipped -- lock held')


# D-01/D-04: the single source of the step order. Nothing else may re-declare this tuple.
STEPS = (
    ('status_refresh', step_status_refresh),
    ('project_sweep', step_project_sweep),
    ('discovery', step_discovery),
    ('reconcile', step_reconcile),
)


def load_state() -> dict:
    """Read the D-11 suppression-state file.

    Returns:
        dict: ``{'failing_steps': [...], 'notified_at': <tz-aware datetime> | None}``.
            A missing file, an unparseable one, one whose top level is not a dict/list
            (WR-03, 36-REVIEW.md), or a ``notified_at`` that is not a valid ISO-8601
            string is treated as "no prior failure" -- never an exception out of
            ``run_tick()``. A naive ``notified_at`` (no tzinfo) is assumed UTC, so
            ``decide_notification()`` can always subtract it from an aware ``now``.
    """
    state_path = Path(settings.FOMO_STATE_DIR) / _STATE_FILENAME
    try:
        with state_path.open() as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {'failing_steps': [], 'notified_at': None}
    if not isinstance(data, dict):
        return {'failing_steps': [], 'notified_at': None}

    failing_steps = data.get('failing_steps')
    if not isinstance(failing_steps, list):
        failing_steps = []

    raw_notified_at = data.get('notified_at')
    try:
        notified_at = datetime.fromisoformat(raw_notified_at) if raw_notified_at else None
    except (TypeError, ValueError):
        notified_at = None
    if notified_at is not None and notified_at.tzinfo is None:
        notified_at = notified_at.replace(tzinfo=dt_timezone.utc)

    return {
        'failing_steps': sorted(failing_steps),
        'notified_at': notified_at,
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
        previous_state: the dict ``load_state()`` returned before this tick ran --
            ``notified_at`` is already a parsed, tz-aware ``datetime`` (or None), never
            a raw string (WR-03, 36-REVIEW.md: ``load_state()`` owns all of that
            parsing so a malformed state file can never raise from in here).
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
    notified_at = previous_state.get('notified_at')

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


def _send_notification(decision: str, results: list[StepResult]) -> bool:
    """Build and send the D-14 notification email for a non-None decision.

    Never raises (D-11/D-17): a mail outage is logged by exception class name only and
    swallowed, so it can never fail the tick.

    Returns:
        bool: True only when the mail was actually attempted *and* delivered --
            ``notifications.notify_staff()`` returns False when there is no staff
            recipient at all, and a raised send is caught and also reported as False
            (WR-02, 36-REVIEW.md). The caller must not record the notification as sent
            unless this is True, or a mail outage on the first failing tick would
            suppress every subsequent notification for the same failing set.
    """
    subject, body = _build_notification_body(decision, results)
    try:
        return notifications.notify_staff(subject, body, fail_silently=False)
    except Exception as exc:  # noqa: BLE001 -- mail sending is a network-ish call, D-17
        logger.error('failed to send unattended notification: %s', type(exc).__name__)
        return False


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
            # WR-04 (36-REVIEW.md): sample a fresh timestamp now that every step has
            # actually run, rather than reusing the START-of-tick `now` -- otherwise the
            # END banner always carries the identical timestamp as its own START line
            # (so no tick's duration is ever readable from the log), and a long tick's
            # reminder timing drifts by the tick's own duration.
            end_time = datetime.now(dt_timezone.utc)

            if not quiet:
                failing_steps = sorted(result.name for result in results if result.failed)
                # WR-03 (36-REVIEW.md): isolate the whole notification/state block --
                # load_state()'s own docstring promised this never raises, but an
                # unwritable/full FOMO_STATE_DIR on save_state() (or any other surprise
                # here) must still not stop the END banner or the exit-code heartbeat
                # ping below from running, the same discipline every other failure path
                # in this module already follows.
                try:
                    previous_state = load_state()
                    decision = decide_notification(previous_state, failing_steps, end_time)
                    # WR-02 (36-REVIEW.md): only record the notification as sent when it
                    # was actually attempted *and* delivered -- otherwise a down SMTP
                    # relay (or every staff email cleared) on the first failing tick
                    # would record notified_at anyway, suppressing all further mail for
                    # the same failing set for 24 hours, and again per reminder window.
                    sent = _send_notification(decision, results) if decision is not None else False
                    if sent and decision in ('failure', 'reminder'):
                        save_state(failing_steps, end_time)
                    elif sent and decision == 'recovered':
                        save_state([], None)
                except Exception as exc:  # noqa: BLE001 -- D-11/D-17, see comment above
                    logger.error('unattended notification/state handling raised: %s', type(exc).__name__)
                ping_heartbeat(str(exit_code))

            _write_banner('END', end_time, exit_code=exit_code)
            return TickResult(exit_code=exit_code, results=tuple(results))
    except LockContended:
        sys.stderr.write('run_unattended: lock held -- skipping this tick\n')
        logger.warning('run_unattended: lock held -- skipping this tick')
        return TickResult(exit_code=0, results=())
