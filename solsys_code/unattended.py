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
import hashlib
import io
import json
import logging
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

import requests
from django.conf import settings
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

from solsys_code import notifications
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.constants import CRON_INTERVAL_MINUTES as _CRON_INTERVAL_MINUTES
from solsys_code.constants import DEFAULT_LOCK_DIR as _DEFAULT_LOCK_DIR
from solsys_code.constants import DEFAULT_LOG_FILE as _DEFAULT_LOG_FILE
from solsys_code.management.commands.backfill_lco_observations import sweep_watched_rows
from solsys_code.management.commands.project_observation_calendar import resolve_observed_site
from solsys_code.models import CampaignRun
from solsys_code.observation_projector import PROJECTED_FACILITIES, project_queryset
from solsys_code.proposal_allocation import refresh_all

logger = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT_SECONDS = 10
_REMINDER_INTERVAL = timedelta(hours=24)
# D-04/WR-16/WR-36/IN-34 (36-REVIEW.md): _CRON_INTERVAL_MINUTES, _DEFAULT_LOCK_DIR and
# _DEFAULT_LOG_FILE are imported from solsys_code.constants (a leaf module with no
# project-local imports of its own) rather than declared here -- both this module and
# check_unattended.py's cron_line()/check_heartbeat() need the same three literals
# without either depending on the other's import graph. WR-36 found unattended.py had
# grown a second, independent copy of the interval (_CRON_TICK_INTERVAL = timedelta(
# minutes=15)) in the same iteration IN-22 established the single-owner rule for the two
# path defaults below -- constants.py is now the one place all three are declared.
# _CRON_TICK_INTERVAL is also the threshold _reap_stale_temp_files() uses, since any of
# _atomic_write_json()'s temp files still around after a full tick interval can only be a
# leftover from a killed process, never an in-flight write (a single write is
# milliseconds of work).
_CRON_TICK_INTERVAL = timedelta(minutes=_CRON_INTERVAL_MINUTES)
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
        exit_code: 0 on a healthy tick (including a lock-contended skip, the heartbeat
            (D-12) is the backstop for that case), 1 if any step failed.
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
    lock_dir = Path(settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR)
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
        response = requests.get(f'{url.rstrip("/")}/{suffix}', timeout=_HEARTBEAT_TIMEOUT_SECONDS)
        # IN-01 (36-REVIEW.md): requests.get() does not raise on a non-2xx response by
        # itself -- an expired/rotated healthchecks.io URL returning 404, or a rate-limit
        # 429, was silently treated as a delivered ping. raise_for_status() turns a 4xx/
        # 5xx into an HTTPError, a requests.exceptions.RequestException subclass, so it
        # is caught by the same except clause below.
        response.raise_for_status()
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


def _refresh_one_facility(facility: Any) -> tuple[int, list[str], int, str | None]:
    """Refresh every non-terminal ObservationRecord for one facility instance (D-03).

    Args:
        facility: an already-constructed ``LCOFacility``/``SOARFacility`` instance -- one
            per call, never shared across facilities or reused between ticks (Phase 34
            D-10).

    Returns:
        tuple[int, list[str], int, str | None]: ``(failed_record_count, class_names,
            omitted_recheck_count, outage_class_name)``. ``class_names`` holds the
            distinct exception class name observed while re-checking each of the first
            ``_MAX_STATUS_RECHECKS`` failed observation ids (WR-08, 36-REVIEW.md) -- a
            transient failure (the re-check succeeds) still counts toward
            ``failed_record_count`` but contributes no class name. ``omitted_recheck_count``
            is how many of the failed records past that cap were never individually
            re-checked at all. ``outage_class_name`` is None for an ordinary per-record
            failure count; it is the raised exception's class name, and
            ``failed_record_count`` is 0, when ``update_all_observation_statuses()``
            itself raised -- IN-05 (36-REVIEW.md): the true affected-record count is
            unknown in that case, so this must never be conflated with "exactly one
            record failed" (the previous behavior: returning a placeholder 1).
    """
    try:
        failed_records = facility.update_all_observation_statuses()
    except Exception as exc:  # noqa: BLE001 -- a portal call, D-17's first bucket
        logger.warning('status refresh raised for %s: %s', type(facility).__name__, type(exc).__name__)
        return 0, [], 0, type(exc).__name__

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
    return len(failed_records), class_names, omitted, None


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
            lco_failed, lco_classes, lco_omitted, lco_outage = _refresh_one_facility(LCOFacility())
            soar_failed, soar_classes, soar_omitted, soar_outage = _refresh_one_facility(SOARFacility())
            total_failed = lco_failed + soar_failed
            total_omitted = lco_omitted + soar_omitted
            classes: list[str] = []
            for name in (*lco_classes, *soar_classes):
                if name not in classes:
                    classes.append(name)

            # IN-05 (36-REVIEW.md): a whole-facility outage (update_all_observation_
            # statuses() itself raised) must read as "outage" with its exception class
            # name, never as "failed 1" -- the true affected-record count is unknown, and
            # reusing the per-record failure count there understated a systemic outage as
            # a single failing record.
            lco_part = f'LCO: outage ({lco_outage})' if lco_outage else f'LCO: failed {lco_failed}'
            soar_part = f'SOAR: outage ({soar_outage})' if soar_outage else f'SOAR: failed {soar_failed}'
            summary_parts = [lco_part, soar_part]
            # IN-05 (36-REVIEW.md): only append the classes segment when there is
            # something to show -- an empty 'classes: ' fragment (reachable whenever
            # every re-checked record's failure turned out transient, which the
            # WR-08 recheck cap makes more likely on a large outage) read as a dangling,
            # broken field rather than "nothing to report here".
            if classes:
                summary_parts.append(f'classes: {", ".join(classes)}')
            if total_omitted:
                # WR-08 (36-REVIEW.md): name how many failed records past the
                # _MAX_STATUS_RECHECKS cap were never individually re-checked, so the
                # failure email/log line does not silently imply every failure was
                # inspected.
                summary_parts.append(f'recheck capped: {total_omitted} omitted')
            summary = ' | '.join(summary_parts)
            failed = total_failed > 0 or bool(lco_outage) or bool(soar_outage)
            return StepResult(name='status_refresh', failed=failed, summary=summary)
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
    # IN-13 (36-REVIEW.md): the query/sweep/bookkeeping/failure-isolation loop itself
    # lives in sweep_watched_rows() -- shared with backfill_lco_observations.Command's
    # own bare-invocation path -- so this step only needs its own StepResult reporting
    # from the counts that helper returns.
    try:
        with command_lock('backfill_lco_observations'):
            # IN-02 (36-REVIEW.md): sweep_proposal()'s own per-request skip-reason lines
            # (missing id/target/instrument_type, a failed block lookup) sink into a
            # throwaway io.StringIO() when no stdout/stderr is given, and this step gave
            # neither -- every such reason was discarded, leaving only the bare failed
            # count in the tick's own summary/log. Capture them here and log at INFO:
            # these are structural skip reasons carrying only portal identifiers
            # (request/observation ids, target names, states) -- never a credential and
            # never a raw response body, request URL, or caught exception's message
            # (D-17, IN-18 36-REVIEW.md).
            #
            # WR-18 (36-REVIEW.md): this project's own LOGGING config pins the root
            # logger to INFO, so logging this at DEBUG (the IN-02 fix's original level)
            # was filtered out before it ever reached the crontab's redirected
            # unattended.log -- identical to the pre-IN-02 behaviour it was meant to
            # fix. INFO matches this same function's sibling operator-facing line below
            # ('0 watched proposals, nothing to discover'). Do NOT fix this by lowering
            # the global log level to DEBUG instead -- that activates WR-22 (a caught
            # portal exception's raw message reaching this same log at DEBUG).
            captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
            rows_swept, failed_count, failed_codes = sweep_watched_rows(
                dry_run=dry_run, stdout=captured_stdout, stderr=captured_stderr
            )
            for sink_name, sink in (('stdout', captured_stdout), ('stderr', captured_stderr)):
                captured_text = sink.getvalue()
                # IN-38 (36-REVIEW.md): one log record per line, not the whole captured
                # buffer as a single multi-line record -- a --dry-run preview writes one
                # "Would create/reuse ..." line per portal request, so a large proposal
                # previously produced one enormous log line that no line-oriented tool
                # (grep, logrotate's size accounting, journald's field limits) handles
                # gracefully.
                for line in captured_text.splitlines():
                    logger.info('discovery %s: %s', sink_name, line)
            if not rows_swept:
                logger.info('0 watched proposals, nothing to discover')
                return StepResult(name='discovery', failed=False, summary='0 watched proposals, nothing to discover')

            overall_summary = f'swept: {rows_swept}, failed: {failed_count}'
            if failed_codes:
                overall_summary += f' ({", ".join(failed_codes)})'
            return StepResult(name='discovery', failed=failed_count > 0, summary=overall_summary)
    except LockContended:
        return StepResult(name='discovery', failed=False, summary='skipped -- lock held')


def step_proposal_allocation(dry_run: bool) -> StepResult:
    """Refresh every stored proposal time allocation via the LCO portal (Phase 37 D-07).

    A dry run returns immediately without instantiating a facility or making any network
    call -- there is no meaningful read-only variant of a portal fetch.

    Args:
        dry_run: report a no-op summary without calling the portal when True.

    Returns:
        StepResult: ``failed`` is True if any proposal's fetch raised. The summary is built
            only from integer counters and (on failure) the first failing exception's class
            name -- never a credential value or a response body (T-37-04/T-37-05). When the
            per-step lock is contended, returns a non-failing ``StepResult`` noting the skip
            instead -- defence in depth behind the runner-level lock.
    """
    if dry_run:
        return StepResult(name='proposal_allocation', failed=False, summary='skipped (dry run)')

    try:
        with command_lock('proposal_allocation'):
            attempted, rows_written, failed, first_exception = refresh_all(LCOFacility())
            summary = f'proposals: {attempted}, rows written: {rows_written}, failed: {failed}'
            if first_exception:
                summary += f', first error: {first_exception}'
            return StepResult(name='proposal_allocation', failed=failed > 0, summary=summary)
    except LockContended:
        return StepResult(name='proposal_allocation', failed=False, summary='skipped -- lock held')


# D-01/D-04: the single source of the step order. Nothing else may re-declare this tuple.
STEPS = (
    ('status_refresh', step_status_refresh),
    ('project_sweep', step_project_sweep),
    ('discovery', step_discovery),
    ('reconcile', step_reconcile),
    ('proposal_allocation', step_proposal_allocation),
)


def _primary_state_path() -> Path:
    """The documented D-11 suppression-state file location (FOMO_STATE_DIR, defaulting
    to FOMO_LOCK_DIR)."""
    return Path(settings.FOMO_STATE_DIR or settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR) / _STATE_FILENAME


def _fallback_state_path() -> Path:
    """Where save_state() falls back to when FOMO_STATE_DIR is unwritable (WR-17,
    36-REVIEW.md).

    ``run_unattended`` is a fresh process per cron tick (no in-process loop), so a
    module-level "already notified" flag cannot survive between ticks -- only something
    written to disk can. ``tempfile.gettempdir()`` is deliberately NOT ``FOMO_STATE_DIR``
    (which is precisely what may have become unwritable): without a location outside it,
    an unwritable state directory makes every tick reach the identical "newly failing"
    decision forever, sending one identical failure email per tick (D-11 failing open).

    CR-05/WR-33 (36-REVIEW.md): a fixed, host-wide filename in a world-writable directory
    let any local account -- or a second FOMO deployment sharing the same host -- plant or
    overwrite this file and steer the notification decision. The filename is instead
    scoped to this deployment (a hash of ``settings.BASE_DIR``, unique per checkout) and
    to this process's own uid, so two FOMO instances on one host (staging/prod, or a test
    run beside a live cron deployment) can never collide on the same fallback path. This
    is a function, not a module-level constant, both so it can react to
    ``settings.BASE_DIR`` under ``override_settings`` in tests and so tests can patch it
    to a temp-directory path instead of touching the real system temp directory
    (WR-34, 36-REVIEW.md).
    """
    tag = hashlib.sha256(str(settings.BASE_DIR).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f'fomo-unattended-state.{os.geteuid()}.{tag}.fallback.json'


def _fallback_is_trustworthy(path: Path) -> bool:
    """CR-05 (36-REVIEW.md): only trust a fallback state file this process itself could
    have written.

    The scoped filename from ``_fallback_state_path()`` alone is not enough: the shared
    system temp directory is still world-writable, so any local account can still create
    a file at that exact name (or plant a symlink there) before this process does. Check
    ownership and permissions with ``lstat`` (never follow a planted symlink), and require
    a regular file owned by this euid with no group/other permission bits -- exactly what
    ``_atomic_write_json()``'s own ``os.chmod(tmp_path, 0o600)`` produces. Anything else
    (missing, a symlink, a different owner, or a looser mode) is ignored rather than
    trusted.
    """
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not (info.st_mode & 0o077)


def _newest_existing_state_path() -> Path | None:
    """Return whichever of the primary/fallback state files was written most recently,
    or None if neither exists (WR-17, 36-REVIEW.md).

    The fallback is only ever newer than the primary while the primary is unwritable --
    the moment a ``save_state()`` succeeds against the primary again, it deletes the
    fallback, so this reverts to reading the primary alone as soon as the outage clears.

    CR-05 (36-REVIEW.md): the fallback candidate is only considered at all when
    ``_fallback_is_trustworthy()`` says this process itself could have written it -- a
    fallback file that fails that check (owned by a different local account, a planted
    symlink, or a looser mode) is never read, no matter how new its mtime is.
    """
    candidates = []
    primary_path = _primary_state_path()
    try:
        candidates.append((primary_path.stat().st_mtime, primary_path))
    except OSError:
        pass

    fallback_path = _fallback_state_path()
    if _fallback_is_trustworthy(fallback_path):
        try:
            candidates.append((fallback_path.stat().st_mtime, fallback_path))
        except OSError:
            pass

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def load_state() -> dict:
    """Read the D-11 suppression-state file.

    Returns:
        dict: ``{'failing_steps': [...], 'notified_at': <tz-aware datetime> | None}``.
            A missing file, an unparseable one, one whose top level is not a dict/list
            (WR-03, 36-REVIEW.md), a ``failing_steps`` whose elements are not all
            strings (WR-10, 36-REVIEW.md -- non-string elements are dropped rather
            than raising when the remaining list is later sorted), or a
            ``notified_at`` that is not a valid ISO-8601 string is treated as "no
            prior failure" -- never an exception out of ``run_tick()``. A naive
            ``notified_at`` (no tzinfo) is assumed UTC, so ``decide_notification()``
            can always subtract it from an aware ``now``.

    WR-17 (36-REVIEW.md): reads whichever of the primary state file and the
    ``save_state()`` fallback location was written most recently, so a tick that could
    only persist to the fallback (because ``FOMO_STATE_DIR`` was unwritable) is still
    seen by the next tick's notification decision, instead of being silently lost.
    """
    state_path = _newest_existing_state_path()
    if state_path is None:
        return {'failing_steps': [], 'notified_at': None}
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
    # WR-10 (36-REVIEW.md): a hand-edited or foreign-written state file can carry a
    # 'failing_steps' list with mixed element types (e.g. `[1, "a"]`) -- sorted() below
    # would then raise TypeError comparing str to int/bool, which is exactly the
    # exception this function's own docstring promises never to raise. Drop anything
    # that is not a step name instead of failing the whole tick's notification path.
    failing_steps = [step for step in failing_steps if isinstance(step, str)]

    raw_notified_at = data.get('notified_at')
    try:
        notified_at = datetime.fromisoformat(raw_notified_at) if raw_notified_at else None
    except (TypeError, ValueError):
        # CR-06 (36-REVIEW.md): an unparseable notified_at makes the whole record
        # untrustworthy, not just this one field -- keeping failing_steps without it
        # wedges decide_notification() on "same set, never notified": failing_steps
        # matches on every later tick, so the 'failure' branch never re-fires, and
        # notified_at is None so the 24-hour reminder branch never fires either. That
        # is silent, permanent loss of mail for exactly the failing set this phase
        # exists to report. Treat the whole record as "no prior failure" instead, which
        # is what this function's own docstring already promises.
        logger.warning('unattended state file has an unparseable notified_at -- ignoring the whole record')
        return {'failing_steps': [], 'notified_at': None}
    if notified_at is not None and notified_at.tzinfo is None:
        notified_at = notified_at.replace(tzinfo=dt_timezone.utc)

    return {
        'failing_steps': sorted(failing_steps),
        'notified_at': notified_at,
    }


def _reap_stale_temp_files(directory: Path, filename: str) -> None:
    """Remove any of ``_atomic_write_json()``'s own temp files older than one tick
    interval (WR-16/D-04: 15 minutes), from a previous write this process (or a prior
    one) never got to clean up.

    IN-21 (36-REVIEW.md): a ``SIGKILL``/OOM kill between ``mkstemp()`` and
    ``os.replace()`` -- the same failure class this function's own except clause
    already handles for a raised exception -- leaves a ``.<filename>.<random>.tmp``
    file behind forever, since ``load_state()`` only ever reads the exact target
    filename. On a host that OOM-kills ticks, the directory (often a small tmpfs, since
    it defaults to ``FOMO_LOCK_DIR``) accumulates one file per occurrence with nothing
    to clean them. Best-effort only: a listing or stat failure here must never block the
    write that follows.
    """
    threshold_seconds = _CRON_TICK_INTERVAL.total_seconds()
    now = datetime.now(dt_timezone.utc).timestamp()
    with suppress(OSError):
        for candidate in directory.glob(f'.{filename}.*.tmp'):
            with suppress(OSError):
                if now - candidate.stat().st_mtime > threshold_seconds:
                    candidate.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically: a fresh temp file in the same
    directory, then ``os.replace()``.

    IN-03 (36-REVIEW.md): so a process kill mid-write (the same OOM/SIGKILL class
    WR-14 already documents for the lock file) can never leave a torn/partial JSON file
    for the next tick's ``load_state()`` to find. Sets an explicit ``0o600`` mode rather
    than relying on the process umask, since the parent directory may not be
    exclusively owned by this process (``FOMO_STATE_DIR`` defaults to
    ``FOMO_LOCK_DIR``; the fallback path is the shared system temp directory).

    IN-21 (36-REVIEW.md): reaps this function's own leftover temp files from an earlier
    kill (see ``_reap_stale_temp_files()``) before creating a new one.

    Raises:
        OSError: the directory or temp file could not be created or written -- the
            caller decides what "could not persist state" means for it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_temp_files(path.parent, path.name)
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, 'w') as fh:
            json.dump(payload, fh)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def save_state(failing_steps: list[str], notified_at: datetime | None) -> None:
    """Write the D-11 suppression-state file.

    Args:
        failing_steps: the currently-failing step names (any order -- sorted before
            writing).
        notified_at: when the notification for this state was sent, or None (the
            recovered/no-prior-failure state).

    WR-17 (36-REVIEW.md): if the primary location (``FOMO_STATE_DIR``) cannot be
    written -- e.g. it became unwritable or full after setup, past
    ``check_state_dir()``'s one-time preflight -- falls back to
    ``_fallback_state_path()`` instead of raising. Without this, ``load_state()`` sees
    no persisted failure on the NEXT tick, ``decide_notification()`` reaches "newly
    failing" again, and the tick mails staff the same failure notice every 15 minutes
    for as long as the outage lasts (D-11 failing open). Once a write to the primary
    location succeeds again, the stale fallback is removed so ``load_state()`` goes
    back to reading the primary alone.
    """
    payload = {
        'failing_steps': sorted(failing_steps),
        'notified_at': notified_at.isoformat() if notified_at else None,
    }
    primary_path = _primary_state_path()
    try:
        _atomic_write_json(primary_path, payload)
    except OSError:
        fallback_path = _fallback_state_path()
        logger.error(
            'could not persist unattended suppression state to %s -- writing to the fallback '
            'location %s instead so a repeat notification for this failing set is still '
            'suppressed (WR-17, 36-REVIEW.md)',
            primary_path,
            fallback_path,
        )
        _atomic_write_json(fallback_path, payload)
        return
    with suppress(OSError):
        _fallback_state_path().unlink()


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

    CR-06 (36-REVIEW.md): belt-and-braces alongside ``load_state()``'s own fix for the
    same disagreement -- a ``previous_state`` whose ``failing_steps`` matches this tick's
    but whose ``notified_at`` is ``None`` (any origin: a hand-edited file, an older/newer
    schema, or a future caller that skips ``load_state()``) must be treated as due for
    notification *now*, not as "already notified, indefinitely". Without this, that
    combination reaches neither the ``'failure'`` branch (the sets match) nor the
    ``'reminder'`` branch (``notified_at is None`` makes the elapsed-time check
    unreachable), and returns ``None`` forever.
    """
    previous_failing = sorted(previous_state.get('failing_steps') or [])
    failing_steps = sorted(failing_steps)
    notified_at = previous_state.get('notified_at')

    if failing_steps:
        if failing_steps != previous_failing:
            return 'failure'
        if notified_at is None:
            # A previously-recorded failure with no recorded send time can never have
            # actually notified anyone -- treat it as due now rather than silently
            # never-again (see CR-06 docstring note above).
            return 'failure'
        if (now - notified_at) >= _REMINDER_INTERVAL:
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
    lines.append(f'Log file: {settings.FOMO_LOG_FILE or _DEFAULT_LOG_FILE}')
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
            with no results and writes a skip line to stderr; the heartbeat (D-12) is
            the structural backstop for a permanently contended lock.

    Raises:
        ValueError: ``only_step`` does not match any registered ``STEPS`` name. IN-06
            (36-REVIEW.md): ``run_unattended``'s own ``--step`` argparse ``choices``
            already rejects an unknown name from the CLI, but ``run_tick()`` is a public
            function any other caller can invoke directly with an arbitrary string --
            before this check, an unrecognized ``only_step`` silently matched zero
            ``STEPS`` entries, ran nothing, and returned ``exit_code=0``: a typo'd step
            name looked exactly like a healthy tick.
    """
    if only_step is not None and only_step not in dict(STEPS):
        raise ValueError(f'Unknown step {only_step!r} -- must be one of {[name for name, _ in STEPS]!r}')

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
