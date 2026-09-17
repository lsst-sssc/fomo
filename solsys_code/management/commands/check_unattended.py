"""Preflight for FOMO's unattended path (D-05, 36-CONTEXT.md, SC 5).

Answers "is this host ready to run FOMO unattended, and what exactly do I paste into the
crontab?" in one run. This command is read-only by construction (D-05, T-36-16): it
creates no directory, writes no file (beyond an optional test email, see
``--send-test-email``), and changes no row. A missing hard prerequisite -- no ``flock``
(or one too old to support ``-E``, WR-11, 36-REVIEW.md), an unwritable lock, log, or
state directory (WR-15, 36-REVIEW.md), the console email backend, or no staff user with
an email -- makes the command exit non-zero, naming every failed hard check in one
``CommandError`` so a fresh-host operator sees the whole list of problems in one run. An
unset heartbeat URL, a ``FOMO_BASE_URL`` left at its localhost dev default (WR-07,
36-REVIEW.md), and an empty watched-proposal list are warnings only (D-08, D-12): the
tick still runs, and mail still sends, without either.

SCHED-10/D-15: every check reports a NAME plus a set/unset status or a count. The only
values ever interpolated into this command's output are filesystem paths,
``sys.executable`` (see ``cron_line()``), and filesystem ownership/permission metadata
(IN-09, 36-REVIEW.md: ``_check_directory_writable()``'s ``os.geteuid()``, owner uid, and
octal mode) -- never a credential, a URL, or any other setting value.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser

from solsys_code import notifications
from solsys_code.models import WatchedProposal


@dataclass
class CheckResult:
    """The outcome of one preflight check.

    Attributes:
        name: a short, stable identifier for the check -- printed alongside its status,
            and named in the ``CommandError`` when the check is hard and failed.
        ok: True when the prerequisite is satisfied.
        hard: True when a failing result should make the command exit non-zero; False
            when it is a warning only (D-08, D-12).
        detail: a human-readable explanation. Never a credential or setting value --
            only paths, counts, and set/unset status (D-15, SCHED-10).
    """

    name: str
    ok: bool
    hard: bool
    detail: str


def check_flock() -> CheckResult:
    """Hard check: the ``flock`` binary is on ``PATH`` and supports ``-E`` -- the
    option the crontab template's skip-detection scheme depends on.

    WR-11 (36-REVIEW.md): ``-E``/``--conflict-exit-code`` was added in util-linux 2.27
    (2015); an older ``flock`` (e.g. RHEL/CentOS 7's 2.23, a still-live deployment
    target) rejects the unknown option and exits before ``run_unattended`` ever
    starts -- every tick becomes a silent no-op, the CR-01 failure class reproduced
    through a different door. Checking only ``shutil.which()`` would pass on such a
    host and hand the operator a cron line that never runs.
    """
    path = shutil.which('flock')
    if path is None:
        return CheckResult(
            name='flock',
            ok=False,
            hard=True,
            detail='not found on PATH -- install the util-linux package, which provides it',
        )
    probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False)  # noqa: S603
    if '--conflict-exit-code' not in (probe.stdout + probe.stderr):
        return CheckResult(
            name='flock',
            ok=False,
            hard=True,
            detail=(
                f'{path} does not support -E/--conflict-exit-code (util-linux < 2.27) -- '
                'the cron line below needs it to distinguish a skipped tick from a failed one'
            ),
        )
    return CheckResult(name='flock', ok=True, hard=True, detail=f'found at {path}, supports -E')


def _owner_mode(path: Path) -> str:
    """Render a path's owner uid and permission mode for a CheckResult's detail text."""
    info = path.stat()
    return f'owner uid {info.st_uid}, mode {oct(info.st_mode & 0o777)}'


def _check_directory_writable(name: str, path: Path) -> CheckResult:
    """Shared read-only writability probe for the lock and log directories.

    Never creates ``path`` -- reports what would need to exist and who would need to
    create it (D-05: the operator acts, not this command).

    WR-06 (36-REVIEW.md): ``os.access(path, os.W_OK)`` answers "can *this* process's
    uid write here" -- not the cron account's, and is effectively unconditional for
    uid 0. Report the resolved owner uid and mode alongside the verdict, and say
    explicitly whose write access was tested, so an operator running this preflight as
    root (a natural thing to do while creating the directories) does not mistake an
    ``[ok]`` tested as root for one tested as the unprivileged cron account.
    """
    if path.exists():
        detail_suffix = f' ({_owner_mode(path)})'
        if os.access(path, os.W_OK):
            return CheckResult(
                name=name,
                ok=True,
                hard=True,
                detail=(
                    f'{path} writable by uid {os.geteuid()}{detail_suffix} -- run this check as the '
                    'account that will actually run unattended to verify it too'
                ),
            )
        return CheckResult(
            name=name,
            ok=False,
            hard=True,
            detail=f'{path} exists but is not writable by uid {os.geteuid()}{detail_suffix}',
        )
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return CheckResult(
            name=name,
            ok=True,
            hard=True,
            detail=(
                f'{path} does not exist yet, but {parent} is writable by uid {os.geteuid()} '
                f'({_owner_mode(parent)}) -- run this check as the account that will actually run '
                'unattended to verify it too'
            ),
        )
    return CheckResult(
        name=name,
        ok=False,
        hard=True,
        detail=f'{path} does not exist and {parent} is not writable -- create it before running unattended',
    )


def check_lock_dir() -> CheckResult:
    """Hard check: ``settings.FOMO_LOCK_DIR`` -- the directory ``command_lock()`` locks in."""
    return _check_directory_writable('FOMO_LOCK_DIR', Path(settings.FOMO_LOCK_DIR))


def check_log_dir() -> CheckResult:
    """Hard check: the parent directory of ``settings.FOMO_LOG_FILE`` -- where the cron
    template's redirect appends."""
    return _check_directory_writable('FOMO_LOG_FILE', Path(settings.FOMO_LOG_FILE).parent)


def check_state_dir() -> CheckResult:
    """Hard check: ``settings.FOMO_STATE_DIR`` -- where ``save_state()`` persists the
    D-11 suppression state.

    WR-15 (36-REVIEW.md): an unwritable ``FOMO_STATE_DIR`` makes every tick send the
    same failure email again, forever -- ``load_state()`` falls back to "no prior
    failure" (WR-03), ``decide_notification()`` therefore decides ``'failure'`` on
    every tick, and ``save_state()``'s own raised ``OSError`` is caught and logged but
    never recorded, so the identical email goes out again next tick. Catching an
    unwritable state directory here, the same way ``check_lock_dir()`` catches an
    unwritable lock directory, lets an operator fix it before that loop starts.
    """
    return _check_directory_writable('FOMO_STATE_DIR', Path(settings.FOMO_STATE_DIR))


def check_email() -> list[CheckResult]:
    """Hard checks (two): the email backend can actually deliver, and there is at least
    one staff recipient on file (D-13).

    Returns:
        list[CheckResult]: ``[backend_result, recipients_result]``.
    """
    results: list[CheckResult] = []

    backend = settings.EMAIL_BACKEND
    is_console = backend == 'django.core.mail.backends.console.EmailBackend'
    results.append(
        CheckResult(
            name='EMAIL_BACKEND',
            ok=not is_console,
            hard=True,
            detail=(
                f'{backend} -- the console backend only prints to a terminal no one is '
                'watching; a failure notice would never be seen'
                if is_console
                else backend
            ),
        )
    )

    recipients = notifications.staff_recipients()
    results.append(
        CheckResult(
            name='staff_recipients',
            ok=bool(recipients),
            hard=True,
            detail=(
                f'{len(recipients)} staff user(s) with an email on file'
                if recipients
                else 'no staff user has an email on file -- a failure notice has nowhere to go'
            ),
        )
    )
    return results


def check_heartbeat() -> CheckResult:
    """Soft check: ``settings.FOMO_HEARTBEAT_URL`` is set (D-12's second visibility layer)."""
    if settings.FOMO_HEARTBEAT_URL:
        return CheckResult(name='heartbeat', ok=True, hard=False, detail='FOMO_HEARTBEAT_URL: set')
    return CheckResult(
        name='heartbeat',
        ok=False,
        hard=False,
        detail='FOMO_HEARTBEAT_URL: unset -- the second visibility layer is off',
    )


def check_base_url() -> CheckResult:
    """Soft check: ``settings.FOMO_BASE_URL`` has been changed from its localhost dev
    default (WR-07, 36-REVIEW.md).

    ``notifications.absolute_url()`` -- used by both the unattended failure email
    (Admin/Calendar links) and the campaign submission notice (the approval-queue link)
    -- joins whatever this setting is with a path. Left at the ``http://localhost:8000``
    default on any real deployment, every one of those links is unusable off this host,
    and nothing else catches it: this preflight is the only place that checks
    ``FOMO_BASE_URL`` at all. A warning, not a hard failure (D-08/D-12's own
    precedent) -- the tick still runs and the mail still sends, just with a dead link.
    """
    is_default = (settings.FOMO_BASE_URL or '').rstrip('/') in ('', 'http://localhost:8000')
    if is_default:
        return CheckResult(
            name='FOMO_BASE_URL',
            ok=False,
            hard=False,
            detail=(
                'FOMO_BASE_URL: unset or still the localhost dev default -- emailed links '
                'will not work off this host'
            ),
        )
    return CheckResult(name='FOMO_BASE_URL', ok=True, hard=False, detail='FOMO_BASE_URL: set')


def cron_line() -> str:
    """Return the exact cron line an operator should install, with real resolved values
    substituted for `deploy/cron/fomo.crontab.example`'s placeholders (D-05).

    Carries the same elements as the committed template -- the `*/15` schedule, the
    `flock -n -E 99` guard, the lock file path, the `run_unattended` command name, the
    `>> ... 2>&1` redirect, and the `lock held` skip tail gated on exit code 99 -- so an
    operator who follows either route ends up with the same behavior (T-36-15).

    WR-01 (36-REVIEW.md): the skip tail is gated on flock's own `-E 99` exit code, not
    on "any non-zero exit", so a tick that actually ran and failed is never mislabeled
    as a skipped one -- `run_unattended` exits 1 on a step failure, which is a disjoint
    code from flock's contention signal.

    WR-09 (36-REVIEW.md): the skip tail's own `[ $? -eq 99 ] && echo ...` is the last
    command in the line, so *its* exit status -- not flock's -- became the line's status.
    That inverted the reporting: 0 only on the one tick where nothing ran, 1 on both a
    healthy tick and a failing one. The line now captures flock's exit code into `$rc`
    immediately, uses `$rc` for the skip-tail test, and ends with an explicit `exit $rc`
    so the line's own status is always `run_unattended`'s (0 healthy, 1 failing, 99 skipped).

    Returns:
        str: the cron line. The only interpolated values are ``sys.executable``, the
            resolved `manage.py` path, the resolved ``flock`` path, and the two
            `FOMO_LOCK_DIR`/`FOMO_LOG_FILE` paths -- never a setting value that is not
            itself a filesystem path.
    """
    python_path = sys.executable
    manage_py_path = Path(settings.BASE_DIR).parent / 'manage.py'
    # NOT 'run_unattended.lock' -- that is the name `command_lock('run_unattended')`
    # locks from inside the process (unattended.py). `flock(2)` locks are per open file
    # description, so a cron guard sharing that filename would be denied by the lock its
    # own child just took, self-deadlocking every scheduled tick (CR-01, 36-REVIEW.md).
    lock_file = Path(settings.FOMO_LOCK_DIR) / 'run_unattended.cron.lock'
    log_file = settings.FOMO_LOG_FILE
    # WR-05 (36-REVIEW.md): resolve the real `flock` path the same way check_flock()
    # already verified it -- the committed template's hardcoded '/usr/bin/flock' is
    # only a placeholder for a host with a non-merged-/usr layout, a venv-provided
    # util-linux, or a container image that keeps it in /bin only.
    flock_path = shutil.which('flock') or '/usr/bin/flock'
    # WR-01 (36-REVIEW.md): `-E 99` makes flock exit 99 specifically on lock contention,
    # so the skip tail can be gated on that one code instead of "any non-zero exit" --
    # `run_unattended` itself exits 1 on a step failure, and a bare `||` tail would
    # therefore mislabel a failing (but genuinely run) tick as "lock held" in the log.
    return (
        f'*/15 * * * * {flock_path} -n -E 99 {lock_file} {python_path} {manage_py_path} run_unattended '
        f'>> {log_file} 2>&1; rc=$?; '
        f'[ $rc -eq 99 ] && echo "$(date -Is) run_unattended skipped: lock held" >> {log_file}; '
        f'exit $rc'
    )


def _send_test_email() -> CheckResult:
    """Send one test email through the configured backend to the staff-with-an-email
    recipient list `notifications.notify_staff()` uses -- the same recipient rule the
    failure notice relies on -- so an operator can prove the mail layer works during
    setup instead of waiting for a real failure.

    Returns:
        CheckResult: treated as hard -- no recipients, or a raised send, is a failure.
            The caught exception is reported by class name only (D-17): a relay
            error's message can embed the mail host credentials.
    """
    manage_py_path = Path(settings.BASE_DIR).parent / 'manage.py'
    subject = 'FOMO check_unattended test email'
    body = f'This is a test email sent by `python manage.py check_unattended --send-test-email` from {manage_py_path}.'
    try:
        sent = notifications.notify_staff(subject, body, fail_silently=False)
    except Exception as exc:  # noqa: BLE001 -- a mail send, D-17's first bucket
        return CheckResult(name='send_test_email', ok=False, hard=True, detail=f'send failed: {type(exc).__name__}')
    if not sent:
        return CheckResult(
            name='send_test_email',
            ok=False,
            hard=True,
            detail='no staff recipient has an email on file -- nothing was sent',
        )
    return CheckResult(name='send_test_email', ok=True, hard=True, detail='sent one test email to staff recipients')


def check_watched_proposals() -> CheckResult:
    """Soft check: at least one active ``WatchedProposal`` row exists (D-08)."""
    count = WatchedProposal.objects.filter(is_active=True).count()
    if count:
        return CheckResult(name='watched_proposals', ok=True, hard=False, detail=f'{count} active proposal(s)')
    return CheckResult(
        name='watched_proposals',
        ok=False,
        hard=False,
        detail='no active WatchedProposal rows -- discovery will be a quiet no-op; add one in the admin',
    )


class Command(BaseCommand):
    """Report every prerequisite the unattended path needs, in one run (D-05, SC 5).

    Read-only: creates no directory, writes no file, and changes no row -- it reports,
    the operator acts. Exits non-zero only when a hard prerequisite is missing; an unset
    heartbeat URL, a ``FOMO_BASE_URL`` left at its localhost dev default (WR-07,
    36-REVIEW.md), and an empty watched-proposal list are warnings (D-08, D-12).
    """

    help = (
        'Report whether this host is ready to run FOMO unattended -- flock, the lock and '
        'log directories, the email backend and staff recipients, the heartbeat URL, the '
        'base URL used to build emailed links, and the watched-proposal list -- in one '
        'run, and print the cron line to install. Read-only: reports, does not fix. '
        '--send-test-email additionally sends one test email through the configured backend.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--send-test-email',
            action='store_true',
            help='Send one test email through the configured backend to the staff recipient list.',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Run every check in order, report each result, print the cron line, and fail
        on any hard failure.

        Returns:
            str | None: a one-line summary of counts when every hard check passes.

        Raises:
            CommandError: when any hard check failed, naming every failed hard check.
        """
        results: list[CheckResult] = []
        results.append(check_flock())
        results.append(check_lock_dir())
        results.append(check_log_dir())
        results.append(check_state_dir())
        results.extend(check_email())
        results.append(check_heartbeat())
        results.append(check_base_url())
        results.append(check_watched_proposals())
        if options.get('send_test_email'):
            results.append(_send_test_email())

        for result in results:
            if result.ok:
                status = 'ok'
            elif result.hard:
                status = 'FAIL'
            else:
                status = 'WARN'
            line = f'[{status}] {result.name}: {result.detail}'
            self.stdout.write(line)
            if status != 'ok':
                self.stderr.write(line)

        # Printed even when a hard check failed -- an operator fixing prerequisites
        # still wants to see the target state (D-05).
        self.stdout.write('')
        self.stdout.write('Cron line to install (both host directories above must exist first):')
        self.stdout.write(cron_line())

        failed_hard = [result for result in results if result.hard and not result.ok]
        if failed_hard:
            names = ', '.join(result.name for result in failed_hard)
            raise CommandError(f'check_unattended: {len(failed_hard)} hard check(s) failed: {names}')

        ok_count = sum(1 for result in results if result.ok)
        return f'check_unattended: {ok_count}/{len(results)} checks passed'
