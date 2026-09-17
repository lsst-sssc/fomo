"""Preflight for FOMO's unattended path (D-05, 36-CONTEXT.md, SC 5).

Answers "is this host ready to run FOMO unattended, and what exactly do I paste into the
crontab?" in one run. This command is read-only by construction (D-05, T-36-16): it
creates no directory, writes no file (beyond an optional test email, see
``--send-test-email``), and changes no row. A missing hard prerequisite -- no ``flock``,
an unwritable lock or log directory, the console email backend, or no staff user with an
email -- makes the command exit non-zero, naming every failed hard check in one
``CommandError`` so a fresh-host operator sees the whole list of problems in one run. An
unset heartbeat URL and an empty watched-proposal list are warnings only (D-08, D-12):
the tick still runs without either.

SCHED-10/D-15: every check reports a NAME plus a set/unset status or a count. The only
values ever interpolated into this command's output are filesystem paths and
``sys.executable`` (see ``cron_line()``) -- never a credential, a URL, or any other
setting value.
"""

import os
import shutil
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
    """Hard check: the ``flock`` binary the crontab template depends on is on ``PATH``."""
    path = shutil.which('flock')
    if path is not None:
        return CheckResult(name='flock', ok=True, hard=True, detail=f'found at {path}')
    return CheckResult(
        name='flock',
        ok=False,
        hard=True,
        detail='not found on PATH -- install the util-linux package, which provides it',
    )


def _check_directory_writable(name: str, path: Path) -> CheckResult:
    """Shared read-only writability probe for the lock and log directories.

    Never creates ``path`` -- reports what would need to exist and who would need to
    create it (D-05: the operator acts, not this command).
    """
    if path.exists():
        if os.access(path, os.W_OK):
            return CheckResult(name=name, ok=True, hard=True, detail=f'{path} exists and is writable')
        return CheckResult(name=name, ok=False, hard=True, detail=f'{path} exists but is not writable')
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return CheckResult(name=name, ok=True, hard=True, detail=f'{path} does not exist yet, but {parent} is writable')
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
    heartbeat URL and an empty watched-proposal list are warnings (D-08, D-12).
    """

    help = (
        'Report whether this host is ready to run FOMO unattended -- flock, the lock and '
        'log directories, the email backend and staff recipients, the heartbeat URL, and '
        'the watched-proposal list -- in one run. Read-only: reports, does not fix.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments. No argument is required."""
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Run every check in order, report each result, and fail on any hard failure.

        Returns:
            str | None: a one-line summary of counts when every hard check passes.

        Raises:
            CommandError: when any hard check failed, naming every failed hard check.
        """
        results: list[CheckResult] = []
        results.append(check_flock())
        results.append(check_lock_dir())
        results.append(check_log_dir())
        results.extend(check_email())
        results.append(check_heartbeat())
        results.append(check_watched_proposals())

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

        failed_hard = [result for result in results if result.hard and not result.ok]
        if failed_hard:
            names = ', '.join(result.name for result in failed_hard)
            raise CommandError(f'check_unattended: {len(failed_hard)} hard check(s) failed: {names}')

        ok_count = sum(1 for result in results if result.ok)
        return f'check_unattended: {ok_count}/{len(results)} checks passed'
