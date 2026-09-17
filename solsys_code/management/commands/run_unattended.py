"""The cron entry point (D-01, 36-CONTEXT.md).

Runs one unattended tick of ``solsys_code.unattended.STEPS`` in this process -- never
shells out to a second ``manage.py`` invocation for any step (D-02). Every step, the
heartbeat, and the failure email all live in ``unattended.run_tick()``; this command is a
thin wrapper matching every other management command in this codebase (the domain logic
lives in a module, the command only wires arguments through and translates the result
into a process exit code).
"""

import sys
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code import unattended


class Command(BaseCommand):
    """Run one unattended tick: reconcile every CampaignRun's calendar projection (and,
    from later plans in this phase, refresh LCO/SOAR observation statuses, sweep the
    observation projector, and discover newly-scheduled observations for every watched
    proposal) -- bracketed by a heartbeat ping and a deduplicated failure email.
    """

    help = (
        'Run one unattended tick of the FOMO pipeline steps. Exits non-zero if any step '
        'failed, so cron sees a failing tick.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what each step would do without writing, pinging, or mailing.',
        )
        parser.add_argument(
            '--step',
            choices=[name for name, _ in unattended.STEPS],
            default=None,
            help=(
                'Run exactly this one step. Like --dry-run, never pings or mails -- an '
                'operator debugging tool, never a substitute for a real tick.'
            ),
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Run a tick and give cron a non-zero exit status when it failed.

        Returns:
            str | None: None on a healthy tick. A failing tick never reaches this
                function's own return -- ``sys.exit()`` raises ``SystemExit`` first.
        """
        result = unattended.run_tick(dry_run=options['dry_run'], only_step=options['step'])
        if result.exit_code != 0:
            sys.exit(result.exit_code)
        return None
