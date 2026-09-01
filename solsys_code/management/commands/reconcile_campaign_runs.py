import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """The single idempotent sweep that projects and refreshes calendar events for every
    ``CampaignRun`` (RECON-01), replacing the retired ``backfill_range_calendar_events``
    command and the per-gap backfill pattern generally (RECON-09). All projection math lives
    in ``solsys_code.campaign_reconciler`` -- this command only loops it. Unlike the retired
    command, which imported the private ``_project_calendar_event`` out of the views module,
    this one imports a public function from a dedicated pure module.
    """

    help = (
        'Sweep every CampaignRun through the shared reconciler, projecting and refreshing '
        'its calendar events. --dry-run reports what would change without writing anything.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be reconciled without writing any CalendarEvent rows.',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Loop reconcile_run() over every CampaignRun and report the D-05 summary.

        Returns:
            str | None: None on completion.
        """
        dry_run = options['dry_run']

        # Deliberately unfiltered: RECON-01 says "every run, regardless of window length,
        # source, or site-resolution state", and reconcile_run()'s own _skip_reason() guard
        # (including its 'not approved' gate) is the single place that decides what does not
        # project -- this command must never grow a second, divergent copy of that rule.
        runs = CampaignRun.objects.all().select_related('site', 'campaign').order_by('pk')

        created = updated = unchanged = blocked = 0
        skipped_count = 0
        failed_count = 0
        run_count = 0

        for run in runs:
            run_count += 1
            try:
                result = reconcile_run(run, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001 -- the only catch point, D-06
                logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
                self.stderr.write(f'Run pk={run.pk}: reconcile failed ({exc}) -- skipping')
                failed_count += 1
                continue

            if result.skipped_reason is not None:
                self.stderr.write(f'Run pk={run.pk}: skipped ({result.skipped_reason})')
                skipped_count += 1
                continue

            created += result.created
            updated += result.updated
            unchanged += result.unchanged
            blocked += result.blocked
            if result.blocked:
                self.stderr.write(f'Run pk={run.pk}: {result.blocked} event(s) blocked -- owned by someone else')

        if dry_run:
            self.stdout.write(
                f'Done (dry run). runs: {run_count}, '
                f'would_create: {created}, '
                f'would_update: {updated}, '
                f'would_leave_unchanged: {unchanged}, '
                f'skipped: {skipped_count}, '
                f'failed: {failed_count}, '
                f'blocked: {blocked}'
            )
        else:
            self.stdout.write(
                f'Done. runs: {run_count}, '
                f'created: {created}, '
                f'updated: {updated}, '
                f'unchanged: {unchanged}, '
                f'skipped: {skipped_count}, '
                f'failed: {failed_count}, '
                f'blocked: {blocked}'
            )
        return
