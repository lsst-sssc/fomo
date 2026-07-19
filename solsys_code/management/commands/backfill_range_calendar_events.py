import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from django.db.models import F, Q
from tom_calendar.models import CalendarEvent

from solsys_code.campaign_views import _project_calendar_event
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """One-off backfill: project CalendarEvents for already-APPROVED range-window CampaignRuns.

    Projection only fires on the approve / resolve_site POST actions (never retroactively), so
    any range-window run approved before Phase 25's per-night projection existed (e.g. the real
    GS-2026A-FT-115 row, CampaignRun pk=34) stays eventless forever without this command
    (D-07/FIX-08). Re-runnable and idempotent: a run that already has a CAMPAIGN:{pk}* event is
    skipped.
    """

    help = (
        'One-off backfill: project CalendarEvents for already-APPROVED, site-resolved '
        'range-window CampaignRuns that were approved before per-night projection existed.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report which runs would be backfilled without writing any CalendarEvent rows.',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Find qualifying runs and delegate projection to campaign_views._project_calendar_event().

        Returns:
            str | None: None on completion.
        """
        dry_run = options['dry_run']

        # Deliberately not filtered by site.observations_type -- D-07's "ground-based site"
        # describes the real motivating data (pk=34), and _project_calendar_event() already
        # routes ground vs satellite correctly, so a hypothetical satellite range candidate is
        # handled safely too.
        candidates = list(
            CampaignRun.objects.filter(
                approval_status=CampaignRun.ApprovalStatus.APPROVED,
                site__isnull=False,
                window_start__isnull=False,
            ).exclude(window_start=F('window_end'))
        )

        backfilled_count = 0
        skipped_count = 0
        failed_count = 0
        would_backfill_count = 0

        for run in candidates:
            # Trailing colon (Pitfall 2): a bare CAMPAIGN:{pk} substring match would also hit
            # CAMPAIGN:{pk}0:... for a longer pk -- combine bare-key and per-night-key lookups.
            already = CalendarEvent.objects.filter(
                Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
            ).exists()
            if already:
                skipped_count += 1
                continue

            if dry_run:
                self.stdout.write(
                    f'Would backfill run pk={run.pk} ({run.campaign.name}: {run.telescope_instrument}) '
                    f'window {run.window_start}..{run.window_end}'
                )
                would_backfill_count += 1
                continue

            try:
                _project_calendar_event(run)
                backfilled_count += 1
            except ValueError as exc:
                logger.debug('_project_calendar_event() raised for run pk=%s: %s', run.pk, exc)
                self.stderr.write(f'Run pk={run.pk}: projection failed ({exc}) -- skipping')
                failed_count += 1
                continue

        if dry_run:
            self.stdout.write(
                f'Done (dry run). candidates: {len(candidates)}, '
                f'would_backfill: {would_backfill_count}, '
                f'skipped: {skipped_count}'
            )
        else:
            self.stdout.write(
                f'Done. candidates: {len(candidates)}, '
                f'backfilled: {backfilled_count}, '
                f'skipped: {skipped_count}, '
                f'failed: {failed_count}'
            )
        return
