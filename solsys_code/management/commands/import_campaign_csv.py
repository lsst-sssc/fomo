import csv
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_targets.models import TargetList

from solsys_code.campaign_utils import (
    insert_or_create_campaign_run,
    map_observation_status,
    parse_obs_window,
    resolve_site,
)
from solsys_code.models import CampaignRun


class Command(BaseCommand):
    """Bootstrap-import a campaign coordination CSV (e.g. the 3I/ATLAS sheet) into CampaignRun rows."""

    help = 'Bootstrap-import a campaign coordination CSV into CampaignRun rows (CAMP-04)'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to the campaign coordination CSV file',
        )
        parser.add_argument(
            '--campaign',
            type=str,
            required=True,
            help='Campaign TargetList name (found-or-created, D-06)',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Import campaign CSV rows into CampaignRun, row-by-row, skip-and-log on natural-key failure.

        Only a failure in a natural-key field (Telescope / Instrument, Obs. Date/UT Time
        Range) skips a row (D-05); every other column defaults to a blank/None value
        rather than aborting the row. Site resolution (D-08/D-09) never skips a row --
        an unresolved site is flagged via `site_needs_review` and counted separately.

        Returns:
            str | None: None on completion.
        """
        filepath = options['filepath']
        campaign, _ = TargetList.objects.get_or_create(name=options['campaign'])

        # D-07: single-target campaigns auto-assign that Target to every imported row.
        auto_target = campaign.targets.first() if campaign.targets.count() == 1 else None

        created_count = 0
        updated_count = 0
        unchanged_count = 0
        skipped_count = 0
        site_needs_review_count = 0

        try:
            with open(filepath, encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
        except OSError as exc:
            raise CommandError(f'Cannot open campaign CSV {filepath!r}: {exc}') from exc

        for row_num, row in enumerate(rows, start=2):  # header is row 1
            telescope_instrument = (row.get('Telescope / Instrument', '') or '').strip()
            try:
                if not telescope_instrument:
                    raise ValueError('Telescope / Instrument is required and was blank')
                obs_date, ut_start, ut_end = parse_obs_window(row.get('Obs. Date', ''), row.get('UT Time Range', ''))
            except ValueError as exc:
                self.stderr.write(f'Row {row_num}: {exc} (row: {row!r})')
                skipped_count += 1
                continue

            site_raw = row.get('Site Code', '') or ''
            site, needs_review = resolve_site(site_raw)
            if needs_review:
                site_needs_review_count += 1

            fields = {
                'target': auto_target,
                'site': site,
                'site_raw': site_raw,
                'site_needs_review': needs_review,
                'obs_date': obs_date,
                'ut_end': ut_end,
                'filters_bandpass': row.get('Filter(s)/Bandpass', '') or '',
                'observation_details': row.get('Observation Details', '') or '',
                'weather': row.get('Weather conditions or forecast', '') or '',
                'run_status': map_observation_status(row.get('Observation Status', '')),
                'approval_status': CampaignRun.ApprovalStatus.APPROVED,  # D-03: bootstrap rows are vetted backfill
                'observation_outcome': row.get('Observation Outcome', '') or '',
                'publication_plans': row.get('Publication Plans', '') or '',
                'open_to_collaboration': (row.get('Open to collaboration?', '') or '').strip().lower() == 'yes',
                'contact_person': row.get('Contact Person', '') or '',
                'contact_email': row.get('Email', '') or '',
                'comments': row.get('Other comments', '') or '',
            }

            run, action = insert_or_create_campaign_run(
                {'campaign': campaign, 'telescope_instrument': telescope_instrument, 'ut_start': ut_start},
                fields,
            )
            if action == 'created':
                created_count += 1
            elif action == 'updated':
                updated_count += 1
            else:
                unchanged_count += 1

        self.stdout.write(
            f'Done. created: {created_count}, '
            f'updated: {updated_count}, '
            f'unchanged: {unchanged_count}, '
            f'skipped: {skipped_count}, '
            f'site_needs_review: {site_needs_review_count}'
        )
        return
