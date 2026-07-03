import csv
from datetime import timedelta
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

    help = (
        'Bootstrap-import a campaign coordination CSV into CampaignRun rows (CAMP-04). '
        "WARNING: re-running this command over the same campaign always resets each row's "
        '`target` to the auto-resolved value (D-07) -- any manual correction a staff user made '
        'to `target` after a previous import will be silently overwritten on re-import (WR-07).'
    )

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

        WR-07: `fields['target']` is unconditionally set to the campaign's auto-resolved
        Target (D-07) on every row, every run -- including on a re-import that updates an
        existing row. This is expected/acceptable for this bootstrap-import command (not
        a bug), but it does mean a staff user's manual `CampaignRun.target` correction
        made via the admin between imports will be reset back to the auto-resolved value
        the next time this command runs over the same campaign.

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
        # CR-02: parse_obs_window's unparseable-time fallback always resolves to the same
        # midnight-UTC timestamp for a given obs_date. Two distinct rows sharing the same
        # telescope/campaign/date that both fall back here would otherwise collide on the
        # (campaign, telescope_instrument, ut_start) natural key and silently merge in
        # insert_or_create_campaign_run. Track how many times each such key has been seen
        # in this batch so repeats get a disambiguating offset instead of merging.
        seen_fallback_keys: dict[tuple[Any, str, Any], int] = {}

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
                obs_date, ut_start, ut_end, ut_needs_review = parse_obs_window(
                    row.get('Obs. Date', ''), row.get('UT Time Range', '')
                )
            except ValueError as exc:
                # WR-06: log only the natural-key fields needed to diagnose the skip --
                # not the full row, which also carries Contact Person/Email PII from the
                # real 3I/ATLAS sheet this command is meant to ingest.
                self.stderr.write(
                    f'Row {row_num}: {exc} (Telescope/Instrument={telescope_instrument!r}, '
                    f'Obs. Date={row.get("Obs. Date")!r})'
                )
                skipped_count += 1
                continue

            if ut_needs_review:
                # See seen_fallback_keys comment above (CR-02): disambiguate repeats of
                # the same fallback natural key within this import so distinct rows don't
                # silently merge, and flag it so the operator knows to check the source
                # rows' UT Time Range cells.
                collision_key = (campaign.pk, telescope_instrument, ut_start)
                collision_count = seen_fallback_keys.get(collision_key, 0)
                if collision_count:
                    ut_start = ut_start + timedelta(seconds=collision_count)
                    self.stderr.write(
                        f'Row {row_num}: WARNING duplicate natural key for unparseable/blank UT Time Range '
                        f'(Telescope/Instrument={telescope_instrument!r}, Obs. Date={obs_date!r}); '
                        f'offsetting ut_start by {collision_count}s to avoid merging distinct rows'
                    )
                seen_fallback_keys[collision_key] = collision_count + 1

            site_raw = row.get('Site Code', '') or ''
            site, needs_review = resolve_site(site_raw)
            if needs_review:
                site_needs_review_count += 1

            fields = {
                # WR-07: unconditionally reset to auto_target on every run, including
                # re-imports -- see handle()'s docstring for why this is expected.
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
