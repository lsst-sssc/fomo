import csv
from datetime import date
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

# WR-09: the D-05 natural-key columns. If the CSV's header doesn't include these exactly
# (e.g. a renamed column in a future sheet export), every row would otherwise be silently
# skipped one-by-one with no single top-level diagnostic that the header shape is wrong.
_REQUIRED_HEADERS = ('Telescope / Instrument', 'Obs. Date', 'UT Time Range')


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

        The window-schema natural key (`campaign`, `telescope_instrument`, `window_start`)
        is date-only (single-night collapse: `window_end` == `window_start` == the parsed
        `Obs. Date`), so two rows sharing the same telescope/campaign/date now collide on
        the key regardless of whether their `UT Time Range` cells parsed. A genuine
        same-date collision is logged and skipped rather than silently merged into one
        `CampaignRun` (D-07/D-08) -- Phase 20 owns real range/TBD disambiguation.

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
        # CR-02 (window-schema rethink): window_start is date-only, so the natural key is
        # now (campaign, telescope_instrument, obs_date) -- any two rows sharing that key
        # collide, not just rows that both fell back to parse_obs_window's
        # unparseable-UT midnight default. Track keys already seen in this batch so a
        # genuine duplicate is logged and skipped rather than silently merged into one
        # CampaignRun via insert_or_create_campaign_run's get_or_create.
        seen_window_keys: set[tuple[Any, str, date]] = set()

        try:
            with open(filepath, encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                # WR-09: fail fast on the header shape itself rather than silently
                # skipping every row one-by-one if a required column is missing/renamed.
                missing_headers = [h for h in _REQUIRED_HEADERS if h not in (reader.fieldnames or [])]
                if missing_headers:
                    raise CommandError(
                        f'Campaign CSV {filepath!r} is missing required column(s): {missing_headers!r}. '
                        f'Found columns: {reader.fieldnames!r}'
                    )
                rows = list(reader)
        except OSError as exc:
            raise CommandError(f'Cannot open campaign CSV {filepath!r}: {exc}') from exc

        for row_num, row in enumerate(rows, start=2):  # header is row 1
            telescope_instrument = (row.get('Telescope / Instrument', '') or '').strip()
            try:
                if not telescope_instrument:
                    raise ValueError('Telescope / Instrument is required and was blank')
                # ut_start/ut_end are no longer stored (window schema is date-only) -- only
                # obs_date (-> window_start/window_end) and ut_needs_review (collision
                # log wording) are used.
                obs_date, _ut_start, _ut_end, ut_needs_review = parse_obs_window(
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

            # See seen_window_keys comment above (CR-02): window_start collapses to
            # obs_date, so this check now runs for every row, not just ones whose UT
            # Time Range fell back to the unparseable-UT default.
            collision_key = (campaign.pk, telescope_instrument, obs_date)
            if collision_key in seen_window_keys:
                self.stderr.write(
                    f'Row {row_num}: WARNING duplicate natural key '
                    f'(Telescope/Instrument={telescope_instrument!r}, Obs. Date={obs_date!r}); '
                    f'skipping row to avoid merging distinct observations into one CampaignRun -- '
                    f"check the source rows' UT Time Range cells"
                    + (' (unparseable/blank UT Time Range)' if ut_needs_review else '')
                )
                skipped_count += 1
                continue
            seen_window_keys.add(collision_key)

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
                'window_end': obs_date,  # single-night collapse: window_end == window_start == obs_date
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
                {'campaign': campaign, 'telescope_instrument': telescope_instrument, 'window_start': obs_date},
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
