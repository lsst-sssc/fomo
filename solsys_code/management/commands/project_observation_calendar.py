"""The observation projector's backstop sweep (TRIG-03): re-project any set of LCO/SOAR
``ObservationRecord``s through ``observation_projector.project_queryset()`` with no required
arguments, so Phase 36's cron can call it with none.

Closes the gap the ``post_save`` receiver alone cannot: ``QuerySet.update()``,
``bulk_create()`` and ``backfill_lco_observations`` all bypass ``Model.save()``, so the
receiver never sees them. This command shares the receiver's own comparison rule
(``calendar_utils.preview_calendar_event_action()``) via ``project_queryset()``, so a
``--dry-run`` count can never disagree with what a real sweep would do.
"""

from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_observations.models import ObservationRecord

from solsys_code.observation_projector import PROJECTED_FACILITIES, project_queryset

# TRIG-03/D-17: single source of truth for this command's own per-facility counter keys,
# used to seed a facility that is in scope but contributed no records -- project_queryset()
# only ever returns keys for facilities it actually saw records from. Mirrors, but is
# intentionally a separate copy from, observation_projector._SWEEP_COUNTER_KEYS.
_COUNTER_KEYS = ('created', 'updated', 'unchanged', 'unprojectable', 'site_lookups', 'site_lookup_failed')


def _new_counters() -> dict[str, int]:
    """Return a fresh zeroed counter dict for one facility (a NEW dict every call)."""
    return dict.fromkeys(_COUNTER_KEYS, 0)


def _parse_proposal_arg(raw: str) -> list[str]:
    """Parse the --proposal argument into a deduped, order-preserving code list.

    Args:
        raw: the raw --proposal argument value (e.g. 'A,B,C', 'A,A,B,').

    Returns:
        list[str]: comma-split, stripped, with empty segments dropped and duplicates removed
            while preserving first-seen order. Codes keep their original casing -- proposal
            codes are case-sensitive, so this never .upper()/.lower()s a code. Unlike the
            retired sync command, there is no 'ALL' sentinel -- omitting --proposal already
            means "every record", so a second way to say the same thing is a second thing to
            get wrong.
    """
    seen: dict[str, None] = {}
    for segment in raw.split(','):
        code = segment.strip()
        if not code:
            continue
        seen.setdefault(code, None)
    return list(seen)


class Command(BaseCommand):
    """Sweep every LCO/SOAR ObservationRecord through the observation projector."""

    help = (
        'Sweep every LCO/SOAR ObservationRecord through the observation projector, projecting '
        'and refreshing its calendar event. --dry-run reports what would change without '
        'writing anything.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments. No argument is required (Phase 36 calls this bare)."""
        parser.add_argument(
            '--proposal',
            type=str,
            default=None,
            help=(
                'Restrict the sweep to these proposal code(s) -- a single code or a '
                "comma-separated list (e.g. 'A,B,C'). Exact-match only, case-sensitive. "
                'Omit to sweep every record.'
            ),
        )
        parser.add_argument(
            '--facility',
            type=str,
            choices=list(PROJECTED_FACILITIES),
            default=None,
            help='Restrict the sweep to one facility (LCO or SOAR). Omit to sweep both.',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would change without writing any CalendarEvent/ObservationRecord row.',
        )

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Sweep matching LCO/SOAR ObservationRecords through the projector and report the summary.

        Returns:
            str | None: None on completion.
        """
        dry_run = options['dry_run']
        facility_filter = options['facility']
        proposal_raw = options['proposal']

        # Facilities "in scope" for this invocation -- every scoped facility gets a summary
        # line even if it contributed zero records, so an operator can see it was considered.
        facilities_in_scope = [facility_filter] if facility_filter else list(PROJECTED_FACILITIES)

        records = ObservationRecord.objects.filter(facility__in=facilities_in_scope)
        if proposal_raw:
            codes = _parse_proposal_arg(proposal_raw)
            if codes:
                records = records.filter(parameters__proposal__in=codes)

        result = project_queryset(records, dry_run=dry_run)

        counters = {facility: _new_counters() for facility in facilities_in_scope}
        for facility, facility_counters in result['counters'].items():
            counters[facility] = facility_counters

        # Per-record failure isolation (D-17): project_queryset() itself never raises -- a
        # row it could not project is reported here, one line per row, and counted
        # separately from the per-facility 'unprojectable' tally already folded into counters.
        failed = 0
        for row in result['rows']:
            if row['action'] == 'unprojectable':
                failed += 1
                self.stderr.write(
                    f'observation_id={row["observation_id"]!r} unprojectable ({row["stage"]}) -- skipping'
                )

        summary = ' | '.join(
            f'{facility}: created: {c["created"]}, updated: {c["updated"]}, unchanged: {c["unchanged"]}, '
            f'unprojectable: {c["unprojectable"]}, site_lookups: {c["site_lookups"]}, '
            f'site_lookup_failed: {c["site_lookup_failed"]}'
            for facility, c in counters.items()
        )
        prefix = 'Done (dry run).' if dry_run else 'Done.'
        self.stdout.write(f'{prefix} failed: {failed} | {summary}')
        return
