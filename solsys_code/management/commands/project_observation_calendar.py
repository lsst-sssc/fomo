"""The observation projector's backstop sweep (TRIG-03): re-project any set of LCO/SOAR
``ObservationRecord``s through ``observation_projector.project_queryset()`` with no required
arguments, so Phase 36's cron can call it with none.

Closes the gap the ``post_save`` receiver alone cannot: ``QuerySet.update()``,
``bulk_create()`` and ``backfill_lco_observations`` all bypass ``Model.save()``, so the
receiver never sees them. This command shares the receiver's own comparison rule
(``calendar_utils.preview_calendar_event_action()``) via ``project_queryset()``, so a
``--dry-run`` count agrees with what a real sweep would do for every field derived from a
record's own already-stored state -- with one documented exception (WR-02): a dry run never
performs the one-time observed-site lookup, so ``site_lookups`` is always 0 and a record whose
only pending change is the coarse-to-observed telescope token is reported ``unchanged`` by
``--dry-run`` but ``updated`` by the real sweep that follows it.
"""

from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import OBSERVED_SITE_PARAMETER_KEYS, derive_telescope, resolve_placement_block
from solsys_code.observation_projector import PROJECTED_FACILITIES, SWEEP_COUNTER_KEYS, project_queryset, stage_for

# TRIG-03/D-17/IN-04: this command's own per-facility counter keys, used to seed a facility
# that is in scope but contributed no records -- project_queryset() only ever returns keys for
# facilities it actually saw records from. Imported from observation_projector rather than
# hand-copied, so the six counters have one source of truth.
_COUNTER_KEYS = SWEEP_COUNTER_KEYS

# D-08: the one-time observed-site lookup only ever fires once a record has reached one of
# these two successful-terminal stages -- a queued or placed record triggers no portal call,
# however many sweeps run.
_SUCCESSFUL_TERMINAL_STAGES = ('observed', 'completed-no-block')


def _new_counters() -> dict[str, int]:
    """Return a fresh zeroed counter dict for one facility (a NEW dict every call)."""
    return dict.fromkeys(_COUNTER_KEYS, 0)


def resolve_observed_site(record: ObservationRecord, facility: Any) -> tuple[dict[str, int] | None, str | None]:
    """The D-07/D-08 one-time observed-site lookup for a successful-terminal record.

    Lives on the command side, never in the projector (34-RESEARCH.md Pitfall 4/Assumption
    A3) -- the projector never makes a network call. Calls the portal-block resolver at most
    once per record, ever: a record whose ``parameters`` already carries ``observed_site``
    (from a previous sweep) is never looked up again, and a queued/placed record is never
    looked up at all.

    A ``None`` block (the resolver's own never-raise failure return) and a returned-but
    -unmapped ``(site, telescope)`` pair are treated as ONE bucket (D-08): both leave the
    coarse aperture token in place, store nothing, and are retried on the next sweep -- two
    counters or two log messages for what is operationally one situation is exactly the
    pitfall this rule exists to prevent.

    There is no ``except`` clause in this function: ``resolve_placement_block()`` already
    converts every failure (network error, auth error, malformed body) to ``None``
    internally, so there is no caught exception object anywhere in this path that a message
    could accidentally embed (SYNC-09/D-13).

    Args:
        record: the ObservationRecord to resolve. Mutated and saved in place on a successful
            lookup (``record.parameters`` gains the three ``OBSERVED_SITE_PARAMETER_KEYS``,
            written with ``update_fields=['parameters']``).
        facility: the record's facility instance (``observation_projector.facility_for()``).

    Returns:
        tuple[dict[str, int] | None, str | None]: ``(None, None)`` when no lookup applies
            (not a successful-terminal stage, or already resolved). ``({'site_lookups': 1},
            None)`` on a successful lookup that wrote ``parameters``. ``({'site_lookup_failed':
            1}, message)`` when the lookup returned no usable block or an unmapped pair --
            ``message`` is a fixed, generic stderr line naming only the observation_id, never
            a caught exception's value.
    """
    site_key, telescope_key, enclosure_key = OBSERVED_SITE_PARAMETER_KEYS
    stage = stage_for(record, facility)
    if stage not in _SUCCESSFUL_TERMINAL_STAGES:
        return None, None
    if record.parameters.get(site_key):
        return None, None

    block = resolve_placement_block(record.observation_id, facility)
    site = block.get('site') if block is not None else None
    telescope = block.get('telescope') if block is not None else None
    enclosure = block.get('enclosure') if block is not None else None

    if derive_telescope(site, telescope) is None:
        message = f'observation_id={record.observation_id!r}: observed-site lookup unavailable -- using fallback label.'
        return {'site_lookup_failed': 1}, message

    record.parameters[site_key] = site
    record.parameters[telescope_key] = telescope
    # IN-05: observed_enclosure has no reader anywhere in this codebase as of Phase 34 -- no
    # later phase in .planning/ROADMAP.md (35 Allocation Layer, 36 Unattended Operation, 37
    # Status Vocabulary) names it as something it will consume. Stored anyway, alongside site
    # and telescope, because it is part of the same portal placement block
    # (resolve_placement_block()) and OBSERVED_SITE_PARAMETER_KEYS already reserves the key --
    # dropping the write now would silently lose data a future consumer might still want,
    # while keeping it costs nothing (same block, same save).
    record.parameters[enclosure_key] = enclosure
    record.save(update_fields=['parameters'])
    return {'site_lookups': 1}, None


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

        def hook(record: ObservationRecord, facility: Any) -> dict[str, int] | None:
            increment, message = resolve_observed_site(record, facility)
            if message:
                self.stderr.write(message)
            return increment

        # D-17: --dry-run performs no site lookup either -- project_queryset() itself also
        # guards this (never calls the hook when dry_run=True), but omitting the hook
        # entirely here means a dry run can never even construct the closure over self.stderr.
        result = project_queryset(records, dry_run=dry_run, pre_fields_hook=None if dry_run else hook)

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
