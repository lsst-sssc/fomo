from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

# Site code -> telescope label, mirroring solsys_code/telescope_runs.py:SITES naming
# convention (e.g. 'FTS'). The 'coj'/'ogg' LCO entries are [ASSUMED] per RESEARCH.md
# Assumptions Log A1/A2 (web-search only, not yet confirmed against real
# ObservationRecord.parameters data for this project's LCO proposal) — confirm against
# real records before relying on this mapping in production. The 'sor' SOAR entry is
# confirmed (not [ASSUMED]) against tom_observations.facilities.soar, which hardcodes
# 'sitecode': 'sor'.
SITE_TELESCOPE_MAP = {
    'coj': 'FTS',
    'ogg': 'FTN',
    'sor': 'SOAR',
}

# TERM-01/D-04: terminal-failure status -> title prefix. COMPLETED is deliberately
# absent here (D-06 research correction) — it is terminal per
# LCOFacility().get_terminal_observing_states() (5 states) but is NOT one of the 4
# failure states returned by LCOFacility().get_failed_observing_states(), so it gets
# a clean title, same as a normally-placed record. This is a hand-typed snapshot of
# the library's current 4 failure states, not auto-derived — if LCOFacility ever adds
# a new failure state, update this dict too (the fallback below still tags it
# '[FAILED]' so a sync never silently skips a real failure state).
_FAILURE_PREFIX_BY_STATUS = {
    'WINDOW_EXPIRED': '[EXPIRED]',
    'CANCELED': '[CANCELLED]',
    'FAILURE_LIMIT_REACHED': '[FAILED]',
    'NOT_ATTEMPTED': '[FAILED]',
}


def _failure_prefix(status: str, facility: LCOFacility) -> str | None:
    """Return the terminal-failure title prefix for a status, or None if not a failure state.

    Args:
        status: the ObservationRecord's status string.
        facility: a shared LCOFacility instance.

    Returns:
        str | None: the TERM-01 prefix (e.g. '[EXPIRED]') if status is one of
            facility.get_failed_observing_states(), else None.
    """
    if status not in set(facility.get_failed_observing_states()):
        return None
    return _FAILURE_PREFIX_BY_STATUS.get(status, '[FAILED]')


def _derive_telescope(site_code: str) -> str:
    """Map an LCO/SOAR site code to a telescope label.

    Args:
        site_code: LCO/SOAR 3-letter site code (e.g. 'coj').

    Returns:
        str: telescope label (e.g. 'FTS').

    Raises:
        KeyError: if site_code is not in SITE_TELESCOPE_MAP.
    """
    try:
        return SITE_TELESCOPE_MAP[site_code]
    except KeyError:
        raise KeyError(f'Unmapped LCO site code {site_code!r}; add it to SITE_TELESCOPE_MAP') from None


def _title_for(record: ObservationRecord, telescope: str, instrument: str, facility: LCOFacility) -> str:
    """Build the CalendarEvent title for a record (D-03/D-04/D-06).

    Args:
        record: the ObservationRecord being synced.
        telescope: derived telescope label.
        instrument: instrument_type from the record's parameters.
        facility: a shared LCOFacility instance.

    Returns:
        str: the title, with a terminal-failure prefix, '[QUEUED]' prefix, or clean
            (no prefix) depending on the record's status/scheduling state.
    """
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    if record.scheduled_start is None:
        return f'[QUEUED] {telescope} {instrument}'
    return f'{telescope} {instrument}'


def _time_window(record: ObservationRecord) -> tuple[datetime, datetime]:
    """Derive the active start/end time window for a record (SYNC-02/SYNC-03).

    Args:
        record: the ObservationRecord being synced.

    Returns:
        tuple[datetime, datetime]: (start_time, end_time), timezone-aware UTC.

    Raises:
        KeyError: if scheduled_start is None and parameters lacks 'start'/'end'.
        ValueError: if parameters['start']/['end'] are not valid ISO datetime strings,
            or if scheduled_start/scheduled_end are inconsistently populated (one set,
            the other None) — a state CalendarEvent's non-nullable times cannot accept.
    """
    if record.scheduled_start is None and record.scheduled_end is None:
        # parameters['start']/['end'] are naive ISO strings (Pitfall 3) -- attach UTC
        # explicitly since LCO request-submission times are conventionally UTC.
        start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
        end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
    elif record.scheduled_start is not None and record.scheduled_end is not None:
        start_time = record.scheduled_start
        end_time = record.scheduled_end
    else:
        raise ValueError(
            f'Inconsistent schedule state: scheduled_start={record.scheduled_start!r}, '
            f'scheduled_end={record.scheduled_end!r}'
        )
    return start_time, end_time


def _build_event_fields(record: ObservationRecord, facility: LCOFacility) -> dict[str, Any]:
    """Build the full set of CalendarEvent field values for a record.

    Args:
        record: the ObservationRecord being synced.
        facility: a shared LCOFacility instance.

    Returns:
        dict[str, Any]: keyword args for CalendarEvent (url, title, description,
            start_time, end_time, telescope, instrument, proposal).

    Raises:
        KeyError: if a required parameters key (site/instrument_type/proposal/
            start/end) is missing.
        ValueError: if parameters['start']/['end'] cannot be parsed as datetimes.
    """
    telescope = _derive_telescope(record.parameters['site'])
    instrument = record.parameters['instrument_type']
    proposal = record.parameters['proposal']
    url = facility.get_observation_url(record.observation_id)
    start_time, end_time = _time_window(record)
    title = _title_for(record, telescope, instrument, facility)
    description = (
        f'Proposal: {proposal}\n'
        f'Status: {record.status}\n'
        f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime("%Y-%m-%dT%H:%M:%S")}'
    )
    return {
        'url': url,
        'title': title,
        'description': description,
        'start_time': start_time,
        'end_time': end_time,
        'telescope': telescope,
        'instrument': instrument,
        'proposal': proposal,
    }


def _parse_proposal_arg(raw: str) -> list[str] | None:
    """Parse the --proposal argument into a deduped code list, or the ALL sentinel.

    Args:
        raw: the raw --proposal argument value (e.g. 'A,B,C', 'ALL', 'A,A,B,').

    Returns:
        list[str] | None: None if raw is the case-insensitive 'all' token
            (SELECT-03/D-02 -- sync every record regardless of proposal). Otherwise
            a list of proposal codes, comma-split, stripped, with empty segments
            dropped and duplicates removed while preserving first-seen order
            (D-03). Codes keep their original casing -- proposal codes are
            case-SENSITIVE (D-01), so this never .upper()/.lower()s a code.
    """
    if raw.strip().lower() == 'all':
        return None
    seen: dict[str, None] = {}
    for segment in raw.split(','):
        code = segment.strip()
        if not code:
            continue
        seen.setdefault(code, None)
    return list(seen)


class Command(BaseCommand):
    """Sync LCO queue ObservationRecords to the FOMO calendar as CalendarEvents."""

    help = 'Sync LCO queue ObservationRecords for a proposal to CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--proposal',
            type=str,
            required=True,
            help=(
                'LCO/SOAR proposal code(s) to filter ObservationRecords by. Accepts a '
                "single code, a comma-separated list (e.g. 'A,B,C'), or the case-"
                "insensitive token 'ALL' to sync every record regardless of proposal."
            ),
        )

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Sync matching LCO/SOAR ObservationRecords to CalendarEvents.

        For each ObservationRecord(facility__in=['LCO', 'SOAR']) matching the
        --proposal selection (a comma-separated code list, or every record when
        --proposal is the ALL sentinel): create a new CalendarEvent if one does not
        exist (keyed on url), or update the existing event in place if any fields
        changed, or leave it untouched if nothing changed (SYNC-04 no-churn
        idempotency). Each record is dispatched through the facility instance
        matching its own `facility` value (SELECT-05) -- never a single shared
        instance reused across both LCO and SOAR records.

        Returns:
            str | None: None on completion.
        """
        proposal = options['proposal']
        # Eager dispatch dict, both keys unconditionally (D-06): each record is
        # processed via the facility instance matching its own `facility` value,
        # never a single shared instance reused across LCO and SOAR (SELECT-05).
        facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}

        # Per-facility counters (D-08): every facility's created/updated/unchanged/
        # skipped counts must be individually visible in the summary line.
        counters = {
            'LCO': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
            'SOAR': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
        }

        records = ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])
        codes = _parse_proposal_arg(proposal)
        if codes is not None:
            records = records.filter(parameters__proposal__in=codes)

        for record in records:
            facility = facilities.get(record.facility)
            if facility is None:
                # D-07 defensive path: an unexpected facility value on a row that
                # otherwise matched facility__in=['LCO', 'SOAR'] shouldn't happen,
                # but skip-and-log rather than abort the whole run.
                self.stderr.write(
                    f'Skipping observation_id={record.observation_id!r}: unrecognized facility {record.facility!r}'
                )
                counters.setdefault(record.facility, {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0})
                counters[record.facility]['skipped'] += 1
                continue

            try:
                fields = _build_event_fields(record, facility)
            except (KeyError, ValueError) as exc:
                self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
                counters[record.facility]['skipped'] += 1
                continue

            url = fields.pop('url')
            event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
            if created:
                counters[record.facility]['created'] += 1
            else:
                changed = any(getattr(event, field_name) != value for field_name, value in fields.items())
                if changed:
                    for field_name, value in fields.items():
                        setattr(event, field_name, value)
                    event.save()
                    counters[record.facility]['updated'] += 1
                else:
                    counters[record.facility]['unchanged'] += 1

        # D-08: per-facility breakdown. Each facility's four counts use the same
        # 'created: N' / 'updated: N' / 'unchanged: N' / 'skipped: N' phrasing as
        # the prior single-facility summary line, kept per-facility for visibility.
        summary = ' | '.join(
            f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
            f'unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}'
            for facility_name, counts in counters.items()
        )
        self.stdout.write(f'Done. proposal: {proposal}, {summary}')
        return
