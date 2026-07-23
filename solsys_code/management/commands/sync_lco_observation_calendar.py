from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

from solsys_code.calendar_utils import (
    InstrumentExtractionError,
    _coarse_telescope_label,
    _derive_telescope,
    _extract_instrument,
    _resolve_placement_block,
    insert_or_create_calendar_event,
)
from solsys_code.models import CalendarEventTelescopeLabel

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


def _title_for(
    record: ObservationRecord, telescope: str, instrument: str, facility: LCOFacility, label_was_fallback: bool
) -> str:
    """Build the CalendarEvent title for a record (D-03/D-04/D-06/D-09).

    Args:
        record: the ObservationRecord being synced.
        telescope: derived telescope label.
        instrument: instrument_type from the record's parameters.
        facility: a shared LCOFacility instance.
        label_was_fallback: True if telescope is a coarse fallback label for a
            PLACED record whose live API resolution failed/timed out/returned an
            unmapped code (D-07) -- never True for a banner-stage record.

    Returns:
        str: the title, with a terminal-failure prefix, '[QUEUED]' prefix,
            '[UNVERIFIED]' prefix, or clean (no prefix), in that priority order
            (D-09): a terminal-failure prefix always wins, even over
            '[UNVERIFIED]'; '[QUEUED]' (banner stage) and '[UNVERIFIED]' (placed +
            fallback) are mutually exclusive by construction since '[UNVERIFIED]'
            only ever applies to a placed record (D-07); clean (no prefix) is a
            placed record whose label was resolved via the live API successfully.
    """
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    if record.scheduled_start is None:
        return f'[QUEUED] {telescope} {instrument}'
    if label_was_fallback:
        return f'[UNVERIFIED] {telescope} {instrument}'
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

    Implements the TELESCOPE-02/03/04 decision tree (D-01/D-02/D-07, Pitfall 4): a
    banner-stage record (scheduled_start is None) gets the coarse fallback label
    with no API call (D-01) and is never counted/flagged as a failure (D-02/D-07). A
    placed record attempts a single live API resolution via
    _resolve_placement_block; an API failure/timeout AND a successfully-returned but
    unmapped (site, telescope_code) pair are the SAME fallback bucket (Pitfall 4) --
    both set label_was_fallback=True, route to the coarse label, and increment the
    same telescope_api_failed counter.

    Args:
        record: the ObservationRecord being synced.
        facility: a shared LCOFacility instance.

    Returns:
        dict[str, Any]: keyword args for CalendarEvent (url, title, description,
            start_time, end_time, telescope, instrument, proposal, target_list),
            plus a 'telescope_api_failed' bool key that the caller (Command.handle())
            pops before constructing CalendarEvent kwargs, exactly like 'url' is
            already popped. 'target_list' is the record's Target's campaign
            TargetList, picked deterministically by name (alphabetically first) when
            the Target belongs to more than one, or None if the Target belongs to
            none.

    Raises:
        KeyError: if a required parameters key (proposal/start/end) is missing.
        ValueError: if parameters['start']/['end'] cannot be parsed as datetimes.
        InstrumentExtractionError: if _extract_instrument (D-01..D-06) finds no
            science config and no exposure-signal config anywhere in parameters.
    """
    instrument = _extract_instrument(record.parameters)
    if instrument is None:
        raise InstrumentExtractionError(
            f'No recognized configuration_type or exposure signal found in observation_id='
            f'{record.observation_id!r} parameters'
        )
    coarse = _coarse_telescope_label(instrument, record.facility)

    if record.scheduled_start is None:
        # D-01: banner stage -- no API call attempted; D-02/D-07: never counted as a
        # failure and never gets the [UNVERIFIED] prefix.
        telescope = coarse
        label_was_fallback = False
    else:
        block = _resolve_placement_block(record.observation_id, facility)
        # T-07-03: a malformed/tampered API block validates 'state' upstream but never
        # 'site'/'telescope' -- read via .get() so a missing key yields None and routes
        # to the same coarse-fallback bucket as an unmapped pair, instead of raising
        # KeyError into the generic except clause one layer up in handle().
        resolved = _derive_telescope(block.get('site'), block.get('telescope')) if block is not None else None
        if resolved is None:
            # Pitfall 4: an API call failure/timeout (block is None) and a
            # successfully-returned but unmapped (site, telescope_code) pair
            # (resolved is None) are the SAME fallback bucket.
            telescope = coarse
            label_was_fallback = True
        else:
            telescope = resolved
            label_was_fallback = False

    proposal = record.parameters['proposal']
    url = facility.get_observation_url(record.observation_id)
    start_time, end_time = _time_window(record)
    title = _title_for(record, telescope, instrument, facility, label_was_fallback)
    # Campaign TargetList association: picked deterministically by name (alphabetically
    # first) when the Target belongs to more than one, None if the Target belongs to
    # none -- CalendarEvent.target_list is a nullable FK, so None is a safe value.
    target_list = record.target.targetlist_set.order_by('name').first()
    description = (
        f'Proposal: {proposal}\n'
        f'Status: {record.status}\n'
        f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime("%Y-%m-%dT%H:%M:%S")}'
    )
    if label_was_fallback:
        # TELESCOPE-04/SYNC-09: a generic, never-exception-derived note. Not logged
        # here -- Command.handle() owns the stderr log line (caller-logging
        # discipline kept in one place).
        description += '\nTelescope label unverified: live API lookup failed or returned an unmapped code.'
    return {
        'url': url,
        'title': title,
        'description': description,
        'start_time': start_time,
        'end_time': end_time,
        'telescope': telescope,
        'instrument': instrument,
        'proposal': proposal,
        'target_list': target_list,
        # D-02 scope: True only for a PLACED record whose label was a fallback --
        # never True for a banner-stage record. Popped by handle() before
        # constructing CalendarEvent kwargs, mirroring 'url'.
        'telescope_api_failed': record.scheduled_start is not None and label_was_fallback,
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
        # skipped/extraction_failed/telescope_api_failed counts must be individually
        # visible in the summary line. 'extraction_failed' (D-06) and
        # 'telescope_api_failed' (SYNC-06/D-02) are dedicated counters distinct from
        # 'skipped' and from each other.
        counters = {
            'LCO': {
                'created': 0,
                'updated': 0,
                'unchanged': 0,
                'skipped': 0,
                'extraction_failed': 0,
                'telescope_api_failed': 0,
            },
            'SOAR': {
                'created': 0,
                'updated': 0,
                'unchanged': 0,
                'skipped': 0,
                'extraction_failed': 0,
                'telescope_api_failed': 0,
            },
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
                counters.setdefault(
                    record.facility,
                    {
                        'created': 0,
                        'updated': 0,
                        'unchanged': 0,
                        'skipped': 0,
                        'extraction_failed': 0,
                        'telescope_api_failed': 0,
                    },
                )
                counters[record.facility]['skipped'] += 1
                continue

            try:
                fields = _build_event_fields(record, facility)
            except InstrumentExtractionError as exc:
                # D-06: a fully-malformed record (no recognized configuration_type, no
                # exposure signal anywhere) is counted separately from 'skipped'.
                self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
                counters[record.facility]['extraction_failed'] += 1
                continue
            except (KeyError, ValueError) as exc:
                self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
                counters[record.facility]['skipped'] += 1
                continue

            url = fields.pop('url')
            telescope_api_failed = fields.pop('telescope_api_failed')
            if telescope_api_failed:
                # SYNC-09/D-11: fixed, generic message -- never interpolates a
                # caught exception (no {exc}/str(exc)/repr(exc) here). SYNC-07: the
                # record still gets a CalendarEvent below; the run continues.
                self.stderr.write(
                    f'Telescope API lookup failed or returned an unmapped code for '
                    f'observation_id={record.observation_id!r}; using fallback label.'
                )
                counters[record.facility]['telescope_api_failed'] += 1

            event, action = insert_or_create_calendar_event({'url': url}, fields)
            counters[record.facility][action] += 1

            # Phase 8 / DISPLAY-01: always reconcile the sidecar row to the current
            # telescope_api_failed signal, regardless of whether CalendarEvent's own
            # fields changed -- kept as a separate statement, never folded into
            # `fields` or `changed`. is_verified reflects the outcome of the most
            # recent sync run that included this record, not real-time state.
            CalendarEventTelescopeLabel.objects.update_or_create(
                event=event, defaults={'is_verified': not telescope_api_failed}
            )

        # D-08: per-facility breakdown. Each facility's six counts use the same
        # 'created: N' / 'updated: N' / 'unchanged: N' / 'skipped: N' /
        # 'extraction_failed: N' / 'telescope_api_failed: N' phrasing as the prior
        # single-facility summary line, kept per-facility for visibility.
        # extraction_failed (D-06) and telescope_api_failed (SYNC-06) are each
        # distinct from skipped and from each other.
        summary = ' | '.join(
            f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
            f'unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}, '
            f'extraction_failed: {counts["extraction_failed"]}, '
            f'telescope_api_failed: {counts["telescope_api_failed"]}'
            for facility_name, counts in counters.items()
        )
        self.stdout.write(f'Done. proposal: {proposal}, {summary}')
        return
