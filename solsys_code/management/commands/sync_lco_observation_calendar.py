from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any
from urllib.parse import urljoin

import requests
from django import forms
from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord

# (site, aperture_class) -> 'SITECODE-CLASS' telescope label (TELESCOPE-01/D-03/D-04).
# Verified, real-data-grounded inventory of the 7 real LCO-network sites this
# codebase's installed LCOSettings/SOARSettings actually confirm (tlv/Wise Observatory
# is deliberately excluded -- confirmed absent from both installed get_sites()
# implementations at the 07-01 Task 1 checkpoint; see 07-01-SUMMARY.md Deviations).
# 'coj'/'ogg'/'sor' migrate the 3 pre-existing entries (D-05) -- 'coj'/'ogg' confirmed
# 2m0 (FTS/FTN), 'sor' confirmed 4m0 (SOAR, tom_observations.facilities.soar hardcodes
# 'sitecode': 'sor'). 'elp'/'lsc'/'cpt'/'tfn' confirmed by operator (LCO staff) at the
# 07-01 Task 1 checkpoint -- see 07-01-SUMMARY.md -- as standard 1m-network sites
# hosting both 1m0 and 0m4 telescope classes.
SITE_TELESCOPE_MAP = {
    ('coj', '2m0'): 'COJ-2m0',
    ('ogg', '2m0'): 'OGG-2m0',
    ('sor', '4m0'): 'SOR-4m0',
    ('elp', '1m0'): 'ELP-1m0',
    ('elp', '0m4'): 'ELP-0m4',
    ('lsc', '1m0'): 'LSC-1m0',
    ('lsc', '0m4'): 'LSC-0m4',
    ('cpt', '1m0'): 'CPT-1m0',
    ('cpt', '0m4'): 'CPT-0m4',
    ('tfn', '1m0'): 'TFN-1m0',
    ('tfn', '0m4'): 'TFN-0m4',
}

# SYNC-08/D-10: explicit timeout, single attempt, no retry/backoff loop. This is the
# first explicit HTTP timeout introduced anywhere in solsys_code/ -- there is no
# existing precedent to follow (JPLSBDBQuery.run_query() calls requests.get() with no
# timeout at all, a known anti-pattern, not a convention to mirror here).
_API_TIMEOUT_SECONDS = 10

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

# EXTRACT-01/D-01: configuration_type values that mark a config as the scientifically
# meaningful one, as opposed to a calibration config (ARC/LAMP_FLAT, SOAR) or an
# NRES-specific config (never in scope). Confirmed against installed tom_observations:
# ocs.py:1025-1030/1213 (flat c_{N}_configuration_type key), lco.py:740-743,757-760,998
# (EXPOSE/REPEAT_EXPOSE/SPECTRUM/REPEAT_SPECTRUM), soar.py:103,118 (SPECTRUM/ARC/
# LAMP_FLAT), blanco.py:177 (STANDARD -- vocabulary adopted now for forward
# compatibility per CONTEXT.md D-01; Blanco facility scope itself stays deferred).
_SCIENCE_CONFIGURATION_TYPES = {'EXPOSE', 'REPEAT_EXPOSE', 'SPECTRUM', 'REPEAT_SPECTRUM', 'STANDARD'}

# D-04: LCO MUSCAT records have no flat c_N_exposure_time, only per-channel
# c_N_ic_M_exposure_time_{suffix} keys (confirmed lco.py:585-596,
# LCOMuscatImagingObservationForm). Detect population by ANY of the 4 channels being
# truthy -- more lenient than the real submission form's all-4-required validation.
_MUSCAT_CHANNEL_SUFFIXES = ('g', 'r', 'i', 'z')


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


def _aperture_class_from_telescope_code(telescope_code: str) -> str | None:
    """Extract the aperture-class token (D-04 vocabulary) from a 4-char telescope code.

    Args:
        telescope_code: e.g. '0m4b', '1m0a', '2m0a' (from the API response's
            'telescope' key) -- a 3-char aperture-class token plus a trailing
            dome-instance letter suffix.

    Returns:
        str | None: '0m4'/'1m0'/'2m0'/'4m0' (strips the trailing dome-instance
            letter), or None if the code doesn't match the expected
            3-char-class + 1-char-suffix shape (routes the caller to fallback
            per TELESCOPE-03).
    """
    if len(telescope_code) >= 4 and telescope_code[:3] in {'0m4', '1m0', '2m0', '4m0'}:
        return telescope_code[:3]
    return None


def _derive_telescope(site: str, telescope_code: str) -> str | None:
    """Map a resolved (site, telescope_code) pair to a verified label via SITE_TELESCOPE_MAP.

    Args:
        site: 3-letter site code from the API response (e.g. 'lsc').
        telescope_code: 4-char telescope code from the API response (e.g. '1m0a').

    Returns:
        str | None: the verified label (e.g. 'LSC-1m0'), or None if the
            (site, class) pair isn't in SITE_TELESCOPE_MAP or telescope_code's
            aperture class couldn't be parsed -- caller falls back to the
            coarse instrument-class label (TELESCOPE-03). Never raises.
    """
    aperture_class = _aperture_class_from_telescope_code(telescope_code)
    if aperture_class is None:
        return None
    return SITE_TELESCOPE_MAP.get((site, aperture_class))


def _resolve_placement_block(observation_id: str, facility: LCOFacility) -> dict[str, Any] | None:
    """Call the LCO Observation Portal API once to resolve a placed record's block.

    Issues a single, timeout-bounded GET to /api/requests/{observation_id}/observations/
    and selects the same COMPLETED-first-else-PENDING block that
    OCSFacility.get_observation_status() selects for scheduled_start/scheduled_end, so
    telescope resolution and timing always come from the same block (Pitfall 3).

    Args:
        observation_id: the record's LCO observation_id.
        facility: a shared LCOFacility/SOARFacility instance (for portal_url/api_key
            settings and auth header construction).

    Returns:
        dict[str, Any] | None: the matched block dict (with 'site'/'enclosure'/
            'telescope'/'state' keys) on success, or None if the API call failed,
            timed out, or returned no usable COMPLETED/PENDING block. Never raises --
            every failure mode (network error, library auth/validation exception,
            malformed/non-JSON body, missing 'state' key) is caught and converted to
            None so the caller always falls through to the coarse fallback (SYNC-07:
            a per-record failure never aborts the run). The except clause never
            references, stringifies, or logs the caught exception (SYNC-09/D-11) --
            ImproperCredentialsException/forms.ValidationError embed response.content
            directly and must never be logged verbatim.
    """
    try:
        response = make_request(
            'GET',
            urljoin(
                facility.facility_settings.get_setting('portal_url'),
                f'/api/requests/{observation_id}/observations/',
            ),
            headers=facility._portal_headers(),
            timeout=_API_TIMEOUT_SECONDS,
        )
        blocks = response.json()
    except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):
        return None

    current_block = None
    for block in blocks:
        if block.get('state') == 'COMPLETED':
            current_block = block
            break
        elif block.get('state') == 'PENDING':
            current_block = block
    return current_block


def _has_muscat_exposure_signal(parameters: dict[str, Any], n: int) -> bool:
    """Check whether config c_{n} has a populated MUSCAT per-channel exposure key (D-04).

    Args:
        parameters: the record's parameters dict.
        n: config index (1-5).

    Returns:
        bool: True if any of c_{n}_ic_1_exposure_time_{g,r,i,z} is truthy.
    """
    return any(parameters.get(f'c_{n}_ic_1_exposure_time_{suffix}') for suffix in _MUSCAT_CHANNEL_SUFFIXES)


def _find_science_config(parameters: dict[str, Any]) -> int | None:
    """Scan c_1..c_5 for the first config whose configuration_type is a science type (D-01).

    Args:
        parameters: the record's parameters dict.

    Returns:
        int | None: the config index (1-5) of the first config whose
            c_{n}_configuration_type is in _SCIENCE_CONFIGURATION_TYPES, or None if no
            config has a recognized science configuration_type.
    """
    for n in range(1, 6):
        configuration_type = parameters.get(f'c_{n}_configuration_type')
        if configuration_type in _SCIENCE_CONFIGURATION_TYPES:
            return n
    return None


def _find_exposure_signal_config(parameters: dict[str, Any]) -> int | None:
    """Scan c_1..c_5 for the first config with a populated exposure signal (D-02 fallback).

    Args:
        parameters: the record's parameters dict.

    Returns:
        int | None: the config index (1-5) of the first config with a truthy flat
            c_{n}_exposure_time, or (D-04) a populated MUSCAT per-channel exposure key,
            or None if no config has any exposure signal at all.
    """
    for n in range(1, 6):
        if parameters.get(f'c_{n}_exposure_time') or _has_muscat_exposure_signal(parameters, n):
            return n
    return None


def _extract_instrument(parameters: dict[str, Any]) -> str | None:
    """Extract the scientifically meaningful instrument_type from a record's parameters.

    Scans the real c_1..c_5-prefixed multi-configuration shape (D-01..D-06): first by
    configuration_type whitelist (science vs. SOAR calibration/NRES configs), falling
    back to the first config with a populated exposure signal (flat or MUSCAT
    per-channel) if no config has a recognized configuration_type. If no c_N_* config
    exists at all (today's legacy single-config shape, pre-dating the c_N_* fields),
    falls back to the flat 'instrument_type' key itself -- D-02's "original EXTRACT-01
    heuristic" applied to the degenerate single-config case.

    Args:
        parameters: the record's parameters dict.

    Returns:
        str | None: the selected config's c_{n}_instrument_type value (D-03, unchanged
            in format), the flat 'instrument_type' value for the legacy shape, or None
            if neither signal selects any config and no flat key is present (D-06 total
            extraction failure -- the caller routes this to a dedicated counter, never
            the existing 'skipped' counter).
    """
    n = _find_science_config(parameters)
    if n is None:
        n = _find_exposure_signal_config(parameters)
    if n is not None:
        return parameters.get(f'c_{n}_instrument_type')
    return parameters.get('instrument_type')


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


class InstrumentExtractionError(Exception):
    """Raised when _extract_instrument finds no usable config (D-06 total extraction failure).

    Caught separately in handle() so a fully-malformed record is routed to the
    dedicated 'extraction_failed' counter, never silently merged into 'skipped'.
    """


def _build_event_fields(record: ObservationRecord, facility: LCOFacility) -> dict[str, Any]:
    """Build the full set of CalendarEvent field values for a record.

    Args:
        record: the ObservationRecord being synced.
        facility: a shared LCOFacility instance.

    Returns:
        dict[str, Any]: keyword args for CalendarEvent (url, title, description,
            start_time, end_time, telescope, instrument, proposal).

    Raises:
        KeyError: if a required parameters key (site/proposal/start/end) is missing.
        ValueError: if parameters['start']/['end'] cannot be parsed as datetimes.
        InstrumentExtractionError: if _extract_instrument (D-01..D-06) finds no
            science config and no exposure-signal config anywhere in parameters.
    """
    # NOTE (Plan 01 -> Plan 02 handoff): this single-class flat lookup is an interim
    # shim, not the live-API resolution path. Plan 01's _derive_telescope(site,
    # telescope_code) now requires a real 4-char telescope code (from
    # _resolve_placement_block's API response), which this still-flat
    # record.parameters['site'] cannot supply. Plan 02 (Wave 2) replaces this call
    # entirely with the API-call + fallback decision tree per TELESCOPE-02/03/04 --
    # it is deliberately NOT implemented here (07-01-PLAN.md's objective explicitly
    # scopes _build_event_fields/Command.handle wiring out of Plan 01). This shim only
    # exists so the 19 pre-existing regression tests (which only ever use the 3
    # single-aperture-class legacy sites coj/ogg/sor) keep passing unmodified until
    # Plan 02 lands.
    _LEGACY_SINGLE_CLASS_SITES = {'coj': '2m0', 'ogg': '2m0', 'sor': '4m0'}
    site_code = record.parameters['site']
    aperture_class = _LEGACY_SINGLE_CLASS_SITES.get(site_code)
    telescope = SITE_TELESCOPE_MAP.get((site_code, aperture_class)) if aperture_class else None
    if telescope is None:
        raise KeyError(f'Unmapped LCO site code {site_code!r}; add it to SITE_TELESCOPE_MAP')
    instrument = _extract_instrument(record.parameters)
    if instrument is None:
        raise InstrumentExtractionError(
            f'No recognized configuration_type or exposure signal found in observation_id='
            f'{record.observation_id!r} parameters'
        )
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
        # skipped/extraction_failed counts must be individually visible in the summary
        # line. 'extraction_failed' (D-06) is a dedicated counter distinct from
        # 'skipped', for fully-malformed records with no extractable instrument config.
        counters = {
            'LCO': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0, 'extraction_failed': 0},
            'SOAR': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0, 'extraction_failed': 0},
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
                    {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0, 'extraction_failed': 0},
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

        # D-08: per-facility breakdown. Each facility's five counts use the same
        # 'created: N' / 'updated: N' / 'unchanged: N' / 'skipped: N' / 'extraction_failed: N'
        # phrasing as the prior single-facility summary line, kept per-facility for
        # visibility. extraction_failed (D-06) is distinct from skipped.
        summary = ' | '.join(
            f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
            f'unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}, '
            f'extraction_failed: {counts["extraction_failed"]}'
            for facility_name, counts in counters.items()
        )
        self.stdout.write(f'Done. proposal: {proposal}, {summary}')
        return
