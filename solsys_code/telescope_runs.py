import re
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, time
from math import sqrt
from zoneinfo import ZoneInfo

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from solsys_code.solsys_code_observatory.models import Observatory

# Maps telescope name to MPC observatory code. `Observatory` (looked up via
# get_site()) remains the single source of truth for location and timezone.
SITES = {
    'Magellan-Clay': '268',
    'Magellan-Baade': '269',
    'NTT': '809',
    'FTS': 'E10',
}

# Known classical-schedule status words/phrases (case-insensitive), per
# docs/design/telescope_runs_calendar.rst "Classical Run Input Format".
KNOWN_STATUSES = {'allocation', 'proposed', 'confirmed', 'cancelled', 'not confirmed'}

# Full month names and 3-letter abbreviations, case-insensitive, mapped to 1-12.
_MONTH_NAMES = {
    'jan': 1,
    'january': 1,
    'feb': 2,
    'february': 2,
    'mar': 3,
    'march': 3,
    'apr': 4,
    'april': 4,
    'may': 5,
    'jun': 6,
    'june': 6,
    'jul': 7,
    'july': 7,
    'aug': 8,
    'august': 8,
    'sep': 9,
    'september': 9,
    'oct': 10,
    'october': 10,
    'nov': 11,
    'november': 11,
    'dec': 12,
    'december': 12,
}

_MONTH_NAME_PATTERN = '|'.join(sorted(_MONTH_NAMES, key=len, reverse=True))

# month-after-range, e.g. 'Jul 8-12'
_MONTH_AFTER_RANGE = re.compile(
    rf"""
    (?P<month1>{_MONTH_NAME_PATTERN})\s+
    (?P<day1>\d{{1,2}})
    \s*-\s*
    (?P<day2>\d{{1,2}})
    """,
    re.VERBOSE | re.IGNORECASE,
)

# month-before-range, e.g. '9-13 July'
_MONTH_BEFORE_RANGE = re.compile(
    rf"""
    (?P<day1>\d{{1,2}})
    \s*-\s*
    (?P<day2>\d{{1,2}})
    \s+
    (?P<month1>{_MONTH_NAME_PATTERN})
    """,
    re.VERBOSE | re.IGNORECASE,
)

# cross-month range, e.g. '28 December-2 January'
_CROSS_MONTH_RANGE = re.compile(
    rf"""
    (?P<day1>\d{{1,2}})\s+
    (?P<month1>{_MONTH_NAME_PATTERN})
    \s*-\s*
    (?P<day2>\d{{1,2}})\s+
    (?P<month2>{_MONTH_NAME_PATTERN})
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Status as a parenthesized phrase, e.g. '(proposed)' or '(not confirmed)'.
_PAREN_STATUS = re.compile(r'\(([^)]+)\)')


def get_site(name: str) -> Observatory:
    """Resolves a telescope name to its Observatory record.

    Args:
        name: Telescope name, a key of SITES (e.g. 'Magellan-Clay').

    Returns:
        Observatory: the observatory record for this telescope's site.

    Raises:
        Observatory.DoesNotExist: if name is not a key in SITES, or no
            Observatory record exists for the resolved MPC obscode.
    """
    try:
        obscode = SITES[name]
    except KeyError as exc:
        raise Observatory.DoesNotExist(f'No site registered in SITES for telescope {name!r}') from exc
    return Observatory.objects.get(obscode=obscode)


def horizon_dip(altitude_m: float) -> u.Quantity:
    """Horizon dip for an observer at altitude_m metres.

    dip = 1.76 arcmin * sqrt(altitude in metres).

    This is the Nautical Almanac dip formula (terrestrial refraction k~1/6
    folded into the spherical-geometry estimate dip ~ sqrt(2h/R), R=6371 km);
    see docs/design/telescope_runs_calendar.rst ("Astronomy: Night
    Boundaries") for the derivation.

    Args:
        altitude_m: Observer altitude above sea level, in metres.

    Returns:
        u.Quantity: dip angle (e.g. 1.4376 deg for 2402 m).

    Raises:
        ValueError: if altitude_m is None or negative.
    """
    if altitude_m is None or altitude_m < 0:
        raise ValueError(f'altitude_m must be a non-negative number, got {altitude_m!r}')
    return 1.76 * sqrt(altitude_m) * u.arcmin


def _solar_altitude(times: Time, location) -> np.ndarray:
    """Solar altitude in degrees for an array of Time objects at a location.

    Args:
        times: astropy Time array of evaluation epochs.
        location: astropy EarthLocation of the observer.

    Returns:
        np.ndarray: solar altitude in degrees for each time.
    """
    sun = get_sun(times)
    altaz = sun.transform_to(AltAz(obstime=times, location=location))
    return altaz.alt.deg


def _find_crossing(
    anchor: Time, location, threshold_deg: float, search_hours: float = 24, coarse_step_min: float = 1.0
) -> list[Time]:
    """Finds UTC times where solar altitude crosses threshold_deg.

    Performs a coarse scan over the window [anchor, anchor + search_hours],
    then refines each sign change with bisection to sub-second precision.
    Anchoring at local noon of the observing date (see _local_noon_utc) and
    scanning forward search_hours=24 guarantees both the evening sunset/dark
    crossing of that date and the following morning's sunrise/dark-end
    crossing fall within the window, in chronological (set, then rise) order.

    Args:
        anchor: astropy Time at the start of the search window (local noon).
        location: astropy EarthLocation of the observer.
        threshold_deg: solar altitude threshold to find crossings of, in degrees.
        search_hours: total width of the search window, in hours.
        coarse_step_min: coarse scan step size, in minutes.

    Returns:
        list[Time]: UTC times of each altitude crossing, in chronological order.
    """
    # +coarse_step_min so the window covers a full, closed [0, search_hours] range
    # (np.arange's exclusive upper bound would otherwise leave the last minute unscanned).
    offsets = np.arange(0, search_hours * 60 + coarse_step_min, coarse_step_min) * u.min
    times = anchor + offsets
    alt = _solar_altitude(times, location)
    crossings = []
    for i in range(len(alt) - 1):
        if (alt[i] - threshold_deg) * (alt[i + 1] - threshold_deg) < 0:
            # Bisection refine between times[i] and times[i+1]
            lo, hi = times[i], times[i + 1]
            lo_alt = alt[i]
            for _ in range(10):  # ~1/1024 of 1-min step -> sub-second precision
                mid = lo + (hi - lo) / 2
                mid_alt = _solar_altitude(Time([mid]), location)[0]
                if (mid_alt - threshold_deg) * (lo_alt - threshold_deg) < 0:
                    hi = mid
                else:
                    lo, lo_alt = mid, mid_alt
            crossings.append(lo)
    return crossings


def _local_noon_utc(local_date: date_cls, tz_name: str) -> Time:
    """Local noon of local_date, converted to UTC, as an astropy Time.

    Args:
        local_date: the local calendar date.
        tz_name: IANA timezone name for the site (e.g. 'America/Santiago').

    Returns:
        Time: local noon of local_date, expressed as a UTC astropy Time.
    """
    tz = ZoneInfo(tz_name)
    local_noon = datetime.combine(local_date, time(12, 0), tzinfo=tz)
    return Time(local_noon.astimezone(ZoneInfo('UTC')))


def sun_event(site: Observatory, date: date_cls, kind: str) -> tuple[Time, Time]:
    """Computes UTC sun-event crossing times for an observing night.

    Args:
        site: Observatory instance (from get_site()).
        date: local calendar date of sunset; the returned events cover the
            observing night starting on the evening of this date.
        kind: 'sun' for the dip-corrected sunset/sunrise threshold
            (-(0.833 + dip) degrees), or 'dark' for the -15 degree
            dark-window threshold (no dip correction).

    Returns:
        tuple[Time, Time]: (setting, rising) as astropy.time.Time objects,
            UTC scale.

    Raises:
        ValueError: if kind is not 'sun' or 'dark'; if site.timezone is
            unset; or if the solar altitude does not cross threshold
            exactly twice in the 24h window following local noon (e.g. a
            high-latitude site in summer where the sun never sets, or never
            gets dark).
    """
    if not site.timezone:
        raise ValueError(
            f'Observatory {site.short_name!r} (obscode={site.obscode}) has no timezone set; '
            'set Observatory.timezone (IANA name, e.g. "America/Santiago") before calling sun_event().'
        )

    location = site.to_earth_location()
    anchor = _local_noon_utc(date, site.timezone)

    if kind == 'sun':
        dip = horizon_dip(site.altitude)
        # 0.833 deg = standard solar semi-diameter (~16') + horizon refraction (~34')
        threshold = -(0.833 + dip.to_value(u.deg))
    elif kind == 'dark':
        threshold = -15.0
    else:
        raise ValueError(f"kind must be 'sun' or 'dark', got {kind!r}")

    crossings = _find_crossing(anchor, location, threshold, search_hours=24)
    if len(crossings) != 2:
        raise ValueError(
            f'Expected 2 sun-event crossings for {site.short_name} on {date} '
            f'(kind={kind!r}), got {len(crossings)}: {crossings}. '
            'This can happen at high latitudes when the sun never sets or never '
            'reaches the requested threshold (e.g. midnight sun or no astronomical darkness).'
        )
    return crossings[0], crossings[1]


@dataclass(frozen=True)
class ParsedRun:
    """Structured result of parse_run_line().

    Attributes:
        telescope: resolved SITES key (e.g. 'NTT').
        instrument: instrument name as it appears in the run line (may be
            hyphenated, e.g. 'Proto-Lightspeed').
        status: lowercase status word/phrase, e.g. 'allocation', 'proposed',
            'not confirmed'. Defaults to 'allocation' if absent (D-05).
        year: four-digit year. Defaults to the current year (PARSE-03), or
            current year + 1 for a run that starts in December and ends in
            January (year roll-over).
        month: month number (1-12) of day1 (the start of the run).
        day1: first day of the run (inclusive).
        day2: last day of the run (inclusive).
    """

    telescope: str
    instrument: str
    status: str
    year: int
    month: int
    day1: int
    day2: int


def _resolve_telescope(token: str) -> str:
    """Resolves a telescope token to a SITES key by prefix match (D-01).

    Args:
        token: the first whitespace-delimited token of a run line.

    Returns:
        str: the resolved SITES key.

    Raises:
        ValueError: if token is a prefix of zero or 2+ SITES keys.
    """
    if token in SITES:
        return token
    candidates = [key for key in SITES if key.startswith(token)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f'Ambiguous telescope {token!r}: matches multiple SITES keys {candidates}; '
            'use a more specific telescope name (e.g. "Magellan-Clay" or "Magellan-Baade").'
        )
    raise ValueError(f'Unknown telescope {token!r}: does not match any SITES key {list(SITES)}')


def _resolve_status(line: str) -> tuple[str, str]:
    """Extracts and validates the status word/phrase from a run line (D-04/05/06).

    Args:
        line: the full run line (including any parenthesized status).

    Returns:
        tuple[str, str]: (status, remainder) where status is the lowercase
            KNOWN_STATUSES member (defaulting to 'allocation' if absent) and
            remainder is the line with the status token(s) removed.

    Raises:
        ValueError: if a parenthesized phrase or trailing status-shaped word
            is present but not in KNOWN_STATUSES.
    """
    paren_match = _PAREN_STATUS.search(line)
    if paren_match:
        candidate = paren_match.group(1).strip().lower()
        if candidate not in KNOWN_STATUSES:
            raise ValueError(
                f'Unrecognized status {candidate!r} in {line!r}; known statuses are {sorted(KNOWN_STATUSES)}'
            )
        remainder = line[: paren_match.start()] + line[paren_match.end() :]
        return candidate, remainder

    # Multi-word statuses (e.g. 'not confirmed') checked before single words.
    for status in sorted(KNOWN_STATUSES, key=len, reverse=True):
        match = re.search(rf'(?<!\S){re.escape(status)}(?!\S)', line, re.IGNORECASE)
        if match:
            remainder = line[: match.start()] + line[match.end() :]
            return status, remainder

    return 'allocation', line


def parse_run_line(line: str) -> ParsedRun:
    """Parses a free-text classical-schedule run line into structured fields.

    Expected format (per docs/design/telescope_runs_calendar.rst "Classical
    Run Input Format"): ``telescope instrument [status] daterange [(status)]``,
    e.g. 'NTT EFOSC2 allocation 9-13 July' or 'Magellan Proto-Lightspeed Jul
    8-12 (proposed)'. The date range may have the month name before or after
    the day range, and no year is given (year defaults per PARSE-03).

    Args:
        line: a single run-line string.

    Returns:
        ParsedRun: the parsed telescope, instrument, status, year, month,
            day1, day2.

    Raises:
        ValueError: if line is empty, the telescope token does not resolve to
            exactly one SITES key (D-01), the status is unrecognized (D-06),
            or no date range can be found.
    """
    stripped = line.strip()
    if not stripped:
        raise ValueError('parse_run_line() received an empty line')

    status, remainder = _resolve_status(stripped)

    # Date range: try month-after-range ('Jul 8-12'), cross-month
    # ('28 December-2 January'), then month-before-range ('9-13 July').
    match = _MONTH_AFTER_RANGE.search(remainder)
    if match:
        day1 = int(match.group('day1'))
        day2 = int(match.group('day2'))
        month = _MONTH_NAMES[match.group('month1').lower()]
    else:
        match = _CROSS_MONTH_RANGE.search(remainder)
        if match:
            day1 = int(match.group('day1'))
            day2 = int(match.group('day2'))
            month = _MONTH_NAMES[match.group('month1').lower()]
        else:
            match = _MONTH_BEFORE_RANGE.search(remainder)
            if not match:
                raise ValueError(f'Could not find a date range (e.g. "9-13 July" or "Jul 8-12") in {line!r}')
            day1 = int(match.group('day1'))
            day2 = int(match.group('day2'))
            month = _MONTH_NAMES[match.group('month1').lower()]

    # Year (PARSE-03): default to current year; roll over to next year if the
    # run starts in December and ends in January (cross-year range).
    year = date_cls.today().year
    if month == 12 and day2 < day1:
        year += 1

    # Telescope (token 0) and instrument (token 1, possibly hyphenated).
    before_range = remainder[: match.start()]
    tokens = before_range.split()
    if len(tokens) < 2:
        raise ValueError(f'Could not find telescope and instrument tokens in {line!r}')
    telescope_token, instrument = tokens[0], tokens[1]

    # D-06: any remaining word(s) between instrument and the date range are a
    # status-shaped token that must be in KNOWN_STATUSES (already checked and
    # consumed by _resolve_status if recognized).
    leftover = ' '.join(tokens[2:]).strip()
    if leftover:
        raise ValueError(f'Unrecognized status {leftover!r} in {line!r}; known statuses are {sorted(KNOWN_STATUSES)}')

    telescope = _resolve_telescope(telescope_token)

    return ParsedRun(
        telescope=telescope,
        instrument=instrument,
        status=status,
        year=year,
        month=month,
        day1=day1,
        day2=day2,
    )
