from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from solsys_code.calendar_utils import insert_or_create_calendar_event
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ESO_NOON_TO_NOON_SITES, ParsedRun, get_site, parse_run_line, sun_event

# The event start_time is a computed sun-event time (telescope_runs.sun_event()), not a
# stable external identifier. It drifts by a second or two between independent ingests of
# the same (site, night) because astropy's IERS Earth-orientation data (UT1-UTC / polar
# motion) is refreshed between runs (see debug/start-time-idempotency-key.md). Match an
# existing CalendarEvent whose start_time is within this window of the freshly computed
# value instead of requiring an exact datetime match, so re-ingesting an unchanged schedule
# updates the existing night rather than silently creating a near-duplicate row. The window
# is ~2 orders of magnitude larger than the largest drift observed (~2s) yet ~3 orders of
# magnitude smaller than the ~24h spacing between any two legitimately distinct events for a
# single telescope+instrument, so it can never merge two genuinely different nights.
_START_TIME_MATCH_TOLERANCE = timedelta(minutes=5)

# Classical-schedule status -> title prefix (D-02). Only 'cancelled' has a visible
# prefix today, mirroring sync_lco_observation_calendar's _FAILURE_PREFIX_BY_STATUS
# idiom; '[CANCELLED]' is already a member of calendar_display_extras._TERMINAL_PREFIXES
# so the terminal box-shadow ring is inherited with no templatetag change.
_CLASSICAL_STATUS_PREFIX = {'cancelled': '[CANCELLED]'}


def _resolve_window_time(window: str, sunset, sunrise, evening_date: date) -> datetime:
    """Convert a window token to a UTC datetime for a single observing night.

    Args:
        window: 'BoN' for computed sunset, 'EoN' for computed sunrise, or a
            4-digit UTC HHMM string. HHMM < 1200 is treated as next-morning
            UTC (evening_date + 1 day); HHMM >= 1200 is evening_date UTC.
        sunset: astropy Time of sunset for this night.
        sunrise: astropy Time of sunrise for this night.
        evening_date: the calendar date of the observing evening.

    Returns:
        datetime: UTC-aware datetime for this window boundary.
    """
    upper = window.upper()
    if upper == 'BON':
        return sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    if upper == 'EON':
        return sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    hh, mm = int(window[:2]), int(window[2:])
    base_date = evening_date + timedelta(days=1) if hh < 12 else evening_date
    return datetime(base_date.year, base_date.month, base_date.day, hh, mm, 0, tzinfo=dt_timezone.utc)


def _iter_run_nights(parsed: ParsedRun) -> list[date]:
    """Returns one evening date per observing night, per the site's night convention.

    Las Campanas (Magellan) Start and End dates are BOTH inclusive observing
    nights, so a run yields E - S + 1 nights (INGEST-01;
    docs/design/telescope_runs_calendar.rst "Night convention"). ESO sites
    (``ESO_NOON_TO_NOON_SITES``, e.g. NTT / La Silla) transcribe their ranges
    verbatim from ESO's Tatoo tool, whose displayed END date is the noon-to-noon
    closing boundary of the last night rather than an observing night itself, so
    their last observing night is day2 - 1 (E - S nights).

    Args:
        parsed: a ParsedRun from parse_run_line().

    Returns:
        list[date]: evening dates for each night of the run.

    Raises:
        ValueError: if day2 < day1 (a descending or malformed same-month day
            range -- e.g. a typo like '20-5 July' -- that parse_run_line does
            not reject upstream; genuine cross-month ranges are already
            rejected at parse time by parse_run_line, PR-REVIEW-F2), or if an
            ESO noon-to-noon range leaves no observing nights after dropping
            its closing boundary (day2 <= day1).
    """
    if parsed.day2 < parsed.day1:
        raise ValueError(f'Invalid or descending same-month day range (day2 < day1): {parsed!r}')
    n_nights = parsed.day2 - parsed.day1 + 1
    if parsed.telescope in ESO_NOON_TO_NOON_SITES:
        # Tatoo's End date is the closing noon boundary of the last night, not an
        # observing night -- drop it so E - S nights remain.
        n_nights -= 1
        if n_nights < 1:
            raise ValueError(
                f'ESO noon-to-noon run range has no observing nights after dropping its '
                f'closing boundary (day1={parsed.day1}, day2={parsed.day2}): {parsed!r}'
            )
    first_night = date(parsed.year, parsed.month, parsed.day1)
    return [first_night + timedelta(days=i) for i in range(n_nights)]


class Command(BaseCommand):
    """Load classical telescope run lines from a file and create or update CalendarEvents."""

    help = 'Load classical telescope run lines from a file and create/update CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to a text file of classical run lines (one per line)',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Load classical schedule lines and create or update CalendarEvents.

        For each observing night derived from a run line: create a new CalendarEvent
        if one does not exist, or update the existing event if any fields have changed,
        or leave it untouched if nothing has changed.

        Returns:
            str | None: None on completion.
        """
        filepath = options['filepath']
        created_count = 0
        updated_count = 0
        unchanged_count = 0
        skipped_count = 0
        lines_processed = 0

        try:
            with open(filepath, encoding='utf-8') as f:
                file_lines = list(f)
        except OSError as exc:
            raise CommandError(f'Cannot open schedule file {filepath!r}: {exc}') from exc

        for line_num, line in enumerate(file_lines, start=1):
            if not line.strip():
                continue
            lines_processed += 1
            try:
                parsed = parse_run_line(line)
                site = get_site(parsed.telescope)
                nights = _iter_run_nights(parsed)
                for d in nights:
                    sunset, sunrise = sun_event(site, d, 'sun')
                    dark_start, dark_end = sun_event(site, d, 'dark')
                    start_time = _resolve_window_time(parsed.start_window or 'BoN', sunset, sunrise, d)
                    end_time = _resolve_window_time(parsed.end_window or 'EoN', sunset, sunrise, d)
                    dark_start_dt = dark_start.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
                    dark_end_dt = dark_end.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)

                    prefix = _CLASSICAL_STATUS_PREFIX.get(parsed.status)
                    title = (
                        f'{prefix} {parsed.telescope} {parsed.instrument}'
                        if prefix
                        else f'{parsed.telescope} {parsed.instrument}'
                    )
                    description = (
                        f'Dark window (-15 deg, UTC): {dark_start_dt.isoformat()} to {dark_end_dt.isoformat()}\n'
                        f'Status: {parsed.status}\n'
                        f'Source line: {line.strip()}'
                    )

                    event, action = insert_or_create_calendar_event(
                        {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
                        {'end_time': end_time, 'title': title, 'description': description},
                        start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
                    )
                    if action == 'created':
                        created_count += 1
                    elif action == 'updated':
                        updated_count += 1
                    else:
                        unchanged_count += 1
            except (ValueError, Observatory.DoesNotExist) as exc:
                self.stderr.write(f'Line {line_num}: {exc} (line text: {line.strip()!r})')
                skipped_count += 1
                continue

        self.stdout.write(
            f'Done. lines processed: {lines_processed}, '
            f'created: {created_count}, '
            f'updated: {updated_count}, '
            f'unchanged: {unchanged_count}, '
            f'skipped: {skipped_count}'
        )
        return
