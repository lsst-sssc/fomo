from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from solsys_code.calendar_utils import insert_or_create_calendar_event
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import ParsedRun, get_site, parse_run_line, sun_event


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
    """Returns one evening date per observing night (E - S + 1 nights, INGEST-01).

    Args:
        parsed: a ParsedRun from parse_run_line().

    Returns:
        list[date]: evening dates for each night of the run.

    Raises:
        ValueError: if day2 < day1 (cross-month ranges are not supported in Phase 3).
    """
    if parsed.day2 < parsed.day1:
        raise ValueError(f'Cross-month run ranges not yet supported in Phase 3: {parsed!r}')
    first_night = date(parsed.year, parsed.month, parsed.day1)
    n_nights = parsed.day2 - parsed.day1 + 1
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

                    title = f'{parsed.telescope} {parsed.instrument}'
                    description = (
                        f'Dark window (-15 deg, UTC): {dark_start_dt.isoformat()} to {dark_end_dt.isoformat()}\n'
                        f'Status: {parsed.status}\n'
                        f'Source line: {line.strip()}'
                    )

                    event, action = insert_or_create_calendar_event(
                        {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
                        {'end_time': end_time, 'title': title, 'description': description},
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
