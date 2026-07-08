import io
import pathlib
import tempfile
from datetime import date

from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent

from solsys_code.management.commands.load_telescope_runs import _iter_run_nights
from solsys_code.models import CalendarEventTelescopeLabel
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import parse_run_line


class TestLoadTelescopeRuns(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        for obscode, fields in {
            '268': dict(
                name='Magellan Clay Telescope',
                short_name='Magellan-Clay',
                lat=-29.0146,
                lon=-70.6926,
                altitude=2402,
                timezone='America/Santiago',
            ),
            '269': dict(
                name='Magellan Baade Telescope',
                short_name='Magellan-Baade',
                lat=-29.0146,
                lon=-70.6926,
                altitude=2402,
                timezone='America/Santiago',
            ),
            '809': dict(
                name='ESO, La Silla',
                short_name='NTT',
                lat=-29.2567,
                lon=-70.7300,
                altitude=2347,
                timezone='America/Santiago',
            ),
            'E10': dict(
                name='Siding Spring Observatory',
                short_name='FTS',
                lat=-31.2734,
                lon=149.0612,
                altitude=1149,
                timezone='Australia/Sydney',
            ),
        }.items():
            Observatory.objects.update_or_create(obscode=obscode, defaults=fields)

    def _write_schedule_file(self, lines: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
        """Write a schedule file to a temporary directory and return (path, tmpdir_ctx).

        The caller must use tmpdir_ctx as a context manager to ensure cleanup:

            path, tmpdir_ctx = self._write_schedule_file([...])
            with tmpdir_ctx:
                call_command(...)
        """
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'schedule.txt'
        path.write_text('\n'.join(lines) + '\n')
        return str(path), tmpdir_ctx

    def test_creates_one_event_per_night(self):
        """INGEST-01: 'NTT EFOSC2 allocation 9-13 July' creates exactly 4 CalendarEvents.

        NTT is an ESO noon-to-noon site: Tatoo's End date (13 July) is the closing
        boundary of the last night, not an observing night, so the run covers only
        the 4 nights of 9, 10, 11, 12 July (E - S, not E - S + 1).
        """
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CalendarEvent.objects.count(), 4)

    def test_iter_run_nights_eso_drops_tatoo_end_boundary(self):
        """ESO regression: 'NTT EFOSC2 allocation 9-13 July' (Tatoo '4.0 nights') expands to
        exactly the 4 evening dates 9-12 July. Tatoo's End date (13 July) is the noon-to-noon
        closing boundary of the last night, not itself an observing night, so it is dropped."""
        parsed = parse_run_line('NTT EFOSC2 allocation 9-13 July')
        year = date.today().year
        self.assertEqual(
            _iter_run_nights(parsed),
            [date(year, 7, 9), date(year, 7, 10), date(year, 7, 11), date(year, 7, 12)],
        )

    def test_iter_run_nights_magellan_both_inclusive_unchanged(self):
        """Magellan/Las Campanas keeps its E - S + 1 both-inclusive convention:
        'Magellan-Baade IMACS 17-18 July' still yields the 2 evening dates 17 and 18 July."""
        parsed = parse_run_line('Magellan-Baade IMACS 17-18 July')
        year = date.today().year
        self.assertEqual(_iter_run_nights(parsed), [date(year, 7, 17), date(year, 7, 18)])

    def test_iter_run_nights_eso_single_night(self):
        """ESO single-night run: 'NTT EFOSC2 9-10 July' (Tatoo '1.0 nights') yields exactly
        the one observing night of 9 July (10 July is the closing boundary only)."""
        parsed = parse_run_line('NTT EFOSC2 9-10 July')
        year = date.today().year
        self.assertEqual(_iter_run_nights(parsed), [date(year, 7, 9)])

    def test_iter_run_nights_eso_zero_length_range_raises(self):
        """ESO degenerate range: an End date equal to the Start date leaves no observing
        nights after dropping the closing boundary, which raises ValueError."""
        parsed = parse_run_line('NTT EFOSC2 9-9 July')
        with self.assertRaises(ValueError):
            _iter_run_nights(parsed)

    def test_event_durations_within_range(self):
        """INGEST-01: every created event has end_time > start_time and duration between 8 and 15 hours."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            for event in CalendarEvent.objects.all():
                self.assertGreater(event.end_time, event.start_time)
                duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
                self.assertGreaterEqual(duration_hours, 8.0, f'Event duration {duration_hours:.1f}h < 8h')
                self.assertLessEqual(duration_hours, 15.0, f'Event duration {duration_hours:.1f}h > 15h')

    def test_event_fields_set_from_parsed_run(self):
        """INGEST-02/D-05/D-06: event has correct telescope/instrument/title and description with all three pieces."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            events = CalendarEvent.objects.all()
            self.assertGreater(events.count(), 0)
            event = events.first()
            self.assertEqual(event.telescope, 'NTT')
            self.assertEqual(event.instrument, 'EFOSC2')
            self.assertEqual(event.title, 'NTT EFOSC2')
            # D-06: description must contain -15 deg dark-window time, status, and source line
            desc = event.description
            # Dark-window time: any ISO-format time string (e.g. 'T' separator)
            self.assertIn('T', desc, 'Expected ISO datetime string in description for dark-window time')
            # Status
            self.assertIn('allocation', desc)
            # Source line text
            self.assertIn('NTT EFOSC2 allocation 9-13 July', desc)

    def test_idempotent_rerun_no_duplicates(self):
        """INGEST-03: running command twice on same file leaves total CalendarEvent count unchanged (still 4)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            first_count = CalendarEvent.objects.count()
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            second_count = CalendarEvent.objects.count()
            self.assertEqual(first_count, 4)
            self.assertEqual(second_count, 4)

    def test_unchanged_rerun_does_not_update_existing_rows(self):
        """D-04: a re-run with unchanged schedule leaves modified timestamps untouched and reports updated: 0."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            stdout1 = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout1, stderr=io.StringIO())
            # CalendarEvent.modified has auto_now=True; it only updates on .save().
            # If unchanged, the command must NOT call .save(), so modified stays constant.
            modified_before = {e.pk: e.modified for e in CalendarEvent.objects.all()}

            stdout2 = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout2, stderr=io.StringIO())
            # No modified timestamp should have changed
            for event in CalendarEvent.objects.all():
                self.assertEqual(
                    event.modified,
                    modified_before[event.pk],
                    f'Event {event.pk} modified timestamp changed on unchanged re-run',
                )
            # Second run summary should report updated: 0
            summary = stdout2.getvalue()
            self.assertIn('updated: 0', summary)

    def test_display_01_no_sidecar_row_for_classically_scheduled_event(self):
        """DISPLAY-01: load_telescope_runs never resolves a telescope label via the LCO
        API, so events it creates have no CalendarEventTelescopeLabel row at all."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(CalendarEventTelescopeLabel.objects.count(), 0)
        event = CalendarEvent.objects.first()
        with self.assertRaises(CalendarEventTelescopeLabel.DoesNotExist):
            _ = event.telescope_label_meta

    def test_unparseable_line_logged_and_skipped(self):
        """D-02: an ambiguous 'Magellan ...' line is logged to stderr with line number; valid lines still process."""
        path, tmpdir_ctx = self._write_schedule_file(
            [
                'NTT EFOSC2 allocation 9-13 July',
                'Magellan IMACS 13-19 July (proposed)',
            ]
        )
        with tmpdir_ctx:
            stderr_buf = io.StringIO()
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=stderr_buf)
            # The NTT line should have created 4 events (ESO noon-to-noon: nights 9-12 July);
            # the ambiguous Magellan line should produce none
            self.assertEqual(CalendarEvent.objects.count(), 4)
            # stderr should contain the line number (2) and the original ambiguous line text
            err = stderr_buf.getvalue()
            self.assertIn('2', err, 'Expected line number in stderr error message')
            self.assertIn('Magellan IMACS 13-19 July (proposed)', err)

    def test_partial_night_bon_to_hhmm_sets_end_time(self):
        """INGEST-WIN-01: a BoN-HHMM window line sets end_time to HHMM UTC on d+1 morning."""
        path, tmpdir_ctx = self._write_schedule_file(['Magellan-Clay Lightspeed 18-20 July BoN-0626'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CalendarEvent.objects.count(), 3)
            for event in CalendarEvent.objects.all():
                # end_time must be clamped to 06:26 UTC (not computed sunrise)
                self.assertEqual(event.end_time.hour, 6)
                self.assertEqual(event.end_time.minute, 26)
                self.assertEqual(event.end_time.second, 0)
                # start_time is computed sunset — before midnight UTC for Santiago in July
                self.assertGreater(event.start_time.hour, 12)
                # duration is shorter than a full night but still at least 6 hours
                duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
                self.assertGreaterEqual(duration_hours, 6.0)
                self.assertLess(duration_hours, 15.0)

    def test_partial_night_hhmm_to_eon_sets_start_time(self):
        """INGEST-WIN-02: a HHMM-EoN window line sets start_time to HHMM UTC on d+1 morning."""
        path, tmpdir_ctx = self._write_schedule_file(['Magellan-Clay LDSS3 18-20 July 0646-EoN'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CalendarEvent.objects.count(), 3)
            for event in CalendarEvent.objects.all():
                # start_time must be clamped to 06:46 UTC (not computed sunset)
                self.assertEqual(event.start_time.hour, 6)
                self.assertEqual(event.start_time.minute, 46)
                self.assertEqual(event.start_time.second, 0)
                # end_time is computed sunrise — early morning UTC for Santiago in July
                self.assertLess(event.end_time.hour, 12)
