import io
import pathlib
import tempfile

from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent

from solsys_code.models import CalendarEventTelescopeLabel
from solsys_code.solsys_code_observatory.models import Observatory


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
        """INGEST-01: 'NTT EFOSC2 allocation 9-13 July' creates exactly 5 CalendarEvents (E - S + 1)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CalendarEvent.objects.count(), 5)

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
        """INGEST-03: running command twice on same file leaves total CalendarEvent count unchanged (still 5)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            first_count = CalendarEvent.objects.count()
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            second_count = CalendarEvent.objects.count()
            self.assertEqual(first_count, 5)
            self.assertEqual(second_count, 5)

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
            # The NTT line should have created 5 events; the ambiguous Magellan line should produce none
            self.assertEqual(CalendarEvent.objects.count(), 5)
            # stderr should contain the line number (2) and the original ambiguous line text
            err = stderr_buf.getvalue()
            self.assertIn('2', err, 'Expected line number in stderr error message')
            self.assertIn('Magellan IMACS 13-19 July (proposed)', err)
