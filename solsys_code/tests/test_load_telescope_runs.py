import io
import pathlib
import tempfile
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from unittest import mock

import astropy.units as u
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code import telescope_runs as tr
from solsys_code.allocation_projector import ALLOC_URL_NAMESPACE
from solsys_code.management.commands.load_telescope_runs import _iter_run_nights
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import get_site, parse_run_line


def _drifted_sun_event(offset_seconds: float):
    """Wrap the real sun_event() to add a fixed offset, simulating cross-session IERS drift.

    The real sun-event math is deterministic within a process, so a genuine cross-session
    start_time drift (astropy refreshing its IERS Earth-orientation data between ingests)
    cannot be produced by simply re-running in one test. This shifts every returned crossing
    time by `offset_seconds`, reproducing the observed ~2s dev-DB drift deterministically.
    """
    real = tr.sun_event

    def _wrapped(site, d, kind):
        setting, rising = real(site, d, kind)
        return setting + offset_seconds * u.s, rising + offset_seconds * u.s

    return _wrapped


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

    def _alloc_events(self):
        """Every CalendarEvent this command's allocation projection wrote (ALLOC: namespace,
        Phase 35) -- the twin of the pre-cutover blank-url (url='') filter this test module
        used before the allocation cutover."""
        return CalendarEvent.objects.filter(url__startswith=ALLOC_URL_NAMESPACE)

    def test_creates_one_event_per_night(self):
        """INGEST-01: 'NTT EFOSC2 allocation 9-13 July' creates exactly 4 ALLOC: CalendarEvents.

        NTT is an ESO noon-to-noon site: Tatoo's End date (13 July) is the closing
        boundary of the last night, not an observing night, so the run covers only
        the 4 nights of 9, 10, 11, 12 July (E - S, not E - S + 1).
        """
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(self._alloc_events().count(), 4)
            self.assertEqual(CalendarEvent.objects.exclude(url__startswith=ALLOC_URL_NAMESPACE).count(), 0)

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
        """INGEST-01: every allocation night has end_time > start_time and duration between
        8 and 15 hours."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            for event in self._alloc_events():
                self.assertGreater(event.end_time, event.start_time)
                duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
                self.assertGreaterEqual(duration_hours, 8.0, f'Event duration {duration_hours:.1f}h < 8h')
                self.assertLessEqual(duration_hours, 15.0, f'Event duration {duration_hours:.1f}h > 15h')

    def test_event_fields_set_from_parsed_run(self):
        """INGEST-02/D-05/D-06: allocation night has correct telescope/instrument/title and
        description with all three pieces (dark window, status, source line)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            events = self._alloc_events()
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

    def test_cancelled_line_gets_bracket_cancelled_title_prefix(self):
        """D-02: a cancelled classical run gets a '[CANCELLED] ' title prefix; description
        still carries the original status/source-line body (plus the shared writer's own
        'Run status: Cancelled' line -- see TestClassicalCalendarUnchangedByCutover for the
        documented divergence this introduces vs. pre-cutover output)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 9-13 July (cancelled)'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            events = self._alloc_events()
            self.assertGreater(events.count(), 0)
            event = events.first()
            self.assertEqual(event.title, '[CANCELLED] NTT EFOSC2')
            self.assertIn('Status: cancelled', event.description)

    def test_non_cancelled_statuses_keep_unprefixed_title(self):
        """D-02: the other four KNOWN_STATUSES words leave the title unprefixed.

        Each status gets its own night/run so the four statuses don't collide on the
        (telescope, instrument, window) natural key.
        """
        lines = [
            'NTT EFOSC2 allocation 9-10 July',
            'NTT EFOSC2 proposed 10-11 July',
            'NTT EFOSC2 confirmed 11-12 July',
            'NTT EFOSC2 12-13 July (not confirmed)',
        ]
        path, tmpdir_ctx = self._write_schedule_file(lines)
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            events = self._alloc_events()
            self.assertEqual(events.count(), 4)
            for event in events:
                with self.subTest(title=event.title):
                    self.assertEqual(event.title, 'NTT EFOSC2')

    def test_reingest_without_cancelled_reverts_title_prefix(self):
        """RESEARCH Pitfall 4: re-ingesting after the 'cancelled' word is removed reverts the
        title to unprefixed, with no stale prefix, no duplicate row, and no double prefix.

        The status word does not participate in source_identifier, so both imports match
        the same CampaignRun and the same allocation nights are updated in place."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 9-13 July (cancelled)'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            cancelled_count = self._alloc_events().count()
            cancelled_pks = set(self._alloc_events().values_list('pk', flat=True))
            for event in self._alloc_events():
                self.assertTrue(event.title.startswith('[CANCELLED] '))

        path2, tmpdir_ctx2 = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx2:
            call_command('load_telescope_runs', path2, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CampaignRun.objects.count(), 1)
            self.assertEqual(self._alloc_events().count(), cancelled_count)
            self.assertEqual(set(self._alloc_events().values_list('pk', flat=True)), cancelled_pks)
            for event in self._alloc_events():
                self.assertEqual(event.title, 'NTT EFOSC2')

    def test_idempotent_rerun_no_duplicates(self):
        """INGEST-03: running the command twice on the same file reports the run and every
        night unchanged, and leaves the total ALLOC: event count unchanged (still 4)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            first_count = self._alloc_events().count()
            self.assertEqual(CampaignRun.objects.count(), 1)

            stdout2 = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout2, stderr=io.StringIO())
            second_count = self._alloc_events().count()

            self.assertEqual(first_count, 4)
            self.assertEqual(second_count, 4)
            self.assertEqual(CampaignRun.objects.count(), 1)
            summary = stdout2.getvalue()
            self.assertIn('unchanged: 1', summary)
            self.assertIn('nights -- created: 0, updated: 0, unchanged: 4', summary)

    def test_unchanged_rerun_does_not_update_existing_rows(self):
        """D-04: a re-run with an unchanged schedule leaves modified timestamps untouched
        and reports the run unchanged and every night unchanged."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            stdout1 = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout1, stderr=io.StringIO())
            # CalendarEvent.modified has auto_now=True; it only updates on .save().
            # If unchanged, the projector must NOT call .save(), so modified stays constant.
            modified_before = {e.pk: e.modified for e in self._alloc_events()}

            stdout2 = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout2, stderr=io.StringIO())
            # No modified timestamp should have changed
            for event in self._alloc_events():
                self.assertEqual(
                    event.modified,
                    modified_before[event.pk],
                    f'Event {event.pk} modified timestamp changed on unchanged re-run',
                )
            # Second run summary should report the run and every night unchanged.
            summary = stdout2.getvalue()
            self.assertIn('unchanged: 1', summary)
            self.assertIn('nights -- created: 0, updated: 0, unchanged: 4', summary)

    def test_reimport_with_drifted_sun_event_does_not_duplicate_or_move_the_night(self):
        """Regression, proven against the new ALLOC: mechanism (replaces the retired
        start-time-tolerance-match test): unlike the old blank-url writer, an allocation
        night is matched by (run.pk, night) -- never by a computed start_time -- so a
        sun_event() drift between imports (astropy IERS Earth-orientation refresh across
        sessions) can never create a near-duplicate row, and D-13 forbids rewriting an
        existing night's stored start_time/end_time in place regardless.

        Reproduces the real dev-DB drift shape (a couple of seconds across sessions)
        deterministically by shifting the second ingest's sun-event times by +2s.
        """
        path, tmpdir_ctx = self._write_schedule_file(['Magellan-Baade IMACS 17-18 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            first_count = self._alloc_events().count()
            first_starts = {e.pk: e.start_time for e in self._alloc_events()}

            stdout2 = io.StringIO()
            with mock.patch(
                'solsys_code.allocation_projector.sun_event',
                side_effect=_drifted_sun_event(2.0),
            ):
                call_command('load_telescope_runs', path, stdout=stdout2, stderr=io.StringIO())

            # No duplicate rows: the two Magellan-Baade nights are matched, not re-created,
            # and their stored start_time never moves even though sun_event() drifted.
            self.assertEqual(first_count, 2)
            self.assertEqual(self._alloc_events().count(), 2)
            for event in self._alloc_events():
                self.assertEqual(event.start_time, first_starts[event.pk])
            self.assertIn('nights -- created: 0, updated: 0, unchanged: 2', stdout2.getvalue())

    def test_display_01_allocation_night_gets_a_calendar_event_meta_row_self_attributed_to_its_own_run(self):
        """INVERTED from pre-cutover behavior (D-08/D-09, Phase 35): the allocation
        projector self-attributes every night it writes via _link_event_to_run(), so a
        classical import's ALLOC: events now DO get a CalendarEventMeta row pointing at
        their own run -- unlike the old blank-url writer, which never went through the
        reconciler's attribution writer at all and left no companion row behind."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        run = CampaignRun.objects.get()
        events = self._alloc_events()
        self.assertEqual(events.count(), 4)
        self.assertEqual(CalendarEventMeta.objects.filter(run=run).count(), 4)
        for event in events:
            self.assertEqual(event.telescope_label_meta.run_id, run.pk)

    def test_classical_schedule_never_adopts_a_projector_owned_event(self):
        """WR-07, strengthened for Phase 35: the classical path has no calendar lookup at
        all -- allocation nights are matched by (run.pk, night), never by (telescope,
        instrument, start_time) -- so an observation-projector-owned event sharing that
        triple is byte-identical after the import, never adopted or rewritten."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            # A real run first, purely to learn the deterministic (telescope, instrument,
            # start_time) triple this schedule line computes for its first night --
            # sun-event math is deterministic within one process (see
            # test_reimport_with_drifted_sun_event_does_not_duplicate_or_move_the_night's
            # own docstring).
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            first_night = self._alloc_events().order_by('start_time').first()
            telescope, instrument, start_time, end_time = (
                first_night.telescope,
                first_night.instrument,
                first_night.start_time,
                first_night.end_time,
            )
            CampaignRun.objects.all().delete()  # cascades its own allocation nights
            CalendarEvent.objects.all().delete()

            # Simulate a projector-owned event that already occupies that exact
            # (telescope, instrument, start_time) triple, carrying a non-ALLOC: url.
            owned = CalendarEvent.objects.create(
                title='[O] NTT owned-by-projector',
                telescope=telescope,
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
                url='https://observe.lco.global/requests/999999/',
            )

            stdout = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout, stderr=io.StringIO())

            owned.refresh_from_db()
            self.assertEqual(owned.title, '[O] NTT owned-by-projector')  # untouched, not adopted
            self.assertEqual(owned.url, 'https://observe.lco.global/requests/999999/')
            matching = CalendarEvent.objects.filter(telescope=telescope, instrument=instrument, start_time=start_time)
            self.assertEqual(matching.count(), 2)  # the owned event, plus a new ALLOC: allocation night
            alloc_match = matching.get(url__startswith=ALLOC_URL_NAMESPACE)
            self.assertNotEqual(alloc_match.pk, owned.pk)
            self.assertIn('nights -- created: 4', stdout.getvalue())

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
            # The NTT line should have created 4 nights (ESO noon-to-noon: nights 9-12 July);
            # the ambiguous Magellan line should produce none
            self.assertEqual(self._alloc_events().count(), 4)
            # stderr should contain the line number (2) and the original ambiguous line text
            err = stderr_buf.getvalue()
            self.assertIn('2', err, 'Expected line number in stderr error message')
            self.assertIn('Magellan IMACS 13-19 July (proposed)', err)

    def test_unknown_status_mapping_is_logged_and_skipped_not_uncaught(self):
        """35-REVIEW.md WR-09: a status word missing from `_CLASSICAL_RUN_STATUS` must be
        skipped and logged per-line (D-02's existing vocabulary), not escape as an uncaught
        `KeyError` that aborts the whole import mid-run."""
        path, tmpdir_ctx = self._write_schedule_file(
            [
                'NTT EFOSC2 allocation 9-13 July',
                'Magellan-Baade IMACS 17-18 July (confirmed)',
            ]
        )
        with tmpdir_ctx:
            with mock.patch.dict(
                'solsys_code.management.commands.load_telescope_runs._CLASSICAL_RUN_STATUS',
                {'confirmed': CampaignRun.RunStatus.PLANNED},  # drop 'allocation'
                clear=True,
            ):
                stderr_buf = io.StringIO()
                call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=stderr_buf)

            # The NTT line's status has no mapping now -- skipped, logged, nothing written.
            self.assertFalse(CampaignRun.objects.filter(telescope_instrument='NTT/EFOSC2').exists())
            err = stderr_buf.getvalue()
            self.assertIn('1', err)
            self.assertIn("'allocation'", err)
            # The other, still-mapped line still processes -- one bad status never aborts
            # the whole import.
            self.assertTrue(CampaignRun.objects.filter(telescope_instrument='Magellan-Baade/IMACS').exists())

    def test_cross_month_line_logged_and_skipped(self):
        """PR-REVIEW-F2: a genuine cross-month run line is rejected at parse time (fail-fast)
        and logged to stderr with its line number, not crashed on; the valid line still
        processes into its allocation nights."""
        path, tmpdir_ctx = self._write_schedule_file(
            [
                'NTT EFOSC2 28 December-2 January',
                'NTT EFOSC2 allocation 9-13 July',
            ]
        )
        with tmpdir_ctx:
            stderr_buf = io.StringIO()
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=stderr_buf)
            # The cross-month line should produce no run; the valid NTT line should still
            # create 4 nights (ESO noon-to-noon: nights 9-12 July).
            self.assertEqual(self._alloc_events().count(), 4)
            err = stderr_buf.getvalue()
            self.assertIn('1', err, 'Expected line number in stderr error message')
            self.assertIn('NTT EFOSC2 28 December-2 January', err)

    def test_partial_night_bon_to_hhmm_sets_end_time(self):
        """INGEST-WIN-01: a BoN-HHMM window line sets end_time to HHMM UTC on d+1 morning."""
        path, tmpdir_ctx = self._write_schedule_file(['Magellan-Clay Lightspeed 18-20 July BoN-0626'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(self._alloc_events().count(), 3)
            for event in self._alloc_events():
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
            self.assertEqual(self._alloc_events().count(), 3)
            for event in self._alloc_events():
                # start_time must be clamped to 06:46 UTC (not computed sunset)
                self.assertEqual(event.start_time.hour, 6)
                self.assertEqual(event.start_time.minute, 46)
                self.assertEqual(event.start_time.second, 0)
                # end_time is computed sunrise — early morning UTC for Santiago in July
                self.assertLess(event.end_time.hour, 12)

    def test_campaign_omitted_leaves_target_list_none(self):
        """Regression guard: without --campaign, the run's campaign and every night's
        target_list are None (zero behavior change from before the flag existed)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())
            run = CampaignRun.objects.get()
            self.assertIsNone(run.campaign)
            events = self._alloc_events()
            self.assertGreater(events.count(), 0)
            for event in events:
                self.assertIsNone(event.target_list)

    def test_campaign_matching_sets_target_list_on_every_event(self):
        """--campaign NAME with a matching TargetList sets run.campaign and target_list on
        every allocation night."""
        campaign = TargetList.objects.create(name='Test Campaign')
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command(
                'load_telescope_runs', path, campaign=campaign.name, stdout=io.StringIO(), stderr=io.StringIO()
            )
            run = CampaignRun.objects.get()
            self.assertEqual(run.campaign_id, campaign.pk)
            events = self._alloc_events()
            self.assertGreater(events.count(), 0)
            for event in events:
                self.assertEqual(event.target_list_id, campaign.pk)

    def test_campaign_no_match_raises_and_creates_nothing(self):
        """--campaign NAME with no matching TargetList raises CommandError before any
        CampaignRun or CalendarEvent is created (fail fast)."""
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            with self.assertRaises(CommandError):
                call_command(
                    'load_telescope_runs',
                    path,
                    campaign='No Such Campaign',
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                )
            self.assertEqual(CampaignRun.objects.count(), 0)
            self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_campaign_multiple_matches_raises(self):
        """--campaign NAME matching more than one TargetList raises CommandError."""
        TargetList.objects.create(name='Ambiguous Campaign')
        TargetList.objects.create(name='Ambiguous Campaign')
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            with self.assertRaises(CommandError):
                call_command(
                    'load_telescope_runs',
                    path,
                    campaign='Ambiguous Campaign',
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                )

    def test_campaign_no_churn_on_rerun(self):
        """Re-running the same file with the same --campaign reports the run and every
        night unchanged (no-churn on the target_list FK field)."""
        campaign = TargetList.objects.create(name='No Churn Campaign')
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command(
                'load_telescope_runs', path, campaign=campaign.name, stdout=io.StringIO(), stderr=io.StringIO()
            )
            first_count = self._alloc_events().count()

            stdout2 = io.StringIO()
            call_command('load_telescope_runs', path, campaign=campaign.name, stdout=stdout2, stderr=io.StringIO())
            self.assertEqual(self._alloc_events().count(), first_count)
            summary = stdout2.getvalue()
            self.assertIn('unchanged: 1', summary)
            self.assertIn('nights -- created: 0, updated: 0, unchanged: 4', summary)


class TestMalformedTimezoneSkipsOneLine(TestCase):
    """35-REVIEW.md NF-21: a mistyped `Observatory.timezone` must be skipped and logged
    per-line -- the runbook's stated "one bad row never aborts the whole run" invariant --
    not escape as an uncaught `ZoneInfoNotFoundError` (a `KeyError` subclass, 35-VERIFICATION.md
    L234) that aborts the whole import mid-batch with a bare traceback, no summary, and no
    subsequent lines processed."""

    @classmethod
    def setUpTestData(cls) -> None:
        Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santigo',  # typo -- the exact reproduction from 35-REVIEW.md
        )
        Observatory.objects.create(
            obscode='269',
            name='Magellan Baade Telescope',
            short_name='Magellan-Baade',
            lat=-29.0146,
            lon=-70.6926,
            altitude=2402,
            timezone='America/Santiago',
        )

    def _write_schedule_file(self, lines: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'schedule.txt'
        path.write_text('\n'.join(lines) + '\n')
        return str(path), tmpdir_ctx

    def test_malformed_timezone_skips_only_its_own_line(self):
        path, tmpdir_ctx = self._write_schedule_file(
            [
                'NTT EFOSC2 allocation 9-13 July',
                'Magellan-Baade IMACS 17-18 July',
            ]
        )
        with tmpdir_ctx:
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout_buf, stderr=stderr_buf)

        # stderr names the Observatory, its obscode and the offending timezone -- not a
        # bare key repr from the parse bucket.
        err = stderr_buf.getvalue()
        self.assertIn('NTT', err)
        self.assertIn('809', err)
        self.assertIn('America/Santigo', err)

        # The command reaches its summary and reports the bad line as skipped.
        summary = stdout_buf.getvalue()
        self.assertIn('skipped: 1', summary)

        # The SECOND line was still processed -- one bad row never aborts the whole run.
        self.assertTrue(CampaignRun.objects.filter(telescope_instrument='Magellan-Baade/IMACS').exists())
        self.assertEqual(CalendarEvent.objects.filter(url__startswith=ALLOC_URL_NAMESPACE).count(), 2)

        # The bad line's CampaignRun does not exist -- transaction.atomic() rolled it back.
        self.assertFalse(CampaignRun.objects.filter(telescope_instrument='NTT/EFOSC2').exists())


def _expected_boundary(token: str | None, sunset, sunrise, night, *, at_full_night: str) -> datetime:
    """Test-local, from-first-principles expression of the BoN/EoN/HHMM sub-night rule
    (D-04) -- deliberately NOT calling any production helper, so this states the contract
    an allocation night's start/end must satisfy rather than echoing the implementation.

    Args:
        token: None, 'BoN'/'EoN' (case-insensitive), or a 4-digit HHMM UTC string.
        sunset: astropy Time of sunset for this night.
        sunrise: astropy Time of sunrise for this night.
        night: the site-local observing night (evening date).
        at_full_night: 'BoN' or 'EoN' -- which sun-event boundary a None token means.

    Returns:
        datetime: the UTC-aware datetime this boundary must equal.
    """
    if token is None:
        token = at_full_night
    upper = token.upper()
    if upper == 'BON':
        return sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    if upper == 'EON':
        return sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    hh, mm = int(token[:2]), int(token[2:])
    base_date = night + timedelta(days=1) if hh < 12 else night
    return datetime(base_date.year, base_date.month, base_date.day, hh, mm, 0, tzinfo=dt_timezone.utc)


class TestClassicalCalendarUnchangedByCutover(TestCase):
    """ROADMAP Success Criterion 4's real gate: a classical import's per-night calendar is
    pinned field-by-field against the contract load_telescope_runs upheld before the
    allocation cutover, for a representative set of lines covering both night conventions,
    a cancelled status, and a partial-night window.

    One deliberate, documented divergence from the pre-cutover output: a cancelled line's
    event description now ALSO carries the shared writer's 'Run status: Cancelled' line
    (event_description()), because allocation descriptions are composed through the same
    helper every other campaign-run event uses -- so a staff mark_cancelled action reaches
    allocation nights too. Titles, spans, telescope, instrument and event counts are
    unchanged.
    """

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
        }.items():
            Observatory.objects.update_or_create(obscode=obscode, defaults=fields)

    def _write_schedule_file(self, lines: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'schedule.txt'
        path.write_text('\n'.join(lines) + '\n')
        return str(path), tmpdir_ctx

    def test_calendar_matches_pre_cutover_contract_field_by_field(self):
        lines = [
            'NTT EFOSC2 allocation 9-13 July',  # ESO multi-night (drops Tatoo closing day)
            'Magellan-Baade IMACS 17-18 July',  # Magellan, both-inclusive
            'NTT EFOSC2 20-22 July (cancelled)',  # cancelled status
            'Magellan-Clay Lightspeed 25-27 July BoN-0626',  # partial-night window
        ]
        path, tmpdir_ctx = self._write_schedule_file(lines)
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        year = date.today().year
        run_specs = [
            # (telescope, instrument, nights, title, start_window, end_window)
            ('NTT', 'EFOSC2', [date(year, 7, d) for d in (9, 10, 11, 12)], 'NTT EFOSC2', None, None),
            ('Magellan-Baade', 'IMACS', [date(year, 7, 17), date(year, 7, 18)], 'Magellan-Baade IMACS', None, None),
            (
                'NTT',
                'EFOSC2',
                [date(year, 7, 20), date(year, 7, 21)],
                '[CANCELLED] NTT EFOSC2',
                None,
                None,
            ),
            (
                'Magellan-Clay',
                'Lightspeed',
                [date(year, 7, 25), date(year, 7, 26), date(year, 7, 27)],
                'Magellan-Clay Lightspeed',
                'BoN',
                '0626',
            ),
        ]

        expected_by_url: dict[str, dict] = {}
        runs = list(CampaignRun.objects.order_by('pk'))
        self.assertEqual(len(runs), len(run_specs))
        for run, (telescope, instrument, nights, title, start_window, end_window) in zip(runs, run_specs, strict=True):
            site = get_site(telescope)
            for night in nights:
                sunset, sunrise = tr.sun_event(site, night, 'sun')
                start = _expected_boundary(start_window, sunset, sunrise, night, at_full_night='BoN')
                end = _expected_boundary(end_window, sunset, sunrise, night, at_full_night='EoN')
                url = f'ALLOC:{run.pk}:{night.isoformat()}'
                expected_by_url[url] = {
                    'telescope': telescope,
                    'instrument': instrument,
                    'title': title,
                    'start_time': start,
                    'end_time': end,
                }

        actual_urls = set(
            CalendarEvent.objects.filter(url__startswith=ALLOC_URL_NAMESPACE).values_list('url', flat=True)
        )
        self.assertEqual(actual_urls, set(expected_by_url))

        for url, expected in expected_by_url.items():
            with self.subTest(url=url):
                event = CalendarEvent.objects.get(url=url)
                self.assertEqual(event.telescope, expected['telescope'])
                self.assertEqual(event.instrument, expected['instrument'])
                self.assertEqual(event.title, expected['title'])
                self.assertEqual(event.start_time, expected['start_time'])
                self.assertEqual(event.end_time, expected['end_time'])

        # Documented divergence: only the cancelled run's nights carry the extra
        # 'Run status: Cancelled' line -- everything else about the description
        # (status word, source line) is unchanged from pre-cutover output.
        cancelled_run = runs[2]
        cancelled_events = CalendarEvent.objects.filter(url__startswith=f'ALLOC:{cancelled_run.pk}:')
        self.assertGreater(cancelled_events.count(), 0)
        for event in cancelled_events:
            self.assertIn('Status: cancelled', event.description)
            self.assertIn('Run status: Cancelled', event.description)

        for run in (runs[0], runs[1], runs[3]):
            for event in CalendarEvent.objects.filter(url__startswith=f'ALLOC:{run.pk}:'):
                self.assertNotIn('Run status:', event.description)
