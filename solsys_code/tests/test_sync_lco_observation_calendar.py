import io
from datetime import datetime
from datetime import timezone as dt_timezone
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory


def _parameters(
    proposal: str = 'TESTCODE123',
    start: str = '2026-07-01T00:00:00',
    end: str = '2026-07-02T00:00:00',
    instrument_type: str = '2M0-SCICAM-MUSCAT',
    site: str | None = 'coj',
) -> dict:
    """Build a parameters dict matching the real ObservationRecord.parameters shape.

    Args:
        proposal: proposal code.
        start: ISO start time string (queue window).
        end: ISO end time string (queue window).
        instrument_type: LCO instrument type code.
        site: LCO 3-letter site code, or None to omit the key entirely.

    Returns:
        dict: a parameters dict suitable for ObservationRecord.parameters.
    """
    params = {
        'proposal': proposal,
        'start': start,
        'end': end,
        'instrument_type': instrument_type,
    }
    if site is not None:
        params['site'] = site
    return params


class TestSyncLcoObservationCalendar(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        cls.user = get_user_model().objects.create(username='sync-test-user')
        cls.target = NonSiderealTargetFactory.create()

    def _create_record(
        self,
        observation_id: str,
        status: str = 'PENDING',
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        facility: str = 'LCO',
        **parameter_overrides,
    ) -> ObservationRecord:
        """Create an ObservationRecord fixture sharing the class-level target/user."""
        return ObservationRecord.objects.create(
            target=self.target,
            user=self.user,
            facility=facility,
            observation_id=observation_id,
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters=_parameters(**parameter_overrides),
        )

    def test_select_01_only_matching_proposal_creates_events(self):
        """SELECT-01: only the matching-proposal record creates a CalendarEvent."""
        self._create_record('111111', proposal='MATCHCODE')
        self._create_record('222222', proposal='OTHERCODE')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_sync_01_d01_url_uses_requests_path_not_requestgroups(self):
        """SYNC-01/D-01: event.url equals LCOFacility().get_observation_url(observation_id)."""
        self._create_record('333333', proposal='MATCHCODE')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        expected_url = LCOFacility().get_observation_url('333333')
        self.assertEqual(event.url, expected_url)
        self.assertIn('/requests/', event.url)
        self.assertIn('333333', event.url)
        self.assertNotIn('requestgroups', event.url)

    def test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title(self):
        """SYNC-02/D-03: scheduled_start=None -> times from parameters, '[QUEUED] ...' title."""
        self._create_record(
            '444444',
            proposal='MATCHCODE',
            start='2026-07-01T00:00:00',
            end='2026-07-02T00:00:00',
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.start_time, datetime(2026, 7, 1, 0, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(event.end_time, datetime(2026, 7, 2, 0, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(event.title, '[QUEUED] FTS 2M0-SCICAM-MUSCAT')

    def test_sync_03_d03_placed_uses_scheduled_times_and_clean_title(self):
        """SYNC-03/D-03: scheduled_start/end populated -> those times, clean title (no [QUEUED])."""
        sched_start = datetime(2026, 7, 5, 10, 0, 0, tzinfo=dt_timezone.utc)
        sched_end = datetime(2026, 7, 5, 12, 0, 0, tzinfo=dt_timezone.utc)
        self._create_record(
            '555555',
            proposal='MATCHCODE',
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.start_time, sched_start)
        self.assertEqual(event.end_time, sched_end)
        self.assertEqual(event.title, 'FTS 2M0-SCICAM-MUSCAT')
        self.assertNotIn('[QUEUED]', event.title)

    def test_sync_05_telescope_instrument_proposal_populated(self):
        """SYNC-05: telescope/instrument/proposal populated from the record."""
        self._create_record(
            '666666',
            proposal='MATCHCODE',
            site='ogg',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, 'FTN')
        self.assertEqual(event.instrument, '2M0-SCICAM-MUSCAT')
        self.assertEqual(event.proposal, 'MATCHCODE')

    def test_term_01_d04_window_expired_gets_expired_prefix(self):
        """TERM-01/D-04: WINDOW_EXPIRED status gets '[EXPIRED]' title prefix; event retained."""
        self._create_record('700001', proposal='MATCHCODE', status='WINDOW_EXPIRED')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[EXPIRED]'))
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_term_01_d04_canceled_gets_cancelled_prefix(self):
        """TERM-01/D-04: CANCELED status gets '[CANCELLED]' title prefix; event retained."""
        self._create_record('700002', proposal='MATCHCODE', status='CANCELED')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[CANCELLED]'))
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_term_01_d04_failure_limit_reached_gets_failed_prefix(self):
        """TERM-01/D-04: FAILURE_LIMIT_REACHED status gets '[FAILED]' title prefix; event retained."""
        self._create_record('700003', proposal='MATCHCODE', status='FAILURE_LIMIT_REACHED')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[FAILED]'))
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_term_01_d04_not_attempted_gets_failed_prefix(self):
        """TERM-01/D-04: NOT_ATTEMPTED status gets '[FAILED]' title prefix; event retained."""
        self._create_record('700004', proposal='MATCHCODE', status='NOT_ATTEMPTED')
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[FAILED]'))
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_d06_completed_gets_clean_title_no_prefix(self):
        """D-06 (research correction): COMPLETED status gets a clean title, no terminal prefix."""
        self._create_record(
            '700005',
            proposal='MATCHCODE',
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 1, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.title, 'FTS 2M0-SCICAM-MUSCAT')
        for prefix in ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[QUEUED]'):
            self.assertNotIn(prefix, event.title)

    def test_sync_04_rerun_updates_in_place_no_churn_on_unchanged(self):
        """SYNC-04: reschedule updates the existing event; unchanged record's modified is untouched."""
        rescheduled = self._create_record('800001', proposal='MATCHCODE')
        self._create_record('800002', proposal='MATCHCODE')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        self.assertEqual(CalendarEvent.objects.count(), 2)
        modified_before = {e.pk: e.modified for e in CalendarEvent.objects.all()}

        # Reschedule one record (now placed, with new times)
        new_start = datetime(2026, 8, 1, 5, 0, 0, tzinfo=dt_timezone.utc)
        new_end = datetime(2026, 8, 1, 7, 0, 0, tzinfo=dt_timezone.utc)
        rescheduled.scheduled_start = new_start
        rescheduled.scheduled_end = new_end
        rescheduled.save()

        stdout2 = io.StringIO()
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=stdout2,
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 2)

        rescheduled_url = LCOFacility().get_observation_url('800001')
        unchanged_url = LCOFacility().get_observation_url('800002')
        rescheduled_event = CalendarEvent.objects.get(url=rescheduled_url)
        unchanged_event = CalendarEvent.objects.get(url=unchanged_url)

        self.assertEqual(rescheduled_event.start_time, new_start)
        self.assertEqual(rescheduled_event.end_time, new_end)
        self.assertNotEqual(rescheduled_event.modified, modified_before[rescheduled_event.pk])
        self.assertEqual(unchanged_event.modified, modified_before[unchanged_event.pk])
        self.assertIn('updated: 1', stdout2.getvalue())
        self.assertIn('unchanged: 1', stdout2.getvalue())

    def test_sync_05_d05_description_contains_proposal_status_and_window(self):
        """SYNC-05/D-05: description contains proposal code, status, and the active time window."""
        self._create_record(
            '900001',
            proposal='MATCHCODE',
            status='PENDING',
            start='2026-09-01T00:00:00',
            end='2026-09-02T00:00:00',
        )
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        event = CalendarEvent.objects.get()
        desc = event.description
        self.assertIn('MATCHCODE', desc)
        self.assertIn('PENDING', desc)
        self.assertIn('2026-09-01', desc)
        self.assertIn('2026-09-02', desc)

    def test_skip_path_missing_site_logged_and_skipped(self):
        """A matching record with no parameters['site'] is logged to stderr and skipped; others still sync."""
        self._create_record('990001', proposal='MATCHCODE', site=None)
        self._create_record('990002', proposal='MATCHCODE', site='coj')

        stderr_buf = io.StringIO()
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=stderr_buf,
        )
        self.assertEqual(CalendarEvent.objects.count(), 1)
        err = stderr_buf.getvalue()
        self.assertIn('990001', err)

    def test_skip_path_inconsistent_scheduled_times_logged_and_skipped(self):
        """A record with only one of scheduled_start/scheduled_end set is skipped, not crashed on."""
        self._create_record(
            '990003',
            proposal='MATCHCODE2',
            scheduled_start=datetime(2026, 8, 1, tzinfo=dt_timezone.utc),
            scheduled_end=None,
        )
        self._create_record('990004', proposal='MATCHCODE2', site='coj')

        stderr_buf = io.StringIO()
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE2',
            stdout=io.StringIO(),
            stderr=stderr_buf,
        )
        self.assertEqual(CalendarEvent.objects.count(), 1)
        err = stderr_buf.getvalue()
        self.assertIn('990003', err)

    def test_zero_match_reports_created_zero_no_command_error(self):
        """Zero matching records creates zero events and reports 'created: 0' (no CommandError)."""
        self._create_record('111222', proposal='SOMEOTHERCODE')
        stdout_buf = io.StringIO()
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'NOMATCHCODE',
            stdout=stdout_buf,
            stderr=io.StringIO(),
        )
        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertIn('created: 0', stdout_buf.getvalue())

    def test_select_02_comma_list_matches_any_no_substring_leakage(self):
        """SELECT-02: --proposal A,B,C matches exactly A/B/C; no substring match on decoy 'AB'."""
        self._create_record('600001', proposal='A')
        self._create_record('600002', proposal='B')
        self._create_record('600003', proposal='C')
        self._create_record('600004', proposal='AB')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'A,B,C',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 3)
        decoy_url = LCOFacility().get_observation_url('600004')
        self.assertFalse(CalendarEvent.objects.filter(url=decoy_url).exists())
        for observation_id in ('600001', '600002', '600003'):
            url = LCOFacility().get_observation_url(observation_id)
            self.assertTrue(CalendarEvent.objects.filter(url=url).exists())

    def test_select_03_all_token_case_insensitive_syncs_everything(self):
        """SELECT-03/D-02: --proposal all (lowercase) syncs every record, regardless of proposal/facility."""
        self._create_record('610001', proposal='PROPA')
        self._create_record('610002', proposal='PROPB')
        self._create_record('610003', proposal='PROPC', facility='SOAR', site='sor', instrument_type='SOAR_GHTS_REDCAM')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'all',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 3)
        for observation_id in ('610001', '610002', '610003'):
            url = LCOFacility().get_observation_url(observation_id)
            self.assertTrue(CalendarEvent.objects.filter(url=url).exists())

    def test_select_04_single_run_covers_both_facilities(self):
        """SELECT-04: a single run produces CalendarEvents for both an LCO and a SOAR record."""
        self._create_record('620001', proposal='SHARED', facility='LCO')
        soar_record = self._create_record(
            '620002', proposal='SHARED', facility='SOAR', site='sor', instrument_type='SOAR_GHTS_REDCAM'
        )

        # Pitfall 4 guard: confirm the fixture actually persisted facility='SOAR'.
        self.assertEqual(ObservationRecord.objects.get(observation_id='620002').facility, 'SOAR')
        self.assertEqual(soar_record.facility, 'SOAR')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'SHARED',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 2)
        lco_url = LCOFacility().get_observation_url('620001')
        soar_url = LCOFacility().get_observation_url('620002')
        self.assertTrue(CalendarEvent.objects.filter(url=lco_url).exists())
        self.assertTrue(CalendarEvent.objects.filter(url=soar_url).exists())

    def test_select_05_soar_record_uses_soar_facility_instance(self):
        """SELECT-05: a SOAR record is dispatched via SOARFacility, never a reused LCOFacility instance.

        Discriminating spy (Pitfall 3): SOARFacility.get_observation_url and
        LCOFacility.get_observation_url return byte-identical strings, so a black-box
        url-equality check cannot prove which class was actually used. Patch both
        methods as imported in the command module and assert the SOAR spy was called
        while the LCO spy was not.
        """
        self._create_record(
            '630001', proposal='SOARCODE', facility='SOAR', site='sor', instrument_type='SOAR_GHTS_REDCAM'
        )

        real_get_observation_url = LCOFacility.get_observation_url

        with (
            patch.object(
                SOARFacility,
                'get_observation_url',
                autospec=True,
                side_effect=real_get_observation_url,
            ) as soar_spy,
            patch.object(
                LCOFacility,
                'get_observation_url',
                autospec=True,
                side_effect=real_get_observation_url,
            ) as lco_spy,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'SOARCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
            soar_spy.assert_called_once()
            lco_spy.assert_not_called()
