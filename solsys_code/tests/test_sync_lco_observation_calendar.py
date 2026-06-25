import io
import re
from datetime import datetime
from datetime import timezone as dt_timezone
from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.management.commands.sync_lco_observation_calendar import (
    SITE_TELESCOPE_MAP,
    _aperture_class_from_telescope_code,
    _derive_telescope,
    _resolve_placement_block,
)
from solsys_code.models import CalendarEventTelescopeLabel


def _parameters(
    proposal: str = 'TESTCODE123',
    start: str = '2026-07-01T00:00:00',
    end: str = '2026-07-02T00:00:00',
    instrument_type: str = '2M0-SCICAM-MUSCAT',
    site: str | None = 'coj',
    extra_params: dict | None = None,
) -> dict:
    """Build a parameters dict matching the real ObservationRecord.parameters shape.

    Args:
        proposal: proposal code.
        start: ISO start time string (queue window).
        end: ISO end time string (queue window).
        instrument_type: LCO instrument type code.
        site: LCO 3-letter site code, or None to omit the key entirely.
        extra_params: additional/override keys merged in last (e.g. c_N_configuration_type,
            c_N_instrument_type, MUSCAT per-channel exposure keys) — used by EXTRACT-02/D-06
            tests to build the real multi-configuration parameter shape without disturbing
            the five named params above.

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
    params.update(extra_params or {})
    return params


def _observations_block_response(
    site: str = 'lsc',
    enclosure: str = 'doma',
    telescope: str = '1m0a',
    state: str = 'COMPLETED',
) -> MagicMock:
    """Build a mock make_request() response for /api/requests/{id}/observations/.

    Args:
        site: 3-letter site code for the single returned block.
        enclosure: 4-char enclosure code for the single returned block.
        telescope: 4-char telescope code for the single returned block.
        state: the block's 'state' value (e.g. 'COMPLETED', 'PENDING').

    Returns:
        MagicMock: a response double whose .json() returns a one-element list
            containing the block dict built from the given keyword args.
    """
    response = MagicMock()
    response.json.return_value = [{'site': site, 'enclosure': enclosure, 'telescope': telescope, 'state': state}]
    return response


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
        """SYNC-02/D-03: scheduled_start=None -> times from parameters, '[QUEUED] ...' title.

        D-01: a banner-stage record makes no API call and gets the coarse fallback
        label ('2m0', derived from the instrument_type), not a SITECODE-CLASS label.
        """
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
        self.assertEqual(event.title, '[QUEUED] 2m0 2M0-SCICAM-MUSCAT')

    def test_sync_03_d03_placed_uses_scheduled_times_and_clean_title(self):
        """SYNC-03/D-03: scheduled_start/end populated -> those times, clean title (no [QUEUED]).

        TELESCOPE-02: a placed record's label comes from a successful live API
        resolution (mocked here), not the flat parameters['site'] key.
        """
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
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ):
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
        self.assertEqual(event.title, 'COJ-2m0 2M0-SCICAM-MUSCAT')
        self.assertNotIn('[QUEUED]', event.title)
        self.assertNotIn('[UNVERIFIED]', event.title)

    def test_display_01_verified_record_creates_sidecar_row_is_verified_true(self):
        """DISPLAY-01: a successfully API-verified record gets a sidecar row with
        is_verified=True."""
        sched_start = datetime(2026, 7, 5, 10, 0, 0, tzinfo=dt_timezone.utc)
        sched_end = datetime(2026, 7, 5, 12, 0, 0, tzinfo=dt_timezone.utc)
        self._create_record(
            '555556',
            proposal='MATCHCODE',
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'MATCHCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        event = CalendarEvent.objects.get()
        self.assertTrue(CalendarEventTelescopeLabel.objects.get(event=event).is_verified)

    def test_display_01_fallback_record_creates_sidecar_row_is_verified_false(self):
        """DISPLAY-01: a placed record whose API call times out (fallback label) gets a
        sidecar row with is_verified=False."""
        self._create_record(
            '800301',
            proposal='SIDECARFALLBACK',
            scheduled_start=datetime(2026, 7, 18, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 18, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='1M0-SCICAM-SINISTRO',
        )
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'SIDECARFALLBACK',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        event = CalendarEvent.objects.get()
        self.assertFalse(CalendarEventTelescopeLabel.objects.get(event=event).is_verified)

    def test_sync_05_telescope_instrument_proposal_populated(self):
        """SYNC-05: telescope/instrument/proposal populated from the record.

        TELESCOPE-02: a placed record's label comes from a successful live API
        resolution (mocked here).
        """
        self._create_record(
            '666666',
            proposal='MATCHCODE',
            scheduled_start=datetime(2026, 7, 6, 10, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 6, 12, 0, 0, tzinfo=dt_timezone.utc),
            site='ogg',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='ogg', telescope='2m0a', state='COMPLETED'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'MATCHCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, 'OGG-2m0')
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
        """D-06 (research correction): COMPLETED status gets a clean title, no terminal prefix.

        TELESCOPE-02: a placed record's label comes from a successful live API
        resolution (mocked here).
        """
        self._create_record(
            '700005',
            proposal='MATCHCODE',
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 1, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'MATCHCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        event = CalendarEvent.objects.get()
        self.assertEqual(event.title, 'COJ-2m0 2M0-SCICAM-MUSCAT')
        for prefix in ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[QUEUED]', '[UNVERIFIED]'):
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

    def test_display_01_rerun_on_unchanged_record_no_duplicate_sidecar_row(self):
        """DISPLAY-01: re-running sync twice on an unchanged record keeps the sidecar
        row count at 1 (no duplicate) and the row's pk (the event's own pk) unchanged."""
        sched_start = datetime(2026, 7, 19, 0, 0, 0, tzinfo=dt_timezone.utc)
        sched_end = datetime(2026, 7, 19, 2, 0, 0, tzinfo=dt_timezone.utc)
        self._create_record(
            '800401',
            proposal='SIDECARNOCHURN',
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'SIDECARNOCHURN',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
            self.assertEqual(CalendarEventTelescopeLabel.objects.count(), 1)
            event_pk_before = CalendarEvent.objects.get().pk

            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'SIDECARNOCHURN',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

        self.assertEqual(CalendarEventTelescopeLabel.objects.count(), 1)
        event = CalendarEvent.objects.get()
        self.assertEqual(event.pk, event_pk_before)
        self.assertTrue(CalendarEventTelescopeLabel.objects.get(event=event).is_verified)

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

    def test_banner_record_missing_site_still_syncs_with_coarse_label(self):
        """D-01: a banner-stage record with no parameters['site'] still syncs (no API call,
        no skip) -- the flat 'site' key is no longer read at all in _build_event_fields."""
        self._create_record('990001', proposal='MATCHCODE', site=None)
        self._create_record('990002', proposal='MATCHCODE', site='coj')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MATCHCODE',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        self.assertEqual(CalendarEvent.objects.count(), 2)
        no_site_url = LCOFacility().get_observation_url('990001')
        event = CalendarEvent.objects.get(url=no_site_url)
        self.assertTrue(event.title.startswith('[QUEUED]'))
        self.assertNotIn('[UNVERIFIED]', event.title)

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

    def test_extract_02_soar_multi_config_picks_spectrum_not_calibration(self):
        """EXTRACT-02: a SOAR SPECTRUM+ARC+LAMP_FLAT record extracts the SPECTRUM config's
        instrument_type, never the ARC/LAMP_FLAT calibration configs."""
        self._create_record(
            '710001',
            proposal='SOAREXTRACT',
            facility='SOAR',
            site='sor',
            instrument_type='NOT-THE-SOURCE',
            extra_params={
                'c_1_configuration_type': 'SPECTRUM',
                'c_1_instrument_type': 'SOAR_GHTS_REDCAM',
                'c_2_configuration_type': 'ARC',
                'c_2_instrument_type': 'SOAR_GHTS_REDCAM_ARC',
                'c_3_configuration_type': 'LAMP_FLAT',
                'c_3_instrument_type': 'SOAR_GHTS_REDCAM_LAMPFLAT',
            },
        )
        # Baseline good record, proving coexistence.
        self._create_record('710002', proposal='SOAREXTRACT', site='ogg')

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'SOAREXTRACT',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 2)
        soar_url = LCOFacility().get_observation_url('710001')
        event = CalendarEvent.objects.get(url=soar_url)
        self.assertEqual(event.instrument, 'SOAR_GHTS_REDCAM')
        self.assertNotEqual(event.instrument, 'SOAR_GHTS_REDCAM_ARC')
        self.assertNotEqual(event.instrument, 'SOAR_GHTS_REDCAM_LAMPFLAT')

    def test_extract_02_muscat_per_channel_exposure_extracts_instrument(self):
        """EXTRACT-02/D-04: an LCO MUSCAT record with only per-channel exposure keys
        (no flat c_N_exposure_time) extracts its instrument_type without raising/empty."""
        self._create_record(
            '710003',
            proposal='MUSCATEXTRACT',
            site='coj',
            instrument_type='NOT-THE-SOURCE',
            extra_params={
                'c_1_configuration_type': 'EXPOSE',
                'c_1_instrument_type': '2M0-SCICAM-MUSCAT',
                'c_1_ic_1_exposure_time_g': 30.0,
                'c_1_ic_1_exposure_time_r': 30.0,
                'c_1_ic_1_exposure_time_i': 30.0,
                'c_1_ic_1_exposure_time_z': 30.0,
            },
        )
        # D-04 leniency: fewer than 4 channels populated still extracts correctly.
        self._create_record(
            '710004',
            proposal='MUSCATEXTRACT',
            site='ogg',
            instrument_type='NOT-THE-SOURCE',
            extra_params={
                'c_1_configuration_type': 'EXPOSE',
                'c_1_instrument_type': '2M0-SCICAM-MUSCAT',
                'c_1_ic_1_exposure_time_g': 30.0,
            },
        )

        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MUSCATEXTRACT',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(CalendarEvent.objects.count(), 2)
        for observation_id in ('710003', '710004'):
            url = LCOFacility().get_observation_url(observation_id)
            event = CalendarEvent.objects.get(url=url)
            self.assertEqual(event.instrument, '2M0-SCICAM-MUSCAT')

    def test_d06_no_extractable_config_logged_and_counted_separately(self):
        """D-06: a fully-malformed record (no recognized configuration_type, no exposure
        signal anywhere) is skipped, logged with its observation_id, and counted in a
        dedicated counter distinct from 'skipped', visible in the run summary."""
        self._create_record(
            '710005',
            proposal='MALFORMEDEXTRACT',
            site='coj',
            instrument_type='NOT-THE-SOURCE',
            extra_params={
                'c_1_configuration_type': 'ARC',
                'c_1_instrument_type': 'SOMETHING',
                'instrument_type': None,
            },
        )
        # Baseline good record, proving coexistence.
        self._create_record('710006', proposal='MALFORMEDEXTRACT', site='ogg')

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        call_command(
            'sync_lco_observation_calendar',
            '--proposal',
            'MALFORMEDEXTRACT',
            stdout=stdout_buf,
            stderr=stderr_buf,
        )

        self.assertEqual(CalendarEvent.objects.count(), 1)
        self.assertIn('710005', stderr_buf.getvalue())
        self.assertIn('extraction_failed: 1', stdout_buf.getvalue())

    def test_telescope_01_verified_dict_covers_all_sites(self):
        """TELESCOPE-01: verified dict covers all 7 real sites with SITECODE-CLASS labels."""
        expected_sites = {'ogg', 'elp', 'lsc', 'cpt', 'coj', 'tfn', 'sor'}
        actual_sites = {site for site, _aperture_class in SITE_TELESCOPE_MAP}
        self.assertEqual(actual_sites, expected_sites)

        label_pattern = re.compile(r'^[A-Z]{3}-(0m4|1m0|2m0|4m0)$')
        for label in SITE_TELESCOPE_MAP.values():
            self.assertRegex(label, label_pattern)

        for migrated_label in ('COJ-2m0', 'OGG-2m0', 'SOR-4m0'):
            self.assertIn(migrated_label, SITE_TELESCOPE_MAP.values())

    def test_telescope_01_aperture_class_from_telescope_code(self):
        """TELESCOPE-01: _aperture_class_from_telescope_code parses/rejects telescope codes."""
        self.assertEqual(_aperture_class_from_telescope_code('1m0a'), '1m0')
        self.assertEqual(_aperture_class_from_telescope_code('0m4b'), '0m4')
        self.assertEqual(_aperture_class_from_telescope_code('2m0a'), '2m0')
        self.assertIsNone(_aperture_class_from_telescope_code('xx'))
        self.assertIsNone(_aperture_class_from_telescope_code('foo9'))

    def test_telescope_01_coj_ogg_full_aperture_class_coverage(self):
        """TELESCOPE-01: coj/ogg's full aperture-class inventory resolves to verified labels.

        Regression for the Phase 7 UAT Test 1 gap (07-UAT.md Gaps section): a real placed
        record (observation_id=4213127) resolved via the live LCO API to
        site='coj', telescope='1m0a' (aperture class '1m0'), but SITE_TELESCOPE_MAP had no
        ('coj', '1m0') entry, so it fell back to the [UNVERIFIED] label instead of COJ-1m0.
        """
        self.assertEqual(_derive_telescope('coj', '1m0a'), 'COJ-1m0')
        self.assertEqual(_derive_telescope('coj', '0m4a'), 'COJ-0m4')
        self.assertEqual(_derive_telescope('ogg', '0m4b'), 'OGG-0m4')

    def test_telescope_02_placed_record_resolves_via_api(self):
        """TELESCOPE-02: a successful mocked API response resolves to the verified label."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(
                site='lsc', enclosure='doma', telescope='1m0a', state='COMPLETED'
            ),
        ):
            block = _resolve_placement_block('12345', mock_facility)

        self.assertIsNotNone(block)
        self.assertEqual(block['site'], 'lsc')
        self.assertEqual(block['enclosure'], 'doma')
        self.assertEqual(block['telescope'], '1m0a')
        self.assertEqual(_derive_telescope(block['site'], block['telescope']), 'LSC-1m0')

    def test_sync_08_single_attempt_no_retry(self):
        """SYNC-08: a timeout results in exactly one make_request call, no retry loop."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ) as mock_make_request:
            block = _resolve_placement_block('12345', mock_facility)

        self.assertIsNone(block)
        mock_make_request.assert_called_once()

    def test_sync_09_no_credential_or_body_leak_in_logs(self):
        """SYNC-09: ImproperCredentialsException/forms.ValidationError are swallowed to None,
        never raised, and the helper never surfaces anything derived from the caught
        exception (which may embed response.content / API-key-adjacent diagnostic text)."""
        mock_facility = MagicMock()
        mock_facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
        mock_facility._portal_headers.return_value = {}

        leak_marker = 'SECRET_API_KEY_LEAK_BODY'

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {leak_marker}'),
        ):
            block = _resolve_placement_block('12345', mock_facility)
        self.assertIsNone(block)

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=forms.ValidationError(f'OCS: {leak_marker}'),
        ):
            block = _resolve_placement_block('12345', mock_facility)
        self.assertIsNone(block)

    def test_telescope_03_api_failure_falls_back_not_skipped(self):
        """TELESCOPE-03: a placed record whose API call times out still gets a CalendarEvent
        (not skipped), telescope = coarse fallback label, skipped count stays 0."""
        self._create_record(
            '800101',
            proposal='FALLBACKCODE',
            scheduled_start=datetime(2026, 7, 15, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 15, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        stdout_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'FALLBACKCODE',
                stdout=stdout_buf,
                stderr=io.StringIO(),
            )

        self.assertEqual(CalendarEvent.objects.count(), 1)
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, '1m0')
        self.assertIn('skipped: 0', stdout_buf.getvalue())

    def test_telescope_03_soar_api_failure_fallback_returns_4m0_label(self):
        """TELESCOPE-03/04/SYNC-06: a placed SOAR record whose API call times out gets the
        coarse '4m0' fallback label (not the raw 'SOAR_GHTS_REDCAM' instrument string), a
        clean '[UNVERIFIED] 4m0 ...' title (not the doubled raw-instrument title), and is
        still counted as a degrade (skipped: 0) -- closes the v1.3 milestone-audit
        zero-coverage gap for SOAR+placed+API-failure."""
        self._create_record(
            '800201',
            proposal='SOARFALLBACKCODE',
            scheduled_start=datetime(2026, 7, 17, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 17, 2, 0, 0, tzinfo=dt_timezone.utc),
            facility='SOAR',
            site='sor',
            instrument_type='SOAR_GHTS_REDCAM',
        )

        stdout_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'SOARFALLBACKCODE',
                stdout=stdout_buf,
                stderr=io.StringIO(),
            )

        self.assertEqual(CalendarEvent.objects.count(), 1)
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, '4m0')
        self.assertTrue(event.title.startswith('[UNVERIFIED] 4m0 '))
        self.assertNotIn('SOAR_GHTS_REDCAM SOAR_GHTS_REDCAM', event.title)
        self.assertIn('skipped: 0', stdout_buf.getvalue())

    def test_telescope_04_fallback_label_visibly_distinguishable(self):
        """TELESCOPE-04: a fallback event has an [UNVERIFIED] title, coarse telescope token,
        and a description failure note; a subsequent successful run flips the label
        visibly (event updates, [UNVERIFIED] drops, telescope becomes the verified label)."""
        sched_start = datetime(2026, 7, 16, 0, 0, 0, tzinfo=dt_timezone.utc)
        sched_end = datetime(2026, 7, 16, 2, 0, 0, tzinfo=dt_timezone.utc)
        self._create_record(
            '800102',
            proposal='FLIPCODE',
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            site='lsc',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'FLIPCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[UNVERIFIED]'))
        self.assertEqual(event.telescope, '1m0')
        self.assertIn('unverified', event.description.lower())
        modified_before = event.modified

        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=_observations_block_response(site='lsc', telescope='1m0a', state='COMPLETED'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'FLIPCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

        event.refresh_from_db()
        self.assertNotIn('[UNVERIFIED]', event.title)
        self.assertEqual(event.telescope, 'LSC-1m0')
        self.assertNotEqual(event.modified, modified_before)

    def test_sync_06_fallback_counter_distinct_from_skipped(self):
        """SYNC-06: telescope_api_failed increments for a placed+failed record while
        skipped stays 0, and the summary line shows 'telescope_api_failed: 1'."""
        self._create_record(
            '800103',
            proposal='COUNTERCODE',
            scheduled_start=datetime(2026, 7, 17, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 17, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        stdout_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=requests.exceptions.Timeout,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'COUNTERCODE',
                stdout=stdout_buf,
                stderr=io.StringIO(),
            )

        summary = stdout_buf.getvalue()
        self.assertIn('telescope_api_failed: 1', summary)
        self.assertIn('skipped: 0', summary)

    def test_sync_07_api_failure_does_not_abort_run(self):
        """SYNC-07: the first-processed of two placed records raises on the API call;
        both still produce CalendarEvents and no exception propagates out of
        call_command. ObservationRecord's default ordering is '-created', so the
        record created SECOND is processed FIRST -- the side_effect list below is
        ordered to match that actual processing order, not creation order."""
        self._create_record(
            '800104',
            proposal='NOABORTCODE',
            scheduled_start=datetime(2026, 7, 18, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 18, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='2M0-SCICAM-MUSCAT',
        )
        self._create_record(
            '800105',
            proposal='NOABORTCODE',
            scheduled_start=datetime(2026, 7, 19, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 19, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='lsc',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        # '-created' default ordering -> 800105 (created last) is processed first.
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=[
                requests.exceptions.Timeout,
                _observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
            ],
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'NOABORTCODE',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

        self.assertEqual(CalendarEvent.objects.count(), 2)
        first_processed_event = CalendarEvent.objects.get(url=LCOFacility().get_observation_url('800105'))
        second_processed_event = CalendarEvent.objects.get(url=LCOFacility().get_observation_url('800104'))
        self.assertTrue(first_processed_event.title.startswith('[UNVERIFIED]'))
        self.assertEqual(second_processed_event.telescope, 'COJ-2m0')

    def test_sync_09_log_line_is_fixed_generic_message(self):
        """SYNC-09: a placed record whose make_request raises an exception embedding a
        leak marker logs the fixed generic message + observation_id, never the leak
        marker token."""
        leak_marker = 'LEAK_MARKER_apikey_body'
        self._create_record(
            '800106',
            proposal='LEAKCODE',
            scheduled_start=datetime(2026, 7, 20, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 20, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='coj',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        stderr_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {leak_marker}'),
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'LEAKCODE',
                stdout=io.StringIO(),
                stderr=stderr_buf,
            )

        err = stderr_buf.getvalue()
        self.assertNotIn(leak_marker, err)
        self.assertIn('800106', err)
        self.assertIn('fallback', err.lower())

    def test_d01_banner_record_no_api_call_no_unverified_prefix(self):
        """D-01: a banner-stage record makes no API call, keeps '[QUEUED]', never gets
        '[UNVERIFIED]', telescope_api_failed stays 0."""
        self._create_record(
            '800107',
            proposal='BANNERCODE',
            site='coj',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        stdout_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
        ) as mock_make_request:
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'BANNERCODE',
                stdout=stdout_buf,
                stderr=io.StringIO(),
            )

        mock_make_request.assert_not_called()
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[QUEUED]'))
        self.assertNotIn('[UNVERIFIED]', event.title)
        self.assertEqual(event.telescope, '1m0')
        self.assertIn('telescope_api_failed: 0', stdout_buf.getvalue())

    def test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped(self):
        """T-07-03: a COMPLETED block returned by the API but missing the 'site' key
        (a malformed/tampered response shape -- only 'state' is validated upstream)
        still produces a coarse-fallback CalendarEvent, not a skipped record. The
        existing _observations_block_response() helper always populates all four
        keys together, so the malformed block is built inline here instead."""
        self._create_record(
            '800108',
            proposal='MISSINGKEYCODE',
            scheduled_start=datetime(2026, 7, 21, 0, 0, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 21, 2, 0, 0, tzinfo=dt_timezone.utc),
            site='lsc',
            instrument_type='1M0-SCICAM-SINISTRO',
        )

        malformed_response = MagicMock()
        malformed_response.json.return_value = [{'state': 'COMPLETED', 'telescope': '1m0a'}]

        stdout_buf = io.StringIO()
        with patch(
            'solsys_code.management.commands.sync_lco_observation_calendar.make_request',
            return_value=malformed_response,
        ):
            call_command(
                'sync_lco_observation_calendar',
                '--proposal',
                'MISSINGKEYCODE',
                stdout=stdout_buf,
                stderr=io.StringIO(),
            )

        self.assertEqual(CalendarEvent.objects.count(), 1)
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, '1m0')
        self.assertTrue(event.title.startswith('[UNVERIFIED]'))
        summary = stdout_buf.getvalue()
        self.assertIn('telescope_api_failed: 1', summary)
        self.assertIn('skipped: 0', summary)
