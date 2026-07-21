import io
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, override_settings
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

GEM_SETTINGS = {
    'GEM': {
        'programs': {
            'GS-2026A-T-999': {
                'MM': 'Std: GMOS-S MOS',
                'QQ': 'Rap: GMOS-S MOS',
            },
            'GN-2026A-T-999': {
                'MM': 'Std: GNIRS',
            },
        }
    }
}


def _gem_parameters(
    prog: str = 'GS-2026A-T-999',
    obsid: list[str] | None = None,
    ready: str = 'true',
    window_date: str | None = None,
    window_time: str | None = None,
    window_duration: str | None = None,
) -> dict:
    """Build a parameters dict matching the Gemini ObservationRecord.parameters shape.

    Args:
        prog: Gemini program ID (e.g. 'GS-2026A-T-999').
        obsid: List of obs codes; defaults to ['MM'].
        ready: ToO readiness flag; 'true' or 'false'.
        window_date: Explicit window date string 'YYYY-MM-DD', or None to omit window keys.
        window_time: Explicit window time string 'HH:MM', added when window_date is set.
        window_duration: Explicit window duration in hours as string, added when window_date is set.

    Returns:
        dict: parameters dict suitable for ObservationRecord.parameters.
    """
    params: dict = {
        'prog': prog,
        'obsid': obsid if obsid is not None else ['MM'],
        'ready': ready,
        'password': '[redacted]',
    }
    if window_date is not None:
        params['windowDate'] = window_date
        params['windowTime'] = window_time or '00:00'
        params['windowDuration'] = window_duration or '1'
    return params


@override_settings(FACILITIES=GEM_SETTINGS)
class TestSyncGeminiObservationCalendar(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        cls.user = get_user_model().objects.create_user(username='testuser', password='pass')
        cls.target = NonSiderealTargetFactory.create(name='test-target')

    def _make_record(self, observation_id: str, params: dict, **kwargs) -> ObservationRecord:
        return ObservationRecord.objects.create(
            observation_id=observation_id,
            target=self.target,
            user=self.user,
            facility='GEM',
            status='PENDING',
            parameters=params,
            **kwargs,
        )

    def test_gem_select_01(self) -> None:
        """GEM-SELECT-01: a single GEM record produces exactly one CalendarEvent; non-GEM is ignored."""
        self._make_record('9001', _gem_parameters())
        ObservationRecord.objects.create(
            observation_id='9999',
            target=self.target,
            user=self.user,
            facility='LCO',
            status='PENDING',
            parameters={},
        )
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_gem_key_01(self) -> None:
        """GEM-KEY-01: event.url == 'GEM:{prog}/{observation_id}'."""
        self._make_record('9001', _gem_parameters(prog='GS-2026A-T-999'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.url, 'GEM:GS-2026A-T-999/9001')

    def test_gem_prop_01(self) -> None:
        """GEM-PROP-01: event.proposal == prog."""
        self._make_record('9001', _gem_parameters(prog='GS-2026A-T-999'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.proposal, 'GS-2026A-T-999')

    def test_gem_tele_01_gemini_south(self) -> None:
        """GEM-TELE-01: GS-* prog -> event.telescope == 'Gemini South'."""
        self._make_record('9001', _gem_parameters(prog='GS-2026A-T-999'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, 'Gemini South')

    def test_gem_tele_01_gemini_north(self) -> None:
        """GEM-TELE-01: GN-* prog -> event.telescope == 'Gemini North'."""
        self._make_record('9001', _gem_parameters(prog='GN-2026A-T-999'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.telescope, 'Gemini North')

    def test_gem_instr_01(self) -> None:
        """GEM-INSTR-01: obs code 'MM' ('Std: GMOS-S MOS') -> event.instrument == 'GMOS-S MOS'."""
        self._make_record('9001', _gem_parameters(obsid=['MM']))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.instrument, 'GMOS-S MOS')

    def test_gem_instr_01_raw_fallback_when_window_present_and_settings_missing(self) -> None:
        """GEM-INSTR-01 raw fallback: explicit window + 'ZZ' absent from settings -> instrument == 'ZZ'.

        'ZZ' is not in GEM_SETTINGS for 'GS-2026A-T-999', so description_str is None.
        Because an explicit window (windowDate/windowTime/windowDuration) is present, the record
        is NOT skipped; instead instrument is set to the raw obs code 'ZZ' (GEM-INSTR-01 raw fallback).
        """
        params = _gem_parameters(obsid=['ZZ'], window_date='2026-07-15', window_time='02:30', window_duration='4')
        self._make_record('9002', params)
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(CalendarEvent.objects.count(), 1)
        event = CalendarEvent.objects.get()
        self.assertEqual(event.instrument, 'ZZ')

    def test_gem_window_01(self) -> None:
        """GEM-WINDOW-01: windowDate/windowTime/windowDuration -> parsed UTC start and end."""
        params = _gem_parameters(window_date='2026-07-15', window_time='02:30', window_duration='4')
        self._make_record('9001', params)
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        expected_start = datetime(2026, 7, 15, 2, 30, tzinfo=dt_timezone.utc)
        expected_end = expected_start + timedelta(hours=4)
        self.assertEqual(event.start_time, expected_start)
        self.assertEqual(event.end_time, expected_end)

    def test_gem_window_02_rap(self) -> None:
        """GEM-WINDOW-02 (Rap): no explicit window, 'QQ' (Rap:) -> start=created, end=created+24h."""
        record = self._make_record('9001', _gem_parameters(obsid=['QQ']))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.start_time, record.created)
        self.assertEqual(event.end_time, record.created + timedelta(hours=24))

    def test_gem_window_02_std(self) -> None:
        """GEM-WINDOW-02 (Std): no explicit window, 'MM' (Std:) -> start=created+24h, end=created+7d."""
        record = self._make_record('9001', _gem_parameters(obsid=['MM']))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.start_time, record.created + timedelta(hours=24))
        self.assertEqual(event.end_time, record.created + timedelta(days=7))

    def test_gem_window_02_skip(self) -> None:
        """GEM-WINDOW-02 (skip): no window + 'ZZ' absent from settings -> no event, skipped: 1.

        When there is no explicit window and the obs code is absent from settings, the ToO-type
        prefix cannot be determined, so the record is skipped (D-01). Contrast with the raw-fallback
        test above where an explicit window forces event creation despite missing settings.
        """
        self._make_record('9001', _gem_parameters(obsid=['ZZ']))
        stdout_buf = io.StringIO()
        call_command('sync_gemini_observation_calendar', stdout=stdout_buf, stderr=io.StringIO())
        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertIn('skipped: 1', stdout_buf.getvalue())

    def test_gem_status_01_on_hold(self) -> None:
        """GEM-STATUS-01: ready='false' -> event.title starts with '[ON_HOLD] '."""
        self._make_record('9001', _gem_parameters(ready='false'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertTrue(event.title.startswith('[ON_HOLD] '))

    def test_gem_status_01_active(self) -> None:
        """GEM-STATUS-01: ready='true' -> event.title == 'Gemini South GMOS-S MOS ToO'."""
        self._make_record('9001', _gem_parameters(ready='true'))
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        self.assertEqual(event.title, 'Gemini South GMOS-S MOS ToO')

    def test_gem_nochurn_01(self) -> None:
        """GEM-NOCHURN-01: re-run on unchanged record leaves modified unchanged and shows unchanged: 1."""
        self._make_record('9001', _gem_parameters())
        call_command('sync_gemini_observation_calendar', stdout=io.StringIO(), stderr=io.StringIO())
        event = CalendarEvent.objects.get()
        modified_before = event.modified

        stdout2 = io.StringIO()
        call_command('sync_gemini_observation_calendar', stdout=stdout2, stderr=io.StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_before)
        self.assertIn('unchanged: 1', stdout2.getvalue())

    def test_gem_secure_01(self) -> None:
        """GEM-SECURE-01: password key/value absent from stdout, stderr, and all CalendarEvent fields.

        Also verifies that a multi-obsid record (obsid=['MM', 'QQ']) runs without error
        (first entry used for instrument/ToO-type lookup).
        """
        params = _gem_parameters(obsid=['MM', 'QQ'])
        params['password'] = 'my-secret-pw-xyz'
        self._make_record('9001', params)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        call_command('sync_gemini_observation_calendar', stdout=stdout_buf, stderr=stderr_buf)

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        self.assertNotIn('password', stdout_val)
        self.assertNotIn('password', stderr_val)
        self.assertNotIn('my-secret-pw-xyz', stdout_val)
        self.assertNotIn('my-secret-pw-xyz', stderr_val)
        for event in CalendarEvent.objects.all():
            for field in event._meta.fields:
                val = str(getattr(event, field.attname, '') or '')
                self.assertNotIn('my-secret-pw-xyz', val, f'Found password in CalendarEvent.{field.name}')
