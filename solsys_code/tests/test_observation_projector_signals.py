"""PROJ-01/PROJ-02/TRIG-01/TRIG-02: the tracer's end-to-end proof that a real
``ObservationRecord.save()`` reaches a real ``CalendarEvent``/``CalendarEventMeta`` row, and
that a second save that only sets the schedule fields narrows that same event in place --
the exact path TOM's own ``observation_change_state`` hook misses (spike 001b scenario S1),
and the real ``updatestatus`` path (spike 001b scenario S4).
"""

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CalendarEventMeta


class ObservationProjectorSignalsTestCase(TestCase):
    """Shared fixture: a queued LCO ObservationRecord for the KEY2026B-004 proposal."""

    def setUp(self) -> None:
        self.target = NonSiderealTargetFactory.create()
        self.window_start = datetime(2026, 9, 15, 22, 0, tzinfo=dt_timezone.utc)
        self.window_end = datetime(2026, 9, 16, 6, 0, tzinfo=dt_timezone.utc)
        self.record = ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='projector-signals-001',
            status='PENDING',
            scheduled_start=None,
            scheduled_end=None,
            parameters={
                'proposal': 'KEY2026B-004',
                'start': self.window_start.isoformat(),
                'end': self.window_end.isoformat(),
                'instrument_type': '2M0-SCICAM-MUSCAT',
            },
        )


class TestPostSaveReceiver(ObservationProjectorSignalsTestCase):
    """S1 (spike 001b): a schedule-only save narrows the record's own event in place."""

    def test_create_projects_a_queued_event_over_the_request_window(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        events = CalendarEvent.objects.filter(url=url)
        self.assertEqual(events.count(), 1)
        event = events.get()
        self.assertEqual(event.start_time, self.window_start)
        self.assertEqual(event.end_time, self.window_end)
        self.assertTrue(event.title.startswith('[Q] '))

    def test_schedule_only_save_narrows_the_same_event_row(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        original_pk = event.pk

        block_start = datetime(2026, 9, 16, 1, 0, tzinfo=dt_timezone.utc)
        block_end = block_start + timedelta(minutes=19)
        self.record.scheduled_start = block_start
        self.record.scheduled_end = block_end
        self.record.save()

        event.refresh_from_db()
        self.assertEqual(event.pk, original_pk)
        self.assertEqual(CalendarEvent.objects.filter(url=url).count(), 1)
        self.assertEqual(event.start_time, block_start)
        self.assertEqual(event.end_time, block_end)
        self.assertTrue(event.title.startswith('[S] '))

    def test_companion_row_links_to_record_and_is_verified(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.observation_record_id, self.record.pk)
        self.assertTrue(meta.is_verified)


class TestUpdateObservationStatusPath(ObservationProjectorSignalsTestCase):
    """S4 (spike 001b): the real ``updatestatus`` path -- no command run, receiver only."""

    def test_updatestatus_narrows_the_event_with_no_command_run(self) -> None:
        block_start = datetime(2026, 9, 16, 2, 0, tzinfo=dt_timezone.utc)
        block_end = block_start + timedelta(minutes=19)

        original_get_status = LCOFacility.get_observation_status

        def fake_get_observation_status(self, observation_id):
            return {'state': 'PENDING', 'scheduled_start': block_start, 'scheduled_end': block_end}

        LCOFacility.get_observation_status = fake_get_observation_status
        try:
            LCOFacility().update_observation_status(self.record.observation_id)
        finally:
            LCOFacility.get_observation_status = original_get_status

        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        self.assertEqual(event.start_time, block_start)
        self.assertEqual(event.end_time, block_end)
        self.assertTrue(event.title.startswith('[S] '))
