"""PROJ-01/PROJ-02/TRIG-01/TRIG-02: the tracer's end-to-end proof that a real
``ObservationRecord.save()`` reaches a real ``CalendarEvent``/``CalendarEventMeta`` row, and
that a second save that only sets the schedule fields narrows that same event in place --
the exact path TOM's own ``observation_change_state`` hook misses (spike 001b scenario S1),
and the real ``updatestatus`` path (spike 001b scenario S4).
"""

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import patch

from django.db import transaction
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.facilities.lco import LCOFacility
from tom_observations.models import ObservationGroup, ObservationRecord
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
        """G-34-2: OCSFacility.get_observation_status() returns the portal's raw ISO-8601
        strings (trailing 'Z'), not datetime objects -- this is the real contract a
        BaseObservationFacility.update_observation_status() save exercises, and the one
        the pre-G-34-2 version of this test (datetime-valued fake) never caught."""
        block_start = datetime(2026, 9, 16, 2, 0, tzinfo=dt_timezone.utc)
        block_end = block_start + timedelta(minutes=19)

        original_get_status = LCOFacility.get_observation_status

        def fake_get_observation_status(self, observation_id):
            return {
                'state': 'PENDING',
                'scheduled_start': block_start.isoformat().replace('+00:00', 'Z'),
                'scheduled_end': block_end.isoformat().replace('+00:00', 'Z'),
            }

        LCOFacility.get_observation_status = fake_get_observation_status
        try:
            with self.assertNoLogs('solsys_code.observation_projector', level='WARNING'):
                LCOFacility().update_observation_status(self.record.observation_id)
        finally:
            LCOFacility.get_observation_status = original_get_status

        url = LCOFacility().get_observation_url(self.record.observation_id)
        self.assertEqual(CalendarEvent.objects.filter(url=url).count(), 1)
        event = CalendarEvent.objects.get(url=url)
        self.assertEqual(event.start_time, block_start)
        self.assertEqual(event.end_time, block_end)
        self.assertTrue(event.title.startswith('[S] '))

    def test_updatestatus_with_datetime_valued_facility_still_narrows_the_event(self) -> None:
        """Task 1 rewrote this class's only test to feed portal ISO strings; this keeps the
        datetime-valued case covered too -- both value types a facility might present stay
        exercised."""
        block_start = datetime(2026, 9, 16, 3, 0, tzinfo=dt_timezone.utc)
        block_end = block_start + timedelta(minutes=19)

        original_get_status = LCOFacility.get_observation_status

        def fake_get_observation_status(self, observation_id):
            return {'state': 'PENDING', 'scheduled_start': block_start, 'scheduled_end': block_end}

        LCOFacility.get_observation_status = fake_get_observation_status
        try:
            with self.assertNoLogs('solsys_code.observation_projector', level='WARNING'):
                LCOFacility().update_observation_status(self.record.observation_id)
        finally:
            LCOFacility.get_observation_status = original_get_status

        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        self.assertEqual(event.start_time, block_start)
        self.assertEqual(event.end_time, block_end)
        self.assertTrue(event.title.startswith('[S] '))

    def test_updatestatus_event_span_matches_the_reloaded_record_no_churn(self) -> None:
        """The no-churn guarantee: the window the receiver wrote from the in-memory portal
        strings is the same window the sweep would later compute from the stored row, so
        the receiver and the sweep cannot fight over the same event."""
        block_start = datetime(2026, 9, 16, 4, 0, tzinfo=dt_timezone.utc)
        block_end = block_start + timedelta(minutes=19)

        original_get_status = LCOFacility.get_observation_status

        def fake_get_observation_status(self, observation_id):
            return {
                'state': 'PENDING',
                'scheduled_start': block_start.isoformat().replace('+00:00', 'Z'),
                'scheduled_end': block_end.isoformat().replace('+00:00', 'Z'),
            }

        LCOFacility.get_observation_status = fake_get_observation_status
        try:
            LCOFacility().update_observation_status(self.record.observation_id)
        finally:
            LCOFacility.get_observation_status = original_get_status

        self.record.refresh_from_db()
        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        self.assertEqual(event.start_time, self.record.scheduled_start)
        self.assertEqual(event.end_time, self.record.scheduled_end)


class TestGroupMembershipReceiver(ObservationProjectorSignalsTestCase):
    """D-15/Pattern 3: the m2m_changed receiver closes the add-after-save gap."""

    def test_group_add_after_save_sets_observation_group(self) -> None:
        group = ObservationGroup.objects.create(name='signals-group-add')
        group.observation_records.add(self.record)

        meta = CalendarEventMeta.objects.get(observation_record=self.record)
        self.assertEqual(meta.observation_group_id, group.pk)

    def test_group_remove_clears_observation_group(self) -> None:
        group = ObservationGroup.objects.create(name='signals-group-remove')
        group.observation_records.add(self.record)
        group.observation_records.remove(self.record)

        meta = CalendarEventMeta.objects.get(observation_record=self.record)
        self.assertIsNone(meta.observation_group_id)

    def test_group_clear_reprojects_every_former_member(self) -> None:
        other_record = ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='projector-signals-002',
            status='PENDING',
            parameters={
                'proposal': 'KEY2026B-004',
                'start': self.window_start.isoformat(),
                'end': self.window_end.isoformat(),
                'instrument_type': '2M0-SCICAM-MUSCAT',
            },
        )
        group = ObservationGroup.objects.create(name='signals-group-clear')
        group.observation_records.add(self.record, other_record)
        group.observation_records.clear()

        meta = CalendarEventMeta.objects.get(observation_record=self.record)
        other_meta = CalendarEventMeta.objects.get(observation_record=other_record)
        self.assertIsNone(meta.observation_group_id)
        self.assertIsNone(other_meta.observation_group_id)

    def test_reverse_direction_add_sets_the_link_too(self) -> None:
        group = ObservationGroup.objects.create(name='signals-group-reverse-add')
        self.record.observationgroup_set.add(group)

        meta = CalendarEventMeta.objects.get(observation_record=self.record)
        self.assertEqual(meta.observation_group_id, group.pk)

    def test_group_add_with_write_failure_logs_a_membership_specific_warning(self) -> None:
        """A record whose event write fails during a membership change (here: a
        pre-existing duplicate-url CalendarEvent making get_or_create() raise
        MultipleObjectsReturned inside project_record()) must not raise, must leave the
        membership change itself intact, and must log a warning naming the membership
        change as the trigger -- on top of project_record()'s own generic warning, not
        instead of it."""
        from solsys_code import observation_projector as op

        facility = op.facility_for(self.record)
        url = facility.get_observation_url(self.record.observation_id)
        # self.record already has its own projector-owned event (created by the fixture's
        # own ObservationRecord.objects.create() post_save call); adding a second row at
        # the same url reproduces the duplicate-url condition get_or_create() cannot
        # tolerate.
        CalendarEvent.objects.create(
            url=url,
            title='pre-existing duplicate',
            start_time=self.window_start,
            end_time=self.window_end,
        )

        group = ObservationGroup.objects.create(name='signals-group-write-failure')
        with self.assertLogs('solsys_code.observation_projector', level='WARNING') as logs:
            group.observation_records.add(self.record)  # must not raise

        self.assertIn(self.record, group.observation_records.all())
        joined = '\n'.join(logs.output)
        self.assertIn(f'unprojectable observation_id={self.record.observation_id!r}', joined)
        self.assertIn(
            f'group membership change left observation_id={self.record.observation_id!r} unprojectable', joined
        )

    def test_gemini_record_added_to_group_writes_no_calendar_event(self) -> None:
        gem_record = ObservationRecord.objects.create(
            target=self.target,
            facility='GEM',
            observation_id='signals-gemini-group-member',
            status='PENDING',
            parameters={},
        )
        group = ObservationGroup.objects.create(name='signals-group-gemini')
        group.observation_records.add(gem_record)

        self.assertFalse(CalendarEventMeta.objects.filter(observation_record=gem_record).exists())


class TestRecordDeleteReceiver(ObservationProjectorSignalsTestCase):
    """D-14/Pattern 4: pre_delete removes only the record's own projector-owned event."""

    def test_delete_deletes_owned_event_and_companion_row(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        event_pk = CalendarEvent.objects.get(url=url).pk

        self.record.delete()

        self.assertFalse(CalendarEvent.objects.filter(pk=event_pk).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event_id=event_pk).exists())

    def test_delete_leaves_a_run_prefixed_companion_event_alive(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        own_meta = CalendarEventMeta.objects.get(event__url=url)
        own_meta.observation_record = None
        own_meta.save()

        run_event = CalendarEvent.objects.create(
            url='RUN:42:2026-09-15',
            title='[CANCELLED] reconciler-owned',
            description='',
            start_time=datetime(2026, 9, 15, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 16, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=run_event, observation_record=self.record)

        self.record.delete()

        self.assertTrue(CalendarEvent.objects.filter(pk=run_event.pk).exists())

    def test_delete_with_no_companion_row_raises_nothing(self) -> None:
        record = ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='signals-delete-no-companion',
            status='PENDING',
            parameters={'proposal': 'NOWINDOW'},  # no start/end -> unprojectable, no companion row
        )
        self.assertFalse(CalendarEventMeta.objects.filter(observation_record=record).exists())

        record.delete()  # must not raise

        self.assertFalse(ObservationRecord.objects.filter(observation_id='signals-delete-no-companion').exists())


class TestReceiverSafetyContract(ObservationProjectorSignalsTestCase):
    """TRIG-02: no receiver may ever abort the operation that triggered it."""

    def test_raising_projector_does_not_block_a_save(self) -> None:
        with patch('solsys_code.observation_projector.project_record', side_effect=RuntimeError('boom')):
            self.record.status = 'COMPLETED'
            self.record.save()  # must not raise
        self.assertEqual(ObservationRecord.objects.get(pk=self.record.pk).status, 'COMPLETED')

    def test_raising_projector_does_not_block_a_group_membership_change(self) -> None:
        group = ObservationGroup.objects.create(name='signals-safety-group')
        with patch('solsys_code.observation_projector.project_record', side_effect=RuntimeError('boom')):
            group.observation_records.add(self.record)  # must not raise
        self.assertIn(self.record, group.observation_records.all())

    def test_raising_projector_does_not_block_a_delete(self) -> None:
        with patch('solsys_code.observation_projector.facility_for', side_effect=RuntimeError('boom')):
            self.record.delete()  # must not raise
        self.assertFalse(ObservationRecord.objects.filter(pk=self.record.pk).exists())

    def test_raw_save_writes_no_calendar_event(self) -> None:
        from solsys_code.observation_projector import receiver_on_record_save

        unsaved = ObservationRecord(
            target=self.target,
            facility='LCO',
            observation_id='signals-raw-save',
            status='PENDING',
            parameters={
                'proposal': 'KEY2026B-004',
                'start': self.window_start.isoformat(),
                'end': self.window_end.isoformat(),
                'instrument_type': '2M0-SCICAM-MUSCAT',
            },
        )
        receiver_on_record_save(sender=ObservationRecord, instance=unsaved, created=True, raw=True)

        url = LCOFacility().get_observation_url('signals-raw-save')
        self.assertFalse(CalendarEvent.objects.filter(url=url).exists())

    def test_queryset_update_bypasses_the_receiver_and_writes_no_new_event(self) -> None:
        url = LCOFacility().get_observation_url(self.record.observation_id)
        event = CalendarEvent.objects.get(url=url)
        title_before = event.title

        ObservationRecord.objects.filter(pk=self.record.pk).update(status='COMPLETED')

        event.refresh_from_db()
        self.assertEqual(event.title, title_before)

    def test_record_saved_inside_a_rolled_back_transaction_leaves_no_event(self) -> None:
        url = LCOFacility().get_observation_url('signals-rollback-record')
        with transaction.atomic():
            ObservationRecord.objects.create(
                target=self.target,
                facility='LCO',
                observation_id='signals-rollback-record',
                status='PENDING',
                parameters={
                    'proposal': 'KEY2026B-004',
                    'start': self.window_start.isoformat(),
                    'end': self.window_end.isoformat(),
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                },
            )
            self.assertTrue(CalendarEvent.objects.filter(url=url).exists())
            transaction.set_rollback(True)

        self.assertFalse(ObservationRecord.objects.filter(observation_id='signals-rollback-record').exists())
        self.assertFalse(CalendarEvent.objects.filter(url=url).exists())

    def test_make_request_is_never_called_during_a_record_save(self) -> None:
        with patch('solsys_code.calendar_utils.make_request') as mock_make_request:
            self.record.status = 'COMPLETED'
            self.record.save()
        mock_make_request.assert_not_called()
