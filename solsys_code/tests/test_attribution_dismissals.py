"""28-CONTEXT.md D-05/D-06/D-08: constraint, cascade and audit-default coverage for both
attribution-dismissal models (CalendarEventDismissal, ObservationRecordDismissal).

Closes 28-VALIDATION.md Wave 0 Requirements gap (the two new dismissal models had no direct
test module before this plan). Follows test_campaign_run_observation.py's fixture and
`transaction.atomic()`-wrapped `assertRaises(IntegrityError)` conventions.
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CalendarEventDismissal, CalendarEventMeta, CampaignRun, ObservationRecordDismissal


class TestCalendarEventDismissal(TestCase):
    """Constraint, cascade and audit-field-default behaviour of the event-side dismissal
    model."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.target = NonSiderealTargetFactory.create()
        cls.dismissing_user = User.objects.create(username='dismissing-staffer-event')

        cls.run_a = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start='2025-07-04',
            window_end='2025-07-04',
        )
        cls.run_b = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start='2025-07-05',
            window_end='2025-07-05',
        )

        cls.event = CalendarEvent.objects.create(
            title='FTN/MuSCAT3 run',
            start_time=datetime(2025, 7, 4, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 7, 5, 6, 0, tzinfo=dt_timezone.utc),
        )

    def test_unique_calendar_event_dismissal_pair_constraint_fires(self):
        """D-05/D-08: a second dismissal row for the same (event, run) pair raises
        IntegrityError."""
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)

    def test_constraint_is_per_pair_not_per_event(self):
        """D-05: dismissing one wrong candidate must leave the same event's OTHER
        candidates -- including a run the matcher surfaces later -- still offered. The same
        event dismissed against a SECOND, different run must not raise."""
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_b)

        self.assertEqual(CalendarEventDismissal.objects.filter(event=self.event).count(), 2)

    def test_deleting_run_cascades_dismissal_but_preserves_event(self):
        dismissal = CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)

        self.run_a.delete()

        self.assertFalse(CalendarEventDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertTrue(CalendarEvent.objects.filter(pk=self.event.pk).exists())

    def test_deleting_event_cascades_dismissal_but_preserves_run(self):
        dismissal = CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)

        self.event.delete()

        self.assertFalse(CalendarEventDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertTrue(CampaignRun.objects.filter(pk=self.run_a.pk).exists())

    def test_plain_orm_create_leaves_dismissed_by_dismissed_at_and_reason_at_defaults(self):
        """The model itself does not require dismissed_by/dismissed_at/reason -- the POST
        action in 28-03 is what enforces a non-empty reason."""
        dismissal = CalendarEventDismissal.objects.create(event=self.event, run=self.run_a)

        self.assertIsNone(dismissal.dismissed_by)
        self.assertIsNone(dismissal.dismissed_at)
        self.assertEqual(dismissal.reason, '')

    def test_deleting_dismissing_user_sets_dismissed_by_null_and_keeps_row(self):
        """D-06/D-07: dismissed_by is SET_NULL -- a departed staff member never erases the
        rejection record itself."""
        dismissal = CalendarEventDismissal.objects.create(
            event=self.event,
            run=self.run_a,
            dismissed_by=self.dismissing_user,
            dismissed_at=datetime(2026, 7, 31, 12, 0, tzinfo=dt_timezone.utc),
            reason='Wrong instrument match.',
        )

        self.dismissing_user.delete()
        dismissal.refresh_from_db()

        self.assertIsNone(dismissal.dismissed_by)
        self.assertTrue(CalendarEventDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertEqual(dismissal.reason, 'Wrong instrument match.')


class TestObservationRecordDismissal(TestCase):
    """Constraint, cascade and audit-field-default behaviour of the observation-side
    dismissal model."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.target = NonSiderealTargetFactory.create()
        # Kept separate from cls.dismissing_user -- ObservationRecord.user is
        # on_delete=DO_NOTHING, so deleting a user still referenced by a live
        # ObservationRecord row would leave a dangling FK and fail SQLite's deferred
        # foreign-key check at transaction commit (the same trap Phase 27-04 hit).
        cls.record_owner = User.objects.create(username='record-owner-dismissal')
        cls.dismissing_user = User.objects.create(username='dismissing-staffer-record')

        cls.run_a = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start='2025-07-04',
            window_end='2025-07-04',
        )
        cls.run_b = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start='2025-07-05',
            window_end='2025-07-05',
        )

        cls.record = ObservationRecord.objects.create(
            target=cls.target,
            user=cls.record_owner,
            facility='LCO',
            observation_id='444444',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )

    def test_unique_observation_record_dismissal_pair_constraint_fires(self):
        """D-05/D-08: a second dismissal row for the same (observation_record, run) pair
        raises IntegrityError."""
        ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)

    def test_constraint_is_per_pair_not_per_record(self):
        """D-05: the same observation record dismissed against a SECOND, different run must
        not raise -- pins the per-pair (not per-orphan) contract."""
        ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)
        ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_b)

        self.assertEqual(ObservationRecordDismissal.objects.filter(observation_record=self.record).count(), 2)

    def test_deleting_run_cascades_dismissal_but_preserves_observation_record(self):
        dismissal = ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)

        self.run_a.delete()

        self.assertFalse(ObservationRecordDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertTrue(ObservationRecord.objects.filter(pk=self.record.pk).exists())

    def test_deleting_observation_record_cascades_dismissal_but_preserves_run(self):
        dismissal = ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)

        self.record.delete()

        self.assertFalse(ObservationRecordDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertTrue(CampaignRun.objects.filter(pk=self.run_a.pk).exists())

    def test_plain_orm_create_leaves_dismissed_by_dismissed_at_and_reason_at_defaults(self):
        dismissal = ObservationRecordDismissal.objects.create(observation_record=self.record, run=self.run_a)

        self.assertIsNone(dismissal.dismissed_by)
        self.assertIsNone(dismissal.dismissed_at)
        self.assertEqual(dismissal.reason, '')

    def test_deleting_dismissing_user_sets_dismissed_by_null_and_keeps_row(self):
        """D-06/D-07: dismissed_by is SET_NULL -- a departed staff member never erases the
        rejection record itself."""
        dismissal = ObservationRecordDismissal.objects.create(
            observation_record=self.record,
            run=self.run_a,
            dismissed_by=self.dismissing_user,
            dismissed_at=datetime(2026, 7, 31, 12, 0, tzinfo=dt_timezone.utc),
            reason='Duplicate suggestion, already confirmed elsewhere.',
        )

        self.dismissing_user.delete()
        dismissal.refresh_from_db()

        self.assertIsNone(dismissal.dismissed_by)
        self.assertTrue(ObservationRecordDismissal.objects.filter(pk=dismissal.pk).exists())
        self.assertEqual(dismissal.reason, 'Duplicate suggestion, already confirmed elsewhere.')


class TestCalendarEventMetaAuditFieldDefaults(TestCase):
    """D-12: confirmed_by/confirmed_at default to None on a freshly created row -- this is
    what makes "NULL means confirmed before audit fields existed" a checked fact rather than
    a comment, mirroring test_campaign_run_observation.py's own audit-field-default test."""

    def test_plain_orm_create_leaves_confirmed_by_and_confirmed_at_blank(self):
        event = CalendarEvent.objects.create(
            title='Freshly resolved event',
            start_time=datetime(2025, 7, 6, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 7, 7, 6, 0, tzinfo=dt_timezone.utc),
        )
        meta = CalendarEventMeta.objects.create(event=event, is_verified=True)

        self.assertIsNone(meta.confirmed_by)
        self.assertIsNone(meta.confirmed_at)
