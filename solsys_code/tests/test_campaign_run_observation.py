"""CANON-04: constraint, cascade and audit-field-default coverage for CampaignRunObservation.

Closes 27-VALIDATION.md Wave 0 Requirements gap 1 (the observation-link model had no direct
test module before this plan).
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

from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation


class TestCampaignRunObservation(TestCase):
    """Constraint, cascade and audit-field-default behaviour of the observation link model."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.target = NonSiderealTargetFactory.create()
        # Kept separate from cls.user (the confirming staffer, deleted by one test below) --
        # ObservationRecord.user is on_delete=DO_NOTHING, so deleting a user still
        # referenced by a live ObservationRecord row would leave a dangling FK and fail
        # SQLite's deferred foreign-key check at transaction commit.
        cls.record_owner = User.objects.create(username='record-owner')
        cls.user = User.objects.create(username='confirming-staffer')

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

        cls.record_1 = ObservationRecord.objects.create(
            target=cls.target,
            user=cls.record_owner,
            facility='LCO',
            observation_id='111111',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )
        cls.record_2 = ObservationRecord.objects.create(
            target=cls.target,
            user=cls.record_owner,
            facility='LCO',
            observation_id='222222',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )

    def test_unique_campaign_run_observation_record_constraint_fires(self):
        """D-02: at most one run per observation record -- a second link row for the same
        observation_record raises IntegrityError, even pointing at a DIFFERENT run.
        """
        CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_1)

        # Pitfall 4 precedent (campaign_views.py:245-266): wrap the failing create in
        # transaction.atomic() so the surrounding test transaction is not poisoned by the
        # uncaught IntegrityError.
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CampaignRunObservation.objects.create(run=self.run_b, observation_record=self.record_1)

    def test_one_run_can_link_multiple_distinct_observation_records(self):
        """The constraint is one-run-per-record, not one-record-per-run (D-02's 'one run
        per observation record' direction) -- two link rows for the SAME run but different
        ObservationRecords must coexist with no IntegrityError.
        """
        CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_1)
        CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_2)

        self.assertEqual(CampaignRunObservation.objects.filter(run=self.run_a).count(), 2)

    def test_deleting_run_cascades_link_but_preserves_observation_record(self):
        """D-04: CASCADE on the run FK deletes the link row; the ObservationRecord itself
        is on the other side of the relation and must be untouched.
        """
        link = CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_1)

        self.run_a.delete()

        self.assertFalse(CampaignRunObservation.objects.filter(pk=link.pk).exists())
        self.assertTrue(ObservationRecord.objects.filter(pk=self.record_1.pk).exists())

    def test_deleting_observation_record_cascades_link_but_preserves_run(self):
        """CASCADE on the observation_record FK deletes the link row; the CampaignRun is
        untouched.
        """
        link = CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_1)

        self.record_1.delete()

        self.assertFalse(CampaignRunObservation.objects.filter(pk=link.pk).exists())
        self.assertTrue(CampaignRun.objects.filter(pk=self.run_a.pk).exists())

    def test_deleting_confirming_user_sets_confirmed_by_null_and_keeps_link_row(self):
        """D-03: confirmed_by is SET_NULL, not CASCADE -- deleting the confirming user must
        not delete the confirmed attribution itself.
        """
        link = CampaignRunObservation.objects.create(
            run=self.run_a,
            observation_record=self.record_1,
            confirmed_by=self.user,
            confirmed_at=datetime(2026, 7, 30, 12, 0, tzinfo=dt_timezone.utc),
        )

        self.user.delete()
        link.refresh_from_db()

        self.assertIsNone(link.confirmed_by)
        self.assertTrue(CampaignRunObservation.objects.filter(pk=link.pk).exists())

    def test_plain_orm_create_leaves_confirmed_by_and_confirmed_at_blank(self):
        """D-01/D-03: the row's existence carries 'confirmed'; only the admin's
        save_formset (Plan 05) stamps confirmed_by/confirmed_at -- nothing else may.
        """
        link = CampaignRunObservation.objects.create(run=self.run_a, observation_record=self.record_1)

        self.assertIsNone(link.confirmed_by)
        self.assertIsNone(link.confirmed_at)

    def test_deleting_run_preserves_calendar_event_and_null_s_out_the_companion_run_link(self):
        """ROADMAP criterion 4's other half: deleting a run must never delete its
        CalendarEvent rows or its CalendarEventMeta rows -- CalendarEventMeta.run is
        SET_NULL (26-DECISION.md), not CASCADE.
        """
        event = CalendarEvent.objects.create(
            title='FTN/MuSCAT3 run',
            start_time=datetime(2025, 7, 4, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 7, 5, 6, 0, tzinfo=dt_timezone.utc),
        )
        meta = CalendarEventMeta.objects.create(event=event, is_verified=True, run=self.run_a)

        self.run_a.delete()
        meta.refresh_from_db()

        self.assertTrue(CalendarEvent.objects.filter(pk=event.pk).exists())
        self.assertTrue(CalendarEventMeta.objects.filter(pk=meta.pk).exists())
        self.assertIsNone(meta.run_id)
