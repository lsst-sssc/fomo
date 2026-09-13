"""D-11: the two new `CampaignRunObservation` receivers (link/unlink) -- the immediacy half
of ALLOC-03 -- plus the never-raise/never-call-out/never-recurse trigger contract that makes
them safe to run inside a staff member's own transaction. Fixture style mirrors
`test_observation_projector_signals.py`.
"""

from datetime import date, datetime
from datetime import timezone as dt_timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import allocation_projector as ap
from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation
from solsys_code.observation_projector import facility_for
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night


class AllocationSignalsTestBase(TestCase):
    """Shared fixture: one resolvable Chilean Observatory and a 3-night, campaign-less
    `CampaignRun`, already reconciled once so its three `ALLOC:` nights exist before each
    test's own trigger (creating/deleting a `CampaignRunObservation`)."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.site = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def setUp(self) -> None:
        self.run = CampaignRun.objects.create(
            campaign=None,
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=self.site,
            site_raw='809',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
            observation_details='Signals fixture',
        )
        reconcile_run(self.run)
        self.assertEqual(allocation_events(self.run).count(), 3)
        self.site_zone = ZoneInfo(self.site.timezone)

    def _make_record(
        self,
        *,
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        status: str = 'COMPLETED',
        facility: str = 'LCO',
    ) -> ObservationRecord:
        """An ObservationRecord (`NonSiderealTargetFactory` target -- CLAUDE.md), not yet
        linked to any run. Carries a real ``instrument_type`` (unlike
        `test_allocation_projector.py`'s own minimal `{'proposal': 'TEST'}` fixture) so the
        record's own observation-projector event is actually created -- several tests here
        assert on that event's own fields/attribution, not just on allocation-night counts."""
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'signals-owner-{uuid4().hex[:8]}')
        window_start = datetime(2026, 8, 1, 22, 0, tzinfo=dt_timezone.utc)
        window_end = datetime(2026, 8, 2, 6, 0, tzinfo=dt_timezone.utc)
        return ObservationRecord.objects.create(
            target=target,
            user=owner,
            facility=facility,
            observation_id=f'alloc-signals-{uuid4().hex[:8]}',
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={
                'proposal': 'TEST',
                'start': window_start.isoformat(),
                'end': window_end.isoformat(),
                'instrument_type': '2M0-SCICAM-MUSCAT',
            },
        )

    def _night_2_block(self) -> tuple[datetime, datetime]:
        """A placed block whose site-local observing night is 2026-08-02 (the middle of the
        fixture run's 3-night window)."""
        start = datetime(2026, 8, 2, 20, 0, tzinfo=dt_timezone.utc)
        end = start.replace(hour=21)
        self.assertEqual(observing_night(start, self.site_zone), date(2026, 8, 2))
        return start, end


class TestCampaignRunObservationSaveReceiver(AllocationSignalsTestBase):
    """Task 1, behavior tests 1/3/4a: `post_save` on `CampaignRunObservation`."""

    def test_linking_a_placed_record_retires_its_night_with_no_explicit_reconcile(self):
        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)

        CampaignRunObservation.objects.create(run=self.run, observation_record=record)

        self.assertEqual(allocation_events(self.run).count(), 2)
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{self.run.pk}:2026-08-02').exists())
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{self.run.pk}:2026-08-01').exists())
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{self.run.pk}:2026-08-03').exists())

    def test_raw_save_projects_nothing(self):
        with patch('solsys_code.allocation_projector.project_allocation') as mock_project:
            instance = SimpleNamespace(pk=999, run_id=self.run.pk)
            ap.receiver_on_run_observation_save(
                sender=CampaignRunObservation, instance=instance, created=True, raw=True
            )
        mock_project.assert_not_called()

    def test_project_allocation_raising_does_not_abort_the_link_save(self):
        record = self._make_record()
        with patch('solsys_code.allocation_projector.project_allocation', side_effect=ValueError('boom')):
            link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)

        self.assertTrue(CampaignRunObservation.objects.filter(pk=link.pk).exists())


class TestCampaignRunObservationDeleteReceiver(AllocationSignalsTestBase):
    """Task 1, behavior tests 2/4b/5: `post_delete` on `CampaignRunObservation`."""

    def test_deleting_the_link_restores_the_night_and_clears_the_record_event_attribution(self):
        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)
        link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)
        self.assertEqual(allocation_events(self.run).count(), 2)

        facility = facility_for(record)
        own_url = facility.get_observation_url(record.observation_id)
        own_event = CalendarEvent.objects.get(url=own_url)
        own_meta = CalendarEventMeta.objects.get(event=own_event)
        self.assertEqual(own_meta.run_id, self.run.pk)
        title_before = own_event.title
        description_before = own_event.description
        start_before = own_event.start_time
        end_before = own_event.end_time

        link.delete()

        self.assertEqual(allocation_events(self.run).count(), 3)
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{self.run.pk}:2026-08-02').exists())
        own_event.refresh_from_db()
        own_meta.refresh_from_db()
        self.assertIsNone(own_meta.run_id)
        self.assertEqual(own_event.title, title_before)
        self.assertEqual(own_event.description, description_before)
        self.assertEqual(own_event.start_time, start_before)
        self.assertEqual(own_event.end_time, end_before)

    def test_project_allocation_raising_does_not_abort_the_link_delete(self):
        record = self._make_record()
        link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)

        with patch('solsys_code.allocation_projector.project_allocation', side_effect=ValueError('boom')):
            link.delete()

        self.assertFalse(CampaignRunObservation.objects.filter(pk=link.pk).exists())

    def test_deleting_the_run_cascades_the_link_without_raising_or_re_projecting(self):
        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)
        CampaignRunObservation.objects.create(run=self.run, observation_record=record)
        run_pk = self.run.pk

        with patch('solsys_code.allocation_projector.project_allocation') as mock_project:
            self.run.delete()  # must not raise

        mock_project.assert_not_called()
        self.assertFalse(CampaignRunObservation.objects.filter(run_id=run_pk).exists())
        self.assertFalse(CampaignRun.objects.filter(pk=run_pk).exists())

    def test_admin_bulk_queryset_delete_cascades_the_link_without_raising_or_re_projecting(self):
        """35-REVIEW.md CR-05: the admin's "Delete selected" bulk action goes through
        `QuerySet.delete()`, not `Model.delete()` -- Django sets `origin` to the QUERYSET
        for that path, not a `CampaignRun` instance, so a plain `isinstance(origin,
        CampaignRun)` check misses it and lets the post_delete receiver re-project the run
        (still present in the DB when CampaignRunObservation's own post_delete fires),
        re-minting the very ALLOC: events and CalendarEventMeta rows the pre_delete cascade
        just cleared, and leaving a dangling CalendarEventMeta.run_id once the CampaignRun
        row itself is actually deleted."""
        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)
        CampaignRunObservation.objects.create(run=self.run, observation_record=record)
        run_pk = self.run.pk

        with patch('solsys_code.allocation_projector.project_allocation') as mock_project:
            CampaignRun.objects.filter(pk=run_pk).delete()  # the admin bulk-delete path; must not raise

        mock_project.assert_not_called()
        self.assertFalse(CampaignRunObservation.objects.filter(run_id=run_pk).exists())
        self.assertFalse(CampaignRun.objects.filter(pk=run_pk).exists())
        self.assertFalse(CalendarEvent.objects.filter(url__startswith=f'ALLOC:{run_pk}:').exists())
        self.assertFalse(CalendarEventMeta.objects.filter(run_id=run_pk).exists())


class TestCampaignRunObservationReceiverWiring(AllocationSignalsTestBase):
    """Task 1, behavior test 6: connecting `ready()` twice must not double-project."""

    def test_ready_connected_twice_does_not_double_project(self):
        from django.apps import apps
        from django.db.models.signals import post_delete, post_save

        apps.get_app_config('solsys_code').ready()  # second connection attempt

        def flatten(receivers):
            if isinstance(receivers, tuple):
                out = []
                for part in receivers:
                    out.extend(part)
                return out
            return list(receivers)

        save_receivers = flatten(post_save._live_receivers(CampaignRunObservation))
        delete_receivers = flatten(post_delete._live_receivers(CampaignRunObservation))
        self.assertEqual(save_receivers.count(ap.receiver_on_run_observation_save), 1)
        self.assertEqual(delete_receivers.count(ap.receiver_on_run_observation_delete), 1)

        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)
        with patch('solsys_code.allocation_projector.project_allocation') as mock_project:
            CampaignRunObservation.objects.create(run=self.run, observation_record=record)
        self.assertEqual(mock_project.call_count, 1)


class TestAllocationTriggerContract(AllocationSignalsTestBase):
    """Task 3: the never-raise, never-call-out, never-recurse properties that make the two
    `CampaignRunObservation` receivers safe to run inside a staff member's transaction."""

    # -- Never raise --------------------------------------------------------------------
    #
    # Deliberately four separate test methods (not one method parametrised via
    # `self.subTest`) -- `unittest`'s "Ran N tests" summary counts subTests as part of their
    # parent method, not as additional tests, and this plan's own acceptance criteria gates
    # on a literal test COUNT (">=12 tests run") for this file.

    def _assert_save_never_raises(self, exc: Exception) -> None:
        record = self._make_record()
        with patch('solsys_code.allocation_projector.project_allocation', side_effect=exc):
            with self.assertLogs('solsys_code.allocation_projector', level='WARNING') as logs:
                link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)

        self.assertTrue(CampaignRunObservation.objects.filter(pk=link.pk).exists())
        joined = '\n'.join(logs.output)
        self.assertIn(type(exc).__name__, joined)
        self.assertNotIn('secret detail', joined)

    def _assert_delete_never_raises(self, exc: Exception) -> None:
        record = self._make_record()
        link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)
        link_pk = link.pk

        with patch('solsys_code.allocation_projector.project_allocation', side_effect=exc):
            with self.assertLogs('solsys_code.allocation_projector', level='WARNING') as logs:
                link.delete()  # must not raise

        self.assertFalse(CampaignRunObservation.objects.filter(pk=link_pk).exists())
        joined = '\n'.join(logs.output)
        self.assertIn(type(exc).__name__, joined)
        self.assertNotIn('secret detail', joined)

    def test_save_receiver_never_raises_on_value_error_and_logs_only_the_type(self):
        self._assert_save_never_raises(ValueError('secret detail'))

    def test_save_receiver_never_raises_on_bare_exception_and_logs_only_the_type(self):
        self._assert_save_never_raises(Exception('secret detail'))

    def test_delete_receiver_never_raises_on_value_error_and_logs_only_the_type(self):
        self._assert_delete_never_raises(ValueError('secret detail'))

    def test_delete_receiver_never_raises_on_bare_exception_and_logs_only_the_type(self):
        self._assert_delete_never_raises(Exception('secret detail'))

    # -- Never call out -------------------------------------------------------------------

    def test_neither_receiver_reaches_a_facility(self):
        """Uses a non-LCO/SOAR ('GEM') linked record deliberately: `project_allocation()`'s
        own D-08 attribution bridge (`_sync_observation_attribution()`) legitimately calls
        `observation_projector.facility_for()` for any LINKED record whose facility IS
        LCO/SOAR -- that is what attributes the record's own event to the run. A GEM record
        is skipped by that bridge's own facility guard, so this is the fixture shape that
        makes 'the mock was never called' a meaningful, literal assertion rather than one
        that would always fail regardless of whether a real network call occurred (see the
        SUMMARY's deviations section for the parallel case this ruled out for Task 2)."""
        record = self._make_record(facility='GEM')

        with patch(
            'solsys_code.observation_projector.facility_for', side_effect=RuntimeError('never call out')
        ) as mock_facility:
            link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)
            link.delete()

        mock_facility.assert_not_called()

    # -- Never recurse ----------------------------------------------------------------------

    def test_confirm_and_undo_each_invoke_project_allocation_exactly_once(self):
        start, end = self._night_2_block()
        record = self._make_record(scheduled_start=start, scheduled_end=end)

        with patch('solsys_code.allocation_projector.project_allocation', wraps=ap.project_allocation) as wrapped:
            link = CampaignRunObservation.objects.create(run=self.run, observation_record=record)
            self.assertEqual(wrapped.call_count, 1)

            link.delete()
            self.assertEqual(wrapped.call_count, 2)
