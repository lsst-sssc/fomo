"""Unit tests for allocation_projector.project_allocation() (Phase 35, plan 35-01).

Covers ALLOC-01 (the `ALLOC:` namespace), ALLOC-02 (site-local nights, both hemispheres),
ALLOC-03 (the observation handoff) and the D-09/D-10 dispatch seam in
`campaign_reconciler.reconcile_run()`. Fixture style mirrors
`CampaignReconcilerTestBase` in `test_campaign_reconciler.py`.
"""

from datetime import date, datetime, time, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_reconciler import event_description, owned_events, reconcile_run
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night, sun_event


class AllocationProjectorTestBase(TestCase):
    """Shared fixture: two resolvable ground Observatory rows, one per hemisphere.

    Both `809` (La Silla / NTT, Chile) and `E10` (Siding Spring / FTS, Australia) are
    created fresh here rather than assumed present in the developer database -- every
    existing test class in this codebase follows the same convention, and ALLOC-02's
    both-hemispheres requirement needs both fixtures to exist for every test module that
    exercises it (Task 3's `TestAllocationNightBoundary`).
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.chilean_site = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.australian_site = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring Observatory',
            short_name='FTS',
            lat=-31.2734,
            lon=149.0612,
            altitude=1149,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def _make_run(self, **overrides) -> CampaignRun:
        """Create a CampaignRun; kwargs override the default (campaign-less, approved,
        CLASSICAL_FILE-sourced, Chilean-sited) field set."""
        kwargs = {
            'campaign': None,
            'source': CampaignRun.Source.CLASSICAL_FILE,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'telescope_instrument': 'NTT/EFOSC2',
            'site': self.chilean_site,
            'site_raw': '809',
            'window_start': date(2026, 7, 9),
            'window_end': date(2026, 7, 11),
            'observation_details': 'Photometric monitoring',
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _link_record(
        self,
        run: CampaignRun,
        *,
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        facility: str = 'LCO',
        status: str = 'COMPLETED',
    ) -> tuple[ObservationRecord, CampaignRunObservation]:
        """Create an ObservationRecord (NonSiderealTargetFactory target -- CLAUDE.md) and
        link it to `run` via a CampaignRunObservation. Returns (record, link)."""
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'obs-owner-{uuid4().hex[:8]}')
        record = ObservationRecord.objects.create(
            target=target,
            user=owner,
            facility=facility,
            observation_id=f'obs-{uuid4().hex[:8]}',
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={'proposal': 'TEST'},
        )
        link = CampaignRunObservation.objects.create(run=run, observation_record=record)
        return record, link


class TestEndToEndAllocationNight(AllocationProjectorTestBase):
    """Task 1's tracer slice: one path end to end, from `CampaignRun` through
    `reconcile_run()` to `ALLOC:` `CalendarEvent` rows."""

    def test_campaign_less_run_projects_one_alloc_event_per_window_night(self):
        run = self._make_run()

        result = reconcile_run(run)

        self.assertEqual(result.created, 3)
        expected_urls = {
            f'ALLOC:{run.pk}:2026-07-09',
            f'ALLOC:{run.pk}:2026-07-10',
            f'ALLOC:{run.pk}:2026-07-11',
        }
        actual_urls = set(allocation_events(run).values_list('url', flat=True))
        self.assertEqual(actual_urls, expected_urls)
        self.assertEqual(CalendarEvent.objects.filter(url__startswith='RUN:').count(), 0)

    def test_each_alloc_event_carries_the_expected_fields(self):
        run = self._make_run()

        reconcile_run(run)

        for night in (date(2026, 7, 9), date(2026, 7, 10), date(2026, 7, 11)):
            event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
            self.assertEqual(event.title, 'NTT EFOSC2')
            self.assertEqual(event.telescope, 'NTT')
            self.assertEqual(event.instrument, 'EFOSC2')
            self.assertIsNone(event.target_list)
            expected_sunset, expected_sunrise = sun_event(self.chilean_site, night, kind='sun')
            self.assertEqual(
                event.start_time, expected_sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
            )
            self.assertEqual(
                event.end_time, expected_sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
            )

    def test_cancelled_run_status_prefixes_title_and_flip_back_refreshes_in_place(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night, run_status=CampaignRun.RunStatus.CANCELLED)

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        pk_before = event.pk
        self.assertEqual(event.title, '[CANCELLED] NTT EFOSC2')

        run.run_status = CampaignRun.RunStatus.PLANNED
        run.save(update_fields=['run_status'])
        reconcile_run(run)

        event.refresh_from_db()
        self.assertEqual(event.pk, pk_before)
        self.assertEqual(event.title, 'NTT EFOSC2')

    def test_queue_sourced_run_keeps_its_single_container_never_fanned_out(self):
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE)

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        self.assertEqual(CalendarEvent.objects.filter(url=f'RUN:{run.pk}').count(), 1)
        self.assertEqual(allocation_events(run).count(), 0)

    def test_single_night_window_projects_exactly_one_night_never_a_bare_alloc_key(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())
        self.assertEqual(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}').count(), 0)

    def test_second_reconcile_of_unchanged_state_reports_unchanged_and_writes_nothing(self):
        run = self._make_run()
        reconcile_run(run)
        pks_before = set(allocation_events(run).values_list('pk', flat=True))

        result = reconcile_run(run)

        self.assertEqual(result.created, 0)
        self.assertEqual(result.updated, 0)
        self.assertEqual(result.unchanged, 3)
        pks_after = set(allocation_events(run).values_list('pk', flat=True))
        self.assertEqual(pks_before, pks_after)


class TestAllocationEventAttribution(AllocationProjectorTestBase):
    """Every minted allocation night is self-attributed (D-12/`_link_event_to_run`), the
    same as a `RUN:`-keyed container event."""

    def test_every_minted_event_has_a_calendar_event_meta_row_linked_to_the_run(self):
        run = self._make_run()

        reconcile_run(run)

        for event in allocation_events(run):
            meta = CalendarEventMeta.objects.get(event=event)
            self.assertEqual(meta.run_id, run.pk)


class TestAllocationEventDescriptionAndTargetList(AllocationProjectorTestBase):
    """The shared `event_description()`/`event_title()` writers reach allocation nights the
    same way they reach container events, so a `mark_cancelled` action is visible there
    too."""

    def test_description_reuses_shared_event_description_body(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertIn(event_description(run), event.description)

    def test_campaign_target_list_is_written_when_a_campaign_is_present(self):
        campaign = TargetList.objects.create(name='3I/ATLAS')
        night = date(2026, 7, 9)
        run = self._make_run(campaign=campaign, window_start=night, window_end=night)

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertEqual(event.target_list_id, campaign.pk)


class TestNoOrphanEventsAfterAllocationDispatch(AllocationProjectorTestBase):
    """A sanity check that owned_events()/RUN: namespace bookkeeping is untouched by the
    new dispatch branch -- the reconciler's own `RUN:` accounting must stay empty for a
    run that never touches the container branch."""

    def test_owned_events_stays_empty_for_a_per_night_allocation_run(self):
        run = self._make_run()

        reconcile_run(run)

        self.assertEqual(owned_events(run).count(), 0)
        self.assertEqual(allocation_events(run).count(), 3)


class TestObservationHandoff(AllocationProjectorTestBase):
    """Task 2: a linked, placed/observed record retires its night; unlinking restores it
    (D-05/D-06/D-07)."""

    def test_linked_placed_record_retires_its_night(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        scheduled_start = datetime(2026, 7, 10, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)
        site_zone = ZoneInfo(self.chilean_site.timezone)
        retired_night = observing_night(scheduled_start, site_zone)
        retired_url = f'ALLOC:{run.pk}:{retired_night.isoformat()}'

        result = reconcile_run(run)

        self.assertEqual(allocation_events(run).count(), 2)
        self.assertEqual(result.retired, 1)
        self.assertFalse(CalendarEvent.objects.filter(url=retired_url).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event__url=retired_url).exists())

    def test_unlinking_restores_the_retired_night_with_a_fresh_event(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        scheduled_start = datetime(2026, 7, 10, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        _record, link = self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)
        reconcile_run(run)

        link.delete()
        # 35-04 D-11: both the link's own creation (via `_link_record()`, above) and its
        # deletion now re-project the run immediately, through the new post_save/post_delete
        # receivers on CampaignRunObservation (wired in SolsysCodeConfig.ready()) -- so this
        # test's two explicit `reconcile_run()` calls (the one right after `_link_record()`
        # and this one after `link.delete()`) are both now redundant no-ops that converge on
        # already-current state. All three nights (07-09/07-10/07-11) already match, so this
        # call reports `unchanged` for all three, not `created` -- mirrors
        # TestEndToEndAllocationNight's own established "second reconcile of unchanged
        # state" contract.
        result = reconcile_run(run)

        self.assertEqual(result.unchanged, 3)
        site_zone = ZoneInfo(self.chilean_site.timezone)
        retired_night = observing_night(scheduled_start, site_zone)
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{retired_night.isoformat()}').exists())

    def test_queued_only_record_retires_nothing(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        self._link_record(run, scheduled_start=None, scheduled_end=None, status='PENDING')

        result = reconcile_run(run)

        self.assertEqual(result.retired, 0)
        self.assertEqual(allocation_events(run).count(), 3)

    def test_terminal_negative_record_keeps_its_night_retired(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        scheduled_start = datetime(2026, 7, 10, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end, status='WINDOW_EXPIRED')

        result = reconcile_run(run)

        self.assertEqual(result.retired, 1)
        self.assertEqual(allocation_events(run).count(), 2)

    def test_reclassified_window_leaves_no_orphan_for_dropped_night(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        reconcile_run(run)
        self.assertEqual(allocation_events(run).count(), 3)

        run.window_end = date(2026, 7, 10)
        run.save(update_fields=['window_end'])
        result = reconcile_run(run)

        self.assertEqual(allocation_events(run).count(), 2)
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:2026-07-11').exists())
        self.assertEqual(result.retired, 1)

    def test_legacy_run_keyed_night_is_rekeyed_in_place(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)
        legacy_url = f'RUN:{run.pk}:{night.isoformat()}'
        legacy_start = datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc)
        legacy_end = datetime(2026, 7, 10, 10, 0, tzinfo=dt_timezone.utc)
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=legacy_url,
            telescope='NTT',
            instrument='EFOSC2',
            start_time=legacy_start,
            end_time=legacy_end,
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)
        legacy_pk = legacy_event.pk

        result = reconcile_run(run)

        self.assertEqual(result.rekeyed, 1)
        alloc_url = f'ALLOC:{run.pk}:{night.isoformat()}'
        event = CalendarEvent.objects.get(url=alloc_url)
        self.assertEqual(event.pk, legacy_pk)
        self.assertEqual(event.start_time, legacy_start)
        self.assertEqual(event.end_time, legacy_end)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.run_id, run.pk)
        self.assertFalse(CalendarEvent.objects.filter(url=legacy_url).exists())


class TestRetirePathLegacyEventGuard(AllocationProjectorTestBase):
    """35-REVIEW.md CR-03: the retire path's own legacy `RUN:{pk}:{night}` delete gets the
    same ownership and human-confirmation guards the takeover branch already applies to the
    same class of row."""

    def test_retiring_a_night_never_deletes_a_human_confirmed_legacy_event(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)
        legacy_url = f'RUN:{run.pk}:{night.isoformat()}'
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=legacy_url,
            telescope='NTT',
            instrument='EFOSC2',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 10, 0, tzinfo=dt_timezone.utc),
        )
        staff_user = User.objects.create(username='retire-staffer')
        CalendarEventMeta.objects.create(
            event=legacy_event, run=run, confirmed_by=staff_user, confirmed_at=timezone.now()
        )
        scheduled_start = datetime(2026, 7, 9, 23, 30, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=1))

        result = reconcile_run(run)

        self.assertTrue(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())
        meta = CalendarEventMeta.objects.get(event=legacy_event)
        self.assertEqual(meta.confirmed_by_id, staff_user.pk)
        self.assertEqual(result.retired, 1)
        self.assertEqual(result.blocked, 1)

    def test_retiring_a_night_never_deletes_a_legacy_event_attributed_to_a_different_run(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)
        other_run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 1))
        legacy_url = f'RUN:{run.pk}:{night.isoformat()}'
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=legacy_url,
            telescope='NTT',
            instrument='EFOSC2',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 10, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=other_run)
        scheduled_start = datetime(2026, 7, 9, 23, 30, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=1))

        result = reconcile_run(run)

        self.assertTrue(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())
        meta = CalendarEventMeta.objects.get(event=legacy_event)
        self.assertEqual(meta.run_id, other_run.pk)
        self.assertEqual(result.retired, 1)
        self.assertEqual(result.blocked, 1)


class TestAttributionBridge(AllocationProjectorTestBase):
    """Task 2, D-08: attribution is a link on the record's OWN event, both directions."""

    def _make_record_event(self, record: ObservationRecord, start: datetime, end: datetime) -> CalendarEvent:
        """Build the record's own observation-projector-style event, including the
        `CalendarEventMeta.observation_record` link `write_event_meta()` would set in
        production -- required for the unlink half's `observation_record__isnull=False`
        filter to see it."""
        facility = op.facility_for(record)
        event = CalendarEvent.objects.create(
            title='LCO record event',
            url=op.event_url(record, facility),
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=start,
            end_time=end,
        )
        op.write_event_meta(event, record)
        return event

    def test_d08_round_trip_link_and_unlink_attribution(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 9))
        scheduled_start = datetime(2026, 7, 9, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        record, link = self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)
        record_event = self._make_record_event(record, scheduled_start, scheduled_end)
        title_before, description_before = record_event.title, record_event.description
        start_before, end_before = record_event.start_time, record_event.end_time

        reconcile_run(run)

        record_event.refresh_from_db()
        meta = CalendarEventMeta.objects.get(event=record_event)
        self.assertEqual(meta.run_id, run.pk)
        self.assertEqual(record_event.title, title_before)
        self.assertEqual(record_event.description, description_before)
        self.assertEqual(record_event.start_time, start_before)
        self.assertEqual(record_event.end_time, end_before)

        link.delete()
        reconcile_run(run)

        record_event.refresh_from_db()
        self.assertFalse(CalendarEventMeta.objects.filter(event=record_event, run_id=run.pk).exists())
        self.assertEqual(record_event.title, title_before)
        self.assertEqual(record_event.description, description_before)
        self.assertEqual(record_event.start_time, start_before)
        self.assertEqual(record_event.end_time, end_before)

    def test_foreign_attribution_is_refused_and_counted(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 9))
        other_run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 1))
        scheduled_start = datetime(2026, 7, 9, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        record, _link = self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)
        record_event = self._make_record_event(record, scheduled_start, scheduled_end)
        meta = CalendarEventMeta.objects.get(event=record_event)
        meta.run = other_run
        meta.save(update_fields=['run'])

        result = reconcile_run(run)

        self.assertEqual(result.blocked, 1)
        record_event.refresh_from_db()
        meta = CalendarEventMeta.objects.get(event=record_event)
        self.assertEqual(meta.run_id, other_run.pk)

    def test_confirmed_attribution_survives_automated_unlink(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 9))
        scheduled_start = datetime(2026, 7, 9, 20, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=1)
        record, link = self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)
        record_event = self._make_record_event(record, scheduled_start, scheduled_end)
        staff_user = User.objects.create(username='staffer')
        confirmed_at = timezone.now()
        meta = CalendarEventMeta.objects.get(event=record_event)
        meta.run = run
        meta.confirmed_by = staff_user
        meta.confirmed_at = confirmed_at
        meta.save(update_fields=['run', 'confirmed_by', 'confirmed_at'])

        link.delete()
        reconcile_run(run)

        meta = CalendarEventMeta.objects.get(event=record_event)
        self.assertEqual(meta.run_id, run.pk)
        self.assertEqual(meta.confirmed_by_id, staff_user.pk)
        self.assertEqual(meta.confirmed_at, confirmed_at)


class TestAllocationDeletionCascade(AllocationProjectorTestBase):
    """Deleting a CampaignRun takes its own allocation nights with it, and never a night
    attributed to a different run (mirrors campaign_reconciler's RUN: cascade twin)."""

    def test_deleting_run_removes_its_alloc_nights(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 10))
        reconcile_run(run)
        self.assertEqual(allocation_events(run).count(), 2)

        run.delete()

        self.assertEqual(CalendarEvent.objects.filter(url__startswith='ALLOC:').count(), 0)

    def test_deleting_run_a_leaves_run_bs_alloc_night_untouched(self):
        run_a = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 9))
        run_b = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 9))
        reconcile_run(run_a)
        reconcile_run(run_b)
        event_a = CalendarEvent.objects.get(url=f'ALLOC:{run_a.pk}:2026-07-09')
        meta = CalendarEventMeta.objects.get(event=event_a)
        meta.run = run_b
        meta.save(update_fields=['run'])

        run_a.delete()

        self.assertTrue(CalendarEvent.objects.filter(pk=event_a.pk).exists())


class TestAllocationNightBoundary(AllocationProjectorTestBase):
    """ALLOC-02: `retired_nights()`'s local-noon anchor decides which night a linked
    record's placed block retires -- not a plain site-local `.date()`. Covers a Sydney
    site (positive UTC offset, +10 in August, no DST) and a Chilean site (negative UTC
    offset, -4 in August, no DST) side by side, mirroring
    `test_campaign_reconciler.TestObservingNightBoundary`'s verified UTC arithmetic. Every
    assertion names the exact surviving/retired `ALLOC:` url, never a count alone."""

    def _assert_retired_and_surviving(self, run: CampaignRun, retired_night: date, surviving_nights: list[date]):
        retired_url = f'ALLOC:{run.pk}:{retired_night.isoformat()}'
        self.assertFalse(CalendarEvent.objects.filter(url=retired_url).exists())
        for night in surviving_nights:
            self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

    def test_sydney_utc_date_differs_from_the_observing_night_it_retires(self):
        """2026-08-02T01:00Z + 10h = 2026-08-02 11:00 local -- before local noon, so the
        2026-08-01 observing night, even though the naive UTC date is 2026-08-02."""
        run = self._make_run(
            site=self.australian_site,
            site_raw='E10',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
        )
        scheduled_start = datetime(2026, 8, 2, 1, 0, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])

    def test_sydney_exact_local_noon_belongs_to_the_date_that_just_started(self):
        """2026-08-02T02:00:00Z + 10h = 2026-08-02 12:00:00 local exactly."""
        run = self._make_run(
            site=self.australian_site,
            site_raw='E10',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
        )
        scheduled_start = datetime(2026, 8, 2, 2, 0, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 2), [date(2026, 8, 1), date(2026, 8, 3)])

    def test_sydney_one_second_before_local_noon_belongs_to_the_previous_date(self):
        """2026-08-02T01:59:59Z + 10h = 2026-08-02 11:59:59 local."""
        run = self._make_run(
            site=self.australian_site,
            site_raw='E10',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
        )
        scheduled_start = datetime(2026, 8, 2, 1, 59, 59, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])

    def test_sydney_post_local_midnight_start_retires_the_previous_date(self):
        """2026-08-01T16:00Z + 10h = 2026-08-02 02:00 local -- after local midnight, so
        belongs to the observing night that started at sunset on Aug 1."""
        run = self._make_run(
            site=self.australian_site,
            site_raw='E10',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
        )
        scheduled_start = datetime(2026, 8, 1, 16, 0, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])

    def test_chile_utc_date_differs_from_the_observing_night_it_retires(self):
        """2026-08-02T01:00Z - 4h = 2026-08-01 21:00 local -- before local midnight, so
        the 2026-08-01 observing night, even though the naive UTC date is 2026-08-02."""
        run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 3))
        scheduled_start = datetime(2026, 8, 2, 1, 0, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])

    def test_chile_exact_local_noon_belongs_to_the_date_that_just_started(self):
        """2026-08-02T16:00:00Z - 4h = 2026-08-02 12:00:00 local exactly."""
        run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 3))
        scheduled_start = datetime(2026, 8, 2, 16, 0, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 2), [date(2026, 8, 1), date(2026, 8, 3)])

    def test_chile_one_second_before_local_noon_belongs_to_the_previous_date(self):
        """2026-08-02T15:59:59Z - 4h = 2026-08-02 11:59:59 local."""
        run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 3))
        scheduled_start = datetime(2026, 8, 2, 15, 59, 59, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])

    def test_chile_post_local_midnight_start_retires_the_previous_date(self):
        """2026-08-02T04:30:00Z - 4h = 2026-08-02 00:30 local -- after local midnight, so
        belongs to the observing night that started at sunset on Aug 1."""
        run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 3))
        scheduled_start = datetime(2026, 8, 2, 4, 30, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        reconcile_run(run)

        self._assert_retired_and_surviving(run, date(2026, 8, 1), [date(2026, 8, 2), date(2026, 8, 3)])


class TestNoSunEventRecompute(AllocationProjectorTestBase):
    """Closes the folded todo (D-13):
    `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` -- an
    idempotent re-reconcile of an existing multi-night run must make zero `sun_event()`
    calls, and a night whose event was deleted out from under the run must call it exactly
    twice (sun + dark) on the next reconcile, so the "never called" assertion is a real
    gate rather than vacuous."""

    def test_second_reconcile_of_unchanged_run_never_calls_sun_event(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 13))
        reconcile_run(run)
        events_before = {e.pk: (e.start_time, e.end_time) for e in allocation_events(run)}

        with patch('solsys_code.allocation_projector.sun_event') as mock_sun_event:
            result = reconcile_run(run)

        mock_sun_event.assert_not_called()
        self.assertEqual(result.unchanged, 5)
        events_after = {e.pk: (e.start_time, e.end_time) for e in allocation_events(run)}
        self.assertEqual(events_before, events_after)

    def test_a_deleted_night_calls_sun_event_exactly_twice_on_next_reconcile(self):
        run = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 13))
        reconcile_run(run)
        deleted_night = date(2026, 7, 11)
        CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{deleted_night.isoformat()}').delete()

        with patch('solsys_code.allocation_projector.sun_event', wraps=sun_event) as mock_sun_event:
            reconcile_run(run)

        calls_for_deleted_night = [call for call in mock_sun_event.call_args_list if call.args[1] == deleted_night]
        self.assertEqual(len(calls_for_deleted_night), 2)
        kinds = sorted(call.kwargs['kind'] for call in calls_for_deleted_night)
        self.assertEqual(kinds, ['dark', 'sun'])


class TestEmptyAndDegenerateWindows(AllocationProjectorTestBase):
    """Degenerate `CampaignRun` states the dispatch's stage-0 guard (`_skip_reason()`)
    handles before ever reaching `project_allocation()`, plus the no-links edge inside it."""

    def test_null_window_is_skipped_as_tbd(self):
        run = self._make_run(window_start=None, window_end=None)

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'TBD window')
        self.assertEqual(allocation_events(run).count(), 0)

    def test_window_end_before_window_start_is_skipped(self):
        run = self._make_run(window_start=date(2026, 7, 11), window_end=date(2026, 7, 9))

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'window_end before window_start')

    def test_no_observation_links_retires_nothing(self):
        run = self._make_run()

        result = reconcile_run(run)

        self.assertEqual(result.retired, 0)


class TestSubNightWindow(AllocationProjectorTestBase):
    """Plan 35-03 Task 2 (D-04/D-13): the projector honours a run's sub-night window
    fields, and re-mints (never rewrites in place) a night whose span changed."""

    def test_null_null_run_keeps_the_sunset_to_sunrise_span_byte_identical(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        expected_sunset, expected_sunrise = sun_event(self.chilean_site, night, kind='sun')
        self.assertEqual(event.start_time, expected_sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0))
        self.assertEqual(event.end_time, expected_sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0))

    def test_set_end_and_null_start_computes_sunset_start_and_next_morning_end(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night, night_end_utc=time(6, 26))

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        expected_sunset, _expected_sunrise = sun_event(self.chilean_site, night, kind='sun')
        self.assertEqual(event.start_time, expected_sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0))
        next_morning = night + timedelta(days=1)
        self.assertEqual(
            event.end_time,
            datetime(next_morning.year, next_morning.month, next_morning.day, 6, 26, 0, tzinfo=dt_timezone.utc),
        )

    def test_set_start_and_null_end_uses_its_own_evening_date_and_computes_sunrise_end(self):
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night, night_start_utc=time(23, 30))

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertEqual(
            event.start_time,
            datetime(night.year, night.month, night.day, 23, 30, 0, tzinfo=dt_timezone.utc),
        )
        _expected_sunset, expected_sunrise = sun_event(self.chilean_site, night, kind='sun')
        self.assertEqual(event.end_time, expected_sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0))

    def test_changing_night_end_utc_remints_only_the_affected_runs_nights(self):
        run_a = self._make_run(window_start=date(2026, 7, 9), window_end=date(2026, 7, 11))
        run_b = self._make_run(window_start=date(2026, 7, 20), window_end=date(2026, 7, 20))
        reconcile_run(run_a)
        reconcile_run(run_b)
        pks_a_before = {e.url: e.pk for e in allocation_events(run_a)}
        pks_b_before = {e.url: e.pk for e in allocation_events(run_b)}

        run_a.night_end_utc = time(6, 26)
        run_a.save(update_fields=['night_end_utc'])
        result = reconcile_run(run_a)

        self.assertEqual(result.created, 3)
        self.assertEqual(result.retired, 3)
        self.assertEqual(result.updated, 0)
        pks_a_after = {e.url: e.pk for e in allocation_events(run_a)}
        self.assertEqual(set(pks_a_after), set(pks_a_before))
        for url, pk_before in pks_a_before.items():
            self.assertNotEqual(pks_a_after[url], pk_before)

        pks_b_after = {e.url: e.pk for e in allocation_events(run_b)}
        self.assertEqual(pks_b_before, pks_b_after)

    def test_reconcile_with_sub_night_fields_set_makes_no_further_sun_event_calls(self):
        run = self._make_run(
            window_start=date(2026, 7, 9),
            window_end=date(2026, 7, 11),
            night_start_utc=time(23, 30),
            night_end_utc=time(6, 26),
        )
        reconcile_run(run)

        with patch('solsys_code.allocation_projector.sun_event') as mock_sun_event:
            result = reconcile_run(run)

        mock_sun_event.assert_not_called()
        self.assertEqual(result.unchanged, 3)

    def test_second_reconcile_with_matching_sub_night_fields_writes_nothing(self):
        run = self._make_run(
            window_start=date(2026, 7, 9),
            window_end=date(2026, 7, 11),
            night_start_utc=time(23, 30),
            night_end_utc=time(6, 26),
        )
        reconcile_run(run)
        pks_before = {e.url: e.pk for e in allocation_events(run)}

        result = reconcile_run(run)

        self.assertEqual(result.created, 0)
        self.assertEqual(result.updated, 0)
        self.assertEqual(result.retired, 0)
        self.assertEqual(result.unchanged, 3)
        pks_after = {e.url: e.pk for e in allocation_events(run)}
        self.assertEqual(pks_before, pks_after)
