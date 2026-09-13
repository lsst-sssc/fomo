"""Unit tests for allocation_projector.project_allocation() (Phase 35, plan 35-01).

Covers ALLOC-01 (the `ALLOC:` namespace), ALLOC-02 (site-local nights, both hemispheres),
ALLOC-03 (the observation handoff) and the D-09/D-10 dispatch seam in
`campaign_reconciler.reconcile_run()`. Fixture style mirrors
`CampaignReconcilerTestBase` in `test_campaign_reconciler.py`.
"""

from datetime import date
from datetime import timezone as dt_timezone

from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_reconciler import event_description, owned_events, reconcile_run
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import sun_event


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
