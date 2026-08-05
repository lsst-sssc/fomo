"""Unit tests for campaign_reconciler.reconcile_run() (plan 29-01, Task 3).

Covers RECON-02 (queue half), RECON-03, RECON-05, RECON-06's dry-run, and RECON-01's
unit-level idempotency, isolated from the Django view/command layer. Fixture style mirrors
CampaignApprovalTestBase in test_campaign_approval.py.
"""

from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import patch

from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.campaign_reconciler import event_title, owned_events, reconcile_run
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import sun_event


class CampaignReconcilerTestBase(TestCase):
    """Shared fixture: one campaign, one resolvable ground Observatory, one satellite one."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.ground_site = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.satellite_site = Observatory.objects.create(
            obscode='250',
            name='Test Space Telescope',
            short_name='TST',
            observations_type=Observatory.SATELLITE_OBSTYPE,
        )

    def _make_run(self, **overrides) -> CampaignRun:
        """Create a CampaignRun; kwargs override the default (approved, ground-sited) field set."""
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site': self.ground_site,
            'site_raw': 'F65',
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 1),
            'observation_details': 'Photometric monitoring',
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)


class TestSkipReasons(CampaignReconcilerTestBase):
    """One test per _skip_reason() branch (D-05's itemized skip vocabulary)."""

    def test_pending_review_run_is_not_approved(self):
        run = self._make_run(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'not approved')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_blank_telescope_instrument_is_missing_telescope_instrument(self):
        run = self._make_run(telescope_instrument='')

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'missing telescope/instrument')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_unset_window_start_is_tbd_window(self):
        run = self._make_run(window_start=None, window_end=None)

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'TBD window')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_no_site_and_no_telescope_class_is_unresolved_site(self):
        run = self._make_run(site=None, site_raw='', telescope_class='')

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'unresolved site')
        self.assertEqual(CalendarEvent.objects.count(), 0)


class TestQueueStage1(CampaignReconcilerTestBase):
    """RECON-02 queue half: a queue-scheduled run projects one bare RUN:{pk} container."""

    def test_lco_queue_multi_night_run_creates_one_bare_container_event(self):
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 15),
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        event = events.get()
        self.assertEqual(event.url, f'RUN:{run.pk}')
        self.assertEqual(event.start_time, datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(event.end_time, datetime(2026, 8, 15, 23, 59, tzinfo=dt_timezone.utc))
        self.assertIn('(window 2026-08-01..2026-08-15)', event.title)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.run_id, run.pk)

    def test_gemini_queue_multi_night_run_creates_one_bare_container_event(self):
        run = self._make_run(
            source=CampaignRun.Source.GEMINI_QUEUE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 15),
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')


class TestClassWideStage2(CampaignReconcilerTestBase):
    """RECON-03: a class-wide (or SPACE-classed) run projects a single bare container."""

    def test_class_wide_site_less_run_creates_one_container_and_is_not_skipped(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.TWO_M0,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )

        result = reconcile_run(run)

        self.assertIsNone(result.skipped_reason)
        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')

    def test_space_classed_run_shares_the_same_container_branch(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.SPACE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )

        result = reconcile_run(run)

        self.assertIsNone(result.skipped_reason)
        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')


class TestSatelliteContainer(CampaignReconcilerTestBase):
    """The ported satellite case: one bare RUN:{pk} whole-day-span event, no sun_event() call."""

    def test_satellite_run_creates_one_container_event_without_calling_sun_event(self):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError('sun_event() must never be called for a satellite run')

        run = self._make_run(
            site=self.satellite_site,
            site_raw='250',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )

        with patch('solsys_code.campaign_reconciler.sun_event', side_effect=_fail_if_called):
            result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        event = events.get()
        self.assertEqual(event.url, f'RUN:{run.pk}')
        self.assertEqual(event.start_time, datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(event.end_time, datetime(2026, 8, 5, 23, 59, tzinfo=dt_timezone.utc))


class TestOwnershipScoping(CampaignReconcilerTestBase):
    """RECON-05: the reconciler never creates, modifies or deletes an event it does not own."""

    def test_unowned_same_window_event_is_left_completely_untouched(self):
        """A hand-made event (blank url, no companion row) whose start_time falls inside the
        run's window is never adopted, modified or linked to a CalendarEventMeta row."""
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )
        orphan = CalendarEvent.objects.create(
            title='Unrelated conference',
            url='',
            start_time=datetime(2026, 8, 2, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 2, 12, 0, tzinfo=dt_timezone.utc),
        )
        modified_before = orphan.modified

        reconcile_run(run)

        orphan.refresh_from_db()
        self.assertEqual(orphan.title, 'Unrelated conference')
        self.assertEqual(orphan.start_time, datetime(2026, 8, 2, 10, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(orphan.end_time, datetime(2026, 8, 2, 12, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(orphan.modified, modified_before)
        self.assertFalse(CalendarEventMeta.objects.filter(event=orphan).exists())

    def test_event_owned_by_a_different_run_is_blocked_and_untouched(self):
        """An event already keyed under this run's RUN:{pk} namespace, but whose companion
        row points at a DIFFERENT run, is blocked -- never written, never re-attributed."""
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )
        other_run = self._make_run(
            telescope_instrument='Other Telescope/Instrument',
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )
        clashing_event = CalendarEvent.objects.create(
            title='Owned by a different run',
            url=f'RUN:{run.pk}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 5, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=clashing_event, run=other_run)
        modified_before = clashing_event.modified

        result = reconcile_run(run)

        self.assertEqual(result.blocked, 1)
        clashing_event.refresh_from_db()
        self.assertEqual(clashing_event.title, 'Owned by a different run')
        self.assertEqual(clashing_event.modified, modified_before)

    def test_owned_events_trailing_colon_guard_excludes_a_different_runs_night(self):
        """owned_events(run) for run pk=3 must not match an event keyed RUN:34:2026-08-01."""
        run = self._make_run()
        # Force a low, predictable pk gap is unnecessary -- just create another run with a
        # numerically-later pk and assert its per-night event never matches run's query.
        other_run = self._make_run(telescope_instrument='Other Telescope/Instrument')
        other_event = CalendarEvent.objects.create(
            title='Other run night',
            url=f'RUN:{other_run.pk}:2026-08-01',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )

        self.assertNotIn(other_event, list(owned_events(run)))


class TestContainerIdempotency(CampaignReconcilerTestBase):
    """RECON-01 (unit level) and RECON-06's dry-run."""

    def test_second_reconcile_is_unchanged_and_dry_run_matches(self):
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE)

        first = reconcile_run(run)
        self.assertEqual(first.created, 1)
        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        modified_after_first = event.modified

        second = reconcile_run(run)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_after_first)

        third = reconcile_run(run, dry_run=True)
        self.assertEqual(third.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_after_first)

    def test_dry_run_on_never_reconciled_run_reports_created_and_writes_nothing(self):
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE)

        result = reconcile_run(run, dry_run=True)

        self.assertEqual(result.created, 1)
        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertEqual(CalendarEventMeta.objects.count(), 0)


class TestAdoptAndRekey(CampaignReconcilerTestBase):
    """D-02: an already-attributed classical night is re-keyed in place, never duplicated."""

    def _make_adopted_event(self, night: date, *, minutes_offset: int = 7) -> CalendarEvent:
        """A CalendarEvent shaped like one `load_telescope_runs` creates: blank url,
        telescope/instrument set, start_time/end_time offset from the reconciler's own
        sunset/sunrise -- the file-derived BoN/EoN window this test proves survives."""
        real_sunset, real_sunrise = sun_event(self.ground_site, night, kind='sun')
        start_time = real_sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0) + timedelta(
            minutes=minutes_offset
        )
        end_time = real_sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0) - timedelta(
            minutes=minutes_offset
        )
        return CalendarEvent.objects.create(
            title='FTN MuSCAT3',
            url='',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=start_time,
            end_time=end_time,
        )

    def test_adopted_night_is_rekeyed_in_place_and_not_duplicated(self):
        first_night = date(2026, 8, 1)
        second_night = date(2026, 8, 2)
        run = self._make_run(window_start=first_night, window_end=second_night)
        adopted_event = self._make_adopted_event(first_night)
        adopted_pk = adopted_event.pk
        meta = CalendarEventMeta.objects.create(event=adopted_event, run=run, is_verified=False)
        adopted_start, adopted_end = adopted_event.start_time, adopted_event.end_time

        result = reconcile_run(run)

        # One adopted (re-keyed) event + one minted for the second night -- never 3.
        self.assertEqual(CalendarEvent.objects.count(), 2)
        adopted_event.refresh_from_db()
        self.assertEqual(adopted_event.pk, adopted_pk)
        self.assertEqual(adopted_event.url, f'RUN:{run.pk}:{first_night.isoformat()}')
        # The file-derived window survived untouched.
        self.assertEqual(adopted_event.start_time, adopted_start)
        self.assertEqual(adopted_event.end_time, adopted_end)
        self.assertEqual(adopted_event.title, event_title(run))
        meta.refresh_from_db()
        self.assertFalse(meta.is_verified)
        self.assertEqual(meta.run_id, run.pk)
        second_night_event = CalendarEvent.objects.get(url=f'RUN:{run.pk}:{second_night.isoformat()}')
        self.assertNotEqual(second_night_event.pk, adopted_pk)
        self.assertEqual(result.updated, 1)
        self.assertEqual(result.created, 1)

    def test_rekey_is_sticky_and_second_reconcile_reports_unchanged(self):
        first_night = date(2026, 8, 1)
        second_night = date(2026, 8, 2)
        run = self._make_run(window_start=first_night, window_end=second_night)
        adopted_event = self._make_adopted_event(first_night)
        CalendarEventMeta.objects.create(event=adopted_event, run=run)

        first = reconcile_run(run)
        self.assertEqual(first.updated, 1)
        self.assertEqual(first.created, 1)
        adopted_event.refresh_from_db()
        modified_after_first = adopted_event.modified

        second = reconcile_run(run)

        self.assertEqual(second.unchanged, 2)
        self.assertEqual(CalendarEvent.objects.count(), 2)
        adopted_event.refresh_from_db()
        self.assertEqual(adopted_event.modified, modified_after_first)

    def test_adopt_matches_on_site_local_night_not_naive_utc_date(self):
        """Mirrors 26-DECISION.md's measured event pk=54 case: a start_time whose naive-UTC
        date is one day before its Australia/Sydney site-local date is still adopted for
        the site-local observing night, not the naive-UTC one."""
        night = date(2026, 7, 9)
        run = self._make_run(window_start=night, window_end=night)
        # 2026-07-08T14:08:19Z + 10h (Sydney AEST, no DST in July) = 2026-07-09 00:08:19
        # local -- naive-UTC date is 2026-07-08, one day before the site-local night.
        start_time = datetime(2026, 7, 8, 14, 8, 19, tzinfo=dt_timezone.utc)
        end_time = start_time + timedelta(hours=8)
        adopted_event = CalendarEvent.objects.create(
            title='FTN MuSCAT3',
            url='',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=start_time,
            end_time=end_time,
        )
        CalendarEventMeta.objects.create(event=adopted_event, run=run)

        reconcile_run(run)

        adopted_event.refresh_from_db()
        self.assertEqual(adopted_event.url, f'RUN:{run.pk}:{night.isoformat()}')
        self.assertEqual(CalendarEvent.objects.count(), 1)
