"""Unit tests for the pure-computation coverage-gap module (GAP-02/GAPB-01) + import guard
(GAP-01).

`campaign_gap.py` depends on `telescope_runs.sun_event`/`observing_night` for ephemerides and
site-local nights, and (as of GAPB-01) on `calendar_utils`, `campaign_attribution`,
`observation_projector` and `status_vocabulary` for the second, observation-event claim
source -- never the heavy SPICE-loading ephemeris/views module. This module's own static
import-guard test mirrors the grep this file's plan verification step also runs, so the two stay
in agreement.

Always uses `tom_targets.tests.factories.NonSiderealTargetFactory` for any Target fixture --
never `SiderealTargetFactory` (CLAUDE.md: FOMO is exclusively for Solar System / non-sidereal
targets).
"""

import inspect
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from unittest import mock

from django.core.cache import cache
from django.db.models.signals import post_save
from django.test import TestCase, override_settings
from django.urls import reverse
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import campaign_gap
from solsys_code.campaign_gap import (
    DEFAULT_WINDOW_DAYS,
    MAX_WINDOW_DAYS,
    build_gap_cache_key,
    claimed_dates,
    clamp_date_range,
    observable_dates,
    observation_claimed_dates,
)
from solsys_code.models import CampaignRun
from solsys_code.observation_projector import receiver_on_record_save
from solsys_code.solsys_code_observatory.models import Observatory

TEST_CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}


class TestClampDateRange(TestCase):
    """D-11: 90-day default window; 180-day hard cap; a smaller request is honoured."""

    def test_default_window_is_90_days(self):
        today = date(2026, 7, 4)
        start, end = clamp_date_range(today, None)
        self.assertEqual(start, today)
        self.assertEqual(end, today + timedelta(days=DEFAULT_WINDOW_DAYS))

    def test_far_future_end_clamps_to_180_days(self):
        today = date(2026, 7, 4)
        start, end = clamp_date_range(today, today + timedelta(days=500))
        self.assertEqual(start, today)
        self.assertEqual(end, today + timedelta(days=MAX_WINDOW_DAYS))

    def test_request_inside_cap_is_honoured(self):
        today = date(2026, 7, 4)
        start, end = clamp_date_range(today, today + timedelta(days=30))
        self.assertEqual(start, today)
        self.assertEqual(end, today + timedelta(days=30))


class TestBuildGapCacheKey(TestCase):
    """D-10: cache key includes all four dimensions; null target encoded as 'none'."""

    def test_key_contains_all_four_dimensions(self):
        d0 = date(2026, 7, 4)
        d1 = date(2026, 10, 2)
        key = build_gap_cache_key(1, None, 5, d0, d1)
        self.assertIn('1', key)
        self.assertIn('none', key)
        self.assertIn('5', key)
        self.assertIn(d0.isoformat(), key)
        self.assertIn(d1.isoformat(), key)

    def test_null_vs_real_target_do_not_collide(self):
        d0 = date(2026, 7, 4)
        d1 = date(2026, 10, 2)
        key_none = build_gap_cache_key(1, None, 5, d0, d1)
        key_real = build_gap_cache_key(1, 7, 5, d0, d1)
        self.assertNotEqual(key_none, key_real)


class TestObservableDates(TestCase):
    """D-03/D-04: non-zero dark window counts as observable; a ValueError date is skipped."""

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='268',
            name='Las Campanas (Magellan-Clay)',
            short_name='Magellan-Clay',
            lon=-70.6926,
            lat=-29.0146,
            altitude=2402.0,
            timezone='America/Santiago',
        )

    def test_returns_dates_with_nonzero_dark_window(self):
        start = date(2026, 6, 10)
        end = date(2026, 6, 12)
        result = observable_dates(self.site, start, end)
        # All 3 nights at a mid-latitude site should have a real dark window.
        self.assertEqual(result, {start, start + timedelta(days=1), end})

    def test_valueerror_date_is_skipped_loop_completes(self):
        start = date(2026, 6, 10)
        end = date(2026, 6, 12)
        middle = start + timedelta(days=1)

        real_sun_event = campaign_gap.sun_event

        def flaky_sun_event(site, d, kind):
            if d == middle:
                raise ValueError('simulated unknown date')
            return real_sun_event(site, d, kind)

        with mock.patch('solsys_code.campaign_gap.sun_event', side_effect=flaky_sun_event):
            result = observable_dates(self.site, start, end)

        self.assertNotIn(middle, result)
        self.assertIn(start, result)
        self.assertIn(end, result)


@override_settings(CACHES=TEST_CACHES)
class TestClaimedDates(TestCase):
    """D-05/D-08: window-range claiming, exclusions, and undated (TBD) flagging."""

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='269',
            name='Las Campanas (Magellan-Baade)',
            short_name='Magellan-Baade',
            lon=-70.6926,
            lat=-29.0146,
            altitude=2402.0,
            timezone='America/Santiago',
        )
        cls.other_site = Observatory.objects.create(
            obscode='809',
            name='La Silla (NTT)',
            short_name='NTT',
            lon=-70.7345,
            lat=-29.2567,
            altitude=2400.0,
            timezone='America/Santiago',
        )
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)

    def setUp(self):
        cache.clear()

    def _make_run(self, **kwargs):
        defaults = {
            'campaign': self.campaign,
            'telescope_instrument': 'Magellan-Baade/IMACS',
            'site': self.site,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'run_status': CampaignRun.RunStatus.OBSERVED,
        }
        defaults.update(kwargs)
        return CampaignRun.objects.create(**defaults)

    def test_approved_run_claims_its_single_night_window(self):
        night = date(2026, 7, 10)
        self._make_run(window_start=night, window_end=night, telescope_instrument='A')
        claimed, undated, unattributed, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(night, claimed)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(undated, [])
        self.assertEqual(unattributed, [])
        self.assertEqual(pending_narrowing, [])

    def test_range_run_claims_every_date_in_window(self):
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 4)
        self._make_run(window_start=window_start, window_end=window_end, telescope_instrument='RANGE')
        claimed, undated, _, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.site)
        expected = {
            date(2026, 8, 1),
            date(2026, 8, 2),
            date(2026, 8, 3),
            date(2026, 8, 4),
        }
        self.assertEqual(claimed, expected)
        self.assertEqual(undated, [])
        # A ground run with a range never lands in pending_narrowing_runs -- that bucket
        # is space-only (ASSET-02).
        self.assertEqual(pending_narrowing, [])

    def test_cancelled_run_not_claimed(self):
        night = date(2026, 7, 11)
        self._make_run(
            window_start=night, window_end=night, telescope_instrument='B', run_status=CampaignRun.RunStatus.CANCELLED
        )
        claimed, _, _, _, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)

    def test_pending_review_run_not_claimed(self):
        night = date(2026, 7, 12)
        self._make_run(
            window_start=night,
            window_end=night,
            telescope_instrument='C',
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        claimed, _, _, _, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)

    def test_undated_runs_flagged(self):
        run = self._make_run(window_start=None, window_end=None, telescope_instrument='E')
        claimed, undated, _, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(run, undated)
        self.assertNotIn(None, claimed)
        # No date should have been added on behalf of this run.
        self.assertEqual(len(claimed), 0)
        # TBD runs never land in pending_narrowing_runs (D-09 explicit distinction).
        self.assertEqual(pending_narrowing, [])

    def test_different_site_not_claimed(self):
        night = date(2026, 7, 15)
        self._make_run(window_start=night, window_end=night, telescope_instrument='F', site=self.other_site)
        claimed, _, _, _, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)


@override_settings(CACHES=TEST_CACHES)
class TestClaimedDatesSpaceMission(TestCase):
    """ASSET-01/ASSET-02/D-09: space-mission runs claim nothing until narrowed to a
    single night; an un-narrowed range lands in pending_narrowing_runs, never
    undated_runs; a TBD space-mission run lands in undated_runs, never
    pending_narrowing_runs (D-09's explicit distinction)."""

    @classmethod
    def setUpTestData(cls):
        cls.space_site = Observatory.objects.create(
            obscode='250',
            name='Test Space Telescope',
            short_name='TST',
            observations_type=Observatory.SATELLITE_OBSTYPE,
        )
        cls.campaign = TargetList.objects.create(name='Space Campaign')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)

    def setUp(self):
        cache.clear()

    def _make_run(self, **kwargs):
        defaults = {
            'campaign': self.campaign,
            'telescope_instrument': 'HST/WFC3',
            'site': self.space_site,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'run_status': CampaignRun.RunStatus.OBSERVED,
        }
        defaults.update(kwargs)
        return CampaignRun.objects.create(**defaults)

    def test_narrowed_space_run_claims_its_single_night(self):
        night = date(2026, 9, 1)
        self._make_run(window_start=night, window_end=night, telescope_instrument='Narrowed')
        claimed, undated, _, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.space_site)
        self.assertEqual(claimed, {night})
        self.assertEqual(undated, [])
        self.assertEqual(pending_narrowing, [])

    def test_unnarrowed_space_run_claims_nothing_and_lands_in_pending_narrowing(self):
        window_start = date(2026, 9, 1)
        window_end = date(2026, 9, 10)
        run = self._make_run(window_start=window_start, window_end=window_end, telescope_instrument='Unnarrowed')
        claimed, undated, _, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.space_site)
        self.assertEqual(claimed, set())
        self.assertEqual(undated, [])
        self.assertIn(run, pending_narrowing)
        self.assertEqual(len(pending_narrowing), 1)

    def test_tbd_space_run_lands_in_undated_not_pending_narrowing(self):
        run = self._make_run(window_start=None, window_end=None, telescope_instrument='TBD')
        claimed, undated, _, pending_narrowing, _, _ = claimed_dates(self.campaign, self.target, self.space_site)
        self.assertEqual(claimed, set())
        self.assertIn(run, undated)
        self.assertEqual(pending_narrowing, [])


@override_settings(CACHES=TEST_CACHES)
class TestClaimedDatesMultiTarget(TestCase):
    """Pitfall 4: a multi-target campaign's target=None runs are unattributed, not counted."""

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring (FTS)',
            short_name='FTS',
            lon=149.0708,
            lat=-31.2733,
            altitude=1165.0,
            timezone='Australia/Sydney',
        )
        cls.campaign = TargetList.objects.create(name='Multi-target Campaign')
        cls.target_a = NonSiderealTargetFactory.create()
        cls.target_b = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target_a, cls.target_b)

    def setUp(self):
        cache.clear()

    def test_target_none_run_is_unattributed_not_claimed_for_either_target(self):
        night = date(2026, 7, 20)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/Sinistro',
            site=self.site,
            target=None,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed_a, _, unattributed_a, _, _, _ = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, unattributed_b, _, _, _ = claimed_dates(self.campaign, self.target_b, self.site)
        self.assertNotIn(night, claimed_a)
        self.assertNotIn(night, claimed_b)
        self.assertEqual(len(unattributed_a), 1)
        self.assertEqual(len(unattributed_b), 1)

    def test_target_specific_run_claimed_only_for_its_own_target(self):
        night = date(2026, 7, 21)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/Sinistro-2',
            site=self.site,
            target=self.target_a,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed_a, _, _, _, _, _ = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, _, _, _, _ = claimed_dates(self.campaign, self.target_b, self.site)
        self.assertIn(night, claimed_a)
        self.assertNotIn(night, claimed_b)


def _create_observation_record_without_projection(**kwargs):
    """Create an ObservationRecord with the observation projector's post_save receiver
    disconnected (34-01/WR-02 precedent, `test_campaign_attribution.py`): these fixtures'
    deliberately incomplete/synthetic `parameters`/schedule fields would otherwise either
    log as 'unprojectable' or, for a would-be-complete fixture, auto-create a
    CalendarEventMeta this module's own manual attribution fixtures would then collide with
    (`CalendarEventMeta.observation_record` is a OneToOneField).
    """
    post_save.disconnect(
        receiver_on_record_save,
        sender=ObservationRecord,
        dispatch_uid='solsys_code.observation_projector.post_save',
    )
    try:
        return ObservationRecord.objects.create(**kwargs)
    finally:
        post_save.connect(
            receiver_on_record_save,
            sender=ObservationRecord,
            weak=False,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )


@override_settings(CACHES=TEST_CACHES)
class TestObservationClaimedDates(TestCase):
    """GAPB-01 (D-16..D-19): observed/scheduled observation blocks claim their site-local
    night alongside approved run windows; a queued/expired/cancelled/failed/inconsistent
    record claims nothing; a record whose site cannot be resolved is counted, never
    silently dropped; the claimed set is a union regardless of which source adds a night
    first.
    """

    @classmethod
    def setUpTestData(cls):
        # Haleakala/FTN -- a real, verified rung-1 (observed_site/observed_telescope
        # parameter) resolution target: derive_telescope('ogg', '2m0a') -> 'FTN' ->
        # campaign_attribution.OBSERVED_TELESCOPE_OBSCODES['FTN'] -> 'F65'.
        cls.site = Observatory.objects.create(
            obscode='F65',
            name='Haleakala (FTN)',
            short_name='FTN',
            lon=-156.2570,
            lat=20.7075,
            altitude=3055.0,
            timezone='Pacific/Honolulu',
        )
        # Siding Spring/FTS -- a real, verified rung-1 SITECODE-CLASS resolution target:
        # derive_telescope('coj', '1m0a') -> 'COJ-1m0' -> LCO_SITE_CODE_TO_OBSCODE['coj'] ->
        # 'E10'. Used as the "resolves to a DIFFERENT observatory" fixture below.
        cls.other_site = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring (FTS)',
            short_name='FTS',
            lon=149.0708,
            lat=-31.2733,
            altitude=1165.0,
            timezone='Australia/Sydney',
        )
        cls.campaign = TargetList.objects.create(name='Observation Claims Campaign')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)

    def setUp(self):
        cache.clear()

    def _make_record(self, **kwargs):
        defaults = {
            'target': self.target,
            'facility': 'LCO',
            'status': 'COMPLETED',
            'parameters': {},
        }
        defaults.update(kwargs)
        return _create_observation_record_without_projection(**defaults)

    def test_observed_block_claims_its_site_local_night(self):
        night = date(2026, 8, 1)
        self._make_record(
            observation_id='OBSCLAIM-OBSERVED',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 1, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 2, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, {night})
        self.assertEqual(site_unknown, 0)

    def test_scheduled_block_claims_its_site_local_night(self):
        night = date(2026, 8, 3)
        self._make_record(
            observation_id='OBSCLAIM-SCHEDULED',
            status='PENDING',
            scheduled_start=datetime(2026, 8, 3, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 4, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, {night})
        self.assertEqual(site_unknown, 0)

    def test_queued_record_with_a_request_window_claims_nothing(self):
        self._make_record(
            observation_id='OBSCLAIM-QUEUED',
            status='PENDING',
            parameters={
                'observed_site': 'ogg',
                'observed_telescope': '2m0a',
                'start': '2026-08-05T22:00:00',
                'end': '2026-08-06T04:00:00',
            },
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 0)

    def test_terminal_and_inconsistent_records_claim_nothing(self):
        for i, status in enumerate(('WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED')):
            self._make_record(
                observation_id=f'OBSCLAIM-TERMINAL-{i}',
                status=status,
                scheduled_start=datetime(2026, 8, 10 + i, 22, 0, tzinfo=dt_timezone.utc),
                scheduled_end=datetime(2026, 8, 11 + i, 4, 0, tzinfo=dt_timezone.utc),
                parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
            )
        # Inconsistent: only scheduled_start set.
        self._make_record(
            observation_id='OBSCLAIM-INCONSISTENT',
            status='PENDING',
            scheduled_start=datetime(2026, 8, 20, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=None,
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 0)

    def test_record_resolving_to_a_different_observatory_claims_nothing_for_this_site(self):
        self._make_record(
            observation_id='OBSCLAIM-OTHERSITE',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 15, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 16, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'coj', 'observed_telescope': '1m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 0)
        # Sanity: the SAME record does claim its night for the site it actually resolves to.
        other_claimed, _ = observation_claimed_dates(self.campaign, self.target, self.other_site)
        self.assertEqual(other_claimed, {date(2026, 8, 15)})

    def test_record_with_unresolvable_site_increments_unknown_count_not_claimed(self):
        self._make_record(
            observation_id='OBSCLAIM-UNKNOWNSITE',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 17, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 18, 4, 0, tzinfo=dt_timezone.utc),
            parameters={},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 1)

    def test_record_with_unconfigured_facility_increments_unknown_count_never_raises(self):
        """CR-03 (37-REVIEW.md) regression: observation_projector.facility_for() raises
        ImportError for a facility name absent from TOM_FACILITY_CLASSES -- this must
        degrade to the same site-unknown bucket an unresolvable site already falls into,
        never a 500 on the anonymous gap-analysis page."""
        self._make_record(
            observation_id='OBSCLAIM-BADFACILITY',
            facility='NOT_A_CONFIGURED_FACILITY',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 17, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 18, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 1)

    def test_record_time_window_raising_is_skipped_never_aborts_the_loop(self):
        # status='COMPLETED' with both schedule fields None still classifies OBSERVED
        # (status_vocabulary.classify_record()'s "completed-no-block" case) -- with no
        # parameters['start']/['end'] either, record_time_window() raises KeyError, which
        # must be skipped as unknown rather than aborting the whole computation. A second,
        # perfectly good record proves the loop really does continue past it.
        self._make_record(
            observation_id='OBSCLAIM-RAISES',
            status='COMPLETED',
            scheduled_start=None,
            scheduled_end=None,
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        good_night = date(2026, 8, 19)
        self._make_record(
            observation_id='OBSCLAIM-RAISES-SIBLING',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 19, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 20, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        observation_claimed, site_unknown = observation_claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(observation_claimed, {good_night})
        self.assertEqual(site_unknown, 0)

    def test_union_with_approved_run_window_produces_one_claimed_date(self):
        night = date(2026, 8, 1)
        self._make_record(
            observation_id='OBSCLAIM-UNION',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 1, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 8, 2, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed, _, _, _, observation_claimed, site_unknown = claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(claimed, {night})
        self.assertEqual(observation_claimed, {night})
        self.assertEqual(site_unknown, 0)

    def test_result_claimed_dates_sorted_and_unique_run_window_seeded_first(self):
        night = date(2026, 9, 1)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        self._make_record(
            observation_id='OBSCLAIM-ORDER-A',
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 2, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        result = campaign_gap._compute_gap(self.campaign, self.target, self.site, night, night)
        self.assertEqual(result['claimed_dates'], [night])

    def test_result_claimed_dates_sorted_and_unique_observation_seeded_first(self):
        night = date(2026, 9, 2)
        self._make_record(
            observation_id='OBSCLAIM-ORDER-B',
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 3, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        result = campaign_gap._compute_gap(self.campaign, self.target, self.site, night, night)
        self.assertEqual(result['claimed_dates'], [night])

    def test_campaign_with_no_observation_events_behaves_like_run_window_only(self):
        night = date(2026, 8, 30)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=night,
            window_end=night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed, _, _, _, observation_claimed, site_unknown = claimed_dates(self.campaign, self.target, self.site)
        self.assertEqual(claimed, {night})
        self.assertEqual(observation_claimed, set())
        self.assertEqual(site_unknown, 0)

    def test_campaign_with_observations_but_no_runs_is_well_formed(self):
        night = date(2026, 8, 31)
        self._make_record(
            observation_id='OBSCLAIM-NORUNS',
            status='COMPLETED',
            scheduled_start=datetime(2026, 8, 31, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 1, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )
        claimed, undated, unattributed, pending_narrowing, observation_claimed, site_unknown = claimed_dates(
            self.campaign, self.target, self.site
        )
        self.assertEqual(claimed, {night})
        self.assertEqual(observation_claimed, {night})
        self.assertEqual(undated, [])
        self.assertEqual(unattributed, [])
        self.assertEqual(pending_narrowing, [])
        self.assertEqual(site_unknown, 0)

    def test_compute_gap_bounds_claimed_dates_to_the_requested_range(self):
        """WR-11 (37-REVIEW.md): claimed_dates()/observation_claimed_dates() are campaign/
        site-wide, not scoped to [start, end] -- _compute_gap()'s own 'claimed_dates'/
        'observation_claimed_dates' result keys (rendered as the "Claimed nights" list)
        must be bounded to the requested range, so a user asking about the next 30 days is
        not shown claimed nights from years ago or years ahead."""
        in_range_night = date(2026, 9, 1)
        out_of_range_run_night = date(2020, 1, 1)
        out_of_range_obs_night = date(2030, 1, 1)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=in_range_night,
            window_end=in_range_night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT4',
            site=self.site,
            window_start=out_of_range_run_night,
            window_end=out_of_range_run_night,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        self._make_record(
            observation_id='OBSCLAIM-OUT-OF-RANGE',
            status='COMPLETED',
            scheduled_start=datetime(2030, 1, 1, 22, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2030, 1, 2, 4, 0, tzinfo=dt_timezone.utc),
            parameters={'observed_site': 'ogg', 'observed_telescope': '2m0a'},
        )

        # Sanity: claimed_dates() itself is genuinely unbounded (its own documented contract).
        claimed, *_ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(out_of_range_run_night, claimed)

        result = campaign_gap._compute_gap(self.campaign, self.target, self.site, in_range_night, in_range_night)
        self.assertEqual(result['claimed_dates'], [in_range_night])
        self.assertNotIn(out_of_range_run_night, result['claimed_dates'])
        self.assertNotIn(out_of_range_obs_night, result['observation_claimed_dates'])
        # gap_dates must still be computed from the UNBOUNDED claimed set -- the
        # out-of-range run does not resurrect the in-range night as a "gap".
        self.assertEqual(result['gap_dates'], [])


@override_settings(CACHES=TEST_CACHES)
class TestGapAnalysisView(TestCase):
    """Integration tests for CampaignGapAnalysisView (GAP-02): the fast table view never
    triggers computation (D-09), a cache hit skips recomputation (D-10), out-of-scope
    target/site pks are rejected server-side (T-17-01/Pitfall 3), and a single-target
    campaign auto-selects its sole target (D-12).
    """

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='097',
            name='Wise Observatory',
            short_name='Wise',
            lon=34.7631,
            lat=30.5958,
            altitude=875.0,
            timezone='Asia/Jerusalem',
        )
        cls.other_site = Observatory.objects.create(
            obscode='I33',
            name='SOAR Cerro Pachon',
            short_name='SOAR',
            lon=-70.7342,
            lat=-30.2379,
            altitude=2738.0,
            timezone='America/Santiago',
        )

        # Single-target campaign: gap_analysis_available is True (has a target + an approved
        # run with a resolved site) -- used for the cache-hit and auto-select tests.
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign = TargetList.objects.create(name='Single-target Campaign')
        cls.campaign.targets.add(cls.target)
        CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Wise/LAST',
            site=cls.site,
            window_start=date(2026, 6, 1),
            window_end=date(2026, 6, 1),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )

        # Multi-target campaign, with its own used site -- used for the IDOR tests.
        cls.target_a = NonSiderealTargetFactory.create()
        cls.target_b = NonSiderealTargetFactory.create()
        cls.multi_campaign = TargetList.objects.create(name='Multi-target Campaign')
        cls.multi_campaign.targets.add(cls.target_a, cls.target_b)
        CampaignRun.objects.create(
            campaign=cls.multi_campaign,
            telescope_instrument='Wise/LAST-2',
            site=cls.site,
            window_start=date(2026, 6, 2),
            window_end=date(2026, 6, 2),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )

        # A wholly separate campaign -- its target and site are never used by either
        # campaign above (T-17-01/Pitfall 3 fixtures for the IDOR tests).
        cls.foreign_target = NonSiderealTargetFactory.create()
        cls.foreign_campaign = TargetList.objects.create(name='Foreign Campaign')
        cls.foreign_campaign.targets.add(cls.foreign_target)
        CampaignRun.objects.create(
            campaign=cls.foreign_campaign,
            telescope_instrument='SOAR/GHTS',
            site=cls.other_site,
            window_start=date(2026, 6, 3),
            window_end=date(2026, 6, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )

    def setUp(self):
        cache.clear()

    def test_table_view_does_not_trigger_computation(self):
        table_url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})
        with mock.patch('solsys_code.campaign_views.get_or_compute_gap') as mocked_gap:
            response = self.client.get(table_url)
        self.assertEqual(response.status_code, 200)
        mocked_gap.assert_not_called()

    def test_cache_hit_skips_recomputation(self):
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': self.campaign.pk})
        end_date = date.today() + timedelta(days=1)
        params = {'site': self.site.pk, 'end_date': end_date.isoformat()}

        # Mock sun_event (rather than the whole computation) so get_or_compute_gap's real
        # cache-or-compute logic is genuinely exercised -- a fixed 2-day window (today,
        # today+1) means exactly 2 sun_event calls total across both requests if (and only
        # if) the second request is served entirely from cache.
        with mock.patch('solsys_code.campaign_gap.sun_event', return_value=None) as mocked_sun_event:
            response1 = self.client.get(gap_url, params)
            response2 = self.client.get(gap_url, params)

        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(mocked_sun_event.call_count, 2)
        self.assertEqual(response1.context['result']['computed_at'], response2.context['result']['computed_at'])

    def test_rejects_out_of_scope_target_and_site(self):
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': self.multi_campaign.pk})

        with mock.patch('solsys_code.campaign_views.get_or_compute_gap') as mocked_gap:
            response_bad_target = self.client.get(gap_url, {'target': self.foreign_target.pk, 'site': self.site.pk})
            response_bad_site = self.client.get(gap_url, {'target': self.target_a.pk, 'site': self.other_site.pk})

        self.assertEqual(response_bad_target.status_code, 400)
        self.assertEqual(response_bad_site.status_code, 400)
        mocked_gap.assert_not_called()

    def test_single_target_autoselects(self):
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': self.campaign.pk})
        fixed_result = {'gap_dates': [], 'computed_at': 'sentinel'}
        with mock.patch('solsys_code.campaign_views.get_or_compute_gap', return_value=fixed_result) as mocked_gap:
            # No target_pk submitted -- the sole campaign target must still be used (D-12).
            response = self.client.get(gap_url, {'site': self.site.pk})

        self.assertEqual(response.status_code, 200)
        mocked_gap.assert_called_once()
        called_target = mocked_gap.call_args[0][1]
        self.assertEqual(called_target, self.target)

    def test_pending_narrowing_alert_shown_for_unnarrowed_space_run(self):
        """D-09: an un-narrowed space-mission run's page shows the distinct
        pending-narrowing alert with its count. The space site has no timezone set, so
        every date in the observable-dates loop raises ValueError and is skipped as
        unknown (D-03) -- that's fine, the pending_narrowing_runs alert is driven purely
        by claimed_dates() bucketing, independent of observable_dates()."""
        space_site = Observatory.objects.create(
            obscode='274',
            name='Test Space Telescope 2',
            short_name='TST2',
            observations_type=Observatory.SATELLITE_OBSTYPE,
        )
        target = NonSiderealTargetFactory.create()
        campaign = TargetList.objects.create(name='Space Pending Campaign')
        campaign.targets.add(target)
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='HST/WFC3',
            site=space_site,
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 10),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': campaign.pk})

        response = self.client.get(gap_url, {'site': space_site.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Pending narrowing: space-mission runs')
        self.assertContains(response, '1 space-mission run(s)')
        self.assertContains(response, "haven't narrowed to a")


@override_settings(CACHES=TEST_CACHES)
class TestGapAnalysisButton(TestCase):
    """Integration tests for the 'Show Coverage Gaps' button's D-14 gating on the per-campaign
    table page (GAP-02): enabled + linked when gap_analysis_available(), disabled with the
    explanatory helper text otherwise -- proven at the rendered-template level, not just view
    context (17-03-PLAN.md Task 2).
    """

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='268',
            name='Las Campanas (Magellan-Clay)',
            short_name='Magellan-Clay',
            lon=-70.6926,
            lat=-29.0146,
            altitude=2402.0,
            timezone='America/Santiago',
        )

    def setUp(self):
        cache.clear()

    def test_button_enabled_with_target_and_resolved_site(self):
        target = NonSiderealTargetFactory.create()
        campaign = TargetList.objects.create(name='Enabled Campaign')
        campaign.targets.add(target)
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='Magellan-Clay/IMACS',
            site=self.site,
            window_start=date(2026, 7, 1),
            window_end=date(2026, 7, 1),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        table_url = reverse('campaigns:table', kwargs={'pk': campaign.pk})
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': campaign.pk})

        response = self.client.get(table_url)

        self.assertContains(response, 'Show Coverage Gaps')
        self.assertContains(response, f'href="{gap_url}"')
        self.assertNotContains(
            response,
            'Coverage-gap analysis needs at least one campaign target and at least one run with a resolved site.',
        )

    def test_button_disabled_with_no_targets(self):
        campaign = TargetList.objects.create(name='No-target Campaign')
        # No .targets.add() -- zero targets, even though a resolved-site run exists, proving
        # the gate is the target count, not merely "no runs at all" (D-14).
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='Magellan-Clay/IMACS',
            site=self.site,
            window_start=date(2026, 7, 2),
            window_end=date(2026, 7, 2),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        table_url = reverse('campaigns:table', kwargs={'pk': campaign.pk})
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': campaign.pk})

        response = self.client.get(table_url)

        self.assertContains(
            response,
            'Coverage-gap analysis needs at least one campaign target and at least one run with a resolved site.',
        )
        self.assertNotContains(response, f'href="{gap_url}"')

    def test_button_disabled_with_no_resolved_site(self):
        target = NonSiderealTargetFactory.create()
        campaign = TargetList.objects.create(name='No-site Campaign')
        campaign.targets.add(target)
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='Unresolved/Site',
            site=None,
            site_raw='Some Unresolved Site',
            window_start=date(2026, 7, 3),
            window_end=date(2026, 7, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        table_url = reverse('campaigns:table', kwargs={'pk': campaign.pk})
        gap_url = reverse('campaigns:gap_analysis', kwargs={'pk': campaign.pk})

        response = self.client.get(table_url)

        self.assertContains(
            response,
            'Coverage-gap analysis needs at least one campaign target and at least one run with a resolved site.',
        )
        self.assertNotContains(response, f'href="{gap_url}"')


class TestNoHeavyEphemerisImport(TestCase):
    """GAP-01 (transitively): no phase module imports the heavy SPICE-loading ephemeris
    module or `solsys_code.views` at module scope -- mirrors the plan's own verify grep."""

    def test_campaign_gap_source_has_no_forbidden_imports(self):
        source = inspect.getsource(campaign_gap)
        for line in source.splitlines():
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith(('from ', 'import ')) and 'ephem_utils' in stripped,
                f'Forbidden ephem_utils import found: {line!r}',
            )
            self.assertNotIn('from solsys_code.views import', stripped)
