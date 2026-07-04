"""Unit tests for the pure-computation coverage-gap module (GAP-02) + import guard (GAP-01).

Depends only on `campaign_gap.py`, which itself depends only on `telescope_runs.sun_event` for
ephemerides -- never the heavy SPICE-loading ephemeris/views module. This module's own static
import-guard test mirrors the grep this file's plan verification step also runs, so the two stay
in agreement.

Always uses `tom_targets.tests.factories.NonSiderealTargetFactory` for any Target fixture --
never `SiderealTargetFactory` (CLAUDE.md: FOMO is exclusively for Solar System / non-sidereal
targets).
"""

import inspect
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from django.core.cache import cache
from django.test import TestCase, override_settings
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
)
from solsys_code.models import CampaignRun
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
    """D-05/D-06/D-07/D-08: claimed-date derivation, exclusions, and undated flagging."""

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

    def test_approved_completed_run_claimed_via_obs_date(self):
        obs_date = date(2026, 7, 10)
        self._make_run(obs_date=obs_date, telescope_instrument='A')
        claimed, undated, unattributed = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(obs_date, claimed)
        self.assertEqual(undated, [])
        self.assertEqual(unattributed, [])

    def test_cancelled_run_not_claimed(self):
        obs_date = date(2026, 7, 11)
        self._make_run(obs_date=obs_date, telescope_instrument='B', run_status=CampaignRun.RunStatus.CANCELLED)
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(obs_date, claimed)

    def test_pending_review_run_not_claimed(self):
        obs_date = date(2026, 7, 12)
        self._make_run(
            obs_date=obs_date,
            telescope_instrument='C',
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(obs_date, claimed)

    def test_ut_start_only_keys_to_site_local_observing_night(self):
        # 2026-07-14 02:00 UTC = 2026-07-13 22:00 America/Santiago (UTC-4) -- evening,
        # so the observing-night label is the local date itself, not the raw UTC date.
        ut_start = datetime(2026, 7, 14, 2, 0, tzinfo=timezone.utc)
        self._make_run(obs_date=None, ut_start=ut_start, telescope_instrument='D')
        claimed, undated, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(date(2026, 7, 13), claimed)
        self.assertNotIn(date(2026, 7, 14), claimed)
        self.assertEqual(undated, [])

    def test_undated_runs_flagged(self):
        run = self._make_run(obs_date=None, ut_start=None, telescope_instrument='E')
        claimed, undated, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(run, undated)
        self.assertNotIn(None, claimed)
        # No date should have been added on behalf of this run.
        self.assertEqual(len(claimed), 0)

    def test_different_site_not_claimed(self):
        obs_date = date(2026, 7, 15)
        self._make_run(obs_date=obs_date, telescope_instrument='F', site=self.other_site)
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(obs_date, claimed)


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
        obs_date = date(2026, 7, 20)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/Sinistro',
            site=self.site,
            target=None,
            obs_date=obs_date,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed_a, _, unattributed_a = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, unattributed_b = claimed_dates(self.campaign, self.target_b, self.site)
        self.assertNotIn(obs_date, claimed_a)
        self.assertNotIn(obs_date, claimed_b)
        self.assertEqual(len(unattributed_a), 1)
        self.assertEqual(len(unattributed_b), 1)

    def test_target_specific_run_claimed_only_for_its_own_target(self):
        obs_date = date(2026, 7, 21)
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/Sinistro-2',
            site=self.site,
            target=self.target_a,
            obs_date=obs_date,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
        claimed_a, _, _ = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, _ = claimed_dates(self.campaign, self.target_b, self.site)
        self.assertIn(obs_date, claimed_a)
        self.assertNotIn(obs_date, claimed_b)


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
