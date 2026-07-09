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
from datetime import date, timedelta
from unittest import mock

from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
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
        claimed, undated, unattributed = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(night, claimed)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(undated, [])
        self.assertEqual(unattributed, [])

    def test_range_run_claims_every_date_in_window(self):
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 4)
        self._make_run(window_start=window_start, window_end=window_end, telescope_instrument='RANGE')
        claimed, undated, _ = claimed_dates(self.campaign, self.target, self.site)
        expected = {
            date(2026, 8, 1),
            date(2026, 8, 2),
            date(2026, 8, 3),
            date(2026, 8, 4),
        }
        self.assertEqual(claimed, expected)
        self.assertEqual(undated, [])

    def test_cancelled_run_not_claimed(self):
        night = date(2026, 7, 11)
        self._make_run(
            window_start=night, window_end=night, telescope_instrument='B', run_status=CampaignRun.RunStatus.CANCELLED
        )
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)

    def test_pending_review_run_not_claimed(self):
        night = date(2026, 7, 12)
        self._make_run(
            window_start=night,
            window_end=night,
            telescope_instrument='C',
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)

    def test_undated_runs_flagged(self):
        run = self._make_run(window_start=None, window_end=None, telescope_instrument='E')
        claimed, undated, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertIn(run, undated)
        self.assertNotIn(None, claimed)
        # No date should have been added on behalf of this run.
        self.assertEqual(len(claimed), 0)

    def test_different_site_not_claimed(self):
        night = date(2026, 7, 15)
        self._make_run(window_start=night, window_end=night, telescope_instrument='F', site=self.other_site)
        claimed, _, _ = claimed_dates(self.campaign, self.target, self.site)
        self.assertNotIn(night, claimed)


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
        claimed_a, _, unattributed_a = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, unattributed_b = claimed_dates(self.campaign, self.target_b, self.site)
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
        claimed_a, _, _ = claimed_dates(self.campaign, self.target_a, self.site)
        claimed_b, _, _ = claimed_dates(self.campaign, self.target_b, self.site)
        self.assertIn(night, claimed_a)
        self.assertNotIn(night, claimed_b)


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
