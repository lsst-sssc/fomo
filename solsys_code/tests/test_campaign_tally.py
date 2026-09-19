"""Unit tests for the public per-run/per-campaign tally module (TALLY-01/02/03, UNUSED-01).

``campaign_tally.py`` depends on ``status_vocabulary`` (the classifier), ``telescope_runs``
(``observing_night()``), ``calendar_utils`` (``record_time_window()``), ``observation_projector``
(``facility_for()``), ``allocation_projector`` (``allocation_events()``) and
``proposal_allocation`` (``estimated_unused_nights()``) -- never ``solsys_code.views`` or
``solsys_code.ephem_utils``. This module's own static import-guard test mirrors the grep this
plan's verify step also runs, so the two stay in agreement.

Always uses ``tom_targets.tests.factories.NonSiderealTargetFactory`` for any Target fixture --
never ``SiderealTargetFactory`` (CLAUDE.md: FOMO is exclusively for Solar System / non-sidereal
targets).
"""

import inspect
from datetime import date, datetime
from datetime import timezone as dt_timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import campaign_tally
from solsys_code.campaign_tally import (
    TALLY_CACHE_TTL_SECONDS,
    build_tally_cache_key,
    get_or_compute_tally,
    link_counts_for_runs,
    night_counts_for_run,
    tallies_for_runs,
    tally_for_run,
)
from solsys_code.models import CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory

TEST_CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}


class CampaignTallyTestBase(TestCase):
    """Shared fixture: one resolvable ground Observatory (Haleakala/FTN -- fixed UTC-10, no
    DST, so the 02:00-local boundary test's arithmetic is unambiguous), one campaign, a
    CampaignRun factory helper and a linked-ObservationRecord factory helper. Mirrors
    ``AllocationProjectorTestBase`` in ``test_allocation_projector.py``.
    """

    @classmethod
    def setUpTestData(cls):
        cls.site = Observatory.objects.create(
            obscode='F65',
            name='Haleakala Observatory',
            short_name='FTN',
            lat=20.7069,
            lon=-156.2570,
            altitude=3055,
            timezone='Pacific/Honolulu',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.site_zone = ZoneInfo(cls.site.timezone)
        cls.campaign = TargetList.objects.create(name='Tally Test Campaign')

    def _make_run(self, **overrides) -> CampaignRun:
        kwargs = {
            'campaign': None,
            'source': CampaignRun.Source.CLASSICAL_FILE,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'telescope_instrument': 'FTN/FLOYDS',
            'site': self.site,
            'site_raw': 'F65',
            'window_start': date(2026, 7, 9),
            'window_end': date(2026, 7, 11),
            'observation_details': 'Photometric monitoring',
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _link_record(self, run: CampaignRun, **overrides) -> ObservationRecord:
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'obs-owner-{uuid4().hex[:8]}')
        kwargs = {
            'target': target,
            'user': owner,
            'facility': 'LCO',
            'observation_id': f'obs-{uuid4().hex[:8]}',
            'status': 'COMPLETED',
            'scheduled_start': None,
            'scheduled_end': None,
            'parameters': {},
        }
        kwargs.update(overrides)
        record = ObservationRecord.objects.create(**kwargs)
        CampaignRunObservation.objects.create(run=run, observation_record=record)
        return record


class TestModuleImportGuard(TestCase):
    """Mirrors campaign_gap.py's own static import-guard test."""

    def test_module_never_imports_views_or_ephem_utils(self):
        src = inspect.getsource(campaign_tally)
        self.assertNotIn('solsys_code.views', src)
        self.assertNotIn('ephem_utils', src)

    def test_module_states_tally_03_invariant_in_words(self):
        doc = campaign_tally.__doc__ or ''
        self.assertIn('run_status', doc)


class TestBuildTallyCacheKey(TestCase):
    """A key that moves with the record change stamp is what makes TALLY-01's "updating as
    the projector narrows" true rather than aspirational."""

    def test_different_stamps_produce_different_keys(self):
        a = build_tally_cache_key(1, datetime(2026, 1, 1, tzinfo=dt_timezone.utc))
        b = build_tally_cache_key(1, datetime(2026, 1, 2, tzinfo=dt_timezone.utc))
        self.assertNotEqual(a, b)

    def test_same_stamp_produces_the_same_key(self):
        stamp = datetime(2026, 1, 1, tzinfo=dt_timezone.utc)
        self.assertEqual(build_tally_cache_key(1, stamp), build_tally_cache_key(1, stamp))

    def test_none_stamp_differs_from_a_real_stamp(self):
        stamp = datetime(2026, 1, 1, tzinfo=dt_timezone.utc)
        self.assertNotEqual(build_tally_cache_key(1, None), build_tally_cache_key(1, stamp))

    def test_none_stamp_is_stable_across_calls(self):
        self.assertEqual(build_tally_cache_key(1, None), build_tally_cache_key(1, None))


class TestLinkCountsForRuns(CampaignTallyTestBase):
    def test_run_with_no_links_reports_zero_and_none_version(self):
        run = self._make_run()
        counts = link_counts_for_runs([run.pk])
        self.assertEqual(counts[run.pk], {'groups': 0, 'records': 0, 'records_version': None})

    def test_records_count_and_version_come_from_one_aggregate(self):
        run = self._make_run()
        record = self._link_record(run)
        counts = link_counts_for_runs([run.pk])[run.pk]
        self.assertEqual(counts['records'], 1)
        self.assertEqual(counts['records_version'], record.modified)

    def test_groups_are_counted_distinct(self):
        run = self._make_run()
        record = self._link_record(run)
        group = ObservationGroup.objects.create(name='tally-group')
        group.observation_records.add(record)
        counts = link_counts_for_runs([run.pk])[run.pk]
        self.assertEqual(counts['groups'], 1)

    def test_whole_set_computed_in_two_queries_never_per_row(self):
        run1 = self._make_run()
        run2 = self._make_run()
        self._link_record(run1)
        self._link_record(run2)
        with self.assertNumQueries(2):
            link_counts_for_runs([run1.pk, run2.pk])


class TestNightCountsForRun(CampaignTallyTestBase):
    def test_no_linked_records_returns_all_zero(self):
        run = self._make_run()
        self.assertEqual(
            night_counts_for_run(run),
            {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0},
        )

    def test_site_unset_returns_all_zero_no_exception(self):
        run = self._make_run(site=None, site_raw='')
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 4, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 10, 0, tzinfo=dt_timezone.utc),
        )
        self.assertEqual(
            night_counts_for_run(run),
            {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0},
        )

    def test_two_observed_records_same_site_local_night_count_once(self):
        run = self._make_run()
        # Both fall on the local evening of 2026-07-09 (Honolulu is fixed UTC-10, no DST).
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 5, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 5, 30, tzinfo=dt_timezone.utc),
        )
        counts = night_counts_for_run(run)
        self.assertEqual(counts['nights_observed'], 1)

    def test_early_morning_local_start_counts_on_the_previous_date(self):
        run = self._make_run()
        # 02:00 local on 2026-07-10 (Honolulu, UTC-10) == 12:00 UTC same date -- must land
        # on the PREVIOUS date's night (2026-07-09), the noon-anchor rule
        # observing_night() already implements (never re-derived here).
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 12, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 12, 30, tzinfo=dt_timezone.utc),
        )
        # Assert directly against observing_night() itself, so this test also pins that no
        # local re-derivation of the night rule has crept into night_counts_for_run().
        from solsys_code.telescope_runs import observing_night

        expected_night = observing_night(datetime(2026, 7, 10, 12, 0, tzinfo=dt_timezone.utc), self.site_zone)
        self.assertEqual(expected_night, date(2026, 7, 9))
        counts = night_counts_for_run(run)
        self.assertEqual(counts['nights_observed'], 1)

    def test_scheduled_record_counts_as_scheduled_not_observed(self):
        run = self._make_run()
        self._link_record(
            run,
            status='PENDING',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        counts = night_counts_for_run(run)
        self.assertEqual(counts, {'nights_observed': 0, 'nights_scheduled': 1, 'nights_failed': 0})

    def test_window_expired_cancelled_and_failed_all_count_as_failed(self):
        run = self._make_run()
        for status in ('WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED'):
            self._link_record(
                run,
                status=status,
                scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
                scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
            )
        counts = night_counts_for_run(run)
        # All three land on the same night -- de-duplicated to one failed night.
        self.assertEqual(counts, {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 1})

    def test_queued_record_with_no_block_contributes_to_no_night_set(self):
        run = self._make_run()
        self._link_record(run, status='PENDING', scheduled_start=None, scheduled_end=None)
        self.assertEqual(
            night_counts_for_run(run),
            {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0},
        )

    def test_inconsistent_record_contributes_to_no_night_set(self):
        run = self._make_run()
        self._link_record(
            run,
            status='PENDING',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=None,
        )
        self.assertEqual(
            night_counts_for_run(run),
            {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0},
        )

    def test_record_time_window_raising_is_skipped_never_aborts(self):
        run = self._make_run()
        # Both fields None with status COMPLETED means record_time_window() falls into the
        # parameters['start']/['end'] branch and raises KeyError -- must be skipped, not
        # raised out of night_counts_for_run().
        self._link_record(run, status='COMPLETED', scheduled_start=None, scheduled_end=None, parameters={})
        self.assertEqual(
            night_counts_for_run(run),
            {'nights_observed': 0, 'nights_scheduled': 0, 'nights_failed': 0},
        )


class TestTallyForRun(CampaignTallyTestBase):
    def test_zero_linked_records_produces_all_zero_and_unused_unknown(self):
        run = self._make_run()
        tally = tally_for_run(run)
        self.assertEqual(
            tally,
            {
                'groups': 0,
                'records': 0,
                'nights_observed': 0,
                'nights_scheduled': 0,
                'nights_failed': 0,
                'nights_unused': None,
                'unused_is_estimate': True,
                'unused_known': False,
            },
        )

    def test_carries_the_eight_required_keys(self):
        run = self._make_run()
        tally = tally_for_run(run)
        self.assertEqual(
            set(tally.keys()),
            {
                'groups',
                'records',
                'nights_observed',
                'nights_scheduled',
                'nights_failed',
                'nights_unused',
                'unused_is_estimate',
                'unused_known',
            },
        )


@override_settings(CACHES=TEST_CACHES)
class TestGetOrComputeTallyFreshness(CampaignTallyTestBase):
    """The regression guard for "updating as the projector narrows" (TALLY-01)."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()

    def test_ttl_is_one_hour(self):
        self.assertEqual(TALLY_CACHE_TTL_SECONDS, 3600)

    def test_saving_a_linked_record_is_reflected_with_no_clock_advance(self):
        run = self._make_run()
        first = get_or_compute_tally(run)
        self.assertEqual(first['records'], 0)

        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )

        second = get_or_compute_tally(run)
        self.assertEqual(second['records'], 1)
        self.assertEqual(second['nights_observed'], 1)

    def test_cache_hit_returns_the_previously_computed_dict_unchanged(self):
        run = self._make_run()
        record = self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        version = link_counts_for_runs([run.pk])[run.pk]['records_version']
        self.assertEqual(version, record.modified)
        first = get_or_compute_tally(run, records_version=version)
        second = get_or_compute_tally(run, records_version=version)
        self.assertEqual(first, second)


@override_settings(CACHES=TEST_CACHES)
class TestTalliesForRuns(CampaignTallyTestBase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()

    def test_returns_one_dict_per_pk(self):
        run1 = self._make_run()
        run2 = self._make_run()
        self._link_record(run1)
        result = tallies_for_runs([run1, run2])
        self.assertEqual(set(result.keys()), {run1.pk, run2.pk})
        self.assertEqual(result[run1.pk]['records'], 1)
        self.assertEqual(result[run2.pk]['records'], 0)

    def test_never_calls_tally_for_run_in_a_row_loop(self):
        """Docstring-level contract check: the function's own source never calls
        tally_for_run() at all (D-08's "never a per-row loop")."""
        src = inspect.getsource(tallies_for_runs)
        self.assertNotIn('tally_for_run(', src)
