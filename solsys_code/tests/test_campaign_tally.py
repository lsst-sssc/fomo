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

import ast
import inspect
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import campaign_gap, campaign_tally, proposal_allocation, status_vocabulary
from solsys_code.allocation_projector import allocation_night_url
from solsys_code.campaign_tally import (
    TALLY_CACHE_TTL_SECONDS,
    build_tally_cache_key,
    campaign_records_version,
    campaign_rollup,
    get_or_compute_tally,
    is_unused_allocation_night,
    link_counts_for_runs,
    night_counts_for_run,
    tallies_for_runs,
    tally_for_run,
    tally_segments,
    unused_nights_for_run,
)
from solsys_code.models import CampaignRun, CampaignRunObservation, ProposalTimeAllocation
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

    def _make_alloc_event(self, run: CampaignRun, night: date, *, end_time: datetime) -> CalendarEvent:
        """Create one ``ALLOC:``-namespaced CalendarEvent for ``run``/``night`` directly --
        cheaper than a full ``project_allocation()`` sweep for a unit test that only cares
        about ``unused_nights_for_run()``'s own counting rule."""
        return CalendarEvent.objects.create(
            title=f'{run.telescope_instrument} allocation',
            start_time=end_time - timedelta(hours=8),
            end_time=end_time,
            url=allocation_night_url(run, night),
        )


class TestModuleImportGuard(TestCase):
    """Mirrors campaign_gap.py's own static import-guard test (test_campaign_gap.py
    TestNoHeavyEphemerisImport) -- checks actual import STATEMENTS, not the module's prose,
    since the module's own docstring legitimately names both modules by way of explaining
    why they must never be imported."""

    def test_module_never_imports_views_or_ephem_utils(self):
        source = inspect.getsource(campaign_tally)
        for line in source.splitlines():
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith(('from ', 'import ')) and 'ephem_utils' in stripped,
                f'Forbidden ephem_utils import found: {line!r}',
            )
            self.assertNotIn('from solsys_code.views import', stripped)

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
        self.assertEqual(
            counts[run.pk],
            {'groups': 0, 'records': 0, 'records_version': None, 'link_version': None},
        )

    def test_records_count_and_version_come_from_one_aggregate(self):
        run = self._make_run()
        record = self._link_record(run)
        counts = link_counts_for_runs([run.pk])[run.pk]
        self.assertEqual(counts['records'], 1)
        self.assertEqual(counts['records_version'], record.modified)

    def test_link_version_is_the_newest_linked_record_id(self):
        """CR-01 (37-REVIEW.md): link_version moves on a create/delete even when the
        linked record's own `modified` stamp does not -- this is what lets
        build_tally_cache_key() see a link creation/deletion that records_version alone
        would miss, since CampaignRunObservation itself carries no timestamp."""
        run = self._make_run()
        first_record = self._link_record(run)
        first_counts = link_counts_for_runs([run.pk])[run.pk]
        self.assertEqual(first_counts['link_version'], first_record.pk)

        second_record = self._link_record(run)
        second_counts = link_counts_for_runs([run.pk])[run.pk]
        self.assertEqual(second_counts['link_version'], max(first_record.pk, second_record.pk))
        self.assertNotEqual(first_counts['link_version'], second_counts['link_version'])

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

    def test_linking_an_older_untouched_record_is_reflected_with_no_clock_advance(self):
        """CR-01 (37-REVIEW.md) regression, exactly as reported: creating a
        CampaignRunObservation link to a pre-existing record whose own `modified` stamp is
        OLDER than the run's current linked-record max must still move the tally on the
        very next call. records_version alone (Max('observation_record__modified')) would
        leave the cache key -- and so the tally -- unchanged, because linking a record does
        not touch that record's own `modified` field and Max(modified) does not move when a
        record older than the current max is newly linked."""
        run = self._make_run()
        newer = self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        first = get_or_compute_tally(run)
        self.assertEqual(first['records'], 1)

        # A pre-existing record whose own `modified` stamp is forced older than `newer`'s --
        # linking it (below) never saves the record itself, so this stamp never moves.
        older_target = NonSiderealTargetFactory.create()
        older_owner = User.objects.create(username=f'obs-owner-older-{uuid4().hex[:8]}')
        older = ObservationRecord.objects.create(
            target=older_target,
            user=older_owner,
            facility='LCO',
            observation_id=f'obs-pre-existing-{uuid4().hex[:8]}',
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 9, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 9, 3, 30, tzinfo=dt_timezone.utc),
            parameters={},
        )
        ObservationRecord.objects.filter(pk=older.pk).update(modified=newer.modified - timedelta(days=1))
        older.refresh_from_db()
        self.assertLess(older.modified, newer.modified)

        CampaignRunObservation.objects.create(run=run, observation_record=older)

        second = get_or_compute_tally(run)
        self.assertEqual(second['records'], 2)

    def test_unused_count_is_live_even_on_a_cache_hit(self):
        """CR-02 (37-REVIEW.md): mirrors the same-named TestTalliesForRuns test for the
        calendar pop-up's own entry point -- get_or_compute_tally() must never serve the
        unused figure from the cached value."""
        run = self._make_run()
        past = timezone.now() - timedelta(hours=1)
        self._make_alloc_event(run, date(2026, 7, 9), end_time=past)

        first = get_or_compute_tally(run)
        self.assertEqual(first['nights_unused'], 1)

        run.run_status = CampaignRun.RunStatus.CANCELLED
        run.save(update_fields=['run_status'])

        second = get_or_compute_tally(run)
        self.assertEqual(second['nights_unused'], 0)
        self.assertEqual(second['records'], first['records'])

    def test_removing_a_non_newest_link_is_reflected_with_no_clock_advance(self):
        """CR-01 (37-REVIEW.md) regression: deleting the link to a record that is NOT the
        run's newest-linked record leaves records_version (Max(modified)) unchanged --
        records_count in the cache key is what makes the removal visible on the very next
        call instead of waiting out the TTL."""
        run = self._make_run()
        older = self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        newer = self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 11, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 11, 3, 30, tzinfo=dt_timezone.utc),
        )
        ObservationRecord.objects.filter(pk=older.pk).update(modified=newer.modified - timedelta(days=1))

        first = get_or_compute_tally(run)
        self.assertEqual(first['records'], 2)

        # Remove the OLDER link -- the run's remaining (newest) linked record's `modified`
        # stamp is unchanged, so records_version alone would not move.
        CampaignRunObservation.objects.filter(run=run, observation_record=older).delete()

        second = get_or_compute_tally(run)
        self.assertEqual(second['records'], 1)


@override_settings(CACHES=TEST_CACHES)
class TestTalliesForRuns(CampaignTallyTestBase):
    def setUp(self):
        cache.clear()

    def test_returns_one_dict_per_pk(self):
        run1 = self._make_run()
        run2 = self._make_run()
        self._link_record(run1)
        result = tallies_for_runs([run1, run2])
        self.assertEqual(set(result.keys()), {run1.pk, run2.pk})
        self.assertEqual(result[run1.pk]['records'], 1)
        self.assertEqual(result[run2.pk]['records'], 0)

    def test_unused_count_is_live_even_on_a_cache_hit(self):
        """CR-02 (37-REVIEW.md): the three unused_* keys must never come from the cached
        value -- they are recomputed on every call, live, from
        is_unused_allocation_night(), so a run_status flip to CANCELLED is visible on the
        very next call with no cache invalidation and no TTL wait, exactly matching
        unused_night_decoration()'s live read for the calendar's [U] marker (D-15)."""
        run = self._make_run()
        past = timezone.now() - timedelta(hours=1)
        self._make_alloc_event(run, date(2026, 7, 9), end_time=past)

        first = tallies_for_runs([run])[run.pk]
        self.assertEqual(first['nights_unused'], 1)

        run.run_status = CampaignRun.RunStatus.CANCELLED
        run.save(update_fields=['run_status'])

        # No linked ObservationRecord changed, so records_version/records/link_version --
        # and so the cache key -- are unchanged: this is a cache hit for the other keys.
        second = tallies_for_runs([run])[run.pk]
        self.assertEqual(second['nights_unused'], 0)
        self.assertEqual(second['records'], first['records'])

    def test_never_calls_tally_for_run_in_a_row_loop(self):
        """Source-level contract check (D-08's "never a per-row loop"): no executable line
        of tallies_for_runs() calls tally_for_run(). Checked line-by-line, skipping the
        function's own docstring, so prose mentioning tally_for_run() by name (to state the
        very rule this test pins) cannot trip the check -- mirrors 37-03-SUMMARY.md's
        documented self-correction for the identical docstring-vs-grep pitfall."""
        source = inspect.getsource(tallies_for_runs)
        func_node = ast.parse(source).body[0]
        # body[0] is the docstring Expr node when one is present -- skip straight to the
        # first REAL statement's line so the docstring's own prose (which names
        # tally_for_run() by name to state this very rule) cannot trip the check.
        first_real_stmt_lineno = func_node.body[1].lineno
        code_only = '\n'.join(source.splitlines()[first_real_stmt_lineno - 1 :])
        self.assertNotIn('tally_for_run(', code_only)


class TestIsUnusedAllocationNight(TestCase):
    """D-14: staff run status always wins; only a truly-ended, truly-empty night reads
    unused. No grace period -- `end_time < now()` in UTC, exactly."""

    def test_past_night_and_ordinary_run_status_is_unused(self):
        past = timezone.now() - timedelta(hours=1)
        self.assertTrue(is_unused_allocation_night(past, CampaignRun.RunStatus.REQUESTED))

    def test_past_night_but_cancelled_run_status_is_not_unused(self):
        past = timezone.now() - timedelta(hours=1)
        self.assertFalse(is_unused_allocation_night(past, CampaignRun.RunStatus.CANCELLED))

    def test_past_night_but_weather_tech_failure_run_status_is_not_unused(self):
        past = timezone.now() - timedelta(hours=1)
        self.assertFalse(is_unused_allocation_night(past, CampaignRun.RunStatus.WEATHER_TECH_FAILURE))

    def test_future_night_is_never_unused_regardless_of_run_status(self):
        future = timezone.now() + timedelta(hours=1)
        self.assertFalse(is_unused_allocation_night(future, CampaignRun.RunStatus.REQUESTED))


class TestUnusedNightsForRun(CampaignTallyTestBase):
    def test_no_allocation_events_returns_none(self):
        run = self._make_run()
        self.assertIsNone(unused_nights_for_run(run))

    def test_one_past_still_standing_night_counts_as_one(self):
        run = self._make_run()
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=1))
        self.assertEqual(unused_nights_for_run(run), 1)

    def test_a_future_night_does_not_count_but_events_are_known(self):
        run = self._make_run()
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() + timedelta(days=1))
        self.assertEqual(unused_nights_for_run(run), 0)

    def test_cancelled_run_status_suppresses_every_past_night(self):
        run = self._make_run(run_status=CampaignRun.RunStatus.CANCELLED)
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=1))
        self.assertEqual(unused_nights_for_run(run), 0)

    def test_two_past_nights_count_both(self):
        run = self._make_run()
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=2))
        self._make_alloc_event(run, date(2026, 7, 10), end_time=timezone.now() - timedelta(days=1))
        self.assertEqual(unused_nights_for_run(run), 2)


@override_settings(CACHES=TEST_CACHES)
class TestTallyForRunUnusedWiring(CampaignTallyTestBase):
    """tally_for_run()'s three unused_* keys, wired to the D-06/D-11 rule."""

    def setUp(self):
        cache.clear()

    def test_run_with_allocation_events_uses_the_exact_count(self):
        run = self._make_run()
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=1))
        tally = tally_for_run(run)
        self.assertEqual(tally['nights_unused'], 1)
        self.assertFalse(tally['unused_is_estimate'])
        self.assertTrue(tally['unused_known'])

    def test_run_with_no_allocation_events_uses_the_proposal_estimate(self):
        run = self._make_run(proposal_code='UTX2026A-002')
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026B',
            instrument_type='1M0-SCICAM-SINISTRO',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=15.0,
            fetched_at=timezone.now(),
        )
        tally = tally_for_run(run)
        self.assertEqual(tally['nights_unused'], 3)  # floor(25/10 + 0.5) == 3
        self.assertTrue(tally['unused_is_estimate'])
        self.assertTrue(tally['unused_known'])

    def test_run_with_no_allocation_events_and_blank_proposal_code_is_unknown_not_zero(self):
        run = self._make_run(proposal_code='')
        tally = tally_for_run(run)
        self.assertIsNone(tally['nights_unused'])
        self.assertFalse(tally['unused_known'])

    def test_run_with_a_proposal_code_but_no_stored_rows_is_unknown_not_zero(self):
        run = self._make_run(proposal_code='NEVER-FETCHED-2026A-001')
        tally = tally_for_run(run)
        self.assertIsNone(tally['nights_unused'])
        self.assertFalse(tally['unused_known'])


class TestTallySegments(TestCase):
    """D-15: the table and the calendar must agree by construction -- tally_segments()
    always returns the same four segments in the same fixed order."""

    _TALLY = {
        'groups': 2,
        'records': 14,
        'nights_observed': 5,
        'nights_scheduled': 2,
        'nights_failed': 1,
        'nights_unused': 3,
        'unused_is_estimate': True,
        'unused_known': True,
    }

    def test_returns_four_segments_in_fixed_order(self):
        segments = tally_segments(self._TALLY)
        self.assertEqual([s['marker'] for s in segments], ['[O]', '[S]', '[X/F]', '[U]'])

    def test_counts_come_from_the_tally_dict(self):
        segments = tally_segments(self._TALLY)
        by_marker = {s['marker']: s for s in segments}
        self.assertEqual(by_marker['[O]']['count'], 5)
        self.assertEqual(by_marker['[S]']['count'], 2)
        self.assertEqual(by_marker['[X/F]']['count'], 1)
        self.assertEqual(by_marker['[U]']['count'], 3)

    def test_unused_segment_carries_the_estimate_and_known_flags(self):
        segments = tally_segments(self._TALLY)
        by_marker = {s['marker']: s for s in segments}
        self.assertTrue(by_marker['[U]']['is_estimate'])
        self.assertTrue(by_marker['[U]']['known'])

    def test_order_is_fixed_regardless_of_which_counts_are_zero(self):
        zero_tally = dict(self._TALLY)
        zero_tally.update(nights_observed=0, nights_scheduled=0, nights_failed=0, nights_unused=0)
        segments = tally_segments(zero_tally)
        self.assertEqual([s['marker'] for s in segments], ['[O]', '[S]', '[X/F]', '[U]'])


@override_settings(CACHES=TEST_CACHES)
class TestCampaignRollup(CampaignTallyTestBase):
    """D-10: the campaign roll-up sums only publicly visible runs, and counts each distinct
    proposal code once, not once per run."""

    def setUp(self):
        cache.clear()

    def test_empty_campaign_returns_a_well_formed_all_zero_rollup(self):
        rollup = campaign_rollup(self.campaign)
        self.assertEqual(rollup['runs'], 0)
        self.assertEqual(rollup['groups'], 0)
        self.assertEqual(rollup['records'], 0)
        self.assertEqual(rollup['nights_observed'], 0)
        self.assertEqual(rollup['nights_scheduled'], 0)
        self.assertEqual(rollup['nights_failed'], 0)
        self.assertIsNone(rollup['nights_unused'])
        self.assertFalse(rollup['unused_known'])

    def test_pending_review_run_contributes_nothing(self):
        run = self._make_run(campaign=self.campaign, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        rollup = campaign_rollup(self.campaign)
        self.assertEqual(rollup['runs'], 0)
        self.assertEqual(rollup['records'], 0)

    def test_pending_review_exclusion_is_applied_at_the_queryset_level(self):
        """The exclusion must be a queryset .exclude(), never CampaignRun.is_publicly_
        visible -- a Python property cannot be used in a filter (the model's own docstring
        note). Checked at the source level, mirroring the plan's own must-have wording."""
        source = inspect.getsource(campaign_rollup)
        self.assertIn('PENDING_REVIEW', source)
        self.assertNotIn('is_publicly_visible', source)

    def test_sums_groups_records_and_nights_across_approved_public_runs(self):
        run1 = self._make_run(campaign=self.campaign, telescope_instrument='FTN/FLOYDS')
        run2 = self._make_run(campaign=self.campaign, telescope_instrument='FTN/Muscat')
        self._link_record(
            run1,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        self._link_record(
            run2,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 11, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 11, 3, 30, tzinfo=dt_timezone.utc),
        )
        rollup = campaign_rollup(self.campaign)
        self.assertEqual(rollup['runs'], 2)
        self.assertEqual(rollup['records'], 2)
        self.assertEqual(rollup['nights_observed'], 2)

    def test_unused_estimate_is_counted_once_per_distinct_proposal_code(self):
        self._make_run(campaign=self.campaign, telescope_instrument='FTN/FLOYDS', proposal_code='SHARED-2026A-001')
        self._make_run(campaign=self.campaign, telescope_instrument='FTN/Muscat', proposal_code='SHARED-2026A-001')
        ProposalTimeAllocation.objects.create(
            proposal_code='SHARED-2026A-001',
            semester='2026A',
            instrument_type='1M0-SCICAM-SINISTRO',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=15.0,
            fetched_at=timezone.now(),
        )
        rollup = campaign_rollup(self.campaign)
        # floor(25/10 + 0.5) == 3 -- counted ONCE for the shared code, never 3+3=6.
        self.assertEqual(rollup['nights_unused'], 3)
        self.assertTrue(rollup['unused_known'])

    def test_exact_allocation_counts_are_added_alongside_the_estimate(self):
        run_exact = self._make_run(campaign=self.campaign, telescope_instrument='FTN/FLOYDS')
        self._make_alloc_event(run_exact, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=1))
        self._make_run(campaign=self.campaign, telescope_instrument='FTN/Muscat', proposal_code='SOLO-2026A-002')
        ProposalTimeAllocation.objects.create(
            proposal_code='SOLO-2026A-002',
            semester='2026A',
            instrument_type='1M0-SCICAM-SINISTRO',
            allocation_type='std',
            allocated_hours=20.0,
            used_hours=10.0,
            fetched_at=timezone.now(),
        )
        rollup = campaign_rollup(self.campaign)
        # 1 exact still-standing night + floor(10/10 + 0.5) == 1 estimated night.
        self.assertEqual(rollup['nights_unused'], 2)
        self.assertTrue(rollup['unused_known'])


class TestCampaignRecordsVersion(CampaignTallyTestBase):
    def test_no_runs_returns_none(self):
        self.assertIsNone(campaign_records_version(self.campaign))

    def test_returns_the_newest_stamp_across_the_campaigns_runs(self):
        run = self._make_run(campaign=self.campaign)
        record = self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        self.assertEqual(campaign_records_version(self.campaign), record.modified)

    def test_pending_review_run_is_excluded(self):
        run = self._make_run(campaign=self.campaign, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        self.assertIsNone(campaign_records_version(self.campaign))


@override_settings(CACHES=TEST_CACHES)
class TestTallyNeverWritesRunStatus(CampaignTallyTestBase):
    """TALLY-03's two-part guard: a behavioural before/after snapshot across every tally
    entry point and both cache orders, plus a syntax-tree assertion that no module in the
    computation path assigns to the field."""

    def setUp(self):
        cache.clear()

    def _exercise_every_entry_point(self, run: CampaignRun) -> None:
        tally_for_run(run)
        get_or_compute_tally(run)  # cache miss
        get_or_compute_tally(run)  # cache hit
        cache.clear()
        get_or_compute_tally(run)  # miss again (hit-then-miss ordering)
        tallies_for_runs([run])
        unused_nights_for_run(run)
        campaign_rollup(self.campaign)

    def test_run_status_unchanged_for_a_fully_observed_run(self):
        run = self._make_run(campaign=self.campaign, run_status=CampaignRun.RunStatus.OBSERVED)
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        before = CampaignRun.objects.get(pk=run.pk).run_status
        self._exercise_every_entry_point(run)
        after = CampaignRun.objects.get(pk=run.pk).run_status
        self.assertEqual(before, after)

    def test_run_status_unchanged_for_a_run_with_no_linked_records(self):
        run = self._make_run(campaign=self.campaign)
        before = CampaignRun.objects.get(pk=run.pk).run_status
        self._exercise_every_entry_point(run)
        after = CampaignRun.objects.get(pk=run.pk).run_status
        self.assertEqual(before, after)

    def test_no_computation_path_module_assigns_to_run_status_attribute(self):
        """Static AST check over the four named modules -- a docstring or comment
        mentioning the field cannot make this pass or fail spuriously, since only actual
        ast.Assign/ast.Call nodes are inspected, never the source text."""
        modules = (campaign_tally, status_vocabulary, proposal_allocation, campaign_gap)
        for module in modules:
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.assertFalse(
                            isinstance(target, ast.Attribute) and target.attr == 'run_status',
                            f'{module.__name__} assigns to .run_status at line {node.lineno}',
                        )
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update':
                    for keyword in node.keywords:
                        self.assertNotEqual(
                            keyword.arg,
                            'run_status',
                            f'{module.__name__} calls update(run_status=...) at line {node.lineno}',
                        )
