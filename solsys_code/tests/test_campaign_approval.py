"""Tests for the staff-facing approval-queue write path (SUBMIT-03 / CAL-01/02/03 / D-01/D-02).

Covers: staff-only gating on both the approval-queue GET and the decision-endpoint POST
(never a soft-filter -- a redirect, never 200-with-pending-content, per 16-RESEARCH.md Pitfall
7); the atomic conditional approve/reject transition and its proven double-approve no-op
(SUBMIT-03); the D-06 hybrid CAMPAIGN:{pk} CalendarEvent projection that fires only for a
single concrete night (window_start == window_end) with a resolved site -- a dip-corrected
sun_event() window for a ground site, a midnight-UTC placeholder for a space site
(CAL-01/CAL-02); no duplicate event and no ``modified`` churn on re-approve (CAL-03); and the
reject path (no event created).

Uses ``TargetList.objects.create(...)`` for the campaign container and plain
``CampaignRun.objects.create(...)`` fixtures. This module never fixtures an individual
``tom_targets.models.Target`` at all (CampaignRun.target is left unset throughout), so
CLAUDE.md's non-sidereal-only target-factory convention doesn't even arise here.
"""

from datetime import date, datetime, timezone
from html.parser import HTMLParser
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code import campaign_utils
from solsys_code.campaign_tables import ApprovalQueueTable, CampaignRunTable
from solsys_code.campaign_utils import NEEDS_REVIEW_NAME_PREFIX, resolve_site
from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher
from solsys_code.telescope_runs import sun_event

CONTACT_PERSON = 'Jane Coordinator'
CONTACT_EMAIL = 'jane@example.org'

# debug/site-search-degraded-pool-recurrence (bug #3): settings.CACHES is a FileBasedCache at
# tempfile.gettempdir() (/tmp), SHARED across processes -- and Django does NOT swap the cache
# backend for tests the way it swaps the database. Without this override, every cache.clear()
# in setUp/tearDown and every real build_site_candidates() call in the tests below reads,
# writes, and WIPES the same cache key ('mpc_obscode_candidates') the dev runserver serves
# site-search from. That is exactly the bug #3 regression: running this suite to verify a fix
# silently wiped the runserver's warmed ~5,700-entry MPC candidate pool, so the live
# site-search reverted to "No matches" until the next successful cold rebuild. Pinning the
# cache-touching test classes to an isolated in-memory LocMemCache keeps ALL test cache traffic
# off the shared file cache the runserver depends on.
ISOLATED_TEST_CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'campaign-tests-isolated',
    }
}

# Wave-0 fixture (SITE-01, RESEARCH.md Pattern 2/3): a small, representative slice of the
# real MPC bulk obscodes response shape ({obscode: {name_utf8, short_name, old_names,
# observations_type, longitude, ...}}). Deliberately does NOT include 'DCT' as a candidate
# string anywhere -- Pitfall 2 confirmed live that difflib cannot bridge the acronym/
# nickname gap even against the full 5,636-string real pool, so 'DCT' must stay a genuine
# no-match case here too (G37's real MPC name is spelled out in full, never abbreviated).
BULK_MPC_FIXTURE = {
    'C65': {
        'name_utf8': 'Observatori Astronòmic del Montsec',
        'short_name': 'OAdM',
        'old_names': None,
        'observations_type': 'fixed',
        'longitude': 1.1937,
    },
    '250': {
        'name_utf8': 'Hubble Space Telescope',
        'short_name': 'HST',
        'old_names': None,
        'observations_type': 'satellite',
        'longitude': None,
    },
    'G37': {
        'name_utf8': 'Lowell Discovery Telescope',
        'short_name': 'Lowell Discovery Telescope',
        'old_names': None,
        'observations_type': 'fixed',
        'longitude': -111.4223,
    },
    'W89': {
        'name_utf8': 'Siding Spring Observatory',
        'short_name': 'SSO',
        'old_names': None,
        'observations_type': 'fixed',
        'longitude': 149.0,
    },
    'F65': {
        'name_utf8': 'Faulkes Telescope South',
        'short_name': 'FTS',
        'old_names': None,
        'observations_type': 'fixed',
        'longitude': 149.0644,
    },
    'X09': {
        'name_utf8': 'Deep Random Survey, Rio Hurtado',
        'short_name': 'Deep Random Survey',
        'old_names': None,
        'observations_type': 'fixed',
        'longitude': -70.9,
    },
}


@override_settings(CACHES=ISOLATED_TEST_CACHES)
class CampaignApprovalTestBase(TestCase):
    """Shared fixture: one campaign, one staff user, one non-staff user.

    Cache-isolated (bug #3, debug/site-search-degraded-pool-recurrence): the decide-endpoint
    POST path resolves a site_selection via ``selection_to_obscode()`` ->
    ``build_site_candidates()``, which reads/writes the ``mpc_obscode_candidates`` cache key.
    The ``@override_settings(CACHES=ISOLATED_TEST_CACHES)`` here (inherited by every subclass)
    keeps that traffic on an in-memory LocMemCache instead of the shared /tmp file cache the
    dev runserver serves live site-search from.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)
        cls.non_staff_user = User.objects.create_user(username='regularobserver', password='pw', is_staff=False)

    def _make_pending_run(self, **overrides):
        """Create a PENDING_REVIEW CampaignRun; kwargs override the default field set."""
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site_raw': 'F65',
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 1),
            'observation_details': 'Photometric monitoring',
            'contact_person': CONTACT_PERSON,
            'contact_email': CONTACT_EMAIL,
            'approval_status': CampaignRun.ApprovalStatus.PENDING_REVIEW,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)


class TestStaffGating(CampaignApprovalTestBase):
    """T-16-03: anonymous/non-staff access must redirect, never render pending content."""

    def test_anonymous_get_approval_queue_redirects(self):
        run = self._make_pending_run()
        response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CampaignRun.objects.get(pk=run.pk).approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)

    def test_non_staff_get_approval_queue_redirects(self):
        self.client.login(username='regularobserver', password='pw')
        response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertEqual(response.status_code, 302)

    def test_anonymous_post_decide_redirects_and_makes_no_change(self):
        run = self._make_pending_run()
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)

    def test_non_staff_post_decide_redirects_and_makes_no_change(self):
        run = self._make_pending_run()
        self.client.login(username='regularobserver', password='pw')
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)

    def test_staff_get_approval_queue_succeeds(self):
        self._make_pending_run()
        self.client.login(username='staffcoordinator', password='pw')
        response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Approval Queue')


class TestApproval(CampaignApprovalTestBase):
    """SUBMIT-03: atomic approve/reject and the proven double-approve no-op."""

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        # D-06: a Tier-1-resolvable ground Observatory for the default fixture's site_raw
        # ('F65') so approve's calendar projection (which now requires a resolved run.site)
        # succeeds deterministically here, without a live MPC API call.
        Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_double_approve_is_noop(self):
        run = self._make_pending_run()
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)

        # Second approve POST on the already-approved row must be a proven no-op.
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)

    def test_second_approve_surfaces_already_decided_warning(self):
        run = self._make_pending_run()
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'}, follow=True
        )
        messages = [str(m) for m in response.context['messages']]
        self.assertIn('This run was already decided by someone else.', messages)

    def test_reject_path_sets_rejected_and_creates_no_event(self):
        run = self._make_pending_run()
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'reject'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.REJECTED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)

    def test_invalid_action_returns_bad_request(self):
        run = self._make_pending_run()
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'bogus'})
        self.assertEqual(response.status_code, 400)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)

    def test_approving_already_resolved_site_does_not_call_resolve_site(self):
        """SITE-03/D-06: a pre-set run.site is trusted and never re-resolved on approve."""
        observatory = Observatory.objects.get(obscode='F65')
        run = self._make_pending_run(site=observatory, site_needs_review=False)
        with patch('solsys_code.campaign_views.resolve_site') as mock_resolve_site:
            response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        mock_resolve_site.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site_id, observatory.pk)

    def test_projection_failure_reverts_site_stays_set_second_approve_skips_resolve_site(self):
        """RESEARCH.md Pitfall 3 regression: a projection failure reverts approval_status to
        PENDING_REVIEW while leaving run.site set (D-06's clobber-fix guard); a second approve
        POST must not re-call resolve_site() (the pre-fix bug re-ran the MPC fetch here)."""
        run = self._make_pending_run()
        with patch(
            'solsys_code.campaign_views.insert_or_create_calendar_event',
            side_effect=RuntimeError('boom'),
        ):
            response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self.assertIsNotNone(run.site)
        resolved_site_pk = run.site.pk

        with patch('solsys_code.campaign_views.resolve_site') as mock_resolve_site:
            response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        mock_resolve_site.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site.pk, resolved_site_pk)

    def test_oversized_site_selection_is_flagged_with_no_network_call_or_fabrication(self):
        """T-21-04: an oversized site_selection is flagged by resolve_site's existing
        _MAX_OBSCODE_LEN guard -- no tier attempted, no network call, no fabricated Observatory."""
        run = self._make_pending_run()
        oversized = 'X' * (Observatory._meta.get_field('obscode').max_length + 1)
        with patch('solsys_code.campaign_utils.MPCObscodeFetcher.query') as mock_query:
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'approve', 'site_selection': oversized},
            )
        self.assertEqual(response.status_code, 302)
        mock_query.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(Observatory.objects.filter(obscode=oversized).count(), 0)


class TestCalendarProjection(CampaignApprovalTestBase):
    """D-06/CAL-01/CAL-02: approving a single-night run with a resolved site projects a
    CAMPAIGN:{pk} event -- a dip-corrected sun_event() window for a ground site, a
    midnight-UTC placeholder for a space site. A range, TBD run, missing
    telescope_instrument, or a sun_event() ValueError all project nothing (the last of
    these without reverting the already-committed approval).
    """

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        # Tier-1-resolvable so approve's site resolution never needs a live MPC API call.
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

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_approve_single_night_ground_run_creates_dip_corrected_calendar_event(self):
        run = self._make_pending_run()
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        event = CalendarEvent.objects.get(url=f'CAMPAIGN:{run.pk}')
        expected_sunset, expected_sunrise = sun_event(self.ground_site, run.window_start, kind='sun')
        self.assertEqual(event.start_time, expected_sunset.to_datetime(timezone=timezone.utc).replace(microsecond=0))
        self.assertEqual(event.end_time, expected_sunrise.to_datetime(timezone=timezone.utc).replace(microsecond=0))
        self.assertEqual(event.target_list_id, self.campaign.pk)
        self.assertEqual(event.telescope, run.telescope_instrument)

    def test_approve_single_night_space_run_creates_midnight_utc_placeholder_event(self):
        space_site = Observatory.objects.create(
            obscode='250',
            name='Test Space Telescope',
            short_name='TST',
            observations_type=Observatory.SATELLITE_OBSTYPE,
        )
        run = self._make_pending_run(site_raw=space_site.obscode)
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        event = CalendarEvent.objects.get(url=f'CAMPAIGN:{run.pk}')
        self.assertEqual(event.start_time, datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(event.end_time, datetime(2026, 8, 1, 23, 59, tzinfo=timezone.utc))

    def test_approve_range_run_creates_no_calendar_event(self):
        run = self._make_pending_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 15))
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)

    def test_approve_tbd_run_creates_no_calendar_event(self):
        run = self._make_pending_run(window_start=None, window_end=None)
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)

    def test_approve_without_telescope_instrument_creates_no_calendar_event(self):
        run = self._make_pending_run(telescope_instrument='')
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_sun_event_valueerror_skips_projection_without_reverting_approval(self):
        """Pitfall 7: a sun_event() ValueError (e.g. blank site.timezone) must be logged and
        skipped, never reach the broad except Exception that reverts a half-committed
        approval back to PENDING_REVIEW."""
        run = self._make_pending_run()
        with patch('solsys_code.campaign_views.sun_event', side_effect=ValueError('no crossings')):
            response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)


class TestApprovalQueueColumns(TestCase):
    """UAT Test 14 gap closure (16-05): ApprovalQueueTable is trimmed/reordered for triage,
    CampaignRunTable stays spreadsheet-parity (Phase 15 D-09 regression guard).

    No DB rows are needed -- both tables are built with an empty data list; only the
    declared column contract (``.columns``) is under test here.
    """

    def test_actions_leads_approval_queue_table(self):
        column_names = [column.name for column in ApprovalQueueTable([]).columns]
        self.assertEqual(column_names[0], 'actions')

    def test_approval_queue_table_excludes_post_observation_columns(self):
        column_names = {column.name for column in ApprovalQueueTable([]).columns}
        self.assertEqual(column_names & {'weather', 'observation_outcome', 'publication_plans'}, set())

    def test_campaign_run_table_unchanged_by_approval_queue_trim(self):
        """D-09 regression guard: the fix is scoped to ApprovalQueueTable only."""
        column_names = {column.name for column in CampaignRunTable([]).columns}
        self.assertTrue({'weather', 'observation_outcome', 'publication_plans'} <= column_names)
        self.assertNotIn('actions', column_names)


class TestApprovalQueueSiteVisibility(CampaignApprovalTestBase):
    """Regression coverage for the visibility gap: pending runs (site_needs_review=False,
    per D-07) must still surface their submitted site_raw text in the site column, not just
    runs where resolution ran and failed."""

    def test_pending_unresolved_site_shows_site_raw(self):
        run = self._make_pending_run(site=None, site_raw='DCT', site_needs_review=False)
        cell = CampaignRunTable([run]).rows[0].get_cell('site')
        self.assertIn('DCT', cell)

    def test_pending_blank_site_raw_renders_empty_cell(self):
        run = self._make_pending_run(site=None, site_raw='', site_needs_review=False)
        cell = CampaignRunTable([run]).rows[0].get_cell('site')
        self.assertEqual(cell, '')

    def test_resolution_failed_site_still_shows_site_raw_with_failure_indicator(self):
        run = self._make_pending_run(site=None, site_raw='DCT', site_needs_review=True)
        cell = CampaignRunTable([run]).rows[0].get_cell('site')
        self.assertIn('DCT', cell)
        self.assertIn('exclamation-triangle', cell)


class TestApprovalSiteResolution(CampaignApprovalTestBase):
    """Approving an unresolvable free-text site must not fabricate a placeholder
    Observatory row (unlike the already-vetted CSV import path), and must not block
    approval (D-07)."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')
        # Keep tier 2 deterministic and offline: simulate an MPC miss/no-network so
        # resolution always falls through past tier 2 to the create_placeholder branch.
        patcher = patch(
            'solsys_code.campaign_utils.MPCObscodeFetcher.query',
            side_effect=requests.exceptions.RequestException,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_approving_unresolvable_free_text_site_creates_no_placeholder_observatory(self):
        run = self._make_pending_run(site_raw='DCT')
        response = self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(Observatory.objects.count(), 0)

    def test_resolve_site_create_placeholder_false_creates_no_observatory(self):
        observatory, needs_review = resolve_site('DCT', create_placeholder=False)
        self.assertIsNone(observatory)
        self.assertTrue(needs_review)
        self.assertEqual(Observatory.objects.count(), 0)

    def test_resolve_site_default_still_creates_placeholder_observatory(self):
        """CSV-import path (default create_placeholder=True) is unaffected."""
        observatory, needs_review = resolve_site('DCT')
        self.assertIsNotNone(observatory)
        self.assertEqual(observatory.obscode, 'DCT')
        self.assertTrue(needs_review)
        self.assertEqual(Observatory.objects.count(), 1)

    def test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review(self):
        """CR-01 (22-REVIEW.md re-review): a Tier 1 hit against a *pre-existing* tier-3
        placeholder (e.g. a repeat CSV Site Code whose first row already created it) must
        still report needs_review=True -- it is not a genuine resolution just because an
        Observatory row exists for that obscode."""
        placeholder = Observatory.objects.create(obscode='DCT', name=f'{NEEDS_REVIEW_NAME_PREFIX}DCT', short_name='DCT')

        site, needs_review = resolve_site('DCT', create_placeholder=False)

        self.assertEqual(site, placeholder)
        self.assertTrue(needs_review)
        self.assertEqual(Observatory.objects.count(), 1)  # no second placeholder fabricated


class TestSiteSelectionResolution(CampaignApprovalTestBase):
    """SITE-02: the staff-submitted site_selection value drives approve-time resolution."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_staff_typed_existing_obscode_resolves_via_site_selection_tier_1_hit(self):
        """A tier-1 hit (existing Observatory) resolves with no fabrication."""
        Observatory.objects.create(
            obscode='G37',
            name='Lowell Discovery Telescope',
            short_name='LDT',
            lat=34.744,
            lon=-111.4223,
            altitude=2361.0,
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        run = self._make_pending_run(site_raw='Lowell Discvery Tel')  # typo -- never resolves via site_raw
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'approve', 'site_selection': 'G37'},
        )
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site.obscode, 'G37')
        self.assertFalse(run.site_needs_review)
        self.assertEqual(Observatory.objects.count(), 1)

    def test_unresolvable_site_selection_leaves_observatory_count_unchanged(self):
        """Regression on 260705-l1v's invariant: an unresolvable site_selection on approve
        creates no placeholder Observatory."""
        run = self._make_pending_run()
        with patch(
            'solsys_code.campaign_utils.MPCObscodeFetcher.query',
            side_effect=requests.exceptions.RequestException,
        ):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'approve', 'site_selection': 'NOWHERE'},
            )
        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(Observatory.objects.count(), 0)

    def test_approve_re_resolves_when_existing_site_is_a_placeholder(self):
        """WR-01 (22-REVIEW.md re-review): a PENDING_REVIEW run whose ``site`` already
        points at a tier-3 placeholder Observatory (not None) must still re-enter site
        resolution on approve, mirroring ``_resolve_site()``'s placeholder-aware guard --
        not just the site-is-None case. Without the fix, ``run.site is None`` is False here
        (a placeholder Observatory is still an Observatory), so resolution never runs and
        the run stays pointed at the unusable placeholder."""
        Observatory.objects.create(
            obscode='G37',
            name='Lowell Discovery Telescope',
            short_name='LDT',
            lat=34.744,
            lon=-111.4223,
            altitude=2361.0,
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        placeholder = Observatory.objects.create(obscode='DCT', name=f'{NEEDS_REVIEW_NAME_PREFIX}DCT', short_name='DCT')
        run = self._make_pending_run(site=placeholder, site_raw='DCT', site_needs_review=True)

        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'approve', 'site_selection': 'G37'},
        )

        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site.obscode, 'G37')
        self.assertFalse(run.site_needs_review)


class TestSiteSelectionNameCandidateResolution(CampaignApprovalTestBase):
    """Permanent CR-01 regression (21-REVIEW-FIX.md / 21-VERIFICATION.md): a name/
    short_name/old_names display-string ``site_selection`` candidate -- NOT a literal
    obscode -- submitted through the real ``campaigns:decide`` POST resolves ``run.site``
    via ``CampaignRunDecisionView.post()``'s ``selection_to_obscode()`` obscode mapping
    (``campaign_utils``). ``TestSiteSelectionResolution`` above only exercises the
    literal-obscode case (``'G37'`` passes through the mapping unchanged), so it never
    proves the display-string lookup itself works. The verifier independently confirmed
    this behavior with a temporary end-to-end test (written, run, then removed per
    verifier convention) before this class existed; this class makes that coverage
    permanent.
    """

    def setUp(self):
        cache.clear()
        self.client.login(username='staffcoordinator', password='pw')
        self.observatory = Observatory.objects.create(
            obscode='G37',
            name='Lowell Discovery Telescope',
            short_name='LDT',
            lat=34.744,
            lon=-111.4223,
            altitude=2361.0,
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        # Explicit candidate pool mapping a name (name_utf8, from the fixture), a
        # short_name, AND an old_names string all to obscode 'G37' -- the three
        # display-string candidate types build_site_candidates()/_flatten_mpc_candidates()
        # produce (RESEARCH.md Open Question 2).
        candidate_pool = {
            **campaign_utils._flatten_mpc_candidates(BULK_MPC_FIXTURE),
            'LDT': 'G37',
            'Historic Lowell Reflector': 'G37',
        }
        # The selection->obscode mapping lives in campaign_utils.selection_to_obscode(), which
        # calls campaign_utils.build_site_candidates() -- patch it at that layer (not the view
        # import site) so the decide POST's mapping sees this deterministic pool.
        patcher = patch('solsys_code.campaign_utils.build_site_candidates', return_value=candidate_pool)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        cache.clear()

    def test_name_short_name_and_old_names_candidates_resolve_via_real_decide_post(self):
        candidates = {
            'name_utf8': 'Lowell Discovery Telescope',
            'short_name': 'LDT',
            'old_names': 'Historic Lowell Reflector',
        }
        for index, (candidate_type, site_selection) in enumerate(candidates.items()):
            with self.subTest(candidate_type=candidate_type, site_selection=site_selection):
                # Distinct window_start per iteration -- CampaignRun's natural-key
                # UniqueConstraint keys on (campaign, telescope_instrument, window_start,
                # window_end), so three runs sharing _make_pending_run()'s default window
                # would otherwise collide on the second/third create().
                run_date = date(2026, 8, 1 + index)
                run = self._make_pending_run(
                    site_raw='Lowell Discvery Tel',  # typo -- never self-resolves
                    window_start=run_date,
                    window_end=run_date,
                )
                response = self.client.post(
                    reverse('campaigns:decide', kwargs={'pk': run.pk}),
                    {'action': 'approve', 'site_selection': site_selection},
                )
                self.assertEqual(response.status_code, 302)
                run.refresh_from_db()
                self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
                self.assertEqual(run.site.obscode, 'G37')
                self.assertEqual(run.site_id, self.observatory.pk)
                self.assertFalse(run.site_needs_review)
                self.assertEqual(Observatory.objects.count(), 1)


class TestSitesNeedingReview(CampaignApprovalTestBase):
    """D-06/D-07/D-08/22-REVIEWS.md findings 3/5/6/8c: the resolve_site decision action for
    approved runs whose site never resolved (``site_needs_review=True``).

    Fixture convention mirrors TestApproval/TestCalendarProjection: a Tier-1-resolvable
    ground Observatory ('F65') so resolution never needs a live MPC API call.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
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

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def _make_needs_review_run(self, **overrides):
        """An APPROVED run with site_needs_review=True (the dead end this phase closes)."""
        kwargs = {
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'site': None,
            'site_needs_review': True,
        }
        kwargs.update(overrides)
        return self._make_pending_run(**kwargs)

    def test_resolve_success_single_night_ground_run_projects_calendar_event(self):
        run = self._make_needs_review_run(site_raw='F65')
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('Site resolved — run added to the calendar.', messages_list)

    def test_resolve_never_re_resolves_already_set_site_but_retries_projection(self):
        """D-06/finding 8c: a run with site already set (the projection-failed retry state)
        must never re-call resolve_site, but its projection IS re-attempted and the flag
        clears on success."""
        run = self._make_needs_review_run(site=self.ground_site, site_raw='F65')
        with patch('solsys_code.campaign_views.resolve_site') as mock_resolve_site:
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site'},
            )
        self.assertEqual(response.status_code, 302)
        mock_resolve_site.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)

    def test_resolve_retryable_projection_failure_stays_approved_site_saved_flag_stays_true(self):
        """Finding 3: a projection failure must not revert approval, must keep the resolved
        site, and must keep site_needs_review=True so the row stays in the retry surface."""
        run = self._make_needs_review_run(site_raw='F65')
        with patch('solsys_code.campaign_views._project_calendar_event', side_effect=RuntimeError('boom')):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'F65'},
                follow=True,
            )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertTrue(run.site_needs_review)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertTrue(any('calendar entry' in m for m in messages_list))

        # Finding 3: the retry surface is preserved -- a subsequent staff GET of the
        # approval queue still lists the run in review_table's underlying data (not just
        # the model field).
        queue_response = self.client.get(reverse('campaigns:approval_queue'))
        review_table = queue_response.context['review_table']
        self.assertIn(run.pk, [row.record.pk for row in review_table.rows])

    def test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event(self):
        """CR-01 (22-REVIEW.md): resolving to a site whose ``timezone`` is blank -- exactly
        what ``MPCObscodeFetcher.to_observatory()`` (Tier 2) produces, since it never sets
        ``timezone`` -- must not silently report success. ``sun_event()`` raises ``ValueError``
        for a blank timezone, and ``_project_calendar_event()`` must now re-raise it (rather
        than swallowing it into a bare ``False``) so ``_resolve_site()``'s existing
        non-reverting except block treats this the same as any other projection failure:
        keep ``site_needs_review=True``, warn instead of claiming success, and create no
        ``CalendarEvent``. Fixtured directly as a local Observatory (Tier 1 hit) with a blank
        ``timezone`` rather than mocking the MPC fetch -- CR-01 only cares about the blank
        timezone, not which tier produced it."""
        blank_tz_site = Observatory.objects.create(
            obscode='T99',
            name='Blank Timezone Site',
            short_name='BTS',
            lat=-30.0,
            lon=149.0,
            altitude=1000.0,
            timezone='',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        run = self._make_needs_review_run(site_raw='T99')
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'T99'},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, blank_tz_site.pk)
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertNotIn('Site resolved.', messages_list)
        self.assertTrue(any('calendar entry' in m for m in messages_list))

        # Finding 3 (still holds under CR-01): the retry surface is preserved -- the run
        # stays listed in review_table's underlying data.
        queue_response = self.client.get(reverse('campaigns:approval_queue'))
        review_table = queue_response.context['review_table']
        self.assertIn(run.pk, [row.record.pk for row in review_table.rows])

    def test_resolve_lost_race_no_op_warns(self):
        """Finding 5: a concurrent resolution landing between the fresh fetch and the site
        write must make the loser's claim update match 0 rows -- no write, no projection."""
        run = self._make_needs_review_run(site_raw='F65')

        def _racing_resolve_site(obscode_selection, create_placeholder=False):
            # Simulate the second staff member's POST winning the race: directly resolve
            # the row's site in the DB before this (the loser's) call returns.
            CampaignRun.objects.filter(pk=run.pk).update(site=self.ground_site, site_needs_review=False)
            return self.ground_site, False

        with patch('solsys_code.campaign_views.resolve_site', side_effect=_racing_resolve_site):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'F65'},
                follow=True,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn("This run's site was already resolved by someone else.", messages_list)

    def test_resolve_rejects_pending_review_run(self):
        run = self._make_pending_run()
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertIsNone(run.site)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('This run is not awaiting site resolution.', messages_list)

    def test_resolve_rejects_already_resolved_run(self):
        run = self._make_needs_review_run(site=self.ground_site, site_needs_review=False)
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('This run is not awaiting site resolution.', messages_list)

    def test_resolve_range_tbd_run_clears_flag_with_no_calendar_event(self):
        run = self._make_needs_review_run(site_raw='F65', window_start=date(2026, 8, 1), window_end=date(2026, 8, 15))
        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 0)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('Site resolved.', messages_list)

    def test_resolve_unresolvable_selection_leaves_site_none_and_flag_true(self):
        run = self._make_needs_review_run(site_raw='')
        with patch(
            'solsys_code.campaign_utils.MPCObscodeFetcher.query',
            side_effect=requests.exceptions.RequestException,
        ):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'NOWHERE'},
                follow=True,
            )
        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn(
            'Could not resolve that site. Try a different search term or an exact MPC code, '
            'or use Create new Observatory.',
            messages_list,
        )

    def test_review_table_context_lists_only_approved_needs_review_runs(self):
        """D-07: review_table lists APPROVED+site_needs_review=True runs only -- not
        pending, not resolved-approved -- INCLUDING a projection-failed retry row (site set,
        flag still True), since the filter is on the flag alone."""
        self._make_pending_run(site_raw='F65')  # PENDING_REVIEW -- must not appear
        self._make_needs_review_run(site_raw='F65', window_start=date(2026, 8, 2), window_end=date(2026, 8, 2))
        resolved_approved = self._make_needs_review_run(
            site=self.ground_site,
            site_needs_review=False,
            window_start=date(2026, 8, 3),
            window_end=date(2026, 8, 3),
        )
        retry_row = self._make_needs_review_run(
            site=self.ground_site, window_start=date(2026, 8, 4), window_end=date(2026, 8, 4)
        )

        response = self.client.get(reverse('campaigns:approval_queue'))

        review_table = response.context['review_table']
        review_pks = {row.record.pk for row in review_table.rows}
        self.assertNotIn(resolved_approved.pk, review_pks)
        self.assertIn(retry_row.pk, review_pks)

    def test_unresolved_review_row_renders_live_search_widget_and_resolve_button(self):
        run = self._make_needs_review_run(site_raw='F65')

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertIn('Sites Needing Review', content)
        self.assertIn('name="site_selection"', content)
        self.assertIn(f'form="resolve-form-{run.pk}"', content)
        self.assertIn('hx-get', content)
        self.assertIn(reverse('campaigns:site_search'), content)
        self.assertIn('input[this.value.length >= 2] changed delay:300ms', content)
        self.assertIn('Create new Observatory', content)
        self.assertIn('value="resolve_site"', content)
        self.assertIn('btn-primary', content)

    def test_retry_row_renders_plain_text_site_and_resolve_button_no_input(self):
        """Finding 8c: a run with site already set (flag still True) shows its resolved
        site as plain text, no site-selection input, but still carries the Resolve button."""
        run = self._make_needs_review_run(site=self.ground_site)

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertNotIn(f'id="site-input-{run.pk}"', content)
        self.assertIn('FTS', content)
        self.assertIn(f'id="resolve-form-{run.pk}"', content)
        self.assertIn('value="resolve_site"', content)

    def test_review_table_empty_state_renders_configured_copy(self):
        response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertContains(response, 'No sites currently need review.')


class TestIsPlaceholderObservatory(TestCase):
    """Unit coverage for campaign_utils.is_placeholder_observatory() (22-06 Task 1) --
    the pure, DB-free string check both render_site() and _resolve_site() key off of."""

    def test_placeholder_observatory_detected(self):
        placeholder = Observatory.objects.create(obscode='DCT', name=f'{NEEDS_REVIEW_NAME_PREFIX}DCT', short_name='DCT')
        self.assertTrue(campaign_utils.is_placeholder_observatory(placeholder))

    def test_real_observatory_not_placeholder(self):
        real = Observatory.objects.create(obscode='F65', name='Faulkes Telescope South', short_name='FTS')
        self.assertFalse(campaign_utils.is_placeholder_observatory(real))

    def test_none_is_not_placeholder(self):
        self.assertFalse(campaign_utils.is_placeholder_observatory(None))

    def test_tier3_create_uses_shared_prefix_constant(self):
        """No behavioral change (Task 1): resolve_site()'s tier-3 fallback still produces
        the exact same name shape, now built from NEEDS_REVIEW_NAME_PREFIX."""
        site, needs_review = resolve_site('ZZZ')
        self.assertTrue(needs_review)
        self.assertEqual(site.name, f'{NEEDS_REVIEW_NAME_PREFIX}ZZZ')
        self.assertTrue(campaign_utils.is_placeholder_observatory(site))


class TestSelectionToObscode(TestCase):
    """Unit coverage for campaign_utils.selection_to_obscode() (debug/site-resolve-list-old-names).

    The site-search suggestion fragment (site_search_results.html) writes a COMBINED
    ``'{display} ({obscode})'`` value into the site_selection input when a suggestion is
    clicked. The approve/resolve handlers must map that combined form -- and a bare display
    string or bare obscode typed/picked verbatim -- back to an obscode; anything unmappable
    passes through unchanged (so resolve_site() can tier-1/2 it, or reject it).
    """

    POOL = {
        'G37': 'G37',
        'Lowell Discovery Telescope': 'G37',
        'G96': 'G96',
        'University of Arizona Mt. Lemmon Survey': 'G96',
    }

    def setUp(self):
        patcher = patch('solsys_code.campaign_utils.build_site_candidates', return_value=self.POOL)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_combined_display_obscode_widget_value_maps_to_obscode(self):
        # The exact string site_search_results.html writes into the input on click.
        self.assertEqual(campaign_utils.selection_to_obscode('Lowell Discovery Telescope (G37)'), 'G37')

    def test_bare_obscode_maps_via_exact_pool_hit(self):
        self.assertEqual(campaign_utils.selection_to_obscode('G37'), 'G37')

    def test_bare_display_string_maps_via_exact_pool_hit(self):
        self.assertEqual(campaign_utils.selection_to_obscode('University of Arizona Mt. Lemmon Survey'), 'G96')

    def test_display_containing_parens_keeps_only_trailing_obscode_group(self):
        # A display that itself contains parentheses: only the LAST '(...)' is the obscode.
        self.assertEqual(campaign_utils.selection_to_obscode('Weird (Annex) Site (G96)'), 'G96')

    def test_combined_form_with_unknown_display_falls_back_to_parenthesized_obscode(self):
        # Display part not in the pool -> recover the parenthesized obscode token directly.
        self.assertEqual(campaign_utils.selection_to_obscode('Some Brand New Scope (X99)'), 'X99')

    def test_unmappable_free_text_passes_through_unchanged(self):
        self.assertEqual(campaign_utils.selection_to_obscode('Totally Unknown Site'), 'Totally Unknown Site')


class TestPlaceholderSiteReplacement(CampaignApprovalTestBase):
    """22-06 gap closure (UAT gap 2B): a Sites Needing Review row whose site is a tier-3
    PLACEHOLDER Observatory now surfaces the correction widget (render_site()) and can be
    replaced via resolve_site (view), while D-06 (never re-resolve a genuine site, racing
    protection) and D-09 (never fabricate from unresolvable input) both stay intact.

    Fixture convention mirrors TestSitesNeedingReview: a Tier-1-resolvable ground
    Observatory ('F65') so resolution never needs a live MPC API call.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
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

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def _make_placeholder_observatory(self, obscode='DCT'):
        """A tier-3 placeholder Observatory shaped exactly like resolve_site()'s fallback --
        NEEDS_REVIEW_NAME_PREFIX name, blank timezone (model default)."""
        return Observatory.objects.create(
            obscode=obscode, name=f'{NEEDS_REVIEW_NAME_PREFIX}{obscode}', short_name=obscode
        )

    def _make_placeholder_run(self, **overrides):
        """An APPROVED run whose site is a placeholder Observatory (site_needs_review=True).

        Only creates its own default placeholder Observatory when the caller doesn't
        already supply a ``site=`` override -- avoids creating a second, unwanted
        placeholder (obscode/name collision) when the caller already made one.
        """
        kwargs = {
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'site_needs_review': True,
        }
        if 'site' not in overrides:
            kwargs['site'] = self._make_placeholder_observatory()
        kwargs.update(overrides)
        return self._make_pending_run(**kwargs)

    def test_placeholder_row_renders_live_search_widget_not_plain_text(self):
        """render_site() (Task 2): a resolve-mode row whose site is a placeholder falls
        through to the correction widget, distinguishing it from the genuine-site retry
        state covered by test_retry_row_renders_plain_text_site_and_resolve_button_no_input."""
        run = self._make_placeholder_run(site_raw='DCT')

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertIn(f'id="site-input-{run.pk}"', content)
        self.assertIn('name="site_selection"', content)
        self.assertIn(f'form="resolve-form-{run.pk}"', content)
        self.assertIn('Create new Observatory', content)

    def test_placeholder_row_read_only_table_never_renders_widget(self):
        """WR-01: even a placeholder-site row must never render the live widget in a
        show_actions=False table, regardless of self.mode -- constructed directly since no
        live show_actions=False + mode='resolve' view exists yet (a hypothetical future
        read-only "resolved sites" audit view, per render_site()'s own docstring)."""
        run = self._make_placeholder_run(site_raw='DCT', approval_status=CampaignRun.ApprovalStatus.APPROVED)
        table = ApprovalQueueTable([run], show_actions=False, mode='resolve')

        rendered = table.render_site(run)

        self.assertNotIn('site_selection', str(rendered))
        self.assertIn('DCT', str(rendered))

    def test_placeholder_replacement_repoints_site_and_clears_review_flag(self):
        """Placeholder replacement: a real site_selection replaces the placeholder site and,
        since preconditions are met (single-night window + telescope_instrument), the flag
        clears and the calendar event projects."""
        placeholder = self._make_placeholder_observatory()
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')

        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)
        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('Site resolved — run added to the calendar.', messages_list)

    def test_placeholder_replacement_via_combined_widget_selection_resolves(self):
        """Regression (debug/site-resolve-list-old-names): the site-search suggestion
        fragment (site_search_results.html) writes the COMBINED ``'{display} ({obscode})'``
        string into the ``site_selection`` input when a staff member clicks a suggestion --
        e.g. ``'Faulkes Telescope South (F65)'``, exactly what the user selected for the
        DCT/G37 row. The pre-fix handler mapped it via
        ``build_site_candidates().get(selection, selection)``, whose pool has no key for the
        combined form, so the 30+ char string reached ``resolve_site()`` and was rejected as
        an oversized obscode (``len > _MAX_OBSCODE_LEN``) -> ``(None, True)`` -> the generic
        "Could not resolve that site" error. ``selection_to_obscode()`` must now round-trip
        the combined value back to F65 and replace the placeholder."""
        placeholder = self._make_placeholder_observatory()
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')
        # A controlled pool so the mapping is deterministic and no live MPC call is made --
        # keyed on the bare display string and bare obscode only (the real pool's shape),
        # NOT the combined 'display (obscode)' form the widget actually submits.
        pool = {'F65': 'F65', 'Faulkes Telescope South': 'F65', 'FTS': 'F65'}
        with patch('solsys_code.campaign_utils.build_site_candidates', return_value=pool):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'Faulkes Telescope South (F65)'},
                follow=True,
            )

        self.assertEqual(response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn('Site resolved — run added to the calendar.', messages_list)
        # The exact pre-fix failure must be gone.
        self.assertNotIn(
            'Could not resolve that site. Try a different search term or an exact '
            'MPC code, or use Create new Observatory.',
            messages_list,
        )

    def test_placeholder_replacement_deletes_orphaned_placeholder_observatory(self):
        """WR-03 (22-REVIEW.md re-review): once a placeholder Observatory is successfully
        replaced and nothing else references it, the now-orphaned placeholder row itself
        must be cleaned up -- not left behind to keep satisfying is_placeholder_observatory()
        and polluting the CR-02 search-suggestion pool."""
        placeholder = self._make_placeholder_observatory()
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')

        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Observatory.objects.filter(pk=placeholder.pk).exists())

    def test_placeholder_replacement_keeps_placeholder_still_referenced_by_another_run(self):
        """WR-03: the orphaned-placeholder cleanup must never delete a placeholder still
        referenced by a *different* CampaignRun (e.g. a second still-unresolved row sharing
        the same not-yet-configured site)."""
        placeholder = self._make_placeholder_observatory()
        other_run = self._make_placeholder_run(
            site=placeholder, site_raw='DCT', window_start=date(2026, 9, 1), window_end=date(2026, 9, 1)
        )
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')

        response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(Observatory.objects.filter(pk=placeholder.pk).exists())
        other_run.refresh_from_db()
        self.assertEqual(other_run.site_id, placeholder.pk)

    def test_placeholder_replacement_failure_fabricates_no_second_placeholder(self):
        """D-09: an unresolvable site_selection on a placeholder-site row must write
        nothing new -- no second placeholder Observatory, the run keeps pointing at its
        existing placeholder, and stays in Sites Needing Review."""
        placeholder = self._make_placeholder_observatory()
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')
        observatory_count_before = Observatory.objects.count()

        with patch(
            'solsys_code.campaign_utils.MPCObscodeFetcher.query',
            side_effect=requests.exceptions.RequestException,
        ):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'NOWHERE'},
                follow=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Observatory.objects.count(), observatory_count_before)
        run.refresh_from_db()
        self.assertEqual(run.site_id, placeholder.pk)
        self.assertTrue(run.site_needs_review)
        messages_list = [str(m) for m in response.context['messages']]
        self.assertIn(
            'Could not resolve that site. Try a different search term or an exact MPC code, '
            'or use Create new Observatory.',
            messages_list,
        )

    def test_genuine_site_still_never_re_resolved_when_replacing_placeholder_would_apply(self):
        """D-06 preserved: a genuinely-resolved (non-placeholder) site is never re-resolved
        by this same placeholder-replacement path -- resolve_site is not called, matching
        the existing finding-8c coverage in TestSitesNeedingReview."""
        run = self._make_placeholder_run(site=self.ground_site, site_raw='F65')
        with patch('solsys_code.campaign_views.resolve_site') as mock_resolve_site:
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site'},
            )
        self.assertEqual(response.status_code, 302)
        mock_resolve_site.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)

    def test_racing_second_resolve_after_placeholder_replacement_does_not_double_write(self):
        """Racing-guard shape: once a placeholder has been replaced by a real site, a
        second resolve_site POST for the same (now-real-site) run must not re-resolve --
        it falls straight to the never-re-resolve path (D-06), never a second write."""
        placeholder = self._make_placeholder_observatory()
        run = self._make_placeholder_run(site=placeholder, site_raw='DCT')

        first_response = self.client.post(
            reverse('campaigns:decide', kwargs={'pk': run.pk}),
            {'action': 'resolve_site', 'site_selection': 'F65'},
            follow=True,
        )
        self.assertEqual(first_response.status_code, 200)
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)
        self.assertFalse(run.site_needs_review)

        # Simulate the retry surface: force the flag back on (as a failed-projection retry
        # row would have it) without touching site, then re-POST resolve_site.
        CampaignRun.objects.filter(pk=run.pk).update(site_needs_review=True)
        with patch('solsys_code.campaign_views.resolve_site') as mock_resolve_site:
            second_response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': run.pk}),
                {'action': 'resolve_site', 'site_selection': 'F65'},
                follow=True,
            )
        self.assertEqual(second_response.status_code, 200)
        mock_resolve_site.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.site_id, self.ground_site.pk)


class TestApprovalQueueSitesNeedingReviewGrouping(CampaignApprovalTestBase):
    """UAT gap 2A closure (22-05): the Sites Needing Review section must be visually
    differentiated from the historical Recently Decided table -- an actionable card, not
    another plain DOM sibling -- while preserving D-07's locked document order (pending /
    decided / sites-needing-review). This is presentation-only: no queryset/table/view
    change, so these assertions hold even against empty tables.
    """

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_sites_needing_review_renders_as_distinguishing_action_required_card(self):
        response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('border-warning', content)
        self.assertIn('Sites Needing Review — action required', content)

    def test_d07_order_preserved_decided_precedes_sites_needing_review(self):
        response = self.client.get(reverse('campaigns:approval_queue'))
        content = response.content.decode()
        decided_index = content.index('Recently Decided')
        review_index = content.index('Sites Needing Review')
        self.assertLess(decided_index, review_index)
        self.assertIn('Pending Review', content)
        self.assertIn('Recently Decided', content)


def _extract_create_observatory_form_fields(html_content: str, form_action_fragment: str) -> dict[str, str]:
    """Extract ``name`` -> ``value`` for every ``<input>`` inside the
    ``observatory_create.html`` form whose ``action`` attribute contains
    ``form_action_fragment`` (CR-02). Stdlib ``html.parser.HTMLParser`` only -- no new
    dependency. Used to replay ONLY the fields the rendered template itself contains, so a
    round-trip test proves the template carries a field rather than merely that the view
    logic works when handed a hand-built POST body."""

    class _FormFieldParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_target_form = False
            self.fields: dict[str, str] = {}

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == 'form':
                if form_action_fragment in (attrs_dict.get('action') or ''):
                    self.in_target_form = True
                return
            if tag == 'input' and self.in_target_form:
                name = attrs_dict.get('name')
                if name:
                    self.fields[name] = attrs_dict.get('value') or ''

        def handle_endtag(self, tag):
            if tag == 'form' and self.in_target_form:
                self.in_target_form = False

    parser = _FormFieldParser()
    parser.feed(html_content)
    return parser.fields


def _stub_to_observatory():
    """Create-and-return an Observatory row, mirroring what a real MPC-backed
    ``MPCObscodeFetcher.to_observatory()`` call would do -- used to fake a successful
    MPC lookup without hitting the live MPC API."""
    return Observatory.objects.create(
        obscode='G37',
        name='Lowell Discovery Telescope',
        short_name='LDT',
        lat=34.744,
        lon=-111.4223,
        altitude=2361.0,
        observations_type=Observatory.OPTICAL_OBSTYPE,
    )


class TestCreateObservatoryRoundTrip(CampaignApprovalTestBase):
    """SITE-02/D-05: the "Create new Observatory" round-trip from the approval queue --
    ``?obscode=`` prefill and a validated ``?next=`` redirect back to the queue."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')
        self.next_url = reverse('campaigns:approval_queue')

    def test_get_with_obscode_and_next_prefills_form_initial(self):
        create_url = reverse('solsys_code_observatory:create')
        response = self.client.get(create_url, {'obscode': 'G37', 'next': self.next_url})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['form'].initial.get('obscode'), 'G37')

    def test_valid_create_with_safe_next_redirects_to_approval_queue(self):
        create_url = reverse('solsys_code_observatory:create')
        with (
            patch('solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.query'),
            patch(
                'solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.to_observatory',
                side_effect=_stub_to_observatory,
            ),
        ):
            response = self.client.post(create_url, {'obscode': 'G37', 'next': self.next_url})
        self.assertRedirects(response, self.next_url)
        self.assertEqual(Observatory.objects.filter(obscode='G37').count(), 1)

    def test_unsafe_next_falls_back_to_detail_redirect(self):
        create_url = reverse('solsys_code_observatory:create')
        with (
            patch('solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.query'),
            patch(
                'solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.to_observatory',
                side_effect=_stub_to_observatory,
            ),
        ):
            response = self.client.post(create_url, {'obscode': 'G37', 'next': 'https://evil.example/steal'})
        observatory = Observatory.objects.get(obscode='G37')
        self.assertRedirects(response, reverse('solsys_code_observatory:detail', kwargs={'pk': observatory.pk}))


class TestCreateObservatoryTemplateNextRoundTrip(CampaignApprovalTestBase):
    """Permanent CR-02 regression (21-REVIEW-FIX.md / 21-VERIFICATION.md): the real
    ``observatory_create.html`` template renders a hidden ``next`` input carrying
    ``request.GET.next``, and POSTing ONLY the fields the rendered form itself contains
    (extracted from the response HTML, not hand-constructed with ``next`` injected)
    redirects to that ``next`` target. ``TestCreateObservatoryRoundTrip`` above proves the
    view logic (``get_success_url()``) works when handed a manually-built POST body, but
    never proves the template actually carries the field. The verifier independently
    confirmed the real-template round-trip with a temporary end-to-end test (written, run,
    then removed per verifier convention) before this class existed; this class makes that
    coverage permanent.
    """

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')
        self.next_url = reverse('campaigns:approval_queue')
        self.create_url = reverse('solsys_code_observatory:create')

    def test_rendered_form_carries_next_field_and_replaying_it_redirects(self):
        response = self.client.get(self.create_url, {'obscode': 'G37', 'next': self.next_url})
        self.assertEqual(response.status_code, 200)

        fields = _extract_create_observatory_form_fields(response.content.decode(), self.create_url)
        # Load-bearing proof the template rendered the hidden field -- NOT hand-injected.
        self.assertEqual(fields.get('next'), self.next_url)

        with (
            patch('solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.query'),
            patch(
                'solsys_code.solsys_code_observatory.views.MPCObscodeFetcher.to_observatory',
                side_effect=_stub_to_observatory,
            ),
        ):
            post_response = self.client.post(self.create_url, fields)

        self.assertRedirects(post_response, self.next_url)
        self.assertEqual(Observatory.objects.filter(obscode='G37').count(), 1)


class TestCalendarNoChurn(CampaignApprovalTestBase):
    """CAL-03: re-approve produces no duplicate event and no modified churn."""

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        # D-06: a Tier-1-resolvable ground Observatory for the default fixture's site_raw
        # ('F65') so the first approve's calendar projection succeeds deterministically.
        Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_second_approve_leaves_event_count_and_modified_unchanged(self):
        run = self._make_pending_run()
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        event = CalendarEvent.objects.get(url=f'CAMPAIGN:{run.pk}')
        modified_after_first_approve = event.modified

        # Second approve on an already-APPROVED row: updated_count == 0 (SUBMIT-03), so the
        # projection block is never re-entered -- no duplicate, no modified churn.
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})

        self.assertEqual(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_after_first_approve)


@override_settings(CACHES=ISOLATED_TEST_CACHES)
class TestSiteFuzzyMatch(TestCase):
    """Wave-0 scaffold (SITE-01): cached MPC candidate pool + difflib fuzzy matching.

    Cache-isolated (bug #3, debug/site-search-degraded-pool-recurrence): this class calls the
    REAL ``build_site_candidates()`` (writing the pool to the ``mpc_obscode_candidates`` cache
    key) and ``cache.clear()`` in setUp/tearDown. Pinned to an in-memory LocMemCache so those
    writes/clears never touch the shared /tmp file cache the dev runserver serves site-search
    from -- without this, running this class wiped the runserver's warmed pool (the bug #3
    regression).

    Not-yet-existing helpers (``MPCObscodeFetcher.query_all``, ``campaign_utils.
    build_site_candidates``, ``campaign_utils.fuzzy_match_candidates``) are referenced via
    module attribute access so RED failures before Tasks 2-3 land are localized
    AttributeErrors on these specific calls, not an ImportError that would de-collect the
    whole test module (this class deliberately does not import the two campaign_utils
    helpers by name at module scope). ``requests.get``/``MPCObscodeFetcher.query_all`` is
    always mocked -- no test in this class hits the live MPC API.

    Uses a plain ``TestCase`` (not ``CampaignApprovalTestBase``) -- this class needs no
    campaign/staff/pending-run fixtures, only ``Observatory`` rows for the local-pool
    fallback case.
    """

    def setUp(self):
        cache.clear()

    def tearDown(self):
        cache.clear()

    @patch('requests.get')
    def test_query_all_returns_fixture_dict_without_mutating_query_contract(self, mock_get):
        mock_response = MagicMock(ok=True)
        mock_response.json.return_value = BULK_MPC_FIXTURE
        mock_get.return_value = mock_response

        fetcher = MPCObscodeFetcher()
        result = fetcher.query_all()

        self.assertEqual(result, BULK_MPC_FIXTURE)
        self.assertEqual(fetcher.obs_data, BULK_MPC_FIXTURE)
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs.get('json'), {})
        # query()'s own single-code contract is untouched by adding query_all -- a fresh
        # fetcher's query() must still exist and behave independently of query_all().
        self.assertTrue(callable(fetcher.query))

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_flattens_obscode_name_and_short_name(self, mock_query_all):
        mock_query_all.return_value = BULK_MPC_FIXTURE

        pool = campaign_utils.build_site_candidates()

        for obscode, rec in BULK_MPC_FIXTURE.items():
            self.assertEqual(pool.get(obscode), obscode)
            self.assertEqual(pool.get(rec['name_utf8']), obscode)
            self.assertEqual(pool.get(rec['short_name']), obscode)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_folds_list_valued_old_names(self, mock_query_all):
        """Regression (debug/site-search-mpc-no-match): the live MPC bulk API returns
        `old_names` as a JSON *list* (e.g. G96 -> ['Mt. Lemmon Survey']), not a string. The
        pre-fix _flatten_mpc_candidates() used the list as a dict key/membership operand and
        raised TypeError: unhashable type: 'list', which build_site_candidates() silently
        swallowed -- discarding the ENTIRE MPC pool and degrading to local-only. Every MPC
        record here has a real name and a list old_names; all must survive, and each prior
        name must be an independently-matchable candidate mapped back to its obscode."""
        fixture = {
            'G96': {
                'name_utf8': 'University of Arizona Mt. Lemmon Survey',
                'short_name': 'University of Arizona Mt. Lemmon Survey',
                'old_names': ['Mt. Lemmon Survey'],
                'observations_type': 'optical',
                'longitude': 249.21128,
            },
            '061': {
                'name_utf8': 'Uzhhorod',
                'short_name': 'Uzh',
                'old_names': ['Uzhgorod', 'Uzhorod'],
                'observations_type': 'optical',
                'longitude': 22.3,
            },
        }
        mock_query_all.return_value = fixture

        pool = campaign_utils.build_site_candidates()

        # The whole MPC pool survived (no swallowed TypeError -> no local-only degradation).
        self.assertEqual(pool.get('G96'), 'G96')
        self.assertEqual(pool.get('061'), '061')
        self.assertEqual(pool.get('University of Arizona Mt. Lemmon Survey'), 'G96')
        # Each list-valued old name is its own candidate, resolving back to the obscode.
        self.assertEqual(pool.get('Mt. Lemmon Survey'), 'G96')
        self.assertEqual(pool.get('Uzhgorod'), '061')
        self.assertEqual(pool.get('Uzhorod'), '061')
        # End-to-end: typing the historical name surfaces the current obscode as a suggestion.
        matches = campaign_utils.substring_or_fuzzy_match_candidates('Mt. Lemmon', pool)
        self.assertIn(('Mt. Lemmon Survey', 'G96'), matches)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_flatten_mpc_candidates_tolerates_string_and_missing_old_names(self, mock_query_all):
        """The list fix must not regress the string / None / missing old_names shapes that
        the single-code query() endpoint and existing fixtures use."""
        fixture = {
            'AAA': {'name_utf8': 'Alpha Obs', 'short_name': 'Alpha', 'old_names': 'Legacy Alpha'},
            'BBB': {'name_utf8': 'Beta Obs', 'short_name': 'Beta', 'old_names': None},
            'CCC': {'name_utf8': 'Gamma Obs', 'short_name': 'Gamma'},  # old_names key absent
        }
        pool = campaign_utils._flatten_mpc_candidates(fixture)

        self.assertEqual(pool.get('Legacy Alpha'), 'AAA')  # string old_names still folded in
        self.assertEqual(pool.get('Beta Obs'), 'BBB')  # None old_names simply skipped, no crash
        self.assertEqual(pool.get('Gamma Obs'), 'CCC')  # missing old_names key, no crash

    def test_flatten_mpc_candidates_survives_shape_surprise_in_any_field(self):
        """Generalized robustness (debug/site-search-degraded-pool-recurrence, bug #3): the bug
        #1 fix normalized only `old_names`, but the SAME failure family applies to every
        candidate field. The live MPC bulk API has already shipped one field (`old_names`) as a
        surprising JSON *list*; if `name_utf8`/`short_name` ever arrive non-str (or a whole
        record is non-dict), the pre-hardening flatten would use the value in a dict-key /
        membership operation and raise `TypeError: unhashable type: 'list'`, which
        build_site_candidates() silently swallows -- dropping the ENTIRE ~2,712-code pool for
        one malformed field. This feeds a fixture where EVERY candidate field takes each
        surprising shape (list, dict, int, None, missing) and asserts (a) flatten never raises,
        and (b) the one well-formed record still resolves -- so a single future shape surprise
        degrades to "skip that field/record", never to a whole-pool drop that reverts live
        site-search to 'No matches'. A live audit confirmed all 2,712 records currently carry
        str name_utf8/short_name, so this is forward-looking hardening, not a current-data fix."""
        fixture = {
            # A genuinely well-formed record must still survive alongside the malformed ones.
            'G37': {'name_utf8': 'Lowell Discovery Telescope', 'short_name': 'LDT', 'old_names': None},
            'LST': {'name_utf8': ['a', 'list'], 'short_name': 'ListName', 'old_names': None},  # list name_utf8
            'DCT': {'name_utf8': 'Dict Name', 'short_name': {'k': 'v'}, 'old_names': None},  # dict short_name
            'INT': {'name_utf8': 12345, 'short_name': 67890, 'old_names': None},  # int scalars
            'LON': {'name_utf8': 'Long Names Site', 'short_name': 'LNS', 'old_names': ['prior', ['nested']]},
            'NON': 'this record is not a dict at all',  # non-dict record
            'EMP': {},  # empty record, every candidate field missing
        }

        # Must not raise despite every shape surprise above.
        pool = campaign_utils._flatten_mpc_candidates(fixture)

        # The well-formed record and every well-formed field are intact.
        self.assertEqual(pool.get('G37'), 'G37')
        self.assertEqual(pool.get('Lowell Discovery Telescope'), 'G37')
        self.assertEqual(pool.get('LDT'), 'G37')
        # A non-str field is treated as absent, but the record's other good fields still fold in.
        self.assertEqual(pool.get('ListName'), 'LST')  # str short_name survives a list name_utf8
        self.assertEqual(pool.get('Dict Name'), 'DCT')  # str name_utf8 survives a dict short_name
        self.assertEqual(pool.get('Long Names Site'), 'LON')
        self.assertEqual(pool.get('prior'), 'LON')  # str element of a mixed old_names list folds in
        # The obscode itself is always a (str) candidate, even when every scalar field is bad.
        self.assertEqual(pool.get('INT'), 'INT')
        self.assertEqual(pool.get('EMP'), 'EMP')
        # No non-str value ever leaked in as a key (the exact unhashable-type crash vector).
        self.assertTrue(all(isinstance(k, str) for k in pool))

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_degraded_pool_uses_short_ttl(self, mock_query_all):
        """A local-only fallback pool (MPC fetch failed) must be cached only briefly, so a
        transient MPC outage cannot poison every site search for the full 24h TTL (the
        amplifier that turned a swallowed error into a persistent, cross-restart outage)."""
        mock_query_all.side_effect = requests.exceptions.RequestException
        Observatory.objects.create(
            obscode='Q64', name='El Sauce Observatory', short_name='El Sauce', lat=-30.47, lon=-70.77, altitude=1500.0
        )

        with patch('solsys_code.campaign_utils.cache.set') as mock_set:
            campaign_utils.build_site_candidates()

        _args, kwargs = mock_set.call_args
        self.assertEqual(kwargs.get('timeout'), campaign_utils.MPC_CANDIDATE_FALLBACK_TTL_SECONDS)
        self.assertLess(
            campaign_utils.MPC_CANDIDATE_FALLBACK_TTL_SECONDS, campaign_utils.MPC_CANDIDATE_CACHE_TTL_SECONDS
        )

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_full_pool_uses_long_ttl(self, mock_query_all):
        """The happy path (MPC fetch succeeded) still caches for the full 24h TTL."""
        mock_query_all.return_value = BULK_MPC_FIXTURE

        with patch('solsys_code.campaign_utils.cache.set') as mock_set:
            campaign_utils.build_site_candidates()

        _args, kwargs = mock_set.call_args
        self.assertEqual(kwargs.get('timeout'), campaign_utils.MPC_CANDIDATE_CACHE_TTL_SECONDS)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_caches_result_under_fixed_key(self, mock_query_all):
        mock_query_all.return_value = BULK_MPC_FIXTURE

        pool = campaign_utils.build_site_candidates()

        self.assertEqual(cache.get('mpc_obscode_candidates'), pool)
        mock_query_all.assert_called_once()

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_second_call_reuses_cache_not_query_all(self, mock_query_all):
        mock_query_all.return_value = BULK_MPC_FIXTURE
        campaign_utils.build_site_candidates()

        # On the second call, a raise here would propagate if the cache were bypassed.
        mock_query_all.side_effect = requests.exceptions.RequestException
        pool_second = campaign_utils.build_site_candidates()

        self.assertIn('C65', pool_second)
        mock_query_all.assert_called_once()

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_cold_cache_mpc_failure_falls_back_to_local_pool(self, mock_query_all):
        mock_query_all.side_effect = requests.exceptions.RequestException
        local = Observatory.objects.create(
            obscode='Q64',
            name='El Sauce Observatory',
            short_name='El Sauce',
            lat=-30.47,
            lon=-70.77,
            altitude=1500.0,
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

        pool = campaign_utils.build_site_candidates()

        self.assertEqual(pool.get(local.obscode), local.obscode)
        self.assertEqual(pool.get(local.name), local.obscode)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_build_site_candidates_excludes_placeholder_observatories(self, mock_query_all):
        """CR-02 (22-REVIEW.md re-review): a tier-3 placeholder Observatory (name prefixed
        NEEDS_REVIEW_NAME_PREFIX) must never surface as a search-suggestion candidate --
        neither by its obscode/short_name nor its 'NEEDS REVIEW: ...' display name -- or a
        staff member could click it and have resolve_site() silently accept it as a genuine
        resolution of itself."""
        mock_query_all.return_value = BULK_MPC_FIXTURE
        Observatory.objects.create(obscode='DCT', name=f'{NEEDS_REVIEW_NAME_PREFIX}DCT', short_name='DCT')

        pool = campaign_utils.build_site_candidates()

        self.assertNotIn('DCT', pool)
        self.assertNotIn(f'{NEEDS_REVIEW_NAME_PREFIX}DCT', pool)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_fuzzy_match_candidates_exact_hit_includes_obscode(self, mock_query_all):
        mock_query_all.return_value = BULK_MPC_FIXTURE
        pool = campaign_utils.build_site_candidates()

        matches = campaign_utils.fuzzy_match_candidates('C65', pool)

        self.assertIn(('C65', 'C65'), matches)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_fuzzy_match_candidates_near_typo_scores_above_cutoff(self, mock_query_all):
        mock_query_all.return_value = BULK_MPC_FIXTURE
        pool = campaign_utils.build_site_candidates()

        matches = campaign_utils.fuzzy_match_candidates('Siding Spring Observatry', pool)

        self.assertIn(('Siding Spring Observatory', 'W89'), matches)

    @patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
    def test_fuzzy_match_candidates_nickname_returns_no_matches(self, mock_query_all):
        """Pitfall 2: 'DCT' cannot bridge to 'Lowell Discovery Telescope' via difflib, even
        against the widened pool -- the free-text/create-new fallback is load-bearing."""
        mock_query_all.return_value = BULK_MPC_FIXTURE
        pool = campaign_utils.build_site_candidates()

        matches = campaign_utils.fuzzy_match_candidates('DCT', pool)

        self.assertEqual(matches, [])


@override_settings(CACHES=ISOLATED_TEST_CACHES)
class TestSiteSearchCacheIsolationRegression(TestCase):
    """Regression (debug/site-search-degraded-pool-recurrence, bug #3): the campaign test suite
    must NEVER read, write, or clear the shared FileBasedCache the dev runserver serves live
    site-search from.

    Root cause of bug #3: settings.CACHES is a FileBasedCache at tempfile.gettempdir() (/tmp),
    shared across processes, and Django does NOT swap the cache backend for tests the way it
    swaps the database. So the campaign tests -- which call cache.clear() in setUp/tearDown and
    the real build_site_candidates() (writing the 'mpc_obscode_candidates' key) -- were
    read/writing/wiping the exact cache entry the runserver depends on. Running the suite to
    verify a fix WIPED the runserver's warmed ~5,700-entry MPC candidate pool, so live
    site-search reverted to "No matches" until the next successful cold rebuild.

    This test writes a sentinel directly into the real /tmp FileBasedCache (a fresh handle,
    deliberately NOT the overridden default alias -- so it points at the runserver's actual
    cache), then performs the exact cache operations the suite performs under
    @override_settings(CACHES=ISOLATED_TEST_CACHES), and asserts the sentinel is untouched.
    Without the isolation decorators this fails: cache.clear() on the shared FileBasedCache
    wipes the sentinel (and the runserver's real pool) too.
    """

    def _runserver_file_cache(self):
        # A direct FileBasedCache handle at the real settings location -- unaffected by this
        # class's CACHES override (which only rebinds the `caches` registry / default proxy).
        import tempfile

        from django.core.cache.backends.filebased import FileBasedCache

        return FileBasedCache(tempfile.gettempdir(), {})

    def test_suite_cache_operations_do_not_touch_the_shared_runserver_file_cache(self):
        import uuid

        runserver_cache = self._runserver_file_cache()
        sentinel_key = f'runserver_warmed_pool_sentinel_{uuid.uuid4().hex}'
        sentinel_value = {'G37': 'G37', 'Lowell Discovery Telescope': 'G37'}
        runserver_cache.set(sentinel_key, sentinel_value, timeout=300)
        try:
            # Confirm the isolation override is actually in effect for the default cache the
            # tests (and campaign_utils) use -- a LocMemCache, not the shared FileBasedCache.
            from django.core.cache import caches
            from django.core.cache.backends.locmem import LocMemCache

            self.assertIsInstance(caches['default'], LocMemCache)

            # The exact operations the campaign test classes perform every run:
            cache.clear()  # setUp/tearDown of TestSiteFuzzyMatch / ThrottleTest / SiteSearchViewTest
            cache.set(campaign_utils._MPC_CANDIDATE_CACHE_KEY, {'X': 'X'})  # a real build_site_candidates() write
            cache.set('site_search_throttle:1.2.3.4', 1)  # a throttle write

            # Under isolation all of the above hit LocMemCache; the runserver's real /tmp pool
            # is untouched. Without the isolation decorators, cache.clear() would have wiped it.
            self.assertEqual(runserver_cache.get(sentinel_key), sentinel_value)
        finally:
            runserver_cache.delete(sentinel_key)


class TestApprovalQueueSiteSearchWidget(CampaignApprovalTestBase):
    """D-10/22-REVIEWS.md findings 1 and 7: the approval queue's pending-row Site column is
    a live-search widget (hx-get to campaigns:site_search), replacing the static datalist
    from Plan 21-03, while keeping the "Create new Observatory" link and stored-XSS
    escaping coverage.

    ``build_site_candidates`` is patched at the view's import site
    (``solsys_code.campaign_views.build_site_candidates``) so every case here is
    deterministic and never hits the live MPC API -- mirrors ``TestSiteFuzzyMatch``'s
    mocking-boundary discipline, but at the view layer instead of the helper layer.
    """

    def setUp(self):
        cache.clear()
        self.client.login(username='staffcoordinator', password='pw')
        self._candidate_pool = campaign_utils._flatten_mpc_candidates(BULK_MPC_FIXTURE)
        patcher = patch('solsys_code.campaign_views.build_site_candidates', return_value=self._candidate_pool)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        cache.clear()

    def test_unresolved_pending_row_renders_live_search_widget_and_create_link(self):
        run = self._make_pending_run(site=None, site_raw='F65', site_needs_review=False)

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertIn('name="site_selection"', content)
        self.assertIn(f'form="decide-form-{run.pk}"', content)
        self.assertIn('hx-get', content)
        self.assertIn(reverse('campaigns:site_search'), content)
        # The raw corrected trigger string (unescaped -- it sits in the format_html
        # literal, not a substituted attribute) -- 22-REVIEWS.md finding 1.
        self.assertIn('input[this.value.length >= 2] changed delay:300ms', content)
        self.assertNotIn('delay:300ms[', content)
        self.assertIn(f'<div id="site-suggestions-site-input-{run.pk}"', content)
        self.assertIn('Create new Observatory', content)
        self.assertNotIn('<datalist', content)

    def test_click_to_fill_wiring_uses_one_consistent_id(self):
        """22-REVIEWS.md finding 7: for the row's actual pk, 'site-input-{pk}' must appear
        as ALL of -- the input id, the hx-target value, the container div id, and the
        hx-vals input_id value -- or the endpoint's onclick fill silently breaks."""
        run = self._make_pending_run(site=None, site_raw='F65', site_needs_review=False)

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        input_id = f'site-input-{run.pk}'
        self.assertIn(f'id="{input_id}"', content)
        self.assertIn(f'hx-target="#site-suggestions-{input_id}"', content)
        self.assertIn(f'<div id="site-suggestions-{input_id}"', content)
        self.assertIn(f'"input_id": "{input_id}"', content)

    def test_site_raw_script_injection_is_escaped_not_rendered_raw(self):
        """T-21-01: format_html auto-escaping must neutralize a stored-XSS attempt in
        site_raw -- no unescaped <script> tag may reach the response body."""
        self._make_pending_run(site=None, site_raw='<script>alert(1)</script>', site_needs_review=False)

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertNotIn('<script>alert(1)</script>', content)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', content)

    def test_resolved_pending_row_renders_no_site_selection_input(self):
        observatory = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        run = self._make_pending_run(site=observatory, site_raw='F65', site_needs_review=False)

        response = self.client.get(reverse('campaigns:approval_queue'))

        content = response.content.decode()
        self.assertNotIn(f'id="site-input-{run.pk}"', content)
        self.assertIn('FTS', content)

    def test_decided_table_renders_no_site_selection_input(self):
        self._make_pending_run(
            site=None,
            site_raw='F65',
            site_needs_review=False,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

        response = self.client.get(reverse('campaigns:approval_queue'))

        self.assertNotIn('name="site_selection"', response.content.decode())
