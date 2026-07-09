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
from unittest.mock import patch

import requests
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.campaign_tables import ApprovalQueueTable, CampaignRunTable
from solsys_code.campaign_utils import resolve_site
from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import sun_event

CONTACT_PERSON = 'Jane Coordinator'
CONTACT_EMAIL = 'jane@example.org'


class CampaignApprovalTestBase(TestCase):
    """Shared fixture: one campaign, one staff user, one non-staff user."""

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
