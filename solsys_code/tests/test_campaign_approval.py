"""Tests for the staff-facing approval-queue write path (SUBMIT-03 / CAL-01/02/03 / D-01/D-02).

Covers: staff-only gating on both the approval-queue GET and the decision-endpoint POST
(never a soft-filter -- a redirect, never 200-with-pending-content, per 16-RESEARCH.md Pitfall
7); the atomic conditional approve/reject transition and its proven double-approve no-op
(SUBMIT-03); the CAMPAIGN:{pk} CalendarEvent projection that fires only when
telescope_instrument + ut_start + ut_end are all present (CAL-01/CAL-02); no duplicate event
and no ``modified`` churn on re-approve (CAL-03); and the reject path (no event created).

Uses ``TargetList.objects.create(...)`` for the campaign container and plain
``CampaignRun.objects.create(...)`` fixtures. This module never fixtures an individual
``tom_targets.models.Target`` at all (CampaignRun.target is left unset throughout), so
CLAUDE.md's non-sidereal-only target-factory convention doesn't even arise here.
"""

from datetime import datetime, timezone

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.campaign_tables import ApprovalQueueTable, CampaignRunTable
from solsys_code.models import CampaignRun

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
            'ut_start': datetime(2026, 8, 1, 3, 0, 0, tzinfo=timezone.utc),
            'ut_end': datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc),
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
    """CAL-01/CAL-02: approving a telescope+date-range run projects a CAMPAIGN:{pk} event."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_approve_with_full_window_creates_calendar_event(self):
        run = self._make_pending_run()
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
        event = CalendarEvent.objects.get(url=f'CAMPAIGN:{run.pk}')
        self.assertEqual(event.target_list_id, self.campaign.pk)
        self.assertEqual(event.start_time, run.ut_start)
        self.assertEqual(event.end_time, run.ut_end)
        self.assertEqual(event.telescope, run.telescope_instrument)

    def test_approve_without_ut_end_creates_no_calendar_event(self):
        """Pitfall 2: CalendarEvent.start_time/end_time are non-nullable -- ut_end missing means
        no event, not a fabricated end_time."""
        run = self._make_pending_run(ut_end=None)
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

    def test_approve_without_ut_start_creates_no_calendar_event(self):
        run = self._make_pending_run(ut_start=None)
        self.client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve'})
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


class TestCalendarNoChurn(CampaignApprovalTestBase):
    """CAL-03: re-approve produces no duplicate event and no modified churn."""

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
