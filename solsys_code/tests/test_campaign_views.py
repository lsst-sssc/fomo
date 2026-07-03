"""Tests for the per-campaign table read path (VIEW-01/03/04).

RED state: `campaign_views.py`/`campaign_urls.py` do not exist yet when this module is first
written (Task 1 of 15-01-PLAN.md) -- every test here MUST fail (URL reverse errors) until
Task 2/3 build the table/filter/view/URL wiring. A passing run at Task 1 is a red flag that
these tests are not actually exercising the new views.

Uses `TargetList.objects.create(...)` (never `SiderealTargetFactory` -- CLAUDE.md mandates
non-sidereal-only fixtures for this project) and a plain `is_staff=True` `User` fixture (no
prior `is_staff` test precedent exists in this codebase per 15-RESEARCH.md Wave 0 Gaps).
"""

from datetime import date, datetime, timedelta, timezone

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList

from solsys_code.models import CampaignRun

# Cycle of run_status values for the "filler" rows -- deliberately excludes PLANNED/OBSERVED/
# CANCELLED, which are pinned to specific rows below so the multi-select filter test (VIEW-04)
# has a small, exactly-known expected result set.
_CYCLE_RUN_STATUSES = [
    CampaignRun.RunStatus.REQUESTED,
    CampaignRun.RunStatus.REDUCED,
    CampaignRun.RunStatus.PUBLISHED,
    CampaignRun.RunStatus.NOT_AWARDED,
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE,
]
_CYCLE_APPROVAL_STATUSES = [
    CampaignRun.ApprovalStatus.PENDING_REVIEW,
    CampaignRun.ApprovalStatus.APPROVED,
    CampaignRun.ApprovalStatus.REJECTED,
]

_TOTAL_RUNS = 30  # > 25 so pagination (D-11) is genuinely exercised
_BASE_DATE = date(2026, 6, 1)
_BASE_UT = datetime(2026, 6, 1, 8, 0, 0, tzinfo=timezone.utc)

CONTACT_PERSON = 'Jane Coordinator'
CONTACT_EMAIL = 'jane@example.org'


class CampaignViewTestBase(TestCase):
    """Shared fixture: one campaign with 30 CampaignRun rows, one empty campaign, one staff user."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.empty_campaign = TargetList.objects.create(name='Empty Campaign')
        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)

        cls.runs = []
        for i in range(_TOTAL_RUNS):
            obs_date = _BASE_DATE + timedelta(days=i)
            ut_start = _BASE_UT + timedelta(days=i)
            kwargs = {
                'campaign': cls.campaign,
                'telescope_instrument': f'FTN/MuSCAT3-{i}',
                'obs_date': obs_date,
                'ut_start': ut_start,
            }
            if i == _TOTAL_RUNS - 1:
                # Most-recent row (highest obs_date -- always page 1, first row per D-10).
                # Carries the seeded contact PII and open_to_collaboration=True so VIEW-03/
                # VIEW-04 assertions never depend on which pagination page a row lands on.
                kwargs.update(
                    run_status=CampaignRun.RunStatus.PLANNED,
                    approval_status=CampaignRun.ApprovalStatus.APPROVED,
                    contact_person=CONTACT_PERSON,
                    contact_email=CONTACT_EMAIL,
                    open_to_collaboration=True,
                )
            elif i == _TOTAL_RUNS - 2:
                kwargs.update(
                    run_status=CampaignRun.RunStatus.OBSERVED,
                    approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
                )
            elif i == _TOTAL_RUNS - 3:
                kwargs.update(
                    run_status=CampaignRun.RunStatus.CANCELLED,
                    approval_status=CampaignRun.ApprovalStatus.REJECTED,
                )
            else:
                kwargs.update(
                    run_status=_CYCLE_RUN_STATUSES[i % len(_CYCLE_RUN_STATUSES)],
                    approval_status=_CYCLE_APPROVAL_STATUSES[i % len(_CYCLE_APPROVAL_STATUSES)],
                )
            cls.runs.append(CampaignRun.objects.create(**kwargs))

        cls.most_recent_run = cls.runs[-1]

    def table_url(self, campaign=None):
        return reverse('campaigns:table', kwargs={'pk': (campaign or self.campaign).pk})

    def list_url(self):
        return reverse('campaigns:list')

    @staticmethod
    def _row_value(record, field):
        """Read a field from a table row's record -- a dict for non-staff (.values()) rows,
        a model instance for staff rows (RESEARCH.md Pitfall 2 dict-vs-model-instance)."""
        if isinstance(record, dict):
            return record[field]
        return getattr(record, field)


class TestCampaignRunTableView(CampaignViewTestBase):
    """VIEW-01: table lists all runs for a campaign, 25/page, default-sorted obs_date desc."""

    def test_anonymous_get_returns_200(self):
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)

    def test_first_page_shows_25_rows_and_second_page_exists(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        self.assertEqual(len(table.page.object_list), 25)
        self.assertGreaterEqual(table.paginator.num_pages, 2)

    def test_default_load_shows_every_seeded_run_status_value(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        seen_statuses = {self._row_value(row.record, 'run_status') for row in table.page.object_list}
        self.assertEqual(seen_statuses, set(CampaignRun.RunStatus.values))

    def test_default_sort_is_obs_date_descending(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        first_record = table.page.object_list[0].record
        self.assertEqual(self._row_value(first_record, 'obs_date'), self.most_recent_run.obs_date)


class TestContactFieldGating(CampaignViewTestBase):
    """VIEW-03: contact_person/contact_email visible only to staff -- proven via context AND content."""

    def test_anonymous_context_rows_have_no_contact_fields(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        for row in table.page.object_list:
            record = row.record
            self.assertIsInstance(record, dict, 'Anonymous rows must be dicts from a .values() queryset')
            self.assertNotIn('contact_person', record)
            self.assertNotIn('contact_email', record)

    def test_anonymous_content_has_no_contact_strings(self):
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertNotIn(CONTACT_PERSON, content)
        self.assertNotIn(CONTACT_EMAIL, content)

    def test_staff_content_includes_contact_strings(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertIn(CONTACT_PERSON, content)
        self.assertIn(CONTACT_EMAIL, content)

    def test_staff_context_rows_have_contact_fields(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        table = response.context['table']
        contact_persons = {self._row_value(row.record, 'contact_person') for row in table.page.object_list}
        self.assertIn(CONTACT_PERSON, contact_persons)


class TestCampaignRunFilterSet(CampaignViewTestBase):
    """VIEW-04: run_status multi-select (OR) + open_to_collaboration boolean; unfiltered default."""

    def test_default_unfiltered_shows_all_rows(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        self.assertEqual(table.paginator.count, _TOTAL_RUNS)

    def test_run_status_multiselect_or_semantics(self):
        response = self.client.get(
            self.table_url(),
            {'run_status': [CampaignRun.RunStatus.PLANNED, CampaignRun.RunStatus.OBSERVED]},
        )
        table = response.context['table']
        self.assertEqual(table.paginator.count, 2)
        for row in table.page.object_list:
            value = self._row_value(row.record, 'run_status')
            self.assertIn(value, [CampaignRun.RunStatus.PLANNED, CampaignRun.RunStatus.OBSERVED])

    def test_open_to_collaboration_filter(self):
        response = self.client.get(self.table_url(), {'open_to_collaboration': 'true'})
        table = response.context['table']
        self.assertEqual(table.paginator.count, 1)
        only_row_record = table.page.object_list[0].record
        self.assertEqual(self._row_value(only_row_record, 'pk'), self.most_recent_run.pk)


class TestCampaignListView(CampaignViewTestBase):
    """D-03: campaigns list page lists only TargetLists with >= 1 CampaignRun."""

    def test_lists_campaign_with_runs(self):
        response = self.client.get(self.list_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.campaign.name)

    def test_does_not_list_campaign_with_zero_runs(self):
        response = self.client.get(self.list_url())
        self.assertNotContains(response, self.empty_campaign.name)
