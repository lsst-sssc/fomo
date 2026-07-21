"""Tests for the per-campaign table read path (VIEW-01/03/04).

RED state: `campaign_views.py`/`campaign_urls.py` do not exist yet when this module is first
written (Task 1 of 15-01-PLAN.md) -- every test here MUST fail (URL reverse errors) until
Task 2/3 build the table/filter/view/URL wiring. A passing run at Task 1 is a red flag that
these tests are not actually exercising the new views.

Uses `TargetList.objects.create(...)` (never `SiderealTargetFactory` -- CLAUDE.md mandates
non-sidereal-only fixtures for this project) and a plain `is_staff=True` `User` fixture (no
prior `is_staff` test precedent exists in this codebase per 15-RESEARCH.md Wave 0 Gaps).
"""

from datetime import date, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.campaign_tables import CampaignRunTable
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
            window_date = _BASE_DATE + timedelta(days=i)
            kwargs = {
                'campaign': cls.campaign,
                'telescope_instrument': f'FTN/MuSCAT3-{i}',
                'window_start': window_date,
                'window_end': window_date,
            }
            if i == _TOTAL_RUNS - 1:
                # Most-recent row (highest window_start -- always page 1, first row per D-10).
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
    """VIEW-01: table lists all runs for a campaign, 25/page, default-sorted window_start desc.

    These assertions are about generic table mechanics (pagination, sort, full row-status
    coverage), not approval-status visibility gating -- exercised via the staff client so
    D-09's non-staff `.exclude(approval_status=PENDING_REVIEW)` (added in Plan 04) doesn't
    change the expected row counts here. D-09 visibility itself is covered separately by
    `TestNonStaffPendingReviewHidden`.
    """

    def test_anonymous_get_returns_200(self):
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)

    def test_first_page_shows_25_rows_and_second_page_exists(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        table = response.context['table']
        self.assertEqual(len(table.page.object_list), 25)
        self.assertGreaterEqual(table.paginator.num_pages, 2)

    def test_default_load_shows_every_seeded_run_status_value(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        table = response.context['table']
        seen_statuses = {self._row_value(row.record, 'run_status') for row in table.page.object_list}
        self.assertEqual(seen_statuses, set(CampaignRun.RunStatus.values))

    def test_default_sort_is_window_start_desc_tbd_last(self):
        """D-04: resolved rows lead (most recent window_start first); a TBD row (both
        window fields null) sorts last -- portably across backends via
        F('window_start').desc(nulls_last=True), never relying on the DB's own implicit
        NULL-ordering default (SQLite/PostgreSQL disagree on that direction)."""
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='TBD-Telescope',
            contact_person='TBD Coordinator',
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        response = self.client.get(self.table_url())
        table = response.context['table']
        first_record = table.page.object_list[0].record
        self.assertEqual(self._row_value(first_record, 'window_start'), self.most_recent_run.window_start)

        last_page_rows = list(table.paginator.page(table.paginator.num_pages).object_list)
        last_record = last_page_rows[-1].record
        self.assertIsNone(self._row_value(last_record, 'window_start'))


class TestWindowColumnRendering(TestCase):
    """D-03/D-05: TBD badge, single-date, and range-arrow rendering for the window column.

    Exercises CampaignRunTable.render_window_start() directly (no HTTP round trip needed --
    this is purely about the render method's output, mirroring test_campaign_approval.py's
    TestApprovalQueueSiteVisibility precedent for render_site()).
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Render Campaign')

    def test_tbd_row_renders_tbd_indicator(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign, telescope_instrument='TBD Scope', contact_person='Render Contact'
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertIn('TBD', cell)

    def test_tbd_row_with_raw_text_renders_tooltip(self):
        """D-08: a TBD row with original_obs_date_raw set carries a title tooltip."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='TBD Scope',
            contact_person='Render Contact Raw',
            original_obs_date_raw='TBD pending Cycle 2',
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertIn('title="TBD pending Cycle 2"', cell)

    def test_tbd_row_with_blank_raw_text_renders_no_title(self):
        """D-08: a blank original_obs_date_raw renders the plain TBD badge, no title attribute."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='TBD Scope',
            contact_person='Render Contact Blank',
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertNotIn('title=', cell)

    def test_tbd_row_with_markup_raw_text_is_escaped(self):
        """T-20-03: angle-bracket markup in original_obs_date_raw is HTML-escaped, not rendered live."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='TBD Scope',
            contact_person='Render Contact Markup',
            original_obs_date_raw='<script>alert(1)</script>',
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertNotIn('<script>', cell)
        self.assertIn('&lt;script&gt;', cell)

    def test_range_row_renders_arrow(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='Range Scope',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 15),
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertIn('-&gt;', cell)

    def test_single_night_row_renders_one_date(self):
        d = date(2026, 8, 1)
        run = CampaignRun.objects.create(
            campaign=self.campaign, telescope_instrument='Single Night Scope', window_start=d, window_end=d
        )
        cell = CampaignRunTable([run]).rows[0].get_cell('window_start')
        self.assertEqual(cell, d)


class TestContactFieldGating(CampaignViewTestBase):
    """VIEW-03: contact_person/contact_email visible only to staff -- proven via context AND content."""

    def test_anonymous_context_rows_have_no_contact_fields(self):
        """VIEW-05: contact_person/contact_email keys are now always present in the non-staff
        .values() dict (queryset-level Case/When annotation, T-21-02), but blank unless the
        row opted in -- none of this base fixture's rows have contact_public_opt_in=True, so
        every anonymous row's value is the empty string, never the raw PII.
        """
        response = self.client.get(self.table_url())
        table = response.context['table']
        for row in table.page.object_list:
            record = row.record
            self.assertIsInstance(record, dict, 'Anonymous rows must be dicts from a .values() queryset')
            self.assertEqual(record['contact_person'], '')
            self.assertEqual(record['contact_email'], '')

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


class TestContactPublicOptIn(CampaignViewTestBase):
    """VIEW-05/T-21-02: opted-in runs expose contact PII to anonymous visitors; opted-out
    runs never emit it from the non-staff SQL SELECT (queryset-level gate, not template-only).
    """

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        cls.opted_in_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Opted-In Scope',
            window_start=_BASE_DATE + timedelta(days=100),
            window_end=_BASE_DATE + timedelta(days=100),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            contact_person='Opted In Person',
            contact_email='optedin@example.org',
            contact_public_opt_in=True,
        )
        cls.opted_out_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Opted-Out Scope',
            window_start=_BASE_DATE + timedelta(days=101),
            window_end=_BASE_DATE + timedelta(days=101),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            contact_person='Opted Out Person',
            contact_email='optedout@example.org',
            contact_public_opt_in=False,
        )

    def _non_staff_values_row(self, pk):
        """Reach into the raw non-staff .values() queryset directly (not via the table/HTTP
        response) to prove the SQL SELECT itself, not just rendered HTML (T-21-02).
        """
        from solsys_code.campaign_views import CampaignRunTableView

        view = CampaignRunTableView()
        view.kwargs = {'pk': self.campaign.pk}
        view.request = type('Req', (), {'user': type('U', (), {'is_staff': False})()})()
        return view.get_queryset().get(pk=pk)

    def test_opted_in_row_exposes_contact_in_non_staff_values(self):
        row = self._non_staff_values_row(self.opted_in_run.pk)
        self.assertEqual(row['contact_person'], 'Opted In Person')
        self.assertEqual(row['contact_email'], 'optedin@example.org')

    def test_opted_out_row_blanks_contact_in_non_staff_values(self):
        row = self._non_staff_values_row(self.opted_out_run.pk)
        self.assertEqual(row['contact_person'], '')
        self.assertEqual(row['contact_email'], '')

    def test_opted_in_content_visible_to_anonymous_visitor(self):
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertIn('Opted In Person', content)
        self.assertIn('optedin@example.org', content)

    def test_opted_out_content_not_visible_to_anonymous_visitor(self):
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertNotIn('Opted Out Person', content)
        self.assertNotIn('optedout@example.org', content)

    def test_staff_sees_both_regardless_of_opt_in(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertIn('Opted In Person', content)
        self.assertIn('Opted Out Person', content)

    def test_allowed_fields_for_non_staff_does_not_list_contact_fields(self):
        """RESEARCH.md Anti-Pattern: contact_person/contact_email must arrive via the
        Case/When F() kwargs in .values(), never be added directly to the allow-list.
        """
        from solsys_code.campaign_views import ALLOWED_FIELDS_FOR_NON_STAFF

        self.assertNotIn('contact_person', ALLOWED_FIELDS_FOR_NON_STAFF)
        self.assertNotIn('contact_email', ALLOWED_FIELDS_FOR_NON_STAFF)


class TestCampaignRunFilterSet(CampaignViewTestBase):
    """VIEW-04: run_status multi-select (OR) + open_to_collaboration boolean; unfiltered default.

    Uses the staff client throughout -- these assertions are about filter semantics over the
    full 30-row fixture, not D-09 approval-status visibility gating (covered separately by
    `TestNonStaffPendingReviewHidden`).
    """

    def test_default_unfiltered_shows_all_rows(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        table = response.context['table']
        self.assertEqual(table.paginator.count, _TOTAL_RUNS)

    def test_run_status_multiselect_or_semantics(self):
        self.client.force_login(self.staff_user)
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

    def test_pending_count_in_context(self):
        response = self.client.get(self.list_url())
        expected_pending = sum(
            1 for run in self.runs if run.approval_status == CampaignRun.ApprovalStatus.PENDING_REVIEW
        )
        self.assertEqual(response.context['pending_count'], expected_pending)


class TestNonStaffPendingReviewHidden(CampaignViewTestBase):
    """D-09/SUBMIT-02: non-staff see approved AND rejected rows; only pending_review is hidden."""

    def test_anonymous_queryset_excludes_pending_review(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        seen_statuses = {self._row_value(row.record, 'approval_status') for row in table.page.object_list}
        self.assertNotIn(CampaignRun.ApprovalStatus.PENDING_REVIEW, seen_statuses)

    def test_anonymous_queryset_still_shows_approved_and_rejected(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        seen_statuses = {self._row_value(row.record, 'approval_status') for row in table.page.object_list}
        self.assertIn(CampaignRun.ApprovalStatus.APPROVED, seen_statuses)
        self.assertIn(CampaignRun.ApprovalStatus.REJECTED, seen_statuses)

    def test_anonymous_total_row_count_excludes_pending(self):
        response = self.client.get(self.table_url())
        table = response.context['table']
        expected_count = sum(1 for run in self.runs if run.approval_status != CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self.assertEqual(table.paginator.count, expected_count)

    def test_staff_sees_all_approval_statuses_including_pending(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        table = response.context['table']
        seen_statuses = {self._row_value(row.record, 'approval_status') for row in table.page.object_list}
        self.assertEqual(seen_statuses, set(CampaignRun.ApprovalStatus.values))


class TestCampaignDetailIntegration(CampaignViewTestBase):
    """VIEW-02: target-detail page shows one campaign link per matching campaign (D-01/D-02),
    discovered via TargetList membership; the navbar exposes a "Campaigns" entry (D-03)."""

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        # cls.campaign (base fixture) already carries 30 CampaignRun rows, so it qualifies as
        # a campaign. Add a member Target via TargetList membership (D-01 -- never via
        # CampaignRun's optional target FK) and a second, unrelated Target in no campaign.
        cls.member_target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.member_target)
        cls.other_target = NonSiderealTargetFactory.create()

    def test_target_detail_shows_campaign_link(self):
        response = self.client.get(reverse('tom_targets:detail', kwargs={'pk': self.member_target.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.table_url())
        self.assertContains(response, f'View {self.campaign.name} Runs')

    def test_target_detail_no_campaign_for_unrelated_target(self):
        response = self.client.get(reverse('tom_targets:detail', kwargs={'pk': self.other_target.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, self.table_url())

    def test_navbar_shows_campaigns_entry(self):
        response = self.client.get(self.list_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'<a class="nav-link" href="{self.list_url()}">Campaigns</a>')
