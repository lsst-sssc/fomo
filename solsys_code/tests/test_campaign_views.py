"""Tests for the per-campaign table read path (VIEW-01/03/04).

RED state: `campaign_views.py`/`campaign_urls.py` do not exist yet when this module is first
written (Task 1 of 15-01-PLAN.md) -- every test here MUST fail (URL reverse errors) until
Task 2/3 build the table/filter/view/URL wiring. A passing run at Task 1 is a red flag that
these tests are not actually exercising the new views.

Uses `TargetList.objects.create(...)` (never `SiderealTargetFactory` -- CLAUDE.md mandates
non-sidereal-only fixtures for this project) and a plain `is_staff=True` `User` fixture (no
prior `is_staff` test precedent exists in this codebase per 15-RESEARCH.md Wave 0 Gaps).
"""

from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from uuid import uuid4

from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import connection
from django.db.models.signals import post_save
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import campaign_tally
from solsys_code.allocation_projector import allocation_night_url
from solsys_code.campaign_tables import CampaignRunTable, _campaign_run_row_id
from solsys_code.models import CampaignRun, CampaignRunObservation, ProposalTimeAllocation
from solsys_code.observation_projector import receiver_on_record_save
from solsys_code.solsys_code_observatory.models import Observatory

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


class TestTelescopeClassVisibleSourceStaffOnly(CampaignViewTestBase):
    """D-18/Phase 27 CANON-01/02: telescope_class is visible to non-staff; source stays
    staff-only, and the existing non-staff approval-gating behaviour is unchanged
    (success criterion 1).
    """

    @classmethod
    def setUpTestData(cls) -> None:
        super().setUpTestData()
        cls.class_wide_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='LCO 1m',
            window_start=_BASE_DATE + timedelta(days=200),
            window_end=_BASE_DATE + timedelta(days=200),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
            source=CampaignRun.Source.CSV_IMPORT,
        )
        cls.pending_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Should Stay Hidden Scope',
            window_start=_BASE_DATE + timedelta(days=201),
            window_end=_BASE_DATE + timedelta(days=201),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

    def test_allowed_fields_includes_telescope_class_excludes_source(self):
        from solsys_code.campaign_views import ALLOWED_FIELDS_FOR_NON_STAFF

        self.assertIn('telescope_class', ALLOWED_FIELDS_FOR_NON_STAFF)
        self.assertNotIn('source', ALLOWED_FIELDS_FOR_NON_STAFF)

    def test_non_staff_queryset_selects_telescope_class(self):
        """D-18: telescope_class is present in the non-staff .values() queryset -- the SQL
        SELECT itself fetches it -- proven directly against the queryset the same way
        TestContactPublicOptIn._non_staff_values_row() proves contact-field gating.
        """
        from solsys_code.campaign_views import CampaignRunTableView

        view = CampaignRunTableView()
        view.kwargs = {'pk': self.campaign.pk}
        view.request = type('Req', (), {'user': type('U', (), {'is_staff': False})()})()
        row = view.get_queryset().get(pk=self.class_wide_run.pk)
        self.assertEqual(row['telescope_class'], CampaignRun.TelescopeClass.ONE_M0)

    def test_non_staff_response_body_renders_telescope_class(self):
        """WR-02: fetching telescope_class into the queryset is not what D-18 promised --
        a non-staff READER has to be able to see it. Asserted against the rendered response
        body, not the queryset, because the previous version of this test passed while the
        field was fetched and then silently discarded by every table column.
        """
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertIn('Telescope class', content)  # column header
        self.assertIn('>1m0<', content)  # the run's own stored value, rendered

    def test_staff_and_non_staff_render_telescope_class_identically(self):
        """WR-02: model-instance rows (staff) would otherwise get django-tables2's automatic
        get_telescope_class_display() label while dict rows (non-staff) get the raw code --
        two reader classes seeing different text for the same run. render_telescope_class
        resolves the raw code for both.
        """
        anonymous = self.client.get(self.table_url()).content.decode()
        self.client.force_login(self.staff_user)
        staff = self.client.get(self.table_url()).content.decode()
        for content in (anonymous, staff):
            self.assertIn('>1m0<', content)
            self.assertNotIn('>1m0 class allocation<', content)

    def test_non_staff_response_body_never_exposes_source(self):
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertNotIn('csv_import', content)
        self.assertNotIn('CSV import', content)

    def test_non_staff_response_still_excludes_pending_review_run(self):
        """Regression guard for success criterion 1: this plan's ALLOWED_FIELDS_FOR_NON_STAFF
        edit leaves the existing non-staff approval-gating queryset behaviour unchanged.
        """
        response = self.client.get(self.table_url())
        content = response.content.decode()
        self.assertNotIn('Should Stay Hidden Scope', content)


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


class TestCampaignListSiteReviewEntryPoint(CampaignViewTestBase):
    """27.1-03: the campaign-list staff banner is driven by either queue -- pending_count or
    site_review_count -- not pending_count alone (T-27.1-08 mitigation for the widened,
    NESTED {% if %} gate in campaign_list.html)."""

    def _clear_pending(self):
        """Approve every PENDING_REVIEW run in the base fixture, so pending_count is 0."""
        CampaignRun.objects.filter(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW).update(
            approval_status=CampaignRun.ApprovalStatus.APPROVED
        )

    def _flag_one_for_site_review(self):
        """Set site_needs_review=True on one already-APPROVED run, so site_review_count is 1."""
        run = CampaignRun.objects.filter(approval_status=CampaignRun.ApprovalStatus.APPROVED).first()
        run.site_needs_review = True
        run.save(update_fields=['site_needs_review'])
        return run

    def test_staff_zero_pending_one_site_review_row(self):
        self._clear_pending()
        self._flag_one_for_site_review()
        self.client.force_login(self.staff_user)
        response = self.client.get(self.list_url())
        # Precondition assertion (plan-mandated): the test must not pass for the wrong reason.
        self.assertEqual(response.context['pending_count'], 0)
        self.assertEqual(response.context['site_review_count'], 1)
        self.assertContains(response, reverse('campaigns:approval_queue'))
        self.assertContains(response, 'needing site review')
        self.assertNotContains(response, 'pending review')

    def test_staff_some_pending_zero_site_review_rows(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.list_url())
        self.assertGreater(response.context['pending_count'], 0)
        self.assertEqual(response.context['site_review_count'], 0)
        self.assertContains(response, reverse('campaigns:approval_queue'))
        self.assertContains(response, 'pending review')
        self.assertNotContains(response, 'needing site review')

    def test_staff_both_queues_non_empty(self):
        self._flag_one_for_site_review()
        self.client.force_login(self.staff_user)
        response = self.client.get(self.list_url())
        self.assertGreater(response.context['pending_count'], 0)
        self.assertEqual(response.context['site_review_count'], 1)
        self.assertContains(response, 'pending review')
        self.assertContains(response, 'needing site review')
        self.assertContains(response, reverse('campaigns:approval_queue'), count=1)

    def test_staff_both_queues_empty(self):
        self._clear_pending()
        self.client.force_login(self.staff_user)
        response = self.client.get(self.list_url())
        self.assertEqual(response.context['pending_count'], 0)
        self.assertEqual(response.context['site_review_count'], 0)
        self.assertNotContains(response, reverse('campaigns:approval_queue'))

    def test_anonymous_both_queues_non_empty(self):
        self._flag_one_for_site_review()
        response = self.client.get(self.list_url())
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse('campaigns:approval_queue'))
        self.assertNotContains(response, 'needing site review')
        self.assertNotContains(response, 'pending review')

    def test_non_staff_authenticated_both_queues_non_empty(self):
        self._flag_one_for_site_review()
        non_staff_user = User.objects.create_user(username='regularvisitor', password='pw', is_staff=False)
        self.client.force_login(non_staff_user)
        response = self.client.get(self.list_url())
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse('campaigns:approval_queue'))
        self.assertNotContains(response, 'needing site review')
        self.assertNotContains(response, 'pending review')

    def test_site_review_count_agrees_with_approval_queue_page(self):
        """The count shown on the campaign list and the rows the approval queue itself lists
        come from one definition (runs_needing_site_review()), so they cannot drift apart."""
        self._flag_one_for_site_review()
        self.client.force_login(self.staff_user)
        list_response = self.client.get(self.list_url())
        queue_response = self.client.get(reverse('campaigns:approval_queue'))
        self.assertEqual(
            list_response.context['site_review_count'],
            len(queue_response.context['review_table'].rows),
        )


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


class TestCampaignRunRowAnchor(CampaignViewTestBase):
    """D-13 (Phase 33 Plan 02): every CampaignRunTable row carries an id="run-{pk}"
    anchor -- the landing spot for the calendar decoration's campaign-table link -- for
    staff (model-instance rows) and anonymous (dict rows from .values()) readers alike,
    and a row whose pk cannot be resolved carries no id attribute at all.
    """

    def test_staff_get_contains_run_row_id(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(f'id="run-{self.most_recent_run.pk}"', content)
        self.assertNotIn('id="run-None"', content)

    def test_anonymous_get_contains_run_row_id(self):
        """The dict-row branch -- the case _campaign_run_row_id's Accessor form exists
        for and the one most likely to regress."""
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(f'id="run-{self.most_recent_run.pk}"', content)
        self.assertNotIn('id="run-None"', content)

    def test_row_attrs_callable_returns_none_for_unresolvable_pk(self):
        """Direct unit assertion on the callable itself: an empty dict is the cheapest
        record with no resolvable pk -- must return None, never the string 'run-None'."""
        self.assertIsNone(_campaign_run_row_id({}))

    def test_staff_get_contains_tr_target_highlight_rule(self):
        """CR-01 (Phase 33 Plan 06): the D-13 highlight rule must actually be served in the
        rendered page, not just sit in the template source outside any rendered block."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'tr:target')

    def test_anonymous_get_contains_tr_target_highlight_rule(self):
        """CR-01, anonymous/dict-row branch -- the highlight rule is page-level CSS, not
        per-row markup, so it must be served identically regardless of viewer."""
        response = self.client.get(self.table_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'tr:target')

    def test_empty_campaign_still_serves_tr_target_highlight_rule(self):
        """ANNOT-02 empty edge: the highlight rule ships with the page, not with a row --
        a campaign with zero run rows must still serve it."""
        response = self.client.get(self.table_url(campaign=self.empty_campaign))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'tr:target')


class TestCampaignRunAnchorPagination(CampaignViewTestBase):
    """WR-08 (Phase 33 Plan 06): pins a known, documented limitation as a tested constraint
    rather than a silent dead link. The calendar decoration's #run-{pk} campaign-table link
    carries no page parameter, so it always lands on page 1 -- a run that sorts past page 1
    under CampaignRunTableView's default window_start-descending order has no id="run-{pk}"
    anchor in the unpaginated page-1 document at all. See docs/runbooks/telescope_runs_calendar.rst
    (added by plan 33-08) for the operator-facing wording.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.pagination_campaign = TargetList.objects.create(name='Pagination Campaign')
        cls.pagination_runs = []
        for i in range(26):
            # Descending-distinct window_start values so run i=0 (earliest date, therefore
            # LAST under window_start-descending sort) is the one run that falls onto page 2
            # under per_page: 25.
            window_date = _BASE_DATE + timedelta(days=i)
            cls.pagination_runs.append(
                CampaignRun.objects.create(
                    campaign=cls.pagination_campaign,
                    telescope_instrument=f'FTN/MuSCAT3-page-{i}',
                    window_start=window_date,
                    window_end=window_date,
                    run_status=CampaignRun.RunStatus.PLANNED,
                    approval_status=CampaignRun.ApprovalStatus.APPROVED,
                )
            )
        # Oldest window_start -- sorts last (descending), so it's the sole page-2 row.
        cls.oldest_run = cls.pagination_runs[0]

    def _pagination_table_url(self):
        return reverse('campaigns:table', kwargs={'pk': self.pagination_campaign.pk})

    def test_oldest_run_has_no_anchor_on_page_1(self):
        response = self.client.get(self._pagination_table_url())
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(f'id="run-{self.oldest_run.pk}"', response.content.decode())

    def test_oldest_run_anchor_present_on_page_2(self):
        response = self.client.get(self._pagination_table_url(), {'page': 2})
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'id="run-{self.oldest_run.pk}"', response.content.decode())


class TestGapAnalysisSiteUnknownCount(TestCase):
    """GAPB-01/D-17: the gap page's site-unknown count line renders only when there's
    something to report, and states the count plainly rather than silently dropping an
    observation the analysis could not place on a site."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.site = Observatory.objects.create(
            obscode='F65',
            name='Haleakala (FTN)',
            short_name='FTN',
            lon=-156.2570,
            lat=20.7075,
            altitude=3055.0,
            timezone='Pacific/Honolulu',
        )
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign = TargetList.objects.create(name='Site Unknown Campaign')
        cls.campaign.targets.add(cls.target)
        # A resolved-site approved run is required for gap_analysis_available() to be True
        # (D-14) -- this run's own window plays no other part in either test below.
        CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            site=cls.site,
            window_start=date(2026, 6, 1),
            window_end=date(2026, 6, 1),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )

    def setUp(self):
        # get_or_compute_gap() caches its result for an hour, keyed by campaign/target/site/
        # date-range -- both tests below hit the same key, so a cache hit from whichever test
        # runs first (alphabetical order, not declaration order) would silently make the
        # second test see a stale result. Never share cache state across test methods here.
        cache.clear()

    def _gap_url(self):
        return reverse('campaigns:gap_analysis', kwargs={'pk': self.campaign.pk})

    def test_site_unknown_count_line_renders_when_nonzero(self):
        # 34-01/WR-02 precedent: disconnect the observation projector's post_save receiver
        # around this fixture -- its deliberately site-less `parameters` would otherwise
        # either log as 'unprojectable' or auto-create a CalendarEventMeta this test never
        # asked for.
        post_save.disconnect(
            receiver_on_record_save,
            sender=ObservationRecord,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )
        try:
            ObservationRecord.objects.create(
                target=self.target,
                facility='LCO',
                observation_id='SITEUNKNOWN-1',
                status='COMPLETED',
                scheduled_start=datetime(2026, 6, 5, 22, 0, tzinfo=dt_timezone.utc),
                scheduled_end=datetime(2026, 6, 6, 4, 0, tzinfo=dt_timezone.utc),
                parameters={},
            )
        finally:
            post_save.connect(
                receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

        response = self.client.get(self._gap_url(), {'site': self.site.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '1 observation(s)')
        self.assertContains(response, 'could not')
        self.assertContains(response, 'not ignored')

    def test_site_unknown_count_line_absent_when_zero(self):
        response = self.client.get(self._gap_url(), {'site': self.site.pk})

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'not ignored')


class CampaignTallyViewTestBase(TestCase):
    """Shared fixture for the TALLY-01/02 Progress-column and roll-up tests: one
    resolvable ground Observatory (fixed UTC-10, no DST -- mirrors
    ``test_campaign_tally.CampaignTallyTestBase``) and a campaign, plus run/link factory
    helpers matching that module's own fixture shape so the two test suites agree on what a
    linked record's state means.
    """

    @classmethod
    def setUpTestData(cls) -> None:
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
        cls.campaign = TargetList.objects.create(name='Tally View Campaign')

    def _make_run(self, **overrides) -> CampaignRun:
        kwargs = {
            'campaign': self.campaign,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'telescope_instrument': f'FTN/FLOYDS-{uuid4().hex[:8]}',
            'site': self.site,
            'site_raw': 'F65',
            'window_start': date(2026, 7, 9),
            'window_end': date(2026, 7, 11),
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _link_record(self, run: CampaignRun, group: ObservationGroup | None = None, **overrides) -> ObservationRecord:
        """Create one linked ObservationRecord (COMPLETED/OBSERVED by default), matching
        ``test_campaign_tally.CampaignTallyTestBase._link_record()``'s fixture shape."""
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
        if group is not None:
            group.observation_records.add(record)
        return record

    def _make_alloc_event(self, run: CampaignRun, night: date, *, end_time: datetime) -> CalendarEvent:
        """One ``ALLOC:``-namespaced CalendarEvent for ``run``/``night`` -- mirrors
        ``test_campaign_tally.CampaignTallyTestBase._make_alloc_event()``."""
        return CalendarEvent.objects.create(
            title=f'{run.telescope_instrument} allocation',
            start_time=end_time - timedelta(hours=8),
            end_time=end_time,
            url=allocation_night_url(run, night),
        )


class TestCampaignRunTableProgressColumn(CampaignTallyViewTestBase):
    """TALLY-01/D-08: a public Progress cell on every run row, computed for the whole table
    in one pass and never a per-row query."""

    def setUp(self):
        # Each test's fresh CampaignRun can land on the same pk a prior test's rolled-back
        # transaction used (sqlite reuses rowids after rollback) -- without clearing the
        # cache, a stale campaign_tally cache entry keyed on that reused pk (with the same
        # "no linked records" records_version) would be returned instead of a fresh
        # computation, exactly as TestCampaignRollup.setUp() already guards against.
        cache.clear()

    def test_progress_cell_shows_groups_records_and_ordered_segments(self):
        run = self._make_run()
        group1 = ObservationGroup.objects.create(name='Group A')
        group2 = ObservationGroup.objects.create(name='Group B')
        self._link_record(
            run,
            group=group1,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        self._link_record(
            run,
            group=group2,
            status='PENDING',
            scheduled_start=datetime(2026, 7, 11, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 11, 3, 30, tzinfo=dt_timezone.utc),
        )
        self._link_record(
            run,
            status='WINDOW_EXPIRED',
            scheduled_start=datetime(2026, 7, 12, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 12, 3, 30, tzinfo=dt_timezone.utc),
        )

        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('2 groups', content)
        self.assertIn('3 records', content)
        # Fixed order: observed, scheduled, expired-or-failed, unused -- and never split
        # across lines in a way that reorders the segments.
        marker_positions = [content.find(marker) for marker in ('[O]', '[S]', '[X/F]', '[U]')]
        self.assertGreater(min(marker_positions), -1)
        self.assertEqual(marker_positions, sorted(marker_positions))
        self.assertIn('[O] 1', content)
        self.assertIn('[S] 1', content)
        self.assertIn('[X/F] 1', content)

    def test_run_with_nothing_linked_renders_zeros_not_an_empty_cell(self):
        self._make_run()
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('0 groups', content)
        self.assertIn('0 records', content)
        self.assertIn('[O] 0', content)
        self.assertIn('[S] 0', content)
        self.assertIn('[X/F] 0', content)
        self.assertNotIn('[U] 0', content)  # never zero for an unknown/not-yet-fetched figure

    def test_unused_segment_reads_exact_when_an_allocation_run_has_still_standing_nights(self):
        run = self._make_run(telescope_instrument='FTN/Exact')
        self._make_alloc_event(run, date(2026, 7, 9), end_time=timezone.now() - timedelta(days=1))
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertContains(response, '[U] 1')

    def test_unused_segment_reads_as_an_estimate_for_a_container_run_with_a_fetched_proposal(self):
        self._make_run(telescope_instrument='FTN/Estimate', proposal_code='EST-2026A-001')
        ProposalTimeAllocation.objects.create(
            proposal_code='EST-2026A-001',
            allocation_type='std',
            allocated_hours=30.0,
            used_hours=0.0,
            fetched_at=timezone.now(),
        )
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertContains(response, '[U] ≈3')

    def test_unused_segment_reads_not_yet_known_before_any_fetch(self):
        self._make_run(telescope_instrument='FTN/Unknown', proposal_code='NEVER-FETCHED-001')
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertContains(response, '[U] not yet known')

    def test_anonymous_and_staff_requests_render_the_same_segments(self):
        staff_user = User.objects.create_user(username='progress-staff', password='pw', is_staff=True)
        run = self._make_run()
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})

        anon_content = self.client.get(url).content.decode()

        staff_client = self.client
        staff_client.force_login(staff_user)
        staff_content = staff_client.get(url).content.decode()

        for marker in ('[O] 1', '[S] 0', '[X/F] 0'):
            self.assertIn(marker, anon_content)
            self.assertIn(marker, staff_content)

    def test_get_queryset_is_unchanged_and_no_field_added_for_the_tally(self):
        """T-37-17: the tally must never widen ALLOWED_FIELDS_FOR_NON_STAFF."""
        from solsys_code.campaign_views import ALLOWED_FIELDS_FOR_NON_STAFF

        self.assertNotIn('progress', ALLOWED_FIELDS_FOR_NON_STAFF)
        self.assertNotIn('tally', ALLOWED_FIELDS_FOR_NON_STAFF)

    def test_page_query_count_does_not_grow_with_additional_cached_rows(self):
        """D-08/T-37-19: adding rows to an already-cached page must not add queries --
        warm the tally cache for every run first (mirrors a page that was already loaded
        once), then assert the SAME query count for two and for three rows."""
        run1 = self._make_run(telescope_instrument='FTN/Q1')
        run2 = self._make_run(telescope_instrument='FTN/Q2')
        campaign_tally.tallies_for_runs([run1, run2])
        url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})

        # Prime any process-level framework caches (e.g. ContentType) with one throwaway
        # request first -- the very first request in a test process costs extra queries for
        # reasons unrelated to this feature, which would bias the two-vs-three comparison.
        self.client.get(url)

        with CaptureQueriesContext(connection) as ctx_two:
            self.client.get(url)
        two_row_count = len(ctx_two.captured_queries)

        run3 = self._make_run(telescope_instrument='FTN/Q3')
        campaign_tally.tallies_for_runs([run3])

        with CaptureQueriesContext(connection) as ctx_three:
            self.client.get(url)
        three_row_count = len(ctx_three.captured_queries)

        self.assertEqual(two_row_count, three_row_count)


class TestCampaignRollup(CampaignTallyViewTestBase):
    """TALLY-02/D-10: a header strip above the runs table and a nights-observed badge on
    the campaign list, rolled up only across publicly visible runs."""

    def setUp(self):
        cache.clear()

    def test_runs_page_shows_rollup_strip_with_group_record_and_segment_counts(self):
        run = self._make_run()
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('1 record', content)
        self.assertIn('Observed', content)
        self.assertIn('[O]', content)

    def test_pending_review_run_excluded_from_rollup_and_not_inferable(self):
        pending_run = self._make_run(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self._link_record(
            pending_run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        rollup = response.context['rollup']
        self.assertEqual(rollup['runs'], 0)
        self.assertEqual(rollup['records'], 0)

    def test_anonymous_get_of_runs_page_returns_200_with_rollup_content(self):
        response = self.client.get(reverse('campaigns:table', kwargs={'pk': self.campaign.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'group')

    def test_campaign_list_badge_reads_runs_and_nights_observed(self):
        run = self._make_run()
        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )
        response = self.client.get(reverse('campaigns:list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '1 run')
        self.assertContains(response, '1 night')
        self.assertContains(response, 'observed')

    def test_campaign_list_badge_omits_nights_clause_when_zero(self):
        self._make_run()
        response = self.client.get(reverse('campaigns:list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '1 run')
        self.assertNotContains(response, 'nights observed')

    def test_saving_a_linked_record_moves_the_rollup_on_next_load_no_clock_advance_no_cache_clear(self):
        run = self._make_run()
        table_url = reverse('campaigns:table', kwargs={'pk': self.campaign.pk})
        list_url = reverse('campaigns:list')

        first_table = self.client.get(table_url)
        self.assertEqual(first_table.context['rollup']['records'], 0)
        first_list = self.client.get(list_url)
        self.assertNotContains(first_list, 'nights observed')

        self._link_record(
            run,
            status='COMPLETED',
            scheduled_start=datetime(2026, 7, 10, 3, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 7, 10, 3, 30, tzinfo=dt_timezone.utc),
        )

        second_table = self.client.get(table_url)
        self.assertEqual(second_table.context['rollup']['records'], 1)
        second_list = self.client.get(list_url)
        self.assertContains(second_list, '1 night')

    def test_build_rollup_cache_key_moves_with_the_change_stamp(self):
        a = campaign_tally.build_rollup_cache_key(1, datetime(2026, 1, 1, tzinfo=dt_timezone.utc))
        b = campaign_tally.build_rollup_cache_key(1, datetime(2026, 1, 2, tzinfo=dt_timezone.utc))
        c = campaign_tally.build_rollup_cache_key(1, datetime(2026, 1, 1, tzinfo=dt_timezone.utc))
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def test_campaign_list_query_count_bound_with_three_campaigns(self):
        """D-10/T-37-19: one extra small query per listed campaign (the change-stamp probe)
        is the accepted, bounded cost -- the cached rollup itself must add no further query."""
        campaigns = [TargetList.objects.create(name=f'Bound Campaign {i}') for i in range(3)]
        for c in campaigns:
            self._make_run(campaign=c, telescope_instrument=f'FTN/{c.pk}')
        # Warm the rollup cache for every campaign (including the fixture's own) first.
        for c in list(campaigns) + [self.campaign]:
            campaign_tally.get_or_compute_rollup(c)

        with CaptureQueriesContext(connection) as ctx:
            self.client.get(reverse('campaigns:list'))
        base_count = len(ctx.captured_queries)

        extra_campaign = TargetList.objects.create(name='Bound Campaign Extra')
        self._make_run(campaign=extra_campaign, telescope_instrument='FTN/extra')
        campaign_tally.get_or_compute_rollup(extra_campaign)

        with CaptureQueriesContext(connection) as ctx:
            self.client.get(reverse('campaigns:list'))
        # One more campaign -> at most one more query (the version probe); the rollup
        # itself must be a cache hit, adding zero.
        self.assertLessEqual(len(ctx.captured_queries), base_count + 1)
