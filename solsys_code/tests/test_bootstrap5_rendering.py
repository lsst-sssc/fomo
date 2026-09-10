"""Browser-driven functional tests proving Bootstrap 5 rendering actually works.

Unit-level template-tag checks cannot prove that the BS5 JavaScript actually runs (e.g.
navbar dropdown toggling) or that django-crispy-forms emits BS5-flavoured layout markup
(no `.form-row`, BS4's class) rather than the pre-upgrade BS4 markup. This module drives a
real headless Chromium browser against a live Django test server with Playwright's
synchronous API to close that gap, as a follow-up to the tomtoolkit 3.0 / Bootstrap 5
upgrade (GitHub issue #45).

Also covers the calendar's Bootstrap 5 modal open path (UAT G-33-2): FOMO's override of
tom_calendar's month partial used to open `#cal-modal` with a jQuery call the tomtoolkit
3.x base page never loads, so every click on the calendar threw silently. The tests below
click a real rendered calendar page and assert the Bootstrap 5 modal actually opens, with
no page error.
"""

import os
from datetime import date, datetime
from datetime import timezone as dt_timezone

from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse
from playwright.sync_api import sync_playwright
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory


class TestBootstrap5Rendering(StaticLiveServerTestCase):
    """Functional suite proving BS5 JS behavior and crispy BS5 layout markup render correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Playwright's synchronous API keeps an asyncio event loop "running" (via greenlet-based
        # dispatch) in whichever thread calls sync_playwright().start(). Django's async-safety
        # guard (django.utils.asyncio.async_unsafe) sees that running loop and raises
        # SynchronousOnlyOperation on every subsequent DB access in this thread -- a false
        # positive, since nothing here is actually concurrent. This is a documented
        # Playwright/Django interaction; the standard fix is to opt this thread out of the
        # async-safety check, scoped to this class's lifetime so it doesn't mask real
        # async-safety bugs in other tests that share this process.
        cls._prev_async_unsafe = os.environ.get('DJANGO_ALLOW_ASYNC_UNSAFE')
        os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        if cls._prev_async_unsafe is None:
            os.environ.pop('DJANGO_ALLOW_ASYNC_UNSAFE', None)
        else:
            os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = cls._prev_async_unsafe
        super().tearDownClass()

    CAL_YEAR = 2026
    CAL_MONTH = 8

    def setUp(self):
        super().setUp()
        self.page = self.browser.new_page()

        # Fixture for the calendar modal-open tests below, built in the shape of
        # MonthCellCampaignMarkerTest (test_calendar_template.py). Built here in setUp(),
        # not setUpTestData -- StaticLiveServerTestCase is a TransactionTestCase subclass,
        # which does not support setUpTestData's class-scoped fixture semantics (only
        # django.test.TestCase wraps setUpTestData in a rolled-back transaction); each
        # TransactionTestCase test flushes the database in its own teardown, so per-class
        # data would only survive for whichever test happened to run first.
        self.campaign = TargetList.objects.create(name='BS5 Modal Fixture Campaign')
        self.approved_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(self.CAL_YEAR, self.CAL_MONTH, 4),
            window_end=date(self.CAL_YEAR, self.CAL_MONTH, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        self.attributed_event = CalendarEvent.objects.create(
            title='BS5 Modal Attributed Event',
            start_time=datetime(self.CAL_YEAR, self.CAL_MONTH, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(self.CAL_YEAR, self.CAL_MONTH, 4, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=self.attributed_event, run=self.approved_run)

        # An event with no CalendarEventMeta row at all -- a click target campaign_decoration()
        # never attributes -- so the fix is proven on an unattributed entry too (ANNOT-02
        # empty-input edge, must_haves Truth 3), not just the attributed one above.
        # Title kept at or under the timed-event loop's truncatechars:16 budget so the
        # locator below can match on the full rendered title text without truncation.
        self.unattributed_event = CalendarEvent.objects.create(
            title='UnattrEvent',
            start_time=datetime(self.CAL_YEAR, self.CAL_MONTH, 12, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(self.CAL_YEAR, self.CAL_MONTH, 12, 21, 0, tzinfo=dt_timezone.utc),
        )

    def tearDown(self):
        self.page.close()
        super().tearDown()

    def _calendar_url(self):
        return f'{self.live_server_url}{reverse("calendar:calendar")}?year={self.CAL_YEAR}&month={self.CAL_MONTH}'

    def test_calendar_modal_opens_for_campaign_attributed_event_with_no_page_errors(self):
        """UAT G-33-2: clicking a campaign-attributed calendar entry opens `#cal-modal` via
        the Bootstrap 5 API, with the 'Attributed campaign run' block and its 'View
        campaign' link, and raises no page error. Against the pre-fix jQuery handler this
        assertion fails with a ReferenceError naming the undefined `$` global (see SUMMARY
        for the exact message observed during the revert-and-restore sensitivity check)."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self.page.goto(self._calendar_url())
        self.page.locator('.cal-event').first.click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()

        modal_body_text = self.page.locator('#cal-modal-body').inner_text()
        assert 'Attributed campaign run' in modal_body_text
        assert self.page.locator('#cal-modal-body a', has_text='View campaign').count() >= 1

        assert page_errors == []

    def test_calendar_modal_opens_for_new_event_button_with_no_page_errors(self):
        """The '+ New Event' button -- a click target with no CalendarEvent behind it at
        all -- opens the same modal through the same Bootstrap 5 handler."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self.page.goto(self._calendar_url())
        self.page.get_by_role('button', name='+ New Event').click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()

        assert page_errors == []

    def test_calendar_modal_opens_for_unattributed_event_with_no_page_errors(self):
        """must_haves Truth 3: an entry with no attributed run must open the pop-up too --
        the defect affected every click target, not just campaign-attributed ones."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self.page.goto(self._calendar_url())
        self.page.locator('.cal-event', has_text='UnattrEvent').first.click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()

        assert page_errors == []

    def test_calendar_modal_opens_for_empty_day_cell_with_no_page_errors(self):
        """must_haves Truth 3: clicking the empty area of a day cell with no events at all
        must also open the pop-up. Click the day-num span specifically -- it sits outside
        the inner event-container div's `event.stopPropagation()` guard, so the click
        bubbles to the day cell's own handler."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self.page.goto(self._calendar_url())
        empty_day_cell = self.page.locator(f'[hx-get*="date={self.CAL_YEAR}-{self.CAL_MONTH:02d}-01"]')
        empty_day_cell.locator('.day-num').click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()

        assert page_errors == []

    def test_navbar_dropdown_toggle_shows_menu(self):
        """Clicking the Observatories navbar toggle reveals a `.dropdown-menu.show` (BS5 JS runs)."""
        self.page.goto(f'{self.live_server_url}/')

        # The TOM base navbar has multiple [data-bs-toggle="dropdown"] toggles (user menu etc.);
        # use .first to avoid a Playwright strict-mode "multiple elements matched" error.
        self.page.locator('[data-bs-toggle="dropdown"]').first.click()

        dropdown_menu = self.page.locator('.dropdown-menu.show').first
        dropdown_menu.wait_for(state='visible')
        assert dropdown_menu.is_visible()

    def test_ephemeris_form_uses_bs5_crispy_layout(self):
        """The makeephem page has zero `.form-row` (BS4) and at least one `form .row` (BS5 crispy)."""
        target = NonSiderealTargetFactory.create()

        self.page.goto(f'{self.live_server_url}{reverse("makeephem", kwargs={"pk": target.pk})}')

        assert self.page.locator('.form-row').count() == 0
        assert self.page.locator('form .row').count() >= 1

    def test_observatory_create_form_submits_to_observatory_url(self):
        """Submitting the obscode create form creates the Observatory and lands on its detail page.

        '/observatory/' alone is not a sufficient check: the create page itself lives at
        '/observatory/create/', so a failed submission that just re-renders the form would also
        satisfy that substring check. Assert the redirect left the create page, and that an
        Observatory row actually exists, so a broken submit button/form is caught.
        """
        self.page.goto(f'{self.live_server_url}{reverse("solsys_code_observatory:create")}')

        self.page.locator('input[name="obscode"]').fill('704')
        self.page.locator('button[type="submit"], input[type="submit"]').first.click()
        self.page.wait_for_load_state('networkidle')

        assert '/observatory/create/' not in self.page.url
        assert Observatory.objects.filter(obscode='704').exists()
