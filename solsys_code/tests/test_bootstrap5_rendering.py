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

Phase 39 (ACCESS-02): an anonymous visitor's modal is a read-only card and the create click
targets (the '+ New Event' button and the day-cell create handler) exist only for a logged-in
user, so the tests that open the modal from those targets log a plain user in first.
"""

import os
import re
import time
from datetime import date, datetime
from datetime import timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import SimpleTestCase, tag
from django.urls import reverse
from django.utils import timezone
from playwright.sync_api import sync_playwright
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import (
    CalendarEventMeta,
    CampaignRun,
    CampaignRunObservation,
    ObservationRecordDismissal,
)
from solsys_code.solsys_code_observatory.models import Observatory


@tag('functional')
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

    def _calendar_editor(self):
        """A plain, non-staff user: any logged-in user may write to the calendar (D-01)."""
        return get_user_model().objects.create_user(username='bs5-calendar-editor', password='pw')

    def _log_in_browser(self, user):
        """Log the browser in by handing it the test client's session cookie."""
        self.client.force_login(user)
        self.page.context.add_cookies(
            [
                {
                    'name': settings.SESSION_COOKIE_NAME,
                    'value': self.client.cookies[settings.SESSION_COOKIE_NAME].value,
                    'url': self.live_server_url,
                }
            ]
        )

    def _wait_for_db(self, predicate, message, timeout=5.0):
        """Poll until predicate() is true; fail with message after timeout seconds.

        The live server answers in its own thread, so a row it writes may land a moment after the
        browser sees the modal close.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        self.fail(message)

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
        all -- opens the same modal through the same Bootstrap 5 handler. The button exists
        only for a logged-in user (Phase 39 D-07); the visitor case is covered by the two
        anonymous tests below."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self._log_in_browser(self._calendar_editor())
        self.page.goto(self._calendar_url())
        self.page.get_by_role('button', name='+ New Event').click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()
        self.page.locator('#cal-modal-body form').wait_for(state='attached')
        assert self.page.locator('#cal-modal-body form').count() == 1

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
        bubbles to the day cell's own handler. The day cell's create handler exists only for a
        logged-in user (Phase 39 D-07); the visitor case is covered by the two anonymous
        tests below."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self._log_in_browser(self._calendar_editor())
        self.page.goto(self._calendar_url())
        empty_day_cell = self.page.locator(f'[hx-get*="date={self.CAL_YEAR}-{self.CAL_MONTH:02d}-01"]')
        empty_day_cell.locator('.day-num').click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()
        self.page.locator('#cal-modal-body form').wait_for(state='attached')
        assert self.page.locator('#cal-modal-body form').count() == 1

        assert page_errors == []

    def test_anonymous_visitor_opens_read_only_event_card_with_no_page_errors(self):
        """D-09 (ACCESS-02): a visitor with no session cookie sees no create click target, can
        still open an event's pop-up through the Bootstrap 5 API, and reads a plain-text card
        (with the attributed-run block and its campaign link) containing no form control."""
        page_errors = []
        requests = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        self.page.on('request', lambda request: requests.append(request.url))

        self.page.goto(self._calendar_url())
        assert self.page.get_by_role('button', name='+ New Event').count() == 0
        assert self.page.locator('[hx-get*="/calendar/create/"]').count() == 0

        self.page.locator('.cal-event').first.click()

        modal = self.page.locator('#cal-modal.show')
        modal.wait_for(state='visible')
        assert modal.is_visible()
        card = self.page.locator('#cal-modal-body #cal-event-card')
        card.wait_for(state='visible')
        assert card.is_visible()
        assert self.page.locator('#cal-modal-body').locator('form, button, input, select, textarea').count() == 0

        modal_body_text = self.page.locator('#cal-modal-body').inner_text()
        assert 'BS5 Modal Attributed Event' in modal_body_text
        assert 'Attributed campaign run' in modal_body_text
        assert self.page.locator('#cal-modal-body a', has_text='View campaign').count() >= 1

        assert not any('/calendar/create/' in url for url in requests)
        assert page_errors == []

    def test_anonymous_click_on_empty_day_cell_opens_nothing(self):
        """D-07: for a visitor the day cell is inert -- clicking an empty one opens no modal and
        sends no request to the create route."""
        page_errors = []
        requests = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        self.page.on('request', lambda request: requests.append(request.url))

        self.page.goto(self._calendar_url())
        self.page.locator('.cal-day.is-current-month').first.locator('.day-num').click()
        self.page.wait_for_timeout(500)

        assert self.page.locator('#cal-modal.show').count() == 0
        assert not any('/calendar/create/' in url for url in requests)
        assert page_errors == []

    def test_signed_in_editor_creates_edits_and_deletes_from_month_view(self):
        """Phase 39 success criterion 2 (D-01, D-03): a plain, non-staff logged-in user still
        creates, edits and deletes a calendar event from the month view, through the guarded
        routes, exactly as before the login guards existed."""
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        # Delete's hx-confirm raises a native confirm dialog; accept it.
        self.page.on('dialog', lambda dialog: dialog.accept())

        self._log_in_browser(self._calendar_editor())
        modal = self.page.locator('#cal-modal.show')
        modal_body = self.page.locator('#cal-modal-body')

        # Create: click the empty 2026-08-20 day cell, fill the title, save.
        self.page.goto(self._calendar_url())
        day_cell = self.page.locator(f'[hx-get*="date={self.CAL_YEAR}-{self.CAL_MONTH:02d}-20"]')
        day_cell.locator('.day-num').click()
        modal.wait_for(state='visible')
        self.page.locator('#cal-modal-body form').wait_for(state='attached')
        # Upstream pre-fills the start time from the clicked day.
        assert self.page.locator('#id_start_time').input_value() == '2026-08-20T00:00'
        self.page.locator('#id_title').fill('EditorTrip')
        modal_body.get_by_role('button', name='Save', exact=True).click()
        modal.wait_for(state='hidden')
        self._wait_for_db(
            lambda: CalendarEvent.objects.filter(title='EditorTrip').count() == 1,
            'the created event never reached the database',
        )
        event = CalendarEvent.objects.get(title='EditorTrip')
        assert event.start_time.date() == date(2026, 8, 20)

        # Edit: reload the month (a save re-renders the current year's month, not 2026-08),
        # reopen the event, change the title, save.
        self.page.goto(self._calendar_url())
        self.page.locator('.cal-event', has_text='EditorTrip').first.click()
        modal.wait_for(state='visible')
        # The editor gets the update form, not the read-only card. (The pop-up of a saved event
        # also holds upstream's separate add-a-todo form, so count the update form specifically.)
        update_form = self.page.locator('#cal-modal-body form[hx-post*="/calendar/update/"]')
        update_form.wait_for(state='attached')
        assert update_form.count() == 1
        assert self.page.locator('#cal-event-card').count() == 0
        self.page.locator('#id_title').fill('EditorTrip2')
        modal_body.get_by_role('button', name='Save', exact=True).click()
        modal.wait_for(state='hidden')
        self._wait_for_db(
            lambda: CalendarEvent.objects.filter(pk=event.pk, title='EditorTrip2').exists(),
            'the edited title never reached the database',
        )

        # Delete: reopen, press Delete, accept the confirm dialog.
        self.page.goto(self._calendar_url())
        self.page.locator('.cal-event', has_text='EditorTrip2').first.click()
        modal.wait_for(state='visible')
        modal_body.get_by_role('button', name='Delete').wait_for(state='visible')
        modal_body.get_by_role('button', name='Delete').click()
        modal.wait_for(state='hidden')
        self._wait_for_db(
            lambda: not CalendarEvent.objects.filter(pk=event.pk).exists(),
            'the deleted event is still in the database',
        )

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

    def test_attribution_page_sections_open_and_close_with_bootstrap5(self):
        """UAT G-37.1-1: the attribution page's Confirmed and Dismissed headings must open and
        close their sections under the Bootstrap 5 that ``tom_common/base.html`` loads. The page
        used the Bootstrap 4 ``data-toggle``/``data-target`` attributes, which Bootstrap 5
        ignores, so neither heading did anything and the system-linked rows (and their Undo
        buttons) were unreachable. Also pins the default state: Confirmed open on load,
        Dismissed folded away."""
        staff = get_user_model().objects.create_user(username='bs5-attribution-staff', password='pw', is_staff=True)
        campaign = TargetList.objects.create(name='BS5 Attribution Campaign')
        target = NonSiderealTargetFactory.create()
        campaign.targets.add(target)
        observatory = Observatory.objects.create(obscode='E10', name='Siding Spring', short_name='SSO')
        # Same shape as AttributionViewTestBase.campaign_run, whose system links
        # TestSystemLinkInConfirmedTable already saves without error.
        run = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=observatory,
            telescope_class='',
        )

        def make_record(night_offset, observation_id):
            return ObservationRecord.objects.create(
                target=target,
                user=staff,
                facility='LCO',
                observation_id=observation_id,
                status='PENDING',
                parameters={
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': datetime(2026, 7, 7 + night_offset, 22, 0).isoformat(),
                    'end': datetime(2026, 7, 8 + night_offset, 6, 0).isoformat(),
                },
            )

        CampaignRunObservation.objects.create(
            run=run, observation_record=make_record(0, 'BS5-ATTR-1'), confirmed_at=timezone.now()
        )
        ObservationRecordDismissal.objects.create(
            observation_record=make_record(1, 'BS5-ATTR-2'),
            run=run,
            dismissed_by=staff,
            dismissed_at=timezone.now(),
            reason='BS5 browser dismissal',
        )

        self._log_in_browser(staff)
        page_errors = []
        self.page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        self.page.goto(f'{self.live_server_url}{reverse("campaigns:attribution")}')

        confirmed = self.page.locator('#attribution-confirmed-section')
        dismissed = self.page.locator('#attribution-dismissed-section')

        # On load: Confirmed is open and shows the system link; Dismissed is folded away.
        confirmed.wait_for(state='visible')
        assert 'System (exact match)' in confirmed.inner_text()
        dismissed.wait_for(state='hidden')

        # Clicking the Dismissed heading opens it, with the dismissal reason showing.
        self.page.locator('button[data-bs-target="#attribution-dismissed-section"]').click()
        dismissed.wait_for(state='visible')
        assert 'BS5 browser dismissal' in dismissed.inner_text()

        # Clicking the Confirmed heading folds it away.
        self.page.locator('button[data-bs-target="#attribution-confirmed-section"]').click()
        confirmed.wait_for(state='hidden')

        assert page_errors == []


class TestTemplatesUseBootstrap5DataAttributes(SimpleTestCase):
    """UAT G-37.1-1 guard: ``tom_common/base.html`` loads Bootstrap 5, whose plugins bind only the
    ``data-bs-*`` attribute spellings. A Bootstrap 4 spelling (``data-toggle``, ``data-target``,
    ...) in a template is inert there and fails silently in a browser -- the attribution page's
    collapse headings did exactly that. This scan needs no browser and no database."""

    # Bootstrap 4's plugin data attributes. Bootstrap 5's carry a `bs-` segment (data-bs-toggle)
    # and so never match: after `data-` the next characters must be one of these names and then `=`.
    BOOTSTRAP4_ATTRIBUTE = re.compile(
        r'\bdata-(?:toggle|target|dismiss|parent|ride|slide-to|slide|spy|backdrop|keyboard|placement)\s*='
    )

    @staticmethod
    def _template_files():
        files = []
        for template_dir in settings.TEMPLATES[0]['DIRS']:
            files.extend(sorted(Path(template_dir).rglob('*.html')))
        return files

    def test_no_template_uses_bootstrap4_data_attributes(self):
        files = self._template_files()
        hits = []
        for path in files:
            for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                if self.BOOTSTRAP4_ATTRIBUTE.search(line):
                    relative = path.relative_to(Path(settings.TEMPLATES[0]['DIRS'][0]))
                    hits.append(f'{relative}:{lineno}: {line.strip()}')
        self.assertEqual(hits, [], 'Bootstrap 4 plugin data attributes found:\n' + '\n'.join(hits))

    def test_the_scan_is_not_vacuous(self):
        """Guard against the scan silently covering nothing (wrong directory, empty glob)."""
        files = self._template_files()
        self.assertGreaterEqual(len(files), 20, f'expected at least 20 templates, scanned {len(files)}')
        names = {path.as_posix() for path in files}
        for expected in ('campaigns/attribution_queue.html', 'campaigns/campaign_list.html'):
            self.assertTrue(any(name.endswith(expected) for name in names), f'{expected} was not covered by the scan')

    def test_the_pattern_matches_bootstrap4_and_not_bootstrap5_spellings(self):
        for bad in ('data-toggle="collapse"', 'data-target="#x"', '<a data-dismiss="modal">', 'data-slide-to="1"'):
            self.assertIsNotNone(self.BOOTSTRAP4_ATTRIBUTE.search(bad), bad)
        for good in (
            'data-bs-toggle="collapse"',
            'data-bs-target="#x"',
            'data-bs-dismiss="modal"',
            'data-bs-slide-to="1"',
        ):
            self.assertIsNone(self.BOOTSTRAP4_ATTRIBUTE.search(good), good)
