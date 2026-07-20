"""Browser-driven functional tests proving Bootstrap 5 rendering actually works.

Unit-level template-tag checks cannot prove that the BS5 JavaScript actually runs (e.g.
navbar dropdown toggling) or that django-crispy-forms emits BS5-flavoured layout markup
(no `.form-row`, BS4's class) rather than the pre-upgrade BS4 markup. This module drives a
real headless Chromium browser against a live Django test server with Playwright's
synchronous API to close that gap, as a follow-up to the tomtoolkit 3.0 / Bootstrap 5
upgrade (GitHub issue #45).
"""

import os

from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse
from playwright.sync_api import sync_playwright
from tom_targets.tests.factories import NonSiderealTargetFactory

# Playwright's synchronous API keeps an asyncio event loop "running" (via greenlet-based
# dispatch) in whichever thread calls sync_playwright().start(). Django's async-safety guard
# (django.utils.asyncio.async_unsafe) sees that running loop and raises SynchronousOnlyOperation
# on every subsequent DB access in this thread -- a false positive, since nothing here is
# actually concurrent. This is a documented Playwright/Django interaction; the standard fix is
# to opt this thread out of the async-safety check.
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')


class TestBootstrap5Rendering(StaticLiveServerTestCase):
    """Functional suite proving BS5 JS behavior and crispy BS5 layout markup render correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()
        super().tearDown()

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
        """Submitting the obscode create form lands on a URL under /observatory/ either way."""
        self.page.goto(f'{self.live_server_url}{reverse("solsys_code_observatory:create")}')

        self.page.locator('input[name="obscode"]').fill('704')
        self.page.locator('button[type="submit"], input[type="submit"]').first.click()
        self.page.wait_for_load_state('networkidle')

        assert '/observatory/' in self.page.url
