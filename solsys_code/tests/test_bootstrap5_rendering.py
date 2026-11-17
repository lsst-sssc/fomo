"""Browser-driven functional tests proving Bootstrap 5 rendering actually works.

Unit-level template-tag checks cannot prove that the BS5 JavaScript actually runs (e.g.
navbar dropdown toggling) or that django-crispy-forms emits BS5-flavoured layout markup
(no `.form-row`, BS4's class) rather than the pre-upgrade BS4 markup. This module drives a
real headless Chromium browser against a live Django test server with Playwright's
synchronous API to close that gap, as a follow-up to the tomtoolkit 3.0 / Bootstrap 5
upgrade (GitHub issue #45).
"""

from django.test import tag
from django.urls import reverse
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.tests.playwright_case import PlaywrightTestCase


@tag('functional')
class TestBootstrap5Rendering(PlaywrightTestCase):
    """Functional suite proving BS5 JS behavior and crispy BS5 layout markup render correctly."""

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
