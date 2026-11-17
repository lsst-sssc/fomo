"""Browser-driven smoke test of the non-sidereal Plan panel on the target detail page.

``test_views.TestNonsiderealTargetPlan`` already checks the rendered HTML. What it cannot show is that
the Plotly figures embedded in that HTML are actually drawn by the browser, or that submitting the Plan
form with a subset of the site checkboxes round-trips; this drives a real headless Chromium to check both.
"""

from django.contrib.auth.models import User
from django.test import tag
from django.urls import reverse
from tom_targets.models import Target

from solsys_code.tests.playwright_case import PlaywrightTestCase


@tag('functional')
class TestNonsiderealPlanPanel(PlaywrightTestCase):
    def setUp(self):
        super().setUp()
        # (33933) is observable in the evening from CPT, and through the night further west, on 2025-05-10
        self.target = Target.objects.create(
            name='33933',
            type='NON_SIDEREAL',
            permissions='PUBLIC',
            scheme='MPC_MINOR_PLANET',
            epoch_of_elements=61000.0,
            mean_anomaly=342.8987983972185,
            arg_of_perihelion=197.2440098291647,
            eccentricity=0.21317079351206,
            lng_asc_node=55.4085914553028,
            inclination=1.0791909799414,
            semimajor_axis=2.186745866749343,
            epoch_of_perihelion=59874.98228566302,
            perihdist=1.72059551512517,
            abs_mag=14.89,
            slope=0.15,
        )
        User.objects.create_user(username='planner', password='pw')
        self.path = reverse('targets:detail', kwargs={'pk': self.target.pk})
        self.url = f'{self.live_server_url}{self.path}'

    def test_plan_form_draws_airmass_and_cadence_figures(self):
        # The target page needs a login; signing in with ?next= lands on the target
        self.page.goto(f'{self.live_server_url}/accounts/login/?next={self.path}')
        self.page.locator('input[name="login"]').fill('planner')
        self.page.locator('input[name="password"]').fill('pw')
        self.page.locator('button[type="submit"], input[type="submit"]').first.click()
        self.page.wait_for_url(self.url)
        panel = self.page.locator('#plan-panel')

        # Before the form is submitted there is a form with every site ticked and no figures
        assert panel.locator('input[name="sites"]').count() == 6
        assert panel.locator('input[name="sites"]:checked').count() == 6
        assert self.page.locator('#nonsidereal-airmass').count() == 0

        panel.locator('input[name="start_time"]').fill('2025-05-10')
        panel.locator('input[name="end_time"]').fill('2025-05-11')
        panel.locator('input[name="airmass"]').fill('3')
        for sitecode in ('coj', 'tfn', 'elp', 'ogg'):
            panel.locator(f'input[name="sites"][value="{sitecode}"]').uncheck()
        panel.locator('button[type="submit"]').click()
        self.page.wait_for_load_state('networkidle')

        # The site selection survives the round trip
        checked = panel.locator('input[name="sites"]:checked').evaluate_all('boxes => boxes.map(box => box.value)')
        assert sorted(checked) == ['cpt', 'lsc']

        # Plotly has drawn the airmass figure with one legend entry per selected site
        airmass = self.page.locator('#nonsidereal-airmass .js-plotly-plot')
        airmass.locator('.main-svg').first.wait_for(state='visible')
        assert airmass.locator('.legend .traces').count() == 2

        # ... and the cadence figure: a bar per site with a window, plus the summary line
        cadence = self.page.locator('#cadence-window')
        cadence.locator('.js-plotly-plot .main-svg').first.wait_for(state='visible')
        assert cadence.locator('.js-plotly-plot .barlayer .point').count() >= 1
        assert 'midpoint' in cadence.inner_text()
