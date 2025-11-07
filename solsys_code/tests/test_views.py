from django.test import Client, TestCase
from django.urls import reverse
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

# Import module to test

MJD_TO_JD_CONVERSION = 2400000.5
JD2000 = 2451545.0  # Reference epoch
CR = 299792.458  # speed of light in km/s


class TestEphemeris(TestCase):
    def setUp(self):
        self.test_observatory, created = Observatory.objects.get_or_create(
            obscode='K93',
            name='Sutherland-LCO Dome C',
            lat=-32.380667412,
            lon=+20.81011,
            altitude=1808.33,
        )
        self.test_rubin_obs, created = Observatory.objects.get_or_create(
            obscode='X05',
            name='Simonyi Survey Telescope, Rubin Observatory',
            lat=-30.244600454210392,
            lon=-70.74942,
            altitude=2683.576,
        )
        self.test_target, created = Target.objects.get_or_create(
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
        self.client = Client()
        return super().setUp()

    def test_K93(self):
        expected_result = f'Ephemeris for {self.test_target.name} at  ({self.test_observatory.obscode})'

        response = self.client.get(
            reverse('ephem', kwargs={'pk': self.test_target.pk}) + f'?obscode={self.test_observatory.obscode}'
        )
        self.assertInHTML(expected_result, response.content.decode())

    def test_no_site_given(self):
        expected_result = f'Ephemeris for {self.test_target.name} at  ({self.test_rubin_obs.obscode})'

        response = self.client.get(reverse('ephem', kwargs={'pk': self.test_target.pk}))
        self.assertInHTML(expected_result, response.content.decode())

    def test_no_site(self):
        expected_result = 'Not Found'

        response = self.client.get(reverse('ephem', kwargs={'pk': self.test_target.pk}) + '?obscode=500')
        self.assertInHTML(expected_result, response.content.decode())
