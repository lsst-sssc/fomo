import json
import logging
from importlib.resources import files
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, unquote, urlparse

from astropy.table import QTable
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.views import JPLSBDBQuery

## Silence logging during tests
logging.disable(logging.CRITICAL)

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


class TestJPLSBDBQuery(SimpleTestCase):
    def setUp(self):
        self.query = JPLSBDBQuery(orbit_class='IEO', orbital_constraints=['q<1.3', 'i<10.5'])
        self.base_url = 'https://ssd-api.jpl.nasa.gov/sbdb_query.api'
        self.fields = 'full_name,first_obs,epoch,e,a,q,i,om,w'
        self.maxDiff = None

        # Load sample response JSON from file
        test_json_fp = files('solsys_code.tests.data').joinpath('test_query_jplsbdb.json')
        self.test_json = json.loads(test_json_fp.read_text())

    def test_translate_constraints(self):
        raw = ['q<1.3', 'i<=10.5', '6<=H<=7']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['q|LT|1.3', 'i|LE|10.5', 'H|RG|6|7']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_is_defined(self):
        raw = ['rot_per IS DEFINED', 'rot_per<=4.2']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['rot_per|DF', 'rot_per|LE|4.2']
        self.assertEqual(query.orbital_constraints, expected)

    def test_build_query_url(self):
        url = self.query.build_query_url()
        self.assertTrue(url.startswith(self.base_url))
        self.assertIn('fields=', url)
        self.assertIn('sb-class=IEO', url)
        self.assertIn('sb-cdata=', url)

        # Parse sb-cdata from URL
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        sb_cdata_encoded = query_params.get('sb-cdata', [None])[0]
        self.assertIsNotNone(sb_cdata_encoded, 'sb-cdata not found in URL')

        sb_cdata_json = json.loads(unquote(sb_cdata_encoded))
        self.assertIn('AND', sb_cdata_json)
        self.assertIn('q|LT|1.3', sb_cdata_json['AND'])
        self.assertIn('i|LT|10.5', sb_cdata_json['AND'])

    @patch('requests.get')
    def test_run_query_success(self, mock_get):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = self.test_json
        mock_get.return_value = mock_response

        results = self.query.run_query()
        self.assertIsInstance(results, dict)
        self.assertIn('fields', results)
        self.assertIn('data', results)

    @patch('requests.get')
    def test_run_query_failure(self, mock_get):
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        results = self.query.run_query()
        self.assertIsNone(results)

    def test_parse_results_valid(self):
        table = self.query.parse_results(self.test_json)
        self.assertIsInstance(table, QTable)
        self.assertEqual(table.colnames, self.test_json['fields'])
        self.assertEqual(len(table), len(self.test_json['data']))
        self.assertEqual(table['full_name'][0], self.test_json['data'][0][0])

    def test_parse_results_empty(self):
        empty_table = self.query.parse_results({})
        self.assertIsInstance(empty_table, QTable)
        self.assertEqual(len(empty_table), 0)

        none_table = self.query.parse_results(None)
        self.assertIsInstance(none_table, QTable)
        self.assertEqual(len(none_table), 0)
