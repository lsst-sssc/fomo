import json
import logging
import requests
from importlib.resources import files
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, unquote, urlparse

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.views import JPLSBId, JPLSBDBQuery, split_number_unit_regex

## Silence logging during tests
logging.disable(logging.CRITICAL)

MJD_TO_JD_CONVERSION = 2400000.5
JD2000 = 2451545.0  # Reference epoch
CR = 299792.458  # speed of light in km/s


class TestSplitNumberUnitRegex(SimpleTestCase):
    def test_int1(self):
        expected_value = 10
        expected_units = 'd'

        value, unit = split_number_unit_regex('10d')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)

    def test_int2(self):
        expected_value = 10
        expected_units = 'days'

        value, unit = split_number_unit_regex('10days')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)

    def test_float1(self):
        expected_value = 10.0
        expected_units = 'd'

        value, unit = split_number_unit_regex('10.0d')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)

    def test_float2(self):
        expected_value = 1.5
        expected_units = 'days'

        value, unit = split_number_unit_regex('1.5days')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)

    def test_float_nounit(self):
        expected_value = 1.5
        expected_units = ''

        value, unit = split_number_unit_regex('1.5')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)

    def test_bad(self):
        expected_value = expected_units = None

        value, unit = split_number_unit_regex('wibble')
        self.assertEqual(expected_value, value)
        self.assertEqual(expected_units, unit)


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


class TestJPLSBDBQuery(TestCase):
    def setUp(self):
        self.query = JPLSBDBQuery(orbit_class='IEO', orbital_constraints=['q<1.3', 'i<10.5'])
        self.base_url = 'https://ssd-api.jpl.nasa.gov/sbdb_query.api'
        self.fields = 'pdes,prefix,epoch_mjd,e,a,q,i,om,w,tp,H,G,M1,K1,condition_code,data_arc,n_obs_used'.split(',')
        self.maxDiff = None

        self.existing_target, _ = Target.objects.get_or_create(
            name='99942',
            defaults=dict(
                type='NON_SIDEREAL',
                permissions='PUBLIC',
                scheme='MPC_MINOR_PLANET',
                epoch_of_elements=60000.0,
                arg_of_perihelion=1.0,
                lng_asc_node=2.0,
                inclination=3.0,
                semimajor_axis=4.0,
                eccentricity=0.1,
                perihdist=0.9,
                epoch_of_perihelion=61000.0,
                abs_mag=19.1,
                slope=0.24,
            ),
        )

        # Load sample response JSON from file
        test_json_fp = files('solsys_code.tests.data').joinpath('test_query_jplsbdb.json')
        self.test_json = json.loads(test_json_fp.read_text())

    def test_default_constraints_when_no_inputs(self):
        query = JPLSBDBQuery()
        self.assertEqual(query.orbital_constraints, ['e|GE|1.2'])

    def test_translate_constraints(self):
        raw = ['q<1.3', 'i<=10.5', '6<=H<=7', '9<a<14']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['q|LT|1.3', 'i|LE|10.5', 'H|RG|6|7', 'a|RE|9|14']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_gt_like(self):
        raw = ['q>1.3', 'i>=10.5', '6>=H>=7', '9>a>14']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['q|GT|1.3', 'i|GE|10.5', 'H|RG|7|6', 'a|RE|14|9']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_is_defined(self):
        raw = ['rot_per IS DEFINED', 'rot_per<=4.2']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['rot_per|DF', 'rot_per|LE|4.2']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_not_defined(self):
        raw = [
            'rot_per IS NOT DEFINED',
        ]
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = [
            'rot_per|ND',
        ]
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_both_defined_and_not(self):
        raw = ['rot_per IS NOT DEFINED', 'H IS DEFINED']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['rot_per|ND', 'H|DF']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_equals(self):
        raw = ['condition_code == 0', 'source == ORB']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['condition_code|EQ|0', 'source|EQ|ORB']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_not_equals(self):
        raw = ['condition_code != 9', 'source != MPC:mpo']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['condition_code|NE|9', 'source|NE|MPC:mpo']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_both_equals_and_not(self):
        raw = ['condition_code == 8', 'source != MPC:mpo']
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ['condition_code|EQ|8', 'source|NE|MPC:mpo']
        self.assertEqual(query.orbital_constraints, expected)

    def test_translate_constraints_invalid_is_defined(self):
        # The code this is trying to test (lines 436--438) is actually unreachable...
        raw = [
            ' is defined',
        ]
        with self.assertRaises(ValueError):
            foo = JPLSBDBQuery(orbital_constraints=raw)
            self.assertEqual(foo.orbital_constraints, '42')

    def test_translate_constraints_invalid_is_not_defined(self):
        raw = [
            ' is not defined',
        ]
        with self.assertRaises(ValueError):
            foo = JPLSBDBQuery(orbital_constraints=raw)
            self.assertEqual(foo.orbital_constraints, '42')

    def test_translate_constraints_invalid(self):
        raw = [
            'foo bar biff splod',
        ]
        with self.assertRaises(ValueError):
            foo = JPLSBDBQuery(orbital_constraints=raw)
            self.assertEqual(foo.orbital_constraints, '42')

    def test_translate_constraints_comp_mismatch1(self):
        raw = [
            '6 > H < 7',
        ]
        with self.assertRaises(ValueError):
            _ = JPLSBDBQuery(orbital_constraints=raw)

    def test_translate_constraints_comp_mismatch2(self):
        raw = [
            '6 < H > 7',
        ]
        with self.assertRaises(ValueError):
            _ = JPLSBDBQuery(orbital_constraints=raw)

    def test_translate_constraints_comp_mismatch3(self):
        raw = [
            '6 >= H <= 7',
        ]
        with self.assertRaises(ValueError):
            _ = JPLSBDBQuery(orbital_constraints=raw)

    def test_translate_constraints_comp_mismatch4(self):
        raw = [
            '6 <= H >= 7',
        ]
        with self.assertRaises(ValueError):
            _ = JPLSBDBQuery(orbital_constraints=raw)

    def test_translate_constraints_comp_mismatch5(self):
        raw = [
            '6 <= H < 7',
        ]
        with self.assertRaises(ValueError):
            _ = JPLSBDBQuery(orbital_constraints=raw)

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
        self.assertEqual(self.fields, results['fields'])
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
        self.assertEqual(table['pdes'][0], self.test_json['data'][0][0])

    def test_parse_results_empty(self):
        empty_table = self.query.parse_results({})
        self.assertIsInstance(empty_table, QTable)
        self.assertEqual(len(empty_table), 0)

        none_table = self.query.parse_results(None)
        self.assertIsInstance(none_table, QTable)
        self.assertEqual(len(none_table), 0)

    def _set_results_table(self, rows):
        self.query.results_table = QTable(rows=rows)

    def test_create_target_asteroid(self):
        asteroid_row = [
            {
                'pdes': '12345',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            }
        ]
        self._set_results_table(asteroid_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='12345')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_MINOR_PLANET')
        self.assertEqual(t.abs_mag, 14.1)
        self.assertEqual(t.slope, 0.15)
        self.assertEqual(t.semimajor_axis, 2.5)
        self.assertEqual(t.epoch_of_perihelion, 61000.0)

    def test_create_target_asteroid2(self):
        asteroid_row = [
            {
                'pdes': '2025 X5',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            }
        ]
        self._set_results_table(asteroid_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='2025 X5')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_MINOR_PLANET')
        self.assertEqual(t.abs_mag, 14.1)
        self.assertEqual(t.slope, 0.15)
        self.assertEqual(t.semimajor_axis, 2.5)
        self.assertEqual(t.epoch_of_perihelion, 61000.0)

    def test_create_target_asteroid3(self):
        # Tests if default slope (G) parameter is created if not present
        asteroid_row = [
            {
                'pdes': '2025 X5',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': '2461000.5',
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': None,
                'M1': None,
                'K1': None,
            }
        ]
        self._set_results_table(asteroid_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='2025 X5')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_MINOR_PLANET')
        self.assertEqual(t.abs_mag, 14.1)
        self.assertEqual(t.slope, 0.15)
        self.assertEqual(t.semimajor_axis, 2.5)
        self.assertEqual(t.epoch_of_perihelion, 61000.0)

    def test_create_target_asteroid4(self):
        # Tests if epoch_of_perihelion parameter is created if not present
        asteroid_row = [
            {
                'pdes': '2025 X5',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': None,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.24,
                'M1': None,
                'K1': None,
            }
        ]
        self._set_results_table(asteroid_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='2025 X5')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_MINOR_PLANET')
        self.assertEqual(t.abs_mag, 14.1)
        self.assertEqual(t.slope, 0.24)
        self.assertEqual(t.semimajor_axis, 2.5)
        self.assertEqual(t.epoch_of_perihelion, None)

    def test_create_target_periodic_comet(self):
        periodic_comet_row = [
            {
                'pdes': '40P',
                'prefix': 'P',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 4.1,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 15.0,
                'K1': 4.0,
            }
        ]
        self._set_results_table(periodic_comet_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='40P')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_COMET')
        self.assertEqual(t.abs_mag, 15.0)
        self.assertEqual(t.slope, 4.0)
        self.assertEqual(t.semimajor_axis, 4.1)
        self.assertEqual(t.epoch_of_perihelion, 61000.0)

    def test_create_target_longperiod_comet(self):
        periodic_comet_row = [
            {
                'pdes': '2021 S3',
                'prefix': 'C',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 100000.0,
                'e': 0.99,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 13.0,
                'K1': 2.0,
            }
        ]
        self._set_results_table(periodic_comet_row)

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(after, before + 1)
        self.assertEqual(len(new_targets), 1)

        t = Target.objects.get(name='C/2021 S3')
        self.assertEqual(t.type, 'NON_SIDEREAL')
        self.assertEqual(t.scheme, 'MPC_COMET')
        self.assertEqual(t.abs_mag, 13.0)
        self.assertEqual(t.slope, 2.0)
        self.assertEqual(t.semimajor_axis, 100000.0)
        self.assertEqual(t.epoch_of_perihelion, 61000.0)

    def test_does_not_duplicate_existing_target(self):
        duplicate_row = [
            {
                'pdes': '99942',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            }
        ]
        self._set_results_table(duplicate_row)
        expected_num_new_targets = 0

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), expected_num_new_targets)
        self.assertEqual(after, before)
        self.assertEqual(Target.objects.filter(name='99942').count(), 1)

    def test_multiple_rows_creates_all_unique_targets(self):
        rows = [
            {
                'pdes': '12345',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            },
            {
                'pdes': '2021 S3',
                'prefix': 'C',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 100000.0,
                'e': 0.99,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 13.0,
                'K1': 2.0,
            },
            {
                'pdes': '40P',
                'prefix': 'P',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 4.1,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 15.0,
                'K1': 4.0,
            },
            {
                'pdes': '2016 P5',
                'prefix': 'P',
                'epoch_mjd': '59604',
                'e': '.05782617848139004',
                'a': '4.704576675254504',
                'q': '4.432528984751853',
                'i': '7.03721301721883',
                'om': '185.406425937313',
                'w': '34.9768095094869',
                'tp': '2460096.977777302902',
                'H': None,
                'G': None,
                'M1': '7.7',
                'K1': '16.5',
                'condition_code': '0',
                'data_arc': '7767',
                'n_obs_used': 233,
            },
        ]
        self._set_results_table(rows)
        expected_num_new_targets = 4

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), expected_num_new_targets)
        self.assertEqual(after, before + expected_num_new_targets)

        self.assertTrue(Target.objects.filter(name='12345').exists())
        self.assertTrue(Target.objects.filter(name='C/2021 S3').exists())
        self.assertTrue(Target.objects.filter(name='40P').exists())
        self.assertTrue(Target.objects.filter(name='P/2016 P5').exists())

    def test_multiple_rows_with_duplicates_only_creates_once_per_target(self):
        rows = [
            {
                'pdes': '99942',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            },
            {
                'pdes': '12345',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            },
            {
                'pdes': '12345',
                'prefix': None,
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 2.5,
                'e': 0.2,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': 14.1,
                'G': 0.15,
                'M1': None,
                'K1': None,
            },
            {
                'pdes': '2021 S3',
                'prefix': 'C',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 100000.0,
                'e': 0.99,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 13.0,
                'K1': 2.0,
            },
            {
                'pdes': '2021 S3',
                'prefix': 'C',
                'w': 10.0,
                'om': 20.0,
                'i': 5.0,
                'a': 100000.0,
                'e': 0.99,
                'epoch_mjd': 61000.0,
                'q': 2.0,
                'tp': 2461000.5,
                'condition_code': '0',
                'data_arc': 100,
                'n_obs_used': 25,
                'H': None,
                'G': None,
                'M1': 13.0,
                'K1': 2.0,
            },
        ]
        self._set_results_table(rows)
        expected_num_new_targets = 2

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), expected_num_new_targets)
        self.assertEqual(after, before + expected_num_new_targets)

        self.assertEqual(Target.objects.filter(name='99942').count(), 1)
        self.assertEqual(Target.objects.filter(name='12345').count(), 1)
        self.assertEqual(Target.objects.filter(name='C/2021 S3').count(), 1)

    def test_empty_results_table(self):
        self._set_results_table([])

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), 0)
        self.assertEqual(after, before)

    def test_null_results_table(self):
        self.query.results_table = None

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), 0)
        self.assertEqual(after, before)

    def test_missing_results_table_attribute(self):
        if hasattr(self.query, 'results_table'):
            delattr(self.query, 'results_table')

        before = Target.objects.count()
        new_targets = self.query.create_targets()
        after = Target.objects.count()

        self.assertEqual(len(new_targets), 0)
        self.assertEqual(after, before)


class TestJPLSBId(SimpleTestCase):
    def setUp(self) -> None:
        self.test_rubin = JPLSBId(mpc_code='X05', fov_ra_hwidth=1.75 * u.deg, fov_dec_hwidth=1.75 * u.deg)
        self.test_ps1 = JPLSBId(mpc_code='F51', fov_ra_hwidth=0.5 * u.deg, fov_dec_hwidth=0.5 * u.deg)
        self.root_url = 'https://ssd-api.jpl.nasa.gov/sb_ident.api'
        self.base_url = self.root_url + '?mpc-code=X05&mag-required=true&two-pass=false'
        test_json_fp = files('solsys_code.tests.data').joinpath('test_query_jplsbid.json')
        self.test_json = json.loads(test_json_fp.read_text())
        test_json_fp = files('solsys_code.tests.data').joinpath('test_query_jplsbid_PS1.json')
        self.test_json_ps1 = json.loads(test_json_fp.read_text())

        self.maxDiff = None
        self.time_fmt = '%Y-%m-%d %H:%M:%S'
        return super().setUp()

    def test_basic_init(self):
        foo = JPLSBId()

        self.assertEqual(self.test_rubin.mpc_code, foo.mpc_code)
        self.assertEqual(self.test_rubin.fov_ra_hwidth, foo.fov_ra_hwidth)
        self.assertEqual(self.test_rubin.fov_dec_hwidth, foo.fov_dec_hwidth)

    def test_build_base_query_defaults(self):
        expected_url = self.root_url + '?mpc-code=X05&mag-required=true&two-pass=false'

        url = self.test_rubin._build_base_query()

        self.assertEqual(expected_url, url)

    def test_build_base_query_defaults_mag_false(self):
        expected_url = self.root_url + '?mpc-code=X05&mag-required=false&two-pass=false'

        self.test_rubin.mag_required = False

        url = self.test_rubin._build_base_query()

        self.assertEqual(expected_url, url)

    def test_build_base_query_defaults_2pass_false(self):
        expected_url = self.root_url + '?mpc-code=X05&mag-required=true&two-pass=true'

        self.test_rubin.two_pass = True

        url = self.test_rubin._build_base_query()

        self.assertEqual(expected_url, url)

    def test_build_query_center_positive_dec(self):
        expected_url = (
            self.base_url
            + '&obs-time=2020-01-01T11:10:01&fov-ra-center=10-10-10&fov-dec-center=42-05-10'
            + '&fov-ra-hwidth=1.75&fov-dec-hwidth=1.75'
        )

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s +42d05m10s', frame='icrs')

        url = self.test_rubin._build_center_query(obs_time, center)

        self.assertEqual(expected_url, url)

    def test_build_query_center_negative_dec(self):
        expected_url = (
            self.base_url
            + '&obs-time=2020-01-01T11:10:01&fov-ra-center=10-10-10&fov-dec-center=M42-05-10'
            + '&fov-ra-hwidth=1.75&fov-dec-hwidth=1.75'
        )

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s -42d05m10s', frame='icrs')

        url = self.test_rubin._build_center_query(obs_time, center)

        self.assertEqual(expected_url, url)

    @patch('requests.get')
    def test_query_center_positive_dec(self, mock_get):
        # Mock the requests.get call and insert the JSON results from file
        mock_response = Mock()
        mock_response.json.return_value = self.test_json_ps1
        mock_response.status_code = 200
        mock_response.ok = True
        mock_get.return_value = mock_response

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s +42d05m10s', frame='icrs')

        expected_keys = ['signature', 'summary', 'fields_first', 'observer', 'n_first_pass', 'data_first_pass']
        expected_sig_ver = '1.1'
        expected_sig_source = 'NASA/JPL Small-Body Identification API'
        expected_obs_date = f"{obs_time.strftime('%Y-%b-%d %H:%M:%S')} ({obs_time.jd:.6f} UT)"

        result = self.test_ps1.query_center(obs_time, center, raw_response=True, verbose=False)

        self.assertEqual(6, len(result))
        for key in expected_keys:
            self.assertTrue(key in result, msg=f'Failure to find key: {key}')
        self.assertEqual(expected_sig_ver, result['signature']['version'])
        self.assertEqual(expected_sig_source, result['signature']['source'])
        observer = result['observer']
        self.assertEqual('Pan-STARRS 1, Haleakala', observer['location'])
        self.assertEqual(expected_obs_date, observer['obs_date'])

    @patch('requests.get')
    def test_query_center_negative_dec(self, mock_get):
        # Mock the requests.get call and insert the JSON results from file
        mock_response = Mock()
        self.test_json['summary']['fov-ra-center'] = '10-10-10'
        self.test_json['summary']['fov-dec-center'] = 'M42-05-10'
        mock_response.json.return_value = self.test_json
        mock_response.status_code = 200
        mock_response.ok = True
        mock_get.return_value = mock_response

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s -42d05m10s', frame='icrs')

        result = self.test_rubin.query_center(obs_time, center, verbose=False)

        self.assertEqual(6, len(result))
        summary = result['summary']
        self.assertEqual(self.test_rubin.mpc_code, summary['mpc-code'])
        self.assertEqual(self.test_rubin.fov_ra_hwidth.value, float(summary['fov-ra-hwidth']))
        self.assertEqual(self.test_rubin.fov_dec_hwidth.value, float(summary['fov-dec-hwidth']))
        self.assertEqual('10-10-10', summary['fov-ra-center'])
        self.assertEqual('M42-05-10', summary['fov-dec-center'])

    @patch('requests.get')
    def test_make_query_failure_bad(self, mock_get):
        """test query of non-existant object"""
        mock_response = Mock()
        mock_response.status_code = 400
        http_error = requests.exceptions.HTTPError()
        mock_response.raise_for_status.side_effect = http_error
        mock_response.json.return_value = (
            b'{"message":"invalid obs-code (should be MPC 3-character string: e.g., '
            + b'\'G96\', \'704\', etc.)","moreInfo":"https://ssd-api.jpl.nasa.gov/doc/sb_ident.html","code":"400"}\n'
        )
        mock_get.return_value = mock_response
        url = self.base_url + '?mpc-code=FOO'

        result = self.test_rubin.make_query(url)
        self.assertEqual(result, None)

    @patch('requests.get')
    def test_make_query_results_PS1(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = self.test_json_ps1
        mock_response.status_code = 200
        mock_response.ok = True
        mock_get.return_value = mock_response

        # We're mocking the response but do this for the look of the thing
        # (and it makes for easier testing of expected values)
        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s -42d05m10s', frame='icrs')
        url = self.test_ps1._build_center_query(obs_time, center)

        expected_keys = ['signature', 'summary', 'fields_first', 'observer', 'n_first_pass', 'data_first_pass']
        expected_sig_ver = '1.1'
        expected_sig_source = 'NASA/JPL Small-Body Identification API'
        expected_first_obj = [
            '90 Antiope (A866 TA)',
            '10:22:20',
            '+12 54\'02"',
            '1.E4',
            '1.E4',
            '1.5E4',
            '14.4',
            '-6.942E+00',
            '4.598E+00',
            '1.1E4',
            '1.1E4',
        ]
        expected_last_obj = [
            'C/2024 J2 (Wierzchos)',
            '12:53:04',
            '-02 48\'06"',
            '1.E5',
            '-5.E4',
            '1.5E5',
            '26.8',
            '1.516E+00',
            '1.154E+00',
            '3.1E5',
            '3.1E5',
        ]

        result = self.test_ps1.make_query(url)

        self.assertEqual(6, len(result))
        for key in expected_keys:
            self.assertTrue(key in result, msg=f'Failure to find key: {key}')
        self.assertEqual(expected_sig_ver, result['signature']['version'])
        self.assertEqual(expected_sig_source, result['signature']['source'])

        summary = result['summary']
        self.assertEqual(self.test_ps1.mpc_code, summary['mpc-code'])
        self.assertEqual(self.test_ps1.two_pass, json.loads(summary['two-pass']))
        self.assertEqual(self.test_ps1.elems_required, json.loads(summary['req-elem']))
        self.assertEqual(obs_time.strftime(self.time_fmt), summary['obs-time'])

        self.assertEqual(5, result['n_first_pass'])
        self.assertEqual(5, len(result['data_first_pass']))
        self.assertEqual(expected_first_obj, result['data_first_pass'][0])
        self.assertEqual(expected_last_obj, result['data_first_pass'][-2])

    @patch('requests.get')
    def test_query_center_PS1(self, mock_get):
        expected_columns = [
            'Object name',
            'Astrometric position',
            'Dist. from center RA',
            'Dist. from center Dec',
            'Dist. from center Norm',
            'V magnitude',
            'RA rate',
            'Dec rate',
            'Pos error RA',
            'Pos error Dec',
        ]
        expected_names = ['90', '1627', '2025 AW11', 'C/2024 J2', '472P']
        expected_dec_error = u.Quantity([11_000, 22_000, 10_000, 310_000, 42_000] * u.arcsec)

        # Mock the requests.get call and insert the JSON results from file
        mock_response = Mock()
        mock_response.json.return_value = self.test_json_ps1
        mock_response.status_code = 200
        mock_response.ok = True
        mock_get.return_value = mock_response

        # We're mocking the response but do this for the look of the thing
        # (and it makes for easier testing of expected values)
        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s -42d05m10s', frame='icrs')

        table = self.test_rubin.query_center(obs_time, center, raw_response=False, verbose=False)

        self.assertTrue(isinstance(table, QTable))
        self.assertEqual(expected_columns, table.colnames)
        self.assertEqual(5, len(table))
        for name1, name2 in zip(expected_names, table['Object name']):
            self.assertEqual(name1, name2)
        assert_quantity_allclose(expected_dec_error, table['Pos error Dec'])

    def test_parse_results_bad_none(self):
        table = self.test_ps1.parse_results(None)

        self.assertEqual(None, table)

    def test_parse_results_bad_empty_dict(self):
        table = self.test_ps1.parse_results({})

        self.assertEqual(None, table)

    def test_parse_results(self):
        expected_num_objs = 5
        expected_columns = [
            'Object name',
            'Astrometric position',
            'Dist. from center RA',
            'Dist. from center Dec',
            'Dist. from center Norm',
            'V magnitude',
            'RA rate',
            'Dec rate',
            'Pos error RA',
            'Pos error Dec',
        ]
        expected_names = ['90', '1627', '2025 AW11', 'C/2024 J2', '472P']
        expected_positions = SkyCoord(
            ['10:22:20', '09:47:52', '10:11:23', '12:53:04', '09:55:30.54',],
            ['+12:54:02', '+12:28:03', '+08:53:27', '-02:48:06', '+00:54:27.4'],
            frame='icrs',
            unit=(u.hourangle, u.deg),
        )
        expected_ra_rates = u.Quantity([-6.942, -18.62, -4.166, 1.516, -18.44], unit=u.arcsec / u.hour)

        table = self.test_ps1.parse_results(self.test_json_ps1)

        self.assertTrue(isinstance(table, QTable))
        self.assertEqual(expected_columns, table.colnames)
        self.assertEqual(expected_num_objs, len(table))
        for name1, name2 in zip(expected_names, table['Object name']):
            self.assertEqual(name1, name2)
        assert_quantity_allclose(expected_ra_rates, table['RA rate'])
        assert_quantity_allclose(expected_positions.ra, table['Astrometric position'].ra)
        assert_quantity_allclose(expected_positions.dec, table['Astrometric position'].dec)
