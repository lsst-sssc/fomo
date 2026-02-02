import json
import logging
from importlib.resources import files
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, unquote, urlparse

from astropy.table import QTable
from django.test import Client, TestCase
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
                abs_mag=19.7,
                slope=0.15,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
                'tp': 61000.0,
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
