import json
import logging
from unittest import skipIf
from unittest.mock import Mock, patch

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from django.test import SimpleTestCase
from importlib_resources import files

# Import code to test
from solsys_code.views import JPLSBId

# Disable logging during testing
logger = logging.getLogger(__name__)
# Disable anything below CRITICAL level
logging.disable(logging.CRITICAL)


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

    @skipIf(True, 'mock needed')
    def test_query_center_positive_dec(self):
        expected_url = (
            self.base_url
            + '&obs-time=2020-01-01T11:10:01&fov-ra-center=10-10-10&fov-dec-center=42-05-10'
            + '&fov-ra-hwidth=1.75&fov-dec-hwidth=1.75'
        )

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s +42d05m10s', frame='icrs')

        url = self.test_rubin.query_center(obs_time, center, verbose=False)

        self.assertEqual(expected_url, url)

    @skipIf(True, 'mock needed')
    def test_query_center_negative_dec(self):
        expected_url = (
            self.base_url
            + '&obs-time=2020-01-01T11:10:01&fov-ra-center=10-10-10&fov-dec-center=M42-05-10'
            + '&fov-ra-hwidth=1.75&fov-dec-hwidth=1.75'
        )

        obs_time = Time('2020-01-01T11:10:01', scale='utc')
        center = SkyCoord('10h10m10s -42d05m10s', frame='icrs')

        url = self.test_rubin.query_center(obs_time, center, verbose=False)

        self.assertEqual(expected_url, url)

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

        self.assertEqual(4, result['n_first_pass'])
        self.assertEqual(4, len(result['data_first_pass']))
        self.assertEqual(expected_first_obj, result['data_first_pass'][0])
        self.assertEqual(expected_last_obj, result['data_first_pass'][-1])
