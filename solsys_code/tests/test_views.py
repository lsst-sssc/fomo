# Disable logging during testing
import logging

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from django.test import SimpleTestCase

# Import code to test
from solsys_code.views import JPLSBId

logger = logging.getLogger(__name__)
# Disable anything below CRITICAL level
logging.disable(logging.CRITICAL)


class TestJPLSBId(SimpleTestCase):
    def setUp(self) -> None:
        self.test_rubin = JPLSBId(mpc_code='X05', fov_ra_hwidth=1.75 * u.deg, fov_dec_hwidth=1.75 * u.deg)
        self.root_url = 'https://ssd-api.jpl.nasa.gov/sb_ident.api'
        self.base_url = self.root_url + '?mpc-code=X05&mag-required=true&two-pass=false'

        self.maxDiff = None
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
