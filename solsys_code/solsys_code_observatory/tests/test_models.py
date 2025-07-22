from datetime import datetime

from django.db import IntegrityError
from django.test import TestCase

# Import models to test
from solsys_code.solsys_code_observatory.models import Observatory


class TestObservatory(TestCase):
    def setUp(self) -> None:
        self.precision = 6
        return super().setUp()

    def test_creation_nocode(self):
        with self.assertRaises(IntegrityError):
            bad = Observatory.objects.create(name='wrong')  # noqa: F841

    def test_creation_noname(self):
        with self.assertRaises(IntegrityError):
            bad = Observatory.objects.create(obscode='X05')  # noqa: F841

    def test_creation_X05(self):
        expected_parallax_consts = (0.864981, -0.500958)

        rubin, created = Observatory.objects.get_or_create(
            obscode='X05',
            name='Simonyi Survey Telescope, Rubin Observatory',
            lat=-30.244600455,
            lon=-70.749420000,
            altitude=2683.57596,
        )

        self.assertTrue(created)
        # Test defaults
        self.assertEqual(rubin.observations_type, Observatory.OPTICAL_OBSTYPE)
        self.assertFalse(rubin.uses_two_line_obs)
        self.assertEqual(type(rubin.created), datetime)
        # test parallax constants properties
        self.assertAlmostEqual(expected_parallax_consts[0], rubin.to_parallax_constants[0])
        self.assertAlmostEqual(expected_parallax_consts[1], rubin.to_parallax_constants[1])

    def test_geocentric_X05(self):
        # Values from Bill Gray's parallax.cgi with lat,lon,alt from below
        expected_XYZ = [+1818.93900671, -5208.47103533, -3195.17141534]

        # Use slightly different values than above to match `EarthLocation.of_site('Rubin')`
        rubin, created = Observatory.objects.get_or_create(
            obscode='X05',
            name='Simonyi Survey Telescope, Rubin Observatory',
            lat=-30.244633333333333,
            lon=-70.74941666666666,
            altitude=2662.75,
        )

        xyz = rubin.to_geocentric()
        self.assertAlmostEqual(expected_XYZ[0], xyz[0], self.precision)
        self.assertAlmostEqual(expected_XYZ[1], xyz[1], self.precision)
        self.assertAlmostEqual(expected_XYZ[2], xyz[2], self.precision)

    def test_ObservatoryXYZ_X05(self):
        expected_XYZ = [+0.2851834, -0.8166132, -0.5009568]

        # Use slightly different values than above to match `EarthLocation.of_site('Rubin')`
        rubin, created = Observatory.objects.get_or_create(
            obscode='X05',
            name='Simonyi Survey Telescope, Rubin Observatory',
            lat=-30.244633333333333,
            lon=-70.74941666666666,
            altitude=2662.75,
        )

        xyz = rubin.ObservatoryXYZ()
        self.assertAlmostEqual(expected_XYZ[0], xyz[0], self.precision)
        self.assertAlmostEqual(expected_XYZ[1], xyz[1], self.precision)
        self.assertAlmostEqual(expected_XYZ[2], xyz[2], self.precision)
