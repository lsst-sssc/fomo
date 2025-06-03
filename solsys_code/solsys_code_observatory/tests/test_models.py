from datetime import datetime

from django.db import IntegrityError
from django.test import TestCase

# Import models to test
from solsys_code.solsys_code_observatory.models import Observatory


class TestObservatory(TestCase):
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
