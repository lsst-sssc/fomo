import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from django.test import TestCase
from numpy.testing import assert_almost_equal  # , assert_array_almost_equal

from solsys_code.etc.views import ETC


class TestETC(TestCase):
    def setUp(self):
        self.test_etc = ETC()
        return super().setUp()

    def tearDown(self):
        return super().tearDown()

    def test_effective_area_default(self):
        expected_area = 3580.630227 * u.cm**2

        area = self.test_etc.effective_area()

        assert_quantity_allclose(expected_area, area)

    def test_effective_area1(self):
        expected_area = 30159.289474 * u.cm**2

        self.test_etc.primary_diam = 2 * u.m
        self.test_etc.obstruction_diam = 0.4 * u.m
        area = self.test_etc.effective_area()

        assert_quantity_allclose(expected_area, area)

    def test_effective_area2(self):
        expected_area = 660.39419171 * u.cm**2

        self.test_etc.primary_diam = 350 * u.mm
        self.test_etc.obstruction_diam = 196 * u.mm
        area = self.test_etc.effective_area()

        assert_quantity_allclose(expected_area, area)
        self.assertEqual(u.cm**2, area.unit)

    def test_fraction_inside(self):
        expected_loss = 0.9999847

        loss_fraction = self.test_etc._fraction_inside()

        assert_almost_equal(expected_loss, loss_fraction)

    def test_fraction_inside_good_seeing(self):
        expected_loss = 1.0

        self.test_etc.fwhm = 0.8 * u.arcsec
        loss_fraction = self.test_etc._fraction_inside()

        assert_almost_equal(expected_loss, loss_fraction)

    def test_internal_snr_from_mag_and_exposure(self):
        """
        Based on expcalc test case:
        ./expcalc -skybrightness 19 -primary 200 -filter V snr 30 20.5 -d
        200.00-cm primary, 40.00-cm obstruction
        Filter V, QE 0.90, read noise 8.00 electrons/pixel
        Pixels are 1.04 arcsec;  aperture 6.00 arcsec, FWHM 3.00 arcsec
        Sky brightness 19.00 mag/arcsec^2; airmass 1.50
        SNR 2.37862 with exposure time 30.0 seconds and magnitude 20.50
        3375.17375 star electrons (total exposure)
        Noise from star 58.09625 (square root of the above line)
        Noise from sky 1415.39471
        Read noise 82.12141 (from 105.37385 pixels)
        Total noise 1418.96486 (above three lines added in quadrature)
        """
        expected_snr = 2.37862

        etc = ETC(
            primary_diam=200 * u.cm,
            obstruction_diam=40 * u.cm,
            pixel_size=1.036 * u.arcsec,
            sky_brightness=19.0,
            airmass=1.5,
        )
        snr = etc.internal_snr_from_mag_and_exposure(20.5, 30)

        assert_almost_equal(expected_snr, snr, 5)
