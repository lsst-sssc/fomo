from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy import units as u


@dataclass
class ETC:
    """
    Exposure Time Calculator
    """

    adc_error: ClassVar = np.sqrt(0.289) * (u.adu / u.pixel)
    # Conversion factor from FWHM to Gaussian standard deviation sigma
    _fwhm2sigma: ClassVar = 2.0 * np.sqrt(2 * np.log(2))

    mpc_code: str = '500'
    filter: str = 'V'
    primary_diam: u.Quantity = 72 * u.cm
    obstruction_diam: u.Quantity = 25 * u.cm  # in cm
    aperture: u.Quantity = 6 * u.arcsec  # arcsec,  _radius_ of the aperture
    fwhm: u.Quantity = 3 * u.arcsec  # arcsec,  _full_ width (i.e., diameter)
    qe: float = 0.9  # unitless,  0-1
    readnoise: float = 8  # counts per pixel
    pixel_size: u.Quantity = 3 * u.arcsec  # arcsec
    sky_brightness: float = 20.0  # magnitudes/arcsec^2
    airmass: float = 1.2  # ~ 1/sin(alt)
    min_alt: float = 0  # degrees
    max_alt: float = 90  # degrees
    min_dec: float = -90  # degrees
    max_dec: float = 90  # degrees
    min_ha: float = -180  # degrees
    max_ha: float = +180  # degrees
    min_elong: float = 0  # degrees
    max_elong: float = 180  # degrees
    sky_brightness_at_zenith: float = 20.0  # mags/arcsec^2

    def _fraction_inside(self) -> float:
        """
        Compute fraction of the PSF within the aperture.
        This could be generalized using the slit_vignette() function from etc
        """
        # FWHM should scale as airmass**0.6 IIRC but this is commented out in the original
        # /* const double real_fwhm = c->fwhm * c->airmass;     */
        r_scaled = self._fwhm2sigma * self.aperture / self.fwhm
        loss = 1.0 - np.exp(-r_scaled * r_scaled / 2.0)
        return loss

    def effective_area(self) -> u.Quantity:
        """
        Returns the effective area of the telescope as area Quantity
        """
        return np.pi * (self.primary_diam * self.primary_diam - self.obstruction_diam * self.obstruction_diam) / 4.0

    def star_electrons_per_second_per_pixel(self, mag):
        """
        Calculate the countrate of electrons/pixel/sec for an object of <mag>
        """
        mag_corr = mag + self.airmass * self.e.extinction
        countrate = (
            np.pow(10.0, -0.4 * mag_corr) * self.e.zero_point * self.effective_area() * self.qe * 1.0
        )  # _fraction_inside( c);
        return countrate

    def internal_snr_from_mag_and_exposure(self, mag, exposure):
        """
        Calculate a SNR from given <mag> and <exposure>
        """
        countrate_obj = self.star_electrons_per_second_per_pixel(mag, self.e)
        signal = countrate_obj * exposure
        noise = np.sqrt(signal)
        return signal / noise
