from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy import units as u


@dataclass
class EtcInternals:
    """Internal current configuration details for the ETC
    adapted from expcalc.cpp's expcalc_internals_t struct
    Defaults for 'V'
    """

    extinction: float = 0.15
    zero_point: float = 8.66e05
    n_pixels_in_aperture: float = 9
    countrate_sky: float = 0.0
    noise2: float = 0.0
    n_star: float = 0.0
    area: float = 0.0


def find_filter(e: EtcInternals, filter: str):
    """Performs a lookup/mapping of the passed <filter>
    and sets the extinction and zero_point in the passed
    <e> instance. 'R' filter is used if the filter is not
    found.

    Args:
        e (ETC_Internals): ETC_Internals (dataclass) instance
        filter (str): Observed filter

    Returns:
        _type_: _description_
    """
    filters = {
        'U': [0.60, 5.50e05],
        'B': [0.40, 3.91e05],
        'V': [0.20, 8.66e05],
        'R': [0.10, 1.10e06],
        'I': [0.08, 6.75e05],
        'N': [0.20, 4.32e06],
        #        'W': [0.15, 2.00e06], # Original value (maybe a VR value?)
        # Value from synphot of obs.countrate() for 1 cm**2 Vega spec and LCO PS-w filter from SVO
        'W': [0.15, 3.69e06],
    }
    filt_info = filters.get(filter, 'R')
    e.extinction = filt_info[0]
    e.zero_point = filt_info[1]
    return 0


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

    def _set_internals(self):
        self.e = EtcInternals()
        if find_filter(self.e, self.filter):
            return -1
        assert self.aperture > 0.1 * u.arcsec and self.aperture < 100.0 * u.arcsec
        assert self.pixel_size > 0.0 * u.arcsec and self.pixel_size < 100.0 * u.arcsec
        assert self.readnoise > 0.0
        assert self.primary_diam > 1.0 * u.cm and self.primary_diam < 31 * u.m
        assert self.obstruction_diam >= 0.0 and self.obstruction_diam < self.primary_diam
        assert self.qe > 0.01 and self.qe <= 1.0
        self.e.zero_point = self.e.zero_point * u.photon / (u.cm**2 * u.s)
        self.e.n_pixels_in_aperture = np.pi * self.aperture * self.aperture / (self.pixel_size * self.pixel_size)
        self.e.noise2 = self.readnoise * self.readnoise
        self.e.countrate_sky = self.sky_electrons_per_second_per_pixel(self.e.zero_point)

    def effective_area(self) -> u.Quantity:
        """
        Returns the effective area of the telescope as area Quantity (converted to cm**2 units)
        """

        pri_diam_cm = self.primary_diam.to(u.cm)
        sec_diam_cm = self.obstruction_diam.to(u.cm)

        return np.pi * (pri_diam_cm * pri_diam_cm - sec_diam_cm * sec_diam_cm) / 4.0

    def star_electrons_per_second_per_pixel(self, mag):
        """
        Calculate the countrate of electrons/pixel/sec for an object of <mag>
        """
        mag_corr = mag + self.airmass * self.e.extinction
        countrate = (
            np.power(10.0, -0.4 * mag_corr)
            * self.e.zero_point
            * self.effective_area()
            * self.qe
            * self._fraction_inside()
        )

        return countrate

    def sky_electrons_per_second_per_pixel(self, zero_point):
        """Calculate the countrate (in electrons/pixels/sec) coming from the sky for the
        given <zero_point

        Args:
            zero_point (float): Zeropoint of system
        """
        area = self.effective_area()
        sky_electrons_per_sec_per_square_arcsec = (
            np.power(10.0, -0.4 * self.sky_brightness) * zero_point * area * self.qe
        )
        pixel_area = self.pixel_size * self.pixel_size
        return sky_electrons_per_sec_per_square_arcsec * pixel_area.to(u.arcsec**2).value

    def internal_snr_from_mag_and_exposure(self, mag, exposure):
        """
        Calculate a SNR from given <mag> and <exposure>
        """
        self._set_internals()
        if not isinstance(exposure, u.Quantity):
            exposure *= u.s
        countrate_obj = self.star_electrons_per_second_per_pixel(mag)
        signal = countrate_obj * exposure
        noise = np.sqrt(
            signal + self.e.n_pixels_in_aperture * (self.e.countrate_sky * exposure + self.e.noise2 * u.photon)
        )

        return signal.value / noise.value
