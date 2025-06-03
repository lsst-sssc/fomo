from math import atan2, cos, degrees, radians, sin

import erfa
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Observatory(models.Model):
    """class/Model to hold and handle MPC observatories defined by their
    observatory code.
    Uses the Observatory Codes API (https://minorplanetcenter.net/mpcops/documentation/obscodes-api/)
    """

    OPTICAL_OBSTYPE = 0
    OCCULTATION_OBSTYPE = 1
    SATELLITE_OBSTYPE = 2
    RADAR_OBSTYPE = 4
    OBSTYPE_CHOICES = (
        (OPTICAL_OBSTYPE, 'Optical'),
        (OCCULTATION_OBSTYPE, 'Occultation'),
        (SATELLITE_OBSTYPE, 'Satellite'),
        (RADAR_OBSTYPE, 'Radar'),
    )
    obscode = models.CharField(
        max_length=4, null=False, blank=False, default=None, unique=True, verbose_name='MPC observatory code'
    )
    name = models.CharField(
        max_length=255, null=False, blank=False, default=None, unique=True, verbose_name='Name of the observatory'
    )
    short_name = models.CharField(max_length=255, verbose_name='Short version of observatory name')
    lon = models.FloatField(
        null=True,
        blank=False,
        verbose_name='Longitude (East is positive) [deg]',
        validators=[
            MinValueValidator(-180.0, message='longitude must be greater than -180.'),
            MaxValueValidator(180.0, message='longitude must be less than 180.'),
        ],
        db_index=True,
    )
    lat = models.FloatField(
        null=True,
        blank=False,
        verbose_name='Latitude (North is positive) [deg]',
        validators=[
            MinValueValidator(-180.0, message='latitude must be greater than -90.'),
            MaxValueValidator(180.0, message='latitude must be less than 90.'),
        ],
        db_index=True,
    )
    altitude = models.FloatField(null=True, blank=False, default=0.0, verbose_name='Altitude [m]')
    observations_type = models.SmallIntegerField(
        'Observations Type', null=False, blank=False, default=0, choices=OBSTYPE_CHOICES
    )
    uses_two_line_obs = models.BooleanField(
        default=False, verbose_name='Whether this observatory uses two-line observations e.g. satellite/radar'
    )
    old_names = models.TextField(verbose_name='Any previous names used by the observatory')
    created = models.DateTimeField(null=True, blank=False, editable=False, auto_now_add=True)
    modified = models.DateTimeField(null=True, blank=True, editable=True, auto_now_add=True)

    def __str__(self) -> str:
        return f'{self.obscode}: {self.name}'

    def from_parallax_constants(self, elong: float, rho_cos_phi: float, rho_sin_phi: float):
        """Convert from MPC parallax constants rho_cos_phi, rho_sin_phi to
        latitude, altitude and store these
        """

        # Get Earth's equatorial radius and flattening factor for WGS84
        # reference ellipsoid
        r, f = erfa.eform(1)

        # Form X,Y,Z vector (geocenter->observatory) from longitude and parallax constants, scaled to meters
        xyz = [r * cos(elong) * rho_cos_phi, r * sin(elong) * rho_cos_phi, r * rho_sin_phi]
        # Transform geocentric to geodetic
        lon, lat, alt = erfa.gc2gde(r, f, xyz)
        self.lon = degrees(lon)
        self.lat = degrees(lat)
        self.altitude = alt

    @property
    def to_parallax_constants(self) -> tuple[float, float]:
        """Convert from latitude and altitude to MPC parallax constants rho_cos_phi, rho_sin_phi
        and return these
        """

        # Get Earth's equatorial radius (in meters) and flattening factor for
        # WGS84 reference ellipsoid
        r, f = erfa.eform(1)
        axis_ratio = 1.0 - f

        u = atan2(sin(radians(self.lat)) * axis_ratio, cos(radians(self.lat)))
        rho_sin_phi = axis_ratio * sin(u) + (self.altitude / r) * sin(radians(self.lat))
        rho_cos_phi = cos(u) + (self.altitude / r) * cos(radians(self.lat))

        return rho_cos_phi, rho_sin_phi
