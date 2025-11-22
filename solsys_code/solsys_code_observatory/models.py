from math import atan2, cos, degrees, radians, sin

import erfa
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


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
            MinValueValidator(-90.0, message='latitude must be greater than -90.'),
            MaxValueValidator(90.0, message='latitude must be less than 90.'),
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
    old_names = models.TextField(blank=True, verbose_name='Any previous names used by the observatory')
    created = models.DateTimeField(null=True, blank=False, editable=False, default=timezone.now)
    modified = models.DateTimeField(null=True, blank=True, editable=True, default=timezone.now)

    # Get Earth's equatorial radius and flattening factor for WGS84
    # reference ellipsoid. `r` is in meters
    _r, _f = erfa.eform(1)

    def __str__(self) -> str:
        return f'{self.obscode}: {self.name}'

    def from_parallax_constants(self, elong: float, rho_cos_phi: float, rho_sin_phi: float):
        """Convert from MPC parallax constants rho_cos_phi, rho_sin_phi to
        latitude, altitude and store these
        """

        # print(f"elong={elong:.6f} rhocosphi={rho_cos_phi:.6f} rhosinphi={rho_sin_phi:.6f}")

        # Form X,Y,Z vector (geocenter->observatory) from longitude and parallax constants, scaled to meters
        xyz = [
            self._r * cos(radians(elong)) * rho_cos_phi,
            self._r * sin(radians(elong)) * rho_cos_phi,
            self._r * rho_sin_phi,
        ]
        # Transform geocentric to geodetic
        lon, lat, alt = erfa.gc2gde(self._r, self._f, xyz)
        self.lon = degrees(lon)
        self.lat = degrees(lat)
        self.altitude = alt

    @property
    def to_parallax_constants(self) -> tuple[float, float]:
        """Convert from latitude and altitude to MPC parallax constants rho_cos_phi, rho_sin_phi
        and return these
        """

        axis_ratio = 1.0 - self._f
        rho_cos_phi = rho_sin_phi = 0.0
        if self.lat and self.lon and self.altitude:
            u = atan2(sin(radians(self.lat)) * axis_ratio, cos(radians(self.lat)))
            rho_sin_phi = axis_ratio * sin(u) + (self.altitude / self._r) * sin(radians(self.lat))
            rho_cos_phi = cos(u) + (self.altitude / self._r) * cos(radians(self.lat))

        return rho_cos_phi, rho_sin_phi

    def to_geocentric(self) -> tuple[float, float, float]:  # Potentially support optional units later
        """Converts the observatory location to geocentric coordinates
        WGS84 ellipsoid is assumed and the values corresponding to that are
        set in the erfa.gd2gce() call

        Returns:
            tuple[float, float, float]: Geocentric position (x,y,z) in km
        """

        xyz = [None, None, None]
        if self.lat and self.lon and self.altitude:
            xyz = erfa.gd2gce(self._r, self._f, radians(self.lon), radians(self.lat), self.altitude)
            # Convert to km
            xyz /= 1000.0
        return xyz

    def to_geodetic(self) -> tuple[float, float, float]:
        """Returns the observatory location in geodetic coordinates
        (longitude, latitude, altitude/height). The longitude and latitude
        values are returned in radians, with East longitude positive and the
        altitude is returned in meters

        Returns
            tuple[float, float, float]: Geodetic position (lon,lat,height) in radians/m
        """
        return (radians(self.lon), radians(self.lat), self.altitude)

    def ObservatoryXYZ(self) -> tuple[float, float, float]:
        """Converts the observatory location to geocentric coordinates (in units of Earth radii)
        Provides similar functionality to Sorcha's Observatory.ObservatoryXYZ()

        Returns:
            tuple[float, float, float]: Geocentric position (x,y,z) in Earth radii
        """

        xyz = self.to_geocentric()
        if all(xyz):
            # Convert to Earth radii
            r_in_km = self._r / 1000.0
            xyz /= r_in_km
        return xyz

    @property
    def get_observations_type(self) -> str:
        """Returns the str version of the stored observations type"""
        obstype_choices = dict(self.OBSTYPE_CHOICES)
        return obstype_choices[self.observations_type]

    class Meta:  # noqa: D106
        verbose_name_plural = 'Observatories'
