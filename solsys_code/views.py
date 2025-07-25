from collections import namedtuple
from pathlib import Path

import numpy as np
import pooch
import spiceypy as spice
from astropy import units as u
from astropy.constants import GM_sun
from astropy.time import Time
from astropy.timeseries import TimeSeries
from django.shortcuts import get_object_or_404, render
from django.views.generic import View
from layup.convert import get_output_column_names_and_types
from layup.predict import predict
from layup.utilities.data_processing_utilities import layup_furnish_spiceypy
from sorcha.ephemeris.orbit_conversion_utilities import universal_cartesian
from sorcha.ephemeris.simulation_geometry import ecliptic_to_equatorial
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

cache_dir = Path(pooch.os_cache('layup'))
# "Furnish" (load) SPICE kernels
layup_furnish_spiceypy(cache_dir)


def convert_target_to_layup(target, sun_dict=None):
    """Converts a `Target` to a numpy array in format needed for 'layup'

    Args:
        target (tom_targets.model.Target): Target
        sun_dict (dict): [Optional] A dict with a key of a JD_TDB pointing
            at a dict of {x,y,z,vx,vy,vz} for position and velocity of the
            Sun to override the internal rebound determination

    Returns:
        output (numpy structured array): Data converted to layup input format
    """

    # Numerical constants needed
    # Solar Gravitional constant (converted to au/days)
    mu_sun = GM_sun.to(u.au**3 / u.day**2).value
    # mu_sun = 2.9591220828559115e-04 # value from tests
    # mu_total = 0.00029630927487993194
    mu = mu_sun
    # Value of au in meters (fixed by IAU 2012 resolution)
    AU_M = 149597870700
    AU_KM = AU_M / 1000.0
    SEC_PER_DAY = 24 * 60 * 60

    if sun_dict is None:
        sun_dict = {}
    # Transform Target orbital elements to cartesian state vector
    time_peri_tdb = Time(target.epoch_of_perihelion, format='mjd', scale='tdb')
    epoch_tdb = Time(target.epoch_of_elements, format='mjd', scale='tdb')
    x, y, z, vx, vy, vz = universal_cartesian(
        mu,
        target.perihdist,
        target.eccentricity,
        np.radians(target.inclination),
        np.radians(target.lng_asc_node),
        np.radians(target.arg_of_perihelion),
        time_peri_tdb.jd,
        epoch_tdb.jd,
    )
    # Get required columns and dtypes
    # Columns which may be added to the output data by the orbit fitting process
    ORBIT_FIT_COLS = [
        ('csq', 'f8'),  # Chi-square value
        ('ndof', 'i4'),  # Number of degrees of freedom
        ('niter', 'i4'),  # Number of iterations
        ('method', 'O'),  # Method used for orbit fitting
        ('flag', 'i4'),  # Single-character flag indicating success of the fit
    ]
    cols_to_keep = ORBIT_FIT_COLS
    primary_id_column_name = 'provID'
    has_covariance = False
    required_column_names, default_column_dtypes = get_output_column_names_and_types(
        primary_id_column_name, has_covariance, cols_to_keep
    )
    # Construct the output dtype for the converted data
    output_dtype = [
        (col, dtype) for col, dtype in zip(required_column_names['BCART_EQ'], default_column_dtypes, strict=False)
    ]
    row = (np.nan,) * 6
    cov = np.full((6, 6), np.nan)
    # First we convert our data into equatorial barycentric cartesian coordinates,
    # regardless of the input format. That allows us to simplify the conversion
    # process below by only having the logic to convert from BCART_EQ to the other formats.

    # Convert to Heliocentric Cartesian (equatorial) by converting from ecliptic coordinates
    ecliptic_coords = np.array((x, y, z))
    ecliptic_velocities = np.array((vx, vy, vz))

    equatorial_coords = np.array(ecliptic_to_equatorial(ecliptic_coords))
    equatorial_velocities = np.array(ecliptic_to_equatorial(ecliptic_velocities))
    # Convert from heliocentric->barycentric using the Sun's position and velocity
    # This can either be passed in as [sun_dict] (e.g. for testing) or we can
    # determine it ourselves using plain SPICE calls
    if epoch_tdb.jd not in sun_dict:
        # Rebound:
        # from layup.utilities.layup_configs import LayupConfigs
        # from sorcha.ephemeris.simulation_setup import _create_assist_ephemeris
        # configs = LayupConfigs()
        # ephem, gm_sun, gm_total = _create_assist_ephemeris(configs.auxiliary, cache_dir)
        # sun_dict[epoch_tdb.jd] = ephem.get_particle("Sun", epoch_tdb.jd - ephem.jd_ref)
        et_sun = spice.str2et(f'jd {epoch_tdb.jd} tdb')
        sun_posvel, sun_ltt = spice.spkezr('SUN', et_sun, 'J2000', 'NONE', 'SSB')
        # Convert from km and km/s to AU and AU/day
        sun_posvel /= AU_KM
        sun_posvel[3:6] *= SEC_PER_DAY
        Sun = namedtuple('Sun', 'x y z vx vy vz')
        sun_dict[epoch_tdb.jd] = Sun(
            x=sun_posvel[0],
            y=sun_posvel[1],
            z=sun_posvel[2],
            vx=sun_posvel[3],
            vy=sun_posvel[4],
            vz=sun_posvel[5],
        )

    sun = sun_dict[epoch_tdb.jd]
    equatorial_coords += np.array((sun.x, sun.y, sun.z))
    equatorial_velocities += np.array((sun.vx, sun.vy, sun.vz))
    row = tuple(np.concatenate([equatorial_coords, equatorial_velocities]))
    row += (epoch_tdb.mjd,)
    row += tuple(0 for col, _ in cols_to_keep)

    results = []
    # Turn our converted row into a structured array
    output_format = 'BCART_EQ'
    cov_res = tuple(val for val in cov.flatten()) if has_covariance else tuple()

    result_struct_array = np.array(
        [(target.name, output_format) + row + cov_res],
        dtype=output_dtype,
    )
    results.append(result_struct_array)

    # Convert the list of results to a numpy structured array
    output = np.squeeze(np.array(results)) if len(results) > 1 else results[0]

    return output


class Ephemeris(View):
    """Generate an ephemeris for a specific `Target`, specified by <pk>,
    for an `Observatory`, specific by <obscode> which are retrieved from
    the query URL.
    Returns the rendered template of the ephemeris.
    """

    def get(self, request, *args, **kwargs):
        """
        Handles the GET requests to this view.

        :param request: request object for this GET request
        :type request: HTTPRequest
        """

        target = get_object_or_404(Target, pk=kwargs['pk'])
        obscode = request.GET.get('obscode', '500')
        # XXX Could replace this by a creation of the missing Observatory
        # relatively easily
        observatory = get_object_or_404(Observatory, obscode=obscode)
        # Construct time series of `Time` objects in UTC.
        # XXX Todo: initialize start, stop, step from query parameters
        start_time = request.GET.get('start', None)
        if start_time is None:
            start_time = Time.now()
            start_time = Time(start_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
        else:
            try:
                start_time = Time(start_time, scale='utc')
            except ValueError:
                start_time = Time.now()
                start_time = Time(start_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))

        ts = TimeSeries(time_start=start_time, time_delta=1 * u.day, n_samples=20)
        # Generate a list of JD_TDB times
        times = ts.time.tdb.jd

        data = convert_target_to_layup(target)

        # Get results from layup's predict routine
        predictions = predict(data, obscode=obscode, times=times, cache_dir=cache_dir)
        ephem_lines = []
        for e in predictions:
            ephem_line = [e['epoch_UTC'], e['ra_deg'], e['dec_deg'], 42.0]
            ephem_lines.append(ephem_line)
        return render(request, 'ephem.html', {'target': target, 'ephem_lines': ephem_lines, 'observatory': observatory})
