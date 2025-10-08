import re
from collections import defaultdict, namedtuple
from csv import writer
from datetime import timezone
from io import StringIO
from pathlib import Path

import assist
import erfa
import numpy as np
import pandas as pd
import pooch
import rebound
import spiceypy as spice
from astropy import units as u
from astropy.constants import GM_sun, c
from astropy.coordinates.builtin_frames.utils import get_jd12, get_polar_motion
from astropy.time import Time
from astropy.timeseries import TimeSeries
from crispy_forms.bootstrap import FormActions
from crispy_forms.layout import HTML, Layout, Submit
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.generic import FormView, View

# from sbpy.photometry import HG
from sorcha.ephemeris.orbit_conversion_utilities import universal_cartesian
from sorcha.ephemeris.simulation_driver import EphemerisGeometryParameters, get_residual_vectors, get_vec
from sorcha.ephemeris.simulation_geometry import (
    barycentricObservatoryRates,
    ecliptic_to_equatorial,
    integrate_light_time,
    vec2ra_dec,
)
from sorcha.ephemeris.simulation_parsing import Observatory as SorchaObservatory
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris, furnish_spiceypy
from sorcha.utilities.sorchaConfigs import auxiliaryConfigs
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

from .forms import EphemerisForm

# Value of au in meters (fixed by IAU 2012 resolution)
AU_M = 149597870700
AU_KM = AU_M / 1000.0
SEC_PER_DAY = 24 * 60 * 60
SPEED_OF_LIGHT = c.to(u.km / u.s).value * SEC_PER_DAY / AU_KM
PI_OVER_2 = np.pi / 2.0  # aka 90 degrees

cache_dir = Path(pooch.os_cache('sorcha'))


# "Furnish" (load) SPICE kernels
# This will download 1.6 GB of SPICE kernels to the `cache_dir` defined
# above (~/.cache/sorcha/) if they don't already exist...
class FakeSorchaArgs:
    """Simple wrapper class to mimic the arguments expected by Sorcha methods."""

    def __init__(self, cache_dir=None):
        # Sorcha allows this argument to be None, so simply use that here
        self.ar_data_file_path = cache_dir


def fomo_furnish_spiceypy(cache_dir):
    """A simple wrapper to furnish spiceypy kernels, adapted from the layup version."""
    # A simple class to mimic the arguments processed by Sorcha's observatory class
    auxconfig = auxiliaryConfigs()
    furnish_spiceypy(FakeSorchaArgs(cache_dir), auxconfig)


fomo_furnish_spiceypy(cache_dir)
args = FakeSorchaArgs(cache_dir)
# Create ASSIST ephemeris
ephem, gm_sun, gm_total = create_assist_ephemeris(args, auxiliaryConfigs())
observatories = SorchaObservatory(args, auxiliaryConfigs())


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
    # Generate columns and dtypes explicitly here for BCART_EQ only since we don't
    # need anything else and it removes dependency on ``layup``
    # Will need modifying if we do anything with covariance
    required_column_names = [primary_id_column_name, 'FORMAT', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot', 'epochMJD_TDB']
    default_column_dtypes = ['O', '<U8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
    for col_name, dtype in cols_to_keep:
        # Add the column name and dtype to the default column dtypes
        required_column_names.append(col_name)
        default_column_dtypes.append(dtype)

    # Construct the output dtype for the converted data
    output_dtype = [(col, dtype) for col, dtype in zip(required_column_names, default_column_dtypes, strict=False)]
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


def generate_assist_simulations(ephem, orbit_data):
    """Creates the ASSIST+Rebound simulations for ephemeris generation.
    This is different from the original in
    `sorcha.ephemeris.simulation_setup.generate_simulations()`:
    * Since we are only concerned with a single object, there is no need to
      create a dictionary keyed on the object id.
    * Similarly there is no need to iterate over rows of orbital parameters
      and perform conversion since the `Target`->state vector has already
      been done in `convert_target_to_layup`. This means that there is no need
      for `gm_sun`, `gm_total` to be passed in.
    * The adaptive timestamp criterion (new default) is used and not turned
      off as in sorcha to better resolve close passes.
    * The full EIH GR model is used since we are less concerned for speed (TBD).

    Parameters
    ----------
    ephem : Ephem
        The ASSIST ephemeris object
    orbit_data : numpy structured array
        Converted orbit data

    Returns
    -------
    sim: Simulation
        The REBOUND Simulation object
    ex: Extras
        The ASSIST Extras object for the forces
    """

    epoch = orbit_data['epochMJD_TDB'][0]
    # convert from MJD to JD, if not done already.
    # XXX This is done repeatedly in both directions... this must lose speed and precision Shirley...
    if epoch < 2400000.5:
        epoch += 2400000.5

    # Instantiate a rebound particle
    x, y, z, vx, vy, vz = (orbit_data[col] for col in ['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])
    ic = rebound.Particle(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    sim = rebound.Simulation()
    sim.t = epoch - ephem.jd_ref
    sim.dt = 0.003  # Approx 5 minutes if the units are days (unclear)
    # This turns on the iterative timestep introduced in arXiv:2401.02849 and default since rebound 4.0.3
    sim.ri_ias15.adaptive_mode = 2
    # Add the particle to the simulation
    sim.add(ic)

    # Attach assist extras to the simulation
    ex = assist.Extras(sim, ephem)

    # (Don't) Change the GR model for speed
    forces = ex.forces
    # forces.remove("GR_EIH")
    # forces.append("GR_SIMPLE")
    ex.forces = forces

    return sim, ex


def calculate_rates_and_geometry(pointing: pd.DataFrame, ephem_geom_params: EphemerisGeometryParameters):
    """Calculate rates and geometry for objects within the field of view

    Parameters
    ----------
    pointing : pandas dataframe
        The dataframe containing the pointing database.
    ephem_geom_params : EphemerisGeometryParameters
        Various parameters necessary to calculate the ephemeris

    Returns
    -------
    : tuple
        Tuple containing the ephemeris parameters needed for Sorcha post processing.
    """

    r_sun = get_vec(pointing, 'r_sun')
    r_obs = get_vec(pointing, 'r_obs')
    v_sun = get_vec(pointing, 'v_sun')
    v_obs = get_vec(pointing, 'v_obs')

    ra0, dec0 = vec2ra_dec(ephem_geom_params.rho_hat)
    drhodt = ephem_geom_params.v_ast - v_obs
    drho_magdt = (1 / ephem_geom_params.rho_mag) * np.dot(ephem_geom_params.rho, drhodt)
    ddeltatdt = drho_magdt / SPEED_OF_LIGHT
    drhodt = ephem_geom_params.v_ast * (1 - ddeltatdt) - v_obs
    A, D = get_residual_vectors(ephem_geom_params.rho_hat)
    drho_hatdt = drhodt / ephem_geom_params.rho_mag - drho_magdt * ephem_geom_params.rho_hat / ephem_geom_params.rho_mag
    dradt = np.dot(A, drho_hatdt)
    ddecdt = np.dot(D, drho_hatdt)
    r_ast_sun = ephem_geom_params.r_ast - r_sun
    v_ast_sun = ephem_geom_params.v_ast - v_sun
    r_ast_obs = ephem_geom_params.r_ast - r_obs
    helio_r = np.linalg.norm(r_ast_sun)
    helio_v = np.linalg.norm(v_ast_sun)
    phase_angle = np.arccos(np.dot(r_ast_sun, r_ast_obs) / (helio_r * np.linalg.norm(r_ast_obs)))
    obs_sun = r_obs - r_sun
    dobs_sundt = v_obs - v_sun

    return (
        ephem_geom_params.obj_id,
        pointing['epoch_UTC'],  # replaces FieldID
        ephem_geom_params.mjd_tai,
        pointing['fieldJD_TDB'],
        ephem_geom_params.rho_mag,  # Range_LTC_{km,au}
        drho_magdt,  # RangeRate_LTC_{km_s,au}
        helio_r,  # Helio_LTC_au
        helio_v,  # HelioRate_LTC_au
        ra0,
        dradt * 180 / np.pi,
        dec0,
        ddecdt * 180 / np.pi,
        r_ast_sun[0],
        r_ast_sun[1],
        r_ast_sun[2],
        v_ast_sun[0],
        v_ast_sun[1],
        v_ast_sun[2],
        obs_sun[0],
        obs_sun[1],
        obs_sun[2],
        dobs_sundt[0],
        dobs_sundt[1],
        dobs_sundt[2],
        phase_angle * 180 / np.pi,
    )


def add_magnitude(pandain, H, G=0.15, comet=False):
    """The apparent magnitude is calculated for the given H, G parameters using
    the classic HG phase function (the HG12 phase function (Penttilä et al. 2016,
      PSS 123 117) may be added later). If [comet]=True, the H and G are
    interpreted as the M1, k1 values for the comet magnitude formula.

    Parameters
    ----------
    pandain : Pandas dataframe
        Dataframe of observations.

    H : float
        absolute magnitude (H) in the H, G model or M1 (nuclear absolute magnitude)
    G : float, optional
        slope parameter (G; default 0.15) or k1
    comet : bool, default False
        Whether this is a comet or not (determines which magnitude formula is used)

    Returns
    -------
    pandain : Pandas dataframe
        Dataframe of observations modified with calculated source apparent
        magnitude column ("APmag")
    """

    # first, get rho, delta and alpha as ndarrays
    # delta, rho (r) are already in au and don't need converting from kilometres
    #  unlike original Sorcha. alpha is in degrees
    delta = pandain['Range_LTC_au'].values

    rho = pandain['Helio_LTC_au'].values

    alpha = pandain['phase_deg'].values
    H = np.full(alpha.shape, H)
    G = np.full(alpha.shape, G)

    # This code (from sorcha) doesn't give values that match (~0.05) those
    # from Horizons, our primary source of truth.
    # calculate reduced magnitude and contribution from phase function
    # reduced magnitude = H + 2.5log10(f(phi))

    # HGm = HG(H=H * u.mag, G=G)
    # reduced_mag = HGm(alpha * u.deg).value

    # # apparent magnitude equation: see equation 1 in Schwamb et al. 2023
    # pandain['APmag'] = 5.0 * np.log10(delta) + 5.0 * np.log10(rho) + reduced_mag

    if comet is True:
        pandain['APmag'] = H + 5.0 * np.log10(delta) + G * np.log10(rho)
    else:
        # Calculate phase functions. Likely need an alpha>~120 wackiness check somewhere
        phi1 = np.exp(-3.33 * (np.tan(np.radians(alpha) / 2.0)) ** 0.63)
        phi2 = np.exp(-1.87 * (np.tan(np.radians(alpha) / 2.0)) ** 1.22)

        pandain['APmag'] = H + 5.0 * np.log10(delta * rho) - 2.5 * np.log10((1.0 - G) * phi1 + G * phi2)

    return pandain


def add_sky_motion(pandain, motion_units=u.arcsec / u.minute):
    """Adds the combined sky motion and position angle columns to the passed
    pandas DataFrame `pandain`

    Parameters
    ----------
    pandain : Pandas dataframe
        Dataframe of observations.
    motion_units : astropy.units.CompositeUnit, optional
        Units for output rate of motion, by default u.arcsec/u.minute

    Returns
    -------
    pandain : Pandas dataframe
        Dataframe of observations modified with calculated apparent
        rate of motion ("sky_motion") and position angle ("sky_motion_PA_deg")
        columns added.
    """

    sky_motion = np.sqrt(pandain['RARateCosDec_deg_day'].values ** 2 + pandain['DecRate_deg_day'].values ** 2)
    sky_motion *= u.deg / u.day
    pandain['sky_motion'] = sky_motion.to(motion_units)
    # I doubt this is all that's needed to get the right angle in all cases but we'll see...
    sky_PA = np.atan2(pandain['RARateCosDec_deg_day'], pandain['DecRate_deg_day'])
    pandain['sky_motion_PA_deg'] = np.degrees(sky_PA)

    return pandain


def build_apco_context(pointing, observatory):
    """
    Wrapper for ``erfa.apco``, used in conversions ICRS <-> AltAz/HADec and ICRS <-> CIRS.

    Parameters
    ----------
    pointing : ``pd.Series``
        A single row (``pd.Series``) pulled from the pointings_df instance
        for which to calculate the calculate the astrom values.
    observatory : ``Observatory``
        An observatory
    """
    lon, lat, height = observatory.to_geodetic()
    obstime = Time(pointing['observationMidpointMJD_TAI'], format='mjd', scale='tai')

    jd1_tt, jd2_tt = get_jd12(obstime, 'tt')
    # Polar motion values (interpolated from IERS data) and TIO locator, s' (`sp`)
    xp, yp = get_polar_motion(obstime)
    sp = erfa.sp00(jd1_tt, jd2_tt)
    # Find the X, Y coordinates of the CIP and the CIO locator, s.
    # x, y, s = get_cip(jd1_tt, jd2_tt)
    # xys00a and xys00b provide the equivalent of get_cip() above namely:
    # - pnm06a or pnm06b: provides the BPN matrix from either IAU 2006/2000A or 2000B
    # - bpn2xy: convert BPN matrix to CIP X,Y coordinates
    # - s06 or s00: CIO locator, s
    # Not sure why astropy rolled its own, maybe it was only added to SOFA/erfa later?
    # We're using the 77 term 2000B nutation over the 1361 term 2000A for speed
    x, y, s = erfa.xys00b(jd1_tt, jd2_tt)
    # Earth rotation angle (modern CIO-based equivalent of GST)
    era = erfa.era00(*get_jd12(obstime, 'ut1'))

    # Earth barycentric position and velocity and heliocentric position
    # XXX TODO can almost certainly get this back out of `pointing`
    jd1_tdb, jd2_tdb = get_jd12(obstime, 'tdb')
    earth_pv_heliocentric, earth_pv = erfa.epv00(jd1_tdb, jd2_tdb)
    earth_heliocentric = earth_pv_heliocentric['p']
    # refraction constants
    refa, refb = 0.0, 0.0  # airless apparent (for now) #_refco(frame_or_coord)

    return erfa.apco(
        jd1_tdb,
        jd2_tdb,
        earth_pv,
        earth_heliocentric,
        x,
        y,
        s,
        era,
        lon,
        lat,
        height,
        xp,
        yp,
        sp,
        refa,
        refb,
    )


def split_number_unit_regex(s):
    """
    Matches a number (integer or float) followed by an optional unit
    """

    match = re.match(r'([-+]?\d*\.?\d+)([a-zA-Z%]+)?', s)
    if match:
        number = float(match.group(1))  # Convert to float for numerical operations
        unit = match.group(2) if match.group(2) else ''  # Handle cases with no unit
        return number, unit
    else:
        return None, None


class MakeEphemerisView(FormView):
    """
    View for making an ephemeris
    """

    template_name = 'ephem_form.html'
    form_class = EphemerisForm

    def get_target_id(self):
        """
        Parses the target id for the given observation from the query parameters.

        Returns
        -------
        int
            id (primary key) of the target for ephemeris generation
        """

        if self.request.method == 'GET':
            return self.kwargs['pk']
        elif self.request.method == 'POST':
            return self.request.POST.get('target_id')

    def get_initial(self):
        """
        Populate form's HiddenField with the target_id
        """
        initial = super().get_initial()
        if not self.get_target_id():
            raise Exception('Must provide target_id')
        target_id = self.get_target_id()

        initial['target_id'] = target_id
        initial.update(self.request.GET.dict())
        return initial

    def get_context_data(self, **kwargs):
        """
        Extract the pk from the kwargs and get the Target and add it to the context.
        """
        context = super().get_context_data(**kwargs)

        target_id = self.kwargs['pk']
        context['target'] = Target.objects.get(id=target_id)

        return context

    def get_form(self, form_class=None):
        """
        Form handler
        """
        form = super().get_form()
        target_id = self.get_target_id()

        cancel_url = reverse('home')
        if target_id:
            cancel_url = reverse('tom_targets:detail', kwargs={'pk': target_id})  # + '?tab=ephemeris'
        form.helper.layout = Layout(
            HTML(
                """<p>Fill in the form to generate an ephemeris. If the Site code doesn't already exist you will be
                 redirected to the Observatory creation form to make it.</p>"""
            ),
            'target_id',
            'start_date',
            'end_date',
            'step',
            'site_code',
            'full_precision',
            FormActions(
                Submit('confirm', 'Create Ephemeris'),
                HTML(f'<a class="btn btn-outline-primary" href={cancel_url}>Cancel</a>'),
            ),
        )
        return form

    def form_valid(self, form: EphemerisForm) -> HttpResponse:
        """form validator for ephemeris generation
        Checks to see if there is a `Observatory` for the requested site_code and converts
        the start time to a naive datetime

        Parameters
        ----------
        form : EphemerisForm
            The filled-in form for validation.

        Returns
        -------
        HttpResponse
            A redirect either to the ephemeris generator (``ephem`` View) with url parameters or
            to the Observatory creation form (``solsys_code_observatory:create``) if the requested `obscode`
            doesn't exist.
        """
        print('In form_valid: ', end='')
        obscode = form.cleaned_data['site_code']
        try:
            _ = Observatory.objects.get(obscode=obscode)
        except Observatory.DoesNotExist:
            return redirect('solsys_code_observatory:create')
        start = form.cleaned_data['start_date']
        # Not sure we want to deal with the horrors of local timezones but as first step, convert it to UTC
        # and then make it naive (as astropy.Time in Ephmeris() can't handle non-naive `datetime`s)
        print(start, start.tzinfo)
        # Convert to UTC (still timezone aware at this stage)
        utc_start = start.astimezone(timezone.utc)
        utc_start = utc_start.replace(tzinfo=None)
        step = form.cleaned_data['step']
        full_precision = form.cleaned_data['full_precision']
        url = (
            reverse('ephem', kwargs={'pk': form.cleaned_data['target_id']})
            + f'?obscode={obscode}&start={utc_start.isoformat()}&step={step}&full_precision={full_precision}'
        )
        print(url)
        return redirect(url)


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
        full_precision = False
        if request.GET.get('full_precision', 'False').lower() in ['true', '1', 'yes']:
            full_precision = True
        # Construct time series of `Time` objects in UTC.
        # XXX Todo: better initialize start, stop, step from query parameters
        start_time = request.GET.get('start', None)
        if start_time is None:
            start_time = Time.now()
            start_time = Time(start_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0), scale='utc')
        else:
            try:
                start_time = Time(start_time, scale='utc')
            except ValueError:
                start_time = Time.now()
                start_time = Time(start_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0), scale='utc')
        step = request.GET.get('step', None)
        if step is None:
            step_size = 1 * u.day
        else:
            number, unit_str = split_number_unit_regex(step)
            unit = u.day
            step_size = number if number is not None else 1
            if unit_str is not None:
                # Do unit handling here
                try:
                    unit = u.Unit(unit_str)
                except ValueError:
                    pass
            step_size *= unit
        ts = TimeSeries(time_start=start_time, time_delta=step_size, n_samples=20)
        # Generate a list of JD_TDB times
        times = ts.time.tdb.jd

        data = convert_target_to_layup(target)

        # Assemble stuff needed for sorcha's version of `integrate_light_time`
        sim, ex = generate_assist_simulations(ephem, data)

        output = StringIO()
        in_memory_csv = writer(output)

        column_names = (
            'ObjID',
            'epoch_UTC',
            'fieldMJD_TAI',
            'fieldJD_TDB',
            'Range_LTC_au',
            'RangeRate_LTC_au_s',
            'Helio_LTC_au',
            'HelioRate_LTC_au',
            'RA_deg',
            'RARateCosDec_deg_day',
            'Dec_deg',
            'DecRate_deg_day',
            'Obj_Sun_x_LTC_au',
            'Obj_Sun_y_LTC_au',
            'Obj_Sun_z_LTC_au',
            'Obj_Sun_vx_LTC_au_s',
            'Obj_Sun_vy_LTC_au_s',
            'Obj_Sun_vz_LTC_au_s',
            'Obs_Sun_x_au',
            'Obs_Sun_y_au',
            'Obs_Sun_z_au',
            'Obs_Sun_vx_au_s',
            'Obs_Sun_vy_au_s',
            'Obs_Sun_vz_au_s',
            'phase_deg',
            'Obs_Az_deg',
            'Obs_Alt_deg',
            'Obs_HA_deg',
        )
        column_types = defaultdict(ObjID=str, FieldID=str).setdefault(float)  # type: ignore
        in_memory_csv.writerow(column_names)

        # Make equivalent `pointings_df`
        pointings_df = pd.DataFrame()
        pointings_df['epoch_UTC'] = ts.time
        pointings_df['fieldJD_TDB'] = times
        pointings_df['observationMidpointMJD_TAI'] = ts.time.tai.mjd
        # et_sun = spice.str2et(f'jd {epoch_tdb.jd} tdb')
        # sun_posvel, sun_ltt = spice.spkezr('SUN', et_sun, 'J2000', 'NONE', 'SSB')
        # # Convert from km and km/s to AU and AU/day
        # sun_posvel /= AU_KM
        # sun_posvel[3:6] *= SEC_PER_DAY

        # Create ET for SPICE
        et = (pointings_df['fieldJD_TDB'] - spice.j2000()) * SEC_PER_DAY

        # create empty arrays for observatory position and velocity to be filled in
        r_obs = np.empty((len(pointings_df), 3))
        v_obs = np.empty((len(pointings_df), 3))

        # SSB->observatory position and velocity vectors
        for idx, et_i in enumerate(et):
            r_obs[idx], v_obs[idx] = barycentricObservatoryRates(et_i, obscode, observatories=observatories)

        r_obs /= AU_KM  # convert to au
        v_obs *= SEC_PER_DAY / AU_KM  # convert to au/day

        pointings_df['r_obs_x'] = r_obs[:, 0]
        pointings_df['r_obs_y'] = r_obs[:, 1]
        pointings_df['r_obs_z'] = r_obs[:, 2]
        pointings_df['v_obs_x'] = v_obs[:, 0]
        pointings_df['v_obs_y'] = v_obs[:, 1]
        pointings_df['v_obs_z'] = v_obs[:, 2]

        # create empty arrays for sun position and velocity to be filled in
        r_sun = np.empty((len(pointings_df), 3))
        v_sun = np.empty((len(pointings_df), 3))
        time_offsets = pointings_df['fieldJD_TDB'] - ephem.jd_ref
        for idx, time_offset_i in enumerate(time_offsets):
            sun = ephem.get_particle('Sun', time_offset_i)
            r_sun[idx] = np.array((sun.x, sun.y, sun.z))
            v_sun[idx] = np.array((sun.vx, sun.vy, sun.vz))

        pointings_df['r_sun_x'] = r_sun[:, 0]
        pointings_df['r_sun_y'] = r_sun[:, 1]
        pointings_df['r_sun_z'] = r_sun[:, 2]
        pointings_df['v_sun_x'] = v_sun[:, 0]
        pointings_df['v_sun_y'] = v_sun[:, 1]
        pointings_df['v_sun_z'] = v_sun[:, 2]

        # Generate ephemeris
        for _, pointing in pointings_df.iterrows():
            mjd_tai = float(pointing['observationMidpointMJD_TAI'])
            r_obs = get_vec(pointing, 'r_obs')
            ephem_geom_params = EphemerisGeometryParameters()
            ephem_geom_params.obj_id = target.name
            ephem_geom_params.mjd_tai = mjd_tai
            (
                ephem_geom_params.rho,
                ephem_geom_params.rho_mag,
                ltt,
                ephem_geom_params.r_ast,
                ephem_geom_params.v_ast,
            ) = integrate_light_time(sim, ex, pointing['fieldJD_TDB'] - ephem.jd_ref, r_obs, lt0=0.01)
            ephem_geom_params.rho_hat = ephem_geom_params.rho / ephem_geom_params.rho_mag

            out_tuple = calculate_rates_and_geometry(pointing, ephem_geom_params)
            # Transform from ICRS RA, Dec -> observed Alt, Az, HA
            # Assemble astrometric context
            astrom = build_apco_context(pointing, observatory)
            # Transform to CIRS (can easily transform further to apparent RA, Dec if needed)
            cirs_ra, cirs_dec = erfa.atciqz(np.radians(out_tuple[8]), np.radians(out_tuple[10]), astrom)
            # Transform from CIRS->observed
            obs_az, obs_zd, obs_ha, obs_dec, obs_ra = erfa.atioq(cirs_ra, cirs_dec, astrom)
            # Convert zenith distance to altitude (in degrees)
            obs_alt = np.degrees(PI_OVER_2 - obs_zd)
            out_tuple = out_tuple + (np.degrees(obs_az), obs_alt, np.degrees(obs_ha))

            in_memory_csv.writerow(out_tuple)
        output.seek(0)
        predictions = pd.read_csv(output, dtype=column_types)
        # Add magnitude column
        H = target.extra_fields.get('H', 22.0)
        G = target.extra_fields.get('G', 0.15)
        comet = False
        if target.scheme == 'MPC_COMET':
            comet = True
        predictions = add_magnitude(predictions, H, G, comet)
        # Add sky motion rate column
        predictions = add_sky_motion(predictions)
        ephem_lines = []
        for _, e in predictions.iterrows():
            ephem_line = [
                e['epoch_UTC'],
                e['RA_deg'],
                e['Dec_deg'],
                e['Obs_Az_deg'],
                e['Obs_Alt_deg'],
                e['APmag'],
                e['Helio_LTC_au'],
                e['Range_LTC_au'],
                e['phase_deg'],
                e['sky_motion'],
                e['sky_motion_PA_deg'],
            ]

            ephem_lines.append(ephem_line)
        return render(
            request,
            'ephem.html',
            {
                'target': target,
                'ephem_lines': ephem_lines,
                'observatory': observatory,
                'full_precision': full_precision,
            },
        )
