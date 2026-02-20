import json
import logging
import re
import urllib.parse
from collections import defaultdict
from csv import writer
from datetime import timezone
from io import StringIO
from math import ceil
from typing import Any, Optional

import erfa
import numpy as np
import pandas as pd
import requests
import spiceypy as spice
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.generic import FormView, View
from sorcha.ephemeris.simulation_driver import EphemerisGeometryParameters, get_vec
from sorcha.ephemeris.simulation_geometry import (
    barycentricObservatoryRates,
    integrate_light_time,
)
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

from .ephem_utils import (
    AU_KM,
    PI_OVER_2,
    SEC_PER_DAY,
    add_magnitude,
    add_sky_motion,
    build_apco_context,
    calculate_rates_and_geometry,
    convert_target_to_layup,
    ephem,
    generate_assist_simulations,
    observatories,
)
from .forms import EphemerisForm


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
        # print('In form_valid: ', end='')
        obs = form.cleaned_data['site_code']
        # Retrieve the start and end times out of the cleaned Form data.
        # Not sure we want to deal with the horrors of local timezones but as first step, convert it to UTC
        # and then make it naive (as astropy.Time in Ephemeris() can't handle non-naive `datetime`s)
        start = form.cleaned_data['start_date']
        # Convert to UTC (still timezone aware at this stage)
        utc_start = start.astimezone(timezone.utc)
        # Replace timezone info making it naive
        utc_start = utc_start.replace(tzinfo=None)

        end = form.cleaned_data['end_date']
        utc_end = end.astimezone(timezone.utc)
        utc_end = utc_end.replace(tzinfo=None)

        step = form.cleaned_data['step']
        full_precision = form.cleaned_data['full_precision']
        url = (
            reverse('ephem', kwargs={'pk': form.cleaned_data['target_id']})
            + f'?obscode={obs.obscode}&start={utc_start.isoformat()}&stop={utc_end.isoformat()}'
            + f'&step={step}&full_precision={full_precision}'
        )
        # print(url)
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
        # Sorcha doesn't support the geocenter (code 500), so until we have our own version
        # of `barycentricObservatoryRates()` which does, default to something else e.g. Rubin (X05) in this case
        obscode = request.GET.get('obscode', 'X05')
        # XXX Could replace this by a creation of the missing Observatory
        # relatively easily
        observatory = get_object_or_404(Observatory, obscode=obscode)
        full_precision = False
        if request.GET.get('full_precision', 'False').lower() in ['true', '1', 'yes']:
            full_precision = True
        # Construct time series of `Time` objects in UTC.
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
        end_time = request.GET.get('stop', None)
        if end_time is None:
            end_time = start_time + TimeDelta(20 * u.day)
            end_time = Time(end_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0), scale='utc')
        else:
            try:
                end_time = Time(end_time, scale='utc')
            except ValueError:
                end_time = start_time + TimeDelta(20 * u.day)
                end_time = Time(end_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0), scale='utc')
        step = request.GET.get('step', None)
        if step is None:
            step_size = 1 * u.day
        else:
            number, unit_str = split_number_unit_regex(step)
            unit = u.day
            step_size = number if number is not None else 1
            if unit_str is not None:
                # Do unit handling here
                err_msg = f'Unit {unit_str} is not compatible with time units, defaulting to days'
                try:
                    unit = u.Unit(unit_str)
                    # Check that unit is compatible with time
                    if not unit.is_equivalent(u.day):
                        # See if we got 'm' for minutes first (which would convert to `Unit('meter')`...), first
                        if unit_str == 'm':
                            unit = u.min
                        else:
                            # Bail on trying to read users' mind and default to days
                            messages.warning(request, err_msg)
                            unit = u.day
                except ValueError:
                    messages.warning(request, err_msg)
            step_size *= unit
        n_steps = (end_time - start_time) / step_size
        ts = TimeSeries(time_start=start_time, time_delta=step_size, n_samples=ceil(n_steps) + 1)
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
        H = target.abs_mag or 22.0
        G = target.slope or 0.15
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


class JPLSBDBQuery:
    """
    The ``JPLSBDBQuery`` provides an interface to JPL's Small Body Database Query
    via its API interface (https://ssd.jpl.nasa.gov/tools/sbdb_query.html)
    """

    base_url = 'https://ssd-api.jpl.nasa.gov/sbdb_query.api'

    _CHAIN_PATTERN = re.compile(
        r"""
        ^\s*
        (?P<a>.+?)\s*
        (?P<op1><=|<|>=|>)\s*
        (?P<field>[A-Za-z_][A-Za-z0-9_\.]*)\s*
        (?P<op2><=|<|>=|>)\s*
        (?P<b>.+?)\s*
        $
        """,
        re.VERBOSE,
    )

    def __init__(self, orbit_class=None, orbital_constraints=None):
        """
        orbit_class: str or None (e.g. 'IEO', 'TJN', etc.)
        orbital_constraints: list of constraint strings, e.g. ['q|LT|1.3', 'i|LT|10.5']
        """
        if orbit_class is None and orbital_constraints is None:
            orbital_constraints = ['e>=1.2']
        self.orbit_class = orbit_class
        self.orbital_constraints_raw = orbital_constraints or []
        self.orbital_constraints = self._translate_constraints(self.orbital_constraints_raw)

    def _translate_constraints(self, constraints):
        translated = []

        for c in constraints:
            s = c.strip()
            lower = s.lower()

            if lower.endswith('is defined'):
                field = s[: -len(' is defined')].strip()
                if field == '':
                    raise ValueError(f'Invalid "is defined" constraint (missing field): {c}')
                translated.append(f'{field}|DF')
                continue

            if lower.endswith('is not defined'):
                field = s[: -len(' is not defined')].strip()
                if field == '':
                    raise ValueError(f'Invalid "is not defined" constraint (missing field): {c}')
                translated.append(f'{field}|ND')
                continue

            # Between 2 values
            m = self._CHAIN_PATTERN.match(s)
            if m:
                a = m.group('a').strip()
                op1 = m.group('op1')
                field = m.group('field').strip()
                op2 = m.group('op2')
                b = m.group('b').strip()

                lt_like = {'<', '<='}
                gt_like = {'>', '>='}

                if op1 in lt_like and op2 in lt_like:
                    # a (min) op1 field op2 b (max)
                    min_val, max_val = a, b
                    left_incl = op1 == '<='
                    right_incl = op2 == '<='

                elif op1 in gt_like and op2 in gt_like:
                    # a (max) op1 field op2 b (min)
                    min_val, max_val = b, a
                    left_incl = op2 == '>='
                    right_incl = op1 == '>='

                else:
                    raise ValueError(f'Unsupported chained comparison direction (must both point same way): {c}')

                # Only allow both-inclusive (RG) or both-exclusive (RE)
                if left_incl and right_incl:
                    translated.append(f'{field}|RG|{min_val}|{max_val}')
                elif (not left_incl) and (not right_incl):
                    translated.append(f'{field}|RE|{min_val}|{max_val}')
                else:
                    raise ValueError(f'Mixed inclusive/exclusive ranges not supported (use <=...<= or <...< ): {c}')

                continue

            # Single value
            if '<=' in s:
                field, value = s.split('<=', 1)
                translated.append(f'{field.strip()}|LE|{value.strip()}')
            elif '>=' in s:
                field, value = s.split('>=', 1)
                translated.append(f'{field.strip()}|GE|{value.strip()}')
            elif '<' in s:
                field, value = s.split('<', 1)
                translated.append(f'{field.strip()}|LT|{value.strip()}')
            elif '>' in s:
                field, value = s.split('>', 1)
                translated.append(f'{field.strip()}|GT|{value.strip()}')
            elif '==' in c:
                field, value = c.split('==')
                translated.append(f'{field.strip()}|EQ|{value.strip()}')
            elif '!=' in c:
                field, value = c.split('!=')
                translated.append(f'{field.strip()}|NE|{value.strip()}')
            else:
                raise ValueError(f'Unsupported constraint format: {c}')

        return translated

    def build_query_url(self):
        """
        Build a query for the JPL SBDB service.
        """
        # Base query fields
        params = {
            'fields': 'pdes,prefix,epoch_mjd,e,a,q,i,om,w,tp,H,G,M1,K1,condition_code,data_arc,n_obs_used',
            'full-prec': 'true',
            'sb-xfrag': 'true',
        }

        # Add sb-class if provided
        if self.orbit_class:
            params['sb-class'] = self.orbit_class

        # Add sb-cdata if constraints provided
        if self.orbital_constraints:
            constraint_obj = {'AND': self.orbital_constraints}
            json_str = json.dumps(constraint_obj, separators=(',', ':'))
            encoded_cdata = urllib.parse.quote(json_str)
            params['sb-cdata'] = encoded_cdata

        # Build URL
        query_parts = [f'{key}={str(value)}' for key, value in params.items()]
        url = f'{self.base_url}?' + '&'.join(query_parts)
        self.url = url
        return url

    def run_query(self) -> Optional[dict[str, Any]]:
        """
        Execute the query and return results as JSON (if successful).
        """
        url = self.build_query_url()
        resp = requests.get(url)

        if resp.ok:
            return resp.json()
        else:
            logger = logging.getLogger(__name__)
            logger.debug(f'Query failed with status {resp.status_code}')
            return None

    def parse_results(self, results: dict[str, Any]) -> QTable:
        """
        Parse JSON results into an Astropy QTable.
        """
        if not results or 'data' not in results:
            logger = logging.getLogger(__name__)
            logger.debug('No data found in results')
            self.results_table = QTable()
            return self.results_table

        data = results['data']
        columns = results['fields']
        self.results_table = QTable(rows=data, names=columns)
        return self.results_table

    def create_targets(self) -> list:
        """
        Create TOM Targets from JPL SBDB Query. Returns a list of the newly created `Target`s.

        Returns
        -------
        list
            A list of the newly created `Target` objects (or an empty list if the needed `self.results_table`
            is empty.
        """

        new_targets = []
        if not getattr(self, 'results_table', None):
            return new_targets
        for result in self.results_table:
            asteroid = True
            name = result['pdes']
            if result['prefix'] in ['C', 'A', 'P', 'D']:
                if name[-1:] == 'P' and result['prefix'] == 'P':
                    # Numbered periodic comet, don't add prefix
                    pass
                else:
                    name = result['prefix'] + '/' + name
            existing_objects = Target.objects.filter(name=name)
            if existing_objects.count() == 0:
                target = Target()
                target.type = 'NON_SIDEREAL'
                if result['prefix'] is None:
                    target.scheme = 'MPC_MINOR_PLANET'
                else:
                    target.scheme = 'MPC_COMET'
                    asteroid = False
                target.name = name
                target.arg_of_perihelion = result['w']  # argument of the perifocus in JPL
                target.lng_asc_node = result['om']  # longitude of asc. node in JPL
                target.inclination = result['i']  # inclination in JPL
                target.semimajor_axis = result['a']  # semi-major axis in JPL
                target.eccentricity = result['e']  # eccentricity in JPL
                target.epoch_of_elements = result['epoch_mjd']  # epoch Julian Date in JPL
                target.perihdist = result['q']  # periapsis distance in JPL
                # convert to mjd from jd (preserving precision)
                try:
                    target.epoch_of_perihelion = float(result['tp'][2:]) - 0.5
                except (IndexError, TypeError):
                    # Already not a string (or None)
                    try:
                        target.epoch_of_perihelion = float(result['tp']) - 2400000.5
                    except (ValueError, TypeError):
                        pass
                target.orbitcode = result['condition_code']
                target.data_arc = result['data_arc']
                target.n_obs_used = result['n_obs_used']
                # Extract absolute magnitude (H) and slope (G) or M1, k1 for comets
                # Default to G=0.15 for asteroids, no instances of comets with M1 defined but k1 not defined were found
                if asteroid:
                    target.abs_mag = result['H']
                    target.slope = result['G'] if result['G'] is not None else 0.15
                else:
                    target.abs_mag = result['M1']
                    target.slope = result['K1']
                target.save()
                new_targets.append(target)
        return new_targets


class JPLSBId:
    """
    The ``JPLSBId`` provides an interface to JPL's Small Body Identification Tool
    via its API interface (https://ssd-api.jpl.nasa.gov/doc/sb_ident.html)
    """

    # XXX better in settings or a conf?
    base_url = 'https://ssd-api.jpl.nasa.gov/sb_ident.api'

    @u.quantity_input
    def __init__(
        self,
        mpc_code='X05',
        fov_ra_hwidth: u.Quantity[u.deg] = 1.75 * u.deg,
        fov_dec_hwidth: u.Quantity[u.deg] = 1.75 * u.deg,
    ) -> None:
        self.mpc_code = mpc_code.upper()
        if self.mpc_code == '500':
            raise ValueError("MPC site code '500' (geocenter) is not valid for this service")
        self.fov_ra_hwidth = fov_ra_hwidth.to(u.deg)
        self.fov_dec_hwidth = fov_dec_hwidth.to(u.deg)
        self.two_pass = False  # whether to request the 2nd numerical integration pass rather than two-body model
        self.mag_required = True  # skip objects without magnitude parameters
        self.elems_required = False  # whether to request orbital elements

    def _build_base_query(self):
        """
        Build the base query from things we already know about such as the site code etc
        """

        url = f'{self.base_url}?mpc-code={self.mpc_code}&mag-required={str(self.mag_required).lower()}'
        url += f'&two-pass={str(self.two_pass).lower()}'
        return url

    @u.quantity_input
    def _build_center_query(self, obs_time: Time, center: SkyCoord):
        """
        Build query for small bodies around <center> (a SkyCoord in the ICRS frame) at <obs_time> (a Time instance)
        """

        url = self._build_base_query()
        # XXX may need '_' not 'T', docs inconsistent
        time_fmt = '%Y-%m-%dT%H:%M:%S'
        url += f'&obs-time={obs_time.utc.strftime(time_fmt)}'
        # Add RA string
        url += f"&fov-ra-center={center.ra.to_string(u.hourangle, sep='-', precision=0, pad=True)}"
        # Add Dec string
        dec_string = center.dec.to_string(u.deg, sep='-', precision=0, pad=True)
        # Turn minus sign into 'M' needed by API
        dec_string = dec_string[0].replace('-', 'M') + dec_string[1:]
        url += f'&fov-dec-center={dec_string}'
        # Add RA and Dec half-widths
        url += f'&fov-ra-hwidth={self.fov_ra_hwidth.to(u.deg).value:.2f}'
        url += f'&fov-dec-hwidth={self.fov_dec_hwidth.to(u.deg).value:.2f}'

        return url

    @u.quantity_input
    def _build_limits_query(self, obs_time: Time, center: SkyCoord):
        """
        Build query for small bodies around within limits <center.ra> - fov_ra_hwidth ... <center.ra> + fov_ra_hwidth
        and <center.dec> - fov_dec_hwidth ... <center.dec> + fov_dec_hwidth
        where <center> is a SkyCoord in the ICRS frame at <obs_time> (a Time instance)
        """

        url = self._build_base_query()
        # XXX may need '_' not 'T', docs inconsistent
        time_fmt = '%Y-%m-%dT%H:%M:%S'
        url += f'&obs-time={obs_time.utc.strftime(time_fmt)}'
        # Add RA string
        ra_lower_lim = center.ra - self.fov_ra_hwidth
        ra_upper_lim = center.ra + self.fov_ra_hwidth
        url += (
            f"&fov-ra-lim={ra_lower_lim.to_string(u.hourangle, sep='-', precision=0, pad=True)},"
            + f"{ra_upper_lim.to_string(u.hourangle, sep='-', precision=0, pad=True)}"
        )
        # Add Dec string
        dec_lower_lim = center.dec - self.fov_dec_hwidth
        dec_upper_lim = center.dec + self.fov_dec_hwidth
        dec_lower_string = dec_lower_lim.to_string(u.deg, sep='-', precision=0, pad=True)
        # Turn minus sign into 'M' needed by API
        dec_lower_string = dec_lower_string[0].replace('-', 'M') + dec_lower_string[1:]
        dec_upper_string = dec_upper_lim.to_string(u.deg, sep='-', precision=0, pad=True)
        # Turn minus sign into 'M' needed by API
        dec_upper_string = dec_upper_string[0].replace('-', 'M') + dec_upper_string[1:]
        url += f'&fov-dec-lim={dec_lower_string},{dec_upper_string}'

        return url

    def make_query(self, url):
        """
        Executes query in <url>. If the response status is good, the results are returned as JSON
        dict structure and the ['signature']['version'] is checked for the known current version (1.1)
        In the case of error, None is returned.
        """
        results = None
        resp = requests.get(url)
        # print("status code=", resp.status_code, resp.ok is True)
        if resp.ok is True:
            # print("Response OK")
            results = resp.json()
            assert results['signature']['version'] == '1.1'

        return results

    def parse_results(self, results: dict[str, Any]) -> QTable:
        """
        Parses the JSON results from the JPL SBID API service returned via ``make_query()` and returns a
        QTable. The seperate Astrometric RA/Dec columns in the original are merged into a single SkyCoord
        column. Other fields with units are turned into Quantity columns.
        """
        table = None

        if results and len(results) > 0 and ('fields_first' in results or 'fields_second' in results):
            column_names = [
                'Object name',
                'Astrometric position',
                'Dist. from center RA',
                'Dist. from center Dec',
                'Dist. from center Norm',
                'V magnitude',
                'RA rate',
                'Dec rate',
                'Pos error RA',
                'Pos error Dec',
            ]
            names = []
            ras = []
            decs = []
            ra_dist = []
            dec_dist = []
            norm_dist = []
            mags = []
            rates_ra = []
            rates_dec = []
            poserr_ra = []
            poserr_dec = []
            numbered_object = re.compile(r'^(\d*) ')
            unnumbered_object = re.compile(r'^\((\d{4}\s[A-Z]{2}\d*)\)')
            comets = re.compile(r'^(\d+[I,P])|([C,P,A]/\d{4} [A-H,J-Y]\d+)')
            results_key = 'data_first_pass'
            have_errors = True
            if 'data_second_pass' in results:
                results_key = 'data_second_pass'
                if len(results['fields_second']) < len(column_names):
                    column_names = column_names[0 : len(results['fields_second']) - 1]
                    have_errors = False
            for sb in results[results_key]:
                # Try to match numbered objects first
                name = sb[0]
                match = re.search(numbered_object, name)
                if match:
                    new_name = match.groups()[0]
                else:
                    # Un-numbered/provisional desiginations
                    match = re.search(unnumbered_object, name)
                    if match:
                        new_name = match.groups()[0]
                    else:
                        # Final try for comets
                        match = re.search(comets, name)
                        new_name = match[0] if match else name
                names.append(new_name)
                ra = Angle(sb[1], unit=u.hourangle)
                dec = Angle(sb[2].replace(' ', 'd').replace("'", 'm').replace('"', 's'), unit=u.deg)
                ras.append(ra)
                decs.append(dec)
                ra_dist.append(float(sb[3]) * u.arcsec)
                dec_dist.append(float(sb[4]) * u.arcsec)
                norm_dist.append(float(sb[5]) * u.arcsec)
                try:
                    mag = float(sb[6])
                except ValueError:
                    # Trim off 'N' or 'T' letter at end
                    mag = float(sb[6][:-1])
                mags.append(mag)
                ra_rate = float(sb[7]) * (u.arcsec / u.hour)
                rates_ra.append(ra_rate)  # .to(u.arcsec/u.minute))
                dec_rate = float(sb[8]) * (u.arcsec / u.hour)
                rates_dec.append(dec_rate)  # .to(u.arcsec/u.minute))
                if have_errors:
                    poserr_ra.append(float(sb[9]) * u.arcsec)
                    poserr_dec.append(float(sb[10]) * u.arcsec)
            # Assemble table from columns
            position = SkyCoord(ras, decs, frame='icrs')
            columns = [names, position, ra_dist, dec_dist, norm_dist, mags, rates_ra, rates_dec]
            if have_errors:
                columns += [poserr_ra, poserr_dec]
            table = QTable(columns, names=column_names)

        return table

    @u.quantity_input
    def query_center(self, obs_time: Time, center: SkyCoord, raw_response: bool = True, verbose: bool = True):
        """
        Query for small bodies around <center> (a SkyCoord in the ICRS frame) at <obs_time> (a Time instance)
        Returns either a parsed QTable of the small bodies (if [raw_response] is False) or the raw JSON
        response from the JPL service (if [raw_resonse] is True). In the event of an error from the API
        endpoint, the query url is returned.
        """

        results = None

        if not isinstance(obs_time, Time):
            obs_time = Time(obs_time, scale='utc')
        if not isinstance(center, SkyCoord):
            if isinstance(center, [tuple, list]):
                center = SkyCoord(center[0], center[1], frame='icrs')
            else:
                raise ValueError('center must be a SkyCoord or a pair of values that can initialize one')
        if verbose:
            print(f'Querying around ({center.ra.deg:.3f}, {center.dec.deg:+.2f}) at {obs_time.utc} UTC')
        url = self._build_center_query(obs_time, center)

        if url:
            if verbose:
                print(url)
            results = self.make_query(url)
            if results is not None:
                if verbose:
                    print(f"Found {results['n_first_pass']} small bodies in FOV")
                if raw_response is False:
                    results = self.parse_results(results)
            else:
                results = url
        return results
