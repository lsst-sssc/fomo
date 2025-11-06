import re
from collections import defaultdict
from csv import writer
from datetime import timezone
from io import StringIO
from math import ceil

import erfa
import numpy as np
import pandas as pd
import spiceypy as spice
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
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
                try:
                    unit = u.Unit(unit_str)
                except ValueError:
                    pass
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
