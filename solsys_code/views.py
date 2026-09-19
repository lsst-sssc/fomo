import json
import logging
import re
import urllib.parse
from datetime import timezone
from math import ceil, isnan
from typing import Any

import requests
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.generic import FormView, View
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

from .ephem_utils import compute_ephemeris
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
        # Default to Rubin (X05) when no site is given. The geocentre (500) is supported by ephem_utils but,
        # like any other site, needs an Observatory row to exist.
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

        predictions = compute_ephemeris(target, observatory, ts.time)

        ephem_lines = []
        for _, e in predictions.iterrows():
            # Az/Alt are NaN for the geocentre; pass None so the template can show 'n.a.'
            ephem_line = [
                e['epoch_UTC'],
                e['RA_deg'],
                e['Dec_deg'],
                None if isnan(e['Obs_Az_deg']) else e['Obs_Az_deg'],
                None if isnan(e['Obs_Alt_deg']) else e['Obs_Alt_deg'],
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

    def run_query(self) -> dict[str, Any] | None:
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
