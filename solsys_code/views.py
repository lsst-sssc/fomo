import re
from typing import Any

import requests
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
from astropy.time import Time

# from django.shortcuts import render


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

        if results and len(results) > 0 and 'fields_first' in results:
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
            for sb in results['data_first_pass']:
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
                mags.append(float(sb[6]))
                ra_rate = float(sb[7]) * (u.arcsec / u.hour)
                rates_ra.append(ra_rate)  # .to(u.arcsec/u.minute))
                dec_rate = float(sb[8]) * (u.arcsec / u.hour)
                rates_dec.append(dec_rate)  # .to(u.arcsec/u.minute))

                poserr_ra.append(float(sb[9]) * u.arcsec)
                poserr_dec.append(float(sb[10]) * u.arcsec)
            # Assemble table from columns
            position = SkyCoord(ras, decs, frame='icrs')
            table = QTable(
                [names, position, ra_dist, dec_dist, norm_dist, mags, rates_ra, rates_dec, poserr_ra, poserr_dec],
                names=column_names,
            )

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
            results = self.make_query(url)
            if results is not None:
                if verbose:
                    print(f"Found {results['n_first_pass']} small bodies in FOV")
                results = self.parse_results(results)
            else:
                results = url
        return results
