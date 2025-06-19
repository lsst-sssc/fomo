from astropy import units as u
from astropy.coordinates import SkyCoord
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
        url += f'&obs-time={obs_time.utc.to_datetime().strftime(time_fmt)}'
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
    def query_center(self, obs_time: Time, center: SkyCoord, verbose: bool = True):
        """
        Query for small bodies around <center> (a SkyCoord in the ICRS frame) at <obs_time> (a Time instance)
        """

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

        return url
