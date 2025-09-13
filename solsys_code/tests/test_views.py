from collections import namedtuple

import erfa
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from django.test import SimpleTestCase, TestCase, tag
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from tom_targets.models import Target

from solsys_code.solsys_code_observatory.models import Observatory

# Import module to test
from solsys_code.views import add_magnitude, add_sky_motion, build_apco_context, convert_target_to_layup

MJD_TO_JD_CONVERSION = 2400000.5
JD2000 = 2451545.0  # Reference epoch
CR = 299792.458  # speed of light in km/s


class TestConvertTargetToLayup(TestCase):
    def setUp(self) -> None:
        epochJD_TDB = 2457545.5
        target_params = {
            'name': 'Fake Rock',  # actually (3666) Holman
            'type': Target.NON_SIDEREAL,
            'scheme': 'MPC_COMET',
            'eccentricity': 0.1273098035049758,
            'perihdist': 2.719440725596252,
            'inclination': 2.363582123773087,
            'lng_asc_node': 120.3869311659506,
            'arg_of_perihelion': 55.06308037812056,
            'epoch_of_perihelion': 57934.05265870551,
            'epoch_of_elements': epochJD_TDB - MJD_TO_JD_CONVERSION,
        }
        self.target, created = Target.objects.get_or_create(**target_params)

        # barycentric cartesian (equatorial) - these are the reference values for comparison
        x_bary_eq = -7.195156074800051e-02
        y_bary_eq = 2.800941663957977e00
        z_bary_eq = 1.148299189842545e00
        vx_bary_eq = -9.914826873812788e-03
        vy_bary_eq = -1.508913222991139e-03
        vz_bary_eq = -2.356455160257992e-04
        self.bary_vec = np.array([x_bary_eq, y_bary_eq, z_bary_eq, vx_bary_eq, vy_bary_eq, vz_bary_eq])
        # Sun's position at epochJD_TDB (from Horizons Vector Table with
        # Target Body: 10 (Sun), Coord Center: @0 (SSB))
        Sun = namedtuple('Sun', 'x y z vx vy vz')
        sun_epoch = Sun(
            x=3.743893517879733e-03,
            y=2.355922092887896e-03,
            z=8.440770737482685e-04,
            vx=-7.096646739414067e-07,
            vy=6.421467712437571e-06,
            vz=2.788964122162865e-06,
        )
        self.sun_dict = {epochJD_TDB: sun_epoch}

    def test_provided_sun_dict(self):
        converted = convert_target_to_layup(self.target, self.sun_dict)
        for name, j in zip(converted.dtype.names[2:8], range(6)):
            assert_almost_equal(converted[name], self.bary_vec[j], 8)

    @tag('spiceypy')
    def test_sun_ephemeris(self):
        """test conversion with in-place Sun position determination
        This tests needs SPICE kernels available (TBD but at least:
        * naif0012.tls (leap second kernel)
        * de440s.bsp (Short version of DE440 ephemeris)
        )"""

        converted = convert_target_to_layup(self.target)
        for name, j in zip(converted.dtype.names[2:8], range(6)):
            assert_almost_equal(converted[name], self.bary_vec[j], 8)


class TestAddMagnitude(SimpleTestCase):
    def test_asteroid_no_default_G(self):
        expected_mags = [17.052, 16.838]

        # Values for (1627) Ivar from K92 calculated by JPL Horizons on 2025-08-25
        # using  soln ref.= JPL#1496
        obs_df = pd.DataFrame(
            {
                'epoch_UTC': ['2025-08-21 00:00:00', '2025-09-09 00:00:00'],
                'Range_LTC_au': np.array([2.068039560205, 1.982726563532]),
                'Helio_LTC_au': np.array([3.010816010789, 2.969972029686]),
                'phase_deg': np.array([8.4394, 4.7247]),
            }
        )

        obs_df = add_magnitude(obs_df, 12.79, 0.6)
        self.assertIn('APmag', obs_df.columns)
        assert_almost_equal(expected_mags, obs_df['APmag'], 3)

    def test_asteroid_default_G(self):
        expected_mags = [23.612, 24.047]

        # Values for 2025 ME74 from X05 calculated by JPL Horizons on 2025-08-25
        # using  soln ref.= JPL#16
        obs_df = pd.DataFrame(
            {
                'epoch_UTC': ['2025-05-01 00:00:00', '2025-05-11 00:00:00'],
                'Range_LTC_au': np.array([1.339060408365, 1.352758415830]),
                'Helio_LTC_au': np.array([0.395257884579, 0.446185902680]),
                'phase_deg': np.array([28.0414, 33.2737]),
            }
        )

        obs_df = add_magnitude(obs_df, 23.75)
        self.assertIn('APmag', obs_df.columns)
        assert_almost_equal(expected_mags, obs_df['APmag'], 3)

    def test_comet(self):
        expected_mags = [21.136, 21.089]

        # Values for C/2023 RS61 from F65 calculated by JPL Horizons on 2025-08-26
        # using  soln ref.= JPL#14
        obs_df = pd.DataFrame(
            {
                'epoch_UTC': ['2025-08-26 00:00:00', '2025-09-05 00:00:00'],
                'Helio_LTC_au': np.array([8.953438259855, 8.939216740294]),
                'Range_LTC_au': np.array([8.79141206043956, 8.62613767932531]),
                'phase_deg': np.array([6.4482, 6.2598]),
            }
        )

        obs_df = add_magnitude(obs_df, 8.8, 8.0, comet=True)
        self.assertIn('APmag', obs_df.columns)
        assert_almost_equal(expected_mags, obs_df['APmag'], 3)


class TestAddSkyMotion(SimpleTestCase):
    def setUp(self) -> None:
        # Define various rate units to simplify things
        self.jpl = u.arcsec / u.hour
        self.sorcha = u.deg / u.day
        self.fomo = u.arcsec / u.min

        return super().setUp()

    def test_default_units(self):
        expected_rate = [1.2440149, 1.3341748]
        expected_pa = [110.94500, 111.53031]

        obs_df = pd.DataFrame(
            {
                'RARateCosDec_deg_day': (([69.70892, 74.46485] * self.jpl).to(self.sorcha)),
                'DecRate_deg_day': (([-26.6820, -29.3780] * self.jpl).to(self.sorcha)),
            }
        )

        obs_df = add_sky_motion(obs_df)

        self.assertIn('sky_motion', obs_df.columns)
        self.assertIn('sky_motion_PA_deg', obs_df.columns)
        assert_almost_equal(expected_rate, obs_df['sky_motion'], 6)
        assert_almost_equal(expected_pa, obs_df['sky_motion_PA_deg'], 6)

    def test_jpl_units(self):
        expected_rate = [1.2440149 * 60, 1.3341748 * 60]
        expected_pa = [110.94500, 111.53031]

        obs_df = pd.DataFrame(
            {
                'RARateCosDec_deg_day': (([69.70892, 74.46485] * self.jpl).to(self.sorcha)),
                'DecRate_deg_day': (([-26.6820, -29.3780] * self.jpl).to(self.sorcha)),
            }
        )

        obs_df = add_sky_motion(obs_df, self.jpl)

        self.assertIn('sky_motion', obs_df.columns)
        self.assertIn('sky_motion_PA_deg', obs_df.columns)
        assert_almost_equal(expected_rate, obs_df['sky_motion'], 5)
        assert_almost_equal(expected_pa, obs_df['sky_motion_PA_deg'], 6)


class TestBuildAPCOContext(TestCase):
    def setUp(self):
        self.test_observatory, created = Observatory.objects.get_or_create(
            obscode='K93',
            name='Sutherland-LCO Dome C',
            lat=-32.380667412,
            lon=+20.81011,
            altitude=1808.33,
        )
        # East +ve longitude and latitude (radians)
        self.elong = np.radians(self.test_observatory.lon)
        self.phi = np.radians(self.test_observatory.lat)
        self.t = Time(2460806.5, format='jd', scale='tdb')
        pointing_df = pd.DataFrame(
            {
                'FieldID': [848],
                'observationMidpointMJD_TAI': self.t.tai.mjd,
            }
        )
        self.test_pointing = pointing_df.iloc[0]
        # Values from JPL Horizons (Observer barycentric position (AU) & velocity (km/s), heliocentric position)
        self.jplh_opb = np.array([-0.6510291307158590, -0.7169649592007841, -0.3106161232372137])
        self.jplh_ovb = np.array([2.278186862894084e01, -1.772608691513609e01, -7.623840806832071e00])
        self.jplh_eph = np.array([-0.6462284900804688, -0.7120159592468899, -0.3086406400831245])
        # Values from erfa.epv00
        self.erfa_epb = np.array([-0.6510155687487159, -0.7169309490408324, -0.3105933888770286])
        # Difference is ~100km or  0.0000006684587122 au) mostly in X

        # Values from test_observatory above, passed through erfa.pvtob, rotated by BPN matrix from
        # erfa.c2ixys/erfa.xys00b
        self.obs_pos = np.array([-1.2840145962839264e-05, -3.3708560834745510e-05, -2.2675569329249562e-05])

        # Observer position barycentric and heliocentric from erfa.apco13
        self.erfa_opb = np.array([-0.6510284090924512, -0.7169646574464681, -0.3106160643791047])
        self.erfa_oph = np.array([-0.6399007072710181, -0.7050634136553813, -0.3056348598551406])

        # Polar motion values "rotated to local meridian" (not sure how this is derived)
        # values from IERS-B columns of https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt
        self.xpl = -2.9954710682224965e-07  # originally np.radians(0.095486/3600.0)
        self.ypl = np.radians(0.425156 / 3600.0)
        self.field_names = (
            'pmt',
            'eb',
            'eh',
            'em',
            'v',
            'bm1',
            # 'bpn',
            'along',
            'phi',  # not initialized/used in apco
            'xpl',
            'ypl',
            'sphi',
            'cphi',
            'diurab',
            'eral',
            'refa',
            'refb',
        )

        return super().setUp()

    def test1(self):
        obs_sun_dist, obs_sun_vec = erfa.pn(self.jplh_eph + self.obs_pos)
        v_vec = self.jplh_ovb / CR
        v = np.linalg.norm(v_vec)
        expected_context = np.array(
            (
                (self.t.tdb.jd - JD2000) / 365.25,  # pmt
                self.jplh_opb,
                obs_sun_vec,
                obs_sun_dist,  # eb, eh, em
                v_vec,
                1 - (v**2),  # v(el), Lorentz factor (bm1)
                erfa.pnm00b(self.t.tt.jd1, self.t.tt.jd2),  # bpn (Bias-Precession-Nutation matrix)
                self.elong,
                0.0,
                self.xpl,
                self.ypl,  # along, phi, xpl, ypl
                np.sin(self.phi),
                np.cos(self.phi),  # sphi, cphi (sin/cos(phi))
                0.0,
                erfa.era00(self.t.ut1.jd1, self.t.ut1.jd2) + self.elong - 2 * np.pi,
                0.0,
                -0.0,
            ),
            dtype=[
                ('pmt', '<f8'),
                ('eb', '<f8', (3,)),
                ('eh', '<f8', (3,)),
                ('em', '<f8'),
                ('v', '<f8', (3,)),
                ('bm1', '<f8'),
                ('bpn', '<f8', (3, 3)),
                ('along', '<f8'),
                ('phi', '<f8'),
                ('xpl', '<f8'),
                ('ypl', '<f8'),
                ('sphi', '<f8'),
                ('cphi', '<f8'),
                ('diurab', '<f8'),
                ('eral', '<f8'),
                ('refa', '<f8'),
                ('refb', '<f8'),
            ],
        )

        context = build_apco_context(self.test_pointing, self.test_observatory)
        np.set_printoptions(precision=16, floatmode='fixed')
        for field in self.field_names:
            precision = 7
            if field in ['eb', 'eh']:
                # Lower precision of Earth's barycentric and heliocentric position since the JPL Horizons-derived
                # values don't match the erfa/VSOP2000 derived ones. This got worse with DE440/441 in 2021 which
                # includes KBOs, which weren't in previous DE's, which shifts the Sol. Sys. barycenter by ~100km
                # (which is ~0.67e-6 or 0.0000006684587122 au)
                precision = 6
            assert_array_almost_equal(
                expected_context[field], context[field], decimal=precision, err_msg=f'Failure on field {field}'
            )
