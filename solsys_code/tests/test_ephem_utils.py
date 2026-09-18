from collections import namedtuple
from datetime import datetime
from pathlib import Path

import erfa
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from django.test import SimpleTestCase, TestCase, tag
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from tom_targets.models import Target

# Import module to test
from solsys_code.ephem_utils import (
    add_magnitude,
    add_sky_motion,
    build_apco_context,
    compute_ephemeris,
    convert_target_to_layup,
    get_nonsidereal_visibility,
)
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.visibility import visibility_windows

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
        for name, j in zip(converted.dtype.names[2:8], range(6), strict=False):
            assert_almost_equal(converted[name], self.bary_vec[j], 8)

    @tag('spiceypy')
    def test_sun_ephemeris(self):
        """test conversion with in-place Sun position determination
        This tests needs SPICE kernels available (TBD but at least:
        * naif0012.tls (leap second kernel)
        * de440s.bsp (Short version of DE440 ephemeris)
        )"""

        converted = convert_target_to_layup(self.target)
        for name, j in zip(converted.dtype.names[2:8], range(6), strict=False):
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
        # Form bpn (Bias-Precession-Nutation matrix) from CIP (X, Y) and CIO locator
        # This is CIO-based so the terms in bpn[0][1] and bpn[1][0] are several orders of magnitude bigger
        # than those in classical equinox-based BPN matrix. So can't just call erfa.pnm00b.
        X, Y, s = erfa.xys06a(self.t.tt.jd1, self.t.tt.jd2)
        self.bpn = erfa.c2ixys(X, Y, s)
        self.field_names = (
            'pmt',
            'eb',
            'eh',
            'em',
            'v',
            'bm1',
            'bpn',
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
                (self.t.tdb.jd - JD2000) / 365.25,  # pmt (for proper motions)
                self.jplh_opb,  # eb, Observer-SSB position vector
                obs_sun_vec,  # eh, Observer-Sun position vector
                obs_sun_dist,  # em, Observer-Sun distance
                v_vec,  # v(elocity)
                1 - (v**2),  # bm1, Lorentz factor
                self.bpn,  # bpn (Bias-Precession-Nutation matrix)
                self.elong,
                0.0,
                self.xpl,  # xpl, ypl, polar motion
                self.ypl,
                np.sin(self.phi),  # sphi, cphi (sin/cos(phi))
                np.cos(self.phi),
                0.0,
                erfa.era00(self.t.ut1.jd1, self.t.ut1.jd2) + self.elong - 2 * np.pi,
                0.0,  # refa, refb, refraction constants
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


class TestGetNonsiderealVisibility(TestCase):
    def setUp(self):
        self.cpt, created = Observatory.objects.get_or_create(
            obscode='K93',
            name='Sutherland-LCO Dome C',
            lat=-32.380667412,
            lon=+20.81011,
            altitude=1808.33,
        )
        self.target, created = Target.objects.get_or_create(
            name='33933',
            type='NON_SIDEREAL',
            permissions='PUBLIC',
            scheme='MPC_MINOR_PLANET',
            epoch_of_elements=61000.0,
            mean_anomaly=342.8987983972185,
            arg_of_perihelion=197.2440098291647,
            eccentricity=0.21317079351206,
            lng_asc_node=55.4085914553028,
            inclination=1.0791909799414,
            semimajor_axis=2.186745866749343,
            epoch_of_perihelion=59874.98228566302,
            perihdist=1.72059551512517,
            abs_mag=14.89,
            slope=0.15,
        )
        self.start = datetime(2025, 5, 10)
        self.end = datetime(2025, 5, 11)

    def test_output_shape_matches_sidereal_visibility(self):
        visibility = get_nonsidereal_visibility(self.target, {'CPT': self.cpt}, self.start, self.end, 60)

        self.assertEqual(list(visibility.keys()), ['CPT'])
        times, airmasses = visibility['CPT']
        self.assertEqual(len(times), 25)
        self.assertEqual(len(airmasses), 25)
        self.assertEqual(times[0], self.start)
        self.assertEqual(times[-1], self.end)

    def test_target_only_visible_in_evening_from_cpt(self):
        # (33933) sets a few hours after the Sun in May 2025; from K93 it is only above the horizon
        # after astronomical twilight between 18:00 and 20:00 UTC on 2025-05-10.
        times, airmasses = get_nonsidereal_visibility(self.target, {'CPT': self.cpt}, self.start, self.end, 60)['CPT']

        valid = {time.hour: airmass for time, airmass in zip(times, airmasses, strict=True) if airmass is not None}
        self.assertEqual(sorted(valid), [18, 19, 20])
        self.assertAlmostEqual(valid[18], 2.02, places=2)
        self.assertAlmostEqual(valid[19], 2.75, places=2)
        self.assertAlmostEqual(valid[20], 5.11, places=2)

    def test_airmass_limit(self):
        times, airmasses = get_nonsidereal_visibility(
            self.target, {'CPT': self.cpt}, self.start, self.end, 60, airmass_limit=3.0
        )['CPT']

        valid_hours = [time.hour for time, airmass in zip(times, airmasses, strict=True) if airmass is not None]
        self.assertEqual(valid_hours, [18, 19])

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            get_nonsidereal_visibility(self.target, {'CPT': self.cpt}, self.end, self.start, 60)


class TestCloseApproach2025FA22(TestCase):
    """
    2025 FA22 passed 0.0056 au from the Earth on 2025-09-18 07:43 TDB, arriving from the daytime sky
    (31 deg from the Sun two days before) and moving at up to 2.6 arcsec/s at closest approach. The
    reference is a JPL Horizons geocentric ephemeris (``data/2025FA22_horizons_geocentric.csv``) from the
    same orbit solution as the elements below.
    """

    def setUp(self):
        # Horizons osculating heliocentric ecliptic elements at 2025-09-01 00:00 TDB (solution of 2026-05-08)
        self.target, created = Target.objects.get_or_create(
            name='2025 FA22',
            type='NON_SIDEREAL',
            permissions='PUBLIC',
            scheme='MPC_MINOR_PLANET',
            epoch_of_elements=60919.0,
            perihdist=0.8818643632074030,
            eccentricity=0.4150181985939446,
            inclination=7.547632751977779,
            lng_asc_node=356.5000374812164,
            arg_of_perihelion=304.9125910339715,
            epoch_of_perihelion=60894.453130597249,
            semimajor_axis=1.507507346532430,
            mean_anomaly=13.07106667639207,
            abs_mag=21.59,
            slope=0.15,
        )
        self.geocentre, created = Observatory.objects.get_or_create(
            obscode='500', name='Geocentric', lat=0.0, lon=0.0, altitude=0.0
        )
        self.horizons = pd.read_csv(Path(__file__).parent / 'data' / '2025FA22_horizons_geocentric.csv', comment='#')

    def test_geocentric_ephemeris_matches_horizons_through_close_approach(self):
        times = Time([datetime.strptime(date, '%Y-%b-%d %H:%M:%S.%f') for date in self.horizons['date_tt']], scale='tt')

        predictions = compute_ephemeris(self.target, self.geocentre, times)

        predicted = SkyCoord(predictions['RA_deg'], predictions['Dec_deg'], unit='deg')
        expected = SkyCoord(self.horizons['ra_deg'], self.horizons['dec_deg'], unit='deg')
        self.assertLess(predicted.separation(expected).arcsec.max(), 0.05)
        assert_array_almost_equal(predictions['Range_LTC_au'], self.horizons['delta_au'], decimal=8)
        assert_array_almost_equal(predictions['phase_deg'], self.horizons['phase_deg'], decimal=1)
        # Closest approach is the 2025-09-18 07:42:59 row
        self.assertEqual(predictions['Range_LTC_au'].idxmin(), 9)

    def test_visibility_switches_on_after_close_approach(self):
        lsc, created = Observatory.objects.get_or_create(
            obscode='W85', name='Cerro Tololo-LCO', lat=-30.167, lon=-70.805, altitude=2198.0
        )
        ogg, created = Observatory.objects.get_or_create(
            obscode='T04', name='Haleakala-LCO', lat=20.707, lon=-156.258, altitude=3055.0
        )

        visibility = get_nonsidereal_visibility(
            self.target, {'LSC': lsc, 'OGG': ogg}, datetime(2025, 9, 16), datetime(2025, 9, 20), 15, airmass_limit=2.5
        )

        # Two days before closest approach the target is 31 deg from the Sun: nothing from either hemisphere
        for site, (times, airmasses) in visibility.items():
            self.assertTrue(
                all(airmass is None for time, airmass in zip(times, airmasses, strict=True) if time.day == 16), site
            )
        # The night after closest approach it is well placed from both hemispheres
        windows = visibility_windows(visibility)
        self.assertEqual(windows['LSC'][-1], (datetime(2025, 9, 19, 5, 30), datetime(2025, 9, 19, 9, 15)))
        self.assertEqual(windows['OGG'][-1], (datetime(2025, 9, 19, 9, 0), datetime(2025, 9, 19, 14, 45)))
