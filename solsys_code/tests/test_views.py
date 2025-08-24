from collections import namedtuple

import numpy as np
from django.test import TestCase, tag
from numpy.testing import assert_almost_equal
from tom_targets.models import Target

# Import module to test
from solsys_code.views import convert_target_to_layup

MJD_TO_JD_CONVERSION = 2400000.5


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
