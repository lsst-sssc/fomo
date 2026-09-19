from datetime import datetime, timedelta, timezone
from math import nan

import numpy as np
from astropy import units as u
from django.test import SimpleTestCase

from solsys_code.visibility import CadenceWindow, airmass_samples, cadence_window, visibility_windows

DAY = timedelta(days=1)


def t(day, hhmm):
    """2026-09-<day> at HH:MM UTC."""
    hour, minute = divmod(hhmm, 100)
    return datetime(2026, 9, day, hour, minute)


# Worked example from docs/plans/visibility_windows.md, night of 2026-09-16 (UTC):
# COJ 09:00->17:15, CPT 17:30->02:00(+1), LSC 23:45->08:00(+1)
COJ = (t(16, 900), t(16, 1715))
CPT = (t(16, 1730), t(17, 200))
LSC = (t(16, 2345), t(17, 800))


class TestAirmassSamples(SimpleTestCase):
    def assertAirmasses(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected, strict=True):
            if want is None:
                self.assertIsNone(got)
            else:
                self.assertIsInstance(got, float)
                self.assertAlmostEqual(got, want, places=3)

    def test_airmass_is_secant_of_zenith_distance(self):
        self.assertAirmasses(airmass_samples([90.0, 30.0], [-30.0, -30.0]), [1.0, 2.0])

    def test_below_horizon_is_none(self):
        self.assertAirmasses(airmass_samples([0.0, -5.0], [-30.0, -30.0]), [None, None])

    def test_undefined_altitude_is_none(self):
        self.assertAirmasses(airmass_samples([nan, 30.0], [-30.0, -30.0]), [None, 2.0])

    def test_sun_up_is_none(self):
        # Sun exactly at the limit is still night; anything above it is rejected
        self.assertAirmasses(airmass_samples([30.0, 30.0, 30.0], [-18.0, -17.9, 10.0]), [2.0, None, None])

    def test_custom_sun_altitude_limit(self):
        self.assertAirmasses(airmass_samples([30.0, 30.0], [-15.0, -10.0], sun_alt_limit_deg=-12.0), [2.0, None])

    def test_default_airmass_limit_is_10(self):
        # airmass 10 corresponds to an altitude of ~5.74 deg
        self.assertAirmasses(airmass_samples([5.7, 5.8], [-30.0, -30.0]), [None, 9.895])

    def test_airmass_limit(self):
        self.assertAirmasses(airmass_samples([30.0, 31.0], [-30.0, -30.0], airmass_limit=2.0), [None, 1.942])

    def test_numpy_inputs(self):
        self.assertAirmasses(airmass_samples(np.array([90.0, -1.0]), np.array([-30.0, -30.0])), [1.0, None])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            airmass_samples([30.0], [-30.0, -30.0])


class TestVisibilityWindows(SimpleTestCase):
    def setUp(self):
        self.times = [t(16, 0) + i * timedelta(hours=1) for i in range(6)]

    def test_empty_samples_gives_no_windows(self):
        self.assertEqual(visibility_windows({}), {})

    def test_runs_of_valid_samples(self):
        samples = {'LSC': (self.times, [None, 1.5, 1.2, None, 2.0, 1.9])}
        self.assertEqual(
            visibility_windows(samples),
            {'LSC': [(t(16, 100), t(16, 200)), (t(16, 400), t(16, 500))]},
        )

    def test_nan_is_invalid(self):
        samples = {'CPT': (self.times, [1.1, nan, 1.3, 1.4, nan, nan])}
        self.assertEqual(visibility_windows(samples), {'CPT': [(t(16, 0), t(16, 0)), (t(16, 200), t(16, 300))]})

    def test_all_invalid_gives_no_windows(self):
        self.assertEqual(visibility_windows({'COJ': (self.times, [None] * 6)}), {'COJ': []})

    def test_multiple_sites_keep_order(self):
        samples = {
            'LSC': (self.times, [1.0] * 6),
            'COJ': (self.times, [None, None, 1.0, 1.0, None, None]),
        }
        self.assertEqual(list(visibility_windows(samples)), ['LSC', 'COJ'])
        self.assertEqual(visibility_windows(samples)['COJ'], [(t(16, 200), t(16, 300))])
        self.assertEqual(visibility_windows(samples)['LSC'], [(t(16, 0), t(16, 500))])

    def test_numpy_inputs(self):
        # Shape of tom_observations.utils.get_sidereal_visibility: object array of datetimes, float array
        times = np.array(self.times, dtype=object)
        airmasses = np.array([np.nan, 1.5, np.float64(1.2), np.nan, 2.0, 1.9])

        self.assertEqual(
            visibility_windows({'LSC': (times, airmasses)}),
            {'LSC': [(t(16, 100), t(16, 200)), (t(16, 400), t(16, 500))]},
        )


class TestCadenceWindow(SimpleTestCase):
    def assertWindow(self, window, start, end, midpoint, duration, coverage=None, gaps=None):
        self.assertIsInstance(window, CadenceWindow)
        self.assertEqual(window.start, start)
        self.assertEqual(window.end, end)
        self.assertEqual(window.midpoint, midpoint)
        self.assertEqual(window.duration, duration)
        if coverage is not None:
            self.assertEqual(window.coverage, coverage)
        if gaps is not None:
            self.assertEqual(window.gaps, gaps)

    def test_worked_example(self):
        window = cadence_window([COJ, CPT, LSC])

        self.assertWindow(window, t(16, 900), t(17, 800), t(16, 2030), timedelta(hours=23))
        self.assertEqual(window.coverage, [(t(16, 900), t(16, 1715)), (t(16, 1730), t(17, 800))])
        self.assertEqual(window.gaps, [(t(16, 1715), t(16, 1730))])

    def test_order_independent(self):
        self.assertEqual(cadence_window([LSC, COJ, CPT]), cadence_window([COJ, CPT, LSC]))

    def test_two_day_horizon_folds_to_same_window(self):
        two_nights = [COJ, CPT, LSC] + [(s + DAY, e + DAY) for s, e in [COJ, CPT, LSC]]

        self.assertEqual(cadence_window(two_nights), cadence_window([COJ, CPT, LSC]))

    def test_window_across_midnight_from_later_anchor(self):
        # Same nightly pattern but the first interval we know about is CPT's, so t0 = 17:30
        window = cadence_window([CPT, LSC, (COJ[0] + DAY, COJ[1] + DAY)])

        self.assertWindow(
            window,
            t(17, 900),
            t(18, 800),
            t(17, 2030),
            timedelta(hours=23),
            coverage=[(t(17, 900), t(17, 1715)), (t(17, 1730), t(18, 800))],
            gaps=[(t(17, 1715), t(17, 1730))],
        )

    def test_single_site(self):
        window = cadence_window([LSC])

        midpoint = datetime(2026, 9, 17, 3, 52, 30)
        self.assertWindow(window, t(16, 2345), t(17, 800), midpoint, timedelta(hours=8, minutes=15))
        self.assertEqual(window.coverage, [LSC])
        self.assertEqual(window.gaps, [])

    def test_largest_gap_not_at_wrap(self):
        # 00:00-02:00 and 10:00-20:00: gaps are 8 h (02-10) and 4 h (20-24); window is 10:00->02:00
        window = cadence_window([(t(16, 0), t(16, 200)), (t(16, 1000), t(16, 2000))])

        self.assertWindow(window, t(16, 1000), t(17, 200), t(16, 1800), timedelta(hours=16))
        self.assertEqual(window.coverage, [(t(16, 1000), t(16, 2000)), (t(17, 0), t(17, 200))])
        self.assertEqual(window.gaps, [(t(16, 2000), t(17, 0))])

    def test_full_coverage(self):
        window = cadence_window([(t(16, 0), t(16, 1300)), (t(16, 1200), t(17, 100))])

        self.assertWindow(window, t(16, 0), t(17, 0), t(16, 1200), DAY)
        self.assertEqual(window.coverage, [(t(16, 0), t(17, 0))])
        self.assertEqual(window.gaps, [])

    def test_interval_longer_than_period_is_full_coverage(self):
        window = cadence_window([(t(16, 300), t(17, 400))])

        self.assertWindow(
            window, t(16, 300), t(17, 300), t(16, 1500), DAY, coverage=[(t(16, 300), t(17, 300))], gaps=[]
        )

    def test_touching_and_contained_intervals_merge(self):
        # 10-12 touches 12-14; 13-13:30 sits inside; one 4 h block, no zero-width gaps
        window = cadence_window([(t(16, 1000), t(16, 1200)), (t(16, 1200), t(16, 1400)), (t(16, 1300), t(16, 1330))])

        self.assertWindow(
            window,
            t(16, 1000),
            t(16, 1400),
            t(16, 1200),
            timedelta(hours=4),
            coverage=[(t(16, 1000), t(16, 1400))],
            gaps=[],
        )

    def test_equal_gaps_break_at_first_after_anchor(self):
        # 00-06 and 12-18 leave two 6 h gaps; the break is the earliest one after t0 (06-12), so the
        # window is 12:00 -> 06:00(+1) with the other gap (18-00) reported as a sub-gap
        window = cadence_window([(t(16, 0), t(16, 600)), (t(16, 1200), t(16, 1800))])

        self.assertWindow(
            window,
            t(16, 1200),
            t(17, 600),
            t(16, 2100),
            timedelta(hours=18),
            coverage=[(t(16, 1200), t(16, 1800)), (t(17, 0), t(17, 600))],
            gaps=[(t(16, 1800), t(17, 0))],
        )

    def test_timezone_aware_datetimes(self):
        aware = [(s.replace(tzinfo=timezone.utc), e.replace(tzinfo=timezone.utc)) for s, e in [COJ, CPT, LSC]]

        window = cadence_window(aware)

        self.assertEqual(window.midpoint, t(16, 2030).replace(tzinfo=timezone.utc))
        self.assertEqual(window.duration, timedelta(hours=23))
        self.assertIsNotNone(window.start.tzinfo)

    def test_non_daily_period(self):
        # Every 12 h: 01:00-04:00 and 13:00-16:00 fold to the same 3 h; window 01:00->04:00, midpoint 02:30
        window = cadence_window([(t(16, 100), t(16, 400)), (t(16, 1300), t(16, 1600))], period=12 * u.h)

        self.assertWindow(
            window, t(16, 100), t(16, 400), t(16, 230), timedelta(hours=3), coverage=[(t(16, 100), t(16, 400))], gaps=[]
        )

    def test_period_as_timedelta(self):
        self.assertEqual(cadence_window([COJ, CPT, LSC], period=DAY), cadence_window([COJ, CPT, LSC]))

    def test_empty_and_reversed_intervals_ignored(self):
        window = cadence_window([COJ, CPT, LSC, (t(16, 500), t(16, 500)), (t(16, 600), t(16, 500))])

        self.assertEqual(window, cadence_window([COJ, CPT, LSC]))

    def test_no_intervals_raises(self):
        with self.assertRaises(ValueError):
            cadence_window([])
        with self.assertRaises(ValueError):
            cadence_window([(t(16, 600), t(16, 500))])
        with self.assertRaises(ValueError):
            cadence_window([(t(16, 500), t(16, 500))])
