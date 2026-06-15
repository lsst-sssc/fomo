"""Unit tests for the Rubin ToO Section 2.1 filter helpers (solsys_code.rubin_too).

These exercise the pure filter logic against lightweight stand-ins for
``tom_jpl.models.ScoutDetail`` (a ``SimpleNamespace`` with the same attributes,
plus a nested ``target`` carrying ``abs_mag``), so no database is needed.
"""

from types import SimpleNamespace

from django.test import SimpleTestCase

from solsys_code.rubin_too import evaluate_filters, passes_filters


def make_detail(**overrides):
    """A ScoutDetail-like object that passes every Section 2.1 filter (Northern, dec > 0).

    Pass ``abs_mag=...`` to override the linked target's H; any ScoutDetail field
    can be overridden by keyword.
    """
    abs_mag = overrides.pop('abs_mag', 25.0)
    fields = dict(
        neo_score=99,
        geocentric_score=1,
        impact_rating=3,
        rms=0.5,
        num_obs=6,
        arc=2.0 / 24.0,  # 2 hours, stored in days
        vmag=21.7,  # fainter than the 21.6 Northern cut
        uncertainty_p1=70.0,  # larger than the 60 arcmin Northern cut
        rate=10.0,
        dec=10.0,  # Northern
        ra=133.5,
    )
    fields.update(overrides)
    fields['target'] = SimpleNamespace(abs_mag=abs_mag)
    return SimpleNamespace(**fields)


class RubinTooFiltersTest(SimpleTestCase):
    def test_baseline_passes(self):
        """The baseline Northern object passes all filters."""
        self.assertIs(passes_filters(make_detail()), True)

    def test_baseline_southern_passes(self):
        """A Southern object passes with fainter Vmag and larger uncertainty."""
        self.assertIs(passes_filters(make_detail(dec=-10.0, vmag=21.9, uncertainty_p1=181.0)), True)

    def test_neo_score(self):
        """neoScore must be >= 98 (None fails)."""
        for value, expected in [(98, True), (97, False), (None, False)]:
            with self.subTest(value=value):
                self.assertIs(passes_filters(make_detail(neo_score=value)), expected)

    def test_geocentric_score(self):
        """geocentricScore must be < 2 (None fails)."""
        for value, expected in [(1, True), (2, False), (None, False)]:
            with self.subTest(value=value):
                self.assertIs(passes_filters(make_detail(geocentric_score=value)), expected)

    def test_abs_mag(self):
        """H must be < 99 (None/undefined fails)."""
        for abs_mag, expected in [(98.0, True), (99.0, False), (None, False)]:
            with self.subTest(abs_mag=abs_mag):
                self.assertIs(passes_filters(make_detail(abs_mag=abs_mag)), expected)

    def test_abs_mag_explicit_argument_overrides_target(self):
        """An explicit abs_mag argument takes precedence over the linked target's H."""
        # Target H would fail, but an explicit in-range H passes (history-row use case).
        self.assertIs(passes_filters(make_detail(abs_mag=200.0), abs_mag=30.0), True)

    def test_impact_rating(self):
        """Impact rating must be >= 3 (None fails)."""
        for value, expected in [(3, True), (4, True), (2, False), (None, False)]:
            with self.subTest(value=value):
                self.assertIs(passes_filters(make_detail(impact_rating=value)), expected)

    def test_rms(self):
        """Orbit-fit RMS must be < 1.0 (None fails)."""
        for value, expected in [(0.99, True), (1.0, False), (None, False)]:
            with self.subTest(value=value):
                self.assertIs(passes_filters(make_detail(rms=value)), expected)

    def test_num_obs(self):
        """Number of observations must be > 5 (None fails)."""
        for num_obs, expected in [(6, True), (5, False), (None, False)]:
            with self.subTest(num_obs=num_obs):
                self.assertIs(passes_filters(make_detail(num_obs=num_obs)), expected)

    def test_arc(self):
        """Arc must exceed 1 hour; ScoutDetail.arc is in days so we convert here."""
        for arc_hours, expected in [(1.5, True), (1.0, False), (0.5, False)]:
            with self.subTest(arc_hours=arc_hours):
                self.assertIs(passes_filters(make_detail(arc=arc_hours / 24.0)), expected)

    def test_arc_none_fails(self):
        """A missing arc fails the obs/arc filter."""
        self.assertIs(passes_filters(make_detail(arc=None)), False)

    def test_vmag_north_south(self):
        """Vmag threshold branches on declination (21.6 N / 21.8 S, dec == 0 is South)."""
        cases = [
            (10.0, 21.7, True),  # North: > 21.6
            (10.0, 21.6, False),  # North: boundary, strict
            (-10.0, 21.7, False),  # South: needs > 21.8
            (-10.0, 21.9, True),  # South: > 21.8
            (0.0, 21.7, False),  # dec == 0 falls in the Southern branch
            (0.0, 21.9, True),
        ]
        for dec, vmag, expected in cases:
            with self.subTest(dec=dec, vmag=vmag):
                self.assertIs(passes_filters(make_detail(dec=dec, vmag=vmag, uncertainty_p1=200.0)), expected)

    def test_unc_p1_north_south(self):
        """uncP1 threshold branches on declination (60 N / 180 S, dec == 0 is South)."""
        cases = [
            (10.0, 61.0, True),  # North: > 60
            (10.0, 60.0, False),  # North: boundary, strict
            (-10.0, 61.0, False),  # South: needs > 180
            (-10.0, 181.0, True),  # South: > 180
            (0.0, 61.0, False),  # dec == 0 falls in the Southern branch
        ]
        # Use a Vmag that satisfies both hemispheres so only uncP1 is under test.
        for dec, unc_p1, expected in cases:
            with self.subTest(dec=dec, unc_p1=unc_p1):
                self.assertIs(passes_filters(make_detail(dec=dec, vmag=21.9, uncertainty_p1=unc_p1)), expected)

    def test_rate(self):
        """Sky-motion rate must be < 25 arcsec/min (None fails)."""
        for value, expected in [(24.9, True), (25.0, False), (None, False)]:
            with self.subTest(value=value):
                self.assertIs(passes_filters(make_detail(rate=value)), expected)

    def test_evaluate_filters_keys_and_values(self):
        """evaluate_filters returns a per-filter pass/fail map with all nine keys."""
        result = evaluate_filters(make_detail(rate=30.0))  # only the rate filter fails
        expected_keys = {
            'neo_score',
            'geocentric_score',
            'abs_mag',
            'impact_rating',
            'rms',
            'obs_arc',
            'vmag',
            'unc_p1',
            'rate',
        }
        self.assertEqual(set(result), expected_keys)
        self.assertIs(result['rate'], False)
        self.assertTrue(all(v for k, v in result.items() if k != 'rate'))
