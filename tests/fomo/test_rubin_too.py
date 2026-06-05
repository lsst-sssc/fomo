"""Unit tests for the Rubin ToO Section 2.1 filter helpers (solsys_code.rubin_too).

These exercise the pure filter logic against lightweight stand-ins for
``tom_jpl.models.ScoutDetail`` (a ``SimpleNamespace`` with the same attributes,
plus a nested ``target`` carrying ``abs_mag``), so no database or Django setup
is needed.
"""

from types import SimpleNamespace

import pytest

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


def test_baseline_passes():
    """The baseline Northern object passes all filters."""
    assert passes_filters(make_detail()) is True


def test_baseline_southern_passes():
    """A Southern object passes with fainter Vmag and larger uncertainty."""
    assert passes_filters(make_detail(dec=-10.0, vmag=21.9, uncertainty_p1=181.0)) is True


@pytest.mark.parametrize('value, expected', [(98, True), (97, False), (None, False)])
def test_neo_score(value, expected):
    """neoScore must be >= 98 (None fails)."""
    assert passes_filters(make_detail(neo_score=value)) is expected


@pytest.mark.parametrize('value, expected', [(1, True), (2, False), (None, False)])
def test_geocentric_score(value, expected):
    """geocentricScore must be < 2 (None fails)."""
    assert passes_filters(make_detail(geocentric_score=value)) is expected


@pytest.mark.parametrize('abs_mag, expected', [(98.0, True), (99.0, False), (None, False)])
def test_abs_mag(abs_mag, expected):
    """H must be < 99 (None/undefined fails)."""
    assert passes_filters(make_detail(abs_mag=abs_mag)) is expected


def test_abs_mag_explicit_argument_overrides_target():
    """An explicit abs_mag argument takes precedence over the linked target's H."""
    # Target H would fail, but an explicit in-range H passes (history-row use case).
    assert passes_filters(make_detail(abs_mag=200.0), abs_mag=30.0) is True


@pytest.mark.parametrize('value, expected', [(3, True), (4, True), (2, False), (None, False)])
def test_impact_rating(value, expected):
    """Impact rating must be >= 3 (None fails)."""
    assert passes_filters(make_detail(impact_rating=value)) is expected


@pytest.mark.parametrize('value, expected', [(0.99, True), (1.0, False), (None, False)])
def test_rms(value, expected):
    """Orbit-fit RMS must be < 1.0 (None fails)."""
    assert passes_filters(make_detail(rms=value)) is expected


@pytest.mark.parametrize('num_obs, expected', [(6, True), (5, False), (None, False)])
def test_num_obs(num_obs, expected):
    """Number of observations must be > 5 (None fails)."""
    assert passes_filters(make_detail(num_obs=num_obs)) is expected


@pytest.mark.parametrize('arc_hours, expected', [(1.5, True), (1.0, False), (0.5, False)])
def test_arc(arc_hours, expected):
    """Arc must exceed 1 hour; ScoutDetail.arc is in days so we convert here."""
    assert passes_filters(make_detail(arc=arc_hours / 24.0)) is expected


def test_arc_none_fails():
    """A missing arc fails the obs/arc filter."""
    assert passes_filters(make_detail(arc=None)) is False


@pytest.mark.parametrize(
    'dec, vmag, expected',
    [
        (10.0, 21.7, True),  # North: > 21.6
        (10.0, 21.6, False),  # North: boundary, strict
        (-10.0, 21.7, False),  # South: needs > 21.8
        (-10.0, 21.9, True),  # South: > 21.8
        (0.0, 21.7, False),  # dec == 0 falls in the Southern branch
        (0.0, 21.9, True),
    ],
)
def test_vmag_north_south(dec, vmag, expected):
    """Vmag threshold branches on declination (21.6 N / 21.8 S, dec == 0 is South)."""
    assert passes_filters(make_detail(dec=dec, vmag=vmag, uncertainty_p1=200.0)) is expected


@pytest.mark.parametrize(
    'dec, unc_p1, expected',
    [
        (10.0, 61.0, True),  # North: > 60
        (10.0, 60.0, False),  # North: boundary, strict
        (-10.0, 61.0, False),  # South: needs > 180
        (-10.0, 181.0, True),  # South: > 180
        (0.0, 61.0, False),  # dec == 0 falls in the Southern branch
    ],
)
def test_unc_p1_north_south(dec, unc_p1, expected):
    """uncP1 threshold branches on declination (60 N / 180 S, dec == 0 is South)."""
    # Use a Vmag that satisfies both hemispheres so only uncP1 is under test.
    assert passes_filters(make_detail(dec=dec, vmag=21.9, uncertainty_p1=unc_p1)) is expected


@pytest.mark.parametrize('value, expected', [(24.9, True), (25.0, False), (None, False)])
def test_rate(value, expected):
    """Sky-motion rate must be < 25 arcsec/min (None fails)."""
    assert passes_filters(make_detail(rate=value)) is expected


def test_evaluate_filters_keys_and_values():
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
    assert set(result) == expected_keys
    assert result['rate'] is False
    assert all(v for k, v in result.items() if k != 'rate')
