"""Rubin Target-of-Opportunity (ToO) filter criteria for JPL Scout NEO candidates.

Implements the primary (Section 2.1) filters from the SSSC NEOs WG document
"Filter Criteria for near-Earth Object (NEO) Rubin ToO Triggers" (v0.2). Each
filter is evaluated against a :class:`tom_jpl.models.ScoutDetail` instance (or
any object exposing the same attributes, e.g. a future ``ScoutDetailHistory``
row), so the same helpers drive both the live "currently passing" view and the
per-year first-pass aggregation.

A candidate "passes" only if *every* Section 2.1 filter is satisfied (and, per
Section 2.3, it should be cancelled once any of them stops being true). The
Section 2.2 airmass/observability filter is intentionally not implemented here
as it requires a site- and time-specific ephemeris.

Note on units: ``ScoutDetail.arc`` is stored in *days* (the Scout API reports it
in hours), so the "arc > 1 hour" criterion is compared against ``1.0 / 24.0``.
"""

# Section 2.1 thresholds
NEO_SCORE_MIN = 98  # neoScore >= 98
GEOCENTRIC_SCORE_MAX = 2  # geocentricScore < 2
ABS_MAG_MAX = 99.0  # H < 99: any size of interest (diameter cut removed)
IMPACT_RATING_MIN = 3  # rating >= 3
RMS_MAX = 1.0  # rmsN < 1.0
NUM_OBS_MIN = 5  # nObs > 5 (strictly greater)
ARC_MIN_DAYS = 1.0 / 24.0  # arc > 1 hour, stored in days
VMAG_MIN_NORTH = 21.6  # Vmag > 21.6 when dec > 0
VMAG_MIN_SOUTH = 21.8  # Vmag > 21.8 when dec <= 0
UNC_P1_MIN_NORTH = 60.0  # uncP1 > 60 arcmin when dec > 0
UNC_P1_MIN_SOUTH = 180.0  # uncP1 > 180 arcmin when dec <= 0
RATE_MAX = 25.0  # rate < 25 arcsec/min


def _f_neo_score(sd, H):
    return sd.neo_score is not None and sd.neo_score >= NEO_SCORE_MIN


def _f_geocentric_score(sd, H):
    return sd.geocentric_score is not None and sd.geocentric_score < GEOCENTRIC_SCORE_MAX


def _f_abs_mag(sd, H):
    return H is not None and H < ABS_MAG_MAX


def _f_impact_rating(sd, H):
    return sd.impact_rating is not None and sd.impact_rating >= IMPACT_RATING_MIN


def _f_rms(sd, H):
    return sd.rms is not None and sd.rms < RMS_MAX


def _f_obs_arc(sd, H):
    return sd.num_obs is not None and sd.num_obs > NUM_OBS_MIN and sd.arc is not None and sd.arc > ARC_MIN_DAYS


def _f_vmag(sd, H):
    if sd.vmag is None or sd.dec is None:
        return False
    threshold = VMAG_MIN_NORTH if sd.dec > 0 else VMAG_MIN_SOUTH
    return sd.vmag > threshold


def _f_unc_p1(sd, H):
    if sd.uncertainty_p1 is None or sd.dec is None:
        return False
    threshold = UNC_P1_MIN_NORTH if sd.dec > 0 else UNC_P1_MIN_SOUTH
    return sd.uncertainty_p1 > threshold


def _f_rate(sd, H):
    return sd.rate is not None and sd.rate < RATE_MAX


# Ordered (key, human-readable label, predicate) for each Section 2.1 filter.
# The order matches the document; keys are reused by the per-filter pass counts.
RUBIN_TOO_FILTERS = [
    ('neo_score', 'NEO score >= 98', _f_neo_score),
    ('geocentric_score', 'Geocentric score < 2', _f_geocentric_score),
    ('abs_mag', 'H < 99', _f_abs_mag),
    ('impact_rating', 'Impact rating >= 3', _f_impact_rating),
    ('rms', 'Orbit-fit RMS < 1.0', _f_rms),
    ('obs_arc', 'nObs > 5 and arc > 1 hr', _f_obs_arc),
    ('vmag', 'V > 21.6 (N) / 21.8 (S)', _f_vmag),
    ('unc_p1', 'Unc. at +1d > 60 (N) / 180 (S) arcmin', _f_unc_p1),
    ('rate', 'Sky motion < 25 arcsec/min', _f_rate),
]


def _resolve_abs_mag(scout_detail, abs_mag):
    """Resolve the absolute magnitude H, falling back to the linked Target's abs_mag.

    ``ScoutDetail`` does not store H itself; it lives on ``Target.abs_mag``. Callers
    working from a history row (which has no contemporaneous H) may pass ``abs_mag``
    explicitly.
    """
    if abs_mag is not None:
        return abs_mag
    target = getattr(scout_detail, 'target', None)
    return getattr(target, 'abs_mag', None) if target is not None else None


def evaluate_filters(scout_detail, abs_mag=None):
    """Return an ordered ``dict`` of ``{filter_key: bool}`` for every Section 2.1 filter."""
    H = _resolve_abs_mag(scout_detail, abs_mag)
    return {key: func(scout_detail, H) for key, _label, func in RUBIN_TOO_FILTERS}


def passes_filters(scout_detail, abs_mag=None):
    """Return ``True`` iff the candidate passes *all* Section 2.1 Rubin ToO filters."""
    H = _resolve_abs_mag(scout_detail, abs_mag)
    return all(func(scout_detail, H) for _key, _label, func in RUBIN_TOO_FILTERS)
