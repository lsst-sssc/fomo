"""
Pure helpers for turning per-site visibility samples into windows and a cadence midpoint.

This module deliberately does not import ``solsys_code.ephem_utils`` (which furnishes SPICE
kernels at import time) so it stays cheap to import and test.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import NamedTuple

from astropy import units as u

Interval = tuple[datetime, datetime]

# LCO site code -> (site name, MPC code of one dome at that site); inter-dome differences are negligible
LCO_SITES = {
    'lsc': ('Cerro Tololo', 'W85'),
    'cpt': ('Sutherland', 'K91'),
    'coj': ('Siding Spring', 'Q63'),
    'tfn': ('Tenerife', 'Z31'),
    'elp': ('McDonald', 'V38'),
    'ogg': ('Haleakala', 'T04'),
}


class CadenceWindow(NamedTuple):
    """
    The observing window to use for a periodic cadence, folded onto one period.

    ``start``/``end`` bracket the longest stretch of the period during which at least one site can
    observe; ``midpoint`` is its centre and ``duration`` its length (``end - start``). These map onto
    an LCO cadence request as ``period``, ``jitter = duration`` and a first window anchored at
    ``midpoint``. ``coverage`` lists the merged intervals inside the window and ``gaps`` the holes
    between them (the largest gap of the period is by construction outside the window).
    """

    start: datetime
    end: datetime
    midpoint: datetime
    duration: timedelta
    coverage: list[Interval]
    gaps: list[Interval]


def _is_valid(airmass) -> bool:
    if airmass is None:
        return False
    try:
        return not math.isnan(airmass)
    except TypeError:
        return True


def airmass_samples(altitudes_deg, sun_altitudes_deg, airmass_limit=None, sun_alt_limit_deg=-18.0) -> list:
    """
    Converts sampled target and Sun altitudes into the ``[airmass | None]`` list used by ``visibility_windows``.

    Mirrors the masking in ``tom_observations.utils.get_sidereal_visibility``: a sample is ``None`` when the
    target is below the horizon, its plane-parallel (sec z) airmass is at or above ``airmass_limit`` (default 10)
    or the Sun is above ``sun_alt_limit_deg`` (default: astronomical twilight).

    :param altitudes_deg: Target altitude at each sample (degrees)
    :param sun_altitudes_deg: Sun altitude at each sample (degrees)
    :param airmass_limit: Maximum acceptable airmass; ``None`` means 10
    :param sun_alt_limit_deg: Samples with the Sun above this altitude are rejected
    :return: Airmass per sample, or ``None`` where the sample is not observable
    :rtype: list
    """
    if airmass_limit is None:
        airmass_limit = 10
    airmasses = []
    for alt, sun_alt in zip(altitudes_deg, sun_altitudes_deg, strict=True):
        # `not alt > 0` rather than `alt <= 0` so that a NaN altitude (e.g. the geocentre) is also rejected
        if not alt > 0 or sun_alt > sun_alt_limit_deg:
            airmasses.append(None)
            continue
        airmass = 1.0 / math.sin(math.radians(alt))
        airmasses.append(None if airmass >= airmass_limit else float(airmass))
    return airmasses


def observability_margins(altitudes_deg, sun_altitudes_deg, airmass_limit=None, sun_alt_limit_deg=-18.0) -> list:
    """
    How far, in degrees, each sample is inside (positive) or outside (negative) its tightest limit.

    The limits are those of ``airmass_samples``: the target above the altitude at which its plane-parallel
    airmass equals ``airmass_limit``, and the Sun below ``sun_alt_limit_deg``. The margin is the smaller of the
    two clearances, so it passes through zero where the sample stops being observable, whichever limit causes
    it. ``visibility_windows`` interpolates that zero crossing to place the window edges.

    :param altitudes_deg: Target altitude at each sample (degrees)
    :param sun_altitudes_deg: Sun altitude at each sample (degrees)
    :param airmass_limit: Maximum acceptable airmass; ``None`` means 10
    :param sun_alt_limit_deg: Samples with the Sun above this altitude are not observable
    :return: Margin per sample in degrees; NaN where the target altitude is NaN (e.g. the geocentre)
    :rtype: list
    """
    if airmass_limit is None:
        airmass_limit = 10
    altitude_limit_deg = math.degrees(math.asin(min(1.0, 1.0 / airmass_limit)))
    return [
        float(min(alt - altitude_limit_deg, sun_alt_limit_deg - sun_alt)) if alt == alt else math.nan
        for alt, sun_alt in zip(altitudes_deg, sun_altitudes_deg, strict=True)
    ]


def _crossing_fraction(margins, valid: int, invalid: int) -> float:
    """
    Fraction of the way from the valid sample to the invalid one at which the margin reaches zero.

    Falls back to the midpoint (0.5) without margins, or where they do not bracket zero.
    """
    if margins is None:
        return 0.5
    inside, outside = margins[valid], margins[invalid]
    if not (inside >= 0 and outside <= 0 and inside > outside):
        return 0.5
    return inside / (inside - outside)


def _run_interval(times: list, first: int, last: int, margins=None) -> Interval:
    """
    The interval covered by the run of valid samples ``times[first:last + 1]``.

    Extended towards the invalid sample on either side, except where the run reaches the end of the
    sampled range. See ``visibility_windows``.
    """
    start, end = times[first], times[last]
    if first > 0:
        start = times[first] - (times[first] - times[first - 1]) * _crossing_fraction(margins, first, first - 1)
    if last < len(times) - 1:
        end = times[last] + (times[last + 1] - times[last]) * _crossing_fraction(margins, last, last + 1)
    return start, end


def visibility_windows(samples: dict[str, tuple]) -> dict[str, list[Interval]]:
    """
    Turns sampled visibility into per-site intervals of contiguous valid samples.

    The edge of each run lies somewhere between its first (last) valid sample and the invalid sample
    before (after) it. Where a site's samples carry margins (see ``observability_margins``) the edge is
    placed where the margin, interpolated linearly between that pair, reaches zero; this is good to a few
    seconds at a 15-minute interval and about 20 s at 35 minutes. Without margins the midpoint of the pair
    is used, which can be wrong by half a sampling interval. Either way a single-sample run keeps a
    non-zero width (``cadence_window`` discards zero-width intervals). A run reaching the first or last
    sample is not extended, as nothing is known beyond the sampled range.

    :param samples: ``{site: (times, airmasses)}`` in the shape returned by
        ``tom_observations.utils.get_sidereal_visibility``, or ``{site: (times, airmasses, margins)}`` as
        returned by ``solsys_code.ephem_utils.get_nonsidereal_visibility``; a sample is valid when its
        airmass is neither ``None`` nor NaN
    :type samples: dict
    :return: ``{site: [(start, end), ...]}``, one interval per run of valid samples
    :rtype: dict
    """
    windows = {}
    for site, (times, airmasses, *rest) in samples.items():
        times = list(times)
        margins = rest[0] if rest else None
        valid = [_is_valid(airmass) for _, airmass in zip(times, airmasses, strict=True)]
        intervals = []
        first = None
        for index, is_valid in enumerate(valid):
            if is_valid:
                if first is None:
                    first = index
            elif first is not None:
                intervals.append(_run_interval(times, first, index - 1, margins))
                first = None
        if first is not None:
            intervals.append(_run_interval(times, first, len(valid) - 1, margins))
        windows[site] = intervals
    return windows


def _as_timedelta(period) -> timedelta:
    if isinstance(period, u.Quantity):
        return timedelta(seconds=period.to_value(u.s))
    return period


def _merge(intervals: list[tuple[timedelta, timedelta]]) -> list[tuple[timedelta, timedelta]]:
    """Merges overlapping or touching intervals; returns them sorted."""
    merged: list[tuple[timedelta, timedelta]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def cadence_window(intervals, period=24 * u.h) -> CadenceWindow:
    """
    Finds the window and midpoint to use for a periodic cadence from a set of visibility intervals.

    All intervals (from any number of sites, over any horizon of at least one period) are folded
    modulo ``period`` and merged; the largest gap on the resulting circle is taken as the break in
    the cadence and the window is its complement. Working on absolute times and folding means a
    window that straddles 00:00 UTC needs no special handling, and the same code serves non-daily
    cadences.

    :param intervals: iterable of ``(start, end)`` datetimes; empty or reversed intervals are ignored
    :param period: cadence period
    :type period: astropy.units.Quantity or datetime.timedelta
    :return: the window, anchored on the first occurrence at or after the earliest interval start
    :rtype: CadenceWindow
    :raises ValueError: if there are no usable intervals
    """
    period = _as_timedelta(period)
    intervals = [(start, end) for start, end in intervals if end > start]
    if not intervals:
        raise ValueError('No intervals to build a cadence window from')
    t0 = min(start for start, _ in intervals)

    folded: list[tuple[timedelta, timedelta]] = []
    for start, end in intervals:
        if end - start >= period:
            folded = [(timedelta(0), period)]
            break
        phase_start = (start - t0) % period
        phase_end = phase_start + (end - start)
        if phase_end > period:
            folded.append((phase_start, period))
            folded.append((timedelta(0), phase_end - period))
        else:
            folded.append((phase_start, phase_end))
    merged = _merge(folded)

    gaps = [(prev[1], nxt[0]) for prev, nxt in zip(merged, merged[1:], strict=False)]
    wrap_gap = (merged[-1][1], merged[0][0] + period)
    if wrap_gap[1] > wrap_gap[0]:
        gaps.append(wrap_gap)

    if not gaps:
        start = t0
        duration = period
        coverage = [(start, start + period)]
        sub_gaps: list[Interval] = []
    else:
        largest = max(gaps, key=lambda gap: gap[1] - gap[0])
        start_phase = largest[1] % period
        duration = period - (largest[1] - largest[0])
        start = t0 + start_phase
        # Rotate the merged intervals so they sit inside [start, start + duration)
        rotated = [((s - start_phase) % period, (s - start_phase) % period + (e - s)) for s, e in merged]
        coverage = [(start + s, start + e) for s, e in _merge(rotated)]
        sub_gaps = [(prev[1], nxt[0]) for prev, nxt in zip(coverage, coverage[1:], strict=False)]

    end = start + duration
    return CadenceWindow(
        start=start,
        end=end,
        midpoint=start + duration / 2,
        duration=duration,
        coverage=coverage,
        gaps=sub_gaps,
    )
