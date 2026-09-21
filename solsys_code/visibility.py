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


def _run_interval(times: list, first: int, last: int) -> Interval:
    """
    The interval covered by the run of valid samples ``times[first:last + 1]``.

    Extended halfway towards the invalid sample on either side, except where the run reaches the end
    of the sampled range. See ``visibility_windows``.
    """
    start = times[first] if first == 0 else times[first - 1] + (times[first] - times[first - 1]) / 2
    end = times[last] if last == len(times) - 1 else times[last] + (times[last + 1] - times[last]) / 2
    return start, end


def visibility_windows(samples: dict[str, tuple]) -> dict[str, list[Interval]]:
    """
    Turns sampled visibility into per-site intervals of contiguous valid samples.

    Each run is extended to the midpoint between its first (last) valid sample and the invalid sample
    before (after) it: the crossing lies somewhere between that pair, so the midpoint is the best
    estimate available without recomputing the ephemeris. Taking the valid sample itself would clip
    every window by up to one sampling interval at each end, always in the same direction, and would
    collapse a single-sample run to zero width (which ``cadence_window`` then discards). A run
    reaching the first or last sample is not extended, as nothing is known beyond the sampled range.

    The residual error is the curvature of the altitude curve across one interval; interpolating the
    crossing from the surrounding samples would remove it.

    :param samples: ``{site: (times, airmasses)}`` in the shape returned by
        ``tom_observations.utils.get_sidereal_visibility``; a sample is valid when its airmass is
        neither ``None`` nor NaN
    :type samples: dict
    :return: ``{site: [(start, end), ...]}``, one interval per run of valid samples
    :rtype: dict
    """
    windows = {}
    for site, (times, airmasses) in samples.items():
        times = list(times)
        valid = [_is_valid(airmass) for _, airmass in zip(times, airmasses, strict=True)]
        intervals = []
        first = None
        for index, is_valid in enumerate(valid):
            if is_valid:
                if first is None:
                    first = index
            elif first is not None:
                intervals.append(_run_interval(times, first, index - 1))
                first = None
        if first is not None:
            intervals.append(_run_interval(times, first, len(valid) - 1))
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
