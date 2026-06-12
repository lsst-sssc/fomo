from datetime import date as date_cls
from datetime import datetime, time
from math import sqrt
from zoneinfo import ZoneInfo

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from solsys_code.solsys_code_observatory.models import Observatory

# Maps telescope name to MPC observatory code. `Observatory` (looked up via
# get_site()) remains the single source of truth for location and timezone.
SITES = {
    'Magellan-Clay': '268',
    'Magellan-Baade': '269',
    'NTT': '809',
    'FTS': 'E10',
}


def get_site(name: str) -> Observatory:
    """Resolves a telescope name to its Observatory record.

    Args:
        name: Telescope name, a key of SITES (e.g. 'Magellan-Clay').

    Returns:
        Observatory: the observatory record for this telescope's site.

    Raises:
        Observatory.DoesNotExist: if name is not a key in SITES, or no
            Observatory record exists for the resolved MPC obscode.
    """
    try:
        obscode = SITES[name]
    except KeyError as exc:
        raise Observatory.DoesNotExist(f'No site registered in SITES for telescope {name!r}') from exc
    return Observatory.objects.get(obscode=obscode)


def horizon_dip(altitude_m: float) -> float:
    """Horizon dip in degrees for an observer at altitude_m metres.

    dip = 1.76 arcmin * sqrt(altitude in metres), converted to degrees.

    Args:
        altitude_m: Observer altitude above sea level, in metres.

    Returns:
        float: dip angle in degrees (e.g. 1.4376 for 2402 m).
    """
    dip_arcmin = 1.76 * sqrt(altitude_m)
    return dip_arcmin / 60.0


def _solar_altitude(times: Time, location) -> np.ndarray:
    """Solar altitude in degrees for an array of Time objects at a location.

    Args:
        times: astropy Time array of evaluation epochs.
        location: astropy EarthLocation of the observer.

    Returns:
        np.ndarray: solar altitude in degrees for each time.
    """
    sun = get_sun(times)
    altaz = sun.transform_to(AltAz(obstime=times, location=location))
    return altaz.alt.deg


def _find_crossing(
    anchor: Time, location, threshold_deg: float, search_hours: float = 24, coarse_step_min: float = 1.0
) -> list[Time]:
    """Finds UTC times where solar altitude crosses threshold_deg.

    Performs a coarse scan over the window [anchor, anchor + search_hours],
    then refines each sign change with bisection to sub-second precision.
    Anchoring at local noon of the observing date (see _local_noon_utc) and
    scanning forward search_hours=24 guarantees both the evening sunset/dark
    crossing of that date and the following morning's sunrise/dark-end
    crossing fall within the window, in chronological (set, then rise) order.

    Args:
        anchor: astropy Time at the start of the search window (local noon).
        location: astropy EarthLocation of the observer.
        threshold_deg: solar altitude threshold to find crossings of, in degrees.
        search_hours: total width of the search window, in hours.
        coarse_step_min: coarse scan step size, in minutes.

    Returns:
        list[Time]: UTC times of each altitude crossing, in chronological order.
    """
    # +coarse_step_min so the window covers a full, closed [0, search_hours] range
    # (np.arange's exclusive upper bound would otherwise leave the last minute unscanned).
    offsets = np.arange(0, search_hours * 60 + coarse_step_min, coarse_step_min) * u.min
    times = anchor + offsets
    alt = _solar_altitude(times, location)
    crossings = []
    for i in range(len(alt) - 1):
        if (alt[i] - threshold_deg) * (alt[i + 1] - threshold_deg) < 0:
            # Bisection refine between times[i] and times[i+1]
            lo, hi = times[i], times[i + 1]
            lo_alt = alt[i]
            for _ in range(10):  # ~1/1024 of 1-min step -> sub-second precision
                mid = lo + (hi - lo) / 2
                mid_alt = _solar_altitude(Time([mid]), location)[0]
                if (mid_alt - threshold_deg) * (lo_alt - threshold_deg) < 0:
                    hi = mid
                else:
                    lo, lo_alt = mid, mid_alt
            crossings.append(lo)
    return crossings


def _local_noon_utc(local_date: date_cls, tz_name: str) -> Time:
    """Local noon of local_date, converted to UTC, as an astropy Time.

    Args:
        local_date: the local calendar date.
        tz_name: IANA timezone name for the site (e.g. 'America/Santiago').

    Returns:
        Time: local noon of local_date, expressed as a UTC astropy Time.
    """
    tz = ZoneInfo(tz_name)
    local_noon = datetime.combine(local_date, time(12, 0), tzinfo=tz)
    return Time(local_noon.astimezone(ZoneInfo('UTC')))


def sun_event(site: Observatory, date: date_cls, kind: str) -> tuple[Time, Time]:
    """Computes UTC sun-event crossing times for an observing night.

    Args:
        site: Observatory instance (from get_site()).
        date: local calendar date of sunset; the returned events cover the
            observing night starting on the evening of this date.
        kind: 'sun' for the dip-corrected sunset/sunrise threshold
            (-(0.833 + dip) degrees), or 'dark' for the -15 degree
            dark-window threshold (no dip correction).

    Returns:
        tuple[Time, Time]: (setting, rising) as astropy.time.Time objects,
            UTC scale.

    Raises:
        ValueError: if kind is not 'sun' or 'dark'; if site.timezone is
            unset; or if the solar altitude does not cross threshold
            exactly twice in the 24h window following local noon (e.g. a
            high-latitude site in summer where the sun never sets, or never
            gets dark).
    """
    if not site.timezone:
        raise ValueError(
            f'Observatory {site.short_name!r} (obscode={site.obscode}) has no timezone set; '
            'set Observatory.timezone (IANA name, e.g. "America/Santiago") before calling sun_event().'
        )

    location = site.to_earth_location()
    anchor = _local_noon_utc(date, site.timezone)

    if kind == 'sun':
        dip = horizon_dip(site.altitude)
        # 0.833 deg = standard solar semi-diameter (~16') + horizon refraction (~34')
        threshold = -(0.833 + dip)
    elif kind == 'dark':
        threshold = -15.0
    else:
        raise ValueError(f"kind must be 'sun' or 'dark', got {kind!r}")

    crossings = _find_crossing(anchor, location, threshold, search_hours=24)
    if len(crossings) != 2:
        raise ValueError(
            f'Expected 2 sun-event crossings for {site.short_name} on {date} '
            f'(kind={kind!r}), got {len(crossings)}: {crossings}. '
            'This can happen at high latitudes when the sun never sets or never '
            'reaches the requested threshold (e.g. midnight sun or no astronomical darkness).'
        )
    return crossings[0], crossings[1]
