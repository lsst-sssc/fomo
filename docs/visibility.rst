.. _visibility:

Visibility and cadence planning for non-sidereal targets
========================================================

The TOM Toolkit's *Plan* panel on the target page (airmass against time for each
observing site) only works for sidereal targets. FOMO replaces it for
``NON_SIDEREAL`` targets with a panel that computes the target's position from
its orbital elements with the same REBOUND + ASSIST pipeline as the ephemeris
view, plots the airmass from each LCO site, and reports the observing window and
midpoint to use for a daily LCO cadence request.

Where to find it
----------------

Open a non-sidereal target, stay on the **Observe** tab and scroll to **Plan**.
Fill in the form and press **Plan**:

Start Time, End Time
   UTC dates (times are taken as 00:00 UTC). One or two nights is the usual
   choice; longer ranges work but are sampled more coarsely (see below).

Maximum Airmass
   Samples above this airmass are treated as not observable. Defaults to 2.5.

Sites
   The LCO sites to sample: Cerro Tololo (LSC), Sutherland (CPT), Siding Spring
   (COJ), Tenerife (TFN), McDonald (ELP) and Haleakala (OGG). All six are
   selected by default; deselect the northern (or southern) sites if the
   programme can only use part of the network.

The parameters are carried in the URL (``?start_time=…&end_time=…&airmass=…&sites=…``)
so a plan can be bookmarked or shared.

What is shown
-------------

**Airmass plot.** One line per site, airmass increasing downwards as in the TOM
Toolkit's own plot. A sample is dropped (the line is broken) when the target is
below the horizon, above the airmass limit, or the Sun is above -18° (i.e.
outside astronomical night). A site from which the target is never observable
in the range is left greyed out in the legend; click it to show the (empty)
trace anyway.

**Cadence window.** Below the airmass plot, FOMO merges the visibility windows
of all selected sites and reports, for example::

   09:00–08:00 UTC (23.0 h): midpoint 20:30 UTC ± 11.5 h; gaps within the window: 17:15–17:30 UTC

This is the longest stretch of the 24-hour day during which at least one site
can observe the target, together with its centre and half-width. It is computed
by folding all windows onto one 24-hour period and taking the complement of the
largest gap, so windows that straddle 00:00 UTC are handled naturally. Short
holes inside the window (here a 15-minute gap between COJ setting and CPT
starting) are listed but do not break it.

The bar chart underneath shows the same information graphically: one row per
site with a bar for each visibility window, an **All sites** row with the merged
coverage inside the cadence window, and a dashed line at the midpoint. Hover
over a bar for its exact start and end times.

Using the result for an LCO cadence request
-------------------------------------------

The summary maps directly onto the fields of an LCO cadence (``period`` /
``jitter``) request:

- ``period``: 24 hours (the folding period);
- ``jitter``: the window duration (23.0 h in the example);
- the first observing window is centred on the midpoint (20:30 UTC), so the
  request's start should be ``midpoint - jitter/2`` on the first night.

The scheduler is then free to place each night's observation at whichever site
is up at the time, and the window excludes only the part of the day when no
selected site can see the target.

Notes and limitations
---------------------

- The sampling interval is 15 minutes for ranges up to about three days and is
  coarsened for longer ranges so that no more than 300 samples per site are
  computed (35 minutes for a week, about 2.5 hours for a month). Windows are
  bounded by the first and last *valid* sample, so they are conservative by up
  to one interval at each end and short windows can be missed at coarse
  intervals. Use a shorter range for objects that are only briefly observable.
- Each sample is a full light-time-corrected integration, so a plan takes a few
  seconds for one or two nights and around ten seconds for a week over all six
  sites.
- The panel needs an :class:`~solsys_code.solsys_code_observatory.models.Observatory`
  row for each site (MPC codes W85, K91, Q63, Z31, V38 and T04). Missing rows
  are created automatically from the LCO facility's site list.
- Positions are topocentric and airless; magnitude is not considered.

Using it from code
------------------

The pieces are available separately for scripts and notebooks:

.. code-block:: python

   from datetime import datetime

   from solsys_code.ephem_utils import get_nonsidereal_visibility
   from solsys_code.solsys_code_observatory.models import Observatory
   from solsys_code.visibility import cadence_window, visibility_windows

   sites = {code: Observatory.objects.get(obscode=obscode) for code, obscode in [('LSC', 'W85'), ('CPT', 'K91'), ('COJ', 'Q63')]}
   visibility = get_nonsidereal_visibility(target, sites, datetime(2026, 9, 16), datetime(2026, 9, 18), 15, airmass_limit=2.5)
   windows = visibility_windows(visibility)
   window = cadence_window([interval for intervals in windows.values() for interval in intervals])
   print(window.midpoint, window.duration)

``get_nonsidereal_visibility`` returns ``{site: (times, airmasses)}`` in the same
shape as ``tom_observations.utils.get_sidereal_visibility``; ``visibility_windows``
and ``cadence_window`` (in ``solsys_code.visibility``) are pure functions with no
Django or SPICE dependencies. ``cadence_window`` accepts a ``period`` other than
24 hours for non-daily cadences.
