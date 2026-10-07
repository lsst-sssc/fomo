Scout Orbital Element History
=============================

This note records the finding (2026-09-10) that FOMO keeps no history of a Scout
candidate's orbital elements, why that is, and the options for adding it.

Background
----------

Beyond driving the Rubin ToO views, this FOMO instance has a secondary purpose:
keeping the history of JPL Scout candidates, which the Scout service itself does
not retain. ``tom_jpl`` 0.3.0 provides ``ScoutDetailHistory`` for this, appending
a row each time Scout recomputes a candidate (keyed on ``last_run``). That
covers the Scout-specific quantities — digest scores, impact rating,
close-approach distance, arc, number of observations, orbit-fit RMS, positional
uncertainties, and the predicted RA/Dec/V/rate at ``t_ephem`` — but **not** the
orbit those quantities were derived from.

Where the elements live and how they change
-------------------------------------------

The orbital elements are on :class:`~tom_targets.models.Target`, not on
``ScoutDetail``. They are refreshed by the hourly ``rundataquery <id>`` cron job:
for every candidate passing the saved query's cuts, ``ScoutDataService`` fetches
the object with ``?tdes=…&orbits=1``, and ``to_target()`` compares the twelve
element fields (``arg_of_perihelion``, ``lng_asc_node``, ``inclination``,
``eccentricity``, ``epoch_of_elements``, ``epoch_of_perihelion``, ``perihdist``,
``abs_mag``, ``slope``, ``semimajor_axis``, ``mean_daily_motion``,
``mean_anomaly``) against the stored ``Target`` and overwrites any that differ.
(``orbits.data[0]`` is the nominal solution; confirmed with the Scout team.)

The previous values are not kept anywhere:

* ``Target`` has only ``created`` / ``modified`` timestamps.
* ``ScoutDetailHistory`` carries no element fields.
* FOMO has no model-audit package installed (no ``django-reversion``,
  ``django-simple-history``, ``django-auditlog`` or ``django-pghistory``).
* Django's admin ``LogEntry`` records only edits made through ``/admin/``, as a
  change message rather than old values; saves from management commands are
  invisible to it.

So the only trace of an element change is ``Target.modified`` advancing, usually
alongside a new ``ScoutDetailHistory`` row with the same ``last_run``. That says
*that* the orbit changed, not *what it was*. The nearest proxy for orbit
evolution is ``rms`` / ``num_obs`` / ``arc`` per history row, which tracks fit
quality over time but not the solution.

A second, related limitation: because the element refresh rides on
``rundataquery``, a candidate that drifts outside the saved query's cuts stops
receiving element updates even though ``updatescout`` continues to track its
``ScoutDetail``. Running an unrestricted saved query avoids this (see the
:ref:`user documentation <scout_rubin_too>`).

Options
-------

In rough order of effort and fit:

1. **Add the element fields to ``ScoutDetailHistory``** (change in ``tom_jpl``).
   ``to_target()`` already holds both the old and new values at the moment it
   diffs them, so writing the elements alongside the existing history row is
   natural: one append-only table, keyed on ``last_run``, queryable per
   candidate, no new dependency. This is the best fit for retrospective analysis
   of how Scout orbits evolve, and the option to raise upstream first. It is a
   migration plus a small change to ``to_target`` / ``store_scout_detail``.

2. **``django-simple-history`` on ``BaseTarget``.** Captures every field on every
   save with ``history_date`` / ``history_user``. Cheap to add, but ``tom_targets``
   would need to register it (or FOMO would ``register(Target)`` from outside),
   it snapshots non-Scout targets and non-element saves too (renames, ``modified``
   churn), and the history table lives in FOMO rather than with the Scout data.

3. **``django-reversion``**, as NEOexchange uses. Similar to (2) but serialises
   whole objects to JSON versions: better for point-in-time restore, worse for
   "plot eccentricity against time".

Decision
--------

Not implemented as of this note. Option 1 is the recommended path and should be
proposed as a ``tom_jpl`` change; options 2 and 3 are fallbacks if a FOMO-only
solution is needed sooner.
