Telescope Runs on the Calendar
==============================

This document records the feasibility study and implementation plan for showing
follow-up telescope runs on the TOM Toolkit calendar (``tom_calendar``).  It was
written after a research spike (2026-06-10) that validated the astronomy and the
data model end-to-end.

Background
----------

FOMO coordinates follow-up of Solar System targets across several telescopes.
The scheduled time on those telescopes falls into two scheduling models:

* **Classically-scheduled nights** — whole nights assigned to a programme in
  advance.  FOMO currently cares about two such telescopes in Chile:

  * **NTT / EFOSC2** at ESO La Silla Observatory.
  * **Magellan** (Baade / Clay) at Las Campanas Observatory.

* **Queue-scheduled blocks** — short blocks placed dynamically by a scheduler
  within an eligible window:

  * **FTS / MuSCAT4** at Siding Spring Observatory, operated by Las Cumbres
    Observatory (LCOGT).  Blocks are roughly six hours per night, schedulable
    across a range of nights bounded by Moon phase.

The goal is to surface this allocated time on the new ``tom_calendar`` calendar
(live during development at ``/calendar/``) so that follow-up can be planned
against known telescope access.

Key finding
-----------

**The feature is feasible with no changes to** ``tom_calendar`` **and no
database migrations.**  The stock :class:`tom_calendar.models.CalendarEvent`
already carries ``title``, ``description``, ``start_time``, ``end_time``,
``url``, ``telescope``, ``instrument``, ``proposal``, ``user`` and a
``target_list`` foreign key.  A telescope run maps directly onto it.
``tom_calendar`` is a third-party install (part of the TOM Toolkit), **not**
FOMO/``tom_jpl`` code, so the design deliberately reuses existing fields rather
than patching the package.

The Data Model
--------------

``CalendarEvent`` fields and how they surface in the UI (verified by reading the
``tom_calendar`` templates):

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Field
     - Type
     - Where it shows
     - Use for runs
   * - ``title``
     - ``CharField(200)``
     - **Grid label** (truncated ~16 chars) and edit modal
     - Short label; the only place to surface status at a glance
   * - ``description``
     - ``TextField``
     - Edit modal only
     - Dark window (UTC), original run string, status, notes
   * - ``start_time`` / ``end_time``
     - ``DateTimeField``
     - Grid (timed events show start) and modal
     - Sunset / sunrise (classical) or block bounds (queue)
   * - ``telescope`` / ``instrument``
     - ``CharField(200)``
     - Edit modal
     - ``NTT`` / ``EFOSC2`` etc., kept clean and queryable
   * - ``url``
     - ``URLField``
     - Edit modal (link)
     - Click-through; **idempotency key** for synced blocks
   * - ``proposal`` / ``user``
     - ``CharField(200)``
     - Edit modal
     - Programme ID / observer, if available
   * - ``target_list``
     - FK ``TargetList``
     - **Grid badge** and modal
     - Attach the night's targets

The related ``EventTodo`` model (per-event checklist; its active count renders on
the grid) and a read-only ``color`` property (derived from ``pk``, so it cannot
be set to encode status) are available but not central to this design.

Because only ``title`` and the ``target_list`` badge render on the calendar
grid, any status that must be glanceable belongs in ``title``; everything else
lives in ``description`` and the typed fields, visible on click-through.

Astronomy: Night Boundaries
---------------------------

For classical runs each night becomes one event spanning **sunset to sunrise**,
with the **-15 deg dark window** recorded in the description.  (-15 deg is a
deliberate FOMO choice for faint targets; textbook nautical twilight is -12 deg
and astronomical is -18 deg.)

Sun altitudes are computed with ``astropy`` (``get_sun`` -> ``AltAz`` at the site
``EarthLocation``).  Two corrections matter:

* **Refraction + solar semidiameter:** geometric sunrise/sunset uses a threshold
  Sun altitude of -0.833 deg.
* **Horizon dip** from the observatory's elevation.  At ~2400 m the visible
  horizon is depressed, so sunset is later and sunrise earlier.  The dip is the
  Nautical Almanac formula

  .. math:: \mathrm{dip} = 1.76' \sqrt{h_\mathrm{metres}}

  derived from spherical geometry (:math:`\theta \approx \sqrt{2h/R}`,
  :math:`R = 6371` km) with terrestrial refraction :math:`k \approx 1/6` folded
  in.  At 2402 m this is 1.44 deg, so the sunset/sunrise threshold is
  -(0.833 + 1.44) = -2.27 deg.  The dip is **not** applied to the -15 deg window
  (the Sun is nowhere near the visible horizon there).

**Validation.** Computed times were checked against Las Campanas Observatory's
own ephemeris tool (John Thorstensen's *skycalc*, served from
``https://www.lco.cl/eph/``) for June 2026:

* Sunset / sunrise agree to **<= 1 minute** once the horizon dip is applied
  (without it there was a consistent ~8 minute error).
* Astronomical twilight (-18 deg) agrees to **<= 1 minute**, confirming the
  twilight solver and hence the -15 deg numbers.
* The tool's "Chilean time (4 hr W)" for June matches ``zoneinfo`` returning
  UTC-4 for ``America/Santiago``.

Observatory Sites
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 12 22 20

   * - Telescope @ Site
     - Lat (deg)
     - Lon (deg)
     - Alt (m)
     - Timezone
     - Source
   * - Magellan @ Las Campanas
     - -29.0146
     - -70.6926
     - 2402
     - ``America/Santiago``
     - Validated vs Las Campanas skycalc tool
   * - NTT @ La Silla
     - -29.2567
     - -70.7300
     - 2347
     - ``America/Santiago``
     - astropy ``of_site('lasilla')``
   * - FTS @ Siding Spring
     - -31.2734
     - 149.0612
     - 1149
     - ``Australia/Sydney``
     - astropy ``of_site('Siding Spring Observatory')``

Chile uses ``America/Santiago`` (DST: UTC-4 in austral winter, UTC-3 summer).
NSW uses ``Australia/Sydney`` (DST: AEST UTC+10 winter, AEDT UTC+11 summer).
Both are handled by ``zoneinfo``; ``tzdata`` is installed.  At +149 deg longitude
a Siding Spring night sits *within* a single UTC date (~07:00-21:00 UTC) rather
than straddling UTC midnight as Chilean nights do.

Classical Run Input Format
--------------------------

Runs are recorded as free text lines, ``telescope instrument [status] daterange
[(status)]``.  Observed examples::

    NTT EFOSC2 allocation 9-13 July
    Magellan IMACS 13-19 July (proposed)
    Magellan Proto-Lightspeed  Jul 8-12 (proposed)

Parsing rules (prototype verified against all three lines):

* ``telescope`` = token 0 (maps to a site: ``NTT`` -> La Silla, ``Magellan`` ->
  Las Campanas); ``instrument`` = token 1 (may be hyphenated, e.g.
  ``Proto-Lightspeed``).
* Date range ``D-D`` with the month name appearing **before or after** the day
  range (``9-13 July`` and ``Jul 8-12`` both occur).
* **No year** is given.  Default to the current year, with special handling for
  runs beginning in late December (roll into the next year).
* Status comes from a parenthetical ``(...)`` or a bare known word
  (``allocation``, ``proposed``, ...).

**Night convention (confirmed from the Las Campanas telescope schedule).**  Run ``Start``
and ``End`` dates are **both observing nights** by evening date; the next run
begins the day *after* ``End``.  (E.g. a Las Campanas run ``Start 2026-06-08 /
End 2026-06-10`` covers the nights of the evenings of the 8th, 9th and 10th; a
different instrument and PI appear on Baade from the 11th.)  Therefore a run from
evening ``S`` to evening ``E`` yields ``E - S + 1`` nights, one event per evening
date ``d`` with ``start = sunset(d)``, ``end = sunrise(d+1)``.  Consecutive runs
tile without overlap.

Queue Runs: #1 + #3
-------------------

Queue scheduling is softer than a classical night, so it is represented two ways:

**#1 - Window banner (the plan).**  One multi-day ``CalendarEvent`` per queue
run.  ``start_time`` / ``end_time`` are the eligible window, supplied **in UTC**
(no timezone conversion).  ``telescope='FTS'``, ``instrument='MuSCAT4'``,
``title='FTS/MuSCAT4 (queue)'``; description records ~6 h/night, total hours and
the Moon constraint.  No per-night blocks are fabricated, since the scheduler may
use only some nights.

**#3 - Real blocks (the truth), synced from observation records.**  FOMO already
has ``LCOFacility`` configured (``settings.py``), so submitted observations exist
as :class:`tom_observations.models.ObservationRecord` rows with
``facility='LCO'``, ``target``, ``parameters``, ``observation_id``, ``status``,
``scheduled_start`` and ``scheduled_end``.  The LCO/OCS facility's
``update_observation_status`` populates the scheduled times from the current
block and reports status from the vocabulary ``PENDING``, ``COMPLETED`` and the
terminal states ``WINDOW_EXPIRED`` / ``CANCELED`` / ``FAILURE_LIMIT_REACHED`` /
``NOT_ATTEMPTED``.

For each LCO record with scheduled times present, the sync **upserts** a
``CalendarEvent`` at the real block time.  The idempotency key is the ``url``
field set to the canonical portal URL
``https://observe.lco.global/requestgroups/<observation_id>/`` — unique per
request and doubling as the click-through.  Status is reflected in ``title`` (and
in full, with ``observation_id``, in ``description``); terminal-failure records
either delete their block or render it struck-through.

Implementation Plan
-------------------

The work is staged so each step is independently testable.

**Stage 1 — site / ephemeris helper.**  A small module (e.g.
``solsys_code/telescope_runs.py``) with:

* a ``SITES`` registry mapping telescope name -> (site label, ``EarthLocation``,
  timezone);
* ``sun_event(site, date, kind)`` returning UTC sunset/sunrise/-15 deg-dark
  crossings using the dip-corrected thresholds above.

**Stage 2 — classical ingest command.**  ``load_telescope_runs`` management
command (modelled on ``fetch_jplsbdb_objects``): parse run lines, expand each to
one ``CalendarEvent`` per night via Stage 1, idempotent on re-run.

**Stage 3 — queue window banners (#1).**  Extend the command (or add a flag /
input mode) to accept FTS UTC windows and create one banner ``CalendarEvent`` per
queue run.

**Stage 4 — observation-record sync (#3).**  ``sync_lco_observation_calendar``
(management command, or a ``post_save`` signal on ``ObservationRecord`` in
``solsys_code`` since FOMO owns that code), run after
``update_all_observation_statuses``; upsert ``CalendarEvent`` rows keyed on the
portal ``url``.

Success Criteria
----------------

Testable, verifiable acceptance criteria.  Django-DB tests live under
``solsys_code/tests/`` (run with ``./manage.py test``); pure helpers may also be
unit-tested.

*Stage 1 — ephemeris helper*

#. For Las Campanas, June 2026, computed sunset and sunrise (dip-corrected) are
   within **2 minutes** of the Las Campanas *skycalc* tool for at least the sample nights
   Jun 1/10/20/30.  (Observed: <= 1 min.)
#. Computed astronomical twilight (-18 deg) for Jun 10 2026 is within 2 minutes
   of the tool's ``twi.end`` / ``twi.beg`` (19:16 / 06:08 local).
#. ``America/Santiago`` resolves to UTC-4 in June and UTC-3 in January;
   ``Australia/Sydney`` to UTC+10 in July and UTC+11 in January (asserted).
#. The horizon-dip helper returns 1.44 deg +/- 0.02 at 2402 m.

*Stage 2 — classical ingest*

#. Parsing the three sample lines yields the expected
   ``(telescope, instrument, status, year, month, day1, day2)`` tuples, including
   month-before and month-after date orders and the hyphenated instrument.
#. ``NTT EFOSC2 allocation 9-13 July`` creates **5** events; ``Magellan IMACS
   13-19 July`` creates **7**; ``Magellan Proto-Lightspeed Jul 8-12`` creates
   **5** (``E - S + 1``, inclusive).
#. Each created event has ``start_time`` = dip-corrected sunset of its evening
   date and ``end_time`` = sunrise of the following morning, both timezone-aware
   UTC, with ``end_time > start_time`` and duration between 8 and 15 hours.
#. ``telescope`` and ``instrument`` are set from the line; the -15 deg dark
   window and the original line appear in ``description``.
#. Running the command **twice** on the same input does not duplicate events
   (idempotent).
#. A line whose run starts in late December creates events in the following
   calendar year.

*Stage 3 — queue window banner (#1)*

#. An FTS UTC window creates exactly **one** ``CalendarEvent`` with
   ``start_time`` / ``end_time`` equal to the supplied UTC bounds (no offset
   applied), ``telescope='FTS'``, ``instrument='MuSCAT4'``.
#. The banner spans multiple dates (renders as an all-day banner) and records the
   per-night hours and Moon constraint in ``description``.

*Stage 4 — observation-record sync (#3)*

#. Given an ``ObservationRecord`` (``facility='LCO'``) with ``scheduled_start`` /
   ``scheduled_end`` set, the sync creates **one** ``CalendarEvent`` at those
   exact times with ``url`` = the portal request URL.
#. Re-running the sync after the record's ``status`` changes updates the existing
   event (matched by ``url``) rather than creating a second one.
#. A record in a terminal-failure state (e.g. ``WINDOW_EXPIRED``) results in its
   event being removed (or marked), per the chosen policy.
#. Records without scheduled times (``PENDING`` only) create no block event.

*End-to-end / manual verification*

#. After ingesting the three sample classical runs plus one FTS window, the
   ``/calendar/`` July 2026 view shows the La Silla and Las Campanas nights and
   the FTS queue banner; opening an event shows the correct telescope,
   instrument, UTC times and dark window.
#. The full suite passes: ``./manage.py test solsys_code`` and ``python -m
   pytest`` both green; ``ruff check .`` and ``ruff format --check .`` clean.

Open Items
----------

* The **FTS queue run input format** (#1) is not yet fixed; an example line of
  how a UTC window + hours + Moon constraint is recorded is needed before
  Stage 3.
* Terminal-failure policy for #3 (delete vs strike-through) to be confirmed.
* Whether ``Magellan`` should distinguish Baade vs Clay in ``telescope`` (the
  ephemeris is identical; both are at Las Campanas).
* The Stage 1 ``SITES`` dict hardcodes telescope name -> MPC obscode, so
  adding a new telescope requires a code change. Consider replacing it with a
  lookup by ``Observatory.short_name`` directly (data-driven, no code change
  to add sites) in Stage 2+. Also note ``to_earth_location()``/``sun_event()``
  assume a ground-based site; a guard against space-based observatories
  (``Observatory.SATELLITE_OBSTYPE``, e.g. JWST/274) would be needed if
  ``SITES`` (or its replacement) is ever extended to non-ground sites.
