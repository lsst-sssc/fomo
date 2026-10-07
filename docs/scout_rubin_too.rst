.. _scout_rubin_too:

JPL Scout candidates and Rubin ToO filtering
=============================================

FOMO ingests unconfirmed NEO candidates from the JPL `Scout
<https://cneos.jpl.nasa.gov/scout/intro.html>`_ service (the objects on the MPC's
NEO Confirmation Page, NEOCP), keeps them current as Scout recomputes their orbits,
retires them when they leave the NEOCP, and evaluates them against the SSSC NEOs
Working Group *Filter Criteria for near-Earth Object (NEO) Rubin ToO Triggers*
(v0.2). This page describes how to set that up and use it.

The Scout ingestion itself is provided by the `tom_jpl
<https://github.com/TOMToolkit/tom_jpl>`_ TOM Toolkit module (0.3.0 or later);
the Rubin ToO filters and views are FOMO's own (``solsys_code.rubin_too`` and
``solsys_code.scout_views``).

How it fits together
--------------------

.. code-block:: text

   MPC NEOCP ──► JPL Scout API ──► rundataquery <id> ──► Target + ScoutDetail (active)
                                     (hourly)                 │ creates new candidates and
                                                              │ refreshes existing ones'
                                                              │ orbital elements
                                   updatescout ───────────────┤ refresh ScoutDetail from the
                                     (hourly)                 │ Scout roster; each new solution
                                                              │ is appended to ScoutDetailHistory
                                                              ▼
                                                    left the NEOCP? → ScoutDetail.active = False
                                                              │
                                   updatescout ───────────────┘ MPC "Previous NEOCP Objects" page:
                                     (daily)                    rename to the IAU designation, or
                                                                record lost / does-not-exist /
                                                                artificial outcomes

Each Scout candidate becomes a ``NON_SIDEREAL`` :class:`~tom_targets.models.Target`
(with the orbital elements Scout provides) plus a ``tom_jpl.models.ScoutDetail`` row
holding the Scout-specific quantities: digest scores, impact rating, close-approach
distance, orbit-fit RMS, number of observations, arc length, position uncertainties
now and at +1 day, and the predicted RA/Dec/V/rate at ``t_ephem``. Scout keeps no
history of its own, so every recomputation is copied into ``ScoutDetailHistory``;
that table is what the first-pass statistics are built from.

Setup
-----

``tom_jpl`` is a FOMO dependency and is already wired into ``settings.py``:

.. code-block:: python

   INSTALLED_APPS = [
       ...
       'tom_jpl',
   ]

   DATA_SERVICES = {
       'Scout': {
           'base_url': 'https://ssd-api.jpl.nasa.gov/scout.api',
       },
   }

After installing or upgrading, apply the ``tom_jpl`` migrations (0001–0005 as of
0.3.0) with ``python manage.py migrate``. Do this **immediately** after a
``pip install`` if the cron jobs below are running: a ``tom_targets``/``tom_jpl``
release that ships a migration will make every ``manage.py`` command fail with
``no such column: tom_targets_basetarget.<field>`` until the migration is applied,
and a cron cycle that fires in between is silently lost.

Interactive use
---------------

**Querying Scout.** Navigate to **Data Services → Scout** in the navbar. The form
lets you apply cuts (NEO score, geocentric score, impact rating, close-approach
distance, uncertainties, …) that the Scout API itself cannot; they are applied
client-side to the full roster. Tick **Save Query** to keep the parameter set for
re-use from the command line (see below). Running the query shows a results table;
tick the candidates you want and press **Create Targets**.

**On a target's detail page**, two tabs are added by ``tom_jpl``:

* **Scout Details** — the current ``ScoutDetail`` values, whether the candidate is
  still active, and (once it has left the NEOCP) the MPC outcome.
* **Scout History** — every stored recomputation, newest first.

**On the target list** (**Targets → Targets**), FOMO adds:

* A **Scout** selector beside the search box: *Active Scout candidates*, *Retired
  Scout candidates*, *Any Scout candidate*, or *Not from Scout*. It combines with
  the other filters and is honoured by **Export Filtered Targets**.
* An **Origin** column (sortable): ``Scout`` for targets with a ``ScoutDetail``,
  ``MPC`` for other non-sidereal targets (e.g. from ``fetch_jplsbdb_objects``),
  blank for sidereal targets.

Keeping candidates up to date
-----------------------------

Saved queries are identified by a **numeric id that is specific to each database**
(so it differs between a laptop and the server). List them with:

.. code-block:: console

   >> python manage.py listqueries
    ID       Name       Data Service             Last Run
   --- ---------------- ------------ --------------------------------
     1 Scout_basic_geo5        Scout 2026-09-10 22:00:02.947468+00:00

Then two commands do the work:

``rundataquery <query_id>``
   Re-runs the saved Scout query and, for **every** candidate passing the query's cuts,
   fetches that object's current orbit (one ``?tdes=…&orbits=1`` request each) and calls
   ``to_target``: new candidates get a ``Target`` + ``ScoutDetail``; existing ones have
   their orbital elements (and ``ScoutDetail``) updated if Scout has recomputed them. This
   is the only path that refreshes a ``Target``'s elements, so it needs to run hourly —
   and a candidate that has drifted outside the saved query's cuts (e.g. its geocentric
   score rose) stops getting element updates while ``updatescout`` still tracks it.

   .. warning::
      ``rundataquery`` catches its own failures and logs them rather than raising, so
      a failed run still exits 0. From ``cron``, a non-zero exit check will not notice
      a broken run; look for ``Finished querying targets`` in the output, or watch
      for a stale ``Last Run`` in ``listqueries``.

``updatescout``
   Two phases, with different natural cadences:

   * ``--skip-designations`` runs only **reconciliation**: one request for the whole
     Scout roster, then for every active candidate either store the new orbit
     solution (if Scout's ``lastRun`` has advanced) or, if it is absent from the
     roster, mark it retired (``active=False``). History is never deleted. If Scout
     returns nothing, or fewer objects than it claims to have, reconciliation is
     skipped with a warning rather than retiring everything.
   * ``--skip-reconcile`` runs only the **designation pass**: scrapes the MPC's
     Previous NEOCP Objects page for retired candidates, renames the ``Target`` to
     its IAU designation (keeping the NEOCP name as an alias), and records
     lost / does-not-exist / artificial-satellite outcomes. The page covers months
     of departures in one request, so daily is plenty.

   ``--dry-run`` reports what either phase would do without writing anything, and is
   the quickest way to check the state of things by hand.

The deployed crontab (times UTC) is the pattern to copy:

.. code-block:: text

   0  * * * *  python manage.py rundataquery 1
   17 * * * *  python manage.py updatescout --skip-designations
   47 4 * * *  python manage.py updatescout --skip-reconcile

Rubin ToO views
---------------

The **Rubin ToO** navbar menu has two pages.

**Current candidates** lists every *active* Scout candidate whose latest stored
values pass **all** of the Section 2.1 filters, with the per-filter thresholds shown
as column tooltips:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Filter
     - Threshold
   * - NEO digest score
     - ≥ 98
   * - Geocentric digest score
     - < 2
   * - Absolute magnitude H
     - < 99 (i.e. no size cut at present)
   * - Scout impact rating
     - ≥ 3
   * - Orbit-fit RMS
     - < 1.0 arcsec
   * - Observations / arc
     - nObs > 5 and arc > 1 hour
   * - Predicted V magnitude
     - > 21.6 (Dec > 0) / > 21.8 (Dec ≤ 0)
   * - 1σ uncertainty at +1 day
     - > 60 arcmin (Dec > 0) / > 180 arcmin (Dec ≤ 0)
   * - Sky motion
     - < 25 arcsec/min

Per Section 2.3 of the criteria, a candidate drops off this page as soon as any
filter stops being satisfied. The Section 2.2 airmass/observability filter is not
applied here — it needs a site- and time-specific ephemeris — so "passing" means
*eligible*, not *observable tonight*.

An empty page is the normal state: ``impact rating ≥ 3`` is the binding constraint
and most of the time no object on the NEOCP meets it.

**First-pass stats** walks ``ScoutDetailHistory`` and counts, per calendar year and
per filter, how many candidates crossed from failing to passing for the first time,
plus a *Combined* row for the first time all filters held simultaneously. It is the
retrospective view of how often a Rubin ToO would have been triggered.

The filter logic lives in ``solsys_code.rubin_too``; ``evaluate_filters(scout_detail)``
returns the per-filter booleans and ``passes_filters`` the conjunction, and both work
on any object with ``ScoutDetail``'s attributes (history rows included).

Checking it is working
----------------------

.. code-block:: console

   >> python manage.py updatescout --dry-run --skip-designations
   Reconciling 81 active Scout candidate(s)...
     [dry-run] would refresh CERTMY2 (Scout has recomputed it)
     [dry-run] would retire P22pL3X (no longer on Scout)
     would refresh 3 candidate(s); would retire 1 candidate(s); 77 unchanged.

A healthy installation shows a handful of refreshes and the occasional retirement;
the number of active candidates should track the count on the `Scout site
<https://cneos.jpl.nasa.gov/scout/>`_ minus whatever the saved query's cuts exclude.

Other things to know:

* ``python manage.py showmigrations | grep '\[ \]'`` lists unapplied migrations —
  the first thing to check when every command starts failing after an upgrade.
* On SQLite, ``manage.py`` prints three ``models.W047`` warnings about
  ``tom_dataproducts`` unique constraints with ``nulls_distinct``. These are from
  tomtoolkit and are harmless for the Scout workflow.
* ``tom_jpl`` checks the Scout API's response signature (currently
  ``{'source': 'NASA/JPL Scout API', 'version': '1.3'}``). On a mismatch it only
  logs ``Signature of response from Scout API does not match expected signature``
  and ingests nothing — which looks exactly like a quiet week on the NEOCP. If the
  candidate count stops changing, check the log for that line.

Related
-------

`scout-alert-bridge <https://github.com/lsst-sssc/scout-alert-bridge>`_ is a headless
sibling of this workflow: the same ``tom_jpl`` ingestion and reconciliation, with the
Rubin ToO filters applied to publish ``new_candidate`` / ``updated`` / ``cancelled`` /
``left_neocp`` events to a SCiMMA Hopskotch Kafka topic for Rubin's ToO system. FOMO's
views are the human-facing equivalent of that stream.
