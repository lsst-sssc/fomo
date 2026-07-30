Telescope Runs Calendar — Operator Runbook
===========================================

This is the how-to-run companion to the
:doc:`/design/telescope_runs_calendar` design document -- see that page for
the *why* (dip-corrected sunset/sunrise, the -15 deg dark window, the
queue-vs-classical scheduling models, and so on). This page is deliberately
task-oriented: it walks through each management command and staff action as
a "How do I...?" question, followed by a quick-reference cheat-sheet and a
troubleshooting section.

This runbook assumes you already have FOMO installed and can run
``python3 manage.py <command>`` from an activated virtual environment; see
:ref:`running-management-commands` if you need that background first.

How do I load a classical telescope schedule?
-----------------------------------------------

``load_telescope_runs`` reads a plain-text schedule file -- one classical
run per line, e.g. ``NTT EFOSC2 allocation 9-13 July`` -- and expands each
run into one ``CalendarEvent`` per observing night, with sunset/sunrise and
the -15 deg dark window computed for that night's site. Running it again on
an unchanged file is a no-op; running it after the file changes creates or
updates only the affected nights.

.. code-block:: console

   >> python3 manage.py load_telescope_runs path/to/schedule.txt

An optional ``--campaign <name>`` flag associates every ``CalendarEvent`` the
file creates or updates with a named campaign (a ``tom_targets.TargetList``),
matched by exact name. It is genuinely optional: if you omit it, no campaign
association is set on any event -- the same behavior this command had before
the flag existed. The name is resolved once, up front, before any schedule
line is processed, so an unknown or ambiguous campaign name fails
immediately rather than half-way through the file.

.. code-block:: console

   >> python3 manage.py load_telescope_runs path/to/schedule.txt --campaign "3I/ATLAS"

.. note::
   Don't confuse this optional ``--campaign`` with ``import_campaign_csv``'s
   ``--campaign`` (below): here, omitting it means "no campaign"; on
   ``import_campaign_csv`` the flag is **required**.

How do I sync LCO/SOAR queue observations?
---------------------------------------------

``sync_lco_observation_calendar`` syncs LCO and SOAR queue
``ObservationRecord`` rows onto the calendar as one ``CalendarEvent`` per
record, keyed on the LCO portal URL. A record still awaiting placement by
the LCO scheduler becomes a ``[QUEUED]`` scheduling-window banner, unless its
status is already a successful terminal state (for example ``COMPLETED``) --
such a record is never bannered as still queued, even if no placement block
was ever resolved for it. Once the scheduler places it, re-running the
command updates the same event in place to the real placed block times.

The required ``--proposal`` flag accepts:

* a single proposal code, e.g. ``--proposal LCO2026A-001``;
* a comma-separated list of codes, e.g. ``--proposal A,B,C`` (matches only
  those exact codes -- no substring leakage, so ``--proposal A`` never also
  matches a proposal literally named ``AB``);
* the case-insensitive token ``ALL``, which syncs every LCO and SOAR record
  regardless of proposal.

.. code-block:: console

   >> python3 manage.py sync_lco_observation_calendar --proposal LCO2026A-001
   >> python3 manage.py sync_lco_observation_calendar --proposal ALL

How do I backfill ObservationRecords for LCO observations submitted outside FOMO?
------------------------------------------------------------------------------------

``backfill_lco_observation_records`` queries the LCO Observation Portal's "Get
All RequestGroups" API for a proposal, keeps only RequestGroups whose name
starts with ``--name-prefix``, and creates one ``ObservationRecord`` per child
request. A request that already has an ``ObservationRecord`` is skipped, so
the command is safe to re-run.

It exists to create the ObservationRecords for observations submitted
directly at the LCO portal rather than through FOMO -- those records are what
``sync_lco_observation_calendar`` above then projects onto the calendar, so
run this command first, then the sync.

The required ``--proposal <code>`` (exact match) and ``--name-prefix
<string>`` flags select which RequestGroups to backfill.

``--campaign <name>`` is optional here too, but omitting it means something
different from omitting it on ``load_telescope_runs``: each request's target
is matched **by name** against the Targets already belonging to this
campaign, and a request whose target isn't a member is skipped and logged,
never guessed at -- but if ``--campaign`` itself is omitted, the command
prints the available campaigns and prompts for a selection interactively.

``--create-missing-targets`` is opt-in, default off. It changes the
unmatched-target case from "skip" to: reuse an existing Target of that name
if one exists anywhere in FOMO, otherwise build a new SIDEREAL field Target
from the request's own RA/Dec -- carrying across epoch, proper motion, and
parallax when the request supplies them -- then add it to the campaign and
process the request normally. A *reused* Target is left untouched; only
newly built ones get those fields populated. This is the single most
surprising detail of the flag.

``--username <user>`` optionally attributes created records to that user;
default is unattributed. An unknown username is a hard error.

Always run with ``--dry-run`` first to see what would be created --
including which field Targets would be created versus reused -- without
writing anything, in the same spirit as ``backfill_range_calendar_events``
above:

.. code-block:: console

   >> python3 manage.py backfill_lco_observation_records --proposal LCO2026A-001 --name-prefix "3I/ATLAS" --campaign "3I/ATLAS" --dry-run
   >> python3 manage.py backfill_lco_observation_records --proposal LCO2026A-001 --name-prefix "3I/ATLAS" --campaign "3I/ATLAS"

Immediately after each new record is saved (non-dry-run only), the command
makes one live best-effort status call to LCO so the record's status,
``scheduled_start``, and ``scheduled_end`` are populated right away instead
of staying unset until the next poll. If that call fails it is logged and
counted, never fatal, and the already-created record is not rolled back.

The final summary line reports these counters::

   Created: 4, already existed: 12, unmatched target: 1, no usable configuration: 0, created field targets: 1, status sync failed: 0

How do I sync Gemini queue observations?
-------------------------------------------

``sync_gemini_observation_calendar`` syncs every submitted Gemini
Target-of-Opportunity ``ObservationRecord`` (``facility='GEM'``) onto the
calendar, unconditionally.

.. code-block:: console

   >> python3 manage.py sync_gemini_observation_calendar

Unlike the LCO/SOAR sync above, this command has **no proposal or filter
flag at all** -- it always processes every Gemini ``ObservationRecord`` in
the database. If you're used to the ``--proposal`` flag from the LCO
section, do not expect an equivalent here; there is nothing to pass. Each
record's observing window comes from its explicit
``windowDate``/``windowTime``/``windowDuration`` parameters when present,
or is otherwise derived from its Target-of-Opportunity type (a Rapid ToO
gets a 24-hour window from submission; a Standard ToO gets a 24-hour to
7-day window).

How do I mark a run cancelled or weathered-out?
--------------------------------------------------

Once a campaign run is approved, the approval queue's **Decided** table
shows "Mark Cancelled" (``action=mark_cancelled``) and "Mark Weathered"
(``action=mark_weather_failure``) buttons on that row's Actions column
(they appear for any approved run regardless of its current observing
status). Clicking one immediately and publicly prepends
``[CANCELLED]`` or ``[WEATHERED]`` to the title of **every**
``CalendarEvent`` associated with that run -- including every per-night
event of a multi-night range-window run -- on the shared campaign calendar
that anonymous visitors can see. There is no separate confirmation step and
no revert button, but the action is a safe, idempotent no-op to re-click:
clicking the same button again, or clicking the other button to correct a
mis-click, simply re-applies the new prefix without creating duplicate
events or losing any data.

How do I bootstrap-import a campaign from a CSV?
----------------------------------------------------

``import_campaign_csv`` bulk-imports a campaign coordination spreadsheet
(for example, a community campaign's shared observing-run tracking sheet)
into ``CampaignRun`` rows, one row per CSV line.

.. code-block:: console

   >> python3 manage.py import_campaign_csv --campaign "3I/ATLAS" path/to/campaign.csv

.. note::
   **Re-import gotcha:** re-running this command over the same
   ``--campaign`` always resets every row's ``target`` field back to its
   auto-resolved value. If a staff member manually corrected a row's
   ``target`` in the Django admin after a previous import, that correction
   is silently overwritten the next time this command runs over the same
   campaign CSV. This is expected behavior for a bootstrap-import command,
   not a bug -- but it is easy to be surprised by, so re-import
   deliberately, not routinely.

.. note::
   **What the command now writes (CANON-01/CANON-02):** every imported row
   records ``source = csv_import`` and is created ``approved`` -- a
   bootstrap import is vetted backfill, not a community submission awaiting
   review, so approval gating applies to web submissions only. A row whose
   ``Site Code`` does not resolve now also gets a derived
   ``telescope_class`` when its ``Telescope / Instrument`` text names a
   telescope class (``2m0``/``1m0``/``0m4``), or ``SPACE`` when it names a
   space observatory with no MPC code, and stays blank otherwise -- blank
   plus a flagged site (``site_needs_review``) is what a genuine resolution
   failure looks like.

How do I re-resolve campaign run sites that have gone stale?
------------------------------------------------------------------

``repair_stale_campaign_run_sites`` is a one-off command for approved
``CampaignRun`` rows whose site never resolved because they were imported
before the JPL Horizons observer-notation alias table existed (added
2026-07-26). It re-runs the real site-resolution path
(``resolve_site()``) against every approved, site-less row, so a row that
would now resolve (for example, a JWST row whose ``Site Code`` is
``500@-170``) gets a genuine chance to.

It deliberately does not touch ``approval_status``, ``run_status``, the
observing window, or ``target`` -- only ``site``, ``site_needs_review``,
and (for one known stale row) ``site_raw`` are ever written -- and it
never creates or updates a calendar event; reconciling a repaired run onto
the calendar is Phase 29's reconciler.

Always run with ``--dry-run`` first. Its limitation: it only performs a
tier-1 (local ``Observatory``) existence check, so a row that would need a
live tier-2 MPC lookup is reported as "would query MPC" rather than
resolved, and nothing is written either way:

.. code-block:: console

   >> python3 manage.py repair_stale_campaign_run_sites --dry-run
   >> python3 manage.py repair_stale_campaign_run_sites

The real (non-dry-run) run may make a live MPC Obscodes API call for any
row that needs a tier-2 lookup. If the network is unavailable, that row
stays site-less and flagged for review -- no placeholder ``Observatory``
is ever fabricated on a network failure (the command always passes
``create_placeholder=False``). It is safe to re-run: a row that resolves
stays resolved, and a row still lacking a site code is skipped again with
no field changes.

How do I backfill calendar events for older approved range-window runs?
----------------------------------------------------------------------------

``backfill_range_calendar_events`` is a one-off command for a narrow
historical gap: a multi-night range-window ``CampaignRun`` that was already
approved and site-resolved *before* per-night calendar projection existed
never got any ``CalendarEvent`` at all, and normal approval/resolve actions
only project events going forward, not retroactively. This command finds
every already-approved, site-resolved range-window run with no existing
calendar event and projects one per night, exactly as if it had just been
approved.

Always run with ``--dry-run`` first to see which runs would be backfilled,
with no database writes:

.. code-block:: console

   >> python3 manage.py backfill_range_calendar_events --dry-run
   >> python3 manage.py backfill_range_calendar_events

The command is safe to re-run: a run that already has a calendar event is
skipped, so running it again after a real backfill is a no-op.

.. _command-cheat-sheet:

Command cheat-sheet
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Command
     - Key flags
     - One-line description
   * - ``load_telescope_runs``
     - ``<filepath>`` (positional), ``--campaign <name>`` (optional)
     - Ingest a classical-schedule text file into per-night CalendarEvents.
   * - ``sync_lco_observation_calendar``
     - ``--proposal <code|A,B,C|ALL>`` (required)
     - Sync LCO/SOAR queue ObservationRecords to CalendarEvents.
   * - ``backfill_lco_observation_records``
     - ``--proposal <code>``, ``--name-prefix <str>`` (both required); ``--campaign <name>``,
       ``--username <user>``, ``--create-missing-targets``, ``--dry-run`` (optional)
     - Backfill ObservationRecords for LCO RequestGroups submitted outside FOMO.
   * - ``sync_gemini_observation_calendar``
     - (none)
     - Sync every Gemini ToO ObservationRecord to CalendarEvents.
   * - ``import_campaign_csv``
     - ``--campaign <name>`` (required), ``<filepath>`` (positional)
     - Bootstrap-import a campaign coordination CSV into CampaignRun rows.
   * - ``repair_stale_campaign_run_sites``
     - ``--dry-run`` (optional)
     - One-off re-resolution of approved CampaignRuns whose site never resolved.
   * - ``backfill_range_calendar_events``
     - ``--dry-run`` (optional)
     - One-off backfill of CalendarEvents for older approved range-window runs.

Troubleshooting
------------------

These are failure modes that have actually been observed running these
commands against real data -- not a speculative list of every possible
exception. Every example below uses synthetic placeholder names, emails,
and telescope/instrument strings; no real contact information appears
anywhere on this page.

Observatory missing timezone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any command that needs to compute sunset/sunrise or the -15 deg dark
window for a site (``sync_lco_observation_calendar``,
``backfill_range_calendar_events``, and any future projection over that
Observatory) will fail with an error like this, observed running a real
backfill against the dev database:

.. code-block:: console

   Observatory 'FTN' (obscode=F65) has no timezone set

**Fix:** the ``Observatory`` record for that site is missing its
``timezone`` field. Set it to a valid IANA timezone name -- for example
``"America/Santiago"`` -- via the Django admin, or via the
``CreateObservatory`` form, then re-run the sync/backfill command for that
site. Until the field is set, every projection or backfill attempt against
that ``Observatory`` record will keep failing with the same error; it is
not a one-time fluke.

Per-line / per-record skip-and-log behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ingest/sync command follows the same shared invariant: **one bad row
never aborts the whole run.** A problem with a single line or record is
logged and skipped, and the command continues to the end, reporting a
summary count.

* ``load_telescope_runs`` skips and logs any schedule line it cannot parse,
  or whose telescope name doesn't resolve to a known ``Observatory``
  (a caught ``ValueError``/``Observatory.DoesNotExist``), and reports a
  ``skipped: N`` count in its final summary line, e.g.::

      Line 12: Observatory 'XYZ' (obscode=???) has no timezone set (line text: 'XYZ Instrument 1-5 July')
      Done. lines processed: 20, created: 95, updated: 0, unchanged: 0, skipped: 1

* ``sync_lco_observation_calendar`` falls back to a coarse, clearly-labelled
  ``[UNVERIFIED]`` telescope name (instead of skipping the record) when its
  per-record live telescope-label API call times out or returns an
  unmapped site/telescope code. This is tracked as its own
  ``telescope_api_failed`` counter, separate from ``skipped``, and the
  record still gets a ``CalendarEvent``.

* ``backfill_range_calendar_events`` skips a candidate run on a
  ``ValueError`` (for example, the Observatory-timezone gap above) and
  continues to the next candidate, never aborting the whole backfill.

``import_campaign_csv`` unresolved rows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A row whose ``Site Code`` cell doesn't resolve to a known ``Observatory``,
or whose ``Obs. Date`` cell doesn't parse into a concrete window, is never
silently dropped. Instead, the row still imports (or updates) as a
``CampaignRun``, flagged with ``site_needs_review`` and/or
``window_needs_review``, and both counts appear in the command's final
summary line, e.g.::

   Done. created: 12, updated: 3, unchanged: 40, skipped: 1, site_needs_review: 2, window_needs_review: 1

Rows flagged ``site_needs_review`` surface in the approval queue's "Sites
Needing Review" card so staff can resolve them without re-running the
import.

If a previously-unresolvable ``Site Code`` has since become resolvable
(for example, a Horizons observer-notation code added to the alias table
after the row was imported), see "How do I re-resolve campaign run sites
that have gone stale?" above -- ``repair_stale_campaign_run_sites`` re-runs
site resolution for every approved, site-less row without re-importing the
whole CSV.

Also recall the re-import ``target``-reset gotcha covered above under "How
do I bootstrap-import a campaign from a CSV?": re-running
``import_campaign_csv`` over the same ``--campaign`` always resets every
row's ``target`` back to its auto-resolved value, silently overwriting any
manual correction made since the previous import.

See also
-----------

* The :ref:`command cheat-sheet <command-cheat-sheet>` above for exact flag
  syntax.
* :doc:`/design/telescope_runs_calendar` for the astronomy and data-model
  rationale behind these commands.
