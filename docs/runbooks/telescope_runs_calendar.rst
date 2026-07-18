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

How do I sync LCO/SOAR queue observations?
---------------------------------------------

``sync_lco_observation_calendar`` syncs LCO and SOAR queue
``ObservationRecord`` rows onto the calendar as one ``CalendarEvent`` per
record, keyed on the LCO portal URL. A record still awaiting placement by
the LCO scheduler becomes a ``[QUEUED]`` scheduling-window banner; once the
scheduler places it, re-running the command updates the same event in
place to the real placed block times.

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

Command cheat-sheet
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Command
     - Key flags
     - One-line description
   * - ``load_telescope_runs``
     - ``<filepath>`` (positional)
     - Ingest a classical-schedule text file into per-night CalendarEvents.
   * - ``sync_lco_observation_calendar``
     - ``--proposal <code|A,B,C|ALL>`` (required)
     - Sync LCO/SOAR queue ObservationRecords to CalendarEvents.
   * - ``sync_gemini_observation_calendar``
     - (none)
     - Sync every Gemini ToO ObservationRecord to CalendarEvents.
   * - ``import_campaign_csv``
     - ``--campaign <name>`` (required), ``<filepath>`` (positional)
     - Bootstrap-import a campaign coordination CSV into CampaignRun rows.
   * - ``backfill_range_calendar_events``
     - ``--dry-run`` (optional)
     - One-off backfill of CalendarEvents for older approved range-window runs.
