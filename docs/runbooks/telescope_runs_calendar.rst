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

How do I reach the approval queue?
---------------------------------------

The approval queue (``campaigns:approval_queue``) hosts **two independent
work queues**, not one:

* **Pending Review** -- public submissions awaiting a staff approve/reject
  decision.
* **Sites Needing Review — action required** -- approved runs whose
  observing site never resolved and for which no ``telescope_class``
  explains the absence (quote the card heading verbatim so it's easy to
  match while scanning the page).

The entry point is the warning banner at the top of ``/campaigns/``,
visible to staff only. As of this phase, it appears whenever **either**
queue has rows, and names each count separately -- for example "3
submissions pending review" and "2 runs needing site review" together, or
either sentence alone if only one queue has rows.

**Behavior change:** before this phase, the banner was driven by the
pending-review count alone. With zero pending submissions -- the normal
steady state -- there was no link to the approval queue at all, even when
the Sites Needing Review queue was full of actionable rows. If you
remember the old all-or-nothing banner, this is the fix: either queue
having rows is now enough to show the banner and its "Review queue" link.

When both queues are empty, the banner does not appear at all; the page
is still reachable directly by URL.

See "``import_campaign_csv`` unresolved rows" below for *why* a row lands
in the Sites Needing Review queue in the first place, and "How do I
re-resolve campaign run sites that have gone stale?" above for the bulk
alternative to resolving rows one at a time from this page.

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

Can I correct a run's source?
----------------------------------

A run that came in through the public submission form (``source = web``)
has no editable ``source`` in the Django admin. The field is not rendered
on its change page **at any approval status** -- pending, approved or
rejected alike.

Why: ``source = web`` combined with ``approval_status`` is the only record
that a human reviewed a public submission. An approved run whose source is
not ``web`` reads as "no approval was required" -- a different fact -- and
nothing on the run stores the old value, so overwriting it cannot be undone
or reconstructed.

What this closes: it used to be possible to open a ``web`` run while it was
still pending, change its ``source`` there (the admin allowed it), and then
approve it. That sequence reached the same lost-provenance state as editing
an already-approved run, just by a longer route.

Every other run keeps an editable ``source``: ``legacy``, ``csv_import``
and the queue sources can all still be corrected in the admin, which is
what that editability was for -- a ``web`` label is never a guess, because
only the submission form can produce it.

**The cost:** if a ``web`` run's source really is wrong, correcting it now
needs a shell or a data migration. This is the same restriction the CSV
re-import path already applies -- see the re-import gotcha note below.

**What stays possible:** the rule looks at the run's current source, so a
non-``web`` run can still be relabelled *to* ``web``, and it locks once
saved. That direction invents a review rather than erasing one, it takes a
deliberate act, and the Django admin's own history log records who changed
the field and when -- so it is visible after the fact, unlike the
direction that was closed.

.. warning::
   **That relabel cannot be taken back from the admin.** The moment you save
   a ``legacy`` or ``csv_import`` run as ``web``, the rule above starts
   applying to it and ``source`` disappears from its change page -- so you
   cannot correct your own mis-click here, only through a shell or a data
   migration. Re-importing the CSV that produced the row will not fix it
   either: ``import_campaign_csv`` leaves ``source`` and ``approval_status``
   alone on any row that already reads ``web``. If the row was also
   ``approved``, it now reads permanently as "a human approved this public
   submission", and the admin history records only that ``source`` changed,
   not what it changed *from*. Treat the ``source`` dropdown on a non-``web``
   run as a one-way door.

Creating a new run in the admin is unaffected: ``source`` is editable on
the add form, so a run can still be created with any source.

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
   campaign CSV.

   The same is true of ``source`` and ``approval_status``: a re-import
   applies ``source = csv_import`` and ``approval_status = approved`` to an
   already-existing row, not just to a newly created one. The one exception
   is a row that came in through the public submission form (``source =
   web``) -- such a row keeps its own ``source`` **and**
   ``approval_status``, so a re-import can never turn an unreviewed public
   submission into something that reads as vetted, publicly-visible
   backfill. Every one of its other fields is still overwritten from the
   CSV. The Django admin applies the same rule -- see "Can I correct a
   run's source?" above.

   All of this is expected behavior for a bootstrap-import command, not a
   bug -- but it is easy to be surprised by, so re-import deliberately, not
   routinely.

   **Site preservation (Phase 27.1, WR-01):** the exception above no longer
   stops at ``source``/``approval_status``. A row whose ``site`` is already
   resolved keeps its ``site``, ``site_raw`` **and** ``site_needs_review``
   when the CSV's own ``Site Code`` cell does not resolve this time (a blank
   cell, or one that only reaches the tier-3 placeholder path) -- so
   re-importing after "How do I re-resolve campaign run sites that have gone
   stale?" above (``repair_stale_campaign_run_sites``) can no longer silently
   revert that repair. A ``Site Code`` cell that *does* genuinely resolve
   still wins, so correcting a wrong code in the sheet and re-importing still
   moves the site as before. The accepted cost: a site can no longer be
   *cleared* through a re-import -- clearing one now requires the Django
   admin or a shell. A non-blank ``telescope_class`` is likewise never
   blanked by a re-import, consistent with the "it is **permanent**: it is
   never cleared by any command" sentence in the note below -- before this
   phase the importer *did* blank it whenever the site resolved, which this
   guard corrects. The ``site_needs_review`` count in the command's summary
   line now reports only flags the command actually wrote, so a preserved
   row no longer inflates it.

.. note::
   **What the command now writes (CANON-01/CANON-02):** every imported row
   records ``source = csv_import`` and is created ``approved`` -- a
   bootstrap import is vetted backfill, not a community submission awaiting
   review, so approval gating applies to web submissions only. On a
   *re-import* those same two values are re-applied to an already-existing
   row, except for a ``source = web`` row (see the re-import gotcha above).
   A row whose
   ``Site Code`` does not resolve now also gets a derived
   ``telescope_class`` when its ``Telescope / Instrument`` text names a
   telescope class (``2m0``/``1m0``/``0m4``), or ``SPACE`` when it names a
   space observatory with no MPC code, and stays blank otherwise. A row that
   gets a derived ``telescope_class`` is **deliberately NOT flagged** for
   site review -- the class is the answer to "why is there no site", not a
   resolution failure, and it is **permanent**: it is never cleared by any
   command, even if a site is later resolved for the same row. Only a row
   with no site *and* no derivable class is flagged (``site_needs_review``)
   -- that combination is what a genuine resolution failure looks like.

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

A candidate row that already carries a ``telescope_class`` is skipped
entirely -- its site, ``site_raw``, and ``site_needs_review`` are all left
untouched, and it is reported under its own ``skipped_class_wide`` counter
in the summary line. A class-carrying row is permanently site-less by
design (the class already answers "why is there no site"), so there is
nothing for this command to repair.

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

.. _campaign-run-block-manual-only:

Why doesn't the calendar pop-up show a "Campaign run" block?
----------------------------------------------------------------

Clicking a calendar entry opens a pop-up that can show a **Campaign run**
block naming the run that owns the event, its window, and its run status.
That block appears only when the event carries a companion record whose
owning-run link is filled in -- and **nothing in FOMO fills that link in
automatically yet.**

Approving a ``CampaignRun`` (or resolving its site from the approval queue)
creates the calendar entries for it, but deliberately does not claim
ownership of them. Writing the ownership link is the job of the
reconciler planned for a later milestone, which is also what will keep the
link correct as runs are re-approved, re-sited, or cancelled. Until then,
the only way an event gets a "Campaign run" block is if a staff member
links it by hand:

1. Go to **Django admin -> Solsys code -> Campaign runs** and open the run.
2. In the **Calendar event metas** inline at the bottom, add a row and pick
   the calendar event that belongs to this run.
3. Save. The pop-up for that event now shows the Campaign run block.

Two things to know about that inline:

* The **calendar event** field is frozen once a row is saved, because it is
  that record's identity. To point the link at a different event, delete
  the row and add a new one -- do not try to edit it in place.
* Clearing the **owning campaign run** value un-owns the event without
  deleting the companion record, so the event's telescope-label
  verification history survives.

An event with no companion record at all, or with the run link left blank,
means "not owned by any campaign run" -- never "needs fixing". That is the
normal state for classical-schedule nights, conferences, and proposal
deadlines, and it is why those entries show no Campaign run block.

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
import -- see the "reach the approval queue" section above for how staff
get there, including the case where zero submissions are pending review.
Only rows with no derivable ``telescope_class`` signal surface
there -- a row whose site failed to resolve but whose instrument text
names a telescope class or a space observatory is not a genuine resolution
failure, so it never appears in this queue and there is nothing to
resolve for it. Per-site detail for a class-wide campaign (e.g. a
multi-site LCO 1m0 network allocation) arrives later, per observation, on
the linked ``ObservationRecord`` rows (CANON-04) -- never by resolving the
run itself to a single site.

If a previously-unresolvable ``Site Code`` has since become resolvable
(for example, a Horizons observer-notation code added to the alias table
after the row was imported), see "How do I re-resolve campaign run sites
that have gone stale?" above -- ``repair_stale_campaign_run_sites`` re-runs
site resolution for every approved, site-less row without re-importing the
whole CSV.

Also recall the re-import reset gotcha covered above under "How do I
bootstrap-import a campaign from a CSV?": re-running ``import_campaign_csv``
over the same ``--campaign`` always resets every row's ``target`` back to
its auto-resolved value, and re-applies ``source = csv_import`` and
``approval_status = approved``, silently overwriting any manual correction
made since the previous import. Rows created by the public submission form
(``source = web``) keep their own ``source`` and ``approval_status``. This
reset does **not** extend to ``site``/``site_raw``/``site_needs_review`` or
``telescope_class``, though: as of Phase 27.1, a row whose site is already
resolved keeps it (and its ``telescope_class``, if any) across a re-import
whose ``Site Code`` cell does not itself resolve -- see "Site preservation"
in the re-import gotcha note above.

See also
-----------

* The :ref:`command cheat-sheet <command-cheat-sheet>` above for exact flag
  syntax.
* :doc:`/design/telescope_runs_calendar` for the astronomy and data-model
  rationale behind these commands.
