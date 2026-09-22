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
run per line, e.g. ``NTT EFOSC2 allocation 9-13 July`` -- and creates or
updates one ``CampaignRun`` per line (``source=CLASSICAL_FILE``). It no
longer writes any ``CalendarEvent`` itself: the allocation projector draws
the same per-night calendar from that run, with sunset/sunrise and the -15
deg dark window computed for that night's site. Running it again on an
unchanged file is a no-op; running it after the file changes creates or
updates only the affected run and lets the projector re-derive the
affected nights.

.. code-block:: console

   >> python3 manage.py load_telescope_runs path/to/schedule.txt

**The calendar entries look exactly as they did before this change** -- same
titles (``NTT EFOSC2``), same sunset-to-sunrise spans, same -15 deg dark
window line in the description. The one visible difference: a cancelled
line's event description now also carries the shared writer's
``Run status: Cancelled`` line, because every allocation description is
composed through the same helper every other campaign-run event uses.

A schedule line is matched to its ``CampaignRun`` by a deterministic key
built from the resolved telescope, instrument, the run's own stored
observing-night window, and its two sub-night tokens -- so re-importing the
same file recomputes the same key and updates the same row, never creating
a duplicate.

**The optional bracketed proposal token.** Two proposals can otherwise
share a telescope, an instrument and a set of nights and be genuinely
indistinguishable -- add a bracketed token naming the proposal to
disambiguate them:

.. code-block:: console

   NTT EFOSC2 allocation 9-13 July [0110.C-0234]

Two lines identical except for their proposal token produce two distinct
``CampaignRun`` rows. Two lines that are otherwise identical and both omit
the token collide: the second is reported on stderr naming both line
numbers and is skipped, never silently merged into the first. **Fix:** add
a proposal token to one of the two lines (or both, if they genuinely are
two different proposals), then re-import.

**The two summary lines a real run prints.** A run-level line reports the
line-by-line tallies (``lines processed``, ``created``/``updated``/
``unchanged``/``skipped``/``skipped_collision``), followed by a
night-level line aggregating what the allocation projector did across
every line's own run (``created``/``updated``/``unchanged``/``retired``/
``rekeyed``/``blocked``/``skipped``)::

   Done. lines processed: 20, created: 19, updated: 0, unchanged: 0, skipped: 1, skipped_collision: 0
   Done. nights -- created: 95, updated: 0, unchanged: 0, retired: 0, rekeyed: 0, blocked: 0, skipped: 0

``--dry-run`` previews both lines without writing anything. For a run that
already exists, the night-level preview comes from the same reconciler
preview every staff action uses; for a brand-new run, a first-time dry run
predicts night counts from the window length rather than computing a real
sun-event time:

.. code-block:: console

   >> python3 manage.py load_telescope_runs path/to/schedule.txt --dry-run

The run-level line's four counters state an invariant that holds on both
passes no matter which arm a line takes: ``created`` + ``updated`` +
``unchanged`` + ``skipped`` equals ``lines processed``. The two arms
reach that invariant differently, and only one of them lets the preview
predict what the real pass will report. For a line whose run already
exists, the preview runs the same reconcile the real pass runs and folds
its own ``created``/``updated``/``unchanged`` counter only after that
call returns -- so the preview and the real pass report the same
decision, and a line whose preview reconcile raises is reported under
``skipped`` alone on both passes, never counted under ``unchanged`` and
``skipped`` at once. For a brand-new line there is no preview reconcile
at all -- the note above already says a first-time dry run predicts
night counts from the window length rather than a sun-event computation
-- so the preview cannot predict a reconcile failure that the real
pass's own call can still raise: a line the real pass drops under
``skipped`` can still preview as ``created``.

An optional ``--campaign <name>`` flag associates every ``CampaignRun`` the
file creates or updates with a named campaign (a ``tom_targets.TargetList``),
matched by exact name. It is genuinely optional: if you omit it, no campaign
association is set on any run -- the same behavior this command had before
the flag existed. The name is resolved once, up front, before any schedule
line is processed, so an unknown or ambiguous campaign name fails
immediately rather than half-way through the file.

.. code-block:: console

   >> python3 manage.py load_telescope_runs path/to/schedule.txt --campaign "3I/ATLAS"

.. note::
   Don't confuse this optional ``--campaign`` with ``import_campaign_csv``'s
   ``--campaign`` (below): here, omitting it means "no campaign"; on
   ``import_campaign_csv`` the flag is **required**.

How do LCO/SOAR queue observations get onto the calendar?
-------------------------------------------------------------

**They get there by themselves.** Saving an ``ObservationRecord`` -- whether
FOMO submitted it, TOM's ``updatestatus`` refreshed it from the LCO portal,
or ``backfill_lco_observation_records`` created it -- draws or updates that
record's own ``CalendarEvent``, keyed on its portal URL, with no operator
command. This is the observation projector (``solsys_code/observation_projector.py``):
a Django ``post_save`` signal receiver connected for every LCO and SOAR
``ObservationRecord``, live since Phase 34.

The event narrows as the record's own fields change:

* the submitted request window while the record is still queued;
* the real placed block once the LCO scheduler places it;
* the observed block once it is actually observed;
* a marked event on the original window if the record expires, is
  cancelled, or fails.

The event's title carries a compact marker naming that stage, e.g.
``[Q] 2m0 3I/ATLAS``. **One module, ``solsys_code/status_vocabulary.py``,
defines every marker below, its label and the calendar legend** -- the
entry titles, the calendar legend and the campaign table's Progress cell
(see "What does a run's or a campaign's public tally show?" below) all
read from that single definition, so there is exactly one spelling of
every state, final as of this phase:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Meaning
   * - ``[Q]``
     - Queued -- awaiting placement by the LCO scheduler.
   * - ``[S]``
     - Scheduled -- the LCO portal's own word for a block the scheduler has
       placed but that has not yet been observed.
   * - ``[O]``
     - Observed -- a successful terminal status.
   * - ``[X]``
     - Window expired before the observation was attempted.
   * - ``[C]``
     - Cancelled -- by whichever layer owns the entry: a staff decision on
       a campaign run (the approval queue's "Mark Cancelled" button, see
       "How do I mark a run cancelled or weathered-out?" below) or a
       portal cancellation on an LCO/SOAR observation record. ``[C]``
       itself never says which -- the pop-up's own "Run status:" line is
       what distinguishes the two.
   * - ``[F]``
     - Failed (failure limit reached, or never attempted).
   * - ``[W]``
     - Weather or technical failure -- a staff decision on a campaign run
       (the approval queue's "Mark Weathered" button, see "How do I mark a
       run cancelled or weathered-out?" below).
   * - ``[?]``
     - Inconsistent record -- only one of ``scheduled_start``/
       ``scheduled_end`` is set; projected anyway so the data problem is
       visible on the calendar rather than only in a log. A failure marker
       always wins over ``[?]`` (``title_for()``), so a record that is both
       inconsistent *and* window-expired/cancelled/failed shows ``[X]``/
       ``[C]``/``[F]`` instead -- the data problem is then visible only in
       a log, not on the calendar, for that combination.
   * - ``[U]``
     - Unused awarded night -- an allocation night that came and went with
       nothing scheduled or observed on it. Never stored in a title: it is
       added only when the calendar page is rendered, on a muted chip, so
       an operator reading a stored ``CalendarEvent.title`` directly (in
       the admin, say) never sees it. See "What does a run's or a
       campaign's public tally show?" below.

You do not need to come back to this runbook to decode a marker on the
calendar page itself: the calendar carries its own status **legend** row,
listing every marker above beside its meaning, so an operator reading a
month cell can decode it at a glance. The ring drawn around a month cell
follows the same vocabulary: a Queued or an Inconsistent record entry is
ringed, a Scheduled or Observed entry is not, and an expired, cancelled,
failed or weathered entry carries the terminal ring. The unused marker
(``[U]``) adds no ring of its own -- it is shown only by that muted chip
and its ``[U]`` text token, precisely so the signal never depends on
colour alone.

Observation series
^^^^^^^^^^^^^^^^^^^^^

For a record belonging to an observation group of two or more, clicking its
calendar entry's pop-up shows an **Observation series** block: the group's
name, which night of how many this entry is, and links back to the group
and to the record's own detail page. Exactly like the "Attributed campaign
run" block described below, this numbering is read from the entry's
companion row (``CalendarEventMeta.observation_group``) every time the page
is drawn -- it is never written into the entry's own title or description,
so re-projecting an entry cannot erase it, and adding or removing a night
never rewrites a sibling's stored title. Nights are numbered by window
start, so a cadence reads in observing order rather than in the order the
entries happen to have been created. An entry with no group, or belonging
to a group of one, simply shows no block. **This block requires logging in,
full stop:** an anonymous (not logged in) visitor never sees it, whether or
not the entry has been attributed to a campaign run, and regardless of that
run's own review status -- because the group's own name is drawn from an
internal LCO Observation Portal ``RequestGroup`` identifier that must not be
published to the public calendar under any circumstance. Logging in shows
the block for an entry with no run and for an entry attributed to an
**approved** run; an entry attributed to a run still pending review stays
hidden even from a logged-in visitor, exactly as that run's own "Attributed
campaign run" block does, until the run clears review.

**One-time title change.** The first sweep of ``project_observation_calendar``
after Phase 34 updates, once, the titles and companion-row links of the
legacy LCO/SOAR calendar entries an earlier sync had already created,
because they still carry that older stopgap wording. This is expected, not
a fault. What survives it: entries keyed on a campaign run, on a Gemini
submission, or on nothing at all are left exactly as they were -- the
projector only ever touches its own facility-URL-keyed events. Every field
the sweep rewrites is re-derived from the observation record itself, so a
later title-wording change plus one more sweep simply re-derives those
titles again.

When would I run the sweep?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``post_save`` receiver above covers ``ObservationRecord.save()``, but it
cannot see a write path that bypasses ``save()`` entirely --
``QuerySet.update()``, ``bulk_create()``, and anything editing rows outside
Django. ``project_observation_calendar`` is the backstop sweep for exactly
those paths, plus the one-time observed-telescope lookup for a newly
observed record (a single live portal call per record, ever). It needs no
arguments -- omitting every flag sweeps every LCO/SOAR record:

.. code-block:: console

   >> python3 manage.py project_observation_calendar
   >> python3 manage.py project_observation_calendar --dry-run

Optional flags: ``--proposal <code|A,B,C>`` restricts the sweep to one or
more exact proposal codes (comma-separated, no substring matching);
``--facility <LCO|SOAR>`` restricts it to one facility; ``--dry-run``
reports what would change without writing anything. ``--proposal`` fails
closed: a value that parses to no usable code at all (e.g. ``--proposal ','``
or a string of nothing but commas/spaces) raises an error rather than
silently widening the sweep to every record in scope -- the opposite of what
naming ``--proposal`` is asking for.

The final summary line reports these counters per facility::

   Done. failed: 0 | LCO: created: 3, updated: 156, unchanged: 0, unprojectable: 0, site_lookups: 59, site_lookup_failed: 1 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0

``created``/``updated``/``unchanged`` are the events the sweep drew, refreshed,
or left alone; ``unprojectable`` counts a record the sweep could not project
at all -- either because its own fields could not be turned into event
values at all (for example, an unparsable request window), or because the
write itself failed once attempted (for example, a pre-existing duplicate
calendar-event URL). Either way the record's event is left exactly as it
was before the sweep touched it, and the top-level ``failed`` count (and a
matching stderr line naming the ``observation_id``) tracks 1:1 with every
row counted ``unprojectable``; ``site_lookups`` counts a successful
one-time observed-telescope lookup; ``site_lookup_failed`` counts a lookup
that has not yet succeeded, retried automatically on the next sweep.

A ``--dry-run`` pass agrees with a real sweep's counts for every field derived
from a record's own already-stored state, with two exceptions. First: a dry
run never performs the one-time observed-site lookup, so its ``site_lookups``
is always 0, and a record whose only pending change is the coarse-to-observed
telescope token (e.g. ``2m0`` to ``FTN``) is reported ``unchanged`` by
``--dry-run`` but ``updated`` by the real sweep that follows it. Second:
``--dry-run`` can only predict a write failure it can detect *without*
writing -- today, a pre-existing duplicate calendar-event URL, which it
counts as ``unprojectable`` just like a real sweep would. A write failure
that only manifests at write time (for example, a database-level error from
an over-length field) has no way to be seen in advance, so
``--dry-run``'s ``unprojectable`` count is a lower bound on what the real
sweep that follows it will report -- it may find more failures than the dry
run predicted, but never fewer.

How do I backfill ObservationRecords for LCO observations submitted outside FOMO?
------------------------------------------------------------------------------------

``backfill_lco_observation_records`` queries the LCO Observation Portal's "Get
All RequestGroups" API for a proposal, keeps only RequestGroups whose name
starts with ``--name-prefix``, and creates one ``ObservationRecord`` per child
request. A request that already has an ``ObservationRecord`` is skipped, so
the command is safe to re-run.

It exists to create the ObservationRecords for observations submitted
directly at the LCO portal rather than through FOMO -- each created
record's own ``save()`` draws its own calendar event automatically (see
"How do LCO/SOAR queue observations get onto the calendar?" above), no
separate sync step needed. It is still worth running
``project_observation_calendar`` afterwards, though: this command links a
multi-request RequestGroup into its own ``ObservationGroup`` *after* each
member record's own save, and the calendar's "Observation series"
decoration and any newly-observed record's telescope label both depend on
that later step (a sweep), not the initial save.

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
writing anything, in the same spirit as ``reconcile_campaign_runs`` below:

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

How do I backfill ObservationRecords without a campaign?
------------------------------------------------------------

``backfill_lco_observations`` is the newer, campaign-agnostic sibling of
``backfill_lco_observation_records`` above -- same LCO Observation Portal
"Get All RequestGroups" source, different contract. Read this section
carefully before choosing between the two; running the wrong one for your
situation is an easy mistake.

**How it differs from** ``backfill_lco_observation_records``:

* **No campaign required.** There is no ``--campaign`` flag and no
  ``--name-prefix`` flag -- every ``RequestGroup`` for the given
  ``--proposal`` is considered, not just ones whose name matches a prefix.
* **Re-running updates in place instead of skipping.** A request that
  already has an ``ObservationRecord`` has its ``status``,
  ``scheduled_start``, ``scheduled_end`` and ``parameters`` refreshed from
  the portal -- ``backfill_lco_observation_records`` skips it entirely once
  created.
* **Unmatched targets are always built as non-sidereal**, from the
  request's own orbital elements -- never a sidereal field ``Target`` from
  RA/Dec, and there is no ``--create-missing-targets`` flag to opt in or
  out of it; this is always the behavior. A request whose target can't be
  matched or built (missing required orbital elements for its scheme, or
  no named target at all) is skipped and the reason is printed to stderr,
  never silently dropped.
* **Multi-request RequestGroups are linked into an** ``ObservationGroup``.
  A ``RequestGroup`` carrying more than one request gets one reusable
  ``ObservationGroup`` linking every record built from it; re-running finds
  and reuses the same group rather than creating a second one. A
  single-request ``RequestGroup`` gets no group at all.
* **Every touched Target is collected into a TargetList.** Every target the
  sweep touches -- matched by fuzzy name or newly built from orbital
  elements -- is added to a ``TargetList`` named ``<proposal>_targets``,
  created on the first run and reused on every re-run; re-runs never
  duplicate a membership. A skipped request contributes nothing to the
  list. There is no way to opt out of the collection.

**--proposal is now optional.** Given, this behaves exactly as documented
above -- a one-off manual sweep of that single code, which does not have
to be an admin-editable watched proposal (see below). Omitted, the command
instead sweeps every active **Watched proposal** row (Django admin ->
Solsys code -> Watched proposals) in proposal-code order, applying each
row's own ``TargetList`` name override and attributed-to user, and this is
the invocation the unattended runner uses -- see
:ref:`unattended-operation` below for the schedule this runs on and the
two failure signals if it stops working. An empty watched list is a quiet
no-op: the command logs one line, writes nothing, and exits 0. Each
watched proposal is swept in its own try block: a portal or data error on
one is caught, recorded on that row's own ``Last sweep summary`` (naming the
exception's class only, never its message), and counted, while every
other row still runs; the command exits non-zero at the end and names the
failing proposal code(s) only if at least one row failed.

.. code-block:: console

   >> python3 manage.py backfill_lco_observations

These four flags apply to the single-proposal override only; combined with the bare
invocation the command now fails with a ``CommandError`` rather than silently
discarding them, because the watched-list sweep takes its overrides from each
``WatchedProposal`` row (WR-12, 36-REVIEW.md).

**Date filtering instead of a name prefix.** ``--created-after`` and
``--created-before`` (ISO-8601 timestamps or bare dates) restrict the
backfill to ``RequestGroup``\\ s created in that window -- sent to the
portal as query parameters, and re-checked client-side against each
``RequestGroup``'s own ``created`` timestamp, so the restriction still
holds even if the portal ignores the query parameters.

``--username <user>`` optionally attributes created/updated records to
that user; default is unattributed. An unknown username is a hard error,
same as the sibling command.

``--target-list <NAME>`` overrides the derived ``<proposal>_targets`` name
for the ``TargetList`` the sweep collects into; the derived name is then
never created. There is no way to opt out of the collection itself.

Always run with ``--dry-run`` first -- it reports every decision (which
targets would be built vs. reused, which records would be created vs.
updated, which groups would be created vs. reused) without writing
anything, and skips the live observed-block lookup described below
entirely. A dry run's counts are what the following real pass over the
same portal payload will report -- the one honest caveat being that a
request needing the live fallback lookup (no embedded ``observations``
block) has its schedule compared by a real run but not by a dry run, so
such a record can be reported ``unchanged`` by a dry run when only its
schedule times would actually move. A dry run also reports which
``TargetList`` it would create or reuse and how many targets it would
add, without creating the list -- so an operator can see a name collision
with an existing list before anything is written:

.. code-block:: console

   >> python3 manage.py backfill_lco_observations --proposal LCO2026A-001 --dry-run
   >> python3 manage.py backfill_lco_observations --proposal LCO2026A-001
   >> python3 manage.py backfill_lco_observations --proposal LCO2026A-001 --created-after 2026-06-01 --created-before 2026-07-01

**Scheduled times.** Unlike the sibling command's post-create live status
call, this command resolves each request's observed block from an embedded
block list on the RequestGroup payload when the portal supplies one, or
otherwise falls back to a live, best-effort
``LCOFacility.get_observation_status()`` call per request (skipped
entirely under ``--dry-run``). A failed fallback lookup is logged and
counted under ``block lookups failed``, never fatal -- the record is still
created or updated with whatever status the request payload itself
reported, just without resolved schedule times.

The summary also reports ``embedded blocks`` and ``fallback lookups
needed`` -- how many requests in this run carried an embedded
``observations`` block versus how many would need (or, on a real run,
used) the live per-request fallback lookup. Both counters are populated in
both modes, so a dry run alone tells an operator which schedule path the
portal actually exercises for a given proposal, without making a single
network call beyond the initial ``RequestGroup`` listing.

The final summary line reports these counters. A real pass::

   requestgroups seen: 6, created: 4, updated: 8, unchanged: 3, skipped: 1, targets created: 2, groups created: 1, groups reused: 2, embedded blocks: 5, fallback lookups needed: 9, block lookups failed: 0, target list: created 'LCO2026A-001_targets', targets added to list: 11

A ``--dry-run`` pass over the same proposal -- same counts, would-forms,
and ``block lookups failed`` reported as not applicable since the live
fallback lookup that would produce it is skipped entirely::

   requestgroups seen: 6, would create: 4, would update: 8, unchanged: 3, skipped: 1, targets would create: 2, groups would create: 1, groups would reuse: 2, embedded blocks: 5, fallback lookups needed: 9, block lookups failed: n/a (dry-run), target list: would reuse 'LCO2026A-001_targets', targets would add to list: 11

**Campaign-surface consequence.** A ``TargetList`` is also what FOMO's
campaign surfaces treat as a campaign, so a backfill-created list shows up
in the campaign picker on the run submission form even though it has no
runs. It does **not** appear on the campaign list page, which only shows
lists that have at least one campaign run.

How do I sync Gemini queue observations?
-------------------------------------------

``sync_gemini_observation_calendar`` syncs every submitted Gemini
Target-of-Opportunity ``ObservationRecord`` (``facility='GEM'``) onto the
calendar, unconditionally.

.. code-block:: console

   >> python3 manage.py sync_gemini_observation_calendar

Unlike LCO/SOAR above, this command has **no proposal or filter flag at
all** -- it always processes every Gemini ``ObservationRecord`` in the
database. If you're used to the ``--proposal`` flag from the LCO section,
do not expect an equivalent here; there is nothing to pass. Each record's
observing window comes from its explicit
``windowDate``/``windowTime``/``windowDuration`` parameters when present,
or is otherwise derived from its Target-of-Opportunity type (a Rapid ToO
gets a 24-hour window from submission; a Standard ToO gets a 24-hour to
7-day window).

**Gemini has no observation-status or observation-URL read-back.** The
Gemini facility class provides no way to ask the portal what actually
happened to a submitted observation, unlike LCO/SOAR (see above). A Gemini
calendar event is therefore a submission-echo only: it shows what was
requested, and it never narrows to a placed or observed block the way an
LCO or SOAR event does, however many times this command is re-run. The
automatic observation projector deliberately ignores Gemini records for
exactly this reason -- there is nothing for it to read back and narrow.
This command is, and stays, the only way a Gemini observation reaches the
calendar. This is an operating limitation to keep in mind when reading a
Gemini entry, not a defect in this command.

How do I reach the approval queue?
---------------------------------------

The approval queue (``campaigns:approval_queue``) hosts **two independent
work queues**, not one:

* **Sites Needing Review — action required** -- approved runs whose
  observing site never resolved and for which no ``telescope_class``
  explains the absence (quote the card heading verbatim so it's easy to
  match while scanning the page).
* **Pending Review** -- public submissions awaiting a staff approve/reject
  decision.

Sites Needing Review now renders first on the page -- it is the only
queue that's actionable when no submissions are pending -- followed by
Pending Review and then Recently Decided (27-UAT.md Test 8 gap closure).

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

How do I attribute existing calendar events and observation records to a run?
--------------------------------------------------------------------------------

The attribution page (``campaigns:attribution``, at ``/campaigns/attribution/``)
is where staff connect a calendar event or an observation record that
already exists to the ``CampaignRun`` that actually produced it. It sits
alongside the approval queue as a second staff decision surface: the
campaign-list warning banner now names a third count -- "N orphans
awaiting attribution" -- with its own "Attribution queue" link, following
the same nested-``{% if %}`` staff-only rule the pending/site-review counts
already use.

**The two worklists, and why an orphan may be absent.** The page lists
"Calendar events awaiting attribution" and "Observation records awaiting
attribution" as two sibling tables. Only an event or record with *at
least one* candidate run appears in either one -- the same campaign/target
boundary check that keeps a suggestion from ever crossing into the wrong
campaign also filters out the noise. A conference or proposal-deadline
calendar event has no campaign at all, so it produces no candidate and
never shows up here. **The queue shows attributable orphans, not every
un-attributed row** -- an empty worklist does not mean nothing is
un-attributed, only that nothing un-attributed has a run to offer it to.

**A rejected run is never offered as a match.** A run a staff member
rejected in the approval queue is never suggested for any orphan, on
either worklist -- an approved run and a still-pending submission both
remain offerable (a pending submission being suggested is useful evidence
that the submission is genuine), only a rejected one is excluded. The
filter is applied once, where candidates are chosen, so the attribution
queue, the campaign-list "N orphans awaiting attribution" banner count
and the calendar modal's staff hint all agree with each other -- none of
them can drift out of sync and show a rejected run as a match somewhere
staff would not think to look. An attribution a staff member already
confirmed stays confirmed if the run is rejected afterwards: rejecting a
run never unlinks an association that already exists.

**What the evidence columns mean.** Every candidate row shows four
separate facts side by side, never collapsed into one cell: the matched
telescope, the date overlap between the orphan's window and the run's
window (stated in words, including the mismatch when there is one), the
campaign the run belongs to, and the instrument-string similarity between
the orphan's instrument text and the run's. The numeric score is
additional to these facts, never a replacement for them -- it renders as a
small, visually subordinate chip after the evidence, there for staff to
sanity-check the banding while the matcher is new, not to be read on its
own. Each row is also tagged with a named confidence band -- **High**,
**Medium** or **Low** -- and the band filter at the top of the page
narrows either worklist to one band at a time.

**The checkbox gate.** A checkbox appears on a candidate row only when it
is a High-band candidate *and* the only High-band candidate for its
orphan -- if two candidates for the same orphan are both High, neither
gets a checkbox, and a staff member must pick one explicitly with the
per-row Confirm button instead. The server re-checks both conditions
again when a bulk "Confirm selected" submission arrives, never trusting
that a checkbox was only rendered for an eligible row. A checkbox is
therefore a shortcut for a decision a human would make unhesitatingly for
an unambiguous pair, not a bulk guess across ambiguous ones.

**What a dismissal means.** Dismissing a candidate records that a
suggested pair was rejected -- who rejected it, when, and why, from a
required free-text reason. A dismissal is **not an association**:
persisting one never creates a link between the orphan and the run, and
an unconfirmed guess can never be mistaken for a confirmed attribution. It exists so
the queue can actually drain -- without it, a rejected candidate would
return on every page load -- and it is fully reversible from the
collapsed "Dismissed" section on the same page, which lists who dismissed
each pair and offers an Undo button for every row. That reason box is
required for Dismiss only -- clicking Confirm on the same row never asks
for one, because a confirmation records the pair itself rather than a
rejection of it.

Undoing a *confirmed* attribution writes a dismissal for that same pair
as part of the same action. This is what keeps the undo attributable (the
link's own audit fields are cleared by the undo, so the trace has to live
somewhere else) and what stops the matcher immediately re-suggesting the
exact pair a staff member just decided was wrong. A freshly-undone
confirmation therefore appears in the Dismissed section, not directly
back in an open worklist, until that dismissal is itself undone.

**The done signal Phase 29 depends on.** The attribution pass is complete
when both worklists are empty and the page shows its "Attribution
complete" heading, naming how many orphans still have no matching run at
all and confirming that the Phase 29 reconcile sweep is safe to run.
There is no backlog-reporting management command -- this signal is read
from the page itself, by design.

**Attributing changes nothing about the entry itself.** Confirming a
candidate here only sets the entry's attribution link
(``CalendarEventMeta.run``) plus who confirmed it and when -- it never
rewrites the entry's title, description, or any other field. The entry
then shows its campaign as a pop-up decoration and a month-cell marker
(see "Why doesn't the calendar pop-up show an 'Attributed campaign run'
block?" below), rendered from that link at request time, and nothing
else changes. An entry backed by a real observation record
(``CalendarEventMeta.observation_record``) is offered in this queue
exactly like any other unattributed entry -- the queue keys off the
attribution link alone, never off whether an observation link is also
present.

**Behavior change:** before this phase, the only mechanism that could
create a run-to-event link was the Django admin's foreign-key picker on
``CalendarEventMeta``/``CampaignRunObservation``, with no evidence, no
worklist, and no undo. That admin path still exists and remains available
for a pair the matcher never offers a candidate for, but the attribution
page above is now the primary, evidence-backed route for the common case.
A link created through the admin's ``CalendarEventMeta`` page is now
stamped automatically with the staff member who saved it and the time
they saved it -- those two fields are no longer editable by hand on that
page, and clearing the run there clears them too -- so an association
created through either surface is attributable, not only one created
through the attribution queue.

How do I mark a run cancelled or weathered-out?
--------------------------------------------------

Once a campaign run is approved, the approval queue's **Decided** table
shows "Mark Cancelled" (``action=mark_cancelled``) and "Mark Weathered"
(``action=mark_weather_failure``) buttons on that row's Actions column
(they appear for any approved run regardless of its current observing
status). Clicking one immediately and publicly prepends
``[C]`` or ``[W]`` to the title of **every**
``CalendarEvent`` associated with that run -- including every per-night
event of a multi-night range-window run -- on the shared campaign calendar
that anonymous visitors can see. ``[C]`` is the same marker a portal
cancellation on an LCO/SOAR observation record uses (see the marker table
above); the pop-up's own "Run status:" line is what tells a visitor this
cancellation came from a staff decision on the run, not the portal. There
is no separate confirmation step and no revert button, but the action is a
safe, idempotent no-op to re-click: clicking the same button again, or
clicking the other button to correct a mis-click, simply re-applies the
new prefix without creating duplicate events or losing any data.

.. note::
   **One-time title change (Phase 37).** Before this phase, a cancelled or
   weathered run's events carried the bracket-word prefix ``[CANCELLED]``
   or ``[WEATHERED]``. The first sweep after this phase (either
   ``reconcile_campaign_runs`` run by hand, or the automatic reconcile
   every staff decision on a run already triggers) rewrites those prefixes
   to the short-letter form ``[C]``/``[W]``, once. This mirrors the
   Phase 34 "One-time title change" note above for the observation
   projector's own titles, and it is expected, not a fault -- a deployment
   that has not yet run the sweep re-titles itself within one unattended
   tick, because both re-titling sweeps run every tick (see "How do I run
   everything unattended?" below).

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

**Correcting a run's source to a queue value changes its calendar entry
from one-per-night to one whole-window entry.** A run's ``source`` decides
which of the two calendar forms it gets, regardless of its site -- see
"How do I get every campaign run onto the calendar?" above. Relabelling a
per-night run's ``source`` to ``lco_queue``/``soar_queue``/
``gemini_queue``/``eso_queue`` does not change anything on the calendar by
itself: the next reconcile (a full sweep, or the run's own next
staff-action reconcile) is what converges on it, deleting the run's
leftover per-night events (counted under ``legacy_deleted``) and replacing
them with a single whole-window entry.

**A ``LEGACY`` row stays per-night until a human relabels it.** Nothing
infers a run's provenance from its site or its telescope name -- a
``LEGACY`` run with a resolved site keeps its per-night entries forever
unless a staff member corrects its ``source`` through this same admin
action, which is the only way a ``LEGACY`` row ever moves into the
queue-sourced whole-window form.

**What happens to an already-reconciled run's calendar events when you
correct its** ``telescope_class`` **or** ``site`` **(its source does not
change this):** ``reconcile_run()`` re-derives which calendar-event family
(a single whole-window entry, or one entry per observing night) a run
belongs to from its *current* ``telescope_class``/``site`` values every time
it runs. If the correction moves the run into the other family -- for
example, setting a ``telescope_class`` on a run that previously had a
resolved site, or correcting a run's ``site`` to a satellite site -- the
next reconcile (either a full ``reconcile_campaign_runs`` sweep, or the
run's own next staff-action reconcile) automatically detaches the old
family's events from the run rather than leaving them on the calendar
looking like a live commitment forever -- unless a staff member has
already confirmed one of those events to this run, in which case the
sweep leaves it alone entirely (see ``detach_declined`` below): a human
decision always outranks an automated sweep. Detaching, not deleting: the
old events stay on the calendar but return to the attribution page's
worklist (``campaigns:attribution``, see "How do I attribute existing
calendar events and observation records to a run?" above), where a staff
member can re-confirm them. Re-confirming a released entry is a permanent
decision -- once a person has confirmed it, no later automated sweep
releases it again. The correction itself does not trigger the detach -- it
happens on the *next* reconcile, same as any other calendar-visibility
change only renders correctly once a sweep runs afterward.

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
   admin or a shell.

   **A correction that does not resolve is discarded, and the command says
   so.** The guard cannot tell "the sheet's cell is still blank" from "someone
   typed a new code that MPC does not know" -- both are "did not resolve", so
   both keep the old ``site`` **and** the old ``site_raw``. Each such row now
   prints a line on stderr naming the site it kept and the ``Site Code`` it
   discarded, and the summary line ends with a ``site_preserved:`` count. Read
   those: the row itself is still reported as ``unchanged``, because nothing
   the command was allowed to write actually changed. If your correction is
   real, fix the site from the Django admin (or the Sites Needing Review
   queue) rather than through the sheet.

   A preserved row also gets **no** newly-derived ``telescope_class``: the
   class records *why there is no site*, and a preserved row still has one, so
   there is nothing for a class to explain. A non-blank ``telescope_class`` is
   never blanked by a re-import, and -- as of this phase (D-04) -- it is also
   **never replaced by a different derived value**: the class this command
   computes is always an inference from the sheet's free text, and an
   inference never overwrites a stored value, it only ever fills a blank one.
   This is consistent with the "it is **permanent**: it is never cleared by
   any command" sentence in the note below -- before Phase 27.1 the importer
   *did* blank it whenever the site resolved, and before this phase a
   re-import could still silently replace a hand-corrected class with a
   different derived one. Each such row prints a line on stderr beginning
   ``kept existing telescope_class``, naming the value that was kept and the
   one the CSV derived and discarded, and the summary line carries a
   ``telescope_class_preserved:`` count alongside ``site_preserved:`` -- the
   row itself is still reported as ``unchanged``, so the stderr line and the
   summary count are the only places a preserved correction is visible.

   The ``site_needs_review`` count in the command's summary
   line reports how many rows **end up** flagged, not how many flags the
   command wrote -- so a preserved row that is already resolved and unflagged
   no longer inflates it, and a preserved row that is still flagged (a
   resolved site whose review flag was never cleared) is still counted, because
   it really is in the Sites Needing Review queue.

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

How do I run the one-time classical cutover?
---------------------------------------------------

``cutover_classical_allocations`` is a one-time command that converts
every legacy blank-url classical ``CalendarEvent`` -- written by
``load_telescope_runs`` before its allocation rewrite -- into a
``CampaignRun`` plus ``ALLOC:``-keyed events, so the calendar ends up with
exactly one writer per night. It needs no schedule file: every fact it
needs (telescope, instrument, status, date range, sub-night window,
optional proposal) is recovered by re-parsing the ``Source line:`` each
legacy event's own description already carries.

This is a one-time migration step, run alongside
``repair_stale_campaign_run_sites`` above (the other one-time, safe-to-repeat
command in this codebase) as part of the cutover to the allocation
projector.

**Run this command BEFORE the first rewritten ``load_telescope_runs``
import of the same schedule file.** An import that runs first creates the
``CampaignRun`` and its ``ALLOC:`` nights itself, so the cutover will then
find those urls already held and refuse to re-key onto them -- reporting
the stranded legacy events as ``key_collision`` instead of converting them.
Running the cutover first avoids that entirely.

It runs the following steps, in order:

.. code-block:: console

   >> python3 manage.py migrate
   # deploy the code
   >> python3 manage.py cutover_classical_allocations --dry-run
   >> python3 manage.py cutover_classical_allocations
   >> python3 manage.py reconcile_campaign_runs --dry-run
   >> python3 manage.py reconcile_campaign_runs

1. ``migrate`` applies the sub-night window fields ``load_telescope_runs``
   now stores on ``CampaignRun``.
2. Deploy the rewritten code.
3. ``cutover_classical_allocations`` converts every explainable legacy
   blank-url classical event once, re-keying it to ``ALLOC:{run_pk}:{night}``
   in place -- same primary key, same ``start_time``/``end_time``, no
   ``sun_event()`` recompute.
4. ``reconcile_campaign_runs`` -- the first sweep after the cutover takes
   over every remaining ``RUN:{pk}:{date}`` night still reachable by its
   own run's per-night dispatch (``rekeyed``), and deletes the rest as
   one-time churn for a run that now dispatches to a whole-window
   container (``legacy_deleted`` -- see "How do I get every campaign run
   onto the calendar?" below).

**Always run ``--dry-run`` first and read the unexplained list.** The
command groups blank-url classical events by their own ``Source line:``,
re-parses each group, and creates or updates the ``CampaignRun`` that
group would have produced under a fresh import. What it converts: every
group whose schedule line parses, whose telescope resolves to a known
``Observatory`` with a timezone set, whose events agree on their campaign,
whose events are not already attributed to a different run, and whose
derived observing nights are not already claimed. What it deliberately
leaves alone: **an event or group it cannot explain is left completely
untouched and reported** with its primary key, title and reason -- no
parseable ``Source line:`` marker; a ``Source line:`` that does not parse
or names an unknown telescope; a resolved site with no timezone; a group
whose events disagree on their campaign; an event already attributed to a
different run; **``key_collision``** -- a second event whose derived
observing night is already claimed, either by another event in this same
run (a duplicate row from a pre-cutover re-ingest) or by a
``CalendarEvent`` row that already holds the derived
``ALLOC:{run_pk}:{night}`` url (the import-ran-first case above);
**``duplicate_identity``** -- either a second GROUP (a second, distinct
``Source line:`` string) whose derived run identity key is the same as an
earlier group's, because the key ignores the schedule line's status word
-- e.g. an ``allocation`` line and a ``cancelled`` line for the same
telescope, instrument and window -- or a ``CampaignRun`` row that
already holds the derived identity key whose own stored
``observation_details`` ``Source line:`` marker is absent or differs
from the group's own line, so the command cannot prove the run came
from it; **``window_mismatch``** -- the event's
own independently-derived observing night falls outside the window its
own schedule line implies (an off-by-one sub-night boundary, or a stored
start time that disagrees with the line's date range); or any other
unexpected error. For a ``key_collision``, the first event to claim a
night is still converted -- only the extra claimants are reported. The
operator action for a ``key_collision``: find the duplicate row in the
Django admin (the printed reason names the colliding pk when the url is
already held elsewhere) and delete it or re-attribute it, then re-run the
command. The operator action for a ``duplicate_identity``: the SECOND
group is never merged into the first group's run -- but only WHEN the
first group's ``CampaignRun`` has a stored ``Source line:`` marker that
is recoverable and matches the group's own line. The database check
catches a claimant left behind by an earlier cutover pass or by a prior
``load_telescope_runs`` import, not only this process's own in-memory
bookkeeping -- but ``observation_details`` is a free-text field editable
from the Django admin, from ``import_campaign_csv.py`` and from the
campaign submission form, so a claimant whose marker is absent or
differs from the group's own line is reported under
``duplicate_identity`` instead of converted, because this command
cannot prove such a row came from the line in hand. This is reachable
through the very Django-admin edit this paragraph itself asks the
operator to perform. The remedy covers both cases: when the two Source
lines differ, edit the affected events' description ``Source line:``
text to disambiguate the two groups in the Django admin -- give the
later group's events a different bracketed proposal token from the
earlier group's; when no marker is recoverable at all, restore or
correct the claimant run's ``observation_details`` ``Source line:``
text in the Django admin so it matches, or disambiguate the two lines --
then re-run the command in either case. The operator action for a ``window_mismatch``: correct the event's stored start time in
the Django admin, or correct the schedule line's date range and
re-import, so the two agree -- the event is refused rather than converted
because re-keying it would write an ``ALLOC:`` url outside the run's own
window, which the next ``reconcile_campaign_runs`` sweep classifies as
stale and deletes. A group **all** of whose events are attributed
elsewhere has no run created or updated for it at all, so a summary line
reading ``runs created: 0`` next to a ``foreign_attribution`` count is the
designed outcome, not a silent failure.

.. note::
   **Re-run gotcha:** a claimant whose stored ``Source line:`` marker
   MATCHES the group's own line is find-and-updated, so ``run_status``,
   ``target``, ``campaign``, ``site``/``site_raw``,
   ``window_start``/``window_end``, the two sub-night fields and
   ``observation_details`` are all re-applied from the schedule line on
   every invocation. A post-import staff edit to any of them does not survive the next cutover run.
   This is expected behaviour for a
   file-authoritative command, not a defect -- the same class of
   behaviour the ``import_campaign_csv`` "Re-import gotcha" note above
   documents at length -- and it applies here because this command CAN
   prove the row came from this schedule line, which is exactly the
   distinction the identity guard above establishes.

   This closes a loop worth naming explicitly: the ``duplicate_identity``
   no-marker remedy above tells an operator to restore the claimant's
   ``observation_details`` ``Source line:`` marker -- doing so converts a
   refused claimant into a matching one, whose fields will then be
   re-applied on the next run. Restore the marker deliberately, not
   reflexively.

``--dry-run`` applies every per-event check the real run applies -- window
containment, in-run collision, and existing-``ALLOC:``-url collision --
through the same shared helper the real run calls, so its counts, its
reason breakdown and its exit status match what the real run then
reports. A second group sharing an earlier group's run identity key is
rejected identically on both passes, before either one attempts a write
for it, so the two passes also agree about which group exists at all. That
is what makes "always run ``--dry-run`` first" worth the operator's time.

**A non-zero exit is expected, not a bug, whenever an unexplained event
remains.** The command raises a self-contained error naming the count and
the reason breakdown; resolve the listed events in the Django admin (fix
the description's ``Source line:``, resolve the telescope, set the site's
timezone, or clear the conflicting attribution, as the reported reason
names), then re-run the command -- it is safe to repeat. **The command
never deletes a ``CalendarEvent`` on any path**, including every failure
path: what it cannot explain, it reports and leaves byte-identical.

Worked example, from a real scratch-copy run against the developer
database (plan 35-06): starting from 241 events (56 date-bearing
``RUN:{pk}:{date}`` nights, 16 bare ``RUN:{pk}`` containers, 10 blank-url
classical events, 159 facility-url observation events), the cutover
converted 9 of the 10 blank-url events (3 groups, 3 new runs) and left 1
unexplained (a pre-existing junk ``tmp`` row with no recoverable
``Source line:``, the known example this command's own docstring names).
The following ``reconcile_campaign_runs`` sweep then re-keyed 48 nights
and deleted 8 leftover per-night events for runs a source correction had
already sent to a container, leaving 233 events total: zero date-bearing
``RUN:`` nights, 57 ``ALLOC:``-keyed events, 16 containers unchanged in
count, 1 blank-url event (the same unexplained junk row), and all 159
facility-url events byte-identical (url, title, description, start, end
and ``modified`` all unchanged). These are the numbers from one real run,
quoted as a worked example -- not a promise about what any other database
will show.

How do I get every campaign run onto the calendar?
---------------------------------------------------------

``reconcile_campaign_runs`` is the one idempotent sweep that projects and
refreshes calendar events for every ``CampaignRun`` in the database, in a
single pass. Running it a second time against unchanged data writes
nothing -- a repeat sweep reports the same runs as ``unchanged`` rather than
touching them again. It replaces the now-retired one-off range-window
backfill command and the whole per-gap-backfill pattern it belonged to:
instead of a new command for each historical gap, this one command
re-derives every run's calendar state from the run itself, so there is
nothing left to backfill separately.

Always run with ``--dry-run`` first to preview what would change, with no
database writes:

.. code-block:: console

   >> python3 manage.py reconcile_campaign_runs --dry-run
   >> python3 manage.py reconcile_campaign_runs

The final summary line reports these counters -- ``would_create``/
``would_update``/``would_leave_unchanged`` in ``--dry-run`` mode, or
``created``/``updated``/``unchanged`` for a real sweep, alongside ``runs``,
``skipped``, ``failed``, ``blocked``, ``skipped_nights``, either
``would_detach`` (``--dry-run``) or ``detached`` (a real sweep),
``detach_declined``, ``remint_declined``, and three allocation-handoff
counters -- ``would_retire``/``retired``, ``would_rekey``/``rekeyed`` and
``would_delete_legacy``/``legacy_deleted``::

   Done (dry run). runs: 19, would_create: 0, would_update: 0, would_leave_unchanged: 15, skipped: 4, failed: 0, blocked: 0, skipped_nights: 2, would_detach: 1, detach_declined: 0, remint_declined: 0, would_retire: 0, would_rekey: 0, would_delete_legacy: 0
   Done. runs: 19, created: 0, updated: 0, unchanged: 15, skipped: 4, failed: 0, blocked: 0, skipped_nights: 2, detached: 1, detach_declined: 0, remint_declined: 0, retired: 0, rekeyed: 0, legacy_deleted: 0

``retired`` counts an allocation night removed from the calendar for any of
five reasons (35-REVIEW.md NF-07): (1) a run's linked ``ObservationRecord``
placed or observed its block on that night, so the projected
sunset-to-sunrise event is no longer needed -- the observation's own
calendar entry is that night's entry now, and unlinking the record restores
the night on the next reconcile (this is the ONLY one of the five reasons
that "unlink to restore" sentence applies to); (2) a boundary-affecting
field changed since the night was last minted -- either a sub-night window
field (the run's own dawn/dusk or dark-window overrides), or a correction
to the run's ``site`` (see "Can I correct a run's source?" above). What a
site correction does to an already-minted night now depends on whether the
run's sub-night window is set: a run whose ``night_start_utc``/
``night_end_utc`` are empty or only half-set has the night deleted and
re-created fresh, at the corrected site's real sunset/sunrise, exactly as
this reason has always described; a run with BOTH fields set has its
boundaries pinned by those fields, so a same-timezone site correction
changes only the dark-window line in the event's description (refreshed
automatically on the next sweep -- see the paragraph on correcting a
site's own definition, below) and nothing is retired for that night; and a
correction that moves such a run to a site in a different timezone makes
the whole run fail to reconcile instead -- reported
``Run pk=N: reconcile failed (...) -- skipping``, with the night's existing
event left untouched, because a sub-night window pinned to one site's
night is not a valid window at a site in a different timezone. The remedy
for that last case is to correct the run's sub-night window fields
together with its ``site``, not the ``site`` alone; (3) the night no
longer falls inside the run's window at all -- a window shrink, or a
re-classification that moves the run off the per-night allocation branch
entirely; (4), after this change (35-REVIEW.md NF-01), a leftover night
whose companion row was deleted outright or had its ``run`` cleared --
previously left on the calendar forever with no counter moved, now removed
and counted here like every other unneeded night; or (5) a one-time
provenance audit: a night minted before this release, or carried across by
the re-key path, whose recorded mint inputs are absent or were recorded in
a pre-release format is resolved once against the computed sun event, and
re-minted when the stored boundary disagrees by more than one minute. This
release repeats that audit a second time, for every night still carrying
the PREVIOUS release's token format -- for the same reason, and with the
same once-per-night bound -- and checks one more input the previous audit
did not: the site's own stored position (latitude, longitude, altitude)
and timezone, alongside the sub-night window and the site identity it
already checked. This happens at most once per night -- the same night
reports ``unchanged`` on every sweep after it, because the resolution
records a current-format provenance token the first time it runs.

**Correcting a site's own definition also re-mints, separately from
correcting a run's** ``site``. Editing an ``Observatory`` row's latitude,
longitude, altitude or timezone in the Django admin -- without touching any
run's ``site`` field at all -- now re-mints every allocation night already
projected at that site, on the next sweep, counted under ``retired`` and
``created`` exactly like any other re-mint. Before this release such an
edit was invisible to the sweep: a night's boundaries were computed once,
at mint time, from whatever the site's position was that day, and nothing
ever compared them against the site's *current* position again -- an
operator correcting a mis-entered coordinate, for example, would see every
night at that site keep reporting ``unchanged`` forever, silently wrong by
up to fifteen hours. One case deliberately does not re-mint: a position
correction too small to move the computed sunset or sunrise by more than a
minute is recognised as within tolerance and reported ``unchanged``, the
same tolerance reason (5) above already applies -- so a small position fix
(rounding a coordinate, for instance) does not churn the calendar.

``rekeyed`` counts a night carried across from the old, retired
``RUN:{pk}:{date}`` key form into the current ``ALLOC:{pk}:{night}`` form,
in place -- same primary key, same start/end time, just re-keyed. This is
the ongoing per-run takeover every reconcile performs for a run that still
dispatches per-night.

``legacy_deleted`` counts one-time churn from a run's dispatch changing --
never per-sweep, always zero again on the next reconcile of the same run --
covering three distinct origins (35-REVIEW.md NF-07): (1) a run with a queue
source (``lco_queue``/``soar_queue``/``gemini_queue``/``eso_queue``) leaves
behind leftover per-night ``RUN:{pk}:{date}`` events from before that
dispatch rule applied; (2) a run re-classified (a ``telescope_class``/
``site`` correction -- see "Can I correct a run's source?" below) OUT of the
per-night ``ALLOC:{pk}:{night}`` allocation branch leaves its old allocation
nights behind, with no other code path left to reach them
(35-REVIEW.md CR-02) -- not a queue source at all; and (3), after this
change (35-REVIEW.md NF-01), a leftover per-night row in either family with
no ``CalendarEventMeta`` companion row at all, or one whose ``run`` is
unset, is deleted here too rather than left unreachable. The sibling claim
"this run now keeps a single whole-window entry" applies ONLY when the run
is actually container-dispatched after the reconcile that reported the
count (origins (1) and (2) above): a run that still dispatches per-night
(origin (3), or a night re-minted under reason (2) above) can report a
non-zero ``legacy_deleted`` while continuing to keep many entries, not one.

``skipped_nights`` counts classical nights whose calendar entry already
comes from another writer attributed to that run -- so
``created: 0, updated: 0`` alongside a non-zero ``skipped_nights`` means
"this run's nights are covered elsewhere", not "already converged" (those
read identically without this counter). ``detached`` counts entries the
reconciler released back into the attribution queue, for either of two
reasons: a night superseded by another attributed entry (the skip rule
below), or events left over from a key family a run no longer belongs to
after a ``telescope_class``/``site`` correction (see the re-classification
note above). ``--dry-run``'s ``would_detach`` reports the same number a
real sweep would detach -- the count is a pure read, so there is nothing
stopping the preview from showing it.

``detach_declined`` counts declines caused ONLY by a human confirmation --
``confirmed_by`` on a night's companion row -- because a person's decision
always outranks an automated sweep, and this counter exists so that fact
is reported rather than left to look identical to "nothing to release".
Two causes, both confirmation-caused:

The first is a companion row the sweep did not RELEASE: a
``telescope_class``/``site`` correction moved the run out of the
calendar-event family that row belongs to, but a staff member had already
confirmed that row's attribution, so it is left alone rather than detached
back into the attribution queue (see "What happens to an already-reconciled
run's calendar events when you correct its ``telescope_class`` or ``site``"
above).

The second, new this round: an allocation night the sweep did not DELETE
when a linked ``ObservationRecord`` would otherwise have retired it (see
``retired`` reason (1) above) -- because a staff member had already
confirmed that night's companion row. A confirmed night now survives its
own retirement instead of being deleted out from under the confirmation.
The consequence an operator actually sees: the calendar shows BOTH the
confirmed allocation night and the linked observation's own entry, on the
same night, until someone clears the confirmation on that night's
companion row and re-runs the sweep -- clearing it lets the retirement
proceed on the next sweep, and the duplicate resolves down to the
observation's entry alone.

For a declined attribution release, there is nothing for an operator to
do: it is a report that a human decision was respected. For a declined
retirement, there is a remedy if the duplicate is unwanted: clear the
confirmation on that night's companion row in the Django admin and re-run
the sweep, or leave it as is and accept the duplicate.

A declined retirement ALSO counts under ``updated`` or ``unchanged`` on
the same sweep -- the same pair the ``remint_declined`` section below
already states for its own decline, and for the same reason: the night
survives and only the delete is declined, so its title, description and
campaign label are still refreshed. This is what keeps a later **Mark
cancelled** / **Mark weather/technical failure** action reaching a night
whose retirement was declined, so the entry picks up its ``[C]``
/ ``[W]`` marker on the next sweep instead of sitting on the
calendar as an ordinary observing night for as long as the confirmation
stands. The night's primary key, start time, end time and
``confirmed_by``/``confirmed_at`` stamp are never touched by that
refresh, and no sun-event calculation is paid for it. The
declined-retirement warning therefore repeats on every sweep until the
confirmation is cleared, which is the standing report for a night the
sweep is refusing to delete, not a fault.

``remint_declined`` counts a night whose boundaries would have changed --
a re-mint that ``retired`` reason (2) above would otherwise have performed
-- but which the sweep declined to delete and re-create, for any of three
causes: a human confirmation (``confirmed_by``) on the night's companion
row; a real ``ObservationRecord``/``ObservationGroup`` link on that row,
whether or not it is confirmed; or an unverified companion row --
``is_verified`` unchecked. This counter is deliberately separate from
``detach_declined``: a re-mint decline destroys nothing and releases no
attribution, so counting it there would tell an operator the wrong thing
happened.

A declined night ALSO counts under ``updated`` or ``unchanged`` on the
same sweep -- deliberately, not a contradiction: its title, description
and campaign label are still refreshed on the same sweep, because those
are not the destructive half of a re-mint; only the boundary rewrite
(start time, end time, primary key) is declined. This warning repeats on
every sweep until the night is resolved one way or the other -- that is
the standing report for a night the sweep is refusing to correct, not a
fault.

The third cause -- "an unverified companion row" -- comes from the
``is_verified`` checkbox on ``CalendarEventMeta`` in the Django admin,
whose label reads ``Whether the telescope label was live-verified against
the LCO API (unchecking also vetoes an automated re-mint)``. Unchecking it
permanently prevents the allocation projector from correcting that night's
boundaries. It does NOT prevent that same night being retired when a
linked observation places a block on it -- see ``detach_declined`` above
for what protects a night from that.

The remedy: clear the confirmation or the link, or check ``is_verified``
to clear the unverified state, on that night's companion row in the
Django admin and re-run the sweep -- or leave it and accept the stored
boundary.

**Post-upgrade deploy note.** After upgrading to a release carrying the
provenance-audit behaviour described under ``retired`` reason (5) above,
run ``reconcile_campaign_runs --dry-run`` first: a non-zero
``would_retire`` on nights nobody edited is that one-time audit, not a
regression -- every night minted before this release carries absent or
pre-release-format mint inputs, and each one is resolved once against the
real sun-event calculation. The same nights report ``unchanged`` on every
sweep afterwards, because the audit records a current-format provenance
token the first time it resolves a night. This applies again to THIS
release: every night still carrying the previous release's token format
resolves once more, now also checking the site's own stored position and
timezone, and a non-zero ``would_retire`` on nights nobody edited is again
the expected one-time audit, not a regression. A non-zero
``remint_declined`` on the same ``--dry-run`` is a different thing to
read: nights carrying a confirmation, an observation link or an
unverified companion row that the sweep will not correct on its own --
see ``remint_declined`` above for the remedy.

A per-run line accompanies each non-zero counter: a skipped-night line on
stdout (normal, expected convergence, not a failure); a detached line on
stderr alongside the existing ``blocked`` line, naming the run and stating
that entries were released back into the attribution queue; and, when a
confirmed row was left alone, a declined line on stderr naming the run and
the count.

A run that does not project onto the calendar at all is reported on stderr
with one of these skip reasons, one line per run:

* ``not approved`` -- the run is still ``pending_review`` or ``rejected``.
  Approve (or reject) it from the approval queue first; an unapproved web
  submission must never reach the shared calendar.
* ``missing telescope/instrument`` -- the run has no
  ``telescope_instrument`` value at all; there is nothing to title the
  calendar entry with until one is set.
* ``TBD window`` -- the run has no concrete ``window_start``/``window_end``
  yet (an unparsed ``Obs. Date``); there is nothing to project until the
  window resolves.
* ``unresolved site`` -- the run has no resolved ``site`` and no
  ``telescope_class`` to explain the absence. Resolve it from the "Sites
  Needing Review" queue on the approval page, or set a ``telescope_class``
  if it is genuinely a class-wide or space allocation.
* ``window_end before window_start`` -- the run's ``window_end`` is earlier
  than its ``window_start`` (a hand-edited admin value, or an upstream
  window-parsing bug). Correct the window fields in the admin; there is
  nothing to project until they describe a real forward-running range.

A run whose site has no ``timezone`` set fails differently -- it reaches the
per-night sunset/sunrise calculation and raises there, so it is reported
separately as ``Run pk=N: reconcile failed (...) -- skipping`` rather than
one of the five skip reasons above. See "Observatory missing timezone" in
Troubleshooting below for the fix. This now also applies to a
queue-scheduled run at such a site: it used to bypass this calculation
entirely (getting a whole-window entry instead), but a queue-scheduled run
with a resolved site follows the same per-night path as a classically-
scheduled run there, so a blank ``timezone`` fails it the same way.

What an operator sees on the calendar afterwards, in plain terms: **a run's
``source`` decides which of the two forms it gets, not its site.** A run
whose source is a queue value (``lco_queue``/``soar_queue``/
``gemini_queue``/``eso_queue``) always keeps a single whole-window entry,
even when it has a fully resolved ground site -- a queue window is a
request, not a set of owned nights, so it is never fanned out into
per-night entries. A ``classical_file``, ``csv_import``, ``web`` or
``legacy`` run with a resolved site and window shows one calendar entry
per observing night instead, spanning that site's sunset-to-sunrise,
sitting alongside the individual observation entries the LCO/Gemini sync
commands already create for it. A class-wide allocation with no fixed
site, and a satellite run, still show a single entry spanning their whole
window, for the same reason a queue-sourced run does -- there is no single
site's dark time to bound a per-night entry to.

The run's free-text ``Telescope / Instrument`` value is split on the first
``/`` or ``+`` into the calendar entry's separate **Telescope** and
**Instrument** fields in the event pop-up; a value with no delimiter goes
wholly into Telescope. The entry's title still shows the full combined text
either way. Entries for class-wide and satellite runs pick this up
automatically on the next sweep, because that whole-window entry is
rewritten from the run every time. Per-night entries -- for a
classically-scheduled run, or a queue-scheduled run with a resolved site --
created before this change keep their old combined value, because a
per-night entry's Telescope/Instrument and its sunset/sunrise window are
deliberately never rewritten after it is first created.

**A night that already has an entry attributed to this run gets no
reconciler entry at all.** The attributed entry -- whether it comes from
``load_telescope_runs``, a hand entry, or any other writer -- IS that
night's calendar entry, and the campaign shows on it as a pop-up
decoration (see "Why doesn't the calendar pop-up show an 'Attributed
campaign run' block?" below) rather than as a second, reconciler-created
entry sitting alongside it. The reconciler now only ever creates, updates
or removes entries it keyed itself (its own ``RUN:{pk}:{date}``/``RUN:{pk}``
urls); it never re-keys or edits another writer's entry.

**Which observing night an entry's start time belongs to is anchored at
local noon**, the same convention the sunset/sunrise calculation itself
uses: the night runs from local noon of a date through local noon of the
next date, so an entry starting after local midnight belongs to the
PREVIOUS date's night, not the date its own local calendar date would
name. A 02:00-local start on 9 August, for example, belongs to the night
that began at sunset on 8 August.

**The rule applies whether or not the reconciler had already made its own
entry for that night.** If it had not, nothing is created for that night
and the skip is the whole story. If it had -- the realistic case once
another writer (a classical-schedule loader, a hand entry, or an
observation record) attributes a real entry to the same run for a night
the reconciler already covered -- that earlier reconciler-created entry is
released (never deleted) back into the attribution queue, where a staff
member can re-confirm it. Re-confirming it is a permanent decision: once a
person has confirmed it, no later automated sweep releases it again.
Clearing the other entry's attribution instead brings the reconciler's own
entry back, in place (same record, same url), on the next sweep.

**One-time title change.** Reconciler-created entries no longer carry the
campaign name in their title -- only the telescope/instrument text (and,
for a cancelled/weathered run, its status prefix). The campaign name now
appears only as the month-cell marker and the pop-up block described in
"Why doesn't the calendar pop-up show an 'Attributed campaign run' block?"
below, rendered from the entry's attribution link rather than written into
the title. The first sweep after this change updates the titles of every
one of the reconciler's own existing entries once, dropping the old
campaign-name prefix; that one-time title change is expected, not a fault.

**You will rarely need to run this by hand.** The same reconciliation now
happens automatically, immediately, for a single run the moment staff
approve it, resolve its site, or mark it cancelled or weather-failed from
the approval queue -- ``reconcile_campaign_runs`` is for sweeping every run
at once (for example, after a bulk site repair) or backfilling a gap found
later, not for routine day-to-day use.

.. _unattended-operation:

How do I run everything unattended?
------------------------------------------

``run_unattended`` is the one FOMO-owned management command a
``flock``-guarded cron line invokes on a fixed schedule. Once it is set up
on a host, none of the sweep commands documented above need to be run by
hand for routine operation -- the runner calls their underlying logic
directly, in one process, every tick.

What runs, and when
^^^^^^^^^^^^^^^^^^^^^^^^

Every 15 minutes (``*/15 * * * *``), ``run_unattended`` runs five steps, in
this fixed order, in one process:

1. **status_refresh** -- the FOMO-owned LCO/SOAR observation-status
   refresh, replacing TOM's stock ``updatestatus`` command. It never
   touches Gemini or ESO, which have no facility read-back to refresh.
2. **project_sweep** -- the observation projector's backstop sweep (the
   same logic ``project_observation_calendar`` runs), including the
   one-time observed-telescope lookup for a newly observed record.
3. **discovery** -- ``backfill_lco_observations`` run with no arguments,
   sweeping every active ``WatchedProposal`` row (see "Adding a proposal
   to watch" below).
4. **reconcile** -- the campaign reconciler sweep, the same logic
   ``reconcile_campaign_runs`` runs.
5. **proposal_allocation** -- refreshes every watched or run-carried
   proposal's time allocation from the LCO Observation Portal
   (``timeallocation_set``) and stores it, one row per proposal code /
   semester / instrument type / allocation type, in
   ``ProposalTimeAllocation``. Public pages only ever read these stored
   rows -- an anonymous visitor's page load never triggers a credentialed
   portal call.

A step that fails never stops the later ones: every tick runs all five
steps, records each one's own outcome, and exits non-zero at the end only
if at least one step failed. If **proposal_allocation** is the one that
fails (a portal outage, an expired API key), the existing failure email
below names it as the failing step exactly like any other; the other four
steps still run on the same tick; and on the public pages, any run whose
unused-nights estimate depends on that proposal shows **not yet known**
rather than a blank page or a stale figure -- a portal outage is never
visible to a public visitor as missing content, only as an honest "not yet
known".

Setting it up on a fresh host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. The runner needs two directories: one for its locks, one for its log.
   Only the log directory's shipped default is safe to use as-is.

   The shipped ``FOMO_LOCK_DIR`` default, ``/var/lock/fomo``, is not
   durable. ``/var/lock`` is a symlink to ``/run/lock``, which is a
   tmpfs, so anything created there -- including that directory itself --
   is gone after the next reboot. This is not something that might
   happen; it is a property of the path.

   That makes the failure permanent, not self-healing. After a reboot,
   the crontab's own ``flock`` guard can no longer open its lock file at
   all, so every tick dies before Python even starts. Pointing the
   crontab line somewhere else does not fix it by itself either --
   ``command_lock()`` tries to create the directory itself on every tick,
   and an unprivileged cron account cannot create anything inside
   root-owned ``/run/lock``. Nothing in the runner repairs this on its
   own; it stays broken until an operator changes the configuration.

   **Recommended: point** ``FOMO_LOCK_DIR`` **at a durable location owned
   by the cron account** -- for example ``~/.local/state/fomo`` -- and
   create it as that account. This route needs no ``sudo`` at all, which
   is part of why it is the recommendation::

      >> mkdir -p ~/.local/state/fomo

   **Alternative, for operators who want to keep the conventional
   location:** add a ``/etc/tmpfiles.d/fomo.conf`` entry so
   ``systemd-tmpfiles`` recreates ``/var/lock/fomo`` at every boot,
   naming the cron account as owner so the directory is never recreated
   world-writable::

      d /var/lock/fomo 0750 <cron-user> <cron-group> -

   This restores the directory on every boot, but not what the previous
   boot left inside it -- so the state file must not live there.

   **The second-order trap.** ``FOMO_STATE_DIR`` takes its default from
   ``FOMO_LOCK_DIR`` at the point ``settings.py`` defines it, and
   ``local_settings.py`` is imported only after that point -- so
   overriding ``FOMO_LOCK_DIR`` alone in ``local_settings.py`` leaves
   ``FOMO_STATE_DIR`` pointed at the old, tmpfs-backed default. Both
   variables must be set explicitly, together. Leave ``FOMO_STATE_DIR``
   behind and the D-11 suppression-state file sits on the tmpfs too: its
   contents are dropped at every reboot, so the runner re-decides
   "newly failing" and re-mails the failure notice all over again after
   each one.

   Set both ``FOMO_LOCK_DIR`` and ``FOMO_STATE_DIR`` in
   ``local_settings.py`` before running the preflight in step 6 below --
   the cron line it prints carries this host's *resolved* lock path, so
   setting them first means the printed line already matches this
   host's configuration.

   ``/var/log/fomo`` is unaffected by any of this -- it is on real disk,
   not tmpfs, and was never part of this failure. Create it exactly as
   before, with ``sudo``, owned by the cron account::

      /var/log/fomo

   If the primary state file cannot be written at tick time -- for
   example ``FOMO_STATE_DIR`` fills up or loses write permission for the
   cron account sometime *after* setup, past this section entirely --
   the tick logs an error naming both paths and writes the suppression
   state to a fallback file in the system temp directory (typically
   ``/tmp``) instead of failing (WR-17, 36-REVIEW.md). That fallback
   location is deliberately outside ``FOMO_STATE_DIR``: that directory is
   precisely what may have gone bad, and ``run_unattended`` is a fresh
   process per cron tick with nothing in memory surviving between ticks,
   so only a file somewhere still writable can carry the suppression
   decision forward. The fallback's filename is scoped to this
   deployment (a hash of the checkout path) and to the cron account's
   uid -- ``fomo-unattended-state.<uid>.<deployment-tag>.fallback.json``
   -- so two FOMO instances on the same host (staging/prod, or a test run
   beside a live cron deployment) can never collide on the same fallback
   path (CR-05/WR-33, 36-REVIEW.md). The next tick reads whichever of the
   primary and fallback files was written most recently, but **only**
   trusts a fallback file it can verify this account itself wrote --
   owned by the same uid, a regular file (never a symlink), with no
   group/other permission bits set (matching the mode every write below
   sets explicitly). A fallback file that fails that check -- for example
   one planted or overwritten by a different local account, since the
   system temp directory remains world-writable regardless of the
   filename -- is ignored rather than trusted, no matter how new its
   mtime is. Honoring a fallback tick's decision instead of ignoring it
   is what stops the next tick from deciding "newly failing" all over
   again and mailing the identical failure notice every 15 minutes for as
   long as the outage lasts (D-11 failing open). Once a later tick
   succeeds in writing the primary location again, it deletes the stale
   fallback by itself, so the runner reverts to reading the primary alone
   with no operator action needed. Both files are written atomically and
   with mode 0600; that mode covers confidentiality only, and it is the
   uid/regular-file check above -- not the mode alone -- that keeps a
   state file living in the shared system temp directory from being a
   silent integrity or availability exposure.

2. Put the real ``EMAIL_BACKEND`` (and its ``EMAIL_HOST_*`` settings) and
   the LCO/SOAR API key (from the LCO Observation Portal and the
   'Profile' link under your username in the top right corner) in this
   host's ``local_settings.py`` file. This should never be in the crontab
   line, never in an environment variable, and never committed to git.
   Write the API key as a flat, top-level assignment --
   ``LCO_API_KEY = '<your key>'`` -- because ``settings.py`` imports
   ``local_settings.py`` into a namespace of its own, near the end of
   the file: names you set there become settings, but anything you try
   to reach *into* -- a dictionary ``settings.py`` already built -- is
   not visible, and the attempt raises ``NameError`` before Django
   finishes starting. ``src/fomo/settings.py`` folds that one key into
   both the LCO facility entry and the SOAR facility entry, because
   SOAR authenticates against the same LCO Observation Portal. Leave
   the setting out and both the LCO and SOAR facility entries stay
   empty, and the tick's ``status_refresh`` step then fails on every
   non-terminal LCO/SOAR record -- a failure email every 15 minutes.
   (Gemini and ESO have their own credentials; this key does not reach
   them.)

   This host must also override three development defaults in the same
   file, or it will serve the site with a signing key that is public in
   this repository: ``SECRET_KEY`` (generate a fresh one --
   ``python -c "from django.core.management.utils import
   get_random_secret_key; print(get_random_secret_key())"``),
   ``DEBUG = False``, and ``ALLOWED_HOSTS`` set to this host's real
   name(s) -- ``DEBUG = False`` with the committed single-entry
   ``ALLOWED_HOSTS`` makes every request return 400, so the two must
   change together.
3. Create the heartbeat check. This is a dead-man's switch on a
   third-party service: it catches a tick that never ran at all or hung
   partway through -- the one failure class FOMO's own error handling
   cannot report itself, because the process that would have mailed
   never got there. A tick that runs and fails is already covered by
   the failure email. It is optional: leave ``FOMO_HEARTBEAT_URL``
   unset in the next step and this layer is simply off, and the
   schedule still runs and still mails on failure.

   Any healthchecks-compatible service works: healthchecks.io's hosted
   free tier, or a self-hosted ``healthchecks`` instance (the same
   open-source Django app) if a third-party dependency for something this
   load-bearing is unwelcome.

   Create ONE check for this schedule. Set the expected interval
   between pings (healthchecks.io calls this ``Period``) to 15
   minutes, matching this cron schedule, and the grace time
   (``Grace``) to about 20 minutes. The drift-free alternative is a
   Cron-type check carrying the same ``*/15 * * * *`` expression the
   crontab line uses, with the same grace; it has no interval to set.
   See "The two failure signals" below for how each check type alerts,
   why these numbers, why the grace must not be shrunk, and what the
   interval's default does if it is left alone.

   Finally, copy that check's own ping URL from the service and keep it
   for the next step, which exports it. On healthchecks.io it has the
   form ``https://hc-ping.com/<uuid>`` (a self-hosted instance gives the
   same shape under your own host). The ``<uuid>`` part is the ping
   token, so this URL is itself a credential and must never go into a
   committed file (D-15). It also must never go into ``local_settings.py``
   on a host that then serves a docs build: Sphinx autoapi/viewcode render
   source verbatim into HTML, and although ``docs/conf.py`` now excludes
   ``local_settings.py`` from that scan (CR-03, 36-REVIEW.md), never build
   and serve HTML docs from a configured production checkout regardless --
   ``_readthedocs/html/`` and ``docs/_build/html/`` are both ``.gitignore``\ d,
   so nothing reaches git, but anything served from either tree is public.
4. Export ``FOMO_HEARTBEAT_URL`` in the environment the cron daemon sees
   (for example via ``/etc/environment``, or a wrapper script the crontab
   line sources) -- never as a literal value in any committed file. This
   is the ping URL the check in the previous step produced -- export it
   exactly as copied.
5. Export ``FOMO_BASE_URL`` in **both** the cron environment and the web
   server's (gunicorn/uWSGI) environment (WR-13, 36-REVIEW.md): the
   campaign-submission approval-queue link is built inside the web
   process, which reads this setting at its own settings-import time, not
   the cron process's. Simplest: set ``FOMO_BASE_URL`` once in this host's
   ``local_settings.py`` instead of an environment variable, so every
   process -- cron and web -- picks up the same value. A deployment that
   exports it for cron only gets a green preflight and unusable
   ``http://localhost:8000/...`` links in every emailed notice the web
   process builds.
6. Run the preflight check **as the account that will actually run
   unattended** (the cron user), not as root:

   .. code-block:: console

      >> python3 manage.py check_unattended

   It reports every prerequisite in one pass -- whether ``flock`` is on
   ``PATH`` *and* new enough to support ``-E`` (util-linux 2.27+, which
   the cron line's skip detection needs), whether the lock, log and
   suppression-state directories exist and are writable (this
   suppression-state check is a setup-time preflight only: it catches an
   unwritable ``FOMO_STATE_DIR`` before an operator ever installs the
   cron line, but it cannot see that directory going bad afterwards -- a
   full ``/var/lock`` tmpfs is the realistic trigger; that case is
   handled at runtime by the fallback described in step 1 above, not by
   this check (WR-17, 36-REVIEW.md)), whether the
   email backend can actually deliver and at least one staff
   user has an email on file, whether ``FOMO_HEARTBEAT_URL`` is set (and
   reminds you that the check at the other end still needs its own
   expected ping interval *and* grace time set -- the preflight can only
   see this host's environment variable, never the remote check's own
   configuration),
   whether ``FOMO_BASE_URL`` has been changed from its localhost dev
   default (WR-07, 36-REVIEW.md -- left at the default, every emailed
   admin/calendar/approval-queue link is unusable off this host), whether
   the LCO/SOAR portal API key is configured (a warning: the tick's
   ``status_refresh`` step fails on every non-terminal record for that
   facility without it), and whether at least one ``WatchedProposal`` row
   is active -- and exits
   non-zero only when a hard prerequisite (flock, the directories, or
   email) is missing; an unset heartbeat URL, a default base URL, an
   unconfigured LCO/SOAR portal API key, and an empty watched-proposal
   list are warnings, not failures (the tick still runs, and mail still
   sends, without any of the four).
   It also prints the exact cron line to install, with the real resolved
   Python interpreter and ``manage.py`` paths already filled in -- printed
   even when a hard check failed, so an operator fixing prerequisites
   still sees the target state. **The directory checks only test the uid
   that ran the command** (WR-06, 36-REVIEW.md): each ``[ok]`` names the
   uid, owner, and mode it actually tested, so running this as root while
   the cron account is unprivileged will print an ``[ok]`` that does not
   mean the cron account can write there -- run it as the cron account to
   get a result that does.

   The passing lines, the blank separator, the cron-line block, and the
   closing summary all go to standard output; a warning line and a
   failure line go to standard error instead. Each line is written once,
   to exactly one stream, so on a terminal -- where the two arrive
   together -- you see every result exactly once, in the order the checks
   ran. Redirecting only standard output to a file (a bare ``>``) still
   captures the passing lines and the cron line, but it silently drops
   every warning and failure, leaving a preflight log that looks entirely
   clean while the real problems went somewhere else. Capture the whole
   report in one file, in check order, with the same ``2>&1`` the
   committed crontab line already appends with:

   .. code-block:: console

      >> python3 manage.py check_unattended >> preflight.log 2>&1
7. Fix whatever it reports, re-running ``check_unattended`` until every
   hard check passes.
8. Copy the printed cron line into the crontab (``crontab -e`` for the
   account that should run it) -- the printed line is authoritative: it
   carries this host's resolved ``flock``, lock-file and log-file paths
   as well as the interpreter and ``manage.py``, which
   ``deploy/cron/fomo.crontab.example`` can only guess at. Starting from
   the template and hand-editing its two placeholder paths is the
   fallback if `check_unattended` cannot run on this host at all --
   confirm ``flock`` really is at ``/usr/bin/flock`` (``command -v
   flock``) and that ``FOMO_LOG_FILE`` is still at its default. The
   template's hardcoded lock path will **not** match a host that
   followed step 1 above -- edit the template's lock path to this
   host's own ``FOMO_LOCK_DIR`` before using it, or use the printed
   line instead.
9. Drop ``deploy/logrotate/fomo.example`` into ``/etc/logrotate.d/fomo`` (or
   wherever this host's logrotate scans) so the log file rotates daily and
   keeps a fortnight instead of growing forever. Writing into
   ``/etc/logrotate.d/`` typically needs ``sudo`` too.

Adding a proposal to watch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to **Django admin -> Solsys code -> Watched proposals**, add a row with
the LCO/SOAR proposal code, and leave **Active** checked. That is the
whole of what widening discovery requires -- no redeploy, and no
command-line argument. **TargetList name override** and **Attribute
records to** are optional per-row overrides (they default to the same
``<code>_targets`` naming and unattributed records the sibling
command-line ``--target-list``/``--username`` flags already produce).
Unchecking **Active** narrows discovery just as immediately, and the row's
history (``Last swept at``/``Last sweep summary``) is kept for reference.

Walking through a first tick
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is how you drive the runner against your own proposal and read each
step's report before you trust the cron line. Every command in this
section is a dry run: it writes nothing, pings no heartbeat, and mails no
one.

1. Add your own LCO/SOAR proposal code as a watched row -- see "Adding a
   proposal to watch" above; that is the only setup this walkthrough needs.
2. Run the first step, **status_refresh**, on its own:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run --step status_refresh

   A healthy dry run reports::

      step status_refresh: ok | skipped (dry run)

   A status refresh is a portal read that also writes through TOM's own
   ``post_save`` machinery, so there is no meaningful read-only variant of
   it -- the dry run for this step does not instantiate either facility,
   makes no portal call, and refreshes no status. It reports that it
   skipped rather than reporting counters.

These result lines are written by the logger to standard error, not
standard output, so redirect with ``2>&1`` if you want them captured in a
file.

``check_unattended`` (see "Setting it up on a fresh host" above) is a
separate, read-only readiness check for host configuration -- flock,
directories, email, the heartbeat URL, and so on. It is not a substitute
for this walkthrough: it tells you the host is ready to run ticks, not
what any one step's own output looks like.

3. Run **project_sweep**:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run --step project_sweep

   A healthy dry run reports::

      step project_sweep: ok | failed: 0 | LCO: created: 0, updated: 1, unchanged: 4, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0

   This dry run writes no calendar event, and it skips the one-time
   observed-telescope lookup entirely, so the ``site_lookups`` counters
   above stay at zero where a real tick would do one.
4. Run **discovery**:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run --step discovery

   Against your newly-watched proposal, a healthy dry run reports::

      step discovery: ok | swept: 1, failed: 0

   An empty watch list reports a different line that is just as healthy::

      step discovery: ok | 0 watched proposals, nothing to discover

   -- that is not a failure, only nothing to do yet. Either way, this dry
   run creates no observation record and writes no bookkeeping back to the
   watched row. Per-request preview lines -- what the sweep would create or
   reuse for each portal request -- also appear in the log, each prefixed
   with ``discovery`` and the stream (``stdout`` or ``stderr``) they came
   from.
5. Run **reconcile**:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run --step reconcile

   A healthy dry run reports::

      step reconcile: ok | runs: 3, failed: 0

   This dry run writes nothing.
6. Run **proposal_allocation**:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run --step proposal_allocation

   A healthy dry run reports::

      step proposal_allocation: ok | skipped (dry run)

   Like ``status_refresh``, this step makes no portal call and stores no
   allocation row in this mode -- it reports that it skipped rather than
   reporting counters.
7. Once each step's own output looks familiar on its own, run the whole
   tick in one go:

   .. code-block:: console

      >> python3 manage.py run_unattended --dry-run

   A healthy dry run reports the start banner, one line per step in
   registry order, and the end banner carrying the process exit code::

      === FOMO unattended run START 2026-09-22T14:00:00+00:00 ===
      step status_refresh: ok | skipped (dry run)
      step project_sweep: ok | failed: 0 | LCO: created: 0, updated: 1, unchanged: 4, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0
      step discovery: ok | 0 watched proposals, nothing to discover
      step reconcile: ok | runs: 3, failed: 0
      step proposal_allocation: ok | skipped (dry run)
      === FOMO unattended run END 2026-09-22T14:00:03+00:00 exit=0 ===

   Each line reads ``step <name>: ok | <summary>`` or
   ``step <name>: FAILED | <summary>`` -- see "What runs, and when" above
   for why a failed step never stops the later ones. The end banner's
   ``exit=`` is the process's own exit code, checkable from the shell on
   the next line:

   .. code-block:: console

      >> echo $?
      0

   A dry run is **not** automatically a success: ``project_sweep`` can
   still report a genuinely unprojectable row, and ``reconcile`` can still
   count a raising run, and either makes the whole tick exit non-zero even
   though nothing was written.

   If a step's own function raises an exception the runner did not already
   catch internally, the tick still continues, and that step's line names
   only the exception's class, never its message::

      step project_sweep: FAILED | raised ConnectionError

   The runner deliberately never lets a caught exception's message reach a
   summary, a log line, or a watched row -- a portal exception's message
   can embed request or response content.

   A note on the ``--step`` flag itself: it accepts exactly the five names
   above, and ``argparse`` rejects anything else before the runner starts
   -- running ``run_unattended --step bogus`` fails immediately, before any
   step, lock, or log line appears (the exact wording is Python-version
   specific, so it is not quoted here). Using ``--step`` on its own,
   without ``--dry-run``, still suppresses the heartbeat ping and the
   failure email -- but it does **not** suppress the work:
   ``run_unattended --step status_refresh`` on its own makes real portal
   calls and real writes, exactly like a real tick's own
   ``status_refresh`` would.
8. **What to check afterwards.** Two places, and after every dry run above
   they should show no change:

   - The watched row in **Django admin -> Solsys code -> Watched
     proposals**. Its bookkeeping columns are labelled **Last swept at**
     and **Last sweep summary**. After the dry runs above, both are
     deliberately unchanged -- that is the dry run working correctly, not
     a failure. They only move after a real discovery sweep: either a cron
     tick, or ``run_unattended --step discovery`` run without
     ``--dry-run``.
   - The calendar. A dry run adds nothing to it; new events appear only
     after a real tick.

The two failure signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Email.** Every staff user with an email on file receives a message,
subject ``FOMO unattended run failed: <step name(s)>``, listing each
failed step's own summary/counter line, the log file path, and links to
the admin and the calendar -- never a traceback, a request URL, or portal
response text. It arrives once for a newly-failing tick, then is
suppressed while the same set of steps keeps failing, with one reminder
every 24 hours for as long as it does. A single
``FOMO unattended run recovered`` message arrives once, the next time a
tick succeeds.

**Heartbeat.** Before the first step of every tick, the runner pings
``<FOMO_HEARTBEAT_URL>/start``; after the last step it pings
``<FOMO_HEARTBEAT_URL>/<exit-code>`` (``0`` on success, the tick's own
non-zero code otherwise). This catches a tick that never ran or hung, not
only one that failed outright -- either kind never reaches the
``/<exit-code>`` ping. Point ``FOMO_HEARTBEAT_URL`` at any
healthchecks-compatible endpoint (hosted or self-hosted) and configure one
check per schedule -- but the check needs two settings, not one: the
check's expected interval between pings (healthchecks.io calls this
``Period``) and its grace time (``Grace``), since other
healthchecks-compatible endpoints may spell the same two concepts
differently. Set the expected interval to 15 minutes, matching
this cron schedule; the drift-free alternative is a Cron-type check
carrying the same ``*/15 * * * *`` expression the crontab line uses,
which pegs lateness to the wall-clock slot instead of to the last ping.
Keep the grace time at about 20 minutes -- a little over one interval, so
one occasional slow tick does not page anyone -- and do not shrink it to
make the total look shorter: because the runner sends a ``/start`` ping,
the grace time also bounds the maximum allowed gap between that ping and
the completion ping, so a grace shorter than a slow tick's real runtime
would page on a healthy-but-slow tick. The service alerts at last ping +
expected interval + grace: with 15 and 20, a stopped schedule shows the
check as late about 15 minutes after the missed tick and alerts about 35
minutes after the last successful ping. Leaving the expected interval
(``Period``) at its default -- 1 day on healthchecks.io -- means the
first alert arrives about a day later while the check looks green and
correctly configured the whole time; the dead-man layer is off with no
visible sign.

When nothing has appeared
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Work through these in order:

1. **The admin's Watched Proposals list.** Is there an active row at all?
   An empty list is a healthy, quiet no-op (see "Adding a proposal to
   watch" above) -- start here before assuming anything is broken. If a
   row exists, its **Last sweep summary** column shows what its most recent
   sweep reported, success or failure.
2. **The log file** (``settings.FOMO_LOG_FILE``, ``/var/log/fomo/unattended.log``
   by default). Every tick writes a START/per-step/END banner with a
   timestamp, so a single tick is readable in isolation even without the
   heartbeat dashboard open.
3. **The heartbeat dashboard's last ping.** A missing or stale ping (older
   than the expected interval plus the grace time -- about 35 minutes
   with the recommended 15/20 settings) means the tick itself never ran
   or never finished -- check the log file next for why.
4. **A repeated "lock held" line in the log.** ``flock -n -E 99`` fails
   immediately rather than queuing, so a permanently contended cron lock
   (``FOMO_LOCK_DIR/run_unattended.cron.lock``) leaves this line on every
   tick -- but since WR-16 (36-REVIEW.md) normalized the skip back to
   exit 0, the crontab line's own exit status looks exactly like a
   healthy no-op. The log line is the only place a contended lock is
   visible at all; do not expect the exit status to tell you::

      2026-09-17T15:00:03+00:00 run_unattended skipped: lock held

   Do not confuse this with the runner's own internal-lock message,
   ``run_unattended: lock held -- skipping this tick`` -- both can appear
   in the same log with the phrase "lock held", but only the cron guard's
   line above (with the ``run_unattended skipped:`` prefix and a leading
   timestamp) is what this checklist item means. One occurrence is normal
   (an overrunning tick colliding with the next scheduled one); several in
   a row means a previous tick is stuck and needs investigating -- see
   "Repeated 'lock held' lines in the unattended log" below.

Running it by hand
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   >> python3 manage.py run_unattended --dry-run
   >> python3 manage.py run_unattended --step reconcile

Both invocations are for debugging: neither pings the heartbeat nor mails
staff, so an operator can run either without paging anyone.
``--step <name>`` accepts ``status_refresh``, ``project_sweep``,
``discovery``, ``reconcile``, or ``proposal_allocation``.

**What the locking does and does not cover.** The cron line's own
``flock -n -E 99`` (against ``run_unattended.cron.lock``) and the
runner's own internal lock (``run_unattended.lock``) together mean two
ticks never overlap -- and neither does a hand-started ``run_unattended``, including
``run_unattended --step <name>``, which takes the exact same lock. But
running one of the underlying sweep commands directly --
``backfill_lco_observations``, ``project_observation_calendar``,
``reconcile_campaign_runs``, or TOM's own ``updatestatus`` -- is **not**
locked against a tick in this release, so a direct run of one of those
commands can coincide with a tick doing the same work. When that matters,
use ``run_unattended --step <name>`` instead -- it is the exclusive route
guaranteed never to overlap a tick. The three FOMO sweep commands are
idempotent and isolate failures per item, so a coincidental overlap is
untidy (both processes doing the same work at once) rather than dangerous
-- but that is not the same guarantee as being locked against a tick, and
this paragraph does not claim it is. The per-step lock files each step
also takes internally are defence in depth behind the runner's own lock
and a reserved name for a possible future caller, not a mechanism that
protects a hand-run ``manage.py`` invocation of the underlying command
today.

.. _campaign-run-block-manual-only:

Why doesn't the calendar pop-up show an "Attributed campaign run" block?
------------------------------------------------------------------------------

Clicking a calendar entry opens a pop-up that can show an **Attributed
campaign run** block naming the run the entry is attributed to, that run's
telescope and instrument, its window and its run status, plus a link to
the campaign table anchored to that run's own row. The block is rendered
from the entry's attribution link (``CalendarEventMeta.run``) every time
the page is drawn -- it is never written into the entry's own title or
description -- so nothing that rewrites those fields (a base-layer
re-projection, a hand edit, anything) can erase it.

Before chasing a missing attribution, first confirm the pop-up opens at
all: clicking a calendar entry opens the pop-up through the Bootstrap 5
modal API, because the TOM Toolkit 3.x base page this site is built on
loads the Bootstrap 5 bundle, htmx and Alpine and no jQuery. These are two
different faults with two different fixes. A pop-up that opens but shows
no Attributed campaign run block means the entry carries no attribution
link -- the situation the rest of this section covers. A pop-up that does
not open at all, on any entry, on a day cell, or on the "+ New Event"
button, is a client-side JavaScript fault in the calendar page, not a
missing attribution. Exactly that fault was found and fixed in Phase 33
(UAT G-33-2); if it recurs, report it as a front-end regression and check
the browser console, rather than looking for a missing campaign link.

**That "View campaign ↗" link carries the run's row anchor but no page
number.** The campaign run table paginates at 25 rows, sorted by window
start descending, so a run that sorts past page 1 -- a campaign with more
runs than fit on one page -- will not be scrolled to; the browser opens on
page 1 with no matching anchor anywhere on it, and nothing highlights. An
active filter on the table can produce the same outcome on any page. If
the link does not appear to do anything, use the table's filter controls
or page forward to find the run's own row by hand. This is a known,
tested constraint (a test pins the >25-run behaviour so it cannot regress
silently), not a bug to report.

A month-view cell shows a small campaign marker on every attributed entry
whose run is publicly visible, with the campaign name as its tooltip -- the
same attribution link the pop-up block reads, rendered a second time at a
glance.

That block/marker appears only when the entry carries a companion record
whose attribution link is filled in.

**As of Phase 33, an entry the reconciler creates never carries the
campaign name in its own title** -- see "How do I get every campaign run
onto the calendar?" above for the one-time title change this caused. The
campaign now shows only through the block and the marker described above,
rendered from the attribution link, never written into the entry's
fields.

**Every entry the reconciler creates gets that link set automatically** --
via ``reconcile_campaign_runs`` and via approving a run, resolving its
site, or marking it cancelled or weather-failed, all of which now
reconcile through the same shared function (see "How do I get every
campaign run onto the calendar?" above). An entry the reconciler creates
therefore shows the Attributed campaign run block the moment it is
created, with no separate linking step. An entry ALREADY attributed to
this run through another writer never gets a second, reconciler-created
entry at all (see the skip rule in that same section above) -- its own
existing attribution is what makes the block appear.

The manual admin path below still exists, and remains the right tool for an
entry the reconciler never touches at all -- a ``load_telescope_runs``- or
sync-command-created entry that has not (yet) been attributed to a run
through the attribution queue, or a hand-created calendar entry:

1. Go to **Django admin -> Solsys code -> Campaign runs** and open the run.
2. In the **Calendar event metas** inline at the bottom, add a row and pick
   the calendar event this run's entry is attributed to.
3. Save. The pop-up for that entry now shows the Attributed campaign run
   block.

Three things to know about that inline:

* The **calendar event** field is frozen once a row is saved, because it is
  that record's identity. To point the link at a different event, delete
  the row and add a new one -- do not try to edit it in place.
* On the run's own inline, the **Attributed campaign run** field is not
  rendered at all -- the row IS the link, so there is no separate value to
  clear there. To un-attribute an entry, delete the row instead. To clear
  the value in place (which also clears the "confirmed by"/"confirmed at"
  record, since a confirmation for an attribution that no longer exists
  would be misleading), open the entry from **Django admin -> Solsys code
  -> Calendar event metas** instead -- the standalone change page for that
  record, not this inline -- and clear the **Attributed campaign run**
  value there. Either way, the entry itself, and its telescope-label
  verification history, survive untouched; nothing is deleted.
* Two further fields on that same inline, **Observation record** and
  **Observation group**, are read-only: they record which real observation
  an entry was drawn from. As of Phase 34, every LCO/SOAR observation
  projector-owned entry carries these automatically -- see "How do LCO/SOAR
  queue observations get onto the calendar?" above -- so they are filled
  in by code only; they stay blank on any entry the projector does not
  own (a reconciler entry, a Gemini entry, a hand entry).

An entry with no companion record at all, or with the attribution link
left blank, still means "not attributed to any campaign run" -- never
"needs fixing". That is the normal state for conferences, proposal
deadlines, and any un-attributed sync-command entry, and it is why those
entries show no Attributed campaign run block.

**27-UAT.md Test 9 gap closure:** when a High-band attribution-queue
candidate already exists for one of these still-unlinked events, the
pop-up now shows a "Possible campaign run match" hint naming the
candidate run and linking straight to the attribution queue (filtered to
the High band) to confirm it. This hint is staff-only -- the attribution
queue itself requires staff, and a candidate run may not yet be publicly
visible -- and an event with zero candidates (the conference/proposal-
deadline case just described) still shows nothing extra. The hint only
ever names a real match that already passed the same scoring the
attribution queue itself uses; see "How do I attribute existing calendar
events and observation records to a run?" above.

What does a run's or a campaign's public tally show?
------------------------------------------------------------------

Every run row on the public campaign table (``campaigns:table``), the
roll-up strip above that table, the ``N runs`` badge on the campaign list,
and the "Attributed campaign run" block in the calendar pop-up (see "Why
doesn't the calendar pop-up show an 'Attributed campaign run' block?"
above) all show the same read-only tally, computed from the run's linked
observation groups and records: how many distinct ``ObservationGroup``\ s
and ``ObservationRecord``\ s are linked to the run, and four night counts
-- observed, scheduled, expired-or-failed, and unused. **Anyone can see
this, staff or not** -- it is not gated behind a login, exactly like the
campaign table and the calendar itself.

Nights are counted on the site-local observing night (the same
noon-anchored rule ``telescope_runs.observing_night()`` uses everywhere
else in FOMO), never the UTC calendar date, so a night that starts late in
the evening and crosses midnight UTC is still counted once, on the night
an observer at the site would call it.

**None of this ever changes a run's own status.** The tally is a read-only
aggregate over the run's linked records; ``CampaignRun.run_status`` stays
exactly what a staff member set it to through the approval queue's
decision buttons (see "How do I mark a run cancelled or weathered-out?"
above) -- no tally computation, cache, or display tag writes to it.

The unused figure needs its own explanation, because it is computed two
different ways depending on the run:

* For a run that projects its own allocation nights (a classical or
  class-wide run with ``ALLOC:``-namespaced calendar events), the figure
  is an **exact** count of that run's still-standing allocation nights
  whose projected sunrise has already passed with nothing scheduled or
  observed on them -- the same rule that puts the ``[U]`` marker on the
  calendar (see "How do LCO/SOAR queue observations get onto the
  calendar?" above).
* For a run with no allocation events of its own (a queue-scheduled run
  whose nights the LCO/SOAR portal itself schedules), the figure is an
  **estimate**: the proposal's unused standard-time hours (the allocation
  the portal reports, minus the hours already used, fetched once per
  unattended tick -- see the fifth runner step in "How do I run everything
  unattended?" below) divided by ten hours per night, a fixed rule of
  thumb. An estimate is always shown with its estimate qualifier, never
  presented as an exact count.
* Before the proposal-allocation step has ever successfully fetched that
  run's proposal, or when the run has no allocation events and no
  proposal code at all, that run's own Progress cell or the pop-up
  block's own figure shows as **not yet known** -- never as a bare zero,
  which would read as "definitely nothing unused" rather than "nothing
  to compute this from yet". In the **campaign roll-up strip**, which
  sums across runs, a run whose own figure is not yet known is never
  counted as zero either: the strip shows the total it CAN account for,
  prefixed with the words **at least**, and names how many runs it
  could not include -- for example ``at least 2 (1 run not yet known)``.
  A total that silently dropped those runs would read as "definitely
  nothing more unused", the same false certainty a bare zero carries for
  a single run.

The estimate qualifier (the ``≈`` sign, spelled ``&approx;`` on the
roll-up strip) means something stricter than "a proposal code was looked
up": it appears only when a proposal-derived estimate actually
contributed a number to the total. A campaign whose only proposal has
never been fetched shows no estimate qualifier at all -- its figure reads
as not yet known instead, on both the run's own cell and the roll-up
strip -- because there is no estimate behind the figure to qualify. This
changed in Phase 37's gap-closure plan (G-37-6/D-20): an operator who
remembers seeing the qualifier on a total with nothing estimated behind
it was seeing the defect, not the design.

**How fresh is the tally?** A change to a linked observation record -- the
projector narrowing a queued request to a placed block, or a placed block
coming back observed -- shows **on the next page load**, with no waiting
period: the cached tally's cache key is built from the run's own primary
key plus the newest linked-record change timestamp, so any such change
produces a fresh cache key and a fresh computation immediately, for the
same reason as before. The unused figure is recomputed **on every page
load**, on every surface that shows it -- the run row's Progress cell, the
campaign roll-up strip above the runs table, the campaign-list badge, and
the calendar pop-up's attributed-run block. All three of its drivers are
immediate: an awarded night passing its projected sunrise, a staff
run-status change to cancelled or weathered, and a freshly fetched proposal
allocation. This is why the roll-up strip can never disagree with the rows
beneath it. What the cache still holds -- and what
``TALLY_CACHE_TTL_SECONDS`` (``solsys_code/campaign_tally.py``, one hour by
default) therefore still bounds -- is the linked group and record counts
and the three record-derived night counts, keyed by the newest
linked-record change stamp: a change that moves neither that stamp nor the
unused rule, such as a run added to the campaign with no linked records
yet, or a site's timezone being edited, is bounded only by that setting.
An operator who wants a shorter bound on that residual should look there.

How do I find nights that were observed but never claimed by any approved run?
-----------------------------------------------------------------------------------

The **Show Coverage Gaps** button on a campaign's runs page
(``campaigns:table``) opens a page that answers one question: of the
nights this campaign's target(s) were actually observable at a chosen
site, which ones are not accounted for by anything on the campaign's
calendar? The button is disabled, with explanatory text in its place, when
the campaign has no targets at all, or none of its runs has a resolved
site -- there is nothing to compute observability against either way.

The page computes over a requested date range: 90 days by default, capped
at 180 days even if a caller asks for more. The result is cached for one
hour (``GAP_CACHE_TTL_SECONDS`` in ``solsys_code/campaign_gap.py``); "Last
computed" on the page names when that cached result was actually
produced, not the time of the current page load.

**What claims a night** (so it does NOT show up as a gap):

* An approved ``CampaignRun``'s window, for every date in its
  ``window_start``..``window_end`` range -- unless that run's own status
  is cancelled, not-awarded, or weather/technical-failure, in which case
  its window claims nothing (a run that fell through in the real world
  frees its dates back up as a gap).
* **As of this phase, also** an observed (``[O]``) or scheduled (``[S]``)
  observation block on the campaign's calendar, on the site-local
  observing night it falls in -- whether or not that observation is
  attributed to any ``CampaignRun``. "On the campaign's calendar" means
  either attributed to one of the campaign's runs, or the record's own
  target belongs to the campaign, so a classical or queue-scheduled
  observation of the campaign's own target counts even before anyone
  works through the attribution queue. The page's "Claimed nights" list
  marks each date with which kind of thing claimed it, when both are
  known.

**What does NOT claim a night:** a queued request's window (a request
that has not yet been placed by the scheduler is not a set of owned
nights), and an expired, cancelled, or failed observation record's
window.

**Site assignment for an observation.** An observation's own recorded
observed site takes priority; failing that, the site of the
``CampaignRun`` it is attributed to; failing that, the observation cannot
be assigned to any site at all. That last case is reported on the page as
a **claimed, site unknown** count -- listed, never silently dropped --
and it closes no gap for any specific site, because the page cannot tell
which site's gaps it would have closed.

The page also carries forward the needs-review lists that predate this
phase: runs with no date at all (**undated**), approved runs not
attributed to either campaign target on a multi-target campaign
(**unattributed**), and space-mission runs whose window has not yet
narrowed to a single night (**pending narrowing**) -- none of these are
counted as claiming any date, and all three are listed so a data-quality
problem is visible rather than silently absorbed into the gap count
either way.

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
     - ``<filepath>`` (positional), ``--campaign <name>``, ``--dry-run`` (both optional)
     - Ingest a classical-schedule text file into one CampaignRun per line; the allocation projector draws the per-night CalendarEvents.
   * - ``project_observation_calendar``
     - ``--proposal <A,B>``, ``--facility <LCO|SOAR>``, ``--dry-run`` (all optional)
     - Backstop sweep: re-project LCO/SOAR ObservationRecords onto the calendar.
   * - ``backfill_lco_observation_records``
     - ``--proposal <code>``, ``--name-prefix <str>`` (both required); ``--campaign <name>``,
       ``--username <user>``, ``--create-missing-targets``, ``--dry-run`` (optional)
     - Backfill ObservationRecords for LCO RequestGroups submitted outside FOMO.
   * - ``backfill_lco_observations``
     - ``--proposal <code>`` (optional -- omit to sweep every active Watched proposal
       row), ``--created-after``/``--created-before``, ``--username <user>``,
       ``--target-list <name>``, ``--dry-run`` (optional; the four non-``--dry-run``
       flags require ``--proposal``)
     - Campaign-agnostic backfill; bare invocation sweeps the admin-editable watched-proposal list (the discovery step of :ref:`unattended-operation`).
   * - ``sync_gemini_observation_calendar``
     - (none)
     - Sync every Gemini ToO ObservationRecord to CalendarEvents.
   * - ``import_campaign_csv``
     - ``--campaign <name>`` (required), ``<filepath>`` (positional)
     - Bootstrap-import a campaign coordination CSV into CampaignRun rows.
   * - ``repair_stale_campaign_run_sites``
     - ``--dry-run`` (optional)
     - One-off re-resolution of approved CampaignRuns whose site never resolved.
   * - ``reconcile_campaign_runs``
     - ``--dry-run`` (optional)
     - Idempotent sweep projecting/refreshing CalendarEvents for every CampaignRun.
   * - ``cutover_classical_allocations``
     - ``--dry-run`` (optional)
     - One-time cutover: converts legacy blank-url classical CalendarEvents into CampaignRuns plus ALLOC:-keyed events.
   * - ``run_unattended``
     - ``--dry-run``, ``--step <name>`` (both optional)
     - The unattended runner: status refresh, projector sweep, discovery, reconcile, and proposal-allocation refresh in one cron-scheduled tick. See :ref:`unattended-operation`.
   * - ``check_unattended``
     - ``--send-test-email`` (optional)
     - Read-only preflight for the unattended path; prints the exact cron line to install. See :ref:`unattended-operation`.

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
window for a site (``load_telescope_runs``, ``reconcile_campaign_runs``,
and any future projection over that Observatory) will fail with an error
like this, observed running a real backfill against the dev database.
``project_observation_calendar`` computes no sun event at all, so it
cannot produce this error:

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

* ``load_telescope_runs`` skips and logs any schedule line it cannot
  parse, or whose telescope name doesn't resolve to a known
  ``Observatory``, or whose classical status word is unrecognised (all
  three caught as ``ValueError``/``Observatory.DoesNotExist``), and --
  caught by its own dedicated clause, ahead of that same catch -- an
  ``Observatory`` whose ``timezone`` field holds a malformed IANA name:
  resolving it raises ``ZoneInfoNotFoundError``, which subclasses
  ``KeyError`` rather than ``ValueError`` and so needs its own handler.
  Either way the command reports a ``skipped: N`` count in its final
  summary line, e.g.::

      Line 1: invalid Observatory.timezone 'America/Santigo' for site 'NTT' (obscode '809'): 'No time zone found with key America/Santigo' (line text: 'NTT EFOSC2 allocation 9-13 July')
      Done. lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1, skipped_collision: 0

* ``project_observation_calendar`` counts, rather than skips, a record it
  cannot project under ``unprojectable`` -- whether its own fields could
  not be turned into event values at all (for example, an unparsable
  request window), or the write itself failed once attempted (for example,
  a pre-existing duplicate calendar-event URL). Either way the record's
  event is left exactly as it stood before this sweep -- no new event, and
  a pre-existing one untouched -- until the underlying data problem is
  fixed and the sweep re-run. A record whose one-time observed-telescope
  lookup has not (yet) succeeded is a different, non-failure case: it keeps
  its coarse aperture-class label (never a fallback marker -- there is no
  ``[UNVERIFIED]`` in this vocabulary) and is counted under
  ``site_lookup_failed``, retried automatically on the next sweep.

* ``reconcile_campaign_runs`` catches any exception a single run's
  reconciliation raises (for example, the Observatory-timezone gap above) at
  the batch-loop level only, reports it as ``Run pk=N: reconcile failed
  (...) -- skipping`` on stderr, and continues to the next run, never
  aborting the whole sweep.

A reported classical-schedule identity-key collision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``load_telescope_runs`` matches a schedule line to its ``CampaignRun`` by a
deterministic key built from the resolved telescope, instrument, the
run's own stored observing-night window and its two sub-night tokens.
Two lines in the same file that yield an identical key are a collision,
reported on stderr naming both line numbers::

   Line 14: source_identifier 'CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-12:BoN:EoN' already claimed by line 9 -- skipping (line text: 'NTT EFOSC2 confirmed 9-12 July')

**Cause:** two proposals share the same telescope, instrument and set of
observing nights, and neither line names a proposal token to
disambiguate them -- this is exactly the real collision Phase 31's
identity spike found in a real schedule sample.

**Fix:** add a bracketed proposal token to one (or both) of the two lines
-- e.g. ``NTT EFOSC2 confirmed 9-12 July [0110.C-0234]`` -- then re-import
the file. The skipped line is never silently merged into the first; it is
reported and dropped until the collision is resolved.

On the legacy cutover path, ``cutover_classical_allocations`` guards
against the identical collision (NF-14, 35-REVIEW.md), but reports it
differently: it groups blank-url events by their own recovered
``Source line:`` text first, so the collision is between two GROUPS, not
two file lines, and it names the derived key ignoring the status word as
the cause -- e.g. an ``allocation`` line and a ``cancelled`` line for the
same telescope, instrument and window resolve to the same key. It is
reported per unexplained event, under the ``duplicate_identity`` reason
(see "How do I run the one-time classical cutover?" above and "A
reported unexplainable event during the classical cutover" below), never
as a separate skipped-line counter -- and the remedy is NOT the same:
``load_telescope_runs`` reads a schedule file the operator can edit and
re-import, while ``cutover_classical_allocations`` reads no such file, so
its remedy is to edit the affected events' description ``Source line:``
text to disambiguate the two groups in the Django admin, then re-run.

A reported unexplainable event during the classical cutover
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cutover_classical_allocations`` exits non-zero and prints a line per
event it could not convert::

   pk=334 ('tmp'): no parseable Source line: marker

**Cause:** the event's description has no recoverable ``Source line:``
marker at all, or the marker's text does not parse (an unknown telescope,
a malformed schedule line), or the resolved site has no timezone set, or
the group's own events disagree on their campaign, or the event is
already attributed to a different ``CampaignRun``, or the reason is
``key_collision`` -- the event's derived observing night is already
claimed, either by another event in this same run or by a
``CalendarEvent`` row that already holds the derived
``ALLOC:{run_pk}:{night}`` url (typically because a rewritten
``load_telescope_runs`` import of the same schedule file already ran) --
or the reason is ``duplicate_identity`` -- either this event's own
group's ``Source line:`` resolves to the same run identity key as an
earlier group's, because the key ignores the line's status word, or a
``CampaignRun`` row already holds the derived identity key whose own
stored ``Source line:`` marker is absent or differs from this event's
own group's line -- or the reason
is ``window_mismatch`` -- the event's own independently-derived observing
night falls outside the window its own schedule line implies. See "How do
I run the one-time classical cutover?" above for the full reason
vocabulary.

**Fix:** resolve the listed event in the Django admin -- correct the
description's ``Source line:``, fix the telescope name, set the
``Observatory``'s timezone, clear the conflicting attribution, or -- for a
``key_collision`` -- find and delete or re-attribute the duplicate row (the
printed reason names the colliding pk when the url is already held
elsewhere), or -- for a ``duplicate_identity`` -- when the two Source
lines differ, edit the affected events' description ``Source line:``
text to disambiguate the two groups in the Django admin; when no marker
is recoverable at all, restore or correct the claimant run's
``observation_details`` ``Source line:`` text in the Django admin so it
matches, or disambiguate the two lines, or -- for a
``window_mismatch`` --
correct the event's stored start time or the schedule line's date range
so the two agree, as the printed reason names -- then re-run the
command. It is safe to re-run: already-converted events drop out of the
candidate set, so only the still-unexplained rows are reported again,
and a repeat pass converts nothing it has not already explained and
updates an existing ``CampaignRun`` only when that run's stored
``Source line:`` matches the line being converted -- reporting anything
else instead, because this command cannot prove an unmarked or
differently-marked claimant came from the line in hand.

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
in the re-import gotcha note above. More generally, and independent of
whether a site is preserved: as of this phase (D-04) a row's already-stored
non-blank ``telescope_class`` is never overwritten by a re-import, even when
the row's own cell derives a genuinely different, non-blank class -- see the
paragraph on ``telescope_class_preserved`` in the re-import gotcha note
above.

The tick reports success but nothing new has appeared on the calendar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** the discovery step's watched-proposal list (Django admin ->
Solsys code -> Watched proposals) is empty, or every row is inactive.
This is the expected, healthy quiet no-op D-08 describes -- an empty list
sweeps nothing and reports no failure -- not a broken discovery step.

**Fix:** add an active row for the proposal code you expect to see
discovered (see "Adding a proposal to watch" in
:ref:`unattended-operation` above); nothing else needs restarting.

Repeated "lock held" lines in the unattended log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** ``flock -n -E 99`` fails immediately rather than queuing, so
every tick that finds the cron guard's own lock
(``FOMO_LOCK_DIR/run_unattended.cron.lock``) already held exits 99 and
the crontab line's tail writes a skip line instead of running::

   2026-09-17T15:00:03+00:00 run_unattended skipped: lock held

This line means the tick genuinely did not run at all -- it is gated on
flock's dedicated exit code 99, so a tick that ran and then *failed*
(``run_unattended`` exits 1) is never mislabeled as "lock held" here; look
for that tick's own START/END banner in the log instead. A single "lock
held" occurrence is normal -- one tick overran its own 15-minute window
and collided with the next scheduled one. Several occurrences in a row
mean a previous tick is genuinely stuck (for example, blocked on a slow
portal response) and never released the lock.

Flock's own exit code 99 is still what the skip line above is gated on --
that has not changed. What has changed (WR-16, 36-REVIEW.md) is what
happens to that 99 next: the crontab line's tail now sets ``rc=0`` once
it has logged the skip line, normalizing the line's own reported status
back to exit 0. So the crontab line's own exit status is only ever
``run_unattended``'s -- 0 healthy, 1 failing -- and never a code
``run_unattended`` itself cannot produce. Anything reading the line's
exit status (cron's own syslog line, a systemd ``OnFailure=`` hook, a
monitoring wrapper) therefore no longer records a routine tick overlap
as a failure -- ``run_tick()``'s own contract is that lock contention is
NOT a failure, and the heartbeat (D-12) is the structural backstop for a
lock that stays contended.

**Fix:** a ``flock`` is always released by the kernel when the holding
process exits -- including a crash, a ``SIGKILL``, or an OOM kill -- so a
dead process never leaves a lock held (WR-14, 36-REVIEW.md). Several
"lock held" lines in a row therefore always mean a tick is *still
running*, never a stale lock file: find it with ``pgrep -af
run_unattended`` and investigate why it is stuck (for example, a slow
portal response) before assuming discovery or reconciliation is broken.
Deleting ``run_unattended.cron.lock`` while a tick is genuinely live does
not help -- the next cron invocation just creates a new inode and takes
its own lock -- and disables the cron guard until the next tick, since
the runner's own internal lock (``run_unattended.lock``, released
automatically when its process exits) is the only thing then still
preventing an overlap. The heartbeat's alert window (expected interval +
grace; see "The two failure signals" in :ref:`unattended-operation`
above) is the structural backstop for exactly this case -- a permanently
contended lock eventually alerts there too.

A failure email arrived once, then went quiet while the problem continued
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** this is the D-11 suppression rule working as designed, not a
lost alert. The runner mails staff once for a newly-failing set of steps,
then suppresses repeat mail while the exact same set keeps failing, with
one reminder every 24 hours for as long as it does. A ``FOMO unattended
run recovered`` message arrives once, the next time a tick succeeds.

**Fix:** nothing to fix in the mail path itself -- check the log file or
the admin's ``Watched proposals``/``last_run_summary`` for the failure
that is still ongoing. If a *different* step starts failing while the
first is still failing, that is reported as a new failing set and mails
immediately, without waiting for the 24-hour reminder interval.

The log shows the suppression state could not be persisted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** ``FOMO_STATE_DIR`` (default ``FOMO_LOCK_DIR``) became
unwritable or full sometime after ``check_unattended`` passed, so the
tick could not write the D-11 suppression-state file to its usual
location and instead wrote it to a per-deployment, per-uid fallback file
(``fomo-unattended-state.<uid>.<deployment-tag>.fallback.json``) in the
system temp directory (WR-17, CR-05/WR-33, 36-REVIEW.md). Nothing is
actually broken by this: the tick itself is not failed by it, no mail was
lost, and no duplicate mail was sent -- the suppression decision
persisted, just somewhere else. The presence of that fallback file in the
temp directory is itself the signal that the primary directory needs
attention.

**Fix:** restore the primary directory -- free up space, or restore
write permission and ownership for the account cron runs as -- and
re-run ``check_unattended`` as that account (not as root) to confirm.
The next tick that writes the primary state file successfully deletes
the fallback by itself, so there is nothing to clean up by hand. Do
**not** delete the fallback file while the primary location is still
unwritable -- it is the only record of the current suppression state,
and removing it makes the very next tick decide "newly failing" again
and re-mail the failure that is already being tracked.

The heartbeat alerts even though the log shows a healthy tick
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** the heartbeat ping itself failed (a network blip, or the
healthchecks-compatible endpoint being briefly unreachable) -- by design,
a ping failure is logged and never fails the tick, so the log's own
START/step/END banner can show every step succeeding on the very tick the
heartbeat service never heard from.

**Fix:** this is a false alarm about the tick's own health, but a real
signal that the heartbeat path itself needs checking -- confirm
``FOMO_HEARTBEAT_URL`` is still correct and the endpoint is reachable from
this host. If the log confirms the tick ran cleanly, there is nothing to
fix on the FOMO side.

The heartbeat never alerted although the schedule stopped
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** the check's expected ping interval (``Period`` on
healthchecks.io) is still at its default -- 1 day -- rather than the 15
minutes the schedule runs at. Because the service alerts at last ping +
expected interval + grace, setting only the grace time leaves the first
alert about a day out; the check stays green for that whole time, so the
absence of an alert is not evidence that the tick ran.

**Fix:** set the expected interval (``Period``) to 15 minutes -- or
switch the check to Cron type with ``*/15 * * * *`` -- and leave the
grace at about 20 minutes; see "The two failure signals" above. With the
crontab line disabled you should then see the check go late about 15
minutes after the missed slot and alert about 35 minutes after the last
ping. Standing check: compare the interval the check is configured with
against the 15-minute cron schedule -- they must match.

The heartbeat went down after a reboot and no email arrived
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** the identifying signature is the symptom pair -- the
healthcheck goes down *and* no failure email arrives. The absence of an
email here is expected, not a second fault: the tick died before Python
ever started, so the runner's own failure-notification path never ran to
send one. The heartbeat is structurally the only signal this failure can
produce.

The trigger is the lock directory: the host rebooted, the directory under
``/run/lock`` went with the tmpfs, and every subsequent tick failed at its
very first step. On the real incident this documents, a kernel-update
reboot wiped the directory and 100 consecutive ticks were lost over about
25 hours before anyone noticed -- calibrate how long this can hide against
that number.

These are two forms of the same root cause, not two separate problems --
the first appears while the crontab's own lock path still points at the
wiped directory, and the second is what replaces it once the crontab line
alone has been repaired but the settings have not::

   >> grep 'flock: cannot open' /var/log/fomo/unattended.log
   >> grep 'PermissionError' /var/log/fomo/unattended.log | grep 'Errno 13'

A hit on the first grep proves the crontab's own ``flock`` guard cannot
open its lock file at all -- the tick never even reached Python. A hit on
the second proves the crontab line itself has already been pointed at a
working path, but ``command_lock()`` still cannot create the configured
``FOMO_LOCK_DIR`` because it sits inside root-owned ``/run/lock`` and the
cron account has no permission there.

**Fix:** point ``FOMO_LOCK_DIR`` and ``FOMO_STATE_DIR`` at durable storage
and change the crontab's ``flock`` path to match -- all three, because
fixing any subset leaves one of the two failure forms above in place.
Then re-run ``check_unattended`` as the cron account, not as root, to
confirm. See "Setting it up on a fresh host" in :ref:`unattended-operation`
above for the durable-path recommendation and the reasoning behind it.

The unused figure says it is not yet known
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** either the **proposal_allocation** unattended step has never
successfully fetched this run's proposal from the LCO Observation Portal
yet (check the log, or the admin's ``Proposal time allocations`` list, for
a row matching the run's proposal code), or the run itself has a blank
``proposal_code`` and no allocation events of its own to count exactly --
there is nothing this figure can be estimated from either way.

**Fix:** for a queue-scheduled run, confirm the run's ``proposal_code`` is
set (the classical loader's bracketed ``[proposal]`` token populates it
automatically; a run imported another way may need it set by hand) and
that the **proposal_allocation** step is completing successfully on the
tick -- see "How do I run everything unattended?" above. For a run that
projects its own allocation nights, this figure should never read "not
yet known" at all; if it does, confirm the run actually has ``ALLOC:``
events (see "How do LCO/SOAR queue observations get onto the calendar?"
above).

See also
-----------

* The :ref:`command cheat-sheet <command-cheat-sheet>` above for exact flag
  syntax.
* :doc:`/design/telescope_runs_calendar` for the astronomy and data-model
  rationale behind these commands.
* :doc:`/notebooks/pre_executed/campaign_lifecycle_demo` for the full
  campaign-lifecycle walkthrough -- a worked, pre-executed example of the
  submission, approval, site-resolution and attribution steps this runbook
  describes.
