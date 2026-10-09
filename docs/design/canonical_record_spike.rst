Canonical-Record Spike
=======================

This document records the investigation spike that settled how FOMO's calendar and
``CampaignRun`` model connect to each other for v2.2's canonical-record and reconciler
work. It was written after a live investigation (2026-07-27) that read the real dev
database, applied a throwaway migration to a disposable copy of it, and measured the
actual behaviour of the existing calendar-sync code rather than reasoning from
documentation alone. No ``CampaignRun`` schema migration, no reconciler module, and no
attribution UI was built during this spike — the deliverable is this durable summary and
its full-detail companion, ``26-DECISION.md`` (originally at
``.planning/phases/26-canonical-record-spike/26-DECISION.md``; this project's
milestone-archival workflow moves completed phase directories to
``.planning/phases-archive/`` once their milestone closes, so check there first if the
original path no longer resolves).

Background
----------

Today, a calendar event created by any of FOMO's sync commands (the classical-night
importer, the LCO queue sync, the Gemini queue sync) has no stored link back to the
``CampaignRun`` that actually produced it — telling which run "owns" a given calendar
row requires matching telescope, instrument and date by hand. Before Phase 27 migrates
the schema, Phase 28 builds a staff-facing tool for connecting existing calendar rows to
their runs, and Phase 29 writes a single reconciler command that keeps every run's
calendar events up to date, this spike settled four concrete questions against the real
dev database so none of that downstream work has to re-derive them from scratch:

* How to record which pipeline created a ``CampaignRun`` (its ``source``), and how that
  interacts with the existing rules that stop two runs from being accidental duplicates.
* How each existing sync command's calendar event maps back to a ``CampaignRun``.
* What key the reconciler should use to create or update a calendar event, and whether a
  run allocated to a whole telescope class (rather than one site) should produce one
  event per candidate site or a single shared event.
* How to rename the existing per-event telescope-label record so it can also carry the
  new run link, and every place in the codebase that rename touches.

Domain correction: queue windows are not sets of owned nights
---------------------------------------------------------------

**Recorded 2026-07-27, after this spike's measurements were taken, from the project
owner (a professional astronomer) — read this before the Decisions section below, which
it qualifies.**

There is a fundamental difference between a **classically scheduled run** — a specific
set of nights at one telescope, each with its own known start and stop time — and a
**queue-scheduled run**. ESO, SOAR and Gemini are queue-scheduled, and queue-scheduled
*networks* of telescopes like LCO especially so: a queue run's window is the span of
time (up to a full six-month semester) during which an observation *could* happen, not a
set of nights the run owns outright. For a queue run, a night inside that window with no
recorded observation is the **normal, correct state** — it is not a gap that is missing
or needs to be filled in.

This directly affects the one real run this spike's comparison was built against:
``CampaignRun`` pk 1 (FTS/MuSCAT4) is an **LCO queue run**, not a classical one. Its
15-night window is the span in which LCO's own scheduler could have placed an
observation; the 11 real calendar events already on the calendar are what it actually
scheduled. The 4 remaining nights in that window are not "uncovered" — they are nights
the queue scheduler simply did not use, exactly as expected for this kind of run.

**Queue-run projection — settled 2026-07-27, closing the one question this correction
had left open:** a queue-scheduled run gets a single whole-window calendar entry, and
its already-scheduled  or already-observed nights keep showing up as their own separate,
more detailed calendar entries exactly as they do today — see Key finding below for the
full verdict.

Two consequences that carry through the rest of this page:

* The event-key scheme below (``RUN:{run_pk}:{date}``, one key per observing night) is
  the right fit for a **classically scheduled** run, where the run genuinely does own a
  specific list of nights. A **queue-scheduled** run instead gets a single whole-window
  calendar entry (keyed on the run alone, no date), coexisting with the individual
  calendar entries its real observations already produce as they are scheduled and
  observed — see Key finding below.
* The very large fan-out number quoted further down for a class-wide queue allocation
  (roughly 400 events for one run) does not survive this correction: a class-wide queue
  allocation gets the same single whole-window entry as any other queue-scheduled run,
  not 400 events and not 80.

``CampaignRun.source`` already distinguishes classical runs from LCO-queue, Gemini-queue
and other run types, so the reconciler has what it needs to treat these differently —
that part of this spike's work is unaffected by this correction.

Key finding
-----------

**All four original questions are now settled with a firm answer, for both classically
scheduled and queue-scheduled runs.** A queue-scheduled run — including ``CampaignRun``
pk 1, the one real run this spike measured against — gets a single whole-window calendar
entry keyed on the run alone (no date), and its real observations keep showing up as
their own separate calendar entries exactly as they do today, narrowing and refining as
they are scheduled and then observed: nothing new needs to be built for that second half,
only left alone. One question remains genuinely open, unrelated to run type: how the
reconciler should treat an *already-existing* calendar entry it might otherwise want to
rewrite (the write-strategy question below) — both candidate answers to that produce an
identical calendar, and the deciding factor between them only fully resolves once a
later milestone rewires the existing sync commands.

Decisions
---------

**Where a run came from, and what needs approval**

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - ``source`` vocabulary
     - Six values: the five ways a run can be created (web submission, classical file,
       LCO queue, Gemini queue, CSV import) plus ``LEGACY`` for the 31 runs that predate
       this tracking. The three adapter-produced values are declared now but not written
       by any code path yet — that starts with a later milestone's adapter rewiring.
     - 27
   * - Existing duplicate-prevention rules
     - Confirmed unaffected: with the new ``source`` and ``telescope_class`` fields
       actually added to a real copy of the database, both existing rules that stop two
       runs from being accidental duplicates still fire exactly as before. Which run
       "owns" a calendar row is tracked by the new run link (see below), not by either
       duplicate-prevention rule.
     - 27
   * - Approval rule
     - A run counts as approved and needing no further review when its
       ``approval_status`` is ``APPROVED`` and its ``source`` is anything other than
       ``WEB`` — that combination means *no approval was required*, not *a human
       approved this*. No new approval value is added; every place that already reads
       ``approval_status`` today would otherwise need updating for one, for a
       distinction ``source`` already carries.
     - 27

**The reconciler's calendar-event key and what it owns (classically scheduled AND queue-scheduled runs)**

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - Event key
     - For a run that owns a specific list of nights (a classically scheduled run), the
       reconciler creates or updates each calendar event under the key
       ``RUN:{run_pk}:{date}``, where ``{date}`` is always the observing night as it
       would be read on a calendar **at the telescope's own site**, not the UTC date of
       whatever timestamp the current stage happens to produce. This matters because
       these can genuinely differ: one of the real events checked during this spike has
       a UTC timestamp that falls on 8 July but is already the night of 9 July at its
       site. Using the wrong date would silently create a duplicate event and orphan the
       original on the next reconcile run. **For a queue-scheduled run, the reconciler
       instead creates or updates one whole-window calendar entry under the key
       ``RUN:{run_pk}`` (no date at all), while its real observations continue to appear
       as their own separate calendar entries produced by the existing LCO/Gemini sync
       commands, narrowing and refining as they are scheduled and observed. Both key
       forms are settled and stable: neither changes as a run moves through the four
       pipeline stages.**
     - 27, 29
   * - Ownership rule
     - A calendar event belongs to the reconciler only if it carries the new companion
       run-link record with the link actually set. No companion record, or one with the
       link left blank, means "not mine, never touch." This is already true for every
       classical-night calendar event in the real database today, since none of them
       have a companion record at all.
     - 27, 29
   * - Class-wide runs
     - A run allocated to a whole telescope class (rather than one resolved site)
       produces a **single class-wide event**, not one event per candidate site and not
       one event per day. Every real class-wide run in the current data is itself a
       queue-scheduled run, so it takes the settled queue-run form above: one
       whole-window entry for the whole run, not the roughly 400 events (or even 80) that
       fanning out per site and per day would have produced. The event narrows to the
       real site once an actual observation record for that run appears.
     - 29
   * - Genuinely site-less runs
     - A run with no ground site resolved at all gets **one event spanning its whole
       window**, not one event per day — honestly showing "sometime in this window"
       rather than claiming every day in it as an observing day. This keeps the
       calendar consistent with the campaign gap-analysis tool, which already refuses to
       count those dates as claimed.

       **Correction (2026-07-29, Phase 27 D-11):** this row originally described the
       genuinely-site-less case as "a space telescope" without qualification. That premise
       is false: space observatories resolve to a real ``Observatory`` like any ground
       site, via an MPC obscode or the JPL Horizons observer-notation alias table at
       ``solsys_code/campaign_utils.py`` — ``Observatory`` already holds ``274`` (JWST),
       ``289`` (Roman) and ``C51`` (WISE). The genuine exception is a space observatory
       with a Horizons code but **no MPC obscode assigned at all** — JUICE
       (``500@-28``); Swift has ``C52``, HST has ``250``, JWST has ``274``. Phase 27's
       ``telescope_class`` vocabulary is therefore the narrower ``2m0``/``1m0``/``0m4``/
       ``SPACE`` set (``SPACE`` meaning specifically "a space observatory with no MPC
       code assigned"), not 26-DECISION.md Criterion 3's originally recommended
       three-meaning scheme. "Unresolved" is deliberately **not** a ``telescope_class``
       value — ``site_needs_review`` already carries that meaning.
     - 27, 29
   * - Allocated-but-unscheduled runs
     - A run with no window start yet produces no calendar event at all, but the
       reconciler counts and reports it in its summary rather than skipping it silently —
       the same way the CSV importer already reports runs whose site needs review.
     - 29

**How this connects the existing 11 real LCO calendar events, and what stays open**

Both measured options below start from the same real starting point: ``CampaignRun``
pk 1's 15-night observing window, 11 of whose nights already have a real calendar event
from the existing LCO queue sync command. **Because pk 1 is itself a queue run (see the
domain correction above), read the "4 newly created" nights in both rows below as the
nights LCO's queue simply did not schedule an observation on — not as coverage gaps that
necessarily should be filled with a calendar entry.** The measured write-behaviour
numbers themselves (what happens to the 11 existing events under each option) are still
valid evidence for Phase 29's decision; only the framing of the 4 new nights as
"uncovered" is corrected.

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Option
     - What happens to the calendar
     - Write behaviour (measured)
     - Trade-off
   * - Adopt
     - The 15-night window shows 15 events: the 11 existing LCO events plus 4 newly
       created for the nights LCO's queue did not schedule.
     - The reconciler updates all 11 existing events in place; creates 4 new ones; stable
       on a repeated run (no further changes).
     - Makes the reconciler responsible for the whole window right away, but see the
       write-conflict finding below.
   * - Gap-fill
     - Also 15 events total — identical calendar to Adopt.
     - The reconciler never touches the 11 existing events; creates only the 4 events for
       the unscheduled nights; stable on a repeated run.
     - The 11 existing nights keep being updated by the existing LCO sync command
       instead of the reconciler, until a later milestone folds that responsibility in.

**Both options were fully measured, and both are viable — this is not a case where one
measurement won and the other lost.** The calendar looks identical either way. The
deciding factor is a write-conflict risk that only affects Adopt: the existing LCO sync
command re-writes an event's fields (including a "last modified" stamp) every time it
runs and finds any field has changed, with no awareness of the reconciler. Under Adopt,
this means the LCO sync command would overwrite the reconciler's own stamp on those 11
rows on its next run, and the reconciler would then re-stamp them right back on its own
next run — a repeating overwrite-and-restamp cycle every reconcile cycle, not a one-time
transitional cost. Under Gap-fill this cannot happen, because the reconciler simply never
writes to those 11 rows.

**This choice is deliberately left open for Phase 29** to decide with the evidence above,
rather than locked here. The condition that resolves it: once a later milestone rewires
the LCO sync command so it no longer writes to these rows directly (folding that
responsibility into the reconciler instead), the write-conflict risk that makes Adopt
worth worrying about today disappears, and Phase 29 should record its own decision at
that point. **A second, separate question — whether a queue run like pk 1 should have
per-night calendar entries minted for its unscheduled nights at all — is now settled:
no. A queue run instead gets a single whole-window entry, and its unscheduled nights
stay exactly as unscheduled, with no calendar entry minted for them at all. See the
Domain correction section above for the settled verdict.**

A third option — the reconciler always creating a fresh event for every night in the
window regardless of what already exists — was measured too, but only as a rejected
baseline: it produces 26 events for this one run (11 originals plus 15 new ones,
including a second, duplicate event for every one of the 11 already-covered nights) —
the concrete, counted version of the double-booked calendar the attribution work (Phase
1)  exists to prevent.

**Migration and the rename checklist**

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - Migration shape
     - Proven against a real copy of the database, not just proposed: rename the
       existing per-event telescope-label record to ``CalendarEventMeta``, then add the
       new run-link field to it and the two new fields to ``CampaignRun``. Written by
       hand rather than auto-generated, because an auto-generated migration cannot tell
       a rename from "delete the old table, create a new one" — and since the telescope-
       label record's own event link *is* its primary key, that would have dropped the
       table and lost all 11 real rows. Row counts before and after the migration were
       identical.
     - 27
   * - Rename checklist
     - Six places actually needed a matching update, not the four originally expected:
       the admin registration and the LCO sync command (both failed loudly, as expected,
       with an import error at startup); two more that weren't on the original list —
       the admin page's own URL name (which Django derives from the model's old class
       name) and the class name as used directly inside four test files. The calendar
       page's own display code needed no change at all, because the underlying link name
       between an event and its telescope-label record was deliberately left unchanged.
     - 27

Timezone gap found during this spike
-------------------------------------

**A concrete Phase 27 prerequisite, not a documentation nicety.** ``CampaignRun`` pk 1's
real site (Siding Spring, MPC code ``E10``) has a blank timezone field in the current
observatory records. This spike's prototype substituted the site's known real-world
timezone so the site-local-observing-night comparison above could still run, but a real
reconciler must not do that silently — ``solsys_code/telescope_runs.py``'s own
``sun_event()`` already raises an error rather than guessing when a site's timezone is
blank, and the reconciler's ``RUN:{run_pk}:{date}`` key depends on the same site-local
date derivation. **Phase 27 should backfill the timezone for this observatory record
before the reconciler ships** — this is the single most actionable, concrete finding
this spike produced, separate from any of the vocabulary or key-scheme decisions above.

Future scope
------------

See ``26-DECISION.md`` (path note above) for the full evidence each of these decisions
rests on — including the real constraint-coexistence test results, the per-adapter
key-construction code citations, the measured rename blast radius (which test files
needed updating and why), the exact code path (file and line) behind the Adopt
write-conflict finding, and the full domain-correction subsection with all six of its
consequences. These are recommendations for Phases 27-29 to implement, plus **one**
still-open question for Phase 29 to settle from the evidence above (the write-strategy
choice for classical runs' already-existing calendar events) — none of it is implemented
in this spike. The other question this domain correction had opened — whether, and
under what key, a queue-scheduled run should be projected onto the calendar at all — is
now settled (see the Domain correction and Decisions sections above): a single
whole-window entry per queue run, coexisting with the per-observation entries the
existing sync commands already produce.

A related documentation gap this domain correction surfaced but does not fix here: the
phase's own planning notes (its context document's write-up of the "4 uncovered nights"
prototype, and the roadmap requirement describing the 19 currently-invisible 3I/ATLAS
runs) both currently read as though every run's window is a set of nights needing full
coverage. Correcting that upstream framing is flagged as a separate follow-up task, not
undertaken by this spike.
