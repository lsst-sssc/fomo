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

Key finding
-----------

**Three of the four questions are settled with a firm answer; the fourth (how the
reconciler should treat a night that already has a calendar event from an existing sync
command) is deliberately left open for Phase 29, because both candidate answers produce
an identical calendar and the deciding factor — whether the reconciler and the existing
LCO sync command fight over the same rows — only fully resolves once a later milestone
rewires those sync commands.** The spike's job here was to produce the measured
comparison, not to pick a winner prematurely.

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

**The reconciler's calendar-event key and what it owns**

.. list-table::
   :header-rows: 1
   :widths: 22 50 12

   * - Topic
     - Decision
     - Phase
   * - Event key
     - The reconciler creates or updates each calendar event under the key
       ``RUN:{run_pk}:{date}``, where ``{date}`` is always the observing night as it
       would be read on a calendar **at the telescope's own site**, not the UTC date of
       whatever timestamp the current stage happens to produce. This matters because
       these can genuinely differ: one of the real events checked during this spike has
       a UTC timestamp that falls on 8 July but is already the night of 9 July at its
       site. Using the wrong date would silently create a duplicate event and orphan the
       original on the next reconcile run.
     - 29
   * - Ownership rule
     - A calendar event belongs to the reconciler only if it carries the new companion
       run-link record with the link actually set. No companion record, or one with the
       link left blank, means "not mine, never touch." This is already true for every
       classical-night calendar event in the real database today, since none of them
       have a companion record at all.
     - 27, 29
   * - Class-wide runs
     - A run allocated to a whole telescope class (rather than one resolved site)
       produces a **single class-wide event per day**, not one event per candidate site.
       One real class-wide run's window, multiplied by how many sites can serve that
       telescope class, works out to roughly 400 events for a single run if fanned out
       per site — nearly all of them describing observations that will never actually
       happen there. The event narrows to the real site once an actual observation
       record for that run appears.
     - 29
   * - Space-mission runs
     - A run with no ground site at all (a space telescope) gets **one event spanning
       its whole window**, not one event per day — honestly showing "sometime in this
       window" rather than claiming every day in it as an observing day. This keeps the
       calendar consistent with the campaign gap-analysis tool, which already refuses to
       count those dates as claimed.
     - 29
   * - Allocated-but-unscheduled runs
     - A run with no window start yet produces no calendar event at all, but the
       reconciler counts and reports it in its summary rather than skipping it silently —
       the same way the CSV importer already reports runs whose site needs review.
     - 29

**How this connects the existing 11 real LCO calendar events, and what stays open**

Both measured options below start from the same real starting point: ``CampaignRun``
pk 1's 15-night observing window, 11 of whose nights already have a real calendar event
from the existing LCO queue sync command.

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Option
     - What happens to the calendar
     - Write behaviour (measured)
     - Trade-off
   * - Adopt
     - The 15-night window shows 15 events: the 11 existing LCO events plus 4 newly
       created for the genuinely uncovered nights.
     - The reconciler updates all 11 existing events in place; creates 4 new ones; stable
       on a repeated run (no further changes).
     - Makes the reconciler responsible for the whole window right away, but see the
       write-conflict finding below.
   * - Gap-fill
     - Also 15 events total — identical calendar to Adopt.
     - The reconciler never touches the 11 existing events; creates only the 4 genuinely
       missing ones; stable on a repeated run.
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
that point.

A third option — the reconciler always creating a fresh event for every night in the
window regardless of what already exists — was measured too, but only as a rejected
baseline: it produces 26 events for this one run (11 originals plus 15 new ones,
including a second, duplicate event for every one of the 11 already-covered nights) —
the concrete, counted version of the double-booked calendar the attribution work (Phase
28) exists to prevent.

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
needed updating and why), and the exact code path (file and line) behind the Adopt
write-conflict finding. These are recommendations for Phases 27-29 to implement, plus one
still-open question for Phase 29 to settle from the evidence above — none of it is
implemented in this spike.
