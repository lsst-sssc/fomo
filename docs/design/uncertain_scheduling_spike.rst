Uncertain-Scheduling Investigation Spike
========================================

This document records the investigation spike that settled five open design
decisions for FOMO's ``CampaignRun`` scheduling model against the real
3I/ATLAS coordination sheet (2026-07-09 snapshot). It was written after a
live investigation that read the actual, publicly-editable Google Sheet
export and probed the live local ``Observatory`` DB and MPC Obscodes API,
rather than reasoning from synthetic examples or documentation alone. No
``CampaignRun`` schema migration, no CSV importer change, and no fuzzy-match
UI code was built during this spike — the deliverable is this durable
summary and its full-detail companion, ``18-DECISION.md`` (originally at
``.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md``;
this project's milestone-archival workflow moves completed phase directories
to ``.planning/phases-archive/`` once their milestone closes, so check there
first if the original path no longer resolves).

Background
----------

FOMO's ``CampaignRun`` model coordinates community-submitted observation
plans for 3I/ATLAS, imported from a real, live, publicly-editable Google
Sheet. The real sheet's space-mission rows (JWST, HST, Swift) frequently
carry a date *range* or a still-``TBD`` observing date rather than a single
known night — the sheet's actual rows are messier than the synthetic
``campaign_sample.csv`` fixture anticipated. Before Phase 19 migrates the
schema, Phase 20 extends the CSV importer, and Phase 21 builds a
staff-facing fuzzy-match site-resolution UI, this spike settled five
concrete questions against the real sheet and the live ``Observatory`` data
so none of that downstream work has to re-derive them from scratch:

* The window field schema for representing a range/TBD observing date.
* The replacement natural key for rows sharing a still-unknown observing
  date.
* The CSV range/TBD text-parsing rules the importer needs.
* The fuzzy-match library choice for staff-facing site-code resolution.
* Whether ``resolve_site()`` correctly resolves real space-observatory MPC
  codes, and whether ``Observatory.obscode`` needs widening.

Key finding
-----------

**Window schema: confirmed as the already-locked nullable
``window_start``/``window_end`` ``DateField`` pair — no schema change needed.**
Every real cell shape in the sheet (single exact date, full-date range,
compact same-month range, and the day-unknown ``TBD`` marker) maps cleanly
onto this pair.

**Fuzzy-match library: difflib is the primary choice — no new dependency
justified by this live test.** ``rapidfuzz`` and stdlib ``difflib`` produced the
same matches (including the same two false positives) and the same clean
misses on the real messy ``Site Code`` corpus; add ``rapidfuzz`` to
``pyproject.toml`` explicitly only if a future, wider candidate pool
demonstrates a case difflib genuinely misses.

Decisions
---------

.. list-table::
   :header-rows: 1
   :widths: 22 48 12

   * - SCHED-01 criterion
     - Decision
     - Phase
   * - Window field schema
     - Nullable ``window_start``/``window_end`` ``DateField`` pair, confirmed
       against real single-date, ranged, and TBD cell shapes.
     - 19
   * - TBD-row natural key
     - Fold ``contact_person`` (existing ``CampaignRun`` field) into the
       natural key for rows where ``window_start IS NULL``, via a
       partial/conditional ``UniqueConstraint`` (exact mechanism is Phase
       19's to design) — evidenced by a real two-row JWST collision in the
       live sheet.
     - 19
   * - CSV range/TBD parsing rules
     - Extend ``parse_obs_window()``'s existing pattern-per-shape discipline
       (the same approach already used for ``_HHMM_RANGE``, ``_APPROX_HOUR``,
       and ``_BARE_HOUR_UTC``) to ``Obs. Date``, one rule per real shape;
       range-detection must inspect both ``Obs. Date`` and
       ``UT Time Range``; never raise on messy non-key fields, flag
       needs-review instead.
     - 20
   * - Fuzzy-match library
     - ``difflib.get_close_matches`` as the primary/default choice; add
       ``rapidfuzz`` to ``pyproject.toml`` explicitly only if a later, wider
       candidate pool proves it necessary.
     - 21
   * - ``resolve_site()`` / obscode widening
     - No widening of ``Observatory.obscode`` (``max_length=4``) needed —
       confirmed against the live field definition; real space-observatory
       MPC codes (``250``, ``274``, ``289``) all fit within 3 characters.
     - None (confirmed as-is)

Future scope
------------

See ``18-DECISION.md`` (path note above) for the full evidence each of these
decisions rests on — including the live rapidfuzz/difflib score comparison
against the real messy ``Site Code`` corpus, the real JWST natural-key
collision, the enumerated CSV cell shapes from the live 2026-07-09 sheet
snapshot, and the unrelated ``to_observatory()`` ``TypeError`` on
satellite-type MPC records discovered while confirming the obscode-widening
verdict. These are recommendations for Phases 19-21 to implement, not
implemented in this spike.
