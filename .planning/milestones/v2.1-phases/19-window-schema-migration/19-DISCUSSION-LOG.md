# Phase 19: Window-Schema Migration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-09
**Phase:** 19-window-schema-migration
**Areas discussed:** Old-field retirement strategy, TBD/window display convention, Calendar projection during the gap, Existing duplicate-row cleanup

---

## Pending Todo Review (pre-area)

| Todo | Score | Folded? |
|------|-------|---------|
| `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — rename calendar_utils.py private helpers | 0.6 (keyword overlap only) | No — left deferred |

**Notes:** Unrelated to CampaignRun/window schema; already marked "deliberately deferred, no second consumer yet" in STATE.md.

---

## Old-field retirement strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Hard cutover | Drop obs_date/ut_start/ut_end in this same migration; update all 15 referencing files | ✓ |
| Transitional dual-schema | Keep old fields alongside new ones for a release cycle | |

**User's choice:** Hard cutover (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Split into steps | 3 separate migrations (add → backfill → drop), each independently reversible | |
| Single combined migration | One migration does add+backfill+drop together | ✓ |

**User's choice:** Single combined migration (user went against the recommended split-into-steps option)
**Notes:** None further — moved to next area after 2 questions.

---

## TBD/window display convention

| Option | Description | Selected |
|--------|-------------|----------|
| Plain "TBD" text | Literal string, no new visual component | |
| "TBD" with a visual flag (badge/icon) | Text plus badge/icon, more discoverable | ✓ (best effort) |

**User's choice:** "Try 'TBD' with visual flag, it would be nice to have some visual appeal but can drop if it doesn't work" — free-text answer, captured as best-effort with plain-text fallback if the badge/icon proves complicated.

| Option | Description | Selected |
|--------|-------------|----------|
| TBD rows first | Unscheduled runs surface at the top (need attention) | |
| TBD rows last | Scheduled runs lead the table; TBD rows fall to the bottom | ✓ |

**User's choice:** TBD rows last

| Option | Description | Selected |
|--------|-------------|----------|
| En-dash range when different | "Aug 1 – Aug 15, 2026" | |
| You decide | No strong preference | |

**User's choice:** Free-text — "Use '->' between window start and window end" (e.g. "Aug 1, 2026 -> Aug 15, 2026"), overriding both offered options.
**Notes:** No range rows exist until Phase 20's CSV import, but the rendering logic is written now per the hard-cutover decision.

---

## Calendar projection during the gap

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse Stage 1's sun_event() | Real dark-window banner via telescope_runs.sun_event() for the resolved site | ✓ (partial) |
| Midnight-UTC placeholder | Simple full-day banner, no ephemeris lookup | ✓ (partial) |
| Stop projecting in Phase 19 | Leave calendar projection disabled until a later phase | |

**User's choice:** Free-text hybrid — "Reuse Stage 1's sun_event() for Earth-based observatories that can be resolved, use midnight-UTC placeholder for space-based observatories." Combines both recommended and alternative options rather than picking one exclusively.
**Notes:** Flagged in CONTEXT.md as a narrow, early use of the ground-vs-space distinction Phase 20's ASSET-01 formalizes — Phase 19 doesn't need the full asset-aware coverage-gap rewrite, just this one projection branch.

---

## Existing duplicate-row cleanup

| Option | Description | Selected |
|--------|-------------|----------|
| Delete the duplicates | Leftover demo/UAT fixture rows, not genuine campaign data | ✓ |
| Keep both, add a disambiguating field | More complex; no real-world case needs true duplicates to coexist | |

**User's choice:** Delete the duplicates (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| Generic de-dup in the migration | Query for any collision group, keep one row per group, log removals | ✓ |
| Remove only the 2 known pairs | Explicit delete by pk; simpler but not portable | |

**User's choice:** Generic de-dup in the migration (Recommended)
**Notes:** Live dev-DB query during discussion found real evidence: pk 15/17 ("Grace Lifecycle") and pk 16/18 ("Hal Lifecycle") are fully-duplicate rows under campaign 3 ("3I/ATLAS (demo)") that would still collide under Phase 18's contact_person-based partial constraint.

---

## Claude's Discretion

- Exact partial/conditional `UniqueConstraint` SQL mechanism (Django `condition=` syntax, portable across SQLite/PostgreSQL) — technical implementation detail for planner/researcher.
- Default date format string and badge/icon visual styling (if the TBD visual flag is pursued) — left to match existing `campaign_tables.py`/template conventions.

## Deferred Ideas

None raised as new scope creep during discussion. The one weakly-matched pending todo (calendar_utils.py helper renaming) was reviewed and left deferred — see Pending Todo Review above.
