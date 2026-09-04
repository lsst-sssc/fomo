# Phase 32: Adapter Consolidation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-03
**Phase:** 32-Adapter Consolidation
**Areas discussed:** Pending todos to fold, Third-adapter facility target, Shared helper's lookup key, Classical proposal-code identity gap, Cutover sequencing (ADAPT-05)

---

## Pending todos to fold

| Option | Description | Selected |
|--------|-------------|----------|
| Skip sun_event for existing nights | Recommended fold — reconcile_run() will run far more often once adapters exist | ✓ |
| Extract site/telescope mapping module | Style cleanup, weak overlap | |
| Neither — leave both for later | | |

**User's choice:** Skip sun_event for existing nights
**Notes:** The retarget-ADAPT-03 todo was deliberately excluded from this list and handled as its own gray area (below) rather than a fold/no-fold checkbox, since it changes the phase's requirements text rather than adding incidental scope.

---

## Third-adapter facility target

| Option | Description | Selected |
|--------|-------------|----------|
| Retarget to SOAR | Extend sync_lco_observation_calendar for SOAR_QUEUE; matches STATE.md's "LCO/SOAR" core value and gives Phase 33 a facility with real read-back | |
| Keep Gemini as scoped | Still rewire sync_gemini_observation_calendar; document it can't support Phase 33 propagation | |
| Do both | Retarget ADAPT-03 to SOAR now AND fold Gemini's rewiring in as a 4th write path | ✓ |

**User's choice:** Do both
**Notes:** Resolves pending todo `2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` (gap G-31-3). Triggered a follow-up decision to correct ROADMAP.md/REQUIREMENTS.md text in the same session (see below) so the locked docs don't contradict this decision. ADAPT-03 retargeted to SOAR; new ADAPT-06 added for Gemini. Ship order becomes classical → LCO/SOAR → Gemini.

---

## Shared helper's lookup key

| Option | Description | Selected |
|--------|-------------|----------|
| source_identifier primary | Look up by source_identifier first, fall back to campaign+window | |
| campaign+window primary, source_identifier as guard | Preserve today's lookup, source_identifier only prevents duplicates | |
| You decide | Let the planner pick against the constraint inventory | ✓ |

**User's choice:** You decide
**Notes:** Recorded as Claude's Discretion in CONTEXT.md with a recommendation (source_identifier-primary) for the planner to confirm against 31-DECISION.md's constraint inventory.

---

## Classical proposal-code identity gap

| Option | Description | Selected |
|--------|-------------|----------|
| Ship the documented risk | Keep tolerance-only key; avoids schedule-file grammar scope creep | ✓ |
| Add proposal-code parsing now | Teach the line grammar to recognize a proposal-code token | |
| You decide | Let the planner assess feasibility | |

**User's choice:** Ship the documented risk
**Notes:** No parsing work added to load_telescope_runs in this phase.

---

## Cutover sequencing (ADAPT-05)

| Option | Description | Selected |
|--------|-------------|----------|
| Hard cutover per adapter, in commit order | Each adapter flips its write path in the same commit it ships in; no dual-write, no flag | ✓ |
| One-shot backfill command first | Convert existing CalendarEvents to CampaignRuns before any adapter flips | |
| You decide | Let the planner design against the CalendarEvent inventory | |

**User's choice:** Hard cutover per adapter, in commit order

---

## Roadmap/requirements text correction

Asked as a direct follow-up to the facility-target decision, since "Do both" left
ROADMAP.md's Phase 32 goal/success-criterion-3 and REQUIREMENTS.md's ADAPT-03 naming
only Gemini — a contradiction with the just-made decision.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, correct ROADMAP.md/REQUIREMENTS.md now | Update goal, success-criterion-3/6, ADAPT-03/ADAPT-06 text so roadmap and CONTEXT.md agree before planning | ✓ |
| No, leave roadmap text as-is | CONTEXT.md decision overrides roadmap text | |

**User's choice:** Yes, correct ROADMAP.md/REQUIREMENTS.md now
**Notes:** ROADMAP.md and REQUIREMENTS.md edited in this same session (before this log was written) — see the "Roadmap Evolution" entry added to STATE.md dated 2026-09-03.

## Claude's Discretion

- Shared write-and-reconcile helper's lookup-key priority (source_identifier vs.
  campaign+window primary) — see CONTEXT.md's Claude's Discretion section.

## Deferred Ideas

- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` —
  reviewed, not folded; left for its own quick task.
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` — reviewed, not folded;
  unrelated to adapter writes.
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — reviewed,
  not folded; unrelated to adapter writes.
