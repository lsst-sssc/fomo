---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 02
subsystem: investigation
tags: [django, campaignrun, schema-spike, uniqueconstraint, sqlite, decision-doc]

requires:
  - phase: 31-01
    provides: "Dated real dev-DB CampaignRun snapshot, campaign-FK read-path blast-radius inventory, and constructed-input constraint probe of all three D-05 candidate schema shapes"
provides:
  - "31-DECISION.md `## Recommendation` section locking SCHEMA-01 (nullable campaign FK, Option A) and SCHEMA-02 (source_identifier field + partial UniqueConstraint) with cited evidence"
  - "Phase 32's inherited obligations: the 5-site campaign-FK null-guard list, the per-ingest-path source_identifier values, and the promote-vs-add-alongside guidance"
affects: [31-03, 31-04, 31-05, 32]

actuals:
  tokens: 3624
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Recommendation-section-after-Findings-section document ordering (26-DECISION.md precedent), using #### for internal subheadings so later plans' ### evidence/verdict sections stay correctly delimited"
    - "Fixed two-tag evidence vocabulary carried forward: Confirmed against real rows / Constructed-input code-path check"

key-files:
  created: []
  modified:
    - .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md

key-decisions:
  - "Task 1 checkpoint (human, blocking-human gate): nullable-fk (Option A - make CampaignRun.campaign nullable) selected over single-sentinel and per-proposal-placeholder. Rationale: cheapest migration (single AlterField, 0/49 existing rows need it) and no placeholder-lifecycle open question, at the cost of a 5-site read-path null-guard obligation this plan hands to Phase 32."
  - "SCHEMA-02: a new source_identifier CharField(max_length=500, null=True, blank=True) with a partial UniqueConstraint (condition=source_identifier__isnull=False) is proposed as an additive identity surface, disjoint by field set from both existing partial constraints, confirmed empirically to coexist without disturbing either."
  - "Classical adapter (load_telescope_runs.py) writes a synthesized deterministic key (CLASSICAL:{telescope}:{instrument}:{start_time}) rather than leaving source_identifier blank, as a default pending plan 31-03's SCHEMA-03 investigation into real classical schedule files."

requirements-completed: [SCHEMA-01, SCHEMA-02]

coverage:
  - id: D1
    description: "Task 1: human decision at a blocking checkpoint between three one-way-door candidate schema shapes for a non-campaign CampaignRun"
    requirement: "SCHEMA-01"
    verification: []
    human_judgment: true
    rationale: "D-05 explicitly rates this a one-way door requiring real human judgment between three evidenced trade-offs, not an automatable choice; the orchestrator already routed this through the actual human user before this plan's Task 2/3 execution."
  - id: D2
    description: "SCHEMA-01 recommendation: nullable campaign FK named as the chosen shape, both rejected shapes ruled out by cited plan 31-01 measurements, Phase 32's read-site obligations recorded, D-06 confirmed at 0/49"
    requirement: "SCHEMA-01"
    verification:
      - kind: other
        ref: "grep -q '^## Recommendation' 31-DECISION.md && grep -q '^### SCHEMA-01 - schema shape for a non-campaign run' 31-DECISION.md && grep -q 'Why not the other two' 31-DECISION.md && grep -q 'What Phase 32 inherits' 31-DECISION.md"
        status: pass
      - kind: other
        ref: "sed -n '/^### SCHEMA-01 - schema shape/,/^### /p' 31-DECISION.md | grep -cE '(nullable|sentinel|placeholder|per-proposal)' -ge 3"
        status: pass
    human_judgment: false
  - id: D3
    description: "SCHEMA-02 recommendation: source_identifier field/type/constraint stated, non-collision argument against both existing partial constraints, three-row per-ingest-path table, classical blank-vs-synthesized decision, Phase 32 promote-vs-add-alongside guidance and invariant-test suggestion"
    requirement: "SCHEMA-02"
    verification:
      - kind: other
        ref: "grep -q '^### SCHEMA-02 - write-time identity field and constraint' 31-DECISION.md && grep -q 'load_telescope_runs' 31-DECISION.md && grep -q 'sync_lco_observation_calendar' 31-DECISION.md && grep -q 'sync_gemini_observation_calendar' 31-DECISION.md && grep -q 'unique_campaign_run_resolved_window' 31-DECISION.md && grep -q 'unique_campaign_run_tbd_natural_key' 31-DECISION.md"
        status: pass
      - kind: other
        ref: "sed -n '/^### SCHEMA-02 - write-time identity field/,/^## /p' 31-DECISION.md | grep -q CLASSICAL_FILE && ... | grep -q LCO_QUEUE && ... | grep -q GEMINI_QUEUE"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-09-02
status: complete
---

# Phase 31 Plan 02: Schema/Identity Recommendation (SCHEMA-01/02) Summary

**Locked nullable campaign FK (Option A) as the chosen schema shape and `source_identifier` as the write-time identity field, ruling out single-sentinel and per-proposal-placeholder on the measured collision evidence from plan 31-01, and handing Phase 32 the 5-site read-path null-guard list plus a per-adapter identity value for classical/LCO/Gemini.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2 (Task 1's checkpoint was resolved by a prior dispatch/human before this continuation started)
- **Files modified:** 1 (`31-DECISION.md`, built up across 2 commits)

## Accomplishments
- Recorded the human's Task 1 checkpoint decision (`nullable-fk`, Option A) as the `### SCHEMA-01` recommendation, with the exact `ForeignKey` field change (`null=True, blank=True`, `related_name`/`on_delete` unchanged), both rejected options ruled out by cited plan 31-01 measurements (Option B: 4 pre-existing colliding telescope/window tuples; Option C: Block D's same-placeholder collision), the full 5-site read-path null-guard obligation handed to Phase 32 (`__str__` named first, the hot `event_title()` site second), and D-06 confirmed directly (0/49 existing rows need migrating).
- Also addressed the plan's backstop truth (what happens to identity if a non-campaign run's proposal later acquires a real campaign): under Option A the `campaign` FK simply moves from `NULL` to the real value with no re-migration of `source_identifier` needed.
- Recorded the `### SCHEMA-02` recommendation: a `source_identifier` `CharField(max_length=500, null=True, blank=True)` with a partial `UniqueConstraint` (`unique_campaign_run_source_identifier`), argued and empirically confirmed disjoint from (and additive alongside) both existing partial constraints; a three-row per-ingest-path table naming the exact value/source-line/availability/probe-result for `load_telescope_runs.py` (synthesized key), `sync_lco_observation_calendar.py` (real portal URL), and `sync_gemini_observation_calendar.py` (constructed `GEM:` key); the classical adapter's explicit blank-vs-synthesized decision (synthesized, pending SCHEMA-03 refinement); Phase 32's promote-vs-add-alongside guidance and suggested invariant test; and the WR-05 race-safety finding (the field is backed by a real constraint).

## Task Commits
1. **Task 1: Decide schema shape (checkpoint:decision)** - resolved by human at a prior dispatch's checkpoint (`nullable-fk`); no commit produced by that step itself.
2. **Task 2: Write the SCHEMA-01 recommendation** - `507f0ac` (docs)
3. **Task 3: Write the SCHEMA-02 recommendation** - `6792748` (docs)

**Plan metadata:** committed alongside this SUMMARY (see below).

## Files Created/Modified
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` - appended `## Recommendation` with `### SCHEMA-01` and `### SCHEMA-02` subsections (created by plan 31-01; this plan only appends to it, per the plan's "leave `## Findings` open, append `## Recommendation` once" instruction)

## Decisions Made
See `key-decisions` in the frontmatter above. Summarized: the human chose nullable-fk (Option A) at Task 1's blocking checkpoint; Task 2/3 then locked the field/constraint details and per-adapter values that follow from that choice, using only cited plan 31-01 measurements, never unlabeled inference.

## Deviations from Plan
None - plan executed exactly as written. Both automated `<verify>` blocks and all `<acceptance_criteria>` passed on the first attempt for both Task 2 and Task 3; no fix-up needed.

## Issues Encountered
None.

## Next Phase Readiness
Plan 31-03 (SCHEMA-03, the classical adapter's identity surface) needs a real classical schedule file from the operator, per this phase's roadmap dependency — the orchestrator will need to obtain one before that plan can fully proceed. This plan's SCHEMA-02 write-up is directly relevant context for it: the classical row's synthesized-key recommendation here is explicitly stated as a *default pending 31-03's findings*, not a final answer — if 31-03 finds a real schedule file reliably carries a proposal code for some run states (per D-07), that supersedes the synthesized key without requiring any change to the `source_identifier` field or its constraint, only to the value `load_telescope_runs.py` writes into it. Plan 31-04 (SCHED-07, scheduling mechanism) and 31-05 (final `## Recommendation`/`## Durable summary` + `docs/design/` page) have no direct dependency on this plan's specific content beyond the phase's overall investigation-only framing already established. No blockers for 31-03 beyond the operator needing to supply a schedule file.

---
*Phase: 31-foundation-spikes-run-identity-unattended-invocation*
*Completed: 2026-09-02*

## Self-Check: PASSED

Both created/modified files verified present (`31-DECISION.md`, `31-02-SUMMARY.md`); all
three commit hashes (`507f0ac`, `6792748`, `388ea68`) confirmed present in
`git log --oneline --all`.
