---
id: SEED-261007-5pe
status: dormant
planted: 2026-10-07T02:59:35.000Z
planted_during: v2.4 intent review walkthrough (before /gsd-complete-milestone)
trigger_when: when the next milestone touches unattended discovery, proposal allocations, or campaign-run creation
scope: medium
---

# SEED-261007-5pe: Proposal-centric model: a FOMO Proposal record linking WatchedProposal, ProposalTimeAllocation and runs

## Why This Matters

FOMO has three free-text `proposal_code` fields with no link or cross-check — `WatchedProposal`, `ProposalTimeAllocation` (Phase 37 D-07) and `CampaignRun`. A run with an unwatched code is never discovered; a watched code with no run has nothing to link to; nothing says so. (Intent review Q8a/Q8b, answered 2026-10-06.) Separately, a new target under a watched cadence proposal has no run to link to, so every new target needs a person to create its per-target run (F11, seen live with CEV8YD2).

## When to Surface

**Trigger:** when the next milestone touches unattended discovery, proposal allocations, or campaign-run creation

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Medium**

## Breadcrumbs

- `.planning/v2.4-INTENT-REVIEW.md` — Q8a, Q8b, F11, F13 (2026-10-06)

## Notes

- A FOMO `Proposal` keyed by the LCO Observation Portal's proposal code, mirroring the portal's `proposals.models` (`Proposal` with the code as primary key; `TimeAllocation` per semester and instrument type; `Semester`), fetched from the portal.
- `WatchedProposal` and `ProposalTimeAllocation` become links to it.
- `CampaignRun.proposal_code` stays free text — non-LCO codes are real (ESO `117.2A2N.001`, Magellan) and never resolve — with an optional link where it does.
- Any cross-check between lists is scoped to fetchable codes by the rule `proposal_codes_to_fetch()` uses since F7, so a non-LCO code is "not fetchable", never a warning.
- F11 options, the developer's preference first: (1) a "create a per-target run from this orphan" action on the attribution page — keeps runs human-declared; (2) discovery creating the run itself — crosses the human-declared line; (3) one proposal-wide run — loses per-target `run_status`.
