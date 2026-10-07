---
id: SEED-261007-5pe
status: dormant
planted: 2026-10-07T02:59:35.000Z
planted_during: v2.4 intent review walkthrough (before /gsd-complete-milestone)
trigger_when: when the next milestone touches unattended discovery, proposal allocations, or campaign-run creation
scope: medium
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
  status: dormant
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

## Upstream context: TOM Toolkit Slack `#tom-toolkit`, "multi-proposal support" thread

Recorded 2026-10-06 from a screenshot of Slack's AI summary of the thread (37 messages from 19 August 2026; the developer could not export the thread itself). Treat the wording as a summary, not quotes.

- **Who:** Carrie Holt asked; Joey Chatelain (TOM Toolkit maintainer) answered.
- **The question:** multi-proposal support in TOM Toolkit — a single facility with several proposals, or several users each with access to designated proposals.
- **Upstream's position:** support is **facility-specific**; there is **no proposal model** in `tom_base`, and none was proposed in the thread.
  - LCO: the regular `LCOFacility` uses one API token in `settings.py`, typically a bot account added as a co-I on each proposal. Per-user proposal selection is **not yet supported**; "under consideration". Co-Is otherwise go to the LCO Observing Portal. LCO, SOAR and Blanco "AEON" time are all reachable through that portal.
  - ESO: credentials entered on a user-profile page.
  - Gemini: the only `tom_base` facility not reachable through the observing portal.
  - Direction of travel: **per-facility user-profile cards** carrying that user's credentials (an LCO token does not work at Gemini and vice versa). All *new* facilities get user-profile support (`tom_keck` will, once Keck's API exists); *older* facilities are not retrofitted unless someone raises a GitHub issue with a specific need.
- **Outcome:** Carrie would assess user needs and may open issues for Gemini and `tom_lt` if user-profile support is required there; she planned to consult Tim.

### What this means for the seed

- There is nothing upstream to align a FOMO `Proposal` record to, and nothing on the way: upstream is solving *who holds the credentials*, not *which proposals exist and what time they have*. A FOMO-local `Proposal` keyed by the LCO portal's proposal code (as the Notes above sketch) does not duplicate planned upstream work.
- The one adjacent upstream item is per-user proposal selection on `LCOFacility`. If FOMO ever wants a user to submit under a proposal of their own rather than the bot's co-I set, that is a `tom_base` issue to raise, not something to build into this seed.
- FOMO's bot-as-co-I token model is exactly the pattern upstream describes as current practice, so `WatchedProposal` / `proposal_codes_to_fetch()` keep working unchanged whatever upstream does with user profiles.
