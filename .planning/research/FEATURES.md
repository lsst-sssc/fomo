# Feature Research

**Domain:** Observatory scheduling / telescope time-allocation coordination — specifically "one canonical run record" reconciliation between an awarded allocation, the calendar events showing it, and the observation records realising it
**Researched:** 2026-07-26
**Confidence:** MEDIUM

## Context: what real systems this draws on

- **LCO Observation Portal** (`observe.lco.global`) — the facility FOMO already syncs from (`sync_lco_observation_calendar.py`). Proposal = TAC award; Request = a submission against that award; sub-request states include `PENDING`/`WINDOW_EXPIRED`/`COMPLETED`. FOMO already models the placed-block vs. window-banner distinction for this facility (SYNC-02/03).
- **Gemini Phase II OT / GPP** — the facility FOMO already syncs from (`sync_gemini_observation_calendar.py`). Program time allocation vs. GEMMA's real-time-updated observing plan; public "Schedules and Queue" pages report percent-of-allocated-time-executed as a *separate* surface from the scheduling tool.
- **ESO P2** — investigated in FOMO's own Phase 13 feasibility spike (not re-researched here beyond the OB/`obStatus` model, which bears directly on the "single record whose status mutates" design question below).
- **ALMA Observing Tool** — Scheduling Blocks (SBs) repeated to reach a target/allocation, tracked against project priority + Executive time balance.
- **JWST APT** — Observation (PI-specified plan) decomposed into Visits (the schedulable unit); Visit Planner checks schedulability.
- **astroplan / TOM Toolkit** — `ObservingBlock` + scheduler abstractions exist upstream, but TOM Toolkit itself (FOMO's base framework) has **no built-in TAC-allocation model** — it stops at the request/`ObservationRecord` layer. This confirms `CampaignRun` is filling a real gap, not duplicating framework functionality.
- **OpenRefine reconciliation** — not astronomy-specific, but the best-documented real system for a confirm/reject matching UX at scale (confidence-scored suggestions, bulk-approve-above-threshold, facet-to-triage).

All findings below are MEDIUM confidence (uncorroborated live web search, cross-checked across multiple independent pages per claim) unless flagged otherwise; see Sources.

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| A single durable record per awarded allocation, separate from its executions | Every real system studied keeps *some* plan/award concept distinct from execution — LCO Proposal, Gemini Program, ALMA project+SB, JWST Observation. `CampaignRun` already exists; v2.2 makes it the thing everything else points at rather than a peer of `CalendarEvent`/`ObservationRecord`. | MEDIUM | Already 90% built (`CampaignRun` model, v2.0-v2.1). v2.2's job is making linkage *structural* (FKs) not incidental (a click-time side effect). |
| Calendar visibility for every awarded run, without a bespoke backfill command per gap | Table stakes for *any* scheduling tool — an award that doesn't show up anywhere is effectively invisible to the people who need to avoid double-booking it. FOMO's own dev DB already demonstrates the cost of not having this: 19 approved, windowed `CampaignRun`s have zero calendar presence today (per PROJECT.md). | MEDIUM-HIGH | This is the reconciler. Complexity is in idempotency (never fabricate a duplicate event) and in covering every entry path (CSV import, submission form, backfill, and future adapters) with one mechanism instead of N. |
| Progressive resolution from "roughly this class of telescope, this day" to "this exact block, this outcome" | Every studied system defers precision until precision is actually known — LCO's window-banner-to-placed-block (already shipped in FOMO, SYNC-02/03), ESO's OB status field maturing from prepared to `C`/`M`/`A`/`F`, STScI's general guidance that JWST tools default to widest constraints and narrow only as real scheduling resolves. Nobody found guesses at exact times before the scheduler has resolved them. | MEDIUM | v2.2's four-stage pipeline (site → class → scheduled → completed) is a direct generalization of what FOMO already does for LCO alone. The work is making it run off `CampaignRun` state for *every* source, not just LCO. |
| A run's status/outcome is visible without cross-referencing multiple tables | LCO/Gemini both surface allocation-vs-used-time on a dedicated status page rather than forcing users to diff two record types by hand. FOMO's coverage-gap analysis is the closest existing analog and is currently blind to non-`CampaignRun` calendar activity (LCO/Gemini-sourced events aren't counted), which the PROJECT.md context calls out as a live defect. | MEDIUM | v2.2 explicitly defers "provenance-blind gap analysis" to v2.3 — correctly scoped out, but flag it as the natural next-milestone follow-on so the roadmap doesn't lose it. |
| Idempotent, non-destructive reconciliation (safe to re-run) | Every FOMO sync command already guarantees this (SYNC-04, GEM-NOCHURN-01, CAL-03) — it's an established codebase convention, not new territory, but the reconciler must hold to the same bar across four pipeline stages instead of one sync path. | MEDIUM | Direct precedent exists in-repo (`insert_or_create_calendar_event`); risk is in stage *transitions* (e.g. stage 2→3 narrowing) not introducing churn or losing manually-set data (see Anti-Features: silent overwrite). |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Operator-assisted attribution (suggested, not automatic, links) | None of the large facility tools studied (LCO/Gemini/ESO/ALMA/JWST) need this feature at all — they're single-source-of-truth systems where the award *is* created in the same system that schedules it. FOMO's distinctive problem is that it aggregates 4+ independent ingest paths (classical file, LCO queue sync, Gemini queue sync, campaign CSV/web submission) that can describe the *same* real allocation without knowing about each other. This is closer to library/catalog reconciliation (OpenRefine) than to any observatory tool. | MEDIUM-HIGH | OpenRefine's pattern is the load-bearing lesson: score/threshold + bulk-approve-the-confident-tail + hand-review only the ambiguous remainder. A queue that shows every candidate link at equal weight (no scoring, no bulk action) will not survive contact with the real backlog (FOMO's own dev DB already has one confirmed double-representation case: `CampaignRun` pk=1 vs. 11 LCO-sourced events for the same FTS/MuSCAT4 run). |
| `source` provenance on every run (web submission / classical file / LCO queue / Gemini queue / CSV import) | No facility tool needs this either, for the same reason — FOMO is unusual in unifying heterogeneous ingest paths under one record. Distinct approval gating per source (only web submissions need staff approval) is a genuinely FOMO-specific requirement, not a pattern borrowed from elsewhere. | LOW-MEDIUM | Already scoped as a target feature; low technical risk, mostly a schema + branching-logic change with existing precedent (`ApprovalStatus`/`RunStatus` TextChoices already exist). |
| Telescope-class-only allocation made visible (stage 2 of the pipeline) | LCO/Gemini/ALMA all support class-wide or facility-wide time (e.g. "any 1m0 in the network") but their own UIs generally show this as an instrument/proposal filter, not as a first-class calendar presence spanning many sites at once. FOMO surfacing a genuine class-wide award as one visible thing (00:00-23:59 that day, not pinned to a site) is a real gap-closer: today `telescope_class=None` and `site=None` look identical, which the milestone context explicitly flags as a live ambiguity bug. | MEDIUM | Directly named in target features; the differentiator is disambiguating "this failed to resolve" from "this is legitimately class-wide," which nothing in the facility tools studied needed to solve because they don't ingest free-text schedules from multiple sources. |
| Unused/never-realised awarded time surfaced explicitly | LCO/Gemini both track this, but on a *separate reporting page*, not inline with the scheduling calendar — that's a deliberate, repeated pattern across two independent real systems worth following. A `CampaignRun` that stops at stage 1 (allocated, never acquires an `ObservationRecord`) is exactly this case in FOMO's model, and it's cheap to compute once the linkage exists. | LOW (data model already there) | Not explicitly named as a v2.2 target feature — flag as a natural, low-cost v2.3 candidate once linkage exists, following the "separate status page, not folded into the calendar" pattern from LCO/Gemini. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Auto-merge/auto-link suspected duplicates above a confidence threshold, no human step | Feels like it saves staff time; OpenRefine itself supports a "match each cell to its best candidate" bulk action, so it's a real pattern elsewhere. | FOMO's milestone context is explicit and correct that this is a merge/deduplication trap: `CampaignRun` pk=1 (the award) and its 11 LCO-sourced events (the realisation) are **not duplicates** — collapsing them would destroy the very distinction v2.2 exists to create. Any auto-link risks quietly merging two genuinely different real-world runs that merely share a telescope+date (e.g. two different campaigns both using FTS the same week). | Keep every suggested link human-confirmed (already the stated design: "never a silent merge"). Reserve threshold-based bulk action, if added later, for *link confirmation* only — never for `CampaignRun` record merging. |
| One unified status vocabulary across all four ingest sources, done now | Feels like obvious cleanup once linkage exists — you'd naturally want `_CLASSICAL_STATUS_PREFIX`, `_FAILURE_PREFIX_BY_STATUS`, `_RUN_STATUS_CALENDAR_PREFIX`, and `calendar_display_extras._TERMINAL_PREFIXES` to agree by more than convention. | Correctly identified in PROJECT.md as **deliberately deferred to v2.3** — bundling it into v2.2 conflates "build the linkage" with "rationalize four independently-evolved status enums," doubling the blast radius of an already-substantial migration/attribution phase. | Ship v2.2's linkage first: prove the canonical-record model works, *then* unify vocabularies once there's one source of truth to unify them against. |
| Rewiring all four ingest adapters to write `CampaignRun` directly instead of writing `CalendarEvent`/`ObservationRecord` and reconciling after the fact | Feels like the "real" fix — if `CampaignRun` is canonical, why not have every adapter create it directly and skip reconciliation entirely? | Correctly deferred to v2.3 in PROJECT.md. Doing it now means changing four working, tested, independently-evolved sync commands (LCO, Gemini, classical, campaign CSV/web) in the same milestone as building and proving the reconciler and attribution model — a large simultaneous-change surface with no fallback if the canonical-record design needs adjustment after real use. | Prove the reconciler against *existing* adapter output first (attribution, not rewrite). Only rewire adapters to write `CampaignRun` natively once the model is validated in production. |
| A fully-automated scheduler that decides real telescope time allocation (à la GEMMA/astroplan priority scheduler) | Tempting scope creep once the domain research turns up real schedulers — "why not have FOMO actually schedule things?" | Out of scope by a wide margin: FOMO doesn't award time (TACs and facility portals do) or control telescopes; its job is coordination/visibility across records that already exist elsewhere. Building a scheduler duplicates work every facility studied already does better, with none of their operational safeguards (weather feeds, instrument constraints, guide-star availability). | Stay a reconciler/visibility layer, not a scheduler. This is consistent with FOMO's existing architecture (`solsys_code/telescope_runs.py` computes *observability*, not allocation). |
| Real-time/live-updating reconciliation (webhooks, sub-minute refresh) matching GEMMA's real-time replanning | Sounds like parity with state-of-the-art facility tools. | FOMO's existing sync commands are all cron/manual-invocation batch jobs (management commands), and nothing in the milestone or the domain requires sub-day latency — `CampaignRun`s are awarded weeks/months ahead, and even LCO's own placed-block sync is periodic, not push-based. Real-time infrastructure (webhooks, queues, always-on workers) is a large new operational surface for no demonstrated user need. | Keep the reconciler an idempotent, periodically-invoked management command, matching every existing FOMO sync command's shape. |
| Carry-forward / rollover semantics for unused allocated time | LCO's own docs make this sound plausible to add ("shouldn't unused nights roll to next window?"). | LCO explicitly does **not** allow unused-hour carryover between semesters — this is a deliberate telescope-time-allocation policy decision made by TACs/facilities, not a data-modeling gap. FOMO surfacing "never used" time is valuable (visibility); FOMO *inventing* rollover semantics on top of a facility's award would misrepresent the actual award and could mislead a PI about time they don't actually have. | Show unused/never-realised time as a fact ("this run never acquired a record"), never as a projected future re-allocation. |

## Feature Dependencies

```
Companion-record generalization (CalendarEventTelescopeLabel -> run FK)
    └──requires──> nothing new (extends existing sidecar model)

ObservationRecord -> CampaignRun linkage
    └──requires──> Companion-record generalization decided first (same migration-shape questions: nullable FK, natural-key mapping)

Operator-assisted attribution UI
    └──requires──> Both linkage FKs existing (companion record + ObservationRecord link)
    └──requires──> source/telescope_class fields (to disambiguate "legitimately class-wide" from "unresolved" before suggesting links)

The reconciler (four-stage pipeline)
    └──requires──> Both linkage FKs existing
    └──requires──> source/telescope_class fields (stage 1 vs stage 2 branch on telescope_class)
    └──enhances──> Coverage-gap analysis (future v2.3: gap analysis becomes provenance-blind once reconciler is the single writer of run-derived events)

Spike (natural keys, adapter identity mapping, migration/attribution strategy)
    └──blocks──> Operator-assisted attribution UI (can't design the confirm/reject queue without knowing what "same run" means per source)
    └──blocks──> The reconciler (can't safely re-derive events without knowing the migration/backfill strategy for existing data)

Unified status vocabulary (v2.3, anti-feature-flagged above if pulled into v2.2)
    └──conflicts──> doing it inside v2.2 (see Anti-Features)

Adapter rewrite to write CampaignRun directly (v2.3)
    └──conflicts──> doing it inside v2.2 (see Anti-Features)
```

### Dependency Notes

- **The spike blocks both the reconciler and the attribution UI:** this is already reflected in the target-feature list ("settles what milestone questioning did not"), and the research above reinforces why it has to go first — the OpenRefine lesson (score + bulk-approve + hand-review-the-tail) can't be designed without first knowing what a "candidate match" even looks like per source, which is exactly the identity-mapping question the spike is scoped to answer.
- **Companion-record generalization should land before the `ObservationRecord` link**, not alongside it, because it's an extension of an existing, tested pattern (`CalendarEventTelescopeLabel`) while the `ObservationRecord` link is new surface (a `CampaignRun`-side many-to-many, per PROJECT.md, since `ObservationRecord` is third-party). Sequencing the known-shape change first de-risks the migration approach for the newer one.
- **The reconciler enhances, but should not be gated on, coverage-gap analysis becoming provenance-blind** — that's explicitly v2.3 scope. The dependency arrow points the other way: gap analysis improvement *depends on* the reconciler existing, not vice versa. Don't let "let's also fix gap analysis" creep into a v2.2 phase.
- **Unused/never-realised-time visibility (identified as a differentiator above) depends on the same linkage the reconciler needs**, so it's a near-zero-marginal-cost addition once v2.2's core model lands — worth flagging to the roadmap as a candidate final v2.2 phase or immediate v2.3 opener, following the LCO/Gemini pattern of a separate status view rather than folding it into the calendar UI itself.

## MVP Definition

### Launch With (v2.2 core, per PROJECT.md target features)

- [ ] Spike settling natural keys, adapter identity mapping, and migration/attribution strategy — nothing else can be soundly designed without this
- [ ] Companion record generalization (`CalendarEventTelescopeLabel` → `run` FK) — extends a proven pattern, lowest-risk linkage
- [ ] `source`/`telescope_class` fields on `CampaignRun` — needed to disambiguate stage-1-vs-stage-2 pipeline branching and to gate approval correctly per source
- [ ] `ObservationRecord` → `CampaignRun` linkage — the second half of "a run owns what realises it"
- [ ] The reconciler (four-stage pipeline) — the actual deliverable; retires `backfill_range_calendar_events`
- [ ] Operator-assisted attribution for *existing* data — without this, the reconciler either fabricates duplicate events for the 19 already-invisible runs and the double-represented FTS run, or leaves them stuck; this is not optional polish, it's how existing data becomes usable under the new model

### Add After Validation (v2.3 candidates, already flagged as deferred in PROJECT.md)

- [ ] Unified status vocabulary across all four prefix maps/enums — do this once there's one canonical model to unify against, not before
- [ ] Rewire the four ingest adapters to write `CampaignRun` directly instead of `CalendarEvent`/`ObservationRecord` — do this once the reconciler is proven against real (not synthetic) attribution outcomes
- [ ] Provenance-blind coverage-gap analysis (count LCO/Gemini/classical-sourced events, not just `CampaignRun` rows) — natural follow-on once the reconciler makes those events reliably `run`-linked
- [ ] Explicit "unused/never-realised allocation" rollup view — cheap once linkage exists; model it as a separate status view (LCO/Gemini pattern), not a calendar overlay

### Future Consideration (defer indefinitely / out of scope)

- [ ] Any form of automated scheduling/allocation decision-making (GEMMA/astroplan-priority-scheduler-style) — not FOMO's job; FOMO coordinates and visualizes, it doesn't award or schedule telescope time
- [ ] Real-time/webhook-driven reconciliation — no demonstrated need beyond FOMO's existing periodic-management-command cadence
- [ ] Allocation rollover/carryover semantics — actively contradicts real facility policy (LCO explicitly disallows it); FOMO should reflect reality, not invent policy

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Spike (natural keys / adapter mapping / migration strategy) | HIGH (blocks everything else) | LOW-MEDIUM | P1 |
| Companion record generalization | MEDIUM | LOW | P1 |
| `source`/`telescope_class` fields | MEDIUM | LOW | P1 |
| `ObservationRecord` → `CampaignRun` linkage | HIGH | MEDIUM | P1 |
| The reconciler (four-stage pipeline) | HIGH | HIGH | P1 |
| Operator-assisted attribution UI | HIGH | MEDIUM-HIGH | P1 |
| Unused/never-realised allocation rollup | MEDIUM | LOW (once linkage exists) | P2 (v2.3 opener candidate) |
| Unified status vocabulary | LOW-MEDIUM (developer-facing, not user-facing) | MEDIUM | P3 (v2.3) |
| Adapters write `CampaignRun` natively | MEDIUM (removes reconciler as sole writer) | HIGH | P3 (v2.3) |
| Provenance-blind gap analysis | MEDIUM | LOW (once linkage exists) | P3 (v2.3) |
| Automated scheduling/allocation | N/A | N/A | Out of scope |
| Real-time reconciliation | LOW | HIGH | Out of scope |
| Allocation carryover semantics | N/A | N/A | Anti-feature |

**Priority key:**
- P1: Must have for v2.2 launch
- P2: Should have, strong v2.3-opener candidate given near-zero marginal cost once P1 linkage exists
- P3: Correctly deferred to v2.3 per PROJECT.md's own scoping

## Competitor Feature Analysis

| Feature | LCO Observation Portal | Gemini OT/GPP | ESO P2 | ALMA OT | FOMO's approach (v2.2) |
|---------|------------------------|---------------|--------|---------|-------------------------|
| Award vs. execution representation | Proposal (award) → Request → sub-request states | Program (award) → OT-defined observations/visits, GEMMA replans in real time | Single OB, `obStatus` field mutates in place (award and execution are the same row) | Project → ObsUnitSet → SB, SB *repeated* to reach allocation | `CampaignRun` (award, canonical) owns many `ObservationRecord`s (executions) via explicit FK/M2M — closer to LCO/ALMA's separated model than ESO's conflated one |
| Unused-time visibility | Separate proposal-accounting page; no semester carryover | Separate "Schedules and Queue" percent-executed page, updated daily | Not surfaced as a distinct concept (status is per-OB, not aggregated) | Tracked via Executive time balance, not user-facing per-project | Deferred to v2.3; recommend following LCO/Gemini's separate-view pattern, not folding into the calendar |
| Progressive window narrowing | Yes — window banner → placed block (already mirrored in FOMO's LCO sync) | Implicit via GEMMA real-time replanning; not user-facing as discrete stages | No — OB is prepared once, then executed; no intermediate narrowing UI | No — SB either executes or doesn't per attempt | Four explicit stages (site → class → scheduled → completed), more granular and more visible than any single system studied |
| Cross-source reconciliation / attribution of independently-created records describing the same real allocation | Not applicable — single source of truth | Not applicable — single source of truth | Not applicable — single source of truth | Not applicable — single source of truth | **Novel to FOMO** — no facility tool needs this because none of them aggregate independent ingest paths; closest real-world analog is OpenRefine-style reconciliation (score + bulk-approve + hand-review), not any observatory tool |

## Sources

- [Open Access Allocation Process - Las Cumbres Observatory](https://lco.global/observatory/proposals/open-access-time-allocation-process/)
- [Time Refund Policy - Las Cumbres Observatory](https://lco.global/documentation/time-refund-policy/)
- [LCO Developers](https://developers.lco.global/)
- [Observing Tool (OT) | Gemini Observatory](https://www.gemini.edu/observing/phase-ii/ot)
- [Schedules and Queue | Gemini Observatory](https://www.gemini.edu/observing/schedules-and-queue)
- [Time Allocation Committees (TAC) processes | Gemini Observatory](https://www.gemini.edu/observing/phase-i-proposing-time/tac)
- [Program execution status - ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/program-execution-status)
- [After the execution my OBs have "status" C, or A and M. What does it mean? - ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/after-the-execution-my-obs-have-status-c-or-a-and-m-what-does-it-mean)
- [What is an observing block? - ESO Operations Helpdesk](https://support.eso.org/en-US/kb/articles/what-is-an-observing-block)
- [ALMA Observing Tool User Manual](https://almascience.nao.ac.jp/documents-and-tools/cycle-0/alma-ot-user-manual)
- [Cycle 13 Proposer's Guide — ALMA Science Portal](https://almascience.eso.org/proposing/proposers-guide)
- [astroplan: An Open Source Observation Planning Package in Python — IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/aaa47e)
- [Observations — TOM Toolkit documentation](https://tom-toolkit.readthedocs.io/en/latest/api/tom_observations/)
- [Planning and Scheduling Observations with Hubble and Webb | STScI](https://www.stsci.edu/contents/newsletters/2025-volume-42-issue-02/planning-and-scheduling-observations-with-hubble-and-webb)
- [APT Visit Planner - JWST User Documentation](https://jwstcf.stsci.edu/jwst-astronomers-proposal-tool-overview/apt-workflow-articles/apt-visit-planner)
- [Observation Specifications - JWST User Documentation](https://jwst-docs.stsci.edu/jppom/observation-specifications)
- [Reconciling | OpenRefine](https://openrefine.org/docs/manual/reconciling)
- [Reconciliation API | OpenRefine](https://openrefine.org/docs/technical-reference/reconciliation-api)
- `.planning/PROJECT.md` (v2.2 "One Canonical Run Record" milestone context, existing v1.0-v2.1 shipped feature history)

---
*Feature research for: observatory scheduling / telescope time-allocation coordination*
*Researched: 2026-07-26*
