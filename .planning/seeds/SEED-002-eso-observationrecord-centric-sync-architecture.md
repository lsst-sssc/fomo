---
id: SEED-002
status: dormant
planted: 2026-07-02
planted_during: post-v1.7 (between milestones; v1.7 ESO Feasibility Spike just shipped)
trigger_when: when ESO/VLT work resumes (v2 ESO milestone scoping, ESO-10/ESO-11 promotion, or tom_eso upstream activity)
scope: large
---

# SEED-002: ESO sync should converge on an ObservationRecord-centric architecture; Phase 13's Bypass verdict is a stepping stone, not the destination

## Why This Matters

Phase 13's Bypass verdict (sync straight from `p2api` to `CalendarEvent`, skipping
`ObservationRecord` for ESO entirely) was a valid *feasibility* finding about the
installed `tom_eso==0.2.4` — but operator judgment (Tim, 2026-07-02) is that it is
not the right medium-term architecture. `ObservationRecord` is the TOM Toolkit's
canonical store of requested observations — target linkage, status history, and
data-product association all hang off it — and `CalendarEvent` is properly a view
derived from it. The better medium-term future is `tom_eso` gaining real facility
functionality (working `submit_observation()`, plus `get_observation_status()` /
`get_observation_url()` / `data_products()`, which all raise `NotImplementedError`
in 0.2.4) so ESO observations flow through `ObservationRecord` like the other
`<Foo>Facility` modules, with calendar events built from records. Under permanent
Bypass, ESO events keyed `ESO:{env}/{obId}` float free of any Target, and FOMO
maintains two sync architectures forever.

ESO work is explicitly deferred right now due to limited ESO usage within LCO
research projects and FOMO — this seed preserves the architectural intent and the
hedging strategy until that changes.

## When to Surface

**Trigger:** when ESO/VLT work resumes — specifically any of:

- a v2 ESO milestone is scoped (promotion of deferred ESO-10
  `sync_eso_observation_calendar` / ESO-11 demo notebook),
- SEED-001's trigger fires (its trigger should be considered fired the moment ESO
  work resumes; `tom_eso#55` is already filed upstream),
- upstream `tom_eso` gains `ObservationRecord`-creation or status-interface
  functionality.

## Scope Estimate

**Large** — a full milestone, likely more than one:

1. When ESO-10 is eventually scoped, structure it in **two layers** regardless of
   architecture choice: a `p2api` → "observation facts" layer (headless auth via
   `FACILITIES['ESO']`, `getOB()`/`getOBExecutions()`/`getNightExecutions()`
   parsing, the captured 12-code `obStatus` mapping — needed under *either*
   architecture) and a thin "where the facts land" layer. Keep `obId` as the join
   key so a later `ObservationRecord` backfill can attach to existing
   `CalendarEvent`s — flipping Bypass → record-backed becomes a rewiring, not a
   rewrite.
2. The **gating unknown** for the ObservationRecord path is OB → FOMO-`Target`
   matching: `ObservationRecord.target` is required, but ESO service-mode OBs are
   authored in ESO's p2 tool, not the TOM, so imported records need a target
   resolution strategy. The Phase 13 spike never investigated this (read-only
   guardrail). Treat it as its own spike-sized question *before* committing to
   Bridge-style local record creation.
3. Upstream track (SEED-001): file the `submit_observation()` empty-ID-list bug
   and the `NotImplementedError` trio as `tom_eso` issues, offering the spike's
   captured response shapes as evidence, so upstream latency runs concurrently
   with local work instead of blocking it.

## Breadcrumbs

- `docs/design/eso_feasibility_spike.rst` — the shipped spike decision record
  (Bypass verdict, capability table, 12-code `obStatus` vocabulary, future-sync
  sketch); its `13-DECISION.md` companion lives in the archived phase directory.
- `.planning/seeds/SEED-001-file-upstream-tom-eso-feature-requests.md` — sibling
  seed; its trigger fires together with this one.
- `.planning/STATE.md` Deferred Items — ESO-10 / ESO-11 requirements deferred at
  v1.7 close, "now unblocked by Phase 13's Bypass verdict".
- `solsys_code/calendar_utils.py:insert_or_create_calendar_event()` — the
  facility-agnostic create-or-update helper any ESO sync reuses unchanged.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` and
  `sync_gemini_observation_calendar.py` — the two existing
  ObservationRecord→CalendarEvent consumers. If ESO converges on
  `ObservationRecord`, the pattern gains a **third consumer**, strengthening the
  currently-deferred case for extracting shared sync-loop machinery
  (per-facility counters + summary-line formatting duplicated across both
  commands today).
- Upstream: [TOMToolkit/tom_eso#55](https://github.com/TOMToolkit/tom_eso/issues/55)
  (La Silla `ESOAPI`/`p1api` wrapper bug, filed during the spike).

## Notes

Captured from a post-v1.7 architecture discussion (session "Post-ESO-spike
milestone", 2026-07-02) reviewing whether Bypass was right for the medium term.
Conclusion: hedge rather than choose — ship calendar value via the facts layer
when ESO work resumes, keep the record-backed flip cheap, and push upstream in
parallel.
