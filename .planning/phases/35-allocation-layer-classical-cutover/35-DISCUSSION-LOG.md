# Phase 35: Allocation Layer & Classical Cutover - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-12
**Phase:** 35-allocation-layer-classical-cutover
**Areas discussed:** Classical line → allocation record, Handoff semantics, Projector home & trigger, Cutover mechanics (ALLOC-05), Queue-scheduled dispatch rule (added at wrap-up)

---

## Todo cross-reference

| Todo | Score | Folded |
|------|-------|--------|
| `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` | 0.6 (routes itself to Phase 35) | ✓ |
| `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` | 0.6 (keyword only) | |
| `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` | 0.6 (keyword only) | |

---

## Classical line → allocation record

**Q1 — collision-safe `source_identifier` (no proposal code in a schedule line)**

| Option | Description | Selected |
|--------|-------------|----------|
| Telescope + instrument + window + sub-night tokens | Deterministic across re-imports; a second line with the same key skipped-and-logged | |
| Same, plus an optional proposal token in the line format | `parse_run_line()` accepts an optional proposal token that joins the key; falls back to the above when absent | ✓ |
| Hash of the normalised source line | Any edit mints a new run; old run left behind | |

**Q2 — run grain**

| Option | Description | Selected |
|--------|-------------|----------|
| One run per line, range window | Projector fans out per-night ALLOC events, like a Gemini FT range run | ✓ |
| One run per night | 5-night line → 5 runs; mismatches campaign table/approval queue | |

**Q3 — parser statuses → `approval_status` / `run_status`**

| Option | Description | Selected |
|--------|-------------|----------|
| All APPROVED; status → run_status only | cancelled→CANCELLED, confirmed/allocation→PLANNED, proposed/not confirmed→REQUESTED; every line projects | ✓ |
| Only confirmed lines project | proposed/not confirmed → PENDING_REVIEW; changes today's behaviour | |

**Q4 — where sub-night windows (`BoN`/`EoN`/`HHMM`) live**

| Option | Description | Selected |
|--------|-------------|----------|
| Two new nullable fields on CampaignRun | e.g. `night_start_utc`/`night_end_utc` TimeField, null = sunset/sunrise | ✓ |
| Keep raw tokens in `observation_details` text | Fragile, staff-editable free text | |
| Drop sub-night windows | Lossy vs today's command | |

**User's choice:** as marked. **Notes:** follow-ons (proposal-token syntax, `--campaign` → `run.campaign`, `target=None`, `site` from `SITES`) left to planner discretion; user moved on.

---

## Handoff semantics

**Q1 — which allocation nights a linked record retires**

| Option | Description | Selected |
|--------|-------------|----------|
| Only its observing night once placed/observed; none while queued | Night of the block start (`_observing_night()`), queued-only retires nothing | ✓ |
| Every night the record's current window overlaps | Queued 3-day window retires 3 nights | |
| Only the start night, regardless of stage | Spike 003 verbatim | |

**Q2 — terminal-negative linked record**

| Option | Description | Selected |
|--------|-------------|----------|
| Night stays retired — the marked observation event is the night | Consistent with 34 D-11; Phase 37 decides the "unused" look | ✓ |
| Night is restored | Two events on the night | |

**Q3 — retire operation**

| Option | Description | Selected |
|--------|-------------|----------|
| Delete the ALLOC event and its meta row | Spike 003's proven op; unlink re-mints | ✓ |
| Detach (clear meta.run) and leave the row | Leaves a duplicate on the night | |

**Q4 — `meta.run` sync with `CampaignRunObservation` (33 D-14 vs 33-10 guard)**

| Option | Description | Selected |
|--------|-------------|----------|
| Set on link; clear on unlink only if not human-confirmed elsewhere | `adopt_event_into_run()` / `unlink_event_from_run()` with the human guard | ✓ |
| Set on link only; never clear | D-14 half-kept | |
| Set and clear unconditionally | Violates "human outranks machine" | |

**Notes:** recorded nuance — a record that expires while still queued never retired anything, so "stays retired" applies only to a placed-then-cancelled/failed block.

---

## Projector home & trigger

**Q1 — module home**

| Option | Description | Selected |
|--------|-------------|----------|
| New peer module; reconcile_run() dispatches to it | `allocation_projector.py` owns ALLOC:; `_reconcile_classical_nights()`/`run_night_url()` deleted | ✓ |
| Rewrite the classical branch in place | Reconciler writes two namespaces | |
| Fully separate entry point and sweep | Second call at every call site and in cron | |

**Q2 — trigger**

| Option | Description | Selected |
|--------|-------------|----------|
| Receivers on CampaignRunObservation + record-side changes | post_save/post_delete on the link; observation projector re-projects linked runs after a record save | ✓ |
| Receivers on CampaignRunObservation only | Stage changes wait for the sweep | |
| No new receivers — call sites + sweep only | | |

**Q3 — night title/description**

| Option | Description | Selected |
|--------|-------------|----------|
| `<telescope> <instrument>` + optional status prefix, dark window in description | Byte-identical to today's classical title | ✓ |
| Reconciler's current per-night title | `(window a..b)` suffix overflows the cell | |
| Add a target token when the run has one | Longer title | |

**Q4 — `sun_event()` cost**

| Option | Description | Selected |
|--------|-------------|----------|
| Never rewritten after creation; sub-night change re-mints | `sun_event()` only on create/re-mint | ✓ |
| Recompute every sweep | The cost the todo exists to remove | |

---

## Cutover mechanics (ALLOC-05)

**Q1 — mechanism and sequence**

| Option | Description | Selected |
|--------|-------------|----------|
| Sweep takeover for RUN: nights + one-time command for legacy classical events | migrate → deploy → cutover command → first sweep re-keys | ✓ |
| One command does everything; the sweep never re-keys | Must run before the first sweep | |
| RunPython data migration for both | Heavy, hard to dry-run | |

**Q2 — existing `RUN:{pk}:{date}` nights**

| Option | Description | Selected |
|--------|-------------|----------|
| Re-key in place | Keep pk, start/end, meta row | ✓ |
| Delete and let the projector re-mint | 56 sun_event scans, churned diff | |

**Q3 — legacy blank-url classical events**

| Option | Description | Selected |
|--------|-------------|----------|
| Re-parse `Source line:`, create the run, re-key the events | No file needed; idempotent | ✓ |
| Operator re-imports the file; command removes matched legacy events | Two-step dance | |
| Delete all blank-url classical events | Lossy | |

**Q4 — unparseable blank-url event (e.g. `tmp` pk 334)**

| Option | Description | Selected |
|--------|-------------|----------|
| Leave untouched, report it, exit non-zero on any leftover | Never deletes what it cannot explain | ✓ |
| Delete it | Risky for hand-made blank-url events | |

---

## Queue-scheduled dispatch rule (surfaced at wrap-up)

**Q1 — how `reconcile_run()` knows a site-resolved run is queue-scheduled**

| Option | Description | Selected |
|--------|-------------|----------|
| By `source`: queue sources → container | LCO/SOAR/GEMINI/ESO_QUEUE → `RUN:{pk}` regardless of site | ✓ |
| By site: LCO network site → container | Couples calendar to a facility list | |
| Keep today's rule unchanged | Queue runs at a fixed site draw nights they never use | |

**Q2 — leftover per-night events on re-classification**

| Option | Description | Selected |
|--------|-------------|----------|
| Delete leftover ALLOC nights; keep detach-only for RUN: containers | SC 5 holds through re-classification | ✓ |
| Detach everywhere, as today | Orphans on the calendar | |

**Q3 — `source=LEGACY` rows**

| Option | Description | Selected |
|--------|-------------|----------|
| LEGACY stays per-night; staff correct `source` via the admin action | No inference | ✓ |
| Cutover relabels LEGACY rows on LCO sites to LCO_QUEUE | Writes provenance from a guess | |

**Notes:** derived consequence recorded in CONTEXT D-16 — a `RUN:{pk}:{date}` night whose run now dispatches to the container is deleted at cutover, not detached (the whole per-night `RUN:` family retires).

---

## Claude's Discretion

Proposal-token syntax; D-04 field names/types; `--dry-run` on `load_telescope_runs` and summary wording; allocation-side receiver error contract and `sun_event` `ValueError` propagation; `ReconcileResult` counters and dry-run preview shape; cutover command name and whether it lives on; keeping `Source line:` in allocation-night descriptions; test file layout and notebook diff expression.

## Deferred Ideas

- Inferring `source` for `LEGACY` runs from site/telescope (declined; staff relabel).
- The visual look of an unused awarded night — Phase 37 UNUSED-01.
