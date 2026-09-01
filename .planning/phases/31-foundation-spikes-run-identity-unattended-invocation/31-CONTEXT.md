# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase is **investigation-only** — no source behaviour changes, no code written for adapters, no migration applied. Its job is to settle, before Phase 32 touches any adapter code:

1. Whether `CampaignRun.campaign` becomes nullable (or an equivalent identity mechanism) so a routine, non-campaign LCO/Gemini queue observation can still get a persistent `CampaignRun` — backed by real dev-DB evidence, not argument (SCHEMA-01, SCHEMA-02).
2. Whether the classical adapter's tolerance-windowed match is a sufficient write-time identity surface on its own, or needs a facility-specific key (SCHEMA-03).
3. The unattended-invocation mechanism (SCHED-07) — verified against the real target host's constraints (overlap prevention, credential handling, missed-invocation visibility), not merely recommended.

Deliverable: a phase decision doc (`31-DECISION.md`) plus a durable `docs/design/` page, following the `18-DECISION.md` → `docs/design/uncertain_scheduling_spike.rst` and `26-DECISION.md` → `docs/design/canonical_record_spike.rst` precedent. No paired notebook/runbook update — CLAUDE.md's paired-docs rule triggers on behaviour changes, and this phase makes none.

</domain>

<decisions>
## Implementation Decisions

### Real host / deployment facts (for the SCHED-07 track)

None of this is recorded anywhere in CLAUDE.md or `.planning/codebase/*` — confirmed by search; this is exactly the gap STATE.md flagged as a blocker only the operator could close.

- **D-01:** FOMO runs today on a local Rocky 9 Linux machine or a WSL2 Ubuntu install. The eventual production target is **LCO's AWS Kubernetes cluster**.
- **D-02:** The scheduling mechanism is **cron + `flock`, running inside the FOMO container** — the same mechanism on the interim host (Rocky 9 / WSL2) and once deployed to AWS (inside the K8s pod's container), rather than a K8s-native `CronJob`. This settles SCHED-07's headline question: no task-queue dependency, and no K8s-native scheduling redesign — cron+flock travels with the container. — **Reversibility:** costly — **rationale:** Phase 34 will build the actual scheduler entry point against this mechanism; switching to K8s-native `CronJob` afterward would mean re-deriving overlap prevention (`concurrencyPolicy` vs. `flock`) and credential wiring (K8s Secrets vs. env vars) from scratch.
- **D-03:** Whether `flock` is actually present in the container image, and whether an outbound heartbeat ping (e.g. healthchecks.io) is acceptable from the eventual AWS deployment, are **not confirmed** — the spike must verify both against the real container/host rather than assume either.
- **D-04:** Credentials are supplied via **environment variables**, extending the existing `FINK_CREDENTIAL_*` pattern already used elsewhere in this codebase (per `.planning/codebase/CONCERNS.md:21`) — no new secrets-file mechanism.

### Schema shape for non-campaign runs (SCHEMA-01/02)

- **D-05:** No lean between the candidate approaches — the user is explicitly open to any of: making `campaign` nullable, a single sentinel/placeholder `TargetList` (e.g. "No Campaign"), or a **per-proposal auto-created default `TargetList`** (one placeholder per LCO/Gemini proposal ID, not a single shared bucket). The spike must investigate at least these three against real dev-DB data and recommend one with evidence, per the roadmap's Success Criterion 1. — **Reversibility:** one-way — **rationale:** whichever shape is chosen becomes the write-time identity surface every Phase 32 adapter targets; changing it after adapters ship means a re-migration across all three adapters' writes (the exact risk research's Executive Summary calls out for "guessing it wrong").
- **D-06:** Migration risk for **existing** `CampaignRun` rows is treated as **low** — every row today came through campaign submission / CSV-import / attribution paths, all of which already require a campaign. The spike should confirm this with a real count from `src/fomo_db.sqlite3` rather than assume it, but should not spend investigation budget hardening a migration path for rows that likely don't need one. Nullability/sentinel design only has to serve **new** adapter-written rows.

### Classical adapter's identity surface (SCHEMA-03)

- **D-07:** A proposal code **may** be present in a classical schedule line, but — per the user's recollection — only for specific run states (e.g. "planned" or "observed"), not necessarily for earlier states (e.g. "requested"). The spike must inspect real classical schedule file samples to confirm which states carry it. Where absent, the existing 5-minute telescope/instrument/start_time tolerance match (Phase 26 Criterion 2, `load_telescope_runs.py:207-216`) remains the fallback — the spike should not assume tolerance-match is either the ceiling or obsolete without checking.

### What counts as "real data" for the spike

- **D-08:** "Real dev-DB rows" (roadmap Success Criterion 1) means the local **`src/fomo_db.sqlite3`** — the standard dev database per CLAUDE.md conventions. No other dataset or snapshot is in play.

### Claude's Discretion

- Exact investigation methodology (which queries to run against `src/fomo_db.sqlite3`, how to sample classical schedule files) is left to the researcher/planner — the decisions above scope *what* to investigate, not *how*.
- The `docs/design/` page's filename and structure follow the `uncertain_scheduling_spike.rst` / `canonical_record_spike.rst` precedent unless the researcher finds a reason to deviate.

### Reviewed Todos (not folded)

Four pending todos scored a weak (0.6) keyword match against this phase during `cross_reference_todos` but were reviewed and **not** folded — none are actually about run-identity schema or unattended invocation, they matched on generic "campaign"/"test" keywords:

- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — extract `calendar_utils.py` mapping/instrument-extraction into its own module. Tangential; belongs with whichever phase next touches that module.
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` — cache `orphans_needing_attribution_count()`. Unrelated to this phase's spike topics.
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — server-side guard on `AttributionDecisionView._dismiss()`. Unrelated.
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — skip `sun_event()` for existing reconciler nights. Unrelated.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap / requirements (this phase's mandate)
- `.planning/ROADMAP.md` §"Phase 31: Foundation Spikes — Run Identity & Unattended Invocation" — goal, depends-on, success criteria, and the v2.3 "Locked constraints" block above it (schema spike blocks adapters; no task-queue unless spike proves cron+flock can't work; new logic must never import `solsys_code.views`/`solsys_code.ephem_utils`)
- `.planning/REQUIREMENTS.md` — SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHED-07 (lines 12-14, 33)

### Research (grounds this phase's investigation scope)
- `.planning/research/SUMMARY.md` — Executive Summary, Critical Pitfalls #1-3 (SQLite write-lock collision, silent scheduled-job failure, credential leakage), Research Flags for Phase 1/3 (real deployment constraints gap, classical write-time identity gap)

### Prior spike precedent (identity findings this phase extends)
- `.planning/milestones/v2.2-phases/26-canonical-record-spike/26-DECISION.md` §"Criterion 2 / SPIKE-02 — per-adapter identity key to run" (lines 835-859) — the existing **read-time** identity mapping per adapter (classical: tolerance match, no `url`; LCO: portal request `url`; Gemini: constructed `GEM:{prog}/{obsid}` `url`, unconfirmed against real rows) that SCHEMA-02 extends to the write-time `CampaignRun` surface
- `docs/design/canonical_record_spike.rst` — durable version of the above, and this phase's own deliverable's structural precedent
- `docs/design/uncertain_scheduling_spike.rst` — structural precedent for the SCHED-07 track's decision doc (from `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`)

### Model under investigation
- `solsys_code/models.py:78-352` (`CampaignRun`) — current `campaign` FK (`NOT NULL`, `on_delete=PROTECT`), the two window `UniqueConstraint`s (`unique_campaign_run_resolved_window`, `unique_campaign_run_tbd_natural_key`), and the `CheckConstraint` pairing `window_start`/`window_end` — all three interact with whatever identity/nullability shape the spike recommends

### Codebase facts confirming the host-facts gap
- `.planning/codebase/CONCERNS.md:21` — existing `FINK_CREDENTIAL_*` env-var pattern (D-04 extends this)
- `.planning/codebase/CONCERNS.md:159` — SQLite concurrent-write-contention concern (grounds D-02/D-03)
- `.planning/codebase/INTEGRATIONS.md` §"CI/CD & Deployment" — confirms no production application-deployment target is documented anywhere in this repo's planning docs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `campaign_views.py::_notify_staff()` — existing `mail_admins()`-based staff-notification idiom; research recommends reusing it for in-command failure notification once Phase 34 builds the scheduler entry point (not this phase's job, but the spike's decision doc should note it as the intended consumer).
- `docs/design/uncertain_scheduling_spike.rst` and `docs/design/canonical_record_spike.rst` — structural templates for this phase's own decision-doc deliverable.

### Established Patterns
- **Decision-doc-then-durable-doc pattern:** every prior investigation-only phase (18, 26) produced a `.planning/`-scoped `{N}-DECISION.md` plus a `docs/design/` page carrying the verdict forward for readers outside `.planning/`. This phase follows the same shape.
- **`CampaignRun.source` vocabulary already anticipates this phase:** `Source.CLASSICAL_FILE`/`LCO_QUEUE`/`GEMINI_QUEUE` are declared in `models.py` (lines 108-137) but unused by any write path until v2.3's adapters — the spike's identity scheme should assume these values are available to key logic on, per-adapter.

### Integration Points
- Whatever identity/nullability shape this phase recommends becomes the schema every Phase 32 adapter plan writes against — the spike output is a direct input to Phase 32's planning, not just documentation.
- The scheduling-mechanism verdict (cron+flock in-container) is a direct input to Phase 34's scheduler entry point.

</code_context>

<specifics>
## Specific Ideas

- The scheduling mechanism must work identically whether FOMO is running on the interim Rocky 9/WSL2 host or inside a container in LCO's AWS Kubernetes cluster later — "cron+flock inside the container" was chosen specifically because it doesn't require a K8s-native redesign when that migration happens.
- The user is genuinely undecided on the schema shape (nullable vs. sentinel vs. per-proposal auto-created placeholder) and wants the spike to investigate and recommend, not just validate a preconceived answer.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (See "Reviewed Todos (not folded)" under `<decisions>` above for the four todos that were considered and explicitly left out.)

</deferred>

---

*Phase: 31-foundation-spikes-run-identity-unattended-invocation*
*Context gathered: 2026-09-01*
