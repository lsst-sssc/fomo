# Project Research Summary

**Project:** FOMO v2.3 "Automatic Run Sync & Outcome Propagation"
**Domain:** Unattended scheduling + adapter rewiring + outcome propagation in a Django/TOM Toolkit app
**Researched:** 2026-09-01
**Confidence:** HIGH (grounded in direct codebase inspection and already-shipped v2.2 precedents)

## Executive Summary

FOMO v2.3 automates three observation-sync commands to run unattended on a schedule, with outcomes propagating automatically to a canonical `CampaignRun` record. The research identifies OS-level cron as the right scheduling mechanism (no new Python dependency, matches this codebase's command-centric design), but reveals a **critical structural decision that must be settled before any adapter is rewritten**: whether `CampaignRun.campaign` becomes nullable, and how non-campaign-linked observations (routine follow-up without a coordinated campaign) get a persistent identity. This is not an edge case — both LCO and Gemini syncs routinely encounter records with no campaign context. Recommended approach: one phase-time investigation spike to settle the schema/identity-key shape for all three adapters simultaneously, then execute the adapters and outcome-propagation sequentially once that foundation is solid. The outcome-derivation rule for mixed-outcome runs (e.g., "any-success-wins" for a multi-night run where some nights weather out but others complete) is explicitly designed rather than implicit, with CI/CD and Kubernetes precedent offering proven patterns.

## Key Findings

### Recommended Stack

**Scheduling mechanism:** OS cron + `flock` for overlap prevention (no new Python dependency; matches this codebase's command-centric design). Optional: healthchecks.io (hosted, free tier) or self-hosted healthchecks for dead-man's-switch monitoring to catch scheduler failures that in-command error handling cannot.

**Core technologies:**
- **OS cron** — unattended invocation of management commands — zero new daemon, zero new infra, runs commands exactly as operators run them by hand.
- **`flock` (util-linux)** — prevent overlapping command invocations — SQLite's single-writer model requires this guard.
- **Django's `mail_admins()` pattern** (via existing `campaign_views.py::_notify_staff()`) — in-command failure notification — reuse existing idiom.
- **healthchecks.io (optional)** — dead-man's-switch visibility — catches scheduler/host failures that in-command exception handling cannot.
- **WatchedProposal model** — watch-list configuration surface — small Django model editable via admin without redeploy, matching how Observatory and CampaignRun are managed.

### Expected Features

**Table stakes (must ship):**
- Unattended recurring invocation with no per-invocation arguments (watch-list replaces `--proposal`/`--name-prefix`)
- Failure visibility combining in-command logging + heartbeat/dead-man's-switch
- Adapter consolidation: adapters write `CampaignRun` instead of `CalendarEvent`; reconciler owns all event writes
- Idempotent, no-churn updates (re-sync identical data twice = zero database churn)
- Terminal-outcome propagation for 1:1 and multi-record runs
- Mixed-outcome aggregation rule: "any-success-wins-once-all-terminal" (prevents single bad observation from regressing otherwise-successful runs)
- Status-vocabulary unification (LCO, Gemini, SOAR statuses onto one shared ranking)

**Competitive differentiators:**
- Per-record status detail preserved (e.g., "3/4 nights completed, 1 weathered")
- Provenance-blind coverage-gap analysis (counts all `CampaignRun`s regardless of source)
- Visual distinction of unused allocations

### Architecture Approach

Three major phases: (1) investigation spike settling schema/identity-key for all adapters, (2) adapter consolidation (adapters write `CampaignRun`, reconciler projects events), (3) outcome propagation (separate pass reading confirmed `CampaignRunObservation` links). The critical risk: `CampaignRun.campaign` is currently NOT NULL, but LCO and Gemini syncs routinely encounter records with no campaign. Must resolve: (a) whether `campaign` becomes nullable, (b) new identity field for queue-sourced runs, (c) new `UniqueConstraint` scoped by identity. These are settled before any adapter rewiring.

**Major architectural components:**
1. Scheduler entry point — orchestrates discovery → 3 adapters → reconcile sweep → outcome propagation
2. Discovery sweep — loops watch-list, creates `ObservationRecord` rows
3. Three rewired adapters — each calls `write_and_reconcile_campaign_run()` helper; LCO/Gemini also write automatic `CampaignRunObservation` link
4. Reconcile sweep — safety net re-deriving calendar events
5. Outcome propagation — separate pass deriving `run_status` from confirmed links, guarded-updating, refreshing calendar title

### Critical Pitfalls

1. **SQLite write-lock collision** between scheduled sync and concurrent staff action — **Mitigation:** `OPTIONS['timeout']` + no-overlap enforcement (flock or scheduler-native).

2. **Silent scheduled-job failure** — in-command error handling cannot detect scheduler itself failing to invoke — **Mitigation:** heartbeat/dead-man's-switch as first-class feature.

3. **Credential leakage through unattended-job logs** — new execution context exposes more of the command — **Mitigation:** audit every log line and exception message in adapters' new code paths; keep job-level credential out of CLI arguments.

4. **Dual-write window during migration** — unmigrated adapter's direct writes + migrated adapter's reconciler-projected writes = duplicates — **Mitigation:** explicit cutover sequencing; verify reconciler's ownership guard leaves unmigrated events alone; treat migration duplicates as attribution-queue resolution.

5. **Adapter's idempotency key breaks against `CampaignRun` schema** — LCO URL/Gemini ID/classical start-time-tolerance key was designed for `CalendarEvent`, not `CampaignRun` — **Mitigation:** explicit per-adapter verification; no-churn test (re-sync identical data twice = zero field changes).

6. **One bad observation regresses run's status** — naive aggregation fails for multi-record runs — **Mitigation:** explicit dominance-order table before implementation; test mixed-outcome fixtures.

7. **Status derivation fires on unconfirmed link, or bypasses confirmed `CampaignRunObservation`** — outcome propagation must read *only* confirmed links, never score candidates — **Mitigation:** guard: run with zero confirmed links stays as-is; test proving unconfirmed candidate never changes `run_status`.

## Implications for Roadmap

**Recommended 8-phase structure:**

1. **Investigation Spike — Schema & Identity** (Phase 1): Settles campaign-nullability, `source_identifier` field, new `UniqueConstraint`. Blocks everything after. Mirrored on Phase 26's role in v2.2.

2. **Shared Helper & Groundwork** (Phase 2): `write_and_reconcile_campaign_run()` helper tested in isolation; gates adapters but independent.

3. **Scheduling Mechanism Spike** (Phase 3): Verifies cron vs. task-queue against real deployment; settles healthchecks/credential handling/overlap prevention. Can research in parallel with Phases 4-6.

4. **ADAPT-01 — Classical Adapter** (Phase 4a): Simplest identity key; validates shared helper against classical data.

5. **ADAPT-02 — LCO Queue Adapter** (Phase 4b): Introduces `source_identifier`; first real automatic `CampaignRunObservation` linking.

6. **ADAPT-03 — Gemini Queue Adapter** (Phase 4c): Validates pattern generalizes to second facility identity scheme.

7. **Discovery Sweep & Watch-List** (Phase 6): Can develop in parallel but wire into orchestrator only after Phase 4b exists.

8. **Outcome Propagation** (Phase 5): Depends on adapters writing `CampaignRunObservation` links; implements the mixed-outcome aggregation rule.

9. **Scheduler Entry Point** (Phase 7): Last; orchestrates every step once independently functional.

10. **Carried-Forward Work** (Phase 8): STATUS-01/02 unification, GAPB-01, UNUSED-01 — direct consequences of adapter rewiring, no independent cost.

**Phase ordering rationale:**
- Schema/identity (Phase 1) gates everything; must come first.
- Adapters sequential (simplest first), each validating the shared pattern; all must ship before outcome propagation has data to read.
- Outcome propagation separate from adapter rewiring (preserves reconciler's pure-projection contract).
- Carried-forward work last (depends on new `CampaignRun`s existing).

## Research Flags

**Phases needing deeper research during planning:**
- **Phase 1 (schema spike):** Whether `campaign` nullability is simple schema change or requires data-migration backfill; whether classical runs have identity surface supporting `source_identifier` field.
- **Phase 3 (scheduling spike):** Real target deployment's cron vs. systemd-timer preference; healthchecks.io acceptability; flock availability.
- **Phase 5 (outcome propagation):** Validate mixed-outcome rule against real historical data — do any existing multi-record runs have outcomes misclassified by "any-success-wins"?

**Phases with standard patterns (skip research):**
- **Phase 2:** Composition of existing building blocks; no research beyond spike's schema decision.
- **Phases 4a-c:** Straightforward refactor; validated by existing SYNC-04-style no-churn tests.
- **Phase 6:** Generalizes existing `backfill_lco_observation_records` logic.
- **Phase 8:** All three features are consequences of adapter rewiring; no independent design research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | HIGH | Direct codebase inspection; cron recommendation corroborated across sources; matches single-server/few-jobs reality. |
| **Features** | HIGH | Rooted in `.planning/PROJECT.md` milestone scope and v2.2 infrastructure; aligns with PR #43's feature-complete bar. |
| **Architecture** | HIGH | Grounded in v2.2 codebase; Phase 26 already settled per-adapter identity keys; research extends to write-time surface. Critical Integration Risk *identified*, not unresolved. |
| **Pitfalls** | HIGH | All seven derive directly from this repo's code patterns and real operational scenarios; prevention strategies mirror existing codebase patterns. |

**Overall: HIGH** — Confidence is high because this is incremental infrastructure on a mature codebase with proven patterns. Main uncertainty (campaign-nullability schema) is identified upfront with clear resolution path (Phase 1 spike).

### Gaps to Address

- **Real deployment scheduling constraints:** CLAUDE.md lacks production deployment infrastructure details; Phase 3 spike must validate against actual target host.
- **Real data validation of mixed-outcome rule:** Needs validation against FOMO's historical run portfolio.
- **Classical adapter's write-time identity surface:** Phase 1 spike must confirm whether classical runs have facility-specific key (like LCO/Gemini) or only tolerance-windowed match.
- **Scheduler credential handling specifics:** Depends on Phase 3's scheduler choice (cron vs. task queue handle credentials differently).

## Sources

### Primary (HIGH)
- `solsys_code/campaign_reconciler.py`, `models.py`, `campaign_utils.py`, management commands
- `.planning/PROJECT.md`, Phase 26 spike decision doc
- `src/fomo/settings.py`

### Secondary (MEDIUM)
- django-crontab package health, Huey docs, Healthchecks.io docs, Django SQLite locking
- Kubernetes Job API documentation, GitHub Actions job-matrix patterns

---

*Research completed: 2026-09-01*
*Ready for roadmap creation: yes*
