# Feature Research

**Domain:** Automatic sync of robotically-scheduled observations + outcome propagation to a canonical run record (FOMO v2.3, "Automatic Run Sync & Outcome Propagation")
**Researched:** 2026-09-01
**Confidence:** MEDIUM (HIGH on FOMO-codebase facts — direct source reads of `campaign_reconciler.py`, `models.py`, `campaign_utils.py`, `sync_lco_observation_calendar.py`; MEDIUM on general cross-domain patterns — CI/CD conclusion aggregation and Kubernetes Job status are well-documented but generically-sourced, not astronomy-specific, since no TOM-Toolkit-specific prior art exists for this exact problem)

## Context

There is no direct astronomy-domain precedent for "sync a robotic scheduler's outcomes up to a program record" — TOM Toolkit itself stops at `ObservationRecord` (one row per facility submission) and has no higher-level "campaign"/"program" concept; FOMO's `CampaignRun` is a FOMO-original abstraction. The closest real prior art is general software engineering: CI/CD systems aggregating per-job status into a pipeline "conclusion" (GitHub Actions, Jenkins), and Kubernetes' Job controller aggregating per-pod-index outcomes into Job status. Both of those domains have already solved "many child task outcomes, one parent record, don't let a single failure erase a partial success" — this is the primary transferable pattern used below. Findings are framed against FOMO's own already-shipped v2.2 infrastructure (`campaign_reconciler.py`, `campaign_attribution.py`, `CampaignRunObservation`), since that infrastructure is what v2.3 has to extend correctly, not invent from scratch.

## Feature Landscape

### Table Stakes (Users Expect These)

Features required for the operator's stated "feature complete" bar (PR #43) — the milestone doesn't ship without these.

| Feature | Why Expected | Complexity | Notes / Dependencies |
|---------|--------------|------------|-----------------------|
| Unattended recurring invocation of all three sync commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`) | This *is* the milestone's stated goal — "no operator running any command" | MEDIUM | Mechanism (cron vs. task queue) is explicitly deferred to the milestone's own phase-time spike; depends on real deployment constraints, not research. No FOMO infra dependency — these commands already run standalone. |
| Failure visibility (a broken/stalled sync is noticed, not silently absent) | Table stakes for *any* unattended job — an operator who has to notice a gap on the calendar by chance is not "unattended," it's "unattended and untrustworthy" | LOW–MEDIUM | Minimum bar: non-zero exit code + logged traceback + some "last successful run" signal an operator can check (log line, status row, or cron's own mail-on-failure). Does not need a dashboard — see Anti-Features. |
| Watch-list-driven discovery sweep (no per-invocation `--proposal`/`--name-prefix` args) | `backfill_lco_observation_records` today requires an explicit proposal + prefix per call — that's fundamentally incompatible with "no operator action" | MEDIUM | Needs a small config surface (Django setting or DB table of proposals/prefixes to watch) — new, but same shape as existing `SITE_TELESCOPE_MAP`-style static config already in `calendar_utils.py`. |
| Adapter consolidation — adapters create/update `CampaignRun`s instead of writing `CalendarEvent`s directly (ADAPT-01..03) | Structural prerequisite: outcome propagation onto `CampaignRun.run_status` is meaningless if the run that "requested" an observation doesn't exist as a `CampaignRun` in the first place for queue/classical syncs | HIGH | Rewires three modules that currently call `insert_or_create_calendar_event()` directly. **Depends on:** `campaign_reconciler.reconcile_run()` (v2.2) as the *only* thing allowed to write `CalendarEvent`s from then on — adapters become pure `CampaignRun` writers, exactly the separation the reconciler was built to enable. Also depends on Phase 26's settled identity-key mapping (each adapter's existing natural key → a `CampaignRun` natural key). |
| Idempotent, no-churn create-or-update for automatically-created `CampaignRun`s | Every existing FOMO ingest path (`insert_or_create_campaign_run`, `insert_or_create_calendar_event`) already enforces this; an automatic path that re-writes/duplicates on every scheduled tick would be a regression, not a new risk | LOW | Direct reuse of `insert_or_create_campaign_run()` — no new mechanism needed, just adapters calling it instead of writing events. |
| Terminal-outcome propagation for the simple 1:1 case (one `CampaignRun` realised by exactly one `ObservationRecord`) | This is issue #37's original Stage 4 ask, in its simplest form, and the case FOMO's existing status vocabularies (`_FAILURE_PREFIX_BY_STATUS`, `map_observation_status`) already model per-record | MEDIUM | **Depends on:** `CampaignRunObservation` (Phase 28) as the *only* legitimate source of "which record(s) belong to this run" — must never infer the link from date/instrument overlap at propagation time; that inference is attribution's job and already requires a staff confirmation (ATTRIB-03) before a `CampaignRunObservation` row exists at all. |
| Automatic derivation never fires ahead of evidence | A run with zero linked `ObservationRecord`s, or all still pending, must not get a terminal `run_status` invented for it | LOW | Pure guard clause — same shape as `campaign_reconciler._skip_reason()`'s existing itemized-skip idiom. |
| Automatic derivation is idempotent / non-flapping across repeated reconcile passes | The reconciler's whole design principle (re-derive from current state every call, per RECON-01) must extend to status derivation, or a scheduled job re-computing `run_status` every N minutes will produce visible churn/flapping | MEDIUM | **Depends on:** `campaign_reconciler.reconcile_run()`'s existing idempotent, level-triggered shape — status derivation should be one more pure function of current linked-record state, computed fresh each call, not an incremental state-machine transition. |
| Status-vocabulary unification (STATUS-01/02, carried from v2.2) | Outcome propagation cannot compare "is this record's status worse/better than that one" without first collapsing LCO's (`WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED`/`COMPLETED`), Gemini's (`ready` flag + ToO type), and `CampaignRun.RunStatus`'s own 8-value vocabulary onto one shared precedence order | MEDIUM–HIGH | **Depends on:** the four already-separate prefix maps (`_CLASSICAL_STATUS_PREFIX`, `_FAILURE_PREFIX_BY_STATUS`, a would-be `_RUN_STATUS_CALENDAR_PREFIX` moved into the reconciler, and `calendar_display_extras._TERMINAL_PREFIXES`) — this milestone is the first time they need to agree on more than *display*, they need to agree on *ranking* for aggregation. |

### Differentiators (Competitive Advantage)

Not required for "feature complete," but where FOMO would visibly beat the ad-hoc status quo (a human periodically checking the LCO portal / an email from Gemini and updating a spreadsheet).

| Feature | Value Proposition | Complexity | Notes / Dependencies |
|---------|-------------------|------------|-----------------------|
| Non-regressive, "any-success-wins" mixed-outcome aggregation for multi-record runs | Directly answers the milestone's own open design question and the stated anti-pattern risk — see "Open Design Question" below for the concrete recommended rule | HIGH | **Depends on:** `CampaignRunObservation` (enumerates the linked set), the unified status vocabulary above (to rank each record), and a guard against overwriting a manually-set terminal status (`mark_cancelled`/`mark_weather_failure`) or a downstream human-owned lifecycle stage (`REDUCED`/`PUBLISHED`). |
| Provenance-blind coverage-gap analysis (GAPB-01, carried from v2.2) | `campaign_gap.claimed_dates()` today only reads `CampaignRun` rows created by CSV import / web submission — once queue/classical syncs also write `CampaignRun`s (via ADAPT-01..03), gap analysis becomes correct for *all* sources instead of undercounting real allocated nights as "unclaimed" | MEDIUM | **Depends on:** ADAPT-01..03 shipping first — gap analysis gets this "for free" once every ingest path writes through the same model, it just needs to stop being source-scoped. |
| Unused-allocation visual distinction (UNUSED-01, carried from v2.2) | Distinguishes "allocated but never observed" (a class-wide/queue window that expired with zero completed records) from "allocated and used" on the calendar — a genuinely new signal, not available from any of LCO/Gemini/SOAR's own UIs in this composed form | MEDIUM | **Depends on:** outcome propagation existing first (needs to know a window's terminal state to render it as unused vs. used) and the reconciler's existing container-vs-per-night event distinction (v2.2 four-stage pipeline). |
| Per-record status detail surfaced alongside the aggregate (e.g. "3/4 nights completed, 1 weathered" rather than collapsing straight to one field) | Mirrors what CI dashboards (GitHub Actions per-job status inside one workflow conclusion) and Kubernetes Job's `succeededIndexes`/`failedIndexes` both preserve — losing this detail is exactly how "one bad observation regresses the whole run" becomes invisible/undebuggable to an operator | MEDIUM | **Depends on:** `CampaignRunObservation` already carrying the per-record link; this is a display/summary feature over data that already exists once ADAPT-01..03 + outcome propagation ship — no new model needed, a computed property or the approval-queue table's own render helpers would do. |
| Reusing the attribution queue's confirm/dismiss gate for outcome-propagation edge cases (e.g. an ambiguous newly-discovered record that could belong to more than one open `CampaignRun`) | Turns an automatic-sync failure mode (mis-attributed outcome) into an operator-visible queue item instead of a silent wrong write — this is exactly the trust property Phase 28 built for record *discovery*, extended to outcome propagation | MEDIUM | **Depends on:** `campaign_attribution.py`'s existing scored-candidate + hard campaign/target boundary gate machinery, and its existing HIGH/MEDIUM/LOW confidence banding — outcome propagation should only auto-apply against a `CampaignRunObservation` row that already exists (staff-confirmed), never against a raw attribution *candidate*. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| "Any failure regresses the run" naive worst-status-wins aggregation | Looks like the simplest possible rule — just take the worst terminal status across all linked records | This is the exact anti-pattern the milestone flags: a class-wide run with 9/10 nights `COMPLETED` and 1 `WEATHER_TECH_FAILURE` would silently report as `WEATHER_TECH_FAILURE` overall, hiding real science data behind a status that reads as "this run got nothing" | Any-success-wins-once-all-terminal rule (see Open Design Question) — only regress to a failure status when *every* linked record is a failure |
| Real-time/webhook-push ingestion from LCO/Gemini/SOAR | Sounds more "automatic" than polling on a schedule | None of the three facilities' APIs FOMO already integrates with expose an inbound webhook FOMO could register for (all existing FOMO/TOM-Toolkit facility clients are pull-based, per-record REST calls); building a webhook receiver is new attack surface and new infra (public endpoint, auth) for a milestone whose own spike is choosing between cron and a task queue, both pull-based | Poll on a documented recurring schedule (the milestone's own chosen mechanism) — "unattended" does not require "instant" |
| Unbounded automatic retry/backoff loops on facility API failures within a single sync pass | Feels more resilient than failing fast | Contradicts FOMO's own existing convention (SYNC-08: explicit timeout, single attempt, no retry loop, precisely to bound a single sync run's worst-case duration and avoid hammering a struggling upstream API); a stuck retry loop inside an unattended job is worse than a fast, visible failure the next scheduled run will naturally retry | Single attempt with a timeout (existing pattern) + let the next scheduled invocation be the retry — this is what "recurring schedule" already buys for free |
| Silent auto-creation of a `CampaignRun` for any newly-discovered orphan `ObservationRecord`, with no staff confirmation | Would make "fully automatic" feel more complete — no queue to check | Directly violates the hard rule Phase 28 was built around (ATTRIB-03 — no association without explicit staff confirmation) and repeats the exact anti-pattern quick task `260705-l1v` fixed once already (silently fabricating a placeholder record rather than surfacing ambiguity) | Auto-*discovery* is fine (the unattended sweep replacing `backfill_lco_observation_records`'s manual invocation); auto-*attribution* to an existing `CampaignRun` stays gated behind the existing confirm/dismiss queue |
| A full notification/alerting pipeline (Slack, email digest, PagerDuty-style escalation) for sync failures | Sounds like the "proper" way to make a failure visible | Well beyond the stated bar ("a failure is visible to an operator rather than silently disappearing") and a genuinely new integration surface (credentials, delivery reliability, its own failure modes) for a milestone that's about sync/propagation, not ops tooling | A log line + an operator-checkable "last successful run" signal (status file, or a queryable timestamp) satisfies the stated requirement; a notification pipeline is a reasonable *future* differentiator, not this milestone's scope |
| Rewriting `run_status` retroactively for runs already in a human-owned downstream lifecycle stage (`REDUCED`/`PUBLISHED`) based on late-arriving observation data | Seems "more correct" to keep everything in sync | Those stages represent human judgment about post-observation work (data reduction, publication) that has nothing to do with an `ObservationRecord`'s facility-reported terminal status — auto-overwriting them would erase real human progress with a machine's stale inference | Automatic derivation only ever writes `run_status` while the run is at or before `OBSERVED`; once a human has advanced it past that point, automatic propagation stops touching the field (see Open Design Question) |

## Open Design Question: Mixed-Outcome `run_status` Derivation

This is the milestone's explicitly flagged open question. Recommended rule, informed by FOMO's own idempotent-reconciler idiom and by CI/CD "conclusion" aggregation precedent (GitHub Actions job-matrix conclusions, Kubernetes Job `succeededIndexes`/`failedIndexes`):

1. **Compute fresh every reconcile pass, from `CampaignRunObservation` alone.** Never mutate `run_status` incrementally from an event stream — derive it as a pure function of the *current* set of linked `ObservationRecord`s, exactly the way `campaign_reconciler.reconcile_run()` already re-derives calendar state from current run fields on every call. This guarantees RECON-01's "running it twice changes nothing" property extends to status.

2. **Classify each linked record into three buckets**, reusing the vocabulary the unification work (STATUS-01/02) produces: `PENDING` (not yet terminal), `SUCCESS` (`COMPLETED`), `FAILURE` (`WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED`, and Gemini's equivalent terminal-negative states).

3. **Aggregate rule, in this precedence order:**
   - Zero linked records → don't touch `run_status` (no evidence yet).
   - Any record still `PENDING` → `PLANNED` (in progress; don't jump to a terminal verdict early).
   - All records terminal, **at least one `SUCCESS`** → `OBSERVED` (any-success-wins — a 9-succeeded/1-weathered class-wide run reports as observed, because usable data exists; this is the concrete fix for "a single bad observation regressing an otherwise-successful multi-record run").
   - All records terminal, **all `FAILURE`** → `WEATHER_TECH_FAILURE` (or `CANCELLED`, if every failure was itself a cancellation) — only a wholly-failed run regresses.

4. **Two sticky exceptions, both borrowed from patterns FOMO already ships:**
   - A staff-set terminal status (`mark_cancelled`/`mark_weather_failure`, already shipped in v2.1/Phase 23) is authoritative and must not be silently overwritten by a later automatic derivation pass — mirror the reconciler's existing `_may_write()`/ownership-guard idiom, applied to the status field instead of the calendar event.
   - Once a human has advanced `run_status` past `OBSERVED` (into `REDUCED`/`PUBLISHED`), automatic derivation stops writing that field entirely — those stages are human judgment, not facility-reported outcome.

**Complexity: HIGH.** Not because the rule itself is complex, but because it touches the status-vocabulary unification, requires a reliable per-record terminal classification for three different facility APIs, and needs the sticky-exception guard to avoid regressing either a staff decision or downstream human lifecycle progress — get any one of those three wrong and the milestone reproduces the exact anti-pattern it's trying to avoid.

## Feature Dependencies

```
Status-vocabulary unification (STATUS-01/02)
    └──requires──> (LCO/Gemini/SOAR per-record terminal-status maps already exist, just unify them)

Adapter consolidation (ADAPT-01..03)
    └──requires──> campaign_reconciler.reconcile_run() (v2.2, already shipped)
    └──requires──> Phase 26's settled per-adapter identity-key mapping (v2.2, already shipped)

Unattended scheduling
    └──requires──> Adapter consolidation (ADAPT-01..03)
    └──requires──> Failure visibility (a silently-broken unattended job is worse than a manually-run one)
    └──requires──> Watch-list config (replaces per-invocation --proposal/--name-prefix args)

Terminal-outcome propagation (simple 1:1 case)
    └──requires──> CampaignRunObservation (Phase 28, already shipped)
    └──requires──> Status-vocabulary unification (STATUS-01/02)

Mixed-outcome aggregation (multi-record case, the open design question)
    └──requires──> Terminal-outcome propagation (simple case)
    └──requires──> CampaignRunObservation as the sole source of "which records belong to this run"

Provenance-blind coverage-gap analysis (GAPB-01)
    └──requires──> Adapter consolidation (ADAPT-01..03) — queue/classical CampaignRuns must exist first

Unused-allocation visual distinction (UNUSED-01)
    └──requires──> Terminal-outcome propagation (needs a window's terminal state to render used vs. unused)
    └──requires──> v2.2 four-stage window pipeline (container vs. per-night events)

Attribution-queue reuse for outcome-propagation edge cases ──enhances──> Terminal-outcome propagation
    (does not gate it — only needed for ambiguous newly-discovered records, not the common case)
```

### Dependency Notes

- **Adapter consolidation must land before unattended scheduling is meaningful:** scheduling the *current* adapters unattended would just automate direct `CalendarEvent` writes with no `CampaignRun` behind them — outcome propagation would have nothing to propagate to. This is why the milestone context lists ADAPT-01..03 as a target feature rather than a "nice to have."
- **Outcome propagation must never bypass `CampaignRunObservation`:** the temptation, once adapters write `CampaignRun`s directly, is to compute status straight from a live `ObservationRecord` query filtered by date/instrument overlap — that recreates attribution logic without its staff-confirmation gate (ATTRIB-03) and would silently regress a hard rule the project already paid a full phase (28) to establish structurally.
- **Status-vocabulary unification is a hard prerequisite, not parallel work,** for both terminal-outcome propagation and mixed-outcome aggregation — you cannot rank "is this record's outcome worse than that one" across three different facility vocabularies without first collapsing them onto one ordered scale.
- **GAPB-01/UNUSED-01 are downstream beneficiaries, not independent features:** both were carried forward from v2.2 specifically *because* they're direct consequences of ADAPT-01..03 landing, per PROJECT.md's own framing — they don't need separate design work beyond "read from the now-complete `CampaignRun` set."

## MVP Definition

### Launch With (v2.3 core — must ship for "feature complete")

- [ ] Phase-time spike settling cron vs. task queue against real deployment constraints — every other unattended-scheduling decision depends on this
- [ ] Adapter consolidation (ADAPT-01..03) — adapters write `CampaignRun`s, reconciler owns all `CalendarEvent` writes
- [ ] Unattended, watch-list-driven recurring invocation of all three sync commands, with failure visible to an operator
- [ ] Watch-list-driven discovery sweep replacing `backfill_lco_observation_records`'s per-invocation args
- [ ] Status-vocabulary unification (STATUS-01/02)
- [ ] Terminal-outcome propagation for the simple 1:1 case (single linked `ObservationRecord` per run)
- [ ] Mixed-outcome aggregation rule for multi-record runs (the "any-success-wins, sticky staff/lifecycle overrides" rule above) — this is explicitly in scope per the milestone's own open question, not deferrable

### Add After Validation (v2.3.x, if the mixed-outcome rule needs iteration)

- [ ] Per-record status detail surfaced in the UI alongside the aggregate (differentiator, not required for the propagation mechanism itself to work)
- [ ] Provenance-blind coverage-gap analysis (GAPB-01) and unused-allocation visual distinction (UNUSED-01) — already carried forward, land once ADAPT-01..03 is stable

### Future Consideration (post-v2.3)

- [ ] Reusing the attribution queue for outcome-propagation edge cases (ambiguous newly-discovered records) — only needed once real unattended operation surfaces such a case
- [ ] Any notification/alerting pipeline beyond log-based failure visibility
- [ ] ESO sync (SEED-001/002 stay dormant this milestone)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|----------------------|----------|
| Adapter consolidation (ADAPT-01..03) | HIGH | HIGH | P1 |
| Unattended scheduling + failure visibility | HIGH | MEDIUM | P1 |
| Status-vocabulary unification | HIGH | MEDIUM | P1 |
| Terminal-outcome propagation (1:1 case) | HIGH | MEDIUM | P1 |
| Mixed-outcome aggregation rule | HIGH | HIGH | P1 |
| Provenance-blind coverage-gap analysis | MEDIUM | LOW (once ADAPT-01..03 lands) | P2 |
| Unused-allocation visual distinction | MEDIUM | MEDIUM | P2 |
| Per-record status detail surfaced | MEDIUM | LOW | P2 |
| Attribution-queue reuse for propagation edge cases | LOW–MEDIUM | MEDIUM | P3 |
| Notification/alerting pipeline | LOW (for this milestone) | HIGH | P3 (explicitly deferred) |

**Priority key:**
- P1: Must have — this is what makes the branch behind PR #43 "feature complete"
- P2: Should have — direct, low-cost consequence of P1 landing
- P3: Nice to have — real, but not part of this milestone's stated bar

## Cross-Domain Pattern Analysis

| Pattern | How CI/CD or Kubernetes Handles It | FOMO's Equivalent | Our Approach |
|---------|--------------------------------------|--------------------|---------------|
| Many child task outcomes → one parent status | GitHub Actions: workflow "conclusion" is `failure` only if a *required* job failed; a matrix with some failing legs still shows per-leg detail, not just one collapsed verdict | `CampaignRun.run_status` from N linked `ObservationRecord`s via `CampaignRunObservation` | Any-success-wins-once-all-terminal (see Open Design Question), never a naive worst-status-wins collapse |
| Partial failure within a batch of indexed sub-tasks | Kubernetes Job (1.28+): `succeededIndexes`/`failedIndexes` are tracked as sets, not collapsed to a single boolean, precisely so partial progress isn't lost | Per-night/per-record status inside a multi-night `CampaignRun` | Preserve per-record detail (via `CampaignRunObservation`) even after computing the aggregate — don't discard the data that produced the verdict |
| Idempotent, level-triggered status computation | Kubernetes controller-runtime: reconcilers must be idempotent — same observed state always yields the same outcome, recomputed from scratch each pass, never incrementally mutated | `campaign_reconciler.reconcile_run()` (v2.2, already ships exactly this for calendar events) | Extend the same idiom to `run_status` derivation — one more pure function of current state, not a new state machine |
| Ownership/authority guard before an automatic writer touches shared state | Kubernetes: a controller must not clobber a field another controller (or a human, via `kubectl edit`) owns | `campaign_reconciler._may_write()` already guards `CalendarEvent` writes against staff/attribution ownership | Apply the identical guard shape to `run_status`: automatic derivation may only write while the run is at/before `OBSERVED` and has no staff-set terminal override |

## Sources

- FOMO codebase (HIGH confidence, primary source): `/home/tlister/git/fomo_devel/.planning/PROJECT.md`, `solsys_code/campaign_reconciler.py`, `solsys_code/models.py` (`CampaignRun`, `CampaignRunObservation`), `solsys_code/campaign_utils.py` (`map_observation_status`), `solsys_code/management/commands/sync_lco_observation_calendar.py` (`_FAILURE_PREFIX_BY_STATUS`), `solsys_code/management/commands/load_telescope_runs.py` (`_CLASSICAL_STATUS_PREFIX`), `solsys_code/templatetags/calendar_display_extras.py` (`_TERMINAL_PREFIXES`)
- [Kubernetes 1.28: Improved failure handling for Jobs](https://kubernetes.io/blog/2023/08/21/kubernetes-1-28-jobapi-update/) — per-index status tracking, MEDIUM confidence (official k8s blog)
- [Kubernetes Jobs documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/) — MEDIUM confidence (official docs)
- [Kubebuilder Book — Good Practices](https://book.kubebuilder.io/reference/good-practices.html) — idempotent reconciler pattern, MEDIUM confidence (official framework docs)
- [The Reconciler Pattern](https://www.farishuskovic.dev/blog/k8s-reconciler-pattern/) — LOW-MEDIUM confidence (independent blog, corroborated by the official docs above)
- General CI/CD "conclusion" aggregation (GitHub Actions job/workflow status model) — LOW-MEDIUM confidence, drawn from general industry knowledge rather than a single cited source

---
*Feature research for: automatic run sync and outcome propagation, FOMO v2.3*
*Researched: 2026-09-01*
