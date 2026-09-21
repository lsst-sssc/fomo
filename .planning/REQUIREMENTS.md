# Requirements: Telescope Runs Calendar — v2.4 Observation-First Calendar

**Defined:** 2026-09-03
**Core Value:** The calendar is driven by what actually happened — one event per `ObservationRecord`, narrowing on every save with no operator action; allocations project intent nights until a real observation retires them; campaigns annotate, never own; the pipeline runs unattended on the real host.

**Grounding:** `.planning/notes/observation-first-calendar-layering.md` (D1–D5), `.planning/spikes/WRAP-UP-SUMMARY.md`, and the `spike-findings-fomo_devel` project skill (non-negotiables from spikes 001–004). v2.3's Phase 31 verdicts and plan 32-01 Tasks 1–2 (`f03553a`, `18ecded`) are kept as the allocation-without-campaign foundation. Phase numbering continues from the superseded v2.3 (last phase: 32).

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Observation Projector (base layer)

- [x] **PROJ-01**: Every LCO/SOAR `ObservationRecord` has exactly one `CalendarEvent`, keyed by `facility.get_observation_url()` (the namespace the existing LCO sync already uses), created or updated in place — never a second event for the same record
- [x] **PROJ-02**: An event's span follows the record's stage: the request window while queued, the placed block once `scheduled_start`/`scheduled_end` are set, the observed block once COMPLETED — the existing `record_time_window` rule
- [x] **PROJ-03**: A terminal-negative record (`WINDOW_EXPIRED` / `CANCELED` / `FAILURE_LIMIT_REACHED`) keeps a visibly marked event on its window night — never silently dropped
- [x] **PROJ-04**: Series identity for a record in an `ObservationGroup` is carried by real foreign keys on `CalendarEventMeta` (`observation_record`, `observation_group`) — a shared title stem and a link back to the group; spike 002's title-suffix stopgap is not the carrier
  - *Scope split (recorded 2026-09-03 during Phase 33 planning):* Phase 33 delivers the carrier — the two foreign keys, their migration and their read-only admin exposure. The **shared title stem** clause is delivered by the Phase 34 projector, which is the only writer of these fields and of an event's title (Phase 33 is schema and semantics only, per 33-CONTEXT.md D-08, and D-12 removes text from titles rather than adding it). Phase 34 must satisfy the title-stem clause alongside PROJ-06's compact-title requirement.
- [x] **PROJ-05**: Re-projecting an unchanged record writes nothing (no-churn); the projector never creates, modifies, or deletes an event it does not own, and never writes the reconciler's `RUN:`/allocation namespace
- [x] **PROJ-06**: Event titles have a compact form that fits a month cell on the calendar
- [x] **SCHED-06** (carried from v2.1/v2.3, re-scoped): A user can watch a record's event narrow queued → scheduled → observed on the calendar with no operator command, proven against the real `KEY2026B-004` records over live nights (closes spike 004's PARTIAL verdict)

### Trigger

- [x] **TRIG-01**: A FOMO-owned Django `post_save` receiver on `ObservationRecord`, registered in `apps.ready()`, re-projects the record's event on every save — including a schedule-only placement save (which TOM's `observation_change_state` hook misses) and the `updatestatus` path
- [x] **TRIG-02**: The receiver is single-record, idempotent, and cheap enough to run inside the caller's transaction on every save; a projector error is logged and never aborts the record save
- [x] **TRIG-03**: A sweep management command re-projects records as the backstop for `queryset.update()`/`bulk_create()` paths and for backfill, with `--dry-run`, per-record failure isolation, and a paired pre-executed demo notebook

### Allocation Layer & Handoff

- [x] **ALLOC-01**: An allocation with a resolved site and a classical/awarded window (a `CampaignRun` with or without a campaign: classical schedule line, approved submission, TBD/range run once resolved) projects one per-night sunset→sunrise event per window night at its site; a queue-scheduled, class-wide, or satellite run keeps its single whole-window container event (Phase 26 verdict — a queue window is not a set of owned nights), annotated, never narrowed
- [x] **ALLOC-02**: Allocation nights are keyed by the site-local observing night (the date `sun_event` takes), not the UTC date — verified for a Chilean and an Australian site
- [x] **ALLOC-03**: An allocation night with a linked `ObservationRecord` (via `CampaignRunObservation`) has no allocation event; unlinking restores it; the observation's own event is untouched by either transition
- [x] **ALLOC-04**: `load_telescope_runs` creates or updates a campaign-less `CampaignRun` (`source=CLASSICAL`, with a collision-safe `source_identifier` per Phase 31's SCHEMA-03 finding) instead of writing calendar events directly, and the allocation projects the same per-night events the command wrote before, idempotently on re-run
- [x] **ALLOC-05**: The cutover from today's `load_telescope_runs`-written events and `RUN:{pk}:{date}` reconciler events to allocation events has an explicit, stated sequencing (migration or one-time command) that never leaves a duplicate or orphaned event on the calendar

### Campaign Annotation (reconciler inversion)

- [x] **ANNOT-01**: `CalendarEventMeta.run` means "attributed to", not "owned by"; `reconcile_run()` no longer adopts, re-keys, or detaches an event attributed to a run — it only annotates — so the base layer and the campaign layer can run side by side without one stealing the other's events
- [x] **ANNOT-02**: Campaign decoration of an observation-backed event (campaign prefix/label, link to its run) is rendered from the `CalendarEventMeta.run` link at display time, never written into the event's own fields, so base re-projection cannot erase it
- [x] **ANNOT-03**: `sync_lco_observation_calendar` is retired in favour of the projector + sweep (one writer per source; same key namespace, same events) with its runbook section, demo notebook, and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-facility-read-back caveat documented

### Unattended Operation (carried from v2.3)

- [x] **SCHED-08**: The projector sweep, the LCO/SOAR discovery backfill, and the reconciler run on a documented recurring cron + `flock -n` schedule (Phase 31's SCHED-07 verdict) with no operator action, guarded against overlapping invocations
- [x] **SCHED-09**: A failed unattended run is visible to an operator through two independent layers — in-command failure notification (reusing the existing `_notify_staff()` email idiom) and a heartbeat/dead-man's switch that also catches the scheduler itself failing to invoke
- [x] **SCHED-10**: No credential value (API keys, passwords) appears in any log line or notification generated by the unattended execution path
- [x] **DISCOVER-01**: An admin-editable watched-proposal list (e.g. a `WatchedProposal` model) replaces `backfill_lco_observations`' per-invocation `--proposal`/name-prefix arguments, so discovery of robotically scheduled observations runs unattended against every currently-watched proposal

### Progress Legibility (public)

- [x] **TALLY-01**: Any user (not only staff) sees on each run — on the campaign table row and the run detail — an ongoing tally of linked `ObservationGroup`s, linked `ObservationRecord`s, and nights observed / scheduled / expired-or-failed / unused-so-far, updating as the projector narrows
- [x] **TALLY-02**: The campaign page rolls the same tally up across the campaign's runs
- [ ] **TALLY-03**: `CampaignRun.run_status` is never set automatically from linked records — it stays a staff decision, made when the run's window has ended
- [x] **UNUSED-01** (carried from v2.2/v2.3): An awarded night that was never scheduled or observed (an allocation event still standing after its night has passed) is visually distinct on the calendar from a realised night

### Status Vocabulary & Gap Analysis (carried from v2.2/v2.3)

- [ ] **STATUS-01**: One status vocabulary replaces the three parallel prefix maps (`_CLASSICAL_STATUS_PREFIX`, `_FAILURE_PREFIX_BY_STATUS`, `_RUN_STATUS_CALENDAR_PREFIX`) which today agree with `calendar_display_extras._TERMINAL_PREFIXES` only by convention, and includes a placed-but-unobserved state (spike 002's `[SCHEDULED]` gap)
- [ ] **STATUS-02**: A general terminal-state classifier replaces the `status == 'COMPLETED'` check, once per-facility `get_terminal_observing_states()` differences are reconciled
- [ ] **GAPB-01**: `campaign_gap.claimed_dates()` counts every observation on the campaign calendar, not only those with a `CampaignRun`, so classical and queue time is no longer reported as unclaimed

## v2 Requirements

Deferred to a future release. Tracked but not in the current roadmap.

### Upstream Contribution

- **UPSTREAM-01**: Extract the projector + `post_save` receiver + sweep into a facility-agnostic module and contribute it to tomtoolkit (`tom_calendar` ships inside tomtoolkit 3.0.1, TOM-org maintained) — SEED-004; only once the projector has shipped in FOMO

### ESO Facility Sync

- **ESO-10**: `sync_eso_observation_calendar` management command (unblocked by Phase 13's Bypass verdict; SEED-001/002 stay dormant)
- **ESO-11**: Paired demo notebook for `sync_eso_observation_calendar`

### Submission Workflow

- **SUBMIT-06**: Trusted-program PI self-approval path
- **SUBMIT-07**: Submission status lookup for submitters

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Automatic `CampaignRun.run_status` aggregation from linked records (v2.3's OUTCOME-01..04) | Dropped, not deferred: a run such as "daily 3I monitoring, LCO 1m0, g/r/i, 2025-07-04 → 2025-09-01" holds several groups and up to ~60 records; any single status collapses information the calendar already shows per night, and an overall state is only meaningful once the window has ended — a staff judgement, not a rule (TALLY-03) |
| Adapters routing through `CampaignRun` (v2.3's ADAPT-01..06) | Dropped: the `CampaignRun` middleman is what the observation-first spikes removed; replaced by the base layer (PROJ/TRIG) and the allocation layer (ALLOC) |
| Per-night allocation events for queue-scheduled / class-wide / satellite runs | Phase 26's domain correction stands — a queue window is not a set of owned nights; those runs keep one whole-window container (ALLOC-01) |
| Live Gemini read-back | `GEMFacility.get_observation_status()`/`get_observation_url()` are hardcoded stubs; `sync_gemini_observation_calendar` remains submission-echo (ANNOT-03) |
| ESO sync (SEED-001/002) | LCO/SOAR only this milestone; ESO stays dormant on its own seeds |
| A task-queue scheduler (Celery/huey/APScheduler) | Phase 31 settled cron + `flock -n` against the real host |
| Auto-attribution of orphan events/records to a `CampaignRun` by scored similarity | Would recreate the unconfirmed-merge risk Phase 28 structurally closed; links are exact identity or human-confirmed only |
| A full alerting/notification pipeline | Email + a dead-man's-switch ping meets the "visible to an operator" bar (SCHED-09) |
| Upstreaming the projector now (SEED-004) | Must first prove itself in FOMO; tracked as UPSTREAM-01 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROJ-01 | Phase 34 | Complete |
| PROJ-02 | Phase 34 | Complete |
| PROJ-03 | Phase 34 | Complete |
| PROJ-04 | Phase 33 (carrier fields), Phase 34 (shared title stem) | Complete |
| PROJ-05 | Phase 34 | Complete |
| PROJ-06 | Phase 34 | Complete |
| SCHED-06 | Phase 34 | Complete |
| TRIG-01 | Phase 34 | Complete |
| TRIG-02 | Phase 34 | Complete |
| TRIG-03 | Phase 34 | Complete |
| ALLOC-01 | Phase 35 | Complete |
| ALLOC-02 | Phase 35 | Complete |
| ALLOC-03 | Phase 35 | Complete |
| ALLOC-04 | Phase 35 | Complete |
| ALLOC-05 | Phase 35 | Complete |
| ANNOT-01 | Phase 33 | Complete |
| ANNOT-02 | Phase 33 | Complete |
| ANNOT-03 | Phase 34 | Complete |
| SCHED-08 | Phase 36 | Complete |
| SCHED-09 | Phase 36 | Complete |
| SCHED-10 | Phase 36 | Complete |
| DISCOVER-01 | Phase 36 | Complete |
| TALLY-01 | Phase 37 | Complete |
| TALLY-02 | Phase 37 | Complete |
| TALLY-03 | Phase 37 | Gaps Found |
| UNUSED-01 | Phase 37 | Complete |
| STATUS-01 | Phase 37 | Gaps Found |
| STATUS-02 | Phase 37 | Gaps Found |
| GAPB-01 | Phase 37 | Gaps Found |

**Coverage:**

- v1 requirements: 29 total
- Mapped to phases: 29 ✓
- Unmapped: 0

**Split requirements:** PROJ-04 is the only requirement whose clauses land in two phases — Phase 33
builds the `observation_record`/`observation_group` carrier, Phase 34's projector writes it and
supplies the shared title stem. See the scope-split note under PROJ-04 above.

---
*Requirements defined: 2026-09-03*
*Last updated: 2026-09-03 after roadmap creation (Phases 33-37; 29/29 mapped, no orphans, no duplicates); PROJ-04 scope split recorded 2026-09-03 during Phase 33 plan revision*
