# Requirements: FOMO — v2.2 One Canonical Run Record

**Defined:** 2026-07-26
**Core Value:** An observing run exists once, as a `CampaignRun`, and everything else is derived from it — the calendar events that show it, the observation records that realise it, and the coverage-gap analysis that counts it.

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Spike

Settles the questions milestone questioning deliberately left open. Blocks every other category.

- [ ] **SPIKE-01**: A decision doc settles the `source` vocabulary and how it interacts with `CampaignRun`'s two existing partial unique constraints, demonstrated by the real `CampaignRun` pk=1 and its 11 LCO-sourced calendar events coexisting without an `IntegrityError`
- [ ] **SPIKE-02**: A decision doc settles how each ingest adapter's existing calendar-event identity key maps onto a run — classical `(telescope, instrument, start_time ±5 min)`, LCO request URL, `GEM:{prog}/{obsid}`, and `CAMPAIGN:{pk}[:{date}]`
- [ ] **SPIKE-03**: A decision doc settles whether a class-wide (stage 2) run produces one event per candidate site or a single class-wide event, and states the reconciler's canonical event-key scheme, stable across all four pipeline stages
- [ ] **SPIKE-04**: A decision doc settles the migration and attribution strategy for the existing calendar events and runs, naming every integration point the companion-record rename touches

### Canonical Record

The model changes that make `CampaignRun` the single canonical observing-run record.

- [ ] **CANON-01**: A `CampaignRun` records which ingest path created it (`source`: web submission, classical file, LCO queue, Gemini queue, CSV import), and approval is required only for web submissions
- [ ] **CANON-02**: A run allocated to a telescope class (`2m0`/`1m0`/`0m4`) is distinguishable from a run whose site failed to resolve, via an explicit `telescope_class` field — today both appear as `site=None`
- [ ] **CANON-03**: A calendar event can be linked to the run it belongs to, through one generalised FOMO companion record that also retains the existing `is_verified` telescope-label flag; existing companion rows survive the change and all four existing integration points (admin registration, LCO sync command, view prefetch, calendar template) keep working
- [ ] **CANON-04**: An `ObservationRecord` can be linked to the `CampaignRun` it realises, through a link that records whether a human confirmed it
- [ ] **CANON-05**: A staff user can see a run's linked calendar events and observation records, and reach the run from an event

### Reconciler

Calendar events become a function of run state instead of a side effect of a staff click.

- [ ] **RECON-01**: Staff can run one command that projects and refreshes calendar events for every run, regardless of window length, source, or site-resolution state; running it a second time changes nothing (no new rows, no `modified` churn)
- [ ] **RECON-02**: A run resolved to a specific telescope produces one calendar event per night, spanning that site's sunset-to-sunrise twilight for that night (stage 1)
- [ ] **RECON-03**: A run allocated only to a telescope class produces one calendar event per day, spanning 00:00–23:59 (stage 2)
- [ ] **RECON-04**: A night whose `ObservationRecord` has been scheduled narrows to that record's window (stage 3), and a completed observation shows the final observed time range marked COMPLETED (stage 4)
- [ ] **RECON-05**: The reconciler never creates, modifies, or deletes a calendar event it does not own — hand-created entries, conference and proposal-deadline events, and un-attributed sync-command events are left untouched
- [ ] **RECON-06**: `--dry-run` reports exactly what would change with no writes; a run that fails to reconcile is reported and skipped rather than aborting the batch
- [ ] **RECON-07**: The approved, site-resolved 3I/ATLAS runs that no existing command can project (19 as of 2026-07-26) become visible on the calendar
- [ ] **RECON-08**: The reconciler is reachable from the existing approve / resolve_site / mark_cancelled / mark_weather_failure staff actions, so a single run reconciles immediately on a staff decision
- [ ] **RECON-09**: `backfill_range_calendar_events` is retired, its behaviour subsumed by the reconciler

### Attribution

Connecting existing events and records to their parent runs, without ever guessing silently.

- [ ] **ATTRIB-01**: Staff see a queue of suggested associations between existing calendar events or observation records and their likely parent run, showing the evidence for each candidate (matched telescope, date overlap, campaign, instrument-string similarity)
- [ ] **ATTRIB-02**: Candidates are confidence-scored and filterable by score, so staff can bulk-confirm the confident tail and hand-review only the ambiguous remainder
- [ ] **ATTRIB-03**: No association is ever created without explicit staff confirmation
- [ ] **ATTRIB-04**: A staff user can undo a confirmed association
- [ ] **ATTRIB-05**: The known real case is surfaced as a candidate — `CampaignRun` pk=1 (FTS/MuSCAT4, 7–21 July, Siding Spring E10) and its 11 LCO queue events (`2m0`/`2M0-SCICAM-MUSCAT`, 7–20 July), which differ in both date span and instrument string
- [ ] **ATTRIB-06**: Attribution can be completed before the first full reconcile sweep, so the calendar never visibly double-books

## v2 Requirements

Deferred to v2.3. Tracked but not in this roadmap.

### Status Unification

- **STATUS-01**: One status vocabulary replaces the three parallel prefix maps (`_CLASSICAL_STATUS_PREFIX`, `_FAILURE_PREFIX_BY_STATUS`, `_RUN_STATUS_CALENDAR_PREFIX`), which today agree with `calendar_display_extras._TERMINAL_PREFIXES` only by convention
- **STATUS-02**: A general terminal-state classifier replaces v2.2's `status == 'COMPLETED'` check, once per-facility `get_terminal_observing_states()` differences are reconciled

### Adapter Consolidation

- **ADAPT-01**: `load_telescope_runs` creates or updates a `CampaignRun` instead of writing calendar events directly
- **ADAPT-02**: `sync_lco_observation_calendar` creates or updates a `CampaignRun` instead of writing calendar events directly
- **ADAPT-03**: `sync_gemini_observation_calendar` creates or updates a `CampaignRun` instead of writing calendar events directly

### Provenance-Blind Gap Analysis

- **GAPB-01**: `campaign_gap.claimed_dates()` counts every observation on the campaign calendar, not only those with a `CampaignRun`, so classical and queue time is no longer reported as unclaimed

### Unused Allocation

- **UNUSED-01**: Awarded nights that were never scheduled or observed are visually distinct from realised nights, making unused allocation legible at a glance

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Automatic merging of suspected duplicate associations | The measured real case (pk=1 vs. its 11 LCO events) differs in both date span and instrument string; any confidence-threshold auto-merge would destroy the run-vs-realisation distinction this milestone exists to create. Research found no observatory tool that does cross-source attribution at all |
| Upstreaming the event→run link into `tom_calendar` | `CampaignRun` is a FOMO concept, so upstream would need a generic relation or a new upstream run concept — a larger discussion that would block this milestone on tomtoolkit's release cycle. Revisit as a seed once the FOMO-side model is proven |
| Renaming `related_name='telescope_label_meta'` | Renaming the model class breaks two import sites the compiler catches; renaming the related_name would silently break the calendar template and the view's `prefetch_related()` with no static check. Rename the class only |
| A new dependency for reconciliation or field-diffing | Research explicitly rejected `django-dirtyfields`/`FieldTracker` (the existing `calendar_utils._update_or_unchanged()` already does this auditably), `django-fsm`, Celery, and `rapidfuzz` (stdlib `difflib` proved sufficient in v2.1) |
| `GenericForeignKey` for the event and record links | Both link targets are fixed and known; a generic relation loses JOIN, prefetch, and admin ergonomics for no benefit |
| Making the `run` link required | Every existing companion row predates the link, and `CalendarEvent` is general-purpose — conferences and proposal deadlines are events with no run at all. The link stays nullable |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SPIKE-01 | Phase 26 — Canonical-Record Spike | Pending |
| SPIKE-02 | Phase 26 — Canonical-Record Spike | Pending |
| SPIKE-03 | Phase 26 — Canonical-Record Spike | Pending |
| SPIKE-04 | Phase 26 — Canonical-Record Spike | Pending |
| CANON-01 | Phase 27 — The Canonical Run Record | Pending |
| CANON-02 | Phase 27 — The Canonical Run Record | Pending |
| CANON-03 | Phase 27 — The Canonical Run Record | Pending |
| CANON-04 | Phase 27 — The Canonical Run Record | Pending |
| CANON-05 | Phase 27 — The Canonical Run Record | Pending |
| ATTRIB-01 | Phase 28 — Operator-Assisted Attribution | Pending |
| ATTRIB-02 | Phase 28 — Operator-Assisted Attribution | Pending |
| ATTRIB-03 | Phase 28 — Operator-Assisted Attribution | Pending |
| ATTRIB-04 | Phase 28 — Operator-Assisted Attribution | Pending |
| ATTRIB-05 | Phase 28 — Operator-Assisted Attribution | Pending |
| ATTRIB-06 | Phase 28 — Operator-Assisted Attribution | Pending |
| RECON-01 | Phase 29 — The Reconciler | Pending |
| RECON-02 | Phase 29 — The Reconciler | Pending |
| RECON-03 | Phase 29 — The Reconciler | Pending |
| RECON-04 | Phase 29 — The Reconciler | Pending |
| RECON-05 | Phase 29 — The Reconciler | Pending |
| RECON-06 | Phase 29 — The Reconciler | Pending |
| RECON-07 | Phase 29 — The Reconciler | Pending |
| RECON-08 | Phase 29 — The Reconciler | Pending |
| RECON-09 | Phase 29 — The Reconciler | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓

**Execution order:** 26 → 27 → 28 → 29. Attribution (28) is deliberately scheduled ahead of the reconciler (29): it is the only mechanism in v2.2 that creates run↔`ObservationRecord` links (adapter rewiring is deferred to v2.3), so without it the reconciler's stages 3 and 4 have no real data to act on — and it makes ATTRIB-06 structural rather than a rollout caveat.

---
*Requirements defined: 2026-07-26*
*Last updated: 2026-07-26 after initial definition*
