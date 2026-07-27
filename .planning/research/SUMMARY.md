# Research Summary: FOMO v2.2 "One Canonical Run Record"

**Project:** FOMO Telescope Runs Calendar — v2.2 Milestone  
**Domain:** Django/TOM Toolkit; idempotent calendar reconciler retrofit onto a live system with multi-source ingest and operator data  
**Researched:** 2026-07-26  
**Confidence:** HIGH (direct source inspection; MEDIUM on Django edge-case behavior from web search)

---

## Executive Summary

FOMO v2.2 consolidates observing runs into a single canonical `CampaignRun` record from which calendar events and observation-record linkages are derived — moving calendar projection from a side effect of a staff click to an idempotent reconciler function. The architecture is pure Django ORM with no new dependencies; the hard work is in (1) a careful migration strategy for the companion record's rename and FK addition that preserves the four existing integration points (admin, management commands, template, view prefetch), (2) settling natural-key semantics and attribution strategy in a spike before building the reconciler, and (3) building the reconciler's four-stage window pipeline with strict ownership scoping so it never silently deletes or mutates unowned calendar events.

**Recommended approach:** Execute the spike first to settle `source`-field identity semantics and the per-adapter mapping strategy. Then sequence migrations (rename + companion-record generalization, then `source`/`telescope_class`, then `ObservationRecord` M2M) as three separate migration files. Build the reconciler in a new `campaign_reconciler.py` logic-layer module (peer to existing `campaign_gap.py`/`campaign_utils.py`, not a views-module helper) to resolve an existing anti-pattern where `backfill_range_calendar_events` imports a private `_project_calendar_event` function from views. Wire the reconciler into both `CampaignRunDecisionView` (per-run on staff actions) and a new `reconcile_campaign_runs` management command (batch sweep), and build the attribution surface last.

**Key risks:** (1) The companion-record rename breaks template/prefetch/admin/command references silently if not re-verified after migration — this is the canonical test case for the spike's deliverable. (2) Stage transitions between the four-window-pipeline stages introduce churn if the key scheme drifts — must design one stable key scheme across all stages and prove idempotency with two consecutive runs. (3) Attribution heuristics will fail silently on the measured real-world case (FTS/MuSCAT4 pk=1 vs. 11 LCO queue events with date-off-by-one and instrument-string mismatch) unless built against that fixture from day one. (4) The reconciler's ownership scoping must be airtight — it must never touch an unattributed hand-created event or a pre-reconciler sync-command event in the same date window, proven by an explicit test fixture.

---

## Key Findings

### Recommended Stack

**No new runtime or development dependencies are warranted.** Every piece of v2.2 — companion-record generalization, `ObservationRecord` linkage, the reconciler, its idempotency tests — is built from Django's own ORM (`ForeignKey`, `ManyToManyField` with custom `through`, migrations `RenameModel`/`AddField`) and the ecosystem already installed for this project (Django 5.2.13 via `tomtoolkit==3.0.0a9`). This is a **milestone addendum**, not a full-project stack review; prior milestones' technologies (astropy, sorcha, ASSIST, SPICE) remain unchanged and in force.

**Core technologies (all pre-installed, no action needed):**
- **Django 5.2.13** — ORM relations, migrations, test framework (`ForeignKey`, `ManyToManyField(through=...)`, `RenameModel`, `CaptureQueriesContext`)
- **`tomtoolkit==3.0.0a9`** (bundles `tom_calendar`, `tom_observations`) — source models (`CalendarEvent`, `ObservationRecord`) are plain Django models with no custom managers or hooks; sidecar/through-model approach is the only attachment point
- **Django test utilities** (`CaptureQueriesContext`) — proves reconciler idempotency (zero writes on second pass), already precedented in this codebase (`test_calendar_template.py:272-289`)

**What NOT to use (explicitly rejected by research):**
- `django-dirtyfields`, `FieldTracker` — the explicit field-diff logic in `calendar_utils._update_or_unchanged()` already solves this, auditably
- `django-fsm`, `django-tasks`/Celery — reconciler is a synchronous management command matching existing sync-command conventions
- `rapidfuzz` — stdlib `difflib` proved sufficient in v2.1 (Phase 18/21 precedent); no match-quality need here
- `GenericForeignKey` — both link targets (`CalendarEvent`, `ObservationRecord`) are fixed and known; loses JOIN, prefetch, admin ergonomics
- Zero-downtime `db_table` pinning — not this project's situation (SQLite dev DB, `DEBUG=True`, no concurrent-deploy constraint)

### Expected Features

**Must have (table stakes for canonical-run model):**
- Single durable `CampaignRun` record per awarded allocation, separate from its executions
- Calendar visibility for every awarded run without a bespoke backfill command per gap (the reconciler solves this)
- Progressive window resolution (site → class → scheduled → completed) matching real facility behavior
- Idempotent, non-destructive reconciliation safe to re-run
- Operator-assisted attribution (suggested, not automatic, links) — the measured real case is FTS pk=1 vs. 11 LCO events with date/instrument mismatch; must be surfaced with confidence scores and per-candidate evidence

**Should have (competitive advantage):**
- `source` provenance field (web submission / classical file / LCO queue / Gemini queue / CSV import) with approval gating per source
- `telescope_class` field to distinguish "legitimately class-wide" from "site failed to resolve" (today both are `site=None`, structurally ambiguous)

**Defer to v2.3 (explicitly out of scope):**
- Unified status vocabulary across all four ingest sources (LCO, Gemini, classical, campaign CSV)
- Adapter rewrite to write `CampaignRun` natively instead of events
- Provenance-blind coverage-gap analysis (count LCO/Gemini/classical events, not just runs)

### Architecture Approach

The v2.2 reconciler extends an established pattern already in this codebase: `campaign_gap.py` and `campaign_utils.py` are pure-logic modules with zero Django-request concerns, imported by views — not the other way around. The reconciler is the third member of this family, living in a new `solsys_code/campaign_reconciler.py` module (not a views-module helper). This resolves an existing anti-pattern where `backfill_range_calendar_events` imports a private `_project_calendar_event()` function from the views layer — both the reconciler and any future management command should import shared logic from the same logic-layer home.

**Major components:**

1. **`campaign_reconciler.py` (new)** — Four-stage window pipeline: `_stage_for()` dispatcher (pure `if`/`elif`, no query), stage-specific window functions (`_site_window()`, `_class_wide_window()`, `_record_window()`), bulk query strategy (3 queries total to fetch runs + prefetch records + bulk-load existing companion rows, independent of run count), and write path (delegates unchanged to existing `insert_or_create_calendar_event()` with no new create-or-update code). Never imports `solsys_code.views` or `solsys_code.ephem_utils` (same SPICE-avoidance contract `campaign_gap.py` already states).

2. **Generalized companion record** — `CalendarEventTelescopeLabel` gains a nullable `run` FK to `CampaignRun` (`on_delete=SET_NULL`), giving runs a one-to-many relation to events via the existing OneToOne sidecar. The event↔companion relation itself stays OneToOne (no cardinality change there); many-to-one comes from many companion rows pointing at the same run. **Critical: The related_name `telescope_label_meta` stays unchanged** — renaming it breaks template/prefetch strings with no static check. The model class itself is renamed (closing the pending naming todo), but the `related_name` doesn't need to change.

3. **Attribution surface (new)** — Staff-facing "suggested associations" queue modeled on existing `ApprovalQueueView` pattern. Never auto-confirms; per-candidate evidence (matched telescope/date/campaign, visible date-overlap/string-similarity signals) required. Uses a through-model carrying `is_confirmed` flag (not a plain M2M), reusing the one-bit-flag idiom already established by `CalendarEventTelescopeLabel.is_verified`.

4. **Wire into callers** — `CampaignRunDecisionView.post()` (approve/resolve_site/mark_cancelled/mark_weather_failure branches) calls `reconcile_run(run)`. New `reconcile_campaign_runs` management command wraps `reconcile_runs()` with `--dry-run` flag.

### Critical Pitfalls (Top 5)

1. **Companion-record rename breaks silent integration points** — Four integration points reference the old name or its `related_name`: `admin.py` import, `sync_lco_observation_calendar.py` import, `views.py` prefetch string, `calendar.html` template lookups. Must re-verify all four in the same commit as the rename migration. Prevention: grep-verify the checklist, prefer keeping `related_name` unchanged (safest).

2. **`run` FK added as required/CASCADE or NULL** — Existing `CalendarEventTelescopeLabel` rows (all LCO/SOAR/Gemini/classical-sync rows) have no `CampaignRun` at all. Must be `null=True, blank=True, on_delete=SET_NULL` so migration is a no-op data-wise; `CASCADE` destroys operator verification history. Prevention: migration reviewed for `null=True`/`SET_NULL`; no `RunPython` backfill.

3. **`source` field collides with existing unique constraints** — Adding `source` to constraint keys risks false collisions on future adapter writes. Keep `source` purely descriptive; let attribution (not the constraint) connect same-physical-run rows from different sources. Prevention: spike tests pk=1/11-LCO-events (both coexist without `IntegrityError`).

4. **Attribution auto-links on loose heuristics** — Date+telescope overlap alone misses the measured case (pk=1 has one-day date discrepancy, instrument strings don't match). Must design against that fixture from day one. Hard filter on target/campaign. Prevention: attribution test includes pk=1 pair; dry run surfaces it with visible evidence.

5. **Reconciler not actually idempotent across stage transitions** — Must design one canonical key scheme per run stable across all stages. Route every write through `insert_or_create_calendar_event()`. Prevention: run twice; assert zero `CalendarEvent.objects.count()` change and zero `modified` churn.

---

## Implications for Roadmap

Based on research, suggested phase structure (6 phases):

### Phase 26: Spike — Natural Keys & Attribution Strategy

**Rationale:** This must come first; every other phase depends on decisions here.  
**Delivers:** Natural-key semantics under `source`, per-adapter identity mapping, migration/backfill strategy, attribution heuristic shape.  
**Must-have:** Reproduce pk=1/11-LCO-events scenario as executable test; document source enum and constraints; settle companion-record rename decision with four integration-point checklist.

### Phase 27: Schema Changes — `source`, `telescope_class`, Companion Generalization

**Rationale:** These three schema additions are independent and execute spike decisions directly.  
**Delivers:** Three separate migrations (not combined); `source` + `telescope_class` on `CampaignRun`; `observation_records` M2M with through-model; companion rename/FK addition.  
**Avoids Pitfalls:** 1 (re-verified integration points), 2 (`null=True`/`SET_NULL`), 3 (`source` out of constraints), 11 (non-web adapters' `approval_status` set explicitly).

### Phase 28: Reconciler Core — Four-Stage Pipeline

**Rationale:** Blocks view wiring and attribution; must not be blocked by attribution.  
**Delivers:** `campaign_reconciler.py` with stage dispatcher/window functions/bulk query strategy; `reconcile_campaign_runs.py` command; idempotency test (two runs, zero writes).  
**Must-have:** Never imports views/ephem_utils; fixture with unrelated event proves it survives untouched; pk=1/11-LCO proves reconciler is blind to unlinked events; non-UTC-friendly sites in date-boundary tests.  
**Avoids Pitfalls:** 5–7 (stable key, no-churn reuse, ownership scoping), 9–10 (bounds, batch isolation), 13 (timezone semantics).

### Phase 29: Wire Reconciler into Views & Commands

**Rationale:** Depends on Phase 28 existing; uses Phase 27 schema.  
**Delivers:** `CampaignRunDecisionView.post()` calls `reconcile_run(run)`; deletes old `_project_calendar_event()` and friends.  
**Must-have:** All existing tests pass; 19 invisible 3I/ATLAS runs now have calendar events.

### Phase 30: Attribution Surface — Operator-Assisted Linking

**Rationale:** Depends on Phase 27 schema + Phase 28 reconciler; needed before first production reconcile (see operational note below).  
**Delivers:** Staff "Suggested Associations" queue with per-candidate evidence; confirm/reject actions; unlink affordance.  
**Must-have:** pk=1 fixture surfaces with visible evidence; target-hard-filter prevents cross-target matches; confirmation is per-candidate and logged; reversible.  
**Avoids Pitfalls:** 4 (fixture-driven attribution), 5 (undo implemented).

### Phase 31: Retire Old Code

**Rationale:** Only after Phase 29 proves all reconciler call sites are in place.  
**Delivers:** Delete `_project_calendar_event()`, related helpers, `backfill_range_calendar_events.py`.  
**Must-have:** All tests pass after deletion.

### Operational Note

**Attribution must run before the first full production reconcile.** An unattributed first sweep will create fresh `CAMPAIGN:{pk}:{date}` events for nights that already have adapter-sourced events, producing visible double-booking until attribution links them. Run the attribution surface before or alongside the first `reconcile_campaign_runs` sweep.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | **HIGH** | Direct reads of installed tomtoolkit source; Django ORM operations are long-stable (predate Django 4). |
| **Features** | **MEDIUM** | Cross-checked against LCO/Gemini/ESO/ALMA/JWST real systems. Attribution heuristics and stage-2 semantics drawn from domain research; spike will validate. |
| **Architecture** | **HIGH** | Grounded in direct inspection of existing `solsys_code/` modules. Pattern (pure-logic modules, not views helpers) already established. |
| **Pitfalls** | **HIGH** | Most from measured dev-DB hazards (19 invisible runs, pk=1/11-LCO collision, FTN timezone). Tier-3 pitfalls confirmed by code inspection. |

**Overall:** **HIGH** — Spike is the only phase requiring new design; every subsequent phase executes established patterns.

### Gaps to Address

1. **Stage-2 class-wide fan-out:** Does stage 2 create one event per candidate site or one class-wide event? Spike deliverable; impacts event count and presentation.

2. **Per-run reconciliation-failed surface:** If reconciler runs as batch sweep (not just per-run), a retry queue is needed (like existing `site_needs_review`). Spike decision; Phase 28 implements accordingly.

3. **Date-boundary correctness:** Must test stage transitions against non-UTC-friendly real sites (Las Campanas, Siding Spring), not just UTC-convenient mocks. Phase 28 must include this.

---

## Sources

**PRIMARY (HIGH confidence):**
- `.planning/research/STACK.md` — Direct inspection of installed package source; Django migration edge cases from community tickets
- `.planning/research/ARCHITECTURE.md` — Direct reads of `solsys_code/` modules and management commands
- `.planning/research/PITFALLS.md` — Measured dev-DB hazards; code-inspection confirmations; established precedents
- `.planning/PROJECT.md` — Current Milestone v2.2 section, concrete defects (19 invisible runs, pk=1/11-LCO collision, FTN timezone gap)

**SECONDARY (MEDIUM confidence):**
- `.planning/research/FEATURES.md` — Facility tool research (LCO, Gemini, ESO, ALMA, JWST); OpenRefine reconciliation pattern

---

*Research completed: 2026-07-26*  
*Confidence: HIGH*  
*Ready for roadmap: YES*
