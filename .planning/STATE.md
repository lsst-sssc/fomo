---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Calendar Visual Clarity
current_phase: 08
current_phase_name: telescope-label-verification-sidecar
status: verifying
stopped_at: Completed 08-02-PLAN.md (Phase 8 complete, ready for verification)
last_updated: "2026-06-25T06:34:13.212Z"
last_activity: 2026-06-25
last_activity_desc: Phase 08 execution started
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-24 after v1.3 milestone close, "Current Milestone" section added for v1.4)

**Core value:** Make `CalendarEvent` color and status convey real meaning (proposal identity, queued/placed/failed state) and add a dedicated field for fallback-resolved telescope labels.
**Current focus:** Phase 08 — telescope-label-verification-sidecar

## Current Position

Phase: 08 (telescope-label-verification-sidecar) — EXECUTING
Plan: 2 of 2
Status: Phase complete — ready for verification
Last activity: 2026-06-25 — Phase 08 execution started

## Performance Metrics

**Velocity:**

- Total plans completed: 9 (v1.0-v1.2) + 5 (v1.3, incl. 07.1) = 14
- Average duration: ~35 min (Phase 2) + 7-8 min/plan (Phase 3); Phase 7 plans ~50 min each
- Total execution time: ~3-4 sessions (v1.0-v1.2) + several sessions (v1.3)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 1 | ~35 min | ~35 min |
| 03 | 2 | - | - |
| 04 | 1 | - | - |
| 05 | 1 | - | - |
| 06 | 1 (6 min, 2 tasks, 2 files) | - | - |
| 07 P01 | 1 (50 min, 3 tasks, 3 files) | - | - |
| 07 P02 | 1 (~50 min, 3 tasks, 3 files) | - | - |
| 07.1 P01 | 1 (25 min, 3 tasks, 3 files) | - | - |
| 08 | TBD | - | - |
| 09 | TBD | - | - |
| Phase 08 P01 | 24min | 3 tasks | 6 files |
| Phase 08 P02 | 11min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

All v1.0-v1.3 decisions logged in PROJECT.md's Key Decisions table (backfilled at each milestone close). v1.4-specific decisions will accumulate here during Phase 8/9 execution and be backfilled to PROJECT.md at v1.4 close.

**Roadmap-time decisions for v1.4 (this roadmapping pass):**

- Phase order: Phase 8 (DISPLAY-02/03/01 sidecar) before Phase 9 (DISPLAY-04/05/06/07 color/status) — research-recommended ordering, isolates the riskier first-ever-migration/cross-app-OneToOneField work from the purely-additive template work in Phase 9.
- DISPLAY-06's open design decision (status visual treatment mechanism: border/opacity/stripe) is resolved via `/gsd:sketch` during Phase 9 planning, not pre-decided by roadmap — kept inside Phase 9 rather than split into its own phase, since "coarse" granularity and its tight coupling to DISPLAY-05's `[QUEUED]` fix and DISPLAY-07's legend (same template, same render pass) don't warrant a third phase.
- N+1 mitigation (batching template tag for the sidecar's reverse O2O reverse accessor) is a Phase 8 planning-time scope call, not decided here — research flags both "accept as-is" and "bulk-prefetch tag" as legitimate options for current calendar-event volume.
- [Phase 08]: Migration generated with explicit --name calendareventtelescopelabel flag to match the plan's must_haves artifact filename; content stays deterministic/unedited
- [Phase 08]: Test expectation for the dashed-border marker count expressed as day-cell occurrences (not event count) because tom_calendar's render_calendar() buckets a multi-day all-day event once per day cell it spans

### Pending Todos

1. Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence) — `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md` — **being addressed by v1.4 Phase 9**
2. Extract site/telescope mapping and instrument extraction into own module (revisit after Phase 7 ships) — `.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`

### Blockers/Concerns

None open. v1.3's Phase 7 research gaps were all resolved and verified live (see PROJECT.md Key Decisions). v1.4 Phase 8/9 each carry one explicit open scope/design decision to make during planning (N+1 mitigation scope for Phase 8; status-treatment mechanism via `/gsd:sketch` for Phase 9) — both already flagged in ROADMAP.md and research docs, not blockers.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260619-f7u | Phase 5 notebook gap: update sync_lco_observation_calendar_demo.ipynb for multi-proposal/multi-facility (SELECT-02/03/04/05); document Stage-vs-Phase numbering mapping in PROJECT.md | 2026-06-19 | 5bef02d | [260619-f7u-phase-5-notebook-gap-update-sync-lco-obs](./quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/) |
| 260619-jpr | Fix sync_lco_observation_calendar SOAR site mapping bug: add SOAR site code 'sor' to SITE_TELESCOPE_MAP, fix SOAR test fixtures and demo notebook fixtures that incorrectly reused LCO site code 'coj' for SOAR records | 2026-06-19 | caf97bd | [260619-jpr-fix-sync-lco-observation-calendar-soar-s](./quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/) |
| 260619-ml8 | Fix pre-commit jupyter-nb-clear-output exclude path mismatch (docs/pre_executed vs docs/notebooks/pre_executed) and remove redundant +00:00 UTC offset from sync_lco_observation_calendar's CalendarEvent description Window (UTC) string | 2026-06-19 | 6b1d37a | [260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e](./quick/260619-ml8-fix-pre-commit-jupyter-nb-clear-output-e/) |
| 260620-v9x | Phase 6 notebook gap: update sync_lco_observation_calendar_demo.ipynb to demonstrate EXTRACT-01/EXTRACT-02 (SOAR multi-config, MUSCAT per-channel, malformed/extraction_failed) | 2026-06-21 | d41cdc7 | [260620-v9x-update-docs-notebooks-pre-executed-sync-](./quick/260620-v9x-update-docs-notebooks-pre-executed-sync-/) |
| 260621-foo | Document demo-notebook-companion convention in CLAUDE.md (planner/plan-checker/executor/verifier responsibilities) to prevent recurrence of the Phase 5/6 notebook-scoping gap | 2026-06-21 | fe378de | [260621-foo-add-a-claude-md-convention-requiring-dem](./quick/260621-foo-add-a-claude-md-convention-requiring-dem/) |
| 260623-ocs | Fix T-07-03 security spec gap in sync_lco_observation_calendar: read resolved API block via .get() instead of bracket indexing in _build_event_fields, None-guard _aperture_class_from_telescope_code, add regression test for a block missing 'site'/'telescope' falling back (not skipped) | 2026-06-24 | 2fa0300 | [260623-ocs-fix-t-07-03-security-spec-gap-in-sync-lc](./quick/260623-ocs-fix-t-07-03-security-spec-gap-in-sync-lc/) |
| 260623-su3 | Fix SITE_TELESCOPE_MAP completeness gap from Phase 7 UAT Test 1: add missing ('coj','1m0'), ('coj','0m4'), ('ogg','0m4') entries confirmed against https://lco.global/observatory/sites/mpccodes/, add regression test for the 3 newly-mapped pairs | 2026-06-24 | 5583400 | [260623-su3-fix-site-telescope-map-completeness-gap-](./quick/260623-su3-fix-site-telescope-map-completeness-gap-/) |

### Key Technical Notes (carried from Phase 4-7 / v1.2-v1.3, still relevant to v1.4)

- `parameters` on `ObservationRecord` is a `TextField` containing JSON (not a JSONField) — filtering requires fetching then parsing in Python.
- `CalendarEvent` upsert keyed on `url` (`LCOFacility().get_observation_url(observation_id)`, confirmed `/requests/<id>` no trailing slash).
- No-churn idempotency: only call `.save()` when fields actually changed — Phase 8's new sidecar write must not be folded into this diff/`fields` comparison (keep it a separate `update_or_create` statement, per ARCHITECTURE.md Pitfall 4).
- `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before DB save.
- DB-dependent tests go in `solsys_code/tests/`, run with `./manage.py test solsys_code`.
- `sync_lco_observation_calendar.py` already computes `telescope_api_failed` (in `_build_event_fields`) and pops it from `fields` before the existing `CalendarEvent.objects.get_or_create()` call — Phase 8's sidecar write goes immediately after that call site, in the same loop iteration, using the already-in-scope `telescope_api_failed` value.
- `load_telescope_runs.py` needs zero code change for Phase 8 — confirmed by direct read, no API call, no fallback concept exists in that command.
- `render_calendar()` (installed `tom_calendar` view) has no `extra_context`/`get_queryset()` hook — any Phase 8/9 read-side fix (N+1 mitigation, color/status logic) must live in the `calendar.html` template-override layer, not the view.
- `calendar.html`'s existing `[QUEUED]` grey override (lines ~158-161) currently discards `event.color` entirely — Phase 9 must fix this in the same task that adds proposal coloring, or the new color signal is invisible for queued events.
- Never use Python's built-in `hash()` for proposal->color mapping (per-process salted, non-deterministic across restarts) — use `hashlib.sha256` per STACK.md.

### Roadmap Evolution

- Phase 07.1 inserted after Phase 7 (v1.3): Close gap: TELESCOPE-03/04/SYNC-06 — SOAR fallback label is facility-unaware (URGENT)
- v1.4 roadmap created 2026-06-24: Phase 8 (Telescope Label Verification Sidecar — DISPLAY-01/02/03) and Phase 9 (Proposal Color & Status Visual Treatment — DISPLAY-04/05/06/07), continuing phase numbering from v1.3's Phase 7/07.1.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| todo | 2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md | being addressed by v1.4 Phase 9 | v1.2 close |
| todo | 2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md | pending (deliberately deferred future work) | v1.3 close |
| requirement | DISPLAY-08 (WCAG contrast-ratio-aware text color switching) | deferred to v2 | v1.4 requirements definition |
| requirement | DISPLAY-09 (batching template tag for sidecar N+1) | deferred to v2 (also a Phase 8 in-phase scope call — see Decisions) | v1.4 requirements definition |

## Session Continuity

Last session: 2026-06-25T06:34:13.188Z
Stopped at: Completed 08-02-PLAN.md (Phase 8 complete, ready for verification)
Resume file: None

## Operator Next Steps

- Review and approve the v1.4 roadmap (Phases 8-9).
- Start Phase 8 with `/gsd-plan-phase 8`.
