---
phase: 27-the-canonical-run-record
verified: 2026-07-30T14:20:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 27: The Canonical Run Record Verification Report

**Phase Goal:** Make `CampaignRun` canonical in the schema — it records how it was created, distinguishes a class-wide allocation from an unresolved site, owns the calendar events that show it, and owns the observation records that realise it — with every existing row and all four companion-record consumers surviving the change.
**Verified:** 2026-07-30T14:20:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every `CampaignRun` records which ingest path created it; a non-web run is never left in the review queue; non-staff visibility unchanged for old and new rows | VERIFIED | `models.py:82-102` `Source` TextChoices (WEB/CLASSICAL_FILE/LCO_QUEUE/GEMINI_QUEUE/CSV_IMPORT/LEGACY, default LEGACY). `import_campaign_csv.py:195-200` sets `approval_status=APPROVED` + `source=CSV_IMPORT` directly, bypassing the review queue. `campaign_views.py:207,237-258` web submissions default to `PENDING_REVIEW` and set `source=WEB`. Dev DB: `source` distinct values are `['legacy', 'csv_import']`, no CSV-imported row sits in `PENDING_REVIEW`. `CampaignRunTableView`'s non-staff `exclude(approval_status=PENDING_REVIEW)` (`campaign_views.py:130`) is byte-unchanged from before the phase. |
| 2 | A class-wide telescope allocation is distinguishable from an unresolved site, and the two coexist without colliding | VERIFIED | `models.py:104-130` `TelescopeClass` TextChoices (`2m0`/`1m0`/`0m4`/`SPACE`, blank default); field is separate from `site`/`site_needs_review`. Migration `0011` backfill and `import_campaign_csv.py` both gate `derive_telescope_class()` on `site is None`, via the single shared `calendar_utils.derive_telescope_class()` helper (confirmed only one `def derive_telescope_class` exists, imported at both call sites). Neither existing partial `UniqueConstraint` on `CampaignRun.Meta` was touched (confirmed by reading `models.py:227-265` — both constraints identical field sets to pre-phase). Dev DB: 3 rows carry a non-blank `telescope_class` (SPACE/1m0/2m0), independent of `site`. |
| 3 | Companion rows survive the rename with `is_verified` intact: dashed-border fallback, LCO sync writes labels, admin registers the model, calendar page loads labels via one prefetch | VERIFIED | `models.py:10-49` `CalendarEventMeta` (renamed from `CalendarEventTelescopeLabel`) via hand-authored `RenameModel` migration `0008` (not Delete/Create — confirmed via AST-level review and migration file content). Dev DB: 11 `CalendarEventMeta` rows present post-migration (matches pre-phase row count). `calendar.html:228,244` still reference `event.telescope_label_meta.is_verified == False` for the dashed-border style, byte-unchanged. `sync_lco_observation_calendar.py:369` still calls `CalendarEventMeta.objects.update_or_create(event=event, defaults={'is_verified': ...})`. `admin.py:99-102,112` registers `CalendarEventMetaAdmin`. `views.py:114` still does a single `.prefetch_related('telescope_label_meta')`. |
| 4 | A calendar event can link to its run; an ObservationRecord can link to the run it realises with confirmation metadata; deleting a run never deletes calendar events, companion rows, or observation records | VERIFIED | `models.py:39-46` `CalendarEventMeta.run` is `SET_NULL` FK to `CampaignRun` (deleting a run un-owns the event, doesn't delete it or the companion row). `models.py:272-333` `CampaignRunObservation` links `run` (CASCADE from run's side — deletes the link row, not the `ObservationRecord`) to `observation_record` (CASCADE from the other direction, untouched by run deletion) with `confirmed_by`/`confirmed_at` and named constraint `unique_campaign_run_observation_record`. `test_campaign_run_observation.py` (7 tests, passing) explicitly asserts both cascade directions and the calendar-event/companion-row preservation guarantee. |
| 5 | A staff user can see a run's linked calendar events and observation records, and get from an event back to its run | VERIFIED (with a known display bug on TBD runs, see Anti-Patterns) | `admin.py:8-31,57` `CalendarEventMetaInline`/`CampaignRunObservationInline` registered on `CampaignRunAdmin.inlines`. `admin.py:64-96` `save_formset` stamps `confirmed_by`/`confirmed_at` only on newly created `CampaignRunObservation` rows (pk-is-None gate), never re-stamping an edit. `event_form.html:100-126` renders a "Campaign run" block with a link to the campaign table, gated on `run.is_publicly_visible`, for both staff and non-staff. `test_calendar_template.py`'s `EventModalCampaignRunLinkTest` and `test_admin.py`'s `CampaignRunAdminInlinesTests` both pass. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/models.py` | `CalendarEventMeta`, `CampaignRun.Source`/`TelescopeClass`/`source`/`telescope_class`/`is_publicly_visible`, `CampaignRunObservation` | VERIFIED | All present, read directly (lines 10-333) |
| `solsys_code/calendar_utils.py` | `derive_telescope_class()`, `NO_OBSCODE_SPACE_OBSERVATORIES`, 5 de-underscored helpers | VERIFIED | `def derive_telescope_class` at line 146; `NO_OBSCODE_SPACE_OBSERVATORIES` at line 118; public helpers confirmed via grep (`aperture_class_from_telescope_code`, `derive_telescope`, `resolve_placement_block`, `extract_instrument`, `coarse_telescope_label`) |
| `solsys_code/admin.py` | Two inlines, `save_formset`, list_display/list_filter additions | VERIFIED | Read directly, matches plan and summary claims |
| `solsys_code/management/commands/repair_stale_campaign_run_sites.py` | One-time repair command, `--dry-run`, `create_placeholder=False` | VERIFIED | Test suite (7 tests) passes; dev-DB before/after table in 27-02-SUMMARY.md is internally consistent with D-16's predicted outcome |
| `solsys_code/migrations/0008,0009,0010,0011` | RenameModel, AddField, CreateModel+AddField, RunPython backfill | VERIFIED | All four read directly; `0008` uses `RenameModel` not Delete/Create; `0011`'s backfill uses the shared `derive_telescope_class` helper, gated on `site__isnull=True` |
| `src/templates/tom_calendar/partials/event_form.html` | Event→run link gated on `is_publicly_visible` | VERIFIED (with WR-04 caveat) | Read directly; renders unconditionally `({{ run.window_start }}–{{ run.window_end }})` which prints literal "None–None" for a TBD, publicly-visible run — see Anti-Patterns |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | Regenerated with source/telescope_class in real output | VERIFIED | `telescope_class` appears 10 times in the notebook |
| `docs/runbooks/telescope_runs_calendar.rst` | Documents importer source behaviour + repair command | VERIFIED | `repair_stale_campaign_run_sites` appears 5 times |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `sync_lco_observation_calendar.py` | `calendar_utils.py` | de-underscored helper imports | WIRED | Confirmed import block + call sites |
| `calendar_utils.py` (`derive_telescope_class`) | `campaign_utils.py` | function-local `HORIZONS_OBSERVER_TO_OBSCODE` import | WIRED | Confirmed function-local, not module-level |
| `views.py` | `models.py` | `prefetch_related('telescope_label_meta')` | WIRED | Unedited, confirmed present |
| `calendar.html` | `models.py` | `event.telescope_label_meta.is_verified` | WIRED | Unedited, confirmed present (2 sites) |
| `migrations/0011` | `calendar_utils.py` | function-local `derive_telescope_class` import | WIRED | Confirmed in migration body |
| `import_campaign_csv.py` | `calendar_utils.py` | `derive_telescope_class(site_raw=..., telescope_instrument=...)` | WIRED | Confirmed; gated on `site is None` |
| `admin.py` (`save_formset`) | `models.py` (`CampaignRunObservation`) | `confirmed_by = request.user` | WIRED | Confirmed, pk-is-None gated |
| `event_form.html` | `models.py` | `event.telescope_label_meta.run.is_publicly_visible` | WIRED | Confirmed, but downstream field rendering has the WR-04 gap for TBD runs |
| `campaign_views.py` | `models.py` | `CampaignRun.objects.create(..., source=Source.WEB)` | WIRED | Confirmed at line 257 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `CampaignRunAdmin` list_display/list_filter | `source`, `telescope_class` | `CampaignRun.objects` (admin ORM query, no custom queryset override) | Yes — dev DB confirmed non-trivial distinct values (`legacy`/`csv_import`, `SPACE`/`1m0`/`2m0`) | FLOWING |
| `event_form.html` run block | `run` (from `event.telescope_label_meta.run`) | Live FK traversal on `CalendarEventMeta` | Dev DB currently has 0 `CalendarEventMeta` rows with `run` set (none of Phase 27's own migrations populate this link — it's a future-phase write path per D-06), so the block is currently dormant on the live DB, by design (Phase 28 is the write path) | FLOWING (mechanism proven by test fixtures; live DB has no data yet, which is expected — not a gap) |
| `CampaignRunAdmin` inlines | `run.calendar_event_metas` / `run.observation_links` | Reverse FK managers | Yes — `test_admin.py`'s `CampaignRunAdminInlinesTests` proves real create/edit/stamp behavior against a live formset POST | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Migrations apply cleanly, no pending model changes | `python manage.py makemigrations --check --dry-run` | "No changes detected" | PASS |
| Dev DB shows real `source`/`telescope_class` values after migration | `manage.py shell` query | `source` distinct `['legacy','csv_import']`; `telescope_class` distinct `['SPACE','1m0','2m0']`; 11 `CalendarEventMeta` rows | PASS |
| Targeted phase-27 test modules pass | `manage.py test solsys_code.tests.test_canonical_record_migration solsys_code.tests.test_campaign_run_observation solsys_code.tests.test_repair_stale_campaign_run_sites solsys_code.tests.test_admin solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_utils solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_submission solsys_code.solsys_code_observatory.tests.test_timezone_backfill_migration` | 157/157 pass | PASS |
| `import_campaign_csv` tests pass | `manage.py test solsys_code.tests.test_import_campaign_csv` | 57/57 pass | PASS |
| Full Django suite (already run by orchestrator) | `manage.py test` | 682/682 pass (excluding pre-existing unrelated segfault in `test_views.TestEphemeris`) | PASS (per already_established) |

### Probe Execution

Not applicable — this phase has no `scripts/*/tests/probe-*.sh` convention; verification relies on the Django test suite and direct dev-DB inspection instead.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| CANON-01 | 27-04, 27-05, 27-06 | `source` field, WEB-only approval gating | SATISFIED | `models.py:82-102,195-200`; `campaign_views.py:257`; `import_campaign_csv.py:195-200` |
| CANON-02 | 27-01, 27-02, 27-04, 27-06 | `telescope_class` field, distinguishable from unresolved site | SATISFIED | `models.py:104-130,204-210`; `calendar_utils.py:146`; migration `0011`; `import_campaign_csv.py` |
| CANON-03 | 27-03 | `CalendarEventMeta` rename + `run` link, 4 integration points preserved | SATISFIED | `models.py:10-49`; migrations `0008`/`0009`; `admin.py`, `sync_lco_observation_calendar.py`, `views.py`, `calendar.html` all confirmed unbroken |
| CANON-04 | 27-04 | `CampaignRunObservation` link model with confirmation metadata | SATISFIED | `models.py:272-333`; `test_campaign_run_observation.py` (7 tests) |
| CANON-05 | 27-05 | Staff sees/edits run's linked events and observations; event links back to run | SATISFIED (with WR-04 display gap noted) | `admin.py:8-31,57,64-96`; `event_form.html:100-126` |

No orphaned requirements — REQUIREMENTS.md's traceability table (lines 100-104) maps all five CANON IDs to Phase 27, and all five appear in at least one plan's `requirements:` frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/templates/tom_calendar/partials/event_form.html` | 121 | Unconditional `({{ run.window_start }}&ndash;{{ run.window_end }})` rendering | WARNING (carried from 27-REVIEW.md WR-04) | A TBD, publicly-visible run linked to an event via the admin's new `CalendarEventMetaInline` renders the literal text "(None–None)" in the public event modal. Does not block truth #5 (the link itself works and is tested for the common resolved-window case), but is a real, demonstrated display bug in new phase-27 surface area. Not covered by `EventModalCampaignRunLinkTest`'s fixtures (all use resolved windows). |
| `solsys_code/management/commands/import_campaign_csv.py` | 177-207 | Every re-import recomputes `site`/`site_raw`/`site_needs_review`/`telescope_class` from the CSV cell unconditionally | WARNING (carried from 27-REVIEW.md WR-01) | A CSV re-import over a campaign can silently revert a site just fixed by `repair_stale_campaign_run_sites` (confirmed real for the dev-DB Swift row, D-16b). Undocumented in the runbook's existing "re-import gotcha" note. Does not violate any of the 5 goal-backward truths directly (both `repair_stale_campaign_run_sites` and `import_campaign_csv` individually work as specified), but is an operational data-integrity risk phase 27 itself created by introducing both commands together. |
| `solsys_code/management/commands/repair_stale_campaign_run_sites.py` | 193-199 | Never clears `telescope_class` when it resolves `site` | CORRECTED (Phase 27.1 — was WARNING, carried from 27-REVIEW.md WR-02) | This report's original Impact cell was wrong: `solsys_code/models.py:207-219` documents the *opposite* invariant — `telescope_class` is never cleared by any writer once set, because a class-wide allocation and a resolved site are not mutually exclusive. Code-review finding CR-01, which proposed clearing `telescope_class` on site resolution, was REJECTED by the user (`27-REVIEW-FIX.md` lines 251-281). The behavior this row originally flagged as a defect is therefore correct as shipped; the stale text was the report's, not the code's. Investigated and withdrawn in Phase 27.1 (plan 27.1-04). |
| `solsys_code/admin.py` | 62 | `source` freely editable in admin (not in `readonly_fields`) | WARNING (carried from 27-REVIEW.md WR-03) | Documented as a deliberate decision (D-19) in a comment, but the comment doesn't address the consequence: any staff user can silently overwrite `source` on an already-`APPROVED` `WEB` row, erasing the CANON-01 derivation signal. Acknowledged, not a blocker for the goal as stated (CANON-01 only requires that `source` records provenance, not that it's tamper-proof). |

No `TBD`/`FIXME`/`XXX` markers found in phase-27-modified files (confirmed via the code review's deep scan, cross-checked with a targeted grep of `key-files` from each SUMMARY).

### Human Verification Required

None. All five observable truths are verifiable programmatically against the codebase and the dev DB, and were verified directly rather than deferred.

### Gaps Summary

No blocking gaps. Phase 27's five goal-backward truths are all verified against the actual codebase (not just SUMMARY claims): the schema changes exist, are migrated, are wired into every consumer the phase's own contract names, and the live dev DB reflects the expected post-migration state (11 companion rows preserved, `source`/`telescope_class` populated with real derived values, zero data loss).

Four WARNING-level anti-patterns carry over from the code review (27-REVIEW.md WR-01 through WR-04) — a TBD-run "(None–None)" display bug in the new calendar-modal run link, a CSV-reimport-can-revert-a-repair operational risk, and an editable `source` field in the admin. None of these prevent any of the five roadmap Success Criteria from being true today; they are genuine, demonstrated risks worth fixing in a follow-up but do not block phase completion. They are surfaced here for visibility, consistent with the code review's own WARNING (not CRITICAL) classification. The fourth item this paragraph previously listed — a `repair_stale_campaign_run_sites` invariant gap around `telescope_class` (WR-02) — was investigated in Phase 27.1 and withdrawn: `solsys_code/models.py:207-219` documents the opposite invariant, and the user rejected code-review finding CR-01, which proposed the behavior this report originally wanted (`27-REVIEW-FIX.md` lines 251-281). See the corrected Anti-Patterns table row above.

_Correction (Phase 27.1, 2026-07-31): the WR-02 anti-pattern row and this paragraph were amended to withdraw the stale `telescope_class` invariant claim — see plan `27.1-04` for the investigation and citation._

---

_Verified: 2026-07-30T14:20:00Z_
_Verifier: Claude (gsd-verifier)_
