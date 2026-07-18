---
phase: 20-range-tbd-import-asset-aware-coverage-gap
plan: 02
subsystem: database
tags: [django, migrations, django-tables2, xss-mitigation]

# Dependency graph
requires:
  - phase: 19-window-schema-migration
    provides: "CampaignRun.window_start/window_end nullable window schema, site_needs_review/site_raw precedent fields, migrations 0001-0005"
provides:
  - "CampaignRun.original_obs_date_raw (CharField) — verbatim Obs. Date sheet text for TBD rows"
  - "CampaignRun.window_needs_review (BooleanField) — window-resolution review flag"
  - "Migration 0006 applied to dev DB"
  - "render_window_start() TBD-badge tooltip surfacing original_obs_date_raw, HTML-escaped"
affects: [20-03-range-tbd-csv-import, 20-04-asset-aware-coverage-gap-demo-notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two new fields added as siblings of site_raw/site_needs_review, same CharField(max_length=255, blank=True, default='') / BooleanField(default=False) shape"
    - "Migration 0006: plain two-AddField migration, no RunPython/backfill — precedent for additive TBD-only fields"
    - "render_window_start() TBD branch: format_html(...) positional-argument tooltip, matching render_site()'s existing escape-by-construction pattern"

key-files:
  created:
    - solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py
  modified:
    - solsys_code/models.py
    - solsys_code/campaign_tables.py
    - solsys_code/tests/test_campaign_models.py
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "original_obs_date_raw is CharField(max_length=255), not TextField — mirrors site_raw exactly per RESEARCH field-type resolution"
  - "Migration 0006 renamed from Django's auto-generated 0006_campaignrun_original_obs_date_raw_and_more.py to the plan's explicit filename for readability/traceability"
  - "Applied migration 0005 (previously pending) together with 0006 to the dev DB, since 0005 had not yet been applied from Plan 19's deferred step"

patterns-established:
  - "TBD-badge tooltip: resolve raw text via Accessor(...).resolve(record, quiet=True) or '', guard for empty string, interpolate via format_html positional arg (never mark_safe/f-string) — reusable for any future dict-vs-model tooltip on this table"

requirements-completed: [IMPORT-01, IMPORT-02]

coverage:
  - id: D1
    description: "CampaignRun gains original_obs_date_raw and window_needs_review fields with site_raw/site_needs_review-style defaults, applied via migration 0006"
    requirement: "IMPORT-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunWindowNeedsReviewFields"
        status: pass
      - kind: other
        ref: "python manage.py makemigrations solsys_code --check --dry-run (exit 0, no pending changes)"
        status: pass
      - kind: other
        ref: "python manage.py migrate solsys_code (0006 applied to src/fomo_db.sqlite3)"
        status: pass
    human_judgment: false
  - id: D2
    description: "TBD badge in the campaign table shows original_obs_date_raw as an HTML-escaped hover tooltip (D-08), blank when the field is empty"
    requirement: "IMPORT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_tbd_row_with_raw_text_renders_tooltip"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_tbd_row_with_blank_raw_text_renders_no_title"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_tbd_row_with_markup_raw_text_is_escaped"
        status: pass
    human_judgment: false

duration: 10min
completed: 2026-07-10
status: complete
---

# Phase 20 Plan 2: Window-Review Fields & TBD Tooltip Summary

**CampaignRun gains original_obs_date_raw/window_needs_review fields (migration 0006, applied to the dev DB) and the campaign table's TBD badge now shows the raw sheet text as an HTML-escaped hover tooltip.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-07-10T19:07:00Z
- **Completed:** 2026-07-10T19:17:00Z
- **Tasks:** 2
- **Files modified:** 5 (1 created: migration; 4 modified: models.py, campaign_tables.py, 2 test files)

## Accomplishments
- `CampaignRun.original_obs_date_raw` (CharField, max_length=255, blank, default `''`) and `CampaignRun.window_needs_review` (BooleanField, default `False`) added, matching the existing `site_raw`/`site_needs_review` shape exactly
- Migration 0006 (two plain `AddField` operations, no `RunPython`/backfill/constraint change) generated and applied to the real dev DB (`src/fomo_db.sqlite3`) — `makemigrations --check --dry-run` confirms model/schema are in sync
- `render_window_start()`'s TBD branch now renders a `title="..."` tooltip with `original_obs_date_raw` when set, via `format_html`'s auto-escaping positional argument (mitigates stored-XSS from community-editable sheet text, T-20-03); blank values fall back to the unchanged plain TBD badge
- Single-night and range branches of `render_window_start()` left untouched (Pitfall 5)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add original_obs_date_raw and window_needs_review fields + migration 0006** - `fe58594` (feat)
2. **Task 2: TBD-badge tooltip showing original_obs_date_raw (D-08)** - `950f649` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified
- `solsys_code/models.py` - Two new `CampaignRun` fields (`original_obs_date_raw`, `window_needs_review`)
- `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py` - Two `AddField` operations, no data migration
- `solsys_code/campaign_tables.py` - `render_window_start()` TBD-badge tooltip
- `solsys_code/tests/test_campaign_models.py` - `TestCampaignRunWindowNeedsReviewFields` (defaults + persistence)
- `solsys_code/tests/test_campaign_views.py` - Three new tests in `TestWindowColumnRendering` (tooltip present, blank-title, HTML-escaping)

## Decisions Made
- Renamed Django's auto-generated migration filename (`..._and_more.py`) to the plan's explicit, more descriptive name (`..._and_window_needs_review.py`) before applying it — purely cosmetic, no functional change to the operations
- Applied migration 0005 (previously pending on this dev DB) together with 0006, since it had not yet been run from Plan 19's deferred-apply step; both applied cleanly with no errors

## Deviations from Plan

None - plan executed exactly as written. Migration content matched the RESEARCH.md-recommended body verbatim (field types, defaults, verbose_names, no RunPython).

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both new `CampaignRun` fields are live in the dev DB and covered by tests; Plan 03 (the range/TBD CSV import command) can now populate `original_obs_date_raw`/`window_needs_review` on unparseable rows
- The TBD-badge tooltip display path is proven end-to-end (empty, populated, and hostile-markup cases), so Plan 03's import doesn't need to touch display code
- No blockers for Plan 03 or Plan 04

---
*Phase: 20-range-tbd-import-asset-aware-coverage-gap*
*Completed: 2026-07-10*

## Self-Check: PASSED

All created/modified files and both task commits (`fe58594`, `950f649`) verified present.
