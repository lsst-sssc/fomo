---
phase: 14-campaign-data-model-bootstrap-import
plan: 01
subsystem: database
tags: [django, orm, textchoices, migrations, campaign-coordination]

# Dependency graph
requires: []
provides:
  - "CampaignRun Django model (solsys_code.models.CampaignRun) with full 3I-sheet field inventory"
  - "CampaignRun.ApprovalStatus (3-value) and CampaignRun.RunStatus (8-value) TextChoices vocabularies"
  - "Migration solsys_code/migrations/0002_campaignrun.py, applied to local dev DB"
affects: [14-02-csv-bootstrap-import, 15-per-campaign-table-view, 16-submission-form-approval-queue]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two independent django.db.models.TextChoices fields on one model for orthogonal status dimensions (admin review vs. real-world lifecycle), instead of one flat vocabulary"
    - "Nullable FK + *_raw CharField + *_needs_review BooleanField sidecar trio for 'flag, don't silently guess' external-resolution fields (mirrors CalendarEventTelescopeLabel.is_verified)"

key-files:
  created:
    - solsys_code/migrations/0002_campaignrun.py
    - solsys_code/tests/test_campaign_models.py
  modified:
    - solsys_code/models.py

key-decisions:
  - "CampaignRun.campaign FK uses on_delete=PROTECT (not CASCADE/SET_NULL) since it's required (null=False) — prevents accidental loss of campaign history if a TargetList is ever deleted"
  - "site_raw stored as CharField(max_length=255), not TextField, matching Observatory.name/short_name convention for short strings"
  - "No DB-level UniqueConstraint on the natural key — follows CalendarEvent precedent of app-level get_or_create only (deferred to Plan 02's import command)"

patterns-established:
  - "Pattern: two-field TextChoices status split (ApprovalStatus + RunStatus) for orthogonal status dimensions"
  - "Pattern: nullable-FK + *_raw + *_needs_review sidecar trio for tiered external-resolution fields"

requirements-completed: [CAMP-01, CAMP-02, CAMP-03]

coverage:
  - id: D1
    description: "CampaignRun model persists the full 3I-sheet field inventory, required campaign TargetList FK, and re-fetches all field values correctly"
    requirement: "CAMP-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunFieldInventory.test_full_field_inventory_persists_and_reloads"
        status: pass
    human_judgment: false
  - id: D2
    description: "target FK is nullable; a single-target campaign works without ever setting it, and a linked-target campaign resolves correctly via NonSiderealTargetFactory"
    requirement: "CAMP-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunOptionalTarget.test_campaign_run_without_target_persists_and_reloads"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunOptionalTarget.test_campaign_run_with_linked_target_persists_and_reloads"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two-field status vocabulary: approval_status (3 values, default pending_review) and run_status (8 values, default requested)"
    requirement: "CAMP-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunStatusVocabulary.test_default_statuses_on_fresh_campaign_run"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunStatusVocabulary.test_approval_status_has_exactly_three_members"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_models.py#TestCampaignRunStatusVocabulary.test_run_status_has_exactly_eight_members"
        status: pass
    human_judgment: false

duration: 24min
completed: 2026-07-03
status: complete
---

# Phase 14 Plan 01: CampaignRun Model Summary

**`CampaignRun` Django model with two-field TextChoices status vocabulary (3-value approval, 8-value run status), required campaign FK, nullable target/site FKs, migration applied, and 6 model-level tests green.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-07-03T06:46:00Z (approx.)
- **Completed:** 2026-07-03T06:10:18Z
- **Tasks:** 3 completed
- **Files modified:** 3 (1 modified, 2 created)

## Accomplishments
- `CampaignRun` model added to `solsys_code/models.py` with the full 18-field 3I-sheet inventory, required `campaign` FK (`TargetList`, `PROTECT`), nullable `target` FK (`Target`, `SET_NULL`), and nullable `site` FK (`Observatory`, `SET_NULL`) with `site_raw`/`site_needs_review` sidecar fields
- `ApprovalStatus` (3 members, default `PENDING_REVIEW`) and `RunStatus` (8 members, default `REQUESTED`) `TextChoices` classes, passed directly to `choices=` per Django 5.2's direct-class support
- Migration `0002_campaignrun.py` auto-generated and applied to the local SQLite dev DB; `makemigrations --check --dry-run` confirms no pending model changes
- 6 model-level tests covering CAMP-01/02/03, using `NonSiderealTargetFactory` per CLAUDE.md's Target test factory rule

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the CampaignRun model with two-field status vocabulary** - `d3c095e` (feat)
2. **Task 2: [BLOCKING] Generate and apply the CampaignRun migration** - `5560194` (feat)
3. **Task 3: Model-level tests for field inventory, optional target, and status vocabulary** - `8ce8264` (test)

## Files Created/Modified
- `solsys_code/models.py` - Added `CampaignRun` model (`ApprovalStatus`/`RunStatus` TextChoices, full field inventory); `CalendarEventTelescopeLabel` left unchanged
- `solsys_code/migrations/0002_campaignrun.py` - Auto-generated migration creating the `campaignrun` table; depends on `0001_calendareventtelescopelabel`
- `solsys_code/tests/test_campaign_models.py` - Model-level tests for CAMP-01/02/03

## Decisions Made
- `campaign` FK uses `on_delete=PROTECT` per RESEARCH A5 (required FK, prevents silent campaign-history loss)
- `site_raw` is a `CharField(max_length=255)`, matching `Observatory.name`/`short_name` convention rather than `TextField`
- No DB-level `UniqueConstraint` on the natural key — deferred to app-level `get_or_create` in Plan 02's import command, following `CalendarEvent` precedent (RESEARCH Open Question 1)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `CampaignRun` schema is stable and migrated; Plan 02 (CSV bootstrap import) can now build `campaign_utils.py` and `import_campaign_csv.py` against it
- Full `./manage.py test solsys_code` suite (207 tests) passes; `ruff check solsys_code/` and `ruff format --check solsys_code/` clean
- No blockers for Plan 02

---
*Phase: 14-campaign-data-model-bootstrap-import*
*Completed: 2026-07-03*
