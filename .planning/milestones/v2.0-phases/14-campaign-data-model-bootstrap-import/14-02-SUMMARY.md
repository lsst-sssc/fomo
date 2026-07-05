---
phase: 14-campaign-data-model-bootstrap-import
plan: 02
subsystem: management-commands
tags: [django, csv-import, mpc-obscodes-api, idempotent-create-or-update, campaign-coordination]

# Dependency graph
requires:
  - "CampaignRun Django model (solsys_code.models.CampaignRun) from 14-01"
provides:
  - "solsys_code.campaign_utils: resolve_site, parse_obs_window, map_observation_status, insert_or_create_campaign_run"
  - "import_campaign_csv management command (solsys_code/management/commands/import_campaign_csv.py)"
affects: [14-03-demo-notebook, 15-per-campaign-table-view, 16-submission-form-approval-queue]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tiered external-lookup resolution (local DB -> external API -> placeholder), adapted from CreateObservatory.form_valid + MPCObscodeFetcher into a plain function"
    - "Never-raise-on-messy-data helper contract: return a usable value plus an explicit flag, mirroring calendar_utils.py's _derive_telescope_class idiom"
    - "No-churn create-or-update (get_or_create + field-diff + conditional save), forked from insert_or_create_calendar_event for a model with no auto-now modified field"

key-files:
  created:
    - solsys_code/campaign_utils.py
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py
  modified: []

key-decisions:
  - "resolve_site length-checks and blank-checks the raw Site Code BEFORE any tier-1/tier-2 attempt, so an oversized code (e.g. JWST's 8-char '500@-170') never reaches Observatory.objects.create() with a truncated/fabricated obscode (D-08/D-09/Pitfall 2)"
  - "parse_obs_window uses three distinct, narrowly-scoped regexes (HH:MM range tolerant of a ';' typo, tilde-prefixed approximate hour, bare-hour-plus-explicit-UTC-marker) rather than a permissive general-purpose date parser, so a stray date-range or free-text garbage cell in UT Time Range never 'succeeds' into a wrong-but-plausible time (RESEARCH.md Anti-Patterns)"
  - "insert_or_create_campaign_run omits 'modified' from update_fields (unlike insert_or_create_calendar_event) because CampaignRun has no auto-now timestamp field"

patterns-established:
  - "Pattern: tiered external-lookup resolution reused verbatim from RESEARCH.md's Pattern 2, only the return-shape adapted from a Django-form side effect to a plain (value, flag) tuple"

requirements-completed: [CAMP-02, CAMP-04]

coverage:
  - id: D1
    description: "resolve_site correctly implements the 3-tier D-08 resolution and never fabricates a site for blank/oversized codes"
    requirement: "CAMP-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_resolve_site_blank_returns_none_needs_review"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_resolve_site_oversized_returns_none_needs_review"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_resolve_site_existing_observatory_hit"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_resolve_site_mpc_miss_creates_placeholder"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_resolve_site_mpc_hit_creates_observatory"
        status: pass
    human_judgment: false
  - id: D2
    description: "parse_obs_window handles HH:MM ranges, semicolon typos, approximate hours, and blank fallback without ever raising on a bad (non-key) UT Time Range; raises only on unparseable Obs. Date"
    requirement: "CAMP-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_hhmm_range"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_semicolon_typo"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_approximate_hour"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_blank_time_falls_back_to_midnight"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_unparseable_date_raises"
        status: pass
    human_judgment: false
  - id: D3
    description: "import_campaign_csv ingests a CSV row-by-row, produces a created/updated/unchanged/skipped/site_needs_review summary, skips-and-logs only natural-key failures, and re-runs idempotently"
    requirement: "CAMP-04"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_creates_campaignrun_with_existing_observatory"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_natural_key_failure_skipped_and_logged"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_idempotent_rerun_no_duplicates"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_unresolvable_site_flags_needs_review_without_skipping_row"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_tier2_mpc_lookup_creates_observatory"
        status: pass
    human_judgment: false
  - id: D4
    description: "A single-target campaign auto-assigns that Target to every imported row (D-07/CAMP-02)"
    requirement: "CAMP-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_auto_resolves_single_target_campaign"
        status: pass
    human_judgment: false

duration: 6min
completed: 2026-07-03
status: complete
---

# Phase 14 Plan 02: Campaign CSV Bootstrap Import Summary

**`campaign_utils.py` (3-tier site resolution, best-effort UT-time parsing, status translation, no-churn create-or-update) plus the `import_campaign_csv` management command, both covered by 20 passing Django tests with the MPC Obscodes API fully mocked.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-03T07:19:55+01:00 (approx.)
- **Completed:** 2026-07-03T07:25:55+01:00
- **Tasks:** 3 completed
- **Files modified:** 3 (all created)

## Accomplishments

- `solsys_code/campaign_utils.py` — four pure helper functions mirroring `calendar_utils.py`'s role for the new command: `resolve_site` (D-08 3-tier resolution reusing `MPCObscodeFetcher`, never fabricating a site for blank/oversized Site Codes), `parse_obs_window` (best-effort UT-time-range parsing tolerant of colon/semicolon typos, approximate hours, and bare-hour-plus-UTC shorthand, raising only on an unparseable `Obs. Date`), `map_observation_status` (case-insensitive substring translation table with a conservative `REQUESTED` default), and `insert_or_create_campaign_run` (no-churn create-or-update forked from `insert_or_create_calendar_event`).
- `solsys_code/management/commands/import_campaign_csv.py` — a `BaseCommand` with a required `--campaign` option and positional `filepath` (D-06); resolves the campaign `TargetList` (found-or-created), auto-assigns a single Target when the campaign has exactly one (D-07), imports the CSV row-by-row via `csv.DictReader`, skip-and-logs only natural-key failures (blank `Telescope / Instrument` or unparseable `Obs. Date`), nulls individual non-key columns on malformed data (D-05), resolves site per row (never skipping on an unresolved site, D-09), backfills bootstrap rows as `approval_status=APPROVED` (D-03), and prints a full `created/updated/unchanged/skipped/site_needs_review` summary. Never imports `solsys_code.views`/`ephem_utils`, avoiding the 1.6 GB SPICE download side effect.
- `solsys_code/tests/test_import_campaign_csv.py` — 20 tests across `TestCampaignUtils` (pure-helper edge cases) and `TestImportCampaignCsv` (command integration, MPC API mocked via `@patch('requests.get')`, no live network calls anywhere), covering D-04 idempotency, D-05 skip-and-log, D-07 auto-target, D-08/D-09 tiered site resolution, and D-03 approval backfill.

## Task Commits

Each task was committed atomically:

1. **Task 1: campaign_utils.py — site resolution, time parsing, status mapping, create-or-update** - `738386b` (feat)
2. **Task 2: import_campaign_csv management command** - `2d7d15b` (feat)
3. **Task 3: Integration + helper tests with mocked MPC API** - `392ef94` (test)

## Files Created/Modified

- `solsys_code/campaign_utils.py` — `resolve_site`, `parse_obs_window`, `map_observation_status`, `insert_or_create_campaign_run`
- `solsys_code/management/commands/import_campaign_csv.py` — the bootstrap-import `BaseCommand`
- `solsys_code/tests/test_import_campaign_csv.py` — `TestCampaignUtils` (10 tests) + `TestImportCampaignCsv` (6 tests) = 20 tests total

## Decisions Made

- `resolve_site` length-checks and blank-checks the raw Site Code before attempting any tier, so an oversized code (JWST's `500@-170`, 8 chars vs. `Observatory.obscode`'s `max_length=4`) is flagged for review with `site=None` rather than truncated or fabricated (Pitfall 2, verified directly by `test_resolve_site_oversized_returns_none_needs_review` and the command-level `test_unresolvable_site_flags_needs_review_without_skipping_row`).
- `parse_obs_window` uses three narrowly-scoped regexes (HH:MM range with `[:;]` separator tolerance, tilde-prefixed approximate hour, bare-hour-plus-explicit-`UTC` marker) instead of a permissive general date parser, per RESEARCH.md's explicit anti-pattern warning against `dateutil.parser.parse` "succeeding" on garbage UT Time Range text.
- `insert_or_create_campaign_run`'s update path uses `save(update_fields=list(fields))` without appending `'modified'` — unlike `insert_or_create_calendar_event`, `CampaignRun` has no auto-now timestamp field, so including a non-existent field name would raise.
- Site-resolution counter (`site_needs_review_count`) reported as a distinct line item in the command's stdout summary, following the Phase 7 `[UNVERIFIED]`-style counter precedent (CONTEXT.md's discretion note).

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written on the implementation side.

### Sequencing note (not a deviation, documented for clarity)

Task 1 is marked `tdd="true"` with a verification command (`./manage.py test
solsys_code.tests.test_import_campaign_csv.TestCampaignUtils`) that targets a test class
defined in Task 3's file (`test_import_campaign_csv.py`). Per the plan's own task
ordering, the test file does not exist until Task 3 completes, so Task 1's automated
verify step could not run standalone at Task 1's commit point. This was resolved by
implementing all three tasks in plan order and running the full verification
(`./manage.py test solsys_code.tests.test_import_campaign_csv`, 20/20 pass, plus the
full `./manage.py test solsys_code` suite, 227/227 pass) once Task 3 landed — consistent
with the plan's own `<verification>` block, which specifies the full-suite command at the
plan level, not per-task.

## Issues Encountered

None.

## User Setup Required

None. The real 3I/ATLAS CSV import (CAMP-04's live run) remains an explicit operator
follow-up outside this plan's automated verification, per the plan's `<objective>` note —
the operator must export the real sheet and run
`import_campaign_csv --campaign "3I/ATLAS" <exported.csv>` separately.

## Next Phase Readiness

- `campaign_utils.py` and `import_campaign_csv.py` are complete and fully tested; `./manage.py test solsys_code` (227 tests) passes, `ruff check solsys_code/` and `ruff format --check solsys_code/` are clean.
- Plan 03 (paired demo notebook + synthetic fixture, CAMP-05) can now build against a stable, tested command.
- No blockers for Plan 03.

---
*Phase: 14-campaign-data-model-bootstrap-import*
*Completed: 2026-07-03*

## Self-Check: PASSED

All created files (`solsys_code/campaign_utils.py`,
`solsys_code/management/commands/import_campaign_csv.py`,
`solsys_code/tests/test_import_campaign_csv.py`, this SUMMARY.md) confirmed present on
disk. All 3 task commit hashes (`738386b`, `2d7d15b`, `392ef94`) confirmed present in
git log.
