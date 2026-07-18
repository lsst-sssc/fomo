---
phase: 20-range-tbd-import-asset-aware-coverage-gap
plan: 03
subsystem: import
tags: [django, csv-import, parsing, regex, natural-key]

# Dependency graph
requires:
  - phase: 20-range-tbd-import-asset-aware-coverage-gap
    plan: 02
    provides: "CampaignRun.original_obs_date_raw/window_needs_review fields + migration 0006, applied to the dev DB"
provides:
  - "parse_obs_window() 7-tuple return (window_start, window_end, original_obs_date_raw, window_needs_review, ut_start, ut_end, ut_needs_review) -- never raises for any Obs. Date input (D-13)"
  - "_DATE_RANGE_FULL / _DATE_RANGE_COMPACT module regexes -- full-date range (D-12) and compact same-month/rollover range (D-11) parsing"
  - "import_campaign_csv: window_needs_review summary counter; resolved-vs-TBD natural-key branch matching CampaignRun.Meta.constraints' two partial UniqueConstraints (Pitfall 2)"
affects: [20-04-asset-aware-coverage-gap-demo-notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Order-of-attempts parser: exact date -> full-date range regex -> compact-range regex (with rollover) -> TBD catch-all, each guarded by try/except ValueError so stdlib date() validation is the never-raise mechanism (no manual day-count guard needed)"
    - "Natural-key branch on `window_start is None` (resolved-window vs TBD lookup/collision-key shape) mirrors the model's two partial UniqueConstraints exactly -- reusable pattern for any future consumer with a nullable-vs-resolved natural key"

key-files:
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py

key-decisions:
  - "Range/TBD rows skip UT-Time-Range parsing entirely (ut_start=ut_end=None, ut_needs_review=False) per RESEARCH.md's A1 assumption -- no current consumer stores these values for a multi-night window"
  - "No dedicated 'YYYY-MM-?' regex -- it falls through steps 1-3 naturally and lands in the TBD catch-all with original_obs_date_raw set to the verbatim text, per RESEARCH.md's documented simplification"
  - "contact_person is pulled out of the unconditional fields dict and only set via the lookup dict for the TBD branch, kept in fields for the resolved-window branch (Pitfall 2) -- avoids lookup/defaults key-overlap ambiguity in get_or_create"

requirements-completed: [IMPORT-01, IMPORT-02]

coverage:
  - id: D1
    description: "parse_obs_window() never raises for any Obs. Date input; unparseable text returns the TBD tuple (window_start=window_end=None, raw text preserved, window_needs_review=True)"
    requirement: "IMPORT-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_blank_date_returns_tbd_tuple"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_yyyy_mm_question_marker_returns_tbd"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_garbage_free_text_returns_tbd"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils.test_parse_obs_window_compact_range_invalid_day_falls_through_to_tbd"
        status: pass
    human_judgment: false
  - id: D2
    description: "A full-date range ('to'/en-dash/em-dash/hyphen) and a compact same-month/rollover range parse to distinct window_start/window_end, including December -> January year rollover"
    requirement: "IMPORT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestCampaignUtils (test_parse_obs_window_full_range_literal_to/en_dash/em_dash/hyphen, test_parse_obs_window_compact_range_same_month/rollover_same_year/rollover_year_crossing)"
        status: pass
    human_judgment: false
  - id: D3
    description: "import_campaign_csv creates a CampaignRun with the matching window for a range row and a flagged window_needs_review=True TBD row for unparseable text, never silently dropped; the summary reports a window_needs_review counter"
    requirement: "IMPORT-01, IMPORT-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_range_row_creates_window"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_tbd_row_flagged_and_counted"
        status: pass
    human_judgment: false
  - id: D4
    description: "Two TBD rows for the same campaign+telescope but different contact_person both import; two with the same natural key in one batch collide and the second is skipped/logged with no Contact Person/Email PII in the log"
    requirement: "IMPORT-01, IMPORT-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_two_tbd_rows_different_contact_both_import"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_import_campaign_csv.py#TestImportCampaignCsv.test_two_tbd_rows_same_contact_second_collides"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-10
status: complete
---

# Phase 20 Plan 3: Range/TBD Import Summary

**`parse_obs_window()` now parses full-date and compact rollover ranges and never raises for any Obs. Date input (7-tuple TBD contract); `import_campaign_csv` persists both range and flagged-TBD rows instead of skipping them, branching its natural key to match the model's two partial UniqueConstraints exactly.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-07-10T20:24:00Z
- **Completed:** 2026-07-10T20:44:00Z
- **Tasks:** 2
- **Files modified:** 3 (campaign_utils.py, import_campaign_csv.py, test_import_campaign_csv.py)

## Accomplishments

- `_DATE_RANGE_FULL` (`' to '`/en-dash/em-dash/hyphen-separated full-date range, D-12) and `_DATE_RANGE_COMPACT` (same-month/rollover compact range, D-11) module regexes added, both anchored `^...$` and matched via `.match()` after `.strip()` -- no `.search()`, no permissive/fuzzy date parser
- `parse_obs_window()` rewritten to an order-of-attempts parser (exact date -> full-date range -> compact range with rollover -> TBD catch-all) returning a 7-tuple (`window_start, window_end, original_obs_date_raw, window_needs_review, ut_start, ut_end, ut_needs_review`) that **never raises** for any Obs. Date input (D-13) -- stdlib `date()`'s own `ValueError` validation is the never-raise mechanism for the rollover's invalid-day case, no manual day-count guard needed
- Compact-range rollover verified for same-month (`'2025-11-02 -25'`), same-year rollover (`'2025-11-28 -05'` -> Dec 5), and December -> January year-crossing rollover (`'2025-12-20 -03'` -> `2026-01-03`)
- Range/TBD rows skip UT-Time-Range parsing entirely (A1); single-night rows keep the existing UT parsing behavior byte-for-byte unchanged
- `import_campaign_csv.Command.handle()` rewritten: the dead `except ValueError` branch that used to wrap `parse_obs_window()` is removed (only the genuine blank-Telescope/Instrument natural-key failure still raises/skips, D-07); a new `window_needs_review_count` summary counter tracks TBD rows; `collision_key` and the `insert_or_create_campaign_run` lookup dict branch on `window_start is None` to produce either the resolved-window key shape or the `(campaign, telescope_instrument, contact_person)` TBD key shape, matching `CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s exactly (Pitfall 2)
- Two fixture-broken tests (`test_natural_key_failure_skipped_and_logged`, `test_natural_key_failure_log_excludes_contact_pii`) fixed: their `'2025-11-02 -25'` "malformed date" fixture is now a valid compact range under D-11, so both now use a blank `Telescope / Instrument` as the genuine natural-key failure (Pitfall 1)
- New tests: 15 parser-level tests (full-range variants, compact rollover same-year/year-crossing, invalid-day fallback, `'YYYY-MM-?'` marker, garbage free text, blank-date TBD) plus 4 import-integration tests (range-creates-window, TBD-flagged-and-counted, two-TBD-different-contact-both-import, two-TBD-same-contact-collision with PII-free collision log assertion)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend parse_obs_window() with range parsing + never-raise TBD contract (D-11/D-12/D-13)** - `6b49081` (feat)
2. **Task 2: Rewrite import_campaign_csv row loop for range/TBD outcomes (D-05/D-06/D-07, Pitfall 2)** - `622699f` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `solsys_code/campaign_utils.py` - `_DATE_RANGE_FULL`/`_DATE_RANGE_COMPACT` module regexes; `parse_obs_window()` rewritten to the 7-tuple never-raise contract
- `solsys_code/management/commands/import_campaign_csv.py` - row loop rewritten for range/TBD outcomes; `window_needs_review_count` counter; resolved-vs-TBD collision-key/lookup branch
- `solsys_code/tests/test_import_campaign_csv.py` - 9 existing `TestCampaignUtils` parser tests migrated to the 7-tuple arity; `test_parse_obs_window_unparseable_date_raises` rewritten as `test_parse_obs_window_blank_date_returns_tbd_tuple`; 14 new parser tests; 2 fixture-broken import tests fixed; 4 new import-integration tests

## Decisions Made

- Range/TBD rows skip UT-Time-Range parsing entirely (`ut_start=ut_end=None, ut_needs_review=False`) per RESEARCH.md's A1 assumption -- zero observable effect since no current `CampaignRun` field stores these values
- No dedicated `'YYYY-MM-?'` regex introduced -- it falls through steps 1-3 naturally and lands in the TBD catch-all with `original_obs_date_raw` set verbatim, per RESEARCH.md's documented implementation simplification
- `contact_person` is pulled out of the unconditional `fields` dict and only set via the `lookup` dict for the TBD branch (kept in `fields` for the resolved-window branch) to avoid `get_or_create`'s lookup/defaults key-overlap ambiguity (Pitfall 2)

## Deviations from Plan

None - plan executed exactly as written. Regex patterns, rollover algorithm, and the 7-tuple return signature all matched 20-RESEARCH.md's "Concrete D-11/D-12/D-13 Parsing Design" verbatim.

## Issues Encountered

None. One ruff `SIM108` lint (nested `if`/`else` in the rollover branch) was auto-flagged and fixed inline before committing (ternary form), and `ruff format` reflowed a few long single-line unpacking statements in the new tests -- both routine formatting, not functional changes.

## User Setup Required

None - no external service configuration or migrations required (this plan only touches parsing/import logic; the schema fields it populates were added in Plan 02).

## Next Phase Readiness

- `parse_obs_window()`'s 7-tuple contract and the import command's resolved-vs-TBD branching are both proven end-to-end against every real-sheet Obs. Date shape (exact date, full range, compact range with same-month/rollover/year-crossing, blank, `'YYYY-MM-?'`, and garbage free text) -- IMPORT-01/IMPORT-02 requirements are now genuinely satisfied by working code (Plan 02 had already flipped their REQUIREMENTS.md rows to Complete prematurely; this plan's work is what makes that status accurate)
- No blockers for Plan 04 (asset-aware coverage-gap demo notebook)

---
*Phase: 20-range-tbd-import-asset-aware-coverage-gap*
*Completed: 2026-07-10*

## Self-Check: PASSED

All modified files (`solsys_code/campaign_utils.py`, `solsys_code/management/commands/import_campaign_csv.py`, `solsys_code/tests/test_import_campaign_csv.py`) and both task commits (`6b49081`, `622699f`) verified present. `python manage.py test solsys_code.tests.test_import_campaign_csv` (47/47) and `python manage.py test solsys_code` (383/383) both pass; `ruff check`/`ruff format --check` clean on all three plan files.
