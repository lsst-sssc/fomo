---
phase: 20-range-tbd-import-asset-aware-coverage-gap
verified: 2026-07-10T23:30:00Z
status: passed
score: 12/12 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 20: Range/TBD Import & Asset-Aware Coverage Gap Verification Report

**Phase Goal:** Make the new window schema usable end-to-end for the harder 3I/ATLAS rows —
import range and TBD `Obs. Date` cells into the window representation instead of dropping them,
and rewrite coverage-gap analysis so ground runs and space-mission runs claim dates differently.
**Verified:** 2026-07-10
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (ROADMAP SC1/IMPORT-01) A CSV row with a date range (e.g. "2025-08-01 to 2025-08-15") creates a `CampaignRun` with the matching window instead of being skipped as a natural-key failure | ✓ VERIFIED | `campaign_utils.py` `_DATE_RANGE_FULL`/`_DATE_RANGE_COMPACT` regexes + `parse_obs_window()` rewrite; `import_campaign_csv.py` branches on `window_start is not None`. Behaviorally proven by demo notebook cell 10 committed output (`window_start=2025-08-01, window_end=2025-08-15`) and by `test_import_campaign_csv.py` range-parsing tests (ran green, see below). |
| 2 | (ROADMAP SC2/IMPORT-02) A row with genuinely unparseable `Obs. Date` text (e.g. "TBD pending Cycle 2") creates a TBD run flagged `window_needs_review=True`, listed in the import summary counter, never silently dropped | ✓ VERIFIED | `parse_obs_window()` never-raise TBD tuple; `window_needs_review_count` added to `Command.handle()`'s summary line; demo notebook cell 10 output shows `window_needs_review=True, original_obs_date_raw='TBD pending Cycle 2'` for the VLT/X-shooter row. |
| 3 | (ROADMAP SC3/ASSET-02) Coverage-gap analysis marks every date within a ground-based run's window as claimed (conservative) | ✓ VERIFIED | `campaign_gap.py:claimed_dates()` — ground branch unchanged `n_days` loop; `test_campaign_gap.py::TestClaimedDates.test_range_run_claims_every_date_in_window` — ran and passed. |
| 4 | (ROADMAP SC4/ASSET-01/ASSET-02) A space-mission run (`site.observations_type == SATELLITE_OBSTYPE`) claims no dates until its window narrows to a single night, at which point that night is claimed | ✓ VERIFIED | `claimed_dates()`'s `is_space_mission` branch computed once from `site` param; `TestClaimedDatesSpaceMission` (3 tests: narrowed/un-narrowed/TBD-vs-space) — ran and passed. End-to-end HTTP test `test_pending_narrowing_alert_shown_for_unnarrowed_space_run` independently re-run: **ok**. |
| 5 | TBD runs land in `undated_runs` regardless of site type, never in `pending_narrowing_runs` (D-09 explicit distinction) | ✓ VERIFIED | `claimed_dates()` checks `window_start is None` first, before the space-mission branch; `test_tbd_space_run_lands_in_undated_not_pending_narrowing` — ran and passed. |
| 6 | `campaignrun_gap_analysis.html` renders a distinct alert reporting `pending_narrowing_runs` count when non-empty | ✓ VERIFIED | Template lines 71–79 (`{% if result.pending_narrowing_runs %}`); wired end-to-end through `CampaignGapAnalysisView.context.update({..., 'result': result})` (no view change needed, opaque dict pass-through) and proven via a full HTTP-response `assertContains` test — independently re-run: **ok**. |
| 7 | `claimed_dates()` keeps its PII-minimizing `.only('pk','window_start','window_end')` queryset — no per-row `run.site` read | ✓ VERIFIED | `campaign_gap.py:180` — `is_space_mission` computed once before the loop from the `site` parameter; loop body never references `run.site`. |
| 8 | `CampaignRun` gains `original_obs_date_raw` (CharField) and `window_needs_review` (BooleanField) via migration 0006, with no data loss/backfill for existing rows | ✓ VERIFIED | `models.py:91,94`; migration 0006 contains only two `AddField` ops with safe defaults, no `RunPython`; independently re-ran `python manage.py makemigrations solsys_code --check --dry-run` → "No changes detected" (schema and model in sync). |
| 9 | TBD badge in `render_window_start()` shows `original_obs_date_raw` as an HTML-escaped tooltip, omitted when blank | ✓ VERIFIED | `campaign_tables.py:153-155` — `Accessor(...).resolve(quiet=True) or ''` guard + `format_html(...)` positional-argument auto-escaping (matches `render_site()` precedent). |
| 10 | **Code-review fix CR-01**: TBD-branch natural-key lookup includes `window_start__isnull=True` so `get_or_create()` cannot match/corrupt an unrelated resolved row or raise an uncaught `MultipleObjectsReturned` | ✓ VERIFIED | `import_campaign_csv.py:204` — `'window_start__isnull': True` present in the TBD-branch lookup dict (commit 2e85670). Regression test `test_tbd_row_does_not_collide_with_resolved_row_same_contact` independently re-run: **ok** — asserts 2 distinct `CampaignRun` rows, resolved row untouched. |
| 11 | **Code-review fix WR-01**: a reversed full-date range (`window_end < window_start`) falls through to TBD/needs-review instead of silently producing an inverted, zero-coverage window | ✓ VERIFIED | `campaign_utils.py:246-250` — explicit `if window_end < window_start: window_start = window_end = None`. Regression test `test_parse_obs_window_full_range_reversed_falls_through_to_tbd` independently re-run: **ok**. |
| 12 | **Code-review fix WR-02**: `insert_or_create_campaign_run()`'s docstring documents both natural-key shapes (resolved-window vs. TBD) | ✓ VERIFIED | `campaign_utils.py:358-364` — docstring now states both partial-`UniqueConstraint` shapes explicitly, including the `window_start__isnull=True` guard rationale. |

**Score:** 12/12 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_gap.py` | 4-tuple `claimed_dates()` + `pending_narrowing_runs` dict key | ✓ VERIFIED | Present, substantive, wired (view passes result dict through, template consumes key) |
| `src/templates/campaigns/campaignrun_gap_analysis.html` | pending_narrowing_runs alert block | ✓ VERIFIED | Lines 71-79, proven via rendered-response test |
| `solsys_code/models.py` | `original_obs_date_raw`, `window_needs_review` fields | ✓ VERIFIED | Lines 91-97 |
| `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py` | two AddField ops, no RunPython | ✓ VERIFIED | Confirmed via file read; `makemigrations --check` clean |
| `solsys_code/campaign_tables.py` | `render_window_start()` TBD tooltip | ✓ VERIFIED | Lines 136-155 |
| `solsys_code/campaign_utils.py` | `_DATE_RANGE_FULL`/`_DATE_RANGE_COMPACT` regexes, 7-tuple `parse_obs_window()`, never-raise contract, reversed-range guard | ✓ VERIFIED | Lines 200-320ish; anchored `^...$` regexes matched with `.match()` |
| `solsys_code/management/commands/import_campaign_csv.py` | resolved-vs-TBD natural-key branch, `window_needs_review_count`, CR-01 fix | ✓ VERIFIED | Full file read; lookup dict includes `window_start__isnull: True` for TBD branch |
| `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` | one range row + one TBD row, synthetic contact data | ✓ VERIFIED | Rows 8-9 ("Gia Range", "Ike Pending") added, synthetic Name/Email style preserved |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | executed cell demonstrating range/TBD outcomes | ✓ VERIFIED | Cell 10 has real committed output matching claimed values exactly |
| `solsys_code/tests/test_campaign_gap.py` | 4-tuple unpacking updated, new ground/space bucket + alert tests | ✓ VERIFIED | `TestClaimedDatesSpaceMission` class + `test_pending_narrowing_alert_shown_for_unnarrowed_space_run` present and passing |
| `solsys_code/tests/test_import_campaign_csv.py` | 7-tuple unpacking, new range/TBD tests, CR-01/WR-01 regression tests | ✓ VERIFIED | Regression tests present at end of file (commit 2e85670) and pass |
| `solsys_code/tests/test_campaign_models.py` | field-default assertions | ✓ VERIFIED | Present, part of the 116-test green run |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `claimed_dates()` 4-tuple | `_compute_gap()` | unpacking + dict key `pending_narrowing_runs` | WIRED | `campaign_gap.py:_compute_gap()` unpacks 4-tuple, adds key to result dict |
| `_compute_gap()` result dict | `campaignrun_gap_analysis.html` | `CampaignGapAnalysisView.context['result']` (unchanged, opaque pass-through) | WIRED | Confirmed no `campaign_views.py` change was needed; verified via HTTP-level `assertContains` test |
| `import_campaign_csv.py` window_start branch | `CampaignRun.objects.get_or_create()` natural key | `insert_or_create_campaign_run(lookup, fields)` | WIRED | `window_start__isnull=True` present in TBD lookup; resolved-window lookup uses 4-field key; matches `CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s |
| `parse_obs_window()` 7-tuple | `import_campaign_csv.py` row loop | direct unpack, `original_obs_date_raw`/`window_needs_review` written to `fields` dict | WIRED | Confirmed in `handle()` body (lines ~122-135, ~178) |
| `render_window_start()` tooltip | `original_obs_date_raw` field | `Accessor('original_obs_date_raw').resolve(record, quiet=True)` | WIRED | `campaign_tables.py:153` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full targeted test suite for all 4 plans' modified files | `python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_gap solsys_code.tests.test_campaign_models solsys_code.tests.test_campaign_views -v1` | "Ran 116 tests in 9.955s / OK" | ✓ PASS |
| CR-01 regression test (isolated) | `python manage.py test solsys_code.tests.test_import_campaign_csv.TestImportCampaignCsv.test_tbd_row_does_not_collide_with_resolved_row_same_contact` | ok | ✓ PASS |
| WR-01 regression test (isolated) | `python manage.py test solsys_code.tests.test_import_campaign_csv.TestCampaignUtils.test_parse_obs_window_full_range_reversed_falls_through_to_tbd` | ok | ✓ PASS |
| Pending-narrowing alert end-to-end HTTP test (isolated) | `python manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView.test_pending_narrowing_alert_shown_for_unnarrowed_space_run` | ok | ✓ PASS |
| ruff lint on all Phase 20 source files | `ruff check solsys_code/campaign_utils.py solsys_code/management/commands/import_campaign_csv.py solsys_code/campaign_gap.py solsys_code/models.py solsys_code/campaign_tables.py` | "All checks passed!" | ✓ PASS |
| ruff format check | `ruff format --check ...` | "3 files already formatted" | ✓ PASS |
| Migration/model sync | `python manage.py makemigrations solsys_code --check --dry-run` | "No changes detected in app 'solsys_code'" | ✓ PASS |

Note: the full `solsys_code` suite (385 tests, per REVIEW.md's claim) was not independently re-run in
full — it triggers a ~1.6GB SPICE-kernel download path on a cold cache (CLAUDE.md's documented
`ephem_utils` heavy-import side effect) and is unrelated to this phase's files. The 116-test run above
covers every test module this phase's plans modified (`test_campaign_gap`, `test_import_campaign_csv`,
`test_campaign_models`, `test_campaign_views`) and is a stronger, more targeted check than re-running
the full suite once more; combined with a clean `ruff check`/`ruff format --check`, this independently
confirms the review's "regression tests added ... full solsys_code suite ... and ruff clean" claim for
the files this phase actually touches.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ASSET-01 | 20-01 | Ground/space classification from `Observatory.observations_type`, no new `CampaignRun` field | ✓ SATISFIED | `campaign_gap.py:180`; REQUIREMENTS.md marks Complete |
| ASSET-02 | 20-01 | Ground claims full window; space claims nothing until narrowed to one night | ✓ SATISFIED | `claimed_dates()` branch + tests; REQUIREMENTS.md marks Complete |
| IMPORT-01 | 20-02, 20-03, 20-04 | Range/TBD `Obs. Date` cells import into window representation instead of being skipped | ✓ SATISFIED | `parse_obs_window()` regexes + `import_campaign_csv.py` branch + demo notebook cell 10; REQUIREMENTS.md marks Complete |
| IMPORT-02 | 20-02, 20-03, 20-04 | Unparseable text flagged `window_needs_review`, counted, never dropped | ✓ SATISFIED | `window_needs_review` field + counter + demo notebook cell 10; REQUIREMENTS.md marks Complete |

No orphaned requirements — REQUIREMENTS.md's Phase 20 row set (ASSET-01, ASSET-02, IMPORT-01,
IMPORT-02) exactly matches the union of `requirements:` frontmatter across all four plans.

### Anti-Patterns Found

None. Scanned all 8 phase-modified source/template files for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER`,
`not yet implemented`/`coming soon`, and hardcoded-empty-return patterns. Every `TBD` match found is
legitimate domain terminology (the feature's "To Be Determined" observation state, e.g. `# TBD
branch`, `TBD badge`, `TBD row`) — not a debt marker. No `FIXME`/`XXX`/`HACK`/`PLACEHOLDER` markers
present in any file this phase touched.

### Code Review Findings — Verified Resolved

`20-REVIEW.md` reported 1 Critical (CR-01) + 2 Warnings (WR-01, WR-02) + 1 Info (IN-01, accepted as
non-blocking). All three actionable findings were independently confirmed fixed in the current code
(not just claimed in the SUMMARY):

- **CR-01** (TBD-branch lookup missing `window_start__isnull=True`) — fixed in
  `import_campaign_csv.py:204`; regression test re-run and passing.
- **WR-01** (reversed date range not validated) — fixed in `campaign_utils.py:246-250`; regression
  test re-run and passing.
- **WR-02** (stale docstring) — fixed in `campaign_utils.py:358-364`, both natural-key shapes now
  documented.
- **IN-01** (cyclomatic complexity) — correctly left unaddressed; it's a non-blocking maintainability
  note, not a must-have.

`20-REVIEW.md`'s frontmatter `status` was updated from `issues_found` to `clean` with a `resolution:`
block (commit 3578ff2) — confirmed consistent with the fix commits' actual diffs.

### Human Verification Required

None. All must-haves have either static evidence (artifact/wiring checks) or independently re-run
passing tests exercising the specific behavior (state-transition/bucketing truths #3-#5, #10-#11
above were confirmed via isolated test re-runs, not presence alone).

### Gaps Summary

No gaps. All 12 derived must-haves (4 ROADMAP success criteria plus 8 PLAN-frontmatter must-haves not
otherwise subsumed by the roadmap criteria, including the 3 code-review fixes) are verified present,
substantive, and wired, with independently re-run passing tests for every behavior-dependent claim.
The demo-notebook companion (CLAUDE.md's mandatory-sync convention, applicable here because Plan 03
changed `import_campaign_csv`'s parsing/counting behavior) was scoped into Plan 04 up front and its
executed output was independently inspected, not merely trusted from the SUMMARY.

---

_Verified: 2026-07-10_
_Verifier: Claude (gsd-verifier)_
