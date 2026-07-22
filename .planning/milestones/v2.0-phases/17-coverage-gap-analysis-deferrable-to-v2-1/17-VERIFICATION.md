---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
verified: 2026-07-05T06:00:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 17: Coverage-Gap Analysis (Deferrable to v2.1) Verification Report

**Phase Goal:** A user can see which observable nights for a campaign target and site are not yet
claimed by any run — FOMO's differentiator over any spreadsheet. GAP-01 is a phase-time research
spike that gates GAP-02's implementation approach.
**Verified:** 2026-07-05T06:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A phase-time research spike produces an explicit dark-window-only vs. target-altitude decision before implementation (GAP-01, roadmap SC1) | VERIFIED | `.planning/phases/17-.../17-GAP-01-DECISION.md` exists, states the dark-window-only decision, and cites pre-milestone research (ARCHITECTURE.md/PITFALLS.md/SUMMARY.md/STACK.md) as rationale, per D-01/D-02. Dated before Plan 01's `campaign_gap.py` commit. |
| 2 | User can view observable-but-unclaimed dates for a campaign target + site, computed on request or cache (GAP-02, roadmap SC2) | VERIFIED | `CampaignGapAnalysisView` (`solsys_code/campaign_views.py`) calls `campaign_gap.get_or_compute_gap`; the gap page (`campaignrun_gap_analysis.html`) renders `result.gap_dates` / "No gaps found" empty state, "Last computed" caption, and is reachable via the `campaigns:gap_analysis` URL and the campaign table's "Show Coverage Gaps" button. Human-verify checkpoint in Plan 03 approved by the user ("approved"). |
| 3 | Gap computation never runs inline in the table view and never imports `ephem_utils`/heavy views module at module scope (roadmap SC3) | VERIFIED | `test_table_view_does_not_trigger_computation` passes (confirmed live run). Import-guard grep over `campaign_gap.py`, `campaign_views.py`, `test_campaign_gap.py` returns zero matches (exit 1), confirmed independently. `TestNoHeavyEphemerisImport` class exists and passes. |
| 4 | `claimed_dates()` excludes cancelled/not_awarded/weather_tech_failure runs and derives the night from `obs_date` else `ut_start` via site-local convention (D-05/D-06/D-07) | VERIFIED | Code inspected in `campaign_gap.py:138-211`; `TestClaimedDates` (5 methods) and `TestClaimedDatesMultiTarget` pass. |
| 5 | A CampaignRun with neither `obs_date` nor `ut_start` is flagged undated, never dropped nor counted as claiming (D-08) | VERIFIED | `claimed_dates()`'s `else: undated_runs.append(run)` branch; `test_undated_runs_flagged` passes. |
| 6 | `clamp_date_range` enforces the 90-day default / 180-day hard cap regardless of client input, and floors a past `end_date` at `start` (D-11, WR-02 fix) | VERIFIED | Code at `campaign_gap.py:44-63`; `TestClampDateRange` (3 methods) passes; WR-02 fix (`max(start, min(...))`) present in code and confirmed by commit `8d93714`. |
| 7 | Cache key includes all four dimensions with an explicit `'none'` encoding for a null target (D-10) | VERIFIED | `build_gap_cache_key()` code inspected; `TestBuildGapCacheKey` (2 methods) passes. |
| 8 | Server-side IDOR validation rejects out-of-scope/non-numeric target and site pks with 400, never crashing (T-17-01, CR-01 fix) | VERIFIED | Independently reproduced: a fresh scratch integration test (not part of the committed suite) hit `/campaigns/<pk>/gaps/?site=abc` against the current code and got HTTP 400 (not a 500), and hit the gap URL with a blank-timezone site + `ut_start`-only run and got HTTP 200 (not a 500) — confirming both CR-01 and CR-02 fixes are real and load-bearing, not just claimed in 17-REVIEW-FIX.md. `test_rejects_out_of_scope_target_and_site` also passes in the committed suite. |

**Score:** 8/8 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/17-.../17-GAP-01-DECISION.md` | Dark-window-only decision doc | VERIFIED | Exists, ~34 lines, Decision/Rationale/Consequences sections present. |
| `solsys_code/campaign_gap.py` | Pure-logic gap computation core | VERIFIED | All required symbols present (`clamp_date_range`, `build_gap_cache_key`, `observable_dates`, `claimed_dates`, `_observing_night_date`, `_compute_gap`, `get_or_compute_gap`, `_EXCLUDED_RUN_STATUSES`, `GAP_CACHE_TTL_SECONDS`, `DEFAULT_WINDOW_DAYS`, `MAX_WINDOW_DAYS`). `ruff check`/`ruff format --check` clean (confirmed live). |
| `solsys_code/campaign_forms.py` | `CampaignGapAnalysisForm` | VERIFIED | Campaign-scoped querysets, GET-method crispy layout confirmed by reading Plan 02 wiring and passing tests. |
| `solsys_code/campaign_views.py` | `CampaignGapAnalysisView` + `gap_analysis_available` + `_as_pk_or_none` | VERIFIED | Read in full; server-side IDOR guard, clamp, cache wiring, and both CR-01/CR-02-related fixes present and exercised. |
| `solsys_code/campaign_urls.py` | `campaigns:gap_analysis` route | VERIFIED | `reverse('campaigns:gap_analysis', kwargs={'pk': 1})` resolves per Plan 02's own verify step (implied by passing `TestGapAnalysisView`/`TestGapAnalysisButton` which use `reverse(...)`). |
| `solsys_code/tests/test_campaign_gap.py` | Unit + integration tests, 8 classes | VERIFIED | 8 test classes / 23 test methods confirmed present by grep; all 23 pass in a live run (`OK`, 3.9s). |
| `src/templates/campaigns/campaignrun_gap_analysis.html` | Full UI-SPEC page | VERIFIED | Read in full; contains title, wait-state notice, "Last computed" caption, gap-date list/empty state, D-08 alert-warning (undated + unattributed), D-03 caveat (count>0 gated), IDOR alert-danger, back link. Loads without `TemplateSyntaxError` (confirmed via Django template loader). |
| `src/templates/campaigns/campaignrun_table.html` | D-14-gated trigger button | VERIFIED | Read in full; live `<a>` link when available, `disabled` `btn-primary` + helper text when not. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `campaign_gap.py` | `solsys_code.telescope_runs.sun_event` | direct import | WIRED | Only ephemeris import in the module; grep confirms no heavy-module import anywhere in the phase's files. |
| `CampaignGapAnalysisView` | `campaign_gap.get_or_compute_gap` | direct call, post-validation | WIRED | Called only after both target and site pass server-side membership checks and the form validates (`form.is_valid()`). |
| `campaign_urls.py` | `CampaignGapAnalysisView` | `path('<int:pk>/gaps/', ...)` | WIRED | Route present; `TestGapAnalysisView`/`TestGapAnalysisButton` exercise it via `reverse()` and pass. |
| `campaignrun_table.html` | `campaigns:gap_analysis` | D-14 `gap_analysis_available` flag | WIRED | `CampaignRunTableView.get_context_data()` supplies the flag (Plan 03 deviation, confirmed present in `campaign_views.py`); `TestGapAnalysisButton` (3 methods) proves all 3 gating branches at the rendered-HTML level. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full `test_campaign_gap` suite passes | `python manage.py test solsys_code.tests.test_campaign_gap -v 1` | `Ran 23 tests in 3.915s / OK` | PASS |
| Full `solsys_code` app suite passes (regression check) | `python manage.py test solsys_code` | `Ran 326 tests in 83.599s / OK` | PASS |
| `ruff check`/`ruff format --check` clean on all phase source files | `ruff check ...` / `ruff format --check ...` | "All checks passed!" / "5 files already formatted" | PASS |
| Import guard: no heavy-ephemeris/views import in phase files | `grep -rnE ...` | exit 1 (zero matches) | PASS |
| CR-01 fix: non-numeric `site` GET param does not crash | scratch integration test, `GET /campaigns/<pk>/gaps/?site=abc` | HTTP 400 | PASS |
| CR-02 fix: blank-timezone site with `ut_start`-only run does not crash | scratch integration test, `GET /campaigns/<pk>/gaps/?site=<blank-tz-site-pk>` | HTTP 200 | PASS |
| No debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) in phase files | `grep -n -E "TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER"` | no matches | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GAP-01 | 17-01 | Phase-time research spike decides dark-window-only vs. altitude filtering before implementation | SATISFIED | `17-GAP-01-DECISION.md` exists, predates `campaign_gap.py`'s implementation commit, and is referenced by module docstrings throughout. |
| GAP-02 | 17-01, 17-02, 17-03 | User can view observable-but-unclaimed dates for a campaign target + site, computed on request or cached, never inline, never importing `ephem_utils` at module scope | SATISFIED | End-to-end path (form → view → cache → template) verified; IDOR/T-17-01, D-09/D-10/D-11/D-12/D-14 all exercised by passing tests, independently re-run. |

No orphaned requirements — REQUIREMENTS.md maps only GAP-01/GAP-02 to Phase 17, both claimed by the plans and both satisfied.

### Anti-Patterns Found

None. No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers in any phase-modified file. No hardcoded empty-data stubs found in `campaign_gap.py`/`campaign_views.py`/templates on inspection.

### Code Review Fix Verification (critical focus of this run)

Two CRITICAL bugs (CR-01, CR-02) were found by `gsd-code-review` after the plans' own SUMMARY.md
files were written, and warnings WR-01 through WR-05 were also raised. `17-REVIEW-FIX.md` claims
all 7 were fixed in commits `8d93714` and `2b7a7e8`, verified against a 326/326 suite. This
verification did not take that claim at face value — it re-read the actual code and independently
re-executed both crash scenarios:

- **CR-01** (non-numeric `target`/`site` GET params → unhandled 500): confirmed fixed. `_as_pk_or_none()` helper (new, `campaign_views.py`) guards both lookups before they reach `.filter(pk=...)`. Reproduced live: `?site=abc` now returns 400, not a `ValueError`-driven 500.
- **CR-02** (`claimed_dates()` crash on blank-timezone site): confirmed fixed. The per-run `_observing_night_date()` call is now wrapped in `try/except ValueError` (`campaign_gap.py:199-207`), routing the run into `undated_runs` instead of crashing. Reproduced live: a run with `ut_start` set and a site with `timezone=''` now renders 200, not a 500.
- **WR-01** (PII fetched into cached result): confirmed fixed — `claimed_dates()`'s queryset now uses `.only('pk', 'obs_date', 'ut_start')` before populating `undated_runs`/`unattributed_runs`.
- **WR-02** (`clamp_date_range` doesn't floor at `start`): confirmed fixed — `max(start, min(requested_end, max_end))` present in code.
- **WR-03** (hand-parsed `end_date` bypassing the form): confirmed fixed — the view now calls `form.is_valid()` and reads `form.cleaned_data.get('end_date')`.
- **WR-04** (redundant TOCTOU `.exists()`+`.get()` pairs): confirmed fixed — both target/site resolution now use a single `.filter(pk=...).first()`.
- **WR-05** (`claimed_dates()` ignores `[start, end]`): confirmed addressed via the documented-invariant approach (docstring note added, not range-scoping the query) — a lower-risk, explicitly-acknowledged choice, consistent with the fix report.

Both critical fixes were re-verified with fresh, independently-authored test code (not the
project's own committed tests), executed against the current checkout, and produced the expected
non-crashing status codes. This phase's must-haves for D-06/D-07 night derivation and server-side
IDOR validation depend on these fixes holding, and they do.

### Human Verification Required

None outstanding. Plan 03's `checkpoint:human-verify` task was presented and the human reviewer
responded "approved" with no issues raised (per 17-03-SUMMARY.md's Checkpoint Resolution section).
No further human verification items were identified during this pass.

### Gaps Summary

No gaps found. All roadmap Success Criteria and PLAN frontmatter must-haves are verified against
the actual codebase, not merely claimed in the SUMMARY files. The two critical post-SUMMARY code
review findings (CR-01, CR-02) were independently re-verified as fixed by exercising the exact
crash scenarios described in 17-REVIEW.md against the current code, rather than trusting
17-REVIEW-FIX.md's narrative alone.

---

*Verified: 2026-07-05T06:00:00Z*
*Verifier: Claude (gsd-verifier)*
