---
phase: 17-coverage-gap-analysis-deferrable-to-v2-1
fixed_at: 2026-07-05T04:40:00Z
review_path: .planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 17: Code Review Fix Report

**Fixed at:** 2026-07-05T04:40:00Z
**Source review:** .planning/phases/17-coverage-gap-analysis-deferrable-to-v2-1/17-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (critical + warning): 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Non-numeric `target`/`site` GET params crash the view with an unhandled 500 instead of the documented 400

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** 2b7a7e8
**Applied fix:** Added a module-level `_as_pk_or_none()` helper that parses a raw GET-param
string as an int and returns `None` on any parse failure. Both the `target` and `site`
lookups in `CampaignGapAnalysisView.get()` now route through this helper before ever reaching
`.filter(pk=...)`, so a non-numeric pk (e.g. `?site=abc`) now returns the documented
`HttpResponseBadRequest` with `idor_error` set, instead of an unhandled `ValueError` → 500.

### CR-02: `claimed_dates()` crashes with an unhandled `ValueError` for any site with a blank `timezone`

**Files modified:** `solsys_code/campaign_gap.py`
**Commit:** 8d93714
**Applied fix:** Wrapped the `_observing_night_date(run.ut_start, site.timezone)` call inside
`claimed_dates()`'s per-run loop in a `try/except ValueError`, mirroring `observable_dates()`'s
existing per-date log+skip discipline. A run whose site has a blank/invalid timezone is now
logged at debug level and appended to `undated_runs` instead of crashing the whole gap-analysis
request with an unhandled 500.

## Warnings — Fixed

### WR-01: Public gap-analysis view fetches full `CampaignRun` rows (including PII fields) into the cached result

**Files modified:** `solsys_code/campaign_gap.py`
**Commit:** 8d93714
**Applied fix:** Restricted the `claimed_dates()` queryset with `.only('pk', 'obs_date',
'ut_start')` before it's used to populate `undated_runs`/`unattributed_runs` (which flow into
the cached `_compute_gap()` result). `.only()` (rather than `.values()`) was chosen deliberately
to preserve `CampaignRun`-instance pk-based equality and attribute access relied on by existing
tests (e.g. `assertIn(run, undated)`), while still ensuring `contact_person`/`contact_email`
are never fetched on this code path — matching `CampaignRunTableView`'s D-13 "restrict the
queryset, not just the rendered output" precedent.

### WR-02: `clamp_date_range()` doesn't enforce `end >= start`

**Files modified:** `solsys_code/campaign_gap.py`
**Commit:** 8d93714
**Applied fix:** `clamp_date_range()` now also floors the requested end date at `start`:
`max(start, min(requested_end, max_end))`. A past `end_date` (e.g. `?end_date=2020-01-01`) can
no longer produce an `end < start` range that silently yields a misleading "no gaps found."

### WR-03: `end_date` parsed by hand from raw `request.GET`, bypassing the bound form

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** 2b7a7e8
**Applied fix:** `CampaignGapAnalysisView.get()` now calls `form.is_valid()` and reads
`form.cleaned_data.get('end_date')` instead of re-parsing `request.GET.get('end_date')` with
`date.fromisoformat()`. An invalid `end_date` (or any other form-level validation failure) now
renders a 400 with the form's own errors surfaced, instead of silently substituting the 90-day
default window with no error shown to the user.

### WR-04: Redundant TOCTOU-prone `.filter(...).exists()` + `.get(...)` pairs

**Files modified:** `solsys_code/campaign_views.py`
**Commit:** 2b7a7e8
**Applied fix:** Collapsed both the `target` and `site` validation blocks to a single
`.filter(pk=...).first()` query each (combined with the CR-01 pk-parsing guard), removing the
redundant round-trip and closing the narrow TOCTOU window between the existence check and the
subsequent `.get()`.

### WR-05: `claimed_dates()` ignores the `[start, end]` window entirely

**Files modified:** `solsys_code/campaign_gap.py`
**Commit:** 8d93714
**Applied fix:** Chose the "document the invariant" option (lower-risk than range-scoping the
query, given `_compute_gap()`'s correctness doesn't currently depend on it). Added an explicit
docstring note to `claimed_dates()` stating that its returned `claimed_dates`/`undated_runs`/
`unattributed_runs` are campaign/site-wide, not scoped to `[start, end]`, even though the cached
result they feed into is keyed by a date range — so a future consumer of the cached dict doesn't
assume otherwise.

## Skipped Issues

None — all in-scope (critical + warning) findings were fixed.

## Test Results

- `python manage.py test solsys_code.tests.test_campaign_gap` — **23/23 passed**, both after the
  `campaign_gap.py` fix commit (8d93714) and again after the `campaign_views.py` fix commit
  (2b7a7e8).
- `python manage.py test solsys_code` (full suite) — **326/326 passed**, run after both fix
  commits. No regressions detected.

Info-level findings (IN-01, IN-02) were out of scope for this run (`fix_scope: critical_warning`)
and were not addressed.

---

_Fixed: 2026-07-05T04:40:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
