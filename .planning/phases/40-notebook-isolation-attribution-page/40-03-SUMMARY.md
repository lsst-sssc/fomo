---
phase: 40-notebook-isolation-attribution-page
plan: 03
subsystem: ui
tags: [bootstrap5, django-templates, playwright, attribution, class-guard]

requires:
  - phase: 40-notebook-isolation-attribution-page
    provides: "40-02's notebook half of the phase (ROADMAP order: notebooks before WARN-04)"
provides:
  - "High-band attribution rows draw a 4px green left bar again (border-0 border-start border-4 border-success), pinned by a GET render test and a Chromium computed-style test"
  - "Eight FOMO templates plus campaign_tables.py on Bootstrap 5.3 class names only; the band filter is a d-flex gap-2 row with a form-select"
  - "TestTemplatesUseBootstrap5ClassNames: fails on any removed Bootstrap 4 class name or a form-control select in any repo-owned template directory"
  - "repo_template_dirs() / repo_template_files() / bootstrap4_hits() shared by the data-attribute and class scans (closes IN-12)"
affects: [40-04 campaign_lifecycle_demo re-execution and WR-15/IN-12 ledger rows]

actuals:
  tokens: 35000
  tasks: 3
  commits: 3

plan_head_before: 1ba08326139047d2b917e0c09731b09da3acf516
plan_head_after: a07545a6976466e498589baa325cb2143d0624ec

tech-stack:
  added: []
  patterns:
    - "Marker classes start with border-0 because Bootstrap reboot gives table rows border-style: solid and .border-4 sets every side"
    - "Template class guard matches whole class tokens with (?<![\\w-])/(?![\\w-]) lookarounds inside class=\"...\" only"
    - "Template scans iterate repo_template_dirs() (DIRS plus in-repo app dirs, never site-packages) and report paths relative to each file's own directory"

key-files:
  created: []
  modified:
    - src/templates/campaigns/attribution_queue.html
    - src/templates/campaigns/campaign_list.html
    - src/templates/campaigns/campaignrun_table.html
    - src/templates/campaigns/approval_queue.html
    - src/templates/campaigns/campaignrun_gap_analysis.html
    - src/templates/campaigns/campaignrun_submit_form.html
    - src/templates/campaigns/submission_thanks.html
    - src/templates/solsys_code/partials/campaign_links.html
    - solsys_code/campaign_tables.py
    - solsys_code/tests/test_attribution_template.py
    - solsys_code/tests/test_campaign_attribution_views.py
    - solsys_code/tests/test_bootstrap5_rendering.py

key-decisions:
  - "Marker per D-18 as amended: High rows drop the border-top separator and the inline border-left-width style"
  - "Left solsys_code/filters.py (Python-built crispy css_class form-group) unchanged: Python-built HTML is not scanned (D-20)"
  - "The render-test fixture's second run uses a later window (12-21 July) so both an event and a record orphan get a non-High candidate"

patterns-established:
  - "Not-vacuous guard on every template scan: at least 20 templates, plus named templates that must be covered"

requirements-completed: [WARN-04]

coverage:
  - id: D1
    description: "A High-band candidate row on the attribution page carries exactly the marker classes and no inline style; no other candidate row carries any of them"
    requirement: WARN-04
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_attribution_template.py#AttributionHighBandMarkerRenderTests"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_attribution_views.py#TestEvidenceColumns.test_evidence_columns_score_and_band_badge_all_present"
        status: pass
    human_judgment: false
  - id: D2
    description: "In Chromium against Bootstrap 5.3.3 a High-band row computes border widths 0/0/0/4px, solid, rgb(25, 135, 84)"
    requirement: WARN-04
    verification:
      - kind: automated_ui
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestBootstrap5Rendering.test_high_band_row_draws_a_green_left_bar_only"
        status: pass
    human_judgment: false
  - id: D3
    description: "Eight FOMO templates and campaign_tables.py use Bootstrap 5.3 class names only; a guard keeps every repo-owned template directory that way"
    requirement: WARN-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestTemplatesUseBootstrap5ClassNames"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestTemplatesUseBootstrap5DataAttributes"
        status: pass
    human_judgment: false
  - id: D4
    description: "The template scans cover every repo-owned template directory, report a hit in a second DIRS entry instead of raising, and never scan third-party app templates (IN-12)"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestTemplatesUseBootstrap5ClassNames.test_a_hit_in_a_second_template_directory_is_reported_not_raised"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#TestTemplatesUseBootstrap5ClassNames.test_third_party_app_template_directories_are_not_scanned"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-10-09
status: complete
---

# Phase 40 Plan 03: Attribution page Bootstrap 5.3 marker and class guard Summary

**High-band attribution rows draw their 4px green left bar again (`border-0 border-start border-4 border-success`), eight templates are on Bootstrap 5.3 class names, and a class-name guard scans every repo-owned template directory**

## Performance

- **Duration:** about 30 min
- **Completed:** 2026-10-09T23:42Z
- **Tasks:** 3 (one tracer, two auto)
- **Files modified:** 12

## Accomplishments

- The attribution page's High-band marker works under Bootstrap 5.3.3: a GET render test pins it to High-band rows only (events and records), and a Chromium test reads the computed style (`0px 0px 0px 4px`, `solid`, `rgb(25, 135, 84)`).
- The band filter is a flex row with a `form-select`; bold labels (`fw-bold`), spacing (`me-N`), `mb-3` wrappers and `text-bg-*` badges replace the removed Bootstrap 4 names in eight templates, and the Create-new-Observatory link in `campaign_tables.py` uses `ms-1`.
- `TestTemplatesUseBootstrap5ClassNames` (six tests) fails on any removed Bootstrap 4 class name inside a `class="..."` value and on any `select` styled `form-control`.
- IN-12 closed: `repo_template_dirs()` returns every `DIRS` entry plus in-repo app template directories outside `site-packages`/`dist-packages`; hits are reported relative to their own directory, so a second `DIRS` entry no longer raises `ValueError`.

## Task Commits

1. **Task 1 (tracer): High-band row marker end to end** - `8688f79` (fix)
2. **Task 2: Bootstrap 5.3 names in the attribution page and four campaign templates** - `888b0cf` (style)
3. **Task 3: last three templates, campaign_tables.py and the class-name guard** - `a07545a` (test)

**Plan metadata:** the docs commit following this summary. The WR-15 and IN-12 ledger rows are marked in 40-04 and cite `8688f79`, `888b0cf` and `a07545a`.

## Files Created/Modified

- `src/templates/campaigns/attribution_queue.html` - marker classes on both candidate-row tags, flex band filter with `form-select`, `fw-*` names
- `src/templates/campaigns/{campaign_list,campaignrun_table,approval_queue,campaignrun_gap_analysis,campaignrun_submit_form,submission_thanks}.html` and `src/templates/solsys_code/partials/campaign_links.html` - Bootstrap 5.3 class renames only
- `solsys_code/campaign_tables.py` - `ml-1` to `ms-1` on the site-cell link
- `solsys_code/tests/test_attribution_template.py` - `AttributionHighBandMarkerRenderTests`
- `solsys_code/tests/test_campaign_attribution_views.py` - evidence-columns test asserts the exact marker class string
- `solsys_code/tests/test_bootstrap5_rendering.py` - computed-style browser test, `repo_template_dirs()`/`repo_template_files()`/`bootstrap4_hits()`, `TestTemplatesUseBootstrap5ClassNames`

## Decisions Made

- Followed the plan's marker and regex decisions as written. `solsys_code/filters.py` (`css_class='form-group col-md-3'`) stays: it is Python-built HTML, not scanned.
- No notebook re-execution here, as planned: `campaign_lifecycle_demo.ipynb` is re-executed once, last, in 40-04.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Render-test fixture: the second run uses a later window**
- **Found during:** Task 1 (RED step)
- **Issue:** The plan's fixture (second run, same window, another site) leaves the record orphan High-band too, because a record carries no telescope and the two runs' instrument names are nearly identical (`record bands ['high', 'high']`). The test could not get a non-High candidate for the record.
- **Fix:** The second run's window is 12-21 July (no overlap with the orphans' 7-8 July night), so both orphans get a Low second candidate. `test_no_marker_when_no_candidate_is_high_band` therefore checks `?band=medium` (empty) and `?band=low` (the two non-High rows) rather than a populated `?band=medium` view.
- **Files modified:** solsys_code/tests/test_attribution_template.py
- **Committed in:** 8688f79

**2. [Rule 3 - Blocking] `--parallel` in Task 2's verify command swallowed the first test label**
- **Found during:** Task 2
- **Issue:** `manage.py test --parallel solsys_code...` parses the label as the worker count.
- **Fix:** Ran with `--parallel auto`; 356 tests passed.
- **Committed in:** n/a (command only)

**3. [Rule 1 - Bug] ruff import-order fix in the Playwright test module**
- **Found during:** Task 3
- **Issue:** `pre-commit run ruff` auto-fixed one finding in `test_bootstrap5_rendering.py` (import ordering after adding `tempfile`/`override_settings`/`get_app_template_dirs`).
- **Fix:** Accepted the autofix; ruff and ruff-format are clean.
- **Committed in:** a07545a

---

**Total deviations:** 3 auto-fixed (1 blocking fixture change, 1 blocking command syntax, 1 lint autofix)
**Impact on plan:** None on scope; all behaviours and artifacts in the plan exist as specified.

## Issues Encountered

Task 1's RED run for the Chromium test was not run against the old template separately; the render tests were (3 failures on the old classes). The Chromium test passes against the new markup, and with the old markup `tr.border-start` does not exist, so its `wait_for` would time out.

## Verification

- `test_attribution_template`, `TestEvidenceColumns`: 18 tests OK.
- Browser tests (new marker test and the existing attribution collapse test): 2 tests OK.
- Swept-page suites (`test_campaign_views`, `test_campaign_approval`, `test_campaign_gap`, `test_campaign_attribution_views`, `test_attribution_template`, data-attribute scan): 356 tests OK.
- Class guard, data-attribute scan, `test_campaign_approval`, `test_campaign_submission`: 164 tests OK.
- Whole `test_bootstrap5_rendering` module plus `test_attribution_template`: 36 tests OK (functional tests included, flaky observatory test passed).
- The django-test hook passed on all three commits; ruff and ruff-format clean.

## Known Stubs

None.

## Threat Flags

None. Only literal class names changed; no new variable output, endpoint or external resource (T-40-09 mitigated: the dismissal-reason XSS test passes in the swept-page suites).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for 40-04: re-execute `campaign_lifecycle_demo.ipynb` with a cell showing the marker, and mark the WR-15, WR-18, IN-12 and IN-15 ledger rows citing `8688f79`, `888b0cf`, `a07545a`.

## Self-Check: PASSED

- Commits `8688f79`, `888b0cf`, `a07545a` are on `issue37-telescope-runs-calendar`.
- All 12 modified files exist and are committed; `solsys_code/filters.py` is unchanged.

---
*Phase: 40-notebook-isolation-attribution-page*
*Completed: 2026-10-09*
