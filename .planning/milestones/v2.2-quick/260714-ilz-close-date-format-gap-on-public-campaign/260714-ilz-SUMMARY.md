---
phase: quick-260714-ilz
plan: 01
subsystem: api
tags: [django, forms, campaign-runs, date-parsing, difflib-adjacent-regex]

requires:
  - phase: 20-range-tbd-import-and-asset-aware-coverage-gap
    provides: parse_obs_window() (campaign_utils.py) — range/TBD-aware Obs. Date parser, already validated against the real 3I/ATLAS sheet.
provides:
  - Public campaign-run submission form (CampaignRunSubmissionForm) now accepts flexible obs_date text (single date, range, blank/TBD) instead of a strict single YYYY-MM-DD DateField.
  - CampaignRunSubmissionView.form_valid() maps the parsed window (window_start/window_end) onto CampaignRun.objects.create() instead of collapsing a single date field.
  - parse_obs_window()'s date-range separator regex now also accepts a double-hyphen ("--") separator, not just single hyphen/en-dash/em-dash/"to".
affects: [submission-form, campaign-run-approval-queue]

tech-stack:
  added: []
  patterns:
    - "Public-form free-text intake delegates to the already-hardened, never-raising campaign_utils.parse_obs_window() parser rather than inventing new date/range parsing logic; a whole-form clean() adapts the parser's needs-review flag into Django's form.add_error() convention (error only on non-blank unparseable text, never on blank/TBD)."

key-files:
  created: []
  modified:
    - solsys_code/campaign_forms.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_utils.py
    - solsys_code/tests/test_campaign_forms.py
    - solsys_code/tests/test_campaign_submission.py

key-decisions:
  - "obs_date changed from forms.DateField to forms.CharField(max_length=255, required=False) with range-aware help_text, parsed in a new whole-form clean() via parse_obs_window(obs_date_raw, '') — no UT-time field exists on the public form, so an empty ut_range_raw is passed through."
  - "Widened parse_obs_window()'s _DATE_RANGE_FULL separator regex from a single hyphen/en-dash/em-dash/\"to\" to -{1,2} (plus en/em dash and \"to\") so a double-hyphen range like '2027-04-20 -- 2027-05-11' — the ASCII stand-in for an en/em dash a public submitter is far more likely to type — parses correctly. This is shared logic also used by the CSV importer; all pre-existing single-hyphen/en-dash/em-dash/\"to\" shapes are unaffected (regression-checked against test_import_campaign_csv.py, 89/89 scoped tests pass)."
  - "Requirement 7 investigated, not fixed: the existing except-IntegrityError handler in CampaignRunSubmissionView.form_valid() already covers a duplicate resolved-window collision (single OR range) via the same unique_campaign_run_resolved_window partial constraint keyed on (campaign, telescope_instrument, window_start, window_end). Only the user-facing message wording was broadened ('on this observing date' -> 'for this observing window') to read correctly for both single dates and ranges. No new handler or schema change was needed."

patterns-established:
  - "Non-model-backed derived fields (window_start/window_end) are set into cleaned_data from a whole-form clean() and consumed directly by the view via form.cleaned_data['window_start']/['window_end'], keeping the parsing contract (parser never raises; both-None-or-both-set) entirely inside the form layer."

requirements-completed: [SUBMIT-01]

coverage:
  - id: D1
    description: "A multi-night range (e.g. '2027-04-20 -- 2027-05-11') submitted through the public form creates one CampaignRun spanning window_start..window_end, with no Django 'Enter a valid date' failure."
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_genuine_multi_night_range_is_valid"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmissionObsDateWindow.test_multi_night_range_creates_one_run_with_resolved_window"
        status: pass
    human_judgment: false
  - id: D2
    description: "Single date, identical double-hyphen range, and identical 'to'-range all still collapse to window_start == window_end (single-night behavior preserved)."
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_single_date_collapses_to_start_equals_end"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_identical_double_hyphen_range_collapses_to_single_night"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_identical_to_separated_range_collapses_to_single_night"
        status: pass
    human_judgment: false
  - id: D3
    description: "Blank obs_date still produces a TBD run: window_start and window_end both None."
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_blank_obs_date_is_valid_and_yields_tbd_window"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmissionObsDateWindow.test_blank_obs_date_creates_one_tbd_run"
        status: pass
    human_judgment: false
  - id: D4
    description: "Genuinely unparseable non-blank obs_date text (or a reversed range) produces a friendly obs_date form error and re-renders the form (HTTP 200), never a 500, and creates no CampaignRun."
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_unparseable_obs_date_text_is_invalid_with_friendly_error"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_reversed_range_is_invalid_with_friendly_error"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmissionObsDateWindow.test_unparseable_obs_date_re_renders_form_creates_no_run"
        status: pass
    human_judgment: false
  - id: D5
    description: "Two submissions proposing the same resolved window (single OR range) surface the existing friendly IntegrityError non-field form error, not a 500 (requirement 7 investigated: existing handler already covers the range case, message wording broadened)."
    requirement: "SUBMIT-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmissionObsDateWindow.test_duplicate_range_submission_shows_friendly_form_error"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_duplicate_natural_key_submission_shows_friendly_form_error"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-07-14
status: complete
---

# Quick Task 260714-ilz: Close date-format gap on public campaign-run submission form Summary

**Public `CampaignRunSubmissionForm.obs_date` now accepts flexible single-date/range/blank text via `parse_obs_window()`, closing the hard Django date-validation failure that blocked multi-night range submissions.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-14T13:30Z (approx., first task commit)
- **Completed:** 2026-07-14T13:39Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `obs_date` changed from a strict `forms.DateField` to a `forms.CharField(max_length=255, required=False)` with range-aware help text, parsed by a new whole-form `clean()` via `parse_obs_window()`.
- `CampaignRunSubmissionView.form_valid()` now maps the parsed `window_start`/`window_end` onto `CampaignRun.objects.create()` instead of collapsing a single `DateField` into both.
- A real parsing gap in the shared `parse_obs_window()` regex was found and fixed: the double-hyphen range separator (`'2027-04-20 -- 2027-05-11'`) used throughout the plan's own examples was not previously supported — only a single hyphen/en-dash/em-dash/`"to"` was.
- The duplicate-submission IntegrityError message was broadened to read correctly for both single dates and ranges; requirement 7 was investigated and confirmed the existing handler needs no schema/handler change.
- Full test coverage added for all six required cases (single date, identical double-hyphen range, identical `'to'`-range, genuine multi-night range, blank/TBD, unparseable text) plus reversed-range and range-duplicate-collision cases, at both the form level and the end-to-end POST level.

## Task Commits

1. **Task 1: Accept flexible date/range input on the public form and map the parsed window on save** - `0e0a6ff` (feat)
2. **Task 2 (deviation): Fix parse_obs_window's separator regex to accept double-hyphen ranges** - `608b06a` (fix, Rule 1)
3. **Task 2: Cover single/identical-range/1-night/multi-night/TBD/unparseable + range-duplicate collision** - `f7b3ca0` (test)

**Plan metadata:** (pending — this docs commit)

## Files Created/Modified

- `solsys_code/campaign_forms.py` - `obs_date` is now a free-text `CharField`; new whole-form `clean()` calls `parse_obs_window()` and sets `cleaned_data['window_start']`/`['window_end']`, erroring only on non-blank unparseable text.
- `solsys_code/campaign_views.py` - `form_valid()` reads the parsed window from `cleaned_data` instead of a single date field; broadened the IntegrityError-collision message wording for windows.
- `solsys_code/campaign_utils.py` - Widened `_DATE_RANGE_FULL`'s separator alternation to also accept a double-hyphen (`-{1,2}`), alongside the existing single-hyphen/en-dash/em-dash/`"to"` shapes.
- `solsys_code/tests/test_campaign_forms.py` - New `CampaignRunSubmissionFormObsDateWindowTest` covering all required form-level cases.
- `solsys_code/tests/test_campaign_submission.py` - New `TestCampaignSubmissionObsDateWindow` covering all required end-to-end POST cases.

## Decisions Made

- Delegated entirely to the already-hardened `parse_obs_window()` rather than writing new parsing logic on the form, per the plan's explicit instruction — the one exception (see Deviations) was a minimal regex widening of the *existing* shared parser, not a new parser.
- Kept the parser's "both-None-or-both-set" contract as an invariant relied upon (not re-implemented) so the model's `campaign_run_window_start_end_null_together` CheckConstraint always holds without extra guard code in the form.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Widened `parse_obs_window()`'s date-range separator regex to accept a double-hyphen**
- **Found during:** Task 2 (writing the required test cases — the plan's own multi-night-range example and identical-range test case both use `'2027-04-20 -- 2027-05-11'` / `'2027-04-20 -- 2027-04-20'`)
- **Issue:** `_DATE_RANGE_FULL` in `campaign_utils.py` only matched a single hyphen, en-dash, em-dash, or literal `"to"` as a range separator — a double-hyphen (`"--"`) fell through to the unparseable/TBD branch, meaning the plan's own must-have example ("Submitting a genuine multi-night range... creates one CampaignRun... no Django 'Enter a valid date' validation failure") would not actually work end-to-end.
- **Fix:** Changed the separator alternation from `[-–—]` to `-{1,2}|[–—]` (plus `to`), so `-{1,2}` matches one *or* two consecutive hyphens while every previously-supported shape (single hyphen, en-dash, em-dash, `"to"`) is unaffected.
- **Files modified:** `solsys_code/campaign_utils.py`
- **Verification:** All 6 required form/submission test cases pass; regression-ran the full pre-existing `test_import_campaign_csv.py` suite (which exercises the original single-hyphen/en-dash/em-dash/`"to"` shapes) — all 89 scoped tests (`test_campaign_forms` + `test_campaign_submission` + `test_import_campaign_csv`) pass, no regressions.
- **Committed in:** `608b06a`

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in shared parser regex)
**Impact on plan:** Necessary for the plan's own stated multi-night-range and identical-range examples to actually work through the public form; no scope creep — the fix is a one-line regex widening in the file `parse_obs_window()` already lives in, shared with (and non-regressive against) the CSV importer.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SUBMIT-01's date-format gap on the public campaign submission form is closed; the public form now has parity with the CSV importer's range/TBD intake.
- `campaign_utils.py`'s `_DATE_RANGE_FULL` regex change is a shared-code fix — any future work touching CSV import date parsing should be aware the separator now also accepts a double-hyphen.
- No paired demo notebook update required: this task touches `campaign_forms.py`/`campaign_views.py`/`campaign_utils.py`, none of which are in CLAUDE.md's notebook-paired module list (`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`).

---
*Phase: quick-260714-ilz*
*Completed: 2026-07-14*

## Self-Check: PASSED

All 5 modified source/test files and the SUMMARY.md itself exist on disk; all 3 task commit hashes (`0e0a6ff`, `608b06a`, `f7b3ca0`) verified present in `git log`.
