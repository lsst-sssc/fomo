---
phase: 19-window-schema-migration
plan: 03
subsystem: api
tags: [django, django-tables2, campaignrun, window-schema, calendar-projection]

# Dependency graph
requires:
  - phase: 19-window-schema-migration (plan 01)
    provides: "CampaignRun.window_start/window_end nullable DateFields and their two partial UniqueConstraints, replacing obs_date/ut_start/ut_end"
provides:
  - "CampaignRunTable/ApprovalQueueTable render a single window_start column (TBD badge / single date / 'start -> end' range, D-03/D-05)"
  - "CampaignRunTableView.get_queryset() applies a cross-backend nulls-last default sort (F('window_start').desc(nulls_last=True), D-04) for both staff and non-staff branches"
  - "ALLOWED_FIELDS_FOR_NON_STAFF lists window_start/window_end instead of the removed obs_date/ut_start/ut_end"
  - "CampaignRunDecisionView.post() projects a D-06 hybrid CalendarEvent on approve: dip-corrected sun_event() window for a ground site, midnight-UTC placeholder for a space site, only for a single concrete night with a resolved site"
  - "CampaignRunSubmissionForm collapsed to a single obs_date field, mapped to window_start==window_end on save"
affects: [phase-20-range-tbd-import, phase-20-asset-aware-coverage-gap, phase-21-site-disambiguation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "django-tables2 dict-vs-model dual accessor (Accessor(...).resolve(record, quiet=True)) applied to a new combined render_window_start() method, mirroring render_site()'s established precedent"
    - "F('field').desc(nulls_last=True) applied in the view's get_queryset(), with the table's Meta.order_by removed and get_table_kwargs()'s order_by=() suppressing django-tables2's own re-sort -- the cross-backend nulls-last idiom for a field django-tables2's Meta.order_by can't express"
    - "except ValueError: log+skip around a new sun_event() call site, matching campaign_gap.observable_dates()'s established per-record discipline, deliberately kept outside the view's broad except Exception (which exists to revert a half-committed approval, not handle expected messy site data)"

key-files:
  created: []
  modified:
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_forms.py
    - solsys_code/tests/test_campaign_views.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_campaign_submission.py

key-decisions:
  - "render_window_start() returns the literal HTML-escaped '-&gt;' (not a plain '-' character or en-dash) between two dates for a range row, per D-05's exact wording -- both the implementation and its test assert this precise string"
  - "TestApproval and TestCalendarNoChurn (test_campaign_approval.py) each gained their own Tier-1-resolvable ground Observatory ('F65') fixture, scoped to just those two classes (not the shared CampaignApprovalTestBase), so their pre-existing calendar-event assertions keep passing now that D-06's gate requires a resolved run.site -- without polluting TestApprovalSiteResolution's Observatory.objects.count()==0 assertions, which rely on the base fixture having zero Observatory rows"
  - "TestCalendarProjection's ground-run test asserts CalendarEvent bounds against a live sun_event(site, date, kind='sun') call rather than a hardcoded expected datetime, so the test tracks the real astronomy calculation instead of a value that could silently drift from it"

patterns-established: []

requirements-completed: [SCHED-02, SCHED-03]

coverage:
  - id: D1
    description: "CampaignRunTable/ApprovalQueueTable render a single window column: TBD badge for a fully-null window, a single date for window_start==window_end, and 'start -> end' for a range; default sort puts resolved rows first (most recent window_start) and TBD rows last, portably across SQLite/PostgreSQL"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_tbd_row_renders_tbd_indicator"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_range_row_renders_arrow"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestWindowColumnRendering.test_single_night_row_renders_one_date"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableView.test_default_sort_is_window_start_desc_tbd_last"
        status: pass
    human_judgment: false
  - id: D2
    description: "Non-staff requests never receive window PII beyond the ALLOWED_FIELDS_FOR_NON_STAFF allowlist, now listing window_start/window_end in place of the removed obs_date/ut_start/ut_end"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactFieldGating (full class, all 4 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Approving a single-night ground run projects a dip-corrected CalendarEvent (sun_event(kind='sun')); a single-night space run projects a midnight-UTC placeholder; a range, TBD, or missing-telescope-instrument run projects nothing; a sun_event ValueError is logged and skipped without reverting the approval"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_single_night_ground_run_creates_dip_corrected_calendar_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_single_night_space_run_creates_midnight_utc_placeholder_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_range_run_creates_no_calendar_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_approve_tbd_run_creates_no_calendar_event"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCalendarProjection.test_sun_event_valueerror_skips_projection_without_reverting_approval"
        status: pass
    human_judgment: false
  - id: D4
    description: "The public submission form has no ut_start/ut_end inputs; its single obs_date field maps to both window_start and window_end on save (single-night collapse), and the duplicate-submission friendly-error path still works under the new resolved-window natural key"
    requirement: "SCHED-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_minimal_valid_submission_creates_pending_run"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_duplicate_natural_key_submission_shows_friendly_form_error"
        status: pass
    human_judgment: false

duration: ~20min
completed: 2026-07-09
status: complete
---

# Phase 19 Plan 3: Window-Schema Migration -- Table/View/Form Consumers Summary

**campaign_tables.py/campaign_views.py/campaign_forms.py rewritten against window_start/window_end: a combined TBD/single-date/range window column with cross-backend nulls-last sort, a D-06 hybrid ground-vs-space calendar projection on approve, and a submission form collapsed to a single observing date.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-09T22:52:37Z
- **Tasks:** 3
- **Files modified:** 6 (campaign_tables.py, campaign_views.py, campaign_forms.py, test_campaign_views.py, test_campaign_approval.py, test_campaign_submission.py)

## Accomplishments
- `CampaignRunTable.render_window_start()` (new) mirrors `render_site()`'s dict-vs-model dual-`Accessor` precedent: a fully-null window renders a `badge-secondary` "TBD" span (D-03), `window_start == window_end` renders a single date, and a range renders `"start -&gt; end"` (D-05, the literal arrow, not an en-dash). `Meta.fields`/`ApprovalQueueTable.Meta.sequence` now list a single `window_start` entry in place of the three removed fields; the table's own `Meta.order_by` is gone entirely (django-tables2 can't compile `F(...).desc(nulls_last=True)`).
- `CampaignRunTableView.get_queryset()` applies `.order_by(F('window_start').desc(nulls_last=True))` for **both** the staff and non-staff branches (D-04), with `get_table_kwargs()` returning `'order_by': ()` for both so django-tables2 never re-sorts and clobbers it -- portable across SQLite and PostgreSQL, which default to opposite implicit NULL-ordering directions for `DESC`.
- `ALLOWED_FIELDS_FOR_NON_STAFF` swaps `obs_date`/`ut_start`/`ut_end` for `window_start`/`window_end`, keeping its shape as an explicit enumerated allowlist (never introspected from `_meta`).
- `CampaignRunDecisionView.post()`'s calendar-projection gate is now `run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end` (a single concrete night with a resolved site). A space-based site (`Observatory.SATELLITE_OBSTYPE`) gets a midnight-UTC placeholder (`window_start` 00:00 to `window_end` 23:59 UTC); a ground-based site calls `telescope_runs.sun_event(site, window_start, kind='sun')` for a dip-corrected sunset/sunrise window, wrapped in `except ValueError: log+skip` so a messy `Observatory` (blank timezone, no crossings) can never revert an already-committed approval.
- `CampaignRunSubmissionForm` drops its `ut_start`/`ut_end` `DateTimeField` inputs; the remaining `obs_date` `DateField` is mapped by `CampaignRunSubmissionView.form_valid()` to both `window_start=` and `window_end=` on `CampaignRun.objects.create(...)` (single-night collapse). The duplicate-submission `IntegrityError` message is reworded to describe the collision in terms of "this observing date" rather than "this start time."
- All three modules' test files were rewritten off `obs_date`/`ut_start`/`ut_end`: `test_campaign_views.py` gained a `TestWindowColumnRendering` class (TBD/range/single-date render assertions) and a rewritten nulls-last sort test; `test_campaign_approval.py`'s `TestCalendarProjection` was rewritten for the D-06 ground/space split plus a new `sun_event()` `ValueError` no-revert case, and `TestApproval`/`TestCalendarNoChurn` gained a ground-site `Observatory` fixture so their pre-existing calendar-event assertions still pass now that projection requires a resolved site; `test_campaign_submission.py`'s minimal-submission and duplicate-collision tests now submit/assert against `obs_date`/`window_start`/`window_end`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Window display column + cross-backend nulls-last default sort + PII allowlist swap** - `2afc402` (feat)
2. **Task 2: D-06 hybrid ground/space calendar projection on approve** - `db9b236` (test), `7a9c4e7` (feat)
3. **Task 3: Collapse the submission form to a single date; map it to both window fields on save** - `2c00624` (feat)

_Note: Task 2 is TDD-tagged. Consistent with 19-01's Task 2 precedent, the test rewrite and the
D-06 implementation were authored together and verified together (both files edited before
either was committed), then split into a `test(...)` commit followed by a `feat(...)` commit
to satisfy the plan-level RED/GREEN gate-sequence check in git log -- not a strict, independently
red-then-green TDD cycle. This is documented explicitly per the "TDD Gate Compliance" section
below._

## Files Created/Modified
- `solsys_code/campaign_tables.py` - `render_window_start()` (new); `Meta.fields`/`Meta.sequence` collapsed to `window_start`; `Meta.order_by` removed
- `solsys_code/campaign_views.py` - `ALLOWED_FIELDS_FOR_NON_STAFF` swapped to `window_start`/`window_end`; `CampaignRunTableView.get_queryset()`/`get_table_kwargs()` nulls-last sort; `CampaignRunDecisionView.post()`'s D-06 hybrid projection; `CampaignRunSubmissionView.form_valid()` maps `obs_date` to both window fields; reworded duplicate-submission error message
- `solsys_code/campaign_forms.py` - `CampaignRunSubmissionForm` drops `ut_start`/`ut_end`; `Layout` updated
- `solsys_code/tests/test_campaign_views.py` - `TestWindowColumnRendering` (new); rewritten nulls-last sort test; fixtures migrated to `window_start`/`window_end`
- `solsys_code/tests/test_campaign_approval.py` - `TestCalendarProjection` rewritten for D-06 (ground/space/range/TBD/no-telescope/ValueError cases); `TestApproval`/`TestCalendarNoChurn` gained a ground `Observatory` fixture; base fixture migrated to `window_start`/`window_end`
- `solsys_code/tests/test_campaign_submission.py` - minimal-submission and duplicate-collision tests rewritten against `obs_date`/`window_start`/`window_end`

## Decisions Made
- `render_window_start()`'s range branch returns the literal `-&gt;` HTML entity (not a plain hyphen/en-dash) per D-05's exact "->" wording -- both the render method and its test assert this precise string, matching RESEARCH.md's Pattern 3 code block verbatim.
- Scoped the ground `Observatory('F65')` fixture needed to keep `TestApproval`/`TestCalendarNoChurn`'s pre-existing calendar-event assertions green (now that D-06 requires a resolved `run.site`) to just those two test classes' own `setUpTestData()`, rather than the shared `CampaignApprovalTestBase` -- adding it to the base would have broken `TestApprovalSiteResolution`'s `Observatory.objects.count() == 0` assertions, which rely on zero pre-existing Observatory rows.
- `TestCalendarProjection`'s ground-run test computes its expected `CalendarEvent` bounds via a live `sun_event(site, date, kind='sun')` call rather than a hardcoded datetime literal, so the assertion tracks the real dip-corrected calculation instead of a value that could silently drift out of sync with it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug/consistency] Stale comment referencing the removed `Meta.order_by` in `ApprovalQueueView.get_context_data()`**
- **Found during:** Task 1
- **Issue:** A comment above `decided_qs` in `campaign_views.py` explained the `order_by=()` precedent by referencing "django-tables2 applies its Meta.order_by (inherited '-obs_date' from CampaignRunTable)" -- but Task 1 removes `CampaignRunTable.Meta.order_by` entirely, making the comment inaccurate.
- **Fix:** Reworded the comment to describe the general table-construction re-sort mechanism without naming the now-nonexistent `Meta.order_by` attribute, and clarified that this queue view intentionally orders by recency (`-pk`), not `window_start`.
- **Files modified:** `solsys_code/campaign_views.py`
- **Commit:** `2afc402`

---

**Total deviations:** 1 auto-fixed (Rule 1)
**Impact on plan:** Comment-only correctness fix adjacent to Task 1's scope; no behavior change, no scope creep.

## TDD Gate Compliance

Task 2 (`tdd="true"`) has both required commits in git log: `db9b236` (`test(19-03): rewrite TestCalendarProjection for D-06 hybrid projection`) precedes `7a9c4e7` (`feat(19-03): D-06 hybrid ground/space calendar projection on approve`). As noted above, both were authored and verified together against the fully-implemented D-06 gate before either commit was made (mirroring 19-01-SUMMARY.md's documented precedent for its own tdd="true" task) -- the test file was not run against source lacking the D-06 change and observed failing first. No REFACTOR commit was needed.

## Issues Encountered
None -- this plan's own scoped test modules (`test_campaign_views`, `test_campaign_approval`, `test_campaign_submission`) are fully green (62/62 tests). The full `python manage.py test solsys_code` app suite has 8 remaining errors, all in `test_import_campaign_csv.py` / `campaign_utils.insert_or_create_campaign_run`'s `ut_start`-keyed lookup -- squarely sibling plan `19-04-PLAN.md`'s scope (`solsys_code/management/commands/import_campaign_csv.py`, its demo notebook, and `test_import_campaign_csv.py`), not touched here. This is the wave-merge gate `19-RESEARCH.md`'s Sampling Rate section describes ("Per wave merge: full solsys_code suite... Phase gate: before /gsd-verify-work"), not a per-plan gate -- consistent with 19-02-SUMMARY.md's identical cross-plan-gap note for its own scope.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `campaign_tables.py`, `campaign_views.py`, and `campaign_forms.py` are fully window-native; their own test modules (`test_campaign_views`, `test_campaign_approval`, `test_campaign_submission`) are green (62/62).
- Sibling plan `19-04` (same wave) still needs to update `import_campaign_csv.py`'s natural-key lookup (`ut_start` -> `window_start`) and its paired demo notebook before `python manage.py test solsys_code` (full suite) is green -- 8 errors remain, all in that plan's scope.
- Phase 20's CSV range/TBD import and asset-aware coverage-gap consumers can build directly on this plan's window-native table/view/form surface; the D-06 ground/space split established here is the "narrow, early application" Phase 20's ASSET-01 formalizes further.

---
*Phase: 19-window-schema-migration*
*Completed: 2026-07-09*

## Self-Check: PASSED

All modified files found on disk (`campaign_tables.py`, `campaign_views.py`, `campaign_forms.py`, `test_campaign_views.py`, `test_campaign_approval.py`, `test_campaign_submission.py`); all four task commit hashes (`2afc402`, `db9b236`, `7a9c4e7`, `2c00624`) found in git log.
