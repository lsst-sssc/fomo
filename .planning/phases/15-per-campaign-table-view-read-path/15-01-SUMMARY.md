---
phase: 15-per-campaign-table-view-read-path
plan: 01
subsystem: ui
tags: [django-tables2, django-filter, django, pii-gating, campaign-run]

requires:
  - phase: 14-campaign-data-model-bootstrap-import
    provides: CampaignRun model (ApprovalStatus/RunStatus TextChoices), TargetList-as-campaign convention
provides:
  - Per-campaign CampaignRun table (sortable/paginated/filterable) at /campaigns/<pk>/
  - Campaigns list page at /campaigns/
  - PII-gated queryset (contact_person/contact_email excluded from SQL for non-staff)
affects: [16-submission-form-approval-queue-calendar-projection]

tech-stack:
  added: []
  patterns:
    - "django-tables2 SingleTableMixin + django-filter FilterView composition (first consumer in this codebase)"
    - "View-layer PII gating via restricted .values() queryset, not template-only hiding"
    - "django_tables2.utils.Accessor used inside render_ methods to resolve raw field values directly from record, bypassing django-tables2's row-level pre-processing (dict-vs-model-instance-safe rendering)"

key-files:
  created:
    - solsys_code/campaign_tables.py
    - solsys_code/campaign_filters.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_urls.py
    - src/templates/campaigns/campaign_list.html
    - src/templates/campaigns/campaignrun_table.html
    - solsys_code/tests/test_campaign_views.py
  modified:
    - src/fomo/urls.py

key-decisions:
  - "render_run_status/render_approval_status resolve the raw field value via Accessor(record) rather than accepting django-tables2's `value` kwarg, since django-tables2 auto-calls get_FOO_display() for model-instance (staff) rows on choice fields before invoking a custom render_ method -- the opposite of what 15-RESEARCH.md Pitfall 2 assumed."
  - "open_to_collaboration filter's NullBooleanSelect widget is left untouched in campaign_filters.py (per Task 2's explicit instruction to leave it to Meta.fields auto-generation); the UI-SPEC 'Unknown'->'Any' copy relabel is done purely in the template via a manually-rendered <select> using the same field name/values, so no FilterSet declaration was needed."

patterns-established:
  - "Pattern: PII-sensitive fields gated by enumerating an explicit ALLOWED_FIELDS_FOR_NON_STAFF list and calling .values(*list) in get_queryset(), never by Meta.exclude alone."

requirements-completed: [VIEW-01, VIEW-03, VIEW-04]

coverage:
  - id: D1
    description: "Anonymous GET /campaigns/<pk>/ lists all CampaignRun rows for that campaign, 25/page, default-sorted obs_date descending"
    requirement: "VIEW-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunTableView"
        status: pass
    human_judgment: false
  - id: D2
    description: "contact_person/contact_email are excluded from SQL SELECT and response.context for anonymous/non-staff requests; present for staff"
    requirement: "VIEW-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestContactFieldGating"
        status: pass
    human_judgment: false
  - id: D3
    description: "run_status multi-select filter (OR semantics) and open_to_collaboration boolean filter narrow rows; unfiltered default shows every row"
    requirement: "VIEW-04"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunFilterSet"
        status: pass
    human_judgment: false
  - id: D4
    description: "Campaigns list page (/campaigns/) lists only TargetLists with >= 1 CampaignRun (D-03)"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignListView"
        status: pass
    human_judgment: false
  - id: D5
    description: "approval_status/run_status render as Bootstrap badges (colored/muted per D-08 and UI-SPEC badge contracts), identically for staff (model-instance) and anonymous (.values() dict) rows"
    verification:
      - kind: manual_procedural
        ref: "Manual smoke test via Django test client against dev DB: response content contains badge markup for both anonymous and staff GET of /campaigns/<pk>/"
        status: pass
    human_judgment: true
    rationale: "Visual badge color/contrast is a UI-SPEC contract item; full visual sign-off deferred to a future UI-review pass, not part of this plan's automated test suite."

duration: 25min
completed: 2026-07-03
status: complete
---

# Phase 15 Plan 01: Per-Campaign Table View (Read Path) Summary

**Anonymous-accessible, PII-gated `django-tables2`/`django-filter` table listing every `CampaignRun` for a campaign, plus a campaigns list page — first real consumer of both libraries in FOMO.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-03T15:58:23Z
- **Completed:** 2026-07-03T16:00:00Z (approx, includes verification)
- **Tasks:** 3
- **Files modified:** 8 (7 created, 1 modified)

## Accomplishments
- `CampaignRunTable` (django-tables2): D-09 spreadsheet-parity 16-column set, `-obs_date` default sort (D-10), Bootstrap4-responsive template, manual `render_run_status`/`render_approval_status`/`render_site`/`render_open_to_collaboration` methods that work identically for staff (model instance) and anonymous (dict) rows
- `CampaignRunFilterSet` (django-filter): `run_status` explicitly declared as `MultipleChoiceFilter` with `CheckboxSelectMultiple` widget for D-12 OR-semantics multi-select; `open_to_collaboration` left to correct auto-generated `BooleanFilter`
- `CampaignRunTableView(SingleTableMixin, FilterView)`: `get_queryset()` returns a `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` restricted queryset for non-staff so `contact_person`/`contact_email` are never fetched by SQL (D-13/VIEW-03/T-15-01); `get_table_kwargs()` adds a redundant `exclude=` as defense in depth
- `CampaignListView(ListView)`: lists only `TargetList`s with `campaign_runs__isnull=False` (Pitfall 3), never `.objects.all()`
- `campaigns` URL namespace wired into `src/fomo/urls.py` before the `tom_common.urls` catch-all
- Two templates (`campaign_list.html`, `campaignrun_table.html`) implementing the UI-SPEC Layout/Copywriting Contracts (filter panel, empty states, badges)
- `test_campaign_views.py`: 13 tests across 4 classes (`TestCampaignRunTableView`, `TestContactFieldGating`, `TestCampaignRunFilterSet`, `TestCampaignListView`) proving VIEW-01/03/04 and D-03; RED confirmed before implementation, GREEN after

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test module + fixtures + failing tests for VIEW-01/03/04** - `d4c9f84` (test)
2. **Task 2: CampaignRunTable (badges, columns) + CampaignRunFilterSet (multi-select)** - `73afcfe` (feat)
3. **Task 3: CampaignRunTableView + CampaignListView + urls + templates (PII-gated)** - `5229a7c` (feat, includes a Rule 1 fix to campaign_tables.py found while wiring the view)

**Plan metadata:** (this commit, following SUMMARY.md write)

## Files Created/Modified
- `solsys_code/tests/test_campaign_views.py` - 30-row CampaignRun fixture, is_staff=True User fixture, 13 tests across 4 classes
- `solsys_code/campaign_tables.py` - `CampaignRunTable`, `APPROVAL_BADGE_CLASSES`, `RUN_STATUS_BADGE_CLASSES`
- `solsys_code/campaign_filters.py` - `CampaignRunFilterSet`
- `solsys_code/campaign_views.py` - `CampaignRunTableView`, `CampaignListView`, `ALLOWED_FIELDS_FOR_NON_STAFF`
- `solsys_code/campaign_urls.py` - `campaigns` namespace (`list`, `table` routes)
- `src/fomo/urls.py` - added `campaigns/` include before the `tom_common.urls` catch-all
- `src/templates/campaigns/campaign_list.html` - D-03 campaigns list page
- `src/templates/campaigns/campaignrun_table.html` - filter panel + `{% render_table table %}`

## Decisions Made
- `render_run_status`/`render_approval_status` resolve the raw stored value via `django_tables2.utils.Accessor('run_status'/'approval_status').resolve(record)` rather than trusting django-tables2's injected `value` argument, because django-tables2 auto-invokes `get_FOO_display()` on model-instance rows for fields with `choices` *before* calling a custom `render_` method — handing staff requests an already-humanized label ("Planned") while anonymous `.values()` requests get the raw code ("planned"). This directly contradicts 15-RESEARCH.md Pitfall 2's stated assumption that `value` is "already the resolved raw string in both cases"; empirically verified false for this django-tables2/Django version combination (found via the GREEN-state test run for Task 3, which raised `ValueError: 'Planned' is not a valid CampaignRun.RunStatus`).
- The `open_to_collaboration` filter's "Unknown"→"Any" copy relabel (UI-SPEC Copywriting Contract) is implemented entirely in `campaignrun_table.html` via a manually-rendered `<select>` reusing the filter form's field name/values — `campaign_filters.py` itself is untouched, per Task 2's explicit instruction to leave that field to `Meta.fields` auto-generation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] render_run_status/render_approval_status received an already-humanized label for staff rows, breaking the RunStatus/ApprovalStatus TextChoices lookup**
- **Found during:** Task 3 (GREEN-state test run)
- **Issue:** `CampaignRun.RunStatus(value).label` raised `ValueError: 'Planned' is not a valid CampaignRun.RunStatus` for staff (model-instance) rows — django-tables2's row-rendering machinery (`django_tables2/rows.py::_get_and_render_with`) auto-calls `record.get_run_status_display()` for model-instance rows on fields with `choices`, before handing `value` to the custom `render_` method, so `value` was `"Planned"` (label) for staff and `"planned"` (raw code) for anonymous `.values()` rows — an inconsistency 15-RESEARCH.md's Pitfall 2 did not anticipate.
- **Fix:** Changed both methods to accept `record` and resolve the raw value directly via `Accessor('run_status'/'approval_status').resolve(record, quiet=True)`, bypassing django-tables2's pre-processed `value` entirely.
- **Files modified:** `solsys_code/campaign_tables.py`
- **Verification:** All 13 `test_campaign_views` tests pass (including staff-path badge rendering); full `solsys_code` suite (255 tests) green
- **Committed in:** `5229a7c` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for the staff-facing table to render at all (would have 500'd on every staff GET). No scope creep — confined to the two render methods already scoped by Task 2.

## Issues Encountered
None beyond the deviation documented above.

## Known Stubs
None — VIEW-05 (submitter contact opt-in / reach-out path for `open_to_collaboration` runs) is explicitly out of scope for this phase (D-14) and no stub/placeholder was added in its place; the column is surfaced as filterable/visible data only, as specified.

## Threat Flags
None — the threat model's single high-severity item (T-15-01, contact PII exposure) is mitigated exactly as planned and proven by `TestContactFieldGating`; no new unmodeled surface was introduced.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Read path (VIEW-01/03/04) is complete and green; Plan 02 of this phase (VIEW-02 — target-detail integration + navbar entry) can proceed independently since it only extends `apps.py`/`solsys_code_extras.py`, neither of which this plan touched.
- Phase 16 (submission form + approval queue) can build directly on `CampaignRunTableView`/`CampaignRunTable` without modification — per D-05, this phase intentionally does not filter by `approval_status`, so Phase 16 must not assume otherwise.
- No blockers.

---
*Phase: 15-per-campaign-table-view-read-path*
*Completed: 2026-07-03*
