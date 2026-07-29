# Phase 19: Window-Schema Migration - Context

**Gathered:** 2026-07-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace `CampaignRun`'s single-night `obs_date`/`ut_start`/`ut_end` representation
with a nullable `window_start`/`window_end` `DateField` pair (locked by Phase 18's
spike), migrating every existing row with no data loss. This is the largest-blast-
radius change in the v2.1 milestone — it touches every consumer of these fields
before Phase 20 (CSV range/TBD import, asset-aware coverage gap) or Phase 21 (site
disambiguation UI) can build on the new schema.

**Requirements:** SCHED-02, SCHED-03, SCHED-04, SCHED-05 (see `.planning/REQUIREMENTS.md`)

**In scope:** schema migration (add window fields, backfill, drop old fields),
updating all direct consumers so they keep working against the new fields
(per-campaign table, approval queue, coverage-gap page, CSV import, calendar
projection), and closing the TBD natural-key duplicate-row gap.

**Out of scope (belongs to later phases):** actually importing range/TBD CSV rows
(Phase 20 — `import_campaign_csv`/`parse_obs_window` still only produces
`window_start == window_end` rows in this phase); the full ground-vs-space-mission
asset-aware coverage-gap rewrite (Phase 20 — this phase only needs the same
ground/space distinction narrowly, for the calendar-projection decision below);
the fuzzy-match site-disambiguation UI (Phase 21).

</domain>

<decisions>
## Implementation Decisions

### Old-field retirement strategy
- **D-01:** Hard cutover — `obs_date`/`ut_start`/`ut_end` are dropped from
  `CampaignRun` in this same phase, not kept as a transitional dual-schema. All 15
  files that currently reference these fields (`models.py`, `campaign_forms.py`,
  `campaign_gap.py`, `campaign_views.py`, `campaign_utils.py`, `campaign_tables.py`,
  `management/commands/import_campaign_csv.py`, plus 6 test files:
  `test_campaign_approval.py`, `test_campaign_gap.py`, `test_campaign_views.py`,
  `test_campaign_submission.py`, `test_campaign_models.py`,
  `test_import_campaign_csv.py`) must be updated in this phase — there is no
  deferred cleanup phase for the old fields.
- **D-02:** The schema change ships as a **single combined migration** (add
  `window_start`/`window_end`, backfill from `obs_date`, drop the old fields and
  old `UniqueConstraint`, add the new one) — not split into separate
  add/backfill/drop migrations. User explicitly chose this over the
  split-into-reversible-steps option.

### TBD/window display convention
- **D-03:** A TBD row (both `window_start`/`window_end` null) should render as
  **"TBD" with a visual flag** (badge/icon) on the per-campaign table and approval
  queue — best effort. If a badge/icon proves complicated given no existing
  badge/icon convention in `campaign_tables.py`, falling back to plain "TBD" text
  is acceptable; the user explicitly said to drop the visual flag if it doesn't
  work out.
- **D-04:** TBD rows sort **last** — scheduled rows (most recent `window_start`
  first) lead the table, replacing today's default `order_by = ('-obs_date',)` in
  `campaign_tables.py`. TBD rows are the least-resolved, lowest-priority-to-display
  rows in this ordering.
- **D-05:** A range row (`window_start != window_end`) displays using **`->`**
  between the two dates — e.g. `"Aug 1, 2026 -> Aug 15, 2026"` — not an en-dash.
  A single-night row (`window_start == window_end`) still displays as one date. No
  range rows exist until Phase 20's CSV import lands, but the column-rendering
  logic for `CampaignRunTable`/`ApprovalQueueTable` is written now per D-01's hard
  cutover.

### Calendar projection during the gap
- **D-06:** `campaign_views.py`'s calendar-projection gate (currently
  `if run.telescope_instrument and run.ut_start and run.ut_end:`, assigning
  `ut_start`/`ut_end` directly as `CalendarEvent` `start_time`/`end_time`) is
  **hybrid** once `ut_start`/`ut_end` are gone:
  - **Ground-based observatories** (resolved `site` where
    `Observatory.observations_type != SATELLITE_OBSTYPE`): reuse Stage 1's
    `telescope_runs.sun_event()` for a real, dip-corrected dark-window banner —
    same accuracy convention the rest of the calendar feature already uses.
  - **Space-based observatories** (`Observatory.observations_type ==
    SATELLITE_OBSTYPE`): use a midnight-UTC placeholder (`window_start` 00:00 UTC
    to `window_end`/`window_start` 23:59 UTC) — `sun_event()` doesn't apply to a
    space telescope with no fixed horizon.
  - Only projects when `window_start == window_end` (a single concrete night) —
    ranges and TBD runs still don't get a `CalendarEvent`, matching today's
    `ut_end`-missing gate behavior.
  - **Note for planner/researcher:** this is a narrow, early application of the
    ground-vs-space distinction that Phase 20's ASSET-01 formalizes for coverage-
    gap analysis. Phase 19 only needs the `Observatory.observations_type` check
    for this one projection code path — it is NOT expected to build the full
    asset-aware `claimed_dates()` rewrite (that's Phase 20's job).

### Existing duplicate-row cleanup
- **D-07:** Live query against the dev DB (see Specific Ideas below) found 2 real
  pairs of fully-duplicate `CampaignRun` rows that would still collide under Phase
  18's contact_person-based partial-constraint recommendation. These are
  identified as leftover demo/UAT fixture rows, not genuine campaign data — **the
  migration deletes the duplicates** as part of its data-cleanup step, before
  applying the new partial `UniqueConstraint`.
- **D-08:** The data migration's de-dup step is **generic**, not hardcoded to the
  2 known pk pairs: it queries for ANY `(campaign, telescope_instrument,
  contact_person)` collision group among null-window rows, keeps one row per
  group (lowest pk), and logs what it removed. This makes the migration
  re-runnable and portable to another environment with different leftover data,
  rather than crashing with an `IntegrityError` on an untested collision the
  discussion didn't spot.

### Claude's Discretion
- Exact partial/conditional `UniqueConstraint` SQL mechanism (Django `condition=`
  syntax, portable across SQLite and PostgreSQL per SCHED-04) — a technical
  implementation choice for the planner/researcher to work out, not re-litigated
  here. The natural-key composition itself (`campaign`, `telescope_instrument`,
  `contact_person`, condition `window_start IS NULL`) is restated as locked by
  Phase 18's `18-DECISION.md`.
- Default table sort direction/format details beyond D-04/D-05 (e.g. exact date
  format string, badge/icon visual styling if pursued) — left to implementation
  to match existing `campaign_tables.py`/template conventions.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 18 spike decisions (locked inputs to this phase)
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` —
  window schema (nullable `window_start`/`window_end` `DateField` pair, criterion
  1), TBD natural-key recommendation (fold `contact_person` into a partial/
  conditional `UniqueConstraint`, criterion 2), real JWST TBD-collision evidence
- `docs/design/uncertain_scheduling_spike.rst` — durable summary of the same
  verdicts, for a quicker read than the full decision doc

### Current implementation (files this phase must change)
- `solsys_code/models.py` — `CampaignRun` model: `obs_date`/`ut_start`/`ut_end`
  fields (lines ~74-76) and `unique_campaign_run_natural_key` `UniqueConstraint`
  (`Meta.constraints`) to be replaced
- `solsys_code/campaign_utils.py` — `parse_obs_window()` (lines 186-244) and
  `insert_or_create_campaign_run` (uses the natural key)
- `solsys_code/campaign_gap.py` — `claimed_dates()` (lines ~152-201) currently
  reads `run.obs_date`/`run.ut_start` directly
- `solsys_code/campaign_views.py` — `ApprovalQueueView`/`CampaignRunDecisionView`
  calendar-projection block (line ~309: `if run.telescope_instrument and
  run.ut_start and run.ut_end:`, lines 318-319 assign `ut_start`/`ut_end` as
  `CalendarEvent` `start_time`/`end_time`); `ALLOWED_FIELDS_FOR_NON_STAFF` list
  (lines 55-57)
- `solsys_code/campaign_tables.py` — `CampaignRunTable`/`ApprovalQueueTable`
  column definitions (lines 58-60, 174-176) and `order_by = ('-obs_date',)`
  (line 73, tagged `# D-10`)
- `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm` fields
  `obs_date`/`ut_start`/`ut_end` (lines 24-26, 51-53)
- `solsys_code/management/commands/import_campaign_csv.py` — bootstrap CSV import,
  consumer of `parse_obs_window()`/`insert_or_create_campaign_run`
- `solsys_code/telescope_runs.py` — `sun_event()`, `get_site()` (Stage 1) reused
  for D-06's ground-based calendar projection
- 6 test files reference the old fields directly:
  `solsys_code/tests/test_campaign_approval.py`, `test_campaign_gap.py`,
  `test_campaign_views.py`, `test_campaign_submission.py`,
  `test_campaign_models.py`, `test_import_campaign_csv.py`

### Project-level conventions
- `CLAUDE.md` — Target test factories (`NonSiderealTargetFactory`), demo-notebook
  companion requirement (does not apply to `campaign_utils.py`/`import_campaign_csv.py`
  per the notebook list in CLAUDE.md — only `telescope_runs.py` and the 3 calendar-sync
  management commands have paired notebooks; `import_campaign_csv.py` already has its
  own separate demo notebook `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`
  from Phase 14 — check whether it needs updating for the field rename)
- `.planning/PROJECT.md` — v2.1 milestone goal, Phase 18 carried decisions section

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `telescope_runs.sun_event()` / `get_site()` (Stage 1) — reused directly for D-06's
  ground-based calendar-projection dark-window calculation; already the
  established convention for every other calendar-sync command.
- `Observatory.observations_type` (`SATELLITE_OBSTYPE` choice) — existing field,
  no schema change needed; D-06 reads it to branch ground vs. space projection.

### Established Patterns
- `parse_obs_window()`'s pattern-per-shape regex discipline (`_HHMM_RANGE`/
  `_APPROX_HOUR`/`_BARE_HOUR_UTC`) stays in place for Phase 20 to extend — Phase
  19 only needs `parse_obs_window()` to keep returning a single exact date
  (mapped to `window_start == window_end`), not to add range/TBD parsing itself.
- `campaign_gap.py`'s existing `unattributed_runs` bucketing pattern (runs that
  can't be attributed to any date are collected separately, never counted as
  claiming a date) should be preserved when `claimed_dates()` is rewritten against
  `window_start`/`window_end`.

### Integration Points
- `CampaignRunTable`/`ApprovalQueueTable` (`campaign_tables.py`) — column
  definitions and `order_by` are the integration point for D-03/D-04/D-05's
  display decisions.
- `CampaignRunDecisionView.post()` (`campaign_views.py`) — the calendar-projection
  gate is the integration point for D-06.
- The `Meta.constraints` list on `CampaignRun` — integration point for the new
  partial `UniqueConstraint` and D-07/D-08's de-dup data migration.

</code_context>

<specifics>
## Specific Ideas

**Live dev-DB evidence gathered during this discussion** (grounds D-07/D-08):
a direct query against the local dev DB (`CampaignRun.objects.filter(obs_date__isnull=True)`)
found 16 total `CampaignRun` rows, 6 with `obs_date` null, and among those, two
genuine duplicate pairs:
- pk 15 & 17: campaign 3 ("3I/ATLAS (demo)"), `telescope_instrument='Demo
  Telescope/DemoCam'`, `contact_person='Grace Lifecycle'`, all other fields
  null/blank/identical.
- pk 16 & 18: same campaign, `telescope_instrument='Demo Telescope/DemoSpec'`,
  `contact_person='Hal Lifecycle'`, same pattern.

These are almost certainly leftover fixture rows from a prior phase's manual UAT
checkpoint (naming convention "Grace Lifecycle"/"Hal Lifecycle" suggests deliberate
lifecycle-state test data, not real submissions) — hence D-07's decision to delete
rather than add a disambiguating field.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. The one weakly-matched pending todo
(`2026-06-23-...rename-calendar-utils-py-private-helpers...md`, score 0.6) was
reviewed and left deferred — see Reviewed Todos below.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` —
  "Extract site/telescope mapping and instrument extraction into own module"
  (renaming `calendar_utils.py`'s private helpers). Reviewed and explicitly left
  deferred: it's about the calendar-sync helpers (Stages 1-3), unrelated to
  `CampaignRun`/window schema, and already marked "deliberately deferred, no
  second consumer yet" in `.planning/STATE.md`.

</deferred>

---

*Phase: 19-window-schema-migration*
*Context gathered: 2026-07-09*
