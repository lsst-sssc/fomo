# Phase 20: Range/TBD Import & Asset-Aware Coverage Gap - Context

**Gathered:** 2026-07-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the Phase 19 window schema (`window_start`/`window_end`) usable end-to-end for
the real 3I/ATLAS sheet's harder rows:

1. **CSV import (IMPORT-01/02)** — `import_campaign_csv`/`parse_obs_window()` currently
   raises `ValueError` and skips any row whose `Obs. Date` isn't an exact `%Y-%m-%d`
   string. This phase extends parsing to real range shapes (`" to "`-separated,
   compact same-month, and now also cross-month/year rollover per discussion) and
   makes genuinely unparseable text (including the `YYYY-MM-?` marker and blank cells)
   import as a flagged TBD row instead of being skipped.
2. **Asset-aware coverage gap (ASSET-01/02)** — `campaign_gap.py`'s `claimed_dates()`
   currently claims every date in every approved run's window regardless of site type.
   This phase branches on `Observatory.observations_type == SATELLITE_OBSTYPE`
   (derived from the run's resolved `site`, no new `CampaignRun` field): ground-based
   runs keep claiming every date in their window (conservative); space-mission runs
   claim nothing until `window_start == window_end` (narrowed to one concrete night).

**Requirements:** IMPORT-01, IMPORT-02, ASSET-01, ASSET-02 (see `.planning/REQUIREMENTS.md`)

**In scope:** `parse_obs_window()` range/TBD parsing extension; a new
`original_obs_date_raw` + `window_needs_review` field pair on `CampaignRun` (and its
migration); `import_campaign_csv.py` catch-all TBD-import path and summary-counter
update; `claimed_dates()`'s ground-vs-space branch and new `pending_narrowing_runs`
bucket; `campaignrun_gap_analysis.html` messaging for that new bucket;
`render_window_start()`'s TBD badge gains a tooltip when `original_obs_date_raw` is set.

**Out of scope (belongs to later phases or was explicitly rejected):**
- Site-disambiguation UI, submitter contact opt-in (VIEW-05) — Phase 21.
- Fixing `resolve_site()`'s `TypeError` on space-observatory `null` longitude
  (Phase 18 spike finding) — flagged for Phase 19/21 awareness, not this phase's job
  unless it blocks ASSET-01's `Observatory.observations_type` check (it doesn't — that
  check only needs an already-resolved `site`, not a fresh `resolve_site()` call).
- An automated "narrowing" mechanism for space-mission runs (e.g. a background job) —
  narrowing is manual/re-import-only, matching the existing model (no new mechanism).
- A distinct model-level sub-status separating "blank Obs. Date" from "month-known,
  day-unknown" (`2025-12-?`) — both collapse to the same TBD state;
  `original_obs_date_raw` alone captures the distinction when needed.

</domain>

<decisions>
## Implementation Decisions

### TBD-row "needs review" persistence
- **D-01:** Add `original_obs_date_raw` (text field) to `CampaignRun`, populated only
  when a row lands as TBD via the new import path (empty/unset for successfully-parsed
  single-date or range rows) — preserves exactly what the sheet said (e.g. `"TBD
  pending Cycle 2"`, `"2025-12-?"`) for staff review.
- **D-02:** Add a dedicated `window_needs_review` boolean field alongside it, mirroring
  the existing `site_needs_review` pattern — set `True` whenever a row is created via
  the new TBD-import path. Supports future filtering the same way `site_needs_review`
  does; not just inferred from non-empty raw text.
- **D-03:** Both "blank `Obs. Date`" and "`YYYY-MM-?` month-known-day-TBD" collapse to
  the same TBD state (`window_start == window_end == None`) at the model level — no
  separate sub-status. `original_obs_date_raw` naturally distinguishes them (empty
  string vs. `"2025-12-?"`) without added model complexity.
- **D-04:** `original_obs_date_raw` is TBD-rows-only — not populated for
  successfully-parsed single-date or range rows, since those are already fully and
  faithfully represented by `window_start`/`window_end`.

### Import summary counters
- **D-05:** Only one new counter: `window_needs_review` (count of rows landing as TBD
  via the new import path). Range rows do **not** get their own counter — they count as
  ordinary `created`/`updated` rows, same as any successfully-parsed row; a range is not
  an exceptional outcome, just a wider window.
- **D-06:** Genuinely unparseable `Obs. Date` text — anything outside the enumerated
  shapes (blank / `" to "` range / compact range / `YYYY-MM-?` marker) — now **imports
  as a TBD row with `window_needs_review=True`**, never skipped. This directly satisfies
  IMPORT-02's "never silently dropped" wording and takes precedence over the Phase 18
  spike's narrower suggestion ("raise only on truly malformed values outside the
  enumerated shapes") — the roadmap criterion is the locked requirement here; the
  spike's suggestion was written before this phase's discussion resolved the open
  question in its favor.
- **D-07:** `skipped_count` is now reserved for non-date failures only — blank
  `Telescope / Instrument` and natural-key collisions (the existing `seen_window_keys`
  duplicate check). No `Obs. Date` outcome (exact, range, or TBD) contributes to
  `skipped_count` anymore, since every outcome now creates or updates a row.
- **D-08:** `original_obs_date_raw` is surfaced in the campaign table UI this phase —
  `CampaignRunTable.render_window_start()`'s existing TBD badge (`campaign_tables.py`)
  gains a `title` attribute showing `original_obs_date_raw` when present, mirroring
  `CalendarEventTelescopeLabel`'s hover-tooltip convention for verification detail.

### Space-mission gap-page bucketing
- **D-09:** `claimed_dates()` gets a **new, distinct return bucket** —
  `pending_narrowing_runs` — for space-mission runs whose window hasn't narrowed to one
  concrete night (still a range, or TBD). This is separate from the existing
  `undated_runs` bucket (which stays reserved for genuinely-TBD rows regardless of
  site type) so staff can tell "no info at all" apart from "a real space-mission run
  with a range, just not scheduled tight enough yet." `campaignrun_gap_analysis.html`
  needs a new alert block for this bucket (e.g. "N space-mission run(s) haven't
  narrowed to a specific night yet and aren't counted as claiming any date").
- **D-10:** No automated narrowing mechanism — a space-mission run only starts claiming
  a date when a staff edit or a future CSV re-import sets `window_start == window_end`
  directly. `claimed_dates()` picks this up naturally on its next computation; no new
  background job or explicit "narrow" UI action is in scope.
- **Note for planner/researcher:** `claimed_dates()`'s ground-vs-space branch reads
  `run.site.observations_type == Observatory.SATELLITE_OBSTYPE`
  (`solsys_code/solsys_code_observatory/models.py`) — the same check Phase 19's D-06
  already used inline for calendar-projection in `campaign_views.py`. Whether to extract
  a small shared helper (e.g. `is_space_mission(site)` in `campaign_utils.py`) or keep
  both call sites inline is left to implementation — not re-litigated here, no
  behavioral difference either way.

### Range-parsing edge-case scope
- **D-11:** The compact same-month range shape (`"2025-11-02 -25"`) is extended to
  detect month/year rollover: if the parsed second-day number is less than the first
  date's day-of-month, roll `window_end` into the next month (and next year for a
  December→January rollover), rather than restricting to same-month only as the Phase
  18 spike's worked example showed. No real sheet row confirms this exact shape yet —
  implement it as a generalization of the confirmed same-month rule, not a new
  from-scratch parser.
- **D-12:** The `" to "`-separated full-date range pattern also accepts en-dash- and
  hyphen-separated variants (e.g. `"2025-07-05–2025-09-22"`), not just the literal
  `" to "` string the spike evidenced — broadens the regex proactively rather than
  waiting for a future sheet edit to use a different separator.
- **D-13:** `parse_obs_window()`'s documented contract changes: it **no longer raises
  `ValueError` for `Obs. Date` failures**. It always returns a tuple now — on anything
  it can't parse into an exact date or a recognized range shape, it returns
  `window_start = window_end = None`, the raw text (for `original_obs_date_raw`), and
  `needs_review = True`. This is a deliberate, phase-scoped contract change (was: "raises
  ValueError... true natural-key failure per D-05") — any future caller of
  `parse_obs_window()`, not just `import_campaign_csv.py`, inherits the never-raise
  behavior. The existing UT-Time-Range-side "never raise, flag needs-review" fallback
  (`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`) is unchanged — this only affects the
  `Obs. Date` side.

### Claude's Discretion
- Exact regex/parsing implementation for month/year rollover (D-11) and the
  en-dash/hyphen separator variants (D-12) — technical detail for the
  researcher/planner, not re-litigated here.
- Whether to extract a shared `is_space_mission(site)` helper vs. keeping the
  `Observatory.observations_type == SATELLITE_OBSTYPE` check inline at each call site
  (see D-09's note) — no behavioral difference, implementation's call.
- `original_obs_date_raw` field type (`CharField` vs `TextField`) and max length, and
  the exact migration mechanics (Django migration file structure) — standard technical
  choices.
- Exact wording of the new `pending_narrowing_runs` alert block in
  `campaignrun_gap_analysis.html` beyond the substance captured in D-09 — match existing
  template tone/structure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 18 spike decisions (locked inputs to this phase)
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` —
  real CSV range/TBD cell shapes (criterion 3), the exact worked example for compact
  same-month range parsing, the "never raise on non-key fields, flag needs-review"
  discipline this phase extends to `Obs. Date`, and the `Observatory.observations_type`
  ground-vs-space confirmation (criterion 5). **Note:** this phase's D-06 decision
  (unparseable text imports as TBD, never skipped) is a deliberate resolution of an
  open question the spike left to "Phase 19/20's call" — it goes further than the
  spike's own narrower suggestion; do not treat the spike's suggestion as still binding
  where this CONTEXT.md's decisions differ.
- `docs/design/uncertain_scheduling_spike.rst` — durable summary of the same findings.

### Phase 19 (immediate predecessor — schema and consumers this phase extends)
- `.planning/phases/19-window-schema-migration/19-CONTEXT.md` — the `window_start`/
  `window_end` schema decisions, the TBD/range display conventions (D-03/D-04/D-05,
  already implemented in `campaign_tables.py`), and the narrow ground-vs-space
  calendar-projection check (D-06) this phase's `claimed_dates()` rewrite parallels.

### Roadmap and requirements
- `.planning/ROADMAP.md` — Phase 20 section: goal, success criteria, requirement IDs.
- `.planning/REQUIREMENTS.md` — IMPORT-01, IMPORT-02, ASSET-01, ASSET-02 full text.

### Current implementation (files this phase must change)
- `solsys_code/campaign_utils.py` — `parse_obs_window()` (lines 185-243): the function
  whose contract changes per D-13; extend its pattern-per-shape discipline
  (`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`) to `Obs. Date` itself.
- `solsys_code/campaign_gap.py` — `claimed_dates()` (lines 114-186): add the
  ground-vs-space branch and `pending_narrowing_runs` bucket (D-09); `_compute_gap()`
  and `get_or_compute_gap()` (cache-key/result shape) need to carry the new bucket
  through to the view.
- `solsys_code/management/commands/import_campaign_csv.py` — `Command.handle()`
  (lines 21-196): remove the `except ValueError: skip` path for `Obs. Date` failures
  (D-06/D-13), add `window_needs_review` counter (D-05), pass `original_obs_date_raw`/
  `window_needs_review` into the `fields` dict for `insert_or_create_campaign_run`.
- `solsys_code/models.py` — `CampaignRun` (lines 30-155): add `original_obs_date_raw`
  and `window_needs_review` fields (D-01/D-02); new migration.
- `solsys_code/campaign_tables.py` — `CampaignRunTable.render_window_start()`
  (lines 135-148): add `title` attribute with `original_obs_date_raw` when present
  (D-08).
- `solsys_code/campaign_views.py` — `CampaignGapAnalysisView`: needs to pass the new
  `pending_narrowing_runs` bucket into template context.
- `src/templates/campaigns/campaignrun_gap_analysis.html` — new alert block for
  `pending_narrowing_runs` (D-09), alongside the existing `undated_runs`/
  `unattributed_runs` alert.
- `solsys_code/solsys_code_observatory/models.py` — `Observatory.observations_type`/
  `SATELLITE_OBSTYPE` (line ~19-56): the field this phase's ground-vs-space branch
  reads; no change needed here, read-only dependency.

### Project-level conventions
- `CLAUDE.md` — Target test factories (`NonSiderealTargetFactory`); the demo-notebook
  companion requirement does **not** apply to `campaign_utils.py`/
  `import_campaign_csv.py` per CLAUDE.md's explicit notebook list (only
  `telescope_runs.py` and the 3 calendar-sync management commands are covered) — but
  `import_campaign_csv.py` has its own separate demo notebook
  (`docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`, from Phase 14) that
  should be checked for whether it needs updating to demonstrate the new range/TBD
  import behavior, since it's a paired deliverable for that command even though it
  isn't on CLAUDE.md's mandatory-sync list.
- `.planning/PROJECT.md` — v2.1 milestone goal, "Current Milestone" section's asset-type
  and CSV-import target-feature bullets.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `parse_obs_window()`'s existing pattern-per-shape regex discipline
  (`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`) — the established convention this
  phase extends to `Obs. Date`, not replaces.
- `Observatory.observations_type`/`SATELLITE_OBSTYPE` — existing field, already used by
  Phase 19's D-06 calendar-projection branch; this phase's ASSET-01 reuses the exact
  same check for `claimed_dates()`.
- `CampaignRunTable.render_window_start()`'s existing TBD badge
  (`<span class="badge badge-secondary">TBD</span>`) — D-08 extends it with a tooltip
  rather than building new display machinery.
- `site_needs_review` field/pattern on `CampaignRun` — direct precedent for the new
  `window_needs_review` field (D-02): same "boolean flag, never fabricate, flag for
  human review" discipline.

### Established Patterns
- "Never raise on a non-key/best-effort field, flag needs-review instead" — already the
  posture for `UT Time Range` parsing; D-13 extends this same posture to `Obs. Date`
  itself, a deliberate widening of what counts as "non-key."
- `claimed_dates()`'s existing `undated_runs`/`unattributed_runs` bucketing pattern (a
  list, never counted as claiming, returned alongside the claimed-dates set) — D-09's
  `pending_narrowing_runs` follows the identical shape, just a third bucket with a
  different membership rule.
- `import_campaign_csv.py`'s skip-and-log discipline for genuine natural-key failures
  (blank `Telescope / Instrument`, natural-key collision) — stays intact; only the
  `Obs. Date`-parsing-failure branch of what used to trigger a skip is removed (D-06/D-07).

### Integration Points
- `CampaignGapAnalysisView`/`get_or_compute_gap()` (`campaign_gap.py`/`campaign_views.py`)
  — the cached gap-computation result shape gains a new key for
  `pending_narrowing_runs`; template context and `campaignrun_gap_analysis.html` both
  need updating in step.
- `insert_or_create_campaign_run` (`campaign_utils.py`) — the natural-key lookup dict
  `import_campaign_csv.py` builds needs to keep working for TBD rows created via the
  new catch-all path, using the existing TBD natural key
  (`campaign`, `telescope_instrument`, `contact_person`, condition
  `window_start IS NULL`) from Phase 19's `Meta.constraints`.

</code_context>

<specifics>
## Specific Ideas

No specific UI mockups or reference examples beyond what's captured in the decisions
above (D-08's tooltip mirrors the existing `CalendarEventTelescopeLabel` hover-tooltip
convention; D-09's new alert block mirrors the existing `undated_runs`/
`unattributed_runs` alert in `campaignrun_gap_analysis.html`).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 20's IMPORT-01/02/ASSET-01/02 scope. Site
disambiguation and VIEW-05 remain explicitly Phase 21 (already scoped there by
ROADMAP.md).

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — weak
  match (score 0.2) from the automated todo cross-reference. Same assessment as Phase
  19's CONTEXT.md: about `calendar_utils.py`'s private helpers (Stages 1-3 calendar
  sync), unrelated to `CampaignRun`/window schema/coverage-gap work. Left deferred.

</deferred>

---

*Phase: 20-range-tbd-import-asset-aware-coverage-gap*
*Context gathered: 2026-07-10*
