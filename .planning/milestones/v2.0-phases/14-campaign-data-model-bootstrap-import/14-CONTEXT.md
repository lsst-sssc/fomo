# Phase 14: Campaign Data Model & Bootstrap Import - Context

**Gathered:** 2026-07-02
**Status:** Ready for planning

<domain>
## Phase Boundary

A `CampaignRun` model — linked to a campaign `TargetList` (required FK), carrying an optional
observed `Target`, the full 3I-sheet field inventory, and a two-field status (`approval_status` +
`run_status`) — plus a management command (`import_campaign_csv` or similar) that bootstrap-imports
the real 3I/ATLAS coordination-sheet CSV into it, idempotently, with skip-and-log error handling and
a created/updated/skipped summary. The paired demo notebook exercises the command against a
synthetic, PII-free fixture.

Out of scope for this phase (belongs to later phases): the per-campaign table view (Phase 15), the
community submission form and approval-queue UI (Phase 16), calendar projection (Phase 16), and
coverage-gap analysis (Phase 17).

**Note on sequencing:** Phase 15 (per-campaign table view) was discussed first in this session; the
operator chose to stop and discuss/plan/build Phase 14 first since Phase 15 depends on the model
defined here.

</domain>

<decisions>
## Implementation Decisions

### Status vocabulary (CAMP-03)
- **D-01:** CAMP-03's status applies **per-`CampaignRun`** (one telescope run), not per-campaign —
  `TargetList` (the campaign) has no status field in this milestone; it's just the container.
- **D-02:** Split into **two fields**, not one flat combined vocabulary:
  - `approval_status` — FOMO admin gatekeeping of the record itself (`pending_review`, `approved`,
    `rejected`). Becomes operationally relevant once Phase 16's submission form exists, but the field
    is defined now.
  - `run_status` — real-world state of the observation itself: `requested` → `planned` → `observed`
    → `reduced` → `published`, plus three distinct terminal/dead-end values: `cancelled`,
    `not_awarded`, `weather_tech_failure`. Eight values total.
  - Rationale for the split: a flat single vocabulary can't represent "a DDT/proposal request whose
    outcome is still pending" independently of admin review state — the operator specifically flagged
    this gap.
- **D-03:** Bootstrap-imported rows (real historical 3I/ATLAS data) get `approval_status='approved'`
  — they're vetted historical data being backfilled, not fresh submissions awaiting review. The full
  `pending_review` → `approved`/`rejected` lifecycle is demonstrated in the demo notebook using
  synthetic data, not via the real import.

### CSV row identity & re-import (CAMP-04)
- **D-04:** Natural key for create-or-update idempotency: **(campaign, telescope, obs date/UT
  start)**. Mirrors the existing `CalendarEvent` find-or-create key pattern (telescope, instrument,
  start_time) already used by `load_telescope_runs`/`sync_lco_observation_calendar`.
- **D-05:** Row-skip granularity: only a failure in one of the **natural-key fields** (campaign,
  telescope, obs date/UT start) skips the whole row (and is logged, per CAMP-04). A malformed
  **non-key** field (e.g. filter, weather, comments) nulls just that column — the row still imports.

### Campaign/target/site resolution
- **D-06:** The import command takes a **required `--campaign` CLI argument**; the `TargetList` is
  found-or-created by name. One CSV run = one campaign (matches the real sheet's shape — there's no
  per-row campaign column in the source data).
- **D-07:** `CampaignRun`'s optional `Target` FK is **auto-resolved**: if the campaign `TargetList`
  has exactly one `Target`, every imported row gets that `Target` assigned automatically (3I/ATLAS is
  single-target).
- **D-08:** `CampaignRun.site` uses a **3-tier resolution** against the `Observatory` model, since
  most 3I-sheet site strings (Palomar P200, VLT/MUSE, etc.) are outside FOMO's existing Observatory
  registry:
  1. Match against existing `Observatory` records.
  2. If not found, query the MPC Obscodes API (the same one `MPCObscodeFetcher`/`CreateObservatory`
     already uses) and create an `Observatory` row if a match is found there.
  3. If still not found, create a **placeholder** `Observatory` row and flag for review.
- **D-09:** Because `site` is not part of the row's natural key, a failed/partial site resolution must
  not skip the row. Storage shape: `CampaignRun.site` (FK to `Observatory`, nullable) +
  `CampaignRun.site_raw` (text, preserves the original sheet string even after resolution) +
  `CampaignRun.site_needs_review` (bool) — mirrors the existing
  `CalendarEventTelescopeLabel.is_verified` sidecar precedent (flag first, don't silently guess).

### PII fixture strategy (CAMP-05)
- **D-10:** Demo-notebook fixture is a **small hand-built synthetic CSV**, same column shape as the
  real sheet, with obviously-fake contact info (e.g. `test@example.com`). ~5-10 rows covering field
  variety: multi-band imaging, spectroscopy, a `not_awarded`/`cancelled` row, and a site needing
  resolution.
- **D-11:** Fixture lives at `docs/notebooks/pre_executed/fixtures/` (colocated with the notebook per
  existing convention). Fixture rows use **only sites already seeded in the local `Observatory`
  table** — the notebook does not make a live MPC API call. This avoids a network dependency inside a
  committed, pre-executed notebook (flaky re-execution, MPC API downtime breaking
  `jupyter nbconvert --execute`). Tier-2 (MPC API lookup) and tier-3 (placeholder creation) resolution
  logic from D-08 is covered by the Django test suite with mocked API responses instead — not
  demonstrated live in the notebook.

### Claude's Discretion
- Exact management command name (e.g. `import_campaign_csv`) — not locked by the user, follow the
  `load_telescope_runs`/`fetch_jplsbdb_objects` naming convention.
- Exact CSV column-name-to-model-field mapping and date/time parsing strategy for the real sheet's
  free-text columns — research/plan should account for messy real-world data (mixed formats,
  multi-band imaging rows) per the `SITE_TELESCOPE_MAP`/v1.2→v1.3 "validate against reality" lesson
  already learned in this codebase.
- Whether `site_needs_review` rows get a distinct counter in the command's created/updated/skipped
  summary output (recommended, following the Phase 7 `[UNVERIFIED]`-style counter precedent, but not
  explicitly asked).

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — proposed extracting
  `SITE_TELESCOPE_MAP`/instrument-extraction logic out of `sync_lco_observation_calendar.py`. Reviewed
  and **not folded**: the operator confirmed this extraction already happened
  (`solsys_code/calendar_utils.py` already holds `SITE_TELESCOPE_MAP`, `_derive_telescope`,
  `_extract_instrument`). The todo file itself is now stale and should probably be deleted/closed
  outside this phase's scope. It was also a different problem from Phase 14's site resolution anyway
  (LCO-only telescope-class labels vs. free-text site/telescope parsing across many non-LCO
  facilities).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 14: Campaign Data Model & Bootstrap Import" — goal, success criteria
- `.planning/REQUIREMENTS.md` §"Campaign Data Model" (CAMP-01..05) — full requirement text

### Research
- `.planning/research/SUMMARY.md` — overall v2.0 approach, no-new-dependencies confirmation,
  PII/messy-CSV risk framing
- `.planning/research/FEATURES.md` — `CampaignRun` field inventory rationale, table-stakes vs.
  differentiator framing
- `.planning/seeds/target-linked-run-submission-form.md` — original seed + 2026-07-02 enrichment
  against the real 3I/ATLAS sheet; full field inventory and PII framing

### Existing code precedent
- `solsys_code/models.py` — `CalendarEventTelescopeLabel` sidecar-model precedent (verified/fallback
  flag pattern), reused for D-09's `site_needs_review`
- `solsys_code/calendar_utils.py` — `SITE_TELESCOPE_MAP`, `_derive_telescope`, `_extract_instrument`
  (already-extracted site/telescope resolution logic; confirmed by operator as the outcome of the
  reviewed-but-not-folded todo above)
- `solsys_code/management/commands/load_telescope_runs.py` — CSV/file ingest command structure,
  create-or-update find-or-create key pattern, per-line skip-and-log error handling
- `solsys_code/solsys_code_observatory/models.py` (`Observatory`) and `solsys_code/
  solsys_code_observatory/utils.py` (`MPCObscodeFetcher`) — site resolution tiers 1 and 2 (D-08)
- `solsys_code/solsys_code_observatory/views.py` (`CreateObservatory.form_valid`) — existing live
  pattern for MPC Obscodes API lookup → `Observatory.from_parallax_constants()` → save

CLAUDE.md's demo-notebook-companion convention applies: this phase's management command needs a
paired demo notebook under `docs/notebooks/pre_executed/` (per D-10/D-11), scoped into
`files_modified` up front.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Observatory` model + `MPCObscodeFetcher` (`solsys_code_observatory/utils.py`) — directly reusable
  for the D-08 site-resolution tiers 1-2.
- `CalendarEventTelescopeLabel` sidecar pattern (`solsys_code/models.py`) — direct structural analog
  for the new `site_needs_review` flag (D-09): a boolean sidecar/field defaulting to "trust it" with
  an explicit "flag, don't silently guess" fallback.
- `load_telescope_runs.py` command structure — CLI argument parsing, per-line try/except
  (`ValueError`, `Observatory.DoesNotExist`) skip-and-log pattern, summary counters — direct template
  for the new `import_campaign_csv`-style command.

### Established Patterns
- Create-or-update idempotency: `get_or_create` + conditional save, keyed on load-bearing fields only,
  to avoid `modified`-timestamp churn on re-run (established across Phases 3/4/10/11). D-04/D-05
  follow this directly.
- "Flag, don't silently guess" for ambiguous/unresolved external data (Phase 7's `[UNVERIFIED]`
  prefix + `is_verified` sidecar). D-08/D-09 follow this directly rather than fail-closed (skip) or
  guess (auto-create with fabricated coordinates).

### Integration Points
- New model(s) likely belong in `solsys_code/models.py` alongside `CalendarEventTelescopeLabel`, or a
  new `campaign_models.py` if the file is judged to be growing unwieldy (planner's call).
- New management command belongs in `solsys_code/management/commands/`, following existing naming
  (`load_telescope_runs.py`, `fetch_jplsbdb_objects.py`).
- Paired demo notebook belongs in `docs/notebooks/pre_executed/`, fixture in
  `docs/notebooks/pre_executed/fixtures/` (D-11) — new subdirectory, first of its kind in this repo;
  check no existing `fixtures/` convention is being contradicted.

</code_context>

<specifics>
## Specific Ideas

- The real 3I/ATLAS sheet is at
  https://docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI/ (per the seed
  doc) — the operator will need to export/provide this CSV for the actual bootstrap import; it is not
  currently in the repo. The demo notebook and its tests must never depend on this real file (per
  CAMP-05/D-10/D-11), but the actual one-off import (CAMP-04's live run) does need it externally.
- Example real rows span FTN/MuSCAT3 multi-band imaging, Palomar P200/NGPS imaging+spectroscopy, and
  VLT/MUSE IFU monitoring — i.e., mostly facilities outside FOMO's existing sync commands. This is why
  D-08's site resolution can't assume matches against `SITE_TELESCOPE_MAP` (which is LCO/Gemini-sync
  specific) and must fall through to the MPC API / placeholder tiers.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 14's scope. (Phase 15 discussion was started first this session
and explicitly deferred until after Phase 14 — see the sequencing note in Phase Boundary above.)

### Reviewed Todos (not folded)
- See `<decisions>` → "Reviewed Todos (not folded)" above.

</deferred>

---

*Phase: 14-Campaign Data Model & Bootstrap Import*
*Context gathered: 2026-07-02*
