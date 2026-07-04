# Phase 17: Coverage-Gap Analysis (Deferrable to v2.1) - Context

**Gathered:** 2026-07-04
**Status:** Ready for planning

<domain>
## Phase Boundary

A user viewing a per-campaign table can request, for one `(campaign target, observing site)`
pair, the list of observable-but-unclaimed nights within a bounded, capped date range — FOMO's
differentiator over any spreadsheet. "Observable" means the site has a non-zero astronomical
(−15°) dark window that night (dark-window-only, per GAP-01's locked decision below) — not true
target-altitude/airmass filtering. "Unclaimed" means no `CampaignRun` with
`approval_status=approved` and a non-terminal-failure `run_status` covers that date at that site.

The computation runs only on explicit user request (a button on the campaign table page), never
inline on table load, and is cached (Django's low-level cache framework, 1-hour TTL) rather than
recomputed on every view. It must never import `solsys_code.ephem_utils` at module scope —
`solsys_code/telescope_runs.py`'s existing `sun_event()`/`get_site()` helpers are the only
ephemeris dependency this phase introduces.

Out of scope for this phase (belongs to later phases or v2.1+): true target-altitude/airmass
filtering (explicitly rejected by GAP-01's locked decision, not just deferred), booking/reserving
a gap as a claimed slot (advisory display only — claiming a gap means submitting a normal
`CampaignRun` through Phase 16's existing form), auto-suggesting `ut_start`/`ut_end` on the Phase
16 submission form (captured as a deferred idea below, not this phase's concern).

</domain>

<decisions>
## Implementation Decisions

### GAP-01: research-spike decision (dark-window-only vs. target-altitude)
- **D-01:** **Locked now: dark-window-only**, reusing `telescope_runs.py`'s `sun_event()`/
  `get_site()` — not true target-altitude/airmass filtering via `ephem_utils`. Pre-milestone
  research (`.planning/research/ARCHITECTURE.md`'s explicit "Decision", `PITFALLS.md`,
  `SUMMARY.md`, `STACK.md`) already unanimously recommends this; re-running the same research
  question at plan time would not surface new information. `gsd-phase-researcher` should treat
  this as settled and focus research effort on implementation details (view/cache design), not
  re-litigating the SPICE-cost tradeoff.
- **D-02:** GAP-01's success criterion ("a phase-time research spike produces an explicit
  decision") is satisfied with a **short written decision doc during execution**
  (`17-GAP-01-DECISION.md` or similar), mirroring Phase 13's `13-DECISION.md` precedent — even
  though the decision itself was reached quickly via this discussion rather than a multi-day
  spike. Gives the phase a citable artifact satisfying the letter of its own success criterion.
- **D-03:** A per-date `sun_event(kind='dark')` `ValueError` (e.g. a hypothetical future polar/
  midnight-sun `Observatory`) **skips that one date as "unknown", does not abort the whole gap
  request**. Matches this codebase's existing per-line/per-record "log+skip, never abort"
  convention (`load_telescope_runs`, `import_campaign_csv`). None of today's 4 `SITES` entries
  (Magellan-Clay/Baade, NTT, FTS) hit this case in practice.
- **D-04:** **Any non-zero dark window counts as "observable"** — no minimum-duration threshold.
  Simplest rule; avoids inventing and justifying an arbitrary cutoff (e.g. "≥1 hour") not
  requested by any requirement.

### "Claimed" definition
- **D-05:** A date is **"claimed"** (excluded from the gap list) when a `CampaignRun` has
  `approval_status='approved'` **and** `run_status` is **not** in
  `{cancelled, not_awarded, weather_tech_failure}`. A run that was approved but then fell through
  in the real world frees its date back up as a gap — matches this phase's purpose of surfacing
  genuinely uncovered nights, not just "was ever approved".
- **D-06:** The claimed date is **`obs_date` if set, else derived from `ut_start`**. `obs_date` is
  the intended single source of truth for "which night"; `ut_start`/`ut_end` supply precise timing
  within that night and are only used to derive a date when `obs_date` was left blank (possible
  for some public submissions).
- **D-07:** When deriving from `ut_start`, convert to the **site's local calendar date**, using the
  same local-noon-anchored "observing night" convention `sun_event()` already uses — not the raw
  UTC date. Keeps "which night" consistent between the observable side (`sun_event`, local-noon
  anchored) and the claimed side; avoids an off-by-one-night mismatch at sites west of UTC where
  local evening has already rolled to the next UTC calendar date.
- **D-08:** A `CampaignRun` with **neither `obs_date` nor `ut_start` set** cannot be attributed to
  any date. It is **flagged separately as "undated, needs review"** alongside the gap-analysis
  result (not silently dropped, and not treated as claiming a date) — surfaces the data-quality
  issue to staff rather than hiding it.

### Trigger, caching & date range
- **D-09:** Gap computation is triggered by **a button on the per-campaign table page** that loads
  a separate gap-analysis section/page via a **normal (non-htmx) request** — no new JS dependency;
  matches the existing `django-tables2`/`crispy-forms`-based view pattern this codebase already
  uses, and satisfies ROADMAP's "UI hint: yes" without adding htmx wiring for what should be a
  fairly rare action.
- **D-10:** Results are cached via **Django's low-level cache framework** (`cache.set`/`cache.get`),
  keyed by `(campaign, target, site, date range)`, with a **1-hour TTL** — not a dedicated
  persistent model. No new migration needed; the TTL naturally handles staleness (a newly-approved
  run shows up as claimed within an hour) without needing invalidation signals wired into every
  `CampaignRun` save. Display a "last computed at" timestamp alongside the result (per
  `PITFALLS.md`'s explicit recommendation) so users know the result isn't necessarily current to
  the second.
- **D-11:** Default date-range window is **the next 90 days from today**; **max allowed span is
  180 days**, enforced server-side regardless of any client-supplied range. 90 days covers a
  typical multi-month interstellar-object observing season without an excessive per-request
  night count; 180 days bounds the worst-case number of `sun_event()` calls per request (per
  `PITFALLS.md`'s explicit warning against an unbounded date range).

### Target + site selection
- **D-12:** Target selection: **auto-use the sole `Target` when `campaign.targets.count() == 1`**;
  otherwise show a dropdown to pick one. Mirrors Phase 14 D-07's existing single-target
  auto-assign convention (`import_campaign_csv.py`) — zero friction for the common single-object
  campaign case (e.g. 3I/ATLAS), explicit choice only when genuinely ambiguous.
- **D-13:** Site selection: a **dropdown of `Observatory` records already used by this campaign's
  `CampaignRun`s** (i.e. distinct non-null `.site` values among the campaign's runs) — not every
  `Observatory` in the DB, and not restricted to `telescope_runs.py`'s 4-entry `SITES` dict.
  Naturally scopes the picker to sites relevant to this campaign; `sun_event()` accepts any
  `Observatory` instance directly (it doesn't require a `SITES`-dict name), so this is not a
  functional restriction.
- **D-14:** If a campaign has **zero `Target`s**, or its `CampaignRun`s have **no resolved site at
  all** (all `site=None`/`site_needs_review=True`), the gap-analysis button is **hidden/disabled
  with an explanatory message** rather than shown and failing on click. Consistent with this
  codebase's "never fabricate/guess" discipline already applied to site resolution — don't offer a
  control with nothing meaningful to select.

### Claude's Discretion
- Exact URL names/paths for the gap-analysis view/section.
- Exact wording of the "last computed at" display and the "undated, needs review" flag (D-08).
- Whether the gap-analysis result lives on its own page or as a section appended to the existing
  per-campaign table page — planner's call, consistent with D-09's non-htmx-button trigger.
- Internal structure/naming of the `17-GAP-01-DECISION.md` artifact from D-02 — follow Phase 13's
  `13-DECISION.md` shape loosely, not verbatim.
- Exact cache key format for D-10 (e.g. a hash of the tuple vs. a delimited string) — any format
  that's stable and collision-free across requests is fine.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 17: Coverage-Gap Analysis (Deferrable to v2.1)" — goal, success
  criteria, explicit deferrability framing
- `.planning/REQUIREMENTS.md` §"Coverage-Gap Analysis (GAP)" (GAP-01, GAP-02) — full requirement
  text

### Pre-milestone research (already settles GAP-01 — see D-01)
- `.planning/research/ARCHITECTURE.md` — explicit "Decision" section recommending
  `telescope_runs.py`'s `sun_event()`/`get_site()` over `ephem_utils`; explains why importing
  `ephem_utils` must never happen at module scope
- `.planning/research/PITFALLS.md` — Pitfall 5 (never compute inline; cache with TTL/invalidation;
  cap date range) and Pitfall 4 (SPICE import cost) — directly shaped D-09/D-10/D-11
- `.planning/research/SUMMARY.md` — coverage-gap phase framing, "HIGH — requires dedicated
  research spike" flag (addressed by D-01/D-02)
- `.planning/research/STACK.md` — confirms stdlib `datetime` + sorting/set-difference is
  sufficient (no interval-arithmetic package needed) for this phase's date-gap computation
- `.planning/research/FEATURES.md` — frames coverage-gap as "genuinely novel synthesis" of
  `telescope_runs.py`'s observable side and `CampaignRun`'s claimed side; confirms advisory-only
  scope (no booking/locking)

### Milestone audit (confirms this phase is being executed, not deferred)
- `.planning/v2.0-MILESTONE-AUDIT.md` §"Routing" — documents the choice to execute Phase 17 before
  closing v2.0 (path 2 of 2 offered)

### Prior phase context
- `.planning/phases/14-campaign-data-model-bootstrap-import/14-CONTEXT.md` — D-02 (`approval_status`/
  `run_status` independent fields, referenced by D-05), D-07 (single-target auto-assign
  convention, referenced by D-12), D-08 (3-tier site resolution, referenced by D-13/D-14)
- `.planning/phases/15-per-campaign-table-view-read-path/15-CONTEXT.md` — D-13 (`is_staff` gating
  precedent; not directly reused here since gap analysis is not staff-only, but establishes the
  page this phase's button attaches to)
- `.planning/phases/16-submission-form-approval-queue-calendar-projection-write-pat/16-CONTEXT.md`
  — D-09 (non-staff visibility of approved/rejected rows; establishes the `CampaignRunTableView`
  page D-09 of this phase's button attaches to), D-05 (submission form field scope — relevant
  context for the deferred `ut_start`/`ut_end` auto-suggest idea below)

### Existing code precedent
- `solsys_code/telescope_runs.py` — `sun_event(site, date, kind='dark')` (the only ephemeris
  dependency this phase should introduce; accepts any `Observatory` instance, not just `SITES`-
  dict names), `get_site()`, `horizon_dip()`; module deliberately has zero `ephem_utils` import
  (documented in `.planning/PROJECT.md`'s Key Decisions)
- `solsys_code/models.py` — `CampaignRun` (`approval_status`, `run_status`, `obs_date`,
  `ut_start`/`ut_end`, `site` FK to `Observatory`, `site_needs_review`), `TargetList.targets`
  (reverse relation used by D-12)
- `solsys_code/management/commands/import_campaign_csv.py` — `auto_target = campaign.targets.first()
  if campaign.targets.count() == 1 else None` (D-07 precedent directly reused by D-12)
- `solsys_code/campaign_utils.py` — `resolve_site()` (3-tier site resolution; relevant precedent
  for D-14's "never guess" discipline, though not directly called by this phase)
- `solsys_code/campaign_views.py`, `campaign_tables.py`, `campaign_urls.py` — Phase 15/16's
  view/table/URL structure the new gap-analysis view/button attaches to
- `solsys_code/forms.py` — existing `EphemerisForm`'s `altitude > 0` `Observatory` restriction
  (considered and rejected for D-13 in favor of scoping to the campaign's own used sites)
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md` — precedent shape
  for D-02's `17-GAP-01-DECISION.md` artifact

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `telescope_runs.sun_event(site, date, kind='dark')` — directly reusable for the observable side
  of the gap computation; already precise to ≤2 min vs. LCO skycalc reference, already tested.
- `telescope_runs.get_site()` / `Observatory.to_earth_location()` — site/location resolution, if
  needed beyond the `CampaignRun.site` FK already available.
- Django's low-level cache framework (`django.core.cache.cache`) — no new dependency, TTL support
  built in.
- `campaign.targets.first()` / `.count()` pattern from `import_campaign_csv.py` — directly reusable
  for D-12's target auto-selection.

### Established Patterns
- Per-record "log+skip, never abort" error handling (`load_telescope_runs`,
  `import_campaign_csv`) — reused by D-03 for per-date `sun_event()` failures.
- "Never fabricate/guess" discipline (Phase 14's 3-tier site resolution) — reused by D-14's
  hide-the-button-rather-than-fail approach.
- No-churn / idempotency conventions elsewhere in this codebase are not directly relevant here
  (gap analysis is read-only/advisory, not a create-or-update path).

### Integration Points
- New gap-analysis view/section attaches to the existing per-campaign table page
  (`CampaignRunTableView` / its template) via D-09's button.
- Reads `CampaignRun` rows filtered by campaign + target + site (D-05/D-06/D-07 define which rows
  count as "claiming" a date).
- Calls `telescope_runs.sun_event()` per candidate date in the requested range (D-11's 90/180-day
  window), catching per-date `ValueError` per D-03.
- Caches the computed result via `django.core.cache.cache`, keyed per D-10.

</code_context>

<specifics>
## Specific Ideas

No new specific external references beyond what pre-milestone research already captured — this
discussion stayed at the decision level (claimed-date definition, trigger/caching mechanics,
target/site selection) rather than naming new external examples or reference implementations.

</specifics>

<deferred>
## Deferred Ideas

- **Auto-calculate/suggest `ut_start`/`ut_end` on the Phase 16 public submission form via JS**,
  based on the entered site (MPC code) + `obs_date`, likely reusing `telescope_runs.sun_event()`.
  Raised during this discussion (Target + site selection area) as a way to improve submission data
  quality (fewer undated/imprecise `CampaignRun` rows feeding D-08's "undated, needs review" flag),
  but it's a Phase 16 submission-form enhancement, not Phase 17 (Coverage-Gap Analysis) scope.
  Candidate for a future phase or quick task.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — matched by
  keyword overlap only (`site`, `telescope`, `instrument`, `extraction`, `module`); already
  resolved per Phase 14/15/16 context (extraction already happened in `calendar_utils.py`) and
  unrelated to this phase. Third time reviewed-not-folded (also Phases 15, 16) — not re-asked of
  the user this time given the identical, already-settled outcome.
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — matched by
  keyword overlap only (`calendar`, `utils`, `helpers`, `module`, `code`); unrelated to
  coverage-gap analysis. Third time reviewed-not-folded (also Phases 15, 16); not re-asked of the
  user this time for the same reason.

</deferred>

---

*Phase: 17-Coverage-Gap Analysis (Deferrable to v2.1)*
*Context gathered: 2026-07-04*
