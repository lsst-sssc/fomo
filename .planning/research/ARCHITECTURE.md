# Architecture Research

**Domain:** Django/TOM-Toolkit campaign-coordination feature — v2.1 range/window scheduling, asset-type-aware gap analysis, and site-disambiguation UI
**Researched:** 2026-07-05
**Confidence:** HIGH (all findings grounded in current repo source, not external ecosystem research — this milestone is pure internal-integration design)

This file supersedes the previous contents (dated 2026-07-02, about the v2.0 Campaign Coordination
milestone's initial build) — that milestone shipped. This is a full rewrite scoped to the v2.1
"Uncertain Scheduling & Site Disambiguation" milestone: how the new range/window scheduling,
ground-vs-space-mission asset distinction, and fuzzy site-disambiguation UI integrate with the
`CampaignRun`/`campaign_gap`/`ApprovalQueueTable` infrastructure v2.0 already shipped.

## Standard Architecture

### System Overview (current v2.0 state, annotated with where v2.1 lands)

```
+------------------------------------------------------------------------------+
|                              Write path (staff)                              |
|  ApprovalQueueView --renders--> ApprovalQueueTable (pending / decided)        |
|        |                              |                                      |
|        |                    [NEW v2.1] render_site() interactive branch      |
|        |                    (fuzzy dropdown + free-text fallback)            |
|        v                              v                                      |
|  CampaignRunDecisionView       [NEW v2.1] CampaignRunSiteResolutionView       |
|  (approve/reject, atomic       (POST site_pk or free-text, StaffRequired,    |
|   conditional .update())        NOT gated on approval_status)                |
|        |                              |                                      |
|        +--------------+---------------+                                     |
|                        v                                                     |
|              CampaignRun.site / site_raw / site_needs_review                 |
|              (via campaign_utils.resolve_site(create_placeholder=False))    |
+------------------------------------------------------------------------------+
                        |
                        v
+------------------------------------------------------------------------------+
|                    CampaignRun (model) - schema change (v2.1)                |
|  campaign, target, telescope_instrument, site, site_raw, site_needs_review   |
|  [CHANGED] obs_date + ut_start + ut_end  ->  window_start/window_end (date)  |
|            + ut_start/ut_end retained as OPTIONAL precise-time fields        |
|  [CHANGED] UniqueConstraint(campaign, telescope_instrument, ut_start)        |
|            ->  UniqueConstraint(campaign, telescope_instrument, window_start)|
+------------------------------------------------------------------------------+
                        |                                   |
                        v                                   v
+---------------------------------------+   +--------------------------------+
|  campaign_gap.py - claimed_dates()     |   |  import_campaign_csv.py /      |
|  [CHANGED] window expansion, gated on  |   |  campaign_utils.parse_obs_window|
|  Observatory.observations_type ==      |   |  [CHANGED] range/TBD-tolerant  |
|  SATELLITE_OBSTYPE (ground claims       |   |  parsing -> window_start/end,  |
|  every date in window; space mission   |   |  window_needs_review flag       |
|  claims none until window narrows)     |   |                                |
+---------------------------------------+   +--------------------------------+
```

### Component Responsibilities

| Component | Responsibility | New / Modified |
|-----------|----------------|-----------------|
| `solsys_code/models.py:CampaignRun` | Owns the scheduling fields and the natural-key `UniqueConstraint` | **Modified** — window fields + constraint change |
| `solsys_code/solsys_code_observatory/models.py:Observatory` | `obscode` max length; `observations_type`/`SATELLITE_OBSTYPE` (already exists, reused not duplicated) | **Modified** — `obscode` max_length widened |
| `solsys_code/campaign_gap.py` | Composes observable dates with claimed dates into a gap | **Modified** — `claimed_dates()` rewritten for windows + asset type |
| `solsys_code/campaign_utils.py` | `resolve_site()` (unchanged contract, reused as-is), `parse_obs_window()` (rewritten), **new** `fuzzy_match_observatories()` | **Modified + new helper** |
| `solsys_code/management/commands/import_campaign_csv.py` | Calls `parse_obs_window`, builds natural-key lookup for `insert_or_create_campaign_run` | **Modified** — call-site + natural-key lookup change |
| `solsys_code/campaign_tables.py:CampaignRunTable` | Column definitions for `obs_date`/`ut_start`/`ut_end` | **Modified** — window columns replace date columns |
| `solsys_code/campaign_tables.py:ApprovalQueueTable` | `render_site()` static badge | **Modified** — interactive dropdown branch, gated on `show_actions` |
| `solsys_code/campaign_views.py:CampaignRunDecisionView` | Atomic approve/reject + calendar projection | **Modified** — guard against clobbering a manually-resolved site; CAL-01 field references re-checked against window fields |
| `solsys_code/campaign_views.py` (new) | `CampaignRunSiteResolutionView` | **New** |
| `solsys_code/campaign_forms.py:CampaignRunSubmissionForm` | Public intake fields | **Modified** — window fields replace `obs_date`/`ut_start`/`ut_end`; VIEW-05 opt-in flag added |
| `solsys_code_observatory` `CreateObservatory` (existing CreateView) | Manual Observatory creation via MPC lookup | **Reused unchanged** as the "free-text -> create new Observatory" fallback destination, not reimplemented |

## Part A — Range/Window Scheduling, Asset-Type Distinction, Gap Analysis

### A1. Recommended schema shape (feeds the milestone's phase-time spike, does not replace it)

Replace `obs_date` (DateField) with `window_start`/`window_end` (both `DateField(null=True, blank=True)`):

- A classically-scheduled single night imports as `window_start == window_end` (the "1-day window" the milestone explicitly calls for) — this is the *same* convention Stage 2's `load_telescope_runs` already uses conceptually for one-night-per-`CalendarEvent`, so it is not a new mental model for this codebase, just applied to `CampaignRun`.
- A true range ("Aug 1-15") sets `window_start != window_end`.
- A "TBD pending Cycle 2" row sets `window_start = window_end = None`, with a new `window_needs_review = BooleanField(default=False)` flag — this directly mirrors the existing `site_needs_review` sidecar-flag pattern already established on this same model, and the `ut_needs_review` return value `parse_obs_window()` already produces today. Reuse the pattern, don't invent a new one.
- **Keep `ut_start`/`ut_end` as-is** (optional precise-time `DateTimeField`s). They are not part of "the range" the milestone is asking about — they exist for a *different* purpose: `CampaignRunDecisionView`'s CAL-01/CAL-02 calendar projection needs a precise `start_time`/`end_time` to create a `CalendarEvent`, which a date-only window cannot supply. Decoupling "what dates does this run claim" (window) from "does this run have a precise enough time to show on the calendar" (ut_start/ut_end) avoids conflating two different consumers of the same fields, and avoids forcing every range/TBD row to fabricate a fake time just to satisfy the calendar-projection code path.

**Natural key**: replace `UniqueConstraint(fields=['campaign', 'telescope_instrument', 'ut_start'])` with `UniqueConstraint(fields=['campaign', 'telescope_instrument', 'window_start'])`. This is a direct field swap, not a redesign — the constraint already relies on a nullable field today (`ut_start` is `null=True`), and SQL's NULL-is-distinct-from-NULL semantics already let multiple no-start-time `CampaignRun`s coexist without collision. `window_start` inherits that exact same nullable-natural-key behavior for TBD rows (multiple TBD rows for the same campaign+telescope simply never collide) — this is a continuation of existing behavior, not a new edge case introduced by this milestone. Phase 14's deterministic-offset hack (built to avoid two distinct unparseable-`ut_start` rows colliding) becomes unnecessary once `window_start=None` rows naturally don't collide — a net simplification worth calling out to whichever phase touches this.

**Blocking prerequisite**: `Observatory.obscode` is `CharField(max_length=4)`; `resolve_site()` already guards `len(code) > _MAX_OBSCODE_LEN` and refuses to create an Observatory for anything longer (flags `needs_review`, no row). JWST's `'500@-170'` (8 chars) can **never** resolve to an `Observatory` row until this field is widened via migration. Since the asset-type distinction (`Observatory.observations_type == SATELLITE_OBSTYPE`) is only checkable once `site` is a resolved FK, **no space-mission run can ever be asset-classified until this migration lands** — this is the single hardest dependency in the whole milestone and must be sequenced first (see Build Order).

### A2. `campaign_gap.claimed_dates()` — asset-aware window expansion

Current behavior (`solsys_code/campaign_gap.py:138-211`): one `CampaignRun` contributes at most one claimed date, derived from `obs_date` if set, else from `ut_start` via `_observing_night_date()` (a timezone-dependent local-noon convention that can raise for a blank-timezone site — the CR-02 fix already guards this with a log+skip).

New behavior required:

```python
def claimed_dates(campaign, target, site) -> tuple[set[date], list, list, list]:
    ...
    claimed: set[date] = set()
    undated_runs: list[CampaignRun] = []       # window_start is None entirely
    pending_narrowing_runs: list[CampaignRun] = []  # NEW: space-mission run with a real
                                                      # but not-yet-narrowed window
    for run in qs:
        if run.window_start is None:
            undated_runs.append(run)
            continue
        end = run.window_end or run.window_start           # 1-day window default
        is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE
        if is_space_mission and run.window_start != end:
            # Recommendation: a space-mission run only "claims" once its window has
            # narrowed to a single concrete day (window_start == window_end) -- a genuine
            # multi-day/range window from a space mission claims nothing yet.
            pending_narrowing_runs.append(run)
            continue
        d = run.window_start
        while d <= end:
            claimed.add(d)
            d += timedelta(days=1)
    return claimed, undated_runs, unattributed_runs, pending_narrowing_runs
```

Notes for whichever phase implements this:
- `site` is already a required parameter to `claimed_dates()` (it's the campaign-scoped `Observatory` selected in the gap-analysis form), so `site.observations_type` is available with zero new plumbing — the asset-type check is a one-line addition to an already-passed argument, not a new dependency.
- Since `window_start`/`window_end` are plain `DateField`s (no time-of-day), the whole `_observing_night_date()` helper (and its timezone lookup, `ZoneInfo`, and blank-timezone `ValueError` guard) becomes **dead code** once the schema migration lands — it existed only to derive a date from a `ut_start` *timestamp*; a *window* is already a date. Removing it is a real simplification opportunity, not just a nice-to-have — flag it to the plan-writer so it isn't left as unreachable code.
- `pending_narrowing_runs` is a genuinely new bucket the gap-analysis view/template must surface (today the template already has a symmetric slot for `undated_runs`/`unattributed_runs` — see `campaignrun_gap_analysis.html:53-64` — so this is an additive third list in the same pattern, not a new UI concept).
- `_compute_gap()`'s `gap = obs - claimed` computation itself does not change; only what feeds `claimed` changes. `observable_dates()` is untouched by this milestone entirely.
- The exact space-mission "narrowing trigger" (`window_start == window_end` is this document's recommendation) is explicitly called out in PROJECT.md as a spike question — treat the code above as the default proposal to validate against real 3I sheet rows in the phase-time spike, not a settled decision.

### A3. CSV import (`import_campaign_csv` / `campaign_utils.parse_obs_window`)

Current contract (`campaign_utils.py:186-244`): `obs_date_raw` **must** parse as `%Y-%m-%d` or the function raises (D-05 "true natural-key failure", causing `import_campaign_csv` to skip-and-log the row); `ut_range_raw` is always best-effort with a midnight-UTC fallback.

Required change: extend the same best-effort discipline to the date column itself, since the milestone explicitly states a range/TBD cell "must import ... instead of being silently dropped." Recommended shape:

- Exact `YYYY-MM-DD` → `window_start = window_end = date`, `window_needs_review = False` (unchanged from today's success path, just renamed).
- A recognizable range pattern (e.g. `"Aug 1-15"`, `"2026-08-01 to 2026-08-15"`) → `window_start`/`window_end` distinct, `window_needs_review = False`.
- Anything else non-blank ("TBD pending Cycle 2", free prose) → `window_start = window_end = None`, `window_needs_review = True`. This **removes** the current `raise ValueError` path entirely — under the new contract there is no longer a "true natural-key failure" case for the date column at all, since the natural key now keys on `window_start` and a `None` value is a valid (if maximally ambiguous) natural-key member per A1's nullable-key reasoning. This is a deliberate behavior change from today's D-05 wording and should be called out explicitly to the plan-writer/spike, since it changes `import_campaign_csv`'s skip-and-log counters (a "TBD" row moves from `skipped` to `created`/`updated` with `window_needs_review=True`).
- Follow the exact same "add a narrow regex for each confirmed real-sheet shape, never a permissive general-purpose date parser" discipline `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` already establish for the UT-time column — the spike's job is to enumerate the *actual* range/TBD shapes in the real 3I sheet (mirroring how the existing UT-time patterns were derived from "RESEARCH.md 'Real 3I/ATLAS Sheet -- Verified Shape'"), not to guess a generic parser up front.

### A4. Consumers that must be updated in lockstep with the schema change

Because `obs_date`/`ut_start` currently appear as literal field names in several places outside `campaign_gap.py`, the schema migration phase's `files_modified` must include all of:
- `solsys_code/campaign_tables.py` — `CampaignRunTable.Meta.fields` lists `obs_date`, `ut_start`, `ut_end` explicitly (line ~58-60); `order_by = ('-obs_date',)` (D-10) needs a new ordering field.
- `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm.obs_date`/`ut_start`/`ut_end` fields and their crispy `Fieldset` layout.
- `solsys_code/campaign_views.py:CampaignRunSubmissionView.form_valid()` — the explicit `CampaignRun.objects.create(... obs_date=..., ut_start=..., ut_end=...)` kwargs.
- `solsys_code/campaign_views.py:CampaignRunDecisionView.post()` — the CAL-01 gate `if run.telescope_instrument and run.ut_start and run.ut_end:` stays keyed on `ut_start`/`ut_end` (unchanged, since calendar projection is decoupled from the window per A1), but any code that assumed `ut_start` was always populated for a "scheduled" run must be re-audited now that a range/TBD row may have `ut_start=None` while still having a real `window_start`/`window_end`.
- `solsys_code/campaign_utils.py:insert_or_create_campaign_run()`'s caller in `import_campaign_csv.py` — the `lookup` dict currently keyed on `ut_start` must switch to `window_start`.
- Paired demo notebook: `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` is not one of the four modules CLAUDE.md's demo-notebook rule names explicitly, but the same spirit applies — if `import_campaign_csv`'s behavior changes (new range/TBD parsing), its existing demo notebook and fixture (`campaign_sample.csv`) should be checked for staleness even though it isn't in the enforced list.

## Part B — Fuzzy-Match Site-Disambiguation UI

### B1. Where it plugs into the existing table/view pair

`ApprovalQueueTable.render_site()` (`campaign_tables.py:111-137`) currently renders one of three states from a **shared** subclass used by *both* the pending table (`show_actions=True`) and the decided table (`show_actions=False`, explicitly documented as read-only). The interactive dropdown must be gated on the same `self.show_actions` flag already threaded through `__init__` for the `actions` column — **no new constructor parameter needed**, direct reuse of an existing mechanism:

```python
def render_site(self, record):
    site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
    if site_short_name:
        return site_short_name          # resolved -- unchanged
    site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
    if not self.show_actions:
        return <today's static badge>   # decided table -- stays read-only, unchanged
    # pending table, unresolved site -- NEW interactive branch
    return self._render_site_disambiguation(record, site_raw)
```

### B2. Fuzzy-candidate generation — new pure-logic helper, no new dependency

Add `fuzzy_match_observatories(site_raw: str, candidates: Iterable[Observatory], limit: int = 5, cutoff: float = 0.5) -> list[Observatory]` to `campaign_utils.py`, next to `resolve_site()` (same module, same "never raise, return a usable value" discipline). Use stdlib `difflib.SequenceMatcher`/`get_close_matches`, scored against each Observatory's `name`, `short_name`, and each newline/comma-split entry of `old_names` (a free `TextField`), keeping the best score per Observatory.

**Do not add `rapidfuzz`/`thefuzz` as a new dependency.** `rapidfuzz` happens to be present in this venv, but only as a transitive dependency of `poetry`'s `cleo` package (confirmed via `pip show rapidfuzz` → `Required-by: cleo`) — it is not a FOMO runtime dependency and pinning on its accidental presence would be fragile. `difflib` is stdlib, matching this project's existing preference for stdlib over new packages where sufficient (the `zoneinfo` precedent noted in this project's own Constraints section). Observatory table size is small (dozens to low hundreds of rows), so `SequenceMatcher`'s O(n*m) string comparison is not a performance concern.

**N+1 avoidance**: fetch `Observatory.objects.all()` **once** in `ApprovalQueueView.get_context_data()` (or in `ApprovalQueueTable.__init__`) and pass the materialized list into the table, rather than having `fuzzy_match_observatories()` re-query per row inside `render_site()`. This mirrors the exact N+1 lesson this codebase already learned and fixed once (`fomo_render_calendar`'s DISPLAY-09 prefetch, v1.6) — worth flagging explicitly since `render_site()` runs once per table row.

### B3. New endpoint — do not fold into `CampaignRunDecisionView`

Add `CampaignRunSiteResolutionView` (`StaffRequiredMixin`, `View`, `http_method_names = ['post']`) as a sibling to `CampaignRunDecisionView`, **not** a new `action` value inside it, because:

1. `CampaignRunDecisionView.post()`'s conditional `.update()` is gated on `approval_status=PENDING_REVIEW` — site resolution must be actionable on *any* row (including already-approved/rejected ones, since staff may need to correct a site after the fact), so it cannot share that gate.
2. The natural-key `UniqueConstraint` does not include `site`, so a site-only update never risks a constraint collision — this endpoint is uniquely low-risk and does not need the same atomic-conditional-update pattern SUBMIT-03 requires; a plain `run.site, run.site_needs_review = ...; run.save(update_fields=[...])` suffices, no new transaction pattern needed.
3. Conflating "decide" (approve/reject, one meaning per POST) with "resolve site" (a different, independently-repeatable action) would break the existing `action in ('approve', 'reject')` validation contract and complicate `updated_count`-based messaging that currently assumes exactly two possible transitions.

Accepts either `site_pk` (one of the fuzzy candidates, or any Observatory at all if the dropdown lists "browse all") or `site_raw_override` (free text). Free-text path calls `resolve_site(text, create_placeholder=False)` (the exact same call already used by `CampaignRunDecisionView`, reused not duplicated) — if that also misses, redirect to the **existing** `CreateObservatory` `CreateView` (in `solsys_code_observatory`, already MPC-code-driven) with a `?next=` back to the approval queue, rather than building a second Observatory-creation form. This satisfies "free-text resolve-or-create fallback" using the app's existing vetted creation path, and keeps `create_placeholder=False`'s "never auto-fabricate" invariant (quick task `260705-l1v`) intact — a human explicitly choosing to create via `CreateObservatory` is categorically different from the code silently fabricating a placeholder.

### B4. Required fix in `CampaignRunDecisionView` — must ship together with B3

`CampaignRunDecisionView.post()` **unconditionally** calls `resolve_site(run.site_raw, create_placeholder=False)` on every approve (`campaign_views.py:302`), overwriting `run.site`/`run.site_needs_review` regardless of whether a human already resolved it via the new disambiguation UI. Without a guard, a staff member who manually fixes a site via B3 and then clicks Approve would have their choice silently clobbered back to whatever the automated tier-1/tier-2 resolver produces (or `None`/`needs_review=True` if it produces nothing). Required change: only call `resolve_site()` when `run.site_id is None` (i.e., skip re-resolution if a site is already set, whether by CSV auto-import, MPC auto-match, or a human's B3 pick). **This guard fix is not optional polish — it must land in the same phase/commit set as the new disambiguation endpoint**, or there is a window where the UI exists but silently doesn't stick.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Deriving "is space mission" from a new `CampaignRun` field

**What people might do:** add an `is_space_mission = BooleanField()` directly on `CampaignRun`.
**Why it's wrong:** the milestone explicitly calls for reusing `Observatory.observations_type` — a `CampaignRun`-level flag would drift out of sync with the resolved `site`'s actual type, require manual double-entry, and duplicate data already modeled on `Observatory`.
**Do this instead:** always derive it at read time from `run.site.observations_type == Observatory.SATELLITE_OBSTYPE` (guarding `run.site is not None` first, since an unresolved site can't be classified either way — it just doesn't claim, same as today).

### Anti-Pattern 2: Folding site-resolution into the atomic approve/reject `.update()`

**What people might do:** add a `site_pk` POST param to the existing `CampaignRunDecisionView` and update both `approval_status` and `site` in the same conditional `.update()`.
**Why it's wrong:** breaks the existing `action in ('approve', 'reject')` two-state contract, couples an unrelated field to the approval-status gate (so site fixes on already-decided rows become impossible), and complicates the `updated_count` messaging logic that currently cleanly maps to "approved / rejected / already-decided / doesn't-exist".
**Instead:** a separate, ungated `CampaignRunSiteResolutionView` (see B3).

### Anti-Pattern 3: A generic/permissive date-range parser for the CSV importer

**What people might do:** reach for `dateutil.parser.parse()` or a broad regex to handle "any" range/TBD text in one shot.
**Why it's wrong:** this codebase's `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` precedent deliberately uses narrow, sheet-verified regexes so a stray unrelated string can never "succeed" into a wrong-but-plausible date — a permissive parser reintroduces exactly the risk that precedent was built to avoid (RESEARCH.md Anti-Patterns, referenced in `campaign_utils.py`'s own docstring).
**Instead:** enumerate the actual range/TBD shapes found in the real 3I sheet during the phase-time spike, and add one narrow pattern per confirmed shape, falling back to `window_needs_review=True` (never a raise, never a guess) for anything else.

## Build Order (dependency-ordered)

1. **Phase-time investigation spike** (already scoped in PROJECT.md, not a separate build phase) — confirms the exact window field names/nullability, the natural-key replacement, the space-mission narrowing-trigger rule, and the CSV range/TBD shapes, against real 3I sheet rows. Everything below assumes its output; treat this document's A1/A2/A3 recommendations as the default answer to validate, not a bypass of the spike.
2. **`Observatory.obscode` max_length migration** (widen past 4 chars). Small, independent, and the hardest blocking dependency: no space-mission `Observatory` row (JWST-style) can exist without it, so the asset-type distinction cannot be validated against real data until this lands. Do this first, even before the `CampaignRun` schema migration, since it touches a different model and carries no risk to existing `CampaignRun` rows.
3. **`CampaignRun` window-field schema migration** (`window_start`/`window_end` + `window_needs_review`, replace natural-key constraint, data-migrate existing `obs_date` rows to `window_start=window_end=obs_date`). This is the single biggest-blast-radius change (touches `CampaignRunTable`, `ApprovalQueueTable`, `CampaignRunSubmissionForm`, `CampaignRunSubmissionView.form_valid()`, `import_campaign_csv.py`, `insert_or_create_campaign_run()`, `campaign_gap.py`) — land it as its own phase before anything downstream, per A4's checklist.
4. **Asset-aware gap analysis** (`campaign_gap.claimed_dates()` rewrite, A2) — depends only on #2 and #3. Can proceed in parallel with #5.
5. **CSV import range/TBD parsing** (`parse_obs_window` rewrite, A3) — depends only on #3. Can proceed in parallel with #4. Remember the paired-notebook convention for anything touching `import_campaign_csv.py`'s behavior.
6. **Site-disambiguation UI** (`fuzzy_match_observatories()`, `ApprovalQueueTable.render_site()` interactive branch, `CampaignRunSiteResolutionView`, `CampaignRunDecisionView.post()` guard fix) — has **no dependency on #2/#3/#4/#5** (it only touches `Observatory` resolution, not scheduling fields), so it can be built first, last, or in parallel with the window/asset work. The one hard internal-ordering rule: the B4 guard fix must ship in the same phase as B3's new endpoint, never split across phases.
7. **VIEW-05 submitter contact opt-in** — fully independent (one new form field + one `ALLOWED_FIELDS_FOR_NON_STAFF`-style conditional). No dependency on any of the above; good candidate for a low-risk first or last phase depending on scheduling preference.

Recommended parallelizable grouping if running multiple phases concurrently: {2, 3} must be sequential (2 before 3 is not a hard dependency but is lower-risk-first); {4, 5} can run in parallel once 3 lands; {6} and {7} can run any time, independent of everything else.

## Sources

- `solsys_code/models.py` (`CampaignRun`, `CalendarEventTelescopeLabel`)
- `solsys_code/solsys_code_observatory/models.py` (`Observatory`, `OBSTYPE_CHOICES`, `SATELLITE_OBSTYPE`)
- `solsys_code/campaign_gap.py` (coverage-gap computation core)
- `solsys_code/campaign_utils.py` (`resolve_site`, `parse_obs_window`, `insert_or_create_campaign_run`)
- `solsys_code/campaign_views.py` (`CampaignRunTableView`, `ApprovalQueueView`, `CampaignRunDecisionView`, `CampaignGapAnalysisView`)
- `solsys_code/campaign_tables.py` (`CampaignRunTable`, `ApprovalQueueTable`)
- `solsys_code/campaign_forms.py` (`CampaignRunSubmissionForm`, `CampaignGapAnalysisForm`)
- `solsys_code/telescope_runs.py` (`sun_event`, `get_site`, `SITES`)
- `.planning/PROJECT.md` (v2.1 milestone scope, v2.0 shipped decisions log)
- Local environment check: `pip show rapidfuzz` (confirmed transitive-only via `poetry`/`cleo`, not a FOMO dependency)

---
*Architecture research for: FOMO campaign-coordination v2.1 (range/window scheduling, asset-type gap analysis, site-disambiguation UI)*
*Researched: 2026-07-05*
