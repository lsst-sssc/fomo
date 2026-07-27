# Architecture Research

**Domain:** Django/TOM Toolkit consolidation — canonical run record + idempotent calendar reconciler
**Researched:** 2026-07-26
**Confidence:** HIGH (grounded entirely in direct reads of `solsys_code/models.py`, `campaign_views.py`, `calendar_utils.py`, `campaign_gap.py`, `campaign_utils.py`, `telescope_runs.py`, `management/commands/backfill_range_calendar_events.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`, and `tom_observations.facility.BaseObservationFacility` in the installed venv — no external/web sources needed for this question)

This file supersedes the previous contents (dated 2026-07-05, about the v2.1 "Uncertain Scheduling &
Site Disambiguation" milestone) — that milestone shipped. This is a full rewrite scoped to the v2.2
"One Canonical Run Record" milestone: where the new companion-record generalization, `source`/
`telescope_class` fields, `ObservationRecord` linkage, and the four-stage reconciler integrate with
the `CampaignRun`/`campaign_views`/`calendar_utils` infrastructure v2.0-v2.1 already shipped.

## Standard Architecture

### Existing Layering (as built, verified by reading imports)

```
┌───────────────────────────────────────────────────────────────────────┐
│  VIEW LAYER (Django views — imports the logic layer, never the reverse)│
│  campaign_views.py                                                     │
│    - imports: calendar_utils, campaign_filters, campaign_forms,        │
│      campaign_gap, campaign_tables, campaign_utils, mixins, models,    │
│      telescope_runs                                                    │
│    - does NOT import solsys_code.views / solsys_code.ephem_utils       │
│      (explicit in its own module docstring — SPICE-avoidance contract) │
├───────────────────────────────────────────────────────────────────────┤
│  LOGIC LAYER ("campaign_*.py" pure-logic modules, no Django view deps) │
│  calendar_utils.py   campaign_gap.py   campaign_utils.py               │
│    - insert_or_create_  - claimed_dates()  - resolve_site()            │
│      calendar_event()   - observable_dates() - parse_obs_window()      │
│    - _extract_instrument - get_or_compute_gap() - build_site_candidates│
│  telescope_runs.py (Stage 1 foundation: SITES, get_site(), sun_event())│
├───────────────────────────────────────────────────────────────────────┤
│  MANAGEMENT COMMANDS (import the logic layer directly — EXCEPT ONE)    │
│  load_telescope_runs.py, sync_lco_observation_calendar.py,             │
│  sync_gemini_observation_calendar.py, import_campaign_csv.py           │
│    → import calendar_utils / campaign_utils. Correct pattern.          │
│  backfill_range_calendar_events.py                                     │
│    → `from solsys_code.campaign_views import _project_calendar_event`  │
│    ⚠ THE ONE VIOLATOR: a management command importing a private        │
│      (`_`-prefixed) symbol from a Django VIEWS module.                 │
├───────────────────────────────────────────────────────────────────────┤
│  MODEL LAYER                                                           │
│  models.py: CalendarEventTelescopeLabel (1:1 sidecar on CalendarEvent),│
│  CampaignRun (window_start/end, site FK, approval_status/run_status)   │
│  Third-party: tom_calendar.CalendarEvent, tom_observations.            │
│  ObservationRecord — FOMO can only extend these via its own sidecar/   │
│  link models, never by editing the pip-installed model classes.        │
└───────────────────────────────────────────────────────────────────────┘
```

**Key structural fact confirmed by reading `campaign_gap.py`'s and `campaign_views.py`'s own module docstrings:** this codebase already has an established, working convention for exactly this kind of shared "logic core" module — `campaign_gap.py` states verbatim: *"a pure-logic helper module with no view/request concerns... must never import the heavy SPICE-loading ephemeris module... at module scope"* and is imported directly by `campaign_views.py` (`from .campaign_gap import clamp_date_range, get_or_compute_gap`). `campaign_utils.py` plays the identical role for site resolution / window parsing. The reconciler should be the third member of this family, not a private helper trapped inside `campaign_views.py`.

### Component Responsibilities (existing, verified)

| Component | Responsibility | File |
|-----------|----------------|------|
| `CampaignRun` | Canonical run record: campaign/target FKs, `telescope_instrument`, `site` FK (nullable), `window_start`/`window_end`, `approval_status`/`run_status` | `solsys_code/models.py:31` |
| `CalendarEventTelescopeLabel` | 1:1 sidecar on `CalendarEvent` (`OneToOneField(primary_key=True)`), today only `is_verified` | `solsys_code/models.py:8` |
| `_project_calendar_event()` | Builds and writes one `CalendarEvent` per night/day for a run (ground: per-night `sun_event()`; satellite: whole-window span). Raises `ValueError` on `sun_event()` failure (CR-01 contract) | `campaign_views.py:404` |
| `_calendar_event_title()` | Single source of truth for event title text (base + window suffix) | `campaign_views.py:392` |
| `_set_run_status()` | Updates every `CalendarEvent` whose `url` matches `CAMPAIGN:{pk}` or `CAMPAIGN:{pk}:*` when a run is marked cancelled/weathered | `campaign_views.py:742` |
| `insert_or_create_calendar_event()` | No-churn create-or-update on an explicit `lookup` dict; used by all 4 writers today (classical, LCO, Gemini, campaign projection) | `calendar_utils.py:318` |
| `sun_event(site, date, kind)` | Dip-corrected sunset/sunrise (`kind='sun'`) or -15° dark window (`kind='dark'`); raises `ValueError` if `site.timezone` unset or no 2 crossings | `telescope_runs.py:251` |
| `claimed_dates()` | Reads `CampaignRun` only (window-based, asset-aware ground/satellite split); ignores `CalendarEvent`/`ObservationRecord` entirely | `campaign_gap.py:116` |
| `backfill_range_calendar_events.py` | One-off command; per-run `.exists()` query in a Python loop (N+1 by construction); `.exclude(window_start=F('window_end'))` structurally skips every single-night run | `management/commands/backfill_range_calendar_events.py` |

## Recommended Integration (v2.2)

### New vs. Modified Components — explicit

**New:**
- `solsys_code/campaign_reconciler.py` — new pure-logic module, peer to `campaign_gap.py`/`campaign_utils.py`. Houses the four-stage pipeline: stage-decision function, per-stage window functions, the write path (delegates to `insert_or_create_calendar_event()`), and the bulk query/orchestration functions (`reconcile_run()`, `reconcile_runs()`). **Never imports `solsys_code.views` or `solsys_code.ephem_utils`** — same contract `campaign_gap.py`/`campaign_views.py` already state and test for (see Anti-Patterns/Sources below). Add a `TestNoHeavyEphemerisImport`-style static source-grep guard test mirroring `solsys_code/tests/test_campaign_gap.py:604`.
- `solsys_code/management/commands/reconcile_campaign_runs.py` — new management command; thin CLI wrapper (`--dry-run`, optionally `--run <pk>`) around `campaign_reconciler.reconcile_runs()`. Replaces `backfill_range_calendar_events.py`.
- Generalized companion model (rename/extend `CalendarEventTelescopeLabel`) — same table, additive nullable `run = models.ForeignKey(CampaignRun, null=True, blank=True, on_delete=models.SET_NULL, related_name='calendar_events')`. The event↔companion relation itself **stays `OneToOneField`** — it is the `run` FK on the companion, not the event↔companion cardinality, that turns "1 run → many events" real (many companion rows, one per event, can point at the same run). This is a low-risk additive-column migration, not a primary-key change.
- `CampaignRun.source` (`TextChoices`: web submission / classical file / LCO queue / Gemini queue / CSV import) and `CampaignRun.telescope_class` (`2m0`/`1m0`/`0m4`, nullable) fields.
- `CampaignRun.observation_records` — `ManyToManyField(ObservationRecord, blank=True, related_name='campaign_runs')` (per PROJECT.md's own stated design: "most likely a many-to-many declared on `CampaignRun`", since `ObservationRecord` is third-party and can't carry the FK itself).
- Attribution surface (staff-facing "suggested associations" queue) — new view(s)/table analogous to the existing `ApprovalQueueView`/"Sites Needing Review" pattern already in `campaign_views.py`.

**Modified:**
- `campaign_views.py`: `_project_calendar_event()` and the calendar-sync loop inside `_set_run_status()` are **deleted**; both the `approve`/`resolve_site` POST branches and `mark_cancelled`/`mark_weather_failure` call `campaign_reconciler.reconcile_run(run)` instead. `_calendar_event_title()` either moves into `campaign_reconciler.py` (preferred — it's pure title logic, no request/view concerns) or stays and is imported by the reconciler; either is fine, but it must have exactly one home, not two divergent copies (this is the exact class of bug CR-01/Pitfall-1 comments throughout this file already warn against).
- `solsys_code/models.py`: `CampaignRun` gains `source`/`telescope_class`/`observation_records`; `CalendarEventTelescopeLabel` gains `run` (and probably gets renamed — see the milestone's "closes the pending 2026-07-02 naming todo" note; a rename is a `SeparateDatabaseAndState` migration concern, not a blocker, but do it in the same migration as the `run` FK addition to avoid two migrations touching the same table for the same conceptual change).
- `campaign_gap.py`: **not modified in v2.2** (deliberately deferred — see Scaling/Anti-Patterns below). Must not be broken by the schema changes: `claimed_dates()` reads only `window_start`/`window_end`/`site`/`target`/`approval_status`/`run_status`, none of which change shape in v2.2.
- `management/commands/backfill_range_calendar_events.py`: retired once `reconcile_campaign_runs.py` covers its cases (the milestone explicitly says the reconciler "retir[es] the backfill-command-per-gap pattern and `backfill_range_calendar_events` with it").

## Architectural Patterns

### Pattern 1: Logic-layer module, not a views-module helper

**What:** Shared business logic that must be callable from both a management command and a Django view lives in a standalone `campaign_*.py` module with zero Django-request/response concerns, imported *by* the views module — never the other way around.

**When to use:** Any time a symbol needs two call sites where one is a management command. `backfill_range_calendar_events.py`'s `from solsys_code.campaign_views import _project_calendar_event` is the counter-example to fix, not a pattern to extend.

**Why this resolves the "circular import" framing of the question:** there is no actual circularity risk once the reconciler lives in `campaign_reconciler.py`, because the dependency graph is already a DAG in this codebase (`campaign_views.py` → `{calendar_utils, campaign_gap, campaign_utils, telescope_runs}`, never the reverse). Adding `campaign_reconciler.py` as one more logic-layer module that `campaign_views.py` imports, and that management commands *also* import directly, keeps the graph a DAG:

```
campaign_reconciler.py  <-- campaign_views.py   (view calls reconciler)
        ^
        +----------------  reconcile_campaign_runs.py  (command calls reconciler)
```

No edge from `campaign_reconciler.py` back to `campaign_views.py` is ever needed — the reconciler doesn't need anything view-specific (no request, no messages framework, no redirect). The two current view-side error-handling differences (`approve()` swallows a `sun_event()` `ValueError` and keeps the approval; `resolve_site()` does not revert but leaves `site_needs_review=True` on failure) stay in `campaign_views.py` as thin try/except wrappers *around* calls to the reconciler's pure functions — the reconciler itself should raise, not decide UI-facing recovery behavior, exactly as `_project_calendar_event()` already documents it does today (its docstring: "this helper does NO error-handling of its own for genuine failures; callers own revert-vs-non-revert behavior").

**Example (shape, not final code):**
```python
# solsys_code/campaign_reconciler.py
def reconcile_run(run: CampaignRun) -> ReconcileResult:
    """Idempotent: (re)computes and writes every CalendarEvent this run should have,
    for its current stage. Never imports solsys_code.views/.ephem_utils."""
    ...

# campaign_views.py
from .campaign_reconciler import reconcile_run
...
try:
    reconcile_run(run)
except ValueError:
    ...  # existing approve()-specific swallow, unchanged in spirit

# management/commands/reconcile_campaign_runs.py
from solsys_code.campaign_reconciler import reconcile_runs
```

### Pattern 2: Four-stage window pipeline as pure functions over already-loaded data

**What:** One function per stage that takes already-fetched Python objects (never issues its own query) and returns an event window (or `None` if that stage doesn't apply), plus one dispatcher that picks the highest applicable stage per run/night.

**Decomposition, grounded in what already exists:**

| Stage | Condition (from `run`/linked data, no new query) | Function | Reuses |
|---|---|---|---|
| 1 | `run.site` set (a specific `Observatory`, not class-wide) | `_site_window(run.site, night) -> (start, end)` | `sun_event(site, night, kind='sun')` — **identical** to the existing ground branch inside `_project_calendar_event()` (lines 468-486 today); this is a lift-and-shift, not new logic. |
| 2 | `run.site is None and run.telescope_class` set | `_class_wide_window(day) -> (start, end)` | New, trivial: `datetime.combine(day, time(0,0), utc)` .. `datetime.combine(day, time(23,59), utc)` — same idiom `_project_calendar_event()` already uses for the satellite branch (line 442-443), just for a day instead of a whole window span. |
| 3 | A linked `ObservationRecord` exists whose window overlaps `night`, and it is not yet in a facility terminal-success state | `_record_window(record) -> (start, end)` | `record.scheduled_start`/`scheduled_end` if set, else `record.parameters['start']`/`['end']` — this is exactly `sync_lco_observation_calendar._time_window()` and `sync_gemini_observation_calendar`'s window derivation generalized; since `ObservationRecord` fields are facility-agnostic (`scheduled_start`, `scheduled_end`, `parameters`), one function suffices across LCO/Gemini/(future ESO). |
| 4 | The linked record's `status` is a **success** terminal state | reuses stage 3's window function (same field), plus a status→"COMPLETED" title/marker | `record.status == 'COMPLETED'` is the convention `sync_lco_observation_calendar.py`'s TERM-01 logic already special-cases (clean title on `COMPLETED`, `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` prefixes on the other terminal states from `BaseObservationFacility.get_terminal_observing_states()`). **Caveat, verified by reading `tom_observations/facility.py`:** `get_terminal_observing_states()` is a per-facility abstract method — the exact string vocabulary is *not* unified across LCO/Gemini/ESO today. CLAUDE.md already flags unifying the three status vocabularies as explicitly deferred to v2.3. Do not build a general "success" classifier in v2.2 beyond `status == 'COMPLETED'`, which is the one value observed to already work for LCO; treat any other terminal status as "stage 3, not stage 4" and leave finer-grained mapping to v2.3. |

**Dispatcher:** `_stage_for(run, night, linked_record) -> int` — pure `if`/`elif` over already-loaded attributes, no query. Per-run, per-night the dispatcher picks stage 4 > 3 > 2 > 1 (highest applicable), matching the milestone's framing ("a classical TAC-awarded run simply stops at stage 1 because it never acquires records").

**Trade-off:** stage 3/4's per-night matching (which `ObservationRecord`, if the run has several, belongs to which night) is new logic that doesn't exist today (today's LCO/Gemini sync commands each own exactly one record → one event, with no "which night of a multi-night run" question). Keep this matching rule simple and explicit in code (e.g. "the record's own scheduled/parameters window's date falls on `night`"), and treat ambiguous cases (a record spanning multiple nights of a range run) as a known v2.2-scope decision to make explicitly during planning, not an implicit behavior.

### Pattern 3: Bulk query strategy to avoid the current N+1

**What people do today (the pattern to fix):** `backfill_range_calendar_events.py`'s `handle()` loops over `candidates` and, **inside the loop**, issues `CalendarEvent.objects.filter(Q(url=...) | Q(url__startswith=...)).exists()` per run — one query per candidate run, i.e. N+1 by construction (visible at `management/commands/backfill_range_calendar_events.py:66-68`).

**What the reconciler should do instead — 3 queries total, independent of run count, for the "which runs need which events" decision phase (the per-event *write* itself is necessarily one query per created/updated `CalendarEvent`, same cost the codebase already accepts everywhere else via `insert_or_create_calendar_event()`):**

```python
# Query 1: candidate runs, with site/campaign already joined (select_related — no N+1
# for run.site.observations_type / run.site.timezone / run.campaign.name reads later).
runs = (CampaignRun.objects
        .filter(approval_status=CampaignRun.ApprovalStatus.APPROVED)
        .exclude(run_status__in=_EXCLUDED_RUN_STATUSES)   # mirror campaign_gap's set
        .select_related('site', 'campaign'))

# Query 2: bulk-prefetch every linked ObservationRecord for those runs in one extra
# query (Django's prefetch_related issues a single WHERE ... IN (...) for the M2M
# through-table), not one query per run.
runs = runs.prefetch_related('observation_records')

# Query 3: bulk-fetch every companion row (and its CalendarEvent) already linked to
# any of these runs, in ONE query, then group by run_id in Python.
from collections import defaultdict
companions = (CalendarEventCompanion.objects
              .filter(run_id__in=[r.pk for r in runs])
              .select_related('event', 'run'))
existing_by_run = defaultdict(list)
for c in companions:
    existing_by_run[c.run_id].append(c)
```

The stage/window decision and the "does this run+night already have an event" check then both run entirely against in-memory data (`runs`, the prefetched `.observation_records.all()`, and `existing_by_run[run.pk]`) — no query inside the per-run/per-night loop except the unavoidable `insert_or_create_calendar_event()` write itself. This is a strict improvement over today's per-run `.exists()` call and is the direct fix for the literal N+1 pattern named in the question.

**One nuance worth flagging for the phase planner:** query 3 above only finds events already linked via the *new* companion `run` FK. It will **not** find the pre-existing, unlinked LCO/Gemini/classical `CalendarEvent`s for the same nights (the Didymos pk=1 double-representation case) — by design, per PROJECT.md ("These are not duplicates to be merged... The fix is attribution, not deduplication"). The reconciler is correct to be blind to those until attribution links them; do not add a second, url-string-based existence check into the reconciler to "catch" them — that would resurrect the exact fragile string-matching (`CAMPAIGN:{pk}` / `CAMPAIGN:{pk}:{date}`) this milestone is trying to retire in favor of the FK.

## Data Flow

### Reconciler Invocation (both call sites end up in the same function)

```
CampaignRunDecisionView.post()            reconcile_campaign_runs (mgmt command)
  (approve / resolve_site /                        |
   mark_cancelled / mark_weather_failure)           |
        |                                           |
        +--------------+----------------------------+
                        v
          campaign_reconciler.reconcile_run(run)
                        |
        +---------------+--------------------+
        v               v                    v
  _stage_for()   _site_window()/       insert_or_create_calendar_event()
  (pure, no      _class_wide_window()/        |
   query)        _record_window()             v
                 (pure, uses already-   CalendarEvent created/updated +
                  loaded run/record      companion row (is_verified, run=run.pk)
                  attributes)            created/updated
```

### Key Data Flows

1. **Live path (new/changed run):** staff approves or resolves a site → `reconcile_run(run)` runs the four-stage dispatcher for every night in `[window_start, window_end]`, writes/updates events via `insert_or_create_calendar_event()`, and stamps the companion row's `run` FK — so a run created by *any* path (not just approve/resolve_site) becomes visible by simply being picked up by the next `reconcile_campaign_runs` sweep, closing the "visible by construction rather than by remembering to run the right backfill command" goal stated in PROJECT.md.
2. **Batch path (sweep):** `reconcile_campaign_runs` management command runs the same 3-query bulk fetch + per-run dispatch over every eligible `CampaignRun`, replacing the narrow `.exclude(window_start=F('window_end'))` filter that currently makes `backfill_range_calendar_events.py` invisible to single-night runs (the concrete defect PROJECT.md names: "its dry-run reports 1 candidate across the whole database").
3. **Attribution path (separate, human-in-the-loop):** a staff-facing surface (new, modeled on the existing `ApprovalQueueView`'s "Sites Needing Review" table pattern) surfaces suspected pre-existing `CalendarEvent`/`ObservationRecord` matches for a run (e.g. window/site/telescope overlap heuristics — ideally reusing the reconciler's own `_site_window`/`_record_window` overlap logic as the *suggestion* engine, so match logic is written once) and writes the companion `run` FK / `observation_records` M2M only on explicit staff confirmation — never a silent merge, per the milestone's own constraint.

## Scaling Considerations

Not a user-scale concern (FOMO is a small internal/community coordination tool) — the relevant "scale" axis here is **number of `CampaignRun`s × nights per sweep**, and query count per sweep, not concurrent users.

| Scale | Current backfill command | Reconciler (v2.2) |
|---|---|---|
| ~20-50 runs (today's real dev-DB scale) | ~50 `.exists()` queries + per-run event writes | 3 bulk queries + per-event writes (same write cost, structural query-count fix) |
| A future full re-ingest sweep (all 4 adapters writing runs instead of events, v2.3 scope) | N/A — command doesn't cover this today | Same 3-query shape scales linearly in run count for the read side; write side is inherently O(events), matching every other writer in this codebase (`insert_or_create_calendar_event()` has no batch-write variant anywhere yet) |

### Scaling Priorities

1. **First real bottleneck, if it ever appears:** the per-event write in `insert_or_create_calendar_event()` is one query per event (a `get_or_create` or a proximity-window `filter().first()`), same as every existing writer. This was already an accepted trade-off in v1.2-v2.1 and is not something v2.2 should try to batch — doing so would be a much larger, riskier rewrite of shared write-path code well outside this milestone's stated scope.
2. **Second:** if `reconcile_campaign_runs` is ever run unconditionally over the full history (not just approved/active runs), narrow the query-1 filter (e.g. exclude terminal `run_status` states, as `campaign_gap._EXCLUDED_RUN_STATUSES` already does) so the sweep doesn't re-touch settled historical rows every time — this is a filter tweak, not an architecture change.

## Anti-Patterns

### Anti-Pattern 1: Management command importing a views-module private symbol

**What people did:** `backfill_range_calendar_events.py: from solsys_code.campaign_views import _project_calendar_event` — a management command reaching into a Django views module and importing an underscore-prefixed (explicitly-private) function.

**Why it's wrong:** (a) layering violation — management commands and views are peer consumers of logic, not consumers of each other; (b) the leading underscore is this codebase's own signal that the symbol has no external-stability contract (its docstring is written in "extracted from the approve branch" terms, i.e. it documents itself as an implementation detail of that view); (c) it made retiring/changing `_project_calendar_event()`'s signature a two-file concern instead of a one-file concern the day this milestone needs to change it.

**Do this instead:** define the shared function in a logic-layer module (`campaign_reconciler.py`), have `campaign_views.py` import it (view depends on logic — correct direction), and have every management command import it from the same place. This is exactly the pattern `campaign_gap.py`/`campaign_utils.py` already established and that `campaign_views.py` already follows for those two modules.

### Anti-Pattern 2: Treating `site=None` as one condition instead of two

**What people did:** today, `_project_calendar_event()` checks `if not (run.telescope_instrument and run.site and run.window_start and run.window_end): return False` — a class-wide allocation and a genuinely-unresolved site are structurally indistinguishable (both `site=None`), so a class-wide run currently gets **zero** calendar presence, silently.

**Why it's wrong:** this is precisely the ambiguity PROJECT.md calls out as the reason `telescope_class` needs to exist as its own field, and it's why stage 2 of the pipeline can't be built on the current schema at all.

**Do this instead:** `telescope_class` must land (schema phase) before stage 2 can be written; the dispatcher then branches on `run.site is not None` (stage 1) vs. `run.site is None and run.telescope_class` (stage 2) vs. neither (genuinely unresolved — no projection, same as today, `site_needs_review` stays the correct signal).

### Anti-Pattern 3: Reconciler re-deriving the no-churn write contract

**What people might be tempted to do:** write a new create-or-update helper inside `campaign_reconciler.py` because the four-stage pipeline's write shape ("narrow the window as more info arrives") feels different from the existing writers'.

**Why it's wrong:** `insert_or_create_calendar_event()` already handles exactly this — SYNC-02→SYNC-03's "banner narrows to placed block" transition in `sync_lco_observation_calendar.py` is the *same* narrowing shape as stage-2→stage-3→stage-4 (each stage just supplies a tighter `fields['start_time']`/`fields['end_time']` and calls the same lookup key). The milestone text itself says as much: "`sync_lco_observation_calendar` already implements stages 3→4... this milestone makes that the general mechanism."

**Do this instead:** the reconciler's job per run/night is to compute `(lookup, fields)` for the current stage and hand it to the existing `insert_or_create_calendar_event()` unchanged — zero new write-path code.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `campaign_views.py` ↔ `campaign_reconciler.py` (new) | Direct Python import, function call | View owns POST validation, messaging, and the approve-swallow/resolve_site-no-revert error-handling asymmetry; reconciler owns pure stage/window logic and the write. |
| `reconcile_campaign_runs.py` (new command) ↔ `campaign_reconciler.py` | Direct Python import, function call | Command owns CLI args (`--dry-run`, output formatting); never talks to `campaign_views.py`. |
| `campaign_reconciler.py` ↔ `calendar_utils.insert_or_create_calendar_event()` | Direct Python import, function call | Reused unchanged — no new write-path code (Anti-Pattern 3). |
| `campaign_reconciler.py` ↔ `telescope_runs.sun_event()` | Direct Python import, function call | Reused unchanged for stage 1 — identical to `campaign_gap.py`'s existing dependency. |
| `campaign_reconciler.py` ↔ `CampaignRun`/companion model/`ObservationRecord` | Django ORM, bulk `select_related`/`prefetch_related` (see Pattern 3) | The one genuinely new query-shape work in this milestone. |
| `campaign_gap.claimed_dates()` ↔ v2.2 schema changes | None required in v2.2 | Reads only pre-existing `CampaignRun` fields; must keep working unmodified through the migration (verified: none of its read fields — `window_start`/`window_end`/`site`/`target`/`approval_status`/`run_status` — change shape). |

### External Services (third-party models FOMO cannot modify directly)

| Model | Integration Pattern | Notes |
|---|---|---|
| `tom_calendar.CalendarEvent` | FOMO-side companion row (`CalendarEventTelescopeLabel` → generalized, `run` FK) | Already the established pattern (v1.4); v2.2 just adds a field to the existing sidecar, doesn't invent a new mechanism. |
| `tom_observations.ObservationRecord` | FOMO-side `ManyToManyField` on `CampaignRun` | No existing FOMO-side link model for `ObservationRecord` to extend (unlike `CalendarEvent`, which already had `CalendarEventTelescopeLabel`) — this is genuinely new, not a generalization of something existing. |
| `tom_observations.facility.BaseObservationFacility.get_terminal_observing_states()` | Read-only, per-facility abstract method | Confirms status vocabularies are NOT unified across facilities today — bounds how ambitious stage 4's "success" detection can safely be in v2.2 (see Pattern 2, stage 4 row). |

## claimed_dates() — what must not be precluded (v2.3 deferred, verify now)

`claimed_dates()` (`campaign_gap.py:116`) is explicitly out of scope for v2.2 (PROJECT.md: "making coverage-gap analysis provenance-blind" is deferred to v2.3). What the v2.2 schema/reconciler design must NOT foreclose for that later work:

1. **Provenance-blind future join.** v2.3 will presumably want `claimed_dates()` to also count `CalendarEvent`s that have no `CampaignRun` at all (the 20 pre-existing Didymos events) alongside `CampaignRun`-derived dates. That requires querying through the companion model's `run` FK (`CalendarEventCompanion.objects.filter(run__isnull=True, ...)` for "unattributed" events, or `run__campaign=...` for attributed ones). **v2.2 must give this FK a normal indexed column** (Django FKs are indexed by default — no special action needed beyond not overriding `db_index=False`).
2. **Class-wide (`telescope_class`, `site=None`) runs are currently invisible to `claimed_dates()`.** Its query is `CampaignRun.objects.filter(campaign=campaign, site=site, ...)` — a run with `site=None` (stage-2 class-wide) will never match any concrete `site` argument. This is a real, foreseeable v2.3-scope gap, but v2.2 doesn't need to fix it — it only needs to avoid making `site` non-nullable or otherwise removing the class-wide representation's ability to exist. Confirmed: `site` stays nullable; `telescope_class` is purely additive. No structural change needed in v2.2, just don't let phase planning quietly make `site` required.
3. **`source`** should be readable by a future `claimed_dates()` without needing a schema change — it's a plain `CharField`/`TextChoices`, trivially filterable (`.exclude(source=CampaignRun.Source.CSV_IMPORT)` etc. if v2.3 wants provenance-based weighting). No action needed beyond choosing a `TextChoices` (not a free-text field) so future filtering is exact-match safe.

No `claimed_dates()` code changes are required in v2.2; the check above is "does the v2.2 schema keep the door open," and the answer is yes for all three items given the field choices already described.

## Build Order

Real blocking dependencies only (not a preferred narrative):

```
0. SPIKE (investigation, no code)
   Settles: source's TextChoices values + natural-key implications, how each
   adapter's existing identity key (5-min-tolerance start_time / LCO url /
   GEM:{prog}/{obsid} / CAMPAIGN:{pk}[:date]) maps onto a CampaignRun, and the
   migration + attribution strategy for pre-existing rows.
   BLOCKS everything below -- every migration's backfill logic and the
   attribution surface's matching rules depend on decisions made here.
        |
        v
1. SCHEMA PHASE (3 additive migrations; no ordering dependency AMONG
   themselves, but all depend on step 0's decisions)
   1a. CampaignRun.source / CampaignRun.telescope_class
   1b. Companion-record generalization: rename CalendarEventTelescopeLabel
       (if the spike decides to) + add nullable `run` FK. Additive column,
       NOT a primary-key change (event<->companion stays OneToOne) -- low risk.
   1c. CampaignRun.observation_records (M2M to ObservationRecord)
        |
        v
2. RECONCILER CORE (campaign_reconciler.py)
   Stage-decision dispatcher + 3 window functions (site/class-wide/record) +
   bulk query strategy (Pattern 3) + write path (delegates to existing
   insert_or_create_calendar_event(), no new write code).
   BLOCKED BY: 1a (needs telescope_class for stage 2), 1b (needs the run FK
   for its bulk existence query and its write-time link), 1c (needs the M2M
   for stage 3/4). NOT blocked by attribution (item 4 below).
        |
        +------------------------------+
        v                              v
3. WIRE INTO CALLERS               4. ATTRIBUTION SURFACE
   campaign_views.py's approve/        Staff-facing suggested-match queue,
   resolve_site/mark_cancelled/        modeled on the existing "Sites
   mark_weather_failure branches       Needing Review" pattern.
   call reconcile_run(); new           BLOCKED BY: 1b + 1c (schema to write
   reconcile_campaign_runs command     into) and step 0 (matching strategy).
   wraps reconcile_runs().             BENEFITS FROM (soft dependency, not a
   BLOCKED BY: 2.                      hard block) step 2 existing first, so
        |                              its "suggest a match" logic reuses the
        v                              reconciler's window-overlap functions
5. RETIRE OLD CODE                     instead of duplicating them.
   Delete _project_calendar_event(),
   _set_run_status()'s manual sync
   loop, and backfill_range_calendar_
   events.py.
   BLOCKED BY: 3 (old code paths must
   be fully replaced first).
```

**Operational note (rollout order, not a code dependency):** run the attribution pass (item 4) against the pre-existing Didymos/LCO/classical events *before or alongside* the first full `reconcile_campaign_runs` sweep over historical data. The reconciler is correct to be blind to unlinked pre-existing events (Pattern 3's nuance) — but that means an unattributed first sweep will create a fresh `CAMPAIGN:{pk}:{date}` event for nights that already have a "real" adapter-sourced event, producing visible double-booking-looking entries on the calendar until attribution links them. This doesn't block writing the reconciler; it's a rollout-sequencing recommendation for whoever runs the first production sweep.

## Sources

- Direct reads of `solsys_code/models.py`, `solsys_code/campaign_views.py`, `solsys_code/calendar_utils.py`, `solsys_code/campaign_gap.py`, `solsys_code/campaign_utils.py`, `solsys_code/telescope_runs.py`, `solsys_code/management/commands/backfill_range_calendar_events.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/management/commands/sync_gemini_observation_calendar.py`, `solsys_code/tests/test_campaign_gap.py` (existing SPICE-import-guard test precedent) — all in this repository, read 2026-07-26.
- `.planning/PROJECT.md` "Current Milestone: v2.2 One Canonical Run Record" section (goal, four-stage pipeline table, key context, deferred-to-v2.3 list).
- `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/facility.py` (`BaseObservationFacility.get_terminal_observing_states()`) — confirms per-facility status vocabularies are not unified, bounding stage 4's design.
- `/home/tlister/git/fomo_devel/CLAUDE.md` — SPICE heavy-import constraint (verbatim: "Heavy import side effect... importing `solsys_code.ephem_utils`... runs `fomo_furnish_spiceypy()`... ~1.6 GB").

---
*Architecture research for: FOMO v2.2 "One Canonical Run Record" milestone*
*Researched: 2026-07-26*
