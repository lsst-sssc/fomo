# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps - Research

**Researched:** 2026-09-18
**Domain:** Django display-layer consolidation (status vocabulary), read-only public aggregation queries (tallies), and a coverage-gap query extension — no new external service integration except one read from the already-integrated LCO Observation Portal API.
**Confidence:** HIGH for the in-repo code grounding (every claim below was read from the actual source this session); MEDIUM for the LCO portal `timeallocation_set` API shape (official docs + example script, not tool-verified against live FOMO credentials).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Status vocabulary (STATUS-01/02)**
- **D-01: Short-letter markers everywhere, final.** Phase 34's provisional `[Q]` queued, `[S]` scheduled, `[O]` observed, `[X]` window expired, `[C]` cancelled, `[F]` failed, `[?]` inconsistent record become the final vocabulary, and the run-level `[CANCELLED]` / `[WEATHERED]` prefixes migrate to the same short form. Every title prefix, the `status_border_css()` ring and the legend derive from one definition in one module — no second hand-maintained copy anywhere. — **Reversibility:** costly — re-titling is one sweep plus one allocation re-project, but every test that asserts on prefix strings (`test_campaign_approval.py` and the calendar template tests) and the runbook's documented prefixes change with it.
- **D-02: Run-level states share the letters.** A staff-cancelled run night is `[C]` (the same letter as a portal-cancelled record); a weathered / technical-failure run night is a new `[W]`. The pop-up's existing `Run status:` line still says which layer the cancellation came from. `[C]` therefore means "cancelled, by whoever owns this event".
- **D-03: `[S]` is called "Scheduled".** The placed-but-unobserved state (block scheduled, not yet observed — spike 002's vocabulary gap) is named with the LCO portal's own word, in the legend, the runbook and docstrings. "Placed" stays an internal code word only.
- **D-04: One legend lists every visible state**, including `[U]` unused awarded night (D-08, a render-time token that never appears in a stored title) and `[?]`, so the legend is the single explanation of everything a visitor can see on the calendar. The legend stays a fixed, ordered vocabulary read from the one module — never data-driven from the database.
- **D-05: The LCO/SOAR OCS vocabulary is the canonical state model; other facilities map onto it.** The classifier's states are the OCS ones (`PENDING`, a placed block, `COMPLETED`, `WINDOW_EXPIRED`, `CANCELED`, `FAILURE_LIMIT_REACHED`, `NOT_ATTEMPTED`). A facility whose TOM `get_terminal_observing_states()` does not fit is *mapped* onto those states by a small FOMO-side table rather than trusted: `tom_gemini` is a limited subset that only schedules disruptive ToOs, so its `TRIGGERED` / `ON_HOLD` "terminal" states mean *submitted*, never observed (and may change with future GPP support); ESO is expected to gain a real Phase 2 read-back vocabulary only far in the future. No facility ever classifies as observed by accident, and the `status == 'COMPLETED'` check in `calendar_utils.py` goes through the classifier.

**Public tallies (TALLY-01/02/03)**
- **D-06: Unused nights for queue-scheduled / class-wide runs come from the LCO portal, not a new manual field.** The portal proposal API's `timeallocation_set` (allocated hours minus used hours, summed over the proposal's allocations) is divided by the fixed rule of thumb **10 hours = 1 night** (the NOIRLab/LCO proposal convention) to give an *estimated* unused night count, labelled as an estimate. The figure is per *proposal* and is attached to every run carrying that proposal code. No staff book-keeping is added for anything the portal can answer. — **Reversibility:** reversible.
- **D-07: The proposal time allocation is fetched in the unattended runner tick and stored in a new small model keyed by proposal code** (planner's naming, e.g. `ProposalTimeAllocation(proposal_code, semester, instrument_type, allocated_hours, used_hours, fetched_at)`), as a new step in `solsys_code/unattended.py`'s `STEPS` (or part of the status-refresh step — planner's choice) with Phase 36's per-step failure isolation. Pages only ever read the stored figure, so an anonymous visitor never triggers a credentialed portal call, and a portal outage surfaces through the existing failure email / heartbeat, never as a blank public page. `WatchedProposal` is not overloaded — it stays a watch list. — **Reversibility:** costly.
- **D-08: One compact "Progress" cell per run row** on the campaign table (django-tables2 column in `campaign_tables.py`), rendering something like `2 groups · 14 records · [O] 5 [S] 2 [X/F] 1 [U] ≈3`, reusing the D-01 letters. Not six sortable columns. The counts must be computed for the whole table in one pass (one `annotate(Count(..., filter=Q(...)))`-style query where SQL can express it; the folded TTL-cache pattern where the site-local night rule cannot be pushed into SQL) — never a per-row loop.
- **D-09: "Run detail" means the calendar pop-up's attributed-run block.** The same tally is added to the `Attributed campaign run` block in `src/templates/tom_calendar/partials/event_form.html`, rendered read-only from `CalendarEventMeta.run` in the style of `campaign_decoration()`, under the same `run.is_publicly_visible` gate. No new run detail page or route.
- **D-10: Campaign roll-up = a header strip above the runs table plus a badge on the campaign list.** `campaignrun_table.html` gets a summary line (sums across the campaign's approved, publicly visible runs); `campaign_list.html`'s existing `N runs` badge gains e.g. `· 5 nights observed`. The unused-nights estimate is counted **once per distinct proposal code**, not once per run, in the roll-up.
- **D-11: Night counters follow the site-local observing night** (`telescope_runs.observing_night`, the same rule the Phase 35 handoff uses), never the UTC date, for every tally and for the gap claims below. Nights observed / scheduled / expired-or-failed are derived from the run's linked records (`CampaignRunObservation`) via the D-05 classifier; nights unused-so-far for an allocation run are the still-standing elapsed `ALLOC:` nights (D-12/D-15), and for a container run the D-06 estimate.
- **TALLY-03 guard (locked by roadmap, restated):** every tally is a read-only aggregate. No code path in this phase — receiver, runner step, view, template tag or migration — writes `CampaignRun.run_status`; it remains set only by the existing staff decision views.

**Unused awarded nights (UNUSED-01)**
- **D-12: Unused is derived at display time; nothing is written.** A template tag in `calendar_display_extras` (in the style of `campaign_decoration()`) classifies an `ALLOC:` event as unused when its night has ended (`end_time` — the projected sunrise — is in the past) and the run's `run_status` carries no `[C]`/`[W]` prefix. The allocation projector's no-churn contract is untouched, there is no time-dependent re-title, and a run's nights never flip one by one across ticks. — **Reversibility:** reversible.
- **D-13: Look = muted chip + a visible `[U]` token prepended at render.** The chip is greyed / desaturated (reduced opacity, dashed border — planner's exact CSS) **and** its rendered label reads `[U] NTT EFOSC2`; the token is added by the template, never stored. Two channels (style and text) so the state is never colour-alone and a screen reader hears it. Not a fourth ring colour.
- **D-14: Staff run status always wins; only truly empty nights read unused.** `[C]`/`[W]` from `run_status` beats unused. A night is unused only when its allocation event is still standing (no linked record retired it) and the night has ended. An *unattributed* observation event on the same site-night does not rescue it: that is an attribution-queue matter, not a display rule.
- **D-15: Unused counts in the tally and is filterable.** The allocation run's "unused so far" is the count of nights the D-12 rule would render as `[U]` — the table and the calendar must agree by construction (one shared classifier function). The legend's `[U]` entry is click-to-filter like the proposal swatches.

**Provenance-blind gap claims (GAPB-01)**
- **D-16: Only observed and scheduled blocks claim a night** — an `[O]` or `[S]` record's placed block, on the site-local observing night it falls in (D-11). A queued request's window claims nothing; expired / cancelled / failed records claim nothing.
- **D-17: Site assignment for an observation event:** the record's `observed_site` (`ObservationRecord.parameters`, Phase 34 D-09) mapped to its `Observatory`; else the site of the `CampaignRun` the event is attributed to; else the event is **not** assignable to a site — it is reported on the gap page as "claimed, site unknown" (a listed count, never silently dropped) and does not close any per-site gap.
- **D-18: "On the campaign calendar" = attributed to one of the campaign's runs (`CalendarEventMeta.run`) OR the record's target belongs to the campaign's `TargetList`.** The union is what makes the analysis provenance-blind.
- **D-19: Approved run windows still claim alongside observation blocks.** `claimed_dates()` keeps today's approved-run-window claims and adds the D-16 blocks; the gap page's claimed list may say which kind covered a night when both apply. The D-05 `_EXCLUDED_RUN_STATUSES` rule is unchanged.

### Claude's Discretion
- **Where the vocabulary lives and its name** — a peer module under `solsys_code/` (e.g. `status_vocabulary.py`), imported by the projector, the allocation projector, the reconciler and `calendar_display_extras`; it must never import `solsys_code.views` or `ephem_utils`.
- **Legacy title migration.** Recommended: a one-time re-title on the next unattended tick.
- **The exact "night has ended" instant** for D-12 — `end_time < now()` in UTC is the simple answer; a grace period is the planner's call.
- **How TALLY-03 is enforced** — at minimum a test proving no module in the phase writes `run_status`, plus the vocabulary module's docstring stating the rule.
- **What the tally shows before the first successful portal fetch** for a proposal (e.g. `unused: not yet fetched`), and how a proposal code is discovered for fetching (the code carried on the run and/or its linked events' `proposal` field — researcher confirms which field is authoritative after Phase 35's parser change). **Researcher finding (this session): neither field is currently populated for any active write path — see Open Questions.**
- **Gap cache invalidation** — the existing one-hour TTL and the page's "cached for one hour" note may stand; invalidating on projector narrowing is optional.
- **Portal API specifics** for `timeallocation_set` (endpoint, fields, semester scoping, credentials via the existing `LCOFacility` settings) — see "LCO Portal `timeallocation_set` API" below.

### Deferred Ideas (OUT OF SCOPE)
- A per-telescope-class hours-per-night table replacing the fixed 10 h rule (D-06 rationale).
- A written `[U]` marker in stored titles (for exports/API consumers) — D-12 chose display-time.
- A "same-site-night observation rescues an unused night" display rule — rejected in D-14 as masking missing attribution; the attribution queue is the fix.
- A public run detail page — D-09 chose the pop-up block; noted as a possible later surface.
- Roadmap-locked out of scope: automatic `run_status` aggregation, a run detail page, live Gemini read-back, ESO sync, any new writer of another layer's events.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STATUS-01 | One status vocabulary replaces the three parallel prefix maps, includes a placed-but-unobserved state | Confirmed exact location of all three maps (`observation_projector._STAGE_MARKER`/`_FAILURE_MARKER_BY_STATUS`, `campaign_reconciler.RUN_STATUS_CALENDAR_PREFIX`, `calendar_display_extras._TERMINAL_PREFIXES`/`_OBSERVATION_STATUS_LEGEND`) with verbatim quotes below; see Architecture Patterns |
| STATUS-02 | General terminal-state classifier replaces `status == 'COMPLETED'` | Confirmed the one real hardcoded check at `calendar_utils.py:336` inside `resolve_placement_block()`; `observation_projector.stage_for()` already has the per-facility `get_terminal_observing_states() - get_failed_observing_states()` logic to generalize from |
| TALLY-01 | Public per-run tally: groups, records, nights by state | `CampaignRunObservation.related_name='observation_links'` gives record count; `ObservationGroup.observation_records` (unnamed reverse `observationgroup_set`) gives group count; night classification needs `stage_for()` + `observing_night()` per linked record — see Code Examples |
| TALLY-02 | Campaign roll-up of the same tally | `CampaignRunTableView`/`CampaignListView` PII-gate pattern is the template to extend; roll-up is a header strip + badge, no new page |
| TALLY-03 | `run_status` never auto-derived | No current code path writes `run_status` outside admin/staff views — confirmed by grep; the guard is a new negative test, not a code change |
| UNUSED-01 | Unused night visibly distinct | `allocation_projector.allocation_events()`/`ALLOC:` namespace and `CalendarEvent.end_time` confirmed as the fields the D-12 tag needs |
| GAPB-01 | Coverage gap counts every observation | `campaign_gap.claimed_dates()` confirmed to query **only** `CampaignRun` window dates today — the exact gap D-16..D-19 close; PII-safe `.only()` pattern must be preserved |
</phase_requirements>

## Summary

This phase is almost entirely a **consolidation and read-time-aggregation** phase: no new
external package, no new outbound integration except one already-integrated portal (the LCO
Observation Portal API, which `tom_observations.facilities.lco.LCOFacility`/`OCSFacility`
already authenticate against for every other portal call in this codebase). The work is
concentrated in four areas, all grounded against the real source this session:

1. **One status-vocabulary module.** Three independent prefix vocabularies exist today and
   the codebase's own comments document that they "must stay byte-identical" to each other —
   `observation_projector.py`'s `_STAGE_MARKER`/`_FAILURE_MARKER_BY_STATUS`,
   `campaign_reconciler.py`'s `RUN_STATUS_CALENDAR_PREFIX` (re-imported by
   `allocation_projector.py`), and `calendar_display_extras.py`'s `_TERMINAL_PREFIXES`/
   `_OBSERVATION_STATUS_LEGEND`. Folding these into one module and routing
   `calendar_utils.resolve_placement_block()`'s hardcoded `'COMPLETED'` string through it is
   the whole of STATUS-01/02. `observation_projector.stage_for()` already implements the
   general classifier shape (`get_terminal_observing_states() - get_failed_observing_states()`);
   the new module should promote that shape to the canonical classifier and give the
   run-level (`CampaignRun.RunStatus`) and per-facility-mapping halves a home in the same file.

2. **Public, read-only tallies.** The record/group counts are a straightforward `Count()`
   annotation over `CampaignRunObservation`/`ObservationGroup`; the per-night state counts
   require iterating each run's linked `ObservationRecord`s through the new classifier and
   `telescope_runs.observing_night()` (the site-local-night rule already proven in Phase 35).
   Because this per-record iteration can't be pushed into SQL, it must follow the existing
   `campaign_gap.py` low-level-cache pattern (`django.core.cache`, TTL) rather than running
   once per row on every table page load — exactly the todo folded into D-08's decision text.

3. **The LCO portal `timeallocation_set` fetch.** Confirmed via official LCO developer docs
   and the LCOGT example-scripts repo (`[CITED]`, not tool-verified against live FOMO
   credentials): `GET /api/proposals/<id>/` (or the list endpoint `/api/proposals/`) returns a
   `timeallocation_set` array of `{std_allocation, std_time_used, semester, instrument_types,
   ...}` per (semester, instrument-type) allocation, authenticated the same way every other
   `LCOFacility`/`OCSFacility` portal call in this codebase already is
   (`Authorization: Token <api_key>` via `facility._portal_headers()`). **Important gap found
   this session:** neither `CampaignRun` nor `CalendarEvent.proposal` is currently populated
   with a proposal code for any *actively written* `CampaignRun` source
   (`WEB`/`CSV_IMPORT`/`CLASSICAL_FILE` — confirmed by exhaustive grep, zero hits for
   "proposal" in `campaign_forms.py` and `import_campaign_csv.py`). The classical loader's
   `[proposal]` token is parsed but only ever concatenated into the free-text
   `observation_details` field. This is a genuine open question for the planner (see Open
   Questions) — D-07's "how a proposal code is discovered" cannot simply read an existing
   field; a structured field must likely be added.

4. **Provenance-blind gap analysis.** `campaign_gap.claimed_dates()` is confirmed, by reading
   its full body, to query only `CampaignRun.objects.filter(campaign=..., site=...,
   approval_status=APPROVED).exclude(run_status__in=_EXCLUDED_RUN_STATUSES)` — it never touches
   `ObservationRecord` or `CalendarEventMeta` at all today. D-16..D-19 require adding a second
   claim source (observed/scheduled event nights, site-resolved per D-17, campaign-membership
   per D-18) unioned with the existing run-window claims, while preserving the function's
   PII-minimizing `.only('pk', 'window_start', 'window_end')` discipline.

**Primary recommendation:** Build the status vocabulary module first (STATUS-01/02) since every
other deliverable (tallies, unused-night tag, gap page's "which kind covered this night" label)
consumes its classifier and marker constants. Then TALLY-01/02/03 and UNUSED-01 together (they
share the per-record night classification and the `ALLOC:` namespace read). GAPB-01 can be
built in parallel since its data source (observation events + run windows) is independent, but
its "claimed, site unknown" listing and its "which kind covered a night" label should reuse the
same classifier and site-resolution helper the tally work builds.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Status vocabulary (markers, ring, legend) | Backend Server (display-time template tags + a peer `solsys_code/` module) | — | Pure Python classification consumed by Django templates; no browser-side logic |
| Public run/campaign tallies | Backend Server (view/template-tag aggregation) | Database (annotated queries) | Counts are computed server-side per request/cache-fill, never in the browser |
| LCO portal `timeallocation_set` fetch | Backend Server (unattended runner step) | Database (`ProposalTimeAllocation` model) | Credentialed portal call must never happen at request time (D-07); stored figure is what public pages read |
| Unused-night display | Backend Server (template tag, display-time) | Browser/Client (CSS chip styling, `[U]` legend click-to-filter JS) | Classification logic lives in Python; the visual/interactive presentation is the existing calendar JS filter pattern |
| Coverage-gap analysis | Backend Server (`campaign_gap.py` query extension) | Database (indexed queries over `CalendarEventMeta`/`ObservationRecord`) | Pure aggregation; `CampaignGapAnalysisView` already the single request-time entry point |

## Standard Stack

No new external package is required by this phase. Every library the work touches is already
an installed, in-use project dependency:

### Core (already installed — verified this session)
| Library | Installed Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django | project-pinned (via tomtoolkit) | ORM annotations (`Count`, `Q`, `Case`/`When`), template tags, `django.core.cache` | Already the whole stack |
| django-tables2 | 3.0.0 `[VERIFIED: pip show django-tables2]` | The D-08 "Progress" column on `CampaignRunTable` | Already used for every other campaign-table column |
| requests | 2.33.1 `[VERIFIED: pip show requests]` | The D-07 portal fetch (`make_request()` wrapper already used by `calendar_utils.resolve_placement_block()` and every facility class) | Already the HTTP client this codebase uses for all portal calls |
| tomtoolkit / tom_observations | project-pinned | `LCOFacility`/`SOARFacility`, `get_terminal_observing_states()`/`get_failed_observing_states()`, `ObservationGroup.observation_records` M2M | Source of the OCS vocabulary D-05 canonicalizes on |

### Supporting
None beyond the above — no new supporting library.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A new `ProposalTimeAllocation` model (D-07) | Storing the fetched figure as JSON on `WatchedProposal` | Rejected by D-07 explicitly — `WatchedProposal` "is not overloaded — it stays a watch list" |
| Reading `timeallocation_set` at request time | Caching it in Django's low-level cache instead of a model | Rejected by D-07 explicitly — an anonymous visitor must never trigger a credentialed portal call; a DB row also survives a cache eviction/outage without a blank page |

**Installation:** None — no `pip install` / migration-only additions beyond the new model(s)
this phase's own plan defines (e.g. `ProposalTimeAllocation`, and possibly a `proposal_code`
field on `CampaignRun` — see Open Questions).

**Version verification performed:** `pip show django-tables2` → 3.0.0; `pip show requests` →
2.33.1. Both already imported and exercised by existing, tested code paths in this repo
(`campaign_tables.py`, `calendar_utils.py`).

## Package Legitimacy Audit

**Not applicable — no new external packages are introduced by this phase.** All libraries used
(Django, django-tables2, requests, tomtoolkit) are pre-existing project dependencies verified
present and in active use via `pip show` and grep against the current codebase this session.

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────┐
                         │   solsys_code/status_vocabulary.py   │
                         │   (new peer module, Claude's         │
                         │    discretion on name/location)      │
                         │                                       │
                         │  - OCS canonical states (D-05)        │
                         │  - facility mapping table (Gemini,    │
                         │    ESO placeholders)                  │
                         │  - MARKER_BY_STATE = {[Q],[S],[O],    │
                         │    [X],[C],[F],[W],[?],[U]}           │
                         │  - LEGEND (ordered tuple, D-04)       │
                         │  - classify(record, facility) -> state│
                         └───────┬───────────────┬───────────────┘
                                 │ imported by   │ imported by
              ┌──────────────────┘               └───────────────────┐
              ▼                                                       ▼
  observation_projector.py                                campaign_reconciler.py /
  (stage_for/title_for -- REPLACE                          allocation_projector.py
   _STAGE_MARKER/_FAILURE_MARKER_BY_STATUS                 (RUN_STATUS_CALENDAR_PREFIX
   with the shared module's marker table)                   -> shared module's [C]/[W])
              │                                                       │
              └───────────────────┬───────────────────────────────────┘
                                   ▼
                    calendar_display_extras.py
                    (_TERMINAL_PREFIXES/_OBSERVATION_STATUS_LEGEND
                     DELETED -- status_border_css()/observation_status_legend()
                     now call into the shared module; NEW: run_tally(),
                     campaign_tally(), unused_night_decoration() tags)
                                   │
                    ┌──────────────┼──────────────────────┐
                    ▼              ▼                      ▼
        event_form.html   campaignrun_table.html   campaign_list.html
        (D-09 tally in     (D-08 Progress column,   (D-10 badge:
         attributed-run     D-10 header strip)        "N runs · M nights
         block)                                        observed")
                                   │
                                   ▼
                    calendar_utils.resolve_placement_block()
                    (line 336's 'COMPLETED' literal -> shared
                     module's classifier)

  ┌───────────────────────────────────────────────────────────────────┐
  │  unattended.py STEPS (new 5th step or folded into status_refresh)  │
  │                                                                     │
  │  LCOFacility()._portal_headers()  ──►  GET /api/proposals/<id>/    │
  │       │                                    (timeallocation_set)    │
  │       ▼                                                             │
  │  ProposalTimeAllocation.objects.update_or_create(                  │
  │      proposal_code=..., semester=..., instrument_type=...)         │
  └──────────────────────┬──────────────────────────────────────────────┘
                          │ read-only, no credentialed call at request time
                          ▼
              campaign_tables.py / calendar_display_extras.py
              (D-06: (allocated_hours - used_hours) / 10 = unused estimate,
               attached to every run carrying that proposal code)

  ┌─────────────────────────────────────────────────────────────────┐
  │  campaign_gap.py claimed_dates() -- TODAY: CampaignRun windows    │
  │  only. GAPB-01 adds a second claim source:                       │
  │                                                                    │
  │   CalendarEventMeta.objects.filter(run__campaign=campaign)        │
  │       UNION (by target)                                           │
  │   ObservationRecord.objects.filter(target__in=campaign.targets)   │
  │       │                                                            │
  │       ▼  (only [O]/[S]-classified records claim a night, D-16)   │
  │   observing_night(record_time_window(record)[0], site_zone)       │
  │       │                                                            │
  │       ▼  (site resolution: observed_site param -> Observatory;    │
  │           else attributed run's site; else "site unknown", D-17) │
  │   unioned into claimed set alongside existing run-window claims   │
  │   (D-19); _EXCLUDED_RUN_STATUSES unchanged                        │
  └─────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

No new top-level directories — this phase adds peer modules/functions to the existing flat
`solsys_code/` app layout:

```
solsys_code/
├── status_vocabulary.py          # NEW — canonical classifier + marker/legend tables (STATUS-01/02)
├── observation_projector.py      # MODIFIED — imports marker table from status_vocabulary
├── campaign_reconciler.py        # MODIFIED — RUN_STATUS_CALENDAR_PREFIX -> shared [C]/[W]
├── allocation_projector.py       # MODIFIED — allocation_night_title() uses shared prefix
├── calendar_utils.py             # MODIFIED — resolve_placement_block()'s 'COMPLETED' -> classifier
├── campaign_gap.py               # MODIFIED — claimed_dates() adds observation-event claims (GAPB-01)
├── campaign_tables.py            # MODIFIED — new Progress column (D-08)
├── campaign_views.py             # MODIFIED — roll-up context data (D-10), still PII-gated
├── unattended.py                 # MODIFIED — new step or extended status_refresh (D-07)
├── models.py                     # MODIFIED — new ProposalTimeAllocation model (+ migration)
├── templatetags/
│   └── calendar_display_extras.py  # MODIFIED — status_border_css()/observation_status_legend()
│                                    #   read from status_vocabulary; new run_tally(),
│                                    #   campaign_tally(), unused_night_decoration() tags
└── tests/
    ├── test_status_vocabulary.py           # NEW
    ├── test_calendar_display_extras.py     # MODIFIED — legend/ring tests updated to new markers
    ├── test_campaign_approval.py           # MODIFIED — prefix-string assertions updated
    ├── test_campaign_gap.py                # MODIFIED — GAPB-01 new claim source tests
    ├── test_campaign_tables.py / test_campaign_views.py  # MODIFIED — tally column/context tests
    └── test_unattended.py                  # MODIFIED — new step test
```

### Pattern 1: Display-time decoration from a link, never a text write

**What:** Every visible campaign/status fact is computed at render time from a foreign-key
link or a record's own fields — never written into `CalendarEvent.title`/`.description` by a
"decoration" step. This is the established Phase 33 pattern (`campaign_decoration()`) and it
is the pattern the tally, unused-night, and status-classifier work must all follow.

**When to use:** Any time a public template needs to show a fact derived from a linked model
(a run's status, a series' position, a night's unused-ness, a run's tally).

**Example (the exact function this phase's new tags should mirror):**
```python
# Source: solsys_code/templatetags/calendar_display_extras.py:493-562 (read this session)
@register.simple_tag
def campaign_decoration(event: CalendarEvent) -> dict | None:
    if not isinstance(event, CalendarEvent):
        return None
    try:
        meta = event.telescope_label_meta
    except ObjectDoesNotExist:
        return None
    run = meta.run
    if run is None or not run.is_publicly_visible:
        return None
    table_url = None
    if run.campaign_id is not None:
        table_url = f"{reverse('campaigns:table', args=[run.campaign_id])}#run-{run.pk}"
    return {
        'campaign_name': run.campaign.name if run.campaign_id is not None else NO_CAMPAIGN_LABEL,
        'run_pk': run.pk,
        'table_url': table_url,
        'telescope_instrument': run.telescope_instrument,
        'window_start': run.window_start,
        'window_end': run.window_end,
        'run_status_display': run.get_run_status_display(),
    }
```
A `run_tally(run)` tag for D-09 should have the identical shape: guard `isinstance`/
`ObjectDoesNotExist`/`is_publicly_visible`, return a plain dict of pre-computed numbers, never
raise.

### Pattern 2: The three status-prefix vocabularies today (STATUS-01 target)

**Vocabulary 1 — the observation projector's own terse markers** (already the D-01 target
form):
```python
# Source: solsys_code/observation_projector.py:86-99 (read this session)
_FAILURE_MARKER_BY_STATUS = {
    'WINDOW_EXPIRED': '[X]',
    'CANCELED': '[C]',
    'FAILURE_LIMIT_REACHED': '[F]',
    'NOT_ATTEMPTED': '[F]',
}

_STAGE_MARKER = {
    'queued': '[Q]',
    'placed': '[S]',
    'observed': '[O]',
    'completed-no-block': '[O]',
}
```

**Vocabulary 2 — the run-level bracket-word prefixes** (D-01/D-02 must migrate these to
`[C]`/`[W]`):
```python
# Source: solsys_code/campaign_reconciler.py:76-79 (read this session)
RUN_STATUS_CALENDAR_PREFIX = {
    CampaignRun.RunStatus.CANCELLED: '[CANCELLED]',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: '[WEATHERED]',
}
```
`RunStatus` is a `models.TextChoices` with exactly these 8 members
`[VERIFIED: solsys_code/models.py:211-218]`:
```
REQUESTED = 'requested', 'Requested'
PLANNED = 'planned', 'Planned'
OBSERVED = 'observed', 'Observed'
REDUCED = 'reduced', 'Reduced'
PUBLISHED = 'published', 'Published'
CANCELLED = 'cancelled', 'Cancelled'
NOT_AWARDED = 'not_awarded', 'Not Awarded'
WEATHER_TECH_FAILURE = 'weather_tech_failure', 'Weather/Technical Failure'
```
Only `CANCELLED` and `WEATHER_TECH_FAILURE` currently map to a calendar prefix; the other six
values are staff-facing states with no calendar-visible marker today (consistent with D-02's
"a staff-cancelled run night is `[C]`... a weathered/technical-failure run night is `[W]`" —
no other `RunStatus` value needs a marker).

**Vocabulary 3 — the display layer's own copies** (both must be deleted once the shared
module exists):
```python
# Source: solsys_code/templatetags/calendar_display_extras.py:132 (read this session)
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X] ', '[C] ', '[F] ', '[?] ')
```
```python
# Source: solsys_code/templatetags/calendar_display_extras.py:140-149 (read this session)
_OBSERVATION_STATUS_LEGEND = (
    {'marker': '[Q]', 'label': 'Queued'},
    {'marker': '[S]', 'label': 'Scheduled'},
    {'marker': '[O]', 'label': 'Observed'},
    {'marker': '[X]', 'label': 'Window expired'},
    {'marker': '[C]', 'label': 'Cancelled'},
    {'marker': '[F]', 'label': 'Failed'},
    {'marker': '[?]', 'label': 'Inconsistent record'},
)
```
Note the legend already has no `[W]`/`[U]` entries — D-04 explicitly adds both. The
`_TERMINAL_PREFIXES` tuple's bracket-WORD entries (`'[EXPIRED]'`, `'[CANCELLED]'`, `'[FAILED]'`,
`'[WEATHERED]'`) are the legacy vocabulary D-01's "no legacy-spelled title" migration note
targets — a real developer-database check that no title still starts with one of these four
strings is the acceptance bar for the migration task.

### Pattern 3: The one real `status == 'COMPLETED'` check (STATUS-02 target)

```python
# Source: solsys_code/calendar_utils.py:330-336 (read this session)
    current_block = None
    for block in blocks:
        if block.get('state') == 'COMPLETED':
            current_block = block
            break
        elif block.get('state') == 'PENDING':
            current_block = block
    return current_block
```
This is inside `resolve_placement_block()`, called by the observed-site sweep-only lookup
(`project_observation_calendar.resolve_observed_site()`) — it selects which portal API *block*
(not which *record*) counts as the observed one, by comparing the block's own `'state'` string
against the literal `'COMPLETED'`. Per D-05, this string literal should be replaced with a call
into the shared classifier's canonical-state constant rather than deleted or generalized away —
the block-selection logic itself (COMPLETED-first-else-PENDING) is correct and stays; only the
literal string comparison should route through one definition.

### Pattern 4: PII gate at the queryset (must be preserved by every new tally query)

```python
# Source: solsys_code/campaign_views.py:92-113, 165-177 (read this session)
ALLOWED_FIELDS_FOR_NON_STAFF = [
    'pk', 'telescope_instrument', 'site__short_name', 'site_raw', 'site_needs_review',
    'telescope_class', 'window_start', 'window_end', 'filters_bandpass',
    'run_status', 'approval_status', 'open_to_collaboration',
    'observation_details', 'weather', 'observation_outcome', 'publication_plans', 'comments',
]
# ... in CampaignRunTableView.get_queryset():
qs = qs.values(*[f for f in ALLOWED_FIELDS_FOR_NON_STAFF if f not in ('contact_person', 'contact_email')])
return qs.annotate(
    contact_person=Case(When(contact_public_opt_in=True, then=F('contact_person')), default=Value(''), output_field=CharField()),
    contact_email=Case(When(contact_public_opt_in=True, then=F('contact_email')), default=Value(''), output_field=EmailField()),
)
```
Any new tally column/annotation added to this queryset for D-08 must either (a) be added as a
`.annotate()` after `.values()` narrows the field list (the same ordering constraint the
existing comment documents — Django's `annotate()` alias-collision check runs against the
model's full field list unless `.values()` has already narrowed it), or (b) be computed in a
separate, PII-free query keyed only on `pk` and joined in Python. Never widen
`ALLOWED_FIELDS_FOR_NON_STAFF` to include a field this phase doesn't explicitly need public.

### Pattern 5: The low-level cache pattern for per-row-in-Python aggregation

```python
# Source: solsys_code/campaign_gap.py:28 (read this session)
GAP_CACHE_TTL_SECONDS = 3600  # D-10: 1-hour result cache
```
`get_or_compute_gap()` wraps `_compute_gap()` in `django.core.cache.cache.get()`/`.set()`
keyed by `build_gap_cache_key(campaign_pk, target_pk, site_pk, start, end)`. D-08's tally
counts that need the site-local-night rule (not expressible as a single SQL aggregate) should
follow this exact shape: a pure `_compute_tally(run)` function, a `build_tally_cache_key(run.pk)`
key builder, and a `get_or_compute_tally(run)` wrapper — this is the "folded TTL-cache pattern"
D-08 names explicitly, and it is also the fix for the already-known
`2026-09-01-add-ttl-cache-to-attribution-banner-count.md` todo's underlying pattern.

### Pattern 6: Coverage-gap's current claim query (the exact GAPB-01 gap)

```python
# Source: solsys_code/campaign_gap.py:164-211 (read this session, abridged)
qs = CampaignRun.objects.filter(campaign=campaign, site=site, approval_status=CampaignRun.ApprovalStatus.APPROVED)
qs = qs.exclude(run_status__in=_EXCLUDED_RUN_STATUSES)
qs = qs.only('pk', 'window_start', 'window_end')
...
for run in qs:
    ...
    n_days = (run.window_end - run.window_start).days + 1
    for i in range(n_days):
        claimed.add(run.window_start + timedelta(days=i))
return claimed, undated_runs, unattributed_runs, pending_narrowing_runs
```
This function **never references `ObservationRecord`, `CalendarEventMeta`, or
`CampaignRunObservation`** — confirmed by reading the entire function body this session. Every
date claimed today comes exclusively from an approved `CampaignRun`'s `window_start`/
`window_end` pair. This is precisely what the roadmap's Success Criterion 5 ("classical and
queue time is no longer reported as unclaimed") is describing: a classical or queue-scheduled
observation whose `CampaignRun` window doesn't cover a night it was actually taken on (or whose
observation isn't yet attributed to any run) reads as an open gap today even though the night
was used.

### Anti-Patterns to Avoid
- **A fourth ring colour for `[U]`:** D-13 explicitly rejects this — unused must be
  distinguished by chip style (opacity/dashed border) *and* the `[U]` text token, never a new
  `status_border_css()` colour, so the signal survives for colour-blind and screen-reader users.
- **Deriving the legend from the classifier's own state list:** D-04 explicitly keeps the
  legend "a fixed, ordered vocabulary read from the one module — never data-driven from the
  database" — the existing `_OBSERVATION_STATUS_LEGEND`'s own comment (line 137-139, read this
  session) already documents why: "deriving it would only let ring-vs-label drift in the other
  direction."
- **A per-row tally query loop over `CampaignRunTable`'s rows:** D-08 explicitly forbids this
  ("never a per-row loop") — the SQL-expressible counts (records, groups) must be one annotated
  query; the night-classification counts must go through the shared TTL-cache pattern (Pattern 5).
- **Trusting a facility's own `get_terminal_observing_states()` as automatically meaning
  "observed":** D-05 explicitly names `tom_gemini`'s `TRIGGERED`/`ON_HOLD` as a case where a
  facility's own "terminal" vocabulary means *submitted*, not *observed* — the classifier must
  map every non-LCO/SOAR facility through an explicit FOMO-side table, never trust the
  facility's own boolean.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Site-local "which night does this belong to" | A new date-bucketing function | `telescope_runs.observing_night(start_time, site_zone)` `[VERIFIED: solsys_code/telescope_runs.py:309-333]` | Already correctly handles the noon-anchor rule (a 02:00-local start belongs to the previous date's night) — Phase 33's CR-02 fix; re-deriving this is a proven regression source |
| Per-facility terminal/failure state sets | A new hardcoded status list per facility | `facility.get_terminal_observing_states()` / `facility.get_failed_observing_states()` (TOM Toolkit) `[VERIFIED: solsys_code/observation_projector.py:103,128]` | Already the source of truth `stage_for()` reads from; re-deriving risks drifting from what `updatestatus`/the sweep actually see |
| PII-safe public queryset | A template-level `{% if request.user.is_staff %}` conditional around contact fields | The existing `.values()` + `Case`/`When` queryset-level gate (Pattern 4) | A template conditional still fetches the real value into the response context; the existing pattern never fetches it from the DB for a non-opted-in row |
| Portal-call caching/backoff for the timeallocation fetch | A bespoke retry/backoff loop inside a template tag or view | The existing `unattended.py` `StepResult`/`command_lock()`/per-step failure-isolation pattern (`[VERIFIED: solsys_code/unattended.py:80-115, 254-311]`) | Phase 36 already solved "credentialed call fails, one email per newly-failing tick, never blocks the other steps" — this phase's new step is a fifth instance of an already-proven pattern |
| Coverage-gap result caching | A new cache decorator | `campaign_gap.py`'s existing `GAP_CACHE_TTL_SECONDS`/`get_or_compute_gap()` (Pattern 5) | GAPB-01 extends the same function; a second caching mechanism would let the two disagree |

**Key insight:** Every "don't hand-roll" item above is not a third-party library recommendation
— it's "the exact function already in this codebase that solves this, verified present and
correct this session." The dominant risk in this phase is re-deriving a rule (the night-anchor,
the terminal-state set, the PII gate, the caching TTL) slightly differently from where it
already lives, producing the drift the phase's own D-01 rationale (`"the three parallel prefix
maps... agree only by convention"`) explicitly names as the disease this phase is meant to
cure.

## Runtime State Inventory

Not applicable — this phase is additive (a new peer module, new template tags, a new model) and
consolidative (replacing three in-code Python dict/tuple literals with one), not a rename,
rebrand, or string-replacement migration across external systems. The one runtime-state item
that *does* apply is covered under Common Pitfalls (legacy title strings already stored in the
database) since it is a data migration, not an external-system concern:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Legacy `[CANCELLED]`/`[WEATHERED]`/`[EXPIRED]`/`[FAILED]` (bracket-word) title prefixes already written to `CalendarEvent.title` rows in the developer database by the pre-Phase-37 reconciler/allocation-projector code paths | Data migration via the recommended one-time re-title sweep (Claude's Discretion, "Legacy title migration") — a re-project/re-reconcile run, not a schema migration |
| Live service config | None — no external service (n8n, Tailscale, etc.) config carries these strings | None |
| OS-registered state | None | None |
| Secrets/env vars | None — the D-07 portal fetch reuses the existing `LCOFacility`/`SOARFacility` `api_key` setting (`LCO_API_KEY` env var, already read by every other portal call in this codebase) — no new secret name | None |
| Build artifacts | None | None |

## Common Pitfalls

### Pitfall 1: Deleting `_TERMINAL_PREFIXES`'s bracket-word entries before the developer database is re-titled

**What goes wrong:** If `status_border_css()`/`observation_status_legend()` are switched to
read only the new short-letter vocabulary before every existing `[CANCELLED]`/`[WEATHERED]`
title in the real developer database has been re-titled, those events silently lose their
status ring — not an error, just a quietly wrong display.
**Why it happens:** `_TERMINAL_PREFIXES` today deliberately carries both vocabularies side by
side (its own comment says "both must stay byte-identical to their producers") specifically
because a migration is not instantaneous.
**How to avoid:** Sequence the plan so the re-title sweep (Claude's Discretion) runs and is
verified (a grep/query proving zero remaining bracket-word titles in the real developer
database, mirroring Phase 34's "one-time title change" runbook note) *before* the legacy
entries are deleted from the ring-matcher.
**Warning signs:** A calendar screenshot/UAT step showing a cancelled/weathered run's chip with
no ring after the vocabulary module ships.

### Pitfall 2: The classical loader's `[proposal]` token is not a structured field anywhere

**What goes wrong:** Assuming `CampaignRun.observation_details` (a free-text field containing
`f'\nProposal: {parsed.proposal}'`, confirmed at
`solsys_code/management/commands/load_telescope_runs.py:276-277` and
`solsys_code/management/commands/cutover_classical_allocations.py:486-487`, both read this
session) can be regex-parsed reliably at tally-fetch time will be fragile — the text is
free-form staff-editable content (`observation_details` is a plain `TextField`, editable in the
admin and the submission form), not a structured value.
**Why it happens:** The proposal token was designed only to disambiguate two classical runs
sharing a telescope/instrument/night (its `source_identifier` role), not to be read back later.
**How to avoid:** Add a real structured field (see Open Questions) rather than parsing
`observation_details` text at fetch time.
**Warning signs:** A tally showing "unused: not yet fetched" forever for every classical run,
or a fetch step matching the wrong proposal because a staff member edited the free-text field.

### Pitfall 3: `campaign_gap.claimed_dates()`'s PII-minimizing `.only()` must be widened carefully

**What goes wrong:** GAPB-01's new observation-event claim source needs to read
`CalendarEventMeta.run`, `ObservationRecord.parameters` (for `observed_site`) and
`ObservationRecord.target` — none of which are covered by the existing
`.only('pk', 'window_start', 'window_end')` restriction (confirmed at
`solsys_code/campaign_gap.py:172`, read this session), because that restriction applies to the
`CampaignRun` queryset, not a new `ObservationRecord`/`CalendarEventMeta` queryset. A careless
implementation could accidentally join back to `CampaignRun.contact_person`/`.contact_email`
through the `CalendarEventMeta.run` FK if a wide `select_related()` is used.
**Why it happens:** The gap page is public (no `StaffRequiredMixin`, confirmed at
`solsys_code/campaign_views.py:869`); any new query added to its data path inherits the same
PII obligation the existing `CampaignRun` query already honors.
**How to avoid:** Give the new `ObservationRecord`/`CalendarEventMeta` query its own explicit
`.only()`/`.values()` restriction naming exactly the fields needed (`pk`, `parameters` [for the
site keys], `scheduled_start`/`scheduled_end`/`status` [for `record_time_window`/`stage_for`],
`run_id`), never a bare `select_related('run')` that could later be widened to include contact
fields by a future editor.
**Warning signs:** A code-review or security-review flag on a new gap-analysis query that
touches `CampaignRun` fields beyond `pk`/`site_id`.

### Pitfall 4: `[C]` collision between run-level and record-level cancellation (D-02, intentional but easy to mis-implement)

**What goes wrong:** A naive classifier implementation might try to give run-level cancellation
and record-level cancellation *different* markers "because they're different things," breaking
D-02's explicit design ("`[C]` therefore means 'cancelled, by whoever owns this event'").
**Why it happens:** The two cancellations come from genuinely different code paths (a staff
`mark_cancelled` action on `CampaignRun.run_status` vs. a portal `CANCELED` observing state on
`ObservationRecord.status`) and it's natural to want to keep them visually distinct.
**How to avoid:** Follow D-02 literally — same `[C]` marker for both; the pop-up's existing
`Run status:` line (already rendered by `campaign_decoration()`'s `run_status_display` key,
confirmed at `calendar_display_extras.py:561`) is the only place the distinction is surfaced.
**Warning signs:** A legend or test expecting two different cancellation markers.

## Code Examples

### The `CampaignRunObservation`/`ObservationGroup` relations the tally counts read

```python
# Source: solsys_code/models.py:575-630 (read this session) -- CampaignRunObservation
run = models.ForeignKey(CampaignRun, on_delete=models.CASCADE, related_name='observation_links', ...)
observation_record = models.ForeignKey(ObservationRecord, on_delete=models.CASCADE, related_name='campaign_run_links', ...)
# One row per confirmed attribution (D-01/D-03 in that model's own docstring): a run's linked
# record COUNT is `run.observation_links.count()`.
```
```python
# Source: tom_observations/models.py:111 (installed package, read this session) -- ObservationGroup
observation_records = models.ManyToManyField(ObservationRecord)
# No related_name declared -> default reverse accessor on ObservationRecord is `observationgroup_set`.
# A run's linked GROUP count (distinct) is:
#   ObservationGroup.objects.filter(
#       observation_records__campaign_run_links__run=run
#   ).distinct().count()
```

### Site timezone lookup for per-record night classification

```python
# Source: solsys_code/solsys_code_observatory/models.py:38-66 (read this session)
obscode = models.CharField(...)
short_name = models.CharField(max_length=255, ...)
timezone = models.CharField(max_length=64, blank=True, default='', verbose_name='IANA timezone name')
# Convert with: from zoneinfo import ZoneInfo; ZoneInfo(observatory.timezone)
```

### The LCO Observation Portal `timeallocation_set` shape

```
# Source: developers.lco.global (official docs, fetched this session) +
# github.com/LCOGT/observation-portal-api-examples query_proposals.py (fetched this session)
# [CITED -- not tool-verified against live FOMO credentials this session]

GET https://observe.lco.global/api/proposals/<proposal_code>/
Headers: Authorization: Token <api_key>

Response (abridged):
{
  "id": "<proposal_code>",
  "timeallocation_set": [
    {
      "semester": "2026B",
      "instrument_types": ["1M0-SCICAM-SINISTRO"],
      "std_allocation": 40.0,
      "std_time_used": 12.5,
      "rr_allocation": 0.0,
      "rr_time_used": 0.0,
      "tc_allocation": 0.0,
      "tc_time_used": 0.0,
      "ipp_limit": ...,
      "ipp_time_available": ...
    },
    ...
  ]
}
```
D-06's "allocated hours minus used hours, summed over the proposal's allocations" maps to
`sum(entry['std_allocation'] - entry['std_time_used'] for entry in timeallocation_set)` for the
standard-time allocation type. Whether Rapid-Response (`rr_*`) or Time-Critical (`tc_*`) hours
should also be summed in depends on which time type FOMO's watched proposals actually use — a
question for the plan's Wave 0 (a live `GET` against one real watched proposal's code, using the
existing `LCOFacility` credentials, should be run and inspected before committing to a formula
that only sums `std_*`).

### The existing per-step unattended runner pattern the D-07 fetch step should copy

```python
# Source: solsys_code/unattended.py:80-96, 421-426 (read this session)
@dataclass
class StepResult:
    name: str
    failed: bool
    summary: str  # credential- and PII-free counter/status line

STEPS = (
    ('status_refresh', step_status_refresh),
    ('project_sweep', step_project_sweep),
    ('discovery', step_discovery),
    ('reconcile', step_reconcile),
    # D-07 candidate: ('proposal_allocation', step_proposal_allocation),
)
```
Each existing `step_*` function follows: dry-run short-circuit -> `command_lock(name)` context
manager (per-step lock, defence in depth behind the runner-level lock) -> real work wrapped in
its own try/except -> a `StepResult` with a `failed` bool and a short summary string. The new
step should be added to this tuple (D-07 explicitly allows either a new tuple entry or folding
into `step_status_refresh` — planner's choice) and must never log the `api_key` value (mirrors
`step_status_refresh`'s own IN-05/WR-08 comment discipline about never leaking credential
values into log lines, per SCHED-10).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Three independently-maintained status-prefix vocabularies with "must stay byte-identical" comments | One canonical classifier module | This phase (STATUS-01/02) | Removes an entire class of silent-drift bug the codebase's own comments already flag as a known risk |
| `sync_lco_observation_calendar`-era bracket-WORD prefixes (`[CANCELLED]`, `[WEATHERED]`, `[EXPIRED]`, `[FAILED]`) | Bracket-LETTER markers (`[C]`, `[W]`, `[X]`, `[F]`) | Phase 34 (partial) / this phase (complete, D-01) | Legend/ring logic collapses from 8 recognized prefix strings to one small marker set |
| `campaign_gap.claimed_dates()` reading only `CampaignRun` windows | Also reading observation events (D-16..D-19) | This phase (GAPB-01) | Classical/queue-scheduled observations no longer read as "gaps" just because no `CampaignRun` window happens to cover the exact night they were taken |

**Deprecated/outdated:**
- The bracket-WORD title vocabulary (`[CANCELLED]`/`[WEATHERED]`/`[EXPIRED]`/`[FAILED]`) — superseded by the D-01 short-letter form; only remains live in already-stored titles until the migration sweep runs.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The LCO Observation Portal's `timeallocation_set` endpoint and field names (`std_allocation`, `std_time_used`, `semester`, `instrument_types`) are current and stable, and `GET /api/proposals/<code>/` (singular) works the same way as the documented `/api/proposals/` list endpoint | Code Examples, "LCO Observation Portal `timeallocation_set` shape" | If the field names or endpoint shape have changed since the docs snapshot fetched this session, D-07's fetch step will need a Wave-0 correction against a real watched proposal before the model schema is finalized |
| A2 | Only `std_allocation`/`std_time_used` (not `rr_*`/`tc_*`) should feed the D-06 unused-hours formula | Code Examples | If FOMO's watched proposals actually use Rapid-Response or Time-Critical time, the unused estimate would undercount; needs a live check against a real proposal |
| A3 | The 8-value `CampaignRun.RunStatus` vocabulary (`REQUESTED`/`PLANNED`/`OBSERVED`/`REDUCED`/`PUBLISHED`/`CANCELLED`/`NOT_AWARDED`/`WEATHER_TECH_FAILURE`) is exhaustive and will not gain a 9th value mid-phase that also needs a calendar marker | Architecture Patterns, Pattern 2 | Low risk — this is a stable, already-migrated `TextChoices` enum read directly from `models.py` this session, not inferred |

**Note:** A1/A2 are the only genuinely `[ASSUMED]`-tier items in this research; everything else
in this document is either `[VERIFIED: <path>:<lines>]` (read from the actual source this
session, with the constant/value quoted verbatim above) or `[CITED: <url>]` (official LCO
documentation / the LCOGT example-scripts repository).

## Open Questions (RESOLVED)

All three questions below were dispositioned during planning; each carries its resolution and the
plan and task that owns it. Nothing in this section is outstanding — an executor should read the
**Resolution** line first and treat the "What was unclear" paragraph as the record of how the
answer was reached, not as a live choice to make.

1. **How is a proposal code actually attached to a `CampaignRun` for the D-07 fetch to key on?**
   - **Resolution (RESOLVED — 37-02 Task 1 and Task 2):** option (a). Plan 37-02 Task 1 is a
     `checkpoint:decision` with `gate="blocking"` that confirms the portal's field names against
     one live credentialed call before any code is written; Task 2 then adds the
     `proposal_code` `CharField(blank=True, default='')` to `CampaignRun` with its migration and
     populates it from `ParsedRun.proposal` in the classical loader, leaving WEB/CSV runs blank
     and covered by the not-yet-known fallback. No planning decision is left open here.
   - What we know: `WatchedProposal.proposal_code` exists as a config list
     (`[VERIFIED: solsys_code/models.py:758]`), and the classical loader's `[proposal]` token
     is parsed into `ParsedRun.proposal`
     (`[VERIFIED: solsys_code/telescope_runs.py:373,401-437]`) — but it is only ever
     concatenated into the free-text `observation_details` field
     (`[VERIFIED: solsys_code/management/commands/load_telescope_runs.py:276-277]`), never
     stored as a structured value on `CampaignRun` or on the resulting `CalendarEvent.proposal`
     field. `CampaignRun` has no `proposal`/`proposal_code` field at all
     (`[VERIFIED: solsys_code/models.py:195-421`, full field list read this session`]`), and a
     repo-wide grep for "proposal" in `campaign_forms.py` and `import_campaign_csv.py` returned
     zero hits — the WEB-submission and CSV-import paths never capture a proposal code either.
   - What was unclear at research time (settled by the Resolution above): whether the planner
     should (a) add a new `proposal_code` field to
     `CampaignRun` populated at write time by each of the three active write paths (classical
     loader, CSV import column if one exists, a new web-form field), or (b) derive it
     differently (e.g. matching `WatchedProposal.proposal_code` against a run's
     `telescope_instrument`/site by some other means, which seems fragile), or (c) scope D-07's
     fetch to only the `WatchedProposal`-driven LCO/SOAR queue path (where a proposal code is
     already known unambiguously from the watch list) and treat classical/CSV/web-submitted
     class-wide runs as "not yet fetchable" until a structured field exists.
   - Recommendation: add a `proposal_code` `CharField(blank=True, default='')` to `CampaignRun`
     as part of this phase's migration, populated by the classical loader (parsing
     `ParsedRun.proposal` into the new field instead of/in addition to the free-text line) and
     left blank for WEB/CSV runs until a follow-up captures it there too — this keeps D-06/D-07
     buildable now without inventing a fragile text-parsing dependency, and the tally's "not yet
     fetched" fallback (Claude's Discretion) already covers the runs left blank.

2. **The runbook has no existing "coverage-gap analysis" section to change.**
   - **Resolution (RESOLVED — 37-07 Task 1, with the behaviour it documents in 37-03):** the
     recommendation was taken. 37-07 Task 1's action says the coverage-gap section is *written
     fresh* ("a full-text search of this file found no existing section") and lists what it must
     cover; 37-07's `must_haves` carries the same as an observable truth. `docs/design/` is left
     alone, being rationale rather than operator how-to. No planning decision is left open here.
   - What we know: the phase's canonical-refs and roadmap text both say "the gap-analysis
     section" in `docs/runbooks/telescope_runs_calendar.rst` changes as a paired doc for this
     phase. A full-text search of that file this session for "Coverage", "gap", "unclaimed", and
     "claimed_dates" returned zero matches (a mention of "gap" appears only in unrelated prose
     about heartbeat timing and a historical backfill command, confirmed by reading the grep
     output line-by-line this session) — no section documenting the coverage-gap feature exists
     in this runbook today, despite the feature (`campaign_gap.py`, `CampaignGapAnalysisView`)
     having shipped in v2.0/v2.1 (GAP-01/GAP-02, ASSET-01/02).
   - What was unclear at research time (settled by the Resolution above): whether this is a
     pre-existing documentation gap the phase should also close (add a new section) or whether
     the gap-analysis feature is documented elsewhere (a `docs/design/*.rst` file) that the
     phase should instead update.
   - Recommendation: treat this as "add a new coverage-gap-analysis section to the runbook" — a
     repo-wide search this session (`grep -rln "coverage.gap\|Coverage-Gap\|claimed_dates" docs/`)
     found only `docs/design/canonical_record_spike.rst` mentions it, and design docs are
     rationale, not the operator-facing how-to the CLAUDE.md paired-docs rule targets. The GAPB-01
     plan should write this section fresh rather than assume it exists to be edited in place.

3. **Which `RunStatus` values other than `CANCELLED`/`WEATHER_TECH_FAILURE` should ever surface a marker, if any, as tallies roll up "observed/scheduled/expired-or-failed/unused"?**
   - **Resolution (RESOLVED — no action needed; enforced by 37-01 Task 1 and 37-04 Task 2):**
     the run-level marker table stays at exactly the two existing entries, migrated to `[C]`/`[W]`
     (37-01), and 37-04 Task 2's `is_unused_allocation_night()` derives its suppression set from
     that same table rather than re-listing statuses — so the other six `RunStatus` values cannot
     reach the classifier by analogy. This question was recorded as a guard rail, not a choice.
   - What we know: `RunStatus` has 8 values total, but only 2 map to a calendar prefix today.
     The tally counts (D-11) are explicitly defined as derived from *linked records* via the
     classifier, not from `run_status` itself (TALLY-03's guard) — so `run_status` values like
     `OBSERVED`/`REDUCED`/`PUBLISHED` are staff-only downstream bookkeeping states that should
     never feed the tally computation, only the pop-up's existing `Run status:` display line.
   - What was unclear at research time: nothing structurally — this is confirmed correct by TALLY-03 and D-14
     ("staff run status always wins" only for the `[C]`/`[W]` unused-suppression case). Listed
     here only so the planner doesn't accidentally wire `REQUESTED`/`PLANNED`/`OBSERVED`/
     `REDUCED`/`PUBLISHED` into the classifier's marker table by analogy with
     `CANCELLED`/`WEATHER_TECH_FAILURE`.
   - Recommendation: no action needed beyond keeping the classifier's run-level marker table to
     exactly the two existing entries, migrated to `[C]`/`[W]`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Django | Entire phase | ✓ | project-pinned | — |
| django-tables2 | D-08 Progress column | ✓ | 3.0.0 `[VERIFIED: pip show django-tables2]` | — |
| requests | D-07 portal fetch | ✓ | 2.33.1 `[VERIFIED: pip show requests]` | — |
| LCO Observation Portal (`observe.lco.global`) | D-07 `timeallocation_set` fetch | Not verified this session (would require a live credentialed call) | — | Runner-step failure isolation (Phase 36 pattern) already covers an outage; public pages show "not yet fetched" per Claude's Discretion note |
| SQLite (dev) | New `ProposalTimeAllocation` model/migration | ✓ (existing `src/fomo_db.sqlite3`) | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** the live portal call itself was not exercised this
session (would require live `LCO_API_KEY` credentials and a real watched-proposal code) — the
existing Phase 36 failure-isolation/heartbeat/email pattern is the documented fallback for a
portal outage at runner-tick time, and "not yet fetched" is the documented fallback for the
public page before the first successful fetch.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django's built-in `TestCase` runner (unittest-based) — **the only functioning suite per CLAUDE.md**; `python -m pytest` does not collect these tests |
| Config file | none — invoked directly via `manage.py test` |
| Quick run command | `python manage.py test solsys_code.tests.test_status_vocabulary` (new module) or `python manage.py test solsys_code.tests.test_calendar_display_extras.TestObservationStatusLegend` (existing, once renamed) |
| Full suite command | `python manage.py test solsys_code` (excluding `test_views.TestEphemeris`, which segfaults in native ASSIST per project memory) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STATUS-01 | One classifier drives title/ring/legend for every marker | unit | `python manage.py test solsys_code.tests.test_status_vocabulary` | ❌ Wave 0 (new file) |
| STATUS-01 | `test_campaign_approval.py`'s existing prefix-string assertions still pass under the new vocabulary | unit | `python manage.py test solsys_code.tests.test_campaign_approval` | ✅ (existing, needs literal-string updates) |
| STATUS-02 | `resolve_placement_block()` routes through the classifier, not a bare string literal | unit | `python manage.py test solsys_code.tests.test_calendar_utils` | ✅ (existing file, needs new test method) |
| TALLY-01/02 | Run/campaign tally counts match a hand-built fixture of linked records at each state | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras` (new test class) | ✅ (existing file, needs new class) |
| TALLY-03 | No module in this phase writes `run_status` | unit (a static/negative test — e.g. `unittest.mock.patch` asserting `CampaignRun.save` is never called with a changed `run_status`, or an AST/grep-based test over the phase's own new files) | `python manage.py test solsys_code.tests.test_status_vocabulary` | ❌ Wave 0 (new test) |
| UNUSED-01 | An `ALLOC:` event past its `end_time` with no `[C]`/`[W]` run status renders `[U]` | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras` | ✅ (existing file, needs new test) |
| GAPB-01 | A classical/queue observation with no covering `CampaignRun` window no longer appears as a gap | unit | `python manage.py test solsys_code.tests.test_campaign_gap` | ✅ (existing file, needs new test class) |

### Sampling Rate
- **Per task commit:** the single most relevant test module from the map above, via
  `python manage.py test solsys_code.tests.<module>`
- **Per wave merge:** `python manage.py test solsys_code` (full app suite, excluding the known
  ASSIST-segfault test per project memory)
- **Phase gate:** full suite green, `pre-commit run ruff --all-files` and
  `pre-commit run ruff-format --all-files` clean (D-07 in CLAUDE.md project constraints), before
  `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_status_vocabulary.py` — new file covering STATUS-01/02's
      classifier, marker table, legend, and the TALLY-03 negative-write guard test
- [ ] A live check (during Wave 0, not a unit test) of the real LCO Observation Portal
      `timeallocation_set` response shape for at least one real watched proposal, to confirm A1/A2
      before the `ProposalTimeAllocation` model schema is finalized
- [ ] Decision on Open Question 1 (`proposal_code` field addition) before the D-07 fetch step's
      "how do I find a proposal code for this run" logic is written

*(No framework install needed — Django's `TestCase` is already fully wired for this app.)*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No new authentication surface — tallies/gap page are intentionally public, mirroring the existing `CampaignRunTableView`/`CampaignGapAnalysisView` posture |
| V3 Session Management | no | No session-state change |
| V4 Access Control | yes | `run.is_publicly_visible` gate (existing, `[VERIFIED: solsys_code/models.py:423-435]`) must gate every new tally/decoration tag exactly as `campaign_decoration()` already does; a `PENDING_REVIEW` run's tally must never leak to a non-staff visitor |
| V5 Input Validation | yes | `CampaignGapAnalysisView`'s existing IDOR-safe target/site pk re-validation pattern (`_as_pk_or_none()`, server-side re-derivation of allowed sets) must be preserved/extended, not bypassed, by any new gap-analysis query parameter |
| V6 Cryptography | no | No new cryptographic operation — the D-07 portal fetch reuses `LCOFacility`'s existing `Authorization: Token` auth, never a new credential mechanism |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| PII leak via a new tally annotation accidentally including `contact_person`/`contact_email` in a non-staff queryset | Information Disclosure | Follow Pattern 4 exactly: annotate after `.values()` narrows the field list to `ALLOWED_FIELDS_FOR_NON_STAFF`; never widen that list for a tally column |
| Credential leak via a logged portal-fetch failure (the D-07 step) | Information Disclosure | Follow `step_status_refresh`'s existing discipline: the except clause never stringifies/logs the caught exception's body when it might embed request/response content with the API key (mirrors `resolve_placement_block()`'s own SYNC-09/D-11 comment, `[VERIFIED: solsys_code/calendar_utils.py:308-315]`) |
| IDOR on the coverage-gap page's new observation-event claim source (a crafted `target`/`site` query param reaching a campaign/target combination the requester shouldn't be able to probe) | Tampering / Information Disclosure | Reuse `CampaignGapAnalysisView`'s existing server-side re-validation of `target`/`site` against the campaign's own allowed sets (`[VERIFIED: solsys_code/campaign_views.py:894-927]`) for any new query parameter GAPB-01's "claimed, site unknown" listing introduces |
| A public tally silently exposing which run a currently-`PENDING_REVIEW` observation belongs to | Information Disclosure | Gate every new tag exactly like `campaign_decoration()`'s `run is None or not run.is_publicly_visible` check |

## Sources

### Primary (HIGH confidence — read from the actual source this session)
- `solsys_code/observation_projector.py` — `stage_for()`, `title_for()`, `_STAGE_MARKER`, `_FAILURE_MARKER_BY_STATUS`, `facility_for()`
- `solsys_code/campaign_reconciler.py` — `RUN_STATUS_CALENDAR_PREFIX`
- `solsys_code/allocation_projector.py` — `allocation_night_title()`, `allocation_events()`, `ALLOC_URL_NAMESPACE`
- `solsys_code/templatetags/calendar_display_extras.py` — `_TERMINAL_PREFIXES`, `_OBSERVATION_STATUS_LEGEND`, `status_border_css()`, `campaign_decoration()`, `observation_series_decoration()`
- `solsys_code/calendar_utils.py` — `resolve_placement_block()` (the line-336 `'COMPLETED'` check)
- `solsys_code/campaign_gap.py` — `claimed_dates()`, `_compute_gap()`, `get_or_compute_gap()`, `GAP_CACHE_TTL_SECONDS`, `_EXCLUDED_RUN_STATUSES`
- `solsys_code/campaign_views.py` — `ALLOWED_FIELDS_FOR_NON_STAFF`, `CampaignRunTableView.get_queryset()`, `CampaignListView`, `CampaignGapAnalysisView`
- `solsys_code/campaign_tables.py` — `CampaignRunTable`
- `solsys_code/unattended.py` — `StepResult`, `STEPS`, `step_status_refresh()`
- `solsys_code/models.py` — `CampaignRun` (`RunStatus`, `Source`, `TelescopeClass`, `is_publicly_visible`), `CalendarEventMeta`, `CampaignRunObservation`, `WatchedProposal`
- `solsys_code/telescope_runs.py` — `observing_night()`, `_resolve_proposal()`, `parse_run_line()`
- `solsys_code/solsys_code_observatory/models.py` — `Observatory` (`obscode`, `short_name`, `timezone`)
- `src/templates/tom_calendar/partials/event_form.html` — the `campaign_decoration`/`observation_series_decoration` render points (D-09's insertion point)
- `docs/runbooks/telescope_runs_calendar.rst` — status-prefix legend section, unattended-operation step list (confirmed no existing coverage-gap section)
- `.venv` site-packages `tom_observations/models.py` (`ObservationGroup.observation_records`) and `tom_calendar/models.py` (`CalendarEvent` field list) — installed package source, read this session
- `pip show django-tables2` / `pip show requests` — installed version verification

### Secondary (MEDIUM confidence)
- [LCO Developers](https://developers.lco.global/) — `timeallocation_set` field documentation (`std_allocation`, `std_time_used`, `semester`, `instrument_types`, and the RR/TC/realtime sibling fields), `/api/proposals/<id>/` single-proposal endpoint, `Authorization: Token` auth scheme
- [LCOGT/observation-portal-api-examples `query_proposals.py`](https://github.com/LCOGT/observation-portal-api-examples) — corroborates `timeallocation_set` field names and the `/api/proposals/` list-endpoint request shape

### Tertiary (LOW confidence)
- None — the WebFetch results above were cross-corroborated across two independent official/semi-official sources and are treated as MEDIUM, not LOW.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; installed versions verified via `pip show`
- Architecture (status vocabulary consolidation, tally computation, gap query extension): HIGH — every cited function/constant was read from the actual source this session with line numbers and verbatim quotes
- LCO portal `timeallocation_set` API shape: MEDIUM — official docs + example script agree, but not tool-verified against live FOMO credentials this session (flagged as Open Question / Assumption A1/A2)
- Proposal-code-on-CampaignRun question: HIGH confidence that the gap exists (exhaustive grep across all active write paths found zero occurrences); LOW/open on which resolution the planner should pick (Open Question 1)

**Research date:** 2026-09-18
**Valid until:** 30 days for the in-repo findings (stable, verified this session); the LCO portal API shape should be re-checked against live credentials before the D-07 plan/task is finalized, since it was not tool-verified this session
