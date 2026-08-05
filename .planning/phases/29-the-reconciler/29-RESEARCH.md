# Phase 29: The Reconciler - Research

**Researched:** 2026-08-04
**Domain:** Django management-command + view-layer refactor; calendar-event projection state machine over existing models (`CampaignRun`, `CalendarEventMeta`, `CampaignRunObservation`, `CalendarEvent`)
**Confidence:** HIGH for code-level mechanics (all claims below are read directly from the live repository, not training knowledge); MEDIUM/LOW flagged inline for the one real data-classification gap this research surfaces (see "Common Pitfalls" #1 and "Open Questions" #1)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: The reconciler takes over `_project_calendar_event()`'s job; `CAMPAIGN:` keys are
  retired.** `campaign_views.py`'s existing `_project_calendar_event()` already creates
  `CAMPAIGN:{run.pk}[:date]`-keyed events on approve/resolve_site. Per this discussion, the
  reconciler's locked `RUN:{run_pk}[:date]` key scheme supersedes it going forward:
  approve/resolve_site/mark_cancelled/mark_weather_failure stop calling
  `_project_calendar_event()` and call the reconciler's per-run function instead. Zero
  `CAMPAIGN:`-namespaced events exist in the dev DB today, so this is a clean cutover —
  `_project_calendar_event()` and `_calendar_event_title()` can be deleted outright rather
  than deprecated in place.

- **D-02: For a classically-scheduled run, the reconciler adopts a pre-existing attributed
  event rather than gap-filling around it.** For a **classical** run whose nights already
  carry a `load_telescope_runs`-created `CalendarEvent` (blank `url`), now linked via Phase
  28's `CalendarEventMeta.run`: the reconciler finds the existing attributed row (via the
  companion FK for this run + night) and updates/re-keys it to `RUN:{run_pk}:{date}` in
  place, rather than leaving it untouched and minting a second, separate event for the same
  physical night. For a **queue-scheduled run**, the reconciler NEVER touches per-night
  events — it mints only one bare `RUN:{run_pk}` whole-window container event, coexisting
  with the run's real `ObservationRecord`-derived events from the existing LCO/Gemini sync
  commands. The same adopt approach extends to D-01's retired `CAMPAIGN:` events (had any
  existed) — the reconciler is the sole authority going forward for any event tied to a
  `CampaignRun`, regardless of which prior mechanism created it.

- **D-03: A single shared core function, `reconcile_run(run)` (naming at planner's
  discretion), in `solsys_code/campaign_reconciler.py` implements all four pipeline stages
  for one `CampaignRun`.** The management command loops this function over every run
  (including under `--dry-run`); each of the four staff actions calls the identical function
  for its single run.

- **D-04: The staff-action call stays synchronous and inline** (no Celery, no async).
  `approve()` continues to swallow a reconcile failure (the run stays approved even if the
  calendar write failed); `resolve_site()` continues to treat a failure as "attempted but
  failed" and keeps `site_needs_review=True` so the action remains retryable.
  `mark_cancelled`/`mark_weather_failure` follow whichever of the two existing patterns they
  already structurally resemble in `_set_run_status()`.

- **D-05: The command's summary follows `import_campaign_csv`'s existing
  created/updated/unchanged/skipped-with-reason counter shape.** `--dry-run` prints the
  identical summary with nothing written. A `skipped` entry always carries a reason string
  (e.g. `blank Observatory.timezone`, `TBD window`, `unresolved site`).

- **D-06: A single run's failure is caught at the batch-loop level** (not per-run
  `transaction.atomic()`), recorded in the skipped-with-reason summary, and the batch
  continues to the next run. A mid-run failure leaves that run's already-written nights in
  place — accepted partial projection, matching Phase 25's precedent.

### Claude's Discretion

- The exact name of the shared per-run function (`reconcile_run` used as a placeholder).
- Whether `_project_calendar_event()`/`_calendar_event_title()` are deleted in the same
  commit that wires the staff actions to the reconciler, or in a preceding cleanup commit.
- Whether the per-stage breakdown (stage 0-4 counts) is worth adding to D-05's summary shape
  as a supplementary line, beyond the required created/updated/unchanged/skipped counts.
- Test organisation, and how `mark_cancelled`/`mark_weather_failure`'s existing
  `_set_run_status()` shape maps onto calling the shared reconcile function.

### Deferred Ideas (OUT OF SCOPE)

- Per-stage (0-4) breakdown in the command's summary output — offered during discussion, not
  chosen, not rejected. Left as Claude's Discretion.
- v2.3 items untouched here: adapter rewiring (ADAPT-01..03 — once shipped, `CampaignRun`
  rows will finally get a real, adapter-written `source` for queue runs; this is the
  "trigger condition" `26-DECISION.md` names for revisiting whether the bare-container-only
  queue verdict still holds); provenance-blind gap analysis (GAPB-01); status-vocabulary
  unification (STATUS-01/02); unused-allocation display (UNUSED-01).
- Rewiring the four ingest adapters themselves to create `CampaignRun`s — the reconciler
  reads existing adapter output, it does not change how classical/LCO/Gemini/CSV-import
  adapters write.
- Automatic merging of suspected duplicate associations, upstreaming the event→run link into
  `tom_calendar`, renaming `related_name='telescope_label_meta'`, any new dependency for
  reconciliation/field-diffing, `GenericForeignKey` for the event/record links, making the
  `run` link required (per REQUIREMENTS.md "Out of Scope" table).

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RECON-01 | One command projects/refreshes calendar events for every run; re-run changes nothing | "Code Examples" #1 (batch loop shape), "Don't Hand-Roll" (`insert_or_create_calendar_event` no-churn contract already proven idempotent by `TestCalendarNoChurn`) |
| RECON-02 | Classical run → one event/night (dip-corrected twilight); queue run → one bare `RUN:{pk}` container | "Architecture Patterns" Pattern 1/2, "Common Pitfalls" #1 (run-type classification gap) |
| RECON-03 | Class-wide run → single whole-window `RUN:{pk}` container | Pattern 2 (same container mechanism as queue-run stage 1 — class-wide is a `telescope_class`-carrying, site-less run; existing `campaign_views.py` satellite branch already does whole-window math to copy) |
| RECON-04 | `CampaignRunObservation`-linked night narrows to the record's window; completed → COMPLETED range | Pattern 3 (stages 3-4), `CampaignRunObservation`/`ObservationRecord` field reference in "Code Examples" #3 |
| RECON-05 | Reconciler never touches events it doesn't own | "Code Examples" #2 (ownership query pattern, `CalendarEventMeta.run__isnull` semantics) |
| RECON-06 | `--dry-run` reports with no writes; a failing run is skipped-with-reason, batch continues | "Code Examples" #1, D-05/D-06 above, "Common Pitfalls" #2 (blank-timezone case is now narrow, not the live blocker it once was) |
| RECON-07 | The 19 approved, site-resolved 3I/ATLAS runs become visible | "Common Pitfalls" #1 — **the real dev-DB rows are not yet classified for this to work automatically; flagged as a data-prerequisite, not a code gap** |
| RECON-08 | approve/resolve_site/mark_cancelled/mark_weather_failure each reconcile immediately | "Architecture Patterns" — exact call-site line numbers and existing try/except shapes documented below |
| RECON-09 | `backfill_range_calendar_events` retired, from code and the runbook | "Code Examples" #4 lists every file/test that references it |

</phase_requirements>

## Summary

Phase 29 is a refactor of an already-well-understood, already-tested projection path, not a
greenfield feature. `campaign_views.py`'s `_project_calendar_event()` (lines 457-553) is the
exact blueprint for the reconciler's classical/space math — the reconciler's job is to move
that logic (minus the retired `CAMPAIGN:` key scheme) into a new pure module
(`solsys_code/campaign_reconciler.py`), add the queue-run bare-container branch that
`_project_calendar_event()` never had, add the D-02 adopt-and-rekey step for classical nights
already linked via `CalendarEventMeta.run`, add stages 3-4 (narrow to `CampaignRunObservation`
data), and wire four existing call sites plus one new command onto it. Every helper the
reconciler needs — `insert_or_create_calendar_event()`, `sun_event()`, the
`Q(url=..) | Q(url__startswith=..)` ownership-scoping idiom, the
created/updated/unchanged/skipped-with-reason summary shape — already exists and is already
exercised by tests; nothing new needs inventing at the plumbing level.

The one real gap this research surfaces, and the single most important finding for planning:
**`CampaignRun.source` is the field 26-DECISION.md names as "the information the reconciler
needs to branch its event-projection strategy on run type," but every one of the 23 real,
approved, site-resolved `CampaignRun` rows on the live `3I/ATLAS` campaign (TargetList pk=3)
currently has `source='legacy'`.** None carry `lco_queue`/`gemini_queue`/`classical_file`.
There is no other field, and no existing code, that mechanically distinguishes "this run is
scheduled through a shared queue network" (e.g. `FTN/MuSCAT3` at F65, LCO's `ogg` site) from
"this run owns a specific night at a single classical facility" (e.g. `HCT` at N50) — the
8-QUEUE/11-CLASSICAL split named in `26-DECISION.md` was produced by a human reading
`telescope_instrument`/`site_raw` text, not by a queryable rule. `source` is staff-editable in
the Django admin for every non-`web` row (confirmed in `admin.py`), so the intended, in-scope
fix is a **manual one-time data-correction step** (staff set `source` on the known queue rows
via the admin) documented in the demo notebook/runbook, not new reconciler code. See "Common
Pitfalls" #1 and "Open Questions" #1 for the full detail and the recommended plan-level
handling.

**Primary recommendation:** build `reconcile_run(run)` as a straight port of
`_project_calendar_event()`'s ground/satellite branches (reusing `sun_event()` and
`insert_or_create_calendar_event()` verbatim), branch stage 1 purely on
`run.source in {LCO_QUEUE, GEMINI_QUEUE}` vs. everything else with a resolved single site
(matching `26-DECISION.md`'s locked verdict), branch stage 2 on `run.telescope_class` being
non-blank, and treat "run has no window yet" / "run's site never resolved and has no class"
as `skipped` reasons in the D-05 summary — then flag the `source`-population gap as a
pre-reconcile data-fix task, not a code task.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-run event projection math (stages 1-4) | API / Backend (pure module, `campaign_reconciler.py`) | — | Django ORM reads/writes, no HTTP concerns; must stay import-clean of `views.py`/`ephem_utils.py` (SPICE kernel load) per the milestone's locked module-home constraint |
| Batch loop + `--dry-run` reporting | API / Backend (management command) | — | `BaseCommand` subclass, thin wrapper over `campaign_reconciler.reconcile_run()` |
| Staff-action trigger (approve/resolve_site/mark_cancelled/mark_weather_failure) | API / Backend (Django views, `campaign_views.py`) | — | Synchronous inline call from existing `CampaignRunDecisionView` methods (D-04); no new HTTP surface |
| Calendar rendering of the resulting events | Frontend Server (SSR, existing `calendar.html`/`calendar_display_extras`) | — | Unchanged by this phase — the reconciler only writes `CalendarEvent` rows through the existing helper; rendering logic is out of scope |
| Ownership/ownership-scoping data (`CalendarEventMeta.run`) | Database / Storage | API / Backend (query pattern) | The FK already exists (Phase 27/28); the reconciler is a new *reader and writer* of it, not a new schema |

## Standard Stack

No new external dependency (locked by REQUIREMENTS.md's "Out of Scope" table and
`26-DECISION.md`'s rejection of `django-dirtyfields`/`FieldTracker`/`django-fsm`/Celery/
`rapidfuzz`). Every building block is already installed and already used by sibling modules:

### Core (all `[VERIFIED: codebase]` — read directly from the live repo, not assumed)

| Component | Location | Purpose | Why reuse it |
|-----------|----------|---------|--------------|
| `insert_or_create_calendar_event()` | `solsys_code/calendar_utils.py:482-542` | No-churn create-or-update for a `CalendarEvent` given a `lookup` dict + `fields` dict | Already the single write path every adapter uses; already proven no-churn via `_update_or_unchanged()` and covered by `TestCalendarNoChurn` |
| `sun_event(site, date, kind='sun')` | `solsys_code/telescope_runs.py:251-299` | Dip-corrected UTC sunset/sunrise for one observing night | Exact math `_project_calendar_event()`'s ground branch already uses; reconciler's stage-1 classical math is a direct reuse |
| `CalendarEventMeta` | `solsys_code/models.py:10-73` | Companion FK carrying `run`, `confirmed_by`, `confirmed_at` | The ownership mechanism RECON-05 depends on; already exists, no schema change needed |
| `CampaignRunObservation` | `solsys_code/models.py:344-405` | Links `CampaignRun` → `ObservationRecord`, one row per confirmed attribution | Stages 3-4's data source; already populated by Phase 28's attribution queue |
| `record_time_window()` | `solsys_code/calendar_utils.py:423-458` | Derives an `ObservationRecord`'s active `(start_time, end_time)` window (banner vs. placed) | Already shared between the LCO sync command and the attribution matcher; stage-3/4 narrowing reuses this instead of re-deriving scheduled_start/scheduled_end logic |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Django ORM (`Q`, `F`, conditional `.update()`) | Django 2.1+ (already installed, per CLAUDE.md tech stack) | Ownership-scoping queries, staleness-safe conditional writes | Every write in this phase — no new query mechanism needed |
| `django.core.management.base.BaseCommand` | stdlib-equivalent (Django) | The new `reconcile_campaign_runs` command | Matches every existing command's shape (`backfill_range_calendar_events.py`, `import_campaign_csv.py`) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A pure-Python module (`campaign_reconciler.py`) | Django signals (`post_save` on `CampaignRun`) | Rejected implicitly by D-03/D-04: the milestone wants one explicit, testable, synchronous function callable from both a loop and a single staff action — a signal handler is harder to call idempotently from a management command's `--dry-run` path and harder to unit-test in isolation |
| `source`-based run-type branching | A new heuristic keyed on `telescope_instrument`/site text (mirroring the spike's human judgment) | Rejected — REQUIREMENTS.md's "Out of Scope" table forbids new dependencies and CONTEXT.md's decisions never mention a new heuristic; `26-DECISION.md` explicitly names `source` as the intended mechanism. A text heuristic would also be unverifiable/untestable in the way `source in {...}` is not |

**Installation:** none — no new packages.

**Version verification:** not applicable — no new package versions to verify.

## Package Legitimacy Audit

Not applicable — this phase introduces zero new external packages (confirmed against
CONTEXT.md's locked decisions and REQUIREMENTS.md's "Out of Scope" table: "A new dependency
for reconciliation or field-diffing" is explicitly excluded). No `package-legitimacy check`
run was needed.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │  Trigger sources (two entry points, D-03)    │
                    │                                               │
   staff POST ──────┤  CampaignRunDecisionView                     │
   (approve/         │    .post() → approve() / _resolve_site() /  │
    resolve_site/     │    _set_run_status() (mark_cancelled/       │
    mark_cancelled/   │    mark_weather_failure)                    │
    mark_weather_     │         │                                   │
    failure)          │         │ calls reconcile_run(run) inline   │
                    │         │ (synchronous, D-04)                │
                    │         ▼                                     │
   `manage.py        │  ┌──────────────────────────────┐            │
   reconcile_         ├─►│ campaign_reconciler.py        │◄─── loop over
   campaign_runs`     │  │  reconcile_run(run) -- one    │     every CampaignRun
   (batch, --dry-run) │  │  CampaignRun in, stage 0-4    │     (RECON-01)
                    │  │  branch + write, out           │            │
                    │  └───────────┬────────────────────┘            │
                    └──────────────┼─────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────┐
                    │ Stage branch on run state (read-only fields) │
                    │                                                │
                    │  window_start is None?           → skip       │
                    │    (stage 0, "allocated, no window yet")      │
                    │  telescope_class non-blank?       → stage 2   │
                    │    (class-wide container, RECON-03)           │
                    │  source in {LCO_QUEUE,GEMINI_QUEUE}?          │
                    │    → stage 1 queue (bare RUN:{pk} container,  │
                    │       RECON-02)                                │
                    │  else (resolved single site)      → stage 1   │
                    │    classical (per-night RUN:{pk}:{date},      │
                    │       RECON-02, D-02 adopt-or-mint)            │
                    │  CampaignRunObservation exists for a night?   │
                    │    → stage 3/4 narrow (RECON-04)               │
                    └──────────────┬────────────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────────────────┐
                    │ Ownership-scoped write (RECON-05)               │
                    │  1. Find events already owned by this run       │
                    │     (CalendarEventMeta.run_id=run.pk, or        │
                    │      companion FK by (run, night) for adopt)    │
                    │  2. insert_or_create_calendar_event(lookup,     │
                    │     fields) -- create/update/unchanged           │
                    │  3. Set/keep CalendarEventMeta.run on write      │
                    │  Never touch an event with no companion row, or │
                    │  one whose run FK points elsewhere/is unset.    │
                    └──────────────┬────────────────────────────────┘
                                   │
                    ┌──────────────▼────────────────────────────────┐
                    │ CalendarEvent (tom_calendar) + CalendarEventMeta │
                    │  → rendered by existing calendar.html/           │
                    │    calendar_display_extras (unchanged)            │
                    └──────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
solsys_code/
├── campaign_reconciler.py         # NEW — reconcile_run(run), stage branch functions,
│                                   #  ownership-scoping query, RUN: key builders
├── campaign_views.py               # _project_calendar_event()/_calendar_event_title()
│                                   #  DELETED; approve()/_resolve_site()/_set_run_status()
│                                   #  import and call campaign_reconciler.reconcile_run()
├── management/commands/
│   ├── reconcile_campaign_runs.py  # NEW — loops reconcile_run() over CampaignRun.objects
│   │                                #  .all(), --dry-run, D-05 summary
│   └── backfill_range_calendar_events.py  # DELETED (RECON-09)
├── tests/
│   ├── test_campaign_reconciler.py         # NEW — unit tests for reconcile_run() itself
│   ├── test_reconcile_campaign_runs.py     # NEW — command/--dry-run/summary tests
│   ├── test_campaign_approval.py           # MODIFIED — TestCalendarProjection's CAMPAIGN:
│   │                                        #  assertions rewritten to RUN: (see Pitfalls #3)
│   └── test_backfill_range_calendar_events.py  # DELETED (RECON-09)
docs/
├── notebooks/pre_executed/
│   └── reconcile_campaign_runs_demo.ipynb  # NEW (CLAUDE.md paired-doc rule)
└── runbooks/
    └── telescope_runs_calendar.rst         # MODIFIED (see "Code Examples" #4)
```

### Pattern 1: Classical per-night projection (RECON-02, ported from `_project_calendar_event`)

**What:** For a run resolved to a single ground `Observatory` and NOT queue-scheduled, loop
`window_start..window_end` inclusive, call `sun_event(run.site, night, kind='sun')` per night,
and either adopt an existing `CalendarEventMeta.run`-linked blank-`url` event for that
(run, night) pair (D-02) or mint a new `RUN:{run_pk}:{date}` event.

**When to use:** `run.source not in {LCO_QUEUE, GEMINI_QUEUE}` and `run.site` is resolved and
`run.site.observations_type != SATELLITE_OBSTYPE` and `run.telescope_class` is blank.

**Example (existing code to port, not hypothetical):**
```python
# Source: solsys_code/campaign_views.py:528-553 (the exact math to reuse; the retired
# CAMPAIGN:{run.pk}/CAMPAIGN:{run.pk}:{date} keys become RUN:{run.pk}/RUN:{run.pk}:{date})
n_nights = (run.window_end - run.window_start).days + 1
is_range = n_nights > 1
for i in range(n_nights):
    night = run.window_start + timedelta(days=i)
    try:
        sunset, sunrise = sun_event(run.site, night, kind='sun')
    except ValueError:
        # D-06: caught at the batch-loop level in the new code, not re-raised past
        # reconcile_run() the way _project_calendar_event() lets callers decide.
        raise
    night_fields = dict(event_fields)
    night_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    night_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    url = f'RUN:{run.pk}' if not is_range else f'RUN:{run.pk}:{night.isoformat()}'
    insert_or_create_calendar_event({'url': url}, fields=night_fields)
```

**D-02's adopt step, before the mint above:** query for a `load_telescope_runs`-created event
already linked to this run for this exact night, via `CalendarEventMeta`:
```python
# Site-local night derivation is load-bearing (26-DECISION.md Criterion 3): compare each
# candidate event's start_time, converted into run.site.timezone, to the target night --
# never the naive UTC date, since a real measured instance (event pk=54) disagrees between
# the two derivations by one day.
from zoneinfo import ZoneInfo
candidates = CalendarEventMeta.objects.filter(
    run_id=run.pk, event__telescope=run.telescope_instrument
).select_related('event')
for meta in candidates:
    local_date = meta.event.start_time.astimezone(ZoneInfo(run.site.timezone)).date()
    if local_date == night:
        # Re-key this event's url to RUN:{run.pk}:{night} in place, in addition to
        # refreshing its start/end/title fields -- CalendarEvent.url has no unique
        # constraint at the DB level (confirmed: tom_calendar.models.CalendarEvent does
        # not declare one), so a direct field assignment + save is safe, but must go
        # through insert_or_create_calendar_event()'s _update_or_unchanged() no-churn path
        # (or an equivalent explicit diff) rather than an unconditional .save() -- see
        # Open Questions #2 for the exact mechanism to lock during planning.
        ...
```

### Pattern 2: Queue/class-wide whole-window container (RECON-02/RECON-03)

**What:** One bare `RUN:{run_pk}` event spanning `window_start`..`window_end`, titled from
the run's telescope/instrument and window, never touching any `ObservationRecord`-derived
event.

**When to use:** `run.source in {LCO_QUEUE, GEMINI_QUEUE}` (queue-scheduled, RECON-02) OR
`run.telescope_class` is non-blank (class-wide, RECON-03) OR `run.telescope_class == 'SPACE'`
(space mission — reuses the existing satellite whole-day-span math, RECON-03's sibling case).

**Example (existing satellite-branch code to generalize):**
```python
# Source: solsys_code/campaign_views.py:503-514 -- the exact whole-window-span mechanism
# already ships for satellite runs; the queue/class-wide branches are the same shape with
# a different date-range source (window_start/window_end still drive it either way).
event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
insert_or_create_calendar_event({'url': f'RUN:{run.pk}'}, fields=event_fields)
```
Measured stability evidence for this exact mechanism (bare key, no date component) already
exists: `26-DECISION.md`'s "Three-way comparison against pk=1's real window" table shows
`span created=1 updated=0`, an idempotent re-run (`created=0 updated=0`), and
`KEY_SET_STABLE=True` under a real window-narrowing edit (only `start_time`/`end_time`
changed, an `updated=1` action) — confirming this branch alone satisfies RECON-01's
idempotency requirement without any extra guard code.

### Pattern 3: Stages 3-4, narrowing to `ObservationRecord` data (RECON-04)

**What:** For a queue-scheduled run whose window has a confirmed `CampaignRunObservation`
link, the per-night detail comes from the existing, **unmodified** LCO/Gemini sync commands
(`sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`) — the reconciler
does not write these events at all. `CampaignRunObservation` merely tells the reconciler these
events exist and belong to the run (for reporting/summary purposes, or for a future need); the
reconciler's own write surface for a queue run is *only* the bare `RUN:{pk}` container from
Pattern 2, per the settled "Queue-run projection" verdict — RECON-04's "narrows to that
record's window" and "shows COMPLETED" behavior is **already implemented** by
`sync_lco_observation_calendar.py`'s `_build_event_fields()`/`_time_window()`
(`calendar_utils.py:423-458`, promoted from the sync command in Phase 28-02) and
`_FAILURE_PREFIX_BY_STATUS` — the reconciler's job for RECON-04 is to leave these events
alone (ownership rule, RECON-05), not to reimplement their narrowing.

**Field reference for `CampaignRunObservation`/`ObservationRecord` (from `models.py`):**
```python
# Source: solsys_code/models.py:344-405 + tom_observations.models.ObservationRecord (installed)
run.observation_links.all()                    # CampaignRunObservation queryset for this run
link.observation_record                          # -> ObservationRecord
link.observation_record.scheduled_start          # None until LCO places it
link.observation_record.scheduled_end
link.observation_record.status                   # e.g. 'COMPLETED', 'PENDING', 'WINDOW_EXPIRED'
link.observation_record.parameters['start']/['end']  # banner-stage fallback (record_time_window())
```

### Anti-Patterns to Avoid

- **Re-deriving the successful-terminal-state set inline.** `sync_lco_observation_calendar.py`
  already has the correct derivation (`get_terminal_observing_states()` minus
  `get_failed_observing_states()`, quick task 260723-r5g's fix) — do not special-case
  `status == 'COMPLETED'` directly in the reconciler; STATUS-02 (v2.3, deferred) is the
  eventual general fix, but for Phase 29 the reconciler should not write these events at all
  (Pattern 3), sidestepping the question entirely.
- **Re-raising `sun_event()`'s `ValueError` out of `reconcile_run()` uncaught.**
  `_project_calendar_event()`'s existing contract lets *callers* decide revert-vs-not; D-06
  instead wants the batch-loop-level catch to be the *only* catch point — `reconcile_run()`
  itself should let the ValueError propagate (matching D-06's "no `transaction.atomic()`
  wrap," meaning no per-run try/except either) so the command's loop and the staff-action call
  sites each apply their own existing (different) handling.
- **Treating `url` as unique for the D-02 re-key write.** `tom_calendar.models.CalendarEvent`
  declares no unique constraint on `url` (confirmed: not present in the installed model) — the
  adopt step must match by `CalendarEventMeta.run_id` + site-local night, not by any assumed
  DB-level uniqueness of the new `url` value being written.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update a `CalendarEvent` with no-churn diffing | A bespoke get-or-create + field comparison | `insert_or_create_calendar_event()` (`calendar_utils.py`) | Already the shared contract every adapter uses; already tested (`TestCalendarNoChurn`) |
| Dip-corrected sunset/sunrise for a night | Astropy `AltAz`/`get_sun` calls inline | `sun_event(site, date, kind='sun')` (`telescope_runs.py`) | Exact math this phase's own `26-DECISION.md` cites as "grounded in" |
| Ownership-scoped event lookup by run pk | A bespoke `url.startswith()` Python filter | `Q(url=f'RUN:{run.pk}') | Q(url__startswith=f'RUN:{run.pk}:')` | Direct precedent already shipped at `campaign_views.py:876-878`; mirror verbatim per `26-DECISION.md`'s own recommendation |
| Command summary reporting | A new dict/dataclass shape | `created_count`/`updated_count`/`unchanged_count`/`skipped_count` int locals + per-row `stderr.write()` (D-05) | Byte-for-byte the shape `import_campaign_csv.py` and `backfill_range_calendar_events.py` already use — no dict/dataclass wrapper exists in either precedent, just plain counters and printed lines |
| Successful-terminal-observation-state detection | `record.status == 'COMPLETED'` | `sync_lco_observation_calendar.py`'s existing derivation, or (preferably) don't write these events at all (Pattern 3) | Facility-specific `get_terminal_observing_states()`/`get_failed_observing_states()` differ; hardcoding one string is the exact anti-pattern STATUS-02 exists to later fix |

**Key insight:** every piece of machinery this phase needs was built for a sibling feature and
is already unit-tested in isolation. The actual engineering work is orchestration (branching on
run state, four call sites, one new command) plus the two genuinely new pieces:
D-02's adopt-and-rekey step (no precedent — the adopt/gap-fill *measurement* exists in
`26-DECISION.md` but no adopt implementation was ever shipped) and stage-1 queue vs. classical
branching (no precedent — `_project_calendar_event()` never had a queue branch at all).

## Common Pitfalls

### Pitfall 1: `CampaignRun.source` cannot yet mechanically distinguish queue vs. classical for the real acceptance-criterion data

**What goes wrong:** RECON-02 requires branching stage-1 projection on whether a run is
"classically-scheduled" or "queue-scheduled." `26-DECISION.md`'s domain correction (point 6)
states `CampaignRun.source` is the field that "already distinguishes classical file / LCO
queue / Gemini queue / etc.," and recommends the reconciler branch on it. **Measured directly
against the live dev DB (2026-08-04):** every one of the 23 approved, site-resolved
`CampaignRun` rows on the real `3I/ATLAS` campaign (`TargetList.objects.get(pk=3)`) has
`source='legacy'`. None carry `lco_queue`/`gemini_queue`/`classical_file` — the vocabulary
exists (declared in `CampaignRun.Source`, `models.py:106-126`) but was never populated for
these rows, because v2.3's ADAPT-01..03 (the adapters that would write it) haven't shipped and
these rows predate CANON-01. `26-DECISION.md`'s own "8 QUEUE / 11 CLASSICAL / 0 SPACE" split
of the RECON-07 baseline was produced by a **human reading `telescope_instrument`/`site_raw`
text** (e.g. recognizing `FTN/MuSCAT3` at obscode F65 as LCO's `ogg` site), not by any
queryable field or existing code path. There is no reverse mapping in this codebase from an
MPC obscode (e.g. `E10`, `F65`) to an LCO/Gemini/SOAR internal site code — confirmed by
grepping `campaign_utils.py`/`solsys_code_observatory/*.py` for `coj`/`ogg`/`sor` and finding
none. If the reconciler branches purely on `source` (the CONTEXT-aligned, in-scope choice),
running it today against the real DB will render **every** approved, site-resolved 3I/ATLAS
run as a classical per-night projection — including the 8 rows a human classified as
queue-scheduled — because `source='legacy'` falls into the "not LCO_QUEUE/GEMINI_QUEUE" branch.

**Why it happens:** `source` is a CANON-01 (Phase 27) field designed for a data-lifecycle that
only fully closes once v2.3's adapter rewiring ships; Phase 29 arrives after CANON-01 but
before ADAPT-01..03, in the gap where the field exists but most real rows were never
classified by it.

**How to avoid:** Branch stage 1 purely on `run.source in {LCO_QUEUE, GEMINI_QUEUE}` in the
reconciler's own code (this is the correct, CONTEXT-aligned, forward-compatible rule — it is
exactly right once ADAPT-01..03 ships). For RECON-07's flagship acceptance criterion to
actually render the 8-QUEUE/11-CLASSICAL split against the real dev DB, plan a **manual
data-correction step** — staff set `source` to `lco_queue` (or `gemini_queue`) on the specific
8 real rows the spike identified, via the Django admin (`source` is staff-editable for any
non-`web` row, confirmed in `admin.py:get_readonly_fields`) — as a one-time task, documented in
the demo notebook and/or runbook, run *before* the first full reconcile sweep. This is a data
task, not a code task, and should not be implemented as reconciler logic (a text-based
heuristic would violate CONTEXT.md's "no new dependency"/no-scope-creep posture and would be
unverifiable). Flag this explicitly for the planner: either (a) scope a `checkpoint:human-verify`
task instructing staff to reclassify the known rows before the acceptance demo, or (b) treat
"correct `source` on demo data" as a pre-condition documented in the paired notebook rather than
a phase deliverable. See "Open Questions" #1.

**Warning signs:** If a test or the demo notebook asserts an 8-vs-11 split against fixtured
`CampaignRun` rows without explicitly setting `source=LCO_QUEUE`/`GEMINI_QUEUE` on the queue
ones, the assertion will silently pass only because the *fixture* set `source` correctly, not
because the reconciler derived it — this is fine and correct for tests (which control their own
fixtures), but must not be mistaken for evidence the real-DB acceptance criterion will render
correctly without the data-fix step above.

### Pitfall 2: The "known blank-`Observatory.timezone` rows" (ROADMAP criterion 4) are narrower than they sound — already effectively closed for the classical-projection path

**What goes wrong:** ROADMAP's success criterion 4 and RECON-06 both reference "the known
blank-`Observatory.timezone` rows" as a case the reconciler must report-and-skip. Read as a
green light to reuse `_project_calendar_event()`'s existing try/except-and-log pattern, this
sounds like a live, common failure mode.

**Why it happens / what's actually true:** **Measured directly against the live dev DB:** only
5 `Observatory` rows have a blank `timezone` today (`Observatory.objects.filter(timezone='')`),
and all 5 are `SATELLITE_OBSTYPE` (WISE, JWST, Roman, HST, Swift) — sites with no `lat`/`lon`
at all, so the ground-only backfill migration (`solsys_code_observatory/migrations/0003_
backfill_observatory_timezone.py`, and the live Tier-2 `MPCObscodeFetcher.to_observatory()`
`timezonefinder` backfill from quick task 260716-h8c, `utils.py:144-154`) never had coordinates
to derive a timezone from in the first place. Every `CampaignRun` currently approved, site-
resolved, and pointing at one of these 5 satellite sites (pks 8, 12, 13, 21, 27, 28) takes the
**satellite whole-day-span branch** (Pattern 2), which never calls `sun_event()` at all — so
the blank timezone never actually raises `ValueError` for any real row today. The case CR-01's
regression test (`test_resolve_blank_timezone_site_keeps_review_flag_and_creates_no_event`)
exercises is a **synthetic ground-type Observatory with `timezone=''`** (`obscode='T99'`), not
a real dev-DB row — it proves the code path is correct, not that it is currently live.

**How to avoid:** Still implement the `sun_event()` `ValueError` → `skipped` handling (D-06
requires it structurally, and a future Tier-2-resolved ground site with missing lat/lon could
still hit it), but do not treat this as the reconciler's primary or most-likely failure mode
against real data — it is a defensive case, already effectively closed for the flagship
acceptance data by Phase 27's backfill. Test it with a synthetic fixture (mirroring the
existing `T99` pattern), not by expecting to reproduce it against the real DB.

**Warning signs:** A plan that budgets significant real-data debugging time for
"blank-timezone runs" is over-scoping this — none of the 3I/ATLAS acceptance-criterion rows
will hit it.

### Pitfall 3: `TestCalendarProjection`/`TestSitesNeedingReview`/`TestRunStatusChange` in `test_campaign_approval.py` assert `CAMPAIGN:`-keyed events and patch `solsys_code.campaign_views.sun_event`/`_project_calendar_event`/`insert_or_create_calendar_event` — all of this breaks under D-01

**What goes wrong:** `test_campaign_approval.py` (2,600+ lines) contains ~15+ test methods
across `TestCalendarProjection`, `TestSitesNeedingReview`, `TestRunStatusChange`, and
`TestCalendarNoChurn` that (a) assert on `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}')`,
(b) `patch('solsys_code.campaign_views.sun_event', ...)`,
`patch('solsys_code.campaign_views._project_calendar_event', ...)`, or
`patch('solsys_code.campaign_views.insert_or_create_calendar_event', ...)`. Once D-01 deletes
`_project_calendar_event()`/`_calendar_event_title()` and rewires the four call sites to import
`campaign_reconciler` instead, every one of these assertions and patch targets breaks — not as
a side effect to discover during verification, but as a **direct, known, first-order
consequence of the locked D-01 decision** that must be planned as rewrite work, not left as an
unplanned test-suite casualty.

**Why it happens:** These tests were written against `_project_calendar_event()` directly
(Phase 25's FIX-01..07 and Phase 21-23's approval-queue work) and haven't been touched since;
D-01 explicitly deletes the function they patch and the key scheme they assert on.

**How to avoid:** Scope a dedicated task (or a clearly-bounded portion of the call-site-rewiring
task) to rewrite `TestCalendarProjection`, `TestSitesNeedingReview`'s calendar-projection
assertions, `TestRunStatusChange`, and `TestCalendarNoChurn` to (a) assert `RUN:` keys instead
of `CAMPAIGN:` keys, and (b) patch `solsys_code.campaign_reconciler.sun_event`/
`insert_or_create_calendar_event` (or whatever the reconciler module's own import names end up
being) instead of the now-deleted `campaign_views` targets. This is a large, mechanical but
non-trivial rewrite — do not underestimate it as "delete two functions."

**Warning signs:** `ruff check .`/`ruff format --check .` will stay green even if these tests
are simply left broken (patch targets that no longer exist raise `AttributeError` at test run
time, not at lint time) — only `./manage.py test solsys_code` surfaces this, so it must be run,
not assumed clean, after the call-site rewiring.

## Code Examples

### 1. Exact retirement checklist for `backfill_range_calendar_events` (RECON-09)

```text
# Source: grep -rln "_project_calendar_event|_calendar_event_title|backfill_range_calendar_events"
# solsys_code/ docs/ (run 2026-08-04, exhaustive)
solsys_code/campaign_views.py                          # _calendar_event_title()/_project_calendar_event()
                                                          # definitions (lines 445-554) + 3 call sites
                                                          # (approve ~line 649, resolve_site ~line 803,
                                                          # the run-status calendar-sync loop uses
                                                          # insert_or_create_calendar_event directly, not
                                                          # _project_calendar_event, at ~line 886)
solsys_code/management/commands/backfill_range_calendar_events.py  # DELETE the whole file
solsys_code/tests/test_backfill_range_calendar_events.py           # DELETE the whole file (124 lines)
solsys_code/tests/test_campaign_approval.py             # rewrite (see Pitfall 3)
docs/runbooks/telescope_runs_calendar.rst                # DELETE "How do I backfill calendar events for
                                                          # older approved range-window runs?" section
                                                          # (lines 473-494); remove its command-cheat-sheet
                                                          # row (lines 565-567); remove its Troubleshooting
                                                          # mention (line ~584, "backfill_range_calendar_
                                                          # events, and any future projection")
solsys_code/tests/test_admin.py                          # comment-only reference (line 1010) -- update
                                                          # wording, no functional dependency
```

### 2. The ownership-scoping query pattern to mirror (RECON-05)

```python
# Source: solsys_code/campaign_views.py:876-878 (existing, shipped, exact precedent)
matching_events = CalendarEvent.objects.filter(
    Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
)
# The RUN: analogue (26-DECISION.md's own stated recommendation, verbatim):
matching_events = CalendarEvent.objects.filter(
    Q(url=f'RUN:{run.pk}') | Q(url__startswith=f'RUN:{run.pk}:')
)
```

For "the event(s) already attributed to run X" (as opposed to "keyed under run X's namespace"
— the ownership check, not the identity check), there is **no existing manager/queryset
helper** — the closest precedent is `AttributionDecisionView._do_confirm_event()`'s direct
filter:
```python
# Source: solsys_code/campaign_views.py:1317 (the exact query shape to mirror for "find this
# run's owned events")
CalendarEventMeta.objects.filter(event_id=orphan_pk, run__isnull=True)
# For the reconciler's "give me every event this run already owns":
CalendarEventMeta.objects.filter(run_id=run.pk).select_related('event')
```
The reconciler needs to write this itself — it is a two-line query, not a gap in existing
tooling, but there is genuinely no pre-existing helper function to import.

### 3. `sun_event()`'s exact signature and raise contract (reused verbatim by Pattern 1)

```python
# Source: solsys_code/telescope_runs.py:251-299
def sun_event(site: Observatory, date: date_cls, kind: str) -> tuple[Time, Time]:
    """...
    Raises:
        ValueError: if kind is not 'sun' or 'dark'; if site.timezone is
            unset; or if the solar altitude does not cross threshold
            exactly twice in the 24h window following local noon.
    """
```
`kind='sun'` (dip-corrected, not `kind='dark'`) is the convention `_project_calendar_event()`'s
ground branch already locks in — the reconciler's classical stage-1 math must use the same
`kind='sun'` value, matching RECON-02's "sunset-to-sunrise twilight" wording exactly (not the
-15° dark window, which `sun_event(kind='dark')` computes but nothing in this phase's
requirements calls for).

### 4. `import_campaign_csv`'s exact summary shape (D-05's literal precedent)

```python
# Source: solsys_code/management/commands/import_campaign_csv.py:400-408 (the exact shape --
# plain int counters, no dict/dataclass wrapper, one final stdout.write() line)
self.stdout.write(
    f'Done. created: {created_count}, '
    f'updated: {updated_count}, '
    f'unchanged: {unchanged_count}, '
    f'skipped: {skipped_count}, '
    f'site_needs_review: {site_needs_review_count}, '
    f'window_needs_review: {window_needs_review_count}, '
    f'site_preserved: {site_preserved_count}'
)
```
`backfill_range_calendar_events.py:90-102` is the `--dry-run`-branching precedent (a
`would_backfill_count` counter substituted for `backfilled_count` under `--dry-run`, sharing
the same `candidates`/`skipped_count` locals) — mirror this shape for
`reconcile_campaign_runs`'s own `--dry-run`: identical stdout line, `would_create`/
`would_update`/`would_unchanged` counters swapped in for `created`/`updated`/`unchanged`, no
writes issued in that branch. Per-row skip reasons are reported via `self.stderr.write(...)`
per row (matching `import_campaign_csv.py:152-157`'s pattern), not accumulated into a
structured list in memory — D-05's "itemized, not a bare count" is satisfied by this existing
stderr-per-row convention, not by inventing a new reporting data structure.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `CAMPAIGN:{run.pk}[:date]`-keyed events, written only on approve/resolve_site clicks, never re-verified | `RUN:{run.pk}[:date]`-keyed events, written by one idempotent function callable from a batch command or a staff click | This phase (D-01) | Every approved run becomes projectable regardless of when it was approved; re-running the command is provably a no-op |
| `backfill_range_calendar_events` (one-off, narrow: only range-window runs missing an event) | `reconcile_campaign_runs` (general: every run, every stage, `--dry-run`) | This phase (RECON-09) | One command replaces a whole class of future one-off backfill commands — this is explicitly named in the phase description as "retiring the backfill-command-per-gap pattern for good" |
| `event.telescope_label_meta.run` always unset for every FOMO-created event (WR-03 gap, `27-REVIEW.md`) | The reconciler sets `CalendarEventMeta.run` at creation for every event it mints or adopts | This phase (D-01/D-02, closing WR-03) | The "Campaign run" modal block (currently manual-admin-only, `telescope_runs_calendar.rst`'s "Why doesn't the calendar pop-up show a 'Campaign run' block?" section) becomes automatic for reconciler-owned events — that runbook section needs updating, not just the two sections RECON-09 names |

**Deprecated/outdated:**
- `_project_calendar_event()`/`_calendar_event_title()` (`campaign_views.py:445-554`): deleted
  outright per D-01, not deprecated in place (zero live `CAMPAIGN:` rows to migrate).
- `backfill_range_calendar_events` management command: deleted per RECON-09.
- The runbook's "manual admin FK picker is the only way to link a Campaign run block" framing
  (`telescope_runs_calendar.rst` lines 496-533): factually superseded by this phase for any
  event the reconciler owns — the admin path remains valid only for events the reconciler
  never touches (unattributed/dismissed-attribution orphans).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Stage 1's queue-vs-classical branch should key on `run.source in {LCO_QUEUE, GEMINI_QUEUE}` alone, with no fallback text heuristic | Common Pitfalls #1, Summary | If wrong, the flagship RECON-07 demo needs a different (more automatic) mechanism than a manual admin data-fix; this is `[ASSUMED]` as the CONTEXT-aligned reading of `26-DECISION.md`'s domain-correction point 6, not confirmed by an explicit CONTEXT.md decision naming the exact branch condition |
| A2 | Stage 2's class-wide branch should key on `run.telescope_class` being non-blank AND not `'SPACE'` (with SPACE handled by the existing satellite whole-day-span math, distinct titling) | Pattern 2 | If wrong, a SPACE-classed run without a resolved site could be mis-projected as a generic class-wide container with a misleading title; this is `[ASSUMED]` — `26-DECISION.md` documents 2m0/1m0/0m4 class-wide behavior and SPACE's satellite-style whole-window behavior separately but does not explicitly state whether they share one code branch or two in the reconciler |
| A3 | The D-02 adopt step matches an existing blank-`url` classical event to a `(run, night)` pair by comparing `CalendarEventMeta.run_id` + the event's `telescope` field + a site-local-date conversion of `start_time`, not by any other combination of fields | Pattern 1 | If wrong (e.g. if `telescope` isn't a reliable enough discriminator when a run's `telescope_instrument` free text doesn't exactly match the classical event's stored `telescope` field), the adopt step could either miss a real match (falling through to gap-fill/mint, contradicting D-02) or match the wrong event; this is `[ASSUMED]` — CONTEXT.md's D-02 states the *outcome* (find the row "via the companion FK for this run + night") but not the exact query shape, which is genuinely new code with no existing precedent to verify against |

**If this table is empty:** N/A — see entries above; all three concern the *exact mechanics*
of decisions that are directionally locked in CONTEXT.md but whose precise implementation was
explicitly left to be worked out during planning/execution (CONTEXT.md's "what research needs
to nail down" list, items 2 and 4).

## Open Questions

1. **How should RECON-07's real-data acceptance criterion actually be satisfied, given
   `source` is uniformly `'legacy'` on the real 3I/ATLAS rows?**
   - What we know: `26-DECISION.md` names `source` as the intended discriminator; the real
     dev-DB rows don't carry it; `source` is staff-editable via the admin for non-`web` rows.
   - What's unclear: whether the plan should include an explicit `checkpoint:human-verify` task
     instructing staff to set `source` on the known 8 queue rows before the demo/acceptance
     check, whether this belongs in the paired demo notebook as a documented prerequisite step
     instead, or whether the planner should treat "renders correctly once `source` is set" as
     sufficient and treat the data-fix as an out-of-plan operator action.
   - Recommendation: raise this explicitly during `/gsd-plan-phase` or a pre-planning
     discussion — it changes whether a task in the plan needs to touch real data, and CLAUDE.md's
     workflow-enforcement rules require going through a GSD command for any direct repo/data
     change, so a manual admin data-fix should itself be scoped as a task (or explicitly
     deferred to the operator) rather than silently assumed.

2. **What is the exact mechanism for D-02's "re-key an adopted event's `url` in place"?**
   - What we know: `insert_or_create_calendar_event()`'s `lookup` parameter is used for
     `CalendarEvent.objects.get_or_create(**lookup, ...)` — it cannot be handed a *different*
     lookup (e.g. `telescope`+`start_time` window) to find a row and then write a *new* `url`
     onto it, because `get_or_create()`'s `defaults` never touches fields that are also lookup
     keys, and `url` would need to move from "found by" to "written to."
   - What's unclear: whether the reconciler should (a) fetch the `CalendarEvent` directly via
     the `CalendarEventMeta` companion query (Pattern 1's sketch above) and call
     `calendar_utils._update_or_unchanged()` directly (it's a module-private function, currently
     only called from within `calendar_utils.py` itself — using it externally would be a new
     cross-module dependency on a "private" helper, the same anti-pattern
     `backfill_range_calendar_events.py`'s docstring calls out for importing a private view
     helper), or (b) `calendar_utils.py` should grow a new small public function for
     "update this specific event's url+fields, no-churn" that both the D-02 adopt step and any
     future similar need can call.
   - Recommendation: plan this as an explicit design task-1 decision (mirroring how
     `26-05-PLAN.md` task 1 was where the adopt-vs-gap-fill/queue-container verdicts got locked)
     — likely resolved by adding a second small public helper to `calendar_utils.py` alongside
     `insert_or_create_calendar_event()`, since `_update_or_unchanged()`'s no-churn diffing logic
     is exactly what's needed and re-implementing it inside `campaign_reconciler.py` would
     duplicate, not reuse, the "Don't Hand-Roll" no-churn contract.

## Environment Availability

Not applicable — this phase has no external service/tool dependencies beyond what's already
installed and used by the modules it extends (Django, the existing SQLite dev DB, `astropy`/
`zoneinfo` via `telescope_runs.sun_event()`, already verified working in this repo). No new
CLI tools, databases, or network services are introduced.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`), NOT pytest — this phase touches `solsys_code/` app code exclusively |
| Config file | none — Django settings module `src.fomo.settings` (per CLAUDE.md) |
| Quick run command | `python manage.py test solsys_code.tests.test_campaign_reconciler` (new file, once created) |
| Full suite command | `python manage.py test solsys_code` (per CLAUDE.md/user memory: use `python manage.py`, not `./manage.py` — see "FOMO test-suite gotchas" memory) |

**CLAUDE.md/memory note carried into this section verbatim:** exclude
`test_views.TestEphemeris` if ever running the full unfiltered Django suite (it segfaults in
native ASSIST) — not relevant to this phase's own new tests, but relevant if a plan task ever
runs `python manage.py test` unfiltered as a smoke check.

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RECON-01 | Command re-run is a no-op (no new rows, no `modified` churn) | integration (Django `TestCase`) | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestIdempotency -v2` | ❌ Wave 0 — new file |
| RECON-02 | Classical run → per-night dip-corrected events; queue run → bare container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassicalStage1 solsys_code.tests.test_campaign_reconciler.TestQueueStage1` | ❌ Wave 0 — new file |
| RECON-03 | Class-wide run → single whole-window container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassWideStage2` | ❌ Wave 0 — new file |
| RECON-04 | Confirmed record narrows/completes correctly | unit — but likely a **non-event** test (Pattern 3: reconciler does not write these) | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestQueueOwnershipDoesNotTouchRecordEvents` | ❌ Wave 0 — new file |
| RECON-05 | Reconciler never touches hand-created/un-attributed events | unit — fixture deliberately places an un-owned event in the same window | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestOwnershipScoping` | ❌ Wave 0 — new file |
| RECON-06 | `--dry-run` writes nothing; a failing run is skipped, batch continues | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestDryRun solsys_code.tests.test_reconcile_campaign_runs.TestFailureIsolation` | ❌ Wave 0 — new file |
| RECON-07 | Real-shape 19-run scenario becomes visible | integration (fixtured, not live-DB — see Pitfall 1) | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestRealDataShapeScenario` | ❌ Wave 0 — new file |
| RECON-08 | Each staff action reconciles immediately | integration, extending existing `test_campaign_approval.py` classes | `python manage.py test solsys_code.tests.test_campaign_approval` | ✅ existing file, needs rewriting (Pitfall 3) |
| RECON-09 | `backfill_range_calendar_events` gone from code + runbook | negative test / grep-based check, or simply file-deletion (no test needed once files are deleted) | n/a (file deletion is self-verifying) | n/a |

### Sampling Rate

- **Per task commit:** `python manage.py test solsys_code.tests.test_campaign_reconciler` (or
  the most specific new-file test module for that task)
- **Per wave merge:** `python manage.py test solsys_code` (excluding `test_views.TestEphemiris`
  per the known segfault, if the wave touches anything importing `ephem_utils` transitively —
  `campaign_reconciler.py` itself must NOT import `ephem_utils`/`views`, per the milestone's
  locked module-home constraint, so this exclusion should not actually be needed for this
  phase's own new tests)
- **Phase gate:** `python manage.py test solsys_code` green, plus `ruff check .` and
  `ruff format --check .` clean, before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `solsys_code/tests/test_campaign_reconciler.py` — unit tests for `reconcile_run()` itself,
      covering RECON-02/03/04/05, isolated from the Django view/command layer
- [ ] `solsys_code/tests/test_reconcile_campaign_runs.py` — command-level tests for RECON-01/06/07
- [ ] No new fixtures/conftest needed beyond what `test_campaign_approval.py` already
      establishes (`Observatory.objects.create(...)` with resolvable `timezone`, plain
      `CampaignRun.objects.create(...)`) — this phase's tests never fixture a `tom_targets.Target`
      individually either (mirroring `test_campaign_approval.py`'s own noted convention), so
      CLAUDE.md's `NonSiderealTargetFactory` rule likely does not arise unless a new test needs
      an actual `Target` instance (e.g. for `CampaignRunObservation`/`ObservationRecord`
      fixtures in the RECON-04 tests) — **if it does, use
      `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory`**,
      per CLAUDE.md.
- [ ] Framework install: none — Django test runner already configured

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Unchanged — `StaffRequiredMixin` already gates the four staff-action call sites |
| V3 Session Management | no | Unchanged |
| V4 Access Control | yes | The new management command is a server-side CLI tool (no HTTP surface, no new access-control decision); the four staff-action call sites keep their existing `StaffRequiredMixin`/POST-only gating, untouched by this phase |
| V5 Input Validation | no (marginal) | The command takes only `--dry-run` (a boolean flag); no user-supplied identifiers flow into the reconciler beyond `CampaignRun.pk` values already validated by the existing view-layer business-logic guards (`_resolve_site()`'s `approval_status`/`site_needs_review` preconditions, unchanged) |
| V6 Cryptography | no | Not applicable |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A reconciler bug silently overwrites/deletes a hand-created or un-attributed calendar event | Tampering | RECON-05's ownership rule (`CalendarEventMeta.run` unset or absent = never touch) — enforce this as the *first* filter condition in every write path, and cover it with the deliberate same-window-orphan fixture RECON-05's success criterion names |
| A management-command failure mid-batch leaves half the runs reconciled and half not, with no record of which failed | Repudiation (of the operation, not a user) | D-05/D-06's skipped-with-reason summary — every failure is reported by pk and reason, never silently swallowed |
| Business-logic bypass: a staff action fires the reconciler for a run in an invalid state (e.g. `resolve_site` on a run that's already resolved) | Elevation of Privilege (business-logic) | Unchanged — the existing guards in `_resolve_site()`/`approve()`/`_set_run_status()` (conditional `.update()`, staleness checks) already gate *when* `reconcile_run()` is called; the reconciler function itself should be safe to call redundantly (idempotent) as defense-in-depth, matching RECON-01's own idempotency requirement |

## Sources

### Primary (HIGH confidence — read directly from the live repository)

- `solsys_code/campaign_views.py` (full file, 1510 lines) — `_calendar_event_title()`,
  `_project_calendar_event()`, `CampaignRunDecisionView.post()`/`_resolve_site()`/
  `_set_run_status()`, `AttributionQueueView`/`AttributionDecisionView`'s `CalendarEventMeta`
  query patterns
- `solsys_code/calendar_utils.py` (full file) — `insert_or_create_calendar_event()`,
  `_update_or_unchanged()`, `record_time_window()`
- `solsys_code/models.py` (full file) — `CalendarEventMeta`, `CampaignRun`,
  `CampaignRunObservation`, `CalendarEventDismissal`/`ObservationRecordDismissal`
- `solsys_code/telescope_runs.py` (full file) — `sun_event()`, `horizon_dip()`
- `solsys_code/management/commands/backfill_range_calendar_events.py` (full file)
- `solsys_code/management/commands/import_campaign_csv.py` (full file)
- `solsys_code/tests/test_campaign_approval.py` (first 470 lines read directly; remainder
  grepped for class/method inventory) — existing fixture and assertion patterns
- `docs/runbooks/telescope_runs_calendar.rst` (sections read: lines 1-620) — exact sections
  needing update/removal
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` — SPIKE-01..04 verdicts, the
  domain correction, the run-type inventory, the D-11 adopt-vs-gap-fill measurement
- `.planning/phases/29-the-reconciler/29-CONTEXT.md` — locked decisions D-01..D-06
- `.planning/REQUIREMENTS.md` — RECON-01..09, Out of Scope table
- Live dev DB (`python manage.py shell`, read-only queries, 2026-08-04): `Observatory`
  blank-timezone census (5 rows, all satellite); `CampaignRun` source-value census for the
  3I/ATLAS campaign (23 rows, all `legacy`); `CampaignRun` pk=1's real fields

### Secondary (MEDIUM confidence)

- None — no external documentation was consulted; this phase's domain is entirely internal to
  the FOMO codebase.

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies, every component read directly from the
  repository
- Architecture: HIGH for the ported classical/satellite math (existing, tested code);
  MEDIUM for the two genuinely new mechanisms (D-02 adopt/re-key, stage-1 queue-vs-classical
  branch) since no prior implementation exists to verify against — flagged in Assumptions Log
- Pitfalls: HIGH — all three pitfalls are measured directly against the live dev DB or the
  live test file, not inferred

**Research date:** 2026-08-04
**Valid until:** 30 days (stable internal-refactor domain; the one time-sensitive fact — the
real dev DB's `source` census — should be re-checked at planning time if significant time has
elapsed, since staff could correct `source` values between now and plan execution)
