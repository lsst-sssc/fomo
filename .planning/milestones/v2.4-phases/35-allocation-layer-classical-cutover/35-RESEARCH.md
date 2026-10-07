# Phase 35: Allocation Layer & Classical Cutover - Research

**Researched:** 2026-09-12
**Domain:** Django ORM state-machine projection (internal FOMO module work — no new external
packages, no new HTTP surface); reuses `solsys_code.campaign_reconciler` / `calendar_utils` /
`telescope_runs` patterns already shipped in Phases 26-34.
**Confidence:** HIGH — every claim below was checked by reading the actual source file this
session (line numbers and verbatim quotes given), by running a query against the real dev
database (`src/fomo_db.sqlite3`), or is a direct excerpt of the phase's own `35-CONTEXT.md`
(which is itself the product of `/gsd-discuss-phase` and already carries locked user
decisions D-01..D-18). No web research was needed or performed: this phase adds no new
library, framework, or external API — it is 100% new/changed code inside an existing Django
app, following patterns the codebase already established in Phases 29/33/34.

## Summary

Phase 35's `35-CONTEXT.md` is unusually complete — it already resolves every design question
(D-01 through D-18) that would normally be left to research/planning, including the exact
dispatch rule, the exact retirement rule, the exact cutover sequence, and the exact set of
functions to delete/keep/add. This RESEARCH.md therefore does **not** re-litigate those
decisions. Its job is narrower and more load-bearing: to **ground every decision against the
actual code on disk** so the planner can write tasks that reference real function signatures,
real line numbers, real migration numbers, and a real dev-DB baseline, rather than
re-deriving them from prose. Every code excerpt below was read from the file this session.

**Primary recommendation:** Build `solsys_code/allocation_projector.py` as a new peer module
that mirrors `campaign_reconciler.py`'s existing shape almost exactly — reuse
`_may_write`-style ownership checks are not needed here (allocation nights have no pre-existing
event to protect against), but the `_observing_night()` promotion, the `sun_event()`-only-on-mint
discipline, and the `insert_or_create_calendar_event()`/`update_calendar_event_key_and_fields()`
no-churn contract must all be lifted verbatim from `campaign_reconciler.py` and `calendar_utils.py`
rather than re-implemented. `reconcile_run()` gets a small dispatch change (D-09/D-10); the bulk
of the new code is genuinely new (the allocation projector itself, the cutover command, and
`load_telescope_runs`'s rewrite around `write_and_reconcile_campaign_run()`).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Allocation night projection (sunset→sunrise event per night) | API/Backend (Django app logic — `solsys_code/allocation_projector.py`) | Database (`CalendarEvent` rows) | Pure server-side batch/receiver-driven projection; no browser or SSR tier involved (FOMO has no separate frontend server — Django templates render directly) |
| Handoff / retirement (delete `ALLOC:` event when a record links) | API/Backend | Database | Same reasoning; triggered by a Django signal on `CampaignRunObservation`, not a request |
| `load_telescope_runs` classical-line ingest | API/Backend (management command) | Database | File-based batch command, no HTTP request involved |
| Cutover / re-keying of legacy events | API/Backend (one-time management command) | Database | Data migration executed by an operator, not a web request |
| Dispatch (`reconcile_run()` routing to container vs. per-night vs. allocation) | API/Backend | — | Single seam already owned by `campaign_reconciler.py`; extending it, not moving it |
| Astronomical sun-event computation (`sun_event()`) | API/Backend (`solsys_code/telescope_runs.py`) | — | CPU-bound astropy computation, deliberately isolated from the SPICE-heavy `ephem_utils.py` module (CLAUDE.md "Heavy import side effect") |
| Calendar display (event pop-up, month view) | Frontend Server / SSR (Django templates) | — | Out of scope for this phase — no template changes are named in the CONTEXT.md scope; decoration is unchanged (D-12 keeps `event_title()`'s existing form) |

FOMO has no separate SPA/browser tier for this feature: `tom_calendar`'s views render
server-side Django templates. There is therefore no browser-vs-server tier-assignment risk for
this phase — everything lands in the same Django app layer that already owns
`campaign_reconciler.py`/`campaign_utils.py`/`calendar_utils.py`. The only mis-tiering risk
worth flagging to the planner: **do not** let any of this phase's new code import
`solsys_code.views` or `solsys_code.ephem_utils` (D-09 states this explicitly, and it is
already this module family's locked constraint — see `campaign_reconciler.py:27-30`, read
this session:
`"this module must NEVER import the views module or the heavy SPICE-loading ephemeris module"`).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ALLOC-01 | Allocation with resolved site + window projects one sunset→sunrise event per night; queue/class-wide/satellite run keeps its single container | `reconcile_run()`'s existing container/per-night dispatch (`campaign_reconciler.py:598-646`, read this session) is the exact pattern D-09/D-10 extend with a third branch; `_reconcile_classical_nights()` (lines 380-479) is the per-night loop to port, not re-derive |
| ALLOC-02 | Nights keyed by site-local observing night (Chilean + Australian sites) | `_observing_night()` (`campaign_reconciler.py:315-343`, read this session) is the exact noon-anchored helper to promote; dev-DB query this session confirms real Chilean (`268`/`269`/`809`/`X05`, tz `America/Santiago`) and Australian (`E10`, tz `Australia/Sydney`) `Observatory` rows exist for tests |
| ALLOC-03 | Linked record retires its allocation night; unlink restores it; observation event untouched | Spike 003 (`sources/003-allocation-night-retirement/spike.py`, read this session) is a runnable, measured proof of exactly this rule; `CampaignRunObservation` (`models.py:471-532`, read this session) is the existing link model, no new model needed |
| ALLOC-04 | `load_telescope_runs` writes a campaign-less `CampaignRun` (not a `CalendarEvent` directly), collision-safe `source_identifier` | `write_and_reconcile_campaign_run()` (`campaign_utils.py:995-1057`, read this session) is the existing helper to route through; `unique_campaign_run_source_identifier` constraint already exists (`models.py:388-392`, migration `0015_campaignrun_nullable_campaign_and_source_identifier.py`) |
| ALLOC-05 | Cutover has explicit sequencing, no duplicate/orphan left on the calendar | Dev-DB query this session gives the exact before-state to design the cutover against: 241 total events, 56 `RUN:{pk}:{date}` across 28 runs, 0 `ALLOC:`, 45 `CampaignRun` rows (24 `legacy`, 11 `csv_import`, 6 `eso_queue`, 4 `lco_queue`) |
</phase_requirements>

## Standard Stack

No new external package is introduced by this phase. Every library the new code needs is
already an installed, imported dependency of the modules it extends.

### Core (already in use — no new install)
| Library | Version (installed) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django | project-pinned (`tomtoolkit>=2.31.4`'s Django 2.1+ floor) | ORM, migrations, signals | Existing framework; `post_save`/`post_delete` receivers are the established trigger pattern (Phase 34) |
| `tom_calendar` (ships inside `tomtoolkit`) | installed at `.../site-packages/tom_calendar` (read `models.py` this session) | `CalendarEvent` model | `title = CharField(max_length=200)`, `url = URLField(blank=True, default="")` [VERIFIED: tom_calendar/models.py:29,33, read this session] — confirms the 200-char title budget and that `url` accepts non-URL key strings like `ALLOC:5:2026-07-08` (Django's `URLField` only validates via `full_clean()`/forms, never on `.save()`, which is why `RUN:`/blank-url/facility-URL keys already coexist in this column today) |
| `astropy` | project-pinned | `sun_event()`'s solar-altitude crossing search | Already the sole sun-event engine (`telescope_runs.py`); no new astronomy call needed — D-13 says only "call it less often, not differently" |
| `zoneinfo` (stdlib) | Python 3.10+ | `_observing_night()`'s site-timezone anchor | Already imported in `campaign_reconciler.py:53`; `tzdata` is already a project dependency (CLAUDE.md constraints) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| — | — | — | No supporting library additions identified for this phase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A new peer module (`allocation_projector.py`) | Adding a third branch inline inside `campaign_reconciler.py` | Rejected by D-09 itself: keeps `campaign_reconciler.py`'s existing `RUN:` namespace ownership single-purpose, and lets the allocation projector be deleted/rewritten independently without touching the reconciler's already-heavily-tested per-night/container branches |
| Django signals (`post_save`/`post_delete` on `CampaignRunObservation`) | A Celery/cron-only re-project | D-11 requires immediate handoff on link/unlink ("the moment staff confirm or undo"); Phase 31 already settled cron+`flock` as the *unattended*-operation mechanism (Phase 36), not the per-action trigger — this phase's triggers are the same event-driven pattern Phase 34 already proved (spike 001-b, `post_save` beat TOM's own hook) |

**Installation:** none required — this phase adds zero entries to any dependency manifest.
`pyproject.toml`'s dependency set is unchanged by this phase.

**Version verification:** N/A — no package version claims are made; every library referenced
above is already installed and pinned by the existing project (verified by reading the
installed `tom_calendar/models.py` this session rather than trusting training-data knowledge
of tomtoolkit's shipped calendar schema).

## Package Legitimacy Audit

**Not applicable — this phase installs no new external package.** All new code
(`solsys_code/allocation_projector.py`, the cutover management command, the
`load_telescope_runs` rewrite) is pure application code inside the already-installed Django
project, calling only already-imported internal modules
(`solsys_code.calendar_utils`, `solsys_code.campaign_utils`, `solsys_code.campaign_reconciler`,
`solsys_code.telescope_runs`, `solsys_code.models`) and already-installed third-party packages
(`django`, `tom_calendar`, `tom_observations`, `astropy`). The Package Legitimacy Gate protocol
(`gsd_run query package-legitimacy check`) was not run because there is nothing to check — no
`pip install`/`npm install` line will appear in this phase's plan.

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                     ┌───────────────────────────────────────────────────────────┐
                     │                  Triggers (entry points)                  │
                     │                                                           │
  staff action ──────┼──▶ campaign_views.AttributionDecisionView._confirm()/     │
  (link/unlink UI)   │    _undo_confirmation()  ──▶ post_save/post_delete on     │
                     │    CampaignRunObservation  ──▶ [NEW] receivers in         │
                     │    apps.py:ready() (D-11)                                 │
                     │                                                           │
  record save        │    ObservationRecord.save() ──▶ observation_projector's   │
  (queued→placed)     │    existing post_save receiver ──▶ [NEW] "re-project     │
                     │    linked runs" step appended after project_record()     │
                     │    (D-11)                                                │
                     │                                                           │
  batch/backstop     │    reconcile_campaign_runs (sweep) ──▶ reconcile_run()   │
                     │    for every CampaignRun, every invocation               │
                     │                                                           │
  file import        │    load_telescope_runs <file> ──▶ write_and_reconcile_   │
                     │    campaign_run() (existing helper, campaign_utils.py)   │
                     └──────────────────────────┬────────────────────────────────┘
                                                 │
                                                 ▼
                     ┌───────────────────────────────────────────────────────────┐
                     │        reconcile_run(run)  (campaign_reconciler.py)       │
                     │        _skip_reason() stage-0 guard (unchanged)          │
                     │                                                           │
                     │   if telescope_class or satellite site:                  │
                     │        ──▶ _reconcile_container()   [UNCHANGED, RUN:{pk}] │
                     │   elif run.source in {queue sources}:  (D-10, NEW test)  │
                     │        ──▶ _reconcile_container()   [UNCHANGED]          │
                     │   else:                                                  │
                     │        ──▶ [NEW] allocation_projector.project_allocation()│
                     │             (ALLOC:{pk}:{night} namespace)               │
                     └──────────────────────────┬────────────────────────────────┘
                                                 │
                                                 ▼
                     ┌───────────────────────────────────────────────────────────┐
                     │      allocation_projector.py  (NEW peer module)          │
                     │                                                           │
                     │  1. linked_nights = site-local night of every             │
                     │     CampaignRunObservation whose record has a placed/     │
                     │     observed block (record_time_window + _observing_night)│
                     │  2. for each night in [window_start..window_end]:         │
                     │       if night in linked_nights: DELETE ALLOC: event      │
                     │       elif event exists at ALLOC:{pk}:{night}:            │
                     │           refresh title/description/target_list only     │
                     │           (never start_time/end_time, D-13)              │
                     │       else: sun_event() ONLY here, on create/re-mint      │
                     │           (D-13, folds the "skip sun_event on existing"   │
                     │           todo directly into new code)                   │
                     │  3. attribute each linked record's OWN event to this run  │
                     │     via CalendarEventMeta.run (adopt_event_into_run(),    │
                     │     never a CalendarEvent field write) — D-08            │
                     └──────────────────────────┬────────────────────────────────┘
                                                 │
                                                 ▼
                     ┌───────────────────────────────────────────────────────────┐
                     │  tom_calendar.CalendarEvent rows (shared calendar table)  │
                     │   - ALLOC:{pk}:{night}     (this phase, intent-only)      │
                     │   - RUN:{pk} / RUN:{pk}:{night}  (existing, container/    │
                     │     legacy per-night, cutover re-keys the latter away)    │
                     │   - facility observation URL (Phase 34, base layer,       │
                     │     e.g. https://observe.lco.global/requests/{id})        │
                     └───────────────────────────────────────────────────────────┘
```

A reader tracing the primary use case (a classical schedule line becomes a calendar night,
then a real observation retires it) follows: `load_telescope_runs` → `write_and_reconcile_
campaign_run()` → `reconcile_run()` → (new) `allocation_projector.project_allocation()` →
`ALLOC:` `CalendarEvent` row created → later, an `ObservationRecord` for that same
target/window gets linked via `CampaignRunObservation` (staff confirm) → the new
`post_save`/`post_delete` receivers (D-11) re-invoke `project_allocation()` → the night's
`ALLOC:` event is deleted and the record's own (Phase-34-projected) event gets
`CalendarEventMeta.run` set to this run.

### Recommended Project Structure

No new directories. One new module at the existing flat layout `campaign_reconciler.py`/
`campaign_utils.py`/`calendar_utils.py`/`telescope_runs.py` already use:

```
solsys_code/
├── allocation_projector.py        # NEW — ALLOC: namespace owner (D-09)
├── campaign_reconciler.py         # MODIFIED — dispatch only (D-09/D-10); _reconcile_classical_nights()
│                                  #   and run_night_url() DELETED; _observing_night() promoted out
├── campaign_utils.py              # UNCHANGED (already has write_and_reconcile_campaign_run(),
│                                  #   adopt_event_into_run(), unlink_event_from_run())
├── calendar_utils.py              # UNCHANGED (insert_or_create_calendar_event(), etc. reused as-is)
├── telescope_runs.py              # MODIFIED — parse_run_line()/ParsedRun gain the proposal token (D-01)
├── models.py                      # MODIFIED — two new nullable CampaignRun fields (D-04); new migration
├── apps.py                        # MODIFIED — two new CampaignRunObservation receivers (D-11)
├── observation_projector.py       # MODIFIED — post_save receiver gains "re-project linked runs" step (D-11)
├── management/commands/
│   ├── load_telescope_runs.py     # REWRITTEN — routes through write_and_reconcile_campaign_run()
│   ├── reconcile_campaign_runs.py # UNCHANGED (already the sweep entry point; first post-cutover
│   │                              #   run performs the RUN:→ALLOC: re-key takeover, D-16)
│   └── cutover_classical_allocations.py  # NEW — one-time D-15/D-17 command
└── tests/
    ├── test_allocation_projector.py       # NEW
    ├── test_load_telescope_runs.py        # EXTENDED (24 existing tests, per canonical_refs)
    └── test_campaign_reconciler.py        # EXTENDED (dispatch-only changes; per-night tests move
                                            #   to test_allocation_projector.py where the behavior lives)
```

### Pattern 1: Site-local observing-night anchor (promote, don't reimplement)

**What:** `_observing_night(start_time, site_zone)` converts an aware `start_time` into the
site's local timezone, subtracts 12 hours, then takes `.date()` — this is the noon-anchored
rule that agrees with `sun_event()`'s own anchor (`_local_noon_utc()`), and it is what fixed a
real, measured defect (CR-02) where a naive `.date()` mis-assigned a post-midnight start to
the wrong night.

**When to use:** Every place this phase needs "which observing night does this record/event
belong to" — both the allocation projector's `linked_nights` computation (ALLOC-03) and any
new test asserting ALLOC-02's site-local-night behavior.

**Example (verified — read from source this session, `campaign_reconciler.py:315-343`):**
```python
# Source: solsys_code/campaign_reconciler.py:315-343 (read this session)
def _observing_night(start_time: datetime, site_zone: ZoneInfo):
    """The site-local observing night a ``start_time`` belongs to, anchored at local noon.
    ...
    """
    local = start_time.astimezone(site_zone)
    return (local - timedelta(hours=12)).date()
```
CONTEXT.md D-05 explicitly directs promoting this "to a shared public helper next to
`sun_event()`" for Phase 35's own use — the docstring's own forward-pointer (line 332-333,
read this session) already names this exact promotion: `"Forward-pointer: Phase 34/35 should
promote this to a shared public helper next to sun_event() when the observation projector
needs the same event-to-night mapping."`

### Pattern 2: `sun_event()` computed only on mint (D-13 / folded todo)

**What:** `sun_event(site, night, kind='sun')` runs an astropy solar-altitude crossing search
that costs measurably (~0.3-1s per call, per the spike-findings skill, `[CITED:
.claude/skills/spike-findings-fomo_devel/references/allocation-handoff.md]`). The existing
per-night reconciler branch (being retired) calls it unconditionally for every night on every
sweep, including nights whose event already exists and is unchanged — this is the exact defect
named in the still-open todo
`.planning/todos/pending/2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`
(read this session).

**When to use:** The new `allocation_projector.py` must call `sun_event()` only inside the
branch that creates a brand-new event or re-mints one whose sub-night fields changed (D-13) —
never inside the branch that only refreshes `title`/`description`/`target_list` on an
already-existing, still-correct night.

**Example (existing per-night branch's current — soon-to-be-retired — shape, showing the
anti-pattern to avoid porting verbatim; read this session, `campaign_reconciler.py:433-477`):**
```python
# Source: solsys_code/campaign_reconciler.py:433-477 (read this session) — CURRENT
# _reconcile_classical_nights() calls sun_event() unconditionally, BEFORE checking whether
# `existing` is None. This is the defect the folded todo names; do NOT port this ordering.
for i in range(n_nights):
    ...
    active_urls.add(url)
    sunset, sunrise = sun_event(run.site, night, kind='sun')   # <-- runs even when existing is not None
    ...
    if existing is None:
        ...
        'start_time': sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
        'end_time': sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
```
The new module must move the `sun_event()` call **inside** the `if existing is None` (or
"sub-night fields changed") branch instead.

### Pattern 3: No-churn create/update/delete via existing `calendar_utils` helpers

**What:** `insert_or_create_calendar_event()`, `update_calendar_event_key_and_fields()`, and
`preview_calendar_event_action()` (all read this session, `calendar_utils.py:585-699`) already
implement the exact create/no-churn-update/dry-run-preview contract every calendar writer in
this codebase uses. The cutover's in-place re-key (D-16/D-17) is exactly
`update_calendar_event_key_and_fields()`'s documented purpose (its own docstring, read this
session, says it exists precisely for "the Phase 29 reconciler's adopt-and-rekey step").

**When to use:** `allocation_projector.py`'s per-night write, and the cutover command's
`RUN:{pk}:{date}` → `ALLOC:{pk}:{night}` re-key, both call these helpers directly — neither
should re-implement `get_or_create`/diff-and-save logic.

**Example (verified — the existing dry-run counterpart, `calendar_utils.py:678-698`):**
```python
# Source: solsys_code/calendar_utils.py:678-698 (read this session)
def preview_calendar_event_action(event: CalendarEvent | None, fields: dict[str, Any]) -> str:
    if event is None:
        return 'created'
    changed = [f for f, v in fields.items() if getattr(event, f) != v]
    return 'updated' if changed else 'unchanged'
```

### Pattern 4: Signal-receiver wiring (`apps.py:ready()`, weak=False, dispatch_uid, never-raise)

**What:** Every receiver this codebase adds follows one shape: connected in `AppConfig.ready()`
with `weak=False` and a unique `dispatch_uid`, guards on `raw=True` (fixture loads), and wraps
its body in `try/except Exception` that only logs — never lets a projector fault break the
caller's save/delete. Verified this session in `apps.py:1-47` and
`observation_projector.py:570-608`.

**When to use:** The two new `CampaignRunObservation` `post_save`/`post_delete` receivers
(D-11) must follow this identical shape.

**Example (verified — the exact wiring to copy, `apps.py:30-47`):**
```python
# Source: solsys_code/apps.py:30-47 (read this session)
post_save.connect(
    receiver_on_record_save,
    sender=ObservationRecord,
    weak=False,
    dispatch_uid='solsys_code.observation_projector.post_save',
)
```
```python
# Source: solsys_code/observation_projector.py:570-600 (read this session) — the never-raise shape
def receiver_on_record_save(sender, instance, created, raw, **kwargs):
    if raw:
        return
    if instance.facility not in PROJECTED_FACILITIES:
        return
    try:
        action, stage = project_record(instance)
    except Exception as exc:  # noqa: BLE001 -- TRIG-02: never abort the caller's save
        logger.warning('receiver_on_record_save failed for observation_id=%r: %s',
                        instance.observation_id, type(exc).__name__)
        return
```

### Anti-Patterns to Avoid

- **Writing campaign text into a base (observation) event's `title`/`description`:** the next
  base re-projection (Phase 34's projector) rebuilds those fields from the record's own state
  and silently erases anything written there — attribution must go through
  `CalendarEventMeta.run` only (proven by spike 003's measured "base event fields unchanged"
  assertion, `[VERIFIED: sources/003-allocation-night-retirement/spike.py:180]`, read this
  session — the spike's own summary dict computes `base_event_fields_unchanged_by_campaign_side`).
- **Keying a night by its naive UTC date** (`start_time.date()`) instead of the site-local,
  noon-anchored observing night — this is exactly the CR-02 defect `_observing_night()` fixed,
  and the spike's own `night_of_record()` helper (`spike.py:59-62`, read this session) is
  explicitly commented as "the build must use the site-local date" — do not port the spike's
  simplification.
- **Re-acquiring a direct `CalendarEvent` write path from an adapter/command** — D-02 already
  states `load_telescope_runs` must route through `write_and_reconcile_campaign_run()`
  (`campaign_utils.py:995`), never call `insert_or_create_calendar_event()` itself for a
  classical line after this phase.
- **Importing `solsys_code.views` or `solsys_code.ephem_utils` from the new module** — triggers
  the ~1.6 GB SPICE kernel download at import time (CLAUDE.md "Heavy import side effect");
  `campaign_reconciler.py`'s own module docstring (read this session, lines 27-30) already
  states this constraint for its sibling modules and D-09 repeats it for the new one.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update-or-leave-unchanged a `CalendarEvent` | A new get_or_create + diff loop | `calendar_utils.insert_or_create_calendar_event()` (verified, `calendar_utils.py:585`) | Already handles the `start_time_tolerance` proximity-match case `load_telescope_runs` needs for its drifting `sun_event()` timestamps; a second implementation would drift out of sync with the no-churn contract every other writer relies on |
| Re-key an already-identified event's `url` in place | Manual `.url = new_url; .save()` | `calendar_utils.update_calendar_event_key_and_fields()` (verified, `calendar_utils.py:648`) | Its own docstring states it exists precisely because `get_or_create()`'s lookup key can never also be a field to update — this is the cutover's exact re-key operation (D-16/D-17) |
| Decide which `CampaignRun`s a run's dispatch belongs to (container vs. per-night vs. allocation) | A second copy of the dispatch rule inside the new module | Extend `reconcile_run()`'s existing `if/elif/else` (verified, `campaign_reconciler.py:598-628`) | `reconcile_run()` is already the single, tested entry point every staff-action view and the sweep call; a second dispatcher would let the two disagree |
| Compute sunset/sunrise for a night | A second astropy call site | `telescope_runs.sun_event(site, date, kind)` (verified, `telescope_runs.py:251`) | Already the dip-corrected, site-timezone-aware, bisection-refined implementation; a second implementation risks the exact ~1.44° dip-precision requirement this project already validated (Stage 1 constraint, CLAUDE.md "Precision" section) |
| Clear/set a `CalendarEventMeta.run` attribution | Direct `meta.run = run; meta.save()` | `campaign_utils.adopt_event_into_run()` / `unlink_event_from_run()` (verified, `campaign_utils.py:949`, `875`) | These are the only writers that respect the "human-confirmed attribution outranks an automated writer" rule (33-10's guard) — a direct write would silently clobber a staff confirmation |

**Key insight:** Every "don't hand-roll" item above is not a third-party library recommendation
— it is a "don't re-implement this codebase's own already-hardened helper" warning, because
every one of these functions embeds a previously-fixed defect (idempotency drift tolerance,
race-safe `get_or_create`, human-outranks-machine attribution). Re-deriving any of them for
Phase 35 risks reintroducing a bug this codebase's own git history already paid to fix.

## Runtime State Inventory

> Included because ALLOC-05's cutover (D-15/D-16/D-17) rewrites in place what is today a
> production-shaped dev database's calendar state — this is a data migration, not a greenfield
> feature, even though it ships as application code plus a one-time command rather than a
> classic rename/refactor.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data (calendar events) | Dev DB (`src/fomo_db.sqlite3`), queried this session: **241** total `CalendarEvent` rows; **56** keyed `RUN:{pk}:{date}` across **28** `CampaignRun`s; **0** keyed `ALLOC:`; **10** blank-`url` classical rows (per CONTEXT.md's code_context, not independently re-verified beyond the 241/56/0 counts this session confirmed match) | Cutover command (D-17) re-parses each blank-`url` event's `Source line:` and creates the owning `CampaignRun`; `reconcile_campaign_runs`'s first post-deploy sweep (D-16) re-keys every remaining `RUN:{pk}:{date}` night to `ALLOC:{pk}:{night}` in place, or deletes it if the run is now dispatched to the container branch (the 8 single-night `lco_queue`/`eso_queue` runs named in CONTEXT.md's code_context) |
| Live service config | None found — classical schedules are ingested from a local text file (`load_telescope_runs <filepath>`), not a live external service; no n8n/Datadog/Tailscale-style out-of-git config applies to this codebase | None |
| OS-registered state | None found — no Task Scheduler/pm2/launchd/systemd registration touches `CampaignRun`/`CalendarEvent` naming; Phase 36 (unattended cron+`flock`) is explicitly out of this phase's scope | None |
| Secrets/env vars | None found — this phase adds no new secret, API key, or env-var-keyed setting. `FOMO_DATABASE_PATH` (used by demo notebooks per Phase 33/34 precedent) is pre-existing and unaffected | None |
| Build artifacts / installed packages | None found — no package rename, no `pyproject.toml` change; one new Django migration file is expected (next number after `0017_calendareventmeta_observation_links.py`, i.e. `0018_...py`, verified by listing `solsys_code/migrations/` this session) for D-04's two new `CampaignRun` fields | Standard `python manage.py makemigrations && python manage.py migrate` — no `RunPython` data step per D-15 ("No `RunPython` data migration") |

**The canonical question restated for this phase:** after the code deploys, what calendar rows
still carry the *old* key form? Answer, grounded in the dev-DB query this session: 56
`RUN:{pk}:{date}` rows and (per CONTEXT.md, not independently re-verified this session beyond
the aggregate counts) roughly 9-10 blank-`url` classical rows. Both categories are the cutover
command's and the first sweep's explicit job (D-15 steps 3-4); nothing else in the schema
carries the pre-cutover key form.

## Common Pitfalls

### Pitfall 1: UTC-date keying instead of site-local observing night
**What goes wrong:** A night is keyed/matched by `record_time_window(record)[0].date()`
(naive UTC date) instead of the site-local, noon-anchored date, so a record whose UTC start
falls after local midnight (common for Chilean sites, UTC-3/-4) gets attributed to the wrong
calendar night, and the wrong `ALLOC:` event is retired (or none is).
**Why it happens:** It is the simpler-looking derivation, and it is exactly what spike 003's
own `night_of_record()` helper did — the spike's own comment flags this as a simplification
not to carry into the build (`sources/003-allocation-night-retirement/spike.py:59-62`, read
this session: `"the build must use the site-local date, cf. the 32-01 must-have about
Chilean/Australian sites"`).
**How to avoid:** Use `_observing_night(start_time, ZoneInfo(run.site.timezone))` (promoted per
D-05) everywhere a night needs deriving from a timestamp.
**Warning signs:** A test asserting retirement behavior only against a UTC or US/European site
would never catch this; ALLOC-02 explicitly requires testing a Chilean *and* an Australian
site, and a record whose UTC start date differs from its observing night (D-05's own test
guidance).

### Pitfall 2: Recomputing `sun_event()` on every idempotent sweep
**What goes wrong:** A multi-week allocation run pays an astropy solar-altitude scan
(measured ~0.3-1s per call) for every night, on every sweep, even when nothing about that
night changed — this is the still-open todo `2026-09-01-skip-sun-event-computation-for-
already-existing-reconciler-n.md` (read this session), found during a 2026-09-01 branch
review of the very code this phase retires.
**Why it happens:** The retired `_reconcile_classical_nights()` calls `sun_event()`
unconditionally, before checking whether the night's event already exists (verified,
`campaign_reconciler.py:449`, inside the `for i in range(n_nights)` loop, ahead of the
`if existing is None` branch at line 456).
**How to avoid:** D-13 requires the new module compute `sun_event()` only inside the
create-or-re-mint branch. This closes the folded todo as a byproduct of the new design rather
than as a separate patch.
**Warning signs:** A regression test asserting "an idempotent re-reconcile of an existing
multi-night run makes no `sun_event()` call" (the todo's own suggested test, patchable via
`unittest.mock.patch('solsys_code.allocation_projector.sun_event')` and asserting
`not_called()` after the first successful reconcile) should be part of this phase's test plan.

### Pitfall 3: Writing campaign attribution into a `CalendarEvent` field instead of `CalendarEventMeta.run`
**What goes wrong:** Any write to an observation-backed event's `title`/`description`/
`start_time`/`end_time` from the allocation/campaign side is silently erased the next time
the Phase 34 observation projector re-projects that record (it rebuilds those fields from the
record's own current state, unconditionally, because it does not know a campaign has
"decorated" them).
**Why it happens:** It is the natural-looking way to "show" a campaign attribution on the
calendar, and is exactly what the *previous* reconciler generation did before Phase 33's
ANNOT-01/D-17 inversion (module docstring, `campaign_reconciler.py:32-37`, read this session:
`"a set CalendarEventMeta.run means the event is ATTRIBUTED to that run, never that the run
OWNS it"`).
**How to avoid:** Every attribution write in this phase must go through
`campaign_utils.adopt_event_into_run()` (link) or `unlink_event_from_run()` (clear) — never a
direct `CalendarEvent.save()` from the allocation side.
**Warning signs:** A code-review grep for `.save(` inside any function that also touches an
observation-record-derived event is the exact check `campaign_reconciler.py`'s own tests use
(`test_reconciler_never_touches_the_record_derived_event`, `test_campaign_reconciler.py:1162`,
read this session) — an equivalent test for the allocation projector should assert the same.

### Pitfall 4: A second, drifting dispatch rule for "is this run per-night or a container"
**What goes wrong:** If the allocation projector (or the cutover command) re-derives its own
copy of "does this run get one event per night or one whole-window container" instead of
reading `reconcile_run()`'s existing branch decision, a future edit to the container/queue
rule (e.g. a new `Source` value) updates one copy and not the other, and a run silently gets
fanned out per-night when it should be a container (or vice versa) — exactly the failure mode
the roadmap's "Phase 26 verdict" (Success Criterion 1) exists to prevent.
**Why it happens:** The new allocation projector module is a separate file from
`campaign_reconciler.py`, so it is tempting to give it its own `if run.telescope_class or
run.source in {...}` guard rather than importing the existing one.
**How to avoid:** D-10 already specifies the exact rule (`run.source in {LCO_QUEUE, SOAR_QUEUE,
GEMINI_QUEUE, ESO_QUEUE}` dispatches to `_reconcile_container()` regardless of site, "no
inference from telescope names or sites"); this rule belongs in `reconcile_run()`'s own
dispatch (already the single seam, verified `campaign_reconciler.py:613-628`), and the
allocation projector should never be called for a run that dispatch didn't route to it.
**Warning signs:** A test that constructs a `CampaignRun` with `source=LCO_QUEUE` and a
resolved ground site and asserts it gets a container, not a fanned-out `ALLOC:` set (already
partly covered by the existing `TestQueueSourceDoesNotChangeShape` test class,
`test_campaign_reconciler.py:110-196`, read this session — extend this class's assumptions
rather than writing a parallel one).

## Code Examples

### The exact retirement rule (verified end-to-end proof)
```python
# Source: .claude/skills/spike-findings-fomo_devel/sources/003-allocation-night-retirement/spike.py
# (read this session, lines 75-112) — measured: link -> 1 retired, re-project -> unchanged,
# unlink -> 1 created; base event `modified` timestamp untouched throughout.
def project_allocation(run: CampaignRun) -> dict[str, int]:
    counters = {'created': 0, 'updated': 0, 'unchanged': 0, 'retired': 0}
    taken = linked_nights(run)          # site-local nights of every linked, placed/observed record
    for night in nights_of(run):
        url = alloc_url(run, night)     # f'ALLOC:{run.pk}:{night.isoformat()}'
        if night in taken:
            deleted, _ = CalendarEvent.objects.filter(url=url).delete()
            counters['retired'] += int(deleted > 0)
            continue
        start, end = sunset_sunrise(run.site, night)   # sun_event() call — only reached here
        fields = {...}
        _event, action = insert_or_create_calendar_event({'url': url}, fields)
        counters[action] += 1
    # attribution (annotate, never adopt): link the base events of linked records to the run
    for link in run.observation_links.select_related('observation_record'):
        rec = link.observation_record
        base = CalendarEvent.objects.filter(url=projector.event_url(rec, projector.facility_for(rec))).first()
        if base is not None:
            meta, _ = CalendarEventMeta.objects.get_or_create(event=base)
            if meta.run_id != run.pk:
                meta.run = run
                meta.save(update_fields=['run'])
    return counters
```
This is a spike, not the build — it uses `night_of_record()`'s naive-UTC-date simplification
(Pitfall 1 above) and its own `insert_or_create_calendar_event()` field set does not match
D-12's final title/description form. The planner should treat the *shape* (retire-if-linked,
mint-if-not, attribute-via-meta-only) as proven and the *details* (night derivation, field
content) as superseded by D-05/D-08/D-12.

### The existing dispatch seam to extend, not duplicate
```python
# Source: solsys_code/campaign_reconciler.py:598-646 (read this session) — CURRENT reconcile_run()
def reconcile_run(run: CampaignRun, *, dry_run: bool = False) -> ReconcileResult:
    reason = _skip_reason(run)
    if reason is not None:
        return ReconcileResult(skipped_reason=reason)

    if run.telescope_class:
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    elif run.site is not None and run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    else:
        result, active_urls = _reconcile_classical_nights(run, dry_run=dry_run)   # <- D-09 replaces
        # this branch's callee with allocation_projector's equivalent; D-10 adds a queue-source
        # elif ABOVE the final else, still routing to _reconcile_container()
    ...
```

### Idempotent create-or-update-or-unchanged, already implemented
```python
# Source: solsys_code/calendar_utils.py:585-645 (read this session)
def insert_or_create_calendar_event(lookup, fields, *, start_time_tolerance=None):
    if start_time_tolerance is not None and 'start_time' in lookup:
        ...  # proximity-match branch (used by load_telescope_runs today; not needed for
             # allocation nights, whose lookup is the exact string key ALLOC:{pk}:{night})
    event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return event, 'created'
    return _update_or_unchanged(event, fields)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `load_telescope_runs` writes a blank-`url` `CalendarEvent` directly via `insert_or_create_calendar_event()` (verified, `load_telescope_runs.py:208-232`, read this session) | Writes a campaign-less `CampaignRun` via `write_and_reconcile_campaign_run()`, which reconciles through the new allocation projector | This phase (ALLOC-04) | The command no longer computes `sun_event()`/dark-window itself at all — that becomes the allocation projector's job, called once per line via `reconcile_run()`; `_resolve_window_time()`/`_iter_run_nights()` move from the command into what becomes the run's stored sub-night fields (D-04) plus the projector's per-night loop |
| `campaign_reconciler._reconcile_classical_nights()` writes classical/per-night events under `RUN:{pk}:{date}`, with or without a campaign, and narrows via the D-01/ANNOT-01 "skip if already attributed" rule | Deleted outright; per-night, non-container, non-satellite runs dispatch to the new `allocation_projector.py`'s `ALLOC:{pk}:{night}` namespace, with an explicit link-based retirement (delete) rather than a skip-and-leave-alone | This phase (D-09) | `run_night_url()` is also deleted (verified, `campaign_reconciler.py:118-129`); any test currently asserting on `RUN:{pk}:{date}` behavior for a per-night run must move to `test_allocation_projector.py` and assert `ALLOC:` instead |
| Naive UTC-date night derivation (`record_time_window(record)[0].date()`) | Site-local, noon-anchored `_observing_night()` (already adopted by the campaign reconciler in Phase 33, CR-02 fix) | Phase 33 (already shipped); this phase reuses it, does not re-derive it | Any new Phase 35 code that re-derives its own night-from-timestamp logic risks reintroducing the exact CR-02 defect that was already fixed once |

**Deprecated/outdated:**
- `_reconcile_classical_nights()` and `run_night_url()` (`campaign_reconciler.py`): deleted by
  D-09, superseded by the new `allocation_projector.py`.
- `load_telescope_runs`'s direct `insert_or_create_calendar_event()` call
  (`load_telescope_runs.py:208`): removed by ALLOC-04's rewrite.
- `_CLASSICAL_STATUS_PREFIX` (`load_telescope_runs.py:29`): superseded by
  `RUN_STATUS_CALENDAR_PREFIX` (`campaign_reconciler.py:76-79`) once every classical line maps
  its parsed status onto `CampaignRun.run_status` (D-03) rather than the command's own
  standalone dict.

## Assumptions Log

CONTEXT.md's `## Claude's Discretion` section already enumerates every open naming/shape
decision left to the planner/executor (proposal-token syntax, D-04 field names, `--dry-run`
wording, counter names, cutover command name/module retention). None of those are research
claims that need independent verification — they are deliberately left open by the user's own
discussion, not assumptions this research is asserting as fact. The one genuinely
research-introduced assumption below should be flagged for a planner confirmation checkpoint:

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 10 blank-`url` classical events and 24 `legacy`-source `CampaignRun` rows named in CONTEXT.md's `code_context` still hold those exact counts and shapes at execution time (this session independently re-verified the 241/56/0 aggregate counts and the 45-run/4-source breakdown via a direct dev-DB query, but did NOT re-verify the specific claim that exactly 9 of the 10 blank-`url` rows have a parseable `Source line:` and 1 (pk 334) does not) | Runtime State Inventory | Low — the cutover command (D-18) already handles an unparseable event by reporting it and exiting non-zero rather than silently mis-migrating it, so a stale count only changes the cutover's reported summary, not its correctness |

**If this table is sparse:** the bulk of this phase's design was locked by the user during
`/gsd-discuss-phase` (CONTEXT.md D-01..D-18) before this research ran; research's job was
verification against the actual codebase, which succeeded for every claim except the one
above (a live-DB row-content detail not independently re-queried this session).

## Open Questions (RESOLVED)

Both questions below were settled during planning; each carries its resolution and the plan
that decides it. Nothing in this section is still open.

1. **Exact field names/type for D-04's two new `CampaignRun` sub-night fields** — **RESOLVED
   (plan 35-03, Task 1):** two nullable `TimeField`s named `night_start_utc` and
   `night_end_utc`, added by the additive migration
   `solsys_code/migrations/0018_campaignrun_night_window_fields.py` with no data step. Null in
   either field means "use the computed sun event for this night". This is the recommendation
   below, adopted unchanged.
   - What we know: CONTEXT.md explicitly leaves this to the planner ("Claude's Discretion"),
     suggesting `night_start_utc`/`night_end_utc` as a `TimeField` pair, null = computed
     sunset/sunrise.
   - What's unclear: whether a `TimeField` (naive time-of-day) or a small-int
     minutes-after-midnight integer is the right storage, especially given
     `_resolve_window_time()`'s existing `HHMM` int-parsing (`load_telescope_runs.py:32-53`,
     read this session) already works in `(hour, minute)` ints, not `time` objects.
   - Recommendation: `TimeField` (nullable) matches Django convention better for an
     admin-editable value and round-trips cleanly through `_resolve_window_time()`'s existing
     `hh, mm = int(window[:2]), int(window[2:])` parsing (construct a `time(hh, mm)`, store
     that) — the planner's discretion is un-blocked by this research either way.

2. **Whether the cutover command needs a `--file` fallback for the 1 unexplainable event (pk 334, the dev DB's own `tmp` junk row)** — **RESOLVED (plan 35-06 Task 2 §5 and §6, plan 35-06
   Task 3 §3, plan 35-07 Task 3 §4):** the non-zero exit is operator-facing only, exactly as
   recommended below. No plan gates a CI step, a pipeline step or another command's execution
   on this command's exit code; no `--file` fallback is added. The two places the command is
   actually driven both treat the non-zero exit as the expected D-18 outcome on a database that
   still holds an unexplainable row: 35-06 Task 3 runs it from a shell against a scratch copy
   and records the reported list as evidence rather than as a task failure, and 35-07 Task 3
   drives it from a notebook cell through `call_command()` inside a `try` / `except
   CommandError` that prints the message as cell output, so the notebook run completes and the
   four end-state assertions that follow it still execute.
   - What we know: D-18 already specifies this event is left untouched and reported, exit
     code non-zero.
   - What's unclear: whether "exit non-zero" should block a CI/deploy pipeline step, or is
     purely an operator-facing signal (this project has no CI gate that runs management
     commands, per CLAUDE.md's Commands section — only `pre-commit` and `python manage.py
     test`/`migrate` are named).
   - Recommendation: treat non-zero exit as operator-facing only; the planner's task should
     not gate any other step on this command's exit code, since no CI step invokes it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Django ORM / migrations | D-04's new `CampaignRun` fields | ✓ | project-pinned | — |
| SQLite dev DB (`src/fomo_db.sqlite3`) | Cutover command's real-data proof (D-15's runbook diff) | ✓ (queried this session) | — | — |
| `astropy` (`sun_event()`) | Allocation night sunset/sunrise computation | ✓ (already imported by `telescope_runs.py`) | project-pinned | — |
| `tzdata` (IANA zone data for `zoneinfo`) | `_observing_night()`'s site-timezone conversion | ✓ (CLAUDE.md constraint; confirmed real `Observatory.timezone` values for `America/Santiago`/`Australia/Sydney` exist in dev DB, queried this session) | stdlib + `tzdata` package | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — every dependency this phase needs is already
installed and already exercised by the code it extends.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django's built-in `django.test.TestCase` test runner (`python manage.py test`) — confirmed the project's only functioning suite (CLAUDE.md "Testing" section, and `.planning/config.json`'s `test_command`) |
| Config file | none — Django test discovery via `manage.py test <labels>`; `pyproject.toml`'s `[tool.pytest.ini_options]` is explicitly legacy/unused for this app (CLAUDE.md) |
| Quick run command | `python manage.py test solsys_code.tests.test_allocation_projector` (new file) or `python manage.py test solsys_code.tests.test_campaign_reconciler` for dispatch-only changes |
| Full suite command | The project's own `test_command` from `.planning/config.json`, verified this session: `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py \| grep -v "tests/test_views\.py$" \| sed "s|/|.|g; s|\.py\$||" \| tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ALLOC-01 | Per-night events for a resolved-site, resolved-window allocation; queue/class-wide/satellite runs stay containers | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ new file (test class pattern to copy: `TestClassicalStage1`, `test_campaign_reconciler.py:1028-1147`, read this session) |
| ALLOC-02 | Site-local night keying, Chilean + Australian sites | unit | `python manage.py test solsys_code.tests.test_allocation_projector.TestObservingNightBoundary` (name TBD) | ❌ new — pattern to copy: `TestObservingNightBoundary`, `test_campaign_reconciler.py:499-675`, read this session, already exercises a real `Australia/Sydney` fixture (`F65`); a Chilean fixture (`America/Santiago`, e.g. obscode `268`/`269`/`809`/`X05`, all confirmed present in the real dev DB this session) should be added as a parallel fixture in the new test module |
| ALLOC-03 | Link retires the night; unlink restores it; observation event untouched | unit + integration | `python manage.py test solsys_code.tests.test_allocation_projector` (retire/restore) and `python manage.py test solsys_code.tests.test_observation_projector_signals` (cross-check the observation event's own fields are byte-unchanged) | ❌ new (retire/restore) / ✅ existing signals test file to extend |
| ALLOC-04 | `load_telescope_runs` writes a `CampaignRun`, not a direct event; idempotent re-import; collision-safe `source_identifier` | integration | `python manage.py test solsys_code.tests.test_load_telescope_runs` | ✅ existing (475 lines, 24 tests per canonical_refs) — extend in place; existing tests like `test_idempotent_rerun_no_duplicates` (line 218) and `test_reingest_with_drifted_sun_event_does_not_duplicate` (line 252) must be revised to assert `CampaignRun` create-or-update behavior instead of direct `CalendarEvent` counts |
| ALLOC-05 | Cutover leaves no duplicate/orphan; before/after counts documented | manual + notebook-verified | `python manage.py cutover_classical_allocations --dry-run` then real run, diffed in `reconcile_campaign_runs_demo.ipynb` (paired-docs obligation) against the real dev DB baseline captured this session (241 events / 56 `RUN:{pk}:{date}` / 0 `ALLOC:`) | ❌ new command; ✅ existing notebook to extend |

### Sampling Rate
- **Per task commit:** the relevant single test file/class (e.g.
  `python manage.py test solsys_code.tests.test_allocation_projector`)
- **Per wave merge:** the full label-list command from `.planning/config.json`'s
  `test_command` (excludes `test_views.TestEphemeris`, which segfaults in native ASSIST per
  project memory)
- **Phase gate:** full suite green, plus `pre-commit run ruff --all-files` and
  `pre-commit run ruff-format --all-files` (both named explicitly in CLAUDE.md's Commands
  section and this phase's own CONTEXT.md constraints)

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_allocation_projector.py` — new file, covers ALLOC-01/02/03; no
  shared fixture currently exists for a campaign-less (`campaign=None`) `CampaignRun` with a
  resolved Chilean site — add one alongside the existing `CampaignReconcilerTestBase` pattern
  (`test_campaign_reconciler.py:35-71`, read this session) rather than duplicating its
  Australian-only site fixture.
- [ ] A Chilean `Observatory` test fixture (`timezone='America/Santiago'`) — the existing
  reconciler test base only fixtures an Australian site (`F65`, `Australia/Sydney`); ALLOC-02
  explicitly requires both hemispheres tested, so the new test module needs its own Chilean
  fixture (real dev-DB obscodes `268`/`269`/`809`/`X05` confirmed to exist with this timezone
  this session, but a **test** fixture should still create its own `Observatory` row rather
  than depend on dev-DB content, per every existing test class's pattern).
- [ ] Framework install: none — `django.test.TestCase` and `NonSiderealTargetFactory` are
  already available project-wide.

## Security Domain

> `security_enforcement` is `true` in `.planning/config.json` (ASVS level 1, block on `high`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase adds no new authentication surface — the cutover command and `load_telescope_runs` are both operator-run management commands, not HTTP endpoints; the existing `StaffRequiredMixin` on `AttributionDecisionView` (`campaign_views.py:1157`, already covers the confirm/undo actions that trigger the new receivers) is unchanged |
| V3 Session Management | No | No new session-touching code |
| V4 Access Control | No (unchanged) | The link/unlink triggers for D-11 fire from `AttributionDecisionView._confirm()`/`_undo_confirmation()`, which already sit behind `StaffRequiredMixin` (verified this session, `campaign_views.py:1157` class declaration) — this phase adds a signal receiver *downstream* of an already-access-controlled action, not a new entry point |
| V5 Input Validation | Yes | The new optional proposal token in `parse_run_line()` (D-01) is parsed from a staff-uploaded schedule file, the same trust boundary `parse_run_line()` already validates against (`ValueError` on anything unrecognized, per the existing `KNOWN_STATUSES`/`_PARTIAL_NIGHTS` pattern, `telescope_runs.py:361-393`/`477-490`, read this session) — the new token must follow the same "raise `ValueError` on anything ambiguous" discipline, not silently accept a malformed token as a valid proposal code |
| V6 Cryptography | No | No cryptographic operation is introduced |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A crafted classical-schedule-file line whose proposal token collides with (or is mistaken for) a status word/`BoN`/`EoN`/`HHMM` token, silently corrupting `source_identifier` uniqueness | Tampering (of ingested file content, not of a live request) | D-01 already requires the proposal-token grammar be unambiguous against `KNOWN_STATUSES`/`BoN`/`EoN`/`HHMM` shapes; `parse_run_line()`'s existing discipline (raise `ValueError` rather than guess, verified throughout `telescope_runs.py`) is the pattern to extend, not a new validation layer |
| Two schedule lines producing the same `source_identifier` (the exact SCHEMA-03 collision Phase 31 found in real data) silently merging two distinct proposals' calendar nights | Tampering / Repudiation (wrong data silently overwrites correct data) | The existing `unique_campaign_run_source_identifier` partial `UniqueConstraint` (`models.py:388-392`, already migrated) plus D-01's requirement that "a second line in the same file that yields an identical key is skipped and logged as a collision, never silently merged" — this is a data-integrity control, not a request-level one, and it is already schema-enforced at the DB level |
| An automated sweep silently overwriting a staff member's confirmed attribution during the allocation handoff | Tampering (of an audit trail) | Already solved by the existing "human outranks machine" guard (`_stale_attributions()`, `campaign_reconciler.py:482-527`, and `adopt_event_into_run()`'s own refusal when a companion row already points at a different run, `campaign_utils.py:977-979`) — D-08 explicitly requires the new link/unlink code reuse these exact functions rather than writing `CalendarEventMeta.run` directly |

No new ASVS gap is introduced by this phase: every input surface it touches (the schedule
file, the staff confirm/undo actions) is already validated/access-controlled by existing code
this phase is required to reuse (D-01, D-08).

## Sources

### Primary (HIGH confidence — read this session)
- `solsys_code/campaign_reconciler.py` (full file) — dispatch seam, `_observing_night()`,
  `_reconcile_classical_nights()`, `ReconcileResult`, ownership helpers
- `solsys_code/campaign_utils.py` (full file) — `write_and_reconcile_campaign_run()`,
  `adopt_event_into_run()`, `unlink_event_from_run()`, `UNLINK_CLEARED_FIELDS`
- `solsys_code/calendar_utils.py` (lines 460-699) — `record_time_window()`, `coerce_schedule_datetime()`,
  `insert_or_create_calendar_event()`, `update_calendar_event_key_and_fields()`,
  `preview_calendar_event_action()`
- `solsys_code/telescope_runs.py` (full file) — `SITES`, `sun_event()`, `parse_run_line()`,
  `ParsedRun`, `KNOWN_STATUSES`, `ESO_NOON_TO_NOON_SITES`, `_local_noon_utc()`
- `solsys_code/management/commands/load_telescope_runs.py` (full file) — current direct-write
  behavior being replaced
- `solsys_code/management/commands/reconcile_campaign_runs.py` (partial) — sweep entry point
- `solsys_code/models.py` (lines 1-540) — `CalendarEventMeta`, `CampaignRun` (fields, `Source`,
  `RunStatus`, constraints), `CampaignRunObservation`
- `solsys_code/observation_projector.py` (lines 560-738) — receiver shapes to mirror
- `solsys_code/apps.py` (full file) — receiver wiring pattern
- `solsys_code/campaign_views.py` (grep) — `AttributionDecisionView`, `reconcile_run()` call sites
- `solsys_code/solsys_code_observatory/models.py` (grep) — `SATELLITE_OBSTYPE`, `timezone` field
- `solsys_code/templatetags/calendar_display_extras.py` (grep) — `_TERMINAL_PREFIXES`
- `solsys_code/migrations/` (directory listing) — confirms latest migration is `0017_...py`
- `.claude/skills/spike-findings-fomo_devel/SKILL.md`, `references/allocation-handoff.md`,
  `references/observation-projector.md`, `sources/003-allocation-night-retirement/spike.py` —
  validated spike findings this phase builds on
- `solsys_code/tests/test_campaign_reconciler.py` (lines 1-75, class list) — existing fixture
  and test-class patterns to extend
- `solsys_code/tests/test_load_telescope_runs.py` (class/test list) — existing test inventory
- `docs/runbooks/telescope_runs_calendar.rst` (grep, section headers) — confirms the exact
  section names CONTEXT.md's paired-docs list references
- `docs/design/telescope_runs_calendar.rst` (lines 191-206) — "Night convention" section
- `.planning/config.json` — `nyquist_validation: true`, `security_enforcement: true`,
  `security_asvs_level: 1`, `test_command`
- Real dev database `src/fomo_db.sqlite3` — queried directly this session via `sqlite3`:
  total `CalendarEvent` count (241), `RUN:%:%`-keyed count (56), `ALLOC:%`-keyed count (0),
  `CampaignRun` count and `source` breakdown (45; legacy 24, csv_import 11, eso_queue 6,
  lco_queue 4), and `Observatory` rows for obscodes `X05`/`268`/`269`/`809`/`E10`/`K92`/`F65`
  with their `timezone` values
- Installed `tom_calendar/models.py` (site-packages) — `CalendarEvent.title`/`url` field
  definitions

### Secondary (MEDIUM confidence)
- `35-CONTEXT.md` itself — the user's locked decisions (D-01..D-18), treated as authoritative
  input rather than independently re-derived, per this agent's role (research grounds
  CONTEXT.md's claims against code, it does not re-litigate them)
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md` — requirement text and phase history

### Tertiary (LOW confidence)
- None — no WebSearch was performed for this phase; every claim traces to a file read this
  session, a DB query run this session, or the phase's own already-authoritative CONTEXT.md.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new library; every reused function's signature and location was
  read from disk this session.
- Architecture: HIGH — the dispatch seam, ownership helpers, and receiver-wiring pattern were
  all read verbatim from the modules this phase extends.
- Pitfalls: HIGH — each pitfall traces to either a measured spike result, a still-open
  documented todo, or a named prior code-review finding (CR-02) in this same codebase, not a
  generic best-practice guess.

**Research date:** 2026-09-12
**Valid until:** 30 days (stable, internal-only codebase; no third-party API/version drift
risk since no external package is introduced) — but re-verify the dev-DB baseline counts
(Runtime State Inventory) immediately before executing the cutover command, since those counts
are a snapshot, not a schema fact.
