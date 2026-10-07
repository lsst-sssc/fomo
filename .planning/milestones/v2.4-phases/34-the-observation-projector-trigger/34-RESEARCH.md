# Phase 34: The Observation Projector & Trigger - Research

**Researched:** 2026-09-10
**Domain:** Django ORM signals (post_save/m2m_changed/pre_delete), TOM Toolkit `tom_observations`/`tom_calendar` integration, idempotent create-or-update projection
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Compact title & series identity (PROJ-04 stem, PROJ-06)**
- **D-01: Stored title = `[marker] <telescope token> <target>`**, e.g. `[Q] 2m0 3I/ATLAS`,
  `[S] 1m0 11P`, `[O] FTS 3I/ATLAS`. The target is `record.target.name`; the telescope token is
  the coarse aperture class (`coarse_telescope_label()`: `0m4`/`1m0`/`2m0`, `4m0` for SOAR)
  while the record is queued or placed, and the observed telescope (D-07) once observed. The
  month cell shows the first 16/18 characters (`calendar.html` `truncatechars`), so the
  marker and token are always visible and the target usually is. Telescope and instrument
  still go into `CalendarEvent.telescope` / `.instrument` for the modal.
  — **Reversibility:** reversible — a title-builder change plus one sweep re-titles every event.
- **D-02: Provisional marker vocabulary — short letters everywhere.** `[Q]` queued,
  `[S]` placed (scheduled block, not yet observed), `[O]` observed (successful terminal
  state), `[X]` `WINDOW_EXPIRED`, `[C]` `CANCELED`, `[F]` `FAILURE_LIMIT_REACHED` and
  `NOT_ATTEMPTED`, `[?]` an inconsistent record (D-13). Phase 37 owns the final wording; this
  phase *extends* `calendar_display_extras._TERMINAL_PREFIXES` / `status_border_css` and the
  legend to recognise the new tokens while keeping the reconciler's `[CANCELLED]`/`[WEATHERED]`
  and the classical `[EXPIRED]`-style prefixes matching, so no existing ring is lost.
- **D-03: Every projector-written title carries exactly one marker; an observed record is
  `[O]`, never a bare title.** "No marker" is reserved to mean "not an observation event"
  across layers (allocation nights, `RUN:` containers and legacy classical events are all
  unmarked), so a clean title cannot be mistaken for "observed".
- **D-04: Series identity is the shared stem, nothing more, in the title.** Grouped records
  look alike in the cell (`[Q] 1m0 11P` × 28). "Night *n* of *N*", the group name and a link
  back to the group are rendered at display time in the modal from
  `meta.observation_group` (siblings ordered by `record_time_window()[0]`), via a template
  tag in the style of Phase 33's `campaign_decoration()`. Nothing group-derived is written
  into `title` or `description`, so adding or removing a sibling never churns the whole group.
- **D-05: Cross-layer telescope-token convention (recorded for Phases 35/37, implemented
  here only for observation events).** LCO/SOAR robotic records use the aperture class /
  observed telescope (D-01, D-07); allocation and classical events use the site short name
  already in `telescope_runs.SITES` (`NTT`, `FTN`, `FTS`, `Magellan-Clay`, …); `GN`/`GS`-style
  names would join `SITES` if a facility with real read-back ever needs them.

**Telescope label — verification without a network call in the hot path**
- **D-06: The projector never verifies; a coarse label while pending is by design.** While a
  record is queued or placed the token is the coarse aperture class with no portal call.
  `[UNVERIFIED]`, the `telescope_api_failed` counter and the "label unverified" description
  line retire with the old command. `CalendarEventMeta.is_verified` has no meaning for an
  observation event any more: the projector normalises it to `True` on every meta row it
  writes and never sets it `False`.
- **D-07: Once a record reaches a successful terminal state, the token becomes the
  telescope it was observed on.** `FTN` for `('ogg','2m0')`, `FTS` for `('coj','2m0')`,
  `SOAR` for `('sor','4m0')`, and `SITE-aperture` (`LSC-1m0`, `OGG-0m4`, …) for the 1m0/0m4
  network — i.e. `SITE_TELESCOPE_MAP` with its 2m0/4m0 values renamed. A COMPLETED record
  whose lookup has not succeeded yet keeps the coarse token under `[O]`.
- **D-08: The sweep makes the lookup, once per newly-observed record; the receiver never
  does.** For a record in a successful terminal state with no stored observed-site, the sweep
  calls `calendar_utils.resolve_placement_block()` once (10s timeout, never raises,
  COMPLETED-first-else-PENDING block), stores the result (D-09), then projects. A failed or
  unmapped lookup leaves the coarse token, is counted (`site_lookup_failed`), and is retried
  on the next sweep.
- **D-09: The observed site is stored on `ObservationRecord.parameters` under generic,
  un-prefixed, self-describing keys** that mirror the OCS observation block's own field
  names — e.g. `observed_site='ogg'`, `observed_telescope='2m0a'`, `observed_enclosure`
  optional. Not FOMO-prefixed. Keys must not collide with the LCO form's submission-constraint
  `site` field. The sweep saves the record with `update_fields=['parameters']`; that save
  fires the receiver once more, which projects and reports `unchanged`. Exact key names are
  the planner's within this rule.
  — **Reversibility:** costly — renaming the keys later means a data fix across every
  observed record's JSON, and any external consumer that learned the keys.

**Stage classification & edge lifecycles (PROJ-02, PROJ-03)**
- **D-10: Stage from record fields only, span from `record_time_window()`.** The spike 002
  classifier: half-set `scheduled_start`/`scheduled_end` → inconsistent (D-13); status in
  `facility.get_failed_observing_states()` → terminal-negative; status in
  `get_terminal_observing_states()` minus failed → observed (block present) or
  completed-no-block (D-12); otherwise placed (block present) or queued. Facility instance
  per record via `get_service_class(record.facility)()`, never one shared instance across
  LCO and SOAR.
- **D-11: A terminal-negative record keeps its full request window, marked** (`[X]`/`[C]`/`[F]`
  spanning the whole submitted window). A terminal-negative record that still carries a
  placed block keeps the block (the rule already prefers it).
- **D-12: COMPLETED with no block → `[O]` on the request window** (a successful-terminal
  record is never bannered as still queued); the observed-site lookup (D-08) still runs for it.
- **D-13: Inconsistent or unprojectable records.** A record with a usable request window but
  a half-set schedule projects as `[?]` on the window. A record with no usable window at all
  (missing `parameters['start'/'end']`, unparsable dates) is *unprojectable*: logged at
  warning (never interpolating an exception that could carry credentials — SYNC-09), counted
  by the sweep, any existing event left untouched, and the record save never aborted.
- **D-14: Deleting a record deletes its event.** A `pre_delete` receiver on `ObservationRecord`
  removes the projector-owned event — found through `instance.calendar_event_meta` *before*
  Phase 33's `SET_NULL` clears the link, and only if the event's `url` is the record's
  facility URL. The sweep cannot do this later (the record is gone), so it must be a receiver.
  — **Reversibility:** costly — a deleted event's attribution audit is gone with it.
- **D-15: Group membership keeps the link current through an `m2m_changed` receiver** on
  `ObservationGroup.observation_records.through`, re-projecting only the records in `pk_set`
  on `post_add` / `post_remove` / `post_clear` (both directions of the relation share the
  through model). Same single-record, never-raise contract as the `post_save` receiver. This
  matters because `backfill_lco_observations` adds group membership with `.add()` *after* the
  record save, which `post_save` never sees.
- **D-16: Receiver contract.** `post_save` on `ObservationRecord`, connected in
  `SolsysCodeConfig.ready()` with `dispatch_uid` and `weak=False`; returns immediately on
  `raw=True` (fixture loads) and for any facility other than `LCO`/`SOAR` (Gemini records
  stay with the submission-echo command); runs inline in the caller's transaction (TRIG-02),
  with no network call, no `sun_event`, and no write unless something changed
  (`insert_or_create_calendar_event()`'s contract); every failure is caught and logged, never
  re-raised. `settings.HOOKS['observation_change_state']` is left pointing at TOM's stock hook.

**Sweep, retirement, takeover & live proof (TRIG-03, ANNOT-03, PROJ-05, SCHED-06)**
- **D-17: Sweep command — zero required arguments, optional narrowing.** e.g.
  `python manage.py project_observation_calendar` sweeps every LCO/SOAR record; optional
  `--proposal A,B` (exact codes, no substring leakage), `--facility LCO|SOAR`, `--dry-run`
  (reports via `preview_calendar_event_action()`). Per-facility summary line in the retired
  command's phrasing: `created / updated / unchanged / unprojectable / site_lookups /
  site_lookup_failed`. Per-record failure isolation; a second run reports everything
  `unchanged`; the `RUN:` count, blank-url and `GEM:` events are provably untouched. Phase 36's
  cron calls it with no arguments.
- **D-18: `sync_lco_observation_calendar` is deleted outright** — the command module, its 38
  tests and its demo notebook. Behaviours worth keeping (no-churn, per-facility dispatch,
  exact-code proposal filter, failure-prefix priority, credential-free logging) are
  re-expressed as projector/sweep tests, not copied. The `calendar_utils` helpers it alone
  called (`resolve_placement_block`, `derive_telescope`, `SITE_TELESCOPE_MAP`,
  `aperture_class_from_telescope_code`) stay. Runbook §"How do I sync LCO/SOAR queue
  observations?" is replaced by a projector/sweep section; the cheat-sheet row,
  `docs/notebooks.rst:15` and CLAUDE.md's notebook map are updated.
  — **Reversibility:** costly — restoring the command means reviving a second writer of the
  same key namespace, the very thing ANNOT-03 removes.
- **D-19: The first sweep takes over the 156 legacy URL-keyed events; no migration.** It
  re-titles them to the D-01 form, links `observation_record` / `observation_group`,
  normalises `is_verified` (D-06), and makes the one-time site lookups for the observed ones.
  The demo notebook snapshots every event's `(url, title, start, end, meta links)` before and
  after and shows the `RUN:` / blank-url / `GEM:` sets byte-identical. One-time churn is
  accepted.
- **D-20: SCHED-06 is proven by a live-narrowing section in the sweep demo notebook**, built
  from spike 004's `recheck.py`: a baseline snapshot of `(status, scheduled_start/end, event
  start/end/title)` for the pending `KEY2026B-004` records; the operator runs *only* TOM's
  `updatestatus` over several real nights; the notebook is re-executed and committed showing
  records that moved `[Q]` → `[S]` → `[O]` with no sweep in between. `34-UAT.md` records the
  dates. Phase verification passes on the baseline plus the mechanism tests; the post-nights
  re-execution is a follow-up commit, not a gate on Phase 35 planning.
- **D-21: ANNOT-03's Gemini caveat.** `sync_gemini_observation_calendar` and its notebook are
  not changed in behaviour; the runbook's Gemini section and the notebook's prose document
  that `GEMFacility.get_observation_status()` / `get_observation_url()` are stubs, so Gemini
  events are submission-echo only and never narrow. The projector ignores `GEM` records (D-16).

### Claude's Discretion

- Module name and home for the projector and its receivers (e.g.
  `solsys_code/observation_projector.py`, receivers in the same module or a sibling
  `signals.py`), and the exact sweep command name (`project_observation_calendar` or an
  equivalent verb-noun name in the existing `*_observation_calendar` family).
- Exact `parameters` key names within D-09's rule; whether `--dry-run` performs the site
  lookups (recommended: no — dry-run must not write the record either).
- Which group wins for a record in more than one `ObservationGroup` (none exist in the dev
  DB; a deterministic pick such as lowest pk is fine) and the `target_list` choice (keep the
  old command's alphabetically-first `TargetList` rule).
- The `description` body (Proposal / Status / Window / Observed at … lines), log levels,
  summary-line wording, and how `[?]` is presented in the legend.
- Whether a settings flag or context manager is offered to silence the receivers during bulk
  test fixtures (the `raw` check already covers `loaddata`).
- Test file layout for the migrated behaviours; how the notebook's before/after diff is
  expressed.

### Deferred Ideas (OUT OF SCOPE)

- A `GN`/`GS`-style telescope token for Gemini or other non-LCO facilities in the `SITES`
  vocabulary — only meaningful once a facility with real read-back exists; noted for
  Phase 35/37 under D-05, not implemented here.
- The allocation layer and `ALLOC:` cutover (Phase 35); cron/`flock` scheduling, the
  watched-proposal list and failure notification (Phase 36); the final status vocabulary,
  status rings for every state, public tallies (Phase 37); any write to the `RUN:` namespace
  or to `CalendarEventMeta.run` / `confirmed_by` / `confirmed_at`; live Gemini read-back; ESO
  sync; upstreaming the projector.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PROJ-01 | Every LCO/SOAR `ObservationRecord` has exactly one `CalendarEvent`, keyed by `facility.get_observation_url()`, created or updated in place | Spike 002 `projector.py` `event_url()`/`project_record()` — proven against 146 real records; `calendar_utils.insert_or_create_calendar_event({'url': url}, fields)` gives the no-churn create/update contract for free |
| PROJ-02 | Event span follows record stage: request window → placed block → observed block | `calendar_utils.record_time_window()` [VERIFIED: solsys_code/calendar_utils.py:423-458] is the existing stage rule; D-10's classifier is a thin layer on top |
| PROJ-03 | Terminal-negative record keeps a visibly marked event on its window night | D-02/D-11 marker vocabulary + spike 002's `_failure_prefix()` reuse from `sync_lco_observation_calendar.py` [VERIFIED: solsys_code/management/commands/sync_lco_observation_calendar.py:28-33,52-65] |
| PROJ-04 (title-stem clause) | Series identity shared stem in title, group link at display time | Spike 002 `series_for()` numbering pattern; Phase 33's `CalendarEventMeta.observation_record`/`observation_group` [VERIFIED: solsys_code/models.py:54-69] already exist as the carrier; `campaign_decoration()` [VERIFIED: solsys_code/templatetags/calendar_display_extras.py:433-497] is the template-tag pattern to mirror |
| PROJ-05 | No-churn re-projection; never writes `RUN:`/allocation namespace | `insert_or_create_calendar_event()`'s `_update_or_unchanged()` [VERIFIED: solsys_code/calendar_utils.py:461-479] already implements the comparison; namespace isolation proven by spike 002's `sweep.py` isolation check |
| PROJ-06 | Compact titles fit a month cell | `calendar.html` `truncatechars:18`/`:16` [VERIFIED: src/templates/tom_calendar/partials/calendar.html:254,282]; D-01's `[marker] <token> <target>` form is sized against this |
| TRIG-01 | FOMO-owned `post_save` receiver registered in `apps.ready()`, catches schedule-only saves and `updatestatus` | Spike 001b `spike.py` scenarios S1/S4 [VERIFIED: sources/001-b-trigger-django-post-save/spike.py] proved both fire; TOM's own hook does not (S1/S4 false for the hook per event-trigger.md) |
| TRIG-02 | Single-record, idempotent, cheap, error never aborts save | Spike 002 `_receiver()`/`project_record()` never-raise pattern [VERIFIED: sources/002-observation-projector/projector.py:158-172,190-196] |
| TRIG-03 | Sweep command backstop with `--dry-run`, per-record isolation, paired notebook | Spike 002 `sweep.py` structure; `reconcile_campaign_runs.py` sibling sweep for `--dry-run`/summary conventions |
| SCHED-06 | Live-night narrowing proof against real `KEY2026B-004` records | Spike 004 `recheck.py` [VERIFIED: sources/004-live-narrowing-updatestatus/recheck.py] — baseline/recheck pattern, PARTIAL verdict closed by re-running post-nights |
| ANNOT-03 | Retire `sync_lco_observation_calendar`; Gemini stays submission-echo | D-18/D-21; `sync_lco_observation_calendar.py` behaviours to re-express [VERIFIED: solsys_code/management/commands/sync_lco_observation_calendar.py] |
</phase_requirements>

## Summary

Phase 34 turns a pattern already proven end-to-end in four spikes (001a/b, 002, 003, 004) into
production code. The core mechanism is small: reuse `calendar_utils.record_time_window()` and
`insert_or_create_calendar_event()` — both already shipped and used by the command being
retired — inside a ~150-line classifier/projector module, wire it to `ObservationRecord` via a
Django `post_save` receiver (not TOM's `observation_change_state` hook, which is silent for
schedule-only placement saves), add an `m2m_changed` receiver for group membership and a
`pre_delete` receiver for record deletion, and add a zero-argument sweep management command as
the backstop for bulk-write paths. Every one of these pieces was run against the real 146
`KEY2026B-004` records in spikes 002/003/004 and works.

The two genuinely new pieces this phase must design (not just port) are: (1) a compact,
single-marker title vocabulary (`[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` + coarse-or-observed
telescope token + target name) that fits a month cell and never collides with non-observation
event titles, and (2) writing the Phase 33 `CalendarEventMeta.observation_record`/
`observation_group` link fields as the real series-identity carrier (replacing spike 002's
title-suffix stopgap), with group decoration rendered at display time via a `campaign_decoration()`-
style template tag so re-projection never erases it.

The single hardest correctness edge, called out repeatedly in the spike sources and the
CONTEXT.md decisions, is namespace isolation: the projector must touch only
`CalendarEvent` rows whose `url` equals `facility.get_observation_url(observation_id)`, and
must never create, modify, or delete a `RUN:`-prefixed (reconciler), `GEM:`-prefixed (Gemini
echo), or blank-url (classical) event. Phase 33 already inverted the reconciler to
annotate-only, so this phase is safe to build; the sweep's own isolation check (`RUN:` count
before == after) is the regression test that guards it forever.

**Primary recommendation:** Port spike 002's `projector.py` classifier and event-building logic
nearly verbatim into a new `solsys_code/observation_projector.py`, wire the three receivers in
`SolsysCodeConfig.ready()` per spike 001b's proven contract, build the sweep as a management
command mirroring `reconcile_campaign_runs.py`'s `--dry-run`/summary conventions, and retire
`sync_lco_observation_calendar` in the same phase since the takeover of its 156 legacy events is
a plain `insert_or_create_calendar_event()` update once the projector exists.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Classify record lifecycle stage (queued/placed/observed/terminal-negative/inconsistent) | API/Backend | — | Pure Python over `ObservationRecord` fields + facility class constants; no I/O (D-10) |
| Build `CalendarEvent` field values (title/description/span) | API/Backend | — | `calendar_utils` helper functions, in-process, no network (D-06/D-16) |
| Persist/update `CalendarEvent` and `CalendarEventMeta` rows | Database/Storage | API/Backend | Django ORM `get_or_create`/`save()`; the projector is the sole writer for its key namespace (PROJ-01/05) |
| Trigger re-projection on every record save | API/Backend | — | Django `post_save`/`m2m_changed`/`pre_delete` signal receivers wired in `apps.ready()` — in-process, same transaction (TRIG-01/02) |
| Backstop bulk-write re-projection | API/Backend | — | Django management command (`project_observation_calendar`), invoked by an operator or (Phase 36) cron (TRIG-03) |
| One-time observed-site network lookup | API/Backend | External Service (LCO portal) | `resolve_placement_block()` — timeout-bounded HTTP GET, sweep-only, never in the receiver's hot path (D-08) |
| Render compact title / status ring in month cell | Frontend Server (Django SSR) | Browser (CSS `truncatechars`) | `calendar_display_extras.py` template tags + `calendar.html` partial — server-rendered HTML, client only applies CSS (PROJ-06/D-02) |
| Render series ("night n of N") and campaign decoration in event modal | Frontend Server (Django SSR) | — | New template tag reading `CalendarEventMeta.observation_group`/`.run` at request time, mirroring `campaign_decoration()` (D-04) |

## Standard Stack

### Core

No new external packages. This phase is entirely composed from libraries already installed and
already imported by the code it replaces.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django | 5.2.17 [VERIFIED: `python -c "import django; print(django.get_version())"`, this session] | ORM, signals (`post_save`/`m2m_changed`/`pre_delete`), management commands | Already the project's web framework; signals are the stdlib mechanism for event-driven re-projection (TRIG-01) |
| tomtoolkit | 3.0.1 [VERIFIED: `pip show tomtoolkit`, this session] | `tom_observations.models.ObservationRecord`/`ObservationGroup`, `tom_observations.facility.get_service_class`, `tom_calendar.models.CalendarEvent` | The models and facility abstraction this phase projects from/into; already a locked dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| requests | (already a transitive dep, used by `calendar_utils.resolve_placement_block`) | One-time observed-site HTTP GET | Sweep-only, 10s timeout, never raises (D-08) — reused unchanged, not re-implemented |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Django `post_save` signal | TOM's `observation_change_state` hook | Rejected: fires only on status change or creation; silent for the scheduler's schedule-only placement save (`scheduled_start`/`scheduled_end` set, `status` unchanged) — exactly the narrowing step that matters (spike 001a PARTIAL verdict) |
| Django `post_save` signal | Celery/task-queue-driven async re-projection | Rejected: v2.3's Phase 31 already settled cron+`flock` over a task queue for this project (locked constraint); the projector must also be synchronous/inline per TRIG-02 |
| One-time sweep site lookup | Live lookup inside the `post_save` receiver | Rejected: violates "no network call in the hot path" (D-08/D-16); would make every save latency-dependent on an external portal |

**Installation:**
```bash
# No new packages. All imports below are already available in the venv.
python3 -c "import django, tom_observations, tom_calendar, requests; print('ok')"
```

**Version verification:** Confirmed this session via `pip show tomtoolkit` (3.0.1) and
`python3 -c "import django; print(django.get_version())"` (5.2.17). No `pyproject.toml` change
is expected for this phase.

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** Every import the projector, its
receivers, and the sweep command need (`django.db.models.signals`, `tom_observations.*`,
`tom_calendar.models`, `solsys_code.calendar_utils`, `requests`) is already installed and
already used by the code being ported from or retired (`sync_lco_observation_calendar.py`,
`calendar_utils.py`). The Package Legitimacy Gate is not applicable — there is nothing to run
`gsd_run query package-legitimacy check` against.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                     ┌───────────────────────────────────────────────┐
                     │            Django signal dispatch              │
                     │        (registered in apps.ready())            │
                     └───────────────────────────────────────────────┘
   ObservationRecord.save()          ObservationGroup.observation_records
   (incl. scheduler placement,       .add()/.remove()/.clear()
   updatestatus, backfill create)              │
        │                                      │
        ▼                                      ▼
  post_save receiver                    m2m_changed receiver
  (raw? -> return;                      (pk_set -> re-project only
   facility LCO/SOAR? else return)       those records)
        │                                      │
        └──────────────┬───────────────────────┘
                        ▼
              project_record(record)                 ObservationRecord.pre_delete
              ─────────────────────                          │
              1. facility_for(record)                         ▼
                 (get_service_class, per-facility     pre_delete receiver
                  instance, never shared)              (find event via
              2. stage_for(record, facility)            instance.calendar_event_meta
                 (D-10 classifier: queued /             BEFORE SET_NULL clears link;
                  placed / observed /                   delete only if url matches
                  completed-no-block /                  this record's facility URL)
                  terminal-negative / inconsistent)
              3. event_fields_for(record, facility)
                 - calendar_utils.record_time_window()  (span, PROJ-02)
                 - coarse_telescope_label() / D-07 observed-site token
                 - title_for(): [marker] <token> <target>  (D-01..D-04)
              4. never raises -> ('unprojectable', reason) on failure (D-13)
                        │
                        ▼
        calendar_utils.insert_or_create_calendar_event({'url': url}, fields)
        ─────────────────────────────────────────────────────────────────
        get_or_create by url -> create | update-if-changed | unchanged (PROJ-05)
                        │
                        ▼
        ┌───────────────────────────────┬───────────────────────────────┐
        │  tom_calendar.CalendarEvent    │   solsys_code.CalendarEventMeta │
        │  (title/description/span/      │   (observation_record OneToOne, │
        │   telescope/instrument)        │    observation_group FK,        │
        │                                │    is_verified=True)            │
        └───────────────────────────────┴───────────────────────────────┘
                        │
                        ▼
        month-cell / modal rendering (calendar.html, calendar_display_extras.py)
        - status_border_css() / _TERMINAL_PREFIXES extended with new markers
        - new template tag renders "night n of N" from observation_group at display time

  Backstop path (bulk writes / backfill / cron, TRIG-03):
  management command `project_observation_calendar`
        │
        ├── --proposal / --facility filters, --dry-run (preview_calendar_event_action)
        ├── sweep-only: resolve_placement_block() one-time observed-site lookup (D-08)
        └── calls project_record() over a queryset, same core logic as the receiver
```

### Recommended Project Structure
```
solsys_code/
├── observation_projector.py   # stage_for(), event_fields_for(), title_for(), project_record(),
│                               # project_queryset() — the core, network-free, never-raise logic
├── signals.py                 # post_save / m2m_changed / pre_delete receiver functions
│                               # (or keep receivers in observation_projector.py — discretionary)
├── apps.py                    # SolsysCodeConfig.ready() connects the three receivers
├── calendar_utils.py          # UNCHANGED except SITE_TELESCOPE_MAP value renames (D-07)
├── management/commands/
│   ├── project_observation_calendar.py   # the sweep (new)
│   └── sync_lco_observation_calendar.py  # DELETED (D-18)
├── templatetags/calendar_display_extras.py  # extend _TERMINAL_PREFIXES/status_border_css;
│                                             # add a series/night-n-of-N template tag
└── tests/
    ├── test_observation_projector.py     # classifier + event-fields unit tests (new)
    ├── test_observation_projector_signals.py  # receiver contract tests (new)
    └── test_project_observation_calendar.py   # sweep command tests (new, migrates the 38
                                                # sync_lco tests' behaviours)
```

### Pattern 1: Never-raise projection wrapped around fallible field-building

**What:** Separate a fallible pure function (`event_fields_for`) that raises on bad input from a
wrapper (`project_record`) that catches everything and converts it to a sentinel action string.
**When to use:** Any function called from inside a `post_save` receiver, where an exception must
never propagate and abort the caller's save (TRIG-02).
**Example:**
```python
# Source: sources/002-observation-projector/projector.py:158-172 (spike, VALIDATED against
# 146 real records) — port this logic into solsys_code/observation_projector.py
def project_record(record: ObservationRecord) -> tuple[str, str]:
    facility = facility_for(record)
    try:
        fields, stage = event_fields_for(record, facility)
    except Exception as exc:  # noqa: BLE001 -- a projector must never break the triggering save
        logger.warning('unprojectable observation_id=%r: %s', record.observation_id, exc)
        return 'unprojectable', f'{type(exc).__name__}: {exc}'
    _event, action = insert_or_create_calendar_event({'url': event_url(record, facility)}, fields)
    return action, stage
```

### Pattern 2: `raw`-guarded, facility-filtered signal receiver

**What:** A `post_save` receiver that returns immediately for fixture loads (`raw=True`) and for
facilities the projector does not own (Gemini), before doing any work.
**When to use:** Any receiver connected globally in `apps.ready()` — it fires for every save of
the sender model across the whole app, including test fixtures and other facilities.
**Example:**
```python
# Source: sources/002-observation-projector/projector.py:190-196 (spike), refined by
# CONTEXT.md D-16 (facility filter added — spike 002 had none)
def _receiver(sender, instance: ObservationRecord, created: bool, raw: bool, **kwargs) -> None:
    if raw:
        return
    if instance.facility not in ('LCO', 'SOAR'):
        return
    action, stage = project_record(instance)
    logger.info('post_save projected observation_id=%r created=%s -> %s (%s)',
                instance.observation_id, created, action, stage)
```

### Pattern 3: `m2m_changed` receiver scoped to `pk_set`

**What:** Re-project only the records named in the signal's `pk_set`, not the whole group, on
`post_add`/`post_remove`/`post_clear`.
**When to use:** `ObservationGroup.observation_records` is a `ManyToManyField` [VERIFIED:
`/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/models.py:110`,
`observation_records = models.ManyToManyField(ObservationRecord)`], so its implicit through
table is `ObservationGroup.observation_records.through` — that is the `sender` to connect to.
**Example:**
```python
from django.db.models.signals import m2m_changed
from tom_observations.models import ObservationGroup, ObservationRecord

def _group_membership_changed(sender, instance, action, pk_set, **kwargs):
    if action not in ('post_add', 'post_remove', 'post_clear'):
        return
    if pk_set is None:  # post_clear
        return
    for pk in pk_set:
        try:
            record = ObservationRecord.objects.get(pk=pk)
        except ObservationRecord.DoesNotExist:
            continue
        if record.facility in ('LCO', 'SOAR'):
            project_record(record)

# apps.py ready():
m2m_changed.connect(_group_membership_changed,
                     sender=ObservationGroup.observation_records.through,
                     weak=False, dispatch_uid='solsys_code.observation_group_projector')
```
This closes the gap `backfill_lco_observations.py`'s `group.observation_records.add(*processed_in_group)`
[VERIFIED: `solsys_code/management/commands/backfill_lco_observations.py:672`] leaves: that
`.add()` call happens *after* each record's own `.save()`/`get_or_create()`, so a plain
`post_save` receiver never sees the group link (D-15).

### Pattern 4: `pre_delete`-based single-record cascade, found before the FK clears

**What:** Look up the projector-owned event through the reverse one-to-one link *before* the
Phase 33 `SET_NULL` on-delete clears it, and delete only if the event's `url` matches this
record's own facility URL (namespace safety).
**When to use:** Deleting a row whose companion link uses `on_delete=SET_NULL` — the standard
`post_delete` signal fires *after* the FK is already null.
**Example (mirrors the existing `CampaignRun` pre_delete receiver):**
```python
# Source: solsys_code/models.py:423-455 [VERIFIED] — same pre_delete-before-SET_NULL pattern,
# different model. Reuse this pattern, not this code, for ObservationRecord.
from django.db.models.signals import pre_delete
from django.dispatch import receiver

@receiver(pre_delete, sender=ObservationRecord)
def _delete_owned_calendar_event_on_record_delete(sender, instance, **kwargs):
    try:
        meta = instance.calendar_event_meta  # CalendarEventMeta.observation_record's related_name
    except ObservationRecord.calendar_event_meta.RelatedObjectDoesNotExist:
        return
    facility = facility_for(instance)
    if meta.event.url == facility.get_observation_url(instance.observation_id):
        meta.event.delete()  # cascades to CalendarEventMeta via CASCADE
```
Note `CalendarEventMeta.observation_record`'s `related_name='calendar_event_meta'`
[VERIFIED: `solsys_code/models.py:54-60`, `related_name='calendar_event_meta'`] — this is the
accessor `instance.calendar_event_meta` above relies on.

### Pattern 5: Title-priority ladder (failure > stage > clean-with-marker)

**What:** A single ordered rule for which marker wins, so failure states always dominate even
over stage markers.
**When to use:** Building `title_for()`.
**Example (spike 002's ladder, adapted to D-01..D-04's single-marker vocabulary):**
```python
# Source: sources/002-observation-projector/projector.py:107-118 (spike source) +
# solsys_code/management/commands/sync_lco_observation_calendar.py:52-65 (_failure_prefix,
# reused verbatim per D-18)
_STAGE_MARKER = {'queued': '[Q]', 'placed': '[S]', 'observed': '[O]', 'completed-no-block': '[O]'}

def title_for(record, stage, token, target_name) -> str:
    marker = _failure_marker(record.status, facility)  # [X]/[C]/[F], D-02
    if marker is None:
        marker = _STAGE_MARKER.get(stage, '[?]')        # D-13 fallback for 'inconsistent'
    return f'{marker} {token} {target_name}'[:200]
```

### Anti-Patterns to Avoid

- **Live telescope-site resolution inside the receiver:** the old sync command's per-record
  portal call for a verified label must not move into the `post_save` path — it makes every
  record save latency-dependent on an external HTTP call (D-08/D-16, "What to Avoid" in
  `references/event-trigger.md` and `references/observation-projector.md`).
- **Writing campaign text into the event's own title/description:** the next re-projection
  rebuilds those fields from record state and silently erases it (spike 003, proven — this is
  why attribution is `CalendarEventMeta.run`, a link, rendered at display time).
- **Trusting `update_fields` to know what changed:** TOM's `update_observation_status()` calls a
  bare `record.save()` [VERIFIED: `/home/tlister/venv/devel_fomo311_venv/.../tom_observations/facility.py:555-565`,
  `record.save()` with no `update_fields` argument], so `update_fields` is `None` on that path —
  do not gate the receiver's work on it being non-`None`.
- **Assuming `QuerySet.update()`/`bulk_create()`/`bulk_update()` will trigger the receiver:**
  they bypass `Model.save()` entirely (spike 001b scenario S5, `expect_fire=False`, confirmed
  fired=False) — this is precisely why the sweep command must exist as a backstop.
- **Recomputing an observed-site lookup on every sweep run:** it must be looked up exactly once
  per record (store on `parameters`, D-09) and skipped once already present, or the sweep pays
  an HTTP round-trip per record on every invocation forever.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stage classification from schedule fields | A new half-set/fully-set/null parser | `calendar_utils.record_time_window()` [VERIFIED: solsys_code/calendar_utils.py:423-458] | Already handles the exact raising contract (`KeyError`/`ValueError`) the projector's `event_fields_for()` must catch |
| Create-or-update-or-unchanged CalendarEvent write | A bespoke `get_or_create` + manual diff | `calendar_utils.insert_or_create_calendar_event()` [VERIFIED: solsys_code/calendar_utils.py:482-542] | Already the no-churn contract three other consumers share; reimplementing risks a second, subtly different diff rule |
| `--dry-run` preview without writing | A parallel "would-be" code path | `calendar_utils.preview_calendar_event_action()` [VERIFIED: solsys_code/calendar_utils.py:575-595] | Uses the identical comparison `_update_or_unchanged()` uses, so dry-run counts can never disagree with a real run |
| Observed-site telescope label resolution | A new LCO API client | `calendar_utils.resolve_placement_block()` + `derive_telescope()` + `SITE_TELESCOPE_MAP` [VERIFIED: solsys_code/calendar_utils.py:255-304,232-252,42-57] | Already timeout-bounded (10s), never-raising, and selects the same COMPLETED-first-else-PENDING block TOM's own status poll uses |
| Per-facility failed/terminal state check | Hardcoding `status == 'COMPLETED'` | `facility.get_failed_observing_states()` / `get_terminal_observing_states()` [VERIFIED: `/home/tlister/venv/.../tom_observations/facilities/ocs.py:1435-1439`, delegates to `facility_settings`; concrete values `['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']` confirmed at `facility_settings.py:118-122` for the shared OCS settings base] | Per-facility differences are real (STATUS-02, Phase 37); hardcoding one facility's states silently mis-classifies another |
| Sweep summary/`--dry-run` command layout | A new CLI argument-parsing pattern | `reconcile_campaign_runs.py`'s existing `--dry-run` + summary-line conventions | Sibling sweep command in the same codebase; consistent operator experience across the two sweeps |

**Key insight:** Nearly the entire correctness burden of this phase was already discharged by
the four validated spikes and by helpers `calendar_utils.py` already ships. The genuinely new
code is: the stage classifier (a ~15-line function), the title builder (the marker vocabulary),
the three signal receivers (thin wrappers with guard clauses), and the sweep's CLI shell. Writing
any of these from scratch without reusing the cited helpers reintroduces bugs the spikes already
found and fixed (e.g., Pitfall 4 in the old sync command: an API-failure fallback and a
successfully-returned-but-unmapped pair must share one fallback bucket).

## Runtime State Inventory

> Included because ANNOT-03/D-19 takes over 156 live `CalendarEvent` rows in place — a
> takeover, not a schema migration, but the same "what runtime state still carries the old form"
> question applies.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | 156 URL-keyed `CalendarEvent` rows carrying spike-002-stopgap titles (title-suffix series identity, no `observation_record`/`observation_group` link, `is_verified` reflecting the old command's `telescope_api_failed` signal) — dev DB baseline 2026-09-10 [context via 34-CONTEXT.md code_context, not independently re-queried this session] | Code edit only, no separate migration script: the first sweep run re-titles, re-links, and normalises these rows in place (D-19). No `RunPython` migration needed — the sweep *is* the one-time takeover mechanism. |
| Live service config | None — no external service (n8n/Datadog/Tailscale-style) holds LCO/SOAR sync configuration outside this repo. The LCO/SOAR portal itself is queried live for the one-time site lookup (D-08), not configured by the retired command. | None |
| OS-registered state | None yet — Phase 36 (not this phase) puts the sweep on cron/`flock`. No OS scheduler entry currently references `sync_lco_observation_calendar`. | None for this phase; Phase 36 will add and must not reference the retired command name |
| Secrets/env vars | None — `sync_lco_observation_calendar` reads no credential of its own; it uses `LCOFacility()`'s existing settings-based portal URL/API key, unchanged by this phase's retirement | None |
| Build artifacts / installed packages | `sync_lco_observation_calendar.py`'s 38-test module (`test_sync_lco_observation_calendar.py`) and its demo notebook (`sync_lco_observation_calendar_demo.ipynb`) are the "stale artifact" risk: deleting the command without migrating these leaves dead test infrastructure and a notebook that fails to execute against a module that no longer exists | Delete the command module; re-express its behaviours as new tests under the projector/sweep test files (D-18); replace the demo notebook with the new sweep notebook rather than deleting it outright (paired-docs rule) |

**Nothing found in category:** Live service config, OS-registered state, and secrets/env vars —
verified by reading `sync_lco_observation_calendar.py` in full this session; it has no
`os.getenv()` calls, no cron/task-scheduler registration, and reads facility credentials only
through the already-shared `LCOFacility()`/`SOARFacility()` instances the projector will also use.

## Common Pitfalls

### Pitfall 1: Using the wrong Django hook as the trigger

**What goes wrong:** Wiring the projector to `settings.HOOKS['observation_change_state']`
instead of a `post_save` signal. The projector silently never fires when the scheduler places a
block (status stays `PENDING`, only `scheduled_start`/`scheduled_end` change).
**Why it happens:** `ObservationRecord.save()` [VERIFIED:
`/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/tom_observations/models.py:58-66`]
only calls `run_hook('observation_change_state', ...)` when `self.status != presave_data.status`
or on creation — a schedule-only save changes neither.
**How to avoid:** Connect a `post_save` receiver directly to `ObservationRecord` in
`apps.ready()`; leave `HOOKS['observation_change_state']` pointed at TOM's stock no-op (D-16).
**Warning signs:** A record's `scheduled_start`/`scheduled_end` update in the admin/DB but its
calendar event never narrows to the placed block.

### Pitfall 2: Assuming bulk writes trigger the receiver

**What goes wrong:** `backfill_lco_observations`, `updatestatus`'s internal bulk-ish loop (it
actually calls `.save()` per record, so it *is* covered — see Pitfall 3), or any future
`QuerySet.update()`/`bulk_create()`/`bulk_update()` call silently produces records with stale or
missing calendar events.
**Why it happens:** `QuerySet.update()`/`bulk_create()`/`bulk_update()` bypass `Model.save()`
entirely and never send `post_save` — proven in spike 001b scenario S5 (`expect_fire=False`,
confirmed).
**How to avoid:** The sweep command (TRIG-03) is the mandatory backstop for any code path that
writes records this way; run it after backfill and after any bulk-update script.
**Warning signs:** A record's DB fields are correct but its `CalendarEvent` is missing, stale, or
absent from the projector's key namespace entirely.

### Pitfall 3: Confusing "TOM's `updatestatus` calls `.save()` per record" with "bulk-safe"

**What goes wrong:** Assuming `updatestatus` is a bulk path that needs the sweep, when in fact
it is `post_save`-safe already (each record gets its own `record.save()` call inside
`update_observation_status()` [VERIFIED: `.../tom_observations/facility.py:555-565`]).
**Why it happens:** `update_all_observation_statuses()` *looks* like a bulk operation (iterates
a queryset) but delegates to per-record `.save()` internally.
**How to avoid:** Confirm this with spike 001b's own S4 scenario (`updatestatus path, block
placed, state unchanged`, `expect_fire=True`, confirmed fired) rather than assuming from the
method name. `updatestatus` genuinely needs no sweep call for correctness (though Phase 36 still
chains a sweep after it, for the observed-site lookup batching, not for correctness of the
projection itself).
**Warning signs:** None expected if the receiver is wired correctly — this pitfall is about
mis-scoping the sweep's necessity, not a runtime bug.

### Pitfall 4: Treating an API-lookup failure and a successfully-resolved-but-unmapped result differently

**What goes wrong:** Splitting "portal call failed/timed out" from "portal call succeeded but
returned a `(site, telescope_code)` pair not in `SITE_TELESCOPE_MAP`" into two different
fallback behaviours (e.g., one silent, one logged) produces inconsistent telescope labels for
what is operationally the same situation: "we don't have a verified label."
**Why it happens:** It is tempting to treat a raised exception and a clean `None` return
differently in application code.
**How to avoid:** `resolve_placement_block()` already returns `None` for both cases (any
exception is caught internally and converted to `None`) [VERIFIED:
solsys_code/calendar_utils.py:280-304]; the caller's `derive_telescope(block.get('site'),
block.get('telescope')) if block is not None else None` treats a `None` block and an unmapped
resolved pair identically, per the original command's D-07/Pitfall-4 note
[VERIFIED: solsys_code/management/commands/sync_lco_observation_calendar.py:176-190].
**Warning signs:** Two different log messages or counters for what should be one
`site_lookup_failed` bucket (D-08).

### Pitfall 5: Logging a caught network exception verbatim

**What goes wrong:** A caught `requests`/portal exception is interpolated into a log line and
ends up carrying response bodies that can include credentials
(`ImproperCredentialsException`/`forms.ValidationError` embed `response.content` directly per
`resolve_placement_block()`'s own docstring).
**Why it happens:** The natural instinct when debugging a failed HTTP call is `logger.warning(f'... {exc}')`.
**How to avoid:** Follow the existing SYNC-09/D-11 discipline: log a fixed, generic message
(`'Telescope API lookup failed or returned an unmapped code for observation_id=%r; using
fallback label.'`) never the exception object itself — the pattern the retired command already
uses at `sync_lco_observation_calendar.py:331-338`.
**Warning signs:** A code reviewer or the security-review pass finds an f-string with `{exc}` or
`{e}` inside any except block that catches a `requests`/portal exception.

### Pitfall 6: Forgetting the group-membership-after-save gap

**What goes wrong:** A record freshly created by `backfill_lco_observations` and then added to
an `ObservationGroup` via `.add()` gets a calendar event from its own `post_save`, but that
event never picks up series identity because the group link didn't exist yet at save time, and
nothing re-projects it afterward.
**Why it happens:** `backfill_lco_observations.py:672` calls
`group.observation_records.add(*processed_in_group)` strictly after each record's own
`update_or_create`/`.save()` in the loop above it [VERIFIED:
solsys_code/management/commands/backfill_lco_observations.py:640-673].
**How to avoid:** The `m2m_changed` receiver (D-15, Pattern 3 above) re-projects every record in
`pk_set` on `post_add`.
**Warning signs:** A multi-request `ObservationGroup`'s events show correct individual titles
but the modal never shows "night n of N" for any of them.

## Code Examples

### Stage classifier (verified pattern, port near-verbatim)
```python
# Source: sources/002-observation-projector/projector.py:91-104 (spike, run against 146 real
# KEY2026B-004 records — see references/observation-projector.md "How to Build It" step 2)
def stage_for(record: ObservationRecord, facility) -> str:
    has_start = record.scheduled_start is not None
    has_end = record.scheduled_end is not None
    if has_start != has_end:
        return 'inconsistent'                       # D-13
    has_block = has_start and has_end
    failed = set(facility.get_failed_observing_states())
    successful = set(facility.get_terminal_observing_states()) - failed
    if record.status in failed:
        return 'terminal-negative'                   # D-11
    if record.status in successful:
        return 'observed' if has_block else 'completed-no-block'  # D-12
    return 'placed' if has_block else 'queued'
```

### Facility instance cache (one per facility name, never shared across LCO/SOAR)
```python
# Source: sources/002-observation-projector/projector.py:39,51-56 (spike)
from tom_observations.facility import get_service_class

_facilities: dict[str, object] = {}

def facility_for(record: ObservationRecord):
    name = record.facility
    if name not in _facilities:
        _facilities[name] = get_service_class(name)()
    return _facilities[name]
```

### Sweep summary-line format (mirrors the retired command's D-08 per-facility breakdown)
```python
# Adapted from solsys_code/management/commands/sync_lco_observation_calendar.py:351-364
# [VERIFIED] -- keep the same per-facility phrasing per D-17, add the two new counters
# (site_lookups / site_lookup_failed) and drop telescope_api_failed/extraction_failed which
# retire with the old command.
summary = ' | '.join(
    f'{facility_name}: created: {counts["created"]}, updated: {counts["updated"]}, '
    f'unchanged: {counts["unchanged"]}, unprojectable: {counts["unprojectable"]}, '
    f'site_lookups: {counts["site_lookups"]}, site_lookup_failed: {counts["site_lookup_failed"]}'
    for facility_name, counts in counters.items()
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `sync_lco_observation_calendar --proposal <codes>` run manually or by cron, one command syncs everything for named proposals | `post_save`/`m2m_changed`/`pre_delete` receivers project every save automatically; `project_observation_calendar` sweep is the backstop, no `--proposal` required by default | Phase 34 (this phase) | An operator no longer needs to remember to run a sync command after every scheduler cycle; the calendar narrows live |
| Live per-record telescope-site API call on every sync run for every placed record | One-time site lookup per record, only once it reaches a successful terminal state, stored on `parameters` and never re-queried | Phase 34 (D-08/D-09) | Eliminates ~60+ portal calls per sweep run that Phase 4-era design required; sweep cost measured at 0.1-0.3s for 146 records in spike 002 with no network calls in the hot path |
| Title-suffix stopgap for series identity (`· <group name> <i>/<n>`) written into the stored title | Real FK carrier (`CalendarEventMeta.observation_record`/`observation_group`, shipped Phase 33), decoration rendered at display time | Phase 33 ships carrier, Phase 34 writes it (PROJ-04 split) | Adding/removing a group member no longer requires re-titling every sibling event |
| Campaign attribution risk: reconciler's adopt/re-key/detach paths could steal base-layer events | Reconciler inverted to annotate-only (Phase 33, ANNOT-01/02) | Phase 33 (2026-09-10, complete) | Makes Phase 34 safe to ship — the base layer and campaign layer can now run side by side |

**Deprecated/outdated:**
- `sync_lco_observation_calendar` management command: retired outright (D-18), not deprecated-in-place — its module, 38 tests, and demo notebook are deleted, with behaviours re-expressed as projector/sweep tests.
- `CalendarEventMeta.is_verified=False` (dashed-border rendering, `calendar.html:247-248,265-268` [VERIFIED]) for observation events: still exists in the template but becomes permanently unreachable for observation-backed events once the projector normalises every meta row it writes to `True` (D-06) — the dashed border will only ever appear for pre-Phase-34 rows until the takeover sweep runs, after which it disappears entirely for this event class.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Dev DB baseline counts (238 events, 156 URL-keyed, 74 pending KEY2026B-004 records, etc.) as stated in 34-CONTEXT.md's `code_context` section are accurate as of 2026-09-10 | Runtime State Inventory, Summary | Not independently re-queried against the live DB this session; if stale, the takeover notebook's before/after diff (D-19) would need re-baselining, but this does not change the projector's design |
| A2 | `facility_settings.get_failed_observing_states()` returns exactly `['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED']` for both `LCOFacility` and `SOARFacility` (SOAR inherits from `LCOFacility`) | Don't Hand-Roll table, Stage classifier | If SOAR's settings ever override these states, the terminal-negative classification could silently diverge between facilities; verified only for the shared `OCSSettings`/base values this session, not SOAR-specific overrides |
| A3 | The exact `parameters` key names for observed-site storage (`observed_site`/`observed_telescope`/`observed_enclosure`) are the planner's discretion per D-09's rule, not yet locked | User Constraints (Claude's Discretion) | If the planner picks names colliding with an existing `parameters` key on some record, the sweep's `update_fields=['parameters']` save could silently overwrite unrelated data — the planner should add a collision check against the LCO submission form's own field vocabulary |

**If this table is empty:** N/A — see above.

## Open Questions (RESOLVED)

All three questions below were settled during planning; each recommendation was adopted
verbatim by the Phase 34 plans, and the plan text that implements it is named in the
resolution line.

1. **Exact module/file layout for the projector and its three receivers** — **RESOLVED:
   receivers are co-located in `observation_projector.py`.**
   - What we know: CONTEXT.md leaves this to Claude's discretion; a plausible split is
     `observation_projector.py` (pure logic) + `signals.py` (receiver wiring) + `apps.py`
     (connection), mirroring the existing `campaign_reconciler.py`/`campaign_utils.py` split.
   - What was unclear: Whether the plan-checker or reviewers prefer receivers co-located with the
     logic they call (fewer files, easier to trace) vs. separated (clearer signal-wiring
     surface for future readers).
   - Recommendation: Co-locate receivers in `observation_projector.py` for Phase 34 (mirrors
     spike 002's `projector.py` which included `_receiver`/`connect`/`disconnect` in one file);
     split out only if the file grows unwieldy.
   - **RESOLVED — adopted.** Plan 34-01 Tasks 1 and 3 put `receiver_on_record_save()`,
     `receiver_on_group_membership_changed()` and `receiver_on_record_delete()` in
     `solsys_code/observation_projector.py` beside the logic they call; no `signals.py` is
     created. `SolsysCodeConfig.ready()` holds the three `dispatch_uid`-keyed connections and
     nothing else, so the signal-wiring surface is still readable in one place.

2. **Which `ObservationGroup` wins when a record belongs to more than one** — **RESOLVED:
   lowest pk, with an explicit multi-group test.**
   - What we know: The dev DB has no record in more than one group today (34-CONTEXT.md
     code_context); spike 002's `series_for()` picks `ObservationGroup.objects.filter(...).order_by('pk').first()`.
   - What was unclear: Whether "lowest pk" is the right long-term tie-break or just a placeholder
     that happens to never be exercised.
   - Recommendation: Keep "lowest pk" (matches spike 002's proven behaviour) and add an explicit
     test for the multi-group case even though no real record exercises it yet, per D-DE (test
     the untested edge before it becomes a real bug).
   - **RESOLVED — adopted.** Plan 34-01 Task 1 specifies
     `series_group_for(record)` as `ObservationGroup.objects.filter(observation_records=record).order_by('pk').first()`,
     and Task 2's behaviour list carries the explicit case "a record in two groups links the
     lowest-pk group" as a committed test even though no real record exercises it yet.

3. **Whether the receiver needs a silencing mechanism for bulk test fixtures beyond `raw=True`** —
   **RESOLVED: no silencing mechanism beyond `raw=True`.**
   - What we know: `raw=True` already covers `loaddata`; CONTEXT.md leaves a settings-flag/
     context-manager option to Claude's discretion.
   - What was unclear: Whether the new projector/sweep test suite will create enough
     `ObservationRecord` fixtures via factories (not `loaddata`) that the receiver's per-save
     work meaningfully slows the test suite or creates test-isolation issues (e.g., events
     leaking between test cases via a receiver writing to the DB during `setUp`).
   - Recommendation: Start without a silencing mechanism (the receiver is cheap and DB-writes in
     tests are normal Django practice); revisit only if test runtime or isolation becomes a
     measured problem during Wave 1 execution.
   - **RESOLVED — adopted.** Plan 34-01 Task 1 gives `receiver_on_record_save()` exactly two
     early returns — `raw` true, and a facility outside `PROJECTED_FACILITIES` — and no settings
     flag or context manager is planned anywhere in the phase. Plan 34-01 Task 3 carries the
     `raw=True` fixture-load test that proves the one mechanism works. If test runtime or
     isolation does become a measured problem during Wave 1, that is a new finding for the
     executor to raise, not a decision reopened here.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| SQLite (dev DB) | ObservationRecord/CalendarEvent/CalendarEventMeta storage | Yes (project default) | — | — |
| Django test runner (`python manage.py test`) | All new unit/integration tests for the projector, receivers, and sweep | Yes | Django 5.2.17 | — |
| LCO/SOAR portal network access | Sweep's one-time observed-site lookup (`resolve_placement_block`), and any test that does not mock `make_request` | Not verifiable in this research session (no network probe attempted) | — | Tests must mock `solsys_code.calendar_utils.make_request` exactly as `test_sync_lco_observation_calendar.py` already does — this is an existing, proven pattern, not a new one |
| `python manage.py test` full-suite run excluding `TestEphemeris` | Verifying no regression across the touched modules | Yes, per prior-session notes (test_command in config.json already excludes `test_views.py`) | — | — |

**Missing dependencies with no fallback:** none identified — the phase's one external-network
dependency (the observed-site lookup) already has a proven mock-based test fallback in the
codebase.

**Missing dependencies with fallback:** LCO/SOAR portal reachability during test runs — covered
by mocking `make_request`, per existing test conventions.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`django.test.TestCase`), run via `python manage.py test` [VERIFIED: CLAUDE.md "Testing" section and `.planning/config.json` `workflow.test_command`] |
| Config file | none — Django's own test runner; `pytest`'s `pyproject.toml` config is explicitly legacy and does not collect these tests (CLAUDE.md) |
| Quick run command | `python manage.py test solsys_code.tests.test_observation_projector` (new, per-module, once created) |
| Full suite command | `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py \| grep -v "tests/test_views\.py$" \| sed "s|/|.|g; s|\.py\$||" \| tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` [VERIFIED: `.planning/config.json` `workflow.test_command`] |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PROJ-01 | One event per record, keyed by facility URL, create/update/unchanged | unit | `python manage.py test solsys_code.tests.test_observation_projector` | ❌ Wave 0 |
| PROJ-02 | Span follows record stage (window/placed/observed) | unit | same module, `TestStageFor`/`TestEventFieldsFor` classes | ❌ Wave 0 |
| PROJ-03 | Terminal-negative record keeps a marked event | unit | same module | ❌ Wave 0 |
| PROJ-04 (title stem) | Series stem in title, no group text elsewhere | unit + template-tag test | `test_observation_projector.py` + `test_calendar_display_extras.py` (extend existing) | ⚠️ existing file extended, new assertions Wave 0 |
| PROJ-05 | No-churn re-projection; namespace isolation | integration | `test_project_observation_calendar.py` — run sweep twice, assert `RUN:`/`GEM:`/blank-url counts unchanged | ❌ Wave 0 |
| PROJ-06 | Title fits `truncatechars:16/18` | unit | assert marker+token always within first 16 chars for realistic target-name lengths | ❌ Wave 0 |
| TRIG-01 | `post_save` fires on schedule-only save and `updatestatus` path | integration | `test_observation_projector_signals.py`, ported from spike 001b scenarios S1/S4 | ❌ Wave 0 |
| TRIG-02 | Receiver never raises, no write on no-op save | unit | same module — force an exception inside `event_fields_for` via a monkeypatch, assert record save still succeeds | ❌ Wave 0 |
| TRIG-03 | Sweep `--dry-run`, per-record isolation | integration | `test_project_observation_calendar.py` | ❌ Wave 0 |
| SCHED-06 | Live narrowing over real nights | manual-only (notebook re-execution) | N/A — proven via committed, re-executed notebook per D-20, not a `manage.py test` assertion | N/A |
| ANNOT-03 | Old command gone, behaviours migrated | unit/integration | delete `test_sync_lco_observation_calendar.py`; assert its 38 tests' behaviours are covered by the new test files | ❌ Wave 0 (deletion + migration) |

### Sampling Rate
- **Per task commit:** `python manage.py test solsys_code.tests.test_observation_projector solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_project_observation_calendar`
- **Per wave merge:** the full-suite command above (excludes `test_views.py`'s SPICE-kernel-triggering `TestEphemeris`, per prior-session notes)
- **Phase gate:** Full suite green, plus `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files`, before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_observation_projector.py` — covers PROJ-01..06 (stage classifier, event-fields builder, title builder)
- [ ] `solsys_code/tests/test_observation_projector_signals.py` — covers TRIG-01/02 (post_save/m2m_changed/pre_delete receiver contracts, ported from spike 001b's 6 scenarios)
- [ ] `solsys_code/tests/test_project_observation_calendar.py` — covers TRIG-03/PROJ-05 (sweep command, `--dry-run`, namespace isolation, migrated from the 38 `test_sync_lco_observation_calendar.py` tests)
- [ ] Extend `solsys_code/tests/test_calendar_display_extras.py` — new marker vocabulary in `status_border_css`/`_TERMINAL_PREFIXES`
- [ ] `docs/notebooks/pre_executed/` — new sweep demo notebook (takeover diff + SCHED-06 live-narrowing section, per D-19/D-20) — not a `manage.py test` gap, but a Wave 0/paired-docs gap the plan must schedule
- [ ] Framework install: none — Django TestCase is already the project's test framework

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | This phase adds no auth surface — receivers and a management command, no new view/endpoint |
| V3 Session Management | No | No session-touching code |
| V4 Access Control | No | The sweep is a management command (staff/operator shell access only); no new URL/view is added |
| V5 Input Validation | Yes | `--proposal`/`--facility` sweep arguments must reject unexpected values gracefully (mirror `_parse_proposal_arg`'s existing dedup/strip logic [VERIFIED: solsys_code/management/commands/sync_lco_observation_calendar.py:227-249]); `record.parameters` values used to build titles/descriptions must not allow HTML/script injection into calendar templates — Django's template auto-escaping already covers this for `calendar.html`/`event_form.html` |
| V6 Cryptography | No | No credential handling added; the sweep reuses `LCOFacility()`'s existing `_portal_headers()` auth construction unchanged |

### Known Threat Patterns for Django signal receivers + management commands

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Credential leakage via exception logging | Information Disclosure | Never interpolate a caught `requests`/portal exception into a log line (SYNC-09/D-11 discipline, Pitfall 5 above); `resolve_placement_block()` already follows this — the projector/sweep must not regress it |
| Denial of service via slow receiver blocking every save | Denial of Service | TRIG-02's "no network call, no `sun_event` scan" hot-path rule keeps the `post_save` receiver's cost bounded and independent of external service latency |
| Namespace confusion / unauthorized event mutation | Tampering | The projector must filter strictly on `url == facility.get_observation_url(...)` before writing or deleting; spike 002's isolation check (`RUN:` count unchanged before/after) is the regression test that catches a namespace leak |
| Unbounded log growth from per-save logging at INFO level on every request in production | Denial of Service (resource exhaustion) | Keep per-save logging at `debug`/`info` per CLAUDE.md's existing logging conventions, not `warning`/`error` for the normal no-churn case |

## Sources

### Primary (HIGH confidence)
- Read this session, verbatim quotes included above: `solsys_code/calendar_utils.py`,
  `solsys_code/models.py`, `solsys_code/apps.py`,
  `solsys_code/management/commands/sync_lco_observation_calendar.py`,
  `solsys_code/management/commands/backfill_lco_observations.py` (lines 640-700),
  `solsys_code/templatetags/calendar_display_extras.py`,
  `src/templates/tom_calendar/partials/calendar.html` (lines 241-285),
  `src/templates/tom_calendar/partials/event_form.html` (lines 100-170),
  `src/fomo/settings.py` (HOOKS block, lines 372-378), `.planning/config.json`
- Read this session, installed package source (site-packages, confirmed paths in output):
  `tom_observations/models.py` (`ObservationRecord.save()`, `ObservationGroup.observation_records`),
  `tom_observations/facility.py` (`get_service_class`, `update_observation_status`,
  `update_all_observation_statuses`), `tom_observations/facilities/ocs.py`
  (`OCSFacility.get_observation_url`, `get_terminal_observing_states`, `get_failed_observing_states`),
  `tom_observations/facilities/soar.py` (`SOARFacility(LCOFacility)`), `tom_calendar/models.py`
  (`CalendarEvent` field definitions)
- Spike source files (`sources/001-b-trigger-django-post-save/spike.py`,
  `sources/002-observation-projector/projector.py`, `sources/002-observation-projector/sweep.py`,
  `sources/004-live-narrowing-updatestatus/recheck.py`) — read in full this session, all
  VALIDATED or PARTIAL (winner/documented) per the skill's Processed Spikes metadata
- `.claude/skills/spike-findings-fomo_devel/references/event-trigger.md`,
  `references/observation-projector.md`, `references/allocation-handoff.md` — synthesized
  findings citing tomtoolkit 3.0.1 source line numbers, cross-checked against the installed
  package this session

### Secondary (MEDIUM confidence)
- `.planning/notes/observation-first-calendar-layering.md` — D1-D5 design decisions and the
  "research findings (admitted, with sources)" block, itself citing tomtoolkit source
- `.planning/phases/34-the-observation-projector-trigger/34-CONTEXT.md` code_context section
  (dev DB baseline counts — not independently re-queried this session, see Assumption A1)

### Tertiary (LOW confidence)
- None — no unverified WebSearch-only claims were needed for this phase; every technical claim
  traces to a file read in this session or a prior spike's forensic log.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; both `tomtoolkit` and `django` versions confirmed live this session
- Architecture: HIGH — every pattern cited traces to either a validated spike run against 146 real records or an existing, currently-shipping FOMO module read this session
- Pitfalls: HIGH — all six pitfalls trace to a specific spike scenario result or a line-numbered read of installed/project source, not general Django folklore

**Research date:** 2026-09-10
**Valid until:** 2026-10-10 (30 days — stable, in-repo/installed-package domain; re-verify sooner
only if `tomtoolkit` or `django` are upgraded before Phase 34 executes)
</content>
