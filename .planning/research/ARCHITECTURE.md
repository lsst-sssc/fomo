# Architecture Research — v2.3 Automatic Run Sync & Outcome Propagation

**Domain:** Unattended scheduling + adapter rewiring + outcome propagation, integrated into an
existing Django/TOM Toolkit app (FOMO) with a pure-logic reconciler already in production.
**Researched:** 2026-09-01
**Confidence:** HIGH (grounded in the actual v2.2 codebase — `campaign_reconciler.py`,
`campaign_utils.py`, `models.py`, the three adapter commands, `reconcile_campaign_runs.py` — and
in the Phase 26 spike's own settled findings, `26-DECISION.md`/`canonical_record_spike.rst`,
which already answered the adapter-identity-key question this milestone must build on).

## Standard Architecture

### System Overview — today (v2.2, shipped) vs. target (v2.3)

```
TODAY (v2.2)
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ load_telescope_runs (file)     │   │ sync_lco_observation_calendar  │
│ sync_gemini_observation_calendar│   │ (ObservationRecord -> LCO/SOAR)│
└───────────────┬────────────────┘   └───────────────┬────────────────┘
                │ insert_or_create_calendar_event()   │
                ▼                                     ▼
                        tom_calendar.CalendarEvent
                                    │
                     CalendarEventMeta (run FK, mostly unset for these adapters)
                                    │
                     Staff actions (approve/resolve_site/mark_cancelled/
                     mark_weather_failure) on CampaignRun --------------> reconcile_run()
                     (only web-submitted / CSV-imported CampaignRuns exist today;
                      the three sync adapters never create one)

TARGET (v2.3)
┌──────────────┐ ┌─────────────────────┐ ┌────────────────────────────┐
│ Scheduler      │ │ Discovery sweep      │ │ (existing) update_all_     │
│ entry point    │→│ (watch-list of       │→│ statuses-style polling     │
│ (cron/task-    │ │ proposals -> new     │ │ keeps ObservationRecord    │
│ queue, spike-  │ │ ObservationRecords)  │ │ status/scheduled_* fresh   │
│ settled)       │ └─────────────────────┘ └──────────────┬─────────────┘
│                │                                         │
│                │  ┌──────────────────────────────────────┴────────┐
│                │→ │ 3 adapters (load_telescope_runs, sync_lco_*,   │
│                │  │ sync_gemini_*) — REWIRED (ADAPT-01..03)         │
│                │  │  each: build (lookup, fields) -> CampaignRun    │
│                │  │  via a shared write_and_reconcile_campaign_run()│
│                │  │  helper -> reconcile_run(run) inline, per run   │
│                │  └──────────────────────┬───────────────────────┘
│                │                          ▼
│                │              CampaignRun (create/update)
│                │                          │
│                │              reconcile_run() (pure fn of run state)
│                │                          │
│                │                          ▼
│                │              tom_calendar.CalendarEvent (RUN: keyed)
│                │
│                │  ┌──────────────────────────────────────────────┐
│                │→ │ reconcile_campaign_runs --dry-run/real         │
│                │  │ (safety-net sweep: catches drift, admin edits, │
│                │  │  a per-record reconcile that raised/failed)    │
│                │  └──────────────────────────────────────────────┘
│                │
│                │  ┌──────────────────────────────────────────────┐
│                │→ │ NEW: outcome propagation step                 │
│                │  │  derive_run_status(run) reads confirmed        │
│                │  │  CampaignRunObservation -> ObservationRecord    │
│                │  │  status; writes run.run_status if changed;      │
│                │  │  THEN calls reconcile_run(run) (never inline    │
│                │  │  inside reconcile_run itself)                   │
│                │  └──────────────────────────────────────────────┘
└──────────────┘
       │  failure isolation per step, per run; mail_admins()/logger.error()
       ▼  on any hard failure so it's visible, not silent
```

### Component Responsibilities

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `campaign_reconciler.reconcile_run()` | Pure, idempotent projection of ONE `CampaignRun`'s current state onto its owned `CalendarEvent`(s) | Existing (v2.2), unchanged in shape — must stay a pure reader of run state |
| `reconcile_campaign_runs` command | Sweeps every `CampaignRun` through `reconcile_run()`, per-run failure isolation, `--dry-run` | Existing (v2.2) — becomes the scheduler's safety-net step, not its primary write path |
| `campaign_utils.insert_or_create_campaign_run()` | No-churn create-or-update for one `CampaignRun` against a caller-supplied natural-key lookup | Existing (v2.2, used today only by `import_campaign_csv`) — becomes the base every adapter calls |
| **NEW** `campaign_utils.write_and_reconcile_campaign_run()` | Thin composition: `insert_or_create_campaign_run()` then `reconcile_run(run)` for that one run, returning both results | New shared helper — the thing that stops ADAPT-01..03 from tripling the same three-line pattern |
| `load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar` | Parse/query a source, build `(lookup, fields)` | REWIRED (ADAPT-01..03): target model changes from `CalendarEvent` to `CampaignRun`; parsing/selection logic unchanged |
| `backfill_lco_observation_records` | One-shot backfill of `ObservationRecord`s for a single `--proposal`/`--campaign`/`--name-prefix` | Existing — becomes the *template* for the new discovery sweep, not itself rewired (still useful as a manual one-off tool) |
| **NEW** discovery-sweep command | Loops a configured watch-list of proposals, calling the same RequestGroups-fetch logic `backfill_lco_observation_records` already has, with zero per-invocation arguments | New — generalizes `backfill_lco_observation_records`'s query logic, does not replace it |
| **NEW** watch-list config surface | Where the set of "proposals to sweep" lives | New — see Pattern 2 below for the settings-vs-model tradeoff |
| **NEW** scheduler entry-point command | One process a cron job / task queue invokes; orchestrates discovery -> 3 adapters -> reconcile sweep -> outcome propagation, in that order, with failure isolation per step | New |
| **NEW** `derive_run_status()` (name TBD, likely `campaign_reconciler.py` or a sibling `campaign_outcomes.py`) | Pure function: given a `CampaignRun`'s confirmed `CampaignRunObservation` links, compute the `RunStatus` its linked observations imply | New — must NOT live inside `reconcile_run()` (see Anti-Patterns) |
| **NEW** outcome-propagation write step | Compares `derive_run_status(run)` to `run.run_status`; on a difference, writes it (mirroring `_set_run_status()`'s existing guarded-update shape) and then calls `reconcile_run(run)` | New |

## Critical Integration Risk — `CampaignRun.campaign` is `NOT NULL`, and most synced observations have no campaign

This is the single biggest architectural fork this milestone must resolve, and it is not
optional detail — it decides whether ADAPT-01..03 is even a straightforward field-mapping
exercise or requires its own schema/spike phase first.

`CampaignRun.campaign` is `models.ForeignKey(TargetList, on_delete=models.PROTECT, null=False, ...)`
(`solsys_code/models.py:167-173`). Both of `CampaignRun`'s existing natural-key
`UniqueConstraint`s (`unique_campaign_run_resolved_window`,
the TBD-branch constraint) are scoped by `campaign` (`solsys_code/models.py:288-299` and the TBD
constraint that follows it). Every existing writer of a `CampaignRun` — the web submission form,
`import_campaign_csv`, the CSV natural key — supplies a real campaign.

But `sync_lco_observation_calendar` and `sync_gemini_observation_calendar` today sync **every**
matching `ObservationRecord`, regardless of whether its `Target` belongs to any campaign
`TargetList` at all (`target_list = record.target.targetlist_set.order_by('name').first()`,
`sync_lco_observation_calendar.py:199` — already nullable and frequently `None` today, e.g. for
routine NEO follow-up that isn't part of a 3I/ATLAS-style community campaign).
`load_telescope_runs`'s `--campaign` flag is optional and commonly omitted
(`load_telescope_runs.py:116-123`, `_resolve_campaign()` returns `None` when omitted). If
ADAPT-01..03 requires every one of these to become a `CampaignRun`, then either:

1. **Every routine, non-campaign sync gets forced into a synthetic/placeholder `TargetList`**
   just to satisfy `NOT NULL` — this pollutes the campaign list UI (`CampaignRunTableView`,
   the campaigns navbar) with fake "campaigns" that are really just routine facility syncs, and
   stretches `CampaignRun`'s documented meaning ("a single target-linked observing run within a
   coordination campaign", `models.py:78-87`) past what it was designed for. **Not recommended.**
2. **`CampaignRun.campaign` becomes nullable.** This is the semantically honest option, and it
   is also what v2.2's own stated goal already implies: "An observing run exists once, as a
   `CampaignRun`, and everything else is derived from it" (PROJECT.md, v2.2 Core Value) is a
   claim about *all* observing runs, not just community-coordinated ones. Making `campaign`
   nullable is what actually finishes that claim.

Option 2 is the right direction, but it is a real schema change with a real consequence: a
`NULL` FK is never considered equal to another `NULL` by either backend's `UniqueConstraint`
(same trap Phase 19 already hit and solved for `window_start IS NULL` — see the TBD-branch
constraint's own comment, `models.py:293-298`). If `campaign` can be `NULL`, the two existing
natural-key branches do not cover a non-campaign, adapter-created run at all — get_or_create
against a lookup containing `campaign=None` would create a fresh, duplicate `CampaignRun` on
every single re-sync, defeating idempotency (SYNC-04's / GEM-NOCHURN-01's whole point).

**Recommendation: this needs its own identity field, not a stretch of the existing natural
keys.** Per `26-DECISION.md`'s own Criterion-2 findings (already settled, not new research):
the true identity for an LCO-queue-sourced run is the LCO portal request `url`
(`sync_lco_observation_calendar.py:361`); for Gemini it is `GEM:{prog}/{obsid}`
(`sync_gemini_observation_calendar.py:150`); for classical runs it is
`(telescope, instrument, start_time)` with a 5-minute tolerance and no `url` at all
(`load_telescope_runs.py:22`,`207-216`). None of these map cleanly onto
`(campaign, telescope_instrument, window_start, window_end)` when `campaign` is absent — a
`telescope_instrument`+window pair is not unique across an entire semester's worth of
class-wide queue traffic the way `campaign`+that pair was for community-coordinated runs.

The cleanest fix, consistent with how this codebase has always resolved exactly this class of
question (Phase 18's TBD natural key, Phase 26's `source`/`telescope_class` spike): **a
phase-time investigation spike, before any adapter is rewired**, that settles (a) whether
`campaign` becomes nullable, (b) a new identity field on `CampaignRun` (e.g.
`source_identifier`, populated with the adapter's own natural key — the LCO url, the Gemini
`GEM:` key, or the classical `(telescope, instrument, start_time)` tuple serialized somehow) and
its `UniqueConstraint` shape scoped by `source` instead of `campaign` for these rows, and (c)
whether the classical adapter's lack of any string identity today (D-19's finding) is acceptable
to carry forward unchanged (matching on `telescope_instrument`+window the way it already does
for classical `CalendarEvent`s) or needs its own new key. **This is exactly the shape of
question Phase 26 already answered for a sibling problem — reuse that pattern, don't reinvent
it inline inside an adapter-rewiring phase.**

## Architectural Patterns

### Pattern 1: Shared "write CampaignRun, then reconcile" helper (avoid triplicating the pattern)

**What:** One function, `write_and_reconcile_campaign_run(lookup, fields) -> tuple[CampaignRun,
str, ReconcileResult]`, added beside `insert_or_create_campaign_run()` in `campaign_utils.py`.
It calls `insert_or_create_campaign_run(lookup, fields)` (existing, unchanged) and then
`campaign_reconciler.reconcile_run(run)` (existing, unchanged) on the same run, returning both
outcomes so the calling command can report create/update/unchanged **and**
created/updated/blocked/skipped-reason in its own summary line — exactly the granularity every
existing adapter's summary line already reports for `CalendarEvent` writes.

**When to use:** Every one of the three rewired adapters' per-record write site. This is the
direct replacement for today's `insert_or_create_calendar_event({...}, fields)` call
(`load_telescope_runs.py:207`, `sync_lco_observation_calendar.py:341`,
`sync_gemini_observation_calendar.py:163`).

**Why not call `insert_or_create_campaign_run()` and `reconcile_run()` separately at each of
the three call sites:** it is the exact pattern this codebase already flagged as worth
extracting once before (`calendar_utils.insert_or_create_calendar_event()` itself was extracted
in Phase 11 — REFAC-01/02 — specifically because it had been copy-pasted across the same three
commands). Skipping the extraction this time would reproduce the identical tech-debt shape
Phase 11 already had to clean up once.

**Trade-off:** a per-record `reconcile_run()` call inside a tight loop over potentially hundreds
of `ObservationRecord`s (e.g. `sync_lco_observation_calendar`'s full-proposal sweep) does more
DB work per record than today's single `insert_or_create_calendar_event()` call — each
`reconcile_run()` call does its own `CalendarEvent`/`CalendarEventMeta` lookups
(`_may_write()`, `_detach_stale_family_events()`) on top of the `CampaignRun` write. This is
the same cost the existing staff-action call sites already pay per single run
(`campaign_views.py:526,680,758`), just now paid at adapter-sync scale. Acceptable for the
current dev-DB scale (dozens of runs); worth a query-count regression test if the LCO/SOAR
sync volume grows materially.

**Example (illustrative, not literal code to be copied verbatim):**
```python
# campaign_utils.py
def write_and_reconcile_campaign_run(
    lookup: dict[str, Any], fields: dict[str, Any]
) -> tuple[CampaignRun, str, ReconcileResult]:
    """Create/update one CampaignRun, then reconcile its calendar projection.

    The one call site every ADAPT-01..03 adapter should use instead of writing a
    CalendarEvent directly -- composes the two existing pure building blocks
    (insert_or_create_campaign_run, campaign_reconciler.reconcile_run) rather than
    duplicating either.
    """
    from solsys_code.campaign_reconciler import reconcile_run  # local import: avoids

    run, action = insert_or_create_campaign_run(lookup, fields)
    result = reconcile_run(run)
    return run, action, result
```

### Pattern 2: Watch-list config surface for the discovery sweep

**What:** `backfill_lco_observation_records` today requires `--proposal`, `--name-prefix`, and
(interactively, if omitted) `--campaign` per invocation — the opposite of "no per-invocation
arguments." The new discovery-sweep command needs a durable, operator-editable list of
`(proposal, name_prefix, campaign)` tuples to loop over unattended.

**Two real options, both consistent with this codebase's existing conventions:**

- **Django settings entry** (e.g. `FOMO_WATCHED_PROPOSALS = [{'proposal': 'LTP2025A-004',
  'name_prefix': '...', 'campaign': None}, ...]` in `local_settings.py`), matching the existing
  `FACILITIES`/`DATA_SERVICES`/`ALERT_STREAMS` pattern already in `src/fomo/settings.py`. No
  migration, no admin UI, editable only by whoever can redeploy/restart.
- **A small DB-backed model** (e.g. `WatchedProposal`), editable via the Django admin without a
  redeploy — closer to how `Observatory` and `CampaignRun` are already managed, and a better
  fit if the set of watched proposals changes often or needs to be edited by non-engineering
  staff.

**Recommendation:** start with the settings-entry form — it is strictly less work, matches the
existing `FACILITIES`-style precedent for "external-service configuration", and the watch-list
is expected to change rarely (new proposal cycles, not daily). Treat the DB-backed model as the
natural upgrade path if that assumption breaks (mirrors this project's own "difflib until
`rapidfuzz` proves necessary" discipline from Phase 18 — do not build the heavier option
speculatively).

### Pattern 3: Keep `derive_run_status()` a pure sibling of `reconcile_run()`, never inside it

**What:** RECON-01..09 lock `reconcile_run()` as a pure, idempotent function of a
`CampaignRun`'s *own* fields (`approval_status`, `window_start`/`end`, `site`,
`telescope_class`, `run_status` only insofar as it picks a title prefix — never insofar as it
*writes* it). Outcome propagation needs to read a *different* model's state
(`CampaignRunObservation` -> `ObservationRecord.status`) and, on a change, *write*
`CampaignRun.run_status`. Folding that read-and-write into `reconcile_run()` itself would turn
it from "project calendar events from run state" into "also mutate run state from a different
model's state, then project" — two responsibilities the module's own docstring already
separates (`campaign_reconciler.py:1-38`: this module owns *projection*, not truth derivation).

**When to use:** exactly the same shape `campaign_views._set_run_status()`
(`campaign_views.py:754-758`) already uses for the staff `mark_cancelled`/`mark_weather_failure`
actions: **mutate `run_status` first** (guarded — see below), **then call `reconcile_run(run)`**
to let the (unchanged) reconciler pick up the new `run_status` for its title/description
prefix. `reconcile_run()` never needs to know *why* `run_status` changed — staff click, or
automatic derivation — only that it changed.

**Concretely, a new function** (a natural home is `campaign_reconciler.py`, since it is the
sibling of `event_title()`/`event_description()` which already read `run.run_status`, or a new
`campaign_outcomes.py` module if keeping `campaign_reconciler.py` scoped strictly to projection
is preferred):

```python
def derive_run_status(run: CampaignRun) -> CampaignRun.RunStatus | None:
    """Pure function: what run_status this run's CONFIRMED observation links imply.

    Reads run.observation_links (CampaignRunObservation, CANON-04) -> their
    ObservationRecord.status. Returns None when there is no confirmed link yet (this
    run's run_status must not be touched -- see the build-order note below), or when
    the linked records' statuses don't yet imply a terminal state.
    """
```

The write step (a new function, or inline in the propagation command) mirrors
`_set_run_status()`'s existing guarded-conditional-update shape (`campaign_views.py:...`,
`_resolve_site()`'s sibling): re-check the run's current `run_status` in the same transaction as
the write (`CampaignRun.objects.filter(pk=run.pk, run_status=old).update(run_status=new)`),
short-circuit if the conditional update affected zero rows (another process already changed it),
`refresh_from_db()`, then call `reconcile_run(run)`.

**Trade-off:** this means outcome propagation runs as its own pass over `CampaignRun`s with
confirmed observation links, separate from (and after) the reconcile sweep — one more step in
the scheduler pipeline, not a free side effect of reconciliation. That is the correct trade for
keeping RECON-01 ("running it a second time changes nothing") true of `reconcile_run()` in
isolation, independent of whether outcome propagation has run yet this cycle.

## Data Flow

### Today (v2.2)

```
[cron/manual] -> load_telescope_runs file
                  -> insert_or_create_calendar_event() -> CalendarEvent (blank url)

[cron/manual] -> sync_lco_observation_calendar --proposal X
                  -> insert_or_create_calendar_event({'url': lco_url}, fields)
                  -> CalendarEvent (LCO-url-keyed) + CalendarEventMeta.is_verified

[cron/manual] -> sync_gemini_observation_calendar
                  -> insert_or_create_calendar_event({'url': 'GEM:...'}, fields)
                  -> CalendarEvent (GEM-url-keyed)

[staff click] -> approve / resolve_site / mark_cancelled / mark_weather_failure
                  -> mutate CampaignRun -> reconcile_run(run) -> CalendarEvent (RUN:-keyed)
                  (only for web-submitted / CSV-imported CampaignRuns -- the sync commands
                   above never touch CampaignRun at all today)
```

### Target (v2.3)

```
[scheduler entry point, cron or task queue -- settled by the phase-time spike]
  1. discovery sweep (watch-list of proposals, zero args)
     -> LCO RequestGroups API -> new ObservationRecord rows (mirrors
        backfill_lco_observation_records's existing per-request logic, generalized to loop)
  2. (existing, unchanged) ObservationRecord status polling keeps status/scheduled_*
     fresh for every record the discovery sweep or normal TOM submission created
  3. load_telescope_runs (classical file, if configured) --REWIRED--
     -> per parsed run-line: write_and_reconcile_campaign_run(lookup, fields)
        -> CampaignRun (create/update) -> reconcile_run(run) -> CalendarEvent (RUN:-keyed,
           per-night, adopting a pre-existing blank-url classical event where one exists)
  4. sync_lco_observation_calendar --REWIRED--
     -> per matching ObservationRecord: write_and_reconcile_campaign_run(lookup, fields)
        -> CampaignRun (create/update, source=LCO_QUEUE) -> reconcile_run(run)
           -> CalendarEvent (bare RUN:{pk} container, since queue runs are site/class-wide
              per the settled Phase 26 verdict -- the record's OWN CalendarEvent, produced
              by this same command's existing per-record logic, still exists separately
              and still supplies per-night/per-observation detail)
  5. sync_gemini_observation_calendar --REWIRED-- (same shape as 4, source=GEMINI_QUEUE)
  6. reconcile_campaign_runs --dry-run (log-only) then real sweep
     -> safety net: catches any CampaignRun whose per-record reconcile_run() call in
        steps 3-5 raised and was skip-logged, or whose state changed out-of-band
        (admin edit, migration, manual staff action between scheduler runs)
  7. NEW: outcome propagation pass
     -> for every CampaignRun with >=1 confirmed CampaignRunObservation link:
        derive_run_status(run); if different from run.run_status, guarded-update it,
        then reconcile_run(run) again (so the calendar title/description refresh)
  -> failure at any step: logged (logger.error) + mail_admins()-style notification,
     mirroring campaign_views._notify_staff()'s existing "email every staff user"
     pattern from Phase 16 -- never silently swallowed between runs
```

### Key Data Flows

1. **Adapter write path (the core rewiring):** parse/query source data -> build
   `(lookup, fields)` for `CampaignRun` (not `CalendarEvent`) -> `write_and_reconcile_campaign_run()`
   -> `CampaignRun` row -> `reconcile_run()` -> `CalendarEvent`. The adapter never touches
   `CalendarEvent` directly again; `campaign_reconciler.py` is the only writer of `RUN:`-namespaced
   events, matching the ownership model `_may_write()` already enforces.
2. **Outcome propagation (new, separate pass):** `ObservationRecord.status` (kept fresh by
   existing polling) -> `CampaignRunObservation` (existence = linkage, confirmed either by staff
   via Phase 28's attribution queue, or — new in v2.3 — automatically when an adapter creates
   the `CampaignRun` FROM the exact `ObservationRecord` it is syncing, since that identity is
   certain, not a scored candidate) -> `derive_run_status()` -> guarded `CampaignRun.run_status`
   write -> `reconcile_run()` refresh.
3. **Discovery sweep (new, upstream of everything else):** watch-list config -> LCO
   RequestGroups API -> new `ObservationRecord` rows, which then flow into data flow 1 on the
   very same scheduler pass (the sweep must run *before* the adapters in the same invocation, or
   newly discovered records wait a full cycle before they get a `CampaignRun`/calendar
   presence).

## Anti-Patterns

### Anti-Pattern 1: Folding `run_status` derivation into `reconcile_run()`

**What people would do:** add an `ObservationRecord`-status read directly inside
`_reconcile_container()`/`_reconcile_classical_nights()` or `reconcile_run()` itself, since it's
already iterating the run and already has `run.run_status` in scope for the title prefix.
**Why it's wrong:** breaks the module's own stated boundary (a pure projector of the run's *own*
state) and makes `reconcile_run()`'s idempotency depend on a second model's state changing
between calls in ways RECON-01's contract never accounted for — two callers of `reconcile_run()`
in the same process (a staff action and the scheduler's outcome-propagation pass) could now see
it silently mutate `run_status` as a side effect of what looks like a read-only projection call.
**Do this instead:** a separate `derive_run_status()` + guarded-write step, always called
*before* `reconcile_run()`, per Pattern 3 above.

### Anti-Pattern 2: Treating "an adapter's own record ⇒ automatic `CampaignRunObservation` link" the same as Phase 28's scored attribution

**What people would do:** when `sync_lco_observation_calendar` creates a `CampaignRun` FROM a
specific `ObservationRecord`, write the `CampaignRunObservation` link the same way Phase 28's
`AttributionDecisionView` does — i.e., go through the scored-candidate matcher
(`campaign_attribution.py`) even though the identity is already certain (the adapter created
the run because of that exact record; there is no ambiguity to score).
**Why it's wrong:** `campaign_attribution.py`'s weighted-sum scoring exists specifically for
*ambiguous, pre-existing* pairs where identity must be inferred (ATRIB-01..06's whole premise).
Running an already-certain pairing through it wastes the scoring machinery and — worse — could
theoretically land below the confirmation threshold for a legitimate, exact pairing if the
scoring weights are ever tuned against the *ambiguous* case.
**Do this instead:** the adapter writes the `CampaignRunObservation` link directly (a plain
`get_or_create`), leaving `confirmed_by`/`confirmed_at` unset — that pair is already nullable
(`models.py`, `CampaignRunObservation.confirmed_by`: `on_delete=SET_NULL, null=True,
blank=True`), so "system-linked, exact identity, no human involved" is already a representable
state, not a schema gap. **Update the CANON-04 docstring's stated invariant** ("row's existence
already means a staff member confirmed this", `models.py:399-404`) when this ships — it becomes
"row's existence means either a staff member confirmed it, or the identity was certain at
creation time" — a comment-accuracy task, not a structural one, but a real one to not skip
(this codebase has a documented history — Phase 28's own 28-05/28-06 gap closures — of an
under-specified confirmation invariant causing a real BLOCKER bug).

### Anti-Pattern 3: A discovery-sweep or scheduler command that swallows a per-item exception silently

**What people would do:** wrap the whole scheduler pass in one broad `try/except: pass` so a
single bad `ObservationRecord`/`CampaignRun` doesn't crash the cron job.
**Why it's wrong:** this is exactly the "failure disappears between runs" outcome this
milestone's own target features explicitly reject ("a failure is visible to an operator rather
than silently disappearing between runs").
**Do this instead:** the existing per-run failure-isolation pattern in
`reconcile_campaign_runs.py:56-62` — a bare `except Exception` **per item**, logged at `debug`
(never interpolating raw exception content that might carry PII/credentials, matching
`sync_gemini_observation_calendar.py`'s `GEM-SECURE-01` discipline), a per-item stderr line, and
a running failure count — plus a **step-level** failure/summary email via the existing
`_notify_staff()`-style "email every staff user" mechanism (`campaign_views.py`, Phase 16) so a
whole step failing (not just one item) actually reaches a human.

### Anti-Pattern 4: Rewiring the three adapters before the campaign-nullability/identity-key question is settled

**What people would do:** start ADAPT-01 (classical adapter) as a "simple" field-mapping task
since its shape (a window range, no `url`) looks closest to the existing natural key, and defer
the LCO/Gemini nullability question to "whichever adapter hits it first."
**Why it's wrong:** the *classical* adapter is the one case that fits today's natural key
cleanly (see Critical Integration Risk above) — starting there risks shipping a working-looking
Phase 1 that then requires a schema change mid-milestone once LCO/Gemini expose the real
`campaign`-nullability problem, exactly the "found late" pattern this codebase's own audit trail
(Phase 19's CR-01, Phase 21's clobbering bug) shows is expensive to unwind after data exists.
**Do this instead:** settle the schema/identity-key question for ALL THREE adapters first (a
single investigation spike), then rewire — see Build Order below.

## Integration Points

### Files that change

| File | Change |
|------|--------|
| `solsys_code/models.py` | `CampaignRun.campaign` likely becomes nullable (pending the spike); a new identity field for queue-sourced runs (e.g. `source_identifier`); a new/adjusted `UniqueConstraint` scoped by `source` for non-campaign rows |
| `solsys_code/migrations/00NN_*.py` | Schema migration for the above — hand-authored if any rename/backfill is load-bearing (mirrors Phase 27's `RenameModel` discipline) |
| `solsys_code/campaign_utils.py` | New `write_and_reconcile_campaign_run()` beside `insert_or_create_campaign_run()` |
| `solsys_code/campaign_reconciler.py` or new `campaign_outcomes.py` | New `derive_run_status()` (pure) + a guarded write helper mirroring `_set_run_status()`'s shape |
| `solsys_code/management/commands/load_telescope_runs.py` | ADAPT-01: replace `insert_or_create_calendar_event()` call with `write_and_reconcile_campaign_run()`; `--campaign` semantics revisited once nullability is settled |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | ADAPT-02: replace `insert_or_create_calendar_event()` call with `write_and_reconcile_campaign_run()`; also write the automatic `CampaignRunObservation` link (Anti-Pattern 2) |
| `solsys_code/management/commands/sync_gemini_observation_calendar.py` | ADAPT-03: same shape as ADAPT-02 |
| `solsys_code/management/commands/backfill_lco_observation_records.py` | Unchanged — remains the manual one-off tool; its RequestGroups-fetch logic is the template the new discovery-sweep command generalizes, not a file it replaces |
| **NEW** `solsys_code/management/commands/discover_watched_proposals.py` (name TBD) | Loops the watch-list, calls the same RequestGroups logic unattended, zero args |
| **NEW** `solsys_code/management/commands/run_unattended_sync.py` (name TBD) | The scheduler entry point: discovery -> 3 adapters -> reconcile sweep -> outcome propagation, per-step failure isolation |
| `src/fomo/settings.py` / `local_settings.py` | New watch-list config entry (Pattern 2) |
| `solsys_code/campaign_attribution.py` | `orphan_observation_records()`/`_eligible_runs_for_record()` need to account for records now linked automatically at creation time (they should simply no longer appear as orphans — verify, don't assume, once ADAPT-02/03 ship) |
| `solsys_code/campaign_gap.py` | GAPB-01 (carried from v2.2): `claimed_dates()` needs to count nights covered by the queue-container's own `ObservationRecord`-derived events, not only `CampaignRun` window fields — a direct consequence of queue runs no longer producing per-night `CampaignRun` rows (they produce one container + real observation events) |
| `solsys_code/templatetags/calendar_display_extras.py` | UNUSED-01 (carried from v2.2): visually distinguish a queue container's un-realized nights from its realized ones; STATUS-01/02: unify `_TERMINAL_PREFIXES`/`_FAILURE_PREFIX_BY_STATUS`/`_CLASSICAL_STATUS_PREFIX`/`RUN_STATUS_CALENDAR_PREFIX` now that `run_status` drives titles more broadly |
| Paired demo notebooks (CLAUDE.md rule) | `load_telescope_runs_demo.ipynb`, `sync_lco_observation_calendar_demo.ipynb`, `sync_gemini_observation_calendar_demo.ipynb`, `reconcile_campaign_runs_demo.ipynb` all need updates showing the new `CampaignRun`-then-`CalendarEvent` flow; `docs/runbooks/telescope_runs_calendar.rst` needs an unattended-scheduling section |

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| LCO Observation Portal RequestGroups API | Already used by `backfill_lco_observation_records` (`GET /api/requestgroups/`, paginated) | The discovery sweep reuses this exact call shape per watch-list entry; no new API surface |
| Cron / task queue | Unattended invocation of the scheduler entry point | Mechanism explicitly deferred to a phase-time spike (cron vs. Celery/task-queue against real deployment constraints) — architecture above is invocation-mechanism-agnostic: whichever mechanism wins, it calls one process/command |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| 3 adapters ↔ `campaign_utils`/`campaign_reconciler` | Direct function call (`write_and_reconcile_campaign_run()`) | Matches the existing `insert_or_create_calendar_event()` call shape being replaced — same import discipline (`campaign_reconciler.py` still never imports the views module or the heavy ephemeris module, per its own module docstring and CLAUDE.md's project-wide constraint) |
| Scheduler entry point ↔ 3 adapters + discovery + reconcile sweep + outcome propagation | Each invoked as a Django management command (`call_command()` or subprocess, TBD by the spike) | Failure isolation per step, not one giant try/except (Anti-Pattern 3) |
| Outcome propagation ↔ `reconcile_run()` | Sequential: write `run_status`, then call `reconcile_run()` — never the reverse, never merged | Pattern 3 |
| `campaign_attribution.py` ↔ new automatic linking | Read-only for automatically-linked records (they should just not appear as orphans); no write coupling | Anti-Pattern 2 |

## Suggested Build Order

1. **Phase-time investigation spike** (mirrors Phase 18/26): settle `CampaignRun.campaign`
   nullability, the new identity field/natural-key shape for queue-sourced runs, and — per the
   milestone's own stated scope — the cron-vs-task-queue invocation mechanism. **Must come
   first**: every later step's schema and call shape depends on this, and this codebase's own
   history (Phase 19/21/27 CR-01-class findings) shows deferred schema decisions get expensive
   once real data exists under the old shape.
2. **Shared helper** (`write_and_reconcile_campaign_run()` in `campaign_utils.py`) — depends only
   on the spike's schema decision, not on any adapter being rewired yet. Write it, test it in
   isolation against the new schema, before touching any command.
3. **ADAPT-01 (classical adapter)** — the adapter whose existing shape needs the least new
   identity-key machinery (Critical Integration Risk above), so it validates the shared helper
   and the new schema against the simplest real case first.
4. **ADAPT-02/03 (LCO, then Gemini)** — same shape, plus the automatic `CampaignRunObservation`
   linking (Anti-Pattern 2), since these are the adapters that actually have a specific
   `ObservationRecord` to link at creation time (the classical adapter has none — it never
   touches `ObservationRecord` at all).
5. **CampaignRunObservation linkage must be live (step 4) before outcome propagation is built.**
   `derive_run_status()` has nothing to read for an adapter-created run until step 4 ships —
   this is the dependency the quality gate calls out explicitly, and it is real: today,
   `CampaignRunObservation` rows exist only via Phase 28's staff-confirmed attribution queue,
   which was never wired to fire automatically for a run the sync adapters themselves just
   created.
6. **Outcome propagation** (`derive_run_status()` + guarded write + `reconcile_run()` call) —
   after step 5, since it needs real confirmed links to read.
7. **Discovery sweep + watch-list config** — can be built in parallel with steps 3-6 (it only
   creates `ObservationRecord`s; it doesn't touch `CampaignRun` itself), but should not be
   wired into the scheduler entry point until step 4 (LCO adapter) is live, or newly discovered
   records will sit with no calendar presence until the next milestone's adapter work lands.
8. **Scheduler entry point** — last, once every step it orchestrates exists independently and
   has its own failure-isolation contract; the entry point itself should add no new business
   logic beyond sequencing and step-level failure reporting.
9. **GAPB-01 / UNUSED-01 / STATUS-01/02** (carried-forward v2.2-deferred items) — after the
   adapters are rewired (steps 3-4), since all three are direct consequences of queue-sourced
   `CampaignRun`s now existing and producing container-plus-observation events rather than raw
   `CalendarEvent`s.

## Sources

- `solsys_code/campaign_reconciler.py` (full read) — the pure-reconciler contract, ownership
  rules, and the two-key-family design this milestone must slot into
- `solsys_code/management/commands/reconcile_campaign_runs.py` — the existing sweep-command
  pattern (per-run failure isolation, `--dry-run`) the new scheduler/discovery commands should
  mirror
- `solsys_code/management/commands/load_telescope_runs.py`,
  `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`,
  `backfill_lco_observation_records.py` — current adapter shapes and identity keys
- `solsys_code/campaign_utils.py` (`insert_or_create_campaign_run`, `resolve_site`) — the
  existing create-or-update contract to extend, not replace
- `solsys_code/models.py` — `CampaignRun`/`CalendarEventMeta`/`CampaignRunObservation` schema,
  including the `NOT NULL campaign` FK and existing `UniqueConstraint`s driving the Critical
  Integration Risk section above
- `.planning/milestones/v2.2-phases/26-canonical-record-spike/26-DECISION.md` and
  `docs/design/canonical_record_spike.rst` — already-settled per-adapter identity-key mappings
  (Criterion 2/SPIKE-02) and the queue-vs-classical event-key verdict this milestone's adapter
  rewiring must be consistent with, not re-litigate
- `.planning/PROJECT.md` — v2.2 SHIPPED section (four-stage window pipeline table), Key
  Decisions (the RECON-*/CANON-*/ATTRIB-* rows this research must not contradict), Current
  Milestone v2.3 scope and Active requirements

---
*Architecture research for: FOMO v2.3 — Automatic Run Sync & Outcome Propagation*
*Researched: 2026-09-01*
