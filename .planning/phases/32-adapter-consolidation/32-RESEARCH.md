# Phase 32: Adapter Consolidation - Research

**Researched:** 2026-09-02
**Domain:** Django backend adapter rewiring — three management commands migrating from direct
`CalendarEvent` writes to `CampaignRun`-write-then-`reconcile_run()`, plus the schema migration
(nullable `campaign`, new `source_identifier` field, new `SOAR_QUEUE` source value) that Phase 31's
spike specified but did not implement.
**Confidence:** HIGH — every claim below is grounded in code read this session (`solsys_code/models.py`,
`campaign_reconciler.py`, the three management commands, `campaign_utils.py`, `calendar_utils.py`,
existing tests) or in Phase 31's `31-DECISION.md`/spike `.rst`, both locked canonical artifacts for
this phase.

## Summary

Phase 32 has two layers of work, and the first is easy to under-scope if the planner reads only
ADAPT-01..06's prose: **(1) a schema migration that nothing has built yet** — despite Phase 31's
spike verdicts reading as settled, `solsys_code/models.py` today still has `campaign` as
`null=False` and no `source_identifier` field at all (confirmed by reading the model directly,
`models.py:167-173`, `models.py:277-317`) — and **(2) the three adapter rewires** the roadmap
names. The spike's own summary is explicit that this is Phase 32's job: "No `CampaignRun` schema
migration, no adapter code, and no scheduler entry point was built during this spike" (spike
`.rst:14-15`). The planner must schedule the migration as this phase's first plan/task, before any
adapter can write a non-campaign `CampaignRun` row.

The shared write-and-reconcile helper CONTEXT.md calls "groundwork" already has 90% of a
precedent to build on: `campaign_utils.insert_or_create_campaign_run(lookup, fields)`
(`campaign_utils.py:817-853`) already implements the exact no-churn create-or-update contract for
`CampaignRun` that `import_campaign_csv.py` uses today. The new helper should wrap this existing
function with a `reconcile_run()` call and (per ADAPT-02) a `CampaignRunObservation` link step —
not reinvent create-or-update from scratch.

`campaign_reconciler.reconcile_run()` is the single downstream entry point every adapter must
call, and it has a hard precondition the planner must carry into every adapter's write path:
`_skip_reason()` (`campaign_reconciler.py:193-214`) returns `'not approved'` unless
`run.approval_status == CampaignRun.ApprovalStatus.APPROVED` — every adapter-created `CampaignRun`
must set `approval_status=APPROVED` at write time (matching the `Source.__doc__`'s derivation
rule: `approval_status == APPROVED` together with `source != WEB` means "no approval was
required", not "a human approved it").

The nullable-`campaign` migration is not free of blast radius: Phase 31 measured **5 call sites**
that dereference `run.campaign.name`/`run.campaign_id` and will raise `AttributeError` the first
time an adapter writes a null-campaign row, one of them (`campaign_reconciler.event_title()`) on
the hot path every `reconcile_run()` call goes through. All 5 must be null-guarded in this phase's
groundwork plan, before the first adapter cuts over.

**Primary recommendation:** Plan Phase 32 as four ordered plans — (0) schema migration +
null-guard fixes + shared write-and-reconcile helper (wrapping `insert_or_create_campaign_run` +
`reconcile_run` + exact-identity `CampaignRunObservation` linking), (1) classical adapter cutover,
(2) LCO/SOAR adapter cutover (including the `SOAR_QUEUE` migration), (3) Gemini adapter cutover —
matching D-05's hard-cutover, ship-order, and D-01/D-02's SOAR-then-Gemini sequencing exactly.

## Architectural Responsibility Map

FOMO is a single-tier Django monolith for this phase's scope — there is no browser/SSR/CDN split;
"backend" here means Django ORM + management-command batch jobs, and "database" means the SQLite
schema and its constraints.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `CampaignRun.campaign` nullable + `source_identifier` field/constraint | Database / Storage | Backend (migration file) | Schema shape change; Phase 31 locked the exact field declaration and constraint (31-DECISION.md) |
| `SOAR_QUEUE` Source value | Database / Storage | Backend (migration file) | New `TextChoices` member, same precedent as `ESO_QUEUE` (migration 0014) |
| Shared write-and-reconcile helper | Backend | — | Peer module under `solsys_code/` (never `campaign_views.py`, never importing `solsys_code.views`/`ephem_utils`) — locked constraint from CONTEXT.md's Integration Points |
| Classical/LCO-SOAR/Gemini adapter rewrites | Backend | — | Existing management commands under `solsys_code/management/commands/` |
| Null-guard fixes (5 sites) | Backend | — | `models.py` `__str__`, `campaign_reconciler.event_title()`, `campaign_tables.py` (x2), `campaign_attribution.py` evidence builder |
| `CampaignRunObservation` exact-identity linking | Backend | Database (constraint) | Adapter-side write, backed by the existing `unique_campaign_run_observation_record` constraint |
| Calendar projection (`reconcile_run()`) | Backend | Database (CalendarEvent writes) | Unchanged this phase — adapters call it, never re-implement it |

## User Constraints (from CONTEXT.md)

<user_constraints>
### Locked Decisions

- **D-01:** ADAPT-03 is retargeted from Gemini to SOAR. `sync_lco_observation_calendar` gains a
  dedicated `CampaignRun.Source.SOAR_QUEUE` value (and its migration) for SOAR-sourced records —
  `SOARFacility` inherits a real portal read-back from `LCOFacility`, so it is the facility that
  actually proves the pattern generalises to a second, live-read-back facility. `GEMFacility` is
  submission-echo only (`get_observation_status()`/`get_observation_url()` are hardcoded stubs;
  the only outbound call is `submit_observation()`) and cannot prove that. — **Reversibility:**
  one-way — `SOAR_QUEUE` is a new `TextChoices` member plus a schema migration; once SOAR-sourced
  rows exist under it, collapsing it back into `LCO_QUEUE` means a data migration, not just a code
  revert.
- **D-02:** Gemini's own write path is kept in scope as a new requirement, ADAPT-06:
  `sync_gemini_observation_calendar` still creates or updates a `CampaignRun` from its
  submission-echo data (real, useful for calendar visibility) instead of writing a `CalendarEvent`
  directly — but both the code and `docs/runbooks/telescope_runs_calendar.rst` must state
  explicitly that a Gemini-sourced run can never receive Phase 33's automatic outcome propagation,
  so this is discovered now, not as a mid-Phase-33 surprise. — **Reversibility:** reversible —
  purely a documentation/scope statement; no schema commitment beyond the already-declared
  `GEMINI_QUEUE` source value.
- **Ship order changes accordingly:** classical → LCO/SOAR (one command, two source values) →
  Gemini. SOAR is not a fourth command to build; it is a second `Source` value inside the existing
  LCO sync command's write path.
- **D-03 (Claude's discretion, see below):** the shared helper's lookup key.
- **D-04:** Ship Phase 31's documented risk as-is — no proposal-code parsing work is added to
  `load_telescope_runs`'s line grammar in this phase. Two different proposals colliding on the
  same telescope/instrument/night remains a documented, low-frequency risk (only 1/3 real sample
  lines carried a proposal code, and it did not parse under today's grammar). —
  **Reversibility:** reversible — closing the gap later is additive parser work, not a schema or
  identity-key change.
- **D-05:** Hard cutover per adapter, in commit order. Each adapter's plan flips its write path
  from direct `CalendarEvent` writes to `CampaignRun`-write-and-reconcile in the same commit that
  ships it — classical first, then LCO/SOAR, then Gemini. No dual-write period, no feature flag. —
  **Reversibility:** costly — reverting a shipped adapter's write path after later adapters and the
  reconciler already depend on its `CampaignRun` rows means an explicit rollback plan, not a
  one-line revert.

### Claude's Discretion

- **D-03:** The write-and-reconcile helper's lookup-key priority. Claude's recommendation (per
  CONTEXT.md): look up an existing `CampaignRun` by `source_identifier` first, falling back to the
  existing campaign+window lookup only when `source_identifier` is absent — matching how
  31-DECISION.md frames `source_identifier` going forward for adapter-written rows. **This
  research's finding, below, refines that recommendation** — see Pattern 1 and Pitfall 1.

### Deferred Ideas (OUT OF SCOPE)

- **`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`** — renaming
  `calendar_utils.py`'s private helpers to reflect shared-module status. Pure style cleanup, no
  behavior change. Left for its own quick task.
- **`2026-09-01-add-ttl-cache-to-attribution-banner-count.md`** — campaign-list page attribution
  banner caching. Unrelated to adapter writes.
- **`2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md`** — attribution
  dismiss-action security guard. Unrelated to adapter writes.
- Phase 33 (outcome propagation), Phase 34 (scheduler entry point), Phase 35 (status vocabulary
  unification) — explicitly out of scope per CONTEXT.md's phase boundary. This research does not
  investigate them.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ADAPT-01 | `load_telescope_runs` creates or updates a `CampaignRun` instead of writing calendar events directly | Pattern 1 (shared helper), Code Example 1; existing `insert_or_create_calendar_event()` call site at `load_telescope_runs.py:207-216` is the exact code to replace |
| ADAPT-02 | `sync_lco_observation_calendar` creates or updates a `CampaignRun`, and automatically links the realising `ObservationRecord` via `CampaignRunObservation` at creation time (exact identity, not scored attribution) | Pattern 2 (exact-identity linking), Pitfall 3; existing `_build_event_fields()`/`insert_or_create_calendar_event()` call site at `sync_lco_observation_calendar.py:317-341` is the exact code to replace |
| ADAPT-03 | Same for SOAR-sourced observations, under `CampaignRun.Source.SOAR_QUEUE` (new value + migration) | Pattern 3 (facility-dispatch → Source-value branch), Migration precedent (migration 0014) |
| ADAPT-04 | Each rewired adapter's own idempotency guarantee is verified against the new `CampaignRun` write path | Validation Architecture section; existing no-churn test pattern in `test_sync_lco_observation_calendar.py:379-421` and `test_load_telescope_runs.py:218-251` is the template to mirror against `CampaignRun` |
| ADAPT-05 | Explicit, stated cutover sequencing that never produces a duplicate or orphaned calendar event | Pitfall 2 (hard-cutover ordering, D-05); Runtime State Inventory |
| ADAPT-06 | `sync_gemini_observation_calendar` creates or updates a `CampaignRun` from submission-echo data, with an explicit caveat it can never receive Phase 33 outcome propagation | Pattern 1, Pitfall 4 (documentation caveat), paired-docs obligation (CLAUDE.md) |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **GSD workflow enforcement:** no direct repo edits outside a GSD command.
- **Test runner:** `python manage.py test <labels>` only — `./manage.py` is not supported;
  `pytest` does not collect these tests. Exclude `solsys_code.tests.test_views.TestEphemeris`
  (segfaults in native ASSIST — MEMORY.md gotcha) from any ad hoc test invocation; the project's
  own `test_command` in `.planning/config.json` already encodes the correct exclusion list.
- **Lint/format:** `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files`
  must stay clean — always run through pre-commit's pinned ruff 0.2.1, not an unpinned `ruff` on
  `PATH`.
- **Target test factories:** always use `tom_targets.tests.factories.NonSiderealTargetFactory` in
  any new/edited test touching `Target`, never `SiderealTargetFactory` — confirmed as the existing
  convention in `test_sync_lco_observation_calendar.py:16` and `test_campaign_reconciler.py:18`.
- **Planning-doc terminology:** write "create or update" / "find-or-create", never "upsert" — this
  RESEARCH.md follows that convention throughout.
- **Paired docs are part of the deliverable, not optional polish.** This phase's CLAUDE.md
  scope note (already updated for Phase 32) names five paired artifacts to update as part of
  `files_modified`, not as a follow-up:
  - `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
  - `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
  - `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`
  - `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (the reconciler now receives
    adapter-created runs)
  - `docs/runbooks/telescope_runs_calendar.rst` (every command's documented effect changes,
    including the new `SOAR_QUEUE` source and the Gemini outcome-propagation caveat)
  Notebooks are regenerated via `jupyter nbconvert --to notebook --execute --inplace` and
  committed **with output** (the `pre_executed/` exception to pre-commit's usual notebook-output
  clearing).
- **Never a private-helper import across modules** — `campaign_reconciler.py`'s existing
  `update_calendar_event_key_and_fields()` docstring calls out this exact anti-pattern
  (`calendar_utils.py:550-559`); the new shared helper must expose a public function, not rely on
  callers reaching into another module's `_private` internals.

## Standard Stack

No new third-party libraries are introduced by this phase — it is a schema migration plus
rewiring of existing Django management commands using the project's existing ORM, `TextChoices`,
and `UniqueConstraint`/`CheckConstraint` machinery. All "stack" for this phase is already
in `pyproject.toml` and already imported by the modules being edited (Django ORM, `zoneinfo`,
`astropy` via `telescope_runs.sun_event()` — unchanged).

### Alternatives Considered

Not applicable — this phase is exclusively internal rewiring of existing code paths against an
already-locked schema design (Phase 31's spike). No library selection decision exists.

## Package Legitimacy Audit

**Not applicable.** This phase installs no new external packages — every module touched
(`solsys_code/models.py`, `campaign_reconciler.py`, `campaign_utils.py`, `calendar_utils.py`, the
three management commands) already exists in the repository and imports only already-present
dependencies (Django, `zoneinfo`, `astropy`/`erfa` via `telescope_runs`, `tom_observations`).

## Architecture Patterns

### System Architecture Diagram

```
Classical schedule file          LCO/SOAR ObservationRecord      Gemini ObservationRecord
        │  (load_telescope_runs)         │ (sync_lco_observation_calendar)   │ (sync_gemini_observation_calendar)
        ▼                                 ▼                                    ▼
  parse_run_line() / sun_event()   _build_event_fields()                derive fields from
  (unchanged this phase)           (unchanged this phase)                parameters JSON
        │                                 │                                    │
        └───────────────┬─────────────────┴──────────────┬─────────────────────┘
                         ▼                                ▼
              write_and_reconcile_campaign_run(lookup, fields, source, source_identifier)
              [NEW shared helper — this phase's groundwork]
                         │
                         ├─► insert_or_create_campaign_run(lookup, fields)   [EXISTING, campaign_utils.py]
                         │        creates/updates CampaignRun, approval_status=APPROVED, source=<adapter's value>
                         │
                         ├─► CampaignRunObservation.get_or_create(observation_record=..., defaults={'run': run})
                         │        [NEW for ADAPT-02/03 — exact-identity link, LCO/SOAR only;
                         │         classical/Gemini have no ObservationRecord to link to]
                         │
                         └─► reconcile_run(run)   [EXISTING, campaign_reconciler.py — UNCHANGED entry point]
                                  │
                                  ├─► _reconcile_container()        (class-wide / satellite runs)
                                  └─► _reconcile_classical_nights()  (per-night ground runs)
                                           │
                                           ▼
                                  CalendarEvent (create/update/unchanged, no-churn)
                                  CalendarEventMeta.run linked
```

### Recommended Project Structure

No new files/directories are needed. The shared helper is a new function in an existing peer
module (see Integration Points below) — CONTEXT.md's locked constraint says it must live
"alongside `campaign_reconciler.py` / `campaign_gap.py` / `campaign_utils.py`", never inside
`campaign_views.py`. `campaign_utils.py` is the natural home: it already owns
`insert_or_create_campaign_run()`, the exact function this helper wraps.

```
solsys_code/
├── campaign_utils.py           # add write_and_reconcile_campaign_run() here — wraps the
│                                # existing insert_or_create_campaign_run() + reconcile_run()
├── campaign_reconciler.py      # unchanged public entry point (reconcile_run); null-guard
│                                # event_title() (models.py:352-equivalent hot site)
├── models.py                   # migration: campaign nullable, source_identifier + constraint,
│                                # SOAR_QUEUE Source value; null-guard __str__()
├── campaign_tables.py          # null-guard both render_run() methods
├── campaign_attribution.py     # null-guard _campaign_evidence()
└── management/commands/
    ├── load_telescope_runs.py                 # ADAPT-01: replace CalendarEvent write
    ├── sync_lco_observation_calendar.py        # ADAPT-02/03: replace CalendarEvent write,
    │                                            # branch Source by facility, link CampaignRunObservation
    └── sync_gemini_observation_calendar.py     # ADAPT-06: replace CalendarEvent write
```

### Pattern 1: Shared write-and-reconcile helper wraps the existing `insert_or_create_campaign_run()`

**What:** `campaign_utils.py` already has a fully-formed no-churn create-or-update function for
`CampaignRun`, used today by `import_campaign_csv.py`:

```python
# Source: solsys_code/campaign_utils.py:817-853 (read this session)
def insert_or_create_campaign_run(lookup: dict[str, Any], fields: dict[str, Any]) -> tuple[CampaignRun, str]:
    """Create or update a CampaignRun, or leave it unchanged if no fields differ."""
    run, created = CampaignRun.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return run, 'created'
    changed = [f for f, v in fields.items() if getattr(run, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(run, f, v)
        run.save(update_fields=list(fields.keys()))
        return run, 'updated'
    return run, 'unchanged'
```

**When to use:** As the lookup/write layer inside the new shared helper. Do not reimplement
`get_or_create` + field-diff logic a second time — this is exactly the "same pattern being
written three times" CONTEXT.md's scope note warns against, except it would now be four times
(CSV import already has it).

**New helper shape (recommended, not existing code):**

```python
# NEW — proposed shape for campaign_utils.py, following the existing
# insert_or_create_campaign_run() signature and no-churn contract exactly.
def write_and_reconcile_campaign_run(
    lookup: dict[str, Any],
    fields: dict[str, Any],
    *,
    observation_record=None,
) -> tuple[CampaignRun, str, ReconcileResult]:
    """Create/update a CampaignRun and immediately reconcile its calendar projection.

    `fields` must include `approval_status=CampaignRun.ApprovalStatus.APPROVED` and the
    adapter's own `source` value — reconcile_run()'s _skip_reason() guard rejects any run
    whose approval_status is not APPROVED (campaign_reconciler.py:204-205).

    `observation_record`, when given (LCO/SOAR only — ADAPT-02/03), gets an exact-identity
    CampaignRunObservation link created alongside the run (see Pattern 2) — never a scored
    attribution candidate.
    """
    run, action = insert_or_create_campaign_run(lookup, fields)
    if observation_record is not None:
        CampaignRunObservation.objects.get_or_create(
            observation_record=observation_record,
            defaults={'run': run, 'confirmed_at': timezone.now()},  # confirmed_by=None: system link
        )
    result = reconcile_run(run)
    return run, action, result
```

### Pattern 2: Exact-identity `CampaignRunObservation` linking (ADAPT-02/03)

**What:** The staff-confirmation queue's existing confirm-write is the pattern to mirror for
field shape, minus the human actor:

```python
# Source: solsys_code/campaign_views.py:1201-1204 (read this session) — the
# human-confirmation path this phase's adapter-side link must structurally match,
# with confirmed_by left None (no staff user acted) instead of request.user.
_, created = CampaignRunObservation.objects.get_or_create(
    observation_record_id=orphan_pk,
    defaults={'run_id': run_pk, 'confirmed_by': request.user, 'confirmed_at': timezone.now()},
)
```

**When to use:** Only for LCO/SOAR (ADAPT-02/03) — the adapter creates one `CampaignRun` per
`ObservationRecord` (the `source_identifier` is that record's own portal request URL), so the
"realising `ObservationRecord`" is exactly the record the adapter is currently processing, not a
scored guess. This is why ADAPT-02's "exact identity, not scored attribution" language is
satisfiable without touching `campaign_attribution.py`'s scorer at all: the link is structural
(the adapter always knows which record it just wrote a run for), not inferred.

Classical and Gemini adapters have **no** `ObservationRecord` to link (classical has no facility
record at all; Gemini's `ObservationRecord` already exists as the *source* of the write, not a
"realising" record found afterward) — `observation_record=None` for both, no
`CampaignRunObservation` row created.

`unique_campaign_run_observation_record` (`models.py:444-447`) already enforces one run per
record globally — the `get_or_create()` lookup keyed on `observation_record` alone is
race-safe against this constraint, matching WR-05's precedent (31-DECISION.md line ~781-787).

### Pattern 3: Facility-dispatch branch selects `Source` value (ADAPT-03)

**What:** `sync_lco_observation_calendar.py`'s existing facility-dispatch dict is already the
natural branch point:

```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:289 (read this session)
facilities = {'LCO': LCOFacility(), 'SOAR': SOARFacility()}
```

**When to use:** Add a parallel `{'LCO': CampaignRun.Source.LCO_QUEUE, 'SOAR': CampaignRun.Source.SOAR_QUEUE}`
mapping keyed the same way, and read it inside the per-record loop (`for record in records:`,
line 303) alongside the existing `facility = facilities.get(record.facility)` lookup — no new
dispatch mechanism needed, this is the same `record.facility` value already driving facility
instance selection.

### Anti-Patterns to Avoid

- **Re-adding a direct `CalendarEvent` write inside an adapter after cutover.** CONTEXT.md's
  locked constraint: "adapters never re-acquire a direct `CalendarEvent` write path." Every
  adapter's only calendar-writing call after this phase must be transitively through
  `reconcile_run()`.
- **Adding `source_identifier` alongside a still-required campaign+window lookup, forever.**
  31-DECISION.md's own Phase 32 guidance (lines ~759-766) is explicit: promote `source_identifier`
  to the primary lookup key for adapter-written rows, don't just add it as a second parallel
  lookup with no plan to retire the first — see Pitfall 1 below for why this matters concretely.
- **Building the new helper inside `campaign_views.py`.** CONTEXT.md's Integration Points locks
  this: "never a private helper inside `campaign_views.py`, and never importing
  `solsys_code.views` or `solsys_code.ephem_utils`" (the 1.6 GB SPICE kernel download is fatal for
  an unattended job).
- **Reformatting the CalendarEvent lookup keys the adapters already use.** The tolerance-match
  key (`{'telescope', 'instrument', 'start_time'}` for classical, `{'url'}` for LCO/SOAR/Gemini)
  is orthogonal to the new `CampaignRun` lookup key — this phase adds a second lookup, it does not
  touch the first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CampaignRun create-or-update with no-churn field diffing | A new `get_or_create` + manual field-diff loop per adapter | `campaign_utils.insert_or_create_campaign_run()` | Already exists, already tested (`import_campaign_csv.py`'s consumer), already matches the project's `save(update_fields=...)` convention |
| Calendar projection from a `CampaignRun` | Adapter-specific `CalendarEvent` construction (what all three commands do today) | `campaign_reconciler.reconcile_run()` | Already handles both the container (class-wide/satellite) and per-night branches, already idempotent, already handles stale-family detachment |
| Human-confirmed vs. exact-identity attribution distinction | A new boolean flag on `CampaignRunObservation` to distinguish "adapter-linked" from "staff-confirmed" | Nothing — the existing model already supports this: `confirmed_by=None` for a system-created link is a legitimate, already-nullable state (`models.py:423-430`) | D-03 of `CampaignRunObservation`'s own docstring explicitly rejected adding a boolean confirmation flag as redundant state |

**Key insight:** almost every piece of this phase's "new" logic already has a shipped analog
somewhere in `solsys_code/` — CSV import already does create-or-update-CampaignRun, the
attribution queue already does exact-key `CampaignRunObservation.get_or_create`, and the
reconciler already does the calendar-projection idempotency. The actual new work is the
migration, the null guards, and threading these three existing patterns together per adapter.

## Runtime State Inventory

**Trigger for this section:** this phase is not a rename/refactor, but it does change a
long-lived *write path* for three production commands with a mandated hard cutover (D-05) — the
"what still points at the old shape after the code changes" question applies to write-path
migrations too, not only renames. Answered per category below.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | 49 existing `CampaignRun` rows (0 null-campaign), all pre-adapter (`legacy`/`csv_import`/`web`/`eso_queue`/`classical_file`/`lco_queue` sources — the last two are pre-v2.3 legacy-labeled rows, not adapter output). No data migration needed for these — the nullable-FK migration is a single `AlterField` with no `RunPython` backfill (31-DECISION.md, confirmed 0/49). | None — schema-only migration for existing rows. |
| Stored data (calendar) | Every existing `CalendarEvent` row the three adapters currently own (keyed by `url` for LCO/SOAR/Gemini, by `(telescope, instrument, start_time)` tolerance for classical) stays exactly as-is at cutover — the adapters' `CalendarEvent`-side lookup keys are unchanged by this phase; only the *new* `CampaignRun`-side write is added in front of them. | None — verify with a before/after event-count assertion per adapter cutover (ADAPT-05). |
| Live service config | None — no external service holds FOMO-specific config for this phase's scope (LCO/SOAR/Gemini credentials are read-only inputs to already-existing sync commands, untouched here). | None. |
| OS-registered state | None — Phase 34 (scheduler entry point), explicitly out of scope, owns cron/flock registration. This phase's commands remain manually invoked exactly as today. | None. |
| Secrets/env vars | None — this phase adds no new credential and reads no existing one differently. | None. |
| Build artifacts | Migration files: next migration number is `0015` (`0014_alter_campaignrun_source.py` is the latest). At least two migrations are needed — one for the nullable-`campaign`/`source_identifier` schema shape (groundwork plan), one for the `SOAR_QUEUE` `Source` value (LCO/SOAR plan, matching the ESO precedent's own single-purpose migration 0014). Whether these are separate files or combined is a planner call; keeping them separate mirrors the existing one-migration-per-schema-concern precedent in this migrations directory. | Write migration(s); no reinstall/package step needed (pure Django ORM). |

## Common Pitfalls

### Pitfall 1: `source_identifier` mirrors the tolerance match's blind spot for classical runs, not a broader identity than it

**What goes wrong:** The recommended classical `source_identifier` value,
`f'CLASSICAL:{telescope}:{instrument}:{bucket}'` (5-minute-bucketed `start_time`), uses the exact
same three fields the existing `_START_TIME_MATCH_TOLERANCE` lookup already uses
(`load_telescope_runs.py:207-216`). Two different proposals allocated the same
telescope+instrument+night still collide on both keys identically — `source_identifier` does not
close SCHEMA-03's documented gap, it inherits it verbatim.

**Why it happens:** The field was designed to give non-campaign rows *some* write-time identity
surface (closing the null-campaign constraint's silent discrimination loss), not to add proposal
discrimination the underlying schedule-file grammar cannot parse today (D-04 accepts this as-is).

**How to avoid:** Do not let the planner or a reviewer treat `source_identifier` as "solving"
SCHEMA-03/the two-proposals-same-night risk for the classical path — it is explicitly *not* a
fix for that (31-DECISION.md, SCHEMA-03 section, "not a contradiction... but it is an incomplete
mitigation"). Document the accepted risk in the classical adapter's plan exactly as D-04 states it.

**Warning signs:** A test asserting two different proposals on the same telescope/instrument/night
produce two distinct `CampaignRun` rows would fail — this is expected and must not be "fixed" by
this phase.

### Pitfall 2: Null-guard the 5 `run.campaign.name`/`run.campaign_id` sites before, not after, the first adapter cutover

**What goes wrong:** The moment any adapter writes a `CampaignRun` with `campaign=None`, these 5
sites raise `AttributeError` on next read:

| File:Line | Expression | Hot/Cold |
|---|---|---|
| `solsys_code/models.py:352` (`CampaignRun.__str__`) | `self.campaign.name` | Cold — admin/UI render paths |
| `solsys_code/campaign_reconciler.py:176` (`event_title()`) | `run.campaign.name` | **Hot — every `reconcile_run()` call** |
| `solsys_code/campaign_tables.py:467` (`DismissalHistoryTable.render_run`) | `record.run.campaign.name` | Cold |
| `solsys_code/campaign_tables.py:538` (second dismissal-table render method) | `record.run.campaign.name` | Cold |
| `solsys_code/campaign_attribution.py:397` (`_campaign_evidence`) | `run.campaign.name` | Cold |

All 5 quoted verbatim (grep-confirmed this session and matching 31-DECISION.md's own inventory
exactly): `models.py:352` → `f'#{self.pk} {self.campaign.name} | ...'`;
`campaign_reconciler.py:176` → `base = f'{run.campaign.name}: {run.telescope_instrument}'`;
`campaign_tables.py:467,538` → `f'{record.run.telescope_instrument} ({record.run.campaign.name})'`;
`campaign_attribution.py:397` → `f"run belongs to campaign '{run.campaign.name}' ..."`.

**Why it happens:** `campaign` has been a required FK for the model's entire history; nothing in
the existing codebase was written expecting `None`.

**How to avoid:** Fix all 5 (e.g. `run.campaign.name if run.campaign_id else '<no campaign>'`,
exact wording is a planner/UX call) inside this phase's groundwork plan, **before** any adapter
plan cuts over. `campaign_reconciler.py:176` is the priority fix — it is the one every
`reconcile_run()` call exercises.

**Warning signs:** A `CommandError`/traceback the first time a null-campaign row reaches
`reconcile_run()` mid-adapter-rollout, or a broken admin changelist/attribution-queue page.

### Pitfall 3: `CampaignRunObservation`'s D-01 docstring reads as "human-confirmation only" — ADAPT-02's automatic link is a different, still-valid case

**What goes wrong:** A reviewer reading `CampaignRunObservation`'s class docstring ("a row exists
only once a staff member confirms the attribution... this keeps ATTRIB-03... structural") could
conclude ADAPT-02's automatic linking contradicts that invariant.

**Why it happens:** The docstring's concern is Phase 28's *scored candidate* attribution queue —
never persisting an unconfirmed guess. It is not a blanket ban on any non-human write to this
model; it is a ban on the attribution *scorer* writing rows on its own.

**How to avoid:** ADAPT-02's link is not the scorer's output — it is structural identity (the
adapter processes exactly one `ObservationRecord` per `CampaignRun` write, so there is no
candidate ambiguity to resolve). Document this distinction explicitly in the adapter plan/task so
a future reader does not "fix" it by routing adapter-created runs through the attribution queue
instead. `confirmed_by=None` correctly represents "no human confirmed this" while
`confirmed_at` can still be set to record when the automatic link was made — this is a legitimate,
already-supported model state, not a new field.

**Warning signs:** A code reviewer or the plan-checker flagging the adapter-side
`CampaignRunObservation.get_or_create()` call as an ATTRIB-03 violation.

### Pitfall 4: A duplicate/orphan CalendarEvent window can open at the exact moment of cutover, not just from a coding bug

**What goes wrong:** D-05's hard cutover means the classical adapter's commit both stops writing
`CalendarEvent` directly and starts writing via `CampaignRun`+`reconcile_run()` — in the same
commit. If the `CampaignRun`-side lookup key (`source_identifier` or campaign+window) does not
find the *same* existing `CalendarEvent` the old direct-write path was already updating in place,
the reconciler could mint a **second** event for a night that already has one from a pre-cutover
run of the same command.

**Why it happens:** The two lookup keys are structurally different (`CalendarEvent`'s own
tolerance-match key vs. `CampaignRun`'s new key) and are resolved independently —
`reconcile_run()`'s `_reconcile_classical_nights()` has its own adopt step
(`_adopted_event_for_night()`, `campaign_reconciler.py:284-335`) specifically for this kind of
"a pre-existing, unattributed classical event should be adopted, not duplicated" case, but it only
adopts events whose `CalendarEventMeta.run_id` already equals the run being reconciled — a
freshly-created `CampaignRun` (this is its first reconcile) has no prior `CalendarEventMeta` link
at all, so this adopt path will not find last night's classical event on the very first
adapter-driven reconcile after cutover unless the adapter's own lookup/`source_identifier` design
accounts for pre-existing, already-`load_telescope_runs`-owned events.

**How to avoid:** ADAPT-05's plan must include an explicit before/after event-count assertion
across the cutover commit for each adapter (Success Criterion 5's "one event per night, no
duplicates, no orphans" bar) — not just a unit test of the new write path in isolation. Consider
whether the classical adapter's *first* `CampaignRun`-driven reconcile pass needs to pre-populate
`CalendarEventMeta.run` links for already-existing, blank-`url` classical events matching the new
run's telescope/instrument/window, mirroring what `_adopted_event_for_night()` already does for
runs reconciled a second time.

**Warning signs:** A cutover integration test showing `CalendarEvent.objects.count()` increasing
by more than the number of genuinely new nights when a previously-`load_telescope_runs`-only
schedule file is re-ingested for the first time under the new adapter.

## Code Examples

### The exact code each adapter's cutover replaces

```python
# Source: solsys_code/management/commands/load_telescope_runs.py:207-216 (read this session)
# ADAPT-01 replaces this insert_or_create_calendar_event() call with a call to the new
# write_and_reconcile_campaign_run() helper — the CalendarEvent-level tolerance-match lookup
# key ({'telescope', 'instrument', 'start_time'}) is unchanged; only the outer write path changes.
event, action = insert_or_create_calendar_event(
    {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
    {
        'end_time': end_time,
        'title': title,
        'description': description,
        'target_list': campaign,
    },
    start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
)
```

```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:329,341 (read this session)
# ADAPT-02/03 replaces this pair with a write_and_reconcile_campaign_run() call carrying
# observation_record=record for the exact-identity CampaignRunObservation link (Pattern 2).
url = fields.pop('url')
...
event, action = insert_or_create_calendar_event({'url': url}, fields)
```

```python
# Source: solsys_code/management/commands/sync_gemini_observation_calendar.py:150,163 (read this session)
# ADAPT-06 replaces this pair. No observation_record link (Pattern 2's classical/Gemini case) —
# GEM-KEY-01's synthesized key becomes the CampaignRun's source_identifier too.
url = f'GEM:{prog}/{record.observation_id}'
...
_event, action = insert_or_create_calendar_event({'url': url}, fields)
```

### Migration precedent to transcribe for the schema groundwork plan

```python
# Source: 31-DECISION.md SCHEMA-01/02 Recommendation sections (locked, transcribe verbatim)
campaign = models.ForeignKey(
    TargetList,
    on_delete=models.PROTECT,   # unchanged
    null=True,                  # was: null=False
    blank=True,                 # new
    related_name='campaign_runs',
    verbose_name='Campaign target list',
)

source_identifier = models.CharField(max_length=500, null=True, blank=True)

# Meta.constraints addition (additive alongside both existing partial constraints):
models.UniqueConstraint(
    fields=('source_identifier',),
    condition=models.Q(source_identifier__isnull=False),
    name='unique_campaign_run_source_identifier',
),
```

### Migration precedent for the `SOAR_QUEUE` `Source` value

```python
# Source: solsys_code/migrations/0014_alter_campaignrun_source.py (read this session) —
# the exact precedent for adding a Source value: a single AlterField transcribing the
# full, ordered choices list. SOAR_QUEUE follows the identical shape, added after GEMINI_QUEUE
# (or wherever ordering convention places it — no ordering constraint was found in the model).
migrations.AlterField(
    model_name='campaignrun',
    name='source',
    field=models.CharField(
        choices=[
            ('web', 'Web submission'),
            ('classical_file', 'Classical run file'),
            ('lco_queue', 'LCO queue'),
            ('soar_queue', 'SOAR queue'),          # NEW
            ('gemini_queue', 'Gemini queue'),
            ('eso_queue', 'ESO queue'),
            ('csv_import', 'CSV import'),
            ('legacy', 'Legacy (pre-v2.2)'),
        ],
        default='legacy',
        max_length=20,
        verbose_name='Ingest source',
    ),
),
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Each adapter writes `CalendarEvent` directly via `insert_or_create_calendar_event()` | Each adapter writes `CampaignRun` via the new shared helper, which calls `reconcile_run()` to project `CalendarEvent` | This phase (v2.3, ADAPT-01..06) | A `CampaignRun` exists for every observation regardless of ingest path — "visible by construction" per the phase goal, not because someone remembered to run `reconcile_campaign_runs` afterward |
| `CampaignRun.campaign` required (`null=False`) | `CampaignRun.campaign` nullable, `source_identifier` is the write-time identity surface for non-campaign rows | This phase (v2.3, schema groundwork, per Phase 31's spike) | Routine queue/classical observations can get a persistent identity without a coordinated campaign |
| `CampaignRun.Source` has `LCO_QUEUE`/`GEMINI_QUEUE`/`ESO_QUEUE` but no SOAR value (SOAR rows would misclassify under `LCO_QUEUE`) | Dedicated `SOAR_QUEUE` value | This phase (ADAPT-03) | SOAR-sourced runs are correctly distinguished from LCO-sourced ones, matching the same reasoning that motivated `ESO_QUEUE`'s own addition |

**Deprecated/outdated:** the three adapters' own `CalendarEvent`-level lookup/tolerance-match keys
are **not** deprecated by this phase — they remain the mechanism for the inner `CalendarEvent`
create-or-update that `reconcile_run()` performs; only the *outer* write path (who calls that
mechanism, and when) changes.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The new shared helper should live in `campaign_utils.py` specifically (not a brand-new module) | Recommended Project Structure | Low — CONTEXT.md only locks "a peer module under `solsys_code/`, never `campaign_views.py`"; `campaign_utils.py` is a reasonable, not mandated, specific choice, since it already owns `insert_or_create_campaign_run()` |
| A2 | `confirmed_at=timezone.now()` (not left `None`) for an adapter-created exact-identity link | Pattern 2 | Low — either choice is internally consistent with the nullable field; leaving it `None` would mean "never confirmed" reads oddly for a link that is, in fact, permanently correct by construction. Planner should confirm this against how `OUTCOME-01`'s Phase-33 reader (not in scope here) expects to interpret `confirmed_at` |
| A3 | Splitting the schema migration into two files (nullable-campaign/source_identifier vs. SOAR_QUEUE) rather than one combined migration | Runtime State Inventory | Low — cosmetic; Django does not care, and the existing migrations directory shows one-concern-per-file as the house style, but this is not stated as a hard rule anywhere read this session |
| A4 | The classical adapter's first `CampaignRun`-driven reconcile needs new pre-existing-event adoption logic beyond what `_adopted_event_for_night()` already does | Pitfall 4 | **Medium** — if wrong (i.e. if the existing lookup/tolerance-match key on the `CalendarEvent` side already prevents any duplicate at cutover for reasons this research didn't fully trace), the planner may add unneeded complexity; if the risk is real and unaddressed, ADAPT-05's "never doubles or orphans" bar fails at the classical cutover specifically. This is the single highest-value thing for the planner to prove or disprove with a concrete before/after test before finalizing the classical adapter's plan |

**If this table is empty:** N/A — see rows above; none of the Standard Stack, Package
Legitimacy, or schema-value claims are assumed (all are `[VERIFIED]` against code or the locked
31-DECISION.md), but the four rows above genuinely need a planner decision or a proving test.

## Open Questions

1. **Does the classical adapter's cutover need explicit pre-existing-event adoption logic, or does the existing lookup already prevent duplication?**
   - What we know: `_adopted_event_for_night()` only adopts events already linked
     (`CalendarEventMeta.run_id == run.pk`) — a brand-new `CampaignRun`'s first reconcile has no
     such link yet.
   - What's unclear: whether the `CampaignRun`'s own lookup key (campaign+window, or
     `source_identifier`) will, in practice, resolve to a `CampaignRun` whose reconcile then finds
     the pre-existing `CalendarEvent` via the *unlinked* fallback path in `_may_write()`
     (`campaign_reconciler.py:217-233`, which allows writing an event with no companion row at all
     if its `url` falls in this run's namespace) — but a classical event's `url` is always blank
     (`''`), never `RUN:{pk}:...`, so `_may_write()`'s namespace-fallback branch would also miss it
     on the first pass.
   - Recommendation: the classical adapter's plan should include a concrete before/after
     `CalendarEvent.objects.count()` test against a real (or fixture) schedule file that has
     already been ingested once under the old direct-write path, then re-ingested once under the
     new adapter — asserting the count does not increase. Treat this as the phase's highest-risk
     correctness question (see Assumption A4).

2. **Should the shared helper's signature take `source`/`source_identifier` as explicit keyword
   arguments, or expect them pre-merged into `fields`?**
   - What we know: `insert_or_create_campaign_run(lookup, fields)`'s existing contract treats
     `source_identifier` as just another field in `fields` (once the migration adds it) — no
     special-casing needed at that layer.
   - What's unclear: whether the *new* helper should validate that `fields` always contains
     `approval_status=APPROVED` and a non-`WEB` `source` (mirroring `import_campaign_csv.py`'s
     WR-01 guard against accidentally overwriting a `WEB`-sourced row's approval state), given all
     three adapters always write non-`WEB` sources and none of them should ever collide with a
     `WEB` row's natural key in practice — but this deserves an explicit test either way.
   - Recommendation: planner's call; either shape is compatible with this research's findings.

## Environment Availability

Not applicable — this phase has no new external tool/service/runtime dependency. All work is
Django ORM + Python against the existing SQLite dev database and existing LCO/SOAR/Gemini
facility classes already vendored via `tom_observations`.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`django.test.TestCase`), via `python manage.py test` |
| Config file | none — settings module is `src.fomo.settings` (set by `manage.py`) |
| Quick run command | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_sync_gemini_observation_calendar` |
| Full suite command | The project's own `.planning/config.json` `workflow.test_command`: runs every `solsys_code/tests/test_*.py` and `solsys_code/solsys_code_observatory/tests/test_*.py` label except `test_views.py` (which segfaults natively in ASSIST — excluded except two named safe tests), matching CLAUDE.md's documented gotcha exactly |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ADAPT-01 | Classical adapter creates/updates `CampaignRun` for each processed night | unit | `python manage.py test solsys_code.tests.test_load_telescope_runs` | ✅ existing file, needs new `CampaignRun`-asserting tests added |
| ADAPT-02 | LCO adapter creates/updates `CampaignRun` and links the realising `ObservationRecord` via `CampaignRunObservation` | unit | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` | ✅ existing file, needs new tests |
| ADAPT-03 | SOAR-sourced records get `Source.SOAR_QUEUE` | unit | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` | ✅ existing file (already has `test_select_05_soar_record_uses_soar_facility_instance` at line 608 to extend) |
| ADAPT-04 | No-churn re-sync against the new `CampaignRun` write path (all three adapters) | unit | `python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_sync_gemini_observation_calendar` | ✅ existing no-churn test pattern (`test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` at `test_sync_lco_observation_calendar.py:379`, `test_idempotent_rerun_no_duplicates`/`test_unchanged_rerun_does_not_update_existing_rows` at `test_load_telescope_runs.py:218,229`) is the template to mirror against `CampaignRun.objects.count()`/field-unchanged assertions |
| ADAPT-05 | Cutover never doubles/orphans a `CalendarEvent` | integration | New test per adapter: ingest under old behavior (or fixture pre-existing rows), then run the new adapter, assert `CalendarEvent.objects.count()` unchanged except for genuinely new nights | ❌ Wave 0 — no existing "simulated cutover" test exists for any of the three adapters |
| ADAPT-06 | Gemini adapter creates/updates `CampaignRun`, with the outcome-propagation-impossible caveat documented | unit + docs | `python manage.py test solsys_code.tests.test_sync_gemini_observation_calendar` plus a `docs/runbooks/telescope_runs_calendar.rst` prose check (manual, docs-only requirement) | ✅ existing test file for the code half; docs half has no automated check (manual review) |

### Sampling Rate

- **Per task commit:** the quick run command above (the four directly-touched test modules).
- **Per wave merge:** the full suite command from `.planning/config.json`.
- **Phase gate:** full suite green before `/gsd-verify-work`, plus a manual UAT pass against
  Success Criterion 5 ("one event per night, no duplicates, no orphans, at every point in the
  cutover sequence").

### Wave 0 Gaps

- [ ] A cutover-simulation integration test per adapter (classical, LCO/SOAR, Gemini) — covers
  ADAPT-05, addresses Open Question 1/Pitfall 4 directly.
- [ ] `CampaignRunObservation` exact-identity-link assertions in
  `test_sync_lco_observation_calendar.py` — covers ADAPT-02/03's linking half specifically (not
  just the `CampaignRun` write half).
- [ ] Null-campaign-row regression tests for the 5 guarded read sites (`models.py.__str__`,
  `campaign_reconciler.event_title`, both `campaign_tables.py` render methods,
  `campaign_attribution._campaign_evidence`) — covers the schema groundwork plan; none of these
  sites currently has a null-campaign test case since no code path could produce one before this
  phase.
- [ ] Framework install: none — `django.test.TestCase` is already fully configured.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase touches only management-command batch writes and internal ORM calls, no new auth surface |
| V3 Session Management | No | Not applicable — no request/session code touched |
| V4 Access Control | No | Management commands are already operator-invoked (unattended invocation is Phase 34, out of scope); no new view/endpoint added |
| V5 Input Validation | Marginal | The three adapters already validate their respective input shapes (`parse_run_line()`, `ObservationRecord.parameters`); this phase adds no new external input surface, only a new internal write target (`CampaignRun`) fed from already-validated fields |
| V6 Cryptography | No | No credential or crypto code touched — SCHED-10's credential-in-log concern is explicitly Phase 34's, not this phase's |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A `CampaignRun` collision silently overwrites a different proposal's attribution (SCHEMA-03's documented gap) | Tampering (data-integrity, not an attacker) | Documented, accepted risk per D-04 — not mitigated in this phase; do not let the plan silently "fix" this beyond what D-04 scopes |
| An adapter-created `CampaignRun` accidentally sets `approval_status=APPROVED` on a row that later natural-key-collides with a genuine `WEB` submission | Elevation of Privilege (a public submission bypasses staff review) | Mirror `import_campaign_csv.py`'s existing guard (`campaign_views.py`/`import_campaign_csv.py:356-358`): never let an adapter write overwrite an existing `source=WEB` row's `source`/`approval_status` fields — the adapters' own natural-key/`source_identifier` design should make this collision structurally rare, but the guard is cheap insurance and matches an established precedent in this codebase |
| A malformed/adversarial classical schedule file line reaching `write_and_reconcile_campaign_run()` with attacker-controlled `telescope_instrument`/`description` text | Tampering | Unchanged from today — classical schedule files are operator-supplied, not public input; no new trust boundary is crossed by this phase |

## Sources

### Primary (HIGH confidence — code and locked planning artifacts read this session)

- `solsys_code/models.py` (lines 78-552) — `CampaignRun`, `Source`, `CampaignRunObservation`,
  constraints, `__str__`
- `solsys_code/campaign_reconciler.py` (full file, 511 lines) — `reconcile_run()`,
  `_skip_reason()`, `event_title()`, `_may_write()`, `_adopted_event_for_night()`
- `solsys_code/campaign_utils.py` (lines 780-853) — `map_observation_status()`,
  `insert_or_create_campaign_run()`
- `solsys_code/calendar_utils.py` (lines 470-596) — `insert_or_create_calendar_event()`,
  `update_calendar_event_key_and_fields()`, `preview_calendar_event_action()`
- `solsys_code/management/commands/load_telescope_runs.py` (full file)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` (full file)
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` (full file)
- `solsys_code/management/commands/import_campaign_csv.py` (lines 330-430) — natural-key
  collision guards precedent
- `solsys_code/campaign_views.py` (lines 1188-1215) — human-confirmation `CampaignRunObservation`
  write precedent
- `solsys_code/migrations/0014_alter_campaignrun_source.py` — `ESO_QUEUE` migration precedent
- `solsys_code/tests/test_sync_lco_observation_calendar.py` (imports/setUp/no-churn tests),
  `solsys_code/tests/test_load_telescope_runs.py` (test list), `solsys_code/tests/test_campaign_reconciler.py`
  (imports/base fixture) — existing test conventions and factory usage
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  (full file, 1126 lines) — locked schema/identity/facility-scope decisions this phase executes
- `docs/design/run_identity_and_unattended_invocation_spike.rst` (full file) — durable summary of
  the above
- `.venv/site-packages/tom_observations/facilities/gemini.py` (lines 480-510),
  `tom_observations/facilities/soar.py` (line 240) — installed-package read confirming
  `GEMFacility`'s stub `get_observation_url`/`get_observation_status` and `SOARFacility(LCOFacility)`
  subclassing, verbatim matching 31-DECISION.md's own claims

### Secondary (MEDIUM confidence)

- `.planning/phases/32-adapter-consolidation/32-CONTEXT.md` — user-locked decisions (treated as
  authoritative constraints, not independently re-verified beyond what's cited above)
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/ROADMAP.md` (grep) — requirement
  text and phase sequencing

### Tertiary (LOW confidence)

None — no WebSearch or unverified external source was used for this phase; it is entirely
internal-codebase and locked-planning-artifact research.

## Metadata

**Confidence breakdown:**
- Standard stack: N/A — no new libraries this phase
- Architecture: HIGH — every pattern is either existing shipped code (`insert_or_create_campaign_run`,
  `reconcile_run`, the confirm-write precedent) or a locked Phase 31 spike verdict
- Pitfalls: HIGH for Pitfalls 1-3 (directly sourced from code + 31-DECISION.md); MEDIUM for
  Pitfall 4 (reasoned from reading `_adopted_event_for_night()`/`_may_write()`, not from an
  observed collision — flagged as Open Question 1 / Assumption A4 for the planner to resolve with
  a concrete test before finalizing the classical adapter's plan)

**Research date:** 2026-09-02
**Valid until:** 30 days (stable internal codebase; no external dependency to go stale) — but
re-check against `31-DECISION.md` if Phase 31's directory is archived to `.planning/phases-archive/`
before this phase plans, per the spike doc's own path-note.
