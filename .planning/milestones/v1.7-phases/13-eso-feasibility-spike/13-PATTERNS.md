# Phase 13: ESO Feasibility Spike - Pattern Map

**Mapped:** 2026-07-01
**Files analyzed:** 3 committed deliverables (2 doc files + 1 optional Observatory record via shell/migration)
**Analogs found:** 3 / 3

This is an investigation-only spike. There is no application code to write —
the only committed artifacts are two documents and (optionally) one DB
record. The investigation script itself (Django shell session or scratch
`.py` file used to exercise `p2api`/`tom_eso` against real ESO credentials)
is explicitly **not** a deliverable per CONTEXT.md D-09 and is intentionally
excluded from this pattern map.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` | planning-doc (full detail) | transform (investigation findings -> structured record) | `.planning/phases-archive/01-site-ephemeris-helper/01-CONTEXT.md` (structure/tone) + `.planning/research/SUMMARY.md` (evidence-and-recommendation format) | role-match |
| `docs/design/eso_feasibility_spike.rst` | durable design doc (RST) | transform (decision doc -> durable summary) | `docs/design/telescope_runs_calendar.rst` | exact |
| Cerro Paranal `Observatory` record (obscode `309`) — discretionary, D-07 | model/data-record | CRUD (create) | `solsys_code/solsys_code_observatory/migrations/0002_observatory_timezone_seed.py` (migration path) or `solsys_code/solsys_code_observatory/views.py:CreateObservatory` (interactive-form path) | exact (two viable analogs; pick per how the record is created) |

## Pattern Assignments

### `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` (planning-doc, full detail)

**Analog:** `.planning/phases-archive/01-site-ephemeris-helper/01-CONTEXT.md` (for structure/section discipline) and `.planning/research/SUMMARY.md` (for evidence -> recommendation framing, since this doc's job is exactly that: "here's what we found, here's the recommendation").

**Structure pattern** (from `01-CONTEXT.md` lines 1-17):
```markdown
# Phase 1: Site & Ephemeris Helper - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers ... [one-paragraph scope statement]

</domain>

<decisions>
## Implementation Decisions

### [Grouping heading]
- **D-01:** ... [locked decision, concrete and testable]
```
Adapt this heading/ID discipline for 13-DECISION.md: use `## Findings` (one
subsection per ESO-0N requirement: connection, OB status/execution capture,
credential/environment situation, Bridge vs Bypass assessment, effort
sizing), each finding written as a locked, falsifiable statement the way
`01-CONTEXT.md`'s `D-0N` items are written — not vague prose. Per D-04, any
verbatim API response block must have credential-adjacent fields redacted
before being pasted in (e.g. replace usernames/program IDs with `<REDACTED>`
placeholders, keep the surrounding JSON/dict structure intact so the shape is
still evidence).

**Evidence-then-recommendation framing** (this repo's closest precedent for
"investigate, then state a recommendation with confidence level" is
`.planning/research/SUMMARY.md`, an executive-summary doc built the same
way: state findings, then a scoped recommendation). Read that file directly
for its exact tone/section ordering (executive summary -> phase rationale ->
confidence assessment -> gaps) and mirror it inside 13-DECISION.md's Bridge/
Bypass/Not-Yet-Feasible recommendation section, per D-11's effort-sizing
requirement (name the methods needing real implementations —
`get_observation_status()`/`get_observation_url()`/`data_products()` — and
size the change as small patch / moderate fork / larger undertaking).

---

### `docs/design/eso_feasibility_spike.rst` (durable design doc, RST)

**Analog:** `docs/design/telescope_runs_calendar.rst` — this is the exact
precedent named in CONTEXT.md's canonical_refs ("durable summary...
alongside `docs/design/telescope_runs_calendar.rst`").

**Document skeleton pattern** (lines 1-44):
```rst
Telescope Runs on the Calendar
==============================

This document records the feasibility study and implementation plan for ...
It was written after a research spike (2026-06-10) that validated the
astronomy and the data model end-to-end.

Background
----------

FOMO coordinates follow-up of Solar System targets across several telescopes.
...

Key finding
-----------

**The feature is feasible with no changes to** ``tom_calendar`` **and no
database migrations.** ...

The Data Model
--------------

... list-table with headers ...
```

Reuse directly for `eso_feasibility_spike.rst`:
- Title = underlined with `=` matching title length (RST convention seen at
  lines 1-2).
- Opening paragraph: what spike this is, when it happened, what it validated
  — same "This document records..." framing.
- `Background` section: ESO Phase 2 scheduling model (VLT/Paranal + NTT/La
  Silla), why FOMO cares (parallel to how `telescope_runs_calendar.rst`
  frames classical vs. queue scheduling).
- `Key finding` section: **bold** one-line verdict up front (Bridge / Bypass
  / Not Yet Feasible), same emphasis convention as
  `` **The feature is feasible...** `` at line 35.
- A `list-table` section (mirror "The Data Model" table at lines 49-80) for
  the OB status vocabulary / method-availability matrix
  (`get_observation_status`, `get_observation_url`, `data_products` — which
  ones work today, which need patching).
- Since `docs/design/gsd_experiment.rst` and `docs/design/design.rst` are
  the other docs in this directory, keep consistent top-level heading style
  (`=` for title, `-` for sections) across all of them — do not invent a new
  heading convention.

**No error handling / no imports / no tests to extract** — this file has no
executable code; only structural/prose conventions apply.

---

### Cerro Paranal `Observatory` record (discretionary, D-07)

**Two viable analogs, pick based on how the record actually gets created:**

**Analog A — data migration path:** `solsys_code/solsys_code_observatory/migrations/0001_initial.py` (schema) plus the intended seed pattern described in `.planning/phases-archive/01-site-ephemeris-helper/01-01-PLAN.md` lines 77-98 (Task 1). Note: despite that plan's Task 1 describing a `RunPython(seed_observatories, unseed_observatories)` step, the actually-committed `0002_observatory_timezone_seed.py` (read directly, lines 1-17) contains **only** the `AddField` operation — no `RunPython` seed function exists in this migration. Treat the RunPython-seed description in the archived plan as an intended-but-not-final approach; if a migration-based seed is chosen for Paranal, model it on Django's standard `RunPython(seed, reverse)` idiom (`apps.get_model(...)`, `update_or_create(obscode=..., defaults={...})` for forward, `filter(obscode=...).delete()` for reverse) rather than copying `0002` verbatim, since `0002` itself doesn't contain that idiom.

**Analog B — interactive form path:** `solsys_code/solsys_code_observatory/views.py:CreateObservatory` (lines 17-42+). This is the "real" production path for adding an Observatory today: a `CreateView` backed by `CreateObservatoryForm` that takes an MPC obscode, queries the MPC Obscodes API via `MPCObscodeFetcher`, and calls `MPCObscodeFetcher.to_observatory()` to build the record — explicitly bypassing the superclass's `form_valid()` to avoid an `IntegrityError` on duplicate creation (see docstring lines 36-42). If the Paranal record is created ad hoc via `./manage.py shell` during the spike (most likely, since this is throwaway investigation infra), the shell one-liner should call the same `MPCObscodeFetcher(...).to_observatory()` path (or `Observatory.objects.get_or_create(obscode='309', defaults={...})` with the coordinates looked up from the MPC API) rather than hand-rolling field values, to match how every other Observatory record in this DB was actually populated.

**Recommendation:** since this is a discretionary, non-committed-to-VCS DB
row (not a migration deliverable per CONTEXT.md's own framing — "in scope
if needed to support the investigation," not a hard requirement), prefer
Analog B (shell one-liner via `MPCObscodeFetcher`) over authoring a new
migration. Only add a migration if the decision doc's evidence-gathering
genuinely depends on Paranal existing across DB resets/CI (unlikely for a
spike).

---

## Shared Patterns

### Redaction of captured API responses (D-04)
No existing code pattern for this in the repo (no prior phase has captured
real vendor API payloads into a committed doc). Apply a simple convention
inline in `13-DECISION.md`: paste the real JSON/dict response, replace any
`username`, `email`, or `progId`-like sensitive fields with
`<REDACTED>` and note next to the block "credential-adjacent fields redacted
per D-04." Keep structural shape (keys, nesting, list lengths) faithful —
that's the whole point of D-04, the shape is the evidence.

### Facility-agnostic sync landing point (context only, not touched)
`solsys_code/calendar_utils.py:insert_or_create_calendar_event()` and
`solsys_code/management/commands/sync_gemini_observation_calendar.py`'s
synthetic-key precedent (`GEM:{prog}/{observation_id}`) are referenced in
CONTEXT.md as the pattern a *future* ESO sync command would follow
(`ESO:{p2_environment}/{obId}`). Not implemented this phase — cite them in
13-DECISION.md's ESO-05 sketch as the landing point/precedent, but do not
write code against them.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Investigation script (Django shell session / scratch `.py`) | N/A — explicitly not a deliverable | N/A | Per CONTEXT.md D-09, this is exploratory-only and not committed; no pattern mapping needed or wanted. |

## Metadata

**Analog search scope:** `.planning/phases-archive/`, `.planning/research/`,
`docs/design/`, `solsys_code/solsys_code_observatory/` (models, views,
migrations)
**Files scanned:** `01-CONTEXT.md`, `01-01-PLAN.md`, `telescope_runs_calendar.rst`,
`gsd_experiment.rst`, `design.rst`, `0001_initial.py`, `0002_observatory_timezone_seed.py`,
`solsys_code_observatory/views.py`
**Pattern extraction date:** 2026-07-01
</content>
