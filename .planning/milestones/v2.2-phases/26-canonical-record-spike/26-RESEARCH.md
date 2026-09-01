# Phase 26: Canonical-Record Spike - Research

**Researched:** 2026-07-27
**Domain:** Investigation methodology for a throwaway-evidence Django spike (no shippable
code) — how to safely mutate a scratch copy of the dev DB, measure a model rename's blast
radius, and produce two durable decision artifacts
**Confidence:** HIGH (nearly every claim below was verified by direct execution against this
repo's own installed Django 5.2.13 in this session, not inferred from documentation)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase Boundary:** This phase is investigation-only, following the Phase 13 (ESO) and Phase 18
(uncertain-scheduling) precedents. It settles the four questions milestone questioning
deliberately left open — the `source` vocabulary and its interaction with `CampaignRun`'s two
existing partial unique constraints (SPIKE-01), how each ingest adapter's existing
calendar-event identity key maps onto a run (SPIKE-02), the canonical reconciler event-key
scheme and the stage-2 class-wide fan-out question (SPIKE-03), and the migration and
attribution strategy including the companion-record rename checklist (SPIKE-04). The
deliverables are a decision doc plus a durable `docs/design/` page. **No schema migration, no
reconciler, no attribution UI ships from this phase** — Phases 27, 28 and 29 consume the
decisions. Any migration or model code written during the spike is throwaway, git-excluded, and
discarded when the phase closes.

**Already locked — this phase executes these, it does not re-open them:**
- The spike blocks everything; Phases 27-29 are gated on it.
- `related_name='telescope_label_meta'` is **not** renamed. Only the model class is.
- No new dependencies (`django-dirtyfields`/`FieldTracker`, `django-fsm`, Celery, `rapidfuzz`,
  `GenericForeignKey` all explicitly rejected by research).
- New reconciler logic will live in `solsys_code/campaign_reconciler.py`, a peer of
  `campaign_gap.py`/`campaign_utils.py` — never a private helper in `campaign_views.py`, and
  never importing `solsys_code.views` or `solsys_code.ephem_utils` (the latter triggers a ~1.6
  GB SPICE kernel download at module load).
- The `run` FK is `null=True, blank=True, on_delete=SET_NULL`; no `RunPython` backfill.
- `source` stays out of both existing partial unique constraints — attribution, not the
  constraint, connects same-physical-run rows from different sources.
- No automatic merging of suspected duplicate associations; no upstreaming the event→run link
  into `tom_calendar`.

**Evidence standard — how the spike proves things:**
- **D-01:** The evidence vehicle is a throwaway migration applied to a copy of the real dev DB.
  Write the real `source`/`telescope_class` migration plus the companion-record rename on a
  scratch branch, apply it to a copy of `src/fomo_db.sqlite3`, record verbatim results in the
  decision doc, then discard. Follows Phase 13's git-excluded `eso_p2_probe.py` precedent.
  Phase 27 writes the real migration afterwards.
- **D-02:** The spike executes and measures the companion-record rename, it does not just
  enumerate it. On the throwaway branch: perform the rename, migrate the DB copy, run the full
  `./manage.py test solsys_code` suite, and load `/calendar/` in the dev server — then record
  which of the four integration points actually broke. **Analytical prediction to test, not
  assume:** because `related_name='telescope_label_meta'` is locked unchanged, the calendar
  template lookups and the view's `prefetch_related()` string are safe by construction — only
  the two class-name imports (`solsys_code/admin.py` and `solsys_code/management/commands/
  sync_lco_observation_calendar.py`) are actually at risk, and those fail loudly as
  `ImportError`. The spike should confirm or refute this rather than restating the checklist
  unexamined.
- **D-03:** The renamed model class is `CalendarEventMeta`. Chosen for generality — it absorbs
  `run`, `is_verified`, and whatever v2.3 adds, without needing a second rename. Reads naturally
  against the locked accessor (`event.telescope_label_meta` → a `CalendarEventMeta`).
  `CalendarEventRunLink` was considered and rejected: it misdescribes the 11 existing rows,
  which are pure telescope-label metadata with no run at all.
- **D-04:** Take a dated, git-excluded snapshot of `src/fomo_db.sqlite3` and pin every number in
  the decision doc to it (e.g. "as of 2026-07-27: 31 runs, 20 calendar events, 0 `CAMPAIGN:`
  events"). Record the PROJECT.md discrepancy (D-16) as an explicit finding so Phases 27-29 do
  not trust it, and open a separate todo to correct PROJECT.md. The spike itself stays
  investigation-only and does not edit PROJECT.md.

**Stage-2 class-wide semantics:**
- **D-05:** A class-wide run produces a single class-wide event per day (00:00–23:59, labelled
  with the class, no site) — not one event per candidate site. Grounded in measured cost:
  `CampaignRun` pk=29 (`LCO 1m`) is an 80-night window and `SITE_TELESCOPE_MAP` carries `1m0` at
  five sites, so fan-out would be 80 × 5 = 400 events for one run. Stage 3 narrows to the real
  site when an `ObservationRecord` appears.
- **D-06:** CANON-02's field widens from a telescope-class-only field to a "why is there no
  site" vocabulary covering telescope-class allocation, space mission, and unresolved/
  failed-to-resolve. Live data has five space-mission rows (pk=8, 12, 13, 21, 26) against two
  class-wide ones (pk=29, 30). The spike settles the vocabulary; Phase 27 implements it.
- **D-07:** A space-mission run gets one spanning event covering the whole window, not one event
  per day (e.g. pk=26 JUICE, 2025-11-02→11-25, becomes one 24-day event). Keeps the calendar
  consistent with v2.1's asset-aware `campaign_gap.claimed_dates()`.
- **D-08:** The spike defines an explicit stage 0 — "allocated but unscheduled." Runs with
  `window_start IS NULL` produce no calendar event but are counted and reported in the
  reconciler's summary, mirroring `import_campaign_csv`'s `site_needs_review` pattern.

**Canonical event key & ownership:**
- **D-09:** Identity and ownership are two separate mechanisms. Identity: the reconciler passes
  a namespaced `url` of the form `RUN:{run_pk}:{date}` as the `insert_or_create_calendar_
  event()` lookup, giving stage-stable idempotency. Ownership (RECON-05): the new companion
  `run` FK, as a hard rule — no companion row, or a companion row with `run=NULL`, means "not
  mine, never touch." `insert_or_create_calendar_event()` takes a caller-supplied `lookup` dict,
  so the reconciler is free to choose its own key without touching the shared helper.
- **D-10:** The `{date}` component is always the site-local observing night, derived in the
  site's timezone — never the UTC date of whatever `start_time` the current stage happens to
  produce. Stages 3 and 4 therefore change an event's times but never its key.
- **D-11:** The adopt-vs-gap-fill question is prototyped, not decided in the abstract. Against
  the real pk=1 case (15-night window 7–21 July; 11 of those nights already carry LCO events),
  build both on the throwaway DB copy and recommend one based on what the calendar actually
  looks like:
  - *Adopt* — update the attributed event in place, keeping its LCO-URL key, and mint
    `RUN:1:{date}` only for the 4 uncovered nights. 15 events. Biggest blast radius.
  - *Gap-fill* — create nothing for nights with an attributed adapter event; fill only the 4
    uncovered nights. Also 15 events, respects RECON-05 more strictly.
  The decision doc must also record the rejected baseline — reconciler always mints its own,
  giving 26 events for one run — and state explicitly that this is the visible double-booking
  ATTRIB-06 exists to prevent.

**`source` vocabulary & approval gating:**
- **D-12:** The 31 existing runs get a distinct `LEGACY` value. Nothing in the data can
  discriminate provenance. `LEGACY` is honest about what is actually known. Blanket
  `CSV_IMPORT` was rejected as an unverifiable assertion written into 31 rows.
- **D-13:** Phase 27 declares the full vocabulary — all five roadmap values plus `LEGACY` — with
  the three adapter values explicitly documented as not yet produced by any code path. Django
  `TextChoices` values are validation-only, so adding/removing them later is a no-op `AlterField`.
- **D-14:** Non-web runs keep `approval_status = APPROVED`; `source` is the disambiguator. The
  derivation rule — `APPROVED` and `source != WEB` means no approval was required — is recorded
  explicitly. A fourth `NOT_REQUIRED` approval value was considered and rejected.

**Measured findings that correct the planning docs** (verified against `src/fomo_db.sqlite3`
during CONTEXT.md's discussion — downstream agents should trust these over the corresponding
planning-doc statements):
- **D-15:** Zero `CAMPAIGN:`-namespaced calendar events in the dev DB. The 20 events are 9
  classical (blank `url`) and 11 LCO. The reconciler's key scheme has a clean slate.
- **D-16:** PROJECT.md's Phase 25 claim (pk=34 GS-2026A-FT-115) does not reproduce — max run pk
  is 31, no FT-115 row, no `CAMPAIGN:` events at all. The DB was re-imported after Phase 25's UAT.
- **D-17:** Zero `PENDING_REVIEW` runs (30 approved, 1 rejected). `import_campaign_csv.py:194`
  already writes `ApprovalStatus.APPROVED`; the importer's real behaviour change is writing
  `source`.
- **D-18:** Zero `GEM:`-namespaced events, so SPIKE-02's Gemini identity mapping can only be
  reasoned from code, not confirmed against real rows — state that confidence difference
  explicitly, following Phase 18's D-09 precedent of never conflating "confirmed against a real
  row" with "confirmed via constructed input."
- **D-19:** The 9 classical events have a blank `url`. SPIKE-02's per-adapter identity mapping
  must account for the classical adapter having no string identity key at all — keyed on
  `(telescope, instrument, start_time ± tolerance)` via `insert_or_create_calendar_event()`'s
  `start_time_tolerance` path.
- **D-20:** Confirmed exactly as the roadmap claims: `CampaignRun` pk=1 (FTS/MuSCAT4,
  2026-07-07→2026-07-21, site 7, campaign 2 "Didymos 2026", approved/observed); its 11 LCO queue
  events (ids 53-63, `2m0`/`COJ-2m0` + `2M0-SCICAM-MUSCAT`, 7–20 July, 8 `[EXPIRED]`, 1
  `[CANCELLED]`); 11 companion rows, all `is_verified=1`; and exactly 19 approved, site-resolved,
  windowed 3I/ATLAS runs with zero calendar presence. Also present: 13 LCO `ObservationRecord`s
  (4 COMPLETED, 8 WINDOW_EXPIRED, 1 CANCELED).

### Claude's Discretion

- Exact structure, wording and section ordering of the decision doc and the `docs/design/` page
  beyond what D-01..D-20 specify, and whether they are one document or two (Phase 13 used
  full-detail-plus-durable-summary; Phase 18 folded both into one).
- How to redact real 3I/ATLAS `contact_person`/`contact_email` values in any quoted evidence —
  carry forward Phase 18's D-01 posture: real people's names may be used to describe a finding,
  but email addresses and full name+email pairings must be omitted or redacted.
- Mechanics of the throwaway branch and DB copy (branch name, where the copy lives, how it is
  git-excluded) — Phase 13's `eso_p2_probe.py` is the precedent, not a prescription.
- How deep to take attribution-scoring prototyping against pk=1's 11 events beyond what D-11
  requires for the adopt-vs-gap-fill comparison.

**Folded Todos:** "Rename `calendar_utils.py` private helpers to reflect shared-module API" —
folded as a recommendation to record, not code to write (this phase ships no code). The decision
doc should state a recommended naming posture so Phase 27 can execute it as part of its own
work. Folding this here does not close the todo; it stays open until code actually lands.

### Deferred Ideas (OUT OF SCOPE)

- Correcting PROJECT.md's stale Phase 25 paragraph (D-16) — a separate todo, not this phase's
  work. The spike records the discrepancy as a finding; the doc fix happens outside the
  investigation-only boundary.
- Renaming `related_name='telescope_label_meta'` to match the new `CalendarEventMeta` class —
  explicitly out of scope per REQUIREMENTS.md; revisit only if a future phase has a reason to
  accept the silent-breakage risk.
- `CalendarEvent.url` non-uniqueness (noted in `23-REVIEW.md` as a pre-existing structural
  issue) — D-09 relies on the `url` key for identity while putting ownership on the companion
  FK, which sidesteps it. If a later phase wants `get_or_create` on `url` to be race-safe, that
  is its own change.
- v2.3 items confirmed still deferred and untouched by this discussion: status vocabulary
  unification (STATUS-01/02), adapter rewiring (ADAPT-01..03), provenance-blind gap analysis
  (GAPB-01), unused-allocation visualisation (UNUSED-01).
- Reviewed but not folded: "Extract site/telescope mapping and instrument extraction into own
  module" — already resolved by Phase 11's `calendar_utils.py` extraction; not relevant here.
</user_constraints>

## Project Constraints (from CLAUDE.md)

Directives extracted from `./CLAUDE.md` that bear on this phase, for the planner to verify
compliance against:

- **GSD workflow enforcement:** file-changing work must go through a GSD command
  (`/gsd:plan-phase`, `/gsd:execute-phase`, etc.) — already the context this research is
  produced within.
- **Serena MCP tools are primary for code reading** per this phase's own additional-context
  instructions. This research pass used direct `Read`/`Bash grep` (Serena's symbolic tools were
  not available in this session's toolset) — the executor should prefer `find_symbol`/
  `find_referencing_symbols` where available for the rename-verification work in item 2.
- **Heavy import side effect:** importing `solsys_code.ephem_utils` (transitively
  `solsys_code.views`) downloads ~1.6 GB of SPICE kernels on first use and runs
  `fomo_furnish_spiceypy()` at module load. This is the single most load-bearing constraint for
  this phase's D-02 measurement methodology — see "Measuring the Rename" above, which
  empirically corrects the naive mitigation (test-module narrowing does not avoid it).
- **Testing split:** DB-dependent tests belong in `solsys_code/tests/`, run via `./manage.py
  test solsys_code` (or a narrower dotted path) — `python -m pytest` does not collect them
  (`testpaths = ["tests", "src", "docs"]` in `pyproject.toml` excludes `solsys_code/`). This
  phase's throwaway test-selection work targets the Django test runner exclusively, never
  pytest.
- **Target test factories:** any fixture touching `Target` must use `tom_targets.tests.
  factories.NonSiderealTargetFactory`, never `SiderealTargetFactory`. Not directly exercised by
  this phase's own throwaway scripts (they query real, already-existing `CampaignRun`/
  `TargetList`/`Target` rows, they don't construct new `Target` fixtures) — flagged here only so
  the plan doesn't introduce a new fixture that violates it if a script needs one.
- **Planning-doc terminology:** plain English over DB jargon ("create or update"/"find-or-create",
  never "upsert") — followed throughout this document; the same convention should carry into the
  decision doc and `docs/design/` page.
- **Paired-docs rule (directory-scoped, `docs/runbooks/` + the notebook-pairing map):** confirmed
  **does not apply to this phase**. ROADMAP.md's own Phase 26 entry already records "None —
  investigation only, no module behaviour changes," and this research confirms that reading
  holds: every module the throwaway work touches (`solsys_code/models.py`, a new migration file)
  is edited only on the scratch branch and discarded before phase close (item 1's discard
  mechanic), so no module's *shipped* behavior changes as a result of this phase. Neither
  `docs/runbooks/telescope_runs_calendar.rst` nor any of the four paired demo notebooks needs an
  update from Phase 26's own deliverables.
- **Ruff style (single quotes, 120-col)** applies to the throwaway `models.py`/migration edits on
  the scratch branch too, for consistency with the rest of the codebase during the measurement
  window, even though none of it survives to be linted for real.

## Summary

This is not feature research — Phase 26 ships no code, so there is no library or pattern to
recommend. What the planner needs is a verified **investigation procedure**: how to point a
throwaway Django migration at a disposable copy of `src/fomo_db.sqlite3` without ever touching
the real dev DB or the working git history, how to measure the companion-record rename's real
blast radius without paying an unnecessary cost or risk, how to produce pasteable executable
evidence for the `IntegrityError` coexistence check, how to cheaply prototype the two
adopt-vs-gap-fill event-write shapes against the real `CampaignRun` pk=1 window, and how to
shape the two durable output documents so they match this repo's two existing spike precedents.

Every mechanical claim below was tested directly in this session against the live venv and a
disposable scratch DB/settings override — not reasoned from Django's documentation. The single
most load-bearing, non-obvious finding is that **`./manage.py test`'s test-module selection does
not control whether the ~1.6 GB SPICE-kernel-triggering `solsys_code.views` import happens** —
Django's own `DiscoverRunner` unconditionally runs `manage.py check` before any test executes,
and `check` imports the full URLconf regardless of which test label was given. What test
selection *does* control is whether the slow, crash-prone ephemeris-integration tests
(`test_ephem_utils.py`, `test_views.py`) actually run — and in this exact sandbox, running the
full `solsys_code` suite end-to-end **segfaulted** inside REBOUND/ASSIST partway through
`test_views.py`, well after the four rename-relevant integration points had already been
exercised cleanly. This directly shapes the recommended test-selection strategy for D-02 below.

**Primary recommendation:** Do the throwaway model/migration work on a dedicated scratch git
branch; point every `manage.py` invocation at a disposable copy of `src/fomo_db.sqlite3` via a
single gitignored `local_settings.py` at the repo root (empirically confirmed to override
`DATABASES` cleanly); prove SPIKE-01 and D-11 with small, verbatim-printing Python scripts run
via `manage.py shell`, not `assert`-based tests; measure D-02 with a **named, narrow list** of
test modules that excludes `test_ephem_utils.py`/`test_views.py` (this does not avoid the fixed
~3-4s SPICE-furnish cost, but it does avoid the segfault and the multi-minute wall time); and
follow Phase 13's full-detail-plus-durable-summary two-document shape, not Phase 18's
single-document shape, given the volume of verbatim executable evidence this phase produces.

## Architectural Responsibility Map

This phase ships no application code, so the usual tier map does not apply. The relevant
"tiers" are investigation infrastructure, not runtime architecture:

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| DB-copy mutation (throwaway migration apply) | Local dev tooling (`manage.py` + gitignored `local_settings.py`) | — | Never touches the API/DB tier that matters at runtime — the real `src/fomo_db.sqlite3` is never opened by any command in this phase |
| Model-rename risk measurement (D-02) | Local dev tooling (`manage.py test`, `manage.py runserver`) | Django/TOM view+template tier (read-only, exercised not modified) | The rename touches `solsys_code/admin.py`, a management command, `views.py`, and a template — all four are *read* during measurement, never shipped changed |
| `IntegrityError` coexistence evidence (SPIKE-01) | Local dev tooling (Django shell script against the DB copy) | Database tier (SQLite constraint enforcement) | The evidence *is* the SQLite constraint engine's own behavior against real rows |
| Decision recording | Documentation tier (`.planning/`, `docs/design/`) | — | The only artifacts that survive phase close |

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SPIKE-01 | `source` vocabulary settled + executable `IntegrityError` coexistence proof against real pk=1/11-LCO-events rows | "IntegrityError Coexistence Check" below gives the exact script shape, the two constraint names to target, and why no rollback is needed (unlike Phase 18's read-only precedent) |
| SPIKE-02 | Per-adapter identity-key-to-run mapping documented | Existing code confirmed: classical events have blank `url` (D-19), LCO events use the real portal URL, no `GEM:`/`CAMPAIGN:` events exist yet (D-15/D-18) — this research adds no new evidence here beyond confirming D-15/D-18/D-19 still hold today (re-verified live, see "Fresh DB Snapshot" below); the mapping itself is CONTEXT.md's job, already locked |
| SPIKE-03 | Canonical reconciler event-key scheme + stage-2 fan-out answer | "Adopt-vs-Gap-Fill Prototype Mechanics" below gives the cheapest way to build both variants against pk=1's real window and compare resulting event counts/keys |
| SPIKE-04 | Migration + attribution strategy, rename checklist | "Throwaway-Evidence Mechanics" gives the safe migration-authoring and DB-copy procedure; "Rename Blast-Radius Measurement" gives the verified four-integration-point checklist and the exact test/command set to prove or refute it |
</phase_requirements>

## Standard Stack

### Core

No new stack. This phase uses only what's already installed and already the project's own
convention:

| Tool | Version (installed, this session) | Purpose | Why Standard |
|------|---------|---------|--------------|
| Django | 5.2.13 | ORM, migrations, `manage.py shell`/`test`/`migrate`/`runserver`, ships its own ad-hoc-script execution surface | Already the project's framework; no reason to reach for anything else for a throwaway probe |
| SQLite3 (`django.db.backends.sqlite3`) | stdlib | The DB engine both the real dev DB and every scratch copy use | `cp`-able as a single file — this is *why* the "disposable copy" strategy works at all; would not be this simple on Postgres |
| `git` (scratch branch) | — | Isolates throwaway `models.py`/migration edits from the real branch history | Matches Phase 13's `eso_p2_probe.py` precedent, adapted for edits to *tracked* files (a script can just not be `git add`ed; a `models.py` edit needs an actual branch to discard cleanly) |

### Supporting

None. Explicitly do not add `pytest`/`pytest-django` for this phase's evidence scripts — the
project's own `pyproject.toml` `testpaths` deliberately excludes `solsys_code/` from pytest
collection (CLAUDE.md), and Django's own `manage.py shell -c` / a short script executed via
`exec()` inside the shell is sufficient and matches Phase 13/18's own "throwaway script, not a
committed dependency or test file" precedent.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A single gitignored `local_settings.py` DATABASES override | `--settings=some_scratch_settings_module` CLI flag per command | Functionally equivalent, but has to be repeated on every `manage.py` invocation (`migrate`, `test`, `runserver`, `shell`) instead of being picked up automatically; `local_settings.py` is also already a first-class, already-gitignored mechanism this project's own `settings.py` explicitly supports (`from local_settings import *` at the file's end) — no reason to invent a second one |
| Hand-authored throwaway migration | `manage.py makemigrations` (autodetected) | STACK.md already documents why autodetection is unsafe here: Django cannot tell "renamed" from "deleted + created" from a field diff alone in non-interactive mode, and would emit `DeleteModel`/`CreateModel` instead of `RenameModel` — which, because the OneToOne's `event_id` is the model's actual primary key, would drop and recreate the table and lose every row on the scratch copy, invalidating the very coexistence evidence this phase needs |
| Two separate scratch DB copies for the D-11 adopt/gap-fill comparison | One copy, reset via a savepoint/rollback between the two scenarios | A second `cp` of a ~950 KB SQLite file costs nothing and gives a clean, independently-inspectable before/after pair for each scenario — simpler than tracking a rollback boundary inside one file |

**Installation:** None required — nothing new is installed. `no packages are added to
`pyproject.toml`` per the CONTEXT.md "Locked constraints" (no new dependencies).

## Package Legitimacy Audit

**Not applicable.** This phase installs no external packages — CONTEXT.md's `<domain>` section
explicitly locks "No new dependencies" as an inherited milestone-level constraint, and nothing
in the investigation procedure below requires one (see "Alternatives Considered" above for why
`pytest`/`rapidfuzz`-style additions were considered and rejected). The Package Legitimacy Gate
protocol is therefore skipped for this phase; the planner does not need a `checkpoint:human-verify`
task for any install step.

## Investigation Methodology

This is the core of this phase's research — the five items the objective asked for, each
grounded in a command actually run against this repo in this session.

### 1. Throwaway-Evidence Mechanics (D-01, D-04)

**The DB path and settings-override mechanism, confirmed by direct inspection and a live test:**

- `src/fomo/settings.py:117-122` sets `DATABASES['default']['NAME'] = os.path.join(BASE_DIR,
  'fomo_db.sqlite3')` where `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
  — i.e. `BASE_DIR` is the repo's `src/` directory, so the real DB is
  `src/fomo_db.sqlite3` exactly as CLAUDE.md states. `[VERIFIED: direct file read]`
- `src/fomo/settings.py`'s very last statement is `try: from local_settings import * except
  ImportError: pass` — this runs *after* `DATABASES` is already assigned, so any name
  `local_settings.py` defines (including `DATABASES`) silently overwrites the value set above.
  `local_settings.py` is already gitignored (`.gitignore:61`, no path prefix so it matches
  wherever it's created) — this file is the project's own designed escape hatch for exactly
  this kind of local override, not something this phase invents. `[VERIFIED: direct file read]`
- **Where must `local_settings.py` live?** `manage.py` does no `sys.path` manipulation; when
  run as `./manage.py ...`, Python puts the script's own directory — the **repo root** — at
  `sys.path[0]`. The unqualified `from local_settings import *` therefore resolves against the
  repo root, not `src/`. Confirmed empirically this session: a `local_settings.py` written at
  the repo root with `DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME':
  '/tmp/probe-local-settings-override.sqlite3'}}` was picked up — `django.conf.settings.DATABASES
  ['default']['NAME']` printed the overridden path, not `src/fomo_db.sqlite3`. The probe file
  was deleted immediately after and `git status` shows no trace (gitignored). `[VERIFIED:
  empirical test run this session]`
- **One override file covers every command.** Because the override happens at Django settings
  load time, it applies uniformly to `manage.py migrate`, `manage.py test`, `manage.py shell`,
  and `manage.py runserver` — the same `local_settings.py` written once at the start of the
  spike session is what makes D-02's "load `/calendar/` in the dev server" step safe too: point
  `runserver` at the same scratch copy, no separate mechanism needed.

**Recommended concrete procedure** (Claude's Discretion per CONTEXT.md — this is a
recommendation, not a prescription):

1. `mkdir -p tmp` at repo root (already covered by `.gitignore:145`, `tmp/`, with no leading
   slash so it's ignored at any depth — nothing under it needs a separate ignore entry).
2. `cp src/fomo_db.sqlite3 tmp/26-spike-db-copy.sqlite3` — the disposable copy every command
   below operates on.
3. Write `local_settings.py` at the repo root (gitignored) with `DATABASES['default']['NAME']`
   pointing at the absolute path of `tmp/26-spike-db-copy.sqlite3`.
4. **Safety guard before any write command:** print `settings.DATABASES['default']['NAME']`
   (e.g. `python manage.py shell -c "from django.conf import settings; print(settings.DATABASES
   ['default']['NAME'])"`) and confirm it ends in `tmp/26-spike-db-copy.sqlite3`, **not**
   `src/fomo_db.sqlite3`, before running `migrate`. This is the single highest-value guard in
   the whole procedure — see Pitfall 1 below.
5. `git checkout -b spike/26-canonical-record-probe` (or similar; branch name is Claude's
   Discretion per CONTEXT.md) off the real phase-26 working branch. Make the throwaway edits on
   this branch only:
   - Rename `CalendarEventTelescopeLabel` -> `CalendarEventMeta` in `solsys_code/models.py`
     (D-03; keep `related_name='telescope_label_meta'` and the `event` field name unchanged —
     see "Rename Blast-Radius Measurement" below for why).
   - Add `run = models.ForeignKey('solsys_code.CampaignRun', null=True, blank=True,
     on_delete=models.SET_NULL, related_name='calendar_links')` to the renamed model — needed
     so the D-11 adopt-vs-gap-fill prototype has something concrete to write ownership into
     (SPIKE-03 needs this proven, not just SPIKE-04's narrower "did the rename break anything"
     question — see the migration skeleton in Code Examples).
   - Add `source`/`telescope_class` to `CampaignRun` (TextChoices per D-12/D-13's vocabulary —
     see Code Examples for the exact skeleton).
   - Hand-author the migration (`solsys_code/migrations/0008_*.py`, next free number confirmed
     by `ls solsys_code/migrations/` this session — `0007_campaignrun_contact_public_opt_in.py`
     is the current head). Do **not** run `makemigrations` non-interactively for the rename step
     — see "Alternatives Considered" above.
6. `python manage.py migrate solsys_code` (with the `local_settings.py` override active — step
   4's guard confirms this) applies the migration to `tmp/26-spike-db-copy.sqlite3` only.
7. Run every evidence-gathering step below against this copy. Because the whole DB file is
   disposable, evidence scripts can **write for real and commit the transaction** — unlike
   Phase 18's read-only precedent (which wrapped every `resolve_site()` call in a rolled-back
   `transaction.atomic()` because it ran against the *real* `Observatory` table), there is no
   need to roll anything back here. This is a genuinely different evidence posture from Phase
   18's, worth stating explicitly in the decision doc so a future reader doesn't assume every
   spike uses the rollback pattern.
8. **Discard, explicitly:** `git checkout <real-phase-26-branch>` (never merge the scratch
   branch), `git branch -D spike/26-canonical-record-probe`, `rm -f local_settings.py`, `rm -rf
   tmp/`. Run `git status --porcelain` on the real branch afterward and confirm it shows nothing
   from this procedure — only the two documentation deliverables should ever be staged.

**Why a branch, not just "don't `git add` it" (Phase 13's script precedent doesn't directly
apply):** Phase 13's `eso_p2_probe.py` was a *new, untracked* file — simply never staging it was
enough. Phase 26's throwaway work edits an *already-tracked* file (`solsys_code/models.py`) and
adds a new file to an already-tracked directory (`solsys_code/migrations/`). Working on a scratch
branch and never merging it is the clean way to guarantee the real phase-26 branch's `git diff`
never contains the throwaway rename/fields — "don't stage it" isn't sufficient once `models.py`
itself has uncommitted throwaway edits sitting in the working tree of the branch you're about to
commit the decision docs from.

### 2. Measuring the Rename (D-02)

**Confirmed integration points, by grep, matching D-02's own analytical prediction and
`STACK.md`'s existing table exactly — no additional consumer was found this session:**

| # | File:line | What it references | Breaks on class rename? |
|---|-----------|---------------------|--------------------------|
| 1 | `solsys_code/admin.py:4,28,41` | `from solsys_code.models import CalendarEventTelescopeLabel`; class + `admin.site.register(...)` | **Yes** — `ImportError`/`AttributeError` at Django startup |
| 2 | `solsys_code/management/commands/sync_lco_observation_calendar.py:18,369` | `from solsys_code.models import CalendarEventTelescopeLabel`; `.objects.update_or_create(event=event, defaults={'is_verified': ...})` | **Yes** — same, at command import |
| 3 | `solsys_code/views.py:114` | `.prefetch_related('telescope_label_meta')` | **No** — this is the FK's `related_name` string, untouched by a class-name-only rename |
| 4 | `src/templates/tom_calendar/partials/calendar.html:228,244` | `{% if event.telescope_label_meta.is_verified == False %}` | **No** — same reason as #3 |

`[VERIFIED: grep -rn "CalendarEventTelescopeLabel" solsys_code --include=*.py, excluding
tests/migrations, this session — exactly these two non-`models.py` files reference the class
name directly]`. `campaign_views.py` does **not** import `CalendarEventTelescopeLabel` at all
(confirmed by the same grep) — it never writes the companion row directly, so D-02's
"four integration points" is complete and no fifth site exists.

**This directly confirms D-02's analytical prediction**: only #1 and #2 are genuinely at risk,
and both fail loudly (`ImportError`) rather than silently. #3/#4 are safe *by construction*
because `related_name` and the OneToOne field name (`event`) are both explicitly kept unchanged
per D-03/CONTEXT.md. The spike's job per D-02 is to *confirm this by actually running the
rename*, not just restate the prediction — the procedure below does that.

**The load-bearing, corrected finding for this section — test-module selection does NOT avoid
the SPICE-triggering import:**

`CLAUDE.md` warns that importing `solsys_code.ephem_utils` (transitively, `solsys_code.views`)
downloads ~1.6 GB of SPICE kernels to `~/.cache/sorcha/` (a symlink to `~/.cache/layup/`,
confirmed 1.8 GB already present on this machine — `[VERIFIED: du -sh ~/.cache/layup]`) on first
use, and that `manage.py test` collecting `solsys_code/tests/` pays this cost. The natural next
question — "can a narrower test selection avoid it?" — was tested directly, three ways, this
session:

1. **Building** a test suite for a single unrelated module (`solsys_code.tests.test_admin`) via
   Django's own `DiscoverRunner.build_suite()` API does **not** import `ephem_utils`/`views` —
   confirmed by checking `sys.modules` immediately after (`False`/`False`). `[VERIFIED]`
2. **Actually running** that same narrow selection via `DiscoverRunner.run_tests()` **does**
   import both — confirmed the same way (`True`/`True`), even though `test_admin.py` never
   references `views`/`ephem_utils` and even for a module (`test_campaign_models.py`) that has
   **no** `self.client`/`reverse()` usage at all. `[VERIFIED, tested twice with two different
   modules]`
3. **Root cause, confirmed by source inspection**: `django/test/runner.py`'s
   `DiscoverRunner.run_checks()` unconditionally calls `call_command('check', ...)` —
   this is not gated by which test label was passed, and is not skippable via a
   `--skip-checks`-style flag on the `test` command (`django.core.management.commands.
   test.Command.requires_system_checks = []` disables *only* the generic `BaseCommand.execute()`
   check call; `DiscoverRunner` runs its own, separate, unconditional `check` regardless).
   `manage.py check` validates the URL configuration (`django.core.checks.urls`), which imports
   the full `ROOT_URLCONF` tree — and `fomo.urls` transitively includes a URL pattern that
   resolves to `solsys_code.views`, triggering the SPICE furnish at that module's import time.
   `[VERIFIED: direct source read of the installed django/test/runner.py and django/core/
   management/commands/test.py, cross-checked against the empirical result above]`

**Practical conclusion for D-02:** there is no `manage.py test <narrow-selection>` invocation
that avoids the fixed ~3-4s furnish cost measured this session (`manage.py check` alone, timed
separately: 3.49s beyond a 1.48s baseline `django.setup()`, with the SPICE cache already warm).
On a machine/session where the cache is **not** warm, this same `check` step would instead pay
the real multi-minute ~1.6 GB download CLAUDE.md warns about — this is a one-time,
machine-level cost (the cache persists across runs, keyed by `~/.cache/sorcha` -> `~/.cache/
layup`), not a per-invocation one, but the spike's plan should not assume every execution
environment starts warm.

**What test selection *does* control — and why it still matters — is the segfault risk:**
running the **full** `./manage.py test solsys_code` suite end-to-end this session (on the
current `issue37-telescope-runs-calendar` branch, unrelated to any Phase 26 change) crashed with
`Fatal Python error: Segmentation fault` inside `assist.extras.integrate_or_interpolate` during
`test_views.py::test_K93` — a REBOUND/ASSIST N-body integration call inside the *ephemeris*
view's own test, not anything Phase 26 touches. `[VERIFIED: full run, timed out at 2 min on a
first attempt, completed to the crash on a second attempt with an 8-min budget; the real
`src/fomo_db.sqlite3` mtime was unchanged before/after, confirming Django used its own isolated
test database and the crash never touched real data]`. Several hundred tests (dots) had already
passed by the time of the crash, including — alphabetically before `test_views.py` — every
module relevant to the four rename integration points. This means:

- The full-suite run is **slow** (timed out past 2 minutes even before the crash) and, in this
  exact sandbox, **unreliable** (segfaults) for reasons that have nothing to do with the rename.
- A **named, narrow module list** gets the same rename-relevant evidence, faster, without the
  crash risk:

  ```
  ./manage.py test solsys_code.tests.test_admin \
                    solsys_code.tests.test_sync_lco_observation_calendar \
                    solsys_code.tests.test_calendar_template \
                    solsys_code.tests.test_campaign_models \
                    solsys_code.tests.test_campaign_views \
                    solsys_code.tests.test_campaign_approval
  ```

  `test_admin.py` exercises integration point #1 end-to-end (`self.client.get(reverse
  ('admin:solsys_code_calendareventtelescopelabel_changelist'))` — this URL name itself is
  derived from the model's *table*, so renaming the model class changes this reverse-URL name
  too; the test's own assertions will surface that as a loud failure if the test isn't also
  updated, which is exactly the kind of consumer this measurement should catch). `test_sync_
  lco_observation_calendar.py` exercises #2 (its sidecar-write assertions call
  `CalendarEventTelescopeLabel.objects.update_or_create` by name today — the test file itself
  needs the same rename applied to compile, which is useful confirming evidence in its own
  right). `test_calendar_template.py` is the load-bearing one for #3/#4: it calls
  `self.client.get(reverse('calendar:calendar'))` — the **same `solsys_code/views.py` module**
  that also holds the crash-prone `Ephemeris`/`MakeEphemerisView`, but `fomo_render_calendar`
  itself (the calendar view, DISPLAY-09's prefetch target) does no REBOUND/ASSIST integration —
  it is a plain queryset + template render. Exercising it pays the same fixed SPICE-furnish
  import cost as any other test in this list (unavoidable per the finding above) but never
  touches the code path that crashed. Its dashed-border/tooltip assertions are exactly the
  template-level check Pitfall 1 in `PITFALLS.md` calls out as the one consumer a Python-level
  `manage.py check` cannot catch on its own.
- **`test_ephem_utils.py` and `test_views.py` should be deliberately excluded** from this
  measurement's test-selection list — not because they'd catch something spurious, but because
  they add several minutes of unrelated ephemeris-integration runtime and, empirically this
  session, a crash, for zero additional evidence about the rename (their own content has nothing
  to do with `CalendarEventTelescopeLabel`).

**The `/calendar/` dev-server load** (the other half of D-02) is a separate, cheap manual step
once the migration is applied to the scratch copy and `local_settings.py` is pointed at it:
`python manage.py runserver` and load `/calendar/` in a browser, or — to get output that pastes
verbatim into the decision doc instead of a screenshot — `python manage.py shell -c` a
`django.test.Client().get('/calendar/?year=2026&month=7')` call and print `response.status_code`
plus a grep of the response content for `telescope_label_meta` occurrences. This reuses the same
Client mechanism `test_calendar_template.py` already uses, just outside the test framework, so
its output is a plain printed string rather than test-runner dots.

**Fresh DB snapshot (corroborates D-04/D-15..D-20, re-measured live this session against the
real, unmodified `src/fomo_db.sqlite3` — read-only, no writes):**

```
CampaignRun count: 31        (max pk: 31)
CalendarEvent count: 20      (9 blank-url classical, 11 https://observe.lco.global/... LCO)
CAMPAIGN: events: 0          GEM: events: 0
CalendarEventTelescopeLabel count: 11
CampaignRun pk=1: FTS/MuSCAT4, 2026-07-07..2026-07-21, site=E10 (Siding Spring-FTS),
                  campaign=Didymos 2026, approval_status=approved, run_status=observed
ObservationRecord count: 13
approval_status breakdown: 30 approved, 1 rejected, 0 pending_review
```

`[VERIFIED: direct read-only ORM query against src/fomo_db.sqlite3, this session, 2026-07-27 —
every figure matches D-04/D-15/D-17/D-20 exactly, one day after CONTEXT.md's own 2026-07-26/27
snapshot]`. This is useful evidence in its own right for D-04's date-pinning requirement: the
decision doc can now cite a second, independently-reproduced timestamp showing the dev DB has
not drifted between context-gathering and this research pass — worth noting explicitly rather
than silently re-deriving the same numbers.

### 3. The `IntegrityError` Coexistence Check (SPIKE-01)

**Recommended form:** a short, self-contained Python script executed via `manage.py shell`
against the scratch DB copy (once the migration from item 1 above is applied), whose **printed
output is the evidence** — plain pass/fail lines, not test-framework dots — matching Phase 13's
`eso_p2_probe.py` and Phase 18's `fuzzy_match_probe.py` precedent of "throwaway script, verbatim
output pasted into the decision doc." Unlike Phase 18's version of this pattern, no
`transaction.atomic()` + rollback is needed (see item 1's discussion of why) — the DB copy is
already disposable, so the script can write for real.

The check needs **two** halves to satisfy SPIKE-01's literal wording ("coexisting... with no
`IntegrityError` **and** no change to either existing partial unique constraint") — a positive
case (the new field doesn't break anything) and a negative control (the *old* constraints still
fire exactly as before, proving `source` was never added to them):

```python
# tmp/26_integrity_check.py -- run via: python manage.py shell < tmp/26_integrity_check.py
# (against the tmp/26-spike-db-copy.sqlite3 copy, per the local_settings.py override -- confirm
# settings.DATABASES['default']['NAME'] before running this.)
from django.db import IntegrityError, transaction
from solsys_code.models import CampaignRun

run1 = CampaignRun.objects.get(pk=1)
print(f'BEFORE: pk=1 source field value = {run1.source!r}')  # field default from the migration

# Positive case (SPIKE-01's literal check): CampaignRun pk=1 and its companion CalendarEvents
# already coexist in the DB by construction (they're real rows) -- the check that matters is
# that giving pk=1 an explicit `source` value, and giving its 11 attributed companion rows a
# `run` FK back to pk=1, does not raise.
try:
    with transaction.atomic():
        run1.source = CampaignRun.Source.LEGACY  # D-12
        run1.save(update_fields=['source'])
    print('PASS: CampaignRun pk=1 source=LEGACY saved, no IntegrityError.')
except IntegrityError as exc:
    print(f'FAIL: {exc}')

from solsys_code.models import CalendarEventMeta  # post-rename name
linked = 0
for companion in CalendarEventMeta.objects.filter(event__url__startswith='https://observe.lco.global'):
    try:
        with transaction.atomic():
            companion.run = run1
            companion.save(update_fields=['run'])
        linked += 1
    except IntegrityError as exc:
        print(f'FAIL linking companion {companion.pk}: {exc}')
print(f'PASS: linked {linked} companion rows to run pk=1 with no IntegrityError.')

# Negative control: the ORIGINAL resolved-window constraint must still fire, unmodified, on
# a genuine duplicate -- proving `source` was never added to its field list. This must still
# raise.
try:
    with transaction.atomic():
        CampaignRun.objects.create(
            campaign=run1.campaign, telescope_instrument=run1.telescope_instrument,
            window_start=run1.window_start, window_end=run1.window_end,
            source=CampaignRun.Source.CSV_IMPORT,  # deliberately a DIFFERENT source
        )
    print('FAIL (unexpected): duplicate-window insert with a different source succeeded -- '
          'the constraint may have silently absorbed `source` into its key.')
except IntegrityError as exc:
    print(f'PASS: unique_campaign_run_resolved_window still fires unmodified '
          f'(source is not in its key): {exc}')
```

The two constraint names to cite verbatim in the decision doc, confirmed from
`solsys_code/models.py:120-160`: `unique_campaign_run_resolved_window`
(`fields=('campaign', 'telescope_instrument', 'window_start', 'window_end')`, condition
`window_start__isnull=False`) and `unique_campaign_run_tbd_natural_key` (`fields=('campaign',
'telescope_instrument', 'contact_person')`, condition `window_start__isnull=True`) — neither
should ever appear with `source` or `telescope_class` added to its `fields` tuple in the
throwaway migration; the script's negative-control block is the executable proof of that.

**Because `source` is backfilled with a single static default (`LEGACY`, D-12) rather than a
per-row-inferred value, the throwaway migration itself doesn't need a `RunPython` step** —
`migrations.AddField('campaignrun', 'source', models.CharField(..., default=CampaignRun.
Source.LEGACY))` backfills all 31 existing rows in one step. This is a genuine simplification
over `PITFALLS.md`'s Pitfall 3 discussion (which worried about per-row inference) — D-12 already
settled that question in CONTEXT.md, so the migration mechanics are simpler than the milestone
research anticipated.

### 4. Prototyping Adopt-vs-Gap-Fill (D-11)

**Cheapest credible prototype:** two separate scratch DB copies (one per scenario — see
"Alternatives Considered" above), each with the throwaway migration applied, each exercised with
a short script that calls `insert_or_create_calendar_event()` (unchanged, imported straight from
`solsys_code.calendar_utils` — no new write-path code, per this repo's own "Don't Hand-Roll"
convention) with a different `lookup` key strategy for pk=1's 15-night window (2026-07-07
through 2026-07-21) against its 11 already-existing LCO-sourced `CalendarEvent`s (ids
confirmed 53-63 in D-20, url-prefixed `https://observe.lco.global/...`):

- **Adopt copy:** for each of the 11 already-covered nights, call `insert_or_create_calendar_
  event({'url': <that event's own existing url>}, fields={...})` — i.e. write into the event
  under its *existing* LCO-URL key, not a fresh `RUN:1:{date}` key. For the 4 uncovered nights
  (2026-07-08's window minus the 11... actually the specific uncovered dates fall out of
  `{window_start..window_end} - {the 11 LCO event dates}`, computed directly from the real
  rows), mint `RUN:1:{date}` per D-09. Expect 15 total events touching pk=1's window: 11
  updated-in-place (still under their original LCO url), 4 newly created under `RUN:1:{date}`.
- **Gap-fill copy:** touch nothing under the 11 existing LCO urls at all — only mint `RUN:1:
  {date}` for the same 4 uncovered nights. Expect 15 events too (11 untouched originals + 4 new
  `RUN:1:{date}` ones), but the 11 originals' `modified` timestamp and companion `run` FK are
  never written by this path.
- **Rejected-baseline control** (D-11 asks the decision doc to record this explicitly): a third
  copy where the reconciler mints its *own* `RUN:1:{date}` key for **every** one of the 15
  nights regardless of existing coverage — expect this to produce **26** events for one run (11
  pre-existing LCO-keyed + 15 fresh `RUN:1:{date}`-keyed), the visible double-booking scenario
  ATTRIB-06 exists to prevent. Running this third variant is cheap (same script, different
  branch) and gives the decision doc a concrete, counted "what we're avoiding" number instead of
  an assertion.

Each script should print a final per-copy summary: total `CalendarEvent` count in pk=1's date
window, the url-key of each, and (adopt copy only) whether the companion row's `run` FK was
set — exactly the shape of evidence the decision doc's D-11 section needs to show its
recommendation "based on what the calendar actually looks like," per CONTEXT.md's own framing.

## Architecture Patterns

### Recommended Scratch/Deliverable Layout

```
tmp/                                     # gitignored (.gitignore:145) -- created, used, deleted
├── 26-spike-db-copy.sqlite3             # SPIKE-01/D-02 scratch copy (migrated, mutated freely)
├── 26-adopt-copy.sqlite3                # D-11 adopt scenario
├── 26-gapfill-copy.sqlite3              # D-11 gap-fill scenario
├── 26-rejected-baseline-copy.sqlite3    # D-11 rejected-baseline control
├── 26_integrity_check.py                # SPIKE-01 script (Code Examples)
└── 26_reconciler_prototype.py           # D-11 script (three modes, one per copy)

local_settings.py                        # gitignored (.gitignore:61) -- repo root, DATABASES override

# On the scratch branch ONLY -- never merged, discarded whole:
solsys_code/models.py                    # throwaway rename + run FK + source/telescope_class
solsys_code/migrations/0008_*.py         # hand-authored, never makemigrations-autodetected

# The only two artifacts that survive phase close, on the REAL phase-26 branch:
.planning/phases/26-canonical-record-spike/26-DECISION.md   # full findings + verbatim evidence
docs/design/canonical_record_spike.rst                      # durable summary (new toctree entry)
```

### Pattern 1: `local_settings.py` as a universal DB-copy pointer

**What:** One gitignored file at the repo root, written once, read by every subsequent
`manage.py` invocation for the rest of the investigation session.
**When to use:** Any time a throwaway Django command needs to run against non-production data
without touching tracked settings.
**Verified this session** — see item 1 above for the exact confirmation.

### Pattern 2: Hand-authored migration, `RenameModel` before `AddField`

**What:** Author the throwaway migration by hand, in the order STACK.md's Django Pattern 2
already specifies for the *real* Phase 27 migration — `RenameModel` first, then `AddField` for
the new `run`/`source`/`telescope_class` fields, referencing the model under its **new**
(post-rename) name since Django's migration state is cumulative within one migration's
`operations` list.
**Why this matters for the spike specifically:** this is the exact migration shape Phase 27
will eventually write for real — prototyping it now on the scratch branch means the spike's
evidence transfers directly (the decision doc can say "this migration shape was proven against
real data," not just reasoned about).

```python
# solsys_code/migrations/0008_scratch_canonical_record_probe.py -- SCRATCH BRANCH ONLY, never
# committed to the real phase-26 branch.
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('solsys_code', '0007_campaignrun_contact_public_opt_in')]

    operations = [
        migrations.RenameModel(old_name='CalendarEventTelescopeLabel', new_name='CalendarEventMeta'),
        migrations.AddField(
            model_name='calendareventmeta',
            name='run',
            field=models.ForeignKey(
                null=True, blank=True, on_delete=django.db.models.deletion.SET_NULL,
                related_name='calendar_links', to='solsys_code.campaignrun',
            ),
        ),
        migrations.AddField(
            model_name='campaignrun', name='source',
            field=models.CharField(max_length=20, default='legacy', choices=[
                ('web', 'Web submission'), ('classical_file', 'Classical file'),
                ('lco_queue', 'LCO queue'), ('gemini_queue', 'Gemini queue'),
                ('csv_import', 'CSV import'), ('legacy', 'Legacy (pre-migration)'),
            ]),
        ),
        migrations.AddField(
            model_name='campaignrun', name='telescope_class',
            field=models.CharField(max_length=10, null=True, blank=True, choices=[
                ('2m0', '2m0'), ('1m0', '1m0'), ('0m4', '0m4'),
            ]),
        ),
    ]
```

### Anti-Patterns to Avoid

- **Running `manage.py migrate` without first printing `settings.DATABASES['default']['NAME']`:**
  the single highest-risk mistake in this whole procedure — see Pitfall 1.
- **Trusting `makemigrations` for the rename step:** already covered in "Alternatives
  Considered" — non-interactive autodetection cannot distinguish rename from delete+create.
- **Including `test_ephem_utils.py`/`test_views.py` in the D-02 measurement selection "to be
  thorough":** costs several minutes and, empirically this session, crashes — for zero
  additional rename-relevant evidence, since neither file references `CalendarEventTelescopeLabel`.
- **Wrapping the D-11/SPIKE-01 scripts in `transaction.atomic()` + rollback "to be safe," the
  way Phase 18 did:** unnecessary here — the whole DB file is disposable. Rollback-wrapping adds
  ceremony without adding safety when the target is already a throwaway copy.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update a `CalendarEvent` in the D-11 prototype | A new ad-hoc write helper for the prototype scripts | `insert_or_create_calendar_event()` (`solsys_code/calendar_utils.py:318`), imported unchanged | It already implements the exact no-churn create-or-update contract the prototype needs, and reusing it means the prototype's evidence transfers directly to what Phase 29's real reconciler will call |
| Verify the rename didn't break something | A new bespoke check script per integration point | The existing test suite (`test_admin.py`, `test_sync_lco_observation_calendar.py`, `test_calendar_template.py`) run against the renamed model on the scratch branch | These tests already assert the exact behaviors (admin changelist renders, sidecar `update_or_create` succeeds, dashed-border class appears) the rename must preserve — re-running them (with the rename's own name changes applied) is strictly better evidence than a new script that might miss an assertion the existing suite already encodes |
| Disposable-copy database isolation | A custom settings-swap shell wrapper or `--database` CLI plumbing | `local_settings.py`, already the project's own designed mechanism | Verified this session to work with zero new code |

**Key insight:** every piece of this investigation's *mechanics* — the DB engine's file-copy
disposability, the settings-override hook, the create-or-update helper, and the existing test
suite's own assertions — was already present in this repo before Phase 26 started. The spike's
job is to combine them safely, not to build new tooling.

## Common Pitfalls

### Pitfall 1: Forgetting the `local_settings.py` override is active and running `migrate`/a
write script against the real `src/fomo_db.sqlite3`

**What goes wrong:** every command in this procedure is a completely ordinary `manage.py`
invocation — nothing about the command line itself signals "this is pointed at a scratch copy."
If `local_settings.py` is deleted, forgotten, or the shell session running the commands doesn't
have it in its working directory, the exact same `migrate`/write commands silently operate on
the real dev DB — the one D-04 explicitly warns is "a moving target" and the one 3I/ATLAS's real
contact-PII rows live in.
**Why it happens:** the override is invisible at the call site; there is no `--database=scratch`
flag making the intent explicit in the command itself.
**How to avoid:** before every `migrate`/write step (not just once at the start), run the guard
one-liner from item 1 step 4 and visually confirm the printed path. Treat any command whose
`DATABASES['default']['NAME']` printout doesn't end in the scratch-copy filename as a stop
condition, not a warning.
**Warning signs:** `src/fomo_db.sqlite3`'s mtime changes during the spike session, or `git
status` on `src/fomo_db.sqlite3` (it's gitignored, so this wouldn't show in `git status`, which
is itself part of the danger — check the file's own mtime/size directly, e.g. `ls -la
src/fomo_db.sqlite3`, not `git status`).

### Pitfall 2: Assuming test-module narrowing avoids the SPICE-triggering import

**What goes wrong:** a plan or executor reasons "select only the four rename-relevant test
modules, and the ~1.6 GB SPICE cost is avoided." Empirically false this session — see
"Measuring the Rename" above. The cost (furnish time, not necessarily a fresh download if the
local cache is warm) is paid by `manage.py check`, called unconditionally inside
`DiscoverRunner.run_checks()`, regardless of test selection.
**How to avoid:** budget for the fixed furnish cost (confirm the local `~/.cache/sorcha`/
`~/.cache/layup` cache is warm before starting, or budget real minutes if it isn't) rather than
trying to engineer around it; use test-module narrowing for its *actual* benefit (avoiding the
slow, crash-prone ephemeris-integration tests), not for a cost-avoidance benefit it doesn't have.

### Pitfall 3: Including the ephemeris-integration test modules and hitting the segfault

**What goes wrong:** running the full `./manage.py test solsys_code` suite (or any selection
that includes `test_views.py`) risks the exact `Fatal Python error: Segmentation fault` observed
this session inside REBOUND/ASSIST, well past the two-minute mark, for reasons unrelated to the
rename.
**How to avoid:** use the named, narrow module list from "Measuring the Rename" above.
**Warning signs:** a `manage.py test` invocation that runs past ~2 minutes with no output beyond
dots — a strong signal it has reached `test_ephem_utils.py`/`test_views.py`.

### Pitfall 4: Trusting `makemigrations` to emit `RenameModel` for the throwaway migration

**What goes wrong:** already covered under "Alternatives Considered" and Pattern 2 above —
non-interactive autodetection would drop and recreate the table, losing every row on the scratch
copy including the 11 real companion rows the coexistence check needs.
**How to avoid:** hand-author the migration; never run `makemigrations` non-interactively for
this rename.

### Pitfall 5: The scratch branch's throwaway edits leaking into the real phase-26 branch

**What goes wrong:** if the throwaway `models.py`/migration edits are made directly on the real
phase-26 working branch (not a dedicated scratch branch) "to save a `git checkout`," an
uncommitted or half-committed throwaway edit can end up in the same `git add` sweep as the real
decision-doc deliverables.
**How to avoid:** the dedicated scratch-branch discipline from item 1 step 5/8; a final `git
status --porcelain` check on the real branch before considering the phase's deliverables ready,
confirming only the two documentation files are staged.

### Pitfall 6: Quoting real contact PII into the decision doc

**What goes wrong:** carried forward from Phase 18's D-01 posture (CONTEXT.md's Claude's
Discretion explicitly reapplies it here) — pasting a `CampaignRun.contact_email` value or a
full name+email pairing verbatim into a committed `.planning/` or `docs/` file.
**How to avoid:** the evidence scripts above never print `contact_person`/`contact_email` at
all — none of the SPIKE-01/D-11 evidence needs those fields. Real people's names (not paired
with email) may be used to describe a finding per the established convention; email addresses
never.

## Code Examples

See "Investigation Methodology" items 1, 3, and "Architecture Patterns" Pattern 2 above for the
`local_settings.py` override, the hand-authored migration skeleton, and the `IntegrityError`
check script — these are the load-bearing, directly-reusable artifacts for this phase's plan and
are written out in full there rather than duplicated here.

## State of the Art

| Old/Naive Approach | Verified Approach | When Corrected | Impact |
|---------------------|--------------------|-----------------|--------|
| "Select a narrow test-module list to avoid the SPICE download" | Narrow selection avoids the slow/crash-prone ephemeris-integration *tests*, but not the fixed `manage.py check`-triggered import cost — those are two different costs | This session, by direct empirical test against the installed Django 5.2.13 | The plan should budget for the fixed furnish cost regardless of selection, and use narrowing specifically to dodge the segfault risk, not the import cost |
| "Wrap every scratch-DB write in `transaction.atomic()` + rollback, mirroring Phase 18" | Unnecessary once the target is a fully disposable file copy — write for real, discard the file | This session, by reasoning through Phase 18's actual rationale (real-DB safety, not a general spike convention) | Simpler evidence scripts; explicitly worth noting in the decision doc so a future reader doesn't assume rollback is always required |

**Deprecated/outdated:** none — this is a fresh investigation with no prior implementation to
supersede.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `manage.py runserver`'s startup also runs Django's system checks (same `check` command path as `test`), so the same `local_settings.py`-override safety guard applies to it too | "Throwaway-Evidence Mechanics" item 1, step 4 | Low — this is long-standing, well-documented Django `runserver` behavior, not verified by direct execution this session (only `check`/`test` were run directly); if wrong, the practical effect is only that the guard step needs to be re-run once more before `runserver`, not a safety gap, since `runserver` uses the same `DATABASES` setting either way |

**All other claims in this research were verified by direct execution against this repo's
installed Django 5.2.13 in this session, or by direct inspection of this repo's own tracked
files** — no other claim needs user confirmation before the plan can proceed.

## Open Questions

1. **Exact uncovered-night dates for pk=1's D-11 prototype (4 nights, per D-11's own count)**
   - What we know: pk=1's window is 2026-07-07..2026-07-21 (15 nights) with 11 LCO events dated
     7-20 July (ids 53-63, 8 `[EXPIRED]`, 1 `[CANCELLED]`).
   - What's unclear: the exact 4 uncovered calendar dates depend on which of the 7-20 July dates
     the 11 events actually land on (some nights may have more than one LCO event, e.g. a
     replacement after an `[EXPIRED]` one) — this wasn't enumerated night-by-night in this
     research pass.
   - Recommendation: the D-11 prototype script itself should compute this directly from the real
     `CalendarEvent` rows (`{d for d in date_range} - {e.start_time.date() for e in events}`)
     rather than the decision doc hand-deriving it — cheap to compute exactly, no need to guess.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python venv (`~/venv/devel_fomo311_venv`) | Every command in this procedure | ✓ | Python 3.11, Django 5.2.13 | — |
| SQLite3 | DB engine, both real and scratch copies | ✓ (stdlib) | — | — |
| SPICE kernel cache (`~/.cache/sorcha` -> `~/.cache/layup`) | Any command that imports `solsys_code.views`/`ephem_utils` (unavoidable per "Measuring the Rename" above) | ✓, warm (1.8 GB present) | — | If cold on the actual execution machine: budget real minutes for a one-time ~1.6 GB download on the first `manage.py check`/`test`/`runserver` invocation of the session |
| `git` (scratch branch support) | Item 1's discard mechanic | ✓ | — | — |
| `sphinx-build` (for verifying the new `docs/design/` page builds clean before commit) | Item 5 / SPIKE-04's docs deliverable | ✓ (pre-commit hook already runs it) | — | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build`, the same invocation pre-commit uses (`.pre-commit-config.yaml`), run manually before committing |

**Missing dependencies with no fallback:** none.

## Validation Architecture

This phase ships no application code, so validation here means **how each claim in the decision
doc gets evidenced** — an executable check's printed output, a real test-suite run's pass/fail
result, or an explicit manual step — not application test coverage. Every runtime number below
was measured directly this session against the live venv, not estimated.

### Test Infrastructure

This repo has **two independent test suites** (CLAUDE.md); this phase's own throwaway evidence
scripts use neither directly (they run via `manage.py shell`, per Investigation Methodology
items 1/3/4), but D-02's measurement step runs the second one, and the planner needs both
characterized:

| Property | pytest suite | Django (`solsys_code`) suite |
|----------|--------------|-------------------------------|
| Config | `pyproject.toml` `[tool.pytest.ini_options]`, `testpaths = ["tests", "src", "docs"]` — does **not** collect `solsys_code/` | Django settings module `src.fomo.settings` (no separate test-settings file; the test DB is Django's own isolated in-memory SQLite DB regardless of `DATABASES['default']['NAME']` — confirmed this session: no `test_*.sqlite3` file ever appeared on disk across every run) |
| Scope relevant to Phase 26 | None — the whole suite is 1 test, unrelated to `solsys_code` (confirmed: `find tests/ src/ docs/ -name test_*.py` -> 1 file) | All of it — the four rename integration points live in `solsys_code/` |
| Quick-run command | `python -m pytest` | `./manage.py test solsys_code.tests.test_calendar_template` (single most load-bearing module — covers integration points #3/#4, the two "safe by construction" ones the spike must actually confirm, not just #1/#2) |
| Quick-run measured time | **0.6s wall** (0.08s test time) `[VERIFIED, this session]` | **21.7s wall** (12.7s test time, 24 tests; the ~9s gap is the fixed `manage.py check`→SPICE-furnish cost from "Measuring the Rename" above, paid once per process) `[VERIFIED, this session]` |
| Full relevant-suite command | `python -m pytest` (same command — the suite is already this small) | The **named, narrow six-module list** from "Measuring the Rename" above (excludes `test_ephem_utils.py`/`test_views.py`) |
| Full relevant-suite measured time | 0.6s wall | **2m 5.6s wall** (114.8s test time, 242 tests, all passing) `[VERIFIED, this session — this number corrects the "completes in seconds" framing in the "Measuring the Rename" narrative above: the accurate figure for the full six-module list is ~2 minutes, not seconds; the single-module quick-run above (~22s) is the genuinely fast per-edit smoke check]` |
| **Literal** `./manage.py test solsys_code` (not recommended for this phase — see Pitfall 3) | N/A | Unsafe: timed out past 2 minutes on a first attempt, then **segfaulted** inside REBOUND/ASSIST on a second attempt (8-minute budget) before reaching a normal exit. Never run this for D-02's measurement; use the narrow list. |

### Evidence Map (ROADMAP success criteria + SPIKE-01..04)

| Criterion | Requirement | Evidence Artifact | Command | Confidence Level |
|-----------|-------------|--------------------|---------|-------------------|
| 1 | SPIKE-01 (`source` vocabulary + `IntegrityError` coexistence) | `tmp/26_integrity_check.py`'s printed PASS/FAIL lines (Investigation Methodology item 3) | `python manage.py shell < tmp/26_integrity_check.py` against the scratch DB copy | **Confirmed against real rows** — `CampaignRun` pk=1 and its real 11 LCO-sourced companion rows |
| 2 | SPIKE-02 (per-adapter identity-key-to-run mapping) | The Fresh DB Snapshot query (Investigation Methodology item 2) for classical (blank `url`) and LCO (real portal `url`) rows; a source-code read of `sync_gemini_observation_calendar.py`'s `GEM:{prog}/{obsid}` key-construction for the Gemini case | `python manage.py shell -c "..."` **read-only** against the real `src/fomo_db.sqlite3` (no scratch copy needed — no write) for classical/LCO; direct source inspection for Gemini | Classical and LCO: **confirmed against real rows**. **Gemini's `GEM:` mapping is confirmed via constructed input / code reading only** — D-18 already establishes zero real `GEM:`-namespaced events exist in the dev DB, so this one row of the evidence map can never move to "confirmed against real rows" without a real Gemini sync having run; the decision doc must state this distinction explicitly per Phase 18's D-09 precedent, not present it with the same confidence as the classical/LCO rows |
| 3 | SPIKE-03 (canonical event-key scheme + stage-2 fan-out) | The three-copy adopt/gap-fill/rejected-baseline event-count comparison (Investigation Methodology item 4) | The three `tmp/26_reconciler_prototype.py` runs, one per scratch copy | **Confirmed against real rows** for the adopt-vs-gap-fill comparison itself (real pk=1 window, real 11 LCO events). The stage-2 80×5=400 fan-out arithmetic (D-05) is a **computed figure from real field values** (`CampaignRun` pk=29's real window length × `SITE_TELESCOPE_MAP`'s real site count) rather than an executable DB check — worth stating that distinction too, since it is not literally "run something and observe an outcome" |
| 4 | SPIKE-04 (migration + rename checklist) | (a) the throwaway migration applying cleanly to the scratch copy with no error; (b) the narrow six-module Django test-suite run's real pass/fail output; (c) the `/calendar/` dev-server manual load below | (a) `python manage.py migrate solsys_code`; (b) the narrow-list command from the Test Infrastructure table; (c) manual, see below | (a)/migration-applies: **confirmed against real rows** (the scratch copy is a real copy of the dev DB). (b)/test-suite pass: **confirmed via constructed input** — Django's `TestCase` machinery builds its own isolated in-memory test DB from factories/fixtures, not from the scratch copy of real data, so a green run proves the rename doesn't break the *tested behaviors*, not that it was proven against the real 11 companion rows specifically (that proof is (a) + the SPIKE-01 script instead). (c): manual, see below |
| 5 | Durable `docs/design/` page | `sphinx-build` clean-build check + toctree entry present | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` (the exact pre-commit invocation, run manually before staging) | Not evidence of a decision — a build-tooling quality gate confirming the page is reachable and RST-valid, not a factual claim needing a confidence tier |

### Sampling Guidance — what to re-run after each investigation step

The goal is to catch a broken scratch copy or a leaked `local_settings.py` override immediately,
not at phase close:

1. **Immediately after writing/refreshing `local_settings.py`:** run the DB-path guard one-liner
   (`python manage.py shell -c "from django.conf import settings; print(settings.DATABASES
   ['default']['NAME'])"`) and confirm it ends in the scratch-copy filename. Re-run this same
   one-liner before every subsequent `migrate` or write-script invocation for the rest of the
   session (Pitfall 1) — it is cheap (well under a second) and is the single highest-value check
   in the whole procedure.
2. **Note on `manage.py test`:** the guard above does **not** need to be re-run before a
   `manage.py test` invocation specifically — Django's test runner always uses its own isolated
   in-memory test DB regardless of `DATABASES['default']['NAME']` (confirmed this session), so a
   leaked/missing `local_settings.py` cannot cause `manage.py test` to touch real data. The guard
   matters for `migrate`/`shell`-based writes and for `runserver`, not for `test`.
3. **Immediately after applying the throwaway migration:** re-run the Fresh DB Snapshot query
   (Investigation Methodology item 2) against the scratch copy and diff it against the real-DB
   baseline captured before the migration (31 `CampaignRun`s, 20 `CalendarEvent`s, 11 companion
   rows). A mismatch here — especially a companion-row count that dropped to 0 — is the signature
   of Pitfall 4 (an autodetected `DeleteModel`/`CreateModel` instead of `RenameModel` silently
   destroying the scratch copy's data); stop and re-author the migration by hand rather than
   proceeding to the coexistence check with corrupted data.
4. **After each `models.py`/migration edit, before running any test selection:** `python
   manage.py check` (against the scratch-pointed settings) as a fast (~3.5s, per "Measuring the
   Rename" above) syntax/import smoke test, before spending the ~22s or ~2min cost of an actual
   test run.
5. **After each D-11 prototype script run:** immediately diff the resulting `CalendarEvent`
   count in pk=1's window against the expected number for that scenario (15 for adopt, 15 for
   gap-fill, 26 for the rejected baseline) — don't batch this check until all three copies are
   done; a script bug caught after one run is a one-line fix, caught after three is a
   re-run-everything problem.
6. **Before considering the session's evidence-gathering complete:** re-run the quick single-
   module command (`test_calendar_template`, ~22s) one more time on the scratch branch as a
   final confirmation nothing regressed since the last edit.
7. **At discard (Pitfall 5):** `git status --porcelain` on the real phase-26 branch, **and**
   explicitly confirm `local_settings.py` no longer exists (`ls local_settings.py` should error)
   and `tmp/` is removed — a leaked override file left in place is invisible to `git status`
   (it's gitignored), so checking for its literal absence on disk is a separate, necessary step,
   not implied by a clean `git status`.

### Manual-Only Verification: `/calendar/` dev-server load (D-02)

This step is inherently manual — there is no automated assertion that substitutes for actually
loading the rendered page, and CONTEXT.md's D-02 asks for it explicitly. Record it as manual,
not as something to script around:

1. Confirm the DB-path guard (Sampling Guidance step 1) before starting `runserver`.
2. `python manage.py runserver` (with `local_settings.py` still pointed at the migrated scratch
   copy) and load `/calendar/?year=2026&month=7` in a browser.
3. **Pass condition:** HTTP 200, no traceback page, and the rendered month shows the same event
   count as the Fresh DB Snapshot baseline (20 events for July 2026's relevant window — exact
   count depends on the month's date range shown).
4. **Known gap in this specific check, worth recording explicitly:** D-20 confirms all 11 real
   companion rows currently have `is_verified=1` — **zero** real rows currently exercise the
   `is_verified=False` dashed-border/fallback-label branch of the template
   (`calendar.html:228,244`). A clean page load therefore proves the `prefetch_related()` string
   and the `event.telescope_label_meta` accessor still resolve without error, but **cannot**
   visually confirm the dashed-border CSS itself renders correctly, because no real row is
   currently in that state. Two ways to close this gap, either acceptable: (a) on the **scratch
   copy only**, temporarily flip one companion row's `is_verified` to `False` before this step,
   reload, and confirm the dashed border appears, then note in the decision doc that this was a
   deliberately-constructed check, not a real-row observation; or (b) rely on the existing
   `test_calendar_template.py` suite instead, which already has fixture-based coverage of this
   exact branch (see Don't Hand-Roll above) — cheaper, and already part of the quick-run command.
   Either way, state explicitly in the decision doc which of the two the evidence rests on,
   following the same "confirmed against real rows" vs. "confirmed via constructed input"
   discipline as the rest of this map.
5. Record verbatim: the HTTP status line and the visible event count (a screenshot, or —
   preferable for a doc that should paste cleanly — the `django.test.Client()`-based text
   substitute already given in Investigation Methodology item 2, run in the same session as a
   corroborating, non-interactive second data point, not a replacement for the actual browser load).

### Wave 0

**None — existing infrastructure covers all phase requirements.** No new test framework, config,
or fixture scaffolding needs to exist before investigation begins:

- The Django test runner, its settings module, and the existing `solsys_code/tests/` suite are
  already in place and already exercise all four rename-relevant integration points (Don't
  Hand-Roll above).
- `local_settings.py` is an existing, already-gitignored project mechanism (Investigation
  Methodology item 1) — nothing to add to `.gitignore` or scaffold.
- `insert_or_create_calendar_event()` — the one piece of application code every throwaway script
  reuses — already exists and is unchanged by this phase.
- The throwaway evidence scripts themselves (`tmp/26_integrity_check.py`,
  `tmp/26_reconciler_prototype.py`) are investigation tooling created *during* the phase's own
  execution, not scaffolding that must pre-exist — they are written, run, and discarded within
  the same investigation session (item 1's discard mechanic), never a Wave 0 deliverable.

## Decision-Doc Shape (item 5)

**Sphinx/RST conventions confirmed from both existing spike pages, this session:**

- Section order: `Background` -> `Key finding` (bolded one-paragraph verdict, before the
  detail table) -> a `.. list-table::` with `:header-rows: 1` and explicit `:widths:` -> a
  closing `Future scope` section pointing at the full-detail companion doc.
- Both existing pages open with an identical boilerplate paragraph explaining that the
  full-detail companion doc originally lived at `.planning/phases/<N>-<slug>/<N>-DECISION.md`
  but may have moved to `.planning/phases-archive/` once the milestone closes — reuse this
  paragraph verbatim (substituting the phase number/slug) for the new page, since it's the
  established reader-facing convention for "where do I find more detail," not phase-specific
  content.
- `docs/design/design.rst`'s own `toctree` (lines 39-46) lists each design page's bare filename
  (no `.rst` extension, no path) — a new `docs/design/canonical_record_spike.rst` needs one new
  line added to that list (after `uncertain_scheduling_spike`, matching chronological order) for
  the page to be reachable; a page that exists but isn't in this toctree still builds without
  error but is orphaned (unreachable from the docs nav) — Sphinx's default build does not warn
  on this unless `-W`/nitpicky mode is enabled (confirmed: `.pre-commit-config.yaml`'s
  `sphinx-build` invocation has no `-W` flag), so this is easy to silently miss and worth calling
  out as an explicit plan checklist item, not something the build will catch.
- Phase 18's D-18/`18-DECISION.md` precedent for confidence-level discipline: every finding that
  rests on a *real* row is stated as "confirmed against real rows"; every finding that rests on a
  *constructed* input (no real row exists to test against) is stated as "confirmed via
  constructed input" — explicitly, in the same sentence, never conflated. This phase has one
  exact analogue already flagged in CONTEXT.md D-18 (SPIKE-02's Gemini mapping has zero real
  `GEM:` rows to test against) — the decision doc must carry the same explicit distinction
  forward for that finding.

**One document or two — recommendation, not a lock (Claude's Discretion per CONTEXT.md):** two
documents, following Phase 13's shape rather than Phase 18's single-document shape. Rationale:
Phase 18 folded findings+recommendation into one `18-DECISION.md` because its evidence, while
real, was compact (five short criteria, each answered in a few paragraphs). Phase 26 has four
spike criteria (SPIKE-01..04) each producing genuinely verbatim executable output (an
`IntegrityError` check's printed pass/fail lines, a rename-measurement test-run's output, a
three-way event-count comparison table) — closer in evidentiary volume to Phase 13's ESO spike
(which also produced substantial verbatim API-response evidence) than to Phase 18's. A single
`docs/design/` page cannot reasonably carry all of that inline without becoming unreadable as a
"durable summary," so the full evidence belongs in `.planning/phases/26-canonical-record-spike/
26-DECISION.md` and only the settled vocabulary/key-scheme/checklist (the parts Phases 27-29
actually consume) belongs in `docs/design/canonical_record_spike.rst`.

## Security Domain

### Applicable ASVS Categories

This phase ships no application code, so most ASVS categories are structurally not applicable —
no new endpoint, no new auth surface, no new input-validation boundary. The one category that
genuinely applies is data protection, because the investigation procedure handles real DB rows
containing PII:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface changes |
| V3 Session Management | No | No session-touching code |
| V4 Access Control | No | No new views/endpoints |
| V5 Input Validation | No | No new user input surface |
| V6 Cryptography | No | No secrets/crypto touched |
| Sensitive-data handling (adjacent to ASVS V9/data-classification concerns, not a numbered V-category this project otherwise tracks) | Yes | Never quote `contact_person`+`contact_email` pairings, or a bare `contact_email`, into any committed `.planning/` or `docs/` file — see Pitfall 6 above; this is the established project convention (Phase 18 D-01), not new to this phase |

### Known Threat Patterns for this phase's scope

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Accidental real-DB mutation from a forgotten `local_settings.py` override | Tampering (of the real dev DB, not a security boundary in the classic sense, but the closest analogue) | Pitfall 1's explicit pre-write guard step |
| PII leakage into a committed decision doc | Information Disclosure | Pitfall 6 / the established redaction convention |

## Sources

### Primary (HIGH confidence — verified by direct execution or direct file read, this session)

- Direct execution against the installed Django 5.2.13 in `~/venv/devel_fomo311_venv`: `local_
  settings.py` override confirmation, `DiscoverRunner.build_suite()` vs. `run_tests()` import
  behavior (two separate modules tested), `manage.py check` timing, full-suite segfault
  reproduction, live read-only DB snapshot.
- Direct source read: `django/test/runner.py` (`DiscoverRunner.run_checks()`), `django/core/
  management/commands/test.py` (`requires_system_checks = []`), `src/fomo/settings.py`,
  `manage.py`, `.gitignore`, `solsys_code/models.py`, `solsys_code/admin.py`, `solsys_code/
  calendar_utils.py`, `solsys_code/campaign_views.py` (`_project_calendar_event`,
  `_calendar_event_title`, `_set_run_status`, `CampaignRunDecisionView`), `.pre-commit-
  config.yaml` (`sphinx-build` invocation).
- `grep -rn "CalendarEventTelescopeLabel"`/`"telescope_label_meta"` across `solsys_code/` and
  `src/templates/`, this session — confirms the four-integration-point list is exhaustive.

### Secondary (MEDIUM confidence — existing milestone research, already HIGH-confidence per its
own sourcing, referenced not re-verified)

- `.planning/research/STACK.md`, `PITFALLS.md`, `ARCHITECTURE.md`, `SUMMARY.md` — the migration-
  mechanics guidance (RenameModel-before-AddField, why makemigrations autodetection is unsafe)
  is drawn directly from `STACK.md`'s own Django Pattern 2, not re-derived here.
- `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-CONTEXT.md`
  and `18-DECISION.md`; `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-CONTEXT.md`
  — precedent for throwaway-script discipline, PII redaction posture, and decision-doc structure.
- `docs/design/uncertain_scheduling_spike.rst`, `docs/design/eso_feasibility_spike.rst`,
  `docs/design/design.rst` — RST/toctree conventions.

### Tertiary (LOW confidence)

- None — every claim in this research was either directly verified this session or drawn from
  this repo's own existing, already-cited research/precedent docs.

## Metadata

**Confidence breakdown:**
- Throwaway-evidence mechanics (item 1): HIGH — the `local_settings.py` override was tested
  end-to-end this session, not inferred.
- Rename blast-radius measurement (item 2): HIGH — the "narrowing doesn't avoid SPICE cost"
  finding and the segfault were both directly reproduced this session; this corrects what would
  otherwise have been a reasonable but wrong assumption.
- `IntegrityError` check shape (item 3): HIGH — grounded directly in `models.py`'s actual
  constraint definitions, cross-checked against D-12's already-locked backfill value.
- Adopt-vs-gap-fill prototype shape (item 4): MEDIUM — the mechanics (reuse `insert_or_create_
  calendar_event()`, two/three DB copies) are HIGH confidence; the exact uncovered-night dates
  are left as an Open Question since enumerating them requires the real per-event data the
  scratch script itself should compute, not something to hand-derive in research.
- Decision-doc shape (item 5): HIGH — both existing spike pages and their toctree wiring were
  read directly.

**Research date:** 2026-07-27
**Valid until:** Treat as valid for the lifetime of Phase 26's execution (this is a
methodology/procedure research pass tied to this repo's exact current state, not a
library/version recommendation that ages independently — re-verify the DB snapshot numbers
if execution happens more than a few days after this research date, per D-04's own concern
about the dev DB being a moving target).
