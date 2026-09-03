# Phase 32: Adapter Consolidation - Context

**Gathered:** 2026-09-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Every ingest path — the classical schedule file (`load_telescope_runs`), the LCO/SOAR
queue (`sync_lco_observation_calendar`), and the Gemini submission-echo queue
(`sync_gemini_observation_calendar`) — creates or updates a `CampaignRun` instead of
writing a `CalendarEvent` directly, and hands projection to
`campaign_reconciler.reconcile_run()`. A shared write-and-reconcile helper is built as
this phase's groundwork (first plan) so the same create-or-update-then-reconcile pattern
is written once, not three times. This phase does not touch outcome propagation (Phase
33), the scheduler entry point (Phase 34), or status vocabulary unification (Phase 35) —
it only makes a `CampaignRun` exist, by construction, for every ingest path.

</domain>

<decisions>
## Implementation Decisions

### Third-adapter facility target (resolves pending todo `2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`, gap G-31-3)

- **D-01:** ADAPT-03 is retargeted from Gemini to SOAR. `sync_lco_observation_calendar`
  gains a dedicated `CampaignRun.Source.SOAR_QUEUE` value (and its migration) for
  SOAR-sourced records — `SOARFacility` inherits a real portal read-back from
  `LCOFacility`, so it is the facility that actually proves the pattern generalises to a
  second, live-read-back facility. `GEMFacility` is submission-echo only
  (`get_observation_status()`/`get_observation_url()` are hardcoded stubs; the only
  outbound call is `submit_observation()`) and cannot prove that. —
  **Reversibility:** one-way — `SOAR_QUEUE` is a new `TextChoices` member plus a schema
  migration; once SOAR-sourced rows exist under it, collapsing it back into `LCO_QUEUE`
  means a data migration, not just a code revert.
- **D-02:** Gemini's own write path is kept in scope as a new requirement, ADAPT-06:
  `sync_gemini_observation_calendar` still creates or updates a `CampaignRun` from its
  submission-echo data (real, useful for calendar visibility) instead of writing a
  `CalendarEvent` directly — but both the code and
  `docs/runbooks/telescope_runs_calendar.rst` must state explicitly that a
  Gemini-sourced run can never receive Phase 33's automatic outcome propagation, so this
  is discovered now, not as a mid-Phase-33 surprise. — **Reversibility:** reversible —
  purely a documentation/scope statement; no schema commitment beyond the
  already-declared `GEMINI_QUEUE` source value.
- **User explicitly chose "do both"** over retargeting only, or keeping Gemini only.
  ROADMAP.md's Phase 32 goal/success-criteria/locked-constraints and REQUIREMENTS.md's
  ADAPT-03/ADAPT-06 text were updated in this same session to match (see canonical refs)
  — coverage moves from 22/22 to 23/23 v1 requirements.
- **Ship order changes accordingly:** classical → LCO/SOAR (one command, two source
  values) → Gemini. SOAR is not a fourth command to build; it is a second `Source` value
  inside the existing LCO sync command's write path.

### Shared helper's lookup key

- **D-03:** Left to planner discretion ("You decide"). Claude's recommendation: the new
  write-and-reconcile helper should look up an existing `CampaignRun` by
  `source_identifier` first, falling back to the existing campaign+window lookup only
  when `source_identifier` is absent — this matches how 31-DECISION.md and the durable
  spike doc already frame `source_identifier` going forward for adapter-written rows.
  Recorded as Claude's discretion below, not a locked decision — the planner should
  confirm this against the constraint inventory in `31-DECISION.md` before committing.

### Classical proposal-code identity gap

- **D-04:** Ship Phase 31's documented risk as-is — no proposal-code parsing work is
  added to `load_telescope_runs`'s line grammar in this phase. Two different proposals
  colliding on the same telescope/instrument/night remains a documented, low-frequency
  risk (only 1/3 real sample lines carried a proposal code, and it did not parse under
  today's grammar). — **Reversibility:** reversible — closing the gap later is additive
  parser work, not a schema or identity-key change.

### Cutover sequencing (ADAPT-05)

- **D-05:** Hard cutover per adapter, in commit order. Each adapter's plan flips its
  write path from direct `CalendarEvent` writes to `CampaignRun`-write-and-reconcile in
  the same commit that ships it — classical first, then LCO/SOAR, then Gemini. No
  dual-write period, no feature flag. This matches the roadmap's locked "ships
  simplest-first" constraint and keeps each adapter's before/after state simple to
  reason about and test. — **Reversibility:** costly — reverting a shipped adapter's
  write path after later adapters and the reconciler already depend on its `CampaignRun`
  rows means an explicit rollback plan, not a one-line revert.

### Claude's Discretion

- The write-and-reconcile helper's lookup-key priority (D-03 above) — recommend
  `source_identifier`-primary, but the planner should verify against the real
  constraint inventory before locking it in.

### Folded Todos

- **`2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`** — move
  `_reconcile_classical_nights()`'s `sun_event()` call inside the `existing is None`
  branch so idempotent reconcile passes stop paying per-night astropy solar scans for
  results that are discarded. Folded because Phase 32's adapters will call
  `reconcile_run()` far more often (every sync), and ADAPT-04 already requires proving
  no-churn re-syncs against the new write path — this fix removes wasted work on exactly
  those passes.

### 2026-09-03 addendum — 32-01's Task 0 checkpoint outcome (blocking decision re-scoped)

32-01-PLAN.md's opening `checkpoint:decision` task (cutover fidelity, classical
identity-key form, one-way `SOAR_QUEUE` re-confirmation, and site-resolution obscodes)
was put to the user before any implementation task ran. Three of its four
sub-decisions are confirmed; the fourth (queue-sourced calendar fidelity, "1a") is
**rejected as scoped** — this addendum records why and what replanning must resolve.
No code was written; the executor stopped at the checkpoint with 0/3 tasks complete.

- **Decision 2 (classical `source_identifier` form): CONFIRMED as recommended.** Use
  `CLASSICAL:{telescope}:{instrument}:{night.isoformat()}` (observing-night date, not a
  5-minute bucket of `start_time`) — drift-free.
- **Decision 3 (`SOAR_QUEUE` one-way): RE-CONFIRMED**, matching D-01 above. Not
  reopened.
- **Decision 4 (SOAR/Gemini site resolution via `Observatory` obscode): CONFIRMED as
  recommended (4a), with one correction.** `568` is the generic "Maunakea" code, not
  Gemini-North-specific — verified against the live `Observatory` table (`568` already
  resolves to a row named plain "Maunakea"). **Gemini North's obscode is `T15`**, per
  the user directly (LCO/astronomy domain knowledge), not `568`. Neither `T15` (Gemini
  North) nor `I33` (SOAR, Cerro Pachón) currently have `Observatory` rows in the dev DB;
  `I11` (Gemini South) already does. The planner must carry `T15` forward everywhere
  `32-01-PLAN.md` currently says `568` for Gemini North, and the runbook's
  self-healing "which obscodes to create" list must name `T15`/`I33` as the two rows an
  operator needs to create, not `568`.
- **Decision 1 (queue-sourced calendar fidelity at cutover): REJECTED AS SCOPED.**
  The checkpoint framed this as two losses (whole-window/whole-day span instead of a
  precise scheduled block, and title-vocabulary collapse from 5 prefixes to 2), with
  only the vocabulary loss named as returning later (Phase 35). The user's actual
  requirement, stated directly: an LCO/SOAR `CampaignRun`'s `CalendarEvent` should
  start life spanning the observation's whole request window (e.g. 8-24h) and then
  **progressively narrow** — automatically, with no staff action, regardless of how
  many times the underlying `ObservationRecord` is rescheduled to a different time or
  site — down to the actual scheduled block, and finally the actual observed block
  (e.g. "01:15-02:17 UTC"), as `record.scheduled_start`/`scheduled_end` become known.
  Investigation into the existing codebase found this is **not a hypothetical ask — it
  is a working feature today that this phase's plan removes with no replacement**:
  - `sync_lco_observation_calendar.py` already computes exactly this precision via
    `_build_event_fields()`/`calendar_utils.record_time_window()` (documented elsewhere
    in this project's history as "RECON-04, stage 3/4") and writes its own
    minute-precise `CalendarEvent`s directly, coexisting today alongside the
    reconciler's coarser container/per-night events (RECON-02's "coexisting with
    sync-command-produced per-observation events").
  - 32-03-PLAN.md (the LCO/SOAR cutover plan) removes that direct write entirely — the
    explicit goal of ADAPT-02/03 — but does not port the precision into the reconciler.
    It takes the already-computed `start_time`/`end_time` and keeps only `.date()`
    (32-03-PLAN.md around the `window_start = event_fields['start_time'].date()` line),
    discarding the time-of-day permanently. 32-03-PLAN.md's own task text states this
    plainly: "A queue observation's calendar entry is no longer the narrower
    portal-scheduled block; it's the run's whole window."
  - Phase 33's `SCHED-06` window-narrowing does **not** cover this gap: its
    requirement text scopes it to "a **space-mission** run's window," a different run
    type from LCO/SOAR ground-based robotic queue runs. Nothing on the current roadmap
    restores per-record time precision for LCO/SOAR after this phase ships.
  - **Approval-loop concern is separately resolved and not blocking:** `models.py`'s
    `Source` docstring already establishes that `approval_status == APPROVED` together
    with `source != WEB` means the adapter itself sets `APPROVED` at write time — no
    staff review gates a reschedule, however many times it happens. This part of the
    user's worry does not require design work; it was already correctly designed.
  - **Replanning must resolve, before 32-03/32-01's schema work resumes:** how the
    reconciler (not the retiring adapter code) renders a precise, automatically
    narrowing `CalendarEvent` window for a `CampaignRun` with a linked
    `CampaignRunObservation` whose `ObservationRecord` carries `scheduled_start`/
    `scheduled_end` (and, later, actual observed times) — most plausibly new reconciler
    logic keyed off that link, since `CampaignRunObservation.confirmed_at`-linked exact
    identity already exists per adapter design (D-03 above) and needs no additional
    schema. Whether this belongs inside Phase 32's own scope (so ADAPT-02/03 ship with
    working precision from day one) or is split into an explicit new phase/plan is a
    replanning call, not a decision to make silently — the user should see the
    trade-off named, not have it resolved by omission the way the original checkpoint
    did.
  - **This is a phase re-scope, per the original checkpoint's own resume-signal text**
    ("anything other than approval on decision 1 or 4 re-scopes the phase and planning
    stops"). The user chose to stop and re-plan rather than accept the regression with
    a documented follow-up.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 31 spike verdicts (schema, identity, invocation — all locked, Phase 32 executes them)
- `docs/design/run_identity_and_unattended_invocation_spike.rst` — durable summary:
  `CampaignRun.campaign` becomes nullable; `source_identifier` (nullable `CharField`,
  partial unique constraint) is the new write-time identity field, additive alongside
  both existing partial constraints; per-adapter identity values (LCO portal URL,
  Gemini's synthesized `GEM:{program}/{observation-id}` key, classical
  `CLASSICAL:{telescope}:{instrument}:{bucket}` key with 5-minute bucketing); the
  classical tolerance match is not sufficient alone (accepted risk, see D-04 above); the
  "Future scope" section is the direct source of this phase's D-01/D-02/D-03 open
  questions.
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  — full evidence behind the durable summary (real dev-DB snapshot, constraint-probe
  output, classical schedule-file sample, null-guard read-site inventory). Consult when
  the rst summary isn't enough detail.

### Facility-scope correction (this session)
- `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`
  — the todo this phase's D-01/D-02 resolve. Marks itself resolved by this discussion;
  no further action needed on it once Phase 32 plans against D-01/D-02.
- `.planning/debug/gemini-vs-soar-facility-scope.md` — full root-cause diagnosis (gap
  G-31-3): `GEMFacility` source reading, `SOARFacility` source reading, dev-DB counts
  confirming zero Gemini/SOAR rows exist yet.
- `solsys_code/models.py` (`CampaignRun.Source`, ~lines 107-134) — current vocabulary
  (`WEB`, `CLASSICAL_FILE`, `LCO_QUEUE`, `GEMINI_QUEUE`, `ESO_QUEUE`, `CSV_IMPORT`,
  `LEGACY`); this phase adds `SOAR_QUEUE`. Note the docstring's own reasoning for why
  ESO got a dedicated value instead of folding into `LCO_QUEUE` — the same reasoning now
  applies to SOAR.

### Roadmap/requirements (updated this session to match D-01/D-02 — read the current text, not memory of the prior wording)
- `.planning/ROADMAP.md` Phase 32 section — goal, requirements list (now
  ADAPT-01..06), scope note, and success criteria 3/6 updated; "Locked constraints"
  bullet on ship order updated to classical → LCO/SOAR → Gemini.
- `.planning/REQUIREMENTS.md` — ADAPT-03 retargeted to SOAR; ADAPT-06 added for Gemini;
  traceability table and coverage count (23/23) updated.

### Existing write/reconcile code this phase rewires
- `solsys_code/management/commands/load_telescope_runs.py` — classical adapter, writes
  `CalendarEvent` directly today (no `CampaignRun` involvement).
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — already handles
  both `LCO` and `SOAR` facilities (`facilities = {'LCO': LCOFacility(), 'SOAR':
  SOARFacility()}`, `facility__in=['LCO', 'SOAR']`); writes `CalendarEvent` directly
  today.
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` — queries local
  `ObservationRecord.objects.filter(facility='GEM')` only, no outbound call; writes
  `CalendarEvent` directly today.
- `solsys_code/campaign_reconciler.py` `reconcile_run()` — the existing, unchanged public
  entry point every adapter must call after writing/updating its `CampaignRun`. Locked
  constraint: adapters never re-acquire a direct `CalendarEvent` write path.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `campaign_reconciler.reconcile_run(run, *, dry_run=False)` — the one entry point the
  new shared helper must call after every create-or-update; already idempotent
  (`ReconcileResult` reports `unchanged` on a no-op second call).
- `sync_lco_observation_calendar.py`'s existing facility-dispatch dict
  (`{'LCO': LCOFacility(), 'SOAR': SOARFacility()}`) — the natural place to also branch
  the `Source` value per record.

### Established Patterns
- All three commands currently build `CalendarEvent` field dicts directly
  (`_build_event_fields()` in the LCO command) — this phase's helper replaces that
  end-state with a `CampaignRun` write followed by `reconcile_run()`.
- `CampaignRun.Source` (`solsys_code/models.py`) is a `TextChoices` vocabulary; adding
  `SOAR_QUEUE` follows the exact precedent already set for `ESO_QUEUE` (plan 29-06).

### Integration Points
- New logic must live in peer modules under `solsys_code/` (alongside
  `campaign_reconciler.py` / `campaign_gap.py` / `campaign_utils.py`) — locked
  constraint, never a private helper inside `campaign_views.py`, and never importing
  `solsys_code.views` or `solsys_code.ephem_utils` (the 1.6 GB SPICE kernel download is
  fatal for what will become an unattended job in Phase 34).

</code_context>

<specifics>
## Specific Ideas

No UI-facing specifics — this phase is backend adapter rewiring. The only user-visible
effect during the cutover is calendar continuity (Success Criterion 5): one event per
night, no duplicates, no orphans, at every point in the per-adapter hard-cutover
sequence.

</specifics>

<deferred>
## Deferred Ideas

### Reviewed Todos (not folded)
- **`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`** —
  renaming `calendar_utils.py`'s private helpers to reflect shared-module status. Pure
  style cleanup, no behavior change, weak overlap with this phase's write-path rewiring.
  Left for its own quick task.
- **`2026-09-01-add-ttl-cache-to-attribution-banner-count.md`** — campaign-list page
  attribution banner caching. Unrelated to adapter writes; belongs near Phase 28/29's
  attribution surfaces, not here.
- **`2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md`** —
  attribution dismiss-action security guard. Unrelated to adapter writes; same area as
  above.

[No scope-creep ideas surfaced during discussion — the four discussed areas were all
already-flagged open questions from Phase 31's own "Future scope" section.]

</deferred>

---

*Phase: 32-Adapter Consolidation*
*Context gathered: 2026-09-03*
