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
