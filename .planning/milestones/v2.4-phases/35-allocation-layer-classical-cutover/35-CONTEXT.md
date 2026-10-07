# Phase 35: Allocation Layer & Classical Cutover - Context

**Gathered:** 2026-09-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 35 gives every allocation — a `CampaignRun` with a resolved site and a resolved window,
campaign or no campaign — its own per-night sunset→sunrise calendar events under a new
`ALLOC:{run_pk}:{night}` key namespace, drawn by a new peer module (`solsys_code/allocation_projector.py`)
that `reconcile_run()` dispatches to in place of today's `_reconcile_classical_nights()` /
`RUN:{pk}:{date}` family. A night with a linked `ObservationRecord` whose block has been placed
or observed has no allocation event; unlinking restores it; the observation's own event is never
edited by either transition. Queue-sourced, class-wide and satellite runs keep their single
whole-window `RUN:{pk}` container (Phase 26 verdict) and are never fanned out.

`load_telescope_runs` stops writing blank-url `CalendarEvent`s and instead creates or updates one
campaign-less `CampaignRun` per schedule line (`source=CLASSICAL_FILE`, a collision-safe
`source_identifier`, two new sub-night window fields), letting the allocation projector draw the
same per-night calendar the command drew before, idempotently on re-run.

The cutover has a stated four-step sequence: schema migration → deploy → a one-time command that
turns the legacy blank-url classical events into allocation runs and re-keys them → the first
`reconcile_campaign_runs` sweep re-keys every `RUN:{pk}:{date}` night into `ALLOC:` in place. After
it, the calendar shows one event per night with no duplicate and no orphan.

Paired docs (CLAUDE.md rule, in `files_modified` up front): `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`,
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, `docs/runbooks/telescope_runs_calendar.rst`
(classical-ingest, campaign-run/reconciler and cheat-sheet sections), plus CLAUDE.md's notebook map if a
new notebook is added.

**In scope:** ALLOC-01, ALLOC-02, ALLOC-03, ALLOC-04, ALLOC-05.

**Out of scope:** cron/`flock` scheduling of the sweep (Phase 36); the final status vocabulary,
status rings, public tallies, UNUSED-01's "unused night" look and GAPB-01 (Phase 37); any write
to an observation event's own fields or to the `facility.get_observation_url()` namespace;
automatic `run_status` aggregation; inferring `source` for `LEGACY` rows; ESO sync; Gemini
read-back.

</domain>

<decisions>
## Implementation Decisions

### Classical line → allocation record (ALLOC-04)

- **D-01: `source_identifier` is a deterministic key from the line's own fields, plus an optional proposal token.**
  Built from the resolved telescope (`SITES` key), instrument, `window_start`, `window_end` and the
  two sub-night tokens (e.g. `CLASSICAL:NTT:EFOSC2:2026-07-09:2026-07-13:BoN:EoN`), and, when the
  line carries one, a proposal token that joins the key. `parse_run_line()` is extended to accept
  the optional proposal token (exact syntax is the planner's; it must not be confusable with a
  status word, a month, or a `BoN`/`EoN`/`HHMM` window token, and `ParsedRun` gains a `proposal`
  attribute). Re-importing the same file matches the same run and updates it (idempotent, ALLOC-04);
  a second line in the same file that yields an identical key is skipped and logged as a
  collision, never silently merged. This is the facility-specific key Phase 31's SCHEMA-03
  finding asked for: two proposals sharing telescope/instrument/nights are distinguishable
  exactly when the schedule names the proposal.
  — **Reversibility:** costly — the key form is persisted on every classical run; changing it
  later means a data fix over every `CLASSICAL_FILE` row and a re-import.
- **D-02: One `CampaignRun` per schedule line, with a range window.** `window_start..window_end`
  are the line's inclusive observing nights after the existing ESO noon-to-noon adjustment
  (`_iter_run_nights()`'s rule moves into the window derivation; it is not re-derived per
  night). `source=CampaignRun.Source.CLASSICAL_FILE` (the enum value that already exists —
  REQUIREMENTS' "CLASSICAL" means this), `campaign` = the `--campaign` `TargetList` or `None`,
  `target=None`, `site=get_site(parsed.telescope)`, `site_raw=parsed.telescope`,
  `telescope_instrument` written in a form that round-trips through
  `campaign_reconciler._split_telescope_instrument()` (`/`-delimited) so the night events carry
  `telescope='NTT'`, `instrument='EFOSC2'` exactly as today. The command routes through
  `write_and_reconcile_campaign_run()` (lookup on `source_identifier`) rather than re-acquiring a
  `CalendarEvent` write path, and reports run-level (`created/updated/unchanged/skipped`) plus the
  reconcile summary per line.
- **D-03: Every line is APPROVED; the parser status maps to `run_status` only.** A schedule file
  is operator-vetted, so `approval_status=APPROVED` for every line and every line projects
  nights, as today. `cancelled` → `RunStatus.CANCELLED` (the existing `RUN_STATUS_CALENDAR_PREFIX`
  gives the `[CANCELLED]` title prefix and ring, replacing `_CLASSICAL_STATUS_PREFIX`);
  `confirmed` / `allocation` → `PLANNED`; `proposed` / `not confirmed` → `REQUESTED`. The status
  word is also kept in the event description (as today).
- **D-04: Two new nullable sub-night window fields on `CampaignRun`.** Names are the planner's
  (e.g. `night_start_utc` / `night_end_utc`, `TimeField`, null = computed sunset / sunrise). `BoN` /
  `EoN` store null; `HHMM` stores the UTC time, and the projector applies today's
  `_resolve_window_time()` rule per night (`< 12:00` → next-morning UTC). Web/CSV/queue runs leave
  both null. One small additive migration; both fields read-only-ish in the admin (staff may
  edit, the projector re-mints affected nights — D-13).
  — **Reversibility:** costly — a schema migration plus every `CLASSICAL_FILE` row's data.

### Handoff semantics (ALLOC-03)

- **D-05: A linked record retires only the site-local night its placed/observed block starts in;
  a queued-only record retires nothing.** For each `CampaignRunObservation` on the run whose record
  has both `scheduled_start`/`scheduled_end` set (the same test `record_time_window()` uses for a
  block), the retired night is `_observing_night(record_time_window(record)[0], ZoneInfo(run.site.timezone))`
  — the run's site zone, noon-anchored (33 CR-02). A record with only a request window is
  intent-that-may-still-move, so the allocation night stays visible until the scheduler places
  it (a queue window is not a set of owned nights). `_observing_night()` is promoted to a shared
  public helper next to `sun_event()` (33's forward pointer) and used by both projectors.
- **D-06: A placed-then-cancelled/failed block keeps its night retired.** A linked record in a
  terminal-negative state that still carries a block occupies the night with its own marked
  (`[X]`/`[C]`/`[F]`, 34 D-11) event; re-drawing an unmarked allocation night beside it would read
  as "still planned". A record that expires while still queued never retired anything (D-05), so
  its allocation night simply remains. What an unused awarded night looks like is Phase 37's
  UNUSED-01.
- **D-07:** Retire = delete the `ALLOC:` event and its `CalendarEventMeta` row. Allocation events
  are derived state, re-creatable from the run plus `sun_event()`, and carry no human audit of
  their own (attribution audit lives on the observation event's meta). Unlink re-mints the
  night, as spike 003 measured (link → 1 retired, re-project → unchanged, unlink → 1 created).
  The reconciler's detach-never-delete rule (CR-01) is unchanged for `RUN:` containers.
- **D-08: `CalendarEventMeta.run` on the observation's event follows the link, with the human
  guard.** On link: set it via `campaign_utils.adopt_event_into_run()` (which refuses when the event
  is attributed to another run — logged and counted, never overwritten). On unlink: clear it via
  `unlink_event_from_run()` only when `meta.run` is this run and the attribution is not
  human-confirmed to a different run (33-10's "human outranks machine"; 33 D-14/D-16). The
  projector never writes an observation event's `title`/`description`/`start`/`end`.

### Projector home, dispatch & trigger (ALLOC-01, ALLOC-02)

- **D-09:** New peer module `solsys_code/allocation_projector.py` owns the `ALLOC:` namespace;
  `reconcile_run()` dispatches to it. `reconcile_run()` keeps its stage-0 guard
  (`_skip_reason()`), its container branch (`RUN:{pk}`) and its convergence step, and calls the
  allocation projector for the per-night case, so the three staff-action views,
  `write_and_reconcile_campaign_run()` and the `reconcile_campaign_runs` sweep keep one entry
  point. `_reconcile_classical_nights()` and `run_night_url()` are deleted; `owned_events()` /
  `writable_events()` keep meaning the `RUN:` namespace only. The projector never imports
  `solsys_code.views` or `solsys_code.ephem_utils`.
  — **Reversibility:** costly — reinstating the `RUN:{pk}:{date}` family means a second
  per-night writer and a reverse cutover.
- **D-10: Queue-sourced runs dispatch to the whole-window container by `source`.**
  `run.source in {LCO_QUEUE, SOAR_QUEUE, GEMINI_QUEUE, ESO_QUEUE}` → `_reconcile_container()`
  regardless of site, alongside today's `telescope_class` and satellite-site rules. `WEB`,
  `CSV_IMPORT`, `CLASSICAL_FILE` and `LEGACY` runs with a resolved site and window are per-night
  allocations. No inference from telescope names or sites; `LEGACY` rows stay per-night until
  staff relabel `source` through the existing admin action ("Can I correct a run's source?"),
  after which the next reconcile flips them.
- **D-11: Triggers — receivers on the link, plus a record-side re-project, plus the sweep.**
  `post_save` / `post_delete` receivers on `CampaignRunObservation` (connected in
  `SolsysCodeConfig.ready()` with `dispatch_uid`, `raw` guard, never-raise, same contract as 34
  D-16) re-project the linked run, so a night retires or restores the moment staff confirm or
  undo. The observation projector's existing `post_save` receiver, after projecting a record that
  has `campaign_run_links`, re-projects each linked run — so a record moving queued → placed
  retires its night without a sweep (still no network call and no `sun_event()` unless a night is
  being minted). `CampaignRun` saves keep reaching `reconcile_run()` through the existing
  staff-action call sites and the sweep; no `post_save` receiver on `CampaignRun` itself.
- **D-12: Allocation night title = `<telescope> <instrument>` with the optional
  `RUN_STATUS_CALENDAR_PREFIX`; no `(window a..b)` suffix on per-night events.** Exactly what
  `load_telescope_runs` writes today (`NTT EFOSC2`, `[CANCELLED] NTT EFOSC2`) — unmarked, site
  token, per 34 D-03/D-05. The description carries the −15° dark window line
  (`sun_event(site, night, 'dark')`, computed only when a night is created or re-minted), the
  status, and the source line / proposal when present. Web/CSV range runs gain the same compact
  per-night form; the container branch's `event_title()` form is unchanged. `target_list` =
  `run.campaign`.
- **D-13: `start_time` / `end_time` of an existing allocation night are never rewritten; a
  sub-night change re-mints the night.** On update, only `title` / `description` / `target_list`
  are written (as the reconciler does today). If the stored night no longer matches the run's
  sub-night fields, the projector deletes and re-creates that night. `sun_event()` (both
  `'sun'` and `'dark'`) runs only for a night being created or re-minted — this is the folded
  todo below, built into the new module rather than patched into the retired branch.
- **D-14:** Re-classification deletes leftover `ALLOC:` nights. When a run changes family (a
  `LEGACY` row relabeled to a queue source, a class set, a site resolved), the convergence step
  deletes any `ALLOC:{pk}:*` event not in this reconcile's active set — the same op as D-07's
  retire — while `RUN:` containers keep CR-01's detach-only rule. SC 5's "no orphan" holds
  through re-classification.
- **ALLOC-02:** nights are `run.window_start + i` (site-local dates by construction), keyed in
  the URL by that date, with `sun_event(run.site, night, 'sun')` for the times. Tests cover a
  Chilean (`X05`/`268`/`269`/`809`) and an Australian (`E10` — timezone must be set) site, including a
  record whose UTC start date differs from its observing night (D-05).

### Cutover (ALLOC-05)

- **D-15: Four stated steps, in the runbook.** (1) `migrate` (D-04's fields); (2) deploy the
  code; (3) run the one-time cutover command (name is the planner's, e.g.
  `cutover_classical_allocations`; `--dry-run`, idempotent, summary line, exits non-zero on
  leftovers) which converts the legacy blank-url classical events (D-17); (4) run
  `reconcile_campaign_runs` once — its first pass over every run takes over the `RUN:{pk}:{date}`
  family (D-16), the 34 D-19 takeover pattern. No `RunPython` data migration. The
  `reconcile_campaign_runs_demo.ipynb` diff proves the result: before/after per-night counts,
  zero `RUN:{pk}:{date}` left, zero blank-url classical left, `RUN:{pk}` containers and every
  URL-keyed observation event byte-identical.
- **D-16:** `RUN:{pk}:{date}` nights are re-keyed in place, or deleted when their run is now a
  container. For a per-night run the projector's takeover updates `url` →
  `ALLOC:{pk}:{night}` and the title to D-12's form, keeping the pk, `start_time`/`end_time` (no
  `sun_event()` recompute) and the self-attributed meta row; a night whose run has a
  placed/observed linked record is deleted in the same pass (D-05/D-07). For a run that D-10
  now sends to the container (the 8 single-night `lco_queue`/`eso_queue` runs on the dev DB),
  its `RUN:{pk}:{date}` events are **deleted**, not detached — the whole per-night `RUN:` family
  retires, so every one of its events is either re-keyed or removed; CR-01's detach-only rule
  applies to `RUN:{pk}` containers only. One-time churn is accepted (33 D-12 / 34 D-19
  precedent).
- **D-17:** Legacy blank-url classical events become allocations by re-parsing their own
  `Source line:`. The cutover command groups blank-url events by the `Source line:` in their
  description, runs `parse_run_line()` + `get_site()`, creates the campaign-less run
  (`source=CLASSICAL_FILE`, `campaign` = the events' `target_list`, sub-night tokens preserved
  into D-04's fields, `source_identifier` per D-01), then re-keys each event to
  `ALLOC:{pk}:{night}` in place and links it (`_link_event_to_run()`-style self-attribution). No
  schedule file is needed; re-running the command is a no-op.
- **D-18: An event the command cannot explain is left untouched and reported.** No parseable
  `Source line:`, an unknown/ambiguous telescope, a blank site timezone, or a `sun_event`
  failure → the event is listed with its reason, nothing is deleted, and the command exits
  non-zero so the operator resolves it in admin (the dev DB's `tmp` event pk 334 is the
  known example). The summary prints the final per-night count SC 5 is checked against.

### Claude's Discretion

- The proposal-token syntax in the schedule line and how `parse_run_line()` disambiguates it;
  the exact `source_identifier` string form within D-01.
- Field names for D-04; whether `TimeField` or a small-int minutes-after-midnight is stored.
- Whether `load_telescope_runs` gains `--dry-run` (recommended: yes, via `reconcile_run(dry_run=True)`)
  and the exact summary wording; whether the old per-night `created/updated/unchanged` counters
  are kept as a second line.
- Receiver error contract on the allocation side: never-raise (34 D-16) for the receivers and the
  record-side re-project; whether `sun_event()`'s `ValueError` keeps propagating out of
  `reconcile_run()` for the staff-action call sites (29 D-06) — recommended: yes, unchanged.
- `ReconcileResult` counters for the new behaviours (`retired` / `restored` / `rekeyed`), the
  `--dry-run` preview shape, and log levels.
- The cutover command's name and module; whether it lives on after Phase 35 (recommended: keep,
  documented as one-time in the runbook, like `repair_stale_campaign_run_sites`).
- Whether the classical event `description`'s `Source line:` is kept verbatim on allocation
  nights (needed by D-17 only for legacy rows; harmless to keep).
- Test file layout (extend `test_load_telescope_runs.py`, `test_campaign_reconciler.py`; new
  `test_allocation_projector.py`) and how the notebooks express the before/after diffs.

### Folded Todos

- **`2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`** — F2: the
  reconciler called `sun_event()` per night on every idempotent sweep and discarded the result for
  existing nights. Its own routing note sends it here; D-13 builds the early exit into the new
  allocation projector (`sun_event()` only when a night is created or re-minted), and the
  retired `_reconcile_classical_nights()` is deleted rather than patched. Close it when D-13
  lands (with the regression test the todo asks for: an idempotent re-reconcile of an existing
  multi-night run makes no `sun_event()` call).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes
- `.planning/ROADMAP.md` §"Phase 35: Allocation Layer & Classical Cutover" — goal, the five
  success criteria, scope note (SCHEMA-03 collision, Phase 26 verdict, cutover sequencing),
  paired-docs list; and the milestone's "Locked constraints" block (one writer per namespace;
  reuse `sun_event()` / `insert_or_create_calendar_event()`; site-local nights; peer modules
  only; never import `views` / `ephem_utils`).
- `.planning/REQUIREMENTS.md` — ALLOC-01..05 (this phase); STATUS-01/02, UNUSED-01, TALLY-*,
  GAPB-01 (Phase 37) and SCHED-08..10 (Phase 36) for what not to pre-empt.
- `.planning/PROJECT.md` §"Current Milestone: v2.4 Observation-First Calendar" — target
  features (allocation layer + handoff), landmines, conventions carried, Phase 34 progress notes.
- `.planning/notes/observation-first-calendar-layering.md` — D2 (base owns, allocation projects
  intent only), D3 (classical runs are allocations, never synthetic records), D4 (narrowing is
  the handoff).

### Spike findings (validated patterns — read before writing the projector)
- `.claude/skills/spike-findings-fomo_devel/SKILL.md` — the non-negotiable requirements list.
- `.claude/skills/spike-findings-fomo_devel/references/allocation-handoff.md` — spike 003: the
  `project_allocation()` sketch, "What to Avoid" (UTC-date keying; recomputing `sun_event` on
  sweeps; writing into a base event), the `CampaignRun` constraints an allocation must satisfy,
  the dev-DB observatories with timezones.
- `.claude/skills/spike-findings-fomo_devel/sources/003-allocation-night-retirement/spike.py` —
  the runnable retire/restore proof D-05..D-08 are built from.
- `.claude/skills/spike-findings-fomo_devel/references/observation-projector.md` — the
  never-raise / no-network contract D-11's record-side re-project must keep.

### Prior-phase decisions this phase builds on or retires
- `.planning/phases/33-series-identity-reconciler-inversion/33-CONTEXT.md` — D-01 (skip-the-night
  is the same sentence as the handoff), D-02 (`_may_write()` blocks on foreign attribution),
  D-03 (past adopts stay in `RUN:` until this phase converts them), D-14 (`meta.run` kept in step
  with `CampaignRunObservation`), D-16 (`unlink_event_from_run()`); plan 33-10's
  human-confirmation guard (`_stale_attributions()`), 33 CR-02's noon-anchored
  `_observing_night()`.
- `.planning/phases/34-the-observation-projector-trigger/34-CONTEXT.md` — D-03 (unmarked =
  not an observation event), D-05 (cross-layer telescope-token convention), D-11 (terminal-negative
  keeps its window, marked), D-14 (`pre_delete` deletes the observation event), D-16 (receiver
  contract), D-19 (first-sweep takeover, no migration).
- `.planning/milestones/v2.2-phases/26-canonical-record-spike/26-DECISION.md` and
  `docs/design/canonical_record_spike.rst` — the "a queue window is not a set of owned nights"
  verdict and the two key families D-10 preserves for containers.
- `.planning/milestones/v2.3-phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  and `docs/design/run_identity_and_unattended_invocation_spike.rst` — SCHEMA-02
  (`source_identifier` + partial unique constraint) and SCHEMA-03 (tolerance match not
  sufficient; the real collision sample) that D-01 answers.
- `.planning/milestones/v2.3-phases/32-adapter-consolidation/32-CONTEXT.md` — why 32-01 Tasks 1–2
  (`write_and_reconcile_campaign_run()`, `adopt_event_into_run()`, nullable `campaign`) exist and
  are the foundation D-02 routes through.
- `docs/design/telescope_runs_calendar.rst` §"Night convention" — the Las Campanas inclusive /
  ESO noon-to-noon rule D-02 carries into the window derivation.
- `.planning/todos/pending/2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`
  — the folded todo (D-13).

### The code this phase writes into, reuses, or retires
- `solsys_code/campaign_reconciler.py` — `reconcile_run()` (dispatch, D-09/D-10),
  `_reconcile_classical_nights()` + `run_night_url()` (deleted), `_reconcile_container()`,
  `_skip_reason()`, `_may_write()`, `_link_event_to_run()`, `_observing_night()` (promoted),
  `_attributed_nights()`, `_stale_attributions()` / `_detach_stale_family_events()` (D-14 adds the
  ALLOC delete), `event_title()` / `event_description()`, `_split_telescope_instrument()`,
  `RUN_STATUS_CALENDAR_PREFIX`, `ReconcileResult`.
- `solsys_code/campaign_utils.py` — `write_and_reconcile_campaign_run()` (D-02's write path),
  `adopt_event_into_run()` (D-08 link), `unlink_event_from_run()` / `UNLINK_CLEARED_FIELDS` (D-08
  unlink), `insert_or_create_campaign_run()`.
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()`,
  `preview_calendar_event_action()`, `update_calendar_event_key_and_fields()` (the re-key op for
  D-16/D-17), `record_time_window()` (D-05's block test).
- `solsys_code/telescope_runs.py` — `sun_event()`, `get_site()`, `SITES`, `ESO_NOON_TO_NOON_SITES`,
  `parse_run_line()` / `ParsedRun` / `KNOWN_STATUSES` (D-01's proposal token, D-03's map),
  `_local_noon_utc()`.
- `solsys_code/management/commands/load_telescope_runs.py` — `_resolve_window_time()`,
  `_iter_run_nights()`, `_START_TIME_MATCH_TOLERANCE`, `_CLASSICAL_STATUS_PREFIX`, `Command.handle()`
  — the behaviours D-02/D-03/D-04/D-12 re-express through a run; the direct
  `insert_or_create_calendar_event()` write is removed.
- `solsys_code/management/commands/reconcile_campaign_runs.py` — the sweep whose first pass is
  step 4 of D-15; `repair_stale_campaign_run_sites.py` — one-time-command precedent for D-15's
  cutover command.
- `solsys_code/observation_projector.py` + `solsys_code/apps.py` `SolsysCodeConfig.ready()` — the
  receiver wiring pattern and the record-side hook point for D-11.
- `solsys_code/models.py` — `CampaignRun` (`Source`, `RunStatus`, `source_identifier` + its
  partial constraint, `window_*`, D-04's new fields), `CampaignRunObservation`
  (`related_name='observation_links'` / `'campaign_run_links'`), `CalendarEventMeta`, the
  `pre_delete` receiver on `CampaignRun` (~line 436).
- `solsys_code/campaign_views.py` — `AttributionDecisionView._confirm()` / `_undo_confirmation()`
  (the link/unlink edges D-11's receivers fire on; ~lines 1245–1380) and the three
  `reconcile_run()` call sites (~lines 558, 714, 801).
- `solsys_code/admin.py` — the `source` correction action D-10 relies on for `LEGACY` rows.
- `solsys_code/templatetags/calendar_display_extras.py` — `_TERMINAL_PREFIXES` /
  `status_border_css()` (the `[CANCELLED]` ring D-03 inherits).
- Tests: `solsys_code/tests/test_load_telescope_runs.py` (24), `test_campaign_reconciler.py`,
  `test_telescope_runs.py`, `test_campaign_attribution_views.py`, `test_observation_projector*.py`.

### Paired docs (CLAUDE.md rule — part of the deliverable)
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` — the command now writes runs.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` — the `RUN:{pk}:{date}` family
  retires; the D-15 cutover diff.
- `docs/runbooks/telescope_runs_calendar.rst` — §"How do I load a classical telescope schedule?",
  §"How do I get every campaign run onto the calendar?", §"Can I correct a run's source?"
  (D-10 consequence), §"Command cheat-sheet", §"Troubleshooting"; a new cutover section (D-15).
- `CLAUDE.md` notebook map — only if a new notebook is added for the cutover command.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `insert_or_create_calendar_event({'url': url}, fields)` / `preview_calendar_event_action()` /
  `update_calendar_event_key_and_fields()` give create, no-churn update, dry-run preview and
  the in-place re-key D-16/D-17 need.
- `write_and_reconcile_campaign_run()` (32-01) is the exact "create-or-update a run keyed on
  `source_identifier`, then reconcile" path `load_telescope_runs` switches to; it already strips
  `source`/`approval_status` for a `WEB` row.
- `adopt_event_into_run()` / `unlink_event_from_run()` are the only attribution writers D-08 uses.
- `_reconcile_classical_nights()` is a working draft of the per-night loop (blocked / attributed
  / existing / mint ordering, `_may_write()` first) — port its ordering, drop its `RUN:` key and
  its per-sweep `sun_event()` call.
- `load_telescope_runs._resolve_window_time()` and `_iter_run_nights()` are the sub-night and
  night-convention rules D-02/D-04 move behind the run.
- Spike 003's `spike.py` is the runnable retire/restore proof; the observation projector's
  receivers in `apps.py` are the wiring pattern for D-11.

### Established Patterns
- Ownership by key namespace: `ALLOC:` is the allocation projector's alone; `RUN:{pk}` stays the
  reconciler's; `http…` observation URLs are the observation projector's; the cutover retires the
  `RUN:{pk}:{date}` family and blank-url classical events entirely.
- Attribution is a link; a human-confirmed attribution outranks any automated writer (33-10).
- Receivers: connected in `ready()` with `dispatch_uid`, `raw` guard, never raise, no network,
  no `sun_event()` unless minting.
- Sweeps report `created / updated / unchanged / …` with per-run failure isolation; one-time
  commands are idempotent, `--dry-run`-able, and documented in the runbook.
- Target fixtures use `NonSiderealTargetFactory`; migrations small and additive; Google
  docstrings; single quotes; 120 cols; `pre-commit run ruff` is the gate.

### Integration Points
- `reconcile_run()` dispatch (D-09/D-10) — the single seam every caller goes through.
- `SolsysCodeConfig.ready()` — two new receivers on `CampaignRunObservation`; the observation
  projector's `post_save` receiver gains the linked-run re-project (D-11).
- `load_telescope_runs.handle()` — rewritten around `write_and_reconcile_campaign_run()`.
- The cutover command (D-15) and the first `reconcile_campaign_runs` sweep after deploy.
- Dev DB baseline (2026-09-12): 241 events — 159 URL-keyed observation events, 72 `RUN:` (56
  `RUN:{pk}:{date}` nights across 28 runs, all self-attributed, 0 human-confirmed, 0 foreign
  attributions; 16 `RUN:{pk}` containers), 10 blank-url (9 classical nights pk 44–52 with a
  parseable `Source line:` and `target_list=2`; 1 junk `tmp` event pk 334 attributed to run 68),
  0 `ALLOC:`; 45 runs (0 without campaign, 0 with `source_identifier`; sources: 24 legacy, 11
  csv_import, 6 eso_queue, 4 lco_queue); 232 meta rows (65 with `run`, 159 with
  `observation_record`); 1 `CampaignRunObservation` link. Under D-10 the 8 single-night
  `lco_queue`/`eso_queue` per-night runs (2, 3, 7, 14, 15, 18, 19, 20) become containers at cutover.

</code_context>

<specifics>
## Specific Ideas

- The handoff rule is deliberately the same sentence as 33 D-01's skip-the-night, refined by
  "placed or observed, not merely queued": what a user sees for an attributed night must not
  change between Phase 33 and Phase 35 except that the night now disappears the moment the
  scheduler places the request.
- A classical import must look byte-for-byte the same on the calendar as before the cutover
  (title `NTT EFOSC2`, dark-window description line, `[CANCELLED]` ring) — SC 4 is a
  regression test against today's `load_telescope_runs` output.
- "Provenance is a stored fact, never a guess": queue-vs-classical dispatch reads `source`;
  `LEGACY` rows are corrected by a human, never inferred.
- The cutover never deletes what it cannot explain; it reports and exits non-zero.

</specifics>

<deferred>
## Deferred Ideas

- Inferring `source` for `LEGACY` runs from their site/telescope (offered as a one-time cutover
  relabel; declined — staff relabel through the admin action instead). Could become a Phase 37
  or later data-hygiene task if many `LEGACY` rows remain.
- How an unused awarded night (allocation night that was never observed) is visually
  distinguished — Phase 37's UNUSED-01, untouched here.

### Reviewed Todos (not folded)
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — attribution-UI
  guard, unrelated to allocation projection; keyword match only.
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` — attribution-UI caching, unrelated;
  keyword match only.

</deferred>

---

*Phase: 35-Allocation Layer & Classical Cutover*
*Context gathered: 2026-09-12*
