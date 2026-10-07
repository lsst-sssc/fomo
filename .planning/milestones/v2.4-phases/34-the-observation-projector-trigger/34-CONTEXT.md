# Phase 34: The Observation Projector & Trigger - Context

**Gathered:** 2026-09-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 34 makes every LCO/SOAR `ObservationRecord` draw and keep current its own
`tom_calendar.CalendarEvent` with no operator command: a projector module (a peer of
`calendar_utils.py` / `campaign_reconciler.py` under `solsys_code/`) that derives one event per
record — keyed by `facility.get_observation_url()`, spanning the request window while queued,
the placed block once scheduled, the observed block once observed, and a visibly marked
window for a terminal-negative record — and writes the Phase 33 link fields
(`CalendarEventMeta.observation_record` / `observation_group`); a FOMO-owned Django
`post_save` receiver (plus an `m2m_changed` receiver for group membership and a `pre_delete`
receiver for record deletion) registered in `apps.ready()`; a zero-argument sweep management
command as the backstop for bulk-write paths and backfill (`--dry-run`, per-record failure
isolation); and the retirement of `sync_lco_observation_calendar` in the projector's favour —
one writer for observation-backed nights, same key namespace, same events.

Alongside: the observed-telescope label (a one-time site lookup made only by the sweep, never
by the receiver), the provisional compact title vocabulary that fits a month cell, the
paired docs (a new pre-executed demo notebook for the sweep replacing the retired command's
notebook, `docs/runbooks/telescope_runs_calendar.rst`'s LCO section, the Gemini
no-read-back caveat, CLAUDE.md's notebook map), and SCHED-06's live-night proof that a real
`KEY2026B-004` record's event narrows queued → scheduled → observed with nobody running
anything.

**In scope:** PROJ-01, PROJ-02, PROJ-03, PROJ-04 (shared title stem clause), PROJ-05,
PROJ-06, TRIG-01, TRIG-02, TRIG-03, SCHED-06, ANNOT-03.

**Out of scope:** the allocation layer and `ALLOC:` cutover (Phase 35); cron/`flock`
scheduling, the watched-proposal list and failure notification (Phase 36); the final status
vocabulary, status rings for every state, public tallies (Phase 37); any write to the
`RUN:` namespace or to `CalendarEventMeta.run` / `confirmed_by` / `confirmed_at`; live Gemini
read-back (`GEMFacility` has none — `sync_gemini_observation_calendar` stays submission-echo);
ESO sync; upstreaming the projector.

</domain>

<decisions>
## Implementation Decisions

### Compact title & series identity (PROJ-04 stem, PROJ-06)

- **D-01: Stored title = `[marker] <telescope token> <target>`**, e.g. `[Q] 2m0 3I/ATLAS`,
  `[S] 1m0 11P`, `[O] FTS 3I/ATLAS`. The target is `record.target.name`; the telescope token is
  the coarse aperture class (`coarse_telescope_label()`: `0m4`/`1m0`/`2m0`, `4m0` for SOAR)
  while the record is queued or placed, and the observed telescope (D-07) once observed. The
  month cell shows the first 16/18 characters (`calendar.html` `truncatechars`), so the
  marker and token are always visible and the target usually is. Telescope and instrument
  still go into `CalendarEvent.telescope` / `.instrument` for the modal.
  — **Reversibility:** reversible — a title-builder change plus one sweep re-titles every
  event.
- **D-02: Provisional marker vocabulary — short letters everywhere.** `[Q]` queued,
  `[S]` placed (scheduled block, not yet observed), `[O]` observed (successful terminal
  state), `[X]` `WINDOW_EXPIRED`, `[C]` `CANCELED`, `[F]` `FAILURE_LIMIT_REACHED` and
  `NOT_ATTEMPTED`, `[?]` an inconsistent record (D-13). Phase 37 owns the final wording; this
  phase *extends* `calendar_display_extras._TERMINAL_PREFIXES` / `status_border_css` and the
  legend to recognise the new tokens while keeping the reconciler's `[CANCELLED]`/`[WEATHERED]`
  and the classical `[EXPIRED]`-style prefixes matching, so no existing ring is lost.
- **D-03: Every projector-written title carries exactly one marker; an observed record is `[O]`, never a bare title.** "No marker" is reserved to mean "not an observation event"
  across layers (allocation nights, `RUN:` containers and legacy classical events are all
  unmarked), so a clean title cannot be mistaken for "observed".
- **D-04: Series identity is the shared stem, nothing more, in the title.** Grouped records
  look alike in the cell (`[Q] 1m0 11P` × 28). "Night *n* of *N*", the group name and a link
  back to the group are rendered at display time in the modal from
  `meta.observation_group` (siblings ordered by `record_time_window()[0]`), via a template
  tag in the style of Phase 33's `campaign_decoration()`. Nothing group-derived is written
  into `title` or `description`, so adding or removing a sibling never churns the whole
  group.
- **D-05: Cross-layer telescope-token convention (recorded for Phases 35/37, implemented here only for observation events).** LCO/SOAR robotic records use the aperture class /
  observed telescope (D-01, D-07); allocation and classical events use the site short name
  already in `telescope_runs.SITES` (`NTT`, `FTN`, `FTS`, `Magellan-Clay`, …); `GN`/`GS`-style
  names would join `SITES` if a facility with real read-back ever needs them. Phase 35's
  allocation projector and Phase 37's vocabulary should keep to this so a month reads
  consistently.

### Telescope label — verification without a network call in the hot path

- **D-06: The projector never verifies; a coarse label while pending is by design.** While a
  record is queued or placed the token is the coarse aperture class with no portal call.
  `[UNVERIFIED]`, the `telescope_api_failed` counter and the "label unverified" description
  line retire with the old command. `CalendarEventMeta.is_verified` has no meaning for an
  observation event any more: the projector normalises it to `True` on every meta row it
  writes (so the dashed border stops showing on the legacy `is_verified=False` rows after
  the takeover, D-19) and never sets it `False`.
- **D-07: Once a record reaches a successful terminal state, the token becomes the telescope it was observed on.** `FTN` for `('ogg','2m0')`, `FTS` for `('coj','2m0')`,
  `SOAR` for `('sor','4m0')`, and `SITE-aperture` (`LSC-1m0`, `OGG-0m4`, …) for the 1m0/0m4
  network — i.e. `SITE_TELESCOPE_MAP` with its 2m0/4m0 values renamed. `[O] LSC-1m0 3I/ATLAS`
  is 20 characters; the cell truncation of the target for 1m0/0m4 observed events is
  accepted. A COMPLETED record whose lookup has not succeeded yet keeps the coarse token
  under `[O]`.
- **D-08: The sweep makes the lookup, once per newly-observed record; the receiver never does.** For a record in a successful terminal state with no stored observed-site, the sweep
  calls the existing `calendar_utils.resolve_placement_block()` once (10 s timeout, never
  raises, same COMPLETED-first-else-PENDING block TOM's own poll selects), stores the result
  (D-09), then projects. A failed or unmapped lookup leaves the coarse token, is counted
  (`site_lookup_failed`), and is retried on the next sweep. Bounded work: one call per record,
  ever (~60 on the first sweep over the dev DB). Phase 36 runs the sweep right after
  `updatestatus`, so the label lands within one cron cycle.
- **D-09: The observed site is stored on `ObservationRecord.parameters` under generic, un-prefixed, self-describing keys that mirror the OCS observation block's own field names** — e.g. `observed_site='ogg'`, `observed_telescope='2m0a'` (verbatim portal values;
  `observed_enclosure` optional). Not FOMO-prefixed (the user wants this usable by other TOMs;
  base TOM has no convention — `parameters` is "what was submitted", and the only observed
  telescope TOM records is `ReducedDatum.telescope`, per datum). Keys must not collide with
  the LCO form's submission-constraint `site` field. The sweep saves the record with
  `update_fields=['parameters']`; that save fires the receiver once more, which projects and
  reports `unchanged`. Exact key names are the planner's within this rule.
  — **Reversibility:** costly — renaming the keys later means a data fix across every
  observed record's JSON, and any external consumer that learned the keys.

### Stage classification & edge lifecycles (PROJ-02, PROJ-03)

- **D-10: Stage from record fields only, span from `record_time_window()`.** The spike 002
  classifier: half-set `scheduled_start`/`scheduled_end` → inconsistent (D-13); status in
  `facility.get_failed_observing_states()` → terminal-negative; status in
  `get_terminal_observing_states()` minus failed → observed (block present) or
  completed-no-block (D-12); otherwise placed (block present) or queued. Facility instance
  per record via `get_service_class(record.facility)()`, never one shared instance across
  LCO and SOAR.
- **D-11: A terminal-negative record keeps its full request window, marked** (`[X]`/`[C]`/`[F]`
  spanning the whole submitted window — what `record_time_window()` already returns and what
  the old sync did). A terminal-negative record that still carries a placed block keeps the
  block (the rule already prefers it).
- **D-12: COMPLETED with no block → `[O]` on the request window** (the old command's D-06
  rule: a successful-terminal record is never bannered as still queued); the observed-site
  lookup (D-08) still runs for it.
- **D-13: Inconsistent or unprojectable records.** A record with a usable request window but
  a half-set schedule projects as `[?]` on the window, so the data problem is visible on the
  calendar and not only in a log. A record with no usable window at all (missing
  `parameters['start'/'end']`, unparsable dates) is *unprojectable*: logged at warning
  (never interpolating an exception that could carry credentials — the SYNC-09 discipline),
  counted by the sweep, any existing event left untouched, and the record save never aborted.
- **D-14: Deleting a record deletes its event.** A `pre_delete` receiver on `ObservationRecord`
  removes the projector-owned event — found through `instance.calendar_event_meta` *before*
  Phase 33's `SET_NULL` clears the link, and only if the event's `url` is the record's
  facility URL (never anything outside the projector's namespace). The companion row and any
  attribution audit on it go with the event. The sweep cannot do this later (the record is
  gone), so it must be a receiver.
  — **Reversibility:** costly — a deleted event's attribution audit (`run`, `confirmed_by`,
  `confirmed_at`) is gone with it; switching to "keep and unlink" later is a code change but
  cannot restore what was already removed.
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
  (`insert_or_create_calendar_event()`'s contract); every failure is caught and logged,
  never re-raised. `settings.HOOKS['observation_change_state']` is left pointing at TOM's stock
  hook.

### Sweep, retirement, takeover & live proof (TRIG-03, ANNOT-03, PROJ-05, SCHED-06)

- **D-17: Sweep command — zero required arguments, optional narrowing.** e.g.
  `python manage.py project_observation_calendar` sweeps every LCO/SOAR record; optional
  `--proposal A,B` (exact codes, no substring leakage), `--facility LCO|SOAR`, `--dry-run`
  (reports via `preview_calendar_event_action()`). Per-facility summary line in the retired
  command's phrasing: `created / updated / unchanged / unprojectable / site_lookups /
  site_lookup_failed`. Per-record failure isolation; a second run reports everything
  `unchanged`; the `RUN:` count, blank-url and `GEM:` events are provably untouched. Phase 36's
  cron calls it with no arguments.
- **D-18: `sync_lco_observation_calendar` is deleted outright** — the command module, its
  38 tests and its demo notebook. Behaviours worth keeping (no-churn, per-facility dispatch,
  exact-code proposal filter, failure-prefix priority, credential-free logging) are
  re-expressed as projector/sweep tests, not copied. The `calendar_utils` helpers it alone
  called (`resolve_placement_block`, `derive_telescope`, `SITE_TELESCOPE_MAP`,
  `aperture_class_from_telescope_code`) stay — the sweep's site lookup (D-08) reuses them.
  Runbook §"How do I sync LCO/SOAR queue observations?" is replaced by a projector/sweep
  section; the cheat-sheet row, `docs/notebooks.rst:15` and CLAUDE.md's notebook map are
  updated to the new command and notebook.
  — **Reversibility:** costly — restoring the command means reviving a second writer of the
  same key namespace, the very thing ANNOT-03 removes.
- **D-19: The first sweep takes over the 156 legacy URL-keyed events; no migration.** It
  re-titles them to the D-01 form (they still carry spike 002's stopgap titles), links
  `observation_record` / `observation_group`, normalises `is_verified` (D-06), and makes the
  one-time site lookups for the observed ones. The demo notebook snapshots every event's
  `(url, title, start, end, meta links)` before and after and shows the `RUN:` / blank-url /
  `GEM:` sets byte-identical. One-time churn is accepted (Phase 33 D-12 set the precedent).
- **D-20: SCHED-06 is proven by a live-narrowing section in the sweep demo notebook**, built
  from spike 004's `recheck.py`: a baseline snapshot of `(status, scheduled_start/end, event
  start/end/title)` for the pending `KEY2026B-004` records; the operator runs *only* TOM's
  `updatestatus` over several real nights (Phase 36 puts it on cron; nothing else is run);
  the notebook is re-executed and committed showing records that moved `[Q]` → `[S]` → `[O]`
  with no sweep in between. `34-UAT.md` records the dates. Phase verification passes on the
  baseline plus the mechanism tests; the post-nights re-execution is a follow-up commit, not a
  gate on Phase 35 planning.
- **D-21: ANNOT-03's Gemini caveat.** `sync_gemini_observation_calendar` and its notebook are
  not changed in behaviour; the runbook's Gemini section and the notebook's prose document
  that `GEMFacility.get_observation_status()` / `get_observation_url()` are stubs, so Gemini
  events are submission-echo only and never narrow. The projector ignores `GEM` records
  (D-16). This closes the folded todo below.

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
- Whether a settings flag or context manager is offered to silence the receivers during
  bulk test fixtures (the `raw` check already covers `loaddata`).
- Test file layout for the migrated behaviours; how the notebook's before/after diff is
  expressed.

### Folded Todos

- **`2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`** — the
  Phase 31 UAT correction that SOAR (a branch inside the LCO path, with real read-back) is the
  second robotic facility and Gemini is submission-echo with no read-back. Its three items are
  all resolved by this phase: `SOAR_QUEUE` already shipped in 32-01; PROJ-01 names LCO/SOAR;
  D-16/D-21 ignore `GEM` records and document the caveat (ANNOT-03). Mark it done when D-21
  lands.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes
- `.planning/ROADMAP.md` §"Phase 34: The Observation Projector & Trigger" — goal, the five
  success criteria, paired-docs list, scope notes on SCHED-06 and ANNOT-03; and the
  milestone's "Locked constraints" block (one writer per source; `post_save` not the TOM hook;
  reuse the shipped helpers; no network / no `sun_event` in the hot path; peer modules only;
  never import `views` / `ephem_utils`).
- `.planning/REQUIREMENTS.md` — PROJ-01..06, TRIG-01..03, SCHED-06, ANNOT-03 (this phase);
  ALLOC-01..05 (Phase 35) and STATUS-01/02 (Phase 37) for what not to pre-empt; PROJ-04's
  scope-split note.
- `.planning/PROJECT.md` §"Current Milestone: v2.4 Observation-First Calendar" — target
  features, landmines, conventions carried.
- `.planning/notes/observation-first-calendar-layering.md` — D1–D5 and the "unresolved"
  list this phase settles (expired/failed rendering: marked, full window).

### Spike findings (validated patterns — read before writing the projector)
- `.claude/skills/spike-findings-fomo_devel/SKILL.md` — the non-negotiable requirements
  list.
- `.claude/skills/spike-findings-fomo_devel/references/observation-projector.md` — the
  stage classifier, title priority, never-raise contract, "What to Avoid" (no per-record
  portal call in the projector; keep the `completed-no-block` branch; long titles).
- `.claude/skills/spike-findings-fomo_devel/references/event-trigger.md` — `apps.ready()`
  wiring, `raw` guard, why TOM's hook and `update_fields` cannot be relied on, bulk paths.
- `.claude/skills/spike-findings-fomo_devel/references/allocation-handoff.md` — the
  attribution-is-a-link rule the projector must respect (never write `meta.run`).
- `.claude/skills/spike-findings-fomo_devel/sources/002-observation-projector/` (`projector.py`,
  `sweep.py`), `sources/001-b-trigger-django-post-save/`, and
  `sources/004-live-narrowing-updatestatus/recheck.py` — the runnable spike code D-10, D-16
  and D-20 are built from.

### Prior-phase decisions this phase builds on
- `.planning/phases/33-series-identity-reconciler-inversion/33-CONTEXT.md` — D-05/D-06/D-07
  (the link fields and `SET_NULL`), D-08 (no backfill in 33 — this phase's sweep does it),
  D-10/D-11 (cell marker + modal decoration pattern D-04 mirrors), D-14 (`meta.run` is the
  single attribution source — never written here), D-15 (observation-backed events stay in
  the attribution queue), D-16 (`unlink_event_from_run()` — the projector never calls it).
- `.planning/phases/33-series-identity-reconciler-inversion/33-01-SUMMARY.md` (the
  `campaign_decoration()` tag and retired adopt path), `33-02-SUMMARY.md` (month-cell chip,
  `fomo_render_calendar` prefetch), `33-03-SUMMARY.md` (migration `0017`, read-only admin).
- `.planning/milestones/v2.2-phases/29-the-reconciler/29-CONTEXT.md` — D-05/D-06: the
  sibling sweep's summary form and per-run failure isolation that D-17 mirrors.
- `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md`
  and `.planning/debug/gemini-vs-soar-facility-scope.md` — the folded todo and its evidence.

### The code this phase writes into, reuses, or retires
- `solsys_code/calendar_utils.py` — `record_time_window()`, `insert_or_create_calendar_event()`,
  `preview_calendar_event_action()`, `extract_instrument()`, `coarse_telescope_label()`,
  `resolve_placement_block()`, `derive_telescope()`, `SITE_TELESCOPE_MAP` (values renamed per
  D-07), `InstrumentExtractionError`.
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — retired (D-18);
  `_FAILURE_PREFIX_BY_STATUS`, `_title_for()`, `_build_event_fields()`, `Command.handle()`
  are the behaviours to re-express.
- `solsys_code/management/commands/backfill_lco_observations.py` — the discovery path whose
  `group.observation_records.add()` (line ~672) motivates D-15; `parameters['start'/'end']`
  precedent for writing into `parameters`.
- `solsys_code/management/commands/reconcile_campaign_runs.py` — the sibling sweep's
  `--dry-run` / summary conventions.
- `solsys_code/models.py` — `CalendarEventMeta` (`observation_record` OneToOne,
  `observation_group` FK, `is_verified`, `run`), the existing `pre_delete` receiver on
  `CampaignRun` (~line 423) as the receiver pattern.
- `solsys_code/apps.py` — `SolsysCodeConfig` (no `ready()` yet).
- `solsys_code/templatetags/calendar_display_extras.py` — `status_border_css()`,
  `_TERMINAL_PREFIXES`, `campaign_decoration()`; `src/templates/tom_calendar/partials/calendar.html`
  (`truncatechars:18` / `:16`, the `is_verified == False` dashed-border branches) and
  `event_form.html` (modal blocks); `solsys_code/views.py` `fomo_render_calendar` prefetch.
- `src/fomo/settings.py` `HOOKS` (~line 372) — left on TOM's stock hook.
- Installed tomtoolkit 3.0.1: `tom_observations/models.py` `ObservationRecord.save()` (hook
  semantics), `tom_observations/facility.py` `update_observation_status()` (bare `save()`,
  no `update_fields`), `tom_observations/facilities/ocs.py` `get_observation_status()`
  (fetches blocks, keeps only start/end), `lco.py` `_build_location()` (`site` is a
  submission constraint).
- Tests to migrate/extend: `solsys_code/tests/test_sync_lco_observation_calendar.py`
  (38 tests, mocks `solsys_code.calendar_utils.make_request`),
  `test_calendar_template.py`, `test_calendar_event_meta_links.py`,
  `test_backfill_lco_observations.py`.

### Paired docs (CLAUDE.md rule — part of the deliverable)
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` → replaced by a new
  pre-executed sweep demo notebook (takeover diff + live-narrowing section), registered at
  `docs/notebooks.rst:15` and in CLAUDE.md's notebook map (`CLAUDE.md:127`).
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` — no-read-back
  caveat in prose (D-21).
- `docs/runbooks/telescope_runs_calendar.rst` — §"How do I sync LCO/SOAR queue
  observations?" (replaced), §"How do I sync Gemini queue observations?" (caveat), the
  command cheat-sheet, §"Troubleshooting".

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `record_time_window()` is the stage rule (window while both schedule fields are null,
  block when both set, `ValueError` when half-set, `KeyError` with no window) — D-10..D-13
  are a thin classifier on top of it.
- `insert_or_create_calendar_event({'url': url}, fields)` / `preview_calendar_event_action()`
  give the no-churn create/update/unchanged contract and the `--dry-run` preview for free.
- `resolve_placement_block()` + `derive_telescope()` + `SITE_TELESCOPE_MAP` are the
  one-time site lookup (D-08); already timeout-bounded and credential-safe.
- `campaign_decoration()` (33-01) and the `.cal-campaign-chip` marker (33-02) are the
  template-tag pattern for D-04's "night n of N" modal rendering.
- `reconcile_campaign_runs` is the sibling sweep for command form, summary and isolation.
- Spike 002's `projector.py` / `sweep.py` and spike 004's `recheck.py` are working drafts of
  the projector, sweep and live-narrowing snapshot.

### Established Patterns
- Ownership by key namespace: the projector owns `url == facility.get_observation_url(id)`
  events only; `RUN:` (reconciler), `GEM:` (Gemini echo), blank-url (classical) are never
  created, modified or deleted by it.
- Attribution is a link: the projector creates or updates the `CalendarEventMeta` row for
  its event to write `observation_record` / `observation_group` / `is_verified`, and never
  touches `run` / `confirmed_by` / `confirmed_at`.
- Credential-free logging (SYNC-09): a caught exception from a portal call is never
  interpolated into a log line.
- Facility instance per record (`get_service_class(record.facility)()`), never shared
  across LCO and SOAR.
- Target fixtures use `NonSiderealTargetFactory`; migrations are small and additive; Google
  docstrings; single quotes; 120 cols; `pre-commit run ruff` is the gate.

### Integration Points
- `SolsysCodeConfig.ready()` gains the three receiver connections (`post_save`,
  `m2m_changed`, `pre_delete`); receivers must not import `solsys_code.views` or
  `solsys_code.ephem_utils`.
- `calendar_display_extras` prefix maps and the legend gain the D-02 tokens; the modal
  (`event_form.html`) gains the series block; `fomo_render_calendar` may need
  `observation_group` in its prefetch.
- Phase 36 will call the zero-argument sweep right after `updatestatus` on cron; Phase 35's
  allocation handoff reads `meta.observation_record` to retire a night.
- Dev DB baseline (2026-09-10): 238 events — 156 URL-keyed (spike-002 stopgap titles),
  72 `RUN:`, 0 `GEM:`, the rest blank-url classical; 83 companion rows, 0 with
  `observation_record`; 159 `ObservationRecord`s, all LCO (74 PENDING, 60 COMPLETED,
  18 WINDOW_EXPIRED, 6 CANCELED, 1 FAILURE_LIMIT_REACHED), 0 SOAR; 9 groups, no record in
  more than one group; real `parameters` carry only `end/instrument_type/proposal/start`.

</code_context>

<specifics>
## Specific Ideas

- The user's own words on labels: "it's OK that it stays as '2m0' while still pending and
  scheduled and waiting to be observed. Once it reaches a terminal observed state, it could
  be updated with the telescope/site it was observed at e.g. FTN/FTS for a 2m0 network
  observation and OGG/ELP/TFN/LSC/CPT/COJ for 1m0 and 0m4 LCO network observations" —
  refined to `SITE-aperture` for the 1m0/0m4 network because LSC/OGG/… host two apertures.
- The user asked that the observed-site keys be generic rather than FOMO-specific so the
  data is useful to non-FOMO TOMs; no base-TOM convention exists, so mirror the OCS block
  field names.
- `[O]` was chosen over a clean observed title because a bare title cannot be told apart
  from an allocation/classical/`RUN:` event that simply has no marker.
- Cross-layer consistency: the same `[marker] <token> <target>` reading should hold when
  Phase 35's allocation nights and Phase 37's vocabulary land (D-05).
- The takeover moment is deliberately used to change the title form once (the 156 events
  are churned anyway), mirroring Phase 33 D-12's accepted one-time churn.

</specifics>

<deferred>
## Deferred Ideas

- A `GN`/`GS`-style telescope token for Gemini or other non-LCO facilities in the `SITES`
  vocabulary — only meaningful once a facility with real read-back exists; noted for
  Phase 35/37 under D-05, not implemented here.

### Reviewed Todos (not folded)
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — its target
  file is deleted by D-18 and the helpers it wanted extracted already live in
  `calendar_utils.py`; the todo can be closed as overtaken once the command is gone.
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — routed to
  Phase 35 by its own note (the allocation projector replaces `_reconcile_classical_nights()`).
- `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` and
  `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` —
  attribution-UI items unrelated to the projector.

</deferred>

---

*Phase: 34-The Observation Projector & Trigger*
*Context gathered: 2026-09-10*
