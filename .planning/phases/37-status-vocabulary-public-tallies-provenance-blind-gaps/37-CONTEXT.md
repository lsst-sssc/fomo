# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps - Context

**Gathered:** 2026-09-18
**Status:** Ready for planning

<domain>
## Phase Boundary

The layered calendar reads correctly to everyone. This phase delivers:

- **STATUS-01/02** — one status vocabulary that drives every calendar title prefix, status ring
  and legend entry, replacing the three parallel prefix maps
  (`observation_projector._STAGE_MARKER` + `_FAILURE_MARKER_BY_STATUS`,
  `campaign_reconciler.RUN_STATUS_CALENDAR_PREFIX` as reused by `allocation_projector`, and
  `calendar_display_extras._TERMINAL_PREFIXES` / `_OBSERVATION_STATUS_LEGEND`) that today agree
  only by convention; a named placed-but-unobserved state; and one facility-aware terminal
  classifier replacing the hardcoded `status == 'COMPLETED'` (`calendar_utils.py:336`).
- **TALLY-01/02** — a public (any visitor), read-only tally on each run and rolled up per
  campaign: linked observation groups and records, nights observed / scheduled /
  expired-or-failed / unused-so-far, updating as the projector narrows.
- **TALLY-03** — a guard: `CampaignRun.run_status` is never derived from linked records.
- **UNUSED-01** — an awarded allocation night that came and went with nothing scheduled or
  observed is visibly different on the calendar from a realised night.
- **GAPB-01** — coverage-gap analysis counts every observation on the campaign calendar, not
  only `CampaignRun` rows.

Out of scope (roadmap-locked): automatic `run_status` aggregation, a run detail page, live
Gemini read-back, ESO sync, any new writer of another layer's events.

</domain>

<decisions>
## Implementation Decisions

### Status vocabulary (STATUS-01/02)

- **D-01: Short-letter markers everywhere, final.** Phase 34's provisional `[Q]` queued, `[S]`
  scheduled, `[O]` observed, `[X]` window expired, `[C]` cancelled, `[F]` failed, `[?]`
  inconsistent record become the final vocabulary, and the run-level `[CANCELLED]` /
  `[WEATHERED]` prefixes migrate to the same short form. Every title prefix, the
  `status_border_css()` ring and the legend derive from one definition in one module — no
  second hand-maintained copy anywhere ("must stay byte-identical" comments go away because
  there is nothing left to keep identical). — **Reversibility:** costly — re-titling is one sweep
  plus one allocation re-project, but every test that asserts on prefix strings
  (`test_campaign_approval.py` and the calendar template tests) and the runbook's documented
  prefixes change with it.
- **D-02: Run-level states share the letters.** A staff-cancelled run night is `[C]` (the same
  letter as a portal-cancelled record); a weathered / technical-failure run night is a new
  `[W]`. The pop-up's existing `Run status:` line still says which layer the cancellation came
  from. `[C]` therefore means "cancelled, by whoever owns this event".
- **D-03: `[S]` is called "Scheduled".** The placed-but-unobserved state (block scheduled, not
  yet observed — spike 002's vocabulary gap) is named with the LCO portal's own word, in the
  legend, the runbook and docstrings. "Placed" stays an internal code word only.
- **D-04: One legend lists every visible state**, including `[U]` unused awarded night (D-08,
  a render-time token that never appears in a stored title) and `[?]`, so the legend is the
  single explanation of everything a visitor can see on the calendar. The legend stays a
  fixed, ordered vocabulary read from the one module — never data-driven from the database.
- **D-05: The LCO/SOAR OCS vocabulary is the canonical state model; other facilities map onto
  it.** The classifier's states are the OCS ones (`PENDING`, a placed block, `COMPLETED`,
  `WINDOW_EXPIRED`, `CANCELED`, `FAILURE_LIMIT_REACHED`, `NOT_ATTEMPTED`). A facility whose
  TOM `get_terminal_observing_states()` does not fit is *mapped* onto those states by a small
  FOMO-side table rather than trusted: `tom_gemini` is a limited subset that only schedules
  disruptive ToOs, so its `TRIGGERED` / `ON_HOLD` "terminal" states mean *submitted*, never
  observed (and may change with future GPP support); ESO is expected to gain a real Phase 2
  read-back vocabulary only far in the future. No facility ever classifies as observed by
  accident, and the `status == 'COMPLETED'` check in `calendar_utils.py` goes through the
  classifier.

### Public tallies (TALLY-01/02/03)

- **D-06: Unused nights for queue-scheduled / class-wide runs come from the LCO portal, not a
  new manual field.** The portal proposal API's `timeallocation_set` (allocated hours minus
  used hours, summed over the proposal's allocations) is divided by the fixed rule of thumb
  **10 hours = 1 night** (the NOIRLab/LCO proposal convention) to give an *estimated* unused
  night count, labelled as an estimate. The figure is per *proposal* and is attached to every
  run carrying that proposal code (a proposal may be one run or several; nothing in the
  proposal says which). No staff book-keeping is added for anything the portal can answer.
  — **Reversibility:** reversible — a per-class hours-per-night table can replace the constant
  later without touching the model or the fetch.
- **D-07: The proposal time allocation is fetched in the unattended runner tick and stored in a
  new small model keyed by proposal code** (planner's naming, e.g.
  `ProposalTimeAllocation(proposal_code, semester, instrument_type, allocated_hours,
  used_hours, fetched_at)`), as a new step in `solsys_code/unattended.py`'s `STEPS` (or part of
  the status-refresh step — planner's choice) with Phase 36's per-step failure isolation. Pages
  only ever read the stored figure, so an anonymous visitor never triggers a credentialed
  portal call, and a portal outage surfaces through the existing failure email / heartbeat,
  never as a blank public page. `WatchedProposal` is not overloaded — it stays a watch list.
  — **Reversibility:** costly — a model and migration plus a runner step; moving the fetch to
  request time later would be a redesign of the public page's trust boundary.
- **D-08: One compact "Progress" cell per run row** on the campaign table (django-tables2
  column in `campaign_tables.py`), rendering something like
  `2 groups · 14 records · [O] 5 [S] 2 [X/F] 1 [U] ≈3`, reusing the D-01 letters so the table
  and the calendar speak the same vocabulary. Not six sortable columns — the responsive table
  is already wide. The counts must be computed for the whole table in one pass (one
  `annotate(Count(..., filter=Q(...)))`-style query where SQL can express it; the folded
  TTL-cache pattern below where the site-local night rule cannot be pushed into SQL) — never a
  per-row loop.
- **D-09: "Run detail" means the calendar pop-up's attributed-run block.** The same tally is
  added to the `Attributed campaign run` block in
  `src/templates/tom_calendar/partials/event_form.html`, rendered read-only from
  `CalendarEventMeta.run` in the style of `campaign_decoration()`, under the same
  `run.is_publicly_visible` gate. No new run detail page or route.
- **D-10: Campaign roll-up = a header strip above the runs table plus a badge on the campaign
  list.** `campaignrun_table.html` gets a summary line (sums across the campaign's approved,
  publicly visible runs); `campaign_list.html`'s existing `N runs` badge gains e.g.
  `· 5 nights observed`. The unused-nights estimate is counted **once per distinct proposal
  code**, not once per run, in the roll-up.
- **D-11: Night counters follow the site-local observing night** (`telescope_runs.observing_night`,
  the same rule the Phase 35 handoff uses), never the UTC date, for every tally and for the
  gap claims below. Nights observed / scheduled / expired-or-failed are derived from the run's
  linked records (`CampaignRunObservation`) via the D-05 classifier; nights unused-so-far for an
  allocation run are the still-standing elapsed `ALLOC:` nights (D-12/D-15), and for a
  container run the D-06 estimate.
- **TALLY-03 guard (locked by roadmap, restated):** every tally is a read-only aggregate. No code
  path in this phase — receiver, runner step, view, template tag or migration — writes
  `CampaignRun.run_status`; it remains set only by the existing staff decision views.

### Unused awarded nights (UNUSED-01)

- **D-12: Unused is derived at display time; nothing is written.** A template tag in
  `calendar_display_extras` (in the style of `campaign_decoration()`) classifies an `ALLOC:`
  event as unused when its night has ended (`end_time` — the projected sunrise — is in the past)
  and the run's `run_status` carries no `[C]`/`[W]` prefix. The allocation projector's
  no-churn contract is untouched, there is no time-dependent re-title, and a run's nights never
  flip one by one across ticks. — **Reversibility:** reversible — a written `[U]` marker could be
  added later by the projector without changing what the tag shows.
- **D-13: Look = muted chip + a visible `[U]` token prepended at render.** The chip is greyed /
  desaturated (reduced opacity, dashed border — planner's exact CSS) **and** its rendered label
  reads `[U] NTT EFOSC2`; the token is added by the template, never stored. Two channels (style
  and text) so the state is never colour-alone and a screen reader hears it. Not a fourth ring
  colour.
- **D-14: Staff run status always wins; only truly empty nights read unused.** `[C]`/`[W]` from
  `run_status` beats unused. A night is unused only when its allocation event is still standing
  (no linked record retired it — the Phase 35 handoff already deletes retired nights) and the
  night has ended. An *unattributed* observation event on the same site-night does not rescue
  it: that is an attribution-queue matter, not a display rule, and checking for it would be a
  per-cell query that silently masks missing attribution.
- **D-15: Unused counts in the tally and is filterable.** The allocation run's "unused so far"
  is the count of nights the D-12 rule would render as `[U]` — the table and the calendar must
  agree by construction (one shared classifier function). The legend's `[U]` entry is
  click-to-filter like the proposal swatches, so "show me the wasted nights" is one click.

### Provenance-blind gap claims (GAPB-01)

- **D-16: Only observed and scheduled blocks claim a night** — an `[O]` or `[S]` record's placed
  block, on the site-local observing night it falls in (D-11). A queued request's window claims
  nothing (a queue window is not a set of owned nights — the Phase 26/35 domain correction);
  expired / cancelled / failed records claim nothing.
- **D-17: Site assignment for an observation event** — the record's `observed_site`
  (`ObservationRecord.parameters`, Phase 34 D-09) mapped to its `Observatory`; else the site of
  the `CampaignRun` the event is attributed to; else the event is **not** assignable to a site —
  it is reported on the gap page as "claimed, site unknown" (a listed count, never silently
  dropped) and does not close any per-site gap.
- **D-18: "On the campaign calendar" = attributed to one of the campaign's runs
  (`CalendarEventMeta.run`) OR the record's target belongs to the campaign's `TargetList`.**
  The union is what makes the analysis provenance-blind: a classical or queue observation of
  the campaign's own target counts even before anyone works the attribution queue.
- **D-19: Approved run windows still claim alongside observation blocks.** `claimed_dates()`
  keeps today's approved-run-window claims (that is how future awarded nights stay covered in a
  forward-looking search) and adds the D-16 blocks; the gap page's claimed list may say which
  kind covered a night when both apply. The D-05 `_EXCLUDED_RUN_STATUSES` rule is unchanged.

### Claude's Discretion

- **Where the vocabulary lives and its name** — a peer module under `solsys_code/` (e.g.
  `status_vocabulary.py`), imported by the projector, the allocation projector, the reconciler
  and `calendar_display_extras`; it must never import `solsys_code.views` or `ephem_utils`.
- **Legacy title migration.** Recommended: a one-time re-title on the next unattended tick —
  the sweep and the allocation re-project already re-derive titles, so `[CANCELLED]` /
  `[WEATHERED]` become `[C]` / `[W]` without a new command — and the ring matcher recognises
  only the new vocabulary once the paired notebook proves the migration leaves the developer
  database with no legacy-spelled title (mirror Phase 34's "one-time title change" runbook note).
- **The exact "night has ended" instant** for D-12 — `end_time < now()` in UTC is the simple
  answer since an allocation night's `end_time` is its projected sunrise; a grace period is the
  planner's call.
- **How TALLY-03 is enforced** — at minimum a test proving no module in the phase writes
  `run_status`, plus the vocabulary module's docstring stating the rule.
- **What the tally shows before the first successful portal fetch** for a proposal (e.g.
  `unused: not yet fetched`), and how a proposal code is discovered for fetching (the code
  carried on the run and/or its linked events' `proposal` field — researcher confirms which
  field is authoritative after Phase 35's parser change).
- **Gap cache invalidation** — the existing one-hour TTL and the page's "cached for one hour"
  note may stand; invalidating on projector narrowing is optional.
- **Portal API specifics** for `timeallocation_set` (endpoint, fields, semester scoping,
  credentials via the existing `LCOFacility` settings) — researcher.

### Folded Todos

- **Add TTL cache to attribution banner count** (`.planning/todos/pending/2026-09-01-add-ttl-cache-to-attribution-banner-count.md`)
  — folded as a *pattern*, not as its own fix: the public tallies (D-08/D-10) are computed for
  every visitor on every campaign-table load, the same exposure the todo measured for the
  banner count. Counters that SQL can aggregate go through one annotated query; anything that
  needs the site-local night rule in Python uses `campaign_gap.py`'s low-level-cache pattern
  (`django.core.cache`, cf. `GAP_CACHE_TTL_SECONDS`) so the cost never fans out per run. The
  banner-count cache itself remains a separate quick task.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and locked constraints
- `.planning/ROADMAP.md` §"Phase 37" (and the v2.4 "Locked constraints" list above Phase 33)
  — goal, success criteria, scope note (tallies public, PII gate, TALLY-03 is a guard), paired docs.
- `.planning/REQUIREMENTS.md` — STATUS-01/02, TALLY-01/02/03, UNUSED-01, GAPB-01 wording; the
  "Out of Scope" table (no `run_status` aggregation, no auto-attribution).
- `.claude/skills/spike-findings-fomo_devel/SKILL.md` and
  `.claude/skills/spike-findings-fomo_devel/references/observation-projector.md` — spike 002's
  `[SCHEDULED]` vocabulary gap and the no-churn / no-network projector rules.

### Prior-phase decisions this phase builds on
- `.planning/phases/34-the-observation-projector-trigger/34-CONTEXT.md` — D-01..D-05 (title
  form, provisional markers, "no marker = not an observation event", cross-layer telescope
  token), D-09 (`observed_site` parameter keys), D-10..D-13 (stage rule, terminal-negative and
  inconsistent records).
- `.planning/phases/35-allocation-layer-classical-cutover/35-CONTEXT.md` — D-05/D-06 (handoff
  retires a night only for a placed/observed block; an expired-while-queued record retires
  nothing), D-12 (allocation night title form), D-13 (no `sun_event` on an unchanged night).
- `.planning/phases/36-unattended-operation/36-CONTEXT.md` — D-01..D-03 (one runner, fixed step
  order, per-step failure isolation, FOMO-owned LCO/SOAR status refresh) — the runner the D-07
  fetch step joins.
- `.planning/phases/33-series-identity-reconciler-inversion/33-CONTEXT.md` — attribution is a
  link rendered at display time (the pattern D-09, D-12 and D-13 reuse).

### Paired docs (CLAUDE.md rule — in `files_modified` up front)
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` — status vocabulary, run tallies
  and gap behaviour are campaign-lifecycle surfaces (roadmap-named).
- `docs/runbooks/telescope_runs_calendar.rst` — the documented status prefixes, the
  gap-analysis section, and the "How do I run everything unattended?" section (a new runner
  step and its failure signal) all change.
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and
  `load_telescope_runs_demo.ipynb` — allocation-night titles change spelling
  (`[CANCELLED]` → `[C]`), so their executed output changes.
- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — only if the
  projector's written titles or the sweep's reported stages change (the letters are unchanged
  by D-01; the classifier route may not alter output — verifier checks).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/observation_projector.py` — `stage_for()` already classifies from
  `get_terminal_observing_states() - get_failed_observing_states()`; `_STAGE_MARKER` /
  `_FAILURE_MARKER_BY_STATUS` / `title_for()` are the projector's half of the vocabulary to
  fold into the one module. `facility_for()` caches one facility instance per name.
- `solsys_code/campaign_reconciler.py:76` `RUN_STATUS_CALENDAR_PREFIX` and
  `solsys_code/allocation_projector.py:197` `allocation_night_title()` — the run-level half.
- `solsys_code/templatetags/calendar_display_extras.py` — `_TERMINAL_PREFIXES`,
  `status_border_css()`, `_OBSERVATION_STATUS_LEGEND` / `observation_status_legend()`,
  `campaign_decoration()` (the read-only, never-raises, PII-safe display-time pattern for
  D-09/D-12/D-13) and `observation_series_decoration()`.
- `solsys_code/calendar_utils.py` — `record_time_window()`, `SITE_TELESCOPE_MAP`,
  `OBSERVED_SITE_PARAMETER_KEYS`, `derive_telescope()`; line 336's `== 'COMPLETED'` is the
  STATUS-02 target.
- `solsys_code/telescope_runs.py` `observing_night()` — the site-local night rule (D-11).
- `solsys_code/allocation_projector.py` `allocation_events()`, `retired_nights()`,
  `allocation_night_url()` — the `ALLOC:` namespace the unused rule reads.
- `solsys_code/campaign_gap.py` — `claimed_dates()`, `_compute_gap()`, `get_or_compute_gap()`,
  `GAP_CACHE_TTL_SECONDS` and the low-level-cache pattern.
- `solsys_code/unattended.py` — `STEPS` registry, `StepResult`, `command_lock()`: where the
  D-07 fetch step registers.
- `solsys_code/campaign_tables.py` — `CampaignRunTable` (django-tables2; note the
  `Accessor`-on-dict-rows rationale in `render_run_status`, because non-staff rows are
  `.values()` dicts).
- `solsys_code/campaign_views.py` — `ALLOWED_FIELDS_FOR_NON_STAFF` and the
  `contact_public_opt_in` `Case/When` PII gate the tally columns must sit inside;
  `CampaignListView` (`run_count` annotation), `CampaignGapAnalysisView`.

### Established Patterns
- **Display-time decoration from a link, never a text write** (Phase 33) — tallies, unused
  and the pop-up block all follow it.
- **One writer per key namespace; no layer edits another's events** — this phase writes no
  events at all except through the existing projectors' re-derivation.
- **Peer modules under `solsys_code/`, never importing `views` or `ephem_utils`.**
- **PII gate at the queryset** (`.values()` for non-staff) — new columns must be in the
  allowed-field list or annotations, never model attribute access on the dict rows.
- **Runner steps are isolated, locked, and summarised** (Phase 36) — the fetch step is one more.
- **Pre-executed notebooks run against a scratch copy of the developer database** (33-09).

### Integration Points
- Titles: `observation_projector.title_for()`, `allocation_projector.allocation_night_title()`,
  `campaign_reconciler` container titles.
- Display: `src/templates/tom_calendar/partials/calendar.html` (legend, filter JS, chip
  markup), `event_form.html` (attributed-run block), `campaignrun_table.html`,
  `campaign_list.html`, `campaignrun_gap_analysis.html`.
- Runner: `unattended.STEPS`; runbook §"How do I run everything unattended?".
- Tests: `solsys_code/tests/test_campaign_approval.py` (asserts on prefix strings),
  calendar template tests, `test_campaign_gap*.py`, `test_unattended*.py`.

</code_context>

<specifics>
## Specific Ideas

- "A night = 10 hours" is a deliberate, fixed rule of thumb with precedent (the NOAO/NOIRLab
  proposal process for LCO time) — present it as an estimate, not a measurement.
- The user does not want any new manual book-keeping for something the portal can answer;
  the proposal `timeallocation_set` is the source of truth for unused time.
- `tom_gemini` is understood as a limited, ToO-only subset whose vocabulary may change with
  GPP support; the classifier's mapping table is where that future change lands.
- Tally cell example the user accepted: `2 groups · 14 records · nights: 5 obs / 2 sched /
  1 fail / ≈3 unused`, with the letter markers reused.

</specifics>

<deferred>
## Deferred Ideas

- A per-telescope-class hours-per-night table replacing the fixed 10 h rule (D-06 rationale).
- A written `[U]` marker in stored titles (for exports/API consumers) — D-12 chose display-time.
- A "same-site-night observation rescues an unused night" display rule — rejected in D-14 as
  masking missing attribution; the attribution queue is the fix.
- A public run detail page — D-09 chose the pop-up block; noted as a possible later surface.

### Reviewed Todos (not folded)
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — re-reviewed
  this session: still open (`campaign_views.py:1339` `_dismiss()` checks only the reason; the
  confirm paths at `:1234`/`:1258`/`:1316` re-validate via `is_offered_candidate()`). Valid,
  staff-only, unrelated to Phase 37 — a `/gsd-quick` candidate.
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — re-reviewed
  this session: overtaken by Phase 35. `_reconcile_classical_nights()` no longer exists; the
  allocation projector implements D-13 (`preserved_dark_window_line()`,
  `allocation_projector.py:232`) and `test_allocation_projector.py:860` asserts `sun_event` is
  not called on an unchanged night. Close as done-by-Phase-35.

</deferred>

---

*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Context gathered: 2026-09-18*
