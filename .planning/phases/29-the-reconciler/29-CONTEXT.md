# Phase 29: The Reconciler - Context

**Gathered:** 2026-08-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 29 makes calendar events a **function of `CampaignRun` state**, computed by one
idempotent reconciler, rather than a side effect of a staff click or a per-gap backfill
command. It ships: `solsys_code/campaign_reconciler.py` (the locked module home) with a
single per-run core function implementing all four window-pipeline stages; a new
`reconcile_campaign_runs` management command that loops that function over every run
(with `--dry-run`); rewiring of the four existing staff actions (approve, resolve_site,
mark_cancelled, mark_weather_failure) to call the same per-run function so a decision
reconciles its run immediately; and retirement of `backfill_range_calendar_events` and
(per this discussion) `_project_calendar_event`/`_calendar_event_title`'s `CAMPAIGN:`-key
projection path, which the reconciler subsumes.

**In scope:** the reconciler module and its per-run core function; the management
command and its `--dry-run`/summary reporting; rewiring the four staff-action call sites;
retiring `backfill_range_calendar_events` (RECON-09) and `_project_calendar_event`'s
`CAMPAIGN:`-keyed projection (this discussion's extension of RECON-09's intent); the
paired demo notebook and runbook updates (CLAUDE.md rule).

**Out of scope:** rewiring the four ingest adapters themselves to create `CampaignRun`s
(v2.3, ADAPT-01..03) — the reconciler reads existing adapter output, it does not change
how classical/LCO/Gemini/CSV-import adapters write; attribution and its confidence
scoring (Phase 28, already shipped); the `source`/`telescope_class` schema (Phase 27,
already shipped); unifying the three status-prefix vocabularies (v2.3, STATUS-01/02).

</domain>

<decisions>
## Implementation Decisions

### The reconciler replaces the existing click-driven projection mechanism entirely

- **D-01: The reconciler takes over `_project_calendar_event()`'s job; `CAMPAIGN:` keys
  are retired.** `campaign_views.py`'s existing `_project_calendar_event()` already
  creates `CAMPAIGN:{run.pk}[:date]`-keyed events on approve/resolve_site — SPIKE-02
  (`26-DECISION.md`) treats this as its own fourth adapter identity mapping, alongside
  classical/LCO/Gemini. Per this discussion, the reconciler's locked `RUN:{run_pk}[:date]`
  key scheme (SPIKE-03) supersedes it going forward: approve/resolve_site/mark_cancelled/
  mark_weather_failure stop calling `_project_calendar_event()` and call the reconciler's
  per-run function instead. **Zero `CAMPAIGN:`-namespaced events exist in the dev DB
  today (D-15, `26-DECISION.md`)**, so this is a clean cutover with nothing to migrate —
  `_project_calendar_event()` and `_calendar_event_title()` can be deleted outright rather
  than deprecated in place. This directly satisfies WR-03's stated deferral in
  `27-REVIEW.md`/`27-REVIEW-FIX.md`: "the automatic writer for [`CalendarEventMeta.run`]
  is deferred to the Phase 29 reconciler... which also owns keeping the link correct as
  runs are re-approved, re-sited or cancelled."

- **D-02: For a classically-scheduled run, the reconciler adopts a pre-existing
  attributed event rather than gap-filling around it.** `26-DECISION.md`'s D-11
  (adopt-vs-gap-fill) was deliberately left open for Phase 29 by explicit human decision.
  Now narrowed by the settled "Queue-run projection" verdict (queue runs never touch
  per-night events at all — bare `RUN:{run_pk}` container only, fully resolved, not
  reopened here): the remaining open question was specifically about **classical** runs
  whose nights already carry a `load_telescope_runs`-created `CalendarEvent` (blank
  `url`), now linked via Phase 28's `CalendarEventMeta.run`.

  **Verdict: adopt.** The reconciler finds the existing attributed row (via the companion
  FK for this run + night) and updates/re-keys it to `RUN:{run_pk}:{date}` in place,
  rather than leaving it untouched and minting a second, separate event for the same
  physical night. This differs from the queue-run verdict (which chose gap-fill)
  precisely because the underlying risk that killed "adopt" for queue runs — a recurring
  churn loop with a second automated writer (`sync_lco_observation_calendar`, which looks
  up by `url`) — **does not apply symmetrically here**: `load_telescope_runs` looks up by
  `(telescope, instrument, start_time ±5min)`, never by `url`, so re-keying a classical
  event's `url` to `RUN:{pk}:{date}` does not break its own idempotent lookup on a future
  re-run. Adopting therefore gives every classical-run night one stable, uniform key
  identity going forward and closes the `event.telescope_label_meta.run`-always-unset gap
  `_project_calendar_event()`'s docstring names, in one pass rather than leaving legacy
  blank-`url` rows permanently inconsistent with new `RUN:`-keyed ones.

  **The same adopt approach extends to D-01's retired `CAMPAIGN:` events** (had any
  existed) — the reconciler is the sole authority going forward for any event tied to a
  `CampaignRun`, regardless of which prior mechanism created it.

### Per-run reconciliation trigger (RECON-08)

- **D-03: A single shared core function, `reconcile_run(run)` (naming at planner's
  discretion), implements all four pipeline stages for one `CampaignRun`.** The
  management command loops this function over every run (including under `--dry-run`);
  each of the four staff actions calls the identical function for its single run. This is
  the only way RECON-01's "running it a second time changes nothing" and RECON-08's
  "immediate reconcile on a staff decision" can be guaranteed to agree with each other —
  and it matches the project's established precedent (Phase 25: `backfill_range_calendar_
  events` delegates 100% of projection math to a single shared function rather than
  reimplementing it).

- **D-04: The staff-action call stays synchronous and inline, preserving the existing
  asymmetric failure-handling split.** No new dependency (Celery was already rejected in
  Phase 26), so the call blocks the HTTP response exactly as `_project_calendar_event()`
  does today. `approve()` continues to swallow a projection failure (the run stays
  approved even if the calendar write failed); `resolve_site()` continues to treat a
  failure as "projection attempted but failed" and keeps `site_needs_review=True` so the
  action remains retryable. `mark_cancelled`/`mark_weather_failure` follow whichever of
  the two existing patterns they already structurally resemble in `_set_run_status()`.

### Dry-run and failure reporting (RECON-06)

- **D-05: The command's summary follows `import_campaign_csv`'s existing
  created/updated/unchanged/skipped-with-reason counter shape.** `--dry-run` prints the
  identical summary with nothing written. A `skipped` entry always carries a reason
  string (e.g. `blank Observatory.timezone`, `TBD window`, `unresolved site`) so RECON-06's
  "reported and skipped" is a concrete, itemized list, not a bare count — matching
  `backfill_range_calendar_events`'s and `import_campaign_csv`'s existing precedent rather
  than inventing a new reporting shape.

- **D-06: A single run's failure is caught at the batch-loop level, recorded in the
  skipped-with-reason summary, and the batch continues to the next run.** No
  `transaction.atomic()` wrap within one run's reconcile — matches Phase 25's accepted
  partial-projection posture (a mid-run failure, e.g. `sun_event()` raising `ValueError`
  for a blank-timezone `Observatory`, leaves that run's already-written nights in place).
  This is safe because a re-run is idempotent and picks up exactly where it left off; it
  is not a "one-time cost" concern the way the queue-run churn-loop risk was, since
  nothing here re-triggers automatically on every reconcile cycle.

### Claude's Discretion

- The exact name of the shared per-run function (`reconcile_run` used as a placeholder
  above).
- Whether `_project_calendar_event()`/`_calendar_event_title()` are deleted in the same
  commit that wires the staff actions to the reconciler, or in a preceding cleanup commit
  — D-01 only fixes that they are deleted, not the commit sequencing.
- Whether the per-stage breakdown (stage 0-4 counts) is worth adding to D-05's summary
  shape as a supplementary line, beyond the required created/updated/unchanged/skipped
  counts — offered during discussion and not chosen, but not explicitly rejected either.
- Test organisation, and how `mark_cancelled`/`mark_weather_failure`'s existing
  `_set_run_status()` shape maps onto calling the shared reconcile function.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes

- `.planning/ROADMAP.md` §"Phase 29: The Reconciler" — the five success criteria, the
  "Depends on" rationale, and the milestone's "Locked constraints" section (module home,
  no new dependencies, `related_name` not renamed).
- `.planning/REQUIREMENTS.md` — RECON-01..09 and the "Out of Scope" table (no automatic
  merging, no upstreaming, no `related_name` rename, no new dependency, no
  `GenericForeignKey`, `run` link stays nullable).
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` — **read in full before
  planning.** §"Criterion 3 / SPIKE-03" for the locked `RUN:{run_pk}:{date}` /
  `RUN:{run_pk}` two-key-family scheme and the full "Queue-run projection — settled"
  verdict (queue runs never get per-night events); §"D-11" for the adopt-vs-gap-fill
  measured evidence (the two-writer churn analysis this discussion's D-02 verdict builds
  on); §"Domain correction — queue windows are not sets of owned nights" for why the
  original "4 uncovered nights" framing does not apply to queue runs; §"Criterion 2 /
  SPIKE-02" for the four adapter identity mappings, including `CAMPAIGN:{pk}[:date]`
  (D-01 above retires this one).
- `docs/design/canonical_record_spike.rst` — the durable, redaction-free form of the same
  spike decisions.
- `.planning/phases/27-the-canonical-run-record/27-REVIEW.md` and `27-REVIEW-FIX.md` —
  WR-03, the `event.telescope_label_meta.run`-always-unset gap this phase's D-01/D-02
  close.
- `.planning/phases/28-operator-assisted-attribution/28-CONTEXT.md` — D-15 ("done" is an
  empty attribution queue plus a stated remaining count) is the structural precondition
  ATTRIB-06 requires before the first full reconcile sweep.
- `docs/runbooks/telescope_runs_calendar.rst` — documents `backfill_range_calendar_events`
  (to be removed per RECON-09) and the four staff decision actions (to be updated to
  describe immediate reconciliation instead of direct projection).

### Code this phase changes or depends on

- `solsys_code/campaign_views.py:445-554` — `_calendar_event_title()` and
  `_project_calendar_event()`, both retired per D-01; their docstrings (especially the
  WR-03 paragraph) document exactly what the reconciler must take over.
- `solsys_code/campaign_views.py:556-` (`CampaignRunDecisionView`) — `approve()`,
  `resolve_site()`, and `_set_run_status()` (used by mark_cancelled/mark_weather_failure)
  — the four call sites D-03/D-04 rewire.
- `solsys_code/campaign_views.py:797` — the existing `Q(url=f'CAMPAIGN:{run.pk}') |
  Q(url__startswith=f'CAMPAIGN:{run.pk}:')` ownership-scoping query, the direct precedent
  a `RUN:` analogue mirrors verbatim.
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()` (the
  caller-supplied-`lookup` create-or-update contract the reconciler plugs into) and
  `_update_or_unchanged()` (the no-churn field-diffing the no-second-writer-churn analysis
  in D-02 depends on).
- `solsys_code/management/commands/backfill_range_calendar_events.py` — retired per
  RECON-09; also the existing anti-pattern (importing a private helper from the views
  module) the milestone's locked constraints call out.
- `solsys_code/management/commands/import_campaign_csv.py` — the
  created/updated/unchanged/skipped-with-reason summary shape D-05 follows.
- `solsys_code/models.py` — `CampaignRun` (`source`, `telescope_class`, window fields,
  both partial unique constraints), `CalendarEventMeta` (`run` FK, `confirmed_by`/
  `confirmed_at`), `CampaignRunObservation` (the confirmed run↔record link stages 3-4
  read).
- `solsys_code/telescope_runs.py` — `sun_event()`, reused for classical per-night
  sunset/sunrise (kind='sun', matching `_project_calendar_event()`'s existing convention).

### Paired docs (CLAUDE.md rule — required in `files_modified` up front)

- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` — new, paired with the
  new command per ROADMAP.md's explicit note.
- `docs/runbooks/telescope_runs_calendar.rst` — document the reconciler command, the
  per-run staff-action reconcile, and remove the `backfill_range_calendar_events` section
  it retires. Both are in `files_modified` from the start, not follow-ups.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`insert_or_create_calendar_event()`** (`calendar_utils.py`) — the shared no-churn
  create-or-update helper every adapter and the reconciler both use; takes an arbitrary
  caller-supplied `lookup` dict, so the reconciler's `RUN:` key scheme plugs in without
  modifying the helper.
- **`sun_event()`** (`telescope_runs.py`) — dip-corrected sunset/sunrise, already used by
  `_project_calendar_event()`'s ground branch; the reconciler's stage-1 classical
  per-night math reuses this directly rather than reimplementing.
- **`CalendarEventMeta.run`** and **`CampaignRunObservation`** (Phase 27/28) — already
  exist; stages 3-4 read `CampaignRunObservation` for the linked `ObservationRecord`'s
  `scheduled_start`/`scheduled_end`/status, and the reconciler's own container/per-night
  events get their `run` FK set at creation.
- **`Q(url=...) | Q(url__startswith=...)` ownership-scoping pattern**
  (`campaign_views.py:797`) — direct precedent for the reconciler's own bare-key-plus-
  prefix query.

### Established Patterns

- **Pure-logic modules never import views/ephem_utils** — `campaign_gap.py`,
  `campaign_utils.py`, and now `campaign_reconciler.py` (locked module home, milestone
  constraint) all follow this; `campaign_reconciler.py` must not import
  `solsys_code.views` or `solsys_code.ephem_utils` (the latter triggers the ~1.6 GB SPICE
  kernel download at module load).
- **A single shared function backs both the batch command and any single-item call
  site** — Phase 25's `backfill_range_calendar_events` delegating to
  `campaign_views._project_calendar_event()` is the direct precedent D-03 follows (the
  roles are now reversed: the reconciler's own function becomes the one shared
  authority).
- **created/updated/unchanged/skipped-with-reason summary dict** —
  `import_campaign_csv`'s and `backfill_range_calendar_events`'s existing reporting shape,
  reused verbatim per D-05.
- **No-churn field diffing via `_update_or_unchanged()`**, not `django-dirtyfields`/
  `FieldTracker` — already rejected by research; the reconciler's idempotency (RECON-01)
  relies on this existing mechanism.

### Integration Points

- `campaign_views.py` loses `_project_calendar_event()`/`_calendar_event_title()`, and its
  four staff-action call sites gain a `campaign_reconciler` import instead.
- `campaign_reconciler.py` is new — the per-run core function, the ownership-scoping
  query, and the two key-family builders (`RUN:{pk}:{date}` for classical,
  `RUN:{pk}` bare for queue/space).
- A new management command imports `campaign_reconciler`'s per-run function and loops it,
  handling `--dry-run` and the summary dict.
- `docs/runbooks/telescope_runs_calendar.rst` loses its
  `backfill_range_calendar_events` section and gains a reconciler section describing both
  the batch command and the "reconciles immediately on approve/resolve_site/mark_cancelled/
  mark_weather_failure" behavior.

</code_context>

<specifics>
## Specific Ideas

- **The acceptance test is real data, same as Phase 28's.** RECON-07's 19 approved,
  site-resolved 3I/ATLAS runs split 8 QUEUE / 11 CLASSICAL / 0 SPACE (measured in
  `26-DECISION.md`'s "Run-type inventory" finding). The 11 CLASSICAL runs must each show
  up via D-02's adopt-or-mint per-night mechanism; the 8 QUEUE runs via the settled bare
  `RUN:{run_pk}` container, coexisting with their real `ObservationRecord`-derived events.
- **`CampaignRun` pk=1 remains the concrete queue-run reference case** (FTS/MuSCAT4,
  2026-07-07..21, Siding Spring E10, 11 real LCO queue events) — but per the domain
  correction, this phase must NOT mint per-night `RUN:1:{date}` events for it; only the
  one bare `RUN:1` container.
- **Zero live `CAMPAIGN:`-keyed events (D-15) means D-01's cutover has no data-migration
  step** — deleting `_project_calendar_event()` is a clean, evidence-backed removal, not a
  deprecation needing a backfill.

</specifics>

<deferred>
## Deferred Ideas

- **Per-stage (0-4) breakdown in the command's summary output** — offered during
  discussion as an alternative/addition to the created/updated/unchanged/skipped shape;
  not chosen but not rejected either. Left as Claude's Discretion (see above), not a
  locked requirement.
- **v2.3 items untouched here:** adapter rewiring (ADAPT-01..03) — once shipped, the
  reconciler's `RUN:` scheme becomes the ONLY writer for queue-run per-night events too,
  which is the trigger condition `26-DECISION.md` names for revisiting whether the
  bare-container-only queue verdict still holds; provenance-blind gap analysis (GAPB-01);
  status-vocabulary unification (STATUS-01/02); unused-allocation display (UNUSED-01).

### Reviewed Todos (not folded)

- **`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`** —
  extract `SITE_TELESCOPE_MAP` and instrument extraction into their own module (matched at
  score 0.6 on telescope/instrument keywords). Reviewed and declined again, consistent
  with Phases 26, 27, and 28: no RECON requirement behind it, and Phase 29 already carries
  a new module, a new command, four call-site rewires, and the retirement of two existing
  code paths.

</deferred>

---

*Phase: 29-the-reconciler*
*Context gathered: 2026-08-04*
