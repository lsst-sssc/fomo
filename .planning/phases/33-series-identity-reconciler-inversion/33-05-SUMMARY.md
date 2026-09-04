---
phase: 33-series-identity-reconciler-inversion
plan: 05
subsystem: campaign-coordination
tags: [reconciler, calendar-events, documentation, jupyter-notebooks, attribution]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion (plan 01)
    provides: "reconcile_run() skip-the-night rule, ReconcileResult.skipped_nights, campaign_decoration() simple_tag, event_form.html decoration block"
  - phase: 33-series-identity-reconciler-inversion (plan 02)
    provides: "Month-cell .cal-campaign-chip marker, anchored campaign-table run-{pk} rows"
  - phase: 33-series-identity-reconciler-inversion (plan 03)
    provides: "CalendarEventMeta.observation_record/observation_group carrier fields, migration 0017"
  - phase: 33-series-identity-reconciler-inversion (plan 04)
    provides: "unlink_event_from_run() -- the single writer that clears an attribution and its audit stamps"
provides:
  - "D-04's real-database proof: a full reconcile_campaign_runs sweep leaves every event outside the RUN: namespace byte-identical, demonstrated against the real developer database (166 non-RUN: events compared, 0 differences)"
  - "A pre-sweep dry-run inspection that previews every touchable url (via reconcile_run(dry_run=True) plus the reconciler's own key builders) and asserts it stays inside RUN_URL_NAMESPACE before any write happens"
  - "A skip-rule demonstration: an attributed night gets no reconciler event, and ReconcileResult.skipped_nights counts it"
  - "A campaign_lifecycle_demo demonstration that the attribution decoration (modal block + month-cell chip) survives a from-scratch rewrite of an event's own title/description (ROADMAP criterion 3)"
  - "A campaign_lifecycle_demo demonstration that unlink_event_from_run() removes only the decoration and audit stamps, deleting nothing (ROADMAP criterion 4)"
  - "An operator runbook whose reconciler, pop-up and attribution-queue sections describe CalendarEventMeta.run as attribution throughout, document the skip rule and the one-time title change, and name the two new read-only observation_record/observation_group admin fields"
affects: []

# Actuals (#2632)
actuals:
  tokens: 22331
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Real-database before/after diff proof: snapshot every row outside a namespace, preview the write in Python via the production function's own dry_run mode (never by parsing a management command's stdout, which may carry only an aggregate summary), run the real write, snapshot again, and assert the diff is empty -- used here for D-04's real-DB half of a reconciler-inversion proof; reusable for any future 'this sweep must not touch rows it doesn't own' claim."
    - "New demo cells inserted after the CODE cell that defines a variable they depend on, located by a marker string unique to that code cell (never a substring that also appears in the preceding markdown prose) -- a marker present in both cells silently inserts new cells between the markdown and its own code cell, breaking variable ordering."
    - "Demonstrating a skip rule without breaking notebook re-run idempotency: delete the row this SAME notebook run already created (not a pre-existing developer-database row) to make room for the stand-in fixture, rather than mutating a CampaignRun's window fields that are part of the model's own get_or_create() natural-key lookup."

key-files:
  created: []
  modified:
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Task 1's D-04 diff is split across two separate code cells (a PRE-SWEEP DRY-RUN INSPECTION cell and a POST-SWEEP DIFF cell), not one -- the plan's own verify script asserts the two headings live in cells at different indices (min(d) < min(p)), which a single combined cell would fail."
  - "The skip-rule demo cell deletes classical_run's own already-existing RUN:{pk}:{date} event for the target night (a row created earlier in this same notebook run) rather than extending the run's window_end -- window_start/window_end are part of the CampaignRun row's own update_or_create() lookup key in this notebook, so mutating window_end in place would make a second run of the notebook create a duplicate CampaignRun instead of finding the existing one."
  - "campaign_lifecycle_demo's four new cell pairs are inserted immediately after the CODE cell that sets meta = CalendarEventMeta.objects.get(...) (located via a marker string unique to that cell), not after the 'Confirm the attribution' markdown cell, whose prose also contains the string used to locate the insertion point -- inserting after the markdown would place the new cells (which read meta) before meta is ever assigned."
  - "The observation_record/observation_group demo cell finds-or-creates its demo Target by name (Target.name is globally unique) rather than always calling NonSiderealTargetFactory.create(), and finds-or-creates the ObservationRecord/ObservationGroup by their own natural keys, so the cell stays safely re-runnable against any developer database."
  - "The runbook's dismissal-section sentence 'can never be mistaken for ownership' was reworded to 'can never be mistaken for a confirmed attribution' -- not one of the plan's three literal forbidden phrases, but the same D-17 attribution-not-ownership concern the plan's whole-page grep sweep asks for."

patterns-established:
  - "A management command's --dry-run stdout is an aggregate summary only; per-row/per-night detail for a notebook proof must come from the production function's own return value (ReconcileResult here), never from parsing printed command output that was never designed to carry that detail."

requirements-completed: [ANNOT-01, ANNOT-02, PROJ-04]

coverage:
  - id: D1
    description: "reconcile_campaign_runs_demo.ipynb proves, with real executed output, that a full sweep leaves every non-RUN: event's url, title and attribution unchanged (empty diff, non-trivial comparison count) -- D-04's real-database half of ROADMAP criterion 2"
    requirement: ANNOT-01
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb#POST-SWEEP DIFF cell -- printed 'Compared 166 non-RUN:-namespaced CalendarEvent row(s)... Differences found: 0'"
        status: pass
    human_judgment: false
  - id: D2
    description: "The dry-run preview (reconcile_run(dry_run=True) per run) runs before the real sweep, and asserts every url the sweep can touch stays inside RUN_URL_NAMESPACE, including a direct check that the one pre-existing non-RUN: attributed row is outside that touchable set"
    requirement: ANNOT-01
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb#PRE-SWEEP DRY-RUN INSPECTION cell -- printed 'every member starts with RUN_URL_NAMESPACE: True' and 'Any of those urls in the touchable set: False'"
        status: pass
    human_judgment: false
  - id: D3
    description: "The inspection cell's per-run detail (including skipped_nights) comes from reconcile_run()'s returned ReconcileResult, not from parsing reconcile_campaign_runs --dry-run stdout, which is labelled in the cell as an aggregate-only cross-check"
    requirement: ANNOT-01
    verification:
      - kind: other
        ref: "grep for 'reconcile_run(' and 'dry_run=True' inside the PRE-SWEEP DRY-RUN INSPECTION cell's source -- 1 match"
        status: pass
    human_judgment: false
  - id: D4
    description: "A classical night with an attributed non-RUN: event gets no reconciler event, and the run's ReconcileResult reports it as a skipped night"
    requirement: ANNOT-01
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb#skip-rule cell -- printed 'RUN:59:2026-09-02' present: False and skip_result.skipped_nights: 1"
        status: pass
    human_judgment: false
  - id: D5
    description: "campaign_lifecycle_demo.ipynb shows the campaign decoration rendered from the attribution link on a real request, still rendered after the event's own title and description are rewritten from scratch"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb#Criterion 3 cell -- printed 'Modal still shows \"Attributed campaign run\": True' and 'Month view still shows the campaign chip marker: True'"
        status: pass
    human_judgment: false
  - id: D6
    description: "campaign_lifecycle_demo.ipynb shows that clearing the attribution removes the decoration and deletes nothing"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb#Criterion 4 cell -- printed 'CalendarEvent.objects.count() before/after unlink' equal, and 'Modal shows \"Attributed campaign run\" after unlink: False'"
        status: pass
    human_judgment: false
  - id: D7
    description: "docs/runbooks/telescope_runs_calendar.rst describes CalendarEventMeta.run as an attribution throughout its three affected sections -- attributed to the run, never owned by it"
    requirement: PROJ-04
    verification:
      - kind: other
        ref: "grep -ci 'attributed to' docs/runbooks/telescope_runs_calendar.rst (6) and grep -n -i 'owning campaign run|owns the event|owned by this run' (no match)"
        status: pass
    human_judgment: false
  - id: D8
    description: "The runbook tells an operator what the reconciler does with a night that already has an attributed event, and that the one-time calendar-title change is expected on the next sweep -- readable and comprehensible without prior access to this phase's planning documents"
    requirement: PROJ-04
    verification: []
    human_judgment: true
    rationale: "The plan's own <human-check> asks a human to read the rewritten pop-up section end to end and confirm an operator with no context could answer what makes the block appear/disappear, what clearing destroys, and what the two new fields are for -- a prose-comprehension judgment no automated check can substitute for. Deferred to phase-level UAT per workflow.human_verify_mode=end-of-phase."
  - id: D9
    description: "Both notebooks are committed with their executed output, regenerated via jupyter nbconvert --to notebook --execute --inplace"
    requirement: PROJ-04
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace exit 0 for both notebooks; every code cell in both committed files carries outputs (0 cells with no outputs)"
        status: pass
    human_judgment: false

# Metrics
duration: 44min
completed: 2026-09-04
status: complete
---

# Phase 33 Plan 05: Paired-Docs Catch-Up -- D-04's Real-Database Proof and the Attribution Story Summary

**Regenerated both v2.2/v2.4 demo notebooks with executed output proving the reconciler touches nothing outside its own namespace on a real 240-event database and that campaign decoration survives a rewrite and disappears cleanly on unlink, then rewrote the operator runbook's reconciler/pop-up/attribution sections from ownership to attribution wording.**

## Performance

- **Duration:** 44 min
- **Started:** 2026-09-04T17:03:14Z
- **Completed:** 2026-09-04T17:47:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `reconcile_campaign_runs_demo.ipynb` gained a two-cell D-04 proof: a `PRE-SWEEP DRY-RUN INSPECTION` cell that snapshots every non-`RUN:`-namespaced `CalendarEvent` (166 rows against the real dev database), previews the sweep by calling `reconcile_run(run, dry_run=True)` for all 53 real + demo `CampaignRun`s, builds the touchable url set from the reconciler's own key builders (`run_container_url`/`run_night_url`/`owned_events`), and asserts every url stays inside `RUN_URL_NAMESPACE` -- including a direct check that the one pre-existing non-`RUN:` attributed row (event pk=159, from `campaign_lifecycle_demo`'s own earlier committed execution) is outside that set. A separate `POST-SWEEP DIFF` cell then runs the real full sweep and proves the before/after diff over those same 166 rows is empty.
- A skip-rule demo cell removes `classical_run`'s own already-created `RUN:{pk}:{date}` event for one night, seeds a blank-url stand-in attributed via `CalendarEventMeta`, reconciles the run again, and prints the attributed event's unchanged url, the absence of a `RUN:`-keyed event for that night, and `skip_result.skipped_nights == 1`.
- `campaign_lifecycle_demo.ipynb` gained four new cell pairs after the existing attribution-confirmation cells: the decoration rendered on a real `calendar:update-event` GET (campaign name, telescope/instrument, window, run status, anchored campaign-table link); the decoration surviving a from-scratch title/description rewrite (ROADMAP criterion 3, including the month-cell `.cal-campaign-chip` marker); `unlink_event_from_run()` clearing only the attribution and its audit stamps while the `CalendarEvent` count and fields survive untouched (ROADMAP criterion 4); and a cell wiring `observation_record`/`observation_group` onto the now-unattributed event's `CalendarEventMeta` row, confirming it is still offered by `orphan_calendar_events()` exactly like any other unattributed event (D-15).
- `docs/runbooks/telescope_runs_calendar.rst`'s "How do I get every campaign run onto the calendar?" section replaced its stale "protects a night adopted from `load_telescope_runs`" framing with the real skip rule and added a "One-time title change" paragraph; "Why doesn't the calendar pop-up show a 'Campaign run' block?" was retitled to "...an 'Attributed campaign run' block?" and rewritten to describe the month-cell marker, the pop-up block's contents and anchored link, the admin-inline clear-removes-only-the-decoration-and-audit-stamps behavior, and the two new read-only `observation_record`/`observation_group` fields; "How do I attribute existing calendar events and observation records to a run?" gained a paragraph stating attribution never changes the entry itself. A whole-page sweep removed every remaining ownership phrase.

## Task Commits

Each task was committed atomically:

1. **Task 1: Reconciler demo notebook -- the skip rule and the empty real-database diff** - `bb0b92e` (feat)
2. **Task 2: Campaign lifecycle notebook -- decoration from the link, surviving a rewrite, gone on unlink** - `91a6556` (feat)
3. **Task 3: Operator runbook -- attribution wording, the skip rule, and the new links** - `c57eb12` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - D-04 real-database diff proof (2 cells) and skip-rule demo (1 cell), executed
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - decoration, criterion 3, criterion 4 and series-identity-links cell pairs (4 pairs), opening/closing markdown updated, executed
- `docs/runbooks/telescope_runs_calendar.rst` - reconciler, pop-up and attribution-queue sections restated as attribution; skip rule, one-time title change, and two new admin fields documented

## Decisions Made

- Split the D-04 diff into two separate code cells so the plan's own cell-ordering verify script (which requires the `PRE-SWEEP DRY-RUN INSPECTION` and `POST-SWEEP DIFF` headings to live in different, ordered cells) passes.
- Demonstrated the skip rule by deleting this same notebook run's own already-created event for the target night, rather than extending `classical_run.window_end` (which is part of that row's own `update_or_create()` natural-key lookup and would break re-run idempotency).
- Located the insertion point for `campaign_lifecycle_demo.ipynb`'s new cells by a marker string unique to the confirm-attribution CODE cell, after discovering the naive marker (present in both the markdown and the code cell) inserted the new cells before `meta` was ever assigned.
- Reworded one incidental "mistaken for ownership" phrase in the attribution-queue section to "mistaken for a confirmed attribution" -- not one of the plan's three literal forbidden strings, but the same attribution-not-ownership concern its whole-page sweep instruction covers.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All three tasks' `<verify>` commands, `<acceptance_criteria>`, and the plan-level `<verification>` block (both notebooks re-executed with 0 cell errors, `pre-commit run --all-files` clean including the Sphinx docs build, the project full-suite test command's 980 + 40 tests passing) were re-run and confirmed after each task.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CLAUDE.md's paired-docs rule is satisfied for every module Phase 33 changed: both demo notebooks and the operator runbook now reflect the reconciler inversion, the display-time decoration, and the new `CalendarEventMeta` link fields.
- D-04's real-database proof exists and is committed with executed output: a full sweep against the real developer database (240 events, 85 companion rows baseline) leaves everything outside the reconciler's own namespace byte-identical.
- ANNOT-01, ANNOT-02 and PROJ-04 -- each also declared by sibling plans in this phase's shared-ID gate -- are now ready to flip to Complete, since this is the last plan in the phase declaring them and every sibling plan (33-01 through 33-04) already has its own SUMMARY.md.
- Phase 33 is complete: all five plans (33-01 through 33-05) have SUMMARY.md files. No blockers for Phase 34 (The Observation Projector & Trigger).

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-04*

## Self-Check: PASSED

All 3 modified files verified present on disk; all 3 task commit hashes (`bb0b92e`, `91a6556`,
`c57eb12`) verified present in git log. Full acceptance-criteria and `<verify>` re-run: both
notebooks re-executed with `jupyter nbconvert --to notebook --execute --inplace` (exit 0, no
cell errors); every plan-authored grep/python one-liner (outputs-count, contact-field-absence,
skipped_nights presence, PRE-SWEEP/POST-SWEEP cell-ordering, `reconcile_run(dry_run=True)`
presence, `RUN_URL_NAMESPACE` presence, three-marker sum, no-outputs count, `solsys_code.views`
literal absence, `attributed to` count, ownership-phrase absence) re-run and matched its
expected non-failure condition; `pre-commit run --all-files` passed (ruff, ruff-format,
notebook-output handling, Sphinx docs build); the project full-suite test command (980 + 40
tests, OK).
