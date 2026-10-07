---
phase: 34-the-observation-projector-trigger
plan: 04
subsystem: calendar-sync
tags: [jupyter-notebook, sphinx-docs, runbook, observation-projector, sched-06]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    plan: "02"
    provides: "project_observation_calendar's real flag set, summary-line counter names, and the FTN/FTS/SOAR telescope label rename this plan's notebook and runbook document"
  - phase: 34-the-observation-projector-trigger
    plan: "03"
    provides: "the on-page marker legend and the Observation series modal block this plan's runbook describes"
provides:
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb: the sweep's paired, pre-executed demo notebook, executed against the real developer database -- the receiver narrowing [Q]->[S]->[O] with no command run, the real one-time takeover proving RUN:/GEM:/blank-url untouched, a converging second sweep, the whole real corpus reconciled with zero unprojectable records, and the SCHED-06 baseline over 74 pending KEY2026B-004 records"
  - "docs/runbooks/telescope_runs_calendar.rst: the LCO/SOAR sync section replaced by a projector-and-sweep narrative, the marker legend/ring/Observation-series documentation, the Gemini no-read-back caveat, and an updated cheat-sheet/troubleshooting section"
  - ".planning/phases/34-the-observation-projector-trigger/34-UAT.md: the SCHED-06 baseline date/count and the dated re-check table that closes spike 004's PARTIAL verdict"
  - "sync_lco_observation_calendar_demo.ipynb retired alongside its command (D-18/D-19), with both registries (docs/notebooks.rst, CLAUDE.md) updated"
affects: [35-allocation-layer-and-classical-cutover, 36-unattended-operation, 37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 52744
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A pre-executed demo notebook that makes a verification-over-time claim (SCHED-06) runs against the real developer database, not a scratch copy -- unlike this repo's other pre_executed/ notebooks (reconcile_campaign_runs_demo.ipynb, campaign_lifecycle_demo.ipynb), which copy src/fomo_db.sqlite3 to a throwaway file precisely so nothing they do persists. A scratch copy discarded at the end of the run would leave nothing for a later re-execution to diff against."
    - "A one-shot destructive demonstration inside a real-database notebook uses transaction.atomic() plus a deliberately-raised sentinel exception to force a guaranteed rollback, rather than get_or_create-based idempotency -- the receiver demo's throwaway record, target, and user all vanish at the end of the cell, leaving the real database exactly as it was before that cell ran."
    - "A notebook cell added to an existing paired notebook that shares state with other cells (campaign_lifecycle_demo.ipynb) creates its own dedicated fixture object (a separate TargetList/campaign) rather than reusing the notebook's already-established shared fixture, so a new demonstration cannot silently widen an unrelated cell's exact-membership assertion."
    - "A management command's own stderr log lines (one per unprojectable row, naming the exception class) are the reliable source for a notebook's after-the-fact reason table -- re-deriving a 'why' by calling the same classification function again after the sweep already ran does not actually prove what the sweep itself hit."

key-files:
  created:
    - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
    - docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json
    - .planning/phases/34-the-observation-projector-trigger/34-UAT.md
  modified:
    - docs/notebooks.rst
    - CLAUDE.md
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst
    - .planning/todos/completed/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md
    - .planning/todos/completed/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md

key-decisions:
  - "The new notebook runs directly against src/fomo_db.sqlite3 (no FOMO_DATABASE_PATH scratch copy) -- the real, one-time takeover this phase performs is itself the deliverable, and SCHED-06's baseline is only useful if it is captured over the same database a later re-execution will re-check."
  - "The receiver-demo section's throwaway user/target/record all live inside one transaction.atomic() block, forced to roll back via a deliberately-raised local exception class -- confirmed by post-execution queries showing zero residue in the real database."
  - "campaign_lifecycle_demo.ipynb's new D-07 cell creates its own 'D-07 site-match demo campaign' TargetList rather than reusing the notebook's shared `campaign` fixture, after a first execution attempt broke a later cell's exact-set membership assertion over the public campaign table."

patterns-established:
  - "Notebook build-and-execute workflow: construct cell content with nbformat (avoiding raw-string/triple-quote collisions when a cell's own source contains a Python docstring), write unexecuted, then jupyter nbconvert --to notebook --execute --inplace exactly once against the real database for a notebook whose narrative depends on before/after state -- validated first against a scratch copy of the same database so the real, one-shot execution is not spent debugging."

requirements-completed: [TRIG-03, ANNOT-03, SCHED-06]

coverage:
  - id: D1
    description: "project_observation_calendar_demo.ipynb, executed against the real developer database, demonstrates the receiver narrowing one throwaway record's event through [Q]->[S]->[O] with no command run (rolled back, confirmed absent afterward), then the real one-time takeover: 156 legacy facility-url-keyed events re-titled, 3 new events created, and the RUN:/GEM:/blank-url families proven byte-identical (72/0/10 events, before == after) via an explicit assertion."
    requirement: "TRIG-03"
    verification:
      - kind: other
        ref: "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb -- executed cell outputs (cells 4, 8, 9, 10), real database, no mocks"
        status: pass
    human_judgment: false
  - id: D2
    description: "The second sweep over the now-converged corpus reports created: 0, updated: 0, site_lookups: 0 for every facility (LCO and SOAR), printed side by side with the first sweep's non-zero created/updated/site_lookups line, with an explicit assertion; the per-corpus reconciliation cell shows 159 LCO/SOAR records against 159 facility-url-keyed events with zero unprojectable rows, and the per-marker tally sums to 159 across all seven markers."
    requirement: "TRIG-03"
    verification:
      - kind: other
        ref: "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb -- executed cell outputs (cells 10, 12, 13)"
        status: pass
    human_judgment: false
  - id: D3
    description: "sync_lco_observation_calendar_demo.ipynb is deleted (git rm), its docs/notebooks.rst toctree entry and CLAUDE.md notebook-map pairing both replaced with project_observation_calendar_demo.ipynb's pairing, and a grep for the retired command name returns 0 across docs/runbooks/telescope_runs_calendar.rst, docs/notebooks.rst, and CLAUDE.md."
    requirement: "ANNOT-03"
    verification:
      - kind: other
        ref: "grep -rc 'sync_lco_observation_calendar' docs/runbooks/telescope_runs_calendar.rst docs/notebooks.rst CLAUDE.md -- 0 for every file"
        status: pass
    human_judgment: false
  - id: D4
    description: "The runbook's LCO/SOAR section is replaced with a projector-and-sweep narrative (marker table, on-page legend/ring documentation, Observation series subsection, One-time title change note, a 'When would I run the sweep?' subsection with real flags and counter meanings), the cheat-sheet and troubleshooting sections are updated to the real vocabulary (unprojectable/site_lookups/site_lookup_failed, no [UNVERIFIED]/telescope_api_failed), and the Gemini section plus its demo notebook both state the no-read-back caveat and that the projector ignores Gemini records by design."
    requirement: "ANNOT-03"
    verification:
      - kind: other
        ref: "grep-based verify commands over docs/runbooks/telescope_runs_calendar.rst and docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb (see plan 34-04 Task 2 <verify>), all passing; sphinx-build and pre-commit ruff/ruff-format green"
        status: pass
    human_judgment: false
  - id: D5
    description: "The SCHED-06 baseline is captured over the real, pending KEY2026B-004 corpus (74 records: 56 queued, 18 placed) and written to a JSON file beside the notebook; 34-UAT.md records the baseline date, record count, and a dated re-check table with the explicit instruction that only `updatestatus` runs between the baseline and the re-check."
    requirement: "SCHED-06"
    verification:
      - kind: manual_procedural
        ref: "34-UAT.md SCHED-06 section -- PENDING until a real re-check row is filled in after real observing nights"
        status: unknown
    human_judgment: true
    rationale: "SCHED-06 is a verification-over-time claim: the baseline mechanism and its recording are complete and verified now, but the claim itself (a real record narrowing with nothing but updatestatus run) can only be confirmed by re-executing this notebook after real observing nights pass. This plan captures the evidence trail; it does not and cannot close the verdict today."

duration: 40min
completed: 2026-09-11
status: complete
---

# Phase 34 Plan 4: Paired Docs, Runbook Rewrite & SCHED-06 Baseline Summary

**A pre-executed demo notebook run against the real developer database proves the projector's one-time takeover leaves the campaign/Gemini/blank-url calendar namespaces untouched, the runbook's LCO section now documents the projector instead of the retired sync command, and a dated SCHED-06 baseline over 74 real pending `KEY2026B-004` records stands ready for a post-observing-nights re-check.**

## Performance

- **Duration:** 40 min
- **Started:** ~2026-09-11T04:16:00Z
- **Completed:** 2026-09-11T04:55:48Z
- **Tasks:** 2
- **Files modified:** 11 (3 created, 7 modified, 1 deleted)

## Accomplishments

- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` (new, 12 executed
  code cells, ~50 KB): built with `nbformat` and executed once, for real, against
  `src/fomo_db.sqlite3` (validated first against a throwaway scratch copy of the same
  database, including live LCO Observation Portal calls, so the one real execution
  would not be spent debugging). It demonstrates the `post_save` receiver narrowing a
  throwaway record's event `[Q]` -> `[S]` -> `[O]` with no operator command (rolled
  back afterward, confirmed absent), the real one-time takeover of 156 legacy
  facility-url-keyed events with the `RUN:`/`GEM:`/blank-url families proven
  byte-identical before/after, a converging second sweep (`created: 0, updated: 0,
  site_lookups: 0` for every facility), the whole real 159-record LCO/SOAR corpus
  reconciled with zero unprojectable rows, a per-marker tally, and the SCHED-06
  baseline over the 74 pending `KEY2026B-004` records.
- `sync_lco_observation_calendar_demo.ipynb` retired outright (`git rm`), with both
  `docs/notebooks.rst` and CLAUDE.md's notebook map repointed at the new notebook.
- `campaign_lifecycle_demo.ipynb` gains a D-07 cell proving `FTN`/`FTS`/`SOAR` still
  resolve a site-level attribution match through `campaign_attribution`'s
  `OBSERVED_TELESCOPE_SITE_CODES` bridge -- scoped to its own dedicated demo campaign
  after a first attempt broke an unrelated cell's exact-membership assertion.
- `docs/runbooks/telescope_runs_calendar.rst`: the LCO/SOAR section is now "How do
  LCO/SOAR queue observations get onto the calendar?" (the projector, the marker
  vocabulary, the on-page legend/ring, the Observation series pop-up block, the
  one-time title change note) plus a new "When would I run the sweep?" subsection;
  the cheat-sheet, the "Observatory missing timezone" command list, and the
  per-record skip-and-log troubleshooting bullet are all updated to the real
  vocabulary; the Gemini section and its demo notebook both gain the D-21
  no-read-back caveat.
- `.planning/phases/34-the-observation-projector-trigger/34-UAT.md` (new): the
  SCHED-06 baseline (captured 2026-09-11T04:44:59Z, 74 pending records, 56 queued/18
  placed) and a dated re-check table with the explicit "only `updatestatus`" rule.
- Both settled todos moved to `.planning/todos/completed/` with closing notes.

## Task Commits

Each task was committed atomically:

1. **Task 1: The sweep's pre-executed demo notebook -- takeover diff and the SCHED-06
   baseline** - `a87f5f8` (docs)
2. **Task 2: The runbook -- projector and sweep replace the LCO sync section, Gemini
   gains its caveat** - `f0f09d4` (docs)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md)

## Files Created/Modified

- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` - the sweep's
  paired, pre-executed demo notebook (new)
- `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` -
  the SCHED-06 baseline snapshot (new)
- `.planning/phases/34-the-observation-projector-trigger/34-UAT.md` - SCHED-06 tracking (new)
- `docs/notebooks.rst` - toctree entry swap
- `CLAUDE.md` - notebook-map pairing swap
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - new D-07 site-match cell
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` - new
  no-read-back caveat markdown cell, no code changed, no re-execution
- `docs/runbooks/telescope_runs_calendar.rst` - LCO/SOAR section rewrite, cheat-sheet,
  troubleshooting, Gemini caveat, stale "empty for every entry today" correction
- `.planning/todos/completed/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` - closed
- `.planning/todos/completed/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` - closed as overtaken
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` - deleted

## Decisions Made

- Ran the new notebook against the real developer database, not a scratch copy --
  SCHED-06's baseline needs to be captured over the same database a later
  re-execution will re-check, unlike this repo's other `pre_executed/` notebooks.
- Used `transaction.atomic()` plus a deliberately-raised local exception to force a
  guaranteed rollback for the receiver demo, confirmed against the real database
  afterward (zero residual rows).
- Gave `campaign_lifecycle_demo.ipynb`'s new cell its own dedicated demo campaign
  (`TargetList`) instead of reusing the notebook's shared `campaign` fixture -- see
  Deviations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] campaign_lifecycle_demo.ipynb's new D-07 cell broke an unrelated
cell's exact-membership assertion**
- **Found during:** Task 1, first `jupyter nbconvert --execute` attempt on
  `campaign_lifecycle_demo.ipynb`
- **Issue:** The D-07 cell's demo `CampaignRun` was created against the notebook's
  shared `campaign` `TargetList` fixture. A later cell in the same notebook asserts
  the public campaign table's visible run set equals an exact 5-pk set; the new demo
  run's pk joined that set and broke the assertion (`AssertionError`), since the new
  run is `APPROVED` and belongs to the same campaign.
- **Fix:** The D-07 cell now creates and uses its own dedicated `TargetList`
  ('D-07 site-match demo campaign') instead of the shared `campaign` variable, so it
  cannot appear in queries scoped to the shared campaign's own pk.
- **Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
- **Verification:** full notebook re-executed clean end to end, all pre-existing cells'
  assertions (including the exact-membership one) still pass; new cell's own
  assertions pass
- **Commit:** `a87f5f8`

**2. [Rule 2 - Missing Critical] The runbook's "Observation record"/"Observation
group" inline-field description had gone stale in the same phase that fills it in**
- **Found during:** Task 2, reading the manual admin-path section this plan's own
  read_first pointed at
- **Issue:** The runbook stated these two Django-admin inline fields "are filled in
  by code only, and are empty for every entry today. The phase that fills them in is
  named in the roadmap (Phase 34, the observation projector)" -- but this very plan
  (Phase 34) is what fills them in; leaving the sentence unchanged would ship
  documentation that contradicts the behavior this same commit delivers.
- **Fix:** Reworded to state that every LCO/SOAR observation projector-owned entry
  now carries these fields automatically, with a cross-reference to the new LCO/SOAR
  section, while noting they stay blank on any entry the projector does not own.
- **Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
- **Verification:** read-through; no verify command specifically targets this
  sentence, but it is the plan's own read_first excerpt (lines ~891-895) and is now
  internally consistent with the rest of the page
- **Commit:** `f0f09d4`

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug in a notebook fixture, 1 Rule 2
missing-critical documentation correction directly caused by this plan's own shipped
behavior). No scope creep beyond what the plan's own acceptance criteria and CLAUDE.md's
paired-docs rule already required.
**Impact on plan:** None outside the stated scope.

## Issues Encountered

None beyond the deviations above. One environmental note for future maintainers: the
plan's task 2 `sphinx-build` verify command (run with `-D
exclude_patterns=notebooks/*,_build`, matching the pre-commit hook's own invocation)
prints `WARNING: toctree contains reference to nonexisting document` lines for
*every* notebook in `docs/notebooks.rst`'s toctree, including notebooks that
genuinely exist on disk and were untouched by this plan (e.g. `telescope_runs_demo`).
This is a pre-existing side effect of that command's `exclude_patterns=notebooks/*`
override (confirmed present before this plan's changes, for unrelated notebooks) --
the build still exits 0 (`build succeeded, N warnings`), which is what the pre-commit
gate and this plan's own acceptance criteria actually check. The retired notebook's
toctree entry is confirmed absent from the warning list entirely (it is no longer
referenced anywhere), which is the substantive thing this verify command exists to
catch.

## User Setup Required

None -- no external service configuration required. The notebook's one live network
dependency (the LCO Observation Portal, for the one-time observed-telescope lookup)
used this developer environment's already-configured API credentials; no new secret
was added or is required to re-run it.

## Next Phase Readiness

Phase 34 is functionally complete: the projector, its three triggers, the backstop
sweep, the display layer (legend/rings/series decoration), and now the paired docs
and SCHED-06 evidence trail are all shipped and committed. The one open item is
SCHED-06 itself, which is *evidenced* but not yet *closed* -- `34-UAT.md` tracks the
dated re-check this requires after real observing nights pass, and is the artifact a
verifier or a future session should consult before treating SCHED-06 as fully
resolved. No blockers for Phase 35 (Allocation Layer & Classical Cutover): this
plan's files (two notebooks, one runbook page, two todos) are disjoint from anything
Phase 35 is expected to touch.

## Self-Check: PASSED

- `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — FOUND
- `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` — FOUND
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` — CONFIRMED ABSENT
- `.planning/phases/34-the-observation-projector-trigger/34-UAT.md` — FOUND
- `.planning/todos/pending/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` — CONFIRMED ABSENT
- `.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — CONFIRMED ABSENT
- `.planning/todos/completed/2026-09-02-retarget-adapt-03-to-soar-and-caveat-phase-33-gemini-outcome.md` — FOUND
- `.planning/todos/completed/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — FOUND
- Commit `a87f5f8` — FOUND in `git log`
- Commit `f0f09d4` — FOUND in `git log`
- `grep -rc 'sync_lco_observation_calendar' docs/runbooks/telescope_runs_calendar.rst docs/notebooks.rst CLAUDE.md` — 0 for every file
- `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files` — both Passed

---
*Phase: 34-the-observation-projector-trigger*
*Completed: 2026-09-11*
