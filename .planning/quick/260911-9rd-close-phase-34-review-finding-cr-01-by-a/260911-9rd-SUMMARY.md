---
phase: quick-260911-9rd
plan: 01
subsystem: testing
tags: [django, observation-projector, calendar, code-review-followup]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    provides: "observation_projector.py's event_url()/facility_for(), CR-01/CR-02 fixes recorded in 34-REVIEW-FIX.md"
provides:
  - "34-REVIEW-FIX.md's CR-01 entry rewritten from a deferred open item to a Resolved by analysis paragraph naming the shared-request-ID rationale and the user's no-schema-change decision"
  - "A regression test pinning that an LCO record and a SOAR record sharing one observation_id converge on exactly one shared CalendarEvent url"
affects: ["Phase 37 (status vocabulary/provenance work that reads the same event/url identity)"]

actuals:
  tokens: 1813
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Planning-doc resolution notes replace deferral prose in place rather than appending a new finding, keeping frontmatter counts (17/17/0, all_fixed) untouched"

key-files:
  created: []
  modified:
    - .planning/phases/34-the-observation-projector-trigger/34-REVIEW-FIX.md
    - solsys_code/tests/test_observation_projector.py

key-decisions:
  - "CR-01's facility-URL-namespace half is closed by analysis, not code: SOARFacility subclasses LCOFacility, both facilities share one LCO Observation Portal request-ID space, so one observation_id producing one shared event_url() is correct identity, not a collision"
  - "No schema change: a partial unique index on tom_calendar_calendarevent(url) for non-blank urls was named and explicitly not chosen -- available as later hardening if the residual hand-edit path (caught by CR-02's fix) is ever judged worth closing further"

patterns-established: []

requirements-completed:
  - CR-01

coverage:
  - id: D1
    description: "34-REVIEW-FIX.md's CR-01 entry states the resolution (SOAR-inherits-LCO rationale, single-identity argument, manual-edit residual path caught by CR-02, 2026-09-11 evidence figures, user's no-schema-change decision); Notes bullet and Summary parenthetical agree; no deferral wording remains; frontmatter counts (17/17/0, all_fixed) unchanged"
    requirement: CR-01
    verification:
      - kind: other
        ref: "bash verify gate in 260911-9rd-PLAN.md Task 1 <verify> (grep-based: presence of resolution content + absence of deferral phrases + frontmatter unchanged)"
        status: pass
    human_judgment: false
  - id: D2
    description: "test_lco_and_soar_records_sharing_an_observation_id_get_one_shared_url added to TestProjectRecordWrites, asserting event_url() is identical for an LCO and a SOAR record sharing an observation_id, and exactly one CalendarEvent row exists at that url; whole test_observation_projector module (48 tests) passes; ruff and ruff-format clean on the file"
    requirement: CR-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_observation_projector.py#TestProjectRecordWrites.test_lco_and_soar_records_sharing_an_observation_id_get_one_shared_url"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_observation_projector (48 tests, OK)"
        status: pass
      - kind: other
        ref: "pre-commit run ruff / ruff-format --files solsys_code/tests/test_observation_projector.py"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-09-11
status: complete
---

# Quick Task 260911-9rd: Close Phase 34 Review Finding CR-01 (facility-URL-namespace half) Summary

**Recorded CR-01's remaining sub-item in `34-REVIEW-FIX.md` as resolved by analysis (LCO/SOAR share one portal request-ID space, no schema change) and added a regression test pinning the shared `event_url()` behaviour for one `observation_id`.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-09-11T07:08:xx-07:00 (approx, first commit 07:09:01)
- **Completed:** 2026-09-11T07:09:53-07:00 (second commit)
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Rewrote `34-REVIEW-FIX.md`'s CR-01 entry from a "Partial by design" deferral into a "Resolved by analysis" paragraph carrying the SOAR-inherits-LCO rationale, the single-identity argument, the manual-edit residual path (already caught by CR-02's fix), the 2026-09-11 evidence figures (241 events, 0 duplicate non-blank urls, 10 blank), and the user's explicit no-schema-change decision.
- Updated the Notes-and-Follow-ups bullet and the Summary block's `Skipped: 0` parenthetical so both agree with the resolution, with no deferral wording left anywhere in the file. Frontmatter counts (`findings_in_scope: 17`, `fixed: 17`, `skipped: 0`, `status: all_fixed`) left byte-for-byte unchanged.
- Added `test_lco_and_soar_records_sharing_an_observation_id_get_one_shared_url` to `TestProjectRecordWrites`, directly after the existing distinct-ids test, proving one LCO record and one SOAR record built with the same `observation_id` resolve to the identical `event_url()` and converge on exactly one `CalendarEvent` row.

## Task Commits

Each task was committed atomically:

1. **Task 1: Record the CR-01 shared-request-ID resolution in 34-REVIEW-FIX.md** - `4eead9b` (docs)
2. **Task 2: Pin the shared LCO/SOAR event URL with a regression test** - `0a87174` (test)

_Note: no separate plan-metadata commit was made by this executor; STATE.md/final-docs commit is the orchestrator's responsibility per this task's constraints._

## Files Created/Modified
- `.planning/phases/34-the-observation-projector-trigger/34-REVIEW-FIX.md` - CR-01 entry, Notes bullet, and Summary parenthetical rewritten to close the open sub-item by analysis
- `solsys_code/tests/test_observation_projector.py` - new regression test pinning the shared LCO/SOAR event url for one `observation_id`

## Decisions Made
- CR-01's facility-URL-namespace half is closed by analysis, not code: `SOARFacility` subclasses `LCOFacility`, both facilities resolve through the same LCO Observation Portal, so an LCO/SOAR pair sharing one `observation_id` sharing one `event_url()` is the correct single identity, not a latent collision.
- No schema change: a partial unique index on `tom_calendar_calendarevent(url)` for non-blank urls was named as a considered-but-unchosen future hardening, not outstanding work.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 34's code-review fix run now has no open sub-item; `34-REVIEW-FIX.md` is fully closed.
- No production code changed; no paired-docs (notebook/runbook) trigger applies per CLAUDE.md, since no module's behaviour changed.
- Exactly two commits on `issue37-telescope-runs-calendar`, touching exactly the two files in scope, not pushed.

## Self-Check: PASSED

All claimed files exist on disk and both task commits (`4eead9b`, `0a87174`) are present in `git log`.

---
*Phase: quick-260911-9rd*
*Completed: 2026-09-11*
