---
phase: quick-260613-eb1
plan: 01
status: complete
subsystem: docs
tags: [jupyter, notebook, telescope_runs, django, documentation-convention]

# Dependency graph
requires:
  - phase: 01-site-ephemeris-helper
    provides: solsys_code/telescope_runs.py (get_site, horizon_dip, sun_event, SITES)
provides:
  - Pre-executed demo notebook for telescope_runs.py
  - "Demo Notebooks" convention documented in PROJECT.md for future phases
affects: [future phase summaries, GSD plan-phase context selection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Demo notebooks for DB-dependent features go under docs/notebooks/pre_executed/ and are excluded from notebooks.rst"

key-files:
  created:
    - docs/notebooks/pre_executed/telescope_runs_demo.ipynb
  modified:
    - .planning/PROJECT.md

key-decisions:
  - "Pre-executed notebook outputs were stripped by the repo's jupyter-nb-clear-output pre-commit hook; the notebook ships with empty outputs/execution_count as a result (repo convention takes precedence over the plan's 'realistic outputs' guidance)"

patterns-established:
  - "Each phase should ship a demo notebook (pre_executed/ for DB-dependent features, notebooks/ otherwise) as part of its Definition of Done"

requirements-completed: [SITE-01, EPHEM-01, EPHEM-03]

# Metrics
duration: 12min
completed: 2026-06-13
---

# Phase quick-260613-eb1: Demo Notebook for telescope_runs.py Summary

**Pre-executed Jupyter notebook demonstrating get_site(), horizon_dip(), and sun_event() for the NTT site, plus a new "Demo Notebooks" convention added to PROJECT.md.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-13T17:13:00Z
- **Completed:** 2026-06-13T17:25:00Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- Created `docs/notebooks/pre_executed/telescope_runs_demo.ipynb`, an 11-cell notebook walking through Django setup, `SITES`, `get_site('NTT')`, `horizon_dip()`, and `sun_event()` for both 'sun' and 'dark' kinds.
- Added a "Demo Notebooks" subsection to `.planning/PROJECT.md` (after Constraints) establishing the per-phase demo-notebook convention, with a location rule (`pre_executed/` vs `notebooks/`) and a reference to the new Phase 01 example.
- Verified `ruff check .` and `ruff format --check .` introduce no new regressions on touched files.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create the pre-executed telescope_runs demo notebook** - `0f52731` (docs)
2. **Task 2: Add Demo Notebooks convention to PROJECT.md and run quality gates** - `1a36914` (docs)

**Plan metadata:** (pending orchestrator docs commit)

## Files Created/Modified
- `docs/notebooks/pre_executed/telescope_runs_demo.ipynb` - Pre-executed demo notebook for `solsys_code/telescope_runs.py` (Django setup, `get_site`, `horizon_dip`, `sun_event`)
- `.planning/PROJECT.md` - New "Demo Notebooks" convention subsection documenting the per-phase demo-notebook expectation

## Decisions Made
- The repo's `jupyter-nb-clear-output` pre-commit hook strips all cell outputs and execution counts on commit. The plan asked for "realistic outputs" to make the notebook read as pre-executed, but the hook (an existing, enforced repo convention per CLAUDE.md) overrides this — the committed notebook has empty `outputs`/`execution_count` fields. The notebook content (markdown explanations, code, and structure) still fully demonstrates `get_site()`, `horizon_dip()`, and `sun_event()` and satisfies the plan's verification checks (valid nbformat-4 JSON, `django.setup` before the `solsys_code` import, and usage of all three helpers).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Pre-commit hook stripped pre-executed notebook outputs**
- **Found during:** Task 1 commit
- **Issue:** The notebook was written with realistic stream outputs and execution counts (per the plan's instructions), but the repo's `jupyter-nb-clear-output` pre-commit hook (enforced, per CLAUDE.md) rewrote the file to remove all outputs and execution counts before allowing the commit.
- **Fix:** Re-staged the hook-modified file and committed as-is. Verified the resulting notebook still satisfies all plan verification checks (nbformat 4, `django.setup`, imports, helper usage present in source).
- **Files modified:** `docs/notebooks/pre_executed/telescope_runs_demo.ipynb`
- **Verification:** Re-ran the plan's automated verify command against the post-hook file; all assertions pass.
- **Committed in:** `0f52731` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, pre-commit hook compliance)
**Impact on plan:** No scope creep. The notebook's instructional content and verification criteria are unaffected; only the example output values (which were illustrative, not load-bearing) were removed by the repo's standard hook.

## Issues Encountered
None beyond the pre-commit hook deviation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The "Demo Notebooks" convention in PROJECT.md is now available for future phase summaries/plans to reference as part of Definition of Done.
- No blockers for Stage 2 decision (deferred per STATE.md).

---
*Phase: quick-260613-eb1*
*Completed: 2026-06-13*

## Self-Check: PASSED
