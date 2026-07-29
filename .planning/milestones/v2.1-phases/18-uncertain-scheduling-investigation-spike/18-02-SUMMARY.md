---
phase: 18-uncertain-scheduling-investigation-spike
plan: 02
subsystem: docs
tags: [rapidfuzz, difflib, campaign_utils, resolve_site, parse_obs_window, sphinx]

# Dependency graph
requires:
  - phase: 18-uncertain-scheduling-investigation-spike (Plan 01)
    provides: Live-tested Findings for all five SCHED-01 criteria (window schema, TBD
      natural-key collision, CSV range/TBD cell shapes, fuzzy-match library scores,
      resolve_site()/obscode confirmation) against the real 3I/ATLAS sheet and the
      live local Observatory DB/MPC Obscodes API.
provides:
  - Locked, falsifiable decisions for all five SCHED-01 criteria, each tied to a
    Plan 01 Finding, gating Phases 19-21 implementation
  - Durable design-doc summary (docs/design/uncertain_scheduling_spike.rst) future
    milestones can reference without digging into .planning/
affects: [19-window-schema-migration, 20-range-tbd-import-and-asset-aware-coverage-gap,
  21-site-disambiguation-and-submitter-contact-opt-in]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Investigation-spike decision doc + durable .rst dual-doc pattern (mirrors
      Phase 13's D-10 precedent) for foundational-architecture findings that gate
      multiple downstream phases

key-files:
  created:
    - docs/design/uncertain_scheduling_spike.rst
  modified:
    - .planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md
    - docs/design/design.rst

key-decisions:
  - "Window schema (SCHED-01 criterion 1): confirmed as-is - nullable window_start/window_end DateField pair, validated against every real cell shape from Plan 01's Finding"
  - "Fuzzy-match library (SCHED-01 criterion 4): split verdict - difflib primary/default (no new dependency), rapidfuzz deferred until a real matching advantage is demonstrated on a wider candidate pool"
  - "Obscode widening (SCHED-01 criterion 5): no widening needed - confirmed against the live Observatory.obscode max_length=4 field definition"
  - "TBD natural key (SCHED-01 criterion 2): fold contact_person into the natural key for null-window rows via a partial/conditional UniqueConstraint, restated for Phase 19 to design the exact mechanism"
  - "CSV range/TBD parsing (SCHED-01 criterion 3): extend parse_obs_window()'s existing pattern-per-shape discipline (_HHMM_RANGE/_APPROX_HOUR/_BARE_HOUR_UTC) to Obs. Date, one rule per real shape, checking both Obs. Date and UT Time Range per D-04"

patterns-established: []

requirements-completed: [SCHED-01]

coverage:
  - id: D1
    description: "18-DECISION.md's Recommendation section completed for all five SCHED-01 criteria, each tied to a Plan 01 Finding"
    requirement: "SCHED-01"
    verification:
      - kind: other
        ref: "grep gate: '## Recommendation' present, no placeholder marker, rapidfuzz|difflib mentioned, contact_person mentioned, _HHMM_RANGE|_APPROX_HOUR|_BARE_HOUR_UTC mentioned, no email-address leak"
        status: pass
    human_judgment: true
    rationale: "Structural grep gates confirm required content is present, but whether each recommendation is genuinely grounded in its Plan 01 Finding (not merely mentioning the right keywords) and whether the fuzzy-library split verdict correctly reflects the recorded scores requires a human read of the reasoning."
  - id: D2
    description: "docs/design/uncertain_scheduling_spike.rst created following the eso_feasibility_spike.rst skeleton, stating the same fuzzy-library and window-schema verdicts as 18-DECISION.md"
    requirement: "SCHED-01"
    verification:
      - kind: other
        ref: "grep gate: file exists, rapidfuzz|difflib mentioned, list-table present, title underline present, no email-address leak"
        status: pass
      - kind: other
        ref: "sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build"
        status: pass
    human_judgment: true
    rationale: "Grep gates and a successful Sphinx build confirm the doc is structurally valid and renders, but whether it reads as a faithful, non-divergent summary of 18-DECISION.md's verdicts (not just keyword-matching) requires a human read."

# Metrics
duration: 12min
completed: 2026-07-09
status: complete
---

# Phase 18 Plan 2: Recommendation and Durable Summary Summary

**Locked all five SCHED-01 decisions (window schema, TBD natural key, CSV range/TBD parsing rules, fuzzy-match library split verdict, no obscode widening) into 18-DECISION.md's Recommendation section and a new durable `docs/design/uncertain_scheduling_spike.rst`, each recommendation tied directly to a Plan 01 Finding.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-07-09
- **Tasks:** 2
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments
- Completed `18-DECISION.md`'s `## Recommendation` section for all five SCHED-01 criteria, replacing the `<!-- completed in Plan 02 -->` placeholder, each recommendation citing the specific Plan 01 Finding it rests on (not asserted independently)
- Reached a split verdict on the fuzzy-match library question: `difflib` as the primary/default choice (no new dependency, no case in the D-09 corpus where `rapidfuzz` demonstrated a real advantage), with `rapidfuzz` deferred to `pyproject.toml` only if a future wider candidate pool proves it necessary
- Created `docs/design/uncertain_scheduling_spike.rst` following the `eso_feasibility_spike.rst` skeleton exactly (title/section underline convention, bold Key finding one-liners, criterion->decision->phase `list-table`), stating the identical fuzzy-library and window-schema verdicts as `18-DECISION.md` with no divergence
- Wired the new `.rst` into `docs/design/design.rst`'s toctree so it is reachable from the built Sphinx docs site, matching the pattern for the other design docs

## Task Commits

Each task was committed atomically:

1. **Task 1: Complete 18-DECISION.md — Recommendation for all five SCHED-01 criteria** - `a17e70d` (docs)
2. **Task 2: Write the durable summary docs/design/uncertain_scheduling_spike.rst** - `9fd8ead` (docs)

_Note: no separate plan-metadata commit yet — this final commit follows below._

## Files Created/Modified
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` - Recommendation section completed for all five SCHED-01 criteria; header status line updated to "Complete"
- `docs/design/uncertain_scheduling_spike.rst` - new durable summary doc (created)
- `docs/design/design.rst` - added `uncertain_scheduling_spike` to the design-docs toctree

## Decisions Made
- Fuzzy-match library split verdict (difflib primary, rapidfuzz deferred): both libraries produced identical matches — including the same two false positives (`X09`->`'309'`, `C65`->`'F65'`) and the same clean misses (`N50`, `X07`, blank) — on Plan 01's real D-09 corpus, so no live-test evidence favored `rapidfuzz`'s extra dependency weight over stdlib `difflib`. This matches RESEARCH Open Questions §1's anticipated "split verdict" outcome.
- No obscode widening: confirmed directly against the live `Observatory._meta.get_field('obscode').max_length` value (4), not merely restated from the pre-spike assumption.
- Header status line in `18-DECISION.md` updated from "Recommendation and Durable summary to be completed in Plan 02" to "Complete" / "now recorded below" to avoid literally containing the string "completed in Plan 02" (which the plan's own placeholder-removal grep gate matches against, not just the HTML-comment marker) — a small wording adjustment made while satisfying the verification gate, not a content change.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Wired the new .rst into docs/design/design.rst's toctree**
- **Found during:** Task 2 (durable summary creation)
- **Issue:** The plan's `files_modified` only listed `docs/design/uncertain_scheduling_spike.rst`, but the house convention (established by every other design doc, including `eso_feasibility_spike.rst`) requires new design docs to be added to `docs/design/design.rst`'s `toctree` directive to be reachable from the built Sphinx site. Without this, Sphinx would emit an "orphaned document" warning and the page would be unreachable from the docs site navigation.
- **Fix:** Added `uncertain_scheduling_spike` as a new line in `docs/design/design.rst`'s `toctree` block, alongside `eso_feasibility_spike` and the other design docs.
- **Files modified:** `docs/design/design.rst`
- **Verification:** `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` — build succeeded with 4 pre-existing warnings, all in unrelated `autoapi/fomo/urls/index.rst` output, none touching the new/modified files.
- **Committed in:** `9fd8ead` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for the new durable summary to actually be part of the built docs site, matching the established convention for every other design doc. No scope creep — no content changes beyond the one added toctree line.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All five SCHED-01 decisions are locked and evidence-backed: Phase 19 (window-schema migration + TBD natural-key constraint) and Phase 20 (CSV range/TBD parsing rules) can proceed without re-deriving anything from the real 3I/ATLAS sheet.
- Phase 21 (fuzzy-match site UI) has its library choice (difflib primary) and the `500@-170`-vs-`274` notation-gap caveat already documented as a distinct problem it must still solve.
- No blockers. Phase 18 (uncertain-scheduling-investigation-spike) is complete; the v2.1 milestone's next phase (19) is ready to plan.

---
*Phase: 18-uncertain-scheduling-investigation-spike*
*Completed: 2026-07-09*

## Self-Check: PASSED

- FOUND: `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`
- FOUND: `docs/design/uncertain_scheduling_spike.rst`
- FOUND: `docs/design/design.rst`
- FOUND: commit `a17e70d`
- FOUND: commit `9fd8ead`
