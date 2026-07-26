---
phase: quick-260726-kdp
plan: 01
subsystem: docs
tags: [sphinx, rst, runbook, claude-md, telescope-runs-calendar]

# Dependency graph
requires:
  - phase: none
    provides: n/a (documentation-only quick task)
provides:
  - Corrected operator runbook (load_telescope_runs --campaign optionality, [QUEUED] guard)
  - New backfill_lco_observation_records operator documentation section + cheat-sheet row
  - Directory-scoped CLAUDE.md paired-deliverable rule covering docs/runbooks/
  - DEFERRED.md recording out-of-scope doc gaps
affects: [future runbook edits, future GSD plans touching docs/runbooks/ or the four paired modules]

# Tech tracking
tech-stack:
  added: []
  patterns: ["CLAUDE.md rule scoped by directory instead of filename enumeration"]

key-files:
  created:
    - .planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - CLAUDE.md

key-decisions:
  - "Disambiguated load_telescope_runs --campaign (optional) from import_campaign_csv --campaign (required) with an explicit cross-reference note"
  - "Corrected the [QUEUED] claim to reflect the successful-terminal-status guard in sync_lco_observation_calendar._title_for, without expanding into full title-prefix coverage (deferred)"
  - "Documented backfill_lco_observation_records read directly from source, including the reuse-vs-build distinction for --create-missing-targets and the different meaning of omitting --campaign there vs on load_telescope_runs"
  - "Rewrote the CLAUDE.md paired-deliverable rule to be directory-scoped on docs/runbooks/ rather than a filename list, while retaining the notebook pairing map, trigger condition, files_modified requirement, nbconvert mechanics, and all four subagent roles"
  - "Left the Troubleshooting section byte-identical and did not add fetch_jplsbdb_objects, per house rules"

requirements-completed: [DOC-01, DOC-02, DOC-03, DOC-04]

# Metrics
duration: ~20min
completed: 2026-07-26
---

# Quick Task 260726-kdp: Close Operator Runbook Drift and Broaden Paired-Deliverable Rule Summary

**Corrected three factual drifts in the telescope-runs-calendar operator runbook (--campaign optionality, the [QUEUED] guard, and the undocumented backfill_lco_observation_records command), and re-scoped CLAUDE.md's paired-deliverable rule from a four-notebook enumeration to a directory-scoped rule covering docs/runbooks/.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- `docs/runbooks/telescope_runs_calendar.rst` no longer misdescribes `load_telescope_runs --campaign` as required-adjacent or conflatable with `import_campaign_csv`'s required `--campaign`, and no longer claims every unplaced LCO record becomes `[QUEUED]`
- `backfill_lco_observation_records` — added 2026-07-19 and extended three times, never documented — now has a full operator section and cheat-sheet row, including the `--create-missing-targets` reuse-vs-build distinction and the post-create status refresh
- CLAUDE.md's paired-deliverable rule is now future-proof: any page under `docs/runbooks/` is covered by directory, not by an enumeration that would need updating every time a new runbook page ships
- `DEFERRED.md` records four genuine but deliberately out-of-scope doc gaps so they aren't silently dropped

## Task Commits

Each task was committed atomically:

1. **Task 1: Correct load_telescope_runs --campaign and the [QUEUED] claim (DOC-01, DOC-03)** - `81dc276` (docs)
2. **Task 2: Document backfill_lco_observation_records (DOC-02)** - `fd19847` (docs)
3. **Task 3: Broaden the CLAUDE.md paired-deliverable rule to docs/runbooks/, record deferrals (DOC-04)** - `e709bf4` (docs)

_All three commits are documentation-only; no Python was touched._

## Files Created/Modified
- `docs/runbooks/telescope_runs_calendar.rst` - Corrected `--campaign` optionality and `[QUEUED]` claim; added a new `backfill_lco_observation_records` operator section and cheat-sheet row (now 6 command rows)
- `CLAUDE.md` - Rewrote the paired-deliverable rule (lines ~106-129) to be directory-scoped on `docs/runbooks/`, net +4 lines
- `.planning/quick/260726-kdp-close-operator-runbook-drift-and-broaden/DEFERRED.md` - New file recording 4 out-of-scope doc gaps

## Decisions Made
- The `backfill_lco_observation_records` documentation was written strictly from reading `solsys_code/management/commands/backfill_lco_observation_records.py` itself (per critical execution note 4), not from commit messages or the plan's paraphrase. On review, the plan's description of the command's behavior matched the source exactly — no discrepancy to flag.
- Kept the CLAUDE.md rule tight: net growth is +4 lines (28 insertions, 24 deletions), well under the ≤15-line cap, while still retaining every element the house rules required to survive (pairing map, trigger, `files_modified` requirement, nbconvert mechanics, all four subagent roles, and the breach history — now with a third entry for `260726-kdp`).
- Deferred the LCO sync's other title prefixes (`[EXPIRED]`/`[FAILED]`/`[CANCELLED]`/`[UNVERIFIED]`) rather than expanding Task 1's `[QUEUED]` fix into full coverage, per house rule 6 and the task's explicit scoping instruction — recorded in `DEFERRED.md` item 4.

## Deviations from Plan

None - plan executed exactly as written. All house rules were followed: no Python touched, no notebook regenerated, no repo-wide ruff run, `src/fomo/settings.py` never staged or modified (it was not modified in this worktree to begin with), the Troubleshooting section is byte-identical (verified via diff against the pre-Task-1 revision), and `fetch_jplsbdb_objects` does not appear anywhere in the runbook.

One environment-setup step outside plan scope: the gitignored setuptools_scm artifact `src/fomo/_version.py` was missing in this worktree and blocking Sphinx's AutoAPI import resolution with a benign warning; per critical execution note 10, it was copied from the main repo checkout (`/home/tlister/git/fomo_devel/src/fomo/_version.py`) and never committed (confirmed gitignored via `git check-ignore -v`).

## Issues Encountered
None - all three Sphinx verification gates (run after each task, and once more at the end) reported zero warning/error lines mentioning `runbooks/telescope_runs_calendar`. The build's other warnings (an autoapi docutils indentation issue in `fomo/urls/index.rst` and five toctree references to not-yet-existing pre_executed notebook pages) are pre-existing and out of this task's scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The runbook is now accurate for `load_telescope_runs`, the LCO sync's `[QUEUED]` behavior, and `backfill_lco_observation_records`.
- `DEFERRED.md` gives a ready-made backlog for a future quick task: campaign association on LCO sync, `import_campaign_csv` header tolerance, the remaining LCO title prefixes, and a possible `fetch_jplsbdb_objects` runbook page.
- CLAUDE.md's paired-deliverable rule will now automatically flag any future plan that touches `docs/runbooks/` without including the affected page in `files_modified`, without needing a filename-list update first.

---
*Phase: quick-260726-kdp*
*Completed: 2026-07-26*

## Self-Check: PASSED

All created/modified files confirmed present on disk (`docs/runbooks/telescope_runs_calendar.rst`,
`CLAUDE.md`, `DEFERRED.md`, `SUMMARY.md`). All three task commit hashes (`81dc276`, `fd19847`,
`e709bf4`) confirmed present in `git log --oneline --all`.
