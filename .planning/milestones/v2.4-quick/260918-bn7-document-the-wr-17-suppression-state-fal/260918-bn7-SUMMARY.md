---
phase: quick-260918-bn7
plan: 01
subsystem: docs
tags: [runbook, unattended, cron, sphinx, rst, wr-16, wr-17]

# Dependency graph
requires:
  - phase: 36-unattended-code-review-fixes
    provides: "WR-16 (eafda16, cron_line() exit-code normalization) and WR-17 (72115bb, save_state()/load_state() fallback) already merged to issue37-telescope-runs-calendar"
provides:
  - "docs/runbooks/telescope_runs_calendar.rst updated to describe the cron line's post-WR-16 exit-status contract"
  - "docs/runbooks/telescope_runs_calendar.rst updated to describe the WR-17 suppression-state fallback file, its read rule, and operator handling"
affects: [36-unattended-code-review-fixes, any future phase touching solsys_code/unattended.py or check_unattended.py]

# Actuals (#2632)
actuals:
  tokens: 1693
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Paired-docs obligation for solsys_code/unattended.py, notifications.py, run_unattended.py and check_unattended.py is satisfied via the runbook's 'How do I run everything unattended?' section, not a demo notebook (per CLAUDE.md's explicit carve-out for this module group)."

key-files:
  created: []
  modified:
    - "docs/runbooks/telescope_runs_calendar.rst"

key-decisions:
  - "Documented WR-16's exit-code normalization in both the troubleshooting entry and the 'When nothing has appeared' checklist item, preserving the true statement that flock's own -E 99 is still the skip-detection mechanism."
  - "Documented WR-17's fallback file in three places (setup step 1, the check_unattended preflight paragraph, and a new troubleshooting entry) rather than one, matching the plan's key_links between setup prose and troubleshooting."
  - "Described the fallback path as 'the system temp directory (typically /tmp)' rather than a guaranteed /tmp literal, since the code uses tempfile.gettempdir()."

requirements-completed: [WR-16, WR-17]

coverage:
  - id: D1
    description: "Runbook's 'Repeated lock held lines' entry and 'When nothing has appeared' item 4 correctly describe the WR-16 exit-code normalization (rc=0, cron line status is always 0/1, never 99)"
    requirement: "WR-16"
    verification:
      - kind: other
        ref: "grep -qF literal check for 'rc=0', 'back to exit 0', 'WR-16, 36-REVIEW.md', '-E 99' in docs/runbooks/telescope_runs_calendar.rst"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files"
        status: pass
    human_judgment: false
  - id: D2
    description: "Runbook documents the WR-17 suppression-state fallback file (name, location, when written, newest-file read rule, self-deleting cleanup) in setup step 1, the check_unattended preflight paragraph, and a new troubleshooting entry"
    requirement: "WR-17"
    verification:
      - kind: other
        ref: "grep -qF literal check for 'fomo-unattended-state.fallback.json' (>=2 occurrences), 'WR-17, 36-REVIEW.md', 'FOMO_STATE_DIR' in docs/runbooks/telescope_runs_calendar.rst"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files"
        status: pass
    human_judgment: false

# Metrics
duration: 25min
completed: 2026-09-18
status: complete
---

# Quick Task 260918-bn7: Document the WR-16/WR-17 suppression-state fallback and exit-code normalization

**Corrected the unattended-operation runbook's stale exit-code claim (WR-16) and added the previously-undocumented suppression-state fallback file behavior (WR-17), closing a CLAUDE.md paired-docs breach.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-18T15:10:00Z
- **Completed:** 2026-09-18T15:35:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Corrected the "Repeated lock held lines" troubleshooting entry and "When nothing has appeared" checklist item 4 to state that a lock-held skip now normalizes the cron line's own exit status back to 0 (WR-16), while preserving the still-true statement that flock's own dedicated code 99 is the skip-detection mechanism.
- Documented the WR-17 suppression-state fallback (`fomo-unattended-state.fallback.json` in the system temp directory) in setup step 1, reframed the `check_unattended` preflight paragraph as setup-time-only with the runtime fallback as the other half of coverage, and added a new troubleshooting entry with an explicit do-not-delete-the-fallback warning.

## Task Commits

Each task was committed atomically:

1. **Task 1: Correct the two places the runbook describes a lock-held skip's exit status (WR-16)** - `74b53b4` (docs)
2. **Task 2: Document the suppression-state fallback file and its operator handling (WR-17)** - `6ded6b4` (docs)

_Note: the final metadata commit (SUMMARY.md/STATE.md) is made separately by the orchestrator, not this executor._

## Files Created/Modified
- `docs/runbooks/telescope_runs_calendar.rst` - WR-16 exit-code corrections (troubleshooting entry + checklist item 4) and WR-17 fallback documentation (setup step 1, preflight paragraph, new troubleshooting entry)

## Decisions Made
- None beyond what's captured in `key-decisions` above - followed the plan's anchor-based editing instructions exactly.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' literal gate checks, diff-discipline checks, and `pre-commit run sphinx-build --all-files` passed on the first attempt for each task.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required. This is a prose-only documentation change.

## Next Phase Readiness
- The CLAUDE.md paired-docs obligation for `solsys_code/unattended.py` and `check_unattended.py` is now satisfied for both WR-16 and WR-17; no notebook was added, matching CLAUDE.md's explicit carve-out for this module group.
- `docs/runbooks/telescope_runs_calendar.rst` is the only file in the diff across both commits; no code, test, notebook, or `deploy/cron/` file was touched.

---
*Phase: quick-260918-bn7*
*Completed: 2026-09-18*

## Self-Check: PASSED

- FOUND: `docs/runbooks/telescope_runs_calendar.rst`
- FOUND commit: `74b53b4`
- FOUND commit: `6ded6b4`
