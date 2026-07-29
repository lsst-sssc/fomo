---
phase: quick-260718-dih
plan: 01
subsystem: campaigns
tags: [django, calendar-sync, error-handling, regex, testing]

# Dependency graph
requires:
  - phase: 23
    provides: CampaignRunDecisionView._set_run_status (D-03/D-04/D-05 status-change endpoint)
  - phase: 14
    provides: telescope_runs.parse_run_line and load_telescope_runs._iter_run_nights
provides:
  - Non-reverting try/except around the calendar-sync loop in _set_run_status
  - Fail-fast ValueError for genuine cross-month run ranges in parse_run_line
  - fullmatch-anchored partial-night token validation in parse_run_line
  - Reworded _iter_run_nights docstring/message for its remaining descending-range guard
  - Corrected, resolution-annotated Findings.md
affects: [campaigns, telescope_runs, load_telescope_runs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Non-reverting try/except around a side-effect loop that follows an already-committed
       write (mirrors _resolve_site's existing projection guard) -- commit first, warn on
       side-effect failure, never revert the committed write."
    - "Fail-fast rejection at parse time instead of a fail-late parser/loader contract
       mismatch (parser and loader must agree on what's unsupported, and the rejection
       should happen as early as possible)."
    - "fullmatch over search for single-token regex validation to reject garbage-wrapped
       input that would otherwise substring-match."

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/telescope_runs.py
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - .planning/Findings.md

key-decisions:
  - "Removed the dead December-to-January year-rollover block in parse_run_line entirely
     (replaced with a plain `year = date_cls.today().year`) rather than keeping it as
     unreachable code, since the only case it ever served (a genuine cross-month range) now
     fails fast before reaching it; any remaining descending same-month range is always
     rejected downstream by _iter_run_nights, so the rollover's effect is never observable."
  - "Reworded _iter_run_nights's day2 < day1 guard's docstring/message to describe its real
     remaining purpose (a descending or malformed same-month day range, e.g. a typo like
     '20-5 July') instead of 'cross-month ranges not yet supported in Phase 3' -- genuine
     cross-month ranges are now rejected upstream in parse_run_line, so the guard's framing
     needed to change even though the guard code itself (`if parsed.day2 < parsed.day1`)
     stays load-bearing and unchanged."
  - "Findings.md finding #2 was softened rather than marked simply 'fixed': the cross-month
     rejection was always a documented, intentional Phase-3 deferral, not an accidental
     contract mismatch -- this task changed it from fail-late (parse succeeds, ingest
     rejects) to fail-fast (parse rejects immediately), resolving the original framing."
  - "Scoped ruff check/format invocations to the 6 files this task actually touched (plus
     --exclude on whole-repo passes) instead of running unscoped `ruff check . --fix` /
     `ruff format .`, after discovering they touch unrelated pre-existing drift in
     migrations, src/fomo/settings.py, two unrelated .planning/quick scripts, and would
     reorder imports in two paired demo notebooks under docs/notebooks/pre_executed/ --
     the latter is explicitly prohibited by both CLAUDE.md's paired-notebook convention and
     this plan's scope note. All out-of-scope drift reverted and logged to
     deferred-items.md, not fixed."

patterns-established:
  - "Side-effect loops that follow an already-committed write get their own non-reverting
     try/except (not shared with revert-capable guards elsewhere in the same view), with
     wording that makes clear the committed change survived and only the side effect needs
     a retry."

requirements-completed: [PR-REVIEW-F1, PR-REVIEW-F2, PR-REVIEW-F3]

coverage:
  - id: D1
    description: "Marking an APPROVED run cancelled/weathered no longer 500s when the
      calendar-sync loop raises -- run_status is committed, a warning is shown, and the
      idempotent action can be retried."
    requirement: "PR-REVIEW-F1"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestRunStatusChange.test_mark_cancelled_survives_calendar_sync_failure"
        status: pass
    human_judgment: false
  - id: D2
    description: "parse_run_line rejects a genuine cross-month range (e.g. '28
      December-2 January') at parse time with ValueError, instead of returning a ParsedRun
      the loader always rejects; the dead year-rollover block was removed."
    requirement: "PR-REVIEW-F2"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_telescope_runs.py#TestTelescopeRuns.test_parse_run_line_cross_month_range_raises"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_load_telescope_runs.py#TestLoadTelescopeRuns.test_cross_month_line_logged_and_skipped"
        status: pass
    human_judgment: false
  - id: D3
    description: "A partial-night token with surrounding garbage (e.g. 'xBoN-0626') is
      rejected via fullmatch instead of silently substring-matched."
    requirement: "PR-REVIEW-F3"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_telescope_runs.py#TestTelescopeRuns.test_parse_run_line_partial_night_token_with_garbage_prefix_raises"
        status: pass
    human_judgment: false
  - id: D4
    description: "Findings.md records each original finding plus a plain-English resolution
      note with corrected post-fix line numbers."
    verification: []
    human_judgment: true
    rationale: "Documentation accuracy/tone is a judgment call best confirmed by a human
      reading the file, not something a test can assert."

# Metrics
duration: 11min
completed: 2026-07-18
status: complete
---

# Quick Task 260718-dih: Fix PR Review Findings Summary

**Hardened three PR-review findings on the telescope-runs-calendar branch: a non-reverting try/except around the run-status-change calendar sync, fail-fast ValueError for cross-month run ranges (replacing a fail-late parser/loader mismatch), and fullmatch-anchored partial-night token validation.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-07-18T09:00:03Z
- **Completed:** 2026-07-18T09:11:17Z
- **Tasks:** 3
- **Files modified:** 7 (6 code/test files + Findings.md)

## Accomplishments
- `_set_run_status`'s calendar-sync loop is wrapped in a non-reverting try/except (mirrors `_resolve_site`'s existing projection guard): a sync failure now logs via `logger.exception`, warns the user their status change was saved and only the calendar sync needs retrying, and redirects -- never an uncaught 500.
- `parse_run_line` now raises `ValueError` immediately when it detects a genuine cross-month range, instead of building a `ParsedRun` that `_iter_run_nights` always rejected downstream. The now-dead December-to-January year-rollover block was removed.
- The partial-night token match switched from `.search(...)` to `.fullmatch(...)`, so a garbage-wrapped token like `'xBoN-0626'` is rejected instead of substring-matched into a plausible-but-wrong window; well-formed tokens (`'BoN-0626'`, `'0646-EoN'`) still parse identically.
- `_iter_run_nights`'s `day2 < day1` guard is kept (still load-bearing for a descending same-month range like a typo'd `'20-5 July'`) with its docstring/message reworded to reflect that narrower remaining case.
- `.planning/Findings.md` retains all three original findings and the coverage-gap list as a historical record, with a plain-English "Resolved:" note and corrected post-fix line numbers appended to each.
- All three coverage gaps closed with new tests: a calendar-sync-failure test in `test_campaign_approval.py`, a rewritten cross-month test plus a new garbage-token test in `test_telescope_runs.py`, and a new command-level cross-month-skip test in `test_load_telescope_runs.py`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Guard the calendar-sync loop in _set_run_status + Finding-1 coverage test** - `f447994` (fix)
2. **Task 2: Fail-fast cross-month rejection + anchored partial-night token + parser/loader tests** - `baac8c2` (fix)
3. **Task 3: Correct Findings.md line numbers + resolution notes; final ruff + full affected-suite run** - `01dbc2a` (docs)

_No plan-metadata commit yet -- SUMMARY.md and STATE.md updates are handled by the orchestrator per this dispatch's constraints._

## Files Created/Modified
- `solsys_code/campaign_views.py` - `_set_run_status`'s calendar-sync loop wrapped in a non-reverting try/except; docstring updated.
- `solsys_code/telescope_runs.py` - `parse_run_line` fails fast on a genuine cross-month range; dead year-rollover block removed; `_PARTIAL_NIGHTS` matched via `fullmatch`.
- `solsys_code/management/commands/load_telescope_runs.py` - `_iter_run_nights`'s day2 < day1 guard docstring/message reworded for its narrower remaining purpose.
- `solsys_code/tests/test_campaign_approval.py` - new `test_mark_cancelled_survives_calendar_sync_failure`.
- `solsys_code/tests/test_telescope_runs.py` - rewrote the December/January rollover test into a cross-month-raises test; added a garbage-prefix partial-night-token test.
- `solsys_code/tests/test_load_telescope_runs.py` - new `test_cross_month_line_logged_and_skipped`.
- `.planning/Findings.md` - corrected post-fix line numbers; appended plain-English resolution notes to each finding and coverage-gap item.
- `.planning/quick/260718-dih-fix-pr-review-findings-unguarded-calenda/deferred-items.md` - logs pre-existing, out-of-scope ruff drift discovered (and reverted) while running whole-repo quality gates.

## Decisions Made
See `key-decisions` in frontmatter above (year-rollover removal rationale, `_iter_run_nights` message rewording, Findings.md finding #2 softened framing, and scoped ruff invocations).

## Deviations from Plan

None beyond the ruff-scoping adjustment already captured as a key decision above (not a
Rule 1-4 deviation in the code being shipped -- it is a verification-tooling adjustment to
avoid touching unrelated files while still satisfying the plan's quality-gate verification
step). No architectural changes, no new dependencies, no auth gates encountered.

## Issues Encountered
- Running `ruff check . --fix` and `ruff format .` unscoped touched unrelated pre-existing
  drift (7 files) including two paired demo notebooks under `docs/notebooks/pre_executed/`
  that CLAUDE.md and this plan's scope note explicitly prohibit modifying. Reverted those
  changes and re-ran ruff scoped to the 6 files this task actually touched (clean, no
  changes needed) plus whole-repo passes with `--exclude "docs/notebooks/pre_executed/*.ipynb"
  --force-exclude` to confirm no in-scope regressions. Logged the out-of-scope drift to
  `deferred-items.md` per the deviation-rules Scope Boundary rather than fixing it.

## Known Stubs
None.

## Threat Flags
None -- all three fixes are pure hardening of already-shipped surfaces per the plan's threat
model (T-DIH-01/02/03), no new trust boundaries or surfaces introduced.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All three PR-review findings are resolved and covered by new regression tests; the 162-test
  combined run across `test_campaign_approval`, `test_telescope_runs`, and
  `test_load_telescope_runs` passes.
- `ruff check` and `ruff format --check` are clean on every file this task touched.
- `.planning/Findings.md` is up to date and safe to reference from future planning docs.
- Pre-existing, unrelated ruff drift in migrations/settings.py/two `.planning/quick` scripts
  remains (see `deferred-items.md`) -- not blocking, out of scope for this task.

## Self-Check: PASSED

All 9 created/modified files verified present on disk; all 3 task commit hashes
(`f447994`, `baac8c2`, `01dbc2a`) verified present in git log.

---
*Quick task: 260718-dih*
*Completed: 2026-07-18*
