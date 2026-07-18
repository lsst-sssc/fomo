---
phase: 24-operator-and-usage-runbook-documentation-for-the-telescope-r
plan: 01
subsystem: docs
tags: [sphinx, rst, runbook, operator-docs, telescope-runs-calendar]

# Dependency graph
requires:
  - phase: 25-range-window-calendarevent-projection-allow-approved-site-re
    provides: backfill_range_calendar_events command and its real observed Observatory-timezone failure text
provides:
  - "docs/runbooks/telescope_runs_calendar.rst — consolidated operator runbook"
  - "docs/installation.rst — Django/manage.py onboarding subsection"
  - "docs/index.rst — Runbooks toctree entry, discoverable"
affects: [complete-milestone-v2.1]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Sphinx runbook page under docs/runbooks/, cross-referenced from docs/design/ (why) and docs/installation.rst (onboarding), never duplicating either"]

key-files:
  created:
    - docs/runbooks/telescope_runs_calendar.rst
  modified:
    - docs/installation.rst
    - docs/index.rst

key-decisions:
  - "OPEN QUESTION 1 resolved: backfill_range_calendar_events INCLUDED in the runbook (same command family, real observed failure mode, low marginal cost)"
  - "OPEN QUESTION 2 resolved: Django/manage.py onboarding content APPENDED as a new subsection to docs/installation.rst (no new file, no second toctree entry)"
  - "mark_cancelled/mark_weather_failure staff actions folded into the calendar-sync grouping (D-07), not given a standalone section"

patterns-established:
  - "Task-oriented 'How do I...?' runbook structure (D-06) as the house style for future operator documentation, distinct from docs/design/'s rationale-only style"

requirements-completed: [D-01, D-02, D-03, D-04, D-05, D-06, D-07, D-08, D-09, D-10, D-11, D-12, D-13]

coverage:
  - id: D1
    description: "docs/installation.rst gains a 'Running FOMO Management Commands' subsection with a .. _running-management-commands: label, positioned after 'Starting up the webserver'"
    requirement: "D-08, D-09"
    verification:
      - kind: other
        ref: "grep -n '^.. _running-management-commands:' docs/installation.rst; grep -n '^Running FOMO Management Commands' docs/installation.rst"
        status: pass
    human_judgment: false
  - id: D2
    description: "docs/runbooks/telescope_runs_calendar.rst is a single consolidated page with six task-oriented 'How do I...?' subsections covering all five commands plus the folded mark_cancelled/mark_weather_failure staff actions, and a five-row cheat-sheet list-table"
    requirement: "D-01, D-02, D-03, D-05, D-06, D-07, D-10"
    verification:
      - kind: other
        ref: "grep -c 'How do I' docs/runbooks/telescope_runs_calendar.rst (returns 7); per-command/action grep loop (all present)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Runbook wired into docs/index.rst's :hidden: toctree (Runbooks <runbooks/telescope_runs_calendar>, between Design and API Reference); notebook-excluding Sphinx build passes with no orphan warning for the new page"
    requirement: "D-04"
    verification:
      - kind: other
        ref: "sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build (build succeeded, 9 pre-existing unrelated warnings, 0 orphan warnings for the runbook page)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Troubleshooting section documents the three real observed failure-mode families (Observatory-missing-timezone with verbatim error + IANA fix-it step; per-line/per-record skip-and-log invariant across all four ingest/sync commands; import_campaign_csv's site_needs_review/window_needs_review flags and re-import target-reset gotcha), with synthetic-PII-only examples"
    requirement: "D-11, D-12, D-13"
    verification:
      - kind: other
        ref: "grep for verbatim timezone error string, America/Santiago, site_needs_review; PII grep confirms no non-@example. email addresses in the file"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-07-18
status: complete
---

# Phase 24 Plan 01: Operator and Usage Runbook Documentation Summary

**Task-oriented Sphinx operator runbook (docs/runbooks/telescope_runs_calendar.rst) covering all five telescope-runs-calendar management commands plus the approval-queue mark_cancelled/mark_weather_failure staff actions, a five-command cheat-sheet, and a troubleshooting section built from real observed failure modes — wired into docs/index.rst's toctree and cross-referenced with a new Django-onboarding subsection in docs/installation.rst.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-07-18T07:15Z (approx.)
- **Completed:** 2026-07-18T07:22Z
- **Tasks:** 3 completed
- **Files modified:** 3 (2 modified, 1 new)

## Accomplishments
- Appended a "Running FOMO Management Commands" onboarding subsection to `docs/installation.rst`, with a `.. _running-management-commands:` cross-reference label that the runbook links to (D-08/D-09).
- Created `docs/runbooks/telescope_runs_calendar.rst` — a single consolidated, task-oriented runbook page ("How do I...?" framing, D-05/D-06) covering all five in-scope commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, `import_campaign_csv`, `backfill_range_calendar_events`), with the `mark_cancelled`/`mark_weather_failure` staff actions folded into the calendar-sync grouping (D-07) and a five-row "Command cheat-sheet" list-table (D-10).
- Wired the new page into `docs/index.rst`'s `:hidden:` toctree (between Design and API Reference) so it is discoverable, not orphaned (D-04) — confirmed via the notebook-excluding Sphinx build with no orphan warning for the new page.
- Added a Troubleshooting section documenting the three real, already-observed failure-mode families: the Observatory-missing-timezone gap (verbatim error string `Observatory 'FTN' (obscode=F65) has no timezone set` + `America/Santiago` IANA fix-it step, D-12), the shared per-line/per-record skip-and-log invariant across `load_telescope_runs`/`sync_lco_observation_calendar`/`backfill_range_calendar_events` (one bad row never aborts the run, D-13), and `import_campaign_csv`'s `site_needs_review`/`window_needs_review` unresolved-row flags plus its re-import `target`-reset gotcha (D-11/D-13) — every example uses synthetic placeholder values, confirmed by a PII grep finding no non-`@example.` email addresses anywhere in the file (security threat T-24-01).

## Task Commits

Each task was committed atomically:

1. **Task 1: Append the "Running FOMO Management Commands" onboarding subsection to docs/installation.rst** - `e220f21` (docs)
2. **Task 2: Create the consolidated runbook page and wire it into the index toctree** - `746440b` (docs)
3. **Task 3: Add the troubleshooting section and run the phase build gate** - `6bd0482` (docs)

_No TDD tasks — documentation-only phase._

## Files Created/Modified
- `docs/installation.rst` - Appended "Running FOMO Management Commands" subsection with a `.. _running-management-commands:` label
- `docs/runbooks/telescope_runs_calendar.rst` (NEW) - Consolidated operator runbook: six "How do I...?" subsections, a command cheat-sheet, and a Troubleshooting section
- `docs/index.rst` - Added one new toctree line: `Runbooks <runbooks/telescope_runs_calendar>`

## Decisions Made
- Both RESEARCH.md open questions resolved exactly as the plan specified: `backfill_range_calendar_events` INCLUDED (same command family, real observed failure mode already documented from Phase 25); Django-onboarding content APPENDED to `docs/installation.rst` (no new file, no second toctree/orphan risk).
- Added a `.. _command-cheat-sheet:` label above the cheat-sheet section so the Troubleshooting section's "See also" pointer could use a resolvable `:ref:` rather than a dangling reference — a small addition beyond the plan's literal text, needed to keep the Sphinx build warning-free (not a scope change, just an internal cross-reference).

## Deviations from Plan

None — plan executed exactly as written, with one minor addition: a `.. _command-cheat-sheet:` label (see Decisions above) so the Troubleshooting section's internal "See also" cross-reference resolves cleanly instead of dangling. No new dependency, no application-code change, no notebook change.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Documentation-only phase; the notebook-excluding Sphinx build (the authoritative pre-commit gate) passed on every task commit.

## Next Phase Readiness

This was Phase 24's only plan. With this phase complete, all four v2.1 roadmap phases (18-21) plus the two organic phases (23, 25) and this documentation phase (24) are done. Per STATE.md's "milestone v2.1 is NOT yet complete" note, Phase 25 declared a dependency on Phase 24 (now satisfied). `/gsd-complete-milestone` can now proceed for v2.1 — no known blockers.

---
*Phase: 24-operator-and-usage-runbook-documentation-for-the-telescope-r*
*Completed: 2026-07-18*
