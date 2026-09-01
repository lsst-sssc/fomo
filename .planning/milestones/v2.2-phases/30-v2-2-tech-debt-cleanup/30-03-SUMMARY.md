---
phase: 30-v2-2-tech-debt-cleanup
plan: 03
subsystem: campaign-csv-import
tags: [import_campaign_csv, telescope_class, re-import-guard, runbook, D-04]

# Dependency graph
requires:
  - phase: 30-v2-2-tech-debt-cleanup
    provides: "30-01's attribution eligibility fix (unrelated file scope, no overlap)"
provides:
  - "import_campaign_csv's telescope_class re-import guard mirrors preserve_site: a stored non-blank value is never blanked and never replaced by a different derived value"
  - "telescope_class_preserved counter and per-row stderr diagnostic, so a preserved correction is never silently indistinguishable from 'unchanged'"
  - "docs/runbooks/telescope_runs_calendar.rst's re-import gotcha note and its executable form in test_import_campaign_csv.py describe the same widened behaviour"
affects: [30-04]

# Actuals (#2632)
actuals:
  tokens: 4300
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "The telescope_class guard mirrors preserve_site's exact shape: a decision boolean computed beside its inputs, a unit pop, a counter and a stderr 'say so' line -- extended to a genuine-difference comparison rather than a blanking-only check"
    - "A widened re-import guard is proven a strict superset of the guard it replaces (freshly-derived '' always differs from a non-blank stored value) rather than by re-deriving the invariant from scratch"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/import_campaign_csv.py
    - solsys_code/tests/test_import_campaign_csv.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "D-04: mirrored preserve_site's shape exactly for the telescope_class guard -- preserve_telescope_class computed immediately after the derivation it gates on (unlike preserve_site, which feeds its derivation), a unit pop, a telescope_class_preserved counter, and a stderr line following the same 'Row {row_num}: kept existing ...; CSV ... discarded' phrasing"
  - "The widened condition (existing.telescope_class truthy AND telescope_class != existing.telescope_class) is a strict superset of the old blanking-only condition (existing.telescope_class truthy AND not telescope_class) -- a freshly-derived '' always differs from any non-blank stored value, so the standing 'never cleared once set' invariant is preserved, not weakened"
  - "Placed the four new tests in a sibling TestReImportTelescopeClassPreservation class rather than appending to TestReImportSitePreservation, keeping the fixture-heavy site-guard tests separate from the telescope_class-only fixtures while both classes still name the runbook-drift contract in their docstrings"
  - "Left test_telescope_class_never_blanked_by_reimport completely unedited, per the plan's explicit instruction -- it passed unchanged, confirming the widened guard is a true superset"

requirements-completed: [D-04]

coverage:
  - id: T1
    description: "A re-import whose derived telescope_class differs from a non-blank stored value keeps the stored value, reports it on stderr, and counts it as telescope_class_preserved"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_import_campaign_csv -v2 -- 77 tests OK, including test_reimport_does_not_replace_a_corrected_telescope_class_with_a_different_derived_one and test_a_preserved_telescope_class_is_reported_on_stderr_and_in_the_summary"
        status: pass
    human_judgment: false
  - id: T2
    description: "First-time derivation onto a blank stored value is unaffected (non-vacuous control); an identical re-derivation is not reported as preserved"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "test_first_derivation_still_writes_telescope_class_when_the_existing_value_is_blank and test_an_identical_derived_telescope_class_is_not_reported_as_preserved, both pass"
        status: pass
    human_judgment: false
  - id: T3
    description: "The pre-existing never-blanked invariant test survives unedited, and the guard's blast radius does not reach the site-repair command or model invariants"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "git diff -U0 solsys_code/tests/test_import_campaign_csv.py shows 0 removed lines; python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_repair_stale_campaign_run_sites solsys_code.tests.test_campaign_models -v1 -- 98 tests OK"
        status: pass
    human_judgment: false
  - id: T4
    description: "The runbook's re-import gotcha note states the widened rule (never blanked AND never replaced), names telescope_class_preserved, and Sphinx builds clean with plan 30-01's attribution paragraph and the attribution section (lines 187-215) untouched"
    requirement: "D-04"
    verification:
      - kind: other
        ref: "grep -c 'telescope_class_preserved' docs/runbooks/telescope_runs_calendar.rst == 2; grep -c 'never replaced' (excluding directives) == 1; grep -c 'A rejected run is never offered as a match' == 1; sed -n '187,215p' unchanged by this task's diff hunks (@@ -433 and @@ -789 only); python -m sphinx -b html docs docs/_build/html -q exits 0 with no warning naming this file"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-08-31
status: complete
---

# Phase 30 Plan 03: Widen the telescope_class Re-Import Guard Summary

**Closed 27-REVIEW WR-01's remaining half by mirroring `preserve_site`'s exact shape for `telescope_class`: a re-import can no longer silently replace a hand-corrected class with a different derived one, every such firing is named on stderr and counted as `telescope_class_preserved` in the summary, and the runbook's re-import gotcha note now says so too.**

## Performance

- **Duration:** ~10 min (commit-to-commit)
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Widened `import_campaign_csv`'s `telescope_class` guard from "never blanked"
  to "never blanked AND never replaced by a different derived value" (D-04),
  computing `preserve_telescope_class` immediately after the derivation it
  gates on and documenting it as a strict superset of the guard it replaces
- Added the `telescope_class_preserved_count` counter, its `self.stderr.write`
  diagnostic (`Row {row_num}: kept existing telescope_class ...; CSV
  telescope_class discarded`), and its `telescope_class_preserved:` field on
  the command's stdout summary line, mirroring `preserve_site`'s "say so"
  reporting exactly
- Extended the `help` string to state the widened rule and both preserved-row
  summary fields
- Pinned both directions of the widened guard with four new tests in a
  sibling `TestReImportTelescopeClassPreservation` class: the preserve case,
  its reporting, the non-vacuous first-derivation control, and the
  no-genuine-difference no-noise case -- while leaving
  `test_telescope_class_never_blanked_by_reimport` completely unedited
- Brought `docs/runbooks/telescope_runs_calendar.rst`'s re-import gotcha note
  and its later site-repair cross-reference back in step with the widened
  code, naming the new stderr diagnostic and `telescope_class_preserved:`
  summary field verbatim

## Task Commits

Each task was committed atomically:

1. **Task 1: Widen the telescope_class guard to mirror preserve_site, with its counter and stderr line** - `9dda336` (feat)
2. **Task 2: Pin both directions of the widened guard with tests** - `3ad8b64` (test)
3. **Task 3: Bring the runbook's re-import gotcha back in step with the widened guard** - `3e0e727` (docs)

## Files Created/Modified

- `solsys_code/management/commands/import_campaign_csv.py` - `preserve_telescope_class` decision boolean computed beside the `telescope_class` derivation; the blanking-only guard block replaced with a widened block that pops the field, increments `telescope_class_preserved_count`, and writes the stderr diagnostic; `help` string and the stdout summary line extended
- `solsys_code/tests/test_import_campaign_csv.py` - new `TestReImportTelescopeClassPreservation` class with 4 tests; `test_telescope_class_never_blanked_by_reimport` left byte-identical
- `docs/runbooks/telescope_runs_calendar.rst` - re-import gotcha note's `telescope_class` paragraph restated to cover both directions and name the new summary field/stderr diagnostic; the site-repair cross-reference extended to state the guard's general case

## Decisions Made

- Mirrored `preserve_site`'s shape exactly per D-04, but computed
  `preserve_telescope_class` immediately AFTER the `telescope_class`
  derivation rather than before it (unlike `preserve_site`, which the
  derivation itself depends on) -- the plan called this out explicitly and
  the ordering is load-bearing: the decision needs the row's actual derived
  value to compare against.
- Verified the widened condition's strict-superset property directly rather
  than asserting it: a freshly-derived `''` always differs from any non-blank
  stored value (`'' != <non-blank>` is always `True`), so every case the old
  blanking-only guard caught (`existing.telescope_class and not
  telescope_class`) is still caught by the new one
  (`existing.telescope_class and telescope_class != existing.telescope_class`).
- Placed the four new tests in a sibling class,
  `TestReImportTelescopeClassPreservation`, rather than appending to the
  already-large `TestReImportSitePreservation` -- CONTEXT.md/PATTERNS.md
  explicitly allowed either, and the telescope_class-only fixtures (no
  `Observatory` seeding needed) read more cleanly on their own.
- Extended the later "How do I re-resolve campaign run sites..." cross-
  reference (not just the primary gotcha note) because it specifically named
  `telescope_class` surviving a re-import in the site-preserved case only --
  leaving the guard's more general, site-independent case (a row whose site
  was never resolved at all, but whose stored class differs from a fresh
  derivation) unstated there.

## Deviations from Plan

None - plan executed exactly as written. No auto-fixes, no blocking issues,
no architectural questions arose.

## Issues Encountered

None. The read-first references in the plan (line numbers for `preserve_site`,
the `help` string, `site_preserved_count`'s initialisation) all matched the
file on disk exactly, so no exploratory reading beyond what the plan specified
was needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- D-04 (27-REVIEW WR-01's `telescope_class` half) is complete and
  independently verified: 77/77 tests in the CSV-import module pass, the
  98-test blast-radius check (import + repair command + campaign models)
  passes, `pre-commit run ruff --all-files` passes with zero files modified,
  and `python -m sphinx -b html docs docs/_build/html -q` exits 0 with no
  warning naming the edited runbook page.
- Plan 30-04 (bookkeeping/audit amendment) can record D-04 as closed; nothing
  in this plan's `files_modified` overlaps with 30-04's scope.
- No blockers.

## Self-Check: PASSED

- All 3 `files_modified` paths verified present on disk
- All 3 task commit hashes (`9dda336`, `3ad8b64`, `3e0e727`) verified in `git log`
- `grep -c 'preserve_telescope_class' solsys_code/management/commands/import_campaign_csv.py` = 4
- `grep -c 'telescope_class_preserved' solsys_code/management/commands/import_campaign_csv.py` = 4
- `grep -c 'kept existing telescope_class' solsys_code/management/commands/import_campaign_csv.py` = 1
- `grep -c 'CSV telescope_class discarded' solsys_code/management/commands/import_campaign_csv.py` = 1
- `grep -c 'D-04' solsys_code/management/commands/import_campaign_csv.py` = 5
- `python manage.py test solsys_code.tests.test_import_campaign_csv -v2` re-run: OK (77 tests)
- `python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_repair_stale_campaign_run_sites solsys_code.tests.test_campaign_models -v1` re-run: OK (98 tests)
- `grep -c 'telescope_class_preserved' docs/runbooks/telescope_runs_calendar.rst` = 2
- `grep -c 'A rejected run is never offered as a match' docs/runbooks/telescope_runs_calendar.rst` = 1 (plan 30-01's paragraph intact)
- `git show --stat` for all 3 commits lists exactly one of the three `files_modified` paths each
- `pre-commit run ruff --all-files` Passed, no files modified

---
*Phase: 30-v2-2-tech-debt-cleanup*
*Completed: 2026-08-31*
