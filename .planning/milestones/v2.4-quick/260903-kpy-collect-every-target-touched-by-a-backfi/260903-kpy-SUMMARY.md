---
phase: quick-260903-kpy
plan: 01
subsystem: backend
tags: [django, management-command, tom-toolkit, target-list, backfill]

# Dependency graph
requires:
  - phase: quick-260903-jid
    provides: backfill_lco_observations summary line contract (no explicit stdout write, single-printed summary)
provides:
  - "backfill_lco_observations collects every touched Target into a create-or-reuse TargetList"
affects: [backfill_lco_observations, telescope_runs_calendar runbook]

# Actuals (#2632)
actuals:
  tokens: 11000
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-invocation collection dict keyed by pk (real mode) or pk-or-name (dry-run mode) to make membership counting mode-symmetric without a second query"

key-files:
  created: []
  modified:
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "Collection keyed by pk in a real run, by pk-or-name in dry-run mode, so a matched target is never double-counted and an unsaved would-be-new target is still collected exactly once (D-01/D-02)"
  - "Two separate insertion points (real branch, dry-run branch) rather than one shared point before the dry_run check, so a real run's just-saved target is keyed by its new pk rather than falling back to name (D-03)"
  - "targets added to list: N reports the collected count, not the newly-added-membership count, so a dry run and the matching real pass over the same payload report the same N (D-04)"
  - "The list is created unconditionally in a real run (get_or_create runs before .add() regardless), even for a sweep that touches zero targets (D-05)"

requirements-completed: [TL-01, TL-02, TL-03, TL-04, TL-05, TL-06, TL-07]

coverage:
  - id: D1
    description: "A real sweep collects every Target it touches (matched and newly built) into a create-or-reuse TargetList named '<proposal>_targets'"
    requirement: "TL-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_created_with_matched_and_new_targets"
        status: pass
    human_judgment: false
  - id: D2
    description: "Re-running the same sweep is idempotent: one TargetList, no duplicate membership, reused verb in the summary"
    requirement: "TL-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_rerun_is_idempotent"
        status: pass
    human_judgment: false
  - id: D3
    description: "--target-list NAME overrides the derived name; the derived name is never created"
    requirement: "TL-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_override_name"
        status: pass
    human_judgment: false
  - id: D4
    description: "Skipped requests (target-step and parameters-step skips alike) contribute nothing to the list"
    requirement: "TL-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_excludes_targets_from_skipped_requests"
        status: pass
    human_judgment: false
  - id: D5
    description: "--dry-run performs zero TargetList writes while reporting would-forms and a would-add count matching the real pass"
    requirement: "TL-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_dry_run_zero_writes_matches_real_pass_count"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py#test_target_list_dry_run_would_reuse_after_real_run"
        status: pass
    human_judgment: false
  - id: D6
    description: "Summary carries 'target list: <verb> <name>' and the added/would-add count fields in both modes, appended after 'block lookups failed', with every pre-existing exact-line assertion extended"
    requirement: "TL-06"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_backfill_lco_observations.py (full module, 30 tests)"
        status: pass
    human_judgment: false
  - id: D7
    description: "Paired demo notebook re-executed with output showing both new fields and printing list membership; cleanup cell deletes the created list; runbook documents the collection, --target-list, both counters, and the accurate campaign-surface consequence"
    requirement: "TL-07"
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files"
        status: pass
    human_judgment: false

duration: ~65min
completed: 2026-09-03
status: complete
---

# Quick Task 260903-kpy Summary

**backfill_lco_observations now collects every touched Target (matched and newly built) into a create-or-reuse TargetList named `<proposal>_targets`, with `--target-list` override, mode-symmetric would-add/added counts, and zero writes under `--dry-run`**

## Performance

- **Duration:** ~65 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added a `collected_targets` dict to `handle()`, keyed by pk (real mode) or pk-or-name (dry-run mode, D-01/D-02), populated at two distinct insertion points downstream of every skip branch (D-03)
- Post-loop step creates-or-reuses a `TargetList` named `<proposal>_targets` (overridable via new `--target-list NAME`), adds the collected targets in one set-like `.add()` call, and reports both a create/reuse verb and a collected-count field (D-04, D-05, D-06) appended to the existing summary line
- Dry-run branch performs only a `TargetList.objects.filter(name=...).exists()` read — no `get_or_create`, no `.add()` — so `--dry-run` remains a true zero-write path (T-kpy-01)
- Six new tests covering derived-name creation, re-run idempotence, the `--target-list` override, dry-run zero-writes matching the real pass's would-add count, dry-run would-reuse after a real run, and skip exclusion (both a target-step skip and a parameters-step skip on an already-matched target); every one of the 24 pre-existing tests' expected summary lines extended via `_expected_summary()`'s three new keyword parameters
- Re-executed the paired demo notebook with real output showing both new summary fields in the dry-run and real passes, printing the resulting `TargetList`'s name and membership, and extending the cleanup cell to delete the demo list
- Extended the runbook's backfill section with the collection bullet, the `--target-list` paragraph, the dry-run would-form sentence, both updated sample summary lines, and the accurate campaign-surface consequence (appears in the campaign picker on the run submission form; does not appear on the campaign list page, per planning fact 4)

## Task Commits

Each task was committed atomically:

1. **Task 1: Collect touched targets into the TargetList, both modes, with tests** - `971b462` (feat)
2. **Task 2: Re-execute the paired demo notebook and update the runbook section** - `1082550` (docs)

## Files Created/Modified
- `solsys_code/management/commands/backfill_lco_observations.py` - `--target-list` arg, `collected_targets` dict, two collection insertion points, post-loop create-or-reuse TargetList step, two new summary fields, docstring updates
- `solsys_code/tests/test_backfill_lco_observations.py` - `_expected_summary()` extended with `list_name`/`list_reused`/`targets_added`; every existing call site updated with explicit values; 6 new tests (24 → 30 total)
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` - intro/dry-run markdown updated, `TargetList` import added, dry-run zero-write print check, list-membership inspection print, cleanup cell deletes the demo list; re-executed with real output
- `docs/runbooks/telescope_runs_calendar.rst` - "How it differs" bullet, `--target-list` paragraph, dry-run paragraph extension, both sample summary blocks, campaign-surface consequence sentence — confined to the backfill section (lines 128-215 origin range)

## Decisions Made
- Collection keyed by pk in real mode, by pk-or-name in dry-run mode (D-01/D-02) — the design decision that makes two portal names fuzzy-matching one existing `Target` collect once, not twice, in both modes
- Two separate insertion points rather than a shared one before the `if dry_run:` branch (D-03) — a real run's newly-saved target must be keyed by its fresh pk, not by name
- `targets added to list: N` is the collected count, not the newly-added-membership count (D-04) — preserves dry-run/real parity on repeated runs
- List created unconditionally in a real run even for a zero-target sweep (D-05) — `get_or_create` runs before `.add()` regardless, and an empty list has no campaign-list-page consequence (planning fact 4)
- Verb computed as a precomputed local rather than a nested ternary, per the plan's explicit style instruction

## Deviations from Plan

None - plan executed exactly as written. All locked design decisions (D-01 through D-07) implemented as specified; no architectural changes needed; no Rule 1-3 auto-fixes required.

## Issues Encountered

`pre-commit run ruff-format` reformatted the notebook's committed JSON on the Task 2 commit attempt (trailing-newline normalization inside three code-cell `source` arrays — a byte-level artifact of the JSON-manipulation script used to edit the notebook, not a content change). Re-staged and re-committed; no further issues.

## Next Phase Readiness

No blockers. The command, its test suite (30 tests), the paired notebook, and the runbook section are all in sync. `backfill_lco_observation_records.py`, its test module, and the campaign modules remain byte-for-byte unchanged (verified via `git status --porcelain` blast-radius check on both task commits).

---
*Phase: quick-260903-kpy*
*Completed: 2026-09-03*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/backfill_lco_observations.py
- FOUND: solsys_code/tests/test_backfill_lco_observations.py
- FOUND: docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND commit: 971b462
- FOUND commit: 1082550
- Re-ran plan-level verification: `python manage.py test solsys_code.tests.test_backfill_lco_observations` → 30 tests, OK
- Re-ran neighbouring regression: `python manage.py test solsys_code.tests.test_backfill_lco_observation_records solsys_code.tests.test_sync_lco_observation_calendar` → 58 tests, OK
- Re-ran `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`, `pre-commit run sphinx-build --all-files` → all clean
- Blast radius check (`backfill_lco_observation_records.py`, its test module, campaign modules) → empty diff
- `reqgroup_2682493.json` confirmed still untracked, appears in no commit
