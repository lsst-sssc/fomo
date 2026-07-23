---
phase: quick-260722-tkt
plan: 01
subsystem: solsys_code/management/commands
tags: [backfill, targets, management-command, lco]
dependency-graph:
  requires: []
  provides:
    - "--create-missing-targets flag on backfill_lco_observation_records"
    - "_request_target_info() helper (name/ra/dec) on requests"
  affects:
    - solsys_code/management/commands/backfill_lco_observation_records.py
tech-stack:
  added: []
  patterns:
    - "create-or-reuse Target by name, mirroring the campaign's find-or-create pattern used elsewhere in FOMO"
key-files:
  created: []
  modified:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py
decisions:
  - "Field Targets created by the flag are always type=SIDEREAL (fixed-sky pointings), not the campaign's NonSideralTargetFactory-fixtured moving-object target -- this is by design, not a CLAUDE.md convention violation."
  - "created_targets counter only reflects actually-persisted creations (0 in --dry-run), since dry-run never saves a Target; the per-request stdout line still reports the would-create/would-reuse intent."
metrics:
  duration: ~25min
  completed: 2026-07-22
status: complete
---

# Quick Task 260722-tkt: Add --create-missing-targets flag to backfill_lco_observation_records Summary

Added an opt-in `--create-missing-targets` flag to `backfill_lco_observation_records` that
auto-creates (or reuses, by name) a SIDEREAL field Target from a request's own RA/Dec when
the request's target isn't a campaign member, adds it to the campaign, and processes the
request through the normal ObservationRecord path instead of skipping it.

## What Was Built

**Task 1 — Extraction + flag + creation flow**
(`solsys_code/management/commands/backfill_lco_observation_records.py`):

- Added `RequestTargetInfo` dataclass and `_request_target_info(request)` helper that walks
  `request['configurations']` and returns `name`/`ra`/`dec` from the first configuration
  with a named target. `_request_target_name` now delegates to it, unchanged for existing
  callers.
- Added `--create-missing-targets` (`store_true`, default off) to `add_arguments`, with help
  text documenting the create-or-reuse behavior and its `--dry-run` interaction.
- In `handle()`, when a request's target name has no campaign match:
  - Flag off: unchanged skip-and-log + `skipped_unmatched_target` increment.
  - Flag on: `_resolve_or_build_field_target()` looks up an existing `Target` by that exact
    name anywhere in FOMO (reuse) or builds a new unsaved `Target(type=Target.SIDEREAL,
    ra=..., dec=...)` (create). In `--dry-run`, nothing is saved/added -- only a descriptive
    stdout line is written, and the (possibly unsaved) Target drives the existing "Would
    create ObservationRecord" line. Otherwise the Target is saved (if new), added to the
    campaign via `campaign.targets.add(target)` (membership only), and inserted into
    `targets_by_name` so the request falls through to the normal `_build_parameters` /
    `ObservationRecord` path.
- Added a `created_targets` counter surfaced in the summary line.
- Updated the `Command` class docstring to document the flag and its `--dry-run` interaction
  using plain English ("create the Target if missing, otherwise reuse it") per CLAUDE.md
  terminology guidance.

**Task 2 — Tests** (`solsys_code/tests/test_backfill_lco_observation_records.py`):

- Extended `_configuration`/`_request` with optional `target_type`/`ra`/`dec` params, and
  added a `_field_request()` builder (type `ICRS`, default ra=170.1/dec=-24.3) for
  field-target requests.
- Added four tests covering all flag scenarios:
  - `test_flag_off_unmatched_field_target_still_skipped` -- default behavior unchanged.
  - `test_flag_on_creates_new_field_target` -- new SIDEREAL Target created with the
    request's ra/dec, added to campaign, ObservationRecord created.
  - `test_flag_on_reuses_existing_field_target` -- pre-created Target (not yet a campaign
    member) is reused (count stays 1), campaign membership added, ObservationRecord created.
  - `test_flag_on_dry_run_creates_nothing` -- zero Target rows, zero membership changes, zero
    ObservationRecords.
- All 13 tests in the suite pass (`./manage.py test
  solsys_code.tests.test_backfill_lco_observation_records`).

**Task 3 — Lint/format**: touched files pass `ruff check` and `ruff format --check`
individually. See Deviations for the repo-wide finding.

## Deviations from Plan

None on the two files this task modified -- plan executed as written.

### Out-of-scope (not fixed, logged only)

Repo-wide `ruff check .` / `ruff format .` surfaced 5 pre-existing findings in
`docs/notebooks/pre_executed/*.ipynb` files (E401/I001 import-organization issues,
one missing docstring, one long line) that are unrelated to this task's changes and were
not modified by it (confirmed via `git status`/`git log`, traces to commit `0dcd4b0`,
Phase 23). Per the executor's scope boundary rule, these were not touched; logged to
`.planning/quick/260722-tkt-add-create-missing-targets-flag-to-backf/deferred-items.md`.

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/backfill_lco_observation_records.py
- FOUND: solsys_code/tests/test_backfill_lco_observation_records.py
- FOUND commit a87dd1a (feat: add --create-missing-targets flag)
- FOUND commit 73581b0 (test: add tests for flag scenarios)
