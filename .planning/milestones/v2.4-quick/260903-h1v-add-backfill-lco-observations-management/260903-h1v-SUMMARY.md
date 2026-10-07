---
phase: quick-260903-h1v
plan: 01
subsystem: lco-ingest
tags: [management-command, lco-portal, non-sidereal-targets, observation-groups]
dependency-graph:
  requires: []
  provides: [backfill_lco_observations-command, non-sidereal-backfill-path]
  affects: [solsys_code-management-commands, docs-runbooks-telescope-runs-calendar]
tech-stack:
  added: []
  patterns:
    - "get_or_create keyed on (facility, observation_id), update-in-place on the existing branch"
    - "portal wire-key -> TOM field mapping inverted from OCSFacility._build_target_fields"
    - "client-side re-check of a server-side date filter, same quantity both ends"
key-files:
  created:
    - solsys_code/management/commands/backfill_lco_observations.py
    - solsys_code/tests/test_backfill_lco_observations.py
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - CLAUDE.md
decisions:
  - "D-A honored: backfill_lco_observation_records left byte-for-byte unchanged; new command is campaign-agnostic, update-in-place, non-sidereal-only"
  - "Scheduled times default to the D-B live get_observation_status() fallback in the primary demo/tests, since the RequestGroups listing payload is not known to embed per-request observation blocks (see Notes below)"
  - "created_after/created_before comparisons fail closed (exclude) when a RequestGroup's own 'created' field is missing/unparseable, matching D-C's anti-silent-backfill intent"
metrics:
  duration: ~55min
  completed: 2026-09-03
actuals:
  tokens: 18365
  tasks: 3
  commits: 3
status: complete
---

# Quick Task 260903-h1v: Add backfill_lco_observations management command Summary

Campaign-agnostic `backfill_lco_observations` management command that backfills
`ObservationRecord`s from the LCO Observation Portal, updates them in place on re-run,
builds only non-sidereal `Target`s from orbital elements, and links multi-request
`RequestGroup`s into reusable `ObservationGroup`s.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | backfill_lco_observations command, end to end | `8a9c797` | `solsys_code/management/commands/backfill_lco_observations.py` |
| 2 | Django TestCase suite over a fully mocked portal | `3f118f6` | `solsys_code/tests/test_backfill_lco_observations.py` |
| 3 | Paired demo notebook, runbook section, and CLAUDE.md map entry | `f874531` | `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`, `docs/runbooks/telescope_runs_calendar.rst`, `CLAUDE.md` |

## Files Created

- `solsys_code/management/commands/backfill_lco_observations.py` — the command itself:
  `--proposal` (required), `--created-after`/`--created-before` (ISO-8601, client-side
  re-checked per D-C), `--username`, `--dry-run`. Pages `GET /api/requestgroups/` via the
  library's own `make_request`/`_portal_headers()`; matches targets with
  `Target.matches.match_fuzzy_name()` (name or alias); builds an unmatched target as
  non-sidereal from the request's `ORBITAL_ELEMENTS` payload (inverse of
  `OCSFacility._build_target_fields`'s wire-key mapping, D-E); writes/updates
  `ObservationRecord` via `get_or_create` + in-place field diff; links multi-request
  `RequestGroup`s into a `get_or_create`d `ObservationGroup` (D-D, `<=50`-char deterministic
  name); resolves `scheduled_start`/`scheduled_end` from an embedded `observations` block
  when present, else `LCOFacility.get_observation_status()` (D-B), best-effort and never
  fatal.
- `solsys_code/tests/test_backfill_lco_observations.py` — 16 `TestCase` tests, no live
  network (`make_request` and `LCOFacility.get_observation_status` both mocked). Covers
  creation, the embedded-block path independently of the fallback path, idempotent
  re-run (status/schedule update in place), third-pass no-churn (`unchanged: 1`), non-sidereal
  target creation with element correctness, existing-target reuse by exact name and by alias,
  unmappable-request skip-and-stderr-report (plus a `Target.SIDEREAL` count-zero assertion),
  multi- vs single-request `ObservationGroup` behavior, dry-run no-write, the
  created-after/created-before client-side re-check, and pagination.
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` — pre-executed,
  committed with output. Walks a hand-built two-request `RequestGroup` fixture through
  `--dry-run` (nothing written), a real pass (records + non-sidereal target + group
  created), and a second real pass with an advanced state and a narrowed observed block
  (one record updated in place, the other's `modified` timestamp provably untouched). Ends
  with a cleanup cell so the notebook is safely re-runnable.

## Files Modified

- `docs/runbooks/telescope_runs_calendar.rst` — new "How do I backfill ObservationRecords
  without a campaign?" section, placed directly after the existing
  `backfill_lco_observation_records` section, spelling out every contract difference
  (no campaign/name-prefix, update-in-place, always-non-sidereal, date filter instead of
  name prefix, `ObservationGroup` linking) so an operator picks the right command.
- `CLAUDE.md` — one new entry in the notebook-pairing map:
  `solsys_code/management/commands/backfill_lco_observations.py -> backfill_lco_observations_demo.ipynb`.

## Decisions Made

- Followed D-A exactly: `backfill_lco_observation_records.py` and its test module are
  untouched (`git status --porcelain` on both stayed empty through every task); the new
  module carries a docstring line pointing at the sibling and naming the contract
  difference.
- Followed D-E's wire-key mapping and D-D's group-naming/truncation rule verbatim, both
  cross-checked directly against `tom_observations/facilities/ocs.py` at execution time
  (`_build_target_fields`'s `field_mapping`, `ObservationGroup.name`'s `max_length=50`).
- `created_after`/`created_before` client-side comparisons **fail closed**: if a window is
  set and a RequestGroup's own `created` field is missing or unparseable, the group is
  excluded rather than included — consistent with D-C's "never silently backfill the whole
  proposal" intent, though the plan didn't specify this edge case explicitly.
- `_parse_datetime_value` accepts bare ISO-8601 dates (not just full timestamps) as a
  minor usability addition for `--created-after`/`--created-before`, beyond what D-C
  strictly required.

## Deviations from Plan

None — plan executed exactly as written. One noteworthy but in-scope adjustment: while
verifying Task 1, the ruff/ruff-format pass on the notebook (Task 3) reformatted several
`print(...)` calls onto multiple lines; the notebook was re-executed with `jupyter nbconvert
--to notebook --execute --inplace` afterward so the committed source and output stay in
sync, per the plan's own instruction for that step.

## Issues Encountered

None. All verify commands and the plan-level verification section passed on first or second
attempt (the ruff-format-triggered notebook re-execution above was the only rework needed).

## Notes for the architectural exploration (per plan's `<output>` requirements)

**(a) The existing `backfill_lco_observation_records` command.** It already existed
(365 lines, verified at planning time) and was deliberately left untouched (D-A). One-sentence
contract difference: it requires a campaign (`TargetList`) and a `--name-prefix`, skips
requests whose `ObservationRecord` already exists, and (with `--create-missing-targets`)
builds *sidereal* field targets from RA/Dec — the new `backfill_lco_observations` needs no
campaign, updates existing records in place, and only ever builds *non-sidereal* targets
from orbital elements.

**(b) Which D-B path actually fires: embedded blocks, or the live fallback?** This could not
be confirmed against a real LCO portal response during this task (no live network access is
used or available in this environment). Based on the code evidence available —
`LCOFacility.get_observation_status()` (`tom_observations/facilities/ocs.py:1548-1575`) makes
its *own* two separate calls (`GET /api/requests/{id}` for state, then
`GET /api/requests/{id}/observations/` for the block list) rather than reading anything already
present in a `GET /api/requestgroups/` response — the working assumption is that the
**RequestGroups listing endpoint does not itself embed per-request observation blocks**, so
the D-B *live fallback* is the path that actually fires in practice. The command still
implements the embedded-block short-circuit exactly as D-B specifies (and it's exercised
directly by `test_uses_embedded_observation_block_when_present` and demonstrated as a
documented-but-secondary path in the notebook), but the primary test/notebook demonstrations
deliberately exercise the fallback path as the realistic one. This assumption should be
verified against a real portal response before the architectural exploration leans on the
embedded-block path being common.

## Self-Check: PASSED

- FOUND: `solsys_code/management/commands/backfill_lco_observations.py`
- FOUND: `solsys_code/tests/test_backfill_lco_observations.py`
- FOUND: `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`
- FOUND: `docs/runbooks/telescope_runs_calendar.rst`
- FOUND: `CLAUDE.md`
- FOUND commit: `8a9c797`
- FOUND commit: `3f118f6`
- FOUND commit: `f874531`
- Re-ran plan-level `<verification>` items 1-7: all passed (help renders 4 flags; new suite
  16/16 green; neighbouring 3 LCO/calendar test modules 96/96 green combined; ruff and
  ruff-format clean repo-wide; sphinx-build clean; sibling command/test `git status
  --porcelain` empty; notebook re-executes with no network, output byte-identical modulo
  execution timestamps, which were reverted to keep the committed notebook clean).
