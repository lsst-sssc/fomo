---
phase: 30-v2-2-tech-debt-cleanup
plan: 02
subsystem: tech-debt-cleanup
tags: [ruff, pre-commit, dev-deps, docstrings, planning-bookkeeping]

# Dependency graph
requires:
  - phase: 30-v2-2-tech-debt-cleanup
    provides: "30-01's attribution eligibility fix (unrelated file scope, no overlap)"
provides:
  - "pyproject.toml's dev extra pins ruff==0.2.1, matching .pre-commit-config.yaml's astral-sh/ruff-pre-commit rev, so a fresh `pip install '.[dev]'` cannot reproduce the phantom drift Phases 26/27/27.1 each logged"
  - "CLAUDE.md documents both lint/format gates as pre-commit-mediated invocations in both places it mentions them (Commands block, Testing constraints bullet)"
  - "campaign_reconciler.py's five docstrings no longer name the deleted _project_calendar_event()/_calendar_event_title() functions"
  - "26-DECISION.md's header preamble reports Phase 26 as complete and names all five plans"
affects: [30-03, 30-04]

# Actuals (#2632)
actuals:
  tokens: 1818
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ruff toolchain drift is closed at the source (dev-dependency pin + documented invocation), never by running a repo-wide reformat -- the repo was already clean under the pinned version"
    - "Docstring dead-symbol repair keeps the historical fact when it explains a current choice (replacing only the dead name), and restates an unverifiable comparison as the present requirement when it does not"

key-files:
  created: []
  modified:
    - pyproject.toml
    - CLAUDE.md
    - solsys_code/campaign_reconciler.py
    - .planning/phases/26-canonical-record-spike/26-DECISION.md

key-decisions:
  - "D-06/D-07: pinned pyproject.toml's ruff entry to exactly match .pre-commit-config.yaml's rev (0.2.1, no leading v), and routed both CLAUDE.md mentions of the lint/format gate through `pre-commit run <hook-id> --all-files` instead of a bare, unpinned ruff invocation"
  - "D-05 held: no formatter or autofix pass was run over the repository; both pre-commit hooks passed with zero files reformatted, confirming the drift was never dirty code"
  - "D-10: kept every load-bearing historical fact in the five repaired docstrings (the retired helper's bare-key-when-n_nights==1 behavior, the preserved 'no event yet' cases, the ground-loop description) and replaced only the dead symbol name; restated the one unverifiable 'byte-identical to' claim as the actual requirement (keep the terminal cancelled/weathered prefix form)"
  - "D-10: extended 26-DECISION.md's per-plan sentence with 26-04 and 26-05 rather than rewriting the paragraph, keeping the accurate 26-01..26-03 sentences untouched"

requirements-completed: [D-05, D-06, D-07, D-10]

coverage:
  - id: D1
    description: "pyproject.toml's dev extra pins ruff to the same version .pre-commit-config.yaml pins"
    requirement: "D-06"
    verification:
      - kind: other
        ref: "grep -oP '(?<=ruff==)[0-9.]+' pyproject.toml == grep -A2 astral-sh/ruff-pre-commit .pre-commit-config.yaml rev"
        status: pass
    human_judgment: false
  - id: D2
    description: "CLAUDE.md documents the lint/format gates as pre-commit-mediated invocations everywhere it mentions them, with no bare unpinned invocation left"
    requirement: "D-07"
    verification:
      - kind: other
        ref: "grep -cE 'ruff (check|format) \\.' CLAUDE.md == 0"
        status: pass
      - kind: other
        ref: "grep -c 'pre-commit run ruff --all-files' CLAUDE.md >= 1"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both gates pass under the pinned version with no file reformatted"
    requirement: "D-05"
    verification:
      - kind: other
        ref: "pre-commit run ruff --all-files && pre-commit run ruff-format --all-files, both Passed, no files-modified output"
        status: pass
    human_judgment: false
  - id: D4
    description: "campaign_reconciler.py's docstrings no longer name deleted functions; the one deliberately-kept mention in reconcile_campaign_runs.py survives; tests still pass"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "grep -cE '_project_calendar_event|_calendar_event_title' solsys_code/campaign_reconciler.py == 0"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs -v1 -- 49 tests OK"
        status: pass
    human_judgment: false
  - id: D5
    description: "26-DECISION.md's header reports the document as complete and names all five plans that built it"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "grep -c 'In progress' 26-DECISION.md == 0; grep -c '26-04' >= 1; grep -c '26-05' >= 1; git diff -U0 hunk count == 2"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-08-31
status: complete
---

# Phase 30 Plan 02: Ruff Toolchain Root Cause & Cosmetic Bookkeeping Summary

**Pinned ruff's dev-extra dependency and CLAUDE.md's documented lint/format command to the exact version pre-commit already enforces, closing the phantom drift Phases 26/27/27.1 each independently logged, and repaired two cosmetic bookkeeping items (stale docstring references, a stuck-in-progress decision-doc header) flagged by the milestone audit.**

## Performance

- **Duration:** ~10 min (commit-to-commit)
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Closed the root cause behind three phases independently rediscovering the same
  non-finding: `pyproject.toml`'s dev extra now pins `ruff==0.2.1`, matching
  `.pre-commit-config.yaml`'s `astral-sh/ruff-pre-commit` rev exactly, so a fresh
  `pip install '.[dev]'` can no longer drift to an unpinned newer ruff (D-06)
- Corrected the documented invocation in both places CLAUDE.md names it (the `##
  Commands` lint block and the `§Project > Constraints` Testing bullet) to
  `pre-commit run ruff --all-files` / `pre-commit run ruff-format --all-files`, so the
  documented command and the enforced version are now the same thing (D-07)
- Verified D-05's premise directly: both pre-commit hooks pass under the pinned version
  with zero files reformatted — the repo was never dirty, only the documented command was
  wrong
- Repaired all five stale docstring references in `campaign_reconciler.py` to functions
  deleted in Phase 29 (`_project_calendar_event()`, `campaign_views._calendar_event_title()`),
  preserving every load-bearing historical fact and restating one unverifiable
  "byte-identical to" comparison as the actual present-tense requirement (D-10)
- Corrected `26-DECISION.md`'s header preamble, which still said "In progress" and named
  only plans 26-01 through 26-03 — it now reports the document as complete and names all
  five plans, including what 26-04 and 26-05 each contributed (D-10)

## Task Commits

Each task was committed atomically:

1. **Task 1: Pin ruff in the dev extra and route CLAUDE.md's documented gate through pre-commit** - `9c4a076` (fix)
2. **Task 2: Repair the five stale docstring references in campaign_reconciler.py** - `e2a2739` (docs)
3. **Task 3: Correct 26-DECISION.md's header preamble** - `b22bf53` (docs)

## Files Created/Modified

- `pyproject.toml` - `ruff` dev-extra entry pinned to `==0.2.1` with a comment citing D-06 and `.pre-commit-config.yaml` as source of truth
- `CLAUDE.md` - `## Commands` lint block and the Testing constraints bullet both now document `pre-commit run ruff --all-files` / `pre-commit run ruff-format --all-files` instead of a bare, unpinned invocation
- `solsys_code/campaign_reconciler.py` - five docstrings (`ReconcileResult`, `run_night_url`, `event_title`, `_skip_reason`, `_reconcile_classical_nights`) repaired in prose only; no executable line changed
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` - header `**Status:**` changed from in-progress to complete, per-plan sentence extended to name 26-04 and 26-05

## Decisions Made

- Reproduced the plan's cited ruff-version contrast directly: the pinned pre-commit
  binary (v0.2.1) reports both hooks Passed with no files modified, confirming the repo
  was already clean and the three prior phases' findings came from an unpinned dev-environment
  ruff, not from dirty code
- For each of the five stale docstrings, applied the plan's per-site rule: kept the fact
  when it explained a current choice (`run_night_url`'s bare-key-when-single-night
  history, `_skip_reason`'s preserved-cases list, `_reconcile_classical_nights`' ground-loop
  description) and replaced only the dead symbol name; restated `event_title`'s
  unverifiable "byte-identical to" comparison as the present requirement instead
- Extended 26-DECISION.md's per-plan sentence rather than rewriting the paragraph, so the
  already-accurate 26-01/26-02/26-03 sentences stayed untouched (confirmed by the `git
  diff -U0` hunk-count acceptance criterion: exactly 2 hunks, both inside the header)

## Deviations from Plan

None — plan executed exactly as written. No auto-fixes, no blocking issues, no
architectural questions arose.

## Issues Encountered

None. `pre-commit --version` (4.6.0) was already on PATH and its hook environments were
already installed from Plan 30-01's execution, so Task 1's precondition was satisfied
with no setup needed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- D-05/D-06/D-07 (ruff toolchain root cause) and D-10 (cosmetic bookkeeping: reconciler
  docstrings, 26-DECISION.md header) are complete and independently verified
- `.planning/v2.2-MILESTONE-AUDIT.md`'s ruff/WR-09/WR-10 stale entries and the two D-10
  cosmetic items are ready for Plan 30-04 to record as closed (D-09 amendment is
  explicitly out of this plan's scope, per the plan's task list)
- Plans 30-03 (Nyquist validation reconciliation) and 30-04 (bookkeeping/audit amendment)
  can proceed independently — neither touches `pyproject.toml`, `CLAUDE.md`,
  `campaign_reconciler.py`, or `26-DECISION.md`
- No blockers

## Self-Check: PASSED

- All 4 `files_modified` paths verified present on disk
- All 3 task commit hashes (`9c4a076`, `e2a2739`, `b22bf53`) verified in `git log`
- `grep -oP '(?<=ruff==)[0-9.]+' pyproject.toml` = `0.2.1`, matches `.pre-commit-config.yaml`'s `rev: v0.2.1`
- `grep -cE 'ruff (check|format) \.' CLAUDE.md` = 0
- `grep -cE '_project_calendar_event|_calendar_event_title' solsys_code/campaign_reconciler.py` = 0
- `grep -c 'In progress' .planning/phases/26-canonical-record-spike/26-DECISION.md` = 0
- `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs -v1` re-run: OK (49 tests)
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both Passed, no files modified

---
*Phase: 30-v2-2-tech-debt-cleanup*
*Completed: 2026-08-31*
