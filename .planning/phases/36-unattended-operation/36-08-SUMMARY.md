---
phase: 36-unattended-operation
plan: 08
subsystem: docs
tags: [settings, credentials, runbook, gap-closure, django-settings]

# Dependency graph
requires:
  - phase: 36-07
    provides: the 9-numbered-step fresh-host setup procedure and its heartbeat-check step, which this plan's step-2 rewrite must coexist with unchanged
provides:
  - "src/fomo/settings.py's LCO_API_KEY fold now fills both FACILITIES['LCO']['api_key'] and FACILITIES['SOAR']['api_key'] from one flat setting"
  - "A committed test (solsys_code/tests/test_settings_api_key_fold.py) that executes the real fold tail of the live settings module against an injected local-settings module, proving the flat assignment reaches both facilities, an absent setting is a clean no-op, the bracketed dict-subscript form raises NameError, and the SOAR accessor reads the fold target"
  - "Runbook step 2 of 'Setting it up on a fresh host' names the flat, top-level LCO_API_KEY assignment an operator can actually write, explains the NameError failure mode in one clause, states what settings.py folds it into and why one key covers both facilities, and states the omitted-setting no-op"
  - "36-VERIFICATION.md's SC-5 sufficiency read-through hold sites now state a release condition (G-36-4 closed by this plan) instead of deferring on an unfixed defect"
affects: [37-status-vocabulary-public-tallies-provenance-blind-gaps, ship-decision-for-v2.4]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
# Same estimateTokens scale (chars/4 over the realized diff), never a harness token count.
actuals:
  tokens: 3597
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Executable-truth test gate: a committed test that slices and executes the REAL tail of a settings module against an injected sys.modules entry, rather than grepping the source for a token, so a future edit that drops a fold line fails the test instead of passing a source-token grep (the exact class of gap G-36-4 was: 36-07's token-level runbook gate passed while the instruction it checked was false)"

key-files:
  created:
    - solsys_code/tests/test_settings_api_key_fold.py
  modified:
    - src/fomo/settings.py
    - docs/runbooks/telescope_runs_calendar.rst
    - .planning/phases/36-unattended-operation/36-VERIFICATION.md

key-decisions:
  - "Promoted the single flat LCO_API_KEY setting to feed both the LCO and the SOAR facility entries (assumption-delta decision in the plan) rather than adding a second SOAR_API_KEY setting -- there is one credential for the LCO Observation Portal, not two, and a second name would give an operator two things to set, rotate, and let drift."
  - "The new test module executes the real fold tail out of the live settings file (sliced from the try/from-fomo.local_settings-import-star anchor to end of file and exec'd against an injected sys.modules entry), rather than re-implementing the fold's logic, so a future edit that drops the SOAR line fails this test instead of a source-token grep."

patterns-established:
  - "Slice-and-exec test pattern for settings-module folds: locate the live settings module via DJANGO_SETTINGS_MODULE, slice its source from a literal anchor to end of file, exec that slice against a synthetic namespace with an injected sys.modules entry standing in for the operator's local settings module -- proves real behavior without ever touching the real, possibly credential-bearing, local settings file on disk."

requirements-completed: [SCHED-08, SCHED-10]

coverage:
  - id: D1
    description: "src/fomo/settings.py's LCO_API_KEY fold extended to also fill FACILITIES['SOAR']['api_key'], proven by a new committed test executing the real fold against an injected local-settings module (flat key reaches both facilities; absent key is a clean no-op; the bracketed dict-subscript form raises NameError; SOARSettings('SOAR').get_setting('api_key') reads the fold target)"
    requirement: "SCHED-10"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_settings_api_key_fold.py (4 tests, all pass, OK)"
        status: pass
      - kind: other
        ref: "python -c 'import django; django.setup(); ...' printing FOLD_TARGETS ['LCO', 'SOAR'] -- confirms the live settings module still imports and both facility entries carry an api_key key"
        status: pass
      - kind: unit
        ref: "solsys_code.tests.test_check_unattended (30 tests) and solsys_code.tests.test_unattended (63 tests), both OK -- unattended path untouched"
        status: pass
      - kind: other
        ref: "git diff --numstat 7d61d6b -- src/fomo/settings.py reports 2 insertions, 0 deletions; git diff --name-only 7d61d6b -- solsys_code/ lists only the new test module"
        status: pass
    human_judgment: false
  - id: D2
    description: "Runbook step 2 of 'Setting it up on a fresh host' rewritten so an operator can follow it verbatim: names the flat LCO_API_KEY placeholder assignment, the NameError failure mode in one clause, what settings.py folds it into and why one key covers both facilities, and the omitted-setting no-op -- with no bracketed settings-dict subscript anywhere in the subsection"
    requirement: "SCHED-08"
    verification:
      - kind: other
        ref: "task2 automated gate: awk-sliced 'Setting it up on a fresh host'..'Adding a proposal to watch' asserts the placeholder assignment form, absence of any FACILITIES['...'] subscript, presence of 'NameError' and 'same LCO Observation Portal', and absence of any UUID-shaped string in the whole runbook"
        status: pass
      - kind: other
        ref: "plan 36-07's slice-scoped gate re-run verbatim: Period/Grace/*/15 * * * */hc-ping.com/<uuid>/healthchecks.io/'The two failure signals'/'35 min'/9th numbered step all present, and the Period line and ping-URL line both precede the export step's anchor phrase"
        status: pass
      - kind: other
        ref: "read-only docutils parse of the whole runbook -- no enumerated-list, indentation, block-quote, literal-block, explicit-markup or title-underline message"
        status: pass
      - kind: other
        ref: "git diff -U0 7d61d6b -- docs/runbooks/telescope_runs_calendar.rst -- exactly one hunk, at line 1461, entirely inside step 2 (before step 3 at line 1471)"
        status: pass
    human_judgment: true
    rationale: "SC 5's real test is sufficiency at point of use for a naive reader working the runbook top-down with no source access -- exactly the SC-5 sufficiency read-through this plan's Task 3 releases the hold on. workflow.human_verify_mode is end-of-phase (default), so per checkpoints.md that human-check is deferred to the phase's end-of-phase UAT rather than run as a mid-flight checkpoint; the automated gates above prove the content is present, correctly formed, and correctly placed, but only a naive human reader can prove it is enough to actually provision a fresh host."
  - id: D3
    description: "36-VERIFICATION.md's SC-5 sufficiency read-through hold sites (all three: frontmatter human_items_still_open entry 1, frontmatter human_verification entry 1, prose Human Verification Required item 1) now state a release condition naming plan 36-08 and the fix, instead of deferring administration on an unfixed defect; no verdict, score, status marker or gaps: entry changed"
    requirement: "SCHED-08"
    verification:
      - kind: other
        ref: "task3 automated gate: grep -cF '36-08' >= 3; grep -cF 'known-broken' == 0; grep -cF 'do not administer until' == 0; grep -qF 'G-36-4'; grep -qF 'LCO_API_KEY'"
        status: pass
      - kind: other
        ref: "marker-order gate: 'SC-5 sufficiency read-through' first-occurrence (line 71) and last-occurrence (line 446) both precede 'Live heartbeat dead-man re-run' first-occurrence (72) and last-occurrence (462)"
        status: pass
      - kind: other
        ref: "verdict-parity gate: counts of '✗ **FAILED**' (3) and '✓ VERIFIED' (74) unchanged from git show 7d61d6b:<file>; status: gaps_found and score: 61/65 must-haves verified unchanged"
        status: pass
      - kind: other
        ref: "git diff -U0 7d61d6b -- .planning/phases/36-unattended-operation/36-VERIFICATION.md -- exactly 3 hunks, at lines 71, 131, and 457 (the three named hold sites)"
        status: pass
    human_judgment: false

# Metrics
duration: ~15min
completed: 2026-09-18
status: complete
---

# Phase 36 Plan 08: Fresh-Host API-Key Fold and Runbook Fix Summary

**Extended the LCO_API_KEY settings fold to also authenticate the SOAR facility, proved it with an executable test against the real fold tail, and rewrote runbook step 2 so a fresh-host operator writes the flat assignment Django can actually import instead of the bracketed FACILITIES[...] path that raised NameError and stopped Django from starting.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-09-18
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- `src/fomo/settings.py`'s `if 'LCO_API_KEY' in globals():` fold now assigns the same flat value into both `FACILITIES['LCO']['api_key']` and `FACILITIES['SOAR']['api_key']`, with one comment line stating SOAR authenticates against the same LCO Observation Portal — closing the SOAR half of gap G-36-4, which had no fold at all before this plan.
- New `solsys_code/tests/test_settings_api_key_fold.py` (4 `SimpleTestCase` tests, no database) executes the real fold tail sliced out of the live settings module against an injected `fomo.local_settings` module in `sys.modules` — never reading or writing the real, possibly credential-bearing, local settings file on disk. Proves: the flat key reaches both facilities; an absent key is a clean no-op; the bracketed dict-subscript form the old runbook documented raises `NameError`; and `SOARSettings('SOAR').get_setting('api_key')` reads the entry the fold fills.
- `docs/runbooks/telescope_runs_calendar.rst` step 2 of "Setting it up on a fresh host" replaced the two sentences that told the operator to write `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` — an assignment that raises `NameError` inside `local_settings.py`'s import namespace and stops Django from starting — with instruction naming the flat `LCO_API_KEY = '<your key>'` placeholder assignment, the `NameError` failure mode in one clause, what `settings.py` folds it into and why one key covers both facilities ("the same LCO Observation Portal"), and what an omitted setting leaves behind (both facility entries stay empty, portal calls go out unauthenticated).
- `.planning/phases/36-unattended-operation/36-VERIFICATION.md`'s SC-5 sufficiency read-through hold — at all three sites (frontmatter `human_items_still_open`, frontmatter `human_verification`, and the prose "Human Verification Required" item 1) — now states a release condition naming this plan and the fix, instead of deferring administration on an unfixed defect. No verdict, score, status marker, or `gaps:` entry was touched; that record belongs to re-verification.

## Task Commits

Each task was committed atomically:

1. **Task 1: Make the one flat key reach both facilities, and prove it end-to-end** - `62d4d78` (feat)
2. **Task 2: Rewrite step 2's API-key instruction so an operator can follow it verbatim, and gate the class** - `39312f4` (docs)
3. **Task 3: Release the SC-5 read-through's hold condition in the verification record** - `96701a3` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/fomo/settings.py` - `LCO_API_KEY` fold extended to also fill `FACILITIES['SOAR']['api_key']` (2 lines added: one comment, one assignment)
- `solsys_code/tests/test_settings_api_key_fold.py` - NEW: executes the real fold tail against an injected local-settings module; 4 test cases (flat key reaches both facilities, absent key is a clean no-op, bracketed subscript raises `NameError`, SOAR accessor reads the fold target)
- `docs/runbooks/telescope_runs_calendar.rst` - Step 2 of "Setting it up on a fresh host" rewritten: flat placeholder assignment, `NameError` clause, what the fold covers and why, omitted-setting no-op
- `.planning/phases/36-unattended-operation/36-VERIFICATION.md` - All three SC-5 sufficiency read-through hold sites now state a release condition naming plan 36-08; no verdict/score/status changed

## Decisions Made

- Promoted the single flat `LCO_API_KEY` setting to feed both facility entries (assumption-delta decision recorded in the plan) rather than adding a second `SOAR_API_KEY` — one credential for the one portal, not two names to let drift.
- The new test executes the real fold tail rather than re-implementing it, so a future regression fails this test instead of a source-token grep — the exact lesson of G-36-4 (36-07's token-level gate passed while the instruction it checked was false).

## Deviations from Plan

None - plan executed exactly as written. The scope fence held: no advisory finding from the "Deferred (do not touch)" table (CR-03, WR-24..27, IN-20, IN-25..28) was folded in, and no file outside the plan's declared `files_modified` was touched.

## Paired-Docs Check (CLAUDE.md)

None owed. `src/fomo/settings.py` has no notebook mapped to it in CLAUDE.md's pairing map. The paired doc for the unattended runner and `check_unattended` is the runbook's "How do I run everything unattended?" section, which this plan's Task 2 already edits (step 2 of the fresh-host setup subsection inside that section) — no separate notebook is owed for it, per CLAUDE.md's explicit carve-out for that section. This is stated explicitly per the plan's Task 3 action item so the paired-docs check has an answer on the record rather than an absence.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Nothing was written to `src/fomo/local_settings.py`; that file is gitignored and untouched by this plan.

## Next Phase Readiness

- Gap G-36-4 is closed in the codebase: the fresh-host procedure's step 2 now names an assignment Django can actually import, and the fold it describes now really does authenticate both the LCO and the SOAR facilities.
- `36-VERIFICATION.md`'s SC-5 sufficiency read-through no longer reads as blocked on an unfixed defect — it has a stated release condition. The read-through itself (a naive reader working the runbook top-down) is still a human item, pending the phase's end-of-phase UAT / re-verification pass.
- No source file outside `src/fomo/settings.py` and the new test module changed; `solsys_code/unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py`, and both existing regression modules (`test_check_unattended.py` 30 tests, `test_unattended.py` 63 tests) are byte-identical to their state at commit `7d61d6b` and remain green.
- Re-verification of Phase 36 can now proceed against a fresh-host procedure that a naive operator can actually follow.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-18*

## Self-Check: PASSED

- FOUND: src/fomo/settings.py
- FOUND: solsys_code/tests/test_settings_api_key_fold.py
- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND: .planning/phases/36-unattended-operation/36-VERIFICATION.md
- FOUND commit: 62d4d78
- FOUND commit: 39312f4
- FOUND commit: 96701a3
- Re-ran plan-level `<verification>` list: `test_settings_api_key_fold` (4 tests, OK); live settings module import printed `FOLD_TARGETS ['LCO', 'SOAR']`; `test_check_unattended` (30, OK) and `test_unattended` (63, OK) both green; `git diff --name-only 7d61d6b -- solsys_code/` lists only the new test module; fresh-host slice contains the placeholder assignment / `NameError` / same-portal phrase and no bracketed subscript; plan 36-07's slice gate still passes; no UUID-shaped string in any of the three touched files; docutils parse of the runbook emits no structural message; `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both Passed; `36-VERIFICATION.md` names plan 36-08 at all three hold sites, keeps the two human-test markers in order, and still reports 3 failed / 74 verified, `status: gaps_found`, `score: 61/65 must-haves verified`.
