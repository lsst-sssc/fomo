---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 07
subsystem: docs-and-vocabulary-migration
tags: [django, sphinx, jupyter, status-vocabulary, test-infrastructure]

requires:
  - phase: 37-01
    provides: "solsys_code/status_vocabulary.py -- MARKER/LABEL/LEGEND/RUN_STATUS_MARKER/RETIRED_TITLE_PREFIXES/state_for_title(), the vocabulary this plan finishes migrating and documents"
  - phase: 37-02
    provides: "solsys_code/proposal_allocation.py and unattended.step_proposal_allocation() -- the fifth runner step this plan documents in the runbook"
  - phase: 37-03
    provides: "campaign_gap.observation_claimed_dates()/observation_site_obscode() -- the second claim source this plan's new coverage-gap runbook section describes"
  - phase: 37-04
    provides: "solsys_code/campaign_tally.py -- the tally module this plan's runbook section and notebook demonstrations describe/exercise"
  - phase: 37-05
    provides: "CampaignRunTable's Progress column and campaign_tally.get_or_compute_rollup() -- what the new notebook cells GET and assert against"
  - phase: 37-06
    provides: "calendar_display_extras.run_tally()/unused_night_decoration() -- what the new notebook cells' pop-up-tally and unused-night demonstrations exercise"
provides:
  - "docs/runbooks/telescope_runs_calendar.rst -- the final nine-marker vocabulary (no 'provisional' language), a public-tally section with the freshness bound, a fresh coverage-gap-analysis section, a five-step unattended-operation section, and a one-time title-change note for the run-level [C]/[W] prefixes"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb -- three new demonstrations (public run tally + campaign roll-up, calendar pop-up tally, an unused awarded night beside a weathered one) with real executed output, plus two stale pre-existing cells fixed so the notebook executes cleanly against current code"
  - "solsys_code/status_vocabulary.py with exactly one spelling per marker -- RETIRED_TITLE_PREFIXES and its state_for_title() branch deleted"
  - "A whole-suite test invocation, python manage.py test solsys_code --exclude-tag=ephemeris_segfault, that actually passes (1703 tests, OK) -- the ephemeris_segfault tag plus a fix for a global logging.disable() bug the single-invocation form exposed for the first time"
affects: []

actuals:
  tokens: 47996
  tasks: 3
  commits: 3
  plan_head_before: 747ecea0508f80e6b7d1782bc8a6fa345491d9fc

tech-stack:
  added: []
  patterns:
    - "logging.disable() scoped to setUpModule()/tearDownModule() rather than bare module-level, so a test module's own noise-suppression never leaks into sibling modules sharing the same whole-suite process"
    - "Django test tags (@tag('name') + --exclude-tag=name) as the mechanism for a documented, known-crash test class to be excluded by the invocation itself rather than by prose beside the command"

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
    - solsys_code/status_vocabulary.py
    - solsys_code/tests/test_status_vocabulary.py
    - solsys_code/tests/test_views.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "Task 3's precondition (both re-title sweeps must leave zero legacy-spelled CalendarEvent titles) passed with `updated: 0` on both sweeps -- the developer database already held no bracket-word titles by the time this plan ran, so RETIRED_TITLE_PREFIXES was deleted immediately with no re-title work actually needed. The precondition check itself still ran and is still the correct gate for a deployment that has NOT already converged."
  - "project_observation_calendar_demo.ipynb was regenerated and confirmed but NOT recommitted: the record-level [Q]/[S]/[O]/[X]/[C]/[F]/[?] letters are genuinely unchanged by this phase, and the only diff a re-execution produces is real developer-database/live-portal timestamp drift (SCHED-06 baseline moving forward with real observing nights) -- exactly the case the plan's own action text says to 'confirm... rather than forcing a diff'. Re-running also regenerated `project_observation_calendar_demo.sched06-baseline.json` as a side effect; that file was reverted via `git checkout --` to avoid committing unrelated real-time drift."
  - "Task 2's regeneration surfaced two genuinely pre-existing, unrelated staleness issues in campaign_lifecycle_demo.ipynb that blocked execution entirely and were fixed as in-scope Rule 1 bugs (not deferred): a stale FTN/FTS/SOAR site-match cell the notebook's own markdown already flagged as unexecuted since 2026-09-11, and a four-way-payoff cell whose assertions predated a dispatch-rule change -- the notebook's classical run now dispatches to the per-night `ALLOC:` family (via `dispatches_per_night()`), not the old per-night `RUN:` family, while the queue-sourced and class-wide runs share the whole-window `RUN:` container. Both cells and the Summary prose describing them were rewritten to match current, real behavior."
  - "classical_run's own three `ALLOC:` nights (window 2026-09-01..2026-09-03, a date hardcoded at notebook-authoring time, now genuinely in the past relative to the real 2026-09-19 execution date) served as the natural 'unused awarded night' demonstration -- no synthetic future-dated run was needed for the ordinary-status half of UNUSED-01's D-14 precedence demo, only for the weathered-run comparison."
  - "Discovered and fixed a real, previously-undetected bug: `test_views.py`'s bare module-level `logging.disable(logging.CRITICAL)` executes at Django's test-discovery import time and silences ALL logging, in every logger, for the rest of the whole-suite process -- breaking any later test's `assertLogs()` (found via `test_allocation_projector.TestDeclinedNightResolutionCostIsBounded`/`TestRemintHumanConfirmationGuard`). This was masked for the project's entire history because the documented `test_command` always ran `test_views.py` in a separate `manage.py test` invocation; Task 3's whole point -- a single `--exclude-tag=ephemeris_segfault` invocation -- is the first time `test_views.py` has ever shared a process with the rest of the suite, which is what exposed it. Fixed by scoping the disable/restore to `setUpModule()`/`tearDownModule()`."

requirements-completed: [STATUS-01, TALLY-01, TALLY-02, UNUSED-01, GAPB-01]

coverage:
  - id: D1
    description: "The runbook documents the final nine-marker vocabulary (adds [W] weathered and [U] unused, corrects [C]/[S]) with no remaining 'provisional' language, and states one module defines it"
    requirement: STATUS-01
    verification:
      - kind: other
        ref: "python -c checks against docs/runbooks/telescope_runs_calendar.rst for '[W]'/'[U]'/absence of 'vocabulary is provisional' (37-07-PLAN.md Task 1 verify)"
        status: pass
      - kind: other
        ref: "pre-commit run --all-files (Sphinx docs build)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A fresh coverage-gap-analysis section (none existed before) names both claim sources, site resolution, and the claimed-site-unknown count"
    requirement: GAPB-01
    verification:
      - kind: other
        ref: "python -c check: src.count('Coverage') > 0 in the runbook (37-07-PLAN.md Task 1 verify)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The unattended-operation section lists five steps (adds proposal_allocation) and describes its failure signal; the cheat-sheet row and a troubleshooting entry are updated to match"
    requirement: TALLY-01
    verification:
      - kind: other
        ref: "python -c checks: 'five steps' present, 'four steps' absent, 'proposal_allocation' present (37-07-PLAN.md Task 1 verify)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The runbook states the tally's freshness bound in operator words -- record-driven changes on the next page load, time-driven ones within TALLY_CACHE_TTL_SECONDS -- naming the constant"
    requirement: TALLY-01
    verification:
      - kind: other
        ref: "python -c checks: 'next page load' and 'TALLY_CACHE_TTL_SECONDS' both present (37-07-PLAN.md Task 1 verify)"
        status: pass
    human_judgment: false
  - id: D5
    description: "campaign_lifecycle_demo.ipynb demonstrates the public run tally, the campaign roll-up, the calendar pop-up tally, and an unused awarded night beside a weathered one, with real executed output and embedded assertions that all pass"
    requirement: TALLY-01
    verification:
      - kind: other
        ref: "jupyter nbconvert --to notebook --execute (real run against a scratch DB copy); every embedded assert in the three new cell pairs passed; every code cell carries a non-null execution_count"
        status: pass
    human_judgment: false
  - id: D6
    description: "campaign_lifecycle_demo.ipynb's Progress-cell/roll-up demonstration asserts the anonymous and staff responses render identical tally segments for the same run"
    requirement: TALLY-02
    verification:
      - kind: other
        ref: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb cell asserting content_public/content_staff segment parity, executed output printed"
        status: pass
    human_judgment: false
  - id: D7
    description: "An unused awarded night (classical_run's own elapsed, ordinary-status ALLOC: nights) renders the muted [U] chip and token; a staff-weathered run's equally elapsed night renders [W] instead (D-14 precedence); rendering never rewrites a stored title"
    requirement: UNUSED-01
    verification:
      - kind: other
        ref: "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb 'An unused awarded night' cell -- cal-event-unused class, data-unused attribute, [U]-token count >= 3, '[W] Y21' present, before/after title byte-identity assertion, executed output printed"
        status: pass
    human_judgment: false
  - id: D8
    description: "After the re-title sweep, no CalendarEvent in the developer database carries a legacy bracket-word status prefix, and RETIRED_TITLE_PREFIXES is deleted from status_vocabulary.py -- exactly one spelling of every marker"
    requirement: STATUS-01
    verification:
      - kind: other
        ref: "python -c query: CalendarEvent legacy-prefix count == 0 against the developer database, after both sweeps"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestVocabularyStructure.test_state_for_title_returns_none_for_retired_bracket_word_prefixes"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_status_vocabulary.py#TestRunStatusMarker.test_cancelled_gets_the_terminal_ring_and_legacy_cancelled_resolves_to_no_state"
        status: pass
    human_judgment: false
  - id: D9
    description: "The whole-suite verification an executor actually runs (python manage.py test solsys_code --exclude-tag=ephemeris_segfault) exits 0 on its own -- the exclusion is a property of the invocation, not prose beside it"
    requirement: STATUS-01
    verification:
      - kind: unit
        ref: "python manage.py test solsys_code --exclude-tag=ephemeris_segfault"
        status: pass
    human_judgment: false

duration: ~135min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 07: Docs, Notebooks & Title-Migration Completion Summary

**Wrote the operator-facing paired docs Phase 37 owed (final nine-marker vocabulary, a public-tally section with its freshness bound, and a from-scratch coverage-gap-analysis section in the runbook), regenerated the four pre-executed notebooks with real executed output demonstrating TALLY-01/02 and UNUSED-01, and closed the STATUS-01 title migration by deleting `RETIRED_TITLE_PREFIXES` once both re-title sweeps proved the developer database held no legacy-spelled title -- discovering and fixing, along the way, a real bug where `test_views.py`'s global `logging.disable()` silently broke `assertLogs()`-based tests suite-wide the moment this plan made the whole-suite test invocation genuinely single-command.**

## Performance

- **Duration:** ~135 min (a large share spent running and diagnosing the plan's own mandated whole-suite verification, `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`, which surfaced a real cross-module test-isolation bug requiring root-cause investigation)
- **Started:** 2026-09-19T08:20Z (approx.)
- **Completed:** 2026-09-19T10:17Z
- **Tasks:** 3
- **Files modified:** 9 (0 created)

## Accomplishments

- `docs/runbooks/telescope_runs_calendar.rst`: the marker table now documents all nine markers ([W] weathered, [U] unused added; [C]/[S] descriptions corrected), the "provisional vocabulary" disclaimer is gone, a new "What does a run's or a campaign's public tally show?" section states the freshness bound in operator words (`TALLY_CACHE_TTL_SECONDS` named), a from-scratch "How do I find nights that were observed but never claimed by any approved run?" coverage-gap section covers both claim sources and the site-unknown count, the unattended-operation section now lists five steps (`proposal_allocation` added) with its failure signal, and a one-time-title-change note documents the run-level `[C]`/`[W]` migration.
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`: three new demonstrations with real executed output -- the public run tally + campaign roll-up (asserting anonymous/staff parity, TALLY-01/02), the calendar pop-up's live tally line, and an unused awarded night (classical_run's own elapsed nights) rendered beside a staff-weathered run's equally elapsed night (UNUSED-01's D-14 precedence) -- plus two pre-existing stale cells fixed so the notebook executes cleanly end-to-end against current code for the first time in this session.
- `reconcile_campaign_runs_demo.ipynb` and `load_telescope_runs_demo.ipynb`: re-executed after updating source assertions/prose from the legacy `[CANCELLED]` spelling to `[C]` (STATUS-01); `load_telescope_runs_demo.ipynb` also shows the newly structured `CampaignRun.proposal_code` field.
- `solsys_code/status_vocabulary.py`: `RETIRED_TITLE_PREFIXES` and its branch of `state_for_title()` are deleted -- confirmed via a live query that the developer database holds zero legacy-spelled `CalendarEvent` titles after running both re-title sweeps.
- `solsys_code/tests/test_views.py`: `TestEphemeris` is tagged `ephemeris_segfault`; `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` now runs the whole app suite in one invocation and passes (1703 tests, `OK`) -- which required also fixing a real, previously-masked bug (see Deviations).

## Task Commits

Each task was committed atomically:

1. **Task 1: Runbook -- the final vocabulary, the tallies, a coverage-gap section written fresh, and the fifth runner step** - `49ef5b5` (docs)
2. **Task 2: Regenerate the four pre-executed notebooks against the shipped behaviour** - `df40929` (docs)
3. **Task 3: Retire the legacy title spellings once the database has none left** - `1850322` (feat)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `docs/runbooks/telescope_runs_calendar.rst` - final vocabulary, public-tally section, coverage-gap section, five-step unattended section, one-time-title-change note
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - three new demonstrations; two stale cells and the approval-loop event count fixed
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` - `[CANCELLED]` -> `[C]` in assertions/prose, re-executed
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` - `[CANCELLED]` -> `[C]`, `proposal_code` demonstrated, re-executed
- `solsys_code/status_vocabulary.py` - `RETIRED_TITLE_PREFIXES` and its `state_for_title()` branch deleted; docstring/comment updated
- `solsys_code/tests/test_status_vocabulary.py` - ring-equivalence and retired-prefix tests updated to document the retirement
- `solsys_code/tests/test_views.py` - `ephemeris_segfault` tag added; module-level `logging.disable()` scoped to `setUpModule()`/`tearDownModule()`
- `solsys_code/tests/test_calendar_display_extras.py` - ring tests migrated from legacy bracket-word fixtures to the final short-letter markers
- `solsys_code/tests/test_calendar_template.py` - one fixture title migrated from `[FAILED]` to `[F]`

## Decisions Made

See `key-decisions` in the frontmatter above for full detail. In brief:

- Task 3's precondition passed with zero re-titling actually needed (`updated: 0` on both sweeps) -- the developer database already had no legacy-spelled titles.
- `project_observation_calendar_demo.ipynb` was regenerated and confirmed, not recommitted -- its record-level letters are genuinely unchanged, and forcing a diff would only commit real-time/live-portal drift.
- Two genuinely pre-existing, unrelated staleness issues in `campaign_lifecycle_demo.ipynb` (a self-documented-stale FTN site-match cell, and a four-way-payoff cell whose assumptions predated a dispatch-rule change) were fixed as in-scope Rule 1 bugs because they blocked this plan's own required regeneration.
- `classical_run`'s own already-elapsed `ALLOC:` nights served as the natural "unused" demonstration, needing no synthetic future-dated run.
- A real, previously-undetected `logging.disable()` scoping bug in `test_views.py` was found and fixed -- see Deviations below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stale FTN/FTS/SOAR site-match demo cell in `campaign_lifecycle_demo.ipynb`**
- **Found during:** Task 2's required notebook regeneration
- **Issue:** The cell's own markdown already documented it as "not re-executed here" since 2026-09-11, asserting `_extract_lco_site_code('FTN') == 'ogg'` via a now-dead `calendar_utils.OBSERVED_TELESCOPE_SITE_CODES` table -- re-executing it as written raised `AssertionError` on the very first loop iteration, since `_extract_lco_site_code()` now returns `None` for all three renamed labels.
- **Fix:** Rewrote the markdown and code cell to demonstrate the real, current bridge (`campaign_attribution.OBSERVED_TELESCOPE_OBSCODES`) directly, per the cell's own documented follow-up instructions.
- **Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
- **Verification:** `jupyter nbconvert --execute` succeeds past this cell; the rewritten cell's own assertions pass.
- **Committed in:** `df40929` (Task 2 commit)

**2. [Rule 1 - Bug] Four-way-payoff cell and its Summary paragraph assumed a dispatch shape the current code no longer produces**
- **Found during:** Task 2's required notebook regeneration
- **Issue:** The cell asserted the classical, LCO-queue and ESO-queue runs all render per-night `RUN:{pk}:{date}` events, with only the class-wide run rendering a bare `RUN:{pk}` container. Re-execution against current code produced `classical_run` with ZERO `RUN:`-namespace events (it now dispatches to the per-night `ALLOC:` family via `campaign_reconciler.dispatches_per_night()`), and the LCO-queue/ESO-queue runs each rendering a single bare `RUN:{pk}` container (not per-night events) -- both contradicting the notebook's committed assertions.
- **Fix:** Rewrote the cell to read both `owned_events()` (`RUN:` namespace) and `allocation_projector.allocation_events()` (`ALLOC:` namespace), and updated the assertions and Summary prose to state the real, current dispatch rule: only the classical run (non-queue source, resolved ground site, no `telescope_class`) takes the per-night `ALLOC:` family; the LCO-queue, ESO-queue and class-wide runs all share the whole-window `RUN:` container.
- **Files modified:** `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`
- **Verification:** Re-executed notebook's assertions pass against the real, current behavior; also fixed the "Approve all four" cell's `events=owned_events(run).count()` line, which was wrongly reporting 0 events for `classical_run` for the same namespace-blindness reason.
- **Committed in:** `df40929` (Task 2 commit)

**3. [Rule 1 - Bug] Legacy `[CANCELLED]` spelling in `reconcile_campaign_runs_demo.ipynb` and `load_telescope_runs_demo.ipynb` assertions and prose**
- **Found during:** Task 2's required notebook regeneration (explicitly named in the plan's own action text)
- **Issue:** Both notebooks asserted `.title.startswith('[CANCELLED]')` and described the marker as `[CANCELLED]`/`_TERMINAL_PREFIXES`, which 37-01 had already migrated to `[C]`/`status_vocabulary.RUN_STATUS_MARKER`.
- **Fix:** Updated assertions and prose to `[C]`/`status_vocabulary.RUN_STATUS_MARKER` throughout both notebooks; added a `proposal_code` demonstration to `load_telescope_runs_demo.ipynb`'s existing proposal-token cell.
- **Files modified:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`
- **Verification:** Both notebooks re-executed cleanly with `[C]`-spelled titles in their committed output.
- **Committed in:** `df40929` (Task 2 commit)

**4. [Rule 2 - Missing critical] Runbook's "How do I mark a run cancelled or weathered-out?" section still documented the retired `[CANCELLED]`/`[WEATHERED]` spelling**
- **Found during:** Task 1's mandated `read_first` review of that exact section
- **Issue:** The section (already updated in code by 37-01) still told an operator that clicking "Mark Cancelled"/"Mark Weathered" prepends `[CANCELLED]`/`[WEATHERED]`, which is now false.
- **Fix:** Updated to `[C]`/`[W]`, added a sentence naming the shared-marker rule (D-02) and a "One-time title change" note mirroring the existing Phase 34 precedent.
- **Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
- **Verification:** `pre-commit run --all-files` (Sphinx build) passes; manual re-read.
- **Committed in:** `49ef5b5` (Task 1 commit)

**5. [Rule 1 - Bug] `test_calendar_display_extras.py` and `test_calendar_template.py` ring-test fixtures broke when `RETIRED_TITLE_PREFIXES` was deleted**
- **Found during:** Task 3's own required verify command (`python manage.py test solsys_code.tests.test_status_vocabulary solsys_code.tests.test_calendar_display_extras`) and a follow-up full-suite run
- **Issue:** Several pre-existing tests asserted `status_border_css('[CANCELLED] x')`/`'[EXPIRED] x'`/`'[FAILED] x'`/`'[WEATHERED] x'` returned the terminal ring -- true only while `RETIRED_TITLE_PREFIXES` recognized those legacy bracket-word forms. Deleting the list (this task's own explicit deliverable) made these assertions false.
- **Fix:** Migrated the fixtures/assertions to the final short-letter markers (`[X]`/`[C]`/`[F]`/`[W]`), documenting in comments that the legacy forms are retired rather than silently dropping the cases.
- **Files modified:** `solsys_code/tests/test_calendar_display_extras.py`, `solsys_code/tests/test_calendar_template.py`
- **Verification:** `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` -- 179 tests, `OK`.
- **Committed in:** `1850322` (Task 3 commit)

**6. [Rule 1/Rule 3 - Bug/Blocking] `test_views.py`'s global `logging.disable(logging.CRITICAL)` silently broke `assertLogs()`-based tests suite-wide**
- **Found during:** Task 3's own mandated whole-suite verify command, `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`
- **Issue:** `test_views.py` executed a bare, module-level `logging.disable(logging.CRITICAL)` at import time. Because Django's test loader imports every test module during discovery -- before running any test in any module -- this permanently disabled ALL logging, in every logger, for the rest of the whole-suite process from the moment `test_views.py` was imported, not just for its own tests. This silently broke `test_allocation_projector.TestDeclinedNightResolutionCostIsBounded` and `TestRemintHumanConfirmationGuard` (both use `self.assertLogs(...)`, which is gated by the same process-wide `logging.Manager.disable` flag `logging.disable()` sets, ahead of the per-logger level `assertLogs()` configures). This bug is genuinely pre-existing -- it predates this plan entirely -- but was masked for the project's whole history because the documented `test_command` (`config.json` `workflow.test_command`) always ran `test_views.py` in a *separate* `manage.py test` invocation from the rest of the suite. This task's own explicit deliverable -- turning the whole-suite invocation into a real single command -- is what, for the first time, put `test_views.py` in the same process as the rest of the suite, exposing the bug. Fixing it was necessary to satisfy this task's own acceptance criterion (`python manage.py test solsys_code --exclude-tag=ephemeris_segfault` exits 0).
- **Fix:** Replaced the bare module-level call with `setUpModule()`/`tearDownModule()` functions that disable logging only for the duration of this module's own tests and restore it (`logging.disable(logging.NOTSET)`) afterward, so the effect never leaks into sibling modules sharing the same whole-suite process.
- **Files modified:** `solsys_code/tests/test_views.py`
- **Verification:** Reproduced the failure in isolation (both named tests fail only when run in the same process as `test_views.py`, and pass standalone); after the fix, `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` -- **Ran 1703 tests in 1343.135s, OK (skipped=1)**.
- **Committed in:** `1850322` (Task 3 commit)

---

**Total deviations:** 6 auto-fixed (5 Rule 1 bugs, 1 Rule 2 missing-critical, with deviation 6 also a Rule 3 blocking-issue fix). **Impact:** All six were necessary to make this plan's own required verification (real notebook re-execution; a genuinely single-command whole-suite test invocation) actually pass, rather than passing only on paper. None represent scope creep into unrelated feature work -- each was a direct, mechanical consequence of either regenerating the paired docs this phase owed or completing the title migration this plan owns. No deviation touched production business logic outside `solsys_code/status_vocabulary.py` (this plan's own declared deliverable) and one test-infrastructure fix (`test_views.py`'s logging scope).

## Issues Encountered

None beyond the deviations documented above -- every issue found was auto-fixed and verified within this plan's own scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

This is the final plan of Phase 37. All five of this plan's declared requirements (`STATUS-01`, `TALLY-01`, `TALLY-02`, `UNUSED-01`, `GAPB-01`) are shared with one or more sibling plans in this phase, all of which have now produced their own `*-SUMMARY.md` -- the shared-ID gate (`requirements.ready-ids`) reports all five ready to mark complete, and this plan's own `update_requirements` step does so.

Verified this session:
- `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` -- **1703 tests, OK (skipped=1)**.
- `pre-commit run --all-files` -- all hooks pass, including the Sphinx documentation build and the project's own `Run unit tests` hook.
- A live query against the developer database confirms zero `CalendarEvent` rows carry a legacy bracket-word status prefix.

No blockers. Phase 37 (Status Vocabulary, Public Tallies & Provenance-Blind Gaps) is complete.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED

- `docs/runbooks/telescope_runs_calendar.rst` -- FOUND
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` -- FOUND
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` -- FOUND
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` -- FOUND
- `solsys_code/status_vocabulary.py` -- FOUND
- `solsys_code/tests/test_status_vocabulary.py` -- FOUND
- `solsys_code/tests/test_views.py` -- FOUND
- `solsys_code/tests/test_calendar_display_extras.py` -- FOUND
- `solsys_code/tests/test_calendar_template.py` -- FOUND
- Commit `49ef5b5` -- FOUND
- Commit `df40929` -- FOUND
- Commit `1850322` -- FOUND
- All plan-level task `<acceptance_criteria>` re-verified true (per-task automated checks re-run above)
- Plan-level `<verification>` fully re-confirmed: `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` exits 0 (1703 tests, OK); `pre-commit run --all-files` exits 0 including the Sphinx docs build; every code cell in the four pre-executed notebooks carries a non-null execution count; a `CalendarEvent` query for legacy bracket-word status prefixes returns 0 rows
