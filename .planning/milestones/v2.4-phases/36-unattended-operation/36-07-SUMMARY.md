---
phase: 36-unattended-operation
plan: 07
subsystem: docs
tags: [runbook, healthchecks, cron, sphinx, gap-closure]

# Dependency graph
requires:
  - phase: 36-05
    provides: the original "How do I run everything unattended?" runbook section, crontab template and check_unattended preflight
  - phase: 36-06
    provides: the corrected two-knob heartbeat arithmetic (Period/Grace/35-min alert window) living in "The two failure signals"
provides:
  - "A create-and-configure-the-heartbeat-check step (new step 3) standing before the FOMO_HEARTBEAT_URL export in the fresh-host setup procedure, naming the service class, both check settings with values, the alert arithmetic, and the placeholder ping-URL form"
  - "The fresh-host setup procedure split from 7 to 9 numbered steps (FOMO_HEARTBEAT_URL export and FOMO_BASE_URL now separate steps; step 1/2/9 carry the sudo and FACILITIES-key-path clarifications the diagnosis swept up)"
  - "deploy/cron/fomo.crontab.example's ping-URL provenance comment and both of its 'for the full setup' pointers now name the 'Setting it up on a fresh host' subsection instead of resolving to the section alone"
  - "36-VERIFICATION.md's human-test script re-ordered so the SC-5 sufficiency read-through runs before the live heartbeat re-run, with Test 6's contaminated round-1 pass recorded as re-opened by G-36-1 rather than closed"
affects: [37-status-vocabulary-public-tallies-provenance-blind-gaps, ship-decision-for-v2.4]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
# Same estimateTokens scale (chars/4 over the realized diff), never a harness token count.
actuals:
  tokens: 6249
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Slice-scoped automated gate: awk-extract the subsection between two headings and assert both token presence AND line-position ordering inside that slice, rather than a whole-file token-presence probe — this is what actually catches a placement defect like G-36-1"

key-files:
  created: []
  modified:
    - docs/runbooks/telescope_runs_calendar.rst
    - deploy/cron/fomo.crontab.example
    - .planning/phases/36-unattended-operation/36-VERIFICATION.md

key-decisions:
  - "Moved the stale 'Test 6 runbook sufficiency' entry out of re_verification.human_items_closed_by_uat into human_items_still_open with a RE-OPENED-by-G-36-1 marker, rather than leaving it in the closed list with an appended marker — the plan left this choice to the planner/executor, and moving it makes 'no longer closed' unambiguous to a reader scanning list membership rather than reading marker text"
  - "Kept the SC5/truth-35 evidence cells' base verdict as '✓ VERIFIED' with a parenthetical human-item qualifier (matching the exact form SC3 already uses), rather than downgrading the marker — the structural half (9 numbered steps, correct order) is genuinely machine-verified; only sufficiency-at-point-of-use is a human item"

patterns-established:
  - "Cross-reference style for pointing at reasoning kept elsewhere in the same doc: a quoted subsection name plus 'above'/'below' (e.g. '(see \"The two failure signals\" below for why...)'), matching the page's existing '(see \"Adding a proposal to watch\" above)' convention — used here to give the new heartbeat-check step the alert-window arithmetic's values without re-copying its justification"

requirements-completed: [SCHED-08, SCHED-09]

coverage:
  - id: D1
    description: "Fresh-host setup procedure gains a create-and-configure-the-heartbeat-check step before the FOMO_HEARTBEAT_URL export (9 numbered steps total), naming the service class, both settings with values, the alert arithmetic, a cross-reference to the canonical reasoning, and the placeholder ping-URL form"
    requirement: "SCHED-09"
    verification:
      - kind: other
        ref: "task1 automated gate: awk-sliced 'Setting it up on a fresh host'..'Adding a proposal to watch' asserts Period/Grace/*/15 * * * */hc-ping.com/<uuid>/healthchecks.io/cross-reference/35 min/9th-step presence AND that the first Period line and the ping-URL line both precede the export step's 'the environment the cron daemon sees' anchor"
        status: pass
      - kind: other
        ref: "pre-commit run sphinx-build --all-files (docs/runbooks/telescope_runs_calendar.rst, no new warning)"
        status: pass
    human_judgment: true
    rationale: "The plan's own success criterion is sufficiency at point of use for a reader who has not been taught the Period/Grace values out of band — this is exactly the SC-5 sufficiency read-through the plan's single <human-check> requires and that 36-VERIFICATION.md now scripts as human-verification item 1. workflow.human_verify_mode is end-of-phase (default) for this project, so per checkpoints.md this human-check is deferred and harvested into the phase's UAT rather than run as a mid-flight checkpoint; automated presence/order gates prove the content is right and in the right place, but only a naive human reader can prove it is enough."
  - id: D2
    description: "deploy/cron/fomo.crontab.example's FOMO_HEARTBEAT_URL entry states where the ping URL comes from, and both of its 'for the full setup' pointers (~32-35, ~58-60) name 'Setting it up on a fresh host' instead of the section alone; runbook steps 2/1/9 name the nested FACILITIES['LCO']['api_key']/FACILITIES['SOAR']['api_key'] key path and the sudo requirement, without contradicting step 6's run-as-cron-account warning"
    requirement: "SCHED-08"
    verification:
      - kind: other
        ref: "task2 automated gate: grep -cF 'Setting it up on a fresh host' deploy/cron/fomo.crontab.example -ge 3; grep -qF healthchecks; no UUID pattern; slice contains FACILITIES['LCO']['api_key'] / FACILITIES['SOAR']['api_key'] / sudo"
        status: pass
      - kind: unit
        ref: "python manage.py test solsys_code.tests.test_check_unattended (30 tests)"
        status: pass
      - kind: other
        ref: "pre-commit run ruff --all-files; pre-commit run ruff-format --all-files; pre-commit run sphinx-build --all-files"
        status: pass
    human_judgment: false
  - id: D3
    description: "36-VERIFICATION.md's human_verification block (frontmatter + prose) now lists the SC-5 sufficiency read-through before the live heartbeat dead-man re-run, each carrying a stable marker string; Test 6's round-1 pass is recorded as re-opened by G-36-1 rather than closed; SC5 and plan-36-05-truth-35 evidence cells cite 9 numbered steps and no longer rest on Test 6's contaminated pass; no verdict or gaps_closed entry was rewritten"
    requirement: "SCHED-09"
    verification:
      - kind: other
        ref: "task3 automated gate: grep -qF 'G-36-1'/'out of band'/'9 numbered steps'; SC-5 and Live-heartbeat markers each appear >=2 times; first-occurrence and last-occurrence line positions confirm SC-5 precedes Live-heartbeat in both frontmatter and prose; '7 numbered steps' absent from the whole file"
        status: pass
      - kind: other
        ref: "python3 yaml.safe_load() over the file's frontmatter block"
        status: pass
    human_judgment: false

# Metrics
duration: ~25min
completed: 2026-09-18
status: complete
---

# Phase 36 Plan 07: Fresh-Host Heartbeat Setup Step Summary

**Inserted a create-and-configure-the-heartbeat-check step before the FOMO_HEARTBEAT_URL export (closing G-36-1's ordering inversion), split FOMO_BASE_URL into its own step, kept the crontab template's setup pointers on the step that does the work, and re-ordered 36-VERIFICATION.md's human tests so sufficiency is measured before the values are taught.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-09-18
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `docs/runbooks/telescope_runs_calendar.rst`'s "Setting it up on a fresh host" subsection grew from 7 to 9 numbered steps: a new step 3 names the heartbeat check's service class (healthchecks.io hosted free tier or self-hosted), creates one check for the schedule, sets both `Period` (15 min, or Cron `*/15 * * * *`) and `Grace` (~20 min), states the alert-window arithmetic (~35 min), points at "The two failure signals" for the underlying reasoning, and gives the placeholder ping-URL form ``https://hc-ping.com/<uuid>``; step 4 is now the `FOMO_HEARTBEAT_URL` export (hygiene only, tied to the check the new step 3 produced); step 5 carries the `FOMO_BASE_URL` text verbatim as its own step.
- Two same-class siblings closed alongside: step 2 now names the nested `FACILITIES['LCO']['api_key']`/`FACILITIES['SOAR']['api_key']` settings key path (no value); steps 1 and 9 note that creating the root-owned `/var/lock/fomo`/`/var/log/fomo` directories and writing into `/etc/logrotate.d/` typically need `sudo`, without contradicting step 6's existing run-as-the-cron-account warning.
- `deploy/cron/fomo.crontab.example` now says the ping URL is the ping URL of a check the operator creates on a healthchecks-compatible service, and both of its "for the full setup" pointers name the "Setting it up on a fresh host" subsection instead of resolving to the section alone.
- `.planning/phases/36-unattended-operation/36-VERIFICATION.md`'s human-verification block (frontmatter and prose) now lists the SC-5 sufficiency read-through before the live heartbeat dead-man re-run, with Test 6's round-1 pass reclassified as re-opened by G-36-1 (it was contaminated by Test 3 teaching the Period/Grace values earlier in the same session) rather than closed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Give the fresh-host procedure a create-and-configure-the-check step, and split the two-variable step** - `280962b` (docs)
2. **Task 2: Keep the paired artifacts in agreement, and close the two same-class siblings the diagnosis swept up** - `a2f1ee9` (docs)
3. **Task 3: Re-script the SC-5 sufficiency read-through so it runs before the tests that teach the knowledge** - `71cdec2` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `docs/runbooks/telescope_runs_calendar.rst` - Fresh-host setup procedure restructured to 9 numbered steps; new create-and-configure-the-check step; FACILITIES key path and sudo clarifications on steps 1/2/9
- `deploy/cron/fomo.crontab.example` - Ping-URL provenance comment extended; both "for the full setup" pointers now name the subsection that does the work
- `.planning/phases/36-unattended-operation/36-VERIFICATION.md` - Human-test order corrected (SC-5 sufficiency read-through before the live heartbeat re-run); Test 6 reclassified as re-opened; SC5/truth-35 evidence cells updated to the 9-step count

## Decisions Made

- Moved Test 6 out of `human_items_closed_by_uat` into `human_items_still_open` with a RE-OPENED-by-G-36-1 marker, rather than leaving it in the closed list annotated — makes "no longer closed" unambiguous by list membership.
- Kept the SC5/truth-35 status cells' `✓ VERIFIED` marker, adding the same parenthetical human-item qualifier SC3 already uses, rather than downgrading the marker — the structural half (9 steps, correct order) is genuinely machine-verified; only point-of-use sufficiency is a human item.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. (The plan's one `<human-check>` — a live read-through against a real healthchecks-compatible account — is deferred to end-of-phase UAT per `workflow.human_verify_mode: end-of-phase`, and is now scripted as `36-VERIFICATION.md`'s human-verification item 1, ordered before the live heartbeat re-run.)

## Next Phase Readiness

- G-36-1 is closed in the codebase: the fresh-host procedure no longer asks the operator to export a URL that only exists after a check has been created, and the service-class/ping-URL-provenance content that existed only in `.planning/research/STACK.md` is now on the operator-facing page.
- The one remaining proof for this gap is the SC-5 sufficiency read-through itself (a naive reader working the runbook top-down) — scripted in `36-VERIFICATION.md` and pending the phase's end-of-phase UAT pass.
- No source file was touched by this plan; `check_heartbeat()`'s reminder, its pinning test, and the whole `test_check_unattended` module (30 tests) are unchanged and green.

---
*Phase: 36-unattended-operation*
*Completed: 2026-09-18*

## Self-Check: PASSED

- FOUND: docs/runbooks/telescope_runs_calendar.rst
- FOUND: deploy/cron/fomo.crontab.example
- FOUND: .planning/phases/36-unattended-operation/36-VERIFICATION.md
- FOUND commit: 280962b
- FOUND commit: a2f1ee9
- FOUND commit: 71cdec2
- Re-ran plan-level `<verification>` list: presence+order slice checks pass, crontab pointer count (3) passes, no UUID pattern in either committed file, `python manage.py test solsys_code.tests.test_check_unattended` (30 tests) passes, ruff/ruff-format/sphinx-build all pass, `git diff --name-only` confirms zero files under `solsys_code/` changed by this plan, and `36-VERIFICATION.md`'s YAML frontmatter parses cleanly.
