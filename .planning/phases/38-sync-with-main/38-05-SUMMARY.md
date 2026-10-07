---
phase: 38-sync-with-main
plan: 05
subsystem: infra
tags: [urls, tomtoolkit-3.1, gap-closure, cr-01, regression-test, tdd]

requires:
  - phase: 38-01
    provides: "merge of origin/main into the v2.5 branch, whose resolution kept the alerts/ include (CR-01)"
  - phase: 38-04
    provides: "refreshed draft PR #43 (not touched here; the PR re-snapshot is 38-06)"
provides:
  - "src/fomo/urls.py without the alerts/ include and its three-line comment, matching main's ada2000"
  - "solsys_code/tests/test_urls.py: regression guard that /alerts/query/list/ is a 404, plus a guard that the project-level shadow routes still win over tom_common.urls"
  - "planning docs (38-01-PLAN, 38-RESEARCH, 38-PATTERNS) no longer call alerts/ a branch route"
  - "full suite green on the corrected tree: 2182 tests, OK, no skips"
affects: [38-06]

actuals:
  tokens: 40000
  tasks: 2
  commits: 3
plan_head_before: 99e4975142efee5fd1ef4cd1777a54d73032b000
plan_head_after: db3ae7c14676c94eae5a3ef0e4af1d55d18e3ae5

tech-stack:
  added: []
  patterns: ["untagged URLconf regression test using resolve()/reverse() plus a logged-in Client(raise_request_exception=False) GET, so it runs in every suite invocation"]

key-files:
  created:
    - solsys_code/tests/test_urls.py
    - .planning/phases/38-sync-with-main/38-05-red-evidence.json
  modified:
    - src/fomo/urls.py
    - .planning/phases/38-sync-with-main/38-01-PLAN.md
    - .planning/phases/38-sync-with-main/38-RESEARCH.md
    - .planning/phases/38-sync-with-main/38-PATTERNS.md

key-decisions:
  - "Took main's side for the alerts/ include (deleted the four lines) rather than re-adding tom_alerts to INSTALLED_APPS; D-03 (keep both sides) is not reopened because the include was merge-base content the branch never changed and main deliberately deleted."
  - "Corrected the same wrong claim in 38-PATTERNS.md and in 38-01's Task 1 action bullet and verify key list (assumptions G4 and G5), each marked 'corrected by 38-05'."

requirements-completed: [SYNC-04, SYNC-07]

duration: 21min
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 05: Drop the alerts/ route main removed (CR-01) Summary

**The four-line `alerts/` include is gone from `src/fomo/urls.py` as in main's ada2000, a regression test written first (RED, then GREEN) proves `/alerts/query/list/` is a 404, and the full suite is green at 2182 tests.**

## Performance

- **Duration:** about 21 min wall clock (the full-suite run took 461 s of that)
- **Started:** 2026-10-07T21:50Z
- **Completed:** 2026-10-07T22:11Z
- **Tasks:** 2 (Task 1 tracer, TDD; Task 2 auto)
- **Files modified:** 6 (two code files, one evidence record, three planning docs)

## Accomplishments

- **RED.** `python manage.py test --noinput solsys_code.tests.test_urls` against the unfixed `urls.py` ran 4 tests and printed `FAILED (failures=3)`. The three `FAIL:` headers were `test_alerts_namespace_cannot_be_reversed` (`NoReverseMatch not raised`), `test_alerts_path_does_not_resolve` (`Resolver404 not raised`) and `test_logged_in_get_of_alerts_list_returns_404` (`500 != 404`). The guard test passed and there was no `ERROR:` header. `gsd_run check tdd-red-evidence 38-05-red-evidence.json` returned `RED_EVIDENCE_OK` (`target_test_failed`, "GREEN authorized"). Semantic assessment: the target test executed and failed on its planned assertion because the include was still in `urls.py`; no load, import or zero-test fault.
- **GREEN.** Deleting the three comment lines and the `path('alerts/', include('tom_alerts.urls', namespace='alerts'))` line made `Ran 4 tests` / `OK`. The fix commit's `git diff --numstat` for `src/fomo/urls.py` is `0	4`. The route-set check passed: the route strings are exactly origin/main's plus `calendar/`, `campaigns/` and `users/<int:pk>/delete/`, no duplicate, all before `include('tom_common.urls')`, and the file has no `tom_alerts` text.
- **Boot and neighbours.** `python manage.py check` printed only `urls.W005` (namespace `calendar` not unique, the expected benign warning). `TestUserDeleteView` plus `test_scout_views` ran 19 tests, OK. `pre-commit run ruff` and `ruff-format` passed on both files (hook-pinned ruff 0.16.9); neither rewrote `test_urls.py`.
- **Full suite (SYNC-07).** `python manage.py test --noinput -v 2 solsys_code --exclude-tag=ephemeris_segfault`: `Ran 2182 tests in 461.288s`, then an exact `OK`, no `FAILED`, no `skipped=`. All four test_urls tests are listed in the log with `... ok`. The known flaky Playwright test did not fail.
- **Docs.** The `alerts/` branch-route wording is corrected in 38-01-PLAN.md (the SYNC-04/ordering truth, the `provides:` line, the Task 1 urls.py bullet and the verify key list), the urls.py row of 38-RESEARCH.md, and the URL list in 38-PATTERNS.md. Docs commit numstat: `4 4` 38-01-PLAN.md, `1 1` 38-RESEARCH.md, `1 1` 38-PATTERNS.md, `10 0` 38-05-red-evidence.json (new file). 38-01's frontmatter still passes `frontmatter.validate --schema plan` and parses with yaml.

## Task Commits

1. **Task 1 RED:** `32dafa2` test(38-05): add failing regression test for the alerts/ route main removed (CR-01)
2. **Task 1 GREEN:** `a4d77f2` fix(38-05): drop the alerts/ route main removed for tomtoolkit 3.1.0 (CR-01)
3. **Task 2:** `db3ae7c` docs(38-05): correct the alerts/ wording in 38-01, RESEARCH and PATTERNS (CR-01)

No refactor commit was needed. Nothing was pushed: this plan publishes nothing (38-06 does, behind its own checkpoint). No notebook and no runbook changed (neither `src/fomo/urls.py` nor `solsys_code/tests/` is in CLAUDE.md's notebook map, and no page under `docs/runbooks/` documents the alerts route).

## TDD Gate Compliance

RED (`32dafa2`, `test(38-05)`) precedes GREEN (`a4d77f2`, `fix(38-05)`), and the RED evidence was classified `RED_EVIDENCE_OK` before the fix. The plan specifies a `fix(...)` subject rather than `feat(...)` for the GREEN commit, since the change deletes a wrongly-restored route; the gate sequence is otherwise intact.

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

None.

## Issues Encountered

None. One tooling note: a `pgrep -f` poll for the background suite matched its own command line and reported the run as still going after it had finished; the log itself showed the completed run.

## Known Stubs

None.

## Threat Flags

None. The change removes a route (T-38-22 mitigated) and adds a guard against its return (T-38-23); the route-set check and numstat `0 4` cover T-38-24; the docs commit replaced lines one for one in the planned files only (T-38-25).

## Next Phase Readiness

Gap items 1, 2 and 4 are closed. Gap item 3 (the PR #43 re-snapshot and publish) is 38-06's, behind its developer checkpoint. The three commits are local only.

## Self-Check: PASSED

- FOUND: solsys_code/tests/test_urls.py, .planning/phases/38-sync-with-main/38-05-red-evidence.json
- FOUND commits: 32dafa2, a4d77f2, db3ae7c (all ancestors of HEAD)
- All task verify blocks and acceptance criteria re-run and passing
