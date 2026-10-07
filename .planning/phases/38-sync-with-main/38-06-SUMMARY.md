---
phase: 38-sync-with-main
plan: 06
subsystem: infra
tags: [pull-request, issue37-code-only, git-worktree, github-actions, django-test-runner, gap-closure]

requires:
  - phase: 38-04
    provides: "draft PR #43 with head issue37-code-only (1a68a76) and the D-11 snapshot recipe"
  - phase: 38-05
    provides: "src/fomo/urls.py without the alerts/ route and the solsys_code/tests/test_urls.py regression test on the v2.5 branch"
provides:
  - "PR #43's head branch issue37-code-only no longer carries the alerts/ route main removed (gap item 3 closed)"
  - "origin/issue37-telescope-runs-calendar carries 38-05's commits and the Phase 38 review/verification docs (plain fast-forward)"
  - "CI proof on the new snapshot: all three pull_request workflows green on Python 3.10-3.12, Django runner with coverage, no pytest job"
affects: [39, 40, 41, 42]

actuals:
  tokens: 25000
  tasks: 3
  commits: 1   # the one snapshot commit 846be34 on issue37-code-only; the primary checkout gained no code commit before this SUMMARY
plan_head_before: 1a68a76415bbb31f926c59334f650baf2e42f28a
plan_head_after: 846be3469427293bc8b9aa36c0352f488eebe468

tech-stack:
  added: []
  patterns: ["code-only PR branch re-snapshotted in a separate git worktree (read-tree of the v2.5 branch tree minus .planning/), plain fast-forward push, CI proof from the PR push"]

key-files:
  created: []
  modified: []

key-decisions:
  - "Developer's Task 2 answer: publish"
  - "No new merge commit was needed (R1): origin/main (a910c17) is already an ancestor of issue37-code-only through 8ef5445"
  - "PR #43 body left untouched (R2): no gh pr edit, ready-for-review or merge command ran"

requirements-completed: [SYNC-04, SYNC-05, SYNC-08]

duration: about 55min including the Task 1 build and the CI wait (the Task 2 checkpoint wait excluded)
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 06: Re-snapshot PR #43 without the alerts/ route Summary

**issue37-code-only gained one snapshot commit (846be34) that drops the alerts/ route main removed and adds the test_urls.py regression test; both branches went out as plain fast-forwards, draft PR #43 is untouched and still a draft, and its CI is green on Python 3.10-3.12.**

## Developer's Task 2 answer

Developer's Task 2 answer: publish

At 2026-10-07T23:07Z `git ls-remote origin` still printed the two recorded pre-push tips, so nothing moved while the checkpoint waited.

## Accomplishments

- One snapshot commit on issue37-code-only: `846be3469427293bc8b9aa36c0352f488eebe468`, parent `1a68a76415bbb31f926c59334f650baf2e42f28a` (the pre-push PR head). Subject: "Sync code-only branch with issue37-telescope-runs-calendar: drop the alerts/ route main removed (v2.5 Phase 38 gap closure)". Its tree equals issue37-telescope-runs-calendar's outside `.planning/`.
- Two-file diff against the old PR head (`git diff --name-only 1a68a76 846be34`): `solsys_code/tests/test_urls.py` and `src/fomo/urls.py`, nothing else.
- No new merge commit (R1): origin/main (a910c178be2e6e8063f8a262b51934ca05cdbb01) is still the tip of main and already an ancestor of issue37-code-only (merged by 8ef5445 in 38-04).
- The published `src/fomo/urls.py` has no `tom_alerts` reference, and `git diff origin/main...origin/issue37-code-only -- src/fomo/urls.py` adds no `tom_alerts` line, so PR #43 no longer re-adds what main's ada2000 removed (SYNC-04).
- Leak check on the snapshot: no `.planning/`, `*.sqlite3`, `local_settings.py` or `reqgroup_*.json` among its tracked files.
- Three-dot diff, `git diff --shortstat origin/main...origin/issue37-code-only`: **177 files changed, 88517 insertions(+), 74 deletions(-)** (38-04 recorded 176 files, 88465 insertions; the extra file and 52 lines are test_urls.py).
- Temporary worktree `/home/tlister/git/fomo_code_only` removed; `git worktree list` shows only the primary checkout, still on issue37-telescope-runs-calendar.

## Pushed tips

| Branch | Before | After |
|--------|--------|-------|
| origin/issue37-telescope-runs-calendar | 62605b8cbb889e9693f4acc78e18947253df148d | 49be149 (`docs(38-05): complete alerts/ route removal gap-closure plan`); 17 commits pushed, first `64c0b7e docs(38-04): complete refresh of draft PR #43 plan`, last `49be149` |
| origin/issue37-code-only | 1a68a76415bbb31f926c59334f650baf2e42f28a | 846be3469427293bc8b9aa36c0352f488eebe468 (1 commit pushed) |

Both pushes were plain `git push origin <branch>` fast-forwards (`62605b8..49be149` and `1a68a76..846be34`); no `--force`, no `--force-with-lease`, no rewrite of existing history. Both pre-push tips are ancestors of the pushed tips, and the origin tips equal the local branches.

## PR and CI evidence

- PR: https://github.com/lsst-sssc/fomo/pull/43 (state OPEN, `isDraft` true, head issue37-code-only, base main). Its body still carries the D-12 sections (`### Observation projector`, `### Allocation layer`, `### Unattended operation`, `### Public tallies`, `### How to try it`), the runbook path `docs/runbooks/telescope_runs_calendar.rst` and the draft line. Only read-only `gh pr view` was used; no PR-editing command ran.
- Snapshot SHA: 846be3469427293bc8b9aa36c0352f488eebe468

| Workflow | Run | Conclusion | Jobs |
|----------|-----|------------|------|
| Unit test and code coverage | https://github.com/lsst-sssc/fomo/actions/runs/37700624248 | success | build (3.10), build (3.11), build (3.12) (step `Run Django unit tests with coverage`), functional-tests |
| Run pre-commit hooks | https://github.com/lsst-sssc/fomo/actions/runs/37700624234 | success | pre-commit-ci |
| Build documentation | https://github.com/lsst-sssc/fomo/actions/runs/37700624246 | success | build |

No job or step name mentions pytest. This was the new test_urls.py module's first run on Python 3.10 and 3.12 (R4): it passed.

## Task Commits

1. **Task 1: build the issue37-code-only re-snapshot** - `846be34` on issue37-code-only, made in the separate worktree by the previous executor.
2. **Task 2: developer approves publishing** - checkpoint, answer "publish" (no commit).
3. **Task 3: push both branches, confirm the draft PR, prove CI, remove the worktree** - no new repo commit (two `git push` commands, read-only `gh pr view`, `gh run list`/`watch`, `git worktree remove`).

**Plan metadata:** the docs commit for this SUMMARY and the tracking files follows.

## Files Created/Modified

None in the primary checkout. The published change is `src/fomo/urls.py` and `solsys_code/tests/test_urls.py` on issue37-code-only, both authored in 38-05.

## Decisions Made

- Publish approved by the developer; OQ1 (the PR push's CI run is the CI proof), R1 (no new merge), R2 (PR body unedited), R3 (both branches pushed) and R4 (first CI run of the new module on 3.10/3.12) all held.
- No paired notebook or runbook change: this plan changes no module behavior and no runbook text.

## Deviations from Plan

None - plan executed exactly as written. (The `commits:` count above is 1 because the plan's only commit is the snapshot on issue37-code-only; the primary checkout's HEAD stayed at 49be149 until this SUMMARY.)

## Issues Encountered

None. The unit-test matrix took roughly 25 minutes; pre-commit and the docs build finished first.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Threat Flags

None - no new network endpoints, auth paths or schema changes; publishing was gated by the leak check (T-38-26), plain pushes only after "publish" (T-38-27), no PR-editing command (T-38-28), and a branch check before every branch-implicit command (T-38-29, T-38-30).

## Known Stubs

None.

## Next Phase Readiness

All six Phase 38 plans are complete: the v2.5 branch and PR #43's head both carry everything main has, with the alerts/ route removed as main does, the PR is still a draft with its D-12 body, and CI is green. Remaining v2.5 phases (39-42) land on issue37-telescope-runs-calendar and will need another snapshot refresh before the PR is updated.

## Self-Check: PASSED

- All four Task 3 verify blocks printed their OK lines.
- FOUND: origin/issue37-code-only = 846be3469427293bc8b9aa36c0352f488eebe468, parent 1a68a76; origin/issue37-telescope-runs-calendar = 49be149.
- Worktree removed; primary checkout on issue37-telescope-runs-calendar.

---
*Phase: 38-sync-with-main*
*Completed: 2026-10-07*
