---
phase: 38-sync-with-main
plan: 04
subsystem: infra
tags: [pull-request, issue37-code-only, git-worktree, github-actions, django-test-runner]

requires:
  - phase: 38-01
    provides: "merge of origin/main into the v2.5 branch (e12158c)"
  - phase: 38-02
    provides: "ruff 0.16.9-clean tree and the Django-runner pre-commit hook"
  - phase: 38-03
    provides: "full suite green on the merged tree; fresh-install proof"
provides:
  - "draft PR #43 refreshed: head issue37-code-only is v2.4 on top of current main, body rewritten per D-12, still a draft"
  - "origin/issue37-telescope-runs-calendar carries all Phase 38 commits (plain fast-forward push)"
  - "CI proof (SC#4, SYNC-05): the PR push ran the Django test runner with coverage; all three workflows green; no pytest job"
affects: [39, 40, 41, 42]

actuals:
  tokens: 30000
  tasks: 3
  commits: 1
plan_head_before: b99566463166b4fd0e65d289da81206bc9fcbe02
plan_head_after: 62605b8cbb889e9693f4acc78e18947253df148d

tech-stack:
  added: []
  patterns: ["code-only PR branch refreshed in a separate git worktree (merge -s ours origin/main, then a read-tree snapshot minus .planning/), pushed as a plain fast-forward", "a push to the PR head branch (a pull_request event to main) is the only CI trigger, so it is the CI proof"]

key-files:
  created: [.planning/phases/38-sync-with-main/38-PR43-BODY.md]
  modified: []

key-decisions:
  - "Developer's Task 2 answer: publish"
  - "Published 372d02c (the previously unpushed v2.4 snapshot) together with the merge and the new snapshot, rather than resetting it away (OQ2): three new commits on issue37-code-only, no history rewritten"
  - "The origin/main merge on issue37-code-only used git merge -s ours (A3); the snapshot commit that follows replaces the tree wholesale, so the final tree equals the v2.5 tree outside .planning/"

requirements-completed: [SYNC-08, SYNC-05]

duration: 32min
completed: 2026-10-07
status: complete
---

# Phase 38 Plan 04: Refresh draft PR #43 Summary

**Draft PR #43 now shows v2.4 on top of current main: issue37-code-only gained a main merge plus a tree snapshot (plain fast-forward push), the body was rewritten per D-12, the PR is still a draft, and its CI run proves the Django test runner with coverage replaced pytest.**

## Performance

- **Duration:** about 32 min wall clock (including the CI wait; the Task 2 checkpoint wait is excluded from the work)
- **Started:** 2026-10-07T18:31Z (Task 1)
- **Completed:** 2026-10-07T19:05Z
- **Tasks:** 3 (Task 2 was the checkpoint)
- **Files modified:** 1 in the repo (`38-PR43-BODY.md`); 2 branches pushed

## Developer's Task 2 answer

Developer's Task 2 answer: publish

## Accomplishments

- issue37-code-only (worktree, primary checkout never left the v2.5 branch) = the PR's four original commits + three new ones; tree equals issue37-telescope-runs-calendar outside `.planning/`; origin/main is an ancestor.
- Both branches pushed as plain fast-forwards; origin tips equal the local branches; 75ad2be and 5a1f27e are ancestors of the new tips.
- PR #43 body replaced with 38-PR43-BODY.md (one section per v2.4 pillar, runbook link, how-to-try-it, draft line). `isDraft` true, head issue37-code-only, base main.
- CI on the snapshot commit: all three workflows succeeded; the unit-test workflow ran `build (3.10)`, `build (3.11)`, `build (3.12)` and `functional-tests`, with the step `Run Django unit tests with coverage`; no job or step mentions pytest.
- Temporary worktree `../fomo_code_only` removed.

## Pushed tips

| Branch | Before | After |
|--------|--------|-------|
| origin/issue37-telescope-runs-calendar | 75ad2be1a6c5f5fc9d24f5ce7194da451df22148 | 62605b8cbb889e9693f4acc78e18947253df148d (87 commits pushed, all Phase 38 work) |
| origin/issue37-code-only | 5a1f27ea16962586ac0fde1fed3f35e3584bb115 | 1a68a76415bbb31f926c59334f650baf2e42f28a |

Commits published on issue37-code-only (on top of the PR's four):

1. `372d02c` Sync code-only branch with issue37-telescope-runs-calendar through v2.4 (previously unpushed snapshot, OQ2)
2. `8ef5445` Merge origin/main into issue37-code-only (tree superseded by the snapshot commit that follows) -- parents 372d02c and a910c17 (origin/main)
3. `1a68a76` Sync code-only branch with issue37-telescope-runs-calendar through v2.5 Phase 38 (merged with main)

Three-dot diff, `git diff --shortstat origin/main...origin/issue37-code-only`: **176 files changed, 88465 insertions(+), 74 deletions(-)**. (RESEARCH's dry run said 174 files; the two extra files are from main's later commits and this plan's snapshot.) Leak check: 260 tracked files, no `.planning/`, `*.sqlite3`, `local_settings.py` or `reqgroup_*.json`.

## PR and CI evidence

- PR: https://github.com/lsst-sssc/fomo/pull/43 (draft, head issue37-code-only, base main, state OPEN)
- Snapshot SHA: 1a68a76415bbb31f926c59334f650baf2e42f28a

| Workflow | Run | Conclusion | Jobs |
|----------|-----|------------|------|
| Unit test and code coverage | https://github.com/lsst-sssc/fomo/actions/runs/37668777675 | success | functional-tests (step `Run Playwright functional tests`), build (3.10), build (3.11), build (3.12) (step `Run Django unit tests with coverage`) |
| Run pre-commit hooks | https://github.com/lsst-sssc/fomo/actions/runs/37668777601 | success | pre-commit-ci |
| Build documentation | https://github.com/lsst-sssc/fomo/actions/runs/37668777472 | success | build |

## Task Commits

1. **Task 1: build the issue37-code-only refresh and draft the PR body** - `62605b8` on issue37-telescope-runs-calendar (38-PR43-BODY.md); refresh commits `8ef5445` and `1a68a76` on issue37-code-only (in the worktree)
2. **Task 2: developer approves publishing** - checkpoint, answer "publish" (no commit)
3. **Task 3: push, rewrite the PR body, prove CI, remove the worktree** - no repo commit (git push, `gh pr edit 43 --body-file`, `git worktree remove`); evidence recorded above

**Plan metadata:** the docs commit for this SUMMARY and the tracking files follows.

## Files Created/Modified

- `.planning/phases/38-sync-with-main/38-PR43-BODY.md` - the D-12 PR body, kept for review of the published text

## Decisions Made

- Publish approved by the developer; OQ1 (the PR push's CI run is SC#4's proof), OQ2 (keep 372d02c), A3 (`merge -s ours`) and A4 (CI not run locally) all stood. A4's risk did not materialise: CI on Python 3.10/3.11/3.12, the docs build and pre-commit all passed first time.
- No paired notebook or runbook change: this plan changes no module behavior and no runbook text.

## Deviations from Plan

None - plan executed exactly as written. (One harmless observation: the pre-push count of v2.5 commits not on origin was 87.)

## Issues Encountered

None. The first `gh run list` after the push already showed all three queued runs for the snapshot SHA.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Threat Flags

None - no new network endpoints, auth paths or schema changes; publishing was gated by the leak check (T-38-17), plain pushes only (T-38-18), body-only PR edit (T-38-19), and a branch check before every branch-implicit command (T-38-20).

## Known Stubs

None.

## Next Phase Readiness

Phase 38 plans are all complete: PR #43 is a draft showing v2.4 on current main with green CI. Remaining v2.5 phases (39-42) land on issue37-telescope-runs-calendar; the PR stays a draft until then and will need another snapshot refresh each time it is updated.

## Self-Check: PASSED

- FOUND: .planning/phases/38-sync-with-main/38-PR43-BODY.md
- FOUND commits: 62605b8 (primary); 8ef5445, 1a68a76 and 372d02c are ancestors of origin/issue37-code-only
- origin tips equal local branches; worktree removed; primary on issue37-telescope-runs-calendar

---
*Phase: 38-sync-with-main*
*Completed: 2026-10-07*
