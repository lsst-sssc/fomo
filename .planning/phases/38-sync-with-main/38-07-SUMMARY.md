---
phase: 38-sync-with-main
plan: 07
subsystem: docs
tags: [local-settings, installation-guide, runbook, pull-request, gap-closure]
gap_ids: [G-38-1]

requires:
  - phase: 38-04
    provides: "draft PR #43 and the gh pr edit 43 --body-file mechanism"
  - phase: 38-06
    provides: "the merged tree with settings.py importing fomo.local_settings (c0f883d)"
provides:
  - "docs/installation.rst section `local-settings`: src/fomo/local_settings.py is the location, a file at the repository root or in src/ is silently ignored, how to move and check it"
  - "runbook fresh-host step 2 and the installation FOMO_BASE_URL note name src/fomo/local_settings.py"
  - "PR #43's live body tells an upgrading host to move local_settings.py to src/fomo/"
affects: [39, 40, 41, 42]

actuals:
  tokens: 600
  tasks: 3
  commits: 4
plan_head_before: a003a36c2fea97c78bcdb48f235924d69ddd15be
plan_head_after: 82a097cff31a617bcc406c38e8d532371e373567

tech-stack:
  added: []
  patterns: ["checkpoint-then-apply for an externally visible PR edit: live body compared with the pre-plan file at every stop before gh pr edit runs"]

key-files:
  created: []
  modified:
    - docs/installation.rst
    - docs/runbooks/telescope_runs_calendar.rst
    - .planning/phases/38-sync-with-main/38-PR43-BODY.md

key-decisions:
  - "Developer chose documentation only (decision (a) at UAT Test 1): src/fomo/local_settings.py is canonical, settings.py unchanged"
  - "After the developer's note about production-deploy / PR #58, the wording became branch-neutral and names src/ as the other old location"
  - "New installation section placed before 'Initializing FOMO and the database' because migrate already loads the host's database setting"

requirements-completed: [SYNC-08]

duration: about 80 min elapsed (2026-10-08T01:08Z to 02:28Z), including two checkpoint waits
completed: 2026-10-08
status: complete
---

# Phase 38 Plan 07: Document src/fomo/local_settings.py (G-38-1) Summary

**The installation guide, the runbook's fresh-host step and the live draft PR #43 body now tell a host upgraded from main to move `local_settings.py` to `src/fomo/local_settings.py`, or it is silently ignored. Docs only: `settings.py` is unchanged and nothing was pushed.**

## Performance

- **Started:** 2026-10-08T01:08Z (BASE recorded)
- **Completed:** 2026-10-08T02:28Z (live PR edit read back)
- **Tasks:** 3 (Task 1 tracer, Task 2 decision checkpoint with two rounds, Task 3 apply)
- **Files modified:** 3 repository files

## Accomplishments

- `docs/installation.rst` has a `.. _local-settings:` section, "Host-specific settings (``local_settings.py``)", before "Initializing FOMO and the database". It says the file lives at `src/fomo/local_settings.py`, is gitignored, and is imported as `fomo.local_settings`. Its warning says a copy at the repository root or in `src/` is no longer read and nothing reports it, and gives both `mv` commands, the check command, and the note that `main` reaches the same location through PR #58.
- The FOMO_BASE_URL note in "Starting up the webserver" and runbook fresh-host step 2 (a 1/1 line change) name `src/fomo/local_settings.py` and link `:ref:`local-settings``.
- `38-PR43-BODY.md`: the Settings checklist line (1/1) carries the move; PR #43's live body was replaced with that file and read back equal.
- The documented check command printed a path ending in `/src/fomo/local_settings.py` on the developer's checkout (output in `$HOME/tmp/phase38-07-check.txt`, not copied into any committed file).

## Final Settings line (verbatim, `38-PR43-BODY.md` line 45 and the live PR #43 body)

> - [x] **Settings:** Changes to `src/fomo/settings.py` that a deployment must mirror in `local_settings.py` are called out above. A deployment sets `FOMO_BASE_URL`, `FOMO_HEARTBEAT_URL`, `FOMO_LOCK_DIR`, `FOMO_STATE_DIR` and `FOMO_LOG_FILE` in its environment, and a real `EMAIL_BACKEND` with its host settings in `local_settings.py`. `tom_registration` is gone from `INSTALLED_APPS` and the middleware list, so a `local_settings.py` must not reference it. `settings.py` loads the host settings from `src/fomo/local_settings.py`, next to `settings.py`: this branch imports it as `fomo.local_settings` (c0f883d), and `main` gets the same location through PR #58's relative `.local_settings` import. A host that still keeps the file at the repository root or in `src/` must move it to `src/fomo/` as part of the upgrade: a file left in either place is silently ignored, and the host runs on the development defaults (the committed `SECRET_KEY`, `DEBUG = True`) with no error.

## Task Commits

BASE (HEAD before the plan): `a003a36c2fea97c78bcdb48f235924d69ddd15be`.

1. **Task 1 (tracer): document the location** - `5421abb` (docs: installation section, FOMO_BASE_URL note, runbook step 2) and `cf24051` (docs: PR body file Settings line)
2. **Task 2 (decision, two rounds): revise, then apply** - `a060f9d` (docs: name `src/` as the other old location and main's PR #58 change) and `82a097c` (docs: rewrap one warning line from 121 to 120 columns)
3. **Task 3: apply the body to PR #43** - no repository commit (`gh pr edit`)

**Plan metadata:** the commit for this SUMMARY and the STATE/ROADMAP update follows (its hash is in the git log, not repeated here).

`git rev-list --count a003a36..82a097c` is 4, matching `actuals.commits`.

## Live-body comparisons (assumption L1, threat T-38-34)

| When | Live PR #43 body (CRLF to LF, ends stripped) vs `git show BASE:...38-PR43-BODY.md` | isDraft / head / base |
|------|---------------------------------------------------------------------------|----------------------|
| Task 1 (start) | match | true / issue37-code-only / main |
| Task 2 round 1 checkpoint | match | true / issue37-code-only / main |
| Task 2 round 2 checkpoint (recheck) | match | true / issue37-code-only / main |
| Task 3 precondition (just before the edit) | match | true / issue37-code-only / main |

No edit made on GitHub was overwritten.

## Developer's Task 2 answers (verbatim)

1. Round 1, on the `cf24051` text: "I have just pushed a commit to the `production-deploy` branch, which will go to `main` soon, which standarizes `local_settings.py` in 'src/fomo/local_settings.py' as other Django projects do". The orchestrator found that PR #58 (`production-deploy` to `main`, commit 9511ef4, `from .local_settings import *`) standardises the same location on main, and that hosts set up for main kept the file in `src/` as well as at the repository root. Offered apply / revise / hold, the developer chose: "revise (Recommended)".
2. Round 2, on the `a060f9d` / `82a097c` text, shown with the full new Settings line, both diffs and a fresh live-body check that matched the pre-plan file: "apply".

The revision made the wording branch-neutral (it no longer implies that only this branch moved the file), named `src/` as the other old location next to the repository root, and recorded main's PR #58 change.

## PR-changing command and post-apply state

- Exactly one PR-changing command ran: `gh pr edit 43 --body-file .planning/phases/38-sync-with-main/38-PR43-BODY.md` (exit 0). The body-only PATCH fallback (assumption L3) was not needed. No title, label, reviewer, ready-for-review, merge or close command ran.
- Read back (`$HOME/tmp/phase38-07-pr43-after.json`): isDraft true, head `issue37-code-only`, base `main`, body equal to `38-PR43-BODY.md` after CRLF to LF and end stripping, with `fomo.local_settings`, `src/fomo/local_settings.py` and the D-12 sections present.

## Nothing pushed, settings.py unchanged

- `git ls-remote origin refs/heads/issue37-code-only refs/heads/issue37-telescope-runs-calendar` equals the tips recorded in `$HOME/tmp/phase38-07-tips.txt`. `issue37-code-only` was not re-snapshotted.
- `git diff BASE HEAD -- src/fomo/settings.py` is empty.

## Decisions Made

- Documentation only, as the developer decided at UAT Test 1 (a); the silent fallback in `settings.py` stays.
- The installation section goes before "Initializing FOMO and the database" (plan discretion): `migrate` already loads the host's database setting.
- The PR body's Settings line does not point at the installation page, because PR #43's diff does not carry that section until the next snapshot refresh.

## Deviations from Plan

### Developer-requested revision (Task 2 "revise" round)

**1. [Revise round] Wording changed after the developer's note on `production-deploy` / PR #58**
- **Found during:** Task 2 round 1
- **Change:** `docs/installation.rst` warning and `38-PR43-BODY.md` Settings line now name `src/` as the other old location, give a second `mv` command, and note that main reaches the same place through PR #58. The wording is branch-neutral.
- **Commit:** `a060f9d`

### Auto-fixed Issues

**2. [Rule 1 - Bug] One added warning line was 121 columns, over Task 1's 120-column check**
- **Found during:** re-running Task 1's width check after `a060f9d`
- **Fix:** rewrapped that line
- **Files modified:** `docs/installation.rst`
- **Verification:** all five Task 1 `<verify>` blocks passed on `82a097c`
- **Commit:** `82a097c`

**Total deviations:** 1 developer-requested revision and 1 auto-fix (rule 1). **Impact:** the plan produced four commits instead of two. No scope change.

## Issues Encountered

None beyond the above.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required. (A host that keeps `local_settings.py` outside `src/fomo/` must move it; that is what the documents now say.)

## Known Stubs

None.

## Threat Flags

None. The new text adds no endpoint, auth path or schema change. T-38-31 to T-38-35 mitigations held: the documents match `settings.py`'s import, the PR edit was gated by the blocking-human checkpoint, and no local path or key was added to a committed file.

## Follow-ups (for the orchestrator, not done here)

- The next `issue37-code-only` snapshot refresh (D-11 recipe, as in 38-04 and 38-06) carries the docs commits onto PR #43's diff.
- The WR-01 row in `38-REVIEW-DISPOSITION.md` (currently `open`) can be set to fixed by the review-disposition step.
- CLAUDE.md's Conventions line names `local_settings.py` with no path. PR #58's commit 9511ef4 also rewrites that line on main, so it will arrive at the next sync with main.
- No notebook or other runbook section changed (no Python module changed, so no paired notebook is due).

## Self-Check: PASSED

- FOUND commits (ancestors of HEAD): 5421abb, cf24051, a060f9d, 82a097c
- FOUND files: docs/installation.rst, docs/runbooks/telescope_runs_calendar.rst, .planning/phases/38-sync-with-main/38-PR43-BODY.md
- Task 3 verify: "OK: PR #43 body equals 38-PR43-BODY.md ..." and "OK: settings.py untouched and no branch pushed or re-snapshotted"

---
*Phase: 38-sync-with-main*
*Completed: 2026-10-08*
