---
status: resolved
trigger: "UAT G-38-1 (Test 1, WR-01): I think `local_settings.py` is supposed to be in src/fomo/ alongside `settings.py` - not sure why it's not in `main`"
created: 2026-10-08T00:40:00Z
updated: 2026-10-08T00:50:00Z
---

## Current Focus

hypothesis: confirmed — see Resolution
test: n/a
expecting: n/a
next_action: hand off to /gsd-verify-work plan_gap_closure (goal was find_root_cause_only)
bug_class: bohrbug
reasoning_checkpoint: null
tdd_checkpoint: null

## Symptoms

expected: docs/installation.rst and the PR #43 body tell a deploying host where local_settings.py must live, so a host set up for main is not silently reverted to dev defaults after PR #43 merges
actual: docs/installation.rst:115 says "in this host's ``local_settings.py``" with no path; 38-PR43-BODY.md:45 lists new settings to mirror but not that the file's required location changed; the live PR #43 (draft) body matches 38-PR43-BODY.md
errors: None reported — the failure mode is silent (the ImportError guard in src/fomo/settings.py:414-423 swallows exactly the missing-module case)
reproduction: Test 1 in 38-UAT.md (carried from 38-VERIFICATION.md WR-01 / 38-REVIEW-DISPOSITION.md)
started: branch commit c0f883d ("refactor: import local_settings from the fomo package", 36-REVIEW WR-32); main still imports the bare module

## Eliminated

- hypothesis: main deliberately placed local_settings.py at the repo root
  evidence: main's `from local_settings import *` (origin/main src/fomo/settings.py:366) is a bare import resolved via sys.path; it never pinned a location — the repo root only worked because it is the cwd for `python manage.py`
  timestamp: 2026-10-08T00:45:00Z
- hypothesis: the notebooks/runbook already document the location well enough
  evidence: docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb does say `src/fomo/local_settings.py`, but docs/installation.rst (the page a new host follows) and docs/runbooks/telescope_runs_calendar.rst:1988 both name the file path-less, and the PR body says nothing about the move
  timestamp: 2026-10-08T00:46:00Z

## Evidence

- timestamp: 2026-10-08T00:42:00Z
  checked: `git show origin/main:src/fomo/settings.py | grep -n local_settings`
  found: line 366 `from local_settings import *  # noqa` inside a bare try/except ImportError: pass
  implication: on main the file is found anywhere on sys.path; repo root works only because of cwd
- timestamp: 2026-10-08T00:42:00Z
  checked: branch src/fomo/settings.py
  found: line 415 `from fomo.local_settings import *`; guard at 416-423 re-raises unless `exc.name == 'fomo.local_settings'`
  implication: only src/fomo/local_settings.py resolves; a repo-root file is ignored with no error
- timestamp: 2026-10-08T00:43:00Z
  checked: `git show --stat c0f883d`
  found: "refactor: import local_settings from the fomo package — keeps local_settings.py alongside settings.py instead of requiring it at the src/ path root"; only src/fomo/settings.py changed
  implication: the location change was deliberate (36-REVIEW WR-32) but shipped with no docs or PR-body update
- timestamp: 2026-10-08T00:44:00Z
  checked: `grep -rn local_settings docs/ .planning/phases/38-sync-with-main/38-PR43-BODY.md`
  found: installation.rst:115 path-less; runbook :1988 path-less; PR body :45 path-less and silent on the move; only campaign_lifecycle_demo.ipynb says src/fomo/
  implication: the two documents WR-01 names are the gap; the developer's own src/fomo/local_settings.py exists and is gitignored (.gitignore:61), so the local checkout is unaffected
- timestamp: 2026-10-08T00:47:00Z
  checked: developer decision (UAT Test 1)
  found: option (a) — src/fomo/local_settings.py is canonical; document it rather than accept both locations
  implication: fix is documentation only; no settings.py change

## Resolution

root_cause: c0f883d moved the import from the bare `local_settings` to `fomo.local_settings` (pinning the file to src/fomo/) without updating docs/installation.rst or the PR #43 body; combined with the deliberate swallow of the missing-module ImportError, a host carrying the file at the repo root reverts to every dev default silently.
fix: (not applied — find_root_cause_only) document the canonical location in docs/installation.rst and in 38-PR43-BODY.md's Settings checklist line, then mirror onto the live PR #43 body with the developer's go-ahead
verification: n/a
oracle_type: n/a
files_changed: []
