---
status: complete
phase: 38-sync-with-main
source: [38-VERIFICATION.md]
started: 2026-10-08T00:40:00Z
updated: 2026-10-08T03:06:33.152Z
---

## Current Test

[testing complete]

## Tests

### 1. WR-01 decision — where local_settings.py lives
The merged settings import `fomo.local_settings` (branch commit c0f883d, deliberate per 36-REVIEW WR-32); main imports top-level `local_settings`. Neither docs/installation.rst nor the PR #43 body tells a host set up for main that the file must now live at src/fomo/local_settings.py, and a missing file falls back silently to dev defaults. (Carried unchanged from the first verification; src/fomo/settings.py not modified since.)
expected: Developer chooses: (a) document the location in docs/installation.rst and 38-PR43-BODY.md / PR #43, or (b) accept both locations during the transition. Either is a follow-up before PR #43 leaves draft.
result: issue
reported: "I think `local_settings.py` is supposed to be in src/fomo/ alongside `settings.py` - not sure why it's not in `main`"
severity: minor
decision: "(a) — src/fomo/local_settings.py is the canonical location; main's bare `from local_settings import *` only ever found it via sys.path (repo root), the branch's `from fomo.local_settings import *` pins it. Follow-up: document the location in docs/installation.rst and the PR #43 body."

### 2. WR-02 decision — no CI job runs TestEphemeris
No CI job runs `TestEphemeris` (tagged `ephemeris_segfault`); main's unit-test matrix ran it. Locked decision D-05 set the CI form `--exclude-tag functional --exclude-tag ephemeris_segfault`, and the phase implemented it exactly (the CI log on 846be34 shows that command in all three build jobs). (Carried unchanged; .github/workflows not modified since the first verification.)
expected: Developer decides whether to add a separate (possibly non-blocking) `python manage.py test --tag ephemeris_segfault` step, or accept the coverage loss as the cost of D-05.
result: pass
decision: "Accept the coverage loss — TestEphemeris segfaults the native ASSIST integrator anyway, so a CI step running it would not produce a usable signal. No separate step added."

### 3. Judgment-tier prohibitions — confirm from the session
Non-authoritative LLM verdicts flagged unverified-prohibition: (1) nothing installed before the developer confirmed the packages at 38-01 Task 2; (2) developer DB not migrated before a verified backup and never with the cron line active; (3) no live LCO/SOAR portal call, real email or heartbeat ping from an executor run; (4) downloaded tomtoolkit wheels only unzipped and diffed; (5) primary checkout never switched to issue37-code-only (38-04 and 38-06).
expected: Developer confirms each from memory of the session. Verifier evidence: SUMMARYs record `approve` and `publish` verbatim; backup src/fomo_db_20261007_pre_phase38.sqlite3 exists (mtime 11:10 PDT, after the 10:19 merge); crontab has no PHASE38-PAUSED line; HEAD's reflog for 2026-10-07 holds only commit entries (no checkout, switch, reset, rebase or amend), including through the 38-06 publish window.
result: pass
note: "Developer confirmed all five prohibitions from the session. On-disk evidence re-checked at UAT time: backup present (2026-10-07 11:10), crontab has 0 PHASE38 lines, every 2026-10-07 HEAD reflog entry is a plain commit (plus the 10:19 commit (merge))."

### 4. WR-01 (2026-10-08 review) — keep or fix the PR #58 sentence in the installation warning
expected: PR #58 is still OPEN and origin/main still has the bare `from local_settings import *`, so the sentence added by 38-07 (`a060f9d`) is false today, though no must-have asserts it and the commits are unpushed. Either accept it as forward-looking (override) or make a one-line docs fix before the next push or snapshot refresh. Source: 38-VERIFICATION.md (2026-10-08, human item 1), 38-REVIEW.md WR-01/WR-02.
result: pass
decision: "pass (the first option) — keep the sentence as a forward-looking statement; override recorded. PR #58 (production-deploy → main) standardises src/fomo/local_settings.py on main; until it merges, main still has the bare import. Advisory WR-02 (keep the absolute `fomo.local_settings` import when PR #58's settings.py conflict is resolved) is recorded in STATE.md Operator Next Steps."

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-38-1
  truth: "docs/installation.rst and the PR #43 body (38-PR43-BODY.md) state that local_settings.py lives at src/fomo/local_settings.py, so a host set up for main (file at the repo root, found via the bare `from local_settings import *`) is told to move it before PR #43 merges"
  status: resolved
  resolved_by: 38-07-PLAN.md
  resolved_at: 2026-10-07
  reason: "User reported: I think `local_settings.py` is supposed to be in src/fomo/ alongside `settings.py` - not sure why it's not in `main`. Decision (a): src/fomo/ is canonical; the documentation follow-up WR-01 names is still outstanding."
  severity: minor
  test: 1
  root_cause: "Branch commit c0f883d (36-REVIEW WR-32) changed src/fomo/settings.py from main's bare `from local_settings import *` (resolved via sys.path, i.e. wherever `python manage.py` is run from — the repo root on hosts set up for main) to `from fomo.local_settings import *`, which only resolves src/fomo/local_settings.py. The two documents a deploying host reads were not updated to say so: docs/installation.rst:115 names `local_settings.py` with no path, and 38-PR43-BODY.md:45 (the PR #43 'Settings' checklist line) lists the new settings a deployment must mirror but not that the file's required location changed. A host that keeps the file at the repo root gets no error — the ImportError guard swallows exactly the missing-module case — and silently runs every dev default."
  artifacts:
    - path: "docs/installation.rst"
      issue: "line 115 says 'in this host's local_settings.py' with no path; nothing in the Installation guide states the file lives at src/fomo/local_settings.py alongside settings.py"
    - path: ".planning/phases/38-sync-with-main/38-PR43-BODY.md"
      issue: "line 45 (Settings checklist) calls out new settings to mirror but not the import-path change from `local_settings` to `fomo.local_settings`; the live PR #43 body (draft) carries the same omission"
  missing:
    - "docs/installation.rst: state that production overrides go in src/fomo/local_settings.py (next to settings.py, gitignored), and that FOMO imports it as fomo.local_settings so a file at the repo root is no longer found"
    - "38-PR43-BODY.md Settings line: add that the local_settings import moved to fomo.local_settings (c0f883d), so a host upgraded from main must move its file from the repo root to src/fomo/"
    - "Mirror the 38-PR43-BODY.md change onto the live PR #43 body (externally visible — confirm with the developer before editing the PR)"
  debug_session: ".planning/debug/local-settings-location-undocumented.md"
