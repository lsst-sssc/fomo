---
status: testing
phase: 38-sync-with-main
source: [38-VERIFICATION.md]
started: 2026-10-08T00:40:00Z
updated: 2026-10-08T00:40:00Z
---

## Current Test

number: 1
name: WR-01 decision — where local_settings.py lives
expected: |
  Developer chooses: (a) document the location in docs/installation.rst and 38-PR43-BODY.md / PR #43, or (b) accept both locations during the transition. Either is a follow-up before PR #43 leaves draft.
awaiting: user response

## Tests

### 1. WR-01 decision — where local_settings.py lives
The merged settings import `fomo.local_settings` (branch commit c0f883d, deliberate per 36-REVIEW WR-32); main imports top-level `local_settings`. Neither docs/installation.rst nor the PR #43 body tells a host set up for main that the file must now live at src/fomo/local_settings.py, and a missing file falls back silently to dev defaults. (Carried unchanged from the first verification; src/fomo/settings.py not modified since.)
expected: Developer chooses: (a) document the location in docs/installation.rst and 38-PR43-BODY.md / PR #43, or (b) accept both locations during the transition. Either is a follow-up before PR #43 leaves draft.
result: [pending]

### 2. WR-02 decision — no CI job runs TestEphemeris
No CI job runs `TestEphemeris` (tagged `ephemeris_segfault`); main's unit-test matrix ran it. Locked decision D-05 set the CI form `--exclude-tag functional --exclude-tag ephemeris_segfault`, and the phase implemented it exactly (the CI log on 846be34 shows that command in all three build jobs). (Carried unchanged; .github/workflows not modified since the first verification.)
expected: Developer decides whether to add a separate (possibly non-blocking) `python manage.py test --tag ephemeris_segfault` step, or accept the coverage loss as the cost of D-05.
result: [pending]

### 3. Judgment-tier prohibitions — confirm from the session
Non-authoritative LLM verdicts flagged unverified-prohibition: (1) nothing installed before the developer confirmed the packages at 38-01 Task 2; (2) developer DB not migrated before a verified backup and never with the cron line active; (3) no live LCO/SOAR portal call, real email or heartbeat ping from an executor run; (4) downloaded tomtoolkit wheels only unzipped and diffed; (5) primary checkout never switched to issue37-code-only (38-04 and 38-06).
expected: Developer confirms each from memory of the session. Verifier evidence: SUMMARYs record `approve` and `publish` verbatim; backup src/fomo_db_20261007_pre_phase38.sqlite3 exists (mtime 11:10 PDT, after the 10:19 merge); crontab has no PHASE38-PAUSED line; HEAD's reflog for 2026-10-07 holds only commit entries (no checkout, switch, reset, rebase or amend), including through the 38-06 publish window.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
