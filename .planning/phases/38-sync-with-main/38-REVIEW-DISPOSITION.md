---
phase: 38
review: 38-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "The merge restores main's deleted `alerts/` route for an app that is no longer installed. Every page view 500s."
  - id: WR-01
    severity: warning
    disposition: open
    title: "The `local_settings.py` import path differs from main's. A host set up for main would silently fall back to dev settings."
  - id: WR-02
    severity: warning
    disposition: open
    title: "No CI job runs `TestEphemeris` any more. The core ephemeris view loses the CI coverage it had on main."
  - id: IN-01
    severity: info
    disposition: open
    title: "`suppress_warnings = ['toc.excluded']` is kept even though the reason for it is gone"
  - id: IN-02
    severity: info
    disposition: open
    title: "Stale stack lines remain in CLAUDE.md beside lines this phase updated"
  - id: IN-03
    severity: info
    disposition: open
    title: "The ruff `exclude` list does not apply under the enforced pre-commit gate"
  - id: IN-04
    severity: info
    disposition: open
    title: "The `setuptools>=62` build floor is too low for a PEP 639 `license` string"
open: 7
total: 7
recorded: 2026-10-07T19:19:03.882Z
---

# Phase 38: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
