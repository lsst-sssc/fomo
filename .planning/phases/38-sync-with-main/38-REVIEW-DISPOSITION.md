---
phase: 38
review: 38-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "\"`src/fomo/local_settings.py` is the location on every current FOMO branch\" is false, and PR #58 is not merged"
  - id: WR-02
    severity: warning
    disposition: open
    title: "PR #58's `from .local_settings import *` is not \"the same change\"; under `manage.py` it breaks the WR-32 guard this page relies on"
  - id: IN-01
    severity: info
    disposition: open
    title: "The check's \"`ModuleNotFoundError` means the file is not where FOMO looks\" also catches a missing dependency inside the file"
  - id: IN-02
    severity: info
    disposition: open
    title: "\"nothing reports it\" is overstated; `check_unattended` (and `check --deploy`) do flag the effects"
  - id: IN-03
    severity: info
    disposition: open
    title: "The upgrade warning doesn't cover what already ran against the development defaults"
  - id: IN-04
    severity: info
    disposition: open
    title: "The location rule in the parentheses is a guess, and `mv` overwrites silently"
  - id: IN-05
    severity: info
    disposition: open
    title: "\"anything set there replaces the default\" contradicts the runbook's FOMO_STATE_DIR trap"
  - id: IN-06
    severity: info
    disposition: open
    title: "Lines edited in place break the surrounding wrap width"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "The merge restores main's deleted `alerts/` route for an app that is no longer installed. Every page view 500s."
open: 8
total: 9
recorded: 2026-10-08T02:41:06.570Z
---

# Phase 38: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| CR-01 | critical | fixed | fixed by 38-05 (test 32dafa2, fix a4d77f2, docs db3ae7c; published to PR #43 by 38-06 snapshot 846be34); the 2026-10-08 re-review of src/fomo/urls.py and solsys_code/tests/test_urls.py confirms the include is gone and nothing reverses `alerts:` (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.

## ID reuse across rounds (2026-10-08)

The 2026-10-08 re-review (`cf01184`, scope `docs/installation.rst` + `docs/runbooks/telescope_runs_calendar.rst`) reused the IDs WR-01, WR-02 and IN-01..IN-06, so every row above except CR-01 now names a 2026-10-08 finding. The 2026-10-07 findings those IDs previously denoted (review `9d916d1`) are accounted for as follows: the old **WR-01** ("The `local_settings.py` import path differs from main's. A host set up for main would silently fall back to dev settings.") became UAT gap G-38-1 and was **fixed by plan 38-07** (docs `5421abb`, `a060f9d`, `82a097c`; PR #43 body `cf24051`, applied to the live PR on the developer's `apply`); the old WR-02 and IN-01..IN-04 were superseded by the 38-05/38-06 fixes and the `b07109a` re-review, which carried only IN-05/IN-06 (now also renumbered).
