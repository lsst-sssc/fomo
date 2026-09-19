---
schema_version: 1
open_count: 3
waived_count: 0
fixed_count: 0
total_count: 3
last_updated: 2026-09-19T06:51:00.204Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 33 | deviation | docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb |  | Plan 33-09 Task 2 acceptance criteria specified a whole-file grep for contact_person/contact_email = 0; satisfied narrowly at the public-table cell (the task's actual scope) since cells 9/10/22 legitimately use those field names for real form submission. | open |  | 2026-09-10T04:19:47.988Z |  |
| 2 | 35 | deviation | solsys_code/tests/test_campaign_reconciler.py |  | Full label-list suite has 29 failures/18 errors after this plan's RUN:{pk}:{date} removal (test_campaign_reconciler.py, test_reconcile_campaign_runs.py, test_campaign_approval.py's run_night_url import + cascading test_campaign_site_search.py) -- expected pre-migration fallout, owned by plan 35-02 per 35-01's own plan text. | open |  | 2026-09-13T03:43:10.774Z |  |
| 3 | 37 | unrun-verify | solsys_code/campaign_tally.py |  | Plan 37-04's full-solsys_code-suite regression verify was not re-confirmed this session (background run inconclusive/still-running); orchestrator runs it as the post-merge gate. | open |  | 2026-09-19T06:51:00.204Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "33",
    "file": "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb",
    "line": null,
    "description": "Plan 33-09 Task 2 acceptance criteria specified a whole-file grep for contact_person/contact_email = 0; satisfied narrowly at the public-table cell (the task's actual scope) since cells 9/10/22 legitimately use those field names for real form submission.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-10T04:19:47.988Z",
    "resolved_at": null
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "35",
    "file": "solsys_code/tests/test_campaign_reconciler.py",
    "line": null,
    "description": "Full label-list suite has 29 failures/18 errors after this plan's RUN:{pk}:{date} removal (test_campaign_reconciler.py, test_reconcile_campaign_runs.py, test_campaign_approval.py's run_night_url import + cascading test_campaign_site_search.py) -- expected pre-migration fallout, owned by plan 35-02 per 35-01's own plan text.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-13T03:43:10.774Z",
    "resolved_at": null
  },
  {
    "id": 3,
    "kind": "unrun-verify",
    "phase": "37",
    "file": "solsys_code/campaign_tally.py",
    "line": null,
    "description": "Plan 37-04's full-solsys_code-suite regression verify was not re-confirmed this session (background run inconclusive/still-running); orchestrator runs it as the post-merge gate.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-19T06:51:00.204Z",
    "resolved_at": null,
    "milestone": "v2.4"
  }
]
````
