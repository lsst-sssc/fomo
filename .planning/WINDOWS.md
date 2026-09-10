---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-09-10T04:19:47.988Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 33 | deviation | docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb |  | Plan 33-09 Task 2 acceptance criteria specified a whole-file grep for contact_person/contact_email = 0; satisfied narrowly at the public-table cell (the task's actual scope) since cells 9/10/22 legitimately use those field names for real form submission. | open |  | 2026-09-10T04:19:47.988Z |  |

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
  }
]
````
