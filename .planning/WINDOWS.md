---
schema_version: 1
open_count: 0
waived_count: 2
fixed_count: 3
total_count: 5
last_updated: 2026-09-22T17:47:40.294Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 33 | deviation | docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb |  | Plan 33-09 Task 2 acceptance criteria specified a whole-file grep for contact_person/contact_email = 0; satisfied narrowly at the public-table cell (the task's actual scope) since cells 9/10/22 legitimately use those field names for real form submission. | waived | Not a defect: 33-09 Task 2's scope was the public-table cell, which has zero contact_person/contact_email references; notebook cells 9/10/22 legitimately use those field names for real form submission. A whole-file grep=0 was an over-broad criterion. | 2026-09-10T04:19:47.988Z | 2026-09-22T17:47:40.209Z |
| 2 | 35 | deviation | solsys_code/tests/test_campaign_reconciler.py |  | Full label-list suite has 29 failures/18 errors after this plan's RUN:{pk}:{date} removal (test_campaign_reconciler.py, test_reconcile_campaign_runs.py, test_campaign_approval.py's run_night_url import + cascading test_campaign_site_search.py) -- expected pre-migration fallout, owned by plan 35-02 per 35-01's own plan text. | fixed |  | 2026-09-13T03:43:10.774Z | 2026-09-22T17:47:39.956Z |
| 3 | 37 | unrun-verify | solsys_code/campaign_tally.py |  | Plan 37-04's full-solsys_code-suite regression verify was not re-confirmed this session (background run inconclusive/still-running); orchestrator runs it as the post-merge gate. | fixed |  | 2026-09-19T06:51:00.204Z | 2026-09-22T17:47:40.040Z |
| 4 | 37 | unrun-verify | solsys_code (full suite) |  | Plan 37-05's plan-level verify command 'python manage.py test solsys_code' (excluding test_views.TestEphemeris) was not re-run this session; the orchestrator runs the full suite as its post-merge gate immediately after this plan returns (per this plan's closeout_discipline instruction). All of this plan's own test_campaign_views/test_campaign_tally tests (127) and inline verify probes passed. | fixed |  | 2026-09-19T07:47:35.888Z | 2026-09-22T17:47:40.123Z |
| 5 | 37 | deviation | solsys_code/tests/test_bootstrap5_rendering.py |  | Observed a flaky, order-dependent Playwright failure in test_observatory_create_form_submits_to_observatory_url during 37-08's full-suite gate; confirmed transient (passes in isolation, with 37-08's changed test modules, and on an immediate full-suite retry). Logged in 37-status-vocabulary-public-tallies-provenance-blind-gaps/deferred-items.md. | waived | Transient, order-dependent Playwright flake in test_observatory_create_form_submits_to_observatory_url; passes in isolation, on retry, and in the 2026-09-22 full-suite run (1776 tests OK). That test and the observatory views are unchanged in v2.4 (only 33-11's calendar-modal test in the same file was edited). Tracked for follow-up in phase 37 deferred-items.md. | 2026-09-21T04:46:53.397Z | 2026-09-22T17:47:40.294Z |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "33",
    "file": "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb",
    "line": null,
    "description": "Plan 33-09 Task 2 acceptance criteria specified a whole-file grep for contact_person/contact_email = 0; satisfied narrowly at the public-table cell (the task's actual scope) since cells 9/10/22 legitimately use those field names for real form submission.",
    "status": "waived",
    "reason": "Not a defect: 33-09 Task 2's scope was the public-table cell, which has zero contact_person/contact_email references; notebook cells 9/10/22 legitimately use those field names for real form submission. A whole-file grep=0 was an over-broad criterion.",
    "recorded_at": "2026-09-10T04:19:47.988Z",
    "resolved_at": "2026-09-22T17:47:40.209Z"
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "35",
    "file": "solsys_code/tests/test_campaign_reconciler.py",
    "line": null,
    "description": "Full label-list suite has 29 failures/18 errors after this plan's RUN:{pk}:{date} removal (test_campaign_reconciler.py, test_reconcile_campaign_runs.py, test_campaign_approval.py's run_night_url import + cascading test_campaign_site_search.py) -- expected pre-migration fallout, owned by plan 35-02 per 35-01's own plan text.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-13T03:43:10.774Z",
    "resolved_at": "2026-09-22T17:47:39.956Z"
  },
  {
    "id": 3,
    "kind": "unrun-verify",
    "phase": "37",
    "file": "solsys_code/campaign_tally.py",
    "line": null,
    "description": "Plan 37-04's full-solsys_code-suite regression verify was not re-confirmed this session (background run inconclusive/still-running); orchestrator runs it as the post-merge gate.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-19T06:51:00.204Z",
    "resolved_at": "2026-09-22T17:47:40.040Z",
    "milestone": "v2.4"
  },
  {
    "id": 4,
    "kind": "unrun-verify",
    "phase": "37",
    "file": "solsys_code (full suite)",
    "line": null,
    "description": "Plan 37-05's plan-level verify command 'python manage.py test solsys_code' (excluding test_views.TestEphemeris) was not re-run this session; the orchestrator runs the full suite as its post-merge gate immediately after this plan returns (per this plan's closeout_discipline instruction). All of this plan's own test_campaign_views/test_campaign_tally tests (127) and inline verify probes passed.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-19T07:47:35.888Z",
    "resolved_at": "2026-09-22T17:47:40.123Z",
    "milestone": "v2.4"
  },
  {
    "id": 5,
    "kind": "deviation",
    "phase": "37",
    "file": "solsys_code/tests/test_bootstrap5_rendering.py",
    "line": null,
    "description": "Observed a flaky, order-dependent Playwright failure in test_observatory_create_form_submits_to_observatory_url during 37-08's full-suite gate; confirmed transient (passes in isolation, with 37-08's changed test modules, and on an immediate full-suite retry). Logged in 37-status-vocabulary-public-tallies-provenance-blind-gaps/deferred-items.md.",
    "status": "waived",
    "reason": "Transient, order-dependent Playwright flake in test_observatory_create_form_submits_to_observatory_url; passes in isolation, on retry, and in the 2026-09-22 full-suite run (1776 tests OK). That test and the observatory views are unchanged in v2.4 (only 33-11's calendar-modal test in the same file was edited). Tracked for follow-up in phase 37 deferred-items.md.",
    "recorded_at": "2026-09-21T04:46:53.397Z",
    "resolved_at": "2026-09-22T17:47:40.294Z",
    "milestone": "v2.4"
  }
]
````
