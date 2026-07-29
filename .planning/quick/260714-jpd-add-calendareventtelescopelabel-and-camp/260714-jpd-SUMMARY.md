---
phase: quick-260714-jpd
plan: 01
subsystem: admin
tags: [django-admin, campaignrun, calendareventtelescopelabel, pii-gating]

requires: []
provides:
  - "solsys_code/admin.py registers CampaignRun (triage-forward list_display, approval_status
    read-only, contact PII excluded from the change-list) and CalendarEventTelescopeLabel
    (standalone, event + is_verified, event__title search)"
  - "solsys_code/tests/test_admin.py proving both models are admin-reachable, approval_status
    is visible-but-non-editable, contact_person/contact_email are gated from the change-list
    but editable in the detail view, and the CalendarEventTelescopeLabel search path resolves"
affects: [admin, campaign-approval, pii-handling]

tech-stack:
  added: []
  patterns:
    - "Custom ModelAdmin + admin.site.register, mirroring solsys_code_observatory/admin.py's
      ObservatoryAdmin style precedent"
    - "readonly_fields (not exclude) used to keep a status field visible-but-non-editable when
      its real transition side effects live in a view, not the model"

key-files:
  created:
    - solsys_code/tests/test_admin.py
  modified:
    - solsys_code/admin.py

key-decisions:
  - "approval_status is readonly_fields, not excluded -- keeps the current value visible in the
    change form while blocking any admin path to APPROVED that bypasses
    CampaignRunDecisionView.post()'s calendar projection + D-06 site guard"
  - "contact_person/contact_email excluded from list_display only, left editable in the detail
    view, extending the VIEW-03/VIEW-05 PII-gating discipline to the admin list view without
    removing staff's ability to fix a submitter's typo'd contact info"
  - "CalendarEventTelescopeLabel registered standalone (not inline) -- no local admin exists for
    tom_calendar.CalendarEvent to inline it onto"

patterns-established:
  - "Admin test fixture attribute for a CampaignRun instance must not be named `run` on a
    django.test.TestCase subclass -- it shadows unittest.TestCase.run(), the method the test
    framework itself invokes to execute the test, producing 'TypeError: X object is not
    callable' with no other diagnostic. Use a more specific name (e.g. campaign_run)."

requirements-completed: [QUICK-260714-jpd]

coverage:
  - id: D1
    description: "CampaignRun and CalendarEventTelescopeLabel are both registered with Django
      admin and reachable at /admin/solsys_code/"
    requirement: "QUICK-260714-jpd"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_campaignrun_changelist_loads"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_calendareventtelescopelabel_changelist_loads"
        status: pass
    human_judgment: false
  - id: D2
    description: "approval_status is displayed but non-editable in the CampaignRun change form
      (no admin path to APPROVED that bypasses CampaignRunDecisionView.post())"
    requirement: "QUICK-260714-jpd"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_approval_status_is_readonly_in_change_form"
        status: pass
    human_judgment: false
  - id: D3
    description: "contact_person/contact_email never appear in the CampaignRun change-list but
      remain editable in the detail view"
    requirement: "QUICK-260714-jpd"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_pii_not_rendered_in_changelist"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_contact_fields_editable_in_change_form"
        status: pass
    human_judgment: false
  - id: D4
    description: "CalendarEventTelescopeLabel is registered standalone with a valid
      event__title search path (no FieldError)"
    requirement: "QUICK-260714-jpd"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_admin.py#AdminRegistrationAndGatingTests.test_calendareventtelescopelabel_search_resolves"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-07-14
status: complete
---

# Quick Task 260714-jpd: Add CalendarEventTelescopeLabel and CampaignRun to admin Summary

**Filled in `solsys_code/admin.py` (previously the untouched startproject stub) to register
`CampaignRun` and `CalendarEventTelescopeLabel`, keeping `approval_status` read-only and contact
PII out of the change-list, proven by a new `solsys_code/tests/test_admin.py` admin-client suite.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-14T00:00:00Z (approx; not separately timestamped by the invoking session)
- **Completed:** 2026-07-14
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `solsys_code/admin.py` now registers `CampaignRunAdmin` (triage-forward `list_display`,
  `approval_status` read-only, PII excluded from the change-list) and
  `CalendarEventTelescopeLabelAdmin` (standalone, `event` + `is_verified`, `event__title` search).
- New `solsys_code/tests/test_admin.py` proves both change-lists load, `approval_status` is
  visible-but-non-editable, `contact_person`/`contact_email` are gated from the list view but
  editable in the detail view, and the label admin's search path resolves without a FieldError.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register CampaignRun and CalendarEventTelescopeLabel in solsys_code/admin.py** - `bc97894` (feat)
2. **Task 2: Add solsys_code/tests/test_admin.py verifying read-only approval_status, PII gating, and registration** - `b6ae100` (test)

**Plan metadata:** committed separately by the orchestrator after this SUMMARY.

## Files Created/Modified
- `solsys_code/admin.py` - Registers `CampaignRunAdmin` and `CalendarEventTelescopeLabelAdmin`
  with `admin.site.register`, mirroring the sibling `ObservatoryAdmin` style.
- `solsys_code/tests/test_admin.py` - Admin-test-client suite covering registration reachability,
  the `approval_status` read-only invariant, PII list-view gating, and the label search path.

## Decisions Made
- Kept `readonly_fields = ['approval_status']` exactly as scoped rather than `exclude`, per the
  plan/CONTEXT: the field must stay visible (shows "Pending Review") but non-editable.
- No other fields were made read-only or excluded beyond what CONTEXT specified — `window_start`,
  `window_end`, `site`, and `run_status` remain admin-editable by default.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Renamed test fixture attribute from `cls.run` to `cls.campaign_run`**
- **Found during:** Task 2 (writing `solsys_code/tests/test_admin.py`)
- **Issue:** The plan's fixture instructions named the `CampaignRun` instance `run` (natural
  English name), but assigning `cls.run = CampaignRun.objects.create(...)` on a
  `django.test.TestCase` subclass shadows `unittest.TestCase.run()`, the method the test runner
  itself calls to execute each test. This produced `TypeError: 'CampaignRun' object is not
  callable` at test-collection time with no test-specific traceback, blocking all 6 tests.
- **Fix:** Renamed the attribute to `cls.campaign_run` and updated its two usages
  (`test_approval_status_is_readonly_in_change_form`,
  `test_contact_fields_editable_in_change_form`).
- **Files modified:** `solsys_code/tests/test_admin.py`
- **Verification:** `./manage.py test solsys_code.tests.test_admin -v 2` — all 6 tests pass.
- **Committed in:** `b6ae100` (Task 2 commit, fixed before first commit of this file — no
  separate commit needed)

Also noted as a new pattern (`patterns-established` above) so future admin/model test authors
don't repeat the same shadowing mistake.

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for the test suite to run at all; no scope creep — same
test coverage as planned, just a safe attribute rename.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both models are now reachable under `/admin/solsys_code/` for staff use as an escape hatch for
  data with no other fix-it path (e.g. correcting a mis-resolved `site` or `window_start`/
  `window_end`).
- The `approval_status` read-only invariant and PII list-view gating are now regression-tested,
  so future admin changes to `CampaignRunAdmin` will be caught if they reintroduce either gap.
- No blockers for Phase 21 verification or subsequent work.

---
*Phase: quick-260714-jpd*
*Completed: 2026-07-14*

## Self-Check: PASSED

- FOUND: solsys_code/admin.py
- FOUND: solsys_code/tests/test_admin.py
- FOUND: .planning/quick/260714-jpd-add-calendareventtelescopelabel-and-camp/260714-jpd-SUMMARY.md
- FOUND: bc97894 (Task 1 commit)
- FOUND: b6ae100 (Task 2 commit)
