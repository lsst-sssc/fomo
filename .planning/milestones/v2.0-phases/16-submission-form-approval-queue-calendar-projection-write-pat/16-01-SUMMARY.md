---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
plan: 01
subsystem: forms
tags: [django, crispy-forms, django-forms, honeypot, email]

# Dependency graph
requires:
  - phase: 15-per-campaign-table-view-read-path
    provides: CampaignRun model, campaign_urls.py/campaign_views.py conventions
provides:
  - StaffRequiredMixin (dispatch-level is_staff gate, redirect to LOGIN_URL)
  - CampaignRunSubmissionForm (plain forms.Form + honeypot, D-05/D-06/SUBMIT-04)
  - EMAIL_BACKEND console setting for dev/test send_mail()
affects: [16-02-submission-view, 16-03-approval-queue]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "StaffRequiredMixin: user_passes_test(lambda u: u.is_staff) on dispatch(), no explicit login_url (first hard is_staff gate in solsys_code)"
    - "Plain forms.Form (never ModelForm) when a model field's required-ness must diverge from the model definition"
    - "Non-raising clean_<field> for a honeypot field so a bot gets no error signal; the view layer decides what to do with a tripped value"

key-files:
  created:
    - solsys_code/mixins.py
    - solsys_code/campaign_forms.py
    - solsys_code/tests/test_campaign_forms.py
  modified:
    - src/fomo/settings.py

key-decisions:
  - "CampaignRunSubmissionForm is a plain forms.Form, not ModelForm, because CampaignRun.telescope_instrument has no blank=True on the model -- a ModelForm would wrongly force it required, contradicting D-05"
  - "EMAIL_BACKEND placed before the local_settings.py import block so a production override always wins"

patterns-established:
  - "StaffRequiredMixin (solsys_code/mixins.py): copy for any future staff-only view gate in this app"
  - "Honeypot field convention: HiddenInput widget + non-raising clean_<field>, trip handled in the view not the form"

requirements-completed: [SUBMIT-01, SUBMIT-04]

coverage:
  - id: D1
    description: "StaffRequiredMixin redirects non-staff/anonymous users to LOGIN_URL via dispatch()"
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "python -c import check for StaffRequiredMixin + EMAIL_BACKEND assertion"
        status: pass
    human_judgment: false
  - id: D2
    description: "CampaignRunSubmissionForm exposes the D-05 field subset as a plain forms.Form with only campaign required at the field level"
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_minimal_valid_submission"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_missing_campaign_invalid"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_telescope_instrument_not_required"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_is_plain_form_not_model_form"
        status: pass
    human_judgment: false
  - id: D3
    description: "contact_person/contact_email required at form-validation level (D-06)"
    requirement: "SUBMIT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_missing_contact_person_invalid"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_missing_contact_email_invalid"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_contact_fields_required"
        status: pass
    human_judgment: false
  - id: D4
    description: "Hidden honeypot field alt_contact_info renders as HiddenInput and its clean method never raises (SUBMIT-04)"
    requirement: "SUBMIT-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_honeypot_filled_still_valid"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_honeypot_widget_is_hidden_input"
        status: pass
    human_judgment: false
  - id: D5
    description: "EMAIL_BACKEND console backend configured so send_mail() never hangs on localhost:25 in dev/tests"
    verification:
      - kind: unit
        ref: "python -c import check asserting 'console' in settings.EMAIL_BACKEND"
        status: pass
    human_judgment: false

# Metrics
duration: 8min
completed: 2026-07-04
status: complete
---

# Phase 16 Plan 01: Leaf Dependencies (StaffRequiredMixin, CampaignRunSubmissionForm, EMAIL_BACKEND) Summary

**Plain `forms.Form` submission form with a non-raising HiddenInput honeypot, a dispatch-level `is_staff` gate mixin, and a console `EMAIL_BACKEND` -- the three self-contained leaf dependencies for Phase 16's submission/approval write path.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-04T09:43:00Z
- **Completed:** 2026-07-04T09:51:00Z
- **Tasks:** 2 (Task 2 run as TDD: RED -> GREEN)
- **Files modified:** 4 (3 created, 1 modified)

## Accomplishments

- `StaffRequiredMixin` (`solsys_code/mixins.py`) -- the first hard `is_staff` dispatch-level gate in this app, copied verbatim from the installed `tom_common.mixins.SuperuserRequiredMixin` shape and renamed/re-predicated
- `CampaignRunSubmissionForm` (`solsys_code/campaign_forms.py`) -- a plain `forms.Form` exposing the full D-05 field set, with `campaign`/`contact_person`/`contact_email` required and a hidden, non-raising `alt_contact_info` honeypot (SUBMIT-04)
- `EMAIL_BACKEND` console setting added to `src/fomo/settings.py`, placed before the `local_settings.py` import so a production override always wins
- 10 new tests in `solsys_code/tests/test_campaign_forms.py`, all green; full `solsys_code` suite (268 tests) still passes

## Task Commits

Each task was committed atomically:

1. **Task 1: StaffRequiredMixin + EMAIL_BACKEND setting** - `7a1c030` (feat)
2. **Task 2: CampaignRunSubmissionForm (TDD)** - RED `9d00300` (test), GREEN `2fde2b2` (feat)

**Plan metadata:** commit to follow (docs: complete plan)

## Files Created/Modified

- `solsys_code/mixins.py` - `StaffRequiredMixin`, dispatch-level `is_staff` gate redirecting to `LOGIN_URL`
- `solsys_code/campaign_forms.py` - `CampaignRunSubmissionForm(forms.Form)` with honeypot and crispy `FormHelper`/`Layout`
- `solsys_code/tests/test_campaign_forms.py` - 10 tests covering all five `<behavior>` items from the plan plus form-type and required-field invariants
- `src/fomo/settings.py` - added `EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'`, documented production override via `local_settings.py`

## Decisions Made

- Kept the honeypot's `clean_alt_contact_info` deliberately non-raising per SUBMIT-04 -- the trip decision belongs to the view (Plan 02), not the form. Verified by test: a filled honeypot still passes `is_valid()`.
- `EMAIL_BACKEND` placed immediately before the `try: from local_settings import *` block so a deployment's real SMTP config always wins over the dev console default.
- No `ModelForm` was used anywhere in `campaign_forms.py` (`grep -c "ModelForm"` = 0), per Pitfall 3 / D-05.

## Deviations from Plan

None - plan executed exactly as written. Task 2 followed the TDD RED -> GREEN sequence specified by `tdd="true"`; no REFACTOR commit was needed (implementation was correct on first GREEN pass, no cleanup required).

## TDD Gate Compliance

RED commit (`9d00300`, `test(16-01): ...`) precedes GREEN commit (`2fde2b2`, `feat(16-01): ...`) in git log -- gate sequence confirmed.

## Issues Encountered

- Pre-commit's initial ruff run flagged a missing docstring (D102) on `StaffRequiredMixin.dispatch` before the Task 1 commit landed; fixed inline (one-line docstring added) and re-committed successfully. Not a plan deviation -- routine lint fix during normal commit flow.
- `ruff format --diff src/fomo/settings.py` reports one unrelated pre-existing formatting fix (a blank line after the module docstring, predating this plan's changes) -- out of scope per the deviation rules' scope boundary; left untouched and logged here rather than fixed.

## User Setup Required

None - no external service configuration required. `EMAIL_BACKEND` defaults to the console backend for dev; production deployments must set real `EMAIL_BACKEND`/`EMAIL_HOST`/`EMAIL_HOST_USER`/`EMAIL_HOST_PASSWORD`/`DEFAULT_FROM_EMAIL` in `local_settings.py` (documented inline in `settings.py`).

## Next Phase Readiness

- `StaffRequiredMixin` ready for `ApprovalQueueView`/approve-reject action views (Plan 03).
- `CampaignRunSubmissionForm` ready for `CampaignRunSubmissionView` (Plan 02) -- the view will construct `CampaignRun.objects.create(**cleaned_data)` (skips `full_clean()`, so model-level blank/required checks never re-trigger) and check `cleaned_data['alt_contact_info']` for the honeypot short-circuit.
- `EMAIL_BACKEND` ready for the staff-notification `send_mail()` call in Plan 02.
- No blockers.

---
*Phase: 16-submission-form-approval-queue-calendar-projection-write-pat*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: solsys_code/mixins.py
- FOUND: solsys_code/campaign_forms.py
- FOUND: solsys_code/tests/test_campaign_forms.py
- FOUND: commit 7a1c030
- FOUND: commit 9d00300
- FOUND: commit 2fde2b2
