---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
plan: 02
subsystem: web
tags: [django, crispy-forms, honeypot, email, integrityerror, forms-form]

# Dependency graph
requires:
  - phase: 16-01
    provides: CampaignRunSubmissionForm (plain forms.Form + honeypot), EMAIL_BACKEND console setting
provides:
  - CampaignRunSubmissionView (FormView) at campaigns:submit -- public intake creating PENDING_REVIEW CampaignRun rows
  - campaigns:submission_thanks -- identical confirmation page for genuine and honeypot-tripped submissions
  - _notify_staff() staff-notification email (PII-free, D-04)
  - Friendly non_field_errors handling for the natural-key UniqueConstraint collision (Pitfall 4)
affects: [16-03-approval-queue, 16-04-calendar-projection-entry-points]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "IntegrityError caught around a create() call must wrap that call in its own transaction.atomic() savepoint, or the caught exception poisons the outer request/test transaction and any subsequent query raises TransactionManagementError instead of the intended graceful handling"
    - "A forward reference to a URL name not yet defined by an earlier-wave plan (reverse('campaigns:approval_queue') before Plan 03 lands) is wrapped in try/except NoReverseMatch with a hardcoded fallback path matching the future URL, so the current plan's tests don't depend on execution order across plans"

key-files:
  created:
    - solsys_code/tests/test_campaign_submission.py
    - src/templates/campaigns/campaignrun_submit_form.html
    - src/templates/campaigns/submission_thanks.html
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/campaign_urls.py

key-decisions:
  - "CampaignRun.objects.create() wrapped in transaction.atomic() inside the try/except IntegrityError block -- without the savepoint, the caught IntegrityError breaks the outer transaction and the form-error re-render's ModelChoiceField query raises TransactionManagementError"
  - "_notify_staff's reverse('campaigns:approval_queue') call is wrapped in try/except NoReverseMatch, falling back to the literal '/campaigns/approval-queue/' path Plan 03 will wire it to, since that URL name does not exist until Plan 03 (Wave 3) lands"

patterns-established:
  - "Public FormView write path pattern: honeypot short-circuit checked FIRST in form_valid (before any DB write), CampaignRun built via cleaned_data mapping (never full_clean()), duplicate-natural-key collisions caught as friendly form errors via a transaction.atomic() savepoint"

requirements-completed: [SUBMIT-01, SUBMIT-04, SUBMIT-05]

coverage:
  - id: D1
    description: "CampaignRunSubmissionView (FormView) at campaigns:submit; a minimal valid POST (campaign + contact_person + contact_email) creates exactly one PENDING_REVIEW CampaignRun and redirects to campaigns:submission_thanks"
    requirement: "SUBMIT-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_minimal_valid_submission_creates_pending_run"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_missing_campaign_invalid"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_missing_contact_person_invalid"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_missing_contact_email_invalid"
        status: pass
    human_judgment: false
  - id: D2
    description: "Duplicate campaign+telescope_instrument+ut_start submission surfaces a friendly non_field_errors banner, never a 500 (Pitfall 4)"
    requirement: "SUBMIT-01"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestCampaignSubmission.test_duplicate_natural_key_submission_shows_friendly_form_error"
        status: pass
    human_judgment: false
  - id: D3
    description: "A honeypot-tripped submission (alt_contact_info populated) creates zero CampaignRun rows, sends zero emails, and redirects to the identical thanks page as a genuine submission"
    requirement: "SUBMIT-04"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestHoneypot.test_honeypot_filled_creates_no_run_and_sends_no_email"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestHoneypot.test_honeypot_response_matches_genuine_submission_redirect"
        status: pass
    human_judgment: false
  - id: D4
    description: "The submission_thanks.html template renders identical markup for genuine and honeypot-tripped submissions -- no conditional branch keyed on any honeypot/status variable"
    requirement: "SUBMIT-04"
    verification:
      - kind: unit
        ref: "grep -c 'alt_contact_info\\|honeypot' src/templates/campaigns/submission_thanks.html == 0"
        status: pass
    human_judgment: false
  - id: D5
    description: "A genuine submission emails every is_staff user with a non-empty email (excluding blank-email staff and non-staff users); subject/body contain no contact_person/contact_email/telescope_instrument/campaign-name PII"
    requirement: "SUBMIT-05"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestStaffNotification.test_genuine_submission_emails_every_staff_user_with_email"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestStaffNotification.test_staff_with_blank_email_not_a_recipient"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestStaffNotification.test_non_staff_user_not_a_recipient"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestStaffNotification.test_email_contains_no_pii"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_submission.py#TestStaffNotification.test_no_staff_with_email_sends_no_email"
        status: pass
    human_judgment: false
  - id: D6
    description: "Manual UI check that the honeypot field is genuinely invisible to a sighted user in the rendered submission form (not just display:none)"
    human_judgment: true
    rationale: "Visual/accessibility inspection of a hidden form field cannot be asserted meaningfully in an integration test -- requires rendering the page in a browser and inspecting the DOM, per 16-VALIDATION.md's Manual-Only Verifications."
    verification: []

# Metrics
duration: 26min
completed: 2026-07-04
status: complete
---

# Phase 16 Plan 02: Submission View, URLs & Templates Summary

**`CampaignRunSubmissionView` (FormView) wired at `campaigns:submit`, backed by a `transaction.atomic()`-guarded `.objects.create()` that turns Pitfall 4's natural-key collision into a friendly form error, a honeypot short-circuit that returns the identical thanks redirect (SUBMIT-04), and a PII-free staff-notification email (SUBMIT-05).**

## Performance

- **Duration:** 26 min
- **Started:** 2026-07-04T10:53:00Z
- **Completed:** 2026-07-04T11:19:00Z
- **Tasks:** 2
- **Files modified:** 5 (4 created, 2 modified — `campaign_views.py` touched by both tasks)

## Accomplishments

- `CampaignRunSubmissionView` (`FormView`) at `campaigns:submit`: honeypot check first (no create/email/error on trip), `CampaignRun.objects.create()` from `cleaned_data` (model default `PENDING_REVIEW` applies, `site`/`site_needs_review` deliberately left unset per D-07), `IntegrityError` on the natural-key collision degrades to a friendly `non_field_errors` banner
- `_notify_staff()`: emails every `is_staff=True` user with a non-blank email a PII-free ping (subject/body carry no contact/telescope/campaign detail) plus a link to the (future) approval queue; `fail_silently=True` so a mail outage never breaks the submission (Pitfall 6)
- Two new URL names (`campaigns:submit`, `campaigns:submission_thanks`) and two new templates matching UI-SPEC Pages 3/4 exactly, including the byte-for-byte identical thanks-page markup for genuine and honeypot-tripped requests
- `solsys_code/tests/test_campaign_submission.py`: 13 new tests covering all three requirements plus the duplicate-collision edge case; full `solsys_code` suite (281 tests) passes

## Task Commits

Each task was committed atomically:

1. **Task 1: CampaignRunSubmissionView + URLs + templates** - `3514c3e` (feat)
2. **Task 2: test_campaign_submission.py (SUBMIT-01/04/05)** - `7642d7f` (test)

**Plan metadata:** commit to follow (docs: complete plan)

## Files Created/Modified

- `solsys_code/campaign_views.py` - added `CampaignRunSubmissionView(FormView)` with `form_valid`/`_notify_staff`
- `solsys_code/campaign_urls.py` - added `submit`/`submission_thanks` URL names
- `src/templates/campaigns/campaignrun_submit_form.html` - new submission form template (UI-SPEC Page 3)
- `src/templates/campaigns/submission_thanks.html` - new confirmation page template (UI-SPEC Page 4)
- `solsys_code/tests/test_campaign_submission.py` - 13 tests: `TestCampaignSubmission` (6), `TestHoneypot` (2), `TestStaffNotification` (5)

## Decisions Made

- Wrapped `CampaignRun.objects.create()` in `transaction.atomic()` inside the `try/except IntegrityError` block (not present in the plan's literal code example) -- without the savepoint, Django's per-test/per-request outer transaction is left broken after the caught `IntegrityError`, and the subsequent form re-render's `ModelChoiceField` queryset query raises `TransactionManagementError` instead of showing the intended friendly error. Verified live: `test_duplicate_natural_key_submission_shows_friendly_form_error` failed with this exact error before the fix, passes after.
- `_notify_staff`'s `reverse('campaigns:approval_queue')` call is wrapped in `try/except NoReverseMatch`, falling back to the literal path (`/campaigns/approval-queue/`) that Plan 03 will register that URL name to. `campaigns:approval_queue` does not exist until Plan 03 (Wave 3) lands, so calling it unconditionally (as the plan's literal code example does) would raise `NoReverseMatch` on every genuine submission at this point in the phase's sequential wave order.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `transaction.atomic()` savepoint around the natural-key `.objects.create()`**
- **Found during:** Task 2 (writing `test_duplicate_natural_key_submission_shows_friendly_form_error`)
- **Issue:** Catching `IntegrityError` around a bare `.objects.create()` call, as written in the plan's action text and 16-RESEARCH.md's code example, leaves the enclosing transaction broken. The second POST in the duplicate test raised `django.db.transaction.TransactionManagementError` when the form was re-rendered (the `campaign` `ModelChoiceField`'s queryset triggers a DB query), instead of showing the intended `non_field_errors` banner.
- **Fix:** Wrapped the `.objects.create()` call in `with transaction.atomic():` inside the existing `try/except IntegrityError` block, isolating the failure to its own savepoint.
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** `test_duplicate_natural_key_submission_shows_friendly_form_error` passes; full `solsys_code` suite (281/281) passes.
- **Committed in:** `7642d7f` (Task 2 commit)

**2. [Rule 3 - Blocking] `NoReverseMatch` fallback for `campaigns:approval_queue`**
- **Found during:** Task 1 (writing `_notify_staff`)
- **Issue:** The plan's action text and 16-RESEARCH.md's code example call `reverse('campaigns:approval_queue')` unconditionally, but that URL name is only registered by Plan 03 (Wave 3), which has not executed yet at this plan's point in the phase's sequential wave order. Calling it unconditionally would raise `NoReverseMatch` on every genuine (non-honeypot) submission, breaking `SUBMIT-01`/`SUBMIT-05` entirely until Plan 03 lands.
- **Fix:** Wrapped the `reverse()` call in `try/except NoReverseMatch`, falling back to the literal path (`/campaigns/approval-queue/`) that 16-03-PLAN.md registers that URL name to (`path('approval-queue/', ...)` mounted under `src/fomo/urls.py`'s `campaigns/` prefix). Once Plan 03 lands, `reverse()` succeeds and this branch becomes dead code -- no further change needed.
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** All `TestStaffNotification` tests pass with the fallback active (Plan 03 not yet executed in this repo state); URL-resolution one-liner confirms `campaigns:submit`/`campaigns:submission_thanks` resolve.
- **Committed in:** `3514c3e` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes were necessary for Task 2's tests to pass without a false-positive/broken-transaction failure and for the view to function correctly ahead of Plan 03 landing. No scope creep — the `approval_queue` URL and template remain entirely Plan 03's responsibility; this plan only adds the temporary forward-compatible fallback.

## Issues Encountered

- Task 1 is marked `tdd="true"` in the plan but its own `<verify>` command (`./manage.py test solsys_code.tests.test_campaign_submission`) targets a test module that the plan's own `<done>` annotation says is "authored in Task 2" -- i.e., the plan sequences implementation (Task 1) before tests (Task 2), the reverse of the standard TDD RED-then-GREEN order. Handled pragmatically: Task 1 was verified manually (URL resolution via a one-line `reverse()` check, `ruff` clean) and committed as a `feat` commit; Task 2 then authored the full test suite, which passed immediately against the already-correct implementation (no RED phase was possible or expected given the plan's task split). Not a deviation from the plan's actual intent, just a note that the `tdd="true"` tag on Task 1 didn't map to a literal RED commit.

## User Setup Required

None - no external service configuration required. `EMAIL_BACKEND` (console) from Plan 01 already covers dev/test email visibility.

## Next Phase Readiness

- `CampaignRunSubmissionView`/`campaigns:submit`/`campaigns:submission_thanks` are live and fully tested; Plan 04's entry-point buttons (campaigns list / per-campaign table "Submit a Run") can link to `campaigns:submit` immediately.
- `_notify_staff`'s `NoReverseMatch` fallback is temporary scaffolding for Plan 03: once Plan 03 registers `campaigns:approval_queue`, the `try/except` branch becomes dead code (no follow-up edit required, but worth a quick `grep -n "NoReverseMatch" solsys_code/campaign_views.py` sanity check after Plan 03 lands to confirm the primary `reverse()` path is now taken).
- No blockers for Plan 03 (approval queue) or Plan 04 (entry points/calendar projection).

---
*Phase: 16-submission-form-approval-queue-calendar-projection-write-pat*
*Completed: 2026-07-04*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/campaign_urls.py
- FOUND: src/templates/campaigns/campaignrun_submit_form.html
- FOUND: src/templates/campaigns/submission_thanks.html
- FOUND: solsys_code/tests/test_campaign_submission.py
- FOUND: commit 3514c3e
- FOUND: commit 7642d7f
