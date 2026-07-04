---
phase: 16
slug: submission-form-approval-queue-calendar-projection-write-pat
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-03
validated: 2026-07-04
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), per CLAUDE.md's "DB-dependent tests go in `solsys_code/tests/`" convention — **not** pytest (`pyproject.toml` `testpaths` excludes `solsys_code/`) |
| **Config file** | none — Django test discovery via `./manage.py test solsys_code` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_campaign_submission` (once created) |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30s quick / ~120s+ full (full suite collection imports `solsys_code.ephem_utils`, which pays a one-time ~1.6GB SPICE kernel download on a cold cache per CLAUDE.md; cached on subsequent runs) |

---

## Sampling Rate

- **After every task commit:** Run the targeted test module for the file(s) touched, e.g. `./manage.py test solsys_code.tests.test_campaign_submission`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green, plus `ruff check .` / `ruff format --check .` clean per CLAUDE.md
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 16-01/16-02/16-04 | 01, 02, 04 | 1, 2, 4 | SUBMIT-01 | — | Valid submission (campaign only) creates a `PENDING_REVIEW` `CampaignRun`; missing `campaign` fails validation; missing `contact_person`/`contact_email` fails validation (D-06) | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_submission.TestCampaignSubmission.test_minimal_valid_submission_creates_pending_run` | ✅ | ✅ green |
| 16-04 | 04 | 4 | SUBMIT-02 | Visibility of unapproved data — see Security Domain below | Anonymous client cannot see a `pending_review` row on the per-campaign table | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_views.TestNonStaffPendingReviewHidden` | ✅ | ✅ green |
| 16-03/16-05 | 03, 05 | 3, 1 (16-05 is a gap-closure plan, own wave 1) | SUBMIT-03 | Race condition / double-processing — see Security Domain below | Approve twice: first call transitions, second call is a proven no-op (`updated_count == 0`, no duplicate `CalendarEvent`, no second email) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestApproval.test_double_approve_is_noop` | ✅ | ✅ green |
| 16-01/16-02 | 01, 02 | 1, 2 | SUBMIT-04 | Bot/automated abuse — see Security Domain below | Honeypot-filled submission: no `CampaignRun` created, no email sent, response is the same success page as a genuine submission | integration | `./manage.py test solsys_code.tests.test_campaign_submission.TestHoneypot` | ✅ | ✅ green |
| 16-02 | 02 | 2 | SUBMIT-05 | PII disclosure — see Security Domain below | Genuine submission triggers `send_mail` to every `is_staff=True` user with a non-empty email (staff with blank email excluded); email body/subject contain no PII (no `contact_person`/`contact_email`/telescope/campaign name), proving D-04 | integration (`django.core.mail.outbox`) | `./manage.py test solsys_code.tests.test_campaign_submission.TestStaffNotification` | ✅ | ✅ green |
| 16-03 | 03 | 3 | CAL-01 | — | Approving a run with telescope + `ut_start` + `ut_end` creates a `CalendarEvent` keyed `CAMPAIGN:{pk}` via `insert_or_create_calendar_event()` | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarProjection` | ✅ | ✅ green |
| 16-03 | 03 | 3 | CAL-02 | — | Created `CalendarEvent.target_list` equals the campaign's `TargetList` | integration | same test module as CAL-01 (`test_approve_with_full_window_creates_calendar_event`) | ✅ | ✅ green |
| 16-03 | 03 | 3 | CAL-03 | — | Re-approving an already-approved run creates no duplicate `CalendarEvent` and causes no `modified` churn (assert `CalendarEvent.objects.count()` unchanged and `modified` timestamp unchanged after the second approve attempt) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarNoChurn` | ✅ | ✅ green |
| 16-04 | 04 | 4 | D-09 (per-campaign table extension, not a numbered requirement) | Visibility of unapproved/rejected data | Non-staff sees `approved` and `rejected` rows, not `pending_review`, on the per-campaign table; staff continues to see every row | integration | `./manage.py test solsys_code.tests.test_campaign_views.TestNonStaffPendingReviewHidden` | ✅ | ✅ green |
| 16-03 | 03 | 3 | D-01/D-02 (approval-queue access, not a numbered requirement) | Access control — see Security Domain below | Anonymous/non-staff GET to approval-queue and approve/reject URLs redirects (never 200 with content) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestStaffGating` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Task ID/Plan/Wave columns resolved against the executed plans (`16-01` through `16-05`); `16-05`
is the post-verification gap-closure plan (UAT Test 14, `ApprovalQueueTable` column trim/reorder),
tagged `SUBMIT-03` in its own frontmatter. All commands above were re-run directly during this
audit (`python manage.py test solsys_code.tests.test_campaign_submission
solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_views
solsys_code.tests.test_campaign_forms` → 61 tests, OK) and independently confirmed green in
16-VERIFICATION.md's 303-test full-suite run.

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_campaign_submission.py` — created (13 tests: `TestCampaignSubmission`, `TestHoneypot`, `TestStaffNotification`), covers SUBMIT-01, SUBMIT-04, SUBMIT-05
- [x] `solsys_code/tests/test_campaign_approval.py` — created (17 tests: `TestStaffGating`, `TestApproval`, `TestCalendarProjection`, `TestApprovalQueueColumns`, `TestCalendarNoChurn`), covers SUBMIT-03, CAL-01, CAL-02, CAL-03, and staff-gating for the approval-queue/action views
- [x] Extended `solsys_code/tests/test_campaign_views.py` — `TestNonStaffPendingReviewHidden` added for D-09 non-staff visibility filter
- [x] No new test framework install needed — `django.test.TestCase` + `django.core.mail` (`outbox`) used as anticipated; no new fixture types beyond those already established in `test_campaign_views.py` from Phase 15

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Approval-queue page layout (pending + recently-decided sections) is legible and the Approve/Reject actions are unambiguous | D-01/D-02 | Layout clarity and visual affordance are subjective judgments; functional presence of both sections and correct action wiring are covered by automated tests above, but visual layout review belongs to UI-SPEC.md / `/gsd-ui-review`, not this functional test suite | Render the approval-queue page in a browser as a staff user, confirm pending and recently-decided rows are visually distinguishable and Approve/Reject controls are discoverable |
| Submission form's honeypot field is genuinely invisible to a sighted human user (not just `display:none` triggering an accessibility complaint) | SUBMIT-04 | Visual/accessibility inspection of a hidden form field is not practical to assert in an integration test | Render the submission form in a browser, confirm no visible extra field, and inspect the rendered HTML for an appropriately hidden/labeled honeypot input |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-07-04

---

## Validation Audit 2026-07-04

State A audit (VALIDATION.md existed as the pre-execution draft with `TBD` task IDs and
`⬜ pending` statuses; all 5 plans/summaries and 16-VERIFICATION.md were available for
cross-reference).

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All 10 rows in the Per-Task Verification Map (SUBMIT-01..05, CAL-01..03, D-09, D-01/D-02) were
matched to real, existing, passing tests in `solsys_code/tests/test_campaign_submission.py`,
`test_campaign_approval.py`, and `test_campaign_views.py`. Re-ran the targeted modules directly
during this audit:

```
python manage.py test solsys_code.tests.test_campaign_submission \
  solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_views \
  solsys_code.tests.test_campaign_forms
→ Ran 61 tests ... OK
```

This corroborates 16-VERIFICATION.md's independent 303-test full-suite run. No auditor subagent
was spawned (no gaps to fill). Task ID/Plan/Wave placeholders resolved against the executed
PLAN.md frontmatter (`requirements:` fields) and file `files_modified` lists. `nyquist_compliant`
set to `true`.
