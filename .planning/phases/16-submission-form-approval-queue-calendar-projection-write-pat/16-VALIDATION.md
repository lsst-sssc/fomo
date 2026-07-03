---
phase: 16
slug: submission-form-approval-queue-calendar-projection-write-pat
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-03
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
| TBD (assigned by planner) | TBD | TBD | SUBMIT-01 | — | Valid submission (campaign only) creates a `PENDING_REVIEW` `CampaignRun`; missing `campaign` fails validation; missing `contact_person`/`contact_email` fails validation (D-06) | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_submission.TestCampaignSubmission.test_minimal_valid_submission_creates_pending_run` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | SUBMIT-02 | Visibility of unapproved data — see Security Domain below | Anonymous client cannot see a `pending_review` row on the per-campaign table | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_views.TestContactFieldGating` (extend) or new module | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | SUBMIT-03 | Race condition / double-processing — see Security Domain below | Approve twice: first call transitions, second call is a proven no-op (`updated_count == 0`, no duplicate `CalendarEvent`, no second email) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestApproval.test_double_approve_is_noop` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | SUBMIT-04 | Bot/automated abuse — see Security Domain below | Honeypot-filled submission: no `CampaignRun` created, no email sent, response is the same success page as a genuine submission | integration | `./manage.py test solsys_code.tests.test_campaign_submission.TestHoneypot` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | SUBMIT-05 | PII disclosure — see Security Domain below | Genuine submission triggers `send_mail` to every `is_staff=True` user with a non-empty email (staff with blank email excluded); email body/subject contain no PII (no `contact_person`/`contact_email`/telescope/campaign name), proving D-04 | integration (`django.core.mail.outbox`) | `./manage.py test solsys_code.tests.test_campaign_submission.TestStaffNotification` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | CAL-01 | — | Approving a run with telescope + `ut_start` + `ut_end` creates a `CalendarEvent` keyed `CAMPAIGN:{pk}` via `insert_or_create_calendar_event()` | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarProjection` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | CAL-02 | — | Created `CalendarEvent.target_list` equals the campaign's `TargetList` | integration | same test module as CAL-01 | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | CAL-03 | — | Re-approving an already-approved run creates no duplicate `CalendarEvent` and causes no `modified` churn (assert `CalendarEvent.objects.count()` unchanged and `modified` timestamp unchanged after the second approve attempt) | integration | same test module as CAL-01 | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | D-09 (per-campaign table extension, not a numbered requirement) | Visibility of unapproved/rejected data | Non-staff sees `approved` and `rejected` rows, not `pending_review`, on the per-campaign table; staff continues to see every row | integration | extend `solsys_code/tests/test_campaign_views.py` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | D-01/D-02 (approval-queue access, not a numbered requirement) | Access control — see Security Domain below | Anonymous/non-staff GET to approval-queue and approve/reject URLs redirects (never 200 with content) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestStaffGating` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Task ID/Plan/Wave columns are placeholders — the planner assigns concrete task IDs when creating PLAN.md files and must update this table (or the plan's own `must_haves`) accordingly.

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_submission.py` — new test module covering SUBMIT-01, SUBMIT-04, SUBMIT-05 (does not exist yet)
- [ ] `solsys_code/tests/test_campaign_approval.py` — new test module covering SUBMIT-03, CAL-01, CAL-02, CAL-03, and staff-gating for the approval-queue/action views (does not exist yet)
- [ ] Extend `solsys_code/tests/test_campaign_views.py` — D-09 non-staff visibility filter (existing module from Phase 15)
- [ ] No new test framework install needed — `django.test.TestCase` + `django.core.mail` (`outbox`) are already available; no fixtures beyond `TargetList.objects.create(...)` + `User.objects.create_user(..., is_staff=True)` (already established in `test_campaign_views.py` from Phase 15)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Approval-queue page layout (pending + recently-decided sections) is legible and the Approve/Reject actions are unambiguous | D-01/D-02 | Layout clarity and visual affordance are subjective judgments; functional presence of both sections and correct action wiring are covered by automated tests above, but visual layout review belongs to UI-SPEC.md / `/gsd-ui-review`, not this functional test suite | Render the approval-queue page in a browser as a staff user, confirm pending and recently-decided rows are visually distinguishable and Approve/Reject controls are discoverable |
| Submission form's honeypot field is genuinely invisible to a sighted human user (not just `display:none` triggering an accessibility complaint) | SUBMIT-04 | Visual/accessibility inspection of a hidden form field is not practical to assert in an integration test | Render the submission form in a browser, confirm no visible extra field, and inspect the rendered HTML for an appropriately hidden/labeled honeypot input |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
