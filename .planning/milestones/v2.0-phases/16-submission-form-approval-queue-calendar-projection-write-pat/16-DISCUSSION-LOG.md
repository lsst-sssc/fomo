# Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-03
**Phase:** 16-submission-form-approval-queue-calendar-projection-write-path
**Areas discussed:** Approval queue interface, Staff notification target, Submission form shape, Post-approval visibility scope

---

## Approval queue interface

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated staff approval-queue page | A FOMO view listing `pending_review` rows with Approve/Reject buttons, reachable from the Campaigns nav | ✓ |
| Django admin bulk actions only | Add 'Approve selected'/'Reject selected' admin actions on `CampaignRunAdmin` | |
| Both — admin actions AND a dedicated queue page | Admin bulk actions for power users plus a friendlier dedicated page | |

**User's choice:** Dedicated staff approval-queue page (Recommended)
**Notes:** ROADMAP.md's `UI hint: yes` and REQUIREMENTS.md's existing exclusion of third-party moderation packages both pointed toward a first-party dedicated page.

| Option | Description | Selected |
|--------|-------------|----------|
| Pending-only queue | Page lists only `pending_review` rows | |
| Pending queue + recent decisions history | Page shows the pending queue plus a "recently decided" section | ✓ |

**User's choice:** Pending queue + recent decisions history
**Notes:** Lets staff spot-check or catch a mis-click without leaving the page.

---

## Staff notification target

| Option | Description | Selected |
|--------|-------------|----------|
| Every is_staff user's email | `User.objects.filter(is_staff=True, email__isnull=False).exclude(email='')` | ✓ |
| A configurable address via settings | New `CAMPAIGN_NOTIFICATION_EMAIL`-style setting | |

**User's choice:** Every is_staff user's email (Recommended)
**Notes:** Matches FOMO's existing `is_staff`-only gating convention; no `settings.ADMINS` currently configured.

| Option | Description | Selected |
|--------|-------------|----------|
| Bare ping + link to queue | Email says a new run was submitted, pending review, with a link | ✓ |
| Full submission details in the email | Telescope, dates, submitter contact info directly in the email | |

**User's choice:** Bare ping + link to queue (Recommended)
**Notes:** Avoids putting submitter PII into email infrastructure outside FOMO's own PII-gating boundary (consistent with VIEW-03).

---

## Submission form shape

| Option | Description | Selected |
|--------|-------------|----------|
| Intake-relevant subset | campaign + telescope_instrument, site_raw, obs_date, ut_start/end, filters_bandpass, observation_details, open_to_collaboration, contact_person, contact_email, comments | ✓ |
| Every CampaignRun field | All ~15 model fields including post-observation fields | |

**User's choice:** Intake-relevant subset (Recommended)
**Notes:** Excludes run_status/observation_outcome/weather/publication_plans/site(FK)/site_needs_review — staff/post-observation fields.

| Option | Description | Selected |
|--------|-------------|----------|
| Require both contact fields | Form-level validation requires contact_person + contact_email; DB column stays optional | ✓ |
| Keep fully optional, matching the model | Submitter can submit with zero contact info | |

**User's choice:** Require both contact fields (Recommended)
**Notes:** Staff must always be able to reach the submitter.

| Option | Description | Selected |
|--------|-------------|----------|
| Free-text site_raw only, resolve on approval | One free-text 'Site' field; FK resolution runs on approval, reusing Phase 14's 3-tier resolution | ✓ |
| Dropdown of existing Observatory records | `<select>` of `Observatory.objects.all()` | |

**User's choice:** Free-text site_raw only, resolve on approval (Recommended)
**Notes:** A dropdown would exclude sites not yet in FOMO's Observatory table — exactly the case Phase 14's 3-tier resolution was built to handle.

---

## Post-approval visibility scope

| Option | Description | Selected |
|--------|-------------|----------|
| Approved-only for non-staff | Non-staff see only `approved` rows; pending AND rejected both hidden | |
| Approved + rejected visible, only pending hidden | Non-staff see approved and rejected rows, just not pending_review | ✓ |

**User's choice:** Approved + rejected visible, only pending hidden
**Notes:** Keeps Phase 15 D-06's original intent that rejected rows aren't hidden from the public table.

| Option | Description | Selected |
|--------|-------------|----------|
| Hide it from non-staff until it has >=1 approved run | Non-staff campaign list only includes TargetLists with >=1 approved CampaignRun | |
| List it regardless (current Phase 15 behavior, unchanged) | Keep D-03 exactly as-is | ✓ |

**User's choice:** List it regardless (current Phase 15 behavior, unchanged)
**Notes:** Operator chose to keep Phase 15's campaign-list behavior unchanged rather than add new list-page filtering logic.

---

## Claude's Discretion

- Honeypot field mechanics (hidden, non-required, non-"honeypot"-named field; silent drop on trip) — informed by web-research best practice, not a locked user decision beyond SUBMIT-04's requirement text.
- Approval atomicity implementation (conditional `.filter().update()` vs `select_for_update()`) — research (`SUMMARY.md` Pitfall 6) already recommends the conditional-update approach.
- Exact URL names/paths for the submission-form and approval-queue views.
- Exact crispy-forms layout/field ordering for the submission form.
- `EMAIL_BACKEND`/`EMAIL_HOST` configuration for local dev and tests.

## Deferred Ideas

None newly deferred. SUBMIT-06, SUBMIT-07, and VIEW-05 were mentioned as research context but are already tracked as v2 requirements in `.planning/REQUIREMENTS.md`.

### Reviewed Todos (not folded)
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — keyword overlap only, unrelated.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — keyword overlap only, already resolved per Phase 14/15 context.
