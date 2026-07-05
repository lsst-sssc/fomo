---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
verified: 2026-07-04T15:58:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 8/8
  gaps_closed:
    - "UAT Test 14 (16-UAT.md): staff approval queue forced horizontal scrolling past 16
       mostly-blank columns to reach Approve/Reject — closed by 16-05's Meta.exclude +
       Meta.sequence on ApprovalQueueTable."
  gaps_remaining: []
  regressions: []
---

# Phase 16: Submission Form, Approval Queue & Calendar Projection Verification Report

**Phase Goal:** Community members (PIs and external observers) can submit runs that stay hidden
until a staff member approves them, and an approved run with a telescope and date range appears
on the shared calendar.
**Verified:** 2026-07-04T15:58:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (16-05 closed the single UAT Test 14 gap; this run
re-verifies the full phase goal end-to-end, not just the delta)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Community member can submit a run via a web form; campaign mandatory, everything else optional (SUBMIT-01) | VERIFIED | `CampaignRunSubmissionForm` is a plain `forms.Form` (0 `ModelForm` hits); `campaigns:submit` → `CampaignRunSubmissionView.form_valid` creates a `CampaignRun`. `test_campaign_submission.py` + `test_campaign_forms.py` pass (ran directly). |
| 2 | New submissions are pending and invisible on public views until approved (SUBMIT-02) | VERIFIED | Model default `approval_status=PENDING_REVIEW`; `CampaignRunTableView.get_queryset` non-staff branch does `.exclude(approval_status=PENDING_REVIEW)` before `.values()`. `test_campaign_views.py::TestNonStaffPendingReviewHidden` passes. |
| 3 | Staff can review and approve/reject pending runs; approval is atomic (double-approve is a no-op, proven by test) (SUBMIT-03) | VERIFIED | `CampaignRunDecisionView.post` uses `.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)`. `test_campaign_approval.py::TestApproval.test_double_approve_is_noop` and `TestCalendarNoChurn` pass. |
| 4 | The public form carries a honeypot field; bot submissions are dropped without processing (SUBMIT-04) | VERIFIED | `alt_contact_info` is `HiddenInput`, `clean_alt_contact_info` never raises; `form_valid` checks it first and short-circuits to the identical thanks redirect with no create/email. `TestHoneypot` passes. |
| 5 | Staff receive an email notification when a new submission lands (SUBMIT-05) | VERIFIED | `_notify_staff` emails every `is_staff=True` user with a non-blank email, PII-free body/subject. `TestStaffNotification` (incl. no-PII, blank-email-excluded, non-staff-excluded) passes. |
| 6 | Approving a run with telescope + date range creates/updates a paired CalendarEvent via `insert_or_create_calendar_event()` keyed `CAMPAIGN:{pk}` (CAL-01) | VERIFIED | `CampaignRunDecisionView.post` calls the helper only when `telescope_instrument`/`ut_start`/`ut_end` all present; 0 direct `CalendarEvent(...)`/`.create()` calls (grep-confirmed). `TestCalendarProjection` (incl. 3 negative cases) passes. |
| 7 | The paired CalendarEvent's target_list is set to the campaign's TargetList (CAL-02) | VERIFIED | `fields={'target_list': run.campaign, ...}` passed to the helper; asserted directly in `test_approve_with_full_window_creates_calendar_event`. |
| 8 | Re-approving or editing an unchanged run causes no duplicate events and no `modified` churn (CAL-03) | VERIFIED | Double-approve matches 0 rows in the conditional update, so the projection block never re-enters. `TestCalendarNoChurn.test_second_approve_leaves_event_count_and_modified_unchanged` asserts both event count and `modified` unchanged. |

**Score:** 8/8 truths verified (0 present-but-behavior-unverified)

### Gap-Closure Verification (16-05 — UAT Test 14)

| Must-have (16-05 frontmatter) | Status | Evidence |
|---|---|---|
| `ApprovalQueueTable` no longer renders `weather`/`observation_outcome`/`publication_plans` | VERIFIED | Live import check: `[c.name for c in ApprovalQueueTable([]).columns]` contains none of the three. `Meta.exclude = ('weather', 'observation_outcome', 'publication_plans')` present at `solsys_code/campaign_tables.py:154`. |
| `actions` is the first column of `ApprovalQueueTable` (Approve/Reject reachable without scrolling) | VERIFIED | Same live check: column list is `['actions', 'approval_status', 'telescope_instrument', 'site', 'obs_date', 'ut_start', 'ut_end', 'filters_bandpass', 'run_status', 'open_to_collaboration', 'observation_details', 'comments', 'contact_person', 'contact_email']` — `actions` leads. `Meta.sequence` at lines 155-164 uses the `'...'` ellipsis token. |
| `CampaignRunTable` completely unchanged — still all 16 spreadsheet-parity columns (Phase 15 D-09 preserved) | VERIFIED | Same live check: `CampaignRunTable([]).columns` = 16 columns including `weather`/`observation_outcome`/`publication_plans`, no `actions`. `CampaignRunTable.Meta` (lines 53-76) has no `exclude`/`sequence` added — byte-identical to pre-16-05. |
| Column-contract regression test exists and passes | VERIFIED | `TestApprovalQueueColumns` (3 tests) in `test_campaign_approval.py:179-199` — ran directly, all pass (part of the 17/17 module run below). |
| Fix scoped to `ApprovalQueueTable.Meta` only — no template/CSS change | VERIFIED | `git show ed7b12a` / `eb9bdaa` touch only `solsys_code/campaign_tables.py` and `solsys_code/tests/test_campaign_approval.py`; `src/templates/campaigns/approval_queue.html` untouched. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/mixins.py` | `StaffRequiredMixin` | VERIFIED | Present, `user_passes_test(lambda u: u.is_staff)` on `dispatch`. |
| `solsys_code/campaign_forms.py` | Plain `forms.Form` + honeypot | VERIFIED | `CampaignRunSubmissionForm(forms.Form)`, 0 `ModelForm` references. |
| `solsys_code/campaign_views.py` | Submission view, approval queue, decision endpoint, visibility filter | VERIFIED | All present, wired, tested (see truths above). |
| `solsys_code/campaign_tables.py::ApprovalQueueTable` | Actions column, trimmed/reordered post-16-05 | VERIFIED | `Meta.exclude`/`Meta.sequence` present; live column-order check confirms; `CampaignRunTable` untouched. |
| `src/templates/campaigns/*` (submit form, thanks, approval queue, list, table) | Entry points + queue UI | VERIFIED | All present; grep-confirmed `campaigns:submit`/`campaigns:approval_queue` wiring; no honeypot leakage in thanks template. |
| `solsys_code/tests/test_campaign_forms.py` | 10 tests | VERIFIED | Ran directly — pass. |
| `solsys_code/tests/test_campaign_submission.py` | 13 tests | VERIFIED | Ran directly — pass (part of combined 303-test run). |
| `solsys_code/tests/test_campaign_approval.py` | 17 tests (14 original + 3 new column-contract) | VERIFIED | Ran directly: `Ran 17 tests ... OK`. |
| `solsys_code/tests/test_campaign_views.py` | D-09 tests | VERIFIED | Ran directly, part of full-suite run. |

**Combined direct test runs (this verification):**
- `python manage.py test solsys_code.tests.test_campaign_approval` → **Ran 17 tests — OK** (includes new `TestApprovalQueueColumns`).
- `python manage.py test solsys_code` (full Django app suite) → **Ran 303 tests — OK**.
- `python -m pytest` → **1 passed**.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `campaigns:submit` URL | `CampaignRunSubmissionView` | `path('submit/', ...)` | WIRED | Resolves live: `/campaigns/submit/`. |
| `campaigns:submission_thanks` URL | `TemplateView` | `path('submission-thanks/', ...)` | WIRED | Resolves live: `/campaigns/submission-thanks/`. |
| `campaigns:approval_queue` / `campaigns:decide` URLs | `ApprovalQueueView` / `CampaignRunDecisionView` | `path(...)` | WIRED | Resolve live: `/campaigns/approval-queue/`, `/campaigns/1/decide/`. Both gated by `StaffRequiredMixin`. |
| `CampaignRunDecisionView.post` (approve) | `calendar_utils.insert_or_create_calendar_event` | Direct call | WIRED | 0 direct `CalendarEvent(...)`/`.create()` hits in `campaign_views.py`. |
| `ApprovalQueueTable.Meta` (exclude/sequence) | `CampaignRunTable.Meta` | `class Meta(CampaignRunTable.Meta)` inheritance, overridden not replaced | WIRED | Live column-list check confirms both the override and the D-09 regression guard hold simultaneously. |
| Non-staff `CampaignRunTableView.get_queryset` | pending-hiding filter | `.exclude(approval_status=PENDING_REVIEW)` before `.values()` | WIRED | Confirmed at `campaign_views.py`. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ApprovalQueueTable column trim/reorder (this verifier's own run, not SUMMARY claim) | Live Python import + column-list assertion (see above) | `actions` leads, 3 columns excluded, `CampaignRunTable` unaffected | PASS |
| 16-05 column-contract test module | `python manage.py test solsys_code.tests.test_campaign_approval` | `Ran 17 tests ... OK` | PASS |
| Full Django app suite (phase gate) | `python manage.py test solsys_code` | `Ran 303 tests ... OK` | PASS |
| pytest suite | `python -m pytest` | `1 passed` | PASS |
| ruff lint on 16-05-touched files | `ruff check solsys_code/campaign_tables.py solsys_code/tests/test_campaign_approval.py` | All checks passed | PASS |
| ruff format --check on 16-05-touched files | `ruff format --check solsys_code/campaign_tables.py solsys_code/tests/test_campaign_approval.py` | 2 files already formatted | PASS |
| URL resolution (submit/thanks/queue/decide) | `reverse(...)` one-liner | All 4 resolve | PASS |
| Debt-marker scan on 16-05-touched files | `grep -E "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER"` | 0 hits | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| SUBMIT-01 | 16-01, 16-02, 16-04 | Community member can submit a run via a web form | SATISFIED | Form + view + entry-point buttons, all tested. |
| SUBMIT-02 | 16-04 | New submissions pending/invisible until approved | SATISFIED | Model default + queryset exclude, tested. |
| SUBMIT-03 | 16-03, 16-05 | Staff review/approve/reject; atomic; double-approve no-op | SATISFIED | Conditional `.update()`, tested; 16-05 also tags SUBMIT-03 for the triage-usability fix. |
| SUBMIT-04 | 16-01, 16-02 | Honeypot; bot submissions dropped | SATISFIED | Non-raising honeypot + view short-circuit, tested. |
| SUBMIT-05 | 16-02 | Staff email notification on new submission | SATISFIED | `_notify_staff`, tested (incl. PII-free assertion). |
| CAL-01 | 16-03 | Approve with telescope+dates creates/updates CAMPAIGN:{pk} CalendarEvent via helper | SATISFIED | Verified in code and tested, incl. negative cases. |
| CAL-02 | 16-03 | Event's target_list = campaign's TargetList | SATISFIED | Verified directly. |
| CAL-03 | 16-03 | Re-approve/unchanged edit: no duplicate, no modified churn | SATISFIED | Verified directly. |

All 8 phase-mapped requirement IDs (SUBMIT-01..05, CAL-01..03) appear in REQUIREMENTS.md marked
"Complete" and in at least one plan's `requirements` frontmatter (16-05 additionally cites
SUBMIT-03 for its usability fix). No orphaned requirements.

### Anti-Patterns Found

None. Scanned `solsys_code/campaign_tables.py` and `solsys_code/tests/test_campaign_approval.py`
(the two files 16-05 touched) for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` — zero hits.
The broader phase's prior anti-pattern scan (16-VERIFICATION.md initial pass) also found none;
nothing in 16-05 reintroduces any.

### Code-Review Cross-Reference (16-REVIEW.md, informational — not a phase gate)

A fresh deep code review ran after 16-05 landed: 0 critical, 3 warnings, 4 info. All warnings are
pre-existing and unrelated to the 16-05 change itself:
- WR-01: CR-01's approve-failure recovery (from a prior review cycle) reverts `approval_status`
  but not `site`/`site_needs_review` — a residual defect in a fix from before this phase's UAT gap.
- WR-02: no test exercises the CR-01 exception/revert path.
- WR-03: public form's `campaign` field exposes every `TargetList` — carried-forward, deliberately
  deferred product-scope decision.

The review explicitly confirmed 16-05's own change is sound: `django_tables2`'s `TableOptions`
supports combining inherited `Meta.fields` with a subclass `exclude`, `TestApprovalQueueColumns`
passes, and `CampaignRunTable` is untouched. None of the 7 findings block this phase's must-haves;
they are advisory follow-up items, not gaps in SUBMIT-01..05/CAL-01..03.

### Human Verification Required

None required to pass this phase. UAT Test 14 (the only outstanding human-flagged item from
16-UAT.md) is closed by 16-05 and verified above with live code evidence, not just SUMMARY claims.
The other 18 UAT tests already passed in the prior human verification round (16-UAT.md) and are
not re-litigated here since 16-05 touched none of the surfaces they covered (submission form,
honeypot, staff notification, entry-point banners/buttons).

### Gaps Summary

No gaps. All 8 roadmap success criteria (SUBMIT-01..05, CAL-01..03) remain satisfied after the
16-05 gap-closure plan, verified independently by this verifier (not from SUMMARY claims):
column contract enforced live via direct Python import, full 17-test approval module green,
full 303-test Django suite green, pytest suite green, ruff clean. The single UAT gap (Test 14,
approval-queue horizontal scrolling) is closed: `actions` now leads `ApprovalQueueTable`'s column
order and the three structurally-blank post-observation columns are excluded, while
`CampaignRunTable` (Phase 15's spreadsheet-parity read path) is confirmed byte-for-byte unchanged
both by direct column-list comparison and by a dedicated regression test.

---

_Verified: 2026-07-04T15:58:00Z_
_Verifier: Claude (gsd-verifier)_
