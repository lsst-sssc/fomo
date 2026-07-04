---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
verified: 2026-07-04T15:30:00Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 16: Submission Form, Approval Queue & Calendar Projection Verification Report

**Phase Goal:** Community intake with staff approval gate; approved runs project onto the
calendar (SUBMIT-01..05, CAL-01..03).
**Verified:** 2026-07-04T15:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Community member can submit a run via a web form; campaign mandatory, everything else optional (SUBMIT-01) | VERIFIED | `CampaignRunSubmissionForm` (`solsys_code/campaign_forms.py`) is a plain `forms.Form` (grep confirms zero `ModelForm` usages); `CampaignRunSubmissionView` at `campaigns:submit` creates a `CampaignRun` from cleaned data. Test suite: `test_campaign_submission.py::TestCampaignSubmission` (6 tests) + `test_campaign_forms.py` (10 tests), all passing (ran directly, not just claimed). |
| 2 | New submissions are pending and invisible on public views until approved (SUBMIT-02) | VERIFIED | Model default `approval_status=PENDING_REVIEW`; `CampaignRunTableView.get_queryset` non-staff branch does `.exclude(approval_status=PENDING_REVIEW)` at the queryset level (`campaign_views.py:87`) before `.values()`. `test_campaign_views.py::TestNonStaffPendingReviewHidden` (4 tests) confirms anonymous excludes pending but keeps approved+rejected; staff sees all. Ran directly — pass. |
| 3 | Staff can review and approve/reject pending runs; approval is atomic (double-approve is a no-op, proven by test) (SUBMIT-03) | VERIFIED | `CampaignRunDecisionView.post` uses a single `.filter(pk=pk, approval_status=PENDING_REVIEW).update(...)` conditional update. `test_campaign_approval.py::TestApproval.test_double_approve_is_noop` and `test_second_approve_surfaces_already_decided_warning` exercise this directly. Ran directly — pass. |
| 4 | The public form carries a honeypot field; bot submissions are dropped without processing (SUBMIT-04) | VERIFIED | `alt_contact_info` is `HiddenInput`, `clean_alt_contact_info` never raises; `form_valid` checks it FIRST and short-circuits to the identical thanks redirect with no create/email. `test_campaign_submission.py::TestHoneypot` (2 tests) + `test_campaign_forms.py` honeypot tests. Ran directly — pass. |
| 5 | Staff receive an email notification when a new submission lands (SUBMIT-05) | VERIFIED | `_notify_staff` emails every `is_staff=True` user with a non-blank email, PII-free body. `test_campaign_submission.py::TestStaffNotification` (5 tests, incl. no-PII assertion, blank-email exclusion, non-staff exclusion). Ran directly — pass. |
| 6 | Approving a run with telescope + date range creates/updates a paired CalendarEvent via `insert_or_create_calendar_event()` keyed `CAMPAIGN:{pk}` (CAL-01) | VERIFIED | `CampaignRunDecisionView.post` calls `insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields={...})` only when `telescope_instrument`/`ut_start`/`ut_end` all present; never constructs `CalendarEvent` directly (grep confirms 0 hits). `test_campaign_approval.py::TestCalendarProjection` (4 tests, incl. 3 negative cases for missing telescope/ut_start/ut_end). Ran directly — pass. |
| 7 | The paired CalendarEvent's target_list is set to the campaign's TargetList (CAL-02) | VERIFIED | `fields={'target_list': run.campaign, ...}` passed into the helper. Asserted directly in `test_approve_with_full_window_creates_calendar_event`. Ran directly — pass. |
| 8 | Re-approving or editing an unchanged run causes no duplicate events and no `modified` churn (CAL-03) | VERIFIED | Double-approve matches 0 rows in the conditional update, so the projection block is never re-entered. `insert_or_create_calendar_event` itself is also no-churn (unchanged fields → no save). `test_campaign_approval.py::TestCalendarNoChurn.test_second_approve_leaves_event_count_and_modified_unchanged` asserts both event count and `modified` timestamp unchanged after a second approve. Ran directly — pass. |

**Score:** 8/8 truths verified (0 present-but-behavior-unverified)

### Code-Review Fix Verification (CR-01, WR-01, WR-03 — not must_haves, but explicitly requested)

| Finding | Claimed Fix | Verified In Code | Behavioral Proof |
|---------|-------------|-------------------|-------------------|
| CR-01 (Critical): approve side-effects not transactionally coupled — mid-flow failure leaves row permanently "approved" with no calendar event | Commit `4241b69`: wrap `resolve_site()` + `run.save()` + `insert_or_create_calendar_event()` in `try/except Exception`; on failure, explicitly revert `approval_status` back to `PENDING_REVIEW`, `logger.exception(...)`, `messages.error(...)` | Present at `campaign_views.py:283-323` — `except Exception:` block calls `CampaignRun.objects.filter(pk=pk).update(approval_status=PENDING_REVIEW)`, logs, and messages the user. Matches the commit's claimed diff (`git show 4241b69`, 44 insertions/23 deletions in `campaign_views.py`). | **Independently exercised** with a temporary, non-committed test that monkeypatches `resolve_site` to raise `RuntimeError` mid-approve: confirmed the row reverts to `PENDING_REVIEW`, no `CalendarEvent` is created, and a subsequent approve POST succeeds cleanly (row becomes `APPROVED` with a `CalendarEvent`, proving the row is not permanently stuck). Test file was deleted after verification — not part of the committed suite (REVIEW-FIX itself flags this path as "requires human verification... consider adding a regression test," which is still valid follow-up advice going forward, but the mechanism itself now has direct behavioral proof from this verification run, not just static code reading). |
| WR-01: misleading "already decided" message for a nonexistent pk | Commit `53358f8`: distinguish `CampaignRun.objects.filter(pk=pk).exists()` (existing-but-decided vs. never-existed) | Present at `campaign_views.py:327-334` — `elif CampaignRun.objects.filter(pk=pk).exists(): messages.warning(...) else: messages.error('This run no longer exists.')` | Logic read directly; consistent with commit diff (`git show 53358f8`, 7 insertions/1 deletion). |
| WR-03: dead `NoReverseMatch` fallback | Commit `a2975c9`: remove `try/except NoReverseMatch`, call `reverse('campaigns:approval_queue')` directly | Present — `_notify_staff` at `campaign_views.py:198` calls `reverse('campaigns:approval_queue')` unconditionally; `grep -n "NoReverseMatch" solsys_code/campaign_views.py` returns 0 hits (only a comment referencing the old name, no import or except clause). | Confirmed by grep + read; consistent with commit diff (5 insertions/10 deletions). |
| WR-02 (deliberately skipped): public form's `campaign` field exposes every `TargetList` | Documented in `16-REVIEW-FIX.md` as an accepted product-scope decision — scoping the queryset would break the "brand-new campaign's first run" use case (D-10) | Confirmed unfixed: `campaign_forms.py:19` still has `queryset=TargetList.objects.all()` | Accepted per task instructions as a known, non-blocking gap — does not contradict any must_have truth in the 4 plans (none of the must_haves claim campaign-list scoping). No action taken. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/mixins.py` | `StaffRequiredMixin`, dispatch-level `is_staff` gate | VERIFIED | Present, `user_passes_test(lambda u: u.is_staff)` on `dispatch`; used by `ApprovalQueueView`/`CampaignRunDecisionView`. |
| `solsys_code/campaign_forms.py` | Plain `forms.Form` + honeypot | VERIFIED | `CampaignRunSubmissionForm(forms.Form)`, 0 `ModelForm` references, honeypot present and non-raising. |
| `solsys_code/campaign_views.py::CampaignRunSubmissionView + _notify_staff` | Submission view + staff email | VERIFIED | Present, wired to `campaigns:submit`, tested. |
| `solsys_code/campaign_views.py::ApprovalQueueView + CampaignRunDecisionView` | Staff queue + atomic decision endpoint | VERIFIED | Present, `StaffRequiredMixin` gates both, `http_method_names=['post']` on decision view. |
| `solsys_code/campaign_tables.py::ApprovalQueueTable` | Actions column with Approve/Reject | VERIFIED | Present, CSRF token minted via `get_token(request)`, `show_actions` toggle for pending vs. decided tables. |
| `src/templates/campaigns/campaignrun_submit_form.html` | Public submission form template | VERIFIED | Renders crispy form; no honeypot leakage. |
| `src/templates/campaigns/submission_thanks.html` | Identical genuine/honeypot confirmation page | VERIFIED | No `{% if %}` branching on honeypot/status (grep confirms 0 hits). |
| `src/templates/campaigns/approval_queue.html` | Two-section staff queue page | VERIFIED | Pending + Recently Decided sections present, `{% render_table %}` used for both. |
| `src/templates/campaigns/campaign_list.html` | Submit button + staff pending banner | VERIFIED | Header row button to `campaigns:submit`; banner gated on `request.user.is_staff and pending_count`, links to `campaigns:approval_queue`. |
| `src/templates/campaigns/campaignrun_table.html` | Submit button (no banner) | VERIFIED | Header row button present; no pending banner (per plan, banner is list-page-only). |
| `solsys_code/tests/test_campaign_forms.py` | 10 tests | VERIFIED | Ran directly — 10/10 pass. |
| `solsys_code/tests/test_campaign_submission.py` | 13 tests | VERIFIED | Ran directly — pass, folded into the 58-test combined run below. |
| `solsys_code/tests/test_campaign_approval.py` | 14 tests | VERIFIED | Ran directly — pass. |
| `solsys_code/tests/test_campaign_views.py` | Extended with D-09 tests | VERIFIED | Ran directly — pass, includes `TestNonStaffPendingReviewHidden` and `TestCampaignListView.test_pending_count_in_context`. |

**Combined direct test run:** `python manage.py test solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_views` → **Ran 58 tests — OK** (executed by this verifier, not taken from SUMMARY claims).

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `campaigns:submit` URL | `CampaignRunSubmissionView` | `path('submit/', CampaignRunSubmissionView.as_view(), name='submit')` | WIRED | Confirmed in `campaign_urls.py`. |
| `campaigns:submission_thanks` URL | `TemplateView` | `path('submission-thanks/', ...)` | WIRED | Confirmed in `campaign_urls.py`. |
| `campaigns:approval_queue` / `campaigns:decide` URLs | `ApprovalQueueView` / `CampaignRunDecisionView` | `path(...)` entries | WIRED | Confirmed in `campaign_urls.py`; both gated by `StaffRequiredMixin`. |
| `CampaignRunDecisionView.post` (approve) | `calendar_utils.insert_or_create_calendar_event` | Direct call, never `CalendarEvent.objects.create()`/`CalendarEvent(...)` | WIRED | `grep -c "CalendarEvent.objects.create\|CalendarEvent("` in `campaign_views.py` returns 0. |
| `CampaignRunDecisionView.post` (approve) | `campaign_utils.resolve_site` | Direct call for site resolution | WIRED | Confirmed at `campaign_views.py:286`. |
| Non-staff `CampaignRunTableView.get_queryset` | pending-hiding filter | `.exclude(approval_status=PENDING_REVIEW)` before `.values()` | WIRED | Confirmed at `campaign_views.py:87-88`. |
| `campaign_list.html` banner | `campaigns:approval_queue` | `{% url %}` inside staff+pending_count guard | WIRED | Confirmed. |
| `campaign_list.html` / `campaignrun_table.html` buttons | `campaigns:submit` | `{% url %}` | WIRED | Confirmed both templates. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Combined phase-16 test suite | `python manage.py test solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_views` | `Ran 58 tests ... OK` | PASS |
| CR-01 revert-on-failure (state-transition/rollback invariant, not covered by any committed test) | Temporary test: monkeypatch `resolve_site` to raise mid-approve, assert row reverts to `PENDING_REVIEW`, no `CalendarEvent`, and a follow-up approve succeeds cleanly | Row reverted, 0 events, second approve succeeded (`APPROVED`, 1 event) | PASS (test file removed after verification — not committed) |
| ruff lint on all phase-touched Python files | `ruff check solsys_code/campaign_views.py solsys_code/campaign_forms.py solsys_code/mixins.py solsys_code/campaign_tables.py solsys_code/campaign_urls.py src/fomo/settings.py` | All checks passed | PASS |
| ruff format --check on same files | `ruff format --check ...` | `src/fomo/settings.py` would reformat (1 file); all others clean | INFO — see below |

**Note on `ruff format --check src/fomo/settings.py`:** the single formatting diff (`ruff format --diff`) is a missing blank line after the module docstring at line 10, unrelated to the `EMAIL_BACKEND` addition (which is at line 385-387, correctly formatted). Confirmed pre-existing via `git log` (the docstring predates the Phase 16 EMAIL_BACKEND commit `7a1c030`; the blank-line issue was already present before this phase touched the file). Not introduced by this phase — informational only, not a blocker.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| SUBMIT-01 | 16-01, 16-02, 16-04 | Community member can submit a run via a web form | SATISFIED | Form + view + entry-point buttons, all tested. |
| SUBMIT-02 | 16-04 | New submissions pending/invisible until approved | SATISFIED | Model default + queryset exclude, tested. |
| SUBMIT-03 | 16-03 | Staff review/approve/reject; atomic; double-approve no-op | SATISFIED | Conditional `.update()`, tested directly. |
| SUBMIT-04 | 16-01, 16-02 | Honeypot; bot submissions dropped | SATISFIED | Non-raising honeypot + view short-circuit, tested. |
| SUBMIT-05 | 16-02 | Staff email notification on new submission | SATISFIED | `_notify_staff`, tested (incl. PII-free assertion). |
| CAL-01 | 16-03 | Approve with telescope+dates creates/updates CAMPAIGN:{pk} CalendarEvent via helper | SATISFIED | Verified in code and tested, incl. negative cases. |
| CAL-02 | 16-03 | Event's target_list = campaign's TargetList | SATISFIED | Verified directly. |
| CAL-03 | 16-03 | Re-approve/unchanged edit: no duplicate, no modified churn | SATISFIED | Verified directly (event count + modified timestamp assertions). |

No orphaned requirements: all 8 phase-mapped requirement IDs (SUBMIT-01..05, CAL-01..03) appear in at least one plan's `requirements` frontmatter and in REQUIREMENTS.md's traceability table (all marked "Complete").

### Anti-Patterns Found

None. Scanned all phase-modified Python and template files for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/stub markers — zero hits.

### Human Verification Required

None required to pass this phase — all must_haves have direct behavioral evidence from this verification run (not just SUMMARY claims). The following are pre-existing, explicitly-deferred manual UI checks noted in the plans' own `<verify>` sections (visual/UX only, not gating this verification):

1. **Honeypot field visual invisibility** — confirm `alt_contact_info` is genuinely hidden (not just `display:none` in a way a sighted user could still tab into) in a real browser render of `campaigns:submit`.
2. **Approval-queue layout legibility** — two-section layout, badge colors, Actions column spacing, and the native `confirm()` Reject dialog copy/behavior, in a real browser.
3. **Entry-point banner/button visual hierarchy** — spacing and Bootstrap alert styling on `campaign_list.html`, staff vs. anonymous view, light/dark theme.

### Gaps Summary

No gaps. All 8 roadmap success criteria (SUBMIT-01..05, CAL-01..03) have both static code evidence and passing automated tests, executed directly by this verifier (58/58 tests green). The one Critical code-review finding (CR-01) was independently re-verified with a targeted behavioral test (not present in the committed suite) proving the revert-on-failure recovery path actually works, not just that the code reads correctly. Both fixed Warnings (WR-01, WR-03) are confirmed present and correct. The one explicitly-skipped Warning (WR-02) is a documented, accepted product-scope decision that does not contradict any must_have truth in the four plans.

---

_Verified: 2026-07-04T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
