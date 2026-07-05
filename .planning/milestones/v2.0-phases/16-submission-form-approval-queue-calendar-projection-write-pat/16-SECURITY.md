---
phase: 16-submission-form-approval-queue-calendar-projection-write-pat
asvs_level: 1
block_on: high
threats_total: 9
threats_closed: 9
threats_open: 0
status: SECURED
audited: 2026-07-04
---

# Phase 16 Security Audit — Submission Form / Approval Queue / Calendar Projection Write Path

Retroactive security audit of an already-shipped phase (Plans 01-05, all complete). Threat
register was authored at plan time across the 5 PLAN.md files (`register_authored_at_plan_time:
true`) — this audit verifies each declared mitigation against the actual implementation and test
suite, it does not hunt for new threats.

## Threat Verification

| Threat ID | Category | Component | Severity | Disposition | Status | Evidence |
|-----------|----------|-----------|----------|-------------|--------|----------|
| T-16-01 | Denial of Service / Tampering | `CampaignRunSubmissionForm` honeypot + `form_valid` short-circuit | medium | mitigate | CLOSED | `solsys_code/campaign_forms.py:32,34-38` — `alt_contact_info` is `forms.HiddenInput`, `clean_alt_contact_info` returns the value without raising. `solsys_code/campaign_views.py:146-149` — `form_valid` checks `form.cleaned_data.get('alt_contact_info')` FIRST and returns the thanks redirect before any `.create()`/`send_mail()`/error. Proven live by `test_campaign_submission.py::TestHoneypot` (both tests pass: 0 rows created, 0 emails sent, identical redirect URL/status to a genuine submission). |
| T-16-02 | Tampering | Approve/reject atomic transition (`CampaignRunDecisionView`) | high | mitigate | CLOSED | `solsys_code/campaign_views.py:278-280` — single conditional `CampaignRun.objects.filter(pk=pk, approval_status=PENDING_REVIEW).update(approval_status=new_status)`; projection block gated on `updated_count == 1 and action == 'approve'` (line 282). Proven live by `test_campaign_approval.py::TestApproval.test_double_approve_is_noop` and `test_second_approve_surfaces_already_decided_warning` — second POST leaves `approval_status` and `CalendarEvent.objects.count()` unchanged and surfaces the "already decided" warning. |
| T-16-03 | Information Disclosure / Elevation of Privilege | `StaffRequiredMixin` gating `ApprovalQueueView`/`CampaignRunDecisionView`; PII columns (contact_person/contact_email); staff pending-banner link; 16-05 column reorder | high | mitigate (mostly) / accept (16-05 slice) | CLOSED | `solsys_code/mixins.py` — `StaffRequiredMixin.dispatch` wrapped in `@method_decorator(user_passes_test(lambda u: u.is_staff))`, no custom `login_url` (redirects to `settings.LOGIN_URL`). `solsys_code/campaign_views.py:208` (`ApprovalQueueView(StaffRequiredMixin, TemplateView)`) and `:260` (`CampaignRunDecisionView(StaffRequiredMixin, View)`) both apply it. `src/templates/campaigns/campaign_list.html:10` gates the pending-review banner/link on `{% if request.user.is_staff and pending_count %}` — convenience only, the real boundary is the mixin on the linked view. 16-05's `ApprovalQueueTable.Meta` exclude/sequence reorders columns but does not touch `StaffRequiredMixin` or add a new endpoint — accept disposition holds (verified: `ApprovalQueueTable` is only ever instantiated inside the already-gated `ApprovalQueueView.get_context_data`). Proven live by `TestStaffGating` (5 tests): anonymous/non-staff GET the queue and POST to decide both 302 with no state change; staff GET succeeds 200. |
| T-16-04 | Information Disclosure | `_notify_staff` email body/subject | high | mitigate | CLOSED | `solsys_code/campaign_views.py:199-204` — `subject='FOMO: new campaign run submission pending review'`, `message=f'A new run submission is pending review: {queue_url}'` — no contact/telescope/campaign-name interpolation anywhere in the string. Proven live by `test_campaign_submission.py::TestStaffNotification.test_email_contains_no_pii`, which asserts `contact_person`/`contact_email`/`telescope_instrument`/campaign name are all absent from both `subject` and `body`. |
| T-16-05 | Tampering | Submission/approval POST forms (CSRF) | medium | mitigate | CLOSED | `src/fomo/settings.py:79` — `CsrfViewMiddleware` active in `MIDDLEWARE`. Submission form: `{% crispy form %}` (`campaignrun_submit_form.html:12`) renders via `crispy_bootstrap4`'s `whole_uni_form.html`, which emits `{% csrf_token %}` whenever `form_method == 'post'` and `disable_csrf` is unset (verified in installed package: `crispy_bootstrap4/templates/bootstrap4/whole_uni_form.html:5-6`) — `CampaignRunSubmissionForm.__init__` never sets `disable_csrf`, so the token is emitted. Approval-queue per-row mini-forms: `solsys_code/campaign_tables.py:176,180,185` — CSRF token minted via `get_token(self.request)` and injected as a hidden `csrfmiddlewaretoken` input in each Approve/Reject `<form>`. |
| T-16-06 | Tampering | GET-triggered state change on decision view | medium | mitigate | CLOSED | `solsys_code/campaign_views.py:270` — `CampaignRunDecisionView.http_method_names = ['post']`, so Django's dispatcher 405s any GET before `post()` runs. Proven live by the absence of a GET-based state-change path in any test, and structurally by Django's `View.http_method_not_allowed` behavior for methods outside `http_method_names`. |
| T-16-07 | Information Disclosure | `CampaignRunTableView.get_queryset` non-staff branch | high | mitigate | CLOSED | `solsys_code/campaign_views.py:84-88` — non-staff branch inserts `qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)` BEFORE `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)`, i.e. pending rows never enter the non-staff SQL SELECT (queryset-level, not a template conditional). Proven live by `test_campaign_views.py::TestNonStaffPendingReviewHidden` (anonymous excludes pending, keeps approved+rejected, correct total count; staff sees all). |
| T-16-08 | Denial of Service | `send_mail` failure breaking the request | low | mitigate | CLOSED | `solsys_code/campaign_views.py:204` — `send_mail(..., fail_silently=True)`. A mail-backend exception cannot propagate to break the submission request. |
| T-16-09 | Tampering | CalendarEvent namespace collision | medium | mitigate | CLOSED | `solsys_code/campaign_views.py:297-307` — projection routed exclusively through `insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields={...})`; no direct `CalendarEvent(...)`/`CalendarEvent.objects.create(...)` call anywhere in `campaign_views.py` (`grep -c` returns 0). LCO/Gemini/classical sync commands key on real `http(s)://` URLs (`sync_lco_observation_calendar.py:190`, `sync_gemini_observation_calendar.py:163`) which can never collide with the `CAMPAIGN:` prefix — distinct namespace confirmed. `insert_or_create_calendar_event`'s `get_or_create` + no-op-if-unchanged contract (`calendar_utils.py:323-332`) also gives CAL-03 (no duplicate, no `modified` churn on re-approve), though in practice T-16-02's `updated_count==1` guard already prevents the projection code from re-running on a second approve. Proven live by `TestCalendarNoChurn.test_second_approve_leaves_event_count_and_modified_unchanged`. |
| T-16-SC | Tampering (supply chain) | Package installs | low | accept | CLOSED (accepted) | `git log --oneline -- pyproject.toml` shows no commit under any of the phase's 5 plans (`16-01`..`16-05`) touches `pyproject.toml` — no new dependency was introduced across the phase, consistent with each plan's RESEARCH.md Package Legitimacy Audit finding "N/A". Accepted risk, no action required. |

## Unregistered Flags

None. No SUMMARY.md in this phase contains a `## Threat Flags` section reporting new attack
surface discovered during implementation; all 5 SUMMARY.md files' "Deviations from Plan" sections
were reviewed and none introduce new trust boundaries beyond what's already registered above
(the CR-01 rollback-on-exception behavior added to `CampaignRunDecisionView.post` — reverting
`approval_status` to `PENDING_REVIEW` if site resolution or calendar projection raises — is a
robustness/data-integrity fix, not a new attack surface; it does not weaken T-16-02's atomicity
guarantee, since the initial `.update()` remains the single point of state transition and the
rollback only fires on unexpected internal failure, not on attacker input).

## Verification Method

- ASVS Level 1 (grep/pattern-presence verification per threat, per `<config>`).
- All 9 threats were also verified live rather than by static grep alone: ran
  `python3 manage.py test solsys_code.tests.test_campaign_forms
  solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_approval
  solsys_code.tests.test_campaign_views` — 61/61 tests passed, exercising the honeypot
  short-circuit, staff gating (anonymous/non-staff/staff), double-approve no-op, PII-free
  notification email, D-09 non-staff visibility filter, and calendar no-churn behaviors described
  above.
- CSRF-emission claim for `{% crispy form %}` was verified against the actually-installed
  `crispy_bootstrap4` package source (`whole_uni_form.html`), not assumed from crispy-forms
  documentation.
- Supply-chain accept disposition was verified against `git log -- pyproject.toml`, not merely
  trusted from RESEARCH.md prose.

## Accepted Risks Log

| Threat ID | Disposition | Rationale | Re-review trigger |
|-----------|-------------|-----------|--------------------|
| T-16-SC | accept | No new third-party packages were introduced by any of the phase's 5 plans; all functionality (honeypot, staff gating, atomic update, calendar projection) uses already-installed Django/crispy-forms/django-tables2 primitives. | Re-review if a future phase adds a new dependency to this write path. |
| T-16-03 (16-05 slice only) | accept | The 16-05 gap-closure plan reorders/trims `ApprovalQueueTable` columns (moves `actions` first, excludes 3 always-blank post-observation columns) but does not change `StaffRequiredMixin` gating, add a new endpoint, or expose any column that wasn't already rendered inside the staff-gated `ApprovalQueueView`. No new exposure. | Re-review if `ApprovalQueueTable` is ever reused outside a `StaffRequiredMixin`-gated view. |

## Gaps / Follow-ups

None. All 9 registered threats resolved to CLOSED; `threats_open` = 0 (no threat at or above the
`block_on: high` threshold is open). Phase 16 is cleared to ship from a security standpoint.
