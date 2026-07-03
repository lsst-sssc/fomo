# Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path) - Research

**Researched:** 2026-07-03
**Domain:** Django forms (crispy-forms) + staff-gated moderation workflow + calendar projection, on top of the existing `CampaignRun`/`CalendarEvent` models
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Approval queue interface**
- **D-01:** Staff review pending submissions through a **dedicated staff-facing approval-queue
  page** (a FOMO view, not Django admin bulk actions, not both) with Approve/Reject actions,
  reachable from the existing "Campaigns" navbar entry (`SolsysCodeConfig.nav_items()`).
- **D-02:** The approval-queue page shows **both** the pending queue (`approval_status
  ='pending_review'`) **and** a "recently decided" section (recently approved/rejected) so staff
  can spot-check or catch a mis-click without leaving the page.

**Staff notification (SUBMIT-05)**
- **D-03:** Notification recipients are **every `User` with `is_staff=True` and a non-empty
  email** — `User.objects.filter(is_staff=True).exclude(email='')`. No new settings-based
  address is needed.
- **D-04:** The email body is a **bare "new submission pending review" ping with a link to the
  approval-queue page** — it does **not** include submission details or submitter contact PII in
  the subject or body.

**Submission form shape (SUBMIT-01)**
- **D-05:** The public form exposes: `campaign` (required) + `telescope_instrument`, `site_raw`
  (D-07), `obs_date`, `ut_start`, `ut_end`, `filters_bandpass`, `observation_details`,
  `open_to_collaboration`, `contact_person`, `contact_email`, `comments`. **Excluded:**
  `run_status`, `observation_outcome`, `weather`, `publication_plans`, `site` (FK),
  `site_needs_review`.
- **D-06:** `contact_person` and `contact_email` are **required at the form-validation level**
  (not the DB level — both stay `blank=True` on the model).
- **D-07:** The form captures the observing site as **free text only** (`site_raw`) — no FK
  resolution at submission time. `CampaignRun.site` resolution runs when **staff approves**,
  reusing `campaign_utils.resolve_site`.

**Post-approval visibility scope (SUBMIT-02, extends Phase 15's table)**
- **D-09:** Non-staff (anonymous) visitors see `approved` AND `rejected` rows — only
  `pending_review` is hidden. Staff continue to see every row regardless of status.
- **D-10:** The campaigns **list** page is **not** changed by this phase.

### Claude's Discretion
- **Honeypot mechanics (SUBMIT-04):** a hidden, non-required, non-obviously-named form field
  (not literally `honeypot`) that silently drops the submission on trip — no `CampaignRun`
  created, no error shown to the bot, no notification email sent. A plain Django form field
  suffices; no third-party honeypot package.
- **Approval atomicity mechanism (SUBMIT-03):** research (`SUMMARY.md` Pitfall 6) already
  recommends a conditional atomic update (e.g.
  `CampaignRun.objects.filter(pk=pk, approval_status='pending_review').update(approval_status=
  'approved')`) so a double-approve is a proven no-op.
- Exact URL names/paths for the submission-form view and the approval-queue view.
- Exact crispy-forms layout/field ordering for the submission form.
- `EMAIL_BACKEND`/`EMAIL_HOST` configuration for local dev and tests.

### Deferred Ideas (OUT OF SCOPE)
None newly deferred by this discussion. SUBMIT-06 (self-service approval bypass), SUBMIT-07
(submitter status-check link), and VIEW-05 (submitter opt-in public contact display) are v2
requirements, not in scope for this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SUBMIT-01 | Community member can submit a run via a web form — campaign mandatory, all other fields optional | Standard Stack (plain `forms.Form`, not `ModelForm`); Code Examples "Submission form" |
| SUBMIT-02 | New submissions are pending and invisible on public views until approved | Architecture Pattern "D-09 queryset exclude"; Code Examples "Non-staff visibility filter" |
| SUBMIT-03 | Staff can review and approve/reject pending runs; approval is atomic (double-approve is a no-op, proven by test) | Common Pitfalls "Approval race condition"; Code Examples "Atomic conditional approve" |
| SUBMIT-04 | The public form carries a honeypot field; bot submissions are dropped without processing | Common Pitfalls "Honeypot must not tip off the bot"; Code Examples "Honeypot field + clean" |
| SUBMIT-05 | Staff receive an email notification when a new submission lands | Common Pitfalls "Email backend + DEFAULT_FROM_EMAIL"; Code Examples "Staff notification email" |
| CAL-01 | Approving a run with telescope + date range creates/updates a paired `CalendarEvent` via `insert_or_create_calendar_event()` keyed `CAMPAIGN:{pk}` | Architecture Pattern "Calendar projection trigger"; Code Examples "Calendar projection on approve" |
| CAL-02 | The paired `CalendarEvent.target_list` is set to the campaign's `TargetList` | Code Examples "Calendar projection on approve" (fields dict) |
| CAL-03 | Re-approving or editing an unchanged run causes no duplicate events and no `modified` churn | Architecture Pattern "CAL-03 is satisfied by the atomicity gate + `insert_or_create_calendar_event`'s no-churn contract" |
</phase_requirements>

## Summary

This phase adds no new dependencies and no new models. `CampaignRun` already has every field
the submission form needs (`campaign`, `telescope_instrument`, `site_raw`, `obs_date`,
`ut_start`, `ut_end`, `filters_bandpass`, `observation_details`, `open_to_collaboration`,
`contact_person`, `contact_email`, `comments`, `approval_status` defaulting to
`PENDING_REVIEW`), and `insert_or_create_calendar_event()` already implements the exact
create-or-update-with-no-churn contract CAL-01/03 need. The work is almost entirely new
view/form/URL/template code following patterns Phase 14/15 already established in this codebase
(`campaign_utils.py`'s "never raise for messy data" discipline, `campaign_views.py`'s
`is_staff`-gated queryset restriction, `insert_or_create_*` idempotent helpers).

Three genuine gotchas emerged from reading the actual models (not assumptions): (1)
`CampaignRun.telescope_instrument` has no `blank=True`, so the submission form **must** be a
plain `forms.Form` + explicit `.objects.create()`, never a `ModelForm` (which would wrongly
force it required); (2) `CalendarEvent.start_time`/`end_time` are both non-nullable, so the
"telescope + date range" trigger condition for CAL-01 should require `ut_start` **and** `ut_end`
both present, not just one; (3) `CampaignRun` has **no** `modified`/timestamp field at all (a
deliberate prior decision, per STATE.md's Phase 14 notes), so "recently decided" (D-02) cannot be
ordered by decision time — order by `-pk` instead, filtered to non-pending rows.

**Primary recommendation:** Add a plain `campaign_forms.py` (submission form) and extend
`campaign_views.py`/`campaign_urls.py`/`campaign_tables.py` with: a `CampaignRunSubmissionView`
(FormView, public), an `ApprovalQueueView` (staff-only, two independent
querysets/`CampaignRunTable` instances — pending + recently-decided), and two POST-only staff-only
action views (approve/reject) that use the atomic `.filter(pk=pk,
approval_status='pending_review').update(...)` pattern and call
`insert_or_create_calendar_event()` only when that update actually flips the row (`updated_count
== 1`) and the date-range condition is met.

## Architectural Responsibility Map

FOMO is a server-rendered Django monolith (Bootstrap4 + crispy-forms + django-tables2, no
client-side framework) — there is effectively no separate "Frontend Server (SSR)" tier distinct
from "API/Backend"; both collapse into the Django view layer.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Public submission form (render + validate) | API/Backend (Django `FormView`) | — | Server-rendered form, no client-side validation framework in this codebase |
| Honeypot spam detection | API/Backend (form `clean()`) | — | Must run server-side; a hidden field alone is a client-side no-op |
| Staff approval-queue display | API/Backend (Django view) | Database/Storage (query) | Two querysets built from `CampaignRun` |
| Approval atomic state transition | Database/Storage (conditional `UPDATE`) | API/Backend (view orchestration) | Atomicity comes from the single SQL statement, not app-level locking |
| Calendar projection (`CalendarEvent` create/update) | Database/Storage (`insert_or_create_calendar_event`) | API/Backend (call site + field mapping) | Reuses existing helper unchanged, per CONTEXT.md canonical refs |
| Staff email notification | API/Backend (Django email backend) | — | Synchronous `send_mail()` call inside the submission view; no task queue needed for this volume |
| Approval-status visibility filter (D-09) | API/Backend (queryset-level `.exclude()`) | Database/Storage | Must restrict at the SQL level, not the template, per Phase 15 D-13 precedent |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django | 5.2.15 `[VERIFIED: pip show, local env]` | Forms, views, ORM atomic `.update()`, email backends | Already the project's framework |
| django-crispy-forms + crispy-bootstrap4 | 2.4 / 2024.10 `[VERIFIED: pip show, local env]` | Submission-form layout, matches `EphemerisForm` precedent | Established pattern in `solsys_code/forms.py` |
| django-tables2 | 3.0.0 `[VERIFIED: pip show, local env]` | Approval-queue's two tables (pending + recently-decided), reuses `CampaignRunTable` | Already used by Phase 15's `CampaignRunTableView` |
| django-filter | 24.3 `[VERIFIED: pip show, local env]` | Not required for this phase's new views (see Architecture Pattern below) — kept for completeness since `CampaignRunFilterSet` still backs the existing table | Already installed |

**No new dependencies required for this phase.** `[VERIFIED: local pip show + SUMMARY.md "No new dependencies required"]`

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `django.core.mail` (stdlib-adjacent, ships with Django) | 5.2.15 | Staff notification email (SUBMIT-05) | `send_mail()` is sufficient; no `EmailMultiAlternatives`/HTML needed for a bare-ping body (D-04) |
| `django.contrib.messages` | bundled, already in `INSTALLED_APPS`/`MIDDLEWARE` `[VERIFIED: src/fomo/settings.py]` | Approve/Reject success/failure feedback on the approval-queue page | Standard Django pattern; no new setting needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Plain `django.core.mail.send_mail()` | `django-anymail` / Celery-backed async email | Overkill for a low-volume, single-recipient-list ping; adds a dependency and (for Celery) a worker process this project doesn't run |
| Hand-rolled honeypot field | `django-honeypot` package | CLAUDE.md's minimal-dependency convention + CONTEXT.md's Claude's Discretion already rules this out; a `forms.CharField(required=False)` + `clean_<field>()` is ~10 lines |
| `django_tables2.MultiTableMixin` for the two approval-queue tables | Two separate `CampaignRunTable` instances built manually in `get_context_data` | `MultiTableMixin` is built for N *symmetric* tables of the *same* queryset shape; here only the pending queue needs zero filtering and the recently-decided queryset needs a different `.exclude()`/ordering — simpler to build both tables explicitly (see Architecture Pattern) `[CITED: django-tables2 docs — MultiTableMixin]` |
| Atomic `.filter().update()` for approve/reject | `transaction.atomic()` + `select_for_update()` | `select_for_update()` provides no additional protection on this project's dev DB (SQLite has no real row-level locking) and the conditional `.update()` is already a single atomic SQL statement — simpler and sufficient `[CITED: Django docs / community race-condition writeups]` |

**Installation:** None — no new packages.

## Package Legitimacy Audit

**Not applicable — this phase introduces no new external packages.** `[VERIFIED: pip show of all
libraries used, local env, 2026-07-03; SUMMARY.md's "No new dependencies required" confirmed]`

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────────────────┐
 Anonymous visitor ─▶│  GET  /campaigns/<pk>/submit/            │
                     │  CampaignRunSubmissionView (FormView)    │
                     └───────────────┬───────────────────────────┘
                                     │ POST (form + hidden honeypot field)
                                     ▼
                     ┌─────────────────────────────────────────┐
                     │ form.is_valid()?                         │
                     │   honeypot field non-empty?               │
                     │     YES → return same "thanks" response,  │
                     │           create NOTHING, send NOTHING    │  SUBMIT-04
                     │     NO  → CampaignRun.objects.create(     │
                     │             approval_status=PENDING_REVIEW│  SUBMIT-01/02
                     │             site=None, site_raw=<text>)   │
                     │           send_mail() to is_staff+email   │  SUBMIT-05
                     └───────────────┬───────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────────────────┐
 Staff (is_staff)  ─▶│  GET /campaigns/approval-queue/           │
                     │  ApprovalQueueView (staff-only)           │
                     │  ┌───────────────┐  ┌───────────────────┐│
                     │  │ pending table │  │ recently-decided   ││  D-01/D-02
                     │  │ (approval_    │  │ table (-approval_  ││
                     │  │  status=      │  │  status=pending,   ││
                     │  │  pending)     │  │  order -pk)        ││
                     │  └───────┬───────┘  └───────────────────┘│
                     └──────────┼─────────────────────────────────┘
                                │ POST /campaigns/<pk>/approve/ (or /reject/)
                                ▼
                     ┌─────────────────────────────────────────┐
                     │ n = CampaignRun.objects.filter(           │
                     │       pk=pk, approval_status='pending_    │  SUBMIT-03
                     │       review').update(approval_status=…)  │  (atomic, race-safe)
                     │ if n == 1 and action == 'approve':         │
                     │     run = resolve_site(run.site_raw)  →   │  D-07
                     │       run.site, run.site_needs_review      │
                     │     if telescope_instrument and ut_start   │
                     │        and ut_end present:                 │  CAL-01
                     │       insert_or_create_calendar_event(     │
                     │         {'url': f'CAMPAIGN:{run.pk}'},     │
                     │         fields={..., target_list=campaign})│  CAL-02/CAL-03
                     └─────────────────────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────────────────┐
 Anonymous visitor ─▶│ GET /campaigns/<pk>/  CampaignRunTableView│
                     │ non-staff queryset EXCLUDES               │  SUBMIT-02/D-09
                     │ approval_status='pending_review'           │
                     └─────────────────────────────────────────┘
```

### Recommended Project Structure
```
solsys_code/
├── campaign_forms.py       # NEW: CampaignRunSubmissionForm (plain forms.Form + honeypot)
├── campaign_views.py       # EXTEND: + CampaignRunSubmissionView, ApprovalQueueView,
│                           #         CampaignRunDecisionView (approve/reject); D-09 filter
│                           #         added to CampaignRunTableView.get_queryset()
├── campaign_tables.py      # EXTEND (optional): ApprovalQueueTable subclass adding an
│                           #         Approve/Reject action column, or reuse CampaignRunTable
│                           #         as-is and put the action buttons in the template instead
├── campaign_urls.py        # EXTEND: submit/, approval-queue/, <pk>/approve/, <pk>/reject/
├── mixins.py                # NEW (or add to an existing shared module): StaffRequiredMixin
└── models.py                # UNCHANGED — no migration needed this phase

src/templates/campaigns/
├── campaignrun_submit_form.html   # NEW: crispy form render
├── submission_thanks.html          # NEW: generic "thanks, pending review" page (shown for
│                                     #      both genuine and honeypot-tripped submissions)
└── approval_queue.html             # NEW: two tables + per-row Approve/Reject POST forms
```

### Pattern 1: Plain `forms.Form`, not `ModelForm`, for the submission form
**What:** `CampaignRunSubmissionForm(forms.Form)` with explicit fields, not
`forms.ModelForm(Meta.model = CampaignRun)`.
**When to use:** Always for this form.
**Why:** `[VERIFIED: solsys_code/models.py]` `CampaignRun.telescope_instrument` is declared
`models.CharField(max_length=255, verbose_name=...)` with **no** `blank=True`. A `ModelForm`
derives field-required-ness from the model field, so it would silently force
`telescope_instrument` to be required — directly contradicting D-05 ("everything except
`campaign` is optional"). A plain form with explicit `required=False` on every non-`campaign`
field sidesteps this entirely, and the view constructs the `CampaignRun` via
`.objects.create(**cleaned_fields)`, which does not call `full_clean()` and therefore never
re-triggers model-level blank/required checks.
**Example:**
```python
# solsys_code/campaign_forms.py — pattern only, not final field list/help text
from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Div, Fieldset, Layout, Submit
from crispy_forms.bootstrap import FormActions
from django import forms
from tom_targets.models import TargetList


class CampaignRunSubmissionForm(forms.Form):
    campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)
    telescope_instrument = forms.CharField(max_length=255, required=False)
    site_raw = forms.CharField(max_length=255, required=False, label='Observing site')
    obs_date = forms.DateField(required=False)
    ut_start = forms.DateTimeField(required=False)
    ut_end = forms.DateTimeField(required=False)
    filters_bandpass = forms.CharField(max_length=255, required=False)
    observation_details = forms.CharField(widget=forms.Textarea, required=False)
    open_to_collaboration = forms.BooleanField(required=False)
    contact_person = forms.CharField(max_length=255, required=True)   # D-06: required at form level
    contact_email = forms.EmailField(required=True)                    # D-06
    comments = forms.CharField(widget=forms.Textarea, required=False)
    # SUBMIT-04: hidden honeypot, non-obvious name, never rendered visibly to a human.
    alt_contact_info = forms.CharField(required=False, widget=forms.HiddenInput())

    def clean_alt_contact_info(self):
        # Deliberately does NOT raise ValidationError -- SUBMIT-04 says the bot gets no error
        # at all. The view checks cleaned_data['alt_contact_info'] and silently short-circuits
        # (see Pattern 3) rather than failing form validation.
        return self.cleaned_data.get('alt_contact_info', '')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'campaign',
            Fieldset('Run details', 'telescope_instrument', 'site_raw', 'obs_date', 'ut_start', 'ut_end',
                     'filters_bandpass', 'observation_details', 'open_to_collaboration'),
            Fieldset('Contact', 'contact_person', 'contact_email', 'comments'),
            # Hidden via widget=HiddenInput above; also keep it out of visible flow/CSS as
            # belt-and-suspenders (e.g. wrap in a Div with a class the site's CSS hides).
            Div('alt_contact_info', css_class='d-none'),
            FormActions(Submit('submit', 'Submit run for review')),
        )
```

### Pattern 2: Staff-only view gating via a local mixin (mirrors `tom_common.mixins.SuperuserRequiredMixin`)
**What:** No existing view in `solsys_code/` currently hard-gates by `is_staff` at the
dispatch/403 level — Phase 15's views instead filter *data* per-request while staying 200-OK for
everyone (`AUTH_STRATEGY='READ_ONLY'` makes the whole site readable by default). D-01's
approval-queue page and the approve/reject actions are the first views in this app that need a
real "staff or redirect/403" gate.
**When to use:** `ApprovalQueueView` and the approve/reject action views.
**Why this pattern:** `[VERIFIED: direct read of installed tom_common/mixins.py]` TOM Toolkit
itself already ships exactly this shape for `is_superuser`:
```python
# installed: tom_common/mixins.py
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import user_passes_test

class SuperuserRequiredMixin():
    @method_decorator(user_passes_test(lambda u: u.is_superuser))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
used by `tom_common.views.GroupCreateView`, `UserCreateView`, etc. Add a local
`StaffRequiredMixin` in `solsys_code/` using the identical shape but checking `u.is_staff`
instead of `u.is_superuser` — `user_passes_test` with no explicit `login_url` redirects
unauthenticated/failing users to `settings.LOGIN_URL` (`'/accounts/login/'`, already configured),
consistent with how the rest of this TOM handles login redirects.
```python
# solsys_code/mixins.py (new, small)
from django.contrib.auth.decorators import user_passes_test
from django.utils.decorators import method_decorator


class StaffRequiredMixin:
    """Redirect to LOGIN_URL unless request.user.is_staff (D-01 approval-queue gate)."""

    @method_decorator(user_passes_test(lambda u: u.is_staff))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```

### Pattern 3: Honeypot — silent success, never an error
**What:** On a tripped honeypot, the view must respond exactly as it would to a genuine
successful submission (same redirect/thanks page, same 200/302 status), just without creating a
`CampaignRun` or sending an email.
**When to use:** `CampaignRunSubmissionView.form_valid()`.
**Why:** `[CITED: web-search-verified honeypot best practice]` A honeypot's value depends on the
bot never learning it was caught — returning a distinguishable error (400, a form-validation
error, a different redirect target) lets a bot operator detect and adapt. `clean_alt_contact_info`
deliberately does not raise (Pattern 1), so `form.is_valid()` still returns `True` on a tripped
honeypot; the short-circuit happens in the view:
```python
def form_valid(self, form):
    if form.cleaned_data.get('alt_contact_info'):
        # SUBMIT-04: bot tripped the honeypot. No CampaignRun, no email, no error --
        # fall straight through to the same success redirect as a genuine submission.
        return redirect('campaigns:submission_thanks')
    run = CampaignRun.objects.create(
        campaign=form.cleaned_data['campaign'],
        telescope_instrument=form.cleaned_data['telescope_instrument'],
        site_raw=form.cleaned_data['site_raw'],
        # ... remaining fields ...
        # approval_status defaults to PENDING_REVIEW (model default) -- do not set explicitly
        # unless you want to be explicit for readability; either is correct.
    )
    self._notify_staff(run)
    return redirect('campaigns:submission_thanks')
```

### Pattern 4: Atomic approve/reject + conditional calendar projection (SUBMIT-03 / CAL-01 / CAL-03)
**What:** A single conditional `UPDATE` proves the double-approve no-op; the calendar-projection
call is gated on that same conditional actually having matched a row.
**When to use:** The approve action view.
**Why this satisfies CAL-03 "for free":** `[VERIFIED: solsys_code/campaign_utils.py,
solsys_code/calendar_utils.py]` If a second "approve" click hits an already-`approved` row, the
conditional `.filter(pk=pk, approval_status='pending_review').update(...)` matches zero rows
(`updated_count == 0`), so `insert_or_create_calendar_event()` is never called a second time —
no duplicate event, no `modified` churn, and no extra `resolve_site()`/DB round trip. This is the
*primary* defense. `insert_or_create_calendar_event()`'s own no-churn field comparison is a
second, independent line of defense if the projection code is ever invoked again with identical
fields (e.g. from a future edit-and-reapprove flow).
```python
def post(self, request, pk):
    action = request.POST.get('action')  # 'approve' or 'reject'
    if action not in ('approve', 'reject'):
        return HttpResponseBadRequest()
    new_status = CampaignRun.ApprovalStatus.APPROVED if action == 'approve' else CampaignRun.ApprovalStatus.REJECTED
    updated_count = CampaignRun.objects.filter(
        pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
    ).update(approval_status=new_status)

    if updated_count == 1 and action == 'approve':
        run = CampaignRun.objects.get(pk=pk)
        site, needs_review = resolve_site(run.site_raw)  # D-07: reuse existing 3-tier resolver
        run.site, run.site_needs_review = site, needs_review
        run.save(update_fields=['site', 'site_needs_review'])

        # CAL-01: "telescope + date range" == telescope_instrument non-blank AND both
        # ut_start and ut_end present (CalendarEvent.start_time/end_time are non-nullable --
        # a run with only ut_start and no ut_end cannot become a valid CalendarEvent).
        if run.telescope_instrument and run.ut_start and run.ut_end:
            insert_or_create_calendar_event(
                {'url': f'CAMPAIGN:{run.pk}'},
                fields={
                    'title': f'{run.campaign.name}: {run.telescope_instrument}',
                    'description': run.observation_details,
                    'start_time': run.ut_start,
                    'end_time': run.ut_end,
                    'target_list': run.campaign,          # CAL-02
                    'telescope': run.telescope_instrument,
                },
            )
        messages.success(request, 'Run approved.')
    elif updated_count == 1:
        messages.success(request, 'Run rejected.')
    else:
        messages.warning(request, 'This run was already decided by someone else.')
    return redirect('campaigns:approval_queue')
```
Use `require_POST` (or restrict `http_method_names = ['post']` on a `View` subclass) so the
state-changing action can never be triggered by a GET (crawler prefetch, `<a href>` link, etc.).

### Pattern 5: Two independent tables on one approval-queue page (D-01/D-02)
**What:** Build two separate `CampaignRunTable` instances from two separate querysets rather than
routing both through `FilterView`/`SingleTableMixin` (which are built around exactly one
queryset).
**When to use:** `ApprovalQueueView`.
**Why:** `[CITED: django-tables2 docs — Class Based Generic Mixins / MultiTableMixin]`
`django_tables2.views.MultiTableMixin` exists for N tables in one view and auto-prefixes each
table's sort/page query params (default `'table_{}-'`) so they don't collide — but it's designed
for a list of *symmetric* tables. Here the two querysets have genuinely different filtering
(`approval_status='pending_review'` vs. `approval_status != 'pending_review'` ordered `-pk`, see
Common Pitfalls "No `modified` field"), so it's simpler and more explicit to skip
`MultiTableMixin`/`FilterView` machinery entirely and build both tables by hand in
`get_context_data`, giving each an explicit `prefix=` so their pagination/sort query params don't
collide if both render on the same page:
```python
class ApprovalQueueView(StaffRequiredMixin, TemplateView):
    template_name = 'campaigns/approval_queue.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        pending_qs = CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).select_related('campaign', 'site')
        decided_qs = CampaignRun.objects.exclude(
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).select_related('campaign', 'site').order_by('-pk')[:20]  # no timestamp field -- see Pitfalls
        context['pending_table'] = ApprovalQueueTable(pending_qs, prefix='pending-')
        context['decided_table'] = ApprovalQueueTable(decided_qs, prefix='decided-', show_actions=False)
        return context
```

### Pattern 6: D-09 non-staff visibility filter on the existing Phase 15 table
**What:** Add a queryset-level `.exclude()`, not a template conditional, mirroring D-13's
existing "restrict the queryset, not just the rendered table" discipline.
**When to use:** `CampaignRunTableView.get_queryset()`.
```python
def get_queryset(self):
    campaign_pk = self.kwargs['pk']
    qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
    if self.request.user.is_staff:
        return qs.select_related('site')
    # D-09: non-staff see approved AND rejected, only pending_review is hidden.
    qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
    return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)
```

### Anti-Patterns to Avoid
- **Using `ModelForm` for the submission form:** forces `telescope_instrument` required
  (contradicts D-05). See Pattern 1.
- **Raising `ValidationError` on a tripped honeypot:** tips off sophisticated bots that they were
  detected. See Pattern 3.
- **Calling `insert_or_create_calendar_event()` unconditionally on every approve POST** (not
  gated on `updated_count == 1`): technically still idempotent (the helper's own no-churn
  contract prevents a duplicate event), but performs a redundant `resolve_site()` MPC-API-capable
  call and DB round trip on every re-click, and produces misleading `'unchanged'` log noise on
  what should be a no-op. See Pattern 4.
- **Ordering "recently decided" by a `modified`/`updated_at` field:** `CampaignRun` has no such
  field (see Common Pitfalls). Order by `-pk` instead.
- **GET-triggered approve/reject:** a `<a href="/campaigns/5/approve/">` link or unauthenticated
  crawler prefetch could silently approve a run. Require POST.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Create-or-update `CalendarEvent` with no duplicate/no churn | A new campaign-specific calendar-write function | `calendar_utils.insert_or_create_calendar_event()` | CONTEXT.md canonical refs mandate this — CAL-01 explicitly requires routing through it; direct `CalendarEvent.objects.create()` risks Pitfall 3 (namespace collision with LCO/Gemini/classical syncs) |
| Resolve a free-text site string to an `Observatory` | New FK-resolution logic in the approval view | `campaign_utils.resolve_site()` | D-07 explicitly reuses the existing 3-tier resolver; it already handles blank/oversized/malformed codes without raising |
| Detect/deflect bot form spam | A third-party honeypot package (`django-honeypot`, etc.) | A plain hidden `forms.CharField(required=False)` | CLAUDE.md's minimal-dependency convention; CONTEXT.md's Claude's Discretion already rules out a package for this scope |
| Race-safe status transition | `transaction.atomic()` + `select_for_update()` | `.filter(pk=pk, approval_status=<expected>).update(...)` | Simpler, and `select_for_update()` buys nothing extra on this project's SQLite dev DB (no real row-level locking) `[CITED]` |

**Key insight:** every piece of "hard" logic this phase needs (idempotent write, site resolution,
staff-only gating) already has a proven, tested implementation somewhere in this codebase or in
TOM Toolkit itself — this phase is primarily new orchestration (form → view → action → template),
not new algorithms.

## Common Pitfalls

### Pitfall 1: `CampaignRun` has no `modified`/timestamp field at all
**What goes wrong:** Trying to order the "recently decided" section (D-02) by "most recently
approved/rejected" fails — there's no field to order by.
**Why it happens:** `[VERIFIED: solsys_code/models.py, STATE.md Phase 14 decisions]` This was a
deliberate prior choice: "`insert_or_create_campaign_run` omits `'modified'` from
`update_fields` since `CampaignRun` has no auto-now timestamp field, unlike
`insert_or_create_calendar_event`." No decision in Phase 16's CONTEXT.md revisits this.
**How to avoid:** Order the recently-decided queryset by `-pk` (a reasonable proxy for
recency of DB write, though not literally "time of decision") and cap it (`[:20]` or similar).
Do **not** add a new `decided_at` field/migration to solve this unless the planner explicitly
decides precise decision-time ordering matters enough to justify a schema change — that's outside
what CONTEXT.md's decisions authorize.
**Warning signs:** A plan task that references `CampaignRun.modified` or tries to `order_by
('-modified')` — this field does not exist on `CampaignRun` (only on `CalendarEvent`).

### Pitfall 2: CAL-01's "date range" condition needs both `ut_start` AND `ut_end`
**What goes wrong:** Approving a run with only `ut_start` set (no `ut_end`) either crashes
(`CalendarEvent.end_time` is non-nullable) or silently defaults `end_time` to something
fabricated (e.g. `ut_start` itself), producing a zero-duration calendar event that misrepresents
the run.
**Why it happens:** `[VERIFIED: tom_calendar/models.py]` `CalendarEvent.start_time` and
`end_time` are both plain `DateTimeField()` with no `null=True`. `CampaignRun.ut_end` is
nullable/optional on the submission form (D-05).
**How to avoid:** Gate calendar projection on `telescope_instrument and ut_start and ut_end` all
being truthy (see Pattern 4). A run missing `ut_end` simply doesn't get a `CalendarEvent` yet —
consistent with the phase's stated trigger condition ("telescope + date range").
**Warning signs:** A plan task that creates a `CalendarEvent` from `ut_start` alone, or that sets
`end_time=ut_start` as a fallback without this being an explicit, separately-confirmed decision.

### Pitfall 3: Submission form must not be a `ModelForm`
**What goes wrong:** `telescope_instrument` (and any other non-`blank=True` `CampaignRun` field)
becomes silently required, breaking D-05's "everything except `campaign` is optional."
**Why it happens:** `[VERIFIED: solsys_code/models.py]` `ModelForm` derives `required` from the
model field's `blank` attribute; `telescope_instrument` has no `blank=True`.
**How to avoid:** Use a plain `forms.Form` (Pattern 1) with explicit `required=False` per field,
and build the `CampaignRun` via `.objects.create(**fields)` (never calls `full_clean()`).
**Warning signs:** `class CampaignRunSubmissionForm(forms.ModelForm)` anywhere in the diff.

### Pitfall 4: `CampaignRun`'s natural-key `UniqueConstraint` can reject a legitimate second submission
**What goes wrong:** The model has `UniqueConstraint(fields=['campaign', 'telescope_instrument',
'ut_start'])` (from Phase 14, for CSV-import idempotency). Two *different* submitters proposing
the same telescope at the same start time for the same campaign (a real, if rare, possibility —
e.g. two people both propose "FTN, 2026-08-01 03:00 UTC" independently) will raise
`IntegrityError` on the second `.objects.create()`.
**Why it happens:** `[VERIFIED: solsys_code/models.py Meta.constraints]` The constraint was
designed for CSV-import idempotency (Phase 14), not anticipated as a public-form collision case.
Note: if either submission leaves `telescope_instrument` blank or `ut_start` unset (`None`), the
constraint does not fire (SQL treats `NULL` as never-equal-to-`NULL`; two rows sharing a blank
string but `ut_start=NULL` don't collide either) — so this only bites when both fields are
genuinely populated and identical.
**How to avoid:** Wrap the `.objects.create()` call in the submission view in a `try/except
IntegrityError` and surface a friendly form error ("A run already exists for this
telescope/time — please check the campaign table or contact staff") rather than a 500. This is a
concrete, testable edge case the plan should include, not just a theoretical note.
**Warning signs:** A plan/test that never exercises submitting two runs with identical
`campaign`+`telescope_instrument`+`ut_start`.

### Pitfall 5: Honeypot field name and hiding must survive both crispy-forms rendering and template CSS
**What goes wrong:** A honeypot rendered as a normal visible crispy-forms field with a label like
"Honeypot" is worse than useless — real users will see and fill it in (breaking their own
submission) while sophisticated bots recognize the literal string "honeypot" and skip it.
**Why it happens:** Default crispy-forms field rendering shows a label + input, fully visible.
**How to avoid:** `widget=forms.HiddenInput()` on the Django field (renders `<input
type="hidden">` with no visible label at all) is sufficient and simplest — it's not just CSS-hidden,
it never appears in the rendered HTML as a fillable, visible control. Avoid literal names like
`honeypot`/`spam_trap` (CONTEXT.md already directs this); a plausible-sounding name (e.g.
`alt_contact_info`) is safest if any bot inspects field names, not just visibility. `[CITED:
web-search-verified honeypot naming guidance]`
**Warning signs:** The honeypot field rendered with a visible crispy `Field(...)` layout entry
and no `HiddenInput` widget.

### Pitfall 6: `send_mail()` failure must not break the submission
**What goes wrong:** If `send_mail()` raises (SMTP misconfiguration, network issue) inside
`form_valid()`, the whole request 500s — the genuine `CampaignRun` the submitter just created
is still in the DB (transaction already committed unless explicitly wrapped), but the submitter
sees an error page instead of the "thanks" confirmation, and may resubmit, risking Pitfall 4's
constraint collision.
**Why it happens:** `send_mail(..., fail_silently=False)` is the default; no `EMAIL_BACKEND` is
currently configured in `src/fomo/settings.py` at all `[VERIFIED: src/fomo/settings.py, grep
confirms no EMAIL_BACKEND/EMAIL_HOST/ADMINS entries]`.
**How to avoid:** Either call `send_mail(..., fail_silently=True)` (simplest — the notification
is a nice-to-have, not the primary deliverable of the request) or wrap the call in `try/except`
and log at `debug`/`warning` level per this codebase's logging convention
(`logger.debug(f'...')`), then still redirect to the thanks page. Add `EMAIL_BACKEND =
'django.core.mail.backends.console.EmailBackend'` to `src/fomo/settings.py` for local dev (visible
in `runserver` stdout) — Django's test runner automatically substitutes the `locmem` backend
during `manage.py test` regardless of this setting, so tests don't need `override_settings` for
basic `mail.outbox` assertions `[CITED: Django docs — testing tools/email]`. Document that
production deployments must set a real `EMAIL_BACKEND`/`EMAIL_HOST`/`EMAIL_HOST_USER`/
`EMAIL_HOST_PASSWORD` (and ideally `DEFAULT_FROM_EMAIL`) in `local_settings.py`.
**Warning signs:** A test that mocks `send_mail` to raise and asserts the view still 500s (should
instead assert graceful degradation), or a plan that never adds `EMAIL_BACKEND` to settings at
all and relies on Django's implicit `smtp.EmailBackend` default (which will hang/fail against
`localhost:25` with no local MTA).

### Pitfall 7: Approval-queue and action views need explicit `is_staff` gating — no existing precedent to copy blindly
**What goes wrong:** Copying Phase 15's `CampaignRunTableView` pattern (which stays 200-OK for
everyone and just restricts *data*) to the approval-queue page would leave it publicly viewable
(pending submissions, including contact PII, exposed to anonymous visitors) — a materially worse
outcome than Phase 15's PII gating gap, since this page's entire *purpose* is staff-only triage.
**Why it happens:** `AUTH_STRATEGY = 'READ_ONLY'` `[VERIFIED: src/fomo/settings.py]` makes every
view readable by default unless explicitly gated; there is no existing `is_staff`-hard-gate
precedent in `solsys_code/` to copy from (grep confirms only the soft data-filtering pattern
exists).
**How to avoid:** Use `StaffRequiredMixin` (Pattern 2) on `ApprovalQueueView` and both
approve/reject action views. Test with an anonymous client and a non-staff-authenticated client,
both expecting a redirect to `LOGIN_URL` (or 403, depending on final mixin config), never a 200
with pending-queue content.
**Warning signs:** `ApprovalQueueView` implemented as a plain `TemplateView`/`ListView` with no
mixin, relying only on the template hiding the "Approve" button for non-staff.

## Code Examples

See Architecture Patterns section above — all code examples for this phase are embedded there
(submission form, staff mixin, honeypot short-circuit, atomic approve/calendar projection, two
independent tables, D-09 queryset filter) since each is tightly coupled to its rationale.

### Staff notification email (SUBMIT-05 / D-03 / D-04)
```python
# in CampaignRunSubmissionView, called from form_valid() after a genuine (non-honeypot) create
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.urls import reverse

def _notify_staff(self, run):
    recipients = list(
        User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True)
    )
    if not recipients:
        return  # no staff with an email on file -- nothing to notify, not an error
    queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
    send_mail(
        subject='FOMO: new campaign run submission pending review',
        message=f'A new run submission is pending review: {queue_url}',  # D-04: no PII, no details
        from_email=None,  # falls back to settings.DEFAULT_FROM_EMAIL
        recipient_list=recipients,
        fail_silently=True,  # Pitfall 6: notification failure must not break the submission
    )
```

## Runtime State Inventory

Not applicable — this is a greenfield-within-phase feature addition (new views/forms/URLs on top
of an unchanged schema), not a rename/refactor/migration phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CAL-01's "date range" trigger condition requires both `ut_start` AND `ut_end` present (not just one) | Architecture Pattern 4, Common Pitfall 2 | If the intended condition was "any of obs_date/ut_start present," some approved runs that should get a `CalendarEvent` won't — low risk (fails safe, no event created rather than a malformed one), but should be confirmed with the user/planner before implementation if there's any ambiguity |
| A2 | "Recently decided" (D-02) should be ordered by `-pk` and capped (e.g. 20 rows), since `CampaignRun` has no timestamp field | Common Pitfall 1, Pattern 5 | If staff expect true chronological-by-decision ordering, `-pk` is a reasonable proxy (matches insertion order) but not exact if `CampaignRun` rows are ever created out of pk order (e.g. after a future bulk-import). Low risk for this phase's launch scope |
| A3 | Honeypot field is safest as `widget=forms.HiddenInput()` (not merely CSS `display:none` on a visible-type field) | Common Pitfall 5 | If a UI reviewer wants a CSS-based hide instead (e.g. to test more sophisticated bot behavior later), the `HiddenInput` approach is easily swapped for a visible field + CSS class in the UI phase without touching the `clean()` logic |
| A4 | `send_mail(..., fail_silently=True)` is preferable to `try/except` + explicit logging | Common Pitfall 6, Code Examples | If staff need visibility into *why* a notification silently failed, `fail_silently=True` hides that; a `try/except` + `logger.warning()` is a safe alternative the planner can choose instead — noted as a genuine open choice, not a hard requirement |

## Open Questions

1. **Exact "date range" trigger condition for CAL-01 (A1 above)**
   - What we know: CONTEXT.md's success criterion says "a telescope and date range" (Success
     Criterion 5); `CalendarEvent.start_time`/`end_time` are both non-nullable.
   - What's unclear: whether `obs_date` alone (without `ut_start`/`ut_end`) should also count as
     a valid trigger, defaulting the calendar event to midnight UTC (mirroring
     `campaign_utils.parse_obs_window`'s CSV-import fallback behavior).
   - Recommendation: require `ut_start` AND `ut_end` both present (strictest, safest — never
     fabricates a time); if the planner/UAT later decides `obs_date`-only submissions should
     still get a (perhaps flagged) calendar event, that's a small, additive follow-up.

2. **Should the approval-queue "recently decided" section include a manual undo/re-open action?**
   - What we know: D-02 says the section exists "so staff can spot-check or catch a mis-click."
   - What's unclear: whether "catching a mis-click" means just *seeing* the mistake (informational
     only, no action) or also *reverting* it (an "undo" button that flips `approved`/`rejected`
     back to `pending_review`).
   - Recommendation: CONTEXT.md's Success Criteria only requires Approve/Reject actions on the
     pending queue — treat the recently-decided section as read-only/informational for this
     phase unless the planner scopes an explicit undo action; leaving both a `pending_review`
     conditional gate on approve/reject already prevents most double-action harm.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| SMTP server / real email delivery | SUBMIT-05 (production) | ✗ (not configured, dev env) | — | Console backend for dev (`EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'`); `locmem` auto-used by Django's test runner; real SMTP config documented as a `local_settings.py` production requirement |

**Missing dependencies with no fallback:** None — email has a full dev/test fallback path.

**Missing dependencies with fallback:** SMTP server (see above).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django's built-in test runner (`django.test.TestCase`), **not** pytest — `pyproject.toml` `testpaths` excludes `solsys_code/` `[VERIFIED: pyproject.toml, CLAUDE.md]` |
| Config file | none — Django test discovery via `./manage.py test` |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_submission` (once created) |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SUBMIT-01 | Valid submission (campaign only) creates a `PENDING_REVIEW` `CampaignRun` | integration | `./manage.py test solsys_code.tests.test_campaign_submission.TestCampaignSubmission.test_minimal_valid_submission_creates_pending_run` | ❌ Wave 0 |
| SUBMIT-01 | Missing `campaign` fails validation; missing `contact_person`/`contact_email` fails validation (D-06) | integration | `./manage.py test solsys_code.tests.test_campaign_submission` | ❌ Wave 0 |
| SUBMIT-02 | Anonymous client cannot see a `pending_review` row on the per-campaign table | integration | `./manage.py test solsys_code.tests.test_campaign_views.TestContactFieldGating` (extend) or new module | ❌ Wave 0 (extend existing) |
| SUBMIT-03 | Approve twice: first call transitions, second call is a proven no-op (`updated_count == 0`, no duplicate `CalendarEvent`, no second email) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestApproval.test_double_approve_is_noop` | ❌ Wave 0 |
| SUBMIT-04 | Honeypot-filled submission: no `CampaignRun` created, no email sent, response is the same success page as a genuine submission | integration | `./manage.py test solsys_code.tests.test_campaign_submission.TestHoneypot` | ❌ Wave 0 |
| SUBMIT-05 | Genuine submission triggers `send_mail` to every `is_staff=True` user with a non-empty email; staff with blank email excluded | integration (`mail.outbox`) | `./manage.py test solsys_code.tests.test_campaign_submission.TestStaffNotification` | ❌ Wave 0 |
| SUBMIT-05 | Email body/subject contain no PII (no `contact_person`/`contact_email`/telescope/campaign name) — proves D-04 | integration | same test module | ❌ Wave 0 |
| CAL-01 | Approving a run with telescope + `ut_start` + `ut_end` creates a `CalendarEvent` keyed `CAMPAIGN:{pk}` | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestCalendarProjection` | ❌ Wave 0 |
| CAL-02 | Created `CalendarEvent.target_list` equals the campaign's `TargetList` | integration | same test module | ❌ Wave 0 |
| CAL-03 | Re-approving an already-approved run creates no duplicate `CalendarEvent` and causes no `modified` churn (assert `CalendarEvent.objects.count()` unchanged and `modified` timestamp unchanged after the second approve attempt) | integration | same test module | ❌ Wave 0 |
| D-09 | Non-staff sees `approved` and `rejected` rows, not `pending_review`, on the per-campaign table | integration | extend `test_campaign_views.py` | ❌ Wave 0 (extend existing) |
| D-01/D-02 | Anonymous/non-staff GET to approval-queue and approve/reject URLs redirects (never 200 with content) | integration | `./manage.py test solsys_code.tests.test_campaign_approval.TestStaffGating` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted test module for the file(s) touched, e.g. `./manage.py test
  solsys_code.tests.test_campaign_submission`
- **Per wave merge:** `./manage.py test solsys_code` (full Django app suite; ephemeris-related
  tests will still trigger the ~1.6 GB SPICE download on a cold cache per CLAUDE.md — expect a
  slow first run, not a phase-16-specific regression)
- **Phase gate:** `./manage.py test solsys_code` green, plus `ruff check .` and `ruff format
  --check .` clean, before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_campaign_submission.py` — covers SUBMIT-01, SUBMIT-04, SUBMIT-05
- [ ] `solsys_code/tests/test_campaign_approval.py` — covers SUBMIT-03, CAL-01, CAL-02, CAL-03,
      and staff-gating for the approval-queue/action views
- [ ] Extend `solsys_code/tests/test_campaign_views.py` — D-09 non-staff visibility filter
- [ ] No new framework install needed — `django.test.TestCase` + `django.core.mail` (`outbox`)
      are already available; no fixtures beyond `TargetList.objects.create(...)` +
      `User.objects.create_user(..., is_staff=True)` (already established in
      `test_campaign_views.py`)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Reuses Django's existing session-based auth (`django.contrib.auth`); no new auth surface introduced |
| V3 Session Management | no | No new session handling code; standard Django session middleware already in place |
| V4 Access Control | yes | `StaffRequiredMixin` (Pattern 2) on `ApprovalQueueView` + approve/reject action views; POST-only enforcement (`require_POST`) so state changes can't be GET-triggered |
| V5 Input Validation | yes | Django `forms.Form` field validation (`EmailField`, `required=True` per D-06); CSRF via `CsrfViewMiddleware` (already in `MIDDLEWARE`) + crispy-forms' automatic `{% csrf_token %}` inclusion |
| V6 Cryptography | no | Nothing in this phase touches secrets, tokens, or crypto primitives |

### Known Threat Patterns for Django form + moderation-queue apps

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Automated bot mass-submission (spam) | Denial of Service / Tampering | Honeypot field (Pattern 3), silently dropped, no error signal to the bot |
| Double-approve / TOCTOU race on the approval action | Tampering | Single atomic conditional `.filter().update()` (Pattern 4) — no read-then-write gap |
| Unauthenticated/non-staff access to the approval queue (pending submissions + contact PII) | Information Disclosure / Elevation of Privilege | `StaffRequiredMixin` hard gate (Pitfall 7) — this is the one view in this app that must NOT follow Phase 15's "soft data filtering" pattern |
| PII leakage via the staff notification email | Information Disclosure | D-04: bare ping + link only, no contact/telescope/campaign details in subject or body — email infrastructure (inboxes, mail server logs) sits outside FOMO's own DB-level PII gate |
| CSRF on the approve/reject POST endpoints | Tampering | Django's `CsrfViewMiddleware` (already active) + form-embedded `{% csrf_token %}` in the per-row Approve/Reject `<form>` |
| GET-triggered state change (crawler prefetch approving/rejecting a run via a bare link) | Tampering | Approve/reject views restricted to POST only (`require_POST` / `http_method_names = ['post']`) |

## Sources

### Primary (HIGH confidence)
- `solsys_code/models.py` — `CampaignRun` field definitions, `UniqueConstraint`, absence of a
  `modified` field
- `solsys_code/campaign_utils.py` — `resolve_site()`, `insert_or_create_campaign_run()`
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()`
- `solsys_code/campaign_views.py`, `campaign_tables.py`, `campaign_filters.py`, `campaign_urls.py`
  — Phase 15's established view/table/filter/URL structure
- `solsys_code/forms.py` — `EphemerisForm` crispy-forms pattern
- `solsys_code/apps.py` — `nav_items()`, `target_detail_buttons()` integration points
- `src/fomo/settings.py` — confirmed no `EMAIL_BACKEND`/`EMAIL_HOST`/`ADMINS`; confirmed
  `django.contrib.messages` already installed/configured; confirmed `AUTH_STRATEGY='READ_ONLY'`
- Installed `tom_calendar/models.py` — `CalendarEvent` field nullability (`start_time`/`end_time`
  non-nullable, no unique DB constraint, `url`/`target_list` fields)
- Installed `tom_common/mixins.py`, `tom_common/views.py` — `SuperuserRequiredMixin` precedent
- `solsys_code/tests/test_campaign_views.py` — existing Phase 15 test fixture/assertion patterns
- `.planning/research/SUMMARY.md` — Pitfall 3 (calendar collision namespace), Pitfall 6 (approval
  race conditions), Pitfall 7 (honeypot + notification)
- `.planning/STATE.md` — Phase 14 decision log (`CampaignRun` has no `modified` field, by design)
- Local `pip show` (2026-07-03) — Django 5.2.15, django-tables2 3.0.0, django-filter 24.3,
  django-crispy-forms 2.4, crispy-bootstrap4 2024.10

### Secondary (MEDIUM confidence)
- `[CITED: django-tables2 docs]` — `MultiTableMixin` shape and prefixing behavior (web-search
  verified against readthedocs)
- `[CITED: Django docs]` — console/locmem email backend behavior, test-runner auto-substitution
  of `locmem`
- `[CITED: community race-condition writeups]` — `select_for_update()` vs. conditional
  `.update()` tradeoffs, SQLite's lack of real row-level locking
- `[CITED: web-search honeypot guidance]` — hidden-field naming/implementation best practices

### Tertiary (LOW confidence)
None — all findings this phase were either verified directly against this codebase's source or
cross-checked against official/community documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all versions confirmed via local `pip show`
- Architecture: HIGH — every pattern grounded in direct reads of this codebase's existing models/
  views/utils, or an installed TOM Toolkit precedent (`SuperuserRequiredMixin`)
- Pitfalls: HIGH — Pitfalls 1-4 and 7 are derived from direct code inspection (model field
  definitions, constraints, settings); Pitfalls 5-6 are MEDIUM (web-search-verified best practice,
  not codebase-specific facts)

**Research date:** 2026-07-03
**Valid until:** 2026-08-02 (30 days — stable Django/library versions, no fast-moving external
APIs in this phase's scope)
