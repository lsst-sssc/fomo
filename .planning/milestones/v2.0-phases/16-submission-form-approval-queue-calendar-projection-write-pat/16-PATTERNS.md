# Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path) - Pattern Map

**Mapped:** 2026-07-04
**Files analyzed:** 9 (new) + 3 (modified)
**Analogs found:** 12 / 12

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `solsys_code/campaign_forms.py` (new) | form | request-response | `solsys_code/forms.py::EphemerisForm` | exact (crispy `forms.Form`, non-`ModelForm`) |
| `solsys_code/mixins.py` (new) | middleware/guard | request-response | installed `tom_common/mixins.py::SuperuserRequiredMixin` | exact (identical shape, different flag) |
| `solsys_code/campaign_views.py` — `CampaignRunSubmissionView` (new class) | controller | request-response (create) | `solsys_code/campaign_views.py::CampaignRunTableView` (file conventions) + Django's generic `FormView` | role-match |
| `solsys_code/campaign_views.py` — `ApprovalQueueView` (new class) | controller | request-response (read, staff-gated) | `solsys_code/campaign_views.py::CampaignRunTableView`/`CampaignListView` | role-match |
| `solsys_code/campaign_views.py` — approve/reject action view(s) (new) | controller | request-response (atomic update + event-driven side effect) | `solsys_code/campaign_utils.py::insert_or_create_campaign_run` (atomic update contract) | role-match |
| `solsys_code/campaign_views.py::CampaignRunTableView.get_queryset` (modify) | controller | CRUD (queryset filter) | itself — extend existing `if self.request.user.is_staff` branch | exact (same file/method) |
| `solsys_code/campaign_tables.py` (extend, optional `ApprovalQueueTable`) | component (table def) | transform | `solsys_code/campaign_tables.py::CampaignRunTable` | exact |
| `solsys_code/campaign_urls.py` (extend) | route | request-response | `solsys_code/campaign_urls.py` (existing file, extend) | exact |
| `src/templates/campaigns/campaignrun_submit_form.html` (new) | component (template) | request-response | `src/templates/*/ephem_form.html` (crispy form render, see forms.py analog) | role-match |
| `src/templates/campaigns/submission_thanks.html` (new) | component (template) | request-response | existing `campaigns/campaign_list.html` / `campaignrun_table.html` (simple template shape) | partial |
| `src/templates/campaigns/approval_queue.html` (new) | component (template) | request-response | `src/templates/campaigns/campaignrun_table.html` (two-table variant) | role-match |
| `solsys_code/apps.py::nav_items` (modify) | config | request-response | itself — extend existing list | exact |
| `src/fomo/settings.py` (add `EMAIL_BACKEND`) | config | — | itself | exact |
| `solsys_code/tests/test_campaign_submission.py` (new) | test | request-response | `solsys_code/tests/test_campaign_views.py` | exact (fixture/assertion conventions) |
| `solsys_code/tests/test_campaign_approval.py` (new) | test | request-response / event-driven | `solsys_code/tests/test_campaign_views.py` | exact |
| `solsys_code/tests/test_campaign_views.py` (extend for D-09) | test | CRUD | itself | exact |

## Pattern Assignments

### `solsys_code/campaign_forms.py` (form, request-response) — NEW FILE

**Analog:** `solsys_code/forms.py::EphemerisForm` (lines 1-77)

**Imports pattern** (`solsys_code/forms.py` lines 1-9):
```python
from datetime import datetime, timedelta, timezone

from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Div, Fieldset, Layout, Submit
from django import forms
from django.urls import reverse

from solsys_code.solsys_code_observatory.models import Observatory
```
For `campaign_forms.py`, swap the last import for `from tom_targets.models import TargetList`.

**Core pattern — plain `forms.Form` + `FormHelper` built in `__init__`** (`solsys_code/forms.py` lines 12-42):
```python
class EphemerisForm(forms.Form):
    """
    This form is for requesting an ephemeris of a Target object
    """
    target_id = forms.IntegerField(required=True, widget=forms.HiddenInput())
    start_date = forms.DateTimeField(required=True, help_text=f'Start {time_text}')
    ...
    site_code = forms.ModelChoiceField(
        Observatory.objects.filter(altitude__gt=0).order_by('name'), blank=False, required=True
    )
    full_precision = forms.BooleanField(
        required=False, initial=True, help_text='Whether to show the full results precision'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        ...
        self.helper.form_action = reverse('makeephem', kwargs={'pk': target_id})
        self.helper.layout = Layout(
            HTML("""<p>...</p>"""),
            'target_id',
            Fieldset('Ephemeris Parameters', Div(Div('start_date', 'end_date', css_class='col'), ...)),
            FormActions(Submit('confirm', 'Create Ephemeris'), HTML(f'<a ...>Cancel</a>')),
        )
```
This is the file's ONLY existing form and establishes the exact idiom (bare `forms.Form`, not
`ModelForm` — no existing `ModelForm` precedent exists in this codebase to accidentally copy from,
which reinforces RESEARCH.md Pitfall 3's warning). Copy the `FormHelper()` + `Layout(Fieldset(...),
FormActions(Submit(...)))` shape directly; RESEARCH.md's Pattern 1 code example
(`CampaignRunSubmissionForm`) already adapts it field-for-field for this phase's honeypot +
`required=False` needs — use that as the concrete template, this analog as the house-style proof.

**No error-handling/validation excerpt needed** — `EphemerisForm` has no custom `clean()`; the new
form's only custom validation is the honeypot's non-raising `clean_alt_contact_info` (RESEARCH.md
Pattern 1), which has no existing in-repo precedent — write it fresh per RESEARCH.md's example.

---

### `solsys_code/mixins.py` (middleware/guard, request-response) — NEW FILE

**Analog:** installed `tom_common/mixins.py::SuperuserRequiredMixin` (verified via research; not
part of this repo's own source but installed and directly importable/inspectable)

```python
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import user_passes_test

class SuperuserRequiredMixin():
    @method_decorator(user_passes_test(lambda u: u.is_superuser))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```

Copy verbatim, renaming to `StaffRequiredMixin` and swapping `u.is_superuser` → `u.is_staff`:
```python
class StaffRequiredMixin:
    """Redirect to LOGIN_URL unless request.user.is_staff (D-01 approval-queue gate)."""

    @method_decorator(user_passes_test(lambda u: u.is_staff))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
This is the **first** hard-gate (redirect-on-fail) view mixin in `solsys_code/` — Phase 15's views
only soft-filter data (`AUTH_STRATEGY='READ_ONLY'` keeps everything 200-OK by default), so there is
no in-repo analog for this pattern; the TOM Toolkit-installed `SuperuserRequiredMixin` is the
closest and only real precedent, already used by `tom_common.views.GroupCreateView`/`UserCreateView`.

---

### `solsys_code/campaign_views.py` — `CampaignRunSubmissionView` (controller, request-response create) — NEW CLASS in EXISTING FILE

**Analog:** `solsys_code/campaign_views.py` itself (module docstring/import conventions, lines 1-19) + Django's generic `FormView`

**Imports pattern to extend with** (existing file, lines 10-19):
```python
from django.db.models import Count
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from tom_targets.models import TargetList

from .campaign_filters import CampaignRunFilterSet
from .campaign_tables import CampaignRunTable
from .models import CampaignRun
```
Add: `from django.views.generic import FormView`, `from django.shortcuts import redirect`,
`from django.contrib.auth.models import User`, `from django.core.mail import send_mail`,
`from django.urls import reverse`, `from .campaign_forms import CampaignRunSubmissionForm`,
`from .mixins import StaffRequiredMixin`.

**Module docstring convention to preserve** (existing file, lines 1-8) — note it deliberately
avoids importing `solsys_code.views`/`.ephem_utils` (the 1.6GB SPICE-kernel-loading module) —
new views must keep this discipline (never `import solsys_code.views`).

**Core pattern — honeypot short-circuit + create + notify** (from RESEARCH.md Pattern 3, concrete
and ready to copy):
```python
def form_valid(self, form):
    if form.cleaned_data.get('alt_contact_info'):
        return redirect('campaigns:submission_thanks')
    run = CampaignRun.objects.create(
        campaign=form.cleaned_data['campaign'],
        telescope_instrument=form.cleaned_data['telescope_instrument'],
        site_raw=form.cleaned_data['site_raw'],
        # ... remaining cleaned_data fields ...
    )
    self._notify_staff(run)
    return redirect('campaigns:submission_thanks')
```
Follow with Pitfall 4's `try/except IntegrityError` around the `.objects.create()` call (the
`UniqueConstraint(['campaign', 'telescope_instrument', 'ut_start'])`, `solsys_code/models.py`
lines ~114-123, can legitimately collide on two independent public submissions).

**Staff-notification pattern** (RESEARCH.md "Staff notification email" example, copy verbatim):
```python
def _notify_staff(self, run):
    recipients = list(
        User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True)
    )
    if not recipients:
        return
    queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
    send_mail(
        subject='FOMO: new campaign run submission pending review',
        message=f'A new run submission is pending review: {queue_url}',
        from_email=None,
        recipient_list=recipients,
        fail_silently=True,
    )
```

---

### `solsys_code/campaign_views.py` — `ApprovalQueueView` (controller, request-response read, staff-gated) — NEW CLASS

**Analog:** `solsys_code/campaign_views.py::CampaignListView` (lines 83-96) for the `TemplateView`/
`ListView`-with-`get_context_data` shape; `CampaignRunTableView.get_context_data` (lines 76-80) for
the "look up a related object, add to context" idiom.

```python
class CampaignListView(ListView):
    queryset = (
        TargetList.objects.filter(campaign_runs__isnull=False).distinct().annotate(run_count=Count('campaign_runs'))
    )
    template_name = 'campaigns/campaign_list.html'
    context_object_name = 'campaigns'
```
```python
def get_context_data(self, **kwargs):
    """Add the campaign (TargetList) to context for the page heading."""
    context = super().get_context_data(**kwargs)
    context['campaign'] = get_object_or_404(TargetList, pk=self.kwargs['pk'])
    return context
```
Combine with `StaffRequiredMixin` (new file above) and RESEARCH.md Pattern 5's two-independent-
tables `get_context_data` body (already concrete — copy that directly, it is not abstracted here
because it's phase-specific, not a codebase idiom to extract).

---

### `solsys_code/campaign_views.py` — approve/reject action view (controller, request-response atomic update + event-driven side effect) — NEW CLASS

**Analog for the atomic no-churn contract:** `solsys_code/campaign_utils.py::insert_or_create_campaign_run`
(lines 264-297) and `solsys_code/calendar_utils.py::insert_or_create_calendar_event` (lines 296-332)

```python
# campaign_utils.py — same "changed = [...]; if changed: save" no-churn idiom the approval
# view's downstream insert_or_create_calendar_event() call already relies on:
def insert_or_create_campaign_run(lookup: dict[str, Any], fields: dict[str, Any]) -> tuple[CampaignRun, str]:
    run, created = CampaignRun.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return run, 'created'
    changed = [f for f, v in fields.items() if getattr(run, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(run, f, v)
        run.save(update_fields=list(fields.keys()))
        return run, 'updated'
    return run, 'unchanged'
```
```python
# calendar_utils.py — the EXACT function CAL-01 must call unchanged (never construct
# CalendarEvent directly):
def insert_or_create_calendar_event(lookup: dict[str, Any], fields: dict[str, Any]) -> tuple[CalendarEvent, str]:
    event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)
    if created:
        return event, 'created'
    changed = [f for f, v in fields.items() if getattr(event, f) != v]
    if changed:
        for f, v in fields.items():
            setattr(event, f, v)
        event.save(update_fields=list(fields.keys()) + ['modified'])
        return event, 'updated'
    return event, 'unchanged'
```
Call signature to use in the approve action (per CONTEXT.md canonical refs — `CAMPAIGN:{pk}` key):
```python
insert_or_create_calendar_event(
    {'url': f'CAMPAIGN:{run.pk}'},
    fields={'title': ..., 'description': run.observation_details, 'start_time': run.ut_start,
            'end_time': run.ut_end, 'target_list': run.campaign, 'telescope': run.telescope_instrument},
)
```

**Atomicity + conditional projection pattern** — this phase's genuinely new pattern (no in-repo
precedent for `.filter(pk=pk, approval_status=...).update(...)` exists yet); use RESEARCH.md
Pattern 4's `post()` body verbatim (already fully worked out with `updated_count == 1` gating and
`messages.success/warning` feedback).

**Site resolution reuse (D-07):** `solsys_code/campaign_utils.py::resolve_site` (lines 85-170) —
call `site, needs_review = resolve_site(run.site_raw)` unchanged; never re-implement any part of
the 3-tier logic.

---

### `solsys_code/campaign_views.py::CampaignRunTableView.get_queryset` (modify, D-09 filter)

**Analog:** itself, existing D-13 branch (lines 62-68):
```python
def get_queryset(self):
    """Restrict to this campaign; non-staff get a PII-safe .values() queryset (D-13)."""
    campaign_pk = self.kwargs['pk']
    qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
    if self.request.user.is_staff:
        return qs.select_related('site')
    return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)
```
D-09's change is additive to the non-staff branch only — insert `.exclude(approval_status=
CampaignRun.ApprovalStatus.PENDING_REVIEW)` before the `.values(...)` call, per RESEARCH.md
Pattern 6:
```python
qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)
```
Keep the queryset-level (not template-level) restriction discipline — this is the same
`ALLOWED_FIELDS_FOR_NON_STAFF` module-level constant (lines 26-44) already enumerated explicitly
rather than introspected, per its own docstring rationale; do not touch that list itself, D-09
only changes row *filtering*, not column *selection*.

---

### `solsys_code/campaign_tables.py` (extend) — optional `ApprovalQueueTable`

**Analog:** `solsys_code/campaign_tables.py::CampaignRunTable` (whole file, 127 lines) — badge
rendering (`render_run_status`/`render_approval_status`, lines 82-107) and the `Meta` shape (lines
51-74) are directly reusable; if an `ApprovalQueueTable` subclass is needed only to add an
Approve/Reject action column, subclass `CampaignRunTable` and add one `tables.Column` with a
`render_actions` method following the same `format_html(...)` idiom used by `render_open_to_
collaboration` (lines 123-127):
```python
def render_open_to_collaboration(self, value):
    if value:
        return format_html('<i class="fa fa-check text-success" aria-hidden="true" title="Yes"></i>')
    return format_html('<i class="fa fa-times text-muted" aria-hidden="true" title="No"></i>')
```
RESEARCH.md's Standard Stack table already flags that the two approval-queue tables (pending +
decided) should be built as **two explicit `CampaignRunTable`-shaped instances** rather than
`MultiTableMixin` — no new base class strictly required unless action buttons are added, in which
case subclass minimally as shown.

---

### `solsys_code/campaign_urls.py` (extend)

**Analog:** itself (whole file, 15 lines):
```python
"""FOMO campaigns URL conf -- the per-campaign table read path (VIEW-01/03/04).

Mirrors solsys_code/calendar_urls.py's structure: app_name + a flat urlpatterns list.
"""

from django.urls import path

from solsys_code.campaign_views import CampaignListView, CampaignRunTableView

app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
```
Extend the import line and `urlpatterns` list with the new views, e.g.:
```python
path('<int:pk>/submit/', CampaignRunSubmissionView.as_view(), name='submit'),
path('submission-thanks/', TemplateView.as_view(template_name='campaigns/submission_thanks.html'),
     name='submission_thanks'),
path('approval-queue/', ApprovalQueueView.as_view(), name='approval_queue'),
path('<int:pk>/approve/', CampaignRunDecisionView.as_view(), name='approve'),  # or a single decision view + POST 'action'
```
Keep the flat-list, `app_name`-namespaced, single-file convention — do not split into a router or
separate URL module.

---

### `solsys_code/apps.py::nav_items` (modify)

**Analog:** itself (lines 23-33) — existing single-entry list; append the approval-queue entry as
a second dict in the same list (or gate it staff-only in the partial template):
```python
def nav_items(self):
    """
    Integration point for adding entries to the navbar (VIEW-02/D-03).
    """
    return [
        {
            'partial': f'{self.name}/partials/campaigns_nav_link.html',
            'context': 'src.templatetags.solsys_code_extras.campaigns_nav_link',
            'position': 'left',
        }
    ]
```
D-01 says the approval-queue page must be "reachable from the existing Campaigns navbar entry" —
simplest compliant approach is a staff-only link *inside* `campaigns_nav_link.html`'s existing
dropdown/partial (check `{% if request.user.is_staff %}` in the template) rather than a second
top-level `nav_items()` entry, unless the planner prefers a separate entry — either is consistent
with this analog's shape.

---

### Test files

**Analog:** `solsys_code/tests/test_campaign_views.py` (lines 1-60 shown; full file 238 lines)

**Imports + fixture conventions** (lines 1-54):
```python
from datetime import date, datetime, timedelta, timezone

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CampaignRun

class CampaignViewTestBase(TestCase):
    """Shared fixture: one campaign with 30 CampaignRun rows, one empty campaign, one staff user."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.empty_campaign = TargetList.objects.create(name='Empty Campaign')
        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)
        ...
```
CLAUDE.md mandates `NonSiderealTargetFactory` (never `SiderealTargetFactory`) wherever a `Target`
fixture is needed — this file already follows that convention (imported even though not directly
shown fixturing a `Target` here; campaigns use `TargetList`, not `Target`, directly). New test
modules (`test_campaign_submission.py`, `test_campaign_approval.py`) should copy this exact
`setUpTestData`/`User.objects.create_user(..., is_staff=True)` shape, and use
`self.client.login(username=..., password=...)` (Django `TestCase` convention already established
elsewhere in this file) for staff-gated assertions, plus `django.core.mail.outbox` for
SUBMIT-05 email assertions (no existing precedent in this repo for `mail.outbox` — new pattern,
but stdlib-Django, no adaptation needed).

## Shared Patterns

### No-churn create-or-update ("upsert" avoided per CLAUDE.md terminology note)
**Source:** `solsys_code/campaign_utils.py::insert_or_create_campaign_run` (lines 264-297),
`solsys_code/calendar_utils.py::insert_or_create_calendar_event` (lines 296-332)
**Apply to:** the approve action's `CalendarEvent` projection (CAL-01/02/03) — call
`insert_or_create_calendar_event()` unchanged, never construct `CalendarEvent` directly.

### Staff-only gating
**Source:** installed `tom_common/mixins.py::SuperuserRequiredMixin` (adapted as new
`solsys_code/mixins.py::StaffRequiredMixin`)
**Apply to:** `ApprovalQueueView`, approve/reject action view(s).

### "Restrict at the queryset, not the template" discipline
**Source:** `solsys_code/campaign_views.py::CampaignRunTableView.get_queryset` (lines 62-68),
`ALLOWED_FIELDS_FOR_NON_STAFF` (lines 26-44)
**Apply to:** D-09's approval-status `.exclude()` addition to the same method.

### Never raise for expected messy/optional data
**Source:** `solsys_code/campaign_utils.py::resolve_site`/`parse_obs_window` (module docstring,
lines 1-9, and each function's own docstring) — "never raise for expected messy data; return a
usable value plus an explicit flag"
**Apply to:** the submission form's honeypot `clean_alt_contact_info` (never raises) and the
approve action's `IntegrityError` handling on a natural-key collision (Pitfall 4) — degrade
gracefully with a friendly error, don't 500.

### crispy-forms `FormHelper` + `Layout(Fieldset(...), FormActions(Submit(...)))`
**Source:** `solsys_code/forms.py::EphemerisForm.__init__` (lines 34-77)
**Apply to:** `CampaignRunSubmissionForm.__init__`.

## No Analog Found

None outright — every new file has at least a role-match analog in this codebase or the installed
TOM Toolkit package. Two constructs are genuinely new to this repo (called out above rather than
listed here since RESEARCH.md already supplies complete, planner-ready code for them):
- The atomic conditional `.filter(pk=pk, approval_status=<expected>).update(...)` double-approve
  guard (RESEARCH.md Pattern 4) — no prior atomic-update precedent in `solsys_code/`.
- A hard staff-gate `dispatch()` mixin used *inside this app's own module* (only the installed
  TOM Toolkit package had the shape to copy from).

## Metadata

**Analog search scope:** `solsys_code/` (forms.py, campaign_views.py, campaign_utils.py,
calendar_utils.py, campaign_tables.py, campaign_urls.py, campaign_filters.py, apps.py, models.py,
tests/test_campaign_views.py), installed `tom_common/mixins.py`
**Files scanned:** 11
**Pattern extraction date:** 2026-07-04
