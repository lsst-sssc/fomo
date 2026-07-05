# Phase 15: Per-Campaign Table View (Read Path) - Pattern Map

**Mapped:** 2026-07-03
**Files analyzed:** 12 (new/modified)
**Analogs found:** 12 / 12 (all matched — either strong role/data-flow analog or explicit
"no direct analog, use RESEARCH.md verified code" per Common Pitfalls / Architecture Patterns)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_views.py` (`CampaignRunTableView`) | controller (class-based view) | CRUD (read, filtered/paginated) | `solsys_code/views.py::MakeEphemerisView` (`FormView`) + RESEARCH.md Pattern 1 (`SingleTableMixin`+`FilterView`, no local analog for this composition) | role-match (view structure), RESEARCH-sourced (table+filter composition) |
| `solsys_code/campaign_views.py` (`CampaignListView`) | controller (class-based `ListView`) | CRUD (read, unsorted list) | `solsys_code/solsys_code_observatory/views.py::ObservatoryList` | exact |
| `solsys_code/campaign_tables.py` (`CampaignRunTable`) | component (django-tables2 Table) | transform (queryset → rendered rows) | none in-repo (first table consumer) — `solsys_code/templatetags/calendar_display_extras.py` for the `render_`-method-as-badge pattern shape | role-match / RESEARCH-sourced |
| `solsys_code/campaign_filters.py` (`CampaignRunFilterSet`) | component (django-filter FilterSet) | request-response (GET params → queryset filter) | none in-repo (first filter consumer) — RESEARCH.md Pattern 4 | no in-repo analog, RESEARCH-sourced |
| `solsys_code/campaign_urls.py` | route (URLconf) | request-response | `solsys_code/calendar_urls.py` | exact |
| `src/fomo/urls.py` (modify — add `campaign_urls` include) | route (URLconf) | request-response | same file, existing `calendar/` include line | exact |
| `solsys_code/apps.py` (modify — extend `target_detail_buttons()`, add `nav_items()`) | config (AppConfig integration hooks) | event-driven (hook called per-render) | same file, existing `target_detail_buttons()`/`data_services()` methods | exact |
| `src/templatetags/solsys_code_extras.py` (modify — add `campaign_links`, `campaigns_nav_link`) | utility (template tag / inclusion tag) | transform (context → partial context) | same file, existing `ephem_button` | exact |
| `src/templates/solsys_code/partials/campaign_links.html` | component (template partial) | request-response (render) | `src/templates/solsys_code/partials/ephem_button.html` (implied sibling; render via `show_individual_app_partial`) | role-match |
| `src/templates/solsys_code/partials/campaigns_nav_link.html` | component (template partial) | request-response (render) | none (first `nav_items()` partial) — RESEARCH.md Pattern 3 verified against `tom_common` source | no in-repo analog, RESEARCH-sourced |
| `src/templates/campaigns/campaign_list.html` | component (template, list page) | request-response (render) | `solsys_code/solsys_code_observatory/templates/.../observatory_list.html` (if present) or generic TOM `object_list` template convention | role-match |
| `src/templates/campaigns/campaignrun_table.html` | component (template, table page) | request-response (render) | none in-repo (first `{% render_table %}` consumer) — django-tables2 doc convention | no in-repo analog, RESEARCH-sourced |
| `solsys_code/tests/test_campaign_views.py` | test | request-response (Django test `Client`) | `solsys_code/tests/test_campaign_models.py` (fixture/style conventions only, not view-testing shape) | role-match (project conventions), no in-repo view-test analog |

## Pattern Assignments

### `solsys_code/campaign_views.py` (controller, request-response)

**Analog for class-based view structure:** `solsys_code/views.py::MakeEphemerisView` (imports/
docstring/type-hint conventions) — but the actual mixin composition must follow RESEARCH.md
Architecture Pattern 1 verbatim (verified against installed package source), since no existing
FOMO view combines `SingleTableMixin`+`FilterView`.

**Imports pattern** (project convention, from `solsys_code/views.py` top of file):
```python
import logging
from django.views.generic import View, FormView
```
Follow this style: stdlib logging import, `logger = logging.getLogger(__name__)`, Django class
imports grouped before local imports (`from .models import ...`, `from .forms import ...`).

**Core pattern — table+filter view** (copy near-verbatim from RESEARCH.md Architecture Pattern 1,
already verified against installed `django_tables2`/`django_filters` 3.0.0/24.3 source):
```python
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin

from .campaign_filters import CampaignRunFilterSet
from .campaign_tables import CampaignRunTable
from .models import CampaignRun


class CampaignRunTableView(SingleTableMixin, FilterView):
    model = CampaignRun
    table_class = CampaignRunTable
    filterset_class = CampaignRunFilterSet
    template_name = 'campaigns/campaignrun_table.html'
    table_pagination = {'per_page': 25}  # D-11

    def get_queryset(self):
        campaign_pk = self.kwargs['pk']
        qs = CampaignRun.objects.filter(campaign_id=campaign_pk).select_related('site')
        if not self.request.user.is_staff:
            # D-13: SQL SELECT itself must never fetch contact_person/contact_email
            # for non-staff — enumerate the exact D-09 column list explicitly here,
            # do not introspect Meta (see RESEARCH.md Open Question 1 re: site__<field>).
            return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)
        return qs

    def get_table_kwargs(self):
        # Belt-and-suspenders per D-13/Pitfall 1: also exclude from rendered table.
        if not self.request.user.is_staff:
            return {'exclude': ('contact_person', 'contact_email')}
        return {}
```

**MRO order is load-bearing** — `SingleTableMixin` MUST be declared before `FilterView`
(RESEARCH.md Pitfall 4). Getting this backwards silently unfilters the table or raises
`AttributeError`.

**Analog for plain list view:** `solsys_code/solsys_code_observatory/views.py::ObservatoryList`
— read this file's `ListView` subclass for the project's `queryset`/`context_object_name`
conventions (annotate + filter pattern for D-03: `TargetList.objects.filter(campaign_runs__isnull=False).distinct()`, per RESEARCH.md Pitfall 3).

---

### `solsys_code/campaign_tables.py` (component, transform)

**No in-repo analog** (first django-tables2 consumer). Use RESEARCH.md Pitfall 2's exact
recommendation — do NOT rely on automatic `get_FOO_display()`:

```python
import django_tables2 as tables
from django.utils.html import format_html

from .models import CampaignRun

APPROVAL_BADGE_CLASSES = {
    CampaignRun.ApprovalStatus.PENDING_REVIEW: 'badge-warning',
    CampaignRun.ApprovalStatus.APPROVED: 'badge-success',
    CampaignRun.ApprovalStatus.REJECTED: 'badge-danger',
}


class CampaignRunTable(tables.Table):
    class Meta:
        model = CampaignRun
        fields = (...)  # D-09 spreadsheet-parity column order
        order_by = ('-obs_date',)  # D-10

    def render_run_status(self, value):
        return CampaignRun.RunStatus(value).label

    def render_approval_status(self, value):
        css = APPROVAL_BADGE_CLASSES.get(value, 'badge-secondary')
        label = CampaignRun.ApprovalStatus(value).label
        return format_html('<span class="badge {}">{}</span>', css, label)
```

**Styling-mechanism precedent** (shape only, not palette): `solsys_code/templatetags/
calendar_display_extras.py` lines 47-100 (`proposal_color`/`status_border_css`) — the pattern
of "fixed internal constant dict, raw DB value never echoed directly into markup" should be
mirrored for `APPROVAL_BADGE_CLASSES`, but use a **plain 3-entry dict** (not the hash-based
`PROPOSAL_PALETTE` approach — that hashing is for an unbounded value set, `approval_status` is a
fixed 3-value `TextChoices`), per RESEARCH.md "Don't Hand-Roll" table.

---

### `solsys_code/campaign_filters.py` (component, request-response)

**No in-repo analog** (first django-filter consumer). Copy verbatim from RESEARCH.md
Architecture Pattern 4 (verified against installed `django_filters` 24.3 source — `Meta.fields`
auto-generation does NOT produce multi-select for a `choices` `CharField`, must declare
explicitly):

```python
import django_filters

from .models import CampaignRun


class CampaignRunFilterSet(django_filters.FilterSet):
    run_status = django_filters.MultipleChoiceFilter(
        choices=CampaignRun.RunStatus.choices,
        label='Run status',
    )

    class Meta:
        model = CampaignRun
        fields = ['run_status', 'open_to_collaboration']
```

---

### `solsys_code/campaign_urls.py` (route, request-response)

**Analog:** `solsys_code/calendar_urls.py` (full file, 18 lines) — exact structural match:
module docstring explaining the namespace's purpose, `app_name = 'campaigns'`, flat
`urlpatterns` list.

```python
# solsys_code/calendar_urls.py (pattern to mirror)
from django.urls import path
from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
    ...
]
```
Apply directly:
```python
from django.urls import path

from solsys_code.campaign_views import CampaignListView, CampaignRunTableView

app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    path('<int:pk>/', CampaignRunTableView.as_view(), name='table'),
]
```

**Wiring into `src/fomo/urls.py`** — follow the existing `calendar/` include line exactly
(same file, lines 20-24):
```python
path('calendar/', include('solsys_code.calendar_urls', namespace='calendar')),  # DISPLAY-09 — before tom_common
```
Add a parallel line for campaigns, placed before `path('', include('tom_common.urls'))` (same
"more specific paths before the catch-all TOM include" ordering convention).

---

### `solsys_code/apps.py` (config, event-driven)

**Analog:** same file, existing `target_detail_buttons()`/`data_services()` methods (24 lines
total — read in full above).

**Current state:**
```python
def target_detail_buttons(self):
    return [
        {
            'partial': f'{self.name}/partials/ephem_button.html',
            'context': 'src.templatetags.solsys_code_extras.ephem_button',
        }
    ]
```

**Extension pattern** (add second dict entry for D-01/D-02, add new `nav_items()` method for
D-03 — copy directly from RESEARCH.md Architecture Patterns 2/3, verified against installed
`tom_targets`/`tom_common` source):
```python
def target_detail_buttons(self):
    return [
        {
            'partial': f'{self.name}/partials/ephem_button.html',
            'context': 'src.templatetags.solsys_code_extras.ephem_button',
        },
        {
            'partial': f'{self.name}/partials/campaign_links.html',
            'context': 'src.templatetags.solsys_code_extras.campaign_links',
        },
    ]

def nav_items(self):
    return [
        {
            'partial': f'{self.name}/partials/campaigns_nav_link.html',
            'context': 'src.templatetags.solsys_code_extras.campaigns_nav_link',
            'position': 'left',
        }
    ]
```
Note: `module_buttons.html` (`src/templates/tom_targets/partials/module_buttons.html`) requires
NO changes — its `{% if 'ephem_button' in button.context %}` check is a dict-key check that
never matches the new `campaign_links` partial, so it falls through to the generic
`{% show_individual_app_partial button %}` `{% else %}` branch already used by every other app.

---

### `src/templatetags/solsys_code_extras.py` (utility, transform)

**Analog:** same file, existing `ephem_button` inclusion tag (full file, 12 lines — read above).

```python
from django import template

register = template.Library()


@register.inclusion_tag('solsys_code/partials/ephem_button.html', takes_context=True)
def ephem_button(context):
    context = {'button_text': 'Ephemeris'}
    return context
```

**Extension pattern** (D-01/D-02 campaign_links, D-03 campaigns_nav_link — copy from
RESEARCH.md Architecture Patterns 2/3):
```python
from tom_targets.models import TargetList


@register.inclusion_tag('solsys_code/partials/campaign_links.html', takes_context=True)
def campaign_links(context):
    target = context.get('target')
    campaigns = (
        TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()
        if target
        else TargetList.objects.none()
    )
    return {'campaigns': campaigns}


@register.inclusion_tag('solsys_code/partials/campaigns_nav_link.html', takes_context=True)
def campaigns_nav_link(context):
    return {}
```

---

### `src/templates/solsys_code/partials/campaign_links.html` (component, request-response)

**Analog:** `src/templates/solsys_code/partials/ephem_button.html` (implied sibling; not read
directly here — locate and mirror its `{% url %}` + Bootstrap button-class conventions before
writing). Loop version per D-02 (one link per campaign):
```html
{% for campaign in campaigns %}
    <a href="{% url 'campaigns:table' pk=campaign.pk %}" class="btn btn-info">{{ campaign.name }}</a>
{% endfor %}
```

### `src/templates/solsys_code/partials/campaigns_nav_link.html` (component, request-response)

**No in-repo analog** (first `nav_items()` partial) — copy from RESEARCH.md Architecture
Pattern 3:
```html
<li class="nav-item {% if request.resolver_match.namespace == 'campaigns' %}active{% endif %}">
    <a class="nav-link" href="{% url 'campaigns:list' %}">Campaigns</a>
</li>
```

### `src/templates/campaigns/campaign_list.html` / `campaignrun_table.html`

**Analog for list template:** locate `solsys_code/solsys_code_observatory/templates/**/observatory_list.html`
(the `ObservatoryList` view's paired template) for the project's base-template-extends and
Bootstrap table/list conventions.

**Table template:** no in-repo analog (first `{% render_table %}` consumer) — follow
django-tables2's documented `{% load django_tables2 %}` + `{% render_table table %}` convention,
plus a `<form>` wrapping the filter widgets from `{{ filter.form }}` (django-filter's standard
`FilterView` context key), matching Bootstrap4 form-rendering conventions already used elsewhere
in `src/templates/` (e.g. crispy-forms usage in `ephem_form.html`).

---

### `solsys_code/tests/test_campaign_views.py` (test, request-response)

**Analog (fixture/style conventions):** `solsys_code/tests/test_campaign_models.py` — read in
full above. Key conventions to copy:
```python
from django.test import TestCase
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory  # CLAUDE.md mandate — never SiderealTargetFactory

from solsys_code.models import CampaignRun


class TestCampaignRunFieldInventory(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            ...
            contact_person='Test Person',
            contact_email='test@example.com',
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )
```
No in-repo precedent exists for view-level tests (`Client()`) or `is_staff=True` user fixtures
in this codebase (RESEARCH.md Wave 0 Gaps confirms `grep -n is_staff` returns no hits) — this is
genuinely new test-fixture territory; use plain `django.contrib.auth.models.User.objects.create_user(..., is_staff=True)`
plus `self.client.force_login(staff_user)` vs. an anonymous `Client()` for the split required by
`TestContactFieldGating`.

---

## Shared Patterns

### View-layer PII gating (not template-only)
**Source:** RESEARCH.md Common Pitfalls §1 (verified against installed `django_tables2`/Django
`MultipleObjectMixin` source — no in-repo precedent exists for this pattern).
**Apply to:** `CampaignRunTableView.get_queryset()` and `get_table_kwargs()`.
```python
if not self.request.user.is_staff:
    return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)  # SQL never selects contact_* columns
```

### `TargetList` "is a campaign" derivation
**Source:** RESEARCH.md Common Pitfalls §3 — no FOMO model flag exists; always filter through
the reverse relation.
**Apply to:** `CampaignListView.get_queryset()` and `campaign_links` inclusion tag.
```python
TargetList.objects.filter(campaign_runs__isnull=False).distinct()                       # list page
TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()        # D-01
```

### Choice-field display via explicit `render_` methods (not automatic `get_FOO_display()`)
**Source:** RESEARCH.md Common Pitfalls §2.
**Apply to:** `CampaignRunTable.render_run_status` / `render_approval_status` — must work
identically whether `record` is a dict (`.values()` non-staff path) or a model instance (staff
path), so never depend on the automatic Django choice-display machinery.

### AppConfig integration hooks (`target_detail_buttons`, `nav_items`)
**Source:** `solsys_code/apps.py` (existing `target_detail_buttons()`), extended per RESEARCH.md
Patterns 2/3.
**Apply to:** All two new hook entries; both route through `tom_common`'s generic
`show_individual_app_partial` — no template-level special-casing needed for either.

## No Analog Found

Files with no close in-repo match — planner should use the RESEARCH.md verified code (all
sourced from directly-inspected installed package source, HIGH confidence) rather than
extrapolating from an unrelated in-repo file:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `solsys_code/campaign_tables.py` | component | transform | First `django-tables2` `Table` subclass in this codebase |
| `solsys_code/campaign_filters.py` | component | request-response | First `django-filter` `FilterSet` in this codebase |
| `src/templates/solsys_code/partials/campaigns_nav_link.html` | component | request-response | First `nav_items()` consumer/partial in this codebase |
| `src/templates/campaigns/campaignrun_table.html` | component | request-response | First `{% render_table %}` template in this codebase |
| Staff-vs-anonymous `Client` test fixture | test | request-response | No `is_staff` test precedent anywhere in the codebase (confirmed via grep in RESEARCH.md) |

## Metadata

**Analog search scope:** `solsys_code/`, `solsys_code/solsys_code_observatory/`,
`src/templates/`, `src/templatetags/`, `src/fomo/urls.py`, `solsys_code/tests/`
**Files scanned:** `solsys_code/apps.py`, `solsys_code/views.py`, `solsys_code/models.py`,
`solsys_code/calendar_urls.py`, `solsys_code/templatetags/calendar_display_extras.py`,
`solsys_code/solsys_code_observatory/views.py`, `solsys_code/solsys_code_observatory/models.py`,
`src/templatetags/solsys_code_extras.py`,
`src/templates/tom_targets/partials/module_buttons.html`, `src/fomo/urls.py`,
`solsys_code/tests/test_campaign_models.py`
**Pattern extraction date:** 2026-07-03
</content>
