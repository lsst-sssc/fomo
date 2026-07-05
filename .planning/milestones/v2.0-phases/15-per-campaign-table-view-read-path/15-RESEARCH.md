# Phase 15: Per-Campaign Table View (Read Path) - Research

**Researched:** 2026-07-03
**Domain:** django-tables2 / django-filter class-based views on Django 2.x-compatible TOM Toolkit; TOM Toolkit app-config integration hooks (`target_detail_buttons`, `nav_items`); view-layer PII gating.
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Campaign discovery & navigation**
- **D-01:** A Target's detail page finds "its" campaign(s) via **TargetList membership**, not
  via `CampaignRun.target`: `TargetList.objects.filter(targets=this_target,
  campaign_runs__isnull=False).distinct()`. This works even for rows where the optional
  `CampaignRun.target` FK was never set (D-07 from Phase 14 only guarantees auto-fill for
  single-target campaigns going forward, not universal population).
- **D-02:** If a Target belongs to 2+ qualifying campaigns, the target-detail integration point
  shows **one button/link per matching campaign** (each labeled with the `TargetList` name) —
  not a single button to an intermediate chooser page.
- **D-03:** The navbar "Campaigns" entry links to a **new dedicated campaigns list page** — a
  view listing every `TargetList` that has ≥1 `CampaignRun`, each linking to its per-campaign
  table. Not a reuse of TOM Toolkit's existing `tom_targets:targetgrouping` view.
- **D-04:** The campaign list page and per-campaign table are **open to anonymous visitors**,
  matching FOMO's existing `AUTH_STRATEGY='READ_ONLY'`/`OPEN` targets convention.

**Approval-status visibility**
- **D-05:** The table **shows all `CampaignRun` rows regardless of `approval_status`** for
  everyone (staff and non-staff/anonymous alike) in this phase — it does **not** filter to
  `approved`-only. Phase 16 planning must not assume Phase 15 already gates on `approval_status`
  — it doesn't.
- **D-06:** Consistent with D-05: staff and non-staff both see `rejected` rows in the main table.
- **D-07:** Default `run_status` filter state on page load is **unfiltered — show everything**.
- **D-08:** `approval_status` gets a **visually distinct badge/highlight** in the table. Follow
  the existing `calendar_display_extras.py` badge/color precedent for implementation style
  (planner/researcher's call on exact mechanism).

**Table columns, sort & paging**
- **D-09:** Column set is **spreadsheet-parity**: telescope_instrument, site, obs_date, UT
  start/end, filters_bandpass, run_status (badged per D-08), open_to_collaboration, contact
  (staff-only), plus observation_details, weather, observation_outcome, publication_plans,
  comments as columns (not hidden behind a detail link).
- **D-10:** Default sort order is **`obs_date`, most recent first**. `django-tables2` still lets
  users re-sort by any column.
- **D-11:** Pagination is **25 rows per page**.
- **D-12:** The `run_status` filter (VIEW-04) is **multi-select** — use `django-filter`'s
  multi-select-capable filter type (e.g. `MultipleChoiceFilter`).

**Staff-only contact gating**
- **D-13:** Non-staff/anonymous viewers get `contact_person`/`contact_email` columns **omitted
  from the table entirely** — not shown as masked/blank placeholders. Gate at the view layer (the
  columns never reach the template context for non-staff) — defense in depth, not just a
  template-level hide.
- **D-14:** Phase 15 adds **no contact/reach-out path** for anonymous visitors. VIEW-05's scope,
  deferred.

### Claude's Discretion
- **Staff check mechanism (D-13's "staff"):** Use `request.user.is_staff` (Django's built-in
  flag) — no new permission/group needed.
- Exact `django-tables2`/`django-filter` implementation details (Table subclass structure,
  FilterSet field wiring, template used).
- Exact URL names/paths for the new campaigns list view and per-campaign table view.
- Exact badge/styling mechanism for D-08's approval_status treatment (new template tag vs. reuse
  of `calendar_display_extras.py` patterns).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within Phase 15's scope. VIEW-05 (submitter contact opt-in) and the
Phase 16 `approval_status` filter (D-05 note) are explicitly out of scope here and already
tracked in REQUIREMENTS.md/ROADMAP.md, not newly deferred by this discussion.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VIEW-01 | User can view a per-campaign table of all its runs (sortable/paginated), replacing the spreadsheet | `SingleTableMixin` + `FilterView` composition (Architecture Patterns §1); `Table.Meta.order_by`/`table_pagination` for D-10/D-11 |
| VIEW-02 | User can reach a target's campaigns from its target-detail page; navbar exposes a campaigns entry | `target_detail_buttons()` extension (D-01/D-02, Architecture Patterns §2) + new `nav_items()` hook (Architecture Patterns §3) |
| VIEW-03 | Contact person/email are visible only to authenticated staff — excluded from view context for anonymous requests and proven by an anonymous-client test | Common Pitfalls §1 (view-layer PII gating: `.values()` dict-row strategy) |
| VIEW-04 | User can filter the table by lifecycle status and the open-to-collaboration flag | `django_filters.MultipleChoiceFilter` for `run_status` (D-12), `BooleanFilter` for `open_to_collaboration` (Architecture Patterns §4) |
</phase_requirements>

## Summary

This phase is the first real consumer of `django-tables2` (3.0.0) and `django-filter` (24.3) in
FOMO — both already installed transitively via `tomtoolkit==3.0.0a9` and listed in
`INSTALLED_APPS`, so **no new dependency needs to be added**. The standard, officially documented
composition is a class-based view that mixes `django_tables2.views.SingleTableMixin` with
`django_filters.views.FilterView` (in that MRO order), backed by a declarative `Table` subclass
and `FilterSet` subclass, following the community convention of splitting them into
`campaign_tables.py` / `campaign_filters.py` alongside a new `campaign_views.py`.

The two integration-hook mechanisms this phase must extend (`target_detail_buttons()` for D-01/D-02
and the new `nav_items()` for D-03) are both driven by `AppConfig` methods discovered by
`apps.get_app_configs()` and rendered through `tom_common`'s generic
`show_individual_app_partial` inclusion tag. Tracing the actual TOM Toolkit source
(`tom_targets/templatetags/targets_extras.py:get_buttons`) revealed that the target-detail
context method receives the **full rendering context** (including `context['target']`, already
set by `{% target_buttons object %}` before `{% get_buttons %}` runs) — so a single new
`target_detail_buttons()` entry, with one partial and one context method that queries D-01's
TargetList membership and returns a list of matching campaigns, satisfies D-02's "one link per
matching campaign" without needing multiple dict entries. By contrast, `nav_items()`'s context
method (`navbar_app_addons`) is called with an **empty dict** (`context_method({})`), so the new
navbar partial's context method takes no useful `context` input and should return a static
"Campaigns" link (its href is just `{% url 'campaigns:list' %}`, no lookups needed).

The highest-risk item is VIEW-03. A django-tables2 column `exclude=` (the documented way to
conditionally drop the `contact_person`/`contact_email` columns) only hides them from the
*rendered table markup* — it does **not** prevent Django's `ListView`/`MultipleObjectMixin`
machinery from also placing the full model instances into `context['object_list']` /
`context['campaignrun_list']`. Because D-13 explicitly requires the fields to "never reach the
template context," this research recommends restricting the underlying queryset itself for
non-staff requests (via `.values(*allowed_fields)`, producing dict rows with `contact_person`/
`contact_email` never selected by SQL and never present in any object in memory), not just hiding
the rendered column. This has a real tradeoff, documented in Common Pitfalls §1: `.values()` rows
are plain dicts, so django-tables2's automatic `get_FOO_display()` call for choice fields
(`run_status`, `approval_status`) — confirmed present in `django_tables2/rows.py` — is silently
skipped for dict rows (it only fires when the accessor's penultimate object `isinstance(...,
models.Model)`), so the badge/label rendering must be done manually via `render_<column>` methods
rather than relying on automatic choice-display, for both the staff and non-staff code paths (for
consistency, use manual `render_` methods unconditionally rather than only for the anonymous
path).

**Primary recommendation:** Build `campaign_views.py` with two views —
`CampaignListView(ListView)` for D-03's list page, and `CampaignRunTableView(SingleTableMixin,
FilterView)` for the per-campaign table — backed by `campaign_tables.py::CampaignRunTable` and
`campaign_filters.py::CampaignRunFilterSet`. Gate PII by overriding `get_queryset()` to select a
restricted field list via `.values()` when `not self.request.user.is_staff`, rather than relying
solely on the Table's `exclude=` kwarg.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Campaign/run table rendering (sort, paginate) | API/Backend (Django view) | Browser (client re-sort via link clicks, no JS) | `django-tables2` renders server-side HTML; sorting/paging are plain GET-param-driven requests, no client state |
| Lifecycle/collaboration filtering | API/Backend (Django view, `FilterSet`) | — | `django-filter` applies `.filter()` to the queryset server-side from GET params |
| PII gating (contact fields) | API/Backend (view `get_queryset`/`get_table_kwargs`) | — | Must happen before template rendering; per D-13 explicitly not a template-only concern |
| Campaign discovery from target detail | API/Backend (`AppConfig.target_detail_buttons` context method) | Database (TargetList membership query) | Existing TOM Toolkit hook pattern; query happens in the context method, not in templates |
| Navbar "Campaigns" entry | API/Backend (`AppConfig.nav_items`) | — | Same `tom_common` hook family as target_detail_buttons, no DB query needed (static link) |
| Approval-status badge styling | API/Backend (Table `render_` method or template tag) | Browser (CSS/Bootstrap badge classes) | Follows `calendar_display_extras.py` precedent: styling constants are Python-side, only CSS class/text delivered to template |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| django-tables2 | 3.0.0 (installed, transitive via `tomtoolkit==3.0.0a9`) [VERIFIED: pip show django-tables2 in project venv] | Sortable/paginated HTML table rendering from a queryset | Already in `INSTALLED_APPS`; de facto standard Django table library, actively maintained |
| django-filter | 24.3 (installed, transitive via `tomtoolkit==3.0.0a9`) [VERIFIED: pip show django-filter in project venv] | Declarative queryset filtering from GET params | Already in `INSTALLED_APPS`; de facto standard Django filtering library, integrates directly with django-tables2 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Django `ListView` / `MultipleObjectMixin` | Django (project-pinned via tomtoolkit) | Base class for `CampaignListView` (D-03's plain list page, no table/filter needed — just `TargetList.objects` with ≥1 `campaign_runs`) | Simple unsorted list of campaigns; no need for full table machinery there |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `django-tables2` + `django-filter` composed manually | `django-tables2`'s bundled `SingleTableView` alone, filtering by hand in `get_queryset` | Loses django-filter's declarative multi-select UI widget generation for D-12; more boilerplate for no benefit since django-filter is already installed |
| `.values()`-restricted queryset for PII gating | `django-tables2` `exclude=` kwarg only | Does not satisfy D-13's "never reach the template context" wording — `ListView`'s default `object_list`/`<model>_list` context keys still carry full model instances even when a Table column is excluded (see Common Pitfalls §1) |

**Installation:**
```bash
# No installation needed — django_tables2 and django_filters are already
# in INSTALLED_APPS (src/fomo/settings.py) and installed transitively via
# tomtoolkit==3.0.0a9. Confirmed via:
#   pip show django-tables2   -> Version: 3.0.0
#   pip show django-filter    -> Version: 24.3
```

**Version verification:** Ran `pip show django-tables2` / `pip show django-filter` in the
project's active venv (`/home/tlister/venv/fomo_venv`) — both resolve and are already imported
successfully by Django's app registry (proven by `tomtoolkit`'s own dependency chain, which
requires both). No `pyproject.toml` entry exists for either package directly; they are transitive
dependencies of `tomtoolkit==3.0.0a9` (`pip show tomtoolkit` lists both in `Requires:`).

## Package Legitimacy Audit

No new external packages are introduced by this phase. `django-tables2` and `django-filter` are
already installed (transitively via `tomtoolkit`) and already declared in `INSTALLED_APPS`; this
phase is simply their first real consumer. The Package Legitimacy Gate does not apply — nothing
new is being added to `pyproject.toml`.

**Packages removed due to [SLOP] verdict:** none — no new packages proposed.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
Anonymous or authenticated browser
        |
        | GET /campaigns/                      GET /campaigns/<pk>/
        v                                              v
+-------------------+                    +------------------------------+
| CampaignListView  |                    | CampaignRunTableView          |
| (ListView)         |                    | (SingleTableMixin + FilterView)|
+-------------------+                    +------------------------------+
        |                                        |            |
        | TargetList.objects                     |            | GET params (run_status=, open_to_collaboration=, sort=)
        | .annotate(Count('campaign_runs'))       v            v
        | .filter(campaign_runs__isnull=False)  CampaignRunFilterSet   CampaignRunTable
        | .distinct()                           (django_filters)        (django_tables2)
        v                                        |            |
   TargetList (campaign) rows                    | .filter()  | .order_by(), .paginate()
        |                                        v            v
        |                              CampaignRun queryset (filtered, sorted)
        |                                        |
        |                          request.user.is_staff?
        |                             /                    \
        |                          yes                       no
        |                           v                         v
        |                 full CampaignRun queryset   .values(*allowed_fields)
        |                 (all columns incl. contact)  (contact_person/contact_email
        |                                                 never selected by SQL)
        v                                        \            /
   campaigns/campaign_list.html          campaign_run_table.html
   (links to each campaign's table)      ({% render_table table %} + filter form)

Target detail page (existing)
        |
        | {% target_buttons object %} sets context['target']
        v
   {% get_buttons %} (tom_targets) --> app.target_detail_buttons()
        |
        v
   SolsysCodeConfig.target_detail_buttons() entry: partial=campaign_links.html,
   context=solsys_code_extras.campaign_links(context)
        |
        | TargetList.objects.filter(targets=context['target'],
        |                            campaign_runs__isnull=False).distinct()   [D-01]
        v
   campaign_links.html: {% for c in campaigns %}<a href="...">{{ c.name }}</a>{% endfor %}  [D-02]

Navbar (existing, every page)
        |
        v
   {% navbar_app_addons %} (tom_common) --> app.nav_items()
        |
        v
   SolsysCodeConfig.nav_items() entry: partial=campaigns_nav_link.html, context={} (static)
        |
        v
   campaigns_nav_link.html: <a href="{% url 'campaigns:list' %}">Campaigns</a>   [D-03]
```

### Recommended Project Structure
```
solsys_code/
├── campaign_views.py       # CampaignListView, CampaignRunTableView
├── campaign_tables.py      # CampaignRunTable (django_tables2.Table subclass)
├── campaign_filters.py     # CampaignRunFilterSet (django_filters.FilterSet subclass)
├── campaign_urls.py        # app_name='campaigns'; list + per-campaign table paths
├── apps.py                 # extend target_detail_buttons(); add nav_items()
src/
├── templatetags/
│   └── solsys_code_extras.py   # add campaign_links(context) alongside ephem_button(context)
├── templates/
│   ├── solsys_code/partials/
│   │   ├── campaign_links.html       # target-detail integration partial (D-01/D-02)
│   │   └── campaigns_nav_link.html   # navbar integration partial (D-03)
│   └── campaigns/
│       ├── campaign_list.html         # D-03 list page
│       └── campaignrun_table.html     # D-09 spreadsheet-parity table + filter form
```

This mirrors the existing `campaign_models.py`-vs-`models.py` split precedent set in Phase 14
(new campaign-specific concerns get their own `campaign_*.py` module rather than growing
`views.py`/`models.py` further), and matches the community convention of separating
`tables.py`/`filters.py` from `views.py` in django-tables2/django-filter projects.
[CITED: django-tables2 filtering docs — https://django-tables2.readthedocs.io/en/latest/pages/filtering.html]

### Pattern 1: SingleTableMixin + FilterView composition
**What:** A class-based view combining `django_tables2.views.SingleTableMixin` with
`django_filters.views.FilterView`, declared in that MRO order so `SingleTableMixin`'s
`get_context_data` (which calls `self.get_table_data()`) runs after `FilterView`'s `get()` has
already set `self.object_list = self.filterset.qs`.
**When to use:** Any list page needing both sorting/pagination and GET-param filtering — exactly
VIEW-01 + VIEW-04.
**Example:**
```python
# Source: https://django-tables2.readthedocs.io/en/latest/pages/filtering.html
# (confirmed against installed django-tables2 3.0.0 source:
#  django_tables2/views.py::SingleTableMixin.get_table_data falls back to
#  self.object_list, which django_filters/views.py::BaseFilterView.get() sets
#  from self.filterset.qs before get_context_data runs)
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
            allowed = [f.name for f in CampaignRun._meta.get_fields()
                       if f.name not in ('contact_person', 'contact_email')
                       and not f.is_relation or f.name in ('site', 'campaign', 'target')]
            # simplified for illustration -- planner should enumerate the exact
            # D-09 column field list explicitly rather than introspecting Meta.
            return qs.values(*allowed)
        return qs

    def get_table_kwargs(self):
        # Belt-and-suspenders: also drop the columns from the rendered table
        # even though get_queryset() already prevents the data from being
        # fetched for non-staff.
        if not self.request.user.is_staff:
            return {'exclude': ('contact_person', 'contact_email')}
        return {}
```

### Pattern 2: `target_detail_buttons()` multi-campaign link
**What:** Extend `SolsysCodeConfig.target_detail_buttons()` with a new entry whose context
method receives the full context (including `context['target']`, set by
`tom_targets.templatetags.targets_extras.target_buttons` before `get_buttons` runs) and returns
**all** matching campaigns in a list, letting the partial template loop and emit one link per
campaign (D-02) from a single `target_detail_buttons()` entry.
**When to use:** VIEW-02's target-detail integration point.
**Example:**
```python
# Source: traced from installed tom_targets 3.0.0a9 source
# (tom_targets/templatetags/targets_extras.py:73 target_buttons sets
#  context['target'] = target, then :503 get_buttons calls
#  context_method(context) -- confirmed by reading target_buttons.html ->
#  {% get_buttons %} -> module_buttons.html chain in this repo)

# solsys_code/apps.py
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

# src/templatetags/solsys_code_extras.py
from tom_targets.models import TargetList

@register.inclusion_tag('solsys_code/partials/campaign_links.html', takes_context=True)
def campaign_links(context):
    target = context.get('target')
    campaigns = TargetList.objects.filter(
        targets=target, campaign_runs__isnull=False
    ).distinct() if target else TargetList.objects.none()
    return {'campaigns': campaigns}
```
```html
<!-- solsys_code/partials/campaign_links.html -->
{% for campaign in campaigns %}
    <a href="{% url 'campaigns:table' pk=campaign.pk %}" class="btn btn-info">{{ campaign.name }}</a>
{% endfor %}
```
Note: this new partial is **not** the string `'ephem_button'`, so the existing customized
`src/templates/tom_targets/partials/module_buttons.html` (which special-cases
`'ephem_button' in button.context` — a check against dict *keys*, which never actually matches
since `ephem_button`'s returned context dict is `{'button_text': ...}`) falls through to its
`{% else %}` branch, which calls the generic `{% show_individual_app_partial button %}` — the
same path every other app's `target_detail_buttons()` entry already uses. **No change to
`module_buttons.html` is required** for the new campaign-links partial to render.

### Pattern 3: `nav_items()` hook (new for this codebase)
**What:** `tom_common`'s `navbar_app_addons` inclusion tag (rendered in
`tom_common/navbar_content.html` via `{% navbar_app_addons %}`) iterates
`apps.get_app_configs()` calling `app.nav_items()` (an `AttributeError` from apps without the
method is silently caught and skipped — confirmed in installed `tom_common` source). FOMO's
`SolsysCodeConfig` does not define `nav_items()` yet; this phase adds it as the first consumer.
Unlike `target_detail_buttons()`, `nav_items()`'s context method is called with an **empty
dict** (`new_context = context_method({})`), not the real page context — the navbar renders
identically on every page, so no per-request context is passed through.
**When to use:** D-03's navbar "Campaigns" entry.
**Example:**
```python
# Source: installed tom_common 3.0.0a9,
# tom_common/templatetags/tom_common_extras.py:navbar_app_addons
#   nav_items = app.nav_items()
#   ...
#   new_context = context_method({})   # <-- always an empty dict, not request context

# solsys_code/apps.py
def nav_items(self):
    return [
        {
            'partial': f'{self.name}/partials/campaigns_nav_link.html',
            'context': 'src.templatetags.solsys_code_extras.campaigns_nav_link',
            'position': 'left',
        }
    ]
```
```python
# src/templatetags/solsys_code_extras.py
@register.inclusion_tag('solsys_code/partials/campaigns_nav_link.html', takes_context=True)
def campaigns_nav_link(context):
    return {}  # static link, no per-request data needed
```
```html
<!-- solsys_code/partials/campaigns_nav_link.html -->
<li class="nav-item {% if request.resolver_match.namespace == 'campaigns' %}active{% endif %}">
    <a class="nav-link" href="{% url 'campaigns:list' %}">Campaigns</a>
</li>
```
If a `context` key is omitted entirely from the `nav_items()` dict, `navbar_app_addons` skips the
`import_string` call and uses `new_context = {}` directly — so a `context` templatetag is
optional here; only include one if useful. Since the FOMO active-nav-link check needs
`request` (present in `takes_context=True` inclusion tags automatically via `context['request']`
if the `django.template.context_processors.request` processor is enabled — confirmed present in
`TEMPLATES.OPTIONS.context_processors` in `src/fomo/settings.py`), the simplest correct approach
is the `takes_context=True` inclusion tag above rather than omitting `context` entirely.

### Pattern 4: `django-filter` multi-select `run_status` + boolean `open_to_collaboration`
**What:** `django_filters.FilterSet.Meta.fields` auto-generates a plain `CharFilter` (exact
match, single value) for a `CharField` with `choices` — **not** a multi-select filter. D-12
requires OR-semantics multi-select, so `run_status` must be **explicitly declared** on the
`FilterSet`, not left to auto-generation.
**When to use:** VIEW-04.
**Example:**
```python
# Source: installed django-filter 24.3,
# django_filters/filterset.py:FILTER_FOR_DBFIELD_DEFAULTS maps models.CharField -> CharFilter
# (no special-casing for `choices`), confirmed by reading the mapping directly.
# django_filters/filters.py:MultipleChoiceFilter docstring confirms OR-by-default semantics
# (matches D-12's "show planned OR observed" requirement with zero extra configuration).
import django_filters

from .models import CampaignRun


class CampaignRunFilterSet(django_filters.FilterSet):
    run_status = django_filters.MultipleChoiceFilter(
        choices=CampaignRun.RunStatus.choices,
        label='Run status',
    )
    # open_to_collaboration is a plain models.BooleanField; Meta.fields
    # auto-generates a correct BooleanFilter for it (FILTER_FOR_DBFIELD_DEFAULTS
    # maps models.BooleanField -> BooleanFilter directly), so no explicit
    # declaration is required for that one.

    class Meta:
        model = CampaignRun
        fields = ['run_status', 'open_to_collaboration']
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sortable/paginated HTML table | Manual `request.GET.get('sort')` + slicing logic | `django-tables2` `Table` + `RequestConfig` (already wired by `SingleTableMixin`) | Already installed, handles sort-direction toggling, pagination, and column-accessor resolution (including `get_FOO_display()` for choice fields on model rows) for free |
| Multi-select GET-param filtering | Manual `request.GET.getlist('run_status')` + `Q()` building | `django_filters.MultipleChoiceFilter` | Already installed; handles form rendering, validation, and OR-combination automatically per D-12 |
| Approval-status color coding | New from-scratch hashing/palette system | Reuse `calendar_display_extras.py`'s pattern *shape* (fixed internal constant, never echoing raw DB value into markup) but with a **plain 3-entry dict** (`pending_review`/`approved`/`rejected` → Bootstrap badge class), not the hash-based `PROPOSAL_PALETTE` approach — that hashing exists specifically to support an *unbounded* set of proposal codes, which does not apply to a fixed 3-value TextChoices field |

**Key insight:** Both stack libraries needed for this phase are already installed and configured
in `INSTALLED_APPS`; the entire implementation surface is composition (which mixins, in which
MRO order) and PII-safe queryset construction, not new tooling.

## Common Pitfalls

### Pitfall 1: `Table.exclude`/`Meta.exclude` alone does not satisfy D-13/VIEW-03
**What goes wrong:** A column excluded from a django-tables2 `Table` (via constructor `exclude=`
or `Meta.exclude`) is hidden from the *rendered HTML*, but the underlying `CampaignRun` objects
still flow into `context['object_list']` / `context['campaignrun_list']` (Django's
`MultipleObjectMixin.get_context_data` populates these from `self.object_list` regardless of what
the Table does with them), and the `Table` object itself (`context['table']`) still wraps the
full model instances with `contact_person`/`contact_email` accessible via `row.record`. A test
that checks `response.context['table'].rows[0].record.contact_email` — or even just an
overly-strict interpretation of "never reach the template context" — would still see the value.
**Why it happens:** django-tables2's `exclude` only affects `BoundColumn` iteration used for
rendering; it does not touch the queryset or the surrounding `ListView` context machinery.
**How to avoid:** Override `get_queryset()` to return a `.values(*allowed_fields)` queryset (no
`contact_person`/`contact_email` in the field list) whenever `not request.user.is_staff`. This
means the SQL `SELECT` itself never fetches those two columns for anonymous/non-staff requests,
so the data genuinely never exists in the process for that request — a stronger and more literally
correct implementation of "excluded from view context" than a template- or table-level hide. Keep
the `Table`'s `exclude=` kwarg as a second, redundant layer (defense in depth, satisfying D-13's
explicit request for that framing) but do not rely on it alone.
**Warning signs:** A VIEW-03 anonymous-client test that inspects `response.context` (not just
`response.content`) for the two field names and finds them present on a `CampaignRun` model
instance, even though the rendered `<table>` markup itself has no matching `<td>`.

### Pitfall 2: `.values()` rows silently break automatic choice-field display
**What goes wrong:** django-tables2's row rendering (`django_tables2/rows.py:_get_and_render_with`)
auto-calls `record.get_run_status_display()` / `record.get_approval_status_display()` **only when**
the resolved "penultimate" object in the accessor chain is a real Django model instance
(`isinstance(penultimate, models.Model)`). A `.values()` queryset yields plain `dict` rows, so this
check is always `False` — `run_status`/`approval_status` would silently render as the raw stored
string (`'pending_review'`) instead of the human label (`'Pending Review'`) for anonymous
requests, while staff (full model-instance queryset) see the correct label — an inconsistent,
easy-to-miss bug.
**Why it happens:** The dict-vs-model-instance distinction introduced by Pitfall 1's fix is
invisible at the `Table` class-definition level; it only manifests at render time, per-request,
depending on `is_staff`.
**How to avoid:** Define explicit `render_run_status`/`render_approval_status` methods on
`CampaignRunTable` that look up the display label from `CampaignRun.RunStatus(value).label` /
`CampaignRun.ApprovalStatus(value).label` directly (works identically whether `record` is a dict
or a model instance, since `value` is already the resolved raw string in both cases) rather than
depending on the automatic `get_FOO_display()` behavior. This also gives D-08's badge column a
natural home (the same `render_approval_status` method can return `format_html` with a Bootstrap
badge class).
**Warning signs:** Manual QA sees "Pending Review" in the staff view and "pending_review" in the
anonymous view for the same row.

### Pitfall 3: `TargetList` has no notion of "is a campaign" — must derive it
**What goes wrong:** Querying `TargetList.objects.all()` for D-03's campaign list page would
include every `TargetList` in the TOM (target-selection groupings, unrelated saved searches,
etc.), not just ones used as campaigns.
**Why it happens:** `TargetList` is a generic TOM Toolkit model with no FOMO-specific "campaign"
flag (confirmed in CONTEXT.md's canonical refs) — "campaign" is purely operational: a
`TargetList` with `campaign_runs__isnull=False`.
**How to avoid:** Always filter/annotate through the `campaign_runs` reverse relation:
`TargetList.objects.filter(campaign_runs__isnull=False).distinct()` (list page) and
`TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()` (target-detail
integration, D-01). Consider `.annotate(run_count=Count('campaign_runs'))` for a useful list-page
column, but this is not required by any locked decision.
**Warning signs:** The campaigns list page shows `TargetList` rows with zero associated runs.

### Pitfall 4: MRO order for `SingleTableMixin` + `FilterView`
**What goes wrong:** Declaring the mixin order backwards (`FilterView, SingleTableMixin`) can
cause `get_context_data`/`get_queryset` resolution to skip the filtering step before the table is
built, or raise `ImproperlyConfigured` if `SingleTableMixin.get_table_data` runs before
`self.object_list` exists.
**Why it happens:** Both mixins define `get_context_data`; Python MRO resolves left-to-right, so
`class CampaignRunTableView(SingleTableMixin, FilterView)` calls `SingleTableMixin`'s
`get_context_data` first, which internally calls `super().get_context_data()` (reaching
`FilterView`'s chain, which already ran `get()` and set `self.object_list = self.filterset.qs`)
before building the table — this is the *correct* order and is the pattern shown in official
docs and confirmed by reading the installed `django_tables2/views.py` and
`django_filters/views.py` source directly.
**How to avoid:** Always declare `SingleTableMixin` before `FilterView` in the class bases, exactly
as shown in Architecture Patterns §1.
**Warning signs:** Table renders unfiltered data, or `AttributeError: 'CampaignRunTableView'
object has no attribute 'object_list'`.

## Code Examples

Verified patterns from official sources / installed package source (all reproduced above under
Architecture Patterns, cross-referenced here for the planner's convenience):

### SingleTableMixin + FilterView composition
See Architecture Patterns §1. [CITED: https://django-tables2.readthedocs.io/en/latest/pages/filtering.html]

### `target_detail_buttons()` multi-link extension
See Architecture Patterns §2. [VERIFIED: read directly from installed `tom_targets` 3.0.0a9 source
and this repo's existing `module_buttons.html`/`solsys_code_extras.py`]

### `nav_items()` hook
See Architecture Patterns §3. [VERIFIED: read directly from installed `tom_common` 3.0.0a9 source]

### Multi-select `run_status` filter
See Architecture Patterns §4. [VERIFIED: read directly from installed `django_filters` 24.3 source]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| N/A — first consumer in this codebase | `django-tables2` 3.0.0 + `django-filter` 24.3 composed via `SingleTableMixin` + `FilterView` | N/A | Sets the pattern precedent for any future FOMO list/table view (e.g. a future `ObservationRecord`-centric table per STATE.md's SEED-002) |

**Deprecated/outdated:** None relevant — both libraries' current major versions are already
pinned transitively and installed; no migration concerns.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `.values(*allowed_fields)` restricted queryset is an acceptable/expected implementation shape for D-13's "excluded from view context" wording (vs. a lighter-weight `exclude=`-only + context-key-scrub approach) | Common Pitfalls §1, Architecture Patterns §1 | If the operator intended a lighter-weight interpretation, the planner may over-engineer the PII gating; low risk since the stronger approach still satisfies the weaker interpretation, but it does add the dict-vs-model-instance complexity of Pitfall 2 |
| A2 | Bootstrap4 badge classes (`badge badge-warning`/`badge-success`/`badge-danger`) are the right visual mechanism for D-08, rather than replicating `calendar_display_extras.py`'s hex/hash palette approach | Don't Hand-Roll | Low risk — this is a UI-SPEC-level decision explicitly deferred to `/gsd-ui-phase` per the phase's "UI hint: yes"; functional behavior (a visually distinct badge exists) is unaffected either way |

**If this table is empty:** N/A — see entries above; both are UI-shape assumptions with low
functional risk, not compliance/security/performance assumptions.

## Open Questions

1. **Exact D-09 column list for `.values()` field enumeration**
   - What we know: D-09 lists the spreadsheet-parity columns explicitly (telescope_instrument,
     site, obs_date, ut_start, ut_end, filters_bandpass, run_status, open_to_collaboration,
     observation_details, weather, observation_outcome, publication_plans, comments,
     approval_status for the badge) plus `pk`/`campaign` for URLs/joins.
   - What's unclear: `site` is a FK to `Observatory` — `.values('site__name')` (or similar) is
     needed rather than `.values('site')` (which would return the raw FK id) if the table should
     display the site's human-readable name; the exact `Observatory` display field to use wasn't
     specified in Phase 14's context.
   - Recommendation: Planner should check `Observatory.__str__`/relevant display field
     (`solsys_code/solsys_code_observatory/models.py`) and use `.values('site__<field>')`
     explicitly, or `.select_related('site')` for the staff (full-instance) path and a
     `.values()` equivalent for the non-staff path.

2. **Whether the campaigns list page (D-03) needs its own filter/sort or is a plain unsorted list**
   - What we know: D-03 only specifies "a view listing every `TargetList` that has ≥1
     `CampaignRun`, each linking to its per-campaign table" — no sort/filter/pagination
     requirement was attached to this page (VIEW-01/04 apply to the per-campaign table, not the
     list page).
   - What's unclear: Whether a plain Django `ListView` (no django-tables2) suffices, or whether
     UI-SPEC will want the same table treatment for consistency.
   - Recommendation: Default to plain `ListView` for the campaigns list page (Recommended Project
     Structure above) since no locked decision requires table machinery there; defer to
     `/gsd-ui-phase` if a richer treatment is wanted.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| django-tables2 | VIEW-01 sortable/paginated table | Yes | 3.0.0 | — |
| django-filter | VIEW-04 filtering | Yes | 24.3 | — |
| Bootstrap4 (`bootstrap4` app) | Badge styling (D-08), filter form widgets | Yes | already in `INSTALLED_APPS` | — |
| SQLite (dev DB) | All queryset operations above | Yes | project default (`src/fomo_db.sqlite3`) | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — this phase has no external service/network
dependency; all required libraries are already installed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`), per CLAUDE.md's "DB-dependent tests go in `solsys_code/tests/`" convention |
| Config file | none — Django test discovery via `./manage.py test solsys_code` |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_views` (new test module) |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VIEW-01 | Table lists all runs for a campaign, sortable/paginated (25/page, D-11) | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunTableView.test_lists_all_runs_paginated` | ❌ Wave 0 |
| VIEW-02 | Target-detail page shows one link per matching campaign; navbar shows "Campaigns" entry | integration (Django `Client`, response content assertions) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignDetailIntegration` | ❌ Wave 0 |
| VIEW-03 | Anonymous client never sees `contact_person`/`contact_email` (context AND content); staff client does | integration (Django `Client()` unauthenticated vs. `Client()` with `is_staff=True` user) | `./manage.py test solsys_code.tests.test_campaign_views.TestContactFieldGating` | ❌ Wave 0 |
| VIEW-04 | `run_status` multi-select filter narrows rows (OR semantics); `open_to_collaboration` filter narrows rows; default (no filter) shows everything (D-07) | integration (Django `Client`, GET with query params) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunFilterSet` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_campaign_views`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full `./manage.py test solsys_code` green before `/gsd-verify-work`, plus
  `ruff check .` / `ruff format --check .` clean per CLAUDE.md.

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_campaign_views.py` — new test module covering VIEW-01..04 (does not
  exist yet; no prior view tests for `CampaignRun` exist — `test_campaign_models.py` from Phase
  14 only covers the model, not views)
- [ ] Fixture data: reuse `NonSiderealTargetFactory` (CLAUDE.md convention) + `TargetList` +
  `CampaignRun.objects.create(...)` rows spanning multiple `run_status`/`approval_status` values
  and at least one row with `contact_person`/`contact_email` populated, following the pattern
  already established in `solsys_code/tests/test_campaign_models.py`
- [ ] A `User(is_staff=True)` test fixture/helper for the staff-vs-anonymous `Client` split in
  `TestContactFieldGating` — no existing precedent for staff-user test fixtures in this codebase
  (grep for `is_staff` across `solsys_code/`/`src/` returned no hits), so this is a genuinely new
  fixture pattern for the test suite.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No new authentication surface — reuses Django's existing session auth and `request.user.is_staff` |
| V3 Session Management | No | No change to session handling |
| V4 Access Control | Yes | View-layer field-level access control: `contact_person`/`contact_email` gated by `request.user.is_staff` via restricted `.values()` queryset (Pitfall 1), not client-side/template-only hiding |
| V5 Input Validation | Yes | `django-filter`'s `FilterSet` validates/coerces GET params (`run_status` choices, `open_to_collaboration` boolean) before building queryset filters — never interpolates raw GET params into raw SQL |
| V6 Cryptography | No | No cryptographic operations in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| PII disclosure via server-rendered context leaking beyond intended audience (contact fields visible to anonymous users) | Information Disclosure | View-layer queryset restriction (`.values()` excluding contact fields for non-staff) rather than template-only `{% if user.is_staff %}` guards — directly matches D-13's explicit defense-in-depth requirement and VIEW-03's ASVS V4 mapping above |
| Un-validated multi-value GET params driving DB filters (e.g. `?run_status=foo&run_status=bar`) | Tampering | `django_filters.MultipleChoiceFilter(choices=CampaignRun.RunStatus.choices)` rejects/ignores values outside the declared `TextChoices` — confirmed via `ChoiceField`-backed `field_class` in the installed `django_filters` source; no manual GET-param parsing needed |
| Open redirect / arbitrary URL construction via campaign `pk` in table URL | Tampering | Standard Django `path('<int:pk>/', ...)` URL converter already constrains the parameter to an integer; `get_object_or_404`-style lookup on an invalid/missing `pk` returns 404, not an error leaking internals |

## Sources

### Primary (HIGH confidence)
- Installed `django_tables2` 3.0.0 source (`django_tables2/views.py`, `django_tables2/rows.py`) — read directly in this session to confirm `SingleTableMixin`/`get_table_kwargs`/choice-display behavior
- Installed `django_filters` 24.3 source (`django_filters/views.py`, `django_filters/filterset.py`, `django_filters/filters.py`) — read directly to confirm `FilterView`, `FILTER_FOR_DBFIELD_DEFAULTS`, and `MultipleChoiceFilter` OR-semantics
- Installed `tom_common` 3.0.0a9 source (`tom_common/templatetags/tom_common_extras.py`) — read directly to confirm `nav_items()`/`navbar_app_addons` mechanism
- Installed `tom_targets` 3.0.0a9 source (`tom_targets/templatetags/targets_extras.py`) — read directly to confirm `target_detail_buttons()`/`get_buttons`/`target_buttons` context-passing mechanism
- This repo's `solsys_code/models.py`, `solsys_code/apps.py`, `solsys_code/templatetags/calendar_display_extras.py`, `src/templatetags/solsys_code_extras.py`, `src/templates/tom_targets/partials/module_buttons.html`, `src/fomo/settings.py`, `src/fomo/urls.py`, `solsys_code/calendar_urls.py` — read directly this session

### Secondary (MEDIUM confidence)
- [django-tables2 filtering docs](https://django-tables2.readthedocs.io/en/latest/pages/filtering.html) — WebSearch-confirmed, matches the installed-source behavior directly inspected above
- [django-tables2 custom rendering / render_FOO docs](https://django-tables2.readthedocs.io/en/latest/pages/custom-rendering.html) — WebSearch-confirmed `render_<column>` method signature convention

### Tertiary (LOW confidence)
- None — all significant claims in this document were verified against installed package source or this repo's own code.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — both libraries already installed/pinned; versions confirmed via `pip show` in the project venv
- Architecture: HIGH — all hook mechanisms (`target_detail_buttons`, `nav_items`, `SingleTableMixin`+`FilterView` MRO) verified by reading the actual installed package source, not just documentation
- Pitfalls: HIGH — the PII-gating pitfall (Pitfall 1) and choice-display pitfall (Pitfall 2) were derived from direct inspection of `django_tables2/rows.py` and Django's `MultipleObjectMixin` context behavior, not assumption

**Research date:** 2026-07-03
**Valid until:** 2026-08-02 (30 days — stable, already-pinned dependency versions; re-verify if `tomtoolkit` is upgraded past `3.0.0a9`, since that could bump `django-tables2`/`django-filter` transitively)
