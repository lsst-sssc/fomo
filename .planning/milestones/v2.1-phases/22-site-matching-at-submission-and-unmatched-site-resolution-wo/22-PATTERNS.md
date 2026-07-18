# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow - Pattern Map

**Mapped:** 2026-07-14
**Files analyzed:** 9 (3 modified-heavily, 4 modified-lightly, 2 new)
**Analogs found:** 9 / 9 (all analogs are in-repo — this phase extends existing modules; RESEARCH.md's Code Examples are already extracted directly from these same files, so "analog" and "target file" frequently coincide)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_utils.py` (+`substring_or_fuzzy_match_candidates()`, +throttle helper) | utility | transform | `fuzzy_match_candidates()` in the same file (lines 306-328) | exact |
| `solsys_code/campaign_views.py` (+`SiteSearchView`) | controller (View) | request-response | `CampaignGapAnalysisView` (GET-only, public, `View`/`TemplateView`, lines 509-583) and `request.htmx` precedent in `solsys_code/views.py:156` | role-match |
| `solsys_code/campaign_views.py` (`CampaignRunDecisionView.post()` +`resolve_site` action, +`_project_calendar_event()`) | controller (View) | CRUD + event-driven | `CampaignRunDecisionView.post()`'s existing `approve` branch (lines 344-477) | exact (self-extension) |
| `solsys_code/campaign_views.py` (`ApprovalQueueView.get_context_data()` + third table) | controller (View) | request-response | `ApprovalQueueView.get_context_data()`'s existing `pending_table`/`decided_table` construction (lines 286-329) | exact (self-extension) |
| `solsys_code/campaign_tables.py` (`ApprovalQueueTable.render_site()` HTMX widget swap) | component (table renderer) | transform | `ApprovalQueueTable.render_site()` itself (lines 208-246) | exact (self-extension) |
| `solsys_code/campaign_forms.py` (`CampaignRunSubmissionForm.site_raw` widget) | component (form field) | transform | `CampaignRunSubmissionForm.__init__`'s existing crispy `Layout` (lines 101-119) | exact (self-extension) |
| `solsys_code/campaign_urls.py` (+`site-search/` path) | route | request-response | Existing flat `urlpatterns` list (lines 20-32) | exact |
| `src/templates/campaigns/partials/site_search_results.html` (NEW) | component (partial template) | transform | `src/templates/solsys_code/partials/campaign_links.html` (dir convention) + `ApprovalQueueTable`'s `format_html_join` escaping convention | role-match (new file, established dir/escaping pattern) |
| `src/templates/campaigns/approval_queue.html` (+third `{% render_table %}` block) | component (template) | transform | The file's own existing two-table structure (lines 1-13) | exact (self-extension) |
| `solsys_code/tests/test_campaign_site_search.py` (NEW) | test | request-response | `solsys_code/tests/test_campaign_approval.py` (`CampaignApprovalTestBase`, `BULK_MPC_FIXTURE`) | role-match |
| `solsys_code/tests/test_campaign_approval.py` (+`TestSitesNeedingReview`) | test | CRUD + event-driven | Existing approve/reject test classes in the same file | exact (self-extension) |

## Pattern Assignments

### `solsys_code/campaign_utils.py` — `substring_or_fuzzy_match_candidates()` (D-04)

**Analog:** `fuzzy_match_candidates()`, same file, lines 306-328.

**Imports already present** (lines 11-25) — no new imports needed for the matcher itself:
```python
import difflib
import logging
...
from django.core.cache import cache
```

**Docstring + "never raise" discipline pattern** (copy structure from `fuzzy_match_candidates`, lines 306-328):
```python
def fuzzy_match_candidates(site_raw: str, candidate_pool: dict[str, str]) -> list[tuple[str, str]]:
    """Fuzzy-match a raw submitted site string against a candidate pool (D-01/A3).
    ...
    """
    text = (site_raw or '').strip()
    if not text:
        return []
    matches = difflib.get_close_matches(text, candidate_pool.keys(), n=5, cutoff=0.6)
    return [(match, candidate_pool[match]) for match in matches]
```

**Core pattern to write** (RESEARCH.md Pattern 2, already fully drafted — copy near-verbatim):
```python
def substring_or_fuzzy_match_candidates(
    site_raw: str, candidate_pool: dict[str, str], *, limit: int = 8
) -> list[tuple[str, str]]:
    """Substring-first, difflib-fallback site match (D-04)."""
    text = (site_raw or '').strip()
    if not text:
        return []
    needle = text.lower()
    hits = [(candidate, obscode) for candidate, obscode in candidate_pool.items() if needle in candidate.lower()]
    if hits:
        hits.sort(key=lambda pair: (len(pair[0]), pair[0]))
        return hits[:limit]
    return fuzzy_match_candidates(text, candidate_pool)[:limit]
```
Place directly below `fuzzy_match_candidates()` (after line 328), not replacing it — `ApprovalQueueTable.render_site()`'s pre-Phase-22 call site depends on it staying intact until the widget swap lands in the same phase.

**Throttle helper** — new module-level constants + function, same "never raise, degrade gracefully" style as `build_site_candidates()` (lines 258-303):
```python
SITE_SEARCH_THROTTLE_LIMIT = 40
SITE_SEARCH_THROTTLE_WINDOW_SECONDS = 60


def _check_and_increment_throttle(client_ip: str) -> bool:
    key = f'site_search_throttle:{client_ip}'
    added = cache.add(key, 1, timeout=SITE_SEARCH_THROTTLE_WINDOW_SECONDS)
    if added:
        return True
    try:
        count = cache.incr(key)
    except ValueError:
        cache.set(key, 1, timeout=SITE_SEARCH_THROTTLE_WINDOW_SECONDS)
        return True
    return count <= SITE_SEARCH_THROTTLE_LIMIT
```
Uses the same `django.core.cache.cache` import already at line 19 — no new import.

---

### `solsys_code/campaign_views.py` — `SiteSearchView` (D-01/D-02/D-03)

**Analog:** `CampaignGapAnalysisView` (public, no `StaffRequiredMixin`, GET-only pattern, lines 509-583) for "public View, no auth mixin"; `solsys_code/views.py:156` for `request.htmx` usage precedent (read separately if needed — not in this phase's files list, cited by RESEARCH.md only).

**Imports pattern** (extend existing block, lines 14-43):
```python
from django.http import HttpResponseBadRequest   # already imported, line 24 — add HttpResponse if not already covered
from django.shortcuts import get_object_or_404, redirect, render   # add `render`
from django.views.generic import FormView, ListView, TemplateView, View   # View already imported, line 27

from .campaign_utils import build_site_candidates, resolve_site, substring_or_fuzzy_match_candidates  # extend line 40
```
`_check_and_increment_throttle` also needs importing from `campaign_utils` alongside the above.

**No-auth-mixin pattern** — mirror `CampaignGapAnalysisView`'s class docstring rationale (lines 509-520): "Public/read-only, same posture as `CampaignRunTableView` — no `StaffRequiredMixin`." Reuse this exact phrasing convention to document why `SiteSearchView` also skips `StaffRequiredMixin` (D-01).

**Core GET-only View pattern** (RESEARCH.md Pattern 1, ready to adapt):
```python
class SiteSearchView(View):
    http_method_names = ['get']

    def get(self, request):
        client_ip = request.META.get('REMOTE_ADDR', '')
        if not _check_and_increment_throttle(client_ip):
            return HttpResponse(status=429)
        query = request.GET.get('q', '')
        input_id = request.GET.get('input_id', 'id_site_raw')
        candidates = substring_or_fuzzy_match_candidates(query, build_site_candidates())
        return render(
            request,
            'campaigns/partials/site_search_results.html',
            {'candidates': candidates, 'input_id': input_id, 'query': query},
        )
```
Note `http_method_names = ['post']` is the existing convention for POST-only views (`CampaignRunDecisionView`, line 342) — mirror that exact attribute name/shape for the GET-only case (`['get']`).

---

### `solsys_code/campaign_views.py` — `CampaignRunDecisionView.post()` `resolve_site` action + `_project_calendar_event()` (D-08)

**Analog:** the view's own existing `approve` branch, lines 344-477 (this is a same-file extraction/refactor, not a cross-file pattern borrow).

**Extraction target — verbatim block to lift into a module-level helper** (lines 391-449, unchanged logic per RESEARCH.md Pattern 4):
```python
def _project_calendar_event(run: CampaignRun) -> None:
    """CAL-01/CAL-02 CalendarEvent projection (D-08). May raise -- callers own error handling."""
    if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
        return
    event_fields = {
        'title': f'{run.campaign.name}: {run.telescope_instrument}',
        'description': run.observation_details,
        'target_list': run.campaign,
        'telescope': run.telescope_instrument,
    }
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        event_fields['start_time'] = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
        event_fields['end_time'] = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
        insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
        return
    try:
        sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
    except ValueError:
        logger.debug('sun_event(sun) raised for site=%s date=%s; skipping projection.', run.site, run.window_start)
        return
    event_fields['start_time'] = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    event_fields['end_time'] = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    insert_or_create_calendar_event({'url': f'CAMPAIGN:{run.pk}'}, fields=event_fields)
```
The `approve` branch (lines 354-466) then calls `_project_calendar_event(run)` in place of its inlined block, **inside its unchanged `except Exception:` revert-to-PENDING_REVIEW block** (lines 450-465) — do not touch that except-block's structure or messaging.

**CR-01 display-string → obscode mapping pattern to reuse verbatim for `resolve_site` action** (lines 377-389):
```python
selection = request.POST.get('site_selection', '').strip() or run.site_raw
obscode_selection = build_site_candidates().get(selection, selection)
site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
run.site, run.site_needs_review = site, needs_review
run.save(update_fields=['site', 'site_needs_review'])
```
The D-06 never-re-resolve guard to copy exactly (comment + `if run.site is None:` check, lines 357-366) — re-fetch `run` fresh from the DB in the new branch (`CampaignRun.objects.get(pk=pk)`, same as line 356) before this check, per Pitfall 2.

**Action dispatch pattern** (extend, lines 346-349):
```python
action = request.POST.get('action')
if action not in ('approve', 'reject', 'resolve_site'):
    return HttpResponseBadRequest()
```
The `resolve_site` branch needs its **own, non-reverting** `try/except` — do NOT reuse the `approve` branch's `except Exception: ... update(approval_status=PENDING_REVIEW) ...` block (lines 450-465). New pattern:
```python
elif action == 'resolve_site':
    run = get_object_or_404(CampaignRun, pk=pk)
    if run.approval_status != CampaignRun.ApprovalStatus.APPROVED or not run.site_needs_review:
        messages.warning(request, 'This run is not awaiting site resolution.')
        return redirect('campaigns:approval_queue')
    if run.site is None:
        selection = request.POST.get('site_selection', '').strip() or run.site_raw
        obscode_selection = build_site_candidates().get(selection, selection)
        site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
        run.site, run.site_needs_review = site, needs_review
        run.save(update_fields=['site', 'site_needs_review'])
        try:
            _project_calendar_event(run)
        except Exception:
            logger.exception('Calendar projection failed for CampaignRun %s during resolve_site.', pk)
            messages.error(request, 'Site resolved but calendar event creation failed. Please try again.')
            return redirect('campaigns:approval_queue')
    messages.success(request, 'Site resolved.')
    return redirect('campaigns:approval_queue')
```
This mirrors the "business-logic bypass" mitigation RESEARCH.md's Security Domain calls out — validate `approval_status`/`site_needs_review` state server-side, not just trust the button was only offered on eligible rows (mirrors `updated_count == 1` staleness-check discipline already used for approve/reject, lines 350-352, 467-476).

---

### `solsys_code/campaign_views.py` — `ApprovalQueueView.get_context_data()` third table (D-07)

**Analog:** the view's own existing `pending_table`/`decided_table` construction, lines 286-329.

**Queryset + table-construction pattern to copy for the third table**:
```python
review_qs = (
    CampaignRun.objects.filter(
        approval_status=CampaignRun.ApprovalStatus.APPROVED, site_needs_review=True
    )
    .select_related('campaign', 'site')
    .order_by('-pk')[:20]
)
...
review_table = ApprovalQueueTable(
    list(review_qs),
    prefix='review-',
    request=self.request,
    candidate_pool=candidate_pool,   # reuse the SAME candidate_pool variable, line 310 — do not re-call build_site_candidates()
    empty_text='No sites currently need review.',
)
RequestConfig(self.request).configure(review_table)
context['review_table'] = review_table
```
This mirrors the `decided_qs`/`pending_qs` docstring precedent about `order_by('-pk')[:20]`, `list(...)` materialization to avoid the "Cannot reorder a query once a slice has been taken" error (lines 292-300), and the Pitfall-5 discipline of building `candidate_pool` exactly once per request (line 310, comment at 306-309) — the third table must consume that same variable, never call `build_site_candidates()` again.

`ApprovalQueueTable.__init__` already accepts `candidate_pool=` (line 202-206) — no table-class change needed for D-07 itself, only for D-10's widget swap (below).

---

### `solsys_code/campaign_tables.py` — `ApprovalQueueTable.render_site()` HTMX widget swap (D-10)

**Analog:** the method's own current static-datalist implementation, lines 208-246.

**Current pattern being replaced** (keep the resolved-row/`show_actions=False` early-return unchanged, lines 219-221):
```python
def render_site(self, record):
    site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
    if site_short_name or not self.show_actions:
        return super().render_site(record)
    pk = Accessor('pk').resolve(record, quiet=True)
    site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
    ...
```
**Escaping convention to preserve** (`format_html`/`format_html_join`, lines 226-246) — every candidate/site string interpolated into the new HTMX markup must go through the same auto-escaping mechanism, e.g.:
```python
options = format_html_join('', '<option value="{}">', ((candidate,) for candidate, _obscode in candidate_pairs))
...
return format_html(
    '<input type="text" name="site_selection" value="{0}" id="site-selection-{5}" '
    'hx-get="{6}" hx-trigger="keyup changed delay:300ms[this.value.length >= 2]" '
    'hx-target="#site-suggestions-{5}" hx-swap="innerHTML" hx-vals=\'{{"input_id": "site-selection-{5}"}}\' '
    'form="{2}" class="form-control form-control-sm" placeholder="MPC code or site name…" autocomplete="off">'
    '<div id="site-suggestions-{5}"></div>'
    '<a href="{4}" class="small ml-1">Create new Observatory</a>',
    site_raw, datalist_id, form_id, options, create_url, pk, reverse('campaigns:site_search'),
)
```
D-10 explicitly keeps the "Create new Observatory" link (`create_url`, lines 232-235) — do not drop it. The `resolve_site` mode of this table (D-07's third table) needs a parallel `render_site`/`render_actions` treatment — the docstring at lines 214-217 explaining "only overridden here, not on the parent" is the precedent to extend/re-document for the new resolve-only rendering branch (distinguish `show_actions` pending-row markup from a `resolve`-row markup via a new constructor flag, e.g. `mode='pending'|'resolve'`, mirroring the existing `show_actions` boolean-flag convention at line 202-206).

**`render_actions()` CSRF/form-id pattern to extend for the resolve action** (lines 248-269):
```python
def render_actions(self, record):
    if not self.show_actions:
        return ''
    decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
    csrf_token = get_token(self.request) if self.request is not None else ''
    form_id = f'decide-form-{record.pk}'
    ...
```
For the review table's "Resolve" button, mirror this exact `<form>`+hidden-CSRF-input+named-submit-button shape with `name="action" value="resolve_site"` in place of `approve`/`reject`.

---

### `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm.site_raw` widget (D-09)

**Analog:** the form's own `__init__`/`Layout` construction, lines 101-119; `reverse_lazy` precedent at `campaign_views.py:197` (`success_url = reverse_lazy(...)`) — Pitfall 6 requires `reverse_lazy`, not `reverse`, at class-body definition time.

**Imports to add** (extend lines 9-16):
```python
from django.urls import reverse_lazy
```

**Widget pattern** (RESEARCH.md Code Examples, ready to adapt into the existing field declaration at line 24):
```python
site_raw = forms.CharField(
    max_length=255,
    required=False,
    label='Observing site',
    widget=forms.TextInput(attrs={
        'hx-get': reverse_lazy('campaigns:site_search'),
        'hx-trigger': "keyup changed delay:300ms[this.value.length >= 2]",
        'hx-target': '#site-suggestions-id_site_raw',
        'hx-swap': 'innerHTML',
        'hx-vals': '{"input_id": "id_site_raw"}',
        'autocomplete': 'off',
    }),
)
```
D-09: **no** "Create new Observatory" link/`HTML(...)` layout addition here — only the suggestions container:
```python
self.helper.layout = Layout(
    'campaign',
    Fieldset(
        'Run details',
        'telescope_instrument',
        'site_raw',
        HTML('<div id="site-suggestions-id_site_raw"></div>'),
        'obs_date',
        ...
    ),
    ...
)
```
`HTML` needs importing from `crispy_forms.layout` — extend the existing import at line 11 (`from crispy_forms.layout import Div, Fieldset, HTML, Layout, Submit`).

---

### `solsys_code/campaign_urls.py` — `site-search/` route

**Analog:** the file's own flat `urlpatterns` list, lines 20-32.

**Pattern to copy exactly** (existing GET-view route shape, `approval-queue/` at line 28):
```python
from solsys_code.campaign_views import (
    ApprovalQueueView,
    CampaignGapAnalysisView,
    CampaignListView,
    CampaignRunDecisionView,
    CampaignRunSubmissionView,
    CampaignRunTableView,
    SiteSearchView,   # add
)
...
urlpatterns = [
    ...
    path('site-search/', SiteSearchView.as_view(), name='site_search'),
    ...
]
```
Add it near `approval-queue/` (both are GET, both new-ish surfaces) but before the catch-all `path('<int:pk>/', ...)` at line 31 — Django resolves top-to-bottom, and `<int:pk>/` would otherwise never conflict with `site-search/` since it's not purely numeric, but keep the convention of static paths before dynamic ones already followed by the existing list (lines 21-31).

---

### `src/templates/campaigns/partials/site_search_results.html` (NEW)

**Analog (directory convention):** `src/templates/solsys_code/partials/` (existing `partials/` subdirectory pattern — `campaign_links.html`, `ephem_button.html`, `campaigns_nav_link.html`); no `campaigns/partials/` directory exists yet, create it.

**Analog (escaping convention):** `ApprovalQueueTable`'s `format_html`/`format_html_join` discipline in `campaign_tables.py` (T-21-01 precedent, lines 226-246) — Django template auto-escaping is the template-side equivalent; RESEARCH.md's Code Examples section has the full ready-to-use template:
```html
{% if candidates %}
<ul class="list-group" id="site-suggestions-list">
  {% for display, obscode in candidates %}
  <li class="list-group-item list-group-item-action"
      style="cursor:pointer;"
      onclick="document.getElementById('{{ input_id }}').value = '{{ display|escapejs }} ({{ obscode|escapejs }})'; document.getElementById('site-suggestions-{{ input_id }}').innerHTML = '';">
    {{ display }} ({{ obscode }})
  </li>
  {% endfor %}
</ul>
{% elif query %}
<p class="text-muted small mb-0">No matches — free text is fine, a staff member will resolve it.</p>
{% endif %}
```
Note the `|escapejs` filter requirement inside the `onclick=` JS-string context — distinct from the auto-escaped `{{ display }} ({{ obscode }})` text node (see RESEARCH.md Security Domain, XSS row).

---

### `src/templates/campaigns/approval_queue.html` — third table block (D-07)

**Analog:** the file's own existing two-`{% render_table %}` structure (full file, 14 lines).

**Pattern to copy exactly**:
```html
{% extends 'tom_common/base.html' %}
{% load render_table from django_tables2 %}
{% block title %}Approval Queue{% endblock %}

{% block content %}
<h4 class="font-weight-bold mb-4">Approval Queue</h4>

<h5 class="font-weight-bold mb-3">Pending Review</h5>
{% render_table pending_table %}

<h5 class="font-weight-bold mb-3 mt-4">Recently Decided</h5>
{% render_table decided_table %}

<h5 class="font-weight-bold mb-3 mt-4">Sites Needing Review</h5>
{% render_table review_table %}
{% endblock %}
```
No new `{% load %}` needed — `render_table` is already loaded at line 2.

---

### `solsys_code/tests/test_campaign_site_search.py` (NEW)

**Analog:** `solsys_code/tests/test_campaign_approval.py` — `CampaignApprovalTestBase` fixture (`campaign`, `staff_user`, `non_staff_user`, `_make_pending_run()`) and `BULK_MPC_FIXTURE` are directly reusable per RESEARCH.md's Wave 0 Gaps section. Not read in full this pass (file exceeds phase scope for pattern excerpting — RESEARCH.md's Validation Architecture section already documents its reusable fixtures by name); read this file directly during planning/execution for exact fixture signatures.

### `solsys_code/tests/test_campaign_approval.py` (+`TestSitesNeedingReview`)

**Analog:** the file's own existing approve/reject test classes (same file) — mirror their `CampaignApprovalTestBase` subclassing and `client.post(reverse('campaigns:decide', kwargs={'pk': run.pk}), {'action': 'approve', ...})` request-shape convention for the new `resolve_site` action tests.

## Shared Patterns

### "Never raise for expected messy data" discipline
**Source:** `solsys_code/campaign_utils.py` — `resolve_site()` docstring (lines 116-141), `build_site_candidates()` docstring (lines 267-272), `fuzzy_match_candidates()` docstring (lines 319-322).
**Apply to:** `substring_or_fuzzy_match_candidates()`, the throttle helper, `SiteSearchView.get()` — every new function in this phase should document and honor "never raises" the same way.

### `format_html`/`format_html_join` auto-escaping for untrusted MPC/local candidate strings
**Source:** `solsys_code/campaign_tables.py` `ApprovalQueueTable.render_site()` (lines 226-246), referenced as the closed T-21-01 XSS mitigation.
**Apply to:** `site_search_results.html` partial (Django template auto-escaping + `|escapejs` for the `onclick=` JS-string context), and `ApprovalQueueTable.render_site()`'s HTMX-widget replacement.

### `StaffRequiredMixin` gating vs. deliberate public access
**Source:** `solsys_code/mixins.py` `StaffRequiredMixin` (full file, 10 lines); `ApprovalQueueView`/`CampaignRunDecisionView` class docstrings explaining why they use it, vs. `CampaignGapAnalysisView`/`CampaignRunSubmissionView` explaining why they deliberately don't.
**Apply to:** `SiteSearchView` must NOT use `StaffRequiredMixin` (D-01); `ApprovalQueueView`'s third table and `CampaignRunDecisionView`'s `resolve_site` action stay behind it (both already inherit it — no change needed there beyond the new branch/table).

### `reverse_lazy()` for URLs referenced at form class-body/import time
**Source:** `solsys_code/campaign_views.py:197` (`success_url = reverse_lazy('campaigns:submission_thanks')`).
**Apply to:** `CampaignRunSubmissionForm.site_raw`'s `hx-get` widget attr (Pitfall 6) — must use `reverse_lazy`, never `reverse`.

### Non-reverting vs. reverting failure-handling split
**Source:** `CampaignRunDecisionView.post()`'s existing `except Exception:` revert-to-PENDING_REVIEW block (lines 450-465), explicitly documented as approve-only behavior.
**Apply to:** the new `resolve_site` action's own `try/except` around `_project_calendar_event(run)` must NOT reuse this revert logic — report failure via `messages.error` only, leave `approval_status`/`site`/`site_needs_review` as already saved.

### `candidate_pool` computed once per request, never per row/per suggestion
**Source:** `ApprovalQueueView.get_context_data()` comment (lines 306-309, "Pitfall 5").
**Apply to:** the new third (`review_table`) table construction must reuse the same `candidate_pool` variable already built for `pending_table`; `SiteSearchView.get()` calls `build_site_candidates()` once per request (it's already 24h-cached, so this is a cache read, not a re-fetch).

## No Analog Found

None — every file this phase touches either already exists (self-extension) or has a direct role-match analog in-repo (new `SiteSearchView`/partial template/URL route, patterned on `CampaignGapAnalysisView` + the existing `campaigns/partials/`-sibling directory convention + RESEARCH.md's fully-drafted Code Examples).

## Metadata

**Analog search scope:** `solsys_code/campaign_*.py`, `solsys_code/mixins.py`, `src/templates/campaigns/`, `src/templates/*/partials/` (directory-convention scan only), `solsys_code/tests/test_campaign_approval.py` (referenced, not fully read — already documented by RESEARCH.md's Validation Architecture section).
**Files scanned:** `campaign_utils.py`, `campaign_views.py`, `campaign_tables.py`, `campaign_forms.py`, `campaign_urls.py`, `mixins.py`, `approval_queue.html`, plus a `find` of every `partials/` directory in `src/templates/`.
**Pattern extraction date:** 2026-07-14
