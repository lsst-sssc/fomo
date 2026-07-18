# Phase 21: Site Disambiguation & Submitter Contact Opt-In - Pattern Map

**Mapped:** 2026-07-11
**Files analyzed:** 8 (modified) + 1 (new migration)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_views.py` (`CampaignRunDecisionView.post`) | controller | request-response | same file, same method (self-modification) | exact |
| `solsys_code/campaign_views.py` (`CampaignRunTableView.get_queryset`, `ApprovalQueueView.get_context_data`) | controller | CRUD (queryset annotation) | same file, same methods (self-modification) | exact |
| `solsys_code/campaign_tables.py` (`render_site`, `render_actions`) | component (django-tables2 Table) | request-response (server-rendered HTML) | same file, same methods (self-modification) | exact |
| `solsys_code/campaign_utils.py` (new `build_site_candidates()`/`fuzzy_match_candidates()` helpers) | utility | transform | `resolve_site()` in the same file | role-match |
| `solsys_code/solsys_code_observatory/utils.py` (`MPCObscodeFetcher.query_all()`) | service | request-response (external HTTP API) | `MPCObscodeFetcher.query()` in the same file | exact |
| `solsys_code/solsys_code_observatory/views.py` (`CreateObservatory`) | controller (CreateView) | request-response | same file, same class (self-modification) | exact |
| `solsys_code/campaign_forms.py` (`CampaignRunSubmissionForm`) | component (Django Form) | request-response | same file, same class (self-modification); `open_to_collaboration` field precedent | exact |
| `solsys_code/models.py` (`CampaignRun`) | model | CRUD | `open_to_collaboration = models.BooleanField(...)` in the same model | exact |
| `solsys_code/migrations/000N_....py` (new) | migration | batch | `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py` | exact |
| `solsys_code/tests/test_campaign_approval.py` (new `TestSiteFuzzyMatch`, extended `TestApproval`/`TestApprovalSiteResolution`) | test | request-response | existing classes in the same file (`CampaignApprovalTestBase`, `@patch` usage) | exact |

## Pattern Assignments

### `solsys_code/campaign_views.py` — D-06 clobber-fix guard (controller, request-response)

**Analog:** the exact same method, current behavior (`CampaignRunDecisionView.post`, lines 304-413).

**Current unconditional resolve call** (lines 314-326):
```python
if updated_count == 1 and action == 'approve':
    try:
        run = CampaignRun.objects.get(pk=pk)
        # D-07: reuse the existing 3-tier site resolver rather than re-implementing it.
        site, needs_review = resolve_site(run.site_raw, create_placeholder=False)
        run.site, run.site_needs_review = site, needs_review
        run.save(update_fields=['site', 'site_needs_review'])
```

**Required change (D-06):** wrap the `resolve_site()` call in `if run.site is None:`. When `run.site`
is already set, read the site-selection field the new UI submits (`request.POST.get('site_selection', '')`)
only in the `run.site is None` branch, falling back to `run.site_raw` when blank — never call
`resolve_site()` a second time once `run.site` is truthy. Keep the existing `except Exception:` revert
block (lines 387-397) unchanged; D-06 only changes what happens *before* it, not the revert-on-failure
contract (Pitfall 3's regression is a second POST hitting this same code path, not the revert logic
itself).

**Error handling pattern to preserve** (lines 387-398):
```python
except Exception:
    logger.exception('Approve side-effects failed for CampaignRun %s; reverted to pending review.', pk)
    CampaignRun.objects.filter(pk=pk).update(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
    messages.error(request, ...)
```
Copy this exact try/except shape unmodified — the fix is purely the guard condition around the
`resolve_site()` call, not the revert path.

---

### `solsys_code/campaign_views.py` — `ApprovalQueueView.get_context_data` (controller, request-response)

**Analog:** the same method, current behavior (lines 252-289).

**Core pattern to extend** — build the merged candidate pool **once per request**, not per row
(Pitfall 5), then pass it into `ApprovalQueueTable.__init__` the same way `request=self.request`
is already passed:
```python
pending_table = ApprovalQueueTable(
    pending_qs,
    prefix='pending-',
    request=self.request,
    empty_text='No submissions waiting for review.',
)
```
Add a `candidate_pool=candidate_pool` kwarg alongside `request=self.request` here, mirroring the
existing `request=` kwarg-passing convention exactly (`ApprovalQueueTable.__init__` already stores
`self.request = request` — extend with `self.candidate_pool = candidate_pool` the same way).

---

### `solsys_code/campaign_views.py` — VIEW-05 PII gating (controller, CRUD)

**Analog:** `CampaignRunTableView.get_queryset()` (lines 87-106), the existing `ALLOWED_FIELDS_FOR_NON_STAFF`
discipline this phase must not regress.

**Imports pattern to add** (top of file, alongside existing `from django.db.models import Count, F` at line 23):
```python
from django.db.models import Case, CharField, EmailField, F, Value, When
```

**Current non-staff branch** (lines 101-106):
```python
qs = qs.exclude(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
qs = qs.order_by(F('window_start').desc(nulls_last=True))
return qs.values(*ALLOWED_FIELDS_FOR_NON_STAFF)
```

**Required change (VIEW-05/D-08, RESEARCH.md Pattern 5):** insert a `Case`/`When` annotation before
`.values()`, and select `contact_person`/`contact_email` from the annotated fields rather than the
raw columns:
```python
qs = qs.annotate(
    _public_contact_person=Case(
        When(contact_public_opt_in=True, then=F('contact_person')),
        default=Value(''), output_field=CharField(),
    ),
    _public_contact_email=Case(
        When(contact_public_opt_in=True, then=F('contact_email')),
        default=Value(''), output_field=EmailField(),
    ),
)
return qs.values(
    *[f for f in ALLOWED_FIELDS_FOR_NON_STAFF if f not in ('contact_person', 'contact_email')],
    contact_person=F('_public_contact_person'),
    contact_email=F('_public_contact_email'),
)
```
`ALLOWED_FIELDS_FOR_NON_STAFF` itself (lines 52-69) does NOT get `contact_person`/`contact_email`
added to it directly — they're supplied via the `contact_person=F(...)` kwarg form of `.values()`
instead, per the anti-pattern warning in RESEARCH.md ("Adding contact_person/contact_email
unconditionally to ALLOWED_FIELDS_FOR_NON_STAFF and gating only in a template/render method").

**`get_table_kwargs()` change** (lines 108-119): remove `'contact_person'`/`'contact_email'` from the
`exclude=(...)` tuple in the non-staff branch — they're now always safe to render (blank string for
opted-out rows, populated for opted-in ones), same discipline as the docstring already states
("Belt-and-suspenders: also drop contact columns from the rendered table").

---

### `solsys_code/campaign_tables.py` — `render_site()` inline dropdown (component, request-response)

**Analog:** `render_site()` itself (lines 108-134) — the dict-vs-model dual-accessor pattern this
phase's new branch must preserve.

**Existing dual-accessor + escaping pattern to mirror** (lines 119-134):
```python
site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
if site_short_name:
    return site_short_name
site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
if not site_raw:
    return ''
if Accessor('site_needs_review').resolve(record, quiet=True):
    return format_html(
        '<span class="text-muted font-italic" title="Site could not be automatically resolved">'
        '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i> {}</span>',
        site_raw,
    )
```
Always use `Accessor(...).resolve(record, quiet=True)` to read `pk`/`site_raw`, never
`record.pk`/`record.site_raw` directly (breaks for `.values()` dict rows on the non-staff table —
though the approval queue is staff-only, keep the same dual-accessor style for consistency and in
case `ApprovalQueueTable` is ever reused for a non-staff view).

**New branch to add (SITE-01/D-04, RESEARCH.md Pattern 4):** when `site_short_name` is empty and
`self.show_actions` is True (only the pending/actionable table needs the input — the read-only
`decided_table` keeps today's plain-text rendering), render an `<input list=...>` + `<datalist>`
bound via the HTML5 `form=` attribute:
```python
def render_site(self, record):
    site_short_name = Accessor('site__short_name').resolve(record, quiet=True)
    if site_short_name:
        return site_short_name
    site_raw = Accessor('site_raw').resolve(record, quiet=True) or ''
    if not self.show_actions:
        return super().render_site(record)  # decided-runs table: unchanged read-only rendering
    pk = Accessor('pk').resolve(record, quiet=True)
    datalist_id = f'site-candidates-{pk}'
    candidate_pairs = fuzzy_match_candidates(site_raw, self.candidate_pool) if self.candidate_pool else []
    options = format_html_join('', '<option value="{}">{}</option>', candidate_pairs)
    create_url = reverse('solsys_code_observatory:create') + f'?obscode=&next={approval_queue_url}'
    return format_html(
        '<input type="text" name="site_selection" value="{0}" list="{1}" '
        'form="decide-form-{2}" class="form-control form-control-sm">'
        '<datalist id="{1}">{3}</datalist> '
        '<a href="{4}" class="small">Create new</a>',
        site_raw, datalist_id, pk, options, create_url,
    )
```
Always use `format_html`/`format_html_join`, never `mark_safe` or f-string interpolation of
`site_raw` or any MPC-sourced candidate name — this is the exact T-20-03 stored-XSS precedent
`render_window_start()` already established (lines 136-159, "The raw text is interpolated as a
positional format_html argument so Django auto-escapes it").

---

### `solsys_code/campaign_tables.py` — `render_actions()` single-form refactor (component, request-response)

**Analog:** `render_actions()` itself (lines 205-228) — the two-`<form>` structure that must
collapse into one `<form id="decide-form-{pk}">` with two named submit buttons so `render_site()`'s
new `form="decide-form-{pk}"` input can target it (D-04).

**Current two-form structure (to replace)** (lines 211-228):
```python
return format_html(
    '<div class="d-flex" style="gap: 0.5rem;">'
    '<form method="post" action="{0}" class="d-inline">'
    '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
    '<input type="hidden" name="action" value="approve">'
    '<button type="submit" class="btn btn-sm btn-success">Approve</button>'
    '</form>'
    '<form method="post" action="{0}" class="d-inline">'
    '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
    '<input type="hidden" name="action" value="reject">'
    '<button type="submit" class="btn btn-sm btn-danger" '
    'onclick="return confirm(\'Reject this submission? '
    'The submitter will not be automatically notified.\')">Reject</button>'
    '</form>'
    '</div>',
    decide_url,
    csrf_token,
)
```
**Required shape (RESEARCH.md Pattern 4's concrete example):**
```python
form_id = f'decide-form-{record.pk}'
return format_html(
    '<form id="{0}" method="post" action="{1}">'
    '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
    '<div class="d-flex" style="gap: 0.5rem;">'
    '<button type="submit" name="action" value="approve" class="btn btn-sm btn-success">Approve</button>'
    '<button type="submit" name="action" value="reject" class="btn btn-sm btn-danger" '
    'onclick="return confirm(\'Reject this submission? '
    'The submitter will not be automatically notified.\')">Reject</button>'
    '</div></form>',
    form_id, decide_url, csrf_token,
)
```
Note the buttons switch from two separate hidden `action` inputs (one per form) to two
`name="action" value="approve"/"reject"` submit buttons in one form — `request.POST.get('action')`
in `CampaignRunDecisionView.post()` continues to work unmodified either way. CSRF token minting
(`get_token(self.request)`, line 210) stays exactly as-is — "one token per form instance" still
holds with one form instead of two.

---

### `solsys_code/campaign_utils.py` — new candidate-pool/fuzzy-match helpers (utility, transform)

**Analog:** `resolve_site()` in the same file (lines 100-198) — the "never raise for expected messy
data; return a usable value plus an explicit flag" discipline this phase's new helpers must follow,
plus the module's existing import/docstring conventions.

**Module docstring/import conventions to mirror** (lines 1-22):
```python
"""...structured as "never raise for expected messy data; return a usable value plus
an explicit flag" per the ``_derive_telescope_class`` precedent in ``calendar_utils.py``.
"""

import re
from datetime import date, datetime
...
import requests
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher
```

**New helpers to add** — `build_site_candidates()` (calls the new `MPCObscodeFetcher.query_all()`,
merges with local `Observatory.objects` fields, flattens per RESEARCH.md Pattern 3) and
`fuzzy_match_candidates()` (wraps `difflib.get_close_matches`, resolves matched strings back to
obscodes via the flattened mapping). Both should follow `resolve_site()`'s docstring shape
(Args/Returns, explicit "never raises" contract) and use `import difflib` alongside the existing
`import re` at the top of the file. Cache access (`django.core.cache.cache`) belongs in
`build_site_candidates()`, mirroring `campaign_gap.py`'s `get_or_compute_gap()` shown below —
`campaign_utils.py` currently has no `django.core.cache` import; add it here for the first time in
this module, following the exact `cache.get`/`cache.set` shape from `campaign_gap.py`.

---

### `solsys_code/campaign_gap.py` — cache pattern to mirror (shared, not modified this phase)

**Analog:** `get_or_compute_gap()` (verified in RESEARCH.md, module lines 1-70 read this session).

**Exact cache pattern to copy** into `campaign_utils.py`'s new `build_site_candidates()`:
```python
from django.core.cache import cache

GAP_CACHE_TTL_SECONDS = 3600  # existing precedent in campaign_gap.py

def get_or_compute_gap(campaign, target, site, start, end):
    key = build_gap_cache_key(campaign.pk, target.pk if target else None, site.pk, start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _compute(...)
    cache.set(key, result, timeout=GAP_CACHE_TTL_SECONDS)
    return result
```
Use a module-level constant `MPC_CANDIDATE_CACHE_TTL_SECONDS = 86400` (24h, per RESEARCH.md
Assumption A2) and a fixed cache key (e.g. `'mpc_obscode_candidates'`, no per-request parameters
needed since it's a single global pool) — simpler than `campaign_gap.py`'s parameterized
`build_gap_cache_key()` since there's no per-campaign/per-site variation here.

---

### `solsys_code/solsys_code_observatory/utils.py` — `MPCObscodeFetcher.query_all()` (service, request-response)

**Analog:** `MPCObscodeFetcher.query()` itself (lines 32-62) — the single-code fetch this phase's
bulk-fetch sibling method must not modify or repurpose.

**Existing single-code fetch pattern** (lines 32-62):
```python
def query(self, obscode: str, dbg: bool = False, timeout: float = 10):
    """Query the MPC obscodes API for the specific <obscode>. ..."""
    self.obs_data = None
    response = requests.get(
        'https://data.minorplanetcenter.net/api/obscodes', json={'obscode': obscode}, timeout=timeout
    )
    if response.ok:
        self.obs_data = response.json()
        ...
    else:
        json_resp = response.json()
        errors = self._flatten_error_dict(json_resp)
        logging.error(f'Error: {response.status_code} Message: {". ".join(errors)}')
        ...
        return json_resp
```

**New sibling method (Pattern 2, verified live 2026-07-11):**
```python
def query_all(self, timeout: float = 30) -> dict:
    """Query the MPC obscodes API for every registered observatory code (bulk mode).

    Omitting the ``obscode`` key from the POST body triggers the bulk-list response
    (confirmed live: 2,710 codes, ~1.5 MB, ~1.3s as of 2026-07-11). Stores the result on
    ``self.obs_data`` like ``query()``, but here it is a dict keyed by 3-char obscode
    rather than a single flat observatory dict -- do not call ``to_observatory()`` on a
    ``query_all()`` result, its ``self.obs_data`` shape contract is for ``query()`` only.

    :param timeout: request timeout in seconds.
    :type timeout: float
    :returns: dict keyed by obscode, e.g. {'X09': {'name_utf8': ..., 'longitude': ..., ...}}
    """
    response = requests.get(
        'https://data.minorplanetcenter.net/api/obscodes', json={}, timeout=timeout
    )
    response.raise_for_status()
    self.obs_data = response.json()
    return self.obs_data
```
Keep this as a distinct method — do not repurpose `query()`'s `self.obs_data` shape (a flat dict of
one observatory), since `resolve_site()`'s tier 2 and `CreateObservatory.form_valid()` both depend
on that exact contract via `to_observatory()`.

---

### `solsys_code/solsys_code_observatory/views.py` — `CreateObservatory` `?next=`/`?obscode=` support (controller, request-response)

**Analog:** the same class, current behavior (lines 17-59) — Pitfall 6's documented gap.

**Current unconditional redirect** (lines 26-30):
```python
def get_success_url(self):
    """Create a custom success_url to redirect to the detail page for the
    newly created Observatory.
    """
    return reverse_lazy('solsys_code_observatory:detail', kwargs={'pk': self.kwargs['pk']})
```

**Required change (SITE-02/D-05, Pitfall 6):**
```python
def get_success_url(self):
    next_url = self.request.GET.get('next') or self.request.POST.get('next')
    if next_url and url_has_allowed_host_and_scheme(
        next_url, allowed_hosts={self.request.get_host()}, require_https=self.request.is_secure()
    ):
        return next_url
    return reverse_lazy('solsys_code_observatory:detail', kwargs={'pk': self.kwargs['pk']})
```
Add `from django.utils.http import url_has_allowed_host_and_scheme` to the imports (alongside the
existing `from django.urls import reverse_lazy` at line 8). Also add `get_initial()` reading
`self.request.GET.get('obscode', '')` for the `obscode` field prefill, following the existing
`get_context_data` override style (lines 32-34) already present in this class — same
"call `super()`, return the dict" shape.

**Existing `form_valid` error-handling to preserve unmodified** (lines 44-59):
```python
obs = MPCObscodeFetcher()
errors = obs.query(form.cleaned_data['obscode'])
try:
    obs = obs.to_observatory()
    self.object = obs
    self.kwargs['pk'] = obs.pk
except MissingDataException:
    if errors:
        form.add_error('obscode', errors.get('message', 'Invalid MPC site code'))
    return self.form_invalid(form)
except IntegrityError:
    print('Attempt to create duplicate Observatory')
    messages.error(self.request, 'Attempt to create duplicate Observatory')
return redirect(self.get_success_url())
```
No changes needed here — `get_success_url()` is already called at the end, so the `?next=` support
flows through automatically once `get_success_url()` itself is fixed.

---

### `solsys_code/campaign_forms.py` — VIEW-05 opt-in checkbox (component, request-response)

**Analog:** `CampaignRunSubmissionForm.open_to_collaboration` — the exact "boolean opt-in on a
public form" precedent CONTEXT.md D-07 explicitly names.

**Field + layout pattern to mirror** (lines 18-62):
```python
open_to_collaboration = forms.BooleanField(required=False, label='Open to collaboration?')
contact_person = forms.CharField(max_length=255, required=True, label='Contact person')  # D-06
contact_email = forms.EmailField(required=True, label='Contact email')  # D-06
...
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper = FormHelper()
    self.helper.layout = Layout(
        'campaign',
        Fieldset(
            'Run details',
            'telescope_instrument', 'site_raw', 'obs_date', 'filters_bandpass',
            'observation_details', 'open_to_collaboration',
        ),
        Fieldset('Contact', 'contact_person', 'contact_email', 'comments'),
        ...
    )
```
**Required change:** add `contact_public_opt_in = forms.BooleanField(required=False, label='Show
contact info publicly?')` immediately after the `contact_email` field declaration (per D-07 "placed
immediately after the existing contact_person/contact_email fields"), and add `'contact_public_opt_in'`
to the `Fieldset('Contact', ...)` layout tuple right after `'contact_email'`. Default `required=False`
gives the unchecked/opt-out default (D-07) automatically — same mechanism as `open_to_collaboration`.

---

### `solsys_code/models.py` — `CampaignRun.contact_public_opt_in` field (model, CRUD)

**Analog:** `open_to_collaboration = models.BooleanField(default=False, verbose_name='Open to collaboration?')`
(line 103) — the exact field to mirror per D-07/A1.

**Required addition:**
```python
contact_public_opt_in = models.BooleanField(default=False, verbose_name='Show contact info publicly?')
```
Place near `contact_person`/`contact_email` field declarations (not necessarily adjacent to
`open_to_collaboration`), matching D-07's "placed immediately after the existing contact_person/
contact_email fields" intent for the model too, for readability parity with the form's field order.

---

### `solsys_code/migrations/000N_....py` — AddField migration (migration, batch)

**Analog:** `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py`
(full file, 28 lines) — the most recent `AddField`-only migration in this app, exact shape to mirror.

**Full pattern to copy:**
```python
# Generated by Django 5.2.15 on 2026-07-11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0006_campaignrun_original_obs_date_raw_and_window_needs_review'),
    ]

    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='contact_public_opt_in',
            field=models.BooleanField(
                default=False,
                verbose_name='Show contact info publicly?',
            ),
        ),
    ]
```
Generate via `./manage.py makemigrations solsys_code` rather than hand-writing, then verify the
output matches this shape (dependency on `0006_...`, single `AddField` operation, `default=False`
needing no backfill/data migration per RESEARCH.md's "Runtime State Inventory" — every existing row
correctly defaults to opted-out).

---

### `solsys_code/tests/test_campaign_approval.py` — new/extended test classes (test, request-response)

**Analog:** existing test module structure (`CampaignApprovalTestBase`, `@patch` usage for
`requests.get` mocking tier-2 MPC lookups) — read this session's grep confirmed the file already
has this base class and `@patch('requests.get', ...)` pattern for mocking MPC API calls without
hitting the live API.

**Pattern to follow for `TestSiteFuzzyMatch` (new class):**
```python
@patch('solsys_code.solsys_code_observatory.utils.MPCObscodeFetcher.query_all')
def test_...(self, mock_query_all):
    mock_query_all.return_value = {
        'X09': {'name_utf8': 'Deep Random Survey, Rio Hurtado', 'short_name': 'X09', 'old_names': None,
                'observations_type': 'fixed', 'longitude': -70.9, ...},
        # 5-10 representative entries per RESEARCH.md Wave 0 Gaps, including one
        # observations_type='satellite' entry with longitude=None (Pitfall 4)
    }
    ...
```
Mock at the `MPCObscodeFetcher.query_all` boundary (never `requests.get` directly for this new
method, and never let a test hit the real live API — RESEARCH.md's explicit Wave 0 gap instruction).

**Pattern to follow for Pitfall 3's regression test (extend `TestApproval`):** approve a run where
site resolution succeeds but calendar projection raises (mock `insert_or_create_calendar_event` or
`sun_event` to raise a non-`ValueError` exception via `@patch`), assert `approval_status` reverts to
`PENDING_REVIEW` while `run.site` stays set; approve again and assert `resolve_site`/the underlying
mocked call is NOT called a second time (`@patch(...).assert_not_called()` after the second POST).

**Target fixture reminder (CLAUDE.md):** any `Target` fixture needed for these tests MUST use
`tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory` — FOMO is
exclusively for Solar System / non-sidereal targets.

## Shared Patterns

### Escaping submitter/external-API-sourced text in table cells
**Source:** `solsys_code/campaign_tables.py:render_window_start()` (lines 136-159), already-cited
T-20-03 precedent.
**Apply to:** `render_site()`'s new `<input>`/`<datalist>` branch (SITE-01/D-04) — every
`site_raw`, MPC-sourced candidate `name_utf8`, and `short_name` value MUST go through
`format_html`/`format_html_join` positional substitution, never `mark_safe` or f-string/`%`
interpolation.

### Django low-level cache for an expensive, rarely-changing external list
**Source:** `solsys_code/campaign_gap.py` (`get_or_compute_gap()`, `GAP_CACHE_TTL_SECONDS = 3600`,
`from django.core.cache import cache`).
**Apply to:** `campaign_utils.py`'s new `build_site_candidates()` helper (D-01/D-02) — same
`cache.get`/`cache.set` shape, longer TTL (24h per RESEARCH.md A2).

### "Never raise for expected messy data; return a usable value plus an explicit flag"
**Source:** `solsys_code/campaign_utils.py:resolve_site()` (module docstring, lines 1-9; function
body lines 100-198 — every external-call branch is wrapped in a narrow `except` that falls through
rather than propagating).
**Apply to:** the new `build_site_candidates()`/`fuzzy_match_candidates()` helpers — an MPC bulk-fetch
network failure must fall back to the local-only `Observatory` candidate pool (RESEARCH.md
"Environment Availability" fallback), never raise into `ApprovalQueueView.get_context_data()` and
break the whole approval-queue page render.

### Queryset-level PII gating (restrict the SELECT, not just the template)
**Source:** `solsys_code/campaign_views.py:ALLOWED_FIELDS_FOR_NON_STAFF` +
`CampaignRunTableView.get_queryset()`/`get_table_kwargs()` (lines 52-119) — the existing discipline
this phase's `Case`/`When` change extends rather than replaces.
**Apply to:** VIEW-05's `contact_public_opt_in`-gated `contact_person`/`contact_email` exposure —
gate in the queryset (`Case`/`When`), and correspondingly stop excluding those two fields in
`get_table_kwargs()`'s non-staff branch. Do not add a template-only conditional as a substitute.

### Staff-only access gating via `StaffRequiredMixin`
**Source:** `solsys_code/campaign_views.py:ApprovalQueueView`/`CampaignRunDecisionView` class
declarations (`class ApprovalQueueView(StaffRequiredMixin, TemplateView):`,
`class CampaignRunDecisionView(StaffRequiredMixin, View):`).
**Apply to:** no new view is added this phase, but confirm the new `site_selection` POST field and
the widened candidate-pool computation stay entirely inside these two already-staff-gated views —
never expose MPC bulk data or the fuzzy-match endpoint on an unauthenticated path.

## No Analog Found

None — every file this phase touches has a direct, in-repo precedent to mirror (see table above).
This phase is almost entirely "extend an existing file/method", not "design from scratch" — RESEARCH.md's
own "Don't Hand-Roll" section makes the same observation.

## Metadata

**Analog search scope:** `solsys_code/` (campaign_views.py, campaign_tables.py, campaign_utils.py,
campaign_forms.py, campaign_gap.py, models.py, migrations/, solsys_code_observatory/utils.py,
solsys_code_observatory/views.py, solsys_code_observatory/forms.py, tests/test_campaign_approval.py)
**Files scanned:** 11 read directly this session (all files listed in CONTEXT.md's "Existing Code
this phase's decisions are about" plus their direct dependencies)
**Pattern extraction date:** 2026-07-11
