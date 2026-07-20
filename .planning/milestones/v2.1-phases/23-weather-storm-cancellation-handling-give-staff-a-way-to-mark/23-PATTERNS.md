# Phase 23: Weather/Storm Cancellation Handling - Pattern Map

**Mapped:** 2026-07-16
**Files analyzed:** 7 (all modified, 0 new)
**Analogs found:** 7 / 7 (all in-repo, most are near-identical siblings within the same file)

No new files are created in this phase — every change lands in an existing module, and in most
cases the closest analog is another function/branch in the *same* file (RESEARCH.md already did
the file-and-line-range identification; this document adds the concrete copy-from excerpts and
classification the planner needs).

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|----------------|------|-----------|-----------------|----------------|
| `solsys_code/management/commands/load_telescope_runs.py` | management command (batch/CRUD) | batch (file → CalendarEvent create-or-update) | `solsys_code/management/commands/sync_lco_observation_calendar.py` (`_FAILURE_PREFIX_BY_STATUS`/`_title_for`) | exact — same title-prefix idiom, different status vocabulary |
| `solsys_code/campaign_views.py` (`CampaignRunDecisionView.post()` + new `_set_run_status()`) | controller (Django `View.post`) | request-response + CRUD | `CampaignRunDecisionView._resolve_site()` (same file, lines 555-684) | exact — same class, same guard/redirect/messages shape |
| `solsys_code/campaign_tables.py` (`ApprovalQueueTable.render_actions()` / `__init__`) | component (django-tables2 column renderer) | request-response (form-in-cell) | `ApprovalQueueTable.render_actions()` resolve-mode branch (same file, lines 299-319) | exact — same class, same per-row mini-form idiom |
| `solsys_code/templatetags/calendar_display_extras.py` (`_TERMINAL_PREFIXES`) | config (module-level constant tuple) | n/a (lookup table) | itself, line 46 | exact — one-line tuple extension |
| `solsys_code/tests/test_load_telescope_runs.py` | test | batch/CRUD | existing title-assertion test in same file (RESEARCH.md: `test_event_fields_set_from_parsed_run`) | exact |
| `solsys_code/tests/test_campaign_approval.py` | test | request-response/CRUD | existing `TestApproval`/`BULK_MPC_FIXTURE`/`ISOLATED_TEST_CACHES` classes in same file | exact |
| `solsys_code/tests/test_calendar_display_extras.py` | test | transform | existing `TestStatusBorderCss`-style parametrized tests in same file | exact |

## Pattern Assignments

### `solsys_code/management/commands/load_telescope_runs.py` (management command, batch)

**Analog:** `solsys_code/management/commands/sync_lco_observation_calendar.py` lines 28-49
(`_FAILURE_PREFIX_BY_STATUS` dict + `_title_for()`).

**Pattern to copy — status→prefix dict, never an if/elif chain:**
```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:28-49
_FAILURE_PREFIX_BY_STATUS = {
    'WINDOW_EXPIRED': '[EXPIRED]',
    'CANCELED': '[CANCELLED]',
    'FAILURE_LIMIT_REACHED': '[FAILED]',
    'NOT_ATTEMPTED': '[FAILED]',
}

def _title_for(record, telescope, instrument, facility, label_was_fallback) -> str:
    prefix = _failure_prefix(record.status, facility)
    if prefix is not None:
        return f'{prefix} {telescope} {instrument}'
    ...
    return f'{telescope} {instrument}'
```

**Exact insertion point in the target file** (`load_telescope_runs.py:139-150`, current code —
title is currently computed unconditionally with no prefix logic):
```python
# Source: solsys_code/management/commands/load_telescope_runs.py:139-150 (BEFORE this phase)
title = f'{parsed.telescope} {parsed.instrument}'
description = (
    f'Dark window (-15 deg, UTC): {dark_start_dt.isoformat()} to {dark_end_dt.isoformat()}\n'
    f'Status: {parsed.status}\n'
    f'Source line: {line.strip()}'
)

event, action = insert_or_create_calendar_event(
    {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
    {'end_time': end_time, 'title': title, 'description': description},
    start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
)
```

**Recommended new code, mirroring the LCO dict pattern exactly (D-01/D-02):**
```python
_CLASSICAL_STATUS_PREFIX = {'cancelled': '[CANCELLED]'}  # D-02: only 'cancelled' has a prefix today

...
prefix = _CLASSICAL_STATUS_PREFIX.get(parsed.status)
title = f'{prefix} {parsed.telescope} {parsed.instrument}' if prefix else f'{parsed.telescope} {parsed.instrument}'
```

**Critical constraint (Pitfall 4 in RESEARCH.md):** `title` MUST be recomputed fresh from
`parsed.telescope`/`parsed.instrument`/`parsed.status` on every `handle()` invocation — never
read/append to `event.title`. `insert_or_create_calendar_event()` already does the no-churn
comparison-and-update, so a freshly-computed unprefixed title on re-ingest automatically reverts
a stale `[CANCELLED]` prefix.

**Status source (`solsys_code/telescope_runs.py:36`, unchanged, read-only reference):**
```python
KNOWN_STATUSES = {'allocation', 'proposed', 'confirmed', 'cancelled', 'not confirmed'}
```

---

### `solsys_code/campaign_views.py` (controller, `CampaignRunDecisionView`)

**Analog:** `CampaignRunDecisionView._resolve_site()`, same file, lines 555-684 — closest sibling
method on the exact same class, same "business-logic guard → conditional queryset `.update()` →
messages → redirect" shape.

**Action-whitelist dispatch pattern to extend** (`campaign_views.py:452-458`):
```python
# Source: solsys_code/campaign_views.py:452-458 (BEFORE this phase)
def post(self, request, pk):
    action = request.POST.get('action')
    if action not in ('approve', 'reject', 'resolve_site'):
        return HttpResponseBadRequest()
    if action == 'resolve_site':
        return self._resolve_site(request, pk)
    new_status = CampaignRun.ApprovalStatus.APPROVED if action == 'approve' else CampaignRun.ApprovalStatus.REJECTED
    updated_count = CampaignRun.objects.filter(
        pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
    ).update(approval_status=new_status)
    ...
```
Extend the whitelist tuple to
`('approve', 'reject', 'resolve_site', 'mark_cancelled', 'mark_weather_failure')` and add an
`elif action in ('mark_cancelled', 'mark_weather_failure'): return self._set_run_status(request, pk, action)`
branch, mirroring the existing `resolve_site` dispatch line exactly.

**Business-logic guard + staleness-safe conditional update pattern to copy**
(`_resolve_site()`, lines 573-580, 629-639):
```python
# Source: solsys_code/campaign_views.py:573-580
run = get_object_or_404(CampaignRun, pk=pk)

# Business-logic bypass guard (Security "business-logic bypass" domain): validate
# state server-side, never just trust the button was only offered on eligible rows.
if run.approval_status != CampaignRun.ApprovalStatus.APPROVED or not run.site_needs_review:
    messages.warning(request, 'This run is not awaiting site resolution.')
    return redirect('campaigns:approval_queue')
```
```python
# Source: solsys_code/campaign_views.py:629-639 — conditional queryset .update() claim pattern
claimed = CampaignRun.objects.filter(
    pk=pk,
    approval_status=CampaignRun.ApprovalStatus.APPROVED,
    site_needs_review=True,
    site_id=previous_site_id,
).update(site=site)
if claimed == 0:
    messages.warning(request, "This run's site was already resolved by someone else.")
    return redirect('campaigns:approval_queue')
run.refresh_from_db()
```

**Recommended new `_set_run_status()` method** (D-03/D-04/D-05, using the guard/update/redirect
shape above, plus the calendar-sync-if-exists guard from Pitfall 1):
```python
_RUN_STATUS_CALENDAR_PREFIX = {
    CampaignRun.RunStatus.CANCELLED: '[CANCELLED]',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: '[WEATHERED]',
}
_ACTION_TO_RUN_STATUS = {
    'mark_cancelled': CampaignRun.RunStatus.CANCELLED,
    'mark_weather_failure': CampaignRun.RunStatus.WEATHER_TECH_FAILURE,
}

def _set_run_status(self, request, pk, action):
    run = get_object_or_404(CampaignRun, pk=pk)
    if run.approval_status != CampaignRun.ApprovalStatus.APPROVED:
        messages.warning(request, 'This run has not been approved yet.')
        return redirect('campaigns:approval_queue')
    new_run_status = _ACTION_TO_RUN_STATUS[action]
    CampaignRun.objects.filter(pk=pk, approval_status=CampaignRun.ApprovalStatus.APPROVED).update(
        run_status=new_run_status
    )
    run.refresh_from_db()
    # Pitfall 1: only update an existing CAMPAIGN:{pk} event -- never fabricate one via
    # get_or_create()'s create-path, which requires non-nullable start_time/end_time that
    # this call deliberately omits (a range/TBD/unresolved-site run never had one projected).
    if CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists():
        prefix = _RUN_STATUS_CALENDAR_PREFIX[new_run_status]
        insert_or_create_calendar_event(
            {'url': f'CAMPAIGN:{run.pk}'},
            fields={
                'title': f'{prefix} {run.campaign.name}: {run.telescope_instrument}',
                'description': run.observation_details,
            },
        )
    messages.success(request, 'Run status updated.')
    return redirect('campaigns:approval_queue')
```

**`_project_calendar_event()` skip-by-design semantics this must mirror**
(`campaign_views.py:372-393`, read-only reference confirming which runs never get a
`CAMPAIGN:{pk}` event in the first place):
```python
# Source: solsys_code/campaign_views.py:388-393
if not (run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end):
    return False
```

**No-churn calendar-event update pattern (D-05), source of truth for both sync commands and
this new branch** (`solsys_code/calendar_utils.py:318-378`, already cited verbatim in
RESEARCH.md):
```python
event, action = insert_or_create_calendar_event(
    {'url': f'CAMPAIGN:{run.pk}'},
    fields={'title': new_title, 'description': new_description},
)
```

---

### `solsys_code/campaign_tables.py` (component, django-tables2 `ApprovalQueueTable`)

**Analog:** `ApprovalQueueTable.render_actions()`, same file, lines 299-319 — the resolve-mode
branch is the closest sibling (single-button mini-form on a non-`show_actions` semantic).

**Imports already present in file (no new imports needed for the recommended shape):**
```python
from django.middleware.csrf import get_token
from django.urls import reverse
from django.utils.html import format_html
from django_tables2 import Accessor
```

**Constructor pattern to extend (`__init__`, lines 202-211)** — add an independent gating flag,
per RESEARCH.md Pitfall 3 (must NOT flip `show_actions=True` for the Decided table, or the
site-search widget leaks in via `render_site()`'s existing `elif not self.show_actions:` branch):
```python
# Source: solsys_code/campaign_tables.py:202-211 (BEFORE this phase)
def __init__(self, *args, show_actions=True, request=None, candidate_pool=None, mode='pending', **kwargs):
    self.show_actions = show_actions
    self.request = request
    self.candidate_pool = candidate_pool
    self.mode = mode
    super().__init__(*args, **kwargs)
```
Add a new `status_actions=False` kwarg (default off, only turned on for the Decided table's
construction) stored as `self.status_actions`, following the exact same
`self.<name> = <name>` assignment style already used for `show_actions`/`mode`.

**`render_actions()` early-return to extend (lines 299-319 for the resolve-mode branch shape;
the current decided-table early-return is a single `if not self.show_actions: return ''`):**
```python
# Source: solsys_code/campaign_tables.py:299-319 (resolve-mode branch, copy this shape)
def render_actions(self, record):
    if not self.show_actions:
        return ''
    decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
    csrf_token = get_token(self.request) if self.request is not None else ''
    if self.mode == 'resolve':
        form_id = f'resolve-form-{record.pk}'
        return format_html(
            '<form id="{0}" method="post" action="{1}">'
            '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
            '<button type="submit" name="action" value="resolve_site" '
            'class="btn btn-sm btn-primary">Resolve</button>'
            '</form>',
            form_id, decide_url, csrf_token,
        )
    ...  # existing pending-mode Approve/Reject branch, unchanged
```

**Recommended new branch (D-04), inserted before the existing `if not self.show_actions: return ''`
so it fires only for the Decided table (`show_actions=False`, `status_actions=True`) — gated on
`approval_status == APPROVED` per RESEARCH.md's Open Question 1 recommendation (buttons always
visible for any APPROVED row, no revert button needed):**
```python
if not self.show_actions:
    if self.status_actions and Accessor('approval_status').resolve(record, quiet=True) == CampaignRun.ApprovalStatus.APPROVED:
        decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
        csrf_token = get_token(self.request) if self.request is not None else ''
        return format_html(
            '<form method="post" action="{0}">'
            '<input type="hidden" name="csrfmiddlewaretoken" value="{1}">'
            '<div class="d-flex" style="gap: 0.5rem;">'
            '<button type="submit" name="action" value="mark_cancelled" '
            'class="btn btn-sm btn-outline-secondary">Mark Cancelled</button>'
            '<button type="submit" name="action" value="mark_weather_failure" '
            'class="btn btn-sm btn-outline-secondary">Mark Weathered</button>'
            '</div></form>',
            decide_url, csrf_token,
        )
    return ''
```

**Call-site to update — `ApprovalQueueView.get_context_data()` decided-table construction**
(`campaign_views.py:335-341`, read-only reference — pass the new flag here, `show_actions` stays
`False`, per Pitfall 3):
```python
# Source: solsys_code/campaign_views.py:335-341 (BEFORE this phase)
decided_table = ApprovalQueueTable(
    list(decided_qs),
    prefix='decided-',
    show_actions=False,
    empty_text='No decisions recorded yet.',
    order_by=(),
)
```
Add `status_actions=True, request=self.request,` to this call (the request is needed for the new
branch's `get_token(self.request)`, same as `pending_table`/`review_table` already pass).

**`RUN_STATUS_BADGE_CLASSES` — confirmed no change needed (`campaign_tables.py:32-41`):**
```python
RUN_STATUS_BADGE_CLASSES = {
    ...
    CampaignRun.RunStatus.CANCELLED: 'badge-light',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: 'badge-light',
}
```

---

### `solsys_code/templatetags/calendar_display_extras.py` (config constant)

**Analog:** itself — a pure one-line tuple extension, no structural analog needed.

**Current state (line 46, confirmed live):**
```python
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]')
```
`'[CANCELLED]'` is already present — D-01/D-02's classical prefix and D-03's `CampaignRun`
`CANCELLED` prefix get the box-shadow ring with NO change here. Only add the new
`WEATHER_TECH_FAILURE` prefix:
```python
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]')
```

**Consumer (line 97, read-only reference, unchanged):**
```python
if any(title.startswith(p) for p in _TERMINAL_PREFIXES):
```

---

## Shared Patterns

### Title-prefix-from-fixed-dict idiom
**Source:** `solsys_code/management/commands/sync_lco_observation_calendar.py:28-49`
(`_FAILURE_PREFIX_BY_STATUS`)
**Apply to:** `load_telescope_runs.py` (`_CLASSICAL_STATUS_PREFIX`) and `campaign_views.py`
(`_RUN_STATUS_CALENDAR_PREFIX`). Both new dicts must be keyed on the fixed vocabulary
(`parsed.status` string / `CampaignRun.RunStatus` enum member), never on free-form/user-supplied
text — this is also the Security "V5 Input Validation" control for this phase.

### No-churn create-or-update via `insert_or_create_calendar_event()`
**Source:** `solsys_code/calendar_utils.py:318-378`
**Apply to:** `load_telescope_runs.py` (existing call, unchanged shape) and the new
`campaign_views.py::_set_run_status()` (new call, guarded by a pre-check
`CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()` — see Pitfall 1 in
RESEARCH.md). Never construct/save a `CalendarEvent` directly in either file.

### Business-logic bypass guard on staff-facing decide actions
**Source:** `CampaignRunDecisionView._resolve_site()`, `campaign_views.py:576-580`
**Apply to:** the new `_set_run_status()` method — re-check `run.approval_status == APPROVED`
server-side even though the Decided-table button is only rendered for APPROVED rows client-side
(V4 Access Control control for this phase).

### CSRF-protected per-row mini-form
**Source:** `ApprovalQueueTable.render_actions()`, `campaign_tables.py:306-319`
(`get_token(self.request)` hidden field, POST to `campaigns:decide`)
**Apply to:** the new Decided-table status-change buttons — reuse the identical
`get_token`/`format_html`/named-submit-button shape, never a hand-rolled CSRF token or a GET link.

## No Analog Found

None — every file in scope has a direct, load-bearing analog already read and excerpted above
(RESEARCH.md's own line-level tracing made this an exceptionally high-coverage phase; no file
needs to fall back to RESEARCH.md's abstract Code Examples section alone).

## Metadata

**Analog search scope:** `solsys_code/management/commands/`, `solsys_code/campaign_views.py`,
`solsys_code/campaign_tables.py`, `solsys_code/templatetags/`, `solsys_code/calendar_utils.py`
**Files scanned:** 7 target files + 3 read-only analog/reference files
(`sync_lco_observation_calendar.py`, `calendar_utils.py`, `telescope_runs.py`)
**Pattern extraction date:** 2026-07-16
</content>
