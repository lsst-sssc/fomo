# Phase 37: Status Vocabulary, Public Tallies & Provenance-Blind Gaps - Pattern Map

**Mapped:** 2026-09-18
**Files analyzed:** 14 (new + modified, including paired docs)
**Analogs found:** 14 / 14 (all files have a same-repo, git-tracked analog; several are
self-modifying, i.e. the analog is the file's own current body being consolidated)

All analog paths below were confirmed git-tracked via `git ls-files -- <path>` before being
named. No path under `.gsd/` or any other gitignored mirror is referenced.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/status_vocabulary.py` (new) | utility (pure-logic classifier module) | transform | `solsys_code/campaign_gap.py` (pure-logic core, no view/request concerns) + `solsys_code/observation_projector.py` (`stage_for()` classifier shape) | role-match (composite) |
| `solsys_code/observation_projector.py` (modified) | service | event-driven / transform | itself (`stage_for()`, `title_for()`, `_STAGE_MARKER`) | exact (self, being refactored to import from status_vocabulary) |
| `solsys_code/campaign_reconciler.py` (modified) | service | CRUD / event-driven | itself (`RUN_STATUS_CALENDAR_PREFIX`) | exact (self) |
| `solsys_code/allocation_projector.py` (modified) | service | batch / transform | itself (`allocation_night_title()`, `allocation_events()`) | exact (self) |
| `solsys_code/calendar_utils.py` (modified) | utility | transform | itself (`resolve_placement_block()` line 330-336) | exact (self) |
| `solsys_code/templatetags/calendar_display_extras.py` (modified) | utility / template-tag provider | request-response (display-time) | itself (`campaign_decoration()`, `status_border_css()`, `_OBSERVATION_STATUS_LEGEND`) | exact (self, new tags mirror `campaign_decoration()`) |
| `solsys_code/campaign_gap.py` (modified) | service | CRUD / batch (set-difference over query results, cached) | itself (`claimed_dates()`, `_compute_gap()`, `get_or_compute_gap()`) | exact (self) |
| `solsys_code/campaign_tables.py` (modified) | component (django-tables2 Table) | request-response | itself (`CampaignRunTable.render_run_status`, `render_telescope_class` — Accessor-on-dict-row pattern) | exact (self) |
| `solsys_code/campaign_views.py` (modified) | controller (Django CBVs) | request-response | itself (`CampaignRunTableView.get_queryset()` PII gate, `CampaignListView`) | exact (self) |
| `solsys_code/unattended.py` (modified — new `step_proposal_allocation`) | service (runner step) | event-driven / batch | `step_status_refresh()` in the same file (credentialed-portal-call step shape) | exact |
| `solsys_code/models.py` (modified — new `ProposalTimeAllocation` model + migration; possible `CampaignRun.proposal_code` field) | model | CRUD | `solsys_code/models.py` `WatchedProposal` (small config-keyed model) and `CampaignRunObservation` (link/attribution model) | role-match |
| `src/templates/tom_calendar/partials/event_form.html` (modified — D-09 tally block) | component (Django template partial) | request-response | itself (`{% campaign_decoration event as deco %}` block, lines 161-183) | exact (self, new block is a sibling `{% run_tally %}` block) |
| `solsys_code/tests/test_status_vocabulary.py` (new) | test | transform (unit) | `solsys_code/tests/test_campaign_gap.py` (pure-logic module test) + `solsys_code/tests/test_calendar_display_extras.py` | role-match (composite) |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` (modified, paired doc) | test/doc (pre-executed demo) | batch | itself (existing cells demonstrating campaign lifecycle) | exact (self) |
| `docs/runbooks/telescope_runs_calendar.rst` (modified, paired doc) | doc (runbook) | — | itself (existing status-prefix legend section, "How do I run everything unattended?" section) | exact (self) |

## Pattern Assignments

### `solsys_code/status_vocabulary.py` (new utility, transform)

**Analogs:** `solsys_code/campaign_gap.py` (module docstring/import discipline) and
`solsys_code/observation_projector.py` (classifier shape to promote).

**Module docstring / heavy-import discipline pattern** (`solsys_code/campaign_gap.py:1-14`):
```python
"""Pure-logic core of the coverage-gap analysis feature (GAP-01/GAP-02).
...
This module depends only on the heavy SPICE-loading ephemeris module's read-only,
already-tested sun-event helper for its ephemeris needs -- it must never import the heavy
SPICE-loading ephemeris module (or any module that imports it, such as ``solsys_code.views``)
at module scope.
"""
```
`status_vocabulary.py` must open with the equivalent statement: it is imported by
`observation_projector.py`, `campaign_reconciler.py`, `allocation_projector.py` and
`calendar_display_extras.py`, none of which may pull in `solsys_code.views`/`ephem_utils` —
state this constraint in the new module's own docstring per Claude's Discretion.

**Classifier shape to promote** (`solsys_code/observation_projector.py:86-99`, quoted verbatim
in RESEARCH.md "Architecture Patterns > Pattern 2"):
```python
_FAILURE_MARKER_BY_STATUS = {
    'WINDOW_EXPIRED': '[X]',
    'CANCELED': '[C]',
    'FAILURE_LIMIT_REACHED': '[F]',
    'NOT_ATTEMPTED': '[F]',
}
_STAGE_MARKER = {
    'queued': '[Q]',
    'placed': '[S]',
    'observed': '[O]',
    'completed-no-block': '[O]',
}
```
`status_for()`/`classify()` in the new module should generalize `stage_for()`'s
`get_terminal_observing_states() - get_failed_observing_states()` logic (verified at
`solsys_code/observation_projector.py:103,128`) into the canonical entry point, with markers,
`RUN_STATUS_MARKER = {CANCELLED: '[C]', WEATHER_TECH_FAILURE: '[W]'}` (migrated from
`campaign_reconciler.RUN_STATUS_CALENDAR_PREFIX`, `campaign_reconciler.py:76-79`), and the
ordered `LEGEND` tuple (D-04) migrated from `_OBSERVATION_STATUS_LEGEND`
(`calendar_display_extras.py:140-149`, quoted in RESEARCH.md) plus new `[W]`/`[U]` entries.

**TALLY-03 guard docstring pattern** — state the invariant directly in the module docstring,
mirroring how `campaign_gap.py`'s docstring states its own invariant ("never raise for
expected messy data"); add one negative test in `test_status_vocabulary.py` (see below).

---

### `solsys_code/observation_projector.py`, `campaign_reconciler.py`, `allocation_projector.py`, `calendar_utils.py` (modified — consolidation)

**Analog:** each file's own current body (self-referential refactor — no external analog
needed; these are the three vocabularies being folded together, per RESEARCH.md "Pattern 2").

**Core pattern — replace local dict with import:**
```python
# Before (solsys_code/observation_projector.py:86-99)
_FAILURE_MARKER_BY_STATUS = {...}
_STAGE_MARKER = {...}

# After
from .status_vocabulary import FAILURE_MARKER_BY_STATUS, STAGE_MARKER
```
Same shape for `campaign_reconciler.RUN_STATUS_CALENDAR_PREFIX` (`campaign_reconciler.py:76-79`,
re-imported today by `allocation_projector.py:197`) — becomes an import of
`status_vocabulary.RUN_STATUS_MARKER`.

**`calendar_utils.py` STATUS-02 target** (`calendar_utils.py:330-336`, quoted verbatim in
RESEARCH.md "Pattern 3"):
```python
current_block = None
for block in blocks:
    if block.get('state') == 'COMPLETED':
        current_block = block
        break
    elif block.get('state') == 'PENDING':
        current_block = block
return current_block
```
Replace the bare `'COMPLETED'` literal with `status_vocabulary`'s canonical-state constant
(e.g. `status_vocabulary.OCSState.COMPLETED`); the block-selection logic itself
(COMPLETED-first-else-PENDING) stays unchanged.

---

### `solsys_code/templatetags/calendar_display_extras.py` (modified — new tags)

**Analog:** `campaign_decoration()` in the same file (`calendar_display_extras.py:493-562`,
quoted verbatim in RESEARCH.md "Pattern 1" — read this session).

**Display-time decoration pattern to copy exactly** for `run_tally()`, `campaign_tally()` and
`unused_night_decoration()`:
```python
@register.simple_tag
def campaign_decoration(event: CalendarEvent) -> dict | None:
    if not isinstance(event, CalendarEvent):
        return None
    try:
        meta = event.telescope_label_meta
    except ObjectDoesNotExist:
        return None
    run = meta.run
    if run is None or not run.is_publicly_visible:
        return None
    ...
    return {  # plain dict of pre-computed values, never raises
        'campaign_name': ...,
        'run_status_display': run.get_run_status_display(),
    }
```
`run_tally(run)` (D-09) must follow this exact shape: `isinstance`/`ObjectDoesNotExist`/
`is_publicly_visible` guard, return a plain dict, never raise. `unused_night_decoration()`
(D-12/D-13) adds the `[U]` token check: `end_time < timezone.now()` and no `[C]`/`[W]` in
`run.run_status` (D-14) — read `run.run_status` via the same guarded-run-lookup, never a bare
attribute chain.

**Legend/ring constants to migrate** (`calendar_display_extras.py:132`, `140-149`, quoted
verbatim in RESEARCH.md "Pattern 2" — both must be deleted once callers read from
`status_vocabulary` instead):
```python
_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X] ', '[C] ', '[F] ', '[?] ')
_OBSERVATION_STATUS_LEGEND = (
    {'marker': '[Q]', 'label': 'Queued'},
    ...
)
```
Note the explicit anti-pattern comment already in this file (RESEARCH.md "Anti-Patterns to
Avoid"): the legend must stay "a fixed, ordered vocabulary read from the one module — never
data-driven from the database" — copy that constraint into `status_vocabulary.LEGEND`, do not
derive it from a DB query.

---

### `solsys_code/campaign_gap.py` (modified — GAPB-01 second claim source)

**Analog:** the function's own current body, `claimed_dates()` (`campaign_gap.py:116-211`, full
text read this session).

**Pattern to extend — PII-minimizing `.only()` before iteration:**
```python
qs = CampaignRun.objects.filter(campaign=campaign, site=site, approval_status=CampaignRun.ApprovalStatus.APPROVED)
qs = qs.exclude(run_status__in=_EXCLUDED_RUN_STATUSES)
qs = qs.only('pk', 'window_start', 'window_end')
...
for run in qs:
    ...
    n_days = (run.window_end - run.window_start).days + 1
    for i in range(n_days):
        claimed.add(run.window_start + timedelta(days=i))
return claimed, undated_runs, unattributed_runs, pending_narrowing_runs
```
The new GAPB-01 observation-event claim source must give **its own** explicit
`.only()`/`.values()` restriction on `ObservationRecord`/`CalendarEventMeta` — never a bare
`select_related('run')` that could pull in `CampaignRun.contact_person`/`.contact_email`
(RESEARCH.md Pitfall 3). Follow the same shape: build a separate query, restrict fields
up front, iterate once, union the resulting date set with `claimed` before it is returned
from `claimed_dates()` (D-19 — union, not replace).

**Constants to reuse, not re-derive:**
```python
GAP_CACHE_TTL_SECONDS = 3600  # D-10: 1-hour result cache
_EXCLUDED_RUN_STATUSES = frozenset({CampaignRun.RunStatus.CANCELLED, ...})
```
`get_or_compute_gap()` / `build_gap_cache_key()` (`campaign_gap.py:66`, `253-277`) already wrap
`_compute_gap()` in `django.core.cache`; GAPB-01 extends `_compute_gap()`'s inputs, not the
caching wrapper itself.

---

### `solsys_code/campaign_tables.py` (modified — D-08 Progress column)

**Analog:** `CampaignRunTable.render_run_status` / `render_telescope_class`
(`campaign_tables.py:129-178`, read this session).

**Accessor-on-dict-row pattern (must copy exactly for the new Progress column):**
```python
def render_run_status(self, record):
    """... Reads the raw stored value from ``record`` via Accessor rather than accepting
    django-tables2's pre-resolved ``value`` kwarg: for model-instance rows (staff),
    django-tables2's row machinery auto-calls ``get_run_status_display()`` *before*
    this method runs ... Resolving from ``record`` directly sidesteps that pre-processing
    and gives the raw code for both dict and model rows.
    """
    value = Accessor('run_status').resolve(record, quiet=True)
    css = RUN_STATUS_BADGE_CLASSES.get(value, 'badge-secondary')
    label = CampaignRun.RunStatus(value).label
    style = 'border: 1px solid #6c757d;' if css == 'badge-light' else ''
    return format_html('<span class="badge {}" style="{}">{}</span>', css, style, label)
```
The new `render_progress(self, record)` for D-08's `2 groups · 14 records · [O] 5 [S] 2 ...`
cell must resolve `record`'s pk via `Accessor('pk').resolve(record, quiet=True)` (works for both
dict and model rows) and look up the pre-computed tally dict from the
`get_or_compute_tally(run_pk)` cache (Pattern 5 below) — **never** a per-row DB query inside
`render_progress` (D-08's explicit "never a per-row loop" ban, RESEARCH.md Anti-Patterns).
Imports to copy: `from django_tables2.utils import Accessor`, `from django.utils.html import
format_html` (`campaign_tables.py:10-15`).

---

### `solsys_code/campaign_views.py` (modified — D-10 roll-up + PII gate)

**Analog:** `CampaignRunTableView.get_queryset()` (`campaign_views.py:92-113,165-177`, quoted
verbatim in RESEARCH.md "Pattern 4" — read this session).

**PII-gate pattern (must be preserved exactly by any new tally annotation):**
```python
ALLOWED_FIELDS_FOR_NON_STAFF = [
    'pk', 'telescope_instrument', 'site__short_name', ..., 'comments',
]
qs = qs.values(*[f for f in ALLOWED_FIELDS_FOR_NON_STAFF if f not in ('contact_person', 'contact_email')])
return qs.annotate(
    contact_person=Case(When(contact_public_opt_in=True, then=F('contact_person')), default=Value(''), output_field=CharField()),
    contact_email=Case(When(contact_public_opt_in=True, then=F('contact_email')), default=Value(''), output_field=EmailField()),
)
```
D-10's campaign roll-up (header strip on `campaignrun_table.html`, badge on
`campaign_list.html`) must annotate the SQL-expressible counts (records, groups) after
`.values()` narrows the field list, exactly as `contact_person`/`contact_email` are today —
never widen `ALLOWED_FIELDS_FOR_NON_STAFF` for a tally column that doesn't need it public.
`CampaignListView`'s existing `run_count` annotation (referenced in RESEARCH.md Code Examples)
is the direct analog for the "N runs" badge gaining "· M nights observed".

---

### `solsys_code/unattended.py` (modified — new D-07 fetch step)

**Analog:** `step_status_refresh()` (`unattended.py:254-311`, read this session in full).

**Runner-step pattern to copy exactly:**
```python
def step_status_refresh(dry_run: bool) -> StepResult:
    if dry_run:
        return StepResult(name='status_refresh', failed=False, summary='skipped (dry run)')
    try:
        with command_lock('status_refresh'):
            ...
            return StepResult(name='status_refresh', failed=failed, summary=summary)
    except LockContended:
        return StepResult(name='status_refresh', failed=False, summary='skipped -- lock held')
```
The new `step_proposal_allocation(dry_run)` (or folded into `step_status_refresh`, D-07
planner's choice) must follow this exact shape: dry-run short-circuit before any network call,
`command_lock('proposal_allocation')` context manager, try/except around the portal call,
`StepResult(name=..., failed=..., summary=...)` on both success and lock-contention paths.
**Never log the API key or raw exception body** — mirror the comment discipline at
`unattended.py:286-306` (IN-05/WR-08) that keeps outage summaries credential- and PII-free.
Register in `STEPS = (...)` (`unattended.py` around line 683, per RESEARCH.md's quoted tuple).

---

### `solsys_code/models.py` (modified — new `ProposalTimeAllocation` model)

**Analog:** `WatchedProposal` (small config-keyed model, `solsys_code/models.py:758` region) and
`CampaignRunObservation` (`solsys_code/models.py:575-630`, quoted in RESEARCH.md Code Examples).

**Pattern:** a small model keyed by a natural business key (proposal code + semester +
instrument type), populated only by a runner step, read-only everywhere else — same shape as
`CampaignRunObservation`'s "one row per confirmed attribution" pattern:
```python
# Source: solsys_code/models.py:575-630 (read this session)
run = models.ForeignKey(CampaignRun, on_delete=models.CASCADE, related_name='observation_links', ...)
observation_record = models.ForeignKey(ObservationRecord, on_delete=models.CASCADE, related_name='campaign_run_links', ...)
```
`ProposalTimeAllocation(proposal_code, semester, instrument_type, allocated_hours,
used_hours, fetched_at)` should use `objects.update_or_create(proposal_code=..., semester=...,
instrument_type=...)` from the runner step (mirrors the "create the record if missing,
otherwise update it in place" phrasing CLAUDE.md prefers over "upsert").

---

### `src/templates/tom_calendar/partials/event_form.html` (modified — D-09 tally block)

**Analog:** the existing `{% campaign_decoration event as deco %}` block
(`event_form.html:161-183`, read this session).

**Sibling-block pattern to copy:**
```html
{% campaign_decoration event as deco %}
{% if deco %}
    <div class="row">
      <div class="col">
        <label>
          Attributed campaign run
          <small>{% if deco.table_url %}<a href="{{ deco.table_url }}" target="_blank">View campaign ↗</a>{% endif %}</small>
        </label>
        <div>
          {{ deco.campaign_name }} &mdash; {{ deco.telescope_instrument }}
          {% if deco.window_start %}({{ deco.window_start }}&ndash;{{ deco.window_end }}){% endif %}
          &mdash; {{ deco.run_status_display }}
        </div>
      </div>
    </div>
{% elif not event.telescope_label_meta.run and request.user.is_staff %}
    ...
{% endif %}
```
D-09's tally line belongs **inside** the same `{% if deco %}` block (the tally is read from
`CalendarEventMeta.run`, same guard `run.is_publicly_visible` already applied by
`campaign_decoration()` itself) — add a `{{ deco.tally }}` sub-line or a second
`{% run_tally event as tally %}` tag call rendered in the same `<div class="col">`, not a new
top-level `{% if %}` gate (avoids re-deriving the `is_publicly_visible` check a second time).
Note the existing WR-04 comment about `window_start` gating (`None`-`None` guard) — the same
"guard before rendering a compound field" discipline applies to any tally sub-fields that can
be `None` before the first portal fetch (D-06's "not yet fetched" fallback).

---

### `solsys_code/tests/test_status_vocabulary.py` (new)

**Analogs:** `solsys_code/tests/test_campaign_gap.py` (pure-logic module test structure) and
`solsys_code/tests/test_calendar_display_extras.py` (existing legend/ring test class shape —
per RESEARCH.md's Validation Architecture table, this file needs a new test class, not a new
file, for TALLY-01/02/UNUSED-01; `test_status_vocabulary.py` itself is net-new for STATUS-01/02
and the TALLY-03 negative-write guard).

**TALLY-03 guard test pattern** (RESEARCH.md's Validation Architecture row): use
`unittest.mock.patch` to assert `CampaignRun.save` is never called with a changed `run_status`
by any function imported from `status_vocabulary`, `calendar_display_extras`'s new tags, or the
new `unattended.py` step — a static/negative test, not a behavior test. Fixture any `Target`
needed for setup via `tom_targets.tests.factories.NonSiderealTargetFactory` per CLAUDE.md.

---

## Shared Patterns

### Display-time decoration from a link, never a text write
**Source:** `solsys_code/templatetags/calendar_display_extras.py:493-562` (`campaign_decoration()`)
**Apply to:** `run_tally()`, `campaign_tally()`, `unused_night_decoration()` (all new tags),
and the D-09 template block. No new tag may write to `CalendarEvent.title`/`.description`.

### PII gate at the queryset (`.values()` narrowed before `.annotate()`)
**Source:** `solsys_code/campaign_views.py:92-113,165-177` (`ALLOWED_FIELDS_FOR_NON_STAFF`)
**Apply to:** `campaign_tables.py`'s new Progress column, `campaign_views.py`'s D-10 roll-up
context, and any GAPB-01 query touching `CalendarEventMeta.run` (must not pull in
`CampaignRun.contact_person`/`.contact_email` through the FK).

### The low-level TTL-cache pattern for per-record-in-Python aggregation
**Source:** `solsys_code/campaign_gap.py:28` (`GAP_CACHE_TTL_SECONDS = 3600`),
`get_or_compute_gap()` (`campaign_gap.py:253-277`)
**Apply to:** D-08's tally counts that need `telescope_runs.observing_night()` per record
(not SQL-expressible): a pure `_compute_tally(run)` function, `build_tally_cache_key(run.pk)`,
and `get_or_compute_tally(run)` wrapper — the exact shape D-08 names explicitly, and the fix
for the folded `2026-09-01-add-ttl-cache-to-attribution-banner-count.md` todo pattern.

### Runner-step isolation, locking and summary discipline
**Source:** `solsys_code/unattended.py:254-311` (`step_status_refresh()`), `command_lock()`
**Apply to:** the new D-07 `step_proposal_allocation` — dry-run short-circuit, per-step lock,
try/except, `StepResult`, never log credential values.

### Site-local observing night (never re-derive)
**Source:** `solsys_code/telescope_runs.py:309-333` (`observing_night()`)
**Apply to:** every tally count (D-11) and GAPB-01's per-event night classification (D-16) —
do not write a new date-bucketing function; this one already handles the noon-anchor rule.

### Facility terminal/failed state sets (never re-derive)
**Source:** `solsys_code/observation_projector.py:103,128`
(`facility.get_terminal_observing_states()` / `.get_failed_observing_states()`)
**Apply to:** `status_vocabulary`'s classifier — read from the facility, then map through the
new FOMO-side per-facility table (D-05) rather than trusting it directly.

## No Analog Found

None — every file in scope has a clear, git-tracked, same-repo analog (mostly itself, since
this phase is predominantly a consolidation of three existing vocabularies plus additive
peer-module/tag work in already-established styles). The one item with only external
(non-repo) reference material is the LCO portal `timeallocation_set` fetch shape itself
(D-07's HTTP call body) — RESEARCH.md's "Code Examples" section already documents the response
shape from official LCO docs; no in-repo analog exists for constructing this specific request
beyond the general `LCOFacility`/`_portal_headers()` authentication pattern already used by
every other portal call in `calendar_utils.py` and `solsys_code/facility` integration points.

## Metadata

**Analog search scope:** `solsys_code/` (all modules named in RESEARCH.md's "Recommended
Project Structure"), `solsys_code/templatetags/`, `solsys_code/tests/`,
`src/templates/tom_calendar/partials/`, `docs/runbooks/`, `docs/notebooks/pre_executed/`.
**Files scanned:** 14 target files plus their own current bodies (self-analogs), confirmed
git-tracked via `git ls-files`.
**Pattern extraction date:** 2026-09-18
