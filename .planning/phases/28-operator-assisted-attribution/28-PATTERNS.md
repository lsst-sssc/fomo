# Phase 28: Operator-Assisted Attribution - Pattern Map

**Mapped:** 2026-08-01
**Files analyzed:** 13 (7 new, 6 modified)
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/campaign_attribution.py` (new matcher module — name is Claude's Discretion) | service (scoring/candidate-generation) | transform / batch (computed on the fly, no persistence) | `solsys_code/campaign_utils.py` `fuzzy_match_candidates()`/`substring_or_fuzzy_match_candidates()` (lines 549, 580) | role-match (difflib scoring precedent; no existing "candidate generation over two querysets with a hard gate" analog exists, so this is the closest available) |
| Attribution view(s) (new, in `campaign_views.py` or a peer module) | controller (staff worklist + POST actions) | request-response, CRUD (writes `CalendarEventMeta.run` / creates `CampaignRunObservation` / creates dismissal rows) | `ApprovalQueueView` (lines 328-397) + `CampaignRunDecisionView` (lines 534-661) | exact (same staff-gated multi-table view + POST-action-view pairing) |
| `src/templates/campaigns/attribution_queue.html` (new) | template | request-response (SSR) | `src/templates/campaigns/approval_queue.html` (24 lines, full file read) | exact (multi-table staff page shape; UI-SPEC explicitly mirrors it) |
| New django-tables2 table classes (candidate/orphan tables, Dismissed, Confirmed) | component (table renderer) | transform (queryset/list → rendered rows) | `ApprovalQueueTable` (campaign_tables.py lines 208-415), `CampaignRunTable` (lines 56-207) | role-match (Meta.exclude/sequence trimming and render_actions() idiom both reusable; the multi-select shape is new — see Divergence Note below) |
| Two dismissal models (`CalendarEventDismissal`, `ObservationRecordDismissal` — names Claude's Discretion) | model | CRUD | `CampaignRunObservation` (models.py lines 330-390) | exact (same shape: real FKs, `confirmed_by`-style audit fields, named `UniqueConstraint`) |
| Migration adding 2 dismissal models + 2 `CalendarEventMeta` fields | migration | batch (schema) | `solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py` (hand-authored `CreateModel` + `AddField`, lines 1-90+) | exact (same operation types: CreateModel with confirmed_by/confirmed_at FK+DateTimeField, plus AddField) |
| New test files (`test_campaign_attribution.py`, `test_campaign_attribution_views.py`, extension to `test_admin.py`) | test | CRUD / event-driven (POST actions), pure-function (matcher) | `solsys_code/tests/test_campaign_run_observation.py` (fixture/constraint style), `solsys_code/tests/test_admin.py::CampaignRunAdminInlinesTests` (lines 217-430, `save_formset` stamping tests) | exact |
| `solsys_code/models.py` (modified: `CalendarEventMeta` gains 2 fields + updated D-05 comment) | model | CRUD | same file, `CampaignRunObservation.confirmed_by`/`confirmed_at` (lines 363-371) | exact (field shape to copy verbatim) |
| `solsys_code/campaign_views.py` (modified: new view(s) + shared orphan-count helper) | controller | request-response | `runs_needing_site_review()` (lines 184-202) + `CampaignListView.get_context_data()` (lines 219-235) | exact |
| `solsys_code/campaign_urls.py` (modified: new routes) | route | request-response | existing `campaigns` namespace `urlpatterns` list (full file, 34 lines) | exact |
| `solsys_code/admin.py` (modified: `save_formset` gains `CalendarEventMeta` branch) | middleware/hook (admin save path) | event-driven (form save) | `CampaignRunAdmin.save_formset()` (lines 220-252) | role-match — same method, but the new branch's condition differs in kind (see Divergence Note) |
| `src/templates/campaigns/campaign_list.html` (modified: count banner) | template | request-response | same file's existing `pending_count`/`site_review_count` banner (lines 9-28) | exact |
| `docs/runbooks/telescope_runs_calendar.rst` (modified: new attribution section) | docs | — | "How do I reach the approval queue?" section (lines 150-179) | exact (structure/tone to match) |

## Pattern Assignments

### `solsys_code/campaign_attribution.py` (service, transform/batch)

**Analog:** `solsys_code/campaign_utils.py:549-577` (`fuzzy_match_candidates()`)

**Imports pattern** (`campaign_utils.py` top of file, representative):
```python
import difflib
```
(the matcher module should import `difflib` the same bare way — no wrapper library.)

**Core difflib scoring pattern to extend** (`campaign_utils.py:549-577`):
```python
def fuzzy_match_candidates(site_raw: str, candidate_pool: dict[str, str], n: int = 5) -> list[tuple[str, str]]:
    text = (site_raw or '').strip()
    if not text:
        return []
    matches = difflib.get_close_matches(text, candidate_pool.keys(), n=n, cutoff=0.6)
    return [(match, candidate_pool[match]) for match in matches]
```
Do **not** reuse the 0.6 cutoff or `get_close_matches` (a best-N-picks API) for D-11's
instrument-similarity signal — RESEARCH.md Pitfall 1 shows the naive whole-string ratio for
`"FTS/MuSCAT4"` vs `"2M0-SCICAM-MUSCAT"` is 0.500, below this codebase's own 0.6 convention.
Use `difflib.SequenceMatcher(None, a, b).ratio()` directly on tokenised strings (split on `/`,
`-`, whitespace) instead, following the same "substring/token pre-pass before falling back to a
raw ratio" discipline `substring_or_fuzzy_match_candidates()` (`campaign_utils.py:580+`)
establishes — that function is the second precedent to read for the *shape* of the fallback
layering, even though its substring logic itself doesn't transfer directly.

**Docstring convention** (from both functions above): Google-style, with an explicit `Args:`/
`Returns:` section and an inline citation of which D-xx decision or research pitfall the
behavior encodes — every new function in the matcher module should carry the same style.

**Telescope-vocabulary bridge — reuse, don't reinvent:**
```python
# Source: solsys_code/calendar_utils.py:40 SITE_TELESCOPE_MAP, :87 aperture_class_from_telescope_code,
# :148 derive_telescope_class, :353 extract_instrument
```
RESEARCH.md Pattern 2 flags that no code today bridges `Observatory.obscode` (e.g. `E10`) to
LCO's 3-letter site codes (e.g. `coj`) used by `SITE_TELESCOPE_MAP`. `telescope_runs.SITES`
(`solsys_code/telescope_runs.py:17-22`) is a *different*, narrower dict (classical-run-file
nickname → obscode, only 4 entries) that happens to confirm `FTS → E10`, but does not cover the
7 LCO/SOAR site codes. Any new alias table the matcher needs must be verified against the real
`Observatory` table or the MPC API per entry, mirroring `HORIZONS_OBSERVER_TO_OBSCODE`'s
extension-rule discipline (`solsys_code/observer_codes.py`) — never hand-typed from memory
(RESEARCH.md Assumption A1).

**Record-side date-window extraction to reuse or promote:**
```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:108-136
def _time_window(record: ObservationRecord) -> tuple[datetime, datetime]:
    if record.scheduled_start is None and record.scheduled_end is None:
        start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
        end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
    elif record.scheduled_start is not None and record.scheduled_end is not None:
        start_time = record.scheduled_start
        end_time = record.scheduled_end
    else:
        raise ValueError(...)
    return start_time, end_time
```
Currently module-private in `sync_lco_observation_calendar.py`. The matcher needs the identical
fallback logic for the record-side date-overlap signal — either import it (promote to non-
underscore, or add a thin re-export) rather than re-deriving the parsing rules. 10 of the 11 real
matching records have `scheduled_start`/`scheduled_end` = `NULL`, so the `parameters['start']`/
`['end']` fallback branch is not a rare edge case here — it is the common case.

**Error handling:** No function in `campaign_utils.py`'s matching helpers raises for expected-
messy input — blank/absent text returns an empty list/None rather than raising (see both
functions' docstrings: "Never raises."). The matcher module should follow the same discipline:
missing `window_start`/`window_end` or missing instrument strings should degrade the relevant
signal's contribution to zero/neutral, never throw, so one orphan's bad data can't crash the
whole worklist render.

---

### Attribution view(s) (controller, request-response + CRUD)

**Analog:** `ApprovalQueueView` (`campaign_views.py:328-397`) for the GET/worklist assembly;
`CampaignRunDecisionView` (`campaign_views.py:534-661`) for the POST confirm/dismiss/undo
actions.

**Staff gate + multi-table GET pattern** (`campaign_views.py:328-341`):
```python
class ApprovalQueueView(StaffRequiredMixin, TemplateView):
    template_name = 'campaigns/approval_queue.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        pending_qs = CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
        ).select_related('campaign', 'site')
        ...
        pending_table = ApprovalQueueTable(pending_qs, prefix='pending-', request=self.request, ...)
        ...
        RequestConfig(self.request).configure(pending_table)
```
Copy: `StaffRequiredMixin` as the first base class, `TemplateView`, materializing querysets to
lists before handing them to a table when the queryset needs Python-side sorting or has already
been sliced (see the `decided_qs`/`[:20]` comment at lines 347-360 — the exact same "Cannot
reorder a query once a slice has been taken" trap applies to any capped Dismissed/Confirmed
section), and `RequestConfig(self.request).configure(table)` for pagination.

**Atomic conditional-update idiom to generalise for the event-side confirm** (`campaign_views.py:558-561`):
```python
updated_count = CampaignRun.objects.filter(
    pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
).update(approval_status=new_status)
```
For `CalendarEventMeta.run`, key the same shape on `run__isnull=True` instead of a status enum:
```python
updated_count = CalendarEventMeta.objects.filter(
    event_id=event_pk, run__isnull=True
).update(run=run_pk, confirmed_by=request.user, confirmed_at=timezone.now())
```
`updated_count == 0` means "already attributed or dismissed by someone else" — reuse the exact
3-way branch at lines 651-660 (`updated_count == 1` / row exists but not updated / row gone) for
the event-side confirm's messaging.

**Record-side confirm is structurally different — a row is CREATED, not a field set.** Use
`get_or_create()` guarded by the existing `unique_campaign_run_observation_record`
`UniqueConstraint` (models.py:384-387), catching `IntegrityError` inside its own
`transaction.atomic()` savepoint the way `CampaignRunSubmissionView.form_valid()` already does
for its own natural-key collision (`campaign_views.py:288-302`) — do not treat this as the same
"conditional `.update()`" shape as the event side; it is `get_or_create()` + `IntegrityError`
catch instead.

**Multi-select bulk confirm (D-09):** loop the single per-row conditional `.update()` inside one
`transaction.atomic()` block — **do not** issue one combined
`CalendarEventMeta.objects.filter(pk__in=checked_pks, run__isnull=True).update(run=...)`, since
each checked row may point at a *different* candidate run and a combined `.update()` can only set
one value for every matched row (RESEARCH.md Anti-Pattern, explicitly called out). Sum
`updated_count` across the loop to report "N candidates confirmed" vs. "M already claimed",
per UI-SPEC's copy contract.

**Undo (D-13):** write a new dismissal row for the pair being un-confirmed, then clear
`CalendarEventMeta.run` (an `.update(run=None)`, still conditional on the current owning run to
guard the same race) or delete the `CampaignRunObservation` row — the two target shapes differ
exactly as the confirm actions do (field-clear vs. row-delete), mirroring the confirm-side
divergence above.

**Messaging tone to copy verbatim** (from `campaign_views.py:658,660`, cited directly in
UI-SPEC's Copywriting Contract):
```python
messages.warning(request, 'This run was already decided by someone else.')
messages.error(request, 'This run no longer exists.')
```
UI-SPEC's exact required strings: `'This candidate was already confirmed or dismissed by
someone else.'` (warning) / `'This candidate no longer exists.'` (error) / `'Attribution
confirmed.'` / `'N candidates confirmed.'` / `'Candidate dismissed.'` / `'Confirmation undone —
back in the queue.'` / `'Dismissal undone — back in the queue.'` (all `messages.success`).

**Shared orphan-count helper (D-02 banner):**
```python
# Source: solsys_code/campaign_views.py:184-202
def runs_needing_site_review():
    """... single definition of "needs site review"; both CampaignListView (a bare .count()
    for the staff banner) and ApprovalQueueView (the full table) call this ..."""
    return CampaignRun.objects.filter(approval_status=CampaignRun.ApprovalStatus.APPROVED, site_needs_review=True)
```
Write one analogous module-level function (e.g. `orphans_needing_attribution_count()` or two
functions, one per orphan kind, summed by the caller) and have **both** the new banner clause in
`campaign_list.html`/`CampaignListView.get_context_data()` and the attribution page's own section
counts call it — never a second inline `.count()` (this is the exact silent-drift hazard the
docstring above names).

---

### `src/templates/campaigns/attribution_queue.html` (template, request-response)

**Analog:** `src/templates/campaigns/approval_queue.html` (full file):
```django
{% extends 'tom_common/base.html' %}
{% load render_table from django_tables2 %}
{% block title %}Approval Queue{% endblock %}

{% block content %}
<h4 class="font-weight-bold mb-4">Approval Queue</h4>

<h5 class="font-weight-bold mb-3">Pending Review</h5>
{% render_table pending_table %}

<h5 class="font-weight-bold mb-3 mt-4">Recently Decided</h5>
{% render_table decided_table %}

<div class="card border-warning mt-4">
  <div class="card-header bg-warning text-dark font-weight-bold">Sites Needing Review — action required</div>
  <div class="card-body">
    <p class="mb-3">...</p>
    {% render_table review_table %}
  </div>
</div>
{% endblock %}
```
Copy: `{% extends 'tom_common/base.html' %}`, `{% load render_table from django_tables2 %}`,
`<h4 class="font-weight-bold mb-4">` page title, `<h5 class="font-weight-bold mb-3">` section
titles, `{% render_table ... %}` per section. **Do not** copy the `card border-warning`
"action required" treatment for the two worklist sections — UI-SPEC explicitly reserves that
styling for the D-02 count banner only; the worklists are routine triage. Use Bootstrap
`collapse` (a `<button data-toggle="collapse">` header, no new JS) for the Dismissed/Confirmed
sections, matching the "Sites Needing Review" card's plain-card precedent in structure but
adding the collapse behavior UI-SPEC requires.

---

### New django-tables2 table classes (component, transform)

**Analog:** `ApprovalQueueTable` (`campaign_tables.py:208-415`), `CampaignRunTable.Meta` (lines
56-207).

**`Meta.exclude`/`Meta.sequence` column-trimming pattern to copy** (`campaign_tables.py:229-238`):
```python
class Meta(CampaignRunTable.Meta):  # noqa: D106
    exclude = ('weather', 'observation_outcome', 'publication_plans')
    sequence = (
        'actions',
        'approval_status',
        'telescope_instrument',
        'site',
        'window_start',
        '...',
    )
```
Apply the same discipline (16-05's fix) to the two new orphan tables: exclude structurally-always-
blank columns for the candidate-pair shape (an orphan row has no `approval_status`; a candidate
row's evidence columns are per UI-SPEC's Evidence Column Contract), and front-load whichever
column needs to be reachable without horizontal scrolling (band badge, per UI-SPEC "leftmost").

**CSRF + `render_actions()` idiom — reference for conventions, NOT a template to copy:**
```python
# Source: solsys_code/campaign_tables.py:351-414 (single-form-per-row POST idiom)
def render_actions(self, record):
    decide_url = reverse('campaigns:decide', kwargs={'pk': record.pk})
    csrf_token = get_token(self.request) if self.request is not None else ''
    form_id = f'decide-form-{record.pk}'
    return format_html(
        '<form id="{0}" method="post" action="{1}">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="{2}">'
        '<div class="d-flex" style="gap: 0.5rem;">'
        '<button type="submit" name="action" value="approve" class="btn btn-sm btn-success" ...>Approve</button>'
        '<button type="submit" name="action" value="reject" class="btn btn-sm btn-danger" ...>Reject</button>'
        '</div></form>',
        form_id, decide_url, csrf_token, record.pk,
    )
```
**What carries over:** CSRF handling via `django.middleware.csrf.get_token(self.request)` minted
per row (never per-template), `reverse()` for URL building, `.btn-sm .btn-success`/`.btn-danger`
button-class conventions, and the HTML5 `form=` cross-reference trick (used by `render_site()`'s
site-search widget at lines 265-299/301-349 to target an out-of-line form) — reuse that exact
`form="..."` attribute mechanism for the per-row single-candidate "Confirm" button so it can
coexist inside a table whose High-band checkboxes belong to one page-spanning bulk `<form>`.

**What does NOT carry over:** the one-`<form>`-per-row structure itself. D-09's multi-select
needs one `<form>` wrapping the *entire* worklist table (checkboxes named `candidate_ids`,
multi-value), with only the single-candidate "Confirm" buttons using the `form=` attribute to
target that one page-spanning form from inside a table row — do not instantiate a `<form>` per
row the way `render_actions()` does today; that shape cannot express "one submit, many selected
rows."

**Candidate-grouping ("pick one of these") has no existing analog** — `ApprovalQueueTable`
renders one row = one `CampaignRun`, with no concept of "one orphan expanded to N candidate
alternatives." UI-SPEC's Candidate Grouping Contract (nested rows under one orphan identity,
light `border-top` between candidates in a group, heavier separator between groups) is new
table-rendering logic this phase must write — there is no `django-tables2` idiom already in this
codebase for grouped/nested rows to extend.

---

### Two dismissal models (model, CRUD)

**Analog:** `CampaignRunObservation` (`models.py:330-390`, full class read):
```python
class CampaignRunObservation(models.Model):
    run = models.ForeignKey(CampaignRun, on_delete=models.CASCADE, related_name='observation_links', ...)
    observation_record = models.ForeignKey(ObservationRecord, on_delete=models.CASCADE, related_name='campaign_run_links', ...)
    confirmed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='confirmed_campaign_run_observations', verbose_name='Confirmed by',
    )
    confirmed_at = models.DateTimeField(null=True, blank=True, verbose_name='Confirmed at')

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=('observation_record',), name='unique_campaign_run_observation_record'),
        ]
```
Copy field shape directly for each of the two new dismissal models, renaming `confirmed_*` to
`dismissed_by`/`dismissed_at` (D-06) and adding a `reason = models.TextField(blank=True, ...)`
free-text field. Each model needs a real FK to its orphan side (`event`/`observation_record`) and
a real FK to `CampaignRun`, plus a **named** `UniqueConstraint` on the `(orphan, run)` pair —
follow the "named `UniqueConstraint` with an explanatory comment" convention this file already
establishes (`CampaignRun.Meta.constraints`, this class's own constraint above) — do **not**
default to a bare `unique_together` tuple, which this codebase never uses.

**`on_delete` choice to note (asymmetric from the CampaignRunObservation precedent):** the
comment at `models.py:352-355` explains why `CampaignRunObservation.observation_record` is
`CASCADE` (an orphaned link "carries nothing and means nothing"). A dismissal row is different:
D-06/D-07 need it to survive as an audit trail even if the orphan or run is later deleted — for
each dismissal model, decide `on_delete` per field with that survivability requirement in mind
(likely `CASCADE` on the orphan FK matching the orphan's own semantics, but this is a genuinely
new judgement call, not a direct copy).

---

### Migration adding 2 dismissal models + 2 `CalendarEventMeta` fields (migration, batch/schema)

**Analog:** `solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py`
(hand-authored, lines 1-90+):
```python
class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0009_calendareventmeta_run'),
        ('tom_observations', '0016_alter_facility_options'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(model_name='campaignrun', name='source', field=models.CharField(...)),
        migrations.AddField(model_name='campaignrun', name='telescope_class', field=models.CharField(...)),
        migrations.CreateModel(
            name='CampaignRunObservation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('confirmed_at', models.DateTimeField(blank=True, null=True, verbose_name='Confirmed at')),
                ('confirmed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                    related_name='confirmed_campaign_run_observations', to=settings.AUTH_USER_MODEL,
                    verbose_name='Confirmed by')),
                ...
            ],
        ),
    ]
```
Copy: top-of-file comment block documenting *why* the migration was hand-authored rather than
`makemigrations`-generated (if the planner chooses to hand-author, matching this file's
precedent) and the deliberate operation ordering (`AddField`s before `CreateModel`, dependency on
`migrations.swappable_dependency(settings.AUTH_USER_MODEL)` since both dismissal models will FK
to the user model). RESEARCH.md Assumption A4 flags that combined-vs-split migration sequencing
is left to the planner (no strong precedent forces either choice); this migration is nonetheless
the correct *field-shape* template for `CreateModel` with a `confirmed_by`/`confirmed_at`-style
FK+DateTimeField pair, adapted to `dismissed_by`/`dismissed_at`/`reason`.

**Do not** use `0012_unflag_class_wide_campaignrun_site_review.py` as the schema-migration
template — that file is a one-way **data** migration (`RunPython`, no schema change at all) and
is the wrong shape for this phase's `CreateModel`/`AddField` needs; it is listed only because it
is the most recent migration in the directory.

---

### `solsys_code/models.py` (modified — `CalendarEventMeta` gains 2 fields)

**Analog:** `CampaignRunObservation.confirmed_by`/`confirmed_at` (`models.py:363-371`):
```python
confirmed_by = models.ForeignKey(
    settings.AUTH_USER_MODEL,
    on_delete=models.SET_NULL,
    null=True,
    blank=True,
    related_name='confirmed_campaign_run_observations',
    verbose_name='Confirmed by',
)
confirmed_at = models.DateTimeField(null=True, blank=True, verbose_name='Confirmed at')
```
Copy verbatim onto `CalendarEventMeta` (D-12), choosing a distinct `related_name` (e.g.
`confirmed_calendar_event_metas`) to avoid a reverse-accessor clash with the existing one.

**The D-05 comment that MUST be updated, not left contradicting the code** (`models.py:34-38`,
current text):
```python
# D-05: deliberately a bare FK with no confirmed_by/confirmed_at. Phase 26 locked this
# shape and accepted the resulting audit asymmetry with the observation link (which does
# carry confirmed_by/confirmed_at) -- an event attribution records no who or when, and a
# future event-side undo will therefore be untraceable. This is a decision, not an
# omission -- do not "fix" it here.
```
D-12 explicitly reopens this. The planner/executor must replace this comment with one
documenting the D-12 revisit (that Phase 28 added `confirmed_by`/`confirmed_at` here, closing
the audit asymmetry the D-05 comment used to describe), per CLAUDE.md's explicit instruction not
to leave a stale contradicting comment.

---

### `solsys_code/campaign_views.py` (modified — new view(s) + shared helper)

Already covered above (Attribution view(s) section) — same file, same analogs
(`ApprovalQueueView`, `CampaignRunDecisionView`, `runs_needing_site_review()`).

### `solsys_code/campaign_urls.py` (modified — new routes)

**Analog:** the existing `campaigns` namespace (full file, 34 lines):
```python
app_name = 'campaigns'

urlpatterns = [
    path('', CampaignListView.as_view(), name='list'),
    ...
    path('approval-queue/', ApprovalQueueView.as_view(), name='approval_queue'),
    path('site-search/', SiteSearchView.as_view(), name='site_search'),
    path('<int:pk>/decide/', CampaignRunDecisionView.as_view(), name='decide'),
    ...
]
```
Copy: flat `urlpatterns` list, `path('<slug>/', View.as_view(), name='...')` style, `<int:pk>/`
prefix pattern for per-object POST-action routes (matching `<int:pk>/decide/`). Add
`attribution/`, plus per-pair POST targets under it (e.g. `attribution/event/<int:pk>/confirm/`,
`attribution/event/<int:pk>/dismiss/`, `attribution/event/<int:pk>/undo/`, and the analogous
`record/` triple) — naming is Claude's Discretion, but should mirror `<int:pk>/decide/`'s
single-dispatching-view-with-an-`action`-POST-param shape (see `CampaignRunDecisionView.post()`)
rather than one URL per action, unless the confirm/dismiss/undo split needs distinct
per-orphan-kind pk-resolution logic that makes a single dispatcher awkward.

### `solsys_code/admin.py` (modified — `save_formset` gains a `CalendarEventMeta` branch)

**Analog (existing branch to extend):**
```python
# Source: solsys_code/admin.py:232-246
instances = formset.save(commit=False)
for instance in instances:
    if isinstance(instance, CampaignRunObservation) and instance.pk is None:
        instance.confirmed_by = request.user
        instance.confirmed_at = timezone.now()
    instance.save()
for obj in formset.deleted_objects:
    obj.delete()
formset.save_m2m()
```
**Divergence Note (RESEARCH.md Pattern 4 / Pitfall 4 — read carefully):** the
`CampaignRunObservation` branch's `instance.pk is None` gate means "this row was just created" —
correct for that model because a `CampaignRunObservation` row's existence *is* the confirmation
(D-01). `CalendarEventMeta` is different: its row already exists (created at telescope-label-
resolution time) with `event` as its primary key, so `instance.pk is None` is **never true** for
it — a copy-pasted `pk is None` gate silently never stamps anything. The correct condition is a
**`run_id` None → not-None transition**, requiring the prior DB value fetched before the mutated
in-memory instance overwrites it:
```python
# New logic this phase must write — no existing helper does this diffing:
if isinstance(instance, CalendarEventMeta):
    prior_run_id = CalendarEventMeta.objects.filter(pk=instance.pk).values_list('run_id', flat=True).first()
    if prior_run_id is None and instance.run_id is not None:
        instance.confirmed_by = request.user
        instance.confirmed_at = timezone.now()
```
(Or capture `prior_run_id` from `formset.initial_forms` before `save(commit=False)` mutates the
instance, per RESEARCH.md's alternative phrasing — either approach is acceptable, but the
`pk is None` gate must NOT be reused verbatim for this branch.)

### `src/templates/campaigns/campaign_list.html` (modified — count banner)

**Analog:** the existing two-queue banner in the same file (lines 9-28, full block):
```django
{% if request.user.is_staff %}
{% if pending_count or site_review_count %}
<div class="alert alert-warning d-flex justify-content-between align-items-center mb-4">
  <span>
    <i class="fa fa-exclamation-circle mr-2"></i>
    {% if pending_count %}{{ pending_count }} submission{{ pending_count|pluralize }} pending review{% endif %}
    {% if pending_count and site_review_count %}&middot;{% endif %}
    {% if site_review_count %}{{ site_review_count }} run{{ site_review_count|pluralize }} needing site review{% endif %}
  </span>
  <a href="{% url 'campaigns:approval_queue' %}" class="btn btn-sm btn-warning">Review queue</a>
</div>
{% endif %}
{% endif %}
```
**Load-bearing structural note (the template comment above the block explicitly warns about
this — copy the nesting, not just the visual result):** the `{% if request.user.is_staff %}`
wrapping a nested `{% if pending_count or site_review_count %}` must stay two **nested** tags,
never flattened, because Django's `{% if %}` binds `and` tighter than `or` — a flattened
condition would leak the banner to anonymous visitors whenever the new orphan count is nonzero.
UI-SPEC direction: extend this same banner's count clauses (or stack a second `alert-warning`
block immediately beneath it) with the new orphan-attribution count — do **not** introduce a
blue `alert-info` banner; this must reuse `alert-warning`/`btn-warning` identically.

### `docs/runbooks/telescope_runs_calendar.rst` (modified — new attribution section)

**Analog:** the "How do I reach the approval queue?" section (lines 150-179, representative
excerpt):
```rst
How do I reach the approval queue?
---------------------------------------

The approval queue (``campaigns:approval_queue``) hosts **two independent
work queues**, not one:

* **Pending Review** -- public submissions awaiting a staff approve/reject
  decision.
* **Sites Needing Review — action required** -- approved runs whose
  observing site never resolved ...

The entry point is the warning banner at the top of ``/campaigns/``,
visible to staff only. As of this phase, it appears whenever **either**
queue has rows, and names each count separately ...

**Behavior change:** before this phase, the banner was driven by the
pending-review count alone. ...
```
Copy: `rst` section-heading underline style (`---...` matching the heading's exact character
count), bulleted enumeration of the sub-sections/worklists with bold lead-in terms, an explicit
**Behavior change:** callout paragraph when a prior phase's behavior is being extended (exactly
this phase's situation for the campaign-list banner), and cross-references via `` :ref: `` /
plain section-name backticks to other runbook sections. New section should cover: what the
attribution page is, the D-15 "done" signal (queue empty + stated remaining count) Phase 29
depends on, and what a dismissal means (D-05: "not an association").

## Shared Patterns

### Staff gating
**Source:** `solsys_code/mixins.py` (full file, 11 lines):
```python
class StaffRequiredMixin:
    @method_decorator(user_passes_test(lambda u: u.is_staff))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
**Apply to:** every new view class this phase adds (the attribution worklist view and all
confirm/dismiss/undo POST-action views) — it is not inherited automatically; each class must
declare `StaffRequiredMixin` as a base explicitly, exactly as `ApprovalQueueView`/
`CampaignRunDecisionView` already do.

### Atomic conditional state-transition idiom
**Source:** `solsys_code/campaign_views.py:558-561` (`CampaignRunDecisionView.post()`).
**Apply to:** the event-side confirm/undo action (keyed on `run__isnull=True` / `run_id=<pk>`
instead of `approval_status`). The record-side confirm/undo uses the sibling
`get_or_create()`/`IntegrityError` idiom instead (see `CampaignRunSubmissionView.form_valid()`,
`campaign_views.py:288-302`) — both are "shared patterns" in the sense that every new POST view
must pick the correct one of these two race-safe idioms per target model, never a
`select_for_update()` pair (RESEARCH.md's Don't-Hand-Roll table explicitly forbids introducing a
second, inconsistent concurrency-control style).

### Messages framework tone
**Source:** `solsys_code/campaign_views.py:644-660` (the `messages.success`/`.warning`/`.error`
triad on approve/reject).
**Apply to:** every confirm/dismiss/undo action — reuse exactly the wording UI-SPEC specifies
(quoted in the Attribution-view section above), matching this codebase's existing tone rather
than inventing new copy.

### CSRF handling in django-tables2 render methods
**Source:** `solsys_code/campaign_tables.py:365-366` (`get_token(self.request)` inside
`render_actions()`), and the class docstring at lines 214-218 explaining why CSRF must be handled
inside the table (not the template loop) because `{% render_table %}` doesn't hand row-rendering
control back to the template.
**Apply to:** every new table class's action-rendering methods; the `request` object must be
passed in at table-construction time (as `ApprovalQueueTable.__init__` already requires) so a
CSRF token can be minted per row.

### Named `UniqueConstraint`s with explanatory comments
**Source:** `solsys_code/models.py:373-388` (`CampaignRunObservation.Meta.constraints`).
**Apply to:** both new dismissal models' `(orphan, run)` pair constraints, and any constraint
touching `CalendarEventMeta`'s new fields (none expected, but the comment-density convention
applies to any new `Meta.constraints` this phase adds).

## No Analog Found

| File/Concern | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Candidate-grouping nested-row rendering ("pick one of these" — UI-SPEC's Candidate Grouping Contract) | component | transform | No existing `django-tables2` table in this codebase groups multiple rows under one shared parent identity; `ApprovalQueueTable` is flat, one row per `CampaignRun`. This is new rendering logic, not an extension of an existing pattern. |
| Obscode ↔ LCO-site-code alias bridge (D-11's telescope-match signal, beyond the confirmed `coj→E10`) | utility | transform | RESEARCH.md Pattern 2/Assumption A1: no existing code bridges these two vocabularies for the other 6 LCO/SOAR sites; must be built new and verified per-entry against the live `Observatory` table or MPC API, following `HORIZONS_OBSERVER_TO_OBSCODE`'s extension-rule discipline as the closest *process* analog (not a code template). |
| `save_formset`'s `run_id` None→not-None transition detection for `CalendarEventMeta` | middleware/hook | event-driven | RESEARCH.md Pattern 4/Pitfall 4: the existing `CampaignRunObservation` branch's `pk is None` gate is the wrong shape for this model and must not be copied verbatim — flagged above in the admin.py Pattern Assignment with the concrete divergent logic required. |

## Metadata

**Analog search scope:** `solsys_code/*.py` (models.py, campaign_views.py, campaign_utils.py,
campaign_tables.py, campaign_urls.py, admin.py, calendar_utils.py, mixins.py, telescope_runs.py,
migrations/, tests/), `src/templates/campaigns/*.html`, `docs/runbooks/telescope_runs_calendar.rst`.
**Files scanned:** 17 source files + 2 templates + 1 runbook + migration directory listing +
test-file directory listing, all read directly (no analog asserted without reading the actual
lines cited above).
**Pattern extraction date:** 2026-08-01
