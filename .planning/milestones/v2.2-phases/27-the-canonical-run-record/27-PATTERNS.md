# Phase 27: The Canonical Run Record - Pattern Map

**Mapped:** 2026-07-29
**Files analyzed:** 11 (create/modify) + 3 test files + 2 paired docs
**Analogs found:** 9 / 11 (2 explicitly have NO in-house analog — see "No Analog Found")

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/models.py` (`CampaignRun` new fields, rename, new link model) | model | CRUD | `solsys_code/models.py` itself (`CampaignRun.Meta.constraints`, `CalendarEventTelescopeLabel`) | exact (self-precedent) |
| `solsys_code/migrations/00XX_rename_calendareventtelescopelabel.py` | migration | batch | `solsys_code/migrations/0004_campaignrun_window_schema.py` (shape only; this one is a pure `RenameModel`, no `RunPython`) | role-match |
| `solsys_code/migrations/00XX_campaignrun_source_telescope_class.py` (`AddField` x3 + new model) | migration | batch | `0004_campaignrun_window_schema.py` (`AddField` ordering precedent) | role-match |
| `solsys_code/migrations/00XX_backfill_source_telescope_class.py` (`RunPython`) | migration | batch/transform | `0004_campaignrun_window_schema.py`'s `backfill_window_fields` / `0005_...`'s `normalize_mismatched_window_pairs` | exact |
| `solsys_code/calendar_utils.py` (`derive_telescope_class`, underscore-strip on 5 helpers) | utility | transform | `solsys_code/calendar_utils.py`'s own `_aperture_class_from_telescope_code` (lines 84-104) | exact (self-precedent) |
| `solsys_code/admin.py` (`CampaignRunAdmin` inlines + `save_formset`, `CalendarEventTelescopeLabelAdmin` rename) | controller (Django admin) | CRUD | `solsys_code/admin.py`'s own current `CampaignRunAdmin` (lines 7-25) for `list_display`/`list_filter`/`readonly_fields` shape; **no in-house inline/`save_formset` precedent exists** — see below | partial (shape only; inline/save_formset is new) |
| `solsys_code/campaign_views.py` (`ALLOWED_FIELDS_FOR_NON_STAFF` +1) | controller | request-response | `solsys_code/campaign_views.py:70-87` itself | exact (self-precedent) |
| `solsys_code/management/commands/import_campaign_csv.py` (writes `source`, calls `derive_telescope_class`) | service (management command) | batch/transform | itself, `fields` dict at line ~182-199 | exact (self-precedent) |
| `src/templates/tom_calendar/partials/event_form.html` (NEW override) | component (template) | request-response | `src/templates/tom_calendar/partials/calendar.html` (lines 215-250, `event.telescope_label_meta.is_verified` usage) | exact |
| `solsys_code/tests/test_canonical_record_migration.py` (NEW) | test | batch | `solsys_code/tests/test_window_schema_migration.py` (full file — `MigrationExecutor` + `TransactionTestCase` shape) | exact |
| `solsys_code/tests/test_campaign_run_observation.py` (or similar, NEW) | test | CRUD | `solsys_code/tests/test_campaign_models.py` (model CRUD test shape — not separately read this session, but is the established sibling for `CampaignRun`-adjacent model tests) | role-match |
| `solsys_code/tests/test_admin.py` (extended — inline formset submission, rename URL fix) | test | request-response | itself, current `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` call sites | exact (self-precedent) |
| D-16 HST tier-2 mock fixture (new `_MPC_OBS_DATA_250`-style dict, wherever the repair-task test lives) | test fixture | event-driven (mocked HTTP) | `solsys_code/tests/test_import_campaign_csv.py:47-58` (`_MPC_OBS_DATA_E10`) + `@patch('requests.get')` call sites | exact |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | file-I/O (notebook) | batch | itself (existing notebook; regenerate via `jupyter nbconvert --execute --inplace`) | exact (self-precedent) |
| `docs/runbooks/telescope_runs_calendar.rst` | doc | — | itself | exact (self-precedent) |

## Pattern Assignments

### `solsys_code/models.py`

**Analog:** the file's own existing `CampaignRun`/`CalendarEventTelescopeLabel` classes (read in full, lines 1-163).

**Rename target** (lines 7-27):
```python
class CalendarEventTelescopeLabel(models.Model):
    """Sidecar record of whether a CalendarEvent's telescope label was live-verified
    against the LCO API or fallback-guessed (TELESCOPE-03/04). One row per
    CalendarEvent at most; no row at all means "verified" by documented default...
    """
    event = models.OneToOneField(
        CalendarEvent, on_delete=models.CASCADE, primary_key=True,
        related_name='telescope_label_meta', verbose_name='Calendar event',
    )
    is_verified = models.BooleanField(default=True, ...)
    def __str__(self):
        return f'{"Verified" if self.is_verified else "Fallback"} label for {self.event.title}'
```
Rename the class only. `related_name='telescope_label_meta'` and `event` field name are
**unchanged** (confirmed by direct read — nothing else references the class name inside this
model body). This is what keeps `views.py`'s `.prefetch_related('telescope_label_meta')` and
`calendar.html`'s `event.telescope_label_meta.is_verified` safe by construction (rename point
3/4 need zero code changes).

**Named partial `UniqueConstraint` pattern to copy for the new link model** (D-02), verbatim
from `CampaignRun.Meta.constraints`:
```python
class Meta:
    constraints = [
        models.UniqueConstraint(
            fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
            condition=models.Q(window_start__isnull=False),
            name='unique_campaign_run_resolved_window',
        ),
        ...
    ]
```
The new observation-link model's constraint should be `fields=('observation_record',)` (a
single-field "at most one link row per observation" constraint — no `condition=` needed since
there's no branching case here, unlike `CampaignRun`'s two-branch design) with an explanatory
comment in the same style: name the constraint, explain *why* a real DB constraint is needed
(concurrent admin saves), and reference D-02 by decision ID the way this file references
`WR-05`/`WR-02`.

**Field style to copy** — every `ForeignKey` here declares `on_delete`, `related_name`, and
`verbose_name` explicitly (see `campaign`, `target`, `site` at lines 60-83); match this for the
new `run` FK on `CalendarEventMeta` (unchanged) and for the new link model's two FKs
(`ForeignKey(CampaignRun, on_delete=CASCADE, related_name=...)`,
`ForeignKey(ObservationRecord, on_delete=CASCADE, related_name=...)` — CASCADE on the run FK per
D-04, and D-04 also implies CASCADE, not SET_NULL, on the observation-record side since the
link row means nothing without either parent... actually re-read D-04: only the run-side
CASCADE is specified; on-delete behavior for the `ObservationRecord` FK is Claude's discretion
since REQUIREMENTS.md doesn't pin it — CASCADE is the safe default matching the run-side choice).

**`TextChoices` pattern to copy for `source`** (D-12 uses the same mechanism), from
`ApprovalStatus`/`RunStatus` (lines 41-58):
```python
class ApprovalStatus(models.TextChoices):
    """Admin review state for a CampaignRun (independent of real-world run outcome)."""
    PENDING_REVIEW = 'pending_review', 'Pending Review'
    APPROVED = 'approved', 'Approved'
    REJECTED = 'rejected', 'Rejected'
```
`source` and `telescope_class` should each be a nested `TextChoices` on `CampaignRun`, matching
this exact shape (docstring one-liner, `snake_case_value = 'snake_case_value', 'Title Case
Label'`). **Reminder from research (Pitfall 1): `telescope_class` values must be stored
lowercase** (`'2m0'`, `'1m0'`, `'0m4'`, `'space'` or `'SPACE'` — confirm casing for `SPACE`
specifically with the planner, since it has no `calendar_utils` lowercase precedent to match
the way `2m0`/`1m0`/`0m4` do) to satisfy D-12's subset assertion against
`_aperture_class_from_telescope_code`'s lowercase set.

**`Field` (not `TextChoices` member) declaration to copy**, from `approval_status`:
```python
approval_status = models.CharField(
    max_length=20, choices=ApprovalStatus, default=ApprovalStatus.PENDING_REVIEW,
    verbose_name='Approval status',
)
```

---

### `solsys_code/migrations/00XX_rename_calendareventtelescopelabel.py` (NEW, hand-authored)

**Analog:** none in `migrations/` is a pure rename — `0004`/`0005` are the closest
`RunPython`-shape precedents but do NOT rename a model. This is a new migration shape for the
project (a `migrations.RenameModel(old_name=..., new_name=...)` operation), but Django's own
`RenameModel` operation is standard and needs no in-house adaptation. Order it first, before any
`AddField` referencing the renamed model, per RESEARCH.md's Pattern 4 ordering.

---

### `solsys_code/migrations/00XX_campaignrun_source_telescope_class.py` (`AddField` x3 + new model)

**Analog:** `0004_campaignrun_window_schema.py`'s header comment style and operation ordering
(lines 1-10) — copy the header-comment convention documenting *why* the ordering matters:
```python
# Generated by Django 5.2.15 on 2026-07-09, hand-edited per
# .planning/phases/19-window-schema-migration/19-RESEARCH.md Pattern 1 to insert the three
# RunPython data-migration steps ... in the load-bearing position: after the new fields are
# added, before the old constraint/fields are removed, and before the two new partial
# constraints are added ...
```
Phase 27's equivalent header should reference `27-RESEARCH.md` Pattern 4 and state the same
kind of load-bearing ordering fact: `AddField(source)`/`AddField(telescope_class)` must precede
the backfill `RunPython` step, and the new `CalendarEventMeta.run` / link-model `CreateModel`
operations are independent of the `AddField`s (no field-value dependency) but should still be
hand-ordered after the rename for readability.

---

### `solsys_code/migrations/00XX_backfill_source_telescope_class.py` (`RunPython`)

**Analog:** `0004_campaignrun_window_schema.py`'s `backfill_window_fields` (lines 20-24) and
`0005_...`'s `normalize_mismatched_window_pairs` (lines 14-39) — both use `apps.get_model()`,
never import the live model, and both log via module-level `logger = logging.getLogger(__name__)`
before any destructive/data-changing step:
```python
def backfill_window_fields(apps, schema_editor):
    """SCHED-05: window_start=window_end=obs_date for every row (NULL stays NULL -> TBD)."""
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    CampaignRun.objects.all().update(window_start=F('obs_date'), window_end=F('obs_date'))
```
and the per-row-with-logging style from `0005`:
```python
def normalize_mismatched_window_pairs(apps, schema_editor):
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    mismatched = CampaignRun.objects.filter(...).order_by('pk')
    for run in mismatched:
        logger.warning('Normalizing ... pk=%s ...', run.pk, ...)
        run.save(update_fields=[...])
```
**RESEARCH.md's own Code Example already drafts the exact shape to use** for
`backfill_telescope_class` (importing `calendar_utils.derive_telescope_class` directly since
it takes primitives — Pitfall 2's one exception) — use it verbatim, with
`reverse_code=migrations.RunPython.noop` matching both precedents (neither `0004` nor `0005`
supplies a real reverse; both are one-way-only data migrations, same posture Phase 27 should
take, per CONTEXT.md's non-reversibility acceptance for this class of migration).

---

### `solsys_code/calendar_utils.py` — `derive_telescope_class()` (D-20)

**Analog:** `_aperture_class_from_telescope_code` (lines 84-104), the existing "never raise,
return value+flag/None" helper right next to `SITE_TELESCOPE_MAP`:
```python
def _aperture_class_from_telescope_code(telescope_code: str | None) -> str | None:
    """Extract the aperture-class token (D-04 vocabulary) from a 4-char telescope code.
    ...
    Returns:
        str | None: '0m4'/'1m0'/'2m0'/'4m0' ... or None if ... (routes the caller to
            fallback per TELESCOPE-03). Never raises.
    """
    if not telescope_code:
        return None
    if len(telescope_code) >= 4 and telescope_code[:3] in {'0m4', '1m0', '2m0', '4m0'}:
        return telescope_code[:3]
    return None
```
`derive_telescope_class(site_raw: str, telescope_instrument: str) -> str` should follow this
same "primitives in, never raise, return the sentinel/blank value on no-match" shape and live
immediately after this function (matching D-20's stated placement "next to
`_aperture_class_from_telescope_code`"). Note per RESEARCH.md Pattern 3: a module docstring
reference to `` `_derive_telescope_class` `` already exists at line 8 of `campaign_utils.py` as
a forward-looking name anticipation — use `derive_telescope_class` (public, no leading
underscore, since the folded todo removes the underscore from its four siblings and this is a
brand-new function that should start in its final public form) or `_derive_telescope_class`
if the planner prefers matching the still-private siblings until the rename lands in the same
phase — either is defensible; RESEARCH.md leaves this as a naming call.

**`SITE_TELESCOPE_MAP` vocabulary this function must stay consistent with** (lines 37-52):
```python
SITE_TELESCOPE_MAP = {
    ('coj', '2m0'): 'COJ-2m0', ('coj', '1m0'): 'COJ-1m0', ('coj', '0m4'): 'COJ-0m4',
    ('ogg', '2m0'): 'OGG-2m0', ('ogg', '0m4'): 'OGG-0m4',
    ('sor', '4m0'): 'SOR-4m0',
    ...
}
```
`4m0` (from `('sor', '4m0')`) is the value D-12's subset-assertion test must explicitly name as
known-excluded from `telescope_class`'s three-value vocabulary.

**Underscore-strip todo** — the five helpers losing their leading underscore:
`_aperture_class_from_telescope_code`, `_derive_telescope`, `_resolve_placement_block`,
`_extract_instrument`, `_coarse_telescope_label`. Simple rename; every call site (in-module and
cross-module) must be grepped and updated together with the def line.

---

### `solsys_code/admin.py`

**Analog for `list_display`/`list_filter`/`readonly_fields` additions:** the file's own current
`CampaignRunAdmin` (lines 7-25, read in full above) — copy the existing list-append style
exactly:
```python
class CampaignRunAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['pk', 'campaign', 'telescope_instrument', 'approval_status',
                     'run_status', 'site', 'window_start', 'window_end']
    list_filter = ['approval_status', 'run_status', 'campaign']
    search_fields = ['telescope_instrument', 'site_raw', 'contact_person']
    readonly_fields = ['approval_status']
```
Append `'source'`/`'telescope_class'` to `list_display` and `list_filter` per D-19; do **not**
add `'source'` to `readonly_fields` (comment why not, mirroring the existing comment style
above `readonly_fields` that explains *why* `approval_status` is there).

**No in-house `TabularInline`/`StackedInline` analog exists anywhere in this project**
(confirmed: `grep -rn "Inline" solsys_code/` returns zero matches). D-06/D-07's two inlines and
the `save_formset` override are **genuinely new patterns for this codebase** — there is nothing
to copy from in-house; RESEARCH.md's own Code Examples section (verified against the installed
Django package's `ModelAdmin.save_formset` source) is the only available precedent:
```python
class CalendarEventMetaInline(admin.TabularInline):
    model = CalendarEventMeta
    fk_name = 'run'
    extra = 0

class CampaignRunObservationInline(admin.TabularInline):
    model = CampaignRunObservation
    fk_name = 'run'
    extra = 0
    readonly_fields = ['confirmed_by', 'confirmed_at']  # set only via save_formset

class CampaignRunAdmin(admin.ModelAdmin):
    inlines = [CalendarEventMetaInline, CampaignRunObservationInline]

    def save_formset(self, request, form, formset, change):
        instances = formset.save(commit=False)
        for instance in instances:
            if isinstance(instance, CampaignRunObservation) and instance.pk is None:
                instance.confirmed_by = request.user
                instance.confirmed_at = timezone.now()
            instance.save()
        formset.save_m2m()
```
Django's base implementation being overridden, confirmed by direct package inspection:
```python
def save_formset(self, request, form, formset, change):
    """Given an inline formset save it to the database."""
    formset.save()
```

**Rename point 1** — the import and the sibling admin class:
```python
from solsys_code.models import CalendarEventTelescopeLabel, CampaignRun
...
class CalendarEventTelescopeLabelAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['event', 'is_verified']
    list_filter = ['is_verified']
    search_fields = ['event__title']
...
admin.site.register(CalendarEventTelescopeLabel, CalendarEventTelescopeLabelAdmin)
```
Both the import and every use of the class name (`CalendarEventTelescopeLabelAdmin`'s own name
is discretionary but conventionally renamed to match; the `register()` call's first argument
must change) need updating together — this is what makes rename point 1 fail loudly
(`ImportError`) if missed.

---

### `solsys_code/campaign_views.py`

**Analog:** the file's own `ALLOWED_FIELDS_FOR_NON_STAFF` (lines 70-87), copy the exact
hand-enumerated style and its explanatory comment above it:
```python
# D-13/VIEW-03/T-15-01: the exact D-09 column list for non-staff requests. Deliberately
# enumerated explicitly (not introspected from CampaignRun._meta) so contact_person/
# contact_email can never accidentally be included ...
ALLOWED_FIELDS_FOR_NON_STAFF = [
    'pk', 'telescope_instrument', 'site__short_name', 'site_raw', 'site_needs_review',
    'window_start', 'window_end', 'filters_bandpass', 'run_status', 'approval_status',
    'open_to_collaboration', 'observation_details', 'weather', 'observation_outcome',
    'publication_plans', 'comments',
]
```
Add `'telescope_class'` to this list per D-18. Per the same decision, add a one-line comment
next to (or above) the list noting that `'source'` is deliberately **not** included and why
(staff-only provenance) — matching this project's convention that a hand-enumerated list's
omissions must read as decisions.

---

### `solsys_code/management/commands/import_campaign_csv.py`

**Analog:** the file's own `fields` dict (lines ~182-199), read in full:
```python
fields = {
    'target': auto_target,
    'site': site,
    'site_raw': site_raw,
    ...
    'approval_status': CampaignRun.ApprovalStatus.APPROVED,  # D-03: bootstrap rows are vetted backfill
    ...
}
```
Add `'source': CampaignRun.Source.CSV_IMPORT` (or chosen enum member name) as a new key,
matching the inline-comment convention (`# D-XX: <why>`). Also add a
`'telescope_class': calendar_utils.derive_telescope_class(site_raw=site_raw,
telescope_instrument=<the row's telescope/instrument value>)` key, calling D-20's shared
helper — this is the second of D-20's two required call sites (the migration's `RunPython` step
is the first).

---

### `src/templates/tom_calendar/partials/event_form.html` (NEW override)

**Analog:** `src/templates/tom_calendar/partials/calendar.html` (lines 215-250, read in full) —
the existing precedent for reading `event.telescope_label_meta` inside a `tom_calendar`
template override:
```django
{% if event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day..." ...
     title="Telescope label is an estimate — could not be verified against the LCO API...">
{% else %}
<div class="cal-event-all-day..." ...>
{% endif %}
```
The new `event_form.html` should read `event.telescope_label_meta.run` (the renamed model's
`related_name` is unchanged, so this accessor works identically to `is_verified` above) and
gate visibility with `event.telescope_label_meta.run.is_publicly_visible` (D-09/D-10) before
rendering a link to the run. **Directory layout that makes Django resolve this override**:
`src/fomo/settings.py:93-107`'s `TEMPLATES['DIRS']` includes `os.path.join(BASE_DIR,
'templates')` **ahead of** `APP_DIRS=True`, so any file placed at
`src/templates/tom_calendar/partials/event_form.html` shadows the installed
`tom_calendar/templates/tom_calendar/partials/event_form.html` — exactly mirroring
`calendar.html`'s existing placement at the identical relative path. `tom_calendar.views.
update_event` renders this partial with `{"form": form, "event": event, "action": "update"}`
already in context (verified against the installed package) — no view override needed.

---

### `solsys_code/tests/test_canonical_record_migration.py` (NEW)

**Analog:** `solsys_code/tests/test_window_schema_migration.py`, full shape (read in full
above) — copy this exactly:
```python
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

class TestWindowSchemaMigrationDataTransform(TransactionTestCase):
    migrate_from = [('solsys_code', '0003_campaignrun_natural_key_unique_constraint')]
    migrate_to = [('solsys_code', '0004_campaignrun_window_schema')]

    def setUp(self):
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps
        TargetList = old_apps.get_model('tom_targets', 'TargetList')
        CampaignRun = old_apps.get_model('solsys_code', 'CampaignRun')
        campaign = TargetList.objects.create(name='3I/ATLAS')
        # ... seed rows against the historical (pre-migration) schema ...
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps
```
The new test module must set `migrate_from`/`migrate_to` bracketing Phase 27's rename +
`AddField` + backfill migrations, seed rows mirroring each D-16 pk shape (JWST-alias,
HST-tier2-mocked via `@patch('requests.get')`, Swift-empty, JUICE-empty, class-wide-empty,
TBD-pair analogue), and assert the `source`/`telescope_class` backfill outcome per row, plus
that the 11 `CalendarEventTelescopeLabel`→renamed-model rows survive the `RenameModel` with
`is_verified` intact (this project's own `RenameModel`-preserves-OneToOne-pk-table fact,
confirmed in RESEARCH.md's Runtime State Inventory against a scratch DB copy).

---

### D-16a HST tier-2 mock fixture

**Analog:** `solsys_code/tests/test_import_campaign_csv.py:47-58` (`_MPC_OBS_DATA_E10`) plus
its `@patch('requests.get')` call sites:
```python
_MPC_OBS_DATA_E10 = {
    'created_at': 'Sat, 25 May 2019 00:11:26 GMT',
    'longitude': '149.07085',
    'name_utf8': 'Siding Spring-Faulkes Telescope South',
    'obscode': 'E10',
    'observations_type': 'optical',
    'old_names': None,
    'rhocosphi': '0.855632',
    'rhosinphi': '-0.516198',
    'short_name': 'Siding Spring-Faulkes Telescope South',
    'updated_at': 'Tue, 15 Apr 2025 20:52:50 GMT',
    'uses_two_line_observations': False,
}
```
and the call-site shape:
```python
@patch('requests.get')
def test_...(self, mock_get):
    mock_response = MagicMock(ok=True)
    mock_response.json.return_value = _MPC_OBS_DATA_E10
    mock_get.return_value = mock_response
```
Build `_MPC_OBS_DATA_250` (HST's real MPC obscode) with the same key shape for D-16a's repair-
task test, and call `resolve_site('250', create_placeholder=False)` per D-22.

---

### Paired docs

**`docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`** — existing notebook, already
committed WITH output (unlike everywhere else in the repo, where pre-commit clears notebook
output). Regenerate via `jupyter nbconvert --to notebook --execute --inplace
docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` after the command's `source`/
`telescope_class` changes land, and add/update cells exercising the new fields with real
executed output (per CLAUDE.md's paired-docs rule).

**`docs/runbooks/telescope_runs_calendar.rst`** — update the section documenting
`import_campaign_csv`'s approval behavior, since CANON-01 changes what it writes for
non-`WEB` sources.

## Shared Patterns

### Data migration shape (`RunPython` + `apps.get_model`)
**Source:** `solsys_code/migrations/0004_campaignrun_window_schema.py`,
`0005_campaignrun_campaign_run_window_start_end_null_together.py`
**Apply to:** the rename migration, the `AddField` migration, and the backfill migration.
```python
def backfill_x(apps, schema_editor):
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')  # never import the live model
    ...
class Migration(migrations.Migration):
    operations = [migrations.RunPython(backfill_x, reverse_code=migrations.RunPython.noop)]
```
The one documented exception: `calendar_utils.derive_telescope_class()` is safe to import
directly inside the `RunPython` function body because it takes only primitives, never a model
instance (D-20's own stated rationale).

### Named partial `UniqueConstraint` with explanatory comment
**Source:** `solsys_code/models.py`, `CampaignRun.Meta.constraints`
**Apply to:** the new observation-link model's uniqueness constraint (D-02).
```python
models.UniqueConstraint(
    fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
    condition=models.Q(window_start__isnull=False),
    name='unique_campaign_run_resolved_window',
),
```
Always name the constraint and explain, in a comment above it, *why* a real DB constraint
(not just app-level validation) is needed — this project's convention ties every constraint to
a decision ID (`WR-05`, `WR-02`) the way this phase should tie its new constraint to `D-02`.

### Migration testing via `MigrationExecutor`
**Source:** `solsys_code/tests/test_window_schema_migration.py`
**Apply to:** the new `test_canonical_record_migration.py`.
See full excerpt above under that file's Pattern Assignment.

### MPC API mocking
**Source:** `solsys_code/tests/test_import_campaign_csv.py:47-58` + `@patch('requests.get')`
**Apply to:** D-16a's HST tier-2 repair-task test.
See full excerpt above.

### Hand-enumerated allow-lists, omissions commented
**Source:** `solsys_code/campaign_views.py:70-87` (`ALLOWED_FIELDS_FOR_NON_STAFF`)
**Apply to:** D-18's `telescope_class` addition / `source` omission.

## No Analog Found

| File/Feature | Role | Data Flow | Reason |
|---|---|---|---|
| `CalendarEventMetaInline` / `CampaignRunObservationInline` (D-06) | admin inline | request-response | Confirmed via `grep -rn "Inline" solsys_code/` — zero matches anywhere in the project. This is a genuinely new pattern; use Django's own standard `TabularInline` idiom (RESEARCH.md's Code Examples), not an in-house adaptation. |
| `CampaignRunAdmin.save_formset()` (D-07) | admin controller method | event-driven (form-submit) | Confirmed via `grep -rn "save_formset\|save_model" solsys_code/` — zero matches anywhere in the project. Use Django's base `ModelAdmin.save_formset()` (verified via package inspection, shown above) as the override target; the stamping logic itself has no in-house precedent to copy, only the general "override a Django framework hook" pattern this project already applies elsewhere (e.g. `CampaignRunSubmissionView.form_valid()`'s own override of Django's `FormView` hook, a *different* hook but the same overriding-a-framework-method idiom). |

## Metadata

**Analog search scope:** `solsys_code/` (models.py, admin.py, calendar_utils.py, campaign_utils.py,
campaign_views.py, migrations/, management/commands/, tests/), `src/templates/tom_calendar/`,
`src/fomo/settings.py`.
**Files scanned:** `models.py`, `admin.py`, `calendar_utils.py` (partial), `campaign_views.py`
(partial), `import_campaign_csv.py` (partial), `migrations/0004_*.py`, `migrations/0005_*.py`,
`tests/test_window_schema_migration.py`, `tests/test_import_campaign_csv.py` (partial),
`templates/tom_calendar/partials/calendar.html` (partial), plus grep sweeps for
`Inline`/`save_formset`/`save_model` (zero hits) across `solsys_code/`.
**Pattern extraction date:** 2026-07-29
