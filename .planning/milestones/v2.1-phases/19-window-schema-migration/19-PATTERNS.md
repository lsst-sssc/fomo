# Phase 19: Window-Schema Migration - Pattern Map

**Mapped:** 2026-07-09
**Files analyzed:** 16 (7 non-test source modules + 1 migration + 6 test files + 1 demo notebook + models.py itself)
**Analogs found:** 16 / 16 — every file to change is itself its own best "analog" (this is a modify-in-place phase, not a greenfield-file phase). Closest external analog cited per file below for the *shape of the change*, not a different file to imitate wholesale.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `solsys_code/models.py` (CampaignRun fields/constraints) | model | CRUD | itself (prior shape) + `migrations/0003_campaignrun_natural_key_unique_constraint.py` for constraint precedent | exact |
| `solsys_code/migrations/0004_campaignrun_window_schema.py` (new) | migration | batch/transform | `solsys_code/migrations/0002_campaignrun.py` (AddField shape) + `0003_campaignrun_natural_key_unique_constraint.py` (AddConstraint shape) | role-match (no prior `RunPython` data migration exists in this app; this is the first) |
| `solsys_code/campaign_utils.py` (`parse_obs_window`, `insert_or_create_campaign_run`) | service/utility | transform + CRUD | itself — `resolve_site()`'s "never raise for expected messy data, return value + flag" discipline in the same file is the pattern to preserve | exact |
| `solsys_code/campaign_gap.py` (`claimed_dates`) | service | transform | itself — `observable_dates()`'s per-date loop + log+skip discipline in the same file | exact |
| `solsys_code/campaign_views.py` (`CampaignRunDecisionView.post()`, `ALLOWED_FIELDS_FOR_NON_STAFF`) | controller | request-response | itself — `campaign_gap.observable_dates()`'s `except ValueError: log+skip` for the new `sun_event()` call site | exact |
| `solsys_code/campaign_tables.py` (`CampaignRunTable`/`ApprovalQueueTable`) | component (django-tables2 Table) | transform (render) | itself — `render_site()`'s dict-vs-model dual accessor | exact |
| `solsys_code/campaign_forms.py` (`CampaignRunSubmissionForm`) | component (Django Form) | request-response | itself — plain `forms.Form` (not `ModelForm`) convention already established | exact |
| `solsys_code/management/commands/import_campaign_csv.py` | controller (management command) | batch | itself — existing `seen_fallback_keys`/`parse_obs_window` collision-disambiguation loop | exact |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | test/demo | file-I/O | itself (prior executed cells) | exact |
| `solsys_code/tests/test_campaign_models.py`, `test_campaign_approval.py`, `test_campaign_gap.py`, `test_campaign_views.py`, `test_campaign_submission.py`, `test_import_campaign_csv.py` | test | CRUD/request-response | themselves — existing `TestCase` fixture/assertion shapes in each file | exact |

## Pattern Assignments

### `solsys_code/models.py` — field replacement + partial constraints

**Analog:** current file (lines 89-91 fields, 114-125 constraint), plus `migrations/0003_campaignrun_natural_key_unique_constraint.py` for the `UniqueConstraint` construction idiom.

**Fields to remove** (lines 89-91):
```python
obs_date = models.DateField(null=True, blank=True, verbose_name='Observation date')
ut_start = models.DateTimeField(null=True, blank=True, verbose_name='UT start time')
ut_end = models.DateTimeField(null=True, blank=True, verbose_name='UT end time')
```

**Fields to add** (same position):
```python
window_start = models.DateField(null=True, blank=True, verbose_name='Observing window start')
window_end = models.DateField(null=True, blank=True, verbose_name='Observing window end')
```

**Constraint to remove/replace** (lines 114-125, keep the race-safety comment style):
```python
class Meta:  # noqa: D106
    constraints = [
        # WR-05: backs the natural key insert_or_create_campaign_run's docstring ...
        models.UniqueConstraint(
            fields=['campaign', 'telescope_instrument', 'ut_start'],
            name='unique_campaign_run_natural_key',
        ),
    ]
```
Replace with the two partial constraints from RESEARCH.md Pattern 2 (`unique_campaign_run_resolved_window` keyed on `campaign, telescope_instrument, window_start, window_end` with `condition=Q(window_start__isnull=False)`; `unique_campaign_run_tbd_natural_key` keyed on `campaign, telescope_instrument, contact_person` with `condition=Q(window_start__isnull=True)`) — copy verbatim from RESEARCH.md's Pattern 2 code block, which is already written against this exact model.

**`__str__` (line 128)** references `self.obs_date` — must become `self.window_start` (or a window-aware rendering); this is a one-line, easy-to-miss update since it's outside the field block being edited.

### `solsys_code/migrations/0004_campaignrun_window_schema.py` (new file)

**Analog:** `solsys_code/migrations/0002_campaignrun.py` (for `AddField`/model-creation shape — read directly, confirms this app's migration file header/`dependencies` convention) and `0003_campaignrun_natural_key_unique_constraint.py` (for `AddConstraint` shape, reproduced above). No prior `RunPython` data migration exists anywhere in this app, so there is no in-repo analog for the backfill/dedup steps — **use RESEARCH.md's Pattern 1 code block verbatim** (`backfill_window_fields`, `dedupe_tbd_collisions`, and the exact 8-step `operations` ordering); RESEARCH.md's version is already written specifically for this model/migration chain (`dependencies = [('solsys_code', '0003_campaignrun_natural_key_unique_constraint')]`).

**Practical generation workflow** (from RESEARCH.md, load-bearing — do not skip): edit `models.py` first, run `./manage.py makemigrations solsys_code`, then hand-insert the two `RunPython` operations at the position Pattern 1 specifies, then smoke-test `./manage.py migrate solsys_code` against a throwaway/copy DB before considering the task done (Pitfall 2/3 in RESEARCH.md — wrong operation order causes `FieldDoesNotExist`/`IntegrityError` mid-migration).

### `solsys_code/campaign_utils.py` — `insert_or_create_campaign_run` lookup key

**Analog:** itself, `resolve_site()`'s messy-data discipline (lines 85-183) is the established tone to match for any new window-parsing edge cases (none are added this phase per CONTEXT.md's scope note — `parse_obs_window()` keeps returning a single exact date).

**Docstring/lookup key to update** (lines 277-310, esp. line 291 comment "D-04: campaign, telescope_instrument, ut_start" and the actual call site in `import_campaign_csv.py` line 162): the `lookup` dict's key changes from `{'campaign':..., 'telescope_instrument':..., 'ut_start': ut_start}` to `{'campaign':..., 'telescope_instrument':..., 'window_start': window_start}` (single-night collapse, `window_start == window_end`). `insert_or_create_campaign_run()` itself (lines 301-310) needs no logic change — it's generic over `lookup`/`fields` dicts; only callers change what they pass.

**`parse_obs_window()` signature/return** (lines 186-244): per CONTEXT.md's explicit scope note, this phase does NOT add range/TBD parsing — it keeps returning a single exact date, but the *field name* in the returned tuple's consumers changes from `obs_date`/`ut_start`/`ut_end` to `window_start == window_end`. The function's internal regex/fallback logic (lines 214-244) is untouched; only its callers' field mapping changes.

### `solsys_code/campaign_gap.py` — `claimed_dates()` rewrite

**Analog:** itself — `observable_dates()`'s per-date loop (lines 109-135) and log+skip discipline is the pattern `claimed_dates()` should now mirror even more closely (iterating a date range directly, no per-run night-derivation).

**Use RESEARCH.md's Code Examples section verbatim** — the rewritten `claimed_dates()` (already read against this exact file) replaces lines 138-211, and **deletes** `_observing_night_date()` (lines 87-106) and its now-dead `from zoneinfo import ZoneInfo`/`from datetime import ... time` imports (Pitfall 4 — `ruff check .` will catch a missed unused import as `F401`).

**`.only()` call to update** (line 182): `qs.only('pk', 'obs_date', 'ut_start')` → `qs.only('pk', 'window_start', 'window_end')`.

### `solsys_code/campaign_views.py` — D-06 calendar projection + PII allowlist

**Analog:** itself — `except ValueError:` log+skip pattern already used by `campaign_gap.observable_dates()` (imported into this same module's dependency graph) is the discipline to copy for the new `sun_event()` call site (Pitfall 7).

**Gate to replace** (line 309, inside `CampaignRunDecisionView.post()`):
```python
if run.telescope_instrument and run.ut_start and run.ut_end:
    insert_or_create_calendar_event(
        {'url': f'CAMPAIGN:{run.pk}'},
        fields={..., 'start_time': run.ut_start, 'end_time': run.ut_end, ...},
    )
```
Replace with RESEARCH.md's "D-06 hybrid calendar projection" Code Example (ground/space branch via `run.site.observations_type == Observatory.SATELLITE_OBSTYPE`, `sun_event(run.site, run.window_start, kind='sun')` wrapped in `try/except ValueError` per Pitfall 6/7) — already written against this exact call site and imports `Observatory` (already imported at line 31) plus `telescope_runs.sun_event` (new import needed).

**`ALLOWED_FIELDS_FOR_NON_STAFF`** (lines 49-67): swap `'obs_date'`, `'ut_start'`, `'ut_end'` (lines 55-57) for `'window_start'`, `'window_end'` — keep the list's shape (explicit enumeration, never introspected — this is a locked security control per RESEARCH.md's Security Domain section, V4 Access Control).

### `solsys_code/campaign_tables.py` — window column + TBD/range rendering + sort

**Analog:** itself — `render_site()` (lines 111-137) is the established dict-vs-model dual-accessor precedent; `render_run_status()`/`render_approval_status()` (lines 84-109) are the established "resolve raw value via `Accessor`, format via `format_html`" precedent for the new `render_window_start()`.

**Column list to update** (`Meta.fields`, lines 55-72): replace `'obs_date'`, `'ut_start'`, `'ut_end'` with `'window_start'` (single combined column per D-03/D-05 — `window_end` need not be a separate visible column, it's read inside the render method).

**`order_by` to remove** (line 73, `order_by = ('-obs_date',)`, tagged `# D-10`): per RESEARCH.md Pattern 4, D-04's nulls-last sort can't be expressed in `Meta.order_by` (django-tables2 only compiles bare accessor strings, not `F(...).desc(nulls_last=True)`) — remove this line, apply the sort in the view's `get_queryset()` instead (`CampaignRunTableView.get_queryset()`, `campaign_views.py`), and pass `order_by=()` in `get_table_kwargs()` so django-tables2 doesn't re-sort and clobber it (mirrors the existing `decided_table = ApprovalQueueTable(..., order_by=())` precedent already in `campaign_views.py` line 261 — read directly, exact same "list already correctly sorted, suppress table's own default" idiom).

**New `render_window_start()` method** — copy RESEARCH.md's Pattern 3 code block verbatim (already written against this file's `Accessor` import at line 14 and `format_html` import at line 13):
```python
def render_window_start(self, record):
    """D-03/D-05: TBD badge/text, single date, or 'start -> end' range."""
    start = Accessor('window_start').resolve(record, quiet=True)
    end = Accessor('window_end').resolve(record, quiet=True)
    if start is None:
        return format_html('<span class="badge badge-secondary">TBD</span>')
    if start == end:
        return start
    return format_html('{} -&gt; {}', start, end)
```
`ApprovalQueueTable.Meta.sequence` (lines 169-178) also references `'obs_date'`, `'ut_start'`, `'ut_end'` (line 174-176) — update to the single `'window_start'` entry.

### `solsys_code/campaign_forms.py` — `CampaignRunSubmissionForm` field collapse

**Analog:** itself — the existing plain-`forms.Form` (never `ModelForm`) convention (module docstring, lines 1-7) must be preserved for whatever field shape replaces `obs_date`/`ut_start`/`ut_end`.

**Fields to remove** (lines 24-26):
```python
obs_date = forms.DateField(required=False, label='Observation date')
ut_start = forms.DateTimeField(required=False, label='UT start time')
ut_end = forms.DateTimeField(required=False, label='UT end time')
```
Per RESEARCH.md's Open Question 1/Assumption A3 (medium-low confidence, flagged for a quick sanity check but not blocking): collapse to a single `obs_date`-style `forms.DateField` that the view (`CampaignRunSubmissionView.form_valid()`, `campaign_views.py` lines 166-182) maps to both `window_start`/`window_end` on `CampaignRun.objects.create(...)` — the two `DateTimeField` UT inputs are dropped entirely, not repurposed. Also update the `Layout(...)` call (lines 45-57) which references `'obs_date'`, `'ut_start'`, `'ut_end'` by name (lines 51-53).

**Consumer to update in the same task:** `CampaignRunSubmissionView.form_valid()` (`campaign_views.py` lines 166-182) currently passes `obs_date=form.cleaned_data['obs_date']`, `ut_start=...`, `ut_end=...` directly into `CampaignRun.objects.create(...)` — becomes `window_start=form.cleaned_data['obs_date'], window_end=form.cleaned_data['obs_date']` (single-night collapse) if the form keeps `obs_date` as its field name, or `window_start`/`window_end` directly if the form field is renamed. The `IntegrityError` friendly-error message at lines 183-192 ("same campaign+telescope_instrument+ut_start") also needs its wording updated to reflect whichever new natural-key description applies (TBD-branch `contact_person`-based collision message differs from the resolved-window message).

### `solsys_code/management/commands/import_campaign_csv.py` — natural-key lookup + collision tracking

**Analog:** itself — the existing `seen_fallback_keys: dict[tuple, int]` collision-disambiguation loop (lines 82, 118-132) is the pattern to preserve; only the key's field composition changes.

**Lookup key to update** (line 162):
```python
run, action = insert_or_create_campaign_run(
    {'campaign': campaign, 'telescope_instrument': telescope_instrument, 'ut_start': ut_start},
    fields,
)
```
becomes `{'campaign': campaign, 'telescope_instrument': telescope_instrument, 'window_start': window_start}` (with `window_end` also set inside `fields`, and `fields['obs_date']` (line 146) removed since `obs_date` no longer exists as a distinct field — `window_start`/`window_end` are the same value for a single night per D-05 test rewrite guidance).

**`seen_fallback_keys` collision key** (line 123, `(campaign.pk, telescope_instrument, ut_start)`) and the offset logic (line 126, `ut_start + timedelta(seconds=collision_count)`): per RESEARCH.md Pitfall 5, this whole mechanism depended on `ut_start` having a time-of-day component to offset by seconds — `window_start` is date-only, so a seconds-offset no longer disambiguates two colliding rows. This needs a **new** disambiguation approach (not a rename) — e.g. tracking the collision count and either logging+skipping the duplicate (consistent with D-07/D-08's dedup philosophy) or another explicit strategy; RESEARCH.md flags the paired test (`test_import_campaign_csv.py::test_duplicate_unparseable_ut_time_rows_do_not_merge`) as needing scenario-level rethinking, not a mechanical rename (Pitfall 5).

### `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — CLAUDE.md-mandated companion update

**Analog:** itself (current executed cells). Line 309 contains `CampaignRun.objects.filter(campaign=campaign).order_by('obs_date', 'ut_start')` — update to `.order_by('window_start')` (or `F('window_start').desc(nulls_last=True)` if mirroring the new default sort), then regenerate via `jupyter nbconvert --to notebook --execute --inplace` and commit with real executed output per CLAUDE.md's demo-notebook-companion rule. **This file is not on CONTEXT.md's 15-file list but must be included in `files_modified`** — RESEARCH.md's Runtime State Inventory explicitly flags this gap.

## Shared Patterns

### Field rename discipline: `obs_date`/`ut_start`/`ut_end` → `window_start`/`window_end`
**Source:** every file in the classification table above independently references the old field names; there is no single shared helper to centralize this (D-01's hard cutover means each of the 7 non-test modules + notebook must be edited directly).
**Apply to:** `models.py`, `campaign_utils.py`, `campaign_gap.py`, `campaign_views.py`, `campaign_tables.py`, `campaign_forms.py`, `import_campaign_csv.py`, the demo notebook, and all 6 test files.
**Rule:** `window_start == window_end` represents a single night (the only shape this phase produces); both `None` represents TBD. Never reintroduce a separate `obs_date`-shaped field.

### "Never raise for expected messy data" discipline
**Source:** `solsys_code/campaign_utils.py::resolve_site()` (lines 85-183) and `solsys_code/campaign_gap.py::observable_dates()` (lines 109-135) — the established codebase-wide convention (also documented in each module's own docstring) for handling site-resolution/ephemeris edge cases without crashing a batch import or view request.
**Apply to:** the new `sun_event()` call site in `campaign_views.py`'s D-06 branch (Pitfall 7 — must `except ValueError: log+skip`, never let it propagate into the existing broad `except Exception:` at line 324, which exists for a different purpose).

### Dict-vs-model dual accessor (django-tables2)
**Source:** `solsys_code/campaign_tables.py::render_site()` (lines 111-137), `render_run_status()`/`render_approval_status()` (lines 84-109) — `Accessor('field').resolve(record, quiet=True)`.
**Apply to:** the new `render_window_start()` method (must work identically whether `record` is a `CampaignRun` instance (staff) or a `.values()` dict (non-staff, per `ALLOWED_FIELDS_FOR_NON_STAFF`)).

### No-churn create-or-update contract
**Source:** `solsys_code/campaign_utils.py::insert_or_create_campaign_run()` (lines 277-310) — `get_or_create(**lookup, defaults=fields)`, diff-and-`save(update_fields=...)` only if changed, else leave untouched.
**Apply to:** unchanged logic-wise this phase — only the `lookup` dict's keys change (see `import_campaign_csv.py` above); the function itself needs no edits.

### Explicit enumerated allowlist (never introspected) for PII gating
**Source:** `solsys_code/campaign_views.py::ALLOWED_FIELDS_FOR_NON_STAFF` (lines 49-67) — a security control documented in RESEARCH.md's Security Domain section (V4 Access Control, Information Disclosure threat pattern).
**Apply to:** must keep the same *shape* (explicit list, not `CampaignRun._meta` introspection) when swapping in `window_start`/`window_end` for the 3 old field names.

## No Analog Found

None — this phase modifies existing files in place; every file has itself (its prior state) as the baseline to diff against, plus in-repo precedent for each new mechanism identified above (partial `UniqueConstraint`, `RunPython` data migration, nulls-last sort) documented directly in RESEARCH.md's Architecture Patterns section (already verified against Django 5.2 docs where no in-repo precedent existed, e.g. the `RunPython` backfill/dedup shape, which is genuinely new to this app).

## Metadata

**Analog search scope:** `solsys_code/` (models, views, forms, tables, utils, gap, management/commands), `solsys_code/migrations/`, `solsys_code/solsys_code_observatory/models.py`, `solsys_code/telescope_runs.py`, `docs/notebooks/pre_executed/`.
**Files scanned:** `models.py`, `campaign_utils.py`, `campaign_gap.py`, `campaign_views.py`, `campaign_tables.py`, `campaign_forms.py`, `management/commands/import_campaign_csv.py`, `migrations/0002_campaignrun.py`, `migrations/0003_campaignrun_natural_key_unique_constraint.py`, `solsys_code_observatory/models.py` (`SATELLITE_OBSTYPE`/`observations_type`), `telescope_runs.py` (`sun_event`/`get_site` signatures).
**Pattern extraction date:** 2026-07-09
