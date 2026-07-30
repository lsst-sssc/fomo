---
phase: 27-the-canonical-run-record
reviewed: 2026-07-30T14:18:07Z
depth: deep
files_reviewed: 28
files_reviewed_list:
  - docs/design/canonical_record_spike.rst
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/calendar_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/backfill_lco_observation_records.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/management/commands/repair_stale_campaign_run_sites.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/migrations/0008_rename_calendareventtelescopelabel_calendareventmeta.py
  - solsys_code/migrations/0009_calendareventmeta_run.py
  - solsys_code/migrations/0010_campaignrun_source_telescope_class_campaignrunobservation.py
  - solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py
  - solsys_code/models.py
  - solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py
  - solsys_code/solsys_code_observatory/tests/test_timezone_backfill_migration.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_calendar_template.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_run_observation.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_canonical_record_migration.py
  - solsys_code/tests/test_import_campaign_csv.py
  - solsys_code/tests/test_load_telescope_runs.py
  - solsys_code/tests/test_repair_stale_campaign_run_sites.py
  - src/templates/tom_calendar/partials/event_form.html
findings:
  critical: 2
  warning: 9
  info: 6
  total: 17
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-07-30T14:18:07Z
**Depth:** deep
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Phase 27 adds `CampaignRun.source`/`telescope_class`, the `CampaignRunObservation` link
model, renames `CalendarEventTelescopeLabel` -> `CalendarEventMeta` (plus a `run` FK), two
admin inlines with attribution stamping, a calendar-modal template override, a one-off
site-repair command, and three data migrations.

Verification actually run during this review (not assumed):

- `python manage.py test solsys_code.tests.test_calendar_utils
  solsys_code.tests.test_campaign_run_observation solsys_code.tests.test_admin
  solsys_code.tests.test_canonical_record_migration
  solsys_code.solsys_code_observatory.tests.test_timezone_backfill_migration
  solsys_code.tests.test_repair_stale_campaign_run_sites` -> 66 tests, OK.
- `python manage.py test solsys_code.tests.test_campaign_views
  solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_submission
  solsys_code.tests.test_import_campaign_csv
  solsys_code.tests.test_sync_lco_observation_calendar` -> 186 tests, OK.
- `ruff check .` -> 1 pre-existing D103 in an unrelated notebook.
  `ruff format --check .` -> 7 files would be reformatted, one of which
  (`import_campaign_csv_demo.ipynb`) is in this changeset; confirmed pre-existing against the
  base commit.
- `src/templates/tom_calendar/partials/event_form.html` was byte-diffed against the installed
  tomtoolkit 3.0.0a9 partial: it is an exact copy plus the inserted block, with no accidental
  edits to the upstream form. Good.

Tests passing is not evidence of correctness here. Both blockers are exactly the cases the
suite does not exercise: a cross-command sequence (`telescope_class` outliving site
resolution) and a doc artifact no test executes. The dominant theme across the warnings is
**write paths added without the matching read/clear paths**: `telescope_class` has three
writers and no clearer, `CalendarEventMeta.run` has one reader (the new template) and no
production writer, and `telescope_class` was added to the non-staff field allowlist but is in
no rendered table column.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `telescope_class` is never cleared when a run's site later resolves, silently violating the model's own invariant

**File:** `solsys_code/campaign_views.py:547-571`, `solsys_code/campaign_views.py:660-707`,
`solsys_code/management/commands/repair_stale_campaign_run_sites.py:189-199`,
`solsys_code/models.py:201-210`

**Issue:** `CampaignRun.telescope_class` is documented as mutually exclusive with a resolved
site: *"telescope_class is never inferred for a run whose site DID resolve"*
(`models.py:203`), and both writers enforce it — migration 0011 filters `site__isnull=True`
(`0011_backfill_campaignrun_telescope_class.py:45`), and `import_campaign_csv.py:205-207`
writes `''` when `site is not None`. **No code path clears the field when a site resolves
after the fact**, and there are three such paths:

1. `CampaignRunDecisionView.post()` approve branch (`campaign_views.py:569-571`) —
   `run.save(update_fields=['site', 'site_needs_review'])`.
2. `CampaignRunDecisionView._resolve_site()` (`campaign_views.py:696-701`) —
   `.update(site=site)` only.
3. `repair_stale_campaign_run_sites` (`repair_stale_campaign_run_sites.py:195-199`) —
   `update_fields = ['site', 'site_needs_review']`.

Fully reachable with no migration involved: a CSV row whose `Site Code` misses (MPC network
down at import time) is stored with `site=None, site_needs_review=True,
telescope_class='1m0'` (`import_campaign_csv.py:205`). Staff then resolve it from the approval
queue's "Sites Needing Review" card, and the row ends up with **both** a real `Observatory`
**and** `telescope_class='1m0'` — which is exactly the ambiguity CANON-02 exists to remove.
Admin `list_filter=['...', 'telescope_class']` (`admin.py:55`, D-19: "how staff find
class-wide runs") now returns site-resolved runs, and Phase 28/29 readers inherit a persisted
contradiction. Neither `test_campaign_views.py` nor `test_repair_stale_campaign_run_sites.py`
covers the resolve-after-derive sequence.

**Fix:** clear the field in every site-resolution path, and add a regression test for the
resolve-after-derive sequence.

```python
# campaign_views.py, approve branch (~line 569)
site, needs_review = resolve_site(obscode_selection, create_placeholder=False)
run.site, run.site_needs_review = site, needs_review
if site is not None:
    # CANON-02: a site-resolved run must never keep a class-wide allocation.
    run.telescope_class = ''
run.save(update_fields=['site', 'site_needs_review', 'telescope_class'])

# campaign_views.py, _resolve_site() conditional claim (~line 696)
claimed = CampaignRun.objects.filter(
    pk=pk,
    approval_status=CampaignRun.ApprovalStatus.APPROVED,
    site_needs_review=True,
    site_id=previous_site_id,
).update(site=site, telescope_class='')

# repair_stale_campaign_run_sites.py (~line 193)
run.site = site
run.site_needs_review = needs_review
update_fields = ['site', 'site_needs_review']
if site is not None and run.telescope_class:
    run.telescope_class = ''
    update_fields.append('telescope_class')
```

### CR-02: the rename broke `sync_lco_observation_calendar_demo.ipynb` — the paired demo notebook now contains code that raises `ImportError`

**File:** `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb:1778`
(also 1782, 1834, 1897; stale prose at 1032, 1236, 1357, 1383, 1625, 1633, 1733, 1749, 1854,
1911, 1952, 1962)

**Issue:** This changeset renames `CalendarEventTelescopeLabel` -> `CalendarEventMeta`
(`solsys_code/migrations/0008_...`, `solsys_code/models.py:10`) and un-privatises five
`calendar_utils` helpers (`_extract_instrument` -> `extract_instrument`, etc.). The paired
notebook for the very module edited by this changeset
(`solsys_code/management/commands/sync_lco_observation_calendar.py` ->
`sync_lco_observation_calendar_demo.ipynb`, per CLAUDE.md's pairing map) was **not** updated.
It still contains an executable cell:

```python
from solsys_code.models import CalendarEventTelescopeLabel   # ImportError today
verified_label = CalendarEventTelescopeLabel.objects.get(event=verified_event)
```

plus `except CalendarEventTelescopeLabel.DoesNotExist:` and a dozen prose references to
function names that no longer exist. It ships with stale pre-executed output claiming PASS
for assertions that can no longer run. The Sphinx build does not catch this
(`docs/conf.py:69` sets `nbsphinx_allow_errors = True`, and `nbsphinx_execute` defaults to
`auto`, which skips notebooks that already carry output), so the breakage is latent until
someone runs `jupyter nbconvert --to notebook --execute --inplace` — at which point the
documented regeneration procedure fails.

CLAUDE.md makes the paired notebook part of the deliverable, not follow-up polish, and names
notebook-scope misses as this project's breach history (Phases 5 and 6). This is the same
class of miss, and here it leaves committed code that cannot run.

**Fix:** update the notebook's model/function names and re-execute it.

```bash
# in the notebook: CalendarEventTelescopeLabel -> CalendarEventMeta,
# _extract_instrument -> extract_instrument, _resolve_placement_block -> resolve_placement_block,
# _derive_telescope -> derive_telescope, _coarse_telescope_label -> coarse_telescope_label
jupyter nbconvert --to notebook --execute --inplace \
  docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
```

## Warnings

### WR-01: CSV re-import unconditionally rewrites `source`, erasing WEB provenance on a colliding public submission

**File:** `solsys_code/management/commands/import_campaign_csv.py:195-200`,
`solsys_code/campaign_utils.py:849-857`

**Issue:** `fields['source'] = CSV_IMPORT` is applied on *update* as well as create —
`insert_or_create_campaign_run()` `setattr`s every key in `fields` onto a matched row
(`campaign_utils.py:852-856`). A `CampaignRun` created by the public form with
`source=WEB, approval_status=PENDING_REVIEW` (`campaign_views.py:257`) that shares the natural
key `(campaign, telescope_instrument, window_start, window_end)` with a CSV row — entirely
plausible, since the sheet and the form describe the same runs — is silently rewritten to
`source=CSV_IMPORT, approval_status=APPROVED`.

The `approval_status` overwrite predates this phase. What this phase adds is the `source`
overwrite, which destroys the one signal that would let staff *detect* it: per the derivation
rule introduced here (`models.py:90-94`), `APPROVED + source != WEB` reads as "no approval was
required". After the re-import, an unreviewed public submission is indistinguishable from
vetted backfill and is publicly visible (`campaign_views.py:130` only excludes
`PENDING_REVIEW`). The handler docstring (`import_campaign_csv.py:73-78`) documents the
analogous `target`-reset hazard but says nothing about `source`, and the runbook note added at
`docs/runbooks/telescope_runs_calendar.rst:188-197` does not mention it either.

**Fix:** preserve WEB provenance on update, and document the behaviour.

```python
# import_campaign_csv.py, before insert_or_create_campaign_run(lookup, fields)
existing = CampaignRun.objects.filter(**lookup).first()
if existing is not None and existing.source == CampaignRun.Source.WEB:
    # CANON-01: never relabel a run that came in through the public form -- the
    # APPROVED + source != WEB derivation rule depends on WEB surviving a re-import.
    fields.pop('source', None)
    fields.pop('approval_status', None)
```

### WR-02: `telescope_class` was added to the non-staff field allowlist but is rendered by no table column — D-18's stated outcome is not delivered

**File:** `solsys_code/campaign_views.py:80-83`, `solsys_code/campaign_tables.py:57-72`,
`solsys_code/tests/test_campaign_views.py:369-385`

**Issue:** `'telescope_class'` was added to `ALLOWED_FIELDS_FOR_NON_STAFF` so non-staff
readers can "distinguish a legitimately class-wide run from a site that failed to resolve"
(`campaign_views.py:80-82`). But `CampaignRunTable.Meta.fields` (`campaign_tables.py:57-72`)
does not include `telescope_class`, and no template renders it — `grep -rn telescope_class
src/` returns nothing. The field is fetched into the `.values()` queryset and then discarded,
for staff and non-staff alike. The new test asserts only that `view.get_queryset()` contains
the key (`test_campaign_views.py:376-385`), never that a reader can see it, so it passes while
the user-visible behaviour is unchanged. The test's own comment ("campaign_tables.py's
rendered-column Meta.fields tuple is out of this plan's scope") documents the gap rather than
closing it.

**Fix:** either render the column, or drop the inert allowlist entry.

```python
# campaign_tables.py
class Meta:
    fields = (
        'telescope_instrument',
        'site',
        'telescope_class',   # CANON-02: class-wide allocation vs. unresolved site
        'window_start',
        ...
    )
```
and assert against `response.content`, not the queryset.

### WR-03: `CalendarEventMeta.run` has no production writer — the new calendar-modal block can never appear from normal operation

**File:** `src/templates/tom_calendar/partials/event_form.html:100-126`,
`solsys_code/campaign_views.py:415-498`, `solsys_code/models.py:39-46`

**Issue:** `_project_calendar_event()` is the only code that creates `CalendarEvent`s for a
`CampaignRun` (`campaign_views.py:458`, `:497`) and it never writes a `CalendarEventMeta` row,
let alone sets `run`. `sync_lco_observation_calendar.py:369` creates meta rows but sets only
`is_verified`. A repo-wide grep confirms the only writers of `run` are `CalendarEventMetaInline`
in the admin (`admin.py:15`) and the tests. So the CANON-05 modal block ships with zero
automatic producers: for every event FOMO creates today, `event.telescope_label_meta.run` is
unset and the block renders nothing.

The design spike does defer the automatic writer to the Phase 29 reconciler
(`docs/design/canonical_record_spike.rst:158-162`), so the deferral is a decision — but
nothing in the runbook or the template tells an operator the "Campaign run" block only appears
if a staff member hand-links the event in Django admin, and the new test module
(`test_calendar_template.py:434-448`) hides this by creating the meta rows directly.

**Fix:** either populate the link where the event is created, or state the manual-only state in
the runbook.

```python
# campaign_views.py, _project_calendar_event(), after each insert_or_create_calendar_event()
event, _ = insert_or_create_calendar_event({'url': url}, fields=night_fields)
CalendarEventMeta.objects.update_or_create(event=event, defaults={'run': run})
```

### WR-04: `repair_stale_campaign_run_sites` counts a placeholder tier-1 hit as "resolved" while leaving the row flagged

**File:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py:201-208`

**Issue:** `resolve_site()` returns `(observatory, needs_review)` and deliberately returns
`needs_review=True` when the tier-1 match is itself a `NEEDS REVIEW: ` placeholder
(`campaign_utils.py:198-210`). The command's summary keys only on `site is not None`:

```python
if site is not None:
    resolved_count += 1
    logger.info('pk=%s: resolved to Observatory %r', run.pk, site.obscode)
else:
    still_flagged_count += 1
```

so a placeholder hit logs "resolved to Observatory 'XXX'" and increments `resolved`, while the
row it just wrote still carries `site_needs_review=True`. The operator-facing summary
("resolved: N, still_flagged: M") over-reports success and under-reports the work queue — for a
one-off data-repair command whose entire output *is* its operator contract, that matters.
Worse, the row is then excluded from any future run of this command (candidate filter is
`site__isnull=True`, lines 145-148), so the misleading count is this command's last word on it.

**Fix:**

```python
if site is not None and not needs_review:
    resolved_count += 1
    logger.info('pk=%s: resolved to Observatory %r', run.pk, site.obscode)
else:
    still_flagged_count += 1
    logger.info('pk=%s: still needs review (site=%r, site_raw=%r)', run.pk, site, site_raw)
```

### WR-05: a hardcoded, database-specific `site_raw` correction ships in a general-purpose command and mutates rows on any database

**File:** `solsys_code/management/commands/repair_stale_campaign_run_sites.py:52-56, 158-163,
196-199`

**Issue:** `_OWNER_SUPPLIED_SITE_RAW = {'swift': 'C52'}` is documented as domain authority for
"the one known stale Swift row (pk=13 on the dev DB)" that "does NOT generalise to another
database" — and is then applied unconditionally on every database the command runs against.
Any `CampaignRun` whose `telescope_instrument` first token is `swift` and whose `site_raw` is
blank has its `site_raw` **written** to `'C52'`, including a Swift/BAT row or a Swift ToO
reported under a different site. The docstring concedes the value does not generalise; nothing
in the code acts on that concession.

**Fix:** gate it behind an explicit opt-in flag so it cannot fire unnoticed on another
database.

```python
parser.add_argument(
    '--apply-owner-site-raw-corrections',
    action='store_true',
    help='D-16b: apply the operator-supplied site_raw corrections (dev-DB specific; off by default).',
)
...
if not site_raw and options['apply_owner_site_raw_corrections']:
    owner_value = _OWNER_SUPPLIED_SITE_RAW.get(_first_instrument_token(run.telescope_instrument))
```

### WR-06: the Observatory timezone data migration imports live application code, including the live `Observatory` model

**File:** `solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py:29`

**Issue:** `from solsys_code.solsys_code_observatory.utils import _get_timezone_finder` pulls in
a module that does `from solsys_code.solsys_code_observatory.models import Observatory`,
`import requests`, and `from tom_dataservices.dataservices import MissingDataException` at
module scope (`utils.py:1-7`). A data migration must stay replayable against a schema that no
longer matches the live model; importing the live model (and two third-party packages) from a
`RunPython` step is exactly the coupling migration 0011's own comment goes out of its way to
avoid (`0011_backfill_campaignrun_telescope_class.py:35-39`). It works today only because the
live model is never *used* — a single future import-time field reference, or a
`tom_dataservices` API change, breaks replay of an already-applied migration.

**Fix:** inline the two-line finder rather than importing the app module.

```python
def backfill_observatory_timezone(apps, schema_editor):
    from timezonefinder import TimezoneFinder

    Observatory = apps.get_model('solsys_code_observatory', 'Observatory')
    rows = Observatory.objects.filter(timezone='', lat__isnull=False, lon__isnull=False)
    if not rows.exists():
        return                      # also avoids the unconditional polygon load, IN-06
    finder = TimezoneFinder()
    for obs in rows:
        ...
```

### WR-07: `derive_telescope_class`'s function-local import does not achieve what its comment claims — migration 0011 imports the live `CampaignRun` anyway

**File:** `solsys_code/calendar_utils.py:175-184`,
`solsys_code/migrations/0011_backfill_campaignrun_telescope_class.py:35-48`

**Issue:** The comment at `calendar_utils.py:175-178` states the import is function-local so a
module-scope one "would drag the live CampaignRun model into calendar_utils' import graph,
which a data migration imports". But the import executes at **call** time, and migration 0011
calls `derive_telescope_class()` once per site-less row (`0011:46`). The first row whose
`site_raw` starts with `500@` — JWST/HST/JUICE rows are precisely the rows this migration
targets — triggers `from solsys_code.campaign_utils import HORIZONS_OBSERVER_TO_OBSCODE`,
which imports `solsys_code.models` (the live `CampaignRun`) *during the migration*. The stated
protection is not delivered; the deferral changes *when* the live model is imported, not
*whether*. `calendar_utils` also imports live models at module scope anyway
(`from tom_calendar.models import CalendarEvent`, line 16), so the migration was never isolated
from live models to begin with.

**Fix:** move the alias table to a model-free module (or freeze it as a literal in the
migration) and delete the misleading comment, so the "safe for migrations" claim becomes true.

```python
# solsys_code/observer_codes.py  (no Django model imports at all)
HORIZONS_OBSERVER_TO_OBSCODE: dict[str, str] = {...}
# campaign_utils.py and calendar_utils.py both import from here, at module scope.
```

### WR-08: `CalendarEventMetaInline` exposes the primary-key `event` field as editable — changing it inserts a duplicate row instead of moving it

**File:** `solsys_code/admin.py:8-17`, `solsys_code/models.py:24-30`

**Issue:** `CalendarEventMeta.event` is a `OneToOneField(..., primary_key=True)`. Django's
`BaseModelFormSet.add_fields()` injects a hidden `id` field only when `pk_is_not_editable(pk)`;
an explicitly declared `OneToOneField(primary_key=True)` is editable and is present in
`form.fields`, so **no hidden pk is added and `event` itself renders as an editable `<select>`**
on every existing inline row. A staff user who changes that select makes `instance.pk` a value
absent from the table; `instance.save()` (`admin.py:90`) issues `UPDATE ... WHERE id=<new pk>`
(0 rows) and falls back to `INSERT`, leaving the original row behind. Result: a duplicate
companion row plus an orphaned `is_verified`/`run` history — the exact history migration 0008's
header comment (`0008_...py:1-7`) was written to protect.

**Fix:** make the identity field non-editable on existing rows, so the only operations are add
and delete.

```python
class CalendarEventMetaInline(admin.TabularInline):
    model = CalendarEventMeta
    fk_name = 'run'
    extra = 0

    def get_readonly_fields(self, request, obj=None):
        # `event` is this model's primary key -- editing it on an existing row
        # INSERTs a second row and orphans the original.
        return ['event'] if obj else []
```

### WR-09: `derive_telescope_class` labels any unrecognised `500@…` string as `SPACE`, including geocentric codes and typos

**File:** `solsys_code/calendar_utils.py:172-184`

**Issue:** The tier-a branch is `site_raw.strip().startswith('500@')` plus "not in
`HORIZONS_OBSERVER_TO_OBSCODE`" -> `return 'SPACE'`. `500@<N>` is JPL Horizons *observer*
notation for "geocentric observer at body N", and body N need not be a spacecraft: `500@399` is
the Earth's centre, `500@10` the Sun, `500@301` the Moon. A mistyped NAIF ID (`'500@-999'` —
the exact string `resolve_site()` cites as unrecognised at `campaign_utils.py:193`) lands here
too. All are permanently recorded as `telescope_class='SPACE'`, i.e. "a space observatory with
a Horizons code but no MPC obscode" (`models.py:104-119`) — a stronger and different claim than
"we could not resolve this". The module's own extension rule for
`NO_OBSCODE_SPACE_OBSERVATORIES` (`calendar_utils.py:111-117`) says "only add a name after
verifying on BOTH sides ... never infer from the name alone"; this branch infers from string
shape alone.

**Fix:** restrict the branch to negative NAIF IDs (spacecraft) and leave everything else blank,
so `site_needs_review` carries the unresolved case as D-13 intends.

```python
if stripped_site.startswith('500@') and stripped_site not in HORIZONS_OBSERVER_TO_OBSCODE:
    naif = stripped_site[4:]
    # Negative NAIF IDs are spacecraft; positive ones are natural bodies
    # (500@399 = geocentre, 500@10 = Sun) and are not observatories at all.
    if naif.startswith('-') and naif[1:].isdigit():
        return 'SPACE'
```

## Info

### IN-01: the six-key counter dict is duplicated three times in `sync_lco_observation_calendar`

**File:** `solsys_code/management/commands/sync_lco_observation_calendar.py:289-306, 322-332`

**Issue:** The literal `{'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0,
'extraction_failed': 0, 'telescope_api_failed': 0}` appears three times. Adding a seventh
counter requires editing three places; missing one produces a `KeyError` on the defensive
unknown-facility path, which no test exercises.

**Fix:** `_COUNTER_KEYS = (...)` plus `def _new_counters(): return dict.fromkeys(_COUNTER_KEYS, 0)`,
used in all three places.

### IN-02: a test module imports a private helper from another test module

**File:** `solsys_code/tests/test_calendar_utils.py:22`

**Issue:** `from solsys_code.tests.test_sync_lco_observation_calendar import
_observations_block_response` couples two test modules: any import-time failure in
`test_sync_lco_observation_calendar` (a renamed facility import, a missing fixture) now also
fails `test_calendar_utils`.

**Fix:** move `_observations_block_response` into a shared non-test helper module (e.g.
`solsys_code/tests/helpers.py`) and import it from both.

### IN-03: `ruff format --check .` fails on `import_campaign_csv_demo.ipynb`

**File:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (cell 3, the multi-line
`assert`)

**Issue:** CLAUDE.md's project constraints require `ruff check .` and `ruff format --check .`
to stay clean. `ruff format --check .` reports 7 files, one of which is this changeset's
regenerated notebook. Confirmed pre-existing (the base commit's copy also fails), so not a
regression — but the phase regenerated and re-committed the file without fixing it.

**Fix:** `ruff format docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (and the other
listed files) in a follow-up.

### IN-04: `campaign_utils`' module docstring cites `_derive_telescope_class`, a name that never existed and now collides with the new public helper

**File:** `solsys_code/campaign_utils.py:8`

**Issue:** The docstring cites "the `_derive_telescope_class` precedent in
`calendar_utils.py`". No such symbol ever existed (the intended reference was
`_derive_telescope`, renamed to `derive_telescope` by this changeset), and this phase now
introduces a real, differently-named `derive_telescope_class`. A reader following the pointer
lands on the wrong function.

**Fix:** change the reference to `derive_telescope`.

### IN-05: the demo notebook's inspection table mixes `legacy` rows into an "every row is `csv_import`" narrative

**File:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (markdown cell 8, output
of code cell 9)

**Issue:** Cell 8 states "Every row imported by this command is `csv_import`", but cell 9 prints
`CampaignRun.objects.filter(campaign=campaign)`, which also picks up the two approval-lifecycle
rows created by cell 17 outside the importer, shown with `source='legacy'`. A reader can
reasonably conclude the importer sometimes writes `legacy`.

**Fix:** filter cell 9 to `source=CampaignRun.Source.CSV_IMPORT` for the claim table, or note in
prose that the two `legacy` rows are the notebook's own lifecycle demo rows.

### IN-06: the timezone backfill migration constructs `TimezoneFinder` even when there is nothing to backfill

**File:** `solsys_code/solsys_code_observatory/migrations/0003_backfill_observatory_timezone.py:32`

**Issue:** `finder = _get_timezone_finder()` runs before the queryset is evaluated, so every
`migrate` — including every test-database creation and every `TransactionTestCase`
re-migration in `test_canonical_record_migration.py` and `test_timezone_backfill_migration.py`
— loads timezonefinder's boundary-polygon data even on a database with zero matching rows.

**Fix:** construct the finder lazily, or return early on an empty queryset (WR-06's snippet
fixes both).

### IN-07: the runbook documents the `target` re-import reset but not the new `source`/`approval_status` reset

**File:** `docs/runbooks/telescope_runs_calendar.rst:188-197, 366-372`

**Issue:** The runbook's re-import warning covers only the `target` reset. The new note
describing what the importer writes ("every imported row records `source = csv_import` and is
created `approved`") is phrased for the create path and does not say a *re-import* applies the
same two values to an already-existing row (see WR-01).

**Fix:** extend the existing "re-import deliberately, not routinely" note to name `source` and
`approval_status` alongside `target`.

---

_Reviewed: 2026-07-30T14:18:07Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
