# Phase 20: Range/TBD Import & Asset-Aware Coverage Gap - Pattern Map

**Mapped:** 2026-07-10
**Files analyzed:** 7 (6 modified + 1 new migration)
**Analogs found:** 7 / 7 (all in-file, self-referential — this phase extends existing functions/patterns rather than introducing new files)

**Note on approach:** Every file in this phase is an existing file being extended, not a new
file. RESEARCH.md already contains direct, verified reads of the exact current bodies with
line numbers, so this pattern map cites those excerpts directly rather than re-reading files.
The "analog" for each change is the *sibling code already in the same file* — this phase's
whole design principle (per RESEARCH.md) is "extend an existing pattern-per-shape/bucketing/
flag convention," not introduce a new one.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `solsys_code/campaign_utils.py` (`parse_obs_window`) | utility (parser) | transform | same function's existing `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` pattern-per-shape regex block | exact (in-file) |
| `solsys_code/campaign_gap.py` (`claimed_dates`) | service (pure logic) | batch/transform | same function's existing `undated_runs` bucketing pattern | exact (in-file) |
| `solsys_code/management/commands/import_campaign_csv.py` (`Command.handle`) | CLI command | batch/file-I/O | same function's existing `seen_window_keys`/lookup-dict natural-key branching | exact (in-file) |
| `solsys_code/models.py` (`CampaignRun`) | model | CRUD | `site_needs_review` field (same model, lines 86-88) | exact (in-file) |
| `solsys_code/migrations/0006_*.py` | migration | schema | `solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py` (immediate predecessor, plain `AddField` shape) | exact |
| `solsys_code/campaign_tables.py` (`render_window_start`) | component (table render) | request-response | `render_site()` tooltip (same file, lines 126-134) | exact (in-file) |
| `src/templates/campaigns/campaignrun_gap_analysis.html` | template | request-response | existing `undated_runs`/`unattributed_runs` alert block (same template, lines 53-69) | exact (in-file) |
| `solsys_code/tests/test_import_campaign_csv.py` (edit 2 existing + add new) | test | — | existing `_WriteCsvMixin`/`_row()` helper pattern (same file) | exact (in-file) |
| `solsys_code/tests/test_campaign_gap.py` (add new tests) | test | — | `TestClaimedDates._make_run` helper (same file, lines 154-163) + space-`Observatory` fixture from `test_campaign_approval.py` (lines 200-206) | exact |
| `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (discretionary update) | notebook | file-I/O | its own existing structure/fixture CSV | exact (in-file) |

No files in this phase lack an analog — every change is an extension of an already-established
in-file (or same-app) convention. "No Analog Found" section omitted.

## Pattern Assignments

### `solsys_code/campaign_utils.py` — `parse_obs_window()` (D-11/D-12/D-13)

**Analog:** the function's own existing `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` regex
cascade for `UT Time Range` (lines ~139-156), which already implements "try each shape in
order, never raise, fall through to a flagged default."

**Current full body** (RESEARCH.md "Current-State Confirmation", verbatim):
```python
_HHMM_RANGE = re.compile(r'(\d{1,2})[:;](\d{2})\s*(am|pm)?\s*-\s*(\d{1,2})[:;](\d{2})\s*(am|pm)?', re.IGNORECASE)
_APPROX_HOUR = re.compile(r'~\s*(\d{1,2})(?::\d{2})?(?::\d{2})?\s*(am|pm)?', re.IGNORECASE)
_BARE_HOUR_UTC = re.compile(r'(\d{1,2})\s*UTC\b', re.IGNORECASE)


def parse_obs_window(obs_date_raw: str, ut_range_raw: str) -> tuple[date, datetime, datetime | None, bool]:
    obs_date = datetime.strptime((obs_date_raw or '').strip(), '%Y-%m-%d').date()  # ValueError propagates

    match = _HHMM_RANGE.search(ut_range_raw or '')
    if match:
        ...  # unchanged UT-time-range parsing, three shapes, never raises
    match = _APPROX_HOUR.search(ut_range_raw or ''):
        ...
    match = _BARE_HOUR_UTC.search(ut_range_raw or ''):
        ...
    # Fallback: obs_date valid, UT range not parseable -- midnight UTC, ut_needs_review=True
    start = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, tzinfo=dt_timezone.utc)
    return obs_date, start, None, True
```

**New module-level regexes to add** (RESEARCH.md "Concrete D-11/D-12/D-13 Parsing Design",
verbatim, place alongside `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`):
```python
# D-12: full-date range, en-dash/em-dash/hyphen or literal "to" separator. Anchored
# start-to-end so it never partially matches inside a longer garbage string (matches this
# module's existing narrowly-scoped-regex discipline, RESEARCH Anti-Patterns precedent).
_DATE_RANGE_FULL = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s*(?:to|[-–—])\s*(\d{4}-\d{2}-\d{2})$',
    re.IGNORECASE,
)

# D-11: compact same-month/rollover range, e.g. "2025-11-02 -25" or "2025-11-28 -05".
# Second group is the day-of-month only (1-2 digits); rollover logic lives in the caller,
# not the regex, per this module's existing convention (see _HHMM_RANGE, which also keeps
# am/pm interpretation out of the pattern itself).
_DATE_RANGE_COMPACT = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\s*-\s*(\d{1,2})$')
```

Use `.match()`/`.fullmatch()` after `.strip()`, never `.search()` — `Obs. Date` is a structured
cell, not free prose (unlike the UT-time regexes above, which correctly use `.search()`).

**Order-of-attempts / control flow** (copy directly, RESEARCH.md):
1. Exact date via existing `strptime`, now wrapped `try/except ValueError` instead of letting
   it propagate.
2. `_DATE_RANGE_FULL` match, each side independently `try/except`-guarded.
3. `_DATE_RANGE_COMPACT` match -> rollover algorithm (below).
4. No match at all -> TBD: `window_start = window_end = None`, `original_obs_date_raw = text`,
   `window_needs_review = True`. (No separate `YYYY-MM-?` regex needed — it falls through here
   naturally.)

**D-11 rollover algorithm** (copy directly, RESEARCH.md verbatim):
```python
match = _DATE_RANGE_COMPACT.match(text)
if match:
    year_s, month_s, day1_s, day2_s = match.groups()
    year, month, day1, day2 = int(year_s), int(month_s), int(day1_s), int(day2_s)
    try:
        window_start = date(year, month, day1)
        if day2 < day1:
            # D-11 rollover: second number is smaller than the first day-of-month --
            # roll into the next month (and next year for a Dec -> Jan crossing).
            if month == 12:
                window_end = date(year + 1, 1, day2)
            else:
                window_end = date(year, month + 1, day2)
        else:
            window_end = date(year, month, day2)
    except ValueError:
        # e.g. day2=35 for a 28/29/30/31-day month, or day1 itself invalid -- stdlib
        # date() already validates this; treat exactly like any other unparseable shape.
        window_start = window_end = None
```

**New return signature** (RESEARCH.md, extends not replaces):
```python
def parse_obs_window(
    obs_date_raw: str, ut_range_raw: str
) -> tuple[date | None, date | None, str, bool, datetime | None, datetime | None, bool]:
    """Returns (window_start, window_end, original_obs_date_raw, window_needs_review,
    ut_start, ut_end, ut_needs_review). Never raises.
    """
```
When the row resolves to range or TBD, skip UT-Time-Range parsing entirely — set
`ut_start = ut_end = None`, `ut_needs_review = False` (A1 in Assumptions Log; zero observable
effect since nothing persists these fields today).

**Error handling pattern:** never raise — every branch either produces a resolved
`window_start`/`window_end` pair or falls through to the TBD tuple. This is a strict widening
of the existing "never raise on UT Time Range, flag needs-review" discipline to `Obs. Date`
itself (D-13).

---

### `solsys_code/campaign_gap.py` — `claimed_dates()` (D-09/D-10, ASSET-01/02)

**Analog:** the function's own existing `undated_runs` bucketing loop (RESEARCH.md verbatim):
```python
claimed: set[date] = set()
undated_runs: list[CampaignRun] = []
for run in qs:
    if run.window_start is None or run.window_end is None:
        undated_runs.append(run)
        continue
    n_days = (run.window_end - run.window_start).days + 1
    for i in range(n_days):
        claimed.add(run.window_start + timedelta(days=i))

return claimed, undated_runs, unattributed_runs
```

**Pattern to copy for the new bucket:** add `pending_narrowing_runs: list[CampaignRun] = []`
alongside `undated_runs`, same declare-list/append/continue shape. Do **not** add a per-row
`run.site.observations_type` check inside the loop (Pitfall 3) — the queryset is already
filtered to a single `site` (the function's own parameter), so compute
`is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE` **once, before
the loop**, exactly as the ground-vs-space precedent below does at the call site.

**Ground-vs-space check to reuse verbatim** (exact precedent, `campaign_views.py:339`, Phase
19 D-06):
```python
if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
    # space-based observatory: no fixed horizon
    ...
```
(Adapt to the pre-computed `is_space_mission` bool inside `claimed_dates()`, not `run.site` —
avoids widening the documented `.only('pk', 'window_start', 'window_end')` PII-minimization
restriction, per Pitfall 3.)

**New bucketing logic (D-09), sketched from the existing shape:**
```python
claimed: set[date] = set()
undated_runs: list[CampaignRun] = []
pending_narrowing_runs: list[CampaignRun] = []
is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE
for run in qs:
    if run.window_start is None or run.window_end is None:
        undated_runs.append(run)
        continue
    if is_space_mission and run.window_start != run.window_end:
        pending_narrowing_runs.append(run)
        continue
    n_days = (run.window_end - run.window_start).days + 1
    for i in range(n_days):
        claimed.add(run.window_start + timedelta(days=i))

return claimed, undated_runs, unattributed_runs, pending_narrowing_runs
```

**Downstream wiring:** `_compute_gap()` (lines 190-224) unpacks the 3-tuple today and builds
the returned dict literal at lines 217-224 — add `'pending_narrowing_runs': pending_narrowing_runs`
there. `get_or_compute_gap()` needs **no change** (opaque dict cache/passthrough).
`campaign_views.py` needs **no change** (Pitfall 4 — `context.update({'result': result})`
already passes the whole dict; template accesses new key via normal dict-key dotted lookup).

---

### `solsys_code/management/commands/import_campaign_csv.py` — `Command.handle()` (D-06/D-07, Pitfall 1/2)

**Analog:** the function's own existing `seen_window_keys` collision-set + lookup-dict
construction (lines 89, 131-142, 169-182).

**Remove:** the `except ValueError as exc: ...; skipped_count += 1; continue` branch around
`parse_obs_window(...)` (lines 108-126) — becomes dead code once `parse_obs_window` never
raises.

**Add:** `window_needs_review` counter, incremented whenever a row lands as TBD via the new
path (D-05).

**Branch lookup/collision-key shape on resolved-vs-TBD** (copy directly, RESEARCH.md Pitfall 2,
matches `CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s exactly):
```python
if window_start is not None:
    collision_key = (campaign.pk, telescope_instrument, window_start, window_end)
    lookup = {
        'campaign': campaign, 'telescope_instrument': telescope_instrument,
        'window_start': window_start, 'window_end': window_end,
    }
else:
    contact_person = row.get('Contact Person', '') or ''
    collision_key = (campaign.pk, telescope_instrument, contact_person)
    lookup = {
        'campaign': campaign, 'telescope_instrument': telescope_instrument,
        'contact_person': contact_person,
    }
```
Pull `contact_person` out of the unconditional `fields` dict when it's promoted into `lookup`
for the TBD branch, to avoid lookup/defaults key overlap ambiguity (Pitfall 2, final paragraph).

**`fields` dict addition:** pass `original_obs_date_raw` and `window_needs_review` into the
`fields` dict for `insert_or_create_campaign_run`, mirroring how `site_needs_review`/`site_raw`
are already passed through in the same command (grep the existing call for the exact key names
used for the site-side pair, and mirror that spelling for the window-side pair).

---

### `solsys_code/models.py` — `CampaignRun` (D-01/D-02)

**Analog:** `site_needs_review` field, exact precedent (RESEARCH.md verbatim, `models.py:86-88`):
```python
site_needs_review = models.BooleanField(
    default=False, verbose_name='Whether the site could not be automatically resolved and needs manual review'
)
```

**New fields to add**, same style:
```python
original_obs_date_raw = models.CharField(
    max_length=255, blank=True, default='', verbose_name='Original Obs. Date text (TBD rows only)'
)
window_needs_review = models.BooleanField(
    default=False,
    verbose_name='Whether the observing window could not be automatically resolved and needs manual review',
)
```
Field-type choice (`CharField(max_length=255)`, not `TextField`) mirrors `site_raw`'s exact
precedent per RESEARCH.md's "Field-type recommendation" (A2 in Assumptions Log).

---

### `solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py`

**Analog:** `solsys_code/migrations/0005_campaignrun_campaign_run_window_start_end_null_together.py`
(immediate predecessor; dependency chain target) and, structurally, the plain-`AddField`-with-
default shape used for `site_needs_review`'s original addition (no `RunPython`, no backfill —
both new columns are TBD-only with safe defaults).

**Full recommended migration** (RESEARCH.md verbatim):
```python
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0005_campaignrun_campaign_run_window_start_end_null_together'),
    ]

    operations = [
        migrations.AddField(
            model_name='campaignrun',
            name='original_obs_date_raw',
            field=models.CharField(
                blank=True, default='', max_length=255, verbose_name='Original Obs. Date text (TBD rows only)'
            ),
        ),
        migrations.AddField(
            model_name='campaignrun',
            name='window_needs_review',
            field=models.BooleanField(
                default=False,
                verbose_name='Whether the observing window could not be automatically resolved and needs manual review',
            ),
        ),
    ]
```
No `RunPython`, no `UniqueConstraint`/`CheckConstraint` change — neither field participates in
any existing constraint.

---

### `solsys_code/campaign_tables.py` — `render_window_start()` (D-08)

**Analog:** `render_site()`'s existing tooltip pattern, exact precedent (RESEARCH.md verbatim,
`campaign_tables.py:126-134`):
```python
if Accessor('site_needs_review').resolve(record, quiet=True):
    return format_html(
        '<span class="text-muted font-italic" title="Site could not be automatically resolved">'
        '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i> {}</span>',
        site_raw,
    )
```

**Target location:** `render_window_start()`'s existing `start is None` (TBD) branch — the
existing TBD badge (`<span class="badge badge-secondary">TBD</span>`) gains a `title=` attribute
showing `original_obs_date_raw` when present, using the same `format_html(...)` construction
style as `render_site()` above (attribute value comes from `record.original_obs_date_raw`, guard
for empty string so the `title` attribute is omitted rather than empty when nothing was
captured). No change needed to the `if start == end: return start` single-night branch
(Pitfall 5) — that path is orthogonal to the TBD badge.

---

### `src/templates/campaigns/campaignrun_gap_analysis.html` (D-09)

**Analog:** the existing `undated_runs`/`unattributed_runs` alert block (RESEARCH.md,
confirmed structure — single `{% if result.undated_runs or result.unattributed_runs %}`
alert-warning `<div>` with one `<h5>`+`<p>` pair per non-empty bucket, lines 53-69).

**Pattern to copy:** add a third conditional inside the same alert-warning div (or a sibling
block — tone/placement is discretionary per CONTEXT.md), following the identical
`{% if result.pending_narrowing_runs %}<h5>...</h5><p>...</p>{% endif %}` shape, with wording
along the lines of "N space-mission run(s) haven't narrowed to a specific night yet and aren't
counted as claiming any date" (D-09's suggested copy). `context['result']` already carries this
key once `_compute_gap()`'s dict literal is updated — no view-level template-context change
needed.

---

### Tests

**`solsys_code/tests/test_import_campaign_csv.py`** — analog is the file's own existing
`_WriteCsvMixin`/`_row(**overrides)` helper pattern; reuse it for new range/TBD-shape fixture
rows rather than a new fixture file. Two existing tests must be *edited*, not just left alone
(Pitfall 1):
- `test_parse_obs_window_unparseable_date_raises` (line 270-272) — rewrite to assert the new
  TBD-tuple contract (`window_start is None`, `window_needs_review is True`,
  `original_obs_date_raw == ''`) instead of `assertRaises(ValueError)`.
- `test_natural_key_failure_skipped_and_logged` (line 470-502) — replace its
  `'Obs. Date': '2025-11-02 -25'` fixture (now validly parseable as a range) with a genuine
  non-date natural-key failure, e.g. blank `Telescope / Instrument`.

**`solsys_code/tests/test_campaign_gap.py`** — analog is `TestClaimedDates._make_run` helper
(lines 154-163) for `CampaignRun` fixtures, extended with new `**kwargs` rather than a parallel
factory; and the space-`Observatory` fixture from `test_campaign_approval.py` (lines 200-206,
exact code to reuse):
```python
space_site = Observatory.objects.create(
    obscode='250', name='Test Space Telescope', short_name='TST',
    observations_type=Observatory.SATELLITE_OBSTYPE,
)
```

**Target factory reminder (CLAUDE.md):** every new/edited test touching `Target` must use
`tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory` — both
existing target test files already follow this; new tests must match.

---

## Shared Patterns

### "Never raise, flag needs-review" discipline
**Source:** `solsys_code/campaign_utils.py`'s existing `_HHMM_RANGE`/`_APPROX_HOUR`/
`_BARE_HOUR_UTC` UT-Time-Range fallback (unchanged this phase).
**Apply to:** `parse_obs_window()`'s `Obs. Date` side (D-13) — the whole phase's parsing design
is a direct widening of this same posture to a second field.

### Boolean "needs review" flag field
**Source:** `site_needs_review` (`solsys_code/models.py:86-88`).
**Apply to:** `window_needs_review` (D-02) — identical `BooleanField(default=False,
verbose_name=...)` shape.

### Bucketed non-claiming run lists
**Source:** `claimed_dates()`'s existing `undated_runs` list (`solsys_code/campaign_gap.py`).
**Apply to:** `pending_narrowing_runs` (D-09) — identical declare/append/continue shape, third
instance of the same pattern.

### Table-cell tooltip via `format_html`
**Source:** `render_site()` (`solsys_code/campaign_tables.py:126-134`).
**Apply to:** `render_window_start()`'s TBD badge (D-08).

### Resolved-vs-TBD natural-key branching
**Source:** `CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s
(`unique_campaign_run_resolved_window` vs. `unique_campaign_run_tbd_natural_key`,
`solsys_code/models.py:113-153`, already established by Phase 19).
**Apply to:** `import_campaign_csv.py`'s `seen_window_keys`/lookup-dict construction (Pitfall 2)
— must branch on `window_start is not None` to match the two constraint shapes exactly.

## No Analog Found

None — every file in this phase's scope is an extension of an existing in-file or same-app
convention (see note under File Classification above).

## Metadata

**Analog search scope:** `solsys_code/` (campaign_utils.py, campaign_gap.py, models.py,
campaign_tables.py, campaign_views.py, management/commands/import_campaign_csv.py, migrations/,
tests/), `src/templates/campaigns/`, `docs/notebooks/pre_executed/`.
**Files scanned:** 7 target files + 2 test files + 1 template + 1 notebook (all pre-read and
verified directly by RESEARCH.md this same session — no additional codebase reads were needed
for this pattern map since every excerpt required was already captured with exact line numbers).
**Pattern extraction date:** 2026-07-10
