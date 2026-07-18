# Phase 20: Range/TBD Import & Asset-Aware Coverage Gap - Research

**Researched:** 2026-07-10
**Domain:** Django ORM model/migration extension, CSV import parsing (regex-based, no new
libraries), and pure-logic coverage-gap analysis, all internal to this codebase.
**Confidence:** HIGH (every claim below is a direct read of the real current source in this
repo, `[VERIFIED: codebase]`, except the D-11/D-12 regex design itself, which is original
reasoning built on stdlib-only primitives and validated against the codebase's existing
pattern-per-shape convention)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01/D-02/D-03/D-04:** Add `original_obs_date_raw` (text field, TBD-rows-only, empty
  for successfully-parsed single-date/range rows) and `window_needs_review` (boolean,
  mirrors `site_needs_review`) to `CampaignRun`. Both "blank `Obs. Date`" and
  `"YYYY-MM-?"` collapse to the same TBD state (`window_start == window_end == None`) —
  no separate sub-status; `original_obs_date_raw` alone distinguishes them.
- **D-05/D-06/D-07:** Only one new import-summary counter, `window_needs_review` (no
  separate range counter — ranges count as ordinary created/updated rows). Any `Obs. Date`
  text outside the enumerated shapes (blank / `" to "` range / compact range / `YYYY-MM-?`
  marker) now imports as a TBD row with `window_needs_review=True`, **never skipped** —
  this takes precedence over the Phase 18 spike's narrower "raise only on truly malformed
  values" suggestion. `skipped_count` is now reserved for non-date failures only (blank
  `Telescope / Instrument`, natural-key collisions).
- **D-08:** `original_obs_date_raw` is surfaced in `CampaignRunTable.render_window_start()`'s
  existing TBD badge via a `title` attribute when present.
- **D-09/D-10:** `claimed_dates()` gets a new, distinct bucket `pending_narrowing_runs` for
  space-mission runs whose window hasn't narrowed to one concrete night (still a range, or
  TBD) — separate from `undated_runs` (reserved for genuinely-TBD rows regardless of site
  type). No automated narrowing mechanism — only a staff edit or CSV re-import that sets
  `window_start == window_end` moves a run out of this bucket.
- **D-11:** The compact same-month range shape (`"2025-11-02 -25"`) is extended to detect
  month/year rollover: if the parsed second-day number is less than the first date's
  day-of-month, roll `window_end` into the next month (and next year for a
  December→January rollover). No real sheet row confirms this exact shape yet — implement
  as a generalization of the confirmed same-month rule, not a new from-scratch parser.
- **D-12:** The `" to "`-separated full-date range pattern also accepts en-dash- and
  hyphen-separated variants (e.g. `"2025-07-05–2025-09-22"`), not just the literal `" to "`
  string.
- **D-13:** `parse_obs_window()` no longer raises `ValueError` for `Obs. Date` failures. It
  always returns a tuple now — on anything unparseable, it returns
  `window_start = window_end = None`, the raw text, and `needs_review = True`. The
  UT-Time-Range-side fallback (`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`) is unchanged.

### Claude's Discretion

- Exact regex/parsing implementation for D-11/D-12 — this document's job (see below).
- Whether to extract a shared `is_space_mission(site)` helper vs. keeping the
  `Observatory.observations_type == SATELLITE_OBSTYPE` check inline at each call site — no
  behavioral difference.
- `original_obs_date_raw` field type/max length, and exact migration mechanics — standard
  technical choices (see Migration Mechanics below).
- Exact wording of the new `pending_narrowing_runs` alert block in
  `campaignrun_gap_analysis.html` beyond D-09's substance.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within Phase 20's IMPORT-01/02/ASSET-01/02 scope. Site
disambiguation and VIEW-05 remain Phase 21.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IMPORT-01 | `parse_obs_window`/`import_campaign_csv` accepts a date range or TBD-style free text and imports the row into the window representation, instead of skipping it | Concrete D-11/D-12 regex design below; current `parse_obs_window()` body confirmed (lines 186-244); `import_campaign_csv.py` lookup/collision-key branching design (Pitfall 1) |
| IMPORT-02 | A row whose `Obs. Date` text still can't be parsed gets a "needs review" flag and is included in the import summary, never silently dropped | D-13 contract change design; `window_needs_review` counter wiring in `Command.handle()` |
| ASSET-01 | Ground vs. space-mission classification derived from `Observatory.observations_type` (`SATELLITE_OBSTYPE`), no new `CampaignRun` field | Confirmed exact check shape and call site precedent (`campaign_views.py:339`, Phase 19 D-06) |
| ASSET-02 | Coverage-gap analysis claims every date in a ground run's window; a space-mission run claims nothing until narrowed to one night | Current `claimed_dates()` body confirmed (lines 115-187); concrete bucketing design below |

</phase_requirements>

## Summary

This phase's real technical work has two independent halves, both pure-Python/Django, with
**no new external dependency** in either: (1) extend `parse_obs_window()`'s existing
pattern-per-shape regex discipline with two more shapes (en-dash/hyphen full-date range,
and month/year-rollover compact range) plus a catch-all "never raise, return TBD" fallback,
and (2) add one `if`/branch to `claimed_dates()` reading the same
`Observatory.observations_type == SATELLITE_OBSTYPE` check Phase 19 already established at
`campaign_views.py:339`, producing a third bucket alongside the existing
`undated_runs`/`unattributed_runs`.

The current real source (read directly this session, not from CONTEXT.md's cited line
numbers, which are close but not exact — see Current-State Confirmation) shows both target
functions are simpler than they might sound: `claimed_dates()` is a ~35-line loop that
already has the exact bucketing shape D-09 wants a third instance of, and `parse_obs_window()`
is a ~60-line function whose *only* Obs.-Date-side logic today is a single
`strptime(...)` call wrapped in nothing (the `ValueError` propagates unguarded) — the entire
D-11/D-12/D-13 change is inserting 2-3 new regex branches ahead of that call and wrapping the
whole thing so nothing escapes as an exception.

The one finding that most changes planning is **not** in CONTEXT.md: two existing tests
(`test_natural_key_failure_skipped_and_logged` and
`test_parse_obs_window_unparseable_date_raises`) use exactly the input shapes this phase
newly makes *parseable* or *non-raising* — those tests' current assertions will fail under
the new contract and must be rewritten, not just left alone (see Pitfall 1 below). A second
finding: `import_campaign_csv.py`'s natural-key `lookup` dict and `seen_window_keys`
collision-detection tuple must branch on resolved-vs-TBD shape to match the model's two
partial `UniqueConstraint`s (Pitfall 2) — CONTEXT.md flags this as an integration point but
doesn't specify the branching logic, which is genuinely load-bearing (get it wrong and a
TBD-path re-import either silently duplicates rows or throws an uncaught `IntegrityError`).

**Primary recommendation:** extend `parse_obs_window()` with two new module-level compiled
regexes (`_DATE_RANGE_FULL`, `_DATE_RANGE_COMPACT`) tried in order after the existing exact-date
`strptime` attempt, wrap the whole Obs.-Date-parsing block in `try/except ValueError` so any
failure falls through to a single TBD return path, and branch `import_campaign_csv.py`'s
lookup/collision keys on `window_start is None` to match `CampaignRun.Meta.constraints`'s two
partial-constraint field sets exactly.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Obs.-Date range/TBD text parsing | API/Backend (`campaign_utils.py`, pure function) | — | Same tier as the existing UT-time parsing it extends; no view/request concerns |
| TBD-row persistence (`original_obs_date_raw`, `window_needs_review`) | Database/Storage (`CampaignRun` model + migration) | API/Backend (import command writes it) | New columns, no new table; write path is the bootstrap-import command |
| Import summary counters & CLI output | API/Backend (`Command.handle()`) | — | Management command, no web-facing surface |
| Ground-vs-space bucketing in coverage-gap | API/Backend (`campaign_gap.py`, pure function) | Database/Storage (`Observatory.observations_type` read) | `claimed_dates()` is pure-logic, already reads `run.site` (FK), no new query shape |
| `pending_narrowing_runs` display | Frontend Server / Django templates (`campaignrun_gap_analysis.html`) | API/Backend (`CampaignGapAnalysisView`, no change needed — see Pitfall 4) | Template already receives the whole `result` dict; new dict key is enough |
| TBD-badge tooltip | Frontend Server (`campaign_tables.py` render method) | — | django-tables2 column-render method, server-side HTML generation |

## Current-State Confirmation

CONTEXT.md's canonical_refs cites line numbers for each file "this phase must change" —
confirmed against the real file bodies read this session (2026-07-10). Differences from
CONTEXT.md's citations are noted; none are material, but the exact current bodies below are
what the plan should diff against, not the cited ranges.

### `solsys_code/campaign_utils.py` — `parse_obs_window()` (actual lines 186-244, not 185-243)

Full current body (verified):

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

**Key finding not stated in CONTEXT.md:** the function's *only* Obs.-Date-side logic is the
single unguarded `strptime` call on line 214. Everything below it is UT-Time-Range parsing
(unchanged by this phase per D-13's own text). This means the D-11/D-12/D-13 change is
localized to inserting new branches **before** (or wrapping) that one line — the three
existing UT-time regexes and their logic need zero changes.

**Second key finding:** the function's return signature today is `(obs_date: date, ut_start:
datetime, ut_end: datetime | None, ut_needs_review: bool)` — a *single* `obs_date`, not a
`window_start`/`window_end` pair. `ut_start`/`ut_end` (the UT-Time-Range-derived datetimes)
are **already vestigial**: `CampaignRun` has had no `ut_start`/`ut_end` fields since Phase
19's migration `0004` dropped them (confirmed — see `models.py` below), and
`import_campaign_csv.py`'s only caller discards them via `_ut_start, _ut_end = ...` (line
114). Only `ut_needs_review` (for collision-log wording) and `obs_date` (to build
`window_start`/`window_end` in the lookup dict, both set to the same value) are actually
used downstream today. See "Concrete D-11/D-12/D-13 Parsing Design" below for how the new
signature should extend this without disturbing the still-relevant parts.

### `solsys_code/campaign_gap.py` — `claimed_dates()` (actual lines 115-187, not 114-186)

Full current body confirmed (see full read above). The load-bearing loop, unchanged parts
kept for context:

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

`qs` is built with `.only('pk', 'window_start', 'window_end')` (line 162) — **this does NOT
include `site` or any `Observatory` field.** ASSET-01's ground-vs-space branch needs
`run.site.observations_type`, so the `.only()` restriction must be widened to include `site`
(and the query needs `select_related('site')` to avoid an N+1 query per run in the loop —
see Pitfall 3). `site` itself is already the query's filter column (`qs = ...filter(...,
site=site, ...)` at line 160), so every row in `qs` shares the *same* `site` object already
passed into `claimed_dates(campaign, target, site)` as a parameter — **the ground-vs-space
check does not need to be per-row at all; it can be computed once, outside the loop**, since
every returned run already has the same site (this simplifies the implementation
significantly versus a naive per-row `run.site.observations_type` read, and avoids the
`.only()`/`select_related` widening entirely — see Pitfall 3 for why this matters).

`_compute_gap()` (lines 190-224) unpacks `claimed, undated_runs, unattributed_runs =
claimed_dates(...)` and assembles the returned dict's keys directly from those three
values — the dict literal at lines 217-224 is the single place a fourth key
(`pending_narrowing_runs`) needs adding.

`get_or_compute_gap()` (lines 227-252) does nothing today that inspects individual dict
keys — it caches/returns the whole dict opaquely. **No change needed here** beyond the dict
literal in `_compute_gap()` already carrying the new key through.

### `solsys_code/management/commands/import_campaign_csv.py` — `Command.handle()` (actual lines 47-197)

Full current natural-key-relevant section confirmed (lines 106-182, see full read above).
Three things to note precisely, since CONTEXT.md's canonical_refs summarizes but doesn't
quote the exact current shape:

1. Line 114: `obs_date, _ut_start, _ut_end, ut_needs_review = parse_obs_window(...)` is
   inside a `try: ... except ValueError as exc: ... skipped_count += 1; continue` block
   (lines 108-126) — this whole `except ValueError` branch becomes dead code once
   `parse_obs_window()` stops raising (D-13), and must be removed, not just left unreachable
   (ruff/lint would likely flag it as unreachable-adjacent dead code, and D-07 requires
   `skipped_count` to no longer count Obs.-Date outcomes).
2. Line 131: `collision_key = (campaign.pk, telescope_instrument, obs_date)` — a
   **single-date** tuple. This must become **two different key shapes** depending on
   resolved-vs-TBD, matching the model's two partial constraints (see Pitfall 2).
3. Lines 169-182: the `insert_or_create_campaign_run({...lookup...}, fields)` call's lookup
   dict is currently `{campaign, telescope_instrument, window_start: obs_date, window_end:
   obs_date}` — always the resolved-window shape. This must also branch (Pitfall 2).

### `solsys_code/models.py` — `CampaignRun` (actual lines 31-157, not 30-155)

Confirmed: `site_needs_review` (lines 86-88) is exactly `models.BooleanField(default=False,
verbose_name='...')` — the direct, simple precedent for `window_needs_review`. No
`original_obs_date_raw` or `window_needs_review` field exists yet (as expected — this phase
adds them). `Meta.constraints` (lines 113-153) confirmed as three constraints:
`unique_campaign_run_resolved_window` (fields: `campaign, telescope_instrument, window_start,
window_end`, condition `window_start__isnull=False`), `unique_campaign_run_tbd_natural_key`
(fields: `campaign, telescope_instrument, contact_person`, condition
`window_start__isnull=True`), and `campaign_run_window_start_end_null_together` (a
`CheckConstraint` enforcing both-null-or-both-set — added in migration `0005`, one migration
later than CONTEXT.md's "0004 single combined migration" framing; see Migration Mechanics).

### `solsys_code/campaign_tables.py` — `render_window_start()` (actual lines 136-149)

Confirmed exact current body (see full read above). The file already has **three** existing
`title="..."` tooltip precedents in `render_site()` (lines 127, 132) and
`render_open_to_collaboration()` (lines 154-155) — D-08's tooltip addition has an in-file
precedent to match, no need to reach for `CalendarEventTelescopeLabel`'s convention
elsewhere.

### `solsys_code/solsys_code_observatory/models.py` — `Observatory.observations_type`/`SATELLITE_OBSTYPE`

Confirmed at lines 17-26: `SATELLITE_OBSTYPE = 2` (an `int`, one of four
`SmallIntegerField` choices: `OPTICAL_OBSTYPE=0`, `OCCULTATION_OBSTYPE=1`,
`SATELLITE_OBSTYPE=2`, `RADAR_OBSTYPE=4`). The exact check used elsewhere in this codebase
(Phase 19 D-06, `campaign_views.py:339`):
```python
if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
```
This is the precedent to reuse verbatim in `claimed_dates()`.

### `src/templates/campaigns/campaignrun_gap_analysis.html`

Confirmed current structure (see full read above): a single
`{% if result.undated_runs or result.unattributed_runs %}` alert-warning block (lines 53-69)
with one `<h5>`+`<p>` pair per non-empty bucket. The new `pending_narrowing_runs` block
should follow the identical shape — a third conditional inside the same `alert-warning` div
(or a new sibling `alert-info`/`alert-warning` block; tone is a template-matching choice, not
re-litigated here per CONTEXT.md).

### `solsys_code/campaign_views.py` — `CampaignGapAnalysisView.get()` (actual lines 461-520)

**Correction to CONTEXT.md's canonical_refs:** CONTEXT.md states this view "needs to pass
the new `pending_narrowing_runs` bucket into template context." Confirmed against the real
code (line 518-519): `result = get_or_compute_gap(...)` then
`context.update({..., 'result': result})` — the **entire** `result` dict (produced by
`_compute_gap()`) is already passed to the template as `context['result']`, and the template
already accesses `result.undated_runs`/`result.unattributed_runs` as dict-key lookups (Django
template `.` resolution works on dict keys automatically). **No `campaign_views.py` change
is needed** — adding `pending_narrowing_runs` to the dict `_compute_gap()` returns is
sufficient; it flows through `get_or_compute_gap()`'s cache-or-compute wrapper and the
existing `context.update()` call unchanged. This simplifies the plan's file list.

### `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb`

Confirmed: uses `docs/notebooks/pre_executed/fixtures/campaign_sample.csv`, a synthetic
6-row fixture where **every** `Obs. Date` value is an exact `YYYY-MM-DD` string (verified by
reading the fixture directly — no range, no TBD, no compact/rollover shape present). The
notebook's markdown only *mentions* `parse_obs_window`'s "best-effort fallback paths" in
prose (line 196) but never exercises a range or TBD row in an executed code cell. Per
CLAUDE.md's explicit note, this notebook is **not** on the mandatory-sync list, but
CONTEXT.md flags it for a check — confirmed finding: **it currently demonstrates none of
this phase's new behavior**, so if the plan wants the notebook to stay a meaningful
demonstration of `import_campaign_csv`, it should add at least one range row and one TBD row
to `campaign_sample.csv` (or a second small CSV) and a cell showing the resulting
`CampaignRun.window_needs_review`/`original_obs_date_raw` values. This is optional per
CLAUDE.md (not one of the four modules on the mandatory notebook list) but strongly
recommended for the notebook to remain accurate — flag as a planner discretion call, not a
must-have gate.

## Concrete D-11/D-12/D-13 Parsing Design

**No new dependency required.** `python-dateutil` is present in the dev environment
(`2.9.0.post0`) but only as a *transitive* dependency of `arrow`/`pandas`/`tomtoolkit` etc.
`[VERIFIED: codebase — grep found no dateutil/python-dateutil entry in pyproject.toml]` —
exactly the same "incidental transitive dependency, not a project dependency" situation
Phase 18's spike documented for `rapidfuzz` (18-DECISION.md criterion 4). D-11's rollover
arithmetic needs nothing beyond stdlib `datetime.date`, which already raises `ValueError` on
an invalid day-of-month (e.g. `date(2025, 2, 35)` -> `ValueError: day is out of range for
month`, confirmed by direct execution this session) — this is exactly the "never raise
uncaught" fallback signal D-13 wants, so no `calendar.monthrange()` day-count guard is even
needed; catching `ValueError` around the whole block is sufficient and matches this
codebase's existing "let the stdlib validate, catch and fall back" style (`parse_obs_window`
already relies on `strptime`'s own `ValueError` for the exact-date case).

### New regexes (module-level, alongside `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`)

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

Both are anchored with `^...$` (unlike the UT-time regexes, which use `.search()` against
free text) because `Obs. Date` cells are structured, not free prose — a `.search()` here
would risk a false-positive match inside a longer TBD sentence like `"TBD pending Cycle
2"` (which contains no digits matching either pattern, but a `.search()`-based design is the
wrong posture for a column that's supposed to be a date). Try these with `.match()` (or
`.fullmatch()`) after `.strip()`, not `.search()`.

**Order of attempts in `parse_obs_window()`** (each guarded, first match wins, no match at
all falls through to the D-13 TBD return):

1. Exact date: `datetime.strptime(text, '%Y-%m-%d').date()` (existing logic, now wrapped in
   `try/except ValueError` instead of letting it propagate) -> `window_start = window_end =
   parsed_date`.
2. `_DATE_RANGE_FULL` match -> parse both groups via `strptime(..., '%Y-%m-%d')` (each
   independently `try/except`-guarded — a malformed date on either side, e.g.
   `"2025-13-05 to 2025-09-22"`, should fall through to TBD, not raise) -> `window_start`,
   `window_end` from the two groups directly (no rollover math needed — both sides are
   already full dates).
3. `_DATE_RANGE_COMPACT` match -> see rollover algorithm below.
4. No match at all (including blank, `"YYYY-MM-?"`, and genuine garbage) -> TBD:
   `window_start = window_end = None`, `original_obs_date_raw = text`, `window_needs_review
   = True`.

**Why no dedicated `YYYY-MM-?` regex is needed:** D-03 already decided this shape collapses
to the identical TBD state as a blank cell. Since step 4 (the catch-all) already captures the
raw text verbatim regardless of *why* nothing matched, a `"2025-12-?"` cell falls through
steps 1-3 (none match) and lands in step 4 with `original_obs_date_raw = "2025-12-?"`
automatically — **no special-cased regex for this shape is needed at all**, simplifying the
implementation versus what CONTEXT.md's canonical_refs might imply ("YYYY-MM-? marker" listed
as a distinct shape to detect). This is a meaningful implementation simplification worth
calling out explicitly to the planner.

### D-11 rollover algorithm (step 3 above)

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

Worked examples (verified by direct execution this session):
- `"2025-11-02 -25"` (Phase 18's confirmed same-month example) -> `day2=25 >= day1=2` ->
  `window_end = date(2025, 11, 25)` — same behavior as today, unchanged.
- `"2025-11-28 -05"` (D-11's new rollover example) -> `day2=5 < day1=28` -> `month != 12` ->
  `window_end = date(2025, 12, 5)`.
- `"2025-12-20 -03"` (December -> January, year rollover) -> `day2=3 < day1=20` -> `month ==
  12` -> `window_end = date(2026, 1, 3)`.
- `"2025-02-01 -35"` (invalid day, no real month has 35 days) -> `date(2025, 2, 35)` raises
  `ValueError` -> falls through to TBD, `original_obs_date_raw = "2025-02-01 -35"`,
  `window_needs_review = True` — never crashes the import.

### New return contract (D-13)

Recommended new signature (extends, not replaces, the existing return tuple shape so the
still-relevant `ut_start`/`ut_end`/`ut_needs_review` values keep working unchanged for the
single-night case):

```python
def parse_obs_window(
    obs_date_raw: str, ut_range_raw: str
) -> tuple[date | None, date | None, str, bool, datetime | None, datetime | None, bool]:
    """Returns (window_start, window_end, original_obs_date_raw, window_needs_review,
    ut_start, ut_end, ut_needs_review). Never raises.
    """
```

**Discretion flagged for the planner:** when the row resolves to a *range* or *TBD*
(`window_start != window_end`, or both `None`), there is no single concrete date to anchor
`ut_start`/`ut_end` to. Recommendation: skip UT-Time-Range parsing entirely in that case
(`ut_start = ut_end = None`, `ut_needs_review = False`) rather than anchoring to
`window_start` — this is a defensible default because (a) `ut_start`/`ut_end` are already
unused by every real caller (`CampaignRun` has no fields for them since Phase 19), and (b) a
range/TBD row's `UT Time Range` cell, if present, describes a single night's timing that
doesn't apply to a multi-night window anyway. This is a behavior change from today (today
always computes *some* `ut_start`, since `obs_date` was always a single valid date by the
time UT parsing ran) but has zero observable effect since nothing persists these values.
Flag this as an assumption for the planner/discuss-phase to confirm if it matters for a
future consumer (see Assumptions Log).

## Pitfalls (Implementation-Level, Beyond CONTEXT.md)

### Pitfall 1: Two existing tests assert the *old* contract on inputs this phase changes

`solsys_code/tests/test_import_campaign_csv.py`:
- `test_parse_obs_window_unparseable_date_raises` (line 270-272): calls
  `parse_obs_window('', '08:50 - 11:50')` and asserts `assertRaises(ValueError)`. Under
  D-13, this must become a TBD-tuple assertion instead (`window_start is None`,
  `window_needs_review is True`, `original_obs_date_raw == ''`) — **not deleted**, since
  blank-date behavior is still real behavior to test, just a different contract.
- `test_natural_key_failure_skipped_and_logged` (line 470-502): uses `'Obs. Date':
  '2025-11-02 -25'` as its "malformed date" fixture — this is **exactly** Phase 18's
  confirmed compact-same-month-range example, which this phase makes successfully
  parseable (`window_start=2025-11-02, window_end=2025-11-25`), not skipped. This test's
  fixture row must be replaced with a genuinely non-natural-key failure (e.g. blank
  `Telescope / Instrument`, which is still a real skip condition per D-07) or the test will
  fail — it is currently testing behavior this phase explicitly reverses. **Any plan for
  this phase must update this test, not just add new ones.**

### Pitfall 2: `import_campaign_csv.py`'s lookup/collision-key must branch resolved-vs-TBD

The model's two partial `UniqueConstraint`s have **different field sets**:
`unique_campaign_run_resolved_window` keys on `(campaign, telescope_instrument,
window_start, window_end)`; `unique_campaign_run_tbd_natural_key` keys on `(campaign,
telescope_instrument, contact_person)`. Today's `import_campaign_csv.py` always builds the
resolved-window lookup shape (line 169-180) because every row was previously guaranteed to
resolve to a concrete date. Once TBD rows can be created via the new catch-all path
(D-06/D-13), both the pre-DB-hit `seen_window_keys` collision set (line 89, 131-142) *and*
the `insert_or_create_campaign_run()` lookup dict (line 169-180) must branch:

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

Getting this wrong (e.g. always using the resolved-window lookup shape even for a TBD row,
with `window_start=None, window_end=None` in the lookup dict) would mean
`get_or_create(**lookup, defaults=fields)`'s own `.filter(**lookup).first()` pre-check
doesn't align with the DB's actual unique index on the TBD branch (which excludes
`window_start`/`window_end` from its field tuple and includes `contact_person` instead) —
two TBD rows for the same telescope/contact re-imported in the same batch could both attempt
`.create()`, and the second would hit the DB's `unique_campaign_run_tbd_natural_key`
constraint as an **uncaught `IntegrityError`** rather than being correctly detected as a
duplicate ahead of time. `fields` (the `defaults=` dict) must **not** also include
`contact_person` when it's promoted into `lookup` for the TBD branch — Django's
`get_or_create` requires lookup and defaults keys to be disjoint... actually they may
overlap in practice (defaults values for keys not in lookup are used only on create), but to
avoid ambiguity/duplication, pull `contact_person` out of `fields` and only set it via
`lookup` for the TBD branch (it's still available in `fields` unconditionally for the
resolved-window branch, where it isn't part of the key).

### Pitfall 3: Widening `claimed_dates()`'s `.only()` for the site check — or better, avoid it

`claimed_dates()`'s queryset explicitly restricts fetched columns to `.only('pk',
'window_start', 'window_end')` (line 162) as a documented PII-minimization discipline
(D-13/WR-01 comment). Naively adding `run.site.observations_type` inside the per-run loop
would either (a) trigger a `DeferredAttributeError`-adjacent lazy-load per row (extra query
per run, an N+1), or (b) require widening `.only()` to include `site` plus a
`select_related('site')`, touching the documented PII-minimization comment. **Better:**
since the queryset is already filtered to a single `site=site` value (the function's own
`site` parameter, line 160), every row shares the identical site — compute
`is_space_mission = site.observations_type == Observatory.SATELLITE_OBSTYPE` **once, outside
the loop**, using the `site` parameter directly (never touching `run.site`). This avoids the
`.only()` change entirely and is both simpler and cheaper than a per-row check.

### Pitfall 4: `campaign_views.py` does not need a code change (see Current-State Confirmation)

Already covered above — noted here again as a pitfall because CONTEXT.md's canonical_refs
lists this file as one "this phase must change." Confirmed it does not need to be touched;
including it in a plan's `files_modified` when no line actually changes would be a false
positive for the plan-checker.

### Pitfall 5: `render_window_start()`'s single-night check after this phase

`render_window_start()`'s `if start == end: return start` branch (line 147-148) already
handles the case where a space-mission run's window has narrowed to one night — no change
needed there for ASSET-02. The only `campaign_tables.py` change this phase needs is D-08's
tooltip addition to the `start is None` (TBD) branch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Month/year rollover arithmetic (D-11) | A custom days-in-month lookup table or manual `if month == 2: ...` chain | stdlib `datetime.date(year, month, day)` — already raises `ValueError` on an invalid combination | This codebase already relies on `strptime`'s own validation for the exact-date case (line 214); the same "let stdlib validate, catch the exception" posture applies directly, and `python-dateutil`/`calendar.monthrange()` add nothing this specific rollover needs |
| En-dash/hyphen-tolerant date-range detection (D-12) | A permissive general-purpose date-range parser (e.g. `dateutil.parser` fuzzy mode) | A narrowly-scoped, anchored regex per shape (`_DATE_RANGE_FULL`) | Matches this module's own documented Anti-Pattern warning (module docstring, line 1-9): "never a permissive general-purpose date/time parser" — a fuzzy parser would happily "succeed" on garbage text into a wrong-but-plausible date, exactly what this module's whole design avoids |
| TBD-row deduplication across a CSV import batch | A DB round-trip per row to check for an existing TBD row before creating | The existing `seen_window_keys` in-memory `set` pattern (already present, line 89), extended with the branching key shape from Pitfall 2 | Already the established pattern for the resolved-window case; no reason to introduce a different mechanism for TBD rows |

**Key insight:** every piece of new logic in this phase is an extension of an existing,
already-battle-tested pattern in this exact codebase (pattern-per-shape regex, branching
natural keys, bucketed non-claiming lists) — there is no case in this phase where reaching
for a new library or a more "clever" general solution is warranted.

## Migration Mechanics

Confirmed by reading the two most recent real migrations (`0004_campaignrun_window_schema.py`,
`0005_campaignrun_campaign_run_window_start_end_null_together.py`) — **note the "single
combined migration 0004" framing in STATE.md/ROADMAP.md is now one migration stale**: Phase
19 shipped **two** migrations (`0004` for the schema+backfill+dedup, `0005` as a follow-up
review-fix adding the `CheckConstraint` with its own defensive `RunPython` normalization
step). The next migration for this phase will be `0006`.

This phase's schema change is much simpler than either predecessor — **no backfill, no
dedup, no data migration needed at all**: `original_obs_date_raw` and `window_needs_review`
are brand-new columns with safe defaults (`''` and `False` respectively), not derived from
any existing column, and no existing row needs its value changed (every current row is
already a resolved single-night row with nothing to backfill into these two new
TBD-only fields). This is exactly the same shape as `site_needs_review`'s original addition
(a plain `AddField` with a default, no `RunPython` step) — confirm by checking migration
`0002_campaignrun.py`'s field-level default if precedent is wanted, though it isn't strictly
necessary here.

Recommended migration `0006_campaignrun_original_obs_date_raw_and_window_needs_review.py`
(naming mirrors the existing `NNNN_campaignrun_<description>.py` convention):

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

**Field-type recommendation (Claude's Discretion item, resolved here):** `CharField(max_length=255,
blank=True, default='')`, not `TextField` — mirrors `site_raw`'s exact precedent
(`models.py` line 85: `models.CharField(max_length=255, blank=True, default='', ...)`) and
every real example seen in the Phase 18 spike (`"TBD pending Cycle 2"`, `"2025-12-?"`) is
well under 255 characters. `TextField` would be inconsistent with this model's established
convention for "raw free-text sidecar of a structured field" (`site_raw` is the direct
precedent D-01 itself cites).

No `RunPython` step, no new `UniqueConstraint`, no `CheckConstraint` change — the two
existing partial constraints and the null-together `CheckConstraint` from migration `0005`
are entirely orthogonal to these two new columns (neither new field participates in any
constraint).

## Testing Conventions

Confirmed by reading `solsys_code/tests/test_campaign_gap.py` and
`solsys_code/tests/test_import_campaign_csv.py` directly:

- **Framework:** `django.test.TestCase`, run via `./manage.py test solsys_code` (not
  pytest) — both target files are already under `solsys_code/tests/`, consistent with
  CLAUDE.md's testing split.
- **Target factory:** `tom_targets.tests.factories.NonSiderealTargetFactory` — used
  throughout both files (`TestClaimedDates.setUpTestData`, `TestImportCampaignCsv`), per
  CLAUDE.md's mandatory rule. No sidereal factory usage found anywhere in either file.
- **`CampaignRun` fixture pattern:** a `_make_run(**kwargs)` helper method with sensible
  defaults, overridden per test (`TestClaimedDates._make_run`, line 154-163) — the
  established local pattern for this file; new ground-vs-space tests should extend this
  helper's `**kwargs`, not introduce a parallel factory.
- **Space-mission `Observatory` fixture precedent:** already exists, in
  `test_campaign_approval.py`'s `test_approve_single_night_space_run_creates_midnight_utc_placeholder_event`
  (line 200-206):
  ```python
  space_site = Observatory.objects.create(
      obscode='250', name='Test Space Telescope', short_name='TST',
      observations_type=Observatory.SATELLITE_OBSTYPE,
  )
  ```
  This is the exact fixture shape `test_campaign_gap.py`'s new ground-vs-space tests should
  reuse (note: this fixture omits `lat`/`lon`/`timezone` — fine for `claimed_dates()`'s
  needs, since the branch only reads `observations_type`, never calls `sun_event()` for a
  space site).
- **CSV-import test helper:** `test_import_campaign_csv.py` uses a `_WriteCsvMixin` +
  `_row(**overrides)` helper (confirmed via grep) to build one-off CSV fixtures per test —
  the same pattern should be used for new range/TBD-shape import tests, not a new fixture
  file per test.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `django.test.TestCase` (Django's own test runner) |
| Config file | none — Django settings module `src.fomo.settings` via `manage.py` |
| Quick run command | `./manage.py test solsys_code.tests.test_import_campaign_csv` / `./manage.py test solsys_code.tests.test_campaign_gap` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IMPORT-01 | `" to "`/en-dash/hyphen full-date range parses to correct `window_start`/`window_end` | unit | `./manage.py test solsys_code.tests.test_import_campaign_csv.TestCampaignUtils` | ✅ (extend existing class) |
| IMPORT-01 | Compact same-month range (`"2025-11-02 -25"`) parses correctly (regression — this exact string currently a "malformed" test fixture, Pitfall 1) | unit | same as above | ✅ (existing test needs editing, not just adding) |
| IMPORT-01 | D-11 rollover: compact range with day2 < day1, same year (Nov -> Dec) | unit | same as above | ❌ Wave 0 — new test |
| IMPORT-01 | D-11 rollover: compact range crossing Dec -> Jan (year increments) | unit | same as above | ❌ Wave 0 — new test |
| IMPORT-01 | Compact range with an invalid resulting day (e.g. day2=35) falls through to TBD, never raises | unit | same as above | ❌ Wave 0 — new test |
| IMPORT-02 | Blank `Obs. Date` -> TBD tuple (`window_start is None`, `window_needs_review=True`, `original_obs_date_raw=''`) | unit | same as above | 🔁 existing test asserts the *old* raise contract, must be rewritten (Pitfall 1) |
| IMPORT-02 | `"YYYY-MM-?"` marker -> TBD tuple with `original_obs_date_raw="2025-12-?"` | unit | same as above | ❌ Wave 0 — new test |
| IMPORT-02 | Genuine garbage text (e.g. `"TBD pending Cycle 2"`) -> TBD tuple, `window_needs_review=True` | unit | same as above | ❌ Wave 0 — new test |
| IMPORT-01/02 | `import_campaign_csv` command: a range-shaped row creates a `CampaignRun` with the matching window (not skipped) | integration | `./manage.py test solsys_code.tests.test_import_campaign_csv.TestImportCampaignCsv` | ❌ Wave 0 — new test |
| IMPORT-01/02 | `import_campaign_csv` command: a TBD-shaped row creates a flagged `window_needs_review=True` row, counted in the new summary counter, never in `skipped_count` | integration | same as above | ❌ Wave 0 — new test |
| IMPORT-01/02 | Two TBD rows for the same telescope+campaign but different `contact_person` both import (TBD natural key uses `contact_person`, per Phase 18's real JWST-collision evidence) | integration | same as above | ❌ Wave 0 — new test |
| IMPORT-01/02 | Two TBD rows for the same telescope+campaign+`contact_person` in one CSV batch collide and the second is skipped/logged (natural-key collision, Pitfall 2) | integration | same as above | ❌ Wave 0 — new test |
| ASSET-01/02 | Ground run with full window -> every date in `claimed` | unit | `./manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates` | ✅ (existing `test_range_run_claims_every_date_in_window` already covers the ground case; confirm it still passes unchanged) |
| ASSET-01/02 | Space run with `window_start == window_end` (narrowed) -> that one date claimed | unit | same as above | ❌ Wave 0 — new test |
| ASSET-01/02 | Space run with `window_start != window_end` (un-narrowed range) -> zero dates claimed, lands in `pending_narrowing_runs`, not `undated_runs` | unit | same as above | ❌ Wave 0 — new test |
| ASSET-01/02 | Space run with TBD (both `None`) -> zero dates claimed, lands in `undated_runs` (not `pending_narrowing_runs` — D-09's explicit distinction) | unit | same as above | ❌ Wave 0 — new test |
| ASSET-02 | Gap-analysis template renders the new `pending_narrowing_runs` alert block with correct count | integration (view-level, `TestGapAnalysisView`) | `./manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView` | ❌ Wave 0 — new test |
| D-08 | TBD badge's `title` attribute shows `original_obs_date_raw` when set, absent when blank | unit (table-render, likely `test_campaign_views.py` or a new `test_campaign_tables.py` if one doesn't exist) | `./manage.py test solsys_code` (locate exact file during planning) | ❌ Wave 0 — new test; confirm target test file during planning (no `test_campaign_tables.py` file was found by name in this session's directory listing — likely lives in `test_campaign_views.py`; verify before scoping) |

### Sampling Rate

- **Per task commit:** `./manage.py test solsys_code.tests.test_import_campaign_csv
  solsys_code.tests.test_campaign_gap` (the two directly-touched test files; fast, no
  SPICE-kernel import cost since neither `campaign_utils.py` nor `campaign_gap.py` imports
  the heavy ephemeris module — confirmed by `campaign_gap.py`'s own module docstring, lines
  9-14, and `TestNoHeavyEphemerisImport` at `test_campaign_gap.py` line 506, which already
  guards this invariant).
- **Per wave merge:** `./manage.py test solsys_code` (full app suite — still pays the heavy
  SPICE-kernel cost once, per CLAUDE.md's documented import side effect, since some other
  test module in the same run likely imports `ephem_utils`/`views` transitively; unavoidable
  regardless of this phase's own scope).
- **Phase gate:** full `./manage.py test solsys_code` green, plus `ruff check .` and `ruff
  format --check .` clean (CLAUDE.md's explicit quality-gate constraint for this project).

### Wave 0 Gaps

- [ ] `test_parse_obs_window_unparseable_date_raises` (line 270-272) — rewrite to assert the
  new TBD-tuple contract instead of `assertRaises(ValueError)`.
- [ ] `test_natural_key_failure_skipped_and_logged` (line 470-502) — replace its
  `'2025-11-02 -25'` fixture row (now a valid parseable range) with a genuine non-date
  natural-key failure (blank `Telescope / Instrument`).
- [ ] New unit tests for `parse_obs_window()`'s D-11/D-12 branches (rollover same-year,
  rollover year-crossing, invalid-day fallback, en-dash/hyphen full-range variants) — none
  exist yet.
- [ ] New unit tests for `claimed_dates()`'s ground-vs-space branch and
  `pending_narrowing_runs` bucket — none exist yet; the `TestClaimedDates` class's existing
  `_make_run` helper and the `test_campaign_approval.py` space-`Observatory` fixture
  (line 200-206) are both directly reusable, no new fixture infrastructure needed.
- [ ] New integration tests for `import_campaign_csv`'s range/TBD import paths — none exist
  yet; the existing `_WriteCsvMixin`/`_row()` helpers in `test_import_campaign_csv.py` are
  directly reusable.
- [ ] Locate (or create) the test file covering `CampaignRunTable.render_window_start()`'s
  tooltip — not confirmed to exist as a dedicated file this session; verify during planning
  before scoping a specific file path.

## Common Pitfalls

See "Pitfalls (Implementation-Level, Beyond CONTEXT.md)" above — Pitfalls 1-5 are the
substantive, non-obvious findings from this research session; they are placed earlier in
this document (immediately after the parsing design) because they are directly actionable
by the planner when scoping tasks, not general background.

## Code Examples

### Ground-vs-space check (exact precedent to reuse, from `campaign_views.py:339`)

```python
# Source: solsys_code/campaign_views.py, Phase 19 D-06 (verified this session)
if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
    # space-based observatory: no fixed horizon
    ...
```

### `site_needs_review` field (exact precedent for `window_needs_review`, from `models.py:86-88`)

```python
# Source: solsys_code/models.py (verified this session)
site_needs_review = models.BooleanField(
    default=False, verbose_name='Whether the site could not be automatically resolved and needs manual review'
)
```

### `render_site()`'s tooltip precedent (exact precedent for D-08, from `campaign_tables.py:126-134`)

```python
# Source: solsys_code/campaign_tables.py (verified this session)
if Accessor('site_needs_review').resolve(record, quiet=True):
    return format_html(
        '<span class="text-muted font-italic" title="Site could not be automatically resolved">'
        '<i class="fa fa-exclamation-triangle" aria-hidden="true"></i> {}</span>',
        site_raw,
    )
```

## State of the Art

No external-library or ecosystem shifts are relevant to this phase — every change is
internal to this codebase's own established conventions. N/A.

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** `python-dateutil` was considered
and explicitly rejected (see "Concrete D-11/D-12/D-13 Parsing Design" above) — it is present
in the dev environment only as a transitive dependency, is not needed for D-11's rollover
arithmetic (stdlib `datetime.date` already validates and raises), and adding it as an
explicit dependency for this phase would repeat the exact "incidental transitive dependency,
not a project dependency" trap Phase 18's spike already flagged for `rapidfuzz`. No
Package Legitimacy Gate run needed — table omitted.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | When a row resolves to a range or TBD, `ut_start`/`ut_end` should be `None`/`None` rather than anchored to `window_start` | Concrete D-11/D-12/D-13 Parsing Design, "New return contract" | Low — `ut_start`/`ut_end` are already unused by every real consumer (no `CampaignRun` field stores them); if a future phase needs a UT time for a range/TBD row, this default would need revisiting, but no current behavior depends on it |
| A2 | `original_obs_date_raw` should be `CharField(max_length=255)`, mirroring `site_raw` | Migration Mechanics | Low — every real example seen (Phase 18 spike) is well under 255 chars; if a future sheet snapshot has a longer TBD note, a follow-up migration to widen `max_length` is cheap and non-breaking |
| A3 | The `pending_narrowing_runs` alert block in `campaignrun_gap_analysis.html` should be a third conditional inside the existing `alert-warning` div, not a separate `alert-info` block | Current-State Confirmation, template section | Low — purely cosmetic; either choice satisfies D-09's substance, and CONTEXT.md explicitly leaves exact wording/styling to implementation |
| A4 | `test_campaign_views.py` (not a not-yet-existing `test_campaign_tables.py`) is the right home for a new `render_window_start()` tooltip test | Validation Architecture, Wave 0 Gaps | Low — a misplaced test still runs under `./manage.py test solsys_code`; only affects file organization, not coverage |

## Open Questions

1. **Should the demo notebook (`import_campaign_csv_demo.ipynb`) be updated in this phase?**
   - What we know: it's not on CLAUDE.md's four-module mandatory-sync list, but it
     currently demonstrates zero range/TBD import behavior (confirmed — its fixture CSV has
     no such rows).
   - What's unclear: whether "strongly recommended but optional" is acceptable to the user,
     or whether they'd rather it be scoped as a required task given it's this command's only
     executed demonstration of end-to-end behavior.
   - Recommendation: planner should scope it as an explicit task (add 1-2 rows to
     `campaign_sample.csv` covering a range and a TBD shape, add a cell showing the
     resulting `window_needs_review`/`original_obs_date_raw` values, re-execute via
     `jupyter nbconvert --to notebook --execute --inplace`) rather than silently skip it —
     matches the spirit of CLAUDE.md's repeated "this gap was hit twice already" warning,
     even though this specific notebook is formally outside the mandatory list.

2. **Exact home for the `render_window_start()` tooltip test.**
   - What we know: `campaign_tables.py` has no dedicated test file found by name this
     session (`test_campaign_tables.py` does not appear in the `solsys_code/tests/`
     directory listing).
   - What's unclear: whether table-render tests live inside `test_campaign_views.py` (which
     tests views that use these tables) or are simply untested today.
   - Recommendation: the planner should grep for `render_window_start` or
     `CampaignRunTable` inside `test_campaign_views.py` during planning to confirm before
     assigning a file path; low risk either way (a new small test file is also acceptable).

## Sources

### Primary (HIGH confidence — direct codebase reads this session)

- `solsys_code/campaign_utils.py` (full file) — `parse_obs_window()`, `resolve_site()`,
  existing regex conventions.
- `solsys_code/campaign_gap.py` (full file) — `claimed_dates()`, `_compute_gap()`,
  `get_or_compute_gap()`.
- `solsys_code/management/commands/import_campaign_csv.py` (full file) — `Command.handle()`.
- `solsys_code/models.py` (full file) — `CampaignRun`, `Meta.constraints`.
- `solsys_code/campaign_tables.py` (full file) — `CampaignRunTable`, `render_window_start()`.
- `solsys_code/campaign_views.py` (lines 330-345, 446-520) — `CampaignGapAnalysisView`,
  Phase 19 D-06 ground-vs-space check.
- `solsys_code/solsys_code_observatory/models.py` (lines 1-70) — `Observatory`,
  `SATELLITE_OBSTYPE`.
- `src/templates/campaigns/campaignrun_gap_analysis.html` (full file).
- `solsys_code/migrations/0004_campaignrun_window_schema.py`,
  `0005_campaignrun_campaign_run_window_start_end_null_together.py` (full files).
- `solsys_code/tests/test_campaign_gap.py`, `test_import_campaign_csv.py`,
  `test_campaign_approval.py` (targeted reads/greps for test patterns and fixtures).
- `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` and its
  `fixtures/campaign_sample.csv` (grep + direct read).
- `pyproject.toml`, live `pip show python-dateutil` — confirmed no explicit dependency.
- Direct Python execution this session — confirmed `date()` rollover arithmetic and
  invalid-day `ValueError` behavior.

### Secondary (MEDIUM confidence)

- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` and
  `docs/design/uncertain_scheduling_spike.rst` — locked upstream findings (real CSV shapes,
  fuzzy-library verdict), treated as given per this phase's scope.
- `.planning/phases/19-window-schema-migration/19-CONTEXT.md` — Phase 19's own decisions,
  cross-checked against the real migration files.

### Tertiary (LOW confidence)

None — no WebSearch or external-library research was needed for this phase; every claim is
either a direct codebase read or original algorithm design grounded in stdlib behavior
verified by direct execution.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase touches no auth surface — CSV import is a management command (staff/CLI-only), coverage-gap page is already public/read-only per existing design |
| V3 Session Management | No | No session-related change |
| V4 Access Control | No | No permission-gate change; `CampaignGapAnalysisView` remains public/read-only per its existing docstring, unchanged by this phase |
| V5 Input Validation | Yes | Regex-anchored, narrowly-scoped parsing (`_DATE_RANGE_FULL`/`_DATE_RANGE_COMPACT`, both `^...$`-anchored) — never a permissive parser; every failure path falls to a flagged, non-crashing TBD state rather than propagating an exception or silently mis-parsing |
| V6 Cryptography | No | No cryptographic surface touched |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/adversarial CSV cell text causing an unhandled exception that aborts the whole batch import (a public/community-editable Google Sheet is the real data source — CLAUDE.md/CONTEXT.md context) | Denial of Service | D-13's "never raise" contract change is itself the mitigation — every `Obs. Date` shape, including deliberately malformed input, now falls through to a flagged TBD row rather than crashing `Command.handle()`'s row loop |
| PII leakage of `contact_person`/`contact_email` via a new code path (e.g. the TBD-branch natural-key lookup now includes `contact_person` directly in a `get_or_create` call, or a new log line) | Information Disclosure | `contact_person` is not PII-sensitive in the same way `contact_email` is (it's already displayed in the staff-facing table per D-08/existing design) and is already an existing `CampaignRun` field used elsewhere in lookups (Phase 19's TBD natural key already includes it) — no new exposure surface; continue the existing `WR-06` discipline of never logging Contact Person/Email in stderr diagnostic lines (`import_campaign_csv.py`'s existing row-skip log already redacts this — new TBD-path log lines, if any, must match) |
| `claimed_dates()`'s new ground-vs-space branch silently mis-classifying a run because `run.site` (or the passed-in `site` parameter) is `None` | Tampering / incorrect authorization-adjacent logic (a bug here could double-book a ground run or wrongly free a space-mission date) | `claimed_dates()`'s queryset is already filtered to `site=site` (a required, non-null parameter — the view never calls it with `site=None`, confirmed at `campaign_views.py` line 518), so `site` is never `None` inside this function; no new null-guard needed, but the plan should assert this invariant in a test rather than silently rely on it |

## Metadata

**Confidence breakdown:**
- Standard stack: N/A — no new library; stdlib-only design, HIGH confidence (verified by
  direct execution)
- Current-state confirmation: HIGH — every cited line/behavior read directly from the real
  file this session, several corrections made to CONTEXT.md's cited line numbers and one
  correction to its `campaign_views.py` "needs a change" claim
- D-11/D-12 parsing design: HIGH — concrete regex + algorithm, validated against real Phase
  18 spike examples and direct Python execution, no untested assumption
- Pitfalls 1/2/3: HIGH — each is a direct read of the exact current code that would break or
  need branching, not a guess
- Validation Architecture: HIGH — test framework/conventions confirmed by reading the real
  existing test files; specific new-test list is derived directly from the locked D-09/D-11/
  D-12/D-13 decisions, not speculative

**Research date:** 2026-07-10
**Valid until:** 30 days (stable, internal-codebase-only research; no external library
version drift risk since no new dependency is introduced)
