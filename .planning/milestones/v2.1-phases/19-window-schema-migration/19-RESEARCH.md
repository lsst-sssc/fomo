# Phase 19: Window-Schema Migration - Research

**Researched:** 2026-07-09
**Domain:** Django schema + data migration (nullable date-range field pair, partial `UniqueConstraint`), plus downstream consumer updates (django-tables2, django-filter, calendar projection, coverage-gap)
**Confidence:** HIGH (migration mechanics, verified against Django 5.2.15 docs and this repo's installed versions) / MEDIUM (exact display/UX shape of the new "window" column, left partly to implementation per CONTEXT.md)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Old-field retirement strategy**
- **D-01:** Hard cutover — `obs_date`/`ut_start`/`ut_end` are dropped from `CampaignRun` in this same
  phase, not kept as a transitional dual-schema. All 15 files that currently reference these fields
  (`models.py`, `campaign_forms.py`, `campaign_gap.py`, `campaign_views.py`, `campaign_utils.py`,
  `campaign_tables.py`, `management/commands/import_campaign_csv.py`, plus 6 test files:
  `test_campaign_approval.py`, `test_campaign_gap.py`, `test_campaign_views.py`,
  `test_campaign_submission.py`, `test_campaign_models.py`, `test_import_campaign_csv.py`) must be
  updated in this phase — there is no deferred cleanup phase for the old fields.
- **D-02:** The schema change ships as a **single combined migration** (add `window_start`/
  `window_end`, backfill from `obs_date`, drop the old fields and old `UniqueConstraint`, add the new
  one) — not split into separate add/backfill/drop migrations. User explicitly chose this over the
  split-into-reversible-steps option.

**TBD/window display convention**
- **D-03:** A TBD row (both `window_start`/`window_end` null) should render as **"TBD" with a visual
  flag** (badge/icon) on the per-campaign table and approval queue — best effort. If a badge/icon
  proves complicated given no existing badge/icon convention in `campaign_tables.py`, falling back to
  plain "TBD" text is acceptable; the user explicitly said to drop the visual flag if it doesn't work
  out.
- **D-04:** TBD rows sort **last** — scheduled rows (most recent `window_start` first) lead the table,
  replacing today's default `order_by = ('-obs_date',)` in `campaign_tables.py`. TBD rows are the
  least-resolved, lowest-priority-to-display rows in this ordering.
- **D-05:** A range row (`window_start != window_end`) displays using **`->`** between the two dates —
  e.g. `"Aug 1, 2026 -> Aug 15, 2026"` — not an en-dash. A single-night row (`window_start ==
  window_end`) still displays as one date. No range rows exist until Phase 20's CSV import lands, but
  the column-rendering logic for `CampaignRunTable`/`ApprovalQueueTable` is written now per D-01's hard
  cutover.

**Calendar projection during the gap**
- **D-06:** `campaign_views.py`'s calendar-projection gate (currently `if run.telescope_instrument and
  run.ut_start and run.ut_end:`, assigning `ut_start`/`ut_end` directly as `CalendarEvent`
  `start_time`/`end_time`) is **hybrid** once `ut_start`/`ut_end` are gone:
  - **Ground-based observatories** (resolved `site` where `Observatory.observations_type !=
    SATELLITE_OBSTYPE`): reuse Stage 1's `telescope_runs.sun_event()` for a real, dip-corrected
    dark-window banner — same accuracy convention the rest of the calendar feature already uses.
  - **Space-based observatories** (`Observatory.observations_type == SATELLITE_OBSTYPE`): use a
    midnight-UTC placeholder (`window_start` 00:00 UTC to `window_end`/`window_start` 23:59 UTC) —
    `sun_event()` doesn't apply to a space telescope with no fixed horizon.
  - Only projects when `window_start == window_end` (a single concrete night) — ranges and TBD runs
    still don't get a `CalendarEvent`, matching today's `ut_end`-missing gate behavior.
  - **Note for planner/researcher:** this is a narrow, early application of the ground-vs-space
    distinction that Phase 20's ASSET-01 formalizes for coverage-gap analysis. Phase 19 only needs the
    `Observatory.observations_type` check for this one projection code path — it is NOT expected to
    build the full asset-aware `claimed_dates()` rewrite (that's Phase 20's job).

**Existing duplicate-row cleanup**
- **D-07:** Live query against the dev DB found 2 real pairs of fully-duplicate `CampaignRun` rows that
  would still collide under Phase 18's contact_person-based partial-constraint recommendation. These
  are identified as leftover demo/UAT fixture rows, not genuine campaign data — **the migration deletes
  the duplicates** as part of its data-cleanup step, before applying the new partial `UniqueConstraint`.
- **D-08:** The data migration's de-dup step is **generic**, not hardcoded to the 2 known pk pairs: it
  queries for ANY `(campaign, telescope_instrument, contact_person)` collision group among null-window
  rows, keeps one row per group (lowest pk), and logs what it removed. This makes the migration
  re-runnable and portable to another environment with different leftover data, rather than crashing
  with an `IntegrityError` on an untested collision the discussion didn't spot.

### Claude's Discretion
- Exact partial/conditional `UniqueConstraint` SQL mechanism (Django `condition=` syntax, portable
  across SQLite and PostgreSQL per SCHED-04) — a technical implementation choice for the
  planner/researcher to work out, not re-litigated here. The natural-key composition itself
  (`campaign`, `telescope_instrument`, `contact_person`, condition `window_start IS NULL`) is restated
  as locked by Phase 18's `18-DECISION.md`. **This research's recommendation for the mechanism and for
  the (undecided) resolved-window branch's key composition is below in Architecture Patterns.**
- Default table sort direction/format details beyond D-04/D-05 (e.g. exact date format string,
  badge/icon visual styling if pursued) — left to implementation to match existing
  `campaign_tables.py`/template conventions.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. The one weakly-matched pending todo
(`2026-06-23-...rename-calendar-utils-py-private-helpers-to-reflect-shared-m...`, score 0.6) was
reviewed and left deferred: it's about `calendar_utils.py`'s private helpers (Stages 1-3), unrelated to
`CampaignRun`/window schema, and already marked "deliberately deferred, no second consumer yet" in
`.planning/STATE.md`.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-------------------|
| SCHED-02 | `obs_date`/`ut_start`/`ut_end` replaced by `window_start`/`window_end`; single night = `window_start == window_end` | Migration design in Architecture Patterns Pattern 1; model field change confirmed against current `models.py` (read directly, lines 89-91, 114-125) |
| SCHED-03 | A `CampaignRun` can exist fully TBD (both window fields null), distinct from a resolved window | Same migration; partial constraint B (Pattern 2) is scoped exactly to `window_start IS NULL` rows, i.e. the TBD state |
| SCHED-04 | Natural-key `UniqueConstraint` closes the NULL-uniqueness gap via a partial/conditional constraint, portable across SQLite and PostgreSQL | Verified directly against Django 5.2 docs (`Index.condition`/`UniqueConstraint.condition`): partial constraints work on SQLite and PostgreSQL, NOT on MySQL/MariaDB/Oracle — see Pitfall 1 and Pattern 2 |
| SCHED-05 | Existing rows migrate with no data loss (`window_start == window_end == obs_date`) | Data-migration `RunPython` pattern (Pattern 1, step 2) — an unconditional `.update(window_start=F('obs_date'), window_end=F('obs_date'))` over every row |
</phase_requirements>

## Summary

This phase is a single Django migration (Django 5.2.15, confirmed installed — `python -c "import
django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)`) plus updates to every direct consumer of
`CampaignRun.obs_date`/`ut_start`/`ut_end` — 7 non-test modules, 6 test modules, and one demo notebook
this research found is *not* on CONTEXT.md's file list (`docs/notebooks/pre_executed/
import_campaign_csv_demo.ipynb`, which contains `.order_by('obs_date', 'ut_start')` at line 309 and
must be regenerated per CLAUDE.md's demo-notebook-companion rule).

The central open technical question CONTEXT.md left to discretion — whether a partial/conditional
`UniqueConstraint` is portable across SQLite and PostgreSQL — is settled directly against the Django
5.2 docs: **yes, `UniqueConstraint(condition=Q(...))` (a partial unique index) works on both SQLite and
PostgreSQL**; only MySQL/MariaDB and Oracle lack support. This is a *different* feature from
`UniqueConstraint(include=[...])` (covering, non-key-column indexes), which really is PostgreSQL-only —
conflating the two is Pitfall 1 below. A second, independent finding closes a subtler correctness gap:
NULL is never considered equal to NULL by *any* database's unique constraint (confirmed via Django
ticket #34357), so the new TBD-branch constraint must key on `(campaign, telescope_instrument,
contact_person)` — fields that are never NULL — using `window_start IS NULL` only in the constraint's
`condition=`, never as one of the uniqued fields itself.

For the resolved-window branch (not discussed in CONTEXT.md, since it wasn't the locked recommendation
from Phase 18), this research recommends keying on `(campaign, telescope_instrument, window_start,
window_end)` rather than `window_start` alone — dropping `window_end` from the key would let a range
starting on the same day as an already-imported single night collide, a case Phase 20's CSV import will
actually produce.

**Primary recommendation:** One migration with two partial `UniqueConstraint`s (resolved-window branch
keyed on all four of `campaign`/`telescope_instrument`/`window_start`/`window_end`, condition
`window_start IS NOT NULL`; TBD branch keyed on `campaign`/`telescope_instrument`/`contact_person`,
condition `window_start IS NULL`), with the backfill (`F('obs_date')` copy) and generic dedup
(`RunPython`, ordered-by-pk Python loop, not a SQL `GROUP BY`) both running *before* the old
`RemoveConstraint`/`RemoveField` operations and *before* the two new `AddConstraint` operations.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Window schema (fields + constraints) | Database / Storage | — | Pure schema/migration change on `CampaignRun`; no view logic |
| Duplicate-row cleanup (D-07/D-08) | Database / Storage | — | A one-time `RunPython` data migration, not an app-layer job |
| Per-campaign table / approval-queue rendering (D-03/D-04/D-05) | API / Backend (Django views + django-tables2) | Browser (badge/icon CSS) | Column rendering and sort order are server-rendered (django-tables2), not client JS |
| Calendar projection (D-06) | API / Backend | — | `CampaignRunDecisionView.post()` server-side branch; no new browser code |
| Coverage-gap claimed-dates | API / Backend | Database (query shape) | `campaign_gap.claimed_dates()` is a pure-logic module reading the DB, no view concerns |
| CSV bootstrap import (`import_campaign_csv`) | API / Backend (management command) | — | Same natural-key lookup, now against `window_start` instead of `ut_start` |

## Standard Stack

### Core
No new packages this phase. `Django==5.2.15` (confirmed installed) already provides everything needed:
`models.UniqueConstraint(condition=...)`, `django.db.models.F`, `migrations.RunPython`. `django-tables2`
3.0.0 (confirmed installed) already provides the `Accessor`/`render_<field>` mechanism the existing
`campaign_tables.py` uses for dict-vs-model duality (`render_site` is the established precedent to
mirror for a new `render_window`/`render_obs_date`-style method — see Pattern 3).

### Supporting
None new.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Two partial `UniqueConstraint`s (resolved / TBD) | A single `CheckConstraint` + one plain `UniqueConstraint` including a computed/coalesced column | Django has no first-class "functional unique constraint on `Coalesce(...)`" ergonomics as clean as `condition=`; the two-partial-constraint design is the documented, idiomatic Django pattern for "different uniqueness rules for different row subsets" (used in Django's own docs example: "only one DRAFT article per user") |
| SQL `GROUP BY ... HAVING COUNT(*) > 1` for D-08's dedup query | A single ordered-by-pk Python loop over `CampaignRun.objects.filter(window_start__isnull=True).order_by('pk')`, tracking seen `(campaign_id, telescope_instrument, contact_person)` keys in a dict | The GROUP BY approach needs a second query per collision group to find the actual duplicate pks to delete (aggregate queries return grouped values, not row identities); with only ~16 total rows in dev and no reason to expect this table to be huge, the plain Python loop is simpler, easier to test, and self-documents "keep lowest pk" without extra subqueries |

**Installation:** None — no new dependencies.

**Version verification:** `python -c "import django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)`
(verified directly in this environment, matches `pyproject.toml`'s unpinned-but-installed version).
`pip show django-tables2` → `3.0.0`. `sqlite3 --version` → `3.45.1` (well above the `3.30.0` minimum
Django's SQLite backend needs for native `NULLS FIRST`/`NULLS LAST` support — see Pattern 4).

## Package Legitimacy Audit

N/A — this phase installs no new packages. `Django` and the local `sqlite3`/PostgreSQL backends are
already-installed project dependencies (see CLAUDE.md "Key Dependencies" / "Runtime"). No new
`pyproject.toml` entries are introduced by this phase's tasks.

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────────────────────┐
                     │   Migration 0004 (single file, Django 5.2)   │
                     │                                               │
  CampaignRun table  │  1. AddField window_start (nullable Date)    │
  (obs_date/         │  2. AddField window_end   (nullable Date)    │
   ut_start/ut_end   │  3. RunPython: backfill window_start/end     │──▶ every existing row now has
   still present) ──▶│     = F('obs_date') for every row            │    window_start==window_end==
                     │  4. RunPython: dedupe (campaign,              │    obs_date, OR both NULL
                     │     telescope_instrument, contact_person)     │    (TBD, if obs_date was NULL)
                     │     collisions among window_start IS NULL     │──▶ D-07/D-08 leftover fixture
                     │     rows, delete all but lowest pk, log it    │    duplicates removed
                     │  5. RemoveConstraint (old natural key)        │
                     │  6. RemoveField obs_date/ut_start/ut_end      │
                     │  7. AddConstraint (resolved-window, D-04-A)   │──▶ SCHED-04 satisfied
                     │  8. AddConstraint (TBD-window, D-04-B)        │
                     └─────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │                     Consumers (all in same phase)                  │
        │                                                                     │
        │  campaign_utils.py           import_campaign_csv.py                 │
        │  parse_obs_window() still    lookup key: (campaign,                 │
        │  returns a single exact      telescope_instrument, window_start)    │
        │  date (Phase 20 adds         instead of (..., ut_start)             │
        │  range/TBD parsing)                                                 │
        │                                                                     │
        │  campaign_gap.py             campaign_views.py                     │
        │  claimed_dates(): claim       CampaignRunDecisionView.post():       │
        │  every date in                only when window_start==window_end:  │
        │  [window_start,window_end];   ground → sun_event() dip-corrected    │
        │  undated_runs only when       night; space → midnight-UTC          │
        │  window_start is NULL         placeholder                          │
        │  (no more ut_start-derived                                         │
        │  night-boundary branch)                                            │
        │                                                                     │
        │  campaign_tables.py                                                │
        │  render_window(): TBD badge/text (D-03) if both null; "start ->    │
        │  end" (D-05) if different; single date if equal; sort key uses     │
        │  F('window_start').desc(nulls_last=True) applied in the VIEW's     │
        │  get_queryset(), table Meta.order_by left empty to preserve it     │
        │  (D-04)                                                            │
        └───────────────────────────────────────────────────────────────────┘
```

### Recommended migration file shape (solsys_code/migrations/0004_*.py)
```
solsys_code/
├── models.py                 # window_start/window_end fields, two new UniqueConstraints
└── migrations/
    └── 0004_campaignrun_window_schema.py   # single combined migration (D-02)
```

### Pattern 1: Single combined migration — field/data/constraint ordering

**What:** All of AddField, the backfill, the dedup, the old-constraint/old-field removal, and the new
constraints in one `Migration.operations` list, in a specific dependency-safe order.

**When to use:** Exactly this phase's D-02 requirement — one non-reversible migration, not a
split-into-steps sequence.

**Why this order is load-bearing (not arbitrary):** Django applies `operations` strictly in list order
against a running "project state." A `RunPython` step that reads/writes a field must come after that
field's `AddField` and before its `RemoveField`. A `RemoveField` for a field still referenced by an
existing `Meta.constraints` entry must be preceded by that constraint's `RemoveConstraint` — otherwise
the migration state is inconsistent (a constraint referencing a field the state says no longer exists).
`makemigrations` will likely auto-order `RemoveConstraint` before `RemoveField(ut_start)` correctly when
you edit `models.py` and regenerate, but it has **no idea about the `RunPython` backfill/dedup steps** —
those must be inserted by hand after generation, in exactly the position shown below.

```python
# Source: pattern derived from Django 5.2 docs (docs.djangoproject.com/en/5.2/ref/migration-operations/)
# + this repo's existing migrations/0002_campaignrun.py, 0003_campaignrun_natural_key_unique_constraint.py
import logging

from django.db import migrations, models
from django.db.models import F

logger = logging.getLogger(__name__)


def backfill_window_fields(apps, schema_editor):
    """SCHED-05: window_start=window_end=obs_date for every row (NULL stays NULL -> TBD)."""
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    CampaignRun.objects.all().update(window_start=F('obs_date'), window_end=F('obs_date'))


def dedupe_tbd_collisions(apps, schema_editor):
    """D-07/D-08: generic cleanup of (campaign, telescope_instrument, contact_person)
    collisions among window_start IS NULL rows -- keeps the lowest pk, deletes the rest,
    logs what was removed. Must run before the new TBD-branch UniqueConstraint is added
    below, or that AddConstraint would fail against real leftover-fixture duplicates.
    """
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    seen: dict[tuple, int] = {}
    # order_by('pk') is what makes "keep the lowest pk" correct: the first row seen for
    # a given key is always the lowest-pk one, so every later match is a dup to delete.
    for run in CampaignRun.objects.filter(window_start__isnull=True).order_by('pk'):
        key = (run.campaign_id, run.telescope_instrument, run.contact_person)
        if key in seen:
            logger.warning(
                'Deleting duplicate TBD CampaignRun pk=%s (kept pk=%s) for '
                'campaign=%s telescope_instrument=%r contact_person=%r',
                run.pk, seen[key], run.campaign_id, run.telescope_instrument, run.contact_person,
            )
            run.delete()
        else:
            seen[key] = run.pk


class Migration(migrations.Migration):
    dependencies = [
        ('solsys_code', '0003_campaignrun_natural_key_unique_constraint'),
    ]

    operations = [
        # 1-2: new nullable fields, added first so RunPython below can populate them.
        migrations.AddField(
            model_name='campaignrun',
            name='window_start',
            field=models.DateField(null=True, blank=True, verbose_name='Observing window start'),
        ),
        migrations.AddField(
            model_name='campaignrun',
            name='window_end',
            field=models.DateField(null=True, blank=True, verbose_name='Observing window end'),
        ),
        # 3-4: data migration, in this order (backfill before dedup -- dedup keys on
        # window_start IS NULL, which only exists after backfill runs).
        migrations.RunPython(backfill_window_fields, reverse_code=migrations.RunPython.noop),
        migrations.RunPython(dedupe_tbd_collisions, reverse_code=migrations.RunPython.noop),
        # 5: old constraint removed BEFORE the fields it references are removed.
        migrations.RemoveConstraint(model_name='campaignrun', name='unique_campaign_run_natural_key'),
        # 6: old fields dropped (D-01 hard cutover).
        migrations.RemoveField(model_name='campaignrun', name='obs_date'),
        migrations.RemoveField(model_name='campaignrun', name='ut_start'),
        migrations.RemoveField(model_name='campaignrun', name='ut_end'),
        # 7-8: new partial constraints -- see Pattern 2 for the exact condition= design.
        migrations.AddConstraint(
            model_name='campaignrun',
            constraint=models.UniqueConstraint(
                fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
                condition=models.Q(window_start__isnull=False),
                name='unique_campaign_run_resolved_window',
            ),
        ),
        migrations.AddConstraint(
            model_name='campaignrun',
            constraint=models.UniqueConstraint(
                fields=('campaign', 'telescope_instrument', 'contact_person'),
                condition=models.Q(window_start__isnull=True),
                name='unique_campaign_run_tbd_natural_key',
            ),
        ),
    ]
```

**Practical generation workflow:** edit `models.py` first (add the two fields, remove the three old
ones, replace `Meta.constraints`), run `./manage.py makemigrations solsys_code`, then hand-insert the
two `RunPython` operations into the generated file at the position shown above (`makemigrations` cannot
know about them — they don't correspond to any model-field change it can detect). Verify the generated
file's operation order matches Pattern 1 before considering the migration file done; if `makemigrations`
produced a different order (e.g. `RemoveField` before `RemoveConstraint`), reorder by hand.

### Pattern 2: Partial `UniqueConstraint` design (SCHED-03/SCHED-04)

**What:** Two `UniqueConstraint`s replace the single unconditional one, each scoped by `condition=` to a
disjoint subset of rows (resolved windows vs. TBD).

**Why two, not one:** [VERIFIED: Django 5.2 docs] `UniqueConstraint(condition=Q(...))` is a genuine
partial unique index — supported on SQLite and PostgreSQL (same restrictions as `Index.condition`; only
MySQL/MariaDB and Oracle lack support). This is the mechanism SCHED-04 asks for. But a single
`UniqueConstraint` cannot itself change *which columns* are being uniqued between two row subsets — the
TBD subset needs `contact_person` in the key (Phase 18's real JWST evidence: two distinct instrument
configs share `window_start IS NULL` + `telescope_instrument='JWST'`), while the resolved-window subset
has no such need (a resolved date/range is itself already a strong natural key). Two independent
partial constraints, each with its own `condition=` and field tuple, express this cleanly:

```python
class Meta:  # noqa: D106
    constraints = [
        # Resolved-window branch: a concrete single night or range. window_end is
        # included (not just window_start) so a range starting on the same day as an
        # existing single-night entry (e.g. Phase 20's "Aug 1-15" alongside an existing
        # lone "Aug 1") is NOT treated as the same row -- CONTEXT.md left this branch's
        # exact key composition undecided; this is this research's recommendation.
        models.UniqueConstraint(
            fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
            condition=models.Q(window_start__isnull=False),
            name='unique_campaign_run_resolved_window',
        ),
        # TBD branch: window_start/window_end are NOT in this constraint's field tuple
        # at all -- they're both NULL for every row this constraint applies to (per its
        # own condition), and NULL is never considered equal by a unique constraint on
        # any backend (confirmed via Django ticket #34357), so including them here would
        # silently defeat the whole point of this constraint. contact_person is the
        # locked Phase 18 discriminator instead (never NULL: CharField(blank=True,
        # default='')).
        models.UniqueConstraint(
            fields=('campaign', 'telescope_instrument', 'contact_person'),
            condition=models.Q(window_start__isnull=True),
            name='unique_campaign_run_tbd_natural_key',
        ),
    ]
```

**Race-safety preserved:** the existing `models.py` comment on the old constraint documents that
`insert_or_create_campaign_run`'s `get_or_create()` is "only race-safe when its lookup fields are backed
by a real DB constraint." Both new constraints are real DB-level constraints (not app-level `clean()`
validation), so this property is preserved for both branches.

### Pattern 3: Dict-vs-model dual accessor for the new "window" column (D-03/D-05)

**What:** `campaign_tables.py` needs a single rendered column that reads two underlying fields
(`window_start`, `window_end`) and must work identically whether `record` is a `CampaignRun` instance
(staff) or a restricted `.values()` dict (non-staff, per `ALLOWED_FIELDS_FOR_NON_STAFF`). This exact
dual-accessor problem is already solved in this file for `render_site()` — mirror it.

```python
# Source: pattern already established in campaign_tables.py's render_site() (read directly)
from django_tables2.utils import Accessor

window_start = tables.Column(accessor='window_start', verbose_name='Observing Window', order_by=('window_start',))

def render_window_start(self, record):
    """D-03/D-05: TBD badge/text, single date, or 'start -> end' range."""
    start = Accessor('window_start').resolve(record, quiet=True)
    end = Accessor('window_end').resolve(record, quiet=True)
    if start is None:  # both null by the model's own invariant
        return format_html('<span class="badge badge-secondary">TBD</span>')  # D-03, best effort
    if start == end:
        return start  # single-night row (D-05)
    return format_html('{} -&gt; {}', start, end)  # D-05: literal "->", not an en-dash
```

`ALLOWED_FIELDS_FOR_NON_STAFF` (`campaign_views.py`) must swap `'obs_date'`, `'ut_start'`, `'ut_end'` for
`'window_start'`, `'window_end'` — the dict path only ever has the keys explicitly listed there.

### Pattern 4: Cross-backend nulls-last default sort (D-04)

**What:** [VERIFIED: Django 5.2 docs / SQLite 3.45.1 installed] `queryset.order_by(F('window_start')
.desc(nulls_last=True))` is the Django-native, cross-backend way to put NULL (TBD) rows last regardless
of whether the DB backend's own default NULL-ordering convention would otherwise put them first (SQLite)
or last (PostgreSQL's own default already differs from SQLite's here, which is exactly why relying on
implicit backend behavior would be non-portable). SQLite 3.45.1 (confirmed installed, well above the
3.30.0 threshold Django's backend requires) supports the native `NULLS LAST` SQL syntax, so no
CASE-expression emulation is even needed here.

**Why this can't live in `campaign_tables.py`'s `Meta.order_by`:** django-tables2's `Meta.order_by` only
accepts bare accessor strings (e.g. `'-window_start'`), which compile to a plain
`queryset.order_by('-window_start')` with the *backend's* default NULL placement — not an `F()`
expression. The fix is applied one layer up, in the view:

```python
# Source: pattern already established in campaign_views.py's ApprovalQueueView (read
# directly) -- decided_table is built from a plain sorted list and passed order_by=()
# specifically so django-tables2 does NOT re-sort it and clobber the pre-established order.
from django.db.models import F

class CampaignRunTableView(SingleTableMixin, FilterView):
    def get_queryset(self):
        qs = CampaignRun.objects.filter(campaign_id=campaign_pk)
        ...
        return qs.order_by(F('window_start').desc(nulls_last=True))

    def get_table_kwargs(self):
        # in addition to the existing PII-exclude kwarg:
        return {..., 'order_by': ()}  # don't let the table's own default sort override this
```

Interactive user-driven sort (clicking a column header) still works normally — `RequestConfig` applies
the `sort=` query param's plain string-based ordering on top, which only affects rows when the user
explicitly asks for a different sort; the nulls-last guarantee only applies to the *default* (D-04's
actual scope).

### Anti-Patterns to Avoid
- **Relying on SQLite/PostgreSQL's implicit default NULL-ordering direction for D-04** instead of
  `F(...).desc(nulls_last=True)`: the two backends default to *opposite* directions for descending sort
  (SQLite: NULLs sort as the smallest value, so `DESC` puts them last automatically today; PostgreSQL:
  `DESC` defaults to `NULLS FIRST`). Code that happens to "work" against dev SQLite would silently
  invert TBD-row placement on a PostgreSQL production database (CLAUDE.md notes PostgreSQL as the
  intended production target).
- **Putting `window_start`/`window_end` inside the TBD branch's uniqued `fields=` tuple**: since both
  are always NULL for every row the TBD constraint's `condition=` selects, including them there does
  nothing (NULL never conflicts with NULL) and creates a false impression the constraint is doing more
  than it is — this is the exact NULL-uniqueness gap SCHED-04 exists to close, and it would still be
  open if implemented this way.
- **Using `UniqueConstraint(include=[...])` thinking it's the "SQLite/PostgreSQL-portable partial
  constraint" mechanism**: `include=` (covering, non-key-column indexes) really is PostgreSQL-only per
  Django's own docs — a genuinely different feature from `condition=` (partial). Conflating the two
  under casual reading of "conditional constraint" language is an easy mistake (see Pitfall 1).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-database partial/conditional uniqueness | A custom `pre_save`/`clean()`-based duplicate check, or raw SQL `CREATE UNIQUE INDEX ... WHERE ...` in a `RunSQL` migration operation | `models.UniqueConstraint(condition=Q(...))` | Already portable across SQLite/PostgreSQL at the Django ORM layer (verified); app-level `clean()` checks are not race-safe (two concurrent requests can both pass validation before either commits) — the existing `models.py` comment on the old constraint already documents this exact race-safety rationale for the codebase |
| Nulls-last default sort | A custom Python-level `sorted(qs, key=...)` post-processing step after fetching all rows | `queryset.order_by(F('field').desc(nulls_last=True))` | Keeps sorting in the DB (works with `SingleTableMixin`'s pagination, which needs a DB-level `ORDER BY` + `LIMIT`, not a Python-level full materialization) |
| Dict-vs-model-instance duality in table columns | A branch (`if isinstance(record, dict): ... else: ...`) in every new render method | `django_tables2.utils.Accessor(...).resolve(record, quiet=True)` | Already the established, working pattern in this exact file (`render_site`, `render_run_status`, `render_approval_status`) — reuse, don't reinvent |

**Key insight:** every "how do I handle this" question this phase raises (partial uniqueness, nulls-last
sort, dict/model duality) already has an established Django or in-repo pattern; the risk in this phase
is not missing tooling, it's getting the *ordering* of migration operations and the *scope* of each
partial constraint's `condition=` subtly wrong.

## Runtime State Inventory

This phase is a database schema + data migration — the Runtime State Inventory protocol applies.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | The `CampaignRun` table itself is the primary stored data being migrated — handled directly by the migration's `AddField`/`RunPython`/`RemoveField` steps (D-02/D-07/D-08). No other model/table stores a copy of `obs_date`/`ut_start`/`ut_end` values. | Data migration (already planned in Pattern 1) — no additional action beyond the migration itself. |
| Live service config | None found. No external service (LCO/Gemini/ESO calendar sync, MPC API) stores or references `CampaignRun.obs_date`/`ut_start`/`ut_end` — those sync commands write to `CalendarEvent`, a separate model untouched by this phase. | None. |
| OS-registered state | None found. No cron/Task-Scheduler/pm2 registration references these field names. | None. |
| Secrets/env vars | None found. No SOPS key, `.env` var, or CI secret is named after `obs_date`/`ut_start`/`ut_end`. | None. |
| Build artifacts | **Found:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` line 309 contains `CampaignRun.objects.filter(campaign=campaign).order_by('obs_date', 'ut_start')` — a committed, pre-executed notebook cell that will `FieldError` after this phase's `RemoveField` operations. This notebook is **not on CONTEXT.md's 15-file list** but is squarely covered by CLAUDE.md's demo-notebook-companion rule (`import_campaign_csv.py`'s paired notebook must stay in sync with behavior changes). | Update the `.order_by(...)` call to reference `window_start`/`window_end`, then regenerate via `jupyter nbconvert --to notebook --execute --inplace` and commit with real executed output (per CLAUDE.md convention) — add this notebook to the plan's `files_modified`. |

**Transient/low-risk item worth a one-line mention (not a required action):** `campaign_gap.py`'s 1-hour
result cache (`GAP_CACHE_TTL_SECONDS`) can hold pickled `CampaignRun` instances fetched via
`.only('pk', 'obs_date', 'ut_start')` from *before* this migration runs. In production this is Django's
low-level cache framework (likely Redis/Memcached with a real TTL) — any entry created before the
migration naturally expires within an hour and is never read past that; no explicit cache-flush step is
required, but the executor should be aware a manual `cache.clear()` is a reasonable belt-and-suspenders
step if deploying this migration outside a maintenance window.

## Common Pitfalls

### Pitfall 1: Confusing `include=` (covering, Postgres-only) with `condition=` (partial, cross-backend)
**What goes wrong:** Assuming any "extra" `UniqueConstraint` keyword argument is PostgreSQL-only, and
concluding SCHED-04's portability requirement can't be met without `RunSQL`/backend-conditional
migration branches.
**Why it happens:** Django's own docs page lists both `include=` and `condition=` on the same
`UniqueConstraint` class, and a casual grep/skim finds "ignored for databases besides PostgreSQL" text
that actually refers to `include=`, not `condition=`.
**How to avoid:** Use `condition=Q(...)` only (never `include=`) for this phase's constraints — verified
directly against Django 5.2 docs that `condition=` partial constraints are supported on SQLite and
PostgreSQL (same restrictions as `Index.condition`).
**Warning signs:** Migration works fine locally (SQLite) but a code review or docs skim raises "isn't
this Postgres-only?" — resolve by checking which keyword argument (`include` vs `condition`) is actually
in use before assuming the two share a portability story.

### Pitfall 2: Migration operation order silently breaking mid-migration
**What goes wrong:** `makemigrations` auto-generates operations in *an* order, but if the two
hand-inserted `RunPython` steps land in the wrong position (e.g. after `RemoveField(obs_date)`), the
backfill's `F('obs_date')` reference raises `FieldDoesNotExist` against the historical model state at
that point in the operations list.
**Why it happens:** `RunPython`'s `apps.get_model(...)` returns the model as it exists in project state
*at that point in the operations list*, not the final state — this is easy to get backwards when
hand-editing a generated migration.
**How to avoid:** Follow Pattern 1's exact operation order; after generating+editing the migration, run
`./manage.py migrate solsys_code` against a copy of the dev DB (or a throwaway SQLite file) and confirm
it succeeds end-to-end before treating the migration as done.
**Warning signs:** `FieldDoesNotExist` or `FieldError` raised during `manage.py migrate`, referencing a
field that should still exist "logically" but has already been removed in the operations list executed
so far.

### Pitfall 3: The dedup migration running *after* the new TBD constraint is added
**What goes wrong:** If `AddConstraint` for the TBD-branch constraint runs before
`dedupe_tbd_collisions`, the `AddConstraint` operation itself fails with an `IntegrityError` against the
real, already-known leftover duplicate rows (Phase 18's D-07 evidence: pk 15/17 and pk 16/18 in the dev
DB) — the migration cannot even complete.
**Why it happens:** Easy to mentally file "add the new constraint" as the "last, obvious step" and
forget the dedup must strictly precede it, not just precede the `RemoveField` steps.
**How to avoid:** Follow Pattern 1's ordering exactly: both `RunPython` steps come before both
`RemoveConstraint`/`RemoveField` **and** both new `AddConstraint` operations.
**Warning signs:** `django.db.utils.IntegrityError: UNIQUE constraint failed` raised specifically during
the `AddConstraint` step when running `manage.py migrate` against the real dev DB (which does contain
the known D-07 duplicates) — this is exactly what a smoke-test migrate run against a dev-DB copy should
catch before the plan is considered done.

### Pitfall 4: Orphaned imports/dead code left behind in `campaign_gap.py`
**What goes wrong:** `_observing_night_date()` (the local-noon-anchored `ut_start`-to-observing-night
helper) has no reason to exist once `ut_start` is gone — `window_start`/`window_end` are already plain
dates, no time-of-day-to-night-boundary conversion is needed. If this helper (and its now-unused
`ZoneInfo`/`time` imports) are left in place "just in case," `ruff check .` fails on `F401` unused
imports, and `_observing_night_date` becomes untested dead code the plan-checker/reviewer would flag.
**Why it happens:** Deleting a whole private helper function feels riskier than leaving it; easy to miss
during a field-rename-focused pass.
**How to avoid:** When rewriting `claimed_dates()` to iterate `[window_start, window_end]` directly (no
per-run night-derivation needed anymore), delete `_observing_night_date()` and its now-dead imports in
the same task, and run `ruff check .` as part of that task's verification.
**Warning signs:** `ruff check .` reports `F401 'zoneinfo.ZoneInfo' imported but unused` (or similar)
after the `campaign_gap.py` rewrite.

### Pitfall 5: Existing tests asserting old-field-specific behavior that no longer exists
**What goes wrong:** Several existing tests assert behavior that the new schema makes structurally
impossible or meaningless, not just renamed — naively renaming field references in these tests would
leave them asserting something false or untestable:
- `test_campaign_gap.py::TestClaimedDates::test_ut_start_only_keys_to_site_local_observing_night` —
  tests the `_observing_night_date()` fallback path directly; this whole code path is deleted per
  Pitfall 4, so this test must be deleted (or replaced with a window-range-claiming test), not renamed.
- `test_campaign_gap.py::TestClaimedDates::test_approved_completed_run_claimed_via_obs_date` — needs
  rewriting to construct a `window_start`/`window_end` pair instead of `obs_date`.
- `test_campaign_views.py::test_default_sort_is_obs_date_descending` — needs rewriting for D-04's
  nulls-last window-based sort, and should add a new case asserting a TBD row sorts after every resolved
  row (the actual behavior change, not just a field rename).
- `test_import_campaign_csv.py::test_duplicate_unparseable_ut_time_rows_do_not_merge` and
  `test_idempotent_rerun_no_duplicates` — depend on the natural key including a fallback timestamp
  (`ut_start`); since `window_start` is date-only (no time-of-day fallback distinction possible anymore),
  these tests' *scenario*, not just field names, needs re-thought for the new key shape.
**Why it happens:** A rename-shaped diff (find `obs_date`, replace `window_start`) looks complete but
silently preserves assertions about mechanics that were deleted, or scenarios the new schema can't
reproduce the same way.
**How to avoid:** Treat each of these 4 tests as requiring a design decision, not a mechanical
find-replace; read what each test is actually verifying before touching it.
**Warning signs:** A test passes after a naive rename but its assertion no longer exercises anything
meaningful (e.g. asserting a fallback path that can't be reached anymore).

### Pitfall 6: Ambiguous `sun_event()` `kind` choice for D-06's "dip-corrected dark-window banner"
**What goes wrong:** CONTEXT.md's D-06 phrase "a real, dip-corrected dark-window banner" conflates two
different `sun_event()` invocations that exist in this codebase for different purposes: `kind='sun'`
(dip-corrected sunset/sunrise, i.e. the *observing night* boundary) and `kind='dark'` (fixed -15°
threshold, NOT dip-corrected, used elsewhere in this codebase for *observability*, e.g.
`campaign_gap.observable_dates()`). Picking the wrong one produces a CalendarEvent with the wrong
duration (the dark window is narrower than the full observing night).
**Why it happens:** The phrase literally says both "dip-corrected" (true only of `kind='sun'`) and
"dark-window" (the *name* used for `kind='dark'`, which isn't dip-corrected at all) in the same
sentence.
**How to avoid:** This research recommends `kind='sun'` for the `CalendarEvent` `start_time`/`end_time`
— it is the literally dip-corrected variant, and it matches the established convention in
`load_telescope_runs.py` (read directly), where classical-run `CalendarEvent`s are bounded by
`sun_event(site, d, 'sun')`'s sunset/sunrise, with `kind='dark'` used only for descriptive text, never
as the event's own start/end. This is tagged `[ASSUMED]` in the Assumptions Log below since CONTEXT.md's
exact wording is ambiguous and this wasn't explicitly re-confirmed with the user.
**Warning signs:** UAT/manual verification of a projected ground-based `CalendarEvent`'s duration looks
noticeably shorter than the site's actual observing night.

### Pitfall 7: `sun_event()` can raise `ValueError` — the D-06 projection must not crash on it
**What goes wrong:** `sun_event()` raises `ValueError` for an unset `site.timezone`, an invalid `kind`,
or a site/date combination with no 2 sun-altitude crossings (e.g. a hypothetical polar site). If the
new D-06 branch calls it unguarded, a single bad site record turns a routine approve-click into an
unhandled 500 (or, worse per the existing `CampaignRunDecisionView.post()`'s `except Exception:` catch,
silently reverts the whole approval back to `PENDING_REVIEW` with a generic error message that obscures
the real cause).
**Why it happens:** `campaign_gap.py`'s `observable_dates()` already has an established log+skip
discipline for exactly this exception; a new call site can easily omit it.
**How to avoid:** Wrap the `sun_event()` call in the same `except ValueError: log+skip` discipline
`observable_dates()` already uses, rather than letting it propagate into the view's broad
`except Exception:` handler (which exists for a *different* purpose — reverting a half-committed
approval — not general error handling for expected messy `Observatory` data).
**Warning signs:** Approving a run for a site with `timezone=''` (blank) unexpectedly reverts the
approval and logs a generic "Approve side-effects failed" message instead of a specific one.

## Code Examples

### `campaign_gap.py` — `claimed_dates()` rewrite (no more `ut_start`/night-boundary derivation)
```python
# Source: this repo's existing observable_dates() loop shape (read directly), applied to
# the claimed side now that window_start/window_end are already plain dates.
def claimed_dates(campaign, target, site) -> tuple[set[date], list, list]:
    qs = CampaignRun.objects.filter(campaign=campaign, site=site, approval_status=CampaignRun.ApprovalStatus.APPROVED)
    qs = qs.exclude(run_status__in=_EXCLUDED_RUN_STATUSES)
    qs = qs.only('pk', 'window_start', 'window_end')  # D-13/WR-01: still PII-free

    unattributed_runs: list[CampaignRun] = []
    single_target = campaign.targets.count() == 1
    if not single_target:
        unattributed_runs = list(qs.filter(target__isnull=True))
        qs = qs.filter(target=target)

    claimed: set[date] = set()
    undated_runs: list[CampaignRun] = []
    for run in qs:
        if run.window_start is None:
            undated_runs.append(run)  # TBD -- can't be attributed to any date (unchanged bucketing rule)
            continue
        n_days = (run.window_end - run.window_start).days + 1
        for i in range(n_days):
            claimed.add(run.window_start + timedelta(days=i))

    return claimed, undated_runs, unattributed_runs
```
Note: this claims every date in a space-mission run's window too — the ground-vs-space distinction
(ASSET-02) is explicitly Phase 20's job per CONTEXT.md's phase-boundary note; Phase 19 only needs
`claimed_dates()` to keep working against the new fields, not to add asset-awareness.

### `campaign_views.py` — D-06 hybrid calendar projection
```python
# Source: sun_event()'s to_datetime() conversion pattern from load_telescope_runs.py
# (read directly, line 40/136-137); Observatory.SATELLITE_OBSTYPE confirmed as a class
# attribute in solsys_code_observatory/models.py (read directly, line 19).
from datetime import datetime, time as dt_time
from datetime import timezone as dt_timezone

from solsys_code.telescope_runs import sun_event

if run.telescope_instrument and run.site and run.window_start and run.window_start == run.window_end:
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        start_time = datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc)
        end_time = datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc)
        insert_or_create_calendar_event(..., fields={..., 'start_time': start_time, 'end_time': end_time, ...})
    else:
        try:
            sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')  # see Pitfall 6
        except ValueError:
            logger.debug('sun_event(sun) raised for site=%s date=%s; skipping projection.', run.site, run.window_start)
        else:
            start_time = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
            end_time = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
            insert_or_create_calendar_event(..., fields={..., 'start_time': start_time, 'end_time': end_time, ...})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `obs_date` (single `DateField`) + `ut_start`/`ut_end` (`DateTimeField`) | `window_start`/`window_end` (nullable `DateField` pair) | This phase (SCHED-02/03) | Represents a single night (`start==end`), a range, or fully-TBD (both null) in one pair instead of three fields with an implicit "TBD means blank" convention |
| `UniqueConstraint(fields=(campaign, telescope_instrument, ut_start))`, unconditional | Two `UniqueConstraint`s, each `condition=`-scoped to resolved vs. TBD rows | This phase (SCHED-04) | Closes the silent-duplicate-TBD-row gap Phase 18 found real evidence for (two distinct JWST rows) |
| `claimed_dates()` deriving a night from `ut_start` via `_observing_night_date()`'s local-noon convention | `claimed_dates()` iterating `[window_start, window_end]` directly (both already plain dates) | This phase | Removes a whole helper function and its `ZoneInfo`/local-noon logic — no longer needed once there's no time-of-day component to convert |

**Deprecated/outdated:**
- `_observing_night_date()` (`campaign_gap.py`): no longer needed once `ut_start` is gone — see Pitfall
  4.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sun_event(site, window_start, kind='sun')` (not `kind='dark'`) is the correct call for D-06's ground-based `CalendarEvent` boundary. | Pitfall 6 / Code Examples | If the user actually intended the -15° dark window as the event's own start/end (not just descriptive text), the projected `CalendarEvent` would be visibly narrower than this research's recommendation produces — a quick UAT check of one projected event's duration against LCO/site sunset-sunrise times would catch this immediately. |
| A2 | The resolved-window branch's `UniqueConstraint` should key on `(campaign, telescope_instrument, window_start, window_end)`, not `window_start` alone. | Pattern 2 / Summary | If `window_end` is omitted, a Phase-20-imported range starting the same day as an existing single-night row would incorrectly collide as a duplicate natural key, blocking a legitimate import. |
| A3 | The public `CampaignRunSubmissionForm`'s `ut_start`/`ut_end` `DateTimeField` inputs have no home in the new schema and should simply be dropped (form collapses to a single date field feeding both `window_start`/`window_end` on save), rather than repurposed some other way. | Open Questions #1 | If the user actually wants sub-day time-of-night detail preserved somewhere (e.g. appended into `comments` as free text), a plan built on this assumption would need a follow-up quick task to add it back. |

**If this table is empty:** N/A — see entries above; all three should be quickly sanity-checked with
the user or via a UAT smoke-check before or during planning, but none blocks starting the plan (each has
a clear, reasonable default this research recommends).

## Open Questions

1. **What happens to the public submission form's UT start/end time fields?**
   - What we know: `CampaignRunSubmissionForm` currently has `obs_date` (`DateField`), `ut_start`/
     `ut_end` (`DateTimeField`) inputs; the new model has only `window_start`/`window_end` (`DateField`,
     no time component at all). D-01's hard cutover means these three model fields are gone regardless.
   - What's unclear: CONTEXT.md's file list requires `campaign_forms.py` to be updated (it's one of the
     15 files), but doesn't specify the new form's exact field shape — whether time-of-night detail is
     dropped entirely, or preserved as free text somewhere (e.g. folded into `comments`).
   - Recommendation: collapse to a single `obs_date`-style `DateField` (renamed `window_start` or kept
     as `obs_date` on the form with the view mapping it to both `window_start`/`window_end` on save,
     single-night collapse per SCHED-02), and drop the two `DateTimeField` inputs entirely — this is the
     simplest, most-consistent-with-D-01 interpretation (tagged `[ASSUMED]`, A3 above). Flag for a quick
     confirmation with the user if the planner wants to be certain before committing to this shape.

## Environment Availability

No new external dependencies. `Django 5.2.15`, `sqlite3 3.45.1` (dev), and `django-tables2 3.0.0` are
already installed and confirmed working in this environment (`python -c "import django; print(...)"`,
`sqlite3 --version`, `pip show django-tables2`, all run directly this session). Production PostgreSQL
compatibility for the new partial constraints is verified against Django's documented cross-backend
`condition=` support (see Summary/Pattern 2), not against a live PostgreSQL instance in this environment
— no PostgreSQL server is running here to test against directly, so this remains a documentation-based
(not live-tested) verification for the PostgreSQL side specifically.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Django | Migration framework, ORM constraints | ✓ | 5.2.15 | — |
| SQLite3 | Dev DB backend | ✓ | 3.45.1 | — |
| django-tables2 | Table column rendering (D-03/D-04/D-05) | ✓ | 3.0.0 | — |
| PostgreSQL | Production DB backend (SCHED-04's portability target) | Not running in this dev environment | — | Portability verified via Django docs only (not live-tested here); recommend a one-time manual smoke-test migration run against a throwaway PostgreSQL instance before shipping to production, if/when this app is deployed there |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** PostgreSQL live verification (see table) — documentation-based
verification stands in for a live test in this environment.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` (`unittest`-style), run via `./manage.py test` |
| Config file | none (Django test runner config lives in `src/fomo/settings.py`'s `TEST_RUNNER` default; no separate pytest config applies to this app's tests — CLAUDE.md: `pyproject.toml`'s `testpaths` deliberately excludes `solsys_code/`) |
| Quick run command | `./manage.py test solsys_code.tests.test_campaign_models` (fastest relevant single-module check) |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirement → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|-------------|
| SCHED-02 | `CampaignRun` savable with `window_start == window_end` (single night) | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ✅ (rewrite `TestCampaignRunFieldInventory`) |
| SCHED-03 | `CampaignRun` savable fully TBD (both window fields null) | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ✅ Wave 0 — needs a new test case, none currently asserts the TBD state explicitly |
| SCHED-04 | Two TBD rows for same campaign+telescope+contact_person collide (IntegrityError); two TBD rows with different contact_person do not | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ✅ Wave 0 — new test case needed (no current test exercises the new partial constraints) |
| SCHED-05 | Migration backfill: every pre-existing row survives with `window_start == window_end == former obs_date` | migration / data | `./manage.py test solsys_code.tests.test_campaign_models` (a `TestCase` that runs against post-migration state) or a dedicated migration test using `django.test.TransactionTestCase` + `MigrationRecorder` if the plan wants to exercise the migration mechanics directly | ❌ Wave 0 — no existing migration-testing convention in this repo; a straightforward `TestCase`-level check against the final schema (create rows with the old shape isn't possible post-migration, so this is really best verified via a manual `manage.py migrate` dry run against a DB copy plus a `TestCase` asserting the model's current field set, not a full migration-replay test) |
| Table/queue rendering (D-03/D-04/D-05) | TBD badge, `->` range display, nulls-last default sort | unit + view | `./manage.py test solsys_code.tests.test_campaign_views` | ✅ (rewrite `test_default_sort_is_obs_date_descending`; add TBD-badge and range-display cases) |
| Calendar projection (D-06) | Ground vs. space branch produces correct `CalendarEvent` window | unit | `./manage.py test solsys_code.tests.test_campaign_approval` | ✅ (existing calendar-projection tests need window-field rewrite + new space-branch case) |
| Coverage-gap `claimed_dates()` rewrite | Every date in `[window_start, window_end]` claimed; TBD → undated bucket | unit | `./manage.py test solsys_code.tests.test_campaign_gap` | ✅ (rewrite `TestClaimedDates`; delete or replace `test_ut_start_only_keys_to_site_local_observing_night` per Pitfall 5) |
| CSV import lookup key (`import_campaign_csv.py`) | Natural-key lookup uses `window_start` not `ut_start` | unit | `./manage.py test solsys_code.tests.test_import_campaign_csv` | ✅ (rewrite `test_duplicate_unparseable_ut_time_rows_do_not_merge`, `test_idempotent_rerun_no_duplicates` per Pitfall 5) |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_campaign_models` (fast, no heavy SPICE
  import — `campaign_gap`/`campaign_views`/`campaign_utils`/`campaign_tables` all deliberately avoid
  importing `solsys_code.views`/`ephem_utils` per CLAUDE.md's documented heavy-import constraint).
- **Per wave merge:** `./manage.py test solsys_code` (full app suite) plus `ruff check .` / `ruff format
  --check .` (project quality gates, per CLAUDE.md and this phase's Constraints).
- **Phase gate:** Full `./manage.py test solsys_code` green, plus a manual `manage.py migrate` dry run
  against a copy of the real dev DB (to actually exercise the D-07/D-08 dedup against the known real
  duplicate rows) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] No existing test asserts `CampaignRun` can be saved fully TBD (both `window_start`/`window_end`
  null) — needed for SCHED-03.
- [ ] No existing test exercises either new partial `UniqueConstraint` directly (a same-key TBD
  collision raising `IntegrityError`; a same-key resolved-window collision raising `IntegrityError`; two
  TBD rows differing only in `contact_person` both saving successfully) — needed for SCHED-04.
- [ ] No migration-level test/dry-run convention exists in this repo yet for verifying `RunPython`
  backfill+dedup correctness against realistic pre-migration data shapes (recommend a throwaway-SQLite-copy
  `manage.py migrate` dry run as the practical substitute, documented above, rather than introducing a
  new `TransactionTestCase`-based migration-testing framework for a single phase).

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|----------------|---------|-------------------|
| V1 Architecture | yes | Migration + constraint changes stay entirely inside Django's ORM/migration framework — no raw SQL (`RunSQL`) introduced; no new trust boundary crossed |
| V5 Input Validation | yes (unchanged surface) | `DateField` itself rejects malformed dates at the Django form/model layer; no new free-text parsing is added by this phase (Phase 20 adds range/TBD text parsing, not this phase) |
| V11 Business Logic | yes | The new partial `UniqueConstraint`s are the enforcement mechanism against a real race condition (two concurrent submissions/imports both passing an app-level uniqueness check before either commits) — this mirrors the existing `models.py` comment's rationale for the *old* constraint's race-safety property, now extended to both new constraints |
| V4 Access Control | no change | Approval-queue staff-only gating (`StaffRequiredMixin`) and PII-restricted non-staff queryset (`ALLOWED_FIELDS_FOR_NON_STAFF`) are unchanged in shape by this phase — only the field *names* inside that already-existing allowlist change (`obs_date`/`ut_start`/`ut_end` → `window_start`/`window_end`) |
| V6 Cryptography | no | Not applicable — no crypto surface touched |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|------------------------|
| Race condition creating duplicate/colliding rows under concurrent writes | Tampering / Repudiation (data integrity) | DB-level `UniqueConstraint` (both branches) — already the established pattern in this codebase (see the old constraint's own docstring rationale); this phase extends, not introduces, this control |
| PII leakage via a newly-added table column bypassing the existing allowlist | Information Disclosure | `ALLOWED_FIELDS_FOR_NON_STAFF` is an explicit enumerated list (not introspected from `CampaignRun._meta`) specifically so a new field can never accidentally leak to non-staff by omission — this phase must update the list's *contents* (swap 3 old names for 2 new ones) but must not change its *shape* (still an explicit allowlist, never auto-derived) |
| Destructive, irreversible data migration deleting rows without an audit trail | Repudiation | D-08 already requires the dedup step to log what it removed (pk, campaign, telescope_instrument, contact_person) before deleting — satisfies a minimal audit-trail requirement for a one-time, non-interactive data-cleanup operation; no additional control needed beyond what D-08 already specifies |

## Sources

### Primary (HIGH confidence — verified directly in this environment)
- Direct reads of `solsys_code/models.py`, `campaign_utils.py`, `campaign_gap.py`, `campaign_views.py`,
  `campaign_tables.py`, `campaign_forms.py`, `management/commands/import_campaign_csv.py`,
  `telescope_runs.py`, `management/commands/load_telescope_runs.py`,
  `solsys_code_observatory/models.py`, `migrations/0002_campaignrun.py`,
  `migrations/0003_campaignrun_natural_key_unique_constraint.py` — full current implementation.
- `python -c "import django; print(django.VERSION)"` → `(5, 2, 15, 'final', 0)`.
- `sqlite3 --version` → `3.45.1`; `pip show django-tables2` → `3.0.0`.
- `grep` sweeps confirming CONTEXT.md's 15-file list is complete (plus the one missed notebook), and
  that `campaign_filters.py`/`admin.py`/`src/templates/` have no direct field references needing
  changes.
- Direct read of `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` line 309
  (`.order_by('obs_date', 'ut_start')`).

### Secondary (MEDIUM confidence — WebSearch/WebFetch verified against official Django docs this session)
- [Django 5.2 Constraints reference](https://docs.djangoproject.com/en/5.2/ref/models/constraints/) —
  `UniqueConstraint.condition`/`include` distinction.
- [Django 5.2 Indexes reference](https://docs.djangoproject.com/en/5.2/ref/models/indexes/) —
  `Index.condition` per-backend support table (SQLite/PostgreSQL yes; MySQL/MariaDB/Oracle no).
- [Django ticket #34357](https://code.djangoproject.com/ticket/34357) — NULL-vs-unique-constraint
  behavior confirmed as standard SQL semantics, not an Django/SQLite-specific bug.
- [Django 5.2 Migration Operations reference](https://docs.djangoproject.com/en/5.2/ref/migration-operations/)
  — operation-ordering semantics for `AddField`/`RemoveField`/`AddConstraint`/`RemoveConstraint`/
  `RunPython`.
- [Django 5.2 Query Expressions reference](https://docs.djangoproject.com/en/5.2/ref/models/expressions/)
  — `F(...).desc(nulls_last=True)` syntax.

### Tertiary (LOW confidence — none used; all package/mechanism claims above were either read directly
from this repo or confirmed against official Django documentation this session)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; exact installed versions confirmed directly this session.
- Migration mechanics / partial constraint design: HIGH — verified directly against Django 5.2 official
  docs and this repo's own prior migration files as precedent.
- Display/UX shape (badge styling, exact TBD wording): MEDIUM — CONTEXT.md D-03 itself says "best
  effort... fall back to plain text if it doesn't work out," so this is intentionally left flexible.
- `sun_event()` `kind` choice for D-06 (Pitfall 6/A1) and submission-form field mapping (Open Question
  1/A3): MEDIUM-LOW — reasonable, precedent-backed recommendations, but not explicitly re-confirmed with
  the user against CONTEXT.md's slightly ambiguous wording.

**Research date:** 2026-07-09
**Valid until:** 30 days (Django/django-tables2 APIs used here are stable, non-fast-moving; the phase's
own dev-DB duplicate-row evidence (D-07) is a point-in-time snapshot that could go stale sooner if the
dev DB is reseeded before this phase is executed).
