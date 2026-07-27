# Stack Research

**Domain:** Django ORM patterns for linking a first-party model to pip-installed third-party models, and idempotent-reconciler design — FOMO v2.2 "One Canonical Run Record"
**Researched:** 2026-07-26
**Confidence:** HIGH (grounded in this repo's already-installed package source for `tom_calendar`/`tom_observations`, this repo's own existing sidecar/no-churn code, and Django's stable migration-operation semantics; MEDIUM on the small set of Django `RenameModel` edge-case tickets pulled via web search)

## Headline Finding

**No new dependency is warranted for v2.2.** Every piece of this milestone — the companion-record generalisation, the `ObservationRecord` linkage, the reconciler, and its idempotency tests — is built from Django's own ORM (`ForeignKey`, `ManyToManyField` with a custom `through`, `migrations.RenameModel`/`AddField`) plus the ecosystem already installed for this project (Django 5.2.13 via `tomtoolkit==3.0.0a9`, `django.test.utils.CaptureQueriesContext`). This section documents the *techniques*, not new packages, because that is what actually blocks or unblocks the roadmap here.

This is a **milestone addendum**, not a full project stack. Prior milestones' STACK.md findings (e.g. v2.1's `rapidfuzz` for site fuzzy-matching) remain in force and are not re-litigated; they're referenced only where directly relevant (see "What NOT to Use").

## Recommended Stack

### Core Technologies (already installed — no action needed)

| Technology | Version (installed) | Purpose in v2.2 | Why |
|------------|---------|---------|-----------------|
| Django | 5.2.13 | ORM relations, migrations, admin, test framework | Pinned transitively via `tomtoolkit>=2.31.4` (currently `3.0.0a9`); `ForeignKey`/`ManyToManyField(through=...)`/`RenameModel`/`RenameField`/`CaptureQueriesContext` have been stable, unchanged APIs since well before Django 4, so nothing here is Django-5.2-specific or at risk from the project's 3.10–3.12 / Django-2.1-floor compatibility window. |
| `tom_calendar` (bundled in `tomtoolkit`, not separately pip-pinned) | ships inside `tomtoolkit==3.0.0a9` | Owns `CalendarEvent` (plain `AutoField` PK, no custom manager) | Confirmed by reading the installed `tom_calendar/models.py` directly — `CalendarEvent` has no hooks for us to attach to except a reverse relation, which is exactly what the existing `CalendarEventTelescopeLabel` sidecar already does. |
| `tom_observations` (bundled in `tomtoolkit`) | ships inside `tomtoolkit==3.0.0a9` | Owns `ObservationRecord` (plain `AutoField` PK) | Confirmed by reading the installed `tom_observations/models.py` — same shape: a plain model with no attachment point of its own, so the link must live on FOMO's side, same constraint the milestone context already states. |

### Supporting Libraries

**None added.** See "What NOT to Use" below for the specific libraries that were considered and rejected, and why.

### Development Tools (already in use, extended not replaced)

| Tool | Purpose | Notes |
|------|---------|-------|
| `django.test.utils.CaptureQueriesContext` | Prove the reconciler is idempotent (no queries / no writes on a repeat run) | Already used exactly this way in `solsys_code/tests/test_calendar_template.py:272-289` (`test_display09_query_count_bounded`) — that test asserts query *count* doesn't grow; the reconciler idempotency test extends the same tool to assert query *content* (no `INSERT`/`UPDATE` SQL) on the second pass. Do not introduce a separate assertion library for this — see Testing section below. |
| `ruff` | Lint/format (single quotes, 120 cols) | No config changes needed; the new companion model, through-model, and reconciler command are ordinary Python. |

## Installation

No installation required — no new runtime or dev dependency is added by this milestone. If a future phase discovers a genuine need, treat that as a signal to re-examine the design rather than a default to reach for — this milestone's own explicit prior ("strong prior: plain ORM") held up under research.

---

## Django Pattern 1 — Generalising `CalendarEventTelescopeLabel` (sidecar OneToOne, kept)

### The decision already made (from PROJECT.md, not re-litigated here)

The milestone context states the companion record **stays a one-to-one sidecar** on `CalendarEvent` (unchanged shape) and gains a nullable `run` FK to `CampaignRun`. This is the right call, and it's worth stating precisely *why*, because the alternatives (`ManyToManyField`, `GenericForeignKey`) were live options and the milestone context correctly rejected both implicitly:

| Approach | Querying | Prefetching | Admin | Cascade behaviour | Verdict for the `CalendarEvent` link |
|----------|----------|-------------|-------|--------------------|----------------------------------------|
| **OneToOne sidecar** (`event = OneToOneField(CalendarEvent, primary_key=True, on_delete=CASCADE)`) — what's already there | Real SQL JOIN, indexed on the shared PK; `event.telescope_label_meta` and `label.event` both single-query | `.prefetch_related('telescope_label_meta')` is O(1) extra query regardless of event count — already proven in production by DISPLAY-09 (`solsys_code/views.py:114`) | Trivial `ModelAdmin`, `list_filter`/`search_fields` work natively — already registered (`solsys_code/admin.py:28-31`) | `on_delete=CASCADE` on `event` (deleting a `CalendarEvent` correctly deletes its companion row — no orphan); the *new* `run` FK should be `on_delete=SET_NULL` (deleting a `CampaignRun` must not delete calendar history — mirrors the existing `CampaignRun.site = ForeignKey(Observatory, on_delete=SET_NULL, ...)` pattern already in `solsys_code/models.py:77-84`) | **Correct, keep it.** Exactly one `CalendarEvent` ever needs exactly one companion row; there is no reason to allow more than one. |
| `ManyToManyField` declared on `CampaignRun` pointing at `CalendarEvent` | Works, but wrong cardinality: a `CalendarEvent` never has more than one owning run in this milestone's model, so M2M would under-constrain (nothing stops two runs both claiming the same event) | Same O(1) prefetch benefit as OneToOne, no advantage here | `filter_horizontal` works but is the wrong widget for a should-be-1:1 relation | Default M2M cascade (only join row removed) — fine, but moot given the cardinality mismatch | **Rejected** — would let a bug or a bad reconciler pass double-attribute an event to two runs with no DB constraint to catch it. |
| `GenericForeignKey` (`content_type` + `object_id` on the companion, pointed at either `CalendarEvent` or something else) | No real JOIN — a separate query per distinct `ContentType`; can't filter by the target's own fields (`.filter(run__telescope_class=...)`) in one query | `prefetch_related` works but issues one extra query *per distinct ContentType* present, not O(1) the way a direct FK is | No native `list_filter`/`search_fields` on the GFK target; needs hand-rolled admin code | **No DB-level referential integrity at all** — deleting the target leaves a dangling `(content_type, object_id)` unless the app manually cleans up; Django's own `GenericRelation` (which *does* provide cleanup) requires adding a field to the target model, which is impossible here since `tom_calendar` can't be edited | **Rejected** — GFK earns its complexity only when the companion needs to attach to an *open-ended* set of third-party model types. Here there are exactly two known, fixed target models (`CalendarEvent`, `ObservationRecord`), each already gets its own purpose-built relation, so GFK buys nothing and costs query-planning and admin ergonomics. |

### Concrete field addition

```python
run = models.ForeignKey(
    'solsys_code.CampaignRun',
    on_delete=models.SET_NULL,
    null=True,
    blank=True,
    related_name='calendar_links',
    verbose_name='Owning campaign run',
)
```

`related_name='calendar_links'` (or whatever name the plan settles on) gives `CampaignRun` its one-to-many reverse relation to the companion rows, and from there to events. A convenience accessor on `CampaignRun` keeps call sites from doing a manual double-hop:

```python
def calendar_events(self) -> models.QuerySet[CalendarEvent]:
    """All CalendarEvents currently attributed to this run, one query, real JOIN."""
    return CalendarEvent.objects.filter(telescope_label_meta__run=self)
```

(`telescope_label_meta` is the *existing* `related_name` on the `event` OneToOneField — see Pattern 2 below on why this name should very likely **not** be touched even though the model class itself is being renamed.) For a list view showing several `CampaignRun`s each with their events (e.g. the campaign table), use `Prefetch('calendar_links', queryset=CompanionModel.objects.select_related('event'))` — same O(1)-extra-query shape as the existing DISPLAY-09 prefetch, not a new pattern.

---

## Django Pattern 2 — `RenameModel`/`RenameField` mechanics: what actually breaks, and safe ordering

This repo has exactly **four** real integration points against `CalendarEventTelescopeLabel` today (confirmed by grep, not assumed):

| # | File | What it references | Breaks on rename? |
|---|------|---------------------|--------------------|
| 1 | `solsys_code/admin.py:4,28,41` | `from solsys_code.models import CalendarEventTelescopeLabel`; `class CalendarEventTelescopeLabelAdmin(...)`; `admin.site.register(CalendarEventTelescopeLabel, ...)` | **Yes, at import time** (`ImportError`/`AttributeError` the moment Django loads `admin.py`) — this is the safest kind of break, caught immediately by `./manage.py check` or the first request. |
| 2 | `solsys_code/management/commands/sync_lco_observation_calendar.py:18,369` | `from solsys_code.models import CalendarEventTelescopeLabel`; `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': ...})` | **Yes, at import time**, same as above — caught by that command's own test suite (`test_sync_lco_observation_calendar.py`, 49 tests including sidecar-write assertions) the moment it runs. |
| 3 | `solsys_code/views.py:114` | `.prefetch_related('telescope_label_meta')` | **No** — this string is the FK's `related_name`, not the model's class name. Renaming the *model* does not touch `related_name` unless you deliberately also rename that. |
| 4 | `src/templates/tom_calendar/partials/calendar.html:228,244` | `{% if event.telescope_label_meta.is_verified == False %}` | **No**, same reason as #3 — Django templates resolve attributes by string; the model's Python class name is invisible here. |

**The load-bearing insight:** renaming the *model class* only breaks Python-level imports (#1, #2), both of which are compile-time-adjacent failures caught the instant the app boots or the command runs — low risk, easy to grep for (`grep -rn CalendarEventTelescopeLabel --include=*.py`). Renaming the `related_name` (`telescope_label_meta`) is the genuinely dangerous move, because #3 and #4 reference it as a bare string with **no static check at all** — a typo or missed occurrence there is a *runtime* `AttributeError`/silent-`None` bug, not an import error, and it can hide in an untested template branch. **Recommendation: keep `related_name='telescope_label_meta'` unchanged** even while renaming the model class and adding the `run` FK. The milestone's stated goal ("closes the pending naming todo") is about the *model's* name being misleading now that it does more than telescope labels — it does not require renaming the accessor, and not renaming the accessor removes two of the four break points from the blast radius entirely.

### Safe migration ordering (concrete, in commit order)

1. **Rename the class and add the field in `models.py` together.** Keep the OneToOne field's own name (`event`) and its `related_name` (`telescope_label_meta`) untouched; only the class name changes, plus the new `run` FK is added.
2. **Hand-author the migration — do not rely on `makemigrations` autodetection.** Django cannot tell "renamed" from "deleted + created" by inspecting field diffs alone; run non-interactively (as CI does) and it will silently emit `DeleteModel`/`CreateModel` instead of `RenameModel`, which — because the OneToOne's `event_id` is the model's actual primary key — would **drop and recreate the table, losing every existing sidecar row**. This repo already hand-authors non-trivial migrations (e.g. `0004_campaignrun_window_schema.py`'s backfill→dedup→constraint-swap), so this is consistent with existing practice, not a new burden. Order **within** the migration matters:
   ```python
   operations = [
       migrations.RenameModel(old_name='CalendarEventTelescopeLabel', new_name='<NewName>'),
       migrations.AddField(
           model_name='<newname>',  # lowercase, post-rename -- Django's migration state is
                                     # cumulative, so by the time AddField runs it must refer
                                     # to the model under its NEW name, not the old one.
           name='run',
           field=models.ForeignKey(null=True, blank=True, on_delete=models.SET_NULL,
                                    related_name='calendar_links', to='solsys_code.campaignrun'),
       ),
   ]
   ```
   `RenameModel` first, `AddField` second, in the same migration or a directly-dependent next one — reversing the order (adding the field to the old model name, then renaming) also works technically but is more confusing to read and out of step with how the actual code change happens (class renamed first).
3. **Data safety check:** `RenameModel` by default also emits `ALTER TABLE ... RENAME TO ...` at the DB level (Django derives the table name from `app_label_modelname` unless `Meta.db_table` is pinned, and this model doesn't pin one). Because the OneToOne's `event` field stays `primary_key=True` and is not touched, **no row data changes** — only the table's own name and Django's bookkeeping change. This satisfies the milestone's explicit "without losing its existing data" requirement without needing the zero-downtime `db_table`-pinning trick some high-traffic Postgres deployments use (that trick is unwarranted complexity for this project — SQLite dev DB, `DEBUG=True`, no rolling-deploy constraint; see "What NOT to Use").
4. **Fix the two import sites** (`admin.py`, `sync_lco_observation_calendar.py`) — plain rename, same commit or immediately after.
5. **Run `./manage.py test solsys_code`** — the existing 49-test `test_sync_lco_observation_calendar.py` suite and the admin test suite (`260714-jpd`) will catch anything missed; `ruff check .`/`ruff format --check .` stay clean since nothing about formatting changes.
6. **Known Django `RenameModel` edge cases that do *not* apply here** (found via targeted search, listed so a future reader doesn't have to re-derive this): `RenameModel` mishandling `related_name='+'` on a *different* model's FK pointing at the renamed one (nothing else FKs to this model), and `RenameModel`-after-`RenameField` ordering bugs affecting M2M `through` tables (this model isn't used as anyone's `through` table). Both are non-issues for a leaf sidecar model with a single inbound OneToOne — flagged only so the plan doesn't need to re-investigate them.

If the model rename and the `source`/`telescope_class` `CampaignRun` migrations land in the same phase, keep them as **separate migration files** even if both touch `solsys_code` — the rename migration should be revertible/reviewable independent of the unrelated `CampaignRun` field additions, and it keeps the "did the rename alone break anything" test run clean.

---

## Django Pattern 3 — `ObservationRecord` linkage: `ManyToManyField` with a custom `through`, declared on `CampaignRun`

`ObservationRecord` is third-party and un-editable, so — per the milestone context — the relation must be declared on `CampaignRun`. A bare `ManyToManyField(ObservationRecord)` would work mechanically (Django creates and owns the join table in `solsys_code`'s own migration state, no cross-app schema coordination needed with `tom_observations`), but it cannot carry the milestone's explicit **"Operator-assisted attribution... never a silent merge"** requirement — a plain M2M join row has no place to record "suggested by the reconciler, not yet confirmed by staff" vs. "staff-confirmed."

**Recommendation: a custom `through` model**, deliberately mirroring the `is_verified` idiom this codebase already established for `CalendarEventTelescopeLabel` — don't invent a new attribution vocabulary when a matching one-bit-flag pattern already exists and is already well-understood by whoever reads this code next:

```python
class CampaignRunObservationRecord(models.Model):
    """Attribution link between a CampaignRun and the ObservationRecord(s) that realise it.

    A custom `through` model rather than a bare ManyToManyField because attribution
    is operator-confirmed, not assumed (v2.2 scope): the reconciler creates rows with
    is_confirmed=False when it *suggests* a match; staff review flips it to True.
    Mirrors the is_verified idiom already established by the CalendarEvent companion.
    """

    run = models.ForeignKey('solsys_code.CampaignRun', on_delete=models.CASCADE, related_name='record_links')
    observation_record = models.ForeignKey(
        'tom_observations.ObservationRecord', on_delete=models.CASCADE, related_name='+'
    )
    is_confirmed = models.BooleanField(default=False, verbose_name='Confirmed by staff (not auto-suggested)')
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=('run', 'observation_record'), name='unique_run_observation_record_link'),
        ]
```

```python
# on CampaignRun:
observation_records = models.ManyToManyField(
    'tom_observations.ObservationRecord',
    through='CampaignRunObservationRecord',
    related_name='campaign_runs',
    blank=True,
)
```

Notes tied to the querying/prefetching/admin/cascade axes the milestone question asked about:

- **Querying:** `run.observation_records.all()` and `record.campaign_runs.all()` both work as ordinary M2M traversal despite the custom `through`; filtering the *link* itself (e.g. only-confirmed links) goes through `run.record_links.filter(is_confirmed=True)`.
- **Prefetching:** `.prefetch_related('record_links__observation_record')` or `Prefetch('observation_records', queryset=...)` — same O(1)-extra-query shape used everywhere else in this codebase (DISPLAY-09 precedent).
- **Admin:** register `CampaignRunObservationRecord` directly (own `ModelAdmin`, `list_filter=['is_confirmed']`) rather than trying to force `filter_horizontal` on the plain M2M field — `filter_horizontal`/`filter_vertical` don't support extra `through` fields, and staff need to see/toggle `is_confirmed`, not just add/remove links.
- **Cascade:** `on_delete=CASCADE` on both `through`-model FKs is correct and is the Django default behaviour for M2M anyway — deleting a `CampaignRun` or an `ObservationRecord` only removes the *link* row, never cascades to delete the other side. This matches the milestone's own framing: "these are not duplicates to be merged... the fix is attribution, not deduplication" — the link is disposable, the run and the record are not.
- `related_name='+'` on the `observation_record` FK (no reverse accessor from `ObservationRecord` back to the through rows) is a deliberate minor choice — nothing in this milestone's scope needs to query "which through-rows reference this record" directly rather than via `record.campaign_runs`; drop the `+` and give it a real `related_name` if a later phase needs that.

---

## Django Pattern 4 — Idempotent reconciliation: plain ORM, extending `insert_or_create_calendar_event()`

**Confirms the milestone's stated strong prior.** `solsys_code/calendar_utils.py:insert_or_create_calendar_event()` already implements exactly the no-churn contract the reconciler needs — `get_or_create()`, then a field-by-field diff (`_update_or_unchanged()`) that only calls `.save(update_fields=...)` when something actually changed, returning `'created'`/`'updated'`/`'unchanged'` for the caller to tally. This is not a coincidence to work around; it's the load-bearing precedent to *reuse directly*, not reinvent:

- The reconciler's per-stage event projection (stage 1 sunset→sunrise, stage 2 all-day, stage 3 narrowed-to-record, stage 4 COMPLETED) is structurally identical to what `sync_lco_observation_calendar` and `backfill_range_calendar_events` already do — key each projected event on a stable, deterministic lookup (this codebase's convention: `{'url': ...}` for URL-keyed records, or a composite `CAMPAIGN:{pk}:{date.isoformat()}`-style key for date-keyed ones), pass changed fields through `insert_or_create_calendar_event()`, and let its existing no-churn logic decide create/update/unchanged. Do not write a second, parallel diff-and-apply helper — extend or call the existing one.
- For the companion-row attribution write (setting `run` on the generalised companion, and `is_confirmed` on `CampaignRunObservationRecord`), the same `update_or_create()`-plus-explicit-field-comparison shape `sync_lco_observation_calendar.py:369` already uses for `CalendarEventTelescopeLabel.objects.update_or_create(event=event, defaults={'is_verified': ...})` applies directly. **Caveat:** `update_or_create()` unconditionally calls `.save()` on an existing match even when nothing changed — for the strict no-churn behaviour the idempotency test needs to prove, prefer the explicit `_update_or_unchanged()`-style dict-diff (exactly as `insert_or_create_calendar_event()` already does) over bare `update_or_create()` for any write path whose second-pass silence is being asserted.
- No `bulk_create`/`bulk_update` batching is needed at this milestone's real data scale (measured: 19 attributable `CampaignRun`s in the dev DB today, tens not thousands) — introducing bulk operations would trade away the per-row no-churn diff this pattern depends on for a performance win this codebase doesn't need yet. If a future milestone's data volume changes that calculus, that's a new, separately-justified decision — not a default to reach for now.

---

## Testing Techniques: proving the reconciler is idempotent

Follow the exact tool and idiom already established at `solsys_code/tests/test_calendar_template.py:272-289` (`test_display09_query_count_bounded`), extended from "query count doesn't grow" to "second pass writes nothing":

```python
from django.db import connection
from django.test.utils import CaptureQueriesContext

class ReconcilerIdempotencyTest(TestCase):
    def test_second_run_is_a_no_op(self):
        call_command('reconcile_campaign_runs')  # first pass: creates/updates as needed

        # Snapshot state that a silent second-pass write would disturb.
        before = list(
            CalendarEvent.objects.order_by('pk').values('pk', 'modified')
        )

        with CaptureQueriesContext(connection) as ctx:
            call_command('reconcile_campaign_runs')  # second pass

        after = list(
            CalendarEvent.objects.order_by('pk').values('pk', 'modified')
        )
        self.assertEqual(before, after)  # no new rows, no modified-timestamp churn

        write_queries = [
            q for q in ctx.captured_queries
            if q['sql'].strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
        ]
        self.assertEqual(write_queries, [])  # second pass performs reads only
```

Key points, tied to what the milestone question specifically asked for:

- **Standard `django.test.TestCase`** (not `TransactionTestCase`) is sufficient — the reconciler doesn't need to observe cross-transaction visibility, and `TestCase`'s wrapping-transaction-per-test is faster and is what every other command test in this codebase already uses.
- **`CaptureQueriesContext(connection)`** is the right tool for two distinct assertions here, both already precedented in this codebase: (a) the existing DISPLAY-09-style "query count doesn't grow" check, useful for catching an accidental N+1 in the reconciler's own per-run loop, and (b) the new "zero write statements on pass two" check, which is a stronger and more direct proof of idempotency than a query *count* comparison alone (a second pass could in principle issue the same *number* of queries while still silently rewriting rows — asserting on `sql` content rules that out).
- **Assert both directions**, mirroring how `test_sync_lco_observation_calendar.py`'s no-churn tests are already structured: run the reconciler against fixture data that should produce changes (created path) and against already-reconciled data that should produce none (unchanged path) — a single "run twice" test alone doesn't distinguish "correctly idempotent" from "silently does nothing on every run."
- **Reuse the counters-dict reporting convention** already shared by all four sync commands (`counters[facility]['created'/'updated'/'unchanged']`) for the reconciler's own summary output — asserting on the command's own stdout/counters in addition to DB state gives a second, independent check of "nothing happened" without needing to inspect SQL text at all, and keeps the reconciler consistent with its siblings' operator-facing output shape (directly relevant to the paired-docs runbook this milestone will need to update per `CLAUDE.md`'s paired-docs rule).

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `django-dirtyfields` / `django-model-utils` `FieldTracker` | Solves exactly the "did this field change" question the codebase already answers explicitly and auditably via `_update_or_unchanged()`'s dict-diff; adding a change-tracking library on top would give the same answer through an opaque signal-based mechanism, one more thing to understand, with no capability gain. | The existing explicit field-diff pattern in `calendar_utils.py`. |
| `django-reversion` (or any audit/versioning package) | This milestone is about *idempotent projection* (does a repeat run change anything), not *change history* (what did this row look like last week). Different problem; out of scope for v2.2. | Django's own `created`/`modified` timestamp fields, already present on every model here. |
| `django-fsm` / other state-machine libraries for `CampaignRun.run_status` or the four-stage window pipeline | `CampaignRun`'s statuses are already plain `TextChoices` with transitions owned by explicit view/command logic (`CampaignRunDecisionView`), which this milestone doesn't change. The reconciler *reads* run state to decide which pipeline stage applies — it doesn't own state transitions, so a state-machine library would be modelling a state machine that doesn't actually exist in the reconciler's own responsibility. | Plain `if`/`elif` stage-selection logic keyed on the fields already present (`approval_status`, resolved site vs. `telescope_class`, presence of scheduled/completed `ObservationRecord`s) — directly mirrors the four-stage table already specified in `PROJECT.md`. |
| A task queue (Celery, or reaching for the already-installed `django-tasks`) for the reconciler | All four existing sync/ingest commands (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, `backfill_range_calendar_events`) are synchronous `BaseCommand`s, run on demand or via cron/operator action, at dev-DB scale (hundreds of rows, not thousands). The reconciler is the same shape of problem (retiring `backfill_range_calendar_events` explicitly, per the milestone goal) — there's no new latency/volume driver here that would justify async execution. | A synchronous management command, same convention as its four siblings. |
| `rapidfuzz` (already declined once, in v2.1) | Not part of this milestone's scope at all, but worth reiterating as house precedent: Phase 18's spike explicitly tested `rapidfuzz` against real messy site-name data for a *different* feature (site fuzzy-matching) and found no match-quality win over stdlib `difflib` sufficient to justify the dependency (`docs/design/uncertain_scheduling_spike.rst`). The same "prove the stdlib is insufficient before adding a dependency" bar applies here, and nothing in v2.2's scope needs fuzzy text matching at all. | N/A — not applicable to v2.2, listed only to preempt the wrong instinct. |
| `GenericForeignKey` for either link | See Pattern 1's comparison table — loses JOIN-ability, prefetch efficiency, admin ergonomics, and DB-level cascade integrity, and buys nothing since both target model types (`CalendarEvent`, `ObservationRecord`) are fixed and already known. | Purpose-built `ForeignKey`/`ManyToManyField(through=...)` per link, as above. |
| Zero-downtime `db_table`-pinning trick for the `RenameModel` migration | A real technique for high-traffic production Postgres deployments doing rolling migrations without locking a live table — not this project's situation (SQLite dev DB, `DEBUG=True`, no concurrent-deploy constraint documented anywhere in `PROJECT.md`/`CLAUDE.md`). | A plain `RenameModel` operation, letting Django rename the underlying table too. |

## Stack Patterns by Variant

**If a future phase needs the reconciler to run on a schedule (not just on-demand):** reach for whatever cron/scheduling mechanism the deployment already uses at the OS/hosting level (this project's `CLAUDE.md` documents no in-app scheduler) — still don't introduce Celery/`django-tasks` for that; a `BaseCommand` invoked by cron is the same pattern the other four sync commands already assume.

**If a later milestone needs the M2M-attribution UI to be richer than a two-state confirmed/unconfirmed flag (e.g. confidence scores, multiple candidate matches per record):** extend `CampaignRunObservationRecord` with more fields (it's already a first-class model, not a bare M2M) rather than reaching for a matching/scoring library — the existing site-fuzzy-match precedent (Phase 18/21/22) shows this codebase's default is to prove stdlib insufficiency first.

## Version Compatibility

| Package | Compatible With | Notes |
|-----------|-----------------|-------|
| Django 5.2.13 | Python 3.10–3.12 (project's tested range) | `RenameModel`/`RenameField`/`ManyToManyField(through=...)`/`CaptureQueriesContext` are long-stable Django APIs (predate Django 4) — nothing in this milestone is gated on Django 5.2 specifically, and nothing here is at risk if `tomtoolkit` moves off the `3.0.0aN` prerelease train to a different Django-supporting range later. |
| `tomtoolkit==3.0.0a9` (bundles `tom_calendar`, `tom_observations`) | Django 5.2.13 (as currently installed in this venv) | Both target models (`CalendarEvent`, `ObservationRecord`) were read directly from the installed package source for this research — confirmed plain `AutoField` PKs, no custom managers/QuerySets that would complicate `select_related`/`prefetch_related`, no existing hooks for third-party attachment (i.e., the sidecar/through-model approach is not working around anything the library changed recently). |

## Sources

- Direct inspection of installed package source (HIGH confidence — primary source, exact version pinned in this project's venv): `tom_calendar/models.py` (`CalendarEvent`, `EventTodo`) and `tom_observations/models.py` (`ObservationRecord`) at `~/venv/devel_fomo311_venv/lib64/python3.11/site-packages/`.
- Direct inspection of this repository's own code (HIGH confidence): `solsys_code/models.py` (`CalendarEventTelescopeLabel`, `CampaignRun`), `solsys_code/calendar_utils.py` (`insert_or_create_calendar_event`, `_update_or_unchanged`), `solsys_code/admin.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/views.py`, `src/templates/tom_calendar/partials/calendar.html`, `solsys_code/tests/test_calendar_template.py` (`CaptureQueriesContext` precedent), `.planning/PROJECT.md` (v2.2 milestone context and prior Key Decisions, including the Phase 18 `difflib`-vs-`rapidfuzz` precedent).
- [Django #23577 — Rename operations should rename indexes, constraints, sequences and triggers named after their former value](https://code.djangoproject.com/ticket/23577) — MEDIUM confidence (community/issue-tracker source, cross-checked against Django's own documented `RenameModel` semantics); informed the "known edge cases that don't apply here" note in Pattern 2.
- [Django #27903 — RenameModel does not change ForeignKey with related_name='+'](https://code.djangoproject.com/ticket/27903) — MEDIUM confidence; same use as above.
- [Django #29000 — RenameModel does not rename M2M column when run after RenameField](https://code.djangoproject.com/ticket/29000) — MEDIUM confidence; confirmed not applicable since the renamed model isn't a `through` table.

---
*Stack research for: FOMO v2.2 "One Canonical Run Record" — Django third-party-model linking, migration mechanics, and reconciler idempotency*
*Researched: 2026-07-26*
