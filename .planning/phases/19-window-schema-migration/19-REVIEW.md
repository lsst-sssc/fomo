---
phase: 19-window-schema-migration
reviewed: 2026-07-10T00:00:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/migrations/0004_campaignrun_window_schema.py
  - solsys_code/models.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/tests/test_campaign_submission.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_import_campaign_csv.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-07-10T00:00:00Z
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found

## Summary

This is a fresh, independent re-review of phase 19's 15-file change set (the hard-cutover of
`CampaignRun`'s `obs_date`/`ut_start`/`ut_end` triple to a `window_start`/`window_end` date-pair
schema), run immediately ahead of a `--fix` pass. **None of the 15 files have changed since the
prior deep review** (commit `3bf7abe`, 2026-07-09T23:30:00Z) — `git diff 3bf7abe..HEAD -- <these
15 paths>` is empty for all of them. I re-derived every finding independently by reading the
current file contents and cross-referencing against the pre-migration schema (migration `0003`)
and the diff against `diff_base`, rather than trusting the prior report's prose, and confirm: **all
5 previously reported findings (CR-01, WR-01, WR-02, WR-03, IN-01) still hold, unaddressed, against
the current file contents.** I also found one new Info-level item not previously flagged.

The consumer-side rewrite (`campaign_gap.py`, `campaign_views.py`, `campaign_tables.py`,
`campaign_forms.py`, `import_campaign_csv.py`) is careful and well-tested for the code paths it
actually reaches. The migration's TBD-branch (`window_start IS NULL`) dedup step is real, correctly
ordered ahead of its `AddConstraint`, and has been validated against known dev-DB duplicates. The
core, still-open problem is that the **resolved-window branch has no equivalent dedup step**, even
though the same backfill operation that makes the TBD dedup necessary also creates duplicate risk on
the resolved-window side — and nothing in this changeset (code or tests) detects or prevents it. Two
further latent (currently-unreachable, but real) gaps affect the resolved-window natural key and the
`window_start`/`window_end` null-pairing invariant, plus a test-coverage gap on the migration's own
data-transformation logic and a factual inconsistency in the paired demo notebook's narrative text.

## Narrative Findings (AI reviewer)

### Critical Issues

#### CR-01: Migration's dedup step only covers the TBD branch, not the structurally identical resolved-window collision the backfill also creates

**File:** `solsys_code/migrations/0004_campaignrun_window_schema.py:17-47,93-100`
**Issue:**

`backfill_window_fields` (lines 17-21) sets `window_start = window_end = obs_date` for **every**
row, discarding the time-of-night distinction the *old* constraint relied on. That old constraint
(`solsys_code/migrations/0003_campaignrun_natural_key_unique_constraint.py:14-19`) was keyed on
`(campaign, telescope_instrument, ut_start)` — a full `DateTimeField`, not `obs_date` — and carried
no `condition=`. Because SQL unique constraints never treat two `NULL`s as equal, this also means
the old schema tolerated unlimited `ut_start IS NULL` rows sharing `(campaign,
telescope_instrument)`; the important case for this finding is the *non-null* one: two pre-migration
rows for the same campaign/telescope/night but two distinct, non-null `ut_start` values (two
separate observing blocks the same night) were perfectly legal, non-colliding rows.

After `backfill_window_fields` runs, both such rows collapse to the same `(window_start,
window_end)` pair — an exact duplicate under the new `unique_campaign_run_resolved_window`
constraint (`AddConstraint` at lines 93-100, mirrored in `models.py:124-128`). `dedupe_tbd_collisions`
(lines 23-47) only queries `CampaignRun.objects.filter(window_start__isnull=True)` — rows whose
`obs_date` was `NULL` — and does nothing for the resolved-window case. There is no equivalent dedup
step anywhere between the backfill and the `unique_campaign_run_resolved_window` `AddConstraint`.

Concretely: if the pre-migration `CampaignRun` table contains **any** two rows sharing `(campaign,
telescope_instrument, obs_date)` with distinct, non-null `ut_start` values — a state the *old*
schema explicitly permitted — this non-reversible, single-file migration will raise `IntegrityError`
when it reaches the `unique_campaign_run_resolved_window` `AddConstraint` and fail to apply, on
whatever environment first encounters that data shape (a fresh deploy re-importing a fuller
historical CSV export before migrating, a staging DB seeded differently from the one dev DB this
phase's manual dry run happened to check, etc.).

No test in this changeset exercises this scenario: `grep`-ing `solsys_code/tests/` for
`dedupe_tbd_collisions`/`backfill_window_fields`/`unique_campaign_run_resolved_window` finds zero
hits outside `test_resolved_window_same_key_collides` (which tests the *new* constraint against
*post-migration* `CampaignRun.objects.create()` calls, not the migration's own `RunPython` backfill
logic against pre-migration row data). The 355 passing tests cited in the review scope all run
against a freshly-migrated, empty test DB and cannot catch this class of bug.

**Fix:** Add a second, symmetric dedup `RunPython` step (mirroring `dedupe_tbd_collisions`) for the
resolved-window branch, inserted anywhere between the backfill and the resolved-window
`AddConstraint`:

```python
def dedupe_resolved_window_collisions(apps, schema_editor):
    """Analogous to dedupe_tbd_collisions, but for the resolved-window branch: two
    pre-migration rows that shared campaign+telescope_instrument+obs_date but differed
    only by the now-dropped ut_start collapse onto an identical (window_start, window_end)
    tuple after backfill_window_fields runs. Must run before the resolved-window
    UniqueConstraint is added below, or that AddConstraint can fail against real
    pre-existing same-night, different-ut_start rows.
    """
    CampaignRun = apps.get_model('solsys_code', 'CampaignRun')
    seen: dict[tuple, int] = {}
    qs = CampaignRun.objects.filter(window_start__isnull=False).order_by('pk')
    for run in qs:
        key = (run.campaign_id, run.telescope_instrument, run.window_start, run.window_end)
        if key in seen:
            logger.warning(
                'Deleting duplicate resolved-window CampaignRun pk=%s (kept pk=%s) for '
                'campaign=%s telescope_instrument=%r window=%s..%s',
                run.pk, seen[key], run.campaign_id, run.telescope_instrument,
                run.window_start, run.window_end,
            )
            run.delete()
        else:
            seen[key] = run.pk
```

Insert `migrations.RunPython(dedupe_resolved_window_collisions, reverse_code=migrations.RunPython.noop)`
immediately after `dedupe_tbd_collisions` and before `RemoveConstraint`. Before shipping, re-run the
manual `manage.py migrate` dry run against a copy of the real dev DB with this step added to confirm
whether it actually deletes anything.

### Warnings

#### WR-01: CSV importer's natural-key lookup omits `window_end`, contradicting the model's own documented resolved-window contract

**File:** `solsys_code/management/commands/import_campaign_csv.py:170-173`
**Issue:**

```python
run, action = insert_or_create_campaign_run(
    {'campaign': campaign, 'telescope_instrument': telescope_instrument, 'window_start': obs_date},
    fields,
)
```

`fields` (built at lines 149-168) sets `'window_end': obs_date`, but the `lookup` dict passed to
`insert_or_create_campaign_run` — and hence to `CampaignRun.objects.get_or_create(**lookup,
defaults=fields)` in `campaign_utils.py:301` — only constrains `window_start`. `models.py:120-124`'s
own Meta comment is explicit about why this is wrong: *"window_end is included (not just
window_start) so a range starting on the same day as an existing single-night entry is not treated
as the same row."* The importer's lookup doesn't follow that rule it documents elsewhere.

This is currently unreachable — every write path shipped in this phase
(`CampaignRunSubmissionView`, this importer) always sets `window_end == window_start`, so no real
range row exists yet to collide with. But it is a landmine for the next feature that creates a range
(Phase 20, or a direct admin/ORM edit): re-running this importer over a campaign that already has a
resolved multi-night `CampaignRun` starting on the CSV row's `obs_date` would either match that range
row via `get()` (since only `window_start` is filtered) and silently overwrite its `window_end` back
down to `obs_date` via `insert_or_create_campaign_run`'s field-diff/update path
(`campaign_utils.py:304-309`) — a silent data-loss collapse of a real range into a single night — or
raise an uncaught `django.core.exceptions.MultipleObjectsReturned` from `get_or_create()` if more
than one range shares that `window_start`, a crash the command's per-row `try/except ValueError`
(`import_campaign_csv.py:108-126`) does not catch, aborting the whole batch import instead of
skip-and-log per row.

**Fix:** Include `window_end` in the lookup, matching the model's actual key:

```python
run, action = insert_or_create_campaign_run(
    {
        'campaign': campaign,
        'telescope_instrument': telescope_instrument,
        'window_start': obs_date,
        'window_end': obs_date,
    },
    fields,  # drop 'window_end' from fields; it's now part of the lookup key
)
```

#### WR-02: No DB-level invariant that `window_start`/`window_end` are null together; `claimed_dates()` will crash on a mismatched row

**File:** `solsys_code/models.py:113-140` (missing `CheckConstraint`); `solsys_code/campaign_gap.py:176-183`
**Issue:**

Every reader of `window_start`/`window_end` in this changeset
(`campaign_tables.render_window_start`, `campaign_views.CampaignRunDecisionView.post`,
`campaign_gap.claimed_dates`) assumes the two fields are either both `NULL` (TBD) or both set
(resolved). Nothing enforces this at the model or DB layer — only the two partial
`UniqueConstraint`s exist, and neither prevents `window_start` set with `window_end` `NULL` (or vice
versa); that combination isn't `NULL` on `window_start`, so it falls into the resolved-window
constraint's `condition=Q(window_start__isnull=False)` branch and is simply allowed to persist as a
unique (if internally inconsistent) row.

`campaign_gap.claimed_dates()` (lines 176-183, verified against current file contents) only guards
the fully-TBD case:

```python
for run in qs:
    if run.window_start is None:
        undated_runs.append(run)
        continue
    n_days = (run.window_end - run.window_start).days + 1   # TypeError if window_end is None
```

If a `CampaignRun` ever exists with `window_start` set and `window_end` `None` (currently
unreachable via any shipped write path, but reachable via direct ORM/admin/future code — e.g. a
partially-completed Phase 20 range-entry form, or a bad manual fixup), this raises an uncaught
`TypeError` that propagates out of `_compute_gap()` / `get_or_compute_gap()` and 500s the entire
`CampaignGapAnalysisView` for that campaign+site+target combination, rather than a graceful per-run
skip. `campaign_tables.render_window_start()` degrades more gracefully in the same scenario (renders
`"2026-08-01 -&gt; None"` rather than crashing), which is itself an inconsistency in how the two
call sites handle the same unguarded state.

**Fix:** Add a DB-level `CheckConstraint` enforcing the pairing:

```python
models.CheckConstraint(
    condition=(
        models.Q(window_start__isnull=True, window_end__isnull=True)
        | models.Q(window_start__isnull=False, window_end__isnull=False)
    ),
    name='campaign_run_window_start_end_null_together',
),
```

and/or defensively guard `claimed_dates()`:

```python
if run.window_start is None or run.window_end is None:
    undated_runs.append(run)
    continue
```

#### WR-03: Migration's `RunPython` backfill/dedup logic has zero automated regression coverage

**File:** `solsys_code/migrations/0004_campaignrun_window_schema.py:17-47`; `solsys_code/tests/test_campaign_models.py`
**Issue:**

`backfill_window_fields` and `dedupe_tbd_collisions` are non-reversible, data-deleting functions
with real production consequences (see CR-01), but no test in this changeset exercises them
directly. `test_campaign_models.py`'s `TestCampaignRunWindowSchema` (verified against current file
contents) only asserts the *post-migration model shape* — constraints and nullability via
`CampaignRun.objects.create()` calls against the already-migrated test schema — never the
migration's own `RunPython` data-transformation functions against a simulated pre-migration state.
The mitigation actually used (a manual dry run against a copy of the dev DB, referenced by the prior
review and CR-01 above) leaves no artifact in the repository proving this logic was exercised, let
alone that CR-01's specific gap was checked for.

**Fix:** At minimum, add a migration test (Django's `MigratorTestCase`-style pattern, or a
hand-rolled one using historical model state via `apps.get_model` inside a
`django.test.TransactionTestCase`) that seeds two rows sharing `obs_date` with distinct `ut_start`
values against the pre-0004 schema, runs the migration, and asserts the outcome (dedup, or a
documented, deliberately-accepted failure mode) rather than relying entirely on an unrepeatable
manual dry run.

### Info

#### IN-01: Demo notebook's committed output contradicts its own narrative text

**File:** `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` (markdown cell `48b89ebf` vs.
code-cell outputs `1093927e`/`16d14cf8`)
**Issue:**

Verified directly against the notebook's current JSON: the "synthetic fixture" markdown cell
(`48b89ebf`, unchanged by this phase's regeneration) says *"all 6 should be `created` on a first
run"*, but the executed output of cell `1093927e` (the actual import run) reads `Done. created: 0,
updated: 0, unchanged: 6, skipped: 0, site_needs_review: 1`, and cell `16d14cf8`'s inspection table
reports a total row count of **10**, not 6 — because the notebook was regenerated against the
shared, already-populated dev DB (`src/fomo_db.sqlite3`) rather than a clean one, as the commit
message for the `19-04` regeneration commit acknowledges. The later-added "Idempotency note" cell
(`c32cae1e`) correctly anticipates this for the approval-lifecycle rows, but the older "synthetic
fixture" cell's "all 6 should be `created`" sentence was never updated to match and now reads as
factually wrong next to the actual printed output immediately below it.

**Fix:** Update the "synthetic fixture" markdown cell's wording to acknowledge the notebook may show
`unchanged` instead of `created` on a rerun against a pre-populated dev DB, matching the tone of the
"Idempotency note" cell added later in the same notebook.

#### IN-02: Ground-vs-space calendar-projection branch treats all non-satellite observatory types identically, without acknowledging RADAR/OCCULTATION

**File:** `solsys_code/campaign_views.py:339-375`
**Issue:**

`CampaignRunDecisionView.post()`'s new D-06 hybrid calendar projection branches only on
`run.site.observations_type == Observatory.SATELLITE_OBSTYPE`; every other value falls into the
`else` branch, which is commented `# Ground-based observatory:` and unconditionally calls
`sun_event(run.site, run.window_start, kind='sun')` to derive a dip-corrected sunset/sunrise window.
`Observatory.OBSTYPE_CHOICES` (`solsys_code/solsys_code_observatory/models.py:17-25`) actually has
four members — `OPTICAL`, `OCCULTATION`, `SATELLITE`, `RADAR` — and this is the only place in the
codebase that branches on `observations_type` at all (`grep -rn SATELLITE_OBSTYPE solsys_code/`
outside `solsys_code_observatory/models.py` and its own tests returns only this one call site). A
`RADAR` observatory (e.g. a planetary-radar facility, which characteristically observes regardless
of solar altitude) or an `OCCULTATION` site gets the same "must be dark, dip-corrected sunset to
sunrise" treatment as an `OPTICAL` telescope, which may not reflect how those sites actually operate
— `sun_event(kind='sun')` could legitimately raise `ValueError` (no two-crossing sun_event) or
produce a semantically meaningless window for such a site, silently skipping calendar projection
(logged at `debug`) rather than surfacing that the ground/space dichotomy doesn't fit every
`observations_type`.

**Fix:** Either explicitly document that the ground/space split is deliberately binary for this
milestone (RADAR/OCCULTATION sites are out of scope, same as ASSET-02 is explicitly deferred to
Phase 20 per the `claimed_dates()` docstring), or branch more precisely, e.g. treat only
`OPTICAL_OBSTYPE` as the "needs a dark window" case and leave `RADAR`/`OCCULTATION` unprojected
(matching the "graceful no projection" behavior already used for unresolved sites/TBD runs) rather
than silently routing them through `sun_event()`.

---

_Reviewed: 2026-07-10T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
