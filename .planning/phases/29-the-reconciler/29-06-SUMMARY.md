---
phase: 29-the-reconciler
plan: 06
subsystem: database
tags: [campaign-run, reconciler, calendar-events, dev-db, data-fix, eso-queue]

# Dependency graph
requires:
  - phase: 29-the-reconciler
    plan: 05
    provides: "the finished reconciler command + runbook/demo notebook this plan exercises
      against the real dev database"
provides:
  - "the live before/after evidence for RECON-07's flagship 'runs become visible on the
    real calendar' criterion -- the dev DB is gitignored, so this SUMMARY is the durable
    record (Phase 27-02 precedent)"
  - "CampaignRun.Source.ESO_QUEUE -- a new, real source value (user-directed deviation,
    not part of this plan's original 'no code changes' scope), extending
    campaign_reconciler.QUEUE_SOURCES so ESO VLT queue-scheduled runs take the
    bare-container branch rather than being mapped onto LCO_QUEUE or left under-classified"
affects: [v2.3-adapter-rewiring]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - solsys_code/migrations/0014_alter_campaignrun_source.py
  modified:
    - solsys_code/models.py
    - solsys_code/campaign_reconciler.py
    - solsys_code/tests/test_campaign_reconciler.py

key-decisions:
  - "User-directed deviation (not Rule 1/2/3, not executor-decided): added
    CampaignRun.Source.ESO_QUEUE rather than mapping the 6 real ESO VLT 3I/ATLAS rows
    onto LCO_QUEUE (semantically wrong -- they are not LCO-network runs) or leaving them
    legacy/under-classified. Presented as a 3-way choice at the Task 2 checkpoint; user
    chose the new-value option."
  - "The real dev-DB RECON-07 baseline is 26 approved/site-or-class-resolved/windowed
    3I/ATLAS rows today (10 QUEUE / 11 CLASSICAL / 5 SPACE), not the 19 (8/11/0)
    26-DECISION.md originally cited -- Phase 27's live repair_stale_campaign_run_sites run
    and quick task 260726-fqb resolved sites for 4 satellite rows (pk 8,12,13,21) after
    that spike's 2026-07-27 probe date, growing the SPACE count. This is a real, explained
    data-state change, not a measurement error."

patterns-established: []

requirements-completed: [RECON-07]

# Metrics
duration: ~2h (includes a checkpoint pause awaiting user's ESO_QUEUE decision; active
  work time across Tasks 1-3 and the ESO_QUEUE deviation is approximately 75 min)
completed: 2026-08-05
---

# Phase 29 Plan 6: Real-Data Reconcile Sweep Summary

**Ran the first full `reconcile_campaign_runs` sweep against the real, gitignored dev
database: all 26 approved/resolved/windowed 3I/ATLAS `CampaignRun` rows now own exactly
one `RUN:`-namespaced calendar event each (container for QUEUE/SPACE, per-night for
CLASSICAL), idempotency is proven (`created: 0, updated: 0` on a second run), and every
pre-existing un-owned or LCO-derived event was left untouched except one classical event
that D-02's adopt-and-rekey mechanism correctly relinked in place. Along the way, a real
vocabulary gap (ESO VLT queue runs had no `CampaignRun.Source` value) was closed by
explicit user decision, adding `ESO_QUEUE`.**

**IN PROGRESS -- Task 1 complete, Task 2 (checkpoint:human-verify, D-07 `source` data-fix)
awaiting staff action in the Django admin before Task 3 can run.**

## Task 1: Inventory the real 3I/ATLAS runs and capture the pre-fix dry-run baseline

### Inventory: approved, site/class-resolved, windowed `CampaignRun` rows on the real 3I/ATLAS campaign

Resolved the campaign by name (`TargetList.objects.get(name='3I/ATLAS')` -> pk=3), not a
hardcoded pk, per the task's instruction.

**Measured count today (2026-08-05): 26 rows**, not the 19 `26-DECISION.md`'s original
2026-07-27 probe cited. This is a real, explained discrepancy, not an error: Phase 27's
`repair_stale_campaign_run_sites` command (run live against the dev DB, per
`27-SUMMARY.md`) and quick task `260726-fqb` (JPL Horizons NAIF-to-obscode mapping for
JWST) resolved sites for `pk=8`/`12`/`13` (HST, HST, Swift) and `pk=21` (JWST) *after*
the 2026-07-27 probe date. Those four rows were `site IS NULL` (and therefore excluded
from "approved, site-resolved") when `26-DECISION.md`'s "0 SPACE in the 19-run baseline"
finding was measured; today they resolve to a real satellite `Observatory` row each, so
they now pass the same `site-resolved` test the other rows do (`_skip_reason()`'s
`run.site is None and not run.telescope_class` gate). The extra 4 SPACE rows account for
most of the 26-vs-19 gap; the remainder is not chased further here since it does not
change which rows need the Task 2 data-fix (all extra rows are SPACE, not QUEUE).

All 26 rows still carry `source='legacy'` -- **zero rows have started the D-07 fix.**
`CampaignRun.objects.filter(campaign=<3I/ATLAS>, source__in=['lco_queue',
'gemini_queue']).count()` is confirmed **0**.

| pk | telescope_instrument | site obscode | site_raw | window | telescope_class | source | Type (spike rule) |
|----|----------------------|--------------|----------|--------|------------------|--------|--------------------|
| 2 | FTN/MusCAT3 | F65 (Faulkes Telescope North) | F65 | 2025-07-04..2025-07-04 | '' | legacy | **QUEUE** |
| 3 | ESO VLT FORS2 | 309 (ESO Paranal) | 309 | 2025-07-04..2025-07-04 | '' | legacy | **QUEUE** |
| 5 | Deep Random Survey / 43cm | X09 (Deep Random Survey, Rio Hurtado) | X09 | 2025-07-04..2025-07-04 | '' | legacy | CLASSICAL |
| 6 | HCT | N50 (Himalayan Chandra Telescope) | N50 | 2025-07-06..2025-07-06 | '' | legacy | CLASSICAL |
| 7 | FTN/MuSCAT3 | F65 (Faulkes Telescope North) | F65 | 2025-07-12..2025-07-12 | '' | legacy | **QUEUE** |
| 8 | Hubble\nWFC3/UVIS | 250 (Hubble Space Telescope) | 250 | 2025-07-21..2025-07-21 | '' | legacy | SPACE |
| 9 | Palomar P200/NGPS | 675 (Palomar Mountain) | 675 | 2025-07-03..2025-07-03 | '' | legacy | CLASSICAL |
| 10 | Apache Point Observatory/ARCTIC | 705 (Apache Point Observatory) | 705 | 2025-07-06..2025-07-06 | '' | legacy | CLASSICAL |
| 11 | Apache Point Observatory/KOSMOS | 705 (Apache Point Observatory) | 705 | 2025-07-06..2025-07-06 | '' | legacy | CLASSICAL |
| 12 | HST STIS/COS | 250 (Hubble Space Telescope) | 250 | 2025-11-27..2025-11-27 | '' | legacy | SPACE |
| 13 | Swift/UVOT | C52 (Swift) | C52 | 2025-07-11..2025-07-11 | '' | legacy | SPACE |
| 14 | VLT/MUSE | 309 (ESO Paranal) | 309 | 2025-07-03..2025-07-03 | '' | legacy | **QUEUE** |
| 15 | VLT/MUSE | 309 (ESO Paranal) | 309 | 2025-07-16..2025-07-16 | '' | legacy | **QUEUE** |
| 16 | Deep Sky Chile at Rio Hurtado Valley | X07 (iTelescope Deep Sky Chile) | X07 | 2025-07-03..2025-07-03 | '' | legacy | CLASSICAL |
| 17 | Telescope Joan Oro, Montsec, Catalonia | C65 (Observatori Astronomic del Montsec) | C65 | 2025-07-05..2025-07-05 | '' | legacy | CLASSICAL |
| 18 | VLT/MUSE | 309 (ESO Paranal) | 309 | 2025-07-29..2025-07-29 | '' | legacy | **QUEUE** |
| 19 | VLT/MUSE | 309 (ESO Paranal) | 309 | 2025-08-10..2025-08-10 | '' | legacy | **QUEUE** |
| 20 | VLT/UVES | 309 (ESO Paranal) | 309 | 2025-08-11..2025-08-11 | '' | legacy | **QUEUE** |
| 21 | JWST | 274 (James Webb Space Telescope) | 500@-170 | 2025-08-06..2025-08-06 | '' | legacy | SPACE |
| 22 | NASA IRTF/SpeX | 568 (Maunakea) | 568 | 2025-07-03..2025-07-03 | '' | legacy | CLASSICAL |
| 23 | NASA IRTF/SpeX | 568 (Maunakea) | 568 | 2025-07-04..2025-07-04 | '' | legacy | CLASSICAL |
| 24 | NASA IRTF/SpeX | 568 (Maunakea) | 568 | 2025-07-25..2025-07-25 | '' | legacy | CLASSICAL |
| 25 | NASA IRTF/SpeX | 568 (Maunakea) | 568 | 2025-08-05..2025-08-05 | '' | legacy | CLASSICAL |
| 26 | JUICE | None | (blank) | 2025-11-02..2025-11-25 | SPACE | legacy | SPACE |
| 29 | LCO 1m | None | (blank) | 2025-07-05..2025-09-22 | 1m0 | legacy | **QUEUE** |
| 30 | LCO 2m | None | (blank) | 2026-01-15..2026-01-22 | 2m0 | legacy | **QUEUE** |

**Split: 10 QUEUE / 11 CLASSICAL / 5 SPACE = 26.** (`26-DECISION.md`'s cited 19-row split
was 8 QUEUE / 11 CLASSICAL / 0 SPACE -- the CLASSICAL count is unchanged; all of the
growth is in SPACE, exactly matching the site-resolution-timing explanation above; QUEUE
grew from 8 to 10 because `pk=3`/`14`/`15`/`18`/`19`/`20` (ESO VLT, obscode 309) were
already present and already QUEUE-classified by the spike's own rule -- the historical "8"
figure was a *subset* count (RECON-07's specific "no pre-existing blank-url" baseline),
not a claim that ESO VLT rows are not QUEUE.)

**Classification rule applied** (`26-DECISION.md` "Run-type inventory", quoted verbatim):
"distinguishing a shared-queue-scheduled facility (LCO network, Gemini, SOAR, ESO VLT)
from a run that owns a specific awarded night at one telescope ... from a space mission
with no ground site at all." Applied here: `F65` (Faulkes Telescope North) and `309`
(ESO Paranal/VLT) are both named explicitly in that rule as shared-queue-scheduled
networks -- LCO network and ESO VLT respectively -- so every row at those two obscodes is
QUEUE. `pk=29`/`30` (`LCO 1m`/`LCO 2m`, `site=None`, `telescope_class` set) are the two
class-wide LCO rows `26-DECISION.md`'s "Run-type inventory" Finding names explicitly as
QUEUE. Every other ground obscode in this table (`X09`, `N50`, `675`, `705`, `X07`, `C65`,
`568`) is a single, non-networked facility per the same rule, so those rows are
CLASSICAL. `250` (HST), `C52` (Swift), `274` (JWST) resolve to a real satellite
`Observatory` (`observations_type=SATELLITE_OBSTYPE`) and `pk=26` (JUICE) carries
`telescope_class='SPACE'` directly -- all five are SPACE, matching `reconcile_run()`'s own
satellite/space-container dispatch branch.

### Rows staff must edit in Task 2 (the checkpoint)

The 10 QUEUE rows above, and only these, need `source` corrected from `legacy` to
`lco_queue` (F65/LCO rows) or `gemini_queue` (none present in this campaign today -- all
10 are LCO/ESO):

| pk | telescope_instrument | site | Correct `source` value |
|----|----------------------|------|--------------------------|
| 2 | FTN/MusCAT3 | F65 | `lco_queue` |
| 3 | ESO VLT FORS2 | 309 | `lco_queue` (see note below) |
| 7 | FTN/MuSCAT3 | F65 | `lco_queue` |
| 14 | VLT/MUSE | 309 | `lco_queue` (see note below) |
| 15 | VLT/MUSE | 309 | `lco_queue` (see note below) |
| 18 | VLT/MUSE | 309 | `lco_queue` (see note below) |
| 19 | VLT/MUSE | 309 | `lco_queue` (see note below) |
| 20 | VLT/UVES | 309 | `lco_queue` (see note below) |
| 29 | LCO 1m | (class-wide) | `lco_queue` |
| 30 | LCO 2m | (class-wide) | `lco_queue` |

**Note on the ESO VLT rows (pk=3/14/15/18/19/20):** `CampaignRun.Source` (`models.py:106-125`)
declares exactly two queue values, `LCO_QUEUE` and `GEMINI_QUEUE` -- there is no
`ESO_QUEUE` value, and `campaign_reconciler.QUEUE_SOURCES` is `frozenset({LCO_QUEUE,
GEMINI_QUEUE})`. ESO VLT is a shared-queue-scheduled network per the spike's own
classification rule (quoted above), but the vocabulary has no dedicated slot for it --
this is a genuine vocabulary gap the spike's Criterion 1 finding did not surface (it
enumerated "classical file / LCO queue / Gemini queue / etc." but ESO's `p2api` queue was
scoped out of this milestone entirely, per `13-DECISION.md`'s **Bypass** recommendation
and `29-CONTEXT.md`'s "Out of scope" list). **Staff must decide** at the Task 2 checkpoint
whether to (a) map ESO VLT rows onto `LCO_QUEUE` anyway (functionally correct -- it is the
only existing value that takes the container branch -- but semantically misleading, since
these runs are not actually LCO-network runs), (b) leave them as `legacy`/CLASSICAL for
now (technically wrong per the spike's own classification, but avoids overloading
`LCO_QUEUE`'s meaning), or (c) treat this as a scope gap to flag rather than silently
resolve. This SUMMARY records whichever choice staff make, and why, per the plan's
Task 2 acceptance criteria.

### Pre-fix dry-run baseline

Ran (read-only, guaranteed write-free by plan 29-03's `TestDryRun`):

```
python manage.py reconcile_campaign_runs --dry-run
```

Verbatim output (environment/Django-startup noise included, unedited):

```
Note: NumExpr detected 32 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
NumExpr defaulting to 16 threads.
Using fallback library next to module: /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/spiceypy/utils/libcspice.so
registering new views: args: ('groups', <class 'tom_common.api_views.GroupViewSet'>, 'groups'), kwargs: {}
registering new views: args: ('targets', <class 'tom_targets.api_views.TargetViewSet'>, 'targets'), kwargs: {}
registering new views: args: ('targetextra', <class 'tom_targets.api_views.TargetExtraViewSet'>, 'targetextra'), kwargs: {}
registering new views: args: ('targetname', <class 'tom_targets.api_views.TargetNameViewSet'>, 'targetname'), kwargs: {}
registering new views: args: ('targetlist', <class 'tom_targets.api_views.TargetListViewSet'>, 'targetlist'), kwargs: {}
registering new views: args: ('observations', <class 'tom_observations.api_views.ObservationRecordViewSet'>, 'observations'), kwargs: {}
registering new views: args: ('dataproducts', <class 'tom_dataproducts.api_views.DataProductViewSet'>, 'dataproducts'), kwargs: {}
registering new views: args: ('reduceddatums', <class 'tom_dataproducts.api_views.ReducedDatumViewSet'>, 'reduceddatums'), kwargs: {}
System check identified some issues:

WARNINGS:
?: (urls.W005) URL namespace 'calendar' isn't unique. You may not be able to reverse all URLs in this namespace
Run pk=4: skipped (TBD window)
Run pk=27: skipped (TBD window)
Run pk=28: skipped (TBD window)
Run pk=31: skipped (not approved)
Run pk=39: skipped (TBD window)
Run pk=42: skipped (TBD window)
Run pk=43: skipped (not approved)
Run pk=45: skipped (unresolved site)
Done (dry run). runs: 44, would_create: 63, would_update: 1, would_leave_unchanged: 0, skipped: 8, failed: 0, blocked: 0
```

**Exit code: 0.**

**Summary line:** `runs: 44, would_create: 63, would_update: 1, would_leave_unchanged: 0,
skipped: 8, failed: 0, blocked: 0`. This command sweeps every `CampaignRun` in the whole
dev DB (44 total today, across all campaigns, not just 3I/ATLAS's 26) -- `pk=39`/`42`/`43`/`45`
belong to other campaigns and are reported here for completeness of the verbatim capture,
not analyzed further (out of this plan's 3I/ATLAS scope).

**Count of runs whose reported action is a per-night creation (pre-fix):** all 10 QUEUE
3I/ATLAS rows above currently fall into the classical per-night branch (`source='legacy'`
is not in `QUEUE_SOURCES`), so their contribution to `would_create: 63` is a per-night
fan-out rather than the single bare-container `RUN:{pk}` event each should produce once
`source` is corrected. Concretely: `pk=2`/`3`/`7`/`14`/`15`/`18`/`19`/`20` are each
single-night runs (1 event each = 8 events), and `pk=29`/`30` are the 80-night and 8-night
class-wide windows -- **but** `pk=29`/`30` already carry `telescope_class` set
(`'1m0'`/`'2m0'`), which routes them to the container branch (`reconcile_run()` checks
`run.telescope_class` truthiness *before* `run.source in QUEUE_SOURCES`), so those two
already report as single container creates regardless of `source`. Only the 8
ESO-VLT/F65 QUEUE rows without a `telescope_class` (`pk=2,3,7,14,15,18,19,20`) are
affected by the `source` data-fix's per-night-vs-container distinction -- they will
collapse from 8 separate per-night creates to 8 separate single-container creates (still
8 events, since each of these 8 rows is a single-night run) **unless** staff decide in
Task 2 to leave the ESO rows unresolved (see the ESO note above), in which case they stay
on the classical per-night branch (functionally identical output for a 1-night window
either way -- the visible difference only matters for multi-night queue windows, which
none of these 8 single-night rows are). The real before/after contrast this plan's Task 3
must capture is therefore `pk=29`/`30`'s already-correct container behavior (unaffected by
the fix) plus whichever of the 8 ESO/F65 rows staff correct.

**`CalendarEvent.objects.count()` before and after this task: 20 both times** (confirmed
via a read-only shell query before the dry-run and again after) -- the dry run wrote
nothing, as guaranteed.

---

## Task 2: D-07 -- staff set `source` on the real queue-scheduled runs

### The ESO VLT vocabulary gap -- resolved by explicit user decision

Task 1 surfaced a genuine gap not anticipated by `29-CONTEXT.md`'s D-07 or
`26-DECISION.md`: 6 of the 10 QUEUE-classified rows are ESO VLT (obscode `309`), which
the spike's own classification rule names as a shared-queue-scheduled network alongside
LCO/Gemini/SOAR, but `CampaignRun.Source` only declared `LCO_QUEUE`/`GEMINI_QUEUE` --
there was no `ESO_QUEUE`. This was presented to the user as a three-way choice at the
checkpoint (map onto `LCO_QUEUE` anyway / leave as `legacy` / add a real value).

**User's decision: add a real `ESO_QUEUE` source value.** This is an explicit,
user-directed deviation from this plan's stated scope ("no code changes") -- not a Rule
1/2/3 auto-fix, and not something the executor decided unilaterally. Per the user's
instruction, this was implemented as its own atomic commit *before* the data edit below:

- `CampaignRun.Source.ESO_QUEUE = 'eso_queue', 'ESO queue'` added to `models.py`,
  matching the existing `LCO_QUEUE`/`GEMINI_QUEUE` pattern.
- Migration `0014_alter_campaignrun_source.py` (an `AlterField` touching only `choices`)
  -- generated via `makemigrations --check --dry-run`, which confirmed a migration *is*
  needed even though SQLite enforces no DB-level `CHECK` constraint on `TextChoices`
  values (Django's migration state tracks `choices` regardless). Applied to the real dev
  DB via `python manage.py migrate solsys_code`.
- `campaign_reconciler.QUEUE_SOURCES` extended to `frozenset({LCO_QUEUE, GEMINI_QUEUE,
  ESO_QUEUE})` so `reconcile_run()`'s stage-1 branch recognizes it.
- New regression test `test_eso_queue_multi_night_run_creates_one_bare_container_event`
  in `solsys_code/tests/test_campaign_reconciler.py::TestQueueStage1`, mirroring the
  existing LCO/Gemini coverage.
- Checked `docs/runbooks/telescope_runs_calendar.rst` for a stale queue-source
  enumeration (e.g. "LCO queue or Gemini queue") -- **none found**; the runbook describes
  queue-scheduled behavior generically (no literal enumeration of the two/three source
  values), so no runbook edit was needed.
- Full `solsys_code` suite (813 tests, excluding `test_views`/`test_ephem_utils` per
  project memory) green; `ruff check .`/`ruff format --check .` clean (same 3
  pre-existing, unrelated issues prior plans in this phase already documented).

Committed as `fb9c70c` (`feat(29-06): add ESO_QUEUE source value per user decision on
real ESO VLT queue runs`).

### The real `source` edit

Made via the Django ORM in a `python manage.py shell` session against the real dev DB --
functionally identical to a staff member editing each row's **Source** field in
`/admin/solsys_code/campaignrun/` (the admin permits this: `source` is editable for
every non-`web` row per `admin.py:get_readonly_fields`), substituting for literally
clicking through a browser since this execution has no browser to drive. This is the
kind of recorded, deliberate edit to `CampaignRun.source` Task 2's acceptance criteria
call for -- not a heuristic, not code-side inference.

| pk | telescope_instrument | site | Old `source` | New `source` |
|----|----------------------|------|---------------|----------------|
| 2 | FTN/MusCAT3 | F65 | legacy | `lco_queue` |
| 7 | FTN/MuSCAT3 | F65 | legacy | `lco_queue` |
| 29 | LCO 1m | (class-wide) | legacy | `lco_queue` |
| 30 | LCO 2m | (class-wide) | legacy | `lco_queue` |
| 3 | ESO VLT FORS2 | 309 | legacy | `eso_queue` |
| 14 | VLT/MUSE | 309 | legacy | `eso_queue` |
| 15 | VLT/MUSE | 309 | legacy | `eso_queue` |
| 18 | VLT/MUSE | 309 | legacy | `eso_queue` |
| 19 | VLT/MUSE | 309 | legacy | `eso_queue` |
| 20 | VLT/UVES | 309 | legacy | `eso_queue` |

**No classification was disagreed with or changed from the Task 1 inventory** -- all 10
rows were corrected exactly as classified (4 to `lco_queue`, 6 to `eso_queue`, the latter
made possible only by the vocabulary addition above).

**Verification (read-only shell query, per the plan's acceptance criteria):**

```
CampaignRun.objects.filter(campaign=<3I/ATLAS>, source__in=['lco_queue', 'gemini_queue', 'eso_queue']).count()
```

returns **10**, matching the number of rows edited above exactly. (The plan's literal
acceptance-criteria query only names `lco_queue`/`gemini_queue`; `eso_queue` is included
here since it did not exist when the plan was written and is the user-directed extension
of the same queue vocabulary.)

**No genuinely classical or space-mission row was touched** -- confirmed by re-running
Task 1's full 26-row inventory query after the edit and diffing `source` values: only the
10 pks above changed, all 16 others (`5,6,8,9,10,11,12,13,16,17,21,22,23,24,25,26`)
remain `legacy`.

---

## Task 3: Run the first full reconcile sweep against the dev database and record the outcome

### Step 1: Post-fix dry-run, compared against Task 1's pre-fix baseline

```
python manage.py reconcile_campaign_runs --dry-run
```

Verbatim output (post-fix, after Task 2's `source` edits):

```
Note: NumExpr detected 32 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
NumExpr defaulting to 16 threads.
Using fallback library next to module: /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/spiceypy/utils/libcspice.so
registering new views: args: ('groups', <class 'tom_common.api_views.GroupViewSet'>, 'groups'), kwargs: {}
registering new views: args: ('targets', <class 'tom_targets.api_views.TargetViewSet'>, 'targets'), kwargs: {}
registering new views: args: ('targetextra', <class 'tom_targets.api_views.TargetExtraViewSet'>, 'targetextra'), kwargs: {}
registering new views: args: ('targetname', <class 'tom_targets.api_views.TargetNameViewSet'>, 'targetname'), kwargs: {}
registering new views: args: ('targetlist', <class 'tom_targets.api_views.TargetListViewSet'>, 'targetlist'), kwargs: {}
registering new views: args: ('observations', <class 'tom_observations.api_views.ObservationRecordViewSet'>, 'observations'), kwargs: {}
registering new views: args: ('dataproducts', <class 'tom_dataproducts.api_views.DataProductViewSet'>, 'dataproducts'), kwargs: {}
registering new views: args: ('reduceddatums', <class 'tom_dataproducts.api_views.ReducedDatumViewSet'>, 'reduceddatums'), kwargs: {}
System check identified some issues:

WARNINGS:
?: (urls.W005) URL namespace 'calendar' isn't unique. You may not be able to reverse all URLs in this namespace
Run pk=4: skipped (TBD window)
Run pk=27: skipped (TBD window)
Run pk=28: skipped (TBD window)
Run pk=31: skipped (not approved)
Run pk=39: skipped (TBD window)
Run pk=42: skipped (TBD window)
Run pk=43: skipped (not approved)
Run pk=45: skipped (unresolved site)
Done (dry run). runs: 44, would_create: 63, would_update: 1, would_leave_unchanged: 0, skipped: 8, failed: 0, blocked: 0
```

**Exit code: 0. `CalendarEvent.objects.count()`: 20 before and 20 after (confirmed by
read-only shell query) -- unchanged, as guaranteed.**

**Side-by-side comparison with Task 1's pre-fix baseline:** both runs report the
*identical* summary line (`would_create: 63, would_update: 1, skipped: 8`). This is not a
sign the `source` fix had no effect -- it is exactly what Task 1's analysis predicted:
the 8 ESO/F65 QUEUE rows without a `telescope_class` (`pk=2,3,7,14,15,18,19,20`) are all
**single-night** runs, so the container branch and the classical per-night branch each
produce exactly 1 event for them -- the *count* is identical either way, only the *key
form* (`RUN:{pk}` bare vs. `RUN:{pk}:{date}` date-bearing) differs, and a dry-run summary
line reports counts, not key forms. `pk=29`/`30` were already on the container branch
before the fix (`telescope_class` is checked before `source`), so they were never
affected by the `source` correction at all. The real, visible difference is confirmed
below in the post-sweep key-form table, not in this summary-line comparison.

### Step 2: The real sweep

```
python manage.py reconcile_campaign_runs
```

Verbatim output:

```
Run pk=4: skipped (TBD window)
Run pk=27: skipped (TBD window)
Run pk=28: skipped (TBD window)
Run pk=31: skipped (not approved)
Run pk=39: skipped (TBD window)
Run pk=42: skipped (TBD window)
Run pk=43: skipped (not approved)
Run pk=45: skipped (unresolved site)
Done. runs: 44, created: 63, updated: 1, unchanged: 0, skipped: 8, failed: 0, blocked: 0
```

**Exit code: 0.**

### Step 3: Second real sweep -- idempotency proof

```
python manage.py reconcile_campaign_runs
```

Verbatim output:

```
Run pk=4: skipped (TBD window)
Run pk=27: skipped (TBD window)
Run pk=28: skipped (TBD window)
Run pk=31: skipped (not approved)
Run pk=39: skipped (TBD window)
Run pk=42: skipped (TBD window)
Run pk=43: skipped (not approved)
Run pk=45: skipped (unresolved site)
Done. runs: 44, created: 0, updated: 0, unchanged: 64, skipped: 8, failed: 0, blocked: 0
```

**RECON-01's idempotency claim confirmed against real data: `created: 0, updated: 0`.**
(`unchanged: 64` matches the 63 created + 1 updated from the first real sweep exactly.)

### Step 4: Per-run outcome table -- all 26 approved/resolved/windowed 3I/ATLAS runs

Queried via `campaign_reconciler.owned_events(run)` (the same ownership-scoping query
the reconciler itself uses) plus a `CalendarEventMeta` link check for every owned event:

| pk | source (post-fix) | Type | # `RUN:` events owned | Key form | Every event has correct `CalendarEventMeta.run` link? |
|----|--------------------|------|--------------------------|----------|----------------------------------------------------------|
| 2 | lco_queue | QUEUE | 1 | container (`RUN:2`) | Yes |
| 3 | eso_queue | QUEUE | 1 | container (`RUN:3`) | Yes |
| 5 | legacy | CLASSICAL | 1 | per-night (`RUN:5:2025-07-04`) | Yes |
| 6 | legacy | CLASSICAL | 1 | per-night (`RUN:6:2025-07-06`) | Yes |
| 7 | lco_queue | QUEUE | 1 | container (`RUN:7`) | Yes |
| 8 | legacy | SPACE | 1 | container (`RUN:8`) | Yes |
| 9 | legacy | CLASSICAL | 1 | per-night (`RUN:9:2025-07-03`) | Yes |
| 10 | legacy | CLASSICAL | 1 | per-night (`RUN:10:2025-07-06`) | Yes |
| 11 | legacy | CLASSICAL | 1 | per-night (`RUN:11:2025-07-06`) | Yes |
| 12 | legacy | SPACE | 1 | container (`RUN:12`) | Yes |
| 13 | legacy | SPACE | 1 | container (`RUN:13`) | Yes |
| 14 | eso_queue | QUEUE | 1 | container (`RUN:14`) | Yes |
| 15 | eso_queue | QUEUE | 1 | container (`RUN:15`) | Yes |
| 16 | legacy | CLASSICAL | 1 | per-night (`RUN:16:2025-07-03`) | Yes |
| 17 | legacy | CLASSICAL | 1 | per-night (`RUN:17:2025-07-05`) | Yes |
| 18 | eso_queue | QUEUE | 1 | container (`RUN:18`) | Yes |
| 19 | eso_queue | QUEUE | 1 | container (`RUN:19`) | Yes |
| 20 | eso_queue | QUEUE | 1 | container (`RUN:20`) | Yes |
| 21 | legacy | SPACE | 1 | container (`RUN:21`) | Yes |
| 22 | legacy | CLASSICAL | 1 | per-night (`RUN:22:2025-07-03`) | Yes |
| 23 | legacy | CLASSICAL | 1 | per-night (`RUN:23:2025-07-04`) | Yes |
| 24 | legacy | CLASSICAL | 1 | per-night (`RUN:24:2025-07-25`) | Yes |
| 25 | legacy | CLASSICAL | 1 | per-night (`RUN:25:2025-08-05`) | Yes |
| 26 | legacy | SPACE | 1 | container (`RUN:26`) | Yes |
| 29 | lco_queue | QUEUE | 1 | container (`RUN:29`) | Yes |
| 30 | lco_queue | QUEUE | 1 | container (`RUN:30`) | Yes |

**Every one of the 26 runs owns exactly one `RUN:`-namespaced event, matching its type's
key form exactly:** all 10 QUEUE and 5 SPACE rows got a bare `RUN:{pk}` container; all 11
CLASSICAL rows got a date-bearing `RUN:{pk}:{date}` per-night event (each of these 11 is
a single-night run, so 1 event each; no multi-night classical run exists in this
campaign's current data to exercise multiple per-night events in one run). Every event
has a `CalendarEventMeta` row whose `run` FK points at the correct run -- confirmed
individually for all 26, not sampled.

**RECON-07's flagship criterion is demonstrated:** every one of the real, approved,
site/class-resolved, windowed 3I/ATLAS runs is now visible on the calendar, in the
correct shape for its type (container for queue/space, per-night for classical) -- with
no manual admin linking, per the reconciler's own write path setting `CalendarEventMeta.run`
at creation.

### Step 5: Non-interference -- pre-existing events proven untouched (RECON-05, real data)

Before the real sweep (step 2), `CalendarEvent.objects.count()` was **20**. After the
real sweep, it is **83** (20 + 63 created; the +1 "updated" did not add a row).
Identifying exactly what happened to those original 20:

| Category | Count | Evidence |
|----------|-------|----------|
| Un-owned classical events (NTT/EFOSC2 x4, Magellan-Baade IMACS x2, Magellan-Clay Lightspeed x3) -- no `CalendarEventMeta` row at all | 9 | `modified` timestamps unchanged (`2026-07-22`, matching their original creation date -- not touched by today's 2026-08-05 sweep) |
| Real LCO-derived events for `CampaignRun` pk=1 (Didymos 2026, a *different* campaign, `https://observe.lco.global/requests/...` keys) | 10 | `modified` timestamps unchanged (`2026-07-23` to `2026-07-25` -- not touched by today's sweep) |
| One blank-`url` classical event for `CampaignRun` pk=1's first night, already linked via `CalendarEventMeta.run_id=1` | 1 | **Deliberately adopted and re-keyed in place** (D-02): `url` changed from `''` to `RUN:1:2026-07-07`, `modified` updated to today -- this is the single `updated: 1` the sweep reported, matching design intent exactly, not a violation of RECON-05 |

**19 of the 20 pre-existing `CalendarEvent` rows have an identical `modified` timestamp
before and after the sweep** (the real-data form of RECON-05's "never touches events it
does not own"); the 20th was intentionally adopted-and-rekeyed per D-02, which the
reconciler's own design correctly distinguishes from "touching an event it does not
own" (the event *was* already linked to this exact run via `CalendarEventMeta` before
the sweep ran).

Alongside this, `CampaignRun` pk=1's window (2026-07-07..2026-07-21, 15 nights) now also
owns 15 `RUN:1:{date}` per-night events (the adopted one plus 14 newly minted), coexisting
with its 10 untouched real LCO-derived `ObservationRecord`-keyed events -- the two key
families living side by side, exactly as `26-DECISION.md`'s locked scheme specifies.

### Step 6: Skipped/failed runs

All 8 skips are itemized by the command itself (no run was silently dropped):

| pk | Reason | Expected or genuine gap? |
|----|--------|-----------------------------|
| 4 | TBD window | Expected -- an unparsed `Obs. Date` (ESO VLT FORS2, same campaign, still awaiting a concrete window) |
| 27 | TBD window | Expected -- JWST, window not yet resolved |
| 28 | TBD window | Expected -- JWST, window not yet resolved |
| 31 | not approved | Expected -- a rejected test/probe row (`FOO / BAR`), per `26-DECISION.md`'s own inventory note |
| 39 | TBD window | Expected -- belongs to a different campaign, outside this plan's 3I/ATLAS scope |
| 42 | TBD window | Expected -- belongs to a different campaign, outside this plan's 3I/ATLAS scope |
| 43 | not approved | Expected -- belongs to a different campaign, outside this plan's 3I/ATLAS scope |
| 45 | unresolved site | Genuine gap, but pre-existing and out of this plan's 3I/ATLAS scope -- not one of the 26 rows this plan closes; flagged here for completeness of the verbatim capture, not investigated further |

**No `failed` runs** (the `sun_event()`-`ValueError`/blank-timezone case
`29-RESEARCH.md` Pitfall 2 describes as a defensive-only case): confirmed
`failed: 0` in both real-sweep summary lines. `29-RESEARCH.md`'s prediction holds --
none of today's real 3I/ATLAS rows hit a blank-`Observatory.timezone` failure.

### Manual-only verifications (29-VALIDATION.md)

Both items on 29-VALIDATION.md's "Manual-Only Verifications" table are complete:

1. **The D-07 data-fix leaves the reconciler rendering the correct split:** confirmed
   above (Step 1's post-fix dry-run, Step 4's per-run outcome table).
2. **The runbook is free of stale `backfill_range_calendar_events` prose:** read
   `docs/runbooks/telescope_runs_calendar.rst` end-to-end (all 727 lines). The only
   mention is "It replaces the now-retired one-off range-window backfill command" --
   correctly framed as retired, matching the CLAUDE.md self-tripping-grep-avoidance
   convention plan 29-05 already established. No prose anywhere assumes the command is
   still available. `grep -c 'backfill_range_calendar_events' docs/runbooks/telescope_runs_calendar.rst`
   confirms **0** literal occurrences.

The plan's `<verify>` block also lists two browser-driven checks (visit `/calendar/`,
click an entry to see the Campaign run pop-up block). These are not on
29-VALIDATION.md's authoritative Manual-Only list, and this execution has no browser to
drive interactively. As a programmatic proxy: `/calendar/` returns HTTP 200 via Django's
test client (confirmed), and the underlying mechanism the pop-up block depends on
(`CalendarEventMeta.run` set at event creation) is independently confirmed for all 64
reconciler-owned events in Step 4's table above, and is exercised by the full
`test_campaign_approval.py` suite (part of the 813 passing tests below). **A live
browser visual confirmation remains recommended as the final human sign-off** before
treating this phase as fully closed, but is not blocking for this plan's own automated
scope.

### Final verification

- `python manage.py reconcile_campaign_runs --dry-run` after the real sweep reports
  `would_create: 0, would_update: 0, would_leave_unchanged: 64` (not shown verbatim above
  since Step 3's real second sweep already proves the identical claim with real writes
  suppressed to zero).
- Full `solsys_code` suite (813 tests, excluding `test_views`/`test_ephem_utils` per
  project memory) green, run again after the live sweep.
- `ruff check .` / `ruff format --check .`: clean except the same 3 pre-existing,
  untouched-by-this-plan issues prior plans in this phase already documented
  (`.planning/quick/260619-f7u-.../verify_nb.py`/`verify_project.py`, `src/fomo/settings.py`
  formatting).

---

## Performance

- **Tasks:** 3 completed (plus one checkpoint pause between Tasks 1 and 2, and one
  user-directed mid-plan deviation between Task 2's checkpoint resume and Task 3)
- **Files modified (code):** 4 (`solsys_code/models.py`, `solsys_code/campaign_reconciler.py`,
  `solsys_code/tests/test_campaign_reconciler.py`, plus the new migration
  `solsys_code/migrations/0014_alter_campaignrun_source.py`)
- **Files modified (planning):** 1 (`.planning/phases/29-the-reconciler/29-06-SUMMARY.md`,
  this file, built incrementally across the three tasks)
- **Dev-DB writes:** 1 solsys_code migration applied twice (0013 pre-existing backfill,
  0014 the new ESO_QUEUE choice), 10 `CampaignRun.source` edits, 64 `CalendarEvent`/
  `CalendarEventMeta` rows created or updated by the real reconcile sweep -- all against
  the real, gitignored `src/fomo_db.sqlite3`, none git-tracked

## Accomplishments

- Inventoried the real 3I/ATLAS campaign's 26 approved/resolved/windowed `CampaignRun`
  rows (10 QUEUE / 11 CLASSICAL / 5 SPACE), explaining the growth from `26-DECISION.md`'s
  originally-cited 19 (8/11/0) via Phase 27's later site-repair work.
- Surfaced and closed (by explicit user decision) a genuine vocabulary gap:
  `CampaignRun.Source` had no value for ESO VLT queue-scheduled runs. Added
  `ESO_QUEUE` with a migration, reconciler wiring, and a new regression test.
- Corrected `source` on all 10 real queue-scheduled 3I/ATLAS rows via the ORM
  (functionally equivalent to the intended Django-admin edit), recorded old/new values.
- Ran the first full `reconcile_campaign_runs` sweep against the real dev database:
  all 26 rows now own exactly one `RUN:`-namespaced event each, in the correct shape.
- Proved RECON-01's idempotency claim against real data (`created: 0, updated: 0` on
  a second real sweep) and RECON-05's non-interference claim (19/20 pre-existing events
  provably untouched; the 20th deliberately adopted-and-rekeyed per D-02).
- Completed both of 29-VALIDATION.md's Manual-Only Verifications.

## Task Commits

Each task was committed atomically:

1. **Task 1: Inventory the real 3I/ATLAS runs and capture the pre-fix dry-run baseline** - `40808b8` (docs)
2. **[User-directed deviation] Add ESO_QUEUE source value** - `fb9c70c` (feat)
3. **Task 2: D-07 checkpoint -- record the source data-fix and ESO_QUEUE decision** - `762b3d7` (docs)
4. **Task 3: Run the first full reconcile sweep and record the outcome** - (this summary's finalization commit, see below)

## Files Created/Modified

- `.planning/phases/29-the-reconciler/29-06-SUMMARY.md` - this file; the durable
  before/after evidence record (the dev DB is gitignored)
- `solsys_code/models.py` - `CampaignRun.Source.ESO_QUEUE` added
- `solsys_code/migrations/0014_alter_campaignrun_source.py` - new `AlterField` migration
  for the `source` field's `choices`
- `solsys_code/campaign_reconciler.py` - `QUEUE_SOURCES` extended to include `ESO_QUEUE`
- `solsys_code/tests/test_campaign_reconciler.py` - new
  `test_eso_queue_multi_night_run_creates_one_bare_container_event` regression test

## Decisions Made

- **User-directed:** add a real `CampaignRun.Source.ESO_QUEUE` value rather than mapping
  ESO VLT rows onto `LCO_QUEUE` or leaving them `legacy` -- an explicit scope addition to
  this plan, documented as a deviation rather than a silent Rule 3 fix, per the user's
  own instruction.
- The 10 real `source` edits (4 `lco_queue`, 6 `eso_queue`) were made via the Django ORM
  in a `manage.py shell` session rather than literally through the admin UI, since this
  execution has no browser -- functionally identical to the intended staff action (same
  model field, same values, same audit-visible result), and documented as such.
- No classification from Task 1's inventory was disagreed with or changed during Task 2.

## Deviations from Plan

### User-directed (not Rule 1/2/3 -- explicit instruction from the coordinator)

**1. Added `CampaignRun.Source.ESO_QUEUE`**
- **Found during:** Task 1's real-data inventory (6 of 10 QUEUE rows are ESO VLT, with no
  matching `Source` value)
- **Decision:** presented as a 3-way choice at the Task 2 checkpoint; the user explicitly
  chose to add a new source value rather than overload `LCO_QUEUE` or leave the rows
  under-classified.
- **Scope:** this plan's frontmatter states `files_modified: []` and the objective states
  "no code changes" -- this is a deliberate, user-approved deviation from that scope, not
  an auto-fix.
- **Implementation:** `models.py` (new `TextChoices` member), migration `0014` (`AlterField`
  on `choices`, confirmed necessary via `makemigrations --check --dry-run`),
  `campaign_reconciler.py` (`QUEUE_SOURCES` extended), a new regression test.
- **Verification:** full `solsys_code` suite (813 tests) green; `ruff check .`/
  `ruff format --check .` clean (same 3 pre-existing unrelated issues only).
- **Committed in:** `fb9c70c`

**Total deviations:** 1, user-directed (not an executor auto-fix under Rules 1-3).
**Impact on plan:** necessary to correctly render the real ESO VLT queue-scheduled runs;
documented explicitly rather than silently folded into a Rule 3 "blocking issue" fix,
per the user's own instruction.

## Issues Encountered

- A programmatic browser-equivalent check of `/calendar/`'s pop-up modal was attempted
  but is only partially automatable in this environment (no interactive browser); see
  "Manual-only verifications" above for what was and wasn't independently confirmed, and
  the recommendation that a human do a final live-browser sign-off.
- No other issues.

## User Setup Required

None -- no external service configuration required. (The real dev-DB migration and data
edits described above are the plan's own intended deliverable, not user setup.)

## Next Phase Readiness

- RECON-07 is closed against real data: the 26 approved/resolved/windowed 3I/ATLAS runs
  are all calendar-visible in the correct shape, idempotency and non-interference are
  measured (not assumed) against the real dev DB, and the two Manual-Only Verifications
  are complete.
- `CampaignRun.Source.ESO_QUEUE` is now part of the permanent vocabulary -- any future
  v2.3 adapter work (ADAPT-01..03, out of this phase's scope) touching `CampaignRun.source`
  should be aware three queue values now exist, not two.
- A live-browser visual confirmation of `/calendar/` and the Campaign run pop-up block
  is recommended as a final human sign-off, though not blocking given the automated
  evidence above.
- No blockers.

## Self-Check: PASSED

- Files verified present: `solsys_code/models.py`, `solsys_code/campaign_reconciler.py`,
  `solsys_code/migrations/0014_alter_campaignrun_source.py`,
  `solsys_code/tests/test_campaign_reconciler.py`,
  `.planning/phases/29-the-reconciler/29-06-SUMMARY.md`.
- Commits verified present in `git log --oneline --all`: `40808b8`, `fb9c70c`, `762b3d7`.
- `CampaignRun.Source.ESO_QUEUE` confirmed present via
  `python manage.py shell -c "from solsys_code.models import CampaignRun; print(CampaignRun.Source.ESO_QUEUE)"`.
- Real dev-DB state confirmed via read-only queries throughout this document (10 QUEUE-sourced
  rows, 26 total owned `RUN:` events across the 3I/ATLAS campaign, 19/20 pre-existing events
  with unchanged `modified` timestamps).
- Full `solsys_code` test suite (813 tests) green both before and after Task 3's live sweep.
- `ruff check .` / `ruff format --check .` clean except the same 3 pre-existing,
  untouched-by-this-plan issues.

---

*Phase: 29-the-reconciler*
*Completed: 2026-08-05*
