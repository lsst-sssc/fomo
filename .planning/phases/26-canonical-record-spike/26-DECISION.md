# Phase 26: Canonical-Record Spike - Decision

**Investigated:** 2026-07-27
**Status:** In progress. This document is built up across plans 26-01, 26-02 and 26-03.
Plan 26-01 records the D-04 dated snapshot, D-16, SPIKE-02's four-adapter identity
mapping, the stage-2/stage-0 inventory (D-05..D-08), the RECON-07 baseline, and
SPIKE-04's migration-application and measured-rename-blast-radius findings. SPIKE-01's
`IntegrityError` coexistence proof and SPIKE-03's adopt-vs-gap-fill prototype (D-11) are
recorded by plan 26-02. `## Recommendation` and `## Durable summary` are completed by
plan 26-03.

This phase is **investigation-only**, following the Phase 13 (ESO) and Phase 18
(uncertain-scheduling) precedents. No `CampaignRun` schema migration, no
`campaign_reconciler.py` module, and no attribution UI ships from this phase. Every
`solsys_code/` edit this plan makes lives on a dedicated scratch git branch
(`spike/26-canonical-record-probe`) that is never merged and is deleted in plan 26-03;
the scratch DB copy (`tmp/26-spike-db-copy.sqlite3`) and `local_settings.py` are
git-excluded and removed at the same point. The sole committed artifact of this plan is
this file.

**Evidence posture, stated explicitly because it differs from Phase 18's:** Phase 18
wrapped every write in a rolled-back `transaction.atomic()` block because it ran against
the live `Observatory` table. Phase 26 instead writes for real against a disposable file
copy of the dev DB (`tmp/26-spike-db-copy.sqlite3`) — there is no rollback anywhere in
this procedure, because the whole file is throwaway. A future reader must not assume
rollback is a universal spike convention; it is specific to a scenario where the target
table is live, which is not the case here.

## Findings

### Snapshot (D-04) — the dev DB as of 2026-07-27

Read-only probe (`tmp/26_snapshot_probe.py`, never selects `contact_person`/
`contact_email`, never calls a write-style ORM method) run against the real, unmodified
`src/fomo_db.sqlite3` via `./manage.py shell -c "exec(open(...).read())"`. The real DB's
`stat -c '%s %Y'` fingerprint (`946176 1785094461`) was recorded before this probe ran
and is identical afterward — this snapshot made zero writes.

As of 2026-07-27: **31 `CampaignRun` rows (max pk 31)**, **20 `CalendarEvent` rows**,
**11 `CalendarEventTelescopeLabel` companion rows** (all 11 `is_verified=True`, 0
`is_verified=False`), **13 `ObservationRecord` rows** (4 COMPLETED, 8 WINDOW_EXPIRED, 1
CANCELED), and a `CampaignRun.approval_status` breakdown of **30 approved, 1 rejected, 0
pending_review**. Every one of these figures matches the CONTEXT.md/RESEARCH.md
2026-07-26/27 pre-planning snapshot exactly, one day later — the dev DB has not drifted
in the interim.

As of 2026-07-27, the 20 `CalendarEvent` rows split as **9 blank-`url`** (classical,
`telescope`/`instrument`/`start_time` values listed in the SPIKE-02 section below), **11
`https://observe.lco.global/...`** (LCO), **0 `GEM:`-namespaced**, and **0
`CAMPAIGN:`-namespaced**.

As of 2026-07-27, `CampaignRun` pk=1 is `FTS/MuSCAT4`, window 2026-07-07 to 2026-07-21,
resolved site obscode `E10` (Siding Spring-FTS), campaign `Didymos 2026`,
`approval_status=approved`, `run_status=observed`. Its 11 attributed LCO events (ids
53-63) run 2026-07-07 through 2026-07-20, telescope `2m0`/`COJ-2m0`, instrument
`2M0-SCICAM-MUSCAT`, with 8 titled `[EXPIRED]` and 1 titled `[CANCELLED]` — this matches
D-20 exactly.

### D-16 — PROJECT.md's Phase 25 claim does not reproduce

As of 2026-07-27, the maximum `CampaignRun` pk is 31, no run's `telescope_instrument`
contains the string `FT-115`, and there are 0 `CAMPAIGN:`-namespaced `CalendarEvent`
rows. PROJECT.md's "Current Milestone" section states `CampaignRun` pk=34
(`GS-2026A-FT-115`) "now has its 4 per-night `CalendarEvent`s" — this claim does not
reproduce against the live dev DB: there is no pk=34, no `FT-115` row of any pk, and no
`CAMPAIGN:` events exist at all. The dev DB was re-imported after Phase 25's UAT.
Phases 27-29 must not trust PROJECT.md's Phase 25 paragraph for any concrete pk or count.
Correcting PROJECT.md is deliberately **not** this phase's work (D-04/D-16); a separate
todo tracks it, and this spike stays investigation-only.

### SPIKE-02 criterion 2 — per-adapter identity key to run

One subsection per ingest adapter, each naming its construction site by file:line, the
exact `lookup` dict it passes to `insert_or_create_calendar_event()`
(`solsys_code/calendar_utils.py:317`), what the resulting key looks like, how a reader
gets from that key back to a `CampaignRun`, and a confidence tag.

**Classical (`load_telescope_runs.py`).** Construction site:
`solsys_code/management/commands/load_telescope_runs.py:22` (`_START_TIME_MATCH_TOLERANCE
= timedelta(minutes=5)`) and `:207-216` (the `insert_or_create_calendar_event()` call).
The lookup passed is `{'telescope': parsed.telescope, 'instrument': parsed.instrument,
'start_time': start_time}` with `start_time_tolerance=_START_TIME_MATCH_TOLERANCE`
(5 minutes) — **there is no `url` in this lookup at all**. As of 2026-07-27 the 9
blank-`url` `CalendarEvent` rows (ids 44-52; `NTT`/`EFOSC2` and `Magellan-Baade`/`IMACS`
and `Magellan-Clay`/`Lightspeed`) are exactly this adapter's output. A reader gets from
one of these events back to a run only by matching `telescope`/`instrument`/date against
`CampaignRun.telescope_instrument`/`window_start`/`window_end` by hand — there is no
stored FK today. This is precisely why RECON-05's ownership scoping (D-09) cannot lean on
`url` for these 9 rows and must use the new companion FK instead.

Tag: **Confirmed against real rows** (9 real blank-`url` events, listed in the D-04
snapshot above).

**LCO (`sync_lco_observation_calendar.py`).** Construction site:
`solsys_code/management/commands/sync_lco_observation_calendar.py:361`
(`event, action = insert_or_create_calendar_event({'url': url}, fields)`). The lookup is
`{'url': <the LCO portal request url>}`, e.g.
`https://observe.lco.global/requests/4229878`. A reader gets from one of these events
back to a run today only by matching telescope/instrument/date against `CampaignRun`
fields, same as the classical case — no FK exists yet. Tag: **Confirmed against real rows**
(11 real LCO-`url` events, ids 53-63, confirmed above against pk=1).

**Gemini (`sync_gemini_observation_calendar.py`).** Construction site:
`solsys_code/management/commands/sync_gemini_observation_calendar.py:150`
(`url = f'GEM:{prog}/{record.observation_id}'`). The lookup would be
`{'url': f'GEM:{prog}/{obsid}'}`. As of 2026-07-27 there are **0** `GEM:`-namespaced
events in the dev DB (D-18), so this mapping can only be reasoned from source code, not
confirmed against a real row — following Phase 18's D-09 precedent, this is stated
explicitly and never presented at the same confidence as the classical/LCO mappings
above. Tag: **Constructed-input code-path check** (zero real `GEM:` rows exist in the
current snapshot to confirm against; reasoned from the adapter's own key-construction
source).

**Campaign projection (`campaign_views.py`).** Construction sites:
`solsys_code/campaign_views.py:447` (`insert_or_create_calendar_event({'url':
f'CAMPAIGN:{run.pk}'}, fields=event_fields)`, single-night runs) and `:485` (`url =
f'CAMPAIGN:{run.pk}' if not is_range else f'CAMPAIGN:{run.pk}:{night.isoformat()}'`,
range runs, one event per night). The existing bare-key-plus-prefix ownership query
already lives at `campaign_views.py:797`
(`Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')`). As of
2026-07-27 there are **0** `CAMPAIGN:`-namespaced events (D-15), so — same as Gemini —
this mapping is reasoned from source, not confirmed against a real row. Tag:
**Constructed-input code-path check** (zero real `CAMPAIGN:` rows in the current
snapshot). D-15's zero-`CAMPAIGN:`-events finding means the reconciler's own key scheme
(D-09's `RUN:{run_pk}:{date}`) has a clean slate — nothing existing to migrate or
reconcile with.

### Stage-2 and stage-0 inventory (D-05..D-08)

As of 2026-07-27, `CampaignRun` rows with `site IS NULL`: pk=8 (`Hubble\nWFC3/UVIS`),
pk=12 (`HST STIS/COS`), pk=13 (`Swift/UVOT`), pk=21 (`JWST`), pk=26 (`JUICE`) — the five
space-mission rows D-06 names — plus pk=29 (`LCO 1m`) and pk=30 (`LCO 2m`), the two
class-wide rows. **One additional site-less row not in CONTEXT.md's expected list: pk=31
(`FOO / BAR`, `approval_status=rejected`, `site_raw='X05'`)** — a rejected test/probe row
whose free-text site never resolved; it is excluded from the space-mission/class-wide
count above because it is neither (its window is a resolved date range, not `NULL`, and
it is not approved), but it is worth recording explicitly as a small additional data
point the pre-planning discussion did not surface. It has no bearing on D-06's vocabulary
decision.

As of 2026-07-27, `CampaignRun` rows with `window_start IS NULL` (stage 0 — "allocated
but unscheduled" per D-08): pk=4 (`ESO VLT FORS2`), pk=27 (`JWST`), pk=28 (`JWST`) —
exactly the three rows CONTEXT.md predicted.

`CampaignRun` pk=29's window is 80 nights inclusive (2026-... window, `window_end -
window_start + 1 day = 80`). `SITE_TELESCOPE_MAP` (`solsys_code/calendar_utils.py:36-53`)
carries `1m0` at **5** sites (`coj`, `cpt`, `elp`, `lsc`, `tfn`) and `2m0` at **2** sites
(`coj`, `ogg`). Combined with pk=29's 80-night window, naive per-site fan-out for that one
class-wide run alone would be 80 x 5 = **400** events — this is a **computed figure from
real field values**, not an executed DB check (RESEARCH.md's own Evidence Map
distinction). `CampaignRun` pk=26 (JUICE)'s window is 23 days end-minus-start (24 nights
inclusive, 2025-11-02 through 2025-11-25) — the concrete real-data instance behind D-07's
"one spanning event per space-mission window" decision.

### RECON-07 baseline

As of 2026-07-27, there are exactly **19** approved, site-resolved, windowed `CampaignRun`
rows on the `3I/ATLAS` campaign with no blank-`url`-classical-style calendar presence —
matching D-20's roadmap-cited figure exactly. (This baseline is computed by the same
read-only probe; it does not yet account for LCO-url-based presence in the same pass, so
it should be read as "no classical-style calendar row currently exists for these runs,"
consistent with how RECON-07's original figure was derived.)

### SPIKE-04 criterion 4 (a) — the migration applies cleanly

The throwaway migration (`solsys_code/migrations/0008_scratch_canonical_record_probe.py`,
scratch branch `spike/26-canonical-record-probe` only, never committed to the real
phase-26 branch) is **hand-authored**, not `makemigrations`-generated: non-interactive
autodetection cannot tell a rename from a delete-plus-create, and because the OneToOne's
`event_id` is the model's actual primary key, a `DeleteModel`/`CreateModel` pair would
have dropped and recreated the table, destroying the 11 real companion rows this spike's
coexistence evidence depends on (RESEARCH.md Pitfall 4). Its operation list, in order
(migration state is cumulative within one `operations` list, so every operation after the
rename refers to the model by its new name): one `RenameModel` (`CalendarEventTelescopeLabel`
-> `CalendarEventMeta`), then three `AddField` operations — `CalendarEventMeta.run` (the
D-11 prototype's ownership FK), `CampaignRun.source` (D-12/D-13, `default='legacy'`, no
`RunPython` step needed since the backfill value is a single static default), and
`CampaignRun.telescope_class` (D-06's widened vocabulary, nullable/blankable).

Applied against `tmp/26-spike-db-copy.sqlite3` (a real copy of the dev DB) via
`python manage.py migrate solsys_code`, verbatim output:

```
Operations to perform:
  Apply all migrations: solsys_code
Running migrations:
  Applying solsys_code.0008_scratch_canonical_record_probe... OK
```

Row counts (`CampaignRun`, `CalendarEvent`, companion-record — via `tmp/26_row_counts.py`,
which imports the companion model by its post-rename name first, falling back to the
pre-rename name) before and after the migration: **31 20 11** both times, byte-identical
— zero row loss, confirming `RenameModel` preserved the OneToOne-primary-key table rather
than dropping and recreating it. This is the exact migration shape (`RenameModel` before
`AddField`, referencing the post-rename name throughout) Phase 27 will write for real —
proven here against a real copy of the dev DB, not just reasoned about. Tag: **Confirmed
against real rows** (the scratch copy is a real copy of the dev DB; the pre/post row
counts are the real 31 `CampaignRun`/20 `CalendarEvent`/11 companion rows, not factory
fixtures).

### SPIKE-04 criterion 4 (b) — measured rename blast radius

Before fixing either class-name consumer, `./manage.py check` was run against the
migrated scratch copy (with the rename/field edits in place on `models.py` but
`admin.py`/`sync_lco_observation_calendar.py` still importing the old class name). It
failed loudly, exactly as D-02's analytical prediction anticipated, via Django's admin
autodiscovery (`app_config.ready()` -> `autodiscover_modules('admin', ...)` ->
`import_module('solsys_code.admin')`), verbatim tail:

```
  File "/home/tlister/git/fomo_devel/solsys_code/admin.py", line 4, in <module>
    from solsys_code.models import CalendarEventTelescopeLabel, CampaignRun
ImportError: cannot import name 'CalendarEventTelescopeLabel' from 'solsys_code.models' (/home/tlister/git/fomo_devel/solsys_code/models.py)
```

This confirms integration point #1 (`solsys_code/admin.py:4,28,41`) fails loudly at
Django startup, not silently — and because `manage.py migrate` itself calls the same
system-check machinery, the rename would have blocked `migrate` too until fixed. Fixing
`admin.py` (the import, the `ModelAdmin` subclass name, and the `admin.site.register`
call) and `sync_lco_observation_calendar.py` (the import and the
`.objects.update_or_create(event=event, ...)` call — only the class name changed; `event=`
and every `related_name`/prefetch string were left untouched) brought `./manage.py check`
back to exit 0. Integration point #2
(`solsys_code/management/commands/sync_lco_observation_calendar.py:18,369`) is confirmed
the same way, at command import.

The remaining checklist rows were measured by running the named, narrow seven-module test
selection twice (`test_admin`, `test_sync_lco_observation_calendar`, `test_calendar_template`,
`test_campaign_models`, `test_campaign_views`, `test_campaign_approval`,
`test_load_telescope_runs` — deliberately excluding `test_ephem_utils.py`/`test_views.py`,
per RESEARCH.md Pitfall 3's segfault finding) against the migrated scratch copy, once
before touching any test file and once after applying the rename to the four affected
test modules. Full output is captured verbatim in the throwaway
`tmp/26-rename-measurement.txt`.

**Pre-fix run: 177 tests collected, 5 errors.** Three modules failed to import outright
(`test_sync_lco_observation_calendar`, `test_calendar_template`, `test_load_telescope_runs`
— each still importing `CalendarEventTelescopeLabel` by name) and two individual tests in
`test_admin.py` failed with `django.urls.exceptions.NoReverseMatch: Reverse for
'solsys_code_calendareventtelescopelabel_changelist' not found` — Django derives an admin
changelist's reverse-URL name from the model's lowercased class name, so the rename changes
this name too, a consumer the original four-point checklist did not name.

**Post-fix run (after renaming the class references in all four test modules, plus the
reverse-URL target in `test_admin.py`): 265 tests, 0 failures, exit 0.**

| # | Integration point | Predicted at risk | What actually happened | How it failed |
|---|--------------------|--------------------|--------------------------|----------------|
| 1 | `solsys_code/admin.py:4,28,41` (import, `ModelAdmin` subclass, `admin.site.register`) | Yes | `ImportError: cannot import name 'CalendarEventTelescopeLabel'` at Django startup (Task 2 evidence, reproduced here at command-import time too) | Loudly, at import |
| 2 | `solsys_code/management/commands/sync_lco_observation_calendar.py:18,369` (import, `.objects.update_or_create(event=event, ...)`) | Yes | Same `ImportError`, at test-module import (`test_sync_lco_observation_calendar.py` failed to import for the same reason before its own rename) | Loudly, at import |
| 3 | `solsys_code/views.py:114` `.prefetch_related('telescope_label_meta')` | No (safe by construction — `related_name` locked unchanged) | **Untouched.** `test_calendar_template.py`'s `self.client.get(reverse('calendar:calendar'))` tests pass post-fix; a separate non-test `django.test.Client()` fetch of `/calendar/?year=2026&month=7` (guard confirmed pointed at the scratch copy) returned **HTTP 200** | N/A — no failure |
| 4 | `src/templates/tom_calendar/partials/calendar.html:228,244` `event.telescope_label_meta.is_verified` | No (safe by construction, same reason as #3) | **Untouched.** `test_calendar_template.py`'s dashed-border/`is_verified=False` fixture-based assertions pass post-fix (part of the seven-module suite) | N/A — no failure |
| 5 (not in the original 4-point checklist) | `test_admin.py`'s `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` | Not named by RESEARCH.md's original grep | `NoReverseMatch` — Django derives this URL name from the model's lowercased class name, so it changes with the class rename even though nothing in `admin.py` had to change to fix it (the `ModelAdmin`'s own registration change is what shifts the derived name) | Loudly, per-test |
| 6 (not in the original 4-point checklist) | Class-name references inside `test_load_telescope_runs.py`, `test_sync_lco_observation_calendar.py`, `test_calendar_template.py` themselves | Not named by RESEARCH.md's original grep (its own grep excluded `tests/`) | `ImportError` at test-module collection for the first two; `test_calendar_template.py` also directly constructs `CalendarEventTelescopeLabel.objects.create(...)` fixture rows that needed the rename applied | Loudly, at import/collection |

**Verdict on D-02's analytical prediction: confirmed-with-additions.** The core prediction
— that `related_name='telescope_label_meta'` being locked unchanged makes the view
`prefetch_related()` string and the calendar template's accessor safe *by construction*,
while the two class-name imports are the only things genuinely at risk, and both fail
loudly — is **confirmed exactly**: rows #1-#4 above match the prediction precisely, and
both real risks failed as loud `ImportError`s, never silently. The **addition** is that the
four-point checklist, scoped to non-test application code, missed two more class-name
consumers that also fail loudly once you include the test suite itself: the admin
reverse-URL name (#5) and the four test modules' own direct references to the class (#6).
Neither addition changes the underlying architectural conclusion (`related_name` is what
matters, not the class name, for runtime behavior) — but a rename executed without
updating these five additional sites would still leave the test suite red even after the
two "real" application consumers are fixed. Phase 27's rename checklist should therefore
name six sites, not four: `admin.py`, `sync_lco_observation_calendar.py`,
`test_admin.py`'s reverse-URL target, and the class-name references inside
`test_load_telescope_runs.py`/`test_sync_lco_observation_calendar.py`/
`test_calendar_template.py`.

**Confidence tagging, per VALIDATION.md's Evidence Map:** the green seven-module test run
is tagged **Constructed-input code-path check** — Django's `TestCase` machinery builds its
own isolated in-memory test database from factories/fixtures, so this proves the rename
does not break *tested behaviors*, not that it held against the real 11 companion rows
specifically. That second proof is Task 2's migration-application finding (criterion 4(a)
above) plus plan 26-02's `IntegrityError` coexistence script — both run against the real
scratch DB copy.

The `django.test.Client()`-based `/calendar/?year=2026&month=7` fetch (run outside the test
framework, `SERVER_NAME='localhost'` supplied because `ALLOWED_HOSTS=[]` in this project's
`settings.py` does not include the test framework's default `testserver` host outside
`manage.py test`) returned **HTTP 200** with **0** literal `telescope_label_meta`
occurrences in the decoded response body — this 0 is expected, not a failure signal: the
template only ever evaluates `event.telescope_label_meta.is_verified` inside an `{% if %}`
conditional, never renders the accessor name itself as text. The actually-meaningful signal
from this check is the 200 status itself: `prefetch_related('telescope_label_meta')`
(integration point #3) raises a `FieldError` at query-execution time if the `related_name`
were broken, and Django does not swallow that — a 200 here is real evidence the accessor
still resolves. This is a non-interactive corroborating data point only, per RESEARCH.md;
it does **not** replace the actual browser-based `/calendar/` load, which is plan 26-02's
task 1.

**Known gap, stated explicitly (VALIDATION.md's Manual-Only Verifications table):** every
one of the 11 real companion rows currently has `is_verified=1` — zero real rows exercise
the `is_verified=False` dashed-border fallback branch (`calendar.html:228,244`). This
finding's evidence for that specific branch rests on **(b)** —
`test_calendar_template.py`'s existing fixture-based coverage (which directly constructs
`is_verified=False` rows and asserts the dashed-border CSS class appears), already exercised
as part of the seven-module suite above — not on a temporarily-flipped real-copy row. Plan
26-02's browser-based `/calendar/` load may still choose to construct such a row on the
scratch copy for a second, visual confirmation; that is out of this task's scope.

**Deferred, out-of-scope finding (not a Phase 26 deviation to fix):** `ruff check .` and
`ruff format --check .` run against the full repository surface pre-existing failures in
files this plan never touched — `src/fomo/settings.py`, four `docs/notebooks/pre_executed/*.ipynb`
files, and two `.planning/quick/260619-f7u-*/` scripts — confirmed identical against the
pre-Phase-26 commit (`77e16b5`), i.e. present before this plan started. Every file this
plan actually created or edited (`solsys_code/models.py`,
`solsys_code/migrations/0008_scratch_canonical_record_probe.py`, `solsys_code/admin.py`,
`solsys_code/management/commands/sync_lco_observation_calendar.py`, and the four edited
test modules) is individually `ruff check`/`ruff format --check` clean. Logged to
`.planning/phases/26-canonical-record-spike/deferred-items.md` per the scope-boundary rule
rather than fixed here.

## Recommendation

<!-- completed in plan 26-03 -->

## Durable summary

<!-- completed in plan 26-03 -->
