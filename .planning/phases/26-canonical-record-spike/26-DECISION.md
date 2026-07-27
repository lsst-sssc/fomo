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

### SPIKE-04 criterion 4 (c) — manual /calendar/ load

A human loaded `http://127.0.0.1:8765/calendar/?year=2026&month=7` in a browser against the
migrated scratch copy (`tmp/26-spike-db-copy.sqlite3`, confirmed by the DB-path guard
immediately before `runserver` started), with `CalendarEventMeta` row pk=53 (event id 53,
`CampaignRun` pk=1's `[CANCELLED] 2m0 2M0-SCICAM-MUSCAT` event) temporarily flipped to
`is_verified=False` on the scratch copy beforehand.

**Pass conditions met, reported verbatim by the human:** the page "rendered fine ... 11 LCO
events appear as expected, along with the classical events. ... the event on July 7 does
appear with [CANCELLED], dashed border and the hover text says (paraphrasing) 'Telescope
label is an estimate - could not be verified against LCO API'." HTTP status, event count, and
dashed-border result all matched expectation. The hover text is a second, independent signal
that the flipped row actually took the unverified-label template branch, not merely a border
CSS coincidence.

**Two observations recorded as observed-and-out-of-scope, not findings against the rename:**
1. The browser's address bar normalized `?year=2026&month=7` to `?year=2026` during the load.
   The rename touched no URL routing or view-parameter handling, so this is pre-existing
   `/calendar/` view behavior, unrelated to D-02/D-03's model rename.
2. The human's browser (Konqueror) did not surface event `url` values on hover/click. This is
   not one of the three pass conditions (HTTP status, event count, dashed-border) and is not
   rename-related; it evidences only that this particular browser's UI doesn't expose the
   `url` field visually, nothing about `prefetch_related('telescope_label_meta')` or the
   `event.telescope_label_meta` accessor (integration points #3/#4), which the 200 status and
   the correctly-rendered dashed border already confirm still resolve.

**Confidence tag:** the HTTP-200/event-count/no-traceback portion of this result is
**Confirmed against real rows** (`CampaignRun` pk=1's real 11 LCO events plus the real
classical events, all unmodified). The dashed-border portion specifically is tagged
**Constructed-input code-path check** — a deliberately-constructed check, not a real-row
observation: `CalendarEventMeta` row pk=53 was temporarily flipped to `is_verified=False`
on the scratch copy for this load and restored to `True` immediately afterward, because
D-20 confirms all 11 real companion rows are `is_verified=1` and so zero real rows
currently exercise that branch. Plan 26-01's
`django.test.Client()`-based fetch of the same URL (also returning HTTP 200) is a corroborating
non-interactive second data point recorded there, not a replacement for this manual load — this
manual browser load is the one step in the phase with no automated substitute (26-VALIDATION.md
"Manual-Only Verifications").

The temporarily-flipped row was restored to `is_verified=True` (confirmed:
`CalendarEventMeta.objects.filter(is_verified=False).count() == 0` immediately afterward) and
the background `runserver` process was stopped before any further write in this plan; `src/
fomo_db.sqlite3`'s fingerprint (`946176 1785094461`) was unchanged throughout.

### SPIKE-01 criterion 1 — source vocabulary and constraint coexistence

Executed via `tmp/26_integrity_check.py` (`python manage.py shell < tmp/26_integrity_check.py`,
captured verbatim to `tmp/26-integrity-check.txt`) against the migrated scratch copy, after
confirming the DB-path guard. Four blocks, five PASS lines, zero FAIL lines:

```
=== Block (A): constraint inventory ===
  UniqueConstraint name='unique_campaign_run_resolved_window' fields=('campaign', 'telescope_instrument', 'window_start', 'window_end') condition=<Q: (AND: ('window_start__isnull', False))>
  UniqueConstraint name='unique_campaign_run_tbd_natural_key' fields=('campaign', 'telescope_instrument', 'contact_person') condition=<Q: (AND: ('window_start__isnull', True))>
  CheckConstraint name='campaign_run_window_start_end_null_together' condition=<Q: (OR: (AND: ('window_end__isnull', True), ('window_start__isnull', True)), (AND: ('window_end__isnull', False), ('window_start__isnull', False)))>
CNAMES: ['campaign_run_window_start_end_null_together', 'unique_campaign_run_resolved_window', 'unique_campaign_run_tbd_natural_key']
NEWFIELD_IN_CONSTRAINTS: False
PASS: neither source nor telescope_class appears in any CampaignRun constraint field set.

=== Block (B): positive case -- source + run-FK coexistence ===
BEFORE: pk=1 source field value = 'legacy'
PASS: CampaignRun pk=1 source=LEGACY saved, no IntegrityError.
LINKED 11
PASS: linked 11 companion rows to run pk=1 with no IntegrityError.

=== Block (C): negative control -- unique_campaign_run_resolved_window ===
PASS: unique_campaign_run_resolved_window still fires unmodified (source is not in its key): UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.window_start, solsys_code_campaignrun.window_end

=== Block (D): negative control -- unique_campaign_run_tbd_natural_key ===
Using TBD-window run pk=4 as the natural-key source (contact_person copied, never printed).
PASS: unique_campaign_run_tbd_natural_key still fires unmodified (source is not in its key): UNIQUE constraint failed: solsys_code_campaignrun.campaign_id, solsys_code_campaignrun.telescope_instrument, solsys_code_campaignrun.contact_person

=== Summary ===
AFTER: pk=1 source field value = 'legacy'
AFTER: CalendarEventMeta rows with run_id=1: 11
```

**Constraint names and field tuples, quoted verbatim from `solsys_code/models.py:120-160`:**
`unique_campaign_run_resolved_window` — `fields=('campaign', 'telescope_instrument',
'window_start', 'window_end')`, condition `window_start__isnull=False`; and
`unique_campaign_run_tbd_natural_key` — `fields=('campaign', 'telescope_instrument',
'contact_person')`, condition `window_start__isnull=True`. Block (A)'s printed boolean
confirms neither `source` nor `telescope_class` was added to either constraint's field tuple
(nor to the third, `campaign_run_window_start_end_null_together`, which is a `CheckConstraint`
with no field tuple at all) — **attribution (the `run` FK), not the constraint, is what
connects same-physical-run rows from different sources**, exactly as CONTEXT.md's locked
constraint states.

Block (B) is SPIKE-01's literal positive case: `CampaignRun` pk=1 given an explicit
`source=LEGACY` value, and all 11 of its real LCO-sourced companion rows (matched by
`event__url__startswith='https://observe.lco.global'`) given a `run` FK back to pk=1 — both
writes committed for real against the disposable scratch copy, zero `IntegrityError`s. This is
also the attribution state plan 26-02's own task 3 (D-11 prototype) depends on, per the plan's
explicit instruction not to skip or roll this step back.

Blocks (C) and (D) are the two negative controls SPIKE-01's wording requires — both existing
partial unique constraints still fire, unmodified, on a genuine duplicate differing only by
`source`, proving `source` was never silently absorbed into either constraint's key. Each
`transaction.atomic()` wrapper exists only so the expected `IntegrityError` doesn't poison the
connection for the next statement — the failed insert left no row behind in either case
(re-running the whole script a second time in this session reproduced the identical five
PASS/zero FAIL result, confirming no partial row was left over from either negative control).

**Evidence posture, restated for this specific check:** unlike Phase 18's rolled-back
`transaction.atomic()` pattern (used because that spike ran against the live `Observatory`
table), this script writes for real against `tmp/26-spike-db-copy.sqlite3`, a disposable file
copy — there is no rollback anywhere in the positive-case writes, only in the two
negative-control blocks, and only to protect the connection from a poisoned transaction, not to
undo the writes.

Tag: **Confirmed against real rows** (real `CampaignRun` pk=1, its real 11 LCO-sourced
companion rows, and the real TBD-window run pk=4 used for the second negative control). `src/
fomo_db.sqlite3`'s fingerprint (`946176 1785094461`) was unchanged throughout this task.

### SPIKE-03 criterion 3 — canonical event key and the adopt-vs-gap-fill comparison

Executed via `tmp/26_reconciler_prototype.py`, run once per scenario
(`RECONCILER_SCENARIO=<adopt|gapfill|rejected-baseline> python manage.py shell <
tmp/26_reconciler_prototype.py`) against its own copy of the post-task-2 scratch DB
(`tmp/26-adopt-copy.sqlite3`, `tmp/26-gapfill-copy.sqlite3`,
`tmp/26-rejected-baseline-copy.sqlite3`), with the DB-path guard re-run before each
scenario's `local_settings.py` repoint. The script imports
`insert_or_create_calendar_event()` from `solsys_code.calendar_utils` unchanged and
defines no substitute write helper; `git diff --quiet -- solsys_code/calendar_utils.py`
on the scratch branch confirms the shared helper was not modified. It imports neither
`solsys_code.views` nor the ephemeris module, so it never triggers the SPICE furnish.

**The `RUN:{run_pk}:{date}` key, D-09/D-10.** The reconciler's own key namespace is
`RUN:{run_pk}:{date}`, where `{date}` is always the **site-local observing night**
(D-10), derived by converting the event's `start_time` into the site's timezone and
taking the local calendar date — never the naive UTC date of whatever timestamp the
current stage happens to produce.

**Measured gap, surfaced by running this for real (not part of D-10's own claim, a
separate finding):** `CampaignRun` pk=1's real site (`Observatory` obscode `E10`,
"Siding Spring-Faulkes Telescope South") has a **blank `timezone` field** in this dev DB
copy — confirmed by direct query, not assumed. The prototype falls back to
`'Australia/Sydney'` (this site's documented IANA zone, per `solsys_code/telescope_runs.py`'s
own `SITES` mapping and CLAUDE.md's "Timezones" constraint) so the D-10 comparison below
could still run against real event timestamps, and prints this substitution explicitly
rather than silently assuming the field was populated. **Phase 27 should backfill
`Observatory.timezone` for E10 before the real reconciler ships** — a blank timezone
would make any real site-local-night derivation raise, the same way
`solsys_code/telescope_runs.py`'s own `sun_event()` already guards against and raises
`ValueError` for a blank `site.timezone`.

**The measured UTC-vs-site-local comparison (D-10 evidence), identical across all three
scenario copies since it reads the same 11 real, unmodified LCO events every time:**

```
event pk=53 start_time=2026-07-07T00:00:00+00:00 utc_date=2026-07-07 local_night=2026-07-07 [same]
event pk=54 start_time=2026-07-08T14:08:19+00:00 utc_date=2026-07-08 local_night=2026-07-09 [DIFFERS]
event pk=55 start_time=2026-07-10T00:00:00+00:00 utc_date=2026-07-10 local_night=2026-07-10 [same]
event pk=56 start_time=2026-07-11T00:00:00+00:00 utc_date=2026-07-11 local_night=2026-07-11 [same]
event pk=57 start_time=2026-07-12T00:00:00+00:00 utc_date=2026-07-12 local_night=2026-07-12 [same]
event pk=58 start_time=2026-07-14T00:00:00+00:00 utc_date=2026-07-14 local_night=2026-07-14 [same]
event pk=59 start_time=2026-07-16T00:00:00+00:00 utc_date=2026-07-16 local_night=2026-07-16 [same]
event pk=60 start_time=2026-07-17T00:00:00+00:00 utc_date=2026-07-17 local_night=2026-07-17 [same]
event pk=61 start_time=2026-07-18T00:00:00+00:00 utc_date=2026-07-18 local_night=2026-07-18 [same]
event pk=62 start_time=2026-07-19T00:00:00+00:00 utc_date=2026-07-19 local_night=2026-07-19 [same]
event pk=63 start_time=2026-07-20T00:00:00+00:00 utc_date=2026-07-20 local_night=2026-07-20 [same]
UTC_COVERED (11): 2026-07-07, 2026-07-08, 2026-07-10, 2026-07-11, 2026-07-12, 2026-07-14, 2026-07-16, 2026-07-17, 2026-07-18, 2026-07-19, 2026-07-20
LOCAL_COVERED (11): 2026-07-07, 2026-07-09, 2026-07-10, 2026-07-11, 2026-07-12, 2026-07-14, 2026-07-16, 2026-07-17, 2026-07-18, 2026-07-19, 2026-07-20
COVERED_SETS_DIFFER: True
UTC_UNCOVERED (4): 2026-07-09, 2026-07-13, 2026-07-15, 2026-07-21
LOCAL_UNCOVERED (4): 2026-07-08, 2026-07-13, 2026-07-15, 2026-07-21
UNCOVERED_SETS_DIFFER: True
```

Only event pk=54 (`start_time=2026-07-08T14:08:19Z`) diverges — its naive UTC date is
2026-07-08 but its site-local night (Sydney, UTC+10, no July DST) is 2026-07-09, because
14:08 UTC is after local midnight (00:08 local the next calendar day). This is the
direct, measured instance of the exact D-10 mechanism CONTEXT.md describes for this
site: a UTC-date key and a site-local-night key disagree for a real event's real
timestamp. The knock-on effect: the two uncovered-night sets have the **same count (4)
but a different specific member** — the naive UTC derivation calls 2026-07-09
uncovered (because it thinks pk=54 covers 07-08), while the site-local derivation calls
**2026-07-08** uncovered instead (because pk=54's site-local night is actually 07-09).
D-11's predicted count of 4 held exactly under both derivations, but *which* night is
the 4th uncovered one is derivation-dependent — precisely the measured evidence
`UNCOVERED_SETS_DIFFER: True` records, turning D-10 from an assertion into something
observed. The scenarios below key on the **site-local** uncovered set
(`2026-07-08, 2026-07-13, 2026-07-15, 2026-07-21`), consistent with D-09/D-10.

**Three-way comparison, measured exactly as D-11 predicted (15 / 15 / 26):**

| Scenario | In-window count | Expected | Matches | Pass-1 tally | Pass-2 (idempotency) |
|----------|-----------------|----------|---------|--------------|------------------------|
| Adopt | 15 | 15 | Yes | `created=4, updated=11, unchanged=0` | `IDEMPOTENT_RERUN adopt created=0 updated=0` |
| Gap-fill | 15 | 15 | Yes | `created=4, updated=0, unchanged=0` | `IDEMPOTENT_RERUN gapfill created=0 updated=0` |
| Rejected baseline | 26 | 26 | Yes | `created=15, updated=0, unchanged=0` | `IDEMPOTENT_RERUN rejected-baseline created=0 updated=0` |

**Adopt** — the 11 real LCO events were updated in place under their own existing
`https://observe.lco.global/...` urls (an `updated` action each, since the adoption
stamp written into `description` differs from the original LCO-sync text), and the 4
site-local-uncovered nights were newly minted under `RUN:1:2026-07-08`,
`RUN:1:2026-07-13`, `RUN:1:2026-07-15`, `RUN:1:2026-07-21`. Companion `run` FK count in
window: 11 (all inherited from task 2's attribution write; the 4 freshly-minted events
have no companion row at all, since this prototype never creates one for them — see the
ownership discussion below). Key list: the 11 unchanged LCO urls plus the 4 `RUN:1:`
keys.

**Gap-fill** — the 11 real LCO events were **never touched at all** (0 created, 0
updated against them); only the same 4 site-local-uncovered nights were newly minted
under the identical `RUN:1:{date}` keys as the adopt scenario. Companion `run` FK count
in window: 11 (unchanged from task 2's inherited state; this scenario's code path never
writes to `CalendarEventMeta`, so the 11 originals' `modified` timestamp is never
touched by the gap-fill path — stages 3-4 for those 11 nights would keep coming from the
LCO sync command until v2.3 rewires the adapters, exactly as D-11 anticipates).

**Rejected baseline** — the reconciler minted its own `RUN:1:{date}` key for **every one
of the 15 window nights** regardless of existing LCO coverage, leaving the 11 originals
untouched. Result: 26 total events in-window (11 pre-existing LCO-keyed + 15 fresh
`RUN:1:{date}`-keyed, including a second, separate event for every one of the 11
already-covered nights) — the concrete, counted instance of the visible double-booking
ATTRIB-06 exists to prevent, not just an assertion.

**Idempotency (D-09's stage-stable-key claim).** Each scenario was run a second time
against the same copy; all three reported `created=0 updated=0` on the re-run
(`IDEMPOTENT_RERUN <scenario> created=0 updated=0` above), and the in-window event count
was identical before and after the re-run in every case. This holds because both
`minted_fields()` and `adopted_fields()` are pure functions of `run1`/`night`/`event.pk`
— none depend on wall-clock time or any other non-deterministic input — so a second call
with the same lookup key always compares against identical stored values and takes the
`unchanged` branch of `insert_or_create_calendar_event()`.

**Identity vs. ownership (D-09).** The namespaced `url` gives an event its *identity*
(what `insert_or_create_calendar_event()`'s `lookup` matches on); the companion `run` FK
gives it *ownership*. The rule holds exactly as CONTEXT.md states it: **no companion
row, or a companion row with `run` unset, means "not mine, never touch."** This is
provable today without any prototype write at all — the 9 real classical `CalendarEvent`
rows (blank `url`, `load_telescope_runs` adapter) have **no companion row whatsoever**,
so neither this reconciler nor any future one may claim them via the `run` FK path; they
are simply outside the ownership mechanism entirely. The adopt scenario's own newly
minted `RUN:1:{date}` events are a related, smaller instance of the same point: this
prototype never creates a `CalendarEventMeta` row for them, so under the "not mine"
rule they are not owned by anything either, despite carrying a `RUN:1:`-namespaced url —
namespace membership alone does not confer ownership; a real reconciler module would
need to explicitly create/attach a companion row with `run` set for its own minted
events if it wants to claim them, which this investigation-only prototype deliberately
does not do (out of this plan's scope, per CONTEXT.md's Claude's Discretion note on how
deep to take attribution-scoring prototyping).

**On D-05's fan-out figure:** the 80×5=400 stage-2 class-wide fan-out number recorded in
plan 26-01 (`CampaignRun` pk=29's real 80-night window × `SITE_TELESCOPE_MAP`'s real
5-site `1m0` count) is a **computed figure from real field values**, not an executable
check the way this task's three scenario runs are — worth restating that distinction
here since both numbers appear in the same decision doc and could otherwise be read as
equally "executed."

**Confidence tags:** the adopt-vs-gap-fill-vs-rejected-baseline comparison itself is
**Confirmed against real rows** (`CampaignRun` pk=1's real 15-night window, its real 11
LCO events, run three times against three independent scratch copies). The
`Australia/Sydney` timezone substitution is a stated, explicit fallback for a real,
measured gap in the `Observatory` row (not itself a "confirmed against real rows"
claim about the site's actual physical timezone — Siding Spring genuinely is UTC+10 in
July, but the *dev-DB row* does not record it).

**The recommendation between adopt and gap-fill is settled in plan 26-03, not here** —
this task's job was to produce the measured comparison plan 26-03's `## Recommendation`
section reasons from, per CONTEXT.md's "prototype both and recommend after" framing.

`src/fomo_db.sqlite3`'s fingerprint (`946176 1785094461`) was unchanged throughout this
task.

### SPIKE-03 gap closure — queue-run projection, measured

26-VERIFICATION.md reopened SPIKE-03 because the locked `RUN:{run_pk}:{date}` key scheme
was settled for classically-scheduled runs only, leaving open whether -- and how -- a
queue-scheduled run should be projected onto the calendar at all. This subsection is the
option-(b) measured closure the verification report's `missing` list named: the same kind
of measured evidence D-11 produced for the write-strategy question, produced here for the
queue-run projection question, against real `CampaignRun` pk=1 and the real dev DB.

#### Run-type inventory (which runs this question actually affects)

Probe run 2026-07-27 (`tmp/26_queue_inventory_probe.py`, read-only, against the
unmodified real `src/fomo_db.sqlite3`). **Judgment applied to real field values, not an
executed check** -- the `source` field that would decide this mechanically does not exist
until Phase 27's CANON-01, so every row's `telescope_instrument`/`site_raw` was read by a
human-auditable rule (full per-row inputs and assigned category recorded in
`tmp/26-queue-inventory.txt`, not just totals) distinguishing a shared-queue-scheduled
facility (LCO network, Gemini, SOAR, ESO VLT) from a run that owns a specific awarded
night at one telescope (the NTT/EFOSC2/Magellan family plus every other single-facility
ground run in this dev DB -- HCT, Palomar P200, Apache Point, IRTF, Deep Sky Chile, Joan
Oró -- none of which schedule through a shared queue network) from a space mission with
no ground site at all.

As of 2026-07-27, of the 31 `CampaignRun` rows: **12 QUEUE**, **12 CLASSICAL**, **7
SPACE**. `CampaignRun` pk=1 (`FTS/MuSCAT4`) and both class-wide rows pk=29 (`LCO 1m`) and
pk=30 (`LCO 2m`) are all **QUEUE**.

The RECON-07 baseline (the 19 approved, site-resolved, windowed `CampaignRun` rows on the
3I/ATLAS campaign with no pre-existing blank-`url`-classical-style calendar presence --
recomputed the same way the original RECON-07 Finding above did, and matching its figure
of 19 exactly) splits **8 QUEUE, 11 CLASSICAL, 0 SPACE**. This is the figure that decides
whether queue-run projection is a corner case or the dominant mechanism for the roadmap's
flagship "19 invisible runs become visible" criterion: it is neither -- a genuine mixed
population, where a majority (11/19) are classically-scheduled single-facility runs
already covered by the locked per-night key scheme, and a substantial minority (8/19,
including the phase goal's own anchor example pk=1) are queue-scheduled runs for which
the projection mechanism was, until this closure, still an open question.

#### The same category error already exists in campaign_gap.claimed_dates()

Calling `claimed_dates()` (imported unchanged from `solsys_code.campaign_gap`, never
modified by this investigation-only plan) with `CampaignRun` pk=1's own campaign, target,
and resolved site (obscode `E10`) returns **15** claimed dates inside pk=1's 15-night
window -- every night of the window. The 11 real LCO events actually cover only **11**
site-local nights (converting each event's `start_time` into the site's timezone and
taking the local calendar date, substituting `Australia/Sydney` explicitly because
obscode `E10`'s `timezone` field is blank in this dev DB, exactly as the earlier SPIKE-03
Finding above already recorded and flagged for a Phase 27 backfill). The set difference is
the same 4 nights the D-11 prototype minted: `2026-07-08`, `2026-07-13`, `2026-07-15`,
`2026-07-21`.

The responsible code, `solsys_code/campaign_gap.py:207-209`'s ground-branch loop inside
`claimed_dates()`, quoted verbatim with real line numbers read from the file:

```
207:        n_days = (run.window_end - run.window_start).days + 1
208:        for i in range(n_days):
209:            claimed.add(run.window_start + timedelta(days=i))
```

**This is the code-level distinguishing fact this closure needs.** For a ground run,
`claimed_dates()`'s loop adds every date in the inclusive `[window_start, window_end]`
range to `claimed` -- it does not distinguish a classically-scheduled run (which
genuinely does own every night in its window) from a queue-scheduled run (whose window is
a span during which observations *could* happen, not a set of owned nights). The same
category error the domain correction identified in the reconciler's own key scheme
already exists, today, in shipped code one module over from the calendar: a decision to
stop minting per-night calendar events for queue runs does not by itself make
`campaign_gap`'s coverage-gap analysis agree with the calendar. Correcting
`claimed_dates()` is **not** this phase's work (investigation-only, flag-not-fix, per the
same D-16/PROJECT.md precedent) -- the requirement that would own it is v2.3's GAPB-01.

#### D-05's 400-event figure

Computed figures from real field values, not an executed DB check (26-VALIDATION.md's own
Evidence Map distinction): `CampaignRun` pk=29's real window is **80** nights inclusive;
`SITE_TELESCOPE_MAP` (`solsys_code/calendar_utils.py:36-53`) carries `1m0` at **5** sites
(`coj`, `cpt`, `elp`, `lsc`, `tfn`). The three arithmetic results a reader needs: naive
per-site-per-night fan-out is **80 x 5 = 400**; per-day (one class-wide event per day,
no site) is **80**; whole-window span (one event covering the entire window) is **1**.

The **site-fanout half** of D-05 -- a single class-wide event, not one per candidate site
-- is independently verified against the real `SITE_TELESCOPE_MAP` and is not in question
here; it is not re-measured by this closure. The **per-day half** (whether pk=29's 80-day
window should produce 80 daily events or a single spanning event) is exactly what the
domain correction reopened, and is not re-verdicted here -- the verdict on which of these
three figures is right for a queue-scheduled class-wide run is plan 26-05's task 1
decision.

#### Three-way comparison against pk=1's real window

Executed via `tmp/26_queue_projection_probe.py`, one run per scenario
(`QUEUE_SCENARIO=<span|none|per-night> python manage.py shell < tmp/26_queue_projection_probe.py`)
against its own disposable copy of the dev DB (`tmp/26-queue-span-copy.sqlite3`,
`tmp/26-queue-none-copy.sqlite3`, `tmp/26-queue-pernight-copy.sqlite3`), with the DB-path
guard printed and asserted before every write. The script imports
`insert_or_create_calendar_event()` from `solsys_code.calendar_utils` unchanged and
defines no substitute write helper; it imports neither `solsys_code.views`,
`solsys_code.ephem_utils`, nor `solsys_code.campaign_views`, reproducing
`_calendar_event_title()` (`campaign_views.py:392-401`) and the `event_fields` dict
(`campaign_views.py:430-436`) inline. Tagged **Confirmed against real rows** throughout:
real `CampaignRun` pk=1, its real 15-night window, its real 11 LCO events, three
independent real copies of the dev DB.

| Scenario | In-window count | Pass-1 tally | Idempotent re-run | `LCO_ROWS_UNTOUCHED` | `KEY_SET_STABLE` under narrowing |
|---|---|---|---|---|---|
| `span` (bare `RUN:1` key, whole-window span) | 12 | `created=1 updated=0 unchanged=0` | `IDEMPOTENT_RERUN span created=0 updated=0` | `True` | `True` (no orphaned keys) |
| `none` (mint nothing for the queue run) | 11 | `created=0 updated=0 unchanged=0` | `IDEMPOTENT_RERUN none created=0 updated=0` | `True` | `True` (trivially -- no `RUN:` keys ever exist) |
| `per-night` (rejected candidate, `RUN:1:{date}` per uncovered night) | 15 | `created=4 updated=0 unchanged=0` | `IDEMPOTENT_RERUN per-night created=0 updated=0` | `True` | **`False`** -- `RUN:1:2026-07-21` orphaned |

**The window-narrowing key-stability probe is the direct, measured answer to which
candidate actually satisfies SPIKE-03's "stable across all four pipeline stages" claim
for a queue run.** After narrowing pk=1's window by one night at each end
(`window_start` 2026-07-08, `window_end` 2026-07-20, a real staff window edit and the
closest available stand-in for a stage-3 narrowing) and re-running each scenario once
more: `span`'s single `RUN:1` key is untouched by the narrowing (only its `start_time`/
`end_time` are updated to the narrowed window, an `updated=1` action, quoted verbatim
from the probe's `PASS3_AFTER_NARROWING span created=0 updated=1 unchanged=0` line) --
the key itself never depends on the window's edges. `per-night`'s uncovered-night set
recomputes against the narrowed window as `2026-07-08, 2026-07-13, 2026-07-15` (three
nights, `2026-07-21` having fallen outside the narrowed window) -- the row already minted
under `RUN:1:2026-07-21` is now **orphaned**: present in the database but no longer
corresponding to any night of the run's current window, with no mechanism in this
prototype (or in the shipped `insert_or_create_calendar_event()` contract it reuses
unchanged) that retires it. `none` has no `RUN:` key at all, so it is trivially stable by
having nothing to destabilize.

`LCO_ROWS_UNTOUCHED` is `True` for all three scenarios (quoted verbatim from each
scenario's `LCO_ROWS_UNTOUCHED <scenario>=True` line, comparing the 11 real events'
`modified` timestamps before the scenario and after its idempotent second pass) -- none
of the three candidate queue-run projections touch the 11 pre-existing LCO-keyed events
at all, unlike D-11's adopt-scenario prototype for classical-style per-night write
strategy. This is a structural difference from D-11's adopt/gap-fill comparison, not an
oversight: every candidate measured here mints under a `RUN:`-namespaced key entirely
separate from the LCO events' own `url`s, so the two-writer-churn question D-11
identified for classical per-night adoption does not arise for any of these three queue
candidates.

#### What a calendar reader actually sees

Measured via `tmp/26_calendar_render_probe.py`, a `django.test.Client()` fetch of
`/calendar/?year=2026&month=7` against each scenario copy after its scenario ran
(`SERVER_NAME` set to an explicitly `ALLOWED_HOSTS`-permitted value for this environment,
following the 26-01-PLAN.md precedent), HTTP 200 in all three cases.

`span`'s single whole-window event (`RUN:1`) renders across **13** day cells after the
window narrowing (one box per day the narrowed 2026-07-08..2026-07-20 span covers,
counted by the exact per-event `hx-get="{% url 'calendar:update-event' event.id %}"`
(`calendar.html:225`) URL token rather than by title text, since several distinct real
rows share identical truncated title text). `per-night`'s four minted events render
**1** cell each (**4** total) -- same-day events, not all-day spans, since each night's
event has identical start/end dates. `none` renders **0** cells, having minted nothing.
As corroboration against a genuinely real row: the real classical NTT/EFOSC2 event
(pk=44, a genuine cross-midnight sunset-to-sunrise event, `start_time` date
`2026-07-09` differing from `end_time` date `2026-07-10`) renders in exactly **2** day
cells in every scenario copy, tagged **Confirmed against real rows** -- the direct,
measured instance of `solsys_code/views.py:126-132`'s containment filter (`all_day_events`
is every event whose `[start_date, end_date]` range contains the cell's date and whose
start and end dates differ) placing a spanning event into every day cell its range
covers.

**What this means for the options, stated plainly:** a reader looking at the rendered
grid cannot tell "one whole-window span row, rendered once per covered day" apart from
"N per-night rows, one real row per night" -- both mechanisms place a box in exactly the
same day cells, with the identical is_verified-gated visual treatment
(`calendar.html:218-240`; the `span`/`per-night` tag on the underlying prototype rows
itself is **Constructed-input code-path check**, the proposal under test, not an existing
row). The actual difference between the options does not live on the calendar grid at
all -- it lives in the **stored rows** (one event vs. N events for the same window), the
**ownership surface** (one companion FK to manage vs. N), and **machine consumers** (any
future code reading `CalendarEvent.objects.filter(url=...)` sees a different row count
depending on which option is chosen, even though a human looking at `/calendar/` sees
the identical picture either way).

#### The key-scheme consequence

From the measurements above, not from argument: a single canonical scheme covering both
run types would have to let each run type take a **different key form** -- a
classically-scheduled run's key is `RUN:{run_pk}:{date}` (date-bearing, one row per
owned night), while a queue-scheduled run's key is the bare `RUN:{run_pk}` (no date
component, one row for the whole window, the `span` candidate above). One ownership
query already covers exactly this bare-key-plus-prefix pair without modification --
`campaign_views.py:797`'s shipped `Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')`
is the direct precedent a `RUN:` analogue would mirror verbatim
(`Q(url=f'RUN:{run.pk}') | Q(url__startswith=f'RUN:{run.pk}:')`). Only the date-bearing
form has a component a stage transition (a window edit, a narrowing) can change -- and
the measured `KEY_SET_STABLE=False`/orphaned-key result above is the direct evidence that
the date-bearing form is the one exposed to that risk, not the bare form. No verdict on
*which* form a queue run should actually use is stated here -- that is plan 26-05 task 1's
decision to lock.

The real `src/fomo_db.sqlite3` size+mtime fingerprint (`946176 1785094461`) was checked
before this plan's first probe ran and again immediately before `tmp/` was deleted at
this plan's close -- identical both times, confirming no command in this plan ever wrote
to the real dev DB. This value is recorded here, inside the committed decision doc,
specifically so the check survives the deletion of `tmp/` (which held the only other copy
of the pre-plan fingerprint, `tmp/26-realdb-fingerprint-before.txt`).

## Recommendation

Four of the five SPIKE-01..04 verdicts below are locked, falsifiable, and grounded
directly in the Findings above. The write-strategy half of SPIKE-03 (adopt-vs-gap-fill)
is the one exception: it is **deliberately left open for Phase 29** per an explicit
human decision at this plan's task-1 checkpoint, not decided here. That deferral is
scoped narrowly to the write strategy alone — the event-key scheme (also part of
SPIKE-03) is fully locked below, not deferred.

### Criterion 1 / SPIKE-01 — the source vocabulary and its constraint interaction

**Locked.** The `source` vocabulary is fixed at **six values**: the five roadmap values
(web submission, classical file, LCO queue, Gemini queue, CSV import) plus `LEGACY` for
the 31 pre-milestone rows. The three adapter values (classical file, LCO queue, Gemini
queue) are declared now but **not yet produced by any code path** — they wait on v2.3's
ADAPT-01..03. Declaring the full set now costs nothing: Django `TextChoices` values are
validation-only, so adding or removing one later is a no-op `AlterField` (D-13).

`LEGACY` is the value the 31 pre-milestone rows get. Nothing in the data can
discriminate provenance for them — `original_obs_date_raw` is set on only 2 of the 31
rows, making it a parse-failure marker rather than an import signature, so any per-row
inference would be guesswork. Blanket `CSV_IMPORT` was considered and rejected as an
unverifiable assertion written into 31 rows, doubly uncertain for pk=1, which is on the
Didymos 2026 campaign rather than 3I/ATLAS (D-12).

`source` and the CANON-02 field (`telescope_class`) stay out of **both** existing
partial unique constraints. This is not asserted — it is the SPIKE-01 Finding's
executable proof: Block (A) of `26_integrity_check.py` printed
`NEWFIELD_IN_CONSTRAINTS: False` against the real constraint definitions, and Blocks (C)
and (D) then fired both `unique_campaign_run_resolved_window` and
`unique_campaign_run_tbd_natural_key` unmodified on two genuine duplicates differing
only by `source` — five PASS, zero FAIL overall. The consequence: **attribution (the new
companion `run` FK), not either constraint, is what connects same-physical-run rows
arriving from different sources.**

The derivation rule is recorded verbatim as a rule downstream code reads, not
re-invents: **`approval_status == APPROVED` and `source != WEB` means *no approval was
required*, as distinct from *a human approved this*.** A fourth `NOT_REQUIRED` approval
value was considered and rejected, because every existing reader of `approval_status` —
the approval-queue filters, the non-staff visibility gate, `CampaignRunTable`,
`ApprovalQueueTable`, and `CampaignRunDecisionView`'s conditional `.update()` — would
have to handle it, for a distinction `source` already carries (D-14).

**Correction for Phase 27's planning:** there are zero `PENDING_REVIEW` rows (D-17), and
`import_campaign_csv.py:194` already writes `ApprovalStatus.APPROVED`. The importer's
real behaviour change under CANON-01 is writing `source`, not changing what it writes
for `approval_status`.

**Consuming phase:** 27 (`CampaignRun` migration and importer changes).

### Criterion 2 / SPIKE-02 — per-adapter identity key to run

**Locked.** Four adapter mappings, carried forward with their confidence distinction
intact rather than flattened:

| Adapter | Lookup key passed to `insert_or_create_calendar_event()` | Confidence |
|---|---|---|
| Classical (`load_telescope_runs.py:207-216`) | `{'telescope': ..., 'instrument': ..., 'start_time': ...}` with a 5-minute tolerance — no `url` at all | Confirmed against real rows (9 real blank-`url` events) |
| LCO (`sync_lco_observation_calendar.py:361`) | `{'url': <LCO portal request url>}` | Confirmed against real rows (11 real LCO-`url` events, pk=1's window) |
| Gemini (`sync_gemini_observation_calendar.py:150`) | `{'url': f'GEM:{prog}/{obsid}'}` | Constructed-input code-path check — zero real `GEM:` rows exist to confirm against |
| Campaign projection (`campaign_views.py:447,485`) | `{'url': f'CAMPAIGN:{run.pk}'}` or `{'url': f'CAMPAIGN:{run.pk}:{night}'}` for ranges | Constructed-input code-path check — zero real `CAMPAIGN:` rows exist to confirm against |

For a reader who wants to take any existing calendar event and say which run it would
belong to: an LCO event's `url` and a Gemini event's `url` both map directly; a
classical event has no string identity at all and must be matched by
telescope/instrument/date; a campaign-projection event's `url` decodes to a `run_pk`
directly by prefix.

**The consequence D-19 forces:** the classical adapter has no string identity key at
all, so RECON-05's ownership scoping cannot lean on `url` for those 9 events — which is
exactly why ownership lives on the companion record instead (criterion 3, below), not on
`url`.

**Consuming phase:** 27 (adapters are not rewired here — that's v2.3/ADAPT-01..03), 29
(reconciler reads these mappings to decide ownership).

### Domain correction — queue windows are not sets of owned nights

**Recorded 2026-07-27, post-execution, from the project owner (a professional
astronomer and the domain authority on this codebase) — this qualifies the entire
Criterion 3 / SPIKE-03 section below and must be read before it, not after.**

**Closed 2026-07-27, by plan 26-05 task 1's human decision, grounded in the `###
SPIKE-03 gap closure — queue-run projection, measured` Finding above (plan 26-04):**
consequence 3 below — whether a queue-scheduled run should be projected onto the
calendar at all, and under what key — is now settled, not an open question. See
`#### Queue-run projection — settled` under Criterion 3 below for the full verdict; the
domain correction itself is not retracted, only its open question closed.

There is a fundamental difference between a **classically scheduled run** — a specific
set of nights at a specific telescope, each with its own known start/stop times — and a
**queue-scheduled run** (ESO, SOAR, Gemini, and particularly queue-scheduled networks
like LCO). A queue run's window is the span of time (up to a full six-month semester)
during which observations *could* take place, not a set of nights the campaign owns. For
a queue run, the absence of an `ObservationRecord` on a given night inside that window is
the **normal, correct state** — not a gap that is "missing" or needs backfilling.

This directly qualifies `CampaignRun` pk=1, the real case the SPIKE-03 prototype below
was built against: it is **FTS/MuSCAT4, an LCO queue run** (its `source` would be `LCO
queue`), not a classical run. Its 2026-07-07..21 window is the span in which LCO's
scheduler *could* place an observation; the 11 real LCO events (D-20) are what it
actually scheduled. Six consequences follow, and are threaded into the relevant
paragraphs below:

1. **The "4 uncovered nights" framing the D-11 prototype measured is wrong for this
   run.** Both the Adopt and Gap-fill scenarios minted the same 4 events
   (`RUN:1:2026-07-08`, `RUN:1:2026-07-13`, `RUN:1:2026-07-15`, `RUN:1:2026-07-21` —
   `created=4` in both scenarios). Minting a calendar entry on those four nights would
   put an event on the calendar for nights when nothing was scheduled and nothing is
   happening — actively misleading to a reader using the calendar to plan observations.
   Adopt and Gap-fill differ only in whether they *also* rewrite the 11 real events; they
   are **identical** in creating these 4 entries, which arguably should not exist at all
   for a queue run.
2. **The prototype was run against the wrong kind of run for the question SPIKE-03 was
   asking.** The per-night `RUN:{run_pk}:{date}` key fits a **classical** run, where the
   run genuinely does own a specific set of nights with definite per-night start/stop
   times — that is the 9 blank-`url` classical events (D-19), not pk=1. This does not
   invalidate the measured numbers below (they are real and reproducible, and the
   site-local-night derivation mechanism they demonstrate is still correct); it
   invalidates the *interpretation* placed on those numbers as "uncovered nights that
   need filling."
3. **The locked SPIKE-03 key-scheme verdict is rescoped, not unlocked, and is now
   settled for both run types.** `RUN:{run_pk}:{date}` with the site-local observing
   night remains the right key **for runs that own specific nights (classical runs)**.
   For a **queue-scheduled run**, the settled verdict (below, `#### Queue-run projection
   — settled`) is a second, coexisting key family: one bare `RUN:{run_pk}` whole-window
   container event (no date component), owned and minted by the reconciler, plus the
   run's real `ObservationRecord`-derived `CalendarEvent`s — already produced today by
   the existing, unchanged LCO/Gemini queue-sync commands — supplying the per-night
   detail as observations are scheduled and observed. The site-local-night derivation
   rule and the D-10 measurement themselves remain valid and locked wherever a per-night
   key is actually used (classical runs); they were never in question and are unaffected
   by this closure.
4. **D-05's 80x5=400 class-wide fan-out figure does not survive; it is replaced by a
   single settled verdict.** `CampaignRun` pk=29 (`LCO 1m`) and pk=30 (`LCO 2m`) are both
   QUEUE run-type (`#### Run-type inventory` Finding above), so both take the settled
   queue-run form: one bare `RUN:{run_pk}` whole-window container event, not 80 per-day
   events and not 400 per-site-per-day events. The **site-fanout half** of D-05 (a single
   class-wide event, not one per candidate site) is independently verified against
   `SITE_TELESCOPE_MAP` and stands unchanged — it was never in question. The **per-day
   half** is the part this domain correction actually reopened, and it is now settled:
   pk=29's real 80-night window produces **1** event under the locked verdict, not 80 and
   not 400.
5. **RECON-07's mechanism is now stated, not left open.** Of the 19 approved,
   site-resolved 3I/ATLAS runs (the RECON-07 Finding above), the measured split
   (`#### Run-type inventory` Finding above) is **8 QUEUE, 11 CLASSICAL, 0 SPACE**. The
   11 CLASSICAL runs become visible via the existing, unchanged mechanism: one
   `RUN:{run_pk}:{date}` event per owned night. The 8 QUEUE runs become visible via the
   newly settled mechanism: one bare `RUN:{run_pk}` whole-window container event, with
   per-night detail supplied by their real `ObservationRecord`-derived events as
   observations are scheduled and observed (see `#### Queue-run projection — settled`
   below). Both the requirement's *intent* (the run is visible at all) and its
   *mechanism* (which key form, per run type) are now settled.
6. **SPIKE-01's vocabulary already supports the fix.** `CampaignRun.source` already
   distinguishes `classical file` / `LCO queue` / `Gemini queue` / etc. (Criterion 1
   above), so the reconciler already has the information it needs to branch its
   event-projection strategy on run type. This part of the spike holds up unchanged.

**Not corrected here — flagged for a separate todo.** `26-CONTEXT.md`'s D-11 framing
(the "4 uncovered nights" language) and RECON-07's wording both currently read as though
every run's window is a set of owned nights needing full per-night coverage. This spike
does not edit those upstream planning docs — a separate todo should be filed to correct
the "uncovered nights"/backfill framing there before Phase 29 is planned.

### Criterion 3 / SPIKE-03 — the canonical event key (locked for classical and queue runs) and the write strategy (deferred)

**Read the domain correction immediately above first — it explains why this section
states two coexisting key forms rather than one.** The event-key scheme is locked for
both classically-scheduled and queue-scheduled runs, following the settled verdict
recorded in `#### Queue-run projection — settled` immediately below. For a
**classically-scheduled run**, the reconciler passes a namespaced
`url` of the form **`RUN:{run_pk}:{date}`** as its `insert_or_create_calendar_event()`
lookup, and `{date}` is **always the site-local observing night**, derived by converting
the event's `start_time` into the site's timezone and taking the local calendar date —
never the naive UTC date of whatever `start_time` the current stage happens to produce.
This is grounded in the SPIKE-03 Finding's measured comparison: of `CampaignRun` pk=1's
11 real LCO events, event pk=54 (`start_time=2026-07-08T14:08:19Z`) has a naive-UTC date
of 2026-07-08 but a site-local night of 2026-07-09 (Sydney, UTC+10, no July DST) — a
real, measured instance of the two derivations disagreeing, not a hypothetical one. The
consequence: stages 3 and 4 change an event's *times* but never its *key*.

#### Queue-run projection — settled

**A queue-scheduled `CampaignRun` gets exactly two coexisting calendar-event key
families, not one.** (1) The reconciler mints and owns a single bare `RUN:{run_pk}`
whole-window container event — no date component — one per queue run, titled from the
run's telescope/instrument and window (matching the human's own example framing, e.g.
"FTS/MuSCAT4, 2026-07-07 to 2026-07-21"). (2) Per-night detail comes from the run's real
`ObservationRecord`s, synced by the existing, unchanged LCO/Gemini queue-sync commands as
their own separate, non-`RUN:`-namespaced `CalendarEvent`s — the reconciler never mints,
and never needs to mint, a per-night `RUN:{run_pk}:{date}` entry for a queue run, because
per-night precision arrives from a real scheduled/observed record instead of a guess
about which nights of the window will be used.

**Verified against live source before writing this verdict, not assumed**
(`solsys_code/management/commands/sync_lco_observation_calendar.py`): `CampaignRun`
pk=1's 11 real LCO `CalendarEvent`s are genuinely `ObservationRecord`-derived — keyed by
`{'url': facility.get_observation_url(record.observation_id)}` (line 186, the LCO portal
request URL, e.g. `https://observe.lco.global/requests/4229878`), with their
`start_time`/`end_time` computed by `_time_window()` (lines 92-121): a banner-stage
record (`scheduled_start is None`) uses `parameters['start']`/`['end']` (lines 107-110);
once LCO's scheduler places it (`scheduled_start`/`scheduled_end` both set, line 112),
the event narrows to the placed block. `_FAILURE_PREFIX_BY_STATUS` (lines 28-33) then
gives a terminal record its `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` title prefix (a clean
title for `COMPLETED`, D-06's research correction). This is exactly the "narrow or refine
as they get scheduled and observed" mechanism the human's decision names — it already
ships today for LCO, and the equivalent Gemini sync command does the same for Gemini ToO
records. Phase 29 does not need to build this half; it needs to leave it alone (the
ownership rule below) and mint only the run-level container alongside it.

**Why this is stable across all four pipeline stages.** A stage transition (allocation ->
scheduled -> observed) can change a `CalendarEvent`'s *times* and *title*, but never its
*key*, for either key family: the bare `RUN:{run_pk}` container has no date component for
any stage to invalidate (measured `KEY_SET_STABLE=True` under a real window-narrowing
edit in the `#### Three-way comparison against pk=1's real window` Finding above — only
the container's own `start_time`/`end_time` changed, `updated=1`); the
`ObservationRecord`-keyed events key on the LCO/Gemini portal identifier, which does not
change as an observation moves from banner to placed to completed — only the fields
`_time_window()`/`_build_event_fields()` compute from it do.

**Ownership rule, unchanged, applying identically to both key families:** no companion
record, or a companion record whose `run` link is unset, means "not mine, never touch."
The reconciler's own bare `RUN:{run_pk}` container gets a companion record with `run` set
at creation; the `ObservationRecord`-derived events keep whatever companion state
attribution (Phase 28) gives them, and until attribution runs, the reconciler must not
touch them regardless of key form.

**Measured evidence this rests on** (`### SPIKE-03 gap closure — queue-run projection,
measured` above, plan 26-04 — cited by heading, not re-derived here): the `span`
candidate — a bare `RUN:1` whole-window event — measured `in-window count=12` (the 1
container plus the 11 real LCO events), `LCO_ROWS_UNTOUCHED=True`, `created=1 updated=0
unchanged=0`, an idempotent re-run (`created=0 updated=0`), and `KEY_SET_STABLE=True`
under window-narrowing. This is exactly the coexistence the locked verdict above
describes, already measured and already passing.

**Rejected options, with their measured in-window counts (to D-11's completeness bar):**
- **`none`** (mint nothing for the queue run): `in-window count=11` (the 11 real LCO
  events only), `LCO_ROWS_UNTOUCHED=True`, trivially `KEY_SET_STABLE=True` (no `RUN:` key
  ever exists). Rejected because it gives a queue run with a genuinely awarded window and
  zero attributed events no calendar presence at all, unlike the container's "this run
  holds this window" assertion.
- **`per-night`** (mint `RUN:1:{date}` for each of the 4 site-local-uncovered nights):
  `in-window count=15`, `created=4 updated=0 unchanged=0`, an idempotent re-run, but
  measured **`KEY_SET_STABLE=False`** — `RUN:1:2026-07-21` was measured orphaned once a
  real window-narrowing edit no longer covered that date. Rejected both because it is
  measurably unstable and because it reintroduces the domain correction's own error:
  asserting an observation happened on a specific night the queue scheduler never used.

**Trigger condition for revisiting:** if a later milestone (v2.3's adapter rewiring,
ADAPT-01..03) changes queue-run adapters so they no longer write per-night
`ObservationRecord`-derived events directly (for example, folding that responsibility
into the reconciler itself, or a new queue-scheduled facility's adapter shipping with no
per-night `CalendarEvent` output at all), Phase 29 or a later phase should revisit
whether the bare `RUN:{run_pk}` container alone remains sufficient, or whether the
reconciler itself must start minting per-night detail for that facility.

**Consuming phase:** 29 (reconciler implements both key families for queue runs); 27/28
(event key and ownership rule for classical runs — closed, above, unaffected by this
verdict).

The identity-versus-ownership split (D-09) is two separate mechanisms: the namespaced
`url` gives an event its identity (what the lookup matches on); the companion `run` FK
gives it ownership. The hard rule: **no companion row, or a companion row whose `run`
link is unset, means "not mine, never touch."** This is already provable against live
data without any prototype write at all — the 9 classical events have no companion row
whatsoever, so they are outside the ownership mechanism entirely, for any reconciler,
present or future.

**Stage-2 fan-out is answered explicitly:** a class-wide run produces a **single
class-wide event per day** (00:00-23:59, labelled with the class, no site), **not** one
event per candidate site. `CampaignRun` pk=29's real 80-night window, multiplied by
`SITE_TELESCOPE_MAP`'s real 5-site `1m0` count, is a computed figure from real field
values (not an executed check) that makes the alternative's cost concrete: naive
per-site fan-out for that one run alone would be 400 events, four-fifths of them
describing observations that will never happen at that site. Stage 3 narrows to the real
site once an `ObservationRecord` appears, which is the pipeline working as designed.
**Per the domain correction above (point 4) and the settled `#### Queue-run projection —
settled` verdict above, this 400-event figure does not survive: pk=29 is QUEUE run-type,
so it takes the settled bare `RUN:{run_pk}` whole-window container form, producing 1
event for its 80-night window — not 80 per-day events and not 400 per-site-per-day
events.**

A space-mission run gets **one spanning event covering the whole window**, not one event
per day (D-07) — the real instance is pk=26 (JUICE, 2025-11-02 through 2025-11-25),
which becomes one 24-day event rather than 24 daily ones. This keeps the calendar
consistent with `campaign_gap.claimed_dates()`'s asset-aware treatment, which already
refuses to claim those dates at all; the uniform "treat every site-less run as daily
00:00-23:59" alternative was rejected for exactly that reason.

The CANON-02 field (`telescope_class`) carries a **three-meaning "why is there no site"
vocabulary** — telescope-class allocation, space mission, and unresolved/failed-to-resolve
— not a telescope-class-only vocabulary, because the live data has five space-mission
rows (pk=8, 12, 13, 21, 26) against two class-wide ones (pk=29, 30). **Recommendation:
keep the field named `telescope_class`** despite the widened meaning — renaming it is a
larger, separate naming discussion that would delay Phase 27 for no functional benefit,
and the widened *values* (not the field name) are what carries the new meaning; Phase 27
should not have to make a naming call mid-implementation, so this recommendation settles
it in advance.

Stage 0, "allocated but unscheduled" (D-08): a run with no window start produces no
calendar event, but the reconciler counts and reports it in its summary the way
`import_campaign_csv` already reports `site_needs_review` — visibly pending rather than
silently skipped. Real rows this covers: pk=4 (ESO VLT FORS2, site-resolved, approved)
and pk=27/28 (JWST, no site). This gives RECON-06's "reported and skipped" a defined
case.

#### D-11 — the adopt-vs-gap-fill write strategy is deliberately left open for Phase 29 (the queue-run projection question is now settled separately, see above)

**This is not an oversight or a soft lean — it is a deliberate deferral made at the
human's explicit direction** at this plan's task-1 checkpoint. The spike's job was to
produce the measurement; the human judged, correctly, that this specific verdict does
not need to be locked now and that nothing downstream is blocked by leaving it open.
Phase 29 makes this call when it writes the real reconciler, using the evidence below.

**Read together with the domain correction above.** The measured numbers below are real
and reproducible, but the 4 "uncovered nights" both scenarios mint are, per the domain
correction, very likely nights that should not carry a calendar event at all for a queue
run like pk=1 — not nights genuinely missing coverage. Phase 29 should treat the table
below as evidence about the *write-conflict* question (adopt vs. gap-fill on the 11 real
LCO events) rather than as settled evidence that minting 4 new events is the right thing
to do for a queue run in the first place; that second question — whether minting
anything per-night for a queue run is right at all — is now settled: see `#### Queue-run
projection — settled` above, which records the locked verdict (a bare whole-window
container plus the run's real `ObservationRecord`-derived events) and states why a
per-night `RUN:{run_pk}:{date}` key was rejected for queue runs specifically.

**Both options are fully measured, and both are viable on the narrower write-conflict
question.** Neither is a lean:

| Scenario | In-window count | Pass-1 tally | Companion `run`-FK count in window | Idempotent re-run |
|---|---|---|---|---|
| Adopt | 15 | `created=4, updated=11, unchanged=0` | 11 | `created=0 updated=0` |
| Gap-fill | 15 | `created=4, updated=0, unchanged=0` | 11 | `created=0 updated=0` |

Both scenarios produce the identical 15-event in-window result, the identical 4 minted
keys (`RUN:1:2026-07-08`, `RUN:1:2026-07-13`, `RUN:1:2026-07-15`, `RUN:1:2026-07-21` —
the site-local-uncovered nights, **now understood per the domain correction as nights
LCO's queue simply did not schedule, not nights lacking coverage**), and the same 11
companion FKs. Both are idempotent on re-run. The two options are **indistinguishable on
the rendered calendar** — they differ only in write surface: which code path is allowed
to write to the 11 pre-existing LCO events.

**The decisive piece of evidence Phase 29 should weigh**, verified directly in the code
during the task-1 checkpoint (not previously recorded in this doc): `sync_lco_observation_
calendar.py:361` calls `insert_or_create_calendar_event({'url': url}, fields)` on every
sync run, and `calendar_utils._update_or_unchanged()` (`calendar_utils.py:297-315`) sets
every key present in `fields` and saves with `update_fields=list(fields.keys()) +
['modified']` whenever any field differs from its stored value. Under **adopt**, this
means the LCO sync command would overwrite the reconciler's stamp on those 11 rows on
its own next run, and the reconciler would then re-stamp them on its next run — a
genuine two-writer churn loop that would report `updated: 11` on *every* reconcile
cycle, not a one-time transitional cost. Under **gap-fill**, this cannot occur: the
reconciler's write surface is limited to the 4 keys it minted itself, so RECON-05 ("the
reconciler never creates, modifies, or deletes a calendar event it does not own") is
satisfied literally rather than by interpreting "has a companion FK" as sufficient
license to overwrite.

**The condition that would settle it:** once v2.3 rewires the adapters so the LCO sync
command no longer writes to these rows (folding that responsibility into the reconciler
itself), the two-writer objection to adopt disappears entirely. That rewiring is the
trigger for revisiting this choice — Phase 29 should record its own decision at that
point rather than defaulting silently to whichever option is easier to implement first.

The **rejected always-mint baseline** — the reconciler minting its own `RUN:1:{date}` key
for all 15 window nights regardless of existing coverage — is not a third option on the
table. It measured 26 total in-window events (11 pre-existing LCO-keyed plus 15 fresh
`RUN:1:`-keyed, including a second, separate event for every one of the 11
already-covered nights): the concrete, counted instance of the visible double-booking
ATTRIB-06 exists to prevent.

**Consuming phase:** 29 (reconciler write strategy — open; whether a queue run should be
projected at all, and under what key, — now settled, see `#### Queue-run projection —
settled` above); 27/28 (event key and ownership rule for classical runs — closed,
above).

### Criterion 4 / SPIKE-04 — migration and attribution strategy

**Locked.** The migration shape is proven, not proposed: one `RenameModel`
(`CalendarEventTelescopeLabel` -> `CalendarEventMeta`) followed by three `AddField`
operations (`CalendarEventMeta.run`, `CampaignRun.source`, `CampaignRun.telescope_class`),
hand-authored rather than autodetected. Non-interactive autodetection cannot tell a
rename from a delete-plus-create, and because the companion record's `event` field is
its actual primary key (a `OneToOneField`), a `DeleteModel`/`CreateModel` pair would drop
and recreate the table, destroying the 11 real companion rows this spike's coexistence
evidence depends on. No data-migration backfill step is needed — `source` backfills via a
single static field default (`default='legacy'`) — and the `run` link is nullable,
blankable, and clears to null on run deletion. The Finding's identical before/after row
counts (31/20/11, byte-identical) are the proof it preserved every real row.

The rename checklist Phase 27 executes, six integration points (not four — the original
research checklist, scoped to non-test application code, missed two):

| # | Integration point | Broke? | How |
|---|---|---|---|
| 1 | `solsys_code/admin.py` (import, `ModelAdmin` subclass, `admin.site.register`) | Yes | Loudly, `ImportError` at Django startup |
| 2 | `solsys_code/management/commands/sync_lco_observation_calendar.py` (import, `.objects.update_or_create(event=event, ...)`) | Yes | Loudly, `ImportError` at command import |
| 3 | `solsys_code/views.py` `.prefetch_related('telescope_label_meta')` | No | Safe by construction (`related_name` unchanged) |
| 4 | `src/templates/tom_calendar/partials/calendar.html` `event.telescope_label_meta.is_verified` | No | Safe by construction (same reason) |
| 5 | `test_admin.py`'s `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` | Yes | Loudly, `NoReverseMatch` — Django derives the admin changelist URL name from the model's lowercased class name |
| 6 | Class-name references inside `test_load_telescope_runs.py`, `test_sync_lco_observation_calendar.py`, `test_calendar_template.py` | Yes | Loudly, `ImportError` at test-module collection |

**Verdict on the pre-spike analytical prediction: confirmed-with-additions.** The core
prediction — that `related_name='telescope_label_meta'` staying unchanged makes the view
prefetch and the calendar template safe by construction, while the two class-name
imports are the only real risk and fail loudly — is confirmed exactly (rows 1-4 above).
The addition: rows 5 and 6, both real, both loud, neither named by the original
four-point checklist because it was scoped to non-test application code. A rename
executed without fixing rows 5-6 would still leave the test suite red even after the two
"real" application consumers are fixed.

The **`related_name='telescope_label_meta'`-stays-unchanged decision** is what makes
rows 3 and 4 safe by construction — renaming it would break both with no static check to
catch it, unlike the class-name rename which the compiler and Django's own startup
checks catch immediately. The renamed class name is `CalendarEventMeta`, chosen for
generality (it absorbs `run`, `is_verified`, and whatever v2.3 adds, without a second
rename) over a link-specific alternative like `CalendarEventRunLink`, which would
misdescribe the 11 existing rows — pure telescope-label metadata, with no run link at
all until attribution sets one.

**Attribution strategy:** ownership lives on the companion record's `run` link; no
automatic merging of suspected duplicates; per-candidate staff confirmation only;
attribution completable before the first full reconcile sweep. Phase 28 builds the
attribution queue; Phase 29's reconciler depends on attribution having already run for
any window it should treat as covered.

**Consuming phase:** 27 (migration and rename), 28 (attribution UI depends on the
companion FK existing), 29 (reconciler ownership check).

### Recorded findings that correct the planning docs

PROJECT.md's Phase 25 paragraph does not reproduce against the live dev DB (D-16): the
maximum `CampaignRun` pk is 31, no run's `telescope_instrument` contains `FT-115`, and
there are 0 `CAMPAIGN:`-namespaced events — the dev DB was re-imported after Phase 25's
UAT. **Phases 27-29 must not trust PROJECT.md's Phase 25 paragraph for any concrete pk or
count.** Correcting PROJECT.md is deliberately a **separate todo**, outside this phase's
investigation-only boundary — PROJECT.md is not edited by this phase.

### Recommended naming posture for calendar_utils.py (folded todo)

This is a recommendation, not code written here. Five cross-module-consumed helpers in
`solsys_code/calendar_utils.py` still carry a leading underscore despite being a de
facto shared API: `_aperture_class_from_telescope_code` (line 84), `_derive_telescope`
(line 106), `_resolve_placement_block` (line 129), `_extract_instrument` (line 229), and
`_coarse_telescope_label` (line 258). Their leading underscore now misrepresents a
module with real cross-module consumers (`load_telescope_runs.py`,
`sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`, and the test
suite). **Recommendation: Phase 27, which will be editing these modules anyway for the
migration and rename work, should drop the leading underscore on these five names as
part of its own work, rather than as a separate cleanup pass.** The todo's second half
also still stands: `calendar_utils.py`-owned tests still live in
`test_sync_lco_observation_calendar.py` and belong in their own module.

Recording this recommendation does **not** close the todo
(`.planning/todos/pending/2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`)
— it stays open until code actually lands.

## Durable summary

See `docs/design/canonical_record_spike.rst` for the durable, redaction-free summary of
these decisions — written for Phases 27-29 to reference without digging into this
findings record. `src/fomo_db.sqlite3` fingerprint at the time this Recommendation was
completed: `946176 1785094461` (unchanged from the D-04 snapshot value recorded at the
top of this document).
