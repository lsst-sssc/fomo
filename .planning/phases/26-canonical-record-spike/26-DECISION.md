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

## Recommendation

<!-- completed in plan 26-03 -->

## Durable summary

<!-- completed in plan 26-03 -->
