---
phase: 261002-dsa
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py
  - .planning/v2.4-INTENT-REVIEW.md
autonomous: true
requirements: [ALLOC-05]

estimate:
  tokens: 50000
  raw_tokens: 50000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - "Running `python manage.py shell < .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py` exactly as committed (`DRY_RUN = True`) prints the resolved database path, a ten-row table (pk, title, UTC span, the matching `ALLOC:` url or 'stray') and the totals, and writes nothing."
    - "If any pre-flight fact is false, the script exits non-zero, names every failure and deletes nothing. The checks are: a pk is missing; a url is not blank; there is a companion row other than pk 334's one unconfirmed run-68 row; there is any todo or dismissal row; a mapped `ALLOC:` night is missing or off by a second; an orphan carries a field value or description line its `ALLOC:` night lacks; or pk 334 is not titled `tmp`."
    - "With `DRY_RUN = False` the ten events are deleted inside one `transaction.atomic()`. The script then prints `deleted: 10`, `cascaded: 1 CalendarEventMeta (pk 334's companion row)` and `ALLOC nights unchanged: 9/9`. Any other per-model delete count rolls the whole transaction back."
    - "Run a second time after a real run, the script fails pre-flight because the pks are gone, and changes nothing."
    - "The executor ran the script only against scratch copies made with sqlite's online backup from a read-only URI, and the script's own `database:` line proves it. When this plan ends, the live `src/fomo_db.sqlite3` still holds all ten orphans. The live run is the operator's."
    - "Setup step 5 in `.planning/v2.4-INTENT-REVIEW.md` is struck through like steps 1-4, with a Done note naming the script, the scratch-copy evidence, the pk 334 finding and the expected live output. The file is left modified and unstaged."
  artifacts:
    - ".planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py: a one-off repair script. It has plain-literal module constants `DRY_RUN = True`, `ORPHAN_PKS`, `EXPECTED_ALLOC_URLS`, `STRAY_PK`, `STRAY_TITLE`, `STRAY_META_RUN_PK`, `CARRIED_FIELDS` and `EXPECTED_DELETE_COUNTS`; functions `preflight()`, `alloc_snapshot()` and `main()`; and an unconditional `main()` call at the bottom. It imports only `django.conf.settings`, `django.db.transaction`, `tom_calendar.models.CalendarEvent` and `solsys_code.models.CalendarEventMeta`."
    - ".planning/v2.4-INTENT-REVIEW.md: setup step 5 is struck through and has a Done note. Working tree only; never staged."
  key_links:
    - "`settings.DATABASES['default']['NAME']` is `os.getenv('FOMO_DATABASE_PATH') or <BASE_DIR>/fomo_db.sqlite3` (src/fomo/settings.py:134), and `src/fomo/local_settings.py` has no `DATABASES` override. That env var is the only thing that keeps the executor off the live database. The script prints the resolved name first, and every executor gate asserts that it is the scratch copy."
    - "Deleting a `CalendarEvent` cascades through exactly three reverse relations: `todos` (tom_calendar `EventTodo`, CASCADE), `telescope_label_meta` (`CalendarEventMeta` one-to-one, CASCADE) and `attribution_dismissals` (`CalendarEventDismissal`, CASCADE). `apps.py` wires no pre/post_delete receiver for either `CalendarEvent` or `CalendarEventMeta`. So the delete touches 10 events plus pk 334's one companion row and nothing else, and `EXPECTED_DELETE_COUNTS` enforces that inside the transaction."
    - "The orphans are blank-url events, so no writer (observation projector, `RUN:` reconciler, `ALLOC:` projector) owns them, and no cron tick recreates them after deletion. The script deletes with `filter(pk__in=ORPHAN_PKS, url='')`, so it can never reach a `RUN:`/`ALLOC:`/facility-url event (the spike-findings skill rule)."
---

<objective>
Write, validate and commit a reviewable one-off script that retires the ten orphan `tom_calendar.CalendarEvent` rows named by setup step 5 of `.planning/v2.4-INTENT-REVIEW.md` ("Walkthrough setup", the "shambles, itemised" bullets). Nine of them (pks 44-52, the hand-entered 2026-07-22 Didymos NTT/Magellan nights) are superseded to the second by the `ALLOC:76/77/78:*` nights that setup step 4's `load_telescope_runs Didymos_runs --campaign 'Didymos 2026'` created. pk 334 (`tmp`) is a stray. Prove the script on scratch copies, dry and real. Record the outcome as a Done note under step 5. The operator runs it against the live database afterwards; this plan never does.

Purpose: the walkthrough's Q1-Q7 calendar checks should read a calendar where each Didymos non-LCO night appears once, as its allocation night, rather than twice. This closes the ALLOC-05 intent ("never leaves a duplicate or orphaned event on the calendar") for the hand-entered events that the Phase 35 cutover never saw, because it converted only `RUN:{pk}:{date}` keys.

Output: the script (Task 1, tracer: written, linted, dry-run on a scratch copy, committed). The scratch-copy real run, the negative and re-run proofs, and the step 5 Done note (Task 2).

The script is at `.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py`, called "the script" below.

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **pk 334 HAS a companion row. The brief's "no `CalendarEventMeta` on any of the ten" is wrong for it.** A read-only query on 2026-10-02 found exactly one `solsys_code_calendareventmeta` row for event 334: `run_id=68`, `confirmed_by_id` NULL, `observation_record_id` NULL, `observation_group_id` NULL, `is_verified=1`. This is consistent with the intent review's own census ("10 blank-url ... 9 do not [carry a meta]"). Run 68 is `FTN/MuSCAT3`, `source=legacy`, window 2025-07-04, in TargetList 10 `WR06 tmp campaign`, and owns its own `ALLOC:68:2025-07-04` event (pk 357). It is itself a leftover from the Phase 35 WR-06 review. The script therefore asserts that exact row for pk 334, and no companion row for 44-52. It expects the delete to cascade that one row, so it reports `deleted: 10` plus `cascaded: 1 CalendarEventMeta (pk 334's companion row)`. Run 68, TargetList 10 and event 357 are out of scope and must not be touched. The Done note flags them for the operator to decide.
2. **Nothing else hangs off the ten.** Reverse relations on `CalendarEvent` are exactly `todos`, `telescope_label_meta` and `attribution_dismissals`. All of them are CASCADE, and only pk 334's meta row is non-empty. Generic references (`django_admin_log`, `django_comments`, guardian user/group object permissions) hold 0 rows for content type `tom_calendar.calendarevent` and these pks. `solsys_code/apps.py` connects receivers only for `ObservationRecord` (post_save, pre_delete), `ObservationGroup` membership (m2m_changed) and `CampaignRunObservation` (post_save, post_delete), with none for `CalendarEvent` or `CalendarEventMeta`. The script still enumerates `CalendarEvent._meta.related_objects` at run time and fails pre-flight on any unexpected non-empty relation, so a relation added later cannot be cascaded silently.
3. **"Carries nothing the `ALLOC:` nights don't recreate" holds beyond the spans.** For each of 44-52 versus its mapped `ALLOC:` event (pks 409-417): `title`, `telescope`, `instrument` and `target_list_id` (2, `Didymos 2026`) are equal; `user` and `proposal` are blank on both. Every line of the orphan's `description` is a line of the `ALLOC:` description (the NTT `ALLOC:` description adds `[117.2A2N.001]` / `Proposal: 117.2A2N.001`). The `ALLOC:` description is the run's `observation_details` plus an optional appended `Run status:` line (`campaign_reconciler.event_description()`), so a line-containment check is robust to a later `run_status` change. The script checks this, not only the spans.
4. **How to point Django at a copy, and prove it.** Set `FOMO_DATABASE_PATH` to the copy in the same command (src/fomo/settings.py:134; `local_settings.py` has no `DATABASES` override). Make the copy with `sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup '<copy>'"`. That is sqlite's online backup from a read-only URI: it gives a consistent snapshot even mid-tick and cannot write to the live file. Both steps were probed at planning time: the copy resolved, the ten pks were present, and the probe was removed. This is the same env var that the 261001-smo notebook setup cell `9084663a` sets; the notebook copied with `shutil.copy2`, and `.backup` is the safer equivalent for a shell. The live DB is in rollback-journal mode and at migration head (`solsys_code` 0023), so the copy needs no `migrate`. Every executor run of the script is prefixed `FOMO_DATABASE_PATH="<copy>"`, and every gate asserts that the script printed `database: <copy>`. Never `export` the variable, and never run the script without it.
5. **How `manage.py shell` runs a file.** In Django 5.2 it execs stdin with one merged namespace dict (`django/core/management/commands/shell.py:257`), so module-level functions see the script's constants normally. It prints `N objects imported automatically` first, which is harmless. Two consequences follow. (a) `__name__` inside that namespace is the shell command's module name, so the script must call `main()` unconditionally at the bottom; a `__name__ == '__main__'` guard would make it silently do nothing. (b) Always feed it with a file redirect (`<`), never a pipe. The stdin check is a zero-timeout `select()`, and a pipe that is not yet readable drops into an interactive shell.
6. **Real mode without editing the committed file.** Make a scratchpad copy with `sed 's/^DRY_RUN = True$/DRY_RUN = False/'` and check that `diff` shows exactly that one line changed. The committed file stays `DRY_RUN = True`, and a gate checks the committed blob. The script's docstring tells the operator to do the same.
7. **Lint scope.** `.planning` is in `[tool.ruff] exclude` (pyproject.toml), and both ruff-pre-commit hooks run `ruff check --force-exclude` / `ruff format --force-exclude`. So `pre-commit run ruff --all-files` and `ruff-format` never see the script; they pass vacuously and say nothing about it. The venv's `ruff` is 0.2.1, the same version `.pre-commit-config.yaml` pins (D-07). Passed an explicit path without `--force-exclude`, it lints the file under the project config (single quotes, 120 columns, isort, D103), as probed on `.planning/spikes/001-a-trigger-tom-hook/hook_receiver.py`. Task 1 runs that directly. No file inside the pre-commit gates' scope changes, so the repo-wide gates are not part of this plan.
8. **The cron.** `run_unattended` ticks every 15 minutes, from this checkout, against `src/fomo_db.sqlite3`. The last tick seen at planning time ended `2026-10-02T16:45:44Z exit=0 duration=41s`. The executor never writes to that file and never runs the script against it. The only sqlite write in this plan is Task 2's one-row edit of a scratch copy, addressed by its scratch path. For the operator's live run: run between ticks, after a `=== FOMO unattended run END` banner in `/var/log/fomo/unattended.log`. A `database is locked` error inside the atomic block means nothing was written; retry after the next END banner.
9. **Paired docs (CLAUDE.md) were considered and are not triggered.** This is a one-off data repair of ten literal pks on one database. It changes no module's behaviour; none of the mapped modules, notebooks or `docs/runbooks/` pages is touched, and no runbook page documents these rows. So no notebook or runbook is in `files_modified`. Also, per the operator's decision, there is no management command, no test and no model change. The precedent `solsys_code/management/commands/repair_stale_campaign_run_sites.py` re-ran a real resolution path over every matching row; this task does not.
10. **`.planning/v2.4-INTENT-REVIEW.md` belongs to the operator and has uncommitted edits**, including the step 4 text itself. Task 2 edits it with one scoped Edit and never stages it. Running `git add` on it would sweep the operator's edits into a task commit.
11. **Branch and staging.** Run `git branch --show-current` before the commit; it must print `issue37-telescope-runs-calendar`. Stage by explicit path only. The untracked `.gsd/`, `reqgroup_2682493.json`, `src/fomo_db_20260929.sqlite3` and `.planning/agent-history.json` stay out of every commit. Every commit message ends with these two trailer lines:
    `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
    `Claude-Session: https://claude.ai/code/session_01R99ttZ39WA1eKiUZXwuoFy`

Source coverage audit. GOAL (intent review setup step 5: retire the ten once step 4 confirms they are superseded) → Task 1 (script plus the live re-verification) and Task 2 (proof plus note). REQ ALLOC-05 (no duplicate or orphaned event on the calendar) → Tasks 1-2. CONTEXT (the orchestrator's deliverable): script requirements → Task 1. Scratch-copy validation, dry and real, with the output recorded and no live run → Task 2. The Done note, struck and left unstaged → Task 2. The explicit paired-docs statement → finding 9. Mutable-scope re-verification first → Task 1 Step 0. The lint check → finding 7 and Task 1. RESEARCH: none (no research phase). Deviation from the brief: pk 334's companion row (finding 1). It is planned for, not omitted: the script asserts it and cascades it. Nothing is unplanned, and no deferred item is present.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/v2.4-INTENT-REVIEW.md
@CLAUDE.md

<interfaces>
These were confirmed at planning time; there is no need to re-read the source for them.

`tom_calendar.models.CalendarEvent` (site-packages `tom_calendar/models.py`): fields `title`, `description`, `start_time`, `end_time`, `url` (blank default `''`), `target_list` (FK TargetList, SET_NULL), `user`, `proposal`, `telescope`, `instrument`, `created`, `modified` (auto_now). Reverse accessors are `todos` (EventTodo), `telescope_label_meta` (CalendarEventMeta) and `attribution_dismissals` (CalendarEventDismissal). `_meta.label` is `'tom_calendar.CalendarEvent'`.

`solsys_code.models.CalendarEventMeta`: `event` is a OneToOneField to CalendarEvent with `on_delete=CASCADE`, `primary_key=True` and `related_name='telescope_label_meta'`, so `event_id` is the pk. Other fields: `is_verified`; `run` (FK CampaignRun, SET_NULL); `confirmed_by`; `confirmed_at`; `observation_record` (OneToOne, SET_NULL); `observation_group` (FK, SET_NULL); `minted_sub_night_window`. `_meta.label` is `'solsys_code.CalendarEventMeta'`.

The live census of the ten, read with `sqlite3` in `mode=ro` on 2026-10-02 (UTC, stored as `YYYY-MM-DD HH:MM:SS`):

| pk | title | start_time | end_time | superseded by (event pk) |
|---|---|---|---|---|
| 44 | NTT EFOSC2 | 2026-07-09 22:06:36 | 2026-07-10 11:29:48 | ALLOC:76:2026-07-09 (409) |
| 45 | NTT EFOSC2 | 2026-07-10 22:07:04 | 2026-07-11 11:29:36 | ALLOC:76:2026-07-10 (410) |
| 46 | NTT EFOSC2 | 2026-07-11 22:07:33 | 2026-07-12 11:29:22 | ALLOC:76:2026-07-11 (411) |
| 47 | NTT EFOSC2 | 2026-07-12 22:08:02 | 2026-07-13 11:29:07 | ALLOC:76:2026-07-12 (412) |
| 48 | Magellan-Baade IMACS | 2026-07-17 22:10:56 | 2026-07-18 11:26:51 | ALLOC:77:2026-07-17 (413) |
| 49 | Magellan-Baade IMACS | 2026-07-18 22:11:27 | 2026-07-19 11:26:28 | ALLOC:77:2026-07-18 (414) |
| 50 | Magellan-Clay Lightspeed | 2026-07-18 22:11:28 | 2026-07-19 06:26:00 | ALLOC:78:2026-07-18 (415) |
| 51 | Magellan-Clay Lightspeed | 2026-07-19 22:12:00 | 2026-07-20 06:26:00 | ALLOC:78:2026-07-19 (416) |
| 52 | Magellan-Clay Lightspeed | 2026-07-20 22:12:31 | 2026-07-21 06:26:00 | ALLOC:78:2026-07-20 (417) |
| 334 | tmp | 2025-07-04 22:00:00 | 2025-07-05 06:00:00 | stray; one companion row, run 68 |

All ten have `url=''`, and none has a todo or dismissal row. 44-52 have `target_list_id=2` and no companion row. The database holds 294 events, of which exactly these ten have a blank url.
</interfaces>
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Re-verify the ten read-only, write the dry-run-default script, prove it end-to-end on a scratch copy, commit</name>
  <files>.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md lines 58-128 ("The shambles, itemised" through the setup order)
    - src/fomo/settings.py lines 126-136 (the `FOMO_DATABASE_PATH` override)
  </read_first>
  <action>
    **Step 0: re-verify the mutable scope before anything else.** Run `git branch --show-current`; it must print `issue37-telescope-runs-calendar`. Then run the first `<automated>` command below. It reads the live `src/fomo_db.sqlite3` through a `mode=ro` URI only, and asserts every fact in the `<interfaces>` census, including pk 334's single run-68 companion row (planning finding 1). If any assertion fails, STOP and report the difference. Do not adapt the script to the new state; a drifted census is a re-plan.

    **Step 1: write the script** with the Write tool at the path in `<files>`. It contains:
    - **A module docstring.** It covers:
      - Purpose: retire the ten orphan calendar events of intent-review setup step 5, quick task 261002-dsa.
      - Why they can go: 44-52 are superseded to the second by the `ALLOC:76/77/78` nights of step 4, and 334 is a stray whose one companion row (attributed to run 68) cascades.
      - That it is for one database only: the pks are literal.
      - How to run it. Dry run: `python manage.py shell < <this path>`. Real run: write a copy with `DRY_RUN` flipped to `False` via sed and run that copy with a `<` redirect, never a pipe (finding 5).
      - When to run it: between cron ticks, and after taking a fresh `.backup` of the live database as the rollback.
      - The expected live output (the lines below).
    - **Imports.** Only `from django.conf import settings`, `from django.db import transaction`, `from solsys_code.models import CalendarEventMeta` and `from tom_calendar.models import CalendarEvent`, ordered as ruff's isort rule wants. It must not import the ephemeris views or the ephemeris utilities module, and must not call or name URL reversal: rendering a URL loads the URLconf, which imports the views and triggers the SPICE download.
    - **Module constants.** Plain assignments with plain literals and no annotations, because the gate reads them with `ast.literal_eval`:
      - `DRY_RUN = True`.
      - `ORPHAN_PKS = [44, 45, 46, 47, 48, 49, 50, 51, 52, 334]`, a list literal with no `range()`.
      - `EXPECTED_ALLOC_URLS`, a dict literal mapping 44-52 to the nine urls in the census table.
      - `STRAY_PK = 334`, `STRAY_TITLE = 'tmp'`, `STRAY_META_RUN_PK = 68`.
      - `CARRIED_FIELDS = ('title', 'user', 'proposal', 'telescope', 'instrument', 'target_list_id')`.
      - `EXPECTED_DELETE_COUNTS = {'tom_calendar.CalendarEvent': 10, 'solsys_code.CalendarEventMeta': 1}`.
    - **`preflight()`, with a docstring.** It returns a pair: a list of failure strings (empty means every check passed) and a list of table rows. It collects every failure rather than stopping at the first. Checks:
      - (a) Fetch the ten with `in_bulk(ORPHAN_PKS)`. Every pk exists.
      - (b) Every one has `url == ''`.
      - (c) Companion rows: `CalendarEventMeta.objects.filter(event_id__in=ORPHAN_PKS)` is exactly one row, with `event_id == STRAY_PK`, `run_id == STRAY_META_RUN_PK`, and `observation_record_id`, `observation_group_id` and `confirmed_by_id` all None. So 44-52 have none (the brief's `.exists()` false), and 334 has exactly the row seen at planning time.
      - (d) For every relation in `CalendarEvent._meta.related_objects` other than the `telescope_label_meta` accessor, `relation.related_model._default_manager.filter(**{f'{relation.field.name}__in': ORPHAN_PKS}).count()` is 0. Failures name the accessor.
      - (e) For each of 44-52, run `CalendarEvent.objects.filter(url=<mapped url>)`. Exactly one event, whose pk is not in `ORPHAN_PKS`, with identical `start_time` and `end_time`. For every name in `CARRIED_FIELDS`, the orphan's value is empty/None or equal to the `ALLOC:` event's. Every non-blank line of the orphan's `description` is in the `ALLOC:` event's `description.splitlines()` (finding 3).
      - (f) pk 334's title is `STRAY_TITLE`.
      Each table row holds pk, title, the span as `start_time.isoformat() -> end_time.isoformat()`, and either `<ALLOC url> (event <pk>)` or `stray (companion row attributed to run 68 cascades)`.
    - **`alloc_snapshot()`, with a docstring.** It returns a dict from each of the nine `ALLOC:` urls to `(pk, start_time, end_time, title, description, modified)`, for those that exist.
    - **`main()`, with a docstring.** It prints these lines, in this order:
      1. `database: <settings.DATABASES['default']['NAME']>`.
      2. `mode: DRY RUN (nothing will be written)` or `mode: DELETE`.
      3. The `pk | title | UTC span | superseded by` table.
      4. If there are failures: one `PRE-FLIGHT FAILED: <reason>` line per failure, then `raise SystemExit` with a message saying how many problems were found and that nothing was written.
      5. `asserted: 10 (9 superseded by ALLOC nights, 1 stray)`.
      6. `companion rows that will cascade: 1 (pk 334 -> run 68)`.
      7. `blank-url events in this database: <count>`.
      Then it takes `alloc_snapshot()`. If `DRY_RUN`: print `DRY RUN: nothing deleted. Run a copy with DRY_RUN = False to delete.` and return. Otherwise, inside one `with transaction.atomic():`:
      - Re-run `preflight()`, and raise `RuntimeError` on any failure, so nothing is deleted.
      - Run `CalendarEvent.objects.filter(pk__in=ORPHAN_PKS, url='').delete()`.
      - Build `counts` from the per-model dict with zero entries dropped. If it differs from `EXPECTED_DELETE_COUNTS`, raise `RuntimeError` naming it, which rolls back.
      After the block:
      - Print `deleted: 10` and `cascaded: 1 CalendarEventMeta (pk 334's companion row)`.
      - Raise `SystemExit` if any of the ten still exists.
      - Compare a fresh `alloc_snapshot()` with the earlier one: all nine urls present, every value identical including `modified`. Print `ALLOC nights unchanged: 9/9`, or raise `SystemExit` listing the differences.
      - Print the blank-url count again.
    - **The last line** is a bare `main()` call, with no `__name__` guard (finding 5).
    <!-- planner-discipline-allow: PRE-FLIGHT FAILED -->
    <!-- planner-discipline-allow: deleted: -->

    **Step 2: lint (finding 7).** Run the third `<automated>` command, the venv ruff 0.2.1 on the explicit path. `ruff check --fix` and `ruff format` on the script alone are fine if it reports something. Re-run until clean.

    **Step 3: dry run on a scratch copy.** Set `SCRATCH` to your session scratchpad directory, never a path inside the repo, in the same Bash call as each command that uses it. Run the fourth `<automated>` command. It makes `$SCRATCH/261002-dsa/dry/fomo_db.sqlite3` with `.backup` from the read-only URI and runs the committed-form script against it via `FOMO_DATABASE_PATH`. It then asserts the copy path is not the live file, that the script printed `database: <copy>`, a clean dry run (all ten rows, every `ALLOC:` url, `asserted: 10`, no deletion line, no pre-flight failure), and that the copy still holds all ten. Keep `$SCRATCH/261002-dsa/dry.out` for the SUMMARY.

    **Step 4: commit.** Run `git branch --show-current` again. Then `git add` the script by explicit path, and commit `chore(261002-dsa): add one-off script retiring the 10 orphan calendar events superseded by the Didymos ALLOC nights` with the two trailer lines from finding 11. Run the last `<automated>` command.
  </action>
  <verify>
    <automated>python -c "
import sqlite3
con = sqlite3.connect('file:src/fomo_db.sqlite3?mode=ro', uri=True)
def q(sql, *args):
    return con.execute(sql, args).fetchall()
census = {
    44: ('NTT EFOSC2', '2026-07-09 22:06:36', '2026-07-10 11:29:48', 'ALLOC:76:2026-07-09'),
    45: ('NTT EFOSC2', '2026-07-10 22:07:04', '2026-07-11 11:29:36', 'ALLOC:76:2026-07-10'),
    46: ('NTT EFOSC2', '2026-07-11 22:07:33', '2026-07-12 11:29:22', 'ALLOC:76:2026-07-11'),
    47: ('NTT EFOSC2', '2026-07-12 22:08:02', '2026-07-13 11:29:07', 'ALLOC:76:2026-07-12'),
    48: ('Magellan-Baade IMACS', '2026-07-17 22:10:56', '2026-07-18 11:26:51', 'ALLOC:77:2026-07-17'),
    49: ('Magellan-Baade IMACS', '2026-07-18 22:11:27', '2026-07-19 11:26:28', 'ALLOC:77:2026-07-18'),
    50: ('Magellan-Clay Lightspeed', '2026-07-18 22:11:28', '2026-07-19 06:26:00', 'ALLOC:78:2026-07-18'),
    51: ('Magellan-Clay Lightspeed', '2026-07-19 22:12:00', '2026-07-20 06:26:00', 'ALLOC:78:2026-07-19'),
    52: ('Magellan-Clay Lightspeed', '2026-07-20 22:12:31', '2026-07-21 06:26:00', 'ALLOC:78:2026-07-20'),
    334: ('tmp', '2025-07-04 22:00:00', '2025-07-05 06:00:00', None),
}
for pk, (title, start, end, alloc) in census.items():
    row = q('select title, start_time, end_time, url from tom_calendar_calendarevent where id = ?', pk)
    assert row == [(title, start, end, '')], (pk, row)
    side = (q('select count(*) from tom_calendar_eventtodo where event_id = ?', pk)[0][0], q('select count(*) from solsys_code_calendareventdismissal where event_id = ?', pk)[0][0])
    assert side == (0, 0), (pk, 'todos/dismissals', side)
    meta = q('select run_id, observation_record_id, observation_group_id, confirmed_by_id from solsys_code_calendareventmeta where event_id = ?', pk)
    assert meta == ([(68, None, None, None)] if pk == 334 else []), (pk, meta)
    if alloc:
        match = q('select start_time, end_time from tom_calendar_calendarevent where url = ?', alloc)
        assert match == [(start, end)], (pk, alloc, match)
print('OK: census holds -- 10 blank-url orphans; 44-52 companion-free and matched to their ALLOC night to the second; 334 has one unconfirmed run-68 companion row; no todos or dismissals')
"</automated>
    <automated>python -c "
import ast
p = '.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py'
t = ast.parse(open(p).read())
mods = set()
for n in ast.walk(t):
    if isinstance(n, ast.ImportFrom):
        mods.add(n.module)
    elif isinstance(n, ast.Import):
        mods.update(a.name for a in n.names)
assert mods <= {'django.conf', 'django.db', 'tom_calendar.models', 'solsys_code.models'}, mods
assert not any((isinstance(n, ast.Name) and n.id == 'reverse') or (isinstance(n, ast.Attribute) and n.attr == 'reverse') for n in ast.walk(t)), 'URL reversal is forbidden'
top = {n.targets[0].id: n.value for n in t.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
assert ast.literal_eval(top['DRY_RUN']) is True, 'committed default must be DRY_RUN = True'
assert ast.literal_eval(top['ORPHAN_PKS']) == [44, 45, 46, 47, 48, 49, 50, 51, 52, 334]
assert ast.literal_eval(top['EXPECTED_ALLOC_URLS']) == {44: 'ALLOC:76:2026-07-09', 45: 'ALLOC:76:2026-07-10', 46: 'ALLOC:76:2026-07-11', 47: 'ALLOC:76:2026-07-12', 48: 'ALLOC:77:2026-07-17', 49: 'ALLOC:77:2026-07-18', 50: 'ALLOC:78:2026-07-18', 51: 'ALLOC:78:2026-07-19', 52: 'ALLOC:78:2026-07-20'}
assert ast.literal_eval(top['EXPECTED_DELETE_COUNTS']) == {'tom_calendar.CalendarEvent': 10, 'solsys_code.CalendarEventMeta': 1}
assert (ast.literal_eval(top['STRAY_PK']), ast.literal_eval(top['STRAY_TITLE']), ast.literal_eval(top['STRAY_META_RUN_PK'])) == (334, 'tmp', 68)
funcs = {n.name for n in t.body if isinstance(n, ast.FunctionDef)}
assert {'preflight', 'alloc_snapshot', 'main'} <= funcs, funcs
last = t.body[-1]
assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and getattr(last.value.func, 'id', None) == 'main', 'last statement must be a bare main() call'
assert not any(isinstance(n, ast.If) and 'name' in ast.dump(n.test) and '__main__' in ast.dump(n.test) for n in t.body), 'no __name__ guard: manage.py shell would skip main()'
print('OK: imports, constants, functions and the bare main() call are as specified')
"</automated>
    <automated>test "$(ruff --version)" = 'ruff 0.2.1' && ruff check .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py && ruff format --check .planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py</automated>
    <automated>SCRATCH="${SCRATCH:?set SCRATCH to your session scratchpad}"; D=.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-; W="$SCRATCH/261002-dsa"; C="$W/dry/fomo_db.sqlite3"; mkdir -p "$W/dry" && sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup '$C'" && FOMO_DATABASE_PATH="$C" python manage.py shell < "$D/retire_orphan_events.py" > "$W/dry.out" 2>&1; RC=$? C="$C" W="$W" python -c "
import os, sqlite3
C, W = os.environ['C'], os.environ['W']
assert os.path.realpath(C) != os.path.realpath('src/fomo_db.sqlite3'), 'the copy path is the live database'
out = open(W + '/dry.out').read()
assert os.environ['RC'] == '0', 'dry run exited ' + os.environ['RC'] + ':' + out[-3000:]
assert 'database: ' + C in out, 'script did not resolve to the scratch copy'
for token in ('mode: DRY RUN', 'asserted: 10', 'DRY RUN: nothing deleted', 'stray', 'ALLOC:76:2026-07-09', 'ALLOC:76:2026-07-12', 'ALLOC:77:2026-07-17', 'ALLOC:77:2026-07-18', 'ALLOC:78:2026-07-18', 'ALLOC:78:2026-07-20', 'companion rows that will cascade: 1'):
    assert token in out, token
assert 'PRE-FLIGHT FAILED' not in out and 'deleted: ' not in out, out[-3000:]
con = sqlite3.connect('file:' + C + '?mode=ro', uri=True)
assert con.execute('select count(*) from tom_calendar_calendarevent where id in (44,45,46,47,48,49,50,51,52,334)').fetchone()[0] == 10
print('OK: dry run on scratch copy', C, '-- clean pre-flight, nothing written')
"</automated>
    <automated>P=.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/retire_orphan_events.py; S="$(git log -1 --format=%s)" && [[ "$S" == 'chore(261002-dsa): '* ]] && F="$(git show --name-only --format= HEAD)" && test "$F" = "$P" && git diff --quiet HEAD -- "$P" && grep -qx 'DRY_RUN = True' "$P" && M="$(git log -1 --format=%B)" && [[ "$M" == *'Claude-Session: https://claude.ai/code/session_01R99ttZ39WA1eKiUZXwuoFy'* ]] && echo 'OK: one-file chore commit; the committed script equals the working copy, which the AST gate proved has DRY_RUN = True'</automated>
  </verify>
  <done>
    The live census was re-verified read-only and matches the plan. The script exists with the specified constants, functions and imports, and lints clean under ruff 0.2.1. A dry run on a `.backup` scratch copy printed `database: <copy>`, the ten-row table and `asserted: 10`, and wrote nothing. One `chore(261002-dsa)` commit touches only the script, with `DRY_RUN = True` in the committed blob.
  </done>
</task>

<task type="auto">
  <name>Task 2: Prove the real run, the refusal and the re-run on scratch copies; record the output; strike setup step 5 with a Done note (working tree only)</name>
  <files>.planning/v2.4-INTENT-REVIEW.md</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md lines 88-128 (setup order; the house style of struck steps 2-4)
  </read_first>
  <action>
    Set `SCRATCH` to your session scratchpad in each Bash call that uses it, as in Task 1. Never run the script without `FOMO_DATABASE_PATH` pointing at a scratch copy.

    **Step 1: refusal proof (first `<automated>`).**
    - Write the real-mode copy `$SCRATCH/261002-dsa/retire_orphan_events_real.py` with `sed 's/^DRY_RUN = True$/DRY_RUN = False/'`, and check `diff` shows exactly one changed line (finding 6).
    - Make copy A with `.backup`. Then shift pk 44's `end_time` by one second ON COPY A ONLY, using its explicit scratch path. This is the plan's only sqlite write.
    - Run the real-mode script against copy A. It must exit non-zero, print `PRE-FLIGHT FAILED` naming pk 44, print no deletion line, and leave all ten rows and pk 334's companion row in copy A. This proves "fail loudly, write nothing".

    **Step 2: real run and re-run (second `<automated>`).**
    - Make copy B with `.backup` and record its event count.
    - Run the real-mode script against copy B. Then run it a second time against copy B.
    - The gate asserts:
      - the first run exited 0 and printed `database: <copy B>`, `mode: DELETE`, `asserted: 10`, `deleted: 10`, `cascaded: 1 CalendarEventMeta` and `ALLOC nights unchanged: 9/9`;
      - copy B lost exactly 10 events, none of the ten pks remains, and pk 334's companion row is gone;
      - all nine `ALLOC:` urls are still present with the census spans, and run 68 and event 357 are still present;
      - the second run exited non-zero with a pre-flight failure and no deletion line;
      - the LIVE database, read through a `mode=ro` URI, still holds all ten orphans.
    - Copy `$SCRATCH/261002-dsa/real.out` verbatim into the SUMMARY (the deliverable's "real-run output"), together with the dry-run output from Task 1 and one line each for the refusal and the re-run. Then delete `$SCRATCH/261002-dsa`.

    **Step 3: the Done note.** Use ONE scoped Edit on `.planning/v2.4-INTENT-REVIEW.md`, never Write. In the setup order, wrap step 5's existing two lines in `~~ ~~`, exactly as steps 2-4 are struck, and keep their wording. After the closing `~~`, add a bold lead-in `**Done 2026-10-02 (quick task 261002-dsa) -- script validated on scratch copies; the live run is the operator's.**` (use an em dash in place of `--`, matching the file). Then add, as continuation lines indented three spaces like step 4's:
    - **The script.** Its path, and one sentence on what pre-flight asserts. It covers: all ten exist with blank urls; 44-52 have no companion row and match their `ALLOC:76/77/78` night to the second, with nothing in title, telescope, instrument, target list or description that the night lacks; 334 is `tmp`; no todos or dismissals. Pre-flight writes nothing if any check fails.
    - **The pk 334 finding** (planning finding 1). 334 is not companion-free. One unconfirmed `CalendarEventMeta` row attributes it to run 68 (`FTN/MuSCAT3` in `WR06 tmp campaign`, a Phase 35 WR-06 leftover with its own `ALLOC:68:2025-07-04`, event 357), and the delete cascades it. Run 68, its TargetList and event 357 were left alone; whether to retire them too is the operator's call.
    - **Scratch evidence.** Quote the real run's `deleted:`, `cascaded:` and `ALLOC nights unchanged:` lines from Step 2. Say that a run against a deliberately broken copy was refused with nothing written, and that a second run was refused because the pks were gone.
    - **The operator's live run.**
      1. Wait for a `=== FOMO unattended run END` banner in `/var/log/fomo/unattended.log`.
      2. Take a fresh `sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup 'src/fomo_db_20261002.sqlite3'"` as the rollback. The 2026-09-29 snapshot predates steps 3-4.
      3. Do a dry run with `python manage.py shell < <script path>` and read the table.
      4. Run the `DRY_RUN = False` copy, made with the sed line in the script's docstring, with a `<` redirect.
      Expected: `asserted: 10`, `deleted: 10`, `cascaded: 1 CalendarEventMeta (pk 334's companion row)`, `ALLOC nights unchanged: 9/9`, and the blank-url count going from 10 to 0. A `database is locked` error means nothing was written; retry after the next END banner.
    Do not tick any checkbox. Do NOT stage or commit this file (finding 10). Run the third `<automated>` command.

    **The SUMMARY also states:** no notebook or runbook was changed, and why (finding 9); the intent-review note is left uncommitted beside the operator's edits; and the live run belongs to the operator.
  </action>
  <verify>
    <automated>SCRATCH="${SCRATCH:?set SCRATCH to your session scratchpad}"; D=.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-; W="$SCRATCH/261002-dsa"; R="$W/retire_orphan_events_real.py"; A="$W/neg/fomo_db.sqlite3"; mkdir -p "$W/neg" && sed 's/^DRY_RUN = True$/DRY_RUN = False/' "$D/retire_orphan_events.py" > "$R" && grep -qx 'DRY_RUN = False' "$R" && ! grep -qx 'DRY_RUN = True' "$R" && diff <(grep -vx 'DRY_RUN = True' "$D/retire_orphan_events.py") <(grep -vx 'DRY_RUN = False' "$R") && sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup '$A'" && sqlite3 "$A" "update tom_calendar_calendarevent set end_time = '2026-07-10 11:29:49' where id = 44" && FOMO_DATABASE_PATH="$A" python manage.py shell < "$R" > "$W/neg.out" 2>&1; RC=$? A="$A" W="$W" python -c "
import os, sqlite3
A, W = os.environ['A'], os.environ['W']
assert os.path.realpath(A) != os.path.realpath('src/fomo_db.sqlite3'), 'copy A is the live database'
out = open(W + '/neg.out').read()
assert 'database: ' + A in out, 'script did not resolve to copy A:' + out[-3000:]
assert os.environ['RC'] != '0', 'a failed pre-flight must exit non-zero'
assert 'PRE-FLIGHT FAILED' in out and '44' in out and 'deleted: ' not in out, out[-3000:]
con = sqlite3.connect('file:' + A + '?mode=ro', uri=True)
assert con.execute('select count(*) from tom_calendar_calendarevent where id in (44,45,46,47,48,49,50,51,52,334)').fetchone()[0] == 10
assert con.execute('select count(*) from solsys_code_calendareventmeta where event_id = 334').fetchone()[0] == 1
print('OK: a broken pre-flight on copy A exits non-zero, names pk 44, and writes nothing')
"</automated>
    <automated>SCRATCH="${SCRATCH:?set SCRATCH to your session scratchpad}"; W="$SCRATCH/261002-dsa"; R="$W/retire_orphan_events_real.py"; B="$W/real/fomo_db.sqlite3"; mkdir -p "$W/real" && sqlite3 'file:src/fomo_db.sqlite3?mode=ro' ".backup '$B'" && sqlite3 "file:$B?mode=ro" 'select count(*) from tom_calendar_calendarevent' > "$W/real.before" && FOMO_DATABASE_PATH="$B" python manage.py shell < "$R" > "$W/real.out" 2>&1; RC=$?; FOMO_DATABASE_PATH="$B" python manage.py shell < "$R" > "$W/rerun.out" 2>&1; RC2=$? RC="$RC" B="$B" W="$W" python -c "
import os, sqlite3
B, W = os.environ['B'], os.environ['W']
assert os.path.realpath(B) != os.path.realpath('src/fomo_db.sqlite3'), 'copy B is the live database'
out = open(W + '/real.out').read()
assert os.environ['RC'] == '0', 'real run exited ' + os.environ['RC'] + ':' + out[-3000:]
for token in ('database: ' + B, 'mode: DELETE', 'asserted: 10', 'deleted: 10', 'cascaded: 1 CalendarEventMeta', 'ALLOC nights unchanged: 9/9'):
    assert token in out, token
rerun = open(W + '/rerun.out').read()
assert os.environ['RC2'] != '0' and 'PRE-FLIGHT FAILED' in rerun and 'deleted: ' not in rerun, rerun[-3000:]
con = sqlite3.connect('file:' + B + '?mode=ro', uri=True)
def one(sql, *a):
    return con.execute(sql, a).fetchone()[0]
before = int(open(W + '/real.before').read().strip())
assert before - one('select count(*) from tom_calendar_calendarevent') == 10
assert one('select count(*) from tom_calendar_calendarevent where id in (44,45,46,47,48,49,50,51,52,334)') == 0
assert one('select count(*) from solsys_code_calendareventmeta where event_id = 334') == 0
alloc = {'ALLOC:76:2026-07-09': ('2026-07-09 22:06:36', '2026-07-10 11:29:48'), 'ALLOC:76:2026-07-10': ('2026-07-10 22:07:04', '2026-07-11 11:29:36'), 'ALLOC:76:2026-07-11': ('2026-07-11 22:07:33', '2026-07-12 11:29:22'), 'ALLOC:76:2026-07-12': ('2026-07-12 22:08:02', '2026-07-13 11:29:07'), 'ALLOC:77:2026-07-17': ('2026-07-17 22:10:56', '2026-07-18 11:26:51'), 'ALLOC:77:2026-07-18': ('2026-07-18 22:11:27', '2026-07-19 11:26:28'), 'ALLOC:78:2026-07-18': ('2026-07-18 22:11:28', '2026-07-19 06:26:00'), 'ALLOC:78:2026-07-19': ('2026-07-19 22:12:00', '2026-07-20 06:26:00'), 'ALLOC:78:2026-07-20': ('2026-07-20 22:12:31', '2026-07-21 06:26:00')}
for url, span in alloc.items():
    assert con.execute('select start_time, end_time from tom_calendar_calendarevent where url = ?', (url,)).fetchall() == [span], url
assert one('select count(*) from solsys_code_campaignrun where id = 68') == 1 and one('select count(*) from tom_calendar_calendarevent where id = 357') == 1
live = sqlite3.connect('file:src/fomo_db.sqlite3?mode=ro', uri=True)
assert live.execute('select count(*) from tom_calendar_calendarevent where id in (44,45,46,47,48,49,50,51,52,334)').fetchone()[0] == 10, 'the LIVE database lost orphans'
print('OK: real run on copy B deleted exactly the ten (+1 cascaded companion row), the 9 ALLOC nights and run 68 are intact, the re-run was refused, and the live database still holds all ten')
"</automated>
    <automated>python -c "
t = open('.planning/v2.4-INTENT-REVIEW.md', encoding='utf-8').read()
s5 = t.index('5. ~~**Retire the 10 orphans (pks 44–52, 334)**')
s6 = t.index('6. Then Q1–Q7', s5)
note = t[s5:s6]
assert 'not by hand.~~' in note, 'step 5 must be struck through like steps 2-4'
for token in ('261002-dsa', 'retire_orphan_events.py', 'asserted: 10', 'deleted: 10', 'ALLOC nights unchanged: 9/9', 'run 68', 'WR06 tmp campaign', 'operator', '.backup'):
    assert token in note, token
print('OK: step 5 struck with its Done note')
" && git diff --cached --quiet -- .planning/v2.4-INTENT-REVIEW.md && echo 'OK: intent review not staged' && git status --porcelain .planning/v2.4-INTENT-REVIEW.md</automated>
    <human-check>Operator, after reading the script: take a fresh `.backup` of `src/fomo_db.sqlite3`, wait for an END banner in `/var/log/fomo/unattended.log`, run it dry, then run the `DRY_RUN = False` copy as described in the step 5 note. Confirm the output reads `asserted: 10`, `deleted: 10`, `cascaded: 1 CalendarEventMeta (pk 334's companion row)` and `ALLOC nights unchanged: 9/9`, and that the calendar at `:8000` shows each Didymos NTT/Magellan night once. Decide separately whether run 68 / `WR06 tmp campaign` / event 357 should also go.</human-check>
  </verify>
  <done>
    On scratch copies only: the deliberately broken copy was refused with nothing written. The real run deleted exactly the ten events plus pk 334's one companion row, and left the nine `ALLOC:` nights, run 68 and event 357 unchanged. A re-run was refused. The live database still holds all ten. The SUMMARY quotes the real-run output verbatim. Setup step 5 is struck, with its Done note, and the file shows as modified but unstaged.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| repair script -> live `src/fomo_db.sqlite3` | An operator-run script deletes rows from the database that the public calendar and the 15-minute cron both read |
| executor -> Django `DATABASES` | Only an env var separates a validation run from the live database |
| cron tick <-> repair transaction | Two writers on one SQLite file |
| task commit -> operator's working tree | `.planning/v2.4-INTENT-REVIEW.md` carries uncommitted operator edits |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-dsa-01 | Tampering | the script deleting a row that is not one of the ten orphans (pk drift, reuse, or an orphan that gained a url or companion row) | high | mitigate | Pre-flight asserts per-pk title, span, blank url, companion rows, every reverse relation and the `ALLOC:` match, and runs again inside the transaction. The delete filter is `pk__in=ORPHAN_PKS, url=''`. Per-model counts must equal `EXPECTED_DELETE_COUNTS`, or a `RuntimeError` rolls back. Task 2's broken-copy run proves the refusal path writes nothing. |
| T-dsa-02 | Tampering | the executor writing to the live database while "validating" | high | mitigate | Every run is prefixed `FOMO_DATABASE_PATH="<copy>"`, never exported. Copies are made by `.backup` from a `mode=ro` URI. Every gate asserts the script printed `database: <copy>` and that the copy path is not the live file. The final gate reads the live DB read-only and requires all ten orphans still present. |
| T-dsa-03 | Denial of service | the operator's live run colliding with a cron tick (lock, or a torn read) | medium | mitigate | One short `transaction.atomic()`. A `database is locked` error rolls back entirely. The note and docstring tell the operator to run after an END banner and to retry after the next. Copies use the online backup API, which is consistent mid-tick. |
| T-dsa-04 | Repudiation | losing pk 334's run-68 attribution row without a trace | low | accept | The row is unconfirmed, has no observation link, and points at a review-leftover run. The script prints the cascade explicitly. The Done note records it with run 68's identity. The operator takes a fresh `.backup` as the rollback before the live run. |
| T-dsa-05 | Denial of service | the script importing the ephemeris views or utilities, which triggers a ~1.6 GB SPICE download on the production host | low | mitigate | The AST gate allows imports from only four modules and forbids any `reverse` name. Planning-time probe: `manage.py shell` on a copy started without the download. |
| T-dsa-06 | Tampering | the task commit sweeping the operator's uncommitted intent-review edits into history | low | mitigate | One scoped Edit, never staged. The gate requires the file absent from `git diff --cached`. The only commit stages the script by explicit path, and the commit gate checks it is the sole file. |
| T-dsa-SC | Tampering | package installs | low | accept | No npm/pip/cargo install in this plan. |
</threat_model>

<verification>
- The live census re-check, run read-only via `mode=ro`, passes before the script is written (Task 1, first gate).
- The script's AST gate passes: four allowed import modules, the literal constants, `preflight`/`alloc_snapshot`/`main`, a bare trailing `main()` call, and no `__name__` guard. `ruff check` and `ruff format --check` under the venv's ruff 0.2.1 are clean on the explicit path, since `.planning/` is excluded from the pre-commit hooks.
- Scratch-copy evidence: a clean dry run; a broken-copy refusal with nothing written; a real run giving `deleted: 10` / `cascaded: 1 CalendarEventMeta` / `ALLOC nights unchanged: 9/9` with exactly 10 events gone; a refused re-run; the live database still holding all ten.
- `git log` shows one `chore(261002-dsa)` commit touching only the script, committed with `DRY_RUN = True`. `.planning/v2.4-INTENT-REVIEW.md` is modified and unstaged. The pre-existing untracked files are still uncommitted.
</verification>

<success_criteria>
- The operator has a reviewable, committed script that, run once against the live database, removes exactly the ten orphans: the nine Didymos hand-entered nights and the `tmp` stray, plus pk 334's one companion row. It touches nothing else, and refuses loudly if the database no longer looks as it did at planning time.
- The script's behaviour (dry, real, refusal, re-run) is proven on scratch copies, with the real output in the SUMMARY. The live database is untouched by this plan.
- Setup step 5 reads as done in the intent review, with everything the operator needs for the live run and the pk 334 / run 68 finding surfaced for their decision.
</success_criteria>

<output>
Create `.planning/quick/261002-dsa-retire-the-10-orphan-calendarevents-pks-/261002-dsa-SUMMARY.md` when done. Include:
- the census re-check result, and the commit SHA;
- the dry-run output (Task 1) and the real-run output (Task 2) verbatim from the scratch copies, plus one line each for the broken-copy refusal and the refused re-run;
- the pk 334 finding (one unconfirmed companion row attributed to run 68, cascaded), and that run 68, TargetList 10 and event 357 were left for the operator;
- that no notebook or runbook changed, and why (planning finding 9); that ruff ran directly because `.planning/` is excluded from the pre-commit hooks (finding 7);
- that the step 5 Done note is in `.planning/v2.4-INTENT-REVIEW.md` uncommitted, and that the live run and its confirmation are the operator's.
Whoever commits the quick-task docs (PLAN/SUMMARY) must stage those two files by explicit path, and never `.planning/v2.4-INTENT-REVIEW.md`.
</output>
