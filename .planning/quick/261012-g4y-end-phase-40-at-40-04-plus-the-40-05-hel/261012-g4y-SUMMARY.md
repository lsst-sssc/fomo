---
phase: 261012-g4y
plan: 01
subsystem: notebook-isolation
tags: [phase-40, rollback, notebook-guard, notebook-helper, pre-executed-notebooks]
status: complete
requirements: [WARN-02, WARN-03, WARN-05]
commits: 4
plan_head_before: c0b1e24f7176b096621075a60845e21b04b558c7
plan_head_after: ef0c58ac66e2e8c7306efe51a72d9535fbd73e41
actuals:
  tokens: 151600    # chars/4 over `git diff c0b1e24 HEAD` (606442 chars, mostly re-executed notebook JSON)
  tasks: 3
  commits: 4
key-files:
  modified:
    - solsys_code/tests/test_pre_executed_notebooks.py
    - pyproject.toml
    - docs/notebooks/fomo_nb_utils.py
    - solsys_code/tests/test_notebook_helper.py
    - docs/notebooks/README.md
    - docs/notebooks/pre_executed/*.ipynb (all eight)
decisions:
  - "c45edbc (40-08) and 7af64d0 (40-11) dropped by developer decision 2026-10-12, knowing the round-2 WR-02 finding"
  - "Notebook source taken from 17168ff; only the eight teardown sentences about the second printed line changed"
duration: about 20 minutes of execution (eight notebook runs took under 10 s each)
completed: 2026-10-12
---

# Quick task 261012-g4y: end Phase 40 at 40-04 plus the 40-05 notebook-helper fix

The round-8 gap plans are reverted, the notebook guard and `pyproject.toml` are back at 8ab5247, the notebook helper
and its tests are back at 819321e, the README is 819321e plus two round-2 IN-01 sentence corrections, and all eight
demo notebooks were re-executed against the 819321e helper. The full suite is green and the override draft is written.

## Commits

| Hash | Subject |
|------|---------|
| c8a8b56 | `Revert "docs(40): create gap-closure plans 40-19..40-21 (round 8)"` (only commit made with `SKIP=django-test`) |
| cc3a60e | revert(quick-261012-g4y): notebook guard and pyproject.toml back to 40-04 (8ab5247), notebook helper and its tests back to 40-05 (819321e) |
| a5ee3d2 | docs(quick-261012-g4y): telescope_runs_demo re-executed on the 40-05 notebook helper; teardown prose matches its output |
| ef0c58a | docs(quick-261012-g4y): seven demo notebooks re-executed on the 40-05 notebook helper; teardown prose matches its output |

`git rev-list --count c0b1e24..HEAD` is 4. Nothing was pushed. The developer's uncommitted files were never staged.

## Restore table

| Path | Source | Check made |
|------|--------|------------|
| solsys_code/tests/test_pre_executed_notebooks.py | 8ab5247 | `git diff --quiet 8ab5247 -- path` clean |
| pyproject.toml | 8ab5247 | clean; the only difference was the `ipykernel>=7` dev-extra line (ccfc578) |
| docs/notebooks/fomo_nb_utils.py | 819321e | clean |
| solsys_code/tests/test_notebook_helper.py | 819321e | clean |
| docs/notebooks/README.md | 819321e plus two sentences | whitespace-normalized text equals 819321e with exactly OLD-A -> NEW-A and OLD-B -> NEW-B |

ipykernel grep (outside `.planning` and notebooks): only `docs/requirements.txt:2` (docs build, there before Phase 40)
and the comment at `docs/notebooks/fomo_nb_utils.py:289`. No Python file imports it, no workflow under `.github`
names it, and no runbook under `docs/runbooks` names the helper or the guard (so no runbook change was needed).

## Record of decisions (developer, 2026-10-12)

**c45edbc dropped, with the planner's WR-02 finding confirmed.** Round-2 review WR-02 (`40-REVIEW.md` at commit
269d8c5) is the bug c45edbc fixed. In 819321e, `_disable_django_databases()` (fomo_nb_utils.py:184) calls
`connections.close_all()` (:199) before it switches the aliases to the dummy backend. If the close raises, the switch
never runs, and Django's exception replaces the helper's own `RuntimeError`. It is reachable by accident:

- (a) A settings module whose `DATABASES` has no `default` alias (the reviewer's repro). `check_database_settings`
  refuses the extra alias, then `close_all()` raises `ImproperlyConfigured("You must define a 'default' database.")`.
  I confirmed in an inert Python session (Django 5.2.17, `settings.configure(DATABASES={'primary': ...})`, no database
  file opened) that both `close_all()` and `connections['primary']` raise that error.
- (b) `DJANGO_ALLOW_ASYNC_UNSAFE` exported as an empty string. The helper's `setdefault` keeps it and
  `django.utils.asyncio.async_unsafe` treats it as unset, so under Jupyter both `migrate` and the close raise
  `SynchronousOnlyOperation` (read from the Django source; not run).

What 819321e does on those paths: the `finally` still removes the scratch directory and restores the environment,
but the dummy-backend switch is skipped and Django's error propagates instead of the helper's. No database is
reachable all the same: in (a) Django refuses every connection while `default` is missing, and in (b) every query
under Jupyter is refused and the only configured file is the removed scratch database. The 819321e helper docstring
(setup_scratch_django's Raises section, fomo_nb_utils.py:265) and the README sentence "A refusal or failure switches
every database alias to Django's dummy backend" (README.md:69) therefore do not hold on those paths. The developer
chose to drop c45edbc knowing this. Neither text was edited.

**7af64d0 dropped with it.** It changed only c45edbc's helper docstring (and guard text that was also rolled back).

**README = 819321e plus c45edbc's two round-2 IN-01 sentence corrections.** Both corrected claims were checked against
the 819321e helper before they were applied:

- NEW-A: after the refusal, or any later failure once the helper has started reading Django's settings, no cell can
  reach a database until restart. Supported by `setup_scratch_django`'s `except BaseException` (fomo_nb_utils.py:306),
  which runs `_disable_django_databases()` for every failure from the environment assignment (:287) onward, including
  the settings check (:295), `django.setup()` and `migrate`; `_disable_django_databases` configures a dummy-only
  settings object when the settings were never read, and later re-runs are refused by :275 because Django is
  configured.
- NEW-A, second claim: a failure before the settings are read (for example snapshot mode with no developer database)
  routes nothing and the setup cell can be run again. Supported by `prepare_scratch_database` (:210-241), which sits
  outside the `try` (:285), validates the mode and the developer database before it creates anything, removes its
  own directory if the snapshot fails, and leaves the environment untouched. The up-front refusal (:275) is also
  before any change.
- NEW-B: re-running the setup cell after the helper has read Django's settings raises `RuntimeError`. Supported by
  :275 (`apps.ready or django_settings.configured`); reading `settings.DATABASES` (:295) configures the lazy settings,
  so a refusal or later failure leaves `configured` true.

I judged each claim by the paths reachable in ordinary use and by accident. The WR-02 paths above are the exception
and are recorded in the previous section. None of c45edbc's other README text (the WR-02 dummy-backend sentence, the
teardown second-line sentence, the guard paragraph) was brought back.

## Notebook source and prose

Notebook source is 17168ff (equal to c0b1e24 for all eight notebooks). The 8ab5247 notebook prose describes the
pre-40-05 order (`django.setup()` first, check afterwards), so the later prose (40-09, 40-11) is kept. An inert
planning-time run of the 8ab5247 guard over all eight notebooks reported no problems, and the suite confirms it: the
restored guard (`test_every_real_pre_executed_notebook_is_isolated`) is green on the re-executed notebooks.

**Setup-cell statements checked against 819321e, per notebook.** In all eight, the setup markdown cell makes the
same three failure-handling statements (the pre-`django.setup()` refusal, "after that refusal, or any later failure
once the helper has started reading Django's settings, no cell can reach a database until the kernel is restarted",
and the re-run `RuntimeError`) plus the scratch-directory, `mode=ro` online-backup and "helper never opens the
developer database in fresh mode" statements. All are true of 819321e (see the NEW-A and NEW-B checks above and
`snapshot_database`, :90). No change was made to any setup cell or other markdown cell.

**Teardown sentence (the only prose change, all eight notebooks).** Old sentence (written for c45edbc, false of
819321e because 819321e prints a fixed line):

> Its second printed line says what the helper checked before `django.setup()` ran: that Django's only configured database was the scratch file in that directory.

New sentence (TEAR_SENT, checked against `teardown_scratch_django` :322-364 and `check_database_settings` :137 and
accurate as planned):

> Its second printed line, that the developer database was never opened for writing, is fixed text rather than a measurement: it rests on the check the helper made before `django.setup()` ran, that Django's only configured database was the scratch file in that directory.

Teardown cells replaced: backfill_lco_observations_demo 46602df0; campaign_lifecycle_demo 234805b6;
import_campaign_csv_demo 9c899021; load_telescope_runs_demo 31ff3edb; project_observation_calendar_demo 317a98bb;
reconcile_campaign_runs_demo 98bba968; sync_gemini_observation_calendar_demo c8f6eeb3; telescope_runs_demo 2e2d138e.
Nothing else in those cells changed (the two snapshot notebooks keep their third sentence). Notebook edits were made
with nbformat from a scratch script under `$HOME/tmp`, by cell id; the diff of each is the one line.

## Re-execution

The committed output came from the c45edbc helper (40-09), so all eight were re-executed against 819321e with the
README's own command (`env TMPDIR=/tmp jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.timeout=1200`, from `docs/notebooks/pre_executed/`), per the CLAUDE.md paired-docs rule. Every
run exited 0 (logs `$HOME/tmp/261012-g4y-*-nbconvert.log`). Per-notebook check on all eight: counts 1..N, no error
output, `Django ready: settings module='src.fomo.settings'`, a `/tmp/fomo-notebook-db-*` Resolved line with the right
mode description, `Removed scratch database directory:` for that directory plus the 819321e second line, directory
gone, `nbsphinx` execute still `never`, no `/home/` in setup or teardown output.

| Notebook | Mode | Resolved directory | Outputs that changed beyond the teardown line |
|----------|------|--------------------|-----------------------------------------------|
| backfill_lco_observations_demo | fresh | /tmp/fomo-notebook-db-bj1udbit | last_run_at timestamps; one stdout/stderr line order swap |
| campaign_lifecycle_demo | fresh | /tmp/fomo-notebook-db-b1czd7d9 | confirmed_at timestamp; stdout line order swap |
| import_campaign_csv_demo | fresh | /tmp/fomo-notebook-db-7pha2wis | none |
| load_telescope_runs_demo | fresh | /tmp/fomo-notebook-db-nkqqkf8j | scratch temp file name |
| sync_gemini_observation_calendar_demo | fresh | /tmp/fomo-notebook-db-zttb072k | time-relative window values |
| project_observation_calendar_demo | snapshot | /tmp/fomo-notebook-db-t6jagsmq | random demo ids |
| reconcile_campaign_runs_demo | snapshot | /tmp/fomo-notebook-db-rrr3g12u | random demo ids; two line order swaps |
| telescope_runs_demo | fresh | /tmp/fomo-notebook-db-j133v2jl | none (dip 1.4376 unchanged) |

Snapshot counts that moved: none. The record pks and run pks in the snapshot notebooks (for example run pk=87, record
pk=524, night pk=713) are the same as at c0b1e24, so the developer database did not change between regenerations and
no quoted count or outcome in the snapshot prose is contradicted; no snapshot prose was changed.

campaign_lifecycle_demo 40-04 cells: `Undo redirected to` occurs once, as at c0b1e24; the High-band marker and
`marker`/`band` counts in the output are unchanged (1, 2, 5).

HOME-PATH lines (not helper output; the only one in all eight notebooks):

- `HOME-PATH (not helper output): campaign_lifecycle_demo.ipynb d687bfab Using fallback library next to module: /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/spiceypy/utils/libcspice.so` (the spiceypy log line already present at c0b1e24 and
  expected from 40-07's inventory).

## Tests and gates

- Targeted run (guard, helper, projector evidence, import_campaign_csv) before the telescope_runs_demo commit: `Ran 147 tests`, `OK`
  (`$HOME/tmp/261012-g4y-t1.log`).
- Full suite: `python manage.py test solsys_code --exclude-tag=ephemeris_segfault --parallel auto --noinput`:
  `Ran 2338 tests in 21.272s`, `OK` (`$HOME/tmp/261012-g4y-full-suite.log`). The three full-hook commits also passed
  the django-test hook.
- `pre-commit run ruff` and `ruff-format` on the guard, helper tests, helper and all eight notebooks: Passed, nothing
  rewritten.

Known difference left as is: `docs/requirements.txt` still lists ipykernel for the docs build.

## Deviations from Plan

None - plan executed as written. The plan's verify blocks ran from scripts saved under `$HOME/tmp` rather than heredocs,
with the same content; the Task 1 per-notebook check and the Task 2 all-eight check were combined into one script.

## DEV-8 confirmations

40-VERIFICATION.md, 40-REVIEW.md and 40-REVIEW-DISPOSITION.md are unchanged since c0b1e24. ROADMAP.md, STATE.md and the whole phase-40
directory equal 5b5912d at HEAD (the revert is the only change). Phase 40 is not marked complete. Nothing was pushed.
The developer's uncommitted files (config.json, state.json, ui-reviews/.gitignore, agent-history.json, milestone.lock,
39.1 .gitkeep, reqgroup_2682493.json) were not staged. This SUMMARY, the draft and the PLAN are uncommitted for the
orchestrator.

## Known Stubs

None.

## The override draft

Path: `.planning/quick/261012-g4y-end-phase-40-at-40-04-plus-the-40-05-hel/261012-g4y-OVERRIDE-DRAFT.md` (95 lines,
uncommitted). Full text:

---

# Override draft for /gsd-verify-work 40 (quick task 261012-g4y)

Draft only. It rests on the 8ab5247 guard and the 819321e helper as restored on 2026-10-12, and it replaces the
override recorded on 2026-10-11 (accepted_at 2026-10-11T13:37:50Z), which described the rolled-back R12.

## Proposed override 1 -- WARN-02 guard completeness

must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"

reason: >-
  The guard reads each notebook's JSON (its literal code and its committed output) and never runs it. In ordinary
  code it catches a notebook that skips the helper or calls it twice (R2), imports Django before the helper call
  (R3), sets Django up itself with `django.setup()`, `from django import setup` or `import django as x` (R4), names
  `FOMO_DATABASE_PATH` or `fomo_db.sqlite3` in a string or passes a `FOMO_DATABASE_PATH=` keyword (R5/R7), or calls
  `shutil.copy`, `copy2` or `copyfile`, also under an import alias (R6). R0, R5, R6 and R7 apply to every notebook;
  the other rules apply only to a notebook that imports Django, the project app or a TOM plugin. It also checks the
  committed output: counts 1..N and no error (R8), a Resolved line inside a `fomo-notebook-db-*` directory (R9), the
  matching teardown at the end (R10), and no `assert` around a write (R11).

  Deliberate evasion is out of scope (developer decision 2026-10-12). The guard does not see code inside a string
  (`exec`, `eval`, a `timeit` string, including the setup call's own argument), IPython shell lines, magics and
  aliases, monkey-patching (for example replacing `tempfile.mkdtemp`), or names, paths and values built at run time.
  It also misses a helper call made through its module attribute, re-pointing Django's settings or connections
  after the helper call without writing `FOMO_DATABASE_PATH` or `fomo_db.sqlite3`, and a byte copy by anything other
  than shutil's `copy`, `copy2` and `copyfile` when the file name is not written out.

  Another process is not seen either (a `!` shell line, `%run`, `subprocess`, `os.system`). Unlike the rest,
  ordinary code can do this: a `!python manage.py ...` cell before the helper call or after the teardown runs with
  `FOMO_DATABASE_PATH` unset, so `settings.py` falls back to `src/fomo_db.sqlite3`.
  [Developer: accept or ask for a fix]

  The run-time backstop is the 819321e helper. It refuses to start if Django is already set up or configured in the
  kernel. Before `django.setup()` it raises `RuntimeError` unless `default` is the only alias, `DATABASE_ROUTERS` is
  empty and `default` resolves to the scratch file it created. After a later failure it closes the connections,
  switches every alias to the dummy backend, removes the scratch directory, restores the environment and re-raises.
  If closing the connections itself fails, the switch is skipped and Django's own error replaces the helper's. A
  `DATABASES` setting with no `default` alias does this (round-2 WR-02, left unfixed by the developer's decision); no
  database is reachable then, because Django refuses every connection without a `default` alias. The teardown's
  second line is fixed text, not a measurement (round-2 WR-01). The helper checks once, at setup, so code that runs
  before the call, after the teardown or in another process is outside its checks.

accepted_by: [developer to fill in]
accepted_at: [developer to fill in]

## Proposed override 2 -- guard wording

must_have: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-18 must_have 6 (every form the texts claim is reported is reported): the guard docstring and README state what R0-R12 check"

reason: >-
  The texts those must-haves judged (the 40-08..40-18 guard docstrings, the README paragraph and the R12 they
  describe) were rolled back with the guard. The texts in force are the 8ab5247 module docstring and the README
  (819321e plus the two IN-01 sentence corrections). Read against the code, they claim more than is checked in four
  places: (1) the docstring says a comment that merely mentions a name cannot satisfy or trip a check, but a cell
  that does not parse is searched as text, comments included (R0; this only ever reports more; round-7 WR-03);
  (2) the README says `/tmp/fomo-notebook-db-*`, but the helper uses `tempfile`, which follows `TMPDIR` (the README's
  own command sets `TMPDIR=/tmp`; round-7 WR-04); (3) the README and docstring say a notebook never sets Django up,
  routes the database itself or byte-copies, which means the forms in override 1 and no others; (4) the README and
  `setup_scratch_django`'s docstring say a refusal or failure switches every alias to the dummy backend, which does
  not hold when closing the connections fails (round-2 WR-02, accepted by dropping c45edbc).

accepted_by: [developer to fill in]
accepted_at: [developer to fill in]

## What each claim rests on

| Claim | Rests on |
|-------|----------|
| Skipped or doubled helper call | `isolation_problems` R2, test_pre_executed_notebooks.py:363-393 |
| Django imported before the call | R3, :394-399 |
| Own `django.setup()` | `_django_setup_problems` (R4), :264-278 |
| Routing by string or keyword | `_database_routing_problems` (R5/R7), :281-295 |
| Byte copy | `_byte_copy_problems` (R6), :298-316 |
| Applies to every notebook / Django-backed only | `isolation_problems` :344-349, `is_django_backed` :190 |
| Output checks | R8 :357-361, R9 :401-412, R10 :414-433, R11 :435 |
| Comments searched as text | R0, :334-342 (`UNPARSEABLE_SUSPECT_TEXT` :83) |
| Refuses a running Django | `setup_scratch_django`, fomo_nb_utils.py:275 |
| Settings checked before `django.setup()` | :295, `check_database_settings` :137, `check_resolved_database` :115 |
| Switch to dummy backend after failure | `except BaseException` :306, `_disable_django_databases` :184 |
| Close can fail first (WR-02) | `connections.close_all()` :199, before the alias switch |
| Teardown line is fixed text | `teardown_scratch_django`, fomo_nb_utils.py:364 |
| Another process falls back to the developer file | `src/fomo/settings.py:100` |
| Wording gaps | guard docstring :19; README.md:59, :69, :90; fomo_nb_utils.py:265 |

## Must-haves the rollback withdraws

The guard, pyproject and helper work of these plans is undone: 40-06 (R12 first form), 40-08 (R12 whole-capability
bans and c45edbc's helper change), 40-10 (R12 at every identifier, R0 allow-list), 40-11 (7af64d0 docstrings),
40-13 (R0 automagic, R6 copy names, ccfc578 ipykernel dev extra), 40-14, 40-15 (R6 and R12 additions), 40-16, 40-17
(R12 shell forms, R3 and R4 reading strings) and 40-18 (docstring and README wording). Their README and docstring
text goes with them.

What stays: 40-05's helper and its tests; the README's two IN-01 sentences from c45edbc; 40-07's and 40-09's
notebook work; and 40-09/40-11's notebook prose (17168ff), whose eight teardown sentences now match 819321e.

The verifier decides at /gsd-verify-work 40 which of those must-haves count as withdrawn.

---

## Self-Check: PASSED

Commits c8a8b56, cc3a60e, a5ee3d2 and ef0c58a are ancestors of HEAD; the restored files, eight notebooks and the draft
exist; the Task 1, Task 2 and Task 3 verify scripts printed their OK lines.
