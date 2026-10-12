---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-12T20:40:40Z
status: passed
score: 88/88 must-haves verified (4 by override; 129 withdrawn by the 261012-g4y rollback; prohibitions 23/23 in scope)
covered_files:
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-05-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-05-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-06-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-06-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-07-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-07-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-08-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-08-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-09-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-09-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-10-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-10-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-11-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-11-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-12-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-12-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-13-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-13-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-14-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-14-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-15-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-15-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-16-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-16-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-17-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-17-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-18-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-18-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-UI-SPEC.md"
  - "CLAUDE.md"
  - "docs/notebooks/README.md"
  - "docs/notebooks/fomo_nb_utils.py"
  - "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/telescope_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "pyproject.toml"
  - "solsys_code/campaign_tables.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/tests/test_attribution_template.py"
  - "solsys_code/tests/test_bootstrap5_rendering.py"
  - "solsys_code/tests/test_campaign_attribution_views.py"
  - "solsys_code/tests/test_notebook_helper.py"
  - "solsys_code/tests/test_pre_executed_notebooks.py"
  - "solsys_code/tests/test_projector_demo_notebook.py"
  - "src/fomo/settings.py"
  - "src/templates/campaigns/approval_queue.html"
  - "src/templates/campaigns/attribution_queue.html"
  - "src/templates/campaigns/campaign_list.html"
  - "src/templates/campaigns/campaignrun_gap_analysis.html"
  - "src/templates/campaigns/campaignrun_submit_form.html"
  - "src/templates/campaigns/campaignrun_table.html"
  - "src/templates/campaigns/partials/attribution_return_state.html"
  - "src/templates/campaigns/submission_thanks.html"
  - "src/templates/solsys_code/partials/campaign_links.html"
covered_digest: "v3:sha256:d97cd134dd35068a286ec4a71dcc991764073f09821a9402e802632cd3c9ebb1"
behavior_unverified: 0
overrides_applied: 4
overrides:
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: >-
      The guard reads each notebook's JSON (its literal code and its committed output) and never runs it. In ordinary
      code it catches a notebook that skips the helper or calls it twice (R2), imports Django before the helper call
      (R3), sets Django up itself with `django.setup()`, `from django import setup` or `import django as x` (R4), names
      `FOMO_DATABASE_PATH` or `fomo_db.sqlite3` in a string or passes a `FOMO_DATABASE_PATH=` keyword (R5/R7), or calls
      `shutil.copy`, `copy2` or `copyfile`, also under an import alias made in the same cell (R6). R0, R5, R6 and R7 apply to every notebook;
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
      ordinary code can do this: a `!python manage.py ...` cell before the helper call runs with
      `FOMO_DATABASE_PATH` unset, so `settings.py` falls back to `src/fomo_db.sqlite3`.
      Accepted as residual by the developer (2026-10-12): an unlikely corner case in niche demo notebooks, no fix planned.

      The run-time backstop is the 819321e helper. It refuses to start if Django is already set up or configured in the
      kernel. Before `django.setup()` it raises `RuntimeError` unless `default` is the only alias, `DATABASE_ROUTERS` is
      empty and `default` resolves to the scratch file it created. After a later failure it closes the connections,
      switches every alias to the dummy backend, removes the scratch directory, restores the environment and re-raises.
      If closing the connections itself fails, the switch is skipped and Django's own error replaces the helper's. A
      `DATABASES` setting with no `default` alias does this (round-2 WR-02, left unfixed by the developer's decision); no
      database is reachable then, because Django refuses every connection without a `default` alias. The teardown's
      second line is fixed text, not a measurement (round-2 WR-01). The helper checks once, at setup, so code that runs
      before the call, after the teardown or in another process is outside its checks.
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-12T20:54:36Z"
  - must_have: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-18 must_have 6 (every form the texts claim is reported is reported): the guard docstring and README state what R0-R12 check"
    reason: >-
      The texts those must-haves judged (the 40-08..40-18 guard docstrings, the README paragraph and the R12 they
      describe) were rolled back with the guard. The texts in force are the 8ab5247 module docstring and the README
      (819321e plus the two IN-01 sentence corrections). Read against the code, they claim more than is checked in five
      places: (1) the docstring says a comment that merely mentions a name cannot satisfy or trip a check, but a cell
      that does not parse is searched as text, comments included (R0; this only ever reports more; round-7 WR-03);
      (2) the README says `/tmp/fomo-notebook-db-*`, but the helper uses `tempfile`, which follows `TMPDIR` (the README's
      own command sets `TMPDIR=/tmp`; round-7 WR-04); (3) the README and docstring say a notebook never sets Django up,
      routes the database itself or byte-copies, which means the forms in override 1 and no others; (4) the README and
      `setup_scratch_django`'s docstring say a refusal or failure switches every alias to the dummy backend, which does
      not hold when closing the connections fails (round-2 WR-02, accepted by dropping c45edbc); (5) the README's "it
      calls the helper first" and the docstring's "before any Django import" hold only for literal import statements
      (R3).
    accepted_by: "Tim Lister"
    accepted_at: "2026-10-12T20:54:36Z"
re_verification:
  previous_status: gaps_found
  previous_score: 210/214 (prohibitions 67/67)
  gaps_closed:
    - "ROADMAP SC 2 (first half) / WARN-02: closed by developer-accepted override 1 (2026-10-12T20:19:00Z), which replaces the 2026-10-11 override; the round-7 CR-01 route (string code in the setup call's own argument) is named in its residual, and the verifier's reproduction still gives [] as the override says"
    - "40-08 must_have 8.14, 40-11 must_have 11.1, 40-18 must_have 6: closed by developer-accepted override 2 (2026-10-12T20:28:10Z); each of its four wording claims checked true against the current tree"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "A Django import done through importlib.import_module('django') or __import__('django') before the helper call is not reported (R3 reads only import statements; R4 only the dotted name django.setup)"
    category: security
    reason: "Deliberate evasion, which override 1 puts out of scope ('Deliberate evasion is out of scope'); no ordinary demo notebook imports Django that way. Recorded so the next guard review sees it"
    evidence_status: "reproduced: isolation_problems() == [] on a deep copy of import_campaign_csv_demo with importlib.import_module('django').setup() before the setup cell; not executed"
  - finding: "R5/R7 read str constants only; b'src/fomo_db.sqlite3' gives []"
    category: security
    reason: "Override 1 says 'in a string', and a bytes literal is not a str, so the claim holds; a bytes path to the developer database is deliberate evasion. Round-4 WR-02 fixed this and was rolled back by 261012-g4y"
    evidence_status: "reproduced on a deep copy; not executed"
  - finding: "Round-7 IN-03 (carried): patching tempfile.mkdtemp before the helper call makes the developer database the 'scratch' file"
    category: security
    reason: "Named in override 1's residual ('monkey-patching (for example replacing tempfile.mkdtemp)'); optional helper hardening is a later-triage candidate"
    evidence_status: "unchanged; helper is 819321e"
  - finding: "Uncommitted developer edit in project_observation_calendar_demo.ipynb markdown cell 4862a7df ('never sidereal sources (CLAUDE.md)' -> 'almost never sidereal sources')"
    category: other
    reason: "Present in the working tree at the start of this verification, not part of any Phase 40 plan or of 261012-g4y; markdown only, so no guard rule or test reads it. It is inside the fingerprinted file, so committing or reverting it changes covered_digest. The new wording also reads against CLAUDE.md's 'FOMO is exclusively for Solar System / non-sidereal targets'"
    evidence_status: "git diff HEAD shows the one-line change; the guard and projector tests are green with it"
  - finding: "Round-1 IN-03 (carried): the runbook's attribution-pass 'each table's page and sort' may read as covering the worklists' event_page/record_page"
    category: other
    reason: "Documentation wording; UAT test 4 passed the runbook section on 2026-10-12. Not in Phase 41's roadmap text; would reach Phase 41 only as a todo"
    evidence_status: "unchanged since the previous report"
human_verification:
  - test: "Re-read override 1's R6 sentence and override 2's 'four places' against the verifier's findings in the 'Override claim check' table, then either confirm both overrides stand as written or amend their wording"
    expected: "Override 1 says R6 catches shutil.copy, copy2 or copyfile 'also under an import alias'. The guard follows an alias only inside the cell that imports it (shutil_names is rebuilt per cell, test_pre_executed_notebooks.py:304-307): `import shutil as sh` plus `from fomo_nb_utils import developer_database_path` in the setup cell and `sh.copy(developer_database_path(repo_root_path), scratch_db_dir / 'copy.sqlite3')` in a later cell gives [] on a copy of import_campaign_csv_demo (same cell, or plain `shutil.copy` in another cell: reported). That byte-copies the developer database, which SC 1 forbids, though it does not route Django to it. Separately, override 2 counts four overclaiming places; the README's 'it calls the helper first' (line 89-90) and the docstring's 'before any Django import' (line 12-13) hold only for literal import statements, a fifth place of the same kind as (3). In the other direction, override 1 says a `!python manage.py` cell after the teardown is not seen, but R10 fails any Django-backed notebook with a cell after the teardown (reproduced). No code change is needed to keep the overrides; a wording change (for example 'under an import alias made in the same cell') needs the developer"
    why_human: "The overrides are the developer's accepted wording; the verifier may not edit or widen them, and the UAT acceptance (test 5) relied on the draft describing the guard 'as it is'"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that
catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3, closing the
remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-12T20:40:40Z (HEAD d6fa287)
**Status:** human_needed
**Re-verification:** Yes, after the developer's rollback (quick task 261012-g4y) that ends Phase 40 at 40-04 plus the
40-05 helper fix

## Summary

The phase goal is met on the current tree, with the guard's known limits accepted by the developer in two overrides.

- **Notebooks.** All eight notebooks use the 819321e helper, were re-executed on it (counts 1..N, no error output, a
  `/tmp/fomo-notebook-db-*` Resolved line, the teardown at the end), and the 8ab5247 guard reports no problem in any of
  them. The developer database's modification time is 2026-10-10 00:16, before the 2026-10-12 re-execution, so the two
  snapshot runs did not write it.
- **Guard and `python -O`.** The guard is back at 8ab5247 and the helper and its tests at 819321e (both `git diff`
  clean). The helper raises a real `RuntimeError` under `-O` and `PYTHONOPTIMIZE=1`. SC 2's first half passes only
  through override 1, as the developer decided.
- **Attribution page.** No attribution file, template, runbook or CLAUDE.md line has changed since 8ab5247 (40-04's
  last commit). The render tests and the three Chromium tests for the marker, the Bootstrap 5 sections and Undo staying
  on the same page all pass.

Both overrides match their must-haves exactly and are applied. Most of their claims are true of the code they cite,
but **one claim is not**: override 1 says R6 catches `shutil.copy`, `copy2` or `copyfile` "also under an import alias".
The guard only follows an alias inside the cell that imports it. With a cross-cell alias, a notebook can byte-copy the
developer database through `developer_database_path()` and the guard gives []. That route does not send Django to the
developer database, so it does not undo override 1's must-have (SC 2, first half). Only the developer can amend the
wording of an accepted override, so the phase is `human_needed` with one item. Two smaller wording points go with it.

Test runs:
- Verifier: guard, helper, projector, attribution template and views modules, plus both template-scan classes: 181
  tests, OK.
- Verifier: the three attribution Chromium tests (High-band bar, sections open and close, Undo on page 2): 3 tests, OK.
- Orchestrator regression gate on the current tree: 2293 + 40 tests, OK.

How the verifier stayed safe: it only called the guard's own functions, loaded from a scratchpad copy, on deep copies of
real notebooks. It ran no notebook, never set `FOMO_DATABASE_PATH` and never opened `src/fomo_db.sqlite3`.

## How the rollback was applied to the must-haves

Quick task 261012-g4y (developer decision 2026-10-12) ended Phase 40 at 40-04 plus the 40-05 helper fix. The verifier
confirmed the restore table:

| Path | Matches | Check |
|------|---------|-------|
| solsys_code/tests/test_pre_executed_notebooks.py | 8ab5247 | `git diff --quiet` clean |
| pyproject.toml | 8ab5247 | clean |
| docs/notebooks/fomo_nb_utils.py | 819321e (40-05 plan_head_after) | clean |
| solsys_code/tests/test_notebook_helper.py | 819321e | clean |
| docs/notebooks/README.md | 819321e plus S1+S2 and S3 (40-08's IN-01 sentences) | `git diff 819321e` shows exactly those two hunks |
| eight notebooks | 17168ff source; only the eight teardown markdown cells changed | cell-by-cell source comparison (one more cell, 4862a7df, is the developer's uncommitted edit, see advisory) |

Must-haves whose guard, pyproject, helper, docstring or README work was rolled back are **withdrawn**, as the developer
decided (261012-g4y OVERRIDE-DRAFT "Must-haves the rollback withdraws"). They are not counted as failed or as verified.
Must-haves whose work stayed are verified against the current tree. Where kept work was later reworded on purpose (the
README's IN-01 sentences, the teardown sentence), the item is marked verified with the supersession noted.

| Plan | Truths | Verified | By override | Withdrawn | Withdrawn because |
|------|--------|----------|-------------|-----------|-------------------|
| 40-01 | 10 | 10 | | | |
| 40-02 | 14 | 14 | | | |
| 40-03 | 8 | 8 | | | |
| 40-04 | 14 | 14 | | | |
| 40-05 | 13 | 13 | | | |
| 40-06 | 8 | | | 8 | R12 and its docstring rolled back |
| 40-07 | 9 | 9 | | | |
| 40-08 | 23 | | 1 (8.14) | 22 | R12 capabilities, c45edbc helper and teardown line, README guard text rolled back (S1-S3 kept: see 9.3) |
| 40-09 | 12 | 9 | | 3 | 9.1 and 9.2 need the c45edbc teardown line; 9.11 override replaced on 2026-10-12 |
| 40-10 | 16 | | | 16 | R0 allow-list, R6/R12 widening rolled back |
| 40-11 | 11 | 2 | 1 (11.1) | 8 | docstrings and README guard text rolled back; 11.8's "never opened" ban reversed with c45edbc |
| 40-12 | 5 | | | 5 | 2026-10-10 override process; override replaced |
| 40-13 | 12 | | | 12 | R0 automagic, R6 names, ipykernel dev extra rolled back |
| 40-14 | 9 | | | 9 | docstring, README and override round 4 rolled back |
| 40-15 | 12 | | | 12 | R6 and R12 additions rolled back |
| 40-16 | 11 | | | 11 | docstring, README and override round 5 rolled back |
| 40-17 | 12 | | | 12 | R12 shell forms, R3/R4 string reading rolled back |
| 40-18 | 12 | | 1 (18.6) | 11 | docstring, README and override round 7 rolled back |
| ROADMAP SC | 6 | 5 | 1 (SC2a) | | |
| **Total** | **217** | **84** | **4** | **129** | |

**Score:** 88/88 in-scope truths (84 verified, 4 passed by override), 0 present but behavior-unverified; 129 withdrawn.

## Goal Achievement

### Observable Truths: ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the dev DB; snapshot notebooks read it only through a read-only online backup, the rest start fresh; the three byte-copying notebooks rebuilt on the helper; all re-executed with output; README rule and lists | ✓ VERIFIED | All eight `isolation_problems()` = [] and Django-backed; modes 6 fresh / 2 snapshot match the README lists (TestReadmeStatesTheIsolationRule green); `snapshot_database` opens `?mode=ro` and uses `Connection.backup()` (fomo_nb_utils.py:104-108); no `shutil.copy*` in any notebook; dev DB mtime 2026-10-10 00:16, before the 2026-10-12 re-execution; D-04 sentence verbatim in README:29, ROADMAP SC 1, REQUIREMENTS WARN-05 |
| SC2a | The guard fails a notebook whose current source routes Django to the dev DB, even with stale output | PASSED (override) | Override 1: accepted by Tim Lister on 2026-10-12T20:19:00Z. Guard is 8ab5247; the residual it names reproduces as stated (see Override claim check) |
| SC2b | The DB-path check raises an ordinary exception under `python -O` | ✓ VERIFIED | `check_resolved_database`/`check_database_settings` use `raise` (fomo_nb_utils.py:130-167); 0 asserts in the helper (AST); `test_a_databases_override_is_refused_before_migrate` runs plain, `-O` and `PYTHONOPTIMIZE=1` (green); no notebook setup cell has an assert and none wraps a write (246 demo asserts, 0 in setup cells, 0 around a write) |
| SC3 | telescope_runs_demo reports the dip in one unit | ✓ VERIFIED | Committed `Las Campanas (2402 m): 1.4376 deg`; no `arcmin deg` in any notebook; pin test green |
| SC4 | High-band marker, template test, Bootstrap 5.3 class names only | ✓ VERIFIED | `border-0 border-start border-4 border-success` in attribution_queue.html; `test_high_band_rows_carry_the_left_bar_marker_and_others_do_not` and TestTemplatesUseBootstrap5ClassNames green; Chromium `test_high_band_row_draws_a_green_left_bar_only` OK |
| SC5 | Undo on a Dismissed row lands on the same page with the section open | ✓ VERIFIED | `attribution_return_state`, `_redirect`, `dismissed_section_open` in campaign_views.py; Undo forms carry the state (campaign_tables.py:577-595); TestAttributionReturnState green; Chromium `test_undo_on_a_dismissed_row_keeps_the_section_open_on_the_same_page` OK |

### Override claim check (what the "What each claim rests on" table cites)

Line numbers in the draft's table were checked against the current files: R2 :363-392, R3 :394-399, R4 :264-278,
R5/R7 :281-295, R6 :298-316, gate :344-349 and :190, R8 :351-361, R9 :401-412, R10 :414-433, R11 :435-446, R0 :334-342,
`UNPARSEABLE_SUSPECT_TEXT` :83, helper :275, :295, :137, :115, :306, :184, :199, :364, settings.py:100, docstring :18-19,
README :59, :69, :90, fomo_nb_utils.py:263-267. All point at the code described.

| Claim | Verifier check | Result |
|-------|----------------|--------|
| R2: skipped helper or a second call | Second bare call reported; a call through `fomo_nb_utils.` gives [] (named as missed) | True |
| R3: Django import before the call | Three 40-02 bypasses before the setup statement on a real copy: reported | True for import statements (see human item on "first") |
| R4: `django.setup()`, `from django import setup`, `import django as x` | Each reported | True |
| R5/R7: `FOMO_DATABASE_PATH` / `fomo_db.sqlite3` in a string, `FOMO_DATABASE_PATH=` keyword | Each reported; a bytes literal gives [] | True ("in a string"; bytes in advisory) |
| R6: `shutil.copy`, `copy2`, `copyfile`, "also under an import alias" | Same-cell alias reported; `from shutil import copy2 as c` reported; **alias imported in another cell gives []** (`shutil_names` is rebuilt per cell, :304-307) | **False for a cross-cell alias** (human item) |
| R0, R5, R6, R7 apply to every notebook; the rest only to Django-backed | Code order :334-349 | True |
| Output checks R8-R11 | Rules present; synthetic tests green | True |
| Not seen: string code incl. the setup call's argument; shell lines; monkey-patching; run-time values | timeit string in the argument: [] (round-7 CR-01); `!python manage.py flush` cell before the helper: [] | True |
| Not seen: `!` cell "after the teardown" | A cell after the teardown is reported by R10 ("the last code cell does not end with ...") | Overstates the residual (safe direction; human item) |
| Not seen: re-pointing settings/connections after the helper | `connections['default'].settings_dict['NAME'] = ...` with a built name: [] | True |
| Helper refuses a running Django (:275) | Up-front check, outside the `try` | True |
| Settings checked before `django.setup()` (:295) | `check_database_settings` order: extra alias, routers, default NAME | True |
| Failure path closes, switches to dummy, removes dir, restores env, re-raises | :306-314 and :184-207 | True |
| Close can fail first; Django's error replaces the helper's | `connections.close_all()` (:199) runs before the switch; no `default` alias makes `ConnectionHandler.configure_settings` raise `ImproperlyConfigured` (Django 5.2.17 source read) | True |
| Teardown second line is fixed text (:364) | `print('The developer database (src/fomo_db.sqlite3) was never opened for writing.')` | True |
| Another process falls back to the dev file | settings.py:100 `os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, 'fomo_db.sqlite3')` | True |
| Override 2 (1): a comment in an unparseable cell trips R0 | `%%time` / `# fomo_db.sqlite3 mention` reported; same comment in a plain cell gives [] | True |
| Override 2 (2): README `/tmp` vs `tempfile`/`TMPDIR` | README :59, :87 say `/tmp`; helper uses `tempfile.mkdtemp` (:233); README :78 sets `TMPDIR=/tmp` | True |
| Override 2 (3): "never sets Django up, routes the database itself or byte-copies" | README :89-90, docstring :13 | True; but README "calls the helper first" / docstring "before any Django import" are a fifth place of the same kind (human item) |
| Override 2 (4): "a refusal or failure switches every alias to the dummy backend" | README :69, helper docstring :263-267 | True |

### Observable Truths: kept plans

| # | Truth (abridged) | Status | Evidence |
|---|------------------|--------|----------|
| 1.1 | Helper tracked; `special_locations` byte-identical to feature/skymap; import leaves no Django | ✓ VERIFIED | `git ls-files`; source segment equal to `origin/feature/skymap`; `test_importing_the_module_does_not_import_django` green |
| 1.2 | Fresh setup routes, migrates, prints Resolved line; teardown removes the directory | ✓ VERIFIED | `test_fresh_setup_routes_migrates_and_tears_down` green |
| 1.3 | DATABASES override refused under python, `-O`, `PYTHONOPTIMIZE=1`; no assert | ✓ VERIFIED | Test green (three subtests); AST: 0 asserts |
| 1.4 | `mode=ro` + backup; snapshot with no dev DB raises before mkdtemp | ✓ VERIFIED | :104-108, :231-233; two tests green |
| 1.5 | Four converted notebooks call the helper with the right mode, top level, before any Django import | ✓ VERIFIED | Inert `notebook_mode()`: load/backfill/campaign fresh, reconcile snapshot; guard [] |
| 1.6 | Re-executed with `TMPDIR=/tmp`; counts 1..N; `/tmp/fomo-notebook-db-*`; teardown last | ✓ VERIFIED | Script over all eight (re-executed on 819321e by 261012-g4y) |
| 1.7 | CLAUDE.md map pairs the helper with every notebook | ✓ VERIFIED | CLAUDE.md:178 |
| 1.8 | Resolved-path check uses `resolve()` | ✓ VERIFIED | :130; two unit tests green |
| 1.9 | Empty input never routes to the dev DB | ✓ VERIFIED | Non-empty `str(scratch_db_path)` assigned; ValueError/RuntimeError before mkdtemp |
| 1.10 | Guard and projector test pass after each commit | ✓ VERIFIED | Historical; both green now |
| 2.1 | WR-13 bypasses reported (synthetic and on a real copy) | ✓ VERIFIED | 30 synthetic tests green; verifier replayed the assignment, aliased-setup and aliased-import bypasses on a copy of import_campaign_csv_demo: each reported |
| 2.2 | Counts not 1..N or error output reported; all eight pass | ✓ VERIFIED | Tests green |
| 2.3 | Not vacuous: eight names, each Django-backed with one setup call | ✓ VERIFIED | `test_the_real_notebook_check_is_not_vacuous` green |
| 2.4 | Resolved line in `fomo-notebook-db-*`; teardown last with output | ✓ VERIFIED | Real-notebook test and verifier script |
| 2.5 | No shutil copy, no literal; comments are not problems | ✓ VERIFIED | Tests green |
| 2.6 | No setup-cell assert; no assert wraps a write; audit | ✓ VERIFIED | Audit now 246 (40-04 added ten demo asserts to campaign_lifecycle: 103 vs 93), 0 in setup cells, 0 around a write |
| 2.7 | No `arcmin deg`; dip 1.42-1.46 | ✓ VERIFIED | 1.4376 |
| 2.8 | Modes of the last four; projector cell 7022f987 kept | ✓ VERIFIED | Ids present; `test_projector_demo_notebook` green |
| 2.9 | D-04 sentence in README, ROADMAP, REQUIREMENTS; README lists tested | ✓ VERIFIED | grep; TestReadmeStatesTheIsolationRule green |
| 2.10 | IN-13: CSV inside `scratch_db_dir`, fixture in a `with` | ✓ VERIFIED | Cells 3b49c87a and 16d14cf8 |
| 2.11 | IN-14: utf-8 | ✓ VERIFIED | :708, :766, :782, :785 |
| 2.12 | No credential-shaped text | ✓ VERIFIED | Test green; verifier regex over all eight files |
| 2.13 | 37.1 ledger WR-13/14/16/17, IN-13/14/17 fixed | ✓ VERIFIED | 37.1-REVIEW-DISPOSITION.md rows 336-347 |
| 2.14 | README lists in file-name order; compared as sets | ✓ VERIFIED | README :33-43 |
| 3.1-3.8 | High-band marker, Chromium computed style, BS 5.3 names, class guard, IN-12 scans, render suites, empty and encoding edges | ✓ VERIFIED | Files unchanged since 8ab5247; test_attribution_template, test_campaign_attribution_views, both scan classes and the Chromium marker test pass |
| 4.1-4.14 | Undo returns to the open Dismissed page; one `_redirect`; carried state; Confirmed landing; last-page clamp; tampering dropped; IN-15 boolean; browser SC 5; runbook and lifecycle notebook; 37.1 ledger; staff gating; edges | ✓ VERIFIED | Files unchanged; TestAttributionReturnState, TestAttributionStaffGating green; Chromium Undo test OK; runbook :968 "same page of the Dismissed section"; lifecycle notebook shows the marker classes and `dismissed-page=` redirect; 37.1 rows WR-15, WR-18, IN-12, IN-15 fixed. 4.9's "last notebook re-executed" is historical (all eight were re-executed later on purpose) |
| 5.1-5.8 | Refusal leaves no reachable database; failure after setup; up-front refusal never disables a working Django; aliases and routers; ready line; snapshot run; teardown location; environment restore | ✓ VERIFIED | Each named test exists and passes (behavioral, child processes) |
| 5.9 | README states the check-before-setup sentence, teardown checks, environment restore; D-04 and lists kept | ✓ VERIFIED (wording superseded) | The 40-05 clause "after a refusal, or any later failure in the setup" was replaced by 40-08's S1+S2, kept by the developer in 261012-g4y; teardown checks :73-75; README test green |
| 5.10-5.12 | Adjacency, empty and ordering edges | ✓ VERIFIED | `test_refuses_a_second_alias_even_when_it_names_the_scratch_file`, `test_refuses_a_missing_or_empty_default`, `test_names_the_extra_alias_before_the_routers`; except-path order at :306-314 |
| 5.13 | All 14 helper tests from 0ad6991 still exist and pass | ✓ VERIFIED | Name diff: none missing; module green |
| 7.1 | IN-05: eight NamedTemporaryFile calls pass `dir=scratch_db_dir` | ✓ VERIFIED | 8/8 in the seven cells; a0c4592f output `/tmp/fomo-notebook-db-nkqqkf8j/...`; directory gone; 2bd45621 explains it |
| 7.2 | All eight re-executed on the 40-05 helper; ready line; Resolved line; counts; teardown; no `/home/` in setup or teardown output | ✓ VERIFIED | 819321e is 40-05's helper; verifier script found no issue in any notebook |
| 7.3 | Setup markdown carries the README's setup-check sentence; none describes setup-before-check | ✓ VERIFIED | S1 (and S2, S3) in all eight setup cells and in the README; no old wording |
| 7.4 | No other code cell changed | ✓ VERIFIED (historical) | Since 488a7bb only 40-09's three planned cells differ |
| 7.5 | Snapshot ids kept; prose matches output | ✓ VERIFIED | Ids present; projector test green; prose: UAT test 1 (human, pass) |
| 7.6 | Home-path inventory | ✓ VERIFIED | Only the spiceypy line in campaign_lifecycle d687bfab |
| 7.7 | Guard, helper, projector tests pass | ✓ VERIFIED | Green (R12 part withdrawn) |
| 7.8, 7.9 | Ledger and gate at the time | ✓ VERIFIED (historical) | Unchanged commits; carried from earlier reports |
| 9.3 | Setup markdown has S1, S2, S3, same as the README; no 2026-10-09 refusal or re-run wording | ✓ VERIFIED | Verifier script, all eight |
| 9.4 | NF-14 label in 2c723427 and 48b9f602; no hard-coded pk | ✓ VERIFIED | `NF14_EXPECTED_UNEXPLAINED` defined in 2c723427, used in 48b9f602; label on each CommandError line |
| 9.5 | 35debc54 cites real headings and output lines | ✓ VERIFIED | Three quoted phrases are headings, one is an output line |
| 9.6 | Code cells changed since 488a7bb: exactly the three | ✓ VERIFIED | Cell comparison against 488a7bb |
| 9.7 | Snapshot ids kept; prose matches | ✓ VERIFIED | As 7.5 |
| 9.8 | Guard, helper, projector tests pass | ✓ VERIFIED | Against the 8ab5247 guard (the 40-08 guard was rolled back) |
| 9.9 | Home-path inventory | ✓ VERIFIED | As 7.6 |
| 9.10, 9.12 | Ledger and gate at the time | ✓ VERIFIED (historical) | Carried |
| 11.7 | Fresh-mode setup sentences F7, F49 (and F23); teardown sentences | ✓ VERIFIED (teardown wording superseded) | F7 in 7dfef438, F49 in 49ff72df, F23 in 23326d63; the three T1 sentences were replaced by 261012-g4y's sentence, which matches 819321e's fixed second line |
| 11.9 | Markdown-only change at 40-11 | ✓ VERIFIED (historical) | Carried |
| 8.14, 11.1, 18.6 | Guard docstring and README state what the rules check | PASSED (override) | Override 2: accepted by Tim Lister on 2026-10-12T20:28:10Z |

### Prohibitions (kept plans: 23)

| Plan | Prohibition | Status | Evidence |
|------|-------------|--------|----------|
| 40-01, 40-02, 40-05, 40-07, 40-09 | Never write, byte-copy or move `src/fomo_db.sqlite3`; snapshot only through `mode=ro` | ✓ VERIFIED (test) | Guard and helper tests; dev DB mtime 2026-10-10 00:16; every Resolved line in `/tmp/fomo-notebook-db-*` |
| 40-01 | No silent fallback when snapshot finds no dev DB | ✓ VERIFIED (test) | `test_snapshot_mode_without_a_developer_database_raises_before_creating_anything` |
| 40-02 | No weakening for a named notebook (judgment) | ✓ VERIFIED (judgment) | EXPECTED_NOTEBOOKS is used only for presence; no skip, expectedFailure or tag decorator in either module |
| 40-02, 40-07, 40-09 | No credential-shaped text | ✓ VERIFIED (test) | Test green |
| 40-03 | No BS4 compatibility stylesheet, no third-party template edits, scan not narrowed (judgment) | ✓ VERIFIED (judgment) | Unchanged since prior verification; `test_third_party_app_template_directories_are_not_scanned` green |
| 40-04 | State carried by hidden inputs and server redirect only (judgment) | ✓ VERIFIED (judgment) | Unchanged; `_return_state_inputs` and the partial include |
| 40-04 | campaign_lifecycle on fresh scratch | ✓ VERIFIED (test) | mode fresh; guard green |
| 40-05 | Up-front refusal outside the `try` | ✓ VERIFIED (test) | :275 before :283; `test_setup_refuses_when_django_is_already_set_up` |
| 40-05 | Call shape and both teardown lines unchanged | ✓ VERIFIED (test) | :317-319, :363-364 |
| 40-05 | Out-of-scope files untouched; no helper test loosened | ✓ VERIFIED | Historical; 14 old tests present |
| 40-07, 40-09 | No `--allow-errors`; counts 1..N; only planned code cells changed; ledger rows; VERIFICATION untouched at the time | ✓ VERIFIED | Script; historical |

The 44 prohibitions of the withdrawn plans are withdrawn with them.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/notebooks/fomo_nb_utils.py` | Scratch helper, settings check before setup, fail-closed path, teardown checks | ✓ VERIFIED | 819321e; `def check_database_settings`, `_active_setup`, dummy backend; used by all eight notebooks |
| `solsys_code/tests/test_notebook_helper.py` | Child-process tests of the helper | ✓ VERIFIED | 27 tests incl. the -O refusal and refusal-state tests; green |
| `solsys_code/tests/test_pre_executed_notebooks.py` | 8ab5247 guard R0-R11, real-notebook and README tests | ✓ VERIFIED | Green; limits accepted by override 1 |
| `docs/notebooks/README.md` | D-04 rule, mode lists, helper section | ✓ VERIFIED | Wording limits accepted by override 2 |
| Eight notebooks | On the helper, re-executed, prose matching | ✓ VERIFIED | Script plus UAT tests 1-2 |
| `src/templates/campaigns/attribution_queue.html`, partial, `campaign_views.py`, `campaign_tables.py`, runbook | Marker, BS 5.3, return state | ✓ VERIFIED | Unchanged since 8ab5247; tests green |
| `CLAUDE.md` | Helper pairing line | ✓ VERIFIED | :178 |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| Each notebook setup cell | fomo_nb_utils.py | `from fomo_nb_utils import setup_scratch_django, teardown_scratch_django` | ✓ WIRED (eight) |
| setup_scratch_django | settings.py DATABASES NAME | `os.environ['FOMO_DATABASE_PATH'] = ...` (:287), checked at :295 before `django.setup()` | ✓ WIRED |
| setup_scratch_django failure path | `_disable_django_databases` | `except BaseException` (:306) | ✓ WIRED |
| teardown_scratch_django | `_active_setup` | recorded dir and environment | ✓ WIRED |
| Guard | helper | `test_the_guard_names_match_the_helper` | ✓ WIRED |
| Guard README test | README headings | `README_SNAPSHOT_HEADING` / `README_FRESH_HEADING` | ✓ WIRED |
| campaign_tables Undo forms | `AttributionDecisionView._redirect` | hidden inputs re-validated by `attribution_return_state(request.POST)` | ✓ WIRED |
| AttributionQueueView | attribution_queue.html | `return_state_fields`, `dismissed_section_open` | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Notebook Resolved line | resolved NAME | lazy settings, checked before setup | Yes | ✓ FLOWING |
| Teardown second line | none | fixed text (:364) | No (by design, stated in override 1 and the teardown prose) | ✓ As documented |
| Attribution page state | band, page, sort | POST, validated, redirected | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Guard, helper, projector, attribution template and views, template scans | `python manage.py test --noinput --parallel auto --exclude-tag ephemeris_segfault solsys_code.tests.test_pre_executed_notebooks ...test_notebook_helper ...test_projector_demo_notebook ...test_attribution_template ...test_campaign_attribution_views ...TestTemplatesUseBootstrap5DataAttributes ...TestTemplatesUseBootstrap5ClassNames` | Ran 181 tests, OK | ✓ PASS |
| Attribution browser tests | the three named TestBootstrap5Rendering attribution tests | Ran 3 tests, OK | ✓ PASS |
| Real notebooks | scratchpad `probe.py` | eight [] , Django-backed, modes as listed | ✓ PASS |
| Override 1 R6 claim | scratchpad `probe2.py` | cross-cell alias [] ; same-cell and plain controls reported | ✗ claim false (human item) |
| Notebook output and prose | scratchpad `nbcheck.py`, `nbcheck2.py` | no issue in any notebook | ✓ PASS |
| Dev DB untouched | `git status --short src/`; mtime | clean; 2026-10-10 00:16 | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No `scripts/*/tests/probe-*.sh` exists, and no plan declares one.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| WARN-02 | 40-02 (kept); 40-06..40-18 (withdrawn) | Guard fails a notebook whose current source routes Django to the dev DB | ✓ SATISFIED (override) | SC2a by override 1 |
| WARN-03 | 40-01, 40-02, 40-05, 40-07, 40-09 | Ordinary exception under `-O` | ✓ SATISFIED | SC2b |
| WARN-04 | 40-03, 40-04 | BS 5.3 only; marker; template test | ✓ SATISFIED | SC4 |
| WARN-05 | 40-01, 40-02, 40-05, 40-07, 40-09 | Audit, shared helper, no byte copy or write, README rule and lists | ✓ SATISFIED | SC1 |
| WARN-06 | 40-02 | Dip in one unit | ✓ SATISFIED | SC3 |
| WARN-07 | 40-04 | Undo keeps Dismissed open on the same page | ✓ SATISFIED | SC5 |

All six IDs appear in plan front matter and in REQUIREMENTS.md (Phase 40). None is orphaned. REQUIREMENTS.md still shows
them unchecked with "Gaps Found"; the orchestrator updates that.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| solsys_code/tests/test_pre_executed_notebooks.py | 304-307 | `shutil_names` rebuilt per cell, so an alias from another cell is not followed | ⚠️ Warning | Makes override 1's R6 sentence false for a cross-cell alias (human item); no Django route |
| docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb | cell 4862a7df | Uncommitted developer edit | ℹ️ Info | Outside the phase; changes covered_digest when committed or reverted |

`TBD` in campaign_tables.py, campaign_views.py and import_campaign_csv_demo is the domain word for an undetermined
observing window, not a debt marker. No `FIXME`, `XXX`, `TODO` or `HACK` in the phase files.

40-REVIEW.md (round 7), 40-VALIDATION.md and 40-SECURITY.md still describe the rolled-back guard. The developer asked
for the review not to be re-run; they are planning records, not must-haves.

### Human Verification Required

UAT (2026-10-12, 10 passed, 7 skipped for rolled-back work) covers 40-01 D6, 40-02 D7, 40-04 D7, 40-05 D7 and the
override acceptance (40-12 D3, 40-14 D4). One new item:

1. **Override wording against the code**
   - **Test:** Read the "Override claim check" table.
   - **Expected:** Confirm both overrides stand as written, or amend them. The R6 sentence holds only when the alias is
     imported in the same cell as the call. Override 2's "four places" leaves out "calls the helper first" and "before
     any Django import". The `!` cell "after the teardown" is in fact reported by R10.
   - **Why human:** Only the developer can change an accepted override. No code change is needed either way.

### Gaps Summary

There are no gaps. Every in-scope truth is verified or passed by a developer-accepted override, and every rolled-back
must-have is recorded as withdrawn. The phase waits only on the developer's answer to the one wording item above.

---

_Verified: 2026-10-12T20:40:40Z_
_Verifier: Claude (gsd-verifier)_
