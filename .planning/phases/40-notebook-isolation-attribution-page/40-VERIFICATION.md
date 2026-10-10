---
phase: 40-notebook-isolation-attribution-page
verified: 2026-10-10T00:55:00Z
status: gaps_found
score: 49/50 must-haves verified (prohibitions 7/8)
covered_files:
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-01-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-02-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-03-SUMMARY.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-PLAN.md"
  - ".planning/phases/40-notebook-isolation-attribution-page/40-04-SUMMARY.md"
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
  - "solsys_code/campaign_tables.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/tests/test_attribution_template.py"
  - "solsys_code/tests/test_bootstrap5_rendering.py"
  - "solsys_code/tests/test_campaign_attribution_views.py"
  - "solsys_code/tests/test_notebook_helper.py"
  - "solsys_code/tests/test_pre_executed_notebooks.py"
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
covered_digest: "v3:sha256:4a8584859769dcd37f734f1459cbe8bc1ad1a4d1c4933f06f240288552201279"
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "40-01 prohibition (test tier): MUST NOT write to, byte-copy or move the developer database from the helper, a notebook, a test or an execution step -- and the phase goal's 'guarded by a check' (review CR-01)"
    status: partial
    reason: >-
      setup_scratch_django() calls django.setup() BEFORE check_resolved_database(). When the check refuses
      (a local_settings.py or DJANGO_SETTINGS_MODULE whose DATABASES beats FOMO_DATABASE_PATH), the except
      block only removes the scratch directory and re-raises; apps.ready stays True and
      connections['default'] still points at the refused database. Reproduced by the verifier in a
      throwaway subprocess (override DATABASES -> scratchpad other.sqlite3): RuntimeError raised, then
      'apps.ready after refusal: True', 'next cell would use: .../other.sqlite3', and a CREATE TABLE
      succeeded against the refused file. The next notebook cell (or Run All Below) would therefore write
      to whatever database the override names -- the developer database in the realistic case. The setup
      cell cannot be re-run in that kernel (apps.ready makes it refuse), so the poisoned state lasts until a
      kernel restart. The literal 40-01 WARN-03 truth (setup itself exits non-zero, prints no Resolved line,
      never creates the other file) holds and is tested; the refusal is not fail-closed. The plan itself
      specified this order (40-01-PLAN.md interfaces, steps 5-6), so the executor followed the plan.
    artifacts:
      - path: "docs/notebooks/fomo_nb_utils.py"
        issue: "lines 199-212: django.setup() precedes check_resolved_database(); the failure path leaves Django configured against the refused database"
      - path: "solsys_code/tests/test_notebook_helper.py"
        issue: "test_a_databases_override_is_refused_before_migrate never checks the state left behind after the refusal (apps.ready, connection usability)"
    missing:
      - "Read django.conf.settings.DATABASES['default']['NAME'] and call check_resolved_database() BEFORE django.setup() (settings load without populating the app registry or opening a connection)"
      - "On any failure after django.setup() (e.g. in migrate), close all connections and leave the ORM unable to reach a database before re-raising"
      - "A TestHelperRunsForReal case that catches the refusal in the child process and asserts django.apps.apps.ready is False and a raw connection.cursor() cannot write the override database"
  - truth: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale (review WR-01)"
    status: partial
    reason: >-
      The guard catches every WR-13 bypass the 40-02 must-have enumerates (all have passing tests), but it
      does not see a notebook that re-routes Django after a correct helper call. The verifier injected plain,
      non-obfuscated cells into a copy of the real import_campaign_csv_demo.ipynb (committed output kept) and
      isolation_problems() returned [] for: (a) `from fomo_nb_utils import developer_database_path`;
      `connections['default'].settings_dict['NAME'] = str(developer_database_path(repo_root_path))`, and
      (b) `override_settings(DATABASES={'default': {..., 'NAME': str(developer_database_path(repo_root_path))}}).enable()`.
      Only a literal 'fomo_db.sqlite3' string is caught (control case). Both use the helper's own public
      function, so this is not an exec()-style evasion; the module docstring's claim that the only blind spot
      is code the AST cannot read is therefore false.
    artifacts:
      - path: "solsys_code/tests/test_pre_executed_notebooks.py"
        issue: "R5/R7 (_database_routing_problems) only match the FOMO_DATABASE_PATH and fomo_db.sqlite3 string constants; no rule for settings_dict / DATABASES / override_settings / modify_settings / developer_database_path / sqlite3.connect references; docstring lines 18-20 overstate coverage"
    missing:
      - "An R-rule over every parsed cell that reports any attribute or subscript reference to DATABASES or settings_dict, any override_settings/modify_settings name, any developer_database_path or prepare_scratch_database/snapshot_database use outside the helper, and any sqlite3 import or sqlite3.connect call (none of the eight notebooks uses these today -- re-run test_every_real_pre_executed_notebook_is_isolated to confirm)"
      - "Synthetic tests for each new rule, including the settings_dict re-route replayed on a copy of a real notebook"
      - "Correct the module docstring to describe what the guard does and does not see"
human_verification:
  - test: "Read the rewritten notebook prose in the eight pre-executed notebooks (setup, teardown and data-dependent claims), especially reconcile_campaign_runs_demo's cutover cells (8b703ea8, ce1fd748, e1edbaa7, d2adacf8), which now describe a snapshot-dependent no-op"
    expected: "Each claim matches the committed output beside it"
    why_human: "Prose accuracy against regenerated output is a reading judgment no test asserts (40-01 D6, 40-02, 40-04 summaries flag it human_judgment: true)"
  - test: "Read docs/runbooks/telescope_runs_calendar.rst lines ~935-990 (attribution pass: green bar, 'The page keeps your place', last-page landing, Confirmed landing)"
    expected: "Accurate and clear; note review IN-03 -- 'each table's page and sort' should not read as covering the two worklists, whose event_page/record_page are not kept"
    why_human: "Documentation wording; tests pin phrases, not meaning"
  - test: "Decide review WR-02: the resolved-path check covers only DATABASES['default'], yet teardown always prints 'The developer database (src/fomo_db.sqlite3) was never opened for writing.' (unmeasured; guard rule R10 requires the line in every notebook)"
    expected: "Either check every alias and refuse DATABASE_ROUTERS in the helper, or reword the teardown line to state only what was checked"
    why_human: "No must-have requires multi-alias handling; whether a published-but-unmeasured claim is acceptable is a developer call"
---

# Phase 40: Notebook Isolation & Attribution Page Verification Report

**Phase Goal:** Every pre-executed demo notebook builds and uses its own scratch database, guarded by a check that catches one that does not, and the staff attribution page renders and pages correctly under Bootstrap 5.3 -- closing the remaining six Phase 37.1 review warnings (WR-13 to WR-18).
**Verified:** 2026-10-10T00:55:00Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Summary

The attribution-page half (WARN-04, WARN-07) is fully delivered and tested, in the browser too. The notebook
half is substantively delivered: all eight notebooks run on the shared helper with the right mode, every
committed output came from one clean run on a `/tmp/fomo-notebook-db-*` scratch database, none byte-copies
the developer database, the README/ROADMAP/REQUIREMENTS carry the D-04 rule verbatim, and the dip is in
degrees (1.4376 deg). Two gaps remain, both in the "guarded" clause of the goal, and both were reproduced by
the verifier: the helper's refusal is not fail-closed (CR-01), and the static guard does not catch a plain
re-route after setup (WR-01). Both fixes are small and local to `fomo_nb_utils.py` and
`test_pre_executed_notebooks.py`.

## Goal Achievement

### Observable Truths -- ROADMAP success criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | No notebook byte-copies or writes the developer DB; snapshot via read-only online backup, others fresh; three byte-copiers rebuilt on `fomo_nb_utils.py`; all re-executed with output; README states rule in same words and lists modes | ✓ VERIFIED | Per-notebook audit: 6 `mode='fresh'`, 2 `mode='snapshot'` (project_observation_calendar, reconcile_campaign_runs); every Resolved line is `/tmp/fomo-notebook-db-*/fomo_db.sqlite3`; counts 1..N, 0 error outputs, last cell `teardown_scratch_django(scratch_db_dir)` with "Removed scratch database directory" output; no `fomo_db.sqlite3` in code. D-04 sentence found once each in README.md, ROADMAP.md, REQUIREMENTS.md; README lists match modes (`test_readme_lists_each_notebook_under_its_mode` passes). `snapshot_database` uses `?mode=ro` URI + `Connection.backup()` (fomo_nb_utils.py:94-98). Normal-path only -- see CR-01 for the refusal path |
| SC2a | The guard fails a notebook whose current source routes Django to the developer DB even with stale output | ✗ FAILED (partial) | WR-13 bypasses all caught (tests at test_pre_executed_notebooks.py:545-695 pass). But a re-route after setup via `connections['default'].settings_dict['NAME'] = str(developer_database_path(repo_root_path))` or `override_settings(DATABASES=...)` on a copy of the real import_campaign_csv_demo returns `[]` (verifier probe). See gap 2 |
| SC2b | Every notebook's DB-path check raises an ordinary exception that fires under `python -O` | ✓ VERIFIED | `check_resolved_database` raises RuntimeError (fomo_nb_utils.py:120-124); no `assert` in helper (`test_module_has_no_assert_statements`); refusal tested under plain, `-O`, `PYTHONOPTIMIZE=1` (`test_a_databases_override_is_refused_before_migrate`, passed in verifier run); no notebook setup cell contains an assert (0 counted) |
| SC3 | telescope_runs_demo reports the horizon dip in a single unit | ✓ VERIFIED | Committed output `Las Campanas (2402 m): 1.4376 deg`; no `arcmin deg` in any notebook output; pinned by `test_telescope_runs_demo_horizon_dip_matches_the_stage1_contract` and `test_no_committed_output_reports_arcmin_deg` |
| SC4 | High-band rows show their marker, a template test asserts it, templates use BS 5.3 names only | ✓ VERIFIED | attribution_queue.html:63,168 `border-0 border-start border-4 border-success`; `AttributionHighBandMarkerRenderTests` (2 tests) pass; Chromium computed-style test `test_high_band_row_draws_a_green_left_bar_only` passed in verifier run; `TestTemplatesUseBootstrap5ClassNames` passes; grep of the 9 swept templates + campaign_tables.py/campaign_views.py finds no BS4 class names |
| SC5 | Undo on a Dismissed row lands on the same page with the section open | ✓ VERIFIED | `_redirect()` (campaign_views.py:1461-1478) keeps validated state + fragment; Undo forms post their own page (campaign_tables.py:577-596); `test_undo_on_a_dismissed_row_reopens_the_dismissed_section_on_the_same_page` (posts the real rendered form) and Playwright `test_undo_on_a_dismissed_row_keeps_the_section_open_on_the_same_page` (27 dismissals, page 2, accepts confirm()) both pass in verifier runs |

### Observable Truths -- plan must_haves (deduplicated against the SCs)

| # | Plan truth (abridged) | Status | Evidence |
|---|-----------------------|--------|----------|
| 1.1 | Helper tracked; `special_locations()` byte-identical to feature/skymap; import pulls in no Django | ✓ VERIFIED | `TestNotebookHelperModule` (4 tests) pass; module imports only stdlib + astropy at top |
| 1.2 | Fresh setup in subprocess creates `fomo-notebook-db-*`, migrates, prints Resolved line; teardown removes it | ✓ VERIFIED | `test_fresh_setup_routes_migrates_and_tears_down` passes |
| 1.3 | DATABASES override refused (RuntimeError, no Resolved line, other file never created) under python/-O/PYTHONOPTIMIZE; no assert | ✓ VERIFIED (literal) | Test passes; but see CR-01 -- the kernel is left usable against the refused DB |
| 1.4 | `snapshot_database` uses `mode=ro` + `backup()`, source bytes unchanged; snapshot with no dev DB raises before mkdtemp | ✓ VERIFIED | `TestSnapshotAndRouting` snapshot tests pass; fomo_nb_utils.py:145-150 raise before `mkdtemp` |
| 1.5 | D-01 verdicts for the four 40-01 notebooks; no own django.setup / FOMO_DATABASE_PATH / fomo_db.sqlite3 / shutil copy | ✓ VERIFIED | Audit above; guard rules R4-R7 green on real notebooks |
| 1.6 | Four notebooks re-executed with TMPDIR=/tmp, counts 1..N, teardown last | ✓ VERIFIED | Audit above |
| 1.7 | CLAUDE.md pairs `fomo_nb_utils.py` with every pre-executed notebook | ✓ VERIFIED | CLAUDE.md:178 |
| 1.8 | Resolved-path check compares `.resolve()`d paths; other file / empty name raises | ✓ VERIFIED | fomo_nb_utils.py:120; two `test_resolved_path_check_*` tests pass |
| 1.9 | Empty input never routes to dev DB: non-empty assignment, unknown mode ValueError, missing dev DB RuntimeError, before any dir | ✓ VERIFIED | fomo_nb_utils.py:145-150,193; `test_unknown_mode_raises_before_creating_anything` passes |
| 1.10 | Old guard and projector test pass after each 40-01 commit | ✓ VERIFIED | Summary self-check + hook ran on both commits; projector test passes now (verifier run) |
| 2.1 | `isolation_problems()` reports each WR-13 bypass, incl. replay on a real notebook copy | ✓ VERIFIED | Tests 545-695 and the real-notebook replay pass (the listed bypasses only; see SC2a) |
| 2.2 | Non-1..N counts / error output reported; all eight pass | ✓ VERIFIED | `test_non_contiguous_execution_counts_are_a_problem`, `test_an_error_output_is_a_problem`, `test_every_real_pre_executed_notebook_is_isolated` |
| 2.3 | Not vacuous: exactly the eight names, each Django-backed, one setup call each | ✓ VERIFIED | `test_the_real_notebook_check_is_not_vacuous` |
| 2.4 | Each setup cell output shows Resolved line in `fomo-notebook-db-*`; teardown last with output | ✓ VERIFIED | Audit above; R9/R10 |
| 2.5 | No shutil copy, no code string naming fomo_db.sqlite3 / FOMO_DATABASE_PATH; comments OK | ✓ VERIFIED | `test_byte_copy_is_a_problem`, `test_developer_database_*`, comment tests |
| 2.6 | No assert in setup cells, none wrapping a write; audit 236 demo asserts | ✓ VERIFIED | Verifier count: 236 at 40-02 end (1ba0832), 246 at HEAD -- the 10 extra are 40-04's new campaign_lifecycle cells (93 -> 103), all read-only per guard R11 (green); 0 in setup cells |
| 2.7 | Fresh mode for import_campaign_csv, sync_gemini, telescope_runs; snapshot for project_observation_calendar in cell 7022f987, projector tests still pass | ✓ VERIFIED | Audit above; `test_projector_demo_notebook` passes |
| 2.8 | D-04 sentence verbatim in README, ROADMAP SC1, REQUIREMENTS WARN-05; README mode lists pinned by test | ✓ VERIFIED | grep -cF = 1/1/1; README tests pass |
| 2.9 | IN-13: CSV written inside `scratch_db_dir`, fixture opened with `with` | ✓ VERIFIED | Ledger + guard green; 40-02 summary deviation notes the correct cell id 16d14cf8 |
| 2.10 | IN-14: guard reads notebooks with `encoding='utf-8'` | ✓ VERIFIED | test_pre_executed_notebooks.py reads with utf-8 |
| 2.11 | No credential-shaped text in any notebook | ✓ VERIFIED | `test_no_credential_shaped_text_in_any_notebook` |
| 2.12 | 37.1 ledger marks WR-13/14/16/17, IN-13/14/17 fixed (front matter + table, citing plan and commit) | ✓ VERIFIED | 37.1-REVIEW-DISPOSITION.md front matter and rows 336-347 |
| 2.13 | README lists each mode's notebooks in file-name order; compared as sets | ✓ VERIFIED | README excerpt is alphabetical |
| 3.1 | Every High-band row has exactly the marker classes, no style; no other row has them | ✓ VERIFIED | `test_high_band_rows_carry_the_left_bar_marker_and_others_do_not` |
| 3.2 | Chromium computed borders 0/0/0/4px solid rgb(25,135,84), no page error | ✓ VERIFIED | Functional test passed in verifier run (WR-03 caveat on its second half: see below) |
| 3.3 | Eight FOMO templates BS 5.3 only; band filter `d-flex align-items-center gap-2` + `form-select form-select-sm`; `ms-1` in campaign_tables.py | ✓ VERIFIED | attribution_queue.html:8,10; campaign_tables.py:433; grep clean |
| 3.4 | `TestTemplatesUseBootstrap5ClassNames` fails on BS4 names / `form-control` select; passes on tree | ✓ VERIFIED | Class passes; pattern self-tests pass |
| 3.5 | IN-12: scans cover every DIRS entry plus in-repo app dirs, never third-party | ✓ VERIFIED | `repo_template_dirs()` test_bootstrap5_rendering.py:54-81; third-party exclusion test at :772-778 |
| 3.6 | Swept-page render suites and data-attribute scan pass | ✓ VERIFIED | Phase modules 168 OK (verifier); orchestrator full suite 2280 + 40 OK on 8ab5247 (no source change since; `git diff 8ab5247 HEAD` touches only .planning) |
| 3.7 | No High-band / empty worklists -> 200 and no marker; class scan not vacuous (>=20 templates) | ✓ VERIFIED | `test_no_marker_when_no_candidate_is_high_band`; `test_the_scan_is_not_vacuous` |
| 3.8 | Whole-token lookaround matching inside `class="..."`, UTF-8 | ✓ VERIFIED | `test_the_pattern_matches_bootstrap4_and_not_bootstrap5_class_names` |
| 4.1 | Real Undo form posted -> `?dismissed-page=1#attribution-dismissed-section`, section `collapse show`, `aria-expanded="true"` | ✓ VERIFIED | `test_undo_on_a_dismissed_row_reopens_the_dismissed_section_on_the_same_page` |
| 4.2 | Every Undo form carries hidden `dismissed-page` (1 on plain load and with request=None) | ✓ VERIFIED | `test_undo_from_the_dismissed_table_always_sends_its_page_even_on_a_plain_load`; campaign_tables.py:595 |
| 4.3 | All five actions redirect via one `_redirect()` keeping exactly the validated keys; forms share one include; worklist actions no fragment | ✓ VERIFIED | Six `return self._redirect(request)` sites; include used 4x in attribution_queue.html; `test_every_action_keeps_band_page_and_sort`, `test_worklist_and_bulk_forms_carry_the_current_state` |
| 4.4 | undo_confirmation -> `#attribution-confirmed-section`; Dismissed open only if it was | ✓ VERIFIED | `test_undo_on_a_confirmed_row_lands_on_the_confirmed_section_and_leaves_dismissed_as_it_was` |
| 4.5 | Last-page undo (26 dismissals) -> `dismissed-page=2`, renders 200 on page 1 open | ✓ VERIFIED | `test_acting_on_the_only_row_of_the_last_page_lands_on_the_new_last_page` |
| 4.6 | ASVS V5 whitelist drops bad band/page/sort/CRLF/unknown keys; path and fragment fixed | ✓ VERIFIED | `test_tampered_return_state_is_dropped`, `test_attribution_return_state_keeps_only_whitelisted_valid_values`; `fullmatch` on `[1-9][0-9]{0,5}` / `-?[a-z_]{1,40}` (campaign_views.py:1197-1198,1229-1236) |
| 4.7 | IN-15: one `dismissed_section_open` boolean drives class, `collapsed`, `aria-expanded` | ✓ VERIFIED | campaign_views.py:1358; attribution_queue.html:253,255,260; three render tests |
| 4.8 | Paired docs: runbook text + campaign_lifecycle_demo marker and Undo cells, re-executed last | ✓ VERIFIED | Runbook lines 938-990; notebook cells print the marker classes and `Undo redirected to '/campaigns/attribution/?dismissed-page=1#attribution-dismissed-section'`; guard green. Wording is a human item |
| 4.9 | 37.1 ledger WR-15, WR-18, IN-12, IN-15 fixed; all D-29 findings fixed | ✓ VERIFIED | Ledger front matter + rows 338-345 |
| 4.10 | Staff gating unchanged | ✓ VERIFIED | `TestAttributionStaffGating` within the 168 OK |
| 4.11 | Last-page clamp by django-tables2; bad page values dropped | ✓ VERIFIED | As 4.5/4.6 |
| 4.12 | Undoing the last dismissal -> `dismissed-page=1`, open, "No dismissals recorded yet." | ✓ VERIFIED | `test_undoing_the_last_dismissal_shows_the_empty_section_open` |
| 4.13 | Posted sort kept when valid; default order otherwise | ✓ VERIFIED | `test_every_action_keeps_band_page_and_sort` |
| UI F1/F3 | `scroll-margin-top: 5rem` on both section divs; "Dismissed section above."; "Confirmation undone -- the pair is now in Dismissed." | ✓ VERIFIED | attribution_queue.html:260,271; campaign_tables.py:790; campaign_views.py:1671; old flash string absent everywhere |

**Score:** 49/50 truths verified (0 present, behavior-unverified). Behavior-dependent truths (redirect state,
section open/closed, last-page landing, refusal) each have a named passing test run by the verifier.

### Prohibitions

| Plan | Prohibition | Tier | Status | Evidence |
|------|-------------|------|--------|----------|
| 40-01 | MUST NOT write/byte-copy/move the developer DB from helper, notebook, test or execution step | test | ✗ FAILED (partial) | Normal path enforced (no copy, `mode=ro`, guard R5-R7). Refusal path: verifier reproduced a write to the refused DB after the RuntimeError (CR-01). Gap 1 |
| 40-01 | MUST NOT fall back silently when snapshot mode finds no dev DB | test | ✓ VERIFIED | `test_snapshot_mode_without_a_developer_database_raises_before_creating_anything` |
| 40-02 | MUST NOT weaken the guard (no exemption list, skipped/deleted real-notebook tests) | judgment | ✓ VERIFIED (judgment) | No per-notebook exemptions in the guard; real-notebook tests are untagged and run in the 168 |
| 40-02 | MUST NOT commit credential-shaped text | test | ✓ VERIFIED | `test_no_credential_shaped_text_in_any_notebook` |
| 40-02 | MUST NOT write/copy the dev DB while converting/re-executing notebooks | test | ✓ VERIFIED | All Resolved lines in `/tmp/fomo-notebook-db-*`; snapshot via helper's `mode=ro` backup |
| 40-03 | MUST NOT satisfy the class guard via BS4 compat CSS, third-party template edits or narrowed scan | judgment | ✓ VERIFIED (judgment) | No compat stylesheet; scan covers every repo-owned dir, not-vacuous test >=20 |
| 40-04 | MUST NOT carry worklist state with HTMX/JS/session | judgment | ✓ VERIFIED (judgment) | Hidden inputs + server-side whitelist only (partial + `_return_state_inputs`) |
| 40-04 | MUST NOT write the dev DB while re-executing campaign_lifecycle_demo | test | ✓ VERIFIED | Its Resolved line is `/tmp/fomo-notebook-db-ej96o7hf/fomo_db.sqlite3`, `mode='fresh'` |

### Code review findings (40-REVIEW.md) -- explicit classification

| Finding | Reproduced? | Classification | Reasoning |
|---------|-------------|----------------|-----------|
| CR-01 refusal leaves Django set up against the refused DB | Yes (verifier subprocess: `apps.ready` True, `CREATE TABLE` succeeded on the refused file) | **Gap (BLOCKER)** -- gap 1 | The goal's "guarded by a check" and the test-tier 40-01 prohibition require the refusal to protect the database; it protects only the setup call itself. T-40-01's mitigation text ("raises before migrate... the other file is never created") holds literally, but the next cell writes. The pre-phase notebooks had the same ordering, and nbconvert / Jupyter Run All stop at the error, which is why an override is a defensible alternative (template below). No later phase covers it |
| WR-01 guard misses post-setup re-routing | Yes (verifier probe: `settings_dict` and `override_settings` via the helper's own `developer_database_path()` -> `[]`) | **Gap** -- gap 2 | ROADMAP SC 2 says the guard fails a notebook whose current source routes Django to the developer DB; this is plain readable source doing exactly that, and the guard docstring claims only exec()-style code escapes. The 40-02 must-have's enumerated WR-13 bypasses are all caught, so an override narrowing SC 2 to that class is the alternative |
| WR-02 only `default` alias checked; unmeasured teardown claim | By reading (fomo_nb_utils.py:206-207, 240) | **Human decision** (WARNING) | No must-have covers multi-alias or router configurations, and none exists in this repo's settings. The published teardown line states something the helper does not measure; reword or extend the check |
| WR-03 functional "other row has no bar" can be vacuous | Not reproduced in practice | **Out of scope** (WARNING, test hardening) | In the current fixture `other_run` shares the campaign and overlapping window, so `candidates_for_event` returns it (window score > 0) and the row exists. The server-side "no other row carries the marker" half is independently pinned by `test_high_band_rows_carry_the_left_bar_marker_and_others_do_not`. Hardening (`assertIn(other_run.pk, bands)` + scoped locator) is worthwhile but does not affect goal achievement |
| IN-01 to IN-07 | -- | Out of scope (Info) | None contradicts a must-have. IN-03 bears on runbook wording (human item 2). IN-05 (temp files left by load_telescope_runs_demo) and IN-06 (teardown prefix-only check) are hygiene |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/notebooks/fomo_nb_utils.py` | helper (setup, snapshot, check, teardown) | ✓ VERIFIED (defect: CR-01) | 240 lines, imported by all 8 notebooks and by both test modules |
| `solsys_code/tests/test_notebook_helper.py` | executed helper tests | ✓ VERIFIED | 14 tests pass |
| `solsys_code/tests/test_pre_executed_notebooks.py` | AST guard, WARN-03/06 checks, README test | ✓ VERIFIED (coverage gap: WR-01) | 799 lines, all tests pass |
| `docs/notebooks/README.md` | D-04 rule, mode lists, usage | ✓ VERIFIED | |
| 8 notebooks under `docs/notebooks/pre_executed/` | on helper, re-executed | ✓ VERIFIED | |
| `src/templates/campaigns/attribution_queue.html` | marker, BS 5.3, open-state boolean, F1 | ✓ VERIFIED | |
| `src/templates/campaigns/partials/attribution_return_state.html` | hidden-input include | ✓ VERIFIED | included 4x |
| `solsys_code/campaign_views.py` | `attribution_return_state`, `_redirect`, `dismissed_section_open` | ✓ VERIFIED | |
| `solsys_code/campaign_tables.py` | Undo forms carry return state + own page | ✓ VERIFIED | |
| `solsys_code/tests/test_bootstrap5_rendering.py` | computed-style tests, class guard, widened scans | ✓ VERIFIED | |
| `solsys_code/tests/test_attribution_template.py` | marker render test | ✓ VERIFIED | |
| `docs/runbooks/telescope_runs_calendar.rst` | kept place, last-page, green bar | ✓ VERIFIED | wording -> human item |
| `CLAUDE.md` | helper paired-docs line | ✓ VERIFIED | |

### Key Link Verification

`verify.key-links` could not evaluate most links (descriptive `from:` fields, exit 1 with "Source file not found"); each was checked by grep instead.

| From | To | Via | Status |
|------|----|-----|--------|
| 8 notebook setup cells | `fomo_nb_utils.py` | `from fomo_nb_utils import setup_scratch_django, teardown_scratch_django` | ✓ WIRED (8/8) |
| `setup_scratch_django` | settings.py `DATABASES` NAME | `os.environ['FOMO_DATABASE_PATH'] = ...` (line 193); settings.py:100 reads it | ✓ WIRED (check ordering defect: CR-01) |
| guard | helper | `SETUP_FUNCTION = 'setup_scratch_django'` vs spec-loaded helper (`test_the_guard_names_match_the_helper`) | ✓ WIRED |
| guard README test | README.md | `README_SNAPSHOT_HEADING` / fresh heading lists | ✓ WIRED |
| attribution_queue.html rows | BS 5.3.3 border utilities | `border-0 border-start border-4 border-success` | ✓ WIRED (computed style verified in Chromium) |
| `repo_template_dirs()` | `settings.TEMPLATES`, `get_app_template_dirs` | shared by both scans | ✓ WIRED |
| campaign_tables.py Undo forms | `AttributionDecisionView._redirect` | `attribution_return_state(request.POST)` (campaign_views.py:1473) | ✓ WIRED |
| `AttributionQueueView.get_context_data` | attribution_queue.html | `return_state_fields`, `dismissed_section_open` (3 places) | ✓ WIRED |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| attribution_queue.html Dismissed section | `dismissed_section_open`, `return_state_fields` | `attribution_return_state(request.GET)` + table prefixed fields | Yes | ✓ FLOWING |
| Undo hidden inputs | `dismissed-page` | `table.page.number` after `RequestConfig.configure` | Yes | ✓ FLOWING |
| Notebook Resolved line | resolved NAME | `django_settings.DATABASES['default']['NAME']` after setup | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-40 unit modules (helper, guard, projector notebook, attribution template/views, BS5 scans) | `python manage.py test <6 modules> --exclude-tag functional --parallel --noinput` | Ran 168 tests, OK | ✓ PASS |
| High-band computed style + Undo browser flow | `python manage.py test ...TestBootstrap5Rendering.test_high_band_row_draws_a_green_left_bar_only ...test_undo_on_a_dismissed_row_keeps_the_section_open_on_the_same_page` | Ran 2 tests, OK | ✓ PASS |
| CR-01 refusal state | throwaway subprocess, override DATABASES -> scratchpad file | RuntimeError, then apps.ready True and CREATE TABLE succeeded on refused file | ✗ FAIL (gap 1) |
| WR-01 guard re-route | `isolation_problems()` on real-notebook copy with injected re-route cells | `[]` for settings_dict and override_settings variants; literal control caught | ✗ FAIL (gap 2) |

### Probe Execution

Step 7c: SKIPPED -- no `scripts/*/tests/probe-*.sh` exist and no plan or summary declares one.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| WARN-02 | 40-02 | Guard fails a notebook whose current source routes Django to the dev DB, regardless of stale output | ✗ BLOCKED (partial) | WR-13 bypass class caught; post-setup re-route not caught (gap 2) |
| WARN-03 | 40-01, 40-02 | No notebook relies on `assert` before migrate; ordinary exception survives `-O` | ✓ SATISFIED | RuntimeError under -O tested; 0 setup-cell asserts; R11 green. Fail-closed weakness tracked under gap 1 |
| WARN-04 | 40-03, 40-04 | Attribution page BS 5.3 only; High-band marker renders; template test asserts it | ✓ SATISFIED | SC4 evidence |
| WARN-05 | 40-01, 40-02 | Audit, three byte-copiers rebuilt on one helper; no byte copy or write; README rule and lists | ✓ SATISFIED (normal path) | SC1 evidence; refusal-path write is gap 1 |
| WARN-06 | 40-02 | Dip in a single unit, re-executed | ✓ SATISFIED | SC3 evidence |
| WARN-07 | 40-04 | Undo keeps Dismissed open on same page | ✓ SATISFIED | SC5 evidence |

All six IDs appear in plan frontmatter and in REQUIREMENTS.md (traceability rows 102-107, all mapped to Phase 40). No orphaned requirements. Note: REQUIREMENTS.md already marks WARN-02 `[x]` Complete; with gap 2 open that is premature.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| docs/notebooks/fomo_nb_utils.py | 199-212 | Guard check after `django.setup()`; failure path does not disable the ORM | 🛑 Blocker | Gap 1 (CR-01) |
| solsys_code/tests/test_pre_executed_notebooks.py | 18-20, 281-295 | Docstring overstates coverage; routing rule matches only two string constants | ⚠️ Warning (backs gap 2) | Gap 2 (WR-01) |
| docs/notebooks/fomo_nb_utils.py | 240 | Unconditional, unmeasured "never opened for writing" line | ⚠️ Warning | Human decision (WR-02) |
| docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb | various | `TBD` | ℹ️ Info | Domain data (`Obs. Date = 'TBD pending Cycle 2'`), not a debt marker |

No TODO/FIXME/XXX/HACK debt markers in lines added by the phase.

### Human Verification Required

These are recorded for the developer; they do not change the gaps_found status.

1. **Notebook prose against regenerated output** -- read the rewritten setup/teardown and data-dependent cells in all eight notebooks, especially `reconcile_campaign_runs_demo`'s cutover cells (now a snapshot-dependent no-op). Expected: every claim matches its committed output. Why human: reading judgment (flagged `human_judgment: true` in 40-01, 40-02 and 40-04 summaries).
2. **Runbook attribution-pass wording** -- `docs/runbooks/telescope_runs_calendar.rst` ~935-990. Expected: accurate; "each table's page and sort" should not imply the worklists' `event_page`/`record_page` are kept (review IN-03). Why human: wording.
3. **WR-02 decision** -- check every DB alias and refuse `DATABASE_ROUTERS` in the helper, or reword the teardown's "never opened for writing" line to what was actually checked. Why human: outside every must-have; acceptability of the published claim is a developer call.

### Gaps Summary

Both gaps share one root concern -- the "guarded" half of the phase goal -- and sit in two files:

1. **CR-01 (BLOCKER):** `setup_scratch_django()` refuses a non-scratch database only after `django.setup()`, and leaves Django configured against it. Fix: resolve and check `settings.DATABASES` before `django.setup()`, disable connections on any later failure, and add a child-process test asserting `apps.ready is False` after the refusal.
2. **WR-01:** the static guard passes a notebook that re-routes the connection after setup through `settings_dict`, `settings.DATABASES`, `override_settings`, or the helper's own `developer_database_path()`. Fix: one more AST rule plus synthetic tests, and correct the docstring.

**If the developer judges either acceptable** (for CR-01: nbconvert and Jupyter Run All stop at the RuntimeError and no `local_settings.py` in this repo overrides `DATABASES`; for WR-01: SC 2 meant the WR-13 bypass class, which is fully caught), add to this file's frontmatter:

```yaml
overrides:
  - must_have: "40-01 prohibition (test tier): MUST NOT write to, byte-copy or move the developer database from the helper, a notebook, a test or an execution step"
    reason: "Refusal raises before migrate; nbconvert and Run All stop at the error; no local_settings DATABASES override exists; fail-closed hardening deferred to <todo/phase>"
    accepted_by: "<name>"
    accepted_at: "<ISO timestamp>"
  - must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"
    reason: "SC 2 targets the WR-13 bypass class, all caught and tested; post-setup re-routing is a deliberate act a static guard cannot fully enumerate; docstring to be corrected"
    accepted_by: "<name>"
    accepted_at: "<ISO timestamp>"
```

No gap is covered by a later phase (Phase 41 triages todos, Phase 42 re-verifies v2.4), so none is deferred.

---

_Verified: 2026-10-10T00:55:00Z_
_Verifier: Claude (gsd-verifier)_
