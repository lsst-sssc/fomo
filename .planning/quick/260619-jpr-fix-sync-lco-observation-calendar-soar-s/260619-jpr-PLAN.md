---
phase: quick-260619-jpr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
autonomous: true
requirements: [QUICK-SOAR-SITE-FIX]

must_haves:
  truths:
    - "A SOAR ObservationRecord with parameters['site']=='sor' produces a CalendarEvent (no longer silently skipped)"
    - "The SOAR test fixtures exercise the real SOAR site code 'sor', not the LCO code 'coj'"
    - "The demo notebook's Phase-5 SOAR fixtures use site 'sor' and a SOAR-identifiable instrument so their CalendarEvent titles read as SOAR, not 'FTS'"
    - "./manage.py test solsys_code.tests.test_sync_lco_observation_calendar passes"
    - "ruff check . and ruff format --check . stay clean"
  artifacts:
    - path: "solsys_code/management/commands/sync_lco_observation_calendar.py"
      provides: "SITE_TELESCOPE_MAP including SOAR site code 'sor'"
      contains: "'sor'"
    - path: "solsys_code/tests/test_sync_lco_observation_calendar.py"
      provides: "SOAR test fixtures using site='sor'"
    - path: "docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb"
      provides: "Phase-5 SOAR fixture cells using site 'sor'"
  key_links:
    - from: "sync_lco_observation_calendar.py:_derive_telescope"
      to: "SITE_TELESCOPE_MAP['sor']"
      via: "dict lookup keyed on parameters['site']"
      pattern: "'sor'"
---

<objective>
Fix the SOAR site-mapping bug in `sync_lco_observation_calendar`: real SOAR
ObservationRecords (`parameters['site'] == 'sor'`) are silently skipped because
`SITE_TELESCOPE_MAP` only maps the LCO site codes `'coj'` and `'ogg'`. The bug
was masked by SOAR test fixtures and demo-notebook fixtures that incorrectly
reuse the LCO site code `'coj'` (and LCO instrument type) for SOAR records, so
the tests passed and the notebook produced LCO-labelled ("FTS") events for SOAR
records — which is why a user reported "no SOAR events".

Purpose: SOAR records must produce SOAR-identifiable CalendarEvents, and the
test/demo fixtures must actually exercise the real SOAR site code so the bug
cannot regress unnoticed.

Output: `'sor'` added to `SITE_TELESCOPE_MAP`; SOAR test fixtures and notebook
SOAR fixture cells corrected to use `site='sor'` and a SOAR-appropriate
`instrument_type`.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
</execution_context>

<context>
@./CLAUDE.md
@solsys_code/management/commands/sync_lco_observation_calendar.py
@solsys_code/tests/test_sync_lco_observation_calendar.py

Confirmed facts (from reading the code during planning):
- `SITE_TELESCOPE_MAP` (lines 16-19) maps only `'coj'->'FTS'` and `'ogg'->'FTN'`.
- `_derive_telescope()` (lines 53-68) does `SITE_TELESCOPE_MAP[site_code]` and
  raises `KeyError` for any unmapped code; `handle()` catches `(KeyError,
  ValueError)` per-record and silently skips (counted under `skipped`).
- SOAR's real site code is `'sor'` (confirmed in
  `tom_observations/facilities/soar.py`: `'sitecode': 'sor'`). SOAR instrument
  codes contain the substring `SOAR` (e.g. the Goodman spectrograph
  `SOAR_GHTS_REDCAM`), unlike the LCO MuSCAT code `'2M0-SCICAM-MUSCAT'`.
- SOAR test fixtures that must change: observation_ids `610003` (test_select_03,
  ~line 396), `620002` (test_select_04, ~line 414), `630001` (test_select_05,
  ~line 443). Each is created via `self._create_record(..., facility='SOAR')`
  and inherits the `_parameters` default `site='coj'` / `instrument_type=
  '2M0-SCICAM-MUSCAT'` (`_parameters` defined ~lines 16-43).
- Notebook SOAR fixtures that must change: `demo-603002` (cell 20) and
  `demo-604002` (cell 22), both currently `'site': 'coj'`,
  `'instrument_type': '2M0-SCICAM-MUSCAT'`.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add SOAR site code to SITE_TELESCOPE_MAP and fix SOAR test fixtures</name>
  <files>solsys_code/management/commands/sync_lco_observation_calendar.py, solsys_code/tests/test_sync_lco_observation_calendar.py</files>
  <action>
In `sync_lco_observation_calendar.py`, add the SOAR site code to
`SITE_TELESCOPE_MAP` (lines 16-19): add the entry `'sor': 'SOAR'`. Use the
telescope label `'SOAR'` (the SOAR telescope's common name), consistent with
the existing `'FTS'`/`'FTN'` short-label style — there is no Faulkes-style
two-telescope abbreviation for SOAR, so its common name is the natural label.
Update the existing comment above the dict to note that `'sor'` is the real
SOAR site code (confirmed against `tom_observations.facilities.soar` —
`'sitecode': 'sor'`), not an [ASSUMED] value like the LCO entries. Do NOT change
the docstrings on `_derive_telescope` that say "LCO site code" beyond what is
needed; if convenient, broaden them to "LCO/SOAR site code" for accuracy.

In `test_sync_lco_observation_calendar.py`, fix the three SOAR fixtures so they
exercise the real SOAR site code instead of inheriting the LCO default. For each
of the three `self._create_record(..., facility='SOAR')` calls — observation_ids
`610003` (test_select_03), `620002` (test_select_04), `630001`
(test_select_05) — add the overrides `site='sor'` and
`instrument_type='SOAR_GHTS_REDCAM'` (a SOAR Goodman spectrograph code; any code
containing 'SOAR' is acceptable, but use this concrete value). These overrides
flow through `_create_record`'s `**parameter_overrides` into `_parameters`.
Leave the LCO fixtures in those same tests unchanged (they correctly use the
`'coj'`/MuSCAT defaults). Do NOT change `_parameters`' defaults — the LCO
default is correct for the many LCO-only fixtures.

Note: the existing pre-fix tests pass precisely because the SOAR fixtures used
`'coj'` (which IS mapped). After this change they use `'sor'`, which only maps
because Task 1 also added `'sor'` to the dict — so the two edits in this task
are mutually dependent and must land together. This is exactly the regression
guard we want: had the test used `'sor'` from the start, the original bug would
have failed the SOAR tests.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; ./manage.py test solsys_code.tests.test_sync_lco_observation_calendar 2>&amp;1 | tail -20</automated>
  </verify>
  <done>SITE_TELESCOPE_MAP contains 'sor': 'SOAR'; the three SOAR test fixtures pass site='sor' and a SOAR instrument_type; the full test_sync_lco_observation_calendar suite passes (in particular test_select_03/04/05 now genuinely exercise site 'sor').</done>
</task>

<task type="auto">
  <name>Task 2: Fix Phase-5 SOAR fixture cells in the demo notebook</name>
  <files>docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb</files>
  <action>
Edit the cell *source* of the two Phase-5 SOAR fixture cells in the notebook so
they create realistic SOAR records (the notebook is a .ipynb — use the
NotebookEdit tool, or carefully edit the JSON cell `source` array; do NOT
re-execute the notebook — it is DB-dependent and out of scope, the user will
re-run it).

Cell with `observation_id='demo-603002'` (the `facility='SOAR'` fixture in the
SELECT-03 cell): change its `parameters` dict from `'site': 'coj'` to
`'site': 'sor'`, and from `'instrument_type': '2M0-SCICAM-MUSCAT'` to
`'instrument_type': 'SOAR_GHTS_REDCAM'`.

Cell with `observation_id='demo-604002'` (the `facility='SOAR'` fixture in the
SELECT-04 cell): make the same two changes — `'site': 'sor'` and
`'instrument_type': 'SOAR_GHTS_REDCAM'`.

Do NOT touch the LCO fixtures in those same cells (`demo-603001`, `demo-604001`)
or any other cell — they correctly use `'coj'`/MuSCAT. Leave the surrounding
LCO fixtures, assertions, teardown, and markdown unchanged. The result is that
the next time someone runs the notebook against a live dev DB, the SOAR records'
CalendarEvent titles will read as 'SOAR SOAR_GHTS_REDCAM' (SOAR-identifiable)
instead of the misleading 'FTS 2M0-SCICAM-MUSCAT'.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python -c "import json; nb=json.load(open('docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb')); src='\n'.join(''.join(c['source']) for c in nb['cells']); soar_sor = src.count(\"'site': 'sor'\"); print('soar sor count:', soar_sor); assert soar_sor == 2, f'expected 2 SOAR site=sor fixtures, got {soar_sor}'; assert src.count('SOAR_GHTS_REDCAM') == 2, 'expected 2 SOAR instrument types'; print('notebook SOAR fixtures OK'); import nbformat; nbformat.read('docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb', as_version=4); print('notebook JSON valid')"</automated>
  </verify>
  <done>Both Phase-5 SOAR fixture cells (demo-603002, demo-604002) use 'site': 'sor' and 'instrument_type': 'SOAR_GHTS_REDCAM'; the notebook remains valid JSON / nbformat-readable; no LCO fixture or other cell was modified.</done>
</task>

<task type="auto">
  <name>Task 3: Confirm quality gates clean</name>
  <files>solsys_code/management/commands/sync_lco_observation_calendar.py, solsys_code/tests/test_sync_lco_observation_calendar.py</files>
  <action>
Run the project quality gates to confirm nothing broke. Run `ruff check .` and
`ruff format --check .` from the repo root. If `ruff format --check` reports the
two edited Python files need reformatting, run `ruff format` on them and re-run
the check. The notebook is exempt from ruff (it is a .ipynb and pre-commit
clears its output rather than linting it), so ruff is not expected to touch the
notebook. This task makes no functional change — it only verifies the gates the
constraints require.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; ruff check . &amp;&amp; ruff format --check . &amp;&amp; echo RUFF_CLEAN</automated>
  </verify>
  <done>ruff check . passes and ruff format --check . reports no files need reformatting (prints RUFF_CLEAN).</done>
</task>

</tasks>

<verification>
- `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` passes, with test_select_03/04/05 now exercising site `'sor'`.
- `SITE_TELESCOPE_MAP` contains `'sor'` so `_derive_telescope('sor')` returns a SOAR label instead of raising KeyError (the bug).
- Demo notebook's two SOAR fixtures use `'site': 'sor'` and a SOAR instrument type; notebook is valid JSON.
- `ruff check .` and `ruff format --check .` are clean.
</verification>

<success_criteria>
A SOAR ObservationRecord (`parameters['site'] == 'sor'`) produces a
SOAR-identifiable CalendarEvent rather than being silently skipped; the test and
notebook SOAR fixtures exercise the real SOAR site code; all listed gates pass.
</success_criteria>

<output>
Create `.planning/quick/260619-jpr-fix-sync-lco-observation-calendar-soar-s/260619-jpr-SUMMARY.md` when done.
</output>
