---
phase: "34"
slug: "the-observation-projector-trigger"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-10"
---

# Phase 34 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` (`django.test.TestCase`), run via `python manage.py test` — the only functioning suite in this repo (CLAUDE.md "Testing") |
| **Config file** | none — Django's own runner. The `pytest` config in `pyproject.toml` is LINCC-template legacy and does not collect these tests |
| **Quick run command** | `python manage.py test solsys_code.tests.test_observation_projector solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_project_observation_calendar` |
| **Full suite command** | `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py \| grep -v "tests/test_views\.py$" \| sed "s\|/\|.\|g; s\|\.py$\|\|" \| tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` (`.planning/config.json` `workflow.test_command`) |
| **Estimated runtime** | ~4 s for a single new module (measured: `test_calendar_display_extras`, 53 tests, 3.5 s wall including interpreter and test-DB setup); full suite measured at 2 m 9 s wall (2026-09-10, this branch) |

**Two runner constraints that shape every command below** (from prior sessions, not assumptions):
`python manage.py`, never `./manage.py`; and never a bare `python manage.py test solsys_code` —
`solsys_code.tests.test_views.TestEphemeris` segfaults in native ASSIST, which is why the full-suite
command enumerates labels and excludes `test_views.py`. Importing `solsys_code.views` or
`solsys_code.ephem_utils` triggers a ~1.6 GB SPICE kernel download on first use; no test module in
this phase imports either, and the calendar page is always reached through the Django test client.

---

## Sampling Rate

- **After every task commit:** the quick run command above, scoped to the modules that task touched
- **After every plan wave:** the full suite command, plus `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files`
- **Before `/gsd-verify-work`:** full suite green and both ruff hooks clean
- **Max feedback latency:** under 10 s for a single-module run; the full suite is a per-wave gate, not a per-task one

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 34-01-01 | 01 | 1 | PROJ-01, PROJ-02, TRIG-01 | T-34-01 / T-34-03 | A record save reaches no network and writes only inside the facility-URL key namespace | integration | `python manage.py test solsys_code.tests.test_observation_projector_signals` | ✅ created by this task | ✅ green |
| 34-01-01 | 01 | 1 | TRIG-02 | T-34-01 | The projector module imports neither the portal-block resolver nor the HTTP request helper | source assertion | `grep -c 'resolve_placement_block\|make_request' solsys_code/observation_projector.py` | ✅ | ✅ green |
| 34-01-01 | 01 | 1 | PROJ-04 (title stem) | T-34-04 | No human-confirmation stamp field is referenced by the projector | source assertion | `python -c "src=open('solsys_code/observation_projector.py').read(); print(sum(src.count(t) for t in ('confirmed_by','confirmed_at')))"` | ✅ | ✅ green |
| 34-01-02 | 01 | 1 | PROJ-01, PROJ-02, PROJ-03, PROJ-05, PROJ-06 | T-34-03 / T-34-04 | `RUN:`, `GEM:` and blank-url events are byte-identical across a projection | unit | `python manage.py test solsys_code.tests.test_observation_projector` | ✅ created by this task | ✅ green |
| 34-01-03 | 01 | 1 | TRIG-01, TRIG-02 | T-34-02 / T-34-03 | A raising projector never aborts a save, a membership change or a delete; a `RUN:` event survives a record delete | integration | `python manage.py test solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_observation_projector` | ✅ | ✅ green |
| 34-01-03 | 01 | 1 | TRIG-01 | — | All three receivers are actually connected in `apps.ready()` | source assertion | `grep -c "dispatch_uid='solsys_code.observation_projector" solsys_code/apps.py` | ✅ | ✅ green |
| 34-01-03 | 01 | 1 | TRIG-01, TRIG-02 | T-34-01 | The whole existing suite still passes with three global receivers live | regression | full suite command — **per-wave gate** (`<automated gate="wave">` in 34-01 Task 3), run once at the end of wave 1, not in the per-task loop; the per-task loop runs the scoped module commands in the rows above | ✅ | ✅ green |
| 34-02-01 | 02 | 2 | TRIG-03, PROJ-05 | T-34-10 / T-34-11 | `--proposal` is an exact-code ORM filter; a sweep touches no foreign key namespace | integration | `python manage.py test solsys_code.tests.test_project_observation_calendar` | ✅ created by this task | ✅ green |
| 34-02-01 | 02 | 2 | TRIG-03 | — | No sweep argument is required, so Phase 36's cron call works | source assertion | `python -c "import re;src=open('solsys_code/management/commands/project_observation_calendar.py').read();print(len(re.findall(r'required\s*=\s*True',src)))"` | ✅ | ✅ green |
| 34-02-02 | 02 | 2 | ANNOT-03 | T-34-12 | The retired command is unreachable — not merely unused | CLI output | `python -c "import django,os;os.environ.setdefault('DJANGO_SETTINGS_MODULE','src.fomo.settings');django.setup();from django.core.management import get_commands;print('sync_lco_observation_calendar' in get_commands())"` | ✅ | ✅ green |
| 34-02-02 | 02 | 2 | ANNOT-03 | T-34-12 | No Python code references the deleted module | source assertion | `grep -rn 'sync_lco_observation_calendar' --include='*.py' solsys_code/ src/ \| wc -l` | ✅ | ✅ green |
| 34-02-03 | 02 | 2 | PROJ-01, PROJ-05 | T-34-07 / T-34-08 / T-34-09 | One portal call per record ever; a failed lookup keeps the coarse token and retries | integration | `python manage.py test solsys_code.tests.test_project_observation_calendar solsys_code.tests.test_observation_projector solsys_code.tests.test_calendar_utils solsys_code.tests.test_campaign_attribution` | ✅ | ✅ green |
| 34-02-03 | 02 | 2 | PROJ-01 | T-34-09 | The renamed labels resolve exactly as specified | CLI output | `python -c "... print(m[('ogg','2m0')], m[('coj','2m0')], m[('sor','4m0')], m[('lsc','1m0')], sorted(o))"` (full form in 34-02 Task 3) | ✅ | ✅ green |
| 34-02-03 | 02 | 2 | ANNOT-03 | — | The label rename breaks no module that reads telescope strings | regression | full suite command — **per-wave gate** (`<automated gate="wave">` in 34-02 Task 3), run once at the end of wave 2 alongside 34-03 Task 2's identical gate, not in the per-task loop | ✅ | ✅ green |
| 34-03-01 | 03 | 2 | PROJ-03, PROJ-06 | T-34-17 / T-34-18 | Every new marker rings correctly and no existing ring is lost | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | ✅ extended by this task | ✅ green |
| 34-03-01 | 03 | 2 | PROJ-03 | T-34-18 | The full thirteen-title ring vector, asserted in one command | CLI output | `python -c "... print([s(t)!='' for t in (…13 titles…)])"` (full form in 34-03 Task 1) | ✅ | ✅ green |
| 34-03-01 | 03 | 2 | PROJ-06 | — | The month-cell truncation budget is unchanged | source assertion | `grep -c 'truncatechars:18' src/templates/tom_calendar/partials/calendar.html` | ✅ | ✅ green |
| 34-03-02 | 03 | 2 | PROJ-04 (title stem), PROJ-05 | T-34-13 / T-34-14 / T-34-16 | The series tag reads only and exposes no PII field | unit + integration | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_event_meta_links` | ✅ | ✅ green |
| 34-03-02 | 03 | 2 | PROJ-05 | T-34-16 | No write method appears in the display-time tag's body | source assertion | `python -c "... print(sum(body.count(t) for t in ('.save(','.update(','.create(','get_or_create(')))"` (full form in 34-03 Task 2) | ✅ | ✅ green |
| 34-03-02 | 03 | 2 | PROJ-05 | T-34-15 | The month view gains no query per grouped event | integration (`assertNumQueries`) | `python manage.py test solsys_code.tests.test_calendar_template` | ✅ | ✅ green |
| 34-03-02 | 03 | 2 | PROJ-03, PROJ-05 | — | No template or view change breaks an unrelated calendar test | regression | full suite command — **per-wave gate** (`<automated gate="wave">` in 34-03 Task 2), run once at the end of wave 2 | ✅ | ✅ green |
| 34-04-01 | 04 | 3 | TRIG-03, SCHED-06 | T-34-19 / T-34-22 | Nothing secret reaches a committed output cell; the SCHED-06 baseline is recorded, not claimed | CLI output + artifact assertion | `python -c "import json;nb=json.load(open('docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb'));print(sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs')))"` | ✅ created by this task | ✅ green |
| 34-04-01 | 04 | 3 | ANNOT-03, PROJ-01 | — | `campaign_attribution.py`'s paired notebook demonstrates the renamed-label resolution branch and keeps its executed output | artifact assertion | the two `campaign_lifecycle_demo.ipynb` probes in 34-04 Task 1 (label-count and output-cell-count) | ✅ | ✅ green |
| 34-04-01 | 04 | 3 | ANNOT-03 | T-34-20 | Both registries point at the new notebook and neither names the retired command | source assertion | `grep -c 'sync_lco_observation_calendar' docs/notebooks.rst CLAUDE.md` and `grep -c 'project_observation_calendar_demo' docs/notebooks.rst CLAUDE.md` | ✅ | ✅ green |
| 34-04-02 | 04 | 3 | ANNOT-03 | T-34-20 / T-34-23 | The runbook documents only runnable commands; the Gemini notebook keeps its committed output | source assertion + CLI output | `grep -rc 'sync_lco_observation_calendar' docs/runbooks/telescope_runs_calendar.rst`, `grep -c 'project_observation_calendar' docs/runbooks/telescope_runs_calendar.rst`, and the Gemini-notebook markdown/output probes in 34-04 Task 2 | ✅ | ✅ green |
| 34-04-02 | 04 | 3 | ANNOT-03 | — | The docs build resolves every toctree entry | CLI output | `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` | ✅ | ✅ green |
| 34-05-01 | 05 | G-34-2 | PROJ-02, TRIG-01 | T-34-01 | Portal ISO strings on an in-memory record coerce to aware UTC on the post_save path, so a real `updatestatus` save projects without `unprojectable` | unit + integration (TDD) | `python manage.py test solsys_code.tests.test_calendar_utils solsys_code.tests.test_observation_projector_signals` (`TestCoerceScheduleDatetime`, `TestRecordTimeWindow`, `TestUpdateObservationStatusPath`) | ✅ extended by this task | ✅ green |
| 34-05-02 | 05 | G-34-2 | TRIG-02 | T-34-02 | The receiver still never raises out of a save; `observation_projector.py` untouched by the fix | source assertion | `git diff 7877a2e -- solsys_code/observation_projector.py` (empty) | ✅ | ✅ green |
| 34-06-01 | 06 | G-34-2 | PROJ-02, SCHED-06 | T-34-19 | Paired notebook re-executed against a scratch copy shows the string-schedule save narrowing; baseline JSON and `src/fomo_db.sqlite3` untouched | artifact assertion | `python manage.py test solsys_code.tests.test_projector_demo_notebook`; `git status --porcelain docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` (empty) | ✅ | ✅ green |
| 34-06-02 | 06 | G-34-2 | TRIG-01, TRIG-02 | T-34-01 | A real portal-backed `updatestatus` over a copy logs zero `unprojectable`; the following dry-run sweep reports `updated: 0` | CLI output (live, gitignored evidence `tmp/34-06-updatestatus.txt`, `tmp/34-06-dry-run.txt`) | `grep -c unprojectable tmp/34-06-updatestatus.txt` (0) | ✅ | ✅ green |
| 34-07-01..03 | 07 | G-34-3 | PROJ-05, TRIG-03, ANNOT-03 | T-34-19 | The committed notebook's takeover evidence is non-vacuous and a repo-level test fails when it goes empty | unit (TDD) + artifact assertion | `python manage.py test solsys_code.tests.test_projector_demo_notebook` (6 tests: 5 pass, 1 skip); `FOMO_DEMO_NOTEBOOK_PATH=<emptied copy>` variant fails on the intended assertion | ✅ created by this task | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Every runnable `<automated>` command in the four plans is paired with a `<fails_when>` statement
naming an observable failure signal (12 pairs in 34-01, 14 in 34-02, 10 in 34-03, 17 in 34-04).
Two of those pairs per plan in 34-01 Task 3, 34-02 Task 3 and 34-03 Task 2 carry `gate="wave"` — the
full-suite run and the two ruff hooks. They match the per-wave sampling rate above rather than the
per-task one, because the full suite is measured at 2 m 9 s wall and would otherwise put the
per-task feedback loop far past its budget.
No task in this phase carries a `MISSING — Wave 0` sentinel.

---

## Wave 0 Requirements

**No separate Wave 0 is needed.** Each new test module is created by the same task whose
`<automated>` command first runs it, so no task's verification references a file that does not yet
exist at the moment it runs:

- `solsys_code/tests/test_observation_projector_signals.py` — created by 34-01 Task 1
- `solsys_code/tests/test_observation_projector.py` — created by 34-01 Task 2
- `solsys_code/tests/test_project_observation_calendar.py` — created by 34-02 Task 1
- `solsys_code/tests/test_calendar_display_extras.py`, `test_calendar_template.py`,
  `test_calendar_utils.py`, `test_campaign_attribution.py` — already exist; extended in place
- Framework install: none — Django's `TestCase` is already this project's test framework

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| A real `KEY2026B-004` record's event narrows queued → scheduled → observed over live nights with nobody running anything | SCHED-06 | It is a claim about the passage of real time and real LCO scheduler decisions; no fixture can produce it, which is exactly why spike 004's verdict was PARTIAL | Baseline is captured and committed by 34-04 Task 1. Over the following nights run **only** `python manage.py updatestatus` — specifically not the sweep. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` with `jupyter nbconvert --to notebook --execute --inplace` and commit it. Record the dates in `34-UAT.md`. Phase verification passes on the baseline plus the mechanism tests; the re-execution is a follow-up commit, not a gate on Phase 35 planning. **Resolved 2026-09-12:** closed in `34-UAT.md` Test 3 — records 4378332 (11P) and 4378046 (10P) narrowed queued → observed via the receiver alone after one real `updatestatus`; notebook re-execution tracked as a paired-docs follow-up |
| A month cell reads correctly at a glance — marker and telescope visible before truncation, terminal ring legible, legend readable | PROJ-06, PROJ-03 | Legibility inside a fixed-width month cell is a visual judgement; the automated tests assert the character budget and the ring value, not whether a person can read it | Open the calendar for a month carrying real LCO nights (`workflow.human_verify_mode` is `end-of-phase`, so this is the `<human-check>` in 34-04) and confirm the titles, rings, series modal and legend by eye. **Resolved 2026-09-11:** passed as Test 1 of the first `34-UAT.md` session (commit 84e8936) |
| Nothing secret reached a committed notebook output cell | T-34-19 | An automated grep can catch known key names but not an unanticipated secret in a pasted response | Read the executed notebook's output cells before committing; confirm no API key, authorization header, raw portal response body or submitter contact detail appears. **Resolved:** performed at each of the 34-04, 34-06 and 34-07 notebook commits (each SUMMARY records the check) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task in all four plans carries at least one)
- [x] Wave 0 covers all MISSING references (there are none — each test module is created by the task that runs it)
- [x] No watch-mode flags
- [x] Feedback latency < 10 s for a single-module run
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-09-12 — all 11 requirements (PROJ-01..06, TRIG-01..03, ANNOT-03, SCHED-06) COVERED; SCHED-06's manual-only item closed by UAT

---

## Validation Audit 2026-09-12

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Audit run: 342 tests across `test_observation_projector`, `test_observation_projector_signals`,
`test_project_observation_calendar`, `test_calendar_display_extras`, `test_calendar_template`,
`test_calendar_utils`, `test_campaign_attribution`, `test_calendar_event_meta_links`,
`test_projector_demo_notebook` — OK (1 skip, by design). Every source-assertion command in the map
returned its expected value (0/0/3/0/0/0/1, notebook output cells: 12). Gap-closure plans 34-05..34-07
added to the map; all three manual-only rows now carry a resolution.
