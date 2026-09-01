---
phase: quick
plan: 260617-mlr
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
  - .planning/PROJECT.md
  - .claude/agents/gsd-planner.md
autonomous: true
requirements: []

must_haves:
  truths:
    - "A Stage 3 demo notebook exists at docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb mirroring the Stage 2 notebook's structure"
    - "The notebook demonstrates the full banner->placed->idempotent lifecycle of sync_lco_observation_calendar"
    - "All notebook output cells are empty/cleared (pre-commit convention)"
    - "PROJECT.md's Working code list names this notebook as the Stage 3 demo"
    - "The GSD planner agent has an explicit rule to include a demo-notebook task when a phase adds a user-facing command/module and PROJECT.md documents a Demo Notebooks convention"
  artifacts:
    - path: "docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb"
      provides: "Stage 3 demo of sync_lco_observation_calendar"
      contains: "sync_lco_observation_calendar"
    - path: ".planning/PROJECT.md"
      provides: "Stage 3 demo bullet in Working code list"
      contains: "Stage 3 demo"
    - path: ".claude/agents/gsd-planner.md"
      provides: "Demo-notebook convention rule for future phase planning"
      contains: "Demo Notebooks"
  key_links:
    - from: "sync_lco_observation_calendar_demo.ipynb"
      to: "sync_lco_observation_calendar command"
      via: "call_command('sync_lco_observation_calendar', '--proposal', ...)"
      pattern: "call_command\\(['\"]sync_lco_observation_calendar"
---

<objective>
Backfill the demo notebook that Phase 04 should have shipped, and close the
process gap so future GSD phases don't silently skip the project's Demo
Notebooks convention again.

Purpose: Phase 04 added the `sync_lco_observation_calendar` management command
but — unlike Stages 1 and 2 — shipped no demo notebook, despite PROJECT.md
documenting a per-phase Demo Notebooks convention. This is a doc/process
gap-fill, NOT a behavior change to the command or its tests.

Output:
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` (Stage 3 demo)
- `.planning/PROJECT.md` Working code list updated with the Stage 3 demo entry
- `.claude/agents/gsd-planner.md` gains a demo-notebook planning rule
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
@solsys_code/management/commands/sync_lco_observation_calendar.py
@solsys_code/tests/test_sync_lco_observation_calendar.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create Stage 3 demo notebook and update PROJECT.md</name>
  <files>docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb, .planning/PROJECT.md</files>
  <action>
Create `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
mirroring the structure of `load_telescope_runs_demo.ipynb` EXACTLY: an intro
markdown cell, a Django-setup markdown cell, the Django setup boilerplate code
cell, then alternating markdown-explanation / code cells for the walkthrough,
ending in a summary markdown cell. Match that notebook's nbformat (4),
nbformat_minor, kernelspec/language_info metadata, and per-cell metadata shape
so it is byte-compatible with the repo's other pre_executed notebooks. Give
each cell a stable hex `id` (any unique 8-char hex string is fine).

Do NOT modify `solsys_code/management/commands/sync_lco_observation_calendar.py`
or `solsys_code/tests/test_sync_lco_observation_calendar.py`.

Cells, in order:

1. Intro markdown: title "# LCO Queue Calendar Sync — Stage 3 Demo
   (sync_lco_observation_calendar)". Explain it demonstrates
   `solsys_code/management/commands/sync_lco_observation_calendar.py` (issue #37
   Stage 3) — syncing LCO queue ObservationRecords to CalendarEvents,
   transitioning from a `[QUEUED]` window banner to a placed block as the LCO
   scheduler acts, with no-churn idempotency. List what it demonstrates
   (mirroring the Stage 2 intro's bullet style). Include the same "lives in
   pre_executed/ because it is DB-dependent and NOT run during Sphinx/CI/
   ReadTheDocs builds, per docs/notebooks/README.md" paragraph as the Stage 2
   notebook.

2. Django-setup markdown cell: same explanation text as the Stage 2 notebook's
   "## Django setup" cell (parents[2] is repo root from docs/notebooks/
   pre_executed/; DJANGO_ALLOW_ASYNC_UNSAFE for sync ORM in Jupyter).

3. Django setup boilerplate CODE cell: copy verbatim from the Stage 2 notebook's
   boilerplate cell (the `import os/sys/django`, `Path.cwd().resolve().parents[2]`
   with the `manage.py` existence assert, sys.path insert, the two
   `os.environ.setdefault` lines, `django.setup()`). This matches PROJECT.md's
   "Django setup boilerplate for notebooks" section.

4. Markdown: "## Create a fixture ObservationRecord". Explain that the command
   queries `ObservationRecord(facility='LCO', parameters__proposal=<code>)` and
   that we build one fixture record using the same parameters shape as
   `solsys_code/tests/test_sync_lco_observation_calendar.py` (`proposal`,
   `start`, `end`, `instrument_type`, `site` keys). Note the FK setup (a Target
   via SiderealTargetFactory and a user) mirrors that test's setUpTestData.

5. Code: build the fixture. Import `get_user_model` from
   `django.contrib.auth`, `SiderealTargetFactory` from
   `tom_targets.tests.factories`, and `ObservationRecord` from
   `tom_observations.models`. Use `get_user_model().objects.get_or_create(
   username='sync-demo-user')` and `SiderealTargetFactory.create()` so the cell
   is re-runnable. Create an unscheduled (`scheduled_start=None`,
   `scheduled_end=None`, `status='PENDING'`) ObservationRecord with
   `facility='LCO'`, a chosen `observation_id` (e.g. `'demo-900001'`), and a
   `parameters` dict literally matching the test's `_parameters` output:
   `{'proposal': <DEMO_PROPOSAL>, 'start': '2026-07-01T00:00:00', 'end':
   '2026-07-02T00:00:00', 'instrument_type': '2M0-SCICAM-MUSCAT', 'site':
   'coj'}`. Assign `DEMO_PROPOSAL = 'DEMOCODE'` near the top of the cell. Print
   the created record's observation_id, status, scheduled_start, and parameters.

6. Markdown: "## Sync the queued (unscheduled) record". Explain `call_command`
   is the Django-recommended way to invoke management commands programmatically,
   and that with `scheduled_start=None` the command derives times from
   `parameters['start']`/`['end']` and prefixes the title with `[QUEUED]`
   (SYNC-02). Note the event url is built via
   `LCOFacility().get_observation_url(observation_id)` (SYNC-01).

7. Code: import `io` and `call_command` from `django.core.management`; run
   `call_command('sync_lco_observation_calendar', '--proposal', DEMO_PROPOSAL,
   stdout=stdout_buf, stderr=stderr_buf)` with StringIO buffers; print the
   stdout summary line and any stderr.

8. Markdown: "## Inspect the queued CalendarEvent". Explain the resulting event:
   `[QUEUED]`-prefixed title, times from the parameters window, and the url
   built from the LCO portal.

9. Code: import `CalendarEvent` from `tom_calendar.models` and `LCOFacility`
   from `tom_observations.facilities.lco`. Fetch the event via
   `CalendarEvent.objects.get(url=LCOFacility().get_observation_url('demo-900001'))`;
   print title, start_time/end_time isoformat, url, telescope, instrument,
   proposal, and description. Note in an inline print or markdown that the title
   starts with `[QUEUED]` and url contains `/requests/`.

10. Markdown: "## Simulate the LCO scheduler placing the observation". Explain
    that setting `scheduled_start`/`scheduled_end` on the record and re-running
    the command transitions the event to the clean placed form — times now come
    from the scheduled fields and the `[QUEUED]` prefix drops (SYNC-03).

11. Code: set `record.scheduled_start = datetime(2026, 7, 5, 10, 0, 0,
    tzinfo=dt_timezone.utc)` and `record.scheduled_end = datetime(2026, 7, 5,
    12, 0, 0, tzinfo=dt_timezone.utc)` (import `datetime` and `timezone as
    dt_timezone` from `datetime`), `record.save()`, then re-run the same
    `call_command`. Re-fetch the event and print its title (now no `[QUEUED]`
    prefix) and updated start_time/end_time.

12. Markdown: "## No-churn idempotency". Explain that re-running the command a
    third time with no changes leaves the event's `modified` timestamp unchanged
    and the stdout summary reports `unchanged: 1` (SYNC-04).

13. Code: capture `modified_before = CalendarEvent.objects.get(url=...).modified`,
    run the command once more into a fresh stdout buffer, re-fetch the event,
    assert/print that `modified` equals `modified_before` and print the stdout
    summary line (showing `unchanged: 1`).

14. Summary markdown: a requirements table (SELECT-01, SYNC-01, SYNC-02,
    SYNC-03, SYNC-04, TERM-01) mapped to what the notebook demonstrates,
    mirroring the Stage 2 notebook's closing "## Summary" table and the
    "pre-executed / excluded from automated doc builds" closing paragraph. Note
    the command is equivalent to `./manage.py sync_lco_observation_calendar
    --proposal <code>`.

CRITICAL — clear all outputs: every code cell's `outputs` list MUST be `[]` and
`execution_count` MUST be `null`. The repo's pre-commit hook clears notebook
output; save the notebook with empty outputs so it is already compliant.

After writing the notebook, update `.planning/PROJECT.md`: in the "Working code"
bullet list (currently ending with the two demo-notebook bullets for Stage 1 and
Stage 2 around lines 27-28), add a new bullet immediately after the Stage 2 demo
line:
`- \`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb\`: Stage 3 demo`
Match the exact formatting of the adjacent Stage 1/Stage 2 demo bullets.
  </action>
  <verify>
    <automated>python -c "import json,sys; nb=json.load(open('docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb')); cells=nb['cells']; code=[c for c in cells if c['cell_type']=='code']; assert nb['nbformat']==4, 'nbformat'; assert all(c.get('outputs',[])==[] for c in code), 'non-empty outputs found'; assert all(c.get('execution_count') is None for c in code), 'execution_count not null'; src=''.join(''.join(c['source']) for c in code); assert \"call_command('sync_lco_observation_calendar'\" in src or 'call_command(\"sync_lco_observation_calendar\"' in src, 'missing call_command'; assert 'get_observation_url' in src, 'missing get_observation_url'; assert 'scheduled_start' in src, 'missing scheduled placement'; assert 'django.setup()' in src, 'missing django setup'; print('notebook OK,', len(cells), 'cells')"</automated>
  </verify>
  <done>
The notebook exists, is valid nbformat-4 JSON with all code-cell outputs empty
and execution_count null, contains the Django setup boilerplate, a fixture
ObservationRecord, a `call_command('sync_lco_observation_calendar', ...)`
invocation, the scheduled-placement transition, and a no-churn re-run.
PROJECT.md's Working code list names the notebook as the Stage 3 demo.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add demo-notebook convention rule to the GSD planner agent</name>
  <files>.claude/agents/gsd-planner.md</files>
  <action>
Add an explicit planning rule so future phases don't silently skip a project's
demo-notebook convention. Edit `.claude/agents/gsd-planner.md`, inside the
existing `<project_context>` block (lines 42-50, which already instructs the
planner to discover project context and honor project conventions). Append a new
bullet/subsection after the "Project skills" item, before the closing
`</project_context>` tag:

```
**Project documentation conventions:** After reading `./CLAUDE.md` and
`.planning/PROJECT.md`, check whether PROJECT.md documents a recurring
per-phase documentation convention (e.g. a "Demo Notebooks" section stating
each phase ships a notebook under a fixed directory). If such a convention
exists AND this phase adds a new user-facing command, module, or feature, the
plan MUST include a task that creates or updates the corresponding artifact
(e.g. a demo notebook under `docs/notebooks/pre_executed/`). Reference the
convention text from PROJECT.md rather than reinventing it per-project — copy
its required structure (intro, setup boilerplate, walkthrough, cleared outputs)
into the task action so the executor matches the project's existing examples.
```

Do not change any other section of the file. Preserve the existing
`@/home/tlister/...project-skills-discovery.md` reference and the surrounding
markdown formatting.
  </action>
  <verify>
    <automated>grep -q "Project documentation conventions" .claude/agents/gsd-planner.md && grep -q "Demo Notebooks" .claude/agents/gsd-planner.md && python -c "t=open('.claude/agents/gsd-planner.md').read(); i=t.index('<project_context>'); j=t.index('</project_context>'); seg=t[i:j]; assert 'Project documentation conventions' in seg, 'rule not inside project_context block'; print('rule placement OK')"</automated>
  </verify>
  <done>
`.claude/agents/gsd-planner.md` contains a "Project documentation conventions"
rule inside the `<project_context>` block that instructs planners to add a
demo-notebook (or equivalent) task whenever PROJECT.md documents such a
convention and the phase adds a user-facing command/module/feature.
  </done>
</task>

</tasks>

<verification>
- Notebook is valid nbformat-4 JSON, structurally mirrors load_telescope_runs_demo.ipynb, all code-cell outputs cleared.
- Notebook walkthrough covers: fixture record -> queued sync ([QUEUED] + parameters window times + get_observation_url) -> scheduled placement (clean title, scheduled times) -> no-churn re-run.
- PROJECT.md Working code list includes the Stage 3 demo bullet.
- gsd-planner.md carries the demo-notebook convention rule inside <project_context>.
- No changes to sync_lco_observation_calendar.py or its test file.
</verification>

<success_criteria>
- [ ] `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` exists, valid, outputs cleared
- [ ] Notebook demonstrates the banner->placed->idempotent lifecycle via call_command
- [ ] PROJECT.md names the notebook as the Stage 3 demo
- [ ] gsd-planner.md has the demo-notebook convention rule
- [ ] sync_lco_observation_calendar.py and test_sync_lco_observation_calendar.py untouched
</success_criteria>

<output>
Create `.planning/quick/260617-mlr-backfill-phase-04-s-missing-demo-noteboo/260617-mlr-SUMMARY.md` when done
</output>
