---
phase: 260929-svk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [DISCOVER-01, PROJ-05, SCHED-08]

estimate:
  tokens: 85000
  raw_tokens: 85000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "An existing LCO `ObservationRecord` whose `parameters` hold `observed_site`/`observed_telescope`/`observed_enclosure` still holds all three keys, with the same values, after `backfill_lco_observations` re-sweeps it over unchanged portal data. The run counts it `unchanged` (real run) and leaves `would update` at 0 (dry run). The row is not saved at all (its `modified` timestamp does not move), so the post_save trigger never re-draws its event with the coarse telescope token."
    - "When the portal really has changed (a moved request window, a new status or a new observed block), discovery still updates the record. `parameters['start']`/`parameters['end']` take the portal's new values, the run counts it `updated` (dry run: `would update`), and the three observed-site keys survive that update too."
    - "A key the sweep stored as `None` (for example `observed_enclosure` when the portal block has no enclosure) is carried forward as `None`. Carry-forward tests key presence, never truthiness. Otherwise the two dicts would still differ and the churn would carry on for exactly those records."
    - "A record that has none of the three keys compares exactly as before this change. Every pre-existing test in `solsys_code/tests/test_backfill_lco_observations.py` passes unmodified."
    - "Dry-run and real-run updated/unchanged decisions cannot drift apart (T-ik7-02). The carry-forward runs inside `_changed_record_fields()`, the one comparison both branches of `sweep_proposal()` already call."
    - "The paired notebook shows the carry-forward with real executed output. It runs against a fresh, throwaway scratch database that it builds and migrates itself, never against `src/fomo_db.sqlite3`, which is this host's live database and carries an active `KEY2026B-004` watched proposal."
    - "The runbook's 'Re-running updates in place instead of skipping.' bullet says that the three observed-site keys the sweep stores are carried forward, never erased."
  artifacts:
    - "solsys_code/management/commands/backfill_lco_observations.py: new pure helper `_preserve_observed_site_keys(existing, rebuilt) -> dict`, called at the top of `_changed_record_fields()`. `OBSERVED_SITE_PARAMETER_KEYS` is imported from `solsys_code.calendar_utils`."
    - "solsys_code/tests/test_backfill_lco_observations.py: new `TestObservedSiteKeysSurviveDiscovery` (a `TestCase`: real-run unchanged, real-run moved window, dry-run unchanged, dry-run moved window, None-valued enclosure) and new `TestPreserveObservedSiteKeys` (a `SimpleTestCase` exercising the pure helper)."
    - "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb: setup cell routed to a fresh scratch database; a new markdown cell plus two code cells after the second-real-pass cell; a scratch-teardown pair at the end; intro and cleanup prose updated. Re-executed top to bottom."
    - "docs/runbooks/telescope_runs_calendar.rst: one added sentence in the `backfill_lco_observations` 'Re-running updates in place' bullet."
  key_links:
    - "`project_observation_calendar.resolve_observed_site()` (the only writer of the three keys) and `backfill_lco_observations._preserve_observed_site_keys()` (the reader that carries them forward) share one contract, `calendar_utils.OBSERVED_SITE_PARAMETER_KEYS`. Both must name the same three keys."
    - "`sweep_proposal()` dry-run branch and real-run branch -> `_changed_record_fields()` -> `_preserve_observed_site_keys()`. The merged dict is also the value `changes['parameters']` returns, so the real-run `setattr` writes the keys back rather than dropping them."
    - "Notebook setup cell: `FOMO_DATABASE_PATH` is assigned before `django.setup()` and read by `src/fomo/settings.py:134`. The cell asserts the resolved `DATABASES['default']['NAME']` is the scratch path before any ORM call."
---

<objective>
Stop the discovery/sweep churn loop that intent-review finding F1 describes (`.planning/v2.4-INTENT-REVIEW.md`, "### F1"). `backfill_lco_observations` (discovery) must stop erasing the three observed-site keys that the projector sweep's one-time lookup stores in `ObservationRecord.parameters`. It must still refresh the portal-owned keys (`proposal`, `instrument_type`, `start`, `end`) whenever the portal changes them.

Purpose: on the live host, every 15-minute tick currently sees discovery report `updated: 133`. Each of those updates overwrites the keys away, and the post_save trigger re-draws each event with the coarse token (`[O] 1m0 248370`). The next tick's sweep then repeats 133 live portal lookups (`site_lookups: 133`) and restores the keys, and discovery erases them again about 30 s later. Tick times grew from 18 s to 431 s on a `flock -n` 15-minute schedule. Event titles are unstable. `campaign_gap.py:160` sees `observed_site` only half the time. This also breaks PROJ-05's no-churn guarantee.

Output: a helper in the discovery command and tests for it; the paired notebook (CLAUDE.md "Paired docs" rule: `backfill_lco_observations.py` -> `backfill_lco_observations_demo.ipynb`), re-executed; one runbook sentence. The projector/sweep side (`project_observation_calendar.py`, `calendar_utils.py`, `observation_projector.py`) is NOT touched.

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **The cron runs from THIS checkout.** `crontab -l` shows `*/15 * * * * /usr/bin/flock -n ... /home/tlister/venv/devel_fomo311_venv/bin/python /home/tlister/git/fomo_devel/manage.py run_unattended`. Any edit you save to `solsys_code/management/commands/backfill_lco_observations.py` is imported by the next tick's process. So every intermediate save of that file must leave it importable and correct. That is why Task 1 writes the tests first (the test file is never imported by the runner) and splits the production edit into two steps that are each safe on their own. The fix therefore goes live on the first tick after Task 1. That is intended: it is how F1 gets confirmed (Task 3's human check).
2. **The paired notebook currently runs against the live database.** Its setup cell (cell index 2, id `2659707b`) calls `django.setup()` with the default `src.fomo.settings`, and nothing redirects `FOMO_DATABASE_PATH`. So it resolves to `src/fomo_db.sqlite3`, the production database this cron writes to. `src/fomo/local_settings.py` does not override `DATABASES`. Worse, a read-only check at planning time showed the live database holds an active `WatchedProposal` `KEY2026B-004`. The notebook's bare-invocation cell (index 16, id `6a1f0c12`) sweeps every active watched row through a `make_request` mock whose payload table only knows the two demo codes. `KEY2026B-004` would hit a `KeyError`, get `last_run_summary = 'failed: KeyError'` written onto the live row, and raise `CommandError`, which aborts the notebook run. The orchestrator's assumption that the notebook "uses its own in-memory/mocked setup" is therefore false. Task 2 adds scratch-database routing, following `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` cell index 2 (id `9084663a`), BEFORE the notebook is ever executed. One deliberate difference: that notebook copies the developer database, and this one must migrate a NEW EMPTY file instead. A copy would bring along the live `KEY2026B-004` row and break cell 16 exactly as above. Copying a SQLite file while a cron job may be writing it can also produce a torn copy. And this demo creates every row it needs itself.
3. **`save(update_fields=['parameters'])` does not write `modified`.** That is how the sweep writes the keys, and the tests and notebook copy that write exactly. `auto_now` only changes the in-memory attribute when `modified` is not in `update_fields`. So always call `refresh_from_db()` after that write and BEFORE capturing a "before" `modified` value. Otherwise a no-churn assertion compares against a timestamp that was never stored.
4. **Runbook check.** The runbook bullet at `docs/runbooks/telescope_runs_calendar.rst` (about lines 365-369) says an existing record has its "`parameters` refreshed from the portal". That reads as a whole-dict replacement, which this change contradicts. So the runbook IS in scope, for one sentence only.
5. **Only discovery rewrites `parameters` wholesale.** A grep of non-test `solsys_code/` found exactly two writers: discovery's `get_or_create` defaults/`setattr` path, and the sweep's `resolve_observed_site()`. The status refresh step does not touch `parameters`. So a fix in this one file closes the loop.

Source coverage: GOAL (F1 fixed, no churn) is covered by Tasks 1 and 3. DISCOVER-01, PROJ-05 and SCHED-08 are covered by Task 1 (behavior) and Task 3 (live confirmation). Orchestrator fix items: the helper and the single call site are Task 1; tests (a), (b) and (c) are Task 1; the notebook is Task 2; the runbook-if-contradicted item is Task 2; the quality gates are Task 3. No item is unplanned.
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
From `solsys_code/calendar_utils.py:95` (import it; do not redefine it):
- `OBSERVED_SITE_PARAMETER_KEYS = ('observed_site', 'observed_telescope', 'observed_enclosure')`. `calendar_utils` imports only stdlib, `requests`, django, `tom_calendar`/`tom_observations`/`tom_common` and two small `solsys_code` modules. It has no SPICE/ephemeris import side effect, so importing it from the discovery command is cheap.

From `solsys_code/management/commands/project_observation_calendar.py:43-104`, `resolve_observed_site(record, facility)`:
- Its guard is `if record.parameters.get(site_key): return None, None` (so a surviving `observed_site` means no further portal lookup).
- It sets `record.parameters[...] = site/telescope/enclosure`. `enclosure` may be `None`, because it comes from `block.get('enclosure')`.
- It then calls `record.save(update_fields=['parameters'])`.

From `solsys_code/management/commands/backfill_lco_observations.py`:
- `_build_parameters(request_group, request) -> dict | None` (line 255). Builds `{'proposal', 'instrument_type', 'start'?, 'end'?}` and never emits any observed-site key.
- `_changed_record_fields(record, status, scheduled_start, scheduled_end, parameters, compare_schedule=True) -> dict` (line 353). Its last check is `if record.parameters != parameters: changes['parameters'] = parameters`. Its docstring already says it is "The single comparison the write branch and the dry-run branch both call ... (T-ik7-02)".
- `sweep_proposal(...)` dry-run branch (about lines 556-575). Calls `_changed_record_fields(existing_record, status, scheduled_start, scheduled_end, parameters, compare_schedule=embedded)`.
- `sweep_proposal(...)` real-run branch (about lines 605-626). Calls `ObservationRecord.objects.get_or_create(..., defaults={... 'parameters': parameters ...})`. When the record already existed it does `changes = _changed_record_fields(record, status, scheduled_start, scheduled_end, parameters)`, then `setattr` for each change, then `record.save()`, and then `updated += 1` or `unchanged += 1`.
- The existing import block ends with `from solsys_code.models import WatchedProposal`.

From `solsys_code/tests/test_backfill_lco_observations.py`:
- Module helpers: `_request(request_id, target_name='Didymos', state='COMPLETED', target_type=..., elements=None, observations=None, windows=None)`. Its default window is `[{'start': '2026-07-01T00:00:00', 'end': '2026-07-02T00:00:00'}]`.
- `_request_group(group_id, name, proposal='LCO2026A-003', requests=None, created=...)` and `_page_response(results, next_url=None)`.
- `_expected_summary(dry_run, requestgroups_seen, created, updated, unchanged, skipped, targets, groups_created, groups_reused, embedded_blocks, fallback_lookups_needed, block_lookups_failed, list_name='LCO2026A-003_targets', list_reused=False, targets_added=0)` builds the exact summary line.
- `TestBackfillLcoObservations` gets its fixture from `setUpTestData` (`NonSiderealTargetFactory.create(name='Didymos')`). Its `setUp` patches `tom_observations.facilities.lco.LCOFacility.get_observation_status` with a default COMPLETED return value, and each test patches `solsys_code.management.commands.backfill_lco_observations.make_request`.
- Reference expectation for a second pass over a single-request, fallback-path payload for the pre-existing `Didymos` target (see `test_dry_run_would_update_when_status_differs`, about line 378): `requestgroups_seen=1, created=0, skipped=0, targets=0, groups_created=0, groups_reused=0, embedded_blocks=0, fallback_lookups_needed=1, block_lookups_failed=0, list_reused=True, targets_added=1`.
- Private helpers are imported directly by tests elsewhere in this repo (for example `test_project_observation_calendar.py:25` imports `_parse_proposal_arg`), so importing `_preserve_observed_site_keys` follows existing convention.

Notebook `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (nbformat 4.5, 25 cells, cell ids stable):
- index 0 `ae33103d` md intro (says it "creates real ... rows in the local dev database")
- index 1 `86dd4f0a` md "Django setup"
- index 2 `2659707b` code setup; no DB routing
- index 4 `a0c6306e` code: `DEMO_PROPOSAL`, `_request(request_id, state)` with a fixed window `2026-07-01T00:00:00`-`2026-07-02T00:00:00`, `_request_group(requests)`
- index 6 `0a7d2d6b` code: `_page_response`, `_status_lookup(schedule_table)`, `FIRST_PASS_SCHEDULE`
- index 12 `dcb271ab` code: defines `record_101`, `record_102`
- index 14 `33d8ee67` code: second real pass, defining `second_pass_request_group` and `SECOND_PASS_SCHEDULE`. Afterwards both 900101 and 900102 are COMPLETED and its output shows `updated: 1, unchanged: 1`.
- index 15 `6a1f0c11` md watched-list section
- index 23 `1c6c298c` md "Cleanup" (says "creates real rows in the local dev database")
- index 24 `ce8337f7` code cleanup (already deletes records 900101/900102)
- Code cells 2 and 6 currently print nothing. `pre_executed/` is excluded from the `jupyter-nb-clear-output` hook, but the `ruff-format` hook DOES format notebooks (`types_or: [python, pyi, jupyter]`).
</interfaces>
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Discovery carries the sweep's observed-site keys forward inside the shared comparison, proven end-to-end through the command</name>
  <files>solsys_code/management/commands/backfill_lco_observations.py, solsys_code/tests/test_backfill_lco_observations.py</files>
  <read_first>
    - solsys_code/management/commands/backfill_lco_observations.py, lines 1-35 (imports and module docstring), 255-285 (`_build_parameters`), 353-395 (`_changed_record_fields`, including its T-ik7-02 docstring) and 540-630 (both `sweep_proposal` branches that call it)
    - solsys_code/management/commands/project_observation_calendar.py, lines 43-104 (`resolve_observed_site`: the writer whose keys must survive, and its `update_fields=['parameters']` save)
    - solsys_code/calendar_utils.py, lines 87-95 (the `OBSERVED_SITE_PARAMETER_KEYS` reservation and its D-09 comment)
    - solsys_code/tests/test_backfill_lco_observations.py, lines 1-150 (helpers, `_expected_summary`, the main class's `setUpTestData`/`setUp`) and 376-480 (the dry-run unchanged/would-update tests to mirror)
  </read_first>
  <behavior>
    - Test A (tracer; must fail before the fix with `updated: 1` and the keys gone): a real run creates record 10. The test then tags it the way the sweep does and calls `refresh_from_db()` before reading `modified`. A second real run over the identical payload prints exactly `_expected_summary(dry_run=False, ..., created=0, updated=0, unchanged=1, ..., fallback_lookups_needed=1, list_reused=True, targets_added=1)`. The reloaded record still has `observed_site`/`observed_telescope`/`observed_enclosure` with the tagged values and still has `proposal`/`instrument_type`/`start`/`end`, and `modified` has not moved.
    - Test B: same setup, but the second real run's request carries `windows=[{'start': '2026-07-03T00:00:00', 'end': '2026-07-04T00:00:00'}]`. The summary is exactly `updated=1, unchanged=0`. The reloaded `parameters['start']` is `'2026-07-03T00:00:00'` and `['end']` is `'2026-07-04T00:00:00'`, and all three observed-site keys still hold their tagged values.
    - Test C: tagged record, then a `--dry-run` over the identical payload. The summary is exactly `_expected_summary(dry_run=True, ..., updated=0, unchanged=1, ...)`, and the record is untouched: keys present and `modified` unchanged.
    - Test D: tagged record, then a `--dry-run` with the moved window from Test B. The summary shows `would update: 1` (exact `_expected_summary(dry_run=True, ..., updated=1, unchanged=0, ...)`), and the stored `start` is still the old `'2026-07-01T00:00:00'` with the keys intact.
    - Test E: tagged with `observed_enclosure=None`, then a second real run over identical data. The result is `unchanged: 1`, `'observed_enclosure' in parameters`, and its value is `None`.
    - Helper tests (a `SimpleTestCase`, no DB): (1) all three keys present on `existing`, one of them `None`, are copied onto the result, and portal-owned keys come from `rebuilt` (an old `start` on `existing` loses to a new `start` on `rebuilt`); (2) an `existing` with none of the keys gives a result equal to `rebuilt` that is a new object, not `rebuilt` itself; (3) a partial `existing` (only `observed_site`) copies only that key and adds no absent key; (4) neither argument is mutated (compare with `copy.deepcopy` snapshots); (5) a non-dict `existing` (`None`) gives a result equal to `rebuilt`.
  </behavior>
  <action>
    RED first, with only the test file touched. The cron runner never imports the test file, so the live host is unaffected while you do this.

    Add two classes at the end of `solsys_code/tests/test_backfill_lco_observations.py`. Spell the three keys literally (`'observed_site'`, `'observed_telescope'`, `'observed_enclosure'`) in both classes, in the same spirit as `_expected_summary` staying independent of the command module. The literal names ARE the contract with the sweep, so a rename of the constant must fail these tests.

    (1) `TestObservedSiteKeysSurviveDiscovery(TestCase)`. Its `setUpTestData` creates the `Didymos` target with `NonSiderealTargetFactory.create(name='Didymos')` (CLAUDE.md rule: never `SiderealTargetFactory`). Its `setUp` copies the main class's class-level `get_observation_status` patch and its default COMPLETED return value, with `addCleanup`. Give it a small `_tag_as_swept(self, observation_id='10', site='elp', telescope='1m0a', enclosure='doma')` helper. The helper loads the record, updates `record.parameters` with the three keys, and calls `record.save(update_fields=['parameters'])`, which is exactly the write `resolve_observed_site()` makes. It then calls `record.refresh_from_db()` and returns the record (planning finding 3: the `modified` value you capture must come from the database). Implement Tests A-E from `<behavior>` as methods, each patching `make_request` the way the existing tests do. Use exact-line `_expected_summary(...)` assertions with the reference counters from `<interfaces>`.

    (2) `TestPreserveObservedSiteKeys(SimpleTestCase)`, covering helper tests (1)-(5). Import `SimpleTestCase` from `django.test` and `copy` from stdlib. Import `_preserve_observed_site_keys` next to `sweep_proposal` in the existing import from the command module.

    Run `python manage.py test solsys_code.tests.test_backfill_lco_observations.TestObservedSiteKeysSurviveDiscovery` and confirm Tests A, B and E fail for the F1 reason (keys gone, or `updated` where `unchanged` was expected). The whole module will first fail with an ImportError for the helper; that is the expected red. Commit the failing tests: `test(260929-svk): pin observed-site keys surviving discovery (F1)`.

    GREEN, in two production edits. Each edit must leave the module importable and behaving correctly on its own, because the next cron tick imports whatever is on disk (planning finding 1). Make each edit in one tool call, and run the import smoke check from `<verify>` right after each.

    Edit 1. Add `from solsys_code.calendar_utils import OBSERVED_SITE_PARAMETER_KEYS` directly above the existing `from solsys_code.models import WatchedProposal`, with a short comment above it. The comment should say this is the parameters-key reservation shared with the projector sweep's one-time observed-site lookup (calendar_utils D-09), and that discovery must carry these keys forward and never erase them (F1, v2.4-INTENT-REVIEW.md). In the same edit, add a module-level function `_preserve_observed_site_keys(existing: Any, rebuilt: dict[str, Any]) -> dict[str, Any]` placed immediately above `_changed_record_fields`. It returns a NEW dict: a shallow copy of `rebuilt`, plus, for each key in `OBSERVED_SITE_PARAMETER_KEYS` that is PRESENT in `existing` (a membership test `key in existing`, never a truthiness test, so a stored `None` enclosure survives), that key set to `existing[key]`. When `existing` is not a `dict`, it returns a plain copy of `rebuilt`. It never mutates either argument. Give it a Google-style docstring (`Args:`/`Returns:`) that says why it exists. The sweep stores these keys once per record, ever, and treats their presence as "already looked up". `_build_parameters()` never produces them. A whole-dict comparison would therefore treat them as portal drift every tick, and discovery would erase them, restarting a live portal lookup plus a coarse-token re-projection for every tagged record (F1). After this edit the helper exists but nothing calls it yet, so behavior is unchanged.

    Edit 2. At the top of `_changed_record_fields()`'s body, rebind `parameters = _preserve_observed_site_keys(record.parameters, parameters)` before any comparison. Leave the rest of the function as it is, so the existing `changes['parameters'] = parameters` now returns the merged dict and the real-run `setattr` writes the keys back instead of dropping them. Placing the carry-forward here, in the one comparison both `sweep_proposal()` branches already call, is what keeps the dry-run and real-run decisions in agreement (T-ik7-02). Do NOT add a second call at either call site. Update the docstring's `parameters` arg description to say the three `OBSERVED_SITE_PARAMETER_KEYS` already on `record.parameters` are carried into it before comparing. Also say that the returned `'parameters'` value, when present, is that merged dict. Do not change `_build_parameters()`, the `get_or_create` defaults (a newly created record has no keys to carry), the summary line, or anything on the projector/sweep side.

    Run the whole test module: every pre-existing test plus the new ones must pass. Then run `pre-commit run ruff --files` and `pre-commit run ruff-format --files` on the two files, and commit: `fix(260929-svk): discovery carries the sweep's observed-site keys forward (F1)`. Stage the two files by explicit path only, never `git add -A` or `git add .`. The working tree also has unrelated pre-existing changes that must stay uncommitted: modified `.planning/v2.4-INTENT-REVIEW.md`, and untracked `.gsd/`, `.planning/agent-history.json`, `reqgroup_2682493.json`, a file literally named `select count(*) from tom_calendar_calendarevent;`, and `src/fomo_db_20260929.sqlite3`.

    Never run any `manage.py` command other than `test` and `shell -c` in this task. In particular, do not run `backfill_lco_observations`, `project_observation_calendar`, `run_unattended` or `migrate` against the default database: it is the live one.
  </action>
  <verify>
    <automated>python manage.py shell -c "import solsys_code.management.commands.backfill_lco_observations as m; print('importable:', m._preserve_observed_site_keys.__name__)"</automated>
    <automated>python manage.py test solsys_code.tests.test_backfill_lco_observations</automated>
    <automated>python -c "import ast; t = ast.parse(open('solsys_code/management/commands/backfill_lco_observations.py').read()); f = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == '_changed_record_fields'); calls = {c.func.id for c in ast.walk(f) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}; assert '_preserve_observed_site_keys' in calls, calls; print('OK: the shared comparison applies the carry-forward')"</automated>
    <automated>pre-commit run ruff --files solsys_code/management/commands/backfill_lco_observations.py solsys_code/tests/test_backfill_lco_observations.py && pre-commit run ruff-format --files solsys_code/management/commands/backfill_lco_observations.py solsys_code/tests/test_backfill_lco_observations.py</automated>
  </verify>
  <done>Tests A-E and helper tests (1)-(5) pass, and so does every pre-existing test in the module. `_changed_record_fields()` calls `_preserve_observed_site_keys()`, and neither call site does. The module imports cleanly. Ruff and ruff-format are clean on both files. Two commits exist, the failing tests first and then the fix, each touching only its own files.</done>
</task>

<task type="auto">
  <name>Task 2: Paired notebook routed to a fresh scratch database and shows the carry-forward; runbook bullet updated</name>
  <files>docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb, cells by id: `ae33103d`, `86dd4f0a`, `2659707b`, `a0c6306e`, `0a7d2d6b`, `0cee7e7a` (the dry-run patch idiom), `dcb271ab`, `33d8ee67`, `6a1f0c12` (why the bare sweep must never see a real watched row), `1c6c298c`, `ce8337f7`
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, cell ids `dfcc0343` and `9084663a` (the scratch-routing pattern to follow, apart from the copy step) and `98bba968`/`d5248b35` (teardown)
    - docs/runbooks/telescope_runs_calendar.rst, lines 354-372 (the `backfill_lco_observations` "How it differs" bullets)
  </read_first>
  <action>
    Part A: make the notebook safe to execute (planning finding 2). This MUST be done before the notebook is executed even once.

    Edit the notebook's JSON with a throwaway Python script kept in the session scratchpad (not the repo). The script should use `nbformat.read(path, as_version=4)` / `nbformat.write`. Locate cells by their `id`, not by index, and create new cells with `nbformat.v4.new_markdown_cell` / `new_code_cell`; under nbformat 4.5 these give each new cell a fresh `id`.

    In setup cell `2659707b`, keep the existing `sys.path`/`DJANGO_SETTINGS_MODULE`/`DJANGO_ALLOW_ASYNC_UNSAFE` lines. Before `django.setup()`, add scratch routing:
    - import `tempfile`;
    - create `scratch_db_dir = Path(tempfile.mkdtemp(prefix='fomo-notebook-db-'))` and `scratch_db_path = scratch_db_dir / 'fomo_db.sqlite3'`;
    - assign `os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)` with a plain assignment, never `setdefault`, so an inherited value can never point the run at a real database.

    After `django.setup()`:
    - assert that `django.conf.settings.DATABASES['default']['NAME'] == str(scratch_db_path)`, with a message saying this notebook must never write to the developer or live database;
    - run `call_command('migrate', verbosity=0)` (import it under a private alias as the reconcile notebook does) to build the schema from empty;
    - print `Resolved database: '<path>' (fresh scratch database, migrated from empty)` in exactly that `Resolved database: '...'` form, because Task 2's gate parses it.

    Do NOT copy `src/fomo_db.sqlite3`, unlike the reconcile notebook. Put the reason in a short code comment: the live copy would bring along active watched proposals that the bare-invocation cell would sweep, the file may be mid-write by the cron, and this demo creates every row it needs itself.

    Then update the prose to match:
    - Extend markdown `86dd4f0a` with a paragraph explaining the scratch database and why it is fresh rather than copied.
    - In intro `ae33103d`, change the "DB-dependent ... creates real ... rows in the local dev database" wording to say the rows are created in a throwaway scratch database the setup cell builds and migrates, never the developer database. Also add a bullet to its "It demonstrates, in order" list for the new observed-site section described in Part B.
    - In cleanup markdown `1c6c298c`, reword the "local dev database" sentence the same way. The cleanup cell itself stays.
    - Append a final markdown cell plus code cell for scratch teardown, modeled on the reconcile notebook's `98bba968`/`d5248b35`: `shutil.rmtree(scratch_db_dir, ignore_errors=True)`, then print `Removed scratch database directory: <path>`. This must be the LAST code cell.

    Part B: demonstrate the fix. Insert, immediately after code cell `33d8ee67` and before markdown `6a1f0c11`, one markdown cell and two code cells.

    The markdown cell's heading should say that discovery keeps the sweep's observed-site keys. Its prose should explain four things in plain English. First, the projector sweep's one-time observed-site lookup stores `observed_site`/`observed_telescope`/`observed_enclosure` in `parameters` and treats their presence as "already looked up". Second, discovery rebuilds `parameters` from the portal with only `proposal`/`instrument_type`/`start`/`end`, and now carries those three keys forward. Third, before this fix, every unattended tick counted each tagged record as `updated` and erased the keys. The post_save trigger then re-drew its event with the coarse telescope token, and the next sweep repeated one live portal lookup per record. Fourth, discovery still refreshes the portal-owned keys when the portal really changes.

    Code cell 1:
    - Import `OBSERVED_SITE_PARAMETER_KEYS` from `solsys_code.calendar_utils`.
    - Tag `record_101` with `('elp', '1m0a', 'doma')` and `record_102` with `('lsc', '1m0a', 'domb')`, zipped onto the constant. For each, call `parameters.update(...)` and then `save(update_fields=['parameters'])`, the same write `resolve_observed_site()` makes.
    - Call `refresh_from_db()` on both, then capture both `modified` values (planning finding 3).
    - Run a `--dry-run` pass over `second_pass_request_group`, patching `make_request` and `get_observation_status` exactly as cell `0cee7e7a` does, and print its stdout.
    - Run a real pass over the same payload with `side_effect=_status_lookup(SECOND_PASS_SCHEDULE)` and print its stdout.
    - Refresh both records and print each one's full `parameters` dict, then whether each `modified` is unchanged.
    - Print a line starting `PASS:` when the dry-run stdout contains `would update: 0`, the real stdout contains `updated: 0, unchanged: 2`, and both records still hold all three keys with their tagged values.

    Code cell 2:
    - Build a payload identical to `second_pass_request_group` except that request 900101's `windows` is `[{'start': '2026-07-02T00:00:00', 'end': '2026-07-03T00:00:00'}]`. Build it by dict-copying `_request(900101, 'COMPLETED')` and replacing `'windows'`, and keep `_request(900102, 'COMPLETED')` unchanged in the same group.
    - Run a real pass with `SECOND_PASS_SCHEDULE`, and print its stdout, which should show `updated: 1, unchanged: 1`.
    - Refresh `record_101` and print its `parameters` (new `start`/`end`, three keys intact).
    - Print a `PASS:` line when both hold.

    Record ids stay 900101/900102, which cell `ce8337f7` already deletes, so the cleanup cell needs no change. Do not print the calendar event title; it is out of this demo's scope.

    Part C: format, then execute.
    - Run `pre-commit run ruff-format --files docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` BEFORE executing, so the executed sources are already in their final formatted form.
    - Then, from the repo root and in the FOREGROUND (Bash tool timeout 600000 ms; never `run_in_background`, never `&`), run `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`. nbconvert runs the kernel in the notebook's own directory, which is what the setup cell's `parents[2]` check requires.
    - Do not export `FOMO_DATABASE_PATH` yourself, and do not pre-create or copy any database: the setup cell does all of it.
    - If a cell errors, fix the source and re-execute the WHOLE notebook. Never hand-write outputs. The committed executed outputs are the deliverable.

    Part D: runbook. In `docs/runbooks/telescope_runs_calendar.rst`, append one or two sentences to the end of the "**Re-running updates in place instead of skipping.**" bullet (about lines 365-369), keeping its indentation. Say that the three observed-site keys (``observed_site``, ``observed_telescope``, ``observed_enclosure``), which the projector sweep's one-time lookup stores in ``parameters``, are carried forward and never erased. Also say that a re-run over unchanged portal data therefore reports the record ``unchanged``. Change nothing else in the runbook.

    Commit the two files, staged by explicit path: `docs(260929-svk): demo observed-site carry-forward on a scratch database; runbook note`.
  </action>
  <verify>
    <automated>python -c "
import json, re
nb = json.load(open('docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb'))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
counts = [c.get('execution_count') for c in code]
assert counts == list(range(1, len(code) + 1)), f'not one fresh top-to-bottom run: {counts}'
errors = [i for i, c in enumerate(code) if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
assert not errors, f'error outputs in code cells {errors}'
def out(c):
    return ''.join(''.join(o.get('text', '')) for o in c.get('outputs', []))
setup = [c for c in code if 'FOMO_DATABASE_PATH' in ''.join(c['source'])]
assert len(setup) == 1, f'expected one setup cell routing FOMO_DATABASE_PATH, found {len(setup)}'
m = re.search(r\"Resolved database: '([^']+)'\", out(setup[0]))
assert m, 'setup cell printed no resolved database'
assert 'fomo-notebook-db-' in m.group(1) and not m.group(1).endswith('src/fomo_db.sqlite3'), m.group(1)
demo = [c for c in code if 'OBSERVED_SITE_PARAMETER_KEYS' in ''.join(c['source'])]
assert demo, 'no observed-site demo cell'
text = ''.join(out(c) for c in demo)
for token in ('would update: 0', 'updated: 0, unchanged: 2', 'updated: 1, unchanged: 1', 'observed_site', 'PASS:'):
    assert token in text, f'demo output lacks {token!r}'
assert 'Removed scratch database directory' in out(code[-1]), 'last code cell is not the scratch teardown'
print(f'OK: {len(code)} code cells, one fresh run, scratch database {m.group(1)}')
"</automated>
    <automated>python -c "
import sqlite3
con = sqlite3.connect('file:src/fomo_db.sqlite3?mode=ro', uri=True, timeout=30)
cur = con.cursor()
records = cur.execute(\"select count(*) from tom_observations_observationrecord where observation_id in ('900101','900102','900301','900401','900402')\").fetchone()[0]
watched = cur.execute(\"select count(*) from solsys_code_watchedproposal where proposal_code like 'BACKFILL-DEMO%'\").fetchone()[0]
keyerror = cur.execute(\"select count(*) from solsys_code_watchedproposal where last_run_summary like 'failed: KeyError%'\").fetchone()[0]
assert (records, watched, keyerror) == (0, 0, 0), (records, watched, keyerror)
print('OK: live database (opened read-only) carries no demo rows and no KeyError sweep summary')
"</automated>
    <automated>python -c "
t = open('docs/runbooks/telescope_runs_calendar.rst').read()
i = t.index('Re-running updates in place instead of skipping')
j = t.index('Unmatched targets are always built as non-sidereal', i)
seg = t[i:j]
for token in ('observed_site', 'observed_telescope', 'observed_enclosure', 'unchanged'):
    assert token in seg, f're-run bullet lacks {token!r}'
print('OK: the re-run bullet states the observed-site carry-forward')
"</automated>
    <automated>pre-commit run ruff-format --files docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb && pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst</automated>
  </verify>
  <done>The notebook was executed once, top to bottom, with no error output. Its setup cell's stored output shows a `fomo-notebook-db-` scratch path. The new cells' outputs show `would update: 0`, `updated: 0, unchanged: 2`, `updated: 1, unchanged: 1` and the surviving `observed_site` keys, and the last cell removed the scratch directory. A read-only query of `src/fomo_db.sqlite3` finds no demo rows and no `failed: KeyError` summary. The runbook bullet states the carry-forward, and Sphinx builds.</done>
</task>

<task type="auto">
  <name>Task 3: Full quality gates, then report the first post-fix live ticks</name>
  <files>solsys_code/management/commands/backfill_lco_observations.py, solsys_code/tests/test_backfill_lco_observations.py, docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - CLAUDE.md, "Commands" and "Testing" sections (always `python manage.py`, never `./manage.py`; ruff is run through pre-commit so the pinned v0.2.1 is what judges, per D-07)
  </read_first>
  <action>
    From the repo root, run `pre-commit run ruff --all-files` and then `pre-commit run ruff-format --all-files`. If either rewrites a file outside this plan's four files, stop and report it instead of committing someone else's file.

    Then run the full Django suite: `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (the tag excludes `TestEphemeris`, which segfaults in native ASSIST). It takes about 8 minutes. Use a Bash timeout of 600000 ms. If that is not enough, re-run it with `run_in_background` and wait for its completion notification. Never skip it or cut it short. The suite uses Django's in-memory test database and never opens the live one.

    Commit only if a gate made a formatting change to this plan's files, staged by explicit path.

    Finally, as a read-only report for SUMMARY.md, never a blocking step: grep `/var/log/fomo/unattended.log` for the `step project_sweep` and `step discovery` lines and the START/END banners of every tick that has STARTED since Task 1's fix commit, and quote them. Expect this pattern. The first post-fix tick should still show about 133 `site_lookups` (the sweep restoring keys that the previous tick's discovery erased), but discovery should now report those records `unchanged`. From the second post-fix tick on, expect `project_sweep ... updated: 0 ... site_lookups: 0`, and tick durations back near the pre-F1 18 s. Do NOT wait for ticks to happen, and do NOT tick the F1 checkbox in `.planning/v2.4-INTENT-REVIEW.md`; that file belongs to the operator's walkthrough and already has uncommitted edits.
  </action>
  <verify>
    <automated>pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
    <automated>python manage.py test solsys_code --exclude-tag=ephemeris_segfault</automated>
    <human-check>Operator, in `/var/log/fomo/unattended.log`: two consecutive post-fix ticks show `step project_sweep: ok ... updated: 0 ... site_lookups: 0` and discovery with no spurious `updated` count, at a steady-state duration (the F1 checkbox in `.planning/v2.4-INTENT-REVIEW.md`). Observation event titles then stay in the observed-site form (for example `[O] TFN-1m0 11P`) across ticks, ready for the D1/D4/Q4 title checks.</human-check>
  </verify>
  <done>Both ruff hooks are clean across the repo. The full `solsys_code` suite passes with the segfault tag excluded. SUMMARY.md quotes whatever post-fix tick lines already exist and states that the two-tick confirmation is the operator's to make.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| discovery (LCO portal payload) -> `ObservationRecord.parameters` | Portal-supplied request data is written onto records that another writer (the projector sweep) also annotates |
| notebook kernel -> Django `DATABASES` | A docs artifact executes real ORM writes on this host, which is also the production host whose cron writes `src/fomo_db.sqlite3` every 15 minutes |
| working tree -> running cron | `run_unattended` imports this checkout's code at every tick, so an edit on disk is live code |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-svk-01 | Tampering | `backfill_lco_observations_demo.ipynb` re-execution against the live database | high | mitigate | Task 2 Part A routes the setup cell to a fresh `tempfile.mkdtemp(prefix='fomo-notebook-db-')` file before `django.setup()`, uses a plain assignment (not `setdefault`), asserts the resolved `NAME` equals the scratch path before any ORM call, and migrates from empty instead of copying the live file, so the live `KEY2026B-004` row is never swept or marked `failed: KeyError`. Task 2's gate reads the resolved path out of the stored output and queries the live database read-only (`mode=ro`) for demo rows and KeyError summaries. |
| T-svk-02 | Denial of service | discovery/sweep churn (F1): ~7-minute ticks on a 15-minute `flock -n` schedule, so a slow portal means a skipped tick | medium | mitigate | `_preserve_observed_site_keys()` inside `_changed_record_fields()` stops discovery from saving a tagged record whose portal data is unchanged. Test A asserts `unchanged: 1` and an unmoved `modified`. Task 3's human check confirms `site_lookups: 0` on two consecutive live ticks. |
| T-svk-03 | Denial of service | a half-edited `backfill_lco_observations.py` imported by the next cron tick | medium | mitigate | Task 1 writes tests first (never imported by the runner), splits the production change into two edits that are each importable and correct on their own, and runs an import smoke check after each. |
| T-svk-04 | Repudiation | dry-run versus real-run disagreement (T-ik7-02), where an operator's dry run would mis-predict what the real sweep writes | low | mitigate | The carry-forward lives only inside the shared `_changed_record_fields()`. An AST gate asserts it is called there, and Tests C/D pin dry-run counts equal to the real-run counts of Tests A/B. |
| T-svk-05 | Tampering | discovery keeping a stale observed-site value | low | accept | Only the three sweep-reserved keys are carried forward. The portal-owned keys still refresh (Test B). The sweep remains the only writer of the reserved keys and writes them once, after a successful terminal state, when the placement no longer changes. |
| T-svk-SC | Tampering | package installs | low | accept | No npm/pip/cargo install in this plan. `nbformat`/`nbconvert` are already installed (nbconvert 7.17.1, nbformat 5.10.4). |
</threat_model>

<verification>
- `python manage.py test solsys_code.tests.test_backfill_lco_observations` passes, including the new `TestObservedSiteKeysSurviveDiscovery` and `TestPreserveObservedSiteKeys`.
- `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` passes.
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` are clean.
- The notebook's stored outputs prove one fresh run on a `fomo-notebook-db-` scratch database, with the carry-forward shown; a read-only query of `src/fomo_db.sqlite3` finds no demo rows.
- `git log` for this task shows commits touching only the four `files_modified` paths plus the task's own planning SUMMARY. The pre-existing dirty or untracked files are still uncommitted.
</verification>

<success_criteria>
- Discovery never erases `observed_site`/`observed_telescope`/`observed_enclosure` (including a `None` value), still refreshes `proposal`/`instrument_type`/`start`/`end`, and the dry-run and real-run counts agree.
- The paired notebook and the runbook describe the new behavior, and the notebook can be re-executed on this host without touching the live database.
- After deployment, which happens as soon as Task 1 lands, because the cron runs this checkout: two consecutive live ticks show `project_sweep: updated: 0, site_lookups: 0`. The operator confirms this and ticks the F1 checkbox.
</success_criteria>

<output>
Create `.planning/quick/260929-svk-stop-the-discovery-sweep-churn-loop-on-o/260929-svk-SUMMARY.md` when done. Include the quoted post-fix tick lines from Task 3, if any, and restate that the two-tick F1 confirmation is the operator's.
</output>
