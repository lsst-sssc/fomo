---
phase: quick-260722-uyz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/tests/test_sync_lco_observation_calendar.py
  - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
autonomous: true
requirements: []
must_haves:
  truths:
    - "A synced CalendarEvent whose ObservationRecord's Target belongs to exactly one campaign TargetList has target_list set to that TargetList."
    - "A synced CalendarEvent whose Target belongs to zero TargetLists has target_list=None (no crash)."
    - "When a Target belongs to 2+ TargetLists, the CalendarEvent's target_list is deterministically the alphabetically-first-by-name TargetList."
    - "Re-syncing an unchanged record whose target_list already matches reports 'unchanged', not 'updated' (no-churn preserved for the new FK field)."
  artifacts:
    - solsys_code/management/commands/sync_lco_observation_calendar.py
    - solsys_code/tests/test_sync_lco_observation_calendar.py
    - docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb
  key_links:
    - "_build_event_fields() returns 'target_list' -> handle() passes fields through -> insert_or_create_calendar_event constructs/updates CalendarEvent.target_list."
    - "record.target.targetlist_set.order_by('name').first() -> CalendarEvent.target_list FK."
---

<objective>
Fix a gap present since the command's original Phase 04 implementation: `sync_lco_observation_calendar`
creates CalendarEvents with no TargetList/campaign association because `_build_event_fields()` never
populates `CalendarEvent.target_list`. Derive it from the ObservationRecord's Target's TargetList
membership, picking deterministically by campaign name when a Target belongs to more than one.

Purpose: after `backfill_lco_observation_records --create-missing-targets` adds a Target to a campaign
TargetList and `sync_lco_observation_calendar` runs, the resulting CalendarEvents show their campaign
association in the FOMO calendar UI instead of appearing unassociated.

Output: updated command source, extended Django tests, and the paired demo notebook (a mandated module
per CLAUDE.md) demonstrating the newly-populated `target_list`.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/management/commands/sync_lco_observation_calendar.py
@solsys_code/tests/test_sync_lco_observation_calendar.py
@solsys_code/calendar_utils.py

# Verified facts (no need to re-derive):
# - TargetList.targets is a ManyToManyField; the reverse accessor on Target is `targetlist_set`.
# - CalendarEvent.target_list is a nullable ForeignKey(TargetList, on_delete=SET_NULL) — safe to leave None.
# - handle() already does: url = fields.pop('url'); telescope_api_failed = fields.pop('telescope_api_failed');
#   insert_or_create_calendar_event({'url': url}, fields) — every remaining key in `fields` rides through
#   to CalendarEvent construction/update, so a new 'target_list' key needs NO change to handle().
# - insert_or_create_calendar_event -> _update_or_unchanged() diffs each field via `getattr(event, f) != v`.
#   For a FK, getattr returns the related model instance and Django model __eq__ compares by (class, pk),
#   so an unchanged FK compares equal and produces no spurious 'updated'. (Task 3 proves this.)
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Populate target_list in _build_event_fields()</name>
  <files>solsys_code/management/commands/sync_lco_observation_calendar.py</files>
  <behavior>
    - When the record's Target belongs to exactly one TargetList, the returned dict's 'target_list' is that TargetList.
    - When the Target belongs to zero TargetLists, 'target_list' is None (no exception on .first()).
    - When the Target belongs to 2+ TargetLists, 'target_list' is the alphabetically-first-by-name TargetList.
    - handle() is unchanged: the new key flows through the existing pop('url')/pop('telescope_api_failed')/pass-through path.
  </behavior>
  <action>
    In `_build_event_fields(record, facility)`, before the returned dict, derive the campaign TargetList
    from the record's Target using the reverse M2M accessor: order the Target's TargetLists by name and
    take the first, so the pick is stable and deterministic whether the Target belongs to zero (None,
    acceptable since the FK is nullable), one, or several TargetLists. Add a matching key to the returned
    dict alongside the existing url/title/description/start_time/end_time/telescope/instrument/proposal/
    telescope_api_failed keys, so it rides through handle()'s existing pass-through to
    insert_or_create_calendar_event exactly like the other CalendarEvent field keys. Do NOT modify
    handle() (the existing pop/pass-through already carries any extra CalendarEvent field key). Do NOT
    add UI or a warning for the multi-membership case — the deterministic pick is the accepted behavior.
    This function is shared by the LCO and SOAR branches; the change applies uniformly to both, which is
    intended. Update the docstring's Returns: section (which enumerates the dict's keys) to include the
    new target_list key and note the deterministic-by-name pick. Use the exact accessor
    `record.target.targetlist_set.order_by('name').first()`.
  </action>
  <verify>
    <automated>DJANGO_SETTINGS_MODULE=src.fomo.settings python -c "import ast,sys; s=open('solsys_code/management/commands/sync_lco_observation_calendar.py').read(); assert \"targetlist_set.order_by('name').first()\" in s and \"'target_list'\" in s, 'target_list derivation missing'; ast.parse(s); print('OK')"</automated>
  </verify>
  <done>_build_event_fields returns a 'target_list' key derived from record.target.targetlist_set.order_by('name').first(); handle() unchanged; docstring Returns: mentions target_list; module still parses.</done>
</task>

<task type="auto">
  <name>Task 2: Add TargetList-membership tests</name>
  <files>solsys_code/tests/test_sync_lco_observation_calendar.py</files>
  <action>
    Add four new test methods to TestSyncLcoObservationCalendar (import TargetList from tom_targets.models
    at the top with the other imports). Note Django TestCase rolls back each test's DB changes, so
    M2M additions to the shared setUpTestData target do not leak between tests; existing tests continue to
    see the target in zero TargetLists and stay green unchanged.
    1. Exactly-one-TargetList: create a TargetList, add the shared target to it, create a record, run the
       command, assert the created CalendarEvent's target_list equals that TargetList.
    2. Zero-TargetLists: create a record whose target belongs to no TargetList, run the command, assert the
       CalendarEvent's target_list is None (proves .first() returning None does not crash and the nullable
       FK stays empty).
    3. Two-TargetLists deterministic pick: create two TargetLists with names whose alphabetical order is
       unambiguous (e.g. 'Alpha Campaign' and 'Beta Campaign'), add the SAME target to both, run the
       command, assert the CalendarEvent's target_list is exactly the alphabetically-first one by identity
       (assertEqual on the TargetList instance/pk), not merely one-of-the-two.
    4. No-churn on the FK field: create a record whose target is in one TargetList, run the command, capture
       the created event's modified timestamp, run the command a second time with target_list resolving to
       the same TargetList, and assert the second run reports 'unchanged' (assert 'unchanged: 1' in stdout)
       and the event's modified timestamp is unchanged — proving the FK-valued field participates in the
       existing change-detection without a spurious diff.
    Reuse the existing _create_record / call_command helpers and stdout capture pattern already in the file.
    Keep every existing test unchanged.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_sync_lco_observation_calendar 2>&1 | tail -5</automated>
  </verify>
  <done>Four new tests (one/zero/two-TargetList and no-churn) pass; all pre-existing tests in the file still pass.</done>
</task>

<task type="auto">
  <name>Task 3: Update paired demo notebook + quality gate</name>
  <files>docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb, solsys_code/management/commands/sync_lco_observation_calendar.py, solsys_code/tests/test_sync_lco_observation_calendar.py</files>
  <action>
    The demo notebook is a mandated companion for this module (CLAUDE.md "Demo notebook companions") and
    its cells already inspect created CalendarEvent fields (cell 9 prints title/telescope/instrument/
    proposal; cell 11 iterates events). Because target_list is a new field on the demonstrated surface and
    today's demo target belongs to no TargetList, add a TargetList to the demo and show the new
    association: in the target-setup cell (cell 4) create-or-get a demo TargetList (e.g. 'Demo Campaign')
    via a re-runnable get_or_create + .targets.add(demo_target) so re-execution does not accumulate rows,
    and extend the event-inspection cell(s) to print event.target_list (expect the demo TargetList's name
    for a synced event, demonstrating the fix's real-world scenario). Then regenerate the notebook with
    executed output committed:
    `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`
    (pre_executed notebooks are committed WITH output per the pre-commit convention). Finally run the
    quality gate on the touched Python files: `ruff check . --fix` and `ruff format .`, and confirm clean.
  </action>
  <verify>
    <automated>jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb && python -c "import json; nb=json.load(open('docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb')); s=''.join(''.join(c['source']) for c in nb['cells']); assert 'target_list' in s, 'notebook does not demonstrate target_list'; print('OK')" && ruff check . && ruff format --check .</automated>
  </verify>
  <done>Notebook demonstrates a populated target_list with committed executed output; re-execution is idempotent; ruff check and ruff format --check are clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| local DB read | target_list is derived from the ObservationRecord's Target's TargetList membership — all local DB data, no new external input crosses a boundary. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-uyz-01 | Tampering | CalendarEvent.target_list assignment | low | accept | Value is a local TargetList FK chosen deterministically by name; no external/untrusted data influences it. Nullable FK safely absorbs the zero-membership case. |
| T-uyz-SC | Tampering | package installs | low | accept | No new packages installed; no install task in this plan. |
</threat_model>

<verification>
- `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — all tests pass (existing + 4 new).
- `ruff check .` and `ruff format --check .` clean.
- Demo notebook re-executes and shows a populated target_list.
</verification>

<success_criteria>
- CalendarEvents synced by the command carry the campaign TargetList (deterministic first-by-name) or None.
- No-churn idempotency holds for the new FK field (unchanged re-sync reports 'unchanged').
- All existing tests remain green; the mandated demo notebook is updated with executed output.
</success_criteria>

<output>
Create `.planning/quick/260722-uyz-fix-sync-lco-observation-calendar-set-ca/260722-uyz-SUMMARY.md` when done
</output>
