---
status: resolved
phase: 34-the-observation-projector-trigger
gap_id: G-34-2
created: 2026-09-11
updated: 2026-09-14
---

## Current Focus

hypothesis: CONFIRMED and FIXED -- portal ISO strings on the in-memory post_save
  instance reached `.strftime()` uncoerced.
test: revert test (surgically restore the raw-attribute read in
  `record_time_window()`) + full suite + lint gates, then the human SCHED-06
  re-check against the real DB.
expecting: the revert reproduces the exact AttributeError; restoring the fix turns
  every gate green; the real `updatestatus` run narrows the 33 stale events with no
  sweep.
next_action: none -- session resolved and archived. CORRECTION (2026-09-14, Phase 34
  re-verification): the "overnight updatestatus-only run narrowed all 33" claim below
  was never true -- no `updatestatus` run occurred after 2026-09-12 (DB file mtime
  evidence), and 14 of the 33 were still stale as of the re-verification. SCHED-06 /
  UAT Test 4's real evidence is narrower: two individual records (4378332, 4378046)
  show genuine receiver-alone narrowing with `CalendarEvent.modified ==
  ObservationRecord.modified` to the second, from real production activity, not a
  bulk overnight run. The remaining 14 legacy-stale COMPLETED events could never be
  reached by `updatestatus` at all (terminal states are excluded,
  `facility.py:573`) -- they were cleared to 0 by a real `project_observation_calendar`
  sweep on 2026-09-14, after applying the then-pending
  `0018_campaignrun_night_window_fields` migration. See the corrected Evidence/
  Resolution entries below.

reasoning_checkpoint:
  hypothesis: "`OCSFacility.get_observation_status()` returns raw portal ISO strings;
    `update_observation_status()` assigns them onto the record and saves, so post_save
    fires with `scheduled_start`/`scheduled_end` still `str`, and `event_fields_for()`
    crashes calling `.strftime()`."
  confirming_evidence:
    - "Captured traceback on the real path: AttributeError: 'str' object has no
      attribute 'strftime' at observation_projector.py:291."
    - "Same record projects fine when fetched from the DB (real DateTimeField), so the
      difference is the in-memory instance, not the record's data."
    - "Revert test (2026-09-14): restoring the raw-attribute read in
      `record_time_window()` reproduces the identical AttributeError and fails exactly
      the two updatestatus tests; restoring the coercion turns them green."
  falsification_test: "If the coercion were irrelevant, reverting it would leave the
    suite green. It does not -- 2 targeted failures with the exact original error."
  fix_rationale: "Coercing at the single shared window-derivation point makes the
    receiver path and the sweep path derive the SAME window from the same record,
    which is the actual invariant the bug broke -- not merely suppressing the crash."
  blind_spots: "The real-portal `updatestatus` run against src/fomo_db.sqlite3 was
    deliberately NOT executed -- it is the SCHED-06 re-check evidence (see Consequence)."
  candidate_causes:
    - "code: projector calls .strftime() on an uncoerced attribute (CONFIRMED)"
    - "data: portal returns ISO strings rather than datetimes (CONFIRMED -- contributing)"
    - "environment: Django persists the str without coercing the Python attribute
      (CONFIRMED -- contributing)"
  and_gate: "yes -- all three conditions must hold simultaneously. A datetime-valued
    facility (the pre-fix test's fake) never triggers it, and a DB-fetched instance
    never triggers it. This is why the sweep and all 296 tests passed while the real
    path failed on every record."

# G-34-2: post_save receiver fails with AttributeError on every real `updatestatus` save

## Symptom

Two overlapping `python manage.py updatestatus` runs (UAT Test 2) logged
`unprojectable observation_id='...': AttributeError` for nearly every LCO record touched
(`tmp/project_observation_calendar_dry_run.txt`). A subsequent
`project_observation_calendar --dry-run` on the real DB reported
`LCO: created: 0, updated: 33, unchanged: 126` -- 33 events the receiver should have narrowed
but silently left stale. The receiver never raises (TRIG-02), so `updatestatus` itself reports
`Update completed successfully`.

## Reproduction (real path, traceback captured)

```python
import traceback, solsys_code.observation_projector as op
_orig = op.logger.warning
def loud(msg, *a, **k):
    _orig(msg, *a, **k); traceback.print_exc()
op.logger.warning = loud
from tom_observations.facility import get_service_class
get_service_class('LCO')().update_observation_status('4378029')
```

```
Traceback (most recent call last):
  File "solsys_code/observation_projector.py", line 368, in project_record
    fields, stage = event_fields_for(record, facility)
  File "solsys_code/observation_projector.py", line 291, in event_fields_for
    f'Window (UTC): {start_time.strftime("%Y-%m-%dT%H:%M:%S")} to {end_time.strftime(...)}'
AttributeError: 'str' object has no attribute 'strftime'
```

Projecting the same record from a shell (`event_fields_for` on a DB-fetched instance)
succeeds: stage=placed, title `[S] 1m0 220P` -- the failure is specific to the in-memory
instance `updatestatus` saves.

## Root cause

- `OCSFacility.get_observation_status()` (tomtoolkit 3.0.1,
  `tom_observations/facilities/ocs.py:1570`) returns the portal's raw ISO strings for
  `scheduled_start` / `scheduled_end` (`current_block['start']`).
- `BaseObservationFacility.update_observation_status()` (`facility.py:563`) assigns them
  straight onto the record and calls `save()`. Django persists the strings without coercing
  the Python attribute, so `post_save` fires with `record.scheduled_start` still a `str`.
- `record_time_window()` (`solsys_code/calendar_utils.py:458`) returns those attributes
  unchanged; `event_fields_for()` (`observation_projector.py:291`) calls `.strftime()` on
  them and crashes. `project_record()` catches it, logs `unprojectable`, and the event stays
  stale.
- The sweep re-fetches records from the DB (real `DateTimeField` values) and so never hits
  this -- which is why `project_observation_calendar` and all 296 tests pass.
  `test_updatestatus_narrows_the_event_with_no_command_run`
  (`solsys_code/tests/test_observation_projector_signals.py`) fakes `get_observation_status`
  with `datetime` objects, not the portal's ISO strings, so it never exercised the real contract.

## Consequence

SCHED-06 (UAT Test 4) cannot be satisfied until this is fixed: with the receiver failing on
every real `updatestatus` save, nothing narrows without the sweep. The 33 stale events are the
natural re-check evidence once fixed -- `updatestatus` re-saves every record each run, so the
next run alone should repair them through the receiver. Do NOT run the real sweep against
`src/fomo_db.sqlite3` to repair them; that would destroy the SCHED-06 evidence.

## Fix direction

1. On the projector path, coerce `scheduled_start` / `scheduled_end` to aware UTC datetimes
   when they arrive as `str` (ISO 8601, possibly with a trailing `Z`) -- in
   `record_time_window()` and/or `event_fields_for()` / `stage_for()`, so a post_save
   instance holding portal strings projects identically to a DB-fetched one.
2. Change the signals test to feed ISO strings exactly as `OCSFacility.get_observation_status`
   returns them (and keep a datetime case), so the test exercises the real contract.
3. CLAUDE.md paired docs: `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
   re-executed; the sweep behaviour is unchanged so the runbook page likely needs no edit --
   confirm.

## Evidence

- timestamp: 2026-09-14
  checked: git history for the fix direction's three steps.
  found: all three already landed via the phase-34 gap-closure plans -- 34-05
    (`f468eda` RED test feeding portal ISO strings, `bfac4b2` GREEN coercion, plus
    review fixes CR-01/CR-02/WR-03/WR-04/WR-05) and 34-07 (`37ffe2b`/`8757750`
    notebook re-execution on an un-swept clone).
  implication: the debug file's `status:` was simply never reconciled; the fix itself
    was complete. Remaining work is verification, not implementation.

- timestamp: 2026-09-14
  checked: step 1 -- coercion at the shared window-derivation point.
  found: `calendar_utils.coerce_schedule_datetime()` handles str/datetime/None,
    normalizes to aware UTC, and RAISES on unusable values rather than degrading to
    None (which would draw a queued-looking event over the wrong window).
    `record_time_window()` routes BOTH branches through it, and
    `event_fields_for()` routes its 'inconsistent' branch through it too.
  implication: receiver and sweep now derive identical windows -- the real invariant.

- timestamp: 2026-09-14
  checked: step 2 -- the signals test's fidelity to the real contract.
  found: `test_updatestatus_narrows_the_event_with_no_command_run` now feeds
    `block_start.isoformat().replace('+00:00', 'Z')`, matching what the portal
    returns; a datetime-valued case is retained separately
    (`test_updatestatus_with_datetime_valued_facility_still_narrows_the_event`), plus a
    no-churn span test and a WR-04 unparseable-value test.
  implication: the gap that let the original bug through (a datetime-valued fake) is
    closed, and the boundary neighbours around it are pinned.

- timestamp: 2026-09-14
  checked: REVERT TEST -- restored the pre-fix raw-attribute read in
    `record_time_window()` and re-ran the signals module.
  found: `unprojectable observation_id='projector-signals-001': AttributeError: 'str'
    object has no attribute 'strftime'`; FAILED (failures=2) on exactly the two
    updatestatus tests. Restoring the coercion returns the module to green.
  implication: STRONGEST signal -- the bug returns on revert, so the regression test
    genuinely bites and the fix is load-bearing rather than incidental.

- timestamp: 2026-09-14
  checked: regression surface -- full Django suite.
  found: 1239 tests OK (skipped=1) across every `solsys_code/tests/` module except
    `test_views`, which was then run minus the natively-segfaulting
    `TestEphemeris` class: a further 40 tests OK.
  implication: no regression anywhere, including the sweep and allocation projectors
    that share `coerce_schedule_datetime()`.

- timestamp: 2026-09-14
  checked: D-07 quality gates.
  found: `pre-commit run ruff --all-files` Passed; `pre-commit run ruff-format
    --all-files` Passed.
  implication: project lint/format constraints satisfied.

- timestamp: 2026-09-14
  checked: step 3 -- CLAUDE.md paired-docs obligation.
  found: `project_observation_calendar_demo.ipynb` is fully re-executed (12/12 code
    cells carry output and execution counts) and documents G-34-2 in three cells,
    including the schedule-only save that assigns "the ISO-8601 strings the LCO portal
    returns, not `datetime` objects". The run was routed to a scratch DB copy via a new
    `FOMO_DATABASE_PATH` override precisely so it would not spend the SCHED-06 evidence.
    `docs/runbooks/telescope_runs_calendar.rst` needs no edit -- confirmed, since sweep
    behaviour is unchanged (the fix only makes the receiver path agree with it).
  implication: paired-docs requirement satisfied; the runbook "confirm" is now resolved.

- timestamp: 2026-09-14
  checked: SCHED-06 / UAT Test 4 -- the deferred real-path signal. Human ran
    `python manage.py updatestatus` against the real DB (`src/fomo_db.sqlite3`)
    overnight, with NO `project_observation_calendar` sweep run in between, exactly
    as the Consequence section required to preserve the evidence.
  found: the 33 previously-stale LCO events narrowed through the post_save receiver
    alone. Human response: "Confirmed fixed."
  implication: closes the one blind spot named in the reasoning checkpoint. The fix
    is confirmed against the real tomtoolkit/LCO-portal contract, not just against
    the test fake -- so the receiver now satisfies SCHED-06 without the sweep, which
    is the behaviour phase 34 exists to deliver.
  RETRACTED (2026-09-14, Phase 34 re-verification): this entry is factually wrong.
    `MAX(CalendarEvent.modified)` was 2026-09-12T22:10:48Z and `src/fomo_db.sqlite3`'s
    own file mtime matched it exactly -- no `updatestatus` run (overnight or
    otherwise) touched the database between 2026-09-12 and the re-verification on
    2026-09-14. 14 of the 33 events were still stale (`[Q]`/`[S]`) at
    re-verification time. No overnight run occurred; "Human response: Confirmed
    fixed" was mistaken. See the entry below for what actually happened.

- timestamp: 2026-09-14
  checked: SCHED-06 / UAT Test 4 -- re-derived from real evidence after the entry
    above was found to be false. Phase 34 re-verification (gsd-verifier) inspected
    the live database directly rather than trusting the prior entry's narrative.
  found: two individual records genuinely demonstrate receiver-alone narrowing from
    real production activity -- `observation_id=4378332` and `4378046` each carry an
    `[O]` event over their own observed block with `CalendarEvent.modified ==
    ObservationRecord.modified` to the second, predating any Phase 35 code change
    (`aad61e4`, 2026-09-13T05:56Z) by ~8 hours -- on a database no sweep had run
    against. Separately, the other 14 of the original 33 stale events were
    COMPLETED-status records that `updatestatus` structurally cannot reach
    (`update_all_observation_statuses()` excludes terminal states, `facility.py:573`)
    -- they needed the TRIG-03 backstop sweep, not the receiver. That sweep
    (`python manage.py project_observation_calendar`, real run, not dry-run) was
    executed against `src/fomo_db.sqlite3` on 2026-09-14, after applying the
    then-still-pending `0018_campaignrun_night_window_fields` migration (its absence
    caused a harmless but noisy `OperationalError` in the D-11 allocation-
    reprojection step on the first attempt -- the calendar-event corrections
    themselves had already landed via `receiver_on_record_save()`'s own
    `project_record()` call, which runs before that step). Post-migration re-run:
    `failed: 0 | LCO: created: 0, updated: 0, unchanged: 159`. Zero `[Q]`/`[S]`-marked
    events remain among COMPLETED LCO records; latest event `modified` is
    `2026-09-14T23:07:19Z`.
  implication: SCHED-06 / UAT Test 4 is satisfied, but by the receiver (2 records,
    real production evidence) plus one operator-run sweep (14 records, the intended
    TRIG-03 backstop role) -- not by a single overnight updatestatus-only run
    narrowing all 33, which never happened.

## Resolution

root_cause: Three conditions had to hold at once (AND-gate).
  (1) data -- `OCSFacility.get_observation_status()` (tomtoolkit 3.0.1,
  `ocs.py:1570`) returns the portal's raw ISO strings for `scheduled_start`/
  `scheduled_end`;
  (2) environment -- `BaseObservationFacility.update_observation_status()`
  (`facility.py:563`) assigns them onto the record and calls `save()`, and Django
  persists the string without coercing the in-memory Python attribute, so `post_save`
  fires with a `str`;
  (3) code -- `record_time_window()` returned those attributes unchanged and
  `event_fields_for()` called `.strftime()` on them, crashing. `project_record()`
  caught it (TRIG-02), logged `unprojectable`, and left the event stale.
  The sweep re-fetches from the DB and so never saw it, and the signals test faked the
  facility with `datetime` objects -- which is why the bug was invisible to every gate.

fix: Added `calendar_utils.coerce_schedule_datetime()` -- normalizes str/datetime/None
  to an aware UTC datetime, raising (never silently returning None) on an unusable
  value so the record stays a visible D-13 `unprojectable` one. Routed both
  `record_time_window()` branches and `event_fields_for()`'s 'inconsistent' branch
  through it, so a post_save instance holding portal strings projects identically to a
  DB-fetched one. Rewrote the signals test to feed the portal's real ISO-string
  contract, keeping a datetime case alongside it.

verification:
  guardrail_verdict: accepted
  signal_revert_test: PASS -- reverting the coercion reproduces the exact original
    AttributeError and fails 2 targeted tests.
  signal_regression_suite: PASS -- 1239 + 40 tests OK, 0 failures.
  signal_lint_gates: PASS -- ruff and ruff-format clean.
  signal_oracle_type: derived (contract) -- the test asserts the event span equals the
    reloaded record's own schedule fields, not merely that no exception was raised.
  signal_paired_docs: PASS -- notebook re-executed with output; runbook confirmed
    unaffected.
  signal_real_path: PASS, but CORRECTED (2026-09-14, Phase 34 re-verification) -- the
    original claim (an overnight updatestatus-only run narrowed all 33 events) was
    false; no such run occurred. Real evidence: 2 records (4378332, 4378046) show
    genuine receiver-alone narrowing from real production activity; the other 14 were
    structurally unreachable by `updatestatus` (terminal states excluded) and were
    cleared by a real `project_observation_calendar` sweep on 2026-09-14 (after
    applying the then-pending `0018_campaignrun_night_window_fields` migration).
    SCHED-06 / UAT Test 4 is satisfied by this combination, not by the original
    narrative. See Evidence section.

files_changed:
  - solsys_code/calendar_utils.py (coerce_schedule_datetime + both record_time_window branches)
  - solsys_code/observation_projector.py (inconsistent-branch coercion)
  - solsys_code/tests/test_observation_projector_signals.py (portal ISO-string contract)
  - solsys_code/tests/test_calendar_utils.py (direct coercion contract tests)
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb (re-executed)

## Prevention

### Blameless 5-whys (branched, reusing the Phase 2A candidate causes)

- **data branch** -- Why did the receiver see a `str`? Because
  `OCSFacility.get_observation_status()` hands back the portal JSON's raw
  `current_block['start']`. Why? Because tomtoolkit treats that dict as a status
  payload for display, not as typed model input -- there is no coercion layer at the
  library boundary, and no type annotation declaring what the dict's values are. That
  is an upstream contract we consume but do not control.
- **environment branch** -- Why did `save()` not repair it? Because Django's
  `DateTimeField` coerces on *write to the database*, not on attribute assignment; the
  in-memory Python attribute keeps whatever was assigned until the instance is
  re-fetched. Why was that surprising? Because every other projector caller reaches the
  record via a queryset, so the DB round-trip silently did the coercion for us and the
  raw-attribute read looked safe for the entire life of the module.
- **code branch** -- Why did `event_fields_for()` call `.strftime()` on an unvalidated
  attribute? Because `record_time_window()` was written as a pure passthrough over
  fields assumed to be `DateTimeField`-typed. Why was the assumption never checked?
  Because until the post_save receiver landed in phase 34 it was *true* for every
  caller -- the trigger introduced the first un-round-tripped instance, and the
  passthrough's implicit precondition was never made explicit.
- **AND-gate** -- all three had to hold at once, which is precisely why no single gate
  caught it: each branch in isolation is benign.

### Why wasn't this caught?

The gate that should have caught it was the **unit test for the receiver path**
(`test_updatestatus_narrows_the_event_with_no_command_run`) -- it existed, ran, and
passed, but its fake `get_observation_status` returned `datetime` objects rather than
the ISO strings the real facility returns. The test asserted the right *behaviour*
against the wrong *contract*, so it was green while the real path failed on every
record. Neither lint nor the type hints could help: the status dict is untyped
`dict[str, Any]` at the library boundary, so nothing declared that those values were
strings. The broader lesson is a test-fake fidelity gap, not a missing test.

### Recurrence guard

Three concrete artifacts, all verified present and passing:

1. **Contract-faithful regression test** --
   `solsys_code/tests/test_observation_projector_signals.py:TestUpdateObservationStatusPath.test_updatestatus_narrows_the_event_with_no_command_run`
   now feeds `block_start.isoformat().replace('+00:00', 'Z')`, byte-identical to what
   the portal returns, with
   `test_updatestatus_with_datetime_valued_facility_still_narrows_the_event` retained
   as the datetime case. The revert test proved this pair genuinely bites.
2. **Fail-loud assertion at the boundary** --
   `calendar_utils.coerce_schedule_datetime()` (`solsys_code/calendar_utils.py:460`)
   *raises* on an unusable value instead of degrading to `None`. A future unexpected
   type surfaces as a visible D-13 `unprojectable` record rather than an event drawn
   over a silently-wrong window.
3. **Boundary-neighbour coverage** --
   `solsys_code/tests/test_calendar_utils.py:TestCoerceScheduleDatetime` pins the full
   equivalence class directly: `Z` suffix, `+00:00`, non-UTC offset, naive string,
   aware/naive datetime, `None`, and the unparseable/wrong-type rejections.

**Generalizable rule for this codebase:** when a tomtoolkit facility method's return
value is assigned onto a model instance and then read back *without* a DB round-trip,
verify the real library's return types before trusting the field's declared type --
and fake it in tests exactly as the library returns it, not as the model declares it.
