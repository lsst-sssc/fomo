---
phase: quick-260722-twe
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/backfill_lco_observation_records.py
  - solsys_code/tests/test_backfill_lco_observation_records.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "--create-missing-targets creating a brand-new field Target sets epoch/pm_ra/pm_dec/parallax from the request target dict's epoch/proper_motion_ra/proper_motion_dec/parallax when present"
    - "A brand-new field Target whose request dict omits those keys has all four fields set to None (unchanged from prior behavior)"
    - "Reusing an existing field Target never overwrites its epoch/pm_ra/pm_dec/parallax, even when the incoming request carries values"
    - "All four scenario tests from 260722-tkt (flag-off, create-new, reuse-existing, dry-run) still pass unchanged"
  artifacts:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py
  key_links:
    - "LCO wire-format keys (epoch/proper_motion_ra/proper_motion_dec/parallax) map to Target fields (epoch/pm_ra/pm_dec/parallax) with no unit conversion"
---

<objective>
Extend `backfill_lco_observation_records --create-missing-targets` so that when it builds a
brand-new SIDEREAL field Target, it also carries over `epoch`, `proper_motion_ra`,
`proper_motion_dec`, and `parallax` from the request's `configuration.target` dict when those
keys are present — mapping them onto `Target.epoch` / `Target.pm_ra` / `Target.pm_dec` /
`Target.parallax` (LCO API units match the Target field units exactly, so no conversion).

Purpose: the real LCO API target dict carries more than name/ra/dec; the 260722-tkt flag only
captured name/ra/dec, dropping proper-motion/epoch/parallax metadata that the model can hold.
Output: strictly additive extraction and pass-through, no migration, no behavior change to the
no-flag or reused-target paths.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@solsys_code/management/commands/backfill_lco_observation_records.py
@solsys_code/tests/test_backfill_lco_observation_records.py

Follow-up to quick task 260722-tkt (commit 73581b0), which added the opt-in
`--create-missing-targets` flag. This extends only the brand-new-Target build path.

Field name / unit mapping (verified — no unit conversion needed):
- LCO `epoch` (Julian years) -> Target.epoch
- LCO `proper_motion_ra` (mas/yr) -> Target.pm_ra
- LCO `proper_motion_dec` (mas/yr) -> Target.pm_dec
- LCO `parallax` (mas) -> Target.parallax

`Target.epoch`/`pm_ra`/`pm_dec`/`parallax` are all `null=True`, so no migration is needed.
Per CLAUDE.md: use plain English in docstrings (no "upsert"); the campaign's own
moving-object fixture stays `NonSiderealTargetFactory`; field Targets under test remain
`type=SIDEREAL` by design.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extract epoch/proper-motion/parallax and pass them to newly-built field Targets</name>
  <files>solsys_code/management/commands/backfill_lco_observation_records.py</files>
  <behavior>
    - Building a new field Target from a request dict carrying epoch/proper_motion_ra/proper_motion_dec/parallax sets Target.epoch/pm_ra/pm_dec/parallax to the matching values.
    - Building a new field Target from a request dict that omits those keys leaves all four as None.
    - Reusing an existing Target (existing is not None branch) leaves its epoch/pm_ra/pm_dec/parallax untouched.
  </behavior>
  <action>
    Extend the `RequestTargetInfo` dataclass (currently `name`, `ra`, `dec`) with four new
    fields: `epoch: float | None`, `pm_ra: float | None`, `pm_dec: float | None`,
    `parallax: float | None`. Update the class docstring to mention plainly that it also
    carries epoch, proper motion, and parallax when the request supplies them.

    In `_request_target_info()`, from the same `target` dict already read for name/ra/dec,
    also read `target.get('epoch')`, `target.get('proper_motion_ra')`,
    `target.get('proper_motion_dec')`, and `target.get('parallax')`, and pass them into the
    returned `RequestTargetInfo` as `epoch=`, `pm_ra=`, `pm_dec=`, `parallax=` respectively.
    Note the wire-key rename: LCO `proper_motion_ra`/`proper_motion_dec` map to the dataclass
    `pm_ra`/`pm_dec`. Extend the docstring's existing "or None if absent" sentence to cover the
    new fields (plain English — "epoch, proper motion, and parallax").

    In `_resolve_or_build_field_target()`, only in the `existing is None` branch (the newly
    built Target), pass `epoch=target_info.epoch, pm_ra=target_info.pm_ra,
    pm_dec=target_info.pm_dec, parallax=target_info.parallax` alongside the existing
    `ra=`/`dec=` kwargs to the `Target(...)` constructor. Leave the `existing is not None`
    branch exactly as-is — a reused Target's metadata is never overwritten.

    Do not touch `extra_params` / `v_magnitude` (no matching Target field). No migration.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_backfill_lco_observation_records 2>&1 | tail -5</automated>
  </verify>
  <done>RequestTargetInfo has the four new fields; _request_target_info reads them with correct wire-key mapping; new-Target build passes them through; reuse branch unchanged; existing 260722-tkt tests still pass.</done>
</task>

<task type="auto">
  <name>Task 2: Extend test helper and add coverage for the new fields, then run quality gates</name>
  <files>solsys_code/tests/test_backfill_lco_observation_records.py</files>
  <action>
    Thread the LCO wire-format keys through the test request builders. Add optional
    `epoch=None`, `pm_ra=None`, `pm_dec=None`, `parallax=None` params to `_configuration()`
    and embed them in the `target` dict under the LCO wire keys `epoch`,
    `proper_motion_ra`, `proper_motion_dec`, `parallax` (only when not None, mirroring how
    `ra`/`dec` are conditionally added — matching the real payload shape). Thread the same
    four params through `_request()` and `_field_request()` so a field request can carry
    them. Keep the wire keys distinct from the Target model's `pm_ra`/`pm_dec` names.

    Add three tests:
    - Flag on, brand-new field name, request target dict includes
      epoch/proper_motion_ra/proper_motion_dec/parallax -> assert the created Target has
      `epoch`, `pm_ra`, `pm_dec`, `parallax` set to the matching values (verifies correct
      field-name mapping, not a naive attribute copy — use distinct non-zero values per field
      so a mis-map would fail).
    - Flag on, brand-new field name, request target dict OMITS those keys -> assert the
      created Target has all four as None (regression guard).
    - Flag on, reuse case: pre-create the field Target elsewhere without epoch/pm/parallax,
      have the incoming request carry values, and assert the reused Target's four fields are
      untouched (still None) — reuse never overwrites.

    Keep all four 260722-tkt scenario tests (flag-off, create-new, reuse-existing, dry-run)
    passing unchanged. Then run the quality gates on touched files:
    `ruff check . --fix` and `ruff format .`.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_backfill_lco_observation_records 2>&1 | tail -5 &amp;&amp; ruff check solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py &amp;&amp; ruff format --check solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py</automated>
  </verify>
  <done>New helper params thread wire keys through; three new tests pass; all prior tests pass; ruff check and ruff format --check both clean on the two touched files.</done>
</task>

</tasks>

<verification>
- `./manage.py test solsys_code.tests.test_backfill_lco_observation_records` — all tests pass (four 260722-tkt scenarios + three new).
- `ruff check .` and `ruff format --check .` clean on the two touched files.
- No new migration files created (fields are already `null=True` on the model).
</verification>

<success_criteria>
- Brand-new field Target created via --create-missing-targets carries epoch/pm_ra/pm_dec/parallax when the request supplies them, mapped from the LCO wire keys.
- Same path with those keys omitted leaves all four None.
- Reused Target's metadata is never overwritten.
- Prior behavior for name/ra/dec/type unchanged; no migration.
</success_criteria>

<output>
Create `.planning/quick/260722-twe-extend-backfill-lco-observation-records-/260722-twe-SUMMARY.md` when done
</output>
