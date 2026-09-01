---
phase: quick-260722-tkt
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
    - "With --create-missing-targets, an unmatched request creates a SIDEREAL Target from the request's ra/dec, adds it to the campaign, and produces an ObservationRecord."
    - "With --create-missing-targets, an unmatched request whose field-name Target already exists elsewhere in FOMO reuses that Target (no duplicate) and only adds campaign membership."
    - "Without the flag, unmatched targets are still skipped-and-logged (unchanged default behavior)."
    - "--create-missing-targets combined with --dry-run writes zero Target rows and zero TargetList membership rows and no ObservationRecord."
  artifacts:
    - solsys_code/management/commands/backfill_lco_observation_records.py
    - solsys_code/tests/test_backfill_lco_observation_records.py
  key_links:
    - "_request_target_name (or a sibling helper) surfaces ra/dec from configuration.target alongside name."
    - "New/reused Target is inserted into targets_by_name so the normal matching/ObservationRecord flow proceeds without skipping."
---

<objective>
Add an opt-in `--create-missing-targets` flag to the `backfill_lco_observation_records`
management command. When set, requests whose target name isn't a campaign member get a
SIDEREAL field Target auto-created (or reused by name) from the request's own RA/Dec,
added to the campaign TargetList, and then processed normally instead of being skipped.

Purpose: LCO campaigns (e.g. Didymos 2026) sometimes observe pre-defined sidereal fields
(e.g. 'Didymos COJ 2026 Field #14') rather than tracking the moving-object target. These
field targets aren't campaign members, so the strict by-name match skips them. This flag
automates the data fix.

Output: extended command with the new flag + docstrings, extended test suite, clean ruff.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@CLAUDE.md
@solsys_code/management/commands/backfill_lco_observation_records.py
@solsys_code/tests/test_backfill_lco_observation_records.py
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Extend target extraction + add --create-missing-targets flag and creation flow</name>
  <files>solsys_code/management/commands/backfill_lco_observation_records.py</files>
  <action>
Implement the full opt-in flag path end-to-end:

(1) Extraction: the module-level `_request_target_name` helper only returns `name` from
the first configuration's `target` dict. Add a sibling helper (e.g.
`_request_target_info(request)`) that walks `request['configurations']`, and for the first
configuration whose `target` dict has a `name`, returns a small structure carrying `name`,
`ra`, and `dec` (read `target.get('ra')` and `target.get('dec')` from that same dict; they
are float degrees in the LCO API). Keep `_request_target_name` working (either leave it and
add the sibling, or have it delegate) so existing call sites and tests are unaffected.

(2) Argument: in `add_arguments`, add
`parser.add_argument('--create-missing-targets', action='store_true', help=...)`. Help text
must state: for a request whose target name isn't a campaign member, auto-create a SIDEREAL
Target from the request's RA/Dec (reusing an existing Target of that name if one exists
anywhere in FOMO), add it to the campaign, then process the request normally; default off;
and that combined with --dry-run it reports what would be created/added without writing.

(3) Handle flow: read `create_missing_targets = options['create_missing_targets']`. In the
per-request loop, when `target is None` (name not in `targets_by_name`): if
`create_missing_targets` is falsy, keep the exact current skip-and-log + increment behavior.
If truthy, resolve-or-create a field Target instead:
  - Look up an existing Target by that exact name anywhere in FOMO
    (`Target.objects.filter(name=<field_name>).first()`); reuse it if found, else build a
    new `Target(name=<field_name>, type=Target.SIDEREAL, ra=<ra>, dec=<dec>)`.
  - In dry-run: do NOT save the Target and do NOT add TargetList membership — write a
    stdout line describing what would happen (create-new vs reuse-existing, and add-to-campaign),
    and use the (possibly unsaved) Target only to drive the existing "Would create
    ObservationRecord" reporting line. Assert nothing is persisted.
  - Not dry-run: save the Target if newly built, add it to the campaign via
    `campaign.targets.add(target)` (membership only — do not remove it from or alter any
    other TargetLists), and insert it into `targets_by_name` so it is treated as matched.
  - Then fall through to the existing `_build_parameters` / ObservationRecord path (no skip).
Import `Target` from `tom_targets.models`. Add a counter (e.g. `created_targets`) and surface
it in the summary line. Do NOT require `ra`/`dec` to be present in the no-flag path.

(4) Docstrings: update the `Command` class docstring (the paragraph describing the
by-name-match / "skipped and logged, never guessed at" contract) to document that
--create-missing-targets opts into auto-creating SIDEREAL field Targets, and note its
--dry-run interaction. Use plain English ("create the Target if missing, otherwise reuse it")
per CLAUDE.md terminology guidance — do not write "upsert".

Do not add fenced code inside comments; keep implementation prose-guided.
  </action>
  <verify>
    <automated>python -c "import ast; ast.parse(open('solsys_code/management/commands/backfill_lco_observation_records.py').read())"</automated>
  </verify>
  <done>
The flag parses; extraction helper surfaces ra/dec; handle() creates-or-reuses a SIDEREAL
Target and adds campaign membership only when the flag is set; default and dry-run contracts
are preserved; class docstring + help text document the new flag.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add tests for the four flag scenarios</name>
  <files>solsys_code/tests/test_backfill_lco_observation_records.py</files>
  <behavior>
    - Flag off (default): an unmatched target is still skipped — no ObservationRecord, no new Target.
    - Flag on, brand-new field name never seen in FOMO: a SIDEREAL Target is created with the
      request's ra/dec, added to the campaign TargetList, and an ObservationRecord is created.
    - Flag on, field name already exists as a Target elsewhere (not yet a campaign member):
      the existing Target is reused (Target count for that name stays 1), campaign membership
      is added, and an ObservationRecord is created.
    - Flag on + --dry-run: reports intent but writes zero new Target rows and zero new
      TargetList membership rows (assert via count / membership queries) and no ObservationRecord.
  </behavior>
  <action>
Extend `test_backfill_lco_observation_records.py`. The existing `_configuration` /
`_request` helpers build a target dict as `{'name': ..., 'type': 'ORBITAL_ELEMENTS'}`; add a
way to include `ra`/`dec` on the request's `configuration.target` (extend `_configuration`
and `_request` with optional `ra`/`dec` params, or add a small field-request builder) so the
new field-target requests carry coordinates like a real sidereal LCO pointing (e.g.
name='Didymos COJ 2026 Field #14', ra=170.1, dec=-24.3, type='ICRS').

Add four tests mirroring the four behaviors above, all passing
`--create-missing-targets` (except the flag-off case). Import `Target` from
`tom_targets.models`. For the reuse case, pre-create the field Target with
`Target.objects.create(name=..., type=Target.SIDEREAL, ra=..., dec=...)` WITHOUT adding it to
the campaign, then assert `Target.objects.filter(name=...).count() == 1` after the run and
that `self.campaign.targets.filter(name=...).exists()`. For the dry-run case assert the
field-name Target does not exist and campaign membership count is unchanged and
`ObservationRecord.objects.exists()` is False.

CLAUDE.md convention: the campaign's own moving-object target (Didymos) is fixtured with
`NonSiderealTargetFactory` (already done in setUpTestData — keep it). The field Targets that
this feature creates are correctly `type=SIDEREAL` by design (fixed-sky pointings); asserting
the new code creates SIDEREAL Targets is expected and is NOT a violation of the non-sidereal
factory convention. Do not use `SiderealTargetFactory` to fixture the campaign target.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_backfill_lco_observation_records</automated>
  </verify>
  <done>All existing tests still pass and the four new flag-scenario tests pass.</done>
</task>

<task type="auto">
  <name>Task 3: Lint and format touched files</name>
  <files>solsys_code/management/commands/backfill_lco_observation_records.py, solsys_code/tests/test_backfill_lco_observation_records.py</files>
  <action>
Run `ruff check . --fix` then `ruff format .` per CLAUDE.md. Resolve any residual lint
findings on the two touched files. Confirm `ruff check .` and `ruff format --check .` are
clean.
  </action>
  <verify>
    <automated>ruff check . && ruff format --check .</automated>
  </verify>
  <done>ruff check and ruff format --check both report clean.</done>
</task>

</tasks>

<verification>
- `./manage.py test solsys_code.tests.test_backfill_lco_observation_records` passes (old + new tests).
- `ruff check .` and `ruff format --check .` are clean.
- Manual read-through confirms: no-flag path byte-for-byte preserves skip-and-log; dry-run
  writes nothing; membership add does not disturb other TargetLists.
</verification>

<success_criteria>
- `--create-missing-targets` is opt-in (default off); without it behavior is unchanged.
- With the flag, unmatched requests yield a created-or-reused SIDEREAL Target added to the
  campaign, then a normal ObservationRecord.
- Dry-run + flag persists nothing (0 Targets, 0 memberships, 0 ObservationRecords).
- Docstring and help text document the flag and its --dry-run interaction.
</success_criteria>

<output>
Create `.planning/quick/260722-tkt-add-create-missing-targets-flag-to-backf/260722-tkt-SUMMARY.md` when done.
</output>
