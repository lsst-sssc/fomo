---
phase: quick-260903-kpy
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
requirements:
  - TL-01
  - TL-02
  - TL-03
  - TL-04
  - TL-05
  - TL-06
  - TL-07

estimate:
  tokens: 45000
  raw_tokens: 45000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - "TL-01: a real sweep collects every Target it touches -- both the ones matched by Target.matches.match_fuzzy_name() and the ones newly built by _build_non_sidereal_target() -- into a TargetList named '<proposal>_targets', created if absent and reused in place if it already exists"
    - "TL-02: re-running the same sweep creates no second TargetList and adds no duplicate membership; the list's target count after run two equals its count after run one"
    - "TL-03: --target-list NAME overrides the derived name; the derived '<proposal>_targets' list is then never created"
    - "TL-04: a request the sweep skips (no id, no named target, unmappable orbital elements, or no usable instrument_type) contributes no target to the list -- collection happens strictly after every skip branch"
    - "TL-05: --dry-run performs zero database writes -- no TargetList row and no membership -- while still reporting the would-forms and a would-add count equal to what the matching real pass reports"
    - "TL-05b: two portal target names that fuzzy-match the same existing Target are collected once, not twice, in both modes -- so the dry-run count and the real count cannot diverge on an alias payload"
    - "TL-06: the summary line carries 'target list: created/reused <name>' and 'targets added to list: N' (dry-run: 'would create'/'would reuse' and 'targets would add to list: N'), appended after the existing 'block lookups failed' field in both modes"
    - "TL-06b: every pre-existing test in test_backfill_lco_observations.py still passes, with its expected summary line extended rather than weakened -- no assertIn is downgraded to a looser fragment"
    - "TL-07: the paired demo notebook is re-executed with output committed, prints the resulting list membership, and its cleanup cell deletes the TargetList it created; the runbook's backfill section documents the list behaviour, --target-list, the two new counters, and the accurate campaign-surface consequence"
    - "backfill_lco_observation_records.py, its test module, campaign_reconciler.py, campaign_utils.py, campaign_attribution.py, calendar_utils.py and models.py are byte-for-byte unchanged"
    - "pre-commit ruff, ruff-format and sphinx-build stay clean; the untracked repo-root file reqgroup_2682493.json is never staged"
  artifacts:
    - path: "solsys_code/management/commands/backfill_lco_observations.py"
      provides: "Per-sweep target collection plus the create-or-reuse TargetList step and its two summary counters"
      contains: "TargetList.objects.get_or_create"
    - path: "solsys_code/tests/test_backfill_lco_observations.py"
      provides: "Six new tests covering derived-name creation, re-run idempotence, the override flag, dry-run zero-writes, dry-run would-reuse, and skip exclusion; every existing expected summary line extended"
      contains: "targets added to list"
    - path: "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
      provides: "Re-executed demo showing the new counters in both passes and the resulting list membership, with cleanup extended to the list"
      contains: "TargetList"
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      provides: "Operator documentation of the TargetList behaviour, --target-list, the two counters, and the campaign-surface note"
      contains: "target-list"
  key_links:
    - "Collection point (real mode) sits after the existing 'if is_new_target: target.save()' block, so every collected Target already has a pk -- matched targets have one, and a new one was just saved"
    - "Collection point (dry-run mode) sits inside the existing 'if dry_run:' branch after the target_verb decision, so it mirrors the dry_run_target_names_seen mechanism it is modelled on"
    - "Both collection points are downstream of every 'skipped += 1; continue' branch, which is what makes TL-04 true"
    - "The post-loop list step reads options['target_list'] (argparse dest for --target-list) with the derived '<proposal>_targets' as the fallback"
    - "The dry-run branch of the list step uses TargetList.objects.filter(name=...).exists() only -- the single line that makes TL-05 true"
---

<objective>
Collect every Target a `backfill_lco_observations` sweep touches into a `TargetList` named
`<proposal>_targets` (override with `--target-list NAME`), created on first run and reused
on every re-run, with the membership add being naturally idempotent and `--dry-run`
reporting the outcome without writing anything.

Purpose: after a sweep, an operator has a single named handle on "everything this proposal
observed" -- usable directly in the TOM's target-list surfaces -- instead of having to
reconstruct the set from ObservationRecords by hand.

Output: the extended command plus its tests (Task 1), and the re-executed paired notebook
plus the updated runbook section (Task 2).
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@/home/tlister/git/fomo_devel/CLAUDE.md
@.planning/STATE.md
@.planning/quick/260903-ik7-fix-backfill-lco-observations-dry-run-su/260903-ik7-SUMMARY.md
@.planning/quick/260903-jid-fix-backfill-lco-observations-doubled-su/260903-jid-SUMMARY.md
@solsys_code/management/commands/backfill_lco_observations.py
@solsys_code/tests/test_backfill_lco_observations.py
</context>

<planning_facts>
Verified by direct inspection at planning time (MUTABLE-SCOPE AUTHORITY #3786) -- do not
re-derive these, and do not contradict them:

1. **`TargetList` model** (`tom_targets/models.py:167-202` in the installed
   `tom_toolkit`): `name = models.CharField(max_length=200)`, `targets =
   models.ManyToManyField(BaseTarget)`, plus auto `created`/`modified`. So the M2M
   attribute is `targets`, `TargetList.objects.get_or_create(name=...)` is valid, and
   `target_list.targets.add(*targets)` is the correct call. `max_length=200` against real
   proposal codes (12-20 characters) plus the 8-character `_targets` suffix leaves enormous
   headroom -- **no truncation helper is needed or wanted here**, unlike `_group_name()`'s
   `max_length=50` squeeze. An over-long operator-supplied `--target-list` value is operator
   error, not a case to defend against.

2. **Import**: `TargetList` is importable from `tom_targets.models` alongside `Target`; the
   sibling command already imports it that way.

3. **Existing house pattern for membership**: `backfill_lco_observation_records.py:259`
   uses `campaign.targets.add(target)`. The set-like semantics of that M2M add are what
   make re-runs non-duplicating.

4. **Campaign surfaces** (needed for the runbook sentence, and it is *not* what the task
   brief assumed): `CampaignListView` (`solsys_code/campaign_views.py:223-233`) is
   deliberately scoped to `TargetList.objects.filter(campaign_runs__isnull=False)` -- a
   `TargetList` with no `CampaignRun`s **does not appear on the campaign list page**. But
   `CampaignRunSubmissionForm.campaign` (`solsys_code/campaign_forms.py:23`) *is*
   `TargetList.objects.all()`, so a backfill-created list **does** appear in that
   campaign picker. The runbook sentence must say this accurately (see Task 2) rather than
   the simpler "it will appear on the campaign views with no runs", which is false for the
   list page.

5. **Every existing test** in `test_backfill_lco_observations.py` runs with
   `--proposal=LCO2026A-003`, so the derived list name throughout the existing suite is
   `LCO2026A-003_targets`. There are 24 tests today; ~10 of them build an exact expected
   summary line via the module-level `_expected_summary()` helper, and
   `test_dry_run_reports_accurate_counters_end_to_end` builds its expected line as a raw
   string literal instead of via the helper -- that literal must be updated too.

6. **Notebook fixture**: `DEMO_PROPOSAL = 'BACKFILL-DEMO-2026A'`, so the demo's derived list
   name is `BACKFILL-DEMO-2026A_targets`. The notebook has 17 cells; cell 8 is the dry-run
   pass, cell 10 the real pass, cell 12 the inspection cell, cell 16 the cleanup cell.

7. **Runbook section**: "How do I backfill ObservationRecords without a campaign?" spans
   lines 128-215 of `docs/runbooks/telescope_runs_calendar.rst`, ending immediately before
   the "How do I sync Gemini queue observations?" heading. It already carries a bullet list
   of "how it differs", a dry-run paragraph, a "Scheduled times" paragraph, and two sample
   summary literal blocks (one real, one dry-run).
</planning_facts>

<locked_design>
These decisions are settled. Implement them as written; do not re-litigate during execution.

**D-01 -- Collection container and key.** One dict, `collected_targets`, declared beside the
existing counters in `handle()`: keys are `target.pk` in a real run (the target is always
saved by the time it is recorded) and, under `--dry-run` only, the target's name when it has
no pk because it would-be-new and was never saved. Values are the `Target` instances, so the
post-loop step can pass them straight to `.add()` without a second query. This mirrors the
existing `dry_run_target_names_seen` per-invocation de-duplication mechanism rather than
inventing a parallel one -- but it is a **separate** container: `dry_run_target_names_seen`
holds would-be-new names only and guards the `targets would create` counter, whereas this
one holds every touched target and must not perturb that counter.

**D-02 -- Keying an existing target by pk in both modes.** Under `--dry-run` a *matched*
target already has a pk, so it is keyed by pk, not by name. This is what stops two portal
names that fuzzy-match the same existing `Target` from being counted twice in a dry run when
the real pass would count them once. Keying by name is the fallback for the unsaved
would-be-new case only.

**D-03 -- Two collection points, not one.** A single shared insertion point before the
`if dry_run:` branch is wrong: in a real run the new target is not saved yet there, so it
would be keyed by name and then keyed again by pk on the next request naming it, inflating
the count. Use one insertion in each branch, as specified in Task 1.

**D-04 -- N is the collected count, not the newly-created-membership count.** `targets added
to list: N` reports how many distinct targets the sweep collected and handed to `.add()`. It
is deliberately *not* "how many memberships were new", so that a dry run and the real pass
over the same portal payload report the same N -- the dry-run parity convention this command
has followed since the ik7 fix. A second identical run therefore reports the same N again
while the list's membership count stays put; those are two different numbers and both tests
assert the right one.

**D-05 -- The list is created unconditionally in a real run**, even when the sweep touched
zero targets, because `get_or_create` runs before `.add()` regardless. Rationale: uniform
reporting, and the blast radius of an accidental empty list is small (planning fact 4 -- an
empty list cannot reach the campaign list page). Do not add a "skip if empty" guard.

**D-06 -- Summary fields are appended at the end**, after the existing `block lookups
failed` field, in that order: `target list: <verb> '<name>'` then the count field. Do not
reorder or re-group the existing fields.

**D-07 -- Locked by the two prior quick tasks; regressing any of these is a failure**: the
shared `_changed_record_fields` four-field comparison helper, the per-invocation
de-duplication of `targets would create`, `_resolve_schedule()` returning the embedded-block
flag, `block lookups failed: n/a (dry-run)`, and `handle()` returning the summary with **no**
explicit `self.stdout.write(summary)` call.
</locked_design>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Collect touched targets into the TargetList, both modes, with tests</name>
  <files>solsys_code/management/commands/backfill_lco_observations.py, solsys_code/tests/test_backfill_lco_observations.py</files>
  <behavior>
    New tests to add (six), all using `NonSiderealTargetFactory` only, the existing
    `make_request` patch and mocked `get_observation_status`, and no live network:
    - Derived-name creation: one RequestGroup with two requests, one naming the existing
      `Didymos` and one naming an absent `2026 AB1` carrying complete elements. A real run
      creates `LCO2026A-003_targets` whose membership is exactly those two targets, and the
      summary reports the created verb with the count 2. This is the test that proves both
      a matched target and a newly created one are collected (TL-01).
    - Re-run idempotence: the same payload run twice. Exactly one `TargetList` exists, its
      target count after run two equals its count after run one, and run two's summary
      reports the reused verb with the same count as run one (TL-02, D-04).
    - Override: the same payload with `--target-list=SweepList`. A list named `SweepList`
      exists with the expected membership and no list named `LCO2026A-003_targets` exists
      at all; the summary names `SweepList` (TL-03).
    - Dry-run zero writes: a dry run over the same payload leaves `TargetList.objects`
      empty and asserts the exact full summary line via the extended `_expected_summary()`
      helper, showing the would-create verb and the would-add count matching what the real
      pass reports for the same payload (TL-05, TL-06).
    - Dry-run would-reuse: a real run followed by a dry run. The dry run's summary shows
      the would-reuse verb, exactly one `TargetList` still exists, and its membership count
      is unchanged by the dry run (TL-05).
    - Skip exclusion: one RequestGroup with three requests -- (a) an absent target
      `2026 AB1` with complete elements, (b) an absent target named `Unmappable Object`
      built with a non-orbital-elements target type so it is skipped at the target step,
      and (c) a request naming the existing `Didymos` whose sole configuration has had its
      `instrument_type` emptied after `_request()` built it, so it is skipped at the
      parameters step. A real run's list membership is exactly `2026 AB1`, the summary
      reports skipped 2 and the added count 1 (TL-04). Note that (c) is the sharp case: its
      target *is* matched before the skip, so it proves the collection point sits after the
      parameters check and not before it.
  </behavior>
  <action>
Extend `solsys_code/management/commands/backfill_lco_observations.py`.

Import: extend the existing `from tom_targets.models import Target` line to also import
`TargetList`.

Argument: add a `--target-list` option to `add_arguments`, `required=False`, help text
saying it overrides the derived `<proposal>_targets` name for the TargetList the sweep
collects into. Argparse exposes it as `options['target_list']`.

Collection container: beside the existing counter locals in `handle()` (next to
`dry_run_target_names_seen`), declare `collected_targets: dict[Any, Target] = {}` with a
comment explaining D-01 and D-02 -- keyed by pk in a real run and, under dry-run only, by
name for a would-be-new target that was never saved and so has no pk; an already-existing
target is keyed by pk in both modes so two portal names fuzzy-matching one Target are
collected once rather than twice.

Collection point, real branch (D-03): immediately after the existing
`if is_new_target: target.save(); targets_created += 1` block and immediately before the
`record, record_created = ObservationRecord.objects.get_or_create(` call, record the target
by its pk. Every target reaching this line has a pk -- a matched one already had it, a new
one was just saved.

Collection point, dry-run branch (D-03): inside the existing `if dry_run:` block, after the
`is_new_target` / `dry_run_target_names_seen` verb decision and before the
`self.stdout.write(f'Would {target_verb} target ...` call, record the target keyed by its pk
when it has one and by `target.name` when it does not. Do **not** add the target's name to
`dry_run_target_names_seen` -- that set guards a different counter and must keep holding
would-be-new names only.

Both insertion points are downstream of every `skipped += 1; continue` branch already in the
loop, which is what makes TL-04 true. Do not move either one earlier.

Post-loop list step: after the request-group loop closes and before the `summary = (`
assignment, resolve `list_name` from `options.get('target_list')` falling back to
`f'{proposal}_targets'`, and set the added count to the length of the collection dict. Under
dry-run, decide reuse-vs-create with `TargetList.objects.filter(name=list_name).exists()` and
perform no write of any kind (T-kpy-01). Otherwise call
`TargetList.objects.get_or_create(name=list_name)` unconditionally (D-05), add the collected
target instances to its `targets` manager in one unpacked `.add()` call, and derive the
reuse flag from the created flag `get_or_create` returned. Add a short comment noting the
M2M add is set-like, which is what makes a re-run non-duplicating.

Summary: compute a verb local before the f-string -- the dry-run pair is the would-forms and
the real pair is the past-tense forms -- then append two fields to the end of the existing
summary f-string after `block lookups failed` (D-06): the list field rendering the name with
`!r` so it appears single-quoted, then the count field whose label is the would-add form
under dry-run and the added form otherwise. Follow the existing per-field inline-ternary
style for the count label; keep the verb as a precomputed local rather than nesting a
ternary inside a ternary.

Docstrings: extend the module docstring's opening sentence to say the command also collects
every touched Target into a TargetList; add a paragraph to the `Command` class docstring
describing the create-or-reuse behaviour, the derived name, the override flag, the fact that
skipped requests contribute nothing, and that a dry run reports the outcome without creating
the list; extend `handle()`'s one-line summary sentence to mention the collection. Google
style, single quotes, 120 columns. Do not touch any other docstring text -- in particular
leave the fallback-schedule caveat paragraph exactly as it stands.

Then extend `solsys_code/tests/test_backfill_lco_observations.py`.

Import `TargetList` alongside `Target` from `tom_targets.models`.

Extend the module-level `_expected_summary()` helper with three keyword parameters --
the list name defaulting to `'LCO2026A-003_targets'`, a reuse flag defaulting to False, and
the added count defaulting to 0 -- and append the two new fields to the string it builds,
spelling both label pairs out literally exactly as the helper already spells every other
label. The helper must stay independent of the command module: never import from it, never
derive a label from it. Then update **every** existing call site with explicit correct
values rather than leaning on the defaults -- several tests run a real pass before their dry
run, so their list is already present and their reuse flag is True, and each test's added
count is the number of distinct targets its payload touches, which is frequently not the
same as its `targets` (would-create) argument. Also update the raw expected-summary string
literal in `test_dry_run_reports_accurate_counters_end_to_end`, which does not go through
the helper. Do not weaken any existing assertion: every exact-line `assertIn` stays an
exact-line `assertIn`, and the exactly-once `count(expected)` assertion added by the prior
quick task stays exactly as it is.

Add one assertion to the existing `test_dry_run_writes_nothing_but_reports_summary` proving
no TargetList row exists after the dry run, alongside its sibling no-writes assertions.

Add the six new tests described in the behavior block above. Keep the existing fixture
helpers (`_request`, `_request_group`, `_configuration`, `_page_response`) as the way every
payload is built; for the emptied-instrument case, build the request with `_request()` and
then clear its first configuration's instrument type on the returned dict rather than adding
a new parameter to the helper.

Commit the command and the tests together as one atomic commit.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_backfill_lco_observations</automated>
    <automated>CMD=solsys_code/management/commands/backfill_lco_observations.py; test "$(grep -v '^\s*#' $CMD | grep -c "'--target-list',")" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'from tom_targets.models import Target, TargetList')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'TargetList.objects.get_or_create')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'TargetList.objects.filter(name=list_name).exists()')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'target list: ')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'targets would add to list')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'targets added to list')" -eq 1 && echo GATES-OK</automated>
    <automated>CMD=solsys_code/management/commands/backfill_lco_observations.py; test "$(grep -v '^\s*#' $CMD | grep -c 'return summary')" -eq 1 && test "$(grep -v '^\s*#' $CMD | grep -c 'self.stdout.write(summary)')" -eq 0 && test "$(grep -v '^\s*#' $CMD | grep -c 'compare_schedule=embedded')" -eq 1 && echo D07-LOCKS-OK</automated>
    <automated>test "$(grep -c 'def test_' solsys_code/tests/test_backfill_lco_observations.py)" -eq 30</automated>
    <automated>python manage.py test solsys_code.tests.test_backfill_lco_observation_records solsys_code.tests.test_sync_lco_observation_calendar</automated>
    <automated>test -z "$(git status --porcelain solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py solsys_code/campaign_reconciler.py solsys_code/campaign_utils.py solsys_code/campaign_attribution.py solsys_code/calendar_utils.py solsys_code/models.py)" && echo BLAST-RADIUS-CLEAN</automated>
    <automated>test "$(git diff --cached --name-only | grep -c 'reqgroup_2682493.json')" -eq 0 && echo UNSTAGED-OK</automated>
    <automated>pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
  </verify>
  <done>
A real sweep creates or reuses `<proposal>_targets` and its membership is exactly the
distinct targets the sweep touched, matched and newly created alike; a second identical run
adds no duplicate membership and creates no second list; `--target-list NAME` overrides the
name; a skipped request contributes nothing; a dry run writes nothing at all yet reports the
would-forms and a would-add count equal to the real pass's; the summary carries both new
fields in both modes; all 30 tests pass; the neighbouring 58-test LCO/calendar regression
passes; the sibling command, its tests, and the campaign/CalendarEvent modules are
untouched; ruff and ruff-format are clean; `reqgroup_2682493.json` is unstaged.
  </done>
  <reversibility rating="reversible">Pure additive code plus tests, revertable by a single git revert; the only persistent side effect is TargetList rows in a local dev database, which are deletable.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Re-execute the paired demo notebook and update the runbook section</name>
  <files>docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <precondition>Task 1 is committed, `jupyter` is on PATH, and the local dev database `src/fomo_db.sqlite3` is migrated -- this notebook is DB-dependent and writes real rows before cleaning them up.</precondition>
  <action>
Notebook (`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`). This is a
CLAUDE.md paired-docs deliverable, not optional polish. Read the cells before editing; the
fixture, the mocking helpers and the `call_command` invocations all stay as they are.

- Intro markdown (cell 0): where it enumerates what the command creates, add the TargetList
  collection so the intro is not silently stale.
- Dry-run markdown (cell 7): add a sentence explaining that the dry-run pass now also reports
  which TargetList it would create or reuse and how many targets it would add, and writes no
  list.
- Dry-run code cell (cell 8): extend the existing `from tom_targets.models import Target`
  line to also import `TargetList`, and after the dry-run call add a check in the same
  `print('PASS: ...')` style cell 12 already uses, showing that no list named
  `f'{DEMO_PROPOSAL}_targets'` exists after the dry run. Use a print, never an `assert` --
  `nbconvert --execute` aborts the whole notebook on an exception, and a stale row from an
  interrupted earlier run would then break the docs build rather than just reading oddly.
- Inspection markdown/code (cells 11 and 12): print the resulting list -- its name and the
  sorted names of its members -- so the reader sees that both the newly created demo target
  and any matched target ended up in it. Keep it in the existing print-a-labelled-line style.
- Cleanup cell (cell 16): delete the TargetList named `f'{DEMO_PROPOSAL}_targets'` alongside
  the existing group/record/target deletions, and add a matching "remaining" print. This
  matters: without it every notebook run leaves a list behind in the dev database, and that
  list is reachable from the campaign picker described below.
- Regenerate with `jupyter nbconvert --to notebook --execute --inplace` and commit the
  notebook **with** its output, per the `pre_executed/` convention.

Runbook (`docs/runbooks/telescope_runs_calendar.rst`), section "How do I backfill
ObservationRecords without a campaign?" (lines 128-215) and **that section only** -- do not
touch the sibling command's section above it or the Gemini section below it.

- Add a bullet to the existing "How it differs" list describing the TargetList collection:
  every target the sweep touches, matched or newly built, is collected into a list named
  `<proposal>_targets`, created on the first run and reused on every re-run, with re-runs
  never duplicating a membership. Say explicitly that a skipped request contributes nothing.
- Document `--target-list NAME` as the override for the derived name, in the same paragraph
  style as the existing `--username` paragraph, and note there is no way to opt out of the
  collection.
- Extend the dry-run paragraph to say a dry run reports which list it would create or reuse,
  and how many targets it would add, without creating the list -- so an operator can see a
  name collision with an existing list *before* anything is written.
- Update **both** sample summary literal blocks so each shows the two new fields at the end
  of the line in its correct label form, consistent with the counts already shown in that
  block.
- Add one sentence noting the campaign-surface consequence, stated accurately per planning
  fact 4: a `TargetList` is also what FOMO's campaign surfaces treat as a campaign, so a
  backfill-created list shows up in the campaign picker on the run submission form even
  though it has no runs; it does **not** appear on the campaign list page, which only shows
  lists that have at least one campaign run. Do not write the simpler claim that it appears
  on the campaign views -- that is false for the list page and was corrected at planning time.

Commit the notebook and the runbook together.
  </action>
  <verify>
    <automated>jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb</automated>
    <automated>NB=docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb; test "$(grep -c 'targets added to list' $NB)" -ge 1 && test "$(grep -c 'targets would add to list' $NB)" -ge 1 && test "$(grep -c 'TargetList' $NB)" -ge 2 && echo NOTEBOOK-GATES-OK</automated>
    <automated>RB=docs/runbooks/telescope_runs_calendar.rst; test "$(grep -c 'target-list' $RB)" -ge 1 && test "$(grep -c 'targets added to list' $RB)" -eq 1 && test "$(grep -c 'targets would add to list' $RB)" -eq 1 && echo RUNBOOK-GATES-OK</automated>
    <automated>test "$(git diff -- docs/runbooks/telescope_runs_calendar.rst | grep -c 'name-prefix')" -eq 0 && echo SECTION-SCOPE-OK   # run before staging</automated>
    <automated>pre-commit run sphinx-build --all-files</automated>
    <automated>python manage.py test solsys_code.tests.test_backfill_lco_observations</automated>
    <automated>test -z "$(git status --porcelain solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py solsys_code/campaign_reconciler.py solsys_code/campaign_utils.py solsys_code/campaign_attribution.py solsys_code/calendar_utils.py solsys_code/models.py)" && echo BLAST-RADIUS-CLEAN</automated>
    <automated>test "$(git diff --cached --name-only | grep -c 'reqgroup_2682493.json')" -eq 0 && echo UNSTAGED-OK</automated>
  </verify>
  <done>
The notebook is re-executed and committed with output showing both new summary fields in the
dry-run and real passes, printing the resulting list's membership, and cleaning the list up
at the end; the runbook's backfill section documents the collection, the override flag, the
two counters in both sample blocks, and the accurate campaign-surface consequence; no other
runbook section is in the diff; the Sphinx build passes; the sibling command, its tests and
the campaign modules remain untouched.
  </done>
  <reversibility rating="reversible">Documentation-only; revertable by git revert with no runtime effect.</reversibility>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| LCO portal -> command | Untrusted remote JSON: target names and orbital elements become Target field values and, now, TargetList membership |
| Operator CLI -> command | `--proposal` and `--target-list` are operator-supplied strings that become a persisted `TargetList.name` |
| `--dry-run` contract | The boundary between "reports what would happen" and "writes"; operators rely on it for unattended-safe inspection |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-kpy-01 | Tampering | dry-run branch of the post-loop list step | high | mitigate | The dry-run path uses `TargetList.objects.filter(name=list_name).exists()` only -- no `get_or_create`, no `.add()`. Pinned by two tests asserting `TargetList.objects` is empty after a dry run and that a dry run following a real run leaves the membership count unchanged, plus a grep gate that `TargetList.objects.get_or_create` appears exactly once in the module. |
| T-kpy-02 | Tampering | `--target-list NAME` colliding with an existing campaign list | medium | mitigate | `get_or_create(name=...)` will silently reuse a real campaign's TargetList if the name matches, adding sweep targets into it. Mitigation is disclosure before the write: the summary distinguishes the created verb from the reused verb, and `--dry-run` reports the would-reuse verb before anything is written. Covered by the dry-run would-reuse test; the runbook's dry-run paragraph tells the operator to look for exactly this. |
| T-kpy-03 | Tampering | Portal-supplied target names reaching TargetList membership | low | accept | Membership is only ever added for targets the command already creates or matches via the pre-existing `match_fuzzy_name` path; this task adds no new trust in portal data beyond what the command already extends. All persistence is through the Django ORM -- no raw SQL, no string-interpolated query. |
| T-kpy-04 | Information disclosure | Summary line printing the list name | low | accept | The emitted name is either operator-supplied or derived from the operator-supplied proposal code; it carries no credential and no portal secret. Consistent with the existing summary, which already echoes counts derived from portal data. |
| T-kpy-05 | Denial of service | Unbounded `.add()` on a very large sweep | low | accept | A single bulk `.add()` of the collected instances is one query regardless of sweep size; the sweep's own per-request portal I/O dominates by orders of magnitude. |
| T-kpy-SC | Tampering | Supply chain | n/a | accept | No npm/pip/cargo install of any kind is performed by this task -- `TargetList` comes from the already-installed `tom_toolkit` dependency -- so the package-legitimacy gate does not apply and no `[ASSUMED]`/`[SUS]` checkpoint is required. |
</threat_model>

<verification>
1. `python manage.py test solsys_code.tests.test_backfill_lco_observations` -- 30 tests pass.
2. `python manage.py test solsys_code.tests.test_backfill_lco_observation_records solsys_code.tests.test_sync_lco_observation_calendar` -- the 58-test neighbouring regression passes, proving no collateral damage.
3. `python manage.py help backfill_lco_observations` -- registers, and now lists 5 command-specific flags (the 4 existing plus `--target-list`).
4. `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`, `pre-commit run sphinx-build --all-files` -- all clean.
5. `git status --porcelain` on `backfill_lco_observation_records.py`, its test module, `campaign_reconciler.py`, `campaign_utils.py`, `campaign_attribution.py`, `calendar_utils.py` and `models.py` -- empty.
6. `git log --oneline -2` -- exactly two commits, one per task, in order.
7. `reqgroup_2682493.json` is still untracked and was never staged: `git status --porcelain reqgroup_2682493.json` reports it as untracked and it appears in no commit's file list.
</verification>

<success_criteria>
- Every Target a sweep touches -- matched or newly created -- lands in `<proposal>_targets`, and nothing a skipped request touched does.
- Re-running is genuinely idempotent: one list, no duplicated memberships, a reused verb in the summary.
- `--target-list NAME` overrides the derived name and nothing else changes.
- `--dry-run` writes nothing and reports would-forms whose counts equal the matching real pass's.
- Both new summary fields appear in both modes, appended after the existing fields, with every pre-existing exact-line assertion extended rather than loosened.
- The paired notebook and the runbook section are updated in the same change, not as follow-up.
- All D-07 locked behaviours from the two prior quick tasks survive untouched.
</success_criteria>

<output>
Create `.planning/quick/260903-kpy-collect-every-target-touched-by-a-backfi/260903-kpy-SUMMARY.md` when done
</output>
