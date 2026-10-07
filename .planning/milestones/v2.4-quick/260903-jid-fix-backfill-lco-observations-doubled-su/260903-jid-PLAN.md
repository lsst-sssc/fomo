---
phase: quick-260903-jid
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/backfill_lco_observations.py
  - solsys_code/tests/test_backfill_lco_observations.py
  - docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb
autonomous: true
requirements:
  - SUMDUP-01
  - SUMDUP-02
  - SUMDUP-03
  - SUMDUP-04

estimate:
  tokens: 9000
  raw_tokens: 9000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - "A single invocation of backfill_lco_observations prints the final summary line exactly once, in both --dry-run and real modes, instead of the byte-identical doubled pair operators see today"
    - "call_command('backfill_lco_observations', ...) still returns the summary string -- the emission path removed is the explicit one, not the return, so any caller reading the return value keeps working"
    - "Every pre-existing test in test_backfill_lco_observations.py still passes unchanged, because Django's BaseCommand.execute() writes the returned value to whatever stdout is bound -- including a redirected StringIO passed as call_command(stdout=buf)"
    - "One test asserts the summary line occurs exactly once in captured stdout, so a future re-introduction of a second emission path fails the suite rather than only showing up on the operator's terminal"
    - "The paired demo notebook is re-executed and committed with output, and no cell records the summary line twice"
    - "docs/runbooks/telescope_runs_calendar.rst's two sample summary blocks each show the line once (verified correct at planning time -- the file is left byte-for-byte untouched)"
    - "backfill_lco_observation_records.py, its test module, and every campaign-related module are byte-for-byte unchanged"
    - "pre-commit ruff and ruff-format stay clean"
  artifacts:
    - path: "solsys_code/management/commands/backfill_lco_observations.py"
      provides: "A single summary emission path at the end of Command.handle -- the return, which Django prints"
      contains: "return summary"
    - path: "solsys_code/tests/test_backfill_lco_observations.py"
      provides: "An exactly-once occurrence assertion on the captured summary line"
      contains: "count("
    - path: "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
      provides: "Re-executed demo notebook whose recorded output shows the summary line once per cell"
      contains: "requestgroups seen"
  key_links:
    - from: "Command.handle's terminal return"
      to: "django.core.management.base.BaseCommand.execute"
      via: "execute() assigns handle()'s value to `output` and writes it to self.stdout when truthy; with the explicit write removed this is the one and only emission, and it honours a redirected stdout= stream"
      pattern: "return summary"
---

<objective>
Remove the duplicate emission of `backfill_lco_observations`'s final summary line so a single
invocation prints it once, and pin that with a test plus a regenerated demo notebook.

Purpose: operators running `python manage.py backfill_lco_observations KEY2026B-004 --dry-run`
against the real LCO portal currently see the summary printed twice, byte-identical, which reads
like the command ran twice or double-counted -- exactly the wrong signal from a line whose whole
job is to be trusted as a count.
Output: a one-line production deletion, one new test assertion, and a re-executed paired notebook.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@CLAUDE.md

# The previous fix to this same command, one hour earlier. Its decisions are LOCKED:
# the single f-string summary with per-field ternaries, the invocation-scoped dry-run target
# de-dup set, the `_changed_record_fields` shared helper, and the test suite's independent
# `_expected_summary` copy of the summary-line contract. This task must not reopen any of them.
@.planning/quick/260903-ik7-fix-backfill-lco-observations-dry-run-su/260903-ik7-SUMMARY.md

@solsys_code/management/commands/backfill_lco_observations.py
@solsys_code/tests/test_backfill_lco_observations.py
</context>

<root_cause>
Verified live at planning time (mutable-scope authority, #3786) -- both halves confirmed by
reading the actual files, not inferred from the bug report:

1. `solsys_code/management/commands/backfill_lco_observations.py` is 660 lines. The last two
   lines of `Command.handle` are, in order: an explicit write of the freshly-built `summary`
   local to `self.stdout` (line 659), then `return summary` (line 660).
2. Installed Django is 5.2.17 at
   `/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/django/core/management/base.py`.
   `BaseCommand.execute()` at lines 464-473 does `output = self.handle(*args, **options)`, then
   `if output:` ... `self.stdout.write(output)` and returns it. `Command` subclasses plain
   `BaseCommand` (line 412) and does not override `output_transaction`, so the default `False`
   applies and the returned string is written back verbatim with no SQL wrapper.

So the line is emitted once by the explicit write and once by Django. Deleting the explicit
write leaves Django's write as the single emission path -- and because `execute()` rebinds
`self.stdout` to an `OutputWrapper` around `options['stdout']` *before* calling `handle()`
(lines 454-455), Django's write lands in exactly the same redirected stream the tests already
capture. That is why removing the explicit write cannot break the existing suite.

Also verified at planning time, so the executor does not have to rediscover it:

- Every existing summary assertion in the test module is an `assertIn(...)` against
  `stdout.getvalue()` (plus one `assertEqual(stderr.getvalue(), '')`). None counts lines or
  asserts an exact full buffer, so all 24 tests keep passing after the change.
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` records the doubled line in
  three cells: the `--dry-run` cell (a bare identical pair), and the real-run and second-run cells
  (each a `stdout: <line>` print of a doubled buffer followed by Django's own bare copy).
  `grep -cF 'requestgroups seen'` over the notebook is **6** today and must become **3**.
- `docs/runbooks/telescope_runs_calendar.rst` shows the summary once in each of its two sample
  literal blocks (a real pass and a `--dry-run` pass), and its surrounding prose says nothing
  about the line being printed twice. **There is nothing to correct in the runbook** -- Task 2
  proves that by check, and leaves the file untouched.
</root_cause>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Single emission path, proven end to end by the test suite</name>
  <files>solsys_code/management/commands/backfill_lco_observations.py, solsys_code/tests/test_backfill_lco_observations.py</files>
  <behavior>
    - Existing behavior preserved: `call_command('backfill_lco_observations', '--proposal=...', stdout=buf)`
      leaves the full summary line in `buf.getvalue()`, and returns that same string to the caller.
    - New assertion: the summary line occurs exactly ONCE in `buf.getvalue()`, not twice.
    - All 24 pre-existing tests continue to pass with no edits to their assertions.
  </behavior>
  <action>
    Production side, `Command.handle` in
    `solsys_code/management/commands/backfill_lco_observations.py`: the method currently ends with
    two statements that each emit the summary. Delete the first one -- line 659, the explicit
    write of the `summary` local to `self.stdout` -- and keep line 660's `return summary` as the
    sole emission path, so Django's `BaseCommand.execute()` prints it exactly once. Change nothing
    else: not the summary f-string, not the two earlier per-request/per-group `self.stdout.write`
    calls at lines 593 and 639 (those are progress lines, not the summary), not the counters, not
    the class docstring.

    Do NOT invert this into write-and-return-`None`. `call_command()` returns whatever
    `execute()` returns, which is `handle()`'s value; the demo notebook and any future caller may
    read it, so the returned string is part of this command's contract and stays.

    Test side, `solsys_code/tests/test_backfill_lco_observations.py`: add ONE assertion to the
    existing `test_dry_run_writes_nothing_but_reports_summary` (around line 320), which already
    captures `stdout = io.StringIO()`, passes it as `call_command(..., stdout=stdout)`, and builds
    the `expected` full line via the module-level `_expected_summary(...)` helper. Immediately
    after that test's existing `assertIn(expected, stdout.getvalue())`, assert the occurrence count
    of the line in the captured buffer equals 1 -- use `_expected_summary`'s returned string as the
    needle (`stdout.getvalue().count(expected)`), so the assertion pins the whole line, not a
    fragment. Give it a short comment explaining WHY the count matters: Django's `execute()` writes
    the value `handle()` returns, so any explicit write of the same string inside `handle()` doubles
    the line on the operator's terminal.

    Do not add a new test method, do not touch `_expected_summary`, and do not modify any other
    test's assertions -- they must pass untouched, which is itself the evidence that Django's write
    reaches the redirected stream.

    Commit the production deletion and the test assertion together as one atomic fix.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_backfill_lco_observations 2>&1 | tail -5</automated>
    <automated>! grep -vE '^[[:space:]]*#' solsys_code/management/commands/backfill_lco_observations.py | grep -qF 'self.stdout.write(summary)' &amp;&amp; echo 'PASS: explicit summary write removed'</automated>
    <automated>test "$(grep -cF 'return summary' solsys_code/management/commands/backfill_lco_observations.py)" = "1" &amp;&amp; echo 'PASS: return kept as sole emission path'</automated>
    <automated>test "$(grep -cE '^[[:space:]]*self\.stdout\.write' solsys_code/management/commands/backfill_lco_observations.py)" = "2" &amp;&amp; echo 'PASS: only the two progress writes remain'</automated>
    <automated>test -z "$(git status --porcelain solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py)" &amp;&amp; echo 'PASS: sibling command untouched'</automated>
    <automated>pre-commit run ruff --all-files &amp;&amp; pre-commit run ruff-format --all-files</automated>
  </verify>
  <done>
    `handle()` ends with `return summary` and no explicit write of it; the module retains exactly
    the two progress-line `self.stdout.write` calls; the test suite is green (25 assertions'
    worth across the same 24 test methods, one method gaining the exactly-once assertion); the
    sibling command and its tests show no diff; ruff and ruff-format clean; one commit made.
  </done>
  <reversibility rating="reversible">Deleting one line and adding one assertion; a `git revert` of the single commit restores the prior behavior exactly.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Re-execute the paired demo notebook, prove the runbook needs no edit</name>
  <files>docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb</files>
  <action>
    Paired-docs house rule (CLAUDE.md, "Paired docs are part of the deliverable"):
    `backfill_lco_observations.py` pairs with
    `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`, whose committed output
    currently shows the doubled line in three cells. Re-execute it in place:

      jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb

    The notebook is self-contained -- mocked portal, hand-built fixture, no live network (it was
    regenerated successfully by quick task 260903-ik7 an hour ago). Do NOT edit its fixture,
    mocking helpers, `call_command` invocations, or cleanup cell, and do NOT add prose about the
    old doubled output; the point is that the recorded output now shows the line once. Only if a
    markdown cell makes a claim the new output contradicts should you adjust that cell, minimally.
    Commit the notebook WITH its output (pre-commit strips notebook output everywhere except
    `pre_executed/`, which is committed with output by convention).

    Runbook: the paired-docs rule also covers any affected page under `docs/runbooks/`. Check the
    `backfill_lco_observations` section of `docs/runbooks/telescope_runs_calendar.rst` (the "How do
    I backfill ObservationRecords without a campaign?" section, roughly lines 128-216) for sample
    output showing the summary twice. Planning-time inspection found each of its two sample literal
    blocks -- one real pass, one `--dry-run` pass -- showing the line exactly once, and no prose
    claiming otherwise, so the expected outcome is NO runbook change. Confirm by check and leave the
    file untouched. Change nothing else in the runbook either way.
  </action>
  <verify>
    <automated>python3 -c "
import json
nb = json.load(open('docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb'))
bad = []
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        text = o.get('text') or []
        lines = [l for l in text if 'requestgroups seen' in l]
        needles = [l.split('stdout: ')[-1].strip() for l in lines]
        if len(needles) != len(set(needles)):
            bad.append(i)
assert not bad, f'summary line still recorded twice in cell index {bad}'
print('PASS: no notebook cell records the summary line twice')
"</automated>
    <automated>test "$(grep -cF 'requestgroups seen' docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb)" = "3" &amp;&amp; echo 'PASS: 6 recorded summary lines reduced to 3 (one per summary-printing cell)'</automated>
    <automated>test "$(grep -cF 'requestgroups seen:' docs/runbooks/telescope_runs_calendar.rst)" = "2" &amp;&amp; test -z "$(git status --porcelain docs/runbooks/telescope_runs_calendar.rst)" &amp;&amp; echo 'PASS: runbook sample blocks already show the line once each; file untouched'</automated>
    <automated>test -z "$(git status --porcelain solsys_code/management/commands/backfill_lco_observation_records.py solsys_code/tests/test_backfill_lco_observation_records.py solsys_code/campaign_views.py solsys_code/campaign_forms.py solsys_code/campaign_reconciler.py solsys_code/campaign_attribution.py)" &amp;&amp; echo 'PASS: sibling command and campaign modules untouched'</automated>
    <automated>pre-commit run sphinx-build --all-files</automated>
  </verify>
  <done>
    The notebook is re-executed and committed with output; no cell's recorded output contains the
    summary line twice; the notebook's total recorded occurrences of `requestgroups seen` is 3
    (down from 6); the runbook is confirmed correct and shows no diff; sphinx-build clean.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| LCO Observation Portal -> command | Untrusted RequestGroup JSON crosses here; unchanged by this task (no parsing, counting, or network code is touched) |
| command -> operator terminal / captured stdout | The only surface this task changes: which code path emits the already-built summary string |
| command -> `call_command()` return value | Programmatic contract consumed by tests and the demo notebook; deliberately preserved |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-jid-01 | Repudiation | `Command.handle` summary emission | low | mitigate | A doubled count line misleads an operator about how much was written -- the audit value of the line is the reason for the fix. Task 1's exactly-once test assertion keeps the single emission path enforced by CI rather than by eyeball. |
| T-jid-02 | Denial of Service | `call_command()` return contract | medium | mitigate | Flipping to write-and-return-`None` would silently break any caller reading the return value (notebook, future automation). Explicitly prohibited in Task 1's action; the positive `return summary` grep gate proves the return survived. |
| T-jid-03 | Information Disclosure | summary line content | low | accept | No new information reaches stdout: the string is byte-for-byte the one already emitted, only once instead of twice. No credential, proposal secret, or portal payload is added. |
| T-jid-04 | Tampering | sibling `backfill_lco_observation_records` and campaign modules | medium | mitigate | Blast-radius containment: both tasks carry a `git status --porcelain` gate asserting the sibling command, its test module, and the four campaign modules are byte-for-byte unchanged. |
| T-jid-SC | Tampering | npm/pip/cargo installs | n/a | accept | No package installs in this task -- no dependency is added, removed, or upgraded, so the package-legitimacy gate does not apply. |
</threat_model>

<verification>
1. `python manage.py test solsys_code.tests.test_backfill_lco_observations` -- all tests pass
   (never `./manage.py`; see CLAUDE.md).
2. Neighboring regression, to prove nothing else regressed:
   `python manage.py test solsys_code.tests.test_backfill_lco_observation_records solsys_code.tests.test_sync_lco_observation_calendar`
   (skip any label that does not exist rather than inventing one).
3. `! grep -vE '^[[:space:]]*#' solsys_code/management/commands/backfill_lco_observations.py | grep -qF 'self.stdout.write(summary)'`
   -- the duplicate emission is gone.
4. `test "$(grep -cF 'return summary' solsys_code/management/commands/backfill_lco_observations.py)" = "1"`
   -- the return contract survived.
5. Notebook: the duplicate-detection python gate from Task 2 passes and
   `grep -cF 'requestgroups seen'` over the notebook is 3.
6. `test -z "$(git status --porcelain docs/runbooks/telescope_runs_calendar.rst)"` -- runbook untouched.
7. `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files`,
   `pre-commit run sphinx-build --all-files` -- all clean.
8. `python manage.py help backfill_lco_observations` still registers the command.
</verification>

<success_criteria>
- One invocation of `backfill_lco_observations` prints its summary line exactly once, in both modes.
- `call_command()` still returns that string.
- 24 pre-existing tests pass with no assertion edits; one of them additionally asserts the
  occurrence count is 1.
- The paired demo notebook is committed with re-executed output showing the line once per cell.
- The runbook is verified correct and left untouched.
- The sibling `backfill_lco_observation_records` command, its tests, and all campaign modules show
  no diff; ruff, ruff-format, and sphinx-build are clean.
- Two atomic commits (Task 1: fix + test; Task 2: notebook).
</success_criteria>

<output>
Create `.planning/quick/260903-jid-fix-backfill-lco-observations-doubled-su/260903-jid-SUMMARY.md` when done
</output>
