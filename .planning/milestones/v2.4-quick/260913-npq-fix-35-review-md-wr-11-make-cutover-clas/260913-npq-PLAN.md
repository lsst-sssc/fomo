---
phase: quick-260913-npq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/management/commands/cutover_classical_allocations.py
  - solsys_code/tests/test_cutover_classical_allocations.py
  - docs/runbooks/telescope_runs_calendar.rst
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb  # conditional -- only if Task 2's cell audit finds a committed cutover cell whose recorded output this change alters; see Task 2 step 3
autonomous: true
requirements:
  - WR-11

estimate:
  tokens: 50000
  raw_tokens: 50000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "Two blank-url legacy events in the same Source line: group whose derived observing nights are the same produce exactly ONE re-keyed event: the first claimant is converted, the second is reported and left byte-identical (url='', no CalendarEventMeta companion row), and the command exits non-zero"
    - "A legacy event whose derived ALLOC:{run_pk}:{night} url is ALREADY held by a different CalendarEvent row (the import-ran-first case, where the rewritten load_telescope_runs already created the run and its nights) is reported and left byte-identical rather than re-keyed onto the taken url"
    - "Both collisions are reported under their own named reason category -- distinct from the generic 'other' unexpected-error bucket -- so the summary breakdown and the CommandError message name a collision as a collision and the operator knows the action is to delete or re-attribute the duplicate row"
    - "After any invocation over a colliding fixture, every non-blank ALLOC:-prefixed url in the CalendarEvent table is still unique -- the command never writes a second row onto a url another row already holds"
    - "--dry-run previews the same outcome with no writes: it detects both collision kinds, counts a collided event as unexplained rather than as a would-be re-key, and its 'events re-keyed:' count equals what the real run then performs on the same fixture"
    - "Every group that has no colliding nights converts exactly as it did before -- the existing test classes in test_cutover_classical_allocations.py stay green, unmodified"
    - "The module docstring's list of reasons an event can be left unexplained names the collision case, and the operator runbook's cutover section lists it with the operator action and pins the cutover-before-load_telescope_runs ordering"
    - "python manage.py test solsys_code.tests.test_cutover_classical_allocations passes in full, and pre-commit's pinned ruff and ruff-format are clean on both changed Python files"
  artifacts:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py
    - docs/runbooks/telescope_runs_calendar.rst
  key_links:
    - "tom_calendar.CalendarEvent.url is URLField(blank=True, default='') with NO unique constraint, and adding one is out of scope (third-party model). The database therefore cannot refuse a duplicate -- the only place the duplicate can be stopped is this command's own per-event loop, before it writes"
    - "The damage lands downstream, not here: project_allocation() reads a night with CalendarEvent.objects.filter(url=url).first(), so it manages ONE of the two rows and never sees the other; because the url IS in active_urls the convergence step will not remove the orphan either. A duplicate written here is permanent and invisible to every counter"
    - "The check must raise BEFORE update_calendar_event_key_and_fields()/adopt_event_into_run() inside the existing per-event savepoint -- that ordering is what keeps the losing event byte-identical (D-18), exactly as the WR-08 window check already does"
    - "A night may only be added to the claimed set AFTER its savepoint commits: an event that failed to re-key for some other reason never wrote its url, so a later event must still be free to claim that night"
    - "--dry-run has run=None whenever the group's CampaignRun does not exist yet; no ALLOC url can already exist for a run with no primary key, so the existing-url probe is simply skipped on that path rather than guessed at"
---

<objective>
Close 35-REVIEW.md WR-11: `cutover_classical_allocations` re-keys every writable legacy
event in a group to `allocation_night_url(run, night)` without ever asking whether that url
is already taken. Two legacy events resolving to the same night (a duplicate row from a
pre-cutover re-ingest, or two schedule lines differing only in fields the identity key
ignores) both silently receive the same `ALLOC:{run_pk}:{night}` url; and if the rewritten
`load_telescope_runs` imported the same schedule file BEFORE the cutover ran, the run and
its `ALLOC:` nights already exist and the cutover re-keys stranded legacy events straight
onto urls other rows already hold. `CalendarEvent.url` has no unique constraint, so nothing
refuses the write, and `project_allocation()`'s `.filter(url=...).first()` then manages one
row and never sees the other -- permanently, invisibly.

Purpose: this command's whole contract is "what it cannot explain, it reports and leaves
byte-identical". A silently duplicated night is the one failure mode that contract cannot
absorb, because no counter in the command or the sweep ever reports it. The fix makes the
collision an explainable, named, operator-actionable outcome instead of a write.

Output: a dedicated collision reason category, an in-run night-claim check and an
existing-url probe on both the real and the `--dry-run` path, a regression test class
covering all three scenarios, an updated module docstring, and the runbook's operator
reason list plus an explicit cutover-before-import ordering sentence. No behaviour change
for any group whose nights are distinct and free.

Scope is fixed and NOT to be revisited during execution: implement BOTH in-run collision
detection AND the existing-url check inside the cutover. Do NOT add a `UniqueConstraint` to
the third-party `tom_calendar` `CalendarEvent` model -- the review records that as worth
doing some day and explicitly out of this scope.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@/home/tlister/git/fomo_devel/CLAUDE.md
@/home/tlister/git/fomo_devel/.planning/STATE.md
@/home/tlister/git/fomo_devel/solsys_code/management/commands/cutover_classical_allocations.py
@/home/tlister/git/fomo_devel/solsys_code/tests/test_cutover_classical_allocations.py

Read only the WR-11 section of
`.planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md`, and only the cutover
section (roughly lines 858-945) plus the "A reported unexplainable event during the
classical cutover" troubleshooting entry (roughly lines 1369-1392) of
`docs/runbooks/telescope_runs_calendar.rst`. Do not read either file whole.

Interfaces already in this module, needed by Task 1 (do not re-derive them):

- `allocation_night_url(run, night) -> str` (`solsys_code/allocation_projector.py`) returns
  `f'ALLOC:{run.pk}:{night.isoformat()}'` -- it needs a run with a primary key.
- `observing_night(start_time, site_zone) -> datetime.date`
  (`solsys_code/telescope_runs.py`), the site-local noon-anchored night derivation already
  used inside the per-event savepoint.
- `_mark_unexplained(events, category, reason)` -- the closure defined in `Command.handle`;
  it appends to `unexplained` and increments `reason_counts[category]`.
- The summary loop prints `_REASON_LABELS[category]` for every category present in
  `reason_counts`, so a new category MUST have a `_REASON_LABELS` entry or the command
  raises `KeyError` while reporting.
- The module currently imports no name from `datetime`; a `set[date]` annotation needs
  `from datetime import date` added to the imports.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Detect ALLOC: key collisions in the cutover and report them instead of writing a duplicate</name>
  <files>solsys_code/management/commands/cutover_classical_allocations.py, solsys_code/tests/test_cutover_classical_allocations.py</files>
  <behavior>
    Write these three tests FIRST, in a new test class in
    `solsys_code/tests/test_cutover_classical_allocations.py`, following the existing
    fixture patterns (`CutoverClassicalAllocationsTestBase`, `_make_legacy_event`,
    `_make_three_night_group`, `_THREE_NIGHT_LINE`, `_THREE_NIGHTS`, `call_command` with
    `StringIO` buffers, `assertRaises(CommandError)`). Confirm they fail against the current
    command before implementing.

    - Test 1 (in-run collision): two blank-url legacy events sharing `_THREE_NIGHT_LINE`
      and the SAME `start_time` (so both derive the same observing night, both inside the
      run's window). Expect: exactly one of the two ends up with an `ALLOC:` url and the
      other still has `url=''` with no `CalendarEventMeta` row for it;
      `CommandError` is raised; the losing event's pk and the collision reason appear on
      stderr; the summary names the collision category with a count of 1; and the set of
      non-blank `ALLOC:`-prefixed urls in `CalendarEvent` has no repeats (compare a list of
      those urls against a set built from it).
    - Test 2 (import-ran-first): build a `CampaignRun` whose `source_identifier` equals
      `_source_identifier(parse_run_line(_THREE_NIGHT_LINE), _THREE_NIGHTS[0],
      _THREE_NIGHTS[-1])` -- the key the cutover itself will derive -- together with a
      `CalendarEvent` already holding `allocation_night_url(run, night)` for one of the
      three nights, as the rewritten `load_telescope_runs` plus the allocation projector
      would have left it. Add one blank-url legacy event for that same night and source
      line. Expect: the legacy event is reported under the collision reason and still has
      `url=''`; `CalendarEvent.objects.filter(url=<that alloc url>).count() == 1`; the
      pre-existing event is untouched. Note the cutover will UPDATE that run in place from
      the parsed line (so its window covers the night and the WR-08 containment check
      passes) -- the collision, not the window, is what must be reported.
    - Test 3 (dry-run preview): `--dry-run` over Test 1's fixture. Expect: no
      `CampaignRun` row and no `CalendarEvent` re-keyed (both still `url=''`); the summary
      reports the collision category with a count of 1 and `events re-keyed: 1`, i.e. the
      same number the real run then performs on that fixture; `CommandError` is still
      raised.

    Use `NonSiderealTargetFactory` if any test ever needs a `Target` (never
    `SiderealTargetFactory`) -- as with the existing classes, none of these should need one.
  </behavior>
  <action>
    Implement in `solsys_code/management/commands/cutover_classical_allocations.py`.

    1. Add a dedicated reason category alongside the existing ones: a constant
       `_KEY_COLLISION = 'key_collision'` placed after `_FOREIGN_ATTRIBUTION` and before
       `_OTHER` in the reason vocabulary, with a matching `_REASON_LABELS` entry reading
       along the lines of "derived ALLOC: night is already claimed by another event". Justify
       the dedicated category over folding into `_OTHER` in a short comment citing WR-11:
       the module's contract is that every reason is printed so an operator can find the
       row, and a collision calls for a different operator action (find and delete or
       re-attribute the duplicate row) than a generic unexpected error, so it needs its own
       name in the summary breakdown and in the CommandError message.

    2. Add a small private exception class (for example `_KeyCollision`, with a docstring)
       raised when a night is already claimed. Raising rather than branching lets the check
       live INSIDE the existing per-event savepoint -- which is what guarantees the losing
       event stays byte-identical -- while a `except _KeyCollision` clause placed BEFORE the
       existing broad `except Exception` clause routes it to the collision category instead
       of `_OTHER`.

    3. Add a per-group set of already-claimed nights, created fresh before the
       `for event in writable_events` loop (it is per group, never shared across groups --
       two different runs legitimately own nights of the same date, since the url is keyed
       by run primary key as well). Move `site_zone = ZoneInfo(site.timezone)` above the
       `if dry_run:` branch, since both paths now need it; the blank-timezone case is
       already rejected earlier in the group loop.

    4. Real path, inside the per-event savepoint, AFTER the WR-08 window-containment check
       and BEFORE `update_calendar_event_key_and_fields()`:
       - if the derived night is already in this group's claimed set, raise the collision
         exception with a message naming the night, along the lines of "a second event
         already claims night {night}";
       - otherwise compute the url with `allocation_night_url(run, night)` and look for
         another row already holding it
         (`CalendarEvent.objects.filter(url=url).exclude(pk=event.pk).first()`); if one
         exists, raise the collision exception with a message naming both the night and the
         holding row's primary key so the operator can go straight to it in the admin.
       Record the night in the claimed set only AFTER the savepoint's `with` block exits
       successfully, next to the existing `events_rekeyed += 1`: an event whose re-key
       failed for some other reason wrote no url, so a later event must still be free to
       claim that night.

    5. Dry-run path: replace the unconditional `events_rekeyed += len(writable_events)`
       with a read-only loop over `writable_events` that derives each event's night, applies
       the identical two checks, counts a collided event via `_mark_unexplained(...,
       _KEY_COLLISION, ...)` instead of as a would-be re-key, and increments
       `events_rekeyed` only for an event that claims its night cleanly. Writes nothing.
       When the group's `CampaignRun` does not exist yet, `run` is None on this path: skip
       the existing-url probe in that case (a run with no primary key can have no `ALLOC:`
       url in the table, so there is nothing to probe) and keep the in-run claimed-night
       check, which needs no run at all. Guard the night derivation with the same
       report-not-crash treatment the real path gives it.

       Deliberate non-goal, to be recorded in the SUMMARY rather than fixed here: the
       dry-run path still does not apply the WR-08 window-containment check, so a dry run
       can over-count a would-be re-key for an out-of-window event. That gap predates WR-11
       and is out of this task's scope -- do not extend the dry-run path beyond the two
       collision checks.

    6. Module docstring: extend the sentence listing the reasons an event or group can be
       left unexplained so it also names the collision case -- a second event claiming a
       night this run has already claimed, or a night whose `ALLOC:` url another
       `CalendarEvent` already holds (which is what an import of the same schedule file
       running before the cutover leaves behind) -- stating that only the first claimant of
       a night is re-keyed and the rest are reported untouched, and that this exists because
       `CalendarEvent.url` carries no unique constraint for the database to enforce.

    Leave every other behaviour alone: the WR-07 all-foreign guard, the WR-08 window check,
    the WR-09 status guard, the group savepoint and the exit-code contract are unchanged,
    and every existing test class in the module must stay green without edits.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_cutover_classical_allocations</automated>
    <automated>grep -n "_KEY_COLLISION" solsys_code/management/commands/cutover_classical_allocations.py</automated>
  </verify>
  <done>
    The new test class passes and every pre-existing class in
    `test_cutover_classical_allocations.py` still passes unmodified. A colliding fixture
    leaves exactly one re-keyed event, one reported event with `url=''` and no meta row, a
    non-zero exit, and no repeated `ALLOC:` url anywhere in the table. `--dry-run` reports
    the same collision and the same would-be re-key count while writing nothing.
  </done>
</task>

<task type="auto">
  <name>Task 2: Document the new reason and pin the cutover-before-import ordering in the runbook, and audit the paired notebook</name>
  <files>docs/runbooks/telescope_runs_calendar.rst, docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb</files>
  <action>
    1. In `docs/runbooks/telescope_runs_calendar.rst`, section "How do I run the one-time
       classical cutover?" (around lines 858-945): add the collision to the operator-facing
       list of what the command deliberately leaves alone, in the same operator language as
       the neighbouring reasons, and state the operator action for it explicitly -- find the
       duplicate row in the Django admin and delete it or re-attribute it, then re-run the
       command. Make clear that the first event to claim a night is still converted; only
       the extra claimants are reported. Name the reason token exactly as the command's own
       summary line prints it, in literal markup, the way this section already names
       ``foreign_attribution`` -- an operator matching a printed breakdown line against the
       runbook needs the same word in both places.

    2. In the same section, add one explicit sentence pinning the ordering: run
       `cutover_classical_allocations` BEFORE the first rewritten `load_telescope_runs`
       import of the same schedule file. An import that runs first creates the
       `CampaignRun` and its `ALLOC:` nights itself, so the cutover will then find those
       urls already held and refuse -- reporting the stranded legacy events as collisions
       rather than re-keying them onto taken urls. Place it with the numbered step list so
       an operator reading the sequence cannot miss it.

    3. Also extend the troubleshooting entry "A reported unexplainable event during the
       classical cutover" (around lines 1369-1392) so its **Cause** list and its **Fix**
       cover the collision reason too, consistent with the section above.

    4. Paired-docs audit (CLAUDE.md), mandatory and explicit: open
       `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and inspect the
       committed cutover cells and their recorded output (the dry-run and real
       `call_command('cutover_classical_allocations', ...)` cells and the assertions that
       consume their stderr). Decide whether this change alters any recorded output: it does
       only if that database's run actually contained a collision. Evidence already on file
       from quick task 260913-ng8: those cells exercise only the `no_source_line` reason (a
       junk pk=334 row), and both the dry run and the real run record `events re-keyed: 9`
       out of 10 candidates -- equal counts, so no night was claimed twice. Confirm that
       independently rather than assuming it.
       - If the audit finds NO output the new category would change: change nothing in the
         notebook and record in the SUMMARY that no re-execution was needed, naming the
         evidence (which cells were inspected, which reason categories their output
         contains, and the matching dry-run/real re-key counts).
       - If the audit DOES find affected output: re-execute with
         `jupyter nbconvert --to notebook --execute --inplace
         docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and commit the
         notebook with its output, per the pre-executed-notebook convention.
  </action>
  <verify>
    <automated>grep -c 'key_collision' docs/runbooks/telescope_runs_calendar.rst</automated>
    <automated>pre-commit run sphinx-build --all-files</automated>
    <human-check>The runbook's cutover section names the collision reason with its operator action, and states in the step sequence that the cutover runs before the first rewritten load_telescope_runs import of the same file.</human-check>
  </verify>
  <done>
    The runbook's cutover section and its troubleshooting entry both cover the collision
    reason and its operator action, and the section states the cutover-before-import
    ordering with the reason it matters. The notebook audit is completed and its outcome
    (re-executed, or unchanged with the evidence named) is recorded in the SUMMARY.
  </done>
</task>

<task type="auto">
  <name>Task 3: Run the project's quality gates on the changed files</name>
  <files>solsys_code/management/commands/cutover_classical_allocations.py, solsys_code/tests/test_cutover_classical_allocations.py</files>
  <action>
    Run pre-commit's pinned ruff and ruff-format on the changed Python files (pre-commit
    pins ruff to the version the repo enforces; an unpinned `ruff` on PATH can report
    findings the enforced gate does not have -- D-07). Apply any fix or reformat they make,
    then re-run the targeted test module so the final tree is the one that passed.

    Use `python manage.py test ...` -- never `./manage.py`, and never the full unfiltered
    suite (`test_views.TestEphemeris` segfaults in native ASSIST, and importing
    `solsys_code.ephem_utils` downloads ~1.6 GB of SPICE kernels).

    Commit convention for this quick task: `fix(quick-260913-npq): ...` for the command and
    test change, `docs(quick-260913-npq): ...` for the runbook (and the notebook, if the
    audit required re-execution). Every commit message ends with the trailer lines:
    `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01MBQXMHen2DLP5owNpwRMKr`.
  </action>
  <verify>
    <automated>pre-commit run ruff --files solsys_code/management/commands/cutover_classical_allocations.py solsys_code/tests/test_cutover_classical_allocations.py</automated>
    <automated>pre-commit run ruff-format --files solsys_code/management/commands/cutover_classical_allocations.py solsys_code/tests/test_cutover_classical_allocations.py</automated>
    <automated>python manage.py test solsys_code.tests.test_cutover_classical_allocations</automated>
  </verify>
  <done>
    Both pre-commit hooks pass on both changed Python files and the targeted test module
    passes on the final tree.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| legacy `CalendarEvent` rows -> cutover | Row content (description, `start_time`) is pre-existing operator/import data this command re-parses and trusts; a duplicate or drifted row is exactly the untrusted input WR-11 is about |
| cutover -> `CalendarEvent.url` keyspace | The command writes a key the database enforces no uniqueness on; the invariant exists only in application code |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-quick-npq-01 | Tampering | `Command.handle` per-event re-key loop | medium | mitigate | Refuse to write a night already claimed in this run or already held in the table; report it under its own reason category and leave the event byte-identical (Task 1) |
| T-quick-npq-02 | Information disclosure | collision reason strings on stderr | low | accept | Reason strings name only a `CalendarEvent` primary key, a night date and an `ALLOC:` url -- the same class of internal identifier the command already prints for every other reason, on an operator-facing console |
| T-quick-npq-03 | Denial of service | extra `CalendarEvent.objects.filter(url=...)` probe per event | low | accept | One indexed-column equality probe per writable event on a one-time command over at most a few hundred legacy rows |
| T-quick-npq-SC | Tampering | npm/pip/cargo installs | high | mitigate | No new dependency is added by this plan; nothing is installed, so the package-legitimacy gate has nothing to audit |
</threat_model>

<verification>
- `python manage.py test solsys_code.tests.test_cutover_classical_allocations` passes, new
  class included, every pre-existing class unmodified and green.
- `pre-commit run ruff --files <the two changed .py files>` and
  `pre-commit run ruff-format --files <same>` are clean.
- The runbook renders and states both the new reason (with its operator action) and the
  cutover-before-import ordering.
- The paired-notebook audit outcome is recorded in the SUMMARY either way.
</verification>

<success_criteria>
A colliding night is reported, never written: only the first claimant of a night is re-keyed,
every other claimant keeps `url=''` and no meta row, no `ALLOC:` url appears twice in the
table, and `--dry-run` predicts exactly that with no writes. The operator can tell a
collision from a generic error by its own named reason, knows to delete or re-attribute the
duplicate row, and is told in the runbook to run the cutover before the first rewritten
import of the same schedule file.
</success_criteria>

<output>
Create `.planning/quick/260913-npq-fix-35-review-md-wr-11-make-cutover-clas/260913-npq-SUMMARY.md` when done
</output>
