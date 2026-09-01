---
phase: quick-260805-qdc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/models.py
autonomous: true
requirements: [T-29-19]

must_haves:
  truths:
    - "Deleting CampaignRun A leaves a calendar event whose CalendarEventMeta.run points at run B fully intact -- both the CalendarEvent row and its B attribution survive"
    - "reconcile_run(run_a) never clears CalendarEventMeta.run to None on an event currently attributed to a different run, even when that event's url sits in run A's RUN: namespace"
    - "Deleting a run still removes every event it genuinely owns inside its RUN: namespace: meta row attributed to that run, meta row with run unset, and no meta row at all"
    - "All 34 pre-existing tests in test_campaign_reconciler.py + test_reconcile_campaign_runs.py still pass unchanged, plus test_campaign_approval.py and test_campaign_attribution.py"
  artifacts:
    - path: "solsys_code/campaign_reconciler.py"
      provides: "writable_events() -- the queryset-level twin of _may_write()'s ownership rule"
      contains: "def writable_events"
    - path: "solsys_code/models.py"
      provides: "pre_delete signal narrowed from namespace-identity to the ownership rule"
      contains: "writable_events"
    - path: "solsys_code/tests/test_campaign_reconciler.py"
      provides: "Three regression tests mirroring the auditor's probes"
      contains: "class TestCrossRunOwnershipGuards"
  key_links:
    - from: "solsys_code/models.py"
      to: "solsys_code.campaign_reconciler.writable_events"
      via: "function-local (lazy) import inside the pre_delete receiver"
      pattern: "from solsys_code\\.campaign_reconciler import writable_events"
    - from: "solsys_code/campaign_reconciler.py::_detach_stale_family_events"
      to: "CalendarEventMeta.run"
      via: "queryset narrowed to rows currently attributed to this run"
      pattern: "filter\\(event__in=stale, run=run\\)"
---

<objective>
Close security finding T-29-19: two write paths added to Phase 29 in a later review-fix round
(commits 9db22f0, 8dcdf58) select calendar events by URL-namespace identity alone
(`owned_events()`), never checking whether `CalendarEventMeta.run` still points at the run doing
the writing. As a result `reconcile_run(run_a)` can silently clear a staff-confirmed Phase 28
attribution that belongs to run B, and deleting run A can hard-delete calendar events that
currently belong to run B.

Purpose: restore Phase 29's T-29-01 ownership guarantee ("the ownership rule is the first
condition checked in every write path") for the two paths that never got it, so no write path can
destroy another run's data. This is a data-integrity fix, not a new feature -- no behavior that
the runbook or the demo notebook documents changes.

Output: a shared `writable_events()` helper encoding the same ownership rule `_may_write()`
already applies to a single event; both write paths routed through it; three regression tests
mirroring the auditor's probes, proven RED before the fix and GREEN after.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

@solsys_code/campaign_reconciler.py
@solsys_code/models.py
@solsys_code/tests/test_campaign_reconciler.py
</context>

<interfaces>
Contracts the executor needs, already verified in the codebase -- do not go hunting for them:

- `CalendarEventMeta.event` is a `OneToOneField` to `tom_calendar.CalendarEvent` with
  `related_name='telescope_label_meta'` and `primary_key=True`
  (`solsys_code/models.py:26-32`). So from a `CalendarEvent` queryset the companion row is
  reachable as `telescope_label_meta__...`, and an event with no companion row matches
  `telescope_label_meta__isnull=True`.
- `CalendarEventMeta.run` is a nullable `ForeignKey(CampaignRun, on_delete=SET_NULL)` with
  `related_name='calendar_event_metas'` (`solsys_code/models.py:36-42`). A row whose `run` is
  unset means "not owned by any CampaignRun", never "touch me".
- The exact OR-of-Q ownership pattern already has precedent in this codebase:
  `campaign_attribution.orphan_calendar_events()` uses
  `Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True)`
  (`solsys_code/campaign_attribution.py:456-459`). Follow that formatting.
- `from django.db.models import Q` is already imported at `campaign_reconciler.py:40` -- no new
  import needed there.
- `_may_write(event, run)` (`campaign_reconciler.py:164-180`) is the single-event form of the rule
  being added at queryset level: when a companion row exists AND its `run` is set, ownership is
  exact-match only; when there is no companion row or its `run` is unset, the run may write the
  event if the event's `url` lives in that run's `RUN:` namespace.
- `owned_events(run)` (`campaign_reconciler.py:111-118`) is ALSO used as a read-only counting
  helper by `test_campaign_approval.py` (lines 253, 407, 437, 459, 466) and by the pre-executed
  demo notebook. Its semantics must NOT change -- add a new function beside it.
- Test fixtures: `CampaignReconcilerTestBase` (`test_campaign_reconciler.py:27-63`) provides
  `self.campaign`, `self.ground_site` (F65, Australia/Sydney), `self.satellite_site` and
  `self._make_run(**overrides)`. `_make_run` defaults to an approved, ground-sited,
  single-night (2026-08-01) run with `telescope_instrument='FTN/MuSCAT3'`. Two runs in one test
  need distinct `telescope_instrument` values (see `test_campaign_reconciler.py:257-262`) because
  of the `unique_campaign_run` natural-key constraint.
- Per CLAUDE.md: if any new test needs a `Target`, use
  `tom_targets.tests.factories.NonSiderealTargetFactory`, never `SiderealTargetFactory`. The
  tests in this plan need only `TargetList` + `CampaignRun` + `CalendarEvent`, so no Target
  fixture should be necessary.
</interfaces>

<paired_docs_assessment>
CLAUDE.md's paired-docs rule was applied up front, and the assessment is: **no doc update is
required**, evidenced rather than assumed.

- `docs/runbooks/telescope_runs_calendar.rst:310-328` describes the detach behavior strictly
  within one run ("detaches the old family's events **from the run**", after a
  `source`/`telescope_class`/`site` correction on that same run). It makes no claim about events
  belonging to a *different* run, and there is no "deleting a run deletes its calendar events"
  passage anywhere in the runbook (grep for delete/run mentions returns nothing).
- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` mentions detaching once
  (cell text "detaches (never deletes) the old family's events", line 211) -- again same-run
  scoped -- and calls `owned_events()` only as a read-only inspection helper (lines 367, 418,
  430). `owned_events()` is deliberately left byte-identical by this plan, so the notebook's
  executed output stays valid and does NOT need regenerating.

Neither artifact asserts the old, buggy cross-run behavior as correct, so per the rule's own
carve-out ("only touch them if you find they actually assert the old, buggy behavior as
correct") they stay untouched. Task 2 re-runs the two greps as a verification step so this
conclusion is re-proven at execution time rather than trusted from planning time.
</paired_docs_assessment>

<tasks>

<task type="auto">
  <name>Task 1: Add the three cross-run ownership regression tests and prove they are RED</name>
  <files>solsys_code/tests/test_campaign_reconciler.py</files>
  <action>
Add a new test class `TestCrossRunOwnershipGuards(CampaignReconcilerTestBase)` immediately after
`TestCampaignRunDeletionCascadesCalendarEvents` (which ends at line 702) and before
`TestWindowEndBeforeWindowStart`. Its class docstring should state that it covers security
finding T-29-19: the two write paths added after T-29-01 that selected events by URL namespace
alone, so a run could destroy another run's staff-confirmed attribution.

Write exactly three tests, mirroring the auditor's three probes. Do NOT modify any existing test
in this file or any production code in this task -- this task establishes RED.

Test 1, `test_deleting_a_run_never_deletes_an_event_attributed_to_a_different_run`: create
`run_a` and `run_b` (distinct `telescope_instrument` values, per the constraint noted in
`<interfaces>`). Create a `CalendarEvent` whose `url` is in run A's namespace -- use the
date-bearing form `f'RUN:{run_a.pk}:2026-08-01'` -- and a `CalendarEventMeta` for it with
`run=run_b`, standing for a staff member having re-attributed a stale event to B via Phase 28's
queue while its url string still carries A's namespace. Capture the event pk, call
`run_a.delete()`, then assert the `CalendarEvent` row still exists AND that
`CalendarEventMeta.objects.get(event_id=event_pk).run_id == run_b.pk` -- B's attribution must be
untouched, not merely surviving as a detached row.

Test 2, `test_reconcile_never_detaches_an_event_attributed_to_a_different_run`: create a
classical `run_a` with `window_start=date(2026, 8, 1)`, `window_end=date(2026, 8, 2)` and a
separate `run_b`. Reconcile `run_a` once so its two per-night events exist. Then create a
stale-family `CalendarEvent` in run A's namespace but OUTSIDE the active window -- url
`f'RUN:{run_a.pk}:2026-09-15'` -- with a `CalendarEventMeta` whose `run` is `run_b`. Call
`reconcile_run(run_a)` again and assert the stale event's companion row still has
`run_id == run_b.pk` (currently it is silently cleared to None by
`_detach_stale_family_events()`), and that the event row itself still exists.

Test 3, `test_deleting_a_run_still_deletes_the_events_it_genuinely_owns`: the
don't-regress-the-fix probe, covering the two cases the existing
`test_deleting_a_run_deletes_its_owned_calendar_events` does not. Create a queue-sourced `run`,
reconcile it (its container event gets a companion row attributed to the run), then add two more
events in the same namespace: one at `f'RUN:{run.pk}:2026-08-03'` with a `CalendarEventMeta`
whose `run` is None (a previously-detached stale-family event -- the WR-01 sweep case), and one
at `f'RUN:{run.pk}:2026-08-04'` with NO companion row at all. Capture all three pks, call
`run.delete()`, and assert all three `CalendarEvent` rows are gone and no `CalendarEventMeta`
rows remain for them. This is what stops the fix from being over-narrowed into re-introducing
WR-01's permanently-orphaned events.

Follow the file's existing style: Google-style docstring on each test explaining the scenario,
single quotes, 120-col lines, `datetime(..., tzinfo=dt_timezone.utc)` for event times (all three
names are already imported at the top of the file).
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python manage.py test solsys_code.tests.test_campaign_reconciler.TestCrossRunOwnershipGuards 2>&amp;1 | tail -40</automated>
  </verify>
  <done>
The run reports `Ran 3 tests` with exactly 2 failures: test 1 (the event attributed to run B was
deleted along with run A) and test 2 (run B's `CalendarEventMeta.run` was cleared to None). Test
3 passes already, since the current namespace-only selection is a superset of correct behavior.
The two failure tracebacks are captured verbatim in the summary as the RED proof -- if either
cross-run test passes at this point, the test does not reproduce the finding and must be
corrected before moving to Task 2.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Route both write paths through the ownership rule and turn the tests GREEN</name>
  <files>solsys_code/campaign_reconciler.py, solsys_code/models.py</files>
  <behavior>
    - `writable_events(run)` returns a namespaced event with no companion row (still writable)
    - `writable_events(run)` returns a namespaced event whose companion row has `run` unset (still writable)
    - `writable_events(run)` returns a namespaced event whose companion row points at this run
    - `writable_events(run)` EXCLUDES a namespaced event whose companion row points at a different run
    - Deleting a run deletes only what `writable_events(run)` selects (Task 1 tests 1 and 3)
    - `_detach_stale_family_events()` clears `run` only on companion rows currently pointing at this run (Task 1 test 2)
  </behavior>
  <action>
In `solsys_code/campaign_reconciler.py`, add a public `writable_events(run: CampaignRun)`
function directly after `owned_events()` (which stays byte-identical -- it is a read-only
identity query used by `test_campaign_approval.py` and the demo notebook). `writable_events()`
returns `owned_events(run)` further filtered by an OR of three `Q` terms:
`telescope_label_meta__isnull=True`, `telescope_label_meta__run__isnull=True`, and
`telescope_label_meta__run=run`. `Q` is already imported at line 40. Format the OR chain the same
way `campaign_attribution.orphan_calendar_events()` does at `campaign_attribution.py:456-459`.

Its docstring must say: this is the queryset-level twin of `_may_write()` -- namespace identity
alone is NOT ownership, because a companion row that points at a different run means a staff
member attributed that event elsewhere (Phase 28), and that attribution outranks a url string
left over from an earlier keying. Note that `owned_events()` (namespace identity) remains the
right query for read-only inspection and counting, while every write path must go through
`writable_events()`.

Narrow `_detach_stale_family_events()` (line 397-398): keep `stale = owned_events(run).exclude(
url__in=active_urls)` and change the update call to
`CalendarEventMeta.objects.filter(event__in=stale, run=run).update(run=None)`. Add a sentence to
its docstring recording why the extra `run=run` term is not redundant: without it, a stale-family
event that staff have since re-attributed to a DIFFERENT run gets its confirmed attribution
silently cleared by a reconcile of the run whose namespace the url happens to carry (security
finding T-29-19). Note it also loses nothing: rows with `run` already unset were a no-op update,
and rows with no companion row were never in the queryset.

In `solsys_code/models.py`, change the `pre_delete` receiver
`_delete_owned_calendar_events_on_campaign_run_delete` (line 355-379) to lazily import
`writable_events` instead of `owned_events` and call `writable_events(instance).delete()`. Keep
the lazy function-local import (the circular-import reason in the existing docstring still
holds). Extend that docstring: the delete is scoped to events this run may write, not merely
events carrying its url prefix -- deleting run A must never destroy an event whose companion row
attributes it to run B (T-29-19), while events with an unset or absent companion row inside A's
namespace ARE still deleted, which is what keeps WR-01's "no permanently-orphaned events" outcome
intact.

If Django rejects `.delete()` on the joined queryset, materialize the pks first
(`CalendarEvent.objects.filter(pk__in=list(writable_events(instance).values_list('pk',
flat=True))).delete()`) rather than weakening the filter -- the ownership condition itself is not
negotiable.

Finally, re-run the paired-docs check from `<paired_docs_assessment>` before finishing: grep
`docs/runbooks/telescope_runs_calendar.rst` and
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` for delete/detach/`owned_events`
mentions and confirm none of them assert the old cross-run behavior as correct. Record the
outcome in the summary. Only edit those two files if the grep contradicts the planning-time
assessment.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_attribution 2>&amp;1 | tail -20 &amp;&amp; ruff check . &amp;&amp; ruff format --check .</automated>
  </verify>
  <done>
All four test modules pass with zero failures and zero errors -- the 34 pre-existing
reconciler/command tests plus the 3 new ones, plus the approval and attribution suites that
consume `owned_events()` and `CalendarEventMeta`. `ruff check .` and `ruff format --check .` both
report clean. `git diff` shows `owned_events()` unchanged, `writable_events()` added, exactly one
line changed in `_detach_stale_family_events()`, and the signal body switched to
`writable_events`. The paired-docs grep result is recorded in the summary.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff admin/UI -> CampaignRun write paths | An authenticated staff action (a `source`/`site` correction that triggers a reconcile, or a run delete from the admin) crosses into code that writes shared, multi-owner calendar rows |
| CampaignRun A's code path -> CalendarEvent/CalendarEventMeta rows owned by run B | The shared calendar is a multi-tenant table: rows here belong to other runs, to `load_telescope_runs` ingest, or to nobody |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-qdc-01 | Tampering | `campaign_reconciler._detach_stale_family_events()` | mitigate | Narrow the bulk update to `filter(event__in=stale, run=run)` so a reconcile can only clear an attribution that currently points at the run being reconciled (Task 2); regression test 2 proves a run-B attribution survives |
| T-qdc-02 | Tampering / destruction of data | `models._delete_owned_calendar_events_on_campaign_run_delete` | mitigate | Route the cascade through `writable_events(instance)` so deleting run A can never hard-delete an event whose companion row attributes it to run B (Task 2); regression test 1 proves the run-B event and its attribution both survive |
| T-qdc-03 | Denial of service (data availability) | over-narrowing the delete signal | mitigate | Regression test 3 pins the three still-deletable cases (companion row on this run, companion row with `run` unset, no companion row) so the fix cannot silently re-introduce WR-01's permanently-orphaned calendar events |
| T-qdc-04 | Repudiation | Phase 28 staff attribution audit trail | mitigate | Both fixes preserve `is_verified`/`confirmed_by`/`confirmed_at` by never touching another run's companion row at all -- no code in this plan writes those fields |
| T-qdc-SC | Tampering | npm/pip/cargo installs | accept | No package-manager installs in this plan; no dependency is added or changed, so the supply-chain surface is unchanged |

Both findings require staff privileges to trigger, so this is a data-integrity fix rather than an
unauthenticated attack surface -- which is why it is dispositioned as a fix-now bug, not an
incident.
</threat_model>

<verification>
1. `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_attribution` -- all pass, no failures, no errors.
2. `ruff check .` and `ruff format --check .` -- clean.
3. RED-then-GREEN evidence: Task 1's captured 2-failure output, followed by Task 2's all-pass output for the same test class.
4. `owned_events()` is byte-identical in `git diff` (its read-only consumers -- `test_campaign_approval.py` and the demo notebook -- are deliberately unaffected).
5. Paired-docs grep re-run and recorded: no runbook or notebook prose asserts the old cross-run delete/detach behavior, so neither is edited and the notebook is not regenerated.
</verification>

<success_criteria>
- Deleting a CampaignRun no longer deletes calendar events attributed to a different run, and no longer clears that run's `CalendarEventMeta.run`.
- `reconcile_run()` no longer clears a `CalendarEventMeta.run` that points at a different run.
- A run's own events -- attributed, detached, or with no companion row -- inside its `RUN:` namespace are still deleted with the run (WR-01 preserved).
- Three regression tests exist that failed before the fix (2 of them) and pass after, so this class of bug cannot silently return.
- The T-29-01 ownership guarantee now holds for every write path in the reconciler, including the two added post-review.
</success_criteria>

<output>
Create `.planning/quick/260805-qdc-fix-t-29-19-phase-29-security-audit-rout/260805-qdc-SUMMARY.md` when done
</output>
