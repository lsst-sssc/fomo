---
phase: quick-260805-tad
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_reconciler.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
  - .planning/phases/29-the-reconciler/29-SECURITY.md
  - .planning/phases/29-the-reconciler/29-CONTEXT.md
  - .planning/phases/29-the-reconciler/29-RESEARCH.md
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [TAD-01, TAD-02, TAD-03, TAD-04]

must_haves:
  truths:
    - "An approved, queue-sourced (lco_queue/gemini_queue/eso_queue) CampaignRun with a resolved, non-satellite site gets one dip-corrected sunset-to-sunrise CalendarEvent per observing night, keyed RUN:{pk}:{date} -- identical treatment to a classical run at that same site"
    - "A run with a non-blank telescope_class still gets exactly one bare RUN:{pk} whole-window container event, whatever its source -- the genuinely site-agnostic/floating case is unchanged"
    - "A run whose resolved site is a satellite site still gets exactly one bare RUN:{pk} whole-window container event, whatever its source -- unchanged"
    - "reconcile_run() dispatch reads no CampaignRun.source value at all; the reconciler module contains no QUEUE_SOURCES reference"
    - "A queue-sourced, site-resolved run that already carries a pre-fix bare RUN:{pk} container event converges on its next reconcile: per-night events are minted, the old container event survives on the calendar un-re-keyed and un-re-timed, and its CalendarEventMeta.run is detached to None"
    - "The reconciler never creates, modifies or deletes an ObservationRecord-derived calendar event -- proven on BOTH the per-night branch and the container branch, since _may_write()'s ownership check is what provides that protection in either branch"
    - "Every pre-existing test that encoded the old source-driven container behaviour has had its assertions/fixtures corrected (not merely supplemented); test_campaign_reconciler.py, test_reconcile_campaign_runs.py and test_campaign_approval.py all pass"
    - "29-SECURITY.md's T-29-07 evidence no longer claims the per-night branch is unreachable for queue runs, stays closed with threats_open: 0, and carries a dated audit-trail entry for this correction"
    - "The paired demo notebook and the operator runbook describe the corrected three-branch dispatch, with the notebook regenerated against a scratch DB carrying real executed output"
  artifacts:
    - path: "solsys_code/campaign_reconciler.py"
      provides: "Three-branch reconcile_run() dispatch: telescope_class -> container, satellite site -> container, else -> classical per-night"
      contains: "def reconcile_run"
    - path: "solsys_code/tests/test_campaign_reconciler.py"
      provides: "Corrected queue-source expectations, the live-shaped container->per-night convergence test, and both-branch record-event non-interference"
      contains: "def test_"
    - path: "solsys_code/tests/test_reconcile_campaign_runs.py"
      provides: "Corrected command-level real-data-shape fixture and assertions"
      contains: "TestRealDataShapeScenario"
    - path: ".planning/phases/29-the-reconciler/29-SECURITY.md"
      provides: "Corrected T-29-07 evidence plus a dated audit-trail row"
      contains: "T-29-07"
    - path: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
      provides: "Paired demo notebook showing the corrected dispatch with real executed output"
      contains: "dispatch"
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      provides: "Operator-facing description of which runs get per-night entries vs. a whole-window entry"
      contains: "whole window"
  key_links:
    - from: "solsys_code/campaign_reconciler.py::reconcile_run"
      to: "_reconcile_classical_nights"
      via: "the final else branch, now also reached by every queue-sourced run with a resolved non-satellite site"
      pattern: "_reconcile_classical_nights\\(run, dry_run=dry_run\\)"
    - from: "solsys_code/campaign_reconciler.py::reconcile_run"
      to: "_reconcile_container"
      via: "telescope_class and satellite-site branches only"
      pattern: "run\\.site\\.observations_type == Observatory\\.SATELLITE_OBSTYPE"
    - from: "solsys_code/campaign_reconciler.py::reconcile_run"
      to: "_detach_stale_family_events"
      via: "convergence step that detaches a pre-fix container event when a run moves to the per-night family"
      pattern: "_detach_stale_family_events\\(run, active_urls\\)"
---

<objective>
`reconcile_run()` currently has four dispatch branches. The third one,
`elif run.source in QUEUE_SOURCES:`, is unreachable for the case it was written for and
wrong for the case it actually catches.

Proof (re-verify it against the code yourself before touching anything):
`_skip_reason()` runs first and returns `'unresolved site'` when
`run.site is None and not run.telescope_class`. So by the time dispatch runs, NOT(site is
None AND telescope_class falsy) holds. The first branch (`if run.telescope_class:`) having
already failed means telescope_class IS falsy. Together those force `run.site is not None`.
The QUEUE_SOURCES branch therefore never fires for a genuinely site-unresolved/floating run
-- it only ever fires for a queue-sourced run that already has a specific, resolved,
non-satellite site (the live case: RUN:3, ESO VLT/FORS2 at MPC 309, Cerro Paranal). Such a
run gets a blanket 00:00-23:59 UTC container event across its whole approved window, when
VLT observations can only happen during dark time at that one fixed location.

The genuinely site-agnostic case (LCO class-wide 1m0/0m4/2m0 allocations, which really can
execute at any matching site in the network) is already fully and exclusively handled by the
earlier `if run.telescope_class:` branch, which this task does not touch.

Purpose: a queue-sourced run with a fixed, resolved, non-satellite site gets per-night
dip-corrected sunset-to-sunrise events for that site, exactly like a classical run there.

Framing note that must hold in every artifact this task edits: this does NOT weaken or
violate RECON-02/RECON-03 or 26-DECISION.md's Criterion 3. The requirement -- a run with no
fixed observing site (class-wide) or no fixed horizon (satellite) gets whole-window container
treatment -- is unchanged and still implemented. What was wrong is using `source` as a proxy
for "no fixed site", when a fixed site can and does coexist with queue-scheduling.

Output: a three-branch dispatch, corrected existing tests, a live-shaped convergence proof,
a corrected T-29-07 security evidence entry, and updated paired docs.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@solsys_code/campaign_reconciler.py
@solsys_code/tests/test_campaign_reconciler.py
@solsys_code/tests/test_reconcile_campaign_runs.py
@.planning/phases/29-the-reconciler/29-SECURITY.md
@docs/runbooks/telescope_runs_calendar.rst
</context>

<facts_already_established>
Confirmed during planning -- do not re-derive these, but do sanity-check any that a change
would invalidate:

1. `QUEUE_SOURCES` has exactly two occurrences in all of `solsys_code/` and `src/`: its
   definition (`campaign_reconciler.py:77`) and the dispatch branch that reads it
   (`:497`). No test module, view, command, template or notebook code cell imports it. The
   notebook mentions it only in markdown prose (two lines), which this task rewrites anyway.
   So removing the branch leaves the frozenset with zero consumers.
2. `test_campaign_approval.py` contains no `source=` / `CampaignRun.Source.` usage at all --
   its runs take the model default and already exercise the classical per-night branch. It
   is expected to pass unchanged; run it as a regression check, do not edit it.
3. `sun_event()` costs ~0.62 s per call (measured 2026-08-05, warm; first call ~1.5 s). This
   matters: every queue fixture that flips from container to per-night multiplies its window
   length by that cost. Baseline before this change:
   `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs`
   = 46 tests in 46.7 s. Keep the post-change combined runtime under ~90 s by shrinking
   fixture windows whose length is not the point of the test (details in Task 1).
4. Model/vocabulary docstrings in `solsys_code/models.py` (`CampaignRun.Source`, lines
   108-137) make no claim about calendar dispatch -- leave that file alone.
5. `docs/design/canonical_record_spike.rst` is a historical Phase 26 spike record and is out
   of scope for CLAUDE.md's paired-docs rule (which covers the paired notebook and
   `docs/runbooks/`). Leave it unchanged.
6. Phase 29 PLAN/SUMMARY/REVIEW/VERIFICATION/UAT files are historical execution records.
   Leave them unchanged. Only `29-SECURITY.md` (a live contract, Task 3) plus dated
   forward-pointers in `29-CONTEXT.md` / `29-RESEARCH.md` get edited.
</facts_already_established>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Remove the source-driven dispatch branch and correct every test that encoded it</name>
  <files>solsys_code/campaign_reconciler.py, solsys_code/tests/test_campaign_reconciler.py, solsys_code/tests/test_reconcile_campaign_runs.py</files>
  <behavior>
    Corrected expectations (write/fix the assertions, then make them pass):
    - queue-sourced (lco_queue / gemini_queue / eso_queue) + resolved ground site -> one
      RUN:{pk}:{date} event per night, no bare RUN:{pk} event at all
    - queue-sourced + non-blank telescope_class -> still exactly one bare RUN:{pk} container
      (source is not what decides)
    - queue-sourced + satellite site -> still exactly one bare RUN:{pk} container, and
      sun_event() is still never called
    - classical + resolved ground site -> unchanged per-night behaviour
    - all five _skip_reason() outcomes -> unchanged
  </behavior>
  <action>
Part A -- the code change (`solsys_code/campaign_reconciler.py`):

Delete the `elif run.source in QUEUE_SOURCES:` branch and its two body lines from
`reconcile_run()` (currently lines 497-500), leaving three branches in this order:
`if run.telescope_class:` -> container (untouched), `elif run.site is not None and
run.site.observations_type == Observatory.SATELLITE_OBSTYPE:` -> container (untouched),
`else:` -> `_reconcile_classical_nights()`. Do not touch `_skip_reason()`, `_may_write()`,
`writable_events()`, `owned_events()`, `_detach_stale_family_events()`'s body, or either
branch helper's implementation.

Also delete the now-consumerless `QUEUE_SOURCES` frozenset (line 77) together with its D-07
comment block (lines 68-76), which asserts "the reconciler branches purely on this set" --
false after this change, and actively misleading if left behind. Fact 1 above establishes it
has no other consumer; re-confirm with a repo-wide grep before deleting.

Correct the module's own prose so it describes what the code now does. At minimum:
  - the module docstring's "Two coexisting key families" paragraph (lines 9-19), which today
    says the bare container is for "queue-scheduled, class-wide and satellite/space runs" and
    names a `source` correction as a re-classification trigger;
  - `run_container_url()`'s docstring (line 96) "(queue/class-wide/satellite branches...)";
  - `run_night_url()`'s docstring (lines 105-108) "the bare form as the queue/container family";
  - `_reconcile_container()`'s docstring (lines 252-258) "shared by queue-scheduled,
    class-wide and satellite runs";
  - `_detach_stale_family_events()`'s docstring (lines 436-440), which lists
    `source`/`telescope_class`/`site` as the fields dispatch reads.
Rewrite each to say: container for class-wide (non-blank `telescope_class`) and satellite-site
runs; per-night for every other approved, windowed run with a resolved ground site --
queue-scheduled or not. Where you cite the requirement, cite it as unchanged (RECON-02/RECON-03
and 26-DECISION.md Criterion 3 still describe the two key families; what changed is how "no
fixed site" is detected). Note the corrected detection in one sentence naming this quick task
(`260805-tad`) so the next reader can find the reasoning.

Grep-literalism warning: the Part A verify greps the module for the literal tokens
`QUEUE_SOURCES` and `run.source`. Phrase the replacement prose to avoid both literals (write
"the run's source field" rather than `run.source`), exactly as the Phase 27-04 migration-header
precedent did.

Part B -- correct `solsys_code/tests/test_campaign_reconciler.py`. Every site below builds a
queue-sourced run on the shared `_make_run()` fixture, whose default `site` is the resolved
ground Observatory F65, so each currently asserts the buggy container shape. These are stale
expectations: FIX them, do not add new tests next to wrong ones. Work through the list and
re-grep for `RUN:{run.pk}'` (bare-key) assertions afterwards to confirm none were missed:

  1. `TestQueueStage1` (lines 108-160, three tests, 15-night windows). Rewrite the class to
     assert the corrected behaviour: an LCO / Gemini / ESO queue-sourced run with a resolved
     ground site gets one date-bearing `RUN:{pk}:{date}` event per night and no bare
     `RUN:{pk}` event. Shrink each window to 2 nights (window length is not this class's
     point; 3 tests x 15 nights x 0.62 s is ~28 s of pure sun_event cost). Rename the class
     and its docstring to say what it now proves. Add one test in the same class covering the
     inverse control: a queue-sourced run that ALSO has a non-blank `telescope_class` still
     gets exactly one bare container -- the assertion that pins "telescope_class decides, the
     source value does not".
  2. `TestOwnershipScoping.test_event_owned_by_a_different_run_is_blocked_and_untouched`
     (line 255). Breaks: the clashing fixture event is keyed at the bare `RUN:{pk}`, which the
     run no longer writes, so nothing is blocked. Correct it to a single-night window with the
     clashing event keyed `RUN:{pk}:2026-08-01`; it must still report `blocked == 1` with the
     event's `title` and `modified` untouched.
  3. `TestOwnershipScoping.test_unowned_same_window_event_is_left_completely_untouched`
     (line 230). Still passes (a blank-url event with no companion row is outside
     `_adopted_event_for_night()`'s candidate query), but now runs the per-night branch over 5
     nights. Shrink its window to 2 nights and update the docstring if it implies the
     container branch.
  4. `TestContainerIdempotency` (lines 301-331, two tests). Breaks: both look up
     `CalendarEvent.objects.get(url=f'RUN:{run.pk}')`. Keep the class's purpose (RECON-01
     idempotency plus RECON-06 dry-run parity ON THE CONTAINER BRANCH) by making the fixture
     genuinely container-shaped -- set `telescope_class` (and clear `site`/`site_raw`) instead
     of setting a queue source. Assertions otherwise unchanged.
  5. `TestQueueOwnershipDoesNotTouchRecordEvents` (lines 553-624). Breaks at the
     `url=f'RUN:{run.pk}'` existence check, `CalendarEvent.objects.count() == 2` and the
     `owned_events(run).count() == 1` pair. This fixture is 29-SECURITY.md's T-29-07 evidence,
     so correct it carefully rather than deleting it: the same queue-sourced, site-resolved run
     now mints one event per night, the LCO-portal-keyed record event must STILL be untouched
     (url/title/start/end/modified identical after two passes), total `CalendarEvent` count is
     the record event plus n nights, and `owned_events(run)` returns exactly the n date-bearing
     rows. Shrink the window to 2 nights. Rename the class to
     `TestRecordEventNonInterference` (Task 2 adds its container-branch twin, and Task 3 cites
     both by name) and rewrite the docstring to say the protection comes from `_may_write()`'s
     ownership check, which applies identically in both branches.
  6. `TestReclassificationConvergence.test_reclassifying_classical_to_queue_detaches_old_per_night_events`
     (line 633). Breaks: a `source` edit no longer moves a site-resolved run between families,
     so the second reconcile now reports `unchanged` and no container appears. Correct it to
     use a trigger that genuinely reclassifies today -- set `telescope_class` on the
     already-reconciled classical run -- and rename it accordingly. The assertions it exists
     for (old per-night events still present, `CalendarEventMeta.run` cleared to None, one new
     container created) all stay.
  7. `TestReclassificationConvergence.test_stale_container_event_is_not_adopted_into_a_classical_night`
     (line 663). Breaks: the first reconcile no longer produces a container for a queue-sourced,
     site-resolved run. Correct it to create the container via a non-blank `telescope_class`,
     then clear `telescope_class` to reclassify the run into the per-night family. Everything
     it proves (a stale own-container event is never adopted into a night slot, survives
     un-re-keyed and un-re-timed, and ends detached) is unchanged and still required.
  8. `TestCampaignRunDeletionCascadesCalendarEvents.test_deleting_a_run_deletes_its_owned_calendar_events`
     (line 698). Breaks at the bare-key lookup. Correct the fixture to a `telescope_class` run
     so the container-deletion case stays covered.
  9. `TestCrossRunOwnershipGuards.test_deleting_a_run_still_deletes_the_events_it_genuinely_owns`
     (line 764). Breaks at the bare-key lookup for `container_event`. Correct the fixture to a
     `telescope_class` run. The other two tests in that class use default classical runs and
     need no change -- confirm, do not assume.
  10. `TestTelescopeInstrumentSplitOnEvents` (lines 812-864): the three container-branch tests
      (`test_container_branch_splits_the_base_fixtures_slash_delimited_value`,
      `test_plus_delimiter_splits_the_same_way`, `test_title_still_carries_the_full_combined_string`)
      break at their bare-key lookups. Correct each fixture to a `telescope_class` run so they
      keep testing the container write path. The two classical tests in that class are unaffected.

Part C -- correct `solsys_code/tests/test_reconcile_campaign_runs.py`:

  - `_seed_mixed_runs()` (lines 74-96): its `queue_run` (LCO_QUEUE, ground site, 6 nights) now
    produces per-night events. `TestIdempotency` and `TestDryRun` assert counts
    generically and should still pass -- confirm by running them, do not assume. Shrink
    `queue_run`'s window to 2 nights (it is swept up to four times per test) and correct the
    helper's docstring, which currently says "one classical multi-night run, one queue run, one
    class-wide run" as if those were three distinct calendar shapes.
  - `TestRealDataShapeScenario.test_19_run_fixture_matching_the_real_split_becomes_calendar_visible`
    (lines 214-270): breaks at `events.count() == 1` / `url == f'RUN:{run.pk}'` for the 8
    queue-sourced runs. Reshape the fixture so it stays an honest model of the real data under
    the corrected rule: keep 19 runs total and the 8/11 queue/classical split, but make 5 of the
    8 queue-sourced runs site-resolved (2-night windows -> per-night events) and the other 3
    class-wide (non-blank `telescope_class`, `site=None`, `site_raw=''` -> one bare container
    each), which mirrors the real mix of site-resolved ESO VLT queue rows alongside LCO
    class-wide allocations recorded in 29-06-SUMMARY.md. Assert per-night keys for the first
    group and a single bare key for the second; keep the `runs: 19`, `failed: 0`, `skipped: 0`,
    `blocked: 0` summary assertions and the final `CalendarEventMeta` total. Update the class
    docstring so it no longer implies queue-sourcing alone determines the shape.

Do not add the new convergence or container-twin tests here -- those are Task 2.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; ! grep -nE 'QUEUE_SOURCES|run\.source' solsys_code/campaign_reconciler.py &amp;&amp; ! grep -rn 'QUEUE_SOURCES' solsys_code/ src/ &amp;&amp; time python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs</automated>
  </verify>
  <done>Dispatch has exactly three branches and reads no source value; `QUEUE_SOURCES` is gone from the codebase; both test modules pass green with corrected (not merely added) expectations; combined runtime is under ~90 s against the 46.7 s baseline; you have read the test diff and can state, per corrected test, what its expectation changed from and to.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Prove convergence of a pre-fix container event and both-branch record-event non-interference</name>
  <files>solsys_code/tests/test_campaign_reconciler.py</files>
  <behavior>
    - A queue-sourced, site-resolved run that ALREADY has a bare RUN:{pk} container event with
      CalendarEventMeta.run pointing at it (the exact live pre-fix state of RUN:3) mints one
      per-night event per night on its next reconcile; the container event still exists, keeps
      its original url, start_time and end_time, is not the same row as any night event, and
      its CalendarEventMeta.run is now None
    - A second reconcile of that run reports unchanged for every night and writes nothing
      (no modified churn on any row)
    - The container branch (class-wide run) never creates, modifies or deletes an
      ObservationRecord-derived calendar event -- the twin of the per-night case Task 1 corrected
  </behavior>
  <action>
Add to `solsys_code/tests/test_campaign_reconciler.py`:

1. In `TestReclassificationConvergence`, a live-shaped test reproducing the RUN:3 transition
rather than assuming the existing convergence tests already cover it. Build a queue-sourced
(`CampaignRun.Source.ESO_QUEUE`) run with the fixture's resolved ground site and a 2-night
window, then hand-create the pre-fix state directly: a `CalendarEvent` keyed at the bare
`RUN:{pk}` spanning `window_start` 00:00 UTC to `window_end` 23:59 UTC, plus a
`CalendarEventMeta` whose `run` points at that run (this is what the old container branch
would have written -- do not try to produce it by reverting the code). Reconcile. Assert:
`result.created` equals the night count; one `RUN:{pk}:{date}` event exists per night; the
container row still exists with its `pk`, `url`, `start_time` and `end_time` unchanged (it was
neither deleted nor adopted-and-re-keyed); its companion row's `run_id` is None; and the
container's `pk` is not any night event's `pk`. Reconcile a second time and assert
`unchanged` equals the night count with every event's `modified` frozen.

2. Mutation probe (record the outcome in the SUMMARY, do not leave the mutation in the tree):
temporarily re-add the `elif run.source in QUEUE_SOURCES:` branch (with a local frozenset) to
`reconcile_run()`, run the new convergence test alone, and confirm it FAILS -- that is what
proves the test is a genuine guard for this change rather than something the pre-existing
convergence tests already passed. Revert the mutation and confirm the module is byte-identical
to its post-Task-1 state (`git diff` clean for `campaign_reconciler.py`) before committing.

3. Add the container-branch twin of `TestRecordEventNonInterference` (renamed in Task 1): a
class-wide run (non-blank `telescope_class`, no site) with an `ObservationRecord` linked via
`CampaignRunObservation` and an LCO-portal-keyed `CalendarEvent` built from
`record_time_window(record)`, mirroring the existing per-night case. Assert the record-derived
event's `url`, `title`, `start_time`, `end_time` and `modified` are unchanged after two
reconcile passes, and that the run's own container event coexists beside it. Use
`NonSiderealTargetFactory` for the Target, never `SiderealTargetFactory` (CLAUDE.md). The
resulting pair -- one test per branch -- is the evidence Task 3 cites for T-29-07.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; python manage.py test solsys_code.tests.test_campaign_reconciler &amp;&amp; git diff --quiet solsys_code/campaign_reconciler.py</automated>
  </verify>
  <done>The convergence test exists, passes after the fix, and was observed to fail with the old branch temporarily restored (outcome recorded in the SUMMARY); the record-event non-interference contract is asserted on both branches; `campaign_reconciler.py` carries no leftover mutation.</done>
</task>

<task type="auto">
  <name>Task 3: Correct T-29-07's evidence in 29-SECURITY.md and add dated forward-pointers to the phase-29 planning docs</name>
  <files>.planning/phases/29-the-reconciler/29-SECURITY.md, .planning/phases/29-the-reconciler/29-CONTEXT.md, .planning/phases/29-the-reconciler/29-RESEARCH.md</files>
  <action>
First verify, do not assume: T-29-07's safety property is "the reconciler never overwrites an
`ObservationRecord`-derived event". Read `_may_write()` and confirm it is still the first
condition checked in BOTH `_reconcile_container()` and `_reconcile_classical_nights()`, and
that a record-derived event (keyed at an LCO portal url, outside the run's `RUN:` namespace,
with no companion row pointing at the run) is refused by it identically in either branch. The
two tests from Task 2 (per-night case + container case) are the fixture evidence. Only if that
investigation finds the property actually broken do you reopen the threat -- it should not be.

`.planning/phases/29-the-reconciler/29-SECURITY.md`:
  - Rewrite T-29-07's "Evidence verified in code" cell (row currently at line 43). Remove the
    now-false sentence "Queue runs dispatch to the container branch only ...; the per-night
    branch is unreachable for them" and the stale `QUEUE_SOURCES at :75` line citation. Replace
    with the real mechanism: `_may_write()`'s ownership check is what refuses a record-derived
    event, and it is the first condition in both write paths, so the protection does not depend
    on which branch a given run takes. Cite the corrected/new test class and both test names
    from Tasks 1-2, and re-derive the `campaign_reconciler.py` line numbers against the
    post-fix file rather than copying the old ones. Keep the row's Status as `closed` and the
    frontmatter's `threats_open: 0`.
  - Add a row to the "Security Audit Trail" table (line 245 onward) in the existing format:
    date `2026-08-05`, threats total 23, closed 23, open 0, Run By naming quick task
    `260805-tad` and scoping it explicitly -- T-29-07 evidence correction only, prompted by the
    dispatch change; state which tests were run and that no other threat was re-scanned, with
    one sentence on why the change cannot affect the others (it removes a branch selector and
    touches no ownership, approval-gate or dry-run code path). Match the tone and specificity of
    the existing `260805-qdc` row.
  - Leave the trust-boundary table's `source / telescope_class / site drive which key family is
    written` row accurate: correct it to name `telescope_class` / `site` only.

`.planning/phases/29-the-reconciler/29-CONTEXT.md`: D-07 (line 123 onward) locks "the
reconciler's own code must branch purely on `run.source in {LCO_QUEUE, GEMINI_QUEUE}`". Do not
rewrite the decision -- append a dated forward-pointer beneath it (the same treatment 27-06
gave 26-CONTEXT.md's D-11 framing), noting that quick task `260805-tad` (2026-08-05) removed
the source-based dispatch branch after proving it could only ever fire for runs that already
had a resolved, non-satellite site; D-07's actual concern -- no free-text heuristic over
`telescope_instrument`/`site_raw` -- is preserved, since dispatch now reads only the structured
`telescope_class` and `site` fields; and the `source` data-fix D-07 scoped remains valuable for
provenance and reporting even though it no longer changes an event's window shape.

`.planning/phases/29-the-reconciler/29-RESEARCH.md`: append the same style of dated
forward-pointer to Pitfall 1 (line 426 onward), which frames `source` as the field the
queue/classical branch must read. State plainly that the RECON-02/RECON-03 requirement is
unchanged and still implemented -- what changed is how "no fixed observing site" is detected
(`telescope_class` / satellite `site`, not `source`).

Do not touch `26-DECISION.md` (its Criterion 3 states the requirement, which is unchanged), and
do not touch the Phase 29 PLAN / SUMMARY / REVIEW / VERIFICATION / UAT files (historical
records). Nowhere describe this change as violating or relaxing RECON-02 or RECON-03. Follow
CLAUDE.md's planning-doc terminology rule: plain English, no DB jargon.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; ! grep -n 'per-night branch is unreachable' .planning/phases/29-the-reconciler/29-SECURITY.md &amp;&amp; grep -q 'threats_open: 0' .planning/phases/29-the-reconciler/29-SECURITY.md &amp;&amp; grep -q '260805-tad' .planning/phases/29-the-reconciler/29-SECURITY.md &amp;&amp; grep -q '260805-tad' .planning/phases/29-the-reconciler/29-CONTEXT.md &amp;&amp; grep -q '260805-tad' .planning/phases/29-the-reconciler/29-RESEARCH.md &amp;&amp; git diff --quiet .planning/phases/26-canonical-record-spike/ 2>/dev/null || true</automated>
  </verify>
  <done>T-29-07 carries accurate evidence naming the two branch tests, stays closed with `threats_open: 0`, and the audit trail records a dated `260805-tad` entry; D-07 and Pitfall 1 each carry a dated forward-pointer that frames the change as a correction to how "no fixed site" is detected; no historical phase record and no `26-DECISION.md` content was edited.</done>
</task>

<task type="auto">
  <name>Task 4: Update the paired demo notebook and the operator runbook</name>
  <files>docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Both artifacts are in scope of CLAUDE.md's paired-docs rule (the notebook is
`campaign_reconciler.py`'s paired demo; the runbook page documents this behaviour), and both
currently describe the old four-branch dispatch as fact. Grep each first to confirm the exact
passages before editing.

Notebook (`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`) -- content edits:
  - Cell 0 (intro bullets): the "three dispatch branches ... a classically-scheduled
    multi-night run, a queue-scheduled run, and a class-wide run" bullet, and the bullet
    promising "a single bare `RUN:{pk}` row each for the queue and class-wide runs".
  - Cell 5 (markdown): rewrite the dispatch-order paragraph -- non-blank `telescope_class`
    wins first, then a satellite `site`, then everything else takes the per-night branch. Drop
    the `campaign_reconciler.QUEUE_SOURCES` reference (the frozenset no longer exists). Keep and
    correct the "why `source` matters for real data" paragraph: `source` is still the
    provenance record and still worth correcting in the admin, but it no longer changes an
    event's window shape -- a queue-scheduled run at a fixed site gets per-night entries just
    like a classical one there.
  - Cell 6 (code): keep the queue-sourced run (it is now the interesting case -- it demonstrates
    that queue-sourcing does not change the shape) but shorten its window to 2026-09-01..09-03
    so the printed output stays readable, and reword its print label/comment. Add a satellite
    run using the `satellite_site` (`X30`) the notebook already seeds but never uses, so the
    container branch's other member is actually demonstrated; the class-wide run stays as-is.
  - Cell 9 (markdown): correct the "two coexisting key families" paragraph so it matches what
    the regenerated output will show.

Notebook -- regeneration (recipe proven by quick task `260805-sgf`; it MUST run against an
empty, freshly-migrated DB, or the sweep reports every real campaign run and the committed
counters become meaningless): if `src/fomo_db.sqlite3` exists in the working tree, move it
aside first (it is gitignored -- never commit it, never delete the operator's copy); run
`python manage.py migrate` to create an empty one; run
`jupyter nbconvert --to notebook --execute --inplace reconcile_campaign_runs_demo.ipynb` with
the working directory set to `docs/notebooks/pre_executed/` (the setup cell's `parents[2]`
resolution depends on it); afterwards delete the scratch DB and move the saved copy back.
Sanity-check the regenerated output before moving on: the queue run's section must list one
date-bearing `RUN:{pk}:{date}` url per night with real dip-corrected sunset/sunrise times, the
class-wide and satellite runs must each show a single bare `RUN:{pk}` url with 00:00/23:59
spans, and the second sweep must still report `created: 0, updated: 0`. Committed with output,
per the `pre_executed/` convention.

Runbook (`docs/runbooks/telescope_runs_calendar.rst`) -- three passages are now factually wrong:
  - Lines 311-329 (what happens to an already-reconciled run's events when you correct its
    fields): the worked example "setting a genuine LCO/Gemini/ESO queue run's `source` from
    `legacy` to the correct queue value, which moves it from the per-night family to the
    whole-window one" no longer happens. Replace the example with one that does move a run
    between families today -- setting a `telescope_class` on a run that had a resolved site, or
    correcting a `site` to a satellite site -- and keep the surrounding detach-not-delete
    explanation intact.
  - Lines 548-553 ("What an operator sees on the calendar afterwards"): correct
    "a queue-scheduled or class-wide run shows a single entry spanning its whole window" to:
    any run with a resolved ground site -- queue-scheduled or classically scheduled -- shows one
    entry per observing night spanning that site's sunset-to-sunrise; only a class-wide
    allocation with no fixed site and a satellite run show a single whole-window entry. Say in
    one sentence why (a run at a fixed site can only observe during that site's dark time),
    and keep the note that these sit alongside the individual observation entries the sync
    commands create.
  - Lines 555-566 (the Telescope/Instrument split paragraph): it lists "queue-scheduled,
    class-wide and satellite runs" as the entries that self-heal on the next sweep because their
    whole-window entry is rewritten every time. Drop queue-scheduled from that list -- a
    site-resolved queue run's per-night entries now follow the same create-only field rule as
    any other per-night entry.
  - Add one sentence, in the skip/failure discussion around lines 542-546, noting that a
    queue-scheduled run at a site with no `timezone` set now fails per-night sun calculation the
    same way a classical run there does (it used to bypass that calculation entirely), pointing
    at the existing "Observatory missing timezone" troubleshooting entry.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel &amp;&amp; ! grep -rn 'QUEUE_SOURCES' docs/ &amp;&amp; ! grep -n 'queue-scheduled or class-wide run' docs/runbooks/telescope_runs_calendar.rst &amp;&amp; python -c 'import json,re; nb=json.load(open("docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb")); out="".join("".join(o.get("text",[])) for c in nb["cells"] for o in c.get("outputs",[])); sec=out.split("--- Queue run")[1].split("---")[1]; urls=re.findall(r"url=.(RUN:[^\x27]+)", sec); assert urls and all(re.fullmatch(r"RUN:\d+:\d{4}-\d{2}-\d{2}", u) for u in urls), urls; assert "created: 0, updated: 0" in out; print("notebook queue-run output OK:", urls)' &amp;&amp; sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build</automated>
  </verify>
  <done>The notebook's prose describes the three-branch dispatch and its committed output shows the queue-sourced run with per-night date-bearing keys plus container entries for the class-wide and satellite runs; the runbook's three stale passages are corrected and Sphinx builds clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| reconciler -> shared `CalendarEvent` table | The reconciler writes rows on a calendar it shares with hand-created entries, `load_telescope_runs` ingest and the LCO/Gemini sync commands. This change alters which rows it writes for one class of run. |
| `CampaignRun` row state -> projection decision | `telescope_class` / `site` (no longer `source`) select the key family; no HTTP input crosses here. |
| pre-fix container events already on the calendar -> post-fix convergence | Existing rows written by the old branch must converge without data loss. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-TAD-01 | Tampering | `_reconcile_classical_nights` now reached by queue-sourced runs | mitigate | `_may_write()` remains the first condition in the per-night write path, unchanged by this task; Task 2 asserts an `ObservationRecord`-derived LCO-portal-keyed event is untouched after two passes on BOTH branches. |
| T-TAD-02 | Tampering | Convergence of a pre-fix bare `RUN:{pk}` container event | mitigate | `_detach_stale_family_events()` detaches, never deletes, and `_adopted_event_for_night()`'s blank-url filter keeps the stale container from being adopted into a night slot; Task 2's live-shaped test asserts the container survives with its url/start/end intact and `run` cleared to None. |
| T-TAD-03 | Tampering | Cross-run attribution during convergence | mitigate | `writable_events()` / the `run=run` filter in the detach step are untouched by this task; `TestCrossRunOwnershipGuards` stays green (Task 1 verify). |
| T-TAD-04 | Repudiation | 29-SECURITY.md carrying false evidence for a closed high-severity threat | mitigate | Task 3 corrects T-29-07's evidence to the real mechanism, cites the two branch tests, and adds a dated audit-trail row scoping the re-verification. |
| T-TAD-05 | Denial of Service | Per-night expansion multiplies `sun_event()` calls for queue-sourced runs (sweep and test runtime) | accept | Measured at ~0.62 s per night per run; the sweep is operator-invoked, never request-triggered, and per-run failures are already isolated. Test fixtures are shrunk in Task 1 to hold the suite under ~90 s. |
| T-TAD-06 | Availability | A queue-sourced run at a site with a blank `Observatory.timezone` now raises where it previously produced a container | accept | Pre-existing, already-isolated failure mode (the command reports it per run and continues); Task 4 documents it in the runbook alongside the existing "Observatory missing timezone" fix. |
| T-TAD-SC | Tampering | npm / pip / cargo installs | accept | No dependency is added or changed; `pyproject.toml` is untouched by this task. |
</threat_model>

<verification>
Full regression, exactly as the bug report requires -- read the diffs of every changed
assertion, do not just count green:

- `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval`
  (use `python manage.py`, not `./manage.py`; do not add `test_views.TestEphemeris`, which
  segfaults in native ASSIST)
- `ruff check .` and `ruff format --check .` clean
- `! grep -rn 'QUEUE_SOURCES' solsys_code/ src/ docs/` returns nothing
- `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` builds clean
- `git status` shows no scratch `src/fomo_db.sqlite3` staged and the operator's DB copy restored
</verification>

<success_criteria>
- `reconcile_run()` has three dispatch branches and reads no `source` value; `QUEUE_SOURCES` is
  gone from the codebase
- A queue-sourced run with a resolved, non-satellite site gets per-night dip-corrected events;
  `telescope_class` runs and satellite-site runs still get one whole-window container each
- Every pre-existing test that asserted the old behaviour was corrected in place, and the
  executor can state per test what changed and why
- The pre-fix-container-to-per-night convergence is proven by a live-shaped test that was
  observed to fail with the old branch temporarily restored
- Record-event non-interference is asserted on both branches, and 29-SECURITY.md's T-29-07
  evidence says so (still closed, `threats_open: 0`, dated audit-trail entry added)
- No artifact anywhere describes this change as violating RECON-02/RECON-03; every mention
  frames it as correcting how "no fixed observing site" is detected
- The paired notebook (regenerated, with output) and the runbook describe the corrected
  behaviour
</success_criteria>

<output>
Create `.planning/quick/260805-tad-fix-window-shape-dispatch-in-the-calenda/260805-tad-SUMMARY.md` when done.
Record in it: the per-test before/after expectation table for every corrected test, the Task 2
mutation-probe outcome, the measured post-change test runtime against the 46.7 s baseline, and
the notebook regeneration evidence.
</output>
