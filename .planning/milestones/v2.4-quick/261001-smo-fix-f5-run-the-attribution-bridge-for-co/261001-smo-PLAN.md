---
phase: 261001-smo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_reconciler.py
  - solsys_code/allocation_projector.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_allocation_projector_signals.py
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - .planning/v2.4-INTENT-REVIEW.md
autonomous: true
requirements: [ANNOT-01, ANNOT-02, TRIG-02]

estimate:
  tokens: 105000
  raw_tokens: 105000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "Take an approved queue-sourced, class-wide run (`source=lco_queue`, `telescope_class='1m0'`, `site=None`) that has a `CampaignRunObservation` link to an LCO/SOAR record, and that record already has its own observation event. `reconcile_run(run)` sets that event's `CalendarEventMeta.run` to the run. It writes none of the event's own fields: title, description, start/end and `modified` stay unchanged. The run still gets exactly one `RUN:{pk}` container and no `ALLOC:` nights. `reconcile_run(run, dry_run=True)` writes no attribution."
    - "Saving the `CampaignRunObservation` for such a run attributes the record's event straight away, with no reconcile call. This goes through the bridge alone: `project_allocation()` is never called, and the trigger writes no `RUN:`/`ALLOC:` event. Deleting the link clears the attribution, unless the event's `CalendarEventMeta.confirmed_by` is set, in which case it stays. If the record itself is saved while the link exists, a missing attribution is put back the same way."
    - "If the event is already attributed to a different run, it is left alone, logged at WARNING ('Allocation attribution blocked') and counted in `ReconcileResult.blocked` by a container reconcile. The bridge runs even when the run's own `RUN:{pk}` container write is itself blocked."
    - "An unapproved container run attributes nothing, either on link save or on reconcile, because the `_skip_reason()` approval gate runs first."
    - "The per-night path is unchanged in behaviour. The ten named per-night tests pass with unedited bodies, and `project_allocation()` is still called exactly once per link save and once per link delete on a per-night run."
    - "There is one bridge implementation. `_sync_observation_attribution()` in `allocation_projector.py` is its only definition. `campaign_reconciler` reaches it only through a function-local import, never at module level. Neither touched module imports `solsys_code.views` or `solsys_code.ephem_utils`."
    - "If the bridge raises inside either link receiver, the staff member's link save or delete still succeeds. Only the exception type name is logged."
    - "The paired notebook was re-executed top to bottom on its scratch copy. Its real output shows a queue/class-wide run's linked record event gaining attribution on link save and on reconcile, not on a dry run, and losing it on unlink. On the copied real data it shows that no linked observation event of a run the reconciler processes is left without attribution. The runbook's 'Attributed campaign run' section says that a linked record's own entry is attributed for every kind of run."
  artifacts:
    - "solsys_code/campaign_reconciler.py: the existing container-write body moves unchanged into a new private `_write_container_event(run, *, dry_run) -> ReconcileResult`. `_reconcile_container(run, *, dry_run)` keeps its name and signature: it calls that helper, then the bridge (function-local import), and adds the bridge's refusals into `blocked`. The module docstring and both function docstrings are updated."
    - "solsys_code/allocation_projector.py: `reproject_allocation_if_dispatched()` keeps the `_skip_reason()` gate and calls `project_allocation(run)` for a per-night run, as now. For a run that does not dispatch per night it calls `_sync_observation_attribution(run, dry_run=False)` and nothing else. The docstrings of the bridge, the entry point and both link receivers are updated."
    - "solsys_code/tests/test_allocation_projector.py: `_make_record_event()` moves from `TestAttributionBridge` to `AllocationProjectorTestBase` unchanged. There is a new `TestContainerAttributionBridge` with 5 reconcile-path tests."
    - "solsys_code/tests/test_allocation_projector_signals.py: there is a new `TestContainerRunObservationReceivers` with 7 trigger-path tests."
    - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb: a new markdown and code cell pair between cells `717711a4` and `a5619fed`, prose updates in `8b703ea8`, `c4cd78c1` and `d2adacf8`, and a shorter print in `b5e5fc86`. It is re-executed on its existing scratch copy."
    - "docs/runbooks/telescope_runs_calendar.rst: one new paragraph in 'Why doesn't the calendar pop-up show an \"Attributed campaign run\" block?'."
    - ".planning/v2.4-INTENT-REVIEW.md: a 'Fix landed' paragraph under F5. It is edited in the working tree only and never committed (planning finding 9)."
  key_links:
    - "`campaign_reconciler.reconcile_run()` dispatches a non-per-night run to `_reconcile_container()`. That function writes the container, then makes the function-local call to `allocation_projector._sync_observation_attribution()`, and the result reaches `ReconcileResult.blocked`. If the import ran at module level, both modules would deadlock at load: `allocation_projector.py:49-57` already imports the reconciler at top level, and so does `campaign_utils.py:30`."
    - "`receiver_on_run_observation_save`/`_delete` (allocation_projector.py ~1594/~1646) and `observation_projector.receiver_on_record_save` (observation_projector.py:699) all call `reproject_allocation_if_dispatched()`. That function runs `_skip_reason()` and then `dispatches_per_night()`, and goes to `project_allocation()` for a per-night run or to `_sync_observation_attribution()` otherwise. Putting the change in this one entry point is what makes all three triggers and the sweep agree."
    - "`_sync_observation_attribution()` calls `campaign_utils.adopt_event_into_run()` to write attribution (it refuses an event that belongs to a different run) and `campaign_utils.unlink_event_from_run()` to clear it, and only for unconfirmed rows with `observation_record` set. `calendar_display_extras.campaign_decoration()` (templatetags, ~481) reads `CalendarEventMeta.run` at request time, so the chip appears once `meta.run` is set."
    - "A container run's convergence step never detaches the new attributions. `_stale_attributions()`/`_stale_dated_events()` scan only the `RUN:` namespace (`owned_events()`), and `_stale_allocation_events()` scans only `ALLOC:`. A facility-url observation event is in neither."
---

<objective>
Fix intent-review finding F5 (`.planning/v2.4-INTENT-REVIEW.md`, "### F5 (2026-10-01)"). A container-dispatched run (class-wide, satellite or queue-sourced) must attribute its linked observation events, exactly as a per-night run already does. Today the D-08 attribution bridge `_sync_observation_attribution()` has a single caller, `project_allocation()`. The container branch `_reconcile_container()` never calls it. The link receivers' entry point `reproject_allocation_if_dispatched()` returns early for every run where `dispatches_per_night()` is false. Live result: KEY2026B-004 runs #69-75 have 181 `CampaignRunObservation` links, and all 181 observation events still have `CalendarEventMeta.run = NULL`, so no campaign chip renders on any of them.

Purpose: the queue and class-wide runs are the ones that actually have observation records, and today they are the ones whose links never decorate. The D2/Q3/Q4 calendar checks in the intent review read the chip, so this fix has to land first.

Output: the fix plus 12 tests (Task 1, tracer). The paired notebook and runbook (Task 2), per CLAUDE.md "Paired docs": `campaign_reconciler.py`/`allocation_projector.py` map to `reconcile_campaign_runs_demo.ipynb`, plus `docs/runbooks/`. Full gates, and the F5 "Fix landed" note (Task 3).

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **The cron runs from THIS checkout, so the fix goes live on the first tick after Task 1's fix commit.** The crontab entry is `*/15 * * * * /usr/bin/flock -n -E 99 ... /home/tlister/git/fomo_devel/manage.py run_unattended`, and `unattended.STEPS` runs status_refresh, project_sweep, discovery, reconcile, proposal_allocation. Every intermediate save of a production module must therefore be importable and correct. Write the tests first (the runner never imports test files). Make each production edit in ONE tool call and run the import smoke check after each. A read-only query of `src/fomo_db.sqlite3` at planning time found: runs 69-75 hold 74/11/28/15/30/8/15 links (181 in total), all LCO; every linked record has an observation-event companion row with `run` NULL; 0 events are attributed to a different run; there are 0 unlinked, unconfirmed observation-event attributions for the unlink half to clear. So the first post-fix tick's reconcile step should adopt 181 events, block 0 and clear 0. Do not run anything against the live database yourself. The live confirmation is the operator's.
2. **One bridge, reached through a function-local import, and not lifted into another module.** `campaign_utils.py:30` imports `campaign_reconciler` at top level, and so does `allocation_projector.py:49-57`. Lifting the bridge into `campaign_utils` would still force the reconciler to use a local import, and would move a heavily cross-referenced function for no gain. So `_sync_observation_attribution()` stays where it is, and `_reconcile_container()` imports it function-locally under its private name. This is the idiom the reconciler already uses at lines 281, 676, 778 and 857, and it is what `allocation_projector.py`'s "Import discipline" docstring (lines 21-28) prescribes for attribution policy: one rule, one owner. Never write a second copy of the bridge.
3. **Discretion: the bridge runs on every container reconcile, including when the run's own `RUN:{pk}` key is blocked** (attributed to a different run). This mirrors `project_allocation()`, which runs the bridge after its per-night loop whatever the nights' outcomes. A run's record links are its own regardless of what happened to its container key. A test pins this. Under `dry_run` the bridge itself returns 0 and writes nothing, so calling it on the dry-run path is harmless and keeps a single code path.
4. **The trigger change goes in `reproject_allocation_if_dispatched()` itself, not a sibling.** It has three callers: both link receivers, and `observation_projector.receiver_on_record_save()` (line 699, once per linked run, after the base projection). Changing this one function makes all three converge. In particular, a record whose own event is drawn only after the link exists gets attributed on that record's own save (NF-04's reasoning, now for container runs too). Cost: each call walks all of the run's links (74 at most live), the same per-save cost the per-night path already pays.
5. **Container convergence never undoes the new attributions.** The proof is in the last key_link above. The run-deletion cascade (`models.py:535`) deletes only `RUN:`/`ALLOC:` events, and `CalendarEventMeta.run` is `on_delete=SET_NULL`, so deleting a container run only nulls its observation-event attributions.
6. **How the notebook gets its database.** Setup cell `9084663a` copies `src/fomo_db.sqlite3` into a new `tempfile.mkdtemp(prefix='fomo-notebook-db-')` directory with `shutil.copy2`, which only reads the source. It points `FOMO_DATABASE_PATH` at the copy before `django.setup()` and asserts that the resolved `DATABASES` name is the copy. Teardown cell `d5248b35` removes it. Keep both cells exactly as they are, and never export `FOMO_DATABASE_PATH` or copy a database yourself. The copy carries the live KEY2026B-004 links, so the cutover section's full sweep (cell `abb9338a`) will attribute those events on the copy, unless a post-fix live tick has already done so. That is the F5 repair on real data. Three consequences follow. (a) Cell `b5e5fc86` prints one line per attributed observation event, about 182 lines, so its print is shortened. (b) The namespace-isolation diff (cell `69ebe899`) stays empty, because cell `abb9338a` already converged those attributions; but the prose in `c4cd78c1` and the intro bullet in `8b703ea8` must stop saying a sweep writes nothing for an observation event. A sweep writes the event's attribution link (D-08), never its fields. (c) Property 4 in cell `803eca78` compares event fields plus `modified` only, so it is unaffected.
7. **Never call `campaign_decoration()`, or anything else that calls `reverse()`, from the notebook.** `reverse()` loads the URLconf, and `src/fomo/urls.py:19` imports `solsys_code.views`, which imports `ephem_utils` and triggers the SPICE kernel download on import. Show `CalendarEventMeta.run` and the run's campaign name instead; `campaign_decoration()` reads exactly that link. The tests assert on `meta.run` for the same reason.
8. **Why the runbook changes.** The section "Why doesn't the calendar pop-up show an \"Attributed campaign run\" block?" (about lines 2138-2250) is the passage that says when the campaign chip appears. It lists how an entry gets its attribution link: reconciler-created entries automatically, the attribution queue, and the admin path "for an entry the reconciler never touches". It is also exactly where an operator lands with the F5 symptom. It has never mentioned the link-driven route for either kind of run, so add one paragraph there. No other runbook passage makes a claim this fix changes. The `blocked` stderr line (about line 1369) and the summary-line description (about line 1158) already cover a bridge refusal.
9. **`.planning/v2.4-INTENT-REVIEW.md` belongs to the operator and has uncommitted edits, including the F5 section itself.** Task 3 adds its paragraph with a scoped Edit and must NOT stage or commit the file. `git add` on that path would sweep the operator's uncommitted setup-step, F3, F4 and F5 text into a task commit. Do not tick the F5 checkbox.
10. **Branch and staging.** Run `git branch --show-current` before the first commit; it must print `issue37-telescope-runs-calendar`. Stage every commit by explicit path. The working tree has unrelated untracked files (`.gsd/`, `reqgroup_2682493.json`, `src/fomo_db_20260929.sqlite3`, `.planning/agent-history.json`) that must stay out of every commit.

Source coverage audit. GOAL: F5, container runs attribute their linked events, via both the sweep and the trigger → Task 1 (fix plus tests), Task 2 (real-data proof on the copy), Task 3 (note). REQ: ANNOT-02, decoration rendered from `meta.run` → Task 1; ANNOT-01, annotation only, no event fields written → Task 1, tests 1/A; TRIG-02, receivers never abort the caller's save → Task 1, tests E-save/E-delete. CONTEXT (orchestrator fix direction): container-reconcile bridge with blocked folding → Task 1 production edit 1; approval-gated, bridge-only trigger path → Task 1 production edit 2; preserved semantics (refusal → blocked, confirmed never cleared, unlink convergence, `PROJECTED_FACILITIES` skip unchanged) → Task 1, bridge body untouched, tests 3/4/C; required tests 1-4 → Task 1 (cases 1-3 as new tests, case 4 by the named pins plus the AST pin gate). Paired docs → Task 2. After-landing note → Task 3. RESEARCH: none (no research phase). Nothing is unplanned, and no deferred item is present.
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
`solsys_code/campaign_reconciler.py` (907 lines), confirmed at planning time:
- Module docstring lines 1-45. The paragraph at 39-45 ("Field authority differs deliberately between this module's own container branch and the per-night allocation branch ...") is where to add one sentence about the bridge.
- Top-level imports at 47-66 import nothing from `allocation_projector` or `campaign_utils`, and must keep it that way.
- `class ReconcileResult(NamedTuple)` at 73, fields `created, updated, unchanged, blocked, skipped_nights, detached, detach_declined, remint_declined, retired, rekeyed, legacy_deleted, skipped_reason`. `_replace(...)` is available.
- `_skip_reason(run) -> str | None` at 226. `_may_write(event, run) -> bool` at 250. Its function-local import comment at 278-281 is the wording template for a local import. `_link_event_to_run(event, run)` at 291. `dispatches_per_night(run) -> bool` at 307.
- `_reconcile_container(run: CampaignRun, *, dry_run: bool) -> ReconcileResult` at 342-374. Body: build `url`/`fields`; `existing = CalendarEvent.objects.filter(url=url).first()`; `if not _may_write(existing, run): logger.warning(...); return ReconcileResult(blocked=1)`; `if dry_run: action = preview_calendar_event_action(existing, fields); return ReconcileResult(**{action: 1})`; insert-or-update; `_link_event_to_run(event, run)`; `return ReconcileResult(**{action: 1})`.
- `reconcile_run(run, *, dry_run=False)` at 825-907. Its else-branch (about 858-865) calls `_reconcile_container(run, dry_run=dry_run)` and sets `active_urls = {run_container_url(run)}`. Leave `reconcile_run()` unchanged.

`solsys_code/allocation_projector.py` (1724 lines):
- Module docstring at 1-28, including "Import discipline" at 21-28. The top-level import of the reconciler is at 49-57.
- `_sync_observation_attribution(run: CampaignRun, *, dry_run: bool) -> int` at 944-1011. When `dry_run` is true it returns 0 and writes nothing. Its link half skips `record.facility not in observation_projector.PROJECTED_FACILITIES`, finds the event by `observation_projector.event_url(record, facility)`, and calls `campaign_utils.adopt_event_into_run(event, run)`, logging a WARNING beginning 'Allocation attribution blocked' and doing `blocked += 1` on refusal. Its unlink half filters `CalendarEventMeta(run=run, confirmed_by__isnull=True, observation_record__isnull=False)`, excludes linked records, and calls `campaign_utils.unlink_event_from_run(...)`. Do NOT change its body; docstring only.
- `project_allocation()` calls the bridge at 1537 (`totals['blocked'] += _sync_observation_attribution(run, dry_run=dry_run)`). Do not change it.
- `reproject_allocation_if_dispatched(run) -> None` at 1571-1591. It does a function-local import of `_skip_reason, dispatches_per_night` from the reconciler (1587), then `if _skip_reason(run) is not None or not dispatches_per_night(run): return`, then `project_allocation(run)`.
- `receiver_on_run_observation_save` at 1594-1643 and `receiver_on_run_observation_delete` at 1646-1724. Each wraps `reproject_allocation_if_dispatched(run)` in `try/except Exception` and logs `'... failed for link pk=%s run pk=%s: %s'` with `type(exc).__name__`. The delete receiver's docstring paragraph starting "This receiver deliberately does NOT clear the removed link's own event attribution itself" (about 1682-1689) names `project_allocation()` as the converging call. Make it name `reproject_allocation_if_dispatched()`: `project_allocation()` for a per-night run, the bridge alone otherwise.

`solsys_code/campaign_utils.py`: `unlink_event_from_run(events: CalendarEvent | int | Any, run: CampaignRun | int | None) -> int` at 898, the single clearing writer (it clears run, confirmed_by and confirmed_at together). `adopt_event_into_run(event, run) -> bool` at 972-1007. Neither changes.

`solsys_code/observation_projector.py`: `PROJECTED_FACILITIES = ('LCO', 'SOAR')` (62). `facility_for(record)` (69). `event_url(record, facility)` (248). `write_event_meta(event, record)` (347) never writes `run`. `receiver_on_record_save` (605-708) projects first, then calls `reproject_allocation_if_dispatched(link.run)` per linked run at 699. Not edited.

`solsys_code/tests/test_allocation_projector.py` (about 3200 lines):
- Imports at 9-36 include `patch`, `User`, `timezone`, `CalendarEvent`, `ObservationRecord`, `NonSiderealTargetFactory`, `op` (observation_projector), `allocation_events`, `from solsys_code.campaign_reconciler import event_description, owned_events, reconcile_run`, `CalendarEventMeta, CampaignRun, CampaignRunObservation`.
- `AllocationProjectorTestBase` (39-143): `_make_run(**overrides)` defaults to campaign None, CLASSICAL_FILE, APPROVED, 'NTT/EFOSC2', `site=self.chilean_site`, `site_raw='809'`, window 2026-07-09..11. `_link_record(run, *, scheduled_start=None, scheduled_end=None, facility='LCO', status='COMPLETED') -> (record, link)` creates a record with minimal `parameters={'proposal': 'TEST'}`, so the observation projector draws NO event for it, then creates the link.
- `TestAttributionBridge` (1020-1107) holds `_make_record_event(self, record, start, end) -> CalendarEvent` (1023-1038, uses `op.facility_for`/`op.event_url`/`op.write_event_meta`), followed by the three per-night pins. `TestAllocationDeletionCascade` starts at 1110.

`solsys_code/tests/test_allocation_projector_signals.py` (about 320 lines):
- Imports at 7-26 include `date, datetime, dt_timezone, SimpleNamespace, patch, uuid4, ZoneInfo, User, TestCase, CalendarEvent, ObservationRecord, NonSiderealTargetFactory`, `ap` (allocation_projector), `allocation_events`, `reconcile_run`, `CalendarEventMeta, CampaignRun, CampaignRunObservation`, `facility_for`, `Observatory`, `observing_night`. It does NOT import `django.utils.timezone`.
- `AllocationSignalsTestBase` (29-102): `setUp` creates and reconciles a per-night run (3 `ALLOC:` nights). `_make_record(*, scheduled_start=None, scheduled_end=None, status='COMPLETED', facility='LCO')` creates a record WITH `instrument_type`, so the observation projector's post_save draws the record's own event. `_night_2_block()` returns an (start, end) block on 2026-08-02. The record-event lookup idiom is `CalendarEvent.objects.get(url=facility_for(record).get_observation_url(record.observation_id))`.
- `TestCampaignRunObservationDeleteReceiver` ends at about 209. `TestCampaignRunObservationReceiverWiring` starts at 212.

Model enums: `CampaignRun.Source.LCO_QUEUE` ('lco_queue'), `CampaignRun.TelescopeClass.ONE_M0` ('1m0'), `CampaignRun.ApprovalStatus.APPROVED` / `PENDING_REVIEW`. `CalendarEventMeta.run` is `ForeignKey(CampaignRun, on_delete=SET_NULL)`. `CalendarEventMeta.event` is a OneToOne field with related_name `telescope_label_meta`.
</interfaces>
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Container runs attribute their linked observation events on reconcile and on link/record save (tests first, then two small production edits)</name>
  <files>solsys_code/tests/test_allocation_projector.py, solsys_code/tests/test_allocation_projector_signals.py, solsys_code/campaign_reconciler.py, solsys_code/allocation_projector.py</files>
  <read_first>
    - solsys_code/campaign_reconciler.py lines 1-45, 226-375, 825-907
    - solsys_code/allocation_projector.py lines 1-28, 944-1011, 1530-1540, 1571-1724
    - solsys_code/tests/test_allocation_projector.py lines 1-146 and 1020-1110
    - solsys_code/tests/test_allocation_projector_signals.py lines 1-212
  </read_first>
  <behavior>
    New `TestContainerAttributionBridge(AllocationProjectorTestBase)` in test_allocation_projector.py. Every test uses a container run made by `_make_container_run()` and one linked record whose own event comes from `_make_record_event()`:
    - 1 `test_reconcile_attributes_the_linked_record_event`: meta.run goes from None to the run. `result.blocked == 0`. The event's title, description, start_time, end_time and `modified` are unchanged. `RUN:{pk}` exists and `allocation_events(run).count() == 0`. A second `reconcile_run(run)` reports `unchanged == 1` and keeps the attribution.
    - 2 `test_dry_run_reconcile_attributes_nothing`: `reconcile_run(run, dry_run=True)` leaves meta.run None. This is a guard and passes before the fix too.
    - 3 `test_event_attributed_to_a_different_run_is_left_alone_and_counted_blocked`: meta.run is pre-set to another run. `reconcile_run(run).blocked == 1`, meta.run is still the other run, and the WARNING log contains 'Allocation attribution blocked'.
    - 4 `test_reconcile_clears_the_attribution_of_a_link_removed_without_signals`: attribute by reconcile. With `solsys_code.allocation_projector.reproject_allocation_if_dispatched` patched out, delete the link; meta.run is still the run. Then `reconcile_run(run)` clears it, and the event's fields are unchanged.
    - 5 `test_bridge_still_runs_when_the_container_key_itself_is_blocked`: reconcile once. Re-point the `RUN:{pk}` companion row to another run with a queryset `.update()`, and null the record event's meta.run with `.update()`. Then `reconcile_run(run)` gives `blocked == 1`, and the record event is attributed to the run again (planning finding 3).
    New `TestContainerRunObservationReceivers(AllocationSignalsTestBase)` in test_allocation_projector_signals.py. Each test uses a container run from `_make_container_run()` and a record from `self._make_record(*self._night_2_block())`, whose own event already exists. Each asserts that event's meta.run is None before linking:
    - A `test_linking_a_record_attributes_its_event_with_no_explicit_reconcile`: inside `patch('solsys_code.allocation_projector.project_allocation', wraps=ap.project_allocation)`, create the link. meta.run is the container run; the wrapped call_count is 0; no `RUN:{pk}` event exists; `allocation_events(container).count() == 0`; the event's own fields are unchanged.
    - B `test_deleting_the_link_clears_its_event_attribution`: after linking (attributed), `link.delete()` gives meta.run None, and the event's fields are unchanged.
    - C `test_deleting_the_link_keeps_a_staff_confirmed_attribution`: link, then set meta.run to the container plus confirmed_by/confirmed_at explicitly with `.update()`, then delete the link. meta.run and confirmed_by are unchanged. This is a guard and passes before the fix.
    - D `test_unapproved_container_run_attributes_nothing`: the container is made with `approval_status=PENDING_REVIEW`. After linking, meta.run is None, and `reconcile_run(container).skipped_reason == 'not approved'` with meta.run still None. This is a guard.
    - E-save `test_bridge_raising_does_not_abort_a_container_link_save` and E-delete `test_bridge_raising_does_not_abort_a_container_link_delete`: with `solsys_code.allocation_projector._sync_observation_attribution` patched to `side_effect=ValueError('secret detail')` and inside `assertLogs('solsys_code.allocation_projector', level='WARNING')`, the save or delete still happens. The joined log contains 'ValueError' and not 'secret detail'. For E-delete, create the link before patching.
    - F `test_record_save_restores_a_missing_container_attribution`: after linking, null meta.run with `CalendarEventMeta.objects.filter(event=event).update(run=None)` (the live pre-fix shape, which bypasses every receiver), then `record.save()`. meta.run is the container run again.
  </behavior>
  <action>
    **Step 0: baseline.** Run `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals --exclude-tag=ephemeris_segfault` before any edit and record the `Ran N tests` figure in the SUMMARY. It is the "before" half of the module-test delta Task 3 quotes. It must pass. If it does not, stop and report.

    **Step 1: RED. Test-only edits; the runner never imports these files.**
    - In test_allocation_projector.py, move `_make_record_event` unchanged (body and docstring) from `TestAttributionBridge` into `AllocationProjectorTestBase`, directly after `_link_record`. Add `dispatches_per_night` to the existing `from solsys_code.campaign_reconciler import ...` line.
    - Insert `TestContainerAttributionBridge` directly after `TestAttributionBridge`, before `TestAllocationDeletionCascade`. Its class docstring says it pins F5 / quick task 261001-smo: container-dispatched runs attribute their linked records' own events through the same D-08 bridge.
    - Give it `_make_container_run(self, **overrides)`. It calls `self._make_run` with `source=CampaignRun.Source.LCO_QUEUE`, `telescope_class=CampaignRun.TelescopeClass.ONE_M0`, `site=None`, `site_raw=''`, `telescope_instrument='LCO 1m0 / Sinistro'`, `window_start=date(2026, 7, 9)` and `window_end=date(2026, 7, 20)`, with overrides merged last, then asserts `dispatches_per_night(run)` is False. Give it `_linked_record_event(self, run)`, which uses a block from 2026-07-10 02:00 to 03:00 UTC, calls `self._link_record(...)` and then `self._make_record_event(...)`, and returns `(record, link, event)`. Use `self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 1))` for "another run".
    - In test_allocation_projector_signals.py, add `from django.utils import timezone` and `from solsys_code.campaign_reconciler import dispatches_per_night, reconcile_run` (merge with the existing `reconcile_run` import).
    - Insert `TestContainerRunObservationReceivers` directly after `TestCampaignRunObservationDeleteReceiver`. Give it `_make_container_run(self, **overrides)`, which builds its kwargs as a dict (campaign None, `source=LCO_QUEUE`, `approval_status=APPROVED`, `telescope_instrument='LCO 1m0 / Sinistro'`, `site=None`, `site_raw=''`, `telescope_class=ONE_M0`, window 2026-08-01..2026-08-03, `observation_details='Container signals fixture'`), updates it with the overrides, creates the run, and asserts `dispatches_per_night` is False. Give it `_own_event(self, record)`, which uses the existing `facility_for(...).get_observation_url(...)` idiom. For the staff user in C, use `User.objects.create(username=f'container-staffer-{uuid4().hex[:8]}')`.
    - Never use `SiderealTargetFactory`. All records come from the existing helpers, which use `NonSiderealTargetFactory` (CLAUDE.md).
    - Run the second `<automated>` command below. Expected RED: 12 new tests. Exactly 3 pass (2, C, D are guards). The other 9 fail with an AssertionError on attribution, `blocked`, or a missing WARNING log. If a failure is an ImportError, NameError, AttributeError or fixture error instead, fix the test before committing. Every pre-existing test must still pass.
    - Commit the two test files by explicit path: `test(261001-smo): pin attribution of linked observation events on queue/class-wide runs (F5)`.

    **Step 2: GREEN. Two production edits, each in ONE Edit tool call, with the first `<automated>` import smoke check after each (planning finding 1).**
    - Edit 1, campaign_reconciler.py. Split `_reconcile_container()`: move its current body unchanged into a new `_write_container_event(run: CampaignRun, *, dry_run: bool) -> ReconcileResult` placed directly above it. Its docstring is the current container docstring (RECON-02/03; sole writer of `RUN:{pk}`, authoritative for every field).
    - `_reconcile_container(run, *, dry_run)` keeps its signature and does three things: `result = _write_container_event(run, dry_run=dry_run)`; a function-local `from solsys_code.allocation_projector import _sync_observation_attribution` with a comment giving both reasons (allocation_projector imports this module at its own top level, so a top-level import would deadlock, which is the same idiom `_may_write()` uses; and the private name is deliberate, one attribution rule with one owner, per that module's "Import discipline"); then return `result._replace(blocked=result.blocked + _sync_observation_attribution(run, dry_run=dry_run))`.
    - Its new docstring explains the D-08 bridge running for container runs (F5, quick task 261001-smo), why it runs even when the container key is blocked (finding 3), and that dry_run attributes nothing because the bridge's own dry-run contract returns 0.
    - Edit 2, allocation_projector.py, `reproject_allocation_if_dispatched()`: keep the existing local import. First, `if _skip_reason(run) is not None: return`. Then, `if not dispatches_per_night(run):` call `_sync_observation_attribution(run, dry_run=False)` and return. Otherwise call `project_allocation(run)` exactly as now.
    - The rewritten docstring says it is still the only receiver path to `project_allocation()`; that a container run gets the bridge alone (no container write, which stays the sweep's and staff actions' job); that this is what makes a link save/delete and a linked record's own save attribute or clear immediately for every dispatch kind (F5); and that the per-night branch is unchanged. Call the bridge by its module-global name so `patch('solsys_code.allocation_projector._sync_observation_attribution')` intercepts it.

    **Step 3: docstrings. Each edit is one Edit call followed by the smoke check; prose only.**
    - `_sync_observation_attribution()`: name its three callers (`project_allocation()`, `campaign_reconciler._reconcile_container()`, and `reproject_allocation_if_dispatched()` for a non-per-night run).
    - `receiver_on_run_observation_save()`: its first paragraph gains one sentence. For a container run it attributes the record's own event instead of retiring a night.
    - `receiver_on_run_observation_delete()`: the "deliberately does NOT clear" paragraph names `reproject_allocation_if_dispatched()` as the converging call (`project_allocation()` per-night, the bridge alone otherwise).
    - campaign_reconciler.py module docstring: the 39-45 paragraph gains one sentence. The container branch also runs the D-08 attribution bridge after writing its container (F5).
    - Do not touch `_sync_observation_attribution()`'s body, `project_allocation()`, `reconcile_run()` or observation_projector.py.

    **Step 4: verify, then commit.**
    - Run every `<automated>` command below. If any PRE-EXISTING test fails, stop and report it. The only acceptable exception is a test whose assertion pins the F5 defect itself, meaning it asserts that a container run's linked record event stays unattributed. Update such a test and name it in the SUMMARY as a deviation.
    - Commit the two production files by explicit path: `fix(261001-smo): run the attribution bridge for container runs on reconcile and on link save/delete (F5)`.
    - From this commit on, the next cron tick attributes the live 181 events. That is intended.
  </action>
  <verify>
    <automated>python manage.py shell -c "import solsys_code.campaign_reconciler as r, solsys_code.allocation_projector as a; print('importable:', r._reconcile_container.__name__, a.reproject_allocation_if_dispatched.__name__, a._sync_observation_attribution.__name__)"</automated>
    <automated>python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_reconcile_campaign_runs --exclude-tag=ephemeris_segfault</automated>
    <automated>python -c "
import ast, glob
heavy = ('solsys_code.views', 'solsys_code.ephem_utils')
trees = {p: ast.parse(open(p).read()) for p in glob.glob('solsys_code/*.py')}
for p in ('solsys_code/campaign_reconciler.py', 'solsys_code/allocation_projector.py'):
    for n in ast.walk(trees[p]):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(heavy):
            raise SystemExit(f'{p} imports {n.module}')
        if isinstance(n, ast.Import) and any(a.name.startswith(heavy) for a in n.names):
            raise SystemExit(f'{p} imports a heavy module')
rec = trees['solsys_code/campaign_reconciler.py']
alp = trees['solsys_code/allocation_projector.py']
top = [n for n in rec.body if isinstance(n, ast.ImportFrom)]
assert not any(n.module in ('solsys_code.allocation_projector', 'solsys_code.campaign_utils') for n in top), 'reconciler must not import allocation_projector/campaign_utils at module level'
defs = [(p, n) for p, t in trees.items() for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == '_sync_observation_attribution']
assert len(defs) == 1 and defs[0][0] == 'solsys_code/allocation_projector.py', f'exactly one bridge implementation: {[p for p, _ in defs]}'
def calls(f):
    return {c.func.id for c in ast.walk(f) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
rf = {n.name: n for n in rec.body if isinstance(n, ast.FunctionDef)}
assert '_write_container_event' in rf, 'missing _write_container_event'
rc = rf['_reconcile_container']
local = [n for n in ast.walk(rc) if isinstance(n, ast.ImportFrom) and n.module == 'solsys_code.allocation_projector']
assert any(a.name == '_sync_observation_attribution' for n in local for a in n.names), 'bridge must be imported function-locally in _reconcile_container'
assert {'_write_container_event', '_sync_observation_attribution'} <= calls(rc), calls(rc)
af = {n.name: n for n in alp.body if isinstance(n, ast.FunctionDef)}
assert {'_skip_reason', 'dispatches_per_night', 'project_allocation', '_sync_observation_attribution'} <= calls(af['reproject_allocation_if_dispatched'])
print('OK: one bridge, function-local import only, container and trigger paths both reach it, no heavy imports')
"</automated>
    <automated>python -c "
import ast, subprocess
red = subprocess.check_output(['git', 'log', '--format=%H', '--grep=^test(261001-smo)', '-1'], text=True).strip()
assert red, 'RED commit test(261001-smo) not found'
pins = {
    'solsys_code/tests/test_allocation_projector.py': ['test_d08_round_trip_link_and_unlink_attribution', 'test_foreign_attribution_is_refused_and_counted', 'test_confirmed_attribution_survives_automated_unlink', 'test_campaign_less_run_projects_one_alloc_event_per_window_night', 'test_linked_placed_record_retires_its_night', 'test_unlinking_restores_the_retired_night_with_a_fresh_event'],
    'solsys_code/tests/test_allocation_projector_signals.py': ['test_linking_a_placed_record_retires_its_night_with_no_explicit_reconcile', 'test_deleting_the_link_restores_the_night_and_clears_the_record_event_attribution', 'test_confirm_and_undo_each_invoke_project_allocation_exactly_once', 'test_ready_connected_twice_does_not_double_project'],
}
def bodies(src):
    return {n.name: ast.dump(n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
for path, names in pins.items():
    old = bodies(subprocess.check_output(['git', 'show', f'{red}^:{path}'], text=True))
    new = bodies(open(path).read())
    for name in names:
        assert old[name] == new[name], f'{path}::{name} changed'
print('OK: all 10 per-night pins are byte-for-byte unchanged in body')
"</automated>
    <automated>pre-commit run ruff --files solsys_code/campaign_reconciler.py solsys_code/allocation_projector.py solsys_code/tests/test_allocation_projector.py solsys_code/tests/test_allocation_projector_signals.py && pre-commit run ruff-format --files solsys_code/campaign_reconciler.py solsys_code/allocation_projector.py solsys_code/tests/test_allocation_projector.py solsys_code/tests/test_allocation_projector_signals.py</automated>
  </verify>
  <done>
    The RED commit and the fix commit exist, each touching only its two files. The RED run showed exactly the 9 expected failures (on assertions) and 3 passing guards, and the SUMMARY records it. All five test modules pass after the fix, with the 12 new tests green. The AST gate confirms one bridge implementation, the function-local import, and both paths reaching the bridge. The pin gate confirms the 10 per-night tests are unchanged. Both ruff hooks are clean on the four files. The SUMMARY records the baseline and post-fix `Ran N tests` for the two edited modules.
  </done>
</task>

<task type="auto">
  <name>Task 2: Paired notebook demonstrates container-run attribution on its scratch copy; runbook says linked entries carry their campaign on every run type</name>
  <files>docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, cells by id (nbformat 4.5, 47 cells): `8b703ea8` (intro), `9084663a` (scratch setup; keep it exactly as is), `abb9338a` (cutover sweep), `7bbf8c0f` (defines `campaign` = TargetList 'Reconciler Demo Campaign'), `8ed4f673` (defines `queue_run`/`class_wide_run`), `c4cd78c1`, `b5e5fc86`, `69ebe899`, `63d961b6` and `717711a4` (observation handoff demo; style template for the new cell), `a5619fed`, `d2adacf8` (summary), `d5248b35` (teardown; keep it exactly as is)
    - docs/runbooks/telescope_runs_calendar.rst, lines 2138-2252 (section "Why doesn't the calendar pop-up show an \"Attributed campaign run\" block?")
  </read_first>
  <action>
    **Part A: notebook edits.** Make them with a throwaway Python script kept in the session scratchpad, never in the repo. Use `nbformat.read(path, as_version=4)` and `nbformat.write`, find cells by `id`, and build new cells with `nbformat.v4.new_markdown_cell` / `new_code_cell`.

    Insert one markdown cell and one code cell immediately after `717711a4` and before `a5619fed`.

    The markdown cell's heading is `## A queue or class-wide run's linked observation carries its campaign too (F5, quick task 261001-smo)`. Its plain-English prose covers these points:
    - Queue-sourced, class-wide and satellite runs get one whole-window container, not per-night `ALLOC:` nights.
    - Until this fix, only the per-night branch ran the D-08 attribution bridge. A container run's linked records therefore never got `CalendarEventMeta.run`, the link the calendar's campaign chip is drawn from. Live, KEY2026B-004's 181 links left all 181 observation entries without a chip.
    - Now both the sweep and the link's own save/delete run the bridge for every kind of run, with the same guards: approval gate first; an entry attributed to a different run is left alone and counted `blocked`; a staff-confirmed attribution is never cleared.
    - The chip itself is not rendered here, because rendering it resolves URLs and that imports the ephemeris views (planning finding 7). The cell prints the link it is drawn from instead.

    The code cell's source must contain the literal run label `LCO 1m0 / Sinistro (F5 container-attribution demo)`. It does the following, in order, with an `assert` after each step that prints a state:

    1. **Imports.** Import exactly what it uses: `date`, `datetime`, `timedelta` and `timezone as dt_timezone` from datetime; `uuid4`; `User`; `ObservationRecord`; `NonSiderealTargetFactory` (never SiderealTargetFactory, per CLAUDE.md); `allocation_events`; `dispatches_per_night`, `reconcile_run` and `run_container_url` from the reconciler; `unlink_event_from_run` from campaign_utils; `CalendarEventMeta`, `CampaignRun` and `CampaignRunObservation`; `PROJECTED_FACILITIES`, `event_url` and `facility_for` from observation_projector. Never import `campaign_decoration` or the views module.
    2. **Real-data check, read-only.** Iterate `CampaignRunObservation.objects.select_related('observation_record', 'run')`, skipping records whose facility is not in `PROJECTED_FACILITIES`. Cache `reconcile_run(run, dry_run=True).skipped_reason` once per distinct run. Find each record's own event with `CalendarEvent.objects.filter(url=event_url(record, facility_for(record))).first()` and classify the link: no event; attributed to the linking run; attributed to a different run; or unattributed, counting separately the unattributed ones on runs whose skipped_reason is None. Print one line of the form `Linked LCO/SOAR records on this database copy: N -- own event attributed to the linking run: A, attributed to a different run: B, unattributed: C (of which on runs the reconciler does not skip: D), no event: E`. Assert `D == 0`. Then print one sentence: on a copy taken before the first post-fix live tick, the cutover section's sweep above is what attributed these, which is the same repair the live tick makes.
    3. **Demo run.** Run `f5_run, _ = CampaignRun.objects.update_or_create(...)` with lookup `campaign=campaign`, `telescope_instrument='LCO 1m0 / Sinistro (F5 container-attribution demo)'`, `window_start=date(2026, 9, 1)` and `window_end=date(2026, 9, 30)`, and defaults `site=None`, `site_raw=''`, `telescope_class=CampaignRun.TelescopeClass.ONE_M0`, `source=CampaignRun.Source.LCO_QUEUE`, `approval_status=CampaignRun.ApprovalStatus.APPROVED`, and an `observation_details` string. Print its pk, source, telescope_class and site, and print `dispatches_per_night: {value}`. Run `reconcile_run(f5_run)` once and print the result. Assert that its `run_container_url` event exists and that `allocation_events(f5_run).count() == 0`.
    4. **Record and its own event.** Create an LCO `ObservationRecord` (NonSiderealTargetFactory target, a fresh `User` named `f5-demo-owner-<hex>`, `observation_id=f'f5-demo-{uuid4().hex[:8]}'`, status 'COMPLETED'). Its scheduled block runs from 2026-09-05 02:00 to 03:00 UTC, and its parameters are `{'proposal': 'DEMO', 'instrument_type': '1M0-SCICAM-SINISTRO', 'start': <block start isoformat>, 'end': <block end isoformat>}`, the shape the signals tests prove the projector draws. Get its own event with `CalendarEvent.objects.get(url=event_url(record, facility_for(record)))`. Snapshot the event's title, description, start_time, end_time and modified. Define a helper `attributed()` that returns `CalendarEventMeta.objects.filter(event=event, run=f5_run).exists()`. Print the event url and title, then `before link: attributed={attributed()}`.
    5. **Link.** `link = CampaignRunObservation.objects.create(run=f5_run, observation_record=record)`. Print `after link save: attributed=...`. Then print `campaign chip source (CalendarEventMeta.run.campaign): {name!r}`, reading the name through `CalendarEventMeta.objects.select_related('run__campaign').get(event=event)`.
    6. **Recreate the pre-fix state.** `unlink_event_from_run(event.pk, f5_run)`, with a comment that this is the state KEY2026B-004's events were left in before the fix. Print `after simulated pre-fix state: attributed=...`.
    7. **Dry run.** `dry = reconcile_run(f5_run, dry_run=True)`. Print it, then `after dry-run reconcile: attributed=...`.
    8. **Real reconcile.** `real = reconcile_run(f5_run)`. Print it, then `after reconcile: attributed=...`. Assert `real.blocked == 0`, and assert the event snapshot is unchanged after `refresh_from_db()`.
    9. **Unlink.** `link.delete()`. Print `after link delete: attributed=...`.
    10. Print a line starting `PASS: F5` that summarises the four transitions.

    Prose updates to existing cells:
    - In `8b703ea8`, reword the bullet "A real sweep touching nothing outside the `RUN:`/`ALLOC:` namespaces ..." to say the sweep never writes an observation event's own fields. The only thing it writes for one is the event's attribution link, when a run links that event's record. Add a bullet for the F5 demo.
    - In `c4cd78c1`, add sentences saying the snapshot's attribution column can legitimately change on the very first sweep of a database whose run links were never attributed. The cutover section's sweep has already done that by this point (the F5 repair, demonstrated further down), so the diff here must still be empty.
    - In `b5e5fc86`, shorten the `foreign_attributed` print only. Keep building `foreign_urls` over the WHOLE queryset and keep the overlap assertion unchanged. Print the total, a per-run count, and at most the first five rows in the existing `event pk=... url=... attributed to run pk=...` format, followed by `... and N more` when there are more.
    - In `d2adacf8`, add a short paragraph naming the F5 demo and its two proofs: the synthetic transitions, and the real-data `D == 0` check.
    - Do not touch `9084663a` or `d5248b35`.

    **Part B: lint, format, then execute.**
    - Run `pre-commit run ruff --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, then `pre-commit run ruff-format --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`. Re-run the first if the second changed anything.
    - Avoid copying the live database mid-tick: read `tail -n 3 /var/log/fomo/unattended.log` and start only when its last banner is an `=== FOMO unattended run END` line. Ticks take about 40 s at :00/:15/:30/:45.
    - Then, from the repo root and in the FOREGROUND (Bash timeout 600000 ms; never `run_in_background`, never `&`), run `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`. If the harness moves it to the background at the limit, wait for its completion notification.
    - Do not export `FOMO_DATABASE_PATH`, and do not create or copy any database yourself; the setup cell does it.
    - If the setup cell fails with "database disk image is malformed", the copy was torn mid-tick: re-run after the next END banner.
    - If a NEW cell fails, fix its source and re-execute the WHOLE notebook. Never hand-write outputs.
    - If a PRE-EXISTING cell's assertion fails, stop and report the cell id and message rather than loosening it. A real-data assertion failing is evidence, as F5 itself was.

    **Part C: runbook.** In `docs/runbooks/telescope_runs_calendar.rst`, inside the section "Why doesn't the calendar pop-up show an \"Attributed campaign run\" block?", insert ONE new paragraph after the paragraph that begins `**Every entry the reconciler creates gets that link set automatically**` and before the paragraph that begins `The manual admin path below still exists`. Its bold lead-in contains the exact words `for every kind of run`. In plain English it says:
    - When a run is linked to an LCO/SOAR observation record (a ``CampaignRunObservation``, made through the attribution page's "Observation records awaiting attribution" table, a script, or the admin), that record's own calendar entry gets its attribution link the moment the link is saved. This holds for a per-night, queue-sourced, class-wide or satellite run alike, and every reconcile re-applies it (each unattended tick's reconcile step, ``reconcile_campaign_runs``, or a staff action on the run).
    - Deleting the link clears it, unless a staff member has confirmed that entry's attribution.
    - It stays unset while the run is not yet approved, or has another skip reason.
    - An entry already attributed to a different run is left alone, counted under ``blocked`` and logged.
    - Once attributed, the entry also leaves the attribution page's "Calendar events awaiting attribution" worklist.
    - Before quick task 261001-smo (2026-10-01) this happened only for per-night runs, so a queue or class-wide run's linked entries showed no campaign chip. The first unattended tick after that fix attributes them, with no operator action.
    Change nothing else in the runbook (planning finding 8).

    Commit both files by explicit path: `docs(261001-smo): demo container-run attribution on the scratch copy; runbook says linked entries carry their campaign on every run type`.
  </action>
  <verify>
    <automated>python -c "
import json, re
nb = json.load(open('docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb'))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
counts = [c.get('execution_count') for c in code]
assert counts == list(range(1, len(code) + 1)), f'not one fresh top-to-bottom run: {counts}'
errors = [i for i, c in enumerate(code) if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
assert not errors, f'error outputs in code cells {errors}'
def out(c):
    return ''.join(''.join(o.get('text', '')) for o in c.get('outputs', []))
setup = [c for c in code if 'FOMO_DATABASE_PATH' in ''.join(c['source'])]
assert len(setup) == 1, len(setup)
m = re.search(r\"Resolved database: '([^']+)'\", out(setup[0]))
assert m and 'fomo-notebook-db-' in m.group(1), 'setup cell did not resolve to a scratch database'
ids = [c.get('id') for c in nb['cells']]
i, j = ids.index('717711a4'), ids.index('a5619fed')
assert j - i == 3, 'expected exactly one new markdown+code pair between the handoff demo and the site-correction section'
demo = nb['cells'][i + 2]
assert demo['cell_type'] == 'code' and 'F5 container-attribution demo' in ''.join(demo['source']), 'F5 demo cell not where expected'
assert 'campaign_decoration' not in ''.join(demo['source']), 'the notebook must not render the chip (reverse() imports the views)'
text = out(demo)
for token in ('of which on runs the reconciler does not skip: 0', 'dispatches_per_night: False', 'before link: attributed=False', 'after link save: attributed=True', \"campaign chip source (CalendarEventMeta.run.campaign): 'Reconciler Demo Campaign'\", 'after simulated pre-fix state: attributed=False', 'after dry-run reconcile: attributed=False', 'after reconcile: attributed=True', 'after link delete: attributed=False', 'PASS: F5'):
    assert token in text, f'F5 demo output lacks {token!r}'
diff_cell = [c for c in code if c.get('id') == '69ebe899'][0]
assert 'Differences found: 0' in out(diff_cell), 'namespace-isolation diff is no longer empty'
assert 'Removed scratch database directory' in out(code[-1]), 'last code cell is not the scratch teardown'
print(f'OK: {len(code)} code cells, one fresh run, scratch database {m.group(1)}')
"</automated>
    <automated>python -c "
t = open('docs/runbooks/telescope_runs_calendar.rst').read()
i = t.index('Clicking a calendar entry opens a pop-up that can show an')
j = t.index('public tally show?', i)
seg = t[i:j]
a = seg.index('Every entry the reconciler creates gets that link set automatically')
k = seg.index('for every kind of run')
b = seg.index('The manual admin path below still exists')
assert a < k < b, 'new paragraph is not between the reconciler-created and manual-admin paragraphs'
para = seg[k:b]
for token in ('CampaignRunObservation', 'blocked', '261001-smo', 'confirmed'):
    assert token in para, f'new paragraph lacks {token!r}'
print('OK: runbook paragraph in place')
"</automated>
    <automated>pre-commit run ruff --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb && pre-commit run ruff-format --files docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb && pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst</automated>
  </verify>
  <done>
    The notebook was executed once, top to bottom, with no error output, on a `fomo-notebook-db-` scratch copy that its last cell removed. The new cell sits between `717711a4` and `a5619fed`. Its stored output shows the real-data `D == 0` line and all four synthetic transitions, with `PASS: F5`. The namespace-isolation diff still reads `Differences found: 0`. The runbook paragraph sits in the right place, and Sphinx builds. One docs commit touches only these two files.
  </done>
</task>

<task type="auto">
  <name>Task 3: Full quality gates, then the F5 "Fix landed" note (working tree only)</name>
  <files>.planning/v2.4-INTENT-REVIEW.md</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md, sections "### F1" through "### F5" (house style of the F1 "Fix landed" and F2 "A landed" paragraphs)
    - CLAUDE.md, "Commands" and "Testing" (always `python manage.py`; ruff through pre-commit, which pins v0.2.1, per D-07)
  </read_first>
  <action>
    **Lint gates.** From the repo root, run `pre-commit run ruff --all-files`, then `pre-commit run ruff-format --all-files`. If either rewrites a file outside this plan's six code/doc files, stop and report rather than committing someone else's file. If a gate reformats one of this plan's files, commit only that change, by explicit path, as `style(261001-smo): apply pre-commit formatting`.

    **Full suite.** Run the second `<automated>` command with `SCRATCH` set to your session scratchpad directory (never a path inside the repo). Use Bash timeout 600000 ms. The last full run took about 24 minutes alongside cron ticks. If the harness moves it to the background at the limit, wait for its completion notification, then read the log's tail and the `exit=` line. Never start a second concurrent run. The suite uses Django's test database and never opens the live one. Quote the `Ran N tests` / `OK` lines in the SUMMARY. The 260930-85d baseline was 1821; this plan adds 12.

    **F5 note. Docs only, no notebook.** Use ONE scoped Edit on `.planning/v2.4-INTENT-REVIEW.md` (never Write). Insert a short paragraph immediately before the line `- [ ] Fixed and confirmed: the 181 KEY2026B-004 observation events show the campaign chip.`, separated from it by a blank line, the same layout as F1. Leave that checkbox unticked. Wording, in the F1/F2 house style:
    - `**Fix landed (quick task 261001-smo, 2026-10-01):**`, then the short SHAs of the test, fix and docs commits from `git log --oneline --grep=261001-smo`, each labelled (tests / fix / notebook + runbook).
    - One parenthetical on the mechanism: `_reconcile_container()` now runs the one bridge `_sync_observation_attribution()` via a function-local import and folds refusals into `blocked`; `reproject_allocation_if_dispatched()` keeps the approval gate and runs the bridge alone for a non-per-night run, so a link save or delete, or a linked record's own save, attributes or clears immediately.
    - The two edited modules' `Ran N` before and after (Task 1's recorded figures), and the full-suite count with `OK`.
    - One sentence of planning-time read-only evidence: 181 links on runs #69-75, all events `run = NULL`, none attributed elsewhere, nothing for the unlink half to clear. The first post-fix tick should therefore adopt 181, block 0 and clear 0.
    - A closing clause that the live confirmation is the operator's to tick: the 181 events showing the chip after the next tick, or after `python manage.py reconcile_campaign_runs`.
    Do NOT stage or commit this file (planning finding 9). Run nothing against the live database. In the SUMMARY, say the note is left uncommitted beside the operator's own edits.
  </action>
  <verify>
    <automated>pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
    <automated>python manage.py test solsys_code --exclude-tag=ephemeris_segfault 2>&1 | tee "$SCRATCH/261001-smo-full-suite.log" | tail -n 5; echo "exit=${PIPESTATUS[0]}"</automated>
    <automated>python -c "
t = open('.planning/v2.4-INTENT-REVIEW.md').read()
f5 = t.index('### F5 (2026-10-01)')
box = t.index('- [ ] Fixed and confirmed: the 181 KEY2026B-004 observation events show the campaign chip.', f5)
note = t.index('**Fix landed (quick task 261001-smo', f5)
assert f5 < note < box, 'note must sit inside F5, before its unticked checkbox'
assert '261001-smo' in t[note:box] and 'operator' in t[note:box]
print('OK: F5 note in place, checkbox untouched')
"</automated>
    <automated>git status --porcelain .planning/v2.4-INTENT-REVIEW.md</automated>
    <human-check>Operator: after the first unattended tick that starts after the fix commit (or after running `python manage.py reconcile_campaign_runs` yourself), open a few KEY2026B-004 observation entries on the calendar and confirm the campaign chip shows `KEY2026B-004_targets`. As a read-only cross-check, run `sqlite3 'file:src/fomo_db.sqlite3?mode=ro' "select count(*) from solsys_code_campaignrunobservation l join solsys_code_calendareventmeta m on m.observation_record_id = l.observation_record_id where l.run_id between 69 and 75 and m.run_id = l.run_id"`, which should read 181. Then tick the F5 checkbox in `.planning/v2.4-INTENT-REVIEW.md`.</human-check>
  </verify>
  <done>
    Both ruff hooks are clean repo-wide. The full `solsys_code` suite passes with exit 0 and the segfault tag excluded, and its count is quoted in the SUMMARY. The F5 note sits inside F5 above its unticked checkbox, naming the commits, the test-count delta and the operator's live confirmation. `git status` shows the intent-review file still modified and unstaged, never committed by this plan.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff attribution action -> `CampaignRunObservation` receivers | A staff member's link save/delete now drives an automated attribution write inside their own transaction |
| automated sweep/trigger -> `CalendarEventMeta.run` on a facility-url event | A machine writes the attribution link that the public calendar's campaign chip renders from |
| working tree -> running cron | `run_unattended` imports this checkout at every tick, so an edit on disk is live code |
| notebook kernel -> Django `DATABASES` | A docs artifact runs real ORM writes on the production host |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-smo-01 | Tampering | `_sync_observation_attribution()` now running for container runs could overwrite a staff or foreign attribution | high | mitigate | The bridge body is unchanged. `adopt_event_into_run()` refuses an event attributed to a different run (logged, counted `blocked`), and the unlink half filters `confirmed_by__isnull=True`. Pinned by tests 3, 5 and C, and by the unchanged per-night pins. |
| T-smo-02 | Information disclosure | a pending-review run's campaign name reaching the public calendar via a newly attributed observation event | medium | mitigate | `reproject_allocation_if_dispatched()` applies `_skip_reason()` before the bridge, and `reconcile_run()` applies it before dispatch (test D). `campaign_decoration()` also independently hides a run that is not `is_publicly_visible`. |
| T-smo-03 | Denial of service | the receiver bridge walks all of a run's links on every link or linked-record save, inside the caller's transaction | low | accept | This is the same per-save cost the per-night path already pays. The largest live run has 74 links. Steady-state ticks save almost no records (F1/F2 evidence: `project_sweep: updated: 0`). The first-tick repair of 181 adoptions is a one-time cost. |
| T-smo-04 | Denial of service | a half-edited module imported by the next cron tick, or a top-level circular import deadlocking module load | high | mitigate | Tests come first. The two production edits are each a single Edit followed by an import smoke check. The AST gate forbids a module-level reconciler import of `allocation_projector`/`campaign_utils` and requires the function-local import. |
| T-smo-05 | Elevation of privilege / Repudiation | a bridge exception aborting a staff member's link save or delete, or leaking exception detail into logs | medium | mitigate | Both receivers keep their `try/except Exception` around `reproject_allocation_if_dispatched()` and log only `type(exc).__name__`. Tests E-save and E-delete assert the row change survives and that 'secret detail' is absent from the log. |
| T-smo-06 | Tampering | notebook re-execution writing to the live database, or a torn copy taken mid-tick | high | mitigate | Setup cell `9084663a` is untouched: it copies to a `fomo-notebook-db-` directory and asserts the resolved DB. The Task 2 gate reads that resolved path from stored output. The executor starts only after an END banner, and never exports `FOMO_DATABASE_PATH` or runs a writing command against `src/fomo_db.sqlite3`. |
| T-smo-07 | Tampering | the task commit sweeping the operator's uncommitted intent-review edits into history | low | mitigate | Task 3 edits `.planning/v2.4-INTENT-REVIEW.md` with one scoped Edit and never stages it. The gate checks `git status --porcelain` still shows it unstaged. All commits stage by explicit path. |
| T-smo-SC | Tampering | package installs | low | accept | No npm/pip/cargo install in this plan. `nbformat` 5.10.4 and `nbconvert` 7.17.1 are already installed. |
</threat_model>

<verification>
- `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_reconcile_campaign_runs --exclude-tag=ephemeris_segfault` passes, with the 12 new tests green.
- `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` passes (exit 0).
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` are clean.
- The AST gate shows one bridge implementation, reached through a function-local import from `_reconcile_container()` and from `reproject_allocation_if_dispatched()`, with no heavy imports. The pin gate shows the 10 per-night tests unchanged.
- The notebook's stored outputs show one fresh run on a scratch copy, the F5 transitions with `PASS: F5`, the real-data `D == 0` line, and an empty namespace-isolation diff.
- For this task, `git log` shows three commits (`test`, `fix`, `docs`; a fourth `style` commit only if a gate reformatted something) touching only the six code/doc paths. `.planning/v2.4-INTENT-REVIEW.md` stays unstaged, and the pre-existing untracked files stay uncommitted.
</verification>

<success_criteria>
- A queue, class-wide or satellite run's linked LCO/SOAR record events carry `CalendarEventMeta.run`, and so the campaign chip, after any reconcile, and immediately on link save. Unlinking clears the attribution unless a staff member confirmed it. A dry run writes nothing.
- Per-night runs behave exactly as before. There is still one bridge, and no event field is ever written by attribution.
- After the fix commit, which is live at the next tick because the cron runs this checkout, the 181 KEY2026B-004 observation events show the `KEY2026B-004_targets` chip. The operator confirms this and ticks F5.
</success_criteria>

<output>
Create `.planning/quick/261001-smo-fix-f5-run-the-attribution-bridge-for-co/261001-smo-SUMMARY.md` when done. Include:
- the baseline and post-fix `Ran N` for the two edited test modules, and the full-suite `Ran N` / `OK` line;
- the RED result (9 failing on assertions, 3 guards passing) and the commit SHAs;
- any pre-existing test changed under Task 1's single allowed exception (expected: none);
- the notebook's real-data line (N / A / B / C / D / E) as executed, and whether the copy predated the first post-fix live tick;
- a statement that the F5 note is in `.planning/v2.4-INTENT-REVIEW.md` uncommitted, and that the live confirmation and the F5 checkbox are the operator's.
</output>
