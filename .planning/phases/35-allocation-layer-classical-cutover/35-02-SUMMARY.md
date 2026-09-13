---
phase: 35-allocation-layer-classical-cutover
plan: 02
subsystem: testing
tags: [django, campaign-reconciler, allocation-projector, calendar-events, test-migration]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-01's ALLOC: namespace (allocation_projector.py), the D-09/D-10 dispatch inversion in campaign_reconciler.reconcile_run(), and the retired RUN:{pk}:{date}/run_night_url()/_reconcile_classical_nights()"
provides:
  - "Seven test modules migrated onto the ALLOC: namespace and the D-10 queue-dispatch inversion, green in one invocation, with a full kept/migrated/retired classification table"
  - "A resolved test_campaign_site_search.py import cascade (it imported test_campaign_approval.py's shared fixtures, which failed to import once run_night_url() no longer existed)"
  - "A production gap found and recorded (not fixed, per this plan's test-only scope): campaign_views._resolve_site()'s 'no new entries' message still names the now-permanently-zero result.skipped_nights"
affects: [35-03, 35-04, 35-05, 35-06, 35-07]

actuals:
  tokens: 40636
  tasks: 3
  commits: 2
  plan_head_before: 74f144ae6e9b56cbbd6fa44497e25167e33341c9

tech-stack:
  added: []
  patterns:
    - "Container-run-plus-hand-made-legacy-event fixture: to keep testing _stale_attributions()/_detach_stale_family_events() (unchanged by Phase 35, RUN:-namespace-only), a container-dispatched run (queue-sourced or class-wide) is reconciled once, then a RUN:{pk}:{date} event is hand-created and attributed to it directly, simulating a leftover per-night artifact -- since a container's active_urls is always exactly {RUN:{pk}}, any such event is structurally stale regardless of trigger."
    - "CampaignRunObservation-linked-placed-record fixture replaces the retired 'attribute a bare non-RUN: event via CalendarEventMeta.run' fixture wherever a test needs to prove a night is retired (D-05) -- the old shape has zero effect on the allocation projector, since retirement is read exclusively from run.observation_links, never from a generic attributed-events set."
    - "mock.patch retargeting: every patch of sun_event/insert_or_create_calendar_event/update_calendar_event_key_and_fields against solsys_code.campaign_reconciler moved to solsys_code.allocation_projector wherever the fixture's CampaignRun is now allocation-dispatched (resolved site, non-queue source) -- campaign_reconciler.py no longer imports sun_event at all after 35-01."

key-files:
  modified:
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_observation_projector.py
    - solsys_code/tests/test_observation_projector_signals.py
    - solsys_code/tests/test_project_observation_calendar.py

key-decisions:
  - "TestClassicalStage1's 5 tests with a confirmed test_allocation_projector.py counterpart are retired; the 2 without one (site-local key-date round-trip, mid-loop sun_event ValueError propagation) are kept, migrated to the ALLOC: key form and the allocation_projector.sun_event patch target, per the plan's own 'verify before retiring' instruction."
  - "TestAttributedNightSkip and TestObservingNightBoundary are retired outright: _attributed_nights() is genuinely dead code after 35-01 (defined in campaign_reconciler.py, called from nowhere), so the skip-the-night rule these classes proved no longer exists to test. TestObservingNightBoundary's noon-anchor boundary coverage is fully duplicated by test_allocation_projector.TestAllocationNightBoundary (confirmed present, 8 tests, both hemispheres) before retiring."
  - "TestReclassificationConvergence.test_pre_fix_container_event_converges_to_per_night_on_next_reconcile is retired: it reproduced a pre-D-10 bug (quick task 260805-tad) where a queue-sourced, resolved-site run wrongly took the per-night branch. D-10 makes container dispatch permanent for every queue source, so the 'converges back to per-night' scenario this test proved can no longer occur at all."
  - "TestSkipAndDetachCounters (test_reconcile_campaign_runs.py) is renamed TestSummaryCounters: its two skipped_nights tests are retired (same dead-code reason as TestAttributedNightSkip) and replaced with tests proving the two counters 35-01 actually added, retired and rekeyed, in both the real and --dry-run summary lines, per Task 2's own instruction."
  - "campaign_views.py's WR-12 'no new entries' message (_resolve_site()) is left as-is, not fixed: it names result.skipped_nights, which is now permanently 0, so the message always reads '0 night(s) are already covered' even when the real reason is a retired night. Recorded as a found-not-fixed production gap per this plan's test-only scope (see Deviations)."

requirements-completed: [ALLOC-01, ALLOC-03]

coverage:
  - id: D1
    description: "test_campaign_reconciler.py's ~100 RUN:{pk}:{date} references migrated onto ALLOC:{pk}:{night}, with a full kept/migrated/retired classification table and a Chilean (America/Santiago) Observatory fixture added"
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py (49 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The six remaining affected test modules (test_reconcile_campaign_runs.py, test_campaign_approval.py, test_calendar_utils.py, test_observation_projector.py, test_observation_projector_signals.py, test_project_observation_calendar.py) migrated and green in one invocation with this plan's own module"
    requirement: ALLOC-03
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval solsys_code.tests.test_calendar_utils solsys_code.tests.test_observation_projector solsys_code.tests.test_observation_projector_signals solsys_code.tests.test_project_observation_calendar"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both formatting gates (pre-commit run ruff / ruff-format) clean over this plan's own seven files"
    verification:
      - kind: other
        ref: "pre-commit run ruff --files <7 paths> && pre-commit run ruff-format --files <7 paths>"
        status: pass
    human_judgment: false

duration: 90min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 02: Test Suite Migration to the ALLOC: Namespace Summary

**All seven test modules affected by 35-01's `RUN:{pk}:{date}` deletion and D-10's queue-dispatch inversion are migrated onto the `ALLOC:` namespace and the container-dispatch rule, green in one invocation, with a full kept/migrated/retired classification table and no silently-dropped coverage.**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-09-13T04:00:00Z (approx)
- **Completed:** 2026-09-13T05:30:00Z (approx)
- **Tasks:** 3
- **Files modified:** 5 (2 files -- `test_calendar_utils.py` -- required no changes at all)

## Accomplishments

- `test_campaign_reconciler.py`'s ~100 `RUN:{pk}:{date}` references migrated: `TestQueueSourceDoesNotChangeShape` renamed to `TestQueueSourceDispatchesToContainer` and its premise inverted for D-10 (queue-sourced + resolved site now gets one bare container, plus a new `SOAR_QUEUE` case); `TestOwnershipScoping`, `TestReconcileThenAttributeOrdering`, `TestAttributedEventsSurviveReconcile`, `TestRecordEventNonInterference`, `TestReclassificationConvergence`, `TestCrossRunOwnershipGuards` and `TestTelescopeInstrumentSplitOnEvents` migrated their per-night fixtures to `ALLOC:`; `TestAttributedNightSkip` and `TestObservingNightBoundary` retired outright (dead-code / fully-duplicated coverage); 5 of `TestClassicalStage1`'s 7 tests retired with a confirmed `test_allocation_projector.py` counterpart, 2 kept with no counterpart.
- A Chilean (`America/Santiago`) `Observatory` fixture added alongside the existing Australian one on `CampaignReconcilerTestBase`.
- A new D-14 test proves a leftover `ALLOC:{pk}:*` night for a shrunk window is deleted, not detached -- the allocation-namespace twin of the existing container-family convergence tests.
- `test_reconcile_campaign_runs.py`'s `TestSkipAndDetachCounters` renamed `TestSummaryCounters`: the dead `skipped_nights` tests retired, replaced with tests proving `retired`/`rekeyed` appear in both the real-run (`retired: N, rekeyed: N`) and `--dry-run` (`would_retire: N, would_rekey: N`) summary lines, including a case where a linked, placed record reports a non-zero `retired`.
- `test_campaign_approval.py`: the deleted `run_night_url()` import replaced with `allocation_projector.allocation_night_url()`/`allocation_events()`/`allocation_night_title()`; every `(window a..b)` per-night title assertion dropped (D-12); five `mock.patch` targets retargeted from `campaign_reconciler` to `allocation_projector` (`sun_event` x2, `insert_or_create_calendar_event`, `update_calendar_event_key_and_fields`); the "every night already covered" and "detaches something" fixtures rebuilt around the real D-05 handoff and D-16 legacy-takeover mechanisms respectively.
- `test_campaign_site_search.py`'s import cascade (it imports `test_campaign_approval.py`'s `BULK_MPC_FIXTURE`/`ISOLATED_TEST_CACHES`) resolved as a side effect -- confirmed green (25 tests) without any direct edit.
- `test_observation_projector.py`, `test_observation_projector_signals.py`, `test_project_observation_calendar.py`: the three `TestNamespaceIsolation`-style foreign-event fixtures re-keyed from `RUN:{pk}:{date}` to `ALLOC:{pk}:{date}` so they keep guarding a key form a real writer produces.
- `test_calendar_utils.py` needed zero changes -- its `RUN:1:2026-08-01`-style strings are `update_calendar_event_key_and_fields()`'s own generic re-key fixtures, unrelated to any dispatch branch.
- All seven modules pass in one `python manage.py test` invocation (337 tests); both `pre-commit run ruff`/`ruff-format` gates clean over the seven files.

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate test_campaign_reconciler.py onto the ALLOC: namespace** - `52c5284` (test)
2. **Task 2: Migrate the six remaining affected test modules** - `cfd63fe` (test)
3. **Task 3: This plan's seven modules green and their formatting gates clean** - no additional commit; Tasks 1-2 already left all seven modules green and both formatting gates clean, confirmed by re-running the full seven-module invocation and both gates fresh (337 tests, `Passed`/`Passed`).

**Plan metadata:** committed alongside this SUMMARY (see final commit below).

## Files Created/Modified

- `solsys_code/tests/test_campaign_reconciler.py` - migrated onto `ALLOC:`/D-10; see classification table
- `solsys_code/tests/test_reconcile_campaign_runs.py` - migrated fixtures + `retired`/`rekeyed` summary-counter coverage
- `solsys_code/tests/test_campaign_approval.py` - migrated imports, patch targets, and per-night title/description assertions
- `solsys_code/tests/test_observation_projector.py` - one foreign-namespace fixture re-keyed to `ALLOC:`
- `solsys_code/tests/test_observation_projector_signals.py` - one foreign-namespace fixture re-keyed to `ALLOC:`
- `solsys_code/tests/test_project_observation_calendar.py` - one foreign-namespace fixture re-keyed to `ALLOC:`
- `solsys_code/tests/test_calendar_utils.py` - unchanged (verified, not modified)

## Classification Table

### test_campaign_reconciler.py (Task 1)

| Class | Outcome | Destination / Reason |
|---|---|---|
| `CampaignReconcilerTestBase` | kept | fixture; added `chile_ground_site` (America/Santiago) alongside the existing Australian `ground_site` |
| `TestSkipReasons` | kept | stage-0 guard untouched by Phase 35 |
| `TestQueueSourceDoesNotChangeShape` | migrated | renamed `TestQueueSourceDispatchesToContainer`; premise inverted for D-10 (queue-sourced + resolved site -> one bare container); added `SOAR_QUEUE` case |
| `TestClassWideStage2` | kept | unaffected by D-09/D-10 |
| `TestSatelliteContainer` | kept | `sun_event` patch target moved from `campaign_reconciler` (no longer imports it) to `telescope_runs` -- production behaviour unaffected |
| `TestOwnershipScoping` | migrated | 2 per-night cases moved to `ALLOC:`; trailing-colon guard test kept (about `owned_events()`'s own prefix, still applies to `RUN:`) |
| `TestContainerIdempotency` | kept | class-wide container branch, unaffected |
| `TestAttributedNightSkip` | retired | `_attributed_nights()` is dead code after 35-01 (defined, never called) -- the skip-the-night rule it proved no longer exists |
| `TestObservingNightBoundary` | retired | destination `test_allocation_projector.TestAllocationNightBoundary` (8 tests, both hemispheres) -- confirmed present before retiring |
| `TestReconcileThenAttributeOrdering` | migrated | detach-and-restore case rebuilt around the D-05/D-07 handoff (delete, not detach, for a per-night run); human-confirmation guard cases moved to a container-run + hand-made-legacy-event fixture |
| `TestAttributedEventsSurviveReconcile` | migrated | now asserts non-interference against an allocation-dispatched run; the retired "skip" concept replaced with "coexists, a fresh night is still created" |
| `TestClassicalStage1` | migrated (2) / retired (5) | 5 tests retired with a confirmed `test_allocation_projector.TestEndToEndAllocationNight`/`TestAllocationEventAttribution` counterpart; 2 (site-local key-date round-trip, mid-loop `ValueError` propagation) kept with no counterpart found, migrated to `ALLOC:` |
| `TestRecordEventNonInterference` | migrated | points at an allocation-dispatched run; record deliberately left un-linked via `CampaignRunObservation` to keep testing pure coexistence, not the handoff |
| `TestContainerRecordEventNonInterference` | kept | container-branch twin, unaffected |
| `TestReclassificationConvergence` | migrated (3) / retired (1) | `test_pre_fix_container_event_converges_to_per_night_on_next_reconcile` retired -- the bug it reproduced (260805-tad) can no longer occur under D-10's permanent container dispatch for queue sources; other 3 migrated + 1 new D-14 leftover-allocation-night case added |
| `TestCampaignRunDeletionCascadesCalendarEvents` | kept | container-only; the `ALLOC:` cascade twin is `test_allocation_projector.TestAllocationDeletionCascade` (35-01) |
| `TestCrossRunOwnershipGuards` | migrated (2) / kept (3) | the 2 per-night guard tests re-keyed to `ALLOC:` (the fixture must collide with the live writer); 3 `RUN:`-namespace-specific guard tests kept unchanged |
| `TestWindowEndBeforeWindowStart` | kept | stage-0 guard, unaffected |
| `TestTelescopeInstrumentSplitOnEvents` | migrated (2) / kept (3) | the classical-create-path tests re-keyed to `ALLOC:`; container-branch tests kept |
| `TestSplitTelescopeInstrumentHelper` | kept | pure-function tests, unaffected |

### test_reconcile_campaign_runs.py (Task 2)

| Class | Outcome | Destination / Reason |
|---|---|---|
| `ReconcileCampaignRunsTestBase` | kept | fixture; docstring updated for the new dispatch rules |
| `TestIdempotency` | migrated | asserts `allocation_events()` for the classical run, `owned_events()` for the two container runs |
| `TestDryRun` | kept | aggregate-count comparison, unaffected by which namespace each fixture run lands in |
| `TestFailureIsolation` | migrated | all three fixture runs are now allocation-dispatched; assertions moved to `allocation_events()` |
| `TestRealDataShapeScenario` | migrated | site-resolved queue runs now assert one bare container each (D-10); classical runs assert `allocation_events()` |
| `TestSkipAndDetachCounters` | migrated (5) / retired (2) | renamed `TestSummaryCounters`; the 2 `skipped_nights` tests retired (dead-code reason, same as `TestAttributedNightSkip`); 5 kept/added proving `detached`/`retired`/`rekeyed` in both real and dry-run summaries |

### test_campaign_approval.py (Task 2)

25 test classes total. 7 touched (migrated); the remaining 18 kept unchanged (no `RUN:`/`owned_events`/per-night title reference).

| Class | Outcome | Destination / Reason |
|---|---|---|
| `TestApproval` | migrated | one `mock.patch` target moved to `allocation_projector.insert_or_create_calendar_event` |
| `TestCalendarProjection` | migrated | classical-run assertions moved to `allocation_events()`/`ALLOC:`; `(window a..b)` suffix dropped for per-night titles; `sun_event` patch targets moved to `allocation_projector` |
| `TestRunStatusChange` | migrated | title/description assertions moved to `allocation_night_title()`; `update_calendar_event_key_and_fields` patch target moved to `allocation_projector` |
| `TestSitesNeedingReview` | migrated | "every night already covered" fixture rebuilt around a `CampaignRunObservation`-linked placed record (D-05); "detaches something" fixture rebuilt around a container-dispatched (`LCO_QUEUE`) run |
| `TestPlaceholderSiteReplacement` | migrated | uses the renamed `allocation_night_url()` helper (mechanical, no behaviour change) |
| `TestCalendarNoChurn` | migrated | uses the renamed `allocation_night_url()` helper (mechanical, no behaviour change) |
| `TestGeminiFtScenario` | migrated | title assertions moved to `allocation_night_title()`; `(window a..b)` suffix dropped |
| `TestStaffGating`, `TestDecidedTableStatusActions`, `TestApprovalQueueColumns`, `TestApprovalQueueSiteVisibility`, `TestApprovalSiteResolution`, `TestSiteSelectionResolution`, `TestSiteSelectionNameCandidateResolution`, `TestIsPlaceholderObservatory`, `TestSelectionToObscode`, `TestApprovalQueueSitesNeedingReviewGrouping`, `TestCreateObservatoryRoundTrip`, `TestCreateObservatoryTemplateNextRoundTrip`, `TestSiteFuzzyMatch`, `TestSiteSearchCacheIsolationRegression`, `TestApprovalQueueSiteSearchWidget`, `TestResolveSiteI11GeminiSouth`, `TestResolveSiteSatelliteObscode`, `TestResolveSiteHorizonsObserverNotation` | kept | no `RUN:`/`owned_events`/per-night key or title reference |

### test_calendar_utils.py (Task 2)

| Class | Outcome | Destination / Reason |
|---|---|---|
| all 10 classes | kept | `RUN:1:2026-08-01`-style strings are `update_calendar_event_key_and_fields()`'s own generic re-key contract fixtures, independent of any dispatch branch |

### test_observation_projector.py (Task 2)

| Class | Outcome | Destination / Reason |
|---|---|---|
| `TestNamespaceIsolation` | migrated | one hand-made foreign-namespace fixture event re-keyed from `RUN:1:2026-09-01` to `ALLOC:1:2026-09-01` |
| `TestStageFor`, `TestTitleAndToken`, `TestEventFieldsFor`, `TestProjectRecordWrites`, `TestMetaLinks`, `TestFieldPopulation`, `TestObservedToken` | kept | no `RUN:`/campaign-reconciler reference |

### test_observation_projector_signals.py (Task 2)

| Class | Outcome | Destination / Reason |
|---|---|---|
| `TestRecordDeleteReceiver` | migrated | one hand-made foreign-namespace fixture event re-keyed from `RUN:42:2026-09-15` to `ALLOC:42:2026-09-15` |
| `TestPostSaveReceiver`, `TestUpdateObservationStatusPath`, `TestGroupMembershipReceiver`, `TestReceiverSafetyContract` | kept | no `RUN:`/campaign-reconciler reference |

### test_project_observation_calendar.py (Task 2)

| Class | Outcome | Destination / Reason |
|---|---|---|
| `TestNamespaceIsolation` | migrated | one hand-made foreign-namespace fixture event re-keyed from `RUN:1:2026-09-01` to `ALLOC:1:2026-09-01` |
| `TestBareInvocationAndSummary`, `TestProposalAndFacilityFiltering`, `TestDryRun`, `TestFailureIsolation`, `TestProjectQuerysetOrdering`, `TestObservedSiteLookup` | kept | no `RUN:`/campaign-reconciler reference |

**Totals:** 20 + 6 + 25 + 10 + 8 + 5 + 7 = 81 classes surveyed across the seven modules. Kept: 55. Migrated: 20. Retired: 6 (0 without a named destination or reason).

## Decisions Made

See `key-decisions` in the frontmatter for the five load-bearing calls: TestClassicalStage1's partial retirement, TestAttributedNightSkip/TestObservingNightBoundary's dead-code retirement, TestReclassificationConvergence's pre-fix-scenario retirement, TestSkipAndDetachCounters's rename/retirement, and the deliberate non-fix of `campaign_views.py`'s stale `skipped_nights` message.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `TestSatelliteContainer`'s `sun_event` patch target no longer exists on `campaign_reconciler`**
- **Found during:** Task 1, first run of `test_campaign_reconciler.py`
- **Issue:** 35-01 removed `campaign_reconciler.py`'s own `sun_event` import entirely (the module no longer calls it, classical or otherwise). `patch('solsys_code.campaign_reconciler.sun_event', ...)` raised `AttributeError` before the assertion under test ever ran.
- **Fix:** Retargeted the patch to `solsys_code.telescope_runs.sun_event` (the source module) -- still guards against a call from anywhere, and `_reconcile_container()` (the branch this satellite run takes) never called it, before or after Phase 35.
- **Files modified:** `solsys_code/tests/test_campaign_reconciler.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_reconciler` green (49 tests).
- **Committed in:** `52c5284` (Task 1 commit)

**2. [Rule 1 - Bug] Five `mock.patch` targets in `test_campaign_approval.py` pointed at the wrong module post-35-01**
- **Found during:** Task 2, first full run of `test_campaign_approval.py`
- **Issue:** `sun_event` (x2), `insert_or_create_calendar_event`, and `update_calendar_event_key_and_fields` were all patched at `solsys_code.campaign_reconciler.*`. For every fixture `CampaignRun` in this module (default `LEGACY` source, resolved site -- always allocation-dispatched under D-09), the real call site is now `solsys_code.allocation_projector.*`; the patches never intercepted anything, so the tests asserted behaviour that never exercised the intended failure path (`test_projection_failure_reverts_...` silently left `approval_status='approved'` instead of reverting).
- **Fix:** Retargeted all five patches to `solsys_code.allocation_projector.*`.
- **Files modified:** `solsys_code/tests/test_campaign_approval.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_approval` green (126 tests).
- **Committed in:** `cfd63fe` (Task 2 commit)

**3. [Rule 1 - Bug] Two `event.description == event_description(run)` assertions failed to account for the allocation night's dark-window prefix line**
- **Found during:** Task 2, first full run of `test_campaign_approval.py`
- **Issue:** `allocation_night_description()` prepends the `-15 deg` dark-window line ahead of the shared `event_description(run)` body; a per-night allocation event's `description` is therefore never byte-equal to `event_description(run)` alone.
- **Fix:** Changed both assertions to `assertIn(event_description(run), event.description)`, matching the pattern already used for the same reason in `test_campaign_reconciler.py`.
- **Files modified:** `solsys_code/tests/test_campaign_approval.py`
- **Verification:** same test run as above.
- **Committed in:** `cfd63fe` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 bugs in test fixtures/patch targets, caught by running each migrated module's own test suite before commit). **Impact:** All three fixes are test-only; no production behaviour changed as a result of any of them.

### Found, Not Fixed (Production Gap -- Rule 4 territory, out of this plan's test-only scope)

**`campaign_views._resolve_site()`'s "no new entries" message names a permanently-zero counter.**
- **Found during:** Task 2, migrating `TestSitesNeedingReview.test_resolve_with_every_night_already_covered_reports_no_new_entries`.
- **Issue:** The `else` branch of `_resolve_site()`'s three-way message logic (`campaign_views.py:734-741`) reads `f'Site resolved — {result.skipped_nights} night(s) are already covered...'`. `result.skipped_nights` is populated only by the now-dead `_attributed_nights()`/skip-the-night rule (see `TestAttributedNightSkip`'s retirement above) and is therefore always `0` under every current dispatch path. The branch is still reachable (e.g. every night in the window is retired via the D-05 handoff, so `created == 0 and updated == 0` with `skipped_reason is None`), and when it fires the message now always reads "0 night(s) are already covered..." regardless of the real reason.
- **Why not fixed here:** Task 1's own action text is explicit -- "This plan is TEST-ONLY: it writes no production file. If a migration exposes a genuine production gap, stop and record it in the summary as a blocker rather than widening the file list."
- **Disposition:** The migrated test (`test_resolve_with_every_night_already_covered_reports_no_new_entries`) asserts the CURRENT (deficient) message text as-is, documented inline in its own docstring, rather than silently asserting a corrected string that the production code doesn't actually produce.
- **Recommended follow-up:** a quick task or a 35-0x plan should teach `_message_reconcile_side_effects()`/`_resolve_site()` to name `result.retired` (and possibly `result.rekeyed`) instead of, or alongside, `result.skipped_nights`.

## Issues Encountered

None blocking beyond the three auto-fixed deviations and the one recorded-not-fixed production gap above, both documented in full.

**35-03 in-flight-state check (per Task 3's own action text):** 35-03 has not started in this sequential run (no `35-03-SUMMARY.md` exists yet), so there is no 35-03 in-flight fallout to attribute or list here.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- This plan's seven test modules are green in one invocation (337 tests) and both formatting gates are clean over them.
- The repo-wide label-list gate (`workflow.test_command`) is deliberately NOT run here -- that is 35-06 Task 3's job, the first wave with a single plan in it and therefore the first place a whole-suite result is attributable to one plan (plan 35-03 runs in the same wave as this one and is mid-flight on `models.py`/`allocation_projector.py`, per this plan's own scoping rationale).
- One production gap (the stale `skipped_nights` message in `campaign_views._resolve_site()`) is recorded above for a future quick task or plan to close -- not a blocker for this plan or this wave.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/tests/test_campaign_reconciler.py
- FOUND: solsys_code/tests/test_reconcile_campaign_runs.py
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND commit: 52c5284
- FOUND commit: cfd63fe
- `python -c "... 'ALLOC:' in src, 'America/Santiago' in src"` -> `1 1` (Task 1 acceptance criterion)
- `python -c "... 'retired' in src + 'rekeyed' in src"` -> `2` (Task 2 acceptance criterion)
- All seven modules re-run together: 337 tests, OK
- `pre-commit run ruff --files <7 paths>` and `pre-commit run ruff-format --files <7 paths>`: both Passed
- All plan-level `<acceptance_criteria>` and `<verify>` commands for Tasks 1-3 re-run and passing (see task-by-task output above)
