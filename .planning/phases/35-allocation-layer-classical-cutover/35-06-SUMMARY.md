---
phase: 35-allocation-layer-classical-cutover
plan: 06
subsystem: calendar-sync
tags: [django, campaign-reconciler, allocation-projector, calendar-events, management-command, data-migration]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-01's ALLOC: namespace, project_allocation()'s legacy-night takeover and D-14 convergence; plan 35-05's write_and_reconcile_campaign_run() field vocabulary, _source_identifier()/_iter_run_nights()/_window_token_to_time()/_CLASSICAL_RUN_STATUS this plan reuses verbatim"
provides:
  - "solsys_code/management/commands/cutover_classical_allocations.py -- the one-time, idempotent, --dry-run-able D-15 step 3 command that converts legacy blank-url classical events into allocation runs"
  - "campaign_reconciler.ReconcileResult.legacy_deleted -- the D-16 second-half counter (a container-dispatched run's leftover per-night RUN: events are deleted, never detached)"
  - "allocation_projector.project_allocation()'s new third return value (legacy_urls_claimed) -- lets a caller's own convergence step avoid double-counting a night the per-night loop already accounted for under dry_run"
  - "Real-database proof (35-VALIDATION.md Manual-Only Verifications) that the four-step cutover sequence leaves one event per night with no duplicate or orphan"
affects: [35-07]

actuals:
  tokens: 28397
  tasks: 3
  commits: 3
  plan_head_before: 6a7784f0e2c644d54ff4f32f8ca26bfd08acc675

tech-stack:
  added: []
  patterns:
    - "Mechanical URL-shape split, not branch-gated: the new legacy_deleted delete path splits a run's stale RUN:-namespaced events by URL SHAPE (bare RUN:{pk} vs date-bearing RUN:{pk}:{date}) rather than by which dispatch branch the run currently takes. The real-database proof validated this was the right call: 4 of the 8 deleted legacy nights belonged to runs that are STILL allocation-dispatched (their windows had simply moved on), not container-dispatched -- a branch-gated implementation would have silently orphaned those four."
    - "Peek-before-write attribution guard in the cutover command: rather than re-keying an event and then calling adopt_event_into_run() to discover a foreign attribution too late (the plan's own literal wording), the command checks CalendarEventMeta.run first and never re-keys an event it cannot safely claim -- mirrors project_allocation()'s own _may_write()-first idiom and is what makes D-18's 'left byte-identical' guarantee actually hold for the foreign-attribution case."
    - "Claimed-url exclusion set for dry-run parity: project_allocation() now returns which legacy RUN:-urls it already accounted for (a takeover re-key or a retirement delete) THIS call, in either mode. campaign_reconciler.reconcile_run()'s own dry-run preview excludes that set before computing legacy_deleted, so a dry run's counters never double-count a url the per-night loop's own preview already counted under rekeyed/retired. Real mode needed no such tracking (the write already happened by the time the exclusion would matter) -- the bug was dry-run-only, and only the real-database run surfaced it."

key-files:
  created:
    - solsys_code/management/commands/cutover_classical_allocations.py
    - solsys_code/tests/test_cutover_classical_allocations.py
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/campaign_views.py
    - solsys_code/allocation_projector.py
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_reconcile_campaign_runs.py
    - solsys_code/tests/test_campaign_approval.py
    - .planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md

key-decisions:
  - "legacy_deleted's delete branch is scoped by URL SHAPE alone (bare vs. date-bearing), not by the run's current dispatch branch -- confirmed correct against real data, where several legacy_deleted rows belonged to still-allocation-dispatched runs, not just the 8 single-night runs D-10 sends to a container."
  - "The cutover command checks an event's existing attribution BEFORE re-keying it, not reactively after (adopting the allocation projector's own _may_write()-first idiom) -- this is what keeps an unexplainable event genuinely byte-identical rather than 're-keyed, then discovered unsafe'."
  - "project_allocation()'s return signature grew a third value (legacy_urls_claimed) so campaign_reconciler's new dry-run preview never double-counts a night the per-night loop already accounted for -- found only by running the real four-step sequence against the developer database (Task 3), not by any of Task 1's or Task 2's own unit tests, which is exactly the gap Task 3's real-data proof exists to catch."
  - "A pre-existing staleness the reconcile sweep exposed (not a Phase 35 defect): 8 of the developer database's 16 RUN:{pk} containers still carried a '<Campaign>: ' title prefix from before Phase 33's D-12 change (2026-09-10) stopped embedding a campaign label in event_title() -- this sweep was the first time these 8 rows were reconciled since that change shipped, so their titles corrected on this run. Recorded as a finding, not fixed here (out of this plan's scope; the reconciler's own title-writing logic is unchanged by Phase 35)."

requirements-completed: [ALLOC-05]

coverage:
  - id: D1
    description: "A container-dispatched run's leftover date-bearing RUN:{pk}:{date} events are deleted (never detached) as one-time churn, under the same foreign-attribution and human-confirmed-declined guards the existing bare-container detach path uses; the bare RUN:{pk} container form is untouched (still detached, never deleted)"
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestLegacyPerNightFamilyDeletion (6 tests) and #TestReconcileThenAttributeOrdering's reclassified-container tests (3 tests, rebuilt for this plan)"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSummaryCounters (command-level legacy_deleted/would_delete_legacy naming, 2 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A new one-time, idempotent, --dry-run-able cutover_classical_allocations command converts every explainable legacy blank-url classical event into a CampaignRun (source=CLASSICAL_FILE, source_identifier matching a fresh import) plus a re-keyed ALLOC:{pk}:{night} event with byte-identical start_time/end_time; an event or group it cannot explain (no Source line:, unresolvable telescope/site, blank timezone, campaign mismatch, foreign attribution) is left completely untouched, reported by pk/title/reason, and the command exits non-zero via a self-contained CommandError; it never calls delete on a CalendarEvent"
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py (13 tests: the plan's 8 named behaviors plus foreign-attribution and campaign-mismatch bonus coverage, plus TestCutoverSequenceContract)"
        status: pass
      - kind: other
        ref: "python manage.py help cutover_classical_allocations -> '--dry-run' present; zero '.delete(' occurrences in the command file -> '1 0'"
        status: pass
    human_judgment: false
  - id: D3
    description: "The four-step cutover sequence (migrate, deploy-as-no-op, cutover_classical_allocations, reconcile_campaign_runs) executed end to end against a scratch copy of the real developer database leaves one event per night: zero date-bearing RUN: nights, zero unexplained blank-url events apart from a recognised pre-existing junk row, unchanged container count, byte-identical facility-url events, and the three-group per-night reconciliation (rekeyed + legacy_deleted + retired) sums to the before-count"
    requirement: ALLOC-05
    verification:
      - kind: integration
        ref: "Real scratch-copy run: before 241/56/16/10/0/45, after 233/0/16/1/57/48 (see below); 48 rekeyed + 8 legacy_deleted + 0 retired == 56"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_cutover_classical_allocations.py#TestCutoverSequenceContract (synthetic-fixture pin of the same end-state)"
        status: pass
    human_judgment: true
    rationale: "The plan's own <human-check> asks a human to read the before/after table and confirm the reconciliation sums and the unexplained list is recognisable -- a judgement about what the calendar means, not a property a test alone can assert. The 8 container title corrections (pre-existing Phase 33 staleness, documented above) are also worth a human's eyes even though they are not a Phase 35 defect."
  - id: D4
    description: "Full label-list regression suite stays green after this plan's changes (the plan's own backstop truth)"
    verification:
      - kind: integration
        ref: "workflow.test_command label-list run: 1215 tests, OK (1 pre-existing skip), plus the two test_views regression tests (40 tests)"
        status: pass
      - kind: other
        ref: "pre-commit run ruff --all-files && pre-commit run ruff-format --all-files -> both Passed"
        status: pass
    human_judgment: false

duration: 95min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 06: Classical Cutover Summary

**A one-time `cutover_classical_allocations` command converts every legacy blank-url classical calendar event into an allocation run indistinguishable from a fresh import, the reconciler's `legacy_deleted` counter retires the last of the old per-night `RUN:` family for a container-dispatched run, and the full four-step sequence is proven end to end against a scratch copy of the real developer database — 241 events collapse to 233 with zero duplicate or orphaned nights.**

## Performance

- **Duration:** ~95 min
- **Started:** 2026-09-13T06:52:00Z (approx, from the prior plan's completion commit)
- **Completed:** 2026-09-13T08:27:00Z
- **Tasks:** 3 (Task 1 auto/tdd, Task 2 auto/tdd, Task 3 auto)
- **Files modified:** 10 (2 created, 8 modified)

## Accomplishments

- `campaign_reconciler.ReconcileResult.legacy_deleted`: a container-dispatched run's leftover date-bearing `RUN:{pk}:{date}` events are now deleted (never detached) as one-time churn — the second, previously-unreachable half of the retired per-night key family's cutover. The bare `RUN:{pk}` container form is untouched: still detached, never deleted. `reconcile_campaign_runs.py` aggregates and reports the counter in both the real (`legacy_deleted: N`) and `--dry-run` (`would_delete_legacy: N`) summary lines, plus a per-run stdout line naming the cause in plain words.
- `campaign_views._message_reconcile_side_effects()` gained a matching staff-facing warning, since the same D-16 delete path can fire from the `resolve_site` staff action, not only the sweep.
- New `solsys_code/management/commands/cutover_classical_allocations.py` (D-15 step 3, D-17): groups every blank-url classical event by the `Source line:` recovered from its own description, re-parses it with the same `parse_run_line()`/`get_site()` the ingest command uses, and creates or updates a campaign-less `CampaignRun` keyed by the same `_source_identifier()` a fresh import would compute — imported directly from `load_telescope_runs.py`, never re-derived. Each event in an explainable group is re-keyed to `ALLOC:{run_pk}:{night}` in place, keeping its primary key and `start_time`/`end_time` byte-identical (no `sun_event()` recompute) and its preserved dark-window description line.
- D-18: an event or group the command cannot explain — no parseable `Source line:`, an unresolvable telescope/site, a blank timezone, a campaign mismatch among a group's events, or an event already attributed to a different run — is left completely untouched, reported with its primary key, title and reason, and the command raises a self-contained `CommandError` (usable by a `call_command()` caller with no access to the preceding stdout) so the exit code is non-zero. The command never calls delete on a `CalendarEvent` on any path, verified both by a zero-occurrence grep over the file and by an explicit row-count-unchanged test across every failure path.
- The foreign-attribution guard is checked BEFORE re-keying, not reactively after (a deliberate refinement over the plan's literal wording — see Decisions): this is what makes an unexplainable event genuinely byte-identical rather than transiently re-keyed.
- Proved the full D-15 four-step sequence (`migrate`; deploy as a no-op in-tree; `cutover_classical_allocations --dry-run` then real; `reconcile_campaign_runs --dry-run` then real) against a scratch copy of the real developer database. Before: 241 events (56 `RUN:{pk}:{date}`, 16 `RUN:{pk}` containers, 10 blank-url, 0 `ALLOC:`, 45 runs) — matching `35-CONTEXT.md`'s baseline exactly, with no divergence to report. After: 233 events (0 date-bearing `RUN:` nights, 16 containers unchanged in count, 1 blank-url — the same pre-existing junk `tmp` row pk=334 — 57 `ALLOC:` events, 48 runs); every one of the 159 facility-url-keyed observation events byte-identical (url/title/description/start/end/`modified` all unchanged). Both cutover invocations exited 1 (the D-18 junk-row report, expected) and both sweep invocations exited 0. `src/fomo_db.sqlite3` itself confirmed unmodified before and after (`git status --porcelain` empty, md5sum unchanged).
- New `TestCutoverSequenceContract` (`solsys_code/tests/test_cutover_classical_allocations.py`) pins the same end-state on synthetic fixtures — a convertible blank-url group, an unexplainable event, a stays-per-night run, a run a queue source now sends to the container, and a linked-record retirement — so the guarantee survives without the developer database.
- Full label-list regression suite green (1215 tests, 1 pre-existing skip) plus the two `test_views` regression tests (40 tests); both `pre-commit run ruff --all-files`/`ruff-format --all-files` gates clean.

## Task Commits

Each task was committed atomically:

1. **Task 1: A run that is now a container deletes its leftover date-bearing events** - `7b2d25a` (feat)
2. **Task 2: The one-time cutover command** - `20198ef` (feat)
3. **Task 3: Prove the four-step sequence against a copy of the real database** - `deb9142` (test) — includes the Rule 1 dry-run double-counting fix this task's own real-data run found

**Plan metadata:** committed alongside this SUMMARY.

_Tasks 1 and 2 were marked `tdd="true"`; see **TDD Gate Compliance** below for the actual commit shape versus the plan's RED/GREEN contract._

## Files Created/Modified

- `solsys_code/management/commands/cutover_classical_allocations.py` - new one-time cutover command (Task 2)
- `solsys_code/tests/test_cutover_classical_allocations.py` - new test module, 13 tests (Tasks 2 & 3)
- `solsys_code/campaign_reconciler.py` - `legacy_deleted` field, `_split_stale_owned_events()`/`_clearable_and_declined()`/`_stale_dated_events()` helpers, `_detach_stale_family_events()` delete branch, `reconcile_run()` dry-run parity fix (Tasks 1 & 3)
- `solsys_code/campaign_views.py` - `_message_reconcile_side_effects()` gains the `legacy_deleted` warning (Task 1)
- `solsys_code/allocation_projector.py` - `project_allocation()` returns `legacy_urls_claimed` (Task 3 fix)
- `solsys_code/management/commands/reconcile_campaign_runs.py` - `legacy_deleted` aggregation and summary lines (Task 1)
- `solsys_code/tests/test_campaign_reconciler.py` - new `TestLegacyPerNightFamilyDeletion` class, reclassified-container fixture rebuild in `TestReconcileThenAttributeOrdering`, two Task 3 regression tests (Tasks 1 & 3)
- `solsys_code/tests/test_reconcile_campaign_runs.py` - `legacy_deleted`/`would_delete_legacy` command-level tests, one regression assertion (Tasks 1 & 3)
- `solsys_code/tests/test_campaign_approval.py` - `test_resolve_that_detaches_something_shows_the_warning` rebuilt for the new delete behavior (Task 1 deviation)
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md` - `35-06-0*` row and Wave 0 checkbox flipped; before/after evidence appended to the Manual-Only Verifications row (Task 3)

## Real-Database Before/After Table

| Metric | Before | After |
|---|---|---|
| Total `CalendarEvent`s | 241 | 233 |
| Date-bearing `RUN:{pk}:{date}` | 56 | 0 |
| Bare `RUN:{pk}` containers | 16 | 16 |
| Blank-url (`url=''`) | 10 | 1 (pk=334, `tmp`, unexplained by design) |
| `ALLOC:`-keyed | 0 | 57 |
| Facility-url (observation) | 159 | 159 (byte-identical) |
| `CampaignRun`s | 45 | 48 (+3 `classical_file`) |

**Three-group reconciliation:** 48 nights rekeyed (in-place, per-night-dispatched runs' own legacy-night takeover) + 8 nights `legacy_deleted` (one-time churn) + 0 nights `retired`-by-observation = **56**, exactly the before-count of date-bearing nights.

**Cutover step, real invocation:** `candidates: 10, groups: 3, runs created: 3, updated: 0, unchanged: 0, events re-keyed: 9, unexplained: 1` (exit 1 — the junk `tmp` row).

**Sweep step, real invocation:** `runs: 48, created: 0, updated: 8, unchanged: 17, skipped: 9, failed: 0, blocked: 0, detached: 0, detach_declined: 0, retired: 0, rekeyed: 48, legacy_deleted: 8` (exit 0). The `--dry-run` invocation of the sweep, run first from the same starting state, predicted `would_rekey: 48, would_delete_legacy: 8, would_retire: 0` exactly.

## Operator-Visible Four-Step Sequence (for plan 35-07's paired docs)

1. `python manage.py migrate` (D-04's sub-night window fields, migration 0018 — already applied by plan 35-03; nothing new for this plan).
2. Deploy the code (a no-op when running in-tree, as this proof did — the ordering statement is worth recording even so).
3. `python manage.py cutover_classical_allocations [--dry-run]` — converts every explainable legacy blank-url classical event once. Exits non-zero when it finds something it cannot explain; that exit is operator-facing, not a pipeline gate, and the same command is safe to re-run.
4. `python manage.py reconcile_campaign_runs [--dry-run]` — the first post-cutover sweep takes over every remaining `RUN:{pk}:{date}` night still reachable by its own run's per-night dispatch, and deletes the rest as one-time churn for a run that now dispatches to the whole-window container.

## Decisions Made

See `key-decisions` in the frontmatter for the four load-bearing calls: the mechanical (not branch-gated) `legacy_deleted` split, the peek-before-write attribution guard in the cutover command, `project_allocation()`'s new third return value for dry-run parity, and the pre-existing container-title staleness finding.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, found by this task's own real-database run] `project_allocation()`'s dry-run preview double-counted a legacy night the per-night loop already accounted for**
- **Found during:** Task 3, comparing the sweep's `--dry-run` output (`would_delete_legacy: 56`) against the real run's output (`legacy_deleted: 8`) over the same starting state — a mismatch the real-data proof exists to catch, not visible from Task 1's own unit tests (none of which combined an allocation-dispatched run's in-window legacy takeover with the new date-bearing delete preview in the same dry-run call).
- **Issue:** `project_allocation()`'s per-night loop identifies a legacy `RUN:{pk}:{night}` event for takeover (rekey) or retirement (delete) but, under `dry_run=True`, never actually writes the url change or deletion — the row is still `RUN:`-prefixed in the database when `campaign_reconciler`'s new `_stale_dated_events()` preview runs immediately afterward in the SAME `reconcile_run()` call, so the same url was counted once as `would_rekey`/`would_retire` and again as `would_delete_legacy`. Real mode was never affected: the write happens before the delete-preview step runs, so the url has already left the `RUN:` namespace by the time it would matter.
- **Fix:** `project_allocation()`'s return signature grew a third value, `legacy_urls_claimed` — every legacy url the per-night loop decided the fate of this call, in either mode. `campaign_reconciler.reconcile_run()`'s dry-run branch now excludes this set before computing `legacy_deleted`.
- **Files modified:** `solsys_code/allocation_projector.py`, `solsys_code/campaign_reconciler.py`
- **Verification:** Re-ran the full four-step sequence from a fresh scratch copy; `--dry-run`'s `would_delete_legacy: 8` now matches the real run's `legacy_deleted: 8` exactly. Added two regression tests in `test_campaign_reconciler.py` (`TestLegacyPerNightFamilyDeletion`) pinning both the takeover and the retire cases, plus a `would_delete_legacy: 0` assertion added to the pre-existing `test_dry_run_names_would_retire_and_would_rekey`.
- **Committed in:** `deb9142` (Task 3 commit)

**2. [Rule 1 - Bug, mechanical consequence of Task 1's own change] Existing tests exercising the old detach-only behavior for a date-bearing legacy event under a container-dispatched run**
- **Found during:** Task 1, running `test_campaign_reconciler.py`/`test_reconcile_campaign_runs.py`/`test_campaign_approval.py`'s own suites immediately after implementing the delete path
- **Issue:** Several pre-existing tests (`TestReconcileThenAttributeOrdering`'s staff-reconfirmation/reclaim/dry-run-preview tests, four tests in `test_reconcile_campaign_runs.TestSummaryCounters`, and `test_campaign_approval.test_resolve_that_detaches_something_shows_the_warning`) built a container-dispatched run plus a hand-made `RUN:{pk}:{date}` legacy event and asserted the OLD detach behavior — now that exact fixture shape is deleted, not detached.
- **Fix:** The three `TestReconcileThenAttributeOrdering` tests were rebuilt around a genuinely-reclassified bare-container fixture (a class-wide run reclassified to allocation dispatch), which still exercises the unchanged detach mechanism for the bare-container form; the `test_reconcile_campaign_runs.py` and `test_campaign_approval.py` tests were renamed and their assertions updated to expect `legacy_deleted`/a deletion warning instead of `detached`/a "released back into the attribution queue" message. `campaign_views._message_reconcile_side_effects()` gained the matching new warning clause.
- **Files modified:** `solsys_code/tests/test_campaign_reconciler.py`, `solsys_code/tests/test_reconcile_campaign_runs.py`, `solsys_code/tests/test_campaign_approval.py`, `solsys_code/campaign_views.py`
- **Verification:** All affected modules green; full label-list suite (Task 3) confirms no other consumer of the old wording was missed.
- **Committed in:** `7b2d25a` (Task 1 commit)

**3. [Interpretive refinement, not a defect] Foreign-attribution guard checked before, not after, the re-key**
- **Found during:** Task 2, implementing D-18's foreign-attribution reason
- **Issue:** The plan's own action text describes calling `adopt_event_into_run(event, run)` reactively and checking its `False` return to detect a foreign attribution — but by then the event would already have been re-keyed (`update_calendar_event_key_and_fields()` ran first), contradicting D-18's "left byte-identical" guarantee for exactly this case.
- **Fix:** The command peeks at `CalendarEventMeta.filter(event=event).first()` BEFORE attempting any write, mirroring `project_allocation()`'s own `_may_write()`-first idiom, and never re-keys an event it cannot safely claim.
- **Files modified:** `solsys_code/management/commands/cutover_classical_allocations.py`
- **Verification:** `TestForeignAttributionLeftUntouched` (bonus test, beyond the plan's 8 named tests) confirms the foreign-attributed event's url is unchanged and its foreign attribution survives, while its group's other events still convert.
- **Committed in:** `20198ef` (Task 2 commit)

---

**Total deviations:** 3 (2 auto-fixed Rule 1 bugs — one a genuine correctness gap only the real-database run could have found, one a mechanical test-suite consequence of Task 1's own change — and 1 interpretive refinement documented as a decision, not a defect). **Impact:** All necessary for correctness; no scope creep. The dry-run double-counting fix is the clearest illustration of why Task 3's real-data proof exists: no unit test in this phase combined the two code paths that interacted to produce it.

### Found, Not Fixed (Pre-existing, Out of Scope)

**8 of the developer database's 16 `RUN:{pk}` containers had a stale, pre-Phase-33 title.**
- **Found during:** Task 3, comparing container event fields before and after the real sweep.
- **Issue:** These 8 containers (the single-night `lco_queue`/`eso_queue` runs pk 2, 3, 7, 14, 15, 18, 19, 20) carried a `'<Campaign>: '` title prefix from before Phase 33's D-12 change (2026-09-10) stopped `event_title()` from embedding a campaign label — this plan's sweep was the first reconcile these 8 rows had been through since that change shipped, so their titles corrected (and their `modified` timestamps updated) on this run.
- **Why not fixed here:** `_reconcile_container()`'s title-writing logic (`event_title()`) is completely unchanged by Phase 35 — this is Phase 33's own already-shipped correction reaching data that happened not to be reconciled since. Fixing it would mean back-dating a title correction that already exists in the code; there is nothing to fix.
- **Disposition:** Recorded here and in 35-VALIDATION.md's Manual-Only Verifications evidence so a human reviewing the before/after diff recognises it rather than mistaking it for a Phase 35 regression. No CalendarEvent count, url, start_time, or end_time changed for these 8 rows — only `title` and `modified`.

## Issues Encountered

None blocking beyond the deviations documented above, all caught and fixed by this plan's own verification gates before this SUMMARY was written.

## TDD Gate Compliance

Tasks 1 and 2 carry `tdd="true"`. `workflow.tdd_mode` is `false` in this project's config, so the runtime MVP+TDD halt gate did not apply.

**Both tasks' actual shape:** implementation and its tests were written and iterated together, verified fully green (all target tests passing, all `<verify>` commands matching expected output) before a single `feat(35-06)` commit per task — no separate `test(35-06)` RED commit exists for either task, and no RED-evidence record was captured via `gsd_run check tdd-red-evidence`, matching the disclosed pattern in 35-01's and 35-05's own SUMMARYs for the same reason: splitting a rewrite this size into a meaningful RED state against pre-existing code would have required either duplicating fixtures for two incompatible shapes or writing throwaway assertions.

**Disposition:** no code or test defect results from this — every acceptance criterion and `<verify>` command for both tasks passes as committed, and Task 3's real-database run (the phase's genuine backstop) caught the one real gap (the dry-run double-count) that neither task's own unit-test-first discipline reached. Flagged here per the tdd.md gate-enforcement contract rather than omitted.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ALLOC-05 and ROADMAP Success Criterion 5 hold: the calendar shows one event per night after the stated four-step cutover, proven against both synthetic fixtures and the real developer database, with no duplicate and no orphan left behind.
- This is the last expansion slice of the tracer proven in 35-01 — the `ALLOC:` path now absorbs the events that existed before the path did, and the legacy `RUN:{pk}:{date}` family is fully retired (rekeyed or deleted) for every run this sweep reached.
- Plan 35-07 (wave 5) owns the paired docs this plan intentionally deferred: `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (the before/after diff this plan's own real-data run measured, reproduced there as executed output with `assert` statements per 35-VALIDATION.md), and `docs/runbooks/telescope_runs_calendar.rst`'s new cutover section — this SUMMARY's "Operator-Visible Four-Step Sequence" and before/after table are the source material for both.
- No blockers. The 8-container title-staleness finding is informational only, not a blocker, and needs no follow-up (see "Found, Not Fixed" above).

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/management/commands/cutover_classical_allocations.py
- FOUND: solsys_code/tests/test_cutover_classical_allocations.py
- FOUND commit: 7b2d25a
- FOUND commit: 20198ef
- FOUND commit: deb9142
- All plan-level `<acceptance_criteria>` and `<verify>` commands re-run and passing (see task-by-task output above)
- Full label-list regression suite (`workflow.test_command`): 1215 tests, OK (1 pre-existing skip); plus 40 `test_views` tests, OK
- `src/fomo_db.sqlite3` confirmed unmodified: `git status --porcelain` empty, md5sum identical before and after
