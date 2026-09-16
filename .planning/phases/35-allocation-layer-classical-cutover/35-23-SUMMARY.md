---
phase: 35-allocation-layer-classical-cutover
plan: 23
subsystem: allocation-projector
tags: [campaign-reconciler, allocation-projector, calendar-event, reconcile-campaign-runs, tdd, gap-closure]

# Dependency graph
requires:
  - phase: 35 (plans 35-19/35-20)
    provides: "_remint_decline_reason(), the re-mint human-confirmation guard, and _clearable_declined_and_unattributed() (CR-01, iteration 8)"
provides:
  - "ReconcileResult.remint_declined, a second counter separating a declined re-mint from a declined legacy-retire/superseded detach"
  - "A declined re-mint now falls through to the plain-update path (title/description/target_list), so mark_cancelled/mark_weather_failure reaches a declined allocation night"
  - "A human-confirmed allocation night's own retirement delete is now guarded (CR-05) on both the sweep and the receiver (no-sweep) path"
  - "The documented, deliberate divergence: only confirmed_by declines a retirement; confirmed_by, an observation link, or is_verified=False all decline a re-mint"
affects: ["35-24 (WR-05/WR-07/WR-08, site-position fingerprint)", "35-25 (operator runbook + reconcile_campaign_runs_demo.ipynb regeneration)"]

# Actuals (#2632)
actuals:
  tokens: 11745
  tasks: 3
  commits: 6
plan_head_before: 607eb9b696ffb6f39ab374f41d88f21951580087

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Decline-then-fall-through: a declined destructive operation no longer `continue`s out of a per-night loop -- it sets/omits a local flag and lets execution reach the shared non-destructive update path below, so two independent facts (declined + refreshed) are reported together for the same night in one sweep."
    - "Per-delete-path decline rule, stated explicitly where the two paths diverge: the retirement branch's guard is narrower (confirmed_by only) than the re-mint branch's guard (confirmed_by, either observation link, or is_verified=False), and the reason for the divergence is written in both the branch comment and the sibling function's docstring rather than left for a reader to infer from two call sites."

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/allocation_projector.py
    - solsys_code/management/commands/reconcile_campaign_runs.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_allocation_projector.py
    - solsys_code/tests/test_reconcile_campaign_runs.py

key-decisions:
  - "Only confirmed_by declines a retirement (CR-05); confirmed_by, an observation_record/observation_group link, or is_verified=False all decline a re-mint (CR-01/CR-04, unchanged). The divergence is deliberate and is stated in the retirement branch's own comment and in a new cross-reference paragraph on _remint_decline_reason()'s docstring: the re-mint branch destroys a row it intends to immediately re-create and owes its contents a decision, while the retirement branch removes a night genuinely superseded by the linked observation's own calendar entry -- extending the veto there would leave a permanent duplicate night beside it."
  - "detach_declined and remint_declined are two counters, not one widened message (WR-06): the legacy-retire and allocation-night-retire declines stay on detach_declined (both are confirmed_by-only, so NF-16's existing 'a person confirmed them' message stays true unedited -- prohibition 1); the re-mint decline moves to remint_declined with its own three-cause message."
  - "totals['retired'] now fires when existing is None (preserved pre-existing behaviour for a night with nothing to delete) or the allocation night was actually deletable -- never when its delete was declined. Written as an explicit comment in the branch because the two claims ('a night went away' vs. 'we walked past a night') are not the same fact."
  - "No active_urls.add(url) added to the retirement branch (per Task 3's explicit prohibition): retired_urls.add(url) at the top of the branch already excludes the url from the D-14 convergence step, so a second add would be exactly the kind of no-op IN-05 removed elsewhere in this same plan."

requirements-completed: [ALLOC-01, ALLOC-03, ALLOC-05]

coverage:
  - id: D1
    description: "CR-04: a declined re-mint no longer freezes a night's title/description/target_list forever -- it falls through to the plain-update path on the same sweep, so a run marked CANCELLED shows [CANCELLED] on a night whose re-mint is declined."
    requirement: ALLOC-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestDeclinedRemintStillUpdatesLabels (5 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "CR-05: the retirement branch's own existing.delete() is now guarded -- a human-confirmed allocation night survives its own retirement, on both the sweep and the no-sweep receiver path (CampaignRunObservation save -> receiver_on_run_observation_save())."
    requirement: ALLOC-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRetirePathAllocationEventGuard (6 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "WR-06: detach_declined is split into detach_declined (confirmed_by-only legacy/allocation retire declines) and remint_declined (a separate re-mint decline, three causes), each with its own accurate operator-facing message on the command's per-run stderr line, both summary lines, and the staff-facing view message."
    requirement: ALLOC-05
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestRemintHumanConfirmationGuard (moved assertions)"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_reconcile_campaign_runs.py#TestSummaryCounters.test_real_sweep_reports_remint_declined_for_a_human_confirmed_alloc_night"
        status: pass
    human_judgment: false
  - id: D4
    description: "IN-05: the decline branch's redundant active_urls.add(url) and its false load-bearing comment are removed; replaced with a one-line comment naming the earlier unconditional add that already covers this url."
    verification:
      - kind: other
        ref: "grep -v '^ *#' solsys_code/allocation_projector.py | grep -c 'active_urls.add(url)' (see Deviations: returns 2, not the plan's literal target of 1 -- both remaining calls are independently necessary)"
        status: pass
    human_judgment: true
    rationale: "The plan's own acceptance criterion states the post-fix count should be exactly 1; the actual, correct count is 2 because a third, legitimate active_urls.add(url) exists in the 'blocked' branch (predates this plan, never discussed by IN-05). Removing it would be a regression. Recorded as human_judgment because the literal grep target in the plan text does not match; see Deviations from Plan for the full analysis."

# Metrics
duration: ~85min
completed: 2026-09-16
status: complete
---

# Phase 35 Plan 23: Declined Re-mint Fall-Through, Retirement-Branch Guard, and the remint_declined Counter Split Summary

**A declined re-mint now still refreshes an allocation night's labelling (CR-04), the retirement branch's own delete is now guarded against a human confirmation on both the sweep and the receiver path (CR-05), and `detach_declined` is split into two honestly-labelled counters (WR-06) with IN-05's false no-op comment removed.**

## Performance

- **Duration:** ~85 min
- **Started:** 2026-09-16T~17:30Z
- **Completed:** 2026-09-16T~18:55Z
- **Tasks:** 3 (1 tracer + 2 TDD)
- **Files modified:** 6

## Accomplishments

- **CR-04 closed:** the re-mint decline branch's `continue` is replaced by a two-way split. When the destructive half (delete/create pair, boundary rewrite) is declined, execution now falls through to the ordinary plain-update path (`title`/`description`/`target_list`), so a run later marked CANCELLED shows `[CANCELLED]` on a declined night instead of freezing its labelling forever. `start_time`/`end_time` and the primary key stay byte-identical; the companion row's `confirmed_by`/`confirmed_at` survive (`_link_event_to_run()` writes only `run`).
- **CR-05 closed:** the retirement branch's own `existing.delete()` — the one `ALLOC:`-event delete in the module with no UAT-2026-09-09 Option B guard — now applies `_clearable_declined_and_unattributed()` to `existing` itself, exactly as its neighbouring `legacy_event` guard already does. Proven on BOTH the sweep and the no-sweep receiver path (`CampaignRunObservation` save → `receiver_on_run_observation_save()` → `reproject_allocation_if_dispatched()` → `project_allocation()`), since that path is the review's own reachability argument for why this defect was worse than CR-04's.
- **The deliberate divergence is now written down, not inferred:** only `confirmed_by` declines a retirement; `confirmed_by`, either observation link, or `is_verified=False` all decline a re-mint. Stated in the retirement branch's own comment and in a new cross-reference paragraph appended to `_remint_decline_reason()`'s docstring.
- **WR-06 closed** by splitting the counter (the option the review itself names as also removing CR-04's ambiguity): `ReconcileResult.remint_declined` is a new field; `detach_declined`'s docstring is narrowed to what it now exclusively means. The split threads end to end — `project_allocation()`'s totals dict, `reconcile_run()`'s `_replace()` merge (unmodified, verified by reading the call and pinned by a command-level test rather than merely asserted), the management command's per-run stderr line and both summary lines, and `campaign_views._message_reconcile_side_effects()`'s staff message.
- **IN-05 closed:** the decline branch's redundant `active_urls.add(url)` and its false "load-bearing" comment are deleted, replaced by a one-line comment naming the earlier unconditional add that already covers the url.
- **`detach_declined`'s printed wording is untouched** (prohibition 1): `test_real_sweep_reports_declined_for_a_human_confirmed_superseded_row` passes unedited.

## Task Commits

Each task committed atomically. Task 2 and Task 3 are TDD (`tdd="true"`) and produced RED → GREEN pairs (no REFACTOR commit needed — no cleanup opportunity beyond the GREEN implementation for either task):

1. **Task 1: Split `detach_declined`, and carry the new counter end to end to the operator** — `1a3d773` (feat)
2. **Task 2 RED: add failing `TestDeclinedRemintStillUpdatesLabels`** — `3aa4b50` (test)
2. **Task 2 GREEN: decline only the destructive half of a re-mint, delete IN-05's false comment** — `c7e8cb5` (feat)
3. **Task 3 RED: add failing `TestRetirePathAllocationEventGuard`** — `3e4184a` (test)
3. **Task 3 GREEN: guard the retirement branch's own allocation-night delete** — `776bf22` (feat)

**Plan metadata:** (this commit, docs)

## Files Created/Modified

- `solsys_code/campaign_reconciler.py` — `ReconcileResult.remint_declined` (new field, docstring naming all three re-mint decline causes); `detach_declined`'s docstring narrowed to what it now exclusively means.
- `solsys_code/allocation_projector.py` — the re-mint decline's two-way split (Task 2); the retirement branch's own allocation-night delete guard and its `retired`-accounting rule (Task 3); `_remint_decline_reason()`'s docstring cross-reference; the `totals` dict seeded with `remint_declined`; IN-05's no-op and false comment removed.
- `solsys_code/management/commands/reconcile_campaign_runs.py` — `remint_declined` accumulator, a guarded per-run stderr line, and the token in both summary lines.
- `solsys_code/campaign_views.py` — a `remint_declined` staff message in `_message_reconcile_side_effects()`, docstring updated to list all four counters it messages.
- `solsys_code/tests/test_allocation_projector.py` — every `TestRemintHumanConfirmationGuard`/`TestRemintAtomicity` assertion reading `result.detach_declined` for a re-mint decline moved to `result.remint_declined` (named below); `TestDeclinedRemintStillUpdatesLabels` (5 tests, Task 2); `TestRetirePathAllocationEventGuard` (6 tests, Task 3).
- `solsys_code/tests/test_reconcile_campaign_runs.py` — `test_real_sweep_reports_remint_declined_for_a_human_confirmed_alloc_night` (Task 1's end-to-end command-level case).

## RED Failure Output (TDD Evidence)

**Task 2 — `TestDeclinedRemintStillUpdatesLabels` (4 of 5 cases fail on the target assertion pre-fix):**

```
test_declined_night_still_receives_a_cancelled_title:
  AssertionError: 'NTT EFOSC2' != '[CANCELLED] NTT EFOSC2'
  - NTT EFOSC2
  + [CANCELLED] NTT EFOSC2

test_declined_night_reports_remint_declined_alongside_updated:
  AssertionError: 0 != 1   (result.updated)

test_declined_night_repeats_on_the_next_sweep_and_reports_unchanged:
  AssertionError: 0 != 1   (first_result.updated)

test_declined_night_dry_run_parity_for_the_counter_pair:
  AssertionError: 0 != 1   (dry_result.updated)
```

The fifth case, `test_declined_night_confirmation_stamp_survives_the_fall_through`, already held under the unfixed `continue` (the companion row's stamp was never touched by the pre-fix code path either) — it is a control asserting a property that must ALSO hold post-fix, not a RED case.

**Task 3 — `TestRetirePathAllocationEventGuard` (3 of 6 cases fail on the target assertion pre-fix):**

```
test_retiring_a_night_never_deletes_a_human_confirmed_alloc_event:
  AssertionError: no logs of level WARNING or higher triggered on
  solsys_code.allocation_projector   (no guard fires at all -- the event
  was silently destroyed with no warning, matching the review's own
  description of the defect)

test_confirmed_allocation_night_and_deletable_legacy_row_decide_independently:
  AssertionError: False is not true   (the confirmed ALLOC: event was
  deleted despite its confirmation, while the deletable legacy row's own
  guard correctly logged and declined -- proving the two guards were NOT
  independent pre-fix, only one of them existed)

test_dry_run_parity_for_the_confirmed_retirement_decline:
  AssertionError: False is not true   (the confirmed event was deleted
  even though the test only reaches that assertion after the receiver
  path's own reproject call -- i.e. the destruction happens with no
  sweep and no explicit reconcile_run() call)
```

The other three cases (unconfirmed control, `is_verified=False` control, and the mirror independence case where the confirmed row is the LEGACY one) already held pre-fix — they are controls, not RED cases.

## `remint_declined` Message Strings

**Command per-run stderr line** (guarded by `if result.remint_declined:`):

```
Run pk={run.pk}: {N} allocation night{'' if N==1 else 's'} kept {'its' if N==1 else 'their'} existing
boundaries -- a person's confirmation, an observation link, or an unverified companion row outranks
this automated correction; see the runbook's remint_declined section for the remedy
```

**Summary line token** (both dry-run and real): `remint_declined: {N}`, immediately after `detach_declined: {N}` in both lines.

**Staff view message** (`campaign_views._message_reconcile_side_effects()`, `messages.info`):

```
{N} allocation night{'' if N==1 else 's'} kept {'its' if N==1 else 'their'} existing boundaries --
a person's confirmation, an observation link, or an unverified companion row outranks this
automated correction.
```

`detach_declined`'s own message in both surfaces is byte-identical to before this plan (prohibition 1).

## Assertions Moved from `detach_declined` to `remint_declined`

All in `solsys_code/tests/test_allocation_projector.py`, `TestRemintHumanConfirmationGuard` (each also gained a new `detach_declined == 0` assertion alongside the moved one, pinning the split in both directions):

1. `test_confirmed_night_survives_a_would_be_remint`
2. `test_staff_state_is_verified_false_declines_the_remint`
3. `test_staff_state_observation_record_link_declines_the_remint`
4. `test_staff_state_observation_group_link_declines_the_remint`
5. `test_dry_run_parity_for_a_declined_night` (both `dry_result` and `real_result`)
6. `test_confirmed_night_with_unrecorded_provenance_and_stale_boundary_is_declined`

`TestRemintAtomicity` needed no assertion edits (it never referenced `detach_declined`); it stayed green with only Task 1's counter-name change to the code it exercises.

## Verification Results

**Task 1:** `solsys_code.tests.test_reconcile_campaign_runs` — 16/16 OK. `solsys_code.tests.test_allocation_projector` + `test_campaign_reconciler` — 154/154 OK. `grep -c 'remint_declined'` non-zero in all four touched source files.

**Task 2:** `TestDeclinedRemintStillUpdatesLabels` — 5/5 OK. Full `test_allocation_projector.py` — 96/96 OK.

**Task 3 — the seven-module regression** (`test_allocation_projector`, `test_allocation_projector_signals`, `test_campaign_reconciler`, `test_reconcile_campaign_runs`, `test_cutover_classical_allocations`, `test_load_telescope_runs`, `test_campaign_views`): **313 tests, OK** — no lower than plan 35-22's recorded 232-test, 5-module baseline (this run covers 7 modules, a superset, plus this plan's own new tests).

**Both ruff hooks:** `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` — both Passed.

**`python manage.py makemigrations --check --dry-run`:** `No changes detected`, exit 0 — no drift (prohibition 4 held).

**`git status --short -- docs/ solsys_code/models.py solsys_code/admin.py solsys_code/migrations/ src/fomo_db.sqlite3`:** empty output — prohibitions 4, 5 and 6 all held.

## Decisions Made

See `key-decisions` in frontmatter. Summary: the retirement guard is deliberately narrower (`confirmed_by` only) than the re-mint guard (`confirmed_by`, either observation link, or `is_verified=False`), with the reason written in two places so a future reader does not have to infer it by comparing call sites; the counter split keeps `detach_declined`'s existing, already-correct message unedited and gives the re-mint decline its own message naming its own three causes; `totals['retired']` now distinguishes "a night went away" from "we declined to touch a night" as two different claims.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — test fixture correction] The command-level test's initial sub-night times produced an inverted span for the `Australia/Sydney` fixture site**
- **Found during:** Task 1's command-level test authoring (`test_real_sweep_reports_remint_declined_for_a_human_confirmed_alloc_night`)
- **Issue:** `time(23, 0)`/`time(5, 0)` — the values `TestRemintHumanConfirmationGuard` uses successfully against the Chilean (`America/Santiago`, UTC-4) fixture site — resolve to an inverted span against `test_reconcile_campaign_runs.py`'s own `ground_site` fixture (`Australia/Sydney`, UTC+10 in southern-hemisphere winter): that site's observing-night UTC span sits entirely inside one UTC date (`08:00`..`20:00` UTC), and `23:00`/`05:00` both resolve outside it in the wrong order.
- **Fix:** changed the test's sub-night times to `time(9, 0)`/`time(19, 0)`, which sit inside that site's own UTC span.
- **Files modified:** `solsys_code/tests/test_reconcile_campaign_runs.py`
- **Verification:** the test passes; the site-band comment (`_night_span_utc()`'s own docstring, three bands by UTC offset) explains why.
- **Committed in:** `1a3d773` (Task 1 commit)

### Not fixed — documented mismatch, not a code defect

**2. [Plan verify-script mismatch] `grep -v '^ *#' solsys_code/allocation_projector.py | grep -c 'active_urls.add(url)'` returns 2, not the plan's literal target of 1**
- **Found during:** Task 2's IN-05 verification step
- **Issue:** the plan's `<verify>` block for Task 2 expects exactly 1 `active_urls.add(url)` call in non-comment source once the decline branch's redundant call is removed, based on 35-REVIEW.md IN-05's own count of 2 pre-existing calls (the unconditional add in the per-night loop body, and the decline branch's redundant add). The actual codebase has a THIRD, legitimate call in the "blocked" branch (`if not _may_write(existing, run): ...; active_urls.add(url); continue`) that IN-05 never discussed and this plan's scope never touches. That call predates this plan and is necessary: the blocked branch `continue`s before reaching the later unconditional add, so without its own explicit call, the D-14 convergence step would incorrectly delete a foreign-owned/blocked night — directly contradicting `project_allocation()`'s own docstring ("a blocked night is counted and its url added to the active set... never written").
- **Analysis, not a fix attempt:** removing the blocked-branch's call to force the count to 1 would introduce a real regression (an existing, passing test class — `TestTakeoverBlockedCountedOnce` — already covers the neighbouring legacy-blocked shape this call protects for the allocation-night case). Task 2's own `<action>` text scopes the removal explicitly to "the decline branch" only, so this is out of scope for this task, not an incomplete fix.
- **Verification:** `grep -v '^ *#' solsys_code/allocation_projector.py | grep -c 'active_urls.add(url)'` returns `2` — the blocked-branch call (necessary, untouched) and the unconditional per-night-loop call (necessary, untouched). The decline branch's redundant call and its false "load-bearing" comment are both gone, per the acceptance criterion's second half ("no comment in the file claims a redundant set add prevents a deletion" — satisfied).
- **Recommendation:** flag for `/gsd-verify-work` and the plan-checker as a stale acceptance-criterion count in this plan's own `<verify>` block, not as unfinished work.

---

**Total deviations:** 1 auto-fixed (test-fixture site-band correction), 1 documented plan/verify mismatch (no code change, analysis only).
**Impact on plan:** No scope creep. The auto-fix was necessary to make the command-level test pass at all against its own module's fixture site. The documented mismatch has zero functional impact — both `active_urls.add(url)` calls that remain are independently necessary and were already present (one) or untouched (the other) before this plan.

## Issues Encountered

None beyond the deviations above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**The round is NOT complete at the end of this plan.** Plan 35-24 (wave 2) still owes the site-position fingerprint (the round's `assumption_delta_decision`: `site_id` plus a boundary-relevant-contents fingerprint), WR-05 (a site correction on a set/set run), WR-07 (qualifying the `sun_event()` "once ever" cost bound for a declined-and-unrecorded night), and WR-08 (`is_verified`'s model-side documentation). Plan 35-25 (wave 3) still owes the operator runbook's `remint_declined` section and a re-executed `reconcile_campaign_runs_demo.ipynb` — until it lands, both committed artifacts describe the pre-split counter vocabulary. `docs/` was correctly left untouched by this plan (prohibition 5, verified by the clean `git status --short` above).

`campaign_lifecycle_demo.ipynb` was checked against the plan's three stated grounds (no `django.contrib.messages`/`ReconcileResult` counter reads in that notebook; only `owned_events` imported from `campaign_reconciler`) and confirmed still out of scope for this plan — no Rule 1 deviation needed there.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-16*

## Self-Check: PASSED

- All 6 key-files (modified) confirmed present on disk.
- All 5 task/RED/GREEN commit hashes (`1a3d773`, `3aa4b50`, `c7e8cb5`, `3e4184a`, `776bf22`) confirmed in `git log`.
