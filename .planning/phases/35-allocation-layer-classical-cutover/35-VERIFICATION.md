---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-15T04:29:09Z
status: gaps_found
score: 74/76 must-haves verified
covered_files:
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-SUMMARY.md"
  - ".planning/REQUIREMENTS.md"
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/apps.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_utils.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/cutover_classical_allocations.py"
  - "solsys_code/management/commands/load_telescope_runs.py"
  - "solsys_code/management/commands/reconcile_campaign_runs.py"
  - "solsys_code/migrations/0018_campaignrun_night_window_fields.py"
  - "solsys_code/models.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/telescope_runs.py"
  - "solsys_code/tests/test_allocation_projector.py"
  - "solsys_code/tests/test_allocation_projector_signals.py"
  - "solsys_code/tests/test_campaign_approval.py"
  - "solsys_code/tests/test_campaign_reconciler.py"
  - "solsys_code/tests/test_cutover_classical_allocations.py"
  - "solsys_code/tests/test_load_telescope_runs.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_reconcile_campaign_runs.py"
  - "solsys_code/tests/test_telescope_runs.py"
  - "solsys_code/tests/test_write_and_reconcile.py"
covered_digest: "v1:sha256:b6a6452163d6d0646ce29b560693aba4b6019a70b029749a35570beb85b47317"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 76/76
  previous_verified: 2026-09-13T12:40:00Z
  human_items_closed:
    - "Cutover before/after diff in the reconciler demo notebook (35-UAT.md test 1 — pass)"
    - "Three-group reconciliation sums in the 35-06 real-database cutover record (35-UAT.md test 2 — pass)"
    - "Judgment-tier prohibition verdicts (35-UAT.md test 3 — pass)"
  security_review: "35-SECURITY.md 2026-09-15 — 18 threats, 17 closed, threats_open: 0 (T-35-17 open, below block_on: high)"
  fingerprint_stale: true
  fingerprint_stale_reason: "covered_files includes solsys_code/observation_projector.py, which commit 24875bf (fix(35): F-34-1) modified after the prior pass, and docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, which carries an uncommitted working-tree edit. Recomputed digest above."
  gaps_remaining: []
  regressions:
    - "NF-21 — commit d57b461 (this phase's own NF-08 fix) removed the only handler for ZoneInfoNotFoundError from load_telescope_runs, making the runbook's stated per-line skip-and-log invariant false"
    - "NF-24 — commits d57b461 and 30112a0 changed module behaviour without their CLAUDE.md-paired notebooks"
gaps:
  - truth: "The operator runbook's classical-ingest and cutover sections describe what the commands now do (35-07 truths 1-4)."
    status: failed
    reason: "The runbook states two operator-facing guarantees the shipped code does not hold. Both were reported in the phase's own committed code review (35-REVIEW.md, iteration 4, commit 8a393cf, status issues_found) and neither has a fix commit."
    artifacts:
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L939-943 — 'the SECOND group is never merged into the first group's run ... the first group's run and events are converted and left untouched either way'. Falsified by an executed probe (see Behavioral Spot-Checks): on the re-run the command itself prescribes at L968, the first group's run silently flips run_status planned -> cancelled and its observation_details is rewritten."
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L1382-1394 — 'one bad row never aborts the whole run' and the load_telescope_runs bullet naming only ValueError/Observatory.DoesNotExist. After commit d57b461, a mistyped Observatory.timezone raises ZoneInfoNotFoundError (a KeyError subclass, confirmed) from ZoneInfo(run.site.timezone) at allocation_projector.py:199/:575 and escapes the per-line handler at load_telescope_runs.py:337 entirely."
      - path: "solsys_code/management/commands/cutover_classical_allocations.py"
        issue: "L622 CommandError text 'then re-run this command -- it is safe to repeat', and the module docstring L47-48 'reported (never silently merged into the earlier group's run)' — both false on the second invocation."
    missing:
      - "NF-19: make the identity-key guard read the database, not only the in-process seen_keys dict — reject a group whose key is already held by a CampaignRun carrying a DIFFERENT stored 'Source line:'. Add a test that runs the command twice over the two-group fixture and asserts run_status/observation_details are unchanged by the second pass."
      - "NF-21: give the write/reconcile call its own `except ZoneInfoNotFoundError` clause (it is not a ValueError) so one malformed site timezone skips its line instead of aborting the batch; add a regression test asserting the following line is still processed and skipped: 1 is reported."
      - "NF-25: replace the duplicate_identity remedy text — this command reads no schedule file, so 'add a bracketed proposal token to one of the two lines' is un-actionable; the operator must edit the events' description 'Source line:' in the admin. Correct the three runbook passages that repeat it."
      - "Correct the runbook's per-line skip-and-log bullet once NF-21 is fixed."
  - truth: "load_telescope_runs_demo.ipynb runs against a throwaway copy of the developer database and shows, with real executed output, one CampaignRun per line and the ALLOC:-keyed nights the projector drew from it (35-07 truth 5) — and CLAUDE.md's paired-docs rule holds for every module this phase changed."
    status: failed
    reason: "CLAUDE.md's paired-docs rule (which explicitly directs the verifier to treat a missing or stale update as a must-have gap) was breached by two of this phase's own fix commits. Reported as NF-24 in 35-REVIEW.md; no fix commit."
    artifacts:
      - path: "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
        issue: "Last touched at 7d7b9d4. Commit d57b461 then changed the command's behaviour — a new per-line stderr message ('Line N: unknown classical status ...'), a new transaction.atomic() boundary, and a change to which exception classes are reported per-line versus aborting the batch. The notebook's cells print 'stderr (skipped lines)' verbatim, i.e. exactly the surface that changed."
      - path: "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
        issue: "Last touched at 8757750 (phase 34). Commit 30112a0 then reordered two signal-triggered writes in observation_projector.py so campaign attribution lands on the creating save rather than a later one — the precise behaviour the D-11 trigger demo exists to show."
    missing:
      - "Add a cell to load_telescope_runs_demo.ipynb exercising the unknown-status (and, once NF-21 is fixed, the invalid-timezone) skip path and its stderr text."
      - "Add a cell to project_observation_calendar_demo.ipynb asserting the attribution lands on the creating save."
      - "Regenerate both with `jupyter nbconvert --to notebook --execute --inplace` and commit with output."
  - truth: "The phase's own deep code review closes — 35-REVIEW.md reports no unresolved blocker."
    status: failed
    reason: "35-REVIEW.md (iteration 4, committed 8a393cf on 2026-09-14) carries `status: issues_found` with 1 BLOCKER and 6 WARNINGs. `git log` shows zero source commits addressing NF-19..NF-25 since; the only source change after that review is 24875bf, which is the phase-34 F-34-1 fix to observation_projector.py. Every finding was re-confirmed present in the current code by this verification."
    artifacts:
      - path: "solsys_code/management/commands/cutover_classical_allocations.py"
        issue: "NF-19 (BLOCKER) unfixed at L344/L387-395/L483-487 — seen_keys is an in-process dict; insert_or_create_campaign_run() at L487 find-and-updates against the database. IN-02 unfixed — seen_keys[key] is claimed at L395 ahead of the checks that decide whether the claiming group is convertible at all."
      - path: "solsys_code/allocation_projector.py"
        issue: "NF-20 unfixed at L679-687 — the _span_needs_remint() re-mint branch still does `if dry_run: continue` with no inversion check, while the create branch at L712-717 has one; a dry run over an operator-inverted window reports would_create/would_retire while the real run raises ValueError. NF-22 unfixed at L655-664 — the takeover branch's `continue` at L663 precedes `legacy_urls_claimed.add()` at L664, so a foreign-attributed legacy RUN:{pk}:{date} event is counted under `blocked` twice."
      - path: "solsys_code/campaign_reconciler.py"
        issue: "NF-23 unfixed at L678 — `-> tuple[int, int, int]` on _detach_stale_family_events(), which returns four values and whose own docstring says tuple[int, int, int, int]."
      - path: "solsys_code/tests/test_cutover_classical_allocations.py"
        issue: "IN-01 unfixed — the comment replacing NF-11's vacuous assertion hard-codes absolute line numbers."
    missing:
      - "Close NF-19 (BLOCKER), NF-20, NF-22, NF-23, IN-01, IN-02, and re-review to a clean 35-REVIEW.md."
      - "No test in test_cutover_classical_allocations.py covers a CampaignRun that already holds the identity key from a prior invocation or from load_telescope_runs — which is why 278 green tests do not catch NF-19."
advisory: []
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.

**Verified:** 2026-09-15T04:29:09Z
**Status:** gaps_found
**Re-verification:** Yes — first pass since the 2026-09-13 `human_needed` result. The prior report carried no `gaps:` block, so this ran as a full initial-depth verification rather than a staleness recheck.

---

## What changed since the prior pass

| Event | Outcome |
|-------|---------|
| 35-UAT.md (commit `1818ef6`) | All 3 human-judgment items **pass**, 0 issues. The three `human_needed` items from 2026-09-13 are closed. |
| 35-SECURITY.md (commit `352bd24`) | 18 threats, 17 closed, `threats_open: 0`. T-35-17 open but below `block_on: high`. |
| `24875bf` fix(35): F-34-1 | `observation_projector.py` hardened (the `campaign_run_links` lookup is now inside its own `try`). This phase's `allocation_projector.py` was untouched. |
| `8a393cf` docs(35): code review iteration 4 | **`status: issues_found` — 1 BLOCKER + 6 WARNINGs + 2 INFO. No fix commit has landed since.** |

The prior pass's three blocking human items are genuinely closed. **The phase does not pass anyway**, because the deep code review committed the day after that pass reports an unresolved BLOCKER that this verification reproduced independently, and because two CLAUDE.md-mandated paired-docs artifacts went stale inside this phase's own fix cycle.

### The CR-01..CR-06 question (explicitly asked)

**Confirmed genuinely resolved, by inspection of the current code — not by trusting the fix commits' subjects or 35-REVIEW-FIX.md.**

| Finding | Verdict in current code | Evidence |
|---|---|---|
| CR-01 dispatch/approval bypass | ✓ Closed | `reconcile_run()` `campaign_reconciler.py:836` is the only non-docstring `project_allocation()` call site outside the module itself; every receiver routes through `reproject_allocation_if_dispatched()` (`allocation_projector.py:794`). `grep` over `solsys_code/**.py` (tests excluded) returns no other call. |
| CR-02 re-classification orphans `ALLOC:` | ✓ Closed | Convergence step wired into both real and dry-run branches (`campaign_reconciler.py:864+`); D-14 delete covered by passing tests. |
| CR-03 unguarded legacy delete on retire | ✓ Closed | `_may_write()`-first idiom on the retire path; "Allocation retire blocked: legacy event pk=… is not owned by run pk=…" observed live in the test log. |
| CR-04 convergence deletes foreign/confirmed `ALLOC:` | ✓ Closed | Ownership-scoped; the residual NF-01 it introduced was closed by the `260913-ti1` quick task and re-confirmed by review iterations 3 and 4. |
| CR-05 admin bulk `QuerySet.delete()` escapes guard | ✓ Closed | `origin_model = getattr(origin, 'model', type(origin))` at `allocation_projector.py:907-908`, covering both the instance and the queryset form. |
| CR-06 inverted span east of UTC | ✓ Closed, and superseded by a stronger rule | `_time_of_day_to_datetime()` (`allocation_projector.py:205-239`) now picks the candidate nearest the site's own observing-night UTC span; the hard-coded `t.hour < 12` threshold is gone entirely, with `_raise_if_inverted()` (`:286-318`) as a backstop. |

The 35-UAT.md Gaps note ("resolve or triage CR-01..CR-06 before treating a passing UAT as phase completion") is therefore satisfied for CR-01..CR-06. It is **not** satisfied for the review iterations that came after them: iteration 4's NF-19..NF-25 and IN-01/IN-02 are all still present.

---

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Per-night events for a resolved-site awarded window; queue/class-wide/satellite keeps one whole-window entry | ✓ VERIFIED | `reconcile_run()` `campaign_reconciler.py:845-862` dispatches on `dispatches_per_night(run)`; container branch otherwise. Behavioural: `test_allocation_projector` + `test_campaign_reconciler` in the 278-test run below — OK. |
| 2 | Nights follow the site-local observing night, Chile and Australia | ✓ VERIFIED | `ZoneInfo(run.site.timezone)` at `allocation_projector.py:199`/`:575`; `_time_of_day_to_datetime()` resolves against the site's own UTC span with no hour threshold. `TestAllocationNightBoundary` passes for `America/Santiago` and `Australia/Sydney`. |
| 3 | Linking an `ObservationRecord` retires that night; unlinking restores it; the observation's own event is untouched | ✓ VERIFIED | Attribution writes route only through `adopt_event_into_run()`/`unlink_event_from_run()`; `test_allocation_projector_signals` (14 tests) and `test_observation_projector_signals` pass. |
| 4 | `load_telescope_runs` produces the same per-night calendar, via an allocation record, idempotently | ✓ VERIFIED | The module imports no calendar writer (`load_telescope_runs.py:1-12`); the only write path is `write_and_reconcile_campaign_run()`. `TestClassicalCalendarUnchangedByCutover` and the idempotence tests pass. |
| 5 | After the cutover, one event per night — no duplicate, no orphan | ✓ VERIFIED | Real-database evidence stands and was human-confirmed today (35-UAT.md tests 1 and 2 — pass): 241→233 total, `RUN:{pk}:{date}` 56→0, containers 16→16, `ALLOC:` 0→57, 159 facility-url events byte-identical; 48 rekeyed + 8 legacy_deleted + 0 retired = 56. Fixture proof: `TestCutoverSequenceContract` passes. **Caveat, not a falsification:** the count-and-orphan property this criterion names holds; the *operator procedure* that reaches it carries the NF-19 defect below. |

**ROADMAP score: 5/5**

### Plan-level Must-Have Truths

| Plan | Truths | Status | Evidence |
|------|--------|--------|----------|
| 35-01 | 16 + 1 backstop | ✓ 17/17 VERIFIED | Dispatch seam, D-10 container, both-hemisphere keying, retire/restore, D-08 both directions, `confirmed_by__isnull=True` guard, D-13 no-recompute, D-14 orphan delete, D-16 in-place re-key, delete cascade. Backstop closed by the 278-test run (all phase modules) below. |
| 35-02 | 6 + 1 backstop | ✓ 7/7 VERIFIED | `TestQueueSourceDispatchesToContainer` asserts the container url; surviving `RUN:{pk}:{date}` strings are pre-seeded legacy fixtures, not expected write-path output; the 81-class classification table in 35-02-SUMMARY.md accounts for all 6 retirements. |
| 35-03 | 7 | ✓ 7/7 VERIFIED | `night_start_utc`/`night_end_utc` present; `0018_campaignrun_night_window_fields.py` is two `AddField` ops and **no `RunPython`** (read in full). `_span_needs_remint()` delete-and-re-create, zero `sun_event()` on an unchanged reconcile. |
| 35-04 | 9 | ✓ 9/9 VERIFIED | Both receivers connected in `apps.py:64-74` with `weak=False` and unique dispatch_uids; no `post_save` on `CampaignRun`; never-raise contract logs `type(exc).__name__` only. 14 signal tests pass. |
| 35-05 | 10 | ✓ 10/10 VERIFIED | No calendar import; `source=CLASSICAL_FILE` + `APPROVED`; `_CLASSICAL_RUN_STATUS` maps status→`run_status` only; `_source_identifier()` includes the proposal token; in-file collision reported and skipped. |
| 35-06 | 10 + 1 backstop | ✓ 11/11 VERIFIED (literal) | Four-step runbook sequence, no `RunPython`, no `.delete()` call site, D-18's six unexplained categories, non-zero exit, `--dry-run` read-only. `TestSecondInvocationIsANoOp` passes, so truth 5's literal wording ("no-op after a **successful** run") holds — the NF-19 defect lives on the *unsuccessful*-run re-run path the command itself prescribes, which no truth names. Recorded as a blocker anti-pattern rather than a falsified truth. |
| 35-07 | 10 | ✗ 8/10 | Truths 2, 3, 6, 7, 8, 9, 10 verified (cutover section present with four ordered steps; source-correction consequences stated at L647-661; both notebooks committed with executed output — HEAD copies 14/14 and 17/17 code cells; CLAUDE.md L136 pairs the cutover command; `docs/index.rst:24` wires the runbook into the toctree). **Truth 1 FAILED** — the runbook's per-line skip-and-log bullet (L1382-1394) and its `duplicate_identity` passage (L939-943) both state guarantees the shipped code does not hold. **Truth 5 FAILED** — `load_telescope_runs_demo.ipynb` predates commit `d57b461`'s behavioural change to the command it demonstrates. |

**Plan-level score: 69/71**

**Overall score: 74/76 truths verified (0 present, behavior-unverified)**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/allocation_projector.py` | Module owning the `ALLOC:` namespace | ⚠️ VERIFIED (defects) | 927 lines (grown from 662 by the review-fix cycle). Imported and used by `campaign_reconciler`, `apps.py`, `observation_projector`, `models.py`, `cutover_classical_allocations`. Wired, data flowing. NF-20 and NF-22 remain open in it. |
| `solsys_code/tests/test_allocation_projector.py` | ALLOC-01/02/03 coverage | ✓ VERIFIED | Passing in the 278-test run. |
| `solsys_code/migrations/0018_campaignrun_night_window_fields.py` | Additive, no data step | ✓ VERIFIED | 23 lines, two `AddField` ops, no `RunPython`. |
| `solsys_code/tests/test_allocation_projector_signals.py` | Trigger contract | ✓ VERIFIED | 14 tests, passing. |
| `solsys_code/management/commands/load_telescope_runs.py` | Writes allocations, not events | ⚠️ VERIFIED (defect) | 360 lines; no calendar import. NF-21 open: `except (ValueError, Observatory.DoesNotExist)` at L337 no longer covers `ZoneInfoNotFoundError`. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | One-time cutover command | ✗ DEFECTIVE | 624 lines; registered and runnable. NF-19 (BLOCKER) and IN-02 open. |
| `solsys_code/tests/test_cutover_classical_allocations.py` | Cutover coverage | ⚠️ INCOMPLETE | 28 tests, all passing — but none covers a `CampaignRun` that already holds the identity key from a prior invocation, which is the NF-19 hole. |
| `solsys_code/tests/test_campaign_reconciler.py` | Migrated onto `ALLOC:`/D-10 | ✓ VERIFIED | Passing. |
| `solsys_code/tests/test_reconcile_campaign_runs.py` | Summary counters | ✓ VERIFIED | Passing. |
| `docs/runbooks/telescope_runs_calendar.rst` | Cutover + ingest + source sections | ✗ INACCURATE | 1447 lines, toctree-wired. Two operator-facing guarantees false (L939-943, L1382-1394). |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Executed allocation-path demo | ✗ STALE | 14/14 code cells with output, but last regenerated at `7d7b9d4`, before `d57b461` changed the command's stderr surface and transaction boundary. |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Executed cutover before/after diff | ⚠️ VERIFIED at HEAD, dirty in tree | Committed copy: 17/17 code cells with non-null execution counts and output. **Working tree** carries an uncommitted hand-edit that nulls the execution counts on code cells 0 and 15 (`[None, 2..15, None, 17]`) — T-35-17. Committing as-is would break 35-07 truth 8. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `campaign_reconciler.reconcile_run()` | `allocation_projector.project_allocation()` | single dispatch seam (D-09) | ✓ WIRED | `campaign_reconciler.py:836`, guarded by `dispatches_per_night(run)`; the only non-container branch. |
| `allocation_projector` | `telescope_runs.observing_night()` / `sun_event()` | shared site-local night anchor (D-05) | ✓ WIRED | `ZoneInfo(run.site.timezone)` at `:199`, `:575`. |
| `allocation_projector` | `campaign_utils.adopt_event_into_run()` / `unlink_event_from_run()` | only attribution writers (D-08) | ✓ WIRED | No direct `meta.run =` in the bridge. |
| `allocation_projector` | `calendar_utils` writers | only calendar writers | ✓ WIRED | `insert_or_create_calendar_event()` / `update_calendar_event_key_and_fields()`. |
| `CampaignRun.night_start_utc/night_end_utc` | per-night span | `night_bounds()` | ✓ WIRED | Consumed in `_mint_fields()`. |
| `AttributionDecisionView` | `CampaignRunObservation` post_save/post_delete → `project_allocation()` | D-11 | ✓ WIRED | `apps.py:64-74`, `weak=False`, unique dispatch_uids. |
| `ObservationRecord.save()` | linked-run re-project | `observation_projector.receiver_on_record_save()` | ✓ WIRED | Hardened by `24875bf`; `test_observation_projector_signals` passes. |
| `load_telescope_runs.handle()` | `write_and_reconcile_campaign_run()` → `reconcile_run()` → `project_allocation()` | ALLOC-04 chain | ✓ WIRED | Now inside `transaction.atomic()` (`d57b461`). |
| `cutover_classical_allocations` | `telescope_runs.parse_run_line()` / `get_site()` | same parse as ingest | ✓ WIRED | Import at `:116`. |
| `cutover_classical_allocations` | `campaign_utils.insert_or_create_campaign_run()` | find-or-create on `source_identifier` | ⚠️ WIRED, UNGUARDED | `:487`. The lookup matches the **database**; the only collision guard (`seen_keys`, `:344`) is in-process. This is the NF-19 seam. |
| `docs/index.rst` toctree | `runbooks/telescope_runs_calendar` | operator-reachable page | ✓ WIRED | `docs/index.rst:24`. |
| `CLAUDE.md` notebook map | `cutover_classical_allocations.py` → `reconcile_campaign_runs_demo.ipynb` | paired-docs enforceability | ✓ WIRED | `CLAUDE.md:136`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `allocation_projector._mint_fields()` | `start`/`end` | `sun_event(run.site, night)` via `night_bounds()` | Yes — real astropy spans in the executed notebook | ✓ FLOWING |
| `allocation_projector` | `target_list` | `run.campaign` FK | Yes | ✓ FLOWING |
| `allocation_projector.retired_nights()` | `nights` | `run.observation_links` → `record.scheduled_start/end` | Yes | ✓ FLOWING |
| `load_telescope_runs` | `night_start_utc`/`night_end_utc` | parsed `BoN`/`EoN`/`HHMM` tokens | Yes | ✓ FLOWING |
| `cutover_classical_allocations` | run fields | re-parsed `Source line:` from each event's own description | Yes — probe created a real run and re-keyed 3 events | ✓ FLOWING |
| `allocation_night_title()` | event title | `run.run_status` → `_RUN_STATUS_CALENDAR_PREFIX` | Yes — **and this is what makes NF-19 calendar-visible**: a silently flipped `run_status` retitles every one of that run's nights `[CANCELLED] …` on the next sweep (step 4 of the documented cutover sequence) | ✓ FLOWING |
| `campaign_views.py:750` | `result.skipped_nights` | nothing — no code path assigns it any more | No | ⚠️ STATIC (cosmetic, pre-disclosed) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All phase-35 test modules | `python manage.py test solsys_code.tests.test_allocation_projector test_allocation_projector_signals test_cutover_classical_allocations test_load_telescope_runs test_campaign_reconciler test_reconcile_campaign_runs test_observation_projector_signals test_telescope_runs` | **Ran 278 tests in 443.6s — OK** (exit 0) | ✓ PASS |
| **NF-19 reproduction (verifier's own probe, run in this process)** | `PYTHONPATH=<scratch> python manage.py test probe_nf19` | Pass 1 over two status-only-colliding groups: correct — `duplicate_identity=3`, non-zero exit, run pk=1 left at `run_status='planned'` / `'Status: allocation'`. **Pass 2 — the re-run the CommandError itself prescribes ("it is safe to repeat") — silently flips run pk=1 to `run_status='cancelled'` and rewrites `observation_details` to `'Status: cancelled'`, while its three `ALLOC:` events still carry group A's `'NTT EFOSC2'` titles, and reports the WRONG reason (`key_collision`, whose documented remedy is "find the duplicate row … and delete it or re-attribute it").** Identical outcome when the claimant run was created by an earlier successful cutover pass. | ✗ **FAIL — reproduced** |
| NF-21 premise | `python -c "issubclass(ZoneInfoNotFoundError, KeyError)"` | `True`; `issubclass(..., ValueError)` → `False`. `ZoneInfo(run.site.timezone)` at `allocation_projector.py:199`/`:575` is reached from the per-line body whose handler is `except (ValueError, Observatory.DoesNotExist)` (`load_telescope_runs.py:337`). | ✗ FAIL |
| Migration has no data step | `cat solsys_code/migrations/0018_*.py` | Two `AddField` ops only | ✓ PASS |
| Classical loader has no calendar write path | import scan of `load_telescope_runs.py:1-12` | No `CalendarEvent` / `calendar_utils` import | ✓ PASS |
| Cutover never deletes a `CalendarEvent` | `grep -n "delete" cutover_classical_allocations.py` | No `.delete()` call site | ✓ PASS |
| Committed notebook execution counts | JSON scan of `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` @ HEAD | 17/17 code cells, execution counts 1..17 | ✓ PASS |
| Working-tree notebook execution counts | JSON scan of the same file, working tree | `[None, 2..15, None, 17]` — cells 0 and 15 nulled by an uncommitted hand-edit | ✗ FAIL (uncommitted) |
| Content fingerprint currency | `gsd_run query verification.fingerprint` | `b6a6452…` vs the prior report's `211d217…` — stale, as expected (`observation_projector.py` changed by `24875bf`; the notebook is dirty in-tree) | ℹ️ Recorded |

*The verifier's probe module was written to the session scratchpad and placed on `PYTHONPATH`; no file was created or modified inside the repository (`git status --short` unchanged before and after).*

### Probe Execution

Not applicable — this project defines no `scripts/*/tests/probe-*.sh` probes and no plan declares one. The Django test runner and the ad-hoc reproduction probe above are the equivalent runnable evidence.

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | no probes declared or conventional in this repo | ? SKIP |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| ALLOC-01 | 35-01, 35-02, 35-03 | Per-night events for resolved-site awarded windows; queue/class-wide/satellite keep one container | ✓ SATISFIED | SC-1; `dispatches_per_night()` dispatch; 278 tests green. |
| ALLOC-02 | 35-01 | Nights keyed by site-local observing night, Chile and Australia | ✓ SATISFIED | SC-2; `_time_of_day_to_datetime()` span-nearest rule; boundary tests both hemispheres. |
| ALLOC-03 | 35-01, 35-02, 35-04 | Linked record ⇒ no allocation event; unlink restores; observation's own event untouched | ✓ SATISFIED | SC-3; signal and attribution-bridge tests. |
| ALLOC-04 | 35-03, 35-05, 35-07 | `load_telescope_runs` writes a campaign-less `CampaignRun` with a collision-safe `source_identifier`; same per-night events, idempotent | ⚠️ PARTIAL | SC-4 holds. But the command's per-line skip-and-log contract (its own documented invariant) is broken for a malformed `Observatory.timezone` (NF-21), and its paired notebook is stale (NF-24). |
| ALLOC-05 | 35-06, 35-07 | Cutover has explicit stated sequencing that never leaves a duplicate or orphan | ⚠️ PARTIAL | The end-state property holds and is human-confirmed on the real database. The *stated sequencing* is where it fails: the command's own remedy instruction and the runbook passage repeating it drive an operator into the reproduced NF-19 silent merge. |

**Orphaned requirements:** none. REQUIREMENTS.md maps exactly ALLOC-01..05 to Phase 35, all five appear in plan frontmatter, and no plan claims an ID not in REQUIREMENTS.md.

### Prohibitions (judgment tier — human-reviewed 2026-09-15)

The seven judgment-tier prohibitions carried forward from the prior report were reviewed and confirmed by the human operator in 35-UAT.md test 3 (result: pass). They are therefore no longer `unverified-prohibition` items and do not drive `human_needed`.

One is re-flagged for the closure plan's attention rather than re-opened:

| # | Plan | Prohibition (abbreviated) | Verdict | Note |
|---|------|---------------------------|---------|------|
| 7 | 35-06 | Cutover must NOT delete/re-key an event it cannot explain, and must NOT reclassify a `LEGACY` row | Upheld — narrowly | Literally upheld: zero `.delete()` call sites, every unexplained category `continue`s before any write, no `LEGACY` row's `source` is rewritten. But NF-19 shows the command silently rewriting a **different object** — a pre-existing `CampaignRun`'s `run_status` and `observation_details` — which the prohibition's wording does not reach. Worth widening the prohibition to "must not silently mutate a record it did not create" when NF-19 is closed. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/management/commands/cutover_classical_allocations.py` | 344, 387-395, 483-487 | Database-scoped find-or-create guarded only by an in-process dict (NF-19) | 🛑 **Blocker** | Reproduced by the verifier's own executed probe. The re-run the command itself prescribes silently flips an APPROVED classical run's lifecycle status; the next sweep then retitles that run's nights `[CANCELLED] …`. |
| `solsys_code/management/commands/load_telescope_runs.py` | 337 | `except (ValueError, Observatory.DoesNotExist)` no longer covers `ZoneInfoNotFoundError` (NF-21) | 🛑 Blocker | One mistyped `Observatory.timezone` aborts the whole import with a bare traceback, no summary, subsequent lines unprocessed — falsifying the runbook's stated invariant. Introduced by this phase's own fix commit `d57b461`. |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`, `project_observation_calendar_demo.ipynb` | — | CLAUDE.md paired-docs rule breach (NF-24) | 🛑 Blocker | CLAUDE.md directs the verifier to treat a missing or stale paired-doc update as a must-have gap. Fifth logged instance of this rule being missed. |
| `solsys_code/allocation_projector.py` | 679-687 | `if dry_run: continue` on the re-mint branch skips the inversion guard the create branch applies (NF-20) | ⚠️ Warning | A dry run over an operator-inverted sub-night window reports `would_create`/`would_retire`; the immediately following real run raises `ValueError`. |
| `solsys_code/allocation_projector.py` | 655-664 | Takeover branch `continue`s before claiming the legacy url (NF-22) | ⚠️ Warning | One foreign-attributed legacy event is reported as `2 event(s) blocked`, with two log lines. |
| `solsys_code/campaign_reconciler.py` | 678 | `-> tuple[int, int, int]` on a function returning four values (NF-23) | ⚠️ Warning | Signature and docstring state different contracts. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 391-392 | `duplicate_identity` remedy names a schedule file this command never reads (NF-25) | ⚠️ Warning | The instruction is un-actionable for this command; the paired notebook's own committed output shows it printed against lines that already carry a token. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 395 | `seen_keys[key]` claimed before the group is known to be convertible (IN-02) | ℹ️ Info | Operator message points at the wrong line to fix first. |
| `solsys_code/tests/test_cutover_classical_allocations.py` | 527-531 | Comment hard-codes absolute line numbers (IN-01) | ℹ️ Info | Rots silently on the next edit above it. |
| `solsys_code/campaign_views.py` | 750 | Interpolates `result.skipped_nights`, which no code path assigns any more | ℹ️ Info | Cosmetic; pre-disclosed in 35-02-SUMMARY.md as a deliberate non-fix. Carried forward unchanged. |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | cells 0, 15 | Uncommitted working-tree hand-edit nulls two execution counts (T-35-17) | ⚠️ Warning | HEAD is clean; committing the working tree as-is would break 35-07 truth 8. Remedy: `jupyter nbconvert --to notebook --execute --inplace`, never hand-patch. |

**Debt-marker gate: clean.** Every `TBD` across this phase's modified files is domain vocabulary (a run whose dates are To Be Determined). No `FIXME`, `XXX`, `HACK`, or un-referenced `TODO` in any file this phase touched.

**Stub scan: clean.** No stub returns on any write path; every rendered value traces to a real `sun_event()` computation, a model field, or a DB query.

### Advisory (New Scope, Unevidenced)

None. Every blocker above is either backed by the verifier's own executed reproduction (NF-19), by a directly demonstrated language-level fact plus the code path that reaches it (NF-21), or by an explicit CLAUDE.md instruction naming the verifier as the enforcer (NF-24). No finding rests on architectural preference alone.

### Human Verification Required

None outstanding. The three items from the 2026-09-13 pass were tested and passed by the operator today (35-UAT.md, 3/3 pass, 0 issues), and the judgment-tier prohibition review that produced the third item is closed. Status is `gaps_found` on code evidence, not on any pending human judgment.

### Deferred Items

None. `roadmap.analyze` shows no later milestone phase whose goal or success criteria cover the cutover identity-key guard, the loader's exception routing, or the paired notebooks. All three gaps are this phase's own work.

### Gaps Summary

**The prior pass's three blocking human items are genuinely closed, and CR-01..CR-06 are genuinely fixed. The phase still does not pass, for reasons neither the UAT nor the security review was looking at.**

The decisive finding is that **this phase's own committed quality gate is red and was never closed.** `35-REVIEW.md` (iteration 4, commit `8a393cf`, 2026-09-14) carries `status: issues_found` with one BLOCKER and six WARNINGs. `git log` over every commit since shows **zero** source changes addressing them — the only source commit after that review is `24875bf`, the phase-34 `F-34-1` fix to a different module. I re-checked all nine findings against the current code by hand and every one is still present.

I did not take the review's word for the BLOCKER. I wrote a probe module into the session scratchpad, put it on `PYTHONPATH`, and ran it against a real Django test database in this process. It reproduces:

- Pass 1 over two status-only-colliding schedule groups behaves exactly as designed — `duplicate_identity=3`, non-zero exit, the first group's run left at `run_status='planned'`.
- Pass 2 — **the re-run the command's own `CommandError` prescribes, verbatim: "then re-run this command -- it is safe to repeat"** — silently flips that run to `run_status='cancelled'`, rewrites its `observation_details`, and reports the whole thing under `key_collision`, a reason whose documented operator remedy ("find the duplicate row … and delete it or re-attribute it") is wrong for what actually happened.

That is not a cosmetic reporting bug. `allocation_night_title()` derives the calendar title prefix from `run_status`, so **step 4 of the documented four-step cutover sequence — the reconciler sweep — then retitles every one of that run's nights `[CANCELLED] NTT EFOSC2`.** An operator who follows the runbook end to end reaches a calendar that marks allocated nights as cancelled. The runbook meanwhile asserts the opposite as settled fact at L939-943: *"the SECOND group is never merged into the first group's run … the first group's run and events are converted and left untouched either way."* Both halves of that sentence are false on the second invocation.

278 tests pass, and they pass honestly — the hole is that no test in `test_cutover_classical_allocations.py` constructs a `CampaignRun` that already holds the identity key. Every collision test is in-process. That is precisely why a green suite, a passing UAT and a clean security audit all coexisted with a live blocker.

Two further gaps, both self-inflicted by this phase's own fix cycle:

1. **The NF-08 fix (`d57b461`) removed the only handler for `ZoneInfoNotFoundError`.** Its own commit message names the subclass relationship — and then narrows the `except` tuple to `(ValueError, Observatory.DoesNotExist)`, which `ZoneInfoNotFoundError` (a `KeyError`) does not match. `ZoneInfo(run.site.timezone)` is reached from inside the per-line body at `allocation_projector.py:199`/`:575`. One typo in an unvalidated `CharField` — which the runbook itself tells operators to hand-type — now aborts the entire import with a bare traceback. The runbook's "**one bad row never aborts the whole run**" (L1382-1394) is now false for the command it names first.

2. **CLAUDE.md's paired-docs rule was breached twice, and CLAUDE.md names the verifier as the enforcer** ("treat a missing or stale update as a must-have gap, not a nice-to-have"). `d57b461` changed `load_telescope_runs.py`'s stderr surface, transaction boundary and exception routing; `load_telescope_runs_demo.ipynb` was last regenerated at `7d7b9d4`, before it, and its cells print `stderr (skipped lines)` verbatim. `30112a0` reordered `observation_projector.py`'s writes so attribution lands on the creating save — the exact behaviour the D-11 trigger demo exists to show — and `project_observation_calendar_demo.ipynb` has not been touched since phase 34. This is the fifth logged instance of this rule being missed.

Finally, a state note rather than a gap: the content fingerprint was stale for exactly the reason suspected — `observation_projector.py` is in this phase's `covered_files` and `24875bf` modified it — and the working tree carries an uncommitted hand-edit to `reconcile_campaign_runs_demo.ipynb` that nulls the execution counts on two of its seventeen code cells (T-35-17). The committed copy is clean; committing the tree as-is would break 35-07 truth 8. Re-execute with `jupyter nbconvert --to notebook --execute --inplace` rather than patching by hand.

**What passes:** the allocation layer itself. SC-1 through SC-4 are solid, all 278 phase tests are green, every key link is wired with real data flowing, the debt-marker and stub gates are clean, requirement traceability is complete with no orphans, and the real-database cutover numbers are human-confirmed. The defects are concentrated in the cutover command's identity guard, the loader's exception routing, and the documentation and notebooks that describe both.

---

_Verified: 2026-09-15T04:29:09Z_
_Verifier: Claude (gsd-verifier)_
