---
phase: 29-the-reconciler
verified: 2026-08-05T22:30:00Z
status: passed
score: 9/9 must-haves verified (automated); 1 traceability inconsistency noted (WARNING); 1 item human-signed-off via UAT
overrides_applied: 1
overrides:
  - must_have: "RECON-04 marked Complete in REQUIREMENTS.md"
    reason: "RECON-04's narrowing/COMPLETED behavior is implemented by pre-existing Phase 28 code; Phase 29's own scope for RECON-04 (non-interference) is fully tested and verified. REQUIREMENTS.md's checkbox appears to be an unupdated tracking artifact, not a functional gap."
    accepted_by: "Tim Lister"
    accepted_at: "2026-08-06T22:27:12Z"
human_verification:
  - test: "Live-browser visual confirmation of /calendar/ and the Campaign run pop-up block for a reconciler-owned event"
    expected: "A queue run shows one whole-window entry, a classical run shows one entry per night, and clicking a reconciler-owned entry shows a 'Campaign run' block naming the run/window/status with no manual admin linking"
    why_human: "Browser-rendered popup content and visual calendar layout cannot be asserted by grep/test; 29-06-SUMMARY.md itself flags this as attempted-but-only-partially-automatable (no interactive browser in that execution) and explicitly recommends a human do the final sign-off before treating the phase as fully closed"
    signed_off: "2026-08-05T23:10:00Z via 29-UAT.md Test 1 (pass) -- confirmed on real dev-DB data for July 2025: RUN:29 renders as one whole-window entry, RUN:9/RUN:22 render one entry per night; RUN:3 (ESO VLT FORS2) confirmed resolving to MPC 309 (Paranal) with correct whole-window display for its eso_queue source"
---

# Phase 29: The Reconciler Verification Report

**Phase Goal:** Calendar events stop being a side effect of a staff click and become a function of
run state — one idempotent command plus per-run reconciliation on every staff decision, projecting
all four window-pipeline stages, safe to re-run, blind to events it does not own, and retiring the
backfill-command-per-gap pattern for good.

**Verified:** 2026-08-05T22:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (merged from ROADMAP.md Success Criteria + PLAN frontmatter must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Staff can run one command that projects/refreshes calendar events for every run; a second identical sweep changes nothing | VERIFIED | `reconcile_campaign_runs` exists (`solsys_code/management/commands/reconcile_campaign_runs.py`), loops `reconcile_run()` unfiltered. `TestIdempotency`/`TestDryRun` pass. Independently re-ran against the real dev DB: `python manage.py reconcile_campaign_runs --dry-run` → `would_create: 0, would_update: 0, would_leave_unchanged: 64` (steady state, confirming the live sweep already recorded in 29-06-SUMMARY.md converged) |
| 2 | A classical, site-resolved run shows one dip-corrected event/night; a queue-scheduled (or class-wide) run shows a single whole-window `RUN:{pk}` container, coexisting with separate real `ObservationRecord`-derived events; a scheduled night narrows to the record's window and a completed one shows COMPLETED | VERIFIED | `_reconcile_classical_nights()`/`_reconcile_container()` in `campaign_reconciler.py`; `TestClassicalStage1`, `TestQueueStage1`, `TestClassWideStage2`, `TestSatelliteContainer` all pass. Non-interference with `ObservationRecord`-derived events proven by `TestQueueOwnershipDoesNotTouchRecordEvents` and, on real data, `CampaignRun` pk=1's 10 untouched LCO-derived events coexisting with its 15 new per-night events (29-06-SUMMARY.md Step 5). The record-window-narrowing/COMPLETED-marking half of this criterion is implemented by pre-existing `sync_lco_observation_calendar.py`/`calendar_utils.record_time_window()` (confirmed present by direct read), which this phase deliberately does not re-implement (RESEARCH.md Pattern 3) — see traceability note below |
| 3 | Events the reconciler does not own are never created/modified/deleted, proven against a same-window fixture | VERIFIED | `_may_write()` is the first condition in every write path; `TestOwnershipScoping` proves a same-window un-owned event and a different-run-owned event are both left alone; real-data evidence: 19/20 pre-existing dev-DB events had an unchanged `modified` timestamp after the live sweep (the 20th was a deliberate D-02 adopt, not a violation) |
| 4 | `--dry-run` writes nothing and reports exactly what would change; a failing run is reported and skipped, batch continues | VERIFIED | `TestDryRun`/`TestFailureIsolation` pass; command's only catch point is the batch loop (`except Exception` around `reconcile_run()`), confirmed by reading `reconcile_campaign_runs.py` directly |
| 5 | The real 3I/ATLAS runs (19 as originally measured; 26 as of this verification, per Phase 27 site-repair growth) become calendar-visible; four staff actions reconcile immediately; `backfill_range_calendar_events` no longer exists in code or runbook | VERIFIED | Independently queried the real dev DB: all 26 eligible, approved 3I/ATLAS runs own at least one `RUN:`-namespaced event (`missing events: []`). `grep -c 'reconcile_run(run)' campaign_views.py` → 3 call sites serving 4 actions (`approve`, `resolve_site`, `_set_run_status` for both `mark_cancelled`/`mark_weather_failure`). `backfill_range_calendar_events.py` and its test module confirmed absent from disk; zero functional references anywhere (remaining `_project_calendar_event`/`_calendar_event_title` grep hits are prose-only docstrings in `campaign_reconciler.py`, a documented residual) |
| 6 | Mid-phase code-review finding CR-01 (stale-family calendar events on run reclassification) is actually fixed, not just claimed | VERIFIED | Read `campaign_reconciler.py` directly: `_detach_stale_family_events()` exists and is called from `reconcile_run()`; `_adopted_event_for_night()`'s candidate query is scoped to `event__url=''`. `TestReclassificationConvergence` (2 tests) passes independently |
| 7 | WR-01 (deleting a `CampaignRun` orphans its calendar events) and WR-02 (`window_end < window_start` silently vanishes) fixes are real | VERIFIED | `pre_delete` signal `_delete_owned_calendar_events_on_campaign_run_delete` present in `models.py`; `_skip_reason()` has the `'window_end before window_start'` branch. `TestCampaignRunDeletionCascadesCalendarEvents` and `TestWindowEndBeforeWindowStart` pass independently |
| 8 | The user-directed `ESO_QUEUE` deviation (plan 29-06) is correctly wired, not just documented | VERIFIED | `CampaignRun.Source.ESO_QUEUE` present in `models.py`; migration `0014_alter_campaignrun_source.py` adds it to `choices`; `campaign_reconciler.QUEUE_SOURCES` includes it; dedicated regression test present and passing |
| 9 | Paired docs (runbook + demo notebook) actually document the reconciler, not the retired command | VERIFIED | `docs/runbooks/telescope_runs_calendar.rst` has 0 occurrences of the literal retired command name (only a paraphrased "retired" reference) and >10 mentions of `reconcile_campaign_runs`; the "Campaign run block" section's stale "nothing in FOMO fills that link in automatically" claim is gone; a note on the CR-01 detach behavior is present. Demo notebook exists, executed with 6/6 code cells carrying output, wired into `docs/notebooks.rst` toctree and `CLAUDE.md`'s pairing map |

**Score:** 9/9 automated must-haves verified. One item (see Human Verification below) requires a live-browser check the phase's own final plan (29-06) could not perform in its execution environment and explicitly deferred.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_reconciler.py` | `reconcile_run()`, stage branches, key builders, ownership query, CR-01/WR-02 fixes | VERIFIED | All symbols present; read directly; 400+ lines |
| `solsys_code/calendar_utils.py` | `update_calendar_event_key_and_fields()`, `preview_calendar_event_action()` | VERIFIED | Both present, tested (`TestUpdateCalendarEventKeyAndFields`/`TestPreviewCalendarEventAction`) |
| `solsys_code/management/commands/reconcile_campaign_runs.py` | Batch command, `--dry-run`, D-05 summary | VERIFIED | Read directly; matches plan exactly; manually re-ran against dev DB |
| `solsys_code/models.py` | `pre_delete` signal (WR-01), `CampaignRun.Source.ESO_QUEUE` | VERIFIED | Both present |
| `solsys_code/migrations/0014_alter_campaignrun_source.py` | `ESO_QUEUE` choice migration | VERIFIED | Present, applied (confirmed via shell query) |
| `solsys_code/tests/test_campaign_reconciler.py` | All 9+ test classes incl. CR-01/WR-01/WR-02 regressions | VERIFIED | 72 tests pass (combined with `test_reconcile_campaign_runs`/`test_calendar_utils`) |
| `solsys_code/management/commands/backfill_range_calendar_events.py` | Must NOT exist | VERIFIED (absent) | Confirmed deleted from disk |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Paired demo notebook, executed | VERIFIED | 6/6 code cells carry output |
| `docs/runbooks/telescope_runs_calendar.rst` | Reconciler documented, backfill retired | VERIFIED | Confirmed by grep + read |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `campaign_reconciler.py` | `calendar_utils.py` | `update_calendar_event_key_and_fields`/`preview_calendar_event_action` imports | WIRED | Confirmed by import statement and call sites |
| `campaign_reconciler.py` | `telescope_runs.py` | `sun_event` import | WIRED | Confirmed |
| `campaign_reconciler.py` | `models.CalendarEventMeta` | `_link_event_to_run`/`_detach_stale_family_events` | WIRED | Confirmed, both exercised by passing tests |
| `reconcile_campaign_runs.py` | `campaign_reconciler.py` | `from solsys_code.campaign_reconciler import reconcile_run` | WIRED | Confirmed by direct read and successful `--dry-run` execution |
| `campaign_views.py` | `campaign_reconciler.py` | 3 `reconcile_run(run)` call sites (4 staff actions) | WIRED | Confirmed by grep; exercised by 124-test `test_campaign_approval.py` (all passing) |
| `models.py` `pre_delete` signal | `campaign_reconciler.owned_events()` | lazy import inside handler | WIRED | Confirmed by direct read and passing `TestCampaignRunDeletionCascadesCalendarEvents` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Reconciler test suite (unit) | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_calendar_utils` | 72 tests, OK | PASS |
| Approval-queue + admin suite | `python manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_admin` | 172 tests, OK | PASS |
| Full `solsys_code` regression (excl. `test_views`/`test_ephem_utils` per project memory) | `python manage.py test <27 modules>` | 817 tests, OK (100.4s) | PASS |
| CR-01/WR-01/WR-02 regression tests specifically | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestReclassificationConvergence solsys_code.tests.test_campaign_reconciler.TestCampaignRunDeletionCascadesCalendarEvents solsys_code.tests.test_campaign_reconciler.TestWindowEndBeforeWindowStart` | 4 tests, OK | PASS |
| Real dev-DB dry-run steady state | `python manage.py reconcile_campaign_runs --dry-run` | `would_create: 0, would_update: 0, would_leave_unchanged: 64` | PASS (confirms 29-06's live sweep is durable/idempotent, not a one-time fluke) |
| Real dev-DB 3I/ATLAS coverage | Shell query: every approved/resolved 3I/ATLAS run has ≥1 owned event | `eligible runs 26, missing events []` | PASS |
| `ruff check .` | — | 1 pre-existing, unrelated `D103` finding (confirmed pre-existing by prior plans' SUMMARYs) | PASS (no new issues) |
| `ruff format --check .` | — | 3 pre-existing, unrelated reformat candidates (confirmed pre-existing) | PASS (no new issues) |
| Sphinx build | `python -m sphinx -b html docs docs/_build/html -q` | Pre-existing, unrelated warnings only (autoapi, `ESO_How_to_download_data.ipynb`) | PASS (no errors attributable to this phase's files) |

### Requirements Coverage

| Requirement | Source Plan(s) | Status | Evidence |
|-------------|-----------------|--------|----------|
| RECON-01 | 29-01, 29-03 | SATISFIED | Idempotency proven at unit and command level, and re-confirmed live |
| RECON-02 | 29-01, 29-02 | SATISFIED | Both classical and queue halves tested and code-reviewed |
| RECON-03 | 29-01 | SATISFIED | Class-wide container branch tested |
| RECON-04 | 29-02 | **Traceability inconsistency — see below** | Non-interference contract tested (`TestQueueOwnershipDoesNotTouchRecordEvents`); the narrowing/COMPLETED half of the literal requirement text is implemented by pre-existing Phase 28 code (`sync_lco_observation_calendar.py`), by deliberate design (RESEARCH.md Pattern 3), not by this phase |
| RECON-05 | 29-01, 29-02 | SATISFIED | Ownership guard tested at unit level and confirmed on real data |
| RECON-06 | 29-01, 29-03 | SATISFIED | Dry-run parity and failure isolation tested |
| RECON-07 | 29-03, 29-06 | SATISFIED | Fixtured 19-run scenario + live sweep against the real (now 26-row) 3I/ATLAS campaign, independently re-confirmed |
| RECON-08 | 29-04 | SATISFIED | All 4 staff actions call `reconcile_run()`, confirmed by grep and passing test suite |
| RECON-09 | 29-04, 29-05 | SATISFIED | Command and test module deleted; runbook rewritten; confirmed by grep and Sphinx build |

**Orphaned requirements:** none — all 9 RECON IDs in REQUIREMENTS.md's traceability table map to a plan in this phase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/campaign_reconciler.py` | 81, 103, 122, 144, 286 | Prose-only docstring mentions of deleted `_project_calendar_event`/`_calendar_event_title` names | Info | Explicitly documented as a known residual in 29-04-SUMMARY.md; confirmed no code imports/calls/patches these names anywhere — cosmetic only, not a functional gap |
| N/A | — | No `TBD`/`FIXME`/`XXX` unreferenced debt markers found in any file this phase touched | — | Debt-marker gate: clean |

### Human Verification Required

### 1. Live-browser confirmation of `/calendar/` rendering and the Campaign run pop-up block

**Test:** Visit `/calendar/` on the dev server; confirm the 3I/ATLAS queue runs each show as a single whole-window entry and the classical runs show one entry per observing night. Click a reconciler-owned entry and confirm its pop-up shows a "Campaign run" block naming the run, its window and its status, with no manual admin linking needed.
**Expected:** Calendar renders the two key families visibly distinct; the pop-up block appears automatically for reconciler-owned events.
**Why human:** Visual/browser-rendered behavior cannot be asserted by grep or the Django test client alone. 29-06-SUMMARY.md itself states this check "is only partially automatable in this environment (no interactive browser)" and explicitly recommends "a live browser visual confirmation... as the final human sign-off... though not blocking given the automated evidence." The underlying mechanism (`CalendarEventMeta.run` set at creation) is independently confirmed by this verification's own real-data query and by the passing `test_campaign_approval.py` suite, so this is a confirmatory check, not a suspected gap.

## Traceability Note (not a blocker, flagged for a documentation decision)

`REQUIREMENTS.md` line 36 still shows `- [ ] **RECON-04**` (unchecked) and its traceability table
(line 114) marks RECON-04 as **"Pending"**, while every other RECON-* requirement in the same table
is **"Complete"**. Plan `29-02-PLAN.md`'s frontmatter declares `requirements: [RECON-02, RECON-04,
RECON-05]` and `29-02-SUMMARY.md` claims `requirements-completed: [RECON-02, RECON-04, RECON-05]`.

Investigation shows this is very likely an unresolved-checkbox oversight, not a live functional gap:
the phase's own `29-RESEARCH.md` ("Pattern 3"), `29-CONTEXT.md`, and `29-VALIDATION.md` all
document, in advance, that RECON-04's stage-3/4 narrowing-to-`ObservationRecord`/COMPLETED behavior
is **already implemented** by pre-existing code from Phase 28 (`sync_lco_observation_calendar.py`'s
`_build_event_fields()`/`_time_window()`, promoted to `calendar_utils.record_time_window()`), and
that this phase's only job for RECON-04 is to prove the reconciler leaves those events alone — which
`TestQueueOwnershipDoesNotTouchRecordEvents` does, and which real dev-DB evidence in
`29-06-SUMMARY.md` (10 untouched `CampaignRun` pk=1 LCO-derived events) corroborates. Confirmed by
direct read that `record_time_window()`/`_build_event_fields()` genuinely exist and are wired into
`sync_lco_observation_calendar.py`.

This is a **WARNING**, not a BLOCKER: the substance of ROADMAP.md's phase-29 Success Criterion 2 is
observably true in the codebase, but `REQUIREMENTS.md`'s own checkbox/table contradicts the phase's
declared completion. Recommend updating `REQUIREMENTS.md` line 36/114 to checked/"Complete" with a
note pointing at Phase 28's pre-existing implementation, or leaving it deliberately "Pending" with an
explicit annotation explaining why (rather than the current silent, unexplained mismatch) — a human
call, not something this verifier should resolve unilaterally.

**This looks intentional and low-risk.** If accepted as-is, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "RECON-04 marked Complete in REQUIREMENTS.md"
    reason: "RECON-04's narrowing/COMPLETED behavior is implemented by pre-existing Phase 28 code; Phase 29's own scope for RECON-04 (non-interference) is fully tested and verified. REQUIREMENTS.md's checkbox appears to be an unupdated tracking artifact, not a functional gap."
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

## Gaps Summary

No functional gaps found. All 9 RECON requirements are observably satisfied in the codebase by
direct inspection and independent test/command execution (not by trusting SUMMARY.md prose). The
mid-phase code review's CR-01 (critical), WR-01, WR-02 and IN-01 findings were verified fixed by
reading the actual code and re-running their regression tests independently — the fixes are real,
not just claimed. The user-directed `ESO_QUEUE` deviation is fully wired (model, migration,
`QUEUE_SOURCES`, test) and is properly recorded in `29-06-SUMMARY.md` as a deviation. The only open
items are: (1) a live-browser sign-off the phase's own final plan flagged as unautomatable and
explicitly deferred to a human, and (2) a cosmetic `REQUIREMENTS.md` traceability inconsistency for
RECON-04 that does not reflect a functional gap. Status is `human_needed` because of item (1); no
gaps are blocking further work.

---

*Verified: 2026-08-05T22:30:00Z*
*Verifier: Claude (gsd-verifier)*
