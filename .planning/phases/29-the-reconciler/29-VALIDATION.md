---
phase: 29
slug: the-reconciler
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-04
---

# Phase 29 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`) — this phase touches `solsys_code/` app code exclusively, not the pytest-only `tests/`/`src/`/`docs/` suite |
| **Config file** | none dedicated — `pyproject.toml` `testpaths` deliberately excludes `solsys_code/` (CLAUDE.md testing split) |
| **Quick run command** | `python manage.py test solsys_code.tests.test_campaign_reconciler` |
| **Full suite command** | `python manage.py test solsys_code` (exclude `test_views.TestEphemeris` — segfaults in native ASSIST, unrelated to this phase) |
| **Estimated runtime** | ~10 s quick / ~2–4 min full |

**Invocation caveat:** use `python manage.py`, never `./manage.py`.

**Module-import caveat:** `solsys_code/campaign_reconciler.py` must NOT import `solsys_code.views`
or `solsys_code.ephem_utils` — the latter triggers the ~1.6 GB SPICE kernel download at module
load (milestone-locked constraint). This phase's own new tests should never need the exclusion
above, since `campaign_reconciler.py` stays import-clean of that module.

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_campaign_reconciler` plus the specific new test file(s) that task touched
- **After every plan wave:** Run `python manage.py test solsys_code` (excluding `TestEphemeris`)
- **Before `/gsd-verify-work`:** Full suite green, plus `ruff check .` and `ruff format --check .` clean
- **Max feedback latency:** ~10 seconds for the quick command

---

## Per-Task Verification Map

Populated by the planner — each plan task must map to a row here. Requirement→test
coverage is fixed by the table below; task IDs are assigned during planning.

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| RECON-01 | Command re-run against unchanged state is a no-op (no new rows, no `modified` churn) | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestIdempotency` | ✅ exists | ✅ green |
| RECON-02 | Classical run → one dip-corrected event/night; queue run → one bare `RUN:{pk}` container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassicalStage1 solsys_code.tests.test_campaign_reconciler.TestQueueStage1` | ✅ exists | ✅ green |
| RECON-03 | Class-wide run → single whole-window `RUN:{pk}` container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassWideStage2` | ✅ exists | ✅ green |
| RECON-04 | A run's confirmed `ObservationRecord` link narrows/completes correctly — reconciler leaves these events alone (Pattern 3: it never writes them) | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestQueueOwnershipDoesNotTouchRecordEvents` | ✅ exists | ✅ green (override accepted — see below) |
| RECON-05 | Reconciler never creates/modifies/deletes an event it does not own, proven against a same-window un-owned fixture | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestOwnershipScoping` | ✅ exists | ✅ green |
| RECON-06 | `--dry-run` writes nothing and reports exactly what would change; a failing run is skipped-with-reason and the batch continues | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestDryRun solsys_code.tests.test_reconcile_campaign_runs.TestFailureIsolation` | ✅ exists | ✅ green |
| RECON-07 | The real 3I/ATLAS runs (26 as of this verification, growing from an initially-measured 19) become calendar-visible | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestRealDataShapeScenario` | ✅ exists | ✅ green |
| RECON-08 | approve/resolve_site/mark_cancelled/mark_weather_failure each call the reconciler and reconcile their run immediately | integration, extending existing `test_campaign_approval.py` classes | `python manage.py test solsys_code.tests.test_campaign_approval` | ✅ exists, rewritten for `RUN:` keys | ✅ green |
| RECON-09 | `backfill_range_calendar_events` no longer exists in code or the operator runbook | file-deletion / grep-based check (self-verifying, no dedicated test) | `ls solsys_code/management/commands/backfill_range_calendar_events.py` (must fail) | ✅ confirmed absent | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Source:** `29-VERIFICATION.md` (2026-08-05, 9/9 automated must-haves, `status: human_needed`
for one live-browser item only). Full targeted suite `python manage.py test
solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs
solsys_code.tests.test_calendar_utils` → 72 tests OK; full `solsys_code` regression (27
modules, excluding `test_views`/`test_ephem_utils` per project memory) → 817 tests OK.
Independently re-ran `python manage.py reconcile_campaign_runs --dry-run` against the real dev
DB during verification: steady state (`would_create: 0, would_update: 0,
would_leave_unchanged: 64`), confirming the live sweep was durable, not a one-time fluke.

**RECON-04 override:** `29-VERIFICATION.md`'s frontmatter records an accepted override —
`REQUIREMENTS.md`'s RECON-04 checkbox was an unupdated tracking artifact, not a functional
gap (RECON-04's narrowing/COMPLETED behavior is pre-existing Phase 28 code by deliberate
design; Phase 29's own scope, non-interference, is fully tested). Accepted by Tim Lister
2026-08-06T22:27:12Z, and `REQUIREMENTS.md` was corrected to `[x]` + annotation by quick task
`260806-lgo` the same day.

---

- [x] `solsys_code/tests/test_campaign_reconciler.py` — present with all required test classes
      (`TestClassicalStage1`, `TestQueueStage1`, `TestClassWideStage2`, `TestOwnershipScoping`,
      `TestReclassificationConvergence`, `TestCampaignRunDeletionCascadesCalendarEvents`,
      `TestWindowEndBeforeWindowStart`, plus the D-02 adopt-and-rekey and ownership-scoping
      coverage), 72 tests combined with the two sibling modules below.
- [x] `solsys_code/tests/test_reconcile_campaign_runs.py` — present: `TestIdempotency`,
      `TestDryRun`, `TestFailureIsolation`, `TestRealDataShapeScenario`.
- [x] `solsys_code/tests/test_campaign_approval.py` — rewritten (not merely extended) to assert
      `RUN:` keys and patch the reconciler module's own import names; part of the 172-test
      approval-queue+admin regression sweep, all passing.
- [x] `solsys_code/tests/test_backfill_range_calendar_events.py` — deleted; the command file
      itself confirmed absent from disk.
- [x] `NonSiderealTargetFactory` used throughout (never `SiderealTargetFactory`), consistent
      with CLAUDE.md.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Status |
|----------|-------------|------------|--------------------|--------|
| Live-browser confirmation of `/calendar/` rendering and the Campaign-run pop-up block | (phase goal, all RECON-*) | Visual/browser-rendered content cannot be asserted by grep or the Django test client alone | Visit `/calendar/`, confirm queue runs show one whole-window entry and classical runs show one entry per night; click a reconciler-owned entry, confirm the pop-up names run/window/status with no manual admin linking | **Signed off** 2026-08-05T23:10:00Z via `29-UAT.md` Test 1 (pass) — confirmed on real dev-DB data: `RUN:29` renders as one whole-window entry, `RUN:9`/`RUN:22` render one entry per night, `RUN:3` (ESO VLT FORS2) confirms resolving to MPC 309 (Paranal) with correct whole-window display |
| The `checkpoint:human-verify` `source` data-fix (CONTEXT.md D-07) actually leaves the real dev DB in a state where the reconciler renders the correct queue/classical split | RECON-07 | Depends on staff editing real `CampaignRun.source` values via the Django admin — not something an automated test can assert against production/dev data | `python manage.py reconcile_campaign_runs --dry-run` against the real dev DB, visually confirm the reported counts | **Completed** — independently re-run during verification: `would_create: 0, would_update: 0, would_leave_unchanged: 64` (steady state) |
| `backfill_range_calendar_events` is genuinely gone from the operator's mental model, not just the codebase | RECON-09 | The runbook prose (not just the command file) needs a human read-through to confirm no stale reference remains | Read `docs/runbooks/telescope_runs_calendar.rst` end-to-end and confirm no remaining prose references the retired command | **Completed** — `29-VERIFICATION.md` confirms 0 literal occurrences of the retired command name, only a paraphrased "retired" reference |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s for the quick command
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated (Phase 30 plan 30-04, D-08 reconciliation)

## Validation Audit 2026-08-31

Reconciled by Phase 30 plan 30-04 (D-08). This file was seeded by plan-phase with 5 Wave 0
gaps (❌) before execution. Cross-referenced against `29-VERIFICATION.md` (2026-08-05, 9/9
automated must-haves, `status: human_needed` for one live-browser item only, since resolved):
all 5 Wave 0 test modules were created/rewritten during execution and independently re-run by
the verifier (72-test targeted suite, 817-test full regression sweep). The one deferred
live-browser check was signed off by the user via `29-UAT.md` Test 1 on 2026-08-05, and the
RECON-04 traceability override was accepted by Tim Lister on 2026-08-06 and corrected in
`REQUIREMENTS.md` by quick task `260806-lgo` the same day. No new test was written by this
reconciliation.

| Metric | Count |
|--------|-------|
| Gaps found | 5 (all pre-existing Wave 0 markers, already closed by execution) |
| Resolved | 5 (confirmed closed via 29-VERIFICATION.md cross-reference, no new work needed) |
| Escalated | 0 |
