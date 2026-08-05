---
phase: 29
slug: the-reconciler
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| RECON-01 | Command re-run against unchanged state is a no-op (no new rows, no `modified` churn) | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestIdempotency` | ❌ W0 |
| RECON-02 | Classical run → one dip-corrected event/night; queue run → one bare `RUN:{pk}` container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassicalStage1 solsys_code.tests.test_campaign_reconciler.TestQueueStage1` | ❌ W0 |
| RECON-03 | Class-wide run → single whole-window `RUN:{pk}` container | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestClassWideStage2` | ❌ W0 |
| RECON-04 | A run's confirmed `ObservationRecord` link narrows/completes correctly — reconciler leaves these events alone (Pattern 3: it never writes them) | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestQueueOwnershipDoesNotTouchRecordEvents` | ❌ W0 |
| RECON-05 | Reconciler never creates/modifies/deletes an event it does not own, proven against a same-window un-owned fixture | unit | `python manage.py test solsys_code.tests.test_campaign_reconciler.TestOwnershipScoping` | ❌ W0 |
| RECON-06 | `--dry-run` writes nothing and reports exactly what would change; a failing run is skipped-with-reason and the batch continues | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestDryRun solsys_code.tests.test_reconcile_campaign_runs.TestFailureIsolation` | ❌ W0 |
| RECON-07 | The 19 approved, site-resolved 3I/ATLAS runs (fixtured shape, not live pks — see Wave 0 note) become calendar-visible | integration | `python manage.py test solsys_code.tests.test_reconcile_campaign_runs.TestRealDataShapeScenario` | ❌ W0 |
| RECON-08 | approve/resolve_site/mark_cancelled/mark_weather_failure each call the reconciler and reconcile their run immediately | integration, extending existing `test_campaign_approval.py` classes | `python manage.py test solsys_code.tests.test_campaign_approval` | ✅ existing file, needs rewriting (RUN: keys + new patch targets) |
| RECON-09 | `backfill_range_calendar_events` no longer exists in code or the operator runbook | file-deletion / grep-based check (self-verifying, no dedicated test) | n/a | n/a |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_reconciler.py` — unit tests for `reconcile_run()` itself,
      isolated from the Django view/command layer: stage-1 classical per-night math (dip-corrected
      `sun_event(kind='sun')`), stage-1 queue bare-container math, stage-2 class-wide container,
      D-02's adopt-and-rekey step (an existing `CalendarEventMeta.run`-linked blank-`url` classical
      event gets its `url` re-keyed to `RUN:{pk}:{date}` in place, not duplicated), the
      `RUN:{pk}[:date]` ownership-scoping query (`Q(url=...) | Q(url__startswith=...)` analogue of
      `campaign_views.py:876-878`), and RECON-05's same-window un-owned-event fixture.
- [ ] `solsys_code/tests/test_reconcile_campaign_runs.py` — command-level tests: RECON-01
      idempotency (second run produces `created=0, updated=0`), RECON-06 `--dry-run` (writes
      nothing, reports the `import_campaign_csv`-style created/updated/unchanged/skipped-with-reason
      summary), a per-run failure (synthetic ground-type `Observatory` with `timezone=''`, mirroring
      the existing `T99` pattern — NOT expected to reproduce against real data per research Pitfall
      2) isolated so the batch continues past it, and RECON-07's fixtured 19-run real-data-shape
      scenario (8 queue-sourced + 11 classical-sourced + 0 space, matching `26-DECISION.md`'s
      measured split — built as an equivalent fixture with `source` set explicitly on the fixture
      rows, not asserted against live pks).
- [ ] `solsys_code/tests/test_campaign_approval.py` — **rewrite, not just extend.** Research
      Pitfall 3: `TestCalendarProjection`, `TestSitesNeedingReview`'s calendar-projection
      assertions, `TestRunStatusChange`, and `TestCalendarNoChurn` currently assert
      `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}')` and
      `patch('solsys_code.campaign_views.sun_event' / '_project_calendar_event' /
      'insert_or_create_calendar_event', ...)` — every one of these breaks under D-01 (the
      functions they patch are deleted, the key scheme they assert on is retired). Rewrite to
      assert `RUN:` keys and patch the reconciler module's own import names. This is planned,
      first-order rewrite work, not incidental breakage to discover during verification.
- [ ] `solsys_code/tests/test_backfill_range_calendar_events.py` — DELETE the whole file (RECON-09;
      124 lines, no longer has a command to test).
- [ ] No new fixtures/conftest needed beyond what `test_campaign_approval.py` already establishes
      (`Observatory.objects.create(...)`, plain `CampaignRun.objects.create(...)`). **If a RECON-04
      test needs an actual `tom_targets.Target` instance** (e.g. for `CampaignRunObservation`/
      `ObservationRecord` fixtures), use `tom_targets.tests.factories.NonSiderealTargetFactory` —
      never `SiderealTargetFactory` — per CLAUDE.md.
- [ ] No framework install needed — Django test runner already configured; no new dependency
      permitted (locked by REQUIREMENTS.md's Out of Scope table).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| The `checkpoint:human-verify` `source` data-fix (CONTEXT.md D-07) actually leaves the real dev DB in a state where the reconciler renders the 8-queue/11-classical split correctly | RECON-07 | Depends on staff editing real `CampaignRun.source` values via the Django admin — not something an automated test can assert against production/dev data, only against a fixture (already covered above) | After staff complete the D-07 data-fix task, run `python manage.py reconcile_campaign_runs --dry-run` against the real dev DB and visually confirm the reported created/updated counts match the expected 19-run split before running it for real |
| `backfill_range_calendar_events` is genuinely gone from the operator's mental model, not just the codebase | RECON-09 | The runbook prose (not just the command file) needs a human read-through to confirm no stale reference remains — grep-based checks catch literal filenames, not surrounding prose that assumes the command still exists | Read `docs/runbooks/telescope_runs_calendar.rst` end-to-end after the edits and confirm no remaining prose references `backfill_range_calendar_events` as a still-available command |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s for the quick command
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
