---
phase: "36"
slug: "unattended-operation"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-16"
---

# Phase 36 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django's built-in test runner (`django.test.TestCase`) via `python manage.py test` — the only functioning suite (CLAUDE.md "Testing") |
| **Config file** | none — `pyproject.toml`'s pytest `testpaths` governs only the legacy `tests/` suite; `solsys_code/tests/` needs no config |
| **Quick run command** | `python manage.py test solsys_code.tests.test_unattended` (new module; narrow to a TestCase/test for a single check) |
| **Full suite command** | `python manage.py test solsys_code` (exclude `solsys_code.tests.test_views.TestEphemeris`, which segfaults in native ASSIST; never import `solsys_code.views`/`ephem_utils` in a probe — ~1.6 GB SPICE download) |
| **Estimated runtime** | ~60 seconds for a single new module; a few minutes for the full `solsys_code` suite |

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_unattended` (or the specific new/changed test module)
- **After every plan wave:** Run `python manage.py test solsys_code` (excluding the known-segfaulting `test_views.TestEphemeris`)
- **Before `/gsd-verify-work`:** Full suite must be green, plus a live `python manage.py check_unattended` run against the developer database (manual UAT — that command only inspects settings/filesystem/DB state)
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 36-XX-XX | — | — | SCHED-08 | — | Runner runs the fixed 4-step sequence, one exit code, step-failure isolation (D-02) | unit | `python manage.py test solsys_code.tests.test_unattended` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | SCHED-08 | — | Two invocations never overlap (per-command `fcntl.flock`, skip-and-log when contended) | unit | `python manage.py test solsys_code.tests.test_unattended.TestLocking` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | SCHED-08 | — | `check_unattended` reports every prerequisite, prints the exact cron line, never prints a credential value | unit | `python manage.py test solsys_code.tests.test_check_unattended` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | SCHED-09 | — | Failed step triggers exactly one email; suppressed on repeat; "cleared" email on recovery (D-11) | unit | `python manage.py test solsys_code.tests.test_unattended.TestNotification` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | SCHED-09 | — | Heartbeat pings `/start` then `/<exit-code>`; ping failure never fails the tick (D-12) | unit | `python manage.py test solsys_code.tests.test_unattended.TestHeartbeat` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | SCHED-10 | T-36-01 | No credential value in any log line / email / stdout across every forced failure path (D-16, D-17) | unit (regression) | `python manage.py test solsys_code.tests.test_unattended.TestCredentialHygiene` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | DISCOVER-01 | — | `WatchedProposal` model + admin (`list_editable` on `is_active`, filter, bookkeeping fields) (D-06) | unit | `python manage.py test solsys_code.tests.test_watched_proposal` | ❌ W0 | ⬜ pending |
| 36-XX-XX | — | — | DISCOVER-01 | — | Bare `backfill_lco_observations` sweeps every active row; per-row isolation; `last_run_summary`/`last_run_at` written (D-07, D-09) | unit (extends existing) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✅ (extend) | ⬜ pending |
| 36-XX-XX | — | — | DISCOVER-01 | — | Empty watched list is a quiet no-op: exit 0, one INFO line (D-08) | unit | `python manage.py test solsys_code.tests.test_backfill_lco_observations.TestEmptyWatchedList` | ❌ W0 (new class, existing file) | ⬜ pending |
| 36-XX-XX | — | — | DISCOVER-01 | — | `--proposal` override still works for an unwatched proposal; the 30 existing tests keep passing | unit (regression) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*The planner replaces the `36-XX-XX` placeholders with real task IDs, plan and wave numbers once PLAN.md files exist.*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_unattended.py` — stubs for SCHED-08, SCHED-09, SCHED-10 (runner orchestration, notification, heartbeat, credential hygiene)
- [ ] `solsys_code/tests/test_check_unattended.py` — stubs for SCHED-08 success criterion 5 (preflight command)
- [ ] `solsys_code/tests/test_watched_proposal.py` — stubs for DISCOVER-01 model/admin surface
- [ ] Extension of `solsys_code/tests/test_backfill_lco_observations.py` — DISCOVER-01 D-07..D-09 (bare-invocation loop, per-proposal isolation, empty-list no-op)
- [ ] Extension of `solsys_code/tests/test_admin.py` — `WatchedProposalAdmin` `list_editable`/`list_filter`
- [ ] No new test framework install needed — `python manage.py test` already covers everything this phase needs

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The installed cron line runs the runner every 15 minutes on the real host and `flock -n` skips a contended tick | SCHED-08 | Needs the real host's crontab, `/var/lock/fomo/`, `/var/log/fomo/` | Follow the runbook's new unattended-operation section: run `check_unattended`, install the printed line, wait one interval, read `/var/log/fomo/unattended.log` |
| The heartbeat service alerts when no `/start` arrives within the grace period | SCHED-09 | Requires the external healthchecks-compatible service and real egress | Configure a check with a 15-minute period / ~20-minute grace; disable the cron line for one interval; confirm the alert fires |
| `check_unattended` against the developer DB prints names and set/unset status only, never values | SCHED-10 | Live settings must not be quoted in any committed artifact | Run `python manage.py check_unattended` locally; inspect the output for variable names only |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
