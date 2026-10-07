---
phase: "36"
slug: "unattended-operation"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-16"
validated: "2026-09-18"
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
| 36-01-01 | 01 | 1 | SCHED-08 | T-36-13 | Runner orchestration: one aggregate exit code, step-failure isolation, fixed registry order, empty-database tick healthy (D-02) | unit | `python manage.py test solsys_code.tests.test_unattended.TestRunUnattended` | ❌ W0 (created by this task) | ✅ green |
| 36-01-01 | 01 | 1 | SCHED-08 | T-36-13 | Two `run_unattended` invocations never overlap — the runner takes its own `fcntl.flock` on the same file cron's `flock -n` guards, and each step takes a named per-command lock as defence in depth; skip-and-log when contended, never queue. The shipped sweep commands do not take these locks in this phase (36-01-PLAN.md `<decisions_this_plan_records>`) | unit | `python manage.py test solsys_code.tests.test_unattended.TestLocking` | ❌ W0 (created by this task) | ✅ green |
| 36-01-01 | 01 | 1 | SCHED-09 | T-36-11 | Failed step triggers exactly one email; suppressed while the same failing set persists; 24-hour reminder; one "recovered" email (D-11, D-14) | unit | `python manage.py test solsys_code.tests.test_unattended.TestNotification` | ❌ W0 (created by this task) | ✅ green |
| 36-01-01 | 01 | 1 | SCHED-09 | T-36-03 | Heartbeat pings `/start` then `/<exit-code>`; ping failure never fails the tick; unset URL skips pinging (D-12) | unit | `python manage.py test solsys_code.tests.test_unattended.TestHeartbeat` | ❌ W0 (created by this task) | ✅ green |
| 36-01-02 | 01 | 1 | SCHED-08 | T-36-02 | The committed crontab template carries the `flock -n` guard, the log redirect, the skip-visible tail and no credential (D-01, D-15, D-18) | source assertion | `python -c "src=open('deploy/cron/fomo.crontab.example').read();print('*/15 * * * *' in src, '/usr/bin/flock -n' in src, 'lock held' in src)"` | ❌ W0 (created by this task) | ✅ green |
| 36-01-03 | 01 | 1 | SCHED-09 | T-36-01 | The submission notice and the runner share one request-free mail sender; a mail outage never breaks a submission (D-11) | unit (regression) | `python manage.py test solsys_code.tests.test_campaign_submission` | ✅ (extend) | ✅ green |
| 36-02-01 | 02 | 1 | DISCOVER-01 | T-36-07, T-36-08 | `WatchedProposal` model + migration + admin (`list_editable` on `is_active`, filter, read-only bookkeeping, unique + stripped code) (D-06) | unit | `python manage.py test solsys_code.tests.test_watched_proposal solsys_code.tests.test_admin` | ❌ W0 (created by this task) | ✅ green |
| 36-02-01 | 02 | 1 | DISCOVER-01 | — | The committed migration matches the model definition and applies cleanly | migration check | `python manage.py makemigrations --check --dry-run && python manage.py migrate` | ✅ (existing tree) | ✅ green |
| 36-02-02 | 02 | 1 | DISCOVER-01 | — | `sweep_proposal()` extraction is behavior-preserving — the 30 existing tests pass unmodified | unit (regression) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✅ (existing 1018-line file) | ✅ green |
| 36-02-03 | 02 | 1 | DISCOVER-01 | T-36-06, T-36-09 | Bare invocation sweeps every active row in code order; per-row failure isolation; `last_run_at`/`last_run_summary` written; failure summary carries a class name only (D-07, D-09) | unit | `python manage.py test solsys_code.tests.test_backfill_lco_observations.TestWatchedListSweep solsys_code.tests.test_backfill_lco_observations.TestPerProposalIsolation` | ❌ W0 (new classes, existing file) | ✅ green |
| 36-02-03 | 02 | 1 | DISCOVER-01 | — | Empty watched list is a quiet no-op: exit 0, one INFO line, no portal call (D-08) | unit | `python manage.py test solsys_code.tests.test_backfill_lco_observations.TestEmptyWatchedList` | ❌ W0 (new class, existing file) | ✅ green |
| 36-02-03 | 02 | 1 | DISCOVER-01 | — | `--proposal` override still works for a proposal that is not a watched row | unit (regression) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✅ | ✅ green |
| 36-03-01 | 03 | 2 | SCHED-08, SCHED-10 | T-36-01 | Status refresh: fresh `LCOFacility()`/`SOARFacility()` per facility, non-empty failure list = step failure, class name re-derived, portal message never logged (D-03) | unit | `python manage.py test solsys_code.tests.test_unattended.TestStatusRefreshStep` | ❌ W0 (new class, file from 36-01-01) | ✅ green |
| 36-03-02 | 03 | 2 | SCHED-08 | T-36-12 | All four steps run in D-01's fixed order even when an early step fails; projector sweep and discovery each call their module function, never `call_command()` | unit | `python manage.py test solsys_code.tests.test_unattended.TestProjectSweepStep solsys_code.tests.test_unattended.TestDiscoveryStep` | ❌ W0 (new classes, file from 36-01-01) | ✅ green |
| 36-03-02 | 03 | 2 | SCHED-08 | — | Expected data-shape outcomes (`unchanged`, `skipped`, `detach_declined`, `remint_declined`) never make a tick non-zero and never mail (D-10) | unit | `python manage.py test solsys_code.tests.test_unattended.TestRunUnattended` | ❌ W0 (new case, file from 36-01-01) | ✅ green |
| 36-03-03 | 03 | 2 | SCHED-10 | T-36-01, T-36-10, T-36-11 | No credential value in any log line / email body / stdout / stderr across all six forced failure paths (D-16, D-17) | unit (regression) | `python manage.py test solsys_code.tests.test_unattended.TestCredentialHygiene` | ❌ W0 (expanded, file from 36-01-01) | ✅ green |
| 36-04-01 | 04 | 2 | SCHED-08 | T-36-16 | `check_unattended` reports every prerequisite in one run; hard failures exit non-zero; heartbeat-unset and empty-list are warnings; the command writes nothing (D-05, D-08, D-12, D-13) | unit | `python manage.py test solsys_code.tests.test_check_unattended.TestHardChecks solsys_code.tests.test_check_unattended.TestWarningChecks` | ❌ W0 (created by this task) | ✅ green |
| 36-04-02 | 04 | 2 | SCHED-08, SCHED-10 | T-36-04, T-36-15 | The printed cron line carries real resolved paths and matches the committed template's shape; the command never prints a setting value; `--send-test-email` proves the mail layer | unit | `python manage.py test solsys_code.tests.test_check_unattended.TestCronLine solsys_code.tests.test_check_unattended.TestTestEmail solsys_code.tests.test_check_unattended.TestNoValueLeakage` | ❌ W0 (created by 36-04-01) | ✅ green |
| 36-04-03 | 04 | 2 | SCHED-08 | — | The committed logrotate example rotates the one log file daily, keeps 14, and uses `copytruncate` (D-18) | source assertion | `python -c "src=open('deploy/logrotate/fomo.example').read();print('daily' in src, 'rotate 14' in src, 'copytruncate' in src)"` | ❌ W0 (created by this task) | ✅ green |
| 36-05-01 | 05 | 3 | SCHED-08, SCHED-09, SCHED-10 | T-36-04 | The runbook's unattended-operation section builds cleanly and covers setup, the two failure signals, and the "nothing has appeared" checklist (SC 5) | docs build + source assertion | `pre-commit run sphinx-build --all-files` | ✅ (extend) | ✅ green |
| 36-05-02 | 05 | 3 | DISCOVER-01 | T-36-17, T-36-18 | The demo notebook shows the watched-proposal contract with committed executed output, makes no live network call, and leaves no rows behind | notebook execution + source assertion | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | ✅ (extend) | ✅ green |
| 36-05-03 | 05 | 3 | SCHED-08 | T-36-19 | `CLAUDE.md`'s paired-docs map records every module this phase adds and why the runner's paired doc is a runbook section | source assertion | `python -c "src=open('CLAUDE.md').read();i=src.index('Paired docs are part of the deliverable');j=src.index('## Project', i);print('unattended.py' in src[i:j], 'notifications.py' in src[i:j])"` | ✅ (extend) | ✅ green |
| 36-06-01..03 | 06 | 4 (gap G-36-3) | SCHED-08, SCHED-09 | T-36-03 | Heartbeat Period/Grace guidance corrected in the runbook and propagated to the crontab backstop comment, the runner docstrings and the `[ok] heartbeat` preflight line; the runbook still builds clean | unit + docs build | `python manage.py test solsys_code.tests.test_check_unattended.TestWarningChecks && pre-commit run sphinx-build --all-files` | ✅ (extend) | ✅ green |
| 36-07-01..03 | 07 | 5 (gap G-36-1) | SCHED-08, SCHED-09 | T-36-04 | The fresh-host setup subsection is self-sufficient: ≥9 numbered steps, Period/Grace values, ping-URL placeholder, `*/15` schedule, a cross-reference to "The two failure signals", and the Period line precedes the ping-URL line which precedes the `4. Export` step | source assertion (slice-scoped) | `python3 -c "import re;src=open('docs/runbooks/telescope_runs_calendar.rst').read();s=src[src.index('Setting it up on a fresh host'):src.index('Adding a proposal to watch')];print(all(t in s for t in ['Period','Grace','*/15 * * * *','hc-ping.com/<uuid>','healthchecks.io','The two failure signals']), len(re.findall(r'^\d+\. ',s,re.M))>=9, s.index('Period')<s.index('hc-ping.com/<uuid>')<s.index('4. Export'))"` | ✅ (extend) | ✅ green |
| 36-08-01..03 | 08 | 6 (gap G-36-4) | SCHED-08, SCHED-10 | T-36-15 | The flat `LCO_API_KEY` local-settings knob folds into both `FACILITIES['LCO']`/`['SOAR']` `api_key` entries, an absent key is a clean no-op, the fold tail leaks no loop variable; the runbook's step 2 names the flat setting and no `FACILITIES[...]` subscript | unit (executes the real settings fold tail) + source assertion | `python manage.py test solsys_code.tests.test_settings_api_key_fold && python3 -c "import re;src=open('docs/runbooks/telescope_runs_calendar.rst').read();s=src[src.index('Setting it up on a fresh host'):src.index('Adding a proposal to watch')];print('LCO_API_KEY' in s, re.search(r'FACILITIES\[',s) is None)"` | ❌ W0 (created by 36-08-01) | ✅ green |
| 36-09-01..03 | 09 | 7 (gap G-36-5) | SCHED-08, SCHED-10 | T-36-16 | `check_unattended` writes each result line exactly once, to one stream chosen by status, flushing stdout before each stderr write; no escape bytes reach a redirected sink; the runbook's step 6 states the routing and the `2>&1` capture form | unit (single-sink `_run_merged()` fixture) + source assertion | `python manage.py test solsys_code.tests.test_check_unattended.TestResultStreamRouting && python3 -c "src=open('docs/runbooks/telescope_runs_calendar.rst').read();s=src[src.index('Setting it up on a fresh host'):src.index('Adding a proposal to watch')];print(all(t in s for t in ['standard output','standard error','2>&1','check_unattended']))"` | ✅ (extend) | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task ID format is `{phase}-{plan}-{task}`. Every row's automated command is also the `<automated>` verify of the named task, so a row going green and its task's verify passing are the same event.*

---

## Wave 0 Requirements

There is no separate Wave 0 plan: every missing test file is created by the same task that needs
it, as the first thing that task does (each of those tasks carries `tdd="true"` and a `<behavior>`
block listing the tests to write before the implementation). The owning task is named against each
item below.

- [x] `solsys_code/tests/test_unattended.py` — created by **36-01-01** (runner orchestration, notification, heartbeat, locking, first credential-hygiene case); extended by **36-03-01**, **36-03-02**, **36-03-03**
- [x] `solsys_code/tests/test_check_unattended.py` — created by **36-04-01**; extended by **36-04-02**
- [x] `solsys_code/tests/test_watched_proposal.py` — created by **36-02-01**
- [x] Extension of `solsys_code/tests/test_backfill_lco_observations.py` — **36-02-02** (`TestSweepProposalFunction`) and **36-02-03** (`TestWatchedListSweep`, `TestPerProposalIsolation`, `TestEmptyWatchedList`); the 30 pre-existing tests must pass unmodified throughout
- [x] Extension of `solsys_code/tests/test_admin.py` — **36-02-01** (`WatchedProposalAdmin` `list_editable` / `list_filter` / changelist render)
- [x] Extension of `solsys_code/tests/test_campaign_submission.py` — **36-01-03** (the shared mail helper's regression cases)
- [x] `solsys_code/tests/test_settings_api_key_fold.py` — created by **36-08-01** (executes the real `settings.py` fold tail against an injected `local_settings` module)
- [x] Extension of `solsys_code/tests/test_check_unattended.py` — **36-09-02** (`TestResultStreamRouting` and the single-sink `_run_merged()` fixture; seven presence tests re-pointed at the merged capture)
- [x] No new test framework install needed — `python manage.py test` already covers everything this phase needs

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The installed cron line runs the runner every 15 minutes on the real host and `flock -n` skips a contended tick | SCHED-08 | Needs the real host's crontab, `/var/lock/fomo/`, `/var/log/fomo/` | Follow the runbook's new unattended-operation section: run `check_unattended`, install the printed line, wait one interval, read `/var/log/fomo/unattended.log` |
| The heartbeat service alerts when no `/start` arrives within the grace period | SCHED-09 | Requires the external healthchecks-compatible service and real egress | Configure a check with a 15-minute period / ~20-minute grace; disable the cron line for one interval; confirm the alert fires |
| `check_unattended` against the developer DB prints names and set/unset status only, never values | SCHED-10 | Live settings must not be quoted in any committed artifact | Run `python manage.py check_unattended` locally; inspect the output for variable names only |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved — validate-phase audit 2026-09-17 (post-execution, all 5 plans summarized)

---

## Validation Audit 2026-09-17

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Evidence: all 23 automated rows' named test classes exist on disk and the full `solsys_code` +
`solsys_code_observatory` suite (1418 tests, excluding the known-segfaulting `test_views.TestEphemeris`,
plus the 40 safe `test_views` cases) ran green after each of the three execution waves; the crontab,
logrotate and `CLAUDE.md` source assertions print `True`; `makemigrations --check --dry-run` reports no
changes; `pre-commit run sphinx-build --all-files` passes; `backfill_lco_observations_demo.ipynb` was
re-executed in place by plan 36-05 with committed output. The three Manual-Only rows remain manual by
design (real-host crontab, external heartbeat service, live-settings inspection) and are carried into
`/gsd-verify-work 36`.

---

## Validation Audit 2026-09-18

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Re-audit after the four gap-closure plans (36-06 `G-36-3`, 36-07 `G-36-1`, 36-08 `G-36-4`, 36-09 `G-36-5`)
and the 36-REVIEW-FIX round landed on top of the 2026-09-17 map. The map gains one row per gap-closure
plan; none was a coverage gap — each plan already carried a committed test or a reproducible source
assertion, the rows only record them so the map stays the single place the phase's verification lives.
Evidence at HEAD `7d1d1ea`: the original 22 test classes named in the map all still exist on disk;
`test_check_unattended` + `test_unattended` + `test_watched_proposal` ran green (122 tests) and
`test_settings_api_key_fold` ran green (9 tests); the 36-07 slice gate, the 36-08 flat-setting gate and
the 36-09 stream-routing prose gate all print `True`. One note on the 36-07 row: its original task-time
gate also probed for the literal `35 min` inside the fresh-host slice; review fix IN-25 (`4ba6e3a`)
deliberately moved that alert-window arithmetic out of step 3 into "The two failure signals" (now
line ~1726, outside the slice), so the committed row asserts the cross-reference to that section
instead of the moved literal. The three Manual-Only rows were discharged by `/gsd-verify-work 36`
rounds 1–4 (36-UAT.md: 32 human checks across four rounds, one issue → G-36-5, closed by 36-09).
