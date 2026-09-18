---
phase: 36-unattended-operation
verified: 2026-09-18T02:05:00Z
status: human_needed
score: 54/56 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/debug/heartbeat-runbook-period-gap.md"
  - ".planning/phases/36-unattended-operation/36-01-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-01-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-02-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-02-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-03-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-03-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-04-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-04-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-05-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-05-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-06-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-06-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-UAT.md"
  - "CLAUDE.md"
  - "deploy/cron/fomo.crontab.example"
  - "deploy/logrotate/fomo.example"
  - "docs/installation.rst"
  - "docs/notebooks.rst"
  - "docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/admin.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/backfill_lco_observations.py"
  - "solsys_code/management/commands/check_unattended.py"
  - "solsys_code/management/commands/run_unattended.py"
  - "solsys_code/migrations/0022_watchedproposal.py"
  - "solsys_code/models.py"
  - "solsys_code/notifications.py"
  - "solsys_code/tests/test_admin.py"
  - "solsys_code/tests/test_backfill_lco_observations.py"
  - "solsys_code/tests/test_campaign_submission.py"
  - "solsys_code/tests/test_check_unattended.py"
  - "solsys_code/tests/test_unattended.py"
  - "solsys_code/tests/test_watched_proposal.py"
  - "solsys_code/unattended.py"
  - "src/fomo/settings.py"
covered_digest: "v1:sha256:596925e0aaa089a610d02354196d5cfb5c559c7d03a621989b6353df124458c4"
behavior_unverified: 1
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 48/48
  previous_verified: 2026-09-17T17:20:00Z
  gaps_closed:
    - "G-36-3 (UAT Test 3): the runbook's heartbeat guidance named only the check's grace time, never its expected ping interval, so a check configured from it first alerted about a day after the schedule stopped. The runbook (4 sites + a new troubleshooting entry), deploy/cron/fomo.crontab.example, two unattended.py docstrings, check_unattended's [ok] heartbeat line and this verification record now all state the two-knob configuration and the last ping + expected interval + grace arithmetic."
  gaps_remaining: []
  regressions: []
  human_items_closed_by_uat:
    - "Test 1 fresh-host preflight — UAT pass (all hard checks [ok], exit 0, cron line printed)"
    - "Test 2 the real crontab — UAT pass (three START/END banners ~15 min apart on the real host)"
    - "Test 4 real mail delivery — UAT pass"
    - "Test 5 logrotate under a live writer — UAT pass (START went with the rotated copy; END landed in the truncated live file)"
    - "Test 6 runbook sufficiency — UAT pass (user verdict)"
  human_items_still_open:
    - "Test 3 re-run: configure a live healthchecks-compatible check from the CORRECTED runbook paragraph alone and confirm late ~15 min / alert ~35 min"
advisory:
  - finding: "WR-22 (36-REVIEW.md iter 3): two logger.debug() sites on the unattended path interpolate a raw exception message rather than its class name — backfill_lco_observations.py:349 (a live, authenticated LCO portal call, reached from step_discovery -> sweep_watched_rows -> sweep_proposal -> _resolve_schedule) and unattended.py:191 (a bare `except Exception` around FOMO's own reconcile_run()). Inert under the shipped configuration because settings.LOGGING pins the root logger to INFO, so no such line is emitted; live the moment anyone raises the level to chase a problem."
    category: security
    reason: "Falsifies plan 36-01 must-have truth 10 as literally written ('no unattended-path log write formats the message of an exception caught from a requests call, a facility/portal call, or send_mail() -- only its class name'). Does NOT falsify SC 4, which is about what appears in a log line the path actually produces: with root at INFO nothing is produced, and the 8 TestCredentialHygiene tests remain green. Resolved by using type(exc).__name__ at both sites and extending TestCredentialHygiene with an assertLogs(level='DEBUG') case."
    evidence_status: "source-confirmed (both lines read directly); no demonstrated leak under the shipped log configuration"
  - finding: "IN-20 (36-REVIEW.md iter 3): check_unattended's [ok] heartbeat detail names only the expected ping interval (Period), not the ~20-minute grace time, and the new test pins only 'Period'. An operator acting on the preflight line alone sets Period=15 and leaves healthchecks.io's 1-hour default grace — a 75-minute window instead of the documented ~35."
    category: other
    reason: "Plan 36-06 truth 6 asks only for an expected-ping-interval reminder, which is present, so this is not a must-have gap. Worth extending the detail string and the test assertion to cover the grace half."
    evidence_status: "source-confirmed; not a must-have breach"
  - finding: "36-REVIEW.md iter 3 carries five further warnings (WR-16 cron exit-99 attribution, WR-17 runtime state-dir mail storm, WR-18 IN-02's skip reasons logged at a dropped DEBUG level, WR-19 check_email accepts dummy/locmem/filebased backends, WR-20 check_flock probe can hang, WR-21 runbook omits two newly-added hard checks)."
    category: other
    reason: "None falsifies a success criterion or a plan must-have truth; recorded here so the ship decision sees them."
    evidence_status: "review-reported; not independently re-tested by this verification"
behavior_unverified_items:
  - truth: "An operator who configures the heartbeat check using only what the corrected runbook names gets a check that goes late about 15 minutes after a missed tick and alerts about 35 minutes after the last successful ping (plan 36-06 truth 1 — G-36-3's failed truth, restored)."
    test: "Create a fresh healthchecks-compatible check pointed at FOMO_HEARTBEAT_URL, configuring ONLY what the corrected runbook paragraph (docs/runbooks/telescope_runs_calendar.rst:1541-1568) names: expected ping interval (Period) 15 minutes — or a Cron-type check with */15 * * * * — and grace time (Grace) about 20 minutes. Then disable the crontab line, simulating the scheduler never invoking the job."
    expected: "The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping, while FOMO itself logs nothing and sends no email — SC 3's second, independent layer."
    why_human: "The signal is produced by the external heartbeat service's own expected-interval-plus-grace timer, not by any FOMO code path. G-36-3 was only findable this way: every internal gate checked the runbook against decision D-12, which itself carried the conflation."
human_verification:
  - test: "Re-run UAT Test 3 against a live healthchecks-compatible check configured ONLY from the corrected runbook paragraph (expected interval / Period 15 min, or Cron type */15 * * * *; grace / Grace ~20 min), then disable the crontab line."
    expected: "Check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping, with FOMO logging nothing and mailing nothing."
    why_human: "External service behaviour; no automated gate can reach it. This is the only proof that closes G-36-3 end-to-end."
  - test: "Decide on WR-22 before shipping: either fix the two logger.debug() sites (backfill_lco_observations.py:349, unattended.py:191) to log type(exc).__name__, or record an explicit acceptance that the class-name-only discipline holds only while settings.LOGGING keeps the root logger at INFO."
    expected: "Either a two-line fix plus an assertLogs(level='DEBUG') credential-hygiene case, or a recorded acceptance with the constraint documented where an operator raising the log level would see it."
    why_human: "A judgment call on a latent-but-real SCHED-10 exposure: the phase's own must-have truth 10 is written absolutely, while SC 4 as written is not breached under the shipped configuration. Which reading governs the ship decision is the developer's to make."
---

# Phase 36: Unattended Operation Verification Report

**Phase Goal:** The projector sweep, the LCO/SOAR discovery backfill and the reconciler run on the real host on a documented schedule with nobody typing anything, against a watched-proposal list an operator edits in the admin — and when it breaks, an operator finds out.
**Verified:** 2026-09-18T02:05:00Z
**Status:** human_needed
**Re-verification:** Yes — rebuilt after UAT gap G-36-3 and plan 36-06's gap-closure commits (`12c51c6`, `f075a7f`, `1f3bbac`, `561f9ea`). All six plans' must-haves re-established from frontmatter; 36-06's truths, prohibitions and artifacts verified line-by-line against the current files; 36-01…36-05's truths regression-checked (all phase test modules re-run green, runner re-executed live).

---

## Goal Achievement

### Observable Truths

#### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Projector sweep, LCO/SOAR discovery backfill and reconciler all run on their documented recurring schedule with no operator action, and two invocations of the same job never overlap | ✓ VERIFIED | Re-executed live on the current code: `FOMO_LOCK_DIR=… FOMO_STATE_DIR=… python manage.py run_unattended --dry-run` → `START 2026-09-18T01:46:33` / `step status_refresh: ok` / `step project_sweep: ok … unchanged: 159` / `step discovery: ok` / `step reconcile: ok \| runs: 45, failed: 0` / `END … exit=0`. Overlap re-proven **cross-process** with the same `/usr/bin/flock -n` binary the crontab line invokes: a second real invocation printed `run_unattended: lock held -- skipping this tick`, ran no step, exited 0. The "on the real host" half is no longer an open item — **UAT Test 2 passed** (three START/END banners ~15 min apart in `/var/log/fomo/unattended.log` from a real crontab) |
| SC2 | Adding a proposal in the admin is enough for its robotically scheduled observations to start appearing — discovery takes no per-invocation arguments and needs no redeploy | ✓ VERIFIED | `--proposal` is `required=False, default=None`; the bare path sweeps `watched_rows()` via the shared `sweep_watched_rows()` (backfill:697-769), the same helper `unattended.step_discovery()` calls (unattended.py:42, :375). Admin add + `list_editable` toggle proven to change what `watched_rows()` returns (prior verification's live POST spot-check; `test_admin`/`test_watched_proposal` re-run green today). Live tick reported `0 watched proposals, nothing to discover` — the correct quiet no-op for this database |
| SC3 | A failed unattended run reaches an operator two independent ways: a notification from the command itself, and a heartbeat that also fires when the scheduler never invoked the job at all | ✓ VERIFIED (configuration guidance re-proof is a human item) | Email layer: `unattended._send_notification()` → `notifications.notify_staff()`; 5 `TestNotification` tests (one mail per newly-failing set, suppression, 24 h reminder, one recovery mail, mail outage never raises) — green today. Heartbeat layer: `ping_heartbeat()` pings `/start` before the first step and `/<exit-code>` after the last, `raise_for_status()`ing since IN-01; `TestHeartbeat` asserts `['…/start','…/0']` and `['…/start','…/1']`. The dead-man half was **empirically confirmed by the operator during UAT**: with expected interval 15 min and grace 20 min the check went Late → Down and the alert email arrived while FOMO logged and mailed nothing. What G-36-3 broke was the *guidance*, now corrected — see truth 44 |
| SC4 | No API key or password appears in any log line, notification or error message the unattended path produces | ✓ VERIFIED (with a ⚠️ warning — see WR-22 below) | `TestCredentialHygiene` (8 tests, green today) seeds a fake LCO api_key, a fake `EMAIL_HOST_PASSWORD` and a fake heartbeat URL, forces failure on every step plus the mail send plus the heartbeat ping, and asserts none reaches the captured log, stdout, stderr, `mail.outbox`, or `WatchedProposal.last_run_summary`. Every `warning`/`error` site in the path logs `type(exc).__name__`. `check_unattended` prints names and set/unset only — live run with `FOMO_HEARTBEAT_URL` seeded printed `[ok] heartbeat: FOMO_HEARTBEAT_URL: set -- …` and never the value. **Warning:** two `logger.debug()` sites do format `str(exc)` (WR-22); `settings.LOGGING` pins the root logger to `INFO` (settings.py:201) so neither line is emitted as shipped, and no credential-bearing log line is produced |
| SC5 | An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source | ✓ VERIFIED | `docs/runbooks/telescope_runs_calendar.rst` `.. _unattended-operation:` — one section covering what runs and when, fresh-host setup (7 numbered steps), adding a watched proposal, both failure signals, the "nothing has appeared" checklist, running by hand, and the exact locking guarantee. **UAT Test 6 passed** on a real read-through (user verdict: pass), with two optional clarification notes recorded in `36-UAT.md` (where `FOMO_HEARTBEAT_URL`/`FOMO_BASE_URL` are read; that logrotate install needs sudo) |

#### Plan 36-01 must-have truths (runner, heartbeat, email, crontab, shared mailer)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All-succeeding tick exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_healthy_tick_exits_zero`, `test_pings_start_then_exit_code`, `test_empty_database_tick_is_healthy` — re-run green |
| 2 | Failing tick exits non-zero, pings `/start` then `/<exit-code>`, sends exactly one email to every staff user with an email | ✓ VERIFIED | `test_step_failure_sets_exit_code`, `test_failing_tick_mails_staff_once` (outbox == 1, excludes the staff user with no email and the non-staff user) |
| 3 | Second consecutive failing tick with the same failing set sends no second email; a later success sends exactly one `FOMO unattended run recovered` | ✓ VERIFIED | `test_repeat_failure_is_suppressed`, `test_recovery_mails_once`. State crosses processes via the real JSON file at `<FOMO_STATE_DIR>/unattended-state.json`, now written atomically (IN-03: mkstemp + chmod 0o600 + `os.replace`) |
| 4 | A raising heartbeat ping never changes the exit code, and neither the URL nor the exception message reaches log/stdout/stderr/email | ✓ VERIFIED | `test_ping_failure_never_fails_the_tick`, `test_heartbeat_ping_failure_leaks_nothing` |
| 5 | With `FOMO_HEARTBEAT_URL` unset: one INFO line per tick, no HTTP call, exit code unaffected | ✓ VERIFIED | `test_unset_url_skips_pinging`; `ping_heartbeat()` early return |
| 6 | Failure subject is `FOMO unattended run failed: <step(s)>`; body carries each failed step's summary, the log path and admin/calendar links — no traceback, no request URL, no portal response text | ✓ VERIFIED | `_build_notification_body()`; `test_failure_email_body_carries_no_secret_and_no_traceback` |
| 7 | A second invocation while the first holds `<FOMO_LOCK_DIR>/run_unattended.lock` runs no step and logs one skip line | ✓ VERIFIED | `test_contended_lock_skips_every_step` **plus** this verification's own cross-process `/usr/bin/flock -n` check, re-run today against the current code |
| 8 | A tick with nothing to do anywhere exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_empty_database_tick_is_healthy`; confirmed live (`0 watched proposals, nothing to discover`, exit 0) |
| 9 | Step sequence fixed and identical every tick regardless of failures | ✓ VERIFIED | `STEPS` is a single module-level tuple; `test_all_four_steps_run_in_order`, `test_step_failure_does_not_abort_the_tick`; live banner in that exact order |
| 10 | (backstop) No unattended-path log/email/stdout/stderr write formats the message of an exception caught from a `requests` call, a facility/portal call, or `send_mail()` — only its class name | ⚠️ UNCERTAIN (WARNING) | Holds at every `warning`/`error` site (`unattended.py:158/225/236/576/625/660`, `backfill…:742`) and is behaviourally backed by the 8 green `TestCredentialHygiene` tests. **But** two `DEBUG` sites format `str(exc)`: `backfill_lco_observations.py:349` wraps the authenticated `facility.get_observation_status()` portal call reached from `step_discovery()`, and `unattended.py:191` catches bare `Exception` around `reconcile_run()`. The prior audit scoped only `unattended.py`/`notifications.py`/`run_unattended.py`, so the backfill site was never in scope; the hygiene tests capture at the default level, so they cannot see either. Inert as shipped (root logger `INFO`), so no line is produced — hence UNCERTAIN, not FAILED. Human decision requested |

#### Plan 36-02 must-have truths (WatchedProposal + watched-list discovery)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | Staff can add a proposal code in the admin and toggle `is_active` from the changelist, no redeploy, no per-invocation argument | ✓ VERIFIED | `WatchedProposalAdmin` `list_display`/`list_editable`/`list_filter`; `WatchedProposalAdminTests` re-run green; prior live HTTP POST spot-check (add → 302, toggle → `is_active=False`, `watched_rows() == []`, re-toggle → `['SPOT2026A-001']`) |
| 12 | Bare `backfill_lco_observations` sweeps every active row, applying that row's `target_list_name` and `attributed_to` | ✓ VERIFIED | `sweep_watched_rows()` (backfill:697-769, extracted by IN-13) called from `handle()` :926; `TestWatchedListSweep`; notebook cell output `Swept 2 watched proposal(s), failed: 0` |
| 13 | `--proposal X` still sweeps exactly X with the existing flags, and X need not be a watched row | ✓ VERIFIED | Override branch passes `created_after`/`created_before`/`target_list`/`user` and never consults `WatchedProposal`; `test_backfill_lco_observations` + `test_campaign_submission` green (prior run, unchanged files since) |
| 14 | After a bare sweep every active row carries `last_run_at` and a `last_run_summary` of either the counter line or `failed: <ExceptionClassName>` | ✓ VERIFIED | `sweep_watched_rows()` writes both with `update_fields`; notebook cells show real timestamps, counter lines, and `failed: ConnectionError` |
| 15 | A portal error on one row is caught, recorded, counted, and the remaining rows are still swept; exit non-zero only because a row failed | ✓ VERIFIED | `sweep_watched_rows()` per-row try/except (backfill:742 logs the class name only); notebook cell 20 shows row B still creating record 900402 after row A failed |
| 16 | Two rows can never share a `proposal_code` | ✓ VERIFIED | `unique=True` in model and migration 0022; `test_duplicate_code_rejected` |
| 17 | Zero active rows → one INFO line, nothing written, exit 0 | ✓ VERIFIED | `TestEmptyWatchedList`; observed live in today's tick |
| 18 | `proposal_code` equality is exact-match, case-sensitive, no Unicode normalization; whitespace stripped on save | ✓ VERIFIED | `WatchedProposal.save()`; `test_code_is_stripped_on_save`, `test_code_comparison_is_case_sensitive` |
| 19 | Active rows are swept in deterministic `proposal_code` ascending order | ✓ VERIFIED | `Meta.ordering = ['proposal_code']` in model and migration options; `watched_rows()` inherits it |

#### Plan 36-03 must-have truths (the remaining three steps, SCHED-10 suite)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 20 | One tick runs all four steps in the fixed order status refresh → projector sweep → discovery → reconcile, in one process, no `manage.py` subprocess | ✓ VERIFIED | `STEPS` tuple; today's live tick in that exact order; no `call_command`/`subprocess`/`Popen`/`os.system` in `unattended.py` or `run_unattended.py` |
| 21 | Status refresh calls `update_all_observation_statuses()` on a fresh `LCOFacility()` and a fresh `SOARFacility()` — never shared — and a non-empty failure list is a step failure | ✓ VERIFIED | `test_calls_both_facilities_with_fresh_instances`, `test_non_empty_failure_list_is_a_step_failure`, `test_facility_exception_is_isolated_per_facility`; IN-05 also stopped a whole-facility outage being reported as "failed 1" |
| 22 | A status-refresh failure is logged as `observation_id` plus the exception class name; TOM's message half never reaches a log line, stdout/stderr, or the email | ✓ VERIFIED | `_refresh_one_facility()` discards `_message` at the unpack; `test_failure_is_reported_by_class_name_not_message`, `test_status_refresh_portal_error_leaks_nothing` |
| 23 | A raising step never prevents later steps; every registry step is attempted every tick and each outcome recorded separately | ✓ VERIFIED | `run_tick()` per-step try/except; `test_step_failure_does_not_abort_the_tick`, `test_all_four_steps_run_in_order` |
| 24 | Projector sweep calls `project_queryset()` directly with the same observed-site hook, counts `unprojectable` as a failure, never `call_command()` | ✓ VERIFIED | `test_unprojectable_row_is_a_failure`, `test_site_lookup_hook_is_passed_on_a_real_run_and_omitted_on_dry_run`, `test_step_never_calls_the_management_command`; live sweep reported the real counter line |
| 25 | Discovery sweeps every active row and fails only when at least one row failed; zero rows logs one INFO line and is healthy | ✓ VERIFIED | `step_discovery()` → `sweep_watched_rows()`; `test_sweeps_every_active_row`, `test_empty_list_is_healthy`, `test_one_failing_row_does_not_stop_the_others`, `test_failing_row_names_the_proposal_in_the_summary` |
| 26 | `unchanged` / `skipped` / `detach_declined` / `remint_declined` never make the tick non-zero and never trigger the email | ✓ VERIFIED | `test_expected_data_shape_outcomes_are_not_failures`; failure derives from typed counters, never from parsing stdout |
| 27 | With a fake api key, mail password and heartbeat URL seeded and every failure path forced in turn, none appears in logs/stdout/stderr/`mail.outbox` | ✓ VERIFIED | The 8-test `TestCredentialHygiene` class, green today. (Scope caveat: captures at the default level, which is why WR-22's DEBUG sites were invisible to it — see truth 10) |

#### Plan 36-04 must-have truths (`check_unattended`, cron line, logrotate)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 28 | `check_unattended` reports every prerequisite in one run | ✓ VERIFIED | Live run today printed flock, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`, `FOMO_STATE_DIR`, `EMAIL_BACKEND`, `staff_recipients`, `heartbeat`, `FOMO_BASE_URL` and `watched_proposals` in one pass |
| 29 | A missing hard prerequisite makes the command exit non-zero and name which check failed | ✓ VERIFIED | `test_missing_flock_fails`, `test_unwritable_lock_dir_fails`, `test_unwritable_log_dir_fails`, `test_console_email_backend_fails`, `test_no_staff_email_fails` (30 tests green today); previously observed live exiting 1 with the correct names before this host was provisioned |
| 30 | Unset heartbeat URL and empty watched list are warnings, not failures; exit stays 0 when every hard check passed | ✓ VERIFIED | `hard=False` on both; live run today exited **0** with `[WARN] watched_proposals` and every hard check `[ok]`; `test_warnings_do_not_mask_a_hard_failure`, `test_all_hard_checks_passing_exits_zero` |
| 31 | The printed cron line carries the real resolved interpreter and `manage.py` path | ✓ VERIFIED | Live output today: `… /home/tlister/venv/devel_fomo311_venv/bin/python /home/tlister/git/fomo_devel/manage.py run_unattended …`; `test_line_has_real_paths` also asserts both placeholders absent |
| 32 | Names and set/unset status only — never a value | ✓ VERIFIED | Live `[ok] heartbeat: FOMO_HEARTBEAT_URL: set -- confirm the check's own expected ping interval (Period) is 15 min, not its 1-day default` with the seeded URL absent; `test_line_carries_no_setting_value`, `test_output_never_contains_a_seeded_value`, and the new `test_set_heartbeat_reminds_about_the_check_period` (asserts `_FAKE_HEARTBEAT_URL` not in stdout) |
| 33 | `--send-test-email` sends one message through the configured backend to the same staff recipient list the failure notice uses | ✓ VERIFIED | `_send_test_email()` → `notifications.notify_staff()` (check_unattended.py:348) — literally the same helper; 3 tests; **UAT Test 4 passed** against real SMTP |
| 34 | The command only reads: creates no directory, writes no file, changes no row | ✓ VERIFIED | `test_command_writes_nothing`; no `mkdir`/write-mode `open()` in the module |

#### Plan 36-05 must-have truths (paired docs)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 35 | An operator can set up or verify the whole schedule from one runbook section without reading source | ✓ VERIFIED | The `_unattended-operation` section's seven subsections; **UAT Test 6 passed** |
| 36 | The section documents both failure signals — email (who, what, repeat, clear) and heartbeat (`/start`, `/<exit-code>`, expected ping interval (Period) plus grace time (Grace), alerting at last ping + Period + Grace) — plus the "nothing has appeared" checklist | ✓ VERIFIED | Runbook:1523-1568 covers recipients, subject form, contents, the 24 h reminder, the recovery mail, and (post-G-36-3) the full two-knob configuration with the arithmetic; "When nothing has appeared" (:1570-1602) gives the D-18 four-step checklist in order, item 3 now bounded by interval + grace |
| 37 | The backfill section documents `--proposal` as optional, the bare watched sweep, per-row overrides and per-row failure isolation | ✓ VERIFIED | All four points stated explicitly, including class-name-only failure recording |
| 38 | The cheat-sheet carries rows for `run_unattended` and `check_unattended`, and `backfill_lco_observations` reflects its optional-argument contract | ✓ VERIFIED | Cheat-sheet rows present for all three |
| 39 | The overlap guarantee is stated at exactly its true strength | ✓ VERIFIED | "What the locking does and does not cover" names `run_unattended --step <name>` as the exclusive manual route and does not overclaim for directly-run sweep commands — matching the code, where only `unattended.py` takes the named locks |
| 40 | The notebook contains executed cells seeding `WatchedProposal` rows, running the command bare, showing `last_run_summary`, and showing one proposal failing without stopping the other | ✓ VERIFIED | Real committed stdout in the watched-proposal cells; `execution_count` sequential across all code cells (IN-11 re-execution confirmed by review iteration 3) |
| 41 | `docs/notebooks.rst` lists `backfill_lco_observations_demo` in the Demonstration Notebooks toctree | ✓ VERIFIED | Toctree line present; referenced file exists; Sphinx build clean |
| 42 | `CLAUDE.md`'s notebook map records the pairing for every module this phase adds, including that the runbook section — not a notebook — is the paired doc for the runner and `check_unattended`, and why | ✓ VERIFIED | CLAUDE.md maps `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py` → the runbook's "How do I run everything unattended?" section with the stated reason |
| 43 | No committed doc, notebook cell or runbook example quotes a live setting, a `local_settings.py` value, or any credential | ✓ VERIFIED | Notebook uses `BACKFILL-DEMO-2026A`/`-B`; crontab template uses `/path/to/venv/bin/python` and `/path/to/checkout/manage.py`; logrotate uses the committed default path; runbook and crontab name env vars only. Re-grepped the gap-closure diff for URL/token/api_key/password patterns: none |

#### Plan 36-06 must-have truths (G-36-3 heartbeat guidance correction)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 44 | An operator configuring the check from the corrected runbook alone gets late ~15 min / alert ~35 min — not a day later (G-36-3's failed truth, restored) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | The guidance now prescribes exactly the configuration the operator's own UAT A/B proved works (Period 15 + Grace 20 → Late → Down → alert email, FOMO silent). But no one has yet configured a check from the corrected paragraph alone, which is the only way G-36-3 was findable. Plan 36-06 declares this as its one `<human-check>`. → Human Verification item 1 |
| 45 | The guidance names BOTH knobs, states the arithmetic (alert at last ping + expected interval + grace), and names the 1-day interval default as the trap | ✓ VERIFIED | runbook:1546-1568 — "the check needs two settings, not one … the check's expected interval between pings (healthchecks.io calls this ``Period``) and its grace time (``Grace``) … The service alerts at last ping + expected interval + grace: with 15 and 20 … late about 15 minutes … alerts about 35 minutes … Leaving the expected interval (``Period``) at its default -- 1 day on healthchecks.io -- means the first alert arrives about a day later while the check looks green" |
| 46 | The Cron-type `*/15 * * * *` alternative is offered, and the grace stays at about 20 min with its `/start`-to-completion role explained | ✓ VERIFIED | runbook:1552-1561 — "the drift-free alternative is a Cron-type check carrying the same ``*/15 * * * *`` expression … Keep the grace time at about 20 minutes … do not shrink it to make the total look shorter: because the runner sends a ``/start`` ping, the grace time also bounds the maximum allowed gap between that ping and the completion ping" |
| 47 | The "When nothing has appeared" triage item judges a stale ping against expected interval + grace, not grace alone | ✓ VERIFIED | runbook:1584-1587 — "older than the expected interval plus the grace time -- about 35 minutes with the recommended 15/20 settings" |
| 48 | A troubleshooting entry covers the exact symptom the gap produced, naming the still-defaulted expected interval as the cause and the fix | ✓ VERIFIED | runbook:2092-2108, title `The heartbeat never alerted although the schedule stopped`, `^`-underlined like its siblings, **Cause:**/**Fix:** form, names the 1-day default, gives the 15-min fix or Cron type, the expected confirmation, and a standing check. Sphinx builds the page with no warning |
| 49 | `deploy/cron/fomo.crontab.example` describes the backstop with both knobs, and `check_unattended`'s `[ok]` heartbeat line reminds the operator the remote check still needs its own expected ping interval | ✓ VERIFIED | crontab:31-35 — "but only if the check at the other end is configured with BOTH its expected ping interval (15 min, matching this schedule; healthchecks.io calls this Period) and its grace time (about 20 min)". Preflight detail confirmed **live**: `[ok] heartbeat: FOMO_HEARTBEAT_URL: set -- confirm the check's own expected ping interval (Period) is 15 min, not its 1-day default`. (See advisory IN-20 — the detail names the interval but not the grace) |
| 50 | The verification record's human-test script and its runbook evidence cell state the corrected time-to-alert, traceable to the gap id | ✓ VERIFIED | Commit `1f3bbac` corrected the frontmatter `human_verification` test-3 entry, its `why_human`, the prose script and the plan 36-05 evidence row, adding a note naming G-36-3 and pointing at `36-UAT.md` and the debug session. This rebuild carries the corrected wording forward and cites the gap in `re_verification.gaps_closed` |
| 51 | Sphinx still builds the runbook with no new warning and `python manage.py test solsys_code.tests.test_check_unattended` stays green | ✓ VERIFIED | Run by this verification: `pre-commit run sphinx-build --all-files` → **Passed**, exit 0, zero `warning`/`error` lines in the log, no `telescope_runs_calendar` warning. `python manage.py test solsys_code.tests.test_check_unattended` → **Ran 30 tests … OK**, and the named new test alone → **Ran 1 test … OK**. `pre-commit run ruff --all-files` → Passed |

**Score:** 54/56 truths verified (1 present-but-behavior-unverified: truth 44; 1 uncertain: truth 10)

---

### Plan 36-06 Prohibitions (must-NOT checks)

All four are judgment-tier (`verification: flagged-unverified`). Per the fail-closed rule these carry a **non-authoritative LLM-judge verdict** and are flagged for human review; none is silently passed.

| # | Prohibition | Judge verdict | Evidence | Flag |
|---|-------------|---------------|----------|------|
| P1 | No new sentence may quote a real heartbeat URL, a real ping UUID, or any other `FOMO_*` value (D-15, T-36-04) | Satisfied | Grepped every added line of `12c51c6..1f3bbac` for URL-with-token / `api_key` / password patterns: none. Runbook and crontab name `FOMO_HEARTBEAT_URL` as a setting only; the preflight's new detail string prints `set`, never the value, and `test_set_heartbeat_reminds_about_the_check_period` asserts the seeded fake URL is absent from stdout | unverified-prohibition — human review recommended |
| P2 | The grace-time recommendation must stay at about 20 minutes (it also bounds `/start`-to-completion) | Satisfied | runbook:1556 "Keep the grace time at about 20 minutes"; :1557-1561 gives exactly the stated reason; crontab:34 "its grace time (about 20 min)"; troubleshooting :2103-2104 "leave the grace at about 20 minutes" | unverified-prohibition — human review recommended |
| P3 | The wording must stay service-agnostic: concept first, healthchecks.io's knob name as the vendor spelling | Satisfied | runbook:1546-1552 keeps "any healthchecks-compatible endpoint (hosted or self-hosted)" and names "the check's expected interval between pings (healthchecks.io calls this ``Period``)", explicitly adding "since other healthchecks-compatible endpoints may spell the same two concepts differently". One stylistic residue: the prose retains the plan's own instruction phrase "Name the concept first:" (see Anti-Patterns) | unverified-prohibition — human review recommended |
| P4 | No task may change `ping_heartbeat()`, the URL construction, or the exit-code path — only two docstring sentences and one preflight detail string | Satisfied | `git diff 12c51c6^..1f3bbac -- solsys_code/unattended.py solsys_code/management/commands/check_unattended.py` shows exactly three hunks: `TickResult` docstring (2 lines), `run_tick()` Returns block (2 lines), and `check_heartbeat()`'s `[ok]` `CheckResult` detail. `ping_heartbeat()` is untouched | unverified-prohibition — human review recommended |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/unattended.py` | Runner, steps, lock, heartbeat, notification | ✓ VERIFIED | 668 lines; imported by `run_unattended.py:16`; executed live today |
| `solsys_code/notifications.py` | Shared request-free mailer | ✓ VERIFIED | 95 lines; called from `unattended.py:574`, `check_unattended.py:348`, `campaign_views.py:340` |
| `solsys_code/management/commands/run_unattended.py` | Cron entry point | ✓ VERIFIED | 60 lines; thin wrapper; executed live |
| `solsys_code/management/commands/check_unattended.py` | Preflight + cron line | ✓ VERIFIED | 446 lines; executed live, exit 0 on this now-provisioned host, correct cron line printed |
| `solsys_code/models.py` (`WatchedProposal`) | Admin-editable watch list | ✓ VERIFIED | Model + `save()` strip + `Meta.ordering`; queried by `watched_rows()` and `check_watched_proposals()` |
| `solsys_code/migrations/0022_watchedproposal.py` | Matching migration | ✓ VERIFIED | 31 lines; model/migration consistency previously confirmed by `makemigrations --check --dry-run` |
| `solsys_code/admin.py` (`WatchedProposalAdmin`) | Registered, `list_editable` | ✓ VERIFIED | `test_admin` green |
| `solsys_code/management/commands/backfill_lco_observations.py` | `sweep_proposal()`, `watched_rows()`, `sweep_watched_rows()`, optional `--proposal` | ✓ VERIFIED | Shared sweep helper imported by `unattended.py:42`; single source for both callers (IN-13) |
| `deploy/cron/fomo.crontab.example` | `*/15`, `flock -n -E 99`, redirect, skip tail, explicit `exit $rc`, no secret | ✓ VERIFIED | Line 52 carries all of them; comment block now names both heartbeat knobs; placeholders only |
| `deploy/logrotate/fomo.example` | Daily, rotate 14, copytruncate | ✓ VERIFIED | 32 lines; **UAT Test 5 passed** against a real forced rotation mid-tick |
| `src/fomo/settings.py` | `FOMO_BASE_URL`/`HEARTBEAT_URL`/`LOCK_DIR`/`STATE_DIR`/`LOG_FILE` | ✓ VERIFIED | All `os.getenv()`-sourced and read by runner, notifications and preflight |
| `docs/runbooks/telescope_runs_calendar.rst` | Unattended section + cheat-sheet + troubleshooting + corrected heartbeat guidance | ✓ VERIFIED | `_unattended-operation` label; corrected at all four G-36-3 sites plus the new entry; Sphinx clean |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | Executed watched-proposal cells | ✓ VERIFIED | 1028 lines; real committed output; sequential `execution_count` |
| `solsys_code/tests/test_check_unattended.py` | New reminder regression test | ✓ VERIFIED | `test_set_heartbeat_reminds_about_the_check_period` at :241-250, with the G-36-3 rationale comment; runs green alone and in the module |
| Tests (`test_unattended`, `test_check_unattended`, `test_watched_proposal`, `test_admin`) | Behavioral coverage | ✓ VERIFIED | **157 tests re-run green today** (127 + 30) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `run_unattended.Command.handle()` | `unattended.run_tick()` → `STEPS` | direct call | ✓ WIRED | `run_unattended.py:16` imports the module; `--step` choices derive from the single `STEPS` tuple |
| `unattended.run_tick()` | `notifications.notify_staff()` | `_send_notification()` | ✓ WIRED | `unattended.py:574`; the same helper `campaign_views._notify_staff()` calls (`campaign_views.py:340`) |
| crontab template's `flock -n` path | `settings.FOMO_LOCK_DIR` | `<dir>/run_unattended.cron.lock` (distinct from the runner's own `run_unattended.lock`, CR-01) | ✓ WIRED | Proven empirically again today: `/usr/bin/flock -n` on the runner's lock path made a real invocation skip every step |
| `unattended.run_tick()` | `<FOMO_STATE_DIR>/unattended-state.json` | `load_state()`/`save_state()` | ✓ WIRED | Real JSON file, now written atomically (IN-03) |
| `WatchedProposal.objects.filter(is_active=True)` | `sweep_proposal()` | `watched_rows()` → `sweep_watched_rows()` | ✓ WIRED | One shared helper, two callers (`backfill…:926`, `unattended.py:375`) |
| `sweep_proposal()` summary | `WatchedProposal.last_run_summary` → admin column | `row.save(update_fields=…)` | ✓ WIRED | `list_display` includes it; notebook shows real values |
| runbook heartbeat paragraph | the heartbeat service's alert rule (last ping + expected interval + grace) | corrected prose | ✓ WIRED | **This is the link G-36-3 proved missing.** Now stated verbatim at runbook:1561-1564, with the 1-day trap named at :1564-1568 |
| runbook heartbeat paragraph | `deploy/cron/fomo.crontab.example:31-35` ↔ `check_heartbeat()`'s `[ok]` detail | shared two-knob wording | ✓ WIRED (partial on the third surface) | Runbook and crontab name both knobs; the preflight detail names the interval only — advisory IN-20, not a must-have breach |
| `check_heartbeat()`'s detail string | `test_set_heartbeat_reminds_about_the_check_period` | assertion on stdout | ✓ WIRED | Asserts `[ok] heartbeat` and `Period` present, seeded URL absent |
| corrected runbook | `36-VERIFICATION.md`'s human-test script | this document | ✓ WIRED | The rebuilt Test 3 script prescribes the two-knob configuration, so re-verification cannot regenerate the gap |
| `check_unattended --send-test-email` | `notifications.notify_staff()` | direct call | ✓ WIRED | `check_unattended.py:348` |
| `step_status_refresh()` | Phase 34 `post_save` receiver | `facility.update_observation_status()` → `.save()` | ✓ WIRED | Receiver connected in `observation_projector`; the sweep step's test disconnects/reconnects it deliberately |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `step_project_sweep()` | `result['counters']` | `project_queryset()` over `ObservationRecord.objects.filter(facility__in=PROJECTED_FACILITIES)` | Yes — today's live tick reported `unchanged: 159` from the dev DB | ✓ FLOWING |
| `step_reconcile()` | `run_count`/`failed_count` | `CampaignRun.objects.all()` | Yes — today's live tick reported `runs: 45, failed: 0` | ✓ FLOWING |
| `step_discovery()` | `rows` | `watched_rows()` → `WatchedProposal` query | Yes — 0 rows live (correct quiet no-op), 2 rows with real portal-shaped payloads in the notebook | ✓ FLOWING |
| `WatchedProposalAdmin` changelist | `last_run_at`/`last_run_summary` | sweep-written model fields | Yes — notebook cells show real written values | ✓ FLOWING |
| `check_unattended` report | `results` | live `shutil.which`, `os.access`, settings, `WatchedProposal` count | Yes — today's live run produced host-accurate verdicts including real uid/owner/mode details | ✓ FLOWING |
| `check_heartbeat()` `[ok]` detail | static reminder string | n/a — deliberately a constant reminder, never the URL value | n/a (by design; printing the value would breach SCHED-10) | ✓ FLOWING |
| Failure email body | `settings.FOMO_LOG_FILE`, `FOMO_BASE_URL` | settings, not hardcoded literals | Yes — asserted present in the body by test | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full tick runs all four steps in D-01 order against the real dev DB | `FOMO_LOCK_DIR=… FOMO_STATE_DIR=… python manage.py run_unattended --dry-run` | `START 2026-09-18T01:46:33` → status_refresh ok → project_sweep ok (159 unchanged) → discovery ok (0 watched) → reconcile ok (45 runs) → `END … exit=0` | ✓ PASS |
| Cross-process overlap guard (cron's own `flock -n` vs the runner's `fcntl.flock`) | `/usr/bin/flock -n <lock> -c "… run_unattended --dry-run"` | `run_unattended: lock held -- skipping this tick`, no step ran, exit 0 | ✓ PASS |
| Preflight on the now-provisioned host, heartbeat URL seeded | `FOMO_HEARTBEAT_URL=… python manage.py check_unattended` | exit 0; every hard check `[ok]`; `[ok] heartbeat: … confirm the check's own expected ping interval (Period) is 15 min, not its 1-day default`; `[WARN] watched_proposals`; correct cron line with real paths; seeded URL never printed | ✓ PASS |
| The new preflight reminder test alone | `python manage.py test …test_check_unattended.TestWarningChecks.test_set_heartbeat_reminds_about_the_check_period` | Ran 1 test — OK | ✓ PASS |
| `check_unattended` module | `python manage.py test solsys_code.tests.test_check_unattended` | Ran 30 tests — OK | ✓ PASS |
| Runner / model / admin modules | `python manage.py test solsys_code.tests.test_unattended test_watched_proposal test_admin` | Ran 127 tests — OK | ✓ PASS |
| Sphinx docs gate | `pre-commit run sphinx-build --all-files` | Passed, exit 0; zero warning/error lines; no `telescope_runs_calendar` warning | ✓ PASS |
| Lint gate | `pre-commit run ruff --all-files` (pinned ruff 0.2.1) | Passed | ✓ PASS |
| Secret leakage in the gap-closure diff | `git diff 12c51c6^..1f3bbac \| grep -E '^\+.*(https?://…token\|api_key\|PASSWORD =)'` | none | ✓ PASS |
| Debt markers in the gap-closure diff | `git diff 12c51c6^..1f3bbac \| grep -E '^\+.*(TBD\|FIXME\|XXX\|HACK\|TODO)'` | none | ✓ PASS |
| Live external heartbeat alerting from the corrected guidance | — | requires a live healthchecks-compatible account | ? SKIP → human verification |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN or SUMMARY declares one; this project's verification contract is the Django test runner (36-VALIDATION.md) | n/a — SKIPPED |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCHED-08 | 36-01, 36-03, 36-04, 36-05, 36-06 | Projector sweep, discovery backfill and reconciler on a documented cron + `flock -n` schedule with no operator action, guarded against overlapping invocations | ✓ SATISFIED | `STEPS` + `run_unattended` + committed crontab template + `check_unattended`'s printed line; overlap re-proven cross-process today; **UAT Test 2 passed on the real host** |
| SCHED-09 | 36-01, 36-03, 36-05, 36-06 | Failure visible through two independent layers — in-command notification and a heartbeat/dead-man's switch | ✓ SATISFIED (one human re-proof open) | `notifications.notify_staff()` (5 tests, UAT Test 4 passed against real SMTP) + `ping_heartbeat()` `/start` / `/<exit-code>` (3 tests; dead-man half empirically confirmed in UAT once the expected interval was set). G-36-3's guidance defect is closed in all five operator-facing surfaces; re-running UAT Test 3 from the corrected text is human item 1 |
| SCHED-10 | 36-01, 36-03, 36-04, 36-05 | No credential value in any log line or notification the unattended path generates | ✓ SATISFIED (⚠️ with WR-22 noted) | Class-name-only discipline at every `warning`/`error` site, 8 `TestCredentialHygiene` tests green, `check_unattended`'s names-only output verified live, no committed artifact carrying a value. Two `DEBUG` sites format `str(exc)` and are inert only because the root logger is pinned to `INFO` — see advisory WR-22 and truth 10 |
| DISCOVER-01 | 36-02, 36-03, 36-05 | Admin-editable watched-proposal list replaces per-invocation `--proposal`/name-prefix arguments | ✓ SATISFIED | `WatchedProposal` model/migration/admin; `--proposal` optional; bare sweep over `watched_rows()` through the shared `sweep_watched_rows()`; admin toggle proven to change discovery scope end-to-end |

**Orphaned requirements:** none. `.planning/REQUIREMENTS.md:121-124` maps exactly SCHED-08, SCHED-09, SCHED-10 and DISCOVER-01 to Phase 36, and all four are claimed by plan frontmatter (36-06 claims SCHED-08 and SCHED-09).

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/management/commands/backfill_lco_observations.py` | 349 | `logger.debug(f'Observed-block lookup failed for observation_id={observation_id!r}: {exc}')` — formats the message of an exception from an authenticated LCO portal call, on the unattended discovery path | ⚠️ Warning | Falsifies plan 36-01 truth 10 as written. Inert as shipped (`settings.LOGGING` root level `INFO`, settings.py:201), so no line is produced and SC 4 holds; live the moment anyone raises the level. Human decision requested (human item 2) |
| `solsys_code/unattended.py` | 191 | `logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)` under a bare `except Exception` | ⚠️ Warning | Same class of exposure. The `# noqa` comment claims D-17's second bucket ("FOMO's own call"), but the bare `except` also catches anything `reconcile_run()` propagates from deeper in the chain |
| `solsys_code/management/commands/check_unattended.py` | 242-246 | `[ok]` heartbeat detail names the expected interval but not the grace time; the new test pins only `'Period'` | ℹ️ Info | Advisory IN-20. Plan 36-06 truth 6 asks only for the interval reminder, so not a must-have breach; an operator acting on the preflight alone would leave the vendor's 1-hour default grace |
| `docs/runbooks/telescope_runs_calendar.rst` | 1548-1549 | The prose retains the plan's own authoring instruction: "…but the check needs two settings, not one. **Name the concept first:** the check's expected interval between pings…" | ℹ️ Info | Reads as a directive to the writer rather than to the operator. Harmless to correctness; worth a one-phrase copy-edit ("These are:" / "They are:") |
| `solsys_code/unattended.py` | 646-660 | An unwritable `FOMO_STATE_DIR` at runtime still permits a per-tick mail storm (only the setup-time preflight check was added) | ℹ️ Info | Review WR-17; pre-existing, guarded at setup time by `check_unattended`'s hard `FOMO_STATE_DIR` check, which passed live on this host |
| phase-modified files (all) | — | `TBD` / `FIXME` / `XXX` / `HACK` debt markers | — none | Scanned every phase-modified source, deploy and doc file. The only `TBD` hits are in `solsys_code/models.py` (`'Original Obs. Date text (TBD rows only)'`, the "TBD branch" constraint comments) — the campaign domain's own window vocabulary, not debt markers. The gap-closure diff added none |

No 🛑 Blockers.

---

### CLAUDE.md Paired-Docs Compliance

Plan 36-06 changed `check_unattended.py` (operator-visible output) and `unattended.py` (docstrings only). CLAUDE.md's map pairs both of those modules with the runbook's "How do I run everything unattended?" section — **not** a notebook — and that section was updated in the same plan, including the step-4 prose describing what the preflight now reports (runbook:1484-1487). Compliant.

No mapped notebook's module changed behavior in 36-06 (`backfill_lco_observations.py` was untouched by this plan), so no notebook re-execution was owed. The wider phase's notebook obligation (`backfill_lco_observations_demo.ipynb`) was met in 36-05 and re-confirmed here.

---

### Human Verification Required

#### 1. Re-run UAT Test 3 — the heartbeat's dead-man half, from the corrected guidance

**Test:** Create a fresh healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL`, configuring **only** what the corrected runbook paragraph (`docs/runbooks/telescope_runs_calendar.rst:1541-1568`) names: an expected ping interval (`Period`) of 15 minutes — or a Cron-type check with `*/15 * * * *` — and a grace time (`Grace`) of about 20 minutes. Then disable the crontab line, simulating the scheduler never invoking the job.
**Expected:** The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping (last ping + expected interval + grace), while FOMO itself logs nothing and sends no email — SC 3's second, independent layer.
**Why human:** The signal comes from the external service's own timer, not from any FOMO code path. G-36-3 was only findable this way: every internal gate checked the runbook against decision D-12, and D-12 itself carried the conflation.

#### 2. Decide on WR-22 before shipping

**Test:** Either fix the two `logger.debug()` sites (`backfill_lco_observations.py:349`, `unattended.py:191`) to log `type(exc).__name__`, and extend `TestCredentialHygiene` with a case that runs a tick under `self.assertLogs(level='DEBUG')` asserting the seeded key and ping URL appear nowhere — or record an explicit acceptance that the class-name-only discipline holds only while `settings.LOGGING` keeps the root logger at `INFO`, documented where an operator who raises the level would see it.
**Expected:** A decision, either way, recorded against the phase.
**Why human:** A judgment call: plan 36-01's truth 10 is written absolutely and is falsified at the source level, while SC 4 as written ("appears in any log line the unattended path **produces**") is not breached under the shipped configuration. Which reading governs the ship decision is the developer's to make.

**Closed by UAT (no longer open):** the fresh-host preflight (Test 1, pass), the real crontab (Test 2, pass), real SMTP delivery (Test 4, pass), logrotate under a live writer (Test 5, pass) and runbook sufficiency (Test 6, pass). These five were carried as human items by the previous verification and have since been exercised on the real host; see `36-UAT.md`.

---

### Gaps Summary

**No gaps.** G-36-3 — the phase's one UAT failure — is closed in the codebase, verified against the files rather than the SUMMARY: the runbook now names both knobs, states the alert arithmetic verbatim, offers the Cron-type alternative, explains why the grace must not be shrunk, names the 1-day default as the trap, fixes the staleness-triage bound and the lock-contention backstop sentence, and adds a troubleshooting entry for the exact silent symptom. The same two-knob wording reached `deploy/cron/fomo.crontab.example`; the preflight's `[ok] heartbeat` line was confirmed **live** to carry the interval reminder without ever printing the URL; a named regression test pins that reminder; and `ping_heartbeat()` — which the debug session cleared — was not touched, exactly as the plan's prohibition required.

The phase is held at `human_needed` for two items. The first is unavoidable: G-36-3's truth can only be closed end-to-end by configuring a live external check from the corrected paragraph alone, which is the same human-only proof that found the gap in the first place. The second is an escalation rather than a defect: review iteration 3 found two `DEBUG`-level sites that format a raw exception message on the unattended path, one of them wrapping an authenticated portal call. They produce nothing under the shipped `INFO` root logger, so SC 4 stands, but plan 36-01's truth 10 is written absolutely and is falsified at the source level — so it is recorded as UNCERTAIN with the decision surfaced rather than silently absorbed into a clean score.

Everything else was re-proven on the current code, not carried over on trust: the runner was executed end-to-end against the real developer database, the overlap guarantee was re-demonstrated cross-process with the same `flock(1)` binary the crontab invokes, `check_unattended` was run live on the now-provisioned host and exited 0 with every hard check green and a correct cron line, 157 phase tests were re-run green, and the Sphinx and ruff gates were re-run by this verification rather than quoted from the SUMMARY.

---

_Verified: 2026-09-18T02:05:00Z_
_Verifier: Claude (gsd-verifier)_
