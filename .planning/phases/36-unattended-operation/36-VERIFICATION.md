---
phase: 36-unattended-operation
verified: 2026-09-17T17:20:00Z
status: human_needed
score: 48/48 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
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
covered_digest: "v1:sha256:f7f55af98aff019f59cde7bf141c66ec0049f7ffdef5800f5e2a04646a670135"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "On the real FOMO host, create /var/lock/fomo and /var/log/fomo writable by the cron account, put the real EMAIL_BACKEND/EMAIL_HOST_* and the LCO/SOAR api_key in local_settings.py, export FOMO_HEARTBEAT_URL and FOMO_BASE_URL for the cron daemon, then run `python manage.py check_unattended`."
    expected: "Every hard check reports [ok], the command exits 0, and it prints a cron line with this host's real interpreter and manage.py paths."
    why_human: "Host filesystem, real SMTP settings and the cron daemon's environment do not exist on the dev machine; on this checkout the command correctly exits 1 naming FOMO_LOCK_DIR/FOMO_LOG_FILE/EMAIL_BACKEND."
  - test: "Install the printed cron line in the FOMO service account's crontab (`crontab -e`) and leave it for ~45 minutes."
    expected: "Three START/per-step/END banners appear in /var/log/fomo/unattended.log, roughly 15 minutes apart, with nobody typing anything."
    why_human: "SC 1's 'on the real host' half needs a real crontab on the production host; only the mechanism, the committed template and the printed line can be verified here."
  - test: "Create a healthchecks-compatible check pointed at FOMO_HEARTBEAT_URL, configuring only what the corrected runbook paragraph names: an expected ping interval (Period) of 15 minutes and a grace time (Grace) of about 20 minutes. Then disable the crontab line (simulating the scheduler never invoking the job) and watch the check."
    expected: "The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping (last ping + expected interval + grace), even though FOMO itself logged nothing and sent no email — the second, independent layer of SC 3."
    why_human: "The 'scheduler never ran the job' signal is produced by the external heartbeat service's own expected-interval-plus-grace timer, not by any FOMO code path."
  - test: "With the real (non-console) email backend configured, run `python manage.py check_unattended --send-test-email`."
    expected: "One message arrives in the mailbox of every staff user with an email on file — the same recipient rule the failure notice uses."
    why_human: "Real SMTP delivery cannot be exercised from the dev host, whose EMAIL_BACKEND is the console backend."
  - test: "Drop deploy/logrotate/fomo.example into /etc/logrotate.d/fomo and run `logrotate -d /etc/logrotate.d/fomo` (dry run), then force one real rotation while a tick is running."
    expected: "The stanza parses, and the copytruncate strategy keeps cron's still-open append redirect writing to the live file rather than the rotated-away inode."
    why_human: "Requires root, a real logrotate installation, and a concurrently running tick holding the log file descriptor open."
  - test: "Hand the runbook's 'How do I run everything unattended?' section to someone who has not read this phase's source and ask them to set the schedule up on a fresh host."
    expected: "They get to a working, checked schedule using only that one section — no source reading, no questions back."
    why_human: "SC 5 is a documentation-usability judgment; completeness can be checked mechanically, sufficiency for a real operator cannot."
---

# Phase 36: Unattended Operation Verification Report

**Phase Goal:** The projector sweep, the LCO/SOAR discovery backfill and the reconciler run on the real host on a documented schedule with nobody typing anything, against a watched-proposal list an operator edits in the admin — and when it breaks, an operator finds out.
**Verified:** 2026-09-17T17:20:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

#### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Projector sweep, LCO/SOAR discovery backfill and reconciler all run on a documented recurring schedule with no operator action, and two invocations never overlap | ✓ VERIFIED (host install is a human item) | Real tick executed against the dev database: `FOMO_LOCK_DIR=… python manage.py run_unattended --dry-run` → `START` / `step status_refresh: ok` / `step project_sweep: ok … unchanged: 159` / `step discovery: ok` / `step reconcile: ok | runs: 45, failed: 0` / `END exit=0`. Overlap proven **cross-process**: holding the lock with the same `/usr/bin/flock -n` the crontab uses made a second real invocation print `run_unattended: lock held -- skipping this tick` and run no step. `deploy/cron/fomo.crontab.example:27` carries the `*/15` + `flock -n` + redirect + skip-tail line |
| SC2 | Adding a proposal in the admin is enough for its robotically scheduled observations to start appearing — discovery takes no per-invocation arguments and needs no redeploy | ✓ VERIFIED | Verifier spot-check on a throwaway test DB drove the real admin: POST to `admin:solsys_code_watchedproposal_add` created the row (`is_active=True`); a `list_editable` POST to the changelist flipped it to `False`; `watched_rows()` then returned `[]`; re-checking it returned `['SPOT2026A-001']`. `backfill_lco_observations` `--proposal` is `required=False, default=None` (line 740-749) and the bare path sweeps `watched_rows()` (line 814) |
| SC3 | A failed run reaches an operator two independent ways: a notification from the command itself, and a heartbeat that also fires when the scheduler never invoked the job | ✓ VERIFIED (external alerting is a human item) | Email layer: `unattended._send_notification()` → `notifications.notify_staff()`; `TestNotification` (5 tests) proves one mail per newly-failing set, suppression, 24 h reminder, one recovery mail, and that a mail outage never raises. Heartbeat layer: `ping_heartbeat()` pings `/start` before the first step and `/<exit_code>` after the last; `TestHeartbeat` asserts `['…/start','…/0']` on a healthy tick and `['…/start','…/1']` on a failing one, that a ping exception never fails the tick, and that an unset URL skips pinging entirely |
| SC4 | No API key or password appears in any log line, notification or error message the unattended path produces | ✓ VERIFIED | `TestCredentialHygiene` (8 tests) seeds a fake LCO api_key, a fake `EMAIL_HOST_PASSWORD` and a fake heartbeat URL, forces failure on every step plus the mail send plus the heartbeat ping, and asserts none of the three literals reaches the captured log, stdout, stderr, `mail.outbox` subject/body, or `WatchedProposal.last_run_summary`. Independent source audit: every `except` in the unattended path logs `type(exc).__name__` (unattended.py:135, 193, 203, 322, 474, 510) — see Anti-Patterns for the single documented exception |
| SC5 | An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source | ✓ VERIFIED (usability judgment is a human item) | `docs/runbooks/telescope_runs_calendar.rst:1410` `.. _unattended-operation:` — a single section covering what runs and when, fresh-host setup (7 numbered steps), adding a watched proposal, both failure signals, the "nothing has appeared" checklist, running by hand, and the exact locking guarantee. `check_unattended` prints the resolved cron line even when hard checks fail (verified live on this host) |

#### Plan 36-01 must-have truths (runner, heartbeat, email, crontab, shared mailer)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All-succeeding tick exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_healthy_tick_exits_zero`, `test_pings_start_then_exit_code`, `test_empty_database_tick_is_healthy` |
| 2 | Failing tick exits non-zero, pings `/start` then `/<exit-code>`, sends exactly one email to every staff user with an email | ✓ VERIFIED | `test_step_failure_sets_exit_code` (SystemExit code 1), `test_failing_tick_mails_staff_once` (outbox == 1, `to == [staff@example.org]`, excludes the staff user with no email and the non-staff user) |
| 3 | Second consecutive failing tick with the same failing set sends no second email; a tick that succeeds afterwards sends exactly one `FOMO unattended run recovered` | ✓ VERIFIED | `test_repeat_failure_is_suppressed` (outbox stays 1 over two invocations), `test_recovery_mails_once` (subject asserted, third healthy tick sends nothing). State crosses processes via the real JSON file at `<FOMO_STATE_DIR>/unattended-state.json` |
| 4 | A raising heartbeat ping never changes the exit code, and neither the URL nor the exception message reaches log/stdout/stderr/email | ✓ VERIFIED | `test_ping_failure_never_fails_the_tick` (`ConnectionError('https://hc.example/UUID-SECRET')`, no raise, `UUID-SECRET` absent from logs), `test_heartbeat_ping_failure_leaks_nothing` |
| 5 | With `FOMO_HEARTBEAT_URL` unset: one INFO line per tick, no HTTP call, exit code unaffected | ✓ VERIFIED | `test_unset_url_skips_pinging`; `ping_heartbeat()` unattended.py:128-131 |
| 6 | Failure subject is `FOMO unattended run failed: <step(s)>`; body carries each failed step's summary, the log path and admin/calendar links — no traceback, no request URL, no portal response text | ✓ VERIFIED | `_build_notification_body()` unattended.py:434-461; `test_failure_email_body_carries_no_secret_and_no_traceback` asserts the log path and step name present, `Traceback` absent, all three seeded secrets absent |
| 7 | A second invocation while the first holds `<FOMO_LOCK_DIR>/run_unattended.lock` runs no step and logs one skip line — never queues, never runs concurrently | ✓ VERIFIED | `test_contended_lock_skips_every_step` (real `fcntl.flock` held on a second fd; `reconcile_run` not called; "lock" on stderr) **plus** the verifier's own cross-process check with `/usr/bin/flock -n` |
| 8 | A tick with nothing to do anywhere exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_empty_database_tick_is_healthy`; confirmed live (discovery step: `0 watched proposals, nothing to discover`, exit 0) |
| 9 | Step sequence fixed and identical every tick regardless of failures; the banner lists them in the same order even when an earlier step failed | ✓ VERIFIED | `STEPS` is a single module-level tuple (unattended.py:346-351); `test_all_four_steps_run_in_order`, `test_step_failure_does_not_abort_the_tick` (asserts `['a','b']` ran and that `step a` precedes `step b` in the log) |
| 10 | (backstop) No unattended-path log/email/stdout/stderr write formats the message of an exception caught from a `requests` call, a facility/portal call, or `send_mail()` — only its class name | ✓ VERIFIED | Exhaustive source audit of every `except` in `unattended.py`/`notifications.py`/`run_unattended.py`: the only site formatting a message is unattended.py:168, scoped to FOMO's own `reconcile_run()` (D-17's second bucket) at DEBUG. `campaign_reconciler.py` imports no network library, so it cannot originate a credential-bearing third-party message. Held-out behavioral evidence: the 8 `TestCredentialHygiene` tests |

#### Plan 36-02 must-have truths (WatchedProposal + watched-list discovery)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | Staff can add a proposal code in the admin and toggle `is_active` from the changelist, no redeploy, no per-invocation argument | ✓ VERIFIED | Verifier spot-check (add POST → 302, changelist `list_editable` POST → `is_active=False`, re-toggle → `True`); `WatchedProposalAdmin` declares `list_display`/`list_editable`/`list_filter` on `is_active` (admin.py:463-474); `WatchedProposalAdminTests` (3 tests) |
| 12 | Bare `backfill_lco_observations` sweeps every active row, applying that row's `target_list_name` and `attributed_to` | ✓ VERIFIED | Command handle() lines 814-834; `TestWatchedListSweep`; notebook cell 16 executed output `Swept 2 watched proposal(s), failed: 0` with row B using its `backfill-demo-watchlist-B` override |
| 13 | `--proposal X` still sweeps exactly X with the existing flags, and X need not be a watched row | ✓ VERIFIED | Override branch lines 800-812 passes `created_after`/`created_before`/`target_list`/`user` and never consults `WatchedProposal`; 67 tests in `test_backfill_lco_observations` + `test_campaign_submission` pass |
| 14 | After a bare sweep every active row carries `last_run_at` and a `last_run_summary` of either the counter line or `failed: <ExceptionClassName>` | ✓ VERIFIED | Lines 845-852; notebook cell 18 executed output shows both rows' real timestamps and counter lines; cell 20 shows `A last_run_summary: failed: ConnectionError` |
| 15 | A portal error on one row is caught, recorded, counted, and the remaining rows are still swept; exit is non-zero only because a row failed | ✓ VERIFIED | Lines 835-843 + 856-860; notebook cell 20 executed output: `CommandError … 1 watched proposal(s) failed: BACKFILL-DEMO-2026A`, `Swept 2 …, failed: 1`, proposal B's record 900402 still created |
| 16 | Two rows can never share a `proposal_code` | ✓ VERIFIED | `unique=True` in model and migration 0022; `test_duplicate_code_rejected` |
| 17 | Zero active rows → one INFO line, nothing written, exit 0 | ✓ VERIFIED | Lines 815-821; `TestEmptyWatchedList`; observed live in the dry-run tick |
| 18 | `proposal_code` equality is exact-match, case-sensitive, no Unicode normalization; whitespace stripped on save | ✓ VERIFIED | `WatchedProposal.save()` models.py strips; `test_code_is_stripped_on_save`, `test_code_comparison_is_case_sensitive` |
| 19 | Active rows are swept in deterministic `proposal_code` ascending order | ✓ VERIFIED | `Meta.ordering = ['proposal_code']` (model + migration options); `watched_rows()` inherits it deliberately rather than restating it |

#### Plan 36-03 must-have truths (the remaining three steps, SCHED-10 suite)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 20 | One tick runs all four steps in the fixed order status refresh → projector sweep → discovery → reconcile, in one process, no `manage.py` subprocess | ✓ VERIFIED | `STEPS` tuple order; live tick output in that exact order; `grep` for `call_command|subprocess|Popen|os.system` in `unattended.py` + `run_unattended.py` returns nothing |
| 21 | Status refresh calls `update_all_observation_statuses()` on a fresh `LCOFacility()` and a fresh `SOARFacility()` — never shared — and a non-empty failure list is a step failure | ✓ VERIFIED | unattended.py:233-241; `test_calls_both_facilities_with_fresh_instances`, `test_non_empty_failure_list_is_a_step_failure`, `test_facility_exception_is_isolated_per_facility` |
| 22 | A status-refresh failure is logged as `observation_id` plus the exception class name; TOM's message half never reaches a log line, stdout/stderr, or the email | ✓ VERIFIED | `_refresh_one_facility()` discards `_message` at the unpack (unattended.py:197) before any string building; `test_failure_is_reported_by_class_name_not_message`, `test_status_refresh_portal_error_leaks_nothing` |
| 23 | A raising step never prevents later steps; every registry step is attempted every tick and each outcome recorded separately | ✓ VERIFIED | `run_tick()` per-step try/except (unattended.py:506-513); `test_step_failure_does_not_abort_the_tick`, `test_all_four_steps_run_in_order` |
| 24 | Projector sweep calls `project_queryset()` directly with the same observed-site hook, counts `unprojectable` as a failure, never `call_command()` | ✓ VERIFIED | unattended.py:263-283; `test_unprojectable_row_is_a_failure`, `test_site_lookup_hook_is_passed_on_a_real_run_and_omitted_on_dry_run`, `test_step_never_calls_the_management_command` |
| 25 | Discovery sweeps every active row through `sweep_proposal()` and fails only when at least one row failed; zero rows logs one INFO line and is healthy | ✓ VERIFIED | unattended.py:302-340; `test_sweeps_every_active_row`, `test_empty_list_is_healthy`, `test_one_failing_row_does_not_stop_the_others`, `test_failing_row_names_the_proposal_in_the_summary` |
| 26 | `unchanged` / `skipped` / `detach_declined` / `remint_declined` never make the tick non-zero and never trigger the email | ✓ VERIFIED | `test_expected_data_shape_outcomes_are_not_failures` (no SystemExit, empty outbox); failure is derived from typed counters, never from parsing stdout text |
| 27 | With a fake api key, mail password and heartbeat URL seeded and every failure path forced in turn, none appears in logs/stdout/stderr/`mail.outbox` | ✓ VERIFIED | The 8-test `TestCredentialHygiene` class, run green |

#### Plan 36-04 must-have truths (`check_unattended`, cron line, logrotate)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 28 | `check_unattended` reports every prerequisite — flock, lock dir, log dir, non-console email backend, staff recipient, heartbeat URL, active watched proposal — in one run | ✓ VERIFIED | Live run on this host printed all seven `[ok]/[FAIL]/[WARN]` lines in one pass; `handle()` lines 251-257 |
| 29 | A missing hard prerequisite makes the command exit non-zero and name which check failed | ✓ VERIFIED | Live run exited 1 naming `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`, `EMAIL_BACKEND`; `test_missing_flock_fails`, `test_unwritable_lock_dir_fails`, `test_unwritable_log_dir_fails`, `test_console_email_backend_fails`, `test_no_staff_email_fails` |
| 30 | Unset heartbeat URL and empty watched list are warnings, not failures; exit stays 0 when every hard check passed | ✓ VERIFIED | `hard=False` on both checks; `test_unset_heartbeat_is_a_warning`, `test_empty_watched_list_is_a_warning`, `test_warnings_do_not_mask_a_hard_failure`, `test_all_hard_checks_passing_exits_zero` |
| 31 | The printed cron line carries the real resolved interpreter and `manage.py` path, not the template's placeholders | ✓ VERIFIED | Live output: `… /home/tlister/venv/devel_fomo311_venv/bin/python /home/tlister/git/fomo_devel/manage.py run_unattended …`; `test_line_has_real_paths` also asserts both placeholders absent |
| 32 | Names and set/unset status only — never a value (api key, mail password, heartbeat URL) | ✓ VERIFIED | `check_heartbeat()` prints `FOMO_HEARTBEAT_URL: set`/`unset`; `test_line_carries_no_setting_value`, `test_output_never_contains_a_seeded_value` (asserts across passing and hard-failing configurations, including the `CommandError` text) |
| 33 | `--send-test-email` sends one message through the configured backend to the same staff recipient list the failure notice uses | ✓ VERIFIED | `_send_test_email()` → `notifications.notify_staff()` — literally the same helper; `test_send_test_email_sends_one_message`, `test_send_test_email_without_recipients_fails`, `test_flag_absent_sends_nothing` |
| 34 | The command only reads: creates no directory, writes no file, changes no row | ✓ VERIFIED | `test_command_writes_nothing` (non-existent lock/log paths still absent after the run, `WatchedProposal.objects.count() == 0`); no `mkdir`/`open(…, 'w')` in the module; confirmed live — `/var/lock/fomo` still absent after my run |

#### Plan 36-05 must-have truths (paired docs)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 35 | An operator can set up or verify the whole schedule from one runbook section without reading source | ✓ VERIFIED (usability is a human item) | `.. _unattended-operation:` section: "What runs, and when", "Setting it up on a fresh host" (7 steps incl. the two directories, local_settings placement, the env vars, `check_unattended`, the crontab, logrotate), "Adding a proposal to watch", "The two failure signals", "When nothing has appeared", "Running it by hand" |
| 36 | The section documents both failure signals — email (who, what, repeat, clear) and heartbeat (`/start`, `/<exit-code>`, expected ping interval (Period) plus grace time (Grace), alerting at last ping + Period + Grace) — plus the "nothing has appeared" checklist | ✓ VERIFIED | Runbook "The two failure signals" covers recipients, subject form, contents, 24 h reminder, recovery mail, and recommends a 15-min Period with a ~20-min Grace (late ~15 min, alert ~35 min after the last ping) — corrected post-G-36-3; "When nothing has appeared" gives the D-18 four-step checklist in order |
| 37 | The backfill section documents `--proposal` as optional, the bare watched sweep, per-row overrides and per-row failure isolation | ✓ VERIFIED | Runbook diff lines 9-23 state all four points explicitly, including class-name-only failure recording |
| 38 | The cheat-sheet carries rows for `run_unattended` and `check_unattended`, and `backfill_lco_observations` reflects its optional-argument contract | ✓ VERIFIED | Cheat-sheet rows added for all three (runbook diff lines 218-235) |
| 39 | The overlap guarantee is stated at exactly its true strength (two ticks never overlap, incl. a hand-started `run_unattended --step`; a directly-run sweep command is not locked against a tick in this release) | ✓ VERIFIED | "What the locking does and does not cover" paragraph says precisely this and names `run_unattended --step <name>` as the exclusive manual route — matching the code, where only `unattended.py` takes the named locks |
| 40 | The notebook contains executed cells seeding `WatchedProposal` rows, running the command bare, showing `last_run_summary`, and showing one proposal failing without stopping the other — real committed output, mocked portal | ✓ VERIFIED | Cells 16/18/20 carry real stdout (`Swept 2 watched proposal(s), failed: 0`; both rows' timestamps and counter lines; `failed: ConnectionError` with B still sweeping). All 11 code cells carry sequential `execution_count` 1-11 |
| 41 | `docs/notebooks.rst` lists `backfill_lco_observations_demo` in the Demonstration Notebooks toctree | ✓ VERIFIED | One added toctree line; the referenced file exists |
| 42 | `CLAUDE.md`'s notebook map records the pairing for every module this phase adds, including that the runbook section — not a notebook — is the paired doc for the runner and `check_unattended`, and why | ✓ VERIFIED | CLAUDE.md diff maps `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py` → the runbook section with the stated reason, and extends the `backfill_lco_observations.py` entry to cover the watched-proposal contract |
| 43 | No committed doc, notebook cell or runbook example quotes a live setting, a `local_settings.py` value, or any credential; examples use placeholder codes and paths | ✓ VERIFIED | Notebook uses `BACKFILL-DEMO-2026A`/`-B`; crontab template uses `/path/to/venv/bin/python` and `/path/to/checkout/manage.py`; logrotate uses `/var/log/fomo/unattended.log` (the committed default, not a host-specific path); runbook quotes env var NAMES only |

**Score:** 48/48 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/unattended.py` | Runner, steps, lock, heartbeat, notification | ✓ VERIFIED | 534 lines; imported by `run_unattended.py`; executed live |
| `solsys_code/notifications.py` | Shared request-free mailer | ✓ VERIFIED | 85 lines; imported by `unattended.py`, `check_unattended.py`, `campaign_views.py` |
| `solsys_code/management/commands/run_unattended.py` | Cron entry point | ✓ VERIFIED | 59 lines; thin wrapper, `sys.exit(result.exit_code)`; executed live |
| `solsys_code/management/commands/check_unattended.py` | Preflight + cron line | ✓ VERIFIED | 285 lines; executed live, exit 1 with correct diagnosis |
| `solsys_code/models.py` (`WatchedProposal`) | Admin-editable watch list | ✓ VERIFIED | Model + `save()` strip + `Meta.ordering`; queried by `watched_rows()` and `check_watched_proposals()` |
| `solsys_code/migrations/0022_watchedproposal.py` | Matching migration | ✓ VERIFIED | `makemigrations --check --dry-run` → "No changes detected" |
| `solsys_code/admin.py` (`WatchedProposalAdmin`) | Registered, `list_editable` | ✓ VERIFIED | Registered on `admin.site`; changelist drives real state changes (spot-check) |
| `solsys_code/management/commands/backfill_lco_observations.py` | `sweep_proposal()`, `watched_rows()`, optional `--proposal` | ✓ VERIFIED | Both functions imported by `unattended.py`; 67 regression tests green |
| `deploy/cron/fomo.crontab.example` | `*/15`, `flock -n`, redirect, skip tail, no secret | ✓ VERIFIED | Line 27 carries all five elements; comments name env vars only |
| `deploy/logrotate/fomo.example` | Daily, rotate 14, copytruncate | ✓ VERIFIED | Path matches the crontab redirect and `FOMO_LOG_FILE`'s default |
| `src/fomo/settings.py` | `FOMO_BASE_URL`/`HEARTBEAT_URL`/`LOCK_DIR`/`STATE_DIR`/`LOG_FILE` | ✓ VERIFIED | All five `os.getenv()`-sourced; read by runner, notifications and preflight |
| `docs/runbooks/telescope_runs_calendar.rst` | Unattended section + cheat-sheet + troubleshooting | ✓ VERIFIED | +269 lines; `_unattended-operation` label referenced from `docs/installation.rst` |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | Executed watched-proposal cells | ✓ VERIFIED | +403 lines; cells 16/18/20/22 with real output; cleanup cell proves zero residue |
| `docs/notebooks.rst`, `docs/installation.rst`, `CLAUDE.md` | Toctree, cross-ref, paired-docs map | ✓ VERIFIED | All three updated |
| Tests (`test_unattended`, `test_check_unattended`, `test_watched_proposal`, `test_admin`, `test_backfill_lco_observations`, `test_campaign_submission`) | Behavioral coverage | ✓ VERIFIED | 122 + 67 = 189 tests, all green |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `run_unattended.Command.handle()` | `unattended.run_tick()` → `STEPS` | direct call | ✓ WIRED | Single `STEPS` declaration (unattended.py:346); `--step` choices derive from it |
| `unattended.run_tick()` | `notifications.notify_staff()` | `_send_notification()` | ✓ WIRED | Same helper `campaign_views.CampaignRunSubmissionView._notify_staff()` now calls — the duplicated `send_mail`/recipient query was deleted from `campaign_views.py` |
| crontab template's `flock -n` path | `settings.FOMO_LOCK_DIR` | `/var/lock/fomo/run_unattended.lock` | ✓ WIRED | Default `FOMO_LOCK_DIR='/var/lock/fomo'`; `command_lock('run_unattended')` opens `<dir>/run_unattended.lock`. **Proven empirically**: `/usr/bin/flock -n` on that path blocked a real runner invocation |
| `unattended.run_tick()` | `<FOMO_STATE_DIR>/unattended-state.json` | `load_state()`/`save_state()` | ✓ WIRED | Real JSON file; `test_reminder_after_interval` writes it directly and the next tick reads it |
| `WatchedProposal.objects.filter(is_active=True)` | `sweep_proposal()` | `watched_rows()` | ✓ WIRED | One call per row from both the command's bare path and `step_discovery()` |
| `sweep_proposal()` summary | `WatchedProposal.last_run_summary` → admin column | `row.save(update_fields=…)` | ✓ WIRED | `list_display` includes `last_run_summary`; notebook shows real values |
| `check_unattended.cron_line()` | `deploy/cron/fomo.crontab.example` | shared seven elements | ✓ WIRED | `test_line_matches_the_committed_template_shape`; live output matches the template line element-for-element |
| `check_unattended` lock/log checks | `settings.FOMO_LOCK_DIR` / `FOMO_LOG_FILE` | same settings the runner uses | ✓ WIRED | Both read the identical settings names |
| `check_unattended --send-test-email` | `notifications.notify_staff()` | direct call | ✓ WIRED | Same recipient rule as the failure notice |
| `step_status_refresh()` | Phase 34 `post_save` receiver | `facility.update_observation_status()` → `.save()` | ✓ WIRED | Receiver connected in `observation_projector`; the sweep step's test disconnects/reconnects it deliberately, proving it is live |
| `step_project_sweep()` | `resolve_observed_site()` | `pre_fields_hook` | ✓ WIRED | Hook passed on a real run, omitted under `--dry-run` (asserted by test) |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `step_project_sweep()` | `result['counters']` | `project_queryset()` over `ObservationRecord.objects.filter(facility__in=PROJECTED_FACILITIES)` | Yes — live tick reported `unchanged: 159` from the dev DB | ✓ FLOWING |
| `step_reconcile()` | `run_count`/`failed_count` | `CampaignRun.objects.all()` | Yes — live tick reported `runs: 45, failed: 0` | ✓ FLOWING |
| `step_discovery()` | `rows` | `watched_rows()` → `WatchedProposal` query | Yes — 0 rows live (correct), 2 rows in the notebook with real portal-shaped payloads | ✓ FLOWING |
| `WatchedProposalAdmin` changelist | `last_run_at`/`last_run_summary` | sweep-written model fields | Yes — notebook cells 18/20 show real written values | ✓ FLOWING |
| `check_unattended` report | `results` | live `shutil.which`, `os.access`, settings, `WatchedProposal` count | Yes — live run produced host-accurate verdicts | ✓ FLOWING |
| Failure email body | `settings.FOMO_LOG_FILE`, `FOMO_BASE_URL` | settings, not hardcoded literals | Yes — asserted present in the body by test | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full tick runs all four steps in D-01 order against the real dev DB | `FOMO_LOCK_DIR=… FOMO_STATE_DIR=… python manage.py run_unattended --dry-run` | `START` → status_refresh ok → project_sweep ok (159 unchanged) → discovery ok (0 watched) → reconcile ok (45 runs) → `END exit=0` | ✓ PASS |
| Cross-process overlap guard (cron's own `flock -n` vs the runner's `fcntl.flock`) | `/usr/bin/flock -n <lock> -c "… run_unattended --dry-run"` | `run_unattended: lock held -- skipping this tick`, no step ran, exit 0 | ✓ PASS |
| Preflight on a host missing prerequisites | `python manage.py check_unattended` | exit 1; `[FAIL]` on FOMO_LOCK_DIR / FOMO_LOG_FILE / EMAIL_BACKEND, `[WARN]` on heartbeat + watched list, `[ok]` on flock + staff recipients, cron line still printed with real paths | ✓ PASS |
| Admin add + changelist toggle actually changes discovery scope | Django test-client POSTs to the `WatchedProposal` add form and the `list_editable` changelist, then `watched_rows()` | add → `is_active=True`; toggle → `False` and `watched_rows() == []`; re-toggle → `True` and `['SPOT2026A-001']` | ✓ PASS |
| Phase test modules | `python manage.py test solsys_code.tests.test_unattended test_check_unattended test_watched_proposal test_admin` | Ran 122 tests — OK | ✓ PASS |
| Regression modules | `python manage.py test solsys_code.tests.test_backfill_lco_observations test_campaign_submission` | Ran 67 tests — OK | ✓ PASS |
| Migration/model consistency | `python manage.py makemigrations --check --dry-run` | "No changes detected", exit 0 | ✓ PASS |
| Lint gate | `ruff check --no-fix solsys_code/ src/fomo/settings.py` (ruff 0.2.1, the pinned version) | no findings, exit 0 | ✓ PASS |
| Format gate | `ruff format --check` on the five new/changed modules | "5 files already formatted" | ✓ PASS |
| Real SMTP delivery, real crontab, external heartbeat alerting | — | requires the production host / external service | ? SKIP → human verification |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and neither the PLANs nor the SUMMARYs declare one; this project's verification contract is the Django test runner (36-VALIDATION.md) | n/a — SKIPPED |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCHED-08 | 36-01, 36-03, 36-04, 36-05 | Projector sweep, discovery backfill and reconciler on a documented cron + `flock -n` schedule with no operator action, guarded against overlapping invocations | ✓ SATISFIED | `STEPS` + `run_unattended` + committed crontab template + `check_unattended`'s printed line; overlap proven cross-process. Host installation is a human item |
| SCHED-09 | 36-01, 36-03, 36-05 | Failure visible through two independent layers — in-command notification and a heartbeat/dead-man's switch | ✓ SATISFIED | `notifications.notify_staff()` (reusing the former `_notify_staff()` idiom exactly as the requirement asks) + `ping_heartbeat()` `/start` / `/<exit-code>`; 5 + 3 tests. External alerting configuration is a human item |
| SCHED-10 | 36-01, 36-03, 36-04, 36-05 | No credential value in any log line or notification the unattended path generates | ✓ SATISFIED | Class-name-only discipline at every `except` site (source-audited), 8 `TestCredentialHygiene` tests, `check_unattended`'s names-only output plus its own 1 leak test, no committed artifact carrying a value |
| DISCOVER-01 | 36-02, 36-03, 36-05 | Admin-editable watched-proposal list replaces per-invocation `--proposal`/name-prefix arguments | ✓ SATISFIED | `WatchedProposal` model/migration/admin; `--proposal` now `required=False`; bare sweep over `watched_rows()`; admin toggle proven to change discovery scope end-to-end |

**Orphaned requirements:** none. `.planning/REQUIREMENTS.md` maps exactly SCHED-08, SCHED-09, SCHED-10 and DISCOVER-01 to Phase 36, and all four are claimed by plan frontmatter.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/unattended.py` | 168 | `logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)` — the one site in the unattended path that formats an exception's message | ℹ️ Info | Deliberate and in scope of D-17's second bucket (FOMO's own call); `campaign_reconciler.py` imports no network library so it cannot originate a credential-bearing third-party message, and DEBUG sits below the project's root INFO level. Residual: a deployment that lowers the root level to DEBUG *and* somehow routes a third-party exception through `reconcile_run()` would emit its message. Not a SCHED-10 breach as the requirement and D-17 are written |
| `solsys_code/unattended.py` | 518-526 | A failing tick with **no** staff recipient on file still calls `save_state(...)`, so suppression starts even though `notify_staff()` sent nothing | ℹ️ Info | Guarded upstream: `check_unattended` treats "no staff user with an email" as a *hard* failure, so a correctly preflighted host cannot reach this state. Worth knowing if someone later removes every staff email |
| phase-added lines (all files) | — | `TBD` / `FIXME` / `XXX` / `TODO` / `HACK` debt markers | — none | `git diff <base>..HEAD` added lines contain no debt marker. Pre-existing `TBD` hits elsewhere in `models.py`/`campaign_views.py`/the runbook are the campaign domain's own "TBD window" vocabulary, not markers |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | cells 2, 6 | Two code cells carry `execution_count` but an empty `outputs` list | ℹ️ Info | Pre-existing structural fact (the Django-setup boilerplate and the mocking-helper import print nothing). Both executed — sequential `execution_count` 1-11 across all 11 code cells confirms a genuine full re-execution. Matches the SUMMARY's own disclosure |

No 🛑 Blockers and no ⚠️ Warnings found.

---

### CLAUDE.md Paired-Docs Compliance

The rule requires that a plan changing a mapped module's behavior carry its paired notebook and any affected `docs/runbooks/` page. Checked and satisfied:

- `backfill_lco_observations.py` changed behavior (optional `--proposal`, watched-list sweep) → `backfill_lco_observations_demo.ipynb` updated with three new executed cells **and** the runbook's backfill section updated. Both were scoped into plan 36-05's `files_modified` up front.
- `docs/runbooks/telescope_runs_calendar.rst` — the only page under `docs/runbooks/` — gained the new unattended section plus three troubleshooting entries and three cheat-sheet rows.
- The four newly-added modules (`unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py`) were added to CLAUDE.md's map, explicitly paired to the runbook section rather than a notebook, with the reason recorded — closing the same enforcement hole Phase 35 was cited for.

---

### Human Verification Required

#### 1. Fresh-host preflight

**Test:** On the real FOMO host, create `/var/lock/fomo` and `/var/log/fomo` writable by the cron account, put the real `EMAIL_BACKEND`/`EMAIL_HOST_*` and the LCO/SOAR `api_key` in `local_settings.py`, export `FOMO_HEARTBEAT_URL` and `FOMO_BASE_URL` for the cron daemon, then run `python manage.py check_unattended`.
**Expected:** Every hard check reports `[ok]`, the command exits 0, and it prints a cron line with this host's real interpreter and `manage.py` paths.
**Why human:** Host filesystem, real SMTP settings and the cron daemon's environment do not exist on the dev machine — there, the command correctly exits 1 naming `FOMO_LOCK_DIR`/`FOMO_LOG_FILE`/`EMAIL_BACKEND`.

#### 2. The real crontab

**Test:** Install the printed cron line in the FOMO service account's crontab (`crontab -e`) and leave it for ~45 minutes.
**Expected:** Three `START`/per-step/`END` banners appear in `/var/log/fomo/unattended.log`, roughly 15 minutes apart, with nobody typing anything.
**Why human:** SC 1's "on the real host" half needs a real crontab on the production host; only the mechanism, the committed template and the printed line can be verified here.

#### 3. The heartbeat's dead-man half

**Test:** Create a healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL`, configuring only what the corrected runbook paragraph names: an expected ping interval (Period) of 15 minutes and a grace time (Grace) of about 20 minutes. Then disable the crontab line (simulating the scheduler never invoking the job) and watch the check.
**Expected:** The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping (last ping + expected interval + grace), even though FOMO itself logged nothing and sent no email — the second, independent layer of SC 3.
**Why human:** That signal is produced by the external heartbeat service's own expected-interval-plus-grace timer, not by any FOMO code path.

**Note (post-G-36-3 correction):** This test script and the plan 36-05 evidence row above (table row 36) were both corrected after gap `G-36-3` found that the original wording named only the check's grace time, leaving its expected ping interval at the vendor default and disabling the alert for about a day. See `36-UAT.md` and `.planning/debug/heartbeat-runbook-period-gap.md` for the original wording and full diagnosis.

#### 4. Real mail delivery

**Test:** With the real (non-console) email backend configured, run `python manage.py check_unattended --send-test-email`.
**Expected:** One message arrives in the mailbox of every staff user with an email on file — the same recipient rule the failure notice uses.
**Why human:** Real SMTP delivery cannot be exercised from the dev host, whose `EMAIL_BACKEND` is the console backend.

#### 5. Log rotation under a live writer

**Test:** Drop `deploy/logrotate/fomo.example` into `/etc/logrotate.d/fomo`, run `logrotate -d` against it, then force one real rotation while a tick is running.
**Expected:** The stanza parses, and `copytruncate` keeps cron's still-open append redirect writing to the live file rather than the rotated-away inode.
**Why human:** Requires root, a real logrotate installation, and a concurrently running tick holding the log file descriptor open.

#### 6. Runbook sufficiency

**Test:** Hand the runbook's "How do I run everything unattended?" section to someone who has not read this phase's source and ask them to set the schedule up on a fresh host.
**Expected:** They get to a working, checked schedule using only that one section — no source reading, no questions back.
**Why human:** SC 5 is a documentation-usability judgment; completeness can be checked mechanically, sufficiency for a real operator cannot.

---

### Gaps Summary

No gaps. Every must-have resolved to VERIFIED against the codebase, not against SUMMARY narrative: the runner was executed end-to-end against the real developer database, the overlap guarantee was proven cross-process using the very `flock(1)` binary the crontab line invokes, the admin toggle was driven through a real HTTP POST and shown to change what `watched_rows()` returns, `check_unattended` was run live and failed exactly the checks this host should fail while still printing a correct cron line, and the SCHED-10 discipline was confirmed by auditing every `except` clause in the unattended path rather than by trusting the credential-hygiene test names.

The phase is held at `human_needed` solely because six items are genuinely host- or external-service-dependent: the production crontab, the external heartbeat's dead-man alert, real SMTP delivery, logrotate under a live writer, the fresh-host preflight passing all-green, and the human judgment of whether the runbook section is sufficient for an operator. Per this phase's own scope note, none of these is a code gap — the mechanism, the committed templates and the preflight that verifies them are all present, wired and behaviorally proven.

---

_Verified: 2026-09-17T17:20:00Z_
_Verifier: Claude (gsd-verifier)_
