---
phase: 36-unattended-operation
verified: 2026-09-18T05:55:00Z
status: gaps_found
score: 61/65 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/debug/heartbeat-runbook-period-gap.md"
  - ".planning/debug/heartbeat-setup-step-context-gap.md"
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
  - ".planning/phases/36-unattended-operation/36-07-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-07-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-REVIEW.md"
  - ".planning/phases/36-unattended-operation/36-UAT.md"
  - "CLAUDE.md"
  - "deploy/cron/fomo.crontab.example"
  - "deploy/logrotate/fomo.example"
  - "docs/conf.py"
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
covered_digest: "v1:sha256:da180c2c717898d835244627e8280f653b0327d1d77297e61edf3bbb086fd032"
behavior_unverified: 1
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 54/56
  previous_verified: 2026-09-18T02:05:00Z
  gaps_closed:
    - "G-36-3 (UAT round 1, Test 3): the runbook's heartbeat guidance named only the check's grace time, never its expected ping interval, so a check configured from it first alerted about a day after the schedule stopped. The runbook (4 sites + a new troubleshooting entry), deploy/cron/fomo.crontab.example, two unattended.py docstrings, check_unattended's [ok] heartbeat line and this verification record now all state the two-knob configuration and the last ping + expected interval + grace arithmetic."
    - "G-36-1 (UAT round 2, Test 1) -- STRUCTURAL HALF CLOSED, plan 36-07 (commits 280962b, a2f1ee9, 71cdec2). The fresh-host procedure no longer asks the operator to export a URL that only exists after a check has been created: it now runs to 9 numbered steps with a create-and-configure-the-heartbeat-check step (runbook:1463-1493) standing BEFORE the FOMO_HEARTBEAT_URL export (:1494), naming the service class (healthchecks.io hosted free tier or a self-hosted healthchecks instance), both check settings with their values, one line of alert arithmetic plus a pointer to the canonical 'The two failure signals' paragraph, and the ping URL's placeholder form ``https://hc-ping.com/<uuid>`` with the warning that the <uuid> IS the credential. Re-proven by this verification, not read from the SUMMARY: the awk-sliced subsection carries Period/Grace/*/15 * * * *,/hc-ping.com/<uuid>/healthchecks.io/'The two failure signals'/'35 min'/a 9th numbered step, and the POSITION gate holds inside the slice (first Period line 31 and ping-URL line 43 both precede the export anchor 'the environment the cron daemon sees' at line 47). deploy/cron/fomo.crontab.example names the subsection at all three pointers (:16, :37, :62). No UUID-shaped string in either file. Zero files under solsys_code/ changed by the three commits. The SUFFICIENCY half stays open as human item 1 -- and see gaps_remaining below: plan 36-07's second, unrelated Task-2 edit introduced a new SC-5 defect in the same subsection."
  gaps_remaining:
    - "G-36-4 (NEW, introduced by this gap-closure round, commit a2f1ee9): runbook step 2 tells the operator to configure the LCO/SOAR API key at FACILITIES['LCO']['api_key'] / FACILITIES['SOAR']['api_key'] in local_settings.py. Following it literally raises NameError at Django startup -- reproduced by this verification. Breaks SC 5 and plan 36-07 truth 6."
  regressions:
    - "docs/runbooks/telescope_runs_calendar.rst:1458-1462 -- the FACILITIES key-path sentence did not exist before a2f1ee9 (git log -S confirms a2f1ee9 is the only commit that introduced it). It is a regression of this round, not a pre-existing finding, so the convergence evidence gate does not apply -- and it is deterministically evidenced regardless."
  human_items_closed_by_uat:
    - "Test 1 fresh-host preflight -- UAT round 1 pass (all hard checks [ok], exit 0, cron line printed)"
    - "Test 2 the real crontab -- UAT round 1 pass (three START/END banners ~15 min apart on the real host)"
    - "Test 4 real mail delivery -- UAT round 1 pass"
    - "Test 5 logrotate under a live writer -- UAT round 1 pass (START went with the rotated copy; END landed in the truncated live file)"
    - "WR-22 ship decision -- UAT round 2 Test 2 pass, option (b), the recorded acceptance (see ## Acknowledged Gaps)"
  human_items_still_open:
    - "SC-5 sufficiency read-through: RE-OPENED by G-36-1 (formerly Test 6, once recorded as closed) -- its round-1 pass came from a reader already taught the Period/Grace values by Test 3 in the same session, so it was not a sufficiency measurement; must be re-run from the fresh-host setup steps alone, BEFORE the live heartbeat re-run below. RELEASE CONDITION: gap G-36-4 (step 2 named a settings path local_settings.py cannot assign) was closed by plan 36-08, which rewrote step 2 to the flat LCO_API_KEY assignment and extended the src/fomo/settings.py fold to also cover the SOAR facility entry; administer this read-through once re-verification confirms G-36-4 closed."
    - "Live heartbeat dead-man re-run (formerly Test 3): configure a live healthchecks-compatible check from the fresh-host setup steps alone and confirm late ~15 min / alert ~35 min; run AFTER the SC-5 sufficiency read-through above, since it teaches the values that read-through measures"
    - "CR-03 credential-in-docs-build decision: decide whether the docs-build render of local_settings.py is fixed (autoapi_ignore) or accepted before shipping -- see Human Verification item 3"
gaps:
  - truth: "An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source (ROADMAP SC 5; plan 36-07 truth 6)"
    status: failed
    reason: "Step 2 of 'Setting it up on a fresh host' -- the one section SC 5 names -- instructs the operator to put the API key at ``FACILITIES['LCO']['api_key']`` / ``FACILITIES['SOAR']['api_key']`` in ``local_settings.py``. That module is imported into its own namespace (src/fomo/settings.py:431-434), so the assignment raises NameError, which the ``except ImportError`` guard does not catch. Reproduced in isolation by this verification against the same module shape and the same guard. The settings module then fails to import, so gunicorn/uWSGI, every ``manage.py`` command, the ``run_unattended`` tick and ``check_unattended`` itself all die before running -- and the failure email that would report it cannot be sent either. The supported name is the flat ``LCO_API_KEY``, documented ONLY in src/fomo/settings.py:436-441's own comment and nowhere in docs/ (`grep -rn \"api_key\\|API key\" docs/*.rst` finds this sentence and nothing else), so the operator must read source to recover -- the exact thing SC 5 forbids. The SOAR half is worse: no ``SOAR_API_KEY`` fold exists anywhere in the repo, so ``FACILITIES['SOAR']['api_key']`` stays `''` whatever the operator writes, and they will believe SOAR is authenticated while status_refresh calls the portal with an empty key."
    artifacts:
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: ":1458-1462 -- step 2's API-key sentence names a settings path that cannot be assigned from local_settings.py; introduced by a2f1ee9 (plan 36-07 Task 2c)"
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: ":1461-1462 -- the ``FACILITIES['SOAR']['api_key']`` half documents a route with no supported mechanism at all (no SOAR_API_KEY fold in src/fomo/settings.py:440-441)"
    missing:
      - "Replace the nested key path with the flat name the fold actually reads: ``LCO_API_KEY = '...'`` in local_settings.py, plus one clause saying why it is flat (local_settings.py is imported into its own namespace, so assigning into FACILITIES[...] raises NameError and stops Django from starting) and that settings.py folds it into FACILITIES['LCO']['api_key']."
      - "Resolve the SOAR half against reality: either document that only LCO has a fold today and FACILITIES['SOAR']['api_key'] is a currently-unfilled slot (SOAR authenticates against the same LCO portal, settings.py:240-247), or extend the fold in src/fomo/settings.py and then document the flat name. The runbook must match whichever is chosen."
      - "A gate with teeth for this class: the setup slice must not name a settings path that local_settings.py cannot assign. The cheapest form is asserting the slice contains ``LCO_API_KEY`` and does NOT contain ``FACILITIES['`` -- the same slice-scoped shape plan 36-07 already built for the heartbeat half."
deferred: []
advisory:
  - finding: "CR-03 (36-REVIEW.md iter 4): docs/conf.py:62-63 sets ``autoapi_dirs = ['../src']`` with an ``autoapi_ignore`` that does not exclude ``local_settings.py``, so sphinx-autoapi and sphinx.ext.viewcode render that file -- the credential home runbook step 2 mandates -- verbatim into generated HTML. Confirmed present in this working tree by this verification: ``_readthedocs/html/autoapi/fomo/local_settings/index.html`` and ``_readthedocs/html/_modules/fomo/local_settings.html`` each contain one UUID-shaped string (this host's live ping token) and the names LCO_API_KEY / EMAIL_HOST_USER / EMAIL_HOST_PASSWORD / FOMO_HEARTBEAT_URL; the same three pages exist under ``docs/_build/html/``. Both trees are .gitignore'd (.gitignore:76-77), so nothing reached git, and ReadTheDocs builds from a checkout with no local_settings.py, so the published site is unaffected. The exposure is on-disk and on anything that serves a build produced from a configured host."
    category: security
    reason: "Does NOT falsify SC 4 as written ('no API key or password appears in any log line, notification or error message the unattended path PRODUCES') -- a docs-build artifact is none of those three, and the unattended path produces none of it. The root cause file (docs/conf.py) was never touched by phase 36; its last three commits predate this milestone. Recorded here rather than as a gap because it is real, reproducible and worth a decision before shipping: the fix is one line (add '*/local_settings.py' to autoapi_ignore), plus deleting the two build trees and rotating the check if either build was ever served or copied, plus correcting the runbook's 'never go into a committed file' boundary claim at :1491-1493, which is scoped to the wrong boundary."
    evidence_status: "reproduced: rendered pages located on disk; UUID-shaped string and credential setting names counted in them without quoting any value"
  - finding: "WR-24 (36-REVIEW.md iter 4): the new step 3 says 'Create ONE check for this schedule and set both of its settings', then offers the Cron-type ``*/15 * * * *`` alternative inside the same sentence and states 'The service alerts at last ping + expected interval + grace'. The canonical paragraph this step defers to says the Cron route 'pegs lateness to the wall-clock slot instead of to the last ping' (runbook:1589-1591), and a Cron-type check has no Period field -- so on the route step 3 recommends as drift-free, both 'both of its settings' and the stated baseline are wrong."
    category: other
    reason: "Plan 36-07 truth 4 explicitly permits ONE line of arithmetic in the new step (the prohibition was on re-copying the /start-to-completion reasoning and the 1-day-default trap, both of which are correctly absent from the slice), so this is not a must-have breach. But it is the first drift cost of the restatement, and the operator who takes the Cron route will mis-predict the alert baseline. Resolved by attaching the formula to the Simple-check route and describing the Cron route as 'missed slot + grace, no interval to set'."
    evidence_status: "source-confirmed: both passages read directly; the two statements contradict each other within the same file"
  - finding: "WR-25 (36-REVIEW.md iter 4): step 3's opening relative clause (runbook:1463-1466) attaches to 'one failed outright', asserting that outright failure is the class FOMO's own error handling cannot report. The code says the reverse -- run_tick() reports an outright failure via _send_notification() and the /<exit-code> ping; the class FOMO cannot report is the tick that never started or hung."
    category: other
    reason: "Cosmetically a misplaced modifier, substantively an inversion of the whole justification for creating the check, in the step whose job is to convince the operator to create it. The operator's action is unaffected (they still create the check), so it does not falsify a must-have."
    evidence_status: "source-confirmed"
  - finding: "WR-26 (36-REVIEW.md iter 4, re-raising iter 3's WR-21): check_state_dir() is a THIRD hard prerequisite (check_unattended.py:177), but step 1 still says 'Create the two directories' and step 6's enumeration lists neither FOMO_STATE_DIR nor check_flock()'s -E/util-linux 2.27 probe, still summarising the hard set as '(flock, the directories, or email)'. ``grep -rn FOMO_STATE_DIR docs/`` returns nothing. Plan 36-07 edited step 1's first sentence and renumbered step 4 to step 6, so both passages were in hand and both were left stale."
    category: other
    reason: "Not a must-have breach: FOMO_STATE_DIR defaults to FOMO_LOCK_DIR (settings.py:423-425), so a fresh host following the runbook passes. A host that points it elsewhere gets a hard preflight failure naming a directory the runbook never mentions. Carried forward open from iteration 3."
    evidence_status: "source-confirmed"
  - finding: "WR-27 (36-REVIEW.md iter 4): runbook step 8 (:1542-1545) says starting from the template and replacing 'its two placeholder paths' means 'either route produces the same line', and the template header says the same. cron_line() (check_unattended.py:309-321) interpolates FIVE resolved values -- sys.executable, manage.py, ``shutil.which('flock')``, FOMO_LOCK_DIR and FOMO_LOG_FILE -- and its own comment calls the template's hardcoded '/usr/bin/flock' a placeholder. On a host whose flock is elsewhere the template route installs a line whose lookup fails with exit 127 (not 99, so no 'lock held' line is written either) and every tick is a silent no-op."
    category: other
    reason: "The printed line -- which the runbook makes the primary route -- is correct, so the documented happy path works; the claim that the hand-edit route is equivalent is what is false. Pre-existing from 36-04/36-05, re-raised because plan 36-07 renumbered the step. Not a must-have breach."
    evidence_status: "source-confirmed: cron_line() body read directly"
  - finding: "IN-25 / IN-26 / IN-27 / IN-28 (36-REVIEW.md iter 4): the 15/20/35 triple now appears in five places across the two files; 'Name each concept first, giving healthchecks.io's spelling in parentheses' (runbook:1476-1477) and the pre-existing 'Name the concept first:' (:1584) are drafting directives surviving into operator prose; deploy/cron/fomo.crontab.example still says '[ $? -eq 99 ]' at :32-33 while the line itself uses '[ $rc -eq 99 ]', and still calls check_unattended and deploy/logrotate/fomo.example '(a later plan in this phase)' though both shipped; runbook :1543 and :1546 mark the two deploy paths with single backticks, which Sphinx renders as italic title references, unlike every other path in the subsection."
    category: other
    reason: "Copy-edit class; none falsifies a success criterion or a plan must-have truth. Recorded so the ship decision sees them."
    evidence_status: "source-confirmed"
  - finding: "WR-22 (36-REVIEW.md iter 3): two logger.debug() sites on the unattended path interpolate a raw exception message rather than its class name -- backfill_lco_observations.py:349 (a live, authenticated LCO portal call, reached from step_discovery -> sweep_watched_rows -> sweep_proposal -> _resolve_schedule) and unattended.py:191 (a bare `except Exception` around FOMO's own reconcile_run()). Inert under the shipped configuration because settings.LOGGING pins the root logger to INFO, so no such line is emitted; live the moment anyone raises the level to chase a problem."
    category: security
    reason: "Falsifies plan 36-01 must-have truth 10 as literally written. Does NOT falsify SC 4, which is about what appears in a log line the path actually produces. SETTLED: the developer recorded the explicit acceptance (option (b)) in UAT round 2 Test 2 -- see ## Acknowledged Gaps. Not re-asked by this re-verification."
    evidence_status: "source-confirmed (both lines read directly); no demonstrated leak under the shipped log configuration; disposition accepted by the developer"
  - finding: "IN-20 (36-REVIEW.md iter 3): check_unattended's [ok] heartbeat detail names only the expected ping interval (Period), not the ~20-minute grace time, and the new test pins only 'Period'. An operator acting on the preflight line alone sets Period=15 and leaves healthchecks.io's 1-hour default grace."
    category: other
    reason: "Plan 36-06 truth 6 asks only for an expected-ping-interval reminder, which is present, so this is not a must-have gap. Less pressing since plan 36-07: the create-and-configure step now sets both knobs before the preflight is ever run."
    evidence_status: "source-confirmed; not a must-have breach"
  - finding: "36-REVIEW.md iter 3 carries five further warnings (WR-16 cron exit-99 attribution, WR-17 runtime state-dir mail storm, WR-18 IN-02's skip reasons logged at a dropped DEBUG level, WR-19 check_email accepts dummy/locmem/filebased backends, WR-20 check_flock probe can hang)."
    category: other
    reason: "None falsifies a success criterion or a plan must-have truth; recorded here so the ship decision sees them."
    evidence_status: "review-reported; not independently re-tested by this verification"
behavior_unverified_items:
  - truth: "An operator who configures the heartbeat check using only what the corrected runbook names gets a check that goes late about 15 minutes after a missed tick and alerts about 35 minutes after the last successful ping (plan 36-06 truth 1 -- G-36-3's failed truth, restored)."
    test: "Create a fresh healthchecks-compatible check pointed at FOMO_HEARTBEAT_URL, configuring ONLY what the fresh-host setup steps name (runbook:1463-1493, step 3): expected ping interval (Period) 15 minutes -- or a Cron-type check with */15 * * * * -- and grace time (Grace) about 20 minutes. Then disable the crontab line, simulating the scheduler never invoking the job."
    expected: "The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping, while FOMO itself logs nothing and sends no email -- SC 3's second, independent layer."
    why_human: "The signal is produced by the external heartbeat service's own expected-interval-plus-grace timer, not by any FOMO code path. G-36-3 was only findable this way: every internal gate checked the runbook against decision D-12, which itself carried the conflation."
human_verification:
  - test: "SC-5 sufficiency read-through: a reader who has not read \"The two failure signals\" and has not been told the expected-interval/grace values reads only \"Setting it up on a fresh host\", top-down, and works the steps. RELEASE CONDITION: gap G-36-4 (step 2 instructed an action that stopped Django from starting) was closed by plan 36-08's rewrite of step 2 to the flat LCO_API_KEY assignment, with the fold in src/fomo/settings.py extended to also cover SOAR; administer once re-verification confirms G-36-4 closed."
    expected: "The reader creates the check, sets both of its settings (Period 15 min, or Cron type */15 * * * *; Grace ~20 min) and fills FOMO_HEARTBEAT_URL without leaving the subsection or reading source (SC 5)."
    why_human: "Sufficiency at point of use is not observable by any token-presence gate, and the verdict is only valid from a reader not already taught the knowledge out of band -- this is why round-1 UAT Test 6 passed while G-36-1 was live."
  - test: "Live heartbeat dead-man re-run: re-run UAT Test 3 against a live healthchecks-compatible check configured ONLY from the fresh-host setup steps (expected interval / Period 15 min, or Cron type */15 * * * *; grace / Grace ~20 min), then disable the crontab line. Run from the fresh-host setup steps, and AFTER the SC-5 sufficiency read-through above, because it teaches the values that read-through measures."
    expected: "Check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping, with FOMO logging nothing and mailing nothing."
    why_human: "External service behaviour; no automated gate can reach it. This is the only proof that closes G-36-3 end-to-end."
  - test: "Decide on CR-03 before shipping: the project's own sphinx-build pre-commit hook renders src/fomo/local_settings.py -- the credential home runbook step 2 mandates -- into _readthedocs/html/ and docs/_build/html/, and both trees in this working tree currently contain this host's ping token and mail/API credential names. Decide between fixing it (add '*/local_settings.py' to docs/conf.py's autoapi_ignore, delete both build trees, rotate the heartbeat check if either build was ever served/copied/shared, and correct the runbook's 'never go into a committed file' claim at :1491-1493 to name the docs-build boundary) and recording an explicit acceptance."
    expected: "Either the one-line autoapi_ignore fix plus tree deletion and the runbook boundary correction, or a recorded acceptance stating that no build is ever served from a configured host."
    why_human: "A judgment call on scope and blast radius, not a correctness question. SC 4 as written is about log lines, notifications and error messages the unattended path produces, and a docs-build artifact is none of those -- so this is not a phase-goal gap -- but it is a real on-disk credential exposure produced by following this phase's own runbook, and the .gitignore that currently contains it is the only thing between it and a public repository."
  - test: "Decide on WR-22 before shipping: either fix the two logger.debug() sites (backfill_lco_observations.py:349, unattended.py:191) to log type(exc).__name__, or record an explicit acceptance that the class-name-only discipline holds only while settings.LOGGING keeps the root logger at INFO. ALREADY DECIDED in UAT round 2 (option (b), the recorded acceptance -- see \"Acknowledged Gaps\" in this same file); a re-verification should not re-ask this settled question."
    expected: "Either a two-line fix plus an assertLogs(level='DEBUG') credential-hygiene case, or a recorded acceptance with the constraint documented where an operator raising the log level would see it."
    why_human: "A judgment call on a latent-but-real SCHED-10 exposure: the phase's own must-have truth 10 is written absolutely, while SC 4 as written is not breached under the shipped configuration. Which reading governs the ship decision is the developer's to make. Settled."
---

# Phase 36: Unattended Operation Verification Report

**Phase Goal:** The projector sweep, the LCO/SOAR discovery backfill and the reconciler run on the real host on a documented schedule with nobody typing anything, against a watched-proposal list an operator edits in the admin — and when it breaks, an operator finds out.
**Verified:** 2026-09-18T05:55:00Z
**Status:** gaps_found
**Re-verification:** Yes — third pass, after gap-closure plan 36-07 (wave 5, G-36-1; commits `280962b`, `a2f1ee9`, `71cdec2`, metadata `95b08ed`). Plan 36-07's nine must-have truths were re-established from its frontmatter and checked line-by-line against the current files; the 36-01…36-06 truths were regression-checked (no file under `solsys_code/` changed in this round, and the two named regression suites were re-run green by this verification). This record preserves the two earlier passes' history rather than replacing it.

**Headline:** G-36-1's *structural* half is genuinely closed — the create-and-configure-the-check step exists, stands before the export, and the presence-and-order gate holds when re-run independently. But plan 36-07's Task 2 also shipped an unrelated "same-class sibling" clarification into the same subsection, and that clarification is wrong in a way that stops Django from starting. **The phase regresses from `human_needed` to `gaps_found` on SC 5, on a defect this gap-closure round introduced.**

---

## Goal Achievement

### Observable Truths

#### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Projector sweep, LCO/SOAR discovery backfill and reconciler all run on their documented recurring schedule with no operator action, and two invocations of the same job never overlap | ✓ VERIFIED | Unchanged by this round — `git diff --name-only 280962b^..71cdec2 -- solsys_code/` returns **0 files**. Carried from the prior pass, whose live evidence stands: `run_unattended --dry-run` ran all four steps in D-01 order against the real dev DB and exited 0; overlap re-proven cross-process with the same `/usr/bin/flock -n` binary the crontab invokes (`run_unattended: lock held -- skipping this tick`, no step ran); **UAT Test 2 passed** on the real host (three START/END banners ~15 min apart in `/var/log/fomo/unattended.log`). `solsys_code.tests.test_unattended` re-run by this verification: **Ran 63 tests — OK** |
| SC2 | Adding a proposal in the admin is enough for its robotically scheduled observations to start appearing — discovery takes no per-invocation arguments and needs no redeploy | ✓ VERIFIED | Unchanged by this round. `--proposal` is `required=False, default=None`; the bare path sweeps `watched_rows()` via the shared `sweep_watched_rows()`, the same helper `unattended.step_discovery()` calls. Admin add + `list_editable` toggle proven to change what `watched_rows()` returns (live POST spot-check, prior pass) |
| SC3 | A failed unattended run reaches an operator two independent ways: a notification from the command itself, and a heartbeat that also fires when the scheduler never invoked the job at all | ✓ VERIFIED (live dead-man re-proof is a human item) | Email layer: `unattended._send_notification()` → `notifications.notify_staff()`; 5 `TestNotification` tests green in today's 63-test run; **UAT Test 4 passed** against real SMTP. Heartbeat layer: `ping_heartbeat()` pings `/start` before the first step and `/<exit-code>` after the last; `TestHeartbeat` asserts both orders. The dead-man half was empirically confirmed by the operator in UAT round 1 once the interval was set. `ping_heartbeat()` untouched by this round (0 source files changed) |
| SC4 | No API key or password appears in any log line, notification or error message the unattended path produces | ✓ VERIFIED (two ⚠️ warnings — WR-22 settled, CR-03 open) | `TestCredentialHygiene` (8 tests, green in today's run) seeds a fake LCO api_key, a fake `EMAIL_HOST_PASSWORD` and a fake heartbeat URL, forces failure on every step plus the mail send plus the heartbeat ping, and asserts none reaches the captured log, stdout, stderr, `mail.outbox`, or `WatchedProposal.last_run_summary`. Every `warning`/`error` site logs `type(exc).__name__`. Neither file this round touched carries a UUID-shaped string (`grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` → **0** in both), and the runbook's only heartbeat URL is the placeholder `https://hc-ping.com/<uuid>` at :1490 with an explicit "the ``<uuid>`` part is the ping token" warning at :1491-1493 — a genuine improvement. **Warning 1 (settled):** WR-22's two `DEBUG` sites format `str(exc)`; root logger pinned to `INFO`, developer acceptance recorded. **Warning 2 (open):** CR-03 — the docs build renders `local_settings.py` into HTML. Verified independently: `_readthedocs/html/autoapi/fomo/local_settings/index.html` and `_readthedocs/html/_modules/fomo/local_settings.html` each contain one UUID-shaped string and the names `LCO_API_KEY`/`EMAIL_HOST_USER`/`EMAIL_HOST_PASSWORD`/`FOMO_HEARTBEAT_URL`. **SC 4 as written is not breached** — that is not a log line, a notification, or an error message the unattended path produces — so it is recorded as an advisory with a human decision, not as a gap |
| SC5 | An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source | ✗ **FAILED** | The heartbeat half is now genuinely right (see plan 36-07 truths below). The **API-key half is not**. Step 2 (`runbook:1458-1462`, added by `a2f1ee9` this round) says "The API key setting is nested: ``FACILITIES['LCO']['api_key']`` and ``FACILITIES['SOAR']['api_key']``" as the instruction for what to put in `local_settings.py`. `src/fomo/settings.py:431-434` imports that module into its own namespace; `:436-439`'s own comment spells out that assigning into `FACILITIES[...]` there **raises `NameError`, which the `ImportError` guard does not catch**. Reproduced in isolation by this verification against the same module shape and the same guard: `NameError: name 'FACILITIES' is not defined`, propagating out of the import. The settings module then fails to import — gunicorn, every `manage.py` command, the `run_unattended` tick and `check_unattended` itself all die before running, and the failure email that would report it cannot be sent. The correct instruction (the flat `LCO_API_KEY`) appears **nowhere in `docs/`** (`grep -rn "api_key\|API key" docs/*.rst` returns this sentence and nothing else), only in `settings.py`'s comment — so recovery requires reading source, which is the precise thing SC 5 forbids. The SOAR half is unactionable in any form: `grep -rn "SOAR_API_KEY" --include=*.py` finds no fold, so `FACILITIES['SOAR']['api_key']` stays `''` whatever is written. → **Gap G-36-4** |

#### Plan 36-07 must-have truths (G-36-1 fresh-host heartbeat setup step)

Re-checked against the files, not the SUMMARY. The slice referred to below is
`awk '/^Setting it up on a fresh host$/,/^Adding a proposal to watch$/'` over the runbook —
104 lines, re-extracted by this verification.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 52 | Reading only the fresh-host subsection top-down, the operator reaches the `FOMO_HEARTBEAT_URL` export already knowing what the heartbeat is, what hosts the check, that they must create one, both values, and which URL to copy — because a create-and-configure step stands BEFORE the export | ✓ VERIFIED (point-of-use sufficiency is a human item) | Slice lines 16-46 are the new step 3; line 47 is the export step. Structurally the inversion is gone. Sufficiency at point of use stays human item 1 |
| 53 | The two things that existed in no operator-facing file — the service class and how the operator obtains the ping URL — are now in the procedure, the URL in placeholder form | ✓ VERIFIED | Slice :24-27 "Any healthchecks-compatible service works: healthchecks.io's hosted free tier, or a self-hosted ``healthchecks`` instance (the same open-source Django app)"; :41-46 "copy that check's own ping URL from the service… On healthchecks.io it has the form ``https://hc-ping.com/<uuid>``… The ``<uuid>`` part is the ping token, so this URL is itself a credential" |
| 54 | Each environment variable has its own numbered step; the heartbeat export carries hygiene only and consumes the previous step's URL; `FOMO_BASE_URL` keeps its treatment verbatim | ✓ VERIFIED | Slice :47-51 (export, hygiene + "This is the ping URL the check in the previous step produced") and :52-61 (`FOMO_BASE_URL`). Verbatim confirmed at the diff level: `git diff 280962b^..280962b` shows the `FOMO_BASE_URL` block's only change is the list marker (`-   Export` → `+5. Export`) |
| 55 | The alert-window reasoning still has exactly ONE home; the new step states the values and points at "The two failure signals" rather than re-copying | ✓ VERIFIED (⚠️ see WR-24) | The slice does **not** contain `bounds the maximum allowed gap` (the `/start`-to-completion reasoning) or the 1-day-default trap sentence; the file still does, at :1594-1596 and :1601-1604. The step carries the permitted single arithmetic line plus "(see \"The two failure signals\" below for why these numbers…)" at :37-39. Advisory WR-24 records that this one restated line is inaccurate for the Cron-type route offered in the same sentence — a drift, not a must-have breach |
| 56 | `deploy/cron/fomo.crontab.example` agrees: its variable listing says where the ping URL comes from (by NAME only), and BOTH "for the full setup" pointers name the subsection | ✓ VERIFIED | `grep -cF 'Setting it up on a fresh host'` → **3**, at :16 (the variable listing — "the ping URL of a check you create on a healthchecks-compatible service… See the \"Setting it up on a fresh host\" subsection… for how to create the check and where its ping URL comes from"), :37 (the heartbeat backstop pointer) and :62 (the fresh-host directories pointer). No value anywhere; no UUID-shaped string |
| 57 | Two same-class siblings closed: step 2 names the nested `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` structure (a settings key path, never a value), and the two steps that need root say so | ✗ **FAILED** | The *sudo* half is fine: slice :4-6 ("creating them under ``/var`` and handing ownership to that account typically needs ``sudo``") and :99-102 ("Writing into ``/etc/logrotate.d/`` typically needs ``sudo`` too"), neither contradicting step 6's run-as-the-cron-account warning at :62-63. The *API-key* half names a structure that **cannot be assigned from `local_settings.py`** — the truth's token-level wording ("names the nested structure… never a value") is satisfied while its purpose (spare the operator from reading source) is inverted: following it stops Django from starting. Task complete, goal missed. See SC 5 and gap G-36-4 |
| 58 | `check_heartbeat()`, its `[ok]` reminder text and `test_set_heartbeat_reminds_about_the_check_period` are untouched and stay green — this plan changes no source file | ✓ VERIFIED | `git diff --name-only 280962b^..71cdec2` lists exactly three files, none under `solsys_code/` (`… -- solsys_code/ \| wc -l` → **0**). `grep -qF "confirm the check's own expected ping"` in `check_unattended.py` → present; the pinning test name → present. `python manage.py test solsys_code.tests.test_check_unattended` re-run by this verification: **Ran 30 tests — OK** |
| 59 | `36-VERIFICATION.md` scripts the SC-5 sufficiency read-through BEFORE the live heartbeat test, and records that a sufficiency verdict from a reader already taught the knowledge out of band is not evidence | ✓ VERIFIED | Checked against the pre-edit file before this rewrite: `SC-5 sufficiency read-through` at lines 62, 63, 83, 332; `Live heartbeat dead-man re-run` at 63, 86, 338 — first-occurrence 62 < 63 and last-occurrence 332 < 338, so the order held in both frontmatter and prose. `G-36-1` ×6, `out of band` ×3, `9 numbered steps` ×2, and zero occurrences of the superseded step count. This rewrite preserves that order and that record |
| 60 | The gate has teeth in BOTH dimensions, presence and ORDER | ✓ VERIFIED | Re-run independently by this verification against the live file, not read from the SUMMARY. **Presence** (all in the slice): `Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`, `healthchecks.io`, `The two failure signals`, `35 min`, and a line matching `^9\. `. **Order** (line numbers *inside* the slice): first `Period` = **31**, ping-URL placeholder = **43**, export anchor `the environment the cron daemon sees` = **47** — so 31 < 47 and 43 < 47, the assertion a mis-ordered-but-complete slice would fail. Supporting gates: `pre-commit run ruff --all-files` → Passed; `pre-commit run ruff-format --all-files` → Passed; `test_check_unattended` → 30 tests OK. Docs structure re-checked read-only via a `docutils` parse of the whole runbook (the sphinx-build hook writes into the gitignored build trees, which this verification was scoped not to touch): **no enumerated-list, indentation or block-quote message** from the renumbered 1–9 list or the new multi-paragraph step 3 — only the expected unknown-`:ref:`/`:doc:`-role notices bare docutils always emits |

#### Plan 36-01 must-have truths (runner, heartbeat, email, crontab, shared mailer)

Regression-checked: no file under `solsys_code/` changed in this round, and
`test_unattended` (63) + `test_check_unattended` (30) were re-run green by this verification.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All-succeeding tick exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_healthy_tick_exits_zero`, `test_pings_start_then_exit_code`, `test_empty_database_tick_is_healthy` |
| 2 | Failing tick exits non-zero, pings `/start` then `/<exit-code>`, sends exactly one email to every staff user with an email | ✓ VERIFIED | `test_step_failure_sets_exit_code`, `test_failing_tick_mails_staff_once` |
| 3 | Second consecutive failing tick with the same failing set sends no second email; a later success sends exactly one `FOMO unattended run recovered` | ✓ VERIFIED | `test_repeat_failure_is_suppressed`, `test_recovery_mails_once`; state crosses processes via the real JSON file, written atomically (IN-03) |
| 4 | A raising heartbeat ping never changes the exit code, and neither the URL nor the exception message reaches log/stdout/stderr/email | ✓ VERIFIED | `test_ping_failure_never_fails_the_tick`, `test_heartbeat_ping_failure_leaks_nothing` |
| 5 | With `FOMO_HEARTBEAT_URL` unset: one INFO line per tick, no HTTP call, exit code unaffected | ✓ VERIFIED | `test_unset_url_skips_pinging`; `ping_heartbeat()` early return |
| 6 | Failure subject is `FOMO unattended run failed: <step(s)>`; body carries each failed step's summary, the log path and admin/calendar links — no traceback, no request URL, no portal response text | ✓ VERIFIED | `_build_notification_body()`; `test_failure_email_body_carries_no_secret_and_no_traceback` |
| 7 | A second invocation while the first holds the lock runs no step and logs one skip line | ✓ VERIFIED | `test_contended_lock_skips_every_step` plus the prior pass's cross-process `/usr/bin/flock -n` check |
| 8 | A tick with nothing to do anywhere exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_empty_database_tick_is_healthy`; confirmed live in the prior pass |
| 9 | Step sequence fixed and identical every tick regardless of failures | ✓ VERIFIED | `STEPS` is a single module-level tuple; `test_all_four_steps_run_in_order`, `test_step_failure_does_not_abort_the_tick` |
| 10 | (backstop) No unattended-path log/email/stdout/stderr write formats the message of an exception caught from a `requests` call, a facility/portal call, or `send_mail()` — only its class name | ⚠️ UNCERTAIN (settled acceptance) | Holds at every `warning`/`error` site and is behaviourally backed by the 8 green `TestCredentialHygiene` tests. Two `DEBUG` sites format `str(exc)` (WR-22). Inert as shipped (root logger `INFO`). **The developer recorded the explicit acceptance in UAT round 2 Test 2 (option (b))** — see ## Acknowledged Gaps. Left UNCERTAIN for traceability; not re-asked |

#### Plan 36-02 must-have truths (WatchedProposal + watched-list discovery)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | Staff can add a proposal code in the admin and toggle `is_active` from the changelist, no redeploy, no per-invocation argument | ✓ VERIFIED | `WatchedProposalAdmin` `list_display`/`list_editable`/`list_filter`; `WatchedProposalAdminTests`; prior live HTTP POST spot-check |
| 12 | Bare `backfill_lco_observations` sweeps every active row, applying that row's `target_list_name` and `attributed_to` | ✓ VERIFIED | `sweep_watched_rows()` called from `handle()`; `TestWatchedListSweep`; notebook output `Swept 2 watched proposal(s), failed: 0` |
| 13 | `--proposal X` still sweeps exactly X with the existing flags, and X need not be a watched row | ✓ VERIFIED | Override branch never consults `WatchedProposal` |
| 14 | After a bare sweep every active row carries `last_run_at` and a `last_run_summary` of either the counter line or `failed: <ExceptionClassName>` | ✓ VERIFIED | `sweep_watched_rows()` writes both with `update_fields`; notebook shows real values |
| 15 | A portal error on one row is caught, recorded, counted, and the remaining rows are still swept | ✓ VERIFIED | Per-row try/except; notebook cell 20 shows row B still creating record 900402 after row A failed |
| 16 | Two rows can never share a `proposal_code` | ✓ VERIFIED | `unique=True` in model and migration 0022; `test_duplicate_code_rejected` |
| 17 | Zero active rows → one INFO line, nothing written, exit 0 | ✓ VERIFIED | `TestEmptyWatchedList`; observed live in the prior pass |
| 18 | `proposal_code` equality is exact-match, case-sensitive; whitespace stripped on save | ✓ VERIFIED | `WatchedProposal.save()`; `test_code_is_stripped_on_save`, `test_code_comparison_is_case_sensitive` |
| 19 | Active rows are swept in deterministic `proposal_code` ascending order | ✓ VERIFIED | `Meta.ordering = ['proposal_code']` in model and migration |

#### Plan 36-03 must-have truths (the remaining three steps, SCHED-10 suite)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 20 | One tick runs all four steps in the fixed order, in one process, no `manage.py` subprocess | ✓ VERIFIED | `STEPS` tuple; no `call_command`/`subprocess`/`Popen`/`os.system` in `unattended.py` or `run_unattended.py` |
| 21 | Status refresh calls `update_all_observation_statuses()` on a fresh `LCOFacility()` and a fresh `SOARFacility()` — never shared — and a non-empty failure list is a step failure | ✓ VERIFIED | `test_calls_both_facilities_with_fresh_instances`, `test_non_empty_failure_list_is_a_step_failure`, `test_facility_exception_is_isolated_per_facility` |
| 22 | A status-refresh failure is logged as `observation_id` plus the exception class name; TOM's message half never reaches a log line, stdout/stderr, or the email | ✓ VERIFIED | `_refresh_one_facility()` discards `_message` at the unpack; two named tests |
| 23 | A raising step never prevents later steps; every registry step is attempted every tick | ✓ VERIFIED | `run_tick()` per-step try/except; two named tests |
| 24 | Projector sweep calls `project_queryset()` directly with the same observed-site hook, counts `unprojectable` as a failure, never `call_command()` | ✓ VERIFIED | Three named tests; live sweep reported the real counter line |
| 25 | Discovery sweeps every active row and fails only when at least one row failed; zero rows logs one INFO line and is healthy | ✓ VERIFIED | `step_discovery()` → `sweep_watched_rows()`; four named tests |
| 26 | `unchanged` / `skipped` / `detach_declined` / `remint_declined` never make the tick non-zero and never trigger the email | ✓ VERIFIED | `test_expected_data_shape_outcomes_are_not_failures`; failure derives from typed counters, never from parsing stdout |
| 27 | With a fake api key, mail password and heartbeat URL seeded and every failure path forced in turn, none appears in logs/stdout/stderr/`mail.outbox` | ✓ VERIFIED | The 8-test `TestCredentialHygiene` class, green in today's run. (Scope caveat: captures at the default level — see truth 10) |

#### Plan 36-04 must-have truths (`check_unattended`, cron line, logrotate)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 28 | `check_unattended` reports every prerequisite in one run | ✓ VERIFIED | Prior pass's live run printed flock, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`, `FOMO_STATE_DIR`, `EMAIL_BACKEND`, `staff_recipients`, `heartbeat`, `FOMO_BASE_URL` and `watched_proposals` in one pass |
| 29 | A missing hard prerequisite makes the command exit non-zero and name which check failed | ✓ VERIFIED | Five named tests inside today's green 30 |
| 30 | Unset heartbeat URL and empty watched list are warnings, not failures; exit stays 0 when every hard check passed | ✓ VERIFIED | `hard=False` on both (`check_unattended.py:241/250`, `:365/369`); two named tests |
| 31 | The printed cron line carries the real resolved interpreter and `manage.py` path | ✓ VERIFIED (⚠️ see WR-27) | `cron_line()` resolves `sys.executable`, `manage.py`, `shutil.which('flock')`, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`; `test_line_has_real_paths`. WR-27 is about the *runbook's* claim that the hand-edit route is equivalent, not about this truth |
| 32 | Names and set/unset status only — never a value | ✓ VERIFIED | Three named tests including `test_output_never_contains_a_seeded_value` |
| 33 | `--send-test-email` sends one message through the configured backend to the same staff recipient list the failure notice uses | ✓ VERIFIED | `_send_test_email()` → `notifications.notify_staff()`; **UAT Test 4 passed** against real SMTP |
| 34 | The command only reads: creates no directory, writes no file, changes no row | ✓ VERIFIED | `test_command_writes_nothing` |

#### Plan 36-05 must-have truths (paired docs)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 35 | An operator can set up or verify the whole schedule from one runbook section without reading source | ✗ **FAILED** | Same defect as SC 5 — see that row and gap G-36-4. The heartbeat half of the section is now right; the API-key sentence added this round is not, and recovering from it requires reading `src/fomo/settings.py` |
| 36 | The section documents both failure signals — email (who, what, repeat, clear) and heartbeat (`/start`, `/<exit-code>`, expected ping interval plus grace time, alerting at last ping + interval + grace) — plus the "nothing has appeared" checklist | ✓ VERIFIED | Canonical paragraph at :1580-1604 read directly: both knobs, the arithmetic, the Cron alternative with its wall-clock-slot caveat, the grace's `/start`-to-completion role, and the 1-day-default trap. "When nothing has appeared" (:1608-) gives the four-step checklist, item 3 bounded by interval + grace |
| 37 | The backfill section documents `--proposal` as optional, the bare watched sweep, per-row overrides and per-row failure isolation | ✓ VERIFIED | All four points stated explicitly |
| 38 | The cheat-sheet carries rows for `run_unattended` and `check_unattended`, and `backfill_lco_observations` reflects its optional-argument contract | ✓ VERIFIED | Cheat-sheet rows present for all three |
| 39 | The overlap guarantee is stated at exactly its true strength | ✓ VERIFIED | "What the locking does and does not cover" names `run_unattended --step <name>` as the exclusive manual route and does not overclaim for directly-run sweep commands |
| 40 | The notebook contains executed cells seeding `WatchedProposal` rows, running the command bare, showing `last_run_summary`, and showing one proposal failing without stopping the other | ✓ VERIFIED | Real committed stdout; `execution_count` sequential across all code cells |
| 41 | `docs/notebooks.rst` lists `backfill_lco_observations_demo` in the Demonstration Notebooks toctree | ✓ VERIFIED | Toctree line present; referenced file exists |
| 42 | `CLAUDE.md`'s notebook map records the pairing for every module this phase adds, including that the runbook section — not a notebook — is the paired doc for the runner and `check_unattended`, and why | ✓ VERIFIED | CLAUDE.md maps `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py` → the runbook's "How do I run everything unattended?" section with the stated reason |
| 43 | No committed doc, notebook cell or runbook example quotes a live setting, a `local_settings.py` value, or any credential | ✓ VERIFIED | Re-grepped this round's two files: zero UUID-shaped strings; the runbook's only URL is the `<uuid>` placeholder; the crontab keeps `/path/to/venv/bin/python` and `/path/to/checkout/manage.py` and names every variable without valuing it. (CR-03 concerns a *generated, gitignored* artifact, not a committed one — see the advisory) |

#### Plan 36-06 must-have truths (G-36-3 heartbeat guidance correction)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 44 | An operator configuring the check from the corrected runbook alone gets late ~15 min / alert ~35 min — not a day later | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Strengthened by plan 36-07 — the values are now in the setup step itself, not 80 lines below — but still nobody has configured a live check from the procedure alone, which is the only way this class of defect is findable. → Human Verification item 2 |
| 45 | The guidance names BOTH knobs, states the arithmetic, and names the 1-day interval default as the trap | ✓ VERIFIED | runbook:1584-1604, read directly this pass |
| 46 | The Cron-type `*/15 * * * *` alternative is offered, and the grace stays at about 20 min with its `/start`-to-completion role explained | ✓ VERIFIED | runbook:1589-1597 |
| 47 | The "When nothing has appeared" triage item judges a stale ping against expected interval + grace, not grace alone | ✓ VERIFIED | runbook:1620-1623 — "older than the expected interval plus the grace time -- about 35 minutes with the recommended 15/20 settings" |
| 48 | A troubleshooting entry covers the exact symptom the gap produced | ✓ VERIFIED | Entry present with `^`-underlined title and **Cause:**/**Fix:** form |
| 49 | `deploy/cron/fomo.crontab.example` describes the backstop with both knobs, and `check_unattended`'s `[ok]` heartbeat line reminds the operator the remote check still needs its own expected ping interval | ✓ VERIFIED | crontab:34-38 read directly; the preflight detail string re-confirmed present and unmodified by this round |
| 50 | The verification record's human-test script and its runbook evidence cell state the corrected time-to-alert, traceable to the gap id | ✓ VERIFIED | Carried forward into this rebuild; `re_verification.gaps_closed` cites G-36-3 |
| 51 | Sphinx still builds the runbook with no new warning and `test_check_unattended` stays green | ✓ VERIFIED | `test_check_unattended` → **Ran 30 tests — OK** (re-run by this verification). Docs structure re-checked read-only via `docutils` (see truth 60); `ruff` and `ruff-format` both Passed |

**Score:** 61/65 truths verified (2 failed: SC 5 and plan-36-07 truth 57, one root cause; 1 present-but-behavior-unverified: truth 44; 1 uncertain-and-accepted: truth 10)

---

### Plan 36-07 Prohibitions (must-NOT checks)

All five are judgment-tier (`verification: flagged-unverified`). Per the fail-closed rule these carry a
**non-authoritative LLM-judge verdict** and are flagged for human review; none is silently passed.

| # | Prohibition | Judge verdict | Evidence | Flag |
|---|-------------|---------------|----------|------|
| P1 | No committed file may gain a real heartbeat URL, a real ping UUID, or any other `FOMO_*` value; the placeholder `https://hc-ping.com/<uuid>` IS allowed | Satisfied | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` → **0** in both `docs/runbooks/telescope_runs_calendar.rst` and `deploy/cron/fomo.crontab.example`. The runbook's only URL is the placeholder at :1490, with the "``<uuid>`` part is the ping token" warning immediately after. The crontab names every variable without valuing it. Residual (a hand-written fake URL of another shape) inspected by eye: none | unverified-prohibition — human review recommended |
| P2 | The grace-time recommendation stays at about 20 minutes (it also bounds `/start`-to-completion) | Satisfied | Slice :34-35 "the grace time (``Grace``) = about 20 minutes"; canonical :1593-1597 keeps the reasoning; crontab :38 "its grace time (about 20 min)" | unverified-prohibition — human review recommended |
| P3 | Wording stays service-agnostic: concept first, healthchecks.io's spelling in parentheses | Satisfied (one stylistic residue) | Slice :24 "Any healthchecks-compatible service works"; :29-34 names each concept with the vendor spelling parenthesised. Residue: the prose retains the plan's own authoring directive "Name each concept first, giving healthchecks.io's spelling in parentheses" (:30), mirroring the pre-existing "Name the concept first:" at :1585 — advisory IN-26 | unverified-prohibition — human review recommended |
| P4 | `ping_heartbeat()`, the URL construction and the exit-code path are untouched; this plan changes NO source file at all — only the runbook, the crontab template and this record | Satisfied | `git diff --name-only 280962b^..71cdec2` returns exactly those three files; `… -- solsys_code/ \| wc -l` → **0**. `check_heartbeat()`'s reminder text and `test_set_heartbeat_reminds_about_the_check_period` both still present and green | unverified-prohibition — human review recommended |
| P5 | No second copy of the alert-window paragraph — the new step gives values and points at "The two failure signals" for the reasoning | Satisfied on the gated half, ⚠️ drifted on the ungated half | The slice does NOT contain `bounds the maximum allowed gap` while the file still does — the marker gate holds. But the step restates the *formula* line, and that restatement is inaccurate for the Cron-type route (advisory WR-24) — precisely the "reworded restatement passes the marker grep" residual the plan itself flagged as judgment. IN-25 counts five copies of the 15/20/35 triple across the two files | unverified-prohibition — human review recommended |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/runbooks/telescope_runs_calendar.rst` | Fresh-host procedure with a create-and-configure-the-check step before the export | ⚠️ **PRESENT, ONE STEP DEFECTIVE** | 9 numbered steps confirmed (`^9\. ` present in the slice); the heartbeat step is at :1463-1493, the export at :1494, order gate holds. But step 2 (:1458-1462) instructs a `NameError`-raising assignment — see gap G-36-4 |
| `deploy/cron/fomo.crontab.example` | Ping-URL provenance + both "full setup" pointers naming the subsection | ✓ VERIFIED | 3 subsection references (:16, :37, :62); `healthchecks` present; no UUID; placeholders intact |
| `.planning/phases/36-unattended-operation/36-VERIFICATION.md` | Human-test order corrected, contamination rule recorded | ✓ VERIFIED | Pre-rewrite check: SC-5 marker first in both frontmatter (62 < 63) and prose (332 < 338); `G-36-1` ×6; zero occurrences of the superseded step count. Order preserved in this rewrite |
| `solsys_code/unattended.py` | Runner, steps, lock, heartbeat, notification | ✓ VERIFIED | Unchanged this round; 63 tests green |
| `solsys_code/notifications.py` | Shared request-free mailer | ✓ VERIFIED | Unchanged; called from `unattended.py`, `check_unattended.py`, `campaign_views.py` |
| `solsys_code/management/commands/run_unattended.py` | Cron entry point | ✓ VERIFIED | Unchanged |
| `solsys_code/management/commands/check_unattended.py` | Preflight + cron line | ✓ VERIFIED | Unchanged; 30 tests green; `[ok]` heartbeat reminder intact |
| `solsys_code/models.py` (`WatchedProposal`) | Admin-editable watch list | ✓ VERIFIED | Unchanged |
| `solsys_code/migrations/0022_watchedproposal.py` | Matching migration | ✓ VERIFIED | Unchanged |
| `solsys_code/admin.py` (`WatchedProposalAdmin`) | Registered, `list_editable` | ✓ VERIFIED | Unchanged |
| `solsys_code/management/commands/backfill_lco_observations.py` | `sweep_proposal()`, `watched_rows()`, `sweep_watched_rows()`, optional `--proposal` | ✓ VERIFIED | Unchanged |
| `deploy/logrotate/fomo.example` | Daily, rotate 14, copytruncate | ✓ VERIFIED | Unchanged; **UAT Test 5 passed** |
| `src/fomo/settings.py` | `FOMO_BASE_URL`/`HEARTBEAT_URL`/`LOCK_DIR`/`STATE_DIR`/`LOG_FILE` | ✓ VERIFIED | All `os.getenv()`-sourced (:415-429); the `LCO_API_KEY` fold and its explanatory comment at :436-441 are what gap G-36-4's fix must agree with |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | Executed watched-proposal cells | ✓ VERIFIED | Unchanged this round |
| `docs/conf.py` | Sphinx/autoapi config | ⚠️ **ADVISORY (CR-03)** | `autoapi_dirs = ['../src']` at :62 with `autoapi_ignore = ['*/__main__.py', '*/_version.py']` at :63 — no `local_settings.py` exclusion. Not touched by phase 36 |
| Tests (`test_unattended`, `test_check_unattended`) | Behavioral coverage | ✓ VERIFIED | **93 tests re-run green by this verification** (63 + 30) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| the new create-and-configure-the-check step | the export step that consumes its output | numbered-step order inside the subsection | ✓ WIRED | Slice line 43 (ping URL) and 31 (`Period`) both precede line 47 (`the environment the cron daemon sees`) — **this is the inversion G-36-1 named, and it is gone**. The export step names its input explicitly: "This is the ping URL the check in the previous step produced" |
| the new step's values | the canonical "Heartbeat." paragraph's reasoning | quoted-subsection cross-reference, no re-copy | ✓ WIRED (⚠️ one restated line drifted) | "(see \"The two failure signals\" below for why these numbers…)" at slice :37-39; the `/start`-to-completion sentence is absent from the slice and present in the file. WR-24: the one permitted arithmetic line is wrong for the Cron route |
| the runbook's setup subsection | `deploy/cron/fomo.crontab.example`'s variable listing and both "full setup" pointers | shared subsection name | ✓ WIRED | All three crontab references name "Setting it up on a fresh host" |
| the runbook's preflight step | `check_heartbeat()`'s `[ok]` detail | `test_set_heartbeat_reminds_about_the_check_period` | ✓ WIRED | The triad was deliberately untouched; all three still agree and the test is green |
| **runbook step 2's API-key instruction** | **`src/fomo/settings.py`'s `LCO_API_KEY` fold** | **the operator writing `local_settings.py`** | ✗ **NOT WIRED** | The runbook names `FACILITIES['LCO']['api_key']`; the fold reads the flat `LCO_API_KEY`. The two do not meet — the documented path raises `NameError` before any fold runs. `FACILITIES['SOAR']['api_key']` has no fold at all. **Gap G-36-4** |
| `run_unattended.Command.handle()` | `unattended.run_tick()` → `STEPS` | direct call | ✓ WIRED | Unchanged; `--step` choices derive from the single `STEPS` tuple |
| `unattended.run_tick()` | `notifications.notify_staff()` | `_send_notification()` | ✓ WIRED | Unchanged |
| crontab template's `flock -n` path | `settings.FOMO_LOCK_DIR` | `<dir>/run_unattended.cron.lock` | ✓ WIRED | Unchanged; proven cross-process in the prior pass |
| `WatchedProposal.objects.filter(is_active=True)` | `sweep_proposal()` | `watched_rows()` → `sweep_watched_rows()` | ✓ WIRED | Unchanged; one shared helper, two callers |
| `step_status_refresh()` | Phase 34 `post_save` receiver | `facility.update_observation_status()` → `.save()` | ✓ WIRED | Unchanged |

---

### Data-Flow Trace (Level 4)

Unchanged by this round (no source file touched); carried from the prior pass, whose live tick
produced real counters from the developer database.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `step_project_sweep()` | `result['counters']` | `project_queryset()` over `ObservationRecord.objects.filter(facility__in=PROJECTED_FACILITIES)` | Yes — prior live tick reported `unchanged: 159` | ✓ FLOWING |
| `step_reconcile()` | `run_count`/`failed_count` | `CampaignRun.objects.all()` | Yes — prior live tick reported `runs: 45, failed: 0` | ✓ FLOWING |
| `step_discovery()` | `rows` | `watched_rows()` → `WatchedProposal` query | Yes — 0 rows live (correct quiet no-op), 2 rows with real portal-shaped payloads in the notebook | ✓ FLOWING |
| `WatchedProposalAdmin` changelist | `last_run_at`/`last_run_summary` | sweep-written model fields | Yes | ✓ FLOWING |
| `check_unattended` report | `results` | live `shutil.which`, `os.access`, settings, `WatchedProposal` count | Yes — host-accurate verdicts including real uid/owner/mode | ✓ FLOWING |
| `cron_line()` | printed cron line | `sys.executable`, `BASE_DIR`, `shutil.which('flock')`, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE` | Yes — five resolved values (⚠️ the runbook claims only two matter; WR-27) | ✓ FLOWING |
| Failure email body | `settings.FOMO_LOG_FILE`, `FOMO_BASE_URL` | settings, not hardcoded literals | Yes — asserted present in the body by test | ✓ FLOWING |

---

### Behavioral Spot-Checks

Run by this verification against the current tree, not quoted from any SUMMARY.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| The slice-scoped presence gate plan 36-07 claims | `awk '/^Setting it up on a fresh host$/,/^Adding a proposal to watch$/' … \| grep -qF` ×7 + `grep -qE '^9\. '` | `Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`, `healthchecks.io`, `The two failure signals`, `35 min`, 9th step — all PRESENT | ✓ PASS |
| The slice-scoped **order** gate (the one that actually catches G-36-1) | line positions inside the slice | `Period` = 31, ping URL = 43, export anchor = 47 → 31 < 47 and 43 < 47 | ✓ PASS |
| The canonical-paragraph no-re-copy gate | `! grep -qF 'bounds the maximum allowed gap'` on the slice; `grep -qF 'also bounds…'` on the file | slice clean, file retains it | ✓ PASS |
| No real ping token in either committed file | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` | runbook **0**, crontab **0** | ✓ PASS |
| Crontab pointers name the subsection | `grep -cF 'Setting it up on a fresh host' deploy/cron/fomo.crontab.example` | **3** (:16, :37, :62) | ✓ PASS |
| Plan 36-07 changed no source file | `git diff --name-only 280962b^..71cdec2 -- solsys_code/ \| wc -l` | **0** | ✓ PASS |
| **The documented API-key assignment** | reproduced the same module shape and the same `except ImportError` guard, then imported it | `NameError: name 'FACILITIES' is not defined` — propagated out of the import; the "imported ok" line never ran | ✗ **FAIL** → gap G-36-4 |
| A `SOAR_API_KEY` fold exists anywhere | `grep -rn "SOAR_API_KEY" --include=*.py .` | no match | ✗ **FAIL** → gap G-36-4 |
| `local_settings.py` rendered into the docs build | `ls _readthedocs/html/autoapi/fomo/local_settings/index.html docs/_build/html/…`; counted credential names and UUID-shaped strings without quoting values | both trees present; each page holds 1 UUID-shaped string and the names `LCO_API_KEY`/`EMAIL_HOST_USER`/`EMAIL_HOST_PASSWORD`/`FOMO_HEARTBEAT_URL`; both trees `.gitignore`d (:76-77) | ✗ FAIL → advisory CR-03 (outside SC 4's wording) |
| `check_unattended` regression suite | `python manage.py test solsys_code.tests.test_check_unattended` | Ran 30 tests — **OK** | ✓ PASS |
| Runner regression suite | `python manage.py test solsys_code.tests.test_unattended` | Ran 63 tests — **OK** | ✓ PASS |
| Lint gate | `pre-commit run ruff --all-files` (pinned ruff 0.2.1) | Passed | ✓ PASS |
| Format gate | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| Runbook RST structure after the 1–9 renumbering | read-only `docutils` parse at `report_level=2` | no enumerated-list / indentation / block-quote message; only the expected unknown-`:ref:`/`:doc:`-role notices | ✓ PASS |
| Debt markers in the two files this round modified | `grep -nE "TBD\|FIXME\|XXX"` | one hit — `runbook:1319 "``TBD window``"`, the campaign domain's own window vocabulary, not a debt marker | ✓ PASS |
| Live external heartbeat alerting from the corrected procedure | — | requires a live healthchecks-compatible account | ? SKIP → human verification |
| Sufficiency of the subsection for a naive reader | — | not observable by any token gate | ? SKIP → human verification |

**Note on the sphinx-build gate:** this verification was scoped to modify no file but
`36-VERIFICATION.md`, and `pre-commit run sphinx-build` writes into `_readthedocs/html/` — which,
per CR-03, is where the credential render lands. The docs check was therefore done read-only with
`docutils` instead. Plan 36-07's own task gates ran the full hook and passed; the independent
structural parse above corroborates the renumbered list is valid RST.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN or SUMMARY declares one; this project's verification contract is the Django test runner (36-VALIDATION.md) | n/a — SKIPPED |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCHED-08 | 36-01, 36-03, 36-04, 36-05, 36-06, 36-07 | Projector sweep, discovery backfill and reconciler on a documented cron + `flock -n` schedule with no operator action, guarded against overlapping invocations | ⚠️ **PARTIALLY BLOCKED** | The mechanism is satisfied and unchanged (`STEPS` + `run_unattended` + committed crontab template + `check_unattended`'s printed line; overlap proven cross-process; **UAT Test 2 passed on the real host**). The *documented* half regresses: the documented setup procedure now contains a step that prevents Django from starting, so an operator provisioning a genuinely fresh host from it does not reach a running schedule. Closing G-36-4 restores this to SATISFIED |
| SCHED-09 | 36-01, 36-03, 36-05, 36-06, 36-07 | Failure visible through two independent layers — in-command notification and a heartbeat/dead-man's switch | ✓ SATISFIED (one human re-proof open) | `notifications.notify_staff()` (5 tests; UAT Test 4 passed against real SMTP) + `ping_heartbeat()` `/start` / `/<exit-code>` (3 tests; dead-man half empirically confirmed in UAT round 1). G-36-3's guidance defect closed in five surfaces; G-36-1's structural half closed by 36-07 — the check-creation step now precedes the export. The live re-run from the procedure alone is human item 2 |
| SCHED-10 | 36-01, 36-03, 36-04, 36-05 | No credential value in any log line or notification the unattended path generates | ✓ SATISFIED (⚠️ WR-22 accepted; ⚠️ CR-03 advisory) | Class-name-only discipline at every `warning`/`error` site, 8 `TestCredentialHygiene` tests green, `check_unattended`'s names-only output verified live, no committed artifact carrying a value, zero UUID-shaped strings in this round's two files. Two `DEBUG` sites format `str(exc)`, inert under the shipped `INFO` root logger — developer acceptance recorded. CR-03's docs-build render is a generated artifact, not a log line or notification the unattended path generates |
| DISCOVER-01 | 36-02, 36-03, 36-05 | Admin-editable watched-proposal list replaces per-invocation `--proposal`/name-prefix arguments | ✓ SATISFIED | `WatchedProposal` model/migration/admin; `--proposal` optional; bare sweep over `watched_rows()` through the shared `sweep_watched_rows()`; admin toggle proven to change discovery scope end-to-end. Untouched by this round |

**Orphaned requirements:** none. `.planning/REQUIREMENTS.md:121-124` maps exactly SCHED-08, SCHED-09,
SCHED-10 and DISCOVER-01 to Phase 36, and all four are claimed by plan frontmatter (36-07 claims
SCHED-08 and SCHED-09). REQUIREMENTS.md marks all four `Complete` at `:121-124` and `[x]` at `:45-48`
— **that status is now premature for SCHED-08** until G-36-4 closes.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `docs/runbooks/telescope_runs_calendar.rst` | 1458-1462 | Operator instruction naming a settings path that the documented configuration file cannot assign (`FACILITIES['LCO']['api_key']` in `local_settings.py`) | 🛑 **Blocker** | Reproduced `NameError`; Django refuses to start; recovery requires reading `src/fomo/settings.py`. **Regression of this gap-closure round** (`git log -S` → introduced by `a2f1ee9`), so the convergence evidence gate does not shield it — and it is deterministically evidenced regardless. → gap G-36-4 |
| `docs/runbooks/telescope_runs_calendar.rst` | 1461-1462 | Documented route with no implementation anywhere (`FACILITIES['SOAR']['api_key']`) | 🛑 **Blocker** (same gap) | An operator believes SOAR is authenticated while `status_refresh` calls the portal with an empty key. Same commit, same fix |
| `docs/conf.py` | 62-63 | `autoapi_dirs = ['../src']` with no `local_settings.py` exclusion, while the runbook mandates that file as the credential home | ⚠️ Warning | CR-03. Real, reproduced on disk, gitignored. Outside SC 4's wording (not a log line/notification/error message the unattended path produces) and rooted in a file phase 36 never touched → advisory + human decision, not a gap |
| `docs/runbooks/telescope_runs_calendar.rst` | 1476-1486 | The one restated arithmetic line is applied to a Cron-type check that has no `Period` and alerts from the wall-clock slot | ⚠️ Warning | WR-24. Contradicts the canonical paragraph at :1589-1591 within the same file |
| `docs/runbooks/telescope_runs_calendar.rst` | 1463-1466 | Relative clause attaches to "one failed outright", inverting which failure class FOMO cannot report | ⚠️ Warning | WR-25. Reverses the justification in the step whose job is to supply it |
| `docs/runbooks/telescope_runs_calendar.rst` | 1451-1456, 1516-1530 | `FOMO_STATE_DIR` is a third hard check (`check_unattended.py:177`) named nowhere in `docs/`; step 6's hard-set enumeration stale | ⚠️ Warning | WR-26 (iter-3 WR-21, still open). Benign while the default holds (`FOMO_STATE_DIR` defaults to `FOMO_LOCK_DIR`) |
| `docs/runbooks/telescope_runs_calendar.rst`, `deploy/cron/fomo.crontab.example` | 1542-1545, 3-11, 56 | "either route produces the same line" / "replace BOTH placeholder paths" vs `cron_line()`'s five resolved values including `shutil.which('flock')` | ⚠️ Warning | WR-27. The hand-edit route can install a line that exits 127 on every tick — silent no-op |
| `solsys_code/management/commands/backfill_lco_observations.py`, `solsys_code/unattended.py` | 349, 191 | `logger.debug()` formatting `str(exc)` on the unattended path | ⚠️ Warning (accepted) | WR-22. Inert under the shipped `INFO` root logger; developer acceptance recorded — see ## Acknowledged Gaps |
| `docs/runbooks/telescope_runs_calendar.rst` | 1476-1477, 1584-1585 | Authoring directives surviving into operator prose ("Name each concept first, giving healthchecks.io's spelling in parentheses") | ℹ️ Info | IN-26 |
| `deploy/cron/fomo.crontab.example` | 32-33, 9-11, 62-65 | `[ $? -eq 99 ]` vs the line's `[ $rc -eq 99 ]`; two "(a later plan in this phase)" parentheticals for artifacts that shipped | ℹ️ Info | IN-27. One of the two sentences was rewritten by `a2f1ee9` with the stale parenthetical preserved |
| `docs/runbooks/telescope_runs_calendar.rst` | 1543, 1546 | Single-backtick deploy paths render as italic title references, unlike every other path in the subsection | ℹ️ Info | IN-28 |
| phase-modified files (all) | — | `TBD` / `FIXME` / `XXX` / `HACK` debt markers | — none | The only `TBD` hit is `runbook:1319`'s `` ``TBD window`` `` — the campaign domain's own window vocabulary, not a debt marker |

**🛑 Blockers: 1 defect (2 rows, one root cause) — gap G-36-4.**

---

### Advisory (New Scope, Unevidenced)

**None.** Every warning recorded above is either a carried-forward item from a prior review
iteration or was independently reproduced by this verification with a concrete command and its
output, so nothing was downgraded for lack of evidence. The `advisory:` frontmatter list carries
findings that are evidenced but fall **outside** the phase's success criteria as written (CR-03,
WR-24…WR-27, IN-25…IN-28, WR-22, IN-20) — a different category from "new scope, unevidenced",
recorded there so the ship decision sees them without their reverting a completed must-have.

---

### CLAUDE.md Paired-Docs Compliance

Plan 36-07 changed **no** module under `solsys_code/` (`git diff --name-only 280962b^..71cdec2 --
solsys_code/` → 0 files), so no paired notebook obligation arose. The change is itself a runbook
edit — `docs/runbooks/telescope_runs_calendar.rst` is the paired doc CLAUDE.md names for the runner
and `check_unattended`, and it is the artifact being corrected. `deploy/cron/fomo.crontab.example`
was updated in the same pass to stay in agreement. **Compliant.**

---

### Human Verification Required

#### 1. SC-5 sufficiency read-through

**Test:** A reader who has not read "The two failure signals" and has not been told the
expected-interval/grace values reads only "Setting it up on a fresh host", top-down, and works the
steps.
**Expected:** The reader creates the check, sets both of its settings (`Period` 15 min, or Cron type
`*/15 * * * *`; `Grace` ~20 min) and fills `FOMO_HEARTBEAT_URL` without leaving the subsection or
reading source (SC 5).
**Why human:** Sufficiency at point of use is not observable by any token-presence gate, and the
verdict is only valid from a reader not already taught the knowledge out of band — this is why
round-1 UAT Test 6 passed while G-36-1 was live.
**RELEASE CONDITION:** gap **G-36-4** (step 2 instructed an assignment that stopped Django from
starting) was closed by plan 36-08, which rewrote step 2 to the flat `LCO_API_KEY` assignment and
extended the fold in `src/fomo/settings.py` to also cover the SOAR facility entry. Administer this
read-through once re-verification confirms G-36-4 closed.

#### 2. Live heartbeat dead-man re-run

**Test:** Create a fresh healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL`, configuring
**only** what the fresh-host setup steps name: an expected ping interval (`Period`) of 15 minutes —
or a Cron-type check with `*/15 * * * *` — and a grace time (`Grace`) of about 20 minutes. Then
disable the crontab line, simulating the scheduler never invoking the job. Run this from the
fresh-host setup steps, and AFTER item 1 above, because it teaches the values item 1 measures.
**Expected:** The check goes late about 15 minutes after the missed tick and alerts about 35 minutes
after the last successful ping (last ping + expected interval + grace), while FOMO itself logs
nothing and sends no email — SC 3's second, independent layer.
**Why human:** The signal comes from the external service's own timer, not from any FOMO code path.
G-36-3 was only findable this way.

#### 3. Decide on CR-03 before shipping

**Test:** The project's own `sphinx-build` pre-commit hook renders `src/fomo/local_settings.py` —
the credential home runbook step 2 mandates — into `_readthedocs/html/` and `docs/_build/html/`.
Both trees in this working tree contain it right now, complete with this host's ping token and the
`LCO_API_KEY` / `EMAIL_HOST_USER` / `EMAIL_HOST_PASSWORD` names. Decide between fixing it (add
`'*/local_settings.py'` to `docs/conf.py`'s `autoapi_ignore`; delete both build trees; rotate the
heartbeat check if either build was ever served, copied or shared; and correct the runbook's "must
never go into a committed file" boundary claim at :1491-1493, which names the wrong boundary) and
recording an explicit acceptance.
**Expected:** Either the one-line `autoapi_ignore` fix plus the cleanup, or a recorded acceptance
stating that no build is ever served from a configured host.
**Why human:** A judgment call on scope and blast radius. SC 4 as written covers log lines,
notifications and error messages *the unattended path produces*, and a docs-build artifact is none
of those — so this is not a phase-goal gap — but it is a real on-disk credential exposure produced
by following this phase's own runbook, and `.gitignore` is the only thing between it and the
repository.

#### 4. Decide on WR-22 before shipping — **SETTLED, do not re-ask**

**Test:** Either fix the two `logger.debug()` sites (`backfill_lco_observations.py:349`,
`unattended.py:191`) to log `type(exc).__name__`, and extend `TestCredentialHygiene` with an
`assertLogs(level='DEBUG')` case — or record an explicit acceptance that the class-name-only
discipline holds only while `settings.LOGGING` keeps the root logger at `INFO`.
**Expected:** A decision, either way, recorded against the phase.
**Why human:** A judgment call. **Already decided in UAT round 2 (option (b), the recorded
acceptance — see "Acknowledged Gaps" below);** a re-verification should not re-ask this settled
question.

**Closed by UAT (no longer open):** the fresh-host preflight (Test 1, pass), the real crontab
(Test 2, pass), real SMTP delivery (Test 4, pass), logrotate under a live writer (Test 5, pass),
and the WR-22 ship decision (round 2 Test 2, pass). See `36-UAT.md`.

**RE-OPENED by G-36-1:** runbook sufficiency (Test 6) is not counted as closed. Its round-1 pass
came from a reader who had already been taught the `Period`/`Grace` values by Test 3 earlier in the
same session, so it was not a sufficiency measurement — see item 1 above and G-36-1 in `36-UAT.md`
/ `.planning/debug/heartbeat-setup-step-context-gap.md`.

The human-test order above was corrected after G-36-1 (see `36-UAT.md` and
`.planning/debug/heartbeat-setup-step-context-gap.md`): a sufficiency verdict from a reader who has
already been taught the knowledge out of band is not evidence. This rewrite preserves that order.

---

### Gaps Summary

**One gap, and it is this round's own doing.**

Plan 36-07 was asked to close G-36-1 — the fresh-host procedure's missing create-the-check step —
and on that job it succeeded, verifiably and independently of its SUMMARY. The subsection now runs
to nine numbered steps; step 3 tells the operator what the heartbeat is, that any
healthchecks-compatible service will do (naming both the hosted and self-hosted routes), to create
one check and set both of its settings to 15 and ~20 minutes, what the resulting alert window is,
where the reasoning lives, and that the ping URL has the form `https://hc-ping.com/<uuid>` and is
itself a credential. The export step that used to stand alone now consumes that URL by name. The
gate the plan built has real teeth: re-run here from scratch, it confirms not just that the tokens
are present but that they are in the right *place* — `Period` at slice line 31 and the ping URL at
43, both ahead of the export anchor at 47. The crontab template's three pointers all resolve to
that subsection. No source file moved, the paired preflight/reminder/test triad is untouched and
green, and neither committed file carries so much as a UUID-shaped string.

**But the same plan's Task 2 carried an unrelated "same-class sibling" clarification into step 2 of
that very subsection, and that clarification is wrong in the most expensive way a runbook sentence
can be.** It tells the operator the API key goes at `FACILITIES['LCO']['api_key']` and
`FACILITIES['SOAR']['api_key']` in `local_settings.py`. `src/fomo/settings.py` imports that module
into its own namespace and says so in its own comment, four lines above the fold that exists
precisely because of this: assigning into `FACILITIES[...]` there raises `NameError`, and the guard
around the import catches `ImportError` only. Reproduced here in isolation. The result is not a
confusing sentence — it is a host whose Django refuses to start, whose `manage.py` commands all die,
and whose failure email cannot be sent to say so. The only correct instruction, the flat
`LCO_API_KEY`, appears nowhere in `docs/`. An operator following the one section SC 5 names has to
go read source to get out. And the SOAR half cannot be made to work at all: no `SOAR_API_KEY` fold
exists, so that slot stays empty however it is configured, while the operator believes SOAR is
authenticated.

The irony is exact and worth recording: the plan's own must-have truth for this edit — "setup step 2
names the nested `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` structure (a
settings key path, never a value)" — is satisfied at the token level and its automated gate passes,
because the gate asked whether those strings are in the slice, not whether they are *true*. This is
the same failure shape as G-36-1 itself (a whole-file token probe that a bare export line
satisfied), reproduced one step to the left.

Everything else in the phase was re-proven rather than carried on trust where it could be: 93 tests
re-run green, ruff and ruff-format re-run, the runbook's RST re-parsed structurally after the
renumbering, the slice gates re-executed from scratch, and the commit range confirmed to have
touched zero source files. The remaining open items are the two human proofs that no automated gate
can reach — the naive-reader sufficiency read-through (now held until G-36-4 is fixed) and the live
external dead-man test — plus one security decision (CR-03) that is real, reproduced, and outside
SC 4's wording.

### Acknowledged Gaps

- **Plan 36-01 truth 10 / 36-REVIEW.md WR-22 — accepted, not fixed (2026-09-18, 36-UAT.md round 2 Test 2).**
  Two `DEBUG`-level sites format `str(exc)` on the unattended path
  (`solsys_code/management/commands/backfill_lco_observations.py:349`, `solsys_code/unattended.py:191`).
  Under the shipped `settings.LOGGING` (root logger `INFO`) neither line is emitted, so SC 4 holds and no
  credential-bearing log line is produced. The developer recorded an explicit acceptance on the condition
  that the root logger stays at `INFO`; the constraint and the required fix (log `type(exc).__name__` only,
  plus an `assertLogs(level='DEBUG')` credential-hygiene test) are documented in a comment directly above
  `LOGGING` in `src/fomo/settings.py`, where an operator raising the level would see it. Truth 10 remains
  recorded as UNCERTAIN above for traceability; it does not block phase completion, and this
  re-verification did not re-open it.

---

_Verified: 2026-09-18T05:55:00Z_
_Verifier: Claude (gsd-verifier)_
_Pass 3 (re-verification after gap-closure plan 36-07). Passes 1 and 2 are preserved in this file's git history._
