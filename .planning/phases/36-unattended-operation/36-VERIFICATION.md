---
phase: 36-unattended-operation
verified: 2026-09-18T07:30:00Z
status: human_needed
score: 75/77 must-haves verified
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
  - ".planning/phases/36-unattended-operation/36-06-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-06-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-07-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-07-SUMMARY.md"
  - ".planning/phases/36-unattended-operation/36-08-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-08-SUMMARY.md"
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
  - "solsys_code/tests/test_settings_api_key_fold.py"
  - "solsys_code/tests/test_unattended.py"
  - "solsys_code/tests/test_watched_proposal.py"
  - "solsys_code/unattended.py"
  - "src/fomo/settings.py"
covered_digest: "v1:sha256:0ab7615b2e33bd77469a5e1b6e3660988753f9c7c0e1cc00d3aa5fa1b43c2890"
behavior_unverified: 1
overrides_applied: 0
decision_coverage:
  honored: 18
  total: 18
  not_honored: []
re_verification:
  previous_status: gaps_found
  previous_score: 61/65
  previous_verified: 2026-09-18T05:55:00Z
  gaps_closed:
    - "G-36-3 (UAT round 1, Test 3) -- closed by plan 36-06. The runbook's heartbeat guidance named only the check's grace time, never its expected ping interval, so a check configured from it first alerted about a day after the schedule stopped. The runbook (4 sites + a troubleshooting entry), deploy/cron/fomo.crontab.example, two unattended.py docstrings and check_unattended's [ok] heartbeat line now all state the two-knob configuration and the last-ping + expected-interval + grace arithmetic. The live external re-proof remains human item 2."
    - "G-36-1 (UAT round 2, Test 1) -- STRUCTURAL HALF closed by plan 36-07 (280962b, a2f1ee9, 71cdec2). The fresh-host procedure runs to 9 numbered steps with a create-and-configure-the-heartbeat-check step standing BEFORE the FOMO_HEARTBEAT_URL export. Re-proven again by THIS pass against the live file after plan 36-08's step-2 edit: slice tokens Period/Grace/'*/15 * * * *'/hc-ping.com/<uuid>/healthchecks.io/'The two failure signals'/'35 min' plus a 9th numbered step all present, and the ORDER gate still holds inside the slice (Period at slice line 39, ping-URL at 51, export anchor 'the environment the cron daemon sees' at 55). The SUFFICIENCY half stays open as human item 1 -- now RELEASED, since its blocking condition (G-36-4) is closed."
    - "G-36-4 (verification pass 3, introduced by plan 36-07 commit a2f1ee9) -- CLOSED by plan 36-08 (62d4d78, 39312f4, 96701a3). Runbook step 2 told the operator to write FACILITIES['LCO']['api_key'] / FACILITIES['SOAR']['api_key'] in local_settings.py, an assignment that raises NameError inside that module's own import namespace and stops Django from starting; and no SOAR fold existed at all. Re-verified at HEAD by THIS pass, not read from the SUMMARY: (a) step 2 now reads ``LCO_API_KEY = '<your key>'`` with the NameError clause, the fold description and the omitted-setting no-op (runbook:1458-1470); (b) the awk-sliced subsection contains ZERO bracketed dict subscripts of any kind (grep -E \"[A-Za-z_]+\\[['\\\"]\" over the 112-line slice -> no match), so nothing nested is copy-pasteable out of the procedure; (c) src/fomo/settings.py:440-443 now folds the one flat key into BOTH FACILITIES['LCO']['api_key'] and FACILITIES['SOAR']['api_key']; (d) the live settings module imports cleanly on this host and reports api_key present and non-empty for BOTH LCO and SOAR (key names only printed, never a value); (e) the new committed test module solsys_code/tests/test_settings_api_key_fold.py runs 4 tests OK, executing the REAL fold tail sliced out of the live settings file against an injected sys.modules entry; (f) LCO_API_KEY is now documented in docs/ (runbook:1462) where before it appeared nowhere, so recovery no longer requires reading source."
  gaps_remaining: []
  regressions: []
  human_items_closed_by_uat:
    - "Test 1 fresh-host preflight -- UAT round 1 pass (all hard checks [ok], exit 0, cron line printed)"
    - "Test 2 the real crontab -- UAT round 1 pass (three START/END banners ~15 min apart on the real host)"
    - "Test 4 real mail delivery -- UAT round 1 pass"
    - "Test 5 logrotate under a live writer -- UAT round 1 pass (START went with the rotated copy; END landed in the truncated live file)"
    - "WR-22 ship decision -- UAT round 2 Test 2 pass, option (b), the recorded acceptance (see ## Acknowledged Gaps)"
  human_items_still_open:
    - "SC-5 sufficiency read-through: RE-OPENED by G-36-1 (formerly Test 6, once recorded as closed) -- its round-1 pass came from a reader already taught the Period/Grace values by Test 3 in the same session, so it was not a sufficiency measurement; must be re-run from the fresh-host setup steps alone, BEFORE the live heartbeat re-run below. HOLD RELEASED: its release condition (gap G-36-4 closed) is confirmed by this pass -- step 2 now names the flat LCO_API_KEY assignment Django can actually import, and the fold reaches both facilities. This item is now administrable."
    - "Live heartbeat dead-man re-run (formerly Test 3): configure a live healthchecks-compatible check from the fresh-host setup steps alone and confirm late ~15 min / alert ~35 min; run AFTER the SC-5 sufficiency read-through above, since it teaches the values that read-through measures"
    - "CR-03 credential-in-docs-build decision: decide whether the docs-build render of local_settings.py is fixed (autoapi_ignore) or accepted before shipping -- see Human Verification item 3"
gaps: []
deferred: []
advisory:
  - finding: "NEW (this pass, plan 36-08): the slice-scoped runbook gate that proves step 2 carries the flat name and no bracketed dict subscript is a TASK-TIME shell gate, not a committed regression test. No file under solsys_code/ or src/ references the 'Setting it up on a fresh host' subsection, and no pre-commit hook checks it, so a future edit that reintroduces a copy-pasteable nested key path into the procedure would ship unblocked. This is the same durability shape that let G-36-4 through (plan 36-07's gate was also task-time and token-level)."
    category: other
    reason: "NOT a must-have breach: plan 36-08 truth 8 is worded 'fails the task instead of shipping', and the gate did run at task time and was re-run independently and green by this verification. Recorded because the protection is per-edit, not permanent. Cheapest durable fix: a SimpleTestCase that awk-slices the runbook subsection and asserts the same two clauses (contains ``LCO_API_KEY``, contains no ``[`'\"]`` dict subscript), which would cost ~15 lines and run in the existing suite."
    evidence_status: "reproduced: `grep -rln \"Setting it up on a fresh host\" solsys_code/ src/ --include=*.py` returns nothing; .pre-commit-config.yaml has no runbook-content hook"
  - finding: "NEW (this pass, plan 36-08): runbook step 2 says local_settings.py 'can only ASSIGN new settings', while the same step also instructs the operator to put the real EMAIL_BACKEND there -- which is an override of an existing top-level setting, not a new one. The operative distinction (top-level assignment works, mutating into an already-built dict does not) is stated correctly in the very next clause."
    category: other
    reason: "Copy-edit class; no operator action is misdirected and nothing fails. Recorded only so the SC-5 sufficiency read-through's reader-confusion signal is not attributed to something unknown. Resolved by 'can only assign settings by name at the top level'."
    evidence_status: "source-confirmed: runbook:1458-1465 read directly"
  - finding: "NEW (this pass, plan 36-08): TestBracketedDictSubscriptRaisesNameError executes a bare literal string in an empty namespace, so it pins Python's own name-resolution semantics rather than FOMO's ``except ImportError`` guard. It can only fail if Python changes."
    category: other
    reason: "Satisfies plan 36-08 truth 6 exactly as that truth is worded ('executed in a module namespace of the same model local_settings.py gets'), and the other three cases in the module are real behavioural pins against the live fold tail. Recorded because this one case contributes near-zero regression protection; pinning the guard itself (that a NameError from the imported module propagates past ``except ImportError``) would."
    evidence_status: "source-confirmed: solsys_code/tests/test_settings_api_key_fold.py:106-115 read directly; all 4 cases run OK"
  - finding: "CR-03 (36-REVIEW.md iter 4): docs/conf.py:62-63 sets ``autoapi_dirs = ['../src']`` with an ``autoapi_ignore`` that does not exclude ``local_settings.py``, so sphinx-autoapi and sphinx.ext.viewcode render that file -- the credential home runbook step 2 mandates -- verbatim into generated HTML. Re-confirmed present in this working tree by this pass: ``_readthedocs/html/autoapi/fomo/local_settings/index.html``, ``_readthedocs/html/_modules/fomo/local_settings.html`` and ``docs/_build/html/autoapi/fomo/local_settings/index.html`` each contain one UUID-shaped string (this host's live ping token). Both trees are .gitignore'd (.gitignore:76-77), so nothing reached git, and ReadTheDocs builds from a checkout with no local_settings.py, so the published site is unaffected."
    category: security
    reason: "Does NOT falsify SC 4 as written ('no API key or password appears in any log line, notification or error message the unattended path PRODUCES') -- a docs-build artifact is none of those three. The root cause file (docs/conf.py) was never touched by phase 36. Recorded rather than gated because it is real, reproducible and worth a decision before shipping: the fix is one line (add '*/local_settings.py' to autoapi_ignore), plus deleting the two build trees and rotating the check if either build was ever served or copied, plus correcting the runbook's 'never go into a committed file' boundary claim at :1500-1501, which is scoped to the wrong boundary."
    evidence_status: "reproduced this pass: all three rendered pages located on disk; UUID-shaped string counted in each without quoting any value"
  - finding: "WR-24 (36-REVIEW.md iter 4): step 3 says 'Create ONE check for this schedule and set both of its settings', then offers the Cron-type ``*/15 * * * *`` alternative inside the same sentence and states 'The service alerts at last ping + expected interval + grace'. The canonical paragraph this step defers to says the Cron route 'pegs lateness to the wall-clock slot instead of to the last ping', and a Cron-type check has no Period field -- so on the route step 3 recommends as drift-free, both 'both of its settings' and the stated baseline are wrong."
    category: other
    reason: "Plan 36-07 truth 4 explicitly permits ONE line of arithmetic in the new step, so this is not a must-have breach. But the operator who takes the Cron route will mis-predict the alert baseline. Resolved by attaching the formula to the Simple-check route and describing the Cron route as 'missed slot + grace, no interval to set'. Untouched by plan 36-08 (its scope fence deferred every advisory by name)."
    evidence_status: "source-confirmed: both passages read directly; unchanged since the prior pass"
  - finding: "WR-25 (36-REVIEW.md iter 4): step 3's opening relative clause (runbook:1471-1474) attaches to 'one failed outright', asserting that outright failure is the class FOMO's own error handling cannot report. The code says the reverse -- run_tick() reports an outright failure via _send_notification() and the /<exit-code> ping; the class FOMO cannot report is the tick that never started or hung."
    category: other
    reason: "Cosmetically a misplaced modifier, substantively an inversion of the whole justification for creating the check. The operator's action is unaffected (they still create the check), so it does not falsify a must-have."
    evidence_status: "source-confirmed; unchanged since the prior pass"
  - finding: "WR-26 (36-REVIEW.md iter 4, re-raising iter 3's WR-21): check_state_dir() is a THIRD hard prerequisite (check_unattended.py:177, appended at :414), but step 1 says 'Create the two directories' and step 6's enumeration lists neither FOMO_STATE_DIR nor check_flock()'s -E/util-linux 2.27 probe, still summarising the hard set as '(flock, the directories, or email)'."
    category: other
    reason: "Not a must-have breach: FOMO_STATE_DIR defaults to FOMO_LOCK_DIR (settings.py:423-425), so a fresh host following the runbook passes. A host that points it elsewhere gets a hard preflight failure naming a directory the runbook never mentions."
    evidence_status: "re-confirmed this pass: `grep -rn FOMO_STATE_DIR docs/ --include=*.rst` still returns nothing; check_state_dir() still hard and still appended to results"
  - finding: "WR-27 (36-REVIEW.md iter 4): runbook step 8 (:1550-1553) says starting from the template and replacing 'its two placeholder paths' means 'either route produces the same line'. cron_line() (check_unattended.py:309-321) interpolates FIVE resolved values -- sys.executable, manage.py, ``shutil.which('flock')``, FOMO_LOCK_DIR and FOMO_LOG_FILE. On a host whose flock is elsewhere the template route installs a line whose lookup fails with exit 127 (not 99, so no 'lock held' line is written either) and every tick is a silent no-op."
    category: other
    reason: "The printed line -- the runbook's primary route -- is correct, so the documented happy path works; the claim that the hand-edit route is equivalent is what is false. Pre-existing from 36-04/36-05. Not a must-have breach."
    evidence_status: "re-confirmed this pass: the 'either route produces' sentence still at runbook:1552; cron_line() body unchanged"
  - finding: "IN-25 / IN-26 / IN-27 / IN-28 (36-REVIEW.md iter 4): the 15/20/35 triple appears in five places across the two files; 'Name each concept first, giving healthchecks.io's spelling in parentheses' (runbook:1484-1485) and the pre-existing 'Name the concept first:' are drafting directives surviving into operator prose; deploy/cron/fomo.crontab.example still says '[ $? -eq 99 ]' at :32 and :53 while the line itself uses '[ $rc -eq 99 ]' at :56, and still calls deploy/logrotate/fomo.example '(a later plan in this phase)' at :64 though it shipped; runbook :1551 and :1554 mark the two deploy paths with single backticks, which Sphinx renders as italic title references, unlike every other path in the subsection."
    category: other
    reason: "Copy-edit class; none falsifies a success criterion or a plan must-have truth. Recorded so the ship decision sees them."
    evidence_status: "re-confirmed this pass by direct grep on both files"
  - finding: "WR-22 (36-REVIEW.md iter 3): two logger.debug() sites on the unattended path interpolate a raw exception message rather than its class name -- backfill_lco_observations.py:349 (a live, authenticated LCO portal call) and unattended.py:191 (a bare `except Exception` around FOMO's own reconcile_run()). Inert under the shipped configuration because settings.LOGGING pins the root logger to INFO, so no such line is emitted; live the moment anyone raises the level."
    category: security
    reason: "Falsifies plan 36-01 must-have truth 10 as literally written. Does NOT falsify SC 4, which is about what appears in a log line the path actually produces. SETTLED: the developer recorded the explicit acceptance (option (b)) in UAT round 2 Test 2 -- see ## Acknowledged Gaps. Not re-asked by this re-verification."
    evidence_status: "source-confirmed in earlier passes; both files byte-identical since (git diff 7d61d6b..HEAD touches neither); disposition accepted by the developer"
  - finding: "IN-20 (36-REVIEW.md iter 3): check_unattended's [ok] heartbeat detail names only the expected ping interval (Period), not the ~20-minute grace time, and the pinning test asserts only 'Period'. An operator acting on the preflight line alone sets Period=15 and leaves healthchecks.io's 1-hour default grace."
    category: other
    reason: "Plan 36-06 truth 6 asks only for an expected-ping-interval reminder, which is present, so this is not a must-have gap. Less pressing since plan 36-07: the create-and-configure step now sets both knobs before the preflight is ever run."
    evidence_status: "source-confirmed; not a must-have breach"
  - finding: "36-REVIEW.md iter 3 carries five further warnings (WR-16 cron exit-99 attribution, WR-17 runtime state-dir mail storm, WR-18 IN-02's skip reasons logged at a dropped DEBUG level, WR-19 check_email accepts dummy/locmem/filebased backends, WR-20 check_flock probe can hang)."
    category: other
    reason: "None falsifies a success criterion or a plan must-have truth; recorded here so the ship decision sees them."
    evidence_status: "review-reported; not independently re-tested by this verification"
  - finding: "BOOKKEEPING (this pass): .planning/REQUIREMENTS.md still marks SCHED-09 and DISCOVER-01 as '[ ]' / 'Gaps Found' (:46, :48, :122, :124). Neither was ever the blocked requirement -- commit a9683ca blanket-reverted all four Phase 36 requirements when G-36-4 was found, and plan 36-08's completion commit c04e245 restored only the two it declared (SCHED-08, SCHED-10). This verification finds SCHED-09 satisfied with one live external re-proof open and DISCOVER-01 fully satisfied."
    category: other
    reason: "A planning-artifact status marker, not a codebase defect -- it cannot falsify a truth, an artifact or a link. Raised so whoever ships the phase corrects the two markers rather than reading them as real open gaps."
    evidence_status: "reproduced: `git diff a9683ca^..c04e245 -- .planning/REQUIREMENTS.md` shows the blanket revert and the partial restore"
behavior_unverified_items:
  - truth: "An operator who configures the heartbeat check using only what the corrected runbook names gets a check that goes late about 15 minutes after a missed tick and alerts about 35 minutes after the last successful ping (plan 36-06 truth 1 -- G-36-3's failed truth, restored)."
    test: "Create a fresh healthchecks-compatible check pointed at FOMO_HEARTBEAT_URL, configuring ONLY what the fresh-host setup steps name (runbook:1471-1501, step 3): expected ping interval (Period) 15 minutes -- or a Cron-type check with */15 * * * * -- and grace time (Grace) about 20 minutes. Then disable the crontab line, simulating the scheduler never invoking the job."
    expected: "The check goes late about 15 minutes after the missed tick and alerts about 35 minutes after the last successful ping, while FOMO itself logs nothing and sends no email -- SC 3's second, independent layer."
    why_human: "The signal is produced by the external heartbeat service's own expected-interval-plus-grace timer, not by any FOMO code path. G-36-3 was only findable this way: every internal gate checked the runbook against decision D-12, which itself carried the conflation."
human_verification:
  - test: "SC-5 sufficiency read-through: a reader who has not read \"The two failure signals\" and has not been told the expected-interval/grace values reads only \"Setting it up on a fresh host\", top-down, and works the steps."
    expected: "The reader creates the check, sets both of its settings (Period 15 min, or Cron type */15 * * * *; Grace ~20 min), writes the flat LCO_API_KEY assignment into local_settings.py, and fills FOMO_HEARTBEAT_URL -- all without leaving the subsection or reading source (SC 5)."
    why_human: "Sufficiency at point of use is not observable by any token-presence gate, and the verdict is only valid from a reader not already taught the knowledge out of band -- this is why round-1 UAT Test 6 passed while G-36-1 was live."
    hold_status: "RELEASED. The blocking condition (gap G-36-4, step 2 instructing an assignment that stopped Django from starting) is confirmed CLOSED by this pass: step 2 now names the flat LCO_API_KEY assignment, the subsection contains no bracketed dict subscript at all, the fold reaches both LCO and SOAR, and the live settings module imports cleanly. Administer this item first."
  - test: "Live heartbeat dead-man re-run: re-run UAT Test 3 against a live healthchecks-compatible check configured ONLY from the fresh-host setup steps (expected interval / Period 15 min, or Cron type */15 * * * *; grace / Grace ~20 min), then disable the crontab line. Run from the fresh-host setup steps, and AFTER the SC-5 sufficiency read-through above, because it teaches the values that read-through measures."
    expected: "Check goes late ~15 min after the missed tick and alerts ~35 min after the last successful ping, with FOMO logging nothing and mailing nothing."
    why_human: "External service behaviour; no automated gate can reach it. This is the only proof that closes G-36-3 end-to-end."
  - test: "Decide on CR-03 before shipping: the project's own sphinx-build pre-commit hook renders src/fomo/local_settings.py -- the credential home runbook step 2 mandates -- into _readthedocs/html/ and docs/_build/html/, and all three rendered pages in this working tree currently contain this host's ping token. Decide between fixing it (add '*/local_settings.py' to docs/conf.py's autoapi_ignore, delete both build trees, rotate the heartbeat check if either build was ever served/copied/shared, and correct the runbook's 'never go into a committed file' claim at :1500-1501 to name the docs-build boundary) and recording an explicit acceptance."
    expected: "Either the one-line autoapi_ignore fix plus tree deletion and the runbook boundary correction, or a recorded acceptance stating that no build is ever served from a configured host."
    why_human: "A judgment call on scope and blast radius, not a correctness question. SC 4 as written is about log lines, notifications and error messages the unattended path produces, and a docs-build artifact is none of those -- so this is not a phase-goal gap -- but it is a real on-disk credential exposure produced by following this phase's own runbook, and the .gitignore that currently contains it is the only thing between it and a public repository."
---

# Phase 36: Unattended Operation Verification Report

**Phase Goal:** The projector sweep, the LCO/SOAR discovery backfill and the reconciler run on the real host on a documented schedule with nobody typing anything, against a watched-proposal list an operator edits in the admin — and when it breaks, an operator finds out.
**Verified:** 2026-09-18T07:30:00Z
**Status:** human_needed
**Re-verification:** Yes — fourth pass, after gap-closure plan 36-08 (wave 6, G-36-4; commits `62d4d78`, `39312f4`, `96701a3`, metadata `c04e245`). Plan 36-08's twelve must-have truths were established from its frontmatter and checked against the files; the SC-5 / truth-35 / truth-57 verdicts that pass 3 marked FAILED were re-tested from scratch; the 36-01…36-07 truths were regression-checked. `git diff --stat 7d61d6b..HEAD` over `solsys_code/ src/ docs/ deploy/` returns exactly **three** files — `docs/runbooks/telescope_runs_calendar.rst` (+12/-2), `solsys_code/tests/test_settings_api_key_fold.py` (new, 125 lines) and `src/fomo/settings.py` (+2/-0) — with a clean working tree, so the regression surface was bounded and verifiable.

**Headline:** **G-36-4 is genuinely closed, and closed at the level the gap demanded — in code, not only in prose.** The runbook's step 2 now names the one assignment `local_settings.py` can actually make, the subsection contains no bracketed dict subscript anywhere for a reader to copy, and the `LCO_API_KEY` fold really does reach the SOAR facility — proven three independent ways: a new committed test that executes the *real* fold tail out of the live settings file, a live import of the actual settings module on this host reporting `api_key` present and non-empty for both `LCO` and `SOAR`, and the flat name now appearing in `docs/` where it previously appeared nowhere. **The phase advances from `gaps_found` to `human_needed`.** No gaps remain. Three items need a human: the SC-5 sufficiency read-through (whose hold is now released), the live external dead-man re-run, and the CR-03 ship decision.

---

## Goal Achievement

### Observable Truths

#### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Projector sweep, LCO/SOAR discovery backfill and reconciler all run on their documented recurring schedule with no operator action, and two invocations of the same job never overlap | ✓ VERIFIED | Unchanged by this round — `git diff --stat 7d61d6b..HEAD -- solsys_code/` lists only the **new test module**; `unattended.py`, `run_unattended.py`, `check_unattended.py` are byte-identical. Carried live evidence stands: `run_unattended --dry-run` ran all four steps in D-01 order against the real dev DB and exited 0; overlap proven cross-process with the same `/usr/bin/flock -n` binary the crontab invokes; **UAT Test 2 passed** on the real host (three START/END banners ~15 min apart in `/var/log/fomo/unattended.log`). `solsys_code.tests.test_unattended` re-run by this verification against the changed settings module: **63 tests, OK** |
| SC2 | Adding a proposal in the admin is enough for its robotically scheduled observations to start appearing — discovery takes no per-invocation arguments and needs no redeploy | ✓ VERIFIED | Unchanged. `--proposal` is `required=False, default=None`; the bare path sweeps `watched_rows()` via the shared `sweep_watched_rows()`, the same helper `unattended.step_discovery()` calls. Admin add + `list_editable` toggle proven to change what `watched_rows()` returns (live POST spot-check, pass 2) |
| SC3 | A failed unattended run reaches an operator two independent ways: a notification from the command itself, and a heartbeat that also fires when the scheduler never invoked the job at all | ✓ VERIFIED (live dead-man re-proof is human item 2) | Email layer: `unattended._send_notification()` → `notifications.notify_staff()`; the `TestNotification` cases green in today's 63-test run; **UAT Test 4 passed** against real SMTP. Heartbeat layer: `ping_heartbeat()` pings `/start` before the first step and `/<exit-code>` after the last; `TestHeartbeat` asserts both orders. The dead-man half was empirically confirmed by the operator in UAT round 1 once the interval was set. `ping_heartbeat()` untouched by this round |
| SC4 | No API key or password appears in any log line, notification or error message the unattended path produces | ✓ VERIFIED (two ⚠️ warnings — WR-22 settled, CR-03 open) | `TestCredentialHygiene` (8 tests, green in today's run) seeds a fake LCO api_key, a fake `EMAIL_HOST_PASSWORD` and a fake heartbeat URL, forces failure on every step plus the mail send plus the heartbeat ping, and asserts none reaches the captured log, stdout, stderr, `mail.outbox`, or `WatchedProposal.last_run_summary`. Every `warning`/`error` site logs `type(exc).__name__`. **Re-checked for this round's three files:** zero UUID-shaped strings in the runbook, `src/fomo/settings.py`, the new test module or the crontab template (`grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` → 0, 0, 0, 0); the runbook's API-key example is the placeholder `'<your key>'` and its only heartbeat URL is `https://hc-ping.com/<uuid>`; the new test's fixture literal is `'fake-portal-key-test-settings-api-key-fold'`, deliberately neither key-shaped nor UUID-shaped. **Warning 1 (settled):** WR-22's two `DEBUG` sites format `str(exc)`; root logger pinned to `INFO`, developer acceptance recorded. **Warning 2 (open):** CR-03 — the docs build renders `local_settings.py` into HTML; re-confirmed on disk this pass. SC 4 as written is not breached |
| SC5 | An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source | ✓ VERIFIED (point-of-use sufficiency is human item 1) | **Was FAILED in pass 3; the defect is gone.** Step 2 (`runbook:1458-1470`) now reads: "Write the API key as a flat, top-level assignment — ``LCO_API_KEY = '<your key>'`` — because this module is imported into its own namespace, so it can only ASSIGN new settings: reaching into a setting already built above it raises ``NameError``, which the import guard does not catch, and Django then refuses to start at all", followed by what `settings.py` folds it into, why one key covers both facilities, and what omitting it leaves behind. Independently re-tested by this pass, not read from the SUMMARY: (a) the 112-line awk-sliced subsection contains **zero** bracketed dict subscripts of any kind — `grep -E "[A-Za-z_]+\[['\"]"` over the slice → no match — so nothing nested is copy-pasteable out of the procedure; (b) `LCO_API_KEY` now appears in `docs/` (`runbook:1462`), where pass 3 found it appeared nowhere; (c) the instruction is *true*: `src/fomo/settings.py:440-443` folds the flat name into both facility entries, and a live import of the actual settings module on this host reports `api_key` present and **non-empty for both `LCO` and `SOAR`** (key names and a boolean printed; no value ever read into this record); (d) the failure mode the old text caused is now pinned by an executable test case. The heartbeat half remains right (36-07 truths below). What is left is sufficiency for a naive reader, which no token gate can measure → human item 1, hold released |

#### Plan 36-08 must-have truths (G-36-4 fresh-host API-key step) — **the new work**

Established from `36-08-PLAN.md` frontmatter and checked against the files at HEAD. The slice below is
`awk '/^Setting it up on a fresh host$/,/^Adding a proposal to watch$/'` over the runbook — 112 lines,
re-extracted by this verification.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 61 | An operator who follows step 2 verbatim gets a Django that starts: the step names the flat top-level `LCO_API_KEY`, and the procedure contains no bracketed settings-dict subscript an operator could copy | ✓ VERIFIED | Slice line 15 carries `` ``LCO_API_KEY = '<your key>'`` ``. Negative gate re-run two ways: `grep -nE "FACILITIES\[\|\['LCO'\]\|\['SOAR'\]"` → no match, and the broader `grep -nE "[A-Za-z_]+\[['\"]"` (ANY bracketed string subscript) → **no match** across all 112 slice lines. The live settings module imports cleanly on this host with a real `local_settings.py` present (Django `setup()` succeeded; 97 tests ran against it) |
| 62 | The step says WHY the name is flat, in one actionable clause: own namespace → reaching into an already-built setting raises `NameError` → the import guard does not catch it → Django refuses to start | ✓ VERIFIED | `runbook:1461-1465` read directly, all four links present in one sentence. Matches the code exactly: `src/fomo/settings.py:431-434` is `try: from fomo.local_settings import *` / `except ImportError: pass`, and `:436-439`'s own comment states the same mechanism |
| 63 | The step says which facilities the one key reaches, and it matches the code: one portal credential, one setting name | ✓ VERIFIED | `runbook:1466-1468` "folds that one key into both the LCO facility entry and the SOAR facility entry, because SOAR authenticates against the same LCO Observation Portal". Code: `settings.py:440-443` assigns `LCO_API_KEY` into `FACILITIES['LCO']['api_key']` **and** `FACILITIES['SOAR']['api_key']`; `settings.py:240-247` documents the SOAR entry as mirroring LCO against the same portal (D-05). `grep -rn "SOAR_API_KEY"` across `.py`/`.rst`/`.example` → **no match**, so no second name was introduced |
| 64 | The SOAR half is true in code, proven by a committed test that executes the real fold rather than grepping for a setting name | ✓ VERIFIED | `solsys_code/tests/test_settings_api_key_fold.py` read line-by-line: `_run_fold()` locates the live settings module via `DJANGO_SETTINGS_MODULE`, slices its source from the literal `try:\n    from fomo.local_settings import *` anchor to EOF, and `exec`s that real slice against a synthetic namespace with an injected `sys.modules['fomo.local_settings']`. `TestFlatKeyReachesBothFacilities` asserts both entries carry the key. **Run by this verification: 4 tests, OK.** Corroborated live: the real settings module on this host reports `api_key` non-empty for both facilities |
| 65 | `SOARSettings('SOAR').get_setting('api_key')` — the accessor the unattended status-refresh step reaches through `SOARFacility` — reads the entry the fold fills | ✓ VERIFIED | `TestSoarAccessorReadsFoldTarget` (green) pins accessor → `FACILITIES['SOAR']['api_key']`; `TestFlatKeyReachesBothFacilities` pins fold → the same key; the chain meets at that key name. Consumer end confirmed in source: `unattended.py:37` imports `SOARFacility` and `:268` calls `_refresh_one_facility(SOARFacility())`. End-to-end closure on the real host from the live import above |
| 66 | The old instruction's failure mode is pinned by an executable case: the bracketed dict-subscript assignment, executed in a module namespace of the same model, raises `NameError` | ✓ VERIFIED (⚠️ near-tautological — see advisory) | `TestBracketedDictSubscriptRaisesNameError` (green) asserts `exec("FACILITIES['LCO']['api_key'] = 'placeholder'", {})` raises `NameError`. Satisfies the truth as worded. Advisory records that this case pins Python's own name resolution rather than FOMO's `except ImportError` guard, so its regression value is near zero |
| 67 | An operator who omits the setting gets the documented no-op: the presence guard skips, both entries stay empty strings, nothing raises, Django starts | ✓ VERIFIED | `TestAbsentKeyIsCleanNoop` (green) executes the real fold tail with an empty injected module and asserts both entries remain `''`. Matches `settings.py:440`'s `if 'LCO_API_KEY' in globals():` guard and the defaults at `:236-247`. Documented at `runbook:1468-1470` including the consequence ("any portal call FOMO makes — including the unattended tick's status refresh — goes out unauthenticated") |
| 68 | The gate has teeth for THIS class and is slice-scoped: the subsection must contain the flat name as a placeholder assignment, must name `NameError`, and must contain no bracketed settings-dict subscript at all | ✓ VERIFIED (task-time gate; ⚠️ not committed — see advisory) | Re-run independently by this verification against the live file: placeholder assignment PRESENT, `NameError` PRESENT, `same LCO Observation Portal` PRESENT, bracketed subscript **absent under both the narrow and the broad pattern**. The truth is worded "fails the task instead of shipping" and the task gate did run; advisory 1 records that no committed test or hook enforces it for future edits |
| 69 | Plan 36-07's slice gate still passes unchanged after this edit: presence of all seven tokens plus a ninth numbered step, and the `Period`/ping-URL lines still precede the export anchor | ✓ VERIFIED | Re-run from scratch. **Presence:** `Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`, `healthchecks.io`, `The two failure signals`, `35 min` — all PRESENT; `^9\. ` PRESENT. **Order inside the slice:** `LCO_API_KEY` = 15, first `Period` = 39, ping-URL placeholder = 51, export anchor `the environment the cron daemon sees` = 55 → 39 < 55 and 51 < 55. **No-re-copy gate:** the slice still does NOT contain `bounds the maximum allowed gap` while the file still does. `git diff -U0` confirms the runbook change is a single hunk entirely inside step 2 |
| 70 | No committed file gains a credential VALUE; the runbook shows a bracketed placeholder assignment only | ✓ VERIFIED | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` → **0** in `docs/runbooks/telescope_runs_calendar.rst`, `src/fomo/settings.py`, `solsys_code/tests/test_settings_api_key_fold.py` and `deploy/cron/fomo.crontab.example`. The documented right-hand side is the literal `'<your key>'`; the test fixture literal is `'fake-portal-key-test-settings-api-key-fold'`, deliberately non-key-shaped. Nothing was written to or read out of `src/fomo/local_settings.py` — the test injects into `sys.modules` and never opens that path |
| 71 | The unattended path itself is untouched, and both regression modules stay green | ✓ VERIFIED | `git diff --name-only 7d61d6b..HEAD -- solsys_code/` lists exactly one file, the **new** test module — so `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py`, `test_unattended.py` and `test_check_unattended.py` are byte-identical. `python manage.py test solsys_code.tests.test_unattended solsys_code.tests.test_check_unattended` re-run by this verification: **Ran 93 tests — OK** |
| 72 | `36-VERIFICATION.md`'s SC-5 hold sites name plan 36-08 and state a release condition, while every verdict, score and status marker was left as the verifier wrote it | ✓ VERIFIED | `git diff 96701a3^..96701a3` inspected hunk by hunk: exactly **three** hunks, at the two frontmatter sites and the prose item, each replacing the HOLD language with a RELEASE CONDITION naming plan 36-08 and the fix. No `status:`, `score:`, `gaps:` entry or verdict cell touched. (This pass now owns those verdicts and rewrites them — as it should) |

#### Plan 36-07 must-have truths (G-36-1 fresh-host heartbeat setup step)

Re-checked against the files at HEAD after 36-08's step-2 edit — the two plans had to coexist.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 52 | Reading only the fresh-host subsection top-down, the operator reaches the `FOMO_HEARTBEAT_URL` export already knowing what the heartbeat is, what hosts the check, that they must create one, both values, and which URL to copy | ✓ VERIFIED (point-of-use sufficiency is human item 1) | Slice lines 24-54 are step 3; line 55 is the export step's anchor. The inversion is gone and 36-08's edit did not disturb it |
| 53 | The service class and how the operator obtains the ping URL are in the procedure, the URL in placeholder form | ✓ VERIFIED | `runbook:1479-1482` "Any healthchecks-compatible service works: healthchecks.io's hosted free tier, or a self-hosted ``healthchecks`` instance"; `:1496-1501` "copy that check's own ping URL from the service… the form ``https://hc-ping.com/<uuid>``… The ``<uuid>`` part is the ping token, so this URL is itself a credential" |
| 54 | Each environment variable has its own numbered step; the heartbeat export carries hygiene only and consumes the previous step's URL; `FOMO_BASE_URL` keeps its treatment verbatim | ✓ VERIFIED | `runbook:1502-1506` (export + "This is the ping URL the check in the previous step produced") and `:1507-1516` (`FOMO_BASE_URL`, unchanged) |
| 55 | The alert-window reasoning still has exactly ONE home; the new step states the values and points at "The two failure signals" | ✓ VERIFIED (⚠️ see WR-24) | Slice does **not** contain `bounds the maximum allowed gap`; the file still does. The step carries the permitted single arithmetic line plus the cross-reference at `:1492-1494` |
| 56 | `deploy/cron/fomo.crontab.example` agrees: the variable listing says where the ping URL comes from, and BOTH "full setup" pointers name the subsection | ✓ VERIFIED | `grep -cF 'Setting it up on a fresh host'` → **3** (:16, :37, :62). No value anywhere; no UUID-shaped string |
| 57 | Step 2 names the nested `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` structure, and the two steps that need root say so | ✓ **PASSED (superseded by plan 36-08)** | The **sudo half still holds**: slice :4-6 ("creating them under ``/var`` and handing ownership to that account typically needs ``sudo``") and :99-102 ("Writing into ``/etc/logrotate.d/`` typically needs ``sudo`` too"). The **API-key half was deliberately reversed** by approved gap-closure plan 36-08 (`gap_closure: true`, `gap_ids: [G-36-4]`), whose `missing` list in pass 3's gap record said in terms: "Replace the nested key path with the flat name the fold actually reads." Truth 57 was itself the defect — token-true, purpose-false — so its reversal is the closure, not a deviation. Recorded as superseded and counted toward the score; **no override was self-accepted.** To record it formally, see the suggested `overrides:` block in the Gaps Summary |
| 58 | `check_heartbeat()`, its `[ok]` reminder text and `test_set_heartbeat_reminds_about_the_check_period` are untouched and green | ✓ VERIFIED | `check_unattended.py` byte-identical since 7d61d6b; `test_check_unattended` → **30 tests, OK** in this verification's 93-test run |
| 59 | `36-VERIFICATION.md` scripts the SC-5 sufficiency read-through BEFORE the live heartbeat test, and records that a sufficiency verdict from a reader already taught the knowledge out of band is not evidence | ✓ VERIFIED | Preserved through both 36-08's edit (marker order re-checked at commit time: SC-5 first at frontmatter 71 and prose 446, ahead of the live re-run at 72 and 462) and through this rewrite: SC-5 read-through is `human_verification` entry 1 and prose item 1; the live dead-man re-run is entry 2 and item 2; the out-of-band-contamination rule is restated in both |
| 60 | The gate has teeth in BOTH dimensions, presence and ORDER | ✓ VERIFIED | Re-run independently against the live file — see truth 69. Supporting gates re-run by this verification: `pre-commit run ruff --all-files` → **Passed**; `pre-commit run ruff-format --all-files` → **Passed**; read-only `docutils` parse of the whole runbook at `report_level=2` → 11 messages, **0 structural** (no enumerated-list, indentation, block-quote, literal-block, explicit-markup or title-underline message); the 11 are the unknown-`:ref:`/`:doc:`-role notices bare docutils always emits |

#### Plan 36-01 must-have truths (runner, heartbeat, email, crontab, shared mailer)

Regression-checked: no file under `solsys_code/` changed except the new test module, and
`test_unattended` (63) + `test_check_unattended` (30) were re-run green by this verification
**against the modified settings module**.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All-succeeding tick exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_healthy_tick_exits_zero`, `test_pings_start_then_exit_code`, `test_empty_database_tick_is_healthy` |
| 2 | Failing tick exits non-zero, pings `/start` then `/<exit-code>`, sends exactly one email to every staff user with an email | ✓ VERIFIED | `test_step_failure_sets_exit_code`, `test_failing_tick_mails_staff_once` |
| 3 | Second consecutive failing tick with the same failing set sends no second email; a later success sends exactly one `FOMO unattended run recovered` | ✓ VERIFIED | `test_repeat_failure_is_suppressed`, `test_recovery_mails_once`; state crosses processes via the real JSON file, written atomically (IN-03) |
| 4 | A raising heartbeat ping never changes the exit code, and neither the URL nor the exception message reaches log/stdout/stderr/email | ✓ VERIFIED | `test_ping_failure_never_fails_the_tick`, `test_heartbeat_ping_failure_leaks_nothing` |
| 5 | With `FOMO_HEARTBEAT_URL` unset: one INFO line per tick, no HTTP call, exit code unaffected | ✓ VERIFIED | `test_unset_url_skips_pinging`; `ping_heartbeat()` early return |
| 6 | Failure subject is `FOMO unattended run failed: <step(s)>`; body carries each failed step's summary, the log path and admin/calendar links — no traceback, no request URL, no portal response text | ✓ VERIFIED | `_build_notification_body()`; `test_failure_email_body_carries_no_secret_and_no_traceback` |
| 7 | A second invocation while the first holds the lock runs no step and logs one skip line | ✓ VERIFIED | `test_contended_lock_skips_every_step` plus pass 2's cross-process `/usr/bin/flock -n` check |
| 8 | A tick with nothing to do anywhere exits 0, pings `/start` then `/0`, sends no email | ✓ VERIFIED | `test_empty_database_tick_is_healthy`; confirmed live in pass 2 |
| 9 | Step sequence fixed and identical every tick regardless of failures | ✓ VERIFIED | `STEPS` is a single module-level tuple; `test_all_four_steps_run_in_order`, `test_step_failure_does_not_abort_the_tick` |
| 10 | (backstop) No unattended-path log/email/stdout/stderr write formats the message of an exception caught from a `requests` call, a facility/portal call, or `send_mail()` — only its class name | ⚠️ UNCERTAIN (settled acceptance) | Holds at every `warning`/`error` site and is behaviourally backed by the 8 green `TestCredentialHygiene` tests. Two `DEBUG` sites format `str(exc)` (WR-22). Inert as shipped (root logger `INFO`). **The developer recorded the explicit acceptance in UAT round 2 Test 2 (option (b))** — see ## Acknowledged Gaps. Left UNCERTAIN for traceability; not re-asked. Both files byte-identical since pass 3 |

#### Plan 36-02 must-have truths (WatchedProposal + watched-list discovery)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | Staff can add a proposal code in the admin and toggle `is_active` from the changelist, no redeploy, no per-invocation argument | ✓ VERIFIED | `WatchedProposalAdmin` `list_display`/`list_editable`/`list_filter`; `WatchedProposalAdminTests`; pass-2 live HTTP POST spot-check |
| 12 | Bare `backfill_lco_observations` sweeps every active row, applying that row's `target_list_name` and `attributed_to` | ✓ VERIFIED | `sweep_watched_rows()` called from `handle()`; `TestWatchedListSweep`; notebook output `Swept 2 watched proposal(s), failed: 0` |
| 13 | `--proposal X` still sweeps exactly X with the existing flags, and X need not be a watched row | ✓ VERIFIED | Override branch never consults `WatchedProposal` |
| 14 | After a bare sweep every active row carries `last_run_at` and a `last_run_summary` of either the counter line or `failed: <ExceptionClassName>` | ✓ VERIFIED | `sweep_watched_rows()` writes both with `update_fields`; notebook shows real values |
| 15 | A portal error on one row is caught, recorded, counted, and the remaining rows are still swept | ✓ VERIFIED | Per-row try/except; notebook cell 20 shows row B still creating record 900402 after row A failed |
| 16 | Two rows can never share a `proposal_code` | ✓ VERIFIED | `unique=True` in model and migration 0022; `test_duplicate_code_rejected` |
| 17 | Zero active rows → one INFO line, nothing written, exit 0 | ✓ VERIFIED | `TestEmptyWatchedList`; observed live in pass 2 |
| 18 | `proposal_code` equality is exact-match, case-sensitive; whitespace stripped on save | ✓ VERIFIED | `WatchedProposal.save()`; `test_code_is_stripped_on_save`, `test_code_comparison_is_case_sensitive` |
| 19 | Active rows are swept in deterministic `proposal_code` ascending order | ✓ VERIFIED | `Meta.ordering = ['proposal_code']` in model and migration |

#### Plan 36-03 must-have truths (the remaining three steps, SCHED-10 suite)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 20 | One tick runs all four steps in the fixed order, in one process, no `manage.py` subprocess | ✓ VERIFIED | `STEPS` tuple; no `call_command`/`subprocess`/`Popen`/`os.system` in `unattended.py` or `run_unattended.py` |
| 21 | Status refresh calls `update_all_observation_statuses()` on a fresh `LCOFacility()` and a fresh `SOARFacility()` — never shared — and a non-empty failure list is a step failure | ✓ VERIFIED | `unattended.py:267-268` constructs each per tick; `test_calls_both_facilities_with_fresh_instances`, `test_non_empty_failure_list_is_a_step_failure`, `test_facility_exception_is_isolated_per_facility`. **Now materially better than at pass 3:** the `SOARFacility()` leg finally has a real credential to authenticate with (truths 63-65) |
| 22 | A status-refresh failure is logged as `observation_id` plus the exception class name; TOM's message half never reaches a log line, stdout/stderr, or the email | ✓ VERIFIED | `_refresh_one_facility()` discards `_message` at the unpack; two named tests |
| 23 | A raising step never prevents later steps; every registry step is attempted every tick | ✓ VERIFIED | `run_tick()` per-step try/except; two named tests |
| 24 | Projector sweep calls `project_queryset()` directly with the same observed-site hook, counts `unprojectable` as a failure, never `call_command()` | ✓ VERIFIED | Three named tests; live sweep reported the real counter line |
| 25 | Discovery sweeps every active row and fails only when at least one row failed; zero rows logs one INFO line and is healthy | ✓ VERIFIED | `step_discovery()` → `sweep_watched_rows()`; four named tests |
| 26 | `unchanged` / `skipped` / `detach_declined` / `remint_declined` never make the tick non-zero and never trigger the email | ✓ VERIFIED | `test_expected_data_shape_outcomes_are_not_failures`; failure derives from typed counters, never from parsing stdout |
| 27 | With a fake api key, mail password and heartbeat URL seeded and every failure path forced in turn, none appears in logs/stdout/stderr/`mail.outbox` | ✓ VERIFIED | The 8-test `TestCredentialHygiene` class, green in today's run. (Scope caveat: captures at the default level — see truth 10) |

#### Plan 36-04 must-have truths (`check_unattended`, cron line, logrotate)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 28 | `check_unattended` reports every prerequisite in one run | ✓ VERIFIED | Pass-2 live run printed flock, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`, `FOMO_STATE_DIR`, `EMAIL_BACKEND`, `staff_recipients`, `heartbeat`, `FOMO_BASE_URL` and `watched_proposals` in one pass |
| 29 | A missing hard prerequisite makes the command exit non-zero and name which check failed | ✓ VERIFIED | Five named tests inside today's green 30 |
| 30 | Unset heartbeat URL and empty watched list are warnings, not failures; exit stays 0 when every hard check passed | ✓ VERIFIED | `hard=False` on both (`check_unattended.py:241/250`, `:365/369`); two named tests |
| 31 | The printed cron line carries the real resolved interpreter and `manage.py` path | ✓ VERIFIED (⚠️ see WR-27) | `cron_line()` resolves `sys.executable`, `manage.py`, `shutil.which('flock')`, `FOMO_LOCK_DIR`, `FOMO_LOG_FILE`; `test_line_has_real_paths`. WR-27 is about the *runbook's* claim that the hand-edit route is equivalent, not about this truth |
| 32 | Names and set/unset status only — never a value | ✓ VERIFIED | Three named tests including `test_output_never_contains_a_seeded_value` |
| 33 | `--send-test-email` sends one message through the configured backend to the same staff recipient list the failure notice uses | ✓ VERIFIED | `_send_test_email()` → `notifications.notify_staff()`; **UAT Test 4 passed** against real SMTP |
| 34 | The command only reads: creates no directory, writes no file, changes no row | ✓ VERIFIED | `test_command_writes_nothing` |

#### Plan 36-05 must-have truths (paired docs)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 35 | An operator can set up or verify the whole schedule from one runbook section without reading source | ✓ VERIFIED (sufficiency is human item 1) | **Was FAILED in pass 3 on the same root cause as SC 5; restored.** The API-key sentence now names the only assignment that works, states why, and states what the fold covers; `LCO_API_KEY` appears in `docs/` so no source read is needed to recover. See SC 5 for the full evidence chain |
| 36 | The section documents both failure signals — email (who, what, repeat, clear) and heartbeat (`/start`, `/<exit-code>`, expected ping interval plus grace time, alerting at last ping + interval + grace) — plus the "nothing has appeared" checklist | ✓ VERIFIED | Canonical paragraph read directly: both knobs, the arithmetic, the Cron alternative with its wall-clock-slot caveat, the grace's `/start`-to-completion role, and the 1-day-default trap. "When nothing has appeared" gives the four-step checklist, item 3 bounded by interval + grace |
| 37 | The backfill section documents `--proposal` as optional, the bare watched sweep, per-row overrides and per-row failure isolation | ✓ VERIFIED | All four points stated explicitly |
| 38 | The cheat-sheet carries rows for `run_unattended` and `check_unattended`, and `backfill_lco_observations` reflects its optional-argument contract | ✓ VERIFIED | Cheat-sheet rows present for all three |
| 39 | The overlap guarantee is stated at exactly its true strength | ✓ VERIFIED | "What the locking does and does not cover" names `run_unattended --step <name>` as the exclusive manual route and does not overclaim for directly-run sweep commands |
| 40 | The notebook contains executed cells seeding `WatchedProposal` rows, running the command bare, showing `last_run_summary`, and showing one proposal failing without stopping the other | ✓ VERIFIED | Real committed stdout; `execution_count` sequential across all code cells; unchanged this round |
| 41 | `docs/notebooks.rst` lists `backfill_lco_observations_demo` in the Demonstration Notebooks toctree | ✓ VERIFIED | Toctree line present; referenced file exists |
| 42 | `CLAUDE.md`'s notebook map records the pairing for every module this phase adds, including that the runbook section — not a notebook — is the paired doc for the runner and `check_unattended`, and why | ✓ VERIFIED | CLAUDE.md maps `unattended.py`, `notifications.py`, `run_unattended.py`, `check_unattended.py` → the runbook's "How do I run everything unattended?" section with the stated reason |
| 43 | No committed doc, notebook cell or runbook example quotes a live setting, a `local_settings.py` value, or any credential | ✓ VERIFIED | Re-grepped this round's three files: zero UUID-shaped strings; the runbook's API-key example is `'<your key>'` and its only URL is the `<uuid>` placeholder; the new test's fixture literal is deliberately non-key-shaped; the crontab keeps `/path/to/venv/bin/python` and `/path/to/checkout/manage.py`. (CR-03 concerns a *generated, gitignored* artifact, not a committed one — see the advisory) |

#### Plan 36-06 must-have truths (G-36-3 heartbeat guidance correction)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 44 | An operator configuring the check from the corrected runbook alone gets late ~15 min / alert ~35 min — not a day later | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | The values are in the setup step itself and the canonical paragraph, and 36-08 did not disturb them (order gate re-run green). But nobody has yet configured a live check from the procedure alone, which is the only way this class of defect is findable. → Human Verification item 2 |
| 45 | The guidance names BOTH knobs, states the arithmetic, and names the 1-day interval default as the trap | ✓ VERIFIED | Canonical paragraph read directly this pass |
| 46 | The Cron-type `*/15 * * * *` alternative is offered, and the grace stays at about 20 min with its `/start`-to-completion role explained | ✓ VERIFIED | Same paragraph; `Grace` ~20 min also restated in step 3 at `runbook:1489-1490` |
| 47 | The "When nothing has appeared" triage item judges a stale ping against expected interval + grace, not grace alone | ✓ VERIFIED | "older than the expected interval plus the grace time -- about 35 minutes with the recommended 15/20 settings" |
| 48 | A troubleshooting entry covers the exact symptom the gap produced | ✓ VERIFIED | Entry present with `^`-underlined title and **Cause:**/**Fix:** form |
| 49 | `deploy/cron/fomo.crontab.example` describes the backstop with both knobs, and `check_unattended`'s `[ok]` heartbeat line reminds the operator the remote check still needs its own expected ping interval | ✓ VERIFIED | crontab:34-38 read directly; the preflight detail string unmodified since pass 3 |
| 50 | The verification record's human-test script and its runbook evidence cell state the corrected time-to-alert, traceable to the gap id | ✓ VERIFIED | Carried into this rebuild; `re_verification.gaps_closed` cites G-36-3 and G-36-1 and G-36-4 by id |
| 51 | Sphinx still builds the runbook with no new warning and `test_check_unattended` stays green | ✓ VERIFIED | `test_check_unattended` → **30 tests, OK**. Docs structure re-checked read-only via `docutils` after 36-08's step-2 rewrite: 11 messages, **0 structural**. `ruff` and `ruff-format` both Passed |

**Score:** 75/77 truths verified — 0 failed; 1 present-but-behavior-unverified (truth 44); 1 uncertain-and-accepted (truth 10); 1 superseded-and-counted (truth 57, reversed by approved gap-closure plan 36-08).

---

### Plan 36-08 Prohibitions (must-NOT checks)

All seven are judgment-tier (`verification: flagged-unverified`). Per the fail-closed rule these carry a
**non-authoritative LLM-judge verdict** and are flagged for human review; none is silently passed.

| # | Prohibition | Judge verdict | Evidence | Flag |
|---|-------------|---------------|----------|------|
| P1 | No committed file may gain a credential VALUE; a bracketed placeholder assignment IS allowed | Satisfied | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` → **0** in the runbook, `src/fomo/settings.py`, the new test module and the crontab template. Documented right-hand side is the literal `'<your key>'`; test fixture is `'fake-portal-key-test-settings-api-key-fold'`. Residual (a hand-written fake key of another shape) inspected by eye: none | unverified-prohibition — human review recommended |
| P2 | Only two source-tree files change; the five named unattended modules and both existing regression modules are untouched, and `ping_heartbeat()` is not read for editing | Satisfied | `git diff --stat 7d61d6b..HEAD -- solsys_code/ src/ docs/ deploy/` → exactly 3 files (runbook, new test module, settings.py), clean working tree. `git diff --name-only ... -- solsys_code/` lists only the new test module, so all five named modules and both regression modules are byte-identical; 93 tests green | unverified-prohibition — human review recommended |
| P3 | Nothing is ever written to `src/fomo/local_settings.py` | Satisfied | The new test injects `types.ModuleType('fomo.local_settings')` into `sys.modules` and restores it via `addCleanup`; the only file it opens is the live settings module's own `__file__`. No commit touches that path (it is gitignored and absent from every diff). This verification likewise never opened it — the live-fold check printed key names and a boolean only | unverified-prohibition — human review recommended |
| P4 | No second flat credential name is introduced — there is no `SOAR_API_KEY` | Satisfied | `grep -rn "SOAR_API_KEY" --include=*.py --include=*.rst --include=*.example .` → **no match**. One setting, folded twice | unverified-prohibition — human review recommended |
| P5 | No advisory finding is folded in (CR-03, WR-24…WR-27, IN-20, IN-25…IN-28 stay untouched) | Satisfied | The runbook diff is a single hunk inside step 2. Every advisory re-checked this pass is still verbatim present: WR-24's Cron sentence, WR-25's relative clause, WR-26's missing `FOMO_STATE_DIR` (`grep -rn FOMO_STATE_DIR docs/ --include=*.rst` → nothing), WR-27's "either route produces the same line" at :1552, IN-27's `[ $? -eq 99 ]` at crontab :32/:53 and "(a later plan in this phase)" at :64, and `docs/conf.py` untouched. **This is the discipline whose absence created G-36-4, and it held** | unverified-prohibition — human review recommended |
| P6 | Inside the runbook only step 2's API-key sentences change; the rest of the nine-step procedure, the canonical "Heartbeat." paragraph, "When nothing has appeared" and both troubleshooting entries stay byte-identical | Satisfied | `git diff 62d4d78^..c04e245 -- docs/runbooks/telescope_runs_calendar.rst` is one hunk: two sentences removed at 1461-1462, ten lines added, all before step 3 at :1471. Nothing else in the 2000+-line file moved | unverified-prohibition — human review recommended |
| P7 | The docs build is not invoked as this plan's RST gate; a read-only `docutils` parse is used instead | Satisfied | This verification used the same read-only route (`publish_doctree` at `report_level=2`) and did not run `pre-commit run sphinx-build`, so neither the plan nor this pass wrote into the gitignored build trees CR-03 concerns | unverified-prohibition — human review recommended |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/fomo/settings.py` | `LCO_API_KEY` fold reaching BOTH the LCO and the SOAR facility entries | ✓ VERIFIED | `:440-443` — `if 'LCO_API_KEY' in globals():` then both assignments, with one comment line naming the shared portal. +2/-0 since pass 3. The live module imports cleanly and both entries carry a non-empty `api_key` on this host |
| `solsys_code/tests/test_settings_api_key_fold.py` | Committed test executing the REAL fold, not a re-implementation | ✓ VERIFIED | NEW, 125 lines, 4 `SimpleTestCase` cases, **all pass**. Slices the live settings source from a literal anchor and `exec`s it against an injected `sys.modules` entry. No skips, no writes, no DB, no `Target` fixture (so the `NonSiderealTargetFactory` rule does not arise). Value-level assertions throughout |
| `docs/runbooks/telescope_runs_calendar.rst` | Fresh-host procedure whose every step an operator can follow verbatim | ✓ VERIFIED | Step 2 rewritten (`:1458-1470`); nine numbered steps intact; heartbeat step at `:1471-1501` still ahead of the export at `:1502`; **no bracketed dict subscript anywhere in the 112-line subsection**; RST structure clean under `docutils` |
| `deploy/cron/fomo.crontab.example` | Ping-URL provenance + both "full setup" pointers naming the subsection | ✓ VERIFIED | 3 subsection references (:16, :37, :62); `healthchecks` present; no UUID; placeholders intact. Unchanged this round |
| `.planning/phases/36-unattended-operation/36-VERIFICATION.md` | Human-test order preserved, hold released, verdicts owned by re-verification | ✓ VERIFIED | Plan 36-08 changed exactly the three hold sites and no verdict; this pass now rewrites the verdicts and preserves the SC-5-before-live-heartbeat order |
| `solsys_code/unattended.py` | Runner, steps, lock, heartbeat, notification | ✓ VERIFIED | Byte-identical since 7d61d6b; 63 tests green |
| `solsys_code/notifications.py` | Shared request-free mailer | ✓ VERIFIED | Unchanged; called from `unattended.py`, `check_unattended.py`, `campaign_views.py` |
| `solsys_code/management/commands/run_unattended.py` | Cron entry point | ✓ VERIFIED | Unchanged |
| `solsys_code/management/commands/check_unattended.py` | Preflight + cron line | ✓ VERIFIED | Unchanged; 30 tests green; `[ok]` heartbeat reminder intact |
| `solsys_code/models.py` (`WatchedProposal`) | Admin-editable watch list | ✓ VERIFIED | Unchanged |
| `solsys_code/migrations/0022_watchedproposal.py` | Matching migration | ✓ VERIFIED | Unchanged |
| `solsys_code/admin.py` (`WatchedProposalAdmin`) | Registered, `list_editable` | ✓ VERIFIED | Unchanged |
| `solsys_code/management/commands/backfill_lco_observations.py` | `sweep_proposal()`, `watched_rows()`, `sweep_watched_rows()`, optional `--proposal` | ✓ VERIFIED | Unchanged |
| `deploy/logrotate/fomo.example` | Daily, rotate 14, copytruncate | ✓ VERIFIED | Unchanged; **UAT Test 5 passed** |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | Executed watched-proposal cells | ✓ VERIFIED | Unchanged this round |
| `docs/conf.py` | Sphinx/autoapi config | ⚠️ **ADVISORY (CR-03)** | `autoapi_dirs = ['../src']` at :62 with `autoapi_ignore = ['*/__main__.py', '*/_version.py']` at :63 — still no `local_settings.py` exclusion. Not touched by phase 36 |
| Tests (`test_unattended`, `test_check_unattended`, `test_settings_api_key_fold`) | Behavioral coverage | ✓ VERIFIED | **97 tests re-run green by this verification** (63 + 30 + 4) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| **runbook step 2's API-key instruction** | **`src/fomo/settings.py`'s `LCO_API_KEY` fold** | **the operator writing `local_settings.py`** | ✓ **WIRED** | **Was ✗ NOT WIRED in pass 3.** The runbook now names the flat `LCO_API_KEY`; the fold reads exactly that name from `globals()`. Same name on both ends, and the name is reachable by a top-level assignment in a star-imported module. Live proof on this host: the settings module imports and both facility `api_key` entries are non-empty |
| the fold | `SOARSettings('SOAR').get_setting('api_key')` | `FACILITIES['SOAR']['api_key']` | ✓ **WIRED** | **Was absent entirely in pass 3 (no SOAR fold existed).** `settings.py:443` writes the key; `TestSoarAccessorReadsFoldTarget` proves the accessor reads that entry; `unattended.py:268` is the consumer (`_refresh_one_facility(SOARFacility())`). Chain closed end to end |
| the new test | the real tail of `src/fomo/settings.py` | sliced from a literal anchor and `exec`'d | ✓ WIRED | `_FOLD_TAIL_ANCHOR = "try:\n    from fomo.local_settings import *"`, located in the live file via `DJANGO_SETTINGS_MODULE`; the assertion `anchor_index != -1` fails loudly if the anchor ever moves. Dropping the SOAR line breaks `test_flat_key_fills_lco_and_soar`, not a grep |
| the create-and-configure-the-check step | the export step that consumes its output | numbered-step order inside the subsection | ✓ WIRED | Slice line 39 (`Period`) and 51 (ping URL) both precede line 55 (`the environment the cron daemon sees`). Preserved through 36-08's edit |
| the new step's values | the canonical "Heartbeat." paragraph's reasoning | quoted-subsection cross-reference, no re-copy | ✓ WIRED (⚠️ one restated line drifted) | Cross-reference present; the `/start`-to-completion sentence absent from the slice and present in the file. WR-24 unchanged |
| the runbook's setup subsection | `deploy/cron/fomo.crontab.example`'s variable listing and both "full setup" pointers | shared subsection name | ✓ WIRED | All three crontab references name "Setting it up on a fresh host" |
| the runbook's preflight step | `check_heartbeat()`'s `[ok]` detail | `test_set_heartbeat_reminds_about_the_check_period` | ✓ WIRED | The triad untouched; all three agree and the test is green |
| `run_unattended.Command.handle()` | `unattended.run_tick()` → `STEPS` | direct call | ✓ WIRED | Unchanged; `--step` choices derive from the single `STEPS` tuple |
| `unattended.run_tick()` | `notifications.notify_staff()` | `_send_notification()` | ✓ WIRED | Unchanged |
| crontab template's `flock -n` path | `settings.FOMO_LOCK_DIR` | `<dir>/run_unattended.cron.lock` | ✓ WIRED | Unchanged; proven cross-process in pass 2 |
| `WatchedProposal.objects.filter(is_active=True)` | `sweep_proposal()` | `watched_rows()` → `sweep_watched_rows()` | ✓ WIRED | Unchanged; one shared helper, two callers |
| `step_status_refresh()` | Phase 34 `post_save` receiver | `facility.update_observation_status()` → `.save()` | ✓ WIRED | Unchanged |

---

### Data-Flow Trace (Level 4)

Source unchanged except the settings fold; carried from pass 2's live tick, plus one new row.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` | the folded key | the operator's `local_settings.py` `LCO_API_KEY`, folded at `settings.py:440-443` | Yes — live import on this host reports both entries present and non-empty (boolean only; no value read) | ✓ FLOWING |
| `step_project_sweep()` | `result['counters']` | `project_queryset()` over `ObservationRecord.objects.filter(facility__in=PROJECTED_FACILITIES)` | Yes — pass-2 live tick reported `unchanged: 159` | ✓ FLOWING |
| `step_reconcile()` | `run_count`/`failed_count` | `CampaignRun.objects.all()` | Yes — pass-2 live tick reported `runs: 45, failed: 0` | ✓ FLOWING |
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
| **The documented API-key assignment actually works** (pass 3's ✗ FAIL) | `python manage.py test solsys_code.tests.test_settings_api_key_fold` — executes the REAL fold tail against an injected local-settings module | Ran 4 tests — **OK**; flat key reaches both facilities, absent key is a clean no-op, bracketed form raises `NameError`, SOAR accessor reads the fold target | ✓ **PASS** |
| **A `SOAR` fold exists** (pass 3's ✗ FAIL) | read `src/fomo/settings.py:440-443`; live `django.setup()` then print `FACILITIES[f].keys()` and `bool(api_key)` for LCO/SOAR | `LCO ['api_key','portal_url'] present: True nonempty: True` / `SOAR ['api_key','portal_url'] present: True nonempty: True` — **no value printed or recorded** | ✓ **PASS** |
| No copy-pasteable nested key path survives in the procedure | `grep -nE "[A-Za-z_]+\[['\"]"` over the 112-line awk slice | **no match** (and the narrow `FACILITIES\[` pattern likewise) | ✓ PASS |
| The flat name is documented in `docs/` | `grep -rn "LCO_API_KEY" docs/ --include=*.rst` | `docs/runbooks/telescope_runs_calendar.rst:1462` — present where pass 3 found nothing | ✓ PASS |
| No `SOAR_API_KEY` second name was introduced | `grep -rn "SOAR_API_KEY" --include=*.py --include=*.rst --include=*.example .` | no match | ✓ PASS |
| Plan 36-07's slice presence gate still holds | awk slice + `grep -qF` ×7 + `grep -qE '^9\. '` | `Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`, `healthchecks.io`, `The two failure signals`, `35 min`, 9th step — all PRESENT | ✓ PASS |
| Plan 36-07's slice ORDER gate still holds | line positions inside the slice | `LCO_API_KEY` = 15, `Period` = 39, ping URL = 51, export anchor = 55 → 39 < 55 and 51 < 55 | ✓ PASS |
| The canonical-paragraph no-re-copy gate | `! grep -qF 'bounds the maximum allowed gap'` on the slice; `grep -qF` on the file | slice clean, file retains it | ✓ PASS |
| No credential value in any file this round touched | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` | runbook **0**, settings.py **0**, new test **0**, crontab **0** | ✓ PASS |
| Change surface is exactly the declared scope | `git diff --stat 7d61d6b..HEAD -- solsys_code/ src/ docs/ deploy/`; `git status --porcelain` on the same paths | 3 files (+137/-2); working tree clean | ✓ PASS |
| Unattended path untouched | `git diff --name-only 7d61d6b..HEAD -- solsys_code/` | only the new test module | ✓ PASS |
| Runner + preflight regression suites | `python manage.py test solsys_code.tests.test_unattended solsys_code.tests.test_check_unattended` | Ran 93 tests — **OK** | ✓ PASS |
| Lint gate | `pre-commit run ruff --all-files` (pinned ruff 0.2.1) | Passed | ✓ PASS |
| Format gate | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| Runbook RST structure after the step-2 rewrite | read-only `docutils` `publish_doctree` at `report_level=2` | 11 messages, **0 structural** (no enumerated-list / indentation / block-quote / literal-block / explicit-markup / title-underline message) | ✓ PASS |
| Debt markers in the three files this round modified | `grep -nE "TBD\|FIXME\|XXX"` | one hit — `runbook:1319 "``TBD window``"`, the campaign domain's own window vocabulary, not a debt marker; `settings.py` and the new test module clean | ✓ PASS |
| Disabled/skipped tests in the new module | `grep -nE "@unittest\.skip\|@pytest\.mark\.skip\|self\.skipTest"` | none | ✓ PASS |
| `local_settings.py` still rendered into the docs build | existence check + UUID-shaped count on the three generated pages | all three present, 1 UUID-shaped string each; both trees `.gitignore`d (:76-77) | ✗ FAIL → advisory CR-03 (outside SC 4's wording) |
| Live external heartbeat alerting from the corrected procedure | — | requires a live healthchecks-compatible account | ? SKIP → human verification item 2 |
| Sufficiency of the subsection for a naive reader | — | not observable by any token gate | ? SKIP → human verification item 1 |

**Note on the sphinx-build gate:** this verification was scoped to modify no file but `36-VERIFICATION.md`,
and `pre-commit run sphinx-build` writes into `_readthedocs/html/` — which, per CR-03, is where the
credential render lands. The docs check was therefore done read-only with `docutils` instead, the same
route plan 36-08's own prohibition P7 mandates.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN or SUMMARY declares one; this project's verification contract is the Django test runner (36-VALIDATION.md) | n/a — SKIPPED |

---

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `solsys_code/tests/test_settings_api_key_fold.py` | SCHED-08, SCHED-10 | 4 | 0 | No — the only file opened is the live settings module, read-only; no fixture is generated from system output | Value (`assertEqual` on the folded key) + exception (`assertRaises`) | **PASS** (one case near-tautological — advisory 3) |
| `solsys_code/tests/test_unattended.py` | SCHED-08/09/10 | 63 | 0 | No | Behavioral (multi-step tick workflows) + value | PASS |
| `solsys_code/tests/test_check_unattended.py` | SCHED-08/09/10 | 30 | 0 | No | Value + behavioral | PASS |

**Disabled tests on requirements:** 0. **Circular patterns detected:** 0. **Insufficient assertions:** 0 blocking; 1 near-tautological case recorded as advisory.

---

### Decision Coverage

**All 18 trackable `36-CONTEXT.md` decisions are honored by shipped artifacts** (`check.decision-coverage-verify` → `{skipped: false, blocking: false, total: 18, honored: 18, not_honored: []}`). Non-blocking gate; recorded for drift tracking.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SCHED-08 | 36-01, 36-03, 36-04, 36-05, 36-06, 36-07, 36-08 | Projector sweep, discovery backfill and reconciler on a documented cron + `flock -n` schedule with no operator action, guarded against overlapping invocations | ✓ **SATISFIED** | **Restored from PARTIALLY BLOCKED.** The mechanism was never in doubt (`STEPS` + `run_unattended` + committed crontab template + `check_unattended`'s printed line; overlap proven cross-process; **UAT Test 2 passed on the real host**). The *documented* half is now correct too: the fresh-host procedure no longer contains a step that prevents Django from starting, the flat setting name is in `docs/`, and the subsection carries no copy-pasteable nested key path. Point-of-use sufficiency is human item 1 |
| SCHED-09 | 36-01, 36-03, 36-05, 36-06, 36-07 | Failure visible through two independent layers — in-command notification and a heartbeat/dead-man's switch | ✓ SATISFIED (one human re-proof open) | `notifications.notify_staff()` (UAT Test 4 passed against real SMTP) + `ping_heartbeat()` `/start` / `/<exit-code>`; dead-man half empirically confirmed in UAT round 1. G-36-3 and G-36-1 both closed. The live re-run from the corrected procedure alone is human item 2. **Note:** REQUIREMENTS.md still marks this `Gaps Found` — a stale marker from a blanket revert, see the bookkeeping advisory |
| SCHED-10 | 36-01, 36-03, 36-04, 36-05, 36-08 | No credential value in any log line or notification the unattended path generates | ✓ SATISFIED (⚠️ WR-22 accepted; ⚠️ CR-03 advisory) | Class-name-only discipline at every `warning`/`error` site, 8 `TestCredentialHygiene` tests green, `check_unattended`'s names-only output verified live, zero UUID-shaped strings in any file this round touched, and the new runbook text keeps the credential as a placeholder. Two `DEBUG` sites format `str(exc)`, inert under the shipped `INFO` root logger — developer acceptance recorded. CR-03's docs-build render is a generated artifact, not a log line or notification the unattended path generates |
| DISCOVER-01 | 36-02, 36-03, 36-05 | Admin-editable watched-proposal list replaces per-invocation `--proposal`/name-prefix arguments | ✓ SATISFIED | `WatchedProposal` model/migration/admin; `--proposal` optional; bare sweep over `watched_rows()` through the shared `sweep_watched_rows()`; admin toggle proven to change discovery scope end-to-end. Untouched by this round. **Note:** REQUIREMENTS.md still marks this `Gaps Found` — a stale marker, see the bookkeeping advisory |

**Orphaned requirements:** none. `.planning/REQUIREMENTS.md:121-124` maps exactly SCHED-08, SCHED-09,
SCHED-10 and DISCOVER-01 to Phase 36, and all four are claimed by plan frontmatter (36-08 claims
SCHED-08 and SCHED-10).

**Bookkeeping discrepancy (not a code gap):** REQUIREMENTS.md currently reads `[x]` SCHED-08, `[ ]`
SCHED-09, `[x]` SCHED-10, `[ ]` DISCOVER-01 (`:45-48`) and `Complete / Gaps Found / Complete / Gaps
Found` (`:121-124`). `git diff a9683ca^..c04e245` shows why: commit `a9683ca` blanket-reverted all four
when G-36-4 was found, and plan 36-08's completion commit `c04e245` restored only the two it declared.
Neither SCHED-09 nor DISCOVER-01 was ever the blocked requirement, and this verification finds both
satisfied. The two markers should be restored when the phase is shipped.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| ~~`docs/runbooks/telescope_runs_calendar.rst`~~ | ~~1458-1462~~ | ~~Operator instruction naming a settings path `local_settings.py` cannot assign~~ | ✅ **RESOLVED** | Pass 3's blocker. Rewritten by `39312f4`; re-tested from scratch this pass — the slice now holds no bracketed subscript at all and the instruction is true in code |
| ~~`docs/runbooks/telescope_runs_calendar.rst`~~ | ~~1461-1462~~ | ~~Documented route with no implementation anywhere (`FACILITIES['SOAR']['api_key']`)~~ | ✅ **RESOLVED** | The fold was extended by `62d4d78`; both entries now carry the key, proven by a committed test and by a live import |
| `docs/runbooks/telescope_runs_calendar.rst` | 1448-1559 (subsection) | The slice-scoped correctness gate is task-time only — no committed test or pre-commit hook enforces it | ⚠️ Warning | A future edit could reintroduce a nested key path unblocked. Same durability shape that let G-36-4 through. Not a must-have breach (truth 68 says "fails the task"). → advisory 1 |
| `docs/conf.py` | 62-63 | `autoapi_dirs = ['../src']` with no `local_settings.py` exclusion, while the runbook mandates that file as the credential home | ⚠️ Warning | CR-03. Real, reproduced on disk again this pass, gitignored. Outside SC 4's wording and rooted in a file phase 36 never touched → advisory + human decision, not a gap |
| `docs/runbooks/telescope_runs_calendar.rst` | 1484-1494 | The one restated arithmetic line is applied to a Cron-type check that has no `Period` and alerts from the wall-clock slot | ⚠️ Warning | WR-24. Contradicts the canonical paragraph within the same file. Unchanged |
| `docs/runbooks/telescope_runs_calendar.rst` | 1471-1474 | Relative clause attaches to "one failed outright", inverting which failure class FOMO cannot report | ⚠️ Warning | WR-25. Unchanged |
| `docs/runbooks/telescope_runs_calendar.rst` | 1451-1456, 1524-1538 | `FOMO_STATE_DIR` is a third hard check (`check_unattended.py:177`, appended at `:414`) named nowhere in `docs/`; step 6's hard-set enumeration stale | ⚠️ Warning | WR-26. Benign while the default holds (`FOMO_STATE_DIR` defaults to `FOMO_LOCK_DIR`). Re-confirmed still open |
| `docs/runbooks/telescope_runs_calendar.rst`, `deploy/cron/fomo.crontab.example` | 1550-1553, 56 | "either route produces the same line" vs `cron_line()`'s five resolved values including `shutil.which('flock')` | ⚠️ Warning | WR-27. The hand-edit route can install a line that exits 127 on every tick — silent no-op. Unchanged |
| `solsys_code/management/commands/backfill_lco_observations.py`, `solsys_code/unattended.py` | 349, 191 | `logger.debug()` formatting `str(exc)` on the unattended path | ⚠️ Warning (accepted) | WR-22. Inert under the shipped `INFO` root logger; developer acceptance recorded — see ## Acknowledged Gaps. Both files byte-identical since pass 3 |
| `docs/runbooks/telescope_runs_calendar.rst` | 1458-1465 | "can only ASSIGN new settings" while the same step also instructs overriding the existing `EMAIL_BACKEND` | ℹ️ Info | Advisory 2. The operative distinction is stated correctly in the next clause; no operator action is misdirected |
| `solsys_code/tests/test_settings_api_key_fold.py` | 106-115 | Regression case pins Python's own name resolution rather than FOMO's `except ImportError` guard | ℹ️ Info | Advisory 3. Near-zero regression value; the module's other three cases are real behavioural pins |
| `docs/runbooks/telescope_runs_calendar.rst` | 1484-1485 | Authoring directive surviving into operator prose ("Name each concept first, giving healthchecks.io's spelling in parentheses") | ℹ️ Info | IN-26 |
| `deploy/cron/fomo.crontab.example` | 32, 53, 56, 64 | `[ $? -eq 99 ]` vs the line's `[ $rc -eq 99 ]`; a "(a later plan in this phase)" parenthetical for an artifact that shipped | ℹ️ Info | IN-27 |
| `docs/runbooks/telescope_runs_calendar.rst` | 1551, 1554 | Single-backtick deploy paths render as italic title references, unlike every other path in the subsection | ℹ️ Info | IN-28 |
| phase-modified files (all) | — | `TBD` / `FIXME` / `XXX` / `HACK` debt markers | — none | The only `TBD` hit is `runbook:1319`'s `` ``TBD window`` `` — the campaign domain's own window vocabulary, not a debt marker. `src/fomo/settings.py` and the new test module are clean |

**🛑 Blockers: 0.**

---

### Advisory (New Scope, Unevidenced)

**None downgraded for lack of evidence.** Every finding recorded above was either carried forward from a
prior review iteration or independently reproduced by this pass with a concrete command and its output.
The three NEW findings this pass raises (the non-committed slice gate, the "can only ASSIGN" phrasing,
the near-tautological test case) are all source-confirmed and all sub-blocker in severity, so the
convergence evidence gate (`verifier-evidence-gate.md`) never had to fire: it bounds 🛑 Blockers only,
and this pass raises none. The `advisory:` frontmatter list carries evidenced findings that fall
**outside** the phase's success criteria as written — recorded so the ship decision sees them without
their reverting a completed must-have.

---

### CLAUDE.md Paired-Docs Compliance

Plan 36-08 changed one module under `src/` (`settings.py`) and added one test module. Neither has a
notebook mapped to it in CLAUDE.md's pairing map. The paired doc CLAUDE.md names for the unattended
runner and `check_unattended` is the runbook's "How do I run everything unattended?" section — and that
section *is* what this plan corrected, in the same commit range, not as a follow-up. No
`docs/runbooks/` page other than `telescope_runs_calendar.rst` is affected, and no notebook's
documented behavior changed (`backfill_lco_observations.py`, `telescope_runs.py`,
`observation_projector.py`, `allocation_projector.py`, `campaign_*` are all byte-identical since
`7d61d6b`). **Compliant.**

---

### Human Verification Required

#### 1. SC-5 sufficiency read-through — **HOLD RELEASED, administer first**

**Test:** A reader who has not read "The two failure signals" and has not been told the
expected-interval/grace values reads only "Setting it up on a fresh host", top-down, and works the
steps.
**Expected:** The reader creates the check, sets both of its settings (`Period` 15 min, or Cron type
`*/15 * * * *`; `Grace` ~20 min), writes the flat `LCO_API_KEY` assignment into `local_settings.py`,
and fills `FOMO_HEARTBEAT_URL` — all without leaving the subsection or reading source (SC 5).
**Why human:** Sufficiency at point of use is not observable by any token-presence gate, and the
verdict is only valid from a reader not already taught the knowledge out of band — this is why
round-1 UAT Test 6 passed while G-36-1 was live.
**Hold status:** **RELEASED.** The blocking condition — gap **G-36-4**, step 2 instructing an
assignment that stopped Django from starting — is confirmed closed by this pass, in code and in prose:
the flat name is documented, the subsection contains no bracketed dict subscript at all, the fold
reaches both facilities, and the live settings module imports cleanly with both `api_key` entries
filled. Administer this item **before** item 2, which teaches the values item 1 measures.

#### 2. Live heartbeat dead-man re-run

**Test:** Create a fresh healthchecks-compatible check pointed at `FOMO_HEARTBEAT_URL`, configuring
**only** what the fresh-host setup steps name: an expected ping interval (`Period`) of 15 minutes —
or a Cron-type check with `*/15 * * * *` — and a grace time (`Grace`) of about 20 minutes. Then
disable the crontab line, simulating the scheduler never invoking the job. Run this from the
fresh-host setup steps, and AFTER item 1 above.
**Expected:** The check goes late about 15 minutes after the missed tick and alerts about 35 minutes
after the last successful ping (last ping + expected interval + grace), while FOMO itself logs
nothing and sends no email — SC 3's second, independent layer.
**Why human:** The signal comes from the external service's own timer, not from any FOMO code path.
G-36-3 was only findable this way. This is the one truth in the phase left
⚠️ PRESENT_BEHAVIOR_UNVERIFIED (truth 44).

#### 3. Decide on CR-03 before shipping

**Test:** The project's own `sphinx-build` pre-commit hook renders `src/fomo/local_settings.py` —
the credential home runbook step 2 mandates — into `_readthedocs/html/` and `docs/_build/html/`.
All three generated pages in this working tree still contain this host's ping token. Decide between
fixing it (add `'*/local_settings.py'` to `docs/conf.py`'s `autoapi_ignore`; delete both build trees;
rotate the heartbeat check if either build was ever served, copied or shared; and correct the
runbook's "must never go into a committed file" boundary claim at `:1500-1501`, which names the wrong
boundary) and recording an explicit acceptance.
**Expected:** Either the one-line `autoapi_ignore` fix plus the cleanup, or a recorded acceptance
stating that no build is ever served from a configured host.
**Why human:** A judgment call on scope and blast radius. SC 4 as written covers log lines,
notifications and error messages *the unattended path produces*, and a docs-build artifact is none
of those — so this is not a phase-goal gap — but it is a real on-disk credential exposure produced
by following this phase's own runbook, and `.gitignore` is the only thing between it and the
repository.

**Closed by UAT (no longer open):** the fresh-host preflight (Test 1, pass), the real crontab
(Test 2, pass), real SMTP delivery (Test 4, pass), logrotate under a live writer (Test 5, pass),
and the WR-22 ship decision (round 2 Test 2, pass, option (b)). See `36-UAT.md` and
## Acknowledged Gaps — **WR-22 is settled and must not be re-asked.**

The human-test order above was corrected after G-36-1 (see `36-UAT.md` and
`.planning/debug/heartbeat-setup-step-context-gap.md`): a sufficiency verdict from a reader who has
already been taught the knowledge out of band is not evidence. This rewrite preserves that order.

---

### Gaps Summary

**No gaps. G-36-4 is closed, and closed in the one place pass 3 said it had to be — in code, not only
in prose.**

Pass 3's finding was that plan 36-07 had shipped a runbook sentence telling the operator to write
`FACILITIES['LCO']['api_key']` into `local_settings.py`, an assignment that raises `NameError` inside
that module's own import namespace and takes Django down with it, and that the SOAR half of the same
sentence described a route with no implementation anywhere in the repository. Both halves are gone.

Plan 36-08 did three things, and this pass re-proved each against the files rather than the SUMMARY.
The runbook's step 2 now names `LCO_API_KEY = '<your key>'` and spends four clauses on *why* it is
flat — own namespace, `NameError`, the `except ImportError` guard that does not catch it, and Django
refusing to start — which is exactly the naming-the-failure-mode that stops the next reader from
"helpfully" nesting it back. The fold in `src/fomo/settings.py` grew the SOAR line it never had, so
the single portal credential now reaches both facility entries; that is the half pass 3 called
"unactionable in any form". And a new committed test module proves it the hard way: it locates the
live settings file, slices its real fold tail from a literal anchor, and executes that slice against
an injected `sys.modules` entry, so a future edit that drops the SOAR line breaks a test rather than
sliding past a source-token grep. The gate that failed to catch G-36-4 was a token-presence probe;
its replacement executes the thing it claims.

The corroboration that matters most is the one no plan could fake: a live import of the actual
settings module on this host, with a real `local_settings.py` present, reports `api_key` present and
non-empty for **both** `LCO` and `SOAR`. No value was read, quoted, or recorded — only the key names
and a boolean. The documented instruction and the running system now agree.

What was checked around the fix is as important as the fix. The scope fence that plan 36-07 breached —
carrying an unrelated "same-class sibling" edit into the subsection under repair, which is precisely
how G-36-4 was born — held this time: the runbook diff is a single hunk entirely inside step 2, every
deferred advisory (CR-03, WR-24…WR-27, IN-20, IN-25…IN-28) is still verbatim where it was, and
`git diff --stat` over the whole source tree since pass 3 returns three files and a clean working
tree. Plan 36-07's slice gates — presence of seven heartbeat tokens, a ninth numbered step, and the
ORDER assertion that `Period` and the ping URL both precede the export anchor — were re-run from
scratch and all still hold. 97 tests are green (63 + 30 + 4), `ruff` and `ruff-format` pass, the
runbook parses with zero structural messages, and all 18 CONTEXT decisions are honored.

**One verdict is recorded as superseded rather than verified.** Plan 36-07's truth 57 asked that step 2
"names the nested `FACILITIES['LCO']['api_key']` / `FACILITIES['SOAR']['api_key']` structure" — that
truth *was* the defect, token-true and purpose-false, and approved gap-closure plan 36-08
(`gap_closure: true`, `gap_ids: [G-36-4]`) exists to reverse it. Its sudo half still holds and is
verified; its API-key half is recorded as `PASSED (superseded)` and counted toward the score. **No
override was self-accepted by this verifier.** To put the reversal on the record formally, add to this
file's frontmatter:

```yaml
overrides:
  - must_have: "setup step 2 names the nested FACILITIES['LCO']['api_key'] / FACILITIES['SOAR']['api_key'] structure"
    reason: "Reversed on purpose by gap-closure plan 36-08: that nested path cannot be assigned from local_settings.py and stopped Django from starting (G-36-4). The flat LCO_API_KEY name, folded into both facility entries, is the corrected contract."
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

**What remains is what no automated gate can reach.** Three human items, in order: the naive-reader
sufficiency read-through for SC 5 (hold now released — this is the first pass in which it can
honestly be administered), the live external dead-man re-run that is the only end-to-end proof of
G-36-3's closure, and the CR-03 ship decision on the docs build rendering `local_settings.py` with
this host's ping token into two gitignored trees. Alongside them sit three sub-blocker observations
worth a glance before shipping: the correctness gate protecting step 2 is per-edit rather than
committed, which is the same durability shape that let G-36-4 through in the first place; the
`NameError` regression case pins Python rather than FOMO's guard; and REQUIREMENTS.md still carries
two stale `Gaps Found` markers on requirements that were never the blocked ones.

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
  re-verification did not re-open it. Both files are byte-identical since pass 3.

---

_Verified: 2026-09-18T07:30:00Z_
_Verifier: Claude (gsd-verifier)_
_Pass 4 (re-verification after gap-closure plan 36-08). Passes 1–3 are preserved in this file's git history._
