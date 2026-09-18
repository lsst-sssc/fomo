---
phase: 36-unattended-operation
verified: 2026-09-18T21:05:00Z
status: passed
score: 88/88 must-haves verified
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
  - ".planning/phases/36-unattended-operation/36-09-PLAN.md"
  - ".planning/phases/36-unattended-operation/36-09-SUMMARY.md"
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
  - "solsys_code/constants.py"
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

covered_digest: "v1:sha256:adb87d35463713c3ddece0aeef13b768452a6422a6f204efa98208355c676e4e"
behavior_unverified: 0
overrides_applied: 1
overrides:

  - must_have: "Plan 36-01 truth 10 -- no log line on the unattended path interpolates a raw exception message (WR-22)"
    reason: "Two logger.debug() sites still interpolate str(exc): backfill_lco_observations.py:349 (live authenticated LCO portal call) and unattended.py:201 (bare except around reconcile_run()). Inert under the shipped configuration -- settings.LOGGING pins the root logger to INFO, so neither line is emitted -- and does NOT falsify SC 4, which is about what appears in a log line the path actually PRODUCES. The developer recorded explicit acceptance, option (b), in UAT round 2 Test 2. step_discovery()'s IN-38 comment now names WR-22 by id as the reason not to lower the global level to DEBUG."
    accepted_by: "tlister (UAT round 2, Test 2)"
    accepted_at: "2026-09-18T09:04:00Z"
re_verification:
  previous_status: human_needed
  previous_score: 75/77
  previous_verified: 2026-09-18T07:30:00Z
  round: 4
  gaps_closed:
    - "G-36-5 (UAT round 3, Test 1) -- CLOSED by plan 36-09 (db8a1cc, b5f40a2, be955cf). Command.handle() wrote every non-ok result line to BOTH self.stdout and self.stderr, so on a terminal or under any 2>&1 the one watched_proposals warning rendered twice, the second copy red (Django sets stderr.style_func = style.ERROR). Verified at HEAD by THIS pass, not read from the SUMMARY: (a) check_unattended.py:606-626 now branches on status -- ok goes to self.stdout only, WARN/FAIL flushes self.stdout then goes to self.stderr only, never both, with a comment naming G-36-5 and explaining why the flush is load-bearing; (b) THE OPERATOR'S OWN REPRODUCTION re-run on this checkout: `python manage.py check_unattended > pf.log 2>&1` (one merged destination, exactly the condition of UAT round 3 Test 1) prints the [WARN] watched_proposals line EXACTLY ONCE, `grep '^\\[' | sort | uniq -d` returns nothing, the file contains zero ESC bytes, and the WARN lands after the nine [ok] lines in check order; (c) the test module can now SEE this defect class -- _run_merged() binds ONE io.StringIO as both stdout= and stderr=, _result_lines() extracts the bracketed-status lines, and TestResultStreamRouting carries the exactly-once case, a forced-multi-failure boundary case, a no-escape-bytes case and a routing case; (d) the three assertions that had pinned the identical non-ok line in BOTH separate captures are gone -- `grep stderr` over the module finds no remaining both-streams assertion; (e) seven presence tests now assert against the merged capture; (f) the module runs 48 tests OK (was 44); (g) the paired runbook section carries it -- step 6 gained the stream-routing passage, what a bare `>` silently drops, and the merged 2>&1 form."
    - "Prior human item 1 (SC-5 sufficiency read-through, hold RELEASED last pass) -- CLOSED by UAT round 3 Tests 2 and 3, both `pass`, administered against the post-review-fix tree. Test 2 read the heartbeat step top-down from the fresh-host steps alone; Test 3 followed step 2's API-key assignment verbatim and Django imported cleanly with check_unattended reporting the credentials present."
    - "Prior human item 2 (live heartbeat dead-man re-run) -- CLOSED by UAT round 3 Test 4, `pass`: a check configured only from the runbook's guidance went late ~15 min after the missed tick and alerted ~35 min after the last successful ping, with FOMO logging nothing and mailing nothing. This restores plan 36-06 truth 1 (prior truth 44), the only ⚠️ PRESENT_BEHAVIOR_UNVERIFIED item of the previous pass, to ✓ VERIFIED on human-administered behavioral evidence."
    - "Prior human item 3 (CR-03 credential-in-docs-build decision) -- CLOSED by fix e2ed553: docs/conf.py:70 now reads autoapi_ignore = ['*/__main__.py', '*/_version.py', '*/local_settings.py'], with a comment naming CR-03 and the three credential classes. The runbook's step 3 boundary claim was corrected in the same round to name the docs-build exposure explicitly."
  gaps_remaining: []
  regressions: []
  scope_note: "This pass covers substantially more than plan 36-09. The previous verification was written at commit 7d61d6b (file mtime 05:46 local, verified: 07:30Z); the entire 36-REVIEW-FIX round -- 35 source commits closing CR-03..CR-06, WR-16..WR-38 and IN-17..IN-40 -- landed AFTER it, as did plan 36-09. Every one of those files was re-read at HEAD (93ef89c) rather than trusted from the fix report; the four ROADMAP Success Criteria and the artifact/key-link tables below are re-derived from HEAD, not carried forward."
gaps: []
deferred: []
advisory:

  - finding: "The runbook's step-6 stream-routing passage added by plan 36-09 landed AFTER UAT round 3's SC-5 sufficiency read-through (Tests 2 and 3) was administered, so no operator has yet read the fresh-host procedure top-down with that passage in it. Plan 36-09's own coverage entry D2 declares human_judgment: true and asks for exactly this skim."
    category: other
    reason: "Not a must-have breach -- plan 36-09 truth 9 is satisfied on content (the passage names which stream carries what, that each line is written once, what a bare `>` drops, and shows the merged 2>&1 form, and every claim it makes matches the code verified above). Raised as the single human-verification item below rather than as a gap, because the risk is prose quality in an added clarification, not a missing or wrong instruction."
    evidence_status: "reproduced: 36-UAT.md round-3 Tests 2/3 timestamps (started 16:38:33Z, updated 18:07:54Z) precede be955cf (2026-09-18 13:20 -0700 = 20:20Z)"
  - finding: "UAT round 3 Test 2's `expected` text names 'the one-line alert arithmetic (~35 min)' as something step 3 must carry, and the operator recorded `pass` -- but review fix IN-25 (4ba6e3a) had already REMOVED that arithmetic from step 3, deliberately, so the alert-window reasoning has exactly one home ('The two failure signals'). Step 3 today states both values (Period 15 min / Cron */15 * * * *; Grace ~20 min) and points at that paragraph; the '35 min' token does not occur anywhere in the 209-line fresh-host slice."
    category: other
    reason: "Plan 36-07 truth 4's operative requirement -- 'states the values ... and points at that paragraph for the why, rather than re-copying it' -- is satisfied, and IN-25 was the resolution of the prior pass's own WR-24/IN-25 advisories. The human explicitly passed the read-through against this exact text, and a human sufficiency verdict outranks a token probe. Recorded so the discrepancy between the UAT expectation string and the shipped prose is not mistaken later for an undetected regression."
    evidence_status: "reproduced: `grep -c '35 min'` over the awk-sliced fresh-host subsection returns 0; step 3 read directly at runbook:1528-1563; 4ba6e3a predates the UAT round-3 window"
  - finding: "BOOKKEEPING (carried forward, still uncorrected): .planning/REQUIREMENTS.md still marks SCHED-09 and DISCOVER-01 as '[ ]' (:46, :48) and 'Gaps Found' (:122, :124). Commit a9683ca blanket-reverted all four Phase 36 requirements when G-36-4 was found; plan 36-08's completion commit c04e245 restored only SCHED-08 and SCHED-10, and plan 36-09 declared only those same two, so the other pair was never restored."
    category: other
    reason: "A planning-artifact status marker, not a codebase defect -- it cannot falsify a truth, an artifact or a link. This pass finds SCHED-09 fully satisfied (email layer + heartbeat layer both wired and behaviorally tested; the live dead-man re-run passed as UAT round 3 Test 4) and DISCOVER-01 fully satisfied (WatchedProposal + admin + bare-invocation sweep, 126 tests green). Whoever ships the phase should flip both markers rather than read them as real open gaps. Second pass in a row this has been raised."
    evidence_status: "reproduced: grep over .planning/REQUIREMENTS.md at HEAD"
  - finding: "docs/runbooks/telescope_runs_calendar.rst:1552 -- a docutils WARNING, 'Inline literal start-string without end-string', caused by ``.gitignore``d (an inline literal whose end-string is immediately followed by a letter). Sphinx renders the paragraph but not that literal as intended."
    category: other
    reason: "Pre-existing, not introduced by plan 36-09: the same warning reproduces at line 1550 of the pre-36-09 file (git show 68fff91). Cosmetic rendering only; no operator instruction is affected. Fixed by writing ``.gitignore``\\ d or 'both .gitignore-d'. Recorded because the file is in this phase's scope."
    evidence_status: "reproduced: docutils publish_doctree at report_level=2 over both HEAD and 68fff91 versions of the file -- 1 warning each, at 1552 and 1550 respectively, with no other structural message in either"
  - finding: "Carried forward from the prior pass and NOT re-tested by this one: the durability advisory on the runbook gates (plan 36-07's presence-and-order gate and plan 36-08's flat-setting gate are TASK-TIME shell gates, not committed regression tests -- no file under solsys_code/ or src/ references the fresh-host subsection); the step-2 'can only ASSIGN new settings' copy-edit; TestBracketedDictSubscriptRaisesNameError pinning Python's own name resolution; WR-26 (FOMO_STATE_DIR now documented by c8b81e2, so this one IS resolved); and 36-REVIEW.md's remaining low-severity items. Plan 36-09 added a third task-time gate of the same shape."
    category: other
    reason: "None falsifies a success criterion or a plan must-have truth. The durability point stands: three runbook-content gates now exist only as task-time shell commands, so a future edit reintroducing a copy-pasteable nested key path, or breaking the stream-routing sentence, would ship unblocked. Cheapest durable fix remains a ~15-line SimpleTestCase that awk-slices the subsection and asserts the same clauses."
    evidence_status: "re-confirmed this pass: `grep -rln 'Setting it up on a fresh host' solsys_code/ src/ --include=*.py` returns nothing"
prohibitions_flagged:

  - plan: "36-09"
    count: 6
    tier: judgment
    note: "All six are verification: flagged-unverified (judgment tier). Per the fail-closed rule each carries a NON-AUTHORITATIVE LLM-judge verdict below and an `unverified-prohibition — human review recommended` flag; none is silently passed."
human_verification:

  - test: "Skim the rendered 'Setting it up on a fresh host' procedure, step 6, in the built docs (or read docs/runbooks/telescope_runs_calendar.rst:1620-1638) with fresh eyes -- ideally the same reader who ran UAT round 3 Tests 2 and 3, since that read-through predates this passage."
    expected: "The added stream-routing paragraph reads in the surrounding operator voice, and a fresh-host operator finishes step 6 knowing that passing lines go to standard output, that warnings and failures go to standard error instead, that each line is written once, that a bare `>` silently drops every warning and failure, and that `2>&1` puts the whole report in one file in check order. Every factual claim in it is already verified against the code -- this is a prose-quality and point-of-use-sufficiency judgment only."
    why_human: "Whether an added operator-facing passage 'reads well' and genuinely helps at point of use is not observable by any token-presence or parse-cleanliness gate -- the same class of judgment that let G-36-1 pass a round-1 read-through. Plan 36-09's own coverage entry D2 declares human_judgment: true and asks for this skim by name."
  - test: "Review the six judgment-tier prohibitions from plan 36-09 recorded in the 'Plan 36-09 Prohibitions' table below and confirm the LLM-judge verdicts."
    expected: "All six judged Satisfied on the evidence shown (only three files changed; the operator's UAT wording survived into db8a1cc with its provenance intact; no explicit style argument reaches the standard-error write; no credential value in any committed file; G-36-1/G-36-3/G-36-4 gates all re-run green; 36-UAT.md and 36-VERIFICATION.md untouched by the plan)."
    why_human: "Judgment-tier prohibitions are must-NOT checks with no wired enforcement in this Python/RST repo. Autonomous verification records a non-authoritative judge verdict and flags them; it never silently passes them."
---

# Phase 36: Unattended Operation — Verification Report (Round 4)

**Phase Goal:** The projector sweep, the LCO/SOAR discovery backfill and the reconciler run on the real host on a documented schedule with nobody typing anything, against a watched-proposal list an operator edits in the admin — and when it breaks, an operator finds out.

**Verified:** 2026-09-18T21:05:00Z at HEAD `93ef89c`
**Status:** passed — UAT round 4 (2026-09-18, `7d1d1ea`) confirmed the two human items below; frontmatter canonicalized by `/gsd-verify-work 36`
**Re-verification:** Yes — round 4, after gap-closure plan 36-09 (`gap_ids: [G-36-5]`) and the full 36-REVIEW-FIX round.

> **Scope note.** The previous report was written before the 36-REVIEW-FIX round landed (35 source commits, `76ed56e`…`27c722a`). This pass re-derives the four ROADMAP Success Criteria, the artifact table and the key-link table from HEAD rather than carrying them forward, and re-runs every gate it cites. Rounds 1–3 are preserved in this file's git history.

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Projector sweep, LCO/SOAR discovery backfill and reconciler all run on their documented recurring schedule with no operator action, and two invocations never overlap | ✓ VERIFIED | `deploy/cron/fomo.crontab.example:68` — the single `*/15 * * * *` line wraps `run_unattended` in `/usr/bin/flock -n -E 99 …/run_unattended.cron.lock`, captures `$rc`, normalizes a 99 (lock held) back to 0 after logging a visible skip line, and `exit $rc`. `unattended.py:115 command_lock()` holds a SECOND, distinct in-process lock (`run_unattended.lock`) plus a per-step lock, deliberately a different path from the cron lock (CR-01). `STEPS` (`:421`) registers all four steps in fixed D-01 order; `run_tick()` (`:781`) catches a raising step so one failure never stops the rest. Behaviorally pinned by `TestLocking`, `TestRunUnattended` — 126 tests green in this pass's run. The live 15-minute schedule on the real host passed UAT round 1 Test 2 (three START/END banners ~15 min apart) |
| SC2 | Adding a proposal in the admin is enough for its robotically scheduled observations to start appearing — discovery takes no per-invocation arguments and needs no redeploy | ✓ VERIFIED | `WatchedProposal` model (`models.py:740`) + migration `0022_watchedproposal.py` + `WatchedProposalAdmin` registered with `list_editable` (`admin.py:463,486`). `backfill_lco_observations.py:807` — `--proposal` is `required=False, default=None`; omitted, `handle()` routes to `sweep_watched_rows()` (`:697`) which sweeps every active row in `proposal_code` order, isolates a per-row failure to that row (class name only, `:740-749`) and writes `last_run_at`/`last_run_summary` either way. `unattended.py:356 step_discovery()` calls the same function with captured sinks, so the unattended path takes no arguments at all. `TestDiscoveryStep` + `test_backfill_lco_observations` green |
| SC3 | A failed unattended run reaches an operator two independent ways: a notification from the command itself, and a heartbeat that also fires when the scheduler never invoked the job at all | ✓ VERIFIED | Layer 1: `run_tick()` `:840-861` → `load_state()` / `decide_notification()` / `_send_notification()`, with D-11 mail-once-per-newly-failing-set, 24 h reminder and one-shot recovery; `notifications.notify_staff()` is the single sender. Layer 2: `ping_heartbeat('start')` before the steps and `ping_heartbeat(str(exit_code))` after (`:818`, `:861`), both skipped for `--dry-run`/`--step`. The layers are independent by construction — the heartbeat fires from the external service's own timer when the tick never starts, which is exactly the class the email cannot report. **Live dead-man re-proof: UAT round 3 Test 4 `pass`** (late ~15 min, alert ~35 min, FOMO silent). `TestHeartbeat`, `TestNotification`, `TestStateFileRobustness`, `TestStateFileAtomicWrite` all green |
| SC4 | No API key or password appears in any log line, notification or error message the unattended path produces | ✓ VERIFIED (1 accepted deviation — WR-22, see overrides) | `TestCredentialHygiene` (`test_unattended.py:976`) exercises every forced failure path plus the failure email's own body. `sweep_watched_rows()` `:740-747` converts a caught portal exception to `f'failed: {type(exc).__name__}'` before it reaches the row, stderr or the log, with a D-17 comment. `step_discovery()`'s IN-38 loop logs the captured sinks one line per record at INFO and names WR-22 by id as the reason not to lower the global level to DEBUG. **Live merged run this pass:** the real `check_unattended` output reports `FOMO_HEARTBEAT_URL: set`, `LCO and SOAR api_key both set` — names and set/unset status only, zero values, zero UUID-shaped strings in any committed phase file. CR-03 (the docs-build render of `local_settings.py`) is now fixed at `docs/conf.py:70` |
| SC5 | An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source | ✓ VERIFIED (one prose-quality skim open — human item 1) | Nine numbered steps in the `Setting it up on a fresh host` slice (209 lines, `runbook:1448-1656`), with the create-and-configure-the-heartbeat step (3) standing before the `FOMO_HEARTBEAT_URL` export (4) — order gate re-run green this pass. Step 2 names the flat `LCO_API_KEY = '<your key>'` and the slice contains ZERO bracketed dict subscripts of any kind. `FOMO_STATE_DIR` and the `flock -E` requirement are now documented (`c8b81e2`), closing the prior WR-26. **Both halves of the sufficiency question passed in UAT round 3** (Test 2 heartbeat step, Test 3 API-key step). The only thing not yet read by an operator is plan 36-09's newly added step-6 paragraph |

### Plan must-have truths

Plans 36-01 … 36-08 contribute 72 truths, re-checked this pass as a regression sweep at HEAD after the 36-REVIEW-FIX round; plan 36-09 contributes 11, verified in full. Truth-by-truth tables for 36-01…36-08 are preserved in this file's git history at `21dc3f8`/round 3; the round-4 disposition of every one of them is:

| Source | Truths | Round-4 status |
|--------|--------|----------------|
| 36-01 (runner, lock, heartbeat, notification, crontab template, shared mailer) | 10 | 9 ✓ VERIFIED + 1 PASSED (override, WR-22 — truth 10) |
| 36-02 (WatchedProposal, admin, `sweep_proposal()` extraction, bare-invocation sweep) | 9 | 9 ✓ VERIFIED |
| 36-03 (the four steps, `STEPS` order, failure isolation, credential hygiene) | 8 | 8 ✓ VERIFIED |
| 36-04 (`check_unattended`, `cron_line()`, `--send-test-email`, no-value output, logrotate) | 7 | 7 ✓ VERIFIED — truth on the result-line report now stronger, see G-36-5 |
| 36-05 (runbook section, backfill contract, demo notebook, CLAUDE.md map) | 9 | 9 ✓ VERIFIED |
| 36-06 (heartbeat two-knob guidance) | 8 | 8 ✓ VERIFIED — truth 1 (prior truth 44) was the only ⚠️ PRESENT_BEHAVIOR_UNVERIFIED item last pass; **upgraded to ✓ VERIFIED on UAT round 3 Test 4's live pass** |
| 36-07 (fresh-host heartbeat step, ordering gate) | 9 | 9 ✓ VERIFIED — truth 6 (nested `FACILITIES[...]` wording) remains superseded-and-counted, reversed by approved gap-closure plan 36-08; truth 4's arithmetic clause was intentionally relocated by review fix IN-25, see advisory 2 |
| 36-08 (flat `LCO_API_KEY`, both-facility fold, committed fold test) | 12 | 12 ✓ VERIFIED — `src/fomo/settings.py` fold re-read; `test_settings_api_key_fold` green |
| **36-09 (G-36-5 stream routing) — the new work** | **11** | **11 ✓ VERIFIED — detailed below** |

#### Plan 36-09 must-have truths (G-36-5)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 78 | `check_unattended` prints each result exactly once; the `[WARN] watched_proposals` line appears a single time, in the operator's real condition (terminal / `2>&1`) | ✓ VERIFIED | **The operator's own reproduction, re-run on this checkout:** `python manage.py check_unattended > pf.log 2>&1` → the `[WARN] watched_proposals` line occurs ONCE; `grep -E '^\[(ok\|WARN\|FAIL)\] ' \| sort \| uniq -d` prints nothing; exit 0, 9/10 checks passed. Behavior-dependent truth, behaviorally proven — not inferred from presence |
| 79 | Each line goes to ONE stream chosen by status: `ok` → stdout; `[WARN]`/`[FAIL]` → stderr INSTEAD of stdout | ✓ VERIFIED | `check_unattended.py:606-626` — `if status == 'ok': self.stdout.write(line)` / `else: self.stdout.flush(); self.stderr.write(line)`. Pinned by `test_warning_and_passing_lines_route_to_separate_streams` (warning in stderr and absent from stdout; `[ok] flock` in stdout and absent from stderr; cron block stays on stdout) — the named test passes |
| 80 | Merged-destination read order is check order — stdout flushed immediately before each stderr write | ✓ VERIFIED | The `self.stdout.flush()` is present in the `else` branch with a comment naming the block-buffered/line-buffered hazard. Observed in the live merged run: the single WARN lands after all nine `[ok]` lines, then the blank line, the cron block and the summary follow |
| 81 | No escape sequence reaches a non-terminal sink | ✓ VERIFIED | `test_merged_capture_has_no_escape_bytes` asserts no `\x1b` and passes; live merged run `grep -c $'\x1b'` → 0. No `style_func=` argument anywhere in the emission loop |
| 82 | Regression pinned by a test using the SAME `io.StringIO` for `stdout=` and `stderr=`, asserting exactly once — plus a harder forced-multi-failure case asserting no duplicate result line | ✓ VERIFIED | `_run_merged()` (`test_check_unattended.py:91`) binds one sink to both. `test_watched_proposals_warning_appears_exactly_once_in_merged_capture` and `test_multiple_non_ok_results_produce_no_duplicate_lines` (deletes the staff user to force a hard `[FAIL]` alongside two `[WARN]`s, asserts ≥3 non-ok lines and `len(lines) == len(set(lines))`) both present and green |
| 83 | A dedicated routing test states the contract the other tests no longer restate | ✓ VERIFIED | `TestResultStreamRouting` (`:519`) with a class docstring naming G-36-5 and the UAT round; the routing case covers all three directions |
| 84 | The three assertions that encoded the defect as the expectation no longer pin the dual-write | ✓ VERIFIED | `grep stderr` over the whole module returns no remaining "same line in both captures" assertion. `test_flock_outside_system_directories_gets_a_sanity_note`, `test_flock_symlink_resolving_outside_system_directories_gets_a_sanity_note` and `test_unset_heartbeat_is_a_warning` all read from `_run_merged()` now |
| 85 | Every test asserting an operator sees a warning asserts against the merged capture | ✓ VERIFIED | Seven presence tests converted (2 flock, 4 warning, 1 facility-credentials). `test_warnings_do_not_mask_a_hard_failure` builds its own `combined = stdout+stderr` — also routing-agnostic. Module grew 44 → **48 tests, all OK** |
| 86 | The paired runbook section carries the change (CLAUDE.md paired-docs rule — `check_unattended.py` → the runbook section, NOT a notebook) | ✓ VERIFIED | `runbook:1623-1638`, inside step 6: names which stream carries what, that each line is written once, that a bare `>` "silently drops every warning and failure, leaving a preflight log that looks entirely clean", and shows the `2>&1` form in a `code-block:: console` using the page's own `>>` prompt convention. Prose quality is human item 1 |
| 87 | The operator's own uncommitted step-2 wording survives into git, typos fixed, whitespace stripped, as its own commit | ✓ VERIFIED | `db8a1cc` is a single hunk inside step 2 adding "(from the LCO Observation Portal and the 'Profile' link under your username in the top right corner)" and "This should never be in the crontab line" — doubled article and missing verb both corrected, no trailing whitespace on any added line. It is the FIRST of the plan's three commits, before either file-touching task |
| 88 | Nothing else moves: 36-07's and 36-08's gates still pass, `unattended.py`/`notifications.py`/`run_unattended.py` untouched, ruff + ruff-format + suite green | ✓ VERIFIED | `git diff --name-only 68fff91..HEAD` → only the three declared files (+ planning artifacts). 36-07 gate re-run over the 209-line slice: `Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`, `healthchecks.io`, `The two failure signals` all present, 9 numbered steps, ORDER holds (Period@96 < ping-URL@107 < export anchor@117). 36-08 gate: `LCO_API_KEY` present, `grep -E "[A-Za-z_]+\[['\"]"` over the slice → **no match**. `pre-commit run ruff` / `ruff-format` → Passed. 48 + 126 tests green in this pass |

**Score:** 88/88 truths verified — 0 failed, 0 present-but-behavior-unverified, 1 PASSED (override, WR-22).

---

### Plan 36-09 Prohibitions (must-NOT checks)

All six are judgment-tier (`verification: flagged-unverified`). Per the fail-closed rule these carry a **non-authoritative LLM-judge verdict** and are flagged for human review; none is silently passed.

| # | Prohibition | Judge verdict | Evidence | Flag |
|---|-------------|---------------|----------|------|
| P1 | The executor must NOT discard/stash/revert the uncommitted operator edit to the runbook | Satisfied | `db8a1cc` is the plan's first commit and contains the operator's phrase verbatim (typos corrected only). No `git checkout --`/`restore`/`stash`/`clean` appears in the SUMMARY's command record, and the wording is present in the file at HEAD | unverified-prohibition — human review recommended |
| P2 | Only three files change; `unattended.py`, `notifications.py`, `run_unattended.py`, every other command, `test_unattended.py` and the crontab template untouched | Satisfied | `git diff --name-only 68fff91..HEAD` lists exactly `check_unattended.py`, `test_check_unattended.py`, `telescope_runs_calendar.rst` plus four `.planning/` files. No other source file moved | unverified-prohibition — human review recommended |
| P3 | No explicit style argument on the standard-error write | Satisfied | `self.stderr.write(line)` — plain string, no `style_func=`. **Behaviorally pinned** by `test_merged_capture_has_no_escape_bytes` and by the live merged run's zero ESC bytes | unverified-prohibition — human review recommended |
| P4 | No credential VALUE enters any committed file | Satisfied | `grep -ciE '[0-9a-f]{8}-[0-9a-f]{4}-…'` → **0** in the runbook, the crontab template, `check_unattended.py`, `test_check_unattended.py` and `src/fomo/settings.py`. Placeholders (`'<your key>'`, `hc-ping.com/<uuid>`) intact | unverified-prohibition — human review recommended |
| P5 | G-36-1/G-36-3/G-36-4 are not re-opened or re-edited; their gates are re-run only to prove nothing broke | Satisfied | Both slice gates re-run green (see truth 88). The 36-09 runbook diff is two hunks: one inside step 2 (the operator's own wording), one appended to step 6. Step 3's heartbeat prose and the canonical "Heartbeat." paragraph are byte-identical | unverified-prohibition — human review recommended |
| P6 | `36-UAT.md` and `36-VERIFICATION.md` are not edited by this plan | Satisfied | Neither appears in `git diff --name-only 68fff91..HEAD`. Both are reconciled here, by re-verification, as the plan intended | unverified-prohibition — human review recommended |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/check_unattended.py` | Preflight + cron line, each result line emitted once to one stream | ✓ VERIFIED | Status-branched emission loop at `:606-626` with the load-bearing `self.stdout.flush()`. Ten checks including `check_state_dir()` and `check_facility_credentials()`. 48 tests green |
| `solsys_code/tests/test_check_unattended.py` | Merged-sink helper + routing regression class | ✓ VERIFIED | `_run_merged()`, `_result_lines()`, `TestResultStreamRouting` (4 cases), 7 converted presence tests. 44 → 48 tests |
| `docs/runbooks/telescope_runs_calendar.rst` | Fresh-host procedure an operator can follow verbatim, now including stream routing | ✓ VERIFIED | 9 numbered steps, 209-line slice; both content gates green; step 6 carries the routing passage; docutils parse shows 0 structural messages and one pre-existing cosmetic warning (advisory 4) |
| `solsys_code/unattended.py` | Runner, four steps, locks, heartbeat, notification, state file | ✓ VERIFIED | Re-read at HEAD after the review-fix round: `_fallback_state_path()`/`_fallback_is_trustworthy()` (CR-05), WR-03's isolated notification block, WR-04's fresh END timestamp, IN-38's per-line discovery logging. Untouched by 36-09. 126 tests green |
| `solsys_code/constants.py` | Single owner of the 15-minute cron interval and default paths | ✓ VERIFIED | New leaf module from IN-34/WR-36; imported by `unattended.py:44` as `_CRON_INTERVAL_MINUTES` |
| `solsys_code/notifications.py` | Shared request-free mailer | ✓ VERIFIED | Single sender; called from `unattended.py`, `check_unattended.py`, `campaign_views.py` |
| `solsys_code/management/commands/run_unattended.py` | Cron entry point | ✓ VERIFIED | Unchanged since 36-01; `--dry-run`/`--step` both quiet the ping and the mail |
| `solsys_code/models.py` (`WatchedProposal`) + `migrations/0022_watchedproposal.py` + `admin.py` | Admin-editable watch list | ✓ VERIFIED | Model `:740`, migration applied on this host, `WatchedProposalAdmin` registered `:486` |
| `solsys_code/management/commands/backfill_lco_observations.py` | `sweep_proposal()`, `watched_rows()`, `sweep_watched_rows()`, optional `--proposal` | ✓ VERIFIED | `--proposal required=False, default=None`; window/name-prefix flags guarded to `--proposal` only (CR-02); per-row isolation returns class name only |
| `deploy/cron/fomo.crontab.example` | `*/15` flock-guarded line, log redirect, visible skip, no secret or host path | ✓ VERIFIED | Line `:68` read directly; `rc=$?` / `rc=0` normalization / `exit $rc`; both "full setup" pointers name the fresh-host subsection; zero UUIDs, placeholders intact |
| `deploy/logrotate/fomo.example` | Daily, rotate 14, copytruncate documented | ✓ VERIFIED | Unchanged; UAT round 1 Test 5 passed under a live writer |
| `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` | Executed watched-proposal cells, in the toctree | ✓ VERIFIED | Last regenerated at `f6be59b`; **no command change landed after it** — `git log f6be59b..HEAD -- backfill_lco_observations.py` is empty, so the paired notebook is not stale |
| `docs/conf.py` | autoapi exclusion for `local_settings.py` | ✓ VERIFIED (was ⚠️ ADVISORY CR-03) | `:70` — `autoapi_ignore = ['*/__main__.py', '*/_version.py', '*/local_settings.py']`, fixed by `e2ed553` |
| `src/fomo/settings.py` | `LCO_API_KEY` fold reaching BOTH LCO and SOAR | ✓ VERIFIED | Live merged preflight reports `LCO and SOAR api_key both set`; `test_settings_api_key_fold` (4 cases) green |
| Tests (`test_unattended`, `test_check_unattended`, `test_backfill_lco_observations`, `test_settings_api_key_fold`) | Behavioral coverage | ✓ VERIFIED | **174 tests re-run green by this verification** (48 + 126) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `check_unattended.py`'s emission loop | a single destination (terminal / `2>&1`) | status-branched write + flush | ✓ **WIRED** | **This is the link G-36-5 lived in.** Proven by running the command into one merged file and counting: no duplicate result line, correct order, no ESC bytes |
| the emission loop | the merged-sink test helper | `_run_merged()` binding one `io.StringIO` to both `stdout=` and `stderr=` | ✓ **WIRED** | Was structurally absent — this is why the defect shipped in the command's first commit and survived six review iterations |
| the stdout flush | the crontab template's `>> …/unattended.log 2>&1` | one file, two streams | ✓ WIRED | Template line `:68` confirmed to carry `2>&1`; step 6 now tells the operator to use the same for the preflight |
| the routing the command implements | the sentence the runbook's step 6 makes about it | operator reading step 6 | ✓ WIRED | Runbook's claim matches the code branch-for-branch (ok→stdout, WARN/FAIL→stderr, once each) |
| cron line ↔ `run_unattended` ↔ `flock` | non-overlap guarantee | two distinct lock paths | ✓ WIRED | `run_unattended.cron.lock` (cron) vs `run_unattended.lock` (`command_lock`) — deliberately different files (CR-01) |
| `WatchedProposal` admin rows | `step_discovery()` | `watched_rows()` → `sweep_watched_rows()` → `sweep_proposal()` | ✓ WIRED | No argument crosses the boundary; adding a row in the admin is the whole configuration surface |
| failing step | operator's inbox | `decide_notification()` → `_send_notification()` → `notifications.notify_staff()` | ✓ WIRED | Mail-once-per-newly-failing-set, 24 h reminder, one-shot recovery; state only recorded when delivery succeeded (WR-02) |
| missed/hung tick | operator's inbox | external heartbeat service timer | ✓ WIRED | `ping_heartbeat('start')` / `ping_heartbeat(str(exit_code))`; live dead-man re-proof passed (UAT round 3 Test 4) |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| G-36-5 reproduction: no duplicate result line in a merged destination | `python manage.py check_unattended > pf.log 2>&1` then `grep -E '^\[(ok\|WARN\|FAIL)\] ' \| sort \| uniq -d` | exit 0, `9/10 checks passed`, **no duplicate lines**, `[WARN] watched_proposals` exactly once | ✓ PASS |
| No ANSI escape bytes in a redirected preflight log | `grep -c $'\x1b' pf.log` | 0 | ✓ PASS |
| Stream-routing + duplicate regression suite | `python manage.py test solsys_code.tests.test_check_unattended` | Ran 48 tests, **OK** | ✓ PASS |
| Runner, discovery and settings-fold suites | `python manage.py test solsys_code.tests.test_unattended solsys_code.tests.test_backfill_lco_observations solsys_code.tests.test_settings_api_key_fold` | Ran 126 tests, **OK** | ✓ PASS |
| Lint / format gate (D-07) | `pre-commit run ruff --files …` / `ruff-format --files …` | Passed / Passed | ✓ PASS |
| Runbook structural cleanliness | `docutils.publish_doctree(report_level=2)` on the full page | 0 structural messages; 1 pre-existing cosmetic warning at `:1552`, identical at `:1550` in the pre-36-09 file; the remaining messages are Sphinx-only `:ref:`/`:doc:` roles | ✓ PASS |
| 36-07 presence-and-order gate over the fresh-host slice | token + line-position probe on the awk slice | all tokens present, 9 numbered steps, Period(96) < ping-URL(107) < export anchor(117) | ✓ PASS |
| 36-08 flat-setting gate over the same slice | `grep -E "[A-Za-z_]+\[['\"]"` | **no match**; `LCO_API_KEY` present | ✓ PASS |
| Full configured suite on HEAD | (post-merge gate, reported by the orchestrator) | 1495 OK (1 skipped) + 40 OK | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` exists in this repository and no plan in phase 36 declares a probe. Step 7c: **SKIPPED (no probes declared or discoverable)** — the phase's runnable checks are the Django test modules and the live command runs recorded above.

---

### Requirements Coverage

| Requirement | Source plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| SCHED-08 | 36-01, 36-03, 36-04, 36-05, 36-09 | Projector sweep, LCO/SOAR discovery backfill and reconciler run on a documented recurring cron + `flock -n` schedule with no operator action, guarded against overlap | ✓ SATISFIED | SC1 evidence; two distinct lock paths; `TestLocking` green; live 15-min schedule passed UAT round 1 Test 2. `[x]` in REQUIREMENTS.md |
| SCHED-09 | 36-01, 36-03, 36-04, 36-06, 36-07 | Failure visible through two independent layers — in-command notification and a heartbeat/dead-man's switch that also catches the scheduler failing to invoke | ✓ SATISFIED | SC3 evidence; live dead-man re-proof passed (UAT round 3 Test 4); `TestNotification` + `TestHeartbeat` green. ⚠️ **REQUIREMENTS.md still marks this `[ ]` / "Gaps Found" — stale bookkeeping, see advisory 3** |
| SCHED-10 | 36-01, 36-03, 36-04, 36-08, 36-09 | No credential value appears in any log line or notification the unattended path generates | ✓ SATISFIED | SC4 evidence; `TestCredentialHygiene`; zero UUIDs in committed phase files; CR-03 docs-build leak fixed. One accepted deviation (WR-22, override). `[x]` in REQUIREMENTS.md |
| DISCOVER-01 | 36-02, 36-03, 36-05 | Admin-editable watched-proposal list replaces per-invocation `--proposal`/name-prefix arguments so discovery runs unattended against every watched proposal | ✓ SATISFIED | SC2 evidence; `WatchedProposal` + admin + bare-invocation sweep + per-row isolation + bookkeeping; UAT Tests 8/9/10 pass. ⚠️ **REQUIREMENTS.md still marks this `[ ]` / "Gaps Found" — stale bookkeeping, see advisory 3** |

No orphaned requirements: `grep -E "Phase 36" .planning/REQUIREMENTS.md` maps exactly these four IDs, and every one is claimed by at least one plan's `requirements` field.

---

### CLAUDE.md Paired-Docs Compliance

| Module changed this round | Paired doc per CLAUDE.md | Updated? | Evidence |
|---------------------------|--------------------------|----------|----------|
| `check_unattended.py` | the runbook's "How do I run everything unattended?" section — explicitly NOT a notebook | ✓ Yes | Step 6 gained the stream-routing passage in the same plan (`be955cf`) |
| `unattended.py`, `notifications.py`, `run_unattended.py` | same runbook section | n/a — untouched by 36-09; the review-fix round updated the runbook for CR-05, WR-16 and WR-17 in `14ae0bf`, `74b53b4`, `6ded6b4` | ✓ compliant |
| `backfill_lco_observations.py` | `backfill_lco_observations_demo.ipynb` | ✓ Not stale | `git log f6be59b..HEAD -- solsys_code/management/commands/backfill_lco_observations.py` is **empty** — the notebook was regenerated after the last command change |
| `observation_projector.py`, `telescope_runs.py`, `campaign_reconciler.py`, `allocation_projector.py` | their own notebooks | n/a | No commit after the notebook baseline touches any of them |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/models.py` | 357, 453, 466, 509 | literal `TBD` | ℹ️ Info — **not a debt marker** | Domain vocabulary: a `CampaignRun` whose observing window is To Be Determined. Pre-existing from an earlier phase, part of the model's public verbose names and constraint comments |
| `docs/runbooks/telescope_runs_calendar.rst` | 1319 | `TBD window` | ℹ️ Info — **not a debt marker** | Same domain term, documented for operators |
| `check_unattended.py`, `test_check_unattended.py`, `unattended.py` | — | `TODO`/`FIXME`/`XXX`/`HACK`/placeholder | — | **None found** |
| `backfill_lco_observations.py` | 349 | `logger.debug(f'… {exc}')` | ⚠️ Warning — **accepted** | WR-22. Inert at the shipped INFO root level; explicit developer acceptance recorded in UAT round 2 Test 2 and carried as an override here. Must not be re-asked |
| `unattended.py` | 201 | `logger.debug('… %s', exc)` | ⚠️ Warning — **accepted** | Same finding, same acceptance |

No blocker anti-pattern found. No unreferenced `TBD`/`FIXME`/`XXX` debt marker exists in any file this phase modified.

---

### Human Verification Required

#### 1. Skim plan 36-09's added step-6 passage in the fresh-host procedure

**Test:** Read `docs/runbooks/telescope_runs_calendar.rst:1620-1638` (step 6 of "Setting it up on a fresh host"), or the rendered page, with fresh eyes — ideally the same reader who ran UAT round 3 Tests 2 and 3, since that read-through was administered before this passage existed.
**Expected:** The paragraph reads in the surrounding operator voice, and a fresh-host operator finishes step 6 knowing: passing lines go to standard output; warnings and failures go to standard error *instead*; each line is written once; a bare `>` silently drops every warning and failure, leaving a preflight log that looks clean; `2>&1` puts the whole report in one file in check order.
**Why human:** Every factual claim in the passage is already verified against the code above — what is left is prose quality and point-of-use sufficiency, which no token-presence or parse gate can measure. Plan 36-09's own coverage entry D2 declares `human_judgment: true` and asks for this skim by name. This is the same class of judgment that let G-36-1 pass a round-1 read-through.

#### 2. Confirm the six judgment-tier prohibition verdicts

**Test:** Review the "Plan 36-09 Prohibitions" table above.
**Expected:** All six judged Satisfied on the evidence shown.
**Why human:** Judgment-tier prohibitions have no wired enforcement in this Python/RST repo. Autonomous verification records a non-authoritative judge verdict and flags it; it never silently passes it.

---

## Acknowledged Gaps — **WR-22 is settled and must not be re-asked.**

The developer accepted option (b) for WR-22 in UAT round 2 Test 2: the two `logger.debug()` sites that interpolate a raw exception message stay as they are, because `settings.LOGGING` pins the root logger to INFO so neither line is ever emitted under the shipped configuration, and SC 4 is about what appears in a log line the path actually *produces*. `step_discovery()` now carries an inline comment naming WR-22 by id as the reason not to lower the global level to DEBUG — the acceptance is documented at the site that would activate it. Recorded here as an `overrides` entry so it counts toward the score rather than reappearing as a finding.

---

### Gaps Summary

**None.** G-36-5 — the last open gap, and the only one raised by UAT round 3 — is closed, and closure was proven by re-running the operator's own reproduction on this checkout rather than by reading plan 36-09's SUMMARY: `python manage.py check_unattended` into a single merged destination now prints every result line exactly once, in check order, with no escape bytes. The underlying reason the defect survived six review iterations is also fixed: the test module now has a helper that binds one sink to both streams, so this entire defect class is reachable by a test at all.

All three previously open human items are closed — the SC-5 sufficiency read-through and the API-key step by UAT round 3 Tests 2 and 3, the live heartbeat dead-man re-run by UAT round 3 Test 4 (which upgrades the one ⚠️ PRESENT_BEHAVIOR_UNVERIFIED truth of the previous pass to ✓ VERIFIED on human-administered behavioral evidence), and the CR-03 docs-build credential exposure by fix `e2ed553`.

The phase goal is achieved: the four steps run on a `*/15` flock-guarded schedule with two distinct lock paths and no operator input; a `WatchedProposal` row added in the admin is the entire configuration surface for discovery; a failure reaches an operator both by email and by an external dead-man's switch that has now been proven live; and no credential value appears in any output the path produces — the one remaining deviation (WR-22) is inert under the shipped configuration and explicitly accepted.

Status is `human_needed`, not `passed`, for two reasons only, neither of which is a defect: plan 36-09's six judgment-tier prohibitions are flagged rather than silently passed, and the step-6 paragraph it added to the runbook has not yet been read by an operator in sufficiency mode. Before shipping, also flip the two stale `SCHED-09` / `DISCOVER-01` markers in `.planning/REQUIREMENTS.md` (advisory 3) — both requirements are satisfied in the codebase; only the checkbox is wrong.

---

_Verified: 2026-09-18T21:05:00Z at `93ef89c`_
_Verifier: Claude (gsd-verifier), round 4_
