---
status: diagnosed
trigger: "UAT Phase 36 Test 3 'The heartbeat's dead-man half' — operator set healthchecks.io Grace to 20 min per the runbook; check still green 28 min after last ping. Only after the operator found the separate 'Period' setting (default 1 day) and set it to 15 min did the check go Late then Down with an alert email. FOMO itself logged nothing and sent nothing (mechanism correct)."
created: 2026-09-17T00:00:00Z
updated: 2026-09-17T00:00:00Z
gap_id: G-36-3
phase: 36-unattended-operation
mode: diagnose-only
---

## Current Focus

hypothesis: CONFIRMED — see Resolution.root_cause
test: (complete)
expecting: (complete)
next_action: "Return diagnosis to caller. Diagnose-only mode: no fix applied."

reasoning_checkpoint:
  hypothesis: "The runbook's heartbeat guidance names only healthchecks.io's Grace knob and never its Period/schedule knob, so an operator following it literally leaves Period at the 1-day default; healthchecks.io alerts at last_ping + Period + Grace, making time-to-alert ~24h20m instead of the intended ~20-35m — the dead-man layer is silently disabled."
  confirming_evidence:
    - "Direct read: runbook:1543-1547 says only 'configure one check per schedule, with a grace period a little above one 15-minute interval -- about 20 minutes is recommended'. No Period/schedule/interval knob named."
    - "Repo-wide grep: the string 'period' appears in a heartbeat context in exactly 4 places (runbook:1545, :1564, :2035, crontab.example:31) and every one is 'grace period'. The healthchecks.io Period knob is named nowhere in docs, deploy, source, or settings."
    - "Vendor docs confirm the mechanism: Period = expected time between pings, Grace = extra wait before alerting; Late at last_ping+Period, Down at last_ping+Period+Grace; auto-provisioned default is Period 1 day / Grace 1 hour."
    - "Operator's own A/B in the trigger is a clean differential: Grace 20 alone = green at 28 min; adding Period 15 = Late then Down + alert email. Only the Period variable changed."
  falsification_test: "Find any operator-facing artifact naming the Period/schedule knob, OR show healthchecks.io alerts on Grace alone independent of Period. Both were run: repo grep found nothing; vendor docs state the opposite."
  fix_rationale: "N/A — diagnose-only. Fix direction is doc-only: name both knobs and state alert_time = Period + Grace."
  blind_spots:
    - "Not empirically re-run against a live healthchecks.io account — relying on vendor docs plus the operator's observed A/B."
    - "Self-hosted healthchecks instances and non-healthchecks 'compatible' endpoints may differ in defaults; the runbook deliberately says 'healthchecks-compatible', so guidance should name the concept (expected interval) as well as the healthchecks.io knob name."
  candidate_causes:
    - "code (ELIMINATED): runner fails to ping. Refuted — 36-UAT.md Test 3 note confirms the mechanism works end-to-end once Period is set."
    - "config (PROXIMATE): operator's check left Period at default. True, but caused by the doc — the operator set exactly what the runbook named."
    - "documentation (ROOT): runbook:1543-1547 names one of the two required knobs."
    - "environment (ENABLING): healthchecks.io's Period default is 1 day, 96x the cron interval — a default that is silent and wrong for this use."
  and_gate: "yes — the failure needs BOTH the doc omission AND the vendor's 1-day default. If the default had been ~the ping interval, the omission would have been harmless. Only the doc half is within FOMO's control, so it is the actionable root cause; the vendor default is recorded as the enabling condition, which is why the fix must state the value explicitly rather than say 'leave the defaults'."

## Symptoms

expected: An operator following the runbook's "How do I run everything unattended?" heartbeat guidance (docs/runbooks/telescope_runs_calendar.rst, the "Heartbeat." paragraph around lines 1538-1547, plus setup step 3 around lines 1461-1463) configures a healthchecks-compatible check pointed at FOMO_HEARTBEAT_URL such that, when the crontab line is disabled, the heartbeat service alerts within roughly 20 minutes (a little over one 15-minute cron interval) with FOMO itself logging nothing and sending no email.
actual: The operator followed the runbook literally — set the healthchecks.io check's Grace to 20 minutes — and 28 minutes after the last ping the check was still green. Only after the operator discovered on their own that healthchecks.io has a separate "Period" setting (expected interval between pings, default 1 day) and set it to 15 minutes did the check go Late (orange) and then Down (red) with an alert email. The runner mechanism itself (pinging <url>/start then <url>/<exit-code>) is confirmed working; the defect is that the runbook never mentions the Period / schedule setting.
errors: None reported (no FOMO-side error; the failure is silent — the alert simply never fires).
reproduction: Test 3 in UAT (.planning/phases/36-unattended-operation/36-UAT.md). Read the runbook heartbeat guidance, configure a healthchecks.io check setting only what the runbook names (grace ~20 min), leave Period at its default, stop the cron line, observe no alert for ~24 h.
started: Discovered during UAT on 2026-09-17/18; the runbook text has said this since Phase 36 landed it.

## Eliminated

- hypothesis: "A FOMO-side code defect — the runner never sent the pings, or sent them to the wrong path."
  evidence: "36-UAT.md Test 3 note records the mechanism working end-to-end once Period=15 was set (Late -> Down -> alert email, FOMO logging and mailing nothing). ping_heartbeat() (solsys_code/unattended.py:135-159) builds `<url>/start` and `<url>/<exit-code>` correctly and raise_for_status()es since IN-01. The operator's own A/B changed only the check's Period."
  timestamp: 2026-09-17

- hypothesis: "The guidance exists but lives in a different operator-facing artifact the operator did not read (crontab example, logrotate example, check_unattended output, settings comments)."
  evidence: "Repo-wide grep over *.py/*.rst/*.example/settings: every heartbeat-adjacent occurrence of 'period' is the phrase 'grace period'. deploy/cron/fomo.crontab.example:31 repeats the same conflation. check_heartbeat() (check_unattended.py:235-244) only reports set/unset. settings.py:410-412 comments only on the unset branch. The omission is repo-wide, not a reading miss."
  timestamp: 2026-09-17

## Evidence

- timestamp: 2026-09-17
  checked: ".planning/debug/knowledge-base.md for a prior matching pattern"
  found: "No match — zero hits for heartbeat/healthcheck/grace/period/runbook."
  implication: "No known-pattern shortcut; investigate from first principles."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:1538-1547, the 'Heartbeat.' paragraph"
  found: "Verbatim: 'Point ``FOMO_HEARTBEAT_URL`` at any healthchecks-compatible endpoint (hosted or self-hosted) and configure one check per schedule, with a grace period a little above one 15-minute interval -- about 20 minutes is recommended, so one occasional slow tick does not page anyone.' One knob named (grace). 'one check per schedule' means one check per cron schedule, not 'set a schedule on the check'."
  implication: "An operator following this literally sets Grace=20 and touches nothing else. Primary defect site."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:1461-1473 (setup step 3) and :1474-1501 (step 4, check_unattended)"
  found: "Step 3 covers only exporting FOMO_HEARTBEAT_URL safely. Neither step tells the operator to create or configure the check at all — the only configuration guidance in the whole runbook is the one sentence at :1545."
  implication: "There is no second place in the setup walkthrough where the Period could have been picked up."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:1563-1565 ('When nothing has appeared' item 3)"
  found: "'A missing or stale ping (older than the configured grace period) means the tick itself never ran or never finished'."
  implication: "Second instance of the same conflation, and it actively misdirects triage: the correct staleness bound is Period + Grace, so an operator eyeballing the dashboard against Grace alone judges staleness against the wrong number."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:2035-2038 and the troubleshooting section :2055-2068"
  found: "':2035: The heartbeat's grace period ... is the structural backstop for exactly this case' — third instance. The only heartbeat troubleshooting entry (:2055) covers the OPPOSITE symptom (alerts fire while the log is healthy). There is no entry for 'the heartbeat never alerted'."
  implication: "The operator hitting this had no troubleshooting path to self-serve; they had to discover the Period knob unaided, which is what the trigger records."

- timestamp: 2026-09-17
  checked: "Repo-wide grep for 'period' across solsys_code/, docs/, deploy/, src/"
  found: "Every heartbeat-adjacent hit is 'grace period': runbook:1545/:1564/:2035, deploy/cron/fomo.crontab.example:31 ('the runner's own heartbeat grace period (D-12) is the structural backstop'), solsys_code/unattended.py:91 and :593-594 (docstrings). check_unattended.py:235-244 check_heartbeat() reports only set/unset. src/fomo/settings.py:410-412 comments only the unset branch."
  implication: "The omission is systemic across every artifact, not a single-line slip. Confirms it is one upstream error propagated, not independent typos."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-CONTEXT.md:133-143 (decision D-12)"
  found: "D-12 reads: 'so a never-invoked tick (no start within the grace period), a hung tick, and a failed tick all alert' and 'The runbook documents a grace period a little above one 15-minute interval (e.g. 20 minutes) and the recommended one-check-per-schedule setup.'"
  implication: "SMOKING GUN. D-12 uses 'grace period' as generic dead-man's-switch jargon meaning the WHOLE time-to-alert budget ('no start within the grace period' is only true if Period is already set to the ping interval). The runbook then transcribed that generic phrase as a vendor-specific instruction, where healthchecks.io has a narrower named knob called Grace. The root cause is upstream of the runbook: the decision itself conflated the two."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-RESEARCH.md:509-525 ('Healthchecks-style ping shape')"
  found: "Research cited healthchecks.io/docs/http_api/ for the PING API only ('You can append /start, /fail or /<exitcode> to the base ping URL...'). No citation, and no section, covering check CONFIGURATION (healthchecks.io/docs/configuring_checks/ — Period, Grace, cron schedules). RESEARCH.md:485 repeats 'the heartbeat's grace period (D-12 recommends ~20 minutes for a 15-minute schedule)'."
  implication: "Provenance of the error: research scoped the vendor read to the SENDING half (what FOMO transmits) and never the RECEIVING half (how the check decides to alert). The receiving-side configuration model was never in the phase's evidence base, so D-12 could not have got it right."

- timestamp: 2026-09-17
  checked: "healthchecks.io/docs/configuring_checks/ via WebSearch"
  found: "'Period is the expected time between pings, and Grace Time is the additional time to wait before sending an alert when a check is late.' Worked example: hourly period + 5 min grace, last ping 12:00 -> Late at 13:00, Down + alerts at 13:05. Auto-provisioned checks default to 'a Period of 1 day and Grace time of 1 hour'. Also: 'If you use the start signal ... Grace Time also specifies the maximum allowed time gap between start and success signals.'"
  implication: "Confirms alert_time = last_ping + Period + Grace, and confirms the 1-day default. With Grace=20m and Period=1 day, first alert is ~24h20m after the last tick — exactly matching 'still green at 28 minutes'. Also answers scope item 5: because the runner sends /start, Grace does DOUBLE duty and must still comfortably exceed a tick's runtime."

- timestamp: 2026-09-17
  checked: "healthchecks.io cron-schedule mode semantics via WebSearch"
  found: "In cron mode the check goes Late at the exact wall-clock moment the cron expression next matches, and Down at that moment + Grace. Worked example: '10 * * * *' with grace 5, last ping 12:30 -> Late 13:10, Down 13:15."
  implication: "A Cron-type check with '*/15 * * * *' is the drift-free alternative to Period=15, and gives the same ~Grace-after-the-missed-slot alert. Viable second option for the fix."

- timestamp: 2026-09-17
  checked: "git log -L 1538,1548 on the runbook"
  found: "The paragraph landed whole in commit c9c6fcf 'docs(36-05): the unattended-operation runbook section' — never edited since. Text has been wrong since Phase 36 landed it, as the symptom report states."
  implication: "Single-origin defect; no later regression to bisect."

- timestamp: 2026-09-17
  checked: "Why every quality gate passed it — 36-REVIEW.md:275 and 36-VERIFICATION.md:147, :288"
  found: "36-REVIEW.md:275, fixing the adjacent WR-14 lock paragraph, explicitly instructs: 'Keep the heartbeat-grace-period sentence as-is.' 36-VERIFICATION.md:147 marks the runbook criterion '✓ VERIFIED' on the grounds that it 'recommends ~20 min grace'. 36-VERIFICATION.md:288 defines the human test itself as 'with a grace period of about 20 minutes ... wait past the grace period'."
  implication: "Every gate checked the runbook for CONSISTENCY WITH D-12, and D-12 carried the conflation, so the error propagated through review and verification unchallenged — and was even written into the UAT test script. Only real-world execution against a live healthchecks.io account could catch it. This is the 'why not caught' answer: no gate validated a decision against the third-party system's actual behaviour."

## Resolution

root_cause: |
  Actionable root cause (documentation): docs/runbooks/telescope_runs_calendar.rst:1543-1547 tells the
  operator to configure the heartbeat check with only a "grace period ... about 20 minutes" and never
  names healthchecks.io's separate Period (expected interval between pings) setting. healthchecks.io
  alerts at last_ping + Period + Grace, and Period defaults to 1 day, so an operator who follows the
  runbook literally gets a check that first alerts ~24h20m after the schedule stops instead of within
  tens of minutes — the dead-man layer is silently disabled while looking correctly configured.

  Enabling condition (environment, outside FOMO's control): healthchecks.io's Period default is 1 day,
  96x the 15-minute cron interval, and is applied silently. The AND-gate fires: the omission is only
  harmful because of this default, which is precisely why the corrected guidance must state the value
  to set rather than rely on defaults.

  Upstream provenance (process): 36-RESEARCH.md:509-525 researched and cited only healthchecks.io's
  *ping API* (the half FOMO sends), never its *check-configuration* model (the half that decides when
  to alert). With the receiving side absent from the evidence base, decision D-12 (36-CONTEXT.md:133-143)
  wrote "grace period" as generic dead-man's-switch jargon for the total time-to-alert ("a never-invoked
  tick (no start within the grace period)"), and plan 36-05 transcribed that generic phrase into a
  vendor-specific instruction where "Grace" is a narrower named knob. Every downstream gate then checked
  the runbook against D-12 rather than against healthchecks.io, so review (36-REVIEW.md:275 — "Keep the
  heartbeat-grace-period sentence as-is") and verification (36-VERIFICATION.md:147 — "✓ VERIFIED ...
  recommends ~20 min grace") both endorsed it, and the same wrong instruction was written into the UAT
  test script itself (36-VERIFICATION.md:288).

fix: "NOT APPLIED — diagnose-only mode (goal: find_root_cause_only). Fix direction reported to caller for plan-phase --gaps."
verification: "N/A — diagnose-only."
files_changed: []
