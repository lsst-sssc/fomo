---
status: diagnosed
trigger: "the \"HeartBeat\" paragraph just starts \"Export `FOMO_HEARTBEAT_URL` in the environment\". There needs to be info or a step before this that says what it is, where to set it up, which website to go to, what to set period and grace time to"
created: 2026-09-17T00:00:00Z
updated: 2026-09-17T00:00:00Z
gap_id: G-36-1
phase: 36-unattended-operation
mode: diagnose-only
bug_class: Bohrbug (deterministic — a top-down read of the setup procedure reproduces the gap every time; no timing, no state)
---

## Current Focus

hypothesis: CONFIRMED — see Resolution.root_cause
test: (complete)
expecting: (complete)
next_action: "Return diagnosis to caller. Diagnose-only mode: no fix applied. No file other than this session file was modified."

reasoning_checkpoint:
  hypothesis: "The fresh-host setup procedure's content was derived from `check_unattended`'s prerequisite list, which by construction enumerates only prerequisites *this host* can verify. The heartbeat therefore entered the procedure as its host-side shadow -- an environment-variable export -- and never as the create-and-configure-a-remote-check task it actually is; every piece of check-side knowledge was assigned instead to the reference subsection 'The two failure signals' 80 lines further down, with no forward pointer from the procedure."
  confirming_evidence:
    - "36-CONTEXT.md D-05:84-85 specifies the setup walk as 'copy the template, run check_unattended, install the line, read the log, watch the heartbeat' -- 'watch', a monitoring verb, never 'create and configure'."
    - "36-05-PLAN.md:132-138 (setup item 2) gives the heartbeat exactly one clause: 'export FOMO_HEARTBEAT_URL and FOMO_BASE_URL in the environment cron sees ... Name every variable and setting, and quote no value'. The same plan's item 4 (:146-149, the failure-signals subsection) is where 'the recommended one-check-per-schedule setup, and a grace period' was assigned."
    - "git blame: the original step 3 (c9c6fcf, `docs(36-05)`) is a verbatim rendering of plan item 2 -- one sentence exporting both variables. The executor wrote precisely what was specified."
    - "Direct read of runbook:1448-1514: the setup subsection contains no cross-reference of any kind to 'The two failure signals', and the word 'heartbeat' appears in the whole document for the first time at :1461 as part of the variable name."
    - "36-06-PLAN.md Task 1:125-181 scopes all corrections to (a) the 'Heartbeat.' paragraph, (b) triage item 3, (c) the lock backstop, (d) a new troubleshooting entry; Task 2(e):244-250 adds one sentence to step 4. Step 3 is named nowhere in the plan."
  falsification_test: "Find (i) a forward pointer from the setup subsection to the heartbeat guidance, (ii) any mention of the heartbeat before step 3 that orients the operator, or (iii) any operator-facing artifact stating which service to use and how to obtain the ping URL. All three were run: (i) grep/read of 1448-1514 -- none; (ii) first 'heartbeat' hit in the file is :1461 -- none; (iii) repo grep for hc-ping/healthchecks.io outside .planning -- only `.planning/research/STACK.md` (not operator-facing) and two 'healthchecks.io calls this Period' asides. Hypothesis survives all three."
  fix_rationale: "N/A -- diagnose-only. Fix direction is doc-only and reported to the caller."
  blind_spots:
    - "SC 5's literal wording is 'from one runbook *section*', which a whole-section read satisfies; the gap's strongest legs are therefore the ordering inversion and the genuinely-absent content, not the placement alone. Called out explicitly in Resolution so the fix planner does not rest the case on placement."
    - "Not re-tested with a naive operator; relying on the reported symptom plus the structural read."
    - "Self-hosted healthchecks instances give a differently-shaped ping URL than hc-ping.com; guidance must name the concept (the check's own ping URL) as well as the hosted form."
  candidate_causes:
    - "code (ELIMINATED): the runner or preflight behaves wrongly. Refuted -- 36-UAT.md round 1 Test 3 confirmed the ping mechanism end-to-end; this round's report is purely about prose."
    - "documentation-structure (ROOT): the setup procedure has no create-the-check step, and the check-side knowledge lives only in a reference subsection 80 lines later with no pointer."
    - "documentation-content (ROOT, second half): which service to use and how to obtain the ping URL are absent from *every* operator-facing artifact, not merely misplaced -- they exist only in `.planning/research/STACK.md`."
    - "process (ENABLING): every gate on this section was a token-presence probe, and the one sufficiency gate (UAT Test 6) was administered to an operator already contaminated by out-of-band knowledge from Test 3 in the same session."
    - "process (ENABLING, second): 36-06's scope was copied from the prior debug session's artifact list, which had *observed* the placement gap and consumed it as corroboration for the one-knob content defect instead of filing it as its own defect."
  and_gate: "yes -- this needs all of: (1) the procedure's scope being host-verifiable prerequisites, so the off-host half had no slot; (2) D-12 assigning check-configuration guidance to 'the runbook' without naming a subsection, letting 36-05 place it in the reference lane; (3) gates that probe for substrings rather than sufficiency-at-point-of-use; (4) UAT Test 6 passing because its subject was no longer naive; (5) 36-06 inheriting its scope from a diagnosis that had already seen the placement gap but did not file it. Remove any one and the gap either never forms or is caught. The actionable pair is (1)+(5)."

## Symptoms

truth: An operator working the runbook's "Setting it up on a fresh host" steps top-down learns, at the point the heartbeat first appears (step 3, "Export FOMO_HEARTBEAT_URL"), what the heartbeat is, where to create the check (a healthchecks-compatible service such as healthchecks.io, hosted or self-hosted), which URL to copy into FOMO_HEARTBEAT_URL, and what to set the check's expected ping interval (Period: 15 min, or Cron type */15 * * * *) and grace time (Grace: ~20 min) to -- without having to discover the "Heartbeat." paragraph ~80 lines later under "The two failure signals".
expected: Same as truth — the fresh-host setup sequence is self-sufficient for the heartbeat (ROADMAP Phase 36 SC 5: "An operator can set up, or verify, the whole schedule on a fresh host from one runbook section without reading source").
actual: The "Heartbeat" setup step just starts "Export `FOMO_HEARTBEAT_URL` in the environment". There needs to be info or a step before this that says what it is, where to set it up, which website to go to, what to set period and grace time to.
errors: None reported (silent documentation-structure gap; nothing fails at runtime).
reproduction: Test 1 in .planning/phases/36-unattended-operation/36-UAT.md — the operator attempted to re-run UAT Test 3 (configure a live heartbeat check from the runbook alone) starting from the "Setting it up on a fresh host" steps in docs/runbooks/telescope_runs_calendar.rst and stopped at step 3 (~line 1461-1463), which only says to export FOMO_HEARTBEAT_URL.
started: Discovered during UAT on 2026-09-18, immediately after gap-closure plan 36-06 (commits 12c51c6, f075a7f, 1f3bbac) corrected the heartbeat alert-window guidance but placed/kept all of it in the "Heartbeat." paragraph under "The two failure signals" (~1541-1568) and in the step-4 preflight sentence (~1484-1487), not in setup step 3 itself.

## Eliminated

- hypothesis: "A regression introduced by gap-closure plan 36-06 — the 36-06 edits removed orientation that step 3 previously had."
  evidence: "git blame + `git show c9c6fcf3:docs/runbooks/telescope_runs_calendar.rst`: the original step 3 as landed by 36-05 was *less* informative, a single sentence covering both variables ('Export ``FOMO_HEARTBEAT_URL`` and ``FOMO_BASE_URL`` in the environment the cron daemon sees ... never as a literal value in any committed file'). 36-06 strictly added (Task 2(e)'s step-4 sentence). The gap has existed since c9c6fcf and 36-06 did not create or worsen it — it only failed to close it."
  timestamp: 2026-09-17

- hypothesis: "The information exists in the setup path but in a sibling operator artifact the operator did not open (the crontab template, the logrotate example, `check_unattended`'s own output)."
  evidence: "deploy/cron/fomo.crontab.example:13-15 lists FOMO_HEARTBEAT_URL by NAME only ('a healthchecks.io-compatible ping URL (optional; unset disables the heartbeat layer, D-12)') and :32-35 names both knobs but explicitly defers: 'See the runbook's \"How do I run everything unattended?\" section for the full setup' — i.e. it points back at the runbook. check_heartbeat() (check_unattended.py:235-252) prints only set/unset plus the Period reminder added by 36-06. Neither says which service, nor how to obtain the URL. The crontab template is also read at step 6, three steps *after* the export."
  timestamp: 2026-09-17

- hypothesis: "Spectrum-based fault localization could rank the defect site."
  evidence: "SKIPPED with note — Phase 1.25 is coverage-gated and this is a prose defect in an .rst file with no executing test that covers it. The nearest automated gates are substring greps (36-05-PLAN.md:195, 36-06-PLAN.md:189/253), which is itself part of the finding, not a localization signal."
  timestamp: 2026-09-17

## Evidence

- timestamp: 2026-09-17
  checked: "Phase 0 — `.planning/debug/knowledge-base.md` for a matching prior pattern (grep for heartbeat / healthcheck / runbook)"
  found: "Zero hits. The immediately-prior, directly-related session (`heartbeat-runbook-period-gap.md`) was diagnose-only and was never archived, so its findings were never written into the knowledge base."
  implication: "The KB offered no shortcut even though a same-file, same-paragraph diagnosis had completed hours earlier. Recurrence-guard observation for the eventual prevention entry: diagnose-only sessions currently leave no KB trace, so a second round starts cold."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst — subsection order of §'How do I run everything unattended?' (:1417) via heading grep"
  found: "Order is: 'What runs, and when' (:1426) -> 'Setting it up on a fresh host' (:1448) -> 'Adding a proposal to watch' (:1515) -> 'The two failure signals' (:1528) -> 'When nothing has appeared' (:1570) -> 'Running it by hand' (:1604). The procedure subsection precedes the reference subsection by 80 lines, and 'Adding a proposal to watch' sits between them."
  implication: "The section is ordered procedure-then-reference. Any prerequisite knowledge that lands in the reference lane is, by construction, unavailable to a top-down reader of the procedure."

- timestamp: 2026-09-17
  checked: "First mention of the heartbeat anywhere in the file (`grep -n -i 'heartbeat\\|healthcheck'`)"
  found: "The first hit in the entire 2100+-line document is :1461, inside setup step 3, as part of the variable name ``FOMO_HEARTBEAT_URL``. The preceding subsection 'What runs, and when' (:1426-1446) describes the four steps and the schedule and never mentions the heartbeat at all."
  implication: "At the moment the operator is told to export the variable, the word 'heartbeat' has no prior referent in the document. Confirms the reported symptom exactly: the step is the term's introduction *and* its only instruction."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:1461-1473 — setup step 3 verbatim"
  found: "Step 3 is three lines on the heartbeat ('Export ``FOMO_HEARTBEAT_URL`` in the environment the cron daemon sees (for example via ``/etc/environment``, or a wrapper script the crontab line sources) -- never as a literal value in any committed file.') followed by *ten* lines on ``FOMO_BASE_URL`` explaining what reads it, in which process, why both environments are needed, the simpler local_settings.py route, and the exact failure mode of getting it wrong."
  implication: "Striking intra-step asymmetry. The same numbered step gives one variable full why/where/failure-mode treatment and the other pure env-var hygiene. Whatever caused the asymmetry is the mechanism to find."

- timestamp: 2026-09-17
  checked: "`git blame -L 1451,1473` on the runbook, plus `git show c9c6fcf3:...` for the original text"
  found: "Lines 1461-1473 all carry 26ee834d, 'fix(36): WR-13 document FOMO_BASE_URL as required by the web process too'. The original (c9c6fcf, 'docs(36-05): the unattended-operation runbook section') read: '3. Export ``FOMO_HEARTBEAT_URL`` and ``FOMO_BASE_URL`` in the environment the cron daemon sees ... -- never as a literal value in any committed file.' — one sentence, both variables, no orientation for either."
  implication: "MECHANISM FOUND for the asymmetry. Both variables started equally bare. FOMO_BASE_URL got its ten lines because code review raised WR-13 against it. Nothing ever raised a finding against the heartbeat's *setup* step, so it kept the original shape. The step's quality tracks which lines a review happened to touch, not what an operator needs."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-05-PLAN.md:132-138 (setup item 2) and :142-149 (failure-signals item 4)"
  found: "Item 2 specifies for the heartbeat only: 'export ``FOMO_HEARTBEAT_URL`` and ``FOMO_BASE_URL`` in the environment cron sees', closing with 'Name every variable and setting, and quote no value -- say what to set, never what it is set to.' Item 4 is where the check-side content was assigned: 'the recommended one-check-per-schedule setup, and a grace period a little above one interval (about 20 minutes for the 15-minute schedule) (D-12)'."
  implication: "SMOKING GUN. The plan itself split the heartbeat in two: its host-side half into the procedure, its check-side half into the reference subsection. The executor's output is a faithful rendering of that split. The defect is in the plan's allocation of content to subsections, not in execution."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-05-PLAN.md:195 and :200-209 — the task's automated verify and acceptance criteria"
  found: "The automated gate is a substring-presence probe: `print('How do I run everything unattended?' in src, ..., 'FOMO_HEARTBEAT_URL' in src, ...)`. The acceptance criterion is 'That section contains all six sub-headings ... and names ``FOMO_HEARTBEAT_URL``, ``FOMO_BASE_URL``, ``check_unattended`` ...'. Nothing tests any subsection for sufficiency at its point of use."
  implication: "Step 3's bare export line satisfies every gate the plan defined. `'FOMO_HEARTBEAT_URL' in src` is True whether the operator can act on it or not."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-CONTEXT.md:24-27 (the SC-5 truth), :77-85 (D-05), :133-143 (D-12)"
  found: "The SC-5 truth enumerates the section's deliverable as 'a committed crontab template and logrotate example, plus a `check_unattended` management command that verifies every prerequisite (flock, lock/log dirs, email backend and staff recipients, heartbeat URL, at least one active watched proposal)'. D-05 lists the same prerequisites and specifies the walk as 'copy the template, run `check_unattended`, install the line, read the log, watch the heartbeat'. D-12 ends: 'The runbook documents a grace period a little above one 15-minute interval (e.g. 20 minutes) and the recommended one-check-per-schedule setup' — 'the runbook', with no subsection named."
  implication: "UPSTREAM ROOT. The setup procedure's content was derived from `check_unattended`'s prerequisite list — a list of things *this host* can be inspected for. The heartbeat appears in it only as 'heartbeat URL', i.e. as an environment variable. The off-host task (create a check on a service, set two knobs on it) is not a host prerequisite and so had no slot in the procedure by construction; D-05's verb for it is 'watch', not 'create'. D-12 then left the placement of the check-side guidance unspecified, so 36-05 put it where it was named — the failure-signals lane."

- timestamp: 2026-09-17
  checked: "Whether the setup subsection (:1448-1514) contains any pointer to the heartbeat guidance"
  found: "No cross-reference of any kind. The only heartbeat hint after step 3 is inside step 4's 30-line `check_unattended` paragraph (:1484-1487): 'whether ``FOMO_HEARTBEAT_URL`` is set (and reminds you that the check at the other end still needs its own expected ping interval set -- the preflight can only see this host's environment variable, never the remote check's own configuration)'. That is the first hint that a remote check exists at all, it names no value, and it points at nothing."
  implication: "Two distinct defects. (a) ORDERING INVERSION: step 3 asks the operator to export a URL that only exists *after* a check has been created, and nothing in steps 1-3 tells them to create one — a numbered procedure with an unstated prerequisite before its own first mention. (b) NO POINTER: the operator has no route from the procedure to the paragraph that would answer them."

- timestamp: 2026-09-17
  checked: "docs/runbooks/telescope_runs_calendar.rst:1541-1568 — the 'Heartbeat.' paragraph as 36-06 left it"
  found: "It is complete on the two knobs: names the concept then healthchecks.io's spelling (``Period``/``Grace``), sets the expected interval to 15 min, offers the Cron-type ``*/15 * * * *`` alternative, keeps grace at ~20 min with the /start-to-completion reason it must not shrink, states alert = last ping + interval + grace (late ~15 min, alert ~35 min), and names the 1-day default as the trap. But its only 'where' is 'Point ``FOMO_HEARTBEAT_URL`` at any healthchecks-compatible endpoint (hosted or self-hosted) and configure one check per schedule'."
  implication: "Even an operator who *does* find the paragraph gets two of the four things they asked for. 'Which website to go to' and 'which URL to copy' are not there. So this is not purely a placement defect — part of the requested content does not exist at the destination either."

- timestamp: 2026-09-17
  checked: "Repo-wide grep for `hc-ping` / `healthchecks.io` / `healthchecks-compatible` across *.rst, *.py, *.example, *.ipynb, *.md, excluding .planning/"
  found: "`hc-ping.com` appears in exactly one place in the repository: `.planning/research/STACK.md` (:22, :47-48, which carry the `https://hc-ping.com/<uuid>` form). In operator-facing files, healthchecks.io appears only as parenthetical asides — runbook :1550 ('healthchecks.io calls this ``Period``'), :1565 ('1 day on healthchecks.io'), :2096, and crontab.example:14/:33. No operator-facing artifact says to sign up for or self-host anything, create a check, or copy its ping URL."
  implication: "Confirms the second defect independently. The service-and-provenance half of the answer was captured during research, never promoted into any shipped doc, and is invisible to an operator. Also confirms 36-CONTEXT.md D-12's 'Phase 31 confirmed `hc-ping.com` egress' never reached an operator-facing page."

- timestamp: 2026-09-17
  checked: ".planning/debug/heartbeat-runbook-period-gap.md:67-71 — the prior session's own Evidence entry on the setup steps"
  found: "Verbatim: checked 'runbook:1461-1473 (setup step 3) and :1474-1501 (step 4)'; found 'Step 3 covers only exporting FOMO_HEARTBEAT_URL safely. **Neither step tells the operator to create or configure the check at all** -- the only configuration guidance in the whole runbook is the one sentence at :1545'; implication 'There is no second place in the setup walkthrough where the Period could have been picked up.'"
  implication: "SECOND SMOKING GUN. The prior diagnosis observed *this exact gap* and consumed it as corroboration for the one-knob content defect ('no second place where Period could have been picked up') rather than filing it as a defect in its own right. Its Resolution.root_cause and its seven-entry `artifacts` list therefore name :1543-1547, :1563-1565, :2035-2038, :2055-2068, the crontab line, check_unattended and two docstrings — and not step 3."
  implication_2: "This is the actionable process failure: an observation that would have prevented this round was recorded in the right file, in the right session, and then not promoted from evidence to artifact."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-06-PLAN.md — Task 1 :118-187, Task 2 :208-251, and both verify blocks :189, :253"
  found: "Task 1's five edits are scoped to (a) rewrite the 'Heartbeat.' paragraph *in place* ('Keep its first three sentences ... Replace the final sentence'), (b) triage item 3, (c) the lock backstop sentence, (d) a new troubleshooting entry. Task 2's five edits cover the crontab comment, two docstrings, `check_heartbeat()`'s detail string, a new test, and (e) 'One sentence in ... the setup step 4 paragraph (~1484)'. Setup step 3 is named in neither task. Both verify blocks are grep probes (`grep -c Period ... -ge 4`, `grep -q 'expected interval'`, `grep -q 'heartbeat never alerted'`)."
  implication: "36-06 inherited the prior diagnosis's framing exactly — 'wrong content at a known site' — and so corrected the content where the wrong content already lived. The one edit it made inside the setup procedure, 2(e), was justified by the paired-docs rule (keeping the runbook in step with `check_unattended`'s new output line), not by operator sufficiency, which is why it landed in step 4 and not step 3."

- timestamp: 2026-09-17
  checked: "Previous UAT round via `git show HEAD~4:.planning/phases/36-unattended-operation/36-UAT.md` — Test 3 and Test 6"
  found: "Test 3 ('The heartbeat's dead-man half') is where the operator discovered the Period knob themselves ('set period to 15 minutes, now have orange pling'). Test 6 ('Runbook sufficiency' — 'given only the runbook's section, reaches a working, checked schedule on a fresh host ... (SC 5)') is recorded `result: pass`, 'User verdict: pass', with a note listing as *optional* clarifications: 'where FOMO_HEARTBEAT_URL / FOMO_BASE_URL are read (env var vs local_settings.py, cron vs web process) and that installing /etc/logrotate.d/fomo and forcing a rotation need sudo.'"
  implication: "WHY NOT CAUGHT. The only gate designed to catch exactly this gap — the SC-5 sufficiency read-through — was administered in the same session, *after* Test 3 had already taught the operator the missing knowledge out of band. A sufficiency test run by a no-longer-naive reader cannot detect missing orientation. The near-miss was even written down ('where FOMO_HEARTBEAT_URL ... is read') and classified as an optional clarification."

- timestamp: 2026-09-17
  checked: ".planning/phases/36-unattended-operation/36-VERIFICATION.md:112 — the SC-5 row"
  found: "'✓ VERIFIED' on the evidence 'one section covering what runs and when, fresh-host setup (7 numbered steps), adding a watched proposal, both failure signals, ... **UAT Test 6 passed** on a real read-through (user verdict: pass), with two optional clarification notes recorded in `36-UAT.md` (where ``FOMO_HEARTBEAT_URL``/``FOMO_BASE_URL`` are read; that logrotate install needs sudo)'."
  implication: "SC 5 was verified structurally (the subsections exist, the steps are numbered) plus the contaminated Test 6 pass. The verification record itself carries the unpromoted observation. Note also the literalism risk: SC 5 says 'from one runbook *section*', which a whole-section read satisfies — so the fix case must rest on the ordering inversion and the absent content, both of which hold regardless of how SC 5 is read."

- timestamp: 2026-09-17
  checked: "Scope item 3 — the same 'procedure assumes knowledge only the reference lane (or nothing) supplies' pattern across the other six setup steps"
  found: |
    Step 1 (:1451-1456, create /var/lock/fomo and /var/log/fomo): names the paths and the ownership
      requirement. Self-sufficient except that creating root-owned dirs and chowning them to the cron
      account needs sudo, which is unstated. MINOR — same class, low impact (round-1 UAT flagged the
      sudo point for logrotate and it was filed as optional).
    Step 2 (:1457-1460, real EMAIL_BACKEND / EMAIL_HOST_* and the LCO/SOAR API key in
      local_settings.py): names the settings but not how to write the API key. The real structure is
      nested — `FACILITIES['LCO']['api_key']` and `FACILITIES['SOAR']['api_key']` (src/fomo/settings.py
      :235-247) — and that nesting appears in NO doc: `grep 'local_settings\|api_key' docs/**/*.rst`
      returns only runbook :1458, :1469 and docs/installation.rst:113. SAME CLASS, MODERATE — but far
      lower impact than the heartbeat, because any working FOMO deployment has already set the API key
      for normal operation; it is not unattended-specific.
    Step 3, FOMO_BASE_URL half (:1464-1473): fully self-sufficient — what reads it, in which process,
      both environments, the simpler route, the failure mode. This is the counter-example that proves
      the class is fixable within a numbered step, and it is only this good because WR-13 forced it.
    Step 3, FOMO_HEARTBEAT_URL half (:1461-1463): the reported gap. MAJOR, and the only step whose
      object lives on a third-party service rather than on this host.
    Step 4 (:1474-1504, check_unattended): self-sufficient and then some.
    Step 5 (:1505-1506, re-run until green): self-sufficient.
    Step 6 (:1507-1510, install the printed cron line): self-sufficient; also points at
      deploy/cron/fomo.crontab.example.
    Step 7 (:1511-1513, drop deploy/logrotate/fomo.example into /etc/logrotate.d/fomo): self-sufficient
      on placement ('or wherever this host's logrotate scans'); sudo unstated. MINOR, same as step 1.
  implication: "The pattern is real but sharply concentrated. The procedure is complete for everything whose object is on this host (directories, settings, environment, the cron line, the logrotate file) and incomplete for (a) the one object that lives off-host — the remote heartbeat check — and (b) two things that need knowledge from outside the section (the FACILITIES nesting; sudo). The heartbeat is the only step whose prerequisite `check_unattended` structurally cannot verify — step 4 :1486-1487 now says so in as many words — which is exactly why deriving the procedure from the preflight's prerequisite list dropped it. Reported for the fix planner's scope decision; not fixed here."

- timestamp: 2026-09-17
  checked: "The healthchecks facts a fix would need to state, against the repository's own prior findings (no new external research needed)"
  found: |
    From `.planning/debug/heartbeat-runbook-period-gap.md` Evidence :97-105 (vendor docs, recorded last
    session): 'Period is the expected time between pings, and Grace Time is the additional time to wait
    before sending an alert when a check is late'; alert at last_ping + Period + Grace; auto-provisioned
    checks default to Period 1 day / Grace 1 hour; with /start in use, Grace also bounds the maximum
    allowed gap between the start and success signals. Cron-type checks go Late at the moment the cron
    expression next matches and Down at that moment + Grace.
    From `.planning/research/STACK.md`:22/:47-48 — a check's ping URL has the form
    `https://hc-ping.com/<uuid>` on the hosted service; the free tier (20 checks, no card) covers this
    project's needs; self-hosting `healthchecks/healthchecks` is the stated fallback.
    From 36-CONTEXT.md D-12 :136-139 — service-agnostic by decision ('any healthchecks-compatible
    endpoint, hosted or self-hosted'), and 'Phase 31 confirmed `hc-ping.com` egress'.
    D-15 (:154-157) restricts *values*: credentials are env vars, and `check_unattended` prints
    'variable *names* and set/unset status, never values'.
  implication: "Every fact the fix needs is already in the repository — none of it requires new research. And D-15 does not block the missing half: `https://hc-ping.com/<uuid>` is a placeholder form, not a value, and the page already uses placeholder paths and placeholder proposal codes by its own convention. 36-06 Task 1's instruction 'Do not quote a real ping URL or UUID anywhere' was correct about real URLs but was applied so as to suppress the generic form too, which is why the provenance never got written. (No real URL, UUID or FOMO_* value appears in this session file.)"

## Resolution

root_cause: |
  Actionable root cause (documentation structure, two halves):

  (1) PLACEMENT — the fresh-host procedure has no create-the-check step, by construction. The setup
  subsection's content was derived from `check_unattended`'s prerequisite list: 36-CONTEXT.md's SC-5
  truth (:24-27) and D-05 (:77-85) both enumerate the deliverable as that list of host-inspectable
  prerequisites, in which the heartbeat appears only as "heartbeat URL" — an environment variable —
  and D-05's verb for it is "watch the heartbeat", never "create and configure a check". 36-05-PLAN.md
  then split the heartbeat along exactly that line: its host-side half into setup item 2 ("export
  `FOMO_HEARTBEAT_URL` and `FOMO_BASE_URL` in the environment cron sees ... quote no value", :132-138)
  and its entire check-side half into failure-signals item 4 ("the recommended one-check-per-schedule
  setup, and a grace period", :146-149). D-12 (:141-143) had said only that "the runbook documents"
  the check setup, naming no subsection, so nothing contested the placement. The executor rendered the
  plan faithfully (c9c6fcf). Consequence: the document's first mention of the heartbeat is the variable
  name in step 3 (:1461) with no prior referent, the step asks the operator to export a URL that only
  exists after a check has been created, nothing in steps 1-3 says to create one, and the procedure
  contains no cross-reference to the paragraph 80 lines later that would explain any of it.

  (2) CONTENT — two of the four things the operator asked for exist in no operator-facing artifact at
  all, so they cannot be reached by moving text. The corrected "Heartbeat." paragraph (:1541-1568) is
  complete on the two knobs but its only "where" is "point `FOMO_HEARTBEAT_URL` at any
  healthchecks-compatible endpoint (hosted or self-hosted)". Which service to use and how to obtain
  the check's ping URL live only in `.planning/research/STACK.md` (:22, :47-48 — the
  `https://hc-ping.com/<uuid>` form, the free tier, the self-hosted fallback), which no operator
  reads; `hc-ping` appears nowhere else in the repository, and healthchecks.io appears in shipped docs
  only as the parenthetical "healthchecks.io calls this `Period`". The provenance half was captured in
  research and never promoted into a shipped page.

  Why 36-06 did not close it (process, actionable): 36-06's scope was inherited verbatim from the
  prior session's artifact list. That session had *already observed this gap* —
  `.planning/debug/heartbeat-runbook-period-gap.md`:67-71 records "Neither step tells the operator to
  create or configure the check at all" — but consumed it as corroboration for the one-knob content
  defect ("there is no second place in the setup walkthrough where the Period could have been picked
  up") instead of filing it as a defect of its own. Its root cause and seven-entry `artifacts` list
  therefore named :1543-1547, :1563-1565, :2035-2038, :2055-2068, the crontab comment,
  `check_heartbeat()` and two docstrings — never step 3. 36-06 Task 1 accordingly rewrote the wrong
  sentence *in place* ("Keep its first three sentences ... Replace the final sentence") and Task 2(e)
  touched the procedure only at step 4, justified by the paired-docs rule (keeping the runbook in step
  with `check_unattended`'s new output line) rather than by operator sufficiency.

  Enabling conditions (process, not directly actionable by a doc edit):
  - Every automated gate on this section is a token-presence probe — 36-05-PLAN.md:195
    (`'FOMO_HEARTBEAT_URL' in src`), :203 ("names `FOMO_HEARTBEAT_URL`..."), 36-06-PLAN.md:189/:253
    (`grep -c Period ... -ge 4`, `grep -q "expected interval"`). Step 3's bare export line satisfies
    all of them. No gate tests a subsection for sufficiency at its point of use.
  - The one gate that could have caught it, UAT Test 6 ("Runbook sufficiency ... given only the
    runbook's section, reaches a working, checked schedule on a fresh host (SC 5)"), was administered
    in the same session *after* Test 3 had taught the operator the missing knowledge out of band. It
    passed ("user verdict: pass") and its own note recorded the near-miss — "where
    `FOMO_HEARTBEAT_URL`/`FOMO_BASE_URL` are read" — as an optional clarification; 36-VERIFICATION.md
    :112 then marked SC 5 "✓ VERIFIED" citing that pass plus the structural presence of 7 numbered
    steps. A sufficiency test read by a no-longer-naive operator cannot detect missing orientation.
  - The contrast inside the same step proves the class is fixable and that quality tracked review
    attention rather than operator need: `FOMO_BASE_URL` got ten lines of what-reads-it / which-process
    / failure-mode treatment because code review raised WR-13 (commit 26ee834d rewrote the whole step);
    `FOMO_HEARTBEAT_URL` kept the original bare shape because no finding was ever raised against it.
  - D-15 restricts *values*, not provenance. 36-06 Task 1's "do not quote a real ping URL or UUID"
    was right about real URLs but was applied so as to suppress the generic `https://hc-ping.com/<uuid>`
    placeholder form too — which the page's own placeholder-path / placeholder-proposal-code convention
    already permits. This is why the "which URL to copy" half never got written.

  AND-gate: fires. The gap requires all five of — the procedure's scope being host-verifiable
  prerequisites; D-12 leaving the check-side guidance's placement unspecified; token-presence gates;
  a contaminated sufficiency test; and 36-06 inheriting a scope that had seen but not filed the
  placement gap. The actionable pair is the first and the last.

fix: "NOT APPLIED — diagnose-only mode (goal: find_root_cause_only). No file other than this session file was modified."
verification: "N/A — diagnose-only."
files_changed: []
