---
status: diagnosed
trigger: "the no active WatchedProposal prints twice, once uncolored and once in red"
created: 2026-09-18T00:00:00Z
updated: 2026-09-18T00:00:00Z
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: CONFIRMED -- `Command.handle()` (check_unattended.py:614-617) builds one `line` string and
  writes that same string to two streams: unconditionally to `self.stdout`, then again to
  `self.stderr` whenever `status != 'ok'`. On a tty (or under any `2>&1`) both streams are one
  destination, so each WARN/FAIL result renders twice. The colour difference is Django's
  `BaseCommand.__init__` setting `self.stderr.style_func = self.style.ERROR` (bold red, ANSI 31;1)
  while `self.stdout` keeps the identity style_func.
test: DONE -- read handle(); inspected Django 5.2.17 BaseCommand/OutputWrapper source; ran
  `python manage.py check_unattended` with streams separated, merged, and under a pty; ran git blame
  over the emission block; read the test helper.
expecting: MET on every count (see Evidence).
next_action: NONE -- goal is find_root_cause_only. Return the diagnosis. No source file modified.

bug_class: Bohrbug (fully deterministic; reproduces on every run given the three co-conditions below).

reasoning_checkpoint:
  hypothesis: "handle()'s unconditional dual-write of one identical string to stdout and stderr makes
    every non-ok result render twice whenever the two streams share a destination; Django's default
    stderr ERROR styling makes the second copy red."
  confirming_evidence:
    - "Direct source read: check_unattended.py:615 self.stdout.write(line); :616-617 if status != 'ok': self.stderr.write(line) -- same `line` variable."
    - "Separated-stream run: out.txt has the WARN line exactly once, err.txt has it exactly once."
    - "Merged run (2>&1): grep -c 'watched_proposals' == 2."
    - "pty capture: line 25 plain, line 26 prefixed ^[[31;1m (bold red) then ^[[0m -- the exact 'uncoloured then red' pair reported."
    - "Django 5.2.17 BaseCommand.__init__ sets self.stderr.style_func = self.style.ERROR; OutputWrapper.style_func setter no-ops unless isatty()."
  falsification_test: "If the duplicate came from anything other than the dual-write (double command
    invocation, a logging handler, a signal receiver), the separated-stream run would show the line
    twice in ONE file, or zero times in the other. It showed exactly one copy in each."
  fix_rationale: "N/A -- diagnose-only run; fix is planned separately."
  blind_spots: "Not tested: behaviour under --no-color (should drop the red but keep the duplicate);
    behaviour when a hard check fails (more non-ok lines, all duplicated) -- both follow from the same
    code path but were not executed."
  candidate_causes:
    - "code: handle()'s unconditional dual-write of one string to two sinks (THE DEFECT)"
    - "environment: stdout and stderr resolving to the same sink -- a tty, or a `2>&1` redirect (necessary co-condition, not a defect)"
    - "data: at least one non-ok CheckResult -- here zero active WatchedProposal rows (necessary trigger, valid data)"
    - "config/framework: Django's default self.stderr.style_func = ERROR (explains the colour asymmetry only, not the duplication)"
  and_gate: "YES -- the visible symptom requires all three of (code dual-write) AND (streams merged)
    AND (>=1 non-ok result) simultaneously. Only the first is actionable; the other two are legitimate
    operating conditions the code failed to account for."

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: `python manage.py check_unattended` prints each CheckResult exactly once; the
  watched_proposals [WARN] line appears a single time, colored consistently with the other result lines.
actual: On the operator's real host (Django dev checkout, uid 10007, all hard checks [ok]) the output
  ends with the same line printed twice, the first uncolored and the second in red:
    [WARN] watched_proposals: no active WatchedProposal rows -- discovery will be a quiet no-op; add one in the admin
    [WARN] watched_proposals: no active WatchedProposal rows -- discovery will be a quiet no-op; add one in the admin
  Every other [ok] line printed once. Exit status and the cron line were fine.
errors: None reported (only the duplicate line).
reproduction: Test 1 in .planning/phases/36-unattended-operation/36-UAT.md -- run
  `python manage.py check_unattended` on a host with zero active WatchedProposal rows and a real (smtp)
  EMAIL_BACKEND, all hard checks passing.
started: Discovered during UAT round 3 (2026-09-18), after phase-36 code-review fix commits
  14ae0bf..27c722a landed. Unknown whether it predates them.

## Eliminated
<!-- APPEND only - prevents re-investigating -->

- hypothesis: "The WR-35 fix (commit 8a4d2bc, 'report the non-system-flock note as [WARN], not [ok]')
    introduced the duplicate -- named in the symptom report as the first place to look."
  evidence: "`git blame -L 607,618` puts the entire emission block, including both the stdout write
    and the conditional stderr write, at fc2e2bc ('feat(36-04): check_unattended -- the prerequisite
    report', 2026-09-17) -- the ORIGINAL feature commit, which predates the whole 14ae0bf..27c722a
    review-fix range. `git log -L 605,620` shows only two commits ever touched those lines: fc2e2bc
    (creation) and d8d7f14 (added the cron-line block AFTER them, leaving the loop untouched). WR-35
    changed only check_flock()'s return value (ok=True -> ok=False/hard=False), which adds a SECOND
    line capable of duplicating but does not create the mechanism."
  timestamp: 2026-09-18T00:00:00Z

- hypothesis: "The command is being invoked twice, or a logging handler / signal receiver re-emits
    the warning."
  evidence: "With stdout and stderr redirected to separate files, out.txt contains the WARN line
    exactly once and err.txt contains it exactly once. A double invocation or an extra emitter would
    put two copies in one file. Also, only the single non-ok line duplicates -- all nine [ok] lines
    appear exactly once -- which matches the `if status != 'ok'` guard precisely."
  timestamp: 2026-09-18T00:00:00Z

- hypothesis: "This is a project-wide management-command convention, so the whole command set is
    affected."
  evidence: "Every other command in solsys_code/management/commands/ that writes to stderr
    (backfill_lco_observation_records, cutover_classical_allocations, fetch_jplsbdb_objects,
    import_campaign_csv, load_telescope_runs, project_observation_calendar, reconcile_campaign_runs,
    sync_gemini_observation_calendar) routes each message to exactly ONE stream -- spot-checked
    project_observation_calendar.py:190-194 and reconcile_campaign_runs.py:63-74. check_unattended is
    the only command that writes the same string to both."
  timestamp: 2026-09-18T00:00:00Z

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-09-18T00:00:00Z
  checked: .planning/debug/knowledge-base.md (Phase 0 known-pattern scan)
  found: No entry matching stdout/stderr duplication, Django management-command output, or terminal
    coloring. Nine entries, all data/model/lookup-key bugs.
  implication: No prior art to reuse; investigate from first principles.

- timestamp: 2026-09-18T00:00:00Z
  checked: solsys_code/management/commands/check_unattended.py:607-617 (Command.handle result loop)
  found: |
    for result in results:
        ... status = 'ok' | 'FAIL' | 'WARN'
        line = f'[{status}] {result.name}: {result.detail}'
        self.stdout.write(line)
        if status != 'ok':
            self.stderr.write(line)
    The SAME `line` string is written to stdout unconditionally and to stderr again whenever the
    status is not 'ok'.
  implication: Two physical writes per non-ok result. On a tty (streams merged) that is one visible
    duplicate per WARN/FAIL line -- and only for non-ok lines, which matches "every other [ok] line
    printed once".

- timestamp: 2026-09-18T00:00:00Z
  checked: Django 5.2.17 source -- BaseCommand.__init__ and OutputWrapper
    (/home/tlister/venv/devel_fomo311_venv/lib64/python3.11/site-packages/django/core/management/base.py)
  found: |
    self.stdout = OutputWrapper(stdout or sys.stdout)
    self.stderr = OutputWrapper(stderr or sys.stderr)
    ...
    else:
        self.style = color_style(force_color)
        self.stderr.style_func = self.style.ERROR
    and OutputWrapper's style_func setter:
        if style_func and self.isatty(): self._style_func = style_func
        else: self._style_func = lambda x: x
  implication: stderr is styled ERROR (bold red) by default; stdout has no style_func, so it stays
    uncoloured. That is exactly the reported "first uncoloured, second in red". It also explains why
    the styling is invisible in tests: the setter no-ops for a non-tty sink like io.StringIO.

- timestamp: 2026-09-18T00:00:00Z
  checked: Live reproduction on this checkout (0 active WatchedProposal rows) --
    `python manage.py check_unattended > out.txt 2> err.txt`, then `... 2>&1 | grep -c watched_proposals`
  found: |
    exit=0 ("9/10 checks passed"). out.txt: nine [ok] lines + exactly ONE
    "[WARN] watched_proposals: no active WatchedProposal rows ..." line, followed by the blank line,
    the "Cron line to install" header, the cron line and the summary. err.txt: startup noise
    (NumExpr/spiceypy/DRF/system-check warnings) + exactly ONE copy of the same WARN line.
    Merged (2>&1): grep -c == 2.
  implication: Reproduces deterministically on this dev checkout. One copy per stream; the doubling is
    purely the merge of the two streams onto one destination.

- timestamp: 2026-09-18T00:00:00Z
  checked: pty reproduction -- `script -qec "python manage.py check_unattended" /dev/null`
  found: |
    line 24: [ok] facility_credentials: LCO and SOAR api_key both set
    line 25: [WARN] watched_proposals: no active WatchedProposal rows ...        <- no ANSI
    line 26: ^[[31;1m[WARN] watched_proposals: no active WatchedProposal rows ... <- bold red
    line 27: ^[[0m
    line 28: Cron line to install (both host directories above must exist first):
  implication: Exact symptom match, including the colour asymmetry. One nuance vs. the report: the
    pair is NOT at the end of the output -- it sits between the facility_credentials [ok] line and the
    cron-line block. Cosmetic discrepancy in the report only; does not affect the diagnosis.

- timestamp: 2026-09-18T00:00:00Z
  checked: git blame -L 607,618 and git log -L 605,620 on check_unattended.py
  found: All of lines 607-617 (the status classification, the stdout write, and the conditional stderr
    write) are attributed to fc2e2bc, "feat(36-04): check_unattended -- the prerequisite report"
    (2026-09-17). Only one later commit, d8d7f14, touched the region, and it only appended the
    cron-line block after the loop. None of 05f03fb / 74fe17b / 8d7b3d1 / 8da8058 / 8a4d2bc (the
    8f9b045..27c722a review-fix commits that touched this file) changed the emission block.
  implication: Not a regression from the review-fix range. The bug has been present since the
    command was first written.

- timestamp: 2026-09-18T00:00:00Z
  checked: .planning/phases/36-unattended-operation/36-04-PLAN.md:182
  found: "`handle()` runs every check in that order, writes one aligned `[ok] / [WARN] / [FAIL]
    <name>: <detail>` line per result to `self.stdout` (warnings and failures also to `self.stderr`)"
  implication: The dual-write was specified in the plan, not introduced by accident during execution.
    The design never considered that on an interactive terminal (or under `2>&1`) the two sinks are
    one destination. The defect is in the design decision as much as the code.

- timestamp: 2026-09-18T00:00:00Z
  checked: solsys_code/tests/test_check_unattended.py -- the `_run()` helper (86-88) and every
    assertion touching stdout/stderr
  found: |
    def _run(*args, **kwargs):
        stdout, stderr = io.StringIO(), io.StringIO()
        call_command('check_unattended', *args, stdout=stdout, stderr=stderr, **kwargs)
        return stdout.getvalue(), stderr.getvalue()
    Two SEPARATE sinks -- the merge never happens, so each capture holds exactly one copy. Worse, two
    tests actively PIN the dual-write as intended behaviour:
      - test_flock_outside_system_directories_gets_a_sanity_note: assertIn('[WARN] flock', stdout) at
        :170 AND assertIn('[WARN] flock', stderr) at :172 (same for the symlink variant at :222/:224)
      - test_unset_heartbeat_is_a_warning: assertIn(..., stdout) at :400 AND assertIn(..., stderr) at :401
    test_warnings_do_not_mask_a_hard_failure (:418) does build `combined = stdout + stderr`, but only
    asserts presence, never a count. Grep for count-based assertions over captured output returns
    nothing (the only `.count(` is WatchedProposal.objects.count() at :353).
  implication: The test suite is structurally blind to this class of bug -- no test ever renders the
    two streams into one sink, and no test counts occurrences. Any fix must also revise :170-172,
    :222-224 and :400-401, which currently encode the buggy behaviour as the expectation.

- timestamp: 2026-09-18T00:00:00Z
  checked: All nine other commands under solsys_code/management/commands/ that call self.stderr.write
  found: Each routes a given message to exactly one stream (spot-checked
    project_observation_calendar.py:190-194 -- `if message: self.stderr.write(message)`, no stdout
    twin; reconcile_campaign_runs.py:63-74 -- stderr-only on the failure and skip paths).
    check_unattended is the sole dual-writer in the repo.
  implication: Scope is one command. Emitting each line to exactly one stream would also bring
    check_unattended back in line with the existing repo convention.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  solsys_code/management/commands/check_unattended.py:614-617 -- Command.handle() builds one `line`
  string per CheckResult and writes that identical string to TWO sinks: unconditionally to
  `self.stdout` (:615), then again to `self.stderr` (:617) whenever `status != 'ok'`. stdout and
  stderr are the same destination on an interactive terminal (and under any `2>&1` redirect), so
  every WARN/FAIL result is rendered twice. The two copies differ in colour because Django's
  `BaseCommand.__init__` sets `self.stderr.style_func = self.style.ERROR` (bold red, ANSI 31;1) while
  `self.stdout` keeps OutputWrapper's identity style_func -- hence "once uncoloured and once in red".
  In the reported run, `watched_proposals` was the ONLY non-ok result (all nine other checks [ok]),
  so it was the only line to double. Present since fc2e2bc, the command's original feature commit;
  specified as intended behaviour in 36-04-PLAN.md:182 and pinned by the existing tests.
fix: [not applied -- goal: find_root_cause_only; fix planned separately]
verification: |
  Reproduced deterministically three ways on this checkout: separated streams (one copy in each file),
  merged streams (grep -c == 2), and under a pty (plain line followed by an ANSI 31;1 red line).
  No source file was modified during this investigation.
files_changed: []
