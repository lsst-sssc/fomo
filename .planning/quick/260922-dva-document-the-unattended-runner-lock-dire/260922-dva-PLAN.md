---
phase: 260922-dva
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [SCHED-08, SCHED-09]

estimate:
  tokens: 50000
  raw_tokens: 25000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - An operator setting up a fresh host is told, before they create anything, that the shipped `FOMO_LOCK_DIR` default does not survive a reboot, and why.
    - The runbook states that the runner cannot repair the wiped directory itself, so the failure is permanent until a human intervenes.
    - The runbook gives one recommended persistent location and one alternative for operators who want to keep the conventional location, with the trade-off of each stated.
    - The runbook states that `FOMO_STATE_DIR` must be set explicitly alongside `FOMO_LOCK_DIR`, and what is lost if it is not.
    - An operator whose heartbeat went down with no failure email can find the entry that names this failure, run the two diagnostic greps from it, and reach the fix without reading any other section first.
    - The `/var/log/fomo` guidance is unchanged — it is on real disk and was never part of this failure.
    - The runbook page builds with zero docutils-category Sphinx warnings of its own.
    - Nothing under `solsys_code/` or `src/` is modified.
  artifacts:
    - docs/runbooks/telescope_runs_calendar.rst — rewritten step 1 of "Setting it up on a fresh host", plus a new `^^^^`-level troubleshooting entry between "The heartbeat never alerted although the schedule stopped" and "The unused figure says it is not yet known".
  key_links:
    - The documented default path `/var/lock/fomo` <-> `src/fomo/settings.py:427` `os.getenv('FOMO_LOCK_DIR', '/var/lock/fomo')`
    - The documented "set both variables" warning <-> `src/fomo/settings.py:430` `FOMO_STATE_DIR = os.getenv('FOMO_STATE_DIR', FOMO_LOCK_DIR)` (default captured at definition time)
    - The documented permission failure <-> `solsys_code/unattended.py:132-133` `Path(settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR).mkdir(parents=True, exist_ok=True)`
    - The documented "set it before running the preflight" advice <-> `cron_line()` in `solsys_code/management/commands/check_unattended.py`, which substitutes this host's resolved lock path into the printed cron line
    - The new troubleshooting entry <-> the corrected fresh-host step 1 (cross-reference, never restate)
---

<objective>
Correct the fresh-host setup instruction that currently tells operators to create the unattended
runner's lock directory on a tmpfs, and add the troubleshooting entry for the failure that
instruction produced on the real host.

Purpose: `/var/lock` is a symlink to `/run/lock`, which is a tmpfs. The `/var/lock/fomo` directory
the runbook instructs operators to create is therefore erased on every reboot. When this host
rebooted for kernel updates on 2026-09-21 the schedule stopped dead — the crontab's `flock` could
not open its lock file, so every tick died before Python started, and because the runner never
started, its own failure-notification path never ran. No email was sent. The healthcheck going down
was the only signal, and it went unnoticed for about 25 hours and 100 consecutive lost ticks. The
runbook's current step 1 is not merely incomplete; it names the wrong directory as the thing to
create, so an operator who follows it exactly reproduces the outage. That is what this plan fixes.

Output: a rewritten step 1 plus one consistency sentence in step 8, and one new troubleshooting
entry — all in `docs/runbooks/telescope_runs_calendar.rst`. Docs-only: nothing under `solsys_code/`
or `src/` is touched.

**Paired-docs assessment (CLAUDE.md, stated explicitly rather than silently omitted):** no notebook
is in scope. CLAUDE.md's paired-docs map routes `solsys_code/unattended.py`,
`solsys_code/notifications.py`, `run_unattended.py` and `check_unattended.py` to the
"How do I run everything unattended?" section of `docs/runbooks/telescope_runs_calendar.rst` —
**not** to a notebook — and that runbook section is precisely what this plan updates. The rule's
trigger is a change to one of those modules' *behavior*; no module behavior changes here at all.
This change documents deployment configuration, so no notebook is added or regenerated.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@docs/runbooks/telescope_runs_calendar.rst
</context>

<grounded_facts>
These were verified on the live host and in this checkout before planning. Do not re-derive them;
do not contradict them. Source files are cited as evidence only and are **not** to be modified.

- `src/fomo/settings.py:427` — `FOMO_LOCK_DIR = os.getenv('FOMO_LOCK_DIR', '/var/lock/fomo')`.
- `src/fomo/settings.py:430` — `FOMO_STATE_DIR = os.getenv('FOMO_STATE_DIR', FOMO_LOCK_DIR)`. The
  default is captured from `FOMO_LOCK_DIR` **at that point in the file**. `local_settings.py` is
  imported at the end of `settings.py`, so an operator who overrides only `FOMO_LOCK_DIR` there does
  **not** move the state directory with it.
- `solsys_code/unattended.py:132-133` — `command_lock()` does
  `Path(settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR).mkdir(parents=True, exist_ok=True)`. On the
  real host this raised `PermissionError: [Errno 13] Permission denied: '/var/lock/fomo'`, because
  an unprivileged account cannot create a directory inside root-owned `/run/lock`.
- The crontab's guard failed first and differently:
  `flock: cannot open lock file /var/lock/fomo/run_unattended.cron.lock: No such file or directory`.
  Two error strings, one root cause; the `PermissionError` form is what appears once the crontab's
  own path has been repaired but the settings have not.
- Incident facts: host rebooted for kernel updates Mon 2026-09-21 08:47 PDT; last good tick
  `2026-09-21T15:45:24Z`; 100 consecutive ticks lost over about 25 hours; no failure email, because
  the runner never started.
- Fix applied on this host: `FOMO_LOCK_DIR` and `FOMO_STATE_DIR` both set in
  `src/fomo/local_settings.py` to a persistent path under the cron account's own state directory,
  and the crontab's `flock` path changed to match. `python manage.py check_unattended` then passed.
- `cron_line()` in `solsys_code/management/commands/check_unattended.py` substitutes this host's
  **resolved** lock path into the cron line it prints — so setting `FOMO_LOCK_DIR` in
  `local_settings.py` *before* running the preflight makes the printed cron line already correct.
- Sphinx baseline for this page: one pre-existing `ref.doc` warning about the excluded notebook, and
  **zero** docutils-category warnings. The docutils gate below is therefore non-vacuous.
</grounded_facts>

<tasks>

<task type="tracer">
  <name>Task 1: Correct the fresh-host lock and state directory step</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Re-locate the `Setting it up on a fresh host` subsection by its heading text, not by line number, and
rewrite its numbered step 1 in place. Keep the surrounding numbered-list structure intact: step 1
stays step 1, and steps 2 through 9 keep their numbers and their text except for the one sentence in
step 8 named at the end of this action.

The rewritten step 1 must make these points, in the runbook's own voice (measured operator prose,
double-backtick inline literals for paths and setting names, `--` for em dashes, and the `::`
literal-block form the page already uses for paths and commands):

- The runner needs two directories: one for its locks, one for its log. Only the log directory's
  shipped default is safe to use as-is.
- The shipped `FOMO_LOCK_DIR` default is not durable. Say why concretely: `/var/lock` is a symlink to
  `/run/lock`, which is a tmpfs, so anything created there is gone after the next reboot. Do not
  hedge this as a possibility — state it as the property of the path that it is.
- State the consequence that makes it permanent rather than self-healing: after the reboot the
  crontab's `flock` guard cannot open its lock file at all, so the tick dies before Python starts;
  and even with the crontab pointed elsewhere, `command_lock()` tries to create the directory itself
  and an unprivileged cron account cannot create anything inside root-owned `/run/lock`. Nothing in
  the runner recovers from this on its own — it stays broken until an operator changes the
  configuration.
- Give the recommended route first: point `FOMO_LOCK_DIR` at a durable location owned by the cron
  account, with `~/.local/state/fomo` as the worked example, and create it with that account rather
  than with `sudo`. Note that this route needs no `sudo` at all, which is part of why it is the
  recommendation.
- Give the alternative second, for operators who want the conventional location: a
  `/etc/tmpfiles.d/fomo.conf` entry that systemd-tmpfiles recreates at every boot, in the
  `d <path> <mode> <owner> <group> -` form, naming the cron account as owner so the directory is not
  recreated world-writable. State its limitation honestly: it restores the directory, but anything
  the previous boot left in it is still gone, so the state file must not live there.
- Warn explicitly about the second-order trap, and give the reason rather than just the rule:
  `FOMO_STATE_DIR` takes its default from `FOMO_LOCK_DIR` at the point `settings.py` defines it, and
  `local_settings.py` is imported after that point, so overriding only `FOMO_LOCK_DIR` there leaves
  the state directory behind. Both must be set. Name what is lost otherwise: the D-11
  suppression-state file sits on a tmpfs and its contents are dropped at every reboot, so the runner
  re-decides "newly failing" and re-mails after each one.
- Add the operational ordering point: set both in `local_settings.py` before running the preflight in
  step 6, because the cron line the preflight prints carries this host's resolved lock path, so the
  printed line then already matches the configuration.
- Keep `/var/log/fomo` exactly as the log location, still created with `sudo` and handed to the cron
  account. It is on real disk and played no part in this failure. Do not restructure or re-argue it.
- Delete the existing claim that the state directory needs no separate handling of its own. That
  sentence is the specific thing this task contradicts, so it cannot survive anywhere on the page.
- Leave the long WR-17 temp-directory fallback paragraph that follows step 1 completely intact. It is
  about the *state* file becoming unwritable at run time and is a different mechanism from the
  lock-directory problem. The new prose must not blur the two: do not describe the tmpfs wipe as
  something that fallback handles, and do not describe the fallback as a remedy for a missing lock
  directory.

Then make one consistency correction in step 8: that step currently tells an operator to confirm
`FOMO_LOCK_DIR` and `FOMO_LOG_FILE` are still at their defaults before trusting the committed
crontab template as a substitute for the printed cron line. After this change the recommended
`FOMO_LOCK_DIR` is deliberately not the default, so reword that sentence to say the template's
hardcoded lock path will not match a host that followed step 1, and that such a host must edit the
template's lock path to its own or use the printed line. Change nothing else in step 8.

Write no fenced code blocks into the page — it is reStructuredText; use the `::` literal-block form
the surrounding steps already use. Do not add a new `.. _label:` anchor: the page's cross-references
use `:ref:`unattended-operation`` plus a quoted subsection name, and Task 2 follows that same idiom.
  </action>
  <verify>
    <automated>test "$(cd /home/tlister/git/fomo_devel &amp;&amp; sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build" -q 2>&amp;1 | grep 'telescope_runs_calendar.rst' | grep -c 'docutils')" = "0"</automated>
    <automated>cd /home/tlister/git/fomo_devel; R=$(awk '/^Setting it up on a fresh host$/{s=1} /^2\. Put the real/{s=0} s' docs/runbooks/telescope_runs_calendar.rst); for t in '/run/lock' 'tmpfs' '.local/state/fomo' 'tmpfiles.d' '/var/log/fomo'; do printf '%s\n' "$R" | grep -qF -- "$t" || exit 1; done</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(awk '/^Setting it up on a fresh host$/{s=1} /^2\. Put the real/{s=0} s' docs/runbooks/telescope_runs_calendar.rst | grep -cF 'FOMO_LOCK_DIR')" -ge 3</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(awk '/^Setting it up on a fresh host$/{s=1} /^2\. Put the real/{s=0} s' docs/runbooks/telescope_runs_calendar.rst | grep -cF 'FOMO_STATE_DIR')" -ge 4</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(grep -c '^2\. Put the real' docs/runbooks/telescope_runs_calendar.rst)" = "1"</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(grep -c '^      /var/lock/fomo$' docs/runbooks/telescope_runs_calendar.rst)" = "0"</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(grep -c 'no third directory is needed' docs/runbooks/telescope_runs_calendar.rst)" = "0"</automated>
    <automated>cd /home/tlister/git/fomo_devel; grep -qF 'fomo-unattended-state.&lt;uid&gt;.&lt;deployment-tag&gt;.fallback.json' docs/runbooks/telescope_runs_calendar.rst</automated>
    <automated>cd /home/tlister/git/fomo_devel; OUT=$(git status --porcelain solsys_code src) || exit 1; test -z "$OUT"</automated>
  </verify>
  <done>
Step 1 of "Setting it up on a fresh host" names a durable `FOMO_LOCK_DIR` as the requirement, states
that the shipped default lives on the `/run/lock` tmpfs and is erased at every reboot, explains that
neither the crontab's `flock` nor `command_lock()` can recover from that on an unprivileged account,
recommends `~/.local/state/fomo` first and offers the `/etc/tmpfiles.d` route as the alternative with
its limitation stated, and warns that `FOMO_STATE_DIR` must be set explicitly alongside
`FOMO_LOCK_DIR` because `settings.py` captures its default at definition time. `/var/log/fomo`
guidance is unchanged. The old claim that no separate state directory is needed is gone from the
page. The WR-17 fallback paragraph is untouched. Step 8 no longer tells an operator to expect the
lock path to be at its default. The page builds with zero docutils-category warnings of its own and
no file under `solsys_code/` or `src/` is modified.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add the post-reboot heartbeat troubleshooting entry</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Add one new entry to the `Troubleshooting` section, inserted after the existing
`The heartbeat never alerted although the schedule stopped` entry and before
`The unused figure says it is not yet known`, so the three heartbeat entries sit together.

Match the sibling house style exactly: a plain-text heading underlined with `^` characters at least
as long as the heading, then a `**Cause:**` paragraph, then a `**Fix:**` paragraph. Use
`The heartbeat went down after a reboot and no email arrived` as the heading.

The `**Cause:**` paragraph must carry these points:

- The symptom pair is the identifying signature: the healthcheck goes down and *no* failure email
  arrives. State why the absence of email is expected here rather than a second fault — the tick died
  before Python started, so the runner's own failure-notification path never executed. The heartbeat
  is structurally the only signal this failure can produce.
- Tie it to the lock directory: the host rebooted, the directory under `/run/lock` went with the
  tmpfs, and every subsequent tick failed at its very first step.
- Reference the real incident concretely, for calibration on how long this hides: a kernel-update
  reboot wiped the directory and 100 consecutive ticks were lost over about 25 hours before anyone
  noticed.

Then give the two diagnostic commands and what each one proves, in a `::` literal block in the page's
existing style — one grep for the `flock: cannot open` line, and one for the `PermissionError` with
errno 13 on the lock directory path. Explain that these are two forms of the same root cause, not two
separate problems: the first appears while the crontab's own lock path is still pointing at the wiped
directory, and the second is what replaces it once the crontab line alone has been repaired but the
settings have not.

The `**Fix:**` paragraph must say: point `FOMO_LOCK_DIR` and `FOMO_STATE_DIR` at durable storage and
change the crontab's `flock` path to match — all three, because fixing any subset leaves one of the
two failure forms in place. Then re-run `check_unattended` as the cron account, not as root, to
confirm. Close by cross-referencing the corrected setup step using the page's existing idiom — quote
"Setting it up on a fresh host" and point at `:ref:`unattended-operation`` — so an operator lands on
the durable-path guidance rather than having the whole of it restated here.

Write no fenced code blocks. Quote no credential, no heartbeat ping URL, no API key and no real
proposal code — paths, setting names and error strings only.
  </action>
  <verify>
    <automated>test "$(cd /home/tlister/git/fomo_devel &amp;&amp; sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build" -q 2>&amp;1 | grep 'telescope_runs_calendar.rst' | grep -c 'docutils')" = "0"</automated>
    <automated>cd /home/tlister/git/fomo_devel; awk '/^The heartbeat never alerted although the schedule stopped$/{n=NR} /^The heartbeat went down after a reboot and no email arrived$/{h=NR} /^The unused figure says it is not yet known$/{u=NR} END{exit !(n&lt;h &amp;&amp; h&lt;u)}' docs/runbooks/telescope_runs_calendar.rst</automated>
    <automated>cd /home/tlister/git/fomo_devel; R=$(awk '/^The heartbeat went down after a reboot and no email arrived$/{s=1} /^The unused figure says it is not yet known$/{s=0} s' docs/runbooks/telescope_runs_calendar.rst); for t in '**Cause:**' '**Fix:**' 'flock: cannot open' 'PermissionError' 'tmpfs' 'reboot' '100' 'FOMO_STATE_DIR' 'FOMO_LOCK_DIR' 'check_unattended' 'Setting it up on a fresh host' 'unattended-operation'; do printf '%s\n' "$R" | grep -qF -- "$t" || exit 1; done</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(grep -c '^The heartbeat went down after a reboot and no email arrived$' docs/runbooks/telescope_runs_calendar.rst)" = "1"</automated>
    <automated>cd /home/tlister/git/fomo_devel; test "$(awk '/^The heartbeat went down after a reboot and no email arrived$/{s=1} /^The unused figure says it is not yet known$/{s=0} s' docs/runbooks/telescope_runs_calendar.rst | grep -c 'hc-ping.com\|LCO_API_KEY\|SECRET_KEY')" = "0"</automated>
    <automated>cd /home/tlister/git/fomo_devel; OUT=$(git status --porcelain solsys_code src) || exit 1; test -z "$OUT"</automated>
    <automated>cd /home/tlister/git/fomo_devel; pre-commit run --all-files</automated>
  </verify>
  <done>
A single new troubleshooting entry sits between "The heartbeat never alerted although the schedule
stopped" and "The unused figure says it is not yet known", in the siblings' exact Cause/Fix house
style. It names the healthcheck-down-with-no-email signature and explains why no email is expected,
ties it to the tmpfs lock directory and the reboot, cites the real incident's 100 lost ticks over
about 25 hours, gives both the `flock: cannot open` and the `PermissionError` diagnostic greps and
explains that they are two forms of one root cause, fixes all three of `FOMO_LOCK_DIR`,
`FOMO_STATE_DIR` and the crontab's `flock` path, tells the operator to re-run `check_unattended` as
the cron account, and cross-references the corrected setup step. No credential or ping URL appears in
the new text. The page builds with zero docutils-category warnings of its own, `pre-commit
run --all-files` is clean, and no file under `solsys_code/` or `src/` is modified.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| runbook instruction -> host filesystem | An operator follows this page literally when choosing where the runner's lock and suppression-state files live. A wrong recommendation here becomes a real deployment defect. |
| other local accounts -> the recommended directory | On a shared host, any location this page names is reachable by other local accounts unless ownership and mode are specified. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-260922-dva-01 | Denial of Service | unattended runner availability (`command_lock()`, crontab `flock` guard) | high | mitigate | This is the incident being documented. Task 1 replaces the instruction that puts the lock directory on a tmpfs with a durable-path recommendation, and Task 2 makes the resulting silent stall diagnosable from the one signal it produces. No code change is in scope, so the documentation is the whole of the mitigation. |
| T-260922-dva-02 | Tampering | the directory this page recommends operators create | medium | mitigate | Task 1 recommends a path under the cron account's own home, created by that account and needing no `sudo`, so it is not shared. Where the `/etc/tmpfiles.d` alternative is offered, the entry form explicitly names the cron account as owner so the directory is never recreated world-writable. |
| T-260922-dva-03 | Information Disclosure | example text in the new prose | low | mitigate | Both tasks are constrained to paths, setting names and error strings. Task 2 carries an explicit gate asserting no ping URL or API-key setting name appears in the new entry, preserving SCHED-10. |
| T-260922-dva-04 | Repudiation | suppression-state durability | medium | mitigate | Task 1 states that leaving `FOMO_STATE_DIR` on the tmpfs drops the D-11 suppression record at every reboot, causing the runner to re-decide "newly failing" and re-mail — so the alert history stops reflecting what actually happened. |
| T-260922-dva-SC | Tampering | npm/pip/cargo installs | low | accept | No package-manager install task exists in this plan. It is documentation-only and adds no dependency, so the package-legitimacy gate has nothing to audit. |

**ASVS level 1, block on high.** T-260922-dva-01 is the only high-severity entry and is fully
mitigated by the two tasks above.
</threat_model>

<planner_contributions_assessment>
Applied honestly rather than assumed to self-skip:

- **api-coverage** — detector run over this task's real scope: `{"detected":false,"signals":[]}`.
  No external API is integrated. Checkpoint skipped, no `COVERAGE.md` produced.
- **assumption-delta** — this is a quick task with no ROADMAP phase section, so the scan resolves to
  `skipped` (`phase_unresolved`) rather than a real negative. Per the contribution's own rule a skip
  is not a verdict: the checkpoint is not raised. Substantively there is no singular-to-plural,
  required-to-optional or derived-to-chosen transition here — the page documents an existing setting
  more accurately and introduces no new modeling decision.
- **schema-gate** — no file in scope matches any ORM schema pattern (no Payload/Prisma/Drizzle/
  Supabase/TypeORM paths; no Django migration either). Skipped silently, no `[BLOCKING]` push task.
- **security** — applied: `<threat_model>` above, ASVS level 1, block on high.
</planner_contributions_assessment>

<out_of_scope>
Identified during planning, deliberately not changed here — recorded so it is a decision rather than
an omission:

- `deploy/cron/fomo.crontab.example` hardcodes `/var/lock/fomo` at lines 9, 67 and 71. It has the
  same defect as the runbook step this plan corrects. It is outside the stated scope of this task
  (which names one documentation file) and outside the content requirements given. Task 1 does make
  the runbook honest about it, by rewording step 8 so an operator is told the template's lock path
  will not match a host that followed the corrected step 1. **Recommend a follow-up quick task** to
  fix the template itself.
- `solsys_code/unattended.py`, `src/fomo/settings.py` and `src/fomo/local_settings.py` are cited as
  evidence only and are not modified. Changing the shipped `FOMO_LOCK_DIR` default is a behavior
  change, not a documentation fix, and would need its own discussion.
- No notebook is touched. See the paired-docs assessment in `<objective>` for the reasoning.
</out_of_scope>

<verification>
- `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D "exclude_patterns=notebooks/*,_build"`
  reports zero docutils-category warnings naming `telescope_runs_calendar.rst`. The page's one
  pre-existing `ref.doc` warning about the excluded notebook is the known baseline and is out of
  scope — the gate greps for `docutils` specifically so that warning cannot mask a new one.
- `pre-commit run --all-files` is clean (this is what actually gates the commit, and it runs the same
  Sphinx build plus ruff).
- `git status --porcelain solsys_code src` is empty after both tasks — the docs-only constraint is
  asserted, not assumed.
</verification>

<success_criteria>
An operator setting up a fresh host from this page cannot reproduce the 2026-09-21 outage by
following it, and an operator already in the outage can identify it from the healthcheck alone and
fix it without reading anything outside the troubleshooting entry and the step it links to.
</success_criteria>

<output>
Create `.planning/quick/260922-dva-document-the-unattended-runner-lock-dire/260922-dva-SUMMARY.md` when done
</output>
</content>
</invoke>
