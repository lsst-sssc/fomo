---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 05
subsystem: investigation-spike (no source code shipped) / documentation publication
tags: [docs, sphinx, design-notes, regression-proof, spike-close-out]
requires:
  - 31-01 (SCHEMA-01/02 evidence)
  - 31-02 (SCHEMA-01/02 recommendation checkpoint)
  - 31-03 (SCHEMA-03 classical-schedule-file findings)
  - 31-04 (SCHED-07 evidence and recommendation)
provides:
  - docs/design/run_identity_and_unattended_invocation_spike.rst — durable, published
    summary of both Phase 31 verdicts, wired into the Sphinx toctree
  - Proof that Phase 31 left the source tree and existing test suite untouched
affects:
  - Phase 32 (Adapter Consolidation) — reads the published identity-scheme verdict
  - Phase 34 (Unattended Scheduling & Discovery) — reads the published scheduling verdict
actuals:
  tokens: 31000
  tasks: 2
  commits: 1
tech-stack:
  added: []
  patterns:
    - "Durable docs/design/ page carrying two decision tracks in one document, following
      canonical_record_spike.rst / uncertain_scheduling_spike.rst precedent"
key-files:
  created:
    - docs/design/run_identity_and_unattended_invocation_spike.rst
  modified:
    - docs/design/design.rst
key-decisions:
  - "Phase 31's two verdicts (schema/identity SCHEMA-01..03, scheduling SCHED-07) are now
    readable outside .planning/, on one published page, closing roadmap Success Criterion
    5's first half."
  - "Task 2 produced no tracked-file changes (it deletes a git-excluded disposable DB copy
    and runs verification only), so no commit was made for it — the atomic close-out
    invariant is satisfied by Task 1's production commit followed by this SUMMARY commit."
requirements-completed: [SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHED-07]
coverage:
  - truth: "A page under docs/design/ carries both this phase's verdicts - the run-identity scheme and the scheduling mechanism - so the decisions are readable outside .planning/"
    human_judgment: false
    rationale: "Deterministically verifiable: file exists, has Background/Decisions/Future scope sections, two list-tables, and a pointer to 31-DECISION.md."
  - truth: "The new page is wired into the Sphinx toctree in docs/design/design.rst"
    human_judgment: false
    rationale: "git diff --numstat confirms exactly one line added, none removed; grep confirms the entry name is present."
  - truth: "The Sphinx documentation build succeeds with the new page included"
    human_judgment: false
    rationale: "sphinx-build -M html ... printed 'build succeeded, 13 warnings' (warnings pre-existing, not from this plan's page)."
  - truth: "The existing test suite is unchanged, because no file under solsys_code/ or src/ was modified by any plan in this phase"
    human_judgment: false
    rationale: "git status --porcelain -- solsys_code src printed nothing; targeted 6-module regression run (143 tests) ended OK."
  - truth: "No file under the git-excluded tmp/ directory was committed, and the disposable database copy is removed"
    human_judgment: false
    rationale: "git ls-files tmp/ printed nothing before and after; tmp/31-spike-db-copy.sqlite3 removed and confirmed absent."
  - truth: "Neither committed artifact - the decision document nor the design page - contains a credential value, a long hex secret or an email address"
    human_judgment: true
    rationale: "The regex gate (assignment-shaped credential token, 32+ char hex, email pattern) is a deterministic backstop, but the plan's own prohibition is explicitly labeled verification: judgment — a regex cannot prove the absence of every possible secret shape, only the absence of the shapes it checks for. Automated gate passed; final confidence rests on this being read by a human before the phase seals, per the plan's own prohibition status."
  - truth: "Task 2's human-check: the published page reads as actionable to someone outside the phase, and every unconfirmed scope is visible on the page itself, not only in the evidence document"
    human_judgment: true
    rationale: "This is the human-check item itself (see Human-Check Items section below) — it did not halt execution per workflow.human_verify_mode=end-of-phase, and is recorded verbatim for the orchestrator's end-of-phase verification pass rather than answered here."
duration: ~25min
completed: 2026-09-02
status: complete
---

# Phase 31 Plan 05: Documentation Publication & Regression Proof Summary

Published run-identity and scheduling verdicts to `docs/design/`, Sphinx build green
(13 pre-existing warnings, none new), targeted 6-module regression suite green (143
tests, OK), source tree and test suite provably untouched, disposable database copy
removed, and both committed artifacts pass the credential/contact-detail scan.

## Performance

- Duration: ~25 minutes
- Tasks completed: 2/2
- Files changed: 2 tracked (`docs/design/run_identity_and_unattended_invocation_spike.rst`
  created, `docs/design/design.rst` modified, one line); 1 git-excluded disposable file
  deleted (`tmp/31-spike-db-copy.sqlite3`, never tracked)

## Accomplishments

- Wrote `docs/design/run_identity_and_unattended_invocation_spike.rst` (Task 1),
  following the `canonical_record_spike.rst` structural template and
  `uncertain_scheduling_spike.rst`'s compact list-table shape: title, opening paragraph
  naming the 2026-09-01 through 2026-09-02 investigation window and what was not built,
  a Background section listing all four settled questions, two Decisions list-tables
  (identity track: schema shape / identity field-and-constraint / per-ingest-path value
  / classical sufficiency verdict; scheduling track: mechanism / overlap prevention /
  credential handling / missed-invocation visibility), a bold-dated correction note
  (the `_notify_staff()`/`mail_admins()` mischaracterization), and a Future scope
  section naming every still-open item for Phase 32 and Phase 34, with every unconfirmed
  scope from `31-DECISION.md` (container image, AWS deployment target, heartbeat policy
  question) carried forward with matching wording rather than dropped or softened.
- Added exactly one line to `docs/design/design.rst`'s toctree, immediately after the
  canonical-record entry.
- Proved (Task 2) that Phase 31 changed no source behaviour: Sphinx build succeeded,
  the targeted 6-module regression suite (`test_campaign_models`,
  `test_load_telescope_runs`, `test_sync_lco_observation_calendar`,
  `test_sync_gemini_observation_calendar`, `test_campaign_reconciler`,
  `test_canonical_record_migration` — none of which imports `ephem_utils`/`views`,
  re-verified before running) ended `OK` (143 tests), `git status --porcelain --
  solsys_code src` printed nothing, no file under `tmp/` is tracked, the disposable
  database copy was deleted, and the credential/email/hex scan over both `31-DECISION.md`
  and the new design page found nothing.
- Confirmed `31-DECISION.md`'s structural integrity before the phase seals: exactly four
  verdict subsections (`### SCHEMA-01`, `### SCHEMA-02`, `### SCHEMA-03`, `### SCHED-07`),
  all below `## Recommendation`; eight `#### ` evidence subsections, all between
  `## Findings` and `## Recommendation`; both track headings (`### Schema/identity track`,
  `### Scheduling track`) present under `## Findings`.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write the durable docs/design page and wire it into the toctree | `39ff60c` | `docs/design/run_identity_and_unattended_invocation_spike.rst`, `docs/design/design.rst` |
| 2 | Prove no source behaviour changed, build the docs, remove the disposable DB copy | *(no commit — no tracked file changed; see Deviations)* | `tmp/31-spike-db-copy.sqlite3` (git-excluded, deleted, never tracked) |

## Files Created/Modified

- `docs/design/run_identity_and_unattended_invocation_spike.rst` (new) — durable summary
  of both Phase 31 verdicts
- `docs/design/design.rst` (modified, one line) — new toctree entry
- `tmp/31-spike-db-copy.sqlite3` (deleted, git-excluded, never tracked) — the disposable
  dev-DB copy created by plan 31-01, no longer needed once its quoted evidence is in
  `31-DECISION.md`

## Decisions Made

- No new decisions were made in this plan — it publishes and proves decisions already
  recorded by plans 31-01 through 31-04. See `31-DECISION.md`'s `## Recommendation`
  section for the substantive verdicts (schema shape: nullable `campaign` FK;
  `source_identifier` write-time identity field; classical tolerance match insufficient
  on its own; SCHED-07: cron + flock inside the container).

## Deviations from Plan

**Task 2 produced no git commit.** The plan's `<files>` for Task 2 is
`tmp/31-spike-db-copy.sqlite3`, which is git-excluded and was never tracked — Task 2's
action is entirely proof-and-cleanup (run the Sphinx build, run the regression suite,
assert an empty `git status` over `solsys_code`/`src`, assert nothing under `tmp/` is
tracked, delete the disposable copy, run the credential scan). None of this touches a
tracked file. Per the task_commit_protocol, a commit requires staged changes; there were
none to stage. This is not a Rule 1-4 deviation — it is the expected shape of a
proof-only task whose only file reference is a git-excluded artifact — and it does not
violate the atomic-close-out invariant, since Task 1's production commit (`39ff60c`) is
followed directly by this SUMMARY's own commit, with no gap in between.

Otherwise: none - both tasks' automated `<verify>` commands and `<acceptance_criteria>`
passed on the first attempt; no Rule 1-4 deviation was needed for Task 1 either.

## Issues Encountered

None. The pre-commit hook's own Sphinx-build check (triggered by Task 1's commit) passed
independently of Task 2's explicit `sphinx-build` invocation, giving two confirmations of
the same "build succeeded" result from two separate invocations.

## Human-Check Items for End-of-Phase

Per `workflow.human_verify_mode = end-of-phase`, none of the three `<human-check>` items
below halted execution anywhere in Phase 31. All three are recorded here verbatim,
consolidated in one place, for the orchestrator's end-of-phase verification pass. Plan
31-04's two items are carried forward from `31-04-SUMMARY.md` since no end-of-phase
verifier has surfaced them yet; this plan's own item is new.

**From plan 31-04, Task 1 human-check:**

> Confirm the shell this phase ran its host probes in is the same machine D-01 refers to -
> the local Rocky 9 or WSL2 install FOMO actually runs on - and not a look-alike sandbox.
> `31-DECISION.md`'s interim-host evidence section quotes the kernel string, the init system
> version and the crontab it found; the crontab entries reference a FOMO checkout path. If
> that path and those entries are yours, the interim-host findings stand. If they are not,
> every finding in that section is about an unrelated machine and the section must be
> re-run on the real host before Phase 34 builds against it.

**From plan 31-04, Task 2 human-check:**

> Confirm whether a FOMO container image exists anywhere yet, and if so where its build
> definition lives, since this repository contains none. `31-DECISION.md`'s container-scope
> section records what was found locally and which branch was taken. If a real image or
> build definition exists somewhere I could not see, the container row of the scope table
> should be re-checked against it before Phase 34 builds the scheduler entry point; if it
> does not exist yet, confirm that whoever writes that definition inherits the requirement
> to install a cron daemon and the lock utility inside it.

**From this plan (31-05), Task 2 human-check:**

> Read `docs/design/run_identity_and_unattended_invocation_spike.rst` as a reader who was not
> part of this phase. Both Decisions tables should be actionable without opening
> `31-DECISION.md`, and every unconfirmed scope - the container image, the AWS target, and the
> classical schedule-file question if no real file was obtained - should be visible on the page
> rather than only in the evidence document. If a verdict reads as more settled on the page than
> the evidence behind it supports, say so and the wording will be corrected before the phase
> seals.

## Next Phase Readiness

This is the final plan of Phase 31. Phase 31 as a whole is ready for the orchestrator's
end-of-phase verification: all five roadmap success criteria have supporting evidence
(the schema/identity track's evidence and recommendation in `31-DECISION.md`; the
scheduling track's evidence and recommendation likewise; both now published on one
`docs/design/` page reachable from the Sphinx toctree; the test suite and source tree
proven unchanged; no disposable artifact committed). All four requirements this phase
carries (SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHED-07) are the last sibling plan to touch
each, so the shared-ID completion gate should now clear all four (confirmed below under
Self-Check).

Two loose ends are already on record for whichever phase reads Phase 31's output next,
named explicitly here so they are not silently assumed closed:

- **Phase 32-facing:** SCHEMA-03's verdict states the classical adapter's `source_identifier`
  mirror inherits the exact same two-proposals-same-night blind spot as the underlying
  tolerance match — populating `source_identifier` for the classical path does not close
  this gap, it merely carries it forward under a new field. Phase 32 inherits two
  concrete, separable follow-on items (recognise a leading proposal-code token in the
  parser; decide whether to fold it into the identity key once reliably extractable), not
  a closed question.
- **Phase 34-facing:** SCHED-07's verdict names a concrete missing dependency, not just a
  caveat — no FOMO container build definition exists anywhere in this repository, so
  whoever writes one inherits the requirement to install both `flock` and an HTTP client
  inside it. The container and AWS deployment scopes remain unconfirmed by construction;
  Phase 34 must not assume either is settled on the strength of the interim-host evidence.

Three human-check items (above) remain outstanding and are the only items blocking full
closure of this phase's evidence trail — none is a code or documentation defect, all
three are confirmations only the operator can give.

## Self-Check: PASSED

- FOUND: `docs/design/run_identity_and_unattended_invocation_spike.rst`
- FOUND: `docs/design/design.rst` contains `run_identity_and_unattended_invocation_spike`
  (one line added, per `git diff --numstat`)
- FOUND commit `39ff60c`
- CONFIRMED: `sphinx-build` printed `build succeeded, 13 warnings`
- CONFIRMED: targeted regression run (6 modules, 143 tests) ended `OK`
- CONFIRMED: `git status --porcelain -- solsys_code src` empty
- CONFIRMED: `git ls-files tmp/` empty; `tmp/31-spike-db-copy.sqlite3` removed
- CONFIRMED: credential/email/hex scan over `31-DECISION.md` and the new design page
  found nothing
- CONFIRMED: `31-DECISION.md` structural gate (4 verdict subsections below
  `## Recommendation`, 8 evidence subsections between `## Findings` and
  `## Recommendation`, both track headings present) passed
</content>
