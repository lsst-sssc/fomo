---
phase: 31-foundation-spikes-run-identity-unattended-invocation
plan: 04
subsystem: investigation-spike (no source code shipped)
tags: [scheduling, cron, flock, unattended-invocation, credentials, dead-mans-switch, spike]
requires:
  - 31-03 (SCHEMA-03 classical-schedule-file findings, prior section of 31-DECISION.md)
provides:
  - SCHED-07 evidence sections and recommendation in 31-DECISION.md
  - Verified real-host mechanism verdict (cron + flock) for Phase 34's scheduler entry point
affects:
  - Phase 34 (Unattended Scheduling & Discovery) — builds its scheduler entry point directly
    against this plan's invocation shape, credential-handling, and visibility recommendation
actuals:
  tokens: 5547
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - "cron + flock (util-linux), one lock file per management command, -n fail-fast flag"
    - "Environment-variable credentials extending the existing FINK_CREDENTIAL_* naming convention"
    - "Two-layer missed-invocation visibility: in-command _notify_staff()-style email + external heartbeat dead-man's-switch"
key-files:
  created:
    - tmp/31-host-probe.txt (git-excluded)
    - tmp/31-container-probe.txt (git-excluded)
  modified:
    - .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md
key-decisions:
  - "SCHED-07 mechanism confirmed: cron + flock inside the FOMO container, identical on the interim host and the eventual AWS deployment. No blocker found to this choice; flock is present on the interim host and the existing crontab already runs FOMO management commands today with zero overlap guards (0/3), which is the strongest evidence for adding flock rather than switching to a task queue or a K8s-native CronJob."
requirements-completed: [SCHED-07]
coverage:
  - truth: "31-DECISION.md states the unattended-invocation mechanism chosen for FOMO, and states it against verified facts about the real target host rather than a recommendation"
    human_judgment: false
    rationale: "Deterministically verifiable by grepping 31-DECISION.md for the '### SCHED-07 - unattended invocation mechanism' heading and its cited flock/cron facts."
  - truth: "31-DECISION.md quotes verbatim, dated terminal output proving whether flock is present and whether outbound HTTPS to a heartbeat endpoint succeeds, run during this phase"
    human_judgment: false
    rationale: "tmp/31-host-probe.txt was captured live during this plan's execution and is quoted verbatim in the doc; deterministically checkable."
  - truth: "31-DECISION.md states overlap prevention concretely: exact invocation shape, lock file path convention, fail-fast flag"
    human_judgment: false
    rationale: "Fully specified invocation shape is present in the document text; deterministically checkable via grep."
  - truth: "31-DECISION.md states how credentials reach an unattended run and that no credential is passed as a management-command argument"
    human_judgment: false
    rationale: "Statement is present in the document text; deterministically checkable."
  - truth: "31-DECISION.md states how a missed invocation becomes visible, as two independent layers"
    human_judgment: false
    rationale: "Both layers (in-command _notify_staff() extension, external heartbeat) are named explicitly; deterministically checkable."
  - truth: "31-DECISION.md labels the deployment question separately for each scope, and does not mark a scope confirmed on the strength of a probe run against a different one"
    human_judgment: true
    rationale: "The document's labeling itself is deterministic (three-scope table present), but whether the interim-host findings genuinely apply to the operator's real D-01 machine, and whether no real FOMO container/build definition exists anywhere the executor could not see, both rest entirely on the two <human-check> items below — human confirmation is the only evidence that can close this."
  - truth: "31-DECISION.md records whether existing crontab entries already run FOMO management commands without an overlap guard, as evidence about the live risk"
    verification: backstop
    human_judgment: false
    rationale: "0/3 unguarded ratio is quoted verbatim from a live probe transcript in the document; deterministically checkable, though it inherits the same host-identity caveat as the row above."
duration: ~20min
completed: 2026-09-02
status: complete
---

# Phase 31 Plan 04: Unattended Invocation Mechanism Spike Summary

Verified cron + flock against the real interim host (flock present, 0/3 existing FOMO cron
entries guarded, heartbeat egress reachable via HTTP 301), found no FOMO container definition
exists anywhere in this repository, and recorded the full SCHED-07 recommendation (invocation
shape, credential handling, two-layer missed-invocation visibility) for Phase 34 to build
against without re-deciding anything.

## Performance

- Duration: ~20 minutes
- Tasks completed: 3/3
- Files changed: 1 tracked (`31-DECISION.md`), 2 git-excluded probe transcripts created

## Accomplishments

- Ran real, dated probes against the interim host (Task 1): `flock` present (util-linux
  2.37.4), the real crontab's 3 FOMO management-command entries carry **0** overlap guards,
  and outbound HTTPS to `hc-ping.com` succeeded (HTTP 301).
- Confirmed this repository tracks no container build file at all, and — since no FOMO image
  exists locally either — ran the same two checks inside a pulled `python:3.11-slim` stand-in
  image instead (Task 2), labeling the result honestly as a stand-in (flock present, curl
  absent) rather than evidence about FOMO's own image. Container and AWS scopes both remain
  explicitly unconfirmed.
- Wrote the complete SCHED-07 recommendation (Task 3): mechanism, exact cron/flock invocation
  shape, adjacency answer (skip, not queue/block), credential-handling constraints (including
  a newly surfaced `local_settings.py` leak-hazard finding), the two-layer missed-invocation
  visibility design (correcting CONTEXT.md's `_notify_staff()` characterization), the SCHED-10
  exception-sanitisation obligation, and the carried-forward same-minute ordering question.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Verify the interim host for real | `14d4a8e` | `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`, `tmp/31-host-probe.txt` (git-excluded) |
| 2 | Reach the container scope, or record honestly it could not be reached | `c3093e2` | `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`, `tmp/31-container-probe.txt` (git-excluded) |
| 3 | Write the SCHED-07 recommendation | `4745ad7` | `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` |

## Files Created/Modified

- `tmp/31-host-probe.txt` (git-excluded) — live transcript: uname, systemd version, flock
  version, redacted crontab, heartbeat egress status code
- `tmp/31-container-probe.txt` (git-excluded) — live transcript: tracked container-build-file
  search (`NONE`), local Docker image inventory, stand-in-image flock/heartbeat check
- `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` —
  three new sections: `### Scheduling track (SCHED-07)`, `#### SCHED-07 evidence - interim
  host verification`, `#### SCHED-07 evidence - container image and AWS scope`, and
  `### SCHED-07 - unattended invocation mechanism` (the final section of the document)

## Decisions Made

- **SCHED-07 mechanism confirmed as cron + flock inside the FOMO container**, identical on
  the interim host and the eventual AWS deployment — no blocker found to D-02's choice.
- **Overlap prevention:** `flock -n <per-command-lock-path> <interpreter> <manage.py>
  <command>`, one lock file per command name, never a shared lock. A tick arriving while the
  lock is held is **skipped**, not queued or blocked.
- **Credential handling:** environment variables, `<SERVICE>_CREDENTIAL_<FIELD>` naming
  convention extending `FINK_CREDENTIAL_*`; never a management-command argument or embedded
  in a cron line.
- **Missed-invocation visibility:** two independent layers — an extended `_notify_staff()`
  in-command email (needs a non-request-based link-building adaptation) and an external
  heartbeat dead-man's-switch (heartbeat fires even on a zero-work run).
- **Container image and AWS scopes remain unconfirmed** — no FOMO container build definition
  exists anywhere in this repository; whoever writes one inherits the requirement to install
  cron and flock inside it, named explicitly as a Phase 34 dependency.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' automated `<verify>` commands and
`<acceptance_criteria>` passed on the first attempt; no Rule 1-4 deviation was needed.

## Issues Encountered

None. `docker pull python:3.11-slim` required live network egress, which task 1 had already
confirmed works from this host; the pull succeeded without incident and the container was
removed automatically via `docker run --rm`.

## Human-Check Items for End-of-Phase

Per `workflow.human_verify_mode = end-of-phase`, both `<human-check>` blocks below did not
halt execution; they are recorded here verbatim for the orchestrator's end-of-phase
verification pass.

**Task 1 human-check:**

> Confirm the shell this phase ran its host probes in is the same machine D-01 refers to -
> the local Rocky 9 or WSL2 install FOMO actually runs on - and not a look-alike sandbox.
> `31-DECISION.md`'s interim-host evidence section quotes the kernel string, the init system
> version and the crontab it found; the crontab entries reference a FOMO checkout path. If
> that path and those entries are yours, the interim-host findings stand. If they are not,
> every finding in that section is about an unrelated machine and the section must be
> re-run on the real host before Phase 34 builds against it.

**Task 2 human-check:**

> Confirm whether a FOMO container image exists anywhere yet, and if so where its build
> definition lives, since this repository contains none. `31-DECISION.md`'s container-scope
> section records what was found locally and which branch was taken. If a real image or
> build definition exists somewhere I could not see, the container row of the scope table
> should be re-checked against it before Phase 34 builds the scheduler entry point; if it
> does not exist yet, confirm that whoever writes that definition inherits the requirement
> to install a cron daemon and the lock utility inside it.

## Next Phase Readiness

Plan 31-05 (docs publication + regression proof) needs: this plan's SCHED-07 verdict is
internally consistent — it explicitly names the Phase 34 dependency for the missing container
definition (install cron + flock inside whatever container gets written) rather than assuming
one already exists, and it does not mark the container or AWS scopes confirmed on the strength
of the interim-host probes. Both human-check items above should be surfaced verbatim in
31-05's end-of-phase verification pass; if the operator's answers to either raise a
contradiction (this is not the real host, or a real FOMO container/build definition already
exists elsewhere), 31-05 should flag that as a correction needed before the durable
`docs/design/` page is published, rather than publishing the current verdict silently.

## Self-Check: PASSED

- FOUND: `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
- FOUND: `tmp/31-host-probe.txt`
- FOUND: `tmp/31-container-probe.txt`
- FOUND commit `14d4a8e`
- FOUND commit `c3093e2`
- FOUND commit `4745ad7`
