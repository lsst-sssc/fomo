---
phase: 13-eso-feasibility-spike
plan: 01
subsystem: integration
tags: [eso, p2api, tom_eso, phase2, feasibility-spike, investigation]

# Dependency graph
requires: []
provides:
  - "Confirmed real ESO Phase 2 production credentials (Paranal/VLT) are obtainable and usable (ESO-01)"
  - "Live, redacted verbatim getOB()/getNightExecutions() response shapes for real OB status/execution data (ESO-02)"
  - "Confirmed viable headless credential-sourcing path via env-var-backed ESOAPI(...), bypassing ESOProfile/session decryption (ESO-03)"
  - "La Silla stretch finding: production_lasilla environment string is unsupported for this account (D-06, documented not blocking)"
  - "13-DECISION.md skeleton with Findings populated and Recommendation/Future-sync-sketch placeholders for Plan 02"
affects: ["13-02 (writes ESO-04 Recommendation and ESO-05 Future-sync sketch on top of these findings)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Read-only P2 API investigation via a throwaway, git-excluded probe script (D-09) rather than committed exploration code"
    - "Headless credential-sourcing via FACILITIES-style env-var-backed ESOAPI(...) construction, confirmed as an alternative to session-bound ESOProfile decryption"

key-files:
  created:
    - ".planning/phases/13-eso-feasibility-spike/13-DECISION.md"
    - "eso_p2_probe.py (repo root, uncommitted/git-excluded per D-09 — not a deliverable)"
  modified: []

key-decisions:
  - "Paranal (VLT) production credentials confirmed obtainable and usable via ESOAPI(environment='production', ...) constructed from env-var-supplied credentials"
  - "La Silla (production_lasilla) fails via tom_eso's ESOAPI wrapper (P1Error, 'environment not supported') because p1api's API_URL has no La Silla entry — but p2api's own API_URL does support production_lasilla, and the operator separately confirmed working La Silla P2 web-portal access with the same credentials, so this is root-caused to ESOAPI's wrapper (not genuine account/API inaccessibility); a direct p2api.ApiConnection bypass is the untested-but-promising next step, per D-06"
  - "Real getOB() response captured for a never-executed draft OB (obStatus='P', empty getOBExecutions()); real getNightExecutions() response captured for an executed-but-failed OB (obStatus='M', grade='X') on the same instrument/night"
  - "Headless credential-sourcing confirmed viable: no ESOProfile/session decryption was needed anywhere in this path, meaning a future FACILITIES['ESO'] settings entry is a workable design for a headless management command"

patterns-established:
  - "D-04 redaction convention applied inline: verbatim API response blocks pasted as-is (their shape is the evidence) with a 'credential-adjacent fields redacted per D-04' note immediately following each block, even when nothing required redaction"

requirements-completed: [ESO-01, ESO-02, ESO-03]

coverage:
  - id: D1
    description: "ESO-01: Paranal production credentials confirmed obtainable/usable, with live connection evidence and a documented La Silla (production_lasilla) failure as a non-blocking stretch finding"
    requirement: "ESO-01"
    verification:
      - kind: manual_procedural
        ref: "Operator ran eso_p2_probe.py against real Paranal production credentials and reported redacted evidence, recorded verbatim in 13-DECISION.md ### ESO-01"
        status: pass
    human_judgment: true
    rationale: "Credential obtainability against a live third-party production API can only be confirmed by the operator holding those credentials running the probe themselves; not automatable by the executor."
  - id: D2
    description: "ESO-02: real getOB() and getNightExecutions() response shapes captured verbatim (obStatus 'P' draft-OB case and obStatus 'M'/grade 'X' executed-and-failed case)"
    requirement: "ESO-02"
    verification:
      - kind: manual_procedural
        ref: "Operator-supplied live p2api responses, recorded verbatim (redacted per D-04) in 13-DECISION.md ### ESO-02"
        status: pass
    human_judgment: true
    rationale: "The evidence is a live third-party API response the operator captured and pasted back; there is no automated test that can independently reproduce a live P2 API call against real production credentials."
  - id: D3
    description: "ESO-03: headless credential-sourcing path confirmed viable via env-var-backed ESOAPI(...), with ESOProfile+session decryption confirmed non-viable for headless use"
    requirement: "ESO-03"
    verification:
      - kind: manual_procedural
        ref: "Operator-observed absence of ESOProfile/session dependency in the successful connection path, recorded in 13-DECISION.md ### ESO-03"
        status: pass
    human_judgment: true
    rationale: "This finding is an interpretive conclusion drawn from the operator's live-run observation plus prior PITFALLS.md research, not something an automated check can verify independently."

duration: ~1h10min (session start to Task 3 commit; includes operator wait time between the Task 1 script and the Task 2 checkpoint being satisfied)
completed: 2026-07-01
status: complete
---

# Phase 13 Plan 01: ESO Feasibility Spike — Live Probe Summary

**Live Paranal (VLT) production P2 API investigation confirms credentials work, real OB status/execution shapes are captured, and headless credential-sourcing via env-var-backed ESOAPI is a viable path — while La Silla's production_lasilla environment fails via tom_eso's ESOAPI wrapper specifically (root-caused to a p1api gap, not genuine API inaccessibility).**

## Performance

- **Duration:** ~1h10min (includes operator credential-run wait time for the Task 2 human-action checkpoint)
- **Started:** 2026-07-01T20:54:55Z (per STATE.md session start)
- **Completed:** 2026-07-01T22:04:00Z (Task 3 commit `48b800d`)
- **Tasks:** 3 (1 auto, 1 checkpoint:human-action, 1 auto)
- **Files modified:** 1 committed (`13-DECISION.md`); 1 uncommitted/git-excluded (`eso_p2_probe.py`, per D-09 — not a deliverable)

## Accomplishments
- Authored a throwaway, git-excluded, read-only ESO P2 probe script (`eso_p2_probe.py`) that walks `getRuns()` → `getItems()`/`getOB()` → `getOBExecutions()`/`getNightExecutions()`, contains zero write-style calls, and reads credentials only from environment variables (Task 1).
- Operator ran the probe against real Paranal (VLT) production credentials and returned redacted evidence directly (Task 2 checkpoint satisfied).
- Recorded ESO-01/02/03 Findings into `.planning/phases/13-eso-feasibility-spike/13-DECISION.md`: credentials obtainable/usable, real `getOB()`/`getNightExecutions()` shapes captured (draft-OB `P` case and executed-and-failed `M`/`X` case), and a viable headless credential-sourcing path confirmed (Task 3).
- Documented the La Silla stretch finding (D-06): `production_lasilla` fails at connection construction with a `P1Error` ("environment not supported") — a distinct, non-blocking finding, not a phase blocker.

## Task Commits

1. **Task 1: Author the throwaway ESO P2 probe script and wire credentials** — no commit (intentional; `eso_p2_probe.py` is a throwaway script per D-09, registered in `.git/info/exclude`, never staged or committed).
2. **Task 2: Operator runs the probe against real Paranal production credentials and returns redacted evidence** — checkpoint:human-action, no commit (operator action; evidence supplied directly to the session).
3. **Task 3: Record the redacted evidence into 13-DECISION.md Findings (ESO-01/02/03)** — `48b800d` (docs)

**Plan metadata:** recorded separately after this SUMMARY (STATE.md/ROADMAP.md/REQUIREMENTS.md docs commit).

_Note: Task 1 and Task 2 intentionally have no commit — Task 1's script is explicitly a non-deliverable throwaway per D-09, and Task 2 is an operator action with no repo changes of its own._

## Files Created/Modified
- `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` - Findings for ESO-01 (credential obtainability), ESO-02 (real OB status/execution data shapes), ESO-03 (headless credential-sourcing path); placeholder headers for ESO-04/ESO-05 left for Plan 02.
- `eso_p2_probe.py` (repo root) - throwaway read-only P2 API probe script; registered in `.git/info/exclude`; never staged or committed per D-09.

## Decisions Made
- Paranal (VLT) production credentials are obtainable and usable via `ESOAPI(environment='production', username, password)` constructed from environment-variable-supplied credentials — confirmed by real data returned in ESO-02, not just a connection-success flag.
- La Silla's `production_lasilla` environment string fails at `ESOAPI` connection-construction time with a `P1Error` ("environment not supported") — a genuinely different exception path than the `p2api.p2api.P2Error` the probe otherwise guards against; documented per D-06 as a valid non-blocking finding, not a phase blocker.
- A future headless management command can source ESO credentials the same way this probe did (env-var/`FACILITIES['ESO']`-style, direct `ESOAPI(...)` construction) rather than relying on the per-user, session-bound, Fernet-encrypted `ESOProfile` path, which is confirmed non-viable for headless use (silent `None` decryption outside an active session, per existing PITFALLS.md research).

## Deviations from Plan

None - plan executed exactly as written. Task 2's checkpoint was satisfied by the operator supplying redacted evidence directly to the session (per the plan's designed human-action flow), and Task 3 recorded that evidence exactly per the plan's structure and D-04 redaction discipline. The redaction re-check requested by the orchestrator prompt (verifying no credential-adjacent fields slipped through) was performed by eye before writing — no additional redaction was needed since none of the pasted content (OB IDs, run IDs, target names/coordinates, timestamps, status codes) is credential-adjacent per D-04's scope (usernames, passwords, sensitive program IDs).

## Issues Encountered
None.

## User Setup Required

None for this plan — the operator already supplied the necessary ESO P2 production credentials directly (via environment variables, per D-03) to run the Task 2 probe, and that evidence has been recorded. No further external service configuration is required to close out this plan.

## Next Phase Readiness

Plan 02 can now write the `## Recommendation (ESO-04)` and `## Future-sync sketch (ESO-05)` sections of `13-DECISION.md` grounded in this plan's real evidence: credentials work for Paranal, real OB status/execution shapes are known (both a never-executed draft OB and an executed-and-failed OB), and a concrete non-`ESOProfile` headless credential-sourcing path is confirmed viable. La Silla remains an open gap (connection failure, environment string needs further investigation) that Plan 02's recommendation should account for rather than assume solved.

---
*Phase: 13-eso-feasibility-spike*
*Completed: 2026-07-01*

## Self-Check: PASSED

- FOUND: `.planning/phases/13-eso-feasibility-spike/13-DECISION.md`
- FOUND: `.planning/phases/13-eso-feasibility-spike/13-01-SUMMARY.md`
- FOUND: commit `48b800d`
