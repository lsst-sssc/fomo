---
status: complete
phase: 31-foundation-spikes-run-identity-unattended-invocation
source: [31-VERIFICATION.md]
started: 2026-09-02T19:00:00Z
updated: 2026-09-02T19:25:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Confirm the probed host is the operator's real FOMO machine
expected: The redacted crontab entries and checkout paths are yours, on the real Rocky 9/WSL2
  FOMO host — confirming the interim-host findings as evidence about the real target host.
result: pass

### 2. Confirm whether a FOMO container image or build definition exists outside this repository
expected: Either none exists anywhere (in which case whoever writes it inherits the requirement
  to install a cron daemon and the lock utility inside it), or one exists (in which case
  31-DECISION.md's container scope row must be re-checked against it before Phase 34 builds the
  scheduler entry point). Independently re-confirmed: this repository tracks no container build
  file of any kind.
result: pass

### 3. Confirm the published design page reads as actionable without overstating confidence
expected: Reading docs/design/run_identity_and_unattended_invocation_spike.rst cold, both
  decision tables are actionable without opening 31-DECISION.md, and every unconfirmed scope
  (container image, AWS target, classical-file question) is visible on the page itself, not
  buried in Future scope. Verifier's own read: the page is, if anything, more hedged than the
  evidence requires.
result: issue
reported: "This phase should not target 'LCO and Gemini sync commands' but rather 'LCO and SOAR' - we have no visibility into any of the Gemini queues through the existing GEMFacility class"
severity: major

### 4. Accept the six judgment-tier prohibitions
expected: Each of the six `verification: judgment` prohibitions carried by this phase's plans
  holds, per the Prohibitions table in 31-VERIFICATION.md — in particular the two
  evidence-integrity ones (no finding tagged as evidence-backed unless its probe actually ran;
  no committed artifact carries a credential value). W-3 (the LCO identity's hardcoded-but-since-
  confirmed-true literal) is the one worth a second opinion.
result: pass

## Summary

total: 4
passed: 3
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-31-3
  truth: "Every ingest path the SCHEMA-02 per-adapter table names (load_telescope_runs, sync_lco_observation_calendar, sync_gemini_observation_calendar) has real, current functional visibility into the facility it claims to sync"
  status: failed
  reason: "User reported: This phase should not target 'LCO and Gemini sync commands' but rather 'LCO and SOAR' - we have no visibility into any of the Gemini queues through the existing GEMFacility class"
  severity: major
  test: 3
  artifacts: []
  missing: []
