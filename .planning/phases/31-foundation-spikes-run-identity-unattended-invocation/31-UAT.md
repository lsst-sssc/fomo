---
status: diagnosed
phase: 31-foundation-spikes-run-identity-unattended-invocation
source: [31-VERIFICATION.md]
started: 2026-09-02T19:00:00Z
updated: 2026-09-02T19:35:00Z
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
  root_cause: |
    Confirmed correct, multi-cause. (1) GEMFacility (tom_observations, upstream library) is
    submit-only: get_observation_status()/get_observation_url()/data_products() are hardcoded
    stubs returning empty state, never real Gemini queue data. (2) sync_gemini_observation_calendar.py
    is live, working code, but it is not a queue sync -- it queries only FOMO's own local
    ObservationRecord rows (facility='GEM') and replays FOMO's own prior ToO submissions onto the
    calendar; it makes zero outbound HTTP calls and never imports GEMFacility. (3) SOAR already has
    real, API-backed queue visibility today, folded inside sync_lco_observation_calendar.py
    (SOARFacility(LCOFacility), a real OCS-API read path) -- there is no separate sync_soar_*
    command needed. (4) CampaignRun.Source (models.py:132-138) has GEMINI_QUEUE but no SOAR_QUEUE
    slot. (5) This was previously recorded and lost: v1.5-REQUIREMENTS.md's Out of Scope section
    already flagged GEMFacility.get_observation_status() as a stub, but that limitation never
    propagated into PROJECT.md, the v2.2 Source vocabulary, or v2.3's REQUIREMENTS/ROADMAP.
    STATE.md's own core-value line already says "LCO/SOAR", while REQUIREMENTS.md ADAPT-03 and
    ROADMAP.md Phase 32 still say Gemini. Phase 31's own source_identifier mechanism, nullable-FK
    decision, and SCHED-07 scheduling track are all unaffected -- this touches only the facility
    inventory. Consequence beyond Phase 31: Phase 32's ADAPT-03 should target SOAR (a branch inside
    the existing LCO adapter, not a third command) and Phase 33's outcome propagation is
    structurally impossible for Gemini (get_observation_status() never leaves its stub state) but
    fully possible for SOAR.
  artifacts:
    - path: ".planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md"
      issue: "Lines 246, 268, 710, 759, 773 frame sync_gemini_observation_calendar.py as one of three facilities with real queue visibility; the cited source line is real but the framing (queue visibility) is not"
    - path: "docs/design/run_identity_and_unattended_invocation_spike.rst"
      issue: "Lines 5, 23, 95-96, 112, 188 carry the same Gemini-as-third-facility framing into the published summary"
  missing:
    - "31-DECISION.md and the published design page need the Gemini/SOAR distinction: source_identifier mechanism is sound and the cited line is real, but Gemini is a submission-echo path with no facility read-back"
    - "Note (for Phase 32/33 planning, not this phase's own fix): CampaignRun.Source needs a SOAR_QUEUE value; ADAPT-03 should re-target SOAR; Phase 33's outcome-propagation feasibility for Gemini needs an explicit caveat"
  debug_session: ".planning/debug/gemini-vs-soar-facility-scope.md"
