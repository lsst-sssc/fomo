---
created: 2026-09-02T21:45:00.000Z
title: Retarget ADAPT-03 to SOAR and caveat Phase 33's Gemini outcome propagation
area: general
severity: major
files:
  - solsys_code/models.py
  - .planning/REQUIREMENTS.md
  - .planning/ROADMAP.md
---

## Problem

Gap `G-31-3` (`31-UAT.md`), diagnosed in full at
`.planning/debug/gemini-vs-soar-facility-scope.md`: Phase 31's own committed artifacts
(`31-DECISION.md`, `docs/design/run_identity_and_unattended_invocation_spike.rst`)
presented Gemini as one of three facilities FOMO can read a queue from. It is not.
`GEMFacility` (`tom_observations/facilities/gemini.py`) exposes no read method that
returns real state: `get_observation_status()` (`gemini.py:506-507`) is a hardcoded
stub returning `{'state': '', 'scheduled_start': None, 'scheduled_end': None}`,
`get_observation_url()` (`gemini.py:490-492`) returns an empty string with the real URL
commented out, and the class's only outbound call is `submit_observation()`
(`gemini.py:453-465`). `sync_gemini_observation_calendar.py` never imports `GEMFacility`
and makes no outbound call of its own — it queries
`ObservationRecord.objects.filter(facility='GEM')` (FOMO's own local rows) and builds its
identity key (`sync_gemini_observation_calendar.py:150`,
`url = f'GEM:{prog}/{record.observation_id}'`) from a static settings dict plus the
record FOMO itself submitted. The command is live, working, submission-echo code — not a
queue sync.

`SOARFacility` (`tom_observations/facilities/soar.py:240`) subclasses `LCOFacility` and
inherits its real read path (`get_observation_status()`,
`tom_observations/facilities/ocs.py:1548`, a live GET to the LCO portal). SOAR is already
folded inside `sync_lco_observation_calendar.py` as a distinct facility
(`sync_lco_observation_calendar.py:289`, `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}`;
`sync_lco_observation_calendar.py:298`, `facility__in=['LCO', 'SOAR']`) — there is no
separate SOAR sync command to build.

This same limitation was already recorded once, in an archived requirements document, and
never propagated forward: `v1.5-REQUIREMENTS.md`'s Out of Scope table logged "Live Gemini
ODB status polling | GEMFacility.get_observation_status() is a stub returning empty
state," but that finding never reached PROJECT.md's active limitations, v2.2's `Source`
vocabulary, or v2.3's REQUIREMENTS.md/ROADMAP.md. This todo exists so the same knowledge
is not lost a second time.

## Solution

Three separable items, each a decision for whichever phase implements it, not an
instruction:

**One — the identity vocabulary is missing a value.** `CampaignRun.Source`
(`solsys_code/models.py:108-138`) declares `GEMINI_QUEUE` and has no `SOAR_QUEUE`, even
though `sync_lco_observation_calendar.py` has treated SOAR as a distinct facility since
v1.3. The vocabulary's own docstring already reasons that a different network (ESO)
deserved a dedicated value rather than being folded into `LCO_QUEUE`
(`solsys_code/models.py:126-128`) — the identical argument was never applied to SOAR.
Whichever phase first writes a SOAR-sourced `CampaignRun` owes a new `Source` value (e.g.
`SOAR_QUEUE`, naming the missing member this todo's title refers to) and its migration.

**Two — ADAPT-03 and Phase 32's third success criterion name the wrong facility.**
`REQUIREMENTS.md`'s ADAPT-03 and `ROADMAP.md`'s Phase 32 goal/success-criterion-3 name
`sync_gemini_observation_calendar` as the second facility proving the adapter pattern
generalises. SOAR is the candidate that actually has facility read-back, and it needs no
third command — it is already a branch inside the existing LCO adapter. This is an
internal contradiction to resolve, not a new scope proposal: STATE.md's own stated core
value already reads "Robotically scheduled LCO/SOAR observations and their outcomes
appear and update on the calendar," naming LCO/SOAR with no Gemini, while REQUIREMENTS.md
and ROADMAP.md still name Gemini.

**Three — Phase 33's outcome propagation is structurally impossible for Gemini.**
Outcome propagation needs to read a terminal observing state back from a facility. For
LCO, including SOAR, that read is real (`ocs.py:1548`'s live portal request). For Gemini,
`get_observation_status()` returns a fixed empty state that is in no terminal-state set
and never changes — propagation is impossible there, not merely unimplemented. Phase 33
needs this recorded as an explicit caveat before it plans OUTCOME-01..04, not discovered
as a surprise mid-phase.

**Not proposed by this todo:** no edit to `.planning/ROADMAP.md` or
`.planning/REQUIREMENTS.md` is made here, and no source file under `solsys_code/` or
`src/` is touched. This todo records a decision three future phases owe, on the surface
their planners read.

Evidence: `.planning/debug/gemini-vs-soar-facility-scope.md` (diagnosed session), gap
`G-31-3` in `.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-UAT.md`,
and the correction section appended to
`.planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`.
