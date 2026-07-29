# Phase 4: LCO Queue Sync Command - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-17
**Phase:** 4-LCO Queue Sync Command
**Areas discussed:** Portal URL format, Telescope field derivation, Title wording for banner vs placed states, Description field content

---

## Portal URL format

| Option | Description | Selected |
|--------|-------------|----------|
| Use LCOFacility.get_observation_url() | Call the real helper from tom_observations.facilities.lco — produces 'https://observe.lco.global/requests/<observation_id>' (no trailing slash). Stays correct automatically if the library changes the format; matches what a user would actually land on when clicking through. | ✓ |
| Hardcode '/requestgroups/<id>/' per REQUIREMENTS.md text | Keep the literal format written in REQUIREMENTS.md/PROJECT.md even though it doesn't match the installed library's actual URL — links would 404 on the real LCO portal. | |
| Hardcode '/requests/<id>' without using the helper | Match the real format but inline it as a string rather than calling get_observation_url() — avoids depending on facility instantiation/config but duplicates the literal. | |

**User's choice:** Use LCOFacility.get_observation_url() (Recommended)
**Notes:** Verified the contradiction by reading the installed `tomtoolkit==3.0.0a9` source (`tom_observations/facilities/ocs.py`) before asking — REQUIREMENTS.md's `/requestgroups/<id>/` format does not match the actual library behavior.

---

## Telescope field derivation

| Option | Description | Selected |
|--------|-------------|----------|
| Map via instrument_type | Build a small instrument_type -> SITES-style label map (e.g. '2M0-SCICAM-MUSCAT' -> 'FTS') so queue events show the same telescope names as classical-run events (Phase 1-3's SITES dict) — consistent calendar view. | ✓ |
| Use raw OCS site code | Store whatever site code appears in parameters (e.g. 'coj') directly — simpler, no mapping table to maintain, but won't match classical-run event telescope labels in the calendar. | |
| You decide | Let Claude/the planner pick the simplest approach that satisfies SYNC-05 without over-engineering a mapping table. | |

**User's choice:** Map via instrument_type (Recommended)
**Notes:** Exact `instrument_type` code strings are dynamic (queried live from the LCO instruments API), so the planner/researcher will need to confirm exact mapping values against real data or test fixtures.

---

## Title wording for banner vs placed states

| Option | Description | Selected |
|--------|-------------|----------|
| '[QUEUED] {telescope} {instrument}' -> '{telescope} {instrument}' | Mirrors Phase 3's clean 'telescope instrument' title once placed; '[QUEUED]' prefix only while scheduled_start is None, matching TERM-01's bracketed-status-prefix convention used for terminal states. | ✓ |
| Status word in title always | Title always carries a parenthetical status word, even after placement, so the calendar always shows current state at a glance. | |
| You decide | Let Claude pick exact wording at planning time, as long as SYNC-02's requirement is satisfied. | |

**User's choice:** '[QUEUED] {telescope} {instrument}' -> '{telescope} {instrument}' (Recommended)
**Notes:** Terminal-state prefixes ([EXPIRED]/[CANCELLED]/[FAILED] per TERM-01) take priority over [QUEUED] when a record is in a terminal state.

---

## Description field content

| Option | Description | Selected |
|--------|-------------|----------|
| Proposal + status + window/placed times | e.g. 'Proposal: PROPOSAL2025A-001 / Status: PENDING / Window: <start> to <end>' (or 'Scheduled: <scheduled_start> to <scheduled_end>' once placed) — mirrors Phase 3's informational-summary convention without needing sun_event() calls. | ✓ |
| Leave blank/unused | Don't populate description at all for this phase — url/telescope/instrument/proposal/title already carry the needed info. | |
| You decide | Let Claude pick reasonable description content at planning time, consistent with Phase 3's style. | |

**User's choice:** Proposal + status + window/placed times (Recommended)
**Notes:** None.

---

## Claude's Discretion

- Exact `instrument_type`/site → telescope-label mapping table values — confirm against real LCO API instrument codes or test fixtures during research/planning.
- Exact wording/line layout of the `description` field beyond the three required pieces.
- Whether to use `get_or_create` + conditional `save()` or `update_or_create` with a pre-comparison for the upsert.
- Exact stdout/stderr summary message format for the command run.

## Deferred Ideas

- Stage 4 full sync (all LCO facilities/instruments) — already future work in PROJECT.md/REQUIREMENTS.md.
- TARG-01/TARG-02 — already captured in REQUIREMENTS.md's "Future Requirements".
- Whether `parameters__proposal=code` ORM-level JSON filtering is reliable on this project's SQLite setup — flagged as a research question, not a user decision.
