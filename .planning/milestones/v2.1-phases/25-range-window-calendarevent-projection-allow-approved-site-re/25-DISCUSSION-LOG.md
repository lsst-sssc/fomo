# Phase 25: Range-window CalendarEvent Projection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-17
**Phase:** 25-range-window-calendarevent-projection-allow-approved-site-re
**Areas discussed:** Ground-branch date math, Banner title format, Backfill for already-approved runs (plus a follow-up clarification on the first area)

---

## Ground-branch date math

| Option | Description | Selected |
|--------|-------------|----------|
| Whole-day span | start_time = window_start 00:00 UTC, end_time = window_end 23:59 UTC, mirroring the satellite branch. One event. | |
| Dip-corrected endpoints | start_time = first night's sunset, end_time = last night's sunrise. One event, more physically precise edges. | |

**User's choice:** Neither as originally framed — user asked "Why not dip-corrected endpoints for each night in the window's range?"

**Notes:** This surfaced a third option neither original framing covered: one `CalendarEvent` PER NIGHT (not one event with adjusted edges). Claude asked a follow-up clarification (below) since this has much larger structural implications (new per-night `url` key scheme; `_set_run_status()` rewrite) than either original option.

### Follow-up: Clarify approach

| Option | Description | Selected |
|--------|-------------|----------|
| One CalendarEvent per night | Mirror `load_telescope_runs`' existing INGEST-01 precedent: E-S+1 dip-corrected nightly events. Requires new per-night `url` key and a `_set_run_status()` rewrite to update all of them. | ✓ |
| One event, dip-corrected edges only | Single CAMPAIGN:{pk} event (no re-keying needed), but start/end come from real sun_event() calls on the first/last night instead of blunt UTC boundaries. | |

**User's choice:** One CalendarEvent per night.

**Notes:** User explicitly picked the larger-scope option, reusing the existing classical-schedule (`load_telescope_runs` INGEST-01) precedent rather than inventing a new single-event-with-adjusted-edges shape. This decision cascaded into CONTEXT.md D-02/D-03/D-04 (new per-night `url` key, `_set_run_status()` rewrite to update multiple events, test-assertion counts changing from `==1` to `== number of nights`).

---

## Banner title format

| Option | Description | Selected |
|--------|-------------|----------|
| Add a window suffix | Range-run titles get `(window {start}..{end})` appended; single-night titles unchanged. | ✓ |
| No change | Range runs get the exact same title format as single-night runs; the multi-day span itself is the only differentiator. | |

**User's choice:** Add a window suffix (recommended option).

**Notes:** This answer was given before the per-night-events clarification above, but remains valid and was carried forward: each per-night event gets the same window-context suffix (not per-night numbering), since — unlike `load_telescope_runs`' genuinely separately-confirmed classical nights — these nights are all part of one awarded allocation the calendar viewer should be able to recognize as connected.

---

## Backfill for already-approved runs

| Option | Description | Selected |
|--------|-------------|----------|
| One-off management command | Finds already-APPROVED, resolved, range-window runs missing events and projects them via the same helper. | ✓ |
| Django data migration | Runs the backfill once at deploy time via a migration. | |
| Manual only — no backfill mechanism | Forward-looking fix only; pk=34 and similar stay eventless unless a human manually re-triggers something. | |

**User's choice:** One-off management command (recommended option).

**Notes:** Directly motivated by the real GS-2026A-FT-115 CampaignRun (pk=34 in the dev DB), which is already APPROVED under the old zero-event behavior — without a backfill mechanism, the user's actual motivating case would stay invisible even after this phase ships.

---

## Claude's Discretion

- Exact per-night `CalendarEvent.url` key format (`CAMPAIGN:{pk}:{date}` recommended default).
- Whether single-night runs keep the bare `CAMPAIGN:{pk}` key unchanged or also move to the new scheme.
- Exact backfill management command name, location, dry-run flag.
- Exact `_set_run_status()` rewrite shape (combined `Q()` filter vs. two filter calls).
- Precise re-derivation of which test assertions change to which values (debug spec's exact counts are superseded by the per-night decision — must re-verify against current file state during planning).

## Deferred Ideas

- Any new calendar visual/UI treatment beyond title text (e.g. a distinct box-shadow ring for "part of a multi-night awarded window") — out of scope for this phase, could be a future phase if the title suffix proves insufficient.
- Extending per-night expansion to the satellite branch — explicitly rejected; satellites have no physical-night concept.
