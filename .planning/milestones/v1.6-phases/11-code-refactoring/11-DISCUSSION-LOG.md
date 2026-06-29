# Phase 11: Code Refactoring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-27
**Phase:** 11-code-refactoring
**Areas discussed:** Module naming & split, Extraction scope for REFAC-01, insert_or_create_calendar_event() API

---

## Module naming & split

| Option | Description | Selected |
|--------|-------------|----------|
| Two files (Recommended) | LCO helpers in one module; insert_or_create_calendar_event() in separate module. Clean separation of LCO-specific vs. generic. | |
| One file | Everything in a single calendar_utils.py. Simpler, mixes LCO-specific and generic. | ✓ |

**User's choice:** One file

**Notes:** None — selected without elaboration. All LCO/SOAR telescope-mapping helpers and `insert_or_create_calendar_event()` go in `solsys_code/calendar_utils.py`.

---

## Module name

| Option | Description | Selected |
|--------|-------------|----------|
| calendar_utils.py (Recommended) | Generic 'calendar utilities' — signals CalendarEvent helpers. Matches _utils.py pattern. | ✓ |
| lco_observation_mapping.py | Todo's suggestion. Accurate for LCO content, misleading for the generic insert_or_create function. | |
| telescope_run_utils.py | Feature-domain-specific. Less precise about contents. | |

**User's choice:** `calendar_utils.py` (recommended)

---

## Extraction scope for REFAC-01

| Option | Description | Selected |
|--------|-------------|----------|
| Just what the TODO names + InstrumentExtractionError (Recommended) | 8 functions/constants from todo + InstrumentExtractionError. Leaves orchestration/CLI helpers in command. | |
| Everything above the Command class | All ~19 module-level symbols to calendar_utils.py. Cleaner but moves CLI helpers that don't belong. | |
| You decide (minimal viable extraction) | Use TODO's named list as boundary; move anything else that clearly belongs. | ✓ |

**User's choice:** Claude's discretion (minimal viable extraction)

**Claude's decision:** TODO-named list + InstrumentExtractionError + `_coarse_telescope_label` + constants needed by moved functions (`_API_TIMEOUT_SECONDS`, `_SCIENCE_CONFIGURATION_TYPES`, `_MUSCAT_CHANNEL_SUFFIXES`). Not moved: `_title_for`, `_time_window`, `_build_event_fields`, `_failure_prefix`, `_parse_proposal_arg` (orchestration/CLI).

---

## insert_or_create_calendar_event() API

| Option | Description | Selected |
|--------|-------------|----------|
| Generic: (lookup_kwargs, fields) → tuple[CalendarEvent, str] (Recommended) | Accepts any dict of ORM lookup kwargs + fields dict. Covers all three commands. Returns (event, 'created'\|'updated'\|'unchanged'). | ✓ |
| URL-only: (url, fields) → tuple[CalendarEvent, str] | Simpler for the common case. load_telescope_runs keeps inline pattern. | |

**User's choice:** Generic API (recommended)

**Notes:** Sidecar write (`CalendarEventTelescopeLabel`) stays in the `sync_lco` command — it is LCO-specific and not part of the shared function.

---

## Claude's Discretion

- **Internal save behaviour:** `save(update_fields=changed_fields)` — list of changed field names, not all fields. More efficient than LCO's current `event.save()`; Gemini already uses this form.
- **Test strategy:** No new unit tests for `calendar_utils.py`. Existing 186 command tests verify behavior-neutrality.

## Deferred Ideas

None — discussion stayed within phase scope.
