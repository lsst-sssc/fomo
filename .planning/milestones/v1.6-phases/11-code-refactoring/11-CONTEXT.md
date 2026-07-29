# Phase 11: Code Refactoring - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract shared telescope-mapping helpers and the no-churn CalendarEvent
create-or-update pattern from `sync_lco_observation_calendar.py` into a new
`solsys_code/calendar_utils.py` module; update all three management commands
to import and use the shared code; replace "upsert" in live docs with plain
English or the function name.

No new features. All tests must pass with no behavior change.

</domain>

<decisions>
## Implementation Decisions

### Module Structure
- **D-01:** One new file: `solsys_code/calendar_utils.py`. Both the LCO/SOAR
  telescope-mapping helpers (REFAC-01) and `insert_or_create_calendar_event()`
  (REFAC-02) live in this single module.

### REFAC-01 — Symbols to extract from sync_lco_observation_calendar.py
- **D-02:** Extract the following symbols from `sync_lco_observation_calendar.py`
  into `calendar_utils.py`:
  - Constants: `SITE_TELESCOPE_MAP`, `_API_TIMEOUT_SECONDS`,
    `_SCIENCE_CONFIGURATION_TYPES`, `_MUSCAT_CHANNEL_SUFFIXES`
  - Exception: `InstrumentExtractionError`
  - Functions (telescope-class helpers):
    `_aperture_class_from_telescope_code`, `_derive_telescope`,
    `_resolve_placement_block`, `_coarse_telescope_label`
  - Functions (instrument-extraction chain):
    `_extract_instrument`, `_find_science_config`,
    `_find_exposure_signal_config`, `_has_muscat_exposure_signal`

  **Not extracted** (stay in the command file — orchestration/CLI logic):
  `_failure_prefix`, `_FAILURE_PREFIX_BY_STATUS`, `_title_for`, `_time_window`,
  `_build_event_fields`, `_parse_proposal_arg`.

  `SITE_TELESCOPE_MAP` must remain a separate data structure from
  `telescope_runs.py:SITES` — different purpose/shape (per Phase 7 decision D-05).

### REFAC-02 — insert_or_create_calendar_event() API
- **D-03:** Function signature:
  ```python
  def insert_or_create_calendar_event(
      lookup: dict[str, Any],
      fields: dict[str, Any],
  ) -> tuple[CalendarEvent, str]:
  ```
  Returns `(event, action)` where `action` is one of `'created'`,
  `'updated'`, or `'unchanged'`. Callers own counter updates.

  The generic `lookup` dict covers all three consumers:
  - `sync_lco` / `sync_gemini`: `{'url': url}`
  - `load_telescope_runs`: `{'telescope': ..., 'instrument': ..., 'start_time': ...}`

  The `CalendarEventTelescopeLabel` sidecar write in `sync_lco` is **not**
  folded into this function — it is LCO-specific (Phase 8 DISPLAY-01) and
  stays in the command after calling `insert_or_create_calendar_event()`.

- **D-04:** "upsert" in `docs/design/telescope_runs_calendar.rst` (2
  occurrences) and `.planning/MILESTONES.md` (1 occurrence) replaced with
  plain English ("creates or updates") or the function name. No "upsert"
  appears in the command files themselves.

### Claude's Discretion
- **Internal save behaviour:** Use `event.save(update_fields=changed_fields)`
  in the shared function (list of changed field names, not all fields). Strictly
  better than the full `event.save()` LCO currently uses; semantically
  identical. Gemini already uses this form.
- **Test strategy:** No new direct unit tests for `calendar_utils.py`. Existing
  `./manage.py test solsys_code` tests (186 tests) verify behavior-neutrality
  end-to-end. This is consistent with success criterion #4.

### Folded Todos
- **Todo: "Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`,
  tagged `resolves_phase: 11`). Addressed by D-02 above. The todo names the
  exact function list and confirms `SITE_TELESCOPE_MAP` must stay separate from
  `telescope_runs.py:SITES`. Its suggested module name (`lco_observation_mapping.py`)
  was superseded by the user's choice of `calendar_utils.py` (D-01).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v1.6 Requirements — REFAC-01, REFAC-02
  (exact success criteria for what must be true)
- `.planning/ROADMAP.md` Phase 11 section — Goal, Depends on, Success Criteria

### Source Code (targets for extraction)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` —
  source of all REFAC-01 extractions and the existing no-churn CalendarEvent
  pattern (lines ~35–500 before the Command class)
- `solsys_code/management/commands/load_telescope_runs.py` —
  consumer; uses `(telescope, instrument, start_time)` lookup key
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` —
  consumer; uses `url` lookup key

### Docs with "upsert" to replace (REFAC-02)
- `docs/design/telescope_runs_calendar.rst` — 2 occurrences
- `.planning/MILESTONES.md` — 1 occurrence

### Context from Prior Investigation
- `.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` —
  Phase 7 review notes, confirmed function list, SNEX2/MOP research, D-05 warning

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/solsys_code_observatory/utils.py` — existing precedent for a
  utility module in `solsys_code/` with no `__all__`, public functions via
  naming convention (no leading underscore = public)
- `solsys_code/telescope_runs.py` — another flat module in `solsys_code/`;
  the new `calendar_utils.py` follows the same pattern (not a sub-package)

### Established Patterns
- No `__all__` in `solsys_code/` modules — public/private signalled by
  leading underscore only
- Google-style docstrings; `D103` (missing public function docstring) is
  enforced by ruff — all public functions in `calendar_utils.py` need one
- Relative imports within `solsys_code/`: `from .calendar_utils import ...`
  (per CONVENTIONS.md dot-notation pattern)
- Constants: UPPER_CASE; private functions: `_lower_case`

### Integration Points
- All three commands update their imports: add `from .calendar_utils import
  SITE_TELESCOPE_MAP, _extract_instrument, InstrumentExtractionError, ...`
  (relative) and `insert_or_create_calendar_event`
- `_build_event_fields` in `sync_lco` calls functions that move to
  `calendar_utils.py` — its import section in the command needs updating
  but the function body stays in the command

</code_context>

<specifics>
## Specific Ideas

- The function name `insert_or_create_calendar_event` (not `upsert_calendar_event`)
  is the canonical name — the requirement explicitly specifies this spelling and
  it replaces "upsert" in docs (REFAC-02).
- `insert_or_create_calendar_event()` should use `save(update_fields=changed_fields)`
  where `changed_fields` is the list of changed field names (not all fields),
  aligning the LCO command with Gemini's more efficient form.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-Code Refactoring*
*Context gathered: 2026-06-27*
