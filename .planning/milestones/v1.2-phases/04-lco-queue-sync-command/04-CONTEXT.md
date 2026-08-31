# Phase 4: LCO Queue Sync Command - Context

**Gathered:** 2026-06-17
**Status:** Ready for planning

<domain>
## Phase Boundary

A `sync_lco_observation_calendar` Django management command that, given a
`--proposal <code>`, queries `tom_observations.models.ObservationRecord`
where `facility='LCO'` and `parameters['proposal']` matches the supplied
code, and upserts one `tom_calendar.models.CalendarEvent` per matching
record — keyed on the LCO portal URL for idempotency and click-through.
Each event transitions from an unscheduled-queue banner
(`parameters['start']`/`parameters['end']`) to a placed block
(`scheduled_start`/`scheduled_end`) once the LCO scheduler acts, updates in
place on rescheduling (no duplicates, no churn on unchanged records), and is
marked with a status prefix on reaching a terminal state
(`WINDOW_EXPIRED`/`CANCELED`/`FAILURE_LIMIT_REACHED`/`NOT_ATTEMPTED`) while
the event itself is retained for audit trail. No new Django models or
migrations — `CalendarEvent` already has the fields needed.

</domain>

<decisions>
## Implementation Decisions

### Portal URL (idempotency key)
- **D-01:** Build `CalendarEvent.url` via the real
  `LCOFacility().get_observation_url(observation_id)` helper from
  `tom_observations.facilities.lco`/`ocs`, **not** the literal
  `https://observe.lco.global/requestgroups/<id>/` format written in
  REQUIREMENTS.md/PROJECT.md. Verified against the installed `tomtoolkit
  3.0.0a9` source: `OCSFacility.get_observation_url()` (in
  `tom_observations/facilities/ocs.py`) returns
  `urljoin(portal_url, f'/requests/{observation_id}')` →
  `https://observe.lco.global/requests/<observation_id>` (no trailing
  slash, `/requests/` not `/requestgroups/`). Using the helper keeps the
  URL correct automatically if the library's format ever changes, and
  matches what a user actually lands on when clicking through — the
  hardcoded format in the requirements text would 404 on the real portal.
  **This corrects the literal URL format in REQUIREMENTS.md SYNC-01/
  PROJECT.md** — the upsert-by-`url` *behavior* in SYNC-01/03/04 is
  unchanged, only the concrete string differs.

### Telescope field derivation
- **D-02:** `ObservationRecord.parameters` (confirmed a real Django
  `JSONField`, not a `TextField` — corrects the Phase 4 "Key Technical
  Notes" in STATE.md) has no flat `'telescope'` key; only `'instrument_type'`
  (e.g. a code containing `'MUSCAT'`) and a site/location code are
  available. Derive `CalendarEvent.telescope` via a small
  `instrument_type`/site → label map so queue-synced events show the same
  telescope naming as classical-run events from Phase 1-3's `SITES` dict
  (e.g. `'FTS'`) — keeps the calendar view consistent across both event
  sources. The exact `instrument_type` code strings are dynamic (queried
  live from the LCO instruments API by `LCOFacility.get_instruments()`),
  so the planner/researcher should confirm exact values against real
  `ObservationRecord.parameters` data or test fixtures rather than
  guessing from static source alone.

### Title wording
- **D-03:** Title format: `'[QUEUED] {telescope} {instrument}'` while
  `scheduled_start` is `None` (satisfies SYNC-02's "title indicates queue/
  unscheduled status"); once placed (`scheduled_start`/`scheduled_end`
  populated), title becomes the clean `'{telescope} {instrument}'` (no
  prefix) — mirrors Phase 3's D-05 clean-title convention.
- **D-04:** Terminal-state prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`
  per TERM-01) take **priority** over `[QUEUED]` — a record in a terminal
  state shows only the terminal-state prefix, never both.

### Description content
- **D-05:** `CalendarEvent.description` is populated (not left blank) with,
  in order: the proposal code, the record's `status`, and the relevant time
  window — `parameters['start']`/`parameters['end']` while unscheduled, or
  `scheduled_start`/`scheduled_end` once placed. Mirrors Phase 3's
  informational-summary convention (D-06 there) without needing
  `sun_event()` calls (queue records already carry their own times).
  Exact line layout/wording is Claude's discretion, as long as all three
  pieces of information are present.

### Claude's Discretion
- Exact `instrument_type`/site → telescope-label mapping table values (D-02)
  — confirm against real LCO API instrument codes or test fixtures during
  research/planning, not guessed from source reading alone.
- Exact wording/line layout of the `description` field beyond the three
  required pieces (D-05).
- Whether to use `get_or_create` + conditional `save()` or
  `update_or_create` with a pre-comparison for the upsert — any approach
  satisfying SYNC-01/03/04's idempotency-without-churn requirement is
  acceptable (same as Phase 3's established pattern).
- Exact stdout/stderr summary message format for the command run (counts of
  created/updated/unchanged/terminal-marked/skipped), following Phase 3's
  `load_telescope_runs` precedent.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `.planning/REQUIREMENTS.md` — SELECT-01, SYNC-01..05, TERM-01 (this
  phase's locked acceptance criteria; note D-01 above corrects the literal
  URL string in SYNC-01's description, not the upsert behavior)
- `.planning/ROADMAP.md` §"Phase 4: LCO Queue Sync Command" — success
  criteria 1-5 and dependency on Phase 3
- `.planning/PROJECT.md` — "Current Milestone: v1.2 LCO Queue Calendar
  Sync" (note: this doc still says "FTS/MuSCAT4" scope; REQUIREMENTS.md's
  SELECT-01 proposal-code filter is the authoritative, broader scope —
  MuSCAT4-only filtering was explicitly moved to Out of Scope)
- `docs/design/telescope_runs_calendar.rst` — Stage 3 design/feasibility
  notes for the original 4-stage plan

### Existing code (Phase 1-3)
- `solsys_code/telescope_runs.py` — `SITES` dict naming convention (D-02
  should produce labels consistent with these keys, e.g. `'FTS'`)
- `solsys_code/management/commands/load_telescope_runs.py` — closest analog
  management command: `add_arguments`, `handle()`, per-item
  `get_or_create` + conditional-save upsert pattern (D-03/D-04/D-05's
  Claude's-Discretion note point back to this), stdout summary reporting
  style
- `solsys_code/tests/test_load_telescope_runs.py` — existing test
  conventions for this kind of command (Django `TestCase`, fixtures via
  `ObservationRecord`/`CalendarEvent` factories or direct `.objects.create`)

### Third-party models/library (not in this repo)
- `tom_observations.models.ObservationRecord` — confirmed fields via
  installed `tomtoolkit==3.0.0a9`: `target` (FK), `user` (FK), `facility`
  (CharField), `parameters` (**JSONField**, not TextField — corrects
  STATE.md), `observation_id` (CharField), `status` (CharField),
  `scheduled_start`/`scheduled_end` (DateTimeField, nullable), `created`,
  `modified`
- `tom_observations.facilities.ocs.OCSFacility` /
  `tom_observations.facilities.lco.LCOFacility` —
  `get_observation_url(observation_id)` (D-01),
  `get_terminal_observing_states()` returns
  `['WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED',
  'NOT_ATTEMPTED']` (matches TERM-01 exactly — confirmed, no correction
  needed), `get_instruments()` (dynamic instrument codes for D-02)
- `tom_observations.facility.BaseObservationForm.serialize_parameters()` —
  `parameters` on a created `ObservationRecord` is
  `copy.deepcopy(cleaned_data)` minus `'groups'`, i.e. a flat dict of the
  submission form's fields (`'proposal'`, `'start'`, `'end'`,
  `'instrument_type'`, etc.) — confirms `parameters['proposal']`/
  `parameters['start']`/`parameters['end']` are directly accessible as
  REQUIREMENTS.md assumes, for the FTS/MuSCAT4 cadence/queue submission
  form. Researcher should verify exact value types (str vs datetime) once
  real or fixture data is available, since JSONField requires
  JSON-serializable values.
- `tom_calendar.models.CalendarEvent` — confirmed fields: `title`,
  `description`, `start_time`, `end_time`, `url` (URLField), `target_list`
  (FK), `user`, `proposal`, `telescope`, `instrument`, `created`,
  `modified` — no unique constraint beyond `pk`, so idempotency is
  app-level (same as Phase 3)
- LCO portal base URL: `'https://observe.lco.global'` (confirmed in
  `src/fomo/settings.py` `FACILITIES['LCO']['portal_url']`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/management/commands/load_telescope_runs.py` — the
  established upsert pattern (`get_or_create` + conditional update only
  when fields differ, per-item error handling, stdout summary) to mirror
  for this command
- `solsys_code/telescope_runs.py:SITES` — naming convention to match for
  the telescope-label mapping (D-02)

### Established Patterns
- Google-style docstrings (`Args`/`Returns`/`Raises`)
- `ValueError`/library exceptions caught per-record, logged, and skipped
  rather than aborting the whole run (Phase 3 D-02 precedent)
- No-churn idempotency: only call `.save()` when fields actually changed,
  to avoid `modified`-timestamp churn (Phase 3 D-04 precedent, restated in
  STATE.md's Phase 4 technical notes)
- DB-dependent tests live in `solsys_code/tests/`, run via
  `./manage.py test solsys_code`

### Integration Points
- `ObservationRecord.objects.filter(facility='LCO')` then Python-side (or,
  since `parameters` is a confirmed JSONField, potentially ORM-level
  `parameters__proposal=code`) filtering for SELECT-01 — research/planning
  should confirm whether DB-level JSON lookup works reliably on this
  project's SQLite + Django JSONField setup before relying on it
- `LCOFacility()` instantiation needed to call `get_observation_url()`
  (D-01) and `get_terminal_observing_states()` (TERM-01) — check
  constructor requirements (e.g. does it need network/API credentials at
  init, or only when actually submitting/querying the portal?)

</code_context>

<specifics>
## Specific Ideas

No additional UI/UX references — this phase is a management command with
no template/UI changes (matches Phase 2/3 precedent). No specific sample
`ObservationRecord` fixtures were provided during discussion; the planner/
researcher will need to construct or find representative
`parameters`/`status`/`scheduled_start` test fixtures for FTS/MuSCAT4-style
LCO queue submissions to validate SYNC-01 through SYNC-05 and TERM-01.

</specifics>

<deferred>
## Deferred Ideas

- Stage 4 full sync (all LCO facilities/instruments, not just records
  matched by `--proposal`) — already noted as future work in PROJECT.md/
  REQUIREMENTS.md, not raised again here.
- `TARG-01` (populate `CalendarEvent.target_list` from the record's
  `target` FK) and `TARG-02` (non-LCO facility sync) — already captured in
  REQUIREMENTS.md's "Future Requirements", not discussed further.
- Verifying whether `parameters__proposal=code` ORM-level JSON filtering is
  reliable on this project's SQLite setup, vs. always filtering in Python
  — flagged as a research question (see code_context) rather than a user
  decision; not resolved during this discussion.

None of these are new capabilities surfaced during discussion — all were
already tracked in REQUIREMENTS.md/PROJECT.md or are implementation
questions for the researcher, not scope creep.

</deferred>

---

*Phase: 4-LCO Queue Sync Command*
*Context gathered: 2026-06-17*
