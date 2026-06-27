# Phase 10: Gemini Calendar Sync Command - Context

**Gathered:** 2026-06-26
**Status:** Ready for planning

<domain>
## Phase Boundary

A management command (`sync_gemini_observation_calendar`) that reads all
`ObservationRecord(facility='GEM')` rows from the database, derives `CalendarEvent`
fields (telescope, instrument, proposal, title, window) from their `parameters` JSON,
and creates/updates events idempotently — no live Gemini API calls, no telescope
resolution, no sidecar model. The command covers all 10 GEM-* requirements and includes
a pre-executed demo notebook. This is a read-from-DB / write-to-calendar transform,
substantially simpler than the LCO sync despite the same idempotency contract.

</domain>

<decisions>
## Implementation Decisions

### ToO-type fallback when settings are missing (GEM-WINDOW-02)

- **D-01:** When an `ObservationRecord` has no explicit `windowDate`/`windowTime` and
  its obs code is absent from `FACILITIES['GEM']['programs']` (so the `Std:`/`Rap:`
  ToO-type prefix can't be determined), the command **skips the record** and increments
  the `skipped` counter. It logs a `WARNING` that names the specific `prog` and `obs_code`
  that are missing from settings, so the operator knows exactly what to add.
  No fallback wide-window event is created — an event with unknown time bounds would be
  misleading on the calendar.
- **D-02:** The `Std:`/`Rap:` ToO-type is read from the settings-description prefix in
  `FACILITIES['GEM']['programs'][prog][obs_code]`, not from any field in `parameters`.
  Same lookup that GEM-INSTR-01 uses for the instrument name — two outputs from one read.

### Multi-obsid submissions

- **D-03 (Claude's discretion):** When `params['obsid']` contains multiple entries
  (standard + rapid of the same template, submitted together), use `params['obsid'][0]`
  for both the instrument-description lookup and the ToO-type detection. When
  `len(params['obsid']) > 1`, emit a `logging.WARNING` that logs the full obsid list so
  operators can see what was found. The instrument label is typically identical for
  Std./Rap. variants of the same physical instrument, so the label is practically correct
  in the common case; the ToO-type window may be off on the second record (documented
  limitation). Direct matching of `observation_id` (numeric suffix only) back to the
  obsid code is not feasible from stored data alone.

### Password scrubbing (GEM-SECURE-01)

- **D-04 (Claude's discretion):** Strip the `password` key from the parameters dict
  **immediately at the start of processing each record**, before any logging, field
  derivation, or exception paths. Pattern: `safe_params = {k: v for k, v in
  record.parameters.items() if k != 'password'}` — use `safe_params` everywhere
  downstream. This is the simplest robust approach: no custom log filters, no risk of
  accidental leakage through Django exception tracebacks. The `CalendarEvent` model has
  no `parameters` field, so there's no write-side exposure.

### Demo notebook fixture design

- **D-05:** Synthetic `ObservationRecord` parameters use realistic program ID structure
  drawn from Gemini South 2026A ToO queue naming conventions
  (`https://www.gemini.edu/observing/schedules-and-queue/queue-summary-bands-dd-lp-ft-pw?semester=2026A&site=South&queue=SQ`)
  but with the program number changed to `999` at the end (e.g. `GS-2026A-T-999`).
  This gives realistic-looking IDs that can't be confused with real submissions.
- **D-06:** The demo notebook must cover all four scenarios explicitly:
  1. **Explicit window** — record with `windowDate`/`windowTime`/`windowDuration` present
     (GEM-WINDOW-01, the primary happy path).
  2. **Rap: derived window** — record with no explicit window, obs code maps to a `'Rap: ...'`
     settings entry → `[record.created, record.created + 24h]` (GEM-WINDOW-02).
  3. **Std: derived window** — record with no explicit window, obs code maps to a `'Std: ...'`
     settings entry → `[record.created + 24h, record.created + 7d]` (GEM-WINDOW-02).
  4. **ON_HOLD + idempotent re-run** — a record with `ready='false'` shows `[ON_HOLD]`
     prefix; re-running the command on all records produces no new events and leaves
     `CalendarEvent.modified` unchanged (GEM-STATUS-01, GEM-NOCHURN-01).
- **D-07:** The settings fixture in the notebook should define a minimal `FACILITIES['GEM']`
  override (patched or via `override_settings`) that maps the `GS-2026A-T-999` program and
  its obs codes to illustrative descriptions (`'Rap: GMOS-S MOS'`, `'Std: GMOS-S MOS'`).
  This lets GEM-INSTR-01 and ToO-type detection run realistically without real credentials.
  Password field in parameters: use `'password': '[redacted]'` as a harmless placeholder.

### Command output format

- **D-08:** Mirror the LCO command's output format exactly: one line per Gemini site
  (South / North) with `created / updated / unchanged / skipped` counts, e.g.:
  ```
  Gemini South: created: 2, updated: 0, unchanged: 1, skipped: 0
  Gemini North: created: 0, updated: 0, unchanged: 0, skipped: 0
  Done.
  ```
  No `extraction_failed` or `telescope_api_failed` counters (those are LCO-specific).
  Operators already familiar with the LCO sync output format will recognise this immediately.

### Claude's Discretion

- Multi-obsid handling (D-03): use first entry + WARNING log.
- Password scrubbing mechanism (D-04): strip at record load time.
- Exact counter label spelling and line format (D-08): mirror LCO as above.
- Whether to add a `--proposal` flag analogous to the LCO command's `--proposal` filter:
  Claude may add it if it fits naturally given the LCO analog, but it is not required.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` (Phase 10 section) — goal, success criteria (5 items), GEM-* requirements list
- `.planning/REQUIREMENTS.md` (all GEM-* requirements, v1.5 section) — 10 locked requirements with
  traceability; also read the "Out of Scope" table (no sidecar model, no ODB polling, no GOATS/GPP,
  no archive retrieval)
- `.planning/PROJECT.md` — current state section (v1.4 shipped; Stage 4 = Gemini is v1.5 scope)

### Analog management command (read in full)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — **direct structural analog**.
  The Gemini command should mirror its `add_arguments()` / `handle()` / private helper pattern,
  its `get_or_create(url=url, defaults=fields)` + `update_fields` no-churn idiom, and its
  `created/updated/unchanged/skipped` counter structure. Key differences: no `LCOFacility` API
  calls, no `CalendarEventTelescopeLabel` sidecar, simpler per-record field derivation.

### Settings structure (instrument lookup)
- `src/fomo/settings.py:227-244` — `FACILITIES['GEM']` block. The `programs` sub-dict maps
  `prog_id → {obs_code → 'Std:/Rap: description'}`. This is the authoritative source for
  instrument descriptions (GEM-INSTR-01) and ToO-type detection (GEM-WINDOW-02 D-02).

### Gemini facility source (TOM Toolkit)
- Installed at (find with `find $VIRTUAL_ENV -path "*/facilities/gemini.py"`) — read for:
  - `GEMObservationForm.serialize_parameters()` parameter key names (`prog`, `obsid`, `password`,
    `ready`, `windowDate`, `windowTime`, `windowDuration`)
  - `windowDate` = `'YYYY-MM-DD'` UTC, `windowTime` = `'HH:MM'` UTC, `windowDuration` = hours as str
  - `submit_observation` returns numeric-suffix-only `observation_id` (response split on `-`, last part)
  - `get_observation_status()` is a stub returning empty state — no live Gemini API in this command

### Demo notebook
- `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` — must be created as
  part of this phase (CLAUDE.md companion-notebook convention). Execute with
  `jupyter nbconvert --to notebook --execute --inplace` and commit with output.
- Reference notebook for structure/style:
  `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`

### Gemini program naming reference
- `https://www.gemini.edu/observing/schedules-and-queue/queue-summary-bands-dd-lp-ft-pw?semester=2026A&site=South&queue=SQ`
  — Gemini South 2026A ToO queue summary. Use for realistic program ID structure in notebook
  fixtures (program number changed to `999`).

### Process conventions
- `CLAUDE.md` — demo notebook is a mandatory deliverable listed by name; password security
  requirement; plain-English-over-jargon convention; ruff single-quote / 120-col style
- Prior phase context (archived, for reference only):
  - `.planning/phases-archive/09-proposal-color-status-visual-treatment/09-CONTEXT.md` —
    confirms `CalendarEvent.proposal` is set from `params['prog']`, which feeds the Phase 9
    `calendar_display_extras` proposal-color template tag automatically; no additional color
    work needed in this phase.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sync_lco_observation_calendar.Command` structure (`add_arguments`, `handle`, per-facility
  counters dict, `get_or_create` + `update_fields` no-churn block, `self.stdout.write` summary
  lines) — copy and simplify for Gemini. The LCO version is 653 lines; the Gemini version should
  be substantially shorter (no live API, no sidecar, simpler field derivation).
- `CalendarEvent` model (from `tom_calendar`) — same model used by LCO sync. Fields needed:
  `url` (unique key), `start_time`, `end_time`, `title`, `telescope`, `instrument`, `proposal`,
  `color`. The `color` property is `BOOTSTRAP_COLORS[self.pk % 9]` (read-only, computed — the
  Phase 9 `calendar_display_extras` template tag bypasses it for display; just leave it as-is).

### Established Patterns
- **No-churn idiom**: `event, created = CalendarEvent.objects.get_or_create(url=url, defaults=fields)`.
  On `not created`: compare each field, collect changed field names, call `event.save(update_fields=changed)`
  only if `changed` is non-empty. Increment `unchanged` if nothing changed. This pattern is
  battle-tested across Phases 3, 4, 5, 7 — do not deviate.
- **Unique URL key**: `f"GEM:{params['prog']}/{record.observation_id}"` — stable, never empty,
  human-readable, parallels LCO's `f"LCO:{record.observation_id}"` key.
- **Password strip at load time**: `safe_params = {k: v for k, v in record.parameters.items()
  if k != 'password'}` — use `safe_params` throughout, never `record.parameters` directly.
- **windowDate/windowTime → datetime**: `datetime.strptime(windowDate, '%Y-%m-%d')` combined with
  `datetime.strptime(windowTime, '%H:%M')`, then `replace(tzinfo=timezone.utc)`. Duration in
  hours as str → `int(windowDuration)`.

### Integration Points
- `ObservationRecord.objects.filter(facility='GEM')` — the query source for the command.
  No additional joins needed; `parameters` is a JSON field already deserialized by Django ORM.
- `CalendarEvent` (write side) — same as every prior sync phase. No migration needed (model
  unchanged since Phase 3).
- `GEMFacility.get_observation_status()` is a stub — this command does NOT call it. Telescope
  and instrument are derived purely from `parameters` + settings.
- `CalendarEventTelescopeLabel` sidecar — NOT used for Gemini events. Per requirements "Out of
  Scope": telescope is deterministic from program prefix (`GS-*` / `GN-*`), so the
  missing-row = "verified" convention in `calendar.html` applies automatically.
- The Phase 9 `calendar_display_extras.proposal_color` template tag reads `CalendarEvent.proposal`
  at render time — setting `proposal = params['prog']` here is all that's needed for Gemini events
  to inherit proposal-keyed coloring automatically.

</code_context>

<specifics>
## Specific Ideas

- Program IDs in the demo notebook should look like real Gemini IDs but end in `999`
  (e.g. `GS-2026A-T-999`) — draw the program-type letter and semester from the actual
  2026A Gemini South ToO queue page linked in canonical refs.
- The settings fixture in the notebook should patch `FACILITIES['GEM']['programs']` with
  at least two obs codes: one `'Rap: ...'` and one `'Std: ...'` for the same program, so
  the notebook can demonstrate both derived-window paths in sequence.
- Password placeholder in notebook params: `'password': '[redacted]'` — never a real key.
- Multi-obsid warning message should explicitly list the full `params['obsid']` list so an
  operator reviewing the log can see exactly which templates were bundled together.
- Skip warning message should name the specific `prog` and `obs_code` missing from settings,
  e.g.: `WARNING: GS-2026A-T-999 obs code 'QQ' not found in FACILITIES['GEM']['programs']
  — skipping ObservationRecord <pk>`.

</specifics>

<deferred>
## Deferred Ideas

- **`--proposal` filter flag** — analogous to `sync_lco_observation_calendar`'s `--proposal`
  argument. Not in requirements but may be added at Claude's discretion if it fits naturally
  given the LCO analog (D-08 / Claude's Discretion). If not added in Phase 10, note for backlog.
- **GOATS / GPP integration** (GEM-GPP-01/02 from REQUIREMENTS.md future requirements) —
  out of scope for v1.5; requires `python < 3.11` and GOATS not installed.
- **Live Gemini ODB status polling** — `GEMFacility.get_observation_status()` is a stub;
  no live API in this command.

</deferred>

---

*Phase: 10-Gemini Calendar Sync Command*
*Context gathered: 2026-06-26*
