# Phase 3: Classical Calendar Ingest - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

A `load_telescope_runs` Django management command that reads a file of
classical-schedule run lines (one per line, parsed via Phase 2's
`telescope_runs.parse_run_line()`), expands each `ParsedRun` into one
`tom_calendar.CalendarEvent` per observing night (`E - S + 1` nights,
`start_time = sunset(d)`, `end_time = sunrise(d+1)` via Stage 1's
`get_site()`/`sun_event()`), and upserts those events idempotently. No new
Django models or migrations — `CalendarEvent` already has the fields needed.

</domain>

<decisions>
## Implementation Decisions

### CLI input & per-line errors
- **D-01:** `load_telescope_runs` takes a **positional file path argument**
  (e.g. `./manage.py load_telescope_runs schedule.txt`), modelled on
  `fetch_jplsbdb_objects`'s command structure but with a required positional
  arg rather than flags.
- **D-02:** Each line is parsed independently via `parse_run_line()`. If a
  line raises `ValueError` (including the Phase 2 D-01 ambiguous-`'Magellan'`
  case, bad date ranges, or unrecognized status), **log the error to
  stderr** (including the line number and original text) and **continue**
  processing remaining lines — do not abort the whole run. At the end, print
  a summary: counts of lines processed, events created, events
  updated/unchanged, and lines skipped due to errors.

### Idempotency / upsert key
- **D-03:** The upsert key for "the same event" across re-runs is
  **`telescope` + `instrument` + `start_time`** (all three). Use
  `CalendarEvent.objects.get_or_create()` (or equivalent get-then-update) on
  this triple.
- **D-04:** On a re-run where the computed `end_time`, `title`, or
  `description` would be **unchanged**, leave the existing row untouched (no
  write) — avoids unnecessary `modified` timestamp churn and preserves any
  manual edits unless the schedule actually changed. If any of those fields
  *would* differ from the existing row, update them in place (still keyed on
  `telescope`+`instrument`+`start_time`, so an instrument change for the same
  telescope/night updates the existing row rather than creating a duplicate).

### Title & description format
- **D-05:** `title = f"{telescope} {instrument}"` (e.g. `"NTT EFOSC2"`,
  `"Magellan-Clay IMACS"`) — clean and queryable per the design doc; status is
  **not** included in the title.
- **D-06:** `description` contains, in order: the -15° dark-window times for
  that night (UTC, from `sun_event(site, d, 'dark')`), the run's `status`
  (e.g. `"Status: allocation"`), and the original run-line text (e.g.
  `"Source line: NTT EFOSC2 allocation 9-13 July"`). Exact formatting/line
  layout is Claude's discretion at planning/implementation time, as long as
  all three pieces of information are present and INGEST-02 is satisfied.

### Status-dependent behavior
- **D-07:** Status (`allocation`/`proposed`/`confirmed`/`not confirmed`/
  `cancelled`) is **informational only** for this phase — it appears in the
  description (D-06) but does not change whether/how events are
  created/updated/deleted. A `'cancelled'`-status line still creates/updates
  events the same as any other status; deleting previously-created events for
  cancelled runs is explicitly out of scope (see Deferred Ideas).

### Claude's Discretion
- Exact per-night iteration approach (loop over `range(day1, day2+1)`,
  handling month/year boundaries within a run — note a run could span a
  month boundary even though `ParsedRun` only stores one `month`/`year` for
  `day1`).
- Exact wording/formatting of the `description` field beyond the three
  required pieces of information (D-06).
- Exact stdout/stderr message formats and the end-of-run summary's exact
  wording, as long as it reports created/updated/skipped counts.
- Whether to use `get_or_create` + conditional `save()`, or `update_or_create`
  with a pre-comparison — any approach satisfying D-03/D-04 is acceptable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/telescope_runs_calendar.rst` — "The Data Model" (CalendarEvent
  field mapping), "Astronomy: Night Boundaries" (night convention, `E - S + 1`
  nights), "Classical Run Input Format" (night convention details), "Stage 2 —
  classical ingest" success criteria
- `.planning/REQUIREMENTS.md` — INGEST-01..03 (this phase)
- `.planning/ROADMAP.md` — Phase 3 success criteria and dependency on Phase 2

### Existing code (Phase 1 & 2)
- `solsys_code/telescope_runs.py` — `ParsedRun`, `parse_run_line()`,
  `SITES`/`get_site()`, `sun_event()` (all consumed by this phase)
- `solsys_code/tests/test_telescope_runs.py` — existing test suite/conventions
  (Django `TestCase`, `solsys_code/tests/`)
- `solsys_code/management/commands/fetch_jplsbdb_objects.py` — existing
  management-command pattern (argument parsing via `add_arguments`,
  `self.stdout.write`/`self.stderr.write` for reporting)

### Third-party model
- `tom_calendar.models.CalendarEvent` (installed package, not in this repo) —
  fields `title`, `description`, `start_time`, `end_time`, `telescope`,
  `instrument`, `url`, `proposal`, `user`, `target_list`; no unique
  constraints beyond `pk`, so idempotency is enforced at the application
  level per D-03/D-04

No other external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/telescope_runs.py:parse_run_line()` / `ParsedRun` — Phase 2's
  parser output is this phase's input
- `solsys_code/telescope_runs.py:get_site()` / `sun_event()` — Phase 1's
  site lookup and sun-event computation, used per-night for `start_time`/
  `end_time`/dark window
- `solsys_code/management/commands/fetch_jplsbdb_objects.py` — closest analog
  for a new management command: `add_arguments`, `handle()`, stdout/stderr
  reporting style

### Established Patterns
- Google-style docstrings (`Args`/`Returns`/`Raises`)
- `ValueError` for invariant violations; per-line error handling here follows
  the same convention but at the command-orchestration layer (catch and log,
  don't propagate)
- Tests for Django-DB-dependent code live in `solsys_code/tests/`, run via
  `./manage.py test solsys_code`

### Integration Points
- `ParsedRun.telescope` is a resolved `SITES` key; pass directly to
  `get_site()`
- For each night `d` in `[day1, day2]` (handling month/year rollover within
  the range), call `sun_event(site, d, 'sun')` for `start_time`/`end_time` and
  `sun_event(site, d, 'dark')` for the description's dark window

</code_context>

<specifics>
## Specific Ideas

No additional UI/UX references — this phase is a management command with no
template/UI changes. The three sample lines from
`docs/design/telescope_runs_calendar.rst` remain the concrete fixtures:
- `NTT EFOSC2 allocation 9-13 July` -> 5 nightly events, title `"NTT EFOSC2"`
- `Magellan IMACS 13-19 July (proposed)` -> per Phase 2's CONTEXT.md D-01,
  bare `'Magellan'` raises `ValueError` at parse time; per D-02 above this
  line is logged and skipped, not a Phase 3 ingest fixture (consistent with
  ROADMAP.md's Phase 3 success-criteria note)
- `Magellan Proto-Lightspeed Jul 8-12 (proposed)` -> same ambiguous-telescope
  skip

</specifics>

<deferred>
## Deferred Ideas

- `'cancelled'`-status lines deleting/striking-through previously-created
  events — informational-only for now (D-07); revisit if real schedule data
  shows cancellations needing calendar cleanup.
- Resolving Magellan-Clay vs Magellan-Baade ambiguity (data-driven
  `Observatory.short_name` lookup or richer input format) — carried forward
  from Phase 2's deferred ideas; still out of scope for Phase 3.

None otherwise — discussion stayed within phase scope.

</deferred>

---

*Phase: 3-Classical Calendar Ingest*
*Context gathered: 2026-06-13*
