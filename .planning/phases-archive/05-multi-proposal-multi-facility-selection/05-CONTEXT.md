# Phase 5: Multi-Proposal & Multi-Facility Selection - Context

**Gathered:** 2026-06-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Generalize the existing `sync_lco_observation_calendar` management command so
that a single invocation can sync any combination of proposals (or the whole
LCO-family network) across **both** `ObservationRecord(facility='LCO')` and
`ObservationRecord(facility='SOAR')` rows, with each record processed using
the facility instance/credentials matching its own `facility` value — never
a single shared facility instance reused across both. No new Django models
or migrations; no instrument-extraction or telescope-label changes (that's
Phase 6/7) — this phase is query/selection/dispatch scope only.

</domain>

<decisions>
## Implementation Decisions

### Multi-proposal / ALL syntax
- **D-01:** `--proposal` accepts a comma-separated list (e.g.
  `--proposal A,B,C`). Each segment is whitespace-stripped but **not**
  case-normalized — proposal codes are case-sensitive (e.g.
  `LTP2025A-004`), so `' B '` becomes `'B'` but casing is preserved as
  typed.
- **D-02:** The special `ALL` token is matched **case-insensitively**
  (`all`/`All`/`ALL` all trigger sync-everything) — no real proposal code
  could collide with the word "all", and case-insensitivity is friendlier
  for command-line operators.
- **D-03:** Duplicate or empty entries in the list (`--proposal A,A,B` or a
  trailing comma `--proposal A,B,`) are **silently deduped and dropped** —
  split, strip, drop empty strings, dedupe — rather than erroring out.

### SOAR facility settings & credentials
- **D-04:** This phase adds a `FACILITIES['SOAR']` entry to
  `src/fomo/settings.py` as part of the code change — SOAR sync cannot
  actually work end-to-end without it (today `SOARSettings` silently
  resolves to blank `portal_url`/`api_key` since the key is absent), and
  SELECT-04's "both facilities synced in one run" success criterion
  implicitly requires real, resolvable SOAR credentials.
- **D-05:** `FACILITIES['SOAR']` reuses the **same** `api_key`/`portal_url`
  env var as `FACILITIES['LCO']` (e.g. both read `os.getenv('LCO_API_KEY')`)
  rather than introducing a separate `SOAR_API_KEY` — matches
  `SOARFacility`'s documented behavior that it authenticates against the
  same LCO Observation Portal API, not a distinct one.

### Per-record facility dispatch & defensive behavior
- **D-06:** Replace the current single shared `LCOFacility()` instance
  (created once at `sync_lco_observation_calendar.py:187`, reused for every
  record regardless of that record's own `facility` field — the exact bug
  SELECT-05 targets) with a dispatch dict
  `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` built **once, eagerly**,
  before the queryset loop, then looked up per record via `record.facility`.
  Build both instances unconditionally regardless of which facilities
  actually appear in a given run's records — simplest, mirrors today's
  single-eager-instance pattern just extended to two keys.
- **D-07:** If a record's `facility` value is something other than `'LCO'`
  or `'SOAR'` (shouldn't occur since the queryset filters on exactly those
  two, but defensively): **skip + log, continue the run** — same per-record
  error-handling convention already used elsewhere in this command and
  established in Phase 3's D-02. A single unexpected row never aborts the
  whole sync.

### Run summary reporting
- **D-08:** The end-of-run summary line reports a **per-facility
  breakdown** (e.g. `LCO: 3 created, 1 updated, 0 skipped | SOAR: 2 created,
  0 updated, 1 skipped`), not just aggregate counts — lets an operator see
  at a glance whether one facility (e.g. SOAR, if credentials are
  misconfigured) had problems, without re-running filtered to a single
  facility.

### Claude's Discretion
- Exact stdout formatting/line layout of the per-facility summary (D-08),
  as long as both facilities' created/updated/unchanged/skipped counts are
  each individually visible.
- Exact log message wording for the D-07 skip-on-unexpected-facility case
  and the D-04 settings-key addition's surrounding comments/docstring.
- Whether the proposal-list parsing (D-01..D-03) is implemented as a small
  helper function or inlined in `add_arguments`/`handle()` — either is fine
  as long as the behavior matches D-01..D-03.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `.planning/REQUIREMENTS.md` — SELECT-02, SELECT-03, SELECT-04, SELECT-05
  (this phase's locked acceptance criteria)
- `.planning/milestones/v1.3-ROADMAP.md` §"Phase 5: Multi-Proposal &
  Multi-Facility Selection" — success criteria 1-4 and dependency on
  Phase 4
- `.planning/PROJECT.md` — "Current Milestone: v1.3 Full LCO Facility Sync"
  section (target features, real-data bug context that drives this
  milestone)
- `.planning/STATE.md` — "Blockers/Concerns" notes a research gap on
  whether `FACILITIES['SOAR']` needs an explicit settings.py entry; D-04/
  D-05 above resolve that gap for this phase
- `docs/design/telescope_runs_calendar.rst` — original Stage 3/4 design
  notes (background only; superseded in scope by v1.3 REQUIREMENTS.md)

### Existing code (Phase 4)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — the
  command this phase modifies. Specifically: `--proposal` arg definition
  (single required str, no comma/ALL handling today), the queryset filter
  `ObservationRecord.objects.filter(facility='LCO',
  parameters__proposal=proposal)` (must become multi-proposal/multi-facility
  aware), and the single `facility = LCOFacility()` instantiation reused via
  `_failure_prefix()`, `_title_for()`, `_build_event_fields()` (the D-06
  anti-pattern to fix)
- `solsys_code/tests/test_sync_lco_observation_calendar.py` — existing test
  conventions: `_create_record()` helper (hardcodes `facility='LCO'`),
  `_parameters()` helper for fixture `parameters` dicts, no mocking of
  `LCOFacility()` (tests call the real instance directly since
  `get_observation_url()` etc. are pure/local). Extend this pattern for new
  SOAR-facility fixtures and comma-list/ALL test cases.
- `.planning/phases/04-lco-queue-sync-command/04-CONTEXT.md` — Phase 4's
  decisions this phase builds on (D-01 `get_observation_url()` usage,
  no-churn upsert pattern, per-record catch-log-continue error handling,
  stdout summary reporting convention)

### Third-party library (not in this repo)
- `tom_observations.facilities.soar.SOARFacility` — `SOARFacility(LCOFacility)`,
  a direct subclass of `LCOFacility` (itself `OCSFacility`); inherits
  `get_observation_url()`, `get_terminal_observing_states()`,
  `get_failed_observing_states()` unchanged from `OCSFacility`, so existing
  helper code works unmodified against a `SOARFacility` instance.
  `SOARFacility.name = 'SOAR'` (matches `ObservationRecord.facility` exactly).
  Defaults to `SOARSettings('SOAR')`, which resolves settings via
  `settings.FACILITIES['SOAR']` — confirms D-04/D-05's settings-key
  requirement.
- `src/fomo/settings.py` `FACILITIES` dict (currently has `'LCO'` and `'GEM'`
  entries only, no `'SOAR'`) — D-04 adds the missing entry here.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — all
  existing helpers (`_failure_prefix`, `_title_for`, `_build_event_fields`,
  the no-churn upsert, the per-record error handling) already accept a
  `facility` instance as a parameter and call only `OCSFacility`-level
  methods on it — they need **no signature changes**, just a different
  instance passed in per record (D-06).
- `solsys_code/tests/test_sync_lco_observation_calendar.py:_create_record()`
  / `_parameters()` — fixture helpers to extend for SOAR-facility and
  multi-proposal test cases. Per CLAUDE.md convention, any new `Target`
  fixtures must use `NonSiderealTargetFactory`, never `SiderealTargetFactory`.

### Established Patterns
- No-churn idempotency: only call `.save()` when fields actually changed
  (Phase 3/4 precedent, restated in STATE.md's Phase 4 technical notes)
- Per-record error handling: catch, log (with line/record identifying
  info), increment a counter, continue — never abort the whole run on one
  bad record (D-07 follows this exactly)
- DB-dependent tests live in `solsys_code/tests/`, run via
  `./manage.py test solsys_code`

### Integration Points
- `ObservationRecord.objects.filter(facility__in=['LCO', 'SOAR'])` combined
  with a proposal filter (`parameters__proposal__in=[...]` when a list is
  given, or no proposal filter at all when `ALL` is given) replaces today's
  single-facility/single-proposal filter
- The `{'LCO': LCOFacility(), 'SOAR': SOARFacility()}` dispatch dict (D-06)
  is the single new integration point threading through the existing
  per-record helper calls

</code_context>

<specifics>
## Specific Ideas

No additional UI/UX references — this phase is a management-command change
only, no template/UI changes. No specific sample SOAR `ObservationRecord`
fixtures were provided during discussion; the planner/researcher should
construct representative SOAR-facility test fixtures (e.g. via the existing
`_create_record()`/`_parameters()` helpers with `facility='SOAR'`) to
validate SELECT-04/05.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Instrument-type extraction
correctness (Phase 6) and telescope-label resolution/fallback (Phase 7) were
not raised as in-scope here; they're already sequenced as later phases per
PROJECT.md's Key Decisions ("Phase ordering follows research's dependency
chain").

</deferred>

---

*Phase: 5-Multi-Proposal & Multi-Facility Selection*
*Context gathered: 2026-06-19*
