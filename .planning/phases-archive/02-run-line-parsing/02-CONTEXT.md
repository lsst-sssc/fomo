# Phase 2: Run Line Parsing - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

A pure, single-line parser that turns a free-text classical-schedule run line
(e.g. `NTT EFOSC2 allocation 9-13 July`, `Magellan IMACS 13-19 July (proposed)`,
`Magellan Proto-Lightspeed Jul 8-12 (proposed)`) into a structured representation
with telescope, instrument, status, year, month, day1, day2. This is the input
contract Phase 3 (Classical Calendar Ingest) consumes — no `CalendarEvent`
creation, no management command, and no `tom_calendar`/Django model work happens
in this phase.

</domain>

<decisions>
## Implementation Decisions

### Telescope name resolution
- **D-01:** The parser resolves the line's telescope token against
  `telescope_runs.SITES` keys (`'Magellan-Clay'`, `'Magellan-Baade'`, `'NTT'`,
  `'FTS'`) by **prefix match**.
  - Exact match (e.g. `'NTT'` -> `'NTT'`, `'FTS'` -> `'FTS'`) resolves directly.
  - A prefix match against multiple SITES keys (e.g. `'Magellan'` matches both
    `'Magellan-Clay'` and `'Magellan-Baade'`) raises `ValueError` listing the
    candidate keys — this is a deliberate ambiguity the parser surfaces rather
    than silently guessing.
  - No match at all raises `ValueError`.
  - Distinguishing Baade vs Clay for a bare `'Magellan'` line remains
    out of scope (per the design doc) — it surfaces as a `ValueError` for now,
    not a silent default.

### Output shape & module location
- **D-02:** Add `parse_run_line()` to the existing `solsys_code/telescope_runs.py`
  (keeps Stage 1 + Stage 2 helpers together; Phase 1 already established this
  module and its `SITES` registry).
- **D-03:** Return value is a small `ParsedRun` dataclass with named fields:
  `telescope` (resolved SITES key, str), `instrument` (str), `status` (str),
  `year` (int), `month` (int), `day1` (int), `day2` (int) — named fields for
  readability in Phase 3, rather than a positional tuple.

### Status vocabulary
- **D-04:** Maintain a small, extendable known-status set (e.g. `'allocation'`,
  `'proposed'`, `'confirmed'`, `'cancelled'`), matched **case-insensitively**,
  whether the status appears as a bare word or inside parentheses, and allowing
  multi-word phrases (e.g. `'(not confirmed)'`).
- **D-05:** If no status word/phrase is present in the line, `status` defaults
  to `'allocation'`.
- **D-06:** If a status word/phrase IS present but does not match the known set,
  raise `ValueError` (forces the known-status set to be kept up to date rather
  than silently passing through unrecognized text).

### Malformed-line & multi-line handling
- **D-07:** `parse_run_line(line: str) -> ParsedRun` operates on a **single
  line**. On anything unparseable (bad/missing date range, unresolved
  telescope, unrecognized status, etc.), it raises `ValueError` with a message
  describing what failed. Phase 3's management command is responsible for
  iterating over a file/block of lines and catching/logging per-line errors —
  this phase does not implement multi-line or file handling.

### Claude's Discretion
- Exact regex/parsing strategy for the two date-range orderings
  (`9-13 July` vs `Jul 8-12`), month-name parsing (full names vs abbreviations,
  case), and hyphenated-instrument handling (`Proto-Lightspeed`) are
  implementation details for planning/execution — the design doc's three
  sample lines are the acceptance fixtures (see Phase 2 success criteria in
  ROADMAP.md).
- Exact wording of `ValueError` messages.
- Whether `ParsedRun` is a `@dataclass` or `@dataclass(frozen=True)` /
  `NamedTuple`-with-defaults variant — any named-field structure satisfying
  D-03 is acceptable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/telescope_runs_calendar.rst` — full feasibility study; "Classical
  Run Input Format" section has the three sample lines, parsing rules, and the
  night convention; "Implementation Plan" / "Success Criteria" sections define
  Stage 2 (= Phase 2 + Phase 3) acceptance criteria
- `.planning/REQUIREMENTS.md` — PARSE-01..03 (this phase), INGEST-01..03 (Phase 3)
- `.planning/ROADMAP.md` — Phase 2 success criteria (4 criteria) and Phase 3
  dependency on this phase's output

### Existing code (Stage 1, Phase 1)
- `solsys_code/telescope_runs.py` — `SITES` registry (lines ~15-21), `get_site()`,
  `sun_event()`; `parse_run_line()` is added alongside these
- `solsys_code/tests/test_telescope_runs.py` — existing Phase 1 test suite and
  conventions for testing this module (Django `TestCase`, `solsys_code/tests/`)

No external specs beyond the above — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/telescope_runs.py:SITES` — dict of telescope name -> MPC obscode;
  the new parser's telescope-resolution step (D-01) matches against `SITES.keys()`
- `solsys_code/views.py:_translate_constraints` — existing precedent in this
  codebase for parsing free-text constraint strings into structured tokens with
  `ValueError` on invalid input; similar error-handling style to follow for
  `parse_run_line()`

### Established Patterns
- Google-style docstrings with `Args`/`Returns`/`Raises` (see `sun_event()`,
  `horizon_dip()` in `telescope_runs.py` for the house style)
- `ValueError` (not custom exceptions) for invariant violations, per
  `.planning/codebase/CONVENTIONS.md`
- Tests for `telescope_runs.py` live in `solsys_code/tests/test_telescope_runs.py`
  (Django `TestCase`, run via `./manage.py test solsys_code`) — even though
  `parse_run_line()` itself is pure-Python with no DB access, keep it in the same
  test module/suite as the rest of `telescope_runs.py` for consistency. A pure
  unit test under `tests/` (pytest) would also be acceptable if it's simpler to
  isolate.

### Integration Points
- `ParsedRun.telescope` (a resolved `SITES` key) feeds directly into Phase 3's
  `get_site(parsed.telescope)` call

</code_context>

<specifics>
## Specific Ideas

No additional specific UI/UX references — this phase is a pure backend parser.
The three sample lines from `docs/design/telescope_runs_calendar.rst` are the
concrete acceptance fixtures:
- `NTT EFOSC2 allocation 9-13 July` -> telescope='NTT', instrument='EFOSC2',
  status='allocation', month=7, day1=9, day2=13
- `Magellan IMACS 13-19 July (proposed)` -> telescope resolution raises
  `ValueError` (ambiguous Magellan-Clay/Magellan-Baade) per D-01 — note this
  sample line is therefore an error-path fixture for this phase, not a
  success-path one
- `Magellan Proto-Lightspeed Jul 8-12 (proposed)` -> same ambiguous-telescope
  `ValueError`; instrument='Proto-Lightspeed', status='proposed', month=7,
  day1=8, day2=12 should still be the values reported in the error/exception
  context if feasible

</specifics>

<deferred>
## Deferred Ideas

- Resolving the Magellan-Clay vs Magellan-Baade ambiguity (e.g. via a richer
  input format that names the specific Magellan instrument/telescope, or a
  data-driven `Observatory.short_name` lookup) — out of scope per the design
  doc's "Open Items"; revisit if Phase 3 testing against the real sample lines
  proves too many real-world lines hit this `ValueError`.

None — discussion otherwise stayed within phase scope.

</deferred>

---

*Phase: 2-Run Line Parsing*
*Context gathered: 2026-06-13*
