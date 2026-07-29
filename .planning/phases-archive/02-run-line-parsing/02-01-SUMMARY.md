---
phase: 02-run-line-parsing
plan: 01
subsystem: telescope-runs-parser
tags: [parsing, regex, telescope-runs, classical-schedule]
dependency_graph:
  requires: []
  provides:
    - solsys_code/telescope_runs.py:ParsedRun
    - solsys_code/telescope_runs.py:parse_run_line
    - solsys_code/telescope_runs.py:KNOWN_STATUSES
  affects:
    - phase-03-classical-calendar-ingest
tech_stack:
  added: []
  patterns:
    - "re.compile(..., re.VERBOSE | re.IGNORECASE) with named groups for date-range parsing (precedent: views.py:_translate_constraints)"
    - "ValueError with f-string {value!r} messages, no custom exceptions (house style)"
key_files:
  created: []
  modified:
    - solsys_code/telescope_runs.py
    - solsys_code/tests/test_telescope_runs.py
decisions:
  - "Telescope resolution by prefix match against SITES.keys(); exact match wins, 2+ matches raise ValueError listing all candidates (D-01)"
  - "Status detection: parenthesized phrase or bare word/phrase matched case-insensitively against KNOWN_STATUSES; any unmatched leftover word between instrument and date range raises ValueError (D-06)"
  - "Date range: three regex patterns tried in order (month-after-range, cross-month, month-before-range) to cover '9-13 July', 'Jul 8-12', and '28 December-2 January'"
  - "Year defaults to current year; rolls to current year + 1 only when month==12 and day2 < day1 (December-start, January-end crossing)"
metrics:
  duration: "~35 min"
  completed: 2026-06-13
---

# Phase 2 Plan 01: Run Line Parsing Summary

Added a pure-Python `parse_run_line()` function and `ParsedRun` dataclass to
`solsys_code/telescope_runs.py` that turns free-text classical-schedule run
lines (telescope, instrument, status, date range) into structured fields,
using prefix-match telescope resolution against `SITES` and three
case-insensitive regex patterns to cover both date-range orderings plus
cross-month ranges.

## What Was Built

- `KNOWN_STATUSES` — module-level set of recognized status words/phrases
  (`'allocation'`, `'proposed'`, `'confirmed'`, `'cancelled'`, `'not
  confirmed'`), matched case-insensitively.
- `ParsedRun` — frozen `@dataclass` with `telescope`, `instrument`, `status`,
  `year`, `month`, `day1`, `day2` fields (Google-style docstring documenting
  each field's meaning, including the year roll-over rule).
- `parse_run_line(line: str) -> ParsedRun` — parses a single run line:
  - Resolves status first (parenthesized phrase or bare word/phrase),
    removing it from the working text.
  - Tries three date-range regexes in order: month-after-range (`Jul 8-12`),
    cross-month (`28 December-2 January`), month-before-range (`9-13 July`).
  - Computes year (current year, +1 for December-start/January-end crossings).
  - Splits the text before the date-range match into telescope + instrument
    tokens; any extra leftover word is treated as an unrecognized status and
    raises `ValueError` (D-06).
  - Resolves the telescope token against `SITES` by prefix match — exact
    match wins; 2+ matches (e.g. `'Magellan'`) raise `ValueError` listing all
    candidate `SITES` keys; 0 matches raise `ValueError`.
- 10 new test methods on `TestTelescopeRuns` covering all 4 ROADMAP Phase 2
  success criteria plus the D-01/D-05/D-06/D-07 error paths.

## Verification

- `python manage.py test solsys_code.tests.test_telescope_runs -v1` — 26/26
  tests pass (16 existing Phase 1 tests + 10 new parser tests).
- `ruff check solsys_code/telescope_runs.py solsys_code/tests/test_telescope_runs.py`
  and `ruff format --check` on both files — clean.
- Manually verified all `<behavior>` fixtures from the plan against the live
  function (NTT success, both bare-Magellan ValueError cases naming both
  `Magellan-Clay`/`Magellan-Baade`, hyphenated `Proto-Lightspeed`, no-year
  default, December->January roll-over, unknown status, unknown telescope).

## Deviations from Plan

None - plan executed exactly as written. The two-pass design (regex for
date-range location plus token-splitting for telescope/instrument/leftover
validation) was an implementation detail left to discretion per CONTEXT.md.

## Self-Check: PASSED

- FOUND: solsys_code/telescope_runs.py (contains ParsedRun, parse_run_line, KNOWN_STATUSES)
- FOUND: solsys_code/tests/test_telescope_runs.py (10 new test_parse_run_line_* methods)
- FOUND commit d00e14f (feat(02-01): add ParsedRun and parse_run_line() to telescope_runs.py)
- FOUND commit b480f07 (test(02-01): add parse_run_line tests covering Phase 2 success criteria)
