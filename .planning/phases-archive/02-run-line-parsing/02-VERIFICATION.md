---
phase: 02-run-line-parsing
verified: 2026-06-13T00:00:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
deferred:
  - truth: "parse_run_line() correctly handles arbitrary cross-month date ranges (e.g. '28 July-2 August') by reporting consistent month/day2 fields, and rejects trailing unrecognized text after the date range (D-06/D-07)"
    addressed_in: "Phase 3"
    evidence: "Phase 3 (Classical Ingest, INGEST-01) is the first consumer of ParsedRun.month/day1/day2 for arbitrary input lines beyond the 3 sample fixtures; CR-01 and CR-02 from 02-REVIEW.md must be resolved before Phase 3 expands ranges into nightly CalendarEvents, since Phase 3 would otherwise compute backwards/invalid date ranges silently for any non-Dec/Jan cross-month line."
---

# Phase 2: Run Line Parsing Verification Report

**Phase Goal:** A run-line parser turns free-text classical-schedule lines (the three sample formats) into structured tuples ready for calendar-event expansion.
**Verified:** 2026-06-13T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Parsing `'NTT EFOSC2 allocation 9-13 July'` and `'Magellan IMACS 13-19 July (proposed)'` (month-before-range) yields correct tuple / documented error | ✓ VERIFIED | Live call returns `ParsedRun(telescope='NTT', instrument='EFOSC2', status='allocation', year=2026, month=7, day1=9, day2=13)`. `'Magellan IMACS 13-19 July (proposed)'` raises `ValueError: Ambiguous telescope 'Magellan': matches multiple SITES keys ['Magellan-Clay', 'Magellan-Baade']...` — matches the design-doc-documented ambiguous-Magellan error path (D-01). Covered by `test_parse_run_line_ntt_efosc2_allocation` and `test_parse_run_line_ambiguous_magellan_imacs`. |
| 2 | Parsing `'Magellan Proto-Lightspeed Jul 8-12 (proposed)'` (month-after-range, hyphenated instrument) yields `instrument='Proto-Lightspeed'` and correct date fields | ✓ VERIFIED | `'Magellan Proto-Lightspeed Jul 8-12 (proposed)'` raises the same ambiguous-Magellan ValueError (D-01, expected per design doc). Substituting an unambiguous telescope (`'NTT Proto-Lightspeed Jul 8-12 (proposed)'`) returns `ParsedRun(telescope='NTT', instrument='Proto-Lightspeed', status='proposed', year=2026, month=7, day1=8, day2=12)` — hyphenated instrument parsed as a single token, month-after-range ordering correct. Covered by `test_parse_run_line_ambiguous_magellan_proto_lightspeed` and `test_parse_run_line_proto_lightspeed_hyphenated_instrument`. |
| 3 | A run line with no year present defaults `year` to the current year | ✓ VERIFIED | `parse_run_line('FTS Spectral confirmed 5-7 Jan')` returns `year=2026` (== `date.today().year`). Covered by `test_parse_run_line_no_year_defaults_to_current_year`. |
| 4 | A run line whose date range starts in late December produces `year = current year + 1` (roll-over) | ✓ VERIFIED | `parse_run_line('NTT EFOSC2 28 December-2 January')` returns `year=2027` (== `date.today().year + 1`), `month=12, day1=28, day2=2` — matches the exact values asserted by `test_parse_run_line_december_january_rolls_over_year`. |

**Score:** 4/4 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | General cross-month date-range handling (CR-02) and trailing-text validation after the date range (CR-01) — both from 02-REVIEW.md | Phase 3 | Phase 3 (INGEST-01) is the first consumer of `ParsedRun.month`/`day1`/`day2` for arbitrary lines beyond the 3 design-doc sample fixtures and the SC4 Dec/Jan special case; these gaps must be closed before nightly `CalendarEvent` expansion to avoid silently-wrong date ranges. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/telescope_runs.py` | `ParsedRun` dataclass, `parse_run_line()`, `KNOWN_STATUSES` constant | ✓ VERIFIED | All three present (lines 26, ~270-289, 352-430). `parse_run_line` is substantive (79 lines), handles status resolution, 3 date-range regex patterns, telescope resolution, year roll-over. |
| `solsys_code/tests/test_telescope_runs.py` | parse_run_line success-path and ValueError-path tests covering all 4 ROADMAP success criteria | ✓ VERIFIED | 10 new `test_parse_run_line_*` methods present (lines 226-303), covering SC1-4 plus D-01/D-05/D-06/D-07 error paths. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `solsys_code/telescope_runs.py:parse_run_line` | `solsys_code/telescope_runs.py:SITES` | `_resolve_telescope` prefix match against `SITES.keys()` | ✓ WIRED | `_resolve_telescope` (lines 292-314) does `token in SITES` / `key.startswith(token)` against the module-level `SITES` dict; called from `parse_run_line` at line 420. |
| `solsys_code/tests/test_telescope_runs.py` | `solsys_code/telescope_runs.py` | `from solsys_code.telescope_runs import ParsedRun, parse_run_line, ...` | ✓ WIRED | Import present at lines 9-18 of test file; all 10 new test methods call `parse_run_line(...)`. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full `solsys_code.tests.test_telescope_runs` suite passes | `python manage.py test solsys_code.tests.test_telescope_runs -v1` | "Ran 26 tests in 4.239s — OK" (16 Phase 1 + 10 Phase 2 tests) | ✓ PASS |
| SC1-4 reproduced live, outside the test harness | Python one-liner calling `parse_run_line` for all 4 success criteria + both sample-line error paths | All 4 SC outputs match expected `ParsedRun` field values and error messages exactly | ✓ PASS |
| Lint/format clean on modified files | `ruff check` + `ruff format --check` on `telescope_runs.py` and `test_telescope_runs.py` | "All checks passed!" / "2 files already formatted" | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PARSE-01 | 02-01 | Parse classical run line into `(telescope, instrument, status, year, month, day1, day2)`, handling both month-before-range and month-after-range orderings | ✓ SATISFIED | SC1/SC2 truths above; `_MONTH_BEFORE_RANGE` and `_MONTH_AFTER_RANGE` regexes both exercised and correct for the documented sample lines. |
| PARSE-02 | 02-01 | Hyphenated instrument names (e.g. `Proto-Lightspeed`) parse as a single token | ✓ SATISFIED | `test_parse_run_line_proto_lightspeed_hyphenated_instrument` + live spot-check both return `instrument='Proto-Lightspeed'`. |
| PARSE-03 | 02-01 | No-year defaults to current year; late-December start rolls to current year + 1 | ✓ SATISFIED | SC3/SC4 truths above, both verified live and by passing tests. |

No orphaned requirements found for Phase 2 in `.planning/REQUIREMENTS.md` (PARSE-01..03 all claimed by plan 02-01 and all SATISFIED).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/telescope_runs.py` | 387-391, 407-418 | CR-02 (cross-month `month2` discarded) and CR-01 (trailing text after date range not validated) from 02-REVIEW.md | ⚠️ Warning | Does not affect any of the 4 locked Phase 2 success criteria or the 3 design-doc sample lines (all pass exactly as specified, including the SC4 Dec/Jan special case). However, for arbitrary input lines beyond these fixtures — which Phase 3's `load_telescope_runs` command will need to handle per INGEST-01 — these gaps would silently produce internally-inconsistent or partially-discarded parse results. Deferred to Phase 3 (see Deferred Items above), not a Phase 2 blocker. |

No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in either modified file.

### Human Verification Required

None. All 4 success criteria and PARSE-01..03 are verifiable programmatically and were confirmed both via the existing automated test suite (26/26 passing) and live, independent re-execution of `parse_run_line()` against all 4 success-criteria inputs and the 3 design-doc sample lines.

### Gaps Summary

No gaps against the Phase 2 goal as scoped. All 4 ROADMAP success criteria pass exactly as specified, PARSE-01/02/03 are satisfied, `ParsedRun`/`parse_run_line`/`KNOWN_STATUSES` exist and are substantively implemented and wired into the test suite, and `./manage.py test solsys_code.tests.test_telescope_runs` passes 26/26.

The two Critical findings from 02-REVIEW.md (CR-01 trailing-text validation, CR-02 general cross-month month2 handling) are real correctness gaps but apply to inputs **outside** the 3 design-doc sample lines and the 4 locked success criteria (SC4's Dec/Jan cross-month case is the one cross-month scenario in scope, and it passes exactly as its test specifies). These are recorded as a deferred item to be resolved as part of, or immediately before, Phase 3 — since Phase 3 (INGEST-01) is the first consumer of `ParsedRun.month`/`day1`/`day2` for arbitrary classical-schedule lines and would otherwise silently compute wrong `CalendarEvent` date ranges for non-Dec/Jan cross-month runs.

---

_Verified: 2026-06-13T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
