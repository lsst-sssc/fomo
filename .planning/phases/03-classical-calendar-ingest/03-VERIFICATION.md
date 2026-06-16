---
phase: 03-classical-calendar-ingest
verified: 2026-06-16T14:30:00Z
status: human_needed
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
deferred:
human_verification:
  - test: "Execute the demo notebook end-to-end and confirm CalendarEvent rows display with title, start_time, end_time, and description"
    expected: "Notebook runs without error; 8 CalendarEvent rows are displayed with non-empty descriptions containing dark-window times, Status, and Source line text"
    why_human: "Notebook output cells are cleared by the pre-commit hook (exclude pattern ^docs/pre_executed does not match docs/notebooks/pre_executed/) so committed output cells cannot be read programmatically. The notebook executes cleanly per 03-02-SUMMARY, but the verifier cannot confirm live output without running it."
---

# Phase 03: Classical Calendar Ingest — Verification Report

**Phase Goal:** Running `load_telescope_runs` against a file of classical run lines populates the calendar with one accurate, idempotent `CalendarEvent` per observing night for each parsed run.
**Verified:** 2026-06-16T14:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Ingesting `NTT EFOSC2 allocation 9-13 July` creates exactly 5 CalendarEvents (E - S + 1) | VERIFIED | `test_creates_one_event_per_night` passes; `CalendarEvent.objects.count() == 5` confirmed live |
| 2 | Each event has `start_time` = dip-corrected sunset, `end_time` = next-morning sunrise, `end_time > start_time`, duration 8-15 h | VERIFIED | `test_event_durations_within_range` passes; `end_time > start_time` and 8-15 h duration asserted |
| 3 | Each event's `telescope`/`instrument`/`title` set from parsed run; `description` contains dark-window times, status, and original run-line text | VERIFIED | `test_event_fields_set_from_parsed_run` passes; `telescope='NTT'`, `instrument='EFOSC2'`, `title='NTT EFOSC2'`; description contains ISO datetime string, `allocation`, and `NTT EFOSC2 allocation 9-13 July` |
| 4 | Running the command twice on the same file creates no duplicates and reports `updated: 0` on the second run | VERIFIED | `test_idempotent_rerun_no_duplicates` and `test_unchanged_rerun_does_not_update_existing_rows` both pass; `get_or_create` with conditional save confirmed in source; second-run summary asserts `updated: 0` |
| 5 | A line raising ValueError is logged to stderr with line number and original text; processing continues | VERIFIED | `test_unparseable_line_logged_and_skipped` passes; stderr contains `'2'` and `'Magellan IMACS 13-19 July (proposed)'`; NTT events still created |

**Score:** 5/5 truths verified

### Demo Notebook — Definition of Done

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | Demo notebook shows load_telescope_runs ingesting a sample schedule file and resulting CalendarEvents, with executed output cells | UNCERTAIN — NEEDS HUMAN | Notebook file exists at `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`. References `load_telescope_runs` 8 times. Does NOT import `ephem_utils` or `solsys_code.views`. However, the pre-commit hook (`exclude: ^docs/pre_executed`) clears all `.ipynb` outputs before commit; 0 output cells are present in the committed file. This matches Phase 01's demo notebook (also 0 output cells). Manual execution is required to confirm the notebook runs and displays CalendarEvent rows. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/management/commands/load_telescope_runs.py` | load_telescope_runs BaseCommand + `_iter_run_nights` helper | VERIFIED | File exists, 119 lines. Contains `class Command(BaseCommand)`, `_iter_run_nights(parsed)` using `timedelta` arithmetic, `add_arguments` with positional `filepath`, `handle` with per-line try/except, `get_or_create` + conditional save. No `update_or_create`. |
| `solsys_code/tests/test_load_telescope_runs.py` | Django TestCase with 6 named test methods | VERIFIED | File exists, 159 lines. Contains `class TestLoadTelescopeRuns(TestCase)` with all 6 exactly-named test methods. `setUpTestData` seeds all 4 Observatory rows (268, 269, 809, E10). |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Executed demo notebook | UNCERTAIN | File exists. Contains 6 code cells and 5 markdown cells. References `load_telescope_runs`. No output cells due to pre-commit hook — requires manual execution to confirm. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `load_telescope_runs.py` | `solsys_code.telescope_runs.parse_run_line` | per-line parse call | VERIFIED | `from solsys_code.telescope_runs import ParsedRun, get_site, parse_run_line, sun_event` at line 9; `parse_run_line(line)` called at line 64 |
| `load_telescope_runs.py` | `solsys_code.telescope_runs.sun_event` | per-night sun/dark event lookup | VERIFIED | `sun_event(site, d, 'sun')` at line 68; `sun_event(site, d, 'dark')` at line 69 |
| `load_telescope_runs.py` | `tom_calendar.models.CalendarEvent` | idempotent get_or_create upsert | VERIFIED | `CalendarEvent.objects.get_or_create(telescope=..., instrument=..., start_time=..., defaults={...})` at lines 82-91. `update_or_create` is absent (grep confirmed 0 occurrences). |
| `load_telescope_runs_demo.ipynb` | `load_telescope_runs` command | `call_command('load_telescope_runs', ...)` | VERIFIED | String `load_telescope_runs` appears 8 times in notebook JSON |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INGEST-01 | 03-01-PLAN.md | `load_telescope_runs` expands parsed run S..E into E - S + 1 nightly CalendarEvents (sunset/sunrise times) | SATISFIED | 5 events created for 9-13 July (5 nights); durations 8-15 h; `_iter_run_nights` uses `timedelta`; tests pass |
| INGEST-02 | 03-01-PLAN.md | Each event sets `telescope`/`instrument` from parsed line, glanceable `title`, `description` with dark-window times and original run-line text | SATISFIED | `title = f'{telescope} {instrument}'` (D-05); description format confirmed: dark-window ISO datetimes, `Status:`, `Source line:`; test asserts all three pieces |
| INGEST-03 | 03-01-PLAN.md | Running command twice on same file produces no duplicate CalendarEvents | SATISFIED | `get_or_create` keyed on `(telescope, instrument, start_time)`; second run leaves count at 5, reports `updated: 0`; `modified` timestamps unchanged |

No orphaned requirements: INGEST-01, INGEST-02, INGEST-03 are the only Phase 3 requirements in REQUIREMENTS.md; all are claimed in 03-01-PLAN.md and verified above.

PARSE-01/02/03 are Phase 2 requirements (already complete); they are not Phase 3 scope.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No TBD/FIXME/XXX/PLACEHOLDER markers found in either new file. No `update_or_create` (grep confirmed 0). No empty returns or stubs. |

Ruff check and format check both pass clean on `load_telescope_runs.py` and `test_load_telescope_runs.py`.

Pre-existing warnings in `solsys_code/tests/test_views.py` (SIM905) and `solsys_code/views.py` (UP045) are documented in 03-01-SUMMARY as out-of-scope (files not modified by this phase).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 6 tests exist and pass | `python manage.py test solsys_code.tests.test_load_telescope_runs -v 2` | Ran 6 tests in 46.5s — OK | PASS |
| Command uses `get_or_create` not `update_or_create` | `grep -c 'update_or_create' solsys_code/management/commands/load_telescope_runs.py` | 0 | PASS |
| Catches both ValueError and Observatory.DoesNotExist | `grep 'except.*ValueError.*Observatory.DoesNotExist' load_telescope_runs.py` | Line 106: `except (ValueError, Observatory.DoesNotExist) as exc:` | PASS |
| Notebook references command | `grep -c 'load_telescope_runs' load_telescope_runs_demo.ipynb` | 8 matches | PASS |
| Notebook avoids ephem_utils import | `grep 'ephem_utils\|solsys_code.views' load_telescope_runs_demo.ipynb` | 0 matches | PASS |

### Human Verification Required

#### 1. Demo Notebook Execution

**Test:** From the repo root, run `jupyter nbconvert --to notebook --execute docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb --output /tmp/demo_out.ipynb` and inspect `/tmp/demo_out.ipynb` output cells.

**Expected:** Notebook completes without error. Output cells show: (a) Observatory seeding confirmation, (b) command summary with `created: 8, skipped: 1`, (c) 8 CalendarEvent rows with non-empty `title`, `start_time`, `end_time`, and `description` containing dark-window ISO datetimes and source line text, (d) idempotency re-run showing `updated: 0, unchanged: 8`.

**Why human:** The pre-commit hook (`exclude: ^docs/pre_executed`) clears all notebook outputs before commit — the committed notebook has 0 output cells. The SUMMARY claims execution succeeds and 03-02-SUMMARY says `jupyter nbconvert --to notebook --execute` exits 0, but the verifier cannot confirm this without running the notebook. The plan's acceptance criterion ("executed with committed output cells") is structurally unmet by this project's convention, but the notebook may still be functionally correct.

### Gaps Summary

No blocking gaps. All 5 core observable truths are verified by passing tests.

The only unresolved item is the demo notebook's committed output state — a consequence of the project's pre-commit hook configuration that clears all notebook outputs regardless of directory. The SUMMARY documents this as a known deviation consistent with the Phase 01 demo notebook behavior. The notebook's functional correctness (does it execute? does it display CalendarEvents?) requires human confirmation.

---

_Verified: 2026-06-16T14:30:00Z_
_Verifier: Claude (gsd-verifier)_
