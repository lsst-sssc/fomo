---
created: 2026-10-02T16:36:10.888Z
title: "load_telescope_runs: skip comment lines and warn on a bare proposal token"
area: telescope-runs
severity: minor
files:
  - solsys_code/management/commands/load_telescope_runs.py:242
  - solsys_code/telescope_runs.py:401
  - solsys_code/telescope_runs.py:455
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
---

## Problem

Found 2026-10-02 during the v2.4 intent-review walkthrough (setup step 4, the first
`load_telescope_runs` run on the live DB). Two usability gaps in the schedule-file format,
neither of which is a parser bug — the file was wrong — but both of which left the operator
guessing at the fix:

1. **No comment syntax.** The loader parses every non-blank line
   (`load_telescope_runs.py:242` skips only blank lines). A file with `# …` comment lines
   produces one `Could not find a date range …` warning per comment line, and a free-text
   block (the Gemini queue program, which legitimately can't be a schedule line) produced
   four more. All skipped harmlessly, but noisy, and the natural way to annotate a hand-kept
   file is a `#` comment.

2. **A bare proposal ID is silently mis-tokenised.** The grammar is
   `telescope instrument [status] daterange`, with the proposal as an optional bracketed
   `[…]` token anywhere on the line. `117.2A2N.001 NTT EFOSC2 allocation 9-13 July` therefore
   reads the ID as the telescope, `NTT` as the instrument and `EFOSC2` as the status, and
   fails with `Unrecognized status 'EFOSC2'; known statuses are […]` — a message that points
   at the wrong token and never mentions brackets. The correct line is
   `NTT EFOSC2 allocation 9-13 July [117.2A2N.001]`.

Dry-run output that exposed both (operator's `Didymos_runs` file, 2026-10-02):

    Line 1: Unrecognized status 'EFOSC2' in '117.2A2N.001 NTT EFOSC2 allocation 9-13 July'
    Line 5: Could not find a date range (e.g. "9-13 July" or "Jul 8-12") in '# Gemini (…'
    Line 9: Unrecognized status 'hours' in 'GMOS-S  6.50 hours  13-16 July'

## Solution

Small, contained — a `/gsd-quick`, post-v2.4:

1. `load_telescope_runs` skips lines whose first non-blank character is `#` (count them
   separately in the summary line, e.g. `comments: N`, so they are not reported as skipped
   parse failures).
2. `telescope_runs.parse_run_line()` (or `_resolve_status()`, `:455`) detects a leading
   token that looks like a proposal code — digits/dots/dashes, e.g. ESO `117.2A2N.001`,
   LCO `KEY2026B-004`, Gemini `GS-2026A-FT-115` — and raises a `ValueError` saying it looks
   like a proposal code and must be bracketed, before the status check runs.
3. Paired docs per CLAUDE.md: a `load_telescope_runs_demo.ipynb` cell exercising both
   behaviours with real output, and the schedule-file format section of
   `docs/runbooks/telescope_runs_calendar.rst` (state that `#` lines are comments; show the
   bare-ID error and its fix).
