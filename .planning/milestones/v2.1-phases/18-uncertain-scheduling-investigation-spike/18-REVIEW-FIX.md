---
phase: 18-uncertain-scheduling-investigation-spike
fixed_at: 2026-07-09T11:05:00Z
review_path: .planning/phases/18-uncertain-scheduling-investigation-spike/18-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 18: Code Review Fix Report

**Fixed at:** 2026-07-09T11:05:00Z
**Source review:** .planning/phases/18-uncertain-scheduling-investigation-spike/18-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 2 (WR-01, WR-02 — critical_warning scope; IN-01 excluded as info-level, out of scope)
- Fixed: 2
- Skipped: 0

## Fixed Issues

### WR-01: Inline-literal markup uses single backticks in the "Key finding" section, inconsistent with the rest of the document and with house style

**Files modified:** `docs/design/uncertain_scheduling_spike.rst`
**Commit:** b3d7c75
**Applied fix:** Converted all five single-backtick spans in the "Key finding" section
(lines 44, 46, 50, 52, 53) to double-backtick RST inline-literal markup, matching the
double-backtick convention used everywhere else in the document (e.g. the "Decisions"
table and the sibling `eso_feasibility_spike.rst`). Verified with a docutils RST parse
pass (no warnings/errors) in addition to re-reading the modified section.

### WR-02: "Decisions" table omits the actual finding for half of the SCHED-01 criterion-5 question it poses

**Files modified:** `docs/design/uncertain_scheduling_spike.rst`
**Commit:** ea6ce06
**Applied fix:** Added a sentence to the `resolve_site()` / obscode-widening row of the
"Decisions" table stating the actual, previously-omitted finding: `resolve_site()`
cannot currently resolve any of the three real space-observatory MPC codes
(`250`/`274`/`289`) because `MPCObscodeFetcher.to_observatory()` raises an unguarded
`TypeError` on the MPC API's `null` longitude for satellite-type records. The addition
points to the "Future scope" section for the fuller writeup, closing the gap where the
table previously answered only the obscode-widening half of the criterion-5 question and
left a future reader trusting the durable-summary framing to wrongly conclude
`resolve_site()` already works for these codes. Verified with a docutils RST parse pass
(no warnings/errors) in addition to re-reading the modified table row.

## Skipped Issues

None — both in-scope findings were fixed. IN-01 was excluded per fix_scope
(`critical_warning`) and left for a future info-level pass if desired.

---

_Fixed: 2026-07-09T11:05:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
