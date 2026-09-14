---
phase: quick-260913-rmd
plan: 01
subsystem: allocation-projector
tags: [django, zoneinfo, campaignrun, allocation-projector, calendar-events]

requires:
  - phase: 35 (Allocation Layer & Classical Cutover)
    provides: night_start_utc/night_end_utc sub-night fields, the ALLOC: allocation projector, the sign-of-offset date rule 35-REVIEW.md NF-03 found broken
provides:
  - "_night_span_utc(): a per-run observing-night UTC span (zoneinfo-only, sun_event()-free) replacing the sign-of-offset boolean"
  - "_time_of_day_to_datetime() rewritten to a distance-to-span candidate resolution, no hour threshold"
  - "Three-band coverage in TestSubNightWindowSiteDirection: +10 (Sydney), -4 (Chile), +2 (SAAO Sutherland), +5:30 (Hanle), -10 (FTN)"
  - "Corrected models.py night_start_utc/night_end_utc field-contract comment (three bands, not two)"
affects: [allocation_projector, campaign_reconciler, classical_cutover]

actuals:
  tokens: 5639
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Per-boundary date resolution against a computed zoneinfo UTC span, picked by distance-to-span with an inside-span short-circuit, instead of a fixed UTC-hour threshold"

key-files:
  created: []
  modified:
    - solsys_code/allocation_projector.py
    - solsys_code/models.py
    - solsys_code/tests/test_allocation_projector.py

key-decisions:
  - "_night_span_utc() computes the site's nominal local 18:00 -> +12h wall-clock instant, both converted to UTC via zoneinfo, as the one per-night span both night_bounds() and _span_needs_remint() resolve boundaries against -- never calling sun_event(), preserving D-13's astropy-free update path."
  - "_time_of_day_to_datetime() picks between the two UTC-date candidates (night, night+1) by distance to the span (zero when inside, else nearer endpoint), tie broken toward night -- replaces the hard-coded t.hour < 12 threshold entirely."
  - "Task 2's FTN 'evening-side end with a computed start' test uses time(6, 30) rather than the plan's literal time(4, 30): FTN's real sun_event()-computed sunset for 2026-07-09 is 05:17:27 UTC on 2026-07-10, after 04:30, so pairing 04:30 with a computed (null) start produces a genuinely inverted span independent of this fix -- the plan's ground-truth table verified only the zoneinfo date-resolution arithmetic for that value, not that it falls after the site's true astronomical sunset. time(6, 30) preserves every property the plan's value was chosen for (early UTC hour, resolved onto night + 1 under both the old and new rule) while sitting safely after the real sunset."

patterns-established:
  - "Per-boundary date resolution against a computed zoneinfo UTC span (distance-to-span), not a fixed UTC-hour threshold, for any future sub-night or sub-window time-of-day resolution in this codebase."

requirements-completed: [ALLOC-02, NF-03]

coverage:
  - id: D1
    description: "Africa/Johannesburg (+2, band 2-east) night_end_utc resolves to the following UTC date, producing a non-inverted span where the old sign-of-offset rule raised ValueError on every reconcile."
    requirement: NF-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection.test_saao_morning_side_end_resolves_to_the_following_utc_date"
        status: pass
    human_judgment: false
  - id: D2
    description: "Asia/Kolkata (+5:30, half-hour offset) 00:00-02:00 window resolves to the following UTC date rather than silently a day early with no error."
    requirement: NF-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection.test_hanle_half_hour_offset_window_resolves_to_the_following_utc_date"
        status: pass
    human_judgment: false
  - id: D3
    description: "Pacific/Honolulu (-10, band 3) resolves BOTH ends onto the following UTC date, including the late-hour end that lands a full day early today; the evening-side end with a computed start still resolves independently; a second reconcile makes no sun_event() call and reports unchanged with the same event primary key."
    requirement: NF-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection.test_ftn_both_ends_resolve_to_the_following_utc_date"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection.test_ftn_evening_side_end_with_computed_start_resolves_independently"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection.test_ftn_re_mint_agreement_makes_no_further_sun_event_calls"
        status: pass
    human_judgment: false
  - id: D4
    description: "Australia/Sydney and America/Santiago behaviour is byte-identical to pre-fix -- the four pre-existing TestSubNightWindowSiteDirection tests (including the inverted-span ValueError guard) pass unmodified."
    requirement: NF-03
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_allocation_projector.py#TestSubNightWindowSiteDirection (4 pre-existing tests, unmodified)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The models.py night_start_utc/night_end_utc field-contract comment and the changed allocation_projector.py docstrings describe the three-band taxonomy, not the superseded two-case rule."
    requirement: NF-03
    verification:
      - kind: other
        ref: "manual review of solsys_code/models.py and solsys_code/allocation_projector.py docstrings"
        status: pass
    human_judgment: false
  - id: D6
    description: "Paired-docs audit: docs/runbooks/telescope_runs_calendar.rst carries no statement of the superseded two-case rule (no edit needed); the two demo notebooks touching this projector (project_observation_calendar_demo.ipynb, reconcile_campaign_runs_demo.ipynb) exercise only the unaffected America/Santiago site and set no sub-night fields in any code cell (no re-execution needed)."
    verification:
      - kind: other
        ref: "grep over docs/runbooks/telescope_runs_calendar.rst; scripted cell scan over both notebooks' committed cells"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-09-14
status: complete
---

# Quick Task 260913-rmd Summary

**Replaced allocation_projector's sign-of-UTC-offset sub-night date rule with a per-boundary resolution against the site's own observing-night UTC span, closing 35-REVIEW.md NF-03 for offsets 0..+6 and at/below -6.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-09-14T03:21:40Z
- **Tasks:** 3
- **Files modified:** 3 (solsys_code/allocation_projector.py, solsys_code/models.py, solsys_code/tests/test_allocation_projector.py)

## Accomplishments

- Replaced `_site_runs_behind_utc(run, night) -> bool` with `_night_span_utc(run, night) -> tuple[datetime, datetime]`: a `zoneinfo`-only per-night UTC span (site-local 18:00 -> +12h wall clock), never calling `sun_event()`.
- Rewrote `_time_of_day_to_datetime()` to pick between the `night`/`night+1` UTC-date candidates by distance to that span (inside-span = zero distance, else nearer endpoint, tie toward `night`) -- deleting the hard-coded `t.hour < 12` threshold entirely.
- `night_bounds()` and `_span_needs_remint()` both now compute the span once per night and resolve each boundary independently through it; the inverted-span `ValueError` guard is untouched (same message, same position).
- Added SAAO Sutherland (K92, `Africa/Johannesburg` +2), Hanle (N50, `Asia/Kolkata` +5:30) and FTN (F65, `Pacific/Honolulu` -10) `Observatory` fixtures, and five new tests to `TestSubNightWindowSiteDirection` explicitly asserting resolved UTC datetimes for every affected band, plus a band-3 re-mint-agreement test.
- Corrected the `models.py` `night_start_utc`/`night_end_utc` field-contract comment and every touched docstring to state the three-band taxonomy (entirely inside its own UTC date above +6; straddling UTC midnight for -6..+6; entirely inside the next UTC date at/below -6) instead of the superseded two-case sign rule.
- Confirmed by RED/GREEN proof: temporarily restored the pre-fix `allocation_projector.py` and re-ran the new SAAO test alone -- it raised the exact `ValueError` the plan's ground truth predicted (`start=2026-07-09T15:52:41+00:00 >= end=2026-07-09T03:00:00+00:00`); restoring the fix made it pass.

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolve each sub-night boundary against the site's own observing-night UTC span** - `d26e53c` (fix)
2. **Task 2: Pin the remaining bands -- half-hour offset, the fully-next-date band, and re-mint agreement** - `09104f5` (test)
3. **Task 3: Paired-docs audit and quality gates** - no code changes (audit found nothing to change; recorded below)

**Plan metadata:** committed separately by the orchestrator (docs: SUMMARY.md, STATE.md, ROADMAP.md, REQUIREMENTS.md)

## Files Created/Modified

- `solsys_code/allocation_projector.py` - `_night_span_utc()` replaces `_site_runs_behind_utc()`; `_time_of_day_to_datetime()` rewritten to distance-to-span resolution; `night_bounds()`/`_span_needs_remint()` updated call sites and docstrings
- `solsys_code/models.py` - `night_start_utc`/`night_end_utc` field-contract comment rewritten to the three-band taxonomy
- `solsys_code/tests/test_allocation_projector.py` - three new `Observatory` fixtures (SAAO Sutherland, Hanle, FTN) and six new tests in `TestSubNightWindowSiteDirection` covering bands 2-east, the half-hour offset, and band 3 (both-ends, evening-side-end, re-mint agreement)

## Decisions Made

- `_night_span_utc()`'s twelve-hour offset is added to the zone-carrying local datetime BEFORE converting to UTC (not computed independently in UTC), so a DST shift inside the night is applied by the conversion rather than assumed away.
- `_time_of_day_to_datetime()`'s tie-break (`min()` over `[night_candidate, night_plus1_candidate]`) deliberately favours `night` on an exact distance tie, matching the plan's specified rule.
- Task 2's FTN "evening-side end with a computed start" test uses `time(6, 30)` instead of the plan's literal `time(4, 30)` -- see Deviations below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan's test data] FTN evening-side-end fixture value produced a genuinely inverted span**
- **Found during:** Task 2 (adding `test_ftn_evening_side_end_with_computed_start_resolves_independently`)
- **Issue:** The plan specified `night_end_utc = time(4, 30)` for the FTN (`Pacific/Honolulu`, -10) evening-side-end test, with `night_start_utc` left null so the start is computed via `sun_event()`. FTN's real `sun_event()`-computed sunset for 2026-07-09 is `2026-07-10T05:17:27+00:00` -- AFTER `2026-07-10T04:30:00+00:00`. Both the pre-fix and the post-fix code resolve `4:30` to the same UTC date (`2026-07-10`), since that value's hour (`4 < 12`) already fell on the correct side of the pre-fix west-of-UTC threshold for this site -- so this is not a defect introduced by the fix. Running the test as literally specified raised `night_bounds()`'s inverted-span `ValueError` (`start=2026-07-10T05:17:27+00:00 >= end=2026-07-10T04:30:00+00:00`), because the plan's ground-truth table verified only the `zoneinfo` date-resolution arithmetic for that value (Task 1's ground truth is explicitly zoneinfo-only, never astropy), not that `04:30` falls after the site's true astronomical sunset for that specific date.
- **Fix:** Changed the test's `night_end_utc` to `time(6, 30)` -- still an early UTC hour that both the pre-fix and post-fix rule resolve onto `night + 1` identically (preserving the "right by luck under the old rule" property the plan's test intends to prove), but safely after the real sunset (`05:17:27` UTC), so the span is not inverted.
- **Files modified:** `solsys_code/tests/test_allocation_projector.py`
- **Verification:** `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler` -- 127/127 pass.
- **Committed in:** `09104f5` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in plan's literal test fixture value, not in the code under test)
**Impact on plan:** The date-resolution behaviour the test proves (an early UTC hour resolving onto `night + 1` for a band-3 site, and the two ends resolving independently) is unchanged from the plan's intent; only the specific numeric fixture value moved to avoid an unrelated astronomical-fact collision. No scope creep.

## Paired-Docs Audit (CLAUDE.md, Task 3)

**Runbook audit:** Searched `docs/runbooks/telescope_runs_calendar.rst` for a 12:00 UTC threshold, a "clock runs behind/ahead of UTC" framing, or any claim that a site ahead of UTC maps its whole night onto a single UTC date (terms: `12:00`, `12 UTC`, `runs behind`, `runs ahead`, `clock runs`, `UTC threshold`, `two-case`, `sign of the`, `sub-night`, `night_start_utc`, `night_end_utc`). Found: the only sub-night-adjacent hits are about the classical cutover's schedule-line re-parsing and the `ALLOC:` identity key (lines 42, 866, 893, 1374) -- no statement of the superseded date-mapping rule. **Outcome: no edit made**, matching the plan's planning-time prediction.

**Notebook audit:** Inspected the committed cells of `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` and `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` via a scripted scan (not assumption). `project_observation_calendar_demo.ipynb` references no `Observatory`/timezone at all. `reconcile_campaign_runs_demo.ipynb` creates two sites (`X29`/`America/Santiago` ground, `X30` satellite with no timezone) -- `America/Santiago` is band 2-west, byte-identical under this change -- and no code cell sets `night_start_utc`/`night_end_utc` (only a markdown cell's prose mentions the field names, explaining why migration 0018 must be applied to the scratch DB copy). **Outcome: no re-execution needed**, sites named above.

## Quality Gates

- `pre-commit run ruff --files solsys_code/allocation_projector.py solsys_code/models.py solsys_code/tests/test_allocation_projector.py` -- clean.
- `pre-commit run ruff-format --files solsys_code/allocation_projector.py solsys_code/models.py solsys_code/tests/test_allocation_projector.py` -- clean.
- `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler` -- 127/127 pass (both after Task 1 alone, 50/50, and after Task 2, 127/127).
- Every per-task commit also ran through the repo's pre-commit hook, which independently re-ran the pinned ruff/ruff-format and the full `pytest` suite (Sphinx build, notebook-output clearing) and passed.

## Issues Encountered

None beyond the fixture-value deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 35-REVIEW.md NF-03 is closed: every `Observatory.timezone` offset band (+10, -4, +2, +5:30, -10) now resolves sub-night boundaries correctly, verified against the plan's ground-truth `zoneinfo` table.
- The six real `Observatory` records this project's developer database already holds with affected offsets (K92/K93 Sutherland-LCO +2, C65 Montsec +2, N50 Hanle +5:30, 500 Geocentric 0, 568 Maunakea -10, F65 FTN -10) are now safe against a classical schedule line naming a partial night, once migration `0018_campaignrun_night_window_fields` is applied to the dev DB (unrelated to this task; tracked separately in STATE.md's Operator Next Steps).
- No new blockers introduced.

---
*Phase: quick-260913-rmd*
*Completed: 2026-09-14*

## Self-Check: PASSED

All modified files and both task commit hashes (`d26e53c`, `09104f5`) verified present on disk / in `git log --oneline --all`.
