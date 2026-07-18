---
phase: 18-uncertain-scheduling-investigation-spike
plan: 01
subsystem: infra
tags: [rapidfuzz, difflib, django-orm, csv, mpc-obscodes-api, investigation-spike]

requires: []
provides:
  - "18-DECISION.md Findings for SCHED-01 criteria 2-5, grounded in real live-test evidence"
  - "Confirmed real CSV Obs. Date/UT Time Range cell shapes (blank; ' to ' range; compact
    same-month range; YYYY-MM-? marker; range-in-UT-column; copy-paste garbage artifact)"
  - "Confirmed real JWST TBD natural-key collision (two distinct rows, Obs. Date = '2025-12-?')"
  - "Live rapidfuzz-vs-difflib scores for the D-09 site-code corpus against the real
    Observatory candidate pool"
  - "Live resolve_site() confirmation for 500@-170/250/274/289, including an unexpected
    finding: resolve_site() cannot currently resolve 250/274/289 via its MPC Tier 2 path
    due to a null-longitude TypeError in MPCObscodeFetcher.to_observatory() for
    satellite-type MPC records"
affects: [19-window-schema-migration, 20-range-tbd-import-asset-gap, 21-site-disambiguation-opt-in]

tech-stack:
  added: [rapidfuzz (scratch-only, not added to pyproject.toml)]
  patterns: [throwaway git-excluded investigation probe (Phase 13 eso_p2_probe.py precedent)]

key-files:
  created:
    - fuzzy_match_probe.py (repo root, git-excluded via .git/info/exclude, NOT committed)
    - .planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md
  modified: []

key-decisions:
  - "rapidfuzz's package legitimacy confirmed by human (Task 1 checkpoint approved) - the
    automated SUS verdict was a documented download-lookup false-positive"
  - "Probe script never staged/committed (per D-08); only 18-DECISION.md is a committed
    deliverable from this plan"
  - "Every resolve_site() call wrapped in a rolled-back transaction.atomic() - Observatory
    count confirmed unchanged (8 before, 8 after) across the whole probe run"

patterns-established:
  - "Throwaway investigation probe convention extended to a second phase (fuzzy_match_probe.py,
    mirroring eso_p2_probe.py): git-excluded via .git/info/exclude, never git-added, run via
    ./manage.py shell < script.py for Django ORM access"

requirements-completed: [SCHED-01]

coverage:
  - id: D1
    description: "18-DECISION.md Findings for SCHED-01 criteria 2-5, each backed by redacted
      real live-test evidence from fuzzy_match_probe.py's stdout"
    requirement: "SCHED-01"
    verification:
      - kind: manual_procedural
        ref: "grep gates: criterion 2/3/4/5 headers present, 'redacted per D-01' note present,
          no email-address pattern in the committed file (all confirmed passing)"
        status: pass
    human_judgment: false

duration: 32min
completed: 2026-07-09
status: complete
---

# Phase 18 Plan 01: Live Fuzzy-Match + resolve_site() + CSV-Shape Investigation Summary

**Live rapidfuzz-vs-difflib comparison, resolve_site()/parse_obs_window() spot-checks, and
CSV cell-shape enumeration against the real 3I/ATLAS sheet and Observatory DB, captured as
redacted evidence in 18-DECISION.md — including an unexpected finding that resolve_site()
currently cannot resolve the standard space-observatory MPC codes 250/274/289 due to a
null-longitude TypeError in MPCObscodeFetcher.to_observatory().**

## Performance

- **Duration:** 32 min (Tasks 2-3; Task 1 checkpoint gate was resolved in a prior session)
- **Started:** 2026-07-09T09:02:20Z
- **Completed:** 2026-07-09T09:34:00Z
- **Tasks:** 3/3 (Task 1 checkpoint gate + Task 2 + Task 3)
- **Files modified:** 2 (1 git-excluded, never committed; 1 committed)

## Accomplishments
- Ran a throwaway, git-excluded `fuzzy_match_probe.py` end-to-end, read-only, against the
  real 3I/ATLAS CSV (D-01 local path) and the live `Observatory` DB via the Django ORM,
  confirming `Observatory.objects.count()` was 8 at both the start and end of the run (no
  persisted writes).
- Enumerated every real `Obs. Date`/`UT Time Range` cell shape in the live 2026-07-09 CSV
  snapshot (30 data rows), confirming all of D-03/D-04/D-05's anticipated shapes plus two
  additional non-key-column shapes already handled gracefully today.
- Confirmed the real D-06 JWST TBD natural-key collision — two distinct rows sharing
  `Telescope / Instrument = 'JWST'` and `Obs. Date = '2025-12-?'`, distinguished only by
  `Filter(s)/Bandpass`.
- Ran a live rapidfuzz (`WRatio`/`token_sort_ratio`/`token_set_ratio`, `score_cutoff=60`) vs.
  difflib (`cutoff=0.6`) comparison against the D-09 real messy site-code corpus and the
  actual local `Observatory` candidate pool, revealing that the current 8-row local
  Observatory table is too narrow to produce meaningful fuzzy matches for arbitrary external
  sites — the two "hits" found were coincidental character-overlap false positives.
- Confirmed `resolve_site()`'s behavior for `500@-170`/`250`/`274`/`289`, uncovering an
  unexpected real bug: `250`/`274`/`289` all exist as valid MPC records (`observations_type:
  satellite`, `longitude: null`), but `MPCObscodeFetcher.to_observatory()`'s unconditional
  `float(self.obs_data['longitude'])` raises `TypeError` on the `null` value, which
  `resolve_site()`'s existing `except (KeyError, ValueError, TypeError)` clause catches and
  treats as a Tier 2 miss — falling through to the same safe `(None, True)` outcome as the
  length-guard case, but via a genuinely different root cause. Independently reproduced and
  confirmed (rolled back, no persistence).
- Wrote `18-DECISION.md` with Findings for SCHED-01 criteria 2-5, each backed by redacted
  verbatim evidence, plus `## Recommendation`/`## Durable summary` placeholders for Plan 02.

## Task Commits

1. **Task 1: Confirm rapidfuzz package legitimacy before the scratch install** - checkpoint
   gate only, no code/commit (resolved via human "approved" response in the prior session)
2. **Task 2: Author, install-for, and run the throwaway fuzzy_match_probe.py** - no commit
   (probe script is intentionally git-excluded and never staged, per the plan's artifact
   spec; `pip install rapidfuzz` confirmed already satisfied, 3.14.3)
3. **Task 3: Record the redacted probe evidence into 18-DECISION.md** - `c52c3dd` (docs)

**Plan metadata:** committed with this SUMMARY/STATE/ROADMAP update

## Files Created/Modified
- `fuzzy_match_probe.py` (repo root) - throwaway, read-only investigation probe; registered
  in `.git/info/exclude`, never `git add`ed
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` - Findings
  for SCHED-01 criteria 2-5, redacted per D-01, committed

## Decisions Made
- Task 1's package-legitimacy checkpoint was resolved "approved" by the human in the prior
  session; rapidfuzz confirmed as the real, well-known library (MIT, ~40M downloads/week,
  `github.com/rapidfuzz/RapidFuzz`) — install stayed scratch-only, never added to
  `pyproject.toml`.
- The probe's CSV header is located by content (`'Obs. Date'` marker) rather than assumed
  to be row 0, since the live sheet export currently carries a 2-row public-editing-notice
  preamble ahead of the real header row — a structural detail not previously documented but
  necessary for the probe to read the file correctly.
- No fuzzy-match library "winner" is picked in 18-DECISION.md's criterion 4 section —
  per the plan, that verdict is explicitly deferred to Plan 02.

## Deviations from Plan

### Auto-fixed Issues

None requiring a code fix — this phase ships no application code. One noteworthy
**unplanned discovery** surfaced during Task 2's live execution (not a deviation from the
plan's instructions, but worth flagging): `resolve_site('250')`/`('274')`/`('289')` do not
actually reach a real Tier 1/2 hit or the length guard as originally assumed — they fail
inside `MPCObscodeFetcher.to_observatory()` on a `TypeError` for satellite-type MPC records'
`null` longitude. This was recorded as a live finding in `18-DECISION.md` (criterion 5)
exactly as the plan's D-08 discipline requires ("live-tested, not reasoned from
documentation alone") — no code was changed, since this phase is investigation-only.

---

**Total deviations:** 0 auto-fixed (no code changes; this is an investigation-only phase)
**Impact on plan:** None — the unexpected `to_observatory()` finding is exactly the kind of
real, live-test evidence D-08 called for, and is documented as a finding, not treated as a
bug to fix in this phase.

## Issues Encountered
None. The probe ran cleanly end-to-end on the first attempt (no `sudo`/permission issues
beyond `./manage.py` needing to be invoked as `python manage.py` rather than an executable
script — resolved trivially, not a deviation).

## User Setup Required
None - no external service configuration required. `rapidfuzz` was already present in the
dev venv (transitive dependency of `cleo`/`poetry`); `pip install rapidfuzz` confirmed
"Requirement already satisfied."

## Next Phase Readiness
`18-DECISION.md`'s Findings for SCHED-01 criteria 2-5 are ready for Plan 02 to complete with
a `## Recommendation` (fuzzy-match library choice, informed by the criterion-4 caveat about
candidate-pool narrowness) and a `## Durable summary`. No blockers. The unexpected
`resolve_site()`/`to_observatory()` `TypeError` finding for satellite-type MPC codes should
be surfaced to Phase 19/21 planners as context, though no fix is required this phase.

---
*Phase: 18-uncertain-scheduling-investigation-spike*
*Completed: 2026-07-09*

## Self-Check: PASSED

- FOUND: `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`
- FOUND: `fuzzy_match_probe.py` (git-excluded, present on disk as expected, never committed)
- FOUND: commit `c52c3dd` in `git log --oneline --all`
