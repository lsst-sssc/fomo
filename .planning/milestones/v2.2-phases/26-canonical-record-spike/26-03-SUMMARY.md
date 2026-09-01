---
phase: 26-canonical-record-spike
plan: 03
subsystem: investigation-spike
tags: [django-migrations, campaign-run, calendar-events, reconciler, sphinx-docs, source-vocabulary]

# Dependency graph
requires:
  - phase: 26-canonical-record-spike (plans 26-01, 26-02)
    provides: >
      D-04 dated snapshot, D-16 PROJECT.md correction, SPIKE-02's four adapter identity
      mappings, the stage-2/stage-0 inventory (D-05..D-08), the RECON-07 baseline,
      SPIKE-04's migration-application and measured-rename-blast-radius findings,
      SPIKE-01's IntegrityError coexistence proof, and SPIKE-03's adopt-vs-gap-fill
      three-way prototype comparison (D-09..D-11).
provides:
  - Completed 26-DECISION.md ## Recommendation section locking SPIKE-01, SPIKE-02,
    SPIKE-04, and SPIKE-03's event-key scheme as falsifiable verdicts for Phases 27-29.
  - A deliberate, human-directed deferral of SPIKE-03's write-strategy half (D-11,
    adopt-vs-gap-fill) to Phase 29, recorded with both measured options and a new
    code-level two-writer-churn finding (sync_lco_observation_calendar.py:361 +
    calendar_utils._update_or_unchanged()).
  - docs/design/canonical_record_spike.rst — durable, redaction-free design page wired
    into docs/design/design.rst's toctree, naming the E10 blank-timezone gap as a
    concrete Phase 27 prerequisite.
  - A fully torn-down investigation: scratch branch deleted unmerged, local_settings.py
    and tmp/ removed, no solsys_code/ diff, real DB fingerprint unchanged.
affects: [27-canonical-record-migration, 28-attribution-ui, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Investigation-spike teardown: scratch git branch + gitignored tmp/ + gitignored
      local_settings.py, deleted/force-deleted only after every finding is committed
      into the durable docs."
    - "docs/design/<name>.rst spike page: opening four-part paragraph (what/date/no-code-
      shipped/pointer-with-archival-caveat), Background, bolded Key finding, per-cluster
      .. list-table:: decision tables with :header-rows: 1 and explicit :widths:,
      Future scope — matching eso_feasibility_spike.rst and uncertain_scheduling_spike.rst."

key-files:
  created:
    - docs/design/canonical_record_spike.rst
  modified:
    - .planning/phases/26-canonical-record-spike/26-DECISION.md
    - docs/design/design.rst

key-decisions:
  - "source vocabulary locked at six values (five roadmap values + LEGACY); LEGACY assigned to all 31 pre-milestone rows; source/telescope_class stay out of both existing partial unique constraints."
  - "Derivation rule locked verbatim: approval_status == APPROVED and source != WEB means no approval was required, not that a human approved it; no fourth approval value added."
  - "Reconciler event key locked: RUN:{run_pk}:{date} with {date} always the site-local observing night, never the naive UTC date."
  - "Class-wide runs get one event per day (not per site); space-mission runs get one spanning event for the whole window; stage 0 (no window_start) produces no event but is counted and reported."
  - "The adopt-vs-gap-fill write strategy (D-11) is deliberately left open for Phase 29, per explicit human decision at the task-1 checkpoint — not a locked verdict, unlike the rest of SPIKE-03."
  - "Migration shape locked: RenameModel (CalendarEventTelescopeLabel -> CalendarEventMeta) then three AddField ops, hand-authored; rename checklist is six integration points, not four (admin reverse-URL name and four test modules' own class references were the two additions)."
  - "telescope_class keeps its existing name despite the widened three-meaning vocabulary, to avoid a separate naming discussion delaying Phase 27."
  - "calendar_utils.py's five underscore-prefixed cross-module helpers get a naming recommendation (drop the underscore, fold into Phase 27's own edits) but the todo stays open, not closed."

requirements-completed: [SPIKE-01, SPIKE-02, SPIKE-03, SPIKE-04]

duration: ~90min (includes a task-1 human decision-checkpoint pause between plan continuations)
completed: 2026-07-27
---

# Phase 26 Plan 03: Recommendation, Durable Publication, and Teardown Summary

**Locked falsifiable verdicts for SPIKE-01, SPIKE-02, SPIKE-04, and SPIKE-03's event-key
scheme in `26-DECISION.md`; deliberately deferred SPIKE-03's adopt-vs-gap-fill write
strategy to Phase 29 per explicit human decision; published `docs/design/canonical_record_spike.rst`
into the Design Notes toctree; and fully tore down every scratch artifact the phase
created, leaving only two documentation deliverables behind.**

## Performance

- **Completed:** 2026-07-27
- **Tasks:** 3 (1 blocking decision checkpoint, 2 auto)
- **Files modified:** 3 (`26-DECISION.md`, `docs/design/canonical_record_spike.rst` [new], `docs/design/design.rst`)
- **Duration:** ~90 min end-to-end, including the pause between the task-1 checkpoint and the human's resolved decision

## Accomplishments
- Completed `26-DECISION.md`'s `## Recommendation` section: one subsection per SPIKE-01..04
  criterion, each restating a locked CONTEXT.md decision and citing the Finding that
  grounds it, plus the recorded D-16/PROJECT.md instruction and the calendar_utils.py
  naming recommendation.
- Recorded a new, code-verified finding (not previously in the doc) that adopt would
  produce a genuine repeating two-writer churn loop, not a one-time cost:
  `sync_lco_observation_calendar.py:361` and `calendar_utils._update_or_unchanged()`
  (lines 297-315) together mean the LCO sync command would overwrite the reconciler's
  stamp on the same 11 rows every cycle, and the reconciler would re-stamp them back.
- Published `docs/design/canonical_record_spike.rst` (the third `docs/design/` spike
  page) with plain-English decision tables, wired it into `docs/design/design.rst`'s
  toctree, and confirmed the exact pre-commit `sphinx-build` invocation exits 0.
- Fully tore down the investigation: deleted the scratch branch
  `spike/26-canonical-record-probe` (force, unmerged, `git log --all` now returns nothing
  for its scratch migration path), removed `local_settings.py` and `tmp/`, confirmed no
  `solsys_code/` file differs from committed state, and confirmed the real
  `src/fomo_db.sqlite3` fingerprint (`946176 1785094461`) is unchanged from the value
  recorded at phase start.
- Confirmed `ruff check .`, `ruff format --check .`, `python -m pytest`, and
  `./manage.py test solsys_code.tests.test_calendar_template` (24 tests, OK) all pass
  from the pristine post-teardown tree — the two lint commands surface only the
  pre-existing repo-wide drift already logged in `deferred-items.md` by plan 26-01,
  confirmed empty-diff against every flagged file back to commit `77e16b5`.

## Task Commits

1. **Task 1: Settle adopt-vs-gap-fill from the prototype's measured numbers (D-11)** —
   checkpoint:decision, no commit (26-DECISION.md explicitly unchanged by this task).
   Presented the measured three-way comparison from `tmp/26-prototype-counts.txt` and a
   recommendation (gap-fill); the human's resolved decision was **defer to Phase 29**,
   not lock either option — see Deviations below.
2. **Task 2: Complete 26-DECISION.md's Recommendation for SPIKE-01..04** — `8829dd6` (docs)
3. **Task 3: Publish the durable docs/design page and discard every throwaway artifact** — `eba1c83` (docs)

_No plan-metadata commit yet — created below, per the final_commit step._

## Files Created/Modified
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` — `## Recommendation`
  section completed (all four SPIKE criteria, D-11's deferral subsection, the D-16
  instruction, the folded-todo naming posture); `## Durable summary` pointer resolved.
- `docs/design/canonical_record_spike.rst` (new) — durable spike page: Background, Key
  finding, four decision-table clusters (source/approval, event-key/ownership,
  stage-semantics, migration/rename), the open D-11 question with both measured options
  and the two-writer-churn finding, and the E10 blank-timezone Phase 27 prerequisite.
- `docs/design/design.rst` — one new toctree line, `canonical_record_spike`, appended
  after `uncertain_scheduling_spike`.

## Decisions Made

**The human overrode the task-1 checkpoint's own recommendation.** Claude's task-1
brief recommended locking `gapfill`; the human's resolved decision was to **not lock
either option** and instead defer the write-strategy choice to Phase 29 entirely. This
is recorded as the actual decision throughout — the doc does not present gap-fill as a
soft lean or imply indecision; it presents both options as fully measured and equally
viable, with the deferral itself as the deliberate outcome.

All other key decisions are listed in the frontmatter `key-decisions` field above —
each restates a CONTEXT.md-locked decision (D-05 through D-19) grounded in a Finding
plans 26-01/26-02 recorded, per the plan's evidence-then-recommendation ordering
(mirroring `18-DECISION.md`'s precedent).

## Deviations from Plan

### Recorded plan-deviation (human-directed, not an auto-fix)

**1. [Human checkpoint override] SPIKE-03's write-strategy half (D-11) deferred to Phase 29 instead of locked**

- **Found during:** Task 1 (checkpoint:decision)
- **What the plan's `must_haves.truths` requires:** "a locked, falsifiable verdict for
  all four of SPIKE-01..04, each grounded in the Findings ... so Phases 27, 28 and 29
  each execute a decision instead of making one."
- **What actually happened:** the human explicitly declined to lock adopt-vs-gap-fill,
  stating the choice felt like "randomly picking something" between two options that are
  indistinguishable on the rendered calendar, and correctly judged that the spike's job
  was to produce the measurement, not force a premature verdict on the one criterion
  where both options are fully viable.
- **Resolution:** SPIKE-03 is split explicitly in `26-DECISION.md` and the durable page:
  the event-key scheme (`RUN:{run_pk}:{date}`, site-local night) **is** locked; only the
  adopt-vs-gap-fill write strategy is left open, with both measured options, the new
  two-writer-churn code finding, and the explicit trigger condition (v2.3's adapter
  rewiring) that should prompt Phase 29 to decide.
- **Impact on the must_have:** SPIKE-01, SPIKE-02, SPIKE-04, and SPIKE-03's key-scheme
  half are locked and falsifiable, satisfying three-and-a-half of the four criteria as
  written. SPIKE-03's write-strategy half is **not** locked — this is a deliberate,
  human-directed deviation from the must_have's literal "all four" wording, recorded
  here rather than silently satisfied by inventing a verdict or silently dropped without
  documentation.
- **Files modified:** `.planning/phases/26-canonical-record-spike/26-DECISION.md`,
  `docs/design/canonical_record_spike.rst`
- **Committed in:** `8829dd6`, `eba1c83`

---

**Total deviations:** 1 (human-directed plan deviation, not a Rule 1-4 auto-fix)
**Impact on plan:** The deviation narrows one criterion's scope from "locked" to
"measured and explicitly deferred with a named decision-maker (Phase 29) and a named
trigger condition." No scope creep; no other must_have or acceptance criterion is
affected.

## Issues Encountered

None beyond the recorded deviation above. All four verification gates (`sphinx-build`,
`ruff check .`, `ruff format --check .`, `python -m pytest`,
`./manage.py test solsys_code.tests.test_calendar_template`) passed on the first run
after teardown; the two lint gates' failures are pre-existing and already logged in
`deferred-items.md`, confirmed via empty `git diff` against commit `77e16b5` for every
flagged file.

## Post-Execution Correction

**After this plan's tasks were committed and torn down, the project owner (a
professional astronomer and the domain authority on this codebase) issued a correction
that partially reframes the locked SPIKE-03 key-scheme verdict.** This is not a
deviation Claude made — it is a domain correction from the human, applied as an
amendment to the already-committed `26-DECISION.md` and `docs/design/canonical_record_spike.rst`.

**The correction:** there is a fundamental difference between a classically scheduled
run (a specific set of owned nights, each with known start/stop times) and a
queue-scheduled run (ESO, SOAR, Gemini, and especially LCO's queue network), whose
window is a span of time during which observations *could* happen, not a set of nights
the run owns. For a queue run, a night inside the window with no scheduled observation is
the normal, correct state — not a gap needing backfill. `CampaignRun` pk=1, the one real
run this plan's SPIKE-03 prototype was built against, is itself an LCO queue run, not a
classical one.

**What this changes:**
- The "4 uncovered nights" framing in the D-11 adopt-vs-gap-fill comparison is corrected:
  those nights are ones LCO's queue scheduler simply did not use, not coverage gaps.
- The locked `RUN:{run_pk}:{date}` event-key scheme is **rescoped, not unlocked** — it
  remains the right key for classically scheduled runs (the site-local-night derivation
  and its measurement stand unchanged); whether a queue run should be projected per-night
  at all is now a second open question for Phase 29, alongside the original write-strategy
  question.
- D-05's 80x5=400 class-wide fan-out figure is flagged as very likely the same category
  error one level up (a class-wide queue allocation treated as owned nights).
- RECON-07's "19 runs become visible" goal is confirmed as still valid in intent; the
  per-night-event *mechanism* is now open for queue runs specifically.
- SPIKE-01's `source` vocabulary is confirmed as already sufficient for the reconciler to
  branch on run type — unaffected by this correction.

**Both files were amended** (`.planning/phases/26-canonical-record-spike/26-DECISION.md`
gained a `### Domain correction — queue windows are not sets of owned nights` subsection
placed immediately before the Criterion 3/SPIKE-03 material it qualifies, plus inline
amendments to the fan-out and D-11 paragraphs; `docs/design/canonical_record_spike.rst`
gained a matching `Domain correction` section and matching inline amendments) and
recommitted as a distinct amendment commit, not folded into the original task commits.
The Sphinx build and the plain-English (`upsert`) / no-email grep gates were re-run
clean after the amendment.

**Flagged, not fixed, per the correction's own scope boundary:** `26-CONTEXT.md`'s D-11
framing and RECON-07's wording both still read as though every run's window is a set of
owned nights needing full per-night coverage. A separate todo should be filed to correct
that upstream framing before Phase 29 is planned — this plan does not edit those docs.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Phase 27 (canonical-record migration) has a proven migration shape, a six-point rename
checklist, the full `source`/`telescope_class` vocabulary, the derivation rule, and one
concrete, actionable prerequisite: backfill `Observatory.timezone` for obscode `E10`
(Siding Spring) before any reconciler work depends on site-local-night derivation.

Phase 28 (attribution UI) has the ownership rule (companion `run` FK, "not mine, never
touch") and the locked event-key scheme to build against.

Phase 29 (reconciler) has everything locked for classically scheduled runs. For
queue-scheduled runs, two questions are explicitly open per the post-execution domain
correction above: the adopt-vs-gap-fill write strategy (both options fully measured, a
concrete code-level risk finding for adopt, and a named trigger condition — v2.3's
adapter rewiring — for when to revisit it), and, newly, whether a queue run should be
projected per-night onto the calendar at all rather than as a single window span.

No blockers. The repository is back to its pre-phase state apart from the two
documentation deliverables — verified, not assumed: `git status --porcelain` is clean,
no `solsys_code/` file differs from its committed state, the scratch branch and its
migration are gone from every ref, and `src/fomo_db.sqlite3`'s fingerprint is unchanged.

---
*Phase: 26-canonical-record-spike*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: `.planning/phases/26-canonical-record-spike/26-DECISION.md`
- FOUND: `docs/design/canonical_record_spike.rst`
- FOUND: `docs/design/design.rst`
- FOUND: `.planning/phases/26-canonical-record-spike/26-03-SUMMARY.md`
- FOUND commit: `8829dd6` (docs(26-03): complete 26-DECISION.md Recommendation)
- FOUND commit: `eba1c83` (docs(26-03): publish canonical-record spike design page)
- FOUND commit: `c57edc9` (docs(26-03): add plan summary)
