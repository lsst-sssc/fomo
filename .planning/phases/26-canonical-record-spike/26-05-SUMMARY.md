---
phase: 26-canonical-record-spike
plan: 05
subsystem: investigation-spike
tags: [django, campaign-run, calendar-utils, campaign-gap, spike, gap-closure, human-decision]

# Dependency graph
requires:
  - phase: 26-canonical-record-spike (plans 26-01..04)
    provides: "the locked D-09/D-10 event-key mechanism for classical runs, the D-11 adopt-vs-gap-fill prototype, the domain-correction reopening of SPIKE-03 for queue-scheduled runs, and plan 26-04's D-11-grade measured three-way (span/none/per-night) comparison against CampaignRun pk=1's real window"
provides:
  - "The locked queue-run projection verdict: a queue-scheduled CampaignRun gets a bare RUN:{run_pk} whole-window container event (reconciler-owned, no date component) coexisting with its real ObservationRecord-derived CalendarEvents (already produced today by the unchanged LCO/Gemini sync commands), which narrow and refine as observations are scheduled and observed"
  - "26-DECISION.md's Criterion 3/SPIKE-03 section and Domain-correction consequences 3/4/5, now true as written for both classically-scheduled and queue-scheduled runs, with a live-source-verified citation (sync_lco_observation_calendar.py) confirming the observation-level half already ships"
  - "docs/design/canonical_record_spike.rst mirroring the same settled verdict in its Domain correction, Key finding, Event key/Class-wide runs rows, Adopt/Gap-fill prose, and Future scope"
  - "ROADMAP.md and REQUIREMENTS.md synced to the settled semantics: Phase 26 status, Phase 29 success criterion 2, SPIKE-03's checkbox/traceability agreement, and RECON-02/RECON-03 reconciled against the verdict"
  - "The follow-up todo 26-03-SUMMARY.md committed to filing: correcting 26-CONTEXT.md's pre-correction owned-nights framing"
affects: [27-canonical-record-migration, 28-attribution-ui, 29-reconciler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A queue-scheduled run's calendar presence is two coexisting, independently-keyed layers: a reconciler-owned bare RUN:{run_pk} container (identity/ownership) plus the existing sync commands' ObservationRecord-derived events (per-night detail) -- neither layer ever writes to the other's rows"
    - "Verify a human's framing against live source (Serena-style read, cited file:line) before recording it as a verdict, rather than assuming it matches what already ships -- confirmed sync_lco_observation_calendar.py's url-key and _time_window() narrowing mechanism actually is the 'refines as scheduled/observed' behavior the decision describes, not a gap Phase 29 needs to build"

key-files:
  created:
    - .planning/todos/pending/2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md
  modified:
    - .planning/phases/26-canonical-record-spike/26-DECISION.md
    - docs/design/canonical_record_spike.rst
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Human-decided verdict (not Claude's own choice, per the plan's blocking checkpoint): a queue-scheduled run gets a whole-window RUN:{run_pk} container event AND keeps its individual ObservationRecord-derived CalendarEvents narrowing/refining as they are scheduled and observed -- correctly understood as the already-measured 'span' scenario (in-window count=12 decomposing as 1 container + the 11 real LCO events, LCO_ROWS_UNTOUCHED=True), not a fourth untested option"
  - "No new measurement was required or performed -- the coexistence the human asked for was already proven by plan 26-04's span scenario; only the write-up needed to state it explicitly as two coexisting key families"
  - "D-05's 80x5=400 class-wide fan-out figure does not survive: pk=29/pk=30 are both QUEUE run-type, so both take the settled bare-container form -- 1 event, not 80 and not 400. The site-fanout half of D-05 (single class-wide event, not one per site) stands unchanged and was never in question"
  - "D-11's adopt-vs-gap-fill write-strategy deferral was left exactly as it stood, per explicit coordinator instruction not to reopen, re-decide, or re-scope it -- only the 3 sentences that would otherwise contradict the new verdict were corrected"
  - "RECON-02 and RECON-03 were amended (not left silently contradicting the verdict) to name the run type each applies to, since both previously stated the pre-correction universal per-night/per-day semantics unconditionally"

requirements-completed: [SPIKE-01, SPIKE-02, SPIKE-03, SPIKE-04]

# Metrics
duration: ~55min
completed: 2026-07-27
---

# Phase 26 Plan 05: Queue-Run Projection Verdict, Doc Sync & Gap-Closure Summary

**Locked the SPIKE-03 gap 26-VERIFICATION.md found: a queue-scheduled `CampaignRun` gets a reconciler-owned whole-window `RUN:{run_pk}` container event coexisting with its real `ObservationRecord`-derived `CalendarEvent`s (verified against live `sync_lco_observation_calendar.py` source to already narrow/refine as observations are scheduled and observed) -- a human decision against plan 26-04's measured evidence, mirrored into `26-DECISION.md`, the durable `docs/design/` page, `ROADMAP.md`, and `REQUIREMENTS.md`, with D-11's write-strategy deferral left untouched as the sole remaining open item.**

## Performance

- **Duration:** ~55 min (includes the blocking decision-checkpoint pause between plan continuations)
- **Tasks:** 3 (1 blocking checkpoint, 2 auto)
- **Files modified:** 5 (`26-DECISION.md`, `docs/design/canonical_record_spike.rst`, `ROADMAP.md`, `REQUIREMENTS.md`, plus 1 new todo file)

## Accomplishments

- **Task 1 (checkpoint:decision):** Presented the full measured evidence from plan 26-04's `### SPIKE-03 gap closure` Finding (run-type inventory 12 QUEUE/12 CLASSICAL/7 SPACE; RECON-07 split 8 QUEUE/11 CLASSICAL/0 SPACE; the `campaign_gap.claimed_dates()` over-claim at `campaign_gap.py:207-209`; the span/none/per-night three-way comparison with idempotency and window-narrowing key-stability tokens; the calendar-render measurement) and a recommendation (`span`). The human's actual decision combined the coexistence already measured by `span` with an explicit naming of the observation-level half: a whole-window container event AND individual `CalendarEvent`s from `ObservationRecord` that narrow/refine as scheduled/observed. Verified against live source (`sync_lco_observation_calendar.py:92-121,186,28-33`) that this second half is exactly what the existing LCO sync command already does -- no new measurement was needed.
- **Task 2:** Amended `26-DECISION.md`'s Domain-correction consequences 3/4/5 and Criterion 3/SPIKE-03 section (renamed heading, new `#### Queue-run projection -- settled` subsection with the locked verdict, key forms, stability argument, ownership rule, measured evidence citation, rejected options with counts, trigger condition, and consuming phase); corrected the fan-out paragraph's closing sentence and 3 sentences in the `#### D-11` subsection without reopening its deferral. Mirrored every change into `docs/design/canonical_record_spike.rst` (Domain correction, Key finding, Event key/Class-wide runs rows, Adopt/Gap-fill prose, Future scope). `sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D exclude_patterns=notebooks/*,_build` succeeded with the same 9 pre-existing warnings 26-VERIFICATION.md already recorded as unrelated to this file.
- **Task 3:** Synced `ROADMAP.md` (Phase 26 status bullet, plan count 3/3->5/5, Phase 29 success criterion 2, Progress-table row) and `REQUIREMENTS.md` (SPIKE-03 line + traceability row now agree as Complete; RECON-02/RECON-03 amended to name run type; SPIKE-01/02/04 and the Coverage block byte-unchanged) to the settled semantics. Filed `.planning/todos/pending/2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md`, the follow-up 26-03-SUMMARY.md committed to.

## Task Commits

Real phase-26 branch (`issue37-telescope-runs-calendar`):

1. **Task 1: Settle whether a queue-scheduled run projects onto the calendar, and under what key** -- checkpoint:decision, no commit (task writes nothing to any file, per its own acceptance criteria)
2. **Task 2: Amend the decision doc and the durable design page so the key-scheme statement is true for both run types** -- `0ac8e58` (docs)
3. **Task 3: Sync the stale upstream wording in ROADMAP.md and REQUIREMENTS.md, and file the outstanding follow-up todo** -- `a6f761a` (docs)

## Files Created/Modified

- `.planning/phases/26-canonical-record-spike/26-DECISION.md` -- dated closure note under the Domain-correction preamble; consequences 3/4/5 rewritten from "open question" to the settled verdict; Criterion 3 heading renamed to cover both run types; new `#### Queue-run projection -- settled` subsection; fan-out paragraph's closing sentence rewritten; 3 sentences in `#### D-11` corrected (heading parenthetical, the "second question is now open too" sentence, the Consuming-phase line) without reopening the deferral.
- `docs/design/canonical_record_spike.rst` -- Domain correction section gains a settled-verdict note and rewritten consequence bullets; Key finding paragraph restated; second `.. list-table::`'s Event key and Class-wide runs rows and its bolded intro rewritten; the Adopt/Gap-fill prose's "second, separate question is also open" passage restated as settled; Future scope reduced from two open questions to one.
- `.planning/ROADMAP.md` -- Phase 26 status bullet, plan count (3/3 -> 5/5), Wave 5 checkbox ticked, Phase 29 success criterion 2, Progress-table row (5/5, Gap closure).
- `.planning/REQUIREMENTS.md` -- SPIKE-03 requirement line and traceability row (now agree: Complete); RECON-02 and RECON-03 amended to name run type.
- `.planning/todos/pending/2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md` (new) -- the follow-up 26-03-SUMMARY.md committed to filing, naming `26-CONTEXT.md`'s stale "4 uncovered nights" framing and pointing at `26-DECISION.md`'s Domain-correction section as current truth.

## Decisions Made

All key decisions are listed in the frontmatter `key-decisions` field above. The headline: the human's verdict was correctly understood as the already-measured `span` scenario plus an explicit statement that the observation-level detail layer is the existing, unchanged `ObservationRecord`-sync mechanism -- confirmed against live source rather than assumed, so no new measurement or scratch environment was needed to close this gap.

## Deviations from Plan

None beyond the coordinator-directed checkpoint resolution itself, which is not a Rule 1-4 auto-fix but an explicit human decision at a blocking checkpoint, handled per the plan's own `<checkpoint_protocol>`.

**Live-source verification performed before writing the verdict (required by the coordinator's resolution message, not a deviation from the plan):** read `solsys_code/management/commands/sync_lco_observation_calendar.py` (lines 28-33, 92-121, 123-216, 186) to confirm, rather than assume, that `CampaignRun` pk=1's 11 real LCO `CalendarEvent`s are genuinely `ObservationRecord`-derived, keyed on the LCO portal request URL (`facility.get_observation_url()`, line 186), and that their `start_time`/`end_time` narrow from a banner (`parameters['start']`/`['end']`) to a placed block (`scheduled_start`/`scheduled_end`) as the LCO scheduler acts -- exactly the "narrow or refine as they get scheduled and observed" mechanism the human's decision names. This already ships today; Phase 29 needs to leave it alone, not build it.

## Issues Encountered

- The Sphinx-build automated verify string's literal `! grep -q 'canonical_record_spike' log` check would fail on any normal build, since Sphinx always prints a `writing output... [N%] design/canonical_record_spike` progress line regardless of warnings. Per 26-VERIFICATION.md's own precedent methodology ("confirmed by grepping warning output for the file name: none"), the actual gate applied was: build exits 0, and no line beginning `WARNING:` names `canonical_record_spike`. Confirmed: 9 warnings total, all pre-existing and unrelated to this file (identical set 26-VERIFICATION.md already recorded).
- `git status --porcelain` filtered for non-`.planning/`/non-`docs/design/canonical_record_spike.rst` paths reports one dirty file: `src/fomo/settings.py` -- the pre-existing, uncommitted local dev modification named in the executor's `working_tree_warning`, never staged, committed, or touched by this plan. Confirmed identical to the same documented exception plan 26-04 recorded.
- `src/fomo_db.sqlite3`'s mtime has moved since plan 26-04 recorded its fingerprint (size unchanged at 946176), because the coordinator's message noted live `runserver`/`manage.py shell` processes are attached to it during this session. This plan made no database writes of its own (documentation-only, no scratch environment stood up) and does not attempt to "restore" the mtime, per the coordinator's explicit instruction.

## User Setup Required

None -- no external service configuration required. The one checkpoint in this plan (`type="checkpoint:decision"`, `gate="blocking"`) was resolved by the coordinator relaying the human's explicit decision.

## Next Phase Readiness

- Phase 27-29 planning can now read `26-DECISION.md`'s `#### Queue-run projection -- settled` subsection and `docs/design/canonical_record_spike.rst`'s matching sections for the complete, locked queue-run verdict -- no open question remains for SPIKE-03's key-scheme scope, closing exactly the gap 26-VERIFICATION.md found.
- `ROADMAP.md`'s Phase 29 success criterion 2 and `REQUIREMENTS.md`'s RECON-02/RECON-03 now state the settled per-run-type semantics, so Phase 29 planning will not inherit stale universal per-night/per-day wording.
- D-11's write-strategy deferral remains the single explicitly open item Phase 29 inherits, with both options fully measured, a concrete two-writer-churn code-level finding, and a named trigger condition (v2.3's adapter rewiring) -- unchanged and untouched by this plan.
- The follow-up todo correcting `26-CONTEXT.md`'s pre-correction "4 uncovered nights" framing is filed and pending; it should be actioned before Phase 29 is planned so the archival discussion record does not contradict the now-settled verdict.
- `solsys_code/` remains byte-identical to its pre-phase-26 state (`git diff --quiet -- solsys_code/` confirmed at every task boundary); `python -m pytest` passes; this plan is documentation-only end to end.
- No blockers.

---
*Phase: 26-canonical-record-spike*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: `.planning/phases/26-canonical-record-spike/26-DECISION.md`
- FOUND: `docs/design/canonical_record_spike.rst`
- FOUND: `.planning/ROADMAP.md`
- FOUND: `.planning/REQUIREMENTS.md`
- FOUND: `.planning/todos/pending/2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md`
- FOUND: `.planning/phases/26-canonical-record-spike/26-05-SUMMARY.md`
- FOUND commit: `0ac8e58` (Task 2)
- FOUND commit: `a6f761a` (Task 3)
