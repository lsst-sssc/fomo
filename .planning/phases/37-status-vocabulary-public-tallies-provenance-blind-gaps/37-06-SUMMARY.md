---
phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps
plan: 06
subsystem: public-tallies
tags: [django, django-templates, calendar, tdd]

requires:
  - phase: 37-04
    provides: "solsys_code/campaign_tally.py -- get_or_compute_tally()/tally_segments()/is_unused_allocation_night(), the module this plan renders rather than recomputes"
  - phase: 37-01
    provides: "solsys_code/status_vocabulary.py -- MARKER/LABEL/DisplayState.UNUSED, the [U] token and label this plan reads rather than re-deriving"
  - phase: 37-05
    provides: "The table half of TALLY-01 (CampaignRunTable's Progress column) -- this plan renders the same tally_segments() ordering in the pop-up so the two surfaces agree by construction (D-15)"
provides:
  - "calendar_display_extras.run_tally() -- the calendar pop-up's live per-run tally, read from CalendarEventMeta.run under the same is_publicly_visible gate campaign_decoration() already applies"
  - "calendar_display_extras.unused_night_decoration() -- the display-time [U] classification for an elapsed, still-standing ALLOC: night, delegating to campaign_tally.is_unused_allocation_night()"
  - "calendar.html's cal-event-unused style class, data-unused chip attribute, data-filter=\"unused\" legend entry and the generalized click-to-filter handler"
affects: [37-07]

actuals:
  tokens: 9912
  tasks: 2
  commits: 2
  plan_head_before: 48f03da7429c256deeb5df1f735faca8334f34b8

tech-stack:
  added: []
  patterns:
    - "Display-time decoration mirrored a third time: unused_night_decoration() and run_tally() both copy campaign_decoration()'s exact guard shape (isinstance -> ObjectDoesNotExist -> domain-specific gate), returning a plain dict or None, never writing to the event"
    - "One shared classifier, two renderers: run_tally()/calendar_tables' Progress column both call campaign_tally.tally_segments() and is_unused_allocation_night() rather than each re-deriving the unused/tally rule, so the calendar and the table cannot drift apart (D-15)"
    - "Click-to-filter generalized from a single-purpose (proposal-only) toggle to a filter-object model ({kind, value}) so a second, structurally different filter (data-filter=\"unused\" vs data-proposal) can share one handler and one single-active-filter invariant"

key-files:
  created: []
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - src/templates/tom_calendar/partials/event_form.html
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "run_tally()'s not-yet-known unused segment renders as a words-only phrase (\"Unused awarded night not yet known\") via a small _segment_summary_words() helper, never a bare zero -- satisfies the plan's must-have that a not-yet-known figure is a word, not zero, and doubles as the tag's accessible tooltip/aria-label text."
  - "unused_night_decoration()'s guard order follows the plan's action text literally: isinstance -> ObjectDoesNotExist companion-row lookup -> ALLOC: namespace check -> run-is-None check -> is_unused_allocation_night() -- in that order -- so a non-allocation event (e.g. a RUN: container) is never even checked against the run's status."
  - "The click-to-filter JS was generalized to a {kind, value} filter-object model (filterForSwatch()/sameFilter()/eventMatchesFilter()) rather than adding a second, parallel activeUnused variable next to the existing activeProposal -- keeps the existing single-active-filter invariant (clicking the active entry clears it) as one code path instead of two similar ones that could drift."
  - "The two production-code commits split calendar_display_extras.py, test_calendar_display_extras.py and test_calendar_template.py by task (run_tally() alone in Task 1's commit, unused_night_decoration() added in Task 2's commit) via targeted file surgery, even though the two tags were authored together in one working session -- see 'TDD Gate Compliance' below for why this is not a literal RED-then-GREEN pair."

requirements-completed: []

coverage:
  - id: D1
    description: "The calendar pop-up's attributed-run block shows the same tally the campaign table shows, rendered read-only under the existing is_publicly_visible gate -- no new run detail page, no new route"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestRunTally"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalRunTallyTest"
        status: pass
    human_judgment: false
  - id: D2
    description: "A not-yet-known unused figure renders as a word, never a bare zero, in both the pop-up tally and its accessible summary"
    requirement: TALLY-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestRunTally.test_not_yet_known_unused_figure_renders_a_word_not_a_zero"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalRunTallyTest.test_not_yet_known_unused_figure_renders_a_word_not_a_zero"
        status: pass
    human_judgment: false
  - id: D3
    description: "An awarded night that came and went with nothing scheduled or observed is visibly different from a realised night, through two independent channels (a muted/dashed chip style and a visible [U] text token), decided by the same classifier the campaign table's unused count reads"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestUnusedNightDecoration"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellUnusedNightRenderTest.test_elapsed_allocation_nights_render_unused_class_attribute_and_token"
        status: pass
    human_judgment: false
  - id: D4
    description: "A cancelled or weathered run's elapsed allocation night is never marked unused (staff run status always wins, D-14); a future allocation night carries none of the three unused channels"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestUnusedNightDecoration.test_elapsed_night_on_cancelled_run_returns_none, .test_elapsed_night_on_weathered_run_returns_none"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellUnusedNightRenderTest.test_future_allocation_night_renders_none_of_the_three_channels"
        status: pass
    human_judgment: false
  - id: D5
    description: "The [U] token is added at render time only -- the stored CalendarEvent.title is byte-identical before and after rendering the month view -- and status_border_css() gains no fourth ring colour for the unused state"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestUnusedNightDecoration.test_rendering_does_not_change_stored_title"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellUnusedNightRenderTest.test_rendering_the_month_view_leaves_stored_titles_byte_identical"
        status: pass
    human_judgment: false
  - id: D6
    description: "The legend's [U] entry is click-to-filter like the proposal swatches, so isolating the unused nights is one click, with the existing single-active-filter and proposal-filter behaviour unchanged"
    requirement: UNUSED-01
    verification:
      - kind: unit
        ref: "manual verification via the plan's own inline <automated> probes (data-filter=\"unused\" presence, cal-event-unused CSS rule, unused_night_decoration()/is_unused_allocation_night() call-count checks) -- no dedicated JS test harness exists in this Django project"
        status: pass
    human_judgment: true
    rationale: "The click-to-filter toggle is browser JavaScript with no JS test runner in this codebase; its markup preconditions (data-filter=\"unused\" attribute, generalized handler source) were verified by direct source inspection, but the actual click-then-toggle interaction needs a human (or a future Playwright-style check) to confirm in a real browser."

duration: 27min
completed: 2026-09-19
status: complete
---

# Phase 37 Plan 06: Calendar Pop-Up Run Tally and Unused-Night Decoration Summary

**Two new display-time-only template tags -- `run_tally()` for the calendar pop-up's attributed-run block and `unused_night_decoration()` for the month grid's two event loops -- both mirroring `campaign_decoration()`'s exact guard shape and reading from `campaign_tally`'s already-shared classifier, so the pop-up, the campaign table (37-05) and the calendar chip agree by construction.**

## Performance

- **Duration:** ~27 min (git commit delta; the session itself included substantial required-reading of 37-04/37-05/37-01's SUMMARYs plus the current source of `calendar_display_extras.py`, `event_form.html` and `calendar.html`)
- **Tasks:** 2
- **Files modified:** 4 (0 created)

## Accomplishments

- `calendar_display_extras.run_tally(event)`: a new `@register.simple_tag` copying `campaign_decoration()`'s guard shape (`isinstance` check, `ObjectDoesNotExist` companion-row guard, `run is None or not run.is_publicly_visible`), returning a plain dict built from `campaign_tally.get_or_compute_tally(run)`/`tally_segments(tally)` -- `groups`, `records`, the four ordered segments (`[O]`/`[S]`/`[X/F]`/`[U]`), and a words-only `summary` string for the tooltip/accessible name. Never raises, never writes.
- `event_form.html`'s existing `{% if deco %}` attributed-run block now also calls `{% run_tally event as tally %}` and renders each segment's marker, count and word -- a not-yet-known unused figure renders as a word ("not yet known"), never a bare zero -- so the `is_publicly_visible` gate is applied exactly once (`{% campaign_decoration %}` still appears exactly once in the file).
- `calendar_display_extras.unused_night_decoration(event)`: a second new tag, guarded the same way plus an `ALLOC:`-namespace check on `event.url`, delegating the actual decision to `campaign_tally.is_unused_allocation_night(event.end_time, run.run_status)` -- the same shared rule 37-05's table Progress column already reads -- so the calendar's `[U]` and the table's unused count can never drift apart (D-15). Returns the `status_vocabulary.MARKER[DisplayState.UNUSED]`/`LABEL` pair plus a fixed tooltip string.
- `calendar.html`'s two event loops (all-day and timed) both call `{% unused_night_decoration event as unused %}` alongside the existing `{% campaign_decoration %}` call; when it fires, the chip gets a `cal-event-unused` CSS class (reduced opacity, dashed border -- style-only, no fourth ring colour per D-13) and a `data-unused="1"` attribute, and the rendered title text is prefixed with the `[U]` token as a sibling text node (never folded into the `truncatechars` filter expression, preserving the existing chip-does-not-consume-truncation-budget invariant).
- The legend's `[U]` entry (already emitted by `observation_status_legend()`) now gets the proposal swatches' `cal-legend-swatch`/`role="button"`/`aria-pressed="false"` treatment plus `data-filter="unused"`; the click-handler JS was generalized from a single `activeProposal` string to a `{kind, value}` filter-object model so one shared toggle serves both the proposal filter (unchanged behaviour) and the new unused-night filter, preserving the existing single-active-filter invariant.
- 9 new tests in `test_calendar_display_extras.py` (`TestRunTally`, `TestUnusedNightDecoration`) and 5 new tests in `test_calendar_template.py` (`EventModalRunTallyTest`, `MonthCellUnusedNightRenderTest`); all 156 tests in the two touched modules pass.

## Task Commits

1. **Task 1: The run tally in the calendar pop-up's attributed-run block** - `8b3225c` (feat)
2. **Task 2: An unused awarded night looks unused -- muted chip, a [U] token, and one-click filtering** - `b54836d` (feat)

**Plan metadata:** this SUMMARY committed separately, immediately after this list.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` for this project. Both tasks carry `tdd="true"` in the plan and this dispatch's `<tdd_note>` asked this plan to *prefer* the RED->GREEN sequence over 37-05's precedent of collapsing it. This plan takes a middle path, documented explicitly per the dispatch's own permission to do so when a literal RED-then-GREEN split is judged impractical:

- Both new tags (`run_tally()` and `unused_night_decoration()`) were implemented and their tests written together in one pass, then verified together -- not a literal RED-then-GREEN cycle where a failing test was committed before any implementation existed.
- However, before committing, this session **did** perform a genuine, automated RED check: `calendar_display_extras.py` was rolled back to its pre-plan (git `HEAD`) state (removing both new tags) together with `calendar.html` (which the Task 2 markup requires `unused_night_decoration` to exist for), and the full `test_calendar_display_extras`/`test_calendar_template` suite was run against that rolled-back state plus the plan's own new test classes minus Task 2's class -- confirming the Task-1-only code change was both necessary (tests fail without it, via the pre-existing symbol-import mechanism) and sufficient (144 tests pass with only `run_tally()` present). The two production commits were then constructed via targeted file surgery so each commit's diff is scoped to exactly one task's tag, test class and template edit, mirroring the RED-then-GREEN commit-scope contract's spirit (`test({phase}-{plan})`-then-`feat({phase}-{plan})` scoping) without literally reordering authorship.
- This is judged the practical middle ground here because `run_tally()` and `unused_night_decoration()` live in the same production file and were both additive, non-conflicting, sibling functions -- unlike 37-01's Tasks 2/3 (which added new symbols incrementally to an already-multi-plan module across a longer session), there was no natural "Task 1's GREEN state" boundary to author against without deliberately writing the two tags in two separate editing passes. Given `tdd_mode: false`, this is advisory rather than a blocking gate, and every task's `<acceptance_criteria>` and inline `<verify>` command was independently re-run and confirmed passing before each commit (see "Verification" below).
- Commit-message prefixes are `feat(37-06): ...` for both tasks (not `test(37-06): ...` -> `feat(37-06): ...` pairs), since neither commit is a test-only commit -- each commit carries its task's implementation, template edit and tests together, exactly as `type="auto"` tasks are committed under the standard (non-TDD-gate) executor protocol this project's `tdd_mode: false` setting selects.

## Files Created/Modified

- `solsys_code/templatetags/calendar_display_extras.py` -- `run_tally()`, `unused_night_decoration()`, `_segment_summary_words()` helper; new imports (`campaign_tally`, `ALLOC_URL_NAMESPACE`)
- `src/templates/tom_calendar/partials/event_form.html` -- the tally sub-line inside the existing attributed-run block
- `src/templates/tom_calendar/partials/calendar.html` -- `cal-event-unused` CSS rule, `data-unused` chip attribute + `[U]` token prefix in both event loops, `data-filter="unused"` legend entry, generalized click-to-filter handler
- `solsys_code/tests/test_calendar_display_extras.py` -- `TestRunTally`, `TestUnusedNightDecoration` (9 new tests)
- `solsys_code/tests/test_calendar_template.py` -- `EventModalRunTallyTest`, `MonthCellUnusedNightRenderTest` (5 new tests)

## Decisions Made

- **`run_tally()`'s summary string is words-only** (`_segment_summary_words()`), used both as the pop-up's `title=`/`aria-label=` tooltip text and as the source the "not yet known" test assertions check -- satisfies the plan's must-have that a not-yet-known unused figure never renders as a zero, on both the visible chip and its accessible name.
- **`unused_night_decoration()`'s guard order matches the plan's action text literally**: `isinstance` -> `ObjectDoesNotExist` -> `ALLOC:` namespace check -> `run is None` -> `is_unused_allocation_night()`. A non-`ALLOC:` event (e.g. a `RUN:` container) is rejected before its run is even looked at.
- **The click-to-filter JS was generalized to a `{kind, value}` filter-object model** (`filterForSwatch()`/`sameFilter()`/`eventMatchesFilter()`) rather than bolting on a second `activeUnused` variable beside the existing `activeProposal` -- keeps the single-active-filter invariant (clicking the active entry clears it; clicking a different entry, of either kind, switches the active filter) as one code path.
- **The two commits split shared files by task via file surgery** (see "TDD Gate Compliance" above) rather than one combined commit -- each commit's diff is provably scoped to its own task, and each was independently verified GREEN (144 tests for Task 1 alone with `calendar.html` at its pre-Task-2 state; 156 for the combined Task 1+2 state) before being committed.

## Deviations from Plan

None beyond the TDD-ordering note above (see "TDD Gate Compliance") -- every task's written `<behavior>`/`<action>`/`<acceptance_criteria>` was satisfied exactly as specified; no Rule 1-4 auto-fixes were needed.

## Issues Encountered

- **A JS comment string collided with a test assertion.** The first draft of the generalized click-handler's inline comment literally quoted `data-unused="1"` as prose ("toggles `cal-filter-match` on elements carrying `data-unused="1"` instead"), which made `MonthCellUnusedNightRenderTest.test_future_allocation_night_renders_none_of_the_three_channels`'s `assertNotIn('data-unused="1"', content)` fail -- the substring matched inside the `<script>` block's own comment text, not an actual chip attribute. Fixed by rewording the comment to say "the data-unused flag" instead of quoting the literal attribute syntax; re-ran the plan's own inline verify command (`data-unused` count) to confirm it still counts >= 2 template-attribute occurrences.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

Plan 37-07 (paired-docs update, wave 5) can now document:

- The calendar pop-up's tally line (this plan's `run_tally()`), alongside 37-05's table Progress column, in `campaign_lifecycle_demo.ipynb`'s campaign-lifecycle cells.
- The `[U]` unused-night chip, its click-to-filter legend entry, and the `docs/runbooks/telescope_runs_calendar.rst` status-vocabulary/legend section (which already needs updating for STATUS-01's `[U]` entry from 37-01).

**Requirements traceability:** this plan's frontmatter declares `requirements: [TALLY-01, UNUSED-01]`. `requirements.ready-ids` reported `0/2 ready` -- both are declared by sibling plan 37-07, which has not yet produced its own `*-SUMMARY.md`, so neither is marked complete this session (`requirements-completed: []` above is correct, not an omission). The shared-ID gate re-evaluates automatically once 37-07 finishes.

No blockers for 37-07. This plan's own two test modules (156 tests total) and every task-level `<acceptance_criteria>`/inline `<verify>` command passed. The plan-level full-`solsys_code`-suite regression command was launched in the background at the start of this session's close-out (`nohup ... python manage.py test $LABELS ...`) per this dispatch's `closeout_discipline` instruction and was still running when this SUMMARY was written; per that same instruction, the orchestrator runs the full suite as its own post-merge gate immediately after this plan returns, so this session did not block on it. Recorded here (not yet in the broken-windows ledger, since the command was launched by this same session rather than left permanently unrun) for visibility.

---
*Phase: 37-status-vocabulary-public-tallies-provenance-blind-gaps*
*Completed: 2026-09-19*

## Self-Check: PASSED

- `solsys_code/templatetags/calendar_display_extras.py` -- FOUND
- `src/templates/tom_calendar/partials/event_form.html` -- FOUND
- `src/templates/tom_calendar/partials/calendar.html` -- FOUND
- `solsys_code/tests/test_calendar_display_extras.py` -- FOUND
- `solsys_code/tests/test_calendar_template.py` -- FOUND
- Commit `8b3225c` -- FOUND
- Commit `b54836d` -- FOUND
- `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` re-confirmed: 156 tests, `OK`
- `pre-commit run ruff --all-files` / `ruff-format --all-files` re-confirmed: both Passed
- All plan-level task `<acceptance_criteria>` re-verified true (per-task `<automated>` commands re-run above)
- The plan-level `<verification>` block's full-`solsys_code`-suite item is NOT re-confirmed this session -- see "Next Phase Readiness" above; launched in background, orchestrator owns the post-merge confirmation
