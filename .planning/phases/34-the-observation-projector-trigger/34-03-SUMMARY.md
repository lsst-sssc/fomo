---
phase: 34-the-observation-projector-trigger
plan: 03
subsystem: calendar-sync
tags: [django-template-tags, django-templates, status-rings, legend, series-identity, n-plus-one]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    plan: "01"
    provides: "the observation projector's marker vocabulary ([Q]/[S]/[O]/[X]/[C]/[F]/[?]) and CalendarEventMeta.observation_record/observation_group, which this plan reads and renders"
provides:
  - "solsys_code/templatetags/calendar_display_extras.py: _TERMINAL_PREFIXES extended with the projector's [X] /[C] /[F] /[?]  markers; status_border_css() recognizes [Q] ; observation_status_legend() (new simple_tag, fixed 7-entry marker legend); observation_series_decoration() (new simple_tag, read-only night-n-of-N decoration)"
  - "src/templates/tom_calendar/partials/calendar.html: the observation-status legend row"
  - "src/templates/tom_calendar/partials/event_form.html: the 'Observation series' modal block, sibling of campaign_decoration"
  - "solsys_code/views.py: fomo_render_calendar's Prefetch widened to select observation_record__target and observation_group"
affects: [37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 11088
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Display-time decoration from a link, never from stored text: observation_series_decoration() mirrors campaign_decoration()'s exact shape (isinstance guard, ObjectDoesNotExist guard, reverse()-in-Python, fixed-key dict, reads-only) so series identity survives a base-layer re-projection the same way campaign attribution already does."
    - "Sort-key substitution for an unprojectable sibling: _window_start_or_max() reimplements the spike's own datetime.max-substitution rule locally rather than importing it across modules, so one bad sibling's window failure sorts it last instead of raising inside a public template tag."
    - "Docstring text that mirrors a grep-gated 'no write call' verify command must avoid literally containing the patterns it's proving absent (e.g. '.save(') -- prose describing the guarantee, not the method names themselves, keeps the plan's own verify script from tripping on its own documentation."

key-files:
  created: []
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - src/templates/tom_calendar/partials/calendar.html
    - src/templates/tom_calendar/partials/event_form.html
    - solsys_code/views.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/tests/test_calendar_template.py

key-decisions:
  - "observation_status_legend() is a fixed, hand-maintained tuple rather than derived from _TERMINAL_PREFIXES/status_border_css() -- deriving it would only let a marker-with-ring-but-no-label (or vice versa) drift silently; Phase 37 owns the final wording."
  - "The new [X] /[C] /[F] /[?]  tokens carry a trailing space in _TERMINAL_PREFIXES deliberately (matching the plan's own instruction), since the tuple is consumed by title.startswith() and a bare '[C]' would also match a hypothetical future '[COMPLETED]'-style bracket-word prefix."
  - "observation_series_decoration()'s docstring was rewritten mid-task to describe 'no database write of any kind' in prose rather than naming .save()/.update()/.create()/get_or_create() literally, after the plan's own verify grep (scoped from `def observation_series_decoration` to end of file) counted those exact substrings inside the docstring itself as false-positive write calls."

requirements-completed: [PROJ-06, PROJ-03, PROJ-05]

coverage:
  - id: D1
    description: "Every projector marker paints the right status ring in a month cell ([Q]/[X]/[C]/[F]/[?] get a ring, [S]/[O] stay ring-free), every pre-existing bracket-word ring is untouched, and a visitor can read the vocabulary off a fixed legend rendered on the calendar page itself."
    requirement: "PROJ-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestProjectorMarkerRings, TestObservationStatusLegend"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#CalendarStatusLegendRenderTest.test_calendar_page_renders_every_legend_marker_and_label"
        status: pass
    human_judgment: false
  - id: D2
    description: "A grouped observation record's event modal shows which night of how many it is, the group's name, and links back to the group list and the record's own detail page -- rendered at request time from CalendarEventMeta.observation_group/.observation_record, numbered by window start (not pk or insertion order), with an unprojectable sibling sorting last rather than raising, and returning None for every documented empty-input edge (no companion row, no group, single-member group, non-CalendarEvent argument)."
    requirement: "PROJ-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestObservationSeriesDecoration"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest"
        status: pass
    human_judgment: false
  - id: D3
    description: "Series identity is never written into the event's own title/description -- projecting a record, rendering the modal, and re-projecting the record leaves event.title and event.description byte-identical -- and the campaign and series decorations render side by side in the same modal without either overwriting the other."
    requirement: "PROJ-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#TestObservationSeriesDecoration.test_render_then_reproject_leaves_title_and_description_byte_identical"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest.test_grouped_and_attributed_event_shows_both_decorations"
        status: pass
    human_judgment: false
  - id: D4
    description: "Rendering a month of grouped observation events issues no per-event query fan-out for the new tag: fomo_render_calendar's Prefetch was widened to select observation_record__target and observation_group, and a second grouped event does not add a query to the month view."
    requirement: "PROJ-05"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest.test_month_view_query_count_does_not_grow_with_second_grouped_event"
        status: pass
    human_judgment: false
  - id: D5
    description: "The month-cell truncatechars:18/:16 budget (PROJ-06) is unchanged, no existing status_border_css() assertion was modified, and the whole pre-existing project test suite plus both ruff gates stay green with the new marker vocabulary, legend and series decoration live."
    requirement: "PROJ-06"
    verification:
      - kind: integration
        ref: "workflow.test_command (.planning/config.json) -- full solsys_code + observatory suite, exit 0"
        status: pass
      - kind: other
        ref: "pre-commit run ruff --all-files && pre-commit run ruff-format --all-files"
        status: pass
    human_judgment: false

duration: 38min
completed: 2026-09-11
status: complete
---

# Phase 34 Plan 3: Status Rings, Marker Legend & Series Identity Summary

**The observation projector's `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers now paint the right calendar ring and appear on a page legend, and a grouped record's event modal shows "night n of N" read live from `CalendarEventMeta.observation_group` — never written into the event itself.**

## Performance

- **Duration:** 38 min
- **Started:** ~2026-09-11T03:36:00Z
- **Completed:** 2026-09-11T04:14:14Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- `_TERMINAL_PREFIXES` extended with `'[X] '`, `'[C] '`, `'[F] '`, `'[?] '` (trailing space, deliberate); `status_border_css()` recognizes `'[Q] '` alongside `'[QUEUED] '`; every pre-existing bracket-word assertion (`[QUEUED]`/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]`) untouched and still passing.
- `observation_status_legend()` (new `simple_tag`): a fixed, ordered 7-entry marker vocabulary, rendered in `calendar.html`'s existing legend row using the `.cal-legend-telescope` class family (no new visual language).
- `observation_series_decoration()` (new `simple_tag`), modelled directly on `campaign_decoration()`: reads a grouped event's `CalendarEventMeta.observation_group`/`.observation_record` at request time, orders siblings by `record_time_window()`'s start (an unprojectable sibling sorts last via a local `datetime.max` substitution rather than raising), and returns group name/pk, 1-based index, size, and two `reverse()`-built URLs. Reads only — verified by a grep gate that counts zero write-method calls and zero PII field names in the function body.
- `event_form.html` renders the new tag as a sibling block (not nested) of `campaign_decoration`, so a record can be in a group with no campaign, attributed to a campaign with no group, or both, and every combination renders.
- `fomo_render_calendar`'s `CalendarEventMeta` `Prefetch` widened to also select `observation_record__target` and `observation_group`, so the modal tag costs no query per event in the month view — proven by an `assertNumQueries`-style count-comparison regression.

## Task Commits

Each task was committed atomically:

1. **Task 1: Status rings and a legend for the compact marker vocabulary** - `637fd5a` (feat)
2. **Task 2: "Night n of N" in the modal — series identity read from the link, never from the title** - `6d759c9` (feat)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Files Created/Modified
- `solsys_code/templatetags/calendar_display_extras.py` - extended `_TERMINAL_PREFIXES`, `status_border_css()`; new `observation_status_legend()`, `_window_start_or_max()`, `observation_series_decoration()`
- `src/templates/tom_calendar/partials/calendar.html` - the observation-status legend row
- `src/templates/tom_calendar/partials/event_form.html` - the "Observation series" modal block
- `solsys_code/views.py` - `fomo_render_calendar`'s widened `Prefetch`
- `solsys_code/tests/test_calendar_display_extras.py` - `TestProjectorMarkerRings`, `TestObservationStatusLegend`, `TestObservationSeriesDecoration` (new classes)
- `solsys_code/tests/test_calendar_template.py` - `CalendarStatusLegendRenderTest`, `EventModalSeriesDecorationTest` (new classes, including the query-count regression)

## Decisions Made
- `observation_status_legend()` is a fixed, hand-maintained tuple rather than derived from `_TERMINAL_PREFIXES`/`status_border_css()` — deriving it risks a marker-with-ring-but-no-label drift; Phase 37 owns the final wording of the whole vocabulary.
- `observation_series_decoration()`'s docstring was rewritten to describe "no database write of any kind" in prose after its first draft (mirroring `campaign_decoration()`'s own phrasing, which literally names `.save()`/`.update()`/`.create()`/`get_or_create()`) caused the plan's own verify grep — scoped from `def observation_series_decoration` to end of file — to count those substrings inside the docstring as false-positive write calls. See Deviations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `observation_series_decoration()`'s docstring self-defeated its own no-write verify gate**
- **Found during:** Task 2, running the plan's own `python -c` grep-count verify command after writing the function
- **Issue:** The docstring, written to mirror `campaign_decoration()`'s "Reads only: never calls ``.save()``, ``.update()``, ``.create()`` or ``get_or_create()``" phrasing, literally contained all four substrings the plan's verify command counts (`.save(`, `.update(`, `.create(`, `get_or_create(`) — the script scopes `body = src[i:]` from `def observation_series_decoration` to end of file, which includes the function's own docstring, so the count came back 4 instead of the required 0. This is a documentation artifact, not an actual write — the function body itself never calls any of these methods — but the plan's acceptance criteria require the literal grep count to be 0.
- **Fix:** Reworded the docstring to state the same guarantee in prose ("performs no database write of any kind (no save, no bulk update, no row creation, no find-or-create call)") without reproducing the method-call syntax.
- **Files modified:** `solsys_code/templatetags/calendar_display_extras.py`
- **Verification:** `python -c "... print(sum(body.count(t) for t in ('.save(','.update(','.create(','get_or_create(')))"` now prints `0`; full scoped test suite still green.
- **Commit:** `6d759c9`

---

**Total deviations:** 1 auto-fixed (Rule 1 — a self-referential false positive in the plan's own verify gate, caused by this task's own docstring wording). No production behaviour changed outside the docstring text; the function's actual read-only contract was correct from the first draft.
**Impact on plan:** None outside the stated scope — no architectural change, no scope creep.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
Phase 34 Plan 4 (the notebook/runbook/docs plan) can now build on a fully-wired, fully-tested display layer: every projector marker paints the right ring and appears on the page legend, and a grouped record's modal reads its series identity from the link with no per-event query cost. No blockers for 34-04 — this plan's files (`calendar_display_extras.py`, the two calendar templates, `views.py`, and the two calendar test modules) are disjoint from 34-04's expected scope (paired notebooks/runbooks), and the sibling wave-2 plan 34-02's own noted gap (verbose-vs-terse marker vocabulary reconciliation, earmarked for Phase 37) is exactly what Task 1 of this plan closes for the ring/legend half; the title-prefix-vocabulary *unification* itself remains Phase 37's STATUS-01/02, as both plans agree.

## Self-Check: PASSED

- `solsys_code/templatetags/calendar_display_extras.py` contains `def observation_status_legend(` and `def observation_series_decoration(` — FOUND
- `src/templates/tom_calendar/partials/calendar.html` contains `observation_status_legend` — FOUND
- `src/templates/tom_calendar/partials/event_form.html` contains `observation_series_decoration` — FOUND
- `solsys_code/views.py` Prefetch selects `observation_record__target` and `observation_group` — FOUND
- Commit `637fd5a` — FOUND in `git log`
- Commit `6d759c9` — FOUND in `git log`

---
*Phase: 34-the-observation-projector-trigger*
*Completed: 2026-09-11*
